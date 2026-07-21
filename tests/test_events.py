"""Tests for the pillar-3 event-level evaluation harness (`rowii.eval.events` +
`scripts/eval_events.py`; Step-2 package 6, design spec D3, plan Task 3).

Synthetic hand-built frames ONLY: the harness is PREPARED-ONLY (no real fault
labels exist until the induced-fault campaign), so every case here pins the
spec-D3 edge semantics on constructed nanosecond timestamps at window_s=1.0 --
inclusive-start/exclusive-end membership, tolerance padding (negative latency
kept), first-alarm-only latency, vacuous-TPR NaN, role filtering, and the
non-event FAR denominators.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rowii.eval.events import EventEvalResult, evaluate_events

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

BASE_NS = int(pd.Timestamp("2026-07-01T00:00:00+00:00").value)
_NS_PER_S = 1_000_000_000

_SUMMARY_COLUMNS = [
    "n_events", "n_detected", "event_tpr", "false_alarm_windows",
    "false_alarm_rate_per_hour", "realized_window_far", "tolerance_s",
]
_PER_EVENT_COLUMNS = ["start_utc", "end_utc", "kind", "detected", "latency_s"]


def _iso(seconds: float) -> str:
    """ISO-8601 UTC instant *seconds* after BASE_NS (tz-aware, '+00:00' offset)."""
    return pd.Timestamp(BASE_NS + int(seconds * _NS_PER_S), tz="UTC").isoformat()


def _alarms(n: int, alarm_at: list[int], role: list[str] | None = None) -> pd.DataFrame:
    """*n* consecutive 1-s windows starting at BASE_NS; alarm=True exactly at the
    window indices in *alarm_at* (timestamps are window STARTS, spec D3)."""
    alarm = np.zeros(n, dtype=bool)
    alarm[alarm_at] = True
    frame = pd.DataFrame(
        {
            "t_utc_ns": BASE_NS + np.arange(n, dtype=np.int64) * _NS_PER_S,
            "alarm": alarm,
        }
    )
    if role is not None:
        frame["role"] = role
    return frame


def _events(
    intervals: list[tuple[float, float]], kind: list[str] | None = None
) -> pd.DataFrame:
    """Events table with tz-aware ISO-8601 strings; *intervals* are
    (start_s, end_s) offsets after BASE_NS -- the string form `pd.read_csv` of a
    campaign events.csv would deliver."""
    frame = pd.DataFrame(
        {
            "start_utc": [_iso(start) for start, _ in intervals],
            "end_utc": [_iso(end) for _, end in intervals],
        }
    )
    if kind is not None:
        frame["kind"] = kind
    return frame


# ---------------------------------------------------------------------------
# 1. Basic detection + result shape (frozen dataclass, to_frame contract)
# ---------------------------------------------------------------------------


def test_single_event_detected_end_to_end_result() -> None:
    result = evaluate_events(_alarms(10, [4]), _events([(2.0, 6.0)]), window_s=1.0)

    assert isinstance(result, EventEvalResult)
    assert result.n_events == 1
    assert result.n_detected == 1
    assert result.event_tpr == pytest.approx(1.0)
    assert result.tolerance_s == 0.0
    # Windows 2..5 lie inside [2, 6); the alarm at t=4 is inside -> no false alarm.
    assert result.false_alarm_windows == 0
    assert result.realized_window_far == pytest.approx(0.0)  # 0 / 6 non-event windows
    assert result.false_alarm_rate_per_hour == pytest.approx(0.0)

    assert list(result.per_event.columns) == _PER_EVENT_COLUMNS
    assert result.per_event["detected"].tolist() == [True]
    assert result.per_event["latency_s"].tolist() == pytest.approx([2.0])
    assert result.per_event["kind"].tolist() == ["fault"]  # no kind column -> default

    frame = result.to_frame()
    assert list(frame.columns) == _SUMMARY_COLUMNS
    assert len(frame) == 1
    assert int(frame.loc[0, "n_detected"]) == 1

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.n_events = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Overlapping events: one alarming window may detect BOTH (per-event
#    membership, never first-match)
# ---------------------------------------------------------------------------


def test_overlapping_events_one_window_detects_both() -> None:
    result = evaluate_events(
        _alarms(10, [5]), _events([(2.0, 6.0), (4.0, 8.0)]), window_s=1.0
    )

    assert result.n_events == 2
    assert result.n_detected == 2
    assert result.event_tpr == pytest.approx(1.0)
    assert result.per_event["detected"].tolist() == [True, True]
    # Same alarming window (t=5), latency measured from EACH event's own start.
    assert result.per_event["latency_s"].tolist() == pytest.approx([3.0, 1.0])
    assert result.false_alarm_windows == 0


# ---------------------------------------------------------------------------
# 3. Zero events: TPR is NaN (vacuous, never 1.0); every alarm is a false alarm
# ---------------------------------------------------------------------------


def test_zero_events_tpr_nan_and_all_alarms_false() -> None:
    result = evaluate_events(_alarms(10, [3, 7]), _events([]), window_s=1.0)

    assert result.n_events == 0
    assert result.n_detected == 0
    assert math.isnan(result.event_tpr)
    assert result.false_alarm_windows == 2
    assert result.realized_window_far == pytest.approx(2 / 10)
    assert result.false_alarm_rate_per_hour == pytest.approx(2 / (10 / 3600))
    assert len(result.per_event) == 0
    assert list(result.per_event.columns) == _PER_EVENT_COLUMNS
    assert math.isnan(result.to_frame().loc[0, "event_tpr"])


# ---------------------------------------------------------------------------
# 4. Zero alarms with events present: TPR 0.0 (NOT NaN), latency NaN
# ---------------------------------------------------------------------------


def test_zero_alarms_tpr_zero_when_events_exist() -> None:
    result = evaluate_events(_alarms(10, []), _events([(2.0, 6.0)]), window_s=1.0)

    assert result.n_events == 1
    assert result.n_detected == 0
    assert result.event_tpr == pytest.approx(0.0)
    assert result.per_event["detected"].tolist() == [False]
    assert math.isnan(result.per_event.loc[0, "latency_s"])
    assert result.false_alarm_windows == 0
    assert result.realized_window_far == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. Event fully outside the alarms' time span: missed, latency NaN; the alarms
#    elsewhere are false alarms
# ---------------------------------------------------------------------------


def test_event_fully_outside_alarm_span_missed() -> None:
    result = evaluate_events(_alarms(10, [1]), _events([(20.0, 25.0)]), window_s=1.0)

    assert result.n_detected == 0
    assert result.event_tpr == pytest.approx(0.0)
    assert math.isnan(result.per_event.loc[0, "latency_s"])
    # No scored window lies inside [20, 25) -> all 10 windows are non-event; the
    # single alarming window is a false alarm.
    assert result.false_alarm_windows == 1
    assert result.realized_window_far == pytest.approx(1 / 10)
    assert result.false_alarm_rate_per_hour == pytest.approx(360.0)


# ---------------------------------------------------------------------------
# 6. Latency uses the FIRST alarming window only
# ---------------------------------------------------------------------------


def test_latency_first_alarm_only() -> None:
    result = evaluate_events(_alarms(10, [5, 6]), _events([(2.0, 8.0)]), window_s=1.0)

    assert result.per_event["detected"].tolist() == [True]
    assert result.per_event["latency_s"].tolist() == pytest.approx([3.0])


# ---------------------------------------------------------------------------
# 7. Boundary semantics: inclusive start, EXCLUSIVE end (timestamps are window
#    starts -- a window starting exactly at end_utc lies entirely after the event)
# ---------------------------------------------------------------------------


def test_boundary_inclusive_start_exclusive_end() -> None:
    at_start = evaluate_events(_alarms(10, [3]), _events([(3.0, 6.0)]), window_s=1.0)
    assert at_start.n_detected == 1
    assert at_start.per_event["latency_s"].tolist() == pytest.approx([0.0])
    assert at_start.false_alarm_windows == 0

    at_end = evaluate_events(_alarms(10, [6]), _events([(3.0, 6.0)]), window_s=1.0)
    assert at_end.n_detected == 0
    assert math.isnan(at_end.per_event.loc[0, "latency_s"])
    # The window starting exactly at end_utc is OUTSIDE the event -> false alarm.
    assert at_end.false_alarm_windows == 1


# ---------------------------------------------------------------------------
# 8. Tolerance pad: an alarm before the logged start (inside the pad) detects
#    the event with a NEGATIVE latency -- kept, not clamped (spec D3)
# ---------------------------------------------------------------------------


def test_negative_latency_inside_tolerance_pad_kept() -> None:
    result = evaluate_events(
        _alarms(10, [4]), _events([(5.0, 8.0)]), window_s=1.0, tolerance_s=2.0
    )

    assert result.n_detected == 1
    assert result.per_event["latency_s"].tolist() == pytest.approx([-1.0])
    assert result.tolerance_s == 2.0
    # Padded interval [3, 10) swallows windows 3..9 -> only 0..2 are non-event,
    # and the alarming window (t=4) is inside the pad -> zero false alarms.
    assert result.false_alarm_windows == 0
    assert result.realized_window_far == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. Role filtering: rows with role != "scored" are dropped BEFORE every metric
#    (numerators AND denominators)
# ---------------------------------------------------------------------------


def test_role_filter_drops_non_scored_rows() -> None:
    role = (
        ["scored"] * 10
        + ["consumed_for_calibration"] * 3
        + ["unknown_state"] * 2
        + ["no_conformal_data"]
    )
    # Alarms at t=9 (scored) and t=12 (consumed -- must be invisible to the harness).
    alarms = _alarms(16, [9, 12], role=role)

    result = evaluate_events(alarms, _events([]), window_s=1.0)

    assert result.false_alarm_windows == 1
    # Denominator counts the 10 scored windows only, never the 6 dropped rows.
    assert result.realized_window_far == pytest.approx(1 / 10)


# ---------------------------------------------------------------------------
# 10. Naive timestamps refuse loudly, telling the user to add a UTC offset
# ---------------------------------------------------------------------------


def test_naive_timestamps_raise() -> None:
    naive = pd.DataFrame(
        {"start_utc": ["2026-07-01T00:00:02"], "end_utc": ["2026-07-01T00:00:06"]}
    )
    with pytest.raises(ValueError, match="UTC offset"):
        evaluate_events(_alarms(10, [4]), naive, window_s=1.0)


# ---------------------------------------------------------------------------
# 11. Missing / mistyped required columns refuse loudly
# ---------------------------------------------------------------------------


def test_missing_required_columns_raise() -> None:
    events = _events([(2.0, 6.0)])

    with pytest.raises(ValueError, match="alarm"):
        evaluate_events(pd.DataFrame({"t_utc_ns": [BASE_NS]}), events, window_s=1.0)

    with pytest.raises(ValueError, match="end_utc"):
        evaluate_events(
            _alarms(10, [4]), pd.DataFrame({"start_utc": [_iso(2.0)]}), window_s=1.0
        )

    floats = pd.DataFrame({"t_utc_ns": [1.5], "alarm": [True]})
    with pytest.raises(ValueError, match="int64"):
        evaluate_events(floats, events, window_s=1.0)


# ---------------------------------------------------------------------------
# 12. FAR denominators are NaN when every scored window lies inside events
# ---------------------------------------------------------------------------


def test_far_nan_when_no_non_event_windows() -> None:
    result = evaluate_events(_alarms(5, [2]), _events([(0.0, 5.0)]), window_s=1.0)

    assert result.n_detected == 1
    assert result.false_alarm_windows == 0
    assert math.isnan(result.realized_window_far)
    assert math.isnan(result.false_alarm_rate_per_hour)


# ---------------------------------------------------------------------------
# 13. end_utc before start_utc is a data-entry error, not a "missed event"
# ---------------------------------------------------------------------------


def test_end_before_start_raises() -> None:
    swapped = _events([(6.0, 2.0)])
    with pytest.raises(ValueError, match="precedes"):
        evaluate_events(_alarms(10, [4]), swapped, window_s=1.0)


# ---------------------------------------------------------------------------
# 14. kind column: carried through when present, "fault" when absent/missing
# ---------------------------------------------------------------------------


def test_kind_column_carried_and_defaulted() -> None:
    with_kind = evaluate_events(
        _alarms(10, [4]),
        _events([(2.0, 6.0), (7.0, 9.0)], kind=["cavitation", None]),  # type: ignore[list-item]
        window_s=1.0,
    )
    assert with_kind.per_event["kind"].tolist() == ["cavitation", "fault"]

    without_kind = evaluate_events(_alarms(10, [4]), _events([(2.0, 6.0)]), window_s=1.0)
    assert without_kind.per_event["kind"].tolist() == ["fault"]


# ---------------------------------------------------------------------------
# 15. window_s / tolerance_s validation
# ---------------------------------------------------------------------------


def test_invalid_window_s_and_tolerance_raise() -> None:
    alarms, events = _alarms(10, [4]), _events([(2.0, 6.0)])

    with pytest.raises(ValueError, match="window_s"):
        evaluate_events(alarms, events, window_s=0.0)
    with pytest.raises(ValueError, match="tolerance_s"):
        evaluate_events(alarms, events, window_s=1.0, tolerance_s=-1.0)


# ---------------------------------------------------------------------------
# 16. CLI smoke: parquet + csv in, event_eval.csv (summary row first) +
#     event_notes.md (honesty framing + D3 conventions) out
# ---------------------------------------------------------------------------


def _write_cli_inputs(tmp_path: Path) -> tuple[Path, Path]:
    alarms = _alarms(10, [4], role=["scored"] * 10)
    # Realistic monitor alarms.parquet schema (extra columns must be ignored).
    alarms["window"] = np.arange(10, dtype=np.int64)
    alarms["state"] = np.int64(0)
    alarms["score"] = 0.1
    alarms["p_value"] = 0.5
    alarms["low_confidence"] = False
    alarms_path = tmp_path / "alarms.parquet"
    alarms.to_parquet(alarms_path, engine="pyarrow", index=False)

    events_path = tmp_path / "events.csv"
    _events([(2.0, 6.0)], kind=["demo-fault"]).to_csv(events_path, index=False)
    return alarms_path, events_path


def test_cli_events_csv_with_comment_provenance_lines(tmp_path: Path) -> None:
    """The real ground-truth files (`docs/groundtruth/080726_events_*.csv`) open
    with `#` provenance lines before the header -- the CLI must skip them
    instead of parsing the first comment as the column row (the P7 execution-B
    failure mode)."""
    import eval_events

    alarms_path, events_path = _write_cli_inputs(tmp_path)
    commented = tmp_path / "events_commented.csv"
    commented.write_text(
        "# Ground truth: induced strikes, provenance line\n"
        "# second provenance line (verification pointer)\n" + events_path.read_text()
    )
    out_dir = tmp_path / "out-commented"

    rc = eval_events.main(
        [
            "--alarms", str(alarms_path),
            "--events", str(commented),
            "--tolerance-s", "1.0",
            "--out", str(out_dir),
        ]
    )
    assert rc == 0
    frame = pd.read_csv(out_dir / "event_eval.csv")
    assert int(frame.loc[0, "n_events"]) == 1
    assert int(frame.loc[0, "n_detected"]) == 1


def test_cli_smoke_writes_event_eval_csv_and_notes(tmp_path: Path) -> None:
    import eval_events

    alarms_path, events_path = _write_cli_inputs(tmp_path)
    out_dir = tmp_path / "out"

    rc = eval_events.main(
        [
            "--alarms", str(alarms_path),
            "--events", str(events_path),
            "--tolerance-s", "1.0",
            "--out", str(out_dir),
        ]
    )
    assert rc == 0

    csv_path = out_dir / "event_eval.csv"
    assert csv_path.is_file()
    frame = pd.read_csv(csv_path)
    assert list(frame.columns) == ["row_type", *_SUMMARY_COLUMNS, *_PER_EVENT_COLUMNS]
    assert frame.loc[0, "row_type"] == "summary"
    assert int(frame.loc[0, "n_events"]) == 1
    assert int(frame.loc[0, "n_detected"]) == 1
    assert frame.loc[0, "tolerance_s"] == pytest.approx(1.0)
    event_rows = frame[frame["row_type"] == "event"]
    assert len(event_rows) == 1
    assert event_rows["kind"].tolist() == ["demo-fault"]
    assert event_rows["latency_s"].tolist() == pytest.approx([2.0])

    notes = (out_dir / "event_notes.md").read_text()
    assert "PREPARED for the induced-fault campaign" in notes
    assert "demo" in notes.lower()
    assert "inclusive start" in notes.lower()
    assert "exclusive" in notes.lower()
    assert "tolerance" in notes.lower()


def test_cli_missing_input_files_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import eval_events

    alarms_path, events_path = _write_cli_inputs(tmp_path)

    rc = eval_events.main(
        [
            "--alarms", str(tmp_path / "nope.parquet"),
            "--events", str(events_path),
            "--out", str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert "not found" in capsys.readouterr().err

    rc = eval_events.main(
        [
            "--alarms", str(alarms_path),
            "--events", str(tmp_path / "nope.csv"),
            "--out", str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert "not found" in capsys.readouterr().err


def test_cli_naive_timestamps_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import eval_events

    alarms_path, _ = _write_cli_inputs(tmp_path)
    naive_path = tmp_path / "naive_events.csv"
    pd.DataFrame(
        {"start_utc": ["2026-07-01T00:00:02"], "end_utc": ["2026-07-01T00:00:06"]}
    ).to_csv(naive_path, index=False)

    rc = eval_events.main(
        [
            "--alarms", str(alarms_path),
            "--events", str(naive_path),
            "--out", str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert "UTC offset" in capsys.readouterr().err


def test_cli_help_documents_every_flag(capsys: pytest.CaptureFixture[str]) -> None:
    import eval_events

    with pytest.raises(SystemExit) as exc_info:
        eval_events.main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--alarms", "--events", "--tolerance-s", "--window-s", "--out"):
        assert flag in out, f"missing {flag!r} in --help output"
