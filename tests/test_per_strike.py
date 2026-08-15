"""Tests for the per-strike / first-alarm-latency evaluation harness
(`rowii.eval.per_strike` + `scripts/eval_per_strike.py`).

Synthetic hand-built frames throughout: every case pins one piece of the
seconds-level strike-eval contract (`docs/groundtruth/080726_strikes_seconds_
{st,pu}.csv` shape) -- window-start-to-mark tolerance matching, strict
`p_value < alpha`, role filtering, the 1.5 s double-impulse dedup rule,
at-or-after first-alarm latency with a search horizon, and the per-kind-group
breakdown (`plate-gen`, `plate-tur`, `landmark`, `vane-sweep`, plus the
overall `"ALL"` row, always five rows regardless of which groups are present
in a given input -- a stable CSV shape across the full representation/session
sweep).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from rowii.eval.per_strike import (
    LatencySummary,
    deduplicate_marks,
    evaluate_event_latency,
    evaluate_strike_latency,
    inter_mark_gaps,
    kind_group,
    mark_detected,
    marks_per_event,
    summarize_latency,
    sweep_strike_detection,
)

BASE = pd.Timestamp("2026-07-08T10:00:00+00:00")
_NS_PER_S = 1_000_000_000

_KIND_GROUPS_ALL = {"plate-gen", "plate-tur", "landmark", "vane-sweep", "ALL"}


def _ts(offset_s: float) -> pd.Timestamp:
    """The UTC instant *offset_s* seconds after BASE."""
    return BASE + pd.Timedelta(seconds=offset_s)


def _alarms(
    n: int,
    *,
    alarming_at: list[int] | None = None,
    role: list[str] | None = None,
    start_s: float = 0.0,
) -> pd.DataFrame:
    """*n* consecutive 1-s scored windows starting at BASE+start_s. p_value is
    0.5 everywhere (never alarms at any of this project's alphas) except at the
    window indices in *alarming_at*, where p_value=0.001 (alarms at every alpha
    in {0.01, 0.05, 0.10}). *role* defaults to "scored" for every row."""
    t = BASE.value + int(start_s * _NS_PER_S) + np.arange(n, dtype=np.int64) * _NS_PER_S
    p = np.full(n, 0.5, dtype=np.float64)
    if alarming_at:
        p[alarming_at] = 0.001
    frame = pd.DataFrame({"t_utc_ns": t.astype(np.int64), "p_value": p})
    frame["role"] = role if role is not None else ["scored"] * n
    return frame


def _marks(rows: list[tuple[str, str, str, int, float]]) -> pd.DataFrame:
    """A marks table from *rows* = (session, event_id, kind, strike_no,
    offset_s); strike_utc is written as an ISO-8601 string, the shape
    `pd.read_csv` would deliver from the real ground-truth CSVs."""
    return pd.DataFrame(
        {
            "session": [r[0] for r in rows],
            "event_id": [r[1] for r in rows],
            "kind": [r[2] for r in rows],
            "strike_no": [r[3] for r in rows],
            "strike_utc": [_ts(r[4]).isoformat() for r in rows],
        }
    )


# ---------------------------------------------------------------------------
# kind_group: the fixed plate-gen / plate-tur / landmark / vane-sweep taxonomy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind", ["plate-gen_0", "plate-gen_90", "plate-gen_180", "plate-gen_270"]
)
def test_kind_group_plate_gen_variants(kind: str) -> None:
    assert kind_group(kind) == "plate-gen"


@pytest.mark.parametrize(
    "kind",
    ["plate-tur_0", "plate-tur_90", "plate-tur_180", "plate-tur_270", "plate-tur_bottom"],
)
def test_kind_group_plate_tur_variants(kind: str) -> None:
    assert kind_group(kind) == "plate-tur"


@pytest.mark.parametrize(
    "kind", ["landmark-A_kugelschieber", "landmark-B_11TG", "landmark-C_EG"]
)
def test_kind_group_landmark_variants(kind: str) -> None:
    assert kind_group(kind) == "landmark"


def test_kind_group_vane_sweep() -> None:
    assert kind_group("vane-sweep") == "vane-sweep"


def test_kind_group_unrecognized_kind_raises() -> None:
    with pytest.raises(ValueError, match="unrecognized"):
        kind_group("something-else")


# ---------------------------------------------------------------------------
# deduplicate_marks: <1.5s-to-previous-mark-of-the-same-event double-impulse
# grouping; "first mark = the strike time"
# ---------------------------------------------------------------------------


def test_dedup_merges_marks_closer_than_gap_into_one_physical_strike() -> None:
    marks = _marks(
        [
            ("st", "02", "plate-gen_0", 1, 0.0),
            ("st", "02", "plate-gen_0", 2, 0.8),
            ("st", "02", "plate-gen_0", 3, 1.6),  # gap to previous = 0.8s < 1.5s
        ]
    )
    out = deduplicate_marks(marks, gap_s=1.5)

    assert len(out) == 1
    assert out.loc[0, "strike_no"] == 1
    assert out.loc[0, "n_impulses"] == 3
    assert out.loc[0, "last_strike_no"] == 3
    assert out.loc[0, "strike_utc"] == _ts(0.0)


def test_dedup_keeps_marks_at_or_beyond_gap_separate() -> None:
    # Gap exactly == gap_s is NOT "closer than" gap_s -> a new physical strike.
    marks = _marks(
        [
            ("st", "01", "landmark-C_EG", 1, 0.0),
            ("st", "01", "landmark-C_EG", 2, 1.5),
        ]
    )
    out = deduplicate_marks(marks, gap_s=1.5)

    assert len(out) == 2
    assert out["n_impulses"].tolist() == [1, 1]
    assert out["strike_no"].tolist() == [1, 2]


def test_dedup_scoped_per_event_even_when_marks_are_close_across_events() -> None:
    marks = _marks(
        [
            ("pu", "02", "plate-gen_90", 6, 3.0),
            ("pu", "03", "plate-gen_180", 1, 3.2),  # 0.2s later, DIFFERENT event
        ]
    )
    out = deduplicate_marks(marks, gap_s=1.5)

    assert len(out) == 2  # never merged across event_id, however close in time


def test_dedup_real_st_event02_pattern_yields_two_physical_strikes() -> None:
    # Actual offsets from docs/groundtruth/080726_strikes_seconds_st.csv event 02
    # (plate-gen_0): two clusters of 3 close marks, ~55s apart.
    marks = _marks(
        [
            ("st", "02", "plate-gen_0", 1, 0.0),
            ("st", "02", "plate-gen_0", 2, 0.805),
            ("st", "02", "plate-gen_0", 3, 1.681),
            ("st", "02", "plate-gen_0", 4, 55.464),
            ("st", "02", "plate-gen_0", 5, 56.199),
            ("st", "02", "plate-gen_0", 6, 57.065),
        ]
    )
    out = deduplicate_marks(marks, gap_s=1.5)

    assert len(out) == 2
    assert out["strike_no"].tolist() == [1, 4]
    assert out["n_impulses"].tolist() == [3, 3]


def test_dedup_invalid_gap_s_raises() -> None:
    marks = _marks([("st", "01", "landmark-C_EG", 1, 0.0)])
    with pytest.raises(ValueError, match="gap_s"):
        deduplicate_marks(marks, gap_s=0.0)


def test_dedup_missing_required_column_raises() -> None:
    with pytest.raises(ValueError, match="event_id"):
        deduplicate_marks(pd.DataFrame({"session": ["st"], "strike_utc": [_ts(0.0).isoformat()]}))


def test_dedup_naive_timestamp_raises() -> None:
    marks = pd.DataFrame(
        {
            "session": ["st"],
            "event_id": ["01"],
            "kind": ["landmark-C_EG"],
            "strike_no": [1],
            "strike_utc": ["2026-07-08T10:00:00"],
        }
    )
    with pytest.raises(ValueError, match="UTC offset"):
        deduplicate_marks(marks)


# ---------------------------------------------------------------------------
# mark_detected: window-start-to-mark tolerance matching at a strict p<alpha
# ---------------------------------------------------------------------------


def test_mark_detected_true_within_tolerance() -> None:
    alarms = _alarms(10, alarming_at=[5])
    marks = _marks([("st", "01", "landmark-C_EG", 1, 5.5)])  # 0.5s from t=5
    assert mark_detected(alarms, marks, tolerance_s=1.0, alpha=0.05).tolist() == [True]


def test_mark_detected_false_outside_tolerance() -> None:
    alarms = _alarms(10, alarming_at=[5])
    marks = _marks([("st", "01", "landmark-C_EG", 1, 7.5)])  # 2.5s away
    assert mark_detected(alarms, marks, tolerance_s=1.0, alpha=0.05).tolist() == [False]


def test_mark_detected_boundary_at_exactly_tolerance_counts() -> None:
    alarms = _alarms(10, alarming_at=[5])
    marks = _marks([("st", "01", "landmark-C_EG", 1, 6.0)])  # exactly 1.0s away
    assert mark_detected(alarms, marks, tolerance_s=1.0, alpha=0.05).tolist() == [True]


def test_mark_detected_alpha_is_strict_less_than() -> None:
    alarms = _alarms(3)
    alarms["p_value"] = [0.05, 0.05, 0.05]
    marks = _marks([("st", "01", "landmark-C_EG", 1, 1.0)])
    assert mark_detected(alarms, marks, tolerance_s=2.0, alpha=0.05).tolist() == [False]
    assert mark_detected(alarms, marks, tolerance_s=2.0, alpha=0.06).tolist() == [True]


def test_mark_detected_ignores_non_scored_rows() -> None:
    alarms = _alarms(5, alarming_at=[2])
    alarms.loc[2, "role"] = "consumed_for_calibration"
    marks = _marks([("st", "01", "landmark-C_EG", 1, 2.0)])
    assert mark_detected(alarms, marks, tolerance_s=0.5, alpha=0.05).tolist() == [False]


def test_mark_detected_no_role_column_treats_everything_as_scored() -> None:
    alarms = _alarms(5, alarming_at=[2]).drop(columns=["role"])
    marks = _marks([("st", "01", "landmark-C_EG", 1, 2.0)])
    assert mark_detected(alarms, marks, tolerance_s=0.5, alpha=0.05).tolist() == [True]


def test_mark_detected_no_alarming_windows_all_false() -> None:
    alarms = _alarms(5)
    marks = _marks(
        [("st", "01", "landmark-C_EG", 1, 1.0), ("st", "01", "landmark-C_EG", 2, 3.0)]
    )
    assert mark_detected(alarms, marks, tolerance_s=5.0, alpha=0.05).tolist() == [False, False]


def test_mark_detected_empty_marks_returns_empty_array() -> None:
    alarms = _alarms(5, alarming_at=[2])
    result = mark_detected(alarms, _marks([]), tolerance_s=1.0, alpha=0.05)
    assert result.shape == (0,)


def test_mark_detected_invalid_tolerance_raises() -> None:
    alarms = _alarms(5)
    marks = _marks([("st", "01", "landmark-C_EG", 1, 1.0)])
    with pytest.raises(ValueError, match="tolerance_s"):
        mark_detected(alarms, marks, tolerance_s=-1.0, alpha=0.05)


def test_mark_detected_invalid_alpha_raises() -> None:
    alarms = _alarms(5)
    marks = _marks([("st", "01", "landmark-C_EG", 1, 1.0)])
    with pytest.raises(ValueError, match="alpha"):
        mark_detected(alarms, marks, tolerance_s=1.0, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        mark_detected(alarms, marks, tolerance_s=1.0, alpha=1.5)


def test_mark_detected_missing_alarms_columns_raise() -> None:
    marks = _marks([("st", "01", "landmark-C_EG", 1, 1.0)])
    with pytest.raises(ValueError, match="p_value"):
        mark_detected(pd.DataFrame({"t_utc_ns": [0]}), marks, tolerance_s=1.0, alpha=0.05)


# ---------------------------------------------------------------------------
# sweep_strike_detection: the (granularity x tolerance_s x alpha x kind_group)
# tidy grid -- always 5 kind_group rows (4 groups + "ALL"), NaN tpr when a
# group has zero marks in the given input.
# ---------------------------------------------------------------------------


def test_sweep_strike_detection_grid_shape_and_columns() -> None:
    alarms = _alarms(20, alarming_at=[5, 15])
    marks = _marks(
        [
            ("st", "01", "landmark-C_EG", 1, 5.0),
            ("st", "02", "plate-gen_0", 1, 15.0),
        ]
    )
    result = sweep_strike_detection(alarms, marks, tolerances_s=[1.0, 2.0], alphas=[0.05, 0.10])

    assert list(result.columns) == [
        "granularity", "tolerance_s", "alpha", "kind_group", "n_marks", "n_detected", "tpr",
    ]
    # 2 granularities x 2 tolerances x 2 alphas x 5 kind-group rows.
    assert len(result) == 2 * 2 * 2 * 5
    assert set(result["granularity"]) == {"impulse", "physical"}
    assert set(result["kind_group"]) == _KIND_GROUPS_ALL


def test_sweep_strike_detection_absent_kind_group_reports_nan_tpr() -> None:
    alarms = _alarms(5, alarming_at=[2])
    marks = _marks([("st", "01", "landmark-C_EG", 1, 2.0)])  # only "landmark" present
    result = sweep_strike_detection(alarms, marks, tolerances_s=[1.0], alphas=[0.05])

    vane = result[(result["kind_group"] == "vane-sweep") & (result["granularity"] == "impulse")]
    assert len(vane) == 1
    assert int(vane["n_marks"].iloc[0]) == 0
    assert math.isnan(vane["tpr"].iloc[0])

    landmark = result[
        (result["kind_group"] == "landmark") & (result["granularity"] == "impulse")
    ]
    assert int(landmark["n_marks"].iloc[0]) == 1
    assert int(landmark["n_detected"].iloc[0]) == 1
    assert landmark["tpr"].iloc[0] == pytest.approx(1.0)


def test_sweep_strike_detection_impulse_vs_physical_n_marks_differ() -> None:
    marks = _marks(
        [
            ("st", "02", "plate-gen_0", 1, 0.0),
            ("st", "02", "plate-gen_0", 2, 0.8),
        ]
    )
    alarms = _alarms(5, alarming_at=[0])
    result = sweep_strike_detection(alarms, marks, tolerances_s=[5.0], alphas=[0.05], gap_s=1.5)

    all_impulse = result[(result["kind_group"] == "ALL") & (result["granularity"] == "impulse")]
    all_physical = result[(result["kind_group"] == "ALL") & (result["granularity"] == "physical")]
    assert int(all_impulse["n_marks"].iloc[0]) == 2
    assert int(all_physical["n_marks"].iloc[0]) == 1


def test_sweep_strike_detection_invalid_alpha_raises() -> None:
    alarms = _alarms(5)
    marks = _marks([("st", "01", "landmark-C_EG", 1, 1.0)])
    with pytest.raises(ValueError, match="alpha"):
        sweep_strike_detection(alarms, marks, tolerances_s=[1.0], alphas=[0.0])


# ---------------------------------------------------------------------------
# evaluate_event_latency: first alarmed scored window AT-OR-AFTER the
# EARLIEST mark of each event ("onset"), missed beyond the search horizon.
# ---------------------------------------------------------------------------


def test_event_latency_uses_first_mark_as_onset_and_measures_to_first_alarm() -> None:
    alarms = _alarms(30, alarming_at=[10])
    marks = _marks(
        [
            ("st", "02", "plate-gen_0", 1, 0.0),
            ("st", "02", "plate-gen_0", 2, 0.8),  # onset = min offset = 0.0
        ]
    )
    result = evaluate_event_latency(alarms, marks, alpha=0.05, search_horizon_s=60.0)

    assert len(result) == 1
    assert bool(result.loc[0, "missed"]) is False
    assert result.loc[0, "latency_s"] == pytest.approx(10.0)
    assert result.loc[0, "kind_group"] == "plate-gen"


def test_event_latency_missed_beyond_horizon() -> None:
    alarms = _alarms(90, alarming_at=[80])
    marks = _marks([("st", "01", "landmark-C_EG", 1, 0.0)])
    result = evaluate_event_latency(alarms, marks, alpha=0.05, search_horizon_s=60.0)

    assert bool(result.loc[0, "missed"]) is True
    assert math.isnan(result.loc[0, "latency_s"])


def test_event_latency_ignores_alarms_before_onset() -> None:
    alarms = _alarms(30, alarming_at=[2, 20])
    marks = _marks([("st", "01", "landmark-C_EG", 1, 10.0)])
    result = evaluate_event_latency(alarms, marks, alpha=0.05, search_horizon_s=60.0)

    assert result.loc[0, "latency_s"] == pytest.approx(10.0)  # from t=20, not t=2


def test_event_latency_ignores_non_scored_rows() -> None:
    alarms = _alarms(20, alarming_at=[5])
    alarms.loc[5, "role"] = "consumed_for_calibration"
    marks = _marks([("st", "01", "landmark-C_EG", 1, 0.0)])
    result = evaluate_event_latency(alarms, marks, alpha=0.05, search_horizon_s=60.0)

    assert bool(result.loc[0, "missed"]) is True


def test_event_latency_invalid_search_horizon_raises() -> None:
    alarms = _alarms(5)
    marks = _marks([("st", "01", "landmark-C_EG", 1, 0.0)])
    with pytest.raises(ValueError, match="search_horizon_s"):
        evaluate_event_latency(alarms, marks, alpha=0.05, search_horizon_s=0.0)


# ---------------------------------------------------------------------------
# evaluate_strike_latency: nearest alarm AT-OR-AFTER each DEDUPLICATED
# physical strike, missed beyond its (shorter) search horizon.
# ---------------------------------------------------------------------------


def test_strike_latency_measures_per_physical_strike_not_per_event() -> None:
    alarms = _alarms(60, alarming_at=[1, 51])
    marks = _marks(
        [
            ("st", "02", "plate-gen_0", 1, 0.0),
            ("st", "02", "plate-gen_0", 2, 0.8),  # folds into physical strike @ t=0.0
            ("st", "02", "plate-gen_0", 3, 50.0),  # separate physical strike @ t=50.0
        ]
    )
    result = evaluate_strike_latency(alarms, marks, alpha=0.05, search_horizon_s=5.0, gap_s=1.5)

    assert len(result) == 2
    assert result["latency_s"].tolist() == pytest.approx([1.0, 1.0])
    assert result["n_impulses"].tolist() == [2, 1]


def test_strike_latency_missed_beyond_five_second_horizon() -> None:
    alarms = _alarms(60, alarming_at=[10])
    marks = _marks([("st", "01", "landmark-C_EG", 1, 0.0)])
    result = evaluate_strike_latency(alarms, marks, alpha=0.05, search_horizon_s=5.0)

    assert bool(result.loc[0, "missed"]) is True
    assert math.isnan(result.loc[0, "latency_s"])


# ---------------------------------------------------------------------------
# summarize_latency: median / IQR over DETECTED latencies only
# ---------------------------------------------------------------------------


def test_summarize_latency_median_and_iqr_over_detected_only() -> None:
    latency = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    missed = np.array([False, False, False, False, True])
    summary = summarize_latency(latency, missed)

    assert summary.n_total == 5
    assert summary.n_detected == 4
    assert summary.n_missed == 1
    assert summary.median_s == pytest.approx(2.5)
    assert summary.iqr_low_s == pytest.approx(1.75)
    assert summary.iqr_high_s == pytest.approx(3.25)


def test_summarize_latency_all_missed_returns_nan() -> None:
    summary = summarize_latency(np.array([np.nan, np.nan]), np.array([True, True]))

    assert summary.n_detected == 0
    assert math.isnan(summary.median_s)
    assert math.isnan(summary.iqr_low_s)
    assert math.isnan(summary.iqr_high_s)


def test_summarize_latency_is_frozen() -> None:
    summary = summarize_latency(np.array([1.0]), np.array([False]))
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.n_total = 99  # type: ignore[misc]


def test_summarize_latency_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        summarize_latency(np.array([1.0, 2.0]), np.array([False]))


def test_summarize_latency_is_a_dataclass_instance() -> None:
    summary = summarize_latency(np.array([1.0]), np.array([False]))
    assert isinstance(summary, LatencySummary)


# ---------------------------------------------------------------------------
# inter_mark_gaps: within-event consecutive-mark spacing (per event, scoped)
# ---------------------------------------------------------------------------


def test_inter_mark_gaps_basic_sequence() -> None:
    marks = _marks(
        [
            ("st", "01", "landmark-C_EG", 1, 0.0),
            ("st", "01", "landmark-C_EG", 2, 0.8),
            ("st", "01", "landmark-C_EG", 3, 1.7),
        ]
    )
    gaps = inter_mark_gaps(marks)

    assert gaps["gap_s"].tolist() == pytest.approx([0.8, 0.9])
    assert gaps["to_strike_no"].tolist() == [2, 3]
    assert (gaps["kind_group"] == "landmark").all()


def test_inter_mark_gaps_single_mark_event_has_no_rows() -> None:
    marks = _marks([("st", "07", "landmark-A_kugelschieber", 1, 0.0)])
    assert len(inter_mark_gaps(marks)) == 0


def test_inter_mark_gaps_scoped_per_event() -> None:
    marks = _marks(
        [
            ("pu", "02", "plate-gen_90", 6, 3.0),
            ("pu", "03", "plate-gen_180", 1, 3.2),
        ]
    )
    assert len(inter_mark_gaps(marks)) == 0  # different events -> no cross-event gap


# ---------------------------------------------------------------------------
# marks_per_event
# ---------------------------------------------------------------------------


def test_marks_per_event_counts_and_kind_group() -> None:
    marks = _marks(
        [
            ("st", "01", "landmark-C_EG", 1, 0.0),
            ("st", "01", "landmark-C_EG", 2, 0.8),
            ("st", "02", "plate-gen_0", 1, 10.0),
        ]
    )
    table = marks_per_event(marks)

    assert len(table) == 2
    row1 = table[table["event_id"] == "01"].iloc[0]
    assert row1["n_marks"] == 2
    assert row1["kind_group"] == "landmark"
    row2 = table[table["event_id"] == "02"].iloc[0]
    assert row2["n_marks"] == 1
    assert row2["kind_group"] == "plate-gen"
