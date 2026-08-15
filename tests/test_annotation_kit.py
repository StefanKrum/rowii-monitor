"""Tests for `scripts/annotation_kit.py`'s PURE helpers (snippet-window arithmetic,
burst-file boundary selection, offset<->UTC conversion, template validation) plus
one deliberate extension of `tests/test_make_demo_assets.py`'s own "pure vs
IO-touching" split: `extract_stream_clip`'s multi-file stitch/clamp behaviour,
exercised against SYNTHETIC Gantner containers (`tests/fixtures/gantner_builder.py`)
rather than real `ROWII_DATA_ROOT` data -- this is exactly the new, highest-risk
logic this task adds (boundary-crossing windows "must not crash"), and synthetic
`.dat` files make it fully, deterministically testable (mirrors `tests/
test_cli_smoke.py`'s own precedent of using `gantner_builder` for CLI-level
integration tests with no real data). Real disk reads against `ROWII_DATA_ROOT`,
WAV writes, spectrogram PNG rendering, and `index.html` assembly are exercised by
actually running the CLI against the real 080726 campaign data instead, not by a
test here.

Import convention mirrors `tests/test_make_demo_assets.py`: `scripts/` is not a
package, so the module under test is imported directly by inserting `scripts/`
onto `sys.path`.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import annotation_kit as ak  # noqa: E402

from rowii.io.dataset import BurstFile  # noqa: E402
from tests.fixtures.gantner_builder import build_gantner_file  # noqa: E402


def _utc(year: int, month: int, day: int, hour: int, minute: int, second: int) -> datetime:
    """`datetime(..., tzinfo=UTC)` shorthand -- mirrors `test_make_demo_assets.
    _utc`."""
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def _burst(start_utc_hint: datetime, path: Path | None = None) -> BurstFile:
    return BurstFile(
        path=path or Path(f"/fake/{start_utc_hint.isoformat()}.dat"),
        stream="RAWGeneratorMic__0",
        start_utc_hint=start_utc_hint,
    )


# ---------------------------------------------------------------------------
# 1. snippet_window
# ---------------------------------------------------------------------------


def test_snippet_window_pads_both_sides() -> None:
    start, end = _utc(2026, 7, 8, 10, 15, 0), _utc(2026, 7, 8, 10, 16, 0)  # 60s event

    w_start, w_end = ak.snippet_window(start, end)

    assert w_start == start - timedelta(seconds=15)
    assert w_end == end + timedelta(seconds=15)
    assert (w_end - w_start).total_seconds() == pytest.approx(90.0)


def test_snippet_window_vane_sweep_yields_3_5_min() -> None:
    start, end = _utc(2026, 7, 8, 10, 24, 0), _utc(2026, 7, 8, 10, 27, 0)  # 180s event

    w_start, w_end = ak.snippet_window(start, end)

    assert (w_end - w_start).total_seconds() == pytest.approx(210.0)


def test_snippet_window_rejects_inverted_event() -> None:
    start, end = _utc(2026, 7, 8, 10, 16, 0), _utc(2026, 7, 8, 10, 15, 0)

    with pytest.raises(ValueError, match="before"):
        ak.snippet_window(start, end)


# ---------------------------------------------------------------------------
# 2. files_covering_window
# ---------------------------------------------------------------------------


def test_files_covering_window_single_file_no_crossing() -> None:
    files = [_burst(_utc(2026, 7, 8, 10, 0, 0)), _burst(_utc(2026, 7, 8, 10, 12, 0))]

    selected = ak.files_covering_window(
        files, _utc(2026, 7, 8, 10, 0, 30), _utc(2026, 7, 8, 10, 2, 0)
    )

    assert [bf.start_utc_hint for bf in selected] == [_utc(2026, 7, 8, 10, 0, 0)]


def test_files_covering_window_stitches_across_one_boundary() -> None:
    # Mirrors the real ST vane-sweep window against the real 080726 file boundary
    # at 10:25:00 UTC.
    files = [
        _burst(_utc(2026, 7, 8, 10, 13, 2)),
        _burst(_utc(2026, 7, 8, 10, 25, 0)),
    ]

    selected = ak.files_covering_window(
        files, _utc(2026, 7, 8, 10, 23, 45), _utc(2026, 7, 8, 10, 27, 15)
    )

    assert [bf.start_utc_hint for bf in selected] == [
        _utc(2026, 7, 8, 10, 13, 2), _utc(2026, 7, 8, 10, 25, 0),
    ]


def test_files_covering_window_deliberately_unsorted_input() -> None:
    files = [_burst(_utc(2026, 7, 8, 10, 24, 0)), _burst(_utc(2026, 7, 8, 10, 0, 0))]

    selected = ak.files_covering_window(
        files, _utc(2026, 7, 8, 10, 0, 30), _utc(2026, 7, 8, 10, 1, 0)
    )

    assert [bf.start_utc_hint for bf in selected] == [_utc(2026, 7, 8, 10, 0, 0)]


def test_files_covering_window_clamps_to_earliest_file_when_window_starts_before_it() -> None:
    files = [_burst(_utc(2026, 7, 8, 10, 0, 0)), _burst(_utc(2026, 7, 8, 10, 12, 0))]

    selected = ak.files_covering_window(
        files, _utc(2026, 7, 8, 9, 50, 0), _utc(2026, 7, 8, 9, 58, 0)
    )

    assert [bf.start_utc_hint for bf in selected] == [_utc(2026, 7, 8, 10, 0, 0)]


def test_files_covering_window_skips_a_missing_middle_file() -> None:
    # A file that would have started ~10:24 (and covered the window) is simply
    # absent -- the next real file starts at 10:36.
    files = [_burst(_utc(2026, 7, 8, 10, 12, 0)), _burst(_utc(2026, 7, 8, 10, 36, 0))]

    selected = ak.files_covering_window(
        files, _utc(2026, 7, 8, 10, 20, 0), _utc(2026, 7, 8, 10, 30, 0)
    )

    # Only the file BEFORE the gap is a candidate: the file after it starts past
    # the window's own end, so it cannot possibly overlap.
    assert [bf.start_utc_hint for bf in selected] == [_utc(2026, 7, 8, 10, 12, 0)]


def test_files_covering_window_rejects_empty_files_and_bad_window() -> None:
    files = [_burst(_utc(2026, 7, 8, 10, 0, 0))]

    with pytest.raises(ValueError, match="non-empty"):
        ak.files_covering_window([], _utc(2026, 7, 8, 10, 0, 0), _utc(2026, 7, 8, 10, 0, 1))
    with pytest.raises(ValueError, match="must be after"):
        ak.files_covering_window(
            files, _utc(2026, 7, 8, 10, 0, 1), _utc(2026, 7, 8, 10, 0, 0)
        )


# ---------------------------------------------------------------------------
# 3. extract_stream_clip -- synthetic Gantner files (gantner_builder), no
#    ROWII_DATA_ROOT. rate_hz=1000.0 (1 ms/sample) keeps fixtures tiny; offset_ns=0
#    throughout (the raw/true-UTC shift itself is `make_demo_assets._shift_ts_ns`,
#    already covered by that module's own reuse -- this suite is about the
#    STITCHING, not the DAQ-clock-quirk arithmetic).
# ---------------------------------------------------------------------------

_RATE_HZ = 1000.0
_N = 2000  # 2 s/file at 1000 Hz


def _build_two_adjacent_files(tmp_path: Path, t0: datetime) -> list[BurstFile]:
    """Two contiguous 2 s synthetic files: A covers [t0, t0+2s), B covers
    [t0+2s, t0+4s). Channel 0 of A holds 0..1999 (its own sample index);
    channel 0 of B holds 10000..11999 -- disjoint ranges so a test can tell,
    from the VALUES alone, which file each returned sample came from.
    """
    t0_ns = round(t0.timestamp() * 1e9)
    data_a = np.zeros((_N, 1), dtype=np.float32)
    data_a[:, 0] = np.arange(_N, dtype=np.float32)
    data_b = np.zeros((_N, 1), dtype=np.float32)
    data_b[:, 0] = np.arange(_N, dtype=np.float32) + 10_000.0

    path_a = build_gantner_file(
        tmp_path / "a.dat", ["Ch0"], data_a, t0_ns=t0_ns, rate_hz=_RATE_HZ
    )
    path_b = build_gantner_file(
        tmp_path / "b.dat", ["Ch0"], data_b,
        t0_ns=t0_ns + round(_N / _RATE_HZ * 1e9), rate_hz=_RATE_HZ,
    )
    return [_burst(t0, path_a), _burst(t0 + timedelta(seconds=2), path_b)]


def test_extract_stream_clip_single_file_no_crossing(tmp_path: Path) -> None:
    t0 = _utc(2026, 7, 8, 10, 0, 0)
    files = _build_two_adjacent_files(tmp_path, t0)

    clip = ak.extract_stream_clip(
        files, offset_ns=0, channel_index=0,
        window_start_utc=t0 + timedelta(seconds=0.1),
        window_end_utc=t0 + timedelta(seconds=0.6),
    )

    assert not clip.clamped
    assert clip.note == ""
    np.testing.assert_allclose(clip.samples, np.arange(100, 600, dtype=np.float32))
    assert clip.rate_hz == pytest.approx(_RATE_HZ, rel=0.01)


def test_extract_stream_clip_stitches_across_the_boundary(tmp_path: Path) -> None:
    t0 = _utc(2026, 7, 8, 10, 0, 0)
    files = _build_two_adjacent_files(tmp_path, t0)

    clip = ak.extract_stream_clip(
        files, offset_ns=0, channel_index=0,
        window_start_utc=t0 + timedelta(seconds=1.5),
        window_end_utc=t0 + timedelta(seconds=2.5),
    )

    assert not clip.clamped, clip.note
    expected = np.concatenate(
        [np.arange(1500, 2000, dtype=np.float32), np.arange(10_000, 10_500, dtype=np.float32)]
    )
    np.testing.assert_allclose(clip.samples, expected)
    assert clip.samples.size == 1000
    # Chronological order preserved: file A's tail strictly precedes file B's head.
    assert clip.covered_start_utc < clip.covered_end_utc
    assert clip.covered_start_utc - (t0 + timedelta(seconds=1.5)) < timedelta(milliseconds=5)


def test_extract_stream_clip_selects_requested_channel(tmp_path: Path) -> None:
    t0 = _utc(2026, 7, 8, 10, 0, 0)
    data = np.zeros((_N, 2), dtype=np.float32)
    data[:, 0] = np.arange(_N, dtype=np.float32)          # channel 0: 0..1999
    data[:, 1] = np.arange(_N, dtype=np.float32) + 5_000.0  # channel 1: 5000..6999
    path = build_gantner_file(
        tmp_path / "two_ch.dat", ["Ch0", "Ch1"], data,
        t0_ns=round(t0.timestamp() * 1e9), rate_hz=_RATE_HZ,
    )
    files = [_burst(t0, path)]

    clip0 = ak.extract_stream_clip(
        files, 0, 0, t0 + timedelta(seconds=0.1), t0 + timedelta(seconds=0.2)
    )
    clip1 = ak.extract_stream_clip(
        files, 0, 1, t0 + timedelta(seconds=0.1), t0 + timedelta(seconds=0.2)
    )

    np.testing.assert_allclose(clip0.samples, np.arange(100, 200, dtype=np.float32))
    np.testing.assert_allclose(clip1.samples, np.arange(5_100, 5_200, dtype=np.float32))


def test_extract_stream_clip_clamps_when_window_starts_before_earliest_file(
    tmp_path: Path,
) -> None:
    t0 = _utc(2026, 7, 8, 10, 0, 0)
    files = _build_two_adjacent_files(tmp_path, t0)

    clip = ak.extract_stream_clip(
        files, offset_ns=0, channel_index=0,
        window_start_utc=t0 - timedelta(seconds=1.0),
        window_end_utc=t0 + timedelta(seconds=0.5),
    )

    assert clip.clamped
    assert "start clamped" in clip.note
    assert "end clamped" not in clip.note
    # Only the real data (0..499) comes back -- nothing invented for the missing
    # second before the recording starts.
    np.testing.assert_allclose(clip.samples, np.arange(0, 500, dtype=np.float32))


def test_extract_stream_clip_clamps_when_window_extends_past_latest_file(
    tmp_path: Path,
) -> None:
    t0 = _utc(2026, 7, 8, 10, 0, 0)
    files = _build_two_adjacent_files(tmp_path, t0)  # B ends at t0+4s

    clip = ak.extract_stream_clip(
        files, offset_ns=0, channel_index=0,
        window_start_utc=t0 + timedelta(seconds=3.5),
        window_end_utc=t0 + timedelta(seconds=5.0),
    )

    assert clip.clamped
    assert "end clamped" in clip.note
    assert "start clamped" not in clip.note
    np.testing.assert_allclose(clip.samples, np.arange(11_500, 12_000, dtype=np.float32))


def test_extract_stream_clip_raises_on_a_window_entirely_inside_a_data_gap(
    tmp_path: Path,
) -> None:
    t0 = _utc(2026, 7, 8, 10, 0, 0)
    # A covers [t0, t0+2s); the next real file D starts at t0+6s (a missing
    # "B"/"C" pair in between) -- the window falls entirely in that gap.
    data_a = np.zeros((_N, 1), dtype=np.float32)
    path_a = build_gantner_file(
        tmp_path / "a.dat", ["Ch0"], data_a, t0_ns=round(t0.timestamp() * 1e9), rate_hz=_RATE_HZ
    )
    t0_d = t0 + timedelta(seconds=6)
    path_d = build_gantner_file(
        tmp_path / "d.dat", ["Ch0"], data_a, t0_ns=round(t0_d.timestamp() * 1e9), rate_hz=_RATE_HZ
    )
    files = [_burst(t0, path_a), _burst(t0_d, path_d)]

    with pytest.raises(ValueError, match="no overlap"):
        ak.extract_stream_clip(
            files, offset_ns=0, channel_index=0,
            window_start_utc=t0 + timedelta(seconds=3),
            window_end_utc=t0 + timedelta(seconds=5),
        )


# ---------------------------------------------------------------------------
# 4. parse_extra_offsets / collect_offsets
# ---------------------------------------------------------------------------


def test_parse_extra_offsets_semicolon_separated_and_empty() -> None:
    assert ak.parse_extra_offsets("12.3;12.9") == pytest.approx([12.3, 12.9])
    assert ak.parse_extra_offsets("") == []
    assert ak.parse_extra_offsets("   ") == []
    # blank tokens (from a trailing/doubled ';') are skipped, not rejected
    assert ak.parse_extra_offsets("12.3;;13.0;") == pytest.approx([12.3, 13.0])


def test_parse_extra_offsets_rejects_malformed_token() -> None:
    with pytest.raises(ValueError, match="not a number"):
        ak.parse_extra_offsets("12.3;abc")


def test_collect_offsets_orders_primary_then_extra_and_allows_gaps() -> None:
    row = {
        "strike1_offset_s": "", "strike2_offset_s": "10.0", "strike3_offset_s": "20.0",
        "extra_offsets_s": "30.0;40.0",
    }

    assert ak.collect_offsets(row) == pytest.approx([10.0, 20.0, 30.0, 40.0])


def test_collect_offsets_all_empty_yields_empty_list() -> None:
    row = {
        "strike1_offset_s": "", "strike2_offset_s": "", "strike3_offset_s": "",
        "extra_offsets_s": "",
    }

    assert ak.collect_offsets(row) == []


def test_collect_offsets_rejects_non_numeric_primary_cell() -> None:
    row = {
        "strike1_offset_s": "twelve", "strike2_offset_s": "", "strike3_offset_s": "",
        "extra_offsets_s": "",
    }

    with pytest.raises(ValueError, match="not a number"):
        ak.collect_offsets(row)


# ---------------------------------------------------------------------------
# 5. validate_offsets
# ---------------------------------------------------------------------------


def test_validate_offsets_accepts_a_valid_increasing_sequence() -> None:
    ak.validate_offsets([1.0, 2.5, 10.0], snippet_duration_s=90.0, event_id="01")  # no raise


def test_validate_offsets_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"out of range"):
        ak.validate_offsets([-0.1, 5.0], snippet_duration_s=90.0, event_id="01")
    with pytest.raises(ValueError, match=r"out of range"):
        ak.validate_offsets([5.0, 91.0], snippet_duration_s=90.0, event_id="01")


def test_validate_offsets_boundary_values_are_inclusive() -> None:
    ak.validate_offsets([0.0, 90.0], snippet_duration_s=90.0, event_id="01")  # no raise


def test_validate_offsets_rejects_non_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ak.validate_offsets([5.0, 5.0], snippet_duration_s=90.0, event_id="01")
    with pytest.raises(ValueError, match="strictly increasing"):
        ak.validate_offsets([10.0, 5.0], snippet_duration_s=90.0, event_id="01")


# ---------------------------------------------------------------------------
# 6. compile_row / compile_template
# ---------------------------------------------------------------------------


def _template_row(**overrides: str) -> dict[str, str]:
    row = {
        "session": "st", "event_id": "02", "kind": "plate-gen_0",
        "snippet_start_utc": "2026-07-08T10:16:45+00:00",
        "snippet_end_utc": "2026-07-08T10:18:15+00:00",
        "gen_wav": "st/event_02_plate-gen_0__gen.wav",
        "tur_wav": "st/event_02_plate-gen_0__tur.wav",
        "expected_strikes": "3",
        "strike1_offset_s": "", "strike2_offset_s": "", "strike3_offset_s": "",
        "extra_offsets_s": "", "confidence": "", "notes": "",
    }
    row.update(overrides)
    return row


def test_compile_row_empty_offsets_yields_no_strikes() -> None:
    assert ak.compile_row(_template_row()) == []


def test_compile_row_converts_offsets_to_absolute_utc() -> None:
    row = _template_row(
        strike1_offset_s="15.2", strike2_offset_s="15.9", strike3_offset_s="16.4",
        confidence="high", notes="clearly audible",
    )

    strikes = ak.compile_row(row)
    snippet_start = datetime.fromisoformat("2026-07-08T10:16:45+00:00")

    assert [s.strike_no for s in strikes] == [1, 2, 3]
    assert strikes[0].strike_utc == snippet_start + timedelta(seconds=15.2)
    assert strikes[-1].strike_utc == snippet_start + timedelta(seconds=16.4)
    assert all(s.confidence == "high" and s.notes == "clearly audible" for s in strikes)
    assert all(s.session == "st" and s.event_id == "02" for s in strikes)
    assert all(s.kind == "plate-gen_0" for s in strikes)


def test_compile_row_vane_sweep_uses_extra_offsets_only() -> None:
    # "sweep" carries no strike1/2/3 -- only optionally-heard individual vane
    # strikes via extra_offsets_s, per the index.html workflow instructions.
    row = _template_row(
        kind="vane-sweep", expected_strikes="sweep",
        snippet_start_utc="2026-07-08T10:23:45+00:00",
        snippet_end_utc="2026-07-08T10:27:15+00:00",
        extra_offsets_s="42.0;90.5", notes="Sweep 15s-195s, 2 individual strikes audible",
    )

    strikes = ak.compile_row(row)

    assert [s.strike_no for s in strikes] == [1, 2]
    assert strikes[0].strike_utc == datetime.fromisoformat("2026-07-08T10:23:45+00:00") + timedelta(
        seconds=42.0
    )


def test_compile_row_propagates_offset_validation_errors() -> None:
    row = _template_row(strike1_offset_s="200.0")  # far past the ~90s snippet

    with pytest.raises(ValueError, match="out of range"):
        ak.compile_row(row)


def test_compile_template_aggregates_rows_and_skips_unannotated() -> None:
    df = pd.DataFrame(
        [
            _template_row(event_id="01", strike1_offset_s="10.0"),
            _template_row(event_id="02"),  # not annotated -> contributes nothing
            _template_row(event_id="03", strike1_offset_s="5.0", strike2_offset_s="6.0"),
        ]
    )

    strikes = ak.compile_template(df)

    assert [s.event_id for s in strikes] == ["01", "03", "03"]


def test_compile_template_rejects_missing_column() -> None:
    df = pd.DataFrame([_template_row()]).drop(columns=["confidence"])

    with pytest.raises(ValueError, match="missing column"):
        ak.compile_template(df)


def test_compile_template_rejects_mixed_sessions() -> None:
    df = pd.DataFrame([_template_row(session="st"), _template_row(session="pu")])

    with pytest.raises(ValueError, match="multiple sessions"):
        ak.compile_template(df)


# ---------------------------------------------------------------------------
# 7. Template/compiled CSV I/O roundtrips
# ---------------------------------------------------------------------------


def test_write_then_read_template_csv_roundtrips_and_keeps_empty_cells_empty(
    tmp_path: Path,
) -> None:
    rows = [
        ak.TemplateRow(
            session="st", event_id="01", kind="landmark-C_EG",
            snippet_start_utc=_utc(2026, 7, 8, 10, 14, 45),
            snippet_end_utc=_utc(2026, 7, 8, 10, 16, 15),
            gen_wav="st/event_01_landmark-C_EG__gen.wav",
            tur_wav="st/event_01_landmark-C_EG__tur.wav",
            expected_strikes="3",
        ),
        ak.TemplateRow(
            session="st", event_id="09", kind="vane-sweep",
            snippet_start_utc=_utc(2026, 7, 8, 10, 23, 45),
            snippet_end_utc=_utc(2026, 7, 8, 10, 27, 15),
            gen_wav="st/event_09_vane-sweep__gen.wav",
            tur_wav="st/event_09_vane-sweep__tur.wav",
            expected_strikes="sweep",
            notes_prefill="tur: end clamped by 1.234s (no later data available)",
        ),
    ]

    path = ak.write_template_csv(rows, tmp_path / "annotation_template_st.csv")
    df = ak.read_template_csv(path)

    assert list(df.columns) == list(ak.TEMPLATE_CSV_COLUMNS)
    assert len(df) == 2
    assert df.iloc[0]["event_id"] == "01"          # zero-padding survives the roundtrip
    assert df.iloc[0]["strike1_offset_s"] == ""     # empty, not "nan"
    assert df.iloc[0]["notes"] == ""
    assert df.iloc[1]["expected_strikes"] == "sweep"
    assert "end clamped" in df.iloc[1]["notes"]


def test_write_compiled_csv_has_provenance_comment_and_correct_rows(tmp_path: Path) -> None:
    strikes = [
        ak.CompiledStrike(
            session="st", event_id="01", kind="landmark-C_EG", strike_no=1,
            strike_utc=_utc(2026, 7, 8, 10, 15, 12), confidence="high", notes="",
        ),
        ak.CompiledStrike(
            session="st", event_id="01", kind="landmark-C_EG", strike_no=2,
            strike_utc=_utc(2026, 7, 8, 10, 15, 30), confidence="high", notes="",
        ),
    ]
    out_path = tmp_path / "080726_strikes_seconds_st.csv"
    template_path = tmp_path / "annotation_template_st.csv"

    ak.write_compiled_csv(
        strikes, out_path, source_path=template_path, compiled_date=date(2026, 8, 11),
        provenance="manual audio/spectrogram annotation against results/annotation-kit/080726/.",
        command_name="compile --template <template>",
    )

    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("#")
    assert "2026-08-11" in text
    assert str(template_path) in text
    assert "manual audio/spectrogram annotation" in text
    assert "compile --template <template>" in text

    df = pd.read_csv(out_path, comment="#")
    assert list(df.columns) == list(ak.COMPILED_CSV_COLUMNS)
    assert df["strike_no"].tolist() == [1, 2]
    assert df["strike_utc"].iloc[0] == "2026-07-08T10:15:12+00:00"


def test_write_compiled_csv_marks_provenance_differs_from_template_provenance(
    tmp_path: Path,
) -> None:
    # compile-marks writes a DIFFERENT provenance sentence than compile (task
    # requirement: "provenance header (interactive per-strike annotation by the
    # author)") -- same writer, different call-site text.
    strikes = [
        ak.CompiledStrike(
            session="pu", event_id="03", kind="plate-gen_180", strike_no=1,
            strike_utc=_utc(2026, 7, 8, 12, 45, 12), confidence="high", notes="",
        ),
    ]
    out_path = tmp_path / "080726_strikes_seconds_pu.csv"
    csv_path = tmp_path / "annotation_marks_pu.csv"

    ak.write_compiled_csv(
        strikes, out_path, source_path=csv_path, compiled_date=date(2026, 8, 15),
        provenance="interactive per-strike annotation (scripts/annotation_kit.py build's "
        "index.html) by the author.",
        command_name="compile-marks --csv <marks csv>",
    )

    text = out_path.read_text(encoding="utf-8")
    assert "interactive per-strike annotation" in text
    assert "by the author" in text
    assert "compile-marks --csv <marks csv>" in text
    assert str(csv_path) in text


# ---------------------------------------------------------------------------
# 8. flat_spectrogram_width_px -- pure duration -> pixel-width formula behind
#    the interactive UI's linear click<->time mapping
# ---------------------------------------------------------------------------


def test_flat_spectrogram_width_px_matches_task_spec_examples() -> None:
    assert ak.flat_spectrogram_width_px(90.0) == 1800
    assert ak.flat_spectrogram_width_px(210.0) == 4200  # the vane-sweep snippet


def test_flat_spectrogram_width_px_rounds_to_nearest_pixel() -> None:
    # 20 px/s * 12.345s = 246.9 -> rounds to 247, not truncated to 246.
    assert ak.flat_spectrogram_width_px(12.345) == 247


def test_flat_spectrogram_width_px_no_min_max_clamp_unlike_labeled_spectrogram() -> None:
    # Deliberately unclamped (unlike render_spectrogram_png's own
    # _SPEC_MIN_WIDTH_PX/_SPEC_MAX_WIDTH_PX): a very short or very long snippet
    # still scales exactly linearly, which the click-to-time mapping depends on.
    assert ak.flat_spectrogram_width_px(1.0) == 20
    assert ak.flat_spectrogram_width_px(600.0) == 12000


# ---------------------------------------------------------------------------
# 9. events_meta.json -- EventMeta generation from template CSVs + write/load
#    roundtrip
# ---------------------------------------------------------------------------


def _write_synthetic_template(tmp_path: Path, session: str, rows: list[ak.TemplateRow]) -> Path:
    return ak.write_template_csv(rows, tmp_path / f"annotation_template_{session}.csv")


def _sample_template_rows(session: str = "st") -> list[ak.TemplateRow]:
    return [
        ak.TemplateRow(
            session=session, event_id="01", kind="landmark-C_EG",
            snippet_start_utc=_utc(2026, 7, 8, 10, 14, 45),
            snippet_end_utc=_utc(2026, 7, 8, 10, 16, 15),
            gen_wav=f"{session}/event_01_landmark-C_EG__gen.wav",
            tur_wav=f"{session}/event_01_landmark-C_EG__tur.wav",
            expected_strikes="3",
        ),
        ak.TemplateRow(
            session=session, event_id="09", kind="vane-sweep",
            snippet_start_utc=_utc(2026, 7, 8, 10, 23, 45),
            snippet_end_utc=_utc(2026, 7, 8, 10, 27, 15),
            gen_wav=f"{session}/event_09_vane-sweep__gen.wav",
            tur_wav=f"{session}/event_09_vane-sweep__tur.wav",
            expected_strikes="sweep",
        ),
    ]


def test_events_meta_from_templates_derives_all_fields(tmp_path: Path) -> None:
    _write_synthetic_template(tmp_path, "st", _sample_template_rows("st"))

    metas = ak.events_meta_from_templates(tmp_path)

    assert len(metas) == 2
    first = metas[0]
    assert first.session == "st"
    assert first.event_id == "01"
    assert first.kind == "landmark-C_EG"
    assert first.snippet_start_utc == _utc(2026, 7, 8, 10, 14, 45)
    assert first.duration_s == pytest.approx(90.0)
    assert first.gen_wav == "st/event_01_landmark-C_EG__gen.wav"
    assert first.tur_wav == "st/event_01_landmark-C_EG__tur.wav"
    assert first.gen_flat_png == "st/event_01_landmark-C_EG__gen_flat.png"
    assert first.tur_flat_png == "st/event_01_landmark-C_EG__tur_flat.png"
    assert first.expected_strikes == "3"

    sweep = metas[1]
    assert sweep.duration_s == pytest.approx(210.0)
    assert sweep.expected_strikes == "sweep"
    assert sweep.gen_flat_png == "st/event_09_vane-sweep__gen_flat.png"


def test_events_meta_from_templates_covers_every_session_on_disk(tmp_path: Path) -> None:
    # Mirrors render_static_index_html's own "rebuild from whatever's on disk"
    # contract: BOTH sessions' templates are picked up, not just one -- a
    # partial `build --session st` followed later by `build --session pu`
    # still yields a complete events_meta.json.
    _write_synthetic_template(tmp_path, "st", _sample_template_rows("st"))
    _write_synthetic_template(tmp_path, "pu", _sample_template_rows("pu"))

    metas = ak.events_meta_from_templates(tmp_path)

    assert {m.session for m in metas} == {"st", "pu"}
    assert len(metas) == 4


def test_events_meta_from_templates_rejects_malformed_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "annotation_template_st.csv"
    path.write_text(
        "session,event_id,kind,snippet_start_utc,snippet_end_utc,gen_wav,tur_wav,"
        "expected_strikes,strike1_offset_s,strike2_offset_s,strike3_offset_s,"
        "extra_offsets_s,confidence,notes\n"
        "st,01,landmark-C_EG,not-a-timestamp,2026-07-08T10:16:15+00:00,"
        "st/event_01__gen.wav,st/event_01__tur.wav,3,,,,,,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed snippet timestamp"):
        ak.events_meta_from_templates(tmp_path)


def test_write_events_meta_json_roundtrips_via_load_events_meta(tmp_path: Path) -> None:
    _write_synthetic_template(tmp_path, "pu", _sample_template_rows("pu"))

    json_path = ak.write_events_meta_json(tmp_path)
    assert json_path == tmp_path / "events_meta.json"
    assert json_path.is_file()

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    assert len(raw) == 2
    assert raw[0]["session"] == "pu"
    assert raw[0]["duration_s"] == pytest.approx(90.0)

    lookup = ak.load_events_meta(json_path)
    assert lookup[("pu", "01")].duration_s == pytest.approx(90.0)
    assert lookup[("pu", "09")].expected_strikes == "sweep"
    assert lookup[("pu", "09")].snippet_start_utc == _utc(2026, 7, 8, 10, 23, 45)


def test_load_events_meta_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        ak.load_events_meta(tmp_path / "does_not_exist.json")


def test_load_events_meta_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "events_meta.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        ak.load_events_meta(path)


def test_load_events_meta_rejects_non_array_json(tmp_path: Path) -> None:
    path = tmp_path / "events_meta.json"
    path.write_text('{"session": "st"}', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON array"):
        ak.load_events_meta(path)


def test_load_events_meta_rejects_entry_missing_a_field(tmp_path: Path) -> None:
    path = tmp_path / "events_meta.json"
    path.write_text(json.dumps([{"session": "st", "event_id": "01"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="entry #0"):
        ak.load_events_meta(path)


# ---------------------------------------------------------------------------
# 10. compile_marks -- offset<->UTC math + validation via an events_meta.json
#     lookup (the interactive index.html's "Export CSV" -> compile-marks path)
# ---------------------------------------------------------------------------


def _sample_events_lookup() -> dict[tuple[str, str], ak.EventMeta]:
    return {
        ("st", "01"): ak.EventMeta(
            session="st", event_id="01", kind="landmark-C_EG",
            snippet_start_utc=_utc(2026, 7, 8, 10, 14, 45), duration_s=90.0,
            gen_wav="st/event_01_landmark-C_EG__gen.wav",
            tur_wav="st/event_01_landmark-C_EG__tur.wav",
            gen_flat_png="st/event_01_landmark-C_EG__gen_flat.png",
            tur_flat_png="st/event_01_landmark-C_EG__tur_flat.png",
            expected_strikes="3",
        ),
        ("st", "09"): ak.EventMeta(
            session="st", event_id="09", kind="vane-sweep",
            snippet_start_utc=_utc(2026, 7, 8, 10, 23, 45), duration_s=210.0,
            gen_wav="st/event_09_vane-sweep__gen.wav",
            tur_wav="st/event_09_vane-sweep__tur.wav",
            gen_flat_png="st/event_09_vane-sweep__gen_flat.png",
            tur_flat_png="st/event_09_vane-sweep__tur_flat.png",
            expected_strikes="sweep",
        ),
    }


def _marks_row(**overrides: str) -> dict[str, str]:
    row = {
        "session": "st", "event_id": "01", "kind": "landmark-C_EG",
        "strike_no": "1", "offset_s": "15.200",
        "strike_utc": "2026-07-08T10:15:00.200+00:00",
        "confidence": "high", "notes": "",
    }
    row.update(overrides)
    return row


def test_compile_marks_converts_offset_to_absolute_utc() -> None:
    df = pd.DataFrame(
        [
            _marks_row(strike_no="1", offset_s="15.200"),
            _marks_row(strike_no="2", offset_s="15.900"),
            _marks_row(strike_no="3", offset_s="16.400"),
        ]
    )

    strikes = ak.compile_marks(df, _sample_events_lookup())

    snippet_start = _utc(2026, 7, 8, 10, 14, 45)
    assert [s.strike_no for s in strikes] == [1, 2, 3]
    assert strikes[0].strike_utc == snippet_start + timedelta(seconds=15.2)
    assert strikes[-1].strike_utc == snippet_start + timedelta(seconds=16.4)
    assert all(s.kind == "landmark-C_EG" for s in strikes)
    assert all(s.session == "st" and s.event_id == "01" for s in strikes)


def test_compile_marks_reorders_by_strike_no_not_row_order() -> None:
    # Rows given OUT of chronological (strike_no) order -- compile_marks must
    # sort by strike_no before validating/converting, not trust raw row order.
    df = pd.DataFrame(
        [
            _marks_row(strike_no="2", offset_s="15.900"),
            _marks_row(strike_no="1", offset_s="15.200"),
        ]
    )

    strikes = ak.compile_marks(df, _sample_events_lookup())

    assert [s.strike_no for s in strikes] == [1, 2]
    assert strikes[0].strike_utc == _utc(2026, 7, 8, 10, 14, 45) + timedelta(seconds=15.2)
    assert strikes[1].strike_utc == _utc(2026, 7, 8, 10, 14, 45) + timedelta(seconds=15.9)


def test_compile_marks_rejects_offset_out_of_range() -> None:
    df = pd.DataFrame([_marks_row(offset_s="95.0")])  # event 01 duration is 90s

    with pytest.raises(ValueError, match="out of range"):
        ak.compile_marks(df, _sample_events_lookup())


def test_compile_marks_rejects_non_increasing_offsets() -> None:
    df = pd.DataFrame(
        [
            _marks_row(strike_no="1", offset_s="10.0"),
            _marks_row(strike_no="2", offset_s="10.0"),
        ]
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        ak.compile_marks(df, _sample_events_lookup())


def test_compile_marks_rejects_unknown_event() -> None:
    df = pd.DataFrame([_marks_row(event_id="99")])

    with pytest.raises(ValueError, match="no matching entry"):
        ak.compile_marks(df, _sample_events_lookup())


def test_compile_marks_rejects_mixed_sessions() -> None:
    df = pd.DataFrame([_marks_row(session="st"), _marks_row(session="pu", event_id="01")])

    with pytest.raises(ValueError, match="multiple sessions"):
        ak.compile_marks(df, _sample_events_lookup())


def test_compile_marks_rejects_missing_column() -> None:
    df = pd.DataFrame([_marks_row()]).drop(columns=["offset_s"])

    with pytest.raises(ValueError, match="missing column"):
        ak.compile_marks(df, _sample_events_lookup())


def test_compile_marks_rejects_non_numeric_offset() -> None:
    df = pd.DataFrame([_marks_row(offset_s="not-a-number")])

    with pytest.raises(ValueError, match="not a number"):
        ak.compile_marks(df, _sample_events_lookup())


def test_compile_marks_vane_sweep_event_uses_its_own_duration() -> None:
    df = pd.DataFrame(
        [_marks_row(event_id="09", kind="vane-sweep", strike_no="1", offset_s="195.0")]
    )

    strikes = ak.compile_marks(df, _sample_events_lookup())

    assert strikes[0].strike_utc == _utc(2026, 7, 8, 10, 23, 45) + timedelta(seconds=195.0)
    assert strikes[0].kind == "vane-sweep"


def test_compile_marks_uses_meta_kind_not_csv_kind() -> None:
    # events_meta.json is authoritative; a stale/incorrect `kind` cell in the
    # marks CSV itself is ignored rather than trusted.
    df = pd.DataFrame([_marks_row(kind="WRONG-KIND")])

    strikes = ak.compile_marks(df, _sample_events_lookup())

    assert strikes[0].kind == "landmark-C_EG"


def test_compile_marks_output_sorted_by_event_id_then_strike_no() -> None:
    df = pd.DataFrame(
        [
            _marks_row(event_id="09", kind="vane-sweep", strike_no="1", offset_s="42.0"),
            _marks_row(event_id="01", strike_no="2", offset_s="15.9"),
            _marks_row(event_id="01", strike_no="1", offset_s="15.2"),
        ]
    )

    strikes = ak.compile_marks(df, _sample_events_lookup())

    assert [(s.event_id, s.strike_no) for s in strikes] == [("01", 1), ("01", 2), ("09", 1)]


# ---------------------------------------------------------------------------
# 11. read_marks_csv -- I/O roundtrip
# ---------------------------------------------------------------------------


def test_read_marks_csv_roundtrips_and_keeps_columns(tmp_path: Path) -> None:
    path = tmp_path / "annotation_marks_st.csv"
    path.write_text(
        "session,event_id,kind,strike_no,offset_s,strike_utc,confidence,notes\n"
        'st,01,landmark-C_EG,1,15.200,2026-07-08T10:15:00.200Z,high,"clear, audible"\n',
        encoding="utf-8",
    )

    df = ak.read_marks_csv(path)

    assert list(df.columns) == list(ak.MARKS_CSV_COLUMNS)
    assert df.iloc[0]["offset_s"] == "15.200"
    assert df.iloc[0]["notes"] == "clear, audible"
