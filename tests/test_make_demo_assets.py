"""Tests for `scripts/make_demo_assets.py`'s PURE helpers only (window-center/dodge
arithmetic, UTC->sample-index conversion, burst-file selection, peak normalization,
longest-segment selection) -- synthetic fixtures throughout, no `ROWII_DATA_ROOT` and
no real `.dat`/`.wav` files anywhere (mirrors `tests/test_conformal.py`'s "pure-math,
no real data" posture). The IO-touching parts of that script (discovery, `read_gantner`
slicing, WAV writing, HTML templating) are exercised by actually running the CLI
against the real 080726 campaign data as part of this task, not by a unit test here.

Import convention mirrors `tests/test_warm_cache.py`: `scripts/` is not a package, so
the module under test is imported directly by inserting `scripts/` onto `sys.path`.
"""
from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import make_demo_assets as mda  # noqa: E402

from rowii.io.dataset import BurstFile  # noqa: E402


def _utc(year: int, month: int, day: int, hour: int, minute: int, second: int) -> datetime:
    """`datetime(..., tzinfo=UTC)` shorthand -- every timestamp this module works
    with is timezone-aware true-UTC (never a naive local time), so every fixture
    below is built the same way real `segments.csv`/events-CSV rows are parsed.
    Explicit named parameters (not `*args: int`): mypy cannot verify a variadic
    int-tuple unpack leaves `datetime`'s own `tzinfo` positional slot unfilled."""
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


# ---------------------------------------------------------------------------
# 1. longest_segment_per_cluster
# ---------------------------------------------------------------------------


def test_longest_segment_per_cluster_picks_longest_and_earliest_tie() -> None:
    segments = [
        mda.Segment(_utc(2026, 7, 8, 11, 0, 0), _utc(2026, 7, 8, 11, 0, 1), cluster_id=-1),
        mda.Segment(_utc(2026, 7, 8, 12, 0, 0), _utc(2026, 7, 8, 12, 0, 5), cluster_id=0),
        mda.Segment(_utc(2026, 7, 8, 14, 0, 0), _utc(2026, 7, 8, 14, 0, 6), cluster_id=0),
        # cluster 1: two 719 s segments (a real tie, as in the actual 080726-pu_strikes
        # segments.csv) -- the EARLIER-starting one must win, deliberately fed here in
        # reverse chronological order to prove the tie-break does not depend on input
        # order.
        mda.Segment(_utc(2026, 7, 8, 13, 0, 0), _utc(2026, 7, 8, 13, 11, 59), cluster_id=1),
        mda.Segment(_utc(2026, 7, 8, 12, 10, 0), _utc(2026, 7, 8, 12, 21, 59), cluster_id=1),
    ]

    result = mda.longest_segment_per_cluster(segments)

    assert set(result) == {0, 1}  # cluster_id == -1 (invalid/transition) never wins
    assert result[0].start_utc == _utc(2026, 7, 8, 14, 0, 0)  # the 6 s one, not the 5 s one
    assert result[0].duration_s == pytest.approx(6.0)
    assert result[1].start_utc == _utc(2026, 7, 8, 12, 10, 0)  # earlier of the two 719 s ties


# ---------------------------------------------------------------------------
# 2. center_window
# ---------------------------------------------------------------------------


def test_center_window_centers_on_midpoint_and_may_overflow_a_short_segment() -> None:
    seg_start, seg_end = _utc(2026, 7, 8, 13, 10, 0), _utc(2026, 7, 8, 13, 21, 59)  # 719 s

    w_start, w_end = mda.center_window(seg_start, seg_end, duration_s=10.0)

    assert (w_end - w_start).total_seconds() == pytest.approx(10.0)
    midpoint = seg_start + (seg_end - seg_start) / 2
    assert w_start + (w_end - w_start) / 2 == midpoint

    # Real case: cluster 0's longest segment in 080726-pu_strikes/.../segments.csv is
    # only 6 s -- shorter than the 10 s clip duration, so the window MUST extend past
    # the segment's own bounds (there is no way to fit a 10 s window inside 6 s).
    short_start, short_end = _utc(2026, 7, 8, 14, 19, 44), _utc(2026, 7, 8, 14, 19, 50)  # 6 s

    short_w_start, short_w_end = mda.center_window(short_start, short_end, duration_s=10.0)

    assert (short_w_end - short_w_start).total_seconds() == pytest.approx(10.0)
    assert short_w_start < short_start
    assert short_w_end > short_end


# ---------------------------------------------------------------------------
# 3. has_collision
# ---------------------------------------------------------------------------


def test_has_collision_true_false_and_padding() -> None:
    events = [(_utc(2026, 7, 8, 12, 43, 0), _utc(2026, 7, 8, 12, 44, 0))]

    # Direct overlap.
    assert mda.has_collision(
        _utc(2026, 7, 8, 12, 43, 30), _utc(2026, 7, 8, 12, 43, 40), events, pad_s=0.0
    )
    # No overlap, far away, even with generous padding.
    assert not mda.has_collision(
        _utc(2026, 7, 8, 14, 0, 0), _utc(2026, 7, 8, 14, 0, 10), events, pad_s=10.0
    )
    # Outside the RAW event but inside the padded event.
    assert mda.has_collision(
        _utc(2026, 7, 8, 12, 44, 5), _utc(2026, 7, 8, 12, 44, 15), events, pad_s=10.0
    )
    # Touching exactly at the (unpadded) boundary is NOT a collision -- half-open
    # interval convention, matching `rowii.eval.events`' own "inclusive start,
    # exclusive end" rule elsewhere in this codebase.
    assert not mda.has_collision(
        _utc(2026, 7, 8, 12, 44, 0), _utc(2026, 7, 8, 12, 44, 10), events, pad_s=0.0
    )


# ---------------------------------------------------------------------------
# 4. dodge_collision
# ---------------------------------------------------------------------------


def test_dodge_collision_shifts_away_or_falls_back_when_there_is_no_room() -> None:
    allowed_start, allowed_end = _utc(2026, 7, 8, 12, 0, 0), _utc(2026, 7, 8, 12, 5, 0)
    events = [(_utc(2026, 7, 8, 12, 2, 0), _utc(2026, 7, 8, 12, 2, 10))]
    w_start, w_end = _utc(2026, 7, 8, 12, 1, 58), _utc(2026, 7, 8, 12, 2, 8)  # collides

    got_start, got_end = mda.dodge_collision(
        w_start, w_end, allowed_start, allowed_end, events, pad_s=0.0
    )

    assert not mda.has_collision(got_start, got_end, events, pad_s=0.0)
    assert allowed_start <= got_start and got_end <= allowed_end
    assert (got_end - got_start) == (w_end - w_start)  # duration preserved

    # Allowed range is EXACTLY the window's own duration -- no slack to shift into
    # (the real-data analogue: cluster 0's 6 s segment vs. a 10 s clip). No collision
    # actually occurs on the real 080726 data for this case, but the fallback must
    # still be well-defined: return the original window unchanged rather than raise.
    tight_start, tight_end = _utc(2026, 7, 8, 14, 19, 42), _utc(2026, 7, 8, 14, 19, 52)
    tight_events = [(_utc(2026, 7, 8, 14, 19, 45), _utc(2026, 7, 8, 14, 19, 46))]

    fallback_start, fallback_end = mda.dodge_collision(
        tight_start, tight_end, tight_start, tight_end, tight_events, pad_s=0.0
    )

    assert (fallback_start, fallback_end) == (tight_start, tight_end)


# ---------------------------------------------------------------------------
# 5. find_burst_file
# ---------------------------------------------------------------------------


def _burst(hour: int, minute: int) -> BurstFile:
    return BurstFile(
        path=Path(f"/fake/RAWGeneratorMic__0_2026-07-08_{hour:02d}-{minute:02d}-00_000000.dat"),
        stream="RAWGeneratorMic__0",
        start_utc_hint=_utc(2026, 7, 8, hour, minute, 0),
    )


def test_find_burst_file_bucket_selection_and_bounds() -> None:
    # Deliberately unsorted input -- the function must sort internally.
    files = [_burst(14, 22), _burst(13, 46), _burst(13, 58), _burst(14, 10)]

    got = mda.find_burst_file(files, _utc(2026, 7, 8, 14, 5, 0))
    assert got.start_utc_hint == _utc(2026, 7, 8, 13, 58, 0)

    # The last bucket has no upper bound -- a target well past the last file's own
    # start still resolves to that last file.
    last_bucket = mda.find_burst_file(files, _utc(2026, 7, 8, 20, 0, 0))
    assert last_bucket.start_utc_hint == _utc(2026, 7, 8, 14, 22, 0)

    with pytest.raises(ValueError, match="no burst file"):
        mda.find_burst_file(files, _utc(2026, 7, 8, 10, 0, 0))  # before the earliest file


# ---------------------------------------------------------------------------
# 6. utc_window_to_sample_range
# ---------------------------------------------------------------------------


def test_utc_window_to_sample_range_exact_and_clamped() -> None:
    t0 = _utc(2026, 7, 8, 13, 46, 3)
    rate_hz = 50_000.0
    n = 100_000  # 2 s of samples
    offsets_ns = (np.arange(n, dtype=np.int64) * round(1e9 / rate_hz)).astype(np.uint64)
    ts_ns = offsets_ns + np.uint64(round(t0.timestamp() * 1e9))

    # A 1 s window starting exactly 0.5 s into the file.
    start_idx, end_idx = mda.utc_window_to_sample_range(
        t0 + timedelta(seconds=0.5), t0 + timedelta(seconds=1.5), ts_ns
    )
    assert start_idx == pytest.approx(25_000, abs=1)
    assert end_idx == pytest.approx(75_000, abs=1)
    assert end_idx - start_idx == pytest.approx(50_000, abs=2)

    # A window reaching outside the file on both sides clamps to [0, n].
    clamped_start, clamped_end = mda.utc_window_to_sample_range(
        t0 - timedelta(seconds=10), t0 + timedelta(seconds=10), ts_ns
    )
    assert (clamped_start, clamped_end) == (0, n)

    with pytest.raises(ValueError):
        mda.utc_window_to_sample_range(t0 + timedelta(seconds=1), t0, ts_ns)


# ---------------------------------------------------------------------------
# 7. peak_normalize
# ---------------------------------------------------------------------------


def test_peak_normalize_hits_target_dbfs_and_handles_silence() -> None:
    samples = np.array([-4.0, 1.0, 3.5, -2.0], dtype=np.float64)

    normalized = mda.peak_normalize(samples, target_dbfs=-1.0)

    expected_peak = 10.0 ** (-1.0 / 20.0)
    assert np.max(np.abs(normalized)) == pytest.approx(expected_peak)
    # Shape/sign preserved: the array is a pure positive rescaling of the input.
    np.testing.assert_allclose(normalized / samples, normalized[0] / samples[0])

    silence = np.zeros(10, dtype=np.float64)
    normalized_silence = mda.peak_normalize(silence, target_dbfs=-1.0)
    np.testing.assert_array_equal(normalized_silence, silence)
    assert np.isfinite(normalized_silence).all()


# ---------------------------------------------------------------------------
# 8. quantile_threshold (feat/demo-replay, Aufgabe A #4: score-histogram line)
# ---------------------------------------------------------------------------


def test_quantile_threshold_is_the_1_minus_alpha_quantile() -> None:
    scores = [float(i) for i in range(1, 101)]  # 1..100

    got = mda.quantile_threshold(scores, alpha=0.01)

    assert got == pytest.approx(np.quantile(scores, 0.99))
    assert mda.quantile_threshold([42.0], alpha=0.5) == pytest.approx(42.0)

    with pytest.raises(ValueError, match="non-empty"):
        mda.quantile_threshold([], alpha=0.01)
    with pytest.raises(ValueError, match="alpha"):
        mda.quantile_threshold(scores, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        mda.quantile_threshold(scores, alpha=1.0)


# ---------------------------------------------------------------------------
# 9. nearest_sorted_index (feat/demo-replay, Aufgabe A #3 window alignment AND the
#    reference implementation for the live-replay JS playhead's own binary search,
#    Aufgabe B #2 -- see this function's docstring)
# ---------------------------------------------------------------------------


def test_nearest_sorted_index_finds_the_closer_neighbor() -> None:
    times = [0.0, 10.0, 10.4, 25.0, 100.0]

    assert mda.nearest_sorted_index(times, 10.4) == 2  # exact match
    assert mda.nearest_sorted_index(times, 10.3) == 2  # closer to 10.4 than to 10.0
    assert mda.nearest_sorted_index(times, 5.0) == 0  # exact tie (5.0/5.0) -> earlier index wins
    assert mda.nearest_sorted_index(times, -5.0) == 0  # before the first entry -> clamp
    assert mda.nearest_sorted_index(times, 500.0) == 4  # after the last entry -> clamp

    with pytest.raises(ValueError, match="non-empty"):
        mda.nearest_sorted_index([], 1.0)


# ---------------------------------------------------------------------------
# 10. column_zscore_stats (feat/demo-replay, Aufgabe A #3 feature-bars day reference)
# ---------------------------------------------------------------------------


def test_column_zscore_stats_computes_mean_std_over_valid_rows_only() -> None:
    features = np.array(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [999.0, 999.0],  # invalid row -- must not pollute the day reference
            [5.0, 30.0],
        ]
    )
    valid_mask = np.array([True, True, False, True])

    mean, std = mda.column_zscore_stats(features, valid_mask)

    np.testing.assert_allclose(mean, [3.0, 20.0])  # mean of the three VALID rows only
    # ddof=0 population std of each column's three VALID values, [1,3,5]/[10,20,30].
    np.testing.assert_allclose(std, [np.std([1.0, 3.0, 5.0]), np.std([10.0, 20.0, 30.0])])

    with pytest.raises(ValueError, match="valid_mask has no True"):
        mda.column_zscore_stats(features, np.zeros(4, dtype=bool))
    with pytest.raises(ValueError, match="rows"):
        mda.column_zscore_stats(features, np.array([True, False]))


# ---------------------------------------------------------------------------
# 11. top_k_abs_z_indices (feat/demo-replay, Aufgabe A #3 feature-bars selection)
# ---------------------------------------------------------------------------


def test_top_k_abs_z_indices_orders_by_largest_absolute_z() -> None:
    row = np.array([10.0, 10.0, 10.0, 10.0])
    mean = np.array([0.0, 0.0, 9.0, 10.0])
    std = np.array([1.0, 5.0, 1.0, 2.0])
    # z = [10.0, 2.0, 1.0, 0.0]

    assert mda.top_k_abs_z_indices(row, mean, std, k=3) == [0, 1, 2]

    # A std == 0 column must never win, even with a huge raw deviation from the mean.
    row2 = np.array([100.0, 5.0])
    mean2 = np.array([0.0, 0.0])
    std2 = np.array([0.0, 1.0])

    assert mda.top_k_abs_z_indices(row2, mean2, std2, k=1) == [1]

    # k larger than the number of columns returns every index, still ordered.
    assert mda.top_k_abs_z_indices(row, mean, std, k=99) == [0, 1, 2, 3]

    with pytest.raises(ValueError, match="k must be positive"):
        mda.top_k_abs_z_indices(row, mean, std, k=0)
    with pytest.raises(ValueError, match="shape"):
        mda.top_k_abs_z_indices(row, mean, std[:2], k=1)


# ---------------------------------------------------------------------------
# 12. shorten_feature_name (feat/demo-replay, Aufgabe A #3 bar labels)
# ---------------------------------------------------------------------------


def test_shorten_feature_name_matches_real_fusion_cache_naming() -> None:
    # Real `feature_names` entries from results/cache/080726-pu_strikes--fusion.npz
    # (module docstring / Aufgabe A #3) -- both stream families that cache actually
    # has (`RAWGeneratorMic__0`/`RAWGeneratorVib__2`/`RAWTurbineMic__1`/
    # `RAWTurbineVib__3`).
    assert mda.shorten_feature_name("RAWGeneratorMic__0::ch0_log_rms") == "GenMic0.ch0.log_rms"
    assert (
        mda.shorten_feature_name("RAWTurbineVib__3::ch5_octave_2000") == "TurVib3.ch5.octave_2000"
    )
    # Anything that doesn't match the expected <Stream>__<n>::ch<i>_<feature> shape
    # is returned UNCHANGED (defensive fallback, never raises).
    assert (
        mda.shorten_feature_name("not-a-recognized-feature-name") == "not-a-recognized-feature-name"
    )


# ---------------------------------------------------------------------------
# 13. extract_window_samples (feat/demo-replay, Aufgabe A #1/#2 waveform+spectrogram)
# ---------------------------------------------------------------------------


def test_extract_window_samples_slices_and_validates_bounds() -> None:
    # 10 s @ 16 kHz, like a real demo clip -- values wrapped into actual int16 range
    # (a real WAV's PCM samples, unlike a bare `np.arange`, never exceed it either).
    pcm = (np.arange(160_000) % 1000).astype(np.int16)

    window = mda.extract_window_samples(pcm, sample_rate_hz=16_000, start_s=4.0, duration_s=1.0)

    assert window.dtype == np.float64
    np.testing.assert_array_equal(window, (np.arange(64_000, 80_000) % 1000).astype(np.float64))

    with pytest.raises(ValueError, match="out of range"):
        mda.extract_window_samples(pcm, sample_rate_hz=16_000, start_s=9.5, duration_s=1.0)
    with pytest.raises(ValueError, match="out of range"):
        mda.extract_window_samples(pcm, sample_rate_hz=16_000, start_s=-1.0, duration_s=1.0)


# ---------------------------------------------------------------------------
# 14. matching_event_kind (feat/demo-dashboard: Alarm-Feed ground-truth tag)
# ---------------------------------------------------------------------------


def test_matching_event_kind_returns_first_overlap_or_none() -> None:
    events = [
        (_utc(2026, 7, 8, 12, 43, 0), _utc(2026, 7, 8, 12, 44, 0), "plate-gen_0"),
        (_utc(2026, 7, 8, 12, 44, 0), _utc(2026, 7, 8, 12, 45, 0), "plate-gen_90"),
    ]

    # Direct overlap with the first event.
    assert (
        mda.matching_event_kind(
            _utc(2026, 7, 8, 12, 43, 10), _utc(2026, 7, 8, 12, 43, 12), events, pad_s=0.0
        )
        == "plate-gen_0"
    )
    # Only reachable via the pad -- lands just before event 2's raw start.
    assert (
        mda.matching_event_kind(
            _utc(2026, 7, 8, 12, 43, 57), _utc(2026, 7, 8, 12, 43, 58), events, pad_s=5.0
        )
        == "plate-gen_0"
    )
    # No event anywhere near, even with generous padding.
    assert (
        mda.matching_event_kind(
            _utc(2026, 7, 8, 14, 0, 0), _utc(2026, 7, 8, 14, 0, 10), events, pad_s=10.0
        )
        is None
    )
    # Touching exactly at the (unpadded) boundary is NOT a collision -- same
    # half-open convention as has_collision.
    assert (
        mda.matching_event_kind(
            _utc(2026, 7, 8, 12, 44, 0), _utc(2026, 7, 8, 12, 44, 10), events, pad_s=0.0
        )
        == "plate-gen_90"  # NOT plate-gen_0: that event ends exactly here
    )


# ---------------------------------------------------------------------------
# 15. state_name_de (feat/demo-dashboard: Zustands-Badge subtitle)
# ---------------------------------------------------------------------------


def test_state_name_de_covers_known_states_and_falls_back() -> None:
    assert mda.state_name_de("turbine") == "Turbinenbetrieb"
    assert mda.state_name_de("pump") == "Pumpbetrieb"
    assert mda.state_name_de("phase-shifter") == "Phasenschieberbetrieb"
    assert mda.state_name_de("standstill") == "Stillstand"
    assert mda.state_name_de("invalid") == "Übergang / ungültig"
    # derive_state_names' own cluster-<id> naming fallback: never invented German
    # prose, just a labelled passthrough.
    assert mda.state_name_de("cluster-2") == "Zustand (cluster-2)"


# ---------------------------------------------------------------------------
# 16. parse_markdown_table / parse_state_table / parse_event_summary_table
#     (feat/demo-dashboard: monitor_notes.md / event_notes.md -> dashboard JSON)
# ---------------------------------------------------------------------------

# Real "## Per-state results" table, byte-for-byte from
# results/step2/once-calibrated/audio-beats/monitor/080726-pu_strikes/recalibrate/
# monitor_notes.md (2026-07-22 run) -- not a hand-crafted fixture, so a real "inf"
# row (state 0, low_confidence) and a real "n/a" row (state 2, never occurs on this
# run) are both exercised as they actually appear on disk.
_REAL_PU_STATE_TABLE_MD = (
    "## Per-state results\n"
    "\n"
    "| state | n_windows | n_scored | n_alarms | alarm_rate | low_confidence | threshold |"
    " n_consumed | status |\n"
    "|---:|---:|---:|---:|---:|:--|---:|---:|:--|\n"
    "| 0 (turbine) | 11 | 6 | 0 | 0.0000 | True | inf | 5 | scored |\n"
    "| 1 (turbine) | 6740 | 4299 | 204 | 0.0475 | False | 0.0348375 | 2441 | scored |\n"
    "| 2 (turbine) | 0 | 0 | 0 | n/a | n/a | n/a | 0 | no_conformal_data |\n"
    "| 3 (standstill) | 6076 | 2315 | 150 | 0.0648 | False | 0.0352036 | 3761 | scored |\n"
    "\n"
    "## Window accounting\n"
)

# Real "## Summary" table, byte-for-byte from
# results/step2/once-calibrated/audio-beats/eval_events/080726-pu_strikes/
# recalibrate/event_notes.md.
_REAL_PU_EVENT_SUMMARY_MD = (
    "## Summary\n"
    "\n"
    "| n_events | n_detected | event_tpr | false_alarm_windows |"
    " false_alarm_rate_per_hour | realized_window_far | tolerance_s |\n"
    "|---:|---:|---:|---:|---:|---:|---:|\n"
    "| 13 | 11 | 0.846154 | 279 | 176.458 | 0.0490162 | 5 |\n"
    "\n"
    "## Per-event results\n"
)


def test_parse_markdown_table_extracts_columns_and_rows_and_stops_at_prose() -> None:
    columns, rows = mda.parse_markdown_table(_REAL_PU_STATE_TABLE_MD, "| state ")

    assert columns == [
        "state", "n_windows", "n_scored", "n_alarms", "alarm_rate",
        "low_confidence", "threshold", "n_consumed", "status",
    ]
    assert len(rows) == 4  # the "## Window accounting" heading below is NOT a row
    assert rows[0] == {
        "state": "0 (turbine)", "n_windows": "11", "n_scored": "6", "n_alarms": "0",
        "alarm_rate": "0.0000", "low_confidence": "True", "threshold": "inf",
        "n_consumed": "5", "status": "scored",
    }

    with pytest.raises(ValueError, match="no markdown table header"):
        mda.parse_markdown_table(_REAL_PU_STATE_TABLE_MD, "| nonexistent ")


def test_parse_state_table_maps_inf_and_na_and_real_thresholds() -> None:
    result = mda.parse_state_table(_REAL_PU_STATE_TABLE_MD)

    assert set(result) == {0, 1, 2, 3}
    assert result[0] == {"name": "turbine", "threshold": math.inf, "low_confidence": True}
    assert result[1]["threshold"] == pytest.approx(0.0348375)
    assert result[1]["low_confidence"] is False
    assert result[2]["threshold"] is None  # "n/a" -- state never occurs on this run
    assert result[3]["name"] == "standstill"
    assert result[3]["threshold"] == pytest.approx(0.0352036)

    with pytest.raises(ValueError, match="unexpected state-table"):
        mda.parse_state_table(
            "| state | threshold | low_confidence |\n"
            "|---|---|---|\n"
            "| not-a-valid-cell | 0.1 | False |\n"
        )


def test_parse_event_summary_table_reads_real_pu_numbers() -> None:
    result = mda.parse_event_summary_table(_REAL_PU_EVENT_SUMMARY_MD)

    assert result == {"n_events": 13, "n_detected": 11, "event_tpr": pytest.approx(0.846154)}

    with pytest.raises(ValueError, match="no markdown table header"):
        mda.parse_event_summary_table("no table here at all")
