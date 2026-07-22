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
