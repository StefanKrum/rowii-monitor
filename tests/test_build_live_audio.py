"""Tests for `scripts/build_live_audio.py`'s PURE helpers: session-bounds/block-window
planning, probe-instant spacing, and -- the two functions `assets/live.js` mirrors by
hand client-side, since this repo has no JS test runner -- the replay-playhead <->
audio-offset UTC mapping (`audio_offset_s`) and the transport-speed <-> playback-rate/
mute mapping (`audio_playback_for_speed`).

Import convention mirrors `tests/test_annotation_kit.py`: `scripts/` is not a package,
so the module under test is imported directly by inserting `scripts/` onto `sys.path`.
Everything that opens a burst file, writes a WAV, or shells out to `afconvert` is
exercised by actually running the CLI against real data instead (mirrors
`annotation_kit.py`'s own "pure vs IO-touching" split, see that module's docstring).
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_live_audio as lva  # noqa: E402


def _utc(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> datetime:
    """`datetime(..., tzinfo=UTC)` shorthand -- mirrors `test_annotation_kit._utc`."""
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


# ---------------------------------------------------------------------------
# 1. session_bounds
# ---------------------------------------------------------------------------


def test_session_bounds_applies_start_and_outer_margins() -> None:
    hints = [_utc(2026, 6, 29, 2, 42, 0), _utc(2026, 6, 29, 0, 30, 57), _utc(2026, 6, 29, 6, 42, 0)]

    start, end = lva.session_bounds(hints, start_margin_s=60.0, outer_margin_s=900.0)

    assert start == _utc(2026, 6, 29, 0, 30, 57) - timedelta(seconds=60)
    assert end == _utc(2026, 6, 29, 6, 42, 0) + timedelta(seconds=900)


def test_session_bounds_rejects_empty_hints() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        lva.session_bounds([])


# ---------------------------------------------------------------------------
# 2. plan_extraction_blocks
# ---------------------------------------------------------------------------


def test_plan_extraction_blocks_steps_by_block_s_across_the_full_span() -> None:
    hints = [_utc(2026, 6, 29, 0, 0, 0)]

    windows = lva.plan_extraction_blocks(
        hints, block_s=300.0, start_margin_s=0.0, outer_margin_s=900.0
    )

    # session_bounds span here is exactly 900s -- 3 whole 300s blocks, no partial tail.
    assert len(windows) == 3
    assert windows[0] == (_utc(2026, 6, 29, 0, 0, 0), _utc(2026, 6, 29, 0, 5, 0))
    assert windows[1] == (_utc(2026, 6, 29, 0, 5, 0), _utc(2026, 6, 29, 0, 10, 0))
    assert windows[2] == (_utc(2026, 6, 29, 0, 10, 0), _utc(2026, 6, 29, 0, 15, 0))


def test_plan_extraction_blocks_windows_are_contiguous_no_gap_no_overlap() -> None:
    hints = [_utc(2026, 6, 29, 0, 30, 57), _utc(2026, 6, 29, 4, 42, 0)]

    windows = lva.plan_extraction_blocks(hints)

    for (_, end_a), (start_b, _) in zip(windows, windows[1:], strict=False):
        assert end_a == start_b


def test_plan_extraction_blocks_matches_the_real_290626_tu_grid_size() -> None:
    # Regression pin against the real build (build_live_audio's own log: 54 planned
    # windows, the last logged non-empty one being "block 53/54"). start_utc_hint
    # values are already UTC (rowii.io.dataset's own local -> UTC filename parse):
    # the first/last RAWGeneratorMic__0 burst files' filenames read local
    # 02-30-57_737000 / 06-42-00_000000 (Europe/Vienna, CEST = UTC+2 in June).
    hints = [
        _utc(2026, 6, 29, 0, 30, 57) + timedelta(microseconds=737_000),
        _utc(2026, 6, 29, 4, 42, 0),
    ]

    windows = lva.plan_extraction_blocks(hints)

    assert len(windows) == 54


def test_plan_extraction_blocks_rejects_empty_hints() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        lva.plan_extraction_blocks([])


# ---------------------------------------------------------------------------
# 3. probe_instants
# ---------------------------------------------------------------------------


def test_probe_instants_includes_both_endpoints() -> None:
    start, end = _utc(2026, 6, 29, 0, 0, 0), _utc(2026, 6, 29, 4, 0, 0)

    instants = lva.probe_instants(start, end, n=5)

    assert instants[0] == start
    assert instants[-1] == end
    assert len(instants) == 5


def test_probe_instants_are_evenly_spaced() -> None:
    start, end = _utc(2026, 6, 29, 0, 0, 0), _utc(2026, 6, 29, 0, 40, 0)  # 2400s span

    instants = lva.probe_instants(start, end, n=5)  # step = 600s = 10 min

    gaps = [(b - a).total_seconds() for a, b in zip(instants, instants[1:], strict=False)]
    assert gaps == pytest.approx([600.0, 600.0, 600.0, 600.0])


def test_probe_instants_rejects_fewer_than_two_probes() -> None:
    with pytest.raises(ValueError, match="n must be >= 2"):
        lva.probe_instants(_utc(2026, 6, 29, 0, 0, 0), _utc(2026, 6, 29, 1, 0, 0), n=1)


def test_probe_instants_rejects_inverted_span() -> None:
    with pytest.raises(ValueError, match="after"):
        lva.probe_instants(_utc(2026, 6, 29, 1, 0, 0), _utc(2026, 6, 29, 0, 0, 0))


# ---------------------------------------------------------------------------
# 4. audio_offset_s -- the replay-playhead <-> audio-offset UTC mapping
#    (assets/live.js's audioOffsetS() is the client-side mirror of this).
# ---------------------------------------------------------------------------


def test_audio_offset_s_zero_when_audio_starts_exactly_at_t0() -> None:
    t0 = "2026-06-29T00:30:57.834000+00:00"

    assert lva.audio_offset_s(0.0, t0, t0) == pytest.approx(0.0)
    assert lva.audio_offset_s(500.0, t0, t0) == pytest.approx(500.0)


def test_audio_offset_s_matches_the_real_290626_tu_gen_stream() -> None:
    # Pinned against the real build_live_audio.py output for RUN=290626-tu.
    t0_utc = "2026-06-29T00:30:57.834000120+00:00"
    audio_start_utc = "2026-06-29T00:30:57.737000+00:00"  # gen stream

    assert lva.audio_offset_s(0.0, t0_utc, audio_start_utc) == pytest.approx(0.097000120, abs=1e-6)
    assert lva.audio_offset_s(15585.0, t0_utc, audio_start_utc) == pytest.approx(
        15585.097000120, abs=1e-6
    )


def test_audio_offset_s_negative_when_audio_starts_after_t0() -> None:
    # The requested instant is before this stream's own extracted audio begins --
    # a valid, negative result (live.js's own job to decide what to do with it, not
    # this function's -- see its docstring).
    t0 = "2026-06-29T00:30:00+00:00"
    audio_start = "2026-06-29T00:30:10+00:00"  # audio starts 10s AFTER t0

    assert lva.audio_offset_s(0.0, t0, audio_start) == pytest.approx(-10.0)
    assert lva.audio_offset_s(10.0, t0, audio_start) == pytest.approx(0.0)


def test_audio_offset_s_is_linear_in_playhead_s() -> None:
    t0 = "2026-06-29T00:30:57.834000+00:00"
    audio_start = "2026-06-29T00:29:00.000000+00:00"

    a = lva.audio_offset_s(100.0, t0, audio_start)
    b = lva.audio_offset_s(200.0, t0, audio_start)

    assert b - a == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 5. audio_playback_for_speed -- the transport-speed <-> playbackRate/mute mapping
#    (assets/live.js's listenPlaybackFor() is the client-side mirror of this).
# ---------------------------------------------------------------------------


def test_audio_playback_for_speed_1x_passes_through_unmuted() -> None:
    assert lva.audio_playback_for_speed(1.0) == (1.0, False)


def test_audio_playback_for_speed_4x_passes_through_unmuted() -> None:
    assert lva.audio_playback_for_speed(4.0) == (4.0, False)


def test_audio_playback_for_speed_16x_is_muted_with_rate_pinned_to_1() -> None:
    assert lva.audio_playback_for_speed(16.0) == (1.0, True)


def test_audio_playback_for_speed_mute_boundary_is_inclusive_at_16() -> None:
    below = lva.audio_playback_for_speed(15.999)
    at = lva.audio_playback_for_speed(16.0)

    assert below == (15.999, False)
    assert at == (1.0, True)


def test_audio_playback_for_speed_above_16x_stays_muted() -> None:
    assert lva.audio_playback_for_speed(32.0) == (1.0, True)


def test_audio_playback_for_speed_rejects_non_positive_speed() -> None:
    with pytest.raises(ValueError, match="positive"):
        lva.audio_playback_for_speed(0.0)
    with pytest.raises(ValueError, match="positive"):
        lva.audio_playback_for_speed(-1.0)
