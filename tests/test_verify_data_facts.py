"""Tests for scripts/verify_data_facts.py (Package-8 D4, A1.6): PURE-logic unit tests
on the variance criterion, the changeover locator (synthetic series, incl. its
reference-windowed search and top-k alternatives, T1 review finding 1/2), the
chronological day-root sort (T1 review finding 3), and the channel-anonymous
mic-profile math (T1 review finding 4) -- IO seams monkeypatched, no real data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import verify_data_facts as vdf  # noqa: E402


def test_block_is_dead_uses_float64_1e9_criterion() -> None:
    dead = np.full((500, 1), -7.0, dtype=np.float32)  # constant channel, float32 input
    assert vdf.block_is_dead(dead[:, 0]) is True
    live = np.random.default_rng(0).normal(size=500).astype(np.float32)
    assert vdf.block_is_dead(live) is False
    # A channel constant to float32 precision but re-cast: std must be exactly 0 in f64.
    near = np.full(500, 3.5, dtype=np.float32)
    assert vdf.block_is_dead(near) is True


# ---------------------------------------------------------------------------
# locate_changeover
# ---------------------------------------------------------------------------


def test_locate_changeover_finds_the_step_index() -> None:
    ts = np.arange(100, dtype=np.uint64) * np.uint64(1_000_000_000)
    values = np.concatenate([np.full(60, 0.0), np.full(40, 378.8)])  # step at index 60
    assert vdf.locate_changeover(ts, values) == 60
    # a noisy plateau then a jump: the JUMP wins over the noise.
    rng = np.random.default_rng(1)
    noisy = np.concatenate([rng.normal(0, 0.01, 60), rng.normal(100, 0.01, 40)])
    assert vdf.locate_changeover(ts, noisy) == 60


def test_locate_changeover_requires_two_points() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        vdf.locate_changeover(np.array([0], dtype=np.uint64), np.array([1.0]))


def test_locate_changeover_reference_window_avoids_wrong_global_step() -> None:
    # 080726-shaped scenario (T1 review finding 1): a POWER-only changeover
    # (pump -> phase-shifter) sits at index 60, but an earlier, slightly LARGER
    # same-channel step (pump start) at index 20 wins the unrestricted global
    # argmax -- the WRONG changeover. A reference-windowed search centred on the
    # true changeover (with a radius generous enough to include it but not index
    # 20) recovers the right one.
    ts = np.arange(120, dtype=np.uint64) * np.uint64(60_000_000_000)  # 1-min windows
    power = np.zeros(120, dtype=np.float64)
    power[20:] = 21.0  # pump start: jump magnitude 21 -- the day's global max
    power[60:] += 20.0  # true changeover: jump magnitude 20, strictly smaller

    assert vdf.locate_changeover(ts, power) == 20  # global argmax: the WRONG step
    assert (
        vdf.locate_changeover(ts, power, reference_index=60, search_radius=30) == 60
    )


def test_locate_changeover_reference_none_preserves_whole_series_behavior() -> None:
    ts = np.arange(100, dtype=np.uint64) * np.uint64(1_000_000_000)
    values = np.concatenate([np.full(60, 0.0), np.full(40, 378.8)])  # existing fixture
    assert (
        vdf.locate_changeover(ts, values)
        == vdf.locate_changeover(ts, values, reference_index=None, search_radius=None)
        == 60
    )


def test_locate_changeover_validates_reference_and_radius_values() -> None:
    ts = np.arange(10, dtype=np.uint64)
    values = np.arange(10, dtype=np.float64)
    with pytest.raises(ValueError, match="together"):
        vdf.locate_changeover(ts, values, reference_index=5)
    with pytest.raises(ValueError, match="together"):
        vdf.locate_changeover(ts, values, search_radius=2)
    with pytest.raises(ValueError, match="out of range"):
        vdf.locate_changeover(ts, values, reference_index=99, search_radius=1)
    with pytest.raises(ValueError, match="search_radius must be"):
        vdf.locate_changeover(ts, values, reference_index=5, search_radius=-1)


def test_locate_changeover_raises_when_reference_window_has_no_candidate() -> None:
    ts = np.arange(5, dtype=np.uint64)
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="no candidate index"):
        vdf.locate_changeover(ts, values, reference_index=0, search_radius=0)


def test_locate_changeover_raises_when_reference_window_is_all_non_finite() -> None:
    ts = np.arange(6, dtype=np.uint64)
    # finite jump only at value-index 5 (outside the window below); value-indices
    # 1-3 (inside the window) all touch a NaN neighbour.
    values = np.array([0.0, np.nan, np.nan, np.nan, 5.0, 5.0])
    with pytest.raises(ValueError, match="no finite consecutive-window difference"):
        vdf.locate_changeover(ts, values, reference_index=1, search_radius=1)


def test_locate_changeover_raises_when_every_difference_is_non_finite() -> None:
    ts = np.arange(5, dtype=np.uint64)
    with pytest.raises(ValueError, match="no finite consecutive-window difference"):
        vdf.locate_changeover(ts, np.full(5, np.nan))
    # alternating NaN: every first-difference still touches a NaN neighbour.
    with pytest.raises(ValueError, match="no finite consecutive-window difference"):
        vdf.locate_changeover(ts, np.array([1.0, np.nan, 2.0, np.nan, 3.0]))


# ---------------------------------------------------------------------------
# top_k_steps
# ---------------------------------------------------------------------------


def test_top_k_steps_ranks_by_magnitude_and_drops_non_finite() -> None:
    values = np.array([0.0, 1.0, np.nan, 1.0, 6.0, 6.0, 6.5, 15.0])
    # value-index: 1 -> |1-0|=1, 2 -> nan, 3 -> nan, 4 -> |6-1|=5, 5 -> 0,
    #              6 -> |6.5-6|=0.5, 7 -> |15-6.5|=8.5
    assert vdf.top_k_steps(values, 3) == [(7, 8.5), (4, 5.0), (1, 1.0)]


def test_top_k_steps_returns_fewer_than_k_when_not_enough_finite_differences() -> None:
    assert vdf.top_k_steps(np.array([np.nan, np.nan]), 3) == []
    assert vdf.top_k_steps(np.array([0.0, 3.0]), 3) == [(1, 3.0)]


# ---------------------------------------------------------------------------
# channel_level_profile / outlier_channel
# ---------------------------------------------------------------------------


def test_channel_level_profile_and_outlier_are_channel_anonymous() -> None:
    # 4 mic channels; channel 2 sits ~6 dB above its ring at strike minutes.
    levels = np.zeros((200, 4), dtype=np.float64)
    levels[:, :] = -40.0
    strike = np.zeros(200, dtype=bool)
    strike[50:70] = True
    levels[strike, 0] = -40.0
    levels[strike, 1] = -40.2
    levels[strike, 2] = -34.0  # the outlier
    levels[strike, 3] = -39.8
    # a zero-sample strike window (all channels NaN, `_run_window_grid_and_levels`'s
    # own fill value for an empty window) must not poison nanmedian for the OTHER
    # strike rows' channels (T1 review finding 4).
    levels[55, :] = np.nan
    profile = vdf.channel_level_profile(levels, strike)
    assert profile.shape == (4,)
    assert np.all(np.isfinite(profile))
    assert vdf.outlier_channel(profile) == 2
    # NO azimuth asserted: the function returns an INDEX only.
    assert isinstance(vdf.outlier_channel(profile), int)


def test_channel_level_profile_raises_when_a_channel_is_all_nan_at_strikes() -> None:
    levels = np.zeros((10, 2), dtype=np.float64)
    levels[:, 1] = np.nan  # channel 1 has no finite value anywhere
    strike = np.zeros(10, dtype=bool)
    strike[2:5] = True
    with pytest.raises(ValueError, match="no finite value"):
        vdf.channel_level_profile(levels, strike)


# ---------------------------------------------------------------------------
# sorted_day_roots
# ---------------------------------------------------------------------------


def test_sorted_day_roots_orders_chronologically_not_alphabetically() -> None:
    names = ["illwerke-010726", "illwerke-250526", "illwerke-290626"]
    assert vdf.sorted_day_roots(names) == [
        "illwerke-250526",  # 2026-05-25
        "illwerke-290626",  # 2026-06-29
        "illwerke-010726",  # 2026-07-01
    ]
