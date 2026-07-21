"""Tests for scripts/verify_data_facts.py (Package-8 D4, A1.6): PURE-logic unit tests
on the variance criterion, the changeover locator (synthetic series), and the
channel-anonymous mic-profile math -- IO seams monkeypatched, no real data."""
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


def test_locate_changeover_finds_the_step_index() -> None:
    ts = np.arange(100, dtype=np.uint64) * np.uint64(1_000_000_000)
    values = np.concatenate([np.full(60, 0.0), np.full(40, 378.8)])  # step at index 60
    assert vdf.locate_changeover(ts, values) == 60
    # a noisy plateau then a jump: the JUMP wins over the noise.
    rng = np.random.default_rng(1)
    noisy = np.concatenate([rng.normal(0, 0.01, 60), rng.normal(100, 0.01, 40)])
    assert vdf.locate_changeover(ts, noisy) == 60


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
    profile = vdf.channel_level_profile(levels, strike)
    assert profile.shape == (4,)
    assert vdf.outlier_channel(profile) == 2
    # NO azimuth asserted: the function returns an INDEX only.
    assert isinstance(vdf.outlier_channel(profile), int)


def test_locate_changeover_requires_two_points() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        vdf.locate_changeover(np.array([0], dtype=np.uint64), np.array([1.0]))
