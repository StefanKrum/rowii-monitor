"""Tests for rowii.anomaly.mad_baseline: the hand-set-threshold MAD baseline's
pure math -- median/MAD commissioning stats, the k <-> threshold <-> flag-rate
triangle (incl. the exact order-statistic k_1pct search), and the log-mel ->
high-band-score derivation (mel-bin selection + the log10-encode/decode
round-trip). Deterministic, synthetic inputs only -- no real data, no partner
number anywhere (module docstring's attribution firewall).
"""
from __future__ import annotations

import numpy as np
import pytest

from rowii.anomaly.mad_baseline import (
    band_energy_score,
    flag_rate,
    high_band_mel_bins,
    k_for_target_rate,
    logmel_geometry,
    median_mad,
    threshold_from_k,
)

_LOGMEL_CACHE_FLOOR = 1e-10
"""Mirrors `rowii.signals.logmel._LOG_FLOOR` -- used ONLY to build synthetic
fixtures that round-trip through the SAME `log10(x + floor)` encoding the real
`logmel` cache stores, so `test_band_energy_score_recovers_known_linear_energy`
exercises the real undo-the-log arithmetic, not a simplified stand-in."""


# ---------------------------------------------------------------------------
# median_mad
# ---------------------------------------------------------------------------


def test_median_mad_matches_manual_computation() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # one big outlier -> MAD robust to it
    median, mad = median_mad(x)
    raw_mad = float(np.median(np.abs(x - np.median(x))))
    assert median == pytest.approx(3.0)
    assert mad == pytest.approx(1.4826 * raw_mad, rel=1e-9)


def test_median_mad_floors_degenerate_zero_mad() -> None:
    x = np.full(10, -40.0)  # every value identical -> raw MAD is exactly 0
    median, mad = median_mad(x)
    assert median == pytest.approx(-40.0)
    assert mad == pytest.approx(1e-8)  # the divide-by-zero floor, never 0.0


def test_median_mad_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        median_mad(np.array([]))


# ---------------------------------------------------------------------------
# threshold_from_k / flag_rate
# ---------------------------------------------------------------------------


def test_threshold_from_k_is_median_plus_k_times_mad() -> None:
    assert threshold_from_k(median=-40.0, mad=2.0, k=5.0) == pytest.approx(-30.0)
    assert threshold_from_k(median=-40.0, mad=2.0, k=0.0) == pytest.approx(-40.0)


def test_flag_rate_counts_strictly_greater() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert flag_rate(x, threshold=3.0) == pytest.approx(2 / 5)  # 4, 5 exceed
    assert flag_rate(x, threshold=5.0) == pytest.approx(0.0)  # threshold itself never flags
    assert flag_rate(x, threshold=0.0) == pytest.approx(1.0)


def test_flag_rate_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        flag_rate(np.array([]), threshold=0.0)


# ---------------------------------------------------------------------------
# k_for_target_rate
# ---------------------------------------------------------------------------


def test_k_for_target_rate_hits_exact_count_for_distinct_scores() -> None:
    x = np.arange(100, dtype=np.float64)  # 100 distinct values, no ties
    median, mad = median_mad(x)
    k, realized = k_for_target_rate(x, median, mad, target_rate=0.01)
    threshold = threshold_from_k(median, mad, k)
    assert flag_rate(x, threshold) == pytest.approx(0.01)  # exactly 1/100
    assert realized == pytest.approx(0.01)


def test_k_for_target_rate_zero_target_flags_nothing() -> None:
    x = np.arange(50, dtype=np.float64)
    median, mad = median_mad(x)
    k, realized = k_for_target_rate(x, median, mad, target_rate=0.0)
    assert realized == pytest.approx(0.0)
    assert flag_rate(x, threshold_from_k(median, mad, k)) == pytest.approx(0.0)


def test_k_for_target_rate_full_target_flags_everything() -> None:
    x = np.arange(50, dtype=np.float64)
    median, mad = median_mad(x)
    k, realized = k_for_target_rate(x, median, mad, target_rate=1.0)
    assert realized == pytest.approx(1.0)
    assert flag_rate(x, threshold_from_k(median, mad, k)) == pytest.approx(1.0)


def test_k_for_target_rate_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        k_for_target_rate(np.array([]), median=0.0, mad=1.0, target_rate=0.01)


def test_k_for_target_rate_raises_outside_unit_interval() -> None:
    x = np.arange(10, dtype=np.float64)
    with pytest.raises(ValueError, match="target_rate"):
        k_for_target_rate(x, median=0.0, mad=1.0, target_rate=1.5)


# ---------------------------------------------------------------------------
# logmel_geometry
# ---------------------------------------------------------------------------


def _names(n_frames: int, n_mels: int) -> list[str]:
    return [
        f"RAWGeneratorMic__0::logmel_f{f}_m{m}"
        for f in range(n_frames)
        for m in range(n_mels)
    ]


def test_logmel_geometry_parses_frame_major_names() -> None:
    names = _names(n_frames=3, n_mels=5)
    assert logmel_geometry(names) == (3, 5)


def test_logmel_geometry_raises_on_wrong_prefix() -> None:
    names = ["RAWTurbineMic__1::logmel_f0_m0"]
    with pytest.raises(ValueError, match="RAWGeneratorMic__0"):
        logmel_geometry(names)


def test_logmel_geometry_raises_on_malformed_local_name() -> None:
    with pytest.raises(ValueError, match="logmel_f"):
        logmel_geometry(["RAWGeneratorMic__0::ch0_log_rms"])


def test_logmel_geometry_raises_on_incomplete_grid() -> None:
    names = _names(n_frames=2, n_mels=2)[:-1]  # drop the last (f=1, m=1) column
    with pytest.raises(ValueError, match="complete"):
        logmel_geometry(names)


def test_logmel_geometry_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        logmel_geometry([])


# ---------------------------------------------------------------------------
# high_band_mel_bins
# ---------------------------------------------------------------------------


def test_high_band_mel_bins_matches_known_plant_geometry() -> None:
    """Pinned against a hand-derived reference (n_mels=64, fmin=20 Hz, the
    `LogmelFeaturizer` default geometry, at the plant's real 50 kHz mic rate):
    24 bins, centers ~5.12-19.92 kHz -- the closest achievable match to a
    literal 5-20 kHz band from this mel filterbank (module docstring)."""
    bins = high_band_mel_bins(n_mels=64, rate_hz=50_000.0)
    expected = np.arange(37, 61)  # indices 37..60 inclusive, 24 bins
    np.testing.assert_array_equal(bins, expected)


def test_high_band_mel_bins_raises_when_band_empty() -> None:
    with pytest.raises(ValueError, match="no mel bin"):
        high_band_mel_bins(n_mels=8, rate_hz=1000.0)  # Nyquist 500 Hz < 5 kHz


# ---------------------------------------------------------------------------
# band_energy_score
# ---------------------------------------------------------------------------


def test_band_energy_score_recovers_known_linear_energy() -> None:
    n_frames, n_mels = 2, 4
    target_bins = np.array([1, 3])
    linear = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],  # frame 0: bins 1,3 -> 2.0 + 4.0 = 6.0
            [1.0, 5.0, 3.0, 7.0],  # frame 1: bins 1,3 -> 5.0 + 7.0 = 12.0
        ]
    )  # mean band power over frames = (6.0 + 12.0) / 2 = 9.0
    cached = np.log10(linear + _LOGMEL_CACHE_FLOOR)  # LogmelFeaturizer's own encoding
    features = cached.reshape(1, n_frames * n_mels)  # frame-major flatten, one window

    score = band_energy_score(features, n_frames, n_mels, target_bins)

    assert score.shape == (1,)
    assert score[0] == pytest.approx(np.log10(9.0 + 1e-12), rel=1e-9)


def test_band_energy_score_handles_multiple_windows() -> None:
    n_frames, n_mels = 1, 2
    target_bins = np.array([0, 1])
    linear = np.array([[1.0, 1.0], [10.0, 10.0]])  # two windows, one frame each
    cached = np.log10(linear + _LOGMEL_CACHE_FLOOR).reshape(2, n_frames * n_mels)

    score = band_energy_score(cached, n_frames, n_mels, target_bins)

    assert score.shape == (2,)
    assert score[1] > score[0]  # window 1 has 10x the band energy of window 0
    assert score[0] == pytest.approx(np.log10(2.0 + 1e-12), rel=1e-9)


def test_band_energy_score_raises_on_geometry_mismatch() -> None:
    features = np.zeros((3, 7))  # 7 != n_frames * n_mels
    with pytest.raises(ValueError, match="n_frames \\* n_mels"):
        band_energy_score(features, n_frames=2, n_mels=4, target_bins=np.array([0]))
