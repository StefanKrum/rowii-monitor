"""Tests for handcrafted audio/vibration featurizers (Task 6)."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from rowii.signals.features import (
    MACHINE_HZ,
    AudioFeaturizer,
    VibFeaturizer,
    fuse,
    zscore,
)


def _windows(signal_1d: np.ndarray, n_windows: int = 1) -> np.ndarray:
    """Broadcast a 1-D signal into a (W, S, 1) single-channel windows array."""
    s = signal_1d.astype(np.float32)
    return np.tile(s, (n_windows, 1))[:, :, np.newaxis]


def _sine(freq_hz: float, rate_hz: float, duration_s: float = 1.0) -> np.ndarray:
    t = np.arange(int(rate_hz * duration_s)) / rate_hz
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


# ---------------------------------------------------------------------------
# MACHINE_HZ constant
# ---------------------------------------------------------------------------


def test_machine_hz_constant_has_the_three_documented_frequencies() -> None:
    assert MACHINE_HZ == {"shaft": 6.25, "blade_pass": 43.75, "guide_vane_pass": 125.0}


# ---------------------------------------------------------------------------
# AudioFeaturizer
# ---------------------------------------------------------------------------


def test_audio_featurizer_has_the_documented_name() -> None:
    assert AudioFeaturizer().name == "audio-handcrafted"


def test_audio_featurizer_44hz_sine_at_50khz_blade_pass_band_dominates() -> None:
    """A 44 Hz tone sits inside the blade_pass band (43.75 Hz +/- 10%) and far
    outside the shaft (6.25 Hz +/- 10%) and guide_vane_pass (125 Hz +/- 10%)
    bands, so its band-energy feature must be the largest of the three for
    the channel carrying the tone.
    """
    rate_hz = 50_000.0
    windows = _windows(_sine(44.0, rate_hz))
    feat = AudioFeaturizer()

    out = feat.transform(windows, rate_hz)
    names = feat.feature_names()

    shaft = out[0, names.index("ch0_band_shaft")]
    blade = out[0, names.index("ch0_band_blade_pass")]
    guide = out[0, names.index("ch0_band_guide_vane_pass")]
    assert blade > shaft
    assert blade > guide


def test_audio_featurizer_white_noise_centroid_exceeds_pure_low_sine_centroid() -> None:
    rate_hz = 50_000.0
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(int(rate_hz)).astype(np.float32)
    low_sine = _sine(MACHINE_HZ["shaft"], rate_hz)
    windows = np.stack([noise, low_sine])[:, :, np.newaxis]
    feat = AudioFeaturizer()

    out = feat.transform(windows, rate_hz)
    names = feat.feature_names()
    idx = names.index("ch0_spectral_centroid")

    assert out[0, idx] > out[1, idx]


def test_audio_featurizer_feature_names_length_matches_transform_width() -> None:
    rate_hz = 50_000.0
    windows = np.random.default_rng(0).standard_normal((3, int(rate_hz), 2)).astype(
        np.float32
    )
    feat = AudioFeaturizer()

    out = feat.transform(windows, rate_hz)

    assert len(feat.feature_names()) == out.shape[1]
    assert out.shape[0] == 3


def test_audio_featurizer_output_dtype_is_float64() -> None:
    rate_hz = 50_000.0
    windows = _windows(_sine(100.0, rate_hz))
    feat = AudioFeaturizer()

    out = feat.transform(windows, rate_hz)

    assert out.dtype == np.float64


def test_audio_featurizer_octave_bands_adapt_to_10khz_rate_no_band_above_nyquist() -> None:
    """At 10 kHz, Nyquist is 5000 Hz: octave bands centered at 4000 Hz (upper
    edge 4000*sqrt(2) ~= 5657 Hz) and 8000 Hz must be dropped, shrinking the
    feature set relative to the 50 kHz case where all 9 octave bands fit.
    """
    rate_50k = 50_000.0
    rate_10k = 10_000.0
    windows_50k = _windows(_sine(100.0, rate_50k))
    windows_10k = _windows(_sine(100.0, rate_10k))

    feat_50k = AudioFeaturizer()
    feat_50k.transform(windows_50k, rate_50k)
    names_50k = feat_50k.feature_names()

    feat_10k = AudioFeaturizer()
    feat_10k.transform(windows_10k, rate_10k)
    names_10k = feat_10k.feature_names()

    assert "ch0_octave_4000" in names_50k
    assert "ch0_octave_8000" in names_50k
    assert "ch0_octave_4000" not in names_10k
    assert "ch0_octave_8000" not in names_10k
    assert len(names_10k) < len(names_50k)


def test_audio_featurizer_does_not_drop_dead_channels() -> None:
    """Audio mics are validated elsewhere; a constant (dead) channel must
    still be represented in the audio feature set, unlike VibFeaturizer.
    """
    rate_hz = 50_000.0
    live = _sine(100.0, rate_hz)
    dead = np.zeros(int(rate_hz), dtype=np.float32)
    windows = np.stack([live, dead], axis=0)[np.newaxis, :, :].transpose(0, 2, 1)
    # windows shape (1, S, 2): channel 0 = live, channel 1 = dead
    feat = AudioFeaturizer()

    out = feat.transform(windows, rate_hz)
    names = feat.feature_names()

    assert any(n.startswith("ch1_") for n in names)
    assert out.shape[1] == len(names)


# ---------------------------------------------------------------------------
# VibFeaturizer
# ---------------------------------------------------------------------------


def test_vib_featurizer_has_the_documented_name() -> None:
    assert VibFeaturizer().name == "vib-handcrafted"


def test_vib_featurizer_dead_channel_dropped_with_warning_and_feature_count_shrinks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rate_hz = 10_000.0
    live = _sine(100.0, rate_hz)
    dead = np.zeros(int(rate_hz), dtype=np.float32)
    # windows shape (1, S, 2): channel 0 = live, channel 1 = dead (std < 1e-9)
    windows = np.stack([live, dead], axis=1)[np.newaxis, :, :]
    assert windows.shape == (1, int(rate_hz), 2)

    feat_both_live = VibFeaturizer()
    both_live_windows = np.stack([live, live], axis=1)[np.newaxis, :, :]
    out_both_live = feat_both_live.transform(both_live_windows, rate_hz)

    feat = VibFeaturizer()
    with caplog.at_level(logging.WARNING):
        out = feat.transform(windows, rate_hz)

    assert out.shape[1] < out_both_live.shape[1]
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "channel 1" in msg and "dead" in msg.lower() for msg in warnings
    ), warnings
    names = feat.feature_names()
    assert len(names) == out.shape[1]
    assert not any(n.startswith("ch1_") for n in names)
    assert any(n.startswith("ch0_") for n in names)


def test_vib_featurizer_feature_names_before_transform_raises_or_is_empty_until_called() -> (
    None
):
    """feature_names() reflects live channels discovered during transform, so
    calling it before any transform() call is only meaningful after at least
    one transform -- calling it beforehand must not silently return a
    stale/garbage list claiming channels exist that were never checked.
    """
    feat = VibFeaturizer()
    with pytest.raises(RuntimeError, match="transform"):
        feat.feature_names()


def test_vib_featurizer_all_live_channels_feature_names_length_matches_transform_width() -> (
    None
):
    rate_hz = 10_000.0
    rng = np.random.default_rng(1)
    windows = rng.standard_normal((4, int(rate_hz), 3)).astype(np.float32)
    feat = VibFeaturizer()

    out = feat.transform(windows, rate_hz)

    assert len(feat.feature_names()) == out.shape[1]
    assert out.shape[0] == 4


def test_vib_featurizer_kurtosis_feature_present_and_matches_scipy() -> None:
    from scipy.stats import kurtosis as scipy_kurtosis

    rate_hz = 10_000.0
    rng = np.random.default_rng(2)
    raw = rng.standard_normal(int(rate_hz)).astype(np.float32)
    windows = raw[np.newaxis, :, np.newaxis]
    feat = VibFeaturizer()

    out = feat.transform(windows, rate_hz)
    names = feat.feature_names()
    idx = names.index("ch0_kurtosis")

    # Compare against scipy applied to the SAME float64 representation the
    # featurizer computes on (all features are computed in float64 for
    # numerical stability); comparing against the raw float32 array would
    # spuriously fail on ~1e-7-level float32 rounding, not a real behavior
    # difference.
    assert out[0, idx] == pytest.approx(
        scipy_kurtosis(raw.astype(np.float64)), rel=1e-9
    )


def test_vib_featurizer_octave_bands_truncated_to_4khz_and_below() -> None:
    rate_hz = 50_000.0  # rate high enough that Nyquist doesn't itself truncate
    rng = np.random.default_rng(3)
    windows = rng.standard_normal((1, int(rate_hz), 1)).astype(np.float32)
    feat = VibFeaturizer()

    feat.transform(windows, rate_hz)
    names = feat.feature_names()

    assert "ch0_octave_4000" in names
    assert "ch0_octave_8000" not in names


def test_vib_featurizer_output_dtype_is_float64() -> None:
    rate_hz = 10_000.0
    windows = np.random.default_rng(4).standard_normal((2, int(rate_hz), 1)).astype(
        np.float32
    )
    feat = VibFeaturizer()

    out = feat.transform(windows, rate_hz)

    assert out.dtype == np.float64


def test_vib_featurizer_all_channels_dead_yields_zero_feature_columns() -> None:
    rate_hz = 10_000.0
    windows = np.zeros((2, int(rate_hz), 2), dtype=np.float32)
    feat = VibFeaturizer()

    with pytest.raises(ValueError, match="all.*dead|no live channel"):
        feat.transform(windows, rate_hz)


# ---------------------------------------------------------------------------
# zscore
# ---------------------------------------------------------------------------


def test_zscore_returns_zero_mean_unit_std_columns() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(loc=10.0, scale=3.0, size=(200, 4))

    z = zscore(x)

    np.testing.assert_allclose(z.mean(axis=0), np.zeros(4), atol=1e-9)
    np.testing.assert_allclose(z.std(axis=0), np.ones(4), atol=1e-9)


def test_zscore_constant_column_survives_as_zeros_not_nan_or_inf() -> None:
    x = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])

    z = zscore(x)

    assert np.all(np.isfinite(z))
    np.testing.assert_array_equal(z[:, 1], np.zeros(3))


def test_zscore_output_dtype_is_float64() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    z = zscore(x)

    assert z.dtype == np.float64


# ---------------------------------------------------------------------------
# fuse
# ---------------------------------------------------------------------------


def test_fuse_shape_is_sum_of_both_feature_dims() -> None:
    rng = np.random.default_rng(6)
    a = rng.normal(size=(10, 3))
    b = rng.normal(size=(10, 5))

    fused = fuse(a, b)

    assert fused.shape == (10, 8)


def test_fuse_is_zscored_concatenation_of_both_inputs() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(loc=100.0, scale=20.0, size=(50, 2))
    b = rng.normal(loc=-5.0, scale=0.1, size=(50, 3))

    fused = fuse(a, b)

    np.testing.assert_allclose(fused[:, :2], zscore(a))
    np.testing.assert_allclose(fused[:, 2:], zscore(b))
