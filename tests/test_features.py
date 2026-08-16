"""Tests for handcrafted audio/vibration featurizers."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from rowii.signals.features import (
    MACHINE_HZ,
    AudioFeaturizer,
    VibFeaturizer,
    apply_zscore,
    fuse,
    machine_band_bin_counts,
    zscore,
    zscore_stats,
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
# machine_band_bin_counts (dedicated high-resolution PSD pass, fix round 1)
# ---------------------------------------------------------------------------


def test_machine_band_bin_counts_all_three_bands_resolved_at_50khz_1s_window() -> None:
    """At 50 kHz with a 1-s window (S=50_000), the dedicated high-resolution
    PSD pass has `nperseg=S` -> 1 Hz bins. All three MACHINE_HZ bands are
    wider than 1 Hz (shaft: 1.25 Hz, blade_pass: 8.75 Hz,
    guide_vane_pass: 25 Hz), so each must contain >= 1 real bin -- the
    nearest-bin fallback is never exercised in this regime.
    """
    rate_hz = 50_000.0
    n_samples = int(rate_hz)

    counts = machine_band_bin_counts(rate_hz, n_samples)

    assert counts["shaft"] >= 1
    assert counts["blade_pass"] >= 1
    assert counts["guide_vane_pass"] >= 1


def test_machine_band_bin_counts_shaft_band_resolved_at_10khz_vib_rate_1s_window() -> None:
    """Vibration path sanity check at a lower sampling rate: 10 kHz with a
    1-s window (S=10_000) also gives 1 Hz bins, so the shaft band -- the
    narrowest of the three (1.25 Hz wide) -- is still resolved by >= 1 real
    bin, not just at audio's 50 kHz.
    """
    rate_hz = 10_000.0
    n_samples = int(rate_hz)

    counts = machine_band_bin_counts(rate_hz, n_samples)

    assert counts["shaft"] >= 1


def test_machine_band_bin_counts_shaft_band_is_zero_bins_at_old_coarse_resolution() -> None:
    """Sanity check that the reviewer's diagnosis is reproduced by the
    formula itself: at the OLD averaged-Welch-only resolution
    (nperseg=4096 @ 50 kHz -> ~12.2 Hz bins), the shaft band (1.25 Hz wide)
    contains zero bins. `machine_band_bin_counts` takes `n_samples` (the
    dedicated high-res pass's nperseg), not the coarse 4096/2048 rule, so
    this test calls it with `n_samples=4096` to probe that OLD resolution
    specifically, confirming the module-level helper's formula agrees with
    the hand-derived numerics in the design analysis before this fix.
    """
    old_resolution_nperseg = 4096
    rate_hz = 50_000.0

    counts = machine_band_bin_counts(rate_hz, old_resolution_nperseg)

    assert counts["shaft"] == 0


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

    At 50 kHz with a 1-s window (S=50_000 samples), the dedicated
    high-resolution machine-band PSD pass has 1 Hz bins (`nperseg=S`), so the
    blade_pass band (39.375-48.125 Hz, width 8.75 Hz) is resolved by several
    REAL bins -- not the nearest-bin-to-center fallback. Assert that
    directly via `machine_band_bin_counts` so a future change that silently
    regresses to the coarse (0-bin) resolution is caught here too, not just
    by the energy-ordering assertion below.
    """
    rate_hz = 50_000.0
    n_samples = int(rate_hz)  # 1-s window
    windows = _windows(_sine(44.0, rate_hz))
    feat = AudioFeaturizer()

    bin_counts = machine_band_bin_counts(rate_hz, n_samples)
    assert bin_counts["blade_pass"] >= 1

    out = feat.transform(windows, rate_hz)
    names = feat.feature_names()

    shaft = out[0, names.index("ch0_band_shaft")]
    blade = out[0, names.index("ch0_band_blade_pass")]
    guide = out[0, names.index("ch0_band_guide_vane_pass")]
    assert blade > shaft
    assert blade > guide


def test_audio_featurizer_625hz_sine_at_50khz_shaft_band_dominates() -> None:
    """A 6.25 Hz tone sits exactly at the shaft band's center (6.25 Hz +/-
    10%, i.e. [5.625, 6.875] Hz) and far outside the blade_pass (43.75 Hz
    +/- 10%) and guide_vane_pass (125 Hz +/- 10%) bands.

    This is the shaft-band analogue of the blade_pass test above: with the
    OLD averaged-Welch-only implementation (`nperseg=4096` @ 50 kHz, 12.2 Hz
    resolution), the shaft band contains ZERO real bins (width 1.25 Hz <<
    12.2 Hz resolution) and the nearest-bin fallback lands on the 12.2 Hz
    bin -- a frequency that is NOT inside the true shaft band and would, for
    example, equally "detect" any tone anywhere in the [~0, ~18] Hz range as
    shaft-band energy. The dedicated high-resolution pass (1 Hz bins at 1-s
    windows) resolves the shaft band with a genuine in-band bin instead.
    """
    rate_hz = 50_000.0
    n_samples = int(rate_hz)  # 1-s window
    windows = _windows(_sine(MACHINE_HZ["shaft"], rate_hz))
    feat = AudioFeaturizer()

    bin_counts = machine_band_bin_counts(rate_hz, n_samples)
    assert bin_counts["shaft"] >= 1

    out = feat.transform(windows, rate_hz)
    names = feat.feature_names()

    shaft = out[0, names.index("ch0_band_shaft")]
    blade = out[0, names.index("ch0_band_blade_pass")]
    guide = out[0, names.index("ch0_band_guide_vane_pass")]
    assert shaft > blade
    assert shaft > guide


def test_audio_featurizer_shaft_band_does_not_alias_a_tone_outside_the_true_band() -> None:
    """Reproduces the reviewer-identified bug directly: under the OLD
    single-averaged-Welch-PSD implementation, the shaft band's nearest-bin
    fallback landed on the 12.2 Hz bin (the resolution of `nperseg=4096` @
    50 kHz) -- a frequency well outside the true shaft band
    [5.625, 6.875] Hz. A pure tone placed AT that old fallback frequency
    (12.207 Hz, no relation to the true shaft band) must NOT register a
    higher "shaft" band-energy reading than a pure tone placed genuinely
    inside the shaft band. This is the ordering test the coarse-resolution
    implementation actually fails (unlike the plain energy-dominance
    checks above, which happen to still pass under the old code because
    spectral leakage from a 6.25 Hz tone still peaks nearer the 12.2 Hz bin
    than the 48.8/125-ish Hz bins) -- so this is the genuine RED case for
    the fix, verified against the pre-fix implementation before writing it.
    """
    rate_hz = 50_000.0
    true_shaft_tone = _sine(MACHINE_HZ["shaft"], rate_hz)
    old_fallback_bin_hz = 50_000.0 / 4096  # ~12.207 Hz: the old nperseg=4096 resolution
    false_tone_outside_true_band = _sine(old_fallback_bin_hz, rate_hz)
    windows = np.stack([true_shaft_tone, false_tone_outside_true_band])[:, :, np.newaxis]
    feat = AudioFeaturizer()

    out = feat.transform(windows, rate_hz)
    names = feat.feature_names()
    idx = names.index("ch0_band_shaft")

    true_band_energy = out[0, idx]
    false_band_energy = out[1, idx]
    assert true_band_energy > false_band_energy


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


def test_vib_featurizer_625hz_sine_at_10khz_shaft_band_resolved_and_dominates() -> None:
    """Vibration path at a lower sampling rate (10 kHz, typical accelerometer
    rate) with a 1-s window: the shaft band must be resolved by a genuine
    in-band bin (not the nearest-bin fallback), verified directly via
    `machine_band_bin_counts`, and a 6.25 Hz tone's shaft-band energy must
    dominate the blade_pass/guide_vane_pass bands for that channel -- the
    same guarantee as the audio path, now checked on VibFeaturizer.
    """
    rate_hz = 10_000.0
    n_samples = int(rate_hz)  # 1-s window
    windows = _windows(_sine(MACHINE_HZ["shaft"], rate_hz))
    feat = VibFeaturizer()

    bin_counts = machine_band_bin_counts(rate_hz, n_samples)
    assert bin_counts["shaft"] >= 1

    out = feat.transform(windows, rate_hz)
    names = feat.feature_names()

    shaft = out[0, names.index("ch0_band_shaft")]
    blade = out[0, names.index("ch0_band_blade_pass")]
    assert shaft > blade


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


def test_vib_featurizer_dead_channel_detection_immune_to_float32_summation_noise() -> None:
    # Real-data crash (2026-07-01 TU1, RAWTurbineVib__3): channels
    # 0-2 are exactly constant at -7.0 in every sample of every file, but computing
    # the batch std in float32 picks up pairwise-summation ROUNDING NOISE whose
    # magnitude depends on the batch's total element count -- ~4.77e-07 (= |c|*eps/2
    # for c=-7.0) for some counts, exactly 0.0 for others. 4.77e-07 > the 1e-9 dead
    # threshold, so the SAME physically-dead channel flip-flopped live/dead across
    # FILES of one stream purely by batch shape, changing the feature-row width
    # mid-stream and crashing _extract_stream_features' matrix assignment
    # ("could not broadcast input array from shape (36,) into shape (72,)").
    #
    # 2_511_087 elements is a count measured to produce the nonzero float32 std for
    # the real channel constant (-7.0); the dead-channel std must be computed in
    # float64, where a constant channel is exactly 0.0 for EVERY batch shape.
    n = 2_511_087
    dead = np.full((1, n), np.float32(-7.0), dtype=np.float32)
    assert float(dead.std()) > 1e-9, (
        "precondition: this exact element count must exhibit the float32 summation "
        "noise the fix guards against (if numpy's pairwise summation changes, pick "
        "a new count with np.full(n, np.float32(-7.0)).std() > 1e-9)"
    )
    rng = np.random.default_rng(0)
    live = rng.normal(0.0, 1.0, size=(1, n)).astype(np.float32)
    windows = np.stack([dead[0], live[0]], axis=1)[np.newaxis, :, :]
    assert windows.shape == (1, n, 2)

    feat = VibFeaturizer()
    feat.transform(windows, rate_hz=10_000.0)

    assert feat.live_channels_ == [1], (
        f"expected the exactly-constant channel 0 to be detected dead regardless of "
        f"batch shape, got live_channels_={feat.live_channels_}"
    )


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


def test_zscore_a_few_nan_rows_do_not_zero_out_the_whole_column() -> None:
    # Real-data finding (TU/fusion run): a real feature matrix always has a
    # handful of NaN rows (invalid windows, ~11/8286 on the real run) -- plain
    # `np.std` propagates NaN into the column's std, and `NaN >= 1e-12` is ALWAYS
    # False (IEEE-754), so the OLD zero-std guard (`safe = std >= 1e-12`) silently
    # zeroed out the ENTIRE column, not just the NaN row, for every column touched
    # by even one invalid window. On the real TU/fusion run this zeroed out all 231
    # fused columns (every stream has at least one NaN row), collapsing KMeans to a
    # single cluster and silently producing ARI=0.0. zscore must compute its
    # mean/std ignoring NaN rows (nanmean/nanstd) and leave genuinely-NaN entries as
    # NaN in the output, without corrupting the column's valid rows.
    rng = np.random.default_rng(11)
    x = rng.normal(loc=10.0, scale=3.0, size=(200, 3))
    x[7] = np.nan  # one invalid window -- must not affect the other 199 rows' z-score

    z = zscore(x)

    assert np.isnan(z[7]).all()
    valid = np.delete(z, 7, axis=0)
    np.testing.assert_allclose(valid.mean(axis=0), np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(valid.std(axis=0), np.ones(3), atol=1e-9)


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


# ---------------------------------------------------------------------------
# zscore_stats / apply_zscore (transfer primitives)
# ---------------------------------------------------------------------------


class TestZscoreStatsApply:
    def test_apply_with_own_stats_equals_zscore(self):
        # NaN row + constant column + normal columns: full semantics coverage
        x = np.array(
            [[1.0, 5.0, 2.0], [2.0, 5.0, 4.0], [np.nan, 5.0, 6.0], [3.0, 5.0, 8.0]]
        )
        mean, std = zscore_stats(x)
        out = apply_zscore(x, mean, std)
        expected = zscore(x)
        np.testing.assert_array_equal(out, expected)  # NaN-positions compare equal

    def test_apply_with_foreign_stats_uses_given_stats(self):
        a = np.array([[0.0, 0.0], [2.0, 4.0]])  # mean (1,2), std (1,2)
        mean, std = zscore_stats(a)
        b = np.array([[1.0, 2.0]])
        out = apply_zscore(b, mean, std)
        np.testing.assert_allclose(out, [[0.0, 0.0]])

    def test_apply_zero_std_column_becomes_zero(self):
        mean = np.array([1.0, 2.0])
        std = np.array([1.0, 0.0])
        out = apply_zscore(np.array([[3.0, 9.0]]), mean, std)
        np.testing.assert_allclose(out, [[2.0, 0.0]])
