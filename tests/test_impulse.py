"""Unit tests for `rowii.anomaly.impulse` -- the band-energy impulse detector
(candidate-kit register criterion #3). Pure numpy math only, synthetic signals,
no `ROWII_DATA_ROOT` needed (real-data validation against the ST-landmark strikes
lives in `tests/test_candidate_kit.py`'s own `@pytest.mark.data` test)."""
from __future__ import annotations

import numpy as np
import pytest

from rowii.anomaly import impulse

# ---------------------------------------------------------------------------
# frame_count: framing math
# ---------------------------------------------------------------------------


def test_frame_count_exact_multiple() -> None:
    # (1000 - 500) // 250 + 1 = 3: frames start at 0, 250, 500 (500+500=1000 fits).
    assert impulse.frame_count(1000, 500, 250) == 3


def test_frame_count_one_short_of_another_frame() -> None:
    # A frame starting at 750 would need samples [750, 1250) -- only 999 available.
    assert impulse.frame_count(999, 500, 250) == 2


def test_frame_count_exactly_one_frame() -> None:
    assert impulse.frame_count(500, 500, 250) == 1


def test_frame_count_shorter_than_one_frame_is_zero() -> None:
    assert impulse.frame_count(499, 500, 250) == 0
    assert impulse.frame_count(0, 500, 250) == 0


# ---------------------------------------------------------------------------
# band_frame_energy: 5-20 kHz band energy per 10 ms frame
# ---------------------------------------------------------------------------

_RATE_HZ = 50_000.0


def _tone(
    freq_hz: float, n_samples: int, rate_hz: float = _RATE_HZ, amp: float = 1.0
) -> np.ndarray:
    t = np.arange(n_samples) / rate_hz
    return amp * np.sin(2 * np.pi * freq_hz * t)


def test_band_frame_energy_returns_frame_count_length() -> None:
    x = _tone(10_000.0, 2000)
    e = impulse.band_frame_energy(x, _RATE_HZ)
    assert e.shape == (impulse.frame_count(2000, 500, 250),)


def test_band_frame_energy_in_band_tone_far_exceeds_out_of_band_tone() -> None:
    # 10 kHz sits inside [5000, 20000); 1 kHz sits well outside it. Same amplitude.
    in_band = impulse.band_frame_energy(_tone(10_000.0, 4000), _RATE_HZ)
    out_of_band = impulse.band_frame_energy(_tone(1_000.0, 4000), _RATE_HZ)
    assert np.all(in_band > out_of_band + 3.0)  # log10 power, >3 decades apart


def test_band_frame_energy_silence_is_far_below_a_tone() -> None:
    silence = impulse.band_frame_energy(np.zeros(4000), _RATE_HZ)
    tone = impulse.band_frame_energy(_tone(10_000.0, 4000), _RATE_HZ)
    assert np.all(silence < tone - 3.0)


def test_band_frame_energy_empty_when_shorter_than_one_frame() -> None:
    e = impulse.band_frame_energy(np.zeros(100), _RATE_HZ)
    assert e.shape == (0,)


def test_band_frame_energy_no_bin_in_band_raises() -> None:
    # At an 8 kHz rate, the whole [5000, 20000) band lies above Nyquist (4 kHz).
    with pytest.raises(ValueError, match="no FFT bin"):
        impulse.band_frame_energy(_tone(1000.0, 4000, rate_hz=8000.0), 8000.0)


# ---------------------------------------------------------------------------
# mad_z_score: rolling-median detrend + MAD z-score
# ---------------------------------------------------------------------------


def test_mad_z_score_flat_signal_is_all_zero() -> None:
    z = impulse.mad_z_score(np.zeros(50), hop_s=1.0, med_win_s=3.0)
    assert np.allclose(z, 0.0)


def test_mad_z_score_isolated_spike_dominates() -> None:
    loge = np.zeros(11)
    loge[5] = 100.0
    z = impulse.mad_z_score(loge, hop_s=1.0, med_win_s=3.0)
    assert z[5] > 1e6
    other = np.delete(z, 5)
    assert np.allclose(other, 0.0)


def test_mad_z_score_removes_a_linear_drift() -> None:
    # A monotone ramp: the rolling median of any interior odd-window is the
    # centre sample itself, so detrending should leave residuals (and z) at
    # exactly zero despite the raw values spanning 0..10.
    loge = np.linspace(0.0, 10.0, 11)
    z = impulse.mad_z_score(loge, hop_s=1.0, med_win_s=3.0)
    assert np.allclose(z, 0.0, atol=1e-9)


def test_mad_z_score_empty_input_returns_empty() -> None:
    z = impulse.mad_z_score(np.zeros(0), hop_s=1.0, med_win_s=3.0)
    assert z.shape == (0,)


# ---------------------------------------------------------------------------
# pick_peaks: greedy descending-z peak-picking with non-max suppression
# ---------------------------------------------------------------------------


def test_pick_peaks_suppresses_the_weaker_of_two_close_peaks() -> None:
    z = np.zeros(20)
    z[5] = 8.0
    z[7] = 7.5  # 2 frames from idx5, inside min_sep (3 frames) -> suppressed
    peaks = impulse.pick_peaks(z, hop_s=1.0, min_sep_s=3.0, z_min=6.0)
    assert [p.time_offset_s for p in peaks] == [5.0]
    assert peaks[0].z == pytest.approx(8.0)


def test_pick_peaks_keeps_two_peaks_far_enough_apart() -> None:
    z = np.zeros(20)
    z[5] = 8.0
    z[12] = 9.0  # 7 frames away, outside min_sep (3 frames)
    peaks = impulse.pick_peaks(z, hop_s=1.0, min_sep_s=3.0, z_min=6.0)
    assert [p.time_offset_s for p in peaks] == [5.0, 12.0]


def test_pick_peaks_visits_in_descending_z_order_not_time_order() -> None:
    z = np.zeros(20)
    z[12] = 9.0
    z[14] = 6.5  # 2 frames from idx12 (earlier in time but weaker) -> suppressed
    peaks = impulse.pick_peaks(z, hop_s=1.0, min_sep_s=3.0, z_min=6.0)
    assert [p.time_offset_s for p in peaks] == [12.0]


def test_pick_peaks_below_threshold_never_returned() -> None:
    z = np.zeros(20)
    z[2] = 5.9  # below the default 6.0 register threshold
    peaks = impulse.pick_peaks(z, hop_s=1.0, min_sep_s=3.0, z_min=6.0)
    assert peaks == []


def test_pick_peaks_empty_input_returns_empty_list() -> None:
    assert impulse.pick_peaks(np.zeros(0), hop_s=1.0, min_sep_s=3.0, z_min=6.0) == []


def test_pick_peaks_uses_module_default_z_min() -> None:
    z = np.zeros(20)
    z[5] = impulse.Z_REGISTER_THRESHOLD - 0.5  # just below the default
    assert impulse.pick_peaks(z, hop_s=1.0, min_sep_s=3.0) == []


# ---------------------------------------------------------------------------
# detect_impulses: end-to-end composition on a synthetic impulse
# ---------------------------------------------------------------------------


def test_detect_impulses_finds_an_injected_in_band_impulse() -> None:
    rng = np.random.default_rng(7)
    n = int(3.0 * _RATE_HZ)  # 3 s of audio
    samples = rng.normal(scale=0.01, size=n)
    impulse_start = int(1.5 * _RATE_HZ)
    samples[impulse_start : impulse_start + 200] += _tone(10_000.0, 200, amp=5.0)

    peaks = impulse.detect_impulses(samples, _RATE_HZ)
    assert len(peaks) >= 1
    closest = min(peaks, key=lambda p: abs(p.time_offset_s - 1.5))
    assert closest.time_offset_s == pytest.approx(1.5, abs=0.05)
    assert closest.z >= impulse.Z_REGISTER_THRESHOLD


def test_detect_impulses_pure_noise_finds_nothing_at_the_register_threshold() -> None:
    rng = np.random.default_rng(3)
    samples = rng.normal(scale=0.01, size=int(2.0 * _RATE_HZ))
    peaks = impulse.detect_impulses(samples, _RATE_HZ)
    assert peaks == []


def test_detect_impulses_z_min_is_configurable() -> None:
    rng = np.random.default_rng(3)
    samples = rng.normal(scale=0.01, size=int(2.0 * _RATE_HZ))
    # An absurdly low z_min must find *something* even in pure noise.
    peaks = impulse.detect_impulses(samples, _RATE_HZ, z_min=0.01)
    assert len(peaks) >= 1
