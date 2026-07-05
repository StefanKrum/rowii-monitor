"""Handcrafted per-window audio/vibration featurizers with machine-frequency bands.

Both featurizers turn a `(W, S, C)` batch of windows (W windows, S samples per
window, C channels) into a `(W, F)` float64 feature matrix. Feature names are
per-channel-expanded using the channel INDEX (e.g. `ch0_log_rms`,
`ch1_band_shaft`, ...) because channel names are not available at this layer
-- callers that need human-readable channel identity must remap `chN_*`
themselves.

Spectral features (band energy, octave energy, spectral centroid, 95%
rolloff) are derived from a Welch PSD estimate
(`scipy.signal.welch(x, fs=rate_hz, nperseg=min(nperseg_cfg, S))`, with
`nperseg_cfg = 4096` for `rate_hz > 20_000` else `2048`). All log-scaled
features use `log10` with a `1e-12` floor to avoid `-inf` on silence.

Band energy is the mean PSD over the FFT bins whose frequency falls inside
the band. Because a Welch PSD has a fixed frequency resolution
(`rate_hz / nperseg`), a band narrower than that resolution can contain ZERO
bins -- this happens for the narrow `MACHINE_HZ` bands (+/-10 % of a few Hz
to ~44 Hz) at `nperseg=4096`/50 kHz (resolution ~12.2 Hz). In that case the
single FFT bin nearest the band's CENTER frequency is used instead, which is
the natural generalisation of "mean PSD over band bins" to a zero-bin band
and keeps the feature meaningful (a tone at 44 Hz still shows up in the
`blade_pass` band because 48.8 Hz -- the nearest bin to the 43.75 Hz center
-- captures the tone's spectral leakage) rather than emitting a fixed
placeholder value.

Both `MACHINE_HZ` bands and the octave bands adapt to the sampling rate:
any band whose upper edge exceeds Nyquist (`rate_hz / 2`) is skipped
entirely (not just zero-filled), so `feature_names()` -- and the transform
output width -- varies with `rate_hz`.
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from scipy import signal
from scipy.stats import kurtosis as _scipy_kurtosis

logger = logging.getLogger(__name__)

MACHINE_HZ: dict[str, float] = {
    "shaft": 6.25,
    "blade_pass": 43.75,
    "guide_vane_pass": 125.0,
}

# Standard octave-band center frequencies, ascending. AudioFeaturizer uses the
# full list; VibFeaturizer truncates to centers <= 4000 Hz per the brief.
_OCTAVE_CENTERS_HZ: tuple[float, ...] = (
    31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0,
)
_VIB_OCTAVE_CENTERS_HZ: tuple[float, ...] = tuple(
    fc for fc in _OCTAVE_CENTERS_HZ if fc <= 4000.0
)

_LOG_FLOOR = 1e-12
_DEAD_CHANNEL_STD = 1e-9
_ROLLOFF_FRACTION = 0.95


class Featurizer(Protocol):
    """Common shape for handcrafted featurizers: name + feature_names + transform."""

    name: str

    def feature_names(self) -> list[str]:
        """Feature column names, in the same order as `transform`'s output columns."""
        ...

    def transform(self, windows: np.ndarray, rate_hz: float) -> np.ndarray:
        """Map `(W, S, C)` float32 windows to a `(W, F)` float64 feature matrix."""
        ...


def _welch_nperseg(rate_hz: float, n_samples: int) -> int:
    base = 4096 if rate_hz > 20_000.0 else 2048
    return min(base, n_samples)


def _log10_floor(x: np.ndarray | float) -> np.ndarray | float:
    return np.log10(np.maximum(x, _LOG_FLOOR))


def _band_energy(freqs: np.ndarray, psd: np.ndarray, lo_hz: float, hi_hz: float) -> float:
    """Mean PSD over bins inside `[lo_hz, hi_hz]`; nearest single bin if none fall inside.

    A band narrower than the PSD's frequency resolution (`freqs[1] - freqs[0]`)
    can contain zero bins -- see module docstring for why the nearest-bin
    fallback is the correct generalisation here, rather than e.g. returning NaN
    or a fixed floor value.
    """
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    if mask.any():
        return float(psd[mask].mean())
    center_hz = (lo_hz + hi_hz) / 2.0
    nearest = int(np.argmin(np.abs(freqs - center_hz)))
    return float(psd[nearest])


def _spectral_centroid(freqs: np.ndarray, psd: np.ndarray) -> float:
    total = psd.sum()
    if total <= 0.0:
        return 0.0
    return float(np.sum(freqs * psd) / total)


def _rolloff95(freqs: np.ndarray, psd: np.ndarray) -> float:
    cumulative = np.cumsum(psd)
    total = cumulative[-1]
    if total <= 0.0:
        return float(freqs[-1])
    threshold = _ROLLOFF_FRACTION * total
    idx = int(np.searchsorted(cumulative, threshold))
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def _machine_bands(nyquist_hz: float) -> list[tuple[str, float, float]]:
    """`(name, lo_hz, hi_hz)` for every `MACHINE_HZ` entry whose band fits under Nyquist."""
    bands = []
    for name, f_hz in MACHINE_HZ.items():
        lo_hz, hi_hz = f_hz * 0.9, f_hz * 1.1
        if hi_hz <= nyquist_hz:
            bands.append((name, lo_hz, hi_hz))
    return bands


def _octave_bands(
    centers_hz: tuple[float, ...], nyquist_hz: float
) -> list[tuple[float, float, float]]:
    """`(center_hz, lo_hz, hi_hz)` for every center whose band fits under Nyquist."""
    bands = []
    for fc in centers_hz:
        lo_hz, hi_hz = fc / np.sqrt(2.0), fc * np.sqrt(2.0)
        if hi_hz <= nyquist_hz:
            bands.append((fc, lo_hz, hi_hz))
    return bands


class AudioFeaturizer:
    """Per-channel handcrafted spectral/level features for audio (mic) windows.

    Features per channel `N` (`chN_*`): `log_rms`, a `band_<name>` energy for
    every `MACHINE_HZ` band that fits under Nyquist, an `octave_<center>`
    energy for every octave band (31.5..8000 Hz) that fits under Nyquist, and
    `spectral_centroid` + `rolloff95`. Dead channels are NOT dropped -- mic
    validity is handled upstream of this featurizer.
    """

    name: str = "audio-handcrafted"

    def __init__(self) -> None:
        self._feature_names: list[str] = []

    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def transform(self, windows: np.ndarray, rate_hz: float) -> np.ndarray:
        n_windows, n_samples, n_channels = windows.shape
        nyquist_hz = rate_hz / 2.0
        machine_bands = _machine_bands(nyquist_hz)
        octave_bands = _octave_bands(_OCTAVE_CENTERS_HZ, nyquist_hz)
        nperseg = _welch_nperseg(rate_hz, n_samples)

        names: list[str] = []
        for ch in range(n_channels):
            prefix = f"ch{ch}"
            names.append(f"{prefix}_log_rms")
            names.extend(f"{prefix}_band_{band_name}" for band_name, _, _ in machine_bands)
            names.extend(
                f"{prefix}_octave_{int(fc)}" for fc, _, _ in octave_bands
            )
            names.append(f"{prefix}_spectral_centroid")
            names.append(f"{prefix}_rolloff95")
        self._feature_names = names

        out = np.empty((n_windows, len(names)), dtype=np.float64)
        for w in range(n_windows):
            col = 0
            for ch in range(n_channels):
                x = windows[w, :, ch].astype(np.float64)
                freqs, psd = signal.welch(x, fs=rate_hz, nperseg=nperseg)

                rms = float(np.sqrt(np.mean(np.square(x))))
                out[w, col] = _log10_floor(rms)
                col += 1

                for _, lo_hz, hi_hz in machine_bands:
                    energy = _band_energy(freqs, psd, lo_hz, hi_hz)
                    out[w, col] = _log10_floor(energy)
                    col += 1

                for _, lo_hz, hi_hz in octave_bands:
                    energy = _band_energy(freqs, psd, lo_hz, hi_hz)
                    out[w, col] = _log10_floor(energy)
                    col += 1

                out[w, col] = _spectral_centroid(freqs, psd)
                col += 1
                out[w, col] = _rolloff95(freqs, psd)
                col += 1

        return out


class VibFeaturizer:
    """Per-live-channel handcrafted features for vibration (accelerometer) windows.

    Features per live channel `N` (`chN_*`): `log_rms`, `kurtosis`, a
    `band_<name>` energy for every `MACHINE_HZ` band that fits under Nyquist,
    and an `octave_<center>` energy for every octave band up to 4000 Hz that
    fits under Nyquist. A channel whose std is `< 1e-9` across the WHOLE
    `(W, S)` input is considered dead: it is dropped from the feature set
    entirely (`logging.warning` names the dropped channel index).

    Stateful design: `transform` discovers live channels from the batch it is
    given and caches them on `self.live_channels_` / `self._feature_names`.
    `feature_names()` is only meaningful AFTER a `transform` call -- it raises
    `RuntimeError` if called first, since there is otherwise no way to know
    which channels are live without having seen data.
    """

    name: str = "vib-handcrafted"

    def __init__(self) -> None:
        self.live_channels_: list[int] | None = None
        self._feature_names: list[str] = []

    def feature_names(self) -> list[str]:
        if self.live_channels_ is None:
            raise RuntimeError(
                "VibFeaturizer.feature_names() is only valid after transform() has "
                "been called at least once (live channels are discovered from data)"
            )
        return list(self._feature_names)

    def transform(self, windows: np.ndarray, rate_hz: float) -> np.ndarray:
        n_windows, n_samples, n_channels = windows.shape
        nyquist_hz = rate_hz / 2.0
        machine_bands = _machine_bands(nyquist_hz)
        octave_bands = _octave_bands(_VIB_OCTAVE_CENTERS_HZ, nyquist_hz)
        nperseg = _welch_nperseg(rate_hz, n_samples)

        live_channels: list[int] = []
        for ch in range(n_channels):
            std = float(windows[:, :, ch].std())
            if std < _DEAD_CHANNEL_STD:
                logger.warning(
                    "VibFeaturizer: channel %d is dead (std=%.3e < %.1e over the full "
                    "batch) -- dropped from the feature set",
                    ch,
                    std,
                    _DEAD_CHANNEL_STD,
                )
            else:
                live_channels.append(ch)
        if not live_channels:
            raise ValueError(
                f"VibFeaturizer.transform: all {n_channels} channel(s) are dead "
                "(std < 1e-9) -- no live channel to build features from"
            )
        self.live_channels_ = live_channels

        names: list[str] = []
        for ch in live_channels:
            prefix = f"ch{ch}"
            names.append(f"{prefix}_log_rms")
            names.append(f"{prefix}_kurtosis")
            names.extend(f"{prefix}_band_{band_name}" for band_name, _, _ in machine_bands)
            names.extend(
                f"{prefix}_octave_{int(fc)}" for fc, _, _ in octave_bands
            )
        self._feature_names = names

        out = np.empty((n_windows, len(names)), dtype=np.float64)
        for w in range(n_windows):
            col = 0
            for ch in live_channels:
                x = windows[w, :, ch].astype(np.float64)
                freqs, psd = signal.welch(x, fs=rate_hz, nperseg=nperseg)

                rms = float(np.sqrt(np.mean(np.square(x))))
                out[w, col] = _log10_floor(rms)
                col += 1

                out[w, col] = float(_scipy_kurtosis(x))
                col += 1

                for _, lo_hz, hi_hz in machine_bands:
                    energy = _band_energy(freqs, psd, lo_hz, hi_hz)
                    out[w, col] = _log10_floor(energy)
                    col += 1

                for _, lo_hz, hi_hz in octave_bands:
                    energy = _band_energy(freqs, psd, lo_hz, hi_hz)
                    out[w, col] = _log10_floor(energy)
                    col += 1

        return out


def zscore(x: np.ndarray) -> np.ndarray:
    """Per-column `(x - mean) / std`, float64. Columns with `std < 1e-12` become zero.

    The zero-std guard avoids `inf`/`NaN` for constant columns (e.g. a
    feature that never varies within a batch) -- such a column carries no
    discriminative information, so an all-zero output is the correct neutral
    value rather than propagating a division blow-up.
    """
    x64 = np.asarray(x, dtype=np.float64)
    mean = x64.mean(axis=0)
    std = x64.std(axis=0)
    out = np.zeros_like(x64)
    safe = std >= 1e-12
    out[:, safe] = (x64[:, safe] - mean[safe]) / std[safe]
    return out


def fuse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Z-scored horizontal concatenation: `np.hstack([zscore(a), zscore(b)])`."""
    return np.hstack([zscore(a), zscore(b)])
