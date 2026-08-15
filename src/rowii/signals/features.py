"""Handcrafted per-window audio/vibration featurizers with machine-frequency bands.

Both featurizers turn a `(W, S, C)` batch of windows (W windows, S samples per
window, C channels) into a `(W, F)` float64 feature matrix. Feature names are
per-channel-expanded using the channel INDEX (e.g. `ch0_log_rms`,
`ch1_band_shaft`, ...) because channel names are not available at this layer
-- callers that need human-readable channel identity must remap `chN_*`
themselves.

Octave energy, spectral centroid, and 95% rolloff are derived from an
AVERAGED Welch PSD estimate
(`scipy.signal.welch(x, fs=rate_hz, nperseg=min(nperseg_cfg, S))`, with
`nperseg_cfg = 4096` for `rate_hz > 20_000` else `2048`). These are broadband
features (octave bands are >= 1 octave wide; centroid/rolloff summarise the
whole spectrum), so averaging several shorter FFT segments to reduce
estimator variance is the right trade-off -- losing some frequency
resolution does not lose information these features actually use.

`MACHINE_HZ` band energies (see below) are narrowband tones, not broadband
content, so they need the OPPOSITE trade-off: resolution over variance
reduction. They are therefore computed from a SEPARATE, DEDICATED
high-resolution PSD pass, `scipy.signal.welch(x, fs=rate_hz,
nperseg=n_samples)` -- i.e. the full window length, no averaging, one
Welch segment. Frequency resolution is then `1 / window_s` (1 Hz at the
project's 1-s windows), which is fine-grained enough that all three
`MACHINE_HZ` bands (shaft: 1.25 Hz wide, blade_pass: 8.75 Hz wide,
guide_vane_pass: 25 Hz wide) contain at least one genuine in-band bin --
see `machine_band_bin_counts` for the formula and
`_band_energy`/`_machine_band_energies` for how it's used. All log-scaled
features use `log10` with a `1e-12` floor to avoid `-inf` on silence.

Band energy is the mean PSD over the FFT bins whose frequency falls inside
the band. The single-bin-nearest-center fallback in `_band_energy` remains
as a last resort for pathological short windows (`window_s < 1 s`, where
even the dedicated high-resolution pass's `1/window_s` resolution could
exceed a band's width) but is NOT exercised in the project's normal
operating regime: with `window_s >= 1 s`, all three `MACHINE_HZ` bands are
resolved by >= 1 real bin, so the fallback path is dead code in practice for
any window at or above 1 s.

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
# full list; VibFeaturizer truncates to centers <= 4000 Hz by design.
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
    can contain zero bins. For `MACHINE_HZ` bands this is avoided in practice
    by feeding this function a dedicated high-resolution PSD (module
    docstring); the nearest-bin fallback below remains only as a defensive
    last resort for windows shorter than 1 s, where even that dedicated
    pass's `1/window_s` resolution could still exceed a band's width -- it is
    the natural generalisation of "mean PSD over band bins" to a zero-bin
    band and keeps the feature meaningful rather than returning NaN or a
    fixed placeholder value.
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


def machine_band_bin_counts(rate_hz: float, n_samples: int) -> dict[str, int]:
    """Count of real (in-band) FFT bins per `MACHINE_HZ` band at the resolution
    the dedicated high-resolution machine-band PSD pass would use for a window
    of `n_samples` samples at `rate_hz` (i.e. `rfftfreq(n_samples, d=1/rate_hz)`,
    matching `nperseg=n_samples` -- see module docstring and
    `_machine_band_energies`, which is the function that actually computes
    machine-band energies at this resolution during `transform`).

    A band whose upper edge exceeds Nyquist is omitted from the result
    entirely (same truncation rule as `_machine_bands`), so the returned dict
    can have fewer than 3 keys at low sampling rates.

    Used both by `_machine_band_energies` (indirectly, via the same
    `rfftfreq` resolution) to document the actual guarantee, and directly by
    tests to assert the guarantee holds (e.g. all three bands resolved by
    >= 1 real bin at 50 kHz/1-s windows) without duplicating the bin-counting
    formula in test code.
    """
    nyquist_hz = rate_hz / 2.0
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / rate_hz)
    counts: dict[str, int] = {}
    for name, lo_hz, hi_hz in _machine_bands(nyquist_hz):
        counts[name] = int(((freqs >= lo_hz) & (freqs <= hi_hz)).sum())
    return counts


def _machine_band_energies(
    x: np.ndarray, rate_hz: float, machine_bands: list[tuple[str, float, float]]
) -> dict[str, float]:
    """Machine-band energies from a DEDICATED high-resolution PSD pass.

    Uses `nperseg=len(x)` (the full window, no Welch averaging/segmenting) so
    the frequency resolution is `rate_hz / len(x) == 1 / window_s` -- fine
    enough (1 Hz at the project's 1-s windows) that all three `MACHINE_HZ`
    bands contain >= 1 real in-band bin (see module docstring and
    `machine_band_bin_counts`). This is intentionally SEPARATE from the
    averaged-Welch PSD used for octave bands/centroid/rolloff
    (`_welch_nperseg`), which trades resolution for lower estimator variance
    -- the right trade-off for broadband features but the wrong one for
    narrowband machine-frequency tones.
    """
    freqs, psd = signal.welch(x, fs=rate_hz, nperseg=len(x))
    return {
        name: _band_energy(freqs, psd, lo_hz, hi_hz)
        for name, lo_hz, hi_hz in machine_bands
    }


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
                machine_energies = _machine_band_energies(x, rate_hz, machine_bands)

                rms = float(np.sqrt(np.mean(np.square(x))))
                out[w, col] = _log10_floor(rms)
                col += 1

                for band_name, _, _ in machine_bands:
                    out[w, col] = _log10_floor(machine_energies[band_name])
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
    entirely (`logging.warning` names the dropped channel index). Caveat:
    this is a batch-GLOBAL check, not per-window -- a channel that is
    constant WITHIN each window but takes a different constant value ACROSS
    windows (nonzero cross-window variance, zero within-window variance)
    counts as live here, since the std is computed over the whole `(W, S)`
    block rather than per window.

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
            # float64, NOT the input's own float32: float32 pairwise summation
            # injects rounding noise of ~|c| * eps/2 into the std of a channel
            # exactly constant at c, and the noise's magnitude depends on the
            # batch's total element count -- measured on real data (2026-07-01
            # TU1 RAWTurbineVib__3, channels constant at -7.0): 4.77e-07 for some
            # per-file batch shapes, exactly 0.0 for others, i.e. ABOVE the 1e-9
            # dead threshold for some files of a stream and below it for the
            # rest, flip-flopping the same physically-dead channel live/dead
            # across files and changing the feature-row width mid-stream (which
            # crashes _extract_stream_features' preallocated-matrix assignment).
            # In float64 a constant channel's std is exactly 0.0 for every batch
            # shape at any realistic batch size.
            std = float(windows[:, :, ch].astype(np.float64).std())
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
                machine_energies = _machine_band_energies(x, rate_hz, machine_bands)

                rms = float(np.sqrt(np.mean(np.square(x))))
                out[w, col] = _log10_floor(rms)
                col += 1

                out[w, col] = float(_scipy_kurtosis(x))
                col += 1

                for band_name, _, _ in machine_bands:
                    out[w, col] = _log10_floor(machine_energies[band_name])
                    col += 1

                for _, lo_hz, hi_hz in octave_bands:
                    energy = _band_energy(freqs, psd, lo_hz, hi_hz)
                    out[w, col] = _log10_floor(energy)
                    col += 1

        return out


def zscore_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column (mean, std) via `nanmean`/`nanstd`, float64 — the statistics
    `zscore` standardizes with, exposed separately so a fitted model can carry its
    FIT-day statistics and re-apply them to another day's features
    (`rowii.state.detect.FittedDetector`, package-2 spec D1).
    """
    x64 = np.asarray(x, dtype=np.float64)
    return np.nanmean(x64, axis=0), np.nanstd(x64, axis=0)


def apply_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """`(x - mean) / std` per column with GIVEN statistics; columns with
    `std < 1e-12` become zero; NaN input rows stay NaN (same semantics as `zscore`,
    which is exactly `apply_zscore(x, *zscore_stats(x))` — regression-gated).
    """
    x64 = np.asarray(x, dtype=np.float64)
    nan_rows = np.isnan(x64).any(axis=1)
    out = np.zeros_like(x64)
    safe = std >= 1e-12
    out[:, safe] = (x64[:, safe] - mean[safe]) / std[safe]
    out[nan_rows] = np.nan
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    """Per-column `(x - mean) / std`, float64, ignoring NaN rows. Columns with
    `std < 1e-12` (over the non-NaN rows) become zero; NaN input rows stay NaN.

    The zero-std guard avoids `inf`/`NaN` for constant columns (e.g. a
    feature that never varies within a batch) -- such a column carries no
    discriminative information, so an all-zero output is the correct neutral
    value rather than propagating a division blow-up.

    Mean/std use `nanmean`/`nanstd` (Task 13 fix): a real feature matrix
    routinely has a handful of NaN rows (invalid windows -- see
    `_StreamFeatureResult.features`'s docstring in `src/rowii/pipeline.py`).
    Plain `.mean()`/`.std()` propagate NaN into EVERY row's statistics for a
    column touched by even one NaN, and the old zero-std guard
    (`std >= 1e-12`) is always False for a NaN std (IEEE-754), which zeroed
    out the WHOLE column -- not just the NaN row -- for any column with any
    invalid window. On a real run (every stream has at least one invalid
    window) this silently zeroed out every fused column, collapsing
    downstream KMeans to a single cluster. Genuinely-NaN rows are restored
    to NaN in the output (not left as whatever the NaN-ignoring arithmetic
    produced) so a caller can still detect and mask them exactly as before.
    """
    return apply_zscore(x, *zscore_stats(x))


def fuse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Z-scored horizontal concatenation: `np.hstack([zscore(a), zscore(b)])`."""
    return np.hstack([zscore(a), zscore(b)])
