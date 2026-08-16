"""Per-window log-mel featurizer for the `logmel` variant.

Feeds the reconstruction scorers (`rowii.anomaly.recon`): each 1-second window
becomes a flattened (frames x mels) log-mel patch whose window-INTERNAL time
axis is the sequence the LSTM/Conv autoencoders consume -- no cross-window
contiguity is needed, so the `Scorer` protocol holds unchanged. Primary mic
stream only (size bound). Pure NumPy (Hann window + rFFT + triangular
mel filterbank); no torch/librosa dependency.

Shape contract: `LogmelFeaturizer.transform` accepts both the pipeline's own
3-D `(B, S, C)` float32 window stacks (`rowii.pipeline._extract_stream_features`
hands every featurizer this shape -- see `AudioFeaturizer`/`BeatsFeaturizer`),
mono-mixing channels by mean exactly like `BeatsFeaturizer.transform`, and
already-mono 2-D `(B, S)` waveforms.
"""
from __future__ import annotations

import numpy as np

_LOG_FLOOR = 1e-10

_DEFAULT_GEOMETRY_RATE_HZ = 50_000.0
"""The plant's real mic sample rate -- with the project's 1-s windows this is the
default geometry `feature_names()` falls back to before any `transform` call."""


def _mel(hz: float) -> float:
    return float(2595.0 * np.log10(1.0 + hz / 700.0))


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return np.asarray(700.0 * (10.0 ** (mel / 2595.0) - 1.0))


def _mel_filterbank(
    n_mels: int, n_fft_bins: int, rate_hz: float, fmin_hz: float = 20.0
) -> np.ndarray:
    """`(n_mels, n_fft_bins)` triangular filters, mel-spaced between fmin and rate/2."""
    fmax_hz = rate_hz / 2.0
    mel_edges = np.linspace(_mel(fmin_hz), _mel(fmax_hz), n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)
    fft_freqs = np.linspace(0.0, fmax_hz, n_fft_bins)
    bank = np.zeros((n_mels, n_fft_bins), dtype=np.float64)
    for m in range(n_mels):
        lo, mid, hi = hz_edges[m], hz_edges[m + 1], hz_edges[m + 2]
        rising = (fft_freqs - lo) / max(mid - lo, 1e-9)
        falling = (hi - fft_freqs) / max(hi - mid, 1e-9)
        bank[m] = np.clip(np.minimum(rising, falling), 0.0, None)
    return bank


class LogmelFeaturizer:
    """Flattened (frames x mels) log-mel patch per window -- see module docstring.

    Frame count follows `n_frames = 1 + (n_samples - frame_len) // hop_len` from
    the window's own sample count and rate -- never hardcoded. At the plant's
    50 kHz / 1-s geometry the defaults (25 ms frame = 1250 samples, 20 ms hop =
    1000 samples) give exactly 49 frames, i.e. 49 x 64 = 3136 features per
    window (pinned by `tests/test_logmel.py` for that geometry only).
    """

    name: str = "logmel"

    def __init__(self, n_mels: int = 64, frame_s: float = 0.025, hop_s: float = 0.020) -> None:
        self.n_mels = n_mels
        self.frame_s = frame_s
        self.hop_s = hop_s
        self._n_frames: int | None = None  # set on first transform, for feature_names

    def _frame_geometry(self, rate_hz: float) -> tuple[int, int]:
        """`(frame_len, hop_len)` in samples at *rate_hz* -- the ONE place the
        seconds -> samples rounding lives (shared by `transform` and
        `feature_names`' default-geometry fallback)."""
        return int(round(self.frame_s * rate_hz)), int(round(self.hop_s * rate_hz))

    def transform(self, stack: np.ndarray, rate_hz: float) -> np.ndarray:
        """`(B, n_samples)` or `(B, n_samples, C)` windows -> `(B, n_frames * n_mels)`
        float64 flattened log-mel patches, frame-major (`reshape(-1)` of a
        `(n_frames, n_mels)` patch).

        A 3-D `(B, S, C)` stack (the pipeline's per-stream featurizer contract,
        `rowii.pipeline._extract_stream_features`) is mono-mixed over channels by
        mean first, mirroring `BeatsFeaturizer.transform`.
        """
        if stack.ndim == 3:
            stack = stack.mean(axis=2)  # mono-mix over channels (BeatsFeaturizer's rule)
        if stack.ndim != 2:
            raise ValueError(
                f"LogmelFeaturizer.transform expects (B, n_samples) or (B, n_samples, C) "
                f"windows, got shape {stack.shape}"
            )
        frame_len, hop_len = self._frame_geometry(rate_hz)
        n_samples = stack.shape[1]
        n_frames = 1 + (n_samples - frame_len) // hop_len
        if n_frames < 1:
            raise ValueError(
                f"window of {n_samples} samples at {rate_hz} Hz is shorter than one "
                f"{frame_len}-sample frame -- cannot compute a log-mel patch"
            )
        self._n_frames = n_frames

        window = np.hanning(frame_len)
        idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
        bank = _mel_filterbank(self.n_mels, frame_len // 2 + 1, rate_hz)

        out = np.empty((stack.shape[0], n_frames * self.n_mels), dtype=np.float64)
        for b in range(stack.shape[0]):
            frames = stack[b][idx] * window            # (n_frames, frame_len)
            power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
            mel_energy = power @ bank.T                # (n_frames, n_mels)
            out[b] = np.log10(mel_energy + _LOG_FLOOR).reshape(-1)
        return out

    def feature_names(self) -> list[str]:
        """`["logmel_f0_m0", ..., "logmel_f{F-1}_m{M-1}"]`, frame-major -- matching
        `transform`'s column order. Before any `transform` call the frame count
        falls back to the 50 kHz / 1-s default geometry (the plant data), so for
        that geometry the names are identical before and after `transform`."""
        if self._n_frames is None:
            # 50 kHz / 1 s default geometry, matching the plant data
            frame_len, hop_len = self._frame_geometry(_DEFAULT_GEOMETRY_RATE_HZ)
            self._n_frames = 1 + (round(_DEFAULT_GEOMETRY_RATE_HZ) - frame_len) // hop_len
        return [
            f"logmel_f{f}_m{m}"
            for f in range(self._n_frames)
            for m in range(self.n_mels)
        ]
