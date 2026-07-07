"""Frozen-BEATs audio featurizer (Task 14, behind the `[beats]` extra).

Only this module (and `rowii.signals.beats_model`, which it delegates
checkpoint loading and fbank computation to) imports `torch`/`torchaudio` --
every other `rowii` module stays torch-free so the core package works
without the optional `[beats]` extra installed (`pyproject.toml`'s
`[project.optional-dependencies].beats`).

`BeatsFeaturizer.transform` mirrors the shape contract of the handcrafted
featurizers (`rowii.signals.features.AudioFeaturizer`/`VibFeaturizer`):
`(W, S, C)` float32 windows in, `(W, D)` float64 features out. Internally:
mono-mix over channels (mean) -> resample *rate_hz* -> 16 kHz
(`torchaudio.functional.resample`) -> 128-mel BEATs fbank
(`rowii.signals.beats_model.beats_fbank`) -> frozen encoder
(`BeatsEncoderProtocol.extract`) -> mean-pool over tokens. Batched over all W
windows in one encoder forward pass (`torch.no_grad()`), not windowed one at
a time -- the pipeline's real workloads are dominated by disk I/O and
feature extraction happens per burst file already (`scripts/run_step1.py`'s
`_extract_stream_features`), so a single-file's worth of windows (at most a
few hundred at 1-s windows / 12-min bursts) comfortably fits one batch on
even CPU-only hardware.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    import torch

    from rowii.vendor.beats.BEATs import BEATs

_FORCE_CPU_ENV = "ROWII_FORCE_CPU"


def best_device() -> torch.device:
    """Best available torch device: `ROWII_FORCE_CPU` env > mps > cuda > cpu.

    `ROWII_FORCE_CPU` (any non-empty value) is a debug escape hatch, checked
    first regardless of what GPU backends are actually available -- matching
    the equivalent `PSHP_FORCE_CPU` convention in the sibling
    `pshp-ssl-transfer` repo's `rowii.core.encoders.device` (this project's
    own env var is prefixed `ROWII_` instead, per this repo's naming).
    """
    import torch

    if os.environ.get(_FORCE_CPU_ENV):
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BeatsEncoderProtocol(Protocol):
    """Anything that maps one window's fbank to per-token embeddings.

    `extract`'s output is mean-pooled over the token (time) dimension by
    `BeatsFeaturizer.transform`, not by the encoder itself -- an encoder only
    needs to return `(n_frames, D)`, whatever `D` it produces.
    """

    def extract(self, fbank: torch.Tensor) -> torch.Tensor:
        """`(n_frames, n_mels)` fbank -> `(n_frames_out, D)` per-token embeddings."""
        ...


class _RealBeatsEncoder:
    """Wraps the loaded (frozen) BEATs model as a `BeatsEncoderProtocol`.

    `extract` batches over the leading window dimension internally (its
    input here is always a single window's fbank, called once per window by
    `BeatsFeaturizer.transform`'s batched path) -- see that method's
    docstring for why the whole-batch forward pass, not a per-window one, is
    what actually executes in the real pipeline.
    """

    def __init__(self, model: BEATs) -> None:
        self._model = model

    def extract(self, fbank: torch.Tensor) -> torch.Tensor:
        """One window's fbank (no padding -- always a single, fully-real
        waveform here) through patch embedding + the transformer encoder.

        Mirrors `rowii.vendor.beats.BEATs.BEATs.extract_features`'s body from
        its OWN `fbank.unsqueeze(1)` step onward (that method starts from a
        raw waveform and computes the fbank itself via `self.preprocess`;
        this class's `BeatsEncoderProtocol.extract` contract instead takes an
        already-computed fbank -- see `BeatsFeaturizer.transform`'s docstring
        for why fbank computation is a separate, independently-testable
        step). `padding_mask` is always `None` here: every call is a single
        complete window, never a padded batch, so every one of the
        reference's `if padding_mask is not None:` branches is a genuine
        no-op, not a shortcut around real masking logic.
        """
        model = self._model
        x = fbank.unsqueeze(0).unsqueeze(1)  # (1, 1, n_frames, n_mels)
        x = model.patch_embedding(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.transpose(1, 2)
        x = model.layer_norm(x)
        if model.post_extract_proj is not None:
            x = model.post_extract_proj(x)
        x = model.dropout_input(x)
        features, _ = model.encoder(x, padding_mask=None)
        # TransformerEncoder.forward (vendored, unannotated) returns
        # (Tensor, list) at runtime; mypy only sees Any for the first element
        # since the vendor module carries no type annotations.
        return cast("torch.Tensor", features[0])


class BeatsFeaturizer:
    """Frozen-BEATs embedding featurizer: `(W, S, C)` windows -> `(W, D)` embeddings."""

    name = "beats"

    def __init__(
        self,
        checkpoint: Path,
        device: torch.device | None = None,
        encoder: BeatsEncoderProtocol | None = None,
    ) -> None:
        """Args:
        checkpoint: Path to the official BEATs `.pt` checkpoint. Ignored if
            *encoder* is given (tests inject a stub encoder and never need a
            real checkpoint on disk).
        device: Torch device to run on; `best_device()` if `None`.
        encoder: Injected `BeatsEncoderProtocol` (e.g. a test stub). If
            `None`, loads the real frozen model from *checkpoint* via
            `rowii.signals.beats_model.load_beats_model`.
        """
        self.device = device if device is not None else best_device()
        if encoder is not None:
            self._encoder = encoder
            self._embed_dim: int | None = None
        else:
            from rowii.signals.beats_model import BEATS_EMBED_DIM, load_beats_model

            model = load_beats_model(checkpoint, self.device)
            self._encoder = _RealBeatsEncoder(model)
            self._embed_dim = BEATS_EMBED_DIM

    def feature_names(self) -> list[str]:
        """`["beats_0", ..., "beats_{D-1}"]`, `D` = the encoder's embedding width.

        `D` is known upfront for the real encoder (`BEATS_EMBED_DIM`, fixed by
        the checkpoint's architecture); for an injected stub encoder (tests),
        `D` is only known after the first `transform()` call, since the stub's
        output width is whatever the stub returns.
        """
        if self._embed_dim is None:
            raise RuntimeError(
                "BeatsFeaturizer.feature_names() with a stub encoder is only valid "
                "after transform() has been called at least once (embedding width is "
                "discovered from the stub's own output)"
            )
        return [f"beats_{i}" for i in range(self._embed_dim)]

    def transform(self, windows: np.ndarray, rate_hz: float) -> np.ndarray:
        """`(W, S, C)` float32 windows -> `(W, D)` float64 BEATs embeddings.

        Pipeline per window: mono-mix over channels (mean) -> resample
        *rate_hz* -> 16 kHz -> BEATs fbank -> frozen encoder -> mean-pool
        over tokens. All W windows are processed in a single Python loop over
        the encoder (batched internally would require padding fbanks of
        possibly-different frame counts to a common length; at this
        pipeline's fixed 1-s windows every window's fbank has an identical
        frame count, but the per-window loop is kept rather than assuming
        that invariant here, since `transform`'s own contract does not
        guarantee equal-length windows across calls).
        """
        import torch
        import torchaudio

        from rowii.signals.beats_model import BEATS_SAMPLE_RATE_HZ, beats_fbank

        n_windows = windows.shape[0]
        mono = windows.mean(axis=2)  # (W, S) -- mean over channels

        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for w in range(n_windows):
                waveform = torch.from_numpy(mono[w].astype(np.float32)).to(self.device)
                if rate_hz != BEATS_SAMPLE_RATE_HZ:
                    waveform = torchaudio.functional.resample(
                        waveform, orig_freq=int(round(rate_hz)), new_freq=BEATS_SAMPLE_RATE_HZ
                    )
                fbank = beats_fbank(waveform)
                tokens = self._encoder.extract(fbank)  # (n_frames_out, D)
                pooled = tokens.mean(dim=0)  # (D,)
                embeddings.append(pooled.cpu().numpy())

        if self._embed_dim is None:
            self._embed_dim = embeddings[0].shape[0]

        return np.stack(embeddings, axis=0).astype(np.float64)
