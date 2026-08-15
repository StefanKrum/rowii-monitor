"""Frozen-TF-C audio featurizer (package-4 spec D1/D4, Time-Frequency
Consistency -- Zhang et al. 2022): the compact industrial-pretraining
counterpart to `rowii.signals.beats.BeatsFeaturizer`, later pre-trained on
public corpora and integrated as a frozen-embedding variant alongside BEATs.

Torch-free story, mirroring `rowii.signals.beats`/`rowii.signals.beats_model`'s
split: this module (`wrapper.py`) never imports `torch` at module level, so
importing or CONSTRUCTING `TfcFeaturizer` never requires the optional
`[beats]` extra -- only actually calling `transform()` on the real
(non-stub-encoder) path does. `rowii.tfc.model` is the one module in this
package that imports torch at module level (the `_recon_models.py`
precedent), and is therefore imported ONLY lazily, from inside this module's
functions/methods, never at `wrapper.py`'s own top level. `TfcConfig` lives
HERE rather than in `model.py` precisely so it stays importable without
torch: callers that only need to describe a TF-C architecture (e.g. to read
a checkpoint's `cfg` field, Task 3/4) never pull in torch as a side effect.

`TfcFeaturizer.transform` mirrors the shape contract of the other featurizers
(`rowii.signals.features.AudioFeaturizer`/`VibFeaturizer`,
`rowii.signals.beats.BeatsFeaturizer`, `rowii.signals.logmel.LogmelFeaturizer`):
`(B, S)` or `(B, S, C)` windows in, `(B, 256)` float64 features out. Internally:
mono-mix over channels (mean, `BeatsFeaturizer`'s rule) -> resample *rate_hz*
to 8 kHz (`_resample_to_8khz`) -> [stub encoder: `encoder.embed()`, done] OR
[real path: per-window standardize -> chunks of `_CHUNK_SIZE` windows through
the frozen `TfcModel` under `torch.no_grad()` -> `h_t ⊕ h_f`, the
PRE-projection 128+128 = 256-d pooled encoder outputs]. Unlike
`BeatsFeaturizer`, construction never raises and never imports torch even
when a real *checkpoint* path is given and no *encoder* stub is injected --
loading the frozen model is deferred to the first `transform()` call and
cached afterwards (`TfcFeaturizer._real_encoder`), which is what lets this
whole module (not just its imports) stay usable without the `[beats]` extra
until a real embedding is actually requested.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.signal import resample_poly

if TYPE_CHECKING:
    import torch

    from rowii.tfc.model import TfcModel

_TFC_SAMPLE_RATE_HZ = 8000
"""Fixed input rate the TF-C encoder was (and will be) pretrained at -- every
window is resampled to exactly this many samples before reaching the model,
regardless of the plant's own mic/vib sample rate (`_resample_to_8khz`). Also
the expected value of a loaded checkpoint's `cfg.sample_rate_hz`/`cfg.n_samples`
fields (`_validate_checkpoint_geometry`) -- both default to this same 8000."""

_EXPECTED_EMBED_DIM = 128
"""The `TfcConfig.embed_dim` every real checkpoint this project trains keeps
fixed (only `channels` -- the CNN body's width -- legitimately varies between
the full-size and tiny/test architectures; see `TfcConfig`'s docstring).
`_N_FEATURES` is defined FROM this constant (not the other way around) so the
two can never silently drift apart; `_validate_checkpoint_geometry` checks a
loaded checkpoint's `cfg.embed_dim` against this same value."""

_N_FEATURES = 2 * _EXPECTED_EMBED_DIM
"""Width of `TfcFeaturizer.transform`'s output: `h_t ⊕ h_f`, each
`_EXPECTED_EMBED_DIM`-wide -- see `TfcFeaturizer.feature_names`."""

_CHUNK_SIZE = 512
"""Windows per forward pass through the frozen model (`TfcFeaturizer`'s real
path) -- bounds peak memory for a large burst file's worth of windows,
mirroring why `BeatsFeaturizer.transform` batches at all (this repo's own
1-s-window bursts rarely exceed a few hundred windows, so this is a safety
cap more than an everyday-active limit)."""

_TORCH_HINT = "TF-C featurizer needs torch: pip install -e '.[beats]'"

_MISSING_CHECKPOINT_MSG = (
    "TfcFeaturizer needs either a checkpoint path or an injected encoder stub "
    "(both were None); pass checkpoint=<Path to a TF-C checkpoint> -- see a "
    "ROWII_TFC_*_CHECKPOINT env var / rowii.config for how callers resolve one "
    "-- or pass encoder=<stub> (tests)."
)


@dataclasses.dataclass(frozen=True)
class TfcConfig:
    """TF-C architecture + pretraining hyperparameters -- torch-free by design
    (a plain dataclass) so it can be read out of a checkpoint's `cfg` field,
    logged, or hashed without ever importing torch. `rowii.tfc.model.TfcModel`
    imports this type FROM this module (not the other way around) to keep the
    package's only import cycle direction fixed: `model.py` -> `wrapper.py`,
    never back.

    Every checkpoint this project actually trains keeps `embed_dim`/`proj_dim`
    at their defaults; only `channels` (and, for fast tests, `temperature`)
    varies between the full-size and tiny/test architectures -- see
    `TfcFeaturizer.feature_names`'s docstring for why that matters.
    """

    sample_rate_hz: int = 8000
    n_samples: int = 8000
    embed_dim: int = 128
    proj_dim: int = 64
    channels: tuple[int, ...] = (32, 64, 128, 128)
    temperature: float = 0.2


class TfcEncoderProtocol(Protocol):
    """Anything that maps a batch of resampled 8 kHz windows to 256-d TF-C
    embeddings. Mirrors `rowii.signals.beats.BeatsEncoderProtocol`'s role:
    lets tests inject a deterministic stub (`tests/test_tfc_wrapper.py`'s
    `_StubEncoder`) without needing the real frozen `TfcModel`, a checkpoint
    file, or torch at all.
    """

    def embed(self, batch_8khz: np.ndarray) -> np.ndarray:
        """`(B, 8000)` float windows -> `(B, 256)` embeddings."""
        ...


def _require_torch() -> None:
    """Raise `RuntimeError` (with the shared install hint) if torch is not
    importable; a no-op otherwise. Callers follow this with their own local
    `import torch` -- mirrors `rowii.anomaly.recon._require_torch`'s
    identical role for the reconstruction scorers.
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(_TORCH_HINT) from e


def _resample_to_8khz(mono: np.ndarray, rate_hz: float) -> np.ndarray:
    """`(B, S)` float64 mono windows at *rate_hz* -> `(B, 8000)` float64 at
    `_TFC_SAMPLE_RATE_HZ`, via `scipy.signal.resample_poly`.

    Two length-normalization steps bracket the actual resampling, both
    needed:

    1. BEFORE resampling, each window is pad/trimmed to exactly
       `round(rate_hz)` samples. `resample_poly(x, up, down)`'s output length
       is `ceil(len(x) * up / down)`; feeding it an input of EXACTLY
       `round(rate_hz)` samples with `down = round(rate_hz)` makes that
       formula collapse to `ceil(up) = up` exactly (8000) for any rate --
       whereas a real window's actual sample count can be off by a few
       samples from the nominal rate (DAQ clock jitter, observed up to a few
       samples/window in this project's real recordings -- see
       `rowii.pipeline._SAMPLE_JITTER_TOLERANCE`), which would otherwise
       leave `resample_poly`'s output length one-off from 8000.
    2. AFTER resampling, the result is pad/trimmed to exactly 8000 samples
       again, defensively: `resample_poly`'s FIR filtering has edge effects
       and `int(round(rate_hz))` truncates fractional rates, so this second
       step is what actually GUARANTEES the invariant every downstream
       consumer (the fixed `n_samples=8000`-shaped model input; a stub
       encoder's `(B, 8000)` contract) depends on, rather than merely making
       it likely.
    """
    target_in = int(round(rate_hz))
    n_samples = mono.shape[1]
    if n_samples < target_in:
        padded = np.pad(mono, ((0, 0), (0, target_in - n_samples)))
    elif n_samples > target_in:
        padded = mono[:, :target_in]
    else:
        padded = mono

    resampled = resample_poly(padded, _TFC_SAMPLE_RATE_HZ, target_in, axis=1)

    n_out = resampled.shape[1]
    if n_out < _TFC_SAMPLE_RATE_HZ:
        resampled = np.pad(resampled, ((0, 0), (0, _TFC_SAMPLE_RATE_HZ - n_out)))
    elif n_out > _TFC_SAMPLE_RATE_HZ:
        resampled = resampled[:, :_TFC_SAMPLE_RATE_HZ]
    return np.ascontiguousarray(resampled, dtype=np.float64)


def _standardize(batch: np.ndarray) -> np.ndarray:
    """Per-window (row) zero-mean/unit-std standardization, `1e-8`-floored
    std to avoid a divide-by-zero on a silent (constant) window -- the same
    floor convention as `rowii.tfc.model.freq_view`'s magnitude-spectrum
    standardization."""
    mean = batch.mean(axis=1, keepdims=True)
    std = np.clip(batch.std(axis=1, keepdims=True), 1e-8, None)
    return (batch - mean) / std


def _validate_checkpoint_geometry(cfg: TfcConfig) -> None:
    """Guard `load_tfc_model` against a checkpoint whose `cfg` does not match
    this module's HARDCODED assumptions:
    `_N_FEATURES`/`_TFC_SAMPLE_RATE_HZ` are fixed constants that
    `TfcFeaturizer` (both `feature_names()` and `_resample_to_8khz`) relies
    on unconditionally, regardless of which checkpoint is actually loaded --
    ONLY `cfg.channels` (the CNN body's width) is free to vary between the
    full-size and tiny/test architectures; `embed_dim`, `sample_rate_hz`, and
    `n_samples` must all match their expected values, or a checkpoint trained
    with different ones would silently mis-shape (or semantically
    mis-resample) every downstream embedding instead of failing at load
    time -- exactly the failure mode this check exists to turn into a loud
    `ValueError` instead.

    Args:
        cfg: The `TfcConfig` rebuilt from a checkpoint's own `cfg` field,
            checked BEFORE it is used to construct a `TfcModel`.

    Raises:
        ValueError: naming, for every mismatching field, BOTH the expected
            and the checkpoint's actual value.
    """
    mismatches = []
    if cfg.embed_dim != _EXPECTED_EMBED_DIM:
        mismatches.append(f"embed_dim: expected {_EXPECTED_EMBED_DIM}, got {cfg.embed_dim}")
    if cfg.sample_rate_hz != _TFC_SAMPLE_RATE_HZ:
        mismatches.append(
            f"sample_rate_hz: expected {_TFC_SAMPLE_RATE_HZ}, got {cfg.sample_rate_hz}"
        )
    if cfg.n_samples != _TFC_SAMPLE_RATE_HZ:
        mismatches.append(f"n_samples: expected {_TFC_SAMPLE_RATE_HZ}, got {cfg.n_samples}")
    if mismatches:
        raise ValueError(
            "TF-C checkpoint cfg does not match TfcFeaturizer's hardcoded assumptions "
            f"({'; '.join(mismatches)}) -- this checkpoint cannot be loaded through "
            "load_tfc_model/TfcFeaturizer without silently mis-shaping features "
            "(only cfg.channels may legitimately differ from a checkpoint's defaults)"
        )


def load_tfc_model(checkpoint: Path, device: torch.device) -> TfcModel:
    """Load a TF-C checkpoint onto *device*, in eval mode.

    Args:
        checkpoint: Path to a `.pt` file containing
            `{"cfg": dataclasses.asdict(TfcConfig(...)), "model": state_dict,
            "corpus_manifest_sha256": str, "epochs": int}` -- the format
            fixed HERE (Task 1) and reused, unmodified, by the later
            pretraining script that actually writes one.
        device: Torch device to place the model on.

    Returns:
        A `TfcModel` (`rowii.tfc.model`, imported lazily here) rebuilt from
        the checkpoint's own `cfg`, `.eval()`'d, with weights loaded
        STRICTLY (any key mismatch between the checkpoint and the freshly
        constructed model is a hard error, matching
        `rowii.signals.beats_model.load_beats_model`'s convention of never
        using `strict=False` as an escape hatch).

    Raises:
        FileNotFoundError: if *checkpoint* does not exist.
        ValueError: if the checkpoint's `cfg.embed_dim`/`sample_rate_hz`/
            `n_samples` do not match this module's hardcoded assumptions
            (`_validate_checkpoint_geometry`).
    """
    import torch

    from rowii.tfc.model import TfcModel

    if not checkpoint.exists():
        raise FileNotFoundError(f"TF-C checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg_dict = dict(state["cfg"])
    # dataclasses.asdict() itself preserves tuple-typed fields as tuples, but a
    # checkpoint's cfg may have round-tripped through something that doesn't
    # (e.g. JSON) before reaching here -- coercing defensively costs nothing
    # when it was already a tuple.
    cfg_dict["channels"] = tuple(cfg_dict["channels"])
    cfg = TfcConfig(**cfg_dict)
    _validate_checkpoint_geometry(cfg)

    model = TfcModel(cfg)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


class _RealTfcEncoder:
    """Wraps the loaded (frozen) `TfcModel` as a `TfcEncoderProtocol`:
    standardize -> chunk -> forward under `no_grad` -> `h_t ⊕ h_f`.

    Mirrors `rowii.signals.beats._RealBeatsEncoder`'s role -- the adapter
    that lets `TfcFeaturizer.transform`'s body stay identical whether it
    ends up calling a test stub or this real encoder.
    """

    def __init__(self, model: TfcModel, device: torch.device) -> None:
        self._model = model
        self._device = device

    def embed(self, batch_8khz: np.ndarray) -> np.ndarray:
        """`(B, 8000)` resampled windows -> `(B, 256)` float64 `h_t ⊕ h_f`
        embeddings (pre-projection -- NOT `z_t`/`z_f`, the 64-d contrastive
        projections that only matter for the pretraining loss)."""
        import torch

        from rowii.tfc.model import freq_view

        standardized = _standardize(batch_8khz)
        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, standardized.shape[0], _CHUNK_SIZE):
                chunk = standardized[start : start + _CHUNK_SIZE]
                x_time = torch.from_numpy(chunk.astype(np.float32)).to(self._device)
                x_freq = freq_view(x_time)
                h_t, h_f, _z_t, _z_f = self._model(x_time, x_freq)
                embeddings.append(torch.cat([h_t, h_f], dim=1).cpu().numpy())
        return np.concatenate(embeddings, axis=0).astype(np.float64)


class TfcFeaturizer:
    """Frozen-TF-C embedding featurizer: `(B, S)`/`(B, S, C)` windows ->
    `(B, 256)` embeddings (spec D1/D4).

    Mirrors `rowii.signals.beats.BeatsFeaturizer`'s stub-injectable design
    (an injected `encoder` short-circuits the real, torch-dependent path
    entirely) but differs from it in one respect: construction NEVER raises
    and NEVER imports torch, even when *checkpoint* is a real path and no
    *encoder* is given -- `BeatsFeaturizer.__init__` loads its model eagerly
    because its `checkpoint` parameter is required (never `None`); TF-C's is
    optional (`None` is valid alongside an injected *encoder*), so loading is
    deferred to the first `transform()` call instead and cached afterwards
    (`self._real_encoder`). This is also what keeps `rowii.tfc.wrapper`
    importable and every `TfcFeaturizer` constructible without the optional
    `[beats]` extra: only calling `transform()` on the real (non-stub) path
    actually needs torch.
    """

    name: str = "tfc"

    def __init__(
        self, checkpoint: Path | None, encoder: TfcEncoderProtocol | None = None
    ) -> None:
        """Args:
        checkpoint: Path to a Task-1-format TF-C checkpoint
            (`load_tfc_model`'s docstring). Ignored if *encoder* is given.
            May be `None` if *encoder* is given (tests) -- `None` with no
            *encoder* is only an error once `transform()` is actually called
            (see class docstring), never at construction.
        encoder: Injected `TfcEncoderProtocol` (e.g. a test stub). If `None`,
            `transform()` lazily loads and caches the real frozen model from
            *checkpoint*, via `load_tfc_model`, on first use.
        """
        self._checkpoint = checkpoint
        self._encoder = encoder
        self._real_encoder: _RealTfcEncoder | None = None

    def feature_names(self) -> list[str]:
        """Fixed `["tfc_e0", ..., "tfc_e255"]`.

        Unlike `BeatsFeaturizer.feature_names()`, this width is never
        discovered from a stub/model at runtime, and calling it before
        `transform()` never raises: every TF-C checkpoint this project
        trains keeps `TfcConfig.embed_dim` at its default (128), so
        `h_t ⊕ h_f` is always exactly 256-wide regardless of which
        checkpoint is loaded (only the CNN body's `channels` -- never
        `embed_dim` -- varies between the full-size and tiny/test configs;
        see `TfcConfig`).
        """
        return [f"tfc_e{i}" for i in range(_N_FEATURES)]

    def transform(self, stack: np.ndarray, rate_hz: float) -> np.ndarray:
        """`(B, S)` or `(B, S, C)` windows -> `(B, 256)` float64 TF-C
        embeddings.

        Pipeline: mono-mix over channels (mean, `BeatsFeaturizer`'s rule) ->
        `_resample_to_8khz` -> `self._encoder.embed(...)` if an encoder stub
        was injected (short-circuits everything torch-related below) ->
        otherwise the real path: lazily load+cache the frozen model
        (`_RealTfcEncoder`) from *checkpoint*, then delegate to it (per-window
        standardize -> chunked forward pass -> `h_t ⊕ h_f`).

        Raises:
            ValueError: *stack* is neither 2-D nor 3-D, or neither *encoder*
                nor *checkpoint* was given.
            RuntimeError: *checkpoint* was given (no *encoder*) but torch is
                not installed.
        """
        mono = stack.mean(axis=2) if stack.ndim == 3 else stack
        if mono.ndim != 2:
            raise ValueError(
                f"TfcFeaturizer.transform expects (B, S) or (B, S, C) windows, "
                f"got shape {stack.shape}"
            )
        batch_8khz = _resample_to_8khz(np.asarray(mono, dtype=np.float64), rate_hz)

        if self._encoder is not None:
            return np.asarray(self._encoder.embed(batch_8khz), dtype=np.float64)

        return np.asarray(self._resolve_real_encoder().embed(batch_8khz), dtype=np.float64)

    def _resolve_real_encoder(self) -> _RealTfcEncoder:
        """Lazily load (once) and cache the real frozen-model encoder. Only
        reached from `transform()` when no *encoder* stub was injected."""
        if self._real_encoder is not None:
            return self._real_encoder
        if self._checkpoint is None:
            raise ValueError(_MISSING_CHECKPOINT_MSG)

        _require_torch()

        from rowii.signals.beats import best_device

        device = best_device()
        model = load_tfc_model(self._checkpoint, device)
        self._real_encoder = _RealTfcEncoder(model, device)
        return self._real_encoder
