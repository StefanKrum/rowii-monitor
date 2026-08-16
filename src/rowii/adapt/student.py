"""Distilled BEATs-student audio featurizer (compactness pair): the compact-CNN counterpart to
`rowii.signals.beats.BeatsFeaturizer`, trained (`scripts/distill_beats.py`) to
regress the frozen BEATs teacher's 768-d embedding from the `logmel` variant's
own input patches -- zero teacher/extraction compute (both sides come from
ALREADY-CACHED `results/cache/<run>--{audio-beats,logmel}.npz` files).

Torch-free story, mirroring `rowii.tfc.wrapper`'s (and, one level down,
`rowii.signals.beats`/`rowii.signals.beats_model`'s) split: this module
(`student.py`) never imports `torch` at module level, so importing or
CONSTRUCTING `StudentFeaturizer` never requires the optional `[beats]` extra --
only actually calling `transform()` on the real (non-stub-encoder) path does.
`rowii.adapt._student_model` is the one module in this pair that imports torch
at module level (the `_recon_models.py`/`tfc/model.py` precedent), and is
therefore imported ONLY lazily, from inside this module's functions/methods,
never at `student.py`'s own top level. `StudentConfig` lives HERE rather than
in `_student_model.py` precisely so it stays importable without torch: callers
that only need to describe a student architecture (e.g. to read a checkpoint's
`cfg` field) never pull in torch as a side effect -- `_student_model.py`
imports `StudentConfig` FROM this module (never the reverse), the SAME
one-directional convention `rowii.tfc.model` uses for `TfcConfig`.

`StudentFeaturizer.transform` mirrors the shape contract of the other
featurizers (`rowii.signals.beats.BeatsFeaturizer`, `rowii.tfc.wrapper.
TfcFeaturizer`, `rowii.signals.logmel.LogmelFeaturizer`): `(B, S)` or `(B, S,
C)` windows in, `(B, 768)` float64 features out. Internally: the SAME
`rowii.signals.logmel.LogmelFeaturizer` front end the `logmel` variant itself
uses (mono-mix + per-window log-mel patch, flattened frame-major) -> [stub
encoder: `encoder.embed(logmel_flat)`, done] OR [real path: the frozen
`_StudentNet` under `torch.no_grad()`, which reshapes the flattened patch back
to a `(1, n_mels, n_frames)` image internally]. The real path is geometry-
LOCKED to the plant's own 50 kHz / 1-s window default (the SAME geometry
`LogmelFeaturizer`'s own docstring pins to exactly 49 frames x 64 mels = 3136):
unlike `TfcFeaturizer` (which explicitly resamples every window to a fixed
8 kHz before its encoder), the student's compactness pair never resamples --
its own framing is "compact CNN on the logmel variant's patches", i.e. this
project's own plant geometry, not a generic geometry-invariant design. A stub
encoder bypasses this constraint entirely (it can do anything with whatever
width `logmel_flat` happens to have), which is what keeps this module's own
tests torch-free regardless of window geometry.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from rowii.signals.logmel import LogmelFeaturizer

if TYPE_CHECKING:
    import torch

    from rowii.adapt._student_model import _StudentNet

_EXPECTED_N_MELS = 64
"""`StudentConfig.n_mels` every real checkpoint this project trains keeps fixed
-- the plant's own log-mel geometry (`rowii.signals.logmel.LogmelFeaturizer`'s
default `n_mels`). `_validate_checkpoint_geometry` checks a loaded checkpoint's
`cfg.n_mels` against this same value."""

_EXPECTED_N_FRAMES = 49
"""`StudentConfig.n_frames` every real checkpoint this project trains keeps
fixed -- the plant's 50 kHz / 1-s window geometry's own frame count (see
`LogmelFeaturizer`'s docstring: `frame_s=0.025`/`hop_s=0.020` at 50 kHz over a
1-s window gives exactly 49 frames)."""

_EXPECTED_OUT_DIM = 768
"""`StudentConfig.out_dim` every real checkpoint this project trains keeps
fixed -- the frozen BEATs teacher's own embedding width
(`rowii.signals.beats_model.BEATS_EMBED_DIM`), what the student is distilled to
regress. `feature_names()` is defined FROM this constant (not the other way
around) so the two can never silently drift apart."""

_CHUNK_SIZE = 512
"""Windows per forward pass through the real `_StudentNet` (`StudentFeaturizer`'s
real path) -- bounds peak memory for a large burst file's worth of windows,
mirroring `rowii.tfc.wrapper.TfcFeaturizer`'s own `_CHUNK_SIZE` (this repo's own
1-s-window bursts rarely exceed a few hundred windows, so this is a safety cap
more than an everyday-active limit)."""

_TORCH_HINT = "Student featurizer needs torch: pip install -e '.[beats]'"

_MISSING_CHECKPOINT_MSG = (
    "StudentFeaturizer needs either a checkpoint path or an injected encoder stub "
    "(both were None); pass checkpoint=<Path to a student checkpoint, e.g. "
    "models/adapted/student_<run>.pt> -- see ROWII_STUDENT_CHECKPOINT / rowii.config "
    "for how callers resolve one -- or pass encoder=<stub> (tests)."
)


@dataclasses.dataclass(frozen=True)
class StudentConfig:
    """Student-CNN architecture -- torch-free by
    design (a plain dataclass) so it can be read out of a checkpoint's `cfg`
    field, logged, or hashed without ever importing torch.
    `rowii.adapt._student_model._StudentNet` imports this type FROM this
    module (not the other way around) to keep the package's only import
    cycle direction fixed: `_student_model.py` -> `student.py`, never back.

    Every checkpoint this project actually trains keeps `n_mels`/`n_frames`/
    `out_dim` at their defaults; only `channels` (and, for fast tests, its own
    length/width) varies between the full-size and tiny/test architectures --
    see `StudentFeaturizer.feature_names`'s docstring for why that matters.
    """

    n_mels: int = 64
    n_frames: int = 49
    out_dim: int = 768
    channels: tuple[int, ...] = (32, 64, 128)


class StudentEncoderProtocol(Protocol):
    """Anything that maps a batch of flattened log-mel patches to 768-d student
    embeddings. Mirrors `rowii.tfc.wrapper.TfcEncoderProtocol`'s role: lets
    tests inject a deterministic stub (`tests/test_student.py`'s
    `_StubEncoder`) without needing the real `_StudentNet`, a checkpoint file,
    or torch at all.
    """

    def embed(self, logmel_flat: np.ndarray) -> np.ndarray:
        """`(B, n_frames * n_mels)` float64 flattened log-mel patches -> `(B,
        out_dim)` embeddings."""
        ...


def _require_torch() -> None:
    """Raise `RuntimeError` (with the shared install hint) if torch is not
    importable; a no-op otherwise. Callers follow this with their own local
    `import torch` -- mirrors `rowii.tfc.wrapper._require_torch`'s identical
    role.
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(_TORCH_HINT) from e


def _validate_checkpoint_geometry(cfg: StudentConfig) -> None:
    """Guard `load_student_model` against a checkpoint whose `cfg` does not
    match this module's HARDCODED assumptions (mirrors `rowii.tfc.wrapper.
    _validate_checkpoint_geometry`): `_EXPECTED_N_MELS`/`_EXPECTED_N_FRAMES`/
    `_EXPECTED_OUT_DIM` are fixed constants that `StudentFeaturizer` (both
    `feature_names()` and the log-mel front end's own plant-geometry
    assumption) relies on unconditionally, regardless of which checkpoint is
    actually loaded -- ONLY `cfg.channels` (the CNN body's width) is free to
    vary between the full-size and tiny/test architectures; `n_mels`,
    `n_frames`, and `out_dim` must all match their expected values, or a
    checkpoint trained with different ones would silently mis-shape (or
    semantically mis-reshape) every downstream embedding instead of failing at
    load time -- exactly the failure mode this check exists to turn into a
    loud `ValueError` instead.

    Args:
        cfg: The `StudentConfig` rebuilt from a checkpoint's own `cfg` field,
            checked BEFORE it is used to construct a `_StudentNet`.

    Raises:
        ValueError: naming, for every mismatching field, BOTH the expected and
            the checkpoint's actual value.
    """
    mismatches = []
    if cfg.n_mels != _EXPECTED_N_MELS:
        mismatches.append(f"n_mels: expected {_EXPECTED_N_MELS}, got {cfg.n_mels}")
    if cfg.n_frames != _EXPECTED_N_FRAMES:
        mismatches.append(f"n_frames: expected {_EXPECTED_N_FRAMES}, got {cfg.n_frames}")
    if cfg.out_dim != _EXPECTED_OUT_DIM:
        mismatches.append(f"out_dim: expected {_EXPECTED_OUT_DIM}, got {cfg.out_dim}")
    if mismatches:
        raise ValueError(
            "student checkpoint cfg does not match StudentFeaturizer's hardcoded "
            f"assumptions ({'; '.join(mismatches)}) -- this checkpoint cannot be loaded "
            "through load_student_model/StudentFeaturizer without silently mis-shaping "
            "features (only cfg.channels may legitimately differ from a checkpoint's "
            "defaults)"
        )


def load_student_model(checkpoint: Path, device: torch.device) -> _StudentNet:
    """Load a student checkpoint onto *device*, in eval mode.

    Args:
        checkpoint: Path to a `.pt` file containing `{"cfg":
            dataclasses.asdict(StudentConfig(...)), "model": state_dict,
            "teacher_variant": str, "run": str, "epochs": int}` -- the format
            fixed HERE and written, unmodified, by `scripts/distill_beats.py`.
        device: Torch device to place the model on.

    Returns:
        A `_StudentNet` (`rowii.adapt._student_model`, imported lazily here)
        rebuilt from the checkpoint's own `cfg`, `.eval()`'d, with weights
        loaded STRICTLY (any key mismatch between the checkpoint and the
        freshly constructed model is a hard error, matching `rowii.tfc.wrapper.
        load_tfc_model`'s/`rowii.signals.beats_model.load_beats_model`'s
        convention of never using `strict=False` as an escape hatch).

    Raises:
        FileNotFoundError: if *checkpoint* does not exist.
        ValueError: if the checkpoint's `cfg.n_mels`/`n_frames`/`out_dim` do
            not match this module's hardcoded assumptions
            (`_validate_checkpoint_geometry`).
    """
    import torch

    from rowii.adapt._student_model import _StudentNet

    if not checkpoint.exists():
        raise FileNotFoundError(f"student checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg_dict = dict(state["cfg"])
    # dataclasses.asdict() itself preserves tuple-typed fields as tuples, but a
    # checkpoint's cfg may have round-tripped through something that doesn't
    # (e.g. JSON) before reaching here -- coercing defensively costs nothing
    # when it was already a tuple (mirrors rowii.tfc.wrapper.load_tfc_model).
    cfg_dict["channels"] = tuple(cfg_dict["channels"])
    cfg = StudentConfig(**cfg_dict)
    _validate_checkpoint_geometry(cfg)

    model = _StudentNet(cfg)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


class _RealStudentEncoder:
    """Wraps the loaded (trained) `_StudentNet` as a `StudentEncoderProtocol`:
    chunk -> forward under `no_grad` -> numpy.

    Mirrors `rowii.tfc.wrapper._RealTfcEncoder`'s role -- the adapter that lets
    `StudentFeaturizer.transform`'s body stay identical whether it ends up
    calling a test stub or this real encoder.
    """

    def __init__(self, model: _StudentNet, device: torch.device) -> None:
        self._model = model
        self._device = device

    def embed(self, logmel_flat: np.ndarray) -> np.ndarray:
        """`(B, n_frames * n_mels)` flattened log-mel patches -> `(B,
        out_dim)` float64 embeddings."""
        import torch

        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, logmel_flat.shape[0], _CHUNK_SIZE):
                chunk = logmel_flat[start : start + _CHUNK_SIZE]
                x = torch.from_numpy(chunk.astype(np.float32)).to(self._device)
                out = self._model(x)
                embeddings.append(out.cpu().numpy())
        return np.concatenate(embeddings, axis=0).astype(np.float64)


class StudentFeaturizer:
    """Distilled-student embedding featurizer: `(B, S)`/`(B, S, C)` windows ->
    `(B, 768)` embeddings.

    Mirrors `rowii.tfc.wrapper.TfcFeaturizer`'s stub-injectable design (an
    injected `encoder` short-circuits the real, torch-dependent path entirely)
    and its deferred-load story: construction NEVER raises and NEVER imports
    torch, even when *checkpoint* is a real path and no *encoder* is given --
    loading the trained model is deferred to the first `transform()` call and
    cached afterwards (`self._real_encoder`), which is what lets this whole
    module (not just its imports) stay usable without the `[beats]` extra
    until a real embedding is actually requested.
    """

    name: str = "student"

    def __init__(
        self, checkpoint: Path | None, encoder: StudentEncoderProtocol | None = None
    ) -> None:
        """Args:
        checkpoint: Path to a `load_student_model`-format checkpoint (that
            function's docstring). Ignored if *encoder* is given. May be
            `None` if *encoder* is given (tests) -- `None` with no *encoder*
            is only an error once `transform()` is actually called (see class
            docstring), never at construction.
        encoder: Injected `StudentEncoderProtocol` (e.g. a test stub). If
            `None`, `transform()` lazily loads and caches the real trained
            model from *checkpoint*, via `load_student_model`, on first use.
        """
        self._checkpoint = checkpoint
        self._encoder = encoder
        self._real_encoder: _RealStudentEncoder | None = None
        self._logmel = LogmelFeaturizer()

    def feature_names(self) -> list[str]:
        """Fixed `["student_e0", ..., "student_e767"]`.

        Unlike `BeatsFeaturizer.feature_names()`, this width is never
        discovered from a stub/model at runtime, and calling it before
        `transform()` never raises: every student checkpoint this project
        trains keeps `StudentConfig.out_dim` at its default (768, the frozen
        BEATs teacher's own embedding width), so this is always exactly
        768-wide regardless of which checkpoint is loaded (only the CNN
        body's `channels` -- never `out_dim` -- varies between the full-size
        and tiny/test configs; see `StudentConfig`).
        """
        return [f"student_e{i}" for i in range(_EXPECTED_OUT_DIM)]

    def transform(self, stack: np.ndarray, rate_hz: float) -> np.ndarray:
        """`(B, S)` or `(B, S, C)` windows -> `(B, 768)` float64 student
        embeddings.

        Pipeline: `rowii.signals.logmel.LogmelFeaturizer.transform` (mono-mix
        over channels + per-window flattened log-mel patch, the SAME front
        end the `logmel` variant itself uses) -> `self._encoder.embed(...)`
        if an encoder stub was injected (short-circuits everything
        torch-related below) -- otherwise the real path: lazily load+cache
        the trained model (`_RealStudentEncoder`) from *checkpoint*, then
        delegate to it (chunked forward pass under `no_grad`).

        Raises:
            ValueError: *stack* is not `(B, S)`/`(B, S, C)` (propagated from
                `LogmelFeaturizer.transform`), or neither *encoder* nor
                *checkpoint* was given.
            RuntimeError: *checkpoint* was given (no *encoder*) but torch is
                not installed.
        """
        logmel_flat = self._logmel.transform(stack, rate_hz)

        if self._encoder is not None:
            return np.asarray(self._encoder.embed(logmel_flat), dtype=np.float64)

        return np.asarray(self._resolve_real_encoder().embed(logmel_flat), dtype=np.float64)

    def _resolve_real_encoder(self) -> _RealStudentEncoder:
        """Lazily load (once) and cache the real trained-model encoder. Only
        reached from `transform()` when no *encoder* stub was injected."""
        if self._real_encoder is not None:
            return self._real_encoder
        if self._checkpoint is None:
            raise ValueError(_MISSING_CHECKPOINT_MSG)

        _require_torch()

        from rowii.signals.beats import best_device

        device = best_device()
        model = load_student_model(self._checkpoint, device)
        self._real_encoder = _RealStudentEncoder(model, device)
        return self._real_encoder
