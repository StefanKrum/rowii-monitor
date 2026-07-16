"""Cross-attention fusion-head wrapper (Step-2 package-5 spec D8, Task 6): the
third fusion level -- a lightweight cross-attention head on FROZEN per-branch
features (`audio-beats`' own 768-d embeddings, the `fusion` cache's own
vibration-branch columns), trained CLIP-style (audio and vibration views of the
SAME window as the positive pair, `scripts/train_xattn.py`) and scored via kNN
on the joint embedding at inference (`--xattn-fusion`, `scripts/run_step2.py`).

Torch-free story, mirroring `rowii.tfc.wrapper`'s/`rowii.adapt.student`'s
identical split: this module (`wrapper.py`) never imports `torch` at module
level, so importing `XattnConfig`, or CALLING `load_xattn_head`/
`joint_embeddings` (whose bodies both `import torch` lazily), never requires the
optional `[beats]` extra merely to be IMPORTED -- only actually LOADING a
checkpoint or running a forward pass does. `rowii.fusionx.model` is the one
module in this package that imports torch at module level (the
`_recon_models.py`/`tfc/model.py`/`_student_model.py` precedent), and is
therefore imported ONLY lazily, from inside this module's own functions, never
at `wrapper.py`'s own top level, and never at this package's `__init__.py`.

`XattnConfig` lives HERE rather than in `model.py` precisely so it stays
importable without torch -- the SAME one-directional import convention
`rowii.tfc.model`/`rowii.adapt._student_model` use for `TfcConfig`/
`StudentConfig` (`rowii.fusionx.model.XattnHead` imports this type FROM this
module, never the reverse).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from rowii.fusionx.model import XattnHead

_EXPECTED_AUDIO_DIM = 768
"""`XattnConfig.audio_dim` every real checkpoint this project trains keeps fixed
-- the frozen BEATs teacher's own embedding width
(`rowii.signals.beats_model.BEATS_EMBED_DIM`), which is what `joint_embeddings`'
*audio* argument is ALWAYS fed from in practice (`audio-beats` cache
embeddings). `_validate_checkpoint_geometry` checks a loaded checkpoint's
`cfg.audio_dim` against this same value -- see that function's own docstring
for why this is the ONE field worth an explicit fail-fast guard (unlike
`lift_dim`/`heads`/`out_dim`, which have no external forcing function and are
trusted to match via `load_state_dict(strict=True)`'s own natural shape
check)."""

_TORCH_HINT = "Cross-attention fusion head needs torch: pip install -e '.[beats]'"

_CHUNK_SIZE = 512
"""Window pairs per forward pass through the frozen head (`joint_embeddings`) --
bounds peak memory for a large window set, mirroring
`rowii.tfc.wrapper.TfcFeaturizer`'s/`rowii.adapt.student.StudentFeaturizer`'s own
`_CHUNK_SIZE` (this repo's own per-run window counts rarely exceed a few
thousand at 1-s windows, so this is a safety cap more than an everyday-active
limit)."""


@dataclasses.dataclass(frozen=True)
class XattnConfig:
    """Cross-attention fusion head architecture (Step-2 package-5 spec D8, Task
    6) -- torch-free by design (a plain dataclass), mirroring
    `rowii.tfc.wrapper.TfcConfig`/`rowii.adapt.student.StudentConfig`'s
    identical role: readable out of a checkpoint's `cfg` field, logged, or
    hashed without ever importing torch.

    Every checkpoint this project trains keeps EVERY field at its default --
    unlike `TfcConfig.channels`/`StudentConfig.channels`, this config has no
    "tiny test architecture" axis at all (`XattnHead` is already small: one
    `nn.MultiheadAttention` plus two lift `Linear`s and one output `Linear`),
    so tests construct `XattnHead` with the plain default `XattnConfig()`
    directly rather than a scaled-down variant.

    `lift_dim` is the ONE shared projection width both branches are lifted to
    BEFORE cross-attention (`rowii.fusionx.model.XattnHead`'s own docstring
    has the full architecture) -- named for what it IS (a single value shared
    by both the audio and the vibration lift), not "vib_dim_lift" (an earlier,
    more confusing name for the identical concept, since the audio branch is
    lifted to this same width too).

    `vib_in_dim` -- the OTHER width `XattnHead.__init__` needs (the fusion
    cache's own vibration-branch column count, `rowii.anomaly.fusion.
    split_branch_columns`) -- is deliberately NOT a field here: it varies per
    TRAINING RUN (however many vibration feature columns that run's `fusion`
    cache happened to have), so it travels as its OWN top-level checkpoint key
    (`"vib_dim"`, `load_xattn_head`'s docstring) rather than living inside this
    otherwise-constant config.
    """

    audio_dim: int = 768
    lift_dim: int = 128
    heads: int = 4
    out_dim: int = 128
    temperature: float = 0.07


def _require_torch() -> None:
    """Raise `RuntimeError` (with the shared install hint) if torch is not
    importable; a no-op otherwise. Callers follow this with their own local
    `import torch` -- mirrors `rowii.tfc.wrapper._require_torch`'s/
    `rowii.adapt.student._require_torch`'s identical role.
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(_TORCH_HINT) from e


def _validate_checkpoint_geometry(cfg: XattnConfig) -> None:
    """Guard `load_xattn_head` against a checkpoint whose `cfg.audio_dim` does
    not match this module's hardcoded assumption (mirrors `rowii.tfc.wrapper.
    _validate_checkpoint_geometry`'s/`rowii.adapt.student.
    _validate_checkpoint_geometry`'s identical shape, scoped to the one field
    that actually needs it here): `joint_embeddings`' *audio* argument is
    ALWAYS real `audio-beats` cache embeddings, unconditionally 768-wide
    (`rowii.signals.beats_model.BEATS_EMBED_DIM`) -- a checkpoint trained with
    a different `audio_dim` would otherwise load FINE (its own stored
    `audio_lift` weights are self-consistent with its own wrong `cfg`) and
    only fail later, cryptically, as a `torch` matmul shape error the first
    time a real 768-d audio embedding is actually fed through it, instead of
    failing loudly and clearly at load time.

    `lift_dim`/`heads`/`out_dim` are deliberately NOT pinned here (unlike
    `audio_dim`, none of them has an external forcing function the way
    `audio_dim` is forced to equal the teacher's own embedding width) -- a
    mismatch on any of those would already be caught by
    `XattnHead.load_state_dict(strict=True)`'s own natural shape check,
    exactly as for every OTHER field `rowii.tfc.wrapper`'s/`rowii.adapt.
    student`'s own geometry guards leave unpinned.

    Args:
        cfg: The `XattnConfig` rebuilt from a checkpoint's own `cfg` field,
            checked BEFORE it is used to construct an `XattnHead`.

    Raises:
        ValueError: naming both the expected and the checkpoint's actual
            `audio_dim` value.
    """
    if cfg.audio_dim != _EXPECTED_AUDIO_DIM:
        raise ValueError(
            "xattn checkpoint cfg does not match load_xattn_head's hardcoded "
            f"assumption (audio_dim: expected {_EXPECTED_AUDIO_DIM}, got "
            f"{cfg.audio_dim}) -- this checkpoint cannot be loaded through "
            "load_xattn_head without silently mis-shaping the audio side of every "
            "downstream joint embedding (only cfg.lift_dim/heads/out_dim may "
            "legitimately differ from a checkpoint's defaults, caught instead by "
            "load_state_dict's own strict shape check)"
        )


def load_xattn_head(checkpoint: Path, device: torch.device) -> XattnHead:
    """Load a cross-attention fusion-head checkpoint onto *device*, in eval
    mode.

    Args:
        checkpoint: Path to a `.pt` file containing `{"cfg":
            dataclasses.asdict(XattnConfig(...)), "model": state_dict, "run":
            str, "vib_dim": int, "epochs": int}` -- the format fixed HERE
            (Task 6) and written, unmodified, by `scripts/train_xattn.py`.
        device: Torch device to place the model on.

    Returns:
        An `XattnHead` (`rowii.fusionx.model`, imported lazily here) rebuilt
        from the checkpoint's own `cfg` + `vib_dim`, `.eval()`'d, with weights
        loaded STRICTLY (mirrors `load_tfc_model`'s/`load_student_model`'s
        convention of never using `strict=False` as an escape hatch).

    Raises:
        FileNotFoundError: if *checkpoint* does not exist.
        ValueError: if the checkpoint's `cfg.audio_dim` does not match this
            module's hardcoded assumption (`_validate_checkpoint_geometry`).
    """
    import torch

    from rowii.fusionx.model import XattnHead

    if not checkpoint.exists():
        raise FileNotFoundError(f"xattn checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = XattnConfig(**dict(state["cfg"]))
    _validate_checkpoint_geometry(cfg)

    model = XattnHead(cfg, vib_in_dim=int(state["vib_dim"]))
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def joint_embeddings(
    head: XattnHead, audio: np.ndarray, vib: np.ndarray, device: torch.device
) -> np.ndarray:
    """Batch *audio*/*vib* window pairs through the frozen *head* under
    `torch.no_grad()` -- the inference-time counterpart of
    `rowii.tfc.wrapper._RealTfcEncoder.embed`/`rowii.adapt.student.
    _RealStudentEncoder.embed`, kept as a plain module-level function (not a
    class wrapping a cached encoder) since `--xattn-fusion`'s own caller
    (`scripts/run_step2.py`) always has a REAL, already-loaded *head* by the
    time it calls this -- there is no stub-injection/lazy-load story to
    preserve here the way `TfcFeaturizer`/`StudentFeaturizer` need one.

    Args:
        head: A loaded (`load_xattn_head`) or freshly constructed `XattnHead`,
            already on *device*.
        audio: `(N, cfg.audio_dim)` float array of audio-beats embeddings.
        vib: `(N, vib_in_dim)` float array of fusion-cache vibration-branch
            columns, row-aligned with *audio* (the SAME window --
            `scripts/run_step2.py`'s/`scripts/train_xattn.py`'s own
            grid-alignment guard is what establishes this alignment).
        device: Torch device to run the forward pass on (must match *head*'s
            own device).

    Returns:
        `(N, cfg.out_dim)` float64 joint embeddings, `N == 0` yielding an
        empty `(0, cfg.out_dim)` array rather than raising.
    """
    import torch

    if audio.shape[0] == 0:
        return np.empty((0, head.cfg.out_dim), dtype=np.float64)

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, audio.shape[0], _CHUNK_SIZE):
            a_chunk = torch.from_numpy(
                np.asarray(audio[start : start + _CHUNK_SIZE], dtype=np.float32)
            ).to(device)
            v_chunk = torch.from_numpy(
                np.asarray(vib[start : start + _CHUNK_SIZE], dtype=np.float32)
            ).to(device)
            out = head(a_chunk, v_chunk)
            embeddings.append(out.cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float64)
