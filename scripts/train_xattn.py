"""Cross-attention fusion-head training CLI (Step-2 package-5 spec D8, plan Task
6): trains `rowii.fusionx.model.XattnHead` CLIP-style on `audio-beats`'
ALREADY-CACHED 768-d embeddings and the `fusion` cache's own vibration-branch
columns (`rowii.anomaly.fusion.split_branch_columns`) -- audio and vibration
views of the SAME window are the ONE positive pair (`rowii.tfc.model.tfc_loss`,
the symmetric InfoNCE this objective needs, reused verbatim -- see `rowii.
fusionx.model`'s own module docstring for the full composite-loss derivation).
Zero extraction compute: this script never runs BEATs or handcrafted feature
extraction itself, never triggers a live `rowii.pipeline.prepare_run` for
either side, and refuses outright (`_load_cache_or_exit`) rather than
triggering one -- the SAME cache-only contract `scripts/distill_beats.py`
already established, duplicated here (not imported) per this project's own
"a script must not depend on a sibling script's internals" rule (`scripts/
warm_cache.py`'s precedent).

Usage: `train_xattn.py --run <name> --epochs 20 --batch-size 256 --lr 1e-3
--seed 7 --out models/adapted/` -> `models/adapted/xattn_<run>.pt` + a sidecar
`<same-stem>.json`. Both caches for *run* must already be warm (`scripts/
warm_cache.py --runs <name> --variants audio-beats fusion`).

## Cache loading + grid alignment

Both caches are loaded via `rowii.pipeline`'s own public cache primitives
(`_cache_npz_path`/`_cache_fingerprint`/`_load_cached_prepared_run` -- the same
cross-module-privates precedent `scripts/analyze_step2.py`/`scripts/warm_cache.
py`/`scripts/distill_beats.py` already set) WITH fingerprint verification: a
missing file or a fingerprint mismatch is a cache MISS, and `_load_cache_or_exit`
refuses outright rather than silently recomputing (naming the exact `warm_cache.
py` invocation to fix it).

`audio-beats`' grid (built from BOTH mic streams' intersection, `rowii.pipeline.
_streams_for_variant`) and `fusion`'s grid (built from all FOUR streams' own
intersection: both mic AND both vibration streams) are not guaranteed to be
byte-identical -- the same physical reason `scripts/run_step2.py`'s own
`--ensemble` guard (`_check_ensemble_grid_alignment`) tolerates a sub-window
offset between a vibration-bearing sweep variant and `logmel` (the vibration
streams start measurably later than the mic streams on real DAQ hardware).
`_check_cache_alignment` mirrors that same tolerance here: `window_ns`/
`n_windows` must match EXACTLY, and the two grids' `t0_ns` must agree within one
window (a warning, not an abort, when nonzero) -- `SystemExit(2)` otherwise,
since a coarser mismatch would make "window index i" refer to different
physical time slots in the two caches.

## Leakage rule (spec D3, reused here)

Training draws on calibration-side windows ONLY: `rowii.anomaly.references.
split_by_segments` on the `fusion` cache's own `segment_ids`/`valid_mask`
(`calibration_frac=0.5`, `seed` -- default 7, the SAME top-level split every
Step-2 sweep draws its own calibration/scoring windows from for this run),
AND-ed with `audio-beats`' own `valid_mask` (so every drawn window's audio side
is finite too -- mirrors `scripts/distill_beats.py`'s own `_select_calibration_
windows`, whose docstring has the full "why AND, not just the anchor side's own
mask" rationale, adapted here: the `fusion` cache is the anchor, `audio-beats`
is the side being AND-ed in). A trained `xattn_<run>.pt` checkpoint therefore
never trains on a window a LATER within-day `--xattn-fusion` sweep for the SAME
run might score against it; the sidecar json restates this caveat (design's
global "adapted/distilled/quantized results always carry their caveat" rule).

## Training

Composite loss (`rowii.fusionx.model`'s own module docstring has the full
derivation): `tfc_loss(lift_a, lift_v, T) + _JOINT_LOSS_WEIGHT * tfc_loss(joint,
lift_a.detach(), T)`. Adam, `--epochs` full passes over the drawn calibration
windows in `--batch-size`-row mini-batches, shuffled each epoch by a
`--seed`-seeded CPU `torch.Generator` -- the SAME established shuffle pattern as
`scripts/pretrain_tfc.py`'s/`scripts/distill_beats.py`'s own training loops.
Deterministic given `--seed` alone (together with `torch.manual_seed(seed)` for
weight init, called BEFORE the model is constructed) -- verified only on CPU
(this project's tests never run torch training on a non-CPU device; the
established MPS/CUDA caveat is `rowii.anomaly.recon`'s own module docstring).

Torch import discipline (plan's Global Constraints): every torch-touching name
here is imported lazily inside the function that needs it, INCLUDING
`discover`/`split_by_segments`, which are a deliberate exception (mirrors
`scripts/distill_beats.py`'s own module docstring: a module-level import
specifically so `tests/test_fusionx.py` can `monkeypatch.setattr(train_xattn,
"discover", ...)`/`monkeypatch.setattr(train_xattn, "split_by_segments", ...)`).
`XattnConfig` (from `rowii.fusionx.wrapper`, torch-free by design) is ALSO a
module-level import for the same reason `distill_beats.py` imports
`StudentConfig` at its own top level: it never pulls in torch merely by being
imported.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rowii.fusionx.model import XattnHead

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.anomaly.fusion import split_branch_columns  # noqa: E402
from rowii.anomaly.references import split_by_segments  # noqa: E402
from rowii.config import Config, load_config  # noqa: E402
from rowii.fusionx.wrapper import XattnConfig  # noqa: E402
from rowii.io.dataset import Run, discover, run_utc_offset_ns  # noqa: E402
from rowii.pipeline import (  # noqa: E402
    PreparedRun,
    _cache_fingerprint,
    _cache_npz_path,
    _load_cached_prepared_run,
    _streams_for_variant,
)

logger = logging.getLogger(__name__)

_TORCH_HINT = "pip install -e '.[beats]'"
_AUDIO_VARIANT = "audio-beats"
_VIB_SOURCE_VARIANT = "fusion"
_CALIBRATION_FRAC = 0.5
"""`split_by_segments`' calibration fraction -- the SAME top-level
calibration/scoring split fraction every Step-2 sweep uses (module docstring's
"Leakage rule" section)."""

_JOINT_LOSS_WEIGHT = 0.5
"""Weight of the joint (post-attention) alignment term in the composite loss
(`rowii.fusionx.model`'s own module docstring has the full derivation) -- HALF
the weight of the primary lift-alignment term, a binding design constant: the
first term (`tfc_loss(lift_a, lift_v, T)`) is the actual CLIP-style objective
that must dominate; this second term is auxiliary, shaping the attention block
+ output projection toward the already-established lift alignment rather than
competing with it on equal footing."""

_LEAKAGE_NOTE = (
    "xattn training trains on calibration-side windows only (rowii.anomaly.references."
    "split_by_segments on the fusion cache's own segment_ids/valid_mask, AND-ed with the "
    "audio-beats cache's own valid_mask, calibration_frac=0.5) -- the SAME top-level split "
    "every Step-2 within-day sweep draws its own calibration/scoring windows from for this "
    "run. Any --xattn-fusion result computed from this checkpoint must restate that the "
    "head was trained on this run's calibration side."
)


def _import_torch_or_exit() -> None:
    """xattn training is inherently a torch operation (mirrors `scripts/
    distill_beats.py`'s/`scripts/pretrain_tfc.py`'s own `_import_torch_or_exit`)
    -- this guard runs unconditionally, early in `main()`, right after argument
    parsing."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"train_xattn needs torch ({exc}); {_TORCH_HINT}") from exc


def _resolve_run_or_exit(run_name: str, cfg: Config) -> Run:
    """*run_name*'s `Run`, discovered under `cfg.data_root` -- mirrors `scripts/
    distill_beats.py`'s own identical helper.

    Raises:
        SystemExit: *run_name* was not discovered under `cfg.data_root` (names
            every discovered run).
    """
    index = discover(cfg.data_root)
    by_name = {r.name: r for r in index.runs}
    run = by_name.get(run_name)
    if run is None:
        raise SystemExit(
            f"train_xattn: run {run_name!r} not discovered under {cfg.data_root} "
            f"(available: {sorted(by_name)})"
        )
    return run


def _load_cache_or_exit(run: Run, variant: str, cfg: Config) -> PreparedRun:
    """Cache-ONLY load of *run*'s (run, variant) `PreparedRun`, refusing outright
    on a miss rather than falling through to a live `rowii.pipeline.prepare_run`
    extraction (module docstring's "Cache loading + grid alignment" section) --
    mirrors `scripts/distill_beats.py`'s identical helper verbatim, duplicated
    (not imported) per this project's own "a script must not depend on a
    sibling script's internals" rule.

    Raises:
        SystemExit: no cache file exists at the expected path, or its stored
            fingerprint does not match the current (run, variant, cfg) -- names
            the exact `scripts/warm_cache.py` invocation to fix it.
    """
    cache_path = _cache_npz_path(cfg.results_root, run.name, variant)
    fingerprint = _cache_fingerprint(run, variant, cfg)
    streams = _streams_for_variant(variant)
    offset_ns = run_utc_offset_ns(run)
    cached = _load_cached_prepared_run(
        cache_path, fingerprint, run, streams, cfg.window.window_s, offset_ns
    )
    if cached is None:
        raise SystemExit(
            f"train_xattn: no warm cache for {run.name} x {variant} (expected "
            f"{cache_path} with a matching fingerprint) -- run `python scripts/"
            f"warm_cache.py --runs {run.name} --variants {variant}` first; this "
            "script refuses to trigger a from-scratch extraction silently."
        )
    return cached


def _check_cache_alignment(run_name: str, audio: PreparedRun, vib_source: PreparedRun) -> int:
    """Grid-alignment guard between the *audio* (`audio-beats`) and
    *vib_source* (`fusion`) caches -- mirrors `scripts/run_step2.py`'s own
    `--ensemble` guard (`_check_ensemble_grid_alignment`) and `scripts/
    distill_beats.py`'s own `_check_cache_alignment`, adapted (module
    docstring's "Cache loading + grid alignment" section has the full
    rationale): the two caches must be STRUCTURALLY identical (`window_ns`,
    `n_windows` -- exact) and t0-aligned WITHIN ONE WINDOW for a shared window
    INDEX to refer to (near enough) the same physical time slot in both --
    `_select_calibration_windows`/`main` index BOTH caches' feature matrices at
    the SAME window indices.

    Returns:
        The absolute t0 offset in ns between the two grids (`0` when exactly
        aligned; always `< window_ns`).

    Raises:
        SystemExit: code 2, with a clear message on stderr, if `window_ns`/
            `n_windows` differ at all, or the t0 offset is one full window or
            more -- either signals a structural inconsistency training cannot
            safely paper over.
    """
    ag, vg = audio.grid, vib_source.grid
    if ag.window_ns != vg.window_ns or ag.n_windows != vg.n_windows:
        print(
            f"train_xattn: cache grid mismatch for run {run_name!r}: audio-beats "
            f"grid (t0_ns={ag.t0_ns}, window_ns={ag.window_ns}, n_windows={ag.n_windows}) "
            f"!= fusion grid (t0_ns={vg.t0_ns}, window_ns={vg.window_ns}, "
            f"n_windows={vg.n_windows}) -- window_ns and n_windows must be identical for "
            "xattn training to index both caches at the same window positions",
            file=sys.stderr,
        )
        raise SystemExit(2)
    t0_offset_ns = abs(ag.t0_ns - vg.t0_ns)
    if t0_offset_ns >= ag.window_ns:
        print(
            f"train_xattn: audio-beats/fusion caches for run {run_name!r} are "
            f"misaligned by >= one window: |t0 offset| = {t0_offset_ns / 1e6:.1f} ms >= "
            f"window {ag.window_ns / 1e6:.0f} ms (audio-beats t0_ns={ag.t0_ns}, fusion "
            f"t0_ns={vg.t0_ns}) -- window index i would refer to non-overlapping time "
            "slots in the two caches",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if t0_offset_ns > 0:
        logger.warning(
            "train_xattn: audio-beats/fusion caches for run %r are offset by %.1f ms "
            "on a %.0f ms window (minimum per-window overlap %.1f%%) -- proceeding: the "
            "head trains on a vibration window shifted by this sub-window DAQ "
            "stream-start offset relative to its paired audio-beats embedding",
            run_name, t0_offset_ns / 1e6, ag.window_ns / 1e6,
            (1.0 - t0_offset_ns / ag.window_ns) * 100.0,
        )
    return t0_offset_ns


def _select_calibration_windows(
    vib_source: PreparedRun, audio: PreparedRun, *, seed: int
) -> np.ndarray:
    """The leakage-safe calibration-side window indices to train on (spec D3's
    rule, reused here per the module docstring's "Leakage rule" section):
    `split_by_segments` on the `fusion` *vib_source* cache's own `segment_ids`/
    `valid_mask` (`_CALIBRATION_FRAC`, *seed*) -- the SAME top-level
    calibration/scoring split every Step-2 sweep draws for this run, so a
    trained xattn checkpoint never trains on a window a later sweep might score
    against it.

    The valid mask fed to `split_by_segments` is the AND of both caches' own
    `valid_mask` (mirrors `scripts/distill_beats.py`'s own `_select_calibration_
    windows`, whose docstring has the full "why AND, not just the anchor side's
    own mask" rationale): after `_check_cache_alignment` confirms the two grids
    share `window_ns`/`n_windows` and are t0-aligned within one window, window
    index i means (near enough) the same physical second in both -- but a
    window can still be coverage-valid for `fusion`'s own four-stream
    intersection while `audio-beats`' own validity (its own two-mic-stream
    intersection, `rowii.pipeline._streams_for_variant("audio-beats")`) says
    otherwise. Training on such a window would pair a real vibration reading
    with an undefined (NaN) audio embedding -- ANDing the masks keeps every
    drawn window's audio side finite, without weakening the leakage rule
    itself (segment ids AND the scoring-side exclusion still come from the
    `fusion` cache alone, per D3).

    Returns:
        `(N,)` int64 ascending window indices, valid in BOTH caches.
    """
    combined_valid = vib_source.valid_mask & audio.valid_mask
    split = split_by_segments(vib_source.segment_ids, combined_valid, _CALIBRATION_FRAC, seed)
    return split.calibration_windows


def _train_xattn_head(
    audio_inputs: np.ndarray,
    vib_inputs: np.ndarray,
    cfg: XattnConfig,
    vib_dim: int,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[XattnHead, list[float]]:
    """CLIP-style composite-loss training (module docstring's "Training"
    section): Adam, *epochs* full passes over *audio_inputs*/*vib_inputs* in
    *batch_size*-row mini-batches, shuffled each epoch by a *seed*-seeded CPU
    `torch.Generator` -- the SAME established pattern as `scripts/distill_beats.
    py`'s own `_train_student`/`scripts/pretrain_tfc.py`'s own `_train`.
    Deterministic given *seed* alone (together with `torch.manual_seed(seed)`
    for weight init, called BEFORE the model is constructed).

    Per training step: draw a shuffled mini-batch -> lift both branches
    (`model.audio_lift`/`model.vib_lift`, read directly -- module docstring's
    composite-loss section) -> the joint embedding (`model(audio_batch,
    vib_batch)`, which internally recomputes the SAME two lifts -- `XattnHead.
    forward`'s own docstring documents this accepted redundancy) -> `rowii.tfc.
    model.tfc_loss(lift_a, lift_v, cfg.temperature) + _JOINT_LOSS_WEIGHT *
    tfc_loss(joint, lift_a.detach(), cfg.temperature)` -> Adam step.

    Args:
        audio_inputs: `(N, cfg.audio_dim)` float64 audio-beats embeddings
            (calibration-side windows only -- caller's responsibility,
            `_select_calibration_windows`).
        vib_inputs: `(N, vib_dim)` float64 fusion-cache vibration-branch
            columns, row *i* paired with `audio_inputs[i]` (the SAME window).
        cfg: Head architecture (`XattnConfig()` in every real run this script
            performs -- no CLI flag varies it).
        vib_dim: `vib_inputs.shape[1]` -- threaded through explicitly (rather
            than read off the array inside this function) so the caller's own
            `split_branch_columns`-derived width is what actually constructs
            the model, with no possibility of the two silently drifting apart.
        epochs: Full passes over *audio_inputs*/*vib_inputs*.
        batch_size: Rows per gradient step (the last batch of an epoch may be
            smaller, `len(audio_inputs) % batch_size`).
        lr: Adam learning rate.
        seed: Seeds `torch.manual_seed` (weight init) AND the shared shuffle
            generator.

    Returns:
        `(model, epoch_losses)`: *model* on `best_device()`, in `.eval()` mode
        (mirrors `_train_student`'s/`pretrain_tfc._train`'s own postcondition
        -- every downstream `joint_embeddings` call reads it under `torch.
        no_grad()`); *epoch_losses* is one MEAN composite loss per epoch, in
        order, for the caller to log.

    Raises:
        ValueError: *audio_inputs*/*vib_inputs* have different row counts.
    """
    import torch

    from rowii.fusionx.model import XattnHead
    from rowii.signals.beats import best_device
    from rowii.tfc.model import tfc_loss

    if audio_inputs.shape[0] != vib_inputs.shape[0]:
        raise ValueError(
            f"audio_inputs.shape[0] ({audio_inputs.shape[0]}) must equal "
            f"vib_inputs.shape[0] ({vib_inputs.shape[0]})"
        )

    torch.manual_seed(seed)
    device = best_device()
    model = XattnHead(cfg, vib_in_dim=vib_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    audio_all = torch.from_numpy(audio_inputs.astype(np.float32)).to(device)
    vib_all = torch.from_numpy(vib_inputs.astype(np.float32)).to(device)
    n = audio_all.shape[0]
    gen = torch.Generator().manual_seed(seed)

    epoch_losses: list[float] = []
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            audio_batch = audio_all[idx]
            vib_batch = vib_all[idx]
            lift_a = model.audio_lift(audio_batch)
            lift_v = model.vib_lift(vib_batch)
            joint = model(audio_batch, vib_batch)
            loss = tfc_loss(lift_a, lift_v, cfg.temperature) + _JOINT_LOSS_WEIGHT * tfc_loss(
                joint, lift_a.detach(), cfg.temperature
            )
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        mean_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(mean_loss)
        logger.info(
            "train_xattn: epoch %d/%d -- mean loss %.6f", epoch + 1, epochs, mean_loss
        )

    model.eval()
    return model, epoch_losses


def _save_checkpoint(
    path: Path,
    cfg: XattnConfig,
    model: XattnHead,
    vib_dim: int,
    run_name: str,
    epochs: int,
) -> None:
    """Write the `rowii.fusionx.wrapper.load_xattn_head`-format checkpoint dict
    (that function's docstring) to *path*, creating parent directories as
    needed."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": dataclasses.asdict(cfg),
            "model": model.state_dict(),
            "run": run_name,
            "vib_dim": vib_dim,
            "epochs": epochs,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the cross-attention fusion head (third fusion level, package-5 "
            "spec D8, Task 6) CLIP-style on ALREADY-CACHED audio-beats embeddings "
            "(results/cache/<run>--audio-beats.npz) and the fusion cache's own "
            "vibration-branch columns (results/cache/<run>--fusion.npz) -- zero "
            "extraction compute. Writes models/adapted/xattn_<run>.pt + a sidecar "
            "<run>.json."
        )
    )
    parser.add_argument(
        "--run", required=True, metavar="NAME",
        help="Run name to train on -- needs BOTH audio-beats and fusion warm caches "
             "(scripts/warm_cache.py --runs <name> --variants audio-beats fusion).",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Training epochs (default: 20).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Mini-batch size (default: 256).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Adam learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--seed", type=int, default=7,
        help="Seeds weight init, the calibration/scoring split, and shuffling "
             "(default: 7).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("models/adapted/"),
        help="Output directory (default: models/adapted/); filename is "
             "xattn_<run>.pt.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    _import_torch_or_exit()

    cfg = load_config()
    run = _resolve_run_or_exit(args.run, cfg)

    t0 = time.monotonic()
    audio = _load_cache_or_exit(run, _AUDIO_VARIANT, cfg)
    vib_source = _load_cache_or_exit(run, _VIB_SOURCE_VARIANT, cfg)

    _check_cache_alignment(run.name, audio, vib_source)

    audio_idx, vib_idx = split_branch_columns(vib_source.feature_names)

    calibration_windows = _select_calibration_windows(vib_source, audio, seed=args.seed)
    if calibration_windows.size == 0:
        print(
            f"train_xattn: no calibration-side window(s) for run {run.name!r} -- "
            "nothing to train on",
            file=sys.stderr,
        )
        return 1

    audio_inputs = audio.features[calibration_windows]
    vib_inputs = vib_source.features[calibration_windows][:, vib_idx]
    vib_dim = int(vib_idx.shape[0])
    logger.info(
        "train_xattn: %d calibration-side window(s) for run %s (of %d/%d valid), "
        "vib_dim=%d (%d audio-branch columns excluded)",
        audio_inputs.shape[0], run.name,
        int(vib_source.valid_mask.sum()), int(audio.valid_mask.sum()),
        vib_dim, int(audio_idx.shape[0]),
    )

    xattn_cfg = XattnConfig()
    model, epoch_losses = _train_xattn_head(
        audio_inputs, vib_inputs, xattn_cfg, vib_dim,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
    )

    checkpoint_path = args.out / f"xattn_{run.name}.pt"
    _save_checkpoint(checkpoint_path, xattn_cfg, model, vib_dim, run.name, args.epochs)
    elapsed_s = time.monotonic() - t0

    sidecar = {
        "run": run.name,
        "audio_variant": _AUDIO_VARIANT,
        "vib_source_variant": _VIB_SOURCE_VARIANT,
        "vib_dim": vib_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "n_calibration_windows": int(audio_inputs.shape[0]),
        "final_loss": epoch_losses[-1],
        "elapsed_s": elapsed_s,
        "note": _LEAKAGE_NOTE,
    }
    sidecar_path = checkpoint_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(
        f"train_xattn: saved {checkpoint_path} ({args.epochs} epoch(s) over "
        f"{audio_inputs.shape[0]} calibration-side window(s), vib_dim={vib_dim}, "
        f"final mean loss {epoch_losses[-1]:.6f}); sidecar {sidecar_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
