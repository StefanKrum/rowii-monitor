"""TF-C pretraining CLI (package-4 spec D3, Task 3): trains the compact TF-C
encoder pair (`rowii.tfc.model.TfcModel`) on either the MIMII pump audio
corpus (`--corpus mimii` -> `tfc_audio.pt`) or the CWRU+Paderborn vibration
corpora (`--corpus bearings` -> `tfc_vib.pt`), writing a Task-1-format
checkpoint (`rowii.tfc.wrapper.load_tfc_model`'s docstring) into `--out`.

Corpus routing (`_corpus_windows`, orchestrator resolution 1): `mimii` walks
`--data-root/mimii/pump_0db` via `iter_windows_wav_dir` (MIMII's own
normal/abnormal layout, abnormal excluded); `bearings` CHAINS
`iter_windows_mat_dir` over `--data-root/cwru` (CWRU's flat `DE_time`
variable, 12 kHz) with `iter_windows_paderborn_dir` over
`--data-root/paderborn` (Task 3's own new adapter for Paderborn KAt's nested
struct layout, `rowii.tfc.corpora`'s own docstring) into ONE combined
stream -- `--limit-clips` applies INDEPENDENTLY to each underlying corpus
directory (a per-source file budget, not a combined one across CWRU and
Paderborn; each `iter_windows_*_dir` call already owns that semantic).

Windows are never all held in memory before subsampling: `_reservoir_sample`
is a genuine STREAMING reservoir sample (Algorithm R, seeded by
`--seed`/`np.random.default_rng`) over the corpus generator, capped at
`--max-windows` (default 200000) -- MIMII alone is tens of thousands of
clips, and every `iter_windows_*_dir` generator already promises to hold only
one clip's windows in memory at a time (`rowii.tfc.corpora`'s own docstring);
naively `list(...)`-ing the whole stream before subsampling would throw that
guarantee away. Every window the stream yields has an equal, uniform
probability of ending up in the final sample, regardless of source order.

Training (`_train`, orchestrator resolution 4): Adam + `rowii.tfc.model.
tfc_loss` (NT-Xent, `TfcConfig.temperature`), `--epochs` full passes over the
sampled windows in `--batch-size`-row mini-batches, shuffled each epoch by a
`--seed`-seeded CPU `torch.Generator` -- the SAME established pattern as
`rowii.anomaly.recon._train_autoencoder`'s shuffle generator (a CPU-resident
generator is fine for `torch.randperm`, whose int64 output only ever INDEXES
a device tensor, never generates data on one). The SAME generator also drives
every draw inside `_augment_time_view` (jitter/scale/mask), so the entire run
is reproducible from `--seed` alone (together with `torch.manual_seed(seed)`
for weight init, called before the model is constructed) -- CPU is the only
backend this reproducibility is verified/tested on (this project's tests
never run torch training against real data or on a non-CPU device; the
established MPS/CUDA caveat is `rowii.anomaly.recon`'s own module docstring).

`_augment_time_view` augments ONLY the time view (gaussian jitter sigma 0.01,
per-window random amplitude scale in [0.9, 1.1], a per-window random
zero-mask capped at 10%); the frequency view is `rowii.tfc.model.freq_view`
of the ALREADY-augmented time signal, never separately augmented -- a
documented simplification beyond design spec D1's per-view augmentation list
(time AND frequency), matching `rowii.tfc.model.tfc_loss`'s own "compact,
honest... non-SOTA-parity" simplification (that module's docstring) and
fixed by orchestrator resolution 4 for this task. Every random draw inside
`_augment_time_view` is generated ON CPU (matching the CPU-resident shuffle
generator) and only THEN moved to the training device with `.to(device)`:
verified directly (2026-07-16, this task) that passing a CPU
`torch.Generator` straight into `torch.rand(..., device="mps")` raises
`RuntimeError: Expected a 'mps' device type for generator but found 'cpu'`
-- generating on CPU first and moving the result is the only pattern that is
correct on every backend (CPU/MPS/CUDA) with one shared generator.

Checkpoint: `{"cfg": dataclasses.asdict(TfcConfig()), "model": state_dict,
"corpus_manifest_sha256": <sha256 of the corpus's MANIFEST.json(s), computed
by `_corpus_manifest_sha256`, or "unknown" if none is present>, "epochs":
<int>}` -- the exact dict format Task 1's `load_tfc_model`/`TfcFeaturizer`
already expect and round-trip-test against. Every checkpoint this script
writes uses `TfcConfig()`'s defaults (no CLI flag varies the architecture);
device via `rowii.signals.beats.best_device()`.

Real pretraining (multi-hour, real corpora) is orchestrator-led (Task 5,
plan's own execution notes) -- this script's own tests exercise it only
against tiny synthetic corpora on CPU (a handful of windows, 1-3 epochs),
never real downloaded corpus bytes.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from rowii.tfc.model import TfcModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.tfc.corpora import (  # noqa: E402
    iter_windows_mat_dir,
    iter_windows_paderborn_dir,
    iter_windows_wav_dir,
)
from rowii.tfc.wrapper import TfcConfig  # noqa: E402

logger = logging.getLogger(__name__)

_TORCH_HINT = "TF-C pretraining needs torch: pip install -e '.[beats]'"

_JITTER_SIGMA = 0.01
"""Additive Gaussian jitter std (spec D1 / orchestrator resolution 4),
applied to the standardized (unit-scale) time view."""

_SCALE_RANGE: tuple[float, float] = (0.9, 1.1)
"""Per-window random amplitude scale range (spec D1 / resolution 4)."""

_ZERO_MASK_MAX_FRAC = 0.10
"""Ceiling on the per-window random zero-mask fraction (spec D1 / resolution
4's "up to 10% of samples"): each window draws its OWN keep-threshold,
uniform in `[0, _ZERO_MASK_MAX_FRAC)`, then masks each sample independently
with that probability -- see `_augment_time_view`'s docstring for the full
semantics and why this bounds the EXPECTED, not a hard per-window, count."""

_CHECKPOINT_NAMES: dict[str, str] = {"mimii": "tfc_audio.pt", "bearings": "tfc_vib.pt"}

_MANIFEST_DIRS: dict[str, tuple[str, ...]] = {
    "mimii": ("mimii",),
    "bearings": ("cwru", "paderborn"),
}


def _import_torch_or_exit() -> None:
    """Mirrors `scripts/warm_cache.py`'s/`scripts/run_step1.py`'s own
    `_import_beats_or_exit` (duplicated rather than imported across
    independent scripts, per `warm_cache.py`'s own documented rationale for
    that choice): torch has no optional code path in THIS script (unlike
    those two, where only SOME variants need it) -- pretraining is
    inherently a torch operation, so this guard runs unconditionally, early
    in `main()`, right after argument parsing.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"torch not available ({exc}); {_TORCH_HINT}") from exc


def _corpus_windows(
    corpus: str, data_root: Path, limit_clips: int | None
) -> Iterator[np.ndarray]:
    """Route `--corpus` to its window iterator(s) (orchestrator resolution 1):
    `mimii` -> MIMII pump_0db WAVs (abnormal-excluded); `bearings` -> CWRU
    `.mat` (`DE_time`, 12 kHz) CHAINED with Paderborn `.mat` (`vibration_1`,
    64 kHz) into one combined stream. `limit_clips` is forwarded UNCHANGED to
    each underlying iterator call -- for `bearings` this means it is a
    PER-SOURCE budget (at most `limit_clips` CWRU files AND, independently,
    at most `limit_clips` Paderborn files), not a combined cap across both;
    the orchestrator's own CLI contract names one `--limit-clips N` flag with
    no further per-corpus split, and this is the simplest, most predictable
    reading of it.
    """
    if corpus == "mimii":
        yield from iter_windows_wav_dir(
            data_root / "mimii" / "pump_0db",
            exclude_substring="abnormal",
            limit_clips=limit_clips,
        )
    elif corpus == "bearings":
        yield from iter_windows_mat_dir(
            data_root / "cwru",
            key_substring="DE_time",
            native_hz=12_000.0,
            limit_clips=limit_clips,
        )
        yield from iter_windows_paderborn_dir(data_root / "paderborn", limit_clips=limit_clips)
    else:
        raise ValueError(f"unknown --corpus: {corpus!r}")


def _reservoir_sample(
    windows: Iterator[np.ndarray], max_windows: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Streaming reservoir sample (Algorithm R) of *windows* (assumed 1-D,
    equal-length rows -- every `iter_windows_*_dir` generator's contract:
    every window is resampled to the same fixed `target_hz`), capped at
    *max_windows* rows, WITHOUT ever materializing the full corpus stream in
    memory first (module docstring's rationale). Deterministic given *rng*'s
    own seed AND the corpus iterators' already-deterministic (sorted-walk)
    file order -- two calls with the same seed and the same underlying
    corpus directory produce byte-identical output.

    Each item seen so far has an equal `max_windows / n_seen` probability of
    surviving to the final sample at any point in the stream (the standard
    Algorithm R guarantee): for the `i`-th item (0-indexed), if fewer than
    `max_windows` items have been kept so far it is kept outright; otherwise
    a uniform random index `j` in `[0, i]` is drawn, and the item replaces
    slot `j` of the reservoir only if `j < max_windows`.

    Args:
        windows: A (possibly very long) stream of equal-shape 1-D arrays.
        max_windows: Reservoir capacity. `0` is valid (every item is
            "considered" -- `n_seen` still counts the full stream -- but
            none is ever kept, since `j < 0` is never true).
        rng: Seeded generator driving every reservoir-replacement draw.

    Returns:
        `(sample, n_seen)`: *sample* is a `(min(n_seen, max_windows),
        window_len)` float32 array (fewer than *max_windows* rows if the
        stream itself yielded fewer windows -- never padded; `(0, 0)` if the
        stream yielded nothing at all); *n_seen* is the TOTAL number of
        windows the stream actually yielded, regardless of how many were
        kept (for the caller's own logging).
    """
    reservoir: np.ndarray | None = None
    n_seen = 0
    for window in windows:
        if reservoir is None:
            reservoir = np.empty((max_windows, window.shape[0]), dtype=np.float32)
        if n_seen < max_windows:
            reservoir[n_seen] = window
        else:
            j = int(rng.integers(0, n_seen + 1))
            if j < max_windows:
                reservoir[j] = window
        n_seen += 1

    if reservoir is None:
        return np.empty((0, 0), dtype=np.float32), 0
    return reservoir[: min(n_seen, max_windows)], n_seen


def _corpus_manifest_sha256(corpus: str, data_root: Path) -> str:
    """Sha256 provenance fingerprint of *corpus*'s downloaded-corpus
    `MANIFEST.json`(s) (`scripts/download_corpora.py`'s own output), baked
    into the checkpoint (Task 1's checkpoint dict format) so a later
    `load_tfc_model` caller can tell WHICH downloaded corpus bytes trained
    this checkpoint. `bearings` combines BOTH `cwru/MANIFEST.json` and
    `paderborn/MANIFEST.json` (whichever of the two actually exist, sorted
    by directory name for a deterministic combination order) into ONE
    sha256 over their concatenated raw bytes -- gracefully degrading to
    whichever single manifest is present if the other corpus was never
    downloaded (Task 2's own documented CWRU-only fallback).

    Returns:
        The combined hex sha256 digest, or `"unknown"` if NO manifest is
        present for *corpus* at all (e.g. a hand-assembled corpus directory
        with no `scripts/download_corpora.py` provenance) -- resolution 4's
        documented fallback, never an error.
    """
    manifest_paths = sorted(data_root / d / "MANIFEST.json" for d in _MANIFEST_DIRS[corpus])
    existing = [p for p in manifest_paths if p.is_file()]
    if not existing:
        return "unknown"
    hasher = hashlib.sha256()
    for p in existing:
        hasher.update(p.read_bytes())
    return hasher.hexdigest()


def _augment_time_view(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """TF-C time-view augmentation (spec D1 / orchestrator resolution 4):
    per-window random amplitude SCALE (uniform in `_SCALE_RANGE`) -> additive
    Gaussian JITTER (std `_JITTER_SIGMA`) -> a per-window random ZERO-MASK.
    The mask draws its own per-window keep-threshold uniform in
    `[0, _ZERO_MASK_MAX_FRAC)`, then masks each of the window's samples
    independently with that probability -- so "up to `_ZERO_MASK_MAX_FRAC`"
    bounds the EXPECTED fraction masked per window (the threshold's own
    ceiling), not a hard per-window count; at `n_samples=8000` the realized
    fraction tracks that expectation tightly (binomial std dev an order of
    magnitude below the mean at any threshold in range).

    All draws come from the SAME *gen* (resolution 4: "via the torch
    Generator"), in this fixed order (scale -> jitter -> mask-threshold ->
    mask), so a full training run is exactly reproducible from one seed
    alone (`_train`'s own docstring).

    Every random tensor here is generated ON CPU (`torch.Generator()` with
    no `device=` is CPU-resident by construction, matching `_train`'s own
    shuffle generator) and only THEN moved to `x.device` with `.to()` --
    verified (module docstring) that a CPU generator cannot drive
    `torch.rand`/`.uniform_()` directly on an MPS-device tensor.

    The frequency view is deliberately NOT separately augmented here -- see
    module docstring's "documented simplification" paragraph; callers derive
    it from THIS function's output (`freq_view(_augment_time_view(x, gen))`).

    Args:
        x: `(B, N)` standardized time-domain windows, already on the
            training device.
        gen: CPU-resident seeded generator (shared with the caller's shuffle
            generator for the epoch this batch belongs to).

    Returns:
        `(B, N)` augmented windows, same dtype/device as *x*.
    """
    import torch

    b, n = x.shape
    scale = torch.empty(b, 1).uniform_(*_SCALE_RANGE, generator=gen).to(x.device)
    jitter = (torch.randn(b, n, generator=gen) * _JITTER_SIGMA).to(x.device)
    augmented = x * scale + jitter

    mask_threshold = torch.empty(b, 1).uniform_(0.0, _ZERO_MASK_MAX_FRAC, generator=gen)
    zero_mask = (torch.rand(b, n, generator=gen) < mask_threshold).to(x.device)
    return augmented.masked_fill(zero_mask, 0.0)


def _train(
    windows: np.ndarray,
    cfg: TfcConfig,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[TfcModel, list[float]]:
    """Train a `TfcModel` on *windows* (Task 1's `TfcModel`/`tfc_loss`/
    `freq_view`; orchestrator resolution 4): Adam, *epochs* full passes over
    *windows* in *batch_size*-row mini-batches, shuffled each epoch by a
    *seed*-seeded CPU `torch.Generator`
    (`rowii.anomaly.recon._train_autoencoder`'s established shuffle
    pattern) -- the SAME generator instance also drives every
    `_augment_time_view` draw, so the whole run is reproducible from *seed*
    alone (together with `torch.manual_seed(seed)` for weight init, called
    BEFORE the model is constructed -- weight init and shuffle/augmentation
    are two independent seeded sources, both fixed by the one *seed* value,
    mirroring `MlpAeScorer.fit`'s identical convention).

    Per training step: draw a shuffled mini-batch -> `_augment_time_view`
    (TIME view only) -> `freq_view` of the AUGMENTED time signal (never the
    raw one -- module docstring) -> forward both through the model ->
    `tfc_loss` on the resulting projections -> Adam step.

    Args:
        windows: `(N, n_samples)` float32 standardized windows (already
            reservoir-sampled by the caller).
        cfg: Architecture (`TfcConfig()` in every real run this script
            performs -- no CLI flag varies it).
        epochs: Full passes over *windows*.
        batch_size: Rows per gradient step (the last batch of an epoch may
            be smaller, `len(windows) % batch_size`).
        lr: Adam learning rate.
        seed: Seeds `torch.manual_seed` (weight init) AND the shared
            shuffle+augmentation generator.

    Returns:
        `(model, epoch_losses)`: *model* on `best_device()`, in `.eval()`
        mode (mirrors `_train_autoencoder`'s postcondition -- every
        downstream embedding call reads it under `torch.no_grad()`);
        *epoch_losses* is one MEAN loss per epoch, in order, for the caller
        to log.
    """
    import torch

    from rowii.signals.beats import best_device
    from rowii.tfc.model import TfcModel, freq_view, tfc_loss

    torch.manual_seed(seed)
    device = best_device()
    model = TfcModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x_all = torch.from_numpy(windows.astype(np.float32)).to(device)
    n = x_all.shape[0]
    gen = torch.Generator().manual_seed(seed)

    epoch_losses: list[float] = []
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch = x_all[perm[start : start + batch_size]]
            opt.zero_grad()
            augmented = _augment_time_view(batch, gen)
            x_freq = freq_view(augmented)
            _h_t, _h_f, z_t, z_f = model(augmented, x_freq)
            loss = tfc_loss(z_t, z_f, cfg.temperature)
            # tfc_loss's explicit `-> torch.Tensor` return annotation (unlike
            # recon.py's `_train_autoencoder`, whose `loss` comes from an
            # nn.Module.__call__ and is therefore already `Any`) makes mypy
            # check this call against torch's own (untyped) Tensor.backward
            # stub -- same gap, same fix as beats_model.py's load_beats_model.
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        mean_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(mean_loss)
        logger.info("pretrain_tfc: epoch %d/%d -- mean loss %.6f", epoch + 1, epochs, mean_loss)

    model.eval()
    return model, epoch_losses


def _save_checkpoint(
    path: Path, cfg: TfcConfig, model: TfcModel, corpus_manifest_sha256: str, epochs: int
) -> None:
    """Write the Task-1-format checkpoint dict (module docstring) to *path*,
    creating parent directories as needed."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": dataclasses.asdict(cfg),
            "model": model.state_dict(),
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "epochs": epochs,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-train the compact TF-C encoder pair (rowii.tfc.model.TfcModel) on "
            "MIMII pump audio (--corpus mimii -> tfc_audio.pt) or CWRU+Paderborn "
            "bearing vibration (--corpus bearings -> tfc_vib.pt), writing a "
            "Task-1-format checkpoint into --out (package-4 spec D3, Task 3)."
        )
    )
    parser.add_argument(
        "--corpus", required=True, choices=("mimii", "bearings"),
        help="Corpus to pretrain on: mimii (audio, -> tfc_audio.pt) or bearings "
             "(CWRU + Paderborn vibration, -> tfc_vib.pt).",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/public"),
        help="Root directory holding the downloaded corpora (default: data/public, "
             "scripts/download_corpora.py's own default --dest).",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs (default: 40).")
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Mini-batch size (default: 256).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Adam learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--seed", type=int, default=7,
        help="Seeds window subsampling, weight init, shuffling, and augmentation "
             "(default: 7).",
    )
    parser.add_argument(
        "--limit-clips", type=int, default=None, metavar="N",
        help="Cap files opened per corpus source (dev subsampling; default: unlimited).",
    )
    parser.add_argument(
        "--max-windows", type=int, default=200_000,
        help="Reservoir-sample cap on total training windows (default: 200000).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("models/pretrained/tfc/"),
        help="Output directory for the checkpoint (default: models/pretrained/tfc/); "
             "filename is tfc_audio.pt (mimii) or tfc_vib.pt (bearings).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    _import_torch_or_exit()

    rng = np.random.default_rng(args.seed)
    sample, n_seen = _reservoir_sample(
        _corpus_windows(args.corpus, args.data_root, args.limit_clips), args.max_windows, rng,
    )
    if sample.shape[0] == 0:
        print(
            f"pretrain_tfc: no windows found for --corpus {args.corpus} under "
            f"{args.data_root} -- nothing to train on",
            file=sys.stderr,
        )
        return 1
    logger.info(
        "pretrain_tfc: --corpus %s -- %d window(s) seen, %d sampled (--max-windows %d)",
        args.corpus, n_seen, sample.shape[0], args.max_windows,
    )

    cfg = TfcConfig()
    model, epoch_losses = _train(
        sample, cfg, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
    )

    manifest_sha256 = _corpus_manifest_sha256(args.corpus, args.data_root)
    checkpoint_path = args.out / _CHECKPOINT_NAMES[args.corpus]
    _save_checkpoint(checkpoint_path, cfg, model, manifest_sha256, args.epochs)

    print(
        f"pretrain_tfc: saved {checkpoint_path} ({args.epochs} epoch(s) over "
        f"{sample.shape[0]} window(s), final mean loss {epoch_losses[-1]:.6f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
