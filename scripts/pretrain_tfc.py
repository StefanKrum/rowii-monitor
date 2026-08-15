"""TF-C pretraining CLI: trains the compact TF-C encoder pair
(`rowii.tfc.model.TfcModel`) on the MIMII pump audio corpus (`--corpus
mimii` -> `tfc_audio.pt`), the CWRU+Paderborn vibration corpora (`--corpus
bearings` -> `tfc_vib.pt`), or the PSHP plant's own pooled calibration-side
audio (`--corpus pshp-pool` -> `tfc_audio_pshp.pt`), writing a checkpoint
(`rowii.tfc.wrapper.load_tfc_model`'s docstring) into `--out`.

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

PSHP pool (`--corpus pshp-pool`, package-7 spec D4 as amended by A3.9):
windows come per `--pool-runs` run from `rowii.adapt.target_windows.
iter_target_windows(run, cfg, target_hz=8000)` -- EXCLUSIVELY the
calibration side of each run's canonical top split (that helper's OWN
pinned seed 7, deliberately NOT `--seed`: the split must stay the one every
Step-2 sweep calibrates/scores against, or pretraining could swallow
scoring-side windows), i.e. the package-5 leakage rule for free. The pool is
MATERIALIZED ONCE to `<out>/pshp_pool_windows.npz` (members: `windows`
float32 `(N, 8000)`, `run_names`, `per_run_counts` int64, `fingerprint` =
sha256 over run names + per-run window counts + target_hz; loaded with
`allow_pickle=False`) and REUSED whenever the stored fingerprint matches the
requested pool (cache HIT logged; any mismatch or unreadable file
re-materializes) -- so D4's continued/from-scratch pretraining PAIR streams
the ~40 GB of Gantner audio once, not twice. Sequential per-run
concatenation is deliberate (no round-robin): pretraining consumes ALL pool
windows, so there is no budget-starvation concern (A3.11's round-robin
rationale applies to capped adaptation draws, not here). Unlike the public
corpora, pshp-pool has NO `download_corpora.py` manifest -- provenance lives
in the repo's own data layout (`rowii.io.dataset.discover`), so the pool
FINGERPRINT plays the checkpoint's `corpus_manifest_sha256` role.

Continued pretraining (`--continue-from PATH`, any corpus): the model is
initialized from that Task-1-format checkpoint's state dict instead of a
fresh seed-derived init, with the architecture following the SOURCE
checkpoint's own `cfg` (mirroring `load_tfc_model`'s reconstruction,
including its strict load -- never `strict=False`). Lineage is recorded as
`continued_from` (absolute path string, or None) in the checkpoint dict and
sidecar, alongside `pool_runs` (None for public corpora). Every checkpoint
now gets a `<same-stem>.json` sidecar (`adapt_beats.py`/`distill_beats.py`'s
own convention); the pshp-pool sidecar's note restates the calibration-side
leakage rule, the pool runs, and the A2.1 universality framing (a
PSHP-adapted encoder is plant-specific; the frozen public-corpus checkpoint
stays the universal reference baseline).

`load_config`/`discover`/`iter_target_windows` are module-level imports that
are only ever CALLED on the pshp-pool path -- module-level for the same
reason `scripts/adapt_beats.py` documents (the `warm_cache.py` precedent):
tests monkeypatch them (`monkeypatch.setattr(pretrain_tfc,
"iter_target_windows", ...)`), which a function-local import would silently
undo by re-resolving the real symbol on every call.

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

Checkpoint: `{"cfg": dataclasses.asdict(<cfg>), "model": state_dict,
"corpus_manifest_sha256": <sha256 of the corpus's MANIFEST.json(s), computed
by `_corpus_manifest_sha256`, "unknown" if none is present, or the pool
fingerprint for pshp-pool>, "epochs": <int>, "continued_from": <str | None>,
"pool_runs": <list[str] | None>}` -- a superset of the dict format Task 1's
`load_tfc_model`/`TfcFeaturizer` already expect and round-trip-test against
(the two lineage keys are additive; that loader reads only the keys it
names). Fresh runs use `TfcConfig()`'s defaults (no CLI flag varies the
architecture); `--continue-from` runs inherit the source checkpoint's `cfg`;
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
import json
import logging
import sys
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from rowii.tfc.model import TfcModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.adapt.target_windows import iter_target_windows  # noqa: E402
from rowii.config import Config, load_config  # noqa: E402
from rowii.io.dataset import Run, discover  # noqa: E402
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

_CHECKPOINT_NAMES: dict[str, str] = {
    "mimii": "tfc_audio.pt",
    "bearings": "tfc_vib.pt",
    "pshp-pool": "tfc_audio_pshp.pt",
}
"""Default checkpoint filename per corpus -- overridable per run via
`--out-name` (package-7 Task 6: D4's from-scratch control writes
`tfc_audio_pshp_scratch.pt`, D5's re-pretrain writes `tfc_vib_v2.pt`)."""

_MANIFEST_DIRS: dict[str, tuple[str, ...]] = {
    "mimii": ("mimii",),
    "bearings": ("cwru", "paderborn"),
    # pshp-pool has NO scripts/download_corpora.py manifest by design: PSHP
    # data provenance lives in the repo's own data layout (rowii.io.dataset.
    # discover), not a download manifest, and main() records the pool
    # FINGERPRINT in the checkpoint's corpus_manifest_sha256 slot instead.
    # The empty tuple keeps _corpus_manifest_sha256 a graceful "unknown"
    # (never a KeyError) if anything ever routes pshp-pool through it.
    "pshp-pool": (),
}

_POOL_TARGET_HZ = 8000
"""TF-C's fixed input rate (`rowii.tfc.wrapper`'s own `_TFC_SAMPLE_RATE_HZ`
-- restated rather than imported: a stable architectural literal, the same
rationale `rowii.adapt.target_windows._PRIMARY_MIC_STREAM` gives for its own
duplication): every pshp-pool window is drawn already resampled to 8 kHz."""

_POOL_WINDOWS_FILENAME = "pshp_pool_windows.npz"
"""Materialize-once cache for the pool's windows, living inside `--out` next
to the checkpoints it feeds (spec A3.9: one ~1.2 GB npz instead of streaming
~40 GB of Gantner audio once PER pretraining of D4's continued/scratch
pair)."""

_DEFAULT_POOL_RUNS = "010726-tu_ph_tu,290626-tu,290626-pu,010726-pu"
"""The canonical final-setup pool (package-7 spec D1): TU+PH+PU coverage,
calibration sides only; 250526-*/270626-* are excluded by design (cross-
config probe / old-config runs, never pooled)."""

_PSHP_POOL_NOTE_TEMPLATE = (
    "PSHP-pool pretraining windows come EXCLUSIVELY from each pool run's calibration side "
    "via rowii.adapt.target_windows.iter_target_windows (per-run top split at its pinned "
    "seed 7 -- the package-5 leakage rule: no scoring-side window ever reaches "
    "pretraining). Pool runs: {runs}. Universality framing (package-7 spec A2.1): a "
    "PSHP-adapted encoder is PLANT-SPECIFIC -- it measures what plant-specific data buys; "
    "the frozen public-corpus checkpoint remains the UNIVERSAL reference baseline in "
    "every comparison."
)
"""Sidecar note for `--corpus pshp-pool` (spec A2.1 + the A4.3 firewall
spirit: our artifacts state their own provenance and framing)."""

_PUBLIC_CORPUS_NOTE = (
    "public-corpus pretraining; data provenance = corpus_manifest_sha256 over "
    "scripts/download_corpora.py's MANIFEST.json(s), 'unknown' when none is present."
)
"""Sidecar note for the public corpora (mimii/bearings)."""


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
    elif corpus == "pshp-pool":
        raise ValueError(
            "pshp-pool windows are materialized once via _load_cached_pool_windows/"
            "_materialize_pool_windows in main() (package-7 Task 6), never streamed "
            "through this public-corpus router"
        )
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
        documented fallback, never an error. `pshp-pool` always lands here
        (its `_MANIFEST_DIRS` entry is empty -- see that dict's own comment;
        `main()` records the pool fingerprint instead of ever consulting
        this function for it).
    """
    manifest_paths = sorted(data_root / d / "MANIFEST.json" for d in _MANIFEST_DIRS[corpus])
    existing = [p for p in manifest_paths if p.is_file()]
    if not existing:
        return "unknown"
    hasher = hashlib.sha256()
    for p in existing:
        hasher.update(p.read_bytes())
    return hasher.hexdigest()


_POOL_SIGNATURE_STREAM = "RAWGeneratorMic__0"
"""The stream whose burst files determine the pool windows (`iter_target_windows`
reads the primary mic) -- the file-signature side of the pool fingerprint stats
exactly these files."""


def _pool_file_signatures(runs: Sequence[Run]) -> list[list[tuple[str, int, int]]]:
    """Per run (in *runs* order): `(basename, size_bytes, mtime_ns)` for every
    `_POOL_SIGNATURE_STREAM` burst file, sorted by basename -- `stat()` only,
    no audio is read. A run without mic files signs as an empty list (it also
    yields zero windows). A file that exists in discovery but cannot be
    stat'ed is a loud SystemExit (the corpus moved under us -- refusing beats
    silently mis-fingerprinting)."""
    signatures: list[list[tuple[str, int, int]]] = []
    for run in runs:
        entries: list[tuple[str, int, int]] = []
        for burst in sorted(
            run.files.get(_POOL_SIGNATURE_STREAM, ()), key=lambda b: b.path.name
        ):
            try:
                stat = burst.path.stat()
            except OSError as exc:
                print(
                    f"pretrain_tfc: cannot stat pool source file {burst.path} "
                    f"({exc}) -- refusing to fingerprint a moving corpus",
                    file=sys.stderr,
                )
                raise SystemExit(2) from exc
            entries.append((burst.path.name, int(stat.st_size), int(stat.st_mtime_ns)))
        signatures.append(entries)
    return signatures


def _pool_fingerprint(
    run_names: Sequence[str],
    target_hz: int,
    file_signatures: Sequence[Sequence[tuple[str, int, int]]],
) -> str:
    """Identity of a materialized PSHP pool (spec A3.9, HARDENED per the T6
    review): sha256 over the pool run names, the target sample rate, and the
    per-run `(basename, size, mtime_ns)` signatures of the mic-stream burst
    files -- serialized as canonical JSON so the digest is stable across
    processes. The file signatures are the load-bearing part: a fingerprint of
    names + window COUNTS alone was proven blind to a same-structure re-ingest
    with different CONTENT (the exact bug class `scripts/scarcity_detection.py`'s
    manifest already guards with size+mtime -- the P6-review precedent this now
    mirrors). Computable BEFORE materialization (stat only), so the cache-HIT
    check validates freshness against the live corpus on every invocation.
    Deliberately NOT hashed: `--seed`/`--max-windows` (they act AFTER
    materialization, on the training subsample) and the split seed (pinned at 7
    inside `iter_target_windows`, module docstring)."""
    payload = json.dumps(
        {
            "runs": list(run_names),
            "target_hz": int(target_hz),
            "files": [
                [[name, size, mtime] for name, size, mtime in run_signature]
                for run_signature in file_signatures
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cached_pool_windows(
    npz_path: Path, pool_runs: list[str], expected_fingerprint: str
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Reuse a previously materialized pool npz (spec A3.9's materialize-ONCE
    rule) if -- and only if -- it is the pool the caller asked for AND the
    live corpus still matches (*expected_fingerprint* is
    the file-signature `_pool_fingerprint` computed by the caller from the
    CURRENT on-disk burst files via a stat-only pass -- a same-structure
    re-ingest with different content therefore MISSES instead of silently
    serving stale windows; no audio is ever read for validation). The stored
    run-name list must equal the requested one, the stored member set/
    geometry/dtype must match the npz contract, and the counts must sum to
    the window rows. Any unreadable or malformed file is a MISS (warned,
    then re-materialized), never an error -- a half-written cache from an
    interrupted run heals itself on the next invocation.

    Returns:
        `(windows, per_run_counts, fingerprint)` on a HIT (logged), else
        `None`.
    """
    if not npz_path.is_file():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            if set(npz.files) != {"windows", "run_names", "per_run_counts", "fingerprint"}:
                logger.warning(
                    "pretrain_tfc: pool windows cache %s has unexpected members %s -- "
                    "re-materializing",
                    npz_path, sorted(npz.files),
                )
                return None
            windows = npz["windows"]
            run_names = [str(name) for name in npz["run_names"].tolist()]
            per_run_counts = npz["per_run_counts"].astype(np.int64)
            stored_fingerprint = str(npz["fingerprint"])
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        logger.warning(
            "pretrain_tfc: pool windows cache %s unreadable (%s) -- re-materializing",
            npz_path, exc,
        )
        return None

    consistent = (
        stored_fingerprint == expected_fingerprint
        and run_names == pool_runs
        and windows.ndim == 2
        and windows.shape[1] == _POOL_TARGET_HZ
        and windows.dtype == np.float32
        and per_run_counts.shape[0] == len(pool_runs)
        and int(per_run_counts.sum()) == int(windows.shape[0])
    )
    if not consistent:
        logger.info(
            "pretrain_tfc: pool windows cache MISS for %s (cached runs %s vs requested %s) "
            "-- re-materializing",
            npz_path, run_names, pool_runs,
        )
        return None
    logger.info(
        "pretrain_tfc: pool windows cache HIT -- reusing %s (%d window(s), fingerprint %s)",
        npz_path, int(windows.shape[0]), stored_fingerprint,
    )
    return windows, per_run_counts, stored_fingerprint


def _materialize_pool_windows(
    runs: list[Run], data_cfg: Config, npz_path: Path, target_hz: int, fingerprint: str
) -> tuple[np.ndarray, np.ndarray, str]:
    """Draw every pool run's calibration-side windows (module docstring's
    leakage rule: `iter_target_windows` at ITS pinned split seed 7, resampled
    to *target_hz*) by simple sequential per-run concatenation, and persist
    them ONCE to *npz_path* for every later pretraining to reuse (spec
    A3.9). A run contributing zero windows is a warning, never an error; an
    entirely empty pool writes NO npz (a cached zero-window "HIT" would
    poison every later run -- `main()`'s no-windows exit handles it) and
    returns an empty array.

    *fingerprint* is the caller-computed `_pool_fingerprint` (file-signature
    based) recorded verbatim in the npz -- this function
    never computes identity itself, so cache-check and write can never drift.

    Returns:
        `(windows, per_run_counts, fingerprint)`: `(N, target_hz)` float32
        windows in pool order, `int64` per-run counts aligned with *runs*,
        and *fingerprint* passed through.
    """
    run_names = [run.name for run in runs]
    logger.info(
        "pretrain_tfc: materializing pool windows for %d run(s) (%s) -> %s",
        len(runs), ", ".join(run_names), npz_path,
    )
    all_windows: list[np.ndarray] = []
    counts: list[int] = []
    for run in runs:
        n_before = len(all_windows)
        for window in iter_target_windows(run, data_cfg, target_hz=target_hz):
            all_windows.append(np.asarray(window, dtype=np.float32))
        n_run = len(all_windows) - n_before
        counts.append(n_run)
        if n_run == 0:
            logger.warning(
                "pretrain_tfc: pool run %s contributed ZERO calibration-side windows",
                run.name,
            )
        else:
            logger.info("pretrain_tfc: pool run %s -- %d window(s)", run.name, n_run)

    per_run_counts = np.asarray(counts, dtype=np.int64)
    if not all_windows:
        return np.empty((0, target_hz), dtype=np.float32), per_run_counts, fingerprint

    windows = np.stack(all_windows)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        windows=windows,
        run_names=np.asarray(run_names),
        per_run_counts=per_run_counts,
        fingerprint=np.asarray(fingerprint),
    )
    logger.info(
        "pretrain_tfc: materialized %d pool window(s) (%.1f MB) -> %s (fingerprint %s)",
        windows.shape[0], windows.nbytes / 1e6, npz_path, fingerprint,
    )
    return windows, per_run_counts, fingerprint


def _load_continue_checkpoint(path: Path) -> tuple[TfcConfig, dict[str, torch.Tensor]]:
    """Read the `--continue-from` source checkpoint (Task-1 dict format):
    returns its architecture (`TfcConfig` rebuilt from the checkpoint's own
    `cfg`, with `load_tfc_model`'s identical defensive `channels`-to-tuple
    coercion) and its model state dict, for `_train` to load as the init.
    Existence is checked by the caller (`main()`, a clean exit-2); a file
    that exists but is not a TF-C checkpoint -- INCLUDING one torch cannot
    even deserialize (garbage bytes) -- prints a pointed
    message to stderr and raises `SystemExit(2)`, never a raw traceback and
    never the bare-string `SystemExit` whose process status is 1."""
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # torch raises many types here; all mean "not a checkpoint"
        print(
            f"pretrain_tfc: --continue-from {path} could not be read as a torch "
            f"checkpoint ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    if not isinstance(state, dict) or "cfg" not in state or "model" not in state:
        print(
            f"pretrain_tfc: --continue-from {path} is not a Task-1-format TF-C checkpoint "
            "(expected a dict with 'cfg' and 'model' keys -- "
            "rowii.tfc.wrapper.load_tfc_model's documented format)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    cfg_dict = dict(state["cfg"])
    cfg_dict["channels"] = tuple(cfg_dict["channels"])
    return TfcConfig(**cfg_dict), state["model"]


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
    init_state: dict[str, torch.Tensor] | None = None,
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
        init_state: If given (`--continue-from`, package-7 spec D4), a model
            state dict loaded over the fresh seed-derived init BEFORE the
            first training step, STRICTLY (any key/shape mismatch between
            *cfg*'s architecture and the state dict is a hard error --
            `load_tfc_model`'s own never-`strict=False` convention). `None`
            keeps the fresh init (from-scratch pretraining).

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
    if init_state is not None:
        # --continue-from: overwrite the fresh init with the source
        # checkpoint's weights (CPU tensors copy into the device parameters).
        try:
            model.load_state_dict(init_state)
        except RuntimeError as exc:
            # strict=True (the default) is load-bearing: under strict=False a
            # MISSING key would silently keep its fresh random init -- a
            # quietly corrupted "continued" run. Normalize
            # the mismatch to the CLI's loud exit-2 contract.
            print(
                f"pretrain_tfc: --continue-from state dict does not match the "
                f"model architecture ({exc})",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
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
    path: Path,
    cfg: TfcConfig,
    model: TfcModel,
    corpus_manifest_sha256: str,
    epochs: int,
    *,
    continued_from: str | None,
    pool_runs: list[str] | None,
) -> None:
    """Write the Task-1-format checkpoint dict (module docstring) to *path*,
    creating parent directories as needed. *continued_from*/*pool_runs* are
    the package-7 lineage keys (spec A3.9) -- always present, `None` for a
    fresh public-corpus run."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": dataclasses.asdict(cfg),
            "model": model.state_dict(),
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "epochs": epochs,
            "continued_from": continued_from,
            "pool_runs": pool_runs,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-train the compact TF-C encoder pair (rowii.tfc.model.TfcModel) on "
            "MIMII pump audio (--corpus mimii -> tfc_audio.pt), CWRU+Paderborn "
            "bearing vibration (--corpus bearings -> tfc_vib.pt), or the PSHP pool's "
            "own calibration-side audio (--corpus pshp-pool -> tfc_audio_pshp.pt; "
            "package-7 spec D4/A3.9), writing a Task-1-format checkpoint into --out "
            "(package-4 spec D3, Task 3)."
        )
    )
    parser.add_argument(
        "--corpus", required=True, choices=("mimii", "bearings", "pshp-pool"),
        help="Corpus to pretrain on: mimii (audio, -> tfc_audio.pt), bearings "
             "(CWRU + Paderborn vibration, -> tfc_vib.pt), or pshp-pool (PSHP "
             "calibration-side plant audio, -> tfc_audio_pshp.pt).",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/public"),
        help="Root directory holding the downloaded PUBLIC corpora (default: data/public, "
             "scripts/download_corpora.py's own default --dest). Not used by pshp-pool, "
             "which discovers plant data via rowii.config's own ROWII_DATA_ROOT.",
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
             "(default: 7). The pshp-pool calibration split itself stays pinned at "
             "iter_target_windows' canonical seed 7 regardless (leakage rule).",
    )
    parser.add_argument(
        "--limit-clips", type=int, default=None, metavar="N",
        help="Cap files opened per PUBLIC corpus source (dev subsampling; default: "
             "unlimited). Not used by pshp-pool (no clip files -- materialized windows).",
    )
    parser.add_argument(
        "--max-windows", type=int, default=200_000,
        help="Reservoir-sample cap on total training windows (default: 200000). For "
             "pshp-pool it caps the TRAINING subsample only -- the materialized npz "
             "always holds the full pool.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("models/pretrained/tfc/"),
        help="Output directory for the checkpoint (default: models/pretrained/tfc/); "
             "filename is tfc_audio.pt (mimii), tfc_vib.pt (bearings), or "
             "tfc_audio_pshp.pt (pshp-pool) unless --out-name overrides it. pshp-pool "
             "also materializes pshp_pool_windows.npz here.",
    )
    parser.add_argument(
        "--pool-runs", default=_DEFAULT_POOL_RUNS, metavar="CSV",
        help="Comma-separated PSHP run names for --corpus pshp-pool (default: the "
             "canonical package-7 D1 pool, %(default)s). Ignored for public corpora.",
    )
    parser.add_argument(
        "--continue-from", type=Path, default=None, metavar="PATH",
        help="Existing TF-C checkpoint whose model state dict initializes training "
             "(continued pretraining, package-7 spec D4; architecture follows THAT "
             "checkpoint's cfg). Omitted -> fresh random init. Recorded as "
             "continued_from in the checkpoint dict and sidecar.",
    )
    parser.add_argument(
        "--out-name", default=None, metavar="NAME.pt",
        help="Checkpoint filename override inside --out, for ANY corpus (default: the "
             "corpus's canonical name). D4's from-scratch control uses "
             "--out-name tfc_audio_pshp_scratch.pt; D5's vib re-pretrain tfc_vib_v2.pt.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    _import_torch_or_exit()

    # --continue-from is resolved BEFORE any corpus work: a typo'd path must
    # fail in seconds, not after a full corpus walk/materialization.
    cfg = TfcConfig()
    init_state: dict[str, torch.Tensor] | None = None
    continued_from: str | None = None
    if args.continue_from is not None:
        if not args.continue_from.is_file():
            print(
                f"pretrain_tfc: --continue-from checkpoint not found: {args.continue_from}",
                file=sys.stderr,
            )
            return 2
        cfg, init_state = _load_continue_checkpoint(args.continue_from)
        continued_from = str(args.continue_from.resolve())
        logger.info(
            "pretrain_tfc: continuing from %s (architecture follows its cfg)", continued_from
        )

    windows_stream: Iterator[np.ndarray]
    pool_runs: list[str] | None = None
    pool_fingerprint: str | None = None
    pool_window_counts: dict[str, int] | None = None
    if args.corpus == "pshp-pool":
        requested = [name.strip() for name in args.pool_runs.split(",") if name.strip()]
        if not requested:
            print("pretrain_tfc: --pool-runs is empty -- nothing to pool", file=sys.stderr)
            return 2
        npz_path = args.out / _POOL_WINDOWS_FILENAME
        # Hardening: discovery + a stat-only file-signature pass run on
        # EVERY invocation (a few seconds) so the cache-HIT check validates
        # freshness against the live corpus -- a counts-only fingerprint was
        # proven blind to same-structure content changes.
        data_cfg = load_config()
        index = discover(data_cfg.data_root)
        runs_by_name = {run.name: run for run in index.runs}
        unknown = [name for name in requested if name not in runs_by_name]
        if unknown:
            available = ", ".join(sorted(runs_by_name)) or "(none discovered)"
            print(
                f"pretrain_tfc: unknown --pool-runs run(s): {', '.join(unknown)}; "
                f"available runs: {available}",
                file=sys.stderr,
            )
            return 2
        pool_run_objs = [runs_by_name[name] for name in requested]
        expected_fp = _pool_fingerprint(
            requested, _POOL_TARGET_HZ, _pool_file_signatures(pool_run_objs)
        )
        cached = _load_cached_pool_windows(npz_path, requested, expected_fp)
        if cached is None:
            cached = _materialize_pool_windows(
                pool_run_objs, data_cfg, npz_path, _POOL_TARGET_HZ, expected_fp
            )
        pool_windows, per_run_counts, pool_fingerprint = cached
        pool_runs = requested
        pool_window_counts = {
            name: int(count)
            for name, count in zip(requested, per_run_counts.tolist(), strict=True)
        }
        # The pool fingerprint plays the manifest role for pshp-pool (module
        # docstring: provenance lives in the repo's own data layout, not a
        # download manifest).
        manifest_sha256 = pool_fingerprint
        note = _PSHP_POOL_NOTE_TEMPLATE.format(runs=", ".join(requested))
        windows_stream = iter(pool_windows)
    else:
        manifest_sha256 = _corpus_manifest_sha256(args.corpus, args.data_root)
        note = _PUBLIC_CORPUS_NOTE
        windows_stream = _corpus_windows(args.corpus, args.data_root, args.limit_clips)

    rng = np.random.default_rng(args.seed)
    sample, n_seen = _reservoir_sample(windows_stream, args.max_windows, rng)
    if sample.shape[0] == 0:
        where = (
            f"in pool runs {', '.join(pool_runs)}"
            if pool_runs is not None
            else f"under {args.data_root}"
        )
        print(
            f"pretrain_tfc: no windows found for --corpus {args.corpus} {where} -- "
            "nothing to train on",
            file=sys.stderr,
        )
        return 1
    logger.info(
        "pretrain_tfc: --corpus %s -- %d window(s) seen, %d sampled (--max-windows %d)",
        args.corpus, n_seen, sample.shape[0], args.max_windows,
    )

    model, epoch_losses = _train(
        sample, cfg, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, init_state=init_state,
    )

    checkpoint_name = args.out_name or _CHECKPOINT_NAMES[args.corpus]
    checkpoint_path = args.out / checkpoint_name
    _save_checkpoint(
        checkpoint_path, cfg, model, manifest_sha256, args.epochs,
        continued_from=continued_from, pool_runs=pool_runs,
    )

    sidecar: dict[str, object] = {
        "corpus": args.corpus,
        "checkpoint": checkpoint_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "max_windows": args.max_windows,
        "n_windows_seen": n_seen,
        "n_windows_trained": int(sample.shape[0]),
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "corpus_manifest_sha256": manifest_sha256,
        "continued_from": continued_from,
        "pool_runs": pool_runs,
        "pool_window_counts": pool_window_counts,
        "pool_fingerprint": pool_fingerprint,
        "note": note,
    }
    sidecar_path = checkpoint_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(
        f"pretrain_tfc: saved {checkpoint_path} ({args.epochs} epoch(s) over "
        f"{sample.shape[0]} window(s), final mean loss {epoch_losses[-1]:.6f}); "
        f"sidecar {sidecar_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
