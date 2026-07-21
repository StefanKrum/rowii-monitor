"""Knowledge-distillation CLI (Step-2 package-5 spec D5, Task 4): trains a
compact CNN student (`rowii.adapt.student.StudentConfig`/
`rowii.adapt._student_model._StudentNet`) to regress the frozen BEATs teacher's
768-d embedding (the PRIMARY-mic column slice of the audio-beats cache, which
concatenates both mic streams -- see `_teacher_target_columns`) from the
`logmel` variant's own input patches, using ONLY
ALREADY-CACHED feature matrices -- `results/cache/<run>--audio-beats.npz` (the
teacher, `rowii.signals.beats.BeatsFeaturizer`'s frozen embeddings) and
`results/cache/<run>--logmel.npz` (the student's input, `rowii.signals.logmel.
LogmelFeaturizer`'s flattened patches). Zero teacher/extraction compute: this
script never runs BEATs or logmel feature extraction itself, never imports
`rowii.signals.beats`, and refuses outright (`_load_cache_or_exit`) rather than
triggering one, mirroring `scripts/analyze_step2.py`'s own `_grid_for_combo`
cache-only-load contract, extended to BOTH sides (that script still falls
through to a live `prepare_run` for non-beats/tfc variants; this one never
does, for either side, since the whole distillation story is "zero
extraction").

Usage: `distill_beats.py (--run <name> | --runs <a,b,c>) --epochs 30
--batch-size 256 --lr 1e-3 --seed 7 --out models/adapted/` ->
`models/adapted/student_<run>.pt` (multi-run: run names joined with `+`,
e.g. `student_<a>+<b>.pt`) + a sidecar `<same-stem>.json`. Both caches for
every named run must already be warm (`scripts/warm_cache.py --runs <name>
--variants audio-beats logmel`).

Multi-run pooling (Step-2 package-7 Task 8, spec D6 as amended by A3.11:
`docs/superpowers/specs/2026-07-18-step2-package7-robustness-design.md`):
`--runs a,b,c` (mutually exclusive with `--run`) applies the single-run
recipe PER RUN -- both caches loaded cache-only, `_check_cache_alignment`,
`_select_calibration_windows` on that run's own top split -- then STACKS the
selected calibration-side rows across runs (student inputs and teacher
primary-mic slices alike, in `--runs` order) and trains ONE student on the
pooled matrix. The sidecar records the runs list and per-run calibration-
window counts (A3.11); the single-run `--run` path is untouched.

## Cache loading + grid alignment

Both caches are loaded via `rowii.pipeline`'s own public cache primitives
(`_cache_npz_path`/`_cache_fingerprint`/`_load_cached_prepared_run` -- the same
cross-module-privates precedent `scripts/analyze_step2.py`/`scripts/warm_cache.
py` already set) WITH fingerprint verification: a missing file or a
fingerprint mismatch is a cache MISS, and `_load_cache_or_exit` refuses
outright rather than silently recomputing (naming the exact `warm_cache.py`
invocation to fix it).

`audio-beats`' grid (built from BOTH mic streams' intersection,
`rowii.pipeline._streams_for_variant`) and `logmel`'s grid (the primary mic
stream ALONE) are not guaranteed to be byte-identical -- the same physical
reason `scripts/run_step2.py`'s own `--ensemble` guard
(`_check_ensemble_grid_alignment`) tolerates a sub-window offset between a
vibration-bearing sweep variant and `logmel`. The stream-set difference can
also shift `t0_ns` by whole windows AND change `n_windows` (the 010726-pu
real-data case: the turbine mic starts 41 ms after the generator mic, so the
both-mics grid loses the one trailing window the mic-alone grid still fits).
`_check_cache_alignment` therefore requires only `window_ns` to match EXACTLY,
computes the integer window shift `round((student_t0 - teacher_t0) /
window_ns)` pairing student window j with teacher window j + shift, restricts
training to the student-index overlap range where a teacher partner exists
(a warning names every dropped window count -- never a silent trim), and
keeps the residual sub-window offset as a warning. `SystemExit(2)` remains
for a `window_ns` mismatch or grids with NO overlapping windows at all --
either signals a structural inconsistency distillation cannot paper over.

## Leakage rule (spec D3, reused here)

Distillation trains on calibration-side windows ONLY: `rowii.anomaly.
references.split_by_segments` on the `logmel` cache's own `segment_ids`/
`valid_mask` (`calibration_frac=0.5`, `seed` -- default 7, the SAME top-level
split every Step-2 sweep draws its own calibration/scoring windows from for
this run), AND-ed with `audio-beats`' own `valid_mask` (so every drawn
window's teacher target is finite -- see `_select_calibration_windows`'s own
docstring for why this is a safe extension of, not a departure from, D3's
"logmel prepared's segment_ids/valid_mask" rule). A distilled `student_<run>
.pt` checkpoint therefore never trains on a window a LATER within-day sweep
for the SAME run might score against it; the sidecar json restates this
caveat (design's global "adapted/distilled/quantized results always carry
their caveat" rule).

## Training

MSE(student(logmel_window), teacher_beats_embedding), Adam, `--epochs` full
passes over the drawn calibration windows in `--batch-size`-row mini-batches,
shuffled each epoch by a `--seed`-seeded CPU `torch.Generator` -- the SAME
established shuffle pattern as `scripts/pretrain_tfc.py`'s/`rowii.anomaly.
recon`'s own `_train`/`_train_autoencoder`. Deterministic given `--seed` alone
(together with `torch.manual_seed(seed)` for weight init, called BEFORE the
model is constructed) -- verified only on CPU (this project's tests never run
torch training on a non-CPU device; the established MPS/CUDA caveat is
`rowii.anomaly.recon`'s own module docstring).

Torch import discipline (plan's Global Constraints): every torch-touching name
here is imported lazily inside the function that needs it, INCLUDING
`discover`, which is a deliberate exception (mirrors `scripts/adapt_beats.py`'s
own module docstring: a module-level import specifically so
`tests/test_student.py` can `monkeypatch.setattr(distill_beats, "discover",
...)`).
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
    from rowii.adapt._student_model import _StudentNet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.adapt.student import StudentConfig  # noqa: E402
from rowii.anomaly.references import split_by_segments  # noqa: E402
from rowii.config import Config, load_config  # noqa: E402
from rowii.io.dataset import Run, discover, run_utc_offset_ns  # noqa: E402
from rowii.pipeline import (  # noqa: E402
    PreparedRun,
    _cache_fingerprint,
    _cache_npz_path,
    _load_cached_prepared_run,
    _streams_for_variant,
    stream_columns,
)

logger = logging.getLogger(__name__)

_TORCH_HINT = "pip install -e '.[beats]'"
_TEACHER_VARIANT = "audio-beats"
_STUDENT_INPUT_VARIANT = "logmel"
_CALIBRATION_FRAC = 0.5
"""`split_by_segments`' calibration fraction -- the SAME top-level
calibration/scoring split fraction every Step-2 sweep uses (module docstring's
leakage-rule section)."""

def _leakage_note(seed: int) -> str:
    """The sidecar leakage note, SEED-CONDITIONAL (T8-review HIGH: the previous
    static string claimed 'the SAME top-level split every Step-2 within-day
    sweep uses' unconditionally -- true at seed=7, FALSE at any other seed,
    yet persisted verbatim into provenance). At the canonical seed the claim
    stands; at any other seed an explicit does-NOT-match caveat is appended."""
    note = _LEAKAGE_NOTE_BASE.format(seed=seed)
    if seed != 7:
        note += (
            f" CAVEAT: seed={seed} != 7 -- this split does NOT match the canonical "
            "seed-7 split the Step-2 evaluations use; the leakage guarantee holds "
            f"only against seed-{seed} evaluations."
        )
    return note


_LEAKAGE_NOTE_BASE = (
    "distillation trains on calibration-side windows only (rowii.anomaly.references."
    "split_by_segments on the logmel student-input cache's own segment_ids/valid_mask, "
    "AND-ed with the audio-beats teacher cache's own valid_mask, calibration_frac=0.5, "
    "seed={seed}) "
    "-- the SAME top-level split every Step-2 within-day sweep draws its own "
    "calibration/scoring windows from for this run. Any Step-1/Step-2 result computed "
    "from this checkpoint's audio-student variant must restate that the student was "
    "distilled on this run's calibration side. Grid caveat: split membership is drawn "
    "on the logmel cache's grid (primary mic alone), while audio-student SCORING runs "
    "on the both-mic grid (_streams_for_variant('audio-student')); if the two mic "
    "streams are not coextensive for a run, the grids can differ in n_windows and "
    "index-based membership need not correspond (coincides on the single-burst days "
    "this checkpoint was produced and evaluated on)."
)

_MULTI_RUN_NOTE = (
    " Multi-run pool (P7 spec D6/A3.11): the same top-split rule is applied PER RUN "
    "(each run's own segment_ids/valid_mask, shared seed) and the selected "
    "calibration-side rows are stacked across the runs listed in this sidecar -- no "
    "run's scoring-side windows are ever trained on, so held-out evaluation on any "
    "run's scoring side (and, per A3.1, on non-pool runs) stays uncontaminated."
)
"""Appended to `_LEAKAGE_NOTE` in `--runs` mode only -- the single-run sidecar
note stays byte-identical."""


def _import_torch_or_exit() -> None:
    """Distillation training is inherently a torch operation (mirrors `scripts/
    pretrain_tfc.py`'s own `_import_torch_or_exit`) -- this guard runs
    unconditionally, early in `main()`, right after argument parsing."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"distill_beats needs torch ({exc}); {_TORCH_HINT}") from exc


def _parse_runs_or_error(parser: argparse.ArgumentParser, raw: str) -> list[str]:
    """`--runs "a,b,c"` -> `["a", "b", "c"]`, with parse-level validation via
    *parser*`.error` (exit 2, argparse's own convention; P7 spec D6/A3.11):
    whitespace around names is stripped; an empty list (nothing but commas/
    blanks) and duplicate names are both rejected -- a duplicated run would
    stack its calibration rows TWICE into the pooled training matrix,
    silently double-weighting that run's distribution in the student's MSE
    objective. Duplicated from `scripts/adapt_beats.py`'s own helper of the
    same name rather than imported (the plan's Global Constraint, set by
    `scripts/warm_cache.py`'s precedent: one script must not depend on a
    sibling script's internals).
    """
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        parser.error("--runs must name at least one run (comma-separated run names)")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        parser.error(
            f"--runs contains duplicate run name(s): {', '.join(duplicates)} -- each run "
            "may appear once (a duplicate would stack its calibration rows twice into "
            "the pooled training matrix)"
        )
    return names


def _resolve_run_or_exit(run_name: str, cfg: Config) -> Run:
    """*run_name*'s `Run`, discovered under `cfg.data_root` -- mirrors `scripts/
    analyze_step2.py`'s own `_grid_for_combo` run-resolution step.

    Raises:
        SystemExit: *run_name* was not discovered under `cfg.data_root` (names
            every discovered run).
    """
    index = discover(cfg.data_root)
    by_name = {r.name: r for r in index.runs}
    run = by_name.get(run_name)
    if run is None:
        raise SystemExit(
            f"distill_beats: run {run_name!r} not discovered under {cfg.data_root} "
            f"(available: {sorted(by_name)})"
        )
    return run


def _resolve_runs_or_exit(run_names: list[str], cfg: Config) -> list[Run]:
    """Multi-run counterpart of `_resolve_run_or_exit` (P7 spec D6/A3.11):
    ONE `discover` walk resolves every name (per-name `_resolve_run_or_exit`
    calls would re-walk the data tree once per pool member), preserving
    *run_names*' order -- the stacking order downstream.

    Raises:
        SystemExit: any name was not discovered under `cfg.data_root` (names
            every unknown name and every discovered run).
    """
    index = discover(cfg.data_root)
    by_name = {r.name: r for r in index.runs}
    unknown = [name for name in run_names if name not in by_name]
    if unknown:
        raise SystemExit(
            f"distill_beats: run(s) {', '.join(map(repr, unknown))} not discovered under "
            f"{cfg.data_root} (available: {sorted(by_name)})"
        )
    return [by_name[name] for name in run_names]


def _load_cache_or_exit(run: Run, variant: str, cfg: Config) -> PreparedRun:
    """Cache-ONLY load of *run*'s (run, variant) `PreparedRun`, refusing outright
    on a miss rather than falling through to a live `rowii.pipeline.prepare_run`
    extraction (module docstring's "Cache loading + grid alignment" section):
    mirrors `scripts/analyze_step2.py`'s `_grid_for_combo`, extended -- THIS
    script never falls through to a live compute for either *variant* it is
    called with, not just a beats/tfc one.

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
            f"distill_beats: no warm cache for {run.name} x {variant} (expected "
            f"{cache_path} with a matching fingerprint) -- run `python scripts/"
            f"warm_cache.py --runs {run.name} --variants {variant}` first; this "
            "script refuses to trigger a from-scratch extraction silently."
        )
    return cached


@dataclasses.dataclass(frozen=True)
class _CacheAlignment:
    """Window-index pairing between the teacher (`audio-beats`) and student
    (`logmel`) caches of one run (module docstring's "Cache loading + grid
    alignment" section): student window j pairs with teacher window
    `j + shift`; only student indices in `[student_lo, student_hi)` have a
    teacher partner; `t0_offset_ns` is the residual sub-window |t0| skew left
    AFTER the integer shift (always `<= window_ns / 2`)."""

    shift: int
    student_lo: int
    student_hi: int
    t0_offset_ns: int

    def teacher_indices(self, student_indices: np.ndarray) -> np.ndarray:
        """The teacher-cache row for each student-cache window index -- callers
        must only pass indices inside `[student_lo, student_hi)` (which
        `_select_calibration_windows`' pairing mask already guarantees)."""
        return student_indices + self.shift


def _check_cache_alignment(
    run_name: str, teacher: PreparedRun, student_input: PreparedRun
) -> _CacheAlignment:
    """Grid-alignment guard between the *teacher* (`audio-beats`) and
    *student_input* (`logmel`) caches -- mirrors `scripts/run_step2.py`'s own
    `--ensemble` guard (`_check_ensemble_grid_alignment`), generalized (module
    docstring's "Cache loading + grid alignment" section has the full
    rationale): the two grids must share `window_ns` EXACTLY, but the variants'
    different stream sets legitimately shift `t0_ns` (sub-window DAQ skew AND
    whole-window offsets) and change `n_windows` -- the returned
    `_CacheAlignment` pairs student window j with teacher window `j + shift`
    over the overlap range, and every window either side loses to the trim is
    logged, never silently dropped.

    Returns:
        The `_CacheAlignment` pairing the two grids.

    Raises:
        SystemExit: code 2, with a clear message on stderr, if `window_ns`
            differs at all, or the two grids share NO overlapping window --
            either signals a structural inconsistency distillation cannot
            safely paper over.
    """
    tg, sg = teacher.grid, student_input.grid
    if tg.window_ns != sg.window_ns:
        print(
            f"distill_beats: cache grid mismatch for run {run_name!r}: audio-beats "
            f"grid (t0_ns={tg.t0_ns}, window_ns={tg.window_ns}, n_windows={tg.n_windows}) "
            f"!= logmel grid (t0_ns={sg.t0_ns}, window_ns={sg.window_ns}, "
            f"n_windows={sg.n_windows}) -- window_ns must be identical for a window "
            "index in one cache to map onto a window index in the other",
            file=sys.stderr,
        )
        raise SystemExit(2)
    # delta is a difference of nearby ns timestamps (int-exact, small), so the
    # float division below is safely inside double precision.
    delta_ns = sg.t0_ns - tg.t0_ns
    shift = int(round(delta_ns / tg.window_ns))
    t0_offset_ns = abs(delta_ns - shift * tg.window_ns)
    student_lo = max(0, -shift)
    student_hi = min(sg.n_windows, tg.n_windows - shift)
    if student_hi <= student_lo:
        print(
            f"distill_beats: audio-beats/logmel caches for run {run_name!r} share "
            f"no overlapping window: audio-beats grid (t0_ns={tg.t0_ns}, "
            f"n_windows={tg.n_windows}) vs logmel grid (t0_ns={sg.t0_ns}, "
            f"n_windows={sg.n_windows}) at window_ns={tg.window_ns} -- nothing "
            "to distill on",
            file=sys.stderr,
        )
        raise SystemExit(2)
    n_paired = student_hi - student_lo
    student_dropped = sg.n_windows - n_paired
    teacher_dropped = tg.n_windows - n_paired
    if student_dropped or teacher_dropped:
        logger.warning(
            "distill_beats: audio-beats/logmel grids for run %r differ in extent "
            "(the variants' stream sets differ -- module docstring): pairing "
            "student window j with teacher window j%+d over student range "
            "[%d, %d), dropping %d unpaired logmel window(s) and %d unpaired "
            "audio-beats window(s) from distillation",
            run_name, shift, student_lo, student_hi, student_dropped, teacher_dropped,
        )
    if t0_offset_ns > 0:
        logger.warning(
            "distill_beats: audio-beats/logmel caches for run %r are offset by %.1f ms "
            "on a %.0f ms window (minimum per-window overlap %.1f%%) -- proceeding: the "
            "student trains on a teacher target shifted by this sub-window DAQ "
            "stream-start offset relative to its own logmel input",
            run_name, t0_offset_ns / 1e6, tg.window_ns / 1e6,
            (1.0 - t0_offset_ns / tg.window_ns) * 100.0,
        )
    return _CacheAlignment(
        shift=shift, student_lo=student_lo, student_hi=student_hi,
        t0_offset_ns=t0_offset_ns,
    )


def _teacher_target_columns(
    teacher: PreparedRun, stream: str, *, expected_dim: int
) -> np.ndarray:
    """Column indices of *teacher*'s feature matrix belonging to *stream*.

    The `audio-beats` teacher cache concatenates BOTH mic streams' 768-d
    embeddings (column order = `_streams_for_variant("audio-beats")`, names
    `"<stream>::<local>"`), while the `logmel` student input covers the primary
    mic alone -- the distillation target is therefore the primary-mic SLICE of
    the teacher matrix, never the full concatenation (regressing a 768-d
    student head onto the 1536-d full matrix was the package-5 execution's
    first real-data failure). Columns are selected by feature-name prefix, not
    by position, so a stream-order change in the cache breaks loudly here
    instead of silently mis-slicing; a slice width that disagrees with the
    student's `out_dim` is likewise a hard exit, not a broadcast. Column
    selection itself is `rowii.pipeline.stream_columns` (shared with
    `scripts/train_xattn.py`'s audio-query slice and `scripts/run_step2.py`'s
    `--xattn-fusion` view -- ONE definition of "this stream's block"); this
    wrapper only adds the CLI's exit-code semantics and the out_dim check.
    """
    try:
        cols = stream_columns(teacher.feature_names, stream)
    except ValueError:
        print(
            f"distill_beats: teacher cache has no columns for stream {stream!r} "
            f"(no feature name starts with '{stream}::') -- cannot slice the "
            "distillation target",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    if int(cols.size) != expected_dim:
        print(
            f"distill_beats: teacher slice for stream {stream!r} has {cols.size} "
            f"column(s) but the student's out_dim is {expected_dim} -- geometry "
            "mismatch between teacher cache and student head",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return cols


def _select_calibration_windows(
    student_input: PreparedRun,
    teacher: PreparedRun,
    *,
    seed: int,
    alignment: _CacheAlignment,
) -> np.ndarray:
    """The leakage-safe calibration-side window indices to distill on (spec D3's
    rule, reused here per the module docstring's "Leakage rule" section):
    `split_by_segments` on the LOGMEL *student_input* cache's own `segment_ids`/
    `valid_mask` (`_CALIBRATION_FRAC`, *seed*) -- the SAME top-level
    calibration/scoring split every Step-2 sweep draws for this run, so a
    distilled student checkpoint never trains on a window a later sweep might
    score against it.

    The valid mask fed to `split_by_segments` is the AND of both caches' own
    `valid_mask` under *alignment*'s pairing (a deliberate, documented
    extension beyond a literal "logmel's own mask" reading): after
    `_check_cache_alignment` establishes which teacher window (`j + shift`)
    each student window j refers to, a window can still be coverage-valid for
    `logmel`'s single primary-mic stream while `audio-beats`' own validity
    (BOTH mic streams, `rowii.pipeline._streams_for_variant("audio-beats")`)
    says otherwise (e.g. the turbine mic missing coverage there), and a
    student window OUTSIDE `[student_lo, student_hi)` has no teacher partner
    at all. Training on such a window would regress the student against a
    teacher target that is undefined (NaN) or nonexistent at that index --
    the combined mask keeps every drawn window's teacher target finite and
    real, without weakening the leakage rule itself (segment ids AND the
    scoring-side exclusion still come from the logmel cache alone, per D3).

    Returns:
        `(N,)` int64 ascending STUDENT-cache window indices, valid in both
        caches and inside *alignment*'s paired range -- the matching teacher
        rows are `alignment.teacher_indices(result)`.
    """
    lo, hi = alignment.student_lo, alignment.student_hi
    teacher_valid_at_student = np.zeros(student_input.grid.n_windows, dtype=bool)
    teacher_valid_at_student[lo:hi] = teacher.valid_mask[
        lo + alignment.shift : hi + alignment.shift
    ]
    combined_valid = student_input.valid_mask & teacher_valid_at_student
    split = split_by_segments(student_input.segment_ids, combined_valid, _CALIBRATION_FRAC, seed)
    return split.calibration_windows


def _train_student(
    student_inputs: np.ndarray,
    teacher_targets: np.ndarray,
    cfg: StudentConfig,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[_StudentNet, list[float]]:
    """MSE(student(logmel_window), teacher_beats_embedding) distillation
    training (module docstring's "Training" section): Adam, *epochs* full
    passes over *student_inputs*/*teacher_targets* in *batch_size*-row
    mini-batches, shuffled each epoch by a *seed*-seeded CPU `torch.Generator`
    -- the SAME established pattern as `scripts/pretrain_tfc.py`'s own
    `_train`. Deterministic given *seed* alone (together with
    `torch.manual_seed(seed)` for weight init, called BEFORE the model is
    constructed).

    Args:
        student_inputs: `(N, n_frames * n_mels)` float64 flattened log-mel
            patches (calibration-side windows only -- caller's
            responsibility, `_select_calibration_windows`).
        teacher_targets: `(N, out_dim)` float64 frozen-BEATs embeddings, row i
            paired with `student_inputs[i]` (the SAME window).
        cfg: Student architecture (`StudentConfig()` in every real run this
            script performs -- no CLI flag varies it).
        epochs: Full passes over *student_inputs*.
        batch_size: Rows per gradient step (the last batch of an epoch may be
            smaller, `len(student_inputs) % batch_size`).
        lr: Adam learning rate.
        seed: Seeds `torch.manual_seed` (weight init) AND the shared shuffle
            generator.

    Returns:
        `(model, epoch_losses)`: *model* on `best_device()`, in `.eval()` mode
        (mirrors `_train_autoencoder`'s/`pretrain_tfc._train`'s own
        postcondition -- every downstream embedding call reads it under
        `torch.no_grad()`); *epoch_losses* is one MEAN loss per epoch, in
        order, for the caller to log.

    Raises:
        ValueError: *student_inputs*/*teacher_targets* have different row
            counts.
    """
    import torch

    from rowii.adapt._student_model import _StudentNet
    from rowii.signals.beats import best_device

    if student_inputs.shape[0] != teacher_targets.shape[0]:
        raise ValueError(
            f"student_inputs.shape[0] ({student_inputs.shape[0]}) must equal "
            f"teacher_targets.shape[0] ({teacher_targets.shape[0]})"
        )

    torch.manual_seed(seed)
    device = best_device()
    model = _StudentNet(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x_all = torch.from_numpy(student_inputs.astype(np.float32)).to(device)
    y_all = torch.from_numpy(teacher_targets.astype(np.float32)).to(device)
    n = x_all.shape[0]
    gen = torch.Generator().manual_seed(seed)

    epoch_losses: list[float] = []
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            pred = model(x_all[idx])
            loss = torch.nn.functional.mse_loss(pred, y_all[idx])
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        mean_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(mean_loss)
        logger.info(
            "distill_beats: epoch %d/%d -- mean loss %.6f", epoch + 1, epochs, mean_loss
        )

    model.eval()
    return model, epoch_losses


def _save_checkpoint(
    path: Path,
    cfg: StudentConfig,
    model: _StudentNet,
    teacher_variant: str,
    run_name: str,
    epochs: int,
) -> None:
    """Write the `rowii.adapt.student.load_student_model`-format checkpoint
    dict (that function's docstring) to *path*, creating parent directories as
    needed."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": dataclasses.asdict(cfg),
            "model": model.state_dict(),
            "teacher_variant": teacher_variant,
            "run": run_name,
            "epochs": epochs,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Distill the frozen BEATs teacher's ALREADY-CACHED embeddings "
            "(results/cache/<run>--audio-beats.npz) into a compact CNN student on "
            "the logmel variant's ALREADY-CACHED input patches "
            "(results/cache/<run>--logmel.npz) -- zero teacher/extraction compute "
            "(package-5 spec D5, Task 4). Writes models/adapted/student_<run>.pt "
            "+ a sidecar <run>.json."
        )
    )
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument(
        "--run", metavar="NAME",
        help="Run name to distill from -- needs BOTH audio-beats and logmel warm "
             "caches (scripts/warm_cache.py --runs <name> --variants audio-beats logmel).",
    )
    run_group.add_argument(
        "--runs", metavar="NAMES",
        help="Comma-separated run names for MULTI-run distillation (P7 spec D6/A3.11): "
             "per run, both caches are loaded (cache-only) and alignment-checked, the "
             "per-run top split selects calibration-side rows, and the selected student "
             "inputs + teacher primary-mic targets are STACKED across runs into one "
             "training set; the checkpoint name joins the run names with '+' and the "
             "sidecar records per-run calibration-window counts. Mutually exclusive "
             "with --run.",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Training epochs (default: 30).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Mini-batch size (default: 256).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Adam learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--seed", type=int, default=7,
        help="Seeds weight init, shuffling, AND the per-run calibration/scoring "
             "SPLIT (default: 7 -- the canonical seed every Step-2 sweep uses). A "
             "non-7 seed voids the leakage guarantee vs the seed-7 evaluations "
             "(the sidecar note then carries an explicit caveat); pretrain_tfc "
             "pins its split at 7, this CLI keeps the shipped P5 contract and "
             "WARNS instead.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("models/adapted/"),
        help="Output directory (default: models/adapted/); filename is "
             "student_<run>.pt.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.seed != 7:
        logger.warning(
            "distill_beats: --seed %d != 7 changes the per-run calibration/scoring "
            "SPLIT -- the checkpoint's leakage guarantee holds only against a "
            "seed-%d evaluation, NOT the canonical seed-7 Step-2 sweeps; the "
            "sidecar note carries this caveat (T8-review seed-tension resolution)",
            args.seed, args.seed,
        )

    # Parse-level --runs validation (empty/duplicates) runs BEFORE the torch
    # import guard: a malformed flag value is a usage error regardless of the
    # environment's torch install (mirrors scripts/adapt_beats.py).
    run_names: list[str] = []
    if args.runs is not None:
        run_names = _parse_runs_or_error(parser, args.runs)

    _import_torch_or_exit()

    cfg = load_config()

    # Pure constructions, shared by both paths (StudentConfig carries no state
    # beyond its defaults; the primary stream is a variant-level constant).
    student_cfg = StudentConfig()
    primary_stream = _streams_for_variant(_STUDENT_INPUT_VARIANT)[0]

    t0: float
    calibration_windows_per_run: dict[str, int] | None = None
    if args.runs is not None:
        # Multi-run pool (P7 spec D6/A3.11 -- module docstring): the
        # single-run recipe applied PER RUN (cache-only loads, alignment
        # guard, per-run top split), the selected calibration-side rows
        # stacked across runs in --runs order, ONE student trained on the
        # pooled matrix.
        pool_runs = _resolve_runs_or_exit(run_names, cfg)
        t0 = time.monotonic()
        student_blocks: list[np.ndarray] = []
        teacher_blocks: list[np.ndarray] = []
        calibration_windows_per_run = {}
        for run in pool_runs:
            teacher = _load_cache_or_exit(run, _TEACHER_VARIANT, cfg)
            student_input = _load_cache_or_exit(run, _STUDENT_INPUT_VARIANT, cfg)
            alignment = _check_cache_alignment(run.name, teacher, student_input)
            calibration_windows = _select_calibration_windows(
                student_input, teacher, seed=args.seed, alignment=alignment
            )
            teacher_cols = _teacher_target_columns(
                teacher, primary_stream, expected_dim=student_cfg.out_dim
            )
            student_blocks.append(student_input.features[calibration_windows])
            teacher_blocks.append(
                teacher.features[alignment.teacher_indices(calibration_windows)][
                    :, teacher_cols
                ]
            )
            # Zero-contribution runs stay visible in the counts (A4.1's
            # coverage-visibility principle: never silently absent) AND get a
            # runtime warning PER RUN -- the whole-branch review caught this
            # check sitting outside the loop, where it only ever saw the last
            # run's count.
            calibration_windows_per_run[run.name] = int(calibration_windows.size)
            if calibration_windows.size == 0:
                logger.warning(
                    "distill_beats: pool run %s contributed ZERO calibration-side "
                    "windows", run.name,
                )
        student_inputs = np.vstack(student_blocks)
        teacher_targets = np.vstack(teacher_blocks)
        if student_inputs.shape[0] == 0:
            print(
                f"distill_beats: no calibration-side window(s) across runs "
                f"{', '.join(run_names)} -- nothing to distill on",
                file=sys.stderr,
            )
            return 1
        label = "+".join(run_names)
        logger.info(
            "distill_beats: %d calibration-side window(s) stacked across %d run(s) "
            "(per-run counts: %s); teacher target = %s slice of the audio-beats cache",
            student_inputs.shape[0], len(pool_runs), calibration_windows_per_run,
            primary_stream,
        )
    else:
        run = _resolve_run_or_exit(args.run, cfg)
        t0 = time.monotonic()
        teacher = _load_cache_or_exit(run, _TEACHER_VARIANT, cfg)
        student_input = _load_cache_or_exit(run, _STUDENT_INPUT_VARIANT, cfg)

        alignment = _check_cache_alignment(run.name, teacher, student_input)

        calibration_windows = _select_calibration_windows(
            student_input, teacher, seed=args.seed, alignment=alignment
        )
        if calibration_windows.size == 0:
            print(
                f"distill_beats: no calibration-side window(s) for run {run.name!r} -- "
                "nothing to distill on",
                file=sys.stderr,
            )
            return 1

        teacher_cols = _teacher_target_columns(
            teacher, primary_stream, expected_dim=student_cfg.out_dim
        )
        student_inputs = student_input.features[calibration_windows]
        teacher_targets = teacher.features[alignment.teacher_indices(calibration_windows)][
            :, teacher_cols
        ]
        label = run.name
        logger.info(
            "distill_beats: %d calibration-side window(s) for run %s (of %d/%d valid); "
            "teacher target = %d-column %s slice of the audio-beats cache",
            student_inputs.shape[0], run.name,
            int(student_input.valid_mask.sum()), int(teacher.valid_mask.sum()),
            int(teacher_cols.size), primary_stream,
        )

    model, epoch_losses = _train_student(
        student_inputs, teacher_targets, student_cfg,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
    )

    checkpoint_path = args.out / f"student_{label}.pt"
    _save_checkpoint(checkpoint_path, student_cfg, model, _TEACHER_VARIANT, label, args.epochs)
    elapsed_s = time.monotonic() - t0

    # Provenance block: the single-run sidecar keeps its exact historical
    # shape ("run", unchanged note); a multi-run sidecar instead carries the
    # runs list, the A3.11-required per-run calibration-window counts, and
    # the pool-suffixed leakage note.
    if calibration_windows_per_run is None:
        provenance: dict[str, object] = {"run": label}
        note = _leakage_note(args.seed)
    else:
        provenance = {
            "runs": run_names,
            "calibration_windows_per_run": calibration_windows_per_run,
        }
        note = _leakage_note(args.seed) + _MULTI_RUN_NOTE
    sidecar: dict[str, object] = {
        **provenance,
        "teacher_variant": _TEACHER_VARIANT,
        "teacher_stream": primary_stream,
        "student_input_variant": _STUDENT_INPUT_VARIANT,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "n_calibration_windows": int(student_inputs.shape[0]),
        "final_loss": epoch_losses[-1],
        "elapsed_s": elapsed_s,
        "note": note,
    }
    sidecar_path = checkpoint_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(
        f"distill_beats: saved {checkpoint_path} ({args.epochs} epoch(s) over "
        f"{student_inputs.shape[0]} calibration-side window(s), final mean loss "
        f"{epoch_losses[-1]:.6f}); sidecar {sidecar_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
