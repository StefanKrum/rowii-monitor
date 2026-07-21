"""Leakage-aware target-normal training-window iterator (Step-2 package-5
Task 2, design spec D3: `docs/superpowers/specs/2026-07-16-step2-package5-
adaptation-design.md`).

`iter_target_windows` feeds `scripts/adapt_beats.py`'s masked-token
adaptation objective (Task 1 as amended by spec Amendment A1,
`rowii.adapt.objective.masked_token_loss`)
1-second windows of a run's PRIMARY MIC stream (`RAWGeneratorMic__0` --
`PreparedRun.segment_ids`'s own primary-stream convention for the "audio"
variant), resampled to BEATs' 16 kHz input rate, drawn ONLY from the
CALIBRATION side of the top split every Step-2 sweep already uses --
`split_by_segments(segment_ids, valid_mask, 0.5, seed=7)`
(`rowii.anomaly.references`). This is the package's leakage rule (spec D3,
restated wherever adapted-model evidence appears): adaptation must never see
a scoring-side segment, or the LATER evaluation of the adapted encoder on
that same segment would be contaminated by having trained on it.

Reuses `rowii.pipeline.prepare_run(run, "audio", cfg, use_cache=True)` for
`grid`/`valid_mask`/`segment_ids` instead of re-deriving them -- the "audio"
variant's primary stream already IS the generator mic, so no separate
single-stream grid/discovery logic is needed, and in practice this is a
cache HIT (the "audio" variant's cache is warm from every earlier Step-1/
Step-2 sweep on the same run). Raw sample slices are then read directly off
the primary stream's own gantner files (bypassing `AudioFeaturizer`/
`BeatsFeaturizer` entirely -- this module hands back RAW 16 kHz waveforms,
not handcrafted or BEATs features), each touched burst file opened via
`read_gantner` at most ONCE (`_iter_indexed_windows`'s read-once cache) --
mirroring `rowii.pipeline._extract_stream_features`'s own
one-file-resident-at-a-time memory discipline (module docstring: "The
pipeline NEVER concatenates a whole stream into memory"; at a file-boundary
window at most TWO files are transiently resident here, see
`_iter_indexed_windows`).

Content-source verification (Task-2 review HIGH): `PreparedRun.segment_ids`
attributes a window to the EARLIEST file with ANY sample overlap
(`_extract_stream_features`'s documented convention) -- which, for a window
straddling a file boundary inside the jitter band, is NOT necessarily the
file whose FULL slice actually produced the window's feature row and
validity. Trusting the attribution blindly would read a near-empty sliver
and zero-pad it to length (reviewer counterexample: a boundary 3 samples
into a window yields a ~99%-padding "window" labeled valid). So before
cutting a window from a file, its slice length is verified against the SAME
`+/-_SAMPLE_JITTER_TOLERANCE` acceptance band `_extract_stream_features`
uses (imported, not duplicated -- the two must stay in lockstep); on
failure the NEXT file is probed, and a window no file covers fully is
SKIPPED with a debug log, never yielded as padded content.

Torch-free by design (numpy/scipy only): `scripts/adapt_beats.py` (an eager-
torch script, Task 3) consumes this module's output, but `target_windows.py`
itself is NOT on the plan's Global-Constraints eager-torch module list
(`objective.py`/`lora.py`/`student.py`'s model part/`fusionx/model.py`) -- it
has no reason to import torch at all, so it stays importable (and
independently testable) without the optional `[beats]` extra installed.
"""
from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
from scipy.signal import resample_poly

from rowii.anomaly.references import split_by_segments
from rowii.config import Config
from rowii.io.dataset import Run, run_utc_offset_ns
from rowii.io.gantner import read_gantner

# _SAMPLE_JITTER_TOLERANCE is deliberately IMPORTED (a private name, like
# `rowii.anomaly.recon`'s import of `scorers._check_query`) rather than
# duplicated: this module's full-window acceptance test must stay in exact
# lockstep with `_extract_stream_features`'s own -- a drifted copy would
# silently re-open the very content-misattribution bug class the check
# exists to close (Task-2 review HIGH; module docstring).
from rowii.pipeline import _SAMPLE_JITTER_TOLERANCE, prepare_run
from rowii.signals.windows import window_slices

logger = logging.getLogger(__name__)

_PRIMARY_MIC_STREAM = "RAWGeneratorMic__0"
"""The "audio" variant's primary stream (`rowii.pipeline._AUDIO_STREAMS[0]`,
restated in `PreparedRun.segment_ids`'s own docstring): `prepare_run(run,
"audio", cfg)` derives `segment_ids` from exactly this stream's own
time-sorted file list, so indexing `run.files[_PRIMARY_MIC_STREAM]` (sorted
the SAME way -- `start_utc_hint` ascending) with a `segment_ids` value lands
on the exact file `rowii.pipeline._extract_stream_features` attributed that
window to. Duplicated here rather than importing `rowii.pipeline`'s private
`_AUDIO_STREAMS` constant -- this module only ever needs the ONE stream, and
a stream NAME is a stable, self-describing literal (unlike the jitter
tolerance above, where correctness depends on staying in lockstep with the
pipeline's own value, hence imported)."""

_TOP_SPLIT_FRAC = 0.5
"""Calibration fraction of the top split -- fixed across every Step-2 sweep
and every package-5 adaptation consumer (spec D3), never a caller-tunable
knob (unlike `seed`, which a caller may deliberately vary)."""


def _shift_ts_ns(ts_ns: np.ndarray, offset_ns: int) -> np.ndarray:
    """`ts_ns + offset_ns`, staying in `uint64` throughout.

    Duplicated from `rowii.pipeline._shift_ts_ns` (itself duplicated from
    `rowii.scada.labels._shift_ts_ns`) rather than imported -- a small,
    self-contained utility, not worth a new cross-module dependency for
    (same rationale that function's own docstring gives). `ts_ns` is
    `uint64` (as produced by `read_gantner`) and so is `WindowGrid.
    edges_ns()`, which `window_slices` compares it against; mixing `int64`
    and `uint64` numpy arrays silently upcasts BOTH to `float64`, losing
    precision above `2**53` (~9.007e15) -- far below these ~1e18 ns
    timestamps, so a naive `ts_ns.astype(np.int64) + offset_ns` would
    corrupt every window boundary.
    """
    if offset_ns >= 0:
        return ts_ns + np.uint64(offset_ns)
    return ts_ns - np.uint64(-offset_ns)


def _pad_trim_1d(x: np.ndarray, n: int) -> np.ndarray:
    """Zero-pad or trim 1-D *x* to exactly *n* samples."""
    if x.shape[0] < n:
        return np.pad(x, (0, n - x.shape[0]))
    if x.shape[0] > n:
        return x[:n]
    return x


def _resample_window(mono: np.ndarray, rate_hz: float, target_hz: int) -> np.ndarray:
    """One 1-D *mono* window at *rate_hz* -> exactly *target_hz* samples.

    Mirrors `rowii.tfc.wrapper._resample_to_8khz`'s two-sided pad/trim
    convention (cross-reference; that function's own docstring has the full
    rationale), generalized from TF-C's hardcoded 8 kHz target to an
    arbitrary *target_hz* (BEATs' 16 kHz here):

    1. BEFORE resampling, pad/trim *mono* to exactly `round(rate_hz)`
       samples -- `resample_poly(x, up, down)`'s output length is
       `ceil(len(x) * up / down)`; feeding it exactly `round(rate_hz)` input
       samples with `down = round(rate_hz)` collapses that formula to
       `ceil(up) = up = target_hz` exactly, regardless of a real window's
       actual sample count being off by a few samples from the nominal rate
       (DAQ clock jitter -- every window reaching this function was already
       verified to hold a FULL slice, within `+/-_SAMPLE_JITTER_TOLERANCE`
       of its source file's expected count, by `_iter_indexed_windows`'s
       content-source check, so this pad/trim only ever absorbs that
       residual few-sample jitter, never a genuinely partial slice).
    2. AFTER resampling, pad/trim again to exactly *target_hz* samples,
       defensively -- `resample_poly`'s FIR filtering has edge effects and
       `int(round(rate_hz))` truncates a fractional rate, so this second
       step is what actually GUARANTEES the output-length invariant every
       caller depends on, rather than merely making it likely.
    """
    source_n = int(round(rate_hz))
    padded = _pad_trim_1d(mono, source_n)
    resampled = resample_poly(padded, target_hz, source_n)
    return np.ascontiguousarray(_pad_trim_1d(resampled, target_hz), dtype=np.float64)


def _standardize_1d(x: np.ndarray) -> np.ndarray:
    """Zero-mean/unit-std standardization, `1e-8`-floored std (silent-window
    divide-by-zero guard) -- same floor convention as `rowii.tfc.wrapper.
    _standardize`/`rowii.tfc.model.freq_view`."""
    mean = x.mean()
    std = np.clip(x.std(), 1e-8, None)
    return (x - mean) / std


@dataclass
class _LoadedPrimaryFile:
    """One primary-mic burst file's raw payload plus everything needed to
    verify and cut windows from it -- `_iter_indexed_windows`'s read-once
    cache entry."""

    data: np.ndarray
    """(n_frames, C) float32 -- the file's raw samples (`GantnerFile.data`)."""
    slices: list[slice]
    """Per-grid-window sample slice into `data`, on the shared true-UTC axis
    (`window_slices` over the file's offset-shifted timestamps)."""
    rate_hz: float
    """The file's own estimated sample rate (`GantnerHeader.sample_rate_hz`)."""
    expected_samples: int
    """`round(rate_hz * window_s)` -- the full-window sample count this
    file's slices are checked against (per-file, from its OWN rate, exactly
    as `_extract_stream_features` computes it)."""


def _iter_indexed_windows(
    run: Run,
    cfg: Config,
    *,
    target_hz: int,
    seed: int,
    max_windows: int | None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Shared implementation behind `iter_target_windows`'s two overloads:
    always yields `(window_index, window)` pairs, ascending by index --
    `iter_target_windows` itself strips the index off unless the caller asks
    for it (`return_indices=True`).

    Content-source verification (module docstring, Task-2 review HIGH): for
    each selected window `w`, candidate source files are probed in the order
    `(segment_ids[w], segment_ids[w] + 1)` and the FIRST one whose slice for
    `w` is full -- `abs(len - expected) <= _SAMPLE_JITTER_TOLERANCE`, the
    identical acceptance band `_extract_stream_features` featurizes under --
    supplies the samples. Files EARLIER than `segment_ids[w]` need no probe:
    the attribution is, by construction, the earliest file (in the same
    sorted order used here) with ANY non-empty slice for `w`
    (`_extract_stream_features`'s never-overwrite attribution loop), so every
    earlier file's slice for `w` is EMPTY -- and an empty slice can never be
    within the jitter band of a full window's expected count. A window for
    which neither candidate holds a full slice is skipped with a debug log
    (it cannot be cut faithfully; yielding it zero-padded would hand the
    adaptation objective fabricated silence) -- unreachable for windows that
    came through the real `valid_mask`-filtered split, whose validity
    REQUIRES a featurized (i.e. full-slice) row somewhere, but pinned
    defensively for hand-built/monkeypatched splits.

    Read-once/memory discipline: each probed file is loaded at most once per
    iteration (`loaded` cache) and evicted as soon as the window sequence has
    advanced past it (attributions are non-decreasing in `w` for contiguous,
    non-overlapping bursts -- the pipeline's own discovery contract -- so a
    file with index below the current window's attribution can never be a
    candidate again). Steady state keeps ONE file resident; a boundary
    window's fallback probe transiently holds two.
    """
    prepared = prepare_run(run, "audio", cfg, use_cache=True)
    top_split = split_by_segments(
        prepared.segment_ids, prepared.valid_mask, _TOP_SPLIT_FRAC, seed
    )
    calibration_windows = top_split.calibration_windows
    if max_windows is not None:
        # Truncate the (already ascending -- SegmentSplit's own contract)
        # index array itself BEFORE any file is touched, so a small
        # max_windows also bounds how many burst files get opened, not just
        # how many windows get yielded.
        calibration_windows = calibration_windows[:max_windows]

    primary_files = sorted(run.files[_PRIMARY_MIC_STREAM], key=lambda f: f.start_utc_hint)
    offset_ns = run_utc_offset_ns(run)
    window_s = prepared.grid.window_ns / 1e9
    loaded: dict[int, _LoadedPrimaryFile] = {}

    def _load(file_index: int) -> _LoadedPrimaryFile:
        entry = loaded.get(file_index)
        if entry is not None:
            return entry
        gf = read_gantner(primary_files[file_index].path)
        rate_hz = gf.header.sample_rate_hz
        ts_ns = _shift_ts_ns(gf.timestamps_ns, offset_ns) if offset_ns else gf.timestamps_ns
        entry = _LoadedPrimaryFile(
            data=gf.data,
            slices=window_slices(ts_ns, prepared.grid),
            rate_hz=rate_hz,
            expected_samples=round(rate_hz * window_s),
        )
        loaded[file_index] = entry
        return entry

    for w in calibration_windows.tolist():
        seg = int(prepared.segment_ids[w])

        # Evict files the ascending window sequence has moved past (index
        # below the current attribution -- never a candidate again, see the
        # docstring's read-once/memory paragraph), freeing their raw data
        # before the next file loads (the pipeline's own per-file gc
        # discipline, `_extract_stream_features`).
        stale = [i for i in loaded if i < seg]
        if stale:
            for i in stale:
                del loaded[i]
            gc.collect()

        source: tuple[_LoadedPrimaryFile, slice] | None = None
        for cand in (seg, seg + 1):
            if not 0 <= cand < len(primary_files):
                continue
            entry = _load(cand)
            sl = entry.slices[w]
            if abs((sl.stop - sl.start) - entry.expected_samples) <= _SAMPLE_JITTER_TOLERANCE:
                source = (entry, sl)
                break
        if source is None:
            logger.debug(
                "iter_target_windows: skipping window %d of run %s -- no primary-mic file "
                "holds a full slice for it (within +/-%d samples of its file's expected "
                "count; candidates probed: file %d and, if present, file %d)",
                w, run.name, _SAMPLE_JITTER_TOLERANCE, seg, seg + 1,
            )
            continue

        entry, sl = source
        raw = entry.data[sl.start : sl.stop, :]
        mono = raw.mean(axis=1).astype(np.float64)
        resampled = _resample_window(mono, entry.rate_hz, target_hz)
        standardized = _standardize_1d(resampled)
        yield w, standardized.astype(np.float32)


@overload
def iter_target_windows(
    run: Run,
    cfg: Config,
    *,
    target_hz: int = 16_000,
    seed: int = 7,
    max_windows: int | None = None,
    return_indices: Literal[False] = False,
) -> Iterator[np.ndarray]: ...


@overload
def iter_target_windows(
    run: Run,
    cfg: Config,
    *,
    target_hz: int = 16_000,
    seed: int = 7,
    max_windows: int | None = None,
    return_indices: Literal[True],
) -> Iterator[tuple[int, np.ndarray]]: ...


def iter_target_windows(
    run: Run,
    cfg: Config,
    *,
    target_hz: int = 16_000,
    seed: int = 7,
    max_windows: int | None = None,
    return_indices: bool = False,
) -> Iterator[np.ndarray] | Iterator[tuple[int, np.ndarray]]:
    """1-s target-NORMAL windows of *run*'s primary mic stream, resampled to
    *target_hz*, drawn ONLY from the top split's calibration side (spec D3
    leakage rule -- module docstring).

    Pipeline per selected window: locate the file actually holding its FULL
    sample slice (verified against the pipeline's own jitter band -- see
    `_iter_indexed_windows` for the probe order and the skip-on-no-full-
    slice rule), reading each touched file once (`read_gantner`) -> slice
    out this window's samples (`rowii.signals.windows.window_slices` against
    `prepare_run`'s own grid) -> mono-mix channels (mean) -> resample to
    *target_hz* (`_resample_window`, `scipy.signal.resample_poly`) ->
    per-window standardize (`_standardize_1d`) -> `float32`.

    Args:
        run: The run to draw target-normal windows from.
        cfg: Project configuration (`prepare_run`'s own `cfg.window.
            window_s`/`cfg.results_root`/cache-fingerprint fields).
        target_hz: Output sample rate AND (since every window is exactly 1 s,
            this project's fixed window duration) output sample COUNT --
            BEATs' own input rate, 16 kHz, by default.
        seed: Seed for the top split's segment shuffle (`rowii.anomaly.
            references.split_by_segments`) -- 7 is the SAME seed every other
            Step-2 top split in this codebase uses (`scripts.
            run_step2_scarcity._TOP_SEED`, `rowii.anomaly.sweep`'s default
            `cfg.seed`); pass a different value only for a deliberately
            different split.
        max_windows: If given, stop after yielding this many windows --
            applied by truncating the (already ascending) calibration-side
            window-index array itself BEFORE any file is opened, so a small
            *max_windows* also bounds how many burst files get read, not just
            how many windows get yielded. `0` is valid (yields nothing,
            touches no file). `None` (default) yields every calibration-side
            window.
        return_indices: If `True`, yield `(window_index, window)` pairs
            instead of bare arrays -- the window index is otherwise
            unobservable from the outside (this module never encodes it into
            the returned samples), so this is the intended way for a caller
            (a test asserting the leakage rule; a future consumer that wants
            to cross-reference a window against `PreparedRun.segment_ids`) to
            know exactly which grid window a given array came from, without a
            second, parallel computation of the same split. `False` (default)
            matches this function's base contract: a plain
            `Iterator[np.ndarray]`.

    Yields:
        `(target_hz,)`-shaped `float32` arrays (or `(window_index, array)`
        pairs when *return_indices* is `True`), in ascending window-index
        order, deterministic for a fixed (*run*, *cfg*, *seed*, *target_hz*,
        *max_windows*). A selected window whose full slice cannot be located
        in any candidate file is skipped (debug-logged), never yielded padded
        -- see `_iter_indexed_windows`.

    Raises:
        ValueError: if *max_windows* is negative -- numpy's `[:negative]`
            slicing would otherwise silently drop windows from the END of
            the calibration set instead of truncating to the first N.
    """
    if max_windows is not None and max_windows < 0:
        raise ValueError(
            f"max_windows must be None or >= 0, got {max_windows!r} -- a negative value "
            "would silently slice from the END of the calibration-side index array "
            "(numpy slicing semantics), the opposite of 'first N windows'"
        )
    indexed = _iter_indexed_windows(
        run, cfg, target_hz=target_hz, seed=seed, max_windows=max_windows
    )
    if return_indices:
        return indexed
    return (window for _, window in indexed)


def _iter_windows_multi(
    runs: list[Run],
    cfg: Config,
    *,
    max_windows: int,
    target_hz: int,
    seed: int,
) -> Iterator[tuple[str, np.ndarray]]:
    """Shared implementation behind `iter_target_windows_multi`'s two
    overloads: always yields `(run_name, window)` pairs in rotation order --
    the public function strips the name off unless the caller asks for it
    (`return_run_names=True`), mirroring `iter_target_windows`'s own
    `_iter_indexed_windows` split.

    A generator function on purpose: the `max_windows == 0` guard and the
    per-run iterator construction only run on first `next()`, so a zero
    budget constructs NO per-run iterator at all (and therefore -- via
    `iter_target_windows`'s own truncate-before-touching-files contract --
    opens no burst file).
    """
    if max_windows == 0:
        return
    rotation: list[tuple[str, Iterator[np.ndarray]]] = [
        (
            run.name,
            iter_target_windows(
                run, cfg, target_hz=target_hz, seed=seed, max_windows=max_windows
            ),
        )
        for run in runs
    ]
    n_yielded = 0
    yielded_per_run: dict[str, int] = {name: 0 for name, _ in rotation}
    while rotation:
        survivors: list[tuple[str, Iterator[np.ndarray]]] = []
        for name, windows in rotation:
            window = next(windows, None)
            if window is None:
                # Exhausted -- this run drops out of the rotation for good
                # (never re-probed: iter_target_windows yields each
                # calibration-side window exactly once). A run that never
                # contributed ANY window is warned about here (T8-review
                # MEDIUM: sidecar-only visibility is not a runtime signal),
                # matching the A4.1 nothing-drops-silently posture.
                if yielded_per_run[name] == 0:
                    logger.warning(
                        "iter_target_windows_multi: pool run %s contributed ZERO "
                        "calibration-side windows", name,
                    )
                continue
            survivors.append((name, windows))
            yield name, window
            yielded_per_run[name] += 1
            n_yielded += 1
            if n_yielded >= max_windows:
                return
        rotation = survivors


@overload
def iter_target_windows_multi(
    runs: list[Run],
    cfg: Config,
    *,
    max_windows: int,
    target_hz: int,
    seed: int = 7,
    return_run_names: Literal[False] = False,
) -> Iterator[np.ndarray]: ...


@overload
def iter_target_windows_multi(
    runs: list[Run],
    cfg: Config,
    *,
    max_windows: int,
    target_hz: int,
    seed: int = 7,
    return_run_names: Literal[True],
) -> Iterator[tuple[str, np.ndarray]]: ...


def iter_target_windows_multi(
    runs: list[Run],
    cfg: Config,
    *,
    max_windows: int,
    target_hz: int,
    seed: int = 7,
    return_run_names: bool = False,
) -> Iterator[np.ndarray] | Iterator[tuple[str, np.ndarray]]:
    """Round-robin multi-run pooling of `iter_target_windows` (Step-2
    package-7 Task 8, spec D6 as amended by A3.11: `docs/superpowers/specs/
    2026-07-18-step2-package7-robustness-design.md`): target-normal windows
    drawn from EVERY run in *runs*, interleaved one window per run per
    rotation pass (a, b, c, a, b, c, ...), stopping after *max_windows*
    TOTAL windows.

    Budget semantics (A3.11, binding): *max_windows* is the TOTAL pool
    budget, NOT a per-run cap. Round-robin is what makes a total budget
    meaningful for a pool: a sequential chain (all of run 1, then run 2,
    ...) under the same total would exhaust the budget almost entirely on
    run 1 whenever run 1 alone offers ~budget-many calibration windows
    (every real pool day does at `scripts/adapt_beats.py`'s 8000-window
    default) -- silently training a "multi-run" model on one run. The
    rotation instead keeps per-run contributions within one window of each
    other for as long as every run still has windows.

    Exhausted runs drop out of the rotation: a run whose calibration side
    is consumed stops occupying turns, and the remaining runs keep rotating
    until the budget is met or every run is exhausted (yielding fewer than
    *max_windows* windows in total is not an error here -- the CLI layer
    decides whether an empty/short pool is fatal).

    Leakage rule, inherited PER RUN (package-5 spec D3, restated by P7's
    D6): each run's windows come from that run's OWN `iter_target_windows`
    iterator and therefore from that run's OWN top-split calibration side
    (`split_by_segments`, `_TOP_SPLIT_FRAC`, *seed*) -- no run's
    scoring-side windows are ever touched. *seed* is forwarded UNCHANGED as
    every run's split seed, with `iter_target_windows`'s own caveat
    inherited: 7 (the default) is the canonical split seed every Step-2
    sweep shares; pass a different value only for a deliberately different
    split.

    Determinism: the rotation adds no randomness of its own -- run order is
    *runs*' own order, per-run window order is `iter_target_windows`'s
    ascending-index order -- so the full yielded sequence is deterministic
    for a fixed (*runs*, *cfg*, *seed*, *target_hz*, *max_windows*).

    Each per-run iterator is constructed with `max_windows=max_windows` as
    its OWN cap too: the rotation stops at the total budget first, so no
    single run can ever be asked for more than *max_windows* windows --
    the per-run truncation therefore changes nothing about which windows
    the rotation can reach, while preserving the single-run path's property
    that a small budget also bounds how many burst files get opened.

    Args:
        runs: Runs to pool, iterated in the given order (= the rotation
            order). Callers are responsible for not repeating a run -- a
            duplicate would occupy two rotation slots and double-weight
            that run inside the shared budget.
        cfg: Project configuration, forwarded to `iter_target_windows`.
        max_windows: TOTAL window budget across all runs (see above). `0`
            yields nothing and constructs no per-run iterator.
        target_hz: Output sample rate/count per window, forwarded per run.
        seed: Per-run SPLIT seed (see the leakage paragraph above).
        return_run_names: If `True`, yield `(run_name, window)` pairs --
            the intended way for a caller (`scripts/adapt_beats.py`'s
            sidecar) to record the per-run window counts A3.11 requires,
            without a second, parallel computation of the rotation.
            `False` (default) yields bare arrays, matching
            `iter_target_windows`'s own base contract.

    Yields:
        `(target_hz,)`-shaped `float32` arrays (or `(run_name, array)`
        pairs when *return_run_names* is `True`), in rotation order.

    Raises:
        ValueError: if *max_windows* is negative (mirrors
            `iter_target_windows`'s own guard; raised eagerly at call
            time, before any iteration).
    """
    if max_windows < 0:
        raise ValueError(
            f"max_windows must be >= 0, got {max_windows!r} -- the multi-run TOTAL "
            "budget has no 'unbounded' mode and a negative value is always a caller bug"
        )
    paired = _iter_windows_multi(
        runs, cfg, max_windows=max_windows, target_hz=target_hz, seed=seed
    )
    if return_run_names:
        return paired
    return (window for _name, window in paired)
