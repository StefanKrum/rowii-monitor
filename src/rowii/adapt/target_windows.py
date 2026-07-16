"""Leakage-aware target-normal training-window iterator (Step-2 package-5
Task 2, design spec D3: `docs/superpowers/specs/2026-07-16-step2-package5-
adaptation-design.md`).

`iter_target_windows` feeds `scripts/adapt_beats.py`'s masked-patch
adaptation objective (Task 1, `rowii.adapt.objective.masked_patch_loss`)
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
not handcrafted or BEATs features), grouped by file so each burst file is
opened via `read_gantner` at most ONCE regardless of how many of its windows
are selected -- mirrors `rowii.pipeline._extract_stream_features`'s own
one-file-resident-at-a-time memory discipline (module docstring: "The
pipeline NEVER concatenates a whole stream into memory").

Torch-free by design (numpy/scipy only): `scripts/adapt_beats.py` (an eager-
torch script, Task 3) consumes this module's output, but `target_windows.py`
itself is NOT on the plan's Global-Constraints eager-torch module list
(`objective.py`/`lora.py`/`student.py`'s model part/`fusionx/model.py`) -- it
has no reason to import torch at all, so it stays importable (and
independently testable) without the optional `[beats]` extra installed.
"""
from __future__ import annotations

import gc
from collections import defaultdict
from collections.abc import Iterator
from typing import Literal, overload

import numpy as np
from scipy.signal import resample_poly

from rowii.anomaly.references import split_by_segments
from rowii.config import Config
from rowii.io.dataset import Run, run_utc_offset_ns
from rowii.io.gantner import read_gantner
from rowii.pipeline import prepare_run
from rowii.signals.windows import window_slices

_PRIMARY_MIC_STREAM = "RAWGeneratorMic__0"
"""The "audio" variant's primary stream (`rowii.pipeline._AUDIO_STREAMS[0]`,
restated in `PreparedRun.segment_ids`'s own docstring): `prepare_run(run,
"audio", cfg)` derives `segment_ids` from exactly this stream's own
time-sorted file list, so indexing `run.files[_PRIMARY_MIC_STREAM]` (sorted
the SAME way -- `start_utc_hint` ascending) with a `segment_ids` value lands
on the exact file `rowii.pipeline._extract_stream_features` attributed that
window to. Duplicated here rather than importing `rowii.pipeline`'s private
`_AUDIO_STREAMS` constant -- this module only ever needs the ONE stream, and
the repo's own convention (`rowii.pipeline._shift_ts_ns`, itself duplicated
from `rowii.scada.labels._shift_ts_ns`) is to duplicate a small, stable
constant/helper rather than reach across a module boundary for a
leading-underscore name."""

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
       (DAQ clock jitter, `rowii.pipeline._SAMPLE_JITTER_TOLERANCE` -- the
       primary-mic window this function receives was already selected via
       `prepare_run`'s own `valid_mask`, which only accepts windows within
       that tolerance, but this pad/trim step absorbs the residual jitter
       either way, without needing to duplicate the tolerance constant
       itself).
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

    # Group by primary-stream file index (PreparedRun.segment_ids), preserving
    # each file's own windows in ascending order -- calibration_windows is
    # already ascending, so a single pass appending in order keeps every
    # per-file group internally ascending too.
    windows_by_file: dict[int, list[int]] = defaultdict(list)
    for w in calibration_windows.tolist():
        windows_by_file[int(prepared.segment_ids[w])].append(w)

    # dict iteration order == insertion order (first-seen file index, in the
    # order calibration_windows itself presents them). Since segment_ids
    # enumerates the primary stream's files in ascending start-time order
    # (PreparedRun.segment_ids' own docstring) and the grid's windows are
    # equally time-ordered, an earlier window index is never attributed to a
    # LATER file than a later window index (contiguous, non-overlapping
    # bursts -- rowii.pipeline._extract_stream_features's own module-level
    # invariant: "each grid window in practice is covered by exactly one
    # file"). So iterating windows_by_file in its natural dict order already
    # yields windows in overall ascending index order, one file read at a
    # time.
    primary_files = sorted(run.files[_PRIMARY_MIC_STREAM], key=lambda f: f.start_utc_hint)
    offset_ns = run_utc_offset_ns(run)

    for file_index, window_indices in windows_by_file.items():
        bf = primary_files[file_index]
        gf = read_gantner(bf.path)
        rate_hz = gf.header.sample_rate_hz
        ts_ns = _shift_ts_ns(gf.timestamps_ns, offset_ns) if offset_ns else gf.timestamps_ns
        slices = window_slices(ts_ns, prepared.grid)

        for w in window_indices:
            sl = slices[w]
            raw = gf.data[sl.start : sl.stop, :]
            mono = raw.mean(axis=1).astype(np.float64)
            resampled = _resample_window(mono, rate_hz, target_hz)
            standardized = _standardize_1d(resampled)
            yield w, standardized.astype(np.float32)

        del gf
        gc.collect()


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

    Pipeline per selected window: read the containing burst file's raw
    samples once (`read_gantner`, grouped and cached per file --
    `_iter_indexed_windows`) -> slice out this window's samples
    (`rowii.signals.windows.window_slices` against `prepare_run`'s own grid)
    -> mono-mix channels (mean) -> resample to *target_hz*
    (`_resample_window`, `scipy.signal.resample_poly`) -> per-window
    standardize (`_standardize_1d`) -> `float32`.

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
        *max_windows*).
    """
    indexed = _iter_indexed_windows(
        run, cfg, target_hz=target_hz, seed=seed, max_windows=max_windows
    )
    if return_indices:
        return indexed
    return (window for _, window in indexed)
