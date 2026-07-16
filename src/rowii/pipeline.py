"""Shared run-preparation pipeline: discover -> grid -> chunked featurize -> validity mask.

Extracted from `scripts/run_step1.py` (Step-2 Task S1, behavior-preserving refactor) so
Step-2's mode-conditioned scoring package (`src/rowii/anomaly/`, design spec
`docs/superpowers/specs/2026-07-09-step2-mode-conditioned-ad-design.md`) can reuse the
exact same feature extraction Step-1's CLI already uses, instead of importing a *script*
module or duplicating the logic. `prepare_run` is the single entry point: given a `Run` +
variant string + `Config`, it returns a `PreparedRun` with the assembled `(W, F)` feature
matrix, the `WindowGrid` it was extracted against, the per-window validity mask, the
feature column names, and per-window burst-segment ids (needed by Step 2's leakage-safe
calibration/scoring splits -- a calibration/scoring split must never cut a 12-minute
burst segment in half).

Memory constraint (unchanged from Step-1's original docstring): mic files are ~800 MB
each, a stream is ~10 GB -- feature extraction is chunked per burst file
(`_extract_stream_features`): each file is read, sliced into windows, featurized, and its
rows written into a preallocated per-stream matrix before the file's raw data is freed.
The pipeline NEVER concatenates a whole stream into memory.

SCADA/ground-truth loading (`rowii.scada.labels`) is deliberately NOT part of this
module, even though `prepare_run` accepts an optional `betriebsdaten` argument (see its
own docstring) -- Step 2's per-state references are built from DETECTED cluster labels,
not GT (design spec §2: GT enters only evaluation views), and Step-1's own GT loading
stays a CLI-layer concern (`scripts/run_step1.py`'s `load_run_gt`), called separately
using this module's `PreparedRun.grid`/`valid_mask`.

On-disk feature cache (Step-2 Task S1): `prepare_run(..., use_cache=True)` (the default)
persists its result to `results/cache/<run.name>--<variant>.npz`, keyed by a sha256
fingerprint of everything that determines the output (variant, window duration, every
burst file's name+size, the beats-checkpoint path, and -- for the two tfc variants,
package-4 spec D4 -- the ONE tfc checkpoint path relevant to that variant) -- see
`_cache_fingerprint`. A fingerprint mismatch (or a missing/corrupt cache file) is
treated as a plain cache miss: recompute, then overwrite. `use_cache=False` bypasses
the cache entirely (never reads, never writes) -- wired to `scripts/run_step1.py
--no-cache`.
"""
from __future__ import annotations

import gc
import hashlib
import logging
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rowii.config import Config
from rowii.io.dataset import BurstFile, Run, run_utc_offset_ns
from rowii.io.gantner import GantnerHeader, read_gantner, read_header
from rowii.signals.features import AudioFeaturizer, VibFeaturizer, fuse
from rowii.signals.logmel import LogmelFeaturizer
from rowii.signals.windows import WindowGrid, common_grid, coverage, window_slices

if TYPE_CHECKING:
    # BeatsFeaturizer needs torch, an optional `[beats]` extra -- never import it
    # unconditionally at module load time (would break the core package for anyone
    # without the extra installed; see `_featurizer_for_stream`, which imports it
    # lazily, only when an actual beats variant runs).
    from rowii.signals.beats import BeatsFeaturizer

    # TfcFeaturizer (package-4 spec D4) shares BeatsFeaturizer's "optional [beats]
    # extra" story -- same TYPE_CHECKING-only import, same lazy real import inside
    # `_featurizer_for_stream`, only when an actual tfc variant runs.
    from rowii.tfc.wrapper import TfcFeaturizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streams per variant (spec: audio streams = both mic files' channels;
# vibration = both vib streams' live channels; fusion = all four; logmel = the
# primary (generator) mic ONLY, package-3 spec D3: size bound, documented).
# ---------------------------------------------------------------------------

_AUDIO_STREAMS: tuple[str, ...] = ("RAWGeneratorMic__0", "RAWTurbineMic__1")
_VIB_STREAMS: tuple[str, ...] = ("RAWGeneratorVib__2", "RAWTurbineVib__3")
_LOGMEL_STREAMS: tuple[str, ...] = ("RAWGeneratorMic__0",)

_COVERAGE_THRESHOLD = 0.8
_MAX_INVALID_FRACTION = 0.05
_SAMPLE_JITTER_TOLERANCE = 4
"""Max |actual - expected| sample count still treated as a "full" window (Task 13 real-data
finding: real DAQ clocks jitter by +/-1 sample/window at 10 kHz/50 kHz -- a sample count off
by 2370+ is a genuine partial window at a file boundary, not jitter; see
`_extract_stream_features`)."""

_BEATS_INSTALL_HINT = (
    'install extra: pip install -e ".[beats]" and set ROWII_BEATS_CHECKPOINT'
)

_TFC_INSTALL_HINT = 'install extra: pip install -e ".[beats]"'
"""TF-C (package-4 spec D4) reuses the SAME `[beats]` extra as BEATs (both need
torch) -- but unlike `_BEATS_INSTALL_HINT`, this hint deliberately does NOT also
name a checkpoint env var: TF-C has TWO independent checkpoints (`audio-tfc` ->
`ROWII_TFC_AUDIO_CHECKPOINT`, `vibration-tfc` -> `ROWII_TFC_VIB_CHECKPOINT`), so
which one to mention depends on the variant -- callers that need a checkpoint-
specific message name the right env var themselves (mirrors `_featurizer_for_
stream`'s own beats-checkpoint-None message shape, e.g. `scripts/run_step1.py`'s
`_import_tfc_or_exit`)."""


def _streams_for_variant(variant: str) -> tuple[str, ...]:
    """The stream ids *variant* needs, in the exact order every downstream per-variant
    assembly (`assemble_variant_features`, `_assemble_feature_names`) concatenates
    columns in. Also determines `PreparedRun.segment_ids`' primary stream: the first
    entry is always the first mic stream for every audio-bearing variant (including
    `"logmel"` and `"audio-tfc"`, the latter's ONLY stream being that same mic pair)
    and the first vib stream for `"vibration"`/`"vibration-tfc"` -- see that field's
    own docstring.
    """
    if variant in ("audio", "audio-beats", "audio-tfc"):
        return _AUDIO_STREAMS
    if variant in ("vibration", "vibration-tfc"):
        return _VIB_STREAMS
    if variant in ("fusion", "fusion-beats"):
        return _AUDIO_STREAMS + _VIB_STREAMS
    if variant == "logmel":
        return _LOGMEL_STREAMS
    raise ValueError(f"unknown variant {variant!r}")


def _is_beats_variant(variant: str) -> bool:
    return variant in ("audio-beats", "fusion-beats")


def _is_tfc_variant(variant: str) -> bool:
    """Mirrors `_is_beats_variant` (package-4 spec D4): `TfcFeaturizer` is likewise
    audio/vibration-branch-specific per variant (unlike BEATs, TF-C is NOT
    audio-only -- `"vibration-tfc"` runs the SAME frozen-embedding contract on the
    vibration branch, pre-trained separately on CWRU/Paderborn bearing data)."""
    return variant in ("audio-tfc", "vibration-tfc")


def _featurizer_for_stream(
    stream: str, variant: str, cfg: Config
) -> AudioFeaturizer | VibFeaturizer | BeatsFeaturizer | LogmelFeaturizer | TfcFeaturizer:
    """One featurizer instance for *stream*, given *variant*.

    Vibration streams get a fresh `VibFeaturizer` for every variant EXCEPT
    `"vibration-tfc"` (package-4 spec D4: unlike BEATs -- audio-branch only
    per the design, so no "vib-beats" variant ever exists -- TF-C IS
    pre-trained on both branches separately, so `TfcFeaturizer(checkpoint=
    cfg.tfc_vib_checkpoint)` is the one case a vibration stream's featurizer
    depends on *variant* at all). Audio streams get `LogmelFeaturizer` for
    the `logmel` variant (whose only stream is the primary mic,
    `_streams_for_variant`), `BeatsFeaturizer` for the two beats variants,
    `TfcFeaturizer(checkpoint=cfg.tfc_audio_checkpoint)` for `"audio-tfc"`,
    and `AudioFeaturizer` otherwise. One `BeatsFeaturizer`/`TfcFeaturizer`
    (and therefore one loaded copy of the frozen checkpoint) is constructed
    PER audio stream, not shared between the two -- mirroring how
    handcrafted `audio`/`fusion` already construct one `AudioFeaturizer` per
    stream, at the cost of loading the checkpoint twice for a variant that
    uses both mic streams. This keeps the per-stream featurizer lifecycle
    uniform across every variant rather than special-casing beats/tfc to
    share a single loaded model.

    Construction never raises for a tfc variant even when the relevant
    `cfg.tfc_*_checkpoint` is `None` (unlike the beats branch below, which
    raises `SystemExit` immediately): `TfcFeaturizer.__init__` never loads a
    checkpoint eagerly (its own module docstring), so a missing checkpoint
    only surfaces once `.transform()` actually runs -- callers that want to
    fail fast, before an expensive sweep starts, use the dedicated script-
    level guard instead (`scripts/run_step1.py`'s `_import_tfc_or_exit`,
    mirrored per script).
    """
    if stream not in _AUDIO_STREAMS:
        if variant == "vibration-tfc":
            from rowii.tfc.wrapper import TfcFeaturizer

            return TfcFeaturizer(checkpoint=cfg.tfc_vib_checkpoint)
        return VibFeaturizer()
    if variant == "logmel":
        return LogmelFeaturizer()
    if _is_beats_variant(variant):
        from rowii.signals.beats import BeatsFeaturizer

        if cfg.beats_checkpoint is None:
            raise SystemExit(
                f"variant {variant!r} needs ROWII_BEATS_CHECKPOINT set; {_BEATS_INSTALL_HINT}"
            )
        return BeatsFeaturizer(checkpoint=cfg.beats_checkpoint)
    if variant == "audio-tfc":
        from rowii.tfc.wrapper import TfcFeaturizer

        return TfcFeaturizer(checkpoint=cfg.tfc_audio_checkpoint)
    return AudioFeaturizer()


# ---------------------------------------------------------------------------
# Run-level grid (no data loaded -- header-only synthesis, spec Task 12 step 2)
# ---------------------------------------------------------------------------


def _synthesize_run_header(files: list[BurstFile]) -> GantnerHeader:
    """One run-level header per stream, from the FIRST and LAST file's headers only.

    `n_frames` is back-computed so that `t0_ns + n_frames / rate * 1e9` reproduces
    the last file's end timestamp -- i.e. the synthesized header describes a
    single virtual stream spanning the whole run without ever reading a sample.
    """
    first = read_header(files[0].path)
    last = read_header(files[-1].path)
    last_end_ns = last.t0_ns + round(last.n_frames / last.sample_rate_hz * 1e9)
    n_frames = round((last_end_ns - first.t0_ns) / 1e9 * first.sample_rate_hz)
    return GantnerHeader(
        source_name=first.source_name,
        channel_names=first.channel_names,
        channel_units=first.channel_units,
        t0_ns=first.t0_ns,
        sample_rate_hz=first.sample_rate_hz,
        n_frames=n_frames,
    )


def build_run_grid(
    run: Run, streams: tuple[str, ...], window_s: float, *, offset_ns: int | None = None
) -> WindowGrid:
    """Common window grid across *streams*, computed from header-only reads.

    Task 10 (D2, DAQ epoch-2000 clock quirk): before `common_grid`, every stream's
    synthesized header t0 is shifted by *run*'s derived UTC offset -- ONE value for
    the whole run, applied identically to every stream, so the returned grid's
    `t0_ns` (and therefore every `WindowGrid.edges_ns()`/`to_segments`-derived
    timestamp downstream) is true UTC, not the raw DAQ axis. A constant per-run
    shift moves every stream's header t0 (and end, since `n_frames`/
    `sample_rate_hz` are untouched) by the same amount, so `common_grid`'s
    intersection-based `t0_ns`/`n_windows` selection is unaffected in every way
    except the absolute axis itself: relative sample alignment, and everything
    `run_detection`/`run_sweep` compute from it (labels, scores, FAR), are
    invariant under this shift by construction (see
    `tests/test_detect_e2e.py`'s/`tests/test_sweep.py`'s dedicated invariance
    tests) -- only RENDERED times move. `prepare_run` also needs this SAME offset
    for `_extract_stream_features`'s own per-file timestamp shift (every raw
    on-disk sample timestamp must move onto the identical true-UTC axis as this
    grid, or nothing in it would ever match a window again -- see that function's
    own docstring); *offset_ns* lets a caller that already derived it
    (`rowii.io.dataset.run_utc_offset_ns(run)`) pass it straight through instead
    of this function deriving it a second time (still cheap either way -- header
    reads only -- but redundant work with no reason to duplicate it when a caller
    already has the value).

    Args:
        run: The `Run` (burst files by stream) to build the grid for.
        streams: The stream ids to build the grid across (a variant's own subset,
            `_streams_for_variant`) -- ALL of *run*'s streams still contribute to
            the offset derivation when *offset_ns* is `None` (see
            `run_utc_offset_ns`'s own docstring on why it is not scoped to just
            these streams).
        window_s: Window duration in seconds.
        offset_ns: Precomputed `run_utc_offset_ns(run)`, if the caller already has
            it (`prepare_run`). `None` (the default -- every other caller, e.g.
            `scripts/analyze_step1.py`'s own direct `build_run_grid` call) derives
            it internally.
    """
    if offset_ns is None:
        offset_ns = run_utc_offset_ns(run)
    synth_headers = []
    for stream in streams:
        files = sorted(run.files[stream], key=lambda f: f.start_utc_hint)
        header = _synthesize_run_header(files)
        if offset_ns:
            header = replace(header, t0_ns=header.t0_ns + offset_ns)
        synth_headers.append(header)
    return common_grid(synth_headers, window_s)


# ---------------------------------------------------------------------------
# Chunked per-stream feature extraction (spec Task 12 step 3 -- memory constraint)
# ---------------------------------------------------------------------------


@dataclass
class _StreamFeatureResult:
    features: np.ndarray
    """(grid.n_windows, F) float64, NaN where a window has no full-window data."""
    coverage: np.ndarray
    """(grid.n_windows,) float64 in [0, 1] -- summed per-file coverage for this stream."""
    feature_names: list[str] = field(default_factory=list)
    """Column names for `features`, in order -- the stream's own featurizer's
    `feature_names()` after its last `transform()` call. Empty if this stream never
    featurized a single full window (`features` is then `(grid.n_windows, 0)`).
    Defaults to `[]` so call sites that only care about `features`/`coverage` (e.g.
    `assemble_variant_features`/`compute_validity_mask` unit tests) do not need to
    supply it."""
    segment_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    """(grid.n_windows,) int64 -- index (0-based, ascending start-time order within
    THIS stream's own file list) of the burst file that contributed a window's
    samples; -1 where no file in this stream covers the window at all (see
    `_extract_stream_features`). Defaults to an empty array for call sites that
    construct a `_StreamFeatureResult` directly without going through
    `_extract_stream_features`."""


def _shift_ts_ns(ts_ns: np.ndarray, offset_ns: int) -> np.ndarray:
    """`ts_ns + offset_ns`, staying in `uint64` throughout -- mirrors `rowii.scada.
    labels._shift_ts_ns` (duplicated, not imported: a small, self-contained utility,
    not worth a new cross-module dependency for). `ts_ns` is `uint64` (as produced
    by `read_gantner`) and so is `WindowGrid.edges_ns()`, which `window_slices`/
    `coverage` compare it against; mixing `int64` and `uint64` numpy arrays silently
    upcasts BOTH to `float64`, losing precision above `2**53` (~9.007e15) -- far
    below these ~1e18 ns timestamps, so a naive `ts_ns.astype(np.int64) + offset_ns`
    would corrupt every window boundary.
    """
    if offset_ns >= 0:
        return ts_ns + np.uint64(offset_ns)
    return ts_ns - np.uint64(-offset_ns)


def _extract_stream_features(
    files: list[BurstFile],
    grid: WindowGrid,
    featurizer: (
        AudioFeaturizer | VibFeaturizer | BeatsFeaturizer | LogmelFeaturizer | TfcFeaturizer
    ),
    offset_ns: int = 0,
) -> _StreamFeatureResult:
    """Featurize one stream's burst files against *grid*, one file at a time.

    Task 10 (D2, DAQ epoch-2000 clock quirk): *grid* is already on the true-UTC
    axis (`build_run_grid`), but each file's OWN raw `timestamps_ns` (`read_gantner`,
    read straight off disk) still carries the raw DAQ axis -- *offset_ns* (the same
    run-level `rowii.io.dataset.run_utc_offset_ns` value `build_run_grid` baked into
    *grid*'s own `t0_ns`) is added to a file's timestamps before EVERY comparison
    against *grid* (`window_slices`, `coverage`) below, so both sides of every
    comparison share one axis again. Shifting both `grid` and every file's
    timestamps by the identical constant leaves every window's sample SET
    unchanged (a pure translation), which is exactly what makes `features`/
    `coverage`/`segment_ids` -- and everything downstream that is built from them
    (labels, scores, FAR) -- invariant under this shift; only the grid's own
    absolute `t0_ns` (and therefore every RENDERED time) moves. Defaults to `0`
    (no-op) for callers that already hand-align a `WindowGrid` to their own raw
    fixture data (this function's own direct unit tests).

    A window is FULL (and gets featurized) if its sample count is within
    `_SAMPLE_JITTER_TOLERANCE` of the expected count for the file's own rate;
    windows further off are left as NaN and contribute to the coverage
    accounting only (a genuinely full window can still be assembled later
    from a DIFFERENT file that covers the same grid index, if the burst
    boundary falls inside that window -- but per the discovery contract runs
    are contiguous with < 15 min gaps and each grid window in practice is
    covered by exactly one file).

    Tolerance rationale (Task 13 real-data finding): real DAQ clocks jitter
    by +/-1 sample per window even with no actual data gap -- `read_gantner`'s
    rate estimate is a median over the whole file, so any single window can
    round to one sample more or fewer than that estimate predicts. A genuine
    partial window at a file boundary is off by thousands of samples (the
    file's own start/end falling mid-window), nowhere near this tolerance, so
    the two cases stay cleanly separated. Accepted windows within tolerance
    but not EXACTLY `expected_samples` long are trimmed to the batch's
    shortest accepted length before stacking (`np.stack` requires uniform
    shape) -- losing a small number of jitter samples out of tens of
    thousands has no meaningful effect on any spectral feature.

    Note on `VibFeaturizer`'s per-file live-channel discovery: `VibFeaturizer.
    transform` re-derives which channels are live from the STD of whatever
    batch it is given (see `rowii.signals.features` module docstring), so it
    runs independently on each file here rather than once for the whole
    stream. In the expected data (channel liveness is a fixed property of the
    sensor for the whole June-25 recording -- GenVib0/TurVib0 dead
    throughout), every file agrees on the same live-channel set and produces
    identically-shaped feature rows. If a channel's liveness genuinely
    differed across files within one run (e.g. a sensor failing mid-run),
    later files would return a different feature-row width and the
    `feature_matrix[window_idx] = row` assignment below would raise a numpy
    `ValueError` (shape mismatch) rather than silently corrupting the
    matrix -- an acceptable fail-fast outcome for a pathological case with no
    test coverage or spec-defined handling.

    `segment_ids` (Step-2 Task S1 addition, alongside the pre-existing
    `features`/`coverage`): derived from the SAME per-file `window_slices` call
    already computed for `coverage` -- a window is attributed to the EARLIEST
    (sorted) file whose slice for that window is non-empty (`sl.stop > sl.start`,
    i.e. the file contributed ANY samples, not necessarily a full window); a
    window no file covers at all keeps the `-1` sentinel it is initialized with.
    A window split across a file boundary (some samples from file *i*, the rest
    from file *i + 1*) is attributed to the earlier file *i* only, since the
    per-window loop below never overwrites an already-claimed (`!= -1`) entry.
    """
    sorted_files = sorted(files, key=lambda f: f.start_utc_hint)
    n_windows = grid.n_windows
    coverage_acc = np.zeros(n_windows, dtype=np.float64)
    segment_ids = np.full(n_windows, -1, dtype=np.int64)
    feature_matrix: np.ndarray | None = None
    feature_names: list[str] = []
    n_features = 0

    for file_index, bf in enumerate(sorted_files):
        gf = read_gantner(bf.path)
        rate_hz = gf.header.sample_rate_hz
        expected_samples = round(rate_hz * (grid.window_ns / 1e9))
        ts_ns = _shift_ts_ns(gf.timestamps_ns, offset_ns) if offset_ns else gf.timestamps_ns
        slices = window_slices(ts_ns, grid)
        file_coverage = coverage(ts_ns, grid, rate_hz)
        coverage_acc += file_coverage

        for w, sl in enumerate(slices):
            if sl.stop > sl.start and segment_ids[w] == -1:
                segment_ids[w] = file_index

        full_window_indices = [
            i for i, sl in enumerate(slices)
            if abs((sl.stop - sl.start) - expected_samples) <= _SAMPLE_JITTER_TOLERANCE
        ]
        if full_window_indices:
            trim_len = min(slices[i].stop - slices[i].start for i in full_window_indices)
            stack = np.stack(
                [gf.data[slices[i].start : slices[i].start + trim_len, :]
                 for i in full_window_indices],
                axis=0,
            ).astype(np.float32)
            batch_features = featurizer.transform(stack, rate_hz)
            if feature_matrix is None:
                n_features = batch_features.shape[1]
                feature_matrix = np.full((n_windows, n_features), np.nan, dtype=np.float64)
            for row, window_idx in zip(batch_features, full_window_indices, strict=True):
                feature_matrix[window_idx] = row

        del gf
        gc.collect()

    if feature_matrix is None:
        # No file in this stream ever produced a single full window.
        feature_matrix = np.full((n_windows, 0), np.nan, dtype=np.float64)
    else:
        feature_names = featurizer.feature_names()

    return _StreamFeatureResult(
        features=feature_matrix,
        coverage=np.clip(coverage_acc, 0.0, 1.0),
        feature_names=feature_names,
        segment_ids=segment_ids,
    )


# ---------------------------------------------------------------------------
# Validity mask (spec Task 12 step 5)
# ---------------------------------------------------------------------------


def compute_validity_mask(
    stream_results: list[_StreamFeatureResult],
) -> np.ndarray:
    """A window is valid iff every used stream has coverage >= 0.8 and no NaN feature.

    Raises:
        RuntimeError: if more than 5% of grid windows are invalid (spec rule).
    """
    n_windows = stream_results[0].features.shape[0]
    valid = np.ones(n_windows, dtype=bool)
    for sr in stream_results:
        valid &= sr.coverage >= _COVERAGE_THRESHOLD
        valid &= ~np.isnan(sr.features).any(axis=1)

    invalid_fraction = 1.0 - valid.mean() if n_windows else 0.0
    if invalid_fraction > _MAX_INVALID_FRACTION:
        raise RuntimeError(
            f"{invalid_fraction:.1%} of {n_windows} grid windows are invalid "
            f"(coverage < {_COVERAGE_THRESHOLD} or NaN features in some used stream), "
            f"exceeding the {_MAX_INVALID_FRACTION:.0%} hard-fail threshold"
        )
    return valid


# ---------------------------------------------------------------------------
# Per-variant feature assembly (spec Task 12 step 4)
# ---------------------------------------------------------------------------


def assemble_variant_features(
    variant: str, stream_results: dict[str, _StreamFeatureResult]
) -> np.ndarray:
    """Combine per-stream feature matrices into the variant's (W, F) matrix.

    audio/vibration/logmel (and their tfc counterparts, package-4 spec D4 --
    `audio-tfc` mirrors `audio`, `vibration-tfc` mirrors `vibration`, both
    single-stream-family variants): `np.hstack` of the variant's stream
    matrices (z-scoring happens inside `run_detection`; for logmel that is a
    single primary-mic matrix, hstack of one). fusion(-beats): `fuse()` on
    the raw audio and vibration matrices (fuse z-scores each side internally
    before concatenating) -- `run_detection`'s own `zscore` call on
    already-z-scored fusion input is then an idempotent-ish
    re-standardization (mean ~0, std ~1 already), documented here rather
    than special-cased, per the brief.
    """
    if variant in ("audio", "audio-beats", "audio-tfc"):
        mats = [stream_results[s].features for s in _AUDIO_STREAMS]
        return np.hstack(mats)
    if variant in ("vibration", "vibration-tfc"):
        mats = [stream_results[s].features for s in _VIB_STREAMS]
        return np.hstack(mats)
    if variant in ("fusion", "fusion-beats"):
        audio_mat = np.hstack([stream_results[s].features for s in _AUDIO_STREAMS])
        vib_mat = np.hstack([stream_results[s].features for s in _VIB_STREAMS])
        return fuse(audio_mat, vib_mat)
    if variant == "logmel":
        mats = [stream_results[s].features for s in _LOGMEL_STREAMS]
        return np.hstack(mats)
    raise ValueError(f"unknown variant {variant!r}")


def _assemble_feature_names(
    variant: str, stream_results: dict[str, _StreamFeatureResult]
) -> list[str]:
    """Column names for `assemble_variant_features(variant, stream_results)`'s output,
    in the exact same per-stream order. fusion(-beats) z-scores each side before
    concatenating but never reorders columns (see `assemble_variant_features`'s own
    docstring), so `_streams_for_variant(variant)`'s stream order alone determines the
    naming for every variant, fused or not. Each stream's own `feature_names()` values
    are prefixed `"<stream>::"` since both audio streams (and both vibration streams)
    independently reuse local channel-index names like `"ch0_log_rms"`, which would
    otherwise collide once concatenated.
    """
    names: list[str] = []
    for stream in _streams_for_variant(variant):
        names.extend(f"{stream}::{name}" for name in stream_results[stream].feature_names)
    return names


# ---------------------------------------------------------------------------
# PreparedRun: the k/clusterer-independent half of the pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedRun:
    """Output of `prepare_run`: everything about a (run, variant) combination that does
    NOT depend on a clusterer or k (Step-1's own `_prepare_run_features` computed this
    inline; Step-2 reuses it as-is via this module)."""

    features: np.ndarray
    """(W, F) float64 -- per-window feature matrix for the requested variant, NaN on
    rows with insufficient stream coverage or missing data (see `valid_mask`)."""
    grid: WindowGrid
    """The window grid `features`/`valid_mask`/`segment_ids` are aligned against."""
    valid_mask: np.ndarray
    """(W,) bool -- True where every stream used by this variant has coverage >= 0.8
    AND a finite (non-NaN) feature row (`compute_validity_mask`)."""
    feature_names: list[str]
    """`features`' column names, one per column, in order -- `"<stream>::<local_name>"`
    (e.g. `"RAWGeneratorMic__0::ch0_log_rms"`); see `_assemble_feature_names`."""
    segment_ids: np.ndarray
    """(W,) int64 -- index of the 12-min source burst segment (file) that contributed
    a window's samples, read off the PRIMARY stream of the variant: the first mic
    stream (`RAWGeneratorMic__0`) for audio/audio-beats/audio-tfc/fusion/fusion-beats/
    logmel (logmel's ONLY stream is that mic; audio-tfc mirrors audio, package-4 spec
    D4), the first vib stream (`RAWGeneratorVib__2`) for vibration/vibration-tfc
    (the latter mirroring vibration) -- i.e. `_streams_for_variant(
    variant)[0]` in every case (that tuple's own ordering already puts the right
    stream first for every variant, so no separate lookup is needed). Segments are
    indexed 0-based, in ascending start-time order, per that ONE stream's own file
    list (not shared/reconciled across streams). `-1` where no file in the primary
    stream covers the window at all (see `_extract_stream_features`). Needed by
    Step 2's leakage-safe calibration/scoring splits (a split must never separate two
    windows that share a segment id)."""


# ---------------------------------------------------------------------------
# On-disk feature cache (Step-2 Task S1)
# ---------------------------------------------------------------------------

_CACHE_SUBDIR = "cache"
# npz archive member names, shared verbatim (as literal strings, not indirected through
# constants) between `_write_cached_prepared_run`'s explicit `np.savez(..., name=value)`
# keywords and `_load_cached_prepared_run`'s `data["name"]` lookups -- `np.savez`'s own
# stub declares a distinct `allow_pickle: bool` keyword alongside `**kwds: ArrayLike`,
# so passing these names through a `**{...}` dict (which WOULD allow a single shared
# constant) makes mypy widen every value's type to satisfy `bool` too; spelling them out
# as ordinary keyword arguments here avoids that entirely, at the cost of the two
# function bodies needing to agree on the member names by inspection rather than a
# shared symbol (verified in both directions by `tests/test_pipeline.py`'s cache
# round-trip tests).


def _cache_npz_path(results_root: Path, run_name: str, variant: str) -> Path:
    """`results/cache/<run_name>--<variant>.npz` -- one (run, variant)'s cache entry."""
    return results_root / _CACHE_SUBDIR / f"{run_name}--{variant}.npz"


def _cache_fingerprint(run: Run, variant: str, cfg: Config) -> str:
    """sha256 hex digest of everything that determines `prepare_run`'s output for one
    (run, variant, cfg): the variant string itself, the window duration, every burst
    file in the run (name + byte size, sorted -- catches both a file being swapped for
    different content, most likely to change size, and files being added/removed;
    scoped to the whole run rather than just the variant's own streams so this stays
    correct even if `_streams_for_variant`'s mapping ever changes), and the `cfg`
    fields that change what a featurizer computes: `cfg.beats_checkpoint`'s path,
    included unconditionally (every variant's payload carries it, even a handcrafted
    one) so a handcrafted-variant cache is never silently reused after switching
    to/from a beats checkpoint; and, ONLY for a tfc variant (package-4 spec D4),
    exactly ONE extra `tfc_*_checkpoint=` line carrying the checkpoint path relevant
    to *variant* itself -- `cfg.tfc_audio_checkpoint` for `"audio-tfc"`,
    `cfg.tfc_vib_checkpoint` for `"vibration-tfc"`. Every other variant's payload
    carries NO tfc line at all, keeping it byte-identical to the pre-package-4
    format: the payload is a PERSISTENCE format, not an implementation detail --
    every existing `results/cache/*.npz` stores a fingerprint computed from it, so
    even a semantically-neutral shape change (e.g. appending blank lines for every
    variant, the exact package-4 execution finding this wording guards against)
    silently invalidates every pre-existing cache entry, including hours-expensive
    BEATs extractions. Scoping the single tfc line to its own variant also keeps a
    checkpoint change from invalidating caches that never depended on it (TF-C has
    TWO independent checkpoints rather than beats' one -- folding both in for every
    variant would force a needless recompute of, say, every `vibration-tfc` cache
    whenever a user merely points `ROWII_TFC_AUDIO_CHECKPOINT` at a new file). Any
    future payload change must consciously update `tests/test_pipeline.py`'s golden
    pins AND deliberately migrate/invalidate the on-disk caches.
    `cfg.detect`/`cfg.gt` are deliberately excluded: they govern clustering/GT
    labeling, never feature EXTRACTION, so changing them must not invalidate this
    cache.
    """
    file_entries = sorted(
        f"{bf.path.name}:{bf.path.stat().st_size}"
        for files in run.files.values()
        for bf in files
    )
    payload_lines = [
        f"variant={variant}",
        f"window_s={cfg.window.window_s!r}",
        f"beats_checkpoint={cfg.beats_checkpoint or ''}",
    ]
    if variant == "audio-tfc":
        payload_lines.append(f"tfc_audio_checkpoint={cfg.tfc_audio_checkpoint or ''}")
    elif variant == "vibration-tfc":
        payload_lines.append(f"tfc_vib_checkpoint={cfg.tfc_vib_checkpoint or ''}")
    payload = "\n".join([*payload_lines, *file_entries])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cached_prepared_run(
    cache_path: Path,
    fingerprint: str,
    run: Run,
    streams: tuple[str, ...],
    window_s: float,
    offset_ns: int,
) -> PreparedRun | None:
    """`PreparedRun` from *cache_path* iff it exists, is readable, and its stored
    fingerprint equals *fingerprint*; `None` otherwise (cache miss -- caller
    recomputes and overwrites). A corrupt or partially-written cache file (e.g. from
    an interrupted previous run) is treated as a miss, not a crash: logged as a
    warning, never raised.

    Task 10 (D4, DAQ epoch-2000 clock quirk -- cache compatibility WITHOUT
    recompute): a cache npz written before this task's fix stores `grid_t0_ns` on
    the RAW axis; a BEATs cache costs hours to rebuild and must never be
    invalidated just to fix a timestamp. On every fingerprint-matched hit, this
    function recomputes the run's grid fresh via `build_run_grid` (headers only,
    cheap -- the EXACT computation the compute path performs, given the SAME
    *run*/*streams*/*window_s*/*offset_ns* `prepare_run` already derived once for
    this call) and OVERRIDES the loaded grid's `t0_ns` with the fresh value when
    they differ, logging one INFO line. `window_ns`/`n_windows` are shift-invariant
    (a constant t0 shift changes neither, `build_run_grid`'s own docstring) so they
    are kept exactly as cached, never taken from the fresh recompute --
    `features`/`valid_mask`/`segment_ids` are likewise shift-invariant and are
    returned exactly as cached, never recomputed. A cache already written POST-fix
    (t0_ns already true-UTC) is a silent no-op here: the freshly computed t0
    matches the cached one exactly, so nothing changes and nothing is logged.
    """
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if str(data["fingerprint"][0]) != fingerprint:
                return None
            grid = WindowGrid(
                t0_ns=int(data["grid_t0_ns"][0]),
                window_ns=int(data["grid_window_ns"][0]),
                n_windows=int(data["grid_n_windows"][0]),
            )
            fresh_grid = build_run_grid(run, streams, window_s, offset_ns=offset_ns)
            if fresh_grid.t0_ns != grid.t0_ns:
                logger.info(
                    "prepare_run: cache at %s carries a raw-axis grid_t0_ns=%d -- "
                    "overriding with the true-UTC t0_ns=%d (window_ns/n_windows/"
                    "features/valid_mask/segment_ids are shift-invariant, kept as "
                    "cached)",
                    cache_path, grid.t0_ns, fresh_grid.t0_ns,
                )
                grid = WindowGrid(
                    t0_ns=fresh_grid.t0_ns, window_ns=grid.window_ns, n_windows=grid.n_windows
                )
            return PreparedRun(
                features=data["features"],
                grid=grid,
                valid_mask=data["valid_mask"],
                feature_names=data["feature_names"].tolist(),
                segment_ids=data["segment_ids"],
            )
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        logger.warning(
            "prepare_run: cache at %s is unreadable (%s) -- recomputing", cache_path, exc
        )
        return None


def _write_cached_prepared_run(cache_path: Path, fingerprint: str, prepared: PreparedRun) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        features=prepared.features,
        valid_mask=prepared.valid_mask,
        segment_ids=prepared.segment_ids,
        feature_names=np.array(prepared.feature_names, dtype=str),
        grid_t0_ns=np.array([prepared.grid.t0_ns], dtype=np.int64),
        grid_window_ns=np.array([prepared.grid.window_ns], dtype=np.int64),
        grid_n_windows=np.array([prepared.grid.n_windows], dtype=np.int64),
        fingerprint=np.array([fingerprint], dtype=str),
    )


# ---------------------------------------------------------------------------
# prepare_run: the public entry point
# ---------------------------------------------------------------------------


def prepare_run(
    run: Run,
    variant: str,
    cfg: Config,
    *,
    betriebsdaten: list[Path] | None = None,
    use_cache: bool = True,
) -> PreparedRun:
    """Grid + chunked featurize + validity mask for one (run, variant) -- the
    expensive, k/clusterer-independent half of the Step-1 pipeline (design spec
    `docs/superpowers/specs/2026-07-05-step1-state-detection-design.md` Task 12 steps
    1-5), extracted from `scripts/run_step1.py`'s original `_prepare_run_features` so
    Step-2's `src/rowii/anomaly/` package can reuse the exact same feature extraction.

    Steps: resolve which streams the variant needs (`_streams_for_variant`) -> build a
    run-level window grid from header-only reads (`build_run_grid`, never loads a
    sample) -> chunked per-file feature extraction per stream (`_extract_stream_
    features`, one burst file resident in memory at a time) -> per-variant column
    assembly (`assemble_variant_features`/`_assemble_feature_names`) -> validity mask
    (`compute_validity_mask`, raises `RuntimeError` if > 5% of windows are invalid) ->
    `PreparedRun`, optionally read from or written to the on-disk cache (module
    docstring).

    SCADA/ground truth loading is deliberately NOT part of this function -- see the
    module docstring. Callers that need it (`scripts/run_step1.py`) call
    `rowii.scada.labels.load_run_gt` themselves afterward, using this function's
    `grid`/`valid_mask`.

    Args:
        run: The `Run` (burst files by stream) to prepare.
        variant: One of the concrete variant strings (`"audio"`, `"audio-beats"`,
            `"audio-tfc"`, `"vibration"`, `"vibration-tfc"`, `"fusion"`,
            `"fusion-beats"`, `"logmel"`).
        cfg: Project configuration (`cfg.window.window_s`, `cfg.beats_checkpoint`,
            `cfg.tfc_audio_checkpoint`/`cfg.tfc_vib_checkpoint`, `cfg.results_root`
            for the cache location).
        betriebsdaten: Accepted for signature symmetry with callers that already have
            this list on hand when they call `prepare_run` (`scripts/run_step1.py`,
            and Step-2's future `run_step2.py`) -- NOT used by this function itself:
            SCADA/GT loading is a separate step (see module docstring), and neither
            the cache fingerprint nor any `PreparedRun` field depends on it.
        use_cache: When `True` (default), load a previously cached `PreparedRun` from
            `results/cache/<run.name>--<variant>.npz` if its stored fingerprint
            matches the current run/variant/config (`_cache_fingerprint`); otherwise
            recompute and write a fresh cache entry. `False` always recomputes and
            never reads or writes the cache (matches `scripts/run_step1.py
            --no-cache`).

    Returns:
        A `PreparedRun` with the assembled `(W, F)` feature matrix, the `WindowGrid` it
        was extracted against, the per-window validity mask, per-column feature names,
        and per-window primary-stream burst-segment ids (see `PreparedRun.segment_ids`
        for the exact convention).

    Raises:
        RuntimeError: if > 5% of grid windows are invalid (`compute_validity_mask`).
        ValueError: if *variant* is not a recognised variant string.
    """
    streams = _streams_for_variant(variant)
    cache_path = _cache_npz_path(cfg.results_root, run.name, variant)
    fingerprint = _cache_fingerprint(run, variant, cfg)
    # Derived ONCE per call (Task 10, D2/D4) -- header-only reads, cheap -- and
    # threaded through every place below that needs the run's true-UTC axis:
    # build_run_grid (both the cache-hit override recompute and the compute
    # path), and _extract_stream_features's own per-file timestamp shift.
    offset_ns = run_utc_offset_ns(run)

    if use_cache:
        cached = _load_cached_prepared_run(
            cache_path, fingerprint, run, streams, cfg.window.window_s, offset_ns
        )
        if cached is not None:
            logger.info("prepare_run: cache hit for %s/%s (%s)", run.name, variant, cache_path)
            return cached
        logger.info("prepare_run: cache miss for %s/%s -- recomputing", run.name, variant)

    grid = build_run_grid(run, streams, cfg.window.window_s, offset_ns=offset_ns)

    stream_results: dict[str, _StreamFeatureResult] = {}
    for stream in streams:
        featurizer = _featurizer_for_stream(stream, variant, cfg)
        stream_results[stream] = _extract_stream_features(
            run.files[stream], grid, featurizer, offset_ns
        )

    valid_mask = compute_validity_mask(list(stream_results.values()))
    features = assemble_variant_features(variant, stream_results)
    feature_names = _assemble_feature_names(variant, stream_results)
    # streams[0] is the PRIMARY stream for every variant (see _streams_for_variant's own
    # docstring): its ordering already puts the first mic stream first for every
    # audio-bearing variant and the first vib stream first for "vibration", so no
    # separate "primary stream" lookup is needed here.
    segment_ids = stream_results[streams[0]].segment_ids

    prepared = PreparedRun(
        features=features,
        grid=grid,
        valid_mask=valid_mask,
        feature_names=feature_names,
        segment_ids=segment_ids,
    )

    if use_cache:
        _write_cached_prepared_run(cache_path, fingerprint, prepared)

    return prepared
