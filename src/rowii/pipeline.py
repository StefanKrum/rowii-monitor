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
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rowii.config import Config
from rowii.io.dataset import BurstFile, Run
from rowii.io.gantner import GantnerHeader, read_gantner, read_header
from rowii.signals.features import AudioFeaturizer, VibFeaturizer, fuse
from rowii.signals.windows import WindowGrid, common_grid, coverage, window_slices

if TYPE_CHECKING:
    # BeatsFeaturizer needs torch, an optional `[beats]` extra -- never import it
    # unconditionally at module load time (would break the core package for anyone
    # without the extra installed; see `_featurizer_for_stream`, which imports it
    # lazily, only when an actual beats variant runs).
    from rowii.signals.beats import BeatsFeaturizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streams per variant (spec: audio streams = both mic files' channels;
# vibration = both vib streams' live channels; fusion = all four).
# ---------------------------------------------------------------------------

_AUDIO_STREAMS: tuple[str, ...] = ("RAWGeneratorMic__0", "RAWTurbineMic__1")
_VIB_STREAMS: tuple[str, ...] = ("RAWGeneratorVib__2", "RAWTurbineVib__3")

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


def _streams_for_variant(variant: str) -> tuple[str, ...]:
    """The stream ids *variant* needs, in the exact order every downstream per-variant
    assembly (`assemble_variant_features`, `_assemble_feature_names`) concatenates
    columns in. Also determines `PreparedRun.segment_ids`' primary stream: the first
    entry is always the first mic stream for every audio-bearing variant and the first
    vib stream for `"vibration"` -- see that field's own docstring.
    """
    if variant in ("audio", "audio-beats"):
        return _AUDIO_STREAMS
    if variant == "vibration":
        return _VIB_STREAMS
    if variant in ("fusion", "fusion-beats"):
        return _AUDIO_STREAMS + _VIB_STREAMS
    raise ValueError(f"unknown variant {variant!r}")


def _is_beats_variant(variant: str) -> bool:
    return variant in ("audio-beats", "fusion-beats")


def _featurizer_for_stream(
    stream: str, variant: str, cfg: Config
) -> AudioFeaturizer | VibFeaturizer | BeatsFeaturizer:
    """One featurizer instance for *stream*, given *variant*.

    Vibration streams always get a fresh `VibFeaturizer` regardless of
    variant (there is no "vib-beats" variant -- `BeatsFeaturizer` is
    audio-branch only per the design). Audio streams get `BeatsFeaturizer`
    for the two beats variants and `AudioFeaturizer` otherwise. One
    `BeatsFeaturizer` (and therefore one loaded copy of the frozen
    checkpoint) is constructed PER audio stream, not shared between the two
    -- mirroring how handcrafted `audio`/`fusion` already construct one
    `AudioFeaturizer` per stream, at the cost of loading the checkpoint
    twice for a beats variant that uses both mic streams. This keeps the
    per-stream featurizer lifecycle uniform across every variant rather than
    special-casing beats to share a single loaded model.
    """
    if stream not in _AUDIO_STREAMS:
        return VibFeaturizer()
    if _is_beats_variant(variant):
        from rowii.signals.beats import BeatsFeaturizer

        if cfg.beats_checkpoint is None:
            raise SystemExit(
                f"variant {variant!r} needs ROWII_BEATS_CHECKPOINT set; {_BEATS_INSTALL_HINT}"
            )
        return BeatsFeaturizer(checkpoint=cfg.beats_checkpoint)
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


def build_run_grid(run: Run, streams: tuple[str, ...], window_s: float) -> WindowGrid:
    """Common window grid across *streams*, computed from header-only reads."""
    synth_headers = []
    for stream in streams:
        files = sorted(run.files[stream], key=lambda f: f.start_utc_hint)
        synth_headers.append(_synthesize_run_header(files))
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


def _extract_stream_features(
    files: list[BurstFile],
    grid: WindowGrid,
    featurizer: AudioFeaturizer | VibFeaturizer | BeatsFeaturizer,
) -> _StreamFeatureResult:
    """Featurize one stream's burst files against *grid*, one file at a time.

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
        slices = window_slices(gf.timestamps_ns, grid)
        file_coverage = coverage(gf.timestamps_ns, grid, rate_hz)
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

    audio/vibration: `np.hstack` of the variant's stream matrices (z-scoring
    happens inside `run_detection`). fusion(-beats): `fuse()` on the raw audio
    and vibration matrices (fuse z-scores each side internally before
    concatenating) -- `run_detection`'s own `zscore` call on already-z-scored
    fusion input is then an idempotent-ish re-standardization (mean ~0, std ~1
    already), documented here rather than special-cased, per the brief.
    """
    if variant in ("audio", "audio-beats"):
        mats = [stream_results[s].features for s in _AUDIO_STREAMS]
        return np.hstack(mats)
    if variant == "vibration":
        mats = [stream_results[s].features for s in _VIB_STREAMS]
        return np.hstack(mats)
    if variant in ("fusion", "fusion-beats"):
        audio_mat = np.hstack([stream_results[s].features for s in _AUDIO_STREAMS])
        vib_mat = np.hstack([stream_results[s].features for s in _VIB_STREAMS])
        return fuse(audio_mat, vib_mat)
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
    stream (`RAWGeneratorMic__0`) for audio/audio-beats/fusion/fusion-beats, the first
    vib stream (`RAWGeneratorVib__2`) for vibration -- i.e. `_streams_for_variant(
    variant)[0]` in every case (that tuple's own ordering already puts the right
    stream first for every variant, so no separate lookup is needed). Segments are
    indexed 0-based, in ascending start-time order, per that ONE stream's own file
    list (not shared/reconciled across streams). `-1` where no file in the primary
    stream covers the window at all (see `_extract_stream_features`). Needed by
    Step 2's leakage-safe calibration/scoring splits (a split must never separate two
    windows that share a segment id)."""


# ---------------------------------------------------------------------------
# prepare_run: the public entry point
# ---------------------------------------------------------------------------


def prepare_run(
    run: Run,
    variant: str,
    cfg: Config,
    *,
    betriebsdaten: list[Path] | None = None,
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
    `PreparedRun`.

    SCADA/ground truth loading is deliberately NOT part of this function -- see the
    module docstring. Callers that need it (`scripts/run_step1.py`) call
    `rowii.scada.labels.load_run_gt` themselves afterward, using this function's
    `grid`/`valid_mask`.

    Args:
        run: The `Run` (burst files by stream) to prepare.
        variant: One of the concrete variant strings (`"audio"`, `"audio-beats"`,
            `"vibration"`, `"fusion"`, `"fusion-beats"`).
        cfg: Project configuration (`cfg.window.window_s`, `cfg.beats_checkpoint`).
        betriebsdaten: Accepted for signature symmetry with callers that already have
            this list on hand when they call `prepare_run` (`scripts/run_step1.py`,
            and Step-2's future `run_step2.py`) -- NOT used by this function itself:
            SCADA/GT loading is a separate step (see module docstring), and no
            `PreparedRun` field depends on it.

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
    grid = build_run_grid(run, streams, cfg.window.window_s)

    stream_results: dict[str, _StreamFeatureResult] = {}
    for stream in streams:
        featurizer = _featurizer_for_stream(stream, variant, cfg)
        stream_results[stream] = _extract_stream_features(run.files[stream], grid, featurizer)

    valid_mask = compute_validity_mask(list(stream_results.values()))
    features = assemble_variant_features(variant, stream_results)
    feature_names = _assemble_feature_names(variant, stream_results)
    # streams[0] is the PRIMARY stream for every variant (see _streams_for_variant's own
    # docstring): its ordering already puts the first mic stream first for every
    # audio-bearing variant and the first vib stream first for "vibration", so no
    # separate "primary stream" lookup is needed here.
    segment_ids = stream_results[streams[0]].segment_ids

    return PreparedRun(
        features=features,
        grid=grid,
        valid_mask=valid_mask,
        feature_names=feature_names,
        segment_ids=segment_ids,
    )
