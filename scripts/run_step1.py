"""Step-1 operating-state detection CLI: discover -> grid -> featurize -> detect -> evaluate.

One CLI drives the whole (run, variant, clusterer) grid described in the design
spec (`docs/superpowers/specs/2026-07-05-step1-state-detection-design.md` §5)
and implementation plan Task 12
(`docs/superpowers/plans/2026-07-05-step1-state-detection.md`). Every stage is
an importable, unit-testable function; `main` only wires them together per the
CLI arguments.

Memory constraint (spec: mic files are ~800 MB each, a stream is ~10 GB): the
pipeline NEVER concatenates a whole stream into memory. Feature extraction is
chunked per burst file (`_extract_stream_features`): each file is read, sliced
into windows, featurized, and its rows written into a preallocated per-stream
matrix before the file's raw data is freed.
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.config import Config, load_config  # noqa: E402
from rowii.eval.metrics import EvalResult, evaluate  # noqa: E402
from rowii.eval.report import write_report  # noqa: E402
from rowii.io.dataset import BurstFile, RecordingIndex, Run, discover  # noqa: E402
from rowii.io.gantner import GantnerHeader, read_gantner, read_header  # noqa: E402
from rowii.scada.labels import gt_labels, load_scada_window_means  # noqa: E402
from rowii.signals.features import AudioFeaturizer, VibFeaturizer, fuse  # noqa: E402
from rowii.signals.windows import WindowGrid, common_grid, coverage, window_slices  # noqa: E402
from rowii.state.detect import DetectionResult, run_detection  # noqa: E402
from rowii.state.segments import to_segments  # noqa: E402

if TYPE_CHECKING:
    # BeatsFeaturizer needs torch, an optional `[beats]` extra -- never import it
    # unconditionally at module load time (would break the core package for
    # anyone without the extra installed; see `_import_beats_or_exit`/
    # `_featurizer_for_stream`, which import it lazily, only when an actual
    # beats variant runs).
    from rowii.signals.beats import BeatsFeaturizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI enums
# ---------------------------------------------------------------------------

ClustererName = Literal["kmeans", "gmm"]
"""Concrete clusterer identifier, matching `rowii.state.detect.run_detection`'s own
`clusterer` parameter type -- used everywhere a clusterer choice flows into it, so no
`type: ignore` is needed at the `run_detection` call site."""

_RUN_CHOICES: tuple[str, ...] = ("tu", "pu-morning", "pu-afternoon", "all")
_VARIANT_CHOICES: tuple[str, ...] = (
    "audio",
    "audio-beats",
    "vibration",
    "fusion",
    "fusion-beats",
    "all",
)
_CLUSTERER_CHOICES: tuple[str, ...] = ("kmeans", "gmm", "all")

_CONCRETE_VARIANTS: tuple[str, ...] = (
    "audio",
    "audio-beats",
    "vibration",
    "fusion",
    "fusion-beats",
)
_CONCRETE_CLUSTERERS: tuple[ClustererName, ...] = ("kmeans", "gmm")
_K_SWEEP_VALUES: tuple[int, ...] = (3, 4, 5, 6)

# Streams used per variant (spec: audio streams = both mic files' channels;
# vibration = both vib streams' live channels; fusion = all four).
_AUDIO_STREAMS: tuple[str, ...] = ("RAWGeneratorMic__0", "RAWTurbineMic__1")
_VIB_STREAMS: tuple[str, ...] = ("RAWGeneratorVib__2", "RAWTurbineVib__3")

_COVERAGE_THRESHOLD = 0.8
_MAX_INVALID_FRACTION = 0.05
_INVALID_LABEL = -1
_SAMPLE_JITTER_TOLERANCE = 4
"""Max |actual - expected| sample count still treated as a "full" window (Task 13 real-data
finding: real DAQ clocks jitter by +/-1 sample/window at 10 kHz/50 kHz -- a sample count off
by 2370+ is a genuine partial window at a file boundary, not jitter; see
`_extract_stream_features`)."""

_BEATS_INSTALL_HINT = (
    'install extra: pip install -e ".[beats]" and set ROWII_BEATS_CHECKPOINT'
)


def _streams_for_variant(variant: str) -> tuple[str, ...]:
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
# argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Step-1 unsupervised operating-state detection for one or more "
            "(run, variant, clusterer) combinations."
        )
    )
    parser.add_argument("--run", choices=_RUN_CHOICES, default="all")
    parser.add_argument("--variant", choices=_VARIANT_CHOICES, default="audio")
    parser.add_argument("--clusterer", choices=_CLUSTERER_CHOICES, default="kmeans")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of clusters (default: cfg.detect.n_states).",
    )
    parser.add_argument(
        "--k-sweep",
        action="store_true",
        help="Sweep k in {3,4,5,6} for the given combination instead of a single k.",
    )
    return parser


def _resolve_choice(choice: str, all_value: str, concrete: tuple[str, ...]) -> tuple[str, ...]:
    return concrete if choice == all_value else (choice,)


def _resolve_clusterers(choice: str) -> tuple[ClustererName, ...]:
    """`_resolve_choice` specialised to `ClustererName`.

    `choice` is `args.clusterer`, already runtime-guaranteed by argparse's
    `choices=_CLUSTERER_CHOICES` to be one of `"kmeans"`, `"gmm"`, `"all"` --
    the `cast` below only tells mypy what argparse already enforces at
    runtime; it does not weaken any actual validation.
    """
    resolved = _resolve_choice(choice, "all", _CONCRETE_CLUSTERERS)
    return cast(tuple[ClustererName, ...], resolved)


def _resolve_runs(choice: str, index: RecordingIndex) -> list[Run]:
    if choice == "all":
        return list(index.runs)
    matches = [r for r in index.runs if r.name == choice]
    if not matches:
        logger.warning("run %r not found in discovered index (no matching Run)", choice)
    return matches


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
    """
    sorted_files = sorted(files, key=lambda f: f.start_utc_hint)
    n_windows = grid.n_windows
    coverage_acc = np.zeros(n_windows, dtype=np.float64)
    feature_matrix: np.ndarray | None = None
    n_features = 0

    for bf in sorted_files:
        gf = read_gantner(bf.path)
        rate_hz = gf.header.sample_rate_hz
        expected_samples = round(rate_hz * (grid.window_ns / 1e9))
        slices = window_slices(gf.timestamps_ns, grid)
        file_coverage = coverage(gf.timestamps_ns, grid, rate_hz)
        coverage_acc += file_coverage

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

    return _StreamFeatureResult(
        features=feature_matrix, coverage=np.clip(coverage_acc, 0.0, 1.0)
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


# ---------------------------------------------------------------------------
# SCADA / GT (spec Task 12 step 6)
# ---------------------------------------------------------------------------


def _betriebsdaten_for_grid(betriebsdaten: list[Path], grid: WindowGrid) -> list[Path]:
    """Betriebsdaten files whose hourly span intersects the grid's UTC time range.

    Cheap (header-only) intersection test: a file (nominally one hour) overlaps
    the grid iff its own [t0, t_end) intersects [grid.t0_ns, grid_end_ns).
    """
    grid_end_ns = int(grid.edges_ns()[-1])
    matched = []
    for path in betriebsdaten:
        header = read_header(path)
        file_end_ns = header.t0_ns + round(header.n_frames / header.sample_rate_hz * 1e9)
        if header.t0_ns < grid_end_ns and file_end_ns > grid.t0_ns:
            matched.append(path)
    return sorted(matched)


def load_run_gt(
    betriebsdaten: list[Path], grid: WindowGrid, cfg: Config, valid_mask: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SCADA window means + GT labels for *grid*, with invalid windows forced to "unknown".

    Returns (scada, gt) -- `scada` for the report's power-curve panel, `gt` for
    evaluation. Invalid windows are not evaluable (their features never fed the
    detector meaningfully), so their GT state is overwritten to "unknown"
    regardless of what the rule-based labeler decided from SCADA alone; the
    count of windows newly marked this way is left to the caller to log.
    """
    matched_files = _betriebsdaten_for_grid(betriebsdaten, grid)
    scada = load_scada_window_means(matched_files, grid)
    gt = gt_labels(scada, cfg.gt, window_s=cfg.window.window_s)
    gt = gt.copy()
    gt.loc[~valid_mask, "state"] = "unknown"
    return scada, gt


# ---------------------------------------------------------------------------
# Minimal fallback report writer for the "no SCADA coverage" case
# ---------------------------------------------------------------------------


def _write_no_gt_report(out_dir: Path, run: str, variant: str, det: DetectionResult) -> None:
    """Minimal report for a run/variant combination with zero known-GT windows.

    `evaluate()` documents (and raises `ValueError` for) exactly this case --
    "no windows with a known ground-truth state" -- since every metric it
    computes is conditioned on having at least one eval window. Rather than
    changing `eval`/`report` (out of scope for this task), this writes a
    reduced `report.md` plus the two artifacts that do not depend on an
    `EvalResult` (`segments.csv`, `frame_labels.parquet`) directly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Run report: {run} / {variant}",
        "",
        "## Metrics",
        "",
        "No SCADA coverage for this run -- metrics skipped (all windows gt.state == "
        '"unknown").',
        "",
        f"| k (clusters) | {det.k} |",
        f"| n_windows | {len(det.frame_labels)} |",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))
    det.segments.to_csv(out_dir / "segments.csv", index=False)
    frame_df = pd.DataFrame(
        {
            "window": np.arange(len(det.frame_labels), dtype=np.int64),
            "cluster": det.frame_labels,
            "mapped_state": ["unknown"] * len(det.frame_labels),
        }
    )
    frame_df.to_parquet(out_dir / "frame_labels.parquet", engine="pyarrow", index=False)


def _silhouette_or_nan(features_valid: np.ndarray, labels: np.ndarray) -> float:
    """`silhouette_score` on z-scored valid features vs *labels*; NaN if <= 1 unique label
    (silhouette is undefined for a single cluster -- spec Task 12 step 8's explicit guard)."""
    from sklearn.metrics import silhouette_score

    from rowii.signals.features import zscore

    if len(np.unique(labels)) <= 1:
        return float("nan")
    return float(silhouette_score(zscore(features_valid), labels))


# ---------------------------------------------------------------------------
# Per-combo pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComboResult:
    """One row of `results/summary.csv`."""

    run: str
    variant: str
    clusterer: str
    k: int
    n_windows: int
    n_valid: int
    n_eval: int
    ari: float | None
    macro_f1: float | None
    boundary_median_abs_s: float | None
    silhouette: float | None
    notes: str


@dataclass
class _RunFeatures:
    """Everything about a (run, variant) combination that does NOT depend on k or
    clusterer -- computed once and reused across every k/clusterer that shares it
    (in particular, the 4 iterations of a `--k-sweep`)."""

    grid: WindowGrid
    features_full: np.ndarray
    valid_mask: np.ndarray
    scada: pd.DataFrame
    gt: pd.DataFrame


def _prepare_run_features(
    run: Run, variant: str, cfg: Config, betriebsdaten: list[Path]
) -> _RunFeatures:
    """Grid + chunked featurize + validity mask + SCADA/GT for one (run, variant).

    This is the expensive, k/clusterer-independent half of the pipeline (spec
    Task 12 steps 1-3, 6): discover a run-level grid without loading data,
    extract features one burst file at a time (never a whole stream in
    memory), compute the validity mask, and load SCADA-derived ground truth.
    """
    streams = _streams_for_variant(variant)
    grid = build_run_grid(run, streams, cfg.window.window_s)

    stream_results: dict[str, _StreamFeatureResult] = {}
    for stream in streams:
        featurizer = _featurizer_for_stream(stream, variant, cfg)
        stream_results[stream] = _extract_stream_features(run.files[stream], grid, featurizer)

    valid_mask = compute_validity_mask(list(stream_results.values()))
    features_full = assemble_variant_features(variant, stream_results)
    scada, gt = load_run_gt(betriebsdaten, grid, cfg, valid_mask)

    return _RunFeatures(
        grid=grid, features_full=features_full, valid_mask=valid_mask, scada=scada, gt=gt
    )


def _combo_out_dir(
    results_root: Path, run_name: str, variant: str, clusterer: ClustererName, k: int, *,
    is_k_sweep: bool,
) -> Path:
    """`results/<run>/<variant>-<clusterer>/` for a single-k run, or
    `results/<run>/<variant>-<clusterer>-k<k>/` for one iteration of a `--k-sweep`
    (each k gets its own directory so a sweep's 4 iterations don't overwrite each
    other's `report.md`/`timeline.png` -- only `summary.csv`'s per-k ROWS are meant
    to accumulate across a sweep, not the detailed per-combo artifacts)."""
    combo_name = f"{variant}-{clusterer}-k{k}" if is_k_sweep else f"{variant}-{clusterer}"
    return results_root / run_name / combo_name


def _detect_and_report(
    run_name: str,
    variant: str,
    clusterer: ClustererName,
    cfg: Config,
    prepared: _RunFeatures,
    results_root: Path,
    *,
    k: int | None,
    notes: str = "",
    is_k_sweep: bool = False,
) -> ComboResult:
    """Cheap, k/clusterer-dependent half: run_detection -> evaluate -> write_report.

    Reuses *prepared* (grid, features, validity mask, GT) computed once by
    `_prepare_run_features` -- the part of the pipeline that a `--k-sweep`'s 4
    iterations must NOT redundantly recompute.
    """
    grid = prepared.grid
    valid_mask = prepared.valid_mask
    n_valid = int(valid_mask.sum())
    features_valid = prepared.features_full[valid_mask]

    valid_grid = WindowGrid(t0_ns=grid.t0_ns, window_ns=grid.window_ns, n_windows=n_valid)
    det_valid = run_detection(features_valid, valid_grid, cfg.detect, clusterer, k=k)

    full_labels = np.full(grid.n_windows, _INVALID_LABEL, dtype=np.int64)
    full_labels[valid_mask] = det_valid.frame_labels
    det = DetectionResult(
        frame_labels=full_labels, segments=to_segments(full_labels, grid), k=det_valid.k
    )

    silhouette = _silhouette_or_nan(features_valid, det_valid.frame_labels)

    out_dir = _combo_out_dir(
        results_root, run_name, variant, clusterer, det.k, is_k_sweep=is_k_sweep
    )
    gt = prepared.gt
    n_unknown = int((gt["state"] == "unknown").sum())

    ev: EvalResult
    try:
        ev = evaluate(det.frame_labels, gt, grid)
    except ValueError:
        logger.info(
            "no SCADA coverage -- metrics skipped for %s/%s-%s (%d/%d unknown windows)",
            run_name, variant, clusterer, n_unknown, grid.n_windows,
        )
        _write_no_gt_report(out_dir, run_name, f"{variant}-{clusterer}", det)
        return ComboResult(
            run=run_name, variant=variant, clusterer=clusterer, k=det.k,
            n_windows=grid.n_windows, n_valid=n_valid, n_eval=0,
            ari=None, macro_f1=None, boundary_median_abs_s=None, silhouette=silhouette,
            notes=_combine_notes("no SCADA coverage", notes),
        )

    write_report(out_dir, run_name, f"{variant}-{clusterer}", det, ev, prepared.scada, gt=gt)

    return ComboResult(
        run=run_name, variant=variant, clusterer=clusterer, k=det.k,
        n_windows=grid.n_windows, n_valid=n_valid, n_eval=ev.n_eval_windows,
        ari=ev.ari, macro_f1=ev.macro_f1,
        boundary_median_abs_s=ev.boundary_median_abs_s, silhouette=silhouette,
        notes=_combine_notes("", notes),
    )


def _combine_notes(auto_note: str, extra_note: str) -> str:
    parts = [n for n in (extra_note, auto_note) if n]
    return ", ".join(parts)


def run_combo(
    run: Run,
    variant: str,
    clusterer: ClustererName,
    cfg: Config,
    betriebsdaten: list[Path],
    results_root: Path,
    *,
    k: int | None = None,
) -> ComboResult:
    """Execute the full pipeline for one (run, variant, clusterer) combination and
    append its row to `results/summary.csv` immediately (spec Task 12 step 8: the
    summary row is an OUTPUT of the combo itself, not a batched end-of-run write --
    so a crash partway through a large `--run all --variant all` grid still leaves
    every already-completed combination's row on disk).

    discover (already done by the caller) -> grid -> chunked featurize ->
    validity mask -> assemble variant features -> SCADA/GT -> run_detection ->
    evaluate (or the no-GT fallback) -> write_report -> ComboResult.
    """
    if _is_beats_variant(variant):
        _import_beats_or_exit()

    prepared = _prepare_run_features(run, variant, cfg, betriebsdaten)
    result = _detect_and_report(
        run.name, variant, clusterer, cfg, prepared, results_root, k=k
    )
    _append_summary_row(results_root, result)
    return result


def run_combo_k_sweep(
    run: Run,
    variant: str,
    clusterer: ClustererName,
    cfg: Config,
    betriebsdaten: list[Path],
    results_root: Path,
) -> list[ComboResult]:
    """Run k in {3,4,5,6} for one (run, variant, clusterer), reusing the same
    (expensive) extracted features across all four k values -- only detection,
    evaluation, and reporting differ per k. Each row is annotated "k-sweep" and
    appended to `results/summary.csv` as it completes (same per-combo append
    contract as `run_combo`)."""
    if _is_beats_variant(variant):
        _import_beats_or_exit()

    prepared = _prepare_run_features(run, variant, cfg, betriebsdaten)
    rows = []
    for k in _K_SWEEP_VALUES:
        result = _detect_and_report(
            run.name, variant, clusterer, cfg, prepared, results_root,
            k=k, notes="k-sweep", is_k_sweep=True,
        )
        _append_summary_row(results_root, result)
        rows.append(result)
    return rows


def _import_beats_or_exit() -> None:
    try:
        import rowii.signals.beats  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"BEATs featurizer not available ({exc}); {_BEATS_INSTALL_HINT}"
        ) from exc


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

_SUMMARY_COLUMNS: tuple[str, ...] = (
    "run", "variant", "clusterer", "k", "n_windows", "n_valid", "n_eval",
    "ari", "macro_f1", "boundary_median_abs_s", "silhouette", "notes",
)


def _append_summary_row(results_root: Path, result: ComboResult) -> None:
    summary_path = results_root / "summary.csv"
    row_df = pd.DataFrame([vars(result)], columns=_SUMMARY_COLUMNS)
    results_root.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(summary_path, index=False)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config()
    index = discover(cfg.data_root)

    runs = _resolve_runs(args.run, index)
    variants = _resolve_choice(args.variant, "all", _CONCRETE_VARIANTS)
    clusterers = _resolve_clusterers(args.clusterer)

    n_rows = 0
    for run in runs:
        for variant in variants:
            for clusterer in clusterers:
                if args.k_sweep:
                    n_rows += len(
                        run_combo_k_sweep(
                            run, variant, clusterer, cfg, index.betriebsdaten, cfg.results_root
                        )
                    )
                else:
                    run_combo(
                        run, variant, clusterer, cfg, index.betriebsdaten,
                        cfg.results_root, k=args.k,
                    )
                    n_rows += 1

    print(f"run_step1: wrote {n_rows} row(s) to {cfg.results_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
