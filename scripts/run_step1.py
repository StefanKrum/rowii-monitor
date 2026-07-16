"""Step-1 operating-state detection CLI: discover -> grid -> featurize -> detect -> evaluate.

One CLI drives the whole (run, variant, clusterer) grid described in the design
spec (`docs/superpowers/specs/2026-07-05-step1-state-detection-design.md` §5)
and implementation plan Task 12
(`docs/superpowers/plans/2026-07-05-step1-state-detection.md`). Every stage is
an importable, unit-testable function; `main` only wires them together per the
CLI arguments.

Run preparation (grid synthesis, chunked per-file feature extraction, validity mask --
the expensive, k/clusterer-independent half of the pipeline, including the memory
constraint that a whole stream is never concatenated into memory) now lives in
`rowii.pipeline.prepare_run` (Step-2 Task S1 extraction, so Step-2's own scoring
pipeline can reuse it without importing this script). This module is a thin caller:
`_prepare_run_features` calls `rowii.pipeline.prepare_run` and pairs its output with
SCADA/GT loading, which stays here (see `rowii.pipeline`'s module docstring for why).
A handful of pipeline-internal names (`_streams_for_variant`, `build_run_grid`,
`_StreamFeatureResult`, `_extract_stream_features`, `assemble_variant_features`,
`_AUDIO_STREAMS`, `_VIB_STREAMS`) are re-exported here for `scripts/analyze_step1.py`
and `tests/test_cli_smoke.py`, which access them as `run_step1.<name>` from before
this refactor.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii import pipeline as _pipeline  # noqa: E402
from rowii.config import Config, load_config  # noqa: E402
from rowii.eval.metrics import EvalResult, evaluate  # noqa: E402
from rowii.eval.report import write_report  # noqa: E402
from rowii.io.dataset import (  # noqa: E402
    RecordingIndex,
    Run,
    betriebsdaten_utc_offset_ns,
    discover,
    run_utc_offset_ns,
)
from rowii.io.gantner import read_header  # noqa: E402
from rowii.pipeline import (  # noqa: E402
    _BEATS_INSTALL_HINT,
    _TFC_INSTALL_HINT,
    PreparedRun,
    _is_beats_variant,
    _is_tfc_variant,
    prepare_run,
)
from rowii.scada.labels import gt_labels, load_scada_window_means  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402
from rowii.state.detect import DetectionResult, run_detection  # noqa: E402
from rowii.state.segments import to_segments  # noqa: E402

logger = logging.getLogger(__name__)

# Re-exported for `scripts/analyze_step1.py` and `tests/test_cli_smoke.py`, which
# access these run-preparation internals as `run_step1.<name>` from before this
# refactor (module docstring) -- plain assignments rather than `from rowii.pipeline
# import <name>` so ruff's unused-import check (these names are never referenced
# anywhere else in THIS module's own code) does not apply.
_AUDIO_STREAMS = _pipeline._AUDIO_STREAMS
_VIB_STREAMS = _pipeline._VIB_STREAMS
_StreamFeatureResult = _pipeline._StreamFeatureResult
_extract_stream_features = _pipeline._extract_stream_features
_streams_for_variant = _pipeline._streams_for_variant
assemble_variant_features = _pipeline.assemble_variant_features
build_run_grid = _pipeline.build_run_grid

# ---------------------------------------------------------------------------
# CLI enums
# ---------------------------------------------------------------------------

ClustererName = Literal["kmeans", "gmm"]
"""Concrete clusterer identifier, matching `rowii.state.detect.run_detection`'s own
`clusterer` parameter type -- used everywhere a clusterer choice flows into it, so no
`type: ignore` is needed at the `run_detection` call site."""

_VARIANT_CHOICES: tuple[str, ...] = (
    "audio",
    "audio-beats",
    "audio-tfc",
    "vibration",
    "vibration-tfc",
    "fusion",
    "fusion-beats",
    "logmel",
    "all",
)
_CLUSTERER_CHOICES: tuple[str, ...] = ("kmeans", "gmm", "all")

_CONCRETE_VARIANTS: tuple[str, ...] = (
    "audio",
    "audio-beats",
    "audio-tfc",
    "vibration",
    "vibration-tfc",
    "fusion",
    "fusion-beats",
    # "logmel" is deliberately NOT expanded by `--variant all` (though it stays
    # explicitly selectable via _VARIANT_CHOICES): package-3 spec D3 scopes logmel
    # as a Step-2 autoencoder INPUT, not a Step-1 clustering candidate -- a
    # 3136-dim z-scored matrix into the full-covariance GMM is statistically
    # underdetermined at typical per-run window counts (and measured at 4.6-8.2 s
    # per fit even on trivial synthetic data). audio-tfc/vibration-tfc draw the
    # OPPOSITE conclusion (package-4 spec D4): TF-C's 256-d embedding IS
    # well-conditioned for the GMM, so both ARE expanded here, same treatment as
    # the handcrafted/beats variants.
)
_CONCRETE_CLUSTERERS: tuple[ClustererName, ...] = ("kmeans", "gmm")
_K_SWEEP_VALUES: tuple[int, ...] = (3, 4, 5, 6)

_INVALID_LABEL = -1


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
    parser.add_argument(
        "--run",
        default="all",
        help=(
            "Run name to process, or 'all' for every discovered run. Run names are "
            "dynamically discovered from ROWII_DATA_ROOT (day-prefixed under a "
            "parent root, e.g. '010726-tu_ph_tu') -- not a fixed enumeration, so any "
            "string is accepted here; an unmatched name warns and processes zero "
            "combinations rather than failing argparse validation."
        ),
    )
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
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Disable rowii.pipeline.prepare_run's on-disk feature cache "
            "(results/cache/<run>--<variant>.npz): always recompute features and "
            "never write a cache entry for this invocation."
        ),
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
# SCADA / GT (spec Task 12 step 6)
# ---------------------------------------------------------------------------


def _betriebsdaten_for_grid(betriebsdaten: list[Path], grid: WindowGrid) -> list[Path]:
    """Betriebsdaten files whose hourly span intersects the grid's UTC time range.

    Cheap (header-only) intersection test: a file (nominally one hour) overlaps
    the grid iff its own [t0, t_end) intersects [grid.t0_ns, grid_end_ns).

    Task 10 (D3 tracing finding): *grid* is true-UTC since `rowii.pipeline.
    build_run_grid` (D2), but each candidate file's own `header.t0_ns` (`read_
    header`, straight off disk) is still the raw DAQ axis -- shifted here by
    *betriebsdaten*'s own derived offset (`rowii.io.dataset.
    betriebsdaten_utc_offset_ns`) before the intersection test, mirroring `rowii.
    scada.labels.load_scada_window_means`'s identical D3 fix. BEFORE this task the
    comparison was RAW-vs-RAW (grid built on the pre-fix raw axis too) -- both
    sides shared the SAME axis by construction, so selection worked correctly by
    accident, not because either side was ever true UTC (see the task report for
    the full derivation).
    """
    grid_end_ns = int(grid.edges_ns()[-1])
    offset_ns = betriebsdaten_utc_offset_ns(betriebsdaten)
    matched = []
    for path in betriebsdaten:
        header = read_header(path)
        file_start_ns = header.t0_ns + offset_ns
        file_end_ns = file_start_ns + round(header.n_frames / header.sample_rate_hz * 1e9)
        if file_start_ns < grid_end_ns and file_end_ns > grid.t0_ns:
            matched.append(path)
    return sorted(matched)


def load_run_gt(
    run: Run, betriebsdaten: list[Path], grid: WindowGrid, cfg: Config, valid_mask: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SCADA window means + GT labels for *grid*, with invalid windows forced to "unknown".

    Returns (scada, gt) -- `scada` for the report's power-curve panel, `gt` for
    evaluation. Invalid windows are not evaluable (their features never fed the
    detector meaningfully), so their GT state is overwritten to "unknown"
    regardless of what the rule-based labeler decided from SCADA alone; the
    count of windows newly marked this way is left to the caller to log.

    *run* is used only to derive `run_utc_offset_ns(run)`, passed to
    `load_scada_window_means` as the audio-side cross-check (D3: never used to
    derive the SCADA-side shift itself, which `load_scada_window_means` always
    derives independently from *betriebsdaten*).
    """
    matched_files = _betriebsdaten_for_grid(betriebsdaten, grid)
    scada = load_scada_window_means(
        matched_files, grid, audio_run_offset_ns=run_utc_offset_ns(run)
    )
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
    state_ari: float | None
    """State-level (mode) ARI (`rowii.eval.metrics.EvalResult.state_ari`) -- the
    primary metric per the design (module docstring: majority cluster->state mapping,
    no 1:1 restriction). `None` for a no-GT combo (see `_write_no_gt_report`)."""
    state_accuracy: float | None
    """State-level accuracy (`EvalResult.state_accuracy`). `None` for a no-GT combo."""
    state_macro_f1: float | None
    """State-level macro-F1 (`EvalResult.state_macro_f1`). `None` for a no-GT combo."""
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
    run: Run, variant: str, cfg: Config, betriebsdaten: list[Path], *, use_cache: bool = True
) -> _RunFeatures:
    """Grid + chunked featurize + validity mask + SCADA/GT for one (run, variant).

    This is the expensive, k/clusterer-independent half of the pipeline (spec
    Task 12 steps 1-3, 6): `rowii.pipeline.prepare_run` does the run-level-grid/
    chunked-featurize/validity-mask work (Step-2 Task S1 extraction -- see that
    module for the memory-bounded per-file extraction and its optional on-disk
    feature cache); SCADA-derived ground truth loading stays here (`load_run_gt`),
    since `rowii.pipeline` deliberately does not own it (see that module's
    docstring for why).
    """
    prepared: PreparedRun = prepare_run(
        run, variant, cfg, betriebsdaten=betriebsdaten, use_cache=use_cache
    )
    scada, gt = load_run_gt(run, betriebsdaten, prepared.grid, cfg, prepared.valid_mask)

    return _RunFeatures(
        grid=prepared.grid,
        features_full=prepared.features,
        valid_mask=prepared.valid_mask,
        scada=scada,
        gt=gt,
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
            state_ari=None, state_accuracy=None, state_macro_f1=None,
            notes=_combine_notes("no SCADA coverage", notes),
        )

    write_report(out_dir, run_name, f"{variant}-{clusterer}", det, ev, prepared.scada, gt=gt)

    return ComboResult(
        run=run_name, variant=variant, clusterer=clusterer, k=det.k,
        n_windows=grid.n_windows, n_valid=n_valid, n_eval=ev.n_eval_windows,
        ari=ev.ari, macro_f1=ev.macro_f1,
        boundary_median_abs_s=ev.boundary_median_abs_s, silhouette=silhouette,
        state_ari=ev.state_ari, state_accuracy=ev.state_accuracy,
        state_macro_f1=ev.state_macro_f1,
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
    use_cache: bool = True,
) -> ComboResult:
    """Execute the full pipeline for one (run, variant, clusterer) combination and
    append its row to `results/summary.csv` immediately (spec Task 12 step 8: the
    summary row is an OUTPUT of the combo itself, not a batched end-of-run write --
    so a crash partway through a large `--run all --variant all` grid still leaves
    every already-completed combination's row on disk).

    discover (already done by the caller) -> grid -> chunked featurize ->
    validity mask -> assemble variant features -> SCADA/GT -> run_detection ->
    evaluate (or the no-GT fallback) -> write_report -> ComboResult.

    *use_cache* is forwarded to `rowii.pipeline.prepare_run` (see `--no-cache`).
    """
    if _is_beats_variant(variant):
        _import_beats_or_exit()
    if _is_tfc_variant(variant):
        _import_tfc_or_exit(cfg, variant)

    prepared = _prepare_run_features(run, variant, cfg, betriebsdaten, use_cache=use_cache)
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
    *,
    use_cache: bool = True,
) -> list[ComboResult]:
    """Run k in {3,4,5,6} for one (run, variant, clusterer), reusing the same
    (expensive) extracted features across all four k values -- only detection,
    evaluation, and reporting differ per k. Each row is annotated "k-sweep" and
    appended to `results/summary.csv` as it completes (same per-combo append
    contract as `run_combo`).

    *use_cache* is forwarded to `rowii.pipeline.prepare_run` (see `--no-cache`).
    """
    if _is_beats_variant(variant):
        _import_beats_or_exit()
    if _is_tfc_variant(variant):
        _import_tfc_or_exit(cfg, variant)

    prepared = _prepare_run_features(run, variant, cfg, betriebsdaten, use_cache=use_cache)
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


def _import_tfc_or_exit(cfg: Config, variant: str) -> None:
    """Extends `_import_beats_or_exit`'s pattern (package-4 spec D4): fails fast,
    BEFORE `_prepare_run_features` starts an expensive extraction, on either of two
    problems `rowii.pipeline._featurizer_for_stream`'s own tfc dispatch would
    otherwise only surface much later (torch missing is never even checked there --
    `TfcFeaturizer` construction never raises; a missing checkpoint only surfaces
    once `.transform()` actually runs on the first full window). Torch is checked
    FIRST (a machine with neither problem should see the more fundamental one), then
    the ONE checkpoint relevant to *variant* itself -- `cfg.tfc_audio_checkpoint` for
    `"audio-tfc"`, `cfg.tfc_vib_checkpoint` for `"vibration-tfc"`.
    """
    try:
        import rowii.tfc.wrapper  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"TF-C featurizer not available ({exc}); {_TFC_INSTALL_HINT}"
        ) from exc
    if variant == "audio-tfc":
        checkpoint, env_var = cfg.tfc_audio_checkpoint, "ROWII_TFC_AUDIO_CHECKPOINT"
    else:
        checkpoint, env_var = cfg.tfc_vib_checkpoint, "ROWII_TFC_VIB_CHECKPOINT"
    if checkpoint is None:
        raise SystemExit(f"variant {variant!r} needs {env_var} set; {_TFC_INSTALL_HINT}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

_SUMMARY_COLUMNS: tuple[str, ...] = (
    "run", "variant", "clusterer", "k", "n_windows", "n_valid", "n_eval",
    "ari", "macro_f1", "boundary_median_abs_s", "silhouette",
    "state_ari", "state_accuracy", "state_macro_f1", "notes",
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
    use_cache = not args.no_cache

    n_rows = 0
    for run in runs:
        # A run must only ever be evaluated against its OWN day tree's
        # Betriebsdaten (spec: docs/superpowers/specs/2026-07-07-step1-
        # multiday-phase-shifter-addendum.md §2) -- passing the flat, pooled
        # `index.betriebsdaten` here would let `_betriebsdaten_for_grid`'s
        # time-overlap filter match a DIFFERENT day's SCADA file whenever two
        # days' burst/Betriebsdaten timestamps happen to overlap (every day
        # tree in this delivery shares the same device-clock convention, so
        # this is a real risk, not a hypothetical one).
        run_betriebsdaten = index.betriebsdaten_by_day.get(run.day_root, [])
        for variant in variants:
            for clusterer in clusterers:
                if args.k_sweep:
                    n_rows += len(
                        run_combo_k_sweep(
                            run, variant, clusterer, cfg, run_betriebsdaten, cfg.results_root,
                            use_cache=use_cache,
                        )
                    )
                else:
                    run_combo(
                        run, variant, clusterer, cfg, run_betriebsdaten,
                        cfg.results_root, k=args.k, use_cache=use_cache,
                    )
                    n_rows += 1

    print(f"run_step1: wrote {n_rows} row(s) to {cfg.results_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
