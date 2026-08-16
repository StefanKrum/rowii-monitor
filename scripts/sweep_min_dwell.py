"""`min_dwell` sweep -- grounding `DetectConfig.min_dwell_s`'s
default in data.

`DetectConfig.min_dwell_s` defaults to 5.0 (`rowii.config`); via
`FittedDetector._finish`'s `min_dwell = max(1, round(min_dwell_s / window_s))`
this is 5 windows at `window_s = 1.0`, and `rowii.state.segments.duration_filter`
merges every detected run shorter than that into a neighbour -- precisely the
mechanism that removes Stefan's "1-second modes" at Step-1. This is Stefan's own
data-grounding motivation (like `rotations-heatmap`/`pillar3-figure` in
`scripts/analyze_days.py`), NOT a replicated partner analysis type -- no
attribution line, and no partner numeric constant is asserted anywhere.

For ONE (fit pool, held-out test run, variant) rotation and each swept
`min_dwell_s` value, this CLI:

1. Rebuilds `Config` via `dataclasses.replace` on `Config.detect.min_dwell_s`,
   refits the pooled DETECTOR arm (`FittedDetector.fit_pooled`) on the pool's
   own FIT side (`rowii.anomaly.pools.build_pool`), and reports Step-1
   `state_ari` on the held-out test run -- the SAME majority-mapped metric as
   `scripts/run_step1.py`'s own `--k-sweep` (`rowii.eval.metrics.
   evaluate(...).state_ari`), so results read alongside it. Detector arm
   ONLY: the mode bank (`rowii.state.modebank.ModeBank`) is dwell-free by
   construction except under its own `--smooth` flag (duration-filter only),
   which this sweep never touches.
2. Runs ONE Step-2 chain FAR spot-check per swept value (e.g. B1->290626-tu
   fusion recalibrate): a trimmed, RECALIBRATE-ONLY duplication of
   `scripts/run_step2.py::_cross_day_pooled_tables`'s recalibrate branch
   (script-sibling rule -- a script never imports another script's internals)
   -- pooled FIT-side references, but a threshold calibrated fresh on the
   TEST run's own calibration-side scores (`scripts/monitor.py`'s recalibrate
   recipe), aggregated across every detected label exactly like
   `rowii.anomaly.sweep.far_row_aggregate` does. This grounds the `min_dwell`
   choice in downstream FAR, not just Step-1 ARI.

Neither `DetectConfig.min_dwell_s`'s default, nor `scripts/monitor.py`'s
`near_transition` window `W` (which reuses the very SAME default) or
`--smooth`'s duration filter, is changed by this script -- the swept
values feed a REPORTED verdict only (`_min_dwell_verdict`), the transplantation
from this sweep's own grounding to those OTHER call sites is a stated, NOT
separately re-optimized, decision. Output:
`results/step2/min-dwell-sweep/<test_run>/<variant>.{csv,json}`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.anomaly.conformal import calibrate  # noqa: E402
from rowii.anomaly.pools import PoolResult, build_pool  # noqa: E402
from rowii.anomaly.references import split_by_segments  # noqa: E402
from rowii.anomaly.sweep import SweepConfig, _make_scorer  # noqa: E402
from rowii.config import Config, load_config  # noqa: E402
from rowii.eval.metrics import evaluate  # noqa: E402
from rowii.io.dataset import (  # noqa: E402
    RecordingIndex,
    Run,
    betriebsdaten_utc_offset_ns,
    discover,
    run_utc_offset_ns,
)
from rowii.io.gantner import read_header  # noqa: E402
from rowii.pipeline import PreparedRun, prepare_run  # noqa: E402
from rowii.scada.labels import gt_labels, load_scada_window_means  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402
from rowii.state.detect import FittedDetector  # noqa: E402

logger = logging.getLogger(__name__)

_INVALID_LABEL = -1
"""Scatter-back sentinel for an invalid window's detected label -- mirrors
`scripts/run_step2.py`'s own `_INVALID_LABEL` (script-sibling rule; a plain
data constant, not a private-API duplication)."""

_DEFAULT_MIN_DWELLS_S: tuple[float, ...] = (5.0, 10.0, 20.0)
_DEFAULT_K = 4
_DEFAULT_ALPHA = 0.05
_DEFAULT_MIN_REF = 20
_SCORER_CHOICES: tuple[str, ...] = (
    "knn", "mahalanobis", "ocsvm", "iforest", "lof", "mlpae", "lstmae", "convae",
)


# ---------------------------------------------------------------------------
# Duplicated script-sibling helpers (rowii.anomaly.pools' module docstring: a
# script must not import another script's internals -- each docstring below
# names which scripts/run_step2.py / scripts/run_modebank.py helper it mirrors)
# ---------------------------------------------------------------------------


def _unknown_run_names(names: list[str], index: RecordingIndex) -> list[str]:
    """Duplicated from `scripts/run_step2.py`'s helper of the same name:
    names in *names* with no matching discovered run, de-duplicated, in the
    order first seen."""
    known = {r.name for r in index.runs}
    return list(dict.fromkeys(n for n in names if n not in known))


def _betriebsdaten_for_grid(betriebsdaten: list[Path], grid: WindowGrid) -> list[Path]:
    """Duplicated from `scripts/run_step2.py`'s helper of the same name:
    Betriebsdaten files whose hourly span intersects *grid*'s true-UTC time
    range."""
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


def _run_gt_states(
    prepared: PreparedRun, run: Run, index: RecordingIndex, cfg: Config
) -> np.ndarray:
    """Duplicated from `scripts/run_modebank.py`'s helper of the same name
    (itself mirroring `scripts/run_step2.py`'s `_load_run_scada` +
    `_gt_state_labels` collapsed into one call): the FULL-length `(W,)` object
    array of GT state strings for *run*. This sweep needs GT on the TEST run
    ONLY (Step-1 `state_ari`) -- fit runs never need GT here (the detector
    transfers unsupervised, and the FAR spot-check scores by DETECTED label,
    not GT), unlike `run_modebank.py`'s bank, which needs it on every run.

    Raises:
        ValueError: *run*'s day has no Betriebsdaten coverage overlapping its
            own grid.
    """
    day_betriebsdaten = index.betriebsdaten_by_day.get(run.day_root, [])
    matched = (
        _betriebsdaten_for_grid(day_betriebsdaten, prepared.grid) if day_betriebsdaten else []
    )
    if not matched:
        raise ValueError(
            f"run {run.name!r} has no Betriebsdaten coverage overlapping its "
            f"own grid -- the min_dwell sweep needs GT on the held-out test "
            f"run (D3b)"
        )
    scada = load_scada_window_means(
        matched, prepared.grid, audio_run_offset_ns=run_utc_offset_ns(run)
    )
    labels: np.ndarray = gt_labels(scada, cfg.gt, window_s=cfg.window.window_s)[
        "state"
    ].to_numpy()
    return labels


def _apply_detector_labels(prepared: PreparedRun, detector: FittedDetector) -> np.ndarray:
    """Duplicated from `scripts/run_step2.py`'s helper of the same name:
    *prepared*'s FULL-length `(W,)` int64 detected labels under *detector*'s
    fit-day parameters (`FittedDetector.apply`, no refit), `_INVALID_LABEL`
    scattered onto every invalid window."""
    valid_mask = prepared.valid_mask
    features_valid = prepared.features[valid_mask]
    n_valid = int(valid_mask.sum())
    valid_grid = WindowGrid(prepared.grid.t0_ns, prepared.grid.window_ns, n_valid)
    det = detector.apply(features_valid, valid_grid)
    full_labels = np.full(prepared.features.shape[0], _INVALID_LABEL, dtype=np.int64)
    full_labels[valid_mask] = det.frame_labels
    return full_labels


def _pool_row_labels(pool: PoolResult, labels_per_run: dict[str, np.ndarray]) -> np.ndarray:
    """Duplicated from `scripts/run_step2.py`'s helper of the same name: per
    stacked pool row, the detected label of its source window (pool rows are
    valid windows by construction, so `_INVALID_LABEL` never appears)."""
    out = np.empty(pool.features.shape[0], dtype=np.int64)
    for member_idx, member in enumerate(pool.members):
        mask = pool.run_index == member_idx
        out[mask] = labels_per_run[member.run_name][pool.window_index[mask]]
    return out


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested directly, tests/test_sweep_min_dwell.py)
# ---------------------------------------------------------------------------


def _min_dwell_windows(min_dwell_s: float, window_s: float) -> int:
    """`max(1, round(min_dwell_s / window_s))` -- the exact `FittedDetector.
    _finish` conversion, duplicated here for the sweep's own reporting/testing."""
    return max(1, round(min_dwell_s / window_s))


def _parse_min_dwells(raw: str) -> tuple[float, ...]:
    """Comma-separated `--min-dwells` values (seconds) -> a float tuple.

    Raises:
        ValueError: *raw* is empty, or any token does not parse as a float.
    """
    values = tuple(float(v.strip()) for v in raw.split(",") if v.strip())
    if not values:
        raise ValueError("--min-dwells got an empty value list")
    return values


def _sweep_state_ari(
    pool_fit_features: np.ndarray,
    prepared_test: PreparedRun,
    gt_test_valid: np.ndarray,
    cfg: Config,
    *,
    k: int,
    min_dwells: Sequence[float],
) -> dict[float, float]:
    """Per swept `min_dwell_s`, refit the pooled
    detector (`FittedDetector.fit_pooled`, `dataclasses.replace` on
    `Config.detect.min_dwell_s`) and report Step-1 `state_ari` on the held-out
    TEST run (`rowii.eval.metrics.evaluate(...).state_ari`, the SAME
    majority-mapped metric as `scripts/run_step1.py`'s own `--k-sweep`).
    Detector arm ONLY (module docstring).

    Raises:
        RuntimeError: `FittedDetector.fit_pooled` could not assign every
            cluster id (k too large for this pool).
        ValueError: `FittedDetector.fit_pooled`/`evaluate` structural guards
            (e.g. k < 1, or the test run has zero known-GT windows).
    """
    out: dict[float, float] = {}
    valid = prepared_test.valid_mask
    n_valid = int(valid.sum())
    for d in min_dwells:
        swept = dataclasses.replace(
            cfg, detect=dataclasses.replace(cfg.detect, min_dwell_s=float(d))
        )
        detector = FittedDetector.fit_pooled(pool_fit_features, swept, k=k)
        grid = WindowGrid(prepared_test.grid.t0_ns, prepared_test.grid.window_ns, n_valid)
        pred = detector.apply(prepared_test.features[valid], grid).frame_labels
        gt = pd.DataFrame({"state": gt_test_valid})
        out[float(d)] = float(evaluate(pred, gt, grid).state_ari)
    return out


def _recalibrate_far(
    pool_fit_features: np.ndarray,
    pool_fit_labels: np.ndarray,
    test_features: np.ndarray,
    labels_test: np.ndarray,
    cal_windows: np.ndarray,
    scoring_windows: np.ndarray,
    sweep_cfg: SweepConfig,
    scorer_name: str,
) -> float:
    """One pooled-detector-label-space realized FAR on the test run's SCORING
    side, RECALIBRATE mode -- the trimmed, recalibrate-only half of
    `scripts/run_step2.py::_cross_day_pooled_tables` (duplicated, script-
    sibling rule): pooled FIT-side references, but a threshold calibrated
    fresh on the TEST run's own calibration-side scores (`scripts/monitor.py`'s
    recalibrate recipe -- references stay the pool's, thresholds only). A
    label without >= `sweep_cfg.min_ref` pooled fit rows, or with zero
    test-calibration windows of its own, contributes no scored/alarm counts
    (mirrors `far_row_excluded`/`far_row_no_conformal_data`'s "contributes
    nothing" rule feeding `rowii.anomaly.sweep.far_row_aggregate`). `NaN` when
    nothing was ever scored at all."""
    all_labels = sorted(
        {int(v) for v in np.unique(pool_fit_labels)}
        | {int(v) for v in np.unique(labels_test[scoring_windows])}
    )
    total_scored = 0
    total_alarms = 0
    for label in all_labels:
        reference = pool_fit_features[pool_fit_labels == label]
        if reference.shape[0] < sweep_cfg.min_ref:
            continue
        scorer = _make_scorer(scorer_name).fit(reference)

        label_cal = cal_windows[labels_test[cal_windows] == label]
        if label_cal.shape[0] == 0:
            continue
        threshold = calibrate(scorer.score(test_features[label_cal]), sweep_cfg.alpha)

        label_scoring = scoring_windows[labels_test[scoring_windows] == label]
        if label_scoring.shape[0] == 0:
            continue
        scores = scorer.score(test_features[label_scoring])
        alarms = scores > threshold.threshold
        total_scored += int(label_scoring.shape[0])
        total_alarms += int(alarms.sum())
    return float(total_alarms / total_scored) if total_scored > 0 else float("nan")


def _sweep_recalibrate_far(
    pool_fit: PoolResult,
    prepared_fit: dict[str, PreparedRun],
    prepared_test: PreparedRun,
    test_run_name: str,
    cal_windows: np.ndarray,
    scoring_windows: np.ndarray,
    cfg: Config,
    *,
    k: int,
    min_dwells: Sequence[float],
    sweep_cfg: SweepConfig,
    scorer_name: str,
) -> dict[float, float]:
    """Per swept `min_dwell_s`: refit the pooled detector, re-derive every
    pool member's + the test run's detected labels under it
    (`_apply_detector_labels`/`_pool_row_labels`), then `_recalibrate_far`.
    Mirrors `_sweep_state_ari`'s own per-value refit loop (detector arm
    only)."""
    out: dict[float, float] = {}
    for d in min_dwells:
        swept = dataclasses.replace(
            cfg, detect=dataclasses.replace(cfg.detect, min_dwell_s=float(d))
        )
        detector = FittedDetector.fit_pooled(pool_fit.features, swept, k=k)
        labels_per_run = {
            name: _apply_detector_labels(prep, detector) for name, prep in prepared_fit.items()
        }
        labels_per_run[test_run_name] = _apply_detector_labels(prepared_test, detector)
        pool_fit_labels = _pool_row_labels(pool_fit, labels_per_run)
        labels_test = labels_per_run[test_run_name]
        out[float(d)] = _recalibrate_far(
            pool_fit.features, pool_fit_labels, prepared_test.features, labels_test,
            cal_windows, scoring_windows, sweep_cfg, scorer_name,
        )
    return out


def _sweep_table(
    min_dwells: Sequence[float],
    ari_by_dwell: dict[float, float],
    far_by_dwell: dict[float, float],
    *,
    window_s: float,
) -> pd.DataFrame:
    """One row per swept `min_dwell_s`, in the given order: the window-count
    conversion (`_min_dwell_windows`), Step-1 `state_ari`, and the recalibrate
    FAR spot-check -- `NaN` where a value was not computed for that dwell."""
    rows = [
        {
            "min_dwell_s": float(d),
            "min_dwell_windows": _min_dwell_windows(float(d), window_s),
            "state_ari": ari_by_dwell.get(float(d), float("nan")),
            "recalibrate_far": far_by_dwell.get(float(d), float("nan")),
        }
        for d in min_dwells
    ]
    return pd.DataFrame(
        rows, columns=["min_dwell_s", "min_dwell_windows", "state_ari", "recalibrate_far"]
    )


def _min_dwell_verdict(ari_by_dwell: dict[float, float], current_default_s: float) -> str:
    """Plain-language verdict on whether this ONE rotation's `state_ari`
    argues for changing `DetectConfig.min_dwell_s`'s default -- reported
    either way ("no DetectConfig default is changed unless the
    data argues for it"). This script never changes the default itself; the
    orchestrator's cross-rotation synthesis reads this verdict
    across all six rotations before any change is considered."""
    finite = {d: v for d, v in ari_by_dwell.items() if np.isfinite(v)}
    if not finite:
        return "no finite state_ari value was computed at any swept min_dwell_s -- no verdict."
    best = max(finite, key=lambda d: finite[d])
    if best == current_default_s:
        return (
            f"min_dwell_s={current_default_s:g}s (the current DetectConfig "
            f"default) already achieves the best state_ari ({finite[best]:.4f}) "
            f"among {sorted(finite)} on this rotation -- no change argued for "
            f"by this rotation alone."
        )
    default_ari = finite.get(current_default_s)
    default_reading = f"{default_ari:.4f}" if default_ari is not None else "not swept"
    return (
        f"min_dwell_s={best:g}s achieves the best state_ari ({finite[best]:.4f}) "
        f"among {sorted(finite)} on this rotation, vs the current DetectConfig "
        f"default {current_default_s:g}s (state_ari={default_reading}) -- "
        f"reported for the orchestrator's cross-rotation synthesis; this "
        f"script does not itself change the default."
    )


def _out_dir(results_root: Path, test_run: str) -> Path:
    """`results/step2/min-dwell-sweep/<test_run>/` -- keyed by the HELD-OUT run,
    mirroring `scripts/run_modebank.py`'s own `_out_dir` convention."""
    return results_root / "step2" / "min-dwell-sweep" / test_run


# ---------------------------------------------------------------------------
# argparse + main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "D3b: sweep DetectConfig.min_dwell_s over a fit pool / held-out "
            "test run rotation, reporting Step-1 state_ari (detector arm "
            "only) plus one recalibrate FAR spot-check per value -- grounds "
            "the min_dwell default in data (spec §3.D3(b))."
        )
    )
    parser.add_argument(
        "--fit-runs", required=True,
        help=(
            "Comma-separated fit-run names, pool order matters "
            "(FittedDetector.fit_pooled's row-order note); the held-out "
            "--test-run must not appear here."
        ),
    )
    parser.add_argument("--test-run", required=True, help="The ONE held-out test run.")
    parser.add_argument("--variant", required=True, help="e.g. audio, vibration, fusion.")
    parser.add_argument(
        "--min-dwells", default="5,10,20",
        help="Comma-separated min_dwell_s values in seconds (default: 5,10,20).",
    )
    parser.add_argument("--k", type=int, default=_DEFAULT_K, help="Pooled cluster count.")
    parser.add_argument(
        "--alpha", type=float, default=_DEFAULT_ALPHA,
        help="Conformal alpha for the recalibrate FAR spot-check.",
    )
    parser.add_argument("--scorer", choices=_SCORER_CHOICES, default="knn")
    parser.add_argument("--min-ref", type=int, default=_DEFAULT_MIN_REF)
    parser.add_argument(
        "--out", default=None, help="Output root (default: <results_root>)."
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable rowii.pipeline.prepare_run's on-disk feature cache.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    fit_run_names = [n.strip() for n in args.fit_runs.split(",") if n.strip()]
    if not fit_run_names:
        parser.error("--fit-runs got an empty run-name list")
    if len(set(fit_run_names)) != len(fit_run_names):
        parser.error(
            "--fit-runs contains duplicate run name(s) -- a run pools its sides once"
        )
    if args.test_run in fit_run_names:
        parser.error(
            f"--test-run {args.test_run!r} is listed in --fit-runs: the held-out "
            f"day must not be a pool member"
        )
    try:
        min_dwells = _parse_min_dwells(str(args.min_dwells))
    except ValueError as exc:
        parser.error(str(exc))

    cfg = load_config()
    index = discover(cfg.data_root)

    unknown = _unknown_run_names([*fit_run_names, args.test_run], index)
    if unknown:
        available = ", ".join(sorted({r.name for r in index.runs})) or "(none discovered)"
        print(
            f"sweep_min_dwell: unknown run name(s): {', '.join(unknown)}; "
            f"available runs: {available}",
            file=sys.stderr,
        )
        return 2

    runs_by_name = {r.name: r for r in index.runs}
    fit_runs = [runs_by_name[name] for name in fit_run_names]
    test_run_obj = runs_by_name[args.test_run]

    prepared_all: dict[str, PreparedRun] = {}
    for run in (*fit_runs, test_run_obj):
        try:
            prepared_all[run.name] = prepare_run(
                run, args.variant, cfg, use_cache=not args.no_cache
            )
        except RuntimeError as exc:
            print(
                f"sweep_min_dwell: prepare_run failed for run {run.name!r} ({exc})",
                file=sys.stderr,
            )
            return 2
    prepared_fit = {name: prepared_all[name] for name in fit_run_names}
    prepared_test = prepared_all[test_run_obj.name]

    feature_names = list(next(iter(prepared_fit.values())).feature_names)
    test_feature_names = list(prepared_test.feature_names)
    if test_feature_names != feature_names:
        diverging = [
            (a, b) for a, b in zip(feature_names, test_feature_names, strict=False) if a != b
        ]
        print(
            f"sweep_min_dwell: test run {test_run_obj.name!r} feature contract "
            f"does not match the fit pool's ({len(test_feature_names)} vs "
            f"{len(feature_names)} column(s); first divergence(s): "
            f"{diverging[:3]}) -- refusing to score a positionally-misaligned "
            f"contract",
            file=sys.stderr,
        )
        return 2

    try:
        gt_full_test = _run_gt_states(prepared_test, test_run_obj, index, cfg)
    except ValueError as exc:
        print(f"sweep_min_dwell: {exc}", file=sys.stderr)
        return 2

    sweep_cfg = SweepConfig(alpha=args.alpha, min_ref=args.min_ref, scorer=args.scorer)
    pool_fit = build_pool(prepared_fit, "fit", sweep_cfg)
    if pool_fit.features.shape[0] == 0:
        print(
            "sweep_min_dwell: the pooled FIT side is empty (every fit run's "
            "splits were degenerate -- see the build_pool warnings above) -- "
            "nothing to fit on",
            file=sys.stderr,
        )
        return 2

    valid = prepared_test.valid_mask
    gt_test_valid = gt_full_test[valid]

    try:
        ari_by_dwell = _sweep_state_ari(
            pool_fit.features, prepared_test, gt_test_valid, cfg,
            k=args.k, min_dwells=min_dwells,
        )
    except (RuntimeError, ValueError) as exc:
        print(
            f"sweep_min_dwell: k too large for this pool: fit_pooled(k={args.k}) "
            f"on {pool_fit.features.shape[0]} pooled fit window(s) failed ({exc}) "
            f"-- pick a smaller --k",
            file=sys.stderr,
        )
        return 2

    try:
        top = split_by_segments(
            prepared_test.segment_ids, prepared_test.valid_mask,
            sweep_cfg.calibration_frac, sweep_cfg.seed,
        )
    except ValueError as exc:
        logger.warning(
            "sweep_min_dwell: test run %r cannot form its own calibration/"
            "scoring split (%s) -- the recalibrate FAR spot-check is skipped "
            "(NaN every value); the state_ari table is still written",
            test_run_obj.name, exc,
        )
        far_by_dwell: dict[float, float] = {float(d): float("nan") for d in min_dwells}
    else:
        cal_windows, scoring_windows = top.calibration_windows, top.scoring_windows
        far_by_dwell = _sweep_recalibrate_far(
            pool_fit, prepared_fit, prepared_test, test_run_obj.name,
            cal_windows, scoring_windows, cfg,
            k=args.k, min_dwells=min_dwells, sweep_cfg=sweep_cfg, scorer_name=args.scorer,
        )

    table = _sweep_table(min_dwells, ari_by_dwell, far_by_dwell, window_s=cfg.window.window_s)
    verdict = _min_dwell_verdict(ari_by_dwell, cfg.detect.min_dwell_s)

    out_root = Path(args.out) if args.out is not None else cfg.results_root
    out_dir = _out_dir(out_root, test_run_obj.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{args.variant}.csv", index=False)

    sidecar = {
        "fit_runs": fit_run_names,
        "test_run": test_run_obj.name,
        "variant": args.variant,
        "k": args.k,
        "alpha": args.alpha,
        "scorer": args.scorer,
        "min_ref": args.min_ref,
        "min_dwells_s": list(min_dwells),
        "current_default_min_dwell_s": cfg.detect.min_dwell_s,
        "verdict": verdict,
        "provenance_note": (
            "Stefan's own min_dwell/dwell-grounding motivation (spec D3(b)); "
            "no partner analysis type or number is involved. state_ari is "
            "the SAME majority-mapped metric as the P7 k-selection "
            "(rowii.eval.metrics.evaluate); recalibrate_far is a trimmed, "
            "recalibrate-only duplication of "
            "scripts/run_step2.py::_cross_day_pooled_tables (script-sibling "
            "rule)."
        ),
    }
    (out_dir / f"{args.variant}.json").write_text(json.dumps(sidecar, indent=2) + "\n")

    print(
        f"sweep_min_dwell: wrote {out_dir / (args.variant + '.csv')} and .json "
        f"({len(min_dwells)} min_dwell value(s), test run {test_run_obj.name!r})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
