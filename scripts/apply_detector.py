"""Detector-transfer qualitative timeline CLI (package-2 spec D2 stretch, `docs/
superpowers/specs/2026-07-15-step2-scarcity-crossday-beats-design.md` D2): apply a
detector fitted on one SCADA-covered day to a SECOND day with no SCADA ground truth
at all (27.06 permanently lacks a Betriebsdaten folder), producing a purely
QUALITATIVE state timeline -- `segments.csv` + `timeline.md`, both carrying an
explicit "labels are transferred, no ground truth" banner and NO accuracy/ARI/F1 or
any other GT-based metric (spec: "narrative cross-check ... report-only, no values
adopted, no metrics claimed").

Detector transfer itself is `rowii.state.detect.FittedDetector.fit`/`.apply`
(package-2 spec D1, no refit/EM on the apply day) -- identical mechanics to
`scripts/run_step2.py`'s own `cross-day-per-state` protocol, just without any
scoring/conformal step and without a second SCADA-covered day to compare against.

Cluster-id -> mode-name mapping (the `mapped_mode` column) is reporting context
only, never part of detection: `rowii.eval.metrics._majority_mapping` maps each
FIT-day cluster id to whichever SCADA-derived GT state is the majority among the
fit day's own eval windows -- the SAME function `scripts/run_step1.py`'s reports use
for their "Cluster -> state mapping (majority)" section (via `rowii.eval.metrics.
evaluate`'s `EvalResult.state_mapping` field, itself `_majority_mapping`'s own
output). Reused directly here since it lives in `src`, not in a sibling script
(`scripts/run_step2_scarcity.py`'s own "a script must not depend on a SIBLING
script's internals" rule). Calling `_majority_mapping` directly rather than going
through `evaluate()` avoids computing (or depending on) any of `evaluate()`'s
ARI/macro-F1/boundary metrics -- which this script's qualitative-only contract must
never surface -- and avoids `evaluate()`'s hard `ValueError` when a day has zero
eval windows, a real possibility here (`_fit_detector_and_mapping` handles that case
by returning an empty mapping instead of raising). The mapping always comes from the
FIT day alone; the apply day never has GT in this path (spec D2).
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.config import Config, load_config  # noqa: E402
from rowii.eval.metrics import _majority_mapping  # noqa: E402
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
from rowii.state.detect import FittedDetector  # noqa: E402
from rowii.state.segments import to_segments  # noqa: E402

logger = logging.getLogger(__name__)

ClustererName = Literal["kmeans", "gmm"]
"""Matches `rowii.state.detect.FittedDetector.fit`'s own `clusterer` parameter type."""

_VARIANT_CHOICES: tuple[str, ...] = (
    "audio", "vibration", "fusion", "audio-beats", "fusion-beats",
    "audio-tfc", "vibration-tfc", "logmel",
)
"""Duplicated from `scripts/run_step2.py`'s own `_VARIANT_CHOICES` (and `scripts/
run_step2_scarcity.py`'s) -- a script must not depend on a sibling script's
internals (this module's own docstring)."""
_CLUSTERER_CHOICES: tuple[str, ...] = ("kmeans", "gmm")

_INVALID_LABEL = -1
"""Sentinel for invalid windows in a full-length label array -- mirrors `scripts/
run_step2.py`'s/`scripts/run_step2_scarcity.py`'s own `_INVALID_LABEL`."""
_UNKNOWN_GT_STATE = "unknown"


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a detector fitted on one SCADA-covered day to a SECOND day with "
            "no SCADA ground truth, producing a purely QUALITATIVE state timeline "
            "(package-2 spec D2 stretch): segments.csv + timeline.md, no "
            "accuracy/ARI/F1 or any other ground-truth metric."
        )
    )
    parser.add_argument(
        "--fit-run", required=True,
        help="Run name to fit the detector on (needs SCADA for a real mode mapping).",
    )
    parser.add_argument(
        "--apply-run", required=True,
        help="Run name to label with the fit-run's detector (SCADA not needed/used).",
    )
    parser.add_argument("--variant", choices=_VARIANT_CHOICES, default="fusion")
    parser.add_argument("--clusterer", choices=_CLUSTERER_CHOICES, default="kmeans")
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Override Config.data_root from load_config() (env ROWII_DATA_ROOT).",
    )
    parser.add_argument(
        "--results-root", type=Path, default=None,
        help=(
            "Override Config.results_root from load_config() (env "
            "ROWII_RESULTS_ROOT) -- affects both rowii.pipeline.prepare_run's "
            "feature-cache location and this script's own results/step2/transfer/"
            "... output root."
        ),
    )
    return parser


def _resolve_clusterer(choice: str) -> ClustererName:
    """`choice` is `args.clusterer`, already runtime-guaranteed by argparse's
    `choices=_CLUSTERER_CHOICES` to be `"kmeans"` or `"gmm"` -- the `cast` below only
    tells mypy what argparse already enforces at runtime (mirrors `scripts/
    run_step1.py`'s own `_resolve_clusterers`)."""
    return cast(ClustererName, choice)


def _import_beats_or_exit() -> None:
    """Mirrors `scripts/run_step2.py`'s/`scripts/run_step2_scarcity.py`'s own
    private helper of the same name (duplicated, not imported -- see this module's
    own docstring)."""
    try:
        import rowii.signals.beats  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"BEATs featurizer not available ({exc}); {_BEATS_INSTALL_HINT}"
        ) from exc


def _import_tfc_or_exit(cfg: Config, variant: str) -> None:
    """Mirrors `scripts/run_step1.py`'s own private helper of the same name
    (duplicated, not imported -- see this module's own docstring). Extends the
    beats-import-guard pattern (package-4 spec D4): torch missing (checked first)
    -> SystemExit naming the shared `[beats]` extra; else the ONE checkpoint
    relevant to *variant* itself missing -> SystemExit naming its own env var."""
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
# Fit day: detector + majority cluster -> GT-state mapping (reporting only)
# ---------------------------------------------------------------------------


def _betriebsdaten_for_grid(betriebsdaten: list[Path], grid: WindowGrid) -> list[Path]:
    """Betriebsdaten files whose hourly span intersects the grid's UTC time range --
    identical logic to `scripts/run_step1.py`'s and `scripts/run_step2.py`'s own
    private helper of the same name (duplicated, not imported -- see this module's
    own docstring).

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


def _fit_detector_and_mapping(
    fit_run: Run, prepared: PreparedRun, index: RecordingIndex, cfg: Config,
    clusterer: ClustererName,
) -> tuple[FittedDetector, dict[int, str]]:
    """Fit a `FittedDetector` on *prepared*'s valid windows, plus the fit day's own
    cluster-id -> GT-state-name majority mapping (module docstring). The mapping is
    `{}` when the fit day has no Betriebsdaten at all, or no window with a known
    (non-"unknown") GT state -- every `mapped_mode` then falls back to "" for that
    fit run (orchestrator resolution 2), never guessed.
    """
    valid_mask = prepared.valid_mask
    features_valid = prepared.features[valid_mask]
    n_valid = int(valid_mask.sum())
    valid_grid = WindowGrid(
        t0_ns=prepared.grid.t0_ns, window_ns=prepared.grid.window_ns, n_windows=n_valid
    )
    detector, det_valid = FittedDetector.fit(
        features_valid, valid_grid, cfg.detect, clusterer=clusterer
    )
    full_labels = np.full(prepared.features.shape[0], _INVALID_LABEL, dtype=np.int64)
    full_labels[valid_mask] = det_valid.frame_labels

    day_betriebsdaten = index.betriebsdaten_by_day.get(fit_run.day_root, [])
    matched = _betriebsdaten_for_grid(day_betriebsdaten, prepared.grid)
    if not matched:
        logger.warning(
            "apply_detector: no Betriebsdaten for fit run %s -- mapped_mode will be "
            "empty for every segment",
            fit_run.name,
        )
        return detector, {}

    scada = load_scada_window_means(
        matched, prepared.grid, audio_run_offset_ns=run_utc_offset_ns(fit_run)
    )
    gt = gt_labels(scada, cfg.gt, window_s=cfg.window.window_s)
    gt = gt.copy()
    gt.loc[~valid_mask, "state"] = _UNKNOWN_GT_STATE
    eval_mask = gt["state"].to_numpy() != _UNKNOWN_GT_STATE
    if not eval_mask.any():
        logger.warning(
            "apply_detector: no known-GT windows for fit run %s -- mapped_mode will "
            "be empty for every segment",
            fit_run.name,
        )
        return detector, {}

    mapping = _majority_mapping(gt.loc[eval_mask, "state"], full_labels[eval_mask])
    return detector, mapping


# ---------------------------------------------------------------------------
# Apply day: transferred labels -> segment table (no GT anywhere in this path)
# ---------------------------------------------------------------------------


def _apply_and_segment(detector: FittedDetector, prepared: PreparedRun) -> pd.DataFrame:
    """Apply *detector* (fit on a DIFFERENT day) to *prepared*'s valid windows, then
    scatter back to the full grid and build the segment table (`to_segments`) --
    mirrors `scripts/run_step2.py`'s own `_apply_detector_labels` scatter-back
    pattern, but returns the segment table directly since this script has no scoring
    step to feed with a raw per-window label array."""
    valid_mask = prepared.valid_mask
    features_valid = prepared.features[valid_mask]
    n_valid = int(valid_mask.sum())
    valid_grid = WindowGrid(
        t0_ns=prepared.grid.t0_ns, window_ns=prepared.grid.window_ns, n_windows=n_valid
    )
    det_valid = detector.apply(features_valid, valid_grid)
    full_labels = np.full(prepared.features.shape[0], _INVALID_LABEL, dtype=np.int64)
    full_labels[valid_mask] = det_valid.frame_labels
    return to_segments(full_labels, prepared.grid)


# ---------------------------------------------------------------------------
# timeline.md (orchestrator resolution 3: banner + one line per segment, no metrics)
# ---------------------------------------------------------------------------


def _timeline_markdown(fit_run: str, apply_run: str, variant: str, segments: pd.DataFrame) -> str:
    lines = [
        f"# State timeline: {apply_run} ({variant}), transferred from {fit_run}",
        "",
        f"**labels are transferred from {fit_run}; this day has no SCADA ground "
        "truth -- qualitative only.** No quantitative evaluation of any kind is "
        f"included below: {fit_run}'s own detector (fit-day standardisation + "
        f"fit-day HMM decode, no refit) is applied unchanged to {apply_run}'s "
        f"features; the mode name in brackets is {fit_run}'s own majority cluster "
        "-> GT-state mapping, reporting context only, never a claim about this day.",
        "",
        "## Segments",
        "",
    ]
    if segments.empty:
        lines.append("(no segments)")
    for _, row in segments.iterrows():
        mode = row["mapped_mode"] or "unmapped"
        lines.append(
            f"- {row['start_utc'].isoformat()} -> {row['end_utc'].isoformat()} "
            f"({row['duration_s']:.1f}s): {mode} [{int(row['cluster_id'])}]"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.data_root is not None:
        cfg = dataclasses.replace(cfg, data_root=args.data_root)
    if args.results_root is not None:
        cfg = dataclasses.replace(cfg, results_root=args.results_root)

    index: RecordingIndex = discover(cfg.data_root)
    by_name: dict[str, Run] = {r.name: r for r in index.runs}

    unknown = [name for name in (args.fit_run, args.apply_run) if name not in by_name]
    if unknown:
        available = ", ".join(sorted(by_name)) or "(none discovered)"
        print(
            f"apply_detector: unknown run name(s): {', '.join(unknown)}; "
            f"available runs: {available}",
            file=sys.stderr,
        )
        return 2

    fit_run = by_name[args.fit_run]
    apply_run = by_name[args.apply_run]

    if _is_beats_variant(args.variant):
        _import_beats_or_exit()
    if _is_tfc_variant(args.variant):
        _import_tfc_or_exit(cfg, args.variant)

    fit_prepared = prepare_run(fit_run, args.variant, cfg, use_cache=True)
    detector, mapping = _fit_detector_and_mapping(
        fit_run, fit_prepared, index, cfg, _resolve_clusterer(args.clusterer)
    )

    apply_prepared = prepare_run(apply_run, args.variant, cfg, use_cache=True)
    raw_segments = _apply_and_segment(detector, apply_prepared)

    segments = raw_segments.rename(columns={"cluster": "cluster_id"})
    segments["mapped_mode"] = segments["cluster_id"].map(lambda c: mapping.get(int(c), ""))

    out_dir = (
        cfg.results_root / "step2" / "transfer"
        / f"{args.fit_run}--to--{args.apply_run}" / args.variant
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    segments.to_csv(out_dir / "segments.csv", index=False)
    (out_dir / "timeline.md").write_text(
        _timeline_markdown(args.fit_run, args.apply_run, args.variant, segments)
    )

    print(
        f"apply_detector: wrote {len(segments)} segment(s) transferred from "
        f"{args.fit_run} to {args.apply_run} -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
