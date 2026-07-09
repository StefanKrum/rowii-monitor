"""Step-2 mode-conditioned anomaly-sweep CLI: prepare -> label -> sweep -> report.

Drives `rowii.anomaly.sweep.run_sweep` (Task S5) over one or more (run, variant,
labels, conditioning, scorer) combinations for two protocols (design spec
`docs/superpowers/specs/2026-07-09-step2-mode-conditioned-ad-design.md` §2-4, plan
`docs/superpowers/plans/2026-07-09-step2-first-package.md` Task S6):

- **within-day**: per selected run, prepare features (`rowii.pipeline.prepare_run`),
  attach labels, then run one sweep per (conditioning, scorer) pair the CLI was asked
  for.
- **cross-day**: calibrate a POOLED reference on one SCADA-covered run ("day A"),
  score every OTHER SCADA-covered run's ("day B") valid windows against it -- a
  cross-day false-alarm-rate matrix, format-compatible with the partner's own
  cross-day comparison table (no values adopted from either side).

Every combo's outputs land under `results/step2/...` (see `_within_day_out_dir`/
`_cross_day_out_dir`), plus two shared, append-only artifacts every combo contributes
a row/section to: `results/step2/summary.csv` and `results/step2/candidate_register.md`.

## Labels (`--labels detected|gt`)

`detected` (the default, and the only run-time-realistic mode -- design spec §2: "Per-
state normal references built from Step-1 detected labels ... GT states used only in
evaluation views"): `rowii.state.detect.run_detection` runs on this run's VALID windows
only (mirrors `scripts/run_step1.py`'s own `_detect_and_report`), then gets scattered
back into a full-length `(W,)` int64 array with the `-1` sentinel on invalid windows.
That sentinel never actually reaches `run_sweep`'s internals: `rowii.anomaly.
references.split_by_segments` (called from `run_sweep`) only ever draws window indices
from `PreparedRun.valid_mask == True` positions, which are EXACTLY the positions this
module filled with a real (>= 0) cluster id -- so `-1` is written for documentation/
debugging clarity only, never read by the sweep.

`gt` loads SCADA (`rowii.scada.labels.load_scada_window_means` + `gt_labels`) and uses
the state STRINGS directly as labels. Binding decision (documented here, not in the
spec's own literal text, which only says GT is for "diagnostics"): windows whose GT
state is `"unknown"` are excluded from the sweep by AND-ing `(state != "unknown")` into
a COPY of `PreparedRun.valid_mask` (`dataclasses.replace` -- the original `PreparedRun`
returned by `prepare_run`/its cache is never mutated), rather than assigning them some
placeholder label -- the same "exclude via mask, not via label value" mechanism
`detected` mode uses for its own invalid windows, so both modes share one exclusion
principle end to end.

## Cross-day: pooled-only (binding simplification, spec §2)

Detected cluster ids from two different days do not refer to the same physical state
(KMeans label 0 on day A is not comparable to label 0 on day B) and GT-state alignment
across days is out of scope for this package (spec: "per-state cross-day needs label
alignment across days ... out of scope; document"). Cross-day therefore NEVER calls
`run_sweep` (which is structurally single-run: one three-way split of ONE `PreparedRun`)
-- `_cross_day_sweep` below hand-builds the same `SweepResult` shape directly: fit a
scorer on day A's fit-part, calibrate its threshold on day A's conformal-part (both from
one `split_by_segments(day_a.segment_ids, day_a_valid_mask, 0.5, seed)` call -- day A's
own top-level calibration/scoring split collapses away since day A never contributes
scoring windows here), then score ALL of day B's valid windows against that one
threshold. The result's single `far_table` row is always labeled `"pooled"`.

`--labels` still applies to cross-day, but its effect is narrower than within-day: since
the pooled path never reads a label VALUE, `gt` mode only tightens which windows are
ELIGIBLE (excluding GT-`"unknown"` windows from both day A and day B, same mechanism as
within-day), while `detected` mode never runs `run_detection` at all for cross-day (there
is no per-label reference to build) and simply uses each day's own `valid_mask` unchanged.

## Output layout

- within-day: `results/step2/within-day/<run>/<variant>-<labels>/<conditioning>-<scorer>/`
  (`far_table.csv`, `scores.parquet`, `candidates.md`) -- exactly the spec's literal path.
- cross-day: the spec only writes `results/step2/cross-day/<variant>/<dayA>__to__<dayB>/`
  literally, with no room for the `--labels`/`--scorer` axes a single invocation can still
  sweep over (`--scorer all`, `--labels gt`) without collision. Binding extension
  (documented here): `results/step2/cross-day/<variant>-<labels>/<dayA>__to__<dayB>/
  <scorer>-pooled/` -- same `<variant>-<labels>` convention as within-day, plus a
  `<scorer>-pooled` leaf (conditioning is always "pooled" for cross-day, so it is folded
  into the leaf name rather than kept as a separate segment).
- `results/step2/summary.csv` (append-only, one row per combo actually written this
  invocation): `run, variant, labels, conditioning, scorer, alpha, per_label_count,
  pooled_realized_far, mean_per_state_far, n_low_confidence, notes`. `run` holds the pair
  string `"<dayA>__to__<dayB>"` for a cross-day row (`notes="cross-day pooled"`, spec's
  literal wording). `per_label_count` is the spec's prose "per-label-count", snake_cased
  to match every other summary column. `pooled_realized_far` is the realized FAR treating
  every non-excluded label's alarms/scored-counts as one combined bucket (identical
  arithmetic to `rowii.anomaly.sweep`'s own per-state aggregate row, recomputed locally
  here for `conditioning="pooled"` sweeps too, which never get that row from `run_sweep`
  itself). `mean_per_state_far` is the plain unweighted mean of each label's own
  `realized_far` (NaN/excluded labels dropped).
- `results/step2/candidate_register.md` (append-only): a static header (written once,
  `_REGISTER_HEADER`) naming this register's purpose plus the ONE external candidate this
  package's spec calls out by name -- the partner's pre-start filling-valve (Fuelldüse)
  observation, clearly source-labeled and never adopted as a value -- followed by one
  `### run / variant-labels / conditioning-scorer` section per combo actually written,
  reproducing that combo's own top-k-per-label table (`source="own sweep"`,
  `assessment="unreviewed"`, both fixed since a script cannot yet judge its own output).

A sweep failing with `ValueError` (e.g. `rowii.anomaly.references.split_by_segments`
finding a degenerate split -- too few segments, or a `calibration_frac` that would empty
one side) is logged and SKIPPED for that one combo, not fatal to the whole invocation --
mirrors `scripts/run_step1.py`'s own "no SCADA coverage" defensive branch, generalized to
every way a single combo can legitimately have nothing to report. The same principle
extends one level earlier: a run whose own `rowii.pipeline.prepare_run` raises
`RuntimeError` (too short/sparse for the requested variant -- e.g. a real "two stray
files" run like `010726-tu1-afternoon`, Task S7 real-data finding) is logged and
excluded -- the whole run for within-day (`_run_within_day_for_run`), or just that one
day and every pair touching it for cross-day (`_run_cross_day`), never the rest of the
invocation.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.anomaly.conformal import ConformalThreshold, calibrate, p_values  # noqa: E402
from rowii.anomaly.references import split_by_segments  # noqa: E402
from rowii.anomaly.scorers import KnnScorer, MahalanobisScorer, Scorer  # noqa: E402
from rowii.anomaly.sweep import SweepConfig, SweepResult, run_sweep  # noqa: E402
from rowii.config import Config, load_config  # noqa: E402
from rowii.io.dataset import RecordingIndex, Run, discover  # noqa: E402
from rowii.io.gantner import read_header  # noqa: E402
from rowii.pipeline import (  # noqa: E402
    _BEATS_INSTALL_HINT,
    PreparedRun,
    _is_beats_variant,
    prepare_run,
)
from rowii.scada.labels import gt_labels, load_scada_window_means  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402
from rowii.state.detect import run_detection  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI enums
# ---------------------------------------------------------------------------

ConditioningName = Literal["per-state", "pooled"]
ScorerName = Literal["knn", "mahalanobis"]

_PROTOCOL_CHOICES: tuple[str, ...] = ("within-day", "cross-day")
_VARIANT_CHOICES: tuple[str, ...] = (
    "audio", "vibration", "fusion", "audio-beats", "fusion-beats",
)
_SCORER_CHOICES: tuple[str, ...] = ("knn", "mahalanobis", "all")
_CONDITIONING_CHOICES: tuple[str, ...] = ("per-state", "pooled", "all")
_LABELS_CHOICES: tuple[str, ...] = ("detected", "gt")

_CONCRETE_SCORERS: tuple[ScorerName, ...] = ("knn", "mahalanobis")
_CONCRETE_CONDITIONINGS: tuple[ConditioningName, ...] = ("per-state", "pooled")

_INVALID_LABEL = -1
"""Sentinel written into `_detected_labels`' full-length label array on invalid
windows -- never read by `run_sweep` (module docstring)."""

_POOLED_LABEL = "pooled"
"""FAR-table label for both `rowii.anomaly.sweep`'s own per-state aggregate row (its
private `_POOLED_ROW_LABEL`, same value, not imported across modules) and this script's
single cross-day row. Collides with a real label only if a state is itself named
"pooled" -- not a name any detected cluster id (int) or GT state string uses."""


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Step-2 mode-conditioned anomaly sweep: per-state (or pooled) "
            "split-conformal scoring over one or more prepared runs, producing a FAR "
            "table, per-window scores, top-k anomaly candidates, and an append-only "
            "candidate register."
        )
    )
    parser.add_argument(
        "--protocol", choices=_PROTOCOL_CHOICES, default="within-day",
        help="'within-day': one blocked split per run. 'cross-day': calibrate on one "
             "SCADA-covered run, score every other one (pooled only).",
    )
    parser.add_argument(
        "--run", default="all",
        help=(
            "Run name to process (within-day only; ignored for --protocol cross-day, "
            "which always sweeps every SCADA-covered run pair). 'all' (default) means "
            "every SCADA-covered run discovered under ROWII_DATA_ROOT."
        ),
    )
    parser.add_argument("--variant", choices=_VARIANT_CHOICES, default="fusion")
    parser.add_argument("--scorer", choices=_SCORER_CHOICES, default="knn")
    parser.add_argument("--conditioning", choices=_CONDITIONING_CHOICES, default="all")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--labels", choices=_LABELS_CHOICES, default="detected",
        help="'detected' (default): Step-1 detected cluster ids. 'gt': SCADA-derived "
             "ground-truth state strings, for diagnostics (design spec §2).",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help=(
            "Disable rowii.pipeline.prepare_run's on-disk feature cache "
            "(results/cache/<run>--<variant>.npz): always recompute features and "
            "never write a cache entry for this invocation."
        ),
    )
    return parser


def _resolve_choice(choice: str, all_value: str, concrete: tuple[str, ...]) -> tuple[str, ...]:
    return concrete if choice == all_value else (choice,)


def _resolve_scorers(choice: str) -> tuple[ScorerName, ...]:
    resolved = _resolve_choice(choice, "all", cast("tuple[str, ...]", _CONCRETE_SCORERS))
    return cast("tuple[ScorerName, ...]", resolved)


def _resolve_conditionings(choice: str) -> tuple[ConditioningName, ...]:
    resolved = _resolve_choice(choice, "all", cast("tuple[str, ...]", _CONCRETE_CONDITIONINGS))
    return cast("tuple[ConditioningName, ...]", resolved)


def _scada_covered_runs(index: RecordingIndex) -> list[Run]:
    """Runs whose own day tree has at least one Betriebsdaten file -- the within-day
    default run set (`--run all`) and cross-day's full "day" set alike. A run with no
    SCADA at all can still be swept in `detected`-labels mode via an explicit
    `--run <name>`, but contributes no SCADA context/GT view, so it is excluded from
    this DEFAULT selection.
    """
    return [r for r in index.runs if index.betriebsdaten_by_day.get(r.day_root)]


def _resolve_runs(choice: str, index: RecordingIndex) -> list[Run]:
    if choice == "all":
        return _scada_covered_runs(index)
    matches = [r for r in index.runs if r.name == choice]
    if not matches:
        logger.warning("run %r not found in discovered index (no matching Run)", choice)
    return matches


def _import_beats_or_exit() -> None:
    try:
        import rowii.signals.beats  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"BEATs featurizer not available ({exc}); {_BEATS_INSTALL_HINT}"
        ) from exc


# ---------------------------------------------------------------------------
# SCADA context (used by both label modes' candidate/register output, and by gt mode
# for the labels themselves)
# ---------------------------------------------------------------------------


def _betriebsdaten_for_grid(betriebsdaten: list[Path], grid: WindowGrid) -> list[Path]:
    """Betriebsdaten files whose hourly span intersects the grid's UTC time range.

    Identical logic to `scripts/run_step1.py`'s own private helper of the same name
    (duplicated rather than imported -- this script must not depend on a SIBLING
    script's internals; `rowii.pipeline`'s own module docstring explains why shared
    Step-1/Step-2 logic lives in the `rowii` package instead, and this ~10-line
    time-overlap filter is small enough that duplicating it costs less than adding a
    script-to-script coupling would).
    """
    grid_end_ns = int(grid.edges_ns()[-1])
    matched = []
    for path in betriebsdaten:
        header = read_header(path)
        file_end_ns = header.t0_ns + round(header.n_frames / header.sample_rate_hz * 1e9)
        if header.t0_ns < grid_end_ns and file_end_ns > grid.t0_ns:
            matched.append(path)
    return sorted(matched)


def _load_run_scada(prepared: PreparedRun, run: Run, index: RecordingIndex) -> pd.DataFrame | None:
    """Per-window SCADA means for *run*'s own day (`rowii.scada.labels.
    load_scada_window_means`), or `None` if this run's day has no Betriebsdaten at all
    (or none whose time range actually overlaps the grid) -- used both for candidate/
    register context columns (P/rpm/KS/Q, any label mode) and, by the caller, to derive
    `gt` labels.
    """
    day_betriebsdaten = index.betriebsdaten_by_day.get(run.day_root, [])
    if not day_betriebsdaten:
        return None
    matched = _betriebsdaten_for_grid(day_betriebsdaten, prepared.grid)
    if not matched:
        return None
    return load_scada_window_means(matched, prepared.grid)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def _detected_labels(prepared: PreparedRun, cfg: Config) -> np.ndarray:
    """Full-length `(W,)` int64 detected cluster-id labels, `_INVALID_LABEL` on invalid
    windows (module docstring) -- mirrors `scripts/run_step1.py`'s own
    `_detect_and_report` scatter-back pattern.
    """
    valid_mask = prepared.valid_mask
    features_valid = prepared.features[valid_mask]
    n_valid = int(valid_mask.sum())
    valid_grid = WindowGrid(
        t0_ns=prepared.grid.t0_ns, window_ns=prepared.grid.window_ns, n_windows=n_valid
    )
    det = run_detection(features_valid, valid_grid, cfg.detect, clusterer="kmeans")
    full_labels = np.full(prepared.features.shape[0], _INVALID_LABEL, dtype=np.int64)
    full_labels[valid_mask] = det.frame_labels
    return full_labels


def _gt_state_labels(scada: pd.DataFrame, cfg: Config) -> np.ndarray:
    """`(W,)` object array of GT state strings for *scada* (module docstring: `gt`
    labels mode)."""
    gt = gt_labels(scada, cfg.gt, window_s=cfg.window.window_s)
    labels: np.ndarray = gt["state"].to_numpy()
    return labels


# ---------------------------------------------------------------------------
# Scorer construction (cross-day only -- within-day delegates to run_sweep, which does
# its own internal dispatch)
# ---------------------------------------------------------------------------


def _make_scorer(name: str) -> Scorer:
    if name == "knn":
        return KnnScorer()
    if name == "mahalanobis":
        return MahalanobisScorer()
    raise ValueError(f"scorer must be 'knn' or 'mahalanobis', got {name!r}")


# ---------------------------------------------------------------------------
# Cross-day sweep (hand-built SweepResult -- see module docstring)
# ---------------------------------------------------------------------------

_FAR_TABLE_COLUMNS: tuple[str, ...] = (
    "label", "n_calibration", "n_scored", "n_alarms", "realized_far", "nominal_alpha",
    "achievable_alpha_floor", "low_confidence", "threshold", "excluded",
)
"""Matches `rowii.anomaly.sweep.SweepResult.far_table`'s documented column contract --
redefined locally rather than importing the sweep module's own private column-name
tuple, since these DataFrames need to share exactly one schema with within-day's
`run_sweep` output (`_write_sweep_outputs`/`_summary_far_metrics` do not otherwise care
which protocol produced them)."""
_SCORES_COLUMNS: tuple[str, ...] = ("window", "label", "score", "p_value", "alarm")
_CANDIDATES_COLUMNS: tuple[str, ...] = ("window", "label", "score", "p_value", "rank")

_CROSS_DAY_SEED = 7
"""Seed for cross-day's single `split_by_segments` call -- matches `SweepConfig.seed`'s
own default (7), for the same reason: an arbitrary-but-fixed, documented choice."""


def _cross_day_sweep(
    prepared_a: PreparedRun,
    valid_a: np.ndarray,
    prepared_b: PreparedRun,
    valid_b: np.ndarray,
    scorer_name: str,
    alpha: float,
    top_k: int,
) -> SweepResult:
    """Fit + calibrate a POOLED scorer on day A's own fit/conformal split, then score
    ALL of day B's eligible windows against that one threshold (module docstring's
    "Cross-day: pooled-only" section) -- returns a `SweepResult` with exactly one
    `far_table` row (`label="pooled"`), so downstream writers never need to
    special-case which protocol produced it.

    Args:
        prepared_a: Day A's `PreparedRun` (calibration source).
        valid_a: Day A's EFFECTIVE valid mask (module docstring: `--labels gt` further
            restricts this beyond `prepared_a.valid_mask`; `detected` mode passes
            `prepared_a.valid_mask` unchanged).
        prepared_b: Day B's `PreparedRun` (scoring target).
        valid_b: Day B's effective valid mask, same convention as `valid_a`.
        scorer_name: `"knn"` or `"mahalanobis"`.
        alpha: Nominal false-alarm rate.
        top_k: Candidate register size.

    Raises:
        ValueError: if `split_by_segments` cannot produce a non-empty fit/conformal
            split of day A's eligible windows, or if day B has zero eligible windows to
            score.
    """
    fit_conformal_split = split_by_segments(prepared_a.segment_ids, valid_a, 0.5, _CROSS_DAY_SEED)
    fit_windows = fit_conformal_split.calibration_windows
    conformal_windows = fit_conformal_split.scoring_windows

    scorer = _make_scorer(scorer_name).fit(prepared_a.features[fit_windows])
    conformal_scores = scorer.score(prepared_a.features[conformal_windows])
    threshold: ConformalThreshold = calibrate(conformal_scores, alpha)

    scoring_windows = np.flatnonzero(valid_b).astype(np.int64)
    if scoring_windows.size == 0:
        raise ValueError("cross-day: day B has zero eligible (valid) windows to score")

    scores = scorer.score(prepared_b.features[scoring_windows])
    p_vals = p_values(scores, conformal_scores)
    alarms = scores > threshold.threshold
    n_scored = int(scoring_windows.shape[0])
    n_alarms = int(alarms.sum())

    far_table = pd.DataFrame(
        [{
            "label": _POOLED_LABEL,
            "n_calibration": float(threshold.n_calibration),
            "n_scored": float(n_scored),
            "n_alarms": float(n_alarms),
            "realized_far": (n_alarms / n_scored) if n_scored > 0 else math.nan,
            "nominal_alpha": alpha,
            "achievable_alpha_floor": threshold.achievable_alpha_floor,
            "low_confidence": threshold.low_confidence,
            "threshold": threshold.threshold,
            "excluded": False,
        }],
        columns=_FAR_TABLE_COLUMNS,
    )
    scores_df = pd.DataFrame(
        {
            "window": scoring_windows,
            "label": [_POOLED_LABEL] * n_scored,
            "score": scores,
            "p_value": p_vals,
            "alarm": alarms,
        },
        columns=_SCORES_COLUMNS,
    )

    # Tie-break identical to rowii.anomaly.sweep._scores_and_candidates: p-value
    # ascending, then score DESCENDING (surfaces the more extreme reading first among
    # windows tied at the achievable-minimum p-value), then window ascending.
    top_order = np.lexsort((scoring_windows, -scores, p_vals))[:top_k]
    candidates_df = pd.DataFrame(
        {
            "window": scoring_windows[top_order],
            "label": [_POOLED_LABEL] * len(top_order),
            "score": scores[top_order],
            "p_value": p_vals[top_order],
            "rank": np.arange(1, len(top_order) + 1, dtype=np.int64),
        },
        columns=_CANDIDATES_COLUMNS,
    )

    return SweepResult(far_table=far_table, scores=scores_df, candidates=candidates_df)


def _cross_day_valid_mask(
    prepared: PreparedRun, run: Run, index: RecordingIndex, cfg: Config, labels_mode: str
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """One day's effective valid mask + SCADA context for cross-day (module docstring:
    `--labels` narrows ELIGIBILITY only, never selects a per-label reference here).
    """
    scada = _load_run_scada(prepared, run, index)
    if labels_mode != "gt":
        return prepared.valid_mask, scada
    if scada is None:
        return np.zeros_like(prepared.valid_mask), None
    state = _gt_state_labels(scada, cfg)
    valid_mask: np.ndarray = prepared.valid_mask & (state != "unknown")
    return valid_mask, scada


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------


def _within_day_out_dir(
    results_root: Path, run_name: str, variant: str, labels_mode: str,
    conditioning: str, scorer: str,
) -> Path:
    return (
        results_root / "step2" / "within-day" / run_name
        / f"{variant}-{labels_mode}" / f"{conditioning}-{scorer}"
    )


def _cross_day_out_dir(
    results_root: Path, variant: str, labels_mode: str, day_a: str, day_b: str, scorer: str
) -> Path:
    return (
        results_root / "step2" / "cross-day" / f"{variant}-{labels_mode}"
        / f"{day_a}__to__{day_b}" / f"{scorer}-pooled"
    )


# ---------------------------------------------------------------------------
# far_table.csv / scores.parquet / candidates.md
# ---------------------------------------------------------------------------

_SCADA_CONTEXT_LABELS: tuple[str, ...] = ("P", "rpm", "KS", "Q")
_SCADA_SOURCE_CHANNELS: dict[str, str] = {
    "P": "power", "rpm": "speed", "KS": "ks_valve", "Q": "reactive",
}
"""Candidate-table context columns -> `rowii.scada.labels.load_scada_window_means`'s own
column names (design spec §2: "SCADA context columns [P, rpm, KS, Q]")."""


def _scada_context_row(scada: pd.DataFrame | None, window: int) -> dict[str, str]:
    """P/rpm/KS/Q values for one candidate window, formatted for markdown -- `"n/a"`
    both when this run has no Betriebsdaten at all (`scada is None`, spec's literal
    "else n/a") and when the specific value is itself NaN for that window.
    """
    if scada is None or window >= len(scada):
        return dict.fromkeys(_SCADA_CONTEXT_LABELS, "n/a")
    row = scada.iloc[window]
    result: dict[str, str] = {}
    for label, channel in _SCADA_SOURCE_CHANNELS.items():
        value = row[channel]
        result[label] = "n/a" if pd.isna(value) else f"{float(value):.3f}"
    return result


def _group_candidates_by_label(candidates: pd.DataFrame) -> list[tuple[object, pd.DataFrame]]:
    """`(label, sub_df)` pairs, one per distinct label in *candidates*, sorted by label
    (never a mix of int/str within one sweep -- `rowii.anomaly.sweep._validate_labels`),
    rows within each group ordered by ascending rank.
    """
    labels = sorted(candidates["label"].unique().tolist())
    return [
        (label, candidates[candidates["label"] == label].sort_values("rank"))
        for label in labels
    ]


def _candidates_markdown(
    candidates: pd.DataFrame, grid: WindowGrid, scada: pd.DataFrame | None, top_k: int
) -> str:
    """`candidates.md` body: top-`top_k` rows PER LABEL with UTC time, score, p-value,
    SCADA context, and a blank `assessment` column for the human review pass (design
    spec §2).
    """
    lines = [f"# Anomaly candidates (top-{top_k} per label)", ""]
    if candidates.empty:
        lines.append("No candidates (every label excluded, or zero scoring windows).")
        lines.append("")
        return "\n".join(lines)

    edges = grid.edges_ns()
    for label, sub in _group_candidates_by_label(candidates):
        lines.append(f"## Label `{label}`")
        lines.append("")
        lines.append(
            "| rank | window | utc_time | score | p_value | P | rpm | KS | Q | assessment |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            window = int(r["window"])
            ts = pd.Timestamp(int(edges[window]), unit="ns", tz="UTC")
            ctx = _scada_context_row(scada, window)
            lines.append(
                f"| {int(r['rank'])} | {window} | {ts.isoformat()} | "
                f"{float(r['score']):.6g} | {float(r['p_value']):.6g} | {ctx['P']} | "
                f"{ctx['rpm']} | {ctx['KS']} | {ctx['Q']} | |"
            )
        lines.append("")
    return "\n".join(lines)


def _write_sweep_outputs(
    out_dir: Path, result: SweepResult, grid: WindowGrid, scada: pd.DataFrame | None, top_k: int
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.far_table.to_csv(out_dir / "far_table.csv", index=False)
    result.scores.to_parquet(out_dir / "scores.parquet", engine="pyarrow", index=False)
    (out_dir / "candidates.md").write_text(
        _candidates_markdown(result.candidates, grid, scada, top_k)
    )


# ---------------------------------------------------------------------------
# Candidate register (append-only)
# ---------------------------------------------------------------------------

_REGISTER_HEADER = """# Step-2 Anomaly Candidate Register

Append-only log of anomaly candidates surfaced by the Step-2 mode-conditioned sweep
(`scripts/run_step2.py`), plus externally reported candidates kept here for
cross-reference only. Every entry carries an explicit **source** (provenance) and an
**assessment** status. No externally sourced value (score, threshold, timing precision,
...) is ever adopted into our own computation -- external entries are comparison-only,
verified independently against our own sweeps where possible (design spec
`docs/superpowers/specs/2026-07-09-step2-mode-conditioned-ad-design.md` §1-2).

## External candidates (partner-reported, provenance labeled)

1. **Pre-start filling-valve (Fuelldüse) sound**, observed before machine start.
   - source: partner slide deck, deck-v3 p.16 (read-only reference)
   - assessment: operator-confirmed normal (partner); to cross-check in our sweeps

## Our sweeps

"""


def _register_path(results_root: Path) -> Path:
    return results_root / "step2" / "candidate_register.md"


def _register_section_markdown(
    run_name: str, variant: str, labels_mode: str, conditioning: str, scorer: str,
    alpha: float, candidates: pd.DataFrame, grid: WindowGrid, scada: pd.DataFrame | None,
) -> str:
    """One append-only register section for a (run, variant, combo) -- the same top-k-
    per-label content as that combo's own `candidates.md` (module docstring), with
    `source`/`assessment` provenance columns appended.
    """
    lines = [
        f"### {run_name} / {variant}-{labels_mode} / {conditioning}-{scorer} (alpha={alpha})",
        "",
    ]
    if candidates.empty:
        lines.append("No candidates (every label excluded, or zero scoring windows).")
        lines.append("")
        return "\n".join(lines) + "\n"

    edges = grid.edges_ns()
    lines.append(
        "| rank | label | window | utc_time | score | p_value | P | rpm | KS | Q | "
        "source | assessment |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, sub in _group_candidates_by_label(candidates):
        for _, r in sub.iterrows():
            window = int(r["window"])
            ts = pd.Timestamp(int(edges[window]), unit="ns", tz="UTC")
            ctx = _scada_context_row(scada, window)
            lines.append(
                f"| {int(r['rank'])} | {label} | {window} | {ts.isoformat()} | "
                f"{float(r['score']):.6g} | {float(r['p_value']):.6g} | {ctx['P']} | "
                f"{ctx['rpm']} | {ctx['KS']} | {ctx['Q']} | own sweep | unreviewed |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def _append_candidate_register(
    results_root: Path, run_name: str, variant: str, labels_mode: str,
    conditioning: str, scorer: str, alpha: float, candidates: pd.DataFrame,
    grid: WindowGrid, scada: pd.DataFrame | None,
) -> None:
    path = _register_path(results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(_REGISTER_HEADER)
    section = _register_section_markdown(
        run_name, variant, labels_mode, conditioning, scorer, alpha, candidates, grid, scada
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(section)


# ---------------------------------------------------------------------------
# summary.csv (append-only)
# ---------------------------------------------------------------------------

_SUMMARY_COLUMNS: tuple[str, ...] = (
    "run", "variant", "labels", "conditioning", "scorer", "alpha", "per_label_count",
    "pooled_realized_far", "mean_per_state_far", "n_low_confidence", "notes",
)


@dataclass(frozen=True)
class _SummaryRow:
    """One `results/step2/summary.csv` row -- see module docstring for column semantics."""

    run: str
    variant: str
    labels: str
    conditioning: str
    scorer: str
    alpha: float
    per_label_count: int
    pooled_realized_far: float
    mean_per_state_far: float
    n_low_confidence: int
    notes: str


def _summary_far_metrics(
    far_table: pd.DataFrame, conditioning: str
) -> tuple[int, float, float, int]:
    """`(per_label_count, pooled_realized_far, mean_per_state_far, n_low_confidence)`
    from one WITHIN-DAY `run_sweep` call's `far_table` (module docstring) -- NOT used
    for cross-day, whose single-row far_table IS the pooled result already
    (`_cross_day_summary_row` reads it directly).

    `conditioning="per-state"`: `run_sweep` already appends an aggregate `label=
    "pooled"` row (`rowii.anomaly.sweep` module docstring point 4) -- read straight
    from it. `conditioning="pooled"`: no such row exists (every per-label row already
    shares one scorer/threshold), so it is recomputed here with the identical
    alarms-over-scored-count arithmetic, restricted to rows that were actually scored
    (`n_scored` not NaN).
    """
    is_aggregate = far_table["label"] == _POOLED_LABEL
    per_label = far_table[~is_aggregate]
    per_label_count = int(len(per_label))
    n_low_confidence = int(per_label["low_confidence"].sum())
    real_far = per_label["realized_far"].dropna()
    mean_per_state_far = float(real_far.mean()) if len(real_far) else math.nan

    if conditioning == "per-state":
        aggregate = far_table[is_aggregate]
        pooled_realized_far = (
            float(aggregate["realized_far"].iloc[0]) if len(aggregate) else math.nan
        )
    else:
        scored = per_label[per_label["n_scored"].notna()]
        total_scored = float(scored["n_scored"].sum())
        total_alarms = float(scored["n_alarms"].sum())
        pooled_realized_far = (total_alarms / total_scored) if total_scored > 0 else math.nan

    return per_label_count, pooled_realized_far, mean_per_state_far, n_low_confidence


def _summary_row(
    run_name: str, variant: str, labels_mode: str, conditioning: str, scorer: str,
    alpha: float, far_table: pd.DataFrame,
) -> _SummaryRow:
    per_label_count, pooled_far, mean_far, n_low_conf = _summary_far_metrics(
        far_table, conditioning
    )
    return _SummaryRow(
        run=run_name, variant=variant, labels=labels_mode, conditioning=conditioning,
        scorer=scorer, alpha=alpha, per_label_count=per_label_count,
        pooled_realized_far=pooled_far, mean_per_state_far=mean_far,
        n_low_confidence=n_low_conf, notes="",
    )


def _cross_day_summary_row(
    day_a: str, day_b: str, variant: str, labels_mode: str, scorer: str,
    alpha: float, far_table: pd.DataFrame,
) -> _SummaryRow:
    """Cross-day's summary row: `far_table` always has exactly one `"pooled"` row
    (`_cross_day_sweep`) -- read it directly rather than through
    `_summary_far_metrics` (which is built around within-day's "aggregate row on top
    of per-label rows" shape, not applicable here). `notes="cross-day pooled"` per the
    spec's literal wording.
    """
    row = far_table.iloc[0]
    far = float(row["realized_far"]) if pd.notna(row["realized_far"]) else math.nan
    return _SummaryRow(
        run=f"{day_a}__to__{day_b}", variant=variant, labels=labels_mode,
        conditioning="pooled", scorer=scorer, alpha=alpha, per_label_count=1,
        pooled_realized_far=far, mean_per_state_far=far,
        n_low_confidence=int(bool(row["low_confidence"])), notes="cross-day pooled",
    )


def _append_summary_row(results_root: Path, row: _SummaryRow) -> None:
    summary_path = results_root / "step2" / "summary.csv"
    row_df = pd.DataFrame([vars(row)], columns=_SUMMARY_COLUMNS)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(summary_path, index=False)


# ---------------------------------------------------------------------------
# within-day orchestration
# ---------------------------------------------------------------------------


def _run_within_day_for_run(
    run: Run,
    variant: str,
    cfg: Config,
    index: RecordingIndex,
    scorers: tuple[ScorerName, ...],
    conditionings: tuple[ConditioningName, ...],
    labels_mode: str,
    alpha: float,
    top_k: int,
    *,
    use_cache: bool,
) -> int:
    """Prepare *run*, attach labels once, then run one sweep per (conditioning,
    scorer) pair, writing that combo's outputs/summary-row/register-section. Returns
    the number of combos actually written (a combo whose sweep raises `ValueError` --
    module docstring -- is logged and skipped, not counted).

    If `prepare_run` itself raises `RuntimeError` (a run too short/sparse for this
    variant -- e.g. a real "two stray files" run, more than 5% of its grid windows
    invalid, `rowii.pipeline.compute_validity_mask`), the WHOLE run is logged and
    skipped (0 combos written) -- the same "not fatal to the whole invocation"
    principle the module docstring already documents for a `run_sweep` `ValueError`,
    extended to this earlier failure mode (Task S7 real-data finding,
    `010726-tu1-afternoon`).
    """
    if _is_beats_variant(variant):
        _import_beats_or_exit()

    try:
        prepared = prepare_run(run, variant, cfg, use_cache=use_cache)
    except RuntimeError as exc:
        logger.warning(
            "run_step2: prepare_run failed for run %r (%s) -- run is too short/sparse "
            "for this variant (module docstring's exclusion principle extended to "
            "prepare_run failures, not just run_sweep ValueErrors) -- skipping (0 "
            "combos written)",
            run.name, exc,
        )
        return 0
    scada = _load_run_scada(prepared, run, index)

    sweep_prepared: PreparedRun
    labels: np.ndarray
    if labels_mode == "gt":
        if scada is None:
            logger.warning(
                "run_step2: no SCADA coverage for run %r -- skipping gt-labels sweep",
                run.name,
            )
            return 0
        labels = _gt_state_labels(scada, cfg)
        valid_mask = prepared.valid_mask & (labels != "unknown")
        sweep_prepared = dataclasses.replace(prepared, valid_mask=valid_mask)
    else:
        labels = _detected_labels(prepared, cfg)
        sweep_prepared = prepared

    n_written = 0
    for conditioning, scorer in itertools.product(conditionings, scorers):
        sweep_cfg = SweepConfig(alpha=alpha, top_k=top_k, conditioning=conditioning, scorer=scorer)
        try:
            result = run_sweep(sweep_prepared, labels, sweep_cfg)
        except ValueError as exc:
            logger.warning(
                "run_step2: sweep failed for %s/%s-%s/%s-%s (%s) -- skipping",
                run.name, variant, labels_mode, conditioning, scorer, exc,
            )
            continue

        out_dir = _within_day_out_dir(
            cfg.results_root, run.name, variant, labels_mode, conditioning, scorer
        )
        _write_sweep_outputs(out_dir, result, sweep_prepared.grid, scada, top_k)
        _append_summary_row(
            cfg.results_root,
            _summary_row(
                run.name, variant, labels_mode, conditioning, scorer, alpha, result.far_table
            ),
        )
        _append_candidate_register(
            cfg.results_root, run.name, variant, labels_mode, conditioning, scorer,
            alpha, result.candidates, sweep_prepared.grid, scada,
        )
        n_written += 1
    return n_written


# ---------------------------------------------------------------------------
# cross-day orchestration
# ---------------------------------------------------------------------------


def _run_cross_day(
    variant: str,
    cfg: Config,
    index: RecordingIndex,
    scorers: tuple[ScorerName, ...],
    labels_mode: str,
    alpha: float,
    top_k: int,
    *,
    use_cache: bool,
) -> int:
    """Every ordered pair of SCADA-covered runs from DIFFERENT day trees (`Run.
    day_root`), same variant (trivially true -- one variant per invocation), skipping
    pairs whose feature dims are incompatible (module docstring). Returns the number
    of (pair, scorer) combos actually written.

    A day whose own `prepare_run` raises `RuntimeError` (too short/sparse for this
    variant, e.g. a real "two stray files" run) is logged and excluded from
    `prepared_by_run` entirely -- every pair touching it is then skipped by the
    `prepared_by_run` membership check below, but pairs between the OTHER, healthy
    days are unaffected (Task S7 real-data finding: `010726-tu1-afternoon` crashed
    this function's `prepared_by_run` prewarm loop, which had no `try/except` at all,
    before this fix -- losing every OTHER day's matrix cell too, not just that one
    day's).
    """
    if _is_beats_variant(variant):
        _import_beats_or_exit()

    days = _scada_covered_runs(index)
    if len(days) < 2:
        logger.warning(
            "run_step2: cross-day needs >= 2 SCADA-covered runs, found %d -- nothing to do",
            len(days),
        )
        return 0

    prepared_by_run: dict[str, tuple[PreparedRun, np.ndarray, pd.DataFrame | None]] = {}
    for run in days:
        try:
            prepared = prepare_run(run, variant, cfg, use_cache=use_cache)
        except RuntimeError as exc:
            logger.warning(
                "run_step2: prepare_run failed for run %r (%s) -- run is too "
                "short/sparse for this variant, excluding from cross-day (every pair "
                "touching it is skipped below, all OTHER days' pairs are unaffected)",
                run.name, exc,
            )
            continue
        valid_mask, scada = _cross_day_valid_mask(prepared, run, index, cfg, labels_mode)
        prepared_by_run[run.name] = (prepared, valid_mask, scada)

    n_written = 0
    for run_a, run_b in itertools.permutations(days, 2):
        if run_a.day_root == run_b.day_root:
            continue  # not genuinely "cross-day" -- same day tree, different session
        if run_a.name not in prepared_by_run or run_b.name not in prepared_by_run:
            continue  # one side failed prepare_run above -- pair has nothing to score

        prepared_a, valid_a, _scada_a = prepared_by_run[run_a.name]
        prepared_b, valid_b, scada_b = prepared_by_run[run_b.name]

        if prepared_a.features.shape[1] != prepared_b.features.shape[1]:
            logger.warning(
                "run_step2: skipping cross-day %s -> %s (incompatible feature dims %d vs %d)",
                run_a.name, run_b.name, prepared_a.features.shape[1], prepared_b.features.shape[1],
            )
            continue

        for scorer in scorers:
            try:
                result = _cross_day_sweep(
                    prepared_a, valid_a, prepared_b, valid_b, scorer, alpha, top_k
                )
            except ValueError as exc:
                logger.warning(
                    "run_step2: cross-day sweep failed for %s -> %s / %s (%s) -- skipping",
                    run_a.name, run_b.name, scorer, exc,
                )
                continue

            out_dir = _cross_day_out_dir(
                cfg.results_root, variant, labels_mode, run_a.name, run_b.name, scorer
            )
            _write_sweep_outputs(out_dir, result, prepared_b.grid, scada_b, top_k)
            _append_summary_row(
                cfg.results_root,
                _cross_day_summary_row(
                    run_a.name, run_b.name, variant, labels_mode, scorer, alpha,
                    result.far_table,
                ),
            )
            _append_candidate_register(
                cfg.results_root, f"{run_a.name}__to__{run_b.name}", variant, labels_mode,
                "pooled", scorer, alpha, result.candidates, prepared_b.grid, scada_b,
            )
            n_written += 1
    return n_written


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config()
    index = discover(cfg.data_root)
    use_cache = not args.no_cache
    scorers = _resolve_scorers(args.scorer)

    if args.protocol == "within-day":
        conditionings = _resolve_conditionings(args.conditioning)
        runs = _resolve_runs(args.run, index)
        n_combos = 0
        for run in runs:
            n_combos += _run_within_day_for_run(
                run, args.variant, cfg, index, scorers, conditionings,
                args.labels, args.alpha, args.top_k, use_cache=use_cache,
            )
        print(
            f"run_step2: wrote {n_combos} within-day combo(s) across {len(runs)} run(s) "
            f"to {cfg.results_root / 'step2'}"
        )
    else:
        if args.conditioning not in ("all", "pooled"):
            logger.info(
                "run_step2: --conditioning=%r is ignored for --protocol cross-day "
                "(always pooled -- module docstring)",
                args.conditioning,
            )
        n_combos = _run_cross_day(
            args.variant, cfg, index, scorers, args.labels, args.alpha, args.top_k,
            use_cache=use_cache,
        )
        print(
            f"run_step2: wrote {n_combos} cross-day pair-combo(s) to "
            f"{cfg.results_root / 'step2'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
