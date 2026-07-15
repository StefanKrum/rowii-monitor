"""Step-1 results-analysis package: consolidates `results/summary.csv` and per-run
artifacts (frame_labels.parquet, segments.csv) into `results/analysis/` (spec: this
task's own brief, no docs/superpowers spec -- a post-hoc analysis pass over the
already-shipped Step-1 grid, not a pipeline change).

Five outputs, each independently rerunnable:

1. `overview.md` + `overview.png` -- consolidated state-level/strict/boundary table
   and a headline dot chart, straight from `summary.csv`.
2. `error_anatomy.md` + `error_vs_boundary.png` -- per-window majority-mapped errors
   bucketed by distance to the nearest GT state-change, plus per-state precision/recall.
3. `gt_sensitivity.md` -- how 010726-tu_ph_tu/fusion-kmeans's EXISTING predictions
   score against six perturbed `GtRules` (predictions fixed, only GT varies).
4. `modality_by_mode.md` -- per-state recall matrix (state x variant) for
   010726-tu_ph_tu and 250526-tu.
5. `crosscheck_partner.md` -- our detected segment timeline vs. the partner's
   documented session facts, independent cross-reference only (see its own header).

Deliverables 2-4 need per-window ground truth that `frame_labels.parquet` does not
carry (it only stores the STRICT Hungarian `mapped_state`, not the majority-mapped
one, and no GT at all -- see `rowii.eval.report.write_report`) -- this module
RECOMPUTES it via the existing pipeline (`rowii.scada.labels.gt_labels`,
`scripts.run_step1.build_run_grid`/`load_run_gt`), never via new labeling logic. The
recompute never re-extracts audio/vibration features: a window's validity is read
directly off its already-written `cluster` column (`-1` marks an invalid window,
`scripts.run_step1._detect_and_report`'s own `_INVALID_LABEL` sentinel), which is
exactly the set `load_run_gt`'s own `valid_mask` argument would force to "unknown" --
so passing `cluster != -1` as that mask reproduces the original pipeline's GT
exactly, without redoing the expensive part.

No existing pipeline module (`src/rowii/**`, `scripts/run_step1.py`) is modified by
this script; it only imports and calls them.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import; headless-safe backend.

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from collections.abc import Mapping, Sequence  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPTS_DIR.parent / "src"
for _extra_path in (str(_SCRIPTS_DIR), str(_SRC_DIR)):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

import run_step1  # noqa: E402

from rowii.config import Config, GtRules, load_config  # noqa: E402
from rowii.eval.metrics import EvalResult, evaluate  # noqa: E402
from rowii.io.dataset import RecordingIndex, Run, discover  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402

logger = logging.getLogger(__name__)

_ANALYSIS_DIR_NAME = "analysis"

# ---------------------------------------------------------------------------
# Palette (dataviz skill's validated categorical/ordinal steps -- see
# `results/analysis` generation notes; `node scripts/validate_palette.js` in the
# skill's own directory confirms every step used here passes light-mode checks).
# ---------------------------------------------------------------------------
_COLOR_ACCURACY = "#2a78d6"  # categorical slot 1 (blue)
_COLOR_ARI = "#1baf7a"  # categorical slot 2 (aqua)
_COLOR_STEM = "#c3c2b7"  # baseline/axis muted gray
_COLOR_GRIDLINE = "#e1e0d9"  # hairline gridline gray
_COLOR_TEXT_SECONDARY = "#52514e"
_COLOR_TEXT_MUTED = "#898781"
_COLOR_SURFACE = "#fcfcfb"
# Ordinal ramp (light -> dark), one hue, for the 5 boundary-distance buckets --
# darkest = closest to a GT boundary (0-5s), lightest = farthest (>60s).
_BUCKET_COLORS: tuple[str, ...] = ("#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef")

_BOUNDARY_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 5.0, "0-5"),
    (5.0, 15.0, "5-15"),
    (15.0, 30.0, "15-30"),
    (30.0, 60.0, "30-60"),
    (60.0, float("inf"), ">60"),
)
_BUCKET_ORDER: tuple[str, ...] = tuple(label for *_ignore, label in _BOUNDARY_BUCKETS)
_NEAR_BOUNDARY_BUCKETS: tuple[str, ...] = ("0-5", "5-15")
"""Buckets counted as "within 15 s of a GT boundary" for the error-anatomy headline."""

_RERUN_RUN = "250526-tu"
_RERUN_VARIANTS: tuple[str, ...] = ("audio", "vibration", "fusion")
_RERUN_CLUSTERER = "kmeans"

_ERROR_ANATOMY_COMBOS: tuple[tuple[str, str], ...] = (
    ("010726-tu_ph_tu", "audio"),
    ("010726-tu_ph_tu", "vibration"),
    ("010726-tu_ph_tu", "fusion"),
    ("010726-tu1-morning", "fusion"),
    ("290626-tu", "fusion"),
    (_RERUN_RUN, "audio"),
    (_RERUN_RUN, "vibration"),
    (_RERUN_RUN, "fusion"),
)
"""Every (run, variant) combo with a kmeans results directory AND real GT coverage
(excludes `270626-pu_ph_pu_ph_pu_ph-1`, which has zero SCADA coverage -- see
`rowii.eval.metrics.evaluate`'s "no eval windows" contract)."""

_MODALITY_BY_MODE_RUNS: tuple[str, ...] = ("010726-tu_ph_tu", _RERUN_RUN)
_MODALITY_BY_MODE_VARIANTS: tuple[str, ...] = ("audio", "vibration", "fusion")

_SENSITIVITY_RUN = "010726-tu_ph_tu"
_SENSITIVITY_VARIANT = "fusion"
_SENSITIVITY_FACTORS: tuple[str, ...] = ("power_eps_mw", "ph_min_dwell_s", "transition_buffer_s")
_SENSITIVITY_MULTIPLIERS: tuple[float, ...] = (0.5, 2.0)
_SENSITIVITY_ROBUST_TOLERANCE = 0.02
"""Absolute-value tolerance for the "robust" verdict on state ARI / state accuracy --
chosen as roughly the natural run-to-run spread already seen in the k-sweep grid
(010726-tu_ph_tu fusion-kmeans state ARI ranges 0.907-0.930 across k=3..6, a spread
of ~0.023; see `results/summary.csv`'s k-sweep rows), i.e. a perturbation that moves
the metric less than a plausible "which k did we pick" swing is not a meaningful
sensitivity."""

_CROSSCHECK_RUNS: tuple[str, ...] = ("010726-tu_ph_tu", "290626-tu")
_PARTNER_PH_HOLD_MINUTES = 37.0
_PARTNER_SOURCE = (
    "Bruno's aggregated-report-v2.md (independent, read-only reference material), "
    "section 5 'June-29 day summary'"
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunVariantEval:
    """Recomputed GT + existing kmeans predictions for one (run, variant) combination."""

    run: str
    variant: str
    grid: WindowGrid
    gt: pd.DataFrame
    """`rowii.scada.labels.gt_labels` output (state, load_bin), index 0..grid.n_windows-1;
    invalid windows (see `cluster`) already forced to "unknown" by `load_run_gt`."""
    cluster: np.ndarray
    """Per-window predicted cluster id from `frame_labels.parquet`'s `cluster` column,
    shape (grid.n_windows,); -1 marks an invalid (never-clustered) window."""
    ev: EvalResult
    """`rowii.eval.metrics.evaluate(cluster, gt, grid)` output -- majority (`state_*`)
    and strict (Hungarian) metrics/mappings for this combination."""

    @property
    def window_s(self) -> float:
        return self.grid.window_ns / 1e9


@dataclass(frozen=True)
class SensitivityObservation:
    """One perturbed-`GtRules` evaluation of the SAME (fixed) predictions."""

    factor: str
    multiplier: float
    perturbed_value: float
    state_ari: float
    state_accuracy: float


@dataclass(frozen=True)
class DurationComparison:
    """Our computed duration vs. a partner-documented duration, both in seconds."""

    label: str
    ours_s: float
    theirs_s: float
    delta_s: float
    delta_pct: float


# ---------------------------------------------------------------------------
# Generic markdown helper (mirrors rowii.eval.report's hand-rolled table helpers --
# no extra dependency such as `tabulate` for `DataFrame.to_markdown()`).
# ---------------------------------------------------------------------------


def _dataframe_to_markdown(df: pd.DataFrame, *, index_label: str | None = None) -> str:
    """Minimal GitHub-flavored markdown table renderer for an already-formatted
    (all-string-or-scalar) DataFrame. Does not round or reformat values -- callers
    format floats before calling this, so every table in this module renders numbers
    the same, explicit way."""
    columns = ([index_label] if index_label else []) + [str(c) for c in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for idx, row in df.iterrows():
        cells = ([str(idx)] if index_label else []) + [str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def _fmt_float(value: float, decimals: int = 3) -> str:
    """`f"{value:.{decimals}f}"`, or the literal string `"n/a"` for NaN/None -- every
    markdown table in this module renders a missing metric this way instead of a
    blank or ambiguous cell."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Deliverable 1: overview
# ---------------------------------------------------------------------------

_OVERVIEW_COLUMNS: tuple[str, ...] = (
    "run", "variant", "clusterer", "k",
    "state_ari", "state_accuracy", "state_macro_f1",
    "ari", "macro_f1", "boundary_median_abs_s",
    "n_eval", "notes",
)
_OVERVIEW_FLOAT_COLUMNS: tuple[str, ...] = (
    "state_ari", "state_accuracy", "state_macro_f1", "ari", "macro_f1",
    "boundary_median_abs_s",
)


def primary_grid_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Rows suitable for the headline table/chart: excludes k-sweep duplicates and
    the no-SCADA-coverage row (`n_eval == 0`) -- one row per (run, variant, clusterer)
    at its default k. Order is preserved (whatever order *summary* itself has)."""
    notes = summary["notes"].fillna("")
    n_eval = pd.to_numeric(summary["n_eval"], errors="coerce").fillna(0)
    mask = (notes == "") & (n_eval > 0)
    return summary[mask].reset_index(drop=True)


def format_overview_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Select + order + round `summary.csv`'s columns for the consolidated overview
    table (state-level primary + strict + boundary). Every float metric renders to 3
    decimals (matching the README grid convention) or the literal string "n/a" for
    NaN (e.g. the no-SCADA-coverage row) -- see `_fmt_float`."""
    out = summary[list(_OVERVIEW_COLUMNS)].copy()
    for col in _OVERVIEW_FLOAT_COLUMNS:
        out[col] = out[col].apply(_fmt_float)
    out["notes"] = out["notes"].fillna("")
    out["n_eval"] = out["n_eval"].fillna(0).astype("int64")
    return out


def plot_overview(rows: pd.DataFrame, out_path: Path) -> None:
    """Horizontal dumbbell/lollipop chart: one row per (run, variant), two dots
    (state accuracy, state ARI) joined by a thin baseline stem -- avoids a dual-axis
    or grouped-bar chart for two same-scale [0, 1] metrics (dataviz skill: "one axis").
    """
    labels = [f"{r['run']} / {r['variant']}" for _, r in rows.iterrows()]
    accuracy = pd.to_numeric(rows["state_accuracy"], errors="coerce").to_numpy()
    ari = pd.to_numeric(rows["state_ari"], errors="coerce").to_numpy()
    y = np.arange(len(rows))

    height = max(2.5, 0.45 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(9, height))
    fig.patch.set_facecolor(_COLOR_SURFACE)
    ax.set_facecolor(_COLOR_SURFACE)

    for yi, acc, ar in zip(y, accuracy, ari, strict=True):
        if np.isnan(acc) or np.isnan(ar):
            continue
        ax.plot([acc, ar], [yi, yi], color=_COLOR_STEM, linewidth=1.5, zorder=1)

    ax.scatter(accuracy, y, s=70, color=_COLOR_ACCURACY, zorder=3, label="state accuracy")
    ax.scatter(ari, y, s=70, color=_COLOR_ARI, zorder=3, label="state ARI")

    for yi, acc, ar in zip(y, accuracy, ari, strict=True):
        if not np.isnan(acc):
            ax.annotate(
                f"{acc:.3f}", (acc, yi), xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=7.5, color=_COLOR_TEXT_SECONDARY,
            )
        if not np.isnan(ar):
            ax.annotate(
                f"{ar:.3f}", (ar, yi), xytext=(0, -11), textcoords="offset points",
                ha="center", fontsize=7.5, color=_COLOR_TEXT_SECONDARY,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=_COLOR_TEXT_SECONDARY)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel("score", color=_COLOR_TEXT_MUTED)
    ax.set_title(
        "State-level accuracy vs. ARI, per run x variant (KMeans, majority mapping)",
        fontsize=11, color="#0b0b0b", loc="left",
    )
    ax.grid(axis="x", color=_COLOR_GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine_name in ("top", "right", "left"):
        ax.spines[spine_name].set_visible(False)
    ax.spines["bottom"].set_color(_COLOR_STEM)
    ax.tick_params(axis="both", colors=_COLOR_TEXT_MUTED, length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08 if height < 4 else -0.04),
              ncol=2, frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_overview(summary: pd.DataFrame, out_dir: Path) -> None:
    table = format_overview_table(summary)
    lines = [
        "# Step-1 overview",
        "",
        "Consolidated state-level (primary), strict (1:1 Hungarian), and boundary "
        "metrics, one row per (run, variant, clusterer) combination in "
        "`results/summary.csv` (`notes` column carries `k-sweep` for k-sweep rows and "
        "`no SCADA coverage` for the SCADA-less 27.06 run). See "
        "`rowii.eval.metrics` for the exact definition of every column.",
        "",
        _dataframe_to_markdown(table),
        "",
        "![state accuracy vs ARI by run x variant](overview.png)",
        "",
    ]
    (out_dir / "overview.md").write_text("\n".join(lines))

    chart_rows = primary_grid_rows(summary)
    plot_overview(chart_rows, out_dir / "overview.png")


# ---------------------------------------------------------------------------
# Deliverable 2: error anatomy
# ---------------------------------------------------------------------------


def gt_boundary_distances_s(gt_states: Sequence[str], window_s: float) -> np.ndarray:
    """Distance (seconds) from every window to the nearest GT state-CHANGE point, on
    the full timeline. A change touching "unknown" on either side is excluded
    (mirrors `rowii.eval.metrics`'s own boundary-deviation convention exactly;
    duplicated here rather than imported since that helper is private and this
    module intentionally stays decoupled from `rowii.eval.metrics`'s internals).

    Returns an array of NaN (one per window) when *gt_states* has zero countable
    changes (e.g. a single-state run).
    """
    n = len(gt_states)
    changes = [
        i
        for i in range(1, n)
        if gt_states[i] != gt_states[i - 1]
        and "unknown" not in (gt_states[i], gt_states[i - 1])
    ]
    if not changes:
        return np.full(n, np.nan)

    idx = np.arange(n, dtype=np.float64)
    changes_arr = np.asarray(changes, dtype=np.float64)
    dist_windows = np.min(np.abs(idx[:, None] - changes_arr[None, :]), axis=1)
    result: np.ndarray = dist_windows * window_s
    return result


def boundary_bucket(distance_s: float) -> str:
    """Bucket label for a boundary distance in seconds (`_BOUNDARY_BUCKETS`, half-open
    `[lo, hi)` intervals). NaN (no GT changes at all in this run) maps to `"n/a"`."""
    if distance_s is None or np.isnan(distance_s):
        return "n/a"
    for lo, hi, label in _BOUNDARY_BUCKETS:
        if lo <= distance_s < hi:
            return label
    return _BOUNDARY_BUCKETS[-1][2]  # unreachable (last bucket's hi is +inf); mypy/defensive


def build_eval_window_table(
    gt_states: Sequence[str],
    cluster: np.ndarray,
    state_mapping: Mapping[int, str],
    window_s: float,
) -> pd.DataFrame:
    """Tidy per-window table restricted to EVAL windows (`gt_state != "unknown"`):
    columns `gt_state`, `cluster`, `pred_state` (majority-mapped), `correct`
    (bool), `boundary_distance_s`, `bucket`.

    `boundary_distance_s` is computed on the FULL timeline (all windows, matching
    `rowii.eval.metrics`'s own boundary-deviation convention) before the eval-window
    restriction is applied below.
    """
    if len(gt_states) != len(cluster):
        raise ValueError(
            f"gt_states ({len(gt_states)}) and cluster ({len(cluster)}) must be the same length"
        )
    distances = gt_boundary_distances_s(gt_states, window_s)
    pred_state = [state_mapping.get(int(c), "unknown") for c in cluster]
    df = pd.DataFrame(
        {
            "gt_state": list(gt_states),
            "cluster": [int(c) for c in cluster],
            "pred_state": pred_state,
            "boundary_distance_s": distances,
        }
    )
    df["correct"] = df["gt_state"] == df["pred_state"]
    df["bucket"] = df["boundary_distance_s"].map(boundary_bucket)
    return df[df["gt_state"] != "unknown"].reset_index(drop=True)


def error_anatomy_histogram(window_table: pd.DataFrame) -> pd.DataFrame:
    """Bucketed error counts/rates from one or more concatenated
    `build_eval_window_table` outputs. One row per bucket (fixed `_BUCKET_ORDER`):
    `n_eval_windows`, `n_errors`, `error_rate` (errors / eval windows in that bucket),
    `pct_of_all_errors` (that bucket's share of all errors COUNTED IN THIS TABLE,
    i.e. across `_BUCKET_ORDER` only).

    A window whose run has zero GT state changes at all gets `bucket == "n/a"`
    (see `boundary_bucket`) and is excluded from every row here; `pct_of_all_errors`
    is normalized against that same exclusion (sums to 1.0 over the returned rows
    whenever at least one error is bucketed), so it never silently double-counts an
    "n/a" error in the denominator while dropping it from every visible bucket.
    """
    counts = [
        (label, len(sub), int((~sub["correct"]).sum()))
        for label, sub in (
            (label, window_table[window_table["bucket"] == label]) for label in _BUCKET_ORDER
        )
    ]
    total_errors = sum(n_err for _label, _n_eval, n_err in counts)
    rows = [
        {
            "bucket_s": label,
            "n_eval_windows": n_eval,
            "n_errors": n_err,
            "error_rate": (n_err / n_eval) if n_eval else float("nan"),
            "pct_of_all_errors": (n_err / total_errors) if total_errors else float("nan"),
        }
        for label, n_eval, n_err in counts
    ]
    return pd.DataFrame(rows)


def per_state_precision_recall(
    window_table: pd.DataFrame, states: Sequence[str] | None = None
) -> pd.DataFrame:
    """Precision/recall/support per GT state from a tidy eval-window table (`gt_state`,
    `pred_state` columns) -- the majority-mapped family. A state absent from GT has
    recall NaN (undefined); one never predicted has precision NaN.
    """
    gt = window_table["gt_state"]
    pred = window_table["pred_state"]
    if states is None:
        states = sorted(set(gt.unique()) | set(pred.unique()))
    rows = []
    for state in states:
        tp = int(((gt == state) & (pred == state)).sum())
        n_gt = int((gt == state).sum())
        n_pred = int((pred == state).sum())
        recall = (tp / n_gt) if n_gt else float("nan")
        precision = (tp / n_pred) if n_pred else float("nan")
        rows.append({"state": state, "precision": precision, "recall": recall, "support": n_gt})
    return pd.DataFrame(rows)


def plot_error_histogram(hist: pd.DataFrame, out_path: Path) -> None:
    """Ordinal bar chart: % of all (pooled) errors per boundary-distance bucket,
    darkest (closest to a GT boundary) to lightest (farthest) -- one hue, monotone
    lightness (dataviz skill's ordinal ramp), since bucket order carries meaning.
    """
    x = np.arange(len(hist))
    pct = (hist["pct_of_all_errors"].to_numpy() * 100.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(_COLOR_SURFACE)
    ax.set_facecolor(_COLOR_SURFACE)

    bars = ax.bar(x, pct, color=_BUCKET_COLORS[: len(hist)], width=0.6, zorder=2)
    for rect, value, n_err in zip(bars, pct, hist["n_errors"], strict=True):
        if np.isnan(value):
            continue
        ax.annotate(
            f"{value:.1f}% (n={int(n_err)})",
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5,
            color=_COLOR_TEXT_SECONDARY,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} s" for b in hist["bucket_s"]], color=_COLOR_TEXT_SECONDARY)
    ax.set_ylabel("% of all errors (pooled across analyzed run x variant combos)",
                   color=_COLOR_TEXT_MUTED, fontsize=9)
    ax.set_ylim(0, max(1.0, float(np.nanmax(pct)) * 1.25) if len(pct) else 1.0)
    ax.set_title(
        "Error concentration vs. distance to nearest GT state-change",
        fontsize=11, color="#0b0b0b", loc="left",
    )
    ax.grid(axis="y", color=_COLOR_GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    ax.spines["bottom"].set_color(_COLOR_STEM)
    ax.spines["left"].set_color(_COLOR_STEM)
    ax.tick_params(axis="both", colors=_COLOR_TEXT_MUTED, length=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_error_anatomy(run_evals: Mapping[tuple[str, str], RunVariantEval], out_dir: Path) -> None:
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    for (run, variant), rv in run_evals.items():
        tables[(run, variant)] = build_eval_window_table(
            list(rv.gt["state"]), rv.cluster, rv.ev.state_mapping, rv.window_s
        )

    pooled = pd.concat(tables.values(), ignore_index=True) if tables else pd.DataFrame(
        columns=["gt_state", "cluster", "pred_state", "boundary_distance_s", "correct", "bucket"]
    )
    pooled_hist = error_anatomy_histogram(pooled)
    plot_error_histogram(pooled_hist, out_dir / "error_vs_boundary.png")

    # Same denominator `error_anatomy_histogram` itself normalizes against (sum over
    # the bucketed rows only) -- see that function's docstring for why this must not
    # be recomputed independently from `pooled` (an "n/a"-bucket window, if any,
    # would then inflate the denominator without ever appearing in a visible bucket).
    total_errors = int(pooled_hist["n_errors"].sum())
    near_boundary_errors = int(
        pooled_hist[pooled_hist["bucket_s"].isin(_NEAR_BOUNDARY_BUCKETS)]["n_errors"].sum()
    )
    near_pct = (near_boundary_errors / total_errors * 100.0) if total_errors else float("nan")

    lines = [
        "# Error anatomy — Step-1 state detection",
        "",
        "For each run x variant (KMeans; majority cluster->state mapping, "
        "`EvalResult.state_mapping`), every EVAL window (`gt.state != \"unknown\"`) "
        "whose majority-mapped prediction differs from GT is bucketed by its "
        "distance to the nearest GT state-change point, found on the FULL per-window "
        "timeline (mirrors `rowii.eval.metrics`'s own boundary-deviation convention).",
        "",
        "## Headline: how much error is near a GT boundary?",
        "",
        f"Pooled across all {len(tables)} analyzed (run, variant) combinations: "
        f"**{near_boundary_errors}/{total_errors} ({near_pct:.1f}%) of all majority-mapped "
        f"errors fall within 15 s of a GT state-change** (buckets 0-5 s + 5-15 s).",
        "",
        _dataframe_to_markdown(
            pooled_hist.assign(
                error_rate=pooled_hist["error_rate"].apply(_fmt_float),
                pct_of_all_errors=pooled_hist["pct_of_all_errors"].apply(
                    lambda v: _fmt_float(v * 100.0, 1) + "%" if not np.isnan(v) else "n/a"
                ),
            )
        ),
        "",
        "![error concentration vs distance to GT boundary](error_vs_boundary.png)",
        "",
        "## Per run x variant",
        "",
    ]
    for (run, variant), wt in tables.items():
        hist = error_anatomy_histogram(wt)
        pr = per_state_precision_recall(wt)
        n_err = int((~wt["correct"]).sum())
        lines += [
            f"### {run} / {variant}",
            "",
            f"{n_err}/{len(wt)} eval windows misclassified (majority mapping).",
            "",
            "Boundary-distance histogram (errors only):",
            "",
            _dataframe_to_markdown(
                hist.assign(
                    error_rate=hist["error_rate"].apply(_fmt_float),
                    pct_of_all_errors=hist["pct_of_all_errors"].apply(
                        lambda v: _fmt_float(v * 100.0, 1) + "%" if not np.isnan(v) else "n/a"
                    ),
                )
            ),
            "",
            "Per-state precision/recall (majority mapping):",
            "",
            _dataframe_to_markdown(
                pr.assign(
                    precision=pr["precision"].apply(_fmt_float),
                    recall=pr["recall"].apply(_fmt_float),
                )
            ),
            "",
        ]

    (out_dir / "error_anatomy.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Deliverable 3: GT-rule sensitivity
# ---------------------------------------------------------------------------


def _baseline_gt_value(rules: GtRules, factor: str) -> float:
    if factor == "power_eps_mw":
        return rules.power_eps_mw
    if factor == "ph_min_dwell_s":
        return rules.ph_min_dwell_s
    if factor == "transition_buffer_s":
        return rules.transition_buffer_s
    raise ValueError(f"unknown sensitivity factor {factor!r}")


def _perturbed_gt_rules(rules: GtRules, factor: str, multiplier: float) -> GtRules:
    if factor == "power_eps_mw":
        return dataclasses.replace(rules, power_eps_mw=rules.power_eps_mw * multiplier)
    if factor == "ph_min_dwell_s":
        return dataclasses.replace(rules, ph_min_dwell_s=rules.ph_min_dwell_s * multiplier)
    if factor == "transition_buffer_s":
        return dataclasses.replace(
            rules, transition_buffer_s=rules.transition_buffer_s * multiplier
        )
    raise ValueError(f"unknown sensitivity factor {factor!r}")


def sensitivity_deltas(
    baseline_state_ari: float,
    baseline_state_accuracy: float,
    observations: Sequence[SensitivityObservation],
) -> pd.DataFrame:
    """One row per perturbation: factor, multiplier, perturbed value, state ARI/accuracy
    under that perturbation, and their delta vs. the (fixed) baseline."""
    rows = []
    for obs in observations:
        rows.append(
            {
                "factor": obs.factor,
                "multiplier": obs.multiplier,
                "perturbed_value": obs.perturbed_value,
                "state_ari": obs.state_ari,
                "delta_state_ari": obs.state_ari - baseline_state_ari,
                "state_accuracy": obs.state_accuracy,
                "delta_state_accuracy": obs.state_accuracy - baseline_state_accuracy,
            }
        )
    return pd.DataFrame(rows)


def sensitivity_verdict(
    deltas: pd.DataFrame, tolerance: float = _SENSITIVITY_ROBUST_TOLERANCE
) -> str:
    """3-sentence-ish verdict: "robust" iff every perturbation's |delta| stays within
    *tolerance* on BOTH state ARI and state accuracy."""
    if len(deltas) == 0:
        return "No perturbations were evaluated; no verdict can be given."
    max_ari = float(deltas["delta_state_ari"].abs().max())
    max_acc = float(deltas["delta_state_accuracy"].abs().max())
    is_robust = max_ari <= tolerance and max_acc <= tolerance
    verdict = "robust" if is_robust else "not robust"
    worst_row = deltas.loc[deltas["delta_state_ari"].abs().idxmax()]
    return (
        f"State-level metrics are **{verdict}** to the {len(deltas)} `GtRules` "
        f"perturbations tested (tolerance {tolerance:.2f} absolute on both state ARI "
        f"and state accuracy): the largest deviation from baseline is "
        f"{max_ari:.4f} in state ARI (factor `{worst_row['factor']}` x"
        f"{worst_row['multiplier']:g}) and {max_acc:.4f} in state accuracy. "
        f"{'No single perturbed rule value moves either headline metric by more than '
         'the tolerance.' if is_robust else 'At least one perturbed rule value moves a '
         'headline metric by more than the tolerance, so the headline numbers should '
         'be read as conditional on the current GtRules defaults.'}"
    )


def write_gt_sensitivity(
    cfg: Config,
    index: RecordingIndex,
    baseline: RunVariantEval,
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    baseline_row = summary[
        (summary["run"] == _SENSITIVITY_RUN)
        & (summary["variant"] == _SENSITIVITY_VARIANT)
        & (summary["clusterer"] == "kmeans")
        & (summary["notes"].fillna("") == "")
    ]
    if len(baseline_row):
        recorded_ari = float(baseline_row.iloc[0]["state_ari"])
        if abs(recorded_ari - baseline.ev.state_ari) > 1e-6:
            logger.warning(
                "recomputed baseline state_ari (%.6f) disagrees with results/summary.csv's "
                "recorded value (%.6f) for %s/%s-kmeans by more than 1e-6 -- the GT-rule "
                "recompute path may not exactly reproduce the original pipeline",
                baseline.ev.state_ari, recorded_ari, _SENSITIVITY_RUN, _SENSITIVITY_VARIANT,
            )

    observations = []
    for factor in _SENSITIVITY_FACTORS:
        for multiplier in _SENSITIVITY_MULTIPLIERS:
            perturbed_rules = _perturbed_gt_rules(cfg.gt, factor, multiplier)
            rv = load_run_variant_eval(
                _SENSITIVITY_RUN, _SENSITIVITY_VARIANT, _RERUN_CLUSTERER, cfg, index,
                gt_rules=perturbed_rules,
            )
            observations.append(
                SensitivityObservation(
                    factor=factor,
                    multiplier=multiplier,
                    perturbed_value=_baseline_gt_value(perturbed_rules, factor),
                    state_ari=rv.ev.state_ari,
                    state_accuracy=rv.ev.state_accuracy,
                )
            )

    deltas = sensitivity_deltas(baseline.ev.state_ari, baseline.ev.state_accuracy, observations)
    verdict = sensitivity_verdict(deltas)

    table = deltas.copy()
    for col in ("state_ari", "delta_state_ari", "state_accuracy", "delta_state_accuracy"):
        table[col] = table[col].apply(_fmt_float)
    table["perturbed_value"] = table["perturbed_value"].apply(lambda v: f"{v:g}")

    lines = [
        "# GT-rule sensitivity — 010726-tu_ph_tu / fusion-kmeans",
        "",
        "The EXISTING fusion-kmeans predictions "
        "(`results/010726-tu_ph_tu/fusion-kmeans/frame_labels.parquet`'s `cluster` "
        "column) are held completely fixed; only the `GtRules` used to recompute GT "
        "(`rowii.scada.labels.gt_labels`) are perturbed, one field at a time, against "
        f"the baseline defaults (power_eps_mw={cfg.gt.power_eps_mw:g}, "
        f"ph_min_dwell_s={cfg.gt.ph_min_dwell_s:g}, "
        f"transition_buffer_s={cfg.gt.transition_buffer_s:g}).",
        "",
        f"Baseline (recomputed via this script; cross-checked against "
        f"`results/summary.csv`): state ARI = {baseline.ev.state_ari:.4f}, "
        f"state accuracy = {baseline.ev.state_accuracy:.4f}.",
        "",
        _dataframe_to_markdown(table),
        "",
        "## Verdict",
        "",
        verdict,
        "",
    ]
    (out_dir / "gt_sensitivity.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Deliverable 4: modality-by-mode synthesis
# ---------------------------------------------------------------------------


def recall_matrix(
    per_variant_tables: Mapping[str, pd.DataFrame], states: Sequence[str] | None = None
) -> pd.DataFrame:
    """Per-state recall (rows = state, cols = variant) from `{variant: window_table}`,
    each `window_table` a `build_eval_window_table` output. A state missing from a
    given variant's table (e.g. zero eval windows) gets NaN recall for that cell."""
    if states is None:
        all_states: set[str] = set()
        for wt in per_variant_tables.values():
            all_states |= set(wt["gt_state"].unique())
        states = sorted(all_states)

    data: dict[str, pd.Series] = {}
    for variant, wt in per_variant_tables.items():
        pr = per_state_precision_recall(wt, states=states).set_index("state")["recall"]
        data[variant] = pr
    return pd.DataFrame(data, index=list(states))


def write_modality_by_mode(
    run_evals: Mapping[tuple[str, str], RunVariantEval], out_dir: Path
) -> None:
    lines = [
        "# Modality-by-mode synthesis",
        "",
        "Per-state recall (majority-mapped family, `EvalResult.state_mapping`) for "
        "each variant's KMeans run, restricted to that run's own eval windows "
        "(`gt.state != \"unknown\"`). Rows = GT state, columns = input variant.",
        "",
        "**Mapping-view caveat.** This matrix uses the PRIMARY (majority "
        "cluster->state) mapping family: a cluster is named after whichever GT state "
        "holds the majority of its own eval windows, so a minority state that is "
        "perfectly CONCENTRATED into one cluster still scores recall 0.000 here "
        "whenever that cluster's majority belongs to a more numerous state. Earlier "
        "prose reports describe exactly such cases through the other (strict, 1:1 "
        "Hungarian) lens -- e.g. the README's multi-day section says "
        "010726-tu_ph_tu's vibration variant catches \"all 117 standstill windows\" "
        "in its quiet cluster, and that run's own `report.md` Hungarian mapping "
        "indeed names that cluster standstill (its column captures 117/117) -- "
        "while this matrix shows vibration standstill recall 0.000 for the same "
        "run, because the same cluster's majority vote resolves to transition (142 "
        "transition windows outvote the 117 standstill ones inside that cluster). "
        "Both views are correct and computed from the same clustering; they answer "
        "different questions (majority: \"what would a mode-labeling system "
        "output?\"; Hungarian/cluster-level: \"is the state separable at all?\"). "
        "When a cell below reads 0.000, consult the run's `report.md` confusion "
        "matrices before concluding the state is not separated.",
        "",
    ]
    for run in _MODALITY_BY_MODE_RUNS:
        tables = {}
        for variant in _MODALITY_BY_MODE_VARIANTS:
            rv = run_evals.get((run, variant))
            if rv is None:
                continue
            tables[variant] = build_eval_window_table(
                list(rv.gt["state"]), rv.cluster, rv.ev.state_mapping, rv.window_s
            )
        if not tables:
            lines += [f"## {run}", "", "No kmeans artifacts available for this run.", ""]
            continue
        matrix = recall_matrix(tables, states=None)
        formatted = matrix.apply(lambda col: col.map(_fmt_float))
        lines += [
            f"## {run}",
            "",
            _dataframe_to_markdown(formatted, index_label="state"),
            "",
        ]

    (out_dir / "modality_by_mode.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Deliverable 5: cross-check vs. partner analysis
# ---------------------------------------------------------------------------


def compare_duration_s(label: str, ours_s: float, theirs_s: float) -> DurationComparison:
    delta_s = ours_s - theirs_s
    delta_pct = (delta_s / theirs_s * 100.0) if theirs_s else float("nan")
    return DurationComparison(
        label=label, ours_s=ours_s, theirs_s=theirs_s, delta_s=delta_s, delta_pct=delta_pct
    )


def _majority_mapped_segments(
    segments: pd.DataFrame, state_mapping: Mapping[int, str]
) -> pd.DataFrame:
    """`segments.csv` (cluster, start_utc, end_utc, duration_s) with a `state` column
    added via the majority cluster->state mapping (`-1`/unmapped clusters -> "unknown")."""
    out = segments.copy()
    out["state"] = out["cluster"].apply(lambda c: state_mapping.get(int(c), "unknown"))
    return out


def _longest_run_of_state(segments_with_state: pd.DataFrame, state: str) -> float | None:
    """Longest single contiguous segment's duration (seconds) matching *state*, or
    None if it never occurs. `to_segments`'s output already has one row per maximal
    run of a single cluster id, but two ADJACENT segments can share the same
    majority-mapped state (different clusters, same state) -- this coalesces
    consecutive same-state rows before taking the max, so a state that is briefly
    split across a cluster boundary is still counted as one contiguous hold.

    NOT gap-tolerant: any non-*state* sliver -- including the 1-second `-1`/unknown
    invalid-window segments the pipeline itself produces at 12-min burst-file
    boundaries (a window straddling two burst files has a full-window sample run in
    NEITHER file, so its features are NaN and the validity mask excludes it; see
    `src/rowii/pipeline.py`'s `_extract_stream_features` / `compute_validity_mask`) -- resets
    the accumulation. For hold-duration comparisons against wall-clock session facts
    use `state_holds` instead; this function is kept as the diagnostic that MAKES the
    fragmentation visible (its value vs. `state_holds`' envelope quantifies it).
    """
    states = segments_with_state["state"].to_numpy()
    durations = segments_with_state["duration_s"].to_numpy()
    if state not in states:
        return None
    best = 0.0
    current = 0.0
    for s, d in zip(states, durations, strict=True):
        if s == state:
            current += float(d)
            best = max(best, current)
        else:
            current = 0.0
    return best


@dataclass(frozen=True)
class StateHold:
    """One gap-tolerant "hold" of a single state: a maximal group of same-state
    segments in which consecutive same-state segments are separated by at most
    `gap_tolerance_s` of other-state/unknown time (see `state_holds`)."""

    start_utc: str
    """First same-state segment's `start_utc` (as read from segments.csv)."""
    end_utc: str
    """Last same-state segment's `end_utc`."""
    envelope_s: float
    """Wall-clock span from the hold's first segment start to its last segment end
    (INCLUDES the bridged sub-tolerance gaps)."""
    summed_s: float
    """Sum of the same-state segments' own durations (EXCLUDES the bridged gaps)."""
    n_fragments: int
    """How many separate same-state segments the hold was fragmented into."""


def state_holds(
    segments_with_state: pd.DataFrame, state: str, gap_tolerance_s: float
) -> list[StateHold]:
    """Gap-tolerant holds of *state* in a `_majority_mapped_segments` output.

    Motivation (2026-07-09 crosscheck fix): `to_segments` output fragments a
    physically continuous operating phase wherever the validity mask dropped a
    window -- on real data this happens like clockwork at 12-min burst-file
    boundaries, where the boundary-straddling window has a full-window sample run
    in neither file, gets NaN features, and is excluded (cluster `-1` ->
    majority-mapped "unknown"). A wall-clock hold duration comparable to an
    operator-log fact ("~37-min PH hold") must therefore bridge sub-tolerance
    interruptions instead of resetting at each 1-second sliver.

    Two same-state segments belong to the same hold iff the wall-clock gap between
    them (`next.start_utc - prev.end_utc`) is <= *gap_tolerance_s*. Each hold
    reports both its envelope (first start -> last end, gaps included) and its
    summed same-state duration (gaps excluded) -- see `StateHold`.

    *segments_with_state* needs `state`, `start_utc`, `end_utc`, `duration_s`
    columns and must be time-ordered (as `to_segments` guarantees). Timestamps may
    be strings (raw segments.csv) or datetimes -- both are parsed via
    `pd.to_datetime`. Returns [] when *state* never occurs.
    """
    matching = segments_with_state[segments_with_state["state"] == state]
    if len(matching) == 0:
        return []

    starts = pd.to_datetime(matching["start_utc"])
    ends = pd.to_datetime(matching["end_utc"])
    durations = matching["duration_s"].to_numpy(dtype=np.float64)

    holds: list[StateHold] = []
    group_start_i = 0
    for i in range(1, len(matching) + 1):
        is_last = i == len(matching)
        gap_s = (
            float((starts.iloc[i] - ends.iloc[i - 1]).total_seconds()) if not is_last else None
        )
        if is_last or (gap_s is not None and gap_s > gap_tolerance_s):
            start_ts = starts.iloc[group_start_i]
            end_ts = ends.iloc[i - 1]
            holds.append(
                StateHold(
                    start_utc=str(start_ts),
                    end_utc=str(end_ts),
                    envelope_s=float((end_ts - start_ts).total_seconds()),
                    summed_s=float(durations[group_start_i:i].sum()),
                    n_fragments=i - group_start_i,
                )
            )
            group_start_i = i
    return holds


_HOLD_GAP_TOLERANCE_S = 60.0
"""Gap tolerance for `state_holds` in the crosscheck: generously above the observed
1-second invalid-window slivers at 12-min burst-file boundaries, far below any
genuine mode dwell (`GtRules.ph_min_dwell_s` = 600 s; the gap between the
290626-tu pre-generation PH hold and the post-generation PH spin-down is >3 h)."""


def write_crosscheck(
    run_evals: Mapping[tuple[str, str], RunVariantEval], cfg: Config, out_dir: Path
) -> None:
    lines = [
        "# Cross-check vs. partner analysis",
        "",
        "**Partner analysis used as independent cross-reference only; no values "
        "adopted into our pipeline or results.** Every number under \"partner\" below "
        "is copied verbatim from Bruno's read-only aggregated-analysis material and "
        "cited by file + section; every number under \"ours\" is computed fresh from "
        "this repository's own `results/*/fusion-kmeans/segments.csv` "
        "(majority cluster->state mapping).",
        "",
    ]

    rv_290626 = run_evals.get(("290626-tu", "fusion"))
    if rv_290626 is not None:
        seg_path = cfg.results_root / "290626-tu" / "fusion-kmeans" / "segments.csv"
        segments = pd.read_csv(seg_path)
        segments = _majority_mapped_segments(segments, rv_290626.ev.state_mapping)
        holds = state_holds(segments, "phase-shifter", _HOLD_GAP_TOLERANCE_S)
        lines += ["## PH-hold duration (290626-tu)", ""]
        if not holds:
            lines += ["No phase-shifter segment found in our detection for this run.", ""]
        else:
            hold = max(holds, key=lambda h: h.summed_s)
            # holds is non-empty, so the state occurs and this can never be None --
            # the fallback only satisfies the Optional type.
            naive_s = _longest_run_of_state(segments, "phase-shifter") or 0.0
            envelope_cmp = compare_duration_s(
                "290626-tu PH-hold envelope", hold.envelope_s, _PARTNER_PH_HOLD_MINUTES * 60.0
            )
            summed_cmp = compare_duration_s(
                "290626-tu PH-hold summed", hold.summed_s, _PARTNER_PH_HOLD_MINUTES * 60.0
            )
            lines += [
                "Ours (from `results/290626-tu/fusion-kmeans/segments.csv`, "
                "majority-mapped, longest gap-tolerant phase-shifter hold with "
                f"{_HOLD_GAP_TOLERANCE_S:.0f} s gap tolerance):",
                "",
                f"- **Envelope** (first PH segment start -> last PH segment end): "
                f"{envelope_cmp.ours_s:.1f} s ({envelope_cmp.ours_s / 60.0:.2f} min).",
                f"- **Summed PH time within that envelope**: {summed_cmp.ours_s:.1f} s "
                f"({summed_cmp.ours_s / 60.0:.2f} min), across {hold.n_fragments} "
                "segment fragments.",
                "",
                f"Partner: ~{_PARTNER_PH_HOLD_MINUTES:.0f} min "
                f"({envelope_cmp.theirs_s:.0f} s), {_PARTNER_SOURCE} "
                '("37-min turbine-direction PH hold").',
                "",
                f"Delta (envelope): {envelope_cmp.delta_s:+.1f} s "
                f"({envelope_cmp.delta_pct:+.1f}%). Delta (summed): "
                f"{summed_cmp.delta_s:+.1f} s ({summed_cmp.delta_pct:+.1f}%).",
                "",
                "### Why gap-tolerant statistics (fragmentation caveat)",
                "",
                "The naive \"longest contiguous majority-PH segment\" statistic reads "
                f"only {naive_s:.0f} s ({naive_s / 60.0:.2f} min) for the same "
                "hold and is NOT comparable to an operator-log wall-clock duration: "
                "the hold is fragmented into "
                f"{hold.n_fragments} pieces by 1-second `unknown` slivers at exactly "
                "the 12-min burst-file boundaries. These slivers are windows the "
                "pipeline's own validity mask drops BEFORE clustering (a window "
                "straddling two burst files has a full-window sample run in neither "
                "file, so its features are NaN and it is excluded -- cluster id `-1`, "
                "majority-mapped \"unknown\"; see "
                "`src/rowii/pipeline.py::_extract_stream_features` / "
                "`compute_validity_mask`), not detector state flicker. The envelope "
                "and summed statistics above bridge interruptions up to "
                f"{_HOLD_GAP_TOLERANCE_S:.0f} s (generously above the observed 1-s "
                "slivers, far below any genuine mode dwell) and are the comparable "
                "quantities.",
                "",
            ]

        sequence = _coalesced_state_sequence(
            _drop_brief_unknown_segments(segments, _HOLD_GAP_TOLERANCE_S)
        )
        lines += [
            "## Session structure (qualitative)",
            "",
            "Partner (same section): \"TU session: 22 groups, converter-assisted "
            "turbine start (new class) via 37-min turbine-direction PH hold, load "
            "steps 91-292 MW, braked shutdown\" -- i.e. PH-hold FIRST, then loaded "
            "generation, ending in a braked shutdown. Our own fusion-kmeans "
            "majority-mapped segment sequence for 290626-tu (sub-"
            f"{_HOLD_GAP_TOLERANCE_S:.0f}-s unknown slivers dropped -- the "
            "invalid-window fragmentation artifact explained above): "
            + " -> ".join(sequence)
            + ".",
            "",
            "## \"22 groups\" (partner) — not a comparable unit",
            "",
            "Partner's \"22 groups\" (same section) counts the partner's own "
            "pre-processing measurement/burst groups for the 29.06 TU session, a "
            "different unit than our unsupervised state-segment count. Our "
            f"fusion-kmeans run produces {len(segments)} raw segments "
            f"({len(sequence)} after dropping sub-{_HOLD_GAP_TOLERANCE_S:.0f}-s "
            "unknown slivers and coalescing adjacent same-state segments) -- "
            "reported here for transparency, not as a like-for-like delta against "
            "the partner's 22.",
            "",
        ]

    rv_010726 = run_evals.get(("010726-tu_ph_tu", "fusion"))
    if rv_010726 is not None:
        n_ph = int((rv_010726.gt["state"] == "phase-shifter").sum())
        lines += [
            "## 010726-tu_ph_tu phase-shifter interval — indirectly corroborated",
            "",
            "aggregated-report-v2.md has no day-specific session facts for 01.07 "
            "(its per-day summary section only covers 29.06), so no direct duration "
            "cross-check is possible for this run. Section 3 (\"PH direction x day\") "
            "does independently confirm a turbine-direction phase-shifter interval "
            "exists in the July-1 delivery (used there to compute "
            "`vib_band_50_200 = 2.491e-3` for \"turbdir_jul01\") -- this corroborates, "
            "without giving a duration to diff against, our own detection of a "
            f"{n_ph}-window ({n_ph * rv_010726.window_s / 60.0:.1f} min) "
            "GT phase-shifter interval in 010726-tu_ph_tu.",
            "",
        ]

    (out_dir / "crosscheck_partner.md").write_text("\n".join(lines))


def _coalesced_state_sequence(segments_with_state: pd.DataFrame) -> list[str]:
    """Majority-mapped state sequence with consecutive duplicates collapsed (e.g.
    `["turbine", "turbine", "phase-shifter"]` -> `["turbine", "phase-shifter"]`)."""
    sequence: list[str] = []
    for state in segments_with_state["state"]:
        if not sequence or sequence[-1] != state:
            sequence.append(state)
    return sequence


def _drop_brief_unknown_segments(
    segments_with_state: pd.DataFrame, max_duration_s: float
) -> pd.DataFrame:
    """Rows of *segments_with_state* minus `"unknown"` segments shorter than or equal
    to *max_duration_s* -- the 1-second invalid-window slivers at burst-file
    boundaries (see `state_holds`' docstring), which are a pipeline masking artifact,
    not a detected state change. LONG unknown stretches (genuine no-coverage regions)
    are kept, so a real gap still shows up in a qualitative sequence."""
    keep = ~(
        (segments_with_state["state"] == "unknown")
        & (segments_with_state["duration_s"] <= max_duration_s)
    )
    return segments_with_state[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Real-data loading (impure: file I/O, real ROWII_DATA_ROOT required)
# ---------------------------------------------------------------------------


def _run_step1_subprocess(run: str, variant: str, clusterer: str) -> None:
    script = _SCRIPTS_DIR / "run_step1.py"
    cmd = [
        sys.executable, str(script),
        "--run", run, "--variant", variant, "--clusterer", clusterer,
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=_SCRIPTS_DIR.parent)


def rerun_missing_variants(cfg: Config, *, skip: bool) -> None:
    """Idempotently regenerate `250526-tu`'s audio/vibration/fusion-kmeans artifacts
    if missing (they were deleted between the June-25 and multi-day grids -- see
    module docstring). A no-op per-variant once its `report.md` exists, so
    re-invoking this (even without `--skip-reruns`) after the first successful run
    never re-appends duplicate `summary.csv` rows."""
    if skip:
        logger.info("--skip-reruns: not regenerating any missing %s artifacts", _RERUN_RUN)
        return
    for variant in _RERUN_VARIANTS:
        out_dir = cfg.results_root / _RERUN_RUN / f"{variant}-{_RERUN_CLUSTERER}"
        if (out_dir / "report.md").exists():
            logger.info("%s already exists, skipping rerun", out_dir)
            continue
        logger.info("regenerating %s (missing) ...", out_dir)
        _run_step1_subprocess(_RERUN_RUN, variant, _RERUN_CLUSTERER)


def _find_run(index: RecordingIndex, name: str) -> Run:
    matches = [r for r in index.runs if r.name == name]
    if not matches:
        raise KeyError(
            f"run {name!r} not found in discovered index "
            f"(available: {sorted(r.name for r in index.runs)})"
        )
    return matches[0]


def load_run_variant_eval(
    run_name: str,
    variant: str,
    clusterer: str,
    cfg: Config,
    index: RecordingIndex,
    *,
    gt_rules: GtRules | None = None,
) -> RunVariantEval:
    """Recompute (grid, GT) for (*run_name*, *variant*) via the existing pipeline
    (`run_step1.build_run_grid`/`load_run_gt`, `rowii.scada.labels.gt_labels`) and
    pair it with the already-written `frame_labels.parquet`'s `cluster` column,
    yielding a fresh `EvalResult` (majority + strict) without re-extracting any
    audio/vibration feature.

    *gt_rules*, when given, overrides `cfg.gt` for this call only (GT-rule
    sensitivity, deliverable 3) -- everything else about the recompute (grid,
    validity mask, predictions) is identical to the baseline call.
    """
    run = _find_run(index, run_name)
    streams = run_step1._streams_for_variant(variant)
    grid = run_step1.build_run_grid(run, streams, cfg.window.window_s)
    betriebsdaten = index.betriebsdaten_by_day.get(run.day_root, [])

    combo_dir = cfg.results_root / run_name / f"{variant}-{clusterer}"
    frame_labels_path = combo_dir / "frame_labels.parquet"
    if not frame_labels_path.exists():
        raise FileNotFoundError(
            f"{frame_labels_path} not found -- run scripts/run_step1.py --run {run_name} "
            f"--variant {variant} --clusterer {clusterer} first"
        )
    frame_labels = pd.read_parquet(frame_labels_path, engine="pyarrow")
    cluster = frame_labels["cluster"].to_numpy()
    if len(cluster) != grid.n_windows:
        raise ValueError(
            f"{frame_labels_path}: {len(cluster)} windows, recomputed grid has "
            f"{grid.n_windows} -- pipeline/grid mismatch, refusing to recompute GT"
        )
    valid_mask = cluster != -1

    cfg_used = cfg if gt_rules is None else dataclasses.replace(cfg, gt=gt_rules)
    _scada, gt = run_step1.load_run_gt(run, betriebsdaten, grid, cfg_used, valid_mask)

    ev = evaluate(cluster, gt, grid)
    return RunVariantEval(run=run_name, variant=variant, grid=grid, gt=gt, cluster=cluster, ev=ev)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the Step-1 results grid (results/summary.csv + per-run "
            "artifacts) into results/analysis/: overview, error anatomy, GT-rule "
            "sensitivity, modality-by-mode recall, and a partner cross-check."
        )
    )
    parser.add_argument(
        "--skip-reruns",
        action="store_true",
        help=(
            "Do not regenerate missing 250526-tu kmeans artifacts (audio/vibration/"
            "fusion) even if absent; the modality-by-mode deliverable will then omit "
            "250526-tu instead of computing it."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config()
    analysis_dir = cfg.results_root / _ANALYSIS_DIR_NAME
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rerun_missing_variants(cfg, skip=args.skip_reruns)

    summary = pd.read_csv(cfg.results_root / "summary.csv")
    write_overview(summary, analysis_dir)

    index = discover(cfg.data_root)

    run_evals: dict[tuple[str, str], RunVariantEval] = {}
    for run_name, variant in _ERROR_ANATOMY_COMBOS:
        try:
            run_evals[(run_name, variant)] = load_run_variant_eval(
                run_name, variant, _RERUN_CLUSTERER, cfg, index
            )
        except FileNotFoundError as exc:
            logger.warning("skipping %s/%s: %s", run_name, variant, exc)

    write_error_anatomy(run_evals, analysis_dir)

    baseline = run_evals.get((_SENSITIVITY_RUN, _SENSITIVITY_VARIANT))
    if baseline is not None:
        write_gt_sensitivity(cfg, index, baseline, summary, analysis_dir)
    else:
        logger.warning(
            "skipping GT sensitivity: %s/%s not available", _SENSITIVITY_RUN, _SENSITIVITY_VARIANT
        )

    write_modality_by_mode(run_evals, analysis_dir)
    write_crosscheck(run_evals, cfg, analysis_dir)

    print(f"analyze_step1: wrote analysis artifacts to {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
