"""ARI / Hungarian-matched macro-F1 / boundary-deviation evaluation against SCADA GT.

`evaluate` compares a detector's per-window cluster labels (anonymous integer ids,
`rowii.state.detect.DetectionResult.frame_labels`) against `rowii.scada.labels.gt_labels`
output. Windows with `gt.state == "unknown"` (no SCADA coverage, or a rule genuinely
could not decide) are dropped BEFORE every metric except the boundary deviation, which
by construction runs on the full per-window timeline (see `_state_change_indices`).

Two families of metrics, both computed by every `evaluate()` call:

- **State-level (mode) metrics -- primary** (`state_mapping`, `state_accuracy`,
  `state_macro_f1`, `state_ari`, `state_confusion`): each predicted cluster maps
  INDEPENDENTLY onto whichever GT state is the majority vote among its own eval
  windows (no 1:1 constraint). This is the metric family that matches the thesis
  design (spec §5: "load sub-structure appears as extra clusters ... or reported as
  sub-clusters") -- a machine operating mode (standstill/turbine/pump) can
  legitimately contain several unsupervised load-level sub-clusters, and a detector
  that cleanly finds such sub-clusters should NOT be penalized as if it had confused
  two different modes.
- **Strict metrics -- secondary** (`ari`, `macro_f1`, `mapping`, `confusion`): a
  strict 1:1 Hungarian correspondence between clusters and GT states (see `_hungarian_mapping`
  below). Kept unchanged for continuity with Task 13's baseline numbers and as a
  diagnostic for genuine over-segmentation (a large gap between `state_macro_f1` and
  `macro_f1` signals exactly the "extra clusters are sub-modes, not confusion"
  pattern this module now also measures directly).

Cluster ids are an arbitrary permutation with no inherent correspondence to GT state
names (KMeans/GMM label ids are assigned by the clustering algorithm, not by this
module) -- `_hungarian_mapping` recovers the best 1:1 correspondence via the Hungarian
algorithm (`scipy.optimize.linear_sum_assignment`) on the GT-state x predicted-cluster
contingency table, maximizing the total number of matched windows. When there are more
predicted clusters than GT states (k > #states, e.g. load sub-structure appearing as
extra clusters -- spec §5), every cluster beyond the 1:1 assignment is left over; each
such leftover cluster maps independently onto whichever GT state its own column in the
contingency table maximizes (not restricted to already-matched states), so a k-sweep
run never leaves a predicted cluster unmapped. `_majority_mapping` (state-level) uses
this same "own column argmax" rule for EVERY cluster, not just the 1:1 leftovers --
i.e. it is what `_hungarian_mapping`'s fallback branch already does, applied
universally.

`load_alignment` (Task 13b item 2) answers a narrower, orthogonal question: within the
single dominant operating mode (turbine, or pump as a fallback -- see its own
docstring), do the detector's sub-clusters track SCADA-derived load LEVEL rather than
just "this is turbine"? This is independent of `evaluate`'s mode-vs-mode metrics above
(which never look at `gt.load_bin` at all) and is scoped to whichever run actually has
turbine/pump windows to analyze.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score

from rowii.signals.windows import WindowGrid

_UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvalResult:
    """Metrics comparing detected states against SCADA ground truth on one run."""

    ari: float
    """Adjusted Rand Index, computed on eval windows (gt.state != "unknown") only."""
    macro_f1: float
    """Macro-averaged F1 of Hungarian-mapped predicted states vs GT states."""
    confusion: pd.DataFrame
    """Cross-tabulation: rows = GT states, columns named via the strict 1:1 Hungarian
    mapping (`mapping`) -- the secondary, strict counterpart to `state_confusion`."""
    boundary_median_abs_s: float | None
    """Median |Δt| (seconds) from each GT state change to the nearest predicted state
    change, both found on the FULL per-window timeline (not eval-windows-only).
    `None` when either side has zero countable state changes."""
    n_eval_windows: int
    """Count of windows with gt.state != "unknown" -- the population every metric
    except `boundary_median_abs_s` is computed over."""
    mapping: dict[int, str]
    """Predicted cluster id -> GT state name, from Hungarian matching (see module
    docstring for the k > #states extra-cluster fallback)."""
    state_mapping: dict[int, str]
    """Predicted cluster id -> GT state name, from an INDEPENDENT per-cluster majority
    vote (no 1:1 constraint) -- the primary, mode-level view (see module docstring)."""
    state_accuracy: float
    """Fraction of eval windows whose `state_mapping`-mapped prediction equals the
    window's own GT state -- the primary, mode-level accuracy."""
    state_macro_f1: float
    """Macro-averaged F1 of `state_mapping`-mapped predictions vs GT states -- the
    primary, mode-level counterpart to `macro_f1`."""
    state_ari: float
    """ARI between the GT state sequence and the `state_mapping`-collapsed predicted
    sequence (cluster ids replaced by their mapped state NAME before scoring, unlike
    `ari`, which compares raw cluster ids) -- rewards a detector for finding pure
    sub-clusters of the correct state instead of penalizing the extra cluster count."""
    state_confusion: pd.DataFrame
    """Cross-tabulation: rows = GT states, columns named via the majority-vote mapping
    (`state_mapping`) -- the primary, mode-level counterpart to `confusion`. A raw
    cluster id can legitimately carry a DIFFERENT column name here than in `confusion`
    (majority vote vs. the globally-optimized 1:1 Hungarian assignment can disagree on
    the same cluster, see module docstring) -- callers must not conflate the two."""


def _hungarian_mapping(gt_states: pd.Series, pred: np.ndarray) -> dict[int, str]:
    """Cluster id -> GT state name, one entry per cluster id present in *pred*.

    *gt_states* and *pred* must already be restricted to eval windows (this function
    assumes every id in *pred* has a genuine, non-"unknown" GT counterpart to be
    matched against).
    """
    state_names = sorted(gt_states.unique())
    cluster_ids = sorted({int(c) for c in np.unique(pred)})

    contingency = pd.DataFrame(0, index=state_names, columns=cluster_ids, dtype=np.int64)
    for state, cluster in zip(gt_states, pred, strict=True):
        contingency.loc[state, int(cluster)] += 1

    # linear_sum_assignment minimizes cost; negate to MAXIMIZE matched-window count.
    # Rectangular support (more clusters than states, or vice versa) yields exactly
    # min(n_states, n_clusters) pairs -- scipy handles this natively, no padding needed.
    row_idx, col_idx = linear_sum_assignment(-contingency.to_numpy())

    mapping: dict[int, str] = {}
    for r, c in zip(row_idx, col_idx, strict=True):
        mapping[cluster_ids[c]] = state_names[r]

    # Clusters left unmatched by the 1:1 Hungarian assignment (k > #states) each fall
    # back independently to whichever GT state their own column maximizes.
    for cluster_id in cluster_ids:
        if cluster_id not in mapping:
            column = contingency[cluster_id]
            mapping[cluster_id] = str(column.idxmax())

    return mapping


def _majority_mapping(gt_states: pd.Series, pred: np.ndarray) -> dict[int, str]:
    """Cluster id -> GT state name, each cluster mapped INDEPENDENTLY to whichever GT
    state is the majority among its own eval windows (no 1:1 constraint across
    clusters -- see module docstring). Every cluster id present in *pred* gets an
    entry; ties fall to `pandas.Series.idxmax`'s first-occurrence-in-index order
    (deterministic given `gt_states.unique()`'s stable ordering), matching
    `_hungarian_mapping`'s own extra-cluster fallback tie-break exactly.
    """
    state_names = sorted(gt_states.unique())
    cluster_ids = sorted({int(c) for c in np.unique(pred)})

    contingency = pd.DataFrame(0, index=state_names, columns=cluster_ids, dtype=np.int64)
    for state, cluster in zip(gt_states, pred, strict=True):
        contingency.loc[state, int(cluster)] += 1

    return {cluster_id: str(contingency[cluster_id].idxmax()) for cluster_id in cluster_ids}


def _state_change_indices(states: list[str]) -> list[int]:
    """Window indices *i* (>= 1) where `states[i] != states[i - 1]`, excluding any
    change touching `"unknown"` on either side (neither a genuine GT transition nor a
    meaningful predicted one -- see module docstring)."""
    return [
        i
        for i in range(1, len(states))
        if states[i] != states[i - 1] and _UNKNOWN not in (states[i], states[i - 1])
    ]


def _boundary_median_abs_s(
    gt_state_col: pd.Series, mapped_full: np.ndarray, grid: WindowGrid
) -> float | None:
    window_s = grid.window_ns / 1e9
    gt_changes = _state_change_indices(list(gt_state_col))
    pred_changes = _state_change_indices(list(mapped_full))

    if not gt_changes or not pred_changes:
        return None

    pred_changes_arr = np.asarray(pred_changes, dtype=np.float64)
    deviations = [
        float(np.min(np.abs(pred_changes_arr - gt_i))) * window_s for gt_i in gt_changes
    ]
    return float(np.median(deviations))


def evaluate(pred: np.ndarray, gt: pd.DataFrame, grid: WindowGrid) -> EvalResult:
    """Compare *pred* (per-window cluster ids) against *gt* (`gt_labels` output).

    Args:
        pred: Per-window predicted cluster ids, shape (grid.n_windows,).
        gt: Ground-truth DataFrame (`rowii.scada.labels.gt_labels` output), indexed
            0..grid.n_windows - 1, with a `state` column (including possibly
            `"unknown"`).
        grid: The `WindowGrid` both *pred* and *gt* were computed against; used only
            to convert the boundary metric's window-index deviations into seconds.

    Returns:
        An `EvalResult` (see field docs for exactly which windows/timeline each metric
        is computed over).

    Raises:
        ValueError: When no windows have a known ground-truth state.
    """
    eval_mask = gt["state"].to_numpy() != _UNKNOWN
    n_eval = int(eval_mask.sum())
    if n_eval == 0:
        raise ValueError(
            "no windows with a known ground-truth state; cannot evaluate "
            "(all gt.state == 'unknown' — check SCADA coverage for this run)"
        )

    gt_eval_states = gt.loc[eval_mask, "state"]
    pred_eval = pred[eval_mask]
    state_names = sorted(gt_eval_states.unique())

    mapping = _hungarian_mapping(gt_eval_states, pred_eval)

    ari = float(adjusted_rand_score(gt_eval_states.to_numpy(), pred_eval))

    mapped_eval = np.array([mapping[int(c)] for c in pred_eval], dtype=object)
    macro_f1 = float(
        f1_score(
            gt_eval_states.to_numpy(),
            mapped_eval,
            average="macro",
            labels=state_names,
            zero_division=0,
        )
    )

    confusion = pd.crosstab(
        pd.Series(gt_eval_states.to_numpy(), name="gt"),
        pd.Series(mapped_eval, name="predicted"),
    )

    # State-level (mode) metrics -- primary view (see module docstring).
    state_mapping = _majority_mapping(gt_eval_states, pred_eval)
    state_mapped_eval = np.array([state_mapping[int(c)] for c in pred_eval], dtype=object)
    state_accuracy = float(accuracy_score(gt_eval_states.to_numpy(), state_mapped_eval))
    state_macro_f1 = float(
        f1_score(
            gt_eval_states.to_numpy(),
            state_mapped_eval,
            average="macro",
            labels=state_names,
            zero_division=0,
        )
    )
    state_ari = float(adjusted_rand_score(gt_eval_states.to_numpy(), state_mapped_eval))

    state_confusion = pd.crosstab(
        pd.Series(gt_eval_states.to_numpy(), name="gt"),
        pd.Series(state_mapped_eval, name="predicted"),
    )

    # Boundary metric runs on the FULL timeline (all windows, including the ones
    # dropped above for every other metric) -- a cluster id outside `mapping` can only
    # occur here if it has zero eval-window presence at all (fully confined to
    # "unknown" zones); such an id maps to a placeholder string that can never equal a
    # real GT state name, so it can still register as a "predicted change" without
    # ever spuriously matching a GT state by coincidence.
    mapped_full = np.array(
        [mapping.get(int(c), "__unmapped__") for c in pred], dtype=object
    )
    boundary_median_abs_s = _boundary_median_abs_s(gt["state"], mapped_full, grid)

    return EvalResult(
        ari=ari,
        macro_f1=macro_f1,
        confusion=confusion,
        boundary_median_abs_s=boundary_median_abs_s,
        n_eval_windows=n_eval,
        mapping=mapping,
        state_mapping=state_mapping,
        state_accuracy=state_accuracy,
        state_macro_f1=state_macro_f1,
        state_ari=state_ari,
        state_confusion=state_confusion,
    )


_LOAD_ALIGNMENT_STATE = "turbine"
_LOAD_ALIGNMENT_FALLBACK_STATE = "pump"
_MIN_DISTINCT_LOAD_BINS = 2


def load_alignment(pred: np.ndarray, gt: pd.DataFrame) -> pd.DataFrame | None:
    """Do sub-clusters within one operating mode track SCADA load LEVEL?

    Restricts to eval windows (`gt.state != "unknown"`) whose GT state is
    `"turbine"` -- falling back to `"pump"` if this run has zero turbine windows
    (e.g. a pump-only recording) -- and cross-tabulates predicted cluster id against
    `gt.load_bin` on exactly that subset. A high alignment here means the detector's
    extra sub-clusters (the ones that make `state_ari`/`macro_f1` diverge, see the
    module docstring) are not noise: they correspond to genuine load-level
    structure the unsupervised detector recovered without ever seeing SCADA.

    Args:
        pred: Per-window predicted cluster ids, shape matching `gt`'s row count.
        gt: Ground-truth DataFrame (`rowii.scada.labels.gt_labels` output) with
            `state` and `load_bin` columns.

    Returns:
        A cluster-id x load-bin crosstab DataFrame with the Adjusted Rand Index
        between `load_bin` and cluster id (on this subset only) attached as
        `df.attrs["ari"]`. `None` when the subset has fewer than 2 distinct load
        bins to align against (an ARI would be undefined/degenerate), including
        when the run has no turbine OR pump windows at all.
    """
    eval_mask = gt["state"].to_numpy() != _UNKNOWN
    state_mask = (gt["state"].to_numpy() == _LOAD_ALIGNMENT_STATE) & eval_mask
    if not state_mask.any():
        state_mask = (gt["state"].to_numpy() == _LOAD_ALIGNMENT_FALLBACK_STATE) & eval_mask
    if not state_mask.any():
        return None

    load_bin = gt.loc[state_mask, "load_bin"].to_numpy()
    pred_subset = pred[state_mask]

    if len(np.unique(load_bin)) < _MIN_DISTINCT_LOAD_BINS:
        return None

    ari = float(adjusted_rand_score(load_bin, pred_subset))

    crosstab = pd.crosstab(
        pd.Series(pred_subset, name="cluster"),
        pd.Series(load_bin, name="load_bin"),
    )
    crosstab.attrs["ari"] = ari
    return crosstab
