"""Multi-run/multi-mode training pools with per-run leakage-safe sides, plus the
A4.1/A4.2 label-coverage tables -- Step-2 package-7 robustness (design spec
`docs/superpowers/specs/2026-07-18-step2-package7-robustness-design.md` D1 +
amendments A3.7 and A4.1/A4.2, plan `docs/superpowers/plans/
2026-07-18-step2-package7-robustness.md` Task 1).

`build_pool` collects ONE side's windows from EVERY run of a pool, each run split
independently with the SAME `split_by_segments` convention `run_sweep` uses
(`rowii.anomaly.sweep`, module docstring points 1-2):

1. **Top split** -- `split_by_segments(segment_ids, valid_mask,
   sweep_cfg.calibration_frac, sweep_cfg.seed)` -> CALIBRATION side / SCORING side.
2. **Nested split of the calibration side only** -- `split_by_segments(segment_ids,
   calibration_mask, 0.5, sweep_cfg.seed + 1)` -> FIT part / CONFORMAL part.

`side="calibration"` pools each run's whole top calibration side; `side="fit"` /
`side="conformal"` pool the nested parts. Pooled references are built on the pooled
FIT part and pooled FROZEN thresholds calibrate on the pooled CONFORMAL part's scores
(spec A3.7) -- the same three-way footing `run_sweep` gives a single run. The
cross-run leakage rule (spec D1) holds by construction: every run's SCORING side is
untouched by every pooled side, so a pooled artifact evaluated on run R's scoring
windows was never fit or calibrated on them -- defensively re-asserted per run here,
the same trust-but-verify posture as `run_sweep._assert_three_way_disjoint`. (The
stronger A3.1 rule -- pool artifacts are never EVALUATED on pool-member runs at all
-- is a protocol-level concern enforced by `scripts/run_step2.py`'s
cross-day-pooled guards, not by this module.)

WARNING (spec A3.7, binding): `scripts/run_step2.py::_cross_day_per_state_sweep`
uses its ONE `split_by_segments` call's two sides directly as fit/conformal -- i.e.
the TOP split doubles as the fit/conformal split. That is correct THERE (its fit day
contributes no scoring windows, so a nested split would waste half the day) but
WRONG for pooled work, where every run keeps a scoring side that pooled artifacts
are later evaluated against. Do NOT copy that convention into this module or its
callers: pooled fit/conformal sides always come from the NESTED split of the
calibration side, exactly as `run_sweep` does.

`coverage_table`/`coverage_warnings` are the A4.1/A4.2 coverage machinery: windows
per (run, label) for any caller-supplied per-window label array -- Step-1 detected
cluster ids, GT state names, or composite `"state|load_bin"` strings assembled by
the caller from `rowii.scada.labels.gt_labels`' two output columns. The pool
machinery is deliberately labels-agnostic (it never derives labels itself, matching
`references.build_references`' "labels is deliberately generic" stance);
`coverage_warnings` compares a TRAINING-side table against an EVALUATION-side table
and warns for every label evaluated somewhere but covered nowhere in training --
"a mode with zero coverage on either side must be visible, never silent" (spec
A4.1), operationalized for load bins by passing composite labels (spec A4.2).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from rowii.anomaly.references import split_by_segments
from rowii.anomaly.sweep import SweepConfig
from rowii.pipeline import PreparedRun

logger = logging.getLogger(__name__)

_COVERAGE_COLUMNS = ["run", "label", "n_windows"]


@dataclass(frozen=True)
class PoolMember:
    """One run's contribution to a pooled side, from `build_pool`."""

    run_name: str
    """The run's name -- the key it had in `build_pool`'s `prepared` dict."""
    windows: np.ndarray
    """(N,) int64 -- ascending window indices into THIS run's own grid (the same
    index space as `PreparedRun.features`' rows). Empty when the run is too
    short/sparse to form the requested side (see `build_pool`)."""
    n_windows: int
    """`windows.shape[0]`, carried explicitly for provenance sidecars."""


@dataclass(frozen=True)
class PoolResult:
    """A pooled side across every run of a pool, from `build_pool`."""

    side: str
    """Which side was pooled: `"calibration"`, `"fit"`, or `"conformal"`."""
    members: list[PoolMember]
    """One member per run, in `prepared`'s own (insertion) order -- including
    empty-side members with `n_windows == 0`."""
    features: np.ndarray
    """(Ntotal, F) float64 -- every member's feature rows stacked in members order;
    always a COPY, never a view into any `PreparedRun.features`."""
    run_index: np.ndarray
    """(Ntotal,) int64 -- per stacked row, the index into `members` of the run the
    row came from."""
    window_index: np.ndarray
    """(Ntotal,) int64 -- per stacked row, the source window index into that run's
    own grid: `features[i] == prepared[members[run_index[i]].run_name]
    .features[window_index[i]]` bitwise."""
    provenance: dict[str, dict[str, int]]
    """Run name -> `{"n_windows": ...}` -- per-run window counts for the sidecars
    (spec D1: "PoolResult carries per-run provenance")."""


def _side_windows(
    run: PreparedRun, side: str, sweep_cfg: SweepConfig, *, run_name: str
) -> np.ndarray:
    """One run's window indices for *side* under the run_sweep split convention
    (module docstring) -- empty, with a WARNING log, when the run is too short or
    sparse to form the requested side (plan Task 1 binding: "empty side for a run ->
    member with n_windows 0 + warning, never a crash"). `split_by_segments` signals
    exactly that condition by raising `ValueError` on a degenerate split, so this is
    the one place a `split_by_segments` error is deliberately absorbed rather than
    propagated."""
    try:
        top = split_by_segments(
            run.segment_ids, run.valid_mask, sweep_cfg.calibration_frac, sweep_cfg.seed
        )
    except ValueError as e:
        logger.warning(
            "build_pool: run %r cannot form a top calibration/scoring split (%s) -- "
            "emitting an empty %r-side member (n_windows=0)",
            run_name,
            e,
            side,
        )
        return np.empty(0, dtype=np.int64)

    if side == "calibration":
        windows = top.calibration_windows
    else:
        calibration_mask = np.zeros(run.valid_mask.shape[0], dtype=bool)
        calibration_mask[top.calibration_windows] = True
        try:
            nested = split_by_segments(
                run.segment_ids, calibration_mask, 0.5, sweep_cfg.seed + 1
            )
        except ValueError as e:
            logger.warning(
                "build_pool: run %r cannot form a nested fit/conformal split of its "
                "calibration side (%s) -- emitting an empty %r-side member "
                "(n_windows=0)",
                run_name,
                e,
                side,
            )
            return np.empty(0, dtype=np.int64)
        windows = nested.calibration_windows if side == "fit" else nested.scoring_windows

    assert set(windows.tolist()).isdisjoint(set(top.scoring_windows.tolist())), (
        f"build_pool: run {run_name!r}'s {side!r}-side windows overlap its own "
        f"scoring side -- cross-run leakage rule violated (spec D1)"
    )
    return windows


def build_pool(
    prepared: dict[str, PreparedRun],
    side: Literal["calibration", "fit", "conformal"],
    sweep_cfg: SweepConfig,
) -> PoolResult:
    """Pool one leakage-safe side's windows and features across every run of
    *prepared* -- see module docstring for the split convention and the A3.7
    WARNING against copying `_cross_day_per_state_sweep`'s top-split-as-fit
    semantics.

    Args:
        prepared: Run name -> `PreparedRun`, at least one entry. Members (and the
            stacked rows) follow this dict's own insertion order. Every run must
            carry the same feature dimensionality (one variant's features -- pooling
            across variants is a caller bug, see Raises).
        side: Which side to pool: `"calibration"` (the whole top calibration side)
            or `"fit"` / `"conformal"` (the nested split's two parts).
        sweep_cfg: Supplies `calibration_frac` and `seed` for the top split (the
            nested split is always `(0.5, seed + 1)`, the fixed `run_sweep`
            convention). Its other fields are not read here.

    Returns:
        A `PoolResult` (see field docs). A run whose requested side cannot be formed
        (too few segments for the top or nested split) contributes an EMPTY member
        (`n_windows == 0`) plus a WARNING log -- never an exception.

    Raises:
        ValueError: if `side` is not one of the three literals (runtime guard, same
            posture as `run_sweep`'s `conditioning` check); if `prepared` is empty;
            if the runs disagree on feature dimensionality.
    """
    if side not in ("calibration", "fit", "conformal"):
        raise ValueError(
            f"side must be 'calibration', 'fit', or 'conformal', got {side!r}"
        )
    if not prepared:
        raise ValueError("build_pool: `prepared` must contain at least one run")

    feature_dims = {name: run.features.shape[1] for name, run in prepared.items()}
    if len(set(feature_dims.values())) > 1:
        raise ValueError(
            f"build_pool: runs disagree on feature dimensionality ({feature_dims}) "
            f"-- a pool must stack ONE variant's features across runs"
        )

    members: list[PoolMember] = []
    provenance: dict[str, dict[str, int]] = {}
    feature_blocks: list[np.ndarray] = []
    run_index_blocks: list[np.ndarray] = []
    window_index_blocks: list[np.ndarray] = []

    for member_idx, (run_name, run) in enumerate(prepared.items()):
        windows = _side_windows(run, side, sweep_cfg, run_name=run_name)
        n_windows = int(windows.shape[0])
        members.append(PoolMember(run_name=run_name, windows=windows, n_windows=n_windows))
        provenance[run_name] = {"n_windows": n_windows}

        block = run.features[windows].astype(np.float64, copy=True)
        assert np.isfinite(block).all(), (
            f"build_pool: run {run_name!r} contributed non-finite feature rows -- "
            f"`split_by_segments` must only return valid windows"
        )
        feature_blocks.append(block)
        run_index_blocks.append(np.full(n_windows, member_idx, dtype=np.int64))
        window_index_blocks.append(windows)

    # `np.concatenate` always allocates fresh output arrays, so `window_index` never
    # aliases any member's own `windows` array (and `features` is a copy per block).
    return PoolResult(
        side=side,
        members=members,
        features=np.vstack(feature_blocks),
        run_index=np.concatenate(run_index_blocks),
        window_index=np.concatenate(window_index_blocks),
        provenance=provenance,
    )


def coverage_table(
    prepared: dict[str, PreparedRun],
    windows_per_run: dict[str, np.ndarray],
    labels_per_run: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Windows per (run, label) over caller-selected window subsets -- the A4.1/A4.2
    coverage table for one side of a pool or evaluation.

    Args:
        prepared: Run name -> `PreparedRun` -- the alignment anchor: every counted
            run's labels must cover that run's FULL window grid, and its windows must
            index into it. May be a superset of `windows_per_run`'s runs.
        windows_per_run: Run name -> (N,) window indices to count (e.g. a
            `PoolMember.windows`, or a test run's scoring windows). Determines which
            runs appear in the table, in this dict's own order.
        labels_per_run: Run name -> (W,) per-window label array aligned with that
            run's grid -- ANY label space the caller chooses: detected cluster ids
            (int), GT state names (str), or composite `"state|load_bin"` strings
            (spec A4.2); this function never derives labels itself.

    Returns:
        DataFrame with columns `run, label, n_windows` -- one row per (run, label)
        pair observed among that run's selected windows (labels sorted per run via
        `np.unique`; a run with an empty selection contributes no rows). Zero-
        coverage VISIBILITY is `coverage_warnings`' job, by comparing two of these
        tables -- absent rows here simply mean zero windows.

    Raises:
        ValueError: if a `windows_per_run` run is missing from `prepared` or
            `labels_per_run`; if a run's labels do not cover its full window grid;
            if a run's windows index outside its grid.
    """
    rows: list[dict[str, int | str]] = []
    for run_name, windows in windows_per_run.items():
        if run_name not in prepared:
            raise ValueError(
                f"coverage_table: run {run_name!r} in windows_per_run is not in "
                f"`prepared` (available: {sorted(prepared)})"
            )
        if run_name not in labels_per_run:
            raise ValueError(
                f"coverage_table: run {run_name!r} in windows_per_run has no entry "
                f"in labels_per_run (available: {sorted(labels_per_run)})"
            )
        n_grid = prepared[run_name].features.shape[0]
        labels = labels_per_run[run_name]
        if labels.shape[0] != n_grid:
            raise ValueError(
                f"coverage_table: run {run_name!r}'s labels have {labels.shape[0]} "
                f"element(s) but its prepared grid has {n_grid} window(s) -- labels "
                f"must cover the run's full window grid"
            )
        if windows.size and (int(windows.min()) < 0 or int(windows.max()) >= n_grid):
            raise ValueError(
                f"coverage_table: run {run_name!r}'s windows index outside its grid "
                f"[0, {n_grid}) (min {int(windows.min())}, max {int(windows.max())})"
            )
        values, counts = np.unique(labels[windows], return_counts=True)
        # `.tolist()` yields native Python ints/strs -- the same no-dtype-branching
        # trick `references.build_references` uses for its label keys.
        for label, count in zip(values.tolist(), counts.tolist(), strict=True):
            rows.append({"run": run_name, "label": label, "n_windows": int(count)})
    return pd.DataFrame(rows, columns=_COVERAGE_COLUMNS)


def _label_totals(table: pd.DataFrame) -> dict[int | str, int]:
    """Label -> summed `n_windows` across every run row of a `coverage_table`
    output. A plain dict loop rather than a pandas groupby so int and str label
    columns behave identically (`.tolist()` restores native Python types)."""
    totals: dict[int | str, int] = {}
    for label, count in zip(table["label"].tolist(), table["n_windows"].tolist(), strict=True):
        totals[label] = totals.get(label, 0) + int(count)
    return totals


def coverage_warnings(train_table: pd.DataFrame, eval_table: pd.DataFrame) -> list[str]:
    """Warnings for every label present in *eval_table* but with ZERO training
    coverage in *train_table* -- the A4.1/A4.2 "never silent" rule: a mode (or
    state x load-bin cell, via composite labels) that is evaluated somewhere but
    trained nowhere must be visible.

    Coverage is pool-wide: a label counts as covered when its `n_windows` SUM across
    every training run is positive (an explicit zero-count row is NOT coverage), and
    only labels with a positive evaluation total can fire (nothing was evaluated
    otherwise). Each finding is BOTH logged at WARNING level (spec A4.2: "the pool
    builder logs a warning") and returned as a string for the caller's notes/sidecar.

    Args:
        train_table: `coverage_table` output for the training side (pool windows).
        eval_table: `coverage_table` output for the evaluation side (held-out runs'
            scored windows) -- built with the SAME label space as *train_table*,
            or the comparison is meaningless (caller's responsibility, matching the
            labels-agnostic stance of this module).

    Returns:
        One message per uncovered label, sorted by the label's string form for
        determinism; empty when every evaluated label has training coverage.
    """
    train_totals = _label_totals(train_table)
    eval_totals = _label_totals(eval_table)
    warnings: list[str] = []
    for label in sorted(eval_totals, key=str):
        n_eval = eval_totals[label]
        if n_eval <= 0:
            continue
        if train_totals.get(label, 0) > 0:
            continue
        message = (
            f"coverage: label {label!r} has {n_eval} evaluation window(s) but zero "
            f"training coverage"
        )
        logger.warning("%s", message)
        warnings.append(message)
    return warnings
