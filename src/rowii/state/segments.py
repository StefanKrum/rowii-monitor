"""Duration-based run merging and segment-table export for per-window state labels.

`duration_filter` removes short spurious runs (e.g. residual single/few-window
flicker that survives `StickyHmmSmoother`) by iteratively merging every maximal
run shorter than `min_dwell` windows into a neighbouring run. `to_segments`
converts a (possibly filtered) per-window label sequence into a segment table
with wall-clock UTC boundaries taken from a `WindowGrid`.

Determinism of `duration_filter`: each iteration selects the SHORTEST run that
is still below `min_dwell` (ties broken by the LEFTMOST such run), and merges
it into whichever neighbour has the LONGER duration (ties broken by the LEFT
neighbour). A run at either array edge has only one neighbour, which absorbs
it unconditionally. Because a merge always deletes exactly one run and never
creates a new one, the run count strictly decreases every iteration, so the
loop is guaranteed to terminate (at latest when a single run remains). This
fixed processing order is what makes pathological inputs such as alternating
patterns resolve to one specific output rather than depending on iteration
order of ties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rowii.signals.windows import WindowGrid


class _Run:
    __slots__ = ("value", "length")

    def __init__(self, value: int, length: int) -> None:
        self.value = value
        self.length = length


def _runs_from_labels(labels: np.ndarray) -> list[_Run]:
    runs: list[_Run] = []
    for value in labels.tolist():
        if runs and runs[-1].value == value:
            runs[-1].length += 1
        else:
            runs.append(_Run(value, 1))
    return runs


def _expand_runs(runs: list[_Run]) -> np.ndarray:
    if not runs:
        return np.array([], dtype=np.int64)
    return np.concatenate(
        [np.full(r.length, r.value, dtype=np.int64) for r in runs]
    )


def _coalesce_same_value_neighbours(runs: list[_Run], idx: int) -> None:
    """Merge runs[idx] with same-value neighbours on either side, in place.

    Needed because a run flanked by two runs of the IDENTICAL value dissolves
    into whichever side wins, leaving that side newly adjacent to the other
    (already-identical-valued) neighbour -- those two must then coalesce into
    one run too, or a same-value adjacency (structurally invalid for a
    maximal-run representation) would remain.
    """
    while idx + 1 < len(runs) and runs[idx].value == runs[idx + 1].value:
        runs[idx].length += runs[idx + 1].length
        del runs[idx + 1]
    while idx - 1 >= 0 and runs[idx - 1].value == runs[idx].value:
        runs[idx - 1].length += runs[idx].length
        del runs[idx]
        idx -= 1


def duration_filter(labels: np.ndarray, min_dwell: int) -> np.ndarray:
    """Merge every maximal run shorter than `min_dwell` windows into a neighbour.

    Args:
        labels: Per-window integer state labels, shape (W,).
        min_dwell: Minimum run length in windows. Runs strictly shorter than
            this are merged away. `min_dwell <= 1` is a no-op (every run has
            length >= 1, so nothing can be shorter than 1).

    Returns:
        Filtered labels, shape (W,), dtype int64. Always a copy (never the
        same array object as `labels`), even when no merge happens.

    See module docstring for the exact deterministic merge order.
    """
    labels_i64 = np.asarray(labels, dtype=np.int64)
    if min_dwell <= 1:
        return labels_i64.copy()

    runs = _runs_from_labels(labels_i64)

    while len(runs) > 1:
        candidates = [i for i, r in enumerate(runs) if r.length < min_dwell]
        if not candidates:
            break
        shortest_length = min(runs[i].length for i in candidates)
        target = next(i for i in candidates if runs[i].length == shortest_length)

        has_left = target > 0
        has_right = target < len(runs) - 1
        if has_left and has_right:
            merge_left = runs[target - 1].length >= runs[target + 1].length
        elif has_left:
            merge_left = True
        else:  # has_right only; target==0 and len(runs) > 1 guarantees this holds
            merge_left = False

        run_length = runs[target].length
        if merge_left:
            absorber_idx = target - 1
            runs[absorber_idx].length += run_length
            del runs[target]
            _coalesce_same_value_neighbours(runs, absorber_idx)
        else:
            absorber_idx = target + 1
            runs[absorber_idx].length += run_length
            del runs[target]
            _coalesce_same_value_neighbours(runs, target)

    return _expand_runs(runs)


def to_segments(labels: np.ndarray, grid: WindowGrid) -> pd.DataFrame:
    """Convert per-window labels into a segment table with UTC boundaries.

    Args:
        labels: Per-window integer state labels, shape (W,). Must have
            `len(labels) == grid.n_windows`.
        grid: The `WindowGrid` whose edge timestamps bound each window.

    Returns:
        DataFrame with one row per maximal run of identical labels, columns:
            - `start_utc` (pd.Timestamp, tz="UTC"): left edge of the run's
              first window.
            - `end_utc` (pd.Timestamp, tz="UTC"): right edge of the run's
              last window (exclusive upper bound of the segment).
            - `duration_s` (float64): `end_utc - start_utc` in seconds.
            - `cluster` (int64): the run's label value.

    Raises:
        ValueError: if `len(labels) != grid.n_windows`.
    """
    if len(labels) != grid.n_windows:
        raise ValueError(
            f"labels length ({len(labels)}) must equal grid.n_windows "
            f"({grid.n_windows})"
        )

    labels_i64 = np.asarray(labels, dtype=np.int64)
    edges_ns = grid.edges_ns()
    runs = _runs_from_labels(labels_i64)

    rows = []
    window_idx = 0
    for r in runs:
        start_ns = int(edges_ns[window_idx])
        end_ns = int(edges_ns[window_idx + r.length])
        rows.append(
            {
                "start_utc": pd.Timestamp(start_ns, unit="ns", tz="UTC"),
                "end_utc": pd.Timestamp(end_ns, unit="ns", tz="UTC"),
                "duration_s": (end_ns - start_ns) / 1e9,
                "cluster": np.int64(r.value),
            }
        )
        window_idx += r.length

    return pd.DataFrame(
        rows, columns=["start_utc", "end_utc", "duration_s", "cluster"]
    ).astype({"cluster": "int64", "duration_s": "float64"})
