"""Leakage-safe calibration/scoring window splits and per-label normal reference sets
for Step-2 mode-conditioned anomaly scoring.

`split_by_segments` partitions a run's valid windows into a calibration side and a
scoring side by shuffling whole 12-minute recording SEGMENTS (`PreparedRun.segment_ids`)
rather than individual windows: a scorer calibrated on part of a segment and then scored
on another part of the SAME segment would leak information across a boundary that,
physically, sits inside one contiguous recording -- the design's leakage rule
("calibration and scoring never share a 12-min recording segment").

`build_references` then collapses one side's windows (typically calibration) into
per-label normal reference matrices -- one reference matrix per label, plus a `pooled`
(label-agnostic) reference spanning every drawn window regardless of label. `labels` is
deliberately generic: the design's central mode-conditioning comparison
(`reference_labels: detected | gt`) calls this same function once with Step-1's DETECTED
cluster ids (run-time realism) and, for diagnostics only, once more with GT state names
-- this module has no opinion on which, it only needs a `(W,)` array of int or str
labels aligned with `features`' rows.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentSplit:
    """Leakage-safe partition of a run's valid windows, from `split_by_segments`."""

    calibration_windows: np.ndarray
    """(Wc,) int64 -- ascending window indices assigned to calibration."""
    scoring_windows: np.ndarray
    """(Ws,) int64 -- ascending window indices assigned to scoring."""


def split_by_segments(
    segment_ids: np.ndarray,
    valid_mask: np.ndarray,
    calibration_frac: float,
    seed: int,
) -> SegmentSplit:
    """Partition *segment_ids*' valid windows into calibration/scoring sets, whole
    segments at a time, so no 12-minute recording segment ever contributes windows to
    both sides.

    Algorithm: the run's distinct segment ids (`np.unique(segment_ids)`, excluding the
    `-1` "uncovered" sentinel -- see `PreparedRun.segment_ids`) are shuffled with a
    `numpy.random.default_rng(seed)` permutation, then walked in that order and
    greedily assigned to calibration until the running count of VALID windows already
    claimed reaches `calibration_frac * (total valid windows across all real
    segments)`; every remaining segment goes to scoring. Because assignment happens at
    segment granularity, the realized calibration fraction can only approximate
    `calibration_frac`, off by at most the size of the one segment that tipped the
    running count over the target.

    Args:
        segment_ids: Per-window source-segment index, shape (W,), e.g.
            `PreparedRun.segment_ids` -- `-1` marks a window no segment covers at all;
            such windows are never assigned to either side.
        valid_mask: Per-window validity, shape (W,), matching `segment_ids`' shape --
            invalid windows never appear in either output array, regardless of which
            side their segment lands on.
        calibration_frac: Target fraction of all valid (real-segment) windows to place
            in calibration. Values that leave one side empty (0, 1, or anything a
            segment's own granularity cannot satisfy, e.g. a single segment) raise
            `ValueError` -- see Raises.
        seed: Seed for the segment-shuffle RNG. The same seed always reproduces the
            identical split; a different seed produces a different shuffle order (and,
            almost always in practice, a different resulting split).

    Returns:
        A `SegmentSplit` with both window-index arrays in ascending order.

    Raises:
        ValueError: if `segment_ids.shape != valid_mask.shape`, or if the resulting
            calibration or scoring side would be empty -- the message names the
            window and segment counts on both sides.
    """
    if segment_ids.shape != valid_mask.shape:
        raise ValueError(
            f"segment_ids.shape ({segment_ids.shape}) must equal valid_mask.shape "
            f"({valid_mask.shape})"
        )

    unique_segments = np.unique(segment_ids[segment_ids != -1])

    valid_real_segment_ids = segment_ids[valid_mask]
    valid_real_segment_ids = valid_real_segment_ids[valid_real_segment_ids != -1]
    counted_segments, counts = np.unique(valid_real_segment_ids, return_counts=True)
    count_by_segment = dict(zip(counted_segments.tolist(), counts.tolist(), strict=True))

    total_valid = int(counts.sum()) if counts.size else 0
    target = calibration_frac * total_valid

    rng = np.random.default_rng(seed)
    shuffled_segments = rng.permutation(unique_segments)

    calib_segment_ids: list[int] = []
    running_count = 0
    for seg in shuffled_segments.tolist():
        if running_count >= target:
            break
        calib_segment_ids.append(seg)
        running_count += count_by_segment.get(seg, 0)

    calib_segment_set = set(calib_segment_ids)
    scoring_segment_ids = [
        seg for seg in unique_segments.tolist() if seg not in calib_segment_set
    ]

    calibration_mask = valid_mask & np.isin(segment_ids, calib_segment_ids)
    scoring_mask = valid_mask & np.isin(segment_ids, scoring_segment_ids)
    calibration_windows = np.flatnonzero(calibration_mask).astype(np.int64)
    scoring_windows = np.flatnonzero(scoring_mask).astype(np.int64)

    if calibration_windows.size == 0 or scoring_windows.size == 0:
        raise ValueError(
            f"split_by_segments: degenerate split from {len(unique_segments)} "
            f"segment(s) at calibration_frac={calibration_frac!r} -- calibration got "
            f"{calibration_windows.size} window(s) across {len(calib_segment_ids)} "
            f"segment(s), scoring got {scoring_windows.size} window(s) across "
            f"{len(scoring_segment_ids)} segment(s); both sides must be non-empty"
        )

    return SegmentSplit(
        calibration_windows=calibration_windows, scoring_windows=scoring_windows
    )


@dataclass(frozen=True)
class ReferenceSet:
    """Per-label normal reference matrices, from `build_references`."""

    references: dict[int | str, np.ndarray]
    """Label -> (Ni, F) reference matrix, one entry per label with >= min_ref drawn
    windows. Keys are plain `int` for cluster-id labels or plain `str` for GT-state
    labels, matching whichever dtype `labels` had (see `build_references`)."""
    pooled: np.ndarray
    """(Nall, F) -- every drawn window's feature row, in `windows`' own order,
    regardless of label and regardless of the min_ref exclusion below."""
    excluded: dict[int | str, int]
    """Label -> count of drawn windows carrying that label, for every label whose
    count fell below `min_ref` (and is therefore absent from `references`). A label
    with zero occurrences in `windows` is not mentioned here at all -- there is
    nothing to exclude."""


def build_references(
    features: np.ndarray,
    labels: np.ndarray,
    windows: np.ndarray,
    *,
    min_ref: int = 20,
) -> ReferenceSet:
    """Collapse `features[windows]` into per-label reference matrices, plus a pooled
    (label-agnostic) reference over every drawn window.

    Args:
        features: Per-window feature matrix, shape (W, F). Rows outside `windows` are
            never read.
        labels: Per-window labels aligned with `features`, shape (W,) -- either an
            integer dtype (detected cluster ids) or a string/object dtype (GT state
            names); both are supported (see module docstring).
        windows: Window indices to draw from, shape (N,) -- typically one side of a
            `SegmentSplit` (e.g. `calibration_windows`). Every drawn feature row must
            already be finite (see Raises); `split_by_segments` guarantees this since
            it never returns an invalid window.
        min_ref: Minimum number of drawn windows a label needs to get its own entry in
            `references`; labels below this are reported in `excluded` instead (and
            still contribute their rows to `pooled`).

    Returns:
        A `ReferenceSet` (see field docs).

    Raises:
        ValueError: if `features.shape[0] != labels.shape[0]`.
        AssertionError: if any drawn window's feature row has a non-finite value.
    """
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"features.shape[0] ({features.shape[0]}) must equal labels.shape[0] "
            f"({labels.shape[0]})"
        )

    drawn_features = features[windows]
    non_finite = ~np.isfinite(drawn_features).all(axis=1)
    assert not non_finite.any(), (
        f"build_references: {int(non_finite.sum())} of {len(windows)} drawn window(s) "
        f"have a non-finite feature value (first offending window index "
        f"{int(windows[non_finite][0])}) -- `windows` must only contain "
        f"already-valid windows"
    )

    drawn_labels = labels[windows]

    references: dict[int | str, np.ndarray] = {}
    excluded: dict[int | str, int] = {}
    # `.tolist()` converts each unique value to its native Python type (int for any
    # numpy integer dtype, str for unicode/object-string dtypes) -- exactly the
    # `int | str` key type `ReferenceSet` declares, with no dtype branching needed.
    for label in np.unique(drawn_labels).tolist():
        label_mask = drawn_labels == label
        count = int(label_mask.sum())
        if count < min_ref:
            excluded[label] = count
            logger.warning(
                "build_references: label %r has %d window(s) in `windows`, below "
                "min_ref=%d -- excluded from per-label references (pooled reference "
                "still includes them)",
                label,
                count,
                min_ref,
            )
            continue
        references[label] = drawn_features[label_mask]

    return ReferenceSet(references=references, pooled=drawn_features, excluded=excluded)
