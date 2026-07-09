"""Tests for `rowii.anomaly.references`: leakage-safe segment splits (`split_by_segments`)
and per-label normal reference sets (`build_references`). Synthetic-only, per the Step-2
plan (`docs/superpowers/plans/2026-07-09-step2-first-package.md` Task S2) -- no real data.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from rowii.anomaly.references import ReferenceSet, SegmentSplit, build_references, split_by_segments

# ---------------------------------------------------------------------------
# split_by_segments
# ---------------------------------------------------------------------------

_WINDOWS_PER_SEGMENT = 50
_N_SEGMENTS = 6


def _six_equal_segments(
    windows_per_segment: int = _WINDOWS_PER_SEGMENT,
) -> tuple[np.ndarray, np.ndarray]:
    """segment_ids/valid_mask for `_N_SEGMENTS` contiguous, fully-valid segments of
    `windows_per_segment` windows each (300 windows total by default) -- the plan's own
    "6 segments x 50 windows" fixture."""
    segment_ids = np.repeat(
        np.arange(_N_SEGMENTS, dtype=np.int64), windows_per_segment
    )
    valid_mask = np.ones(segment_ids.shape[0], dtype=bool)
    return segment_ids, valid_mask


def _six_uneven_segments() -> tuple[np.ndarray, np.ndarray]:
    """segment_ids/valid_mask for 6 segments of DELIBERATELY unequal sizes (10..60
    windows, 210 total). Used only where a test needs the accumulated running count to
    genuinely depend on shuffle ORDER: with equal-sized segments (as in
    `_six_equal_segments`), the NUMBER of segments needed to clear a given
    calibration_frac threshold is invariant to shuffle order (only WHICH segments are
    picked varies), so a naive "different seed -> different split" test could
    coincidentally re-pick the same segment subset. Unequal sizes make both the count
    and the membership of the accumulated subset depend on order.
    """
    sizes = [10, 20, 30, 40, 50, 60]
    segment_ids = np.concatenate(
        [np.full(size, seg, dtype=np.int64) for seg, size in enumerate(sizes)]
    )
    valid_mask = np.ones(segment_ids.shape[0], dtype=bool)
    return segment_ids, valid_mask


def test_split_by_segments_no_segment_straddles_both_sides() -> None:
    segment_ids, valid_mask = _six_equal_segments()

    split = split_by_segments(segment_ids, valid_mask, calibration_frac=0.5, seed=0)

    assert isinstance(split, SegmentSplit)
    calib_segments = set(segment_ids[split.calibration_windows].tolist())
    scoring_segments = set(segment_ids[split.scoring_windows].tolist())
    assert calib_segments, "calibration side must not be empty"
    assert scoring_segments, "scoring side must not be empty"
    assert calib_segments.isdisjoint(scoring_segments)

    # Every window is valid here, so every one of the 300 windows must land on exactly
    # one side -- no window silently dropped, none duplicated.
    all_assigned = np.concatenate([split.calibration_windows, split.scoring_windows])
    assert sorted(all_assigned.tolist()) == list(range(segment_ids.shape[0]))


def test_split_by_segments_only_valid_windows_appear_in_either_side() -> None:
    segment_ids, valid_mask = _six_equal_segments()
    # Invalidate every 10th window (5 per 50-window segment) -- every segment still
    # has plenty of valid windows left, so this only tests the validity filter, not
    # segment-count degeneracy.
    valid_mask[::10] = False

    split = split_by_segments(segment_ids, valid_mask, calibration_frac=0.5, seed=0)

    all_assigned = np.concatenate([split.calibration_windows, split.scoring_windows])
    assert valid_mask[all_assigned].all()
    invalid_windows = np.flatnonzero(~valid_mask)
    assert not np.isin(invalid_windows, all_assigned).any()


def test_split_by_segments_deterministic_for_same_seed() -> None:
    segment_ids, valid_mask = _six_equal_segments()

    split_a = split_by_segments(segment_ids, valid_mask, calibration_frac=0.4, seed=7)
    split_b = split_by_segments(segment_ids, valid_mask, calibration_frac=0.4, seed=7)

    np.testing.assert_array_equal(split_a.calibration_windows, split_b.calibration_windows)
    np.testing.assert_array_equal(split_a.scoring_windows, split_b.scoring_windows)


def test_split_by_segments_different_seed_changes_the_split() -> None:
    # Verified empirically (not just assumed): seeds 0 vs 1 on the uneven-segment
    # fixture pick different segment subsets for calibration_frac=0.5.
    segment_ids, valid_mask = _six_uneven_segments()

    split_a = split_by_segments(segment_ids, valid_mask, calibration_frac=0.5, seed=0)
    split_b = split_by_segments(segment_ids, valid_mask, calibration_frac=0.5, seed=1)

    assert not np.array_equal(split_a.calibration_windows, split_b.calibration_windows)


def test_split_by_segments_respects_calibration_frac_within_one_segment_granularity() -> None:
    # 6 equal segments of 50 -> target for frac=0.3 is 90 windows; the greedy algorithm
    # must stop as soon as it clears 90, which -- since segments are uniform -- always
    # happens after exactly 2 segments (100 windows), regardless of shuffle order.
    segment_ids, valid_mask = _six_equal_segments()
    total = segment_ids.shape[0]
    frac = 0.3

    split = split_by_segments(segment_ids, valid_mask, calibration_frac=frac, seed=3)

    target = frac * total
    actual = split.calibration_windows.shape[0]
    assert actual >= target
    assert actual - target < _WINDOWS_PER_SEGMENT
    assert actual % _WINDOWS_PER_SEGMENT == 0


def test_split_by_segments_raises_on_single_segment() -> None:
    segment_ids = np.zeros(50, dtype=np.int64)
    valid_mask = np.ones(50, dtype=bool)

    with pytest.raises(ValueError, match="segment"):
        split_by_segments(segment_ids, valid_mask, calibration_frac=0.5, seed=0)


def test_split_by_segments_raises_on_calibration_frac_zero() -> None:
    segment_ids, valid_mask = _six_equal_segments()

    with pytest.raises(ValueError, match="non-empty"):
        split_by_segments(segment_ids, valid_mask, calibration_frac=0.0, seed=0)


def test_split_by_segments_raises_on_calibration_frac_one() -> None:
    segment_ids, valid_mask = _six_equal_segments()

    with pytest.raises(ValueError, match="non-empty"):
        split_by_segments(segment_ids, valid_mask, calibration_frac=1.0, seed=0)


def test_split_by_segments_error_message_names_the_window_and_segment_counts() -> None:
    segment_ids = np.zeros(50, dtype=np.int64)
    valid_mask = np.ones(50, dtype=bool)

    with pytest.raises(ValueError) as exc_info:
        split_by_segments(segment_ids, valid_mask, calibration_frac=0.5, seed=0)

    message = str(exc_info.value)
    # One side got all 50 windows (1 segment), the other got 0 windows (0 segments) --
    # the message must name these counts, not just say "degenerate".
    assert "50 window" in message
    assert "0 window" in message
    assert "1 segment" in message


def test_split_by_segments_raises_on_shape_mismatch() -> None:
    segment_ids = np.zeros(50, dtype=np.int64)
    valid_mask = np.ones(40, dtype=bool)

    with pytest.raises(ValueError, match="shape"):
        split_by_segments(segment_ids, valid_mask, calibration_frac=0.5, seed=0)


# ---------------------------------------------------------------------------
# build_references
# ---------------------------------------------------------------------------


def _synthetic_features(rng: np.random.Generator, n: int, n_features: int = 4) -> np.ndarray:
    return rng.normal(size=(n, n_features))


def test_build_references_partitions_by_int_label() -> None:
    rng = np.random.default_rng(0)
    features = _synthetic_features(rng, 90)
    labels = np.array([0] * 30 + [1] * 30 + [2] * 30, dtype=np.int64)
    windows = np.arange(90)

    result = build_references(features, labels, windows, min_ref=20)

    assert isinstance(result, ReferenceSet)
    assert set(result.references.keys()) == {0, 1, 2}
    assert all(isinstance(k, int) for k in result.references)
    for label, ref in result.references.items():
        np.testing.assert_array_equal(ref, features[labels == label])
    np.testing.assert_array_equal(result.pooled, features[windows])
    assert result.excluded == {}


def test_build_references_partitions_by_str_label() -> None:
    rng = np.random.default_rng(1)
    features = _synthetic_features(rng, 60)
    labels = np.array(["standstill"] * 20 + ["turbine"] * 20 + ["pump"] * 20)
    windows = np.arange(60)

    result = build_references(features, labels, windows, min_ref=20)

    assert set(result.references.keys()) == {"standstill", "turbine", "pump"}
    assert all(isinstance(k, str) for k in result.references)
    for label, ref in result.references.items():
        np.testing.assert_array_equal(ref, features[labels == label])
    assert result.excluded == {}


def test_build_references_excludes_labels_below_min_ref_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rng = np.random.default_rng(2)
    features = _synthetic_features(rng, 55)
    # label 0: 50 windows (kept, default min_ref=20); label 1: 5 windows (excluded).
    labels = np.array([0] * 50 + [1] * 5, dtype=np.int64)
    windows = np.arange(55)

    with caplog.at_level(logging.WARNING):
        result = build_references(features, labels, windows)  # default min_ref=20

    assert set(result.references.keys()) == {0}
    assert result.excluded == {1: 5}
    # Pooled must still include every drawn window, including the excluded label's.
    np.testing.assert_array_equal(result.pooled, features[windows])
    assert result.pooled.shape[0] == 55

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("label 1" in msg and "5 window" in msg for msg in warnings)


def test_build_references_only_uses_the_drawn_window_subset() -> None:
    rng = np.random.default_rng(3)
    features = _synthetic_features(rng, 20)
    labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    windows = np.arange(10, 20)  # only label-1 windows are drawn

    result = build_references(features, labels, windows, min_ref=5)

    assert set(result.references.keys()) == {1}
    np.testing.assert_array_equal(result.references[1], features[10:20])
    assert result.pooled.shape[0] == 10
    assert result.excluded == {}  # label 0 has zero occurrences in `windows`, not "excluded"


def test_build_references_asserts_on_non_finite_feature_row() -> None:
    features = np.zeros((10, 3))
    features[4, 1] = np.nan
    labels = np.zeros(10, dtype=np.int64)
    windows = np.arange(10)

    with pytest.raises(AssertionError, match="non-finite"):
        build_references(features, labels, windows, min_ref=1)


def test_build_references_raises_on_features_labels_length_mismatch() -> None:
    features = np.zeros((10, 3))
    labels = np.zeros(9, dtype=np.int64)
    windows = np.arange(9)

    with pytest.raises(ValueError, match="labels"):
        build_references(features, labels, windows, min_ref=1)
