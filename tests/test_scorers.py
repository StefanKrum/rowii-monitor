"""Tests for `rowii.anomaly.scorers`: `KnnScorer` (cosine/euclidean) and
`MahalanobisScorer` (diagonal + shrinkage). Synthetic-only, per the Step-2 plan
(`docs/superpowers/plans/2026-07-09-step2-first-package.md` Task S3) -- no real data.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from rowii.anomaly.scorers import KnnScorer, MahalanobisScorer

# ---------------------------------------------------------------------------
# Constructed inliers-vs-outliers (item 1)
# ---------------------------------------------------------------------------


def test_knn_cosine_directional_outlier_scores_exceed_inlier_max() -> None:
    # Reference + inliers cluster tightly around the e0 axis; outliers sit near the
    # orthogonal e1 axis -- a directional (angular) outlier, exactly what cosine
    # distance is meant to catch regardless of vector magnitude.
    rng = np.random.default_rng(0)
    n_features = 8
    e0 = np.zeros(n_features)
    e0[0] = 1.0
    e1 = np.zeros(n_features)
    e1[1] = 1.0

    reference = e0 + rng.normal(scale=0.02, size=(100, n_features))
    inliers = e0 + rng.normal(scale=0.02, size=(20, n_features))
    outliers = e1 + rng.normal(scale=0.02, size=(20, n_features))

    scorer = KnnScorer(k=1, metric="cosine").fit(reference)
    inlier_scores = scorer.score(inliers)
    outlier_scores = scorer.score(outliers)

    assert outlier_scores.min() > inlier_scores.max()


def test_mahalanobis_magnitude_outlier_along_low_variance_feature_exceeds_inlier_max() -> None:
    # Feature 0 has tiny reference variance (std 0.05), feature 1 has large reference
    # variance (std 3.0). Inliers vary only in feature 0, within its own typical
    # range; outliers are shifted by 1.0 in feature 0 alone (20x its raw std) while
    # feature 1 is held at the reference mean for BOTH groups -- isolating the
    # low-variance-feature deviation as the only difference between the groups, so
    # separation can only come from the per-feature (variance-aware) normalization a
    # naive Euclidean distance would not apply.
    rng = np.random.default_rng(1)
    n = 200
    feature0 = rng.normal(scale=0.05, size=n)
    feature1 = rng.normal(scale=3.0, size=n)
    reference = np.column_stack([feature0, feature1])
    reference_mean = reference.mean(axis=0)

    rng_in = np.random.default_rng(2)
    inliers = np.column_stack(
        [rng_in.normal(scale=0.05, size=20), np.full(20, reference_mean[1])]
    )
    outliers = np.column_stack(
        [np.full(20, reference_mean[0] + 1.0), np.full(20, reference_mean[1])]
    )

    scorer = MahalanobisScorer().fit(reference)
    inlier_scores = scorer.score(inliers)
    outlier_scores = scorer.score(outliers)

    assert outlier_scores.min() > inlier_scores.max()


# ---------------------------------------------------------------------------
# KnnScorer cosine equivalence to sklearn (item 2)
# ---------------------------------------------------------------------------


def test_knn_cosine_k1_matches_sklearn_cosine_similarity() -> None:
    rng = np.random.default_rng(3)
    reference = rng.normal(size=(15, 5))
    x = rng.normal(size=(6, 5))

    scores = KnnScorer(k=1, metric="cosine").fit(reference).score(x)

    expected = 1.0 - cosine_similarity(x, reference).max(axis=1)
    np.testing.assert_allclose(scores, expected, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Chunked == unchunked (item 3)
# ---------------------------------------------------------------------------


def test_knn_cosine_chunked_matches_unchunked() -> None:
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(30, 6))
    x = rng.normal(size=(50, 6))

    scores_chunked = KnnScorer(k=1, metric="cosine", chunk_size=3).fit(reference).score(x)
    scores_unchunked = (
        KnnScorer(k=1, metric="cosine", chunk_size=1_000_000).fit(reference).score(x)
    )

    np.testing.assert_allclose(scores_chunked, scores_unchunked, rtol=1e-12, atol=1e-14)


def test_knn_euclidean_chunked_matches_unchunked() -> None:
    rng = np.random.default_rng(5)
    reference = rng.normal(size=(30, 6))
    x = rng.normal(size=(50, 6))

    scores_chunked = KnnScorer(k=3, metric="euclidean", chunk_size=3).fit(reference).score(x)
    scores_unchunked = (
        KnnScorer(k=3, metric="euclidean", chunk_size=1_000_000).fit(reference).score(x)
    )

    np.testing.assert_allclose(scores_chunked, scores_unchunked, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# k=3 top-k mean semantics, hand-computed (item 4)
# ---------------------------------------------------------------------------


def test_knn_cosine_k3_top_k_mean_matches_hand_computed_case() -> None:
    # Five 2-D unit vectors at known angles from the query (angle 0): cosine
    # similarity to the query is exactly cos(angle). The top-3 (by similarity) are
    # the three smallest angles -- 0, 10, 20 degrees -- so the expected score is
    # independently derived by sorting the five known similarities, not by re-using
    # the implementation's own top-k selection.
    angles_deg = np.array([0.0, 10.0, 20.0, 90.0, 180.0])
    reference = np.column_stack(
        [np.cos(np.radians(angles_deg)), np.sin(np.radians(angles_deg))]
    )
    query = np.array([[1.0, 0.0]])

    score = KnnScorer(k=3, metric="cosine").fit(reference).score(query)[0]

    similarities = np.cos(np.radians(angles_deg))
    top3_mean = np.sort(similarities)[-3:].mean()
    expected = 1.0 - top3_mean
    np.testing.assert_allclose(score, expected, rtol=1e-12)


def test_knn_euclidean_kth_neighbor_distance_hand_computed() -> None:
    # Five collinear reference points 10 apart; k=2 -> score is the distance to the
    # SECOND-nearest reference point (not a mean), hand-derived from sorted
    # point-to-point distances.
    reference = np.array(
        [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]]
    )
    x = np.array([[0.0, 0.0], [5.0, 0.0]])

    scores = KnnScorer(k=2, metric="euclidean").fit(reference).score(x)

    # query [0,0] -> distances [0,10,20,30,40] -> 2nd-nearest = 10.
    # query [5,0] -> distances [5,5,15,25,35] -> 2nd-nearest = 5.
    np.testing.assert_allclose(scores, [10.0, 5.0])


def test_knn_euclidean_k1_matches_sklearn_nearest_neighbors_distance() -> None:
    rng = np.random.default_rng(6)
    reference = rng.normal(size=(20, 4))
    x = rng.normal(size=(7, 4))

    scores = KnnScorer(k=1, metric="euclidean").fit(reference).score(x)

    expected_distances, _ = NearestNeighbors(n_neighbors=1).fit(reference).kneighbors(x)
    np.testing.assert_allclose(scores, expected_distances[:, 0], rtol=1e-10)


# ---------------------------------------------------------------------------
# Mahalanobis shrinkage=1.0 equals isotropic (item 5)
# ---------------------------------------------------------------------------


def test_mahalanobis_shrinkage_one_equals_isotropic_variance() -> None:
    rng = np.random.default_rng(7)
    n = 300
    reference = np.column_stack(
        [
            rng.normal(scale=0.1, size=n),
            rng.normal(scale=2.0, size=n),
            rng.normal(scale=5.0, size=n),
        ]
    )
    x = rng.normal(size=(10, 3))

    scores = MahalanobisScorer(shrinkage=1.0).fit(reference).score(x)

    mean = reference.mean(axis=0)
    isotropic_var = reference.var(axis=0).mean()
    expected = np.sqrt(np.sum(np.square(x - mean) / isotropic_var, axis=1))
    np.testing.assert_allclose(scores, expected, rtol=1e-10)


def test_mahalanobis_shrinkage_zero_uses_raw_per_feature_variance() -> None:
    # Complements the shrinkage=1.0 case: shrinkage=0.0 must use the raw per-feature
    # variance unmixed (the other end of the shrinkage formula).
    rng = np.random.default_rng(8)
    n = 300
    reference = np.column_stack(
        [rng.normal(scale=0.1, size=n), rng.normal(scale=4.0, size=n)]
    )
    x = rng.normal(size=(10, 2))

    scores = MahalanobisScorer(shrinkage=0.0).fit(reference).score(x)

    mean = reference.mean(axis=0)
    raw_var = reference.var(axis=0)
    expected = np.sqrt(np.sum(np.square(x - mean) / raw_var, axis=1))
    np.testing.assert_allclose(scores, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Determinism (item 6)
# ---------------------------------------------------------------------------


def test_knn_scoring_is_deterministic() -> None:
    rng = np.random.default_rng(9)
    reference = rng.normal(size=(40, 4))
    x = rng.normal(size=(15, 4))

    # Same fitted instance, scored twice.
    scorer = KnnScorer(k=2, metric="cosine").fit(reference)
    repeat_a = scorer.score(x)
    repeat_b = scorer.score(x)
    np.testing.assert_array_equal(repeat_a, repeat_b)

    # Two independently constructed-and-fitted instances on the same reference.
    fresh_a = KnnScorer(k=2, metric="cosine").fit(reference).score(x)
    fresh_b = KnnScorer(k=2, metric="cosine").fit(reference).score(x)
    np.testing.assert_array_equal(fresh_a, fresh_b)


def test_mahalanobis_scoring_is_deterministic() -> None:
    rng = np.random.default_rng(10)
    reference = rng.normal(size=(40, 4))
    x = rng.normal(size=(15, 4))

    scorer = MahalanobisScorer(shrinkage=0.2).fit(reference)
    repeat_a = scorer.score(x)
    repeat_b = scorer.score(x)
    np.testing.assert_array_equal(repeat_a, repeat_b)

    fresh_a = MahalanobisScorer(shrinkage=0.2).fit(reference).score(x)
    fresh_b = MahalanobisScorer(shrinkage=0.2).fit(reference).score(x)
    np.testing.assert_array_equal(fresh_a, fresh_b)


# ---------------------------------------------------------------------------
# name attribute
# ---------------------------------------------------------------------------


def test_scorer_name_attributes() -> None:
    assert KnnScorer().name == "knn"
    assert MahalanobisScorer().name == "mahalanobis"


# ---------------------------------------------------------------------------
# Error paths (item 7)
# ---------------------------------------------------------------------------


def test_knn_score_before_fit_raises_value_error() -> None:
    with pytest.raises(ValueError, match="fit"):
        KnnScorer().score(np.zeros((3, 2)))


def test_knn_euclidean_score_before_fit_raises_value_error() -> None:
    with pytest.raises(ValueError, match="fit"):
        KnnScorer(metric="euclidean").score(np.zeros((3, 2)))


def test_mahalanobis_score_before_fit_raises_value_error() -> None:
    with pytest.raises(ValueError, match="fit"):
        MahalanobisScorer().score(np.zeros((3, 2)))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_knn_fit_raises_on_non_finite_reference(bad_value: float) -> None:
    reference = np.ones((10, 3))
    reference[3, 1] = bad_value

    with pytest.raises(ValueError, match="finite"):
        KnnScorer().fit(reference)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_mahalanobis_fit_raises_on_non_finite_reference(bad_value: float) -> None:
    reference = np.ones((10, 3))
    reference[3, 1] = bad_value

    with pytest.raises(ValueError, match="finite"):
        MahalanobisScorer().fit(reference)


def test_knn_fit_raises_when_k_exceeds_reference_size() -> None:
    reference = np.random.default_rng(11).normal(size=(3, 2))

    with pytest.raises(ValueError, match="k"):
        KnnScorer(k=5).fit(reference)


def test_knn_fit_raises_on_empty_reference() -> None:
    reference = np.empty((0, 4))

    with pytest.raises(ValueError, match="empty"):
        KnnScorer().fit(reference)


def test_mahalanobis_fit_raises_on_empty_reference() -> None:
    reference = np.empty((0, 4))

    with pytest.raises(ValueError, match="empty"):
        MahalanobisScorer().fit(reference)


def test_knn_cosine_fit_raises_on_zero_norm_reference_row() -> None:
    reference = np.random.default_rng(12).normal(size=(10, 3))
    reference[4] = 0.0

    with pytest.raises(ValueError, match="zero-norm"):
        KnnScorer(metric="cosine").fit(reference)


def test_knn_euclidean_fit_does_not_require_nonzero_rows() -> None:
    # The zero-norm guard is a cosine-only concern (L2-normalization); Euclidean
    # fitting must accept an all-zero row without raising.
    reference = np.random.default_rng(13).normal(size=(10, 3))
    reference[4] = 0.0

    KnnScorer(metric="euclidean").fit(reference)  # must not raise


def test_knn_constructor_raises_on_invalid_metric() -> None:
    with pytest.raises(ValueError, match="metric"):
        KnnScorer(metric="manhattan")


def test_knn_constructor_raises_on_k_less_than_one() -> None:
    with pytest.raises(ValueError, match="k"):
        KnnScorer(k=0)


def test_knn_constructor_raises_on_chunk_size_less_than_one() -> None:
    """chunk_size must be >= 1: chunk_size=0 crashes low-level, chunk_size=-1 silently
    returns uninitialized memory from np.empty."""
    with pytest.raises(ValueError, match="chunk_size"):
        KnnScorer(chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
        KnnScorer(chunk_size=-1)


def test_knn_cosine_score_raises_on_nan_in_query() -> None:
    """Query with NaN must be rejected before scoring."""
    reference = np.random.default_rng(14).normal(size=(10, 3))
    scorer = KnnScorer(k=1, metric="cosine").fit(reference)
    x = np.ones((5, 3))
    x[2, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        scorer.score(x)


def test_knn_cosine_score_raises_on_inf_in_query() -> None:
    """Query with inf must be rejected before scoring."""
    reference = np.random.default_rng(15).normal(size=(10, 3))
    scorer = KnnScorer(k=1, metric="cosine").fit(reference)
    x = np.ones((5, 3))
    x[1, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        scorer.score(x)


def test_knn_cosine_score_raises_on_neginf_in_query() -> None:
    """Query with -inf must be rejected before scoring."""
    reference = np.random.default_rng(16).normal(size=(10, 3))
    scorer = KnnScorer(k=1, metric="cosine").fit(reference)
    x = np.ones((5, 3))
    x[0, 2] = -np.inf
    with pytest.raises(ValueError, match="non-finite"):
        scorer.score(x)


def test_knn_euclidean_score_raises_on_nan_in_query() -> None:
    """Query with NaN must be rejected before scoring (euclidean metric)."""
    reference = np.random.default_rng(17).normal(size=(10, 3))
    scorer = KnnScorer(k=1, metric="euclidean").fit(reference)
    x = np.ones((5, 3))
    x[2, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        scorer.score(x)


def test_knn_euclidean_score_raises_on_inf_in_query() -> None:
    """Query with inf must be rejected before scoring (euclidean metric)."""
    reference = np.random.default_rng(18).normal(size=(10, 3))
    scorer = KnnScorer(k=1, metric="euclidean").fit(reference)
    x = np.ones((5, 3))
    x[1, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        scorer.score(x)


def test_mahalanobis_score_raises_on_nan_in_query() -> None:
    """Query with NaN must be rejected before scoring."""
    reference = np.random.default_rng(19).normal(size=(10, 3))
    scorer = MahalanobisScorer().fit(reference)
    x = np.ones((5, 3))
    x[2, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        scorer.score(x)


def test_mahalanobis_score_raises_on_inf_in_query() -> None:
    """Query with inf must be rejected before scoring."""
    reference = np.random.default_rng(20).normal(size=(10, 3))
    scorer = MahalanobisScorer().fit(reference)
    x = np.ones((5, 3))
    x[1, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        scorer.score(x)
