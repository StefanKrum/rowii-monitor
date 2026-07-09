"""kNN embedding-distance and Mahalanobis anomaly scorers for Step-2 mode-conditioned
scoring (design spec `docs/superpowers/specs/2026-07-09-step2-mode-conditioned-ad-design.md`,
plan `docs/superpowers/plans/2026-07-09-step2-first-package.md` Task S3).

Both scorers share the same two-step contract (`Scorer` protocol below): `fit(reference)`
consumes a `(N, F)` matrix of NORMAL reference embeddings/features for one operating
mode (or the pooled, mode-agnostic reference -- `rowii.anomaly.references.build_references`'s
output), then `score(x)` maps a `(W, F)` batch of windows to `(W,)` anomaly scores, higher
always meaning more anomalous -- so both scorers, and anything added to this module later,
can be thresholded and ranked identically by callers (`rowii.anomaly.conformal`, Task S4).

`KnnScorer` (k=1, cosine) is the design's cited default: distance to the nearest normal
reference embedding is a simple yet effective anomaly score for pretrained-audio
embeddings (design §3.5). For cosine, `score = 1 - max_similarity` at k=1, generalizing
to `1 - mean(top-k similarities)` for k>1 (the k=1 case is the mean of a single value,
so one formula covers both). An `"euclidean"` metric option scores by the distance to
the k-th nearest neighbour instead (via `sklearn.neighbors.NearestNeighbors`) -- a
different aggregation (the single k-th distance, not a top-k mean) since Euclidean
neighbour search does not hand back a bounded per-neighbour similarity to average.

`MahalanobisScorer` (diagonal covariance + shrinkage) summarises the reference by its
per-feature mean and variance and scores by the resulting diagonal Mahalanobis
distance -- the classical alternative that accounts for the normal class's covariance
structure (design §3.5). `shrinkage` (0 = pure per-feature variance, 1 = fully
isotropic) keeps a reference feature with near-zero variance from letting a
correspondingly tiny deviation dominate the score; see `MahalanobisScorer.fit`.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.neighbors import NearestNeighbors

_VAR_FLOOR = 1e-12


class Scorer(Protocol):
    """Common shape for anomaly scorers: name + fit(reference) + score(windows)."""

    name: str

    def fit(self, reference: np.ndarray) -> Scorer:
        """Fit on a `(N, F)` finite reference matrix of normal windows; returns self."""
        ...

    def score(self, x: np.ndarray) -> np.ndarray:
        """Map `(W, F)` windows to `(W,)` float64 scores, higher = more anomalous."""
        ...


def _check_reference(reference: np.ndarray) -> None:
    """Shared `fit()` precondition for both scorers: all-finite, at least one row.

    Raises:
        ValueError: if `reference` contains a NaN/inf value, or has zero rows.
    """
    non_finite = ~np.isfinite(reference)
    if non_finite.any():
        bad_rows = np.flatnonzero(non_finite.any(axis=1))
        raise ValueError(
            f"reference contains {int(non_finite.sum())} non-finite value(s) "
            f"(first offending row index {int(bad_rows[0])}) -- fit() requires an "
            f"all-finite reference matrix"
        )
    if reference.shape[0] == 0:
        raise ValueError(f"reference must be non-empty, got shape {reference.shape}")


def _check_query(x: np.ndarray) -> None:
    """Shared `score()` precondition: query contains only finite values.

    Raises:
        ValueError: if `x` contains a NaN/inf value.
    """
    if not np.isfinite(x).all():
        raise ValueError(
            "query contains non-finite values; filter via valid_mask before scoring"
        )


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize, leaving any zero-norm row unchanged (avoids 0/0) --
    matches `sklearn.preprocessing.normalize`'s zero-vector convention. A zero-norm
    query row scores exactly 1.0 (orthogonal-equivalent to all reference vectors),
    not maximally dissimilar (range max is 2.0). Only ever applied to a
    `score()`-side query batch, never to the reference (see `KnnScorer.fit`, which
    raises instead of silently tolerating a zero-norm reference row)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return x / safe_norms


def _chunk_bounds(n: int, chunk_size: int) -> list[tuple[int, int]]:
    """`[(start, end), ...]` slice bounds covering `range(n)` in steps of `chunk_size`."""
    return [(start, min(start + chunk_size, n)) for start in range(0, n, chunk_size)]


class KnnScorer:
    """kNN embedding-distance scorer (design spec §3.5) -- see module docstring for
    the cosine (top-k mean) vs. euclidean (k-th neighbour distance) formulas."""

    name: str = "knn"

    def __init__(self, k: int = 1, metric: str = "cosine", chunk_size: int = 4096) -> None:
        """Args:
            k: Neighbours to aggregate. Cosine: mean of the top-k similarities (k=1
                reduces to the plain max). Euclidean: distance to the k-th nearest
                neighbour.
            metric: `"cosine"` or `"euclidean"`.
            chunk_size: Number of `x` rows `score()` processes per batch, to cap peak
                memory when scoring a large `x` against a large reference. Does not
                change the result (see `test_scorers.py`'s chunked/unchunked tests).

        Raises:
            ValueError: if `k < 1`, `chunk_size < 1`, or `metric` is neither
                `"cosine"` nor `"euclidean"`.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric!r}")
        self.k = k
        self.metric = metric
        self.chunk_size = chunk_size
        self._reference: np.ndarray | None = None
        """Cosine only: L2-normalized reference, set by `fit`."""
        self._nn: NearestNeighbors | None = None
        """Euclidean only: fitted neighbour index, set by `fit`."""

    def fit(self, reference: np.ndarray) -> KnnScorer:
        """Store *reference* for scoring: L2-normalize it (cosine) or index it with
        `sklearn.neighbors.NearestNeighbors` (euclidean).

        Args:
            reference: `(N, F)` finite matrix of normal reference windows.

        Returns:
            self.

        Raises:
            ValueError: if `reference` is empty or has a non-finite value (see
                `_check_reference`); if `k` exceeds `N`; if `metric == "cosine"` and
                any reference row has zero L2 norm (cannot be normalized).
        """
        _check_reference(reference)
        if self.k > reference.shape[0]:
            raise ValueError(
                f"k ({self.k}) must be <= reference row count ({reference.shape[0]})"
            )
        if self.metric == "cosine":
            norms = np.linalg.norm(reference, axis=1)
            zero_rows = np.flatnonzero(norms == 0.0)
            if zero_rows.size:
                raise ValueError(
                    f"reference contains {zero_rows.size} zero-norm row(s) (first at "
                    f"index {int(zero_rows[0])}) -- cannot L2-normalize for cosine "
                    f"similarity"
                )
            self._reference = reference / norms[:, None]
        else:
            self._nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean").fit(reference)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """`(W, F)` windows -> `(W,)` float64 scores, higher = more anomalous.

        Raises:
            ValueError: if called before `fit()`, or if `x` contains non-finite values.
        """
        _check_query(x)
        if self.metric == "cosine":
            if self._reference is None:
                raise ValueError("KnnScorer.score() called before fit()")
            return self._score_cosine(x)
        if self._nn is None:
            raise ValueError("KnnScorer.score() called before fit()")
        return self._score_euclidean(x)

    def _score_cosine(self, x: np.ndarray) -> np.ndarray:
        reference = self._reference
        assert reference is not None  # for mypy; score() already checked this
        n = x.shape[0]
        scores = np.empty(n, dtype=np.float64)
        for start, end in _chunk_bounds(n, self.chunk_size):
            chunk = _l2_normalize(x[start:end])
            similarity = chunk @ reference.T
            top_k = np.partition(similarity, -self.k, axis=1)[:, -self.k :]
            scores[start:end] = 1.0 - top_k.mean(axis=1)
        return scores

    def _score_euclidean(self, x: np.ndarray) -> np.ndarray:
        nn = self._nn
        assert nn is not None  # for mypy; score() already checked this
        n = x.shape[0]
        scores = np.empty(n, dtype=np.float64)
        for start, end in _chunk_bounds(n, self.chunk_size):
            distances, _ = nn.kneighbors(x[start:end], n_neighbors=self.k)
            scores[start:end] = distances[:, -1]
        return scores


class MahalanobisScorer:
    """Diagonal-covariance Mahalanobis scorer with shrinkage (design spec §3.5) --
    see module docstring."""

    name: str = "mahalanobis"

    def __init__(self, shrinkage: float = 0.1) -> None:
        """Args:
            shrinkage: Mixing weight toward the isotropic (mean) variance:
                `var_shrunk = (1 - shrinkage) * var + shrinkage * mean(var)`. 0 = pure
                per-feature variance, 1 = fully isotropic (see `fit`).
        """
        self.shrinkage = shrinkage
        self._mean: np.ndarray | None = None
        self._var_shrunk: np.ndarray | None = None

    def fit(self, reference: np.ndarray) -> MahalanobisScorer:
        """Fit the per-feature mean and shrinkage-adjusted variance.

        `var_shrunk = max((1 - shrinkage) * var + shrinkage * mean(var), 1e-12)`,
        elementwise, where `var` is *reference*'s per-feature (population, ddof=0)
        variance and `mean(var)` is that variance vector's own scalar mean across all
        F features. The `1e-12` floor keeps a reference feature with near-zero (or,
        under a shrinkage > 1, negative) shrunk variance from producing a
        division-by-zero or negative score term.

        Args:
            reference: `(N, F)` finite matrix of normal reference windows.

        Returns:
            self.

        Raises:
            ValueError: if `reference` is empty or has a non-finite value.
        """
        _check_reference(reference)
        variance = reference.var(axis=0)
        var_shrunk = (1.0 - self.shrinkage) * variance + self.shrinkage * variance.mean()
        self._mean = reference.mean(axis=0)
        self._var_shrunk = np.maximum(var_shrunk, _VAR_FLOOR)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """`(W, F)` windows -> `(W,)` float64 scores, higher = more anomalous:
        `sqrt(sum((x - mean) ** 2 / var_shrunk, axis=1))`.

        Raises:
            ValueError: if called before `fit()`, or if `x` contains non-finite values.
        """
        _check_query(x)
        mean = self._mean
        var_shrunk = self._var_shrunk
        if mean is None or var_shrunk is None:
            raise ValueError("MahalanobisScorer.score() called before fit()")
        diff_sq = np.square(x - mean)
        return np.sqrt(np.sum(diff_sq / var_shrunk, axis=1))
