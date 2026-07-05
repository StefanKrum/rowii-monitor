import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class KMeansClusterer:
    """K-means clustering with fixed random seed for reproducibility."""

    def __init__(self, n_clusters: int, random_seed: int) -> None:
        """Initialize KMeans clusterer.

        Args:
            n_clusters: Number of clusters.
            random_seed: Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self._kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_seed,
            n_init=10,
        )

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        """Fit KMeans and return cluster labels.

        Args:
            x: Input data, shape (W, F).

        Returns:
            Cluster labels, shape (W,), dtype int64.
        """
        labels = self._kmeans.fit_predict(x)
        return np.asarray(labels, dtype=np.int64)


class GmmClusterer:
    """Gaussian Mixture Model clustering with full covariance."""

    def __init__(self, n_clusters: int, random_seed: int) -> None:
        """Initialize GMM clusterer.

        Args:
            n_clusters: Number of components.
            random_seed: Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self._gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=random_seed,
        )

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        """Fit GMM and return cluster labels.

        Args:
            x: Input data, shape (W, F).

        Returns:
            Cluster labels, shape (W,), dtype int64.
        """
        labels = self._gmm.fit_predict(x)
        return np.asarray(labels, dtype=np.int64)
