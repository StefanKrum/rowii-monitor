import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from rowii.state.cluster import GmmClusterer, KMeansClusterer


class TestKMeansClusterer:
    def test_kmeans_recovers_blobs(self):
        """KMeans recovers 3 separated Gaussian blobs with ARI == 1.0."""
        rng = np.random.default_rng(0)

        # Three well-separated Gaussian blobs
        blob1 = rng.normal(loc=(0, 0), scale=0.5, size=(50, 2))
        blob2 = rng.normal(loc=(10, 10), scale=0.5, size=(50, 2))
        blob3 = rng.normal(loc=(-10, 10), scale=0.5, size=(50, 2))

        x = np.vstack([blob1, blob2, blob3])
        true_labels = np.array([0] * 50 + [1] * 50 + [2] * 50)

        clusterer = KMeansClusterer(n_clusters=3, random_seed=0)
        pred_labels = clusterer.fit_predict(x)

        ari = adjusted_rand_score(true_labels, pred_labels)
        assert ari == 1.0, f"Expected ARI=1.0, got {ari}"

    def test_kmeans_deterministic(self):
        """Same seed produces identical labels across two calls."""
        rng = np.random.default_rng(42)
        blob1 = rng.normal(loc=(0, 0), scale=0.5, size=(50, 2))
        blob2 = rng.normal(loc=(10, 10), scale=0.5, size=(50, 2))
        blob3 = rng.normal(loc=(-10, 10), scale=0.5, size=(50, 2))
        x = np.vstack([blob1, blob2, blob3])

        clusterer1 = KMeansClusterer(n_clusters=3, random_seed=42)
        labels1 = clusterer1.fit_predict(x)

        clusterer2 = KMeansClusterer(n_clusters=3, random_seed=42)
        labels2 = clusterer2.fit_predict(x)

        np.testing.assert_array_equal(labels1, labels2)


class TestGmmClusterer:
    def test_gmm_recovers_blobs(self):
        """GMM recovers 3 separated Gaussian blobs with ARI == 1.0."""
        rng = np.random.default_rng(0)

        # Three well-separated Gaussian blobs
        blob1 = rng.normal(loc=(0, 0), scale=0.5, size=(50, 2))
        blob2 = rng.normal(loc=(10, 10), scale=0.5, size=(50, 2))
        blob3 = rng.normal(loc=(-10, 10), scale=0.5, size=(50, 2))

        x = np.vstack([blob1, blob2, blob3])
        true_labels = np.array([0] * 50 + [1] * 50 + [2] * 50)

        clusterer = GmmClusterer(n_clusters=3, random_seed=0)
        pred_labels = clusterer.fit_predict(x)

        ari = adjusted_rand_score(true_labels, pred_labels)
        assert ari == 1.0, f"Expected ARI=1.0, got {ari}"


class TestPredict:
    def test_kmeans_predict_matches_fit_predict_on_same_data(self):
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(0, 0.1, (50, 3)), rng.normal(5, 0.1, (50, 3))])
        c = KMeansClusterer(n_clusters=2, random_seed=7)
        fit_labels = c.fit_predict(x)
        np.testing.assert_array_equal(c.predict(x), fit_labels)
        assert c.predict(x).dtype == np.int64

    def test_kmeans_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit_predict"):
            KMeansClusterer(n_clusters=2, random_seed=7).predict(np.zeros((3, 2)))

    def test_gmm_predict_new_points_assigned_to_nearest_component(self):
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(0, 0.1, (50, 2)), rng.normal(5, 0.1, (50, 2))])
        c = GmmClusterer(n_clusters=2, random_seed=7)
        fit_labels = c.fit_predict(x)
        label_at_origin = fit_labels[0]
        assert c.predict(np.array([[0.05, -0.05]]))[0] == label_at_origin

    def test_gmm_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit_predict"):
            GmmClusterer(n_clusters=2, random_seed=7).predict(np.zeros((3, 2)))
