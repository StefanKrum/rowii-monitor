"""Reconstruction-scorer tests (package-3 spec D2). CPU-forced, seeded."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rowii.anomaly.recon import ConvAeScorer, LstmAeScorer, MlpAeScorer  # noqa: E402


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setenv("ROWII_FORCE_CPU", "1")


def _vector_data(seed=0, f=32):
    rng = np.random.default_rng(seed)
    reference = rng.normal(0.0, 0.1, (400, f))
    inliers = rng.normal(0.0, 0.1, (40, f))
    outliers = rng.normal(3.0, 0.1, (10, f))
    return reference, inliers, outliers


def _patch_data(seed=0, frames=7, mels=8):
    rng = np.random.default_rng(seed)
    f = frames * mels
    reference = rng.normal(0.0, 0.1, (300, f))
    inliers = rng.normal(0.0, 0.1, (30, f))
    outliers = rng.normal(3.0, 0.1, (10, f))
    return reference, inliers, outliers


class TestMlpAe:
    def test_outliers_reconstruct_worse(self):
        reference, inliers, outliers = _vector_data()
        s = MlpAeScorer(hidden=(16, 4), epochs=60, seed=7).fit(reference)
        assert s.score(outliers).min() > s.score(inliers).max()

    def test_deterministic_given_seed(self):
        reference, inliers, _ = _vector_data()
        a = MlpAeScorer(hidden=(16, 4), epochs=10, seed=7).fit(reference).score(inliers)
        b = MlpAeScorer(hidden=(16, 4), epochs=10, seed=7).fit(reference).score(inliers)
        np.testing.assert_allclose(a, b)


class TestPatchAes:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: LstmAeScorer(hidden=8, epochs=40, seed=7, n_mels=8),
            lambda: ConvAeScorer(channels=(4, 8), epochs=40, seed=7, n_mels=8),
        ],
    )
    def test_outliers_reconstruct_worse(self, factory):
        reference, inliers, outliers = _patch_data()
        s = factory().fit(reference)
        assert s.score(outliers).min() > s.score(inliers).max()

    def test_non_divisible_width_rejected(self):
        with pytest.raises(ValueError, match="n_mels"):
            LstmAeScorer(n_mels=8).fit(np.zeros((30, 30)))


# ---------------------------------------------------------------------------
# score() before fit() (orchestrator resolution 6): AssertionError, matching
# this module's own `assert self._model is not None` guard (see module
# docstring for why this scorer family diverges from `rowii.anomaly.scorers`'
# ValueError convention here).
# ---------------------------------------------------------------------------


class TestScoreBeforeFit:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: MlpAeScorer(),
            lambda: LstmAeScorer(n_mels=8),
            lambda: ConvAeScorer(n_mels=8),
        ],
    )
    def test_raises_assertion_error(self, factory):
        with pytest.raises(AssertionError):
            factory().score(np.zeros((5, 56)))
