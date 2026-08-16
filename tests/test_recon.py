"""Reconstruction-scorer tests. CPU-forced, seeded."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rowii.anomaly._recon_models import _ConvAe  # noqa: E402
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
# score() before fit() -> ValueError, wording-consistent with
# rowii.anomaly.scorers' own precondition errors ("<Class>.score() called
# before fit()", e.g. KnnScorer.score) -- the first
# cut raised AssertionError here; one convention across the Scorer family now.
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
    def test_raises_value_error(self, factory):
        with pytest.raises(ValueError, match="called before fit"):
            factory().score(np.zeros((5, 56)))


# ---------------------------------------------------------------------------
# Conv-AE decoder hits the input patch shape EXACTLY via closed-form
# output_padding (replaces the earlier
# interpolate-based resize) -- asserted on the RAW decoder output, layer by
# layer, so no resampling step could mask a mismatch. Covers both the test
# geometry (8 mels x 7 frames) and the real logmel geometry (64 mels x 49
# frames).
# ---------------------------------------------------------------------------


class TestConvAeDecoderShape:
    @pytest.mark.parametrize("mels,frames", [(8, 7), (64, 49)])
    def test_raw_decoder_output_matches_input_patch_shape(self, mels, frames):
        model = _ConvAe(n_frames=frames, n_mels=mels, channels=(4, 8))
        patch = torch.zeros(2, 1, mels, frames)

        h = model.relu(model.enc1(patch))
        h = model.relu(model.enc2(h))
        h = model.relu(model.dec1(h))
        raw = model.dec2(h)

        assert raw.shape == (2, 1, mels, frames)
