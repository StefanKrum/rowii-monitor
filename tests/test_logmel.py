"""LogmelFeaturizer unit tests (package-3 spec D3)."""
from __future__ import annotations

import numpy as np

from rowii.signals.logmel import LogmelFeaturizer


class TestLogmelFeaturizer:
    def test_shape_and_names_at_50khz(self):
        rng = np.random.default_rng(0)
        stack = rng.normal(0, 0.1, (3, 50_000))
        f = LogmelFeaturizer()
        out = f.transform(stack, 50_000.0)
        assert out.shape == (3, 49 * 64)
        assert out.dtype == np.float64
        names = f.feature_names()
        assert len(names) == 49 * 64
        assert names[0] == "logmel_f0_m0" and names[64] == "logmel_f1_m0"

    def test_tone_lands_in_matching_mel_band(self):
        t = np.arange(50_000) / 50_000.0
        tone = np.sin(2 * np.pi * 1000.0 * t)[None, :]
        f = LogmelFeaturizer()
        out = f.transform(tone, 50_000.0).reshape(49, 64)
        band_energy = out.mean(axis=0)
        # the 1 kHz band must dominate quiet bands far away in mel space
        assert band_energy.argmax() not in (0, 63)
        assert band_energy.max() > band_energy.min() + 2.0  # log10 domain

    def test_deterministic(self):
        rng = np.random.default_rng(1)
        stack = rng.normal(0, 0.1, (2, 50_000))
        f = LogmelFeaturizer()
        np.testing.assert_array_equal(
            f.transform(stack, 50_000.0), f.transform(stack, 50_000.0)
        )

    def test_feature_names_stable_before_and_after_transform_at_default_geometry(self):
        """`feature_names()` on a FRESH instance (default 50 kHz / 1 s geometry
        fallback) must equal `feature_names()` after a real transform at that same
        geometry -- the cache/`_extract_stream_features` path calls it after
        transform, but nothing may depend on call order for the plant's own
        geometry (orchestrator resolution 2)."""
        fresh_names = LogmelFeaturizer().feature_names()

        f = LogmelFeaturizer()
        rng = np.random.default_rng(2)
        f.transform(rng.normal(0, 0.1, (1, 50_000)), 50_000.0)

        assert f.feature_names() == fresh_names
        assert len(fresh_names) == 49 * 64

    def test_accepts_pipeline_3d_stack_and_mono_mixes_channels(self):
        """The pipeline's `_extract_stream_features` hands featurizers a 3-D
        `(B, S, C)` float32 stack (see `AudioFeaturizer`/`BeatsFeaturizer`) --
        a 3-D input must be mono-mixed over channels (mean), mirroring
        `BeatsFeaturizer.transform`, and give the exact same result as the
        equivalent already-mono 2-D input."""
        rng = np.random.default_rng(3)
        mono = rng.normal(0, 0.1, (2, 50_000))
        stacked = np.stack([mono, mono], axis=2)  # (2, 50_000, 2), identical channels

        f = LogmelFeaturizer()
        out_3d = f.transform(stacked.astype(np.float32), 50_000.0)
        out_2d = f.transform(mono.astype(np.float32), 50_000.0)

        assert out_3d.shape == (2, 49 * 64)
        assert out_3d.dtype == np.float64
        np.testing.assert_array_equal(out_3d, out_2d)
