"""TfcFeaturizer contract tests (package-4 spec D1/D4): stub encoder, no torch.

Stub-encoder tests inject a deterministic `TfcEncoderProtocol` stub via
`TfcFeaturizer`'s `encoder=` parameter, so they exercise the mono-mix/resample/
shape wiring without needing torch or a real checkpoint at all -- this whole
module is import-time torch-free (no `pytest.importorskip("torch")` at module
scope), unlike `tests/test_tfc_model.py`. The one exception is the checkpoint
round-trip test at the bottom, which needs a real `TfcModel`/torch and guards
itself with a function-scoped `pytest.importorskip("torch")` instead of
gating the whole file.
"""
from __future__ import annotations

import numpy as np
import pytest

from rowii.tfc.wrapper import TfcConfig, TfcFeaturizer


class _StubEncoder:
    def __init__(self):
        self.calls = []

    def embed(self, batch_8khz: np.ndarray) -> np.ndarray:
        self.calls.append(batch_8khz.shape)
        return np.tile(batch_8khz.mean(axis=1, keepdims=True), (1, 256))


class TestTfcFeaturizer:
    def test_transform_2d_shape_names_dtype(self):
        f = TfcFeaturizer(checkpoint=None, encoder=_StubEncoder())
        rng = np.random.default_rng(0)
        out = f.transform(rng.normal(0, 1, (3, 50_000)), 50_000.0)
        assert out.shape == (3, 256) and out.dtype == np.float64
        names = f.feature_names()
        assert names[0] == "tfc_e0" and names[-1] == "tfc_e255" and len(names) == 256

    def test_transform_3d_mono_mix(self):
        stub = _StubEncoder()
        f = TfcFeaturizer(checkpoint=None, encoder=stub)
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, (2, 10_000, 3))
        out = f.transform(x, 10_000.0)
        assert out.shape == (2, 256)
        assert stub.calls[0] == (2, 8000)  # resampled to 8 kHz

    def test_resample_normalizes_length(self):
        stub = _StubEncoder()
        f = TfcFeaturizer(checkpoint=None, encoder=stub)
        x = np.random.default_rng(2).normal(0, 1, (1, 50_004))  # jittered window
        f.transform(x, 50_000.0)
        assert stub.calls[0] == (1, 8000)

    def test_module_imports_without_torch(self):
        # wrapper.py itself must not import torch at module level; this test file
        # already imported it torch-free above. Assert the real-encoder path guards:
        f = TfcFeaturizer(checkpoint=None, encoder=None)
        with pytest.raises((RuntimeError, ValueError)):
            f.transform(np.zeros((1, 8000)), 8000.0)  # no checkpoint, no stub


# ---------------------------------------------------------------------------
# Checkpoint round-trip (real TfcModel + torch): build a tiny model, save it
# in the exact dict format the (later, not-yet-written) pretrain_tfc.py
# script will write, reload it via load_tfc_model, and drive it through
# TfcFeaturizer end to end. Function-scoped importorskip keeps the rest of
# this file torch-free at collection time even when torch is unavailable.
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_transform_is_deterministic(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import dataclasses

    from rowii.tfc.model import TfcModel
    from rowii.tfc.wrapper import load_tfc_model

    # CPU-forced like tests/test_tfc_model.py's autouse fixture -- TfcFeaturizer
    # has no device= override, so this keeps its internal best_device() pick
    # (and this test's runtime) predictable on MPS/CUDA dev machines too.
    monkeypatch.setenv("ROWII_FORCE_CPU", "1")

    cfg = TfcConfig(channels=(4, 8))  # tiny CNN body; embed_dim stays default (128)
    model = TfcModel(cfg)
    checkpoint_path = tmp_path / "tfc_checkpoint.pt"
    torch.save(
        {
            "cfg": dataclasses.asdict(cfg),
            "model": model.state_dict(),
            "corpus_manifest_sha256": "test",
            "epochs": 1,
        },
        checkpoint_path,
    )

    loaded = load_tfc_model(checkpoint_path, torch.device("cpu"))
    assert isinstance(loaded, TfcModel)
    assert loaded.training is False  # load_tfc_model must leave it in eval mode

    featurizer = TfcFeaturizer(checkpoint=checkpoint_path)
    windows = np.random.default_rng(3).normal(0, 1, (2, 8000))

    out1 = featurizer.transform(windows, 8000.0)
    out2 = featurizer.transform(windows, 8000.0)

    assert out1.shape == (2, 256) and out1.dtype == np.float64
    assert np.isfinite(out1).all()
    np.testing.assert_array_equal(out1, out2)
