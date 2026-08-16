"""TF-C model/loss unit tests. CPU-forced, seeded."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rowii.tfc.model import TfcModel, freq_view, tfc_loss  # noqa: E402
from rowii.tfc.wrapper import TfcConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setenv("ROWII_FORCE_CPU", "1")


def _batch(b=8, n=8000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, generator=g)


class TestTfcModel:
    def test_forward_shapes(self):
        cfg = TfcConfig()
        m = TfcModel(cfg)
        x = _batch()
        h_t, h_f, z_t, z_f = m(x, freq_view(x))
        assert h_t.shape == (8, 128) and h_f.shape == (8, 128)
        assert z_t.shape == (8, 64) and z_f.shape == (8, 64)

    def test_loss_decreases_with_training(self):
        torch.manual_seed(7)
        cfg = TfcConfig(channels=(8, 16), embed_dim=32, proj_dim=16)
        m = TfcModel(cfg)
        x = _batch(b=16, n=8000, seed=1)
        xf = freq_view(x)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        first = tfc_loss(*m(x, xf)[2:], cfg.temperature).item()
        for _ in range(30):
            opt.zero_grad()
            loss = tfc_loss(*m(x, xf)[2:], cfg.temperature)
            loss.backward()
            opt.step()
        last = tfc_loss(*m(x, xf)[2:], cfg.temperature).item()
        assert last < first

    def test_loss_prefers_aligned_pairs(self):
        # perfectly aligned projections -> lower loss than shuffled pairing
        torch.manual_seed(0)
        z = torch.nn.functional.normalize(torch.randn(16, 64), dim=1)
        aligned = tfc_loss(z, z, 0.2).item()
        perm = z[torch.randperm(16, generator=torch.Generator().manual_seed(1))]
        shuffled = tfc_loss(z, perm, 0.2).item()
        assert aligned < shuffled

    def test_freq_view_shape_and_determinism(self):
        x = _batch(b=3)
        f1, f2 = freq_view(x), freq_view(x)
        assert f1.shape == (3, 4001)
        assert torch.equal(f1, f2)
