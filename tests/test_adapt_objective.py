"""Masked-patch reconstruction objective tests (Step-2 package-5 spec D1,
Task 1). CPU-forced, eager-torch target module -- same
`pytest.importorskip("torch")`-at-module-scope convention as
`tests/test_adapt_lora.py` / `tests/test_recon.py`.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rowii.adapt.objective import masked_patch_loss  # noqa: E402


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setenv("ROWII_FORCE_CPU", "1")


def test_loss_on_masked_frames_only():
    torch.manual_seed(0)
    fbank = torch.randn(2, 20, 16)  # (B, frames, mels)
    head = torch.nn.Linear(16, 16)
    calls = {}

    def encoder(x):
        calls["input"] = x.detach().clone()
        return x  # identity encoder, dim-preserving

    gen = torch.Generator().manual_seed(7)
    loss = masked_patch_loss(encoder, fbank, head, mask_frac=0.3, generator=gen)
    assert loss.ndim == 0 and torch.isfinite(loss)
    masked = (calls["input"] == 0).all(dim=2)  # zeroed frames
    frac = masked.float().mean().item()
    assert 0.15 < frac < 0.45  # ~0.3 of frames masked


def test_loss_decreases_with_training():
    torch.manual_seed(0)
    fbank = torch.randn(8, 20, 16)
    enc = torch.nn.Linear(16, 16)
    head = torch.nn.Linear(16, 16)
    opt = torch.optim.Adam([*enc.parameters(), *head.parameters()], lr=1e-2)
    gen = torch.Generator().manual_seed(7)
    first = masked_patch_loss(lambda x: enc(x), fbank, head, generator=gen).item()
    for _ in range(50):
        g = torch.Generator().manual_seed(7)
        opt.zero_grad()
        loss = masked_patch_loss(lambda x: enc(x), fbank, head, generator=g)
        loss.backward()
        opt.step()
    g = torch.Generator().manual_seed(7)
    assert masked_patch_loss(lambda x: enc(x), fbank, head, generator=g).item() < first
