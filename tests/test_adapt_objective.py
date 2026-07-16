"""Masked-reconstruction objective tests (Step-2 package-5 spec D1 as amended
by Amendment A1, Task 1 + Task 3 rework): the frame-level `masked_patch_loss`
(retained for position-preserving encoders) and the native token-level
`masked_token_loss` (the BEATs adaptation objective). CPU-forced, eager-torch
target module -- same `pytest.importorskip("torch")`-at-module-scope
convention as `tests/test_adapt_lora.py` / `tests/test_recon.py`.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rowii.adapt.objective import masked_patch_loss, masked_token_loss  # noqa: E402


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


# ---------------------------------------------------------------------------
# masked_token_loss (spec D1 as amended by Amendment A1, Task-3 rework): the
# native token-level latent-target MAE actually used for BEATs adaptation.
# ---------------------------------------------------------------------------


def test_token_loss_masks_token_rows_only():
    torch.manual_seed(0)
    tokens = torch.randn(2, 24, 12)  # (B, T, D) native pre-encoder tokens
    head = torch.nn.Linear(12, 12)
    calls = {}

    def encoder(x):
        calls["input"] = x.detach().clone()
        return x  # identity encoder, dim-preserving

    gen = torch.Generator().manual_seed(7)
    loss = masked_token_loss(tokens, encoder, head, mask_frac=0.3, generator=gen)

    assert loss.ndim == 0 and torch.isfinite(loss)
    masked = (calls["input"] == 0).all(dim=2)  # zeroed token rows
    frac = masked.float().mean().item()
    assert 0.15 < frac < 0.45  # ~0.3 of token rows masked
    # the caller's tokens tensor is never mutated (masking works on a clone)
    assert not (tokens == 0).all(dim=2).any()


def test_token_loss_decreases_with_training():
    torch.manual_seed(0)
    tokens = torch.randn(8, 24, 12)
    enc = torch.nn.Linear(12, 12)
    head = torch.nn.Linear(12, 12)
    opt = torch.optim.Adam([*enc.parameters(), *head.parameters()], lr=1e-2)
    gen = torch.Generator().manual_seed(7)
    first = masked_token_loss(tokens, lambda x: enc(x), head, generator=gen).item()
    for _ in range(50):
        g = torch.Generator().manual_seed(7)
        opt.zero_grad()
        loss = masked_token_loss(tokens, lambda x: enc(x), head, generator=g)
        loss.backward()
        opt.step()
    g = torch.Generator().manual_seed(7)
    assert masked_token_loss(tokens, lambda x: enc(x), head, generator=g).item() < first


def test_token_target_is_detached_masked_rows_get_no_gradient():
    """The stop-gradient contract (masked_token_loss's docstring): the
    reconstruction TARGET (`tokens[mask]`) is detached, so a grad-requiring
    `tokens` (the `--mode full` case, where the token construction is
    trainable) receives gradient ONLY through the legitimate input path --
    the unmasked rows the encoder mixes into its predictions -- never
    through the target side. With a position-MIXING encoder (each row plus
    the per-sample row mean), the masked positions' predictions genuinely
    depend on the unmasked rows, so: unmasked rows get NONZERO gradient
    (input path alive), while masked rows get EXACTLY ZERO gradient (their
    input-path contribution is overwritten by the zeroing write; their
    target-path contribution is detached). Without the detach, every masked
    row would receive the MSE target gradient -2/N * (pred - target) -- a
    collapse-enabling path this test would catch immediately.
    """
    torch.manual_seed(0)
    tokens = torch.randn(2, 24, 12, requires_grad=True)
    head = torch.nn.Linear(12, 12)
    calls = {}

    def mixing_encoder(x):
        calls["input"] = x.detach().clone()
        return x + x.mean(dim=1, keepdim=True)  # masked rows see unmasked rows

    gen = torch.Generator().manual_seed(7)
    loss = masked_token_loss(tokens, mixing_encoder, head, generator=gen)
    loss.backward()

    mask = (calls["input"] == 0).all(dim=2)  # which rows were zeroed
    assert mask.any() and (~mask).any()
    assert tokens.grad is not None
    masked_grads = tokens.grad[mask]
    unmasked_grads = tokens.grad[~mask]
    assert torch.all(masked_grads == 0), "target must be detached (and zeroing must cut input path)"
    assert torch.any(unmasked_grads != 0), "input path through unmasked rows must stay alive"
