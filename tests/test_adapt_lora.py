"""LoRA injection/merge tests (Step-2 package-5 spec D2, Task 1). CPU-forced,
eager-torch target module -- mirrors `tests/test_recon.py`'s
`pytest.importorskip("torch")`-at-module-scope convention (this repo's core
dependencies do not include torch; it is opt-in via the `[beats]` extra, and
`rowii.adapt.lora` is declared an EAGER-torch module by the package-5 plan's
Global Constraints, so there is no torch-free import path worth testing here,
unlike `rowii.tfc.wrapper`).
"""
from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")

from rowii.adapt.lora import LoraLinear, inject_lora, lora_parameters, merge_lora  # noqa: E402


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setenv("ROWII_FORCE_CPU", "1")


class _TinyAttnModel(torch.nn.Module):
    """Stand-in mirroring the vendored naming shape: layers[i].self_attn.{q,k,v}_proj."""

    def __init__(self):
        super().__init__()
        attn = torch.nn.Module()
        attn.q_proj = torch.nn.Linear(8, 8)
        attn.k_proj = torch.nn.Linear(8, 8)
        attn.v_proj = torch.nn.Linear(8, 8)
        layer = torch.nn.Module()
        layer.self_attn = attn
        self.layers = torch.nn.ModuleList([layer])
        self.unrelated = torch.nn.Linear(8, 8)

    def forward(self, x):
        a = self.layers[0].self_attn
        return a.q_proj(x) + a.v_proj(x) + a.k_proj(x) + self.unrelated(x)


def test_inject_targets_only_qv_under_self_attn():
    m = _TinyAttnModel()
    n = inject_lora(m, r=2)
    assert n == 2  # q_proj + v_proj; k_proj and unrelated untouched
    assert isinstance(m.layers[0].self_attn.q_proj, LoraLinear)
    assert isinstance(m.layers[0].self_attn.k_proj, torch.nn.Linear)
    assert isinstance(m.unrelated, torch.nn.Linear)


def test_injection_starts_as_identity():
    torch.manual_seed(0)
    m = _TinyAttnModel()
    x = torch.randn(4, 8)
    before = m(x).detach().clone()
    inject_lora(m, r=2)
    torch.testing.assert_close(m(x), before)  # B init zeros


def test_base_frozen_adapters_trainable():
    m = _TinyAttnModel()
    inject_lora(m, r=2)
    q = m.layers[0].self_attn.q_proj
    assert not q.base.weight.requires_grad
    lora_named = {id(p) for p in lora_parameters(m)}
    assert id(q.lora_a.weight) in lora_named and id(q.lora_b.weight) in lora_named


def test_merge_restores_plain_linear_and_forward():
    torch.manual_seed(1)
    m = _TinyAttnModel()
    inject_lora(m, r=2)
    # push adapters off zero
    for p in lora_parameters(m):
        torch.nn.init.normal_(p, std=0.1)
    x = torch.randn(4, 8)
    unmerged = m(x).detach().clone()
    n = merge_lora(m)
    assert n == 2
    assert isinstance(m.layers[0].self_attn.q_proj, torch.nn.Linear)
    torch.testing.assert_close(m(x), unmerged, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Structural match against the REAL vendored BEATs class (not the hand-rolled
# stand-in above): constructs `rowii.vendor.beats.BEATs.BEATs` with a tiny
# config (encoder_layers=2, small embed/ffn/head dims, default 16-sample
# patch size) -- module CONSTRUCTION only, no checkpoint load and no forward
# pass (nothing in `BEATs.__init__`/`TransformerEncoder.__init__` runs a
# forward pass; `nn.Conv2d`/`nn.Conv1d`/`nn.Linear` construction never
# validates a downstream sequence length), so this needs neither a real
# checkpoint nor `@pytest.mark.data`. `q_noise` defaults to 0.0 in the
# vendored `MultiheadAttention.__init__` and is never overridden by
# `TransformerEncoder`/`BEATsConfig`, so `modules.quant_noise(..., 0.0, ...)`
# always returns its input module unchanged -- q/k/v/out_proj are plain
# `nn.Linear` at ANY config, tiny or real. This is what proves
# `inject_lora`'s generic `named_modules()`-path walk actually lines up with
# the REAL vendored attribute names/nesting (`encoder.layers[i].self_attn.
# {q,k,v,out}_proj`), not just the hand-rolled stand-in's shape above.
# ---------------------------------------------------------------------------


def test_structural_match_on_real_vendored_class():
    from rowii.vendor.beats.BEATs import BEATs, BEATsConfig

    cfg = BEATsConfig(
        {
            "input_patch_size": 16,
            "embed_dim": 32,
            "encoder_layers": 2,
            "encoder_embed_dim": 32,
            "encoder_ffn_embed_dim": 64,
            "encoder_attention_heads": 4,
        }
    )

    start = time.monotonic()
    model = BEATs(cfg)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, (
        f"tiny BEATs() construction took {elapsed:.2f}s (budget 2s; no checkpoint "
        "weights are loaded here, so this should be near-instant)"
    )

    n = inject_lora(model)

    assert n == 2 * cfg.encoder_layers
    for layer in model.encoder.layers:
        assert isinstance(layer.self_attn.q_proj, LoraLinear)
        assert isinstance(layer.self_attn.v_proj, LoraLinear)
        assert isinstance(layer.self_attn.k_proj, torch.nn.Linear)
        assert isinstance(layer.self_attn.out_proj, torch.nn.Linear)
