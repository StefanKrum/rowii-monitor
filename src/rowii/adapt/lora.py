"""LoRA (Low-Rank Adaptation, Hu et al. 2021) injection for the vendored
BEATs encoder (Step-2 package-5 spec D2, Task 1): the ONE module under
`rowii.adapt` (alongside `objective.py`) that imports torch at module level,
mirroring `rowii.anomaly._recon_models`'s / `rowii.tfc.model`'s role in their
own packages -- adaptation only ever runs with the `[beats]` extra installed
(there is no "adapt a model that doesn't exist" torch-free code path worth
guarding here, unlike `rowii.tfc.wrapper`/`rowii.signals.beats`), so this
module is eager rather than lazy-behind-a-wrapper.

Design (spec D2): rank-8, alpha-16, dropout-0.0 low-rank adapters are
injected into the QUERY and VALUE projections of every self-attention block
of the vendored BEATs encoder (`rowii.vendor.beats.backbone.
TransformerSentenceEncoderLayer.self_attn.{q,v}_proj`, both plain
`nn.Linear`s at any config -- see `tests/test_adapt_lora.py::
test_structural_match_on_real_vendored_class`'s docstring for why) --
the literature-cited placement for transformer LoRA. `dropout=0.0` per spec
means "no dropout", so `LoraLinear` carries no dropout layer at all rather
than a `nn.Dropout(0.0)` no-op module.

Injection wraps the EXISTING `nn.Linear` objects by attribute replacement
(`inject_lora`) -- no vendored-code edits, no new class hierarchy for BEATs
itself. `merge_lora` folds the adapters back into the wrapped base Linears so
an adapted encoder is a checkpoint-format-compatible, ordinary BEATs module
again: it reloads through the EXISTING `rowii.signals.beats_model.
load_beats_model` path ({"cfg", "model"} state dict), so no new featurizer
code is needed downstream just to score with an adapted model.
"""
from __future__ import annotations

import math
from collections.abc import Iterator

import torch

_DEFAULT_TARGET_NAMES = ("q_proj", "v_proj")
"""Attribute names LoRA targets by default (spec D2: query and value
projections only -- key/output projections are left frozen, matching the
literature-cited placement for transformer LoRA)."""


class LoraLinear(torch.nn.Module):
    """Wraps one frozen `base` `nn.Linear` with a trainable low-rank
    residual: `forward(x) = base(x) + scale * lora_b(lora_a(x))`, `scale =
    alpha / r`.

    `lora_a` (`Linear(in_features, r, bias=False)`) is kaiming-uniform
    initialized -- the same distribution `nn.Linear`'s own default
    `reset_parameters` uses, made explicit here rather than relied upon,
    matching the reference LoRA implementation's own initialization choice.
    `lora_b` (`Linear(r, out_features, bias=False)`) is ZERO initialized, so
    `lora_b(lora_a(x)) == 0` identically at construction time regardless of
    `lora_a`'s weights -- injecting a fresh `LoraLinear` never changes a
    model's forward output (`tests/test_adapt_lora.py::
    test_injection_starts_as_identity`), which is what lets `inject_lora` be
    called on an already-trained/frozen model with no forward-pass side
    effects until the adapters are actually trained.

    `base`'s parameters (weight AND bias, if present) are frozen
    (`requires_grad_(False)`) in `__init__` -- only `lora_a`/`lora_b` are
    meant to train (`lora_parameters` yields exactly these two Linears'
    parameters and nothing else, including nothing from `base`).
    """

    def __init__(self, base: torch.nn.Linear, r: int = 8, alpha: int = 16) -> None:
        super().__init__()
        base.requires_grad_(False)
        self.base = base
        self.lora_a = torch.nn.Linear(base.in_features, r, bias=False)
        self.lora_b = torch.nn.Linear(r, base.out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b.weight)
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out: torch.Tensor = self.base(x)
        residual: torch.Tensor = self.lora_b(self.lora_a(x))
        return base_out + self.scale * residual


def _parent_path_and_attr(dotted_name: str) -> tuple[str, str]:
    """Splits a `named_modules()`-style dotted path (e.g.
    `"encoder.layers.0.self_attn.q_proj"`) into its parent's qualified path
    string (`"encoder.layers.0.self_attn"`) and the final attribute name
    (`"q_proj"`). A name with no dot (a direct top-level attribute) has an
    empty parent path.
    """
    parent_path, sep, attr_name = dotted_name.rpartition(".")
    return (parent_path if sep else ""), (attr_name if sep else dotted_name)


def _resolve_parent(root: torch.nn.Module, parent_path: str) -> torch.nn.Module:
    """Walks *parent_path* (a dot-joined string as produced by
    `_parent_path_and_attr`, e.g. `"encoder.layers.0.self_attn"`) from
    *root*, returning the submodule the final attribute in the original
    dotted name lives on. Empty *parent_path* returns *root* itself.

    Plain `getattr` suffices for EVERY path component, including numeric
    `ModuleList`/`Sequential` indices like `"0"`: `nn.Module.__setattr__`
    registers child modules into `self._modules` (an `OrderedDict` keyed by
    name, never `self.__dict__`) precisely so that `nn.Module.__getattr__`'s
    fallback -- triggered whenever normal attribute lookup misses, which it
    always does for submodules -- can resolve them by that same string key,
    whether the key is an identifier like `"self_attn"` or a stringified
    index like `"0"` (`ModuleList.append` internally calls
    `add_module(str(len(self)), module)`). This is also, structurally, the
    exact inverse of how `named_modules()` builds its dotted names in the
    first place (it walks `self._modules.items()` recursively, joining keys
    with `.`), so walking the same keys back via `getattr` is guaranteed to
    reach the same submodule `named_modules()` reported.
    """
    obj = root
    if not parent_path:
        return obj
    for part in parent_path.split("."):
        obj = getattr(obj, part)
    return obj


def inject_lora(
    module: torch.nn.Module,
    r: int = 8,
    alpha: int = 16,
    target_names: tuple[str, ...] = _DEFAULT_TARGET_NAMES,
) -> int:
    """Replaces every `nn.Linear` in *module* whose attribute name is in
    *target_names* AND whose parent's qualified `named_modules()` path
    contains `"self_attn"` as a whole PATH COMPONENT with a `LoraLinear`
    wrapping it (same *r*/*alpha* for every replacement), in place.

    The `"self_attn"` check is COMPONENT-EXACT: the parent's dot-joined
    path is split on `"."` and one whole segment must equal `"self_attn"`
    (`"encoder.layers.0.self_attn"` matches; a top-level attribute's parent
    path is `""`, whose only "segment" is `""`, so it never does). A bare
    substring test against the joined path would over-match parents whose
    names merely CONTAIN `"self_attn"` -- not a hypothetical: the vendored
    `TransformerSentenceEncoderLayer` itself carries a sibling module named
    `self_attn_layer_norm` (a `LayerNorm`, hence no Linear children on the
    real model, which is the only reason a substring test happens not to
    misfire there today; any structure hanging a `q_proj` Linear under such
    a name would be silently injected). `tests/test_adapt_lora.py::
    test_inject_skips_substring_decoy_parents` pins the component-exact
    semantics against exactly those decoys (`self_attn_layer_norm` and
    `not_self_attn_thing`, both carrying a `q_proj` Linear child that must
    stay untouched).

    Matching targets are collected into a list FIRST (`list(module.
    named_modules())`, fully materialized) and only replaced (via `setattr`
    on each resolved parent) in a second pass -- `named_modules()` is a
    generator that walks `_modules` dicts live, and mutating a module's
    attributes (which rewrites its `_modules` dict) while that same walk is
    still in progress is unsafe.

    Args:
        module: Root module to walk (e.g. a whole `BEATs` instance, or just
            its `.encoder`).
        r: LoRA rank (spec D2 default: 8).
        alpha: LoRA scale numerator (spec D2 default: 16); `LoraLinear.scale
            = alpha / r`.
        target_names: Attribute names to target (spec D2 default: query and
            value projections only).

    Returns:
        Count of `nn.Linear`s replaced.
    """
    targets: list[tuple[torch.nn.Module, str, torch.nn.Linear]] = []
    for name, child in list(module.named_modules()):
        if not isinstance(child, torch.nn.Linear):
            continue
        parent_path, attr_name = _parent_path_and_attr(name)
        if attr_name in target_names and "self_attn" in parent_path.split("."):
            parent = _resolve_parent(module, parent_path)
            targets.append((parent, attr_name, child))

    for parent, attr_name, linear in targets:
        setattr(parent, attr_name, LoraLinear(linear, r=r, alpha=alpha))

    return len(targets)


def merge_lora(module: torch.nn.Module) -> int:
    """Folds every `LoraLinear` in *module* back into its wrapped `base`
    Linear, in place, and swaps the plain Linear back in via `setattr` --
    the inverse of `inject_lora`, producing an ordinary (checkpoint-format
    compatible) module again.

    For each `LoraLinear`: `base.weight.data += scale * (lora_b.weight @
    lora_a.weight)` (both `(out, r) @ (r, in) -> (out, in)`, matching
    `base.weight`'s own `(out_features, in_features)` shape -- this is the
    closed-form fold of `base(x) + scale * lora_b(lora_a(x))` into a single
    Linear: `lora_b(lora_a(x)) = x @ lora_a.weight.T @ lora_b.weight.T = x @
    (lora_b.weight @ lora_a.weight).T`, so adding `scale * (lora_b.weight @
    lora_a.weight)` to `base.weight` reproduces the wrapped forward exactly,
    `tests/test_adapt_lora.py::test_merge_restores_plain_linear_and_forward`
    verifies this numerically). `base`'s bias is untouched -- LoRA never
    adds a bias term. The mutated `base` object itself (not a freshly
    constructed Linear) is what gets swapped back in via `setattr`, so
    identity/`requires_grad` state on it survives merge unchanged except for
    the weight update (merged models are typically used for frozen
    inference afterward, e.g. via `load_beats_model`, not further training).

    Args:
        module: Root module to walk (same contract as `inject_lora`).

    Returns:
        Count of `LoraLinear`s merged.
    """
    targets: list[tuple[torch.nn.Module, str, LoraLinear]] = []
    for name, child in list(module.named_modules()):
        if not isinstance(child, LoraLinear):
            continue
        parent_path, attr_name = _parent_path_and_attr(name)
        parent = _resolve_parent(module, parent_path)
        targets.append((parent, attr_name, child))

    for parent, attr_name, lora in targets:
        delta = lora.scale * (lora.lora_b.weight @ lora.lora_a.weight)
        lora.base.weight.data += delta
        setattr(parent, attr_name, lora.base)

    return len(targets)


def lora_parameters(module: torch.nn.Module) -> Iterator[torch.nn.Parameter]:
    """Yields ONLY the trainable adapter parameters (`lora_a`/`lora_b`
    weights) of every `LoraLinear` in *module* -- never anything from a
    `base` Linear, whether or not it happens to have `requires_grad=True`
    (`inject_lora` always freezes `base`, but this function does not rely on
    that: it filters by module identity, not by gradient flag). This is the
    parameter set an adaptation optimizer should train (spec D2: "only
    adapter params train"); passing `module.parameters()` directly to an
    optimizer would also (uselessly, since frozen) include every base
    weight.
    """
    for child in module.modules():
        if isinstance(child, LoraLinear):
            yield from child.lora_a.parameters()
            yield from child.lora_b.parameters()
