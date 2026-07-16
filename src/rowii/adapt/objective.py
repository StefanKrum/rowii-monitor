"""Masked-patch reconstruction proxy objective (Step-2 package-5 spec D1,
Task 1): the adaptation objective shared by LoRA and full fine-tune. Eager
torch at module level -- see `rowii.adapt.lora`'s module docstring for why
that is the right discipline for this package (adaptation only ever runs
with the `[beats]` extra installed).

BEATs' OWN pre-training objective (discrete acoustic-token distillation via
its own tokenizer network) is not reproducible here -- that tokenizer is not
part of the vendored inference-time module, and training one from scratch is
out of this package's scope (design spec D1, non-goals). The objective
implemented here is therefore a DOCUMENTED PROXY, always described as such
wherever adapted-model results appear (spec D1, acceptance criteria): mask a
random subset of the input fbank's TIME FRAMES (zeroing them), run the
(possibly-adapted) encoder on the masked input, project its output back to
fbank-frame width with a small linear `head`, and take the MSE against the
ORIGINAL (unmasked) fbank values at the masked positions only -- an MAE
(He et al. 2021)-style masked-reconstruction loss, self-supervised, using
target NORMAL windows only.

Frame-level masking (whole `(mels,)` rows of the fbank zeroed) rather than
BEATs' own internal PATCH tokens (the vendored encoder's actual pre-training
unit, produced by `patch_embedding`'s strided `Conv2d` over the fbank image)
is a deliberate simplification: this objective operates on the fbank BEFORE
`encoder_forward` is ever called, so it has no dependency on how any given
`encoder_forward` closure internally patches/projects/pools its input --
`adapt_beats.py` (Task 3) can close over the vendored BEATs' patch-embedding
+ transformer stack, or over a bare tiny stand-in (this module's own test
suite uses `nn.Linear` as `encoder_forward`), with the exact same masking
logic either way. The tradeoff is that "one masked frame" here is a coarser
unit than "one masked patch" in the original BEATs recipe (a 16x16 fbank
patch spans multiple consecutive frames) -- accepted because this project
needs A self-supervised target-normal objective for measuring the
LoRA-vs-full-FT-vs-frozen adaptation axis (spec D1's question), not a
faithful reproduction of BEATs' own pre-training procedure.
"""
from __future__ import annotations

from collections.abc import Callable

import torch


def masked_patch_loss(
    encoder_forward: Callable[[torch.Tensor], torch.Tensor],
    fbank: torch.Tensor,
    head: torch.nn.Linear,
    mask_frac: float = 0.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Masked-frame reconstruction MSE (module docstring: the adaptation
    proxy objective).

    Gradient-flow caveat for test authors (Task-1 review note): with a
    purely POSITION-WISE `encoder_forward` (e.g. this module's own test
    suite's bare `nn.Linear`, applied independently per frame), the
    encoder's WEIGHT gradients are exactly zero -- the loss reads
    predictions only at masked positions, where the input frame is zeroed,
    and `d(Wx + b)/dW` is an outer product with `x = 0` there; only bias
    terms (and `head`, whose input is the encoder's nonzero bias output)
    receive gradient. `tests/test_adapt_objective.py::
    test_loss_decreases_with_training` still passes (bias + head training
    suffices for a decreasing loss), but no test with a position-wise
    encoder can prove that ADAPTER weights train. A real attention encoder
    mixes information across positions (a masked query attends to unmasked
    keys/values), so LoRA q/v adapter weights DO receive gradients there --
    Task 3's integration tests against the real vendored encoder must
    assert nonzero LoRA-adapter gradients after a backward pass, precisely
    because the unit tests here structurally cannot.

    Args:
        encoder_forward: Maps a `(B, frames, mels)` fbank (already
            frame-masked by this function) to `(B, frames, D)` per-frame
            encoder output, any `D` (`head` reconciles it back to `mels`).
            Typically a closure over an (adapted) BEATs encoder's frame-level
            forward path (`adapt_beats.py`, Task 3); the test suite closes
            over a bare `nn.Linear`/identity function instead, since this
            function's own contract does not care what `encoder_forward`
            actually is, only its input/output shapes.
        fbank: `(B, frames, mels)` UNMASKED input; never mutated (masking
            works on an internal clone) so callers can reuse the same tensor
            across calls (e.g. the training-loop pattern in
            `tests/test_adapt_objective.py::test_loss_decreases_with_training`,
            which calls this function once per optimizer step on the same
            `fbank`).
        head: `nn.Linear(D, mels)` projecting `encoder_forward`'s output
            back to fbank-frame width; trained jointly with the encoder
            (`adapt_beats.py` includes `head.parameters()` in its optimizer,
            same pattern as `tests/test_adapt_objective.py::
            test_loss_decreases_with_training`).
        mask_frac: Fraction of TIME FRAMES (dim 1) masked per sample, spec
            D1 default 0.3 (30%). Rounded to the nearest whole frame count,
            clamped to `[1, frames]` (at least one frame is always masked,
            even for a tiny `frames` where `round(mask_frac * frames)` would
            floor to 0).
        generator: Optional `torch.Generator` (CPU-typed, matching
            `torch.randperm`'s own default -- this function never moves it
            to `fbank`'s device) controlling which frames are masked, for
            reproducibility (every call in
            `tests/test_adapt_objective.py::test_loss_decreases_with_training`
            passes a generator freshly re-seeded to the SAME value, so every
            optimizer step is scored against the identical masked positions
            -- what makes gradient descent on this loss well-defined across
            steps). `None` draws from the global torch RNG.

    Returns:
        Scalar (0-dim) MSE loss tensor over the masked positions only.
    """
    batch, n_frames, _n_mels = fbank.shape
    n_masked = min(n_frames, max(1, round(mask_frac * n_frames)))

    # Per-sample independent random frame subset (spec: "per-sample random
    # frame subset via the generator") -- built on CPU regardless of
    # `fbank`'s device (matching `generator`'s own CPU-only contract above),
    # then moved once as a whole boolean mask.
    mask = torch.zeros(batch, n_frames, dtype=torch.bool)
    for b in range(batch):
        perm = (
            torch.randperm(n_frames, generator=generator)
            if generator is not None
            else torch.randperm(n_frames)
        )
        mask[b, perm[:n_masked]] = True
    mask = mask.to(fbank.device)

    masked_fbank = fbank.clone()
    masked_fbank[mask] = 0.0

    encoded = encoder_forward(masked_fbank)
    predicted = head(encoded)

    return torch.nn.functional.mse_loss(predicted[mask], fbank[mask])
