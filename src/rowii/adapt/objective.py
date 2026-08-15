"""Masked-reconstruction proxy objectives (Step-2 package-5 spec D1 as
amended by Amendment A1, Task 1 + Task 3 rework): the adaptation objectives
shared by LoRA and full fine-tune. Eager torch at module level -- see
`rowii.adapt.lora`'s module docstring for why that is the right discipline
for this package (adaptation only ever runs with the `[beats]` extra
installed).

BEATs' OWN pre-training objective (discrete acoustic-token distillation via
its own tokenizer network) is not reproducible here -- that tokenizer is not
part of the vendored inference-time module, and training one from scratch is
out of this package's scope (design spec D1, non-goals). Both objectives here
are therefore DOCUMENTED PROXIES, always described as such wherever
adapted-model results appear. Two variants:

- `masked_token_loss` -- THE adaptation objective for BEATs: operates on the model's own NATIVE
  pre-encoder patch tokens (`(B, T, D)`, produced by the frozen preprocess ->
  patch_embedding -> layer_norm -> post_extract_proj pipeline in
  `scripts/adapt_beats.py`), masks a random subset of token ROWS (zeroing
  them), runs the (possibly-adapted) encoder on the masked token sequence,
  and reconstructs the ORIGINAL (pre-mask) token embeddings with a small
  linear `head`, MSE on masked positions only -- a latent-target MAE.
  Because train-time inputs are the SAME tokens the deployed inference path
  (`BEATs.extract_features`) feeds the encoder, adapters trained through this
  objective train on the distribution they will be scored on.

- `masked_patch_loss` -- the original frame-level variant (D1's pre-amendment
  wording), masking whole `(mels,)` TIME-FRAME rows of the fbank BEFORE the
  encoder. RETAINED for encoders that consume the fbank frame-by-frame
  (position-preserving encoders: its `encoder_forward` must map `(B, frames,
  mels)` -> `(B, frames, D)`, the SAME `frames` count in and out). It is
  STRUCTURALLY INCOMPATIBLE with BEATs' native forward (Amendment A1's
  trigger, discovered in Task 3): BEATs' `patch_embedding` is a strided
  `Conv2d` over the fbank "image" that downsamples BOTH the time and mel
  axes and then FLATTENS the two patch axes into one token axis (~98 frames
  -> 48 tokens for the real checkpoint), so no closure over the native
  BEATs stack can satisfy this loss's frame-count-preserving contract --
  and bridging the gap with a NON-native frame-preserving projection (Task
  3's first, retired attempt) trains the adapters on an input distribution
  decoupled from the deployed inference path. Do not use this objective for
  BEATs adaptation; use `masked_token_loss`.
"""
from __future__ import annotations

from collections.abc import Callable

import torch


def _random_row_mask(
    batch: int, n_rows: int, mask_frac: float, generator: torch.Generator | None
) -> torch.Tensor:
    """`(batch, n_rows)` boolean mask with ~*mask_frac* of each sample's rows
    True -- the shared masking core of both objectives in this module
    (factored out in the Task-3 rework; `masked_patch_loss`'s draw sequence
    is byte-identical to its pre-refactor behaviour: same per-sample
    `torch.randperm` calls against the same generator, in the same order).

    The masked count is `round(mask_frac * n_rows)`, clamped to
    `[1, n_rows]` (at least one row is always masked, even for a tiny
    *n_rows* where the rounding would floor to 0). Built on CPU regardless
    of where the caller's data lives (matching `torch.randperm`'s own
    CPU-generator default; callers move the mask once, as a whole).
    """
    n_masked = min(n_rows, max(1, round(mask_frac * n_rows)))
    mask = torch.zeros(batch, n_rows, dtype=torch.bool)
    for b in range(batch):
        perm = (
            torch.randperm(n_rows, generator=generator)
            if generator is not None
            else torch.randperm(n_rows)
        )
        mask[b, perm[:n_masked]] = True
    return mask


def masked_token_loss(
    tokens: torch.Tensor,
    encoder_forward: Callable[[torch.Tensor], torch.Tensor],
    head: torch.nn.Linear,
    mask_frac: float = 0.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Native token-level masked-reconstruction MSE (module docstring: the
    BEATs adaptation objective, spec D1 as amended by Amendment A1) -- a
    latent-target MAE: mask ~*mask_frac* of *tokens*' rows (zeroed), encode
    the masked sequence, and regress the encoder's output at the masked
    positions (through *head*) onto the ORIGINAL pre-mask token embeddings.

    The reconstruction TARGET (`tokens[mask]`) is DETACHED from the graph --
    the stop-gradient every latent-target self-supervision recipe applies to
    its targets (MAE regresses constant pixels; data2vec/BYOL stop-gradient
    their teacher branch). Without it, a `--mode full` run (where the token
    construction itself is trainable) could lower the loss by moving the
    TARGETS toward the predictions -- collapsing the token embeddings --
    instead of learning to reconstruct; with it, gradient reaches the token
    construction only through the legitimate INPUT path (the unmasked rows
    the encoder attends to). In lora mode the distinction is moot (the token
    construction is frozen, so `tokens` never requires grad), but the detach
    is unconditional -- correctness must not depend on which mode built the
    input. `tests/test_adapt_objective.py::
    test_token_target_is_detached_masked_rows_get_no_gradient` pins exactly
    this: masked rows of a grad-requiring `tokens` receive ZERO gradient
    (their input-path contribution is overwritten by the zeroing write, and
    the target path is detached) while unmasked rows DO receive gradient
    through a position-mixing encoder.

    Gradient-flow caveat, inherited from `masked_patch_loss` (see its
    docstring for the full derivation): with a purely POSITION-WISE
    *encoder_forward*, encoder weight gradients at masked positions are
    exactly zero (the input rows there are zeroed). A real attention encoder
    mixes information across positions (a masked query attends to unmasked
    keys/values), so LoRA q/v adapter weights DO receive gradients there --
    `tests/test_adapt_beats.py::
    test_lora_adapter_grads_nonzero_against_real_tiny_encoder` asserts this
    against the real vendored encoder.

    Args:
        tokens: `(B, T, D)` NATIVE pre-encoder token embeddings (for BEATs:
            the output of preprocess -> patch_embedding -> layer_norm ->
            post_extract_proj, `scripts/adapt_beats.py`'s `_native_tokens`).
            Never mutated (masking works on an internal clone), so callers
            can reuse the same tensor across calls.
        encoder_forward: Maps the `(B, T, D)` masked token sequence to
            `(B, T, D_out)` per-token encoder output, any `D_out` (*head*
            reconciles it back to `D`). Token count `T` must be preserved --
            trivially true for a transformer encoder consuming its own
            native tokens (the whole point of Amendment A1).
        head: `nn.Linear(D_out, D)` projecting the encoder output back to
            token-embedding width; trained jointly with the encoder
            (`scripts/adapt_beats.py` includes `head.parameters()` in its
            optimizer for both modes).
        mask_frac: Fraction of token rows (dim 1) masked per sample, spec
            D1/A1 default 0.3 (30%); rounding/clamping per
            `_random_row_mask`.
        generator: Optional CPU `torch.Generator` controlling which rows are
            masked (same contract as `masked_patch_loss`'s parameter of the
            same name). `None` draws from the global torch RNG.

    Returns:
        Scalar (0-dim) MSE loss tensor over the masked positions only.
    """
    batch, n_tokens, _dim = tokens.shape
    mask = _random_row_mask(batch, n_tokens, mask_frac, generator).to(tokens.device)

    masked_tokens = tokens.clone()
    masked_tokens[mask] = 0.0

    encoded = encoder_forward(masked_tokens)
    predicted = head(encoded)

    return torch.nn.functional.mse_loss(predicted[mask], tokens[mask].detach())


def masked_patch_loss(
    encoder_forward: Callable[[torch.Tensor], torch.Tensor],
    fbank: torch.Tensor,
    head: torch.nn.Linear,
    mask_frac: float = 0.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Masked-frame reconstruction MSE -- the frame-level objective variant,
    for POSITION-PRESERVING encoders only (module docstring; NOT for BEATs'
    native patchifying forward, whose `patch_embedding` collapses ~98 fbank
    frames into 48 flattened patch tokens and therefore cannot satisfy this
    function's frame-count-preserving `encoder_forward` contract -- Amendment
    A1 retargeted BEATs adaptation to `masked_token_loss` for exactly this
    reason; this function is retained for non-patchifying encoders).

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
    the Task-3 integration tests assert nonzero LoRA-adapter gradients after
    a backward pass (through `masked_token_loss`, the objective actually
    used for BEATs adaptation), precisely because the unit tests here
    structurally cannot.

    Args:
        encoder_forward: Maps a `(B, frames, mels)` fbank (already
            frame-masked by this function) to `(B, frames, D)` per-frame
            encoder output, any `D` (`head` reconciles it back to `mels`).
            MUST preserve the frame axis -- see the patchify-incompatibility
            paragraph above for why no native-BEATs closure can.
        fbank: `(B, frames, mels)` UNMASKED input; never mutated (masking
            works on an internal clone) so callers can reuse the same tensor
            across calls (e.g. the training-loop pattern in
            `tests/test_adapt_objective.py::test_loss_decreases_with_training`,
            which calls this function once per optimizer step on the same
            `fbank`).
        head: `nn.Linear(D, mels)` projecting `encoder_forward`'s output
            back to fbank-frame width; trained jointly with the encoder.
        mask_frac: Fraction of TIME FRAMES (dim 1) masked per sample, spec
            D1 default 0.3 (30%); rounding/clamping per `_random_row_mask`.
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
    mask = _random_row_mask(batch, n_frames, mask_frac, generator).to(fbank.device)

    masked_fbank = fbank.clone()
    masked_fbank[mask] = 0.0

    encoded = encoder_forward(masked_fbank)
    predicted = head(encoded)

    return torch.nn.functional.mse_loss(predicted[mask], fbank[mask])
