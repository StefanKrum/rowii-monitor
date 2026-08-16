"""Cross-attention fusion head model: the third fusion level -- a lightweight
cross-attention head on FROZEN per-branch
features (audio-beats' own 768-d embeddings, the `fusion` cache's own
vibration-branch columns), trained CLIP-style (audio and vibration views of the
SAME window as the positive pair) and scored via kNN on the joint embedding at
inference (`--xattn-fusion`, `scripts/run_step2.py`).

Mirrors `rowii.tfc.model`'s/`rowii.adapt._student_model`'s identical role in
their own packages: the ONE module under `rowii.fusionx` that imports torch at
module level (a lazily-acquired torch handle can only be typed `Any`, and mypy
rejects subclassing a value of type `Any`, so an `nn.Module` subclass needs a
REAL top-level `import torch` behind it) -- imported ONLY lazily, from inside
`rowii.fusionx.wrapper`'s functions (`load_xattn_head`, `joint_embeddings`) and
`scripts/train_xattn.py`'s own training loop, never at `wrapper.py`'s own top
level, never at this package's `__init__.py`.

Architecture (binding contract): two independent lift `Linear`s project each
branch's own raw feature width down to `cfg.lift_dim` -- `audio_lift:
Linear(cfg.audio_dim, cfg.lift_dim)`, `vib_lift: Linear(vib_in_dim,
cfg.lift_dim)` (`vib_in_dim` is a CONSTRUCTOR argument, not an `XattnConfig`
field -- that dataclass's own docstring explains why) -- followed by ONE
`torch.nn.MultiheadAttention(cfg.lift_dim, cfg.heads, batch_first=True)`
treating the lifted audio embedding as a single-token QUERY and the lifted
vibration embedding as the single-token KEY/VALUE (i.e. "let the audio view of
this window attend to the vibration view of the SAME window"), a residual
connection (audio lift + attention output) + `LayerNorm`, and a final
`Linear(cfg.lift_dim, cfg.out_dim)` projecting to the joint embedding
`--xattn-fusion` actually scores with kNN.

Training objective (this module never trains itself -- `scripts/train_xattn.py`
owns the training loop; this docstring records the binding design so both stay
in sync): `tfc_loss` (`rowii.tfc.model`) IS the symmetric InfoNCE / NT-Xent this
CLIP-style alignment objective needs -- imported and reused VERBATIM, not
reimplemented. `rowii.tfc.model`'s own module docstring already describes
exactly this structure ("each window's own (z_t, z_f) pair as the single
positive, every other projection in the batch (both views) as a negative"); the
only change here is relabelling "time view / frequency view" as "audio lift /
vibration lift" -- the SAME cross-view contrastive-alignment shape, applied to a
different pair of views of the same underlying instance (a plant window, not a
signal).

The composite training loss (binding design choice, applied in `scripts/
train_xattn.py`'s own training loop -- this module's `forward` only ever
returns the joint embedding, never the loss):

    loss = tfc_loss(lift_a, lift_v, T) + 0.5 * tfc_loss(joint, lift_a.detach(), T)

The FIRST term is the actual CLIP-style alignment objective: it shapes
`audio_lift`/`vib_lift` so the two "views" of one window (audio, vibration) end
up close in a SHARED aligned subspace, exactly `rowii.tfc.model.tfc_loss`'s own
job for the time/frequency pair. The SECOND term additionally shapes the JOINT
(post-attention) embedding toward that SAME aligned lift space, at HALF weight
and through a DETACHED `lift_a`. Be precise about what the detach does and does
NOT do (the original wording here overclaimed, caught by the final whole-branch
review with a runtime gradient probe): `joint = model(audio, vib)` recomputes
`audio_lift(audio)` inside `forward` (query + residual), so the second term's
gradient DOES flow into `audio_lift` through the prediction side -- measured
per-submodule grad norms on a fresh head under the term-2-only loss:
`audio_lift` 11.18 (the largest), `out` 4.63, `attn.out_proj` 2.00, `vib_lift`
0.70. What `lift_a.detach()` actually removes is the TARGET side: without it,
term 2 could reduce its loss by dragging the alignment target toward the joint
prediction (target chasing), instead of moving the prediction toward a target
the FIRST term alone owns. The auxiliary term's influence is bounded by its
`0.5` weight, not by the detach. Both terms
share one `temperature` (`cfg.temperature`). The `0.5` weight and the
`lift_a.detach()` choice are this task's own binding design constants -- `scripts/
train_xattn.py`'s `_JOINT_LOSS_WEIGHT` module constant is where they are
actually applied.
"""
from __future__ import annotations

from typing import cast

import torch

from rowii.fusionx.wrapper import XattnConfig


class XattnHead(torch.nn.Module):
    """Cross-attention fusion head: audio lift + vibration lift -> one-token
    cross-attention (audio as query, vibration as key/value) -> residual +
    `LayerNorm` -> output projection.

    `forward` returns ONLY the joint (post-attention) embedding -- `scripts/
    train_xattn.py`'s training loop reads the two LIFTS separately, via
    `self.audio_lift`/`self.vib_lift` directly (both plain, PUBLIC submodules
    -- module docstring's composite-loss section), not through `forward`,
    since the alignment term needs the lifts BEFORE attention ever mixes them.
    This means `audio_lift`/`vib_lift` are computed TWICE per training step
    (once standalone for the alignment term, once again inside `forward` for
    the joint term) -- a deliberate, accepted redundancy: `forward`'s own
    return-type contract is fixed to `(B, cfg.out_dim)` (the binding
    interface), and the head is tiny enough (one attention block, two small
    lifts) on the small per-run calibration-side window counts this project
    trains on that the extra pass costs nothing that matters.
    """

    def __init__(self, cfg: XattnConfig, vib_in_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.audio_lift = torch.nn.Linear(cfg.audio_dim, cfg.lift_dim)
        self.vib_lift = torch.nn.Linear(vib_in_dim, cfg.lift_dim)
        self.attn = torch.nn.MultiheadAttention(cfg.lift_dim, cfg.heads, batch_first=True)
        self.norm = torch.nn.LayerNorm(cfg.lift_dim)
        self.out = torch.nn.Linear(cfg.lift_dim, cfg.out_dim)

    def forward(self, audio: torch.Tensor, vib: torch.Tensor) -> torch.Tensor:
        """`audio`: `(B, cfg.audio_dim)`. `vib`: `(B, vib_in_dim)`, row-aligned
        with *audio* (the SAME window). Returns `(B, cfg.out_dim)` joint
        embeddings.

        Internally: lift both branches to `(B, 1, cfg.lift_dim)` single-token
        sequences -> `MultiheadAttention(query=audio_token, key=value=
        vib_token)` (`need_weights=False`: the attention weights themselves
        are never consumed downstream, so computing them is skipped) ->
        squeeze back to `(B, cfg.lift_dim)` -> residual-add the audio lift
        (the block's own query input) + `LayerNorm` -> `Linear` to
        `cfg.out_dim`.
        """
        a = self.audio_lift(audio).unsqueeze(1)  # (B, 1, lift_dim)
        v = self.vib_lift(vib).unsqueeze(1)  # (B, 1, lift_dim)
        attn_out, _weights = self.attn(a, v, v, need_weights=False)  # (B, 1, lift_dim)
        fused = self.norm(attn_out.squeeze(1) + a.squeeze(1))  # residual + LayerNorm
        return cast(torch.Tensor, self.out(fused))
