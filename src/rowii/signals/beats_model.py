"""Thin loader for the official BEATs checkpoint format.

The BEATs model architecture (patch embedding, transformer encoder stack with
relative-position/GRU-rel-pos/deep-norm variants, mel-filterbank
preprocessing) is Microsoft's published implementation
(`BEATs: Audio Pre-Training with Acoustic Tokenizers`,
https://arxiv.org/abs/2212.09058, https://github.com/microsoft/unilm/tree/master/beats,
MIT-licensed) -- not something this project reimplements from scratch. That
reference code has no `pip install`-able package; the Task 14 brief
sanctions two paths for using it: a minimal fresh reimplementation of only
the inference-time pieces, or vendoring the reference module with
attribution. A fresh reimplementation was attempted first and rejected: the
encoder alone (`rowii.vendor.beats.backbone.TransformerEncoder`) implements
T5-style relative-position bucketing, a GRU-gated relative-position variant,
and DeepNorm residual scaling, all of which the pretrained
`BEATs_iter3_plus_AS2M.pt` checkpoint's state dict depends on exactly
matching (verified here by `torch`'s `load_state_dict` returning
`<All keys matched successfully>`, no `strict=False` escape hatch) --
reimplementing that surface "fresh" risks a subtly wrong forward pass that
still loads without error, which is a worse failure mode than an attributed,
unmodified-in-substance vendor copy. The vendored code lives in
`rowii.vendor.beats` (`BEATs.py`, `backbone.py`, `modules.py`, each carrying
Microsoft's original MIT header plus a vendoring provenance note); this
module is the ONLY place in `rowii`'s own code that imports it, and is
itself fresh, minimal, project-specific code: checkpoint loading, fbank
preprocessing, and pooling -- exactly what Step 1's featurizer contract
needs and nothing from the reference wrapper's training/LoRA paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from rowii.vendor.beats.BEATs import BEATs

BEATS_SAMPLE_RATE_HZ = 16_000
"""BEATs' required input sample rate -- fixed by how the checkpoint was pretrained."""

BEATS_EMBED_DIM = 768
"""`BEATsConfig.encoder_embed_dim` for `BEATs_iter3_plus_AS2M.pt` (BEATs-base)."""

_FBANK_MEAN = 15.41663
_FBANK_STD = 6.55582
"""Global fbank normalisation constants from the official `BEATs.preprocess`
(`rowii.vendor.beats.BEATs.BEATs.preprocess`'s own defaults) -- fixed statistics
baked into how the checkpoint was pretrained, not something to refit per dataset."""


def load_beats_model(checkpoint: Path, device: torch.device) -> BEATs:
    """Load the frozen BEATs encoder from *checkpoint* onto *device*, in eval mode.

    Args:
        checkpoint: Path to the official BEATs `.pt` file (e.g.
            `BEATs_iter3_plus_AS2M.pt`). Must contain `"cfg"` and `"model"` keys,
            the format the official checkpoints ship in.
        device: Torch device to place the model on.

    Returns:
        The vendored `BEATs` module, `.eval()`'d, with every parameter's
        `requires_grad` left at its state-dict default (loading does not
        itself freeze gradients -- callers that need a hard freeze wrap
        forward calls in `torch.no_grad()`, as `BeatsFeaturizer.transform`
        does).

    Raises:
        FileNotFoundError: if *checkpoint* does not exist.
    """
    import torch

    from rowii.vendor.beats.BEATs import BEATs, BEATsConfig

    if not checkpoint.exists():
        raise FileNotFoundError(f"BEATs checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    # BEATsConfig.__init__ is unannotated vendored code (Microsoft's original,
    # kept verbatim -- see module docstring); the call itself is correct at
    # runtime (verified by the real-checkpoint smoke test), just untyped.
    cfg = BEATsConfig(state["cfg"])  # type: ignore[no-untyped-call]
    model = BEATs(cfg)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def beats_fbank(waveform: torch.Tensor) -> torch.Tensor:
    """128-mel log-fbank for one 16 kHz mono *waveform*, BEATs-normalised.

    Reimplements the data-only body of the official
    `rowii.vendor.beats.BEATs.BEATs.preprocess` for a single waveform (that
    method takes no `self` state -- it is pure `torchaudio` + arithmetic --
    so calling it would require an already-constructed `BEATs` instance for
    no reason; this module intentionally does not depend on model
    construction just to fbank one waveform):
    `torchaudio.compliance.kaldi.fbank` with `num_mel_bins=128`, 25 ms
    frames / 10 ms shift at 16 kHz, scaled by `2**15` first (the reference
    implementation's int16-range convention for a float waveform in
    `[-1, 1]`), then `(fbank - mean) / (2 * std)` with the fixed constants
    above.

    Args:
        waveform: `(n_samples,)` float tensor at `BEATS_SAMPLE_RATE_HZ` Hz.

    Returns:
        `(n_frames, 128)` float tensor.
    """
    from typing import cast

    import torch
    import torchaudio.compliance.kaldi as ta_kaldi

    scaled = waveform.unsqueeze(0) * 2**15
    fbank = ta_kaldi.fbank(
        scaled,
        num_mel_bins=128,
        sample_frequency=BEATS_SAMPLE_RATE_HZ,
        frame_length=25,
        frame_shift=10,
    )
    # torchaudio's compliance.kaldi stubs are incomplete, so fbank's dtype is
    # Any to mypy; the runtime type is torch.Tensor (verified by every test
    # in tests/test_beats.py exercising this exact call).
    return cast(torch.Tensor, (fbank - _FBANK_MEAN) / (2 * _FBANK_STD))
