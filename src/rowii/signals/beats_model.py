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

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

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

_QUANTIZED_ENGINE_PREFERENCE: tuple[str, ...] = ("fbgemm", "x86", "qnnpack")
"""Preference order for `torch.backends.quantized.engine` (`select_quantized_
engine`): `"fbgemm"`/`"x86"` are the x86 CPU kernels PyTorch ships for
`torch.ao.quantization.quantize_dynamic` -- this project's actual deployment
target (an x86-64 on-premise server, design spec D6); `"qnnpack"` (ARM/
mobile-oriented) is the fallback that lets the SAME code path work on this
project's own Apple Silicon dev machines. `torch.backends.quantized.engine`
defaults to `"none"`, and BOTH quantizing (`scripts/quantize_beats.py`) and
merely `torch.load`-ing (never mind running) a previously-saved dynamically-
quantized module (`load_quantized_beats_model` below, in a FRESH process
where nothing has set this yet) raise `RuntimeError` ("Unknown qengine" /
"NoQEngine") unless this is set to one of `torch.backends.quantized.
supported_engines` FIRST -- a real, verified-by-hand finding (not documented
anywhere in the design spec): unpickling a dynamically-quantized `nn.Linear`'s
packed weight dispatches through the SAME backend-engine selection
`quantize_dynamic` itself does."""


def select_quantized_engine(supported: Sequence[str] | None = None) -> str:
    """The best available `torch.backends.quantized.engine` value for this
    machine (see `_QUANTIZED_ENGINE_PREFERENCE`'s docstring for why this
    selection needs to exist at all). Callers assign the result to `torch.
    backends.quantized.engine` themselves, BEFORE `torch.ao.quantization.
    quantize_dynamic` (`scripts/quantize_beats.py`) or `torch.load`-ing a
    previously-quantized module (`load_quantized_beats_model` below).

    Args:
        supported: Override for `torch.backends.quantized.supported_engines`
            (tests only -- this torch attribute cannot itself be
            monkeypatched: `torch.backends.quantized`'s own
            `_SupportedQEnginesProp.__set__` unconditionally raises
            `RuntimeError("Assignment not supported")`, since it reflects
            what this torch BUILD was actually compiled with, not process
            state). `None` (every real caller) reads the genuine value.

    Returns:
        The first of `_QUANTIZED_ENGINE_PREFERENCE` present in the supported
        list; if neither preferred engine is present (a torch build with only
        exotic/future backends), the first supported entry other than the
        always-present `"none"` sentinel.

    Raises:
        RuntimeError: the supported list contains nothing but `"none"` -- this
            torch build was compiled without ANY quantized CPU backend, so
            INT8 dynamic quantization is unavailable regardless of preference.
    """
    import torch

    if supported is not None:
        engines = list(supported)
    else:
        engines = torch.backends.quantized.supported_engines
    for preferred in _QUANTIZED_ENGINE_PREFERENCE:
        if preferred in engines:
            return preferred
    for engine in engines:
        if engine != "none":
            return str(engine)
    raise RuntimeError(
        f"no usable torch quantized backend engine on this machine (supported: "
        f"{engines!r}) -- INT8 dynamic quantization needs fbgemm/x86/qnnpack "
        "compiled into this torch build"
    )


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


def load_quantized_beats_model(checkpoint: Path) -> BEATs:
    """Load a `scripts/quantize_beats.py`-produced quantized-module pickle
    from *checkpoint*, in eval mode, on CPU (design spec D6).

    Unlike `load_beats_model` (which reconstructs a `BEATs` instance from a
    `{"cfg", "model"}` state-dict checkpoint), this function `torch.load`s the
    QUANTIZED MODULE OBJECT ITSELF. `torch.ao.quantization.quantize_dynamic`'s
    dynamically-quantized `nn.Linear` submodules (`torch.ao.nn.quantized.
    dynamic.Linear`) pack their int8 weight plus a separate scale/zero-point
    into an opaque `LinearPackedParams` object, not a plain tensor a `state_
    dict` round trip alone can reconstruct -- so `scripts/quantize_beats.py`
    pickles the whole module (`torch.save(quantized_module, path)`) instead
    of saving `{"cfg", "model"}`, and this is the dedicated counterpart
    loader (mirrors `load_beats_model`'s/`rowii.tfc.wrapper.load_tfc_model`'s/
    `rowii.adapt.student.load_student_model`'s own "one loader function per
    checkpoint format" convention).

    Always CPU -- no *device* argument, unlike `load_beats_model`: dynamically
    quantized Linear kernels are a CPU-only PyTorch backend (no MPS/CUDA
    dynamic-quantization support in eager mode), which happens to match this
    project's own deployment target for the compact/quantized pole (an
    on-premise server with no GPU, design spec D6) -- there is no other
    device this loader could sensibly target.

    Sets `torch.backends.quantized.engine` (`select_quantized_engine`) BEFORE
    `torch.load` -- see `_QUANTIZED_ENGINE_PREFERENCE`'s docstring for the
    verified-by-hand rationale (unpickling a `LinearPackedParams` dispatches
    through the same backend-engine selection `quantize_dynamic` itself does,
    and fails with a bare `RuntimeError` otherwise).

    Args:
        checkpoint: Path to a `.pt` file written by `torch.save(quantized_
            module, path)` (`scripts/quantize_beats.py`'s own output format).

    Returns:
        The unpickled `BEATs` module (the SAME top-level class `load_beats_
        model` returns -- `quantize_dynamic` only swaps `nn.Linear` LEAVES for
        quantized counterparts, never the enclosing module's own type or its
        OTHER submodules' callable surface, verified by hand -- so every
        non-Linear stage `BeatsFeaturizer`'s `_RealBeatsEncoder.extract`/
        `BEATs.extract_features` calls, e.g. `patch_embedding`/`layer_norm`/
        `encoder`, is exactly the same attribute path as the fp32 model),
        `.eval()`'d, on CPU.

    Raises:
        FileNotFoundError: if *checkpoint* does not exist (mirrors `load_
            beats_model`'s/`load_tfc_model`'s/`load_student_model`'s own
            convention -- names the exact path in the message).
    """
    import torch

    if not checkpoint.exists():
        raise FileNotFoundError(f"quantized BEATs checkpoint not found: {checkpoint}")

    torch.backends.quantized.engine = select_quantized_engine()
    # torch.load's return type is untyped (Any) to mypy; the runtime type is
    # always the pickled BEATs instance quantize_beats.py itself saved (see
    # this function's own docstring on why the enclosing module type is
    # unaffected by quantize_dynamic).
    model = cast("BEATs", torch.load(checkpoint, map_location="cpu", weights_only=False))
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
