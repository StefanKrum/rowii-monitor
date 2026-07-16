"""Tests for the frozen-BEATs featurizer (Task 14, `[beats]` extra).

Stub-encoder tests (below `# stub-encoder tests`) inject a deterministic
`BeatsEncoderProtocol` stub via `BeatsFeaturizer`'s `encoder=` parameter, so
they exercise `best_device`/mono-mix/resample/fbank/shape wiring without
needing the real pretrained checkpoint -- but constructing `torch.Tensor`s at
all still needs `torch` importable, hence `pytest.importorskip("torch")` at
module scope (this repo's core dependencies do not include torch; it is
opt-in via the `[beats]` extra).

`select_quantized_engine`/`load_quantized_beats_model` (below `# quantized
backend engine selection`/`# load_quantized_beats_model`) and `BeatsFeaturizer`'s
own `int8_model_path` branch (below `# int8 branch`) are Step-2 package-5 spec
D6, Task 5's additions: post-training INT8 dynamic quantization support. These
sections build a tiny REAL (2-layer) vendored `BEATs` instance and quantize it
FOR REAL (`torch.ao.quantization.quantize_dynamic`, proven <1s even at the
real `BEATS_EMBED_DIM`=768 width -- no checkpoint weights loaded, mirrors
`tests/test_adapt_beats.py`'s own `_tiny_beats()` recipe), rather than a
hand-rolled stand-in -- `scripts/quantize_beats.py`'s own tests
(`tests/test_quantize_beats.py`) reuse the identical recipe.

The single `@pytest.mark.data` test at the bottom exercises the REAL
checkpoint end-to-end (loading + real encoder forward pass), consistent with
this repo's other `@pytest.mark.data` real-data smoke tests
(`tests/test_real_data.py`).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rowii.signals.beats import BeatsFeaturizer, best_device  # noqa: E402
from rowii.signals.beats_model import (  # noqa: E402
    BEATS_EMBED_DIM,
    load_quantized_beats_model,
    select_quantized_engine,
)

# ---------------------------------------------------------------------------
# Shared fixtures/helpers (int8-quantization sections, below) -- mirrors
# tests/test_adapt_beats.py's own "helpers first, at module top" convention.
# ---------------------------------------------------------------------------


def _tiny_beats_config() -> object:
    """Tiny (2-layer) REAL vendored `BEATsConfig` at `BEATS_EMBED_DIM`=768
    (`embed_dim == encoder_embed_dim`, so `post_extract_proj` is `None`, the
    simplest case) -- 768 matches `BeatsFeaturizer`'s own hardcoded `feature_
    names()` assumption for the real (non-stub) int8 path exactly, unlike
    `tests/test_adapt_beats.py`'s own 32-dim tiny config (that file never
    calls `feature_names()`, so the mismatch never mattered there). Duplicated,
    not imported -- test modules are not a shared library other test modules
    import from in this project (`tests/test_adapt_beats.py`'s own established
    convention). Proven <1s to construct even at this width (no checkpoint
    weights loaded)."""
    from rowii.vendor.beats.BEATs import BEATsConfig

    return BEATsConfig(
        {
            "input_patch_size": 16,
            "embed_dim": BEATS_EMBED_DIM,
            "encoder_layers": 2,
            "encoder_embed_dim": BEATS_EMBED_DIM,
            "encoder_ffn_embed_dim": 128,
            "encoder_attention_heads": 8,
        }
    )


def _tiny_beats() -> object:
    from rowii.vendor.beats.BEATs import BEATs

    return BEATs(_tiny_beats_config())


def _save_tiny_quantized_beats(path: Path) -> None:
    """Builds a tiny REAL BEATs instance, dynamically quantizes it FOR REAL
    (`torch.ao.quantization.quantize_dynamic`, Linear layers, qint8,
    `inplace=True` -- see `rowii.signals.beats_model.load_quantized_beats_
    model`'s own module-level docstring note for why `inplace=True` is
    load-bearing: the vendored `TransformerEncoder` wraps its positional conv
    in `weight_norm`, which `quantize_dynamic`'s default deepcopy-based path
    cannot handle), and `torch.save()`s the quantized MODULE (not a state
    dict) to *path* -- mirrors exactly what `scripts/quantize_beats.py` itself
    does, so these tests exercise a genuinely representative artifact, not a
    hand-rolled stand-in."""
    model = _tiny_beats()
    model.eval()  # type: ignore[attr-defined]
    torch.backends.quantized.engine = select_quantized_engine()
    torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
    )
    torch.save(model, path)

# ---------------------------------------------------------------------------
# best_device
# ---------------------------------------------------------------------------


def test_best_device_honours_rowii_force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROWII_FORCE_CPU", "1")

    device = best_device()

    assert device.type == "cpu"


def test_best_device_without_force_cpu_prefers_mps_or_cuda_over_cpu_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On this dev machine (Apple Silicon, MPS available, no CUDA), the default
    (no ROWII_FORCE_CPU set) must resolve to MPS, matching the documented
    priority `ROWII_FORCE_CPU env > mps > cuda > cpu`."""
    monkeypatch.delenv("ROWII_FORCE_CPU", raising=False)

    device = best_device()

    if torch.backends.mps.is_available():
        assert device.type == "mps"
    elif torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


# ---------------------------------------------------------------------------
# stub-encoder tests
# ---------------------------------------------------------------------------


class _StubEncoder:
    """Deterministic stand-in for the real frozen BEATs encoder.

    Returns a fixed-width (n_frames, EMBED_DIM) tensor of the fbank's own
    per-frame mean, broadcast across the embedding dimension -- enough to
    let tests assert real shape/pooling/mono-mix behavior of
    `BeatsFeaturizer.transform` without loading any checkpoint weights.
    """

    EMBED_DIM = 8

    def extract(self, fbank: torch.Tensor) -> torch.Tensor:
        per_frame_mean = fbank.mean(dim=-1, keepdim=True)  # (n_frames, 1)
        return per_frame_mean.expand(-1, self.EMBED_DIM)  # (n_frames, EMBED_DIM)


def _make_featurizer() -> BeatsFeaturizer:
    return BeatsFeaturizer(
        checkpoint=None,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        encoder=_StubEncoder(),
    )


def test_beats_featurizer_has_the_documented_name() -> None:
    feat = _make_featurizer()

    assert feat.name == "beats"


def test_beats_featurizer_feature_names_before_transform_raises_with_a_stub_encoder() -> None:
    """Mirrors `VibFeaturizer.feature_names()`'s established pattern
    (`rowii.signals.features`): with an injected stub encoder, the embedding
    width is only known once the stub has actually run, so calling
    `feature_names()` first must fail loudly rather than guess."""
    feat = _make_featurizer()

    with pytest.raises(RuntimeError, match="transform"):
        feat.feature_names()


def test_beats_featurizer_feature_names_length_matches_embed_dim_after_transform() -> None:
    rate_hz = 50_000.0
    windows = np.random.default_rng(9).standard_normal((1, int(rate_hz), 1)).astype(np.float32)
    feat = _make_featurizer()

    feat.transform(windows, rate_hz)
    names = feat.feature_names()

    assert names == [f"beats_{i}" for i in range(_StubEncoder.EMBED_DIM)]


def test_beats_featurizer_transform_output_shape_is_w_by_embed_dim() -> None:
    rate_hz = 50_000.0
    n_windows, n_samples, n_channels = 3, int(rate_hz), 2
    windows = np.random.default_rng(0).standard_normal(
        (n_windows, n_samples, n_channels)
    ).astype(np.float32)
    feat = _make_featurizer()

    out = feat.transform(windows, rate_hz)

    assert out.shape == (n_windows, _StubEncoder.EMBED_DIM)
    assert out.dtype == np.float64


def test_beats_featurizer_transform_output_is_finite() -> None:
    rate_hz = 50_000.0
    windows = np.random.default_rng(1).standard_normal(
        (2, int(rate_hz), 2)
    ).astype(np.float32)
    feat = _make_featurizer()

    out = feat.transform(windows, rate_hz)

    assert np.isfinite(out).all()


def test_beats_featurizer_mono_mix_is_mean_over_channels_not_first_channel() -> None:
    """Two channels with genuinely different content: the mono mix the
    featurizer resamples/fbanks must be their MEAN, not (e.g.) just channel 0
    -- verified indirectly by comparing against a hand-mixed single-channel
    input through the same pipeline, using the stub's fbank-mean-passthrough
    behavior as the observable.
    """
    rate_hz = 50_000.0
    n_samples = int(rate_hz)
    rng = np.random.default_rng(2)
    ch0 = rng.standard_normal(n_samples).astype(np.float32)
    ch1 = rng.standard_normal(n_samples).astype(np.float32)
    two_channel = np.stack([ch0, ch1], axis=-1)[np.newaxis, :, :]  # (1, S, 2)
    hand_mixed = ((ch0 + ch1) / 2.0)[np.newaxis, :, np.newaxis]  # (1, S, 1)

    feat_two_channel = _make_featurizer()
    feat_hand_mixed = _make_featurizer()

    out_two_channel = feat_two_channel.transform(two_channel, rate_hz)
    out_hand_mixed = feat_hand_mixed.transform(hand_mixed, rate_hz)

    np.testing.assert_allclose(out_two_channel, out_hand_mixed, atol=1e-4)


def test_beats_featurizer_mono_mix_differs_from_single_channel_alone() -> None:
    """Negative control for the mono-mix test above: if the featurizer wrongly
    used only channel 0 (ignoring channel 1 entirely), this test would
    spuriously pass the assertion above too -- so also assert the mixed
    result is NOT equal to channel 0 alone, ruling out that bug directly.
    """
    rate_hz = 50_000.0
    n_samples = int(rate_hz)
    rng = np.random.default_rng(3)
    ch0 = rng.standard_normal(n_samples).astype(np.float32)
    ch1 = rng.standard_normal(n_samples).astype(np.float32) * 5.0  # very different scale
    two_channel = np.stack([ch0, ch1], axis=-1)[np.newaxis, :, :]
    channel0_alone = ch0[np.newaxis, :, np.newaxis]

    feat_two_channel = _make_featurizer()
    feat_channel0 = _make_featurizer()

    out_two_channel = feat_two_channel.transform(two_channel, rate_hz)
    out_channel0 = feat_channel0.transform(channel0_alone, rate_hz)

    assert not np.allclose(out_two_channel, out_channel0, atol=1e-4)


def test_beats_featurizer_resamples_10khz_vib_rate_without_hard_failing() -> None:
    """`BeatsFeaturizer` is documented as audio-branch only, but must not
    hard-fail on other input rates (e.g. the 10 kHz vibration rate) -- it
    should resample to 16 kHz like any other rate.
    """
    rate_hz = 10_000.0
    windows = np.random.default_rng(4).standard_normal(
        (2, int(rate_hz), 1)
    ).astype(np.float32)
    feat = _make_featurizer()

    out = feat.transform(windows, rate_hz)

    assert out.shape == (2, _StubEncoder.EMBED_DIM)
    assert np.isfinite(out).all()


def test_beats_featurizer_device_selection_defaults_to_best_device_when_unset() -> None:
    """When `device=None` is passed explicitly (the constructor default), the
    featurizer must resolve via `best_device()` -- observable here through
    `ROWII_FORCE_CPU`, since injecting an `encoder=` stub means no real model
    is ever moved to a device, so this only tests the resolved attribute."""
    old = os.environ.get("ROWII_FORCE_CPU")
    os.environ["ROWII_FORCE_CPU"] = "1"
    try:
        feat = BeatsFeaturizer(
            checkpoint=None,  # type: ignore[arg-type]
            device=None,
            encoder=_StubEncoder(),
        )
    finally:
        if old is None:
            os.environ.pop("ROWII_FORCE_CPU", None)
        else:
            os.environ["ROWII_FORCE_CPU"] = old

    assert feat.device.type == "cpu"


# ---------------------------------------------------------------------------
# quantized backend engine selection (rowii.signals.beats_model, spec D6)
# ---------------------------------------------------------------------------


class TestSelectQuantizedEngine:
    def test_prefers_fbgemm_when_available(self) -> None:
        assert select_quantized_engine(["qnnpack", "fbgemm", "onednn"]) == "fbgemm"

    def test_prefers_x86_over_qnnpack(self) -> None:
        assert select_quantized_engine(["qnnpack", "x86"]) == "x86"

    def test_falls_back_to_qnnpack_when_neither_fbgemm_nor_x86_available(self) -> None:
        assert select_quantized_engine(["none", "qnnpack"]) == "qnnpack"

    def test_falls_back_to_any_remaining_non_none_engine(self) -> None:
        assert select_quantized_engine(["none", "onednn"]) == "onednn"

    def test_raises_when_only_none_is_supported(self) -> None:
        with pytest.raises(RuntimeError, match="no usable"):
            select_quantized_engine(["none"])

    def test_default_reads_the_real_torch_supported_engines(self) -> None:
        # No override -- exercises the real torch.backends.quantized.
        # supported_engines path (whatever this machine's torch build
        # actually supports; every dev/CI machine has at least ONE real
        # backend compiled in, verified indirectly by this not raising).
        engine = select_quantized_engine()
        assert engine in torch.backends.quantized.supported_engines


# ---------------------------------------------------------------------------
# load_quantized_beats_model (rowii.signals.beats_model, spec D6)
# ---------------------------------------------------------------------------


class TestLoadQuantizedBeatsModel:
    def test_missing_checkpoint_raises_file_not_found_naming_the_path(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "does-not-exist.pt"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_quantized_beats_model(missing)
        assert str(missing) in str(exc_info.value)

    def test_reloads_and_forward_runs_on_cpu(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "beats_int8.pt"
        _save_tiny_quantized_beats(ckpt)

        model = load_quantized_beats_model(ckpt)

        assert model.training is False
        probe = torch.randn(2, 16_000)
        with torch.no_grad():
            features, _ = model.extract_features(probe)
        assert features.shape[0] == 2
        assert features.shape[-1] == BEATS_EMBED_DIM
        assert torch.isfinite(features).all()


# ---------------------------------------------------------------------------
# BeatsFeaturizer(int8_model_path=...) branch (spec D6)
# ---------------------------------------------------------------------------


class TestBeatsFeaturizerInt8Branch:
    def test_forces_cpu_even_with_an_explicit_non_cpu_device(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "beats_int8.pt"
        _save_tiny_quantized_beats(ckpt)
        # An explicit MPS device= would be nonsensical for a dynamically
        # quantized module (CPU-only kernels) -- the int8 branch must force
        # CPU regardless of what the caller passes. Falls back to a
        # deliberately non-CPU "meta" device on machines without MPS, so this
        # assertion is meaningful either way.
        requested = (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("meta")
        )

        feat = BeatsFeaturizer(
            checkpoint=None,  # type: ignore[arg-type]  -- unused on the int8 path
            device=requested,
            int8_model_path=ckpt,
        )

        assert feat.device.type == "cpu"

    def test_transform_runs_the_real_pipeline_and_matches_embed_dim(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "beats_int8.pt"
        _save_tiny_quantized_beats(ckpt)
        feat = BeatsFeaturizer(checkpoint=None, int8_model_path=ckpt)  # type: ignore[arg-type]

        rate_hz = 50_000.0
        windows = np.random.default_rng(11).standard_normal(
            (2, int(rate_hz), 2)
        ).astype(np.float32)

        out = feat.transform(windows, rate_hz)

        assert out.shape == (2, BEATS_EMBED_DIM)
        assert np.isfinite(out).all()
        assert feat.feature_names() == [f"beats_{i}" for i in range(BEATS_EMBED_DIM)]

    def test_feature_names_available_immediately_without_calling_transform_first(
        self, tmp_path: Path
    ) -> None:
        """Unlike the stub-encoder path (embed dim discovered from transform()'s
        own first call), the real int8 path -- like the real fp32 path -- knows
        its embedding width upfront (BEATS_EMBED_DIM, the standard BEATs-base
        architecture every real checkpoint this project quantizes FROM) --
        feature_names() must be callable right after construction."""
        ckpt = tmp_path / "beats_int8.pt"
        _save_tiny_quantized_beats(ckpt)

        feat = BeatsFeaturizer(checkpoint=None, int8_model_path=ckpt)  # type: ignore[arg-type]

        assert feat.feature_names() == [f"beats_{i}" for i in range(BEATS_EMBED_DIM)]

    def test_missing_int8_checkpoint_file_raises_clear_error_naming_the_path(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "does-not-exist.pt"

        with pytest.raises(FileNotFoundError) as exc_info:
            BeatsFeaturizer(checkpoint=None, int8_model_path=missing)  # type: ignore[arg-type]
        assert str(missing) in str(exc_info.value)

    def test_stores_int8_model_path_and_checkpoint_for_introspection(self, tmp_path: Path) -> None:
        # Mirrors TfcFeaturizer's/StudentFeaturizer's own established
        # `._checkpoint` introspection attribute (tests/test_pipeline.py's own
        # dispatch tests read it directly) -- BeatsFeaturizer eager-loads, so
        # neither attribute is needed for transform() to work, but storing them
        # keeps the class's own introspection story consistent regardless.
        ckpt = tmp_path / "beats_int8.pt"
        _save_tiny_quantized_beats(ckpt)
        beats_ckpt = tmp_path / "unused_fp32.pt"

        feat = BeatsFeaturizer(checkpoint=beats_ckpt, int8_model_path=ckpt)

        assert feat._int8_model_path == ckpt
        assert feat._checkpoint == beats_ckpt


# ---------------------------------------------------------------------------
# real-checkpoint smoke test
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_beats_featurizer_real_checkpoint_synthetic_audio_smoke() -> None:
    """3 windows of 1-s synthetic audio at 50 kHz through the REAL frozen
    checkpoint -> (3, 768), finite, non-constant across windows (rules out a
    degenerate all-same-embedding forward pass)."""
    from rowii.config import load_config

    cfg = load_config()
    if cfg.beats_checkpoint is None:
        pytest.skip("ROWII_BEATS_CHECKPOINT not set")

    rate_hz = 50_000.0
    n_windows, n_samples, n_channels = 3, int(rate_hz), 2
    rng = np.random.default_rng(42)
    windows = rng.standard_normal((n_windows, n_samples, n_channels)).astype(np.float32)

    feat = BeatsFeaturizer(checkpoint=cfg.beats_checkpoint)
    out = feat.transform(windows, rate_hz)

    assert out.shape == (3, 768)
    assert np.isfinite(out).all()
    # Non-constant: different synthetic-noise windows must not collapse to
    # identical embeddings.
    assert not np.allclose(out[0], out[1])
    assert not np.allclose(out[1], out[2])
