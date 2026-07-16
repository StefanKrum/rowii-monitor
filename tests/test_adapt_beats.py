"""Tests for `scripts/adapt_beats.py` (Step-2 package-5 spec D4, Task 3):
CLI-level end-to-end tests against a monkeypatched `discover`/`load_beats_model`/
`iter_target_windows` (mirrors `tests/test_warm_cache.py`'s/`tests/
test_pretrain_tfc.py`'s own established patterns -- no real data tree, no real
BEATs checkpoint, no network anywhere in this file) plus focused unit tests for
the pieces of new logic that have no natural home in Task 1/2's own test files:
the frame-preserving bridge (`_build_frame_bridge`/`_encoder_forward`), mode
preparation/optimizer-param scoping (`_prepare_model_for_mode`/
`_trainable_params`), and mode-default resolution (`_resolve_mode_defaults`).

`load_beats_model` is monkeypatched to a stub that distinguishes its TWO
distinct call sites by whether *checkpoint* exists on disk yet: the initial
"load the base checkpoint" call (a deliberately nonexistent fake path) gets a
freshly-constructed TINY REAL `rowii.vendor.beats.BEATs.BEATs` instance (same
tiny config `tests/test_adapt_lora.py::test_structural_match_on_real_vendored_
class` uses -- 2 layers, 32-dim, proven <2s to construct); the later
reload-verification call (a file `adapt_beats.py` itself just wrote) delegates
to the REAL `rowii.signals.beats_model.load_beats_model`, so the reload check
genuinely exercises `torch.save`/`torch.load`/`load_state_dict` fidelity
against that real file, not a second canned instance.

CPU-forced throughout (`ROWII_FORCE_CPU=1`), matching every other eager-torch
test module in this repo.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rowii.adapt.lora import LoraLinear, lora_parameters  # noqa: E402
from rowii.adapt.objective import masked_patch_loss  # noqa: E402
from rowii.config import Config  # noqa: E402
from rowii.io.dataset import RecordingIndex, Run  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import adapt_beats  # noqa: E402


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setenv("ROWII_FORCE_CPU", "1")


# ---------------------------------------------------------------------------
# Shared fixtures/helpers
# ---------------------------------------------------------------------------


def _tiny_beats_config():
    from rowii.vendor.beats.BEATs import BEATsConfig

    # Identical to tests/test_adapt_lora.py::test_structural_match_on_real_
    # vendored_class's own tiny config (embed_dim == encoder_embed_dim == 32,
    # so BEATs.__init__ sets post_extract_proj=None -- proven <2s to
    # construct, no checkpoint weights involved).
    return BEATsConfig(
        {
            "input_patch_size": 16,
            "embed_dim": 32,
            "encoder_layers": 2,
            "encoder_embed_dim": 32,
            "encoder_ffn_embed_dim": 64,
            "encoder_attention_heads": 4,
        }
    )


def _tiny_beats():
    from rowii.vendor.beats.BEATs import BEATs

    return BEATs(_tiny_beats_config())


def _fake_index(run_names: list[str]) -> RecordingIndex:
    """Mirrors `tests/test_warm_cache.py::_fake_index` exactly (duplicated,
    not imported -- test modules are not a shared library other test modules
    import from in this project, per that module's own established
    convention)."""
    runs = [Run(name=name, files={}, day_root=Path("/fake-day-root")) for name in run_names]
    return RecordingIndex(runs=runs, betriebsdaten=[], betriebsdaten_by_day={})


def _stub_load_beats_model(checkpoint, device):
    """Monkeypatch target for `adapt_beats.load_beats_model` (module
    docstring / this file's own docstring): a *checkpoint* that does not
    exist yet (this file's fake `ROWII_BEATS_CHECKPOINT` value) -> a fresh
    tiny REAL BEATs instance; a *checkpoint* that DOES exist (a file
    `adapt_beats.py` itself just wrote, during `_save_and_verify`'s reload
    step) -> the REAL loader, genuinely exercising save/load fidelity.
    """
    if not checkpoint.exists():
        return _tiny_beats().to(device)
    from rowii.signals.beats_model import load_beats_model as real_load_beats_model

    return real_load_beats_model(checkpoint, device)


def _stub_iter_target_windows(run, cfg, *, target_hz=16_000, seed=7, max_windows=None):
    """Monkeypatch target for `adapt_beats.iter_target_windows`: *n* synthetic
    `(target_hz,)` float32 windows, deterministic given *seed* (own `rng`,
    independent of torch's global RNG) -- ignores *run*/*cfg* entirely (no
    real gantner tree anywhere in this file)."""
    n = 24 if max_windows is None else min(24, max_windows)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield rng.normal(0.0, 1.0, target_hz).astype(np.float32)


def _patch_for_e2e(monkeypatch, tmp_path, *, run_name: str = "test-run") -> None:
    monkeypatch.setattr(adapt_beats, "discover", lambda data_root: _fake_index([run_name]))
    monkeypatch.setattr(adapt_beats, "load_beats_model", _stub_load_beats_model)
    monkeypatch.setattr(adapt_beats, "iter_target_windows", _stub_iter_target_windows)
    monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", str(tmp_path / "fake_base_checkpoint.pt"))
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))


# ---------------------------------------------------------------------------
# 1. --help / parser basics (mirrors tests/test_pretrain_tfc.py's own section)
# ---------------------------------------------------------------------------


def test_help_exits_zero_and_documents_every_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        adapt_beats.main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--mode", "--run", "--epochs", "--batch-size", "--lr",
        "--seed", "--max-windows", "--out",
    ):
        assert flag in out, f"missing {flag!r} in --help output"


def test_mode_and_run_are_required():
    with pytest.raises(SystemExit) as exc_info:
        adapt_beats.build_parser().parse_args([])
    assert exc_info.value.code == 2


def test_invalid_mode_choice_exits_2():
    with pytest.raises(SystemExit) as exc_info:
        adapt_beats.build_parser().parse_args(["--mode", "bogus", "--run", "x"])
    assert exc_info.value.code == 2


def test_build_parser_flat_defaults():
    args = adapt_beats.build_parser().parse_args(["--mode", "lora", "--run", "x"])
    assert args.batch_size == 16
    assert args.seed == 7
    assert args.max_windows == 8000
    assert args.out == Path("models/adapted/")
    # epochs/lr default to None at the PARSE level -- _resolve_mode_defaults
    # is what fills in the mode-specific value (contract (e) below).
    assert args.epochs is None
    assert args.lr is None


# ---------------------------------------------------------------------------
# 2. Mode-default resolution (contract (e): "mode defaults applied when
#    flags omitted", both as parse-level assertions and via the dedicated
#    resolver `_resolve_mode_defaults` unit-tested in isolation)
# ---------------------------------------------------------------------------


class TestResolveModeDefaults:
    def test_lora_defaults_when_omitted(self):
        epochs, lr = adapt_beats._resolve_mode_defaults("lora", None, None)
        assert epochs == 5
        assert lr == pytest.approx(1e-4)

    def test_full_defaults_when_omitted(self):
        epochs, lr = adapt_beats._resolve_mode_defaults("full", None, None)
        assert epochs == 2
        assert lr == pytest.approx(1e-5)

    def test_explicit_epochs_wins_lr_still_defaults(self):
        epochs, lr = adapt_beats._resolve_mode_defaults("lora", 3, None)
        assert epochs == 3
        assert lr == pytest.approx(1e-4)

    def test_explicit_lr_wins_epochs_still_defaults(self):
        epochs, lr = adapt_beats._resolve_mode_defaults("full", None, 2e-3)
        assert epochs == 2
        assert lr == pytest.approx(2e-3)

    def test_both_explicit_flags_win(self):
        epochs, lr = adapt_beats._resolve_mode_defaults("lora", 9, 5e-2)
        assert epochs == 9
        assert lr == pytest.approx(5e-2)

    def test_parser_epochs_lr_are_none_until_resolved(self):
        """Parse-level half of contract (e): argparse itself must hand back
        None for an omitted --epochs/--lr (never silently substituting a
        flat default at the parser level) -- _resolve_mode_defaults is the
        ONLY place the mode-specific value is filled in."""
        args = adapt_beats.build_parser().parse_args(
            ["--mode", "full", "--run", "x", "--epochs", "7"]
        )
        assert args.epochs == 7
        assert args.lr is None
        epochs, lr = adapt_beats._resolve_mode_defaults("full", args.epochs, args.lr)
        assert (epochs, lr) == (7, pytest.approx(1e-5))


# ---------------------------------------------------------------------------
# 3. Import/checkpoint guards
# ---------------------------------------------------------------------------


def test_import_beats_or_exit_raises_systemexit_with_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rowii.signals.beats":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit) as exc_info:
        adapt_beats._import_beats_or_exit()

    message = str(exc_info.value)
    assert "pip install -e" in message
    assert "ROWII_BEATS_CHECKPOINT" in message


def test_require_beats_checkpoint_exits_with_hint_when_unset():
    cfg = Config(data_root=Path("."), results_root=Path("."), beats_checkpoint=None)

    with pytest.raises(SystemExit) as exc_info:
        adapt_beats._require_beats_checkpoint(cfg)

    assert "ROWII_BEATS_CHECKPOINT" in str(exc_info.value)


def test_require_beats_checkpoint_returns_path_when_set():
    checkpoint = Path("/fake/beats.pt")
    cfg = Config(data_root=Path("."), results_root=Path("."), beats_checkpoint=checkpoint)

    assert adapt_beats._require_beats_checkpoint(cfg) == checkpoint


# ---------------------------------------------------------------------------
# 4. Frame-preserving bridge + mode preparation (unit level, no CLI)
# ---------------------------------------------------------------------------


class TestPrepareModelForMode:
    def test_lora_freezes_base_and_injects_only_qv_adapters(self):
        model = _tiny_beats()

        n_injected = adapt_beats._prepare_model_for_mode("lora", model)

        assert n_injected == 2 * model.cfg.encoder_layers
        for layer in model.encoder.layers:
            assert isinstance(layer.self_attn.q_proj, LoraLinear)
            assert isinstance(layer.self_attn.v_proj, LoraLinear)
            assert not layer.self_attn.k_proj.weight.requires_grad
            assert not layer.self_attn.out_proj.weight.requires_grad
        assert not model.patch_embedding.weight.requires_grad
        lora_ids = {id(p) for p in lora_parameters(model)}
        assert lora_ids  # non-empty
        assert all(p.requires_grad for p in lora_parameters(model))

    def test_full_marks_every_parameter_trainable(self):
        model = _tiny_beats()
        for p in model.parameters():
            p.requires_grad_(False)  # start frozen, prove "full" flips ALL of it

        n_injected = adapt_beats._prepare_model_for_mode("full", model)

        assert n_injected == 0
        assert all(p.requires_grad for p in model.parameters())


class TestTrainableParams:
    def test_lora_is_adapters_and_head_only(self):
        model = _tiny_beats()
        adapt_beats._prepare_model_for_mode("lora", model)
        head = torch.nn.Linear(model.cfg.encoder_embed_dim, 128)

        params = adapt_beats._trainable_params("lora", model, head)

        expected = {id(p) for p in lora_parameters(model)} | {id(p) for p in head.parameters()}
        assert {id(p) for p in params} == expected
        assert len(params) == len(expected)  # no duplicates
        # sanity: a strict, much smaller subset of the model's own full param set
        assert len(params) < len(list(model.parameters())) + len(list(head.parameters()))

    def test_full_is_every_model_param_and_head(self):
        model = _tiny_beats()
        adapt_beats._prepare_model_for_mode("full", model)
        head = torch.nn.Linear(model.cfg.encoder_embed_dim, 128)

        params = adapt_beats._trainable_params("full", model, head)

        expected = {id(p) for p in model.parameters()} | {id(p) for p in head.parameters()}
        assert {id(p) for p in params} == expected
        assert len(params) == len(expected)


class TestFrameBridgeAndEncoderForward:
    def test_bridge_is_frozen_linear_with_expected_shape(self):
        bridge = adapt_beats._build_frame_bridge(128, 32, torch.device("cpu"))

        assert isinstance(bridge, torch.nn.Linear)
        assert bridge.in_features == 128
        assert bridge.out_features == 32
        assert not bridge.weight.requires_grad
        assert not bridge.bias.requires_grad

    def test_encoder_forward_preserves_frame_axis(self):
        """The load-bearing shape contract masked_patch_loss depends on
        (module docstring): encoder_forward's OUTPUT frame count must equal
        its INPUT frame count, unlike BEATs' own patch_embedding."""
        model = _tiny_beats()
        bridge = adapt_beats._build_frame_bridge(
            128, model.cfg.encoder_embed_dim, torch.device("cpu")
        )
        encoder_forward = adapt_beats._encoder_forward(model, bridge)

        fbank = torch.randn(3, 41, 128)
        out = encoder_forward(fbank)

        assert out.shape == (3, 41, model.cfg.encoder_embed_dim)


# ---------------------------------------------------------------------------
# 5. LoRA-adapter gradient flow against a REAL attention encoder (contract
#    (b): the concrete counterpart to rowii.adapt.objective's own documented
#    caveat that no position-wise-encoder test can prove this)
# ---------------------------------------------------------------------------


def test_lora_adapter_grads_nonzero_against_real_tiny_encoder():
    """`rowii.adapt.objective.masked_patch_loss`'s own docstring: a
    POSITION-WISE encoder_forward (e.g. that module's own test suite's bare
    nn.Linear) provably yields zero WEIGHT gradient at masked positions --
    masked frames are zeroed before the encoder ever sees them, and a
    position-wise map never mixes information across positions, so nothing
    downstream of a masked position can depend on any OTHER position's
    weights. A REAL multi-head self-attention encoder is different: a masked
    query attends to unmasked keys/values, so information genuinely crosses
    positions. This test proves gradient reaches the LoRA adapters when
    adapt_beats.py's frame-preserving bridge feeds masked fbank frames into
    the REAL vendored TransformerEncoder.

    Two backward passes, not one -- LoRA's own zero-init for lora_b (Task 1
    design: "B init zeros") makes lora_a's gradient EXACTLY zero at the very
    first step, by the chain rule alone (dLoss/dA = dLoss/dB_out @ B.weight,
    and B.weight is all-zero pre-training), REGARDLESS of which encoder is
    downstream -- a well-known LoRA training-dynamics property, not a
    position-wise-encoder artifact. lora_b's OWN gradient is unaffected by
    this (a Linear's weight-grad depends on the upstream grad and its INPUT
    activation, not its own current value), so it IS already nonzero at step
    1. One optimizer step moves lora_b off zero; the second backward pass
    then shows lora_a's gradient becomes nonzero too -- proving the FULL
    adapter (both A and B), not just B, genuinely trains through the real
    encoder.
    """
    torch.manual_seed(0)
    model = _tiny_beats()
    adapt_beats._prepare_model_for_mode("lora", model)
    first_lora = next(m for m in model.encoder.modules() if isinstance(m, LoraLinear))

    waveform = torch.randn(2, 16_000)
    with torch.no_grad():
        fbank = model.preprocess(waveform)
    bridge = adapt_beats._build_frame_bridge(
        fbank.shape[-1], model.cfg.encoder_embed_dim, torch.device("cpu")
    )
    head = torch.nn.Linear(model.cfg.encoder_embed_dim, fbank.shape[-1])
    encoder_forward = adapt_beats._encoder_forward(model, bridge)

    gen = torch.Generator().manual_seed(7)
    loss = masked_patch_loss(encoder_forward, fbank, head, generator=gen)
    loss.backward()

    assert first_lora.lora_b.weight.grad is not None
    assert torch.any(first_lora.lora_b.weight.grad != 0)
    assert first_lora.lora_a.weight.grad is not None
    assert torch.all(first_lora.lora_a.weight.grad == 0)  # expected LoRA-at-init behaviour

    opt = torch.optim.SGD(lora_parameters(model), lr=0.1)
    opt.step()
    opt.zero_grad()

    gen2 = torch.Generator().manual_seed(11)
    loss2 = masked_patch_loss(encoder_forward, fbank, head, generator=gen2)
    loss2.backward()

    assert first_lora.lora_a.weight.grad is not None
    assert torch.any(first_lora.lora_a.weight.grad != 0)

    # The contract's own literal ask, restated generically over every
    # injected adapter (true from step 1 already, via lora_b above).
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in lora_parameters(model)
    )


# ---------------------------------------------------------------------------
# 6. End-to-end CLI: lora mode (contract (a))
# ---------------------------------------------------------------------------


def test_e2e_lora_mode_checkpoint_sidecar_and_reload_check(tmp_path, monkeypatch, caplog):
    _patch_for_e2e(monkeypatch, tmp_path)
    out_dir = tmp_path / "out"

    with caplog.at_level(logging.INFO):
        exit_code = adapt_beats.main(
            [
                "--mode", "lora", "--run", "test-run", "--epochs", "1",
                "--batch-size", "8", "--max-windows", "16", "--seed", "7",
                "--out", str(out_dir),
            ]
        )

    assert exit_code == 0
    checkpoint_path = out_dir / "beats_lora_test-run.pt"
    assert checkpoint_path.is_file()

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert set(state.keys()) == {"cfg", "model"}
    assert isinstance(state["model"], dict)
    assert len(state["model"]) > 0
    # the merged checkpoint must NOT contain any lora_a/lora_b keys (D2:
    # merge_lora folds them back into plain Linears before export)
    assert not any("lora_a" in k or "lora_b" in k for k in state["model"])

    deviation_logs = [
        r.message for r in caplog.records if "reload-forward deviation" in r.message
    ]
    assert len(deviation_logs) == 1, "reload-forward deviation check did not log exactly once"

    sidecar_path = checkpoint_path.with_suffix(".json")
    assert sidecar_path.is_file()
    sidecar = json.loads(sidecar_path.read_text())
    for key in ("mode", "run", "epochs", "n_windows", "final_loss", "elapsed_s", "note"):
        assert key in sidecar, f"sidecar missing {key!r}"
    assert sidecar["mode"] == "lora"
    assert sidecar["run"] == "test-run"
    assert sidecar["epochs"] == 1
    assert sidecar["n_windows"] == 16
    assert math.isfinite(sidecar["final_loss"])
    assert sidecar["elapsed_s"] >= 0
    assert "proxy" in sidecar["note"].lower()
    assert sidecar["max_reload_deviation"] < 1e-4


# ---------------------------------------------------------------------------
# 7. End-to-end CLI: full mode smoke
# ---------------------------------------------------------------------------


def test_e2e_full_mode_smoke(tmp_path, monkeypatch):
    _patch_for_e2e(monkeypatch, tmp_path)
    out_dir = tmp_path / "out"

    exit_code = adapt_beats.main(
        [
            "--mode", "full", "--run", "test-run", "--epochs", "1",
            "--batch-size", "8", "--max-windows", "16", "--seed", "7",
            "--out", str(out_dir),
        ]
    )

    assert exit_code == 0
    checkpoint_path = out_dir / "beats_ft_test-run.pt"
    assert checkpoint_path.is_file()

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert set(state.keys()) == {"cfg", "model"}

    sidecar = json.loads(checkpoint_path.with_suffix(".json").read_text())
    assert sidecar["mode"] == "full"
    assert sidecar["epochs"] == 1
    assert math.isfinite(sidecar["final_loss"])


# ---------------------------------------------------------------------------
# 8. Determinism: two full runs, same seed -> identical checkpoint tensors
#    (contract (d))
# ---------------------------------------------------------------------------


def test_determinism_same_seed_produces_identical_lora_checkpoint(tmp_path, monkeypatch):
    _patch_for_e2e(monkeypatch, tmp_path)
    argv_base = [
        "--mode", "lora", "--run", "test-run", "--epochs", "2",
        "--batch-size", "8", "--max-windows", "24", "--seed", "7",
    ]
    out1, out2 = tmp_path / "out1", tmp_path / "out2"

    assert adapt_beats.main([*argv_base, "--out", str(out1)]) == 0
    assert adapt_beats.main([*argv_base, "--out", str(out2)]) == 0

    state1 = torch.load(
        out1 / "beats_lora_test-run.pt", map_location="cpu", weights_only=False
    )["model"]
    state2 = torch.load(
        out2 / "beats_lora_test-run.pt", map_location="cpu", weights_only=False
    )["model"]

    assert state1.keys() == state2.keys()
    compared = 0
    for key in state1:
        torch.testing.assert_close(state1[key], state2[key], rtol=0, atol=0, msg=f"key={key}")
        compared += 1
    assert compared > 0


def test_determinism_same_seed_produces_identical_full_checkpoint(tmp_path, monkeypatch):
    _patch_for_e2e(monkeypatch, tmp_path)
    argv_base = [
        "--mode", "full", "--run", "test-run", "--epochs", "1",
        "--batch-size", "8", "--max-windows", "16", "--seed", "3",
    ]
    out1, out2 = tmp_path / "out1", tmp_path / "out2"

    assert adapt_beats.main([*argv_base, "--out", str(out1)]) == 0
    assert adapt_beats.main([*argv_base, "--out", str(out2)]) == 0

    state1 = torch.load(
        out1 / "beats_ft_test-run.pt", map_location="cpu", weights_only=False
    )["model"]
    state2 = torch.load(
        out2 / "beats_ft_test-run.pt", map_location="cpu", weights_only=False
    )["model"]

    assert state1.keys() == state2.keys()
    for key in state1:
        torch.testing.assert_close(state1[key], state2[key], rtol=0, atol=0, msg=f"key={key}")


def test_determinism_different_seed_produces_different_checkpoint(tmp_path, monkeypatch):
    _patch_for_e2e(monkeypatch, tmp_path)
    argv_base = [
        "--mode", "lora", "--run", "test-run", "--epochs", "2",
        "--batch-size", "8", "--max-windows", "24",
    ]
    out1, out2 = tmp_path / "out1", tmp_path / "out2"

    assert adapt_beats.main([*argv_base, "--seed", "7", "--out", str(out1)]) == 0
    assert adapt_beats.main([*argv_base, "--seed", "8", "--out", str(out2)]) == 0

    state1 = torch.load(
        out1 / "beats_lora_test-run.pt", map_location="cpu", weights_only=False
    )["model"]
    state2 = torch.load(
        out2 / "beats_lora_test-run.pt", map_location="cpu", weights_only=False
    )["model"]

    any_different = any(not torch.equal(state1[k], state2[k]) for k in state1)
    assert any_different


# ---------------------------------------------------------------------------
# 9. Run-resolution / no-windows error paths
# ---------------------------------------------------------------------------


def test_unknown_run_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(adapt_beats, "discover", lambda data_root: _fake_index(["run-a"]))
    monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", str(tmp_path / "fake.pt"))

    exit_code = adapt_beats.main(
        ["--mode", "lora", "--run", "bogus-run", "--out", str(tmp_path / "out")]
    )

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "bogus-run" in err
    assert "run-a" in err


def test_no_windows_found_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(adapt_beats, "discover", lambda data_root: _fake_index(["test-run"]))
    monkeypatch.setattr(adapt_beats, "iter_target_windows", lambda *a, **k: iter(()))
    monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", str(tmp_path / "fake.pt"))

    exit_code = adapt_beats.main(
        ["--mode", "lora", "--run", "test-run", "--out", str(tmp_path / "out")]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "no target-normal windows" in err.lower()


def test_missing_checkpoint_env_exits_via_systemexit(tmp_path, monkeypatch):
    # load_config() falls back to the repo's own .env file whenever the
    # process env var is merely ABSENT (rowii.config.load_config's own
    # documented priority: explicit env > process env > .env file) -- this
    # repo's .env sets a real ROWII_BEATS_CHECKPOINT for dev use, so a plain
    # monkeypatch.delenv would not actually reach the "unset" branch here.
    # An explicit empty string DOES override it (process env always wins
    # over .env, and load_config's own "ckpt = merged.get(...) or None"
    # treats "" the same as unset).
    monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", "")
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))

    with pytest.raises(SystemExit) as exc_info:
        adapt_beats.main(["--mode", "lora", "--run", "test-run", "--out", str(tmp_path / "out")])

    assert "ROWII_BEATS_CHECKPOINT" in str(exc_info.value)
