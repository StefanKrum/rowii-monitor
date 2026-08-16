"""Tests for `scripts/quantize_beats.py`:
post-training INT8 dynamic quantization of the frozen BEATs encoder.

CLI-level end-to-end tests against a monkeypatched `load_beats_model` (mirrors
`tests/test_adapt_beats.py`'s/`tests/test_student.py`'s own established
pattern -- no real BEATs checkpoint, no network anywhere in this file), built
on the SAME tiny-REAL-BEATs recipe `tests/test_beats.py` uses for its own
`select_quantized_engine`/`load_quantized_beats_model`/`BeatsFeaturizer`
int8-branch tests, plus focused unit tests for `cosine_drift`, the
embedding-drift helper exposed for the execution phase.

`-m "not data"` only; no real checkpoint, no network anywhere in this file.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import quantize_beats  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures/helpers
# ---------------------------------------------------------------------------


def _tiny_beats_config():
    # Duplicated from tests/test_adapt_beats.py/tests/test_beats.py -- test
    # modules are not a shared library other test modules import from in this
    # project (tests/test_adapt_beats.py's own established convention). A
    # small (32-dim) config suffices here: this file never calls
    # feature_names()/checks embed width, unlike tests/test_beats.py's own
    # 768-dim tiny config.
    from rowii.vendor.beats.BEATs import BEATsConfig

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


def _stub_load_beats_model(checkpoint, device):
    """Monkeypatch target for `quantize_beats.load_beats_model`: ignores
    *checkpoint*'s actual content (a dummy file this module's own tests
    create purely so `Path.stat()` has something real to report for
    `size_fp32_bytes`) and returns a fresh tiny REAL BEATs instance -- mirrors
    `tests/test_adapt_beats.py`'s own `_stub_load_beats_model`, simplified:
    unlike `adapt_beats.py`, this script never reloads a checkpoint a second
    time, so there is no exists()-based branching to mirror.
    """
    return _tiny_beats().to(device)


def _patch_beats_checkpoint(monkeypatch, tmp_path: Path, *, size_bytes: int = 4096) -> Path:
    """Writes a dummy fp32 checkpoint file at a controlled tmp_path (so
    `checkpoint.stat().st_size` in `main()` reports a REAL, controlled
    number), points `ROWII_BEATS_CHECKPOINT` at it, and monkeypatches
    `load_beats_model` -- mirrors `tests/test_student.py`'s/`tests/
    test_adapt_beats.py`'s own defensive env-neutralization: this repo's own
    `.env` sets a REAL `ROWII_BEATS_CHECKPOINT` for dev use (process env >
    `.env`, but only once the KEY is present), so every test must set every
    relevant env var explicitly rather than relying on defaults.
    """
    monkeypatch.setattr(quantize_beats, "load_beats_model", _stub_load_beats_model)
    checkpoint = tmp_path / "fake_beats_fp32.pt"
    checkpoint.write_bytes(b"\x00" * size_bytes)
    monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    monkeypatch.delenv("ROWII_BEATS_INT8_CHECKPOINT", raising=False)
    return checkpoint


# ---------------------------------------------------------------------------
# 1. --help / parser basics (mirrors tests/test_adapt_beats.py's own section)
# ---------------------------------------------------------------------------


def test_help_exits_zero_and_documents_every_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        quantize_beats.main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--out" in out


def test_build_parser_default_out():
    args = quantize_beats.build_parser().parse_args([])
    assert args.out == Path("models/adapted/beats_int8.pt")


def test_build_parser_out_override():
    args = quantize_beats.build_parser().parse_args(["--out", "/tmp/x/beats_int8.pt"])
    assert args.out == Path("/tmp/x/beats_int8.pt")


# ---------------------------------------------------------------------------
# 2. main() end-to-end: smoke test with a monkeypatched tiny model
# ---------------------------------------------------------------------------


def test_main_saves_a_reloadable_forward_running_quantized_module(
    tmp_path, monkeypatch, caplog
):
    fp32_checkpoint = _patch_beats_checkpoint(monkeypatch, tmp_path)
    out_path = tmp_path / "out" / "beats_int8.pt"

    with caplog.at_level(logging.INFO):
        rc = quantize_beats.main(["--out", str(out_path)])

    assert rc == 0
    assert out_path.is_file()

    # Reloads and forward-runs on CPU.
    reloaded = torch.load(out_path, map_location="cpu", weights_only=False)
    reloaded.eval()
    probe = torch.randn(2, 16_000)
    with torch.no_grad():
        features, _ = reloaded.extract_features(probe)
    assert features.shape[0] == 2
    assert torch.isfinite(features).all()

    # Sidecar: exactly the 4 contracted keys, sane values.
    sidecar_path = out_path.with_suffix(".json")
    assert sidecar_path.is_file()
    sidecar = json.loads(sidecar_path.read_text())
    assert set(sidecar) == {
        "source_checkpoint", "size_fp32_bytes", "size_int8_bytes", "created_at",
    }
    assert sidecar["source_checkpoint"] == str(fp32_checkpoint)
    assert sidecar["size_fp32_bytes"] == fp32_checkpoint.stat().st_size
    assert sidecar["size_int8_bytes"] == out_path.stat().st_size
    assert sidecar["size_int8_bytes"] > 0
    assert isinstance(sidecar["created_at"], str) and sidecar["created_at"]

    # Parameter count and both on-disk sizes logged.
    messages = [r.getMessage() for r in caplog.records]
    assert any("quantize_beats" in m for m in messages)
    assert any("parameter" in m.lower() for m in messages)
    assert any("MB" in m for m in messages)


def test_main_writes_out_directory_when_missing(tmp_path, monkeypatch):
    _patch_beats_checkpoint(monkeypatch, tmp_path)
    out_path = tmp_path / "does" / "not" / "exist" / "beats_int8.pt"

    rc = quantize_beats.main(["--out", str(out_path)])

    assert rc == 0
    assert out_path.is_file()


def test_main_missing_beats_checkpoint_env_exits(tmp_path, monkeypatch):
    monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", "")
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))

    with pytest.raises(SystemExit) as exc_info:
        quantize_beats.main([])
    assert "ROWII_BEATS_CHECKPOINT" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. cosine_drift
# ---------------------------------------------------------------------------


class TestCosineDrift:
    def test_identical_rows_give_similarity_one(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(5, 8))

        assert quantize_beats.cosine_drift(a, a.copy()) == pytest.approx(1.0)

    def test_orthogonal_rows_give_similarity_zero(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[0.0, 1.0], [1.0, 0.0]])

        assert quantize_beats.cosine_drift(a, b) == pytest.approx(0.0)

    def test_mean_over_rows_not_just_the_first(self):
        a = np.array([[1.0, 0.0], [1.0, 0.0]])
        b = np.array([[1.0, 0.0], [0.0, 1.0]])  # row 0 identical, row 1 orthogonal

        assert quantize_beats.cosine_drift(a, b) == pytest.approx(0.5)

    def test_opposite_rows_give_similarity_minus_one(self):
        a = np.array([[1.0, 2.0]])
        b = -a

        assert quantize_beats.cosine_drift(a, b) == pytest.approx(-1.0)

    def test_scale_invariant(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(4, 6))
        b = a * 3.7

        assert quantize_beats.cosine_drift(a, b) == pytest.approx(1.0)

    def test_shape_mismatch_raises_value_error(self):
        a = np.zeros((3, 4))
        b = np.zeros((3, 5))

        with pytest.raises(ValueError, match="shape"):
            quantize_beats.cosine_drift(a, b)

    def test_zero_norm_row_does_not_raise_or_produce_nan(self):
        a = np.array([[0.0, 0.0], [1.0, 0.0]])
        b = np.array([[1.0, 1.0], [1.0, 0.0]])

        result = quantize_beats.cosine_drift(a, b)

        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# 4. --drift-run (persisted fp32-vs-int8 drift stats, final-review finding)
# ---------------------------------------------------------------------------


def test_drift_run_persists_row_cosine_stats_in_sidecar(tmp_path, monkeypatch):
    _patch_beats_checkpoint(monkeypatch, tmp_path)
    out_path = tmp_path / "out" / "beats_int8.pt"

    # 6 fake raw windows (fewer than the 2000 default: exercises the clamp).
    rng = np.random.default_rng(7)
    fake_windows = rng.normal(0.0, 0.1, (6, 100, 1)).astype(np.float32)
    monkeypatch.setattr(
        quantize_beats, "_drift_windows_for_run", lambda run, n, cfg: fake_windows
    )

    base = rng.normal(size=(6, 4))

    class _StubFeaturizer:
        """fp32 vs int8 telling: the int8 instance (int8_model_path set)
        returns a slightly perturbed copy of the fp32 embeddings, so the
        cosine stats are non-trivial (< 1) but high (> 0.9)."""

        def __init__(self, checkpoint, device=None, encoder=None, int8_model_path=None):
            self._perturb = int8_model_path is not None

        def transform(self, windows, rate_hz):
            assert windows.shape == fake_windows.shape
            out = base.copy()
            if self._perturb:
                out[:, 0] += 0.05
            return out

    import rowii.signals.beats as beats_mod

    monkeypatch.setattr(beats_mod, "BeatsFeaturizer", _StubFeaturizer)

    rc = quantize_beats.main(
        ["--out", str(out_path), "--drift-run", "some-run", "--drift-windows", "2000"]
    )
    assert rc == 0

    sidecar = json.loads(out_path.with_suffix(".json").read_text())
    drift = sidecar["drift"]
    assert drift["run"] == "some-run"
    assert drift["n_windows"] == 6  # clamped to what the fake reader returned
    assert 0.9 < drift["mean_cosine"] < 1.0
    assert drift["min_cosine"] <= drift["p5_cosine"] <= drift["mean_cosine"]
    assert "CPU" in drift["note"]


def test_without_drift_run_sidecar_has_no_drift_key(tmp_path, monkeypatch):
    _patch_beats_checkpoint(monkeypatch, tmp_path)
    out_path = tmp_path / "out2" / "beats_int8.pt"
    rc = quantize_beats.main(["--out", str(out_path)])
    assert rc == 0
    sidecar = json.loads(out_path.with_suffix(".json").read_text())
    assert "drift" not in sidecar
