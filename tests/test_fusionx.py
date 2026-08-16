"""Tests for the cross-attention fusion head:
`rowii.fusionx.wrapper`/`rowii.fusionx.model` (the `XattnConfig`/`XattnHead`
pair, mirroring `rowii.tfc.wrapper`/`rowii.tfc.model`'s own torch-free-wrapper/
eager-model split) and `scripts/train_xattn.py` (the CLIP-style training CLI
that turns ALREADY-CACHED `audio-beats` embeddings + the `fusion` cache's own
vibration-branch columns into an `xattn_<run>.pt` checkpoint -- zero extraction
compute).

Sections:
1. `XattnHead` direct shape/determinism tests (mirrors `tests/test_tfc_model.py`).
2. `load_xattn_head` geometry guard + checkpoint round-trip (mirrors
   `tests/test_tfc_wrapper.py`).
3. `joint_embeddings` shape/dtype/empty-input contract.
4. Composite-loss sanity: decreases with training on synthetic aligned pairs;
   a trained head prefers the true (aligned) pairing over a shuffled one
   (mirrors `tests/test_tfc_model.py`'s own `test_loss_prefers_aligned_pairs`).
5. `scripts/train_xattn.py`: cache-only loading with fingerprint verification,
   the grid-alignment guard, the leakage-safe calibration-only split, the
   composite-loss training loop, determinism, and checkpoint I/O -- all against
   synthetic `results/cache/*.npz` files built by hand to match `rowii.pipeline.
   _write_cached_prepared_run`'s exact on-disk schema (mirrors `tests/
   test_student.py`'s own section 5; never a real BEATs/fusion extraction
   anywhere in this file).

`-m "not data"` only; no real caches, no real data root, no network anywhere in
this file.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from rowii.anomaly.references import SegmentSplit
from rowii.config import Config
from rowii.fusionx.wrapper import XattnConfig
from rowii.io.dataset import BurstFile, RecordingIndex, Run
from rowii.pipeline import PreparedRun
from rowii.signals.windows import WindowGrid
from tests.fixtures.gantner_builder import build_gantner_file

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import train_xattn  # noqa: E402

from rowii import pipeline as _pipeline  # noqa: E402


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setenv("ROWII_FORCE_CPU", "1")


_VIB_DIM = 40
"""Vibration-branch width used throughout this file's direct `XattnHead`
tests -- an arbitrary but fixed value distinct from `cfg.audio_dim`/
`cfg.lift_dim`/`cfg.out_dim`, so a shape bug that accidentally reuses the wrong
dimension cannot pass by coincidence."""


# ---------------------------------------------------------------------------
# 1. XattnHead direct shape/determinism tests
# ---------------------------------------------------------------------------


class TestXattnHeadShapes:
    def test_forward_shape_default_config(self):
        torch = pytest.importorskip("torch")
        from rowii.fusionx.model import XattnHead

        torch.manual_seed(0)
        cfg = XattnConfig()
        model = XattnHead(cfg, vib_in_dim=_VIB_DIM)
        audio = torch.randn(5, cfg.audio_dim)
        vib = torch.randn(5, _VIB_DIM)
        out = model(audio, vib)
        assert out.shape == (5, cfg.out_dim)

    def test_forward_rejects_mismatched_audio_width(self):
        torch = pytest.importorskip("torch")
        from rowii.fusionx.model import XattnHead

        model = XattnHead(XattnConfig(), vib_in_dim=_VIB_DIM)
        with pytest.raises(RuntimeError):
            model(torch.randn(2, 100), torch.randn(2, _VIB_DIM))

    def test_forward_is_deterministic_given_same_init_seed(self):
        torch = pytest.importorskip("torch")
        from rowii.fusionx.model import XattnHead

        cfg = XattnConfig()
        torch.manual_seed(0)
        model1 = XattnHead(cfg, vib_in_dim=_VIB_DIM)
        torch.manual_seed(0)
        model2 = XattnHead(cfg, vib_in_dim=_VIB_DIM)

        audio = torch.randn(3, cfg.audio_dim)
        vib = torch.randn(3, _VIB_DIM)
        out1 = model1(audio, vib)
        out2 = model2(audio, vib)
        torch.testing.assert_close(out1, out2)

    def test_two_calls_on_the_same_eval_model_are_identical(self):
        """Determinism of `forward` itself (not just init) -- no dropout/BN
        randomness anywhere in this architecture, so `.eval()` is not even
        strictly required here, but set anyway to mirror every OTHER
        checkpoint-loaded model's own inference-time convention."""
        torch = pytest.importorskip("torch")
        from rowii.fusionx.model import XattnHead

        torch.manual_seed(1)
        model = XattnHead(XattnConfig(), vib_in_dim=_VIB_DIM).eval()
        audio = torch.randn(4, XattnConfig().audio_dim)
        vib = torch.randn(4, _VIB_DIM)
        with torch.no_grad():
            out1 = model(audio, vib)
            out2 = model(audio, vib)
        torch.testing.assert_close(out1, out2)

    def test_audio_lift_and_vib_lift_are_public_submodules(self):
        """`scripts/train_xattn.py`'s own training loop reads `model.audio_lift`/
        `model.vib_lift` directly (module docstring's composite-loss section) --
        assert they exist, are plain `nn.Linear`s, and produce the expected
        widths, so a future refactor that renames/removes them fails a test
        here instead of surfacing as a cryptic AttributeError deep inside the
        training script."""
        torch = pytest.importorskip("torch")
        from rowii.fusionx.model import XattnHead

        cfg = XattnConfig()
        model = XattnHead(cfg, vib_in_dim=_VIB_DIM)
        assert isinstance(model.audio_lift, torch.nn.Linear)
        assert isinstance(model.vib_lift, torch.nn.Linear)
        audio = torch.randn(2, cfg.audio_dim)
        vib = torch.randn(2, _VIB_DIM)
        assert model.audio_lift(audio).shape == (2, cfg.lift_dim)
        assert model.vib_lift(vib).shape == (2, cfg.lift_dim)


# ---------------------------------------------------------------------------
# 2. load_xattn_head geometry guard + checkpoint round-trip
# ---------------------------------------------------------------------------


def _save_xattn_checkpoint(
    path: Path,
    torch,
    cfg: XattnConfig,
    vib_dim: int,
    *,
    run_name: str = "test-run",
    epochs: int = 1,
) -> None:
    from rowii.fusionx.model import XattnHead

    model = XattnHead(cfg, vib_in_dim=vib_dim)
    torch.save(
        {
            "cfg": dataclasses.asdict(cfg),
            "model": model.state_dict(),
            "run": run_name,
            "vib_dim": vib_dim,
            "epochs": epochs,
        },
        path,
    )


class TestLoadXattnHead:
    def test_rejects_off_default_audio_dim(self, tmp_path):
        torch = pytest.importorskip("torch")
        from rowii.fusionx.wrapper import load_xattn_head

        cfg = dataclasses.replace(XattnConfig(), audio_dim=512)
        checkpoint_path = tmp_path / "bad_audio_dim.pt"
        _save_xattn_checkpoint(checkpoint_path, torch, cfg, vib_dim=_VIB_DIM)

        with pytest.raises(ValueError) as exc_info:
            load_xattn_head(checkpoint_path, torch.device("cpu"))

        message = str(exc_info.value)
        assert "audio_dim" in message
        assert "512" in message
        assert "768" in message

    def test_accepts_default_geometry(self, tmp_path):
        torch = pytest.importorskip("torch")
        from rowii.fusionx.model import XattnHead
        from rowii.fusionx.wrapper import load_xattn_head

        cfg = XattnConfig()
        checkpoint_path = tmp_path / "good.pt"
        _save_xattn_checkpoint(checkpoint_path, torch, cfg, vib_dim=_VIB_DIM)

        loaded = load_xattn_head(checkpoint_path, torch.device("cpu"))
        assert isinstance(loaded, XattnHead)
        assert loaded.training is False
        assert loaded.cfg == cfg
        assert loaded.vib_lift.in_features == _VIB_DIM

    def test_missing_file_raises_file_not_found_error(self, tmp_path):
        torch = pytest.importorskip("torch")
        from rowii.fusionx.wrapper import load_xattn_head

        with pytest.raises(FileNotFoundError):
            load_xattn_head(tmp_path / "does-not-exist.pt", torch.device("cpu"))

    def test_round_trip_forward_is_deterministic(self, tmp_path):
        torch = pytest.importorskip("torch")
        from rowii.fusionx.wrapper import load_xattn_head

        cfg = XattnConfig()
        checkpoint_path = tmp_path / "roundtrip.pt"
        _save_xattn_checkpoint(checkpoint_path, torch, cfg, vib_dim=_VIB_DIM)

        loaded = load_xattn_head(checkpoint_path, torch.device("cpu"))
        audio = torch.randn(3, cfg.audio_dim)
        vib = torch.randn(3, _VIB_DIM)
        with torch.no_grad():
            out1 = loaded(audio, vib)
            out2 = loaded(audio, vib)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# 3. joint_embeddings shape/dtype/empty-input contract
# ---------------------------------------------------------------------------


class TestJointEmbeddings:
    def test_shape_dtype_finite(self):
        torch = pytest.importorskip("torch")
        from rowii.fusionx.model import XattnHead
        from rowii.fusionx.wrapper import joint_embeddings

        cfg = XattnConfig()
        model = XattnHead(cfg, vib_in_dim=_VIB_DIM).eval()
        rng = np.random.default_rng(0)
        audio = rng.normal(size=(10, cfg.audio_dim))
        vib = rng.normal(size=(10, _VIB_DIM))

        out = joint_embeddings(model, audio, vib, torch.device("cpu"))
        assert out.shape == (10, cfg.out_dim)
        assert out.dtype == np.float64
        assert np.isfinite(out).all()

    def test_empty_input_returns_empty_array_not_error(self):
        torch = pytest.importorskip("torch")
        from rowii.fusionx.model import XattnHead
        from rowii.fusionx.wrapper import joint_embeddings

        cfg = XattnConfig()
        model = XattnHead(cfg, vib_in_dim=_VIB_DIM).eval()
        audio = np.empty((0, cfg.audio_dim))
        vib = np.empty((0, _VIB_DIM))

        out = joint_embeddings(model, audio, vib, torch.device("cpu"))
        assert out.shape == (0, cfg.out_dim)

    def test_chunking_matches_a_single_unchunked_pass(self, monkeypatch):
        """`_CHUNK_SIZE`-batched forward passes must reproduce the SAME result
        as one unchunked pass (mirrors `rowii.anomaly.scorers.KnnScorer`'s own
        documented chunking invariant) -- monkeypatch the module constant down
        to force multiple chunks over a small, fast input."""
        torch = pytest.importorskip("torch")
        import rowii.fusionx.wrapper as wrapper_mod
        from rowii.fusionx.model import XattnHead

        cfg = XattnConfig()
        model = XattnHead(cfg, vib_in_dim=_VIB_DIM).eval()
        rng = np.random.default_rng(1)
        audio = rng.normal(size=(9, cfg.audio_dim))
        vib = rng.normal(size=(9, _VIB_DIM))

        unchunked = wrapper_mod.joint_embeddings(model, audio, vib, torch.device("cpu"))
        monkeypatch.setattr(wrapper_mod, "_CHUNK_SIZE", 2)
        chunked = wrapper_mod.joint_embeddings(model, audio, vib, torch.device("cpu"))

        np.testing.assert_allclose(chunked, unchunked, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Composite-loss sanity (train_xattn._train_xattn_head)
# ---------------------------------------------------------------------------


def _aligned_pairs(n: int, vib_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic ALIGNED (audio, vib) window pairs: the vibration view is a fixed
    random linear projection of the audio view plus small noise, so the two views
    of one window are mutually informative -- exactly the structure the CLIP-style
    alignment objective needs to be learnable at all (mirrors `tests/test_student.
    py`'s own "synthetic teacher = fixed random projection" fixture idea)."""
    rng = np.random.default_rng(seed)
    audio = rng.normal(size=(n, XattnConfig().audio_dim))
    projection = rng.normal(size=(XattnConfig().audio_dim, vib_dim))
    vib = audio @ projection / np.sqrt(XattnConfig().audio_dim)
    vib += 0.05 * rng.normal(size=vib.shape)
    return audio, vib


class TestCompositeLoss:
    def test_loss_decreases_with_training_on_aligned_pairs(self):
        pytest.importorskip("torch")
        audio, vib = _aligned_pairs(48, _VIB_DIM, seed=0)

        model, losses = train_xattn._train_xattn_head(
            audio, vib, XattnConfig(), _VIB_DIM,
            epochs=15, batch_size=16, lr=1e-3, seed=7,
        )

        assert len(losses) == 15
        assert all(np.isfinite(loss_value) for loss_value in losses)
        assert losses[-1] < losses[0]
        assert model.training is False

    def test_trained_head_prefers_aligned_over_shuffled_pairing(self):
        """After training on aligned pairs, the composite loss on the TRUE
        (index-aligned) pairing must be lower than on a SHUFFLED pairing of the
        same rows -- the CLIP-style objective's whole point (mirrors `tests/
        test_tfc_model.py`'s own `test_loss_prefers_aligned_pairs`, lifted from
        the raw `tfc_loss` level to the trained head)."""
        torch = pytest.importorskip("torch")
        from rowii.tfc.model import tfc_loss

        cfg = XattnConfig()
        audio, vib = _aligned_pairs(48, _VIB_DIM, seed=1)
        model, _losses = train_xattn._train_xattn_head(
            audio, vib, cfg, _VIB_DIM, epochs=15, batch_size=16, lr=1e-3, seed=7,
        )

        audio_t = torch.from_numpy(audio.astype(np.float32))
        vib_t = torch.from_numpy(vib.astype(np.float32))
        perm = torch.randperm(48, generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            lift_a = model.audio_lift(audio_t)
            aligned = tfc_loss(lift_a, model.vib_lift(vib_t), cfg.temperature).item()
            shuffled = tfc_loss(lift_a, model.vib_lift(vib_t[perm]), cfg.temperature).item()
        assert aligned < shuffled

    def test_determinism_same_seed_identical_losses_and_state_dict(self):
        torch = pytest.importorskip("torch")
        audio, vib = _aligned_pairs(24, _VIB_DIM, seed=2)

        model1, losses1 = train_xattn._train_xattn_head(
            audio, vib, XattnConfig(), _VIB_DIM, epochs=3, batch_size=8, lr=1e-3, seed=7,
        )
        model2, losses2 = train_xattn._train_xattn_head(
            audio, vib, XattnConfig(), _VIB_DIM, epochs=3, batch_size=8, lr=1e-3, seed=7,
        )

        assert losses1 == losses2
        sd1, sd2 = model1.state_dict(), model2.state_dict()
        assert sd1.keys() == sd2.keys()
        for key in sd1:
            torch.testing.assert_close(sd1[key], sd2[key])

    def test_mismatched_row_counts_raises_value_error(self):
        pytest.importorskip("torch")
        with pytest.raises(ValueError, match=r"shape\[0\]"):
            train_xattn._train_xattn_head(
                np.zeros((4, XattnConfig().audio_dim)), np.zeros((3, _VIB_DIM)),
                XattnConfig(), _VIB_DIM, epochs=1, batch_size=2, lr=1e-3, seed=7,
            )


# ---------------------------------------------------------------------------
# 5. scripts/train_xattn.py (mirrors tests/test_student.py's own section 5)
# ---------------------------------------------------------------------------

_CACHE_RATE_HZ = 100.0
"""Synthetic burst files' own sample rate -- only `_cache_fingerprint`'s file
name+size and `build_run_grid`'s header reads ever touch these files; the
actual sample data/duration is irrelevant to the synthetic-cache tests below,
which fully control the LOGICAL window count via each cache npz's own
`grid_n_windows`/`grid_window_ns` -- kept AS CACHED by `rowii.pipeline.
_load_cached_prepared_run`, never recomputed (mirrors `tests/test_student.py`'s
identical constant)."""

_N_AUDIO_COLS = 4
_N_VIB_COLS = 6
"""Synthetic `fusion` cache's per-branch column counts -- deliberately small,
distinct values so a bug that grabs the wrong branch (or the whole matrix)
cannot produce the right `vib_dim` by coincidence."""


def _fusion_feature_names() -> list[str]:
    """`split_branch_columns`-compatible names: audio columns FIRST (matching
    the real `fusion` variant's own stream order, `rowii.pipeline.
    _streams_for_variant`), then vibration columns."""
    audio = [f"RAWGeneratorMic__0::a{i}" for i in range(_N_AUDIO_COLS)]
    vib = [f"RAWGeneratorVib__2::v{i}" for i in range(_N_VIB_COLS)]
    return audio + vib


def _cache_burst(
    path: Path, stream: str, n_seconds: float, *, t0_ns: int, start_utc_hint: datetime
) -> BurstFile:
    n_samples = round(_CACHE_RATE_HZ * n_seconds)
    data = np.ones((n_samples, 4), dtype=np.float32)
    build_gantner_file(
        path, ["ch0", "ch1", "ch2", "ch3"], data, t0_ns=t0_ns, rate_hz=_CACHE_RATE_HZ
    )
    return BurstFile(path=path, stream=stream, start_utc_hint=start_utc_hint)


def _tiny_fusion_run(burst_dir: Path, *, name: str = "xattn-run", n_seconds: float = 2.0) -> Run:
    """Minimal REAL four-stream `Run` for cache-fingerprint/grid derivation --
    mirrors `tests/test_student.py`'s own `_tiny_audio_run`, extended to all
    four streams: `audio-beats` (`_streams_for_variant` = both mic streams) and
    `fusion` (both mic + both vib streams) can BOTH resolve against this one
    Run."""
    burst_dir.mkdir(parents=True, exist_ok=True)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    streams = (
        "RAWGeneratorMic__0", "RAWTurbineMic__1", "RAWGeneratorVib__2", "RAWTurbineVib__3",
    )
    files = {
        stream: [
            _cache_burst(
                burst_dir / f"{stream}.dat", stream, n_seconds,
                t0_ns=0, start_utc_hint=t0,
            )
        ]
        for stream in streams
    }
    return Run(name=name, files=files, day_root=burst_dir)


def _xattn_cfg(results_root: Path) -> Config:
    return Config(data_root=results_root, results_root=results_root)


def _write_synthetic_cache(
    cfg: Config,
    run: Run,
    variant: str,
    *,
    features: np.ndarray,
    valid_mask: np.ndarray,
    segment_ids: np.ndarray,
    feature_names: list[str] | None = None,
    window_ns: int = 1_000_000_000,
    t0_ns: int = 0,
) -> Path:
    """Writes a real `results/cache/<run>--<variant>.npz` via `rowii.pipeline`'s
    OWN `_write_cached_prepared_run` (the exact on-disk schema, no hand-rolled
    npz keys) + a REAL fingerprint (`_cache_fingerprint`) -- so `train_xattn.
    _load_cache_or_exit`'s fingerprint verification is genuinely exercised, not
    bypassed (mirrors `tests/test_student.py`'s identical helper, plus a
    `feature_names` override for the `fusion` cache's branch-splittable
    names)."""
    n_windows = features.shape[0]
    prepared = PreparedRun(
        features=features,
        grid=WindowGrid(t0_ns=t0_ns, window_ns=window_ns, n_windows=n_windows),
        valid_mask=valid_mask,
        feature_names=(
            feature_names if feature_names is not None
            else [f"f{i}" for i in range(features.shape[1])]
        ),
        segment_ids=segment_ids,
    )
    cache_path = _pipeline._cache_npz_path(cfg.results_root, run.name, variant)
    fingerprint = _pipeline._cache_fingerprint(run, variant, cfg)
    _pipeline._write_cached_prepared_run(cache_path, fingerprint, prepared)
    return cache_path


def _grid(t0_ns: int = 0, window_ns: int = 1_000_000_000, n_windows: int = 10) -> WindowGrid:
    return WindowGrid(t0_ns=t0_ns, window_ns=window_ns, n_windows=n_windows)


def _prepared_with_grid(grid: WindowGrid) -> PreparedRun:
    n = grid.n_windows
    return PreparedRun(
        features=np.zeros((n, 1)),
        grid=grid,
        valid_mask=np.ones(n, dtype=bool),
        feature_names=["f0"],
        segment_ids=np.zeros(n, dtype=np.int64),
    )


def _fake_index(runs: list[Run]) -> RecordingIndex:
    return RecordingIndex(runs=runs, betriebsdaten=[], betriebsdaten_by_day={})


# --- 5a. _check_cache_alignment (ensemble-style tolerance) -------------------


class TestCheckCacheAlignment:
    def test_exact_match_returns_zero_offset(self):
        audio = _prepared_with_grid(_grid())
        vib_source = _prepared_with_grid(_grid())
        assert train_xattn._check_cache_alignment("run", audio, vib_source) == 0

    def test_sub_window_offset_tolerated_with_warning(self, caplog):
        audio = _prepared_with_grid(_grid(t0_ns=0))
        vib_source = _prepared_with_grid(_grid(t0_ns=26_000_000))
        with caplog.at_level(logging.WARNING):
            offset = train_xattn._check_cache_alignment("run", audio, vib_source)
        assert offset == 26_000_000
        assert any("offset" in r.message.lower() for r in caplog.records)

    def test_one_window_offset_exits_2(self, capsys):
        audio = _prepared_with_grid(_grid(t0_ns=1_000_000_000))
        vib_source = _prepared_with_grid(_grid(t0_ns=0))
        with pytest.raises(SystemExit) as exc_info:
            train_xattn._check_cache_alignment("run", audio, vib_source)
        assert exc_info.value.code == 2
        assert "misaligned by >= one window" in capsys.readouterr().err

    def test_structural_n_windows_mismatch_exits_2(self, capsys):
        audio = _prepared_with_grid(_grid(n_windows=10))
        vib_source = _prepared_with_grid(_grid(n_windows=9))
        with pytest.raises(SystemExit) as exc_info:
            train_xattn._check_cache_alignment("run", audio, vib_source)
        assert exc_info.value.code == 2
        assert "grid mismatch" in capsys.readouterr().err

    def test_structural_window_ns_mismatch_exits_2(self, capsys):
        audio = _prepared_with_grid(_grid(window_ns=1_000_000_000))
        vib_source = _prepared_with_grid(_grid(window_ns=500_000_000))
        with pytest.raises(SystemExit) as exc_info:
            train_xattn._check_cache_alignment("run", audio, vib_source)
        assert exc_info.value.code == 2
        assert "grid mismatch" in capsys.readouterr().err


# --- 5b. _select_calibration_windows (leakage) ------------------------------


class TestSelectCalibrationWindows:
    def test_leakage_known_split_excludes_scoring_side(self, monkeypatch):
        n = 10
        common_segment_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
        vib_source = PreparedRun(
            features=np.zeros((n, 1)), grid=_grid(n_windows=n),
            valid_mask=np.ones(n, dtype=bool), feature_names=["f0"],
            segment_ids=common_segment_ids,
        )
        audio = PreparedRun(
            features=np.zeros((n, 1)), grid=_grid(n_windows=n),
            valid_mask=np.ones(n, dtype=bool), feature_names=["f0"],
            segment_ids=common_segment_ids,
        )
        known_split = SegmentSplit(
            calibration_windows=np.array([0, 1, 4, 5], dtype=np.int64),
            scoring_windows=np.array([2, 3, 6, 7, 8, 9], dtype=np.int64),
        )
        calls: list[tuple[float, int]] = []

        def fake_split_by_segments(segment_ids, valid_mask, calibration_frac, seed):
            calls.append((calibration_frac, seed))
            return known_split

        monkeypatch.setattr(train_xattn, "split_by_segments", fake_split_by_segments)

        result = train_xattn._select_calibration_windows(vib_source, audio, seed=7)

        np.testing.assert_array_equal(result, known_split.calibration_windows)
        assert not (set(result.tolist()) & set(known_split.scoring_windows.tolist())), (
            "xattn training must never draw a scoring-side window (spec D3 leakage rule)"
        )
        assert calls == [(0.5, 7)], (
            "must call split_by_segments(segment_ids, valid_mask, 0.5, seed) with the "
            "default seed=7 when the caller does not override it"
        )

    def test_combines_both_caches_valid_masks(self, monkeypatch):
        # A window valid for fusion but NOT for audio-beats must never reach
        # split_by_segments' valid_mask as True -- its audio side would be
        # undefined (NaN) there (module docstring's rationale for why the AND
        # is a safe extension of, not a departure from, the leakage rule).
        n = 4
        vib_source = PreparedRun(
            features=np.zeros((n, 1)), grid=_grid(n_windows=n),
            valid_mask=np.array([True, True, True, True]), feature_names=["f0"],
            segment_ids=np.array([0, 0, 1, 1], dtype=np.int64),
        )
        audio = PreparedRun(
            features=np.zeros((n, 1)), grid=_grid(n_windows=n),
            valid_mask=np.array([True, False, True, True]), feature_names=["f0"],
            segment_ids=np.array([0, 0, 1, 1], dtype=np.int64),
        )
        seen_masks: list[np.ndarray] = []

        def fake_split_by_segments(segment_ids, valid_mask, calibration_frac, seed):
            seen_masks.append(valid_mask.copy())
            return SegmentSplit(
                calibration_windows=np.array([0], dtype=np.int64),
                scoring_windows=np.array([2], dtype=np.int64),
            )

        monkeypatch.setattr(train_xattn, "split_by_segments", fake_split_by_segments)

        train_xattn._select_calibration_windows(vib_source, audio, seed=7)

        np.testing.assert_array_equal(seen_masks[0], np.array([True, False, True, True]))


# --- 5c. _load_cache_or_exit (public cache-load path + fingerprint check) --


class TestLoadCacheOrExit:
    def test_hit_returns_prepared_run_matching_cached_features(self, tmp_path):
        run = _tiny_fusion_run(tmp_path / "burst")
        cfg = _xattn_cfg(tmp_path / "results")
        features = np.arange(20.0).reshape(10, 2)
        valid_mask = np.ones(10, dtype=bool)
        segment_ids = np.zeros(10, dtype=np.int64)
        _write_synthetic_cache(
            cfg, run, "fusion", features=features, valid_mask=valid_mask,
            segment_ids=segment_ids,
        )

        prepared = train_xattn._load_cache_or_exit(run, "fusion", cfg)

        np.testing.assert_array_equal(prepared.features, features)
        np.testing.assert_array_equal(prepared.valid_mask, valid_mask)
        assert prepared.grid.n_windows == 10

    def test_miss_no_file_exits_naming_warm_cache_hint(self, tmp_path):
        run = _tiny_fusion_run(tmp_path / "burst")
        cfg = _xattn_cfg(tmp_path / "results")

        with pytest.raises(SystemExit) as exc_info:
            train_xattn._load_cache_or_exit(run, "fusion", cfg)

        message = str(exc_info.value)
        assert "warm_cache.py" in message
        assert "fusion" in message

    def test_miss_stale_fingerprint_exits(self, tmp_path):
        run = _tiny_fusion_run(tmp_path / "burst")
        cfg = _xattn_cfg(tmp_path / "results")
        cache_path = _write_synthetic_cache(
            cfg, run, "fusion", features=np.zeros((5, 2)),
            valid_mask=np.ones(5, dtype=bool), segment_ids=np.zeros(5, dtype=np.int64),
        )
        with np.load(cache_path, allow_pickle=False) as data:
            raw = dict(data.items())
        raw["fingerprint"] = np.array(["stale-fingerprint"], dtype=str)
        np.savez(cache_path, **raw)

        with pytest.raises(SystemExit) as exc_info:
            train_xattn._load_cache_or_exit(run, "fusion", cfg)

        assert "warm_cache.py" in str(exc_info.value)


# --- 5d. main() end-to-end ---------------------------------------------------


class TestTrainXattnMainEndToEnd:
    def _write_both_caches(self, cfg: Config, run: Run, *, n: int = 12) -> None:
        rng = np.random.default_rng(3)
        # >= 2 distinct segments so split_by_segments(0.5) yields a non-degenerate
        # calibration/scoring split (mirrors tests/test_student.py's e2e fixture).
        segment_ids = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=np.int64)
        _write_synthetic_cache(
            cfg, run, "audio-beats",
            # Faithful to the real cache: BOTH mic streams' 768-d embeddings
            # concatenated (1536 columns), stream-prefixed names -- the trainer
            # must slice out the primary mic's 768, never project 1536.
            features=rng.normal(size=(n, 1536)),
            valid_mask=np.ones(n, dtype=bool), segment_ids=segment_ids,
            feature_names=(
                [f"RAWGeneratorMic__0::beats_{i}" for i in range(768)]
                + [f"RAWTurbineMic__1::beats_{i}" for i in range(768)]
            ),
        )
        _write_synthetic_cache(
            cfg, run, "fusion",
            features=rng.normal(size=(n, _N_AUDIO_COLS + _N_VIB_COLS)),
            valid_mask=np.ones(n, dtype=bool), segment_ids=segment_ids,
            feature_names=_fusion_feature_names(),
        )

    def _patch_env(self, monkeypatch, tmp_path: Path, cfg: Config, run: Run) -> None:
        monkeypatch.setattr(train_xattn, "discover", lambda data_root: _fake_index([run]))
        monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
        monkeypatch.setenv("ROWII_RESULTS_ROOT", str(cfg.results_root))
        # This repo's own .env sets a real ROWII_BEATS_CHECKPOINT for dev use
        # (process env > .env, but only once the KEY is present) -- an explicit
        # empty string neutralizes it so main()'s own load_config() computes the
        # SAME fingerprint _write_synthetic_cache used above (mirrors tests/
        # test_student.py's identical fix for the identical contamination).
        monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", "")

    def test_full_run_writes_checkpoint_and_sidecar(self, tmp_path, monkeypatch):
        torch = pytest.importorskip("torch")
        run = _tiny_fusion_run(tmp_path / "burst", name="e2e-run")
        cfg = _xattn_cfg(tmp_path / "results")
        self._write_both_caches(cfg, run)
        self._patch_env(monkeypatch, tmp_path, cfg, run)

        out_dir = tmp_path / "out"
        rc = train_xattn.main(
            [
                "--run", run.name, "--epochs", "2", "--batch-size", "4", "--seed", "7",
                "--out", str(out_dir),
            ]
        )

        assert rc == 0
        checkpoint_path = out_dir / f"xattn_{run.name}.pt"
        sidecar_path = checkpoint_path.with_suffix(".json")
        assert checkpoint_path.is_file()
        assert sidecar_path.is_file()

        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["run"] == run.name
        assert sidecar["audio_variant"] == "audio-beats"
        assert sidecar["audio_stream"] == "RAWGeneratorMic__0"
        assert sidecar["vib_source_variant"] == "fusion"
        assert sidecar["vib_dim"] == _N_VIB_COLS
        assert sidecar["epochs"] == 2
        assert sidecar["n_calibration_windows"] > 0
        assert np.isfinite(sidecar["final_loss"])
        assert "calibration-side" in sidecar["note"]

        from rowii.fusionx.wrapper import load_xattn_head

        loaded = load_xattn_head(checkpoint_path, torch.device("cpu"))
        assert loaded.training is False
        assert loaded.vib_lift.in_features == _N_VIB_COLS

    def test_determinism_two_same_seed_runs_write_identical_tensors(
        self, tmp_path, monkeypatch
    ):
        torch = pytest.importorskip("torch")
        run = _tiny_fusion_run(tmp_path / "burst", name="det-run")
        cfg = _xattn_cfg(tmp_path / "results")
        self._write_both_caches(cfg, run)
        self._patch_env(monkeypatch, tmp_path, cfg, run)

        out_a, out_b = tmp_path / "out_a", tmp_path / "out_b"
        for out_dir in (out_a, out_b):
            rc = train_xattn.main(
                ["--run", run.name, "--epochs", "2", "--batch-size", "4",
                 "--seed", "7", "--out", str(out_dir)]
            )
            assert rc == 0

        state_a = torch.load(out_a / f"xattn_{run.name}.pt", weights_only=False)
        state_b = torch.load(out_b / f"xattn_{run.name}.pt", weights_only=False)
        assert state_a["model"].keys() == state_b["model"].keys()
        for key in state_a["model"]:
            torch.testing.assert_close(state_a["model"][key], state_b["model"][key])

    def test_grid_misalignment_exits_2(self, tmp_path, monkeypatch, capsys):
        """main() must run `_check_cache_alignment` between the two cache loads
        and training, propagating its `SystemExit(2)`. The misaligned pair is
        injected by monkeypatching `_load_cache_or_exit` (not by writing
        misaligned t0 values into synthetic cache files): `rowii.pipeline.
        _load_cached_prepared_run` deliberately RECOMPUTES grid t0 from the
        run's real file headers on every hit (its own raw-axis-cache
        compatibility story), so a cached t0 can never disagree with the
        fresh header-derived one for a same-files synthetic run -- the guard's
        own decision logic is unit-tested directly in TestCheckCacheAlignment
        above; this test pins only main()'s wiring of it."""
        pytest.importorskip("torch")
        run = _tiny_fusion_run(tmp_path / "burst", name="misaligned-run")
        cfg = _xattn_cfg(tmp_path / "results")
        self._patch_env(monkeypatch, tmp_path, cfg, run)

        n = 12
        misaligned = {
            "audio-beats": _prepared_with_grid(_grid(t0_ns=0, n_windows=n)),
            "fusion": _prepared_with_grid(_grid(t0_ns=2_000_000_000, n_windows=n)),
        }
        monkeypatch.setattr(
            train_xattn, "_load_cache_or_exit",
            lambda run_, variant, cfg_: misaligned[variant],
        )

        with pytest.raises(SystemExit) as exc_info:
            train_xattn.main(
                ["--run", run.name, "--epochs", "1", "--out", str(tmp_path / "out")]
            )

        assert exc_info.value.code == 2
        assert "misaligned by >= one window" in capsys.readouterr().err

    def test_unknown_run_exits(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        monkeypatch.setattr(train_xattn, "discover", lambda data_root: _fake_index([]))
        monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
        monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
        monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", "")

        with pytest.raises(SystemExit):
            train_xattn.main(["--run", "does-not-exist"])

    def test_missing_cache_exits_naming_warm_cache_hint(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        run = _tiny_fusion_run(tmp_path / "burst", name="no-cache-run")
        monkeypatch.setattr(train_xattn, "discover", lambda data_root: _fake_index([run]))
        monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
        monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
        monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", "")

        with pytest.raises(SystemExit) as exc_info:
            train_xattn.main(["--run", run.name])

        assert "warm_cache.py" in str(exc_info.value)

    def test_help_exits_zero_and_documents_every_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            train_xattn.main(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        for flag in ("--run", "--epochs", "--batch-size", "--lr", "--seed", "--out"):
            assert flag in out
