"""Tests for the distilled BEATs-student compactness pair (Step-2 package-5 spec
D5, plan Task 4): `rowii.adapt.student`/`rowii.adapt._student_model` (the
`StudentFeaturizer`/`_StudentNet` pair, mirroring `rowii.tfc.wrapper`/
`rowii.tfc.model`'s own torch-free-wrapper/eager-model split) and
`scripts/distill_beats.py` (the training CLI that turns ALREADY-CACHED
`audio-beats` teacher embeddings + `logmel` student-input caches into a
`student_<run>.pt` checkpoint -- zero teacher/extraction compute).

Sections:
1. `StudentFeaturizer` contract (stub encoder, torch-free import "poison-style" --
   mirrors `tests/test_tfc_wrapper.py`'s own `test_module_imports_without_torch`:
   this whole section never calls `pytest.importorskip("torch")` at module scope,
   proving the stub-encoder path never needs torch).
2. `_StudentNet` direct shape test (mirrors `tests/test_tfc_model.py`).
3. `load_student_model` geometry guard (mirrors `tests/test_tfc_wrapper.py`'s
   `_validate_checkpoint_geometry` tests).
4. Checkpoint round-trip determinism (mirrors `tests/test_tfc_wrapper.py`'s own).
5. `scripts/distill_beats.py`: cache-only loading with fingerprint verification,
   the grid-alignment guard, the leakage-safe calibration-only split, the
   MSE-distillation training loop (loss decreases), determinism, and checkpoint
   I/O -- all against synthetic `results/cache/*.npz` files built by hand to match
   `rowii.pipeline._write_cached_prepared_run`'s exact on-disk schema (never a
   real BEATs/logmel extraction anywhere in this file).

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

from rowii.adapt.student import StudentConfig, StudentFeaturizer
from rowii.anomaly.references import SegmentSplit
from rowii.config import Config
from rowii.io.dataset import BurstFile, RecordingIndex, Run
from rowii.pipeline import PreparedRun
from rowii.signals.windows import WindowGrid
from tests.fixtures.gantner_builder import build_gantner_file

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import distill_beats  # noqa: E402

from rowii import pipeline as _pipeline  # noqa: E402


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setenv("ROWII_FORCE_CPU", "1")


_PLANT_RATE_HZ = 50_000.0
"""The plant's real mic sample rate -- with 1-s windows this is the geometry at
which `LogmelFeaturizer` (the student's front end) produces exactly 49 frames x
64 mels = 3136 features, matching `StudentConfig`'s defaults (`n_frames=49,
n_mels=64`) -- see `rowii.signals.logmel.LogmelFeaturizer`'s own docstring."""


# ---------------------------------------------------------------------------
# 1. StudentFeaturizer contract: stub encoder, no torch needed at all
# ---------------------------------------------------------------------------


class _StubEncoder:
    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def embed(self, logmel_flat: np.ndarray) -> np.ndarray:
        self.calls.append(logmel_flat.shape)
        return np.tile(logmel_flat.mean(axis=1, keepdims=True), (1, 768))


class TestStudentFeaturizer:
    def test_transform_2d_shape_names_dtype(self):
        f = StudentFeaturizer(checkpoint=None, encoder=_StubEncoder())
        rng = np.random.default_rng(0)
        out = f.transform(rng.normal(0, 1, (3, 50_000)), _PLANT_RATE_HZ)
        assert out.shape == (3, 768)
        assert out.dtype == np.float64
        names = f.feature_names()
        assert names[0] == "student_e0"
        assert names[-1] == "student_e767"
        assert len(names) == 768

    def test_transform_3d_mono_mix(self):
        stub = _StubEncoder()
        f = StudentFeaturizer(checkpoint=None, encoder=stub)
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, (2, 50_000, 3))
        out = f.transform(x, _PLANT_RATE_HZ)
        assert out.shape == (2, 768)
        assert stub.calls[0] == (2, 3136)  # 49 frames x 64 mels, mono-mixed first

    def test_stub_receives_flattened_logmel_patch(self):
        """The stub's `embed(logmel_flat)` contract -- `(B, n_frames*n_mels)`,
        the SAME flattened shape `_StudentNet.forward` reshapes internally --
        proving `StudentFeaturizer.transform` never reshapes before handing off
        to an injected encoder (module docstring's "reshape happens inside the
        model" design)."""
        stub = _StubEncoder()
        f = StudentFeaturizer(checkpoint=None, encoder=stub)
        x = np.random.default_rng(2).normal(0, 1, (1, 50_000))
        f.transform(x, _PLANT_RATE_HZ)
        assert stub.calls[0] == (1, 3136)

    def test_feature_names_available_before_any_transform_call(self):
        # Unlike BeatsFeaturizer (embedding width discovered from the encoder at
        # runtime), out_dim=768 is a fixed StudentConfig default every checkpoint
        # this project trains keeps -- mirrors TfcFeaturizer.feature_names().
        f = StudentFeaturizer(checkpoint=None, encoder=None)
        names = f.feature_names()
        assert len(names) == 768
        assert names[0] == "student_e0" and names[-1] == "student_e767"

    def test_module_imports_without_torch(self):
        # student.py itself must not import torch at module level -- this whole
        # test file section never imports torch either. Assert the real-encoder
        # path guards instead: no checkpoint, no stub -> a clean error, not an
        # accidental ImportError from some top-level `import torch`.
        f = StudentFeaturizer(checkpoint=None, encoder=None)
        with pytest.raises((RuntimeError, ValueError)):
            f.transform(np.zeros((1, 50_000)), _PLANT_RATE_HZ)


# ---------------------------------------------------------------------------
# 2. _StudentNet direct shape test (mirrors tests/test_tfc_model.py)
# ---------------------------------------------------------------------------


def test_student_net_forward_shape_default_config():
    torch = pytest.importorskip("torch")
    from rowii.adapt._student_model import _StudentNet

    torch.manual_seed(0)
    cfg = StudentConfig()
    model = _StudentNet(cfg)
    x = torch.randn(4, cfg.n_frames * cfg.n_mels)
    out = model(x)
    assert out.shape == (4, 768)


def test_student_net_forward_shape_tiny_config():
    torch = pytest.importorskip("torch")
    from rowii.adapt._student_model import _StudentNet

    torch.manual_seed(0)
    cfg = StudentConfig(channels=(4, 8, 16))
    model = _StudentNet(cfg)
    x = torch.randn(2, cfg.n_frames * cfg.n_mels)
    out = model(x)
    assert out.shape == (2, 768)


def test_student_net_rejects_mismatched_input_width():
    torch = pytest.importorskip("torch")
    from rowii.adapt._student_model import _StudentNet

    model = _StudentNet(StudentConfig())
    with pytest.raises(ValueError, match="3136"):
        model(torch.randn(2, 100))


# ---------------------------------------------------------------------------
# 3. load_student_model geometry guard (mirrors test_tfc_wrapper.py)
# ---------------------------------------------------------------------------


def _save_student_checkpoint(path: Path, torch, cfg: StudentConfig) -> None:
    from rowii.adapt._student_model import _StudentNet

    model = _StudentNet(cfg)
    torch.save(
        {
            "cfg": dataclasses.asdict(cfg),
            "model": model.state_dict(),
            "teacher_variant": "audio-beats",
            "run": "test-run",
            "epochs": 1,
        },
        path,
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("n_mels", 32), ("n_frames", 20), ("out_dim", 256)],
)
def test_load_student_model_rejects_off_default_geometry(tmp_path, monkeypatch, field, bad_value):
    torch = pytest.importorskip("torch")
    from rowii.adapt.student import load_student_model

    monkeypatch.setenv("ROWII_FORCE_CPU", "1")

    cfg = dataclasses.replace(StudentConfig(channels=(4, 8, 16)), **{field: bad_value})
    checkpoint_path = tmp_path / "off_default.pt"
    _save_student_checkpoint(checkpoint_path, torch, cfg)

    with pytest.raises(ValueError) as exc_info:
        load_student_model(checkpoint_path, torch.device("cpu"))

    message = str(exc_info.value)
    assert field in message
    assert str(bad_value) in message


def test_load_student_model_accepts_default_geometry_with_only_channels_overridden(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    from rowii.adapt._student_model import _StudentNet
    from rowii.adapt.student import load_student_model

    monkeypatch.setenv("ROWII_FORCE_CPU", "1")

    cfg = StudentConfig(channels=(4, 8, 16))  # n_mels/n_frames/out_dim all default
    checkpoint_path = tmp_path / "tiny_but_valid.pt"
    _save_student_checkpoint(checkpoint_path, torch, cfg)

    loaded = load_student_model(checkpoint_path, torch.device("cpu"))
    assert isinstance(loaded, _StudentNet)


def test_load_student_model_missing_file_raises_file_not_found_error(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from rowii.adapt.student import load_student_model

    monkeypatch.setenv("ROWII_FORCE_CPU", "1")

    with pytest.raises(FileNotFoundError):
        load_student_model(tmp_path / "does-not-exist.pt", torch.device("cpu"))


# ---------------------------------------------------------------------------
# 4. Checkpoint round-trip determinism (mirrors test_tfc_wrapper.py)
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_transform_is_deterministic(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from rowii.adapt._student_model import _StudentNet
    from rowii.adapt.student import load_student_model

    monkeypatch.setenv("ROWII_FORCE_CPU", "1")

    cfg = StudentConfig(channels=(4, 8, 16))
    checkpoint_path = tmp_path / "student_checkpoint.pt"
    _save_student_checkpoint(checkpoint_path, torch, cfg)

    loaded = load_student_model(checkpoint_path, torch.device("cpu"))
    assert isinstance(loaded, _StudentNet)
    assert loaded.training is False  # load_student_model must leave it in eval mode

    featurizer = StudentFeaturizer(checkpoint=checkpoint_path)
    windows = np.random.default_rng(3).normal(0, 1, (2, 50_000))

    out1 = featurizer.transform(windows, _PLANT_RATE_HZ)
    out2 = featurizer.transform(windows, _PLANT_RATE_HZ)

    assert out1.shape == (2, 768) and out1.dtype == np.float64
    assert np.isfinite(out1).all()
    np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# 5. scripts/distill_beats.py
# ---------------------------------------------------------------------------

_CACHE_RATE_HZ = 100.0
"""Synthetic burst files' own sample rate -- only `_cache_fingerprint`'s file
name+size and `build_run_grid`'s header reads ever touch these files (module
docstring's "5." section); the actual sample data/duration is irrelevant to
the synthetic-cache tests below, which fully control the LOGICAL window count
via each cache npz's own `grid_n_windows`/`grid_window_ns` -- kept AS CACHED
by `rowii.pipeline._load_cached_prepared_run`, never recomputed."""


def _cache_burst(
    path: Path, stream: str, n_seconds: float, *, t0_ns: int, start_utc_hint: datetime
) -> BurstFile:
    n_samples = round(_CACHE_RATE_HZ * n_seconds)
    data = np.ones((n_samples, 4), dtype=np.float32)
    build_gantner_file(
        path, ["ch0", "ch1", "ch2", "ch3"], data, t0_ns=t0_ns, rate_hz=_CACHE_RATE_HZ
    )
    return BurstFile(path=path, stream=stream, start_utc_hint=start_utc_hint)


def _tiny_audio_run(burst_dir: Path, *, name: str = "distill-run", n_seconds: float = 2.0) -> Run:
    """Minimal REAL (both-mic-stream) `Run` for cache-fingerprint/grid derivation
    -- mirrors `tests/test_pipeline.py`'s own `_single_file_audio_run`. Both
    `audio-beats` (`_streams_for_variant` = both mic streams) and `logmel`
    (primary mic only) can resolve against this one Run."""
    burst_dir.mkdir(parents=True, exist_ok=True)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    files = {
        "RAWGeneratorMic__0": [
            _cache_burst(
                burst_dir / "gen_mic.dat", "RAWGeneratorMic__0", n_seconds,
                t0_ns=0, start_utc_hint=t0,
            )
        ],
        "RAWTurbineMic__1": [
            _cache_burst(
                burst_dir / "tur_mic.dat", "RAWTurbineMic__1", n_seconds,
                t0_ns=0, start_utc_hint=t0,
            )
        ],
    }
    return Run(name=name, files=files, day_root=burst_dir)


def _distill_cfg(results_root: Path) -> Config:
    return Config(data_root=results_root, results_root=results_root)


def _write_synthetic_cache(
    cfg: Config,
    run: Run,
    variant: str,
    *,
    features: np.ndarray,
    valid_mask: np.ndarray,
    segment_ids: np.ndarray,
    window_ns: int = 1_000_000_000,
    t0_ns: int = 0,
) -> Path:
    """Writes a real `results/cache/<run>--<variant>.npz` via `rowii.pipeline`'s
    OWN `_write_cached_prepared_run` (the exact on-disk schema, no hand-rolled
    npz keys) + a REAL fingerprint (`_cache_fingerprint`) -- so `distill_beats.
    _load_cache_or_exit`'s fingerprint verification is genuinely exercised, not
    bypassed."""
    n_windows = features.shape[0]
    prepared = PreparedRun(
        features=features,
        grid=WindowGrid(t0_ns=t0_ns, window_ns=window_ns, n_windows=n_windows),
        valid_mask=valid_mask,
        feature_names=[f"f{i}" for i in range(features.shape[1])],
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


# --- 5a. _check_cache_alignment ---------------------------------------------


class TestCheckCacheAlignment:
    def test_exact_match_returns_zero_offset(self):
        teacher = _prepared_with_grid(_grid())
        student_input = _prepared_with_grid(_grid())
        assert distill_beats._check_cache_alignment("run", teacher, student_input) == 0

    def test_sub_window_offset_tolerated_with_warning(self, caplog):
        teacher = _prepared_with_grid(_grid(t0_ns=26_000_000))
        student_input = _prepared_with_grid(_grid(t0_ns=0))
        with caplog.at_level(logging.WARNING):
            offset = distill_beats._check_cache_alignment("run", teacher, student_input)
        assert offset == 26_000_000
        assert any("offset" in r.message.lower() for r in caplog.records)

    def test_one_window_offset_exits_2(self, capsys):
        teacher = _prepared_with_grid(_grid(t0_ns=1_000_000_000))
        student_input = _prepared_with_grid(_grid(t0_ns=0))
        with pytest.raises(SystemExit) as exc_info:
            distill_beats._check_cache_alignment("run", teacher, student_input)
        assert exc_info.value.code == 2
        assert "misaligned by >= one window" in capsys.readouterr().err

    def test_structural_n_windows_mismatch_exits_2(self, capsys):
        teacher = _prepared_with_grid(_grid(n_windows=10))
        student_input = _prepared_with_grid(_grid(n_windows=9))
        with pytest.raises(SystemExit) as exc_info:
            distill_beats._check_cache_alignment("run", teacher, student_input)
        assert exc_info.value.code == 2
        assert "grid mismatch" in capsys.readouterr().err

    def test_structural_window_ns_mismatch_exits_2(self, capsys):
        teacher = _prepared_with_grid(_grid(window_ns=1_000_000_000))
        student_input = _prepared_with_grid(_grid(window_ns=500_000_000))
        with pytest.raises(SystemExit) as exc_info:
            distill_beats._check_cache_alignment("run", teacher, student_input)
        assert exc_info.value.code == 2
        assert "grid mismatch" in capsys.readouterr().err


# --- 5b. _select_calibration_windows (leakage) ------------------------------


class TestSelectCalibrationWindows:
    def test_leakage_known_split_excludes_scoring_side(self, monkeypatch):
        n = 10
        common_segment_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
        student_input = PreparedRun(
            features=np.zeros((n, 1)), grid=_grid(n_windows=n),
            valid_mask=np.ones(n, dtype=bool), feature_names=["f0"],
            segment_ids=common_segment_ids,
        )
        teacher = PreparedRun(
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

        monkeypatch.setattr(distill_beats, "split_by_segments", fake_split_by_segments)

        result = distill_beats._select_calibration_windows(student_input, teacher, seed=7)

        np.testing.assert_array_equal(result, known_split.calibration_windows)
        assert not (set(result.tolist()) & set(known_split.scoring_windows.tolist())), (
            "distillation must never train on a scoring-side window (spec D3 leakage rule)"
        )
        assert calls == [(0.5, 7)], (
            "must call split_by_segments(segment_ids, valid_mask, 0.5, seed) with the "
            "default seed=7 when the caller does not override it"
        )

    def test_custom_seed_is_forwarded(self, monkeypatch):
        n = 4
        prepared = PreparedRun(
            features=np.zeros((n, 1)), grid=_grid(n_windows=n),
            valid_mask=np.ones(n, dtype=bool), feature_names=["f0"],
            segment_ids=np.array([0, 0, 1, 1], dtype=np.int64),
        )
        calls: list[tuple[float, int]] = []

        def fake_split_by_segments(segment_ids, valid_mask, calibration_frac, seed):
            calls.append((calibration_frac, seed))
            return SegmentSplit(
                calibration_windows=np.array([0], dtype=np.int64),
                scoring_windows=np.array([2], dtype=np.int64),
            )

        monkeypatch.setattr(distill_beats, "split_by_segments", fake_split_by_segments)

        distill_beats._select_calibration_windows(prepared, prepared, seed=99)

        assert calls == [(0.5, 99)]

    def test_combines_both_caches_valid_masks(self, monkeypatch):
        # A window valid for logmel but NOT for audio-beats must never reach
        # split_by_segments' valid_mask as True -- the teacher target would be
        # undefined (NaN) there (module docstring's rationale for why the AND
        # is a safe extension of, not a departure from, D3's rule).
        n = 4
        student_input = PreparedRun(
            features=np.zeros((n, 1)), grid=_grid(n_windows=n),
            valid_mask=np.array([True, True, True, True]), feature_names=["f0"],
            segment_ids=np.array([0, 0, 1, 1], dtype=np.int64),
        )
        teacher = PreparedRun(
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

        monkeypatch.setattr(distill_beats, "split_by_segments", fake_split_by_segments)

        distill_beats._select_calibration_windows(student_input, teacher, seed=7)

        np.testing.assert_array_equal(seen_masks[0], np.array([True, False, True, True]))


# --- 5c. _load_cache_or_exit (public cache-load path + fingerprint check) --


class TestLoadCacheOrExit:
    def test_hit_returns_prepared_run_matching_cached_features(self, tmp_path):
        run = _tiny_audio_run(tmp_path / "burst")
        cfg = _distill_cfg(tmp_path / "results")
        features = np.arange(20.0).reshape(10, 2)
        valid_mask = np.ones(10, dtype=bool)
        segment_ids = np.zeros(10, dtype=np.int64)
        _write_synthetic_cache(
            cfg, run, "logmel", features=features, valid_mask=valid_mask, segment_ids=segment_ids,
        )

        prepared = distill_beats._load_cache_or_exit(run, "logmel", cfg)

        np.testing.assert_array_equal(prepared.features, features)
        np.testing.assert_array_equal(prepared.valid_mask, valid_mask)
        assert prepared.grid.n_windows == 10

    def test_miss_no_file_exits_naming_warm_cache_hint(self, tmp_path):
        run = _tiny_audio_run(tmp_path / "burst")
        cfg = _distill_cfg(tmp_path / "results")

        with pytest.raises(SystemExit) as exc_info:
            distill_beats._load_cache_or_exit(run, "logmel", cfg)

        message = str(exc_info.value)
        assert "warm_cache.py" in message
        assert "logmel" in message

    def test_miss_stale_fingerprint_exits(self, tmp_path):
        run = _tiny_audio_run(tmp_path / "burst")
        cfg = _distill_cfg(tmp_path / "results")
        cache_path = _write_synthetic_cache(
            cfg, run, "logmel", features=np.zeros((5, 2)),
            valid_mask=np.ones(5, dtype=bool), segment_ids=np.zeros(5, dtype=np.int64),
        )
        with np.load(cache_path, allow_pickle=False) as data:
            raw = dict(data.items())
        raw["fingerprint"] = np.array(["stale-fingerprint"], dtype=str)
        np.savez(cache_path, **raw)

        with pytest.raises(SystemExit) as exc_info:
            distill_beats._load_cache_or_exit(run, "logmel", cfg)

        assert "warm_cache.py" in str(exc_info.value)


# --- 5d. _train_student (MSE distillation) ----------------------------------


class TestTrainStudent:
    def test_loss_decreases_with_training(self):
        pytest.importorskip("torch")
        rng = np.random.default_rng(0)
        # Synthetic teacher = independent random targets (mirrors tests/
        # test_tfc_model.py's/tests/test_adapt_objective.py's own
        # "loss decreases" sanity checks) -- a tiny channel config + enough
        # epochs lets the network memorize this small a batch.
        student_inputs = rng.normal(size=(16, 49 * 64)).astype(np.float64)
        teacher_targets = rng.normal(size=(16, 768)).astype(np.float64)
        cfg = StudentConfig(channels=(4, 8, 16))

        model, losses = distill_beats._train_student(
            student_inputs, teacher_targets, cfg,
            epochs=25, batch_size=8, lr=1e-2, seed=7,
        )

        assert len(losses) == 25
        assert all(np.isfinite(loss_value) for loss_value in losses)
        assert losses[-1] < losses[0]
        assert model.training is False

    def test_determinism_same_seed_identical_losses_and_state_dict(self):
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(1)
        student_inputs = rng.normal(size=(12, 49 * 64)).astype(np.float64)
        teacher_targets = rng.normal(size=(12, 768)).astype(np.float64)
        cfg = StudentConfig(channels=(4, 8, 16))

        model1, losses1 = distill_beats._train_student(
            student_inputs, teacher_targets, cfg, epochs=3, batch_size=4, lr=1e-2, seed=7,
        )
        model2, losses2 = distill_beats._train_student(
            student_inputs, teacher_targets, cfg, epochs=3, batch_size=4, lr=1e-2, seed=7,
        )

        assert losses1 == losses2
        sd1, sd2 = model1.state_dict(), model2.state_dict()
        assert sd1.keys() == sd2.keys()
        for key in sd1:
            torch.testing.assert_close(sd1[key], sd2[key])

    def test_different_seed_produces_different_losses(self):
        pytest.importorskip("torch")
        rng = np.random.default_rng(2)
        student_inputs = rng.normal(size=(12, 49 * 64)).astype(np.float64)
        teacher_targets = rng.normal(size=(12, 768)).astype(np.float64)
        cfg = StudentConfig(channels=(4, 8, 16))

        _model_a, losses_a = distill_beats._train_student(
            student_inputs, teacher_targets, cfg, epochs=3, batch_size=4, lr=1e-2, seed=7,
        )
        _model_b, losses_b = distill_beats._train_student(
            student_inputs, teacher_targets, cfg, epochs=3, batch_size=4, lr=1e-2, seed=8,
        )

        assert losses_a != losses_b

    def test_mismatched_row_counts_raises_value_error(self):
        pytest.importorskip("torch")
        cfg = StudentConfig(channels=(4, 8, 16))
        with pytest.raises(ValueError, match=r"shape\[0\]"):
            distill_beats._train_student(
                np.zeros((4, 49 * 64)), np.zeros((3, 768)), cfg,
                epochs=1, batch_size=2, lr=1e-3, seed=7,
            )


# --- 5e. main() end-to-end -----------------------------------------------


class TestDistillBeatsMainEndToEnd:
    def test_full_run_writes_checkpoint_and_sidecar(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        run = _tiny_audio_run(tmp_path / "burst", name="e2e-run")
        cfg = _distill_cfg(tmp_path / "results")

        rng = np.random.default_rng(3)
        n = 12
        # >= 2 distinct segments so split_by_segments(0.5) yields a non-degenerate
        # calibration/scoring split.
        segment_ids = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
        _write_synthetic_cache(
            cfg, run, "audio-beats",
            features=rng.normal(size=(n, 768)),
            valid_mask=np.ones(n, dtype=bool), segment_ids=segment_ids,
        )
        _write_synthetic_cache(
            cfg, run, "logmel",
            features=rng.normal(size=(n, 49 * 64)),
            valid_mask=np.ones(n, dtype=bool), segment_ids=segment_ids,
        )

        monkeypatch.setattr(distill_beats, "discover", lambda data_root: _fake_index([run]))
        monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
        monkeypatch.setenv("ROWII_RESULTS_ROOT", str(cfg.results_root))
        # This repo's own .env sets a real ROWII_BEATS_CHECKPOINT for dev use
        # (process env > .env, but only once the KEY is present) -- an explicit
        # empty string neutralizes it so main()'s own load_config() computes the
        # SAME fingerprint _write_synthetic_cache used above (mirrors tests/
        # test_adapt_beats.py's identical fix for the identical contamination).
        monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", "")

        out_dir = tmp_path / "out"
        rc = distill_beats.main(
            [
                "--run", run.name, "--epochs", "2", "--batch-size", "4", "--seed", "7",
                "--out", str(out_dir),
            ]
        )

        assert rc == 0
        checkpoint_path = out_dir / f"student_{run.name}.pt"
        sidecar_path = checkpoint_path.with_suffix(".json")
        assert checkpoint_path.is_file()
        assert sidecar_path.is_file()

        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["run"] == run.name
        assert sidecar["teacher_variant"] == "audio-beats"
        assert sidecar["student_input_variant"] == "logmel"
        assert sidecar["epochs"] == 2
        assert sidecar["n_calibration_windows"] > 0
        assert "calibration-side" in sidecar["note"]

        import torch as _torch

        from rowii.adapt.student import load_student_model

        loaded = load_student_model(checkpoint_path, _torch.device("cpu"))
        assert loaded.training is False

    def test_unknown_run_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(distill_beats, "discover", lambda data_root: _fake_index([]))
        monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
        monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
        monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", "")

        with pytest.raises(SystemExit):
            distill_beats.main(["--run", "does-not-exist"])

    def test_missing_cache_exits_naming_warm_cache_hint(self, tmp_path, monkeypatch, capsys):
        run = _tiny_audio_run(tmp_path / "burst", name="no-cache-run")
        monkeypatch.setattr(distill_beats, "discover", lambda data_root: _fake_index([run]))
        monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
        monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
        monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", "")

        with pytest.raises(SystemExit) as exc_info:
            distill_beats.main(["--run", run.name])

        assert "warm_cache.py" in str(exc_info.value)

    def test_help_exits_zero_and_documents_every_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            distill_beats.main(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--run" in out
        assert "--epochs" in out
        assert "--batch-size" in out
        assert "--lr" in out
        assert "--seed" in out
        assert "--out" in out
