"""Tests for the distilled BEATs-student compactness pair:
`rowii.adapt.student`/`rowii.adapt._student_model` (the
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
    feature_names: list[str] | None = None,
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
        feature_names=(
            feature_names
            if feature_names is not None
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


# --- 5a. _check_cache_alignment ---------------------------------------------


class TestCheckCacheAlignment:
    def test_exact_match_is_identity_alignment(self):
        teacher = _prepared_with_grid(_grid())
        student_input = _prepared_with_grid(_grid())
        alignment = distill_beats._check_cache_alignment("run", teacher, student_input)
        assert (alignment.shift, alignment.student_lo, alignment.student_hi) == (0, 0, 10)
        assert alignment.t0_offset_ns == 0

    def test_sub_window_offset_tolerated_with_warning(self, caplog):
        teacher = _prepared_with_grid(_grid(t0_ns=26_000_000))
        student_input = _prepared_with_grid(_grid(t0_ns=0))
        with caplog.at_level(logging.WARNING):
            alignment = distill_beats._check_cache_alignment("run", teacher, student_input)
        assert alignment.t0_offset_ns == 26_000_000
        assert (alignment.shift, alignment.student_lo, alignment.student_hi) == (0, 0, 10)
        assert any("offset" in r.message.lower() for r in caplog.records)

    def test_whole_window_offset_becomes_integer_shift(self, caplog):
        # Teacher grid starts exactly one window later: student j pairs with
        # teacher j-1 over the overlap; the first student window has no partner.
        teacher = _prepared_with_grid(_grid(t0_ns=1_000_000_000))
        student_input = _prepared_with_grid(_grid(t0_ns=0))
        with caplog.at_level(logging.WARNING):
            alignment = distill_beats._check_cache_alignment("run", teacher, student_input)
        assert (alignment.shift, alignment.student_lo, alignment.student_hi) == (-1, 1, 10)
        assert alignment.t0_offset_ns == 0
        assert any("dropp" in r.message.lower() for r in caplog.records)

    def test_n_windows_mismatch_trims_to_overlap(self, caplog):
        # The 010726-pu real-data case: logmel's primary-mic-only
        # grid starts 41 ms earlier and fits one MORE window than audio-beats'
        # both-mics grid -- pair i<->i over the teacher's range, drop the
        # student's extra final window, keep the sub-window residual warning.
        teacher = _prepared_with_grid(_grid(t0_ns=41_000_000, n_windows=10))
        student_input = _prepared_with_grid(_grid(t0_ns=0, n_windows=11))
        with caplog.at_level(logging.WARNING):
            alignment = distill_beats._check_cache_alignment("run", teacher, student_input)
        assert (alignment.shift, alignment.student_lo, alignment.student_hi) == (0, 0, 10)
        assert alignment.t0_offset_ns == 41_000_000
        assert any("dropp" in r.message.lower() for r in caplog.records)

    def test_disjoint_grids_exit_2(self, capsys):
        teacher = _prepared_with_grid(_grid(t0_ns=20_000_000_000, n_windows=10))
        student_input = _prepared_with_grid(_grid(t0_ns=0, n_windows=10))
        with pytest.raises(SystemExit) as exc_info:
            distill_beats._check_cache_alignment("run", teacher, student_input)
        assert exc_info.value.code == 2
        assert "no overlapping window" in capsys.readouterr().err

    def test_structural_window_ns_mismatch_exits_2(self, capsys):
        teacher = _prepared_with_grid(_grid(window_ns=1_000_000_000))
        student_input = _prepared_with_grid(_grid(window_ns=500_000_000))
        with pytest.raises(SystemExit) as exc_info:
            distill_beats._check_cache_alignment("run", teacher, student_input)
        assert exc_info.value.code == 2
        assert "grid mismatch" in capsys.readouterr().err

    def test_teacher_indices_applies_shift(self):
        alignment = distill_beats._CacheAlignment(
            shift=-1, student_lo=1, student_hi=10, t0_offset_ns=0
        )
        student_idx = np.array([1, 4, 9], dtype=np.int64)
        np.testing.assert_array_equal(
            alignment.teacher_indices(student_idx), np.array([0, 3, 8])
        )


# --- 5a2. _teacher_target_columns (primary-mic slice) ------------------------


def _teacher_with_names(feature_names: list[str]) -> PreparedRun:
    n, width = 4, len(feature_names)
    return PreparedRun(
        features=np.zeros((n, width)), grid=_grid(n_windows=n),
        valid_mask=np.ones(n, dtype=bool), feature_names=feature_names,
        segment_ids=np.zeros(n, dtype=np.int64),
    )


class TestTeacherTargetColumns:
    def test_selects_only_primary_stream_columns_by_name_prefix(self):
        teacher = _teacher_with_names(
            ["MicA::beats_0", "MicA::beats_1", "MicB::beats_0", "MicB::beats_1"]
        )
        cols = distill_beats._teacher_target_columns(teacher, "MicA", expected_dim=2)
        np.testing.assert_array_equal(cols, np.array([0, 1]))

    def test_selection_is_by_prefix_not_position(self):
        # Reversed stream order in the cache must still find MicA's columns.
        teacher = _teacher_with_names(
            ["MicB::beats_0", "MicB::beats_1", "MicA::beats_0", "MicA::beats_1"]
        )
        cols = distill_beats._teacher_target_columns(teacher, "MicA", expected_dim=2)
        np.testing.assert_array_equal(cols, np.array([2, 3]))

    def test_no_matching_stream_exits_2(self, capsys):
        teacher = _teacher_with_names(["MicB::beats_0", "MicB::beats_1"])
        with pytest.raises(SystemExit) as exc_info:
            distill_beats._teacher_target_columns(teacher, "MicA", expected_dim=2)
        assert exc_info.value.code == 2
        assert "MicA" in capsys.readouterr().err

    def test_width_mismatch_with_student_out_dim_exits_2(self, capsys):
        # 4 teacher columns for the stream but a 2-d student head: geometry error.
        teacher = _teacher_with_names(
            ["MicA::beats_0", "MicA::beats_1", "MicA::beats_2", "MicA::beats_3"]
        )
        with pytest.raises(SystemExit) as exc_info:
            distill_beats._teacher_target_columns(teacher, "MicA", expected_dim=2)
        assert exc_info.value.code == 2
        assert "out_dim" in capsys.readouterr().err


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

        alignment = distill_beats._check_cache_alignment("run", teacher, student_input)
        result = distill_beats._select_calibration_windows(
            student_input, teacher, seed=7, alignment=alignment
        )

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

        alignment = distill_beats._check_cache_alignment("run", prepared, prepared)
        distill_beats._select_calibration_windows(
            prepared, prepared, seed=99, alignment=alignment
        )

        assert calls == [(0.5, 99)]

    def test_combines_both_caches_valid_masks(self, monkeypatch):
        # A window valid for logmel but NOT for audio-beats must never reach
        # split_by_segments' valid_mask as True -- the teacher target would be
        # undefined (NaN) there (module docstring's rationale for why the AND
        # is a safe extension of, not a departure from, the leakage rule).
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

        alignment = distill_beats._check_cache_alignment("run", teacher, student_input)
        distill_beats._select_calibration_windows(
            student_input, teacher, seed=7, alignment=alignment
        )

        np.testing.assert_array_equal(seen_masks[0], np.array([True, False, True, True]))

    def test_alignment_trim_masks_unpaired_student_windows(self, monkeypatch):
        # Student cache has one MORE window than the teacher (the 010726-pu
        # case): the unpaired final student window must reach split_by_segments
        # as invalid even though both caches' own masks are all-True there.
        student_input = PreparedRun(
            features=np.zeros((5, 1)), grid=_grid(n_windows=5),
            valid_mask=np.ones(5, dtype=bool), feature_names=["f0"],
            segment_ids=np.array([0, 0, 1, 1, 1], dtype=np.int64),
        )
        teacher = PreparedRun(
            features=np.zeros((4, 1)), grid=_grid(n_windows=4),
            valid_mask=np.ones(4, dtype=bool), feature_names=["f0"],
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

        alignment = distill_beats._check_cache_alignment("run", teacher, student_input)
        distill_beats._select_calibration_windows(
            student_input, teacher, seed=7, alignment=alignment
        )

        np.testing.assert_array_equal(
            seen_masks[0], np.array([True, True, True, True, False])
        )

    def test_alignment_shift_reads_teacher_mask_at_shifted_index(self, monkeypatch):
        # shift=-1 (teacher grid one window later): student window j must AND
        # against teacher.valid_mask[j-1], and the unpaired student window 0
        # must come out invalid.
        student_input = PreparedRun(
            features=np.zeros((4, 1)), grid=_grid(t0_ns=0, n_windows=4),
            valid_mask=np.ones(4, dtype=bool), feature_names=["f0"],
            segment_ids=np.array([0, 0, 1, 1], dtype=np.int64),
        )
        teacher = PreparedRun(
            features=np.zeros((4, 1)), grid=_grid(t0_ns=1_000_000_000, n_windows=4),
            valid_mask=np.array([True, False, True, True]), feature_names=["f0"],
            segment_ids=np.array([0, 0, 1, 1], dtype=np.int64),
        )
        seen_masks: list[np.ndarray] = []

        def fake_split_by_segments(segment_ids, valid_mask, calibration_frac, seed):
            seen_masks.append(valid_mask.copy())
            return SegmentSplit(
                calibration_windows=np.array([2], dtype=np.int64),
                scoring_windows=np.array([3], dtype=np.int64),
            )

        monkeypatch.setattr(distill_beats, "split_by_segments", fake_split_by_segments)

        alignment = distill_beats._check_cache_alignment("run", teacher, student_input)
        assert (alignment.shift, alignment.student_lo, alignment.student_hi) == (-1, 1, 4)
        distill_beats._select_calibration_windows(
            student_input, teacher, seed=7, alignment=alignment
        )

        # Student j=1 -> teacher 0 (True), j=2 -> teacher 1 (False), j=3 -> teacher 2
        # (True); student 0 has no teacher partner.
        np.testing.assert_array_equal(
            seen_masks[0], np.array([False, True, False, True])
        )


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
        # Faithful to the real audio-beats cache: BOTH mic streams' 768-d
        # embeddings concatenated (1536 columns), stream-prefixed names -- the
        # script must slice out the primary mic's 768, never regress onto 1536.
        _write_synthetic_cache(
            cfg, run, "audio-beats",
            features=rng.normal(size=(n, 1536)),
            valid_mask=np.ones(n, dtype=bool), segment_ids=segment_ids,
            feature_names=(
                [f"RAWGeneratorMic__0::beats_{i}" for i in range(768)]
                + [f"RAWTurbineMic__1::beats_{i}" for i in range(768)]
            ),
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
        assert sidecar["teacher_stream"] == "RAWGeneratorMic__0"
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


# ---------------------------------------------------------------------------
# 5f. --runs: multi-run stacking: per run, BOTH caches load (cache-only) and
#     alignment-check
#     exactly as in single-run mode, the per-run top split selects
#     calibration-side rows, and the selected student inputs + teacher
#     primary-mic slices are STACKED across runs into ONE training set. The
#     tests below capture `_train_student`'s inputs to pin the stacked
#     matrices bitwise -- including the 1-run-list parity case against the
#     unchanged single-run path.
# ---------------------------------------------------------------------------

_GEN_TUR_BEATS_NAMES: list[str] = (
    [f"RAWGeneratorMic__0::beats_{i}" for i in range(768)]
    + [f"RAWTurbineMic__1::beats_{i}" for i in range(768)]
)


def _write_run_with_cache_pair(
    tmp_path: Path,
    cfg: Config,
    name: str,
    *,
    n: int,
    seed: int,
    tur_shift_s: float = 0.0,
):
    """One run + BOTH synthetic caches (audio-beats teacher with gen+tur
    stream-prefixed names, logmel student input) -- the multi-run tests'
    per-run building block, mirroring TestDistillBeatsMainEndToEnd's own
    fixture construction.

    *tur_shift_s* > 0 starts the TURBINE mic's burst file that many seconds
    after the generator mic's, which misaligns the two variants' grids at
    the SOURCE: `rowii.pipeline._load_cached_prepared_run` overrides a
    cached `grid_t0_ns` with the freshly recomputed true-UTC t0 on every
    hit (the DAQ epoch-2000 compatibility path), so a t0 shift written only
    into the cache npz would be erased at load time -- the audio-beats grid
    (both-mic intersection, `build_run_grid`) then genuinely starts
    *tur_shift_s* later than the logmel grid (primary mic alone).
    """
    from datetime import timedelta

    burst_dir = tmp_path / f"burst-{name}"
    if tur_shift_s == 0.0:
        run = _tiny_audio_run(burst_dir, name=name)
    else:
        burst_dir.mkdir(parents=True, exist_ok=True)
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        files = {
            "RAWGeneratorMic__0": [
                _cache_burst(
                    burst_dir / "gen_mic.dat", "RAWGeneratorMic__0", 2.0 + tur_shift_s,
                    t0_ns=0, start_utc_hint=t0,
                )
            ],
            "RAWTurbineMic__1": [
                _cache_burst(
                    burst_dir / "tur_mic.dat", "RAWTurbineMic__1", 2.0,
                    t0_ns=round(tur_shift_s * 1e9),
                    start_utc_hint=t0 + timedelta(seconds=tur_shift_s),
                )
            ],
        }
        run = Run(name=name, files=files, day_root=burst_dir)
    rng = np.random.default_rng(seed)
    segment_ids = np.repeat([0, 1], n // 2).astype(np.int64)
    _write_synthetic_cache(
        cfg, run, "audio-beats",
        features=rng.normal(size=(n, 1536)),
        valid_mask=np.ones(n, dtype=bool), segment_ids=segment_ids,
        feature_names=list(_GEN_TUR_BEATS_NAMES),
    )
    _write_synthetic_cache(
        cfg, run, "logmel",
        features=rng.normal(size=(n, 49 * 64)),
        valid_mask=np.ones(n, dtype=bool), segment_ids=segment_ids,
    )
    return run


class _TrainStubModel:
    """Minimal stand-in for the trained `_StudentNet` in tests that capture
    `_train_student`'s inputs: only `state_dict()` is touched afterwards
    (`_save_checkpoint`'s torch.save)."""

    def state_dict(self):
        return {}


def _capture_train_student(monkeypatch) -> list[tuple[np.ndarray, np.ndarray]]:
    """Monkeypatches `distill_beats._train_student` to record each call's
    `(student_inputs, teacher_targets)` (copied) and skip real training."""
    captured: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_train(student_inputs, teacher_targets, cfg, *, epochs, batch_size, lr, seed):
        captured.append((student_inputs.copy(), teacher_targets.copy()))
        return _TrainStubModel(), [0.123]

    monkeypatch.setattr(distill_beats, "_train_student", fake_train)
    return captured


def _multi_run_env(monkeypatch, tmp_path: Path, cfg: Config, runs: list[Run]) -> None:
    """The established e2e env recipe (TestDistillBeatsMainEndToEnd), shared
    by the multi-run tests: monkeypatched discover + env vars, with
    ROWII_BEATS_CHECKPOINT explicitly emptied so main()'s own load_config()
    computes the SAME cache fingerprint `_write_synthetic_cache` used."""
    monkeypatch.setattr(distill_beats, "discover", lambda data_root: _fake_index(runs))
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(cfg.results_root))
    monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", "")


def test_distill_run_and_runs_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as exc_info:
        distill_beats.build_parser().parse_args(["--run", "x", "--runs", "a,b"])
    assert exc_info.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_distill_one_of_run_or_runs_is_required(capsys):
    with pytest.raises(SystemExit) as exc_info:
        distill_beats.build_parser().parse_args([])
    assert exc_info.value.code == 2
    assert "--runs" in capsys.readouterr().err


def test_distill_help_documents_runs_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        distill_beats.main(["--help"])
    assert exc_info.value.code == 0
    # "--runs NAMES" is the option's own usage form -- the bare string
    # "--runs" already appears in --run's warm_cache.py hint, so it would
    # not distinguish a parser that actually has the flag.
    assert "--runs NAMES" in capsys.readouterr().out


def test_distill_runs_with_duplicate_names_is_a_parser_error(capsys):
    with pytest.raises(SystemExit) as exc_info:
        distill_beats.main(["--runs", "run-a,run-a"])
    assert exc_info.value.code == 2
    assert "duplicate" in capsys.readouterr().err.lower()


def test_multi_run_single_item_list_matches_single_run_bitwise(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    cfg = _distill_cfg(tmp_path / "results")
    run = _write_run_with_cache_pair(tmp_path, cfg, "parity-run", n=12, seed=11)
    _multi_run_env(monkeypatch, tmp_path, cfg, [run])
    captured = _capture_train_student(monkeypatch)

    rc_single = distill_beats.main(
        ["--run", run.name, "--epochs", "1", "--out", str(tmp_path / "out-single")]
    )
    rc_multi = distill_beats.main(
        ["--runs", run.name, "--epochs", "1", "--out", str(tmp_path / "out-multi")]
    )

    assert rc_single == 0 and rc_multi == 0
    assert len(captured) == 2
    (single_x, single_y), (multi_x, multi_y) = captured
    assert single_x.dtype == multi_x.dtype and single_y.dtype == multi_y.dtype
    np.testing.assert_array_equal(single_x, multi_x, err_msg=(
        "--runs with a 1-run list must hand _train_student the BITWISE-same student "
        "inputs as the single-run path"
    ))
    np.testing.assert_array_equal(single_y, multi_y, err_msg=(
        "--runs with a 1-run list must hand _train_student the BITWISE-same teacher "
        "targets as the single-run path"
    ))

    single_sidecar = json.loads(
        (tmp_path / "out-single" / f"student_{run.name}.json").read_text()
    )
    multi_sidecar = json.loads(
        (tmp_path / "out-multi" / f"student_{run.name}.json").read_text()
    )
    assert single_sidecar["run"] == run.name
    assert "runs" not in single_sidecar, "single-run sidecar stays exactly as before"
    assert multi_sidecar["runs"] == [run.name]
    assert multi_sidecar["calibration_windows_per_run"] == {
        run.name: single_sidecar["n_calibration_windows"]
    }
    assert (
        multi_sidecar["n_calibration_windows"] == single_sidecar["n_calibration_windows"]
    )


def test_multi_run_stacks_calibration_rows_across_runs_in_runs_order(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    cfg = _distill_cfg(tmp_path / "results")
    run1 = _write_run_with_cache_pair(tmp_path, cfg, "stack-run-1", n=12, seed=21)
    run2 = _write_run_with_cache_pair(tmp_path, cfg, "stack-run-2", n=8, seed=22)
    _multi_run_env(monkeypatch, tmp_path, cfg, [run1, run2])
    captured = _capture_train_student(monkeypatch)

    out_dir = tmp_path / "out"
    rc = distill_beats.main(
        ["--runs", f"{run1.name},{run2.name}", "--epochs", "1", "--out", str(out_dir)]
    )
    assert rc == 0

    expected_x_blocks: list[np.ndarray] = []
    expected_y_blocks: list[np.ndarray] = []
    expected_counts: dict[str, int] = {}
    for run in (run1, run2):
        teacher = distill_beats._load_cache_or_exit(run, "audio-beats", cfg)
        student_input = distill_beats._load_cache_or_exit(run, "logmel", cfg)
        alignment = distill_beats._check_cache_alignment(run.name, teacher, student_input)
        calib = distill_beats._select_calibration_windows(
            student_input, teacher, seed=7, alignment=alignment
        )
        assert calib.size > 0, "fixture sanity: every run must contribute rows"
        expected_x_blocks.append(student_input.features[calib])
        # Primary-mic slice hand-pinned from the fixture's feature-name order
        # (generator mic first): columns 0..767 -- independent ground truth,
        # not routed through _teacher_target_columns.
        expected_y_blocks.append(teacher.features[calib][:, :768])
        expected_counts[run.name] = int(calib.size)

    assert len(captured) == 1, "multi-run mode trains ONE student on the stacked pool"
    got_x, got_y = captured[0]
    np.testing.assert_array_equal(got_x, np.vstack(expected_x_blocks))
    np.testing.assert_array_equal(got_y, np.vstack(expected_y_blocks))

    checkpoint_path = out_dir / f"student_{run1.name}+{run2.name}.pt"
    assert checkpoint_path.is_file(), "multi-run checkpoint name joins run names with '+'"
    sidecar = json.loads(checkpoint_path.with_suffix(".json").read_text())
    assert sidecar["runs"] == [run1.name, run2.name]
    assert sidecar["calibration_windows_per_run"] == expected_counts
    assert sidecar["n_calibration_windows"] == sum(expected_counts.values())


def test_multi_run_disjoint_second_run_exits_2_before_training(
    tmp_path, monkeypatch, capsys
):
    pytest.importorskip("torch")
    cfg = _distill_cfg(tmp_path / "results")
    run1 = _write_run_with_cache_pair(tmp_path, cfg, "align-run-1", n=8, seed=31)
    # run2's turbine mic starts a full 8 windows (= n) after its generator mic,
    # so the freshly recomputed audio-beats grid (both-mic intersection) shares
    # NO window with the logmel grid (primary mic alone) -- whole-window shifts
    # within the overlap are now paired by _CacheAlignment, but a fully
    # disjoint pair remains the hard refusal (see _write_run_with_cache_pair's
    # docstring for why the misalignment must live in the burst files, not the
    # cache npz).
    run2 = _write_run_with_cache_pair(
        tmp_path, cfg, "align-run-2", n=8, seed=32, tur_shift_s=8.0
    )
    _multi_run_env(monkeypatch, tmp_path, cfg, [run1, run2])
    captured = _capture_train_student(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        distill_beats.main(
            ["--runs", f"{run1.name},{run2.name}", "--out", str(tmp_path / "out")]
        )

    assert exc_info.value.code == 2
    assert "no overlapping window" in capsys.readouterr().err
    assert captured == [], "the per-run alignment guard must fire BEFORE any training"


def test_seed_not_seven_sidecar_note_carries_caveat_and_warns(
    tmp_path, monkeypatch, caplog
):
    """The leakage note is persisted into
    provenance -- at seed != 7 it must carry the does-NOT-match caveat and the
    CLI must warn; the previous static string was silently false."""
    import logging

    pytest.importorskip("torch")
    run = _tiny_audio_run(tmp_path / "burst", name="seed-run")
    cfg = _distill_cfg(tmp_path / "results")
    rng = np.random.default_rng(3)
    n = 12
    segment_ids = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    _write_synthetic_cache(
        cfg, run, "audio-beats",
        features=rng.normal(size=(n, 1536)),
        valid_mask=np.ones(n, dtype=bool), segment_ids=segment_ids,
        feature_names=(
            [f"RAWGeneratorMic__0::beats_{i}" for i in range(768)]
            + [f"RAWTurbineMic__1::beats_{i}" for i in range(768)]
        ),
    )
    _write_synthetic_cache(
        cfg, run, "logmel",
        features=rng.normal(size=(n, 49 * 64)),
        valid_mask=np.ones(n, dtype=bool), segment_ids=segment_ids,
    )
    monkeypatch.setattr(distill_beats, "discover", lambda data_root: _fake_index([run]))
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(cfg.results_root))
    monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", "")

    out_dir = tmp_path / "out"
    with caplog.at_level(logging.WARNING):
        rc = distill_beats.main(
            [
                "--run", run.name, "--epochs", "1", "--batch-size", "4",
                "--seed", "9", "--out", str(out_dir),
            ]
        )
    assert rc == 0
    sidecar = json.loads((out_dir / f"student_{run.name}.json").read_text())
    assert "seed=9" in sidecar["note"]
    assert "does NOT match" in sidecar["note"]
    warned = [r.getMessage() for r in caplog.records]
    assert any("canonical seed-7" in m for m in warned)

    # The canonical seed keeps the unqualified claim (no caveat).
    with caplog.at_level(logging.WARNING):
        rc = distill_beats.main(
            [
                "--run", run.name, "--epochs", "1", "--batch-size", "4",
                "--seed", "7", "--out", str(tmp_path / "out7"),
            ]
        )
    assert rc == 0
    sidecar7 = json.loads((tmp_path / "out7" / f"student_{run.name}.json").read_text())
    assert "seed=7" in sidecar7["note"]
    assert "does NOT match" not in sidecar7["note"]


def test_distill_runs_with_only_blank_names_is_a_parser_error(capsys):
    with pytest.raises(SystemExit) as exc_info:
        distill_beats.main(["--runs", "  ,  ,", "--out", "/tmp/unused"])
    assert exc_info.value.code == 2


def test_multi_run_zero_contribution_warning_fires_for_a_non_last_run(
    tmp_path, monkeypatch, caplog
):
    """The zero-contribution warning sat
    OUTSIDE the pool loop and therefore only ever inspected the LAST run's
    count -- a zero-contribution run in any earlier position was silently
    unflagged (the never-silently-absent principle). Pin: run 1 contributes
    zero, run 2 contributes rows -> the warning fires and names run 1."""
    import logging

    pytest.importorskip("torch")
    cfg = _distill_cfg(tmp_path / "results")
    run1 = _write_run_with_cache_pair(tmp_path, cfg, "zero-run", n=8, seed=41)
    run2 = _write_run_with_cache_pair(tmp_path, cfg, "rich-run", n=8, seed=42)
    _multi_run_env(monkeypatch, tmp_path, cfg, [run1, run2])
    captured = _capture_train_student(monkeypatch)

    real_select = distill_beats._select_calibration_windows

    def fake_select(student_input, teacher, *, seed, alignment):
        # Deterministically starve the FIRST run only.
        if fake_select.calls == 0:
            fake_select.calls += 1
            return np.empty(0, dtype=np.int64)
        fake_select.calls += 1
        return real_select(student_input, teacher, seed=seed, alignment=alignment)

    fake_select.calls = 0
    monkeypatch.setattr(distill_beats, "_select_calibration_windows", fake_select)

    with caplog.at_level(logging.WARNING):
        rc = distill_beats.main(
            ["--runs", f"{run1.name},{run2.name}", "--out", str(tmp_path / "out")]
        )
    assert rc == 0

    zero_warnings = [
        r.message for r in caplog.records if "ZERO calibration-side" in r.message
    ]
    assert any("zero-run" in m for m in zero_warnings), (
        "the zero-contribution warning must fire for a NON-LAST pool run"
    )
    assert not any("rich-run" in m for m in zero_warnings)
    # Training saw only the rich run's rows; the sidecar records both counts.
    assert len(captured) == 1
    import json as _json
    sidecar = _json.loads(
        (tmp_path / "out" / f"student_{run1.name}+{run2.name}.json").read_text()
    )
    assert sidecar["calibration_windows_per_run"][run1.name] == 0
    assert sidecar["calibration_windows_per_run"][run2.name] > 0
