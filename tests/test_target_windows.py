"""Tests for `rowii.adapt.target_windows.iter_target_windows` (Step-2
package-5 Task 2, design spec D3: `docs/superpowers/specs/2026-07-16-step2-
package5-adaptation-design.md`): the leakage-aware, primary-mic-stream,
16 kHz target-normal training-window iterator BEATs adaptation draws from.

Torch-free module under test -- no `pytest.importorskip("torch")` needed,
unlike `tests/test_adapt_lora.py`/`tests/test_adapt_objective.py` (Task 1's
eager-torch modules); `iter_target_windows` never imports torch at all.

No `@pytest.mark.data`: every run here is a synthetic gantner tree built via
`tests/fixtures/gantner_builder.build_gantner_file`, the SAME fixture (and
the same hand-built-`Run`-bypassing-`discover()` convention)
`tests/test_pipeline.py` uses for `rowii.pipeline.prepare_run`'s own unit
tests.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from rowii.adapt import target_windows
from rowii.adapt.target_windows import iter_target_windows
from rowii.anomaly.references import SegmentSplit
from rowii.config import Config
from rowii.io.dataset import BurstFile, Run
from rowii.pipeline import PreparedRun, prepare_run
from tests.fixtures.gantner_builder import build_gantner_file

_RATE_HZ = 8000.0
"""Synthetic files' own sample rate -- deliberately != 16_000 (the default
`target_hz`) so every test genuinely exercises `_resample_window`'s
upsampling path (16000/8000 = an exact 2x ratio after gcd reduction), not a
degenerate no-op resample."""
_CHANNELS = 4
_SECONDS_PER_SEGMENT = 6
_N_SEGMENTS = 4
_T0 = datetime(2026, 1, 1, tzinfo=UTC)
"""Midnight UTC -- exactly hour-aligned, so `rowii.io.dataset.
run_utc_offset_ns`'s per-file deviation-from-rounded-offset warning never
fires here (mirrors `tests/test_pipeline.py`'s own `t0` choice; see that
module's fixtures for the same reasoning)."""


def _burst(
    path: Path, stream: str, n_seconds: int, *, t0_ns: int, start_utc_hint: datetime,
    rng: np.random.Generator,
) -> BurstFile:
    """One synthetic burst file: *n_seconds* of NON-constant (seeded random
    normal) data on `_CHANNELS` channels. Non-constant matters: a constant
    window standardizes to all-zeros (std floored to 1e-8, so
    `(x - mean) / 1e-8 == 0` everywhere), which would silently defeat this
    file's "standardized output has std ~= 1" assertions.
    """
    n_samples = round(_RATE_HZ * n_seconds)
    data = rng.standard_normal((n_samples, _CHANNELS)).astype(np.float32)
    build_gantner_file(path, ["ch0", "ch1", "ch2", "ch3"], data, t0_ns=t0_ns, rate_hz=_RATE_HZ)
    return BurstFile(path=path, stream=stream, start_utc_hint=start_utc_hint)


def _build_run(
    tmp_path: Path,
    *,
    n_segments: int = _N_SEGMENTS,
    seconds_per_segment: int = _SECONDS_PER_SEGMENT,
    name: str = "target-windows-run",
) -> Run:
    """*n_segments* back-to-back files per mic stream (`RAWGeneratorMic__0`/
    `RAWTurbineMic__1` -- both required, `_streams_for_variant("audio")`),
    each its own primary-stream "segment" (`PreparedRun.segment_ids`):
    contiguous in time, full coverage, no invalid windows. Built by hand
    (`Run`/`BurstFile` constructed directly), bypassing
    `rowii.io.dataset.discover()` entirely -- mirrors `tests/test_pipeline.
    py`'s own `_single_file_audio_run` convention (a `Run` is cheap to build
    by hand for a focused unit test); the >15-min gap-splitting
    `discover()` does is irrelevant here since segment ids come from
    per-FILE indices, not from time gaps.
    """
    burst_dir = tmp_path / "burst"
    burst_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260716)
    gen_files: list[BurstFile] = []
    tur_files: list[BurstFile] = []
    for k in range(n_segments):
        t0_ns = k * seconds_per_segment * 1_000_000_000
        hint = _T0 + timedelta(seconds=k * seconds_per_segment)
        gen_files.append(
            _burst(
                burst_dir / f"gen_{k}.dat", "RAWGeneratorMic__0", seconds_per_segment,
                t0_ns=t0_ns, start_utc_hint=hint, rng=rng,
            )
        )
        tur_files.append(
            _burst(
                burst_dir / f"tur_{k}.dat", "RAWTurbineMic__1", seconds_per_segment,
                t0_ns=t0_ns, start_utc_hint=hint, rng=rng,
            )
        )
    return Run(
        name=name,
        files={"RAWGeneratorMic__0": gen_files, "RAWTurbineMic__1": tur_files},
        day_root=burst_dir,
    )


def _cfg(tmp_path: Path) -> Config:
    return Config(data_root=tmp_path, results_root=tmp_path / "results")


def _split_by_known_segments(prepared: PreparedRun, calib_segment_ids: set[int]) -> SegmentSplit:
    """A REAL, valid `SegmentSplit` restricted to a caller-chosen set of
    segment (file) ids -- mirrors `split_by_segments`'s own calibration/
    scoring mask construction (spec D3) exactly, but with the partition
    chosen directly instead of coming from the RNG-driven segment shuffle.
    Used to monkeypatch a fully deterministic, known-in-advance split so
    tests can assert precisely which windows are and are not yielded.
    """
    calib_ids = list(calib_segment_ids)
    calib_mask = prepared.valid_mask & np.isin(prepared.segment_ids, calib_ids)
    scoring_mask = prepared.valid_mask & ~np.isin(prepared.segment_ids, calib_ids)
    return SegmentSplit(
        calibration_windows=np.flatnonzero(calib_mask).astype(np.int64),
        scoring_windows=np.flatnonzero(scoring_mask).astype(np.int64),
    )


# ---------------------------------------------------------------------------
# 1. Leakage: only calibration-side windows are ever yielded (spec D3)
# ---------------------------------------------------------------------------


def test_only_calibration_side_windows_are_yielded(tmp_path, monkeypatch) -> None:
    run = _build_run(tmp_path)
    cfg = _cfg(tmp_path)
    prepared = prepare_run(run, "audio", cfg, use_cache=False)

    known_split = _split_by_known_segments(prepared, calib_segment_ids={0, 2})
    assert known_split.calibration_windows.size > 0
    assert known_split.scoring_windows.size > 0

    calls: list[tuple[float, int]] = []

    def fake_split_by_segments(segment_ids, valid_mask, calibration_frac, seed):
        calls.append((calibration_frac, seed))
        return known_split

    monkeypatch.setattr(target_windows, "split_by_segments", fake_split_by_segments)

    yielded = list(iter_target_windows(run, cfg, return_indices=True))
    yielded_indices = [idx for idx, _window in yielded]

    assert yielded_indices == known_split.calibration_windows.tolist(), (
        "must yield exactly the calibration-side window indices, in ascending order"
    )
    scoring_set = set(known_split.scoring_windows.tolist())
    assert not (set(yielded_indices) & scoring_set), (
        "adaptation must never see a scoring-side window (spec D3 leakage rule)"
    )
    assert calls == [(0.5, 7)], (
        "must call split_by_segments(segment_ids, valid_mask, 0.5, seed) with the "
        "default seed=7 when the caller does not override it"
    )


def test_custom_seed_is_forwarded_to_split_by_segments(tmp_path, monkeypatch) -> None:
    run = _build_run(tmp_path)
    cfg = _cfg(tmp_path)
    prepared = prepare_run(run, "audio", cfg, use_cache=False)
    known_split = _split_by_known_segments(prepared, calib_segment_ids={1, 3})

    calls: list[tuple[float, int]] = []

    def fake_split_by_segments(segment_ids, valid_mask, calibration_frac, seed):
        calls.append((calibration_frac, seed))
        return known_split

    monkeypatch.setattr(target_windows, "split_by_segments", fake_split_by_segments)

    list(iter_target_windows(run, cfg, seed=99))

    assert calls == [(0.5, 99)]


# ---------------------------------------------------------------------------
# 2. Output contract: (16000,) float32, standardized (mean ~= 0, std ~= 1)
# ---------------------------------------------------------------------------


def test_output_contract_shape_dtype_and_standardization(tmp_path) -> None:
    run = _build_run(tmp_path)
    cfg = _cfg(tmp_path)

    windows = list(iter_target_windows(run, cfg))

    assert len(windows) > 0
    for window in windows:
        assert isinstance(window, np.ndarray)
        assert window.shape == (16_000,)
        assert window.dtype == np.float32
        assert abs(float(window.mean())) < 1e-3
        assert abs(float(window.std()) - 1.0) < 1e-2


def test_default_return_type_yields_bare_arrays_not_index_pairs(tmp_path) -> None:
    run = _build_run(tmp_path)
    cfg = _cfg(tmp_path)

    first = next(iter(iter_target_windows(run, cfg)))

    assert isinstance(first, np.ndarray)


# ---------------------------------------------------------------------------
# 3. Determinism: two independent calls yield identical sequences
# ---------------------------------------------------------------------------


def test_determinism_across_two_calls(tmp_path) -> None:
    run = _build_run(tmp_path)
    cfg = _cfg(tmp_path)

    first = np.stack(list(iter_target_windows(run, cfg)))
    second = np.stack(list(iter_target_windows(run, cfg)))

    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# 4. max_windows truncates to the first N yielded, deterministic order
# ---------------------------------------------------------------------------


def test_max_windows_truncates_to_first_n_yielded(tmp_path, monkeypatch) -> None:
    run = _build_run(tmp_path)
    cfg = _cfg(tmp_path)
    prepared = prepare_run(run, "audio", cfg, use_cache=False)
    known_split = _split_by_known_segments(prepared, calib_segment_ids={0, 1, 2, 3})
    assert known_split.calibration_windows.size > 5, "fixture must offer more than N windows"

    monkeypatch.setattr(
        target_windows, "split_by_segments",
        lambda segment_ids, valid_mask, calibration_frac, seed: known_split,
    )

    n = 5
    truncated_indices = [
        idx for idx, _window in iter_target_windows(run, cfg, max_windows=n, return_indices=True)
    ]

    assert truncated_indices == known_split.calibration_windows[:n].tolist()


def test_max_windows_zero_yields_nothing_and_opens_no_files(tmp_path, monkeypatch) -> None:
    run = _build_run(tmp_path)
    cfg = _cfg(tmp_path)
    prepared = prepare_run(run, "audio", cfg, use_cache=False)
    known_split = _split_by_known_segments(prepared, calib_segment_ids={0, 1, 2, 3})

    monkeypatch.setattr(
        target_windows, "split_by_segments",
        lambda segment_ids, valid_mask, calibration_frac, seed: known_split,
    )
    calls: list[Path] = []
    monkeypatch.setattr(target_windows, "read_gantner", lambda path: calls.append(path))

    result = list(iter_target_windows(run, cfg, max_windows=0))

    assert result == []
    assert calls == []


# ---------------------------------------------------------------------------
# 5. Efficiency: each touched primary-mic file is read exactly once;
#    untouched (scoring-only) files are never opened at all.
# ---------------------------------------------------------------------------


def test_reads_each_touched_primary_file_exactly_once_and_skips_untouched_files(
    tmp_path, monkeypatch
) -> None:
    run = _build_run(tmp_path)
    cfg = _cfg(tmp_path)
    prepared = prepare_run(run, "audio", cfg, use_cache=False)
    # Segments 0 and 2 only -- segments 1 and 3's files must never be opened.
    known_split = _split_by_known_segments(prepared, calib_segment_ids={0, 2})
    monkeypatch.setattr(
        target_windows, "split_by_segments",
        lambda segment_ids, valid_mask, calibration_frac, seed: known_split,
    )

    real_read_gantner = target_windows.read_gantner
    calls: list[Path] = []

    def spy(path):
        calls.append(path)
        return real_read_gantner(path)

    monkeypatch.setattr(target_windows, "read_gantner", spy)

    list(iter_target_windows(run, cfg))

    sorted_gen_files = sorted(run.files["RAWGeneratorMic__0"], key=lambda f: f.start_utc_hint)
    expected_paths = {sorted_gen_files[0].path, sorted_gen_files[2].path}
    assert set(calls) == expected_paths, "only files backing a calibration window may be opened"
    assert len(calls) == len(expected_paths), "each touched file must be read exactly once"
