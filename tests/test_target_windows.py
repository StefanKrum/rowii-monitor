"""Tests for `rowii.adapt.target_windows.iter_target_windows`: the leakage-aware,
primary-mic-stream, 16 kHz target-normal training-window iterator BEATs adaptation draws from.

Torch-free module under test -- no `pytest.importorskip("torch")` needed,
unlike `tests/test_adapt_lora.py`/`tests/test_adapt_objective.py` (the
eager-torch modules); `iter_target_windows` never imports torch at all.

No `@pytest.mark.data`: every run here is a synthetic gantner tree built via
`tests/fixtures/gantner_builder.build_gantner_file`, the SAME fixture (and
the same hand-built-`Run`-bypassing-`discover()` convention)
`tests/test_pipeline.py` uses for `rowii.pipeline.prepare_run`'s own unit
tests.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

from rowii.adapt import target_windows
from rowii.adapt.target_windows import iter_target_windows
from rowii.anomaly.references import SegmentSplit
from rowii.config import Config
from rowii.io.dataset import BurstFile, Run
from rowii.io.gantner import read_gantner
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
    scoring mask construction exactly, but with the partition
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
# 1. Leakage: only calibration-side windows are ever yielded
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


# ---------------------------------------------------------------------------
# 6. Content correctness at a file boundary inside the jitter band (review
#    HIGH, reviewer-constructed counterexample): `PreparedRun.segment_ids`'
#    attribution semantic is earliest-file-with-ANY-overlap, while the actual
#    feature/valid_mask for a boundary-straddling window comes from whichever
#    file holds a FULL (within +/-4 samples) slice -- the iterator must yield
#    the FULL file's content, never a near-empty slice zero-padded to length.
# ---------------------------------------------------------------------------

_BOUNDARY_RATE_HZ = 1000.0
"""Boundary/gap fixtures' own sample rate: 1 kHz -> exactly 1000 expected
samples per 1-s window, so a boundary "3 samples into a window" is easy to
place by sample count (1 sample per ms; `t0_ns` steps of 1_000_000)."""


def _mk_raw_file(path: Path, n_samples: int, t0_ns: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_samples, _CHANNELS)).astype(np.float32)
    build_gantner_file(
        path, ["ch0", "ch1", "ch2", "ch3"], data, t0_ns=t0_ns, rate_hz=_BOUNDARY_RATE_HZ
    )
    return path


def _build_boundary_run(tmp_path: Path) -> Run:
    """Primary mic split across two files with the boundary 3 samples INTO
    window 2 (the reviewer's counterexample): file A holds samples
    [0, 2003) -- so only 3 samples of window 2 ([2000 ms, 3000 ms)) -- and
    file B holds samples [2003, 6000), i.e. 997 of window 2's expected 1000
    samples, within `rowii.pipeline._SAMPLE_JITTER_TOLERANCE` (+/-4). The
    pipeline attributes window 2 to file A (earliest ANY-overlap) but
    features/validates it from file B (the only FULL slice). Turbine mic:
    one contiguous file, so validity is governed purely by the gen mic.
    """
    work = tmp_path / "boundary"
    work.mkdir(parents=True, exist_ok=True)
    a = _mk_raw_file(work / "gen_a.dat", 2003, t0_ns=0, seed=1)
    b = _mk_raw_file(work / "gen_b.dat", 3997, t0_ns=2003 * 1_000_000, seed=2)
    t = _mk_raw_file(work / "tur.dat", 6000, t0_ns=0, seed=3)
    gen = [
        BurstFile(path=a, stream="RAWGeneratorMic__0", start_utc_hint=_T0),
        BurstFile(
            path=b, stream="RAWGeneratorMic__0",
            start_utc_hint=_T0 + timedelta(microseconds=2003 * 1000),
        ),
    ]
    tur = [BurstFile(path=t, stream="RAWTurbineMic__1", start_utc_hint=_T0)]
    return Run(
        name="boundary-run",
        files={"RAWGeneratorMic__0": gen, "RAWTurbineMic__1": tur},
        day_root=work,
    )


def _reference_window(raw: np.ndarray, rate_hz: float, target_hz: int = 16_000) -> np.ndarray:
    """Independent reference computation of the iterator's per-window output
    contract (mono-mix -> pad/trim to round(rate) -> resample_poly ->
    pad/trim to target -> standardize -> float32), written out inline from
    the CONTRACT rather than calling any `target_windows` helper -- so the
    boundary test's content assertions cannot be satisfied by a bug that
    lives inside those helpers themselves.
    """
    mono = raw.mean(axis=1).astype(np.float64)
    n_in = int(round(rate_hz))
    mono = np.pad(mono, (0, n_in - mono.shape[0])) if mono.shape[0] < n_in else mono[:n_in]
    res = resample_poly(mono, target_hz, n_in)
    if res.shape[0] < target_hz:
        res = np.pad(res, (0, target_hz - res.shape[0]))
    else:
        res = res[:target_hz]
    res = (res - res.mean()) / np.clip(res.std(), 1e-8, None)
    return res.astype(np.float32)


def test_boundary_window_content_comes_from_the_full_window_file(tmp_path, monkeypatch) -> None:
    run = _build_boundary_run(tmp_path)
    cfg = _cfg(tmp_path)
    prepared = prepare_run(run, "audio", cfg, use_cache=False)

    # Fixture sanity: all 6 windows valid; attribution assigns window 2 to
    # file 0 (its 3-sample sliver is the earliest overlap) even though the
    # full slice lives in file 1 -- exactly the mismatch under test.
    assert prepared.grid.n_windows == 6
    assert prepared.valid_mask.all()
    assert prepared.segment_ids.tolist() == [0, 0, 0, 1, 1, 1]

    known_split = _split_by_known_segments(prepared, calib_segment_ids={0, 1})
    monkeypatch.setattr(
        target_windows, "split_by_segments",
        lambda segment_ids, valid_mask, calibration_frac, seed: known_split,
    )

    yielded = dict(iter_target_windows(run, cfg, return_indices=True))
    assert 2 in yielded
    actual = yielded[2]

    gen_files = sorted(run.files["RAWGeneratorMic__0"], key=lambda f: f.start_utc_hint)
    # Window 2 spans [2000 ms, 3000 ms): file B (t0 = 2003 ms) contributes its
    # own samples [0, 997); file A (t0 = 0, 2003 samples) contributes only its
    # samples [2000, 2003). Slice bounds hardcoded from the fixture layout --
    # first-principles ground truth, independent of any windowing helper.
    full_file_raw = read_gantner(gen_files[1].path).data[0:997, :]
    short_file_raw = read_gantner(gen_files[0].path).data[2000:2003, :]
    expected_full = _reference_window(full_file_raw, _BOUNDARY_RATE_HZ)
    wrong_padded = _reference_window(short_file_raw, _BOUNDARY_RATE_HZ)

    assert np.allclose(actual, expected_full, rtol=1e-5, atol=1e-6), (
        "window 2's content must come from the file holding its FULL slice (file B), "
        "matching the file _extract_stream_features actually featurized it from"
    )
    assert not np.allclose(actual, wrong_padded, rtol=1e-5, atol=1e-6), (
        "window 2's content must never be the 3-sample file's zero-padded sliver"
    )
    # Content signature: a genuine ~1000-real-sample window resamples to varied
    # content; a 3-sample sliver zero-padded to 1000 standardizes to a single
    # dominant repeated value on ~99% of positions.
    _vals, counts = np.unique(np.round(actual, 6), return_counts=True)
    assert counts.max() / actual.size < 0.5


def test_window_with_no_full_slice_in_any_file_is_skipped_with_debug_log(
    tmp_path, monkeypatch, caplog
) -> None:
    """Defensive skip path: a window whose every candidate file holds only a
    partial slice must be SKIPPED (with a debug log), never yielded as padded
    content. Such a window is pipeline-invalid (no file featurizes it, so its
    feature row is NaN and `valid_mask` excludes it) and thus can only reach
    the iterator through a hand-built split -- pinned here exactly so the
    iterator's own last line of defense never silently regresses into
    yielding padding.
    """
    work = tmp_path / "gap"
    work.mkdir(parents=True, exist_ok=True)
    # 40 windows at 1 kHz; file A [0, 20003), file B [20600, 40000): window 20
    # ([20000 ms, 21000 ms)) gets 3 samples from A and 400 from B -- no full
    # slice anywhere. 1 invalid window of 40 (2.5%) stays under prepare_run's
    # 5% hard-fail threshold.
    a = _mk_raw_file(work / "gen_a.dat", 20_003, t0_ns=0, seed=1)
    b = _mk_raw_file(work / "gen_b.dat", 19_400, t0_ns=20_600 * 1_000_000, seed=2)
    t = _mk_raw_file(work / "tur.dat", 40_000, t0_ns=0, seed=3)
    run = Run(
        name="gap-run",
        files={
            "RAWGeneratorMic__0": [
                BurstFile(path=a, stream="RAWGeneratorMic__0", start_utc_hint=_T0),
                BurstFile(
                    path=b, stream="RAWGeneratorMic__0",
                    start_utc_hint=_T0 + timedelta(microseconds=20_600 * 1000),
                ),
            ],
            "RAWTurbineMic__1": [
                BurstFile(path=t, stream="RAWTurbineMic__1", start_utc_hint=_T0)
            ],
        },
        day_root=work,
    )
    cfg = _cfg(tmp_path)
    prepared = prepare_run(run, "audio", cfg, use_cache=False)
    assert prepared.grid.n_windows == 40
    assert not prepared.valid_mask[20], "fixture sanity: the gap window is pipeline-invalid"

    forced_split = SegmentSplit(
        calibration_windows=np.array([19, 20, 21], dtype=np.int64),
        scoring_windows=np.array([22], dtype=np.int64),
    )
    monkeypatch.setattr(
        target_windows, "split_by_segments",
        lambda segment_ids, valid_mask, calibration_frac, seed: forced_split,
    )

    with caplog.at_level(logging.DEBUG, logger="rowii.adapt.target_windows"):
        yielded = dict(iter_target_windows(run, cfg, return_indices=True))

    assert sorted(yielded) == [19, 21], "windows with a full slice yield; the gap window not"
    skip_messages = [
        rec.getMessage() for rec in caplog.records if "skip" in rec.getMessage().lower()
    ]
    assert any("20" in msg for msg in skip_messages), (
        f"expected a debug log naming the skipped window 20, got: {skip_messages!r}"
    )


# ---------------------------------------------------------------------------
# 7. Negative max_windows is rejected (review Low): numpy's [:negative]
#    slicing would otherwise silently drop windows from the END instead of
#    truncating to the first N.
# ---------------------------------------------------------------------------


def test_negative_max_windows_raises_value_error(tmp_path) -> None:
    run = Run(name="unused", files={}, day_root=tmp_path)
    cfg = _cfg(tmp_path)

    with pytest.raises(ValueError, match="max_windows"):
        iter_target_windows(run, cfg, max_windows=-1)


# ---------------------------------------------------------------------------
# 8. iter_target_windows_multi: round-robin pooling across runs. The per-run iterators
#    are monkeypatched to known finite sequences (the SAME
#    `target_windows.iter_target_windows` monkeypatch seam the split tests
#    above use), so every assertion below pins the ROTATION itself --
#    interleaving order, exhausted-run dropout, the TOTAL budget -- against
#    hand-written ground truth, independent of any real gantner tree.
# ---------------------------------------------------------------------------


def _fake_run(tmp_path: Path, name: str) -> Run:
    """A files-free `Run` -- the monkeypatched per-run iterator below never
    touches files, only `run.name`."""
    return Run(name=name, files={}, day_root=tmp_path)


def _window_of(value: float) -> np.ndarray:
    """A distinguishable fake window: constant *value*, so `w[0]` identifies
    exactly which (run, position) a yielded array came from."""
    return np.full(3, value, dtype=np.float32)


_MULTI_SEQUENCES: dict[str, list[float]] = {
    # Unequal lengths on purpose (the test shape): run-b exhausts after
    # its first window, run-c after its second -- the pinned rotation is then
    # a, b, c, a, c, a.
    "run-a": [10.0, 11.0, 12.0],
    "run-b": [20.0],
    "run-c": [30.0, 31.0],
}


def _install_fake_per_run_iterators(monkeypatch) -> list[dict[str, object]]:
    """Monkeypatches `target_windows.iter_target_windows` with a fake serving
    `_MULTI_SEQUENCES[run.name]`, recording each call's kwargs at CALL time
    (a plain function returning an iterator, NOT a generator function -- so
    the call log also proves whether a run's iterator was constructed at
    all, e.g. for the zero-budget test)."""
    calls: list[dict[str, object]] = []

    def fake(run, cfg, *, target_hz=16_000, seed=7, max_windows=None, return_indices=False):
        calls.append(
            {"run": run.name, "target_hz": target_hz, "seed": seed, "max_windows": max_windows}
        )
        return iter([_window_of(v) for v in _MULTI_SEQUENCES[run.name]])

    monkeypatch.setattr(target_windows, "iter_target_windows", fake)
    return calls


def _three_fake_runs(tmp_path: Path) -> list[Run]:
    return [_fake_run(tmp_path, name) for name in ("run-a", "run-b", "run-c")]


def test_multi_round_robin_order_with_exhausted_run_dropout(tmp_path, monkeypatch) -> None:
    runs = _three_fake_runs(tmp_path)
    _install_fake_per_run_iterators(monkeypatch)

    out = list(
        target_windows.iter_target_windows_multi(
            runs, _cfg(tmp_path), max_windows=100, target_hz=16_000
        )
    )

    assert [float(w[0]) for w in out] == [10.0, 20.0, 30.0, 11.0, 31.0, 12.0], (
        "rotation must interleave a, b, c, a, c, a: one window per run per pass, an "
        "exhausted run dropping out of the rotation (A3.11)"
    )
    assert all(isinstance(w, np.ndarray) for w in out), (
        "base contract yields bare arrays, matching iter_target_windows' own"
    )


def test_multi_return_run_names_yields_run_window_pairs(tmp_path, monkeypatch) -> None:
    runs = _three_fake_runs(tmp_path)
    _install_fake_per_run_iterators(monkeypatch)

    pairs = list(
        target_windows.iter_target_windows_multi(
            runs, _cfg(tmp_path), max_windows=100, target_hz=16_000, return_run_names=True
        )
    )

    assert [name for name, _w in pairs] == [
        "run-a", "run-b", "run-c", "run-a", "run-c", "run-a"
    ]
    assert [float(w[0]) for _name, w in pairs] == [10.0, 20.0, 30.0, 11.0, 31.0, 12.0]


def test_multi_total_budget_is_total_not_per_run(tmp_path, monkeypatch) -> None:
    runs = _three_fake_runs(tmp_path)
    _install_fake_per_run_iterators(monkeypatch)

    out = list(
        target_windows.iter_target_windows_multi(
            runs, _cfg(tmp_path), max_windows=4, target_hz=16_000
        )
    )

    assert [float(w[0]) for w in out] == [10.0, 20.0, 30.0, 11.0], (
        "max_windows is the TOTAL across runs (A3.11): exactly 4 windows, cut "
        "mid-rotation, never 4 per run"
    )


def test_multi_forwards_target_hz_seed_and_budget_cap_to_each_run(tmp_path, monkeypatch) -> None:
    runs = _three_fake_runs(tmp_path)
    calls = _install_fake_per_run_iterators(monkeypatch)

    list(
        target_windows.iter_target_windows_multi(
            runs, _cfg(tmp_path), max_windows=5, target_hz=8_000, seed=42
        )
    )

    assert calls == [
        {"run": "run-a", "target_hz": 8_000, "seed": 42, "max_windows": 5},
        {"run": "run-b", "target_hz": 8_000, "seed": 42, "max_windows": 5},
        {"run": "run-c", "target_hz": 8_000, "seed": 42, "max_windows": 5},
    ], (
        "each run's iterator is constructed exactly once, in runs order, with the "
        "caller's target_hz, the shared split seed, and the TOTAL budget as its own "
        "per-run cap (no single run can contribute more than the total)"
    )


def test_multi_default_split_seed_is_7(tmp_path, monkeypatch) -> None:
    runs = _three_fake_runs(tmp_path)
    calls = _install_fake_per_run_iterators(monkeypatch)

    list(
        target_windows.iter_target_windows_multi(
            runs, _cfg(tmp_path), max_windows=2, target_hz=16_000
        )
    )

    assert calls and all(c["seed"] == 7 for c in calls), (
        "omitting seed must forward the canonical split seed 7 to every per-run "
        "iterator (the same default iter_target_windows itself pins)"
    )


def test_multi_zero_budget_yields_nothing_and_constructs_no_iterator(
    tmp_path, monkeypatch
) -> None:
    runs = _three_fake_runs(tmp_path)
    calls = _install_fake_per_run_iterators(monkeypatch)

    out = list(
        target_windows.iter_target_windows_multi(
            runs, _cfg(tmp_path), max_windows=0, target_hz=16_000
        )
    )

    assert out == []
    assert calls == [], "a zero budget must not even construct a per-run iterator"


def test_multi_negative_budget_raises_value_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_windows"):
        target_windows.iter_target_windows_multi(
            [], _cfg(tmp_path), max_windows=-1, target_hz=16_000
        )


def test_multi_zero_contribution_run_warns(monkeypatch, caplog) -> None:
    """A pool member contributing zero windows must produce a
    RUNTIME warning, not just sidecar-JSON visibility."""
    import logging
    from types import SimpleNamespace

    from rowii.adapt import target_windows as tw

    def fake_iter(run, cfg, *, target_hz, seed=7, max_windows=None):
        values = {"run-a": [1.0, 2.0], "run-empty": []}[run.name]
        return iter([np.full(4, v, dtype=np.float32) for v in values])

    monkeypatch.setattr(tw, "iter_target_windows", fake_iter)
    runs = [SimpleNamespace(name="run-a"), SimpleNamespace(name="run-empty")]
    with caplog.at_level(logging.WARNING):
        out = list(
            tw.iter_target_windows_multi(
                runs, cfg=None, max_windows=10, target_hz=4  # type: ignore[arg-type]
            )
        )
    assert len(out) == 2
    warned = [r.getMessage() for r in caplog.records if "ZERO" in r.getMessage()]
    assert any("run-empty" in m for m in warned)
