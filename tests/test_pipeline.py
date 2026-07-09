"""Unit tests for `rowii.pipeline` (Step-2 Task S1): `prepare_run`'s on-disk feature
cache (round-trip + fingerprint invalidation) and its two fields Step-1's own CLI never
consumed, `feature_names` and `segment_ids`.

`scripts/run_step1.py`'s own CLI-level behavior for this same underlying logic is
covered by `tests/test_cli_smoke.py`'s miniature end-to-end test (which also pins exact
pre-refactor summary-row values); this file exercises `rowii.pipeline.prepare_run`
directly, bypassing discovery/the CLI entirely, since a `Run` object is cheap to build
by hand for a focused unit test.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from rowii import pipeline
from rowii.config import Config
from rowii.io.dataset import BurstFile, Run
from rowii.pipeline import PreparedRun, prepare_run
from rowii.signals.features import AudioFeaturizer
from rowii.signals.windows import WindowGrid
from tests.fixtures.gantner_builder import build_gantner_file

_RATE_HZ = 100.0


def _cfg(results_root: Path) -> Config:
    return Config(data_root=results_root, results_root=results_root)


def _burst(
    path: Path, stream: str, n_seconds: float, *, t0_ns: int, start_utc_hint: datetime
) -> BurstFile:
    n_samples = round(_RATE_HZ * n_seconds)
    data = np.ones((n_samples, 4), dtype=np.float32)
    build_gantner_file(
        path, ["ch0", "ch1", "ch2", "ch3"], data, t0_ns=t0_ns, rate_hz=_RATE_HZ
    )
    return BurstFile(path=path, stream=stream, start_utc_hint=start_utc_hint)


def _single_file_audio_run(burst_dir: Path, *, n_seconds: int = 5) -> Run:
    """One file per mic stream, contiguous full coverage, no invalid windows --
    `_AUDIO_STREAMS[0]` ("RAWGeneratorMic__0") is the only stream `segment_ids` is
    read from, but BOTH mic streams need a file present for the "audio" variant to
    resolve at all (`_streams_for_variant("audio")` requires both)."""
    burst_dir.mkdir(parents=True, exist_ok=True)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    files = {
        "RAWGeneratorMic__0": [
            _burst(
                burst_dir / "gen_mic.dat", "RAWGeneratorMic__0", n_seconds,
                t0_ns=0, start_utc_hint=t0,
            )
        ],
        "RAWTurbineMic__1": [
            _burst(
                burst_dir / "tur_mic.dat", "RAWTurbineMic__1", n_seconds,
                t0_ns=0, start_utc_hint=t0,
            )
        ],
    }
    return Run(name="unit-test-run", files=files, day_root=burst_dir)


# ---------------------------------------------------------------------------
# 1. Basic correctness: feature_names / segment_ids on a single-file, full-coverage run
# ---------------------------------------------------------------------------


def test_prepare_run_audio_variant_populates_feature_names_and_segment_ids(tmp_path) -> None:
    run = _single_file_audio_run(tmp_path / "burst", n_seconds=5)
    cfg = _cfg(tmp_path / "results")

    prepared = prepare_run(run, "audio", cfg, use_cache=False)

    assert isinstance(prepared, PreparedRun)
    assert prepared.grid.n_windows == 5
    assert prepared.features.shape == (5, len(prepared.feature_names))
    assert prepared.valid_mask.shape == (5,)
    assert prepared.valid_mask.all(), "single contiguous file, full coverage everywhere"

    # Both mic streams' feature names appear, each prefixed by its own stream id (see
    # _assemble_feature_names -- disambiguates the two streams' otherwise-identical
    # local "chN_*" names).
    assert any(n.startswith("RAWGeneratorMic__0::") for n in prepared.feature_names)
    assert any(n.startswith("RAWTurbineMic__1::") for n in prepared.feature_names)

    # A single burst file per stream -> every window's primary-stream (first mic
    # stream, per _streams_for_variant("audio")[0]) segment id is 0.
    assert prepared.segment_ids.shape == (5,)
    assert prepared.segment_ids.dtype == np.int64
    assert (prepared.segment_ids == 0).all()


def test_extract_stream_features_segment_ids_track_files_in_time_order_with_gap(
    tmp_path,
) -> None:
    """`_extract_stream_features`'s segment_ids: 0-based, time-sorted index of the file
    that contributed a window's samples; -1 where no file covers the window at all
    (PreparedRun.segment_ids' documented convention, tested directly at the function
    that derives it -- going through the full `prepare_run`/`compute_validity_mask`
    would hard-fail here, since a deliberate 2-window gap out of 8 windows is 25%
    invalid, far above the 5% threshold; this is a single-stream, function-level test,
    not a validity-mask scenario).
    """
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    file_a = _burst(
        tmp_path / "a.dat", "RAWGeneratorMic__0", 3, t0_ns=0, start_utc_hint=t0
    )
    # File B starts 5s after file A's t0 (file A itself spans only 3s) -- a 2s/2-window
    # gap with NO samples from either file in between.
    file_b = _burst(
        tmp_path / "b.dat",
        "RAWGeneratorMic__0",
        3,
        t0_ns=5_000_000_000,
        start_utc_hint=datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC),
    )
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=8)

    result = pipeline._extract_stream_features([file_a, file_b], grid, AudioFeaturizer())

    np.testing.assert_array_equal(
        result.segment_ids, np.array([0, 0, 0, -1, -1, 1, 1, 1], dtype=np.int64)
    )


# ---------------------------------------------------------------------------
# 2. Cache round-trip: second call with an unchanged run/variant/cfg is a cache HIT
# ---------------------------------------------------------------------------


def test_prepare_run_cache_round_trip_avoids_recompute(tmp_path, monkeypatch) -> None:
    run = _single_file_audio_run(tmp_path / "burst")
    cfg = _cfg(tmp_path / "results")

    call_count = {"n": 0}
    real_extract = pipeline._extract_stream_features

    def counting_extract(*args, **kwargs):
        call_count["n"] += 1
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_extract_stream_features", counting_extract)

    first = prepare_run(run, "audio", cfg, use_cache=True)
    assert call_count["n"] == 2, "one _extract_stream_features call per audio stream"

    cache_path = tmp_path / "results" / "cache" / "unit-test-run--audio.npz"
    assert cache_path.is_file()

    second = prepare_run(run, "audio", cfg, use_cache=True)
    assert call_count["n"] == 2, "second call must be a cache HIT -- no new extraction"

    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.valid_mask, second.valid_mask)
    np.testing.assert_array_equal(first.segment_ids, second.segment_ids)
    assert first.feature_names == second.feature_names
    assert first.grid == second.grid


def test_prepare_run_no_cache_never_reads_or_writes_cache_file(tmp_path, monkeypatch) -> None:
    run = _single_file_audio_run(tmp_path / "burst")
    cfg = _cfg(tmp_path / "results")

    call_count = {"n": 0}
    real_extract = pipeline._extract_stream_features

    def counting_extract(*args, **kwargs):
        call_count["n"] += 1
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_extract_stream_features", counting_extract)

    prepare_run(run, "audio", cfg, use_cache=False)
    prepare_run(run, "audio", cfg, use_cache=False)

    assert call_count["n"] == 4, "use_cache=False must recompute every single call"
    cache_path = tmp_path / "results" / "cache" / "unit-test-run--audio.npz"
    assert not cache_path.exists()


def test_prepare_run_cache_invalidates_when_source_file_size_changes(
    tmp_path, monkeypatch
) -> None:
    burst_dir = tmp_path / "burst"
    run = _single_file_audio_run(burst_dir, n_seconds=5)
    cfg = _cfg(tmp_path / "results")

    call_count = {"n": 0}
    real_extract = pipeline._extract_stream_features

    def counting_extract(*args, **kwargs):
        call_count["n"] += 1
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_extract_stream_features", counting_extract)

    prepare_run(run, "audio", cfg, use_cache=True)
    assert call_count["n"] == 2

    prepare_run(run, "audio", cfg, use_cache=True)
    assert call_count["n"] == 2, "unchanged inputs -- second call must be a cache HIT"

    # Rebuild ONE stream's file with more data (6 s instead of 5 s) at the SAME path --
    # a genuine byte-size change, exactly what the fingerprint must catch.
    gen_mic_path = run.files["RAWGeneratorMic__0"][0].path
    longer_data = np.ones((round(_RATE_HZ * 6), 4), dtype=np.float32)
    build_gantner_file(
        gen_mic_path, ["ch0", "ch1", "ch2", "ch3"], longer_data, t0_ns=0, rate_hz=_RATE_HZ
    )
    assert gen_mic_path.stat().st_size != (burst_dir / "tur_mic.dat").stat().st_size

    prepare_run(run, "audio", cfg, use_cache=True)
    assert call_count["n"] == 4, "a changed source file size must force a recompute"


def test_prepare_run_cache_survives_a_corrupt_cache_file(tmp_path, monkeypatch) -> None:
    """A cache file that fails to parse (e.g. truncated by an interrupted previous
    write) is a cache MISS, not a crash -- `prepare_run` recomputes and overwrites it."""
    run = _single_file_audio_run(tmp_path / "burst")
    cfg = _cfg(tmp_path / "results")

    prepare_run(run, "audio", cfg, use_cache=True)
    cache_path = tmp_path / "results" / "cache" / "unit-test-run--audio.npz"
    assert cache_path.is_file()
    cache_path.write_bytes(b"not a real npz file")

    call_count = {"n": 0}
    real_extract = pipeline._extract_stream_features

    def counting_extract(*args, **kwargs):
        call_count["n"] += 1
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_extract_stream_features", counting_extract)

    prepared = prepare_run(run, "audio", cfg, use_cache=True)

    assert call_count["n"] == 2, "corrupt cache must trigger a real recompute"
    assert isinstance(prepared, PreparedRun)
    # The cache file must have been rewritten with valid data (readable again).
    reloaded = prepare_run(run, "audio", cfg, use_cache=True)
    assert call_count["n"] == 2, "the rewritten cache must be a HIT on the next call"
    np.testing.assert_array_equal(prepared.features, reloaded.features)


# ---------------------------------------------------------------------------
# 3. Unknown variant raises ValueError (unchanged from the pre-refactor behavior)
# ---------------------------------------------------------------------------


def test_prepare_run_unknown_variant_raises_value_error(tmp_path) -> None:
    run = _single_file_audio_run(tmp_path / "burst")
    cfg = _cfg(tmp_path / "results")

    with pytest.raises(ValueError, match="unknown variant"):
        prepare_run(run, "not-a-real-variant", cfg, use_cache=False)
