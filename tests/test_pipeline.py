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

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from rowii import pipeline
from rowii.config import Config
from rowii.io.dataset import BurstFile, Run
from rowii.pipeline import PreparedRun, prepare_run
from rowii.signals.features import AudioFeaturizer, VibFeaturizer
from rowii.signals.logmel import LogmelFeaturizer
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


# ---------------------------------------------------------------------------
# 4. build_run_grid: DAQ epoch-2000 clock quirk (Task 10, D2) -- the run's derived
# UTC offset (rowii.io.dataset.run_utc_offset_ns) must be baked into grid.t0_ns
# BEFORE common_grid, so PreparedRun.grid.t0_ns is true UTC, not the raw DAQ axis.
# ---------------------------------------------------------------------------


def _naive_unix_decode_ns(dt: datetime) -> int:
    """Nanoseconds since the Unix epoch that *dt* decodes to when its own digits
    are read naively AS IF it already were a Unix timestamp -- mirrors
    `tests.test_dataset._naive_unix_decode_ns` (duplicated, not imported)."""
    return int((dt - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds()) * 10**9


# CEST (UTC+2) worked example, identical constant to tests/test_dataset.py's own.
_CEST_OFFSET_NS = 946_677_600 * 10**9


def _quirky_two_stream_run(burst_dir: Path, *, n_seconds: int = 2) -> Run:
    """Two single-file streams sharing the SAME quirky raw axis: `header.t0_ns`
    decodes naively to `1996-06-27T06:41:03Z`; the true instant (filename hint) is
    `2026-06-27T04:41:03Z` -- the CEST worked example, same as
    `tests/test_dataset.py`'s own `run_utc_offset_ns` tests."""
    burst_dir.mkdir(parents=True, exist_ok=True)
    raw_t0_ns = _naive_unix_decode_ns(datetime(1996, 6, 27, 6, 41, 3, tzinfo=UTC))
    hint = datetime(2026, 6, 27, 4, 41, 3, tzinfo=UTC)
    files = {
        "RAWGeneratorMic__0": [
            _burst(
                burst_dir / "gen_mic.dat", "RAWGeneratorMic__0", n_seconds,
                t0_ns=raw_t0_ns, start_utc_hint=hint,
            )
        ],
        "RAWTurbineMic__1": [
            _burst(
                burst_dir / "tur_mic.dat", "RAWTurbineMic__1", n_seconds,
                t0_ns=raw_t0_ns, start_utc_hint=hint,
            )
        ],
    }
    return Run(name="quirky-two-stream-run", files=files, day_root=burst_dir)


def test_build_run_grid_shifts_t0_ns_onto_true_utc_for_quirky_clock(tmp_path) -> None:
    run = _quirky_two_stream_run(tmp_path / "burst")

    grid = pipeline.build_run_grid(
        run, ("RAWGeneratorMic__0", "RAWTurbineMic__1"), window_s=1.0
    )

    expected_true_utc_t0_ns = _naive_unix_decode_ns(datetime(2026, 6, 27, 4, 41, 3, tzinfo=UTC))
    assert grid.t0_ns == expected_true_utc_t0_ns
    assert grid.n_windows == 2  # unaffected by the axis shift -- 2 s of data, 1 s windows


def test_build_run_grid_leaves_already_correct_clock_grid_unshifted(tmp_path) -> None:
    # header.t0_ns IS the true UTC instant already (a few seconds of ordinary DAQ
    # jitter aside, well under the 1-hour plausibility gate) -- grid.t0_ns must come
    # out exactly as `common_grid` alone would have computed it, no offset invented.
    burst_dir = tmp_path / "burst"
    burst_dir.mkdir(parents=True, exist_ok=True)
    hint = datetime(2030, 3, 1, 12, 0, 0, tzinfo=UTC)
    true_t0_ns = _naive_unix_decode_ns(hint)
    files = {
        "RAWGeneratorMic__0": [
            _burst(
                burst_dir / "gen_mic.dat", "RAWGeneratorMic__0", 2,
                t0_ns=true_t0_ns, start_utc_hint=hint,
            )
        ],
        "RAWTurbineMic__1": [
            _burst(
                burst_dir / "tur_mic.dat", "RAWTurbineMic__1", 2,
                t0_ns=true_t0_ns, start_utc_hint=hint,
            )
        ],
    }
    run = Run(name="correct-clock-run", files=files, day_root=burst_dir)

    grid = pipeline.build_run_grid(
        run, ("RAWGeneratorMic__0", "RAWTurbineMic__1"), window_s=1.0
    )

    assert grid.t0_ns == true_t0_ns


# ---------------------------------------------------------------------------
# 5. prepare_run cache: DAQ epoch-2000 clock quirk (Task 10, D4) -- a cache written
# before the true-UTC fix carries a raw-axis grid_t0_ns; a fingerprint-matched hit
# must OVERRIDE it with the true-UTC value on load, without recomputing features.
# ---------------------------------------------------------------------------


def test_prepare_run_cache_hit_overrides_raw_axis_t0_ns_with_true_utc(
    tmp_path, monkeypatch
) -> None:
    run = _quirky_two_stream_run(tmp_path / "burst")
    cfg = _cfg(tmp_path / "results")

    call_count = {"n": 0}
    real_extract = pipeline._extract_stream_features

    def counting_extract(*args, **kwargs):
        call_count["n"] += 1
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_extract_stream_features", counting_extract)

    # First call: real compute + cache write. grid.t0_ns is ALREADY true-UTC (D2),
    # so simulate a cache written BEFORE the fix by round-tripping the cache file
    # with grid_t0_ns rewritten back to the raw axis -- exactly what an npz on disk
    # from before this task would carry.
    first = prepare_run(run, "audio", cfg, use_cache=True)
    assert call_count["n"] == 2

    cache_path = tmp_path / "results" / "cache" / "quirky-two-stream-run--audio.npz"
    assert cache_path.is_file()
    with np.load(cache_path, allow_pickle=False) as data:
        raw_cache = dict(data.items())
    raw_axis_t0_ns = first.grid.t0_ns - _CEST_OFFSET_NS
    raw_cache["grid_t0_ns"] = np.array([raw_axis_t0_ns], dtype=np.int64)
    np.savez(cache_path, **raw_cache)

    second = prepare_run(run, "audio", cfg, use_cache=True)

    assert call_count["n"] == 2, "cache hit must not trigger a recompute"
    assert second.grid.t0_ns == first.grid.t0_ns, "override must recover the true-UTC t0_ns"
    assert second.grid.window_ns == first.grid.window_ns
    assert second.grid.n_windows == first.grid.n_windows
    np.testing.assert_array_equal(second.features, first.features)
    np.testing.assert_array_equal(second.valid_mask, first.valid_mask)
    np.testing.assert_array_equal(second.segment_ids, first.segment_ids)


def test_prepare_run_cache_hit_is_a_no_op_when_t0_ns_already_true_utc(
    tmp_path, monkeypatch, caplog
) -> None:
    """A cache already written POST-fix (t0_ns already true-UTC) must be a silent
    override no-op -- `_load_cached_prepared_run` recomputes the fresh grid t0 every
    time (module docstring), but when it already matches the cached value there is
    nothing to log or change."""
    run = _quirky_two_stream_run(tmp_path / "burst")
    cfg = _cfg(tmp_path / "results")

    first = prepare_run(run, "audio", cfg, use_cache=True)
    cache_path = tmp_path / "results" / "cache" / "quirky-two-stream-run--audio.npz"
    assert cache_path.is_file()

    with caplog.at_level(logging.INFO):
        second = prepare_run(run, "audio", cfg, use_cache=True)

    assert second.grid.t0_ns == first.grid.t0_ns
    messages = [r.message for r in caplog.records]
    assert not any("overriding" in m.lower() for m in messages), messages


# ---------------------------------------------------------------------------
# 6. logmel variant (package-3 spec D3): stream mapping, featurizer dispatch, and a
# cache round-trip for the logmel-shaped (W, 49 * 64 = 3136) feature matrix.
# ---------------------------------------------------------------------------

_LOGMEL_RATE_HZ = 50_000.0
"""The plant's real mic rate -- the geometry at which logmel produces exactly
49 frames x 64 mels = 3136 features per 1-s window."""


def test_streams_for_variant_logmel_is_primary_mic_only() -> None:
    assert pipeline._streams_for_variant("logmel") == ("RAWGeneratorMic__0",)


def test_featurizer_for_stream_dispatches_logmel_variant(tmp_path) -> None:
    cfg = _cfg(tmp_path / "results")
    assert isinstance(
        pipeline._featurizer_for_stream("RAWGeneratorMic__0", "logmel", cfg),
        LogmelFeaturizer,
    )
    # The pre-existing dispatch stays untouched: audio -> AudioFeaturizer, vib
    # streams -> VibFeaturizer regardless of variant.
    assert isinstance(
        pipeline._featurizer_for_stream("RAWGeneratorMic__0", "audio", cfg),
        AudioFeaturizer,
    )
    assert isinstance(
        pipeline._featurizer_for_stream("RAWGeneratorVib__2", "vibration", cfg),
        VibFeaturizer,
    )


def _single_file_logmel_run(burst_dir: Path, *, n_seconds: int = 3) -> Run:
    """One 50 kHz single-channel file for the primary mic stream ONLY --
    `_streams_for_variant("logmel")` needs no other stream (spec D3: primary mic,
    size bound)."""
    burst_dir.mkdir(parents=True, exist_ok=True)
    path = burst_dir / "gen_mic.dat"
    n_samples = round(_LOGMEL_RATE_HZ * n_seconds)
    data = np.ones((n_samples, 1), dtype=np.float32)
    build_gantner_file(path, ["mic0"], data, t0_ns=0, rate_hz=_LOGMEL_RATE_HZ)
    files = {
        "RAWGeneratorMic__0": [
            BurstFile(
                path=path,
                stream="RAWGeneratorMic__0",
                start_utc_hint=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ],
    }
    return Run(name="unit-test-logmel-run", files=files, day_root=burst_dir)


# ---------------------------------------------------------------------------
# 7. audio-tfc / vibration-tfc variants (package-4 spec D4): stream mapping,
# featurizer dispatch (with the RELEVANT checkpoint path injected via cfg),
# feature-column assembly, and cache-fingerprint checkpoint scoping.
# ---------------------------------------------------------------------------


def test_streams_for_variant_tfc_variants_mirror_audio_and_vibration() -> None:
    assert pipeline._streams_for_variant("audio-tfc") == pipeline._streams_for_variant("audio")
    assert pipeline._streams_for_variant("vibration-tfc") == pipeline._streams_for_variant(
        "vibration"
    )


def test_featurizer_for_stream_dispatches_tfc_variants_with_cfg_checkpoint(tmp_path) -> None:
    from rowii.tfc.wrapper import TfcFeaturizer

    audio_ckpt = tmp_path / "tfc_audio.pt"
    vib_ckpt = tmp_path / "tfc_vib.pt"
    cfg = Config(
        data_root=tmp_path, results_root=tmp_path / "results",
        tfc_audio_checkpoint=audio_ckpt, tfc_vib_checkpoint=vib_ckpt,
    )

    audio_feat = pipeline._featurizer_for_stream("RAWGeneratorMic__0", "audio-tfc", cfg)
    assert isinstance(audio_feat, TfcFeaturizer)
    assert audio_feat._checkpoint == audio_ckpt

    vib_feat = pipeline._featurizer_for_stream("RAWGeneratorVib__2", "vibration-tfc", cfg)
    assert isinstance(vib_feat, TfcFeaturizer)
    assert vib_feat._checkpoint == vib_ckpt

    # Pre-existing dispatch stays untouched by the restructuring this needed
    # (_featurizer_for_stream's vibration-stream branch could no longer be a flat
    # "any non-audio stream -> VibFeaturizer" rule once vibration-tfc existed):
    # ordinary "vibration" still gets a plain VibFeaturizer, ordinary "audio" still
    # gets a plain AudioFeaturizer.
    assert isinstance(
        pipeline._featurizer_for_stream("RAWGeneratorVib__2", "vibration", cfg), VibFeaturizer
    )
    assert isinstance(
        pipeline._featurizer_for_stream("RAWGeneratorMic__0", "audio", cfg), AudioFeaturizer
    )


def test_assemble_variant_features_tfc_variants_hstack_like_audio_and_vibration() -> None:
    audio_result = pipeline._StreamFeatureResult(
        features=np.array([[1.0, 2.0]]), coverage=np.array([1.0])
    )
    vib_result = pipeline._StreamFeatureResult(
        features=np.array([[3.0]]), coverage=np.array([1.0])
    )
    stream_results = {
        "RAWGeneratorMic__0": audio_result,
        "RAWTurbineMic__1": audio_result,
        "RAWGeneratorVib__2": vib_result,
        "RAWTurbineVib__3": vib_result,
    }

    np.testing.assert_array_equal(
        pipeline.assemble_variant_features("audio-tfc", stream_results),
        pipeline.assemble_variant_features("audio", stream_results),
    )
    np.testing.assert_array_equal(
        pipeline.assemble_variant_features("vibration-tfc", stream_results),
        pipeline.assemble_variant_features("vibration", stream_results),
    )


def test_cache_fingerprint_tfc_checkpoint_change_is_scoped_to_its_own_variant(
    tmp_path,
) -> None:
    """Variant-SCOPED checkpoint sensitivity (unlike `beats_checkpoint`'s
    unconditional payload line): TF-C has TWO independent checkpoints (audio vs
    vibration branch, unlike BEATs' one), so a fingerprint must depend only on the
    ONE relevant to ITS OWN variant -- changing ROWII_TFC_AUDIO_CHECKPOINT must
    change audio-tfc's fingerprint but leave vibration-tfc's and fusion's alone
    (the reverse would force a needless recompute of every OTHER variant's cache
    whenever a user points at a new audio-branch checkpoint). Three cfgs, changing
    exactly ONE field relative to the baseline each, isolate the two checkpoint
    axes independently. This test pins RELATIVE behavior only; the absolute payload
    SHAPE (backward compatibility with pre-package-4 caches) is pinned by the
    golden test below."""
    run = _single_file_audio_run(tmp_path / "burst")
    audio_a, audio_b = tmp_path / "tfc_audio_a.pt", tmp_path / "tfc_audio_b.pt"
    vib_a, vib_b = tmp_path / "tfc_vib_a.pt", tmp_path / "tfc_vib_b.pt"
    baseline = Config(
        data_root=tmp_path, results_root=tmp_path,
        tfc_audio_checkpoint=audio_a, tfc_vib_checkpoint=vib_a,
    )
    audio_changed = Config(
        data_root=tmp_path, results_root=tmp_path,
        tfc_audio_checkpoint=audio_b, tfc_vib_checkpoint=vib_a,  # only audio differs
    )
    vib_changed = Config(
        data_root=tmp_path, results_root=tmp_path,
        tfc_audio_checkpoint=audio_a, tfc_vib_checkpoint=vib_b,  # only vib differs
    )

    def fp(variant: str, cfg: Config) -> str:
        return pipeline._cache_fingerprint(run, variant, cfg)

    assert fp("audio-tfc", baseline) != fp("audio-tfc", audio_changed), (
        "audio-tfc's fingerprint must change when ROWII_TFC_AUDIO_CHECKPOINT changes"
    )
    assert fp("audio-tfc", baseline) == fp("audio-tfc", vib_changed), (
        "audio-tfc's fingerprint must not depend on tfc_vib_checkpoint"
    )
    assert fp("vibration-tfc", baseline) == fp("vibration-tfc", audio_changed), (
        "vibration-tfc's fingerprint must not depend on tfc_audio_checkpoint"
    )
    assert fp("vibration-tfc", baseline) != fp("vibration-tfc", vib_changed), (
        "vibration-tfc's fingerprint must change when ROWII_TFC_VIB_CHECKPOINT changes"
    )
    assert fp("fusion", baseline) == fp("fusion", audio_changed) == fp("fusion", vib_changed), (
        "a non-tfc variant's fingerprint must not depend on either tfc checkpoint"
    )


# ---------------------------------------------------------------------------
# 8. audio-student variant (Step-2 package-5 spec D5): stream mapping (mirrors
# audio-beats), featurizer dispatch (with cfg.student_checkpoint injected),
# feature-column assembly, and cache-fingerprint checkpoint scoping isolated to
# ITS OWN variant only.
# ---------------------------------------------------------------------------


def test_streams_for_variant_audio_student_mirrors_audio_beats() -> None:
    assert pipeline._streams_for_variant("audio-student") == pipeline._streams_for_variant(
        "audio-beats"
    )


def test_is_student_variant() -> None:
    assert pipeline._is_student_variant("audio-student") is True
    assert pipeline._is_student_variant("audio-beats") is False
    assert pipeline._is_student_variant("audio-tfc") is False
    assert pipeline._is_student_variant("audio") is False


def test_featurizer_for_stream_dispatches_audio_student_with_cfg_checkpoint(tmp_path) -> None:
    from rowii.adapt.student import StudentFeaturizer

    student_ckpt = tmp_path / "student.pt"
    cfg = Config(
        data_root=tmp_path, results_root=tmp_path / "results",
        student_checkpoint=student_ckpt,
    )

    feat = pipeline._featurizer_for_stream("RAWGeneratorMic__0", "audio-student", cfg)
    assert isinstance(feat, StudentFeaturizer)
    assert feat._checkpoint == student_ckpt

    feat_turbine = pipeline._featurizer_for_stream("RAWTurbineMic__1", "audio-student", cfg)
    assert isinstance(feat_turbine, StudentFeaturizer)
    assert feat_turbine._checkpoint == student_ckpt

    # Pre-existing dispatch stays untouched: ordinary "audio" still gets a plain
    # AudioFeaturizer, "audio-beats"/"audio-tfc" are unaffected by this branch.
    assert isinstance(
        pipeline._featurizer_for_stream("RAWGeneratorMic__0", "audio", cfg), AudioFeaturizer
    )


def test_featurizer_for_stream_audio_student_never_raises_with_no_checkpoint(tmp_path) -> None:
    # Mirrors TfcFeaturizer's own deferred-load story: construction never raises
    # even when cfg.student_checkpoint is None -- only transform() would.
    from rowii.adapt.student import StudentFeaturizer

    cfg = Config(data_root=tmp_path, results_root=tmp_path / "results")
    feat = pipeline._featurizer_for_stream("RAWGeneratorMic__0", "audio-student", cfg)
    assert isinstance(feat, StudentFeaturizer)
    assert feat._checkpoint is None


def test_assemble_variant_features_audio_student_hstacks_like_audio_beats() -> None:
    audio_result = pipeline._StreamFeatureResult(
        features=np.array([[1.0, 2.0]]), coverage=np.array([1.0])
    )
    stream_results = {
        "RAWGeneratorMic__0": audio_result,
        "RAWTurbineMic__1": audio_result,
    }

    np.testing.assert_array_equal(
        pipeline.assemble_variant_features("audio-student", stream_results),
        pipeline.assemble_variant_features("audio-beats", stream_results),
    )


def test_cache_fingerprint_student_checkpoint_change_is_scoped_to_its_own_variant(
    tmp_path,
) -> None:
    """Variant-SCOPED checkpoint sensitivity (mirrors the tfc test above, package-5
    spec D5): a fingerprint must depend on `cfg.student_checkpoint` ONLY for
    `"audio-student"` -- changing it must leave `"audio-beats"` (the teacher
    variant `audio-student` distills FROM), `"audio-tfc"`, and every other
    variant's fingerprint untouched (the reverse would force a needless
    recompute of, say, the hours-expensive `audio-beats` cache whenever a user
    merely points `ROWII_STUDENT_CHECKPOINT` at a freshly distilled checkpoint).
    This test pins RELATIVE behavior only; the absolute payload SHAPE (backward
    compatibility with pre-package-5 caches) is pinned by the golden test below.
    """
    run = _single_file_audio_run(tmp_path / "burst")
    student_a = tmp_path / "student_a.pt"
    student_b = tmp_path / "student_b.pt"
    baseline = Config(
        data_root=tmp_path, results_root=tmp_path, student_checkpoint=student_a,
    )
    changed = Config(
        data_root=tmp_path, results_root=tmp_path, student_checkpoint=student_b,
    )

    def fp(variant: str, cfg: Config) -> str:
        return pipeline._cache_fingerprint(run, variant, cfg)

    assert fp("audio-student", baseline) != fp("audio-student", changed), (
        "audio-student's fingerprint must change when ROWII_STUDENT_CHECKPOINT changes"
    )
    assert fp("audio-beats", baseline) == fp("audio-beats", changed), (
        "audio-beats' fingerprint (the teacher cache audio-student distills FROM) must "
        "not depend on student_checkpoint"
    )
    assert fp("audio-tfc", baseline) == fp("audio-tfc", changed), (
        "audio-tfc's fingerprint must not depend on student_checkpoint"
    )
    assert fp("audio", baseline) == fp("audio", changed), (
        "a non-student variant's fingerprint must not depend on student_checkpoint"
    )
    assert fp("fusion", baseline) == fp("fusion", changed), (
        "fusion's fingerprint must not depend on student_checkpoint either"
    )


# ---------------------------------------------------------------------------
# 9. beats_int8_checkpoint (Step-2 package-5 spec D6): cache-fingerprint
# scoping to beats variants ONLY (audio-beats/fusion-beats), and ONLY when the
# int8 path is actually set -- unset must be payload byte-identical to the
# pre-existing (pre-Task-5) beats-variant fingerprint (no int8-checkpoint line
# at all), and `_featurizer_for_stream`'s beats dispatch must thread
# cfg.beats_int8_checkpoint through to BeatsFeaturizer's int8_model_path arg.
# ---------------------------------------------------------------------------


def test_cache_fingerprint_beats_int8_checkpoint_changes_beats_variants_only(
    tmp_path,
) -> None:
    """Mirrors `test_cache_fingerprint_student_checkpoint_change_is_scoped_to_
    its_own_variant`'s own relative-behavior shape: a fingerprint must depend
    on `cfg.beats_int8_checkpoint` for BOTH beats variants (`audio-beats`,
    `fusion-beats` -- `_is_beats_variant`), change again when the int8 path
    itself changes, and leave every OTHER variant's fingerprint (including
    the fp32-only `beats_checkpoint`-scoped ones) untouched. This test pins
    RELATIVE behavior only; the absolute payload SHAPE (backward
    compatibility with pre-Task-5 caches) is pinned by the two tests below."""
    run = _single_file_audio_run(tmp_path / "burst")
    unset = Config(data_root=tmp_path, results_root=tmp_path)
    set_a = Config(
        data_root=tmp_path, results_root=tmp_path,
        beats_int8_checkpoint=tmp_path / "beats_int8_a.pt",
    )
    set_b = Config(
        data_root=tmp_path, results_root=tmp_path,
        beats_int8_checkpoint=tmp_path / "beats_int8_b.pt",
    )

    def fp(variant: str, cfg: Config) -> str:
        return pipeline._cache_fingerprint(run, variant, cfg)

    assert fp("audio-beats", unset) != fp("audio-beats", set_a), (
        "audio-beats' fingerprint must change when ROWII_BEATS_INT8_CHECKPOINT is set"
    )
    assert fp("audio-beats", set_a) != fp("audio-beats", set_b), (
        "audio-beats' fingerprint must change again when the int8 path itself changes"
    )
    assert fp("fusion-beats", unset) != fp("fusion-beats", set_a), (
        "fusion-beats (the other beats variant) must be scoped the same way"
    )
    assert fp("audio", unset) == fp("audio", set_a), (
        "a non-beats variant's fingerprint must not depend on beats_int8_checkpoint"
    )
    assert fp("audio-tfc", unset) == fp("audio-tfc", set_a), (
        "audio-tfc's fingerprint must not depend on beats_int8_checkpoint"
    )
    assert fp("audio-student", unset) == fp("audio-student", set_a), (
        "audio-student's fingerprint must not depend on beats_int8_checkpoint"
    )


def test_cache_fingerprint_beats_int8_checkpoint_unset_is_byte_identical_to_pre_task5_format(
    tmp_path,
) -> None:
    """The literal 'unset -> payload byte-identical' contract (package-5 spec
    D6, Task 5): reconstructs the EXACT payload `_cache_fingerprint` produced
    for a beats variant before this task existed (three fixed lines + sorted
    file entries, no int8 line at all) independently, and asserts today's
    function -- called with beats_int8_checkpoint left at its default `None`
    -- still hashes to that same value. A regression here means an EXISTING
    `audio-beats`/`fusion-beats` cache (hours-expensive real BEATs
    extractions) would be silently invalidated on next use, even though
    nothing about how those variants are actually computed changed.
    """
    run = _single_file_audio_run(tmp_path / "burst")
    cfg = Config(data_root=tmp_path, results_root=tmp_path)  # beats_int8_checkpoint=None

    file_entries = sorted(
        f"{bf.path.name}:{bf.path.stat().st_size}"
        for files in run.files.values() for bf in files
    )
    expected_payload = "\n".join(
        ["variant=audio-beats", "window_s=1.0", "beats_checkpoint=", *file_entries]
    )
    expected = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()

    assert pipeline._cache_fingerprint(run, "audio-beats", cfg) == expected


def test_featurizer_for_stream_threads_beats_int8_checkpoint_for_beats_variants(
    tmp_path,
) -> None:
    """`_featurizer_for_stream`'s beats branch (package-5 spec D6) must pass
    `cfg.beats_int8_checkpoint` through as `BeatsFeaturizer`'s `int8_model_
    path` -- proven here via the missing-file guard (cheap: no real torch
    model construction needed) rather than a full real-checkpoint round trip
    (covered directly by `tests/test_beats.py`'s own `BeatsFeaturizer`
    tests): if the wiring were dropped (int8_model_path never passed
    through), `BeatsFeaturizer` would silently fall back to trying
    `cfg.beats_checkpoint` instead and raise `FileNotFoundError` naming THAT
    (wrong) path instead.
    """
    pytest.importorskip("torch")
    beats_ckpt = tmp_path / "beats.pt"  # deliberately never created -- must never
    # be read once int8_model_path is set (BeatsFeaturizer's own contract), so
    # its non-existence must never surface in the raised error below.
    int8_ckpt = tmp_path / "beats_int8.pt"  # deliberately missing
    cfg = Config(
        data_root=tmp_path, results_root=tmp_path / "results",
        beats_checkpoint=beats_ckpt, beats_int8_checkpoint=int8_ckpt,
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        pipeline._featurizer_for_stream("RAWGeneratorMic__0", "audio-beats", cfg)

    assert str(int8_ckpt) in str(exc_info.value)
    assert str(beats_ckpt) not in str(exc_info.value)


def _golden_fingerprint_run(burst_dir: Path) -> Run:
    """Fixed-name, fixed-size dummy files for the golden-fingerprint test below --
    `_cache_fingerprint` only ever `stat()`s a burst file (name + byte size enter the
    payload; content is never read or parsed), so plain zero-filled bytes suffice and
    keep the golden fully deterministic across machines and tmp dirs (only
    `path.name`, never the tmp-dir-dependent parent path, is hashed)."""
    burst_dir.mkdir(parents=True, exist_ok=True)
    gen, tur = burst_dir / "gen_mic.dat", burst_dir / "tur_mic.dat"
    gen.write_bytes(b"\x00" * 100)
    tur.write_bytes(b"\x00" * 200)
    hint = datetime(2026, 1, 1, tzinfo=UTC)
    return Run(
        name="golden-run",
        files={
            "RAWGeneratorMic__0": [
                BurstFile(path=gen, stream="RAWGeneratorMic__0", start_utc_hint=hint)
            ],
            "RAWTurbineMic__1": [
                BurstFile(path=tur, stream="RAWTurbineMic__1", start_utc_hint=hint)
            ],
        },
        day_root=burst_dir,
    )


def test_cache_fingerprint_golden_pins_payload_backward_compatibility(tmp_path) -> None:
    """GOLDEN regression pins (package-4 execution finding): the fingerprint payload
    SHAPE is a persistence format -- every pre-existing `results/cache/*.npz` stores
    a fingerprint computed from it, and hours-expensive BEATs caches silently
    invalidate (full re-extraction on next use) if the payload for their variant
    ever changes shape, even with identical semantics. Task 4's first cut emitted
    blank `tfc_*_checkpoint=` lines for EVERY variant and did exactly that (real
    finding: 250526-tu/audio-beats stored d455ca9b... vs recomputed 9bacaa14...).

    The hex literals below are sha256 digests computed ONCE from the intended
    payloads, independently of `_cache_fingerprint` itself:

    - non-tfc/non-student variants: the PRE-package-4 payload format,
      byte-identical (`variant=` / `window_s=` / `beats_checkpoint=` / sorted
      `name:size` file entries -- NO extra checkpoint lines at all);
    - tfc variants: that same format + exactly ONE extra line (its own
      variant's checkpoint) inserted after `beats_checkpoint=`;
    - `audio-student` (package-5 spec D5, Task 4, added in THIS commit): same
      format + exactly ONE extra `student_checkpoint=` line, same insertion
      point. No cache-migration is needed for this addition -- `audio-student`
      is a BRAND NEW variant with no pre-existing `results/cache/*.npz`
      entries to orphan, and every OTHER variant's own golden pin above stays
      byte-for-byte UNCHANGED (proven by their tests staying green, untouched,
      in this same commit) -- the banked lesson applied consciously, not
      skipped.
    - `audio-beats` + `beats_int8_checkpoint` SET (package-5 spec D6, Task 5,
      added in THIS commit): same format + exactly ONE extra `beats_int8_
      checkpoint=` line, inserted right after `beats_checkpoint=` -- scoped to
      beats variants ONLY (`_is_beats_variant`: `audio-beats`/`fusion-beats`)
      and ONLY when the int8 path is actually set at all; the UNSET case is
      covered by a separate dedicated test (`test_cache_fingerprint_beats_
      int8_checkpoint_unset_is_byte_identical_to_pre_task5_format`), which
      independently reconstructs the SAME pre-Task-5 payload rather than
      duplicating another opaque hex literal here. No cache-migration is
      needed for the unset case: every existing `results/cache/*--audio-
      beats.npz`/`*--fusion-beats.npz` entry was written under a
      `beats_int8_checkpoint=None` config, so its fingerprint is UNCHANGED by
      this addition (proven by that dedicated test, and by every OTHER golden
      pin above staying untouched). Setting `ROWII_BEATS_INT8_CHECKPOINT` for
      the FIRST time is a deliberate, one-time fork: the very next
      `prepare_run` call for that (run, beats-variant) pair computes a
      DIFFERENT fingerprint, misses the existing fp32-computed cache, and
      recomputes into a fresh entry -- expected and intended (the int8
      embeddings genuinely differ numerically from fp32's), not a bug.

    Any future change to `_cache_fingerprint`'s payload must consciously update
    these constants AND deliberately migrate/invalidate the on-disk caches -- a
    failing golden here means "you are about to orphan every existing cache entry",
    not "just update the constant".
    """
    run = _golden_fingerprint_run(tmp_path / "burst")
    cfg_plain = Config(data_root=tmp_path, results_root=tmp_path)  # all checkpoints None
    cfg_tfc = Config(
        data_root=tmp_path, results_root=tmp_path,
        # FIXED absolute paths (never tmp_path-derived): the checkpoint path string
        # itself enters the payload, so a tmp-dir-dependent path would break the pin.
        tfc_audio_checkpoint=Path("/fixed/tfc_audio.pt"),
        tfc_vib_checkpoint=Path("/fixed/tfc_vib.pt"),
    )
    cfg_student = Config(
        data_root=tmp_path, results_root=tmp_path,
        student_checkpoint=Path("/fixed/student.pt"),
    )
    cfg_beats_int8 = Config(
        data_root=tmp_path, results_root=tmp_path,
        beats_int8_checkpoint=Path("/fixed/beats_int8.pt"),
    )

    assert pipeline._cache_fingerprint(run, "audio", cfg_plain) == (
        "1f9c34523161e0dcd58de6ab3f55da9f321d35304588d786aab8ba06a49513c0"
    ), "non-tfc payload shape changed -- this would invalidate every pre-existing cache"
    assert pipeline._cache_fingerprint(run, "audio-tfc", cfg_tfc) == (
        "5dedd7f15ee2e08d66b2f44ddb90ca3e5be3ebb8498ffcfeb71f8929128be656"
    ), "audio-tfc payload shape changed (expected: pre-T4 format + ONE tfc_audio line)"
    assert pipeline._cache_fingerprint(run, "vibration-tfc", cfg_tfc) == (
        "ccf33c4e323dc21795b30833a623408a93222661fdc148290faf257d6c014b49"
    ), "vibration-tfc payload shape changed (expected: pre-T4 format + ONE tfc_vib line)"
    assert pipeline._cache_fingerprint(run, "audio-student", cfg_student) == (
        "b7d7e12e85c315ddb1bd486fe25d217e81cc08a425105eca58306d62aaedc578"
    ), "audio-student payload shape changed (expected: pre-T4 format + ONE student_checkpoint line)"
    assert pipeline._cache_fingerprint(run, "audio-beats", cfg_beats_int8) == (
        "71010e76d13ab3b70bf7d44df6bb41113f7c2f1bec9bbb5efa8e60c71a585b92"
    ), (
        "audio-beats+int8 payload shape changed (expected: pre-T5 format + ONE "
        "beats_int8_checkpoint line)"
    )


def test_prepare_run_logmel_cache_round_trip(tmp_path, monkeypatch) -> None:
    """End-to-end `prepare_run(run, "logmel", ...)` over real (synthetic) 50 kHz
    burst data: (W, 3136) features, stream-prefixed logmel feature names, and a
    cache round-trip (second call is a HIT -- no re-extraction; same pattern as
    the audio-variant cache tests above)."""
    run = _single_file_logmel_run(tmp_path / "burst", n_seconds=3)
    cfg = _cfg(tmp_path / "results")

    call_count = {"n": 0}
    real_extract = pipeline._extract_stream_features

    def counting_extract(*args, **kwargs):
        call_count["n"] += 1
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_extract_stream_features", counting_extract)

    first = prepare_run(run, "logmel", cfg, use_cache=True)
    assert call_count["n"] == 1, "logmel uses exactly ONE stream (primary mic)"

    assert first.features.shape == (3, 49 * 64)
    assert first.valid_mask.all()
    assert len(first.feature_names) == 49 * 64
    assert first.feature_names[0] == "RAWGeneratorMic__0::logmel_f0_m0"
    assert first.feature_names[64] == "RAWGeneratorMic__0::logmel_f1_m0"
    assert (first.segment_ids == 0).all()

    cache_path = tmp_path / "results" / "cache" / "unit-test-logmel-run--logmel.npz"
    assert cache_path.is_file()

    second = prepare_run(run, "logmel", cfg, use_cache=True)
    assert call_count["n"] == 1, "second call must be a cache HIT -- no new extraction"

    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(first.valid_mask, second.valid_mask)
    np.testing.assert_array_equal(first.segment_ids, second.segment_ids)
    assert first.feature_names == second.feature_names
    assert first.grid == second.grid


# --- stream_columns (per-stream column-block selection) ----------------------


class TestStreamColumns:
    def test_selects_contiguous_block_by_name_prefix(self):
        names = ["MicA::e0", "MicA::e1", "MicB::e0", "MicB::e1"]
        np.testing.assert_array_equal(
            pipeline.stream_columns(names, "MicA"), np.array([0, 1])
        )
        np.testing.assert_array_equal(
            pipeline.stream_columns(names, "MicB"), np.array([2, 3])
        )

    def test_selection_is_by_prefix_not_position(self):
        names = ["MicB::e0", "MicA::e0", "MicB::e1", "MicA::e1"]
        np.testing.assert_array_equal(
            pipeline.stream_columns(names, "MicA"), np.array([1, 3])
        )

    def test_prefix_match_is_exact_up_to_separator(self):
        # "Mic" must not swallow "MicA::..." -- the separator is part of the prefix.
        names = ["MicA::e0", "MicA::e1"]
        with pytest.raises(ValueError, match="Mic"):
            pipeline.stream_columns(names, "Mic")

    def test_unknown_stream_raises_value_error(self):
        with pytest.raises(ValueError, match="MicC"):
            pipeline.stream_columns(["MicA::e0"], "MicC")
