import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_live_replay as blr


def test_live_sessions_registry() -> None:
    assert set(blr.LIVE_SESSIONS) == {
        "290626-tu", "080726-pu_strikes", "010726-tu1-morning", "270626-pu_ph_pu_ph_pu_ph-1",
    }
    assert blr.LIVE_SESSIONS["290626-tu"]["out"] == "live.html"
    outs = [s["out"] for s in blr.LIVE_SESSIONS.values()]
    assert len(outs) == len(set(outs))


def test_session_summary_shape() -> None:
    s = blr.session_summary("080726-pu_strikes", duration_s=3600.0, n_episodes=7)
    assert s["id"] == "080726-pu_strikes"
    assert s["display_name"] == "Hammer-strike day"
    assert s["events"] is True
    assert s["duration_s"] == 3600.0
    assert s["n_episodes"] == 7
    date_label = s["date_label"]
    assert isinstance(date_label, str)
    assert date_label.startswith("WED · 08 JUL 2026")


def test_humanized_names_helper() -> None:
    names = blr.humanize_names(["RAWGeneratorMic__0::ch0_log_rms", "bogus"])
    assert names[0] == "Generator mic 1 · loudness (log RMS)"
    assert names[1] == "bogus"


def test_pages_nav_fractions_and_href() -> None:
    summaries = {
        "290626-tu": {
            "id": "290626-tu", "display_name": "Turbine day",
            "blurb": "quiet reference day", "events": False,
            "date_label": "MON · 29 JUN 2026", "duration_s": 100.0, "n_episodes": 2,
        },
    }
    timelines = {
        "290626-tu": {
            "segments": [
                {"start_s": 0.0, "end_s": 40.0, "state_name": "standstill"},
                {"start_s": 40.0, "end_s": 100.0, "state_name": "turbine"},
            ],
            "tick_s": [50.0],
        },
    }
    nav = blr.pages_nav(summaries, timelines)
    assert nav[0]["href"] == "live.html"
    assert nav[0]["ribbon"] == [
        {"start_frac": 0.0, "end_frac": 0.4, "state": "standstill"},
        {"start_frac": 0.4, "end_frac": 1.0, "state": "turbine"},
    ]
    assert nav[0]["ticks"] == [0.5]


def test_pages_nav_skips_sessions_not_yet_built() -> None:
    """A `--run` subset build only has SOME registry sessions in `summaries` --
    `pages_nav` must list exactly those, not dangle a link to an unbuilt page."""
    summaries = {
        "290626-tu": {
            "id": "290626-tu", "display_name": "Turbine day",
            "blurb": "quiet reference day", "events": False,
            "date_label": "MON · 29 JUN 2026", "duration_s": 10.0, "n_episodes": 0,
        },
    }
    timelines: dict[str, dict[str, Any]] = {"290626-tu": {"segments": [], "tick_s": []}}
    nav = blr.pages_nav(summaries, timelines)
    assert [n["id"] for n in nav] == ["290626-tu"]


def test_make_run_context_once_calibrated_session() -> None:
    ctx = blr.make_run_context("290626-tu")
    assert ctx.run == "290626-tu"
    assert ctx.representation == "fusion"
    assert ctx.regime == "frozen"
    assert ctx.monitor_dir == (
        blr.RESULTS_ROOT / "step2" / "once-calibrated" / "fusion" / "monitor"
        / "290626-tu" / "frozen"
    )
    assert ctx.fusion_json == (
        blr.RESULTS_ROOT / "step2" / "once-calibrated" / "fusion" / "fusion.json"
    )
    assert ctx.audio_meta_json == blr.AUDIO_DIR / "290626-tu_audio_meta.json"


def test_make_run_context_monitor_ext_session() -> None:
    """`270626-pu_ph_pu_ph_pu_ph-1` is a `candidate_kit._MONITOR_EXT_SESSIONS`
    coverage-extension session: its alarms live under `results/monitor-ext/<run>/
    <representation>/`, NOT the pinned once-calibrated tree (no `<regime>/`
    segment there) -- `make_run_context` must resolve the SAME path
    `candidate_kit._alarms_path_for` does, not the once-calibrated formula."""
    run = "270626-pu_ph_pu_ph_pu_ph-1"
    ctx = blr.make_run_context(run)
    assert ctx.regime == "recalibrate"
    assert ctx.monitor_dir == blr.RESULTS_ROOT / "monitor-ext" / run / "fusion"


def test_main_unknown_run_lists_valid_ids() -> None:
    with pytest.raises(SystemExit) as exc_info:
        blr.main(["--run", "not-a-real-session"])
    message = str(exc_info.value)
    assert "not-a-real-session" in message
    for valid_id in blr.LIVE_SESSIONS:
        assert valid_id in message


def test_preflight_reports_missing_inputs_for_unbuilt_run() -> None:
    run = "zzz-never-built-session"
    ctx = blr.RunContext(
        run=run,
        representation="fusion",
        regime="frozen",
        sentinel_representation="audio-beats",
        unit_name=blr.UNIT_NAME,
        monitor_dir=(
            blr.RESULTS_ROOT / "step2" / "once-calibrated" / "fusion" / "monitor" / run / "frozen"
        ),
        sentinel_json=(
            blr.RESULTS_ROOT / "step2" / "once-calibrated" / "audio-beats" / "audio-beats.json"
        ),
        fusion_json=blr.RESULTS_ROOT / "step2" / "once-calibrated" / "fusion" / "fusion.json",
        candidates_meta=blr.RESULTS_ROOT / "candidate-kit" / "candidates_meta.json",
        audio_dir=blr.AUDIO_DIR,
        audio_meta_json=blr.AUDIO_DIR / f"{run}_audio_meta.json",
    )
    problems = blr.preflight(ctx)
    assert problems
    assert all(run in p for p in problems)


def test_preflight_reports_missing_sentinel_and_fusion_json_files() -> None:
    """Reviewer fix (round 1, item 2): an absent `sentinel_json`/`fusion_json`
    must abort in preflight, not crash mid-build with FileNotFoundError."""
    run = "zzz-never-built-session"
    ctx = blr.RunContext(
        run=run,
        representation="fusion",
        regime="frozen",
        sentinel_representation="audio-beats",
        unit_name=blr.UNIT_NAME,
        monitor_dir=(
            blr.RESULTS_ROOT / "step2" / "once-calibrated" / "fusion" / "monitor" / run / "frozen"
        ),
        sentinel_json=blr.RESULTS_ROOT / "does-not-exist" / "audio-beats.json",
        fusion_json=blr.RESULTS_ROOT / "does-not-exist" / "fusion.json",
        candidates_meta=blr.RESULTS_ROOT / "candidate-kit" / "candidates_meta.json",
        audio_dir=blr.AUDIO_DIR,
        audio_meta_json=blr.AUDIO_DIR / f"{run}_audio_meta.json",
    )
    problems = blr.preflight(ctx)
    assert any("audio-beats.json" in p and "run_once_calibrated.py" in p for p in problems)
    assert any("fusion.json" in p and "run_once_calibrated.py" in p for p in problems)


def test_sentinel_payload_full_when_trigger_and_regime_rows_both_present() -> None:
    trig = {
        "era": "B", "s1_rate": 0.0767, "s1_threshold": 0.0805,
        "s1_fired": False, "s2_fired": False, "s2_attribution": "machine",
        "decision": "frozen",
    }
    regime = {
        "once_triggered_far": 0.075, "always_frozen_far": 0.075,
        "always_recalibrate_far": 0.0439, "far_basis": "common-window",
    }
    payload = blr.sentinel_payload(trig, regime)
    assert payload == {
        "available": "full",
        "era": "B",
        "s1_rate": 0.0767,
        "s1_threshold": 0.0805,
        "s1_fired": False,
        "s2_fired": False,
        "s2_attribution": "machine",
        "decision": "frozen",
        "nominal_alpha": 0.05,
        "realized_far": 0.075,
        "always_frozen_far": 0.075,
        "always_recalibrate_far": 0.0439,
        "far_basis": "common-window",
    }


def test_sentinel_payload_trigger_only_when_regime_row_missing() -> None:
    """270626-pu_ph_pu_ph_pu_ph-1's real shape: a trigger_log row exists (this
    session WAS sentinel-scored, with a REAL recorded decision) but no
    regimes row (no regime REPLAY/FAR bookkeeping exists for this session --
    it was never monitored under the pinned once-calibrated tree). `decision`
    carries the trigger row's own real value VERBATIM -- it is real recorded
    data, not something a missing regime replay should discard (contract
    correction, review round 2: an earlier version of this shape wrongly
    forced `decision: None` here). Only the FAR-derived fields
    (`nominal_alpha`, `realized_far`, `always_frozen_far`,
    `always_recalibrate_far`, `far_basis`) are genuinely unavailable without a
    regime row, so those -- and only those -- stay absent."""
    trig = {
        "era": "A", "s1_rate": 0.3273, "s1_threshold": 0.0805,
        "s1_fired": True, "s2_fired": False, "s2_attribution": "machine",
        "decision": "recalibrate",
    }
    payload = blr.sentinel_payload(trig, None)
    assert payload == {
        "available": "trigger_only",
        "era": "A",
        "s1_rate": 0.3273,
        "s1_threshold": 0.0805,
        "s1_fired": True,
        "s2_fired": False,
        "s2_attribution": "machine",
        "decision": "recalibrate",
        "note": (
            "sentinel decision recorded; no regime replay (FAR evaluation) exists for "
            "this session"
        ),
    }
    for far_key in (
        "nominal_alpha", "realized_far", "always_frozen_far",
        "always_recalibrate_far", "far_basis",
    ):
        assert far_key not in payload


def test_sentinel_payload_none_when_neither_row_present() -> None:
    """010726-tu1-morning's real shape: no trigger_log row and no regimes
    row -- never scored by the once-calibrated sentinel driver at all."""
    payload = blr.sentinel_payload(None, None)
    assert payload == {
        "available": "none",
        "note": (
            "session not scored by the once-calibrated sentinel driver; alarms come "
            "from the coverage-extension monitor run (recalibrate thresholds)"
        ),
    }


def test_truncate_to_common_length_leaves_equal_length_arrays_untouched(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The common case (290626-tu, 080726-pu_strikes, 010726-tu1-morning): every
    array a caller passes already shares one length -- truncation must be a
    no-op, not just a coincidentally-correct slice, and must stay SILENT (no
    `logger.warning`) since nothing was actually discarded."""
    a = np.arange(5, dtype=np.float64)
    b = np.ones(5, dtype=bool)
    with caplog.at_level(logging.WARNING):
        out_a, out_b = blr._truncate_to_common_length(a, b, label="equal_case")
    np.testing.assert_array_equal(out_a, a)
    np.testing.assert_array_equal(out_b, b)
    assert caplog.records == []


def test_truncate_to_common_length_real_270626_case(caplog: pytest.LogCaptureFixture) -> None:
    """270626-pu_ph_pu_ph_pu_ph-1's real cache shapes: audio.npz's 2-mic grid
    intersection is 18716 windows, vibration.npz's 2-accelerometer grid
    intersection is 19436 windows -- both share the identical `grid_t0_ns`
    and 1.0 s `grid_window_ns` (verified directly against the real caches), so
    index `i` means the same real second in both and truncating the longer
    array down to the shorter one's length is exact alignment, not a lossy
    approximation. This is the real shape that crashed `load_feature_snapshot`
    with a boolean-index length mismatch before this fix (`_zscore`'s
    `x[valid]`, `valid` being audio-length).

    Reviewer finding (round 1): truncation that actually discards data was
    silent -- a real mismatch like this one produced no log trace anywhere.
    `_truncate_to_common_length` must now `logger.warning` exactly once,
    naming the caller (*label*) and the per-array discarded window counts."""
    shorter = np.arange(3, dtype=np.float64)  # stands in for the 18716-long audio array
    longer = np.arange(5, dtype=np.float64)  # stands in for the 19436-long vibration array
    valid = np.array([True, False, True])
    with caplog.at_level(logging.WARNING):
        out_shorter, out_longer, out_valid = blr._truncate_to_common_length(
            shorter, longer, valid, label="mismatch_case"
        )
    np.testing.assert_array_equal(out_shorter, shorter)
    np.testing.assert_array_equal(out_longer, longer[:3])
    np.testing.assert_array_equal(out_valid, valid)
    # The whole point: indexing the truncated array by `valid` must not raise.
    out_longer[out_valid]
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "mismatch_case" in message
    assert "[3, 5, 3]" in message  # input lengths (shorter, longer, valid), in call order
    assert "[0, 2, 0]" in message  # per-array discarded windows -- only the 5-long one loses (2)


def test_truncate_to_common_length_preserves_2d_feature_matrix_columns() -> None:
    """`load_feature_snapshot` truncates 2D `(n_windows, n_features)` matrices
    along axis 0 only -- the feature/column count must survive untouched."""
    audio_feats = np.zeros((4, 135))
    vib_feats = np.zeros((6, 96))
    out_audio, out_vib = blr._truncate_to_common_length(
        audio_feats, vib_feats, label="feature_matrix_case"
    )
    assert out_audio.shape == (4, 135)
    assert out_vib.shape == (4, 96)
