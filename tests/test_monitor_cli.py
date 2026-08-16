"""Tests for `scripts/monitor.py`: CLI-level tests against a monkeypatched `discover`/`load_config`/
`prepare_run` seam feeding hand-built `PreparedRun`s directly (the established
`tests/test_apply_detector.py` style), with a REAL `MonitorSnapshot` built through
`rowii.runtime.snapshot.fit_snapshot`/`save_snapshot` into tmp_path -- the monitor
consumes the genuine artifact, never a mock of it.

Fixture sizing note (empirically verified before hardcoding, the
`tests/test_runtime_snapshot.py` practice): the two-state fixture uses TEN
alternating segments so BOTH labels survive into the snapshot at `SweepConfig`
defaults (that file's module docstring derives why eight is not enough), AND so
that the monitor day's own recalibration split (`split_by_segments` at the
snapshot's calibration_frac=0.5, seed=7) puts segments of BOTH parities on each
side -- verified: calibration side {0, 1, 3, 7, 8}, scoring side {2, 4, 5, 6, 9},
so every snapshot label has >= 1 calibration-side AND >= 1 scoring-side window in
recalibrate mode (verified counts 60/90 and 90/60). The no-conformal-data fixture
below derives its lone-segment placement from that same split at test time rather
than hardcoding segment 2, so it can never drift from `split_by_segments`.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rowii.anomaly.normalize import fit_pool_stats, fit_session_stats
from rowii.anomaly.references import split_by_segments
from rowii.anomaly.sweep import SweepConfig
from rowii.config import Config, DetectConfig
from rowii.io.dataset import RecordingIndex, Run
from rowii.pipeline import PreparedRun
from rowii.runtime.snapshot import MonitorSnapshot, fit_snapshot, save_snapshot
from rowii.signals.windows import WindowGrid

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

_FIT_RUN = "fit-day"
_MONITOR_RUN = "monitor-day"
_FIT_T0_NS = 0
_MON_T0_NS = 900_000_000_000
"""Deliberately different from `_FIT_T0_NS` (the `test_apply_detector.py` device):
pins that every UTC timestamp in the outputs comes from the MONITORED run's grid,
never the fit grid."""
_WINDOW_NS = 1_000_000_000

_N_SEGMENTS = 10
"""See module docstring -- 10, not 8, so both labels survive into the snapshot AND
both land on each side of the monitor day's recalibration split."""
_SEG_LEN = 30

_ALARM_COLUMNS = [
    "window", "t_utc_ns", "state", "score", "p_value", "alarm", "low_confidence", "role",
    "threshold_source", "near_transition", "state_name",
]
"""The exact alarms.parquet column contract (order included) --
`threshold_source` appended: per-window
`rolling`/`fit_day_fallback` on rolling-mode scored rows, the constant mode name in
the other modes, so the column exists uniformly. `near_transition`/`state_name`
appended at the END: BOTH ALWAYS present,
regardless of whether the snapshot carries `state_names`."""


# ---------------------------------------------------------------------------
# Hand-built fixtures (no gantner tree -- test_apply_detector.py's style)
# ---------------------------------------------------------------------------


def _fake_index() -> RecordingIndex:
    runs = [
        Run(name=_FIT_RUN, files={}, day_root=Path("/fake-fit-day-root")),
        Run(name=_MONITOR_RUN, files={}, day_root=Path("/fake-monitor-day-root")),
    ]
    return RecordingIndex(runs=runs, betriebsdaten=[], betriebsdaten_by_day={})


def _two_state_prepared(t0_ns: int, seed: int, n_features: int = 2) -> PreparedRun:
    """Two well-separated 'states' (feature values 0 and 20), alternating by
    segment -- `_N_SEGMENTS` segments of `_SEG_LEN` windows each, every window valid."""
    rng = np.random.default_rng(seed)
    feats: list[np.ndarray] = []
    seg_ids: list[np.ndarray] = []
    for s in range(_N_SEGMENTS):
        value = 20.0 * (s % 2)
        feats.append(rng.normal(value, 0.1, (_SEG_LEN, n_features)))
        seg_ids.append(np.full(_SEG_LEN, s, dtype=np.int64))
    features = np.vstack(feats)
    n = len(features)
    return PreparedRun(
        features=features,
        grid=WindowGrid(t0_ns=t0_ns, window_ns=_WINDOW_NS, n_windows=n),
        valid_mask=np.ones(n, dtype=bool),
        feature_names=[f"f{i}" for i in range(n_features)],
        segment_ids=np.concatenate(seg_ids),
    )


def _lone_high_state_prepared(t0_ns: int, seed: int) -> PreparedRun:
    """A monitor day whose 20-blob appears in exactly ONE segment, chosen (at test
    time, from `split_by_segments` itself) to land on the SCORING side of the
    monitor's recalibration split -- so the 20-blob's detected label has zero
    calibration-side windows and must take the `no_conformal_data` path."""
    seg_ids = np.repeat(np.arange(_N_SEGMENTS, dtype=np.int64), _SEG_LEN)
    valid = np.ones(_N_SEGMENTS * _SEG_LEN, dtype=bool)
    cfg = SweepConfig()  # the snapshot carries these exact defaults
    top = split_by_segments(seg_ids, valid, cfg.calibration_frac, cfg.seed)
    target_seg = int(min(set(seg_ids[top.scoring_windows].tolist())))

    rng = np.random.default_rng(seed)
    feats = [
        rng.normal(20.0 if s == target_seg else 0.0, 0.1, (_SEG_LEN, 2))
        for s in range(_N_SEGMENTS)
    ]
    features = np.vstack(feats)
    n = len(features)
    return PreparedRun(
        features=features,
        grid=WindowGrid(t0_ns=t0_ns, window_ns=_WINDOW_NS, n_windows=n),
        valid_mask=np.ones(n, dtype=bool),
        feature_names=["f0", "f1"],
        segment_ids=seg_ids,
    )


def _cfg(results_root: Path) -> Config:
    """Constructed directly (not `load_config()`) so ambient env vars / .env files
    can never leak into these CLI tests (`test_runtime_snapshot.py`'s rationale)."""
    return Config(
        data_root=Path("/fake-data-root"),
        results_root=results_root,
        detect=DetectConfig(n_states=2, min_dwell_s=3.0),
    )


def _make_snapshot(tmp_path: Path) -> tuple[Path, MonitorSnapshot]:
    """Fit a REAL snapshot on the fit-day fixture and persist it -- both labels
    survive (module docstring sizing note)."""
    fit_prepared = _two_state_prepared(_FIT_T0_NS, seed=0)
    snapshot, _ = fit_snapshot(
        fit_prepared, _cfg(tmp_path / "unused"), SweepConfig(),
        variant="fusion", fit_run=_FIT_RUN,
    )
    assert len(snapshot.thresholds) == 2  # fixture guarantee, not a monitor property
    path = tmp_path / "snapshot.npz"
    save_snapshot(path, snapshot)
    return path, snapshot


def _install_common_monkeypatches(
    monkeypatch: pytest.MonkeyPatch,
    monitor: object,
    results_root: Path,
    mon_prepared: PreparedRun,
) -> None:
    monkeypatch.setattr(monitor, "discover", lambda data_root: _fake_index())
    monkeypatch.setattr(monitor, "load_config", lambda: _cfg(results_root))

    def _fake_prepare_run(
        run: Run, variant: str, cfg: Config, *, use_cache: bool
    ) -> PreparedRun:
        assert variant == "fusion"  # the SNAPSHOT's variant must drive preparation
        if run.name == _MONITOR_RUN:
            return mon_prepared
        raise AssertionError(f"unexpected run name {run.name!r}")

    monkeypatch.setattr(monitor, "prepare_run", _fake_prepare_run)


# ---------------------------------------------------------------------------
# 1. Recalibrate mode end to end (the DEFAULT mode)
# ---------------------------------------------------------------------------


def test_recalibrate_mode_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import monitor

    snapshot_path, snapshot = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out"

    exit_code = monitor.main(
        ["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN, "--out", str(out_dir)]
    )
    assert exit_code == 0

    for name in (
        "segments.csv", "timeline.md", "alarms.parquet", "alarm_segments.csv",
        "monitor_notes.md",
    ):
        assert (out_dir / name).is_file()

    alarms = pd.read_parquet(out_dir / "alarms.parquet", engine="pyarrow")
    assert list(alarms.columns) == _ALARM_COLUMNS
    # One row per VALID window (all 300 are valid in this fixture), ascending.
    assert len(alarms) == int(mon_prepared.valid_mask.sum())
    assert alarms["window"].is_monotonic_increasing

    # t_utc_ns comes from the MONITORED run's grid: t0_ns + window * window_ns.
    expected_t = _MON_T0_NS + alarms["window"].to_numpy(dtype=np.int64) * _WINDOW_NS
    assert np.array_equal(alarms["t_utc_ns"].to_numpy(dtype=np.int64), expected_t)

    # Roles: both states are snapshot-known and both split sides are populated, so
    # ONLY scored / consumed_for_calibration may appear -- and both do.
    roles = set(alarms["role"].unique().tolist())
    assert roles == {"scored", "consumed_for_calibration"}

    # Outside rolling mode, threshold_source is the constant
    # mode name on EVERY row (uniform-column rule).
    assert (alarms["threshold_source"] == "recalibrate").all()

    scored = alarms[alarms["role"] == "scored"]
    assert set(scored["state"].unique().tolist()) == set(snapshot.thresholds)
    assert ((scored["p_value"] > 0.0) & (scored["p_value"] <= 1.0)).all()
    # Same-distribution sanity: realized alarm rate per state stays near alpha.
    for _state, group in scored.groupby("state"):
        assert group["alarm"].mean() <= 3 * snapshot.alpha

    # Calibration-bias rule: consumed windows are NEVER alarmed, no p-value.
    consumed = alarms[alarms["role"] == "consumed_for_calibration"]
    assert len(consumed) > 0
    assert not consumed["alarm"].any()
    assert consumed["p_value"].isna().all()

    notes = (out_dir / "monitor_notes.md").read_text()
    assert "recalibrate" in notes
    assert "did NOT hold" not in notes  # the frozen-mode warning must not leak here
    assert "no fault labels" in notes.lower()  # honesty framing
    assert "candidate" in notes.lower()
    assert snapshot.fit_run in notes

    # State half mirrors apply_detector's conventions, on the MONITOR grid.
    segments = pd.read_csv(out_dir / "segments.csv", parse_dates=["start_utc", "end_utc"])
    assert list(segments.columns) == ["start_utc", "end_utc", "duration_s", "cluster_id"]
    assert segments["start_utc"].min() == pd.Timestamp(_MON_T0_NS, unit="ns", tz="UTC")
    assert set(segments["cluster_id"].unique().tolist()) <= set(snapshot.thresholds) | {-1}


# ---------------------------------------------------------------------------
# 2. Frozen mode: verdict for EVERY valid known-state window + the
#    distribution-shift warning
# ---------------------------------------------------------------------------


def test_frozen_mode_flags_shift_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monitor

    snapshot_path, _snapshot = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out"

    exit_code = monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--thresholds", "frozen", "--out", str(out_dir),
        ]
    )
    assert exit_code == 0

    alarms = pd.read_parquet(out_dir / "alarms.parquet", engine="pyarrow")
    # Every valid window's state is snapshot-known here -> ALL rows are scored.
    assert len(alarms) == int(mon_prepared.valid_mask.sum())
    assert (alarms["role"] == "scored").all()
    assert ((alarms["p_value"] > 0.0) & (alarms["p_value"] <= 1.0)).all()
    # Constant mode name outside rolling mode.
    assert (alarms["threshold_source"] == "frozen").all()

    notes = (out_dir / "monitor_notes.md").read_text()
    assert "frozen" in notes
    assert "did NOT hold" in notes  # the warning, verbatim phrase


# ---------------------------------------------------------------------------
# 3. Geometry guard: the snapshot's trained columns are the scoring contract --
#    extra prepared columns are projected away (channel-availability drift, the
#    080726 TurbineVib-ch0 case); a MISSING snapshot column stays exit 2.
# ---------------------------------------------------------------------------


def test_geometry_extra_prepared_columns_projected_to_snapshot_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import monitor

    snapshot_path, _snapshot = _make_snapshot(tmp_path)  # fitted on 2 feature columns
    base = _two_state_prepared(_MON_T0_NS, seed=1)

    # Reference run: exact-geometry prepared run, the pre-drift behavior.
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", base)
    ref_dir = tmp_path / "out-ref"
    assert monitor.main(
        ["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN, "--out", str(ref_dir)]
    ) == 0

    # Drifted run: same two snapshot columns PLUS two live extra channels.
    rng = np.random.default_rng(99)
    extended = replace(
        base,
        features=np.hstack([base.features, rng.normal(5.0, 1.0, (len(base.features), 2))]),
        feature_names=[*base.feature_names, "extra_a", "extra_b"],
    )
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", extended)
    out_dir = tmp_path / "out"
    with caplog.at_level(logging.WARNING):
        exit_code = monitor.main(
            ["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN, "--out", str(out_dir)]
        )
    assert exit_code == 0

    projection_warnings = [
        r.message for r in caplog.records if "extra_a" in r.message
    ]
    assert projection_warnings, "projection must WARN, naming the dropped columns"
    assert "extra_b" in projection_warnings[0]

    # The extra columns must not influence scoring at all: identical alarms.
    ref = pd.read_parquet(ref_dir / "alarms.parquet", engine="pyarrow")
    got = pd.read_parquet(out_dir / "alarms.parquet", engine="pyarrow")
    pd.testing.assert_frame_equal(got, ref)


def test_geometry_missing_snapshot_column_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import monitor

    snapshot_path, snapshot = _make_snapshot(tmp_path)  # fitted on 2 feature columns
    base = _two_state_prepared(_MON_T0_NS, seed=1)
    renamed = replace(base, feature_names=["f0", "g1"])  # 'f1' gone, width equal
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", renamed)
    out_dir = tmp_path / "out"

    exit_code = monitor.main(
        ["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN, "--out", str(out_dir)]
    )
    assert exit_code == 2

    err = capsys.readouterr().err
    assert "missing" in err
    assert "'f1'" in err  # the absent snapshot column, named
    assert "fusion" in err  # the snapshot's variant, named
    assert str(len(snapshot.feature_names)) in err  # snapshot width (2)
    assert not out_dir.exists()  # refused BEFORE writing anything


# ---------------------------------------------------------------------------
# 4. Unknown detected state (absent from the snapshot) -> role="unknown_state",
#    counted in the notes (force it by stripping one label)
# ---------------------------------------------------------------------------


def test_unknown_state_windows_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monitor

    fit_prepared = _two_state_prepared(_FIT_T0_NS, seed=0)
    snapshot, _ = fit_snapshot(
        fit_prepared, _cfg(tmp_path / "unused"), SweepConfig(),
        variant="fusion", fit_run=_FIT_RUN,
    )
    dropped = max(snapshot.thresholds)
    stripped = replace(
        snapshot,
        references={la: v for la, v in snapshot.references.items() if la != dropped},
        calibration_scores={
            la: v for la, v in snapshot.calibration_scores.items() if la != dropped
        },
        thresholds={la: v for la, v in snapshot.thresholds.items() if la != dropped},
    )
    snapshot_path = tmp_path / "snapshot.npz"
    save_snapshot(snapshot_path, stripped)

    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out"

    exit_code = monitor.main(
        ["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN, "--out", str(out_dir)]
    )
    assert exit_code == 0

    alarms = pd.read_parquet(out_dir / "alarms.parquet", engine="pyarrow")
    unknown = alarms[alarms["state"] == dropped]
    assert len(unknown) > 0
    assert (unknown["role"] == "unknown_state").all()
    assert not unknown["alarm"].any()
    assert unknown["p_value"].isna().all()
    assert unknown["score"].isna().all()
    # The kept state still gets its normal recalibrate-mode roles.
    kept_roles = set(alarms.loc[alarms["state"] != dropped, "role"].unique().tolist())
    assert kept_roles == {"scored", "consumed_for_calibration"}

    notes = (out_dir / "monitor_notes.md").read_text()
    assert "unknown_state" in notes
    assert str(len(unknown)) in notes  # the count itself is reported


# ---------------------------------------------------------------------------
# 5. A snapshot-known state with ZERO calibration-side windows on the new run ->
#    ALL its windows role="no_conformal_data", no verdicts (binding rule)
# ---------------------------------------------------------------------------


def test_no_conformal_data_state_gets_no_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monitor

    snapshot_path, snapshot = _make_snapshot(tmp_path)
    mon_prepared = _lone_high_state_prepared(_MON_T0_NS, seed=5)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out"

    exit_code = monitor.main(
        ["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN, "--out", str(out_dir)]
    )
    assert exit_code == 0

    alarms = pd.read_parquet(out_dir / "alarms.parquet", engine="pyarrow")
    no_conf = alarms[alarms["role"] == "no_conformal_data"]
    assert len(no_conf) == _SEG_LEN  # exactly the lone 20-blob segment's windows
    lone_states = set(no_conf["state"].unique().tolist())
    assert len(lone_states) == 1
    lone_state = lone_states.pop()
    assert lone_state in set(snapshot.thresholds)
    # No verdicts for that state anywhere: never scored, never alarmed.
    assert (alarms.loc[alarms["state"] == lone_state, "role"] == "no_conformal_data").all()
    assert not no_conf["alarm"].any()
    assert no_conf["p_value"].isna().all()
    # The other state proceeds normally.
    other_roles = set(alarms.loc[alarms["state"] != lone_state, "role"].unique().tolist())
    assert other_roles == {"scored", "consumed_for_calibration"}

    notes = (out_dir / "monitor_notes.md").read_text()
    assert "no_conformal_data" in notes


# ---------------------------------------------------------------------------
# 6. alarm_segments.csv: exact schema + durations consistent with alarm count
#    (default --out location exercised here too: results/monitor/<run>/)
# ---------------------------------------------------------------------------


def test_alarm_segments_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import monitor

    snapshot_path, _snapshot = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    results_root = tmp_path / "results"
    _install_common_monkeypatches(monkeypatch, monitor, results_root, mon_prepared)

    exit_code = monitor.main(["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN])
    assert exit_code == 0

    out_dir = results_root / "monitor" / _MONITOR_RUN  # the documented default
    segments = pd.read_csv(out_dir / "alarm_segments.csv", parse_dates=["start_utc", "end_utc"])
    assert list(segments.columns) == ["start_utc", "end_utc", "duration_s"]

    # Maximal alarm runs must jointly cover exactly the alarmed windows (1 s each).
    alarms = pd.read_parquet(out_dir / "alarms.parquet", engine="pyarrow")
    n_alarm_windows = int(alarms["alarm"].sum())
    assert n_alarm_windows > 0  # fixture produces alarms (verified rates ~ alpha)
    assert len(segments) > 0
    assert segments["duration_s"].sum() == pytest.approx(n_alarm_windows * 1.0)


# ---------------------------------------------------------------------------
# 7. Unknown run name: exit 2, names the available runs (established convention)
# ---------------------------------------------------------------------------


def test_unknown_run_name_exits_2_and_lists_available_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import monitor

    snapshot_path, _snapshot = _make_snapshot(tmp_path)
    monkeypatch.setattr(monitor, "discover", lambda data_root: _fake_index())
    monkeypatch.setattr(monitor, "load_config", lambda: _cfg(tmp_path / "results"))

    def _boom_prepare_run(
        run: Run, variant: str, cfg: Config, *, use_cache: bool
    ) -> PreparedRun:
        raise AssertionError("prepare_run must not be called for an unknown run name")

    monkeypatch.setattr(monitor, "prepare_run", _boom_prepare_run)

    exit_code = monitor.main(
        ["--snapshot", str(snapshot_path), "--run", "totally-bogus-run"]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "totally-bogus-run" in err
    assert _FIT_RUN in err
    assert _MONITOR_RUN in err


# ---------------------------------------------------------------------------
# 8. --help documents every flag
# ---------------------------------------------------------------------------


def test_help_documents_every_flag(capsys: pytest.CaptureFixture[str]) -> None:
    import monitor

    with pytest.raises(SystemExit) as exc_info:
        monitor.main(["--help"])
    assert exc_info.value.code == 0

    out = capsys.readouterr().out
    for flag in ("--snapshot", "--run", "--thresholds", "--alpha", "--no-cache", "--out"):
        assert flag in out


# ---------------------------------------------------------------------------
# --alpha coverage (mutation gap) + CLI exit-2 paths
# ---------------------------------------------------------------------------


def test_alpha_override_changes_recalibrated_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kills a surviving mutant (hardcoded alpha in the
    recalibrate path): the applied per-state threshold must be bitwise
    `calibrate(cal_scores, 0.30)`, computed independently here from the
    parquet's own consumed-side scores -- not the default-alpha threshold."""
    import monitor

    from rowii.anomaly.conformal import calibrate

    snapshot_path, snapshot = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)

    out_default = tmp_path / "out-default"
    out_loose = tmp_path / "out-loose"
    assert monitor.main(
        ["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN, "--out", str(out_default)]
    ) == 0
    assert monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--alpha", "0.30", "--out", str(out_loose),
        ]
    ) == 0

    loose = pd.read_parquet(out_loose / "alarms.parquet", engine="pyarrow")
    default = pd.read_parquet(out_default / "alarms.parquet", engine="pyarrow")
    for state in snapshot.thresholds:
        cal_scores = loose.loc[
            (loose["role"] == "consumed_for_calibration") & (loose["state"] == state),
            "score",
        ].to_numpy()
        expected = calibrate(cal_scores, 0.30).threshold
        scored_loose = loose[(loose["role"] == "scored") & (loose["state"] == state)]
        scored_default = default[
            (default["role"] == "scored") & (default["state"] == state)
        ]
        # The alarm SET at alpha=0.30 must be exactly "score > expected" ...
        assert (
            scored_loose["alarm"].to_numpy()
            == (scored_loose["score"].to_numpy() > expected)
        ).all()
        # ... and looser than the default-alpha run on the same windows.
        assert scored_loose["alarm"].sum() >= scored_default["alarm"].sum()
    # At least one state must actually alarm MORE at the loose alpha, otherwise
    # this test cannot distinguish the mutant (same-distribution fixture makes
    # this deterministic at these seeds -- verified when writing the test).
    assert loose["alarm"].sum() > default["alarm"].sum()
    assert "0.3" in (out_loose / "monitor_notes.md").read_text()


@pytest.mark.parametrize("bad", ["0", "1", "1.5", "-0.1"])
def test_alpha_out_of_range_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    import monitor

    snapshot_path, _ = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    exit_code = monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--alpha", bad, "--out", str(tmp_path / "out-bad"),
        ]
    )
    assert exit_code == 2
    assert not (tmp_path / "out-bad").exists()


def test_missing_snapshot_file_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monitor

    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    exit_code = monitor.main(
        [
            "--snapshot", str(tmp_path / "does-not-exist.npz"), "--run", _MONITOR_RUN,
            "--out", str(tmp_path / "out-missing"),
        ]
    )
    assert exit_code == 2
    assert not (tmp_path / "out-missing").exists()


# ---------------------------------------------------------------------------
# --session-norm -- refusal on stats-less
#     snapshots, scoring-space-only wiring, detector-RAW invariance
# ---------------------------------------------------------------------------


def _stats_snapshot(
    tmp_path: Path, *, norm_minutes: float = 20.0
) -> tuple[Path, MonitorSnapshot]:
    """A REAL fit_snapshot product upgraded with fit-day session stats (the
    monitor's reference-side transform) -- the 300 s fixture sits entirely
    inside the default 20-minute prefix, so the stats cover every valid window."""
    fit_prepared = _two_state_prepared(_FIT_T0_NS, seed=0)
    snapshot, _ = fit_snapshot(
        fit_prepared, _cfg(tmp_path / "unused"), SweepConfig(),
        variant="fusion", fit_run=_FIT_RUN,
    )
    if norm_minutes > 0.0:
        stats = fit_session_stats(
            fit_prepared.features, fit_prepared.valid_mask, fit_prepared.grid,
            norm_minutes=norm_minutes,
        )
    else:
        stats = fit_pool_stats(fit_prepared.features)  # the pool-global sentinel
    snapshot = replace(snapshot, session_stats=stats)
    path = tmp_path / "snapshot_v2_stats.npz"
    save_snapshot(path, snapshot)
    return path, snapshot


def test_session_norm_refuses_snapshot_without_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A snapshot with `session_stats=None` (a v1 FILE or any snapshot fitted
    without session stats) must be REFUSED under --session-norm with exit 2 and a
    message naming the v1/no-stats cause."""
    import monitor

    snapshot_path, _ = _make_snapshot(tmp_path)  # no session stats
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out-refused"

    exit_code = monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--session-norm", "--out", str(out_dir),
        ]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "--session-norm" in err
    assert "v1" in err  # names the v1/no-stats cause
    assert "session" in err.lower()
    assert not out_dir.exists()  # refused BEFORE writing anything


def test_session_norm_removes_global_shift_in_recalibrate_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property end to end: a GLOBAL affine shift of the monitored run is
    absorbed by --session-norm -- (shifted run + norm) produces the SAME states,
    roles and alarm set as (unshifted run + norm), because each run's own first-N
    median/MAD stats map both into the same normalized scoring space."""
    import monitor

    snapshot_path, snapshot = _stats_snapshot(tmp_path)
    base = _two_state_prepared(_MON_T0_NS, seed=1)
    shifted = replace(base, features=base.features + 5.0)

    out_base = tmp_path / "out-base-norm"
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", base)
    assert monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--session-norm", "--out", str(out_base),
        ]
    ) == 0

    out_shifted = tmp_path / "out-shifted-norm"
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", shifted)
    assert monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--session-norm", "--out", str(out_shifted),
        ]
    ) == 0

    alarms_base = pd.read_parquet(out_base / "alarms.parquet", engine="pyarrow")
    alarms_shift = pd.read_parquet(out_shifted / "alarms.parquet", engine="pyarrow")

    # Non-vacuousness guard: the equality below must compare a REAL alarm set
    # (probed at test-writing time: 5 alarm windows at these seeds).
    assert int(alarms_base["alarm"].sum()) > 0

    # The shift is REMOVED: identical states/roles/alarm sets, scores equal to
    # floating-point noise (the shifted run's median shifts by exactly the offset).
    np.testing.assert_array_equal(
        alarms_shift["state"].to_numpy(), alarms_base["state"].to_numpy()
    )
    np.testing.assert_array_equal(
        alarms_shift["role"].to_numpy(), alarms_base["role"].to_numpy()
    )
    np.testing.assert_array_equal(
        alarms_shift["alarm"].to_numpy(), alarms_base["alarm"].to_numpy()
    )
    np.testing.assert_allclose(
        alarms_shift["score"].to_numpy(), alarms_base["score"].to_numpy(),
        rtol=1e-9, atol=1e-9,
    )

    # Recalibrate-mode FAR sanity on the shifted+normed run: near alpha per state.
    scored = alarms_shift[alarms_shift["role"] == "scored"]
    for _state, group in scored.groupby("state"):
        assert group["alarm"].mean() <= 3 * snapshot.alpha

    notes = (out_shifted / "monitor_notes.md").read_text()
    assert "session" in notes.lower()
    # Both stats' n_windows are named (the fit fixture and the monitored fixture
    # each have 300 valid windows inside the 20-minute prefix).
    assert "reference-side stats: n_windows=300" in notes
    assert "monitored-run stats: n_windows=300" in notes


def test_session_norm_detector_labels_bitwise_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Binding boundary: the DETECTOR always consumes RAW features -- the
    state column (and segments.csv) must be bitwise identical with and without
    --session-norm on the same run."""
    import monitor

    snapshot_path, _ = _stats_snapshot(tmp_path)
    shifted = replace(
        _two_state_prepared(_MON_T0_NS, seed=1),
        features=_two_state_prepared(_MON_T0_NS, seed=1).features + 5.0,
    )

    out_norm = tmp_path / "out-with-norm"
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", shifted)
    assert monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--session-norm", "--out", str(out_norm),
        ]
    ) == 0
    out_raw = tmp_path / "out-without-norm"
    assert monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--out", str(out_raw),
        ]
    ) == 0

    with_norm = pd.read_parquet(out_norm / "alarms.parquet", engine="pyarrow")
    without_norm = pd.read_parquet(out_raw / "alarms.parquet", engine="pyarrow")
    np.testing.assert_array_equal(
        with_norm["state"].to_numpy(), without_norm["state"].to_numpy()
    )
    assert (
        (out_norm / "segments.csv").read_text() == (out_raw / "segments.csv").read_text()
    )


def test_session_norm_pool_global_sentinel_falls_back_to_default_minutes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pooled snapshot's stats carry the `norm_minutes == 0.0` sentinel (pool-
    global, not first-N) -- the monitor must fit the MONITORED run's own stats over
    the DEFAULT 20-minute prefix instead of a zero-length one, and proceed."""
    import monitor

    snapshot_path, _ = _stats_snapshot(tmp_path, norm_minutes=0.0)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out-sentinel"

    assert monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--session-norm", "--out", str(out_dir),
        ]
    ) == 0
    notes = (out_dir / "monitor_notes.md").read_text()
    assert "pool-global" in notes
    assert "monitored-run stats: n_windows=300" in notes


def test_help_documents_session_norm(capsys: pytest.CaptureFixture[str]) -> None:
    import monitor

    with pytest.raises(SystemExit) as exc_info:
        monitor.main(["--help"])
    assert exc_info.value.code == 0
    assert "--session-norm" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --level-recal -- refusal on
#     medians-less snapshots, mutual exclusion with --session-norm, applying
#     AFTER the snapshot-contract projection with a shape-preserving recentre
# ---------------------------------------------------------------------------

_LEVEL_FEATURE_NAMES = ["f0_log_rms", "f1_octave_125"]
"""Level-bearing column names (both match `rowii.anomaly.levelrecal`'s
`_LEVEL_SUBSTRINGS`) -- the fixture needs a snapshot whose feature_names
actually carry a level column, unlike the module's default `f0`/`f1` names."""


def _level_prepared(t0_ns: int, seed: int) -> PreparedRun:
    """`_two_state_prepared`'s exact blob/segment layout (module docstring sizing
    note applies unchanged), with LEVEL-bearing feature names."""
    return replace(_two_state_prepared(t0_ns, seed), feature_names=list(_LEVEL_FEATURE_NAMES))


def _level_snapshot(
    tmp_path: Path, *, medians: dict[str, float] | None
) -> tuple[Path, MonitorSnapshot]:
    """A REAL fit_snapshot product optionally upgraded with level-recal reference
    medians (`level_recal_medians`) -- `medians=None` reproduces the
    v1/no-recal refusal fixture; a concrete dict reproduces a `run_step2
    --level-recal --save-snapshot` artifact (`_stats_snapshot`'s own device,
    injecting the field `fit_snapshot` itself never sets)."""
    fit_prepared = _level_prepared(_FIT_T0_NS, seed=0)
    snapshot, _ = fit_snapshot(
        fit_prepared, _cfg(tmp_path / "unused"), SweepConfig(),
        variant="fusion", fit_run=_FIT_RUN,
    )
    assert len(snapshot.thresholds) == 2
    if medians is not None:
        snapshot = replace(snapshot, level_recal_medians=medians)
    path = tmp_path / "snapshot_level_recal.npz"
    save_snapshot(path, snapshot)
    return path, snapshot


def test_monitor_level_recal_refuses_snapshot_without_medians(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A snapshot with `level_recal_medians=None` (a v1 file, or any
    snapshot fitted without --level-recal) must be REFUSED under --level-recal
    with exit 2 -- never a silent raw-space fallback (mirrors --session-norm's
    stats-less refusal, `test_session_norm_refuses_snapshot_without_stats`)."""
    import monitor

    snapshot_path, _snapshot = _level_snapshot(tmp_path, medians=None)
    mon_prepared = _level_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out-refused"

    exit_code = monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--level-recal", "--out", str(out_dir),
        ]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "--level-recal" in err
    assert not out_dir.exists()  # refused BEFORE writing anything


def test_monitor_level_recal_and_session_norm_mutually_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import monitor

    medians = {name: 0.0 for name in _LEVEL_FEATURE_NAMES}
    snapshot_path, _snapshot = _level_snapshot(tmp_path, medians=medians)
    mon_prepared = _level_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)

    exit_code = monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--level-recal", "--session-norm",
        ]
    )
    assert exit_code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_monitor_level_recal_applies_after_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--level-recal computes its run-side median AFTER the snapshot-
    contract projection, name-keyed against the PROJECTED feature_names -- an
    extra prepared column beyond the snapshot's own 2 must be dropped BEFORE the
    recal runs (else column_medians/apply_level_recal would see the wrong width
    and the run would wrongly exit 2). The recentring property mirrors the
    session-norm shift-removal test: additively shifting the monitored run's LEVEL columns and
    recentring with --level-recal reproduces the UNSHIFTED run's alarm set."""
    import monitor

    anchor = {name: 0.0 for name in _LEVEL_FEATURE_NAMES}
    snapshot_path, _snapshot = _level_snapshot(tmp_path, medians=anchor)

    base = _level_prepared(_MON_T0_NS, seed=1)
    out_base = tmp_path / "out-lrecal-base"
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", base)
    assert monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--level-recal", "--out", str(out_base),
        ]
    ) == 0

    rng = np.random.default_rng(123)
    extended = replace(
        base,
        # An EXTRA prepared column beyond the snapshot's 2 -- must be projected
        # away BEFORE level-recal ever sees it; plus a +5.0 shift on
        # ONLY the two level columns (never touching the extra column).
        features=np.hstack(
            [base.features + 5.0, rng.normal(5.0, 1.0, (len(base.features), 1))]
        ),
        feature_names=[*base.feature_names, "extra_shape_col"],
    )
    out_shifted = tmp_path / "out-lrecal-shifted"
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", extended)
    exit_code = monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--level-recal", "--out", str(out_shifted),
        ]
    )
    assert exit_code == 0

    alarms_base = pd.read_parquet(out_base / "alarms.parquet", engine="pyarrow")
    alarms_shift = pd.read_parquet(out_shifted / "alarms.parquet", engine="pyarrow")
    assert list(alarms_shift.columns) == _ALARM_COLUMNS  # schema unchanged

    # Shift removed + extra column safely projected away: identical verdicts.
    np.testing.assert_array_equal(
        alarms_shift["state"].to_numpy(), alarms_base["state"].to_numpy()
    )
    np.testing.assert_array_equal(
        alarms_shift["role"].to_numpy(), alarms_base["role"].to_numpy()
    )
    np.testing.assert_array_equal(
        alarms_shift["alarm"].to_numpy(), alarms_base["alarm"].to_numpy()
    )
    np.testing.assert_allclose(
        alarms_shift["score"].to_numpy(), alarms_base["score"].to_numpy(),
        rtol=1e-9, atol=1e-9,
    )

    notes = (out_shifted / "monitor_notes.md").read_text()
    assert "level" in notes.lower()
    assert "recal" in notes.lower()


def test_help_documents_level_recal(capsys: pytest.CaptureFixture[str]) -> None:
    import monitor

    with pytest.raises(SystemExit) as exc_info:
        monitor.main(["--help"])
    assert exc_info.value.code == 0
    assert "--level-recal" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Equivalence tripwire pinning
# monitor's own `_first_n_minutes_rows` to `rowii.anomaly.normalize.
# fit_session_stats`'s window-membership rule -- mirrors
# `tests/test_step2_pooled_cli.py`'s `test_first_n_minutes_rows_matches_
# fit_session_stats_window_membership` (run_step2's own duplicate against the
# same canonical rule). The rule is now triplicated (run_step2, monitor,
# fit_session_stats each independently encode "window START offset <
# norm_minutes*60s AND valid_mask") -- a failure here means monitor's copy
# drifted from the other two.
# ---------------------------------------------------------------------------


def test_first_n_minutes_rows_matches_fit_session_stats_window_membership() -> None:
    import monitor

    rng = np.random.default_rng(11)
    n = 40
    window_ns = 10_000_000_000  # 10 s/window -> cutoff at 5 min = window index 30
    grid = WindowGrid(t0_ns=0, window_ns=window_ns, n_windows=n)
    valid_mask = np.ones(n, dtype=bool)
    valid_mask[[2, 29]] = False  # invalid windows inside AND at the time-cutoff edge
    features = rng.normal(0.0, 100.0, (n, 3))  # well-spread, non-degenerate values
    prepared = PreparedRun(
        features=features, grid=grid, valid_mask=valid_mask,
        feature_names=["f0", "f1", "f2"], segment_ids=np.zeros(n, dtype=np.int64),
    )

    norm_minutes = 5.0
    rows = monitor._first_n_minutes_rows(prepared, norm_minutes)
    stats = fit_session_stats(features, valid_mask, grid, norm_minutes=norm_minutes)

    assert rows.shape[0] == stats.n_windows
    # `fit_session_stats` exposes no raw rows -- recompute its own documented
    # `_center_scale` formula (median; MAD * 1.4826, floored at 1e-8) from
    # `_first_n_minutes_rows`' selected rows: if the two rules ever select a
    # DIFFERENT row set, this well-spread fixture makes the median/MAD diverge.
    expected_center = np.median(rows, axis=0)
    expected_mad = np.median(np.abs(rows - expected_center), axis=0)
    expected_scale = np.maximum(expected_mad * 1.4826, 1e-8)
    np.testing.assert_array_equal(stats.center, expected_center)
    np.testing.assert_array_equal(stats.scale, expected_scale)


# ---------------------------------------------------------------------------
# --thresholds rolling --
#     per-window trailing thresholds with conformal-floor fallback, the
#     threshold_source column, coverage stats, and the all-invalid guard
# ---------------------------------------------------------------------------

_ROLLING_ALPHA = 0.30
"""Rolling tests' alpha: the conformal floor is ceil(1/0.30) - 1 = 3 trailing
windows, small enough that a 1-minute trailing window over the 10-segment fixture
exercises BOTH branches deterministically. Derivation (module-docstring split:
calibration segments {0, 1, 3, 7, 8}, scoring {2, 4, 5, 6, 9}; segment s = windows
[30s, 30s + 30), 1-s windows, states alternate by segment parity): the even-parity
state's scored segment 2 rolls for w = 60..87 (trailing calibration windows
90 - w from segment 0) and falls back for w = 88..89 and ALL of segments 4/6 (no
same-state calibration window within 60 s); the odd-parity state rolls for
w = 150..177 and 270..297 and falls back for w = 178..179 and 298..299."""

_ROLLING_M_NS = 60 * 1_000_000_000
"""1 minute (--rolling-minutes 1) in grid nanoseconds."""

_ROLLING_FLOOR = 3
"""ceil(1 / _ROLLING_ALPHA) - 1 (the conformal floor at alpha=0.30)."""


def _run_rolling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    monitor: object,
    out_dir: Path,
    *extra: str,
) -> tuple[MonitorSnapshot, pd.DataFrame, str]:
    """One rolling-mode monitor pass over the standard two-state fixture at
    `--rolling-minutes 1 --alpha 0.30`; returns (snapshot, alarms, notes)."""
    snapshot_path, snapshot = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    exit_code = monitor.main(  # type: ignore[attr-defined]  # object-typed seam
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--thresholds", "rolling", "--rolling-minutes", "1",
            "--alpha", str(_ROLLING_ALPHA), "--out", str(out_dir), *extra,
        ]
    )
    assert exit_code == 0
    alarms = pd.read_parquet(out_dir / "alarms.parquet", engine="pyarrow")
    notes = (out_dir / "monitor_notes.md").read_text()
    return snapshot, alarms, notes


def test_rolling_mode_fallback_and_rolling_branches_bitwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core semantics, verified from first principles: for EVERY scored
    window, the trailing set is recomputed here from the parquet's own consumed rows
    (same state, window start in [t_w - M, t_w)) and the emitted threshold_source,
    alarm bit, p-value and low_confidence must match the branch that trailing count
    dictates -- bitwise (`calibrate`/`p_values` on the recomputed trailing set for
    rolling windows, the snapshot's STORED threshold/calibration scores for
    fallback windows). Both branches must be non-empty for both states
    (`_ROLLING_ALPHA` docstring derivation)."""
    import monitor

    from rowii.anomaly.conformal import calibrate as _calibrate
    from rowii.anomaly.conformal import p_values as _p_values

    snapshot, alarms, _notes = _run_rolling(
        tmp_path, monkeypatch, monitor, tmp_path / "out-rolling"
    )
    assert list(alarms.columns) == _ALARM_COLUMNS

    # Same roles as recalibrate mode: rolling replaces HOW thresholds are derived,
    # not WHICH windows get verdicts (calibration side stays consumed).
    assert set(alarms["role"].unique().tolist()) == {"scored", "consumed_for_calibration"}
    consumed = alarms[alarms["role"] == "consumed_for_calibration"]
    scored = alarms[alarms["role"] == "scored"]
    assert not consumed["alarm"].any()
    assert consumed["p_value"].isna().all()
    assert (consumed["threshold_source"] == "").all()
    assert set(scored["threshold_source"].unique().tolist()) == {
        "rolling", "fit_day_fallback",
    }

    for state, group in scored.groupby("state"):
        cal = consumed[consumed["state"] == state].sort_values("t_utc_ns")
        cal_t = cal["t_utc_ns"].to_numpy(dtype=np.int64)
        cal_scores = cal["score"].to_numpy(dtype=np.float64)
        frozen = snapshot.thresholds[int(state)]
        stored_cal = snapshot.calibration_scores[int(state)]
        n_roll = n_fall = 0
        for row in group.itertuples(index=False):
            t_w = int(row.t_utc_ns)
            trailing = cal_scores[(cal_t >= t_w - _ROLLING_M_NS) & (cal_t < t_w)]
            if trailing.size >= _ROLLING_FLOOR:
                n_roll += 1
                assert row.threshold_source == "rolling"
                expected = _calibrate(trailing, _ROLLING_ALPHA)
                assert bool(row.alarm) == bool(row.score > expected.threshold)
                assert row.p_value == _p_values(np.array([row.score]), trailing)[0]
                assert not row.low_confidence  # count >= floor => real threshold
            else:
                n_fall += 1
                assert row.threshold_source == "fit_day_fallback"
                assert bool(row.alarm) == bool(row.score > frozen.threshold)
                assert row.p_value == _p_values(np.array([row.score]), stored_cal)[0]
                assert bool(row.low_confidence) == frozen.low_confidence
        # Non-vacuousness: BOTH branches occur for BOTH states at these settings.
        assert n_roll > 0
        assert n_fall > 0

    # Hand-computed chosen window (docstring derivation): w=60 is the first
    # scoring-side window (segment 2); the ONLY same-state windows starting in
    # [t_60 - 1 min, t_60) are calibration segment 0 = windows 0..29 -> it rolls
    # on exactly that 30-window trailing set.
    w60 = scored[scored["window"] == 60].iloc[0]
    assert w60["threshold_source"] == "rolling"
    trail = consumed[
        (consumed["state"] == w60["state"])
        & (consumed["t_utc_ns"] >= int(w60["t_utc_ns"]) - _ROLLING_M_NS)
        & (consumed["t_utc_ns"] < int(w60["t_utc_ns"]))
    ]
    assert sorted(trail["window"].tolist()) == list(range(0, 30))
    expected_60 = _calibrate(
        trail["score"].to_numpy(dtype=np.float64), _ROLLING_ALPHA
    ).threshold
    assert bool(w60["alarm"]) == bool(w60["score"] > expected_60)


def test_rolling_coverage_stats_in_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MANDATORY output: the notes carry a per-state trailing-coverage table
    whose counts/fractions match the parquet, the motivating measurement as the
    rationale line, and the double-reference honesty note (slow-fault
    absorption; the threshold_source column as the visibility)."""
    import monitor

    _snapshot, alarms, notes = _run_rolling(
        tmp_path, monkeypatch, monitor, tmp_path / "out-cov"
    )
    scored = alarms[alarms["role"] == "scored"]
    assert "## Rolling trailing coverage" in notes
    assert f"conformal floor = {_ROLLING_FLOOR}" in notes
    assert "M = 1 min" in notes
    for state, group in scored.groupby("state"):
        n_scored = len(group)
        n_roll = int((group["threshold_source"] == "rolling").sum())
        n_fall = int((group["threshold_source"] == "fit_day_fallback").sum())
        assert n_scored == n_roll + n_fall
        expected_row = (
            f"| {int(state)} | {n_scored} | {n_roll} | {n_fall} "
            f"| {n_roll / n_scored:.4f} |"
        )
        assert expected_row in notes
    # The motivating measurement is cited as the rationale.
    assert "46.8%" in notes
    assert "M=20" in notes
    assert "290626" in notes
    # The double-reference honesty note.
    assert "absorb" in notes.lower()
    assert "side by side" in notes.lower()
    assert "fit_day_fallback" in notes
    assert "rolling" in notes


def test_rolling_composes_with_session_norm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rolling must compose with --session-norm (scores in whichever space is
    active): smoke -- exit 0, both threshold sources present (the source PATTERN is
    norm-invariant: labels come from raw features and the split is fixed, so
    trailing COUNTS are unchanged), scores finite, both notes sections present."""
    import monitor

    snapshot_path, _snapshot = _stats_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out-rolling-norm"

    exit_code = monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--thresholds", "rolling", "--rolling-minutes", "1",
            "--alpha", str(_ROLLING_ALPHA), "--session-norm", "--out", str(out_dir),
        ]
    )
    assert exit_code == 0

    alarms = pd.read_parquet(out_dir / "alarms.parquet", engine="pyarrow")
    scored = alarms[alarms["role"] == "scored"]
    assert set(scored["threshold_source"].unique().tolist()) == {
        "rolling", "fit_day_fallback",
    }
    assert np.isfinite(scored["score"].to_numpy(dtype=np.float64)).all()
    notes = (out_dir / "monitor_notes.md").read_text()
    assert "Session normalization" in notes
    assert "## Rolling trailing coverage" in notes


def test_rolling_default_minutes_is_60(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --rolling-minutes, M defaults to 60 (binding). The 300-s
    fixture sits entirely inside 60 minutes, so every scored window's trailing set
    is its state's FULL earlier calibration side (>= 30 windows >= floor 3) -- all
    sources must be `rolling` and the notes must name M = 60."""
    import monitor

    snapshot_path, _snapshot = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out-default-m"

    exit_code = monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--thresholds", "rolling", "--alpha", str(_ROLLING_ALPHA),
            "--out", str(out_dir),
        ]
    )
    assert exit_code == 0
    alarms = pd.read_parquet(out_dir / "alarms.parquet", engine="pyarrow")
    scored = alarms[alarms["role"] == "scored"]
    assert (scored["threshold_source"] == "rolling").all()
    assert "M = 60 min" in (out_dir / "monitor_notes.md").read_text()


def test_eval_events_consumes_rolling_alarms_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The events harness must keep working on a rolling-mode parquet: its role
    filter selects the scored rows and the extra threshold_source column is
    ignored (evaluate_events consumes t_utc_ns/alarm/role only)."""
    import monitor

    from rowii.eval.events import evaluate_events

    _snapshot, alarms, _notes = _run_rolling(
        tmp_path, monkeypatch, monitor, tmp_path / "out-events"
    )
    # Windows 150..179 are scoring-side (segment 5, module-docstring split).
    events = pd.DataFrame(
        {
            "start_utc": [
                pd.Timestamp(_MON_T0_NS + 150 * _WINDOW_NS, unit="ns", tz="UTC").isoformat()
            ],
            "end_utc": [
                pd.Timestamp(_MON_T0_NS + 180 * _WINDOW_NS, unit="ns", tz="UTC").isoformat()
            ],
        }
    )
    result = evaluate_events(alarms, events, window_s=1.0)
    assert result.n_events == 1
    assert result.n_detected in (0, 1)
    # The role filter really applied: non-event scored windows only (150 scored
    # windows total, 30 inside the event) -- consumed rows never count.
    n_scored = int((alarms["role"] == "scored").sum())
    assert result.false_alarm_windows <= n_scored - 30


def test_all_invalid_run_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A monitored run whose valid_mask is all False must be
    refused with a clean exit 2 naming the zero-valid-windows cause BEFORE the
    detector apply (previously a raw sklearn ValueError), writing nothing."""
    import monitor

    snapshot_path, _snapshot = _make_snapshot(tmp_path)
    base = _two_state_prepared(_MON_T0_NS, seed=1)
    all_invalid = replace(
        base, valid_mask=np.zeros(base.features.shape[0], dtype=bool)
    )
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", all_invalid)
    out_dir = tmp_path / "out-invalid"

    exit_code = monitor.main(
        ["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN, "--out", str(out_dir)]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "zero valid windows" in err
    assert _MONITOR_RUN in err
    assert not out_dir.exists()  # refused BEFORE writing anything


def test_rolling_minutes_requires_rolling_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--rolling-minutes with any other --thresholds mode is a usage error (exit 2
    naming both flags), not a silently ignored knob."""
    import monitor

    snapshot_path, _snapshot = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out-mins-guard"

    exit_code = monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--rolling-minutes", "30", "--out", str(out_dir),
        ]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "--rolling-minutes" in err
    assert "rolling" in err
    assert not out_dir.exists()


@pytest.mark.parametrize("bad", ["0", "-5", "nan"])
def test_rolling_minutes_must_be_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    import monitor

    snapshot_path, _snapshot = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out-mins-bad"

    exit_code = monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--thresholds", "rolling", "--rolling-minutes", bad, "--out", str(out_dir),
        ]
    )
    assert exit_code == 2
    assert not out_dir.exists()


def test_help_documents_rolling_flags(capsys: pytest.CaptureFixture[str]) -> None:
    import monitor

    with pytest.raises(SystemExit) as exc_info:
        monitor.main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--rolling-minutes" in out
    assert "rolling" in out  # the third --thresholds choice is documented


# ---------------------------------------------------------------------------
# Boundary unit pin, rolling-off-norm warning, branches
# ---------------------------------------------------------------------------


def test_trailing_bounds_upper_edge_exclusive_lower_inclusive() -> None:
    """cal/scoring segment-disjointness makes cal_t ==
    scr_t unreachable through the CLI, so the upper-edge exclusivity had NO
    test (the side="right" mutation survived the suite). Pin both edges on
    synthetic arrays directly."""
    import monitor

    m_ns = 10
    cal_t = np.array([0, 5, 10, 15, 20], dtype=np.int64)
    # Scored window exactly AT a calibration time: that calibration window is
    # EXCLUDED (upper edge exclusive). Scored window exactly M after one: that
    # one is INCLUDED (lower edge inclusive).
    scr_t = np.array([10, 25], dtype=np.int64)
    lo, hi = monitor._trailing_bounds(cal_t, scr_t, m_ns)
    # Window at t=10, interval [0, 10): cal 0 and 5 in, cal 10 OUT.
    assert (lo[0], hi[0]) == (0, 2)
    # Window at t=25, interval [15, 25): cal 15 and 20 in.
    assert (lo[1], hi[1]) == (3, 5)


def test_rolling_without_session_norm_warns_and_fallback_matches_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The stats-bearing-snapshot warning must name
    rolling's fallback branch, and that branch must inherit frozen mode's
    verdicts bitwise (probed correct by the review; pinned here)."""
    import logging

    import monitor

    snapshot_path, _ = _stats_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)

    out_roll = tmp_path / "out-roll"
    with caplog.at_level(logging.WARNING):
        assert monitor.main(
            [
                "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
                "--thresholds", "rolling", "--rolling-minutes", "10000",
                "--out", str(out_roll),
            ]
        ) == 0
    warned = " ".join(r.getMessage() for r in caplog.records)
    assert "fit_day_fallback" in warned

    out_frozen = tmp_path / "out-frozen"
    assert monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--thresholds", "frozen", "--out", str(out_frozen),
        ]
    ) == 0

    roll = pd.read_parquet(out_roll / "alarms.parquet", engine="pyarrow")
    frozen = pd.read_parquet(out_frozen / "alarms.parquet", engine="pyarrow")
    fb = roll[roll["threshold_source"] == "fit_day_fallback"]
    if len(fb):
        fz = frozen.set_index("window").loc[fb["window"]]
        assert (fb["alarm"].to_numpy() == fz["alarm"].to_numpy()).all()
        assert np.array_equal(fb["p_value"].to_numpy(), fz["p_value"].to_numpy())


def test_rolling_zero_calibration_state_all_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot-known state with zero calibration-side
    windows on the monitored run takes the flagged fallback for ALL its scored
    windows in rolling mode (the designed behavior)."""
    import monitor

    snapshot_path, snapshot = _make_snapshot(tmp_path)
    mon_prepared = _lone_high_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)

    out_dir = tmp_path / "out"
    assert monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--thresholds", "rolling", "--out", str(out_dir),
        ]
    ) == 0
    alarms = pd.read_parquet(out_dir / "alarms.parquet", engine="pyarrow")
    scored = alarms[alarms["role"] == "scored"]
    lone = scored[scored["threshold_source"] == "fit_day_fallback"]
    assert len(lone) > 0


def test_rolling_minutes_overflow_guard_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import monitor

    snapshot_path, _ = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    exit_code = monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--thresholds", "rolling", "--rolling-minutes", "1000000000",
            "--out", str(tmp_path / "out"),
        ]
    )
    assert exit_code == 2
    assert "decade" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 10. --exclude-calibration-events: induced-event intervals are
#     BANNED from the calibration side and rescued into the scoring side, so
#     strike days calibrate on strike-free windows and every event is evaluable.
# ---------------------------------------------------------------------------


def _anomalous_calibration_prepared() -> PreparedRun:
    """The two-state fixture with an injected anomalous burst at windows 5..9
    (inside calibration-side segment 0 -- module docstring split): raw value
    (30, 14) is DIRECTIONALLY off the 20-blob's (1, 1) axis, so its cosine kNN
    score (~6e-2) sits orders of magnitude above the blob's internal spread
    (~5e-5) -- an alarm is guaranteed once the window is actually scored --
    while its standardized position labels it as the 20-blob state."""
    base = _two_state_prepared(_MON_T0_NS, seed=1)
    feats = base.features.copy()
    feats[5:10] = (30.0, 14.0)
    return replace(base, features=feats)


def _write_events_csv(path: Path, rows: list[tuple[str, str]]) -> Path:
    lines = ["# provenance comment line (must be skipped)", "start_utc,end_utc,kind"]
    lines += [f"{s},{e},induced-strike" for s, e in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


_EVENT_5S_10S = ("1970-01-01T00:15:05+00:00", "1970-01-01T00:15:10+00:00")
"""Covers exactly windows 5..9 of the fixture grid (t0 = 00:15:00 UTC, 1-s
windows) at --exclude-tolerance-s 0."""


def test_exclude_calibration_events_rescues_and_alarms_event_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monitor

    snapshot_path, _ = _make_snapshot(tmp_path)
    mon_prepared = _anomalous_calibration_prepared()
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    ev = _write_events_csv(tmp_path / "events.csv", [_EVENT_5S_10S])

    base_dir, excl_dir = tmp_path / "out-base", tmp_path / "out-excl"
    assert monitor.main(
        ["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN, "--out", str(base_dir)]
    ) == 0
    assert monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--exclude-calibration-events", str(ev), "--exclude-tolerance-s", "0",
            "--out", str(excl_dir),
        ]
    ) == 0

    base = pd.read_parquet(base_dir / "alarms.parquet", engine="pyarrow")
    excl = pd.read_parquet(excl_dir / "alarms.parquet", engine="pyarrow")
    burst = base["window"].isin(range(5, 10))

    # Without exclusion the burst is CONSUMED into calibration: never alarmed,
    # and it silently inflates the state's threshold.
    assert (base.loc[burst, "role"] == "consumed_for_calibration").all()
    assert not base.loc[burst, "alarm"].any()

    # With exclusion the same windows are scored against a strike-free
    # threshold -- and this burst is a guaranteed outlier.
    burst_x = excl["window"].isin(range(5, 10))
    assert (excl.loc[burst_x, "role"] == "scored").all()
    assert excl.loc[burst_x, "p_value"].notna().all()
    assert excl.loc[burst_x, "alarm"].all()

    # Exactly the five burst windows moved sides; nothing else changed roles.
    assert (base["role"] == "consumed_for_calibration").sum() - 5 == (
        excl["role"] == "consumed_for_calibration"
    ).sum()
    assert (excl["role"] == "scored").sum() == (base["role"] == "scored").sum() + 5

    notes = (excl_dir / "monitor_notes.md").read_text()
    assert "exclude" in notes.lower()
    assert "events.csv" in notes


def test_exclude_events_covering_all_of_a_states_calibration_goes_no_conformal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monitor

    snapshot_path, _ = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    # The 0-blob's calibration-side segments are 0 and 8 (module docstring
    # split; even segments carry the 0-blob) -- cover both fully.
    ev = _write_events_csv(
        tmp_path / "events.csv",
        [
            ("1970-01-01T00:15:00+00:00", "1970-01-01T00:15:30+00:00"),
            ("1970-01-01T00:19:00+00:00", "1970-01-01T00:19:30+00:00"),
        ],
    )
    out_dir = tmp_path / "out"
    assert monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--exclude-calibration-events", str(ev), "--exclude-tolerance-s", "0",
            "--out", str(out_dir),
        ]
    ) == 0

    alarms = pd.read_parquet(out_dir / "alarms.parquet", engine="pyarrow")
    zero_state = int(alarms.loc[alarms["window"] == 0, "state"].iloc[0])
    zero_rows = alarms["state"] == zero_state
    # Every calibration window of the 0-blob state was excluded -> the
    # no-conformal path for ALL its windows (rescued ones included).
    assert (alarms.loc[zero_rows, "role"] == "no_conformal_data").all()
    # The other state still calibrates and scores normally.
    other_rows = alarms["state"] != zero_state
    assert set(alarms.loc[other_rows, "role"]) == {"scored", "consumed_for_calibration"}


def test_exclude_events_naive_timestamps_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import monitor

    snapshot_path, _ = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    ev = _write_events_csv(
        tmp_path / "events.csv", [("1970-01-01T00:15:05", "1970-01-01T00:15:10")]
    )
    out_dir = tmp_path / "out"
    exit_code = monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--exclude-calibration-events", str(ev), "--out", str(out_dir),
        ]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "tz-aware" in err
    assert not out_dir.exists()


def test_exclude_events_frozen_mode_warns_and_has_no_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import monitor

    snapshot_path, _ = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    ev = _write_events_csv(tmp_path / "events.csv", [_EVENT_5S_10S])

    plain_dir, excl_dir = tmp_path / "out-plain", tmp_path / "out-excl"
    assert monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--thresholds", "frozen", "--out", str(plain_dir),
        ]
    ) == 0
    with caplog.at_level(logging.WARNING):
        assert monitor.main(
            [
                "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
                "--thresholds", "frozen",
                "--exclude-calibration-events", str(ev), "--out", str(excl_dir),
            ]
        ) == 0
    assert any(
        "frozen" in r.message and "exclude" in r.message.lower() for r in caplog.records
    )
    plain = pd.read_parquet(plain_dir / "alarms.parquet", engine="pyarrow")
    excl = pd.read_parquet(excl_dir / "alarms.parquet", engine="pyarrow")
    pd.testing.assert_frame_equal(excl, plain)


def test_exclude_events_rolling_mode_scores_rescued_windows_via_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monitor

    snapshot_path, _ = _make_snapshot(tmp_path)
    mon_prepared = _anomalous_calibration_prepared()
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    ev = _write_events_csv(tmp_path / "events.csv", [_EVENT_5S_10S])

    out_dir = tmp_path / "out"
    assert monitor.main(
        [
            "--snapshot", str(snapshot_path), "--run", _MONITOR_RUN,
            "--thresholds", "rolling", "--rolling-minutes", "2", "--alpha", "0.05",
            "--exclude-calibration-events", str(ev), "--exclude-tolerance-s", "0",
            "--out", str(out_dir),
        ]
    ) == 0

    alarms = pd.read_parquet(out_dir / "alarms.parquet", engine="pyarrow")
    burst = alarms["window"].isin(range(5, 10))
    # Rescued into scoring; no trailing same-state calibration exists that
    # early in the run, so the fit-day fallback threshold applies -- and the
    # burst is an outlier against it.
    assert (alarms.loc[burst, "role"] == "scored").all()
    assert (alarms.loc[burst, "threshold_source"] == "fit_day_fallback").all()
    assert alarms.loc[burst, "alarm"].all()


# ---------------------------------------------------------------------------
# 11. Named states + near_transition / --suppress-transition-alarms
# ---------------------------------------------------------------------------


def test_near_transition_mask_marks_boundary_windows_valid_subseq() -> None:
    import monitor
    # labels: run of 0s, an invalid gap (-1), run of 1s. The 0->1 change is at the
    # FIRST valid 1 (index 6); invalid window 3 is NOT a change.
    labels = np.array([0, 0, 0, -1, 1, 1, 1, 1], dtype=np.int64)
    valid = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    mask = monitor._near_transition_mask(labels, valid, window_ns=1_000_000_000, w_seconds=1.0)
    assert mask[3] == False  # invalid window never flagged   # noqa: E712
    assert mask[4] == True   # first valid 1 (boundary onset) # noqa: E712
    assert mask[2] == True   # last valid 0, within 1 window of the boundary onset  # noqa: E712
    assert mask[7] == False  # 3 valid windows past the boundary -> outside +-1  # noqa: E712


def test_near_transition_mask_floors_w_windows_at_one() -> None:
    """`_near_transition_mask` must floor its w_windows
    conversion at 1, for literal parity with `FittedDetector._finish`'s own
    `max(1, round(min_dwell_s / window_s))` -- a w_seconds small enough that
    `round()` truncates to 0 steps must still flag the immediate (+-1 step)
    neighbours of a transition, not degenerate to "only the boundary window
    itself"."""
    import monitor
    labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    valid = np.ones(5, dtype=bool)
    # window_ns=1s, w_seconds=0.4 -> round(0.4) == 0 without the floor.
    mask = monitor._near_transition_mask(labels, valid, window_ns=1_000_000_000, w_seconds=0.4)
    assert mask[3] == True    # boundary onset itself                       # noqa: E712
    assert mask[2] == True    # ONE step before onset -- floored in, not dropped  # noqa: E712
    assert mask[4] == True    # ONE step after onset                        # noqa: E712
    assert mask[1] == False   # TWO steps before -- still outside +-1       # noqa: E712
    assert mask[0] == False   # TWO steps... clearly outside                # noqa: E712


def test_apply_transition_suppression_forces_false_and_counts() -> None:
    import monitor
    alarm = np.array([True, True, False, True], dtype=bool)
    near = np.array([True, False, True, True], dtype=bool)
    role = np.array(["scored", "scored", "scored", "consumed_for_calibration"], dtype=object)
    n = monitor._apply_transition_suppression(alarm, near, role)
    assert n == 1                                        # only window 0: near & scored & was-True
    assert alarm.tolist() == [False, True, False, True]  # window 3 (consumed) untouched


def test_state_name_column_always_present_fallback_cluster_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monitor
    # snapshot WITHOUT state_names -> every alarm row's state_name is cluster-<id>.
    snapshot_path, _snapshot = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out"
    assert monitor.main(
        ["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN, "--out", str(out_dir)]
    ) == 0
    alarms = pd.read_parquet(out_dir / "alarms.parquet")
    assert list(alarms.columns) == _ALARM_COLUMNS  # near_transition + state_name appended
    assert alarms["state_name"].str.startswith("cluster-").all()
    assert alarms["near_transition"].dtype == bool


def test_named_snapshot_surfaces_state_name_and_mapped_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monitor
    _path, snapshot = _make_snapshot(tmp_path)
    labels = sorted(snapshot.thresholds)  # both fixture labels survive (module docstring)
    named = replace(snapshot, state_names={labels[0]: "turbine", labels[1]: "pump"})
    named_path = tmp_path / "named.npz"
    save_snapshot(named_path, named)  # save_snapshot does not re-validate state_names
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out"
    assert monitor.main(
        ["--snapshot", str(named_path), "--run", _MONITOR_RUN, "--out", str(out_dir)]
    ) == 0
    alarms = pd.read_parquet(out_dir / "alarms.parquet")
    assert set(alarms["state_name"].unique()) <= {"turbine", "pump"}
    segments = pd.read_csv(out_dir / "segments.csv")
    assert "mapped_mode" in segments.columns  # present because state_names present
    timeline = (out_dir / "timeline.md").read_text()
    assert "(turbine)" in timeline or "(pump)" in timeline


def test_suppress_transition_alarms_invariant_and_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monitor
    snapshot_path, _snapshot = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out"
    assert monitor.main(
        ["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN, "--out", str(out_dir),
         "--suppress-transition-alarms"]
    ) == 0
    alarms = pd.read_parquet(out_dir / "alarms.parquet")
    scored = alarms[alarms["role"] == "scored"]
    # invariant: no scored near_transition window remains an alarm; audit columns retained.
    assert not bool((scored["alarm"] & scored["near_transition"]).any())
    assert scored["score"].notna().any()
    assert "suppressed_by_transition" in (out_dir / "monitor_notes.md").read_text()
