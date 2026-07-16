"""Tests for `scripts/monitor.py` (Step-2 package-6 Task 2, design spec
`docs/superpowers/specs/2026-07-16-step2-package6-runtime-pillar3-design.md` D2 +
amendment A1.3): CLI-level tests against a monkeypatched `discover`/`load_config`/
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

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
]
"""The plan's exact alarms.parquet column contract (order included)."""


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
    calibration-side windows and must take the `no_conformal_data` path (A1.3)."""
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
# 1. Recalibrate mode end to end (the DEFAULT mode -- plan Task 2 test 1)
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

    scored = alarms[alarms["role"] == "scored"]
    assert set(scored["state"].unique().tolist()) == set(snapshot.thresholds)
    assert ((scored["p_value"] > 0.0) & (scored["p_value"] <= 1.0)).all()
    # Same-distribution sanity: realized alarm rate per state stays near alpha.
    for _state, group in scored.groupby("state"):
        assert group["alarm"].mean() <= 3 * snapshot.alpha

    # Calibration-bias rule (A1.3): consumed windows are NEVER alarmed, no p-value.
    consumed = alarms[alarms["role"] == "consumed_for_calibration"]
    assert len(consumed) > 0
    assert not consumed["alarm"].any()
    assert consumed["p_value"].isna().all()

    notes = (out_dir / "monitor_notes.md").read_text()
    assert "recalibrate" in notes
    assert "did NOT hold" not in notes  # the frozen-mode warning must not leak here
    assert "no fault labels" in notes.lower()  # spec §4 honesty framing
    assert "candidate" in notes.lower()
    assert snapshot.fit_run in notes

    # State half mirrors apply_detector's conventions, on the MONITOR grid.
    segments = pd.read_csv(out_dir / "segments.csv", parse_dates=["start_utc", "end_utc"])
    assert list(segments.columns) == ["start_utc", "end_utc", "duration_s", "cluster_id"]
    assert segments["start_utc"].min() == pd.Timestamp(_MON_T0_NS, unit="ns", tz="UTC")
    assert set(segments["cluster_id"].unique().tolist()) <= set(snapshot.thresholds) | {-1}


# ---------------------------------------------------------------------------
# 2. Frozen mode: verdict for EVERY valid known-state window + the package-2
#    distribution-shift warning (plan Task 2 test 2)
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

    notes = (out_dir / "monitor_notes.md").read_text()
    assert "frozen" in notes
    assert "did NOT hold" in notes  # the package-2 warning, verbatim phrase


# ---------------------------------------------------------------------------
# 3. Geometry guard: mismatched prepared features -> exit 2 naming both widths
# ---------------------------------------------------------------------------


def test_geometry_mismatch_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import monitor

    snapshot_path, snapshot = _make_snapshot(tmp_path)  # fitted on 2 feature columns
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1, n_features=4)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out"

    exit_code = monitor.main(
        ["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN, "--out", str(out_dir)]
    )
    assert exit_code == 2

    err = capsys.readouterr().err
    assert str(len(snapshot.feature_names)) in err  # snapshot width (2)
    assert str(len(mon_prepared.feature_names)) in err  # prepared width (4)
    assert "fusion" in err  # the snapshot's variant, named
    assert not out_dir.exists()  # refused BEFORE writing anything


# ---------------------------------------------------------------------------
# 4. Unknown detected state (absent from the snapshot) -> role="unknown_state",
#    counted in the notes (plan Task 2 test 4: force it by stripping one label)
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
#    ALL its windows role="no_conformal_data", no verdicts (A1.3 binding rule)
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
# 8. --help documents every flag (plan Task 2 test 6)
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
# T2-review hardening: --alpha coverage (mutation gap) + CLI exit-2 paths
# ---------------------------------------------------------------------------


def test_alpha_override_changes_recalibrated_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kills the T2 review's surviving mutant (hardcoded alpha in the
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
