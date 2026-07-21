"""Tests for `scripts/run_step2.py --protocol cross-day-pooled` (Step-2 package-7
Task 3, design spec `docs/superpowers/specs/2026-07-18-step2-package7-robustness-
design.md` D2 + A3.1/A3.7/A3.8 + A4.1/A4.2/A4.5): held-out-day-group evaluation of
pooled references under BOTH threshold modes in one invocation.

Style-2 fixtures throughout (no synthetic Gantner trees): `run_step2.discover` and
`run_step2.prepare_run` are monkeypatched to hand-built `Run`s/`PreparedRun`s, the
same seam `test_step2_cli.py`'s xattn test uses. The three synthetic runs follow
`tests/test_detect_pooled.py`'s disjoint-mode blob fixture -- fit-a carries blobs
1+2, fit-b blobs 2+3, the held-out test-c all three -- so the pooled detector
(k=3) must recover every mode and every run's labels live in ONE pooled id space.

Fixture sizing note (verified empirically before hardcoding, the
`test_runtime_snapshot.py` practice): at `SweepConfig` defaults (top split 0.5/seed
7, nested 0.5/seed 8), 10 segments per fit run and 9 for the test run give EVERY
pooled cluster id >= 30 pooled fit-side rows (>= min_ref 20), >= 30 pooled
conformal-side rows, >= 30 test-run calibration-side rows, and >= 30 test-run
scoring-side rows (probe counts: id 0: 30/30/60/30, id 1: 90/60/60/30,
id 2: 60/30/30/60), with exact blob recovery (ARI 1.0) on all three runs.

The bitwise threshold tests re-derive the full expected pipeline INDEPENDENTLY
(build_pool + fit_pooled + apply + hand-run `calibrate`), mirroring
`tests/test_pools.py`'s hand-run split-parity stance; far tables are read back with
`float_precision="round_trip"` so "bitwise" survives the CSV round trip honestly.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from rowii.anomaly.conformal import ConformalThreshold, calibrate
from rowii.anomaly.pools import PoolResult, build_pool
from rowii.anomaly.references import split_by_segments
from rowii.anomaly.scorers import KnnScorer
from rowii.anomaly.sweep import SweepConfig
from rowii.config import Config
from rowii.io.dataset import BurstFile, RecordingIndex, Run
from rowii.pipeline import PreparedRun
from rowii.runtime.snapshot import load_snapshot, to_detector
from rowii.signals.windows import WindowGrid
from rowii.state.detect import FittedDetector

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Synthetic runs (blob fixture, sizing note in the module docstring)
# ---------------------------------------------------------------------------

_SEG_LEN = 30
_FIT_SEGMENTS = 10
_TEST_SEGMENTS = 9
_K = 3
_CENTERS: dict[int, tuple[float, float]] = {
    1: (0.0, 0.0),
    2: (12.0, 12.0),
    3: (-12.0, 18.0),
}

_FIT_RUN_NAMES = ("fit-a", "fit-b")
_TEST_RUN_NAME = "test-c"

_BASE_ARGS = [
    "--protocol", "cross-day-pooled", "--fit-runs", "fit-a,fit-b",
    "--test-run", "test-c", "--variant", "fusion", "--scorer", "knn", "--k", str(_K),
]


def _blob_run(blob_cycle: list[int], n_segments: int, seed: int) -> PreparedRun:
    """Contiguous `_SEG_LEN`-window segments, one blob per segment, cycling
    *blob_cycle* -- `tests/test_detect_pooled.py`'s layout with per-segment ids so
    `split_by_segments` has real segments to draw."""
    rng = np.random.default_rng(seed)
    feats: list[np.ndarray] = []
    seg_ids: list[np.ndarray] = []
    for s in range(n_segments):
        blob = blob_cycle[s % len(blob_cycle)]
        feats.append(rng.normal(_CENTERS[blob], 0.3, (_SEG_LEN, 2)))
        seg_ids.append(np.full(_SEG_LEN, s, dtype=np.int64))
    features = np.vstack(feats)
    n = len(features)
    return PreparedRun(
        features=features,
        grid=WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=n),
        valid_mask=np.ones(n, dtype=bool),
        feature_names=["f0", "f1"],
        segment_ids=np.concatenate(seg_ids),
    )


def _pooled_prepared() -> dict[str, PreparedRun]:
    return {
        "fit-a": _blob_run([1, 2], _FIT_SEGMENTS, seed=0),
        "fit-b": _blob_run([2, 3], _FIT_SEGMENTS, seed=1),
        "test-c": _blob_run([1, 2, 3], _TEST_SEGMENTS, seed=2),
    }


def _fake_run(name: str, date: str) -> Run:
    """A discovery-shaped `Run` whose single burst file's NAME carries *date* --
    exactly what the A3.8 day-group guard parses."""
    return Run(
        name=name,
        files={
            "RAWGeneratorMic__0": [
                BurstFile(
                    path=Path(f"/fake/{name}/RAWGeneratorMic__0_{date}_06-00-00_000000.dat"),
                    stream="RAWGeneratorMic__0",
                    start_utc_hint=datetime.fromisoformat(f"{date}T06:00:00+00:00"),
                )
            ]
        },
        day_root=Path(f"/fake/{name}"),
    )


def _fake_index(runs: list[Run]) -> RecordingIndex:
    """No Betriebsdaten anywhere -- the GT state x load-bin coverage overlay must
    take its skip-with-log path, never crash."""
    return RecordingIndex(runs=runs, betriebsdaten=[], betriebsdaten_by_day={})


def _default_index() -> RecordingIndex:
    return _fake_index(
        [
            _fake_run("fit-a", "2026-06-29"),
            _fake_run("fit-b", "2026-07-01"),
            _fake_run("test-c", "2026-07-03"),
        ]
    )


def _install_fakes(monkeypatch, tmp_path, prepared: dict[str, PreparedRun],
                   index: RecordingIndex) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))

    import run_step2

    monkeypatch.setattr(run_step2, "discover", lambda root: index)

    def fake_prepare(run, variant, cfg, use_cache=True):
        assert variant == "fusion"
        return prepared[run.name]

    monkeypatch.setattr(run_step2, "prepare_run", fake_prepare)


def _out_dir(tmp_path) -> Path:
    return (
        tmp_path / "results" / "step2" / "cross-day-pooled" / _TEST_RUN_NAME
        / "fusion-pooled"
    )


def _read_far(path: Path) -> pd.DataFrame:
    """`float_precision="round_trip"` so bitwise threshold comparisons survive the
    CSV round trip (module docstring)."""
    return pd.read_csv(path, float_precision="round_trip")


# ---------------------------------------------------------------------------
# Hand-run mirror of the pooled pipeline (independent expected values)
# ---------------------------------------------------------------------------


class _HandRun(NamedTuple):
    detector: FittedDetector
    labels: dict[str, np.ndarray]
    pool_fit: PoolResult
    pool_conformal: PoolResult
    fit_labels: np.ndarray
    conformal_labels: np.ndarray
    cal_windows: np.ndarray
    scoring_windows: np.ndarray


def _pool_row_labels(pool: PoolResult, labels: dict[str, np.ndarray]) -> np.ndarray:
    out = np.empty(pool.features.shape[0], dtype=np.int64)
    for i, member in enumerate(pool.members):
        mask = pool.run_index == i
        out[mask] = labels[member.run_name][pool.window_index[mask]]
    return out


def _hand_pipeline(prepared: dict[str, PreparedRun]) -> _HandRun:
    sweep_cfg = SweepConfig()
    prepared_fit = {name: prepared[name] for name in _FIT_RUN_NAMES}
    pool_fit = build_pool(prepared_fit, "fit", sweep_cfg)
    pool_conformal = build_pool(prepared_fit, "conformal", sweep_cfg)
    rowii_cfg = Config(data_root=Path("/unused"), results_root=Path("/unused"))
    detector = FittedDetector.fit_pooled(pool_fit.features, rowii_cfg, k=_K)
    labels = {
        name: detector.apply(p.features, p.grid).frame_labels
        for name, p in prepared.items()
    }
    test = prepared[_TEST_RUN_NAME]
    top = split_by_segments(
        test.segment_ids, test.valid_mask, sweep_cfg.calibration_frac, sweep_cfg.seed
    )
    return _HandRun(
        detector=detector,
        labels=labels,
        pool_fit=pool_fit,
        pool_conformal=pool_conformal,
        fit_labels=_pool_row_labels(pool_fit, labels),
        conformal_labels=_pool_row_labels(pool_conformal, labels),
        cal_windows=top.calibration_windows,
        scoring_windows=top.scoring_windows,
    )


def _expected_mode_thresholds(
    hand: _HandRun, prepared: dict[str, PreparedRun], alpha: float
) -> tuple[dict[int, ConformalThreshold], dict[int, ConformalThreshold], dict[int, int]]:
    """(frozen, recalibrate, n_scoring_alarms_frozen) per pooled label id --
    frozen from the pool's CONFORMAL side (A3.7), recalibrate from the test run's
    own calibration side."""
    test = prepared[_TEST_RUN_NAME]
    lab_test = hand.labels[_TEST_RUN_NAME]
    frozen: dict[int, ConformalThreshold] = {}
    recal: dict[int, ConformalThreshold] = {}
    frozen_alarms: dict[int, int] = {}
    for lid in range(_K):
        reference = hand.pool_fit.features[hand.fit_labels == lid]
        scorer = KnnScorer().fit(reference)
        conf_scores = scorer.score(hand.pool_conformal.features[hand.conformal_labels == lid])
        frozen[lid] = calibrate(conf_scores, alpha)
        label_cal = hand.cal_windows[lab_test[hand.cal_windows] == lid]
        recal[lid] = calibrate(scorer.score(test.features[label_cal]), alpha)
        label_scr = hand.scoring_windows[lab_test[hand.scoring_windows] == lid]
        frozen_alarms[lid] = int(
            (scorer.score(test.features[label_scr]) > frozen[lid].threshold).sum()
        )
    return frozen, recal, frozen_alarms


# ---------------------------------------------------------------------------
# 1. --help mentions the new flags
# ---------------------------------------------------------------------------


def test_help_mentions_pooled_flags(capsys) -> None:
    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--fit-runs", "--test-run", "--k", "--save-snapshot"):
        assert flag in out, f"missing {flag!r} in --help output"
    assert "cross-day-pooled" in out


# ---------------------------------------------------------------------------
# 2. End-to-end: exit 0, both FAR tables with the contracted schema
# ---------------------------------------------------------------------------


def test_end_to_end_writes_both_far_tables(tmp_path, monkeypatch) -> None:
    prepared = _pooled_prepared()
    _install_fakes(monkeypatch, tmp_path, prepared, _default_index())

    import run_step2

    exit_code = run_step2.main(_BASE_ARGS)
    assert exit_code == 0

    out_dir = _out_dir(tmp_path)
    for filename in ("far_table_frozen.csv", "far_table_recalibrate.csv"):
        path = out_dir / filename
        assert path.is_file(), f"missing {path}"
        table = _read_far(path)
        assert list(table.columns) == list(run_step2._FAR_TABLE_COLUMNS)
        # One row per pooled label id (all three modes survive the min_ref floor,
        # sizing note) + the run_sweep-style aggregate row, aggregate LAST.
        assert list(table["label"]) == ["0", "1", "2", "pooled"]
        per_label = table[table["label"] != "pooled"]
        assert not per_label["excluded"].any()
        assert (per_label["n_scored"] > 0).all()
        assert (per_label["nominal_alpha"] == 0.05).all()


# ---------------------------------------------------------------------------
# 3. FROZEN thresholds are bitwise calibrate(pool CONFORMAL scores, alpha) (A3.7)
# ---------------------------------------------------------------------------


def test_frozen_thresholds_bitwise_from_pool_conformal_side(tmp_path, monkeypatch) -> None:
    prepared = _pooled_prepared()
    _install_fakes(monkeypatch, tmp_path, prepared, _default_index())

    import run_step2

    assert run_step2.main(_BASE_ARGS) == 0

    hand = _hand_pipeline(prepared)
    frozen, _recal, frozen_alarms = _expected_mode_thresholds(hand, prepared, alpha=0.05)

    table = _read_far(_out_dir(tmp_path) / "far_table_frozen.csv").set_index("label")
    for lid in range(_K):
        row = table.loc[str(lid)]
        expected = frozen[lid]
        assert float(row["threshold"]) == expected.threshold
        assert int(row["n_calibration"]) == expected.n_calibration
        assert float(row["achievable_alpha_floor"]) == expected.achievable_alpha_floor
        assert int(row["n_alarms"]) == frozen_alarms[lid]
        lab_test = hand.labels[_TEST_RUN_NAME]
        n_scored = int((lab_test[hand.scoring_windows] == lid).sum())
        assert int(row["n_scored"]) == n_scored


# ---------------------------------------------------------------------------
# 4. RECALIBRATE thresholds are bitwise calibrate(test calibration side, alpha)
# ---------------------------------------------------------------------------


def test_recalibrate_thresholds_bitwise_from_test_calibration_side(
    tmp_path, monkeypatch
) -> None:
    prepared = _pooled_prepared()
    _install_fakes(monkeypatch, tmp_path, prepared, _default_index())

    import run_step2

    assert run_step2.main(_BASE_ARGS) == 0

    hand = _hand_pipeline(prepared)
    _frozen, recal, _ = _expected_mode_thresholds(hand, prepared, alpha=0.05)

    table = _read_far(_out_dir(tmp_path) / "far_table_recalibrate.csv").set_index("label")
    for lid in range(_K):
        row = table.loc[str(lid)]
        expected = recal[lid]
        assert float(row["threshold"]) == expected.threshold
        assert int(row["n_calibration"]) == expected.n_calibration
        # Both modes score the SAME test scoring side -- only thresholds differ.
        lab_test = hand.labels[_TEST_RUN_NAME]
        assert int(row["n_scored"]) == int((lab_test[hand.scoring_windows] == lid).sum())


# ---------------------------------------------------------------------------
# 5. A3.1 guard: the test run must never be a pool member
# ---------------------------------------------------------------------------


def test_pool_member_as_test_run_rejected(tmp_path, monkeypatch, capsys) -> None:
    _install_fakes(monkeypatch, tmp_path, _pooled_prepared(), _default_index())

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(
            [
                "--protocol", "cross-day-pooled", "--fit-runs", "fit-a,test-c",
                "--test-run", "test-c", "--variant", "fusion",
            ]
        )
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "A3.1" in err
    assert "test-c" in err


# ---------------------------------------------------------------------------
# 6. A3.8 guard: fit and test day groups must be disjoint (sibling-run case)
# ---------------------------------------------------------------------------


def test_same_day_group_rejected_for_sibling_runs(tmp_path, monkeypatch, capsys) -> None:
    """`010726-tu1` and `010726-tu2` style siblings share ONE calendar day
    (2026-07-01 in both burst-file names) -- a rotation that fits on one and tests
    on the other is not held-out-day-group evaluation and must exit 2."""
    prepared = {
        "010726-tu1": _blob_run([1, 2], _FIT_SEGMENTS, seed=0),
        "010726-tu2": _blob_run([1, 2], _FIT_SEGMENTS, seed=1),
    }
    index = _fake_index(
        [_fake_run("010726-tu1", "2026-07-01"), _fake_run("010726-tu2", "2026-07-01")]
    )
    _install_fakes(monkeypatch, tmp_path, prepared, index)

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(
            [
                "--protocol", "cross-day-pooled", "--fit-runs", "010726-tu1",
                "--test-run", "010726-tu2", "--variant", "fusion",
            ]
        )
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "2026-07-01" in err
    assert "010726-tu1" in err
    assert "day group" in err


# ---------------------------------------------------------------------------
# 7. Unknown run names are listed (exit 2, warm_cache precedent)
# ---------------------------------------------------------------------------


def test_unknown_run_names_listed(tmp_path, monkeypatch, capsys) -> None:
    _install_fakes(monkeypatch, tmp_path, _pooled_prepared(), _default_index())

    import run_step2

    exit_code = run_step2.main(
        [
            "--protocol", "cross-day-pooled", "--fit-runs", "fit-a,nope",
            "--test-run", "test-c", "--variant", "fusion",
        ]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "nope" in err
    assert "fit-b" in err  # the available-runs listing


# ---------------------------------------------------------------------------
# 8. k too large -> exit 2 with a clear message
# ---------------------------------------------------------------------------


def test_k_too_large_exits_2(tmp_path, monkeypatch, capsys) -> None:
    _install_fakes(monkeypatch, tmp_path, _pooled_prepared(), _default_index())

    import run_step2

    args = [a for a in _BASE_ARGS]
    args[args.index("--k") + 1] = "9999"
    exit_code = run_step2.main(args)
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "k too large" in err


# ---------------------------------------------------------------------------
# 9. Coverage tables (A4.1/A4.2) + notes sections
# ---------------------------------------------------------------------------


def test_coverage_tables_and_notes_written(tmp_path, monkeypatch) -> None:
    prepared = _pooled_prepared()
    _install_fakes(monkeypatch, tmp_path, prepared, _default_index())

    import run_step2

    assert run_step2.main(_BASE_ARGS) == 0
    out_dir = _out_dir(tmp_path)

    train = pd.read_csv(out_dir / "coverage_train.csv")
    assert list(train.columns) == ["run", "label", "n_windows"]
    assert set(train["run"]) == set(_FIT_RUN_NAMES)
    assert set(train["label"]) == {0, 1, 2}

    eval_table = pd.read_csv(out_dir / "coverage_eval.csv")
    assert list(eval_table.columns) == ["run", "label", "n_windows"]
    assert set(eval_table["run"]) == {_TEST_RUN_NAME}
    hand = _hand_pipeline(prepared)
    assert int(eval_table["n_windows"].sum()) == int(hand.scoring_windows.shape[0])

    notes = (out_dir / "notes.md").read_text()
    assert "held-out-day-group" in notes
    for name in (*_FIT_RUN_NAMES, _TEST_RUN_NAME):
        assert name in notes
    assert "frozen" in notes
    assert "recalibrate" in notes
    assert "A3.1" in notes
    # A4.5 estimator-vs-final framing.
    assert "estimator" in notes.lower()
    assert "final" in notes.lower()
    # No SCADA on the synthetic runs and full detected-label coverage -> the
    # coverage-warning section reports the "(none)" sentinel (and the GT overlay
    # was skipped).
    assert "(none)" in notes
    assert not (out_dir / "coverage_train_gt.csv").exists()
    assert not (out_dir / "coverage_eval_gt.csv").exists()


def test_notes_surface_coverage_warnings_verbatim() -> None:
    """Seam test for the warning plumbing: detected-label warnings cannot fire on
    this fixture (the pooled detector's id space is covered by construction), so the
    notes builder is asserted directly with a crafted A4.2 composite-label warning."""
    import run_step2

    warning = (
        "coverage: label 'pump|2' has 7 evaluation window(s) but zero training coverage"
    )
    notes = run_step2._cross_day_pooled_notes(
        ["fit-a", "fit-b"],
        "test-c",
        "fusion",
        "knn",
        0.05,
        3,
        {
            "calibration": {"fit-a": {"n_windows": 12}, "fit-b": {"n_windows": 10}},
            "fit": {"fit-a": {"n_windows": 6}, "fit-b": {"n_windows": 5}},
            "conformal": {"fit-a": {"n_windows": 6}, "fit-b": {"n_windows": 5}},
        },
        [warning],
    )
    assert warning in notes
    assert "(none)" not in notes


# ---------------------------------------------------------------------------
# 10. --save-snapshot: fit_snapshot_from_parts round trip + provenance sidecar
# ---------------------------------------------------------------------------


def test_save_snapshot_round_trip_geometry_and_provenance(tmp_path, monkeypatch) -> None:
    prepared = _pooled_prepared()
    _install_fakes(monkeypatch, tmp_path, prepared, _default_index())

    import run_step2

    snap_path = tmp_path / "snapshots" / "pool_snapshot.npz"
    assert run_step2.main([*_BASE_ARGS, "--save-snapshot", str(snap_path)]) == 0
    assert snap_path.is_file()

    loaded = load_snapshot(snap_path)
    assert loaded.fit_run == "pool:fit-a,fit-b"
    assert loaded.variant == "fusion"
    assert loaded.feature_names == ["f0", "f1"]
    assert loaded.mean.shape == (2,)
    assert set(loaded.thresholds) == {0, 1, 2}

    hand = _hand_pipeline(prepared)
    frozen, _recal, _ = _expected_mode_thresholds(hand, prepared, alpha=0.05)
    for lid in range(_K):
        # The snapshot stores the FROZEN pool-conformal thresholds (A3.7) and the
        # pooled fit-side references, bitwise.
        assert loaded.thresholds[lid] == frozen[lid]
        np.testing.assert_array_equal(
            loaded.references[lid], hand.pool_fit.features[hand.fit_labels == lid]
        )

    # Detector round trip: the rebuilt detector labels the held-out run exactly
    # like the pooled detector the CLI fit.
    test = prepared[_TEST_RUN_NAME]
    np.testing.assert_array_equal(
        to_detector(loaded).apply(test.features, test.grid).frame_labels,
        hand.labels[_TEST_RUN_NAME],
    )

    sidecar = json.loads(snap_path.with_suffix(".json").read_text())
    assert sidecar["provenance"]["fit_runs"] == ["fit-a", "fit-b"]
    assert sidecar["provenance"]["held_out_test_run"] == "test-c"
    assert sidecar["provenance"]["k"] == _K
    assert "pool_members" in sidecar["provenance"]


# ---------------------------------------------------------------------------
# 11. --alpha override flows into both tables
# ---------------------------------------------------------------------------


def test_alpha_override_flows_into_both_tables(tmp_path, monkeypatch) -> None:
    prepared = _pooled_prepared()
    _install_fakes(monkeypatch, tmp_path, prepared, _default_index())

    import run_step2

    assert run_step2.main([*_BASE_ARGS, "--alpha", "0.1"]) == 0

    hand = _hand_pipeline(prepared)
    frozen, recal, _ = _expected_mode_thresholds(hand, prepared, alpha=0.1)
    out_dir = _out_dir(tmp_path)
    for filename, expected in (
        ("far_table_frozen.csv", frozen),
        ("far_table_recalibrate.csv", recal),
    ):
        table = _read_far(out_dir / filename)
        assert (table["nominal_alpha"] == 0.1).all()
        by_label = table.set_index("label")
        for lid in range(_K):
            assert float(by_label.loc[str(lid)]["threshold"]) == expected[lid].threshold


# ---------------------------------------------------------------------------
# 12. Flag/protocol guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        ["--fit-runs", "fit-a"],
        ["--test-run", "test-c"],
        ["--k", "3"],
        ["--save-snapshot", "snap.npz"],
    ],
)
def test_pooled_flags_require_cross_day_pooled_protocol(
    tmp_path, monkeypatch, capsys, extra
) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    (tmp_path / "data").mkdir()

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(["--protocol", "within-day", "--variant", "fusion", *extra])
    assert exc_info.value.code == 2
    assert "cross-day-pooled" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [
        ["--fit-runs", "fit-a,fit-b"],  # missing --test-run
        ["--test-run", "test-c"],  # missing --fit-runs
    ],
)
def test_cross_day_pooled_requires_fit_runs_and_test_run(
    tmp_path, monkeypatch, capsys, args
) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    (tmp_path / "data").mkdir()

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(
            ["--protocol", "cross-day-pooled", "--variant", "fusion", *args]
        )
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--fit-runs" in err
    assert "--test-run" in err


def test_gt_labels_and_scorer_all_rejected(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    (tmp_path / "data").mkdir()

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main([*_BASE_ARGS, "--labels", "gt"])
    assert exc_info.value.code == 2
    assert "detected-labels only" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main([*_BASE_ARGS, "--scorer", "all"])
    assert exc_info.value.code == 2
    assert "one scorer" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# T3-review hardening: midnight-crossing day groups + untested guard branches
# ---------------------------------------------------------------------------


def _fake_run_two_dates(name: str, date_a: str, date_b: str) -> Run:
    """A discovery-shaped run whose burst files span TWO calendar dates -- the
    midnight-crossing case the T3 review proved bypassed a first-file-only day
    group (real near-miss: 010726-tu_ph_tu's last file starts 23:57 local)."""
    return Run(
        name=name,
        files={
            "RAWGeneratorMic__0": [
                BurstFile(
                    path=Path(
                        f"/fake/{name}/RAWGeneratorMic__0_{date_a}_23-58-00_000000.dat"
                    ),
                    stream="RAWGeneratorMic__0",
                    start_utc_hint=datetime.fromisoformat(f"{date_a}T23:58:00+00:00"),
                ),
                BurstFile(
                    path=Path(
                        f"/fake/{name}/RAWGeneratorMic__0_{date_b}_00-10-00_000000.dat"
                    ),
                    stream="RAWGeneratorMic__0",
                    start_utc_hint=datetime.fromisoformat(f"{date_b}T00:10:00+00:00"),
                ),
            ]
        },
        day_root=Path(f"/fake/{name}"),
    )


def test_midnight_crossing_fit_run_shares_test_day_rejected(
    tmp_path, monkeypatch, capsys
) -> None:
    """The overnight run's TAIL date collides with the test run's day -- the
    date-SET guard must refuse (a first-file-only group would have passed)."""
    overnight = _fake_run_two_dates("fit-night", "2026-07-01", "2026-07-02")
    fit_b = _fake_run("fit-b", "2026-06-29")
    test_c = _fake_run("test-c", "2026-07-02")
    _install_fakes(
        monkeypatch, tmp_path, _pooled_prepared(),
        _fake_index([overnight, fit_b, test_c]),
    )

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(
            [
                "--protocol", "cross-day-pooled", "--fit-runs", "fit-night,fit-b",
                "--test-run", "test-c", "--variant", "fusion",
            ]
        )
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "fit-night" in err
    assert "2026-07-02" in err


def test_duplicate_fit_runs_exit_2(tmp_path, monkeypatch, capsys) -> None:
    _install_fakes(monkeypatch, tmp_path, _pooled_prepared(), _default_index())

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(
            [
                "--protocol", "cross-day-pooled", "--fit-runs", "fit-a,fit-a",
                "--test-run", "test-c", "--variant", "fusion",
            ]
        )
    assert exc_info.value.code == 2
    assert "duplicate" in capsys.readouterr().err


def test_fit_pooled_runtime_error_exits_2_not_traceback(
    tmp_path, monkeypatch, capsys
) -> None:
    """T3-review mutation probe: narrowing the except clause to ValueError let
    fit_pooled's own RuntimeError (near-constant pool, unassigned cluster ids)
    escape as a traceback. Pin the RuntimeError branch through the CLI."""
    _install_fakes(monkeypatch, tmp_path, _pooled_prepared(), _default_index())

    import run_step2

    from rowii.state.detect import FittedDetector

    def _raise_runtime(*args: object, **kwargs: object) -> object:
        raise RuntimeError("pooled KMeans did not assign every cluster id 0..4")

    monkeypatch.setattr(FittedDetector, "fit_pooled", classmethod(
        lambda cls, *a, **k: _raise_runtime()
    ))

    exit_code = run_step2.main(
        [
            "--protocol", "cross-day-pooled", "--fit-runs", "fit-a,fit-b",
            "--test-run", "test-c", "--variant", "fusion", "--k", "5",
        ]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "k too large" in err
    assert "0..4" in err


def test_save_snapshot_with_non_runtime_scorer_exits_2(
    tmp_path, monkeypatch, capsys
) -> None:
    _install_fakes(monkeypatch, tmp_path, _pooled_prepared(), _default_index())

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(
            [
                "--protocol", "cross-day-pooled", "--fit-runs", "fit-a,fit-b",
                "--test-run", "test-c", "--variant", "fusion",
                "--scorer", "ocsvm", "--save-snapshot", str(tmp_path / "s.npz"),
            ]
        )
    assert exc_info.value.code == 2
    assert "runtime" in capsys.readouterr().err.lower()


def test_k_below_one_exits_2(tmp_path, monkeypatch, capsys) -> None:
    _install_fakes(monkeypatch, tmp_path, _pooled_prepared(), _default_index())

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(
            [
                "--protocol", "cross-day-pooled", "--fit-runs", "fit-a,fit-b",
                "--test-run", "test-c", "--variant", "fusion", "--k", "0",
            ]
        )
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# T4: --session-norm / --norm-minutes (package-7 Task 4, spec D3/A3.5) --
#     cross-day-pooled wiring ONLY (within-day/cross-day wiring is deferred)
# ---------------------------------------------------------------------------

_SNORM_ARGS = [*_BASE_ARGS, "--session-norm"]


def _snorm_out_dir(tmp_path, minutes: str = "20") -> Path:
    """Session-norm runs land in a `-snorm<N>` suffixed leaf so the A2.2 N-sweep
    never overwrites the un-normed baseline (or another N's outputs)."""
    return (
        tmp_path / "results" / "step2" / "cross-day-pooled" / _TEST_RUN_NAME
        / f"fusion-pooled-snorm{minutes}"
    )


def test_session_norm_smoke_tables_and_thresholds_differ(tmp_path, monkeypatch) -> None:
    prepared = _pooled_prepared()
    _install_fakes(monkeypatch, tmp_path, prepared, _default_index())

    import run_step2

    assert run_step2.main(_BASE_ARGS) == 0  # un-normed baseline first
    assert run_step2.main(_SNORM_ARGS) == 0

    out_dir = _snorm_out_dir(tmp_path)
    for filename in ("far_table_frozen.csv", "far_table_recalibrate.csv"):
        path = out_dir / filename
        assert path.is_file(), f"missing {path}"
        table = _read_far(path)
        assert list(table.columns) == list(run_step2._FAR_TABLE_COLUMNS)
        assert list(table["label"]) == ["0", "1", "2", "pooled"]
        per_label = table[table["label"] != "pooled"]
        assert not per_label["excluded"].any()
        assert (per_label["n_scored"] > 0).all()

    # Scoring in the session-normalized space must actually CHANGE the thresholds
    # vs the un-normed baseline (same labels, same window sets -- different space).
    for filename in ("far_table_frozen.csv", "far_table_recalibrate.csv"):
        base = _read_far(_out_dir(tmp_path) / filename).set_index("label")
        norm = _read_far(out_dir / filename).set_index("label")
        base_thr = base.loc[[str(lid) for lid in range(_K)], "threshold"].to_numpy()
        norm_thr = norm.loc[[str(lid) for lid in range(_K)], "threshold"].to_numpy()
        assert not np.allclose(base_thr, norm_thr), filename

    notes = (out_dir / "notes.md").read_text()
    assert "session" in notes.lower()
    assert "deferred" in notes.lower()  # within-day/cross-day wiring deferral, documented
    assert "confound" in notes.lower()  # the A3.5 state-mix caveat travels with results


def test_session_norm_uses_per_run_stats(tmp_path, monkeypatch) -> None:
    """Per-run stats are BINDING (Task 4: "per-run stats for pool members!") -- the
    CLI must fit session stats exactly once per run (2 fit runs + the test run),
    each on that run's OWN feature matrix, at the requested norm minutes."""
    prepared = _pooled_prepared()
    _install_fakes(monkeypatch, tmp_path, prepared, _default_index())

    import run_step2

    calls: list[tuple[np.ndarray, float]] = []
    real_fit = run_step2.fit_session_stats

    def spy(features, valid_mask, grid, *, norm_minutes):
        calls.append((features, norm_minutes))
        return real_fit(features, valid_mask, grid, norm_minutes=norm_minutes)

    monkeypatch.setattr(run_step2, "fit_session_stats", spy)

    assert run_step2.main([*_SNORM_ARGS, "--norm-minutes", "5"]) == 0
    assert len(calls) == 3  # fit-a, fit-b, test-c -- once each
    assert {minutes for _, minutes in calls} == {5.0}
    for name in (*_FIT_RUN_NAMES, _TEST_RUN_NAME):
        assert any(feats is prepared[name].features for feats, _ in calls), name
    # The N goes into the leaf name too (N-sweep separability).
    assert (_snorm_out_dir(tmp_path, "5") / "far_table_frozen.csv").is_file()


def test_session_norm_snapshot_stores_pool_global_stats_and_raw_references(
    tmp_path, monkeypatch
) -> None:
    """Task 4's pooled-snapshot design decision: references stay RAW (the
    MonitorSnapshot field contract), `session_stats` = pool-global median/MAD over
    the RAW pooled fit matrix (`norm_minutes == 0.0` sentinel), and the stored
    conformal scores/thresholds are SELF-CONSISTENT in the exact space the monitor
    reconstructs (scorer on stats-transformed references, stats-transformed
    conformal rows) -- deliberately NOT far_table_frozen.csv's per-run-normalized
    thresholds (FAR-level-only comparability, A3.5)."""
    prepared = _pooled_prepared()
    _install_fakes(monkeypatch, tmp_path, prepared, _default_index())

    import run_step2

    from rowii.anomaly.normalize import apply_session_norm

    snap_path = tmp_path / "snapshots" / "pool_snorm.npz"
    assert run_step2.main([*_SNORM_ARGS, "--save-snapshot", str(snap_path)]) == 0

    loaded = load_snapshot(snap_path)
    stats = loaded.session_stats
    assert stats is not None
    assert stats.norm_minutes == 0.0  # pool-global sentinel

    hand = _hand_pipeline(prepared)
    center = np.median(hand.pool_fit.features, axis=0)
    mad = np.median(np.abs(hand.pool_fit.features - center), axis=0)
    scale = np.maximum(mad * 1.4826, 1e-8)
    np.testing.assert_array_equal(stats.center, center)
    np.testing.assert_array_equal(stats.scale, scale)
    assert stats.n_windows == hand.pool_fit.features.shape[0]

    for lid in range(_K):
        raw_reference = hand.pool_fit.features[hand.fit_labels == lid]
        np.testing.assert_array_equal(loaded.references[lid], raw_reference)
        # Self-consistency in the monitor-reconstructed space, bitwise.
        scorer = KnnScorer().fit(apply_session_norm(raw_reference, stats))
        conf_raw = hand.pool_conformal.features[hand.conformal_labels == lid]
        expected_scores = scorer.score(apply_session_norm(conf_raw, stats))
        np.testing.assert_array_equal(loaded.calibration_scores[lid], expected_scores)
        assert loaded.thresholds[lid] == calibrate(expected_scores, 0.05)

    sidecar = json.loads(snap_path.with_suffix(".json").read_text())
    assert sidecar["provenance"]["session_norm"]["norm_minutes"] == 20.0
    assert sidecar["provenance"]["session_norm"]["pool_stats_n_windows"] == stats.n_windows


def test_norm_minutes_requires_session_norm(tmp_path, monkeypatch, capsys) -> None:
    _install_fakes(monkeypatch, tmp_path, _pooled_prepared(), _default_index())

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main([*_BASE_ARGS, "--norm-minutes", "5"])
    assert exc_info.value.code == 2
    assert "--session-norm" in capsys.readouterr().err


def test_norm_minutes_must_be_positive(tmp_path, monkeypatch, capsys) -> None:
    _install_fakes(monkeypatch, tmp_path, _pooled_prepared(), _default_index())

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main([*_SNORM_ARGS, "--norm-minutes", "0"])
    assert exc_info.value.code == 2
    assert "> 0" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [["--session-norm"], ["--session-norm", "--norm-minutes", "5"]],
)
def test_session_norm_requires_cross_day_pooled_protocol(
    tmp_path, monkeypatch, capsys, extra
) -> None:
    """Task 4 wires --session-norm into cross-day-pooled ONLY (the plan's
    within-day/cross-day wiring is DEFERRED) -- other protocols must refuse."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    (tmp_path / "data").mkdir()

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(["--protocol", "within-day", "--variant", "fusion", *extra])
    assert exc_info.value.code == 2
    assert "cross-day-pooled" in capsys.readouterr().err
