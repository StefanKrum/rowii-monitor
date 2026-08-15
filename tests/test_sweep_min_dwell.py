"""Tests for scripts/sweep_min_dwell.py (Package-9 D3b): the min_dwell->window-count
conversion (5/10/20; duration_filter no-op at min_dwell<=1) + a monkeypatched
fit_pooled/GT-seam CLI writing a state_ari-by-min_dwell table -- no real data.

Plan's own RED tests (verbatim) are `test_min_dwell_windows_conversion` and
`test_duration_filter_noop_at_min_dwell_one`. This file extends them with direct
unit tests of the D3b-only `_recalibrate_far` aggregation (the "plus one Step-2
chain FAR spot-check" the plan describes only in prose, `_min_dwell_verdict`/
`_sweep_table`'s pure shape, and the "monkeypatched CLI ARI table" the plan's own
Task 4 checkbox names but does not spell out in code -- an end-to-end `main()` run
on hand-built PreparedRuns (`tests/test_run_modebank.py`'s own `_prepared`/
`_install` seam shape, reused here for the SAME reason: `build_pool` +
`FittedDetector.fit_pooled` need real segment-blocked splits to exercise for real).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from rowii.anomaly.sweep import SweepConfig  # noqa: E402
from rowii.config import Config, DetectConfig  # noqa: E402
from rowii.io.dataset import RecordingIndex, Run  # noqa: E402
from rowii.pipeline import PreparedRun  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402

_W = 1_000_000_000
_NAMES = [f"ch0_octave_{i}" for i in range(4)]


# ---------------------------------------------------------------------------
# RED tests written directly against the design's own acceptance criteria
# ---------------------------------------------------------------------------


def test_min_dwell_windows_conversion() -> None:
    import sweep_min_dwell as sd
    assert sd._min_dwell_windows(5.0, 1.0) == 5
    assert sd._min_dwell_windows(10.0, 1.0) == 10
    assert sd._min_dwell_windows(20.0, 1.0) == 20
    assert sd._min_dwell_windows(0.5, 1.0) == 1  # floored at 1 -> duration_filter no-op


def test_duration_filter_noop_at_min_dwell_one() -> None:
    from rowii.state.segments import duration_filter
    labels = np.array([0, 1, 0, 0, 0], dtype=np.int64)
    np.testing.assert_array_equal(duration_filter(labels, min_dwell=1), labels)


# ---------------------------------------------------------------------------
# `_recalibrate_far`: the D3b "plus one Step-2 chain FAR spot-check" (spec
# §3.D3(b)) -- the trimmed, recalibrate-only duplication of
# scripts/run_step2.py::_cross_day_pooled_tables's recalibrate branch.
# ---------------------------------------------------------------------------


def test_recalibrate_far_aggregates_across_labels() -> None:
    import sweep_min_dwell as sd

    rng = np.random.default_rng(0)
    # Two well-separated pooled-fit labels (ample reference rows each).
    pool_fit_features = np.vstack(
        [rng.normal(0.0, 0.1, (40, 3)), rng.normal(10.0, 0.1, (40, 3))]
    )
    pool_fit_labels = np.array([0] * 40 + [1] * 40, dtype=np.int64)

    # Test run: label 0's windows cluster near 0, label 1's near 10 -- every
    # window interleaved calibration/scoring so both sides see both labels.
    n = 20
    test_features = np.vstack([rng.normal(0.0, 0.1, (n, 3)), rng.normal(10.0, 0.1, (n, 3))])
    labels_test = np.array([0] * n + [1] * n, dtype=np.int64)
    cal_windows = np.arange(0, 2 * n, 2, dtype=np.int64)
    scoring_windows = np.arange(1, 2 * n, 2, dtype=np.int64)

    sweep_cfg = SweepConfig(alpha=0.2, min_ref=10)
    far = sd._recalibrate_far(
        pool_fit_features, pool_fit_labels, test_features, labels_test,
        cal_windows, scoring_windows, sweep_cfg, "knn",
    )
    assert 0.0 <= far <= 1.0


def test_recalibrate_far_skips_labels_below_min_ref() -> None:
    """A pooled-fit label with < sweep_cfg.min_ref rows contributes NOTHING to
    the aggregate (mirrors far_row_excluded's "contributes nothing" rule) --
    here it is the ONLY label, so the aggregate has zero scored windows -> NaN."""
    import sweep_min_dwell as sd

    pool_fit_features = np.zeros((3, 2), dtype=np.float64)  # only 3 rows
    pool_fit_labels = np.zeros(3, dtype=np.int64)
    test_features = np.zeros((4, 2), dtype=np.float64)
    labels_test = np.zeros(4, dtype=np.int64)
    cal_windows = np.array([0, 1], dtype=np.int64)
    scoring_windows = np.array([2, 3], dtype=np.int64)

    sweep_cfg = SweepConfig(alpha=0.1, min_ref=10)  # 3 < 10 -> excluded
    far = sd._recalibrate_far(
        pool_fit_features, pool_fit_labels, test_features, labels_test,
        cal_windows, scoring_windows, sweep_cfg, "knn",
    )
    assert np.isnan(far)


def test_recalibrate_far_nan_when_nothing_scored() -> None:
    """A label with a real reference but zero test-calibration windows of its
    own contributes nothing either (mirrors far_row_no_conformal_data)."""
    import sweep_min_dwell as sd

    rng = np.random.default_rng(1)
    pool_fit_features = rng.normal(1.0, 0.1, (20, 2))
    pool_fit_labels = np.zeros(20, dtype=np.int64)
    test_features = rng.normal(1.0, 0.1, (4, 2))
    labels_test = np.zeros(4, dtype=np.int64)
    cal_windows = np.array([], dtype=np.int64)  # nothing to calibrate on
    scoring_windows = np.array([0, 1], dtype=np.int64)

    sweep_cfg = SweepConfig(alpha=0.1, min_ref=1)
    far = sd._recalibrate_far(
        pool_fit_features, pool_fit_labels, test_features, labels_test,
        cal_windows, scoring_windows, sweep_cfg, "knn",
    )
    assert np.isnan(far)


# ---------------------------------------------------------------------------
# `_sweep_table` / `_min_dwell_verdict`: pure shape + reporting helpers.
# ---------------------------------------------------------------------------


def test_sweep_table_shape_and_window_count() -> None:
    import sweep_min_dwell as sd

    table = sd._sweep_table(
        [5.0, 10.0], {5.0: 0.9, 10.0: 0.8}, {5.0: 0.1, 10.0: 0.05}, window_s=1.0,
    )
    assert list(table["min_dwell_s"]) == [5.0, 10.0]
    assert list(table["min_dwell_windows"]) == [5, 10]
    assert list(table["state_ari"]) == [0.9, 0.8]
    assert list(table["recalibrate_far"]) == [0.1, 0.05]


def test_min_dwell_verdict_names_the_best_and_the_default() -> None:
    import sweep_min_dwell as sd

    ari = {5.0: 0.5, 10.0: 0.9, 20.0: 0.7}
    text = sd._min_dwell_verdict(ari, current_default_s=5.0)
    assert "10" in text and "5" in text

    ari_default_best = {5.0: 0.95, 10.0: 0.5}
    text2 = sd._min_dwell_verdict(ari_default_best, current_default_s=5.0)
    assert "no change" in text2.lower()


# ---------------------------------------------------------------------------
# CLI end-to-end: monkeypatched fit_pooled/GT seams on hand-built PreparedRuns
# (mirrors tests/test_run_modebank.py's own `_prepared`/`_install`, the SAME
# fixture shape build_pool + FittedDetector.fit_pooled need to exercise for
# real -- no real data).
# ---------------------------------------------------------------------------


def _prepared(t0: int, seed: int, n_seg: int = 8, seg: int = 30) -> tuple[PreparedRun, np.ndarray]:
    """Contiguous *seg*-window segments alternating GT mode ('turbine'/'pump'),
    two well-separated blobs -- mirrors tests/test_run_modebank.py's own helper."""
    rng = np.random.default_rng(seed)
    feats, ids, gt = [], [], []
    for s in range(n_seg):
        mode = s % 2
        feats.append(rng.normal(0.0 if mode == 0 else 10.0, 0.2, (seg, 4)))
        ids.append(np.full(seg, s, dtype=np.int64))
        gt.append(np.array((["turbine"] if mode == 0 else ["pump"]) * seg, dtype=object))
    f = np.vstack(feats)
    p = PreparedRun(
        features=f,
        grid=WindowGrid(t0, _W, len(f)),
        valid_mask=np.ones(len(f), dtype=bool),
        feature_names=list(_NAMES),
        segment_ids=np.concatenate(ids),
    )
    return p, np.concatenate(gt)


def _install(monkeypatch, mod, results_root, prepared_by_run, gt_by_run) -> None:
    runs = [Run(name=n, files={}, day_root=Path(f"/d/{n}")) for n in prepared_by_run]
    monkeypatch.setattr(
        mod, "discover",
        lambda dr: RecordingIndex(runs=runs, betriebsdaten=[], betriebsdaten_by_day={}),
    )
    monkeypatch.setattr(
        mod, "load_config",
        lambda: Config(
            data_root=Path("/d"), results_root=results_root,
            detect=DetectConfig(n_states=2, min_dwell_s=3.0),
        ),
    )
    monkeypatch.setattr(
        mod, "prepare_run",
        lambda run, variant, cfg, *, use_cache: prepared_by_run[run.name],
    )
    monkeypatch.setattr(
        mod, "_run_gt_states",
        lambda prepared, run, index, cfg: gt_by_run[run.name],
    )


def test_sweep_min_dwell_cli_writes_state_ari_and_far_table(tmp_path, monkeypatch) -> None:
    import sweep_min_dwell as sd

    pf1, g1 = _prepared(0, 1)
    pf2, g2 = _prepared(0, 2)
    pt, gt = _prepared(9_000_000_000, 3)
    prepared = {"fitA": pf1, "fitB": pf2, "testC": pt}
    gts = {"fitA": g1, "fitB": g2, "testC": gt}
    _install(monkeypatch, sd, tmp_path / "results", prepared, gts)

    code = sd.main(
        [
            "--fit-runs", "fitA,fitB", "--test-run", "testC", "--variant", "fusion",
            "--k", "2", "--min-dwells", "1,3", "--alpha", "0.1", "--min-ref", "5",
        ]
    )
    assert code == 0

    out_dir = tmp_path / "results" / "step2" / "min-dwell-sweep" / "testC"
    table = pd.read_csv(out_dir / "fusion.csv")
    assert list(table["min_dwell_s"]) == [1.0, 3.0]
    assert {"min_dwell_windows", "state_ari", "recalibrate_far"} <= set(table.columns)
    assert table["state_ari"].between(-1.0 - 1e-9, 1.0 + 1e-9).all()
    far_col = table["recalibrate_far"]
    assert ((far_col >= 0.0) & (far_col <= 1.0) | far_col.isna()).all()

    sidecar = json.loads((out_dir / "fusion.json").read_text())
    assert sidecar["fit_runs"] == ["fitA", "fitB"]
    assert sidecar["test_run"] == "testC"
    assert sidecar["variant"] == "fusion"
    assert sidecar["min_dwells_s"] == [1.0, 3.0]
    assert "verdict" in sidecar and isinstance(sidecar["verdict"], str)


def test_duplicate_fit_run_names_exits_2() -> None:
    import sweep_min_dwell as sd

    with pytest.raises(SystemExit) as exc_info:
        sd.main(["--fit-runs", "fitA,fitA", "--test-run", "testC", "--variant", "fusion"])
    assert exc_info.value.code == 2


def test_test_run_listed_in_fit_runs_exits_2() -> None:
    import sweep_min_dwell as sd

    with pytest.raises(SystemExit) as exc_info:
        sd.main(["--fit-runs", "fitA,testC", "--test-run", "testC", "--variant", "fusion"])
    assert exc_info.value.code == 2


def test_unknown_run_names_exit_2(tmp_path, monkeypatch, capsys) -> None:
    import sweep_min_dwell as sd

    pf1, g1 = _prepared(0, 1)
    pt, gt = _prepared(9_000_000_000, 3)
    _install(
        monkeypatch, sd, tmp_path / "results",
        {"fitA": pf1, "testC": pt}, {"fitA": g1, "testC": gt},
    )

    code = sd.main(["--fit-runs", "fitA,nope", "--test-run", "testC", "--variant", "fusion"])
    assert code == 2
    assert "nope" in capsys.readouterr().err
