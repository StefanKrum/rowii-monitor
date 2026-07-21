"""Tests for scripts/run_modebank_chain.py (Package-8 D1 chain probe): split-parity
vs run_step2's top split (BINDING) + FAR math on synthetic two-mode data, the chain's
own per-mode reference/threshold/FAR assembly (`_build_far_table`, unit-tested
directly on hand-built BANK-ASSIGNED label arrays -- never GT, mirroring run_step2
cross-day-pooled's "detected-labels only" convention with the bank standing in for
the detector), and a full CLI end-to-end pass on a monkeypatched two-mode fixture
mirroring `tests/test_run_modebank.py`'s just-committed seam.

Style-2 fixtures throughout (no synthetic Gantner trees): `discover`/`load_config`/
`prepare_run`/`_run_gt_states` are monkeypatched, no real data root ever touched.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from rowii.anomaly.references import split_by_segments  # noqa: E402
from rowii.anomaly.sweep import SweepConfig  # noqa: E402
from rowii.config import Config, DetectConfig  # noqa: E402
from rowii.io.dataset import BurstFile, RecordingIndex, Run  # noqa: E402
from rowii.pipeline import PreparedRun  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402

_W = 1_000_000_000
_NAMES = [f"ch0_octave_{i}" for i in range(4)]


# ---------------------------------------------------------------------------
# 1. Plan's own mandatory RED tests: the two pure, load-bearing helpers.
# ---------------------------------------------------------------------------


def test_top_split_parity_with_run_step2_convention() -> None:
    import run_modebank_chain as rc

    seg = np.repeat(np.arange(10, dtype=np.int64), 20)
    valid = np.ones(200, dtype=bool)
    got = rc._top_split(seg, valid)  # the chain's own call
    ref = split_by_segments(seg, valid, 0.5, 7)  # run_step2 cross-day-pooled convention
    np.testing.assert_array_equal(got.calibration_windows, ref.calibration_windows)
    np.testing.assert_array_equal(got.scoring_windows, ref.scoring_windows)


def test_top_split_literals_match_sweepconfig_defaults() -> None:
    """Tripwire (T4-review follow-up F1): `_TOP_FRAC`/`_TOP_SEED` are HARD-CODED
    literals, not read off `SweepConfig`'s own fields (module docstring's BINDING
    split-parity rationale) -- so if `SweepConfig`'s `calibration_frac`/`seed`
    defaults ever drift, nothing else here would notice: the held-out top split
    would silently diverge from run_step2's while the pools kept following the new
    defaults. A failure here means the split-parity contract must be RE-DECIDED
    (should the literals now track the new default, or intentionally keep the old
    one?) -- not that the literals should be blindly bumped to make this pass again.
    """
    import run_modebank_chain as rc

    assert SweepConfig().calibration_frac == rc._TOP_FRAC
    assert SweepConfig().seed == rc._TOP_SEED


def test_far_math_on_synthetic_two_mode_scoring() -> None:
    import run_modebank_chain as rc

    # 8 scored windows, 2 flagged as alarms -> realized_far = 0.25.
    scores = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 9.0, 9.0])
    threshold = 1.0
    far, n_alarm, n = rc._far(scores, threshold)
    assert (n, n_alarm) == (8, 2)
    assert far == 0.25


# ---------------------------------------------------------------------------
# 2. `_build_far_table`: the chain's own per-mode reference/threshold/FAR
#    assembly, unit-tested directly on hand-built BANK-ASSIGNED label arrays
#    (no CLI, no ModeBank fitting needed) -- covers the min_ref exclusion,
#    no-conformal-data, empty-scoring, and scored+aggregate paths that mirror
#    run_step2.py's `_cross_day_pooled_tables`/`far_row_*` dispatch.
# ---------------------------------------------------------------------------


def _sweep_cfg(min_ref: int = 20, alpha: float = 0.05) -> SweepConfig:
    return SweepConfig(alpha=alpha, min_ref=min_ref, scorer="knn")


def test_build_far_table_excludes_label_below_min_ref() -> None:
    import run_modebank_chain as rc

    rng = np.random.default_rng(0)
    fit_features = rng.normal(size=(25, 3))
    fit_labels = np.array(["common"] * 22 + ["rare"] * 3, dtype=object)
    conformal_features = rng.normal(size=(10, 3))
    conformal_labels = np.array(["common"] * 8 + ["rare"] * 2, dtype=object)
    scoring_features = rng.normal(size=(5, 3))
    scoring_labels = np.array(["common"] * 4 + ["rare"] * 1, dtype=object)

    table = rc._build_far_table(
        fit_features, fit_labels, conformal_features, conformal_labels,
        scoring_features, scoring_labels, _sweep_cfg(min_ref=20),
    )
    rare_row = table[table["label"] == "rare"].iloc[0]
    assert bool(rare_row["excluded"]) is True
    assert np.isnan(rare_row["realized_far"])
    common_row = table[table["label"] == "common"].iloc[0]
    assert bool(common_row["excluded"]) is False


def test_build_far_table_no_conformal_data() -> None:
    import run_modebank_chain as rc

    rng = np.random.default_rng(1)
    fit_features = rng.normal(size=(25, 3))
    fit_labels = np.array(["only"] * 25, dtype=object)
    conformal_features = np.empty((0, 3))
    conformal_labels = np.empty(0, dtype=object)
    scoring_features = rng.normal(size=(5, 3))
    scoring_labels = np.array(["only"] * 5, dtype=object)

    table = rc._build_far_table(
        fit_features, fit_labels, conformal_features, conformal_labels,
        scoring_features, scoring_labels, _sweep_cfg(min_ref=20),
    )
    row = table[table["label"] == "only"].iloc[0]
    assert bool(row["excluded"]) is False
    assert np.isnan(row["realized_far"])
    assert bool(row["low_confidence"]) is True  # far_row_no_conformal_data's own convention


def test_build_far_table_empty_scoring_side() -> None:
    import run_modebank_chain as rc

    rng = np.random.default_rng(2)
    fit_features = rng.normal(size=(25, 3))
    fit_labels = np.array(["a"] * 25, dtype=object)
    conformal_features = rng.normal(size=(10, 3))
    conformal_labels = np.array(["a"] * 10, dtype=object)
    scoring_features = rng.normal(size=(4, 3))
    scoring_labels = np.array(["b"] * 4, dtype=object)  # a different bank-assigned mode

    table = rc._build_far_table(
        fit_features, fit_labels, conformal_features, conformal_labels,
        scoring_features, scoring_labels, _sweep_cfg(min_ref=20),
    )
    row_a = table[table["label"] == "a"].iloc[0]
    assert row_a["n_scored"] == 0.0
    assert np.isnan(row_a["realized_far"])
    assert bool(row_a["excluded"]) is False
    # label "b" has NO fit-side reference at all -- excluded, never no-conformal-data.
    row_b = table[table["label"] == "b"].iloc[0]
    assert bool(row_b["excluded"]) is True


def test_build_far_table_scored_rows_plus_pooled_aggregate() -> None:
    import run_modebank_chain as rc

    rng = np.random.default_rng(3)
    fit_features = np.vstack(
        [rng.normal(0.0, 0.2, (30, 3)), rng.normal(10.0, 0.2, (30, 3))]
    )
    fit_labels = np.array(["a"] * 30 + ["b"] * 30, dtype=object)
    conformal_features = np.vstack(
        [rng.normal(0.0, 0.2, (30, 3)), rng.normal(10.0, 0.2, (30, 3))]
    )
    conformal_labels = np.array(["a"] * 30 + ["b"] * 30, dtype=object)
    scoring_features = np.vstack(
        [rng.normal(0.0, 0.2, (10, 3)), rng.normal(10.0, 0.2, (10, 3))]
    )
    scoring_labels = np.array(["a"] * 10 + ["b"] * 10, dtype=object)

    table = rc._build_far_table(
        fit_features, fit_labels, conformal_features, conformal_labels,
        scoring_features, scoring_labels, _sweep_cfg(min_ref=20),
    )
    assert set(table["label"]) == {"a", "b", "pooled"}
    for lbl in ("a", "b"):
        row = table[table["label"] == lbl].iloc[0]
        assert bool(row["excluded"]) is False
        assert row["n_scored"] == 10.0
        assert 0.0 <= row["realized_far"] <= 1.0
    pooled = table[table["label"] == "pooled"].iloc[0]
    non_agg = table[table["label"] != "pooled"]
    assert pooled["n_scored"] == non_agg["n_scored"].sum()
    assert pooled["n_alarms"] == non_agg["n_alarms"].sum()
    assert pooled["realized_far"] == pytest.approx(
        non_agg["n_alarms"].sum() / non_agg["n_scored"].sum()
    )


# ---------------------------------------------------------------------------
# 3. Full CLI end-to-end: monkeypatched discover/load_config/prepare_run/
#    _run_gt_states seam (mirrors tests/test_run_modebank.py). n_seg=20
#    fixtures -- verified empirically to give BOTH GT modes non-degenerate
#    fit- AND conformal-side pool coverage at SweepConfig defaults (0.5/7 top
#    split, 0.5/8 nested split), unlike test_run_modebank.py's own n_seg=8
#    fixture (which starves one mode's conformal side under the SAME defaults
#    -- fine there since that test only asserts generic bank-metrics shape,
#    but this CLI's happy-path test wants BOTH bank-assigned modes to reach
#    real (non-excluded) far_table rows).
# ---------------------------------------------------------------------------


def _prepared(
    t0: int, seed: int, n_seg: int = 20, seg: int = 30
) -> tuple[PreparedRun, np.ndarray]:
    rng = np.random.default_rng(seed)
    feats, ids, gt = [], [], []
    for s in range(n_seg):
        mode = s % 2  # alternate turbine / pump
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


def test_run_modebank_chain_writes_far_table_with_both_modes_and_pooled_row(
    tmp_path, monkeypatch
) -> None:
    import run_modebank_chain as rc

    pf1, g1 = _prepared(0, 1)
    pf2, g2 = _prepared(0, 2)
    pt, gt = _prepared(9_000_000_000, 3)
    prepared = {"fitA": pf1, "fitB": pf2, "testC": pt}
    # Only the FIT runs' GT is ever looked up (module docstring: the chain
    # conditions on the bank's OWN label-free assignment, never the test
    # run's GT) -- "testC" deliberately absent from gt_by_run to pin that.
    gts = {"fitA": g1, "fitB": g2}
    _install(monkeypatch, rc, tmp_path / "results", prepared, gts)
    monkeypatch.setattr(rc, "_run_day_groups", lambda run: {run.name})  # force disjoint

    code = rc.main([
        "--fit-runs", "fitA,fitB", "--test-run", "testC",
        "--variant", "fusion", "--family", "gaussian", "--alpha", "0.05",
    ])
    assert code == 0

    out = tmp_path / "results" / "step2" / "modebank-chain" / "testC" / "fusion-gaussian"
    table = pd.read_csv(out / "far_table.csv")
    assert list(table.columns) == list(rc._FAR_TABLE_COLUMNS)
    assert set(table["label"]) == {"turbine", "pump", "pooled"}
    assert (table["excluded"] == False).all()  # noqa: E712 -- both modes survive at this fixture size
    pooled = table[table["label"] == "pooled"].iloc[0]
    non_agg = table[table["label"] != "pooled"]
    assert pooled["n_scored"] == non_agg["n_scored"].sum()
    assert pooled["n_alarms"] == non_agg["n_alarms"].sum()

    notes = (out / "notes.md").read_text()
    assert "fitA" in notes and "fitB" in notes and "testC" in notes
    assert "inspired by the partner" in notes
    # Threshold-regime self-description (T4-review follow-up F3): notes.md must
    # name its own regime and point at the correct P7 comparison file, not the
    # recalibrate one this probe has no arm for.
    assert "frozen" in notes
    assert "far_table_frozen.csv" in notes


# ---------------------------------------------------------------------------
# 4. Argument-shape guards (pure, no data touched) -- mirrors
#    scripts/run_modebank.py's own guards verbatim.
# ---------------------------------------------------------------------------


def test_duplicate_fit_run_names_exits_2(capsys) -> None:
    import run_modebank_chain as rc

    with pytest.raises(SystemExit) as exc_info:
        rc.main([
            "--fit-runs", "fitA,fitA", "--test-run", "testC",
            "--variant", "fusion", "--family", "gaussian",
        ])
    assert exc_info.value.code == 2
    assert "duplicate" in capsys.readouterr().err


def test_test_run_listed_in_fit_runs_exits_2(capsys) -> None:
    import run_modebank_chain as rc

    with pytest.raises(SystemExit) as exc_info:
        rc.main([
            "--fit-runs", "fitA,testC", "--test-run", "testC",
            "--variant", "fusion", "--family", "gaussian",
        ])
    assert exc_info.value.code == 2
    assert "testC" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 5. Data-dependent guards (discover/prepare_run seam, return 2).
# ---------------------------------------------------------------------------


def test_unknown_run_names_exit_2(tmp_path, monkeypatch, capsys) -> None:
    import run_modebank_chain as rc

    pf1, g1 = _prepared(0, 1)
    pt, gt = _prepared(9_000_000_000, 3)
    _install(
        monkeypatch, rc, tmp_path / "results",
        {"fitA": pf1, "testC": pt}, {"fitA": g1},
    )

    code = rc.main([
        "--fit-runs", "fitA,nope", "--test-run", "testC",
        "--variant", "fusion", "--family", "gaussian",
    ])
    assert code == 2
    assert "nope" in capsys.readouterr().err


def test_day_group_overlap_between_fit_and_test_exits_2(tmp_path, monkeypatch, capsys) -> None:
    """The A3.8-style day-group guard (duplicated from run_step2.py/run_modebank.py's
    `_run_day_groups`) fires on the REAL (unmocked) day-group computation -- fitA and
    testC share one calendar day."""
    import run_modebank_chain as rc

    def _fake_run(name: str, date: str) -> Run:
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

    pf1, g1 = _prepared(0, 1)
    pt, gt = _prepared(9_000_000_000, 3)
    prepared = {"fitA": pf1, "testC": pt}
    gts = {"fitA": g1}
    runs = [_fake_run("fitA", "2026-07-01"), _fake_run("testC", "2026-07-01")]
    monkeypatch.setattr(
        rc, "discover",
        lambda dr: RecordingIndex(runs=runs, betriebsdaten=[], betriebsdaten_by_day={}),
    )
    monkeypatch.setattr(
        rc, "load_config",
        lambda: Config(data_root=Path("/d"), results_root=tmp_path / "results"),
    )
    monkeypatch.setattr(
        rc, "prepare_run",
        lambda run, variant, cfg, *, use_cache: prepared[run.name],
    )
    monkeypatch.setattr(
        rc, "_run_gt_states",
        lambda prepared_run, run, index, cfg: gts[run.name],
    )

    code = rc.main([
        "--fit-runs", "fitA", "--test-run", "testC",
        "--variant", "fusion", "--family", "gaussian",
    ])
    assert code == 2
    err = capsys.readouterr().err
    assert "fitA" in err
    assert "2026-07-01" in err
