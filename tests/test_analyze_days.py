"""Tests for scripts/analyze_days.py (Package-8 D3): per-subcommand artifact-shape
tests on synthetic inputs + pure-helper math (flag-rate matrix, segment-block
bootstrap, 3dB classification boundary, era-step column math). A1.8: no partner
number appears as an expected value.

Beyond the plan's own RED block (flag-rate matrix, classify-shift boundary,
block-bootstrap, rotations-heatmap CLI), this file also exercises the
feature-stability and era-step subcommands end-to-end via the shared
`_run_features_and_gt` seam (monkeypatched directly, mirroring how
`tests/test_modebank.py` bypasses IO entirely) -- both are required deliverables
of Task 8's own interface section, not just their pure helpers.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_days as ad  # noqa: E402

from rowii.config import Config, DetectConfig  # noqa: E402
from rowii.io.dataset import RecordingIndex, Run  # noqa: E402

# ---------------------------------------------------------------------------
# Plan's own RED tests (verbatim, docs/superpowers/plans/
# 2026-07-21-step2-package8-modebank-explain.md, Task 8)
# ---------------------------------------------------------------------------


def test_flag_rate_matrix_reads_pooled_aggregate_row() -> None:
    far = {("010726-pu", "290626-tu"): 0.03, ("290626-tu", "010726-pu"): 0.51}
    m = ad._flag_rate_matrix(far)
    assert m.loc["010726-pu", "290626-tu"] == 0.03
    assert m.loc["290626-tu", "010726-pu"] == 0.51


def test_classify_shift_boundary_is_the_named_cutoff() -> None:
    assert ad._classify_shift(2.9, cutoff=3.0) == "slow"
    assert ad._classify_shift(3.1, cutoff=3.0) == "drifting"
    assert ad._classify_shift(3.0, cutoff=3.0) == "drifting"  # >= cutoff


def test_block_bootstrap_uses_segment_ids_not_wall_clock() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, 120)
    seg = np.repeat(np.arange(12), 10)  # 12 recording segments
    lo, hi = ad._block_bootstrap_ci(values, seg, n_boot=200, seed=1)
    assert lo < np.median(values) < hi
    # a degenerate single-segment array still returns a finite interval.
    lo1, hi1 = ad._block_bootstrap_ci(values, np.zeros(120, dtype=np.int64), n_boot=50, seed=1)
    assert np.isfinite(lo1) and np.isfinite(hi1)


def test_rotations_heatmap_subcommand_writes_png_and_csv(tmp_path, monkeypatch) -> None:
    # synthetic cross-day-pooled tree: two far tables with a 'pooled' aggregate row.
    root = tmp_path / "results" / "step2" / "cross-day-pooled"
    for test, fit, far in (("290626-tu", "010726-pu", 0.03), ("010726-pu", "290626-tu", 0.51)):
        d = root / test / "audio-pooled"
        d.mkdir(parents=True)
        pd.DataFrame(
            [{"label": "0", "realized_far": 0.0}, {"label": "pooled", "realized_far": far}]
        ).to_csv(d / "far_table_frozen.csv", index=False)
        (d / "notes.md").write_text(f"- fit pool: {fit} (pool order = `--fit-runs` order)\n")
    out = tmp_path / "results" / "analysis-days"
    code = ad.main(
        [
            "rotations-heatmap", "--root", str(root), "--out", str(out),
            "--variant", "audio", "--mode", "frozen",
        ]
    )
    assert code == 0
    assert (out / "rotations-heatmap" / "audio-frozen.png").is_file()
    assert (out / "rotations-heatmap" / "audio-frozen.csv").is_file()


# ---------------------------------------------------------------------------
# Extension 1: `_era_step_row` pure helper (named in the plan's own Interfaces
# section -- "Pure helper `_era_step_row(levels_by_stream, gt_mode_mask) -> dict`"
# -- but not itself part of the plan's given RED block).
# ---------------------------------------------------------------------------


def test_era_step_row_computes_per_stream_median_over_the_mask() -> None:
    levels_by_stream = {
        "RAWGeneratorMic__0": np.array([-40.0, -40.0, -10.0, -10.0]),
        "RAWTurbineVib__3": np.array([1.0, 1.0, 99.0, 99.0]),
    }
    mask = np.array([True, True, False, False])
    row = ad._era_step_row(levels_by_stream, mask)
    assert row == {"RAWGeneratorMic__0": -40.0, "RAWTurbineVib__3": 1.0}
    with pytest.raises(ValueError, match="zero windows"):
        ad._era_step_row(levels_by_stream, np.zeros(4, dtype=bool))


# ---------------------------------------------------------------------------
# Extension 2: feature-stability end-to-end (monkeypatched `_run_features_and_gt`
# seam -- Task 8's interface bullet requires the subcommand, not just its two
# pure helpers).
# ---------------------------------------------------------------------------

_STAB_NAMES = ["s::ch0_log_rms", "s::ch0_octave_500"]


def _stab_run(
    run_name: str, seed: int, *, shift: float = 0.0, has_gt: bool = True
) -> ad._RunFeatures:
    rng = np.random.default_rng(seed)
    n = 60
    features = rng.normal(0.0, 0.05, (n, 2))
    features[:, 0] += shift  # only feature 0 drifts across days
    segment_ids = np.repeat(np.arange(6), 10)
    gt_states = np.array((["turbine"] if has_gt else ["unknown"]) * n, dtype=object)
    return ad._RunFeatures(
        run_name=run_name, features=features, gt_states=gt_states,
        segment_ids=segment_ids, feature_names=list(_STAB_NAMES), has_gt=has_gt,
    )


def _install_features_and_gt(
    monkeypatch, mod, run_names: list[str], fake_by_run: dict[str, ad._RunFeatures], results_root
) -> None:
    runs = [Run(name=n, files={}, day_root=Path(f"/d/{n}")) for n in run_names]
    monkeypatch.setattr(
        mod, "discover",
        lambda dr: RecordingIndex(runs=runs, betriebsdaten=[], betriebsdaten_by_day={}),
    )
    monkeypatch.setattr(
        mod, "load_config",
        lambda: Config(data_root=Path("/d"), results_root=results_root, detect=DetectConfig()),
    )
    monkeypatch.setattr(
        mod, "_run_features_and_gt",
        lambda run_name, variant, cfg, index, **kw: fake_by_run[run_name],
    )


def test_feature_stability_writes_sorted_table_and_excludes_non_gt_runs(
    tmp_path, monkeypatch, caplog
) -> None:
    fake = {
        "dayA": _stab_run("dayA", 1, shift=0.0),
        "dayB": _stab_run("dayB", 2, shift=5.0),  # feature 0 shifts by ~5.0 (>= 3.0 cutoff)
        "dayNoGT": _stab_run("dayNoGT", 3, has_gt=False),
    }
    _install_features_and_gt(
        monkeypatch, ad, ["dayA", "dayB", "dayNoGT"], fake, tmp_path / "results"
    )

    out = tmp_path / "results" / "analysis-days"
    with caplog.at_level(logging.WARNING):
        code = ad.main(
            [
                "feature-stability", "--runs", "dayA,dayB,dayNoGT", "--variant", "audio",
                "--out", str(out), "--n-boot", "50", "--seed", "1", "--min-mode-windows", "5",
            ]
        )
    assert code == 0
    # A1.2: the non-GT-bearing day is excluded, with a warning naming it.
    assert any("dayNoGT" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)

    csv_path = out / "feature-stability" / "audio.csv"
    png_path = out / "feature-stability" / "audio.png"
    assert csv_path.is_file() and png_path.is_file()

    table = pd.read_csv(csv_path)
    assert {"feature", "mode", "day", "shift_abs", "classification"} <= set(table.columns)
    assert set(table["day"]) == {"dayA", "dayB"}  # dayNoGT never contributes a row

    shifted = table[table["feature"] == "s::ch0_log_rms"]
    assert (shifted["classification"] == "drifting").all()
    stable = table[table["feature"] == "s::ch0_octave_500"]
    assert (stable["classification"] == "slow").all()
    # sorted by shift_abs descending -- the drifting feature's rows come first.
    assert table.iloc[0]["feature"] == "s::ch0_log_rms"


def test_feature_stability_unknown_run_name_exits_2(tmp_path, monkeypatch, capsys) -> None:
    fake = {"dayA": _stab_run("dayA", 1)}
    _install_features_and_gt(monkeypatch, ad, ["dayA"], fake, tmp_path / "results")

    code = ad.main(
        [
            "feature-stability", "--runs", "dayA,nope", "--variant", "audio",
            "--out", str(tmp_path / "results" / "analysis-days"),
        ]
    )
    assert code == 2
    assert "nope" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Extension 3: era-step end-to-end -- the 27.06-style unmatched point (A1.2)
# and the --include-080726 gate (A1.2).
# ---------------------------------------------------------------------------

_ERA_NAMES = ["RAWGeneratorMic__0::ch0_log_rms", "RAWTurbineVib__3::ch0_log_rms"]


def _era_run(run_name: str, *, level: float, has_gt: bool) -> ad._RunFeatures:
    n = 40
    features = np.zeros((n, 2), dtype=np.float64)
    features[:, 0] = level
    features[:, 1] = level + 1.0
    gt_states = np.array((["turbine"] if has_gt else ["unknown"]) * n, dtype=object)
    return ad._RunFeatures(
        run_name=run_name, features=features, gt_states=gt_states,
        segment_ids=np.arange(n) // 10, feature_names=list(_ERA_NAMES), has_gt=has_gt,
    )


def test_era_step_marks_unmatched_day_and_gates_080726(tmp_path, monkeypatch, caplog) -> None:
    fake = {
        "dayGT": _era_run("dayGT", level=0.0, has_gt=True),
        "dayNoGT": _era_run("dayNoGT", level=10.0, has_gt=False),
        "080726-pu_strikes": _era_run("080726-pu_strikes", level=20.0, has_gt=True),
    }
    _install_features_and_gt(
        monkeypatch, ad, ["dayGT", "dayNoGT", "080726-pu_strikes"], fake, tmp_path / "results"
    )
    out = tmp_path / "results" / "analysis-days"

    # Without the gate flag: 080726 is excluded (warning names it); dayNoGT is an
    # unmatched point (no GT at all -- A1.2's exact wording), dayGT is matched.
    with caplog.at_level(logging.WARNING):
        code = ad.main(
            [
                "era-step", "--runs", "dayGT,dayNoGT,080726-pu_strikes",
                "--variant", "audio", "--gt-mode", "turbine", "--out", str(out),
            ]
        )
    assert code == 0
    assert any(
        "080726-pu_strikes" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )
    table = pd.read_csv(out / "era-step" / "audio.csv")
    assert set(table["run"]) == {"dayGT", "dayNoGT"}
    no_gt_rows = table[table["run"] == "dayNoGT"]
    assert (~no_gt_rows["matched"]).all()
    assert (no_gt_rows["note"] == "no GT -- era-A anchor by MeasName only").all()
    gt_rows = table[table["run"] == "dayGT"]
    assert gt_rows["matched"].all()
    assert (out / "era-step" / "audio.png").is_file()

    # With the gate flag: 080726 is included as a normal (matched) point.
    code2 = ad.main(
        [
            "era-step", "--runs", "dayGT,dayNoGT,080726-pu_strikes",
            "--variant", "audio", "--gt-mode", "turbine", "--out", str(out),
            "--include-080726",
        ]
    )
    assert code2 == 0
    table2 = pd.read_csv(out / "era-step" / "audio.csv")
    assert set(table2["run"]) == {"dayGT", "dayNoGT", "080726-pu_strikes"}
    strikes_rows = table2[table2["run"] == "080726-pu_strikes"]
    assert strikes_rows["matched"].all()
