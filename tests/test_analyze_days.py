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


# ---------------------------------------------------------------------------
# Task 9 (plan docs/superpowers/plans/2026-07-21-step2-package8-modebank-
# explain.md): `mode-signatures`, `tonal-table`, `pillar3-figure`, `digest` --
# three more pure helpers (`_tonal_contrast`, `_mode_profile`, `_tpr_by_alpha`,
# all explicitly named "Pure helper" in the plan's own Task 9 Interfaces
# section) + their subcommands, going through the SAME `_run_features_and_gt`/
# `_RunFeatures` seam Task 8 built (mode-signatures/tonal-table) or reading
# `results/pillar3/**/event_eval.csv` directly (pillar3-figure, no seam
# needed, mirrors rotations-heatmap's own direct-filesystem-read style).
# ---------------------------------------------------------------------------


def test_tonal_contrast_is_band_minus_floor() -> None:
    assert ad._tonal_contrast(band_energy=-30.0, octave_floor=-45.0) == 15.0  # our own definition


# ---------------------------------------------------------------------------
# Task 9 extension 1: `_mode_profile` + `_band_octave_columns` pure helpers,
# `mode-signatures` subcommand (per-day artifact-shape; plan's own RED block
# names the subcommand test literally -- extended here to two days to also
# pin the "one PNG per RUN, not per variant" file-naming contract).
# ---------------------------------------------------------------------------


def test_band_octave_columns_excludes_log_rms_and_shape_columns() -> None:
    names = [
        "s::ch0_log_rms", "s::ch0_band_shaft", "s::ch0_octave_125",
        "s::ch0_spectral_centroid", "s::ch0_rolloff95",
    ]
    assert ad._band_octave_columns(names) == [1, 2]


def test_mode_profile_reports_median_and_iqr_per_mode_excluding_unknown() -> None:
    # 3 columns; mode 'turbine' rows all at 0.0, mode 'pump' rows all at 10.0,
    # a GT 'unknown' row is present but must never surface as its own mode.
    features = np.array(
        [
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0],
            [99.0, 99.0, 99.0],
        ]
    )
    gt = np.array(["turbine", "turbine", "pump", "pump", "unknown"], dtype=object)
    table = ad._mode_profile(features, gt, level_cols=[0, 2])
    assert set(table["mode"]) == {"turbine", "pump"}
    assert set(table["column"]) == {0, 2}
    turbine_col0 = table[(table["mode"] == "turbine") & (table["column"] == 0)].iloc[0]
    assert turbine_col0["median"] == 0.0
    assert turbine_col0["q25"] == 0.0 and turbine_col0["q75"] == 0.0
    assert turbine_col0["n_windows"] == 2
    pump_col2 = table[(table["mode"] == "pump") & (table["column"] == 2)].iloc[0]
    assert pump_col2["median"] == 10.0


_MODESIG_NAMES = [f"s::ch0_octave_{fc}" for fc in (125, 250)] + ["s::ch0_band_shaft"]


def _modesig_run(run_name: str, seed: int) -> ad._RunFeatures:
    rng = np.random.default_rng(seed)
    n_per = 30
    feats = np.vstack([rng.normal(0.0, 0.1, (n_per, 3)), rng.normal(8.0, 0.1, (n_per, 3))])
    gt = np.array((["turbine"] * n_per) + (["pump"] * n_per), dtype=object)
    return ad._RunFeatures(
        run_name=run_name, features=feats, gt_states=gt,
        segment_ids=np.arange(2 * n_per) // 10, feature_names=list(_MODESIG_NAMES), has_gt=True,
    )


def test_mode_signatures_subcommand_writes_artifacts(tmp_path, monkeypatch) -> None:
    fake = {"290626-tu": _modesig_run("290626-tu", 0)}
    _install_features_and_gt(monkeypatch, ad, ["290626-tu"], fake, tmp_path / "results")
    out = tmp_path / "results" / "analysis-days"
    code = ad.main(
        ["mode-signatures", "--runs", "290626-tu", "--variant", "audio", "--out", str(out)]
    )
    assert code == 0
    assert (out / "mode-signatures" / "290626-tu.png").is_file()
    csv_path = out / "mode-signatures" / "290626-tu.csv"
    assert csv_path.is_file()
    table = pd.read_csv(csv_path)
    assert {"mode", "feature", "median", "q25", "q75", "n_windows"} <= set(table.columns)
    assert set(table["mode"]) == {"turbine", "pump"}


def test_mode_signatures_writes_one_png_per_run(tmp_path, monkeypatch) -> None:
    fake = {"dayA": _modesig_run("dayA", 1), "dayB": _modesig_run("dayB", 2)}
    _install_features_and_gt(monkeypatch, ad, ["dayA", "dayB"], fake, tmp_path / "results")
    out = tmp_path / "results" / "analysis-days"
    code = ad.main(
        ["mode-signatures", "--runs", "dayA,dayB", "--variant", "audio", "--out", str(out)]
    )
    assert code == 0
    assert (out / "mode-signatures" / "dayA.png").is_file()
    assert (out / "mode-signatures" / "dayB.png").is_file()


# ---------------------------------------------------------------------------
# Task 9 extension 2: `_nearest_octave_hz` + `_tonal_table` pure/impure
# helpers, `tonal-table` subcommand (artifact-shape).
# ---------------------------------------------------------------------------


def test_nearest_octave_hz_picks_closest_by_absolute_distance() -> None:
    assert ad._nearest_octave_hz(43.75, [31.5, 63.0, 125.0]) == 31.5
    assert ad._nearest_octave_hz(125.0, [31.5, 63.0, 125.0, 250.0]) == 125.0
    with pytest.raises(ValueError, match="empty"):
        ad._nearest_octave_hz(6.25, [])


_TONAL_NAMES = [
    "RAWGeneratorMic__0::ch0_band_shaft",
    "RAWGeneratorMic__0::ch0_octave_31",
    "RAWGeneratorMic__0::ch0_octave_63",
]


def _tonal_run(
    run_name: str, *, shaft: float, floor: float, has_gt: bool = True
) -> ad._RunFeatures:
    n = 20
    features = np.zeros((n, 3), dtype=np.float64)
    features[:, 0] = shaft  # ch0_band_shaft
    features[:, 1] = floor  # ch0_octave_31 -- nearest neighbour to MACHINE_HZ['shaft']=6.25
    features[:, 2] = floor + 50.0  # ch0_octave_63 -- must NOT be picked as the floor
    gt = np.array((["turbine"] if has_gt else ["unknown"]) * n, dtype=object)
    return ad._RunFeatures(
        run_name=run_name, features=features, gt_states=gt,
        segment_ids=np.arange(n) // 5, feature_names=list(_TONAL_NAMES), has_gt=has_gt,
    )


def test_tonal_table_contrasts_machine_band_against_nearest_octave() -> None:
    rf = _tonal_run("dayA", shaft=-30.0, floor=-45.0)
    table = ad._tonal_table([rf])
    row = table[(table["band"] == "shaft") & (table["mode"] == "turbine")].iloc[0]
    assert row["stream"] == "RAWGeneratorMic__0"
    assert row["band_energy"] == pytest.approx(-30.0)
    assert row["octave_floor"] == pytest.approx(-45.0)
    assert row["octave_floor_hz"] == pytest.approx(31.0)  # parsed from '_octave_31'
    assert row["tonal_contrast"] == pytest.approx(15.0)
    # only 'shaft' has a '_band_*' column in this fixture -- blade_pass/guide_vane_pass absent.
    assert set(table["band"]) == {"shaft"}


def test_tonal_table_excludes_non_gt_runs() -> None:
    rf_no_gt = _tonal_run("dayNoGT", shaft=-30.0, floor=-45.0, has_gt=False)
    table = ad._tonal_table([rf_no_gt])
    assert table.empty


def test_tonal_table_subcommand_writes_artifacts(tmp_path, monkeypatch) -> None:
    fake = {"dayA": _tonal_run("dayA", shaft=-30.0, floor=-45.0)}
    _install_features_and_gt(monkeypatch, ad, ["dayA"], fake, tmp_path / "results")
    out = tmp_path / "results" / "analysis-days"
    code = ad.main(["tonal-table", "--runs", "dayA", "--variant", "audio", "--out", str(out)])
    assert code == 0
    assert (out / "tonal-table" / "audio.png").is_file()
    table = pd.read_csv(out / "tonal-table" / "audio.csv")
    assert {"run", "mode", "stream", "band", "tonal_contrast"} <= set(table.columns)


# ---------------------------------------------------------------------------
# Task 9 extension 3: `_tpr_by_alpha` pure helper + `pillar3-figure`
# subcommand -- the REAL on-disk `event_eval.csv` schema verified against
# `results/pillar3/080726-{pu,st}_strikes/**/event_eval.csv` on this branch:
# `row_type == "summary"`, columns n_events/n_detected/event_tpr/
# realized_window_far (`scripts/eval_events.py`'s own `_CSV_COLUMNS`).
# ---------------------------------------------------------------------------


def test_tpr_by_alpha_pivots_representation_by_alpha() -> None:
    table = pd.DataFrame(
        [
            {"representation": "audio", "alpha": 0.01, "event_tpr": 0.5},
            {"representation": "audio", "alpha": 0.05, "event_tpr": 0.8},
            {"representation": "fusion", "alpha": 0.01, "event_tpr": 0.6},
        ]
    )
    m = ad._tpr_by_alpha(table)
    assert m.loc["audio", 0.01] == 0.5
    assert m.loc["audio", 0.05] == 0.8
    assert m.loc["fusion", 0.01] == 0.6
    assert np.isnan(m.loc["fusion", 0.05])


def _write_event_eval(
    path: Path, *, n_events: float, n_detected: float, event_tpr: float, realized_window_far: float
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "row_type": "summary", "n_events": n_events, "n_detected": n_detected,
                "event_tpr": event_tpr, "false_alarm_windows": 4.0,
                "false_alarm_rate_per_hour": 10.0, "realized_window_far": realized_window_far,
                "tolerance_s": 5.0, "start_utc": "", "end_utc": "", "kind": "", "detected": "",
                "latency_s": "",
            },
            {
                "row_type": "event", "n_events": "", "n_detected": "", "event_tpr": "",
                "false_alarm_windows": "", "false_alarm_rate_per_hour": "",
                "realized_window_far": "", "tolerance_s": "",
                "start_utc": "2026-07-08 10:00:00+00:00", "end_utc": "2026-07-08 10:01:00+00:00",
                "kind": "plate-gen_0", "detected": "True", "latency_s": 3.0,
            },
        ]
    ).to_csv(path / "event_eval.csv", index=False)


def test_pillar3_figure_subcommand_writes_artifacts_and_skips_non_alpha_leaves(tmp_path) -> None:
    root = tmp_path / "results" / "pillar3"
    _write_event_eval(
        root / "080726-pu_strikes" / "fusion-a0.01",
        n_events=10.0, n_detected=6.0, event_tpr=0.6, realized_window_far=0.01,
    )
    _write_event_eval(
        root / "080726-pu_strikes" / "audio-a0.05",
        n_events=10.0, n_detected=10.0, event_tpr=1.0, realized_window_far=0.06,
    )
    _write_event_eval(
        root / "080726-st_strikes" / "fusion-a0.01",
        n_events=10.0, n_detected=9.0, event_tpr=0.9, realized_window_far=0.02,
    )
    # 'fusion-frozen' carries no '-a<alpha>' suffix -- not part of the alpha
    # grid this figure compares, must be silently skipped.
    _write_event_eval(
        root / "080726-st_strikes" / "fusion-frozen",
        n_events=10.0, n_detected=10.0, event_tpr=1.0, realized_window_far=0.03,
    )

    out = tmp_path / "results" / "analysis-days"
    code = ad.main(["pillar3-figure", "--root", str(root), "--out", str(out)])
    assert code == 0
    assert (out / "pillar3-figure" / "pillar3.png").is_file()
    table = pd.read_csv(out / "pillar3-figure" / "pillar3.csv")
    assert {
        "session", "representation", "alpha", "n_events", "n_detected",
        "event_tpr", "realized_window_far",
    } <= set(table.columns)
    assert set(table["representation"]) == {"fusion", "audio"}  # 'fusion-frozen' excluded
    assert len(table) == 3


def test_pillar3_figure_missing_root_exits_1(tmp_path, capsys) -> None:
    code = ad.main(
        ["pillar3-figure", "--root", str(tmp_path / "nope"), "--out", str(tmp_path / "out")]
    )
    assert code == 1
    assert "nope" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Task 9 extension 4: `digest` (plan's own RED block, extended to also check
# every subcommand name and the discovered-PNG link are present).
# ---------------------------------------------------------------------------


def test_digest_writes_readme_with_attribution_lines(tmp_path) -> None:
    out = tmp_path / "results" / "analysis-days"
    (out / "rotations-heatmap").mkdir(parents=True)
    (out / "rotations-heatmap" / "audio-frozen.png").write_bytes(b"x")
    assert ad.main(["digest", "--out", str(out)]) == 0
    readme = (out / "README.md").read_text()
    assert "Rodrigues & Zhang (2026)" in readme  # attribution present
    assert "z-score" in readme and "fusion" in readme  # A1.1 finding documented
    assert "rotations-heatmap/audio-frozen.png" in readme  # figure actually linked
    for name in (
        "rotations-heatmap", "feature-stability", "era-step",
        "mode-signatures", "tonal-table", "pillar3-figure",
    ):
        assert name in readme
