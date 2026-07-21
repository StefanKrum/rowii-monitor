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
# T8-review item 4: rotations-heatmap `--leaf-suffix` discovery + the <2-
# rotations guard (never render a 1-cell "matrix" silently).
# ---------------------------------------------------------------------------


def test_rotations_heatmap_leaf_suffix_discovers_suffixed_leaves(tmp_path) -> None:
    root = tmp_path / "results" / "step2" / "cross-day-pooled"
    for test, fit, far in (("290626-tu", "010726-pu", 0.03), ("010726-pu", "290626-tu", 0.51)):
        d = root / test / "fusion-pooled-a0.05"
        d.mkdir(parents=True)
        pd.DataFrame(
            [{"label": "0", "realized_far": 0.0}, {"label": "pooled", "realized_far": far}]
        ).to_csv(d / "far_table_frozen.csv", index=False)
        (d / "notes.md").write_text(f"- fit pool: {fit} (pool order = `--fit-runs` order)\n")
    # a PLAIN (no-suffix) leaf also exists for one test run -- must NOT be
    # picked up when --leaf-suffix selects the '-a0.05' leaves only.
    plain = root / "290626-tu" / "fusion-pooled"
    plain.mkdir(parents=True)
    pd.DataFrame(
        [{"label": "0", "realized_far": 0.0}, {"label": "pooled", "realized_far": 0.99}]
    ).to_csv(plain / "far_table_frozen.csv", index=False)
    (plain / "notes.md").write_text("- fit pool: 010726-pu (pool order = `--fit-runs` order)\n")

    out = tmp_path / "results" / "analysis-days"
    code = ad.main(
        [
            "rotations-heatmap", "--root", str(root), "--out", str(out),
            "--variant", "fusion", "--mode", "frozen",
            # NOTE: argparse's "looks like another option" heuristic misreads a
            # bare `-a0.05` token as a flag -- the `--leaf-suffix=...` form
            # keeps it inside one token (a real CLI-usage gotcha, not a bug in
            # the discovery logic itself, which _run_rotations_heatmap reads
            # via `str(args.leaf_suffix)` either way).
            "--leaf-suffix=-a0.05",
        ]
    )
    assert code == 0
    assert (out / "rotations-heatmap" / "fusion-frozen.png").is_file()
    csv_path = out / "rotations-heatmap" / "fusion-frozen.csv"
    assert csv_path.is_file()
    table = pd.read_csv(csv_path, index_col=0)
    # the plain leaf's 0.99 value must never surface -- only the '-a0.05' pair.
    assert table.loc["010726-pu", "290626-tu"] == pytest.approx(0.03)


def test_rotations_heatmap_single_leaf_exits_2_with_hint(tmp_path, capsys) -> None:
    root = tmp_path / "results" / "step2" / "cross-day-pooled"
    d = root / "290626-tu" / "audio-pooled"
    d.mkdir(parents=True)
    pd.DataFrame(
        [{"label": "0", "realized_far": 0.0}, {"label": "pooled", "realized_far": 0.2}]
    ).to_csv(d / "far_table_frozen.csv", index=False)
    (d / "notes.md").write_text("- fit pool: 010726-pu (pool order = `--fit-runs` order)\n")

    out = tmp_path / "results" / "analysis-days"
    code = ad.main(
        [
            "rotations-heatmap", "--root", str(root), "--out", str(out),
            "--variant", "audio", "--mode", "frozen",
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "--leaf-suffix" in err
    assert "010726-pu" in err  # what WAS found is listed, not just a bare count
    assert not (out / "rotations-heatmap").exists()  # never rendered


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
# T8-review item 1 (BLOCKER, unit coherence): feature-stability's slow/
# drifting cutoff must compare against a genuine dB figure, not a raw log10
# shift -- `_log_rms` stores log10(RMS AMPLITUDE) (dB = 20*log10(ratio)),
# `_band_`/`_octave_` store log10(mean Welch PSD, POWER) (dB =
# 10*log10(ratio)). Exact fixture values chosen so the family factor itself
# is pinned: 0.2 log10 units reads "slow" for a band/octave column (x10 -> 2
# dB) but "drifting" for a log_rms column (x20 -> 4 dB) -- these would
# collapse to the SAME classification under the old, unconverted comparison.
# ---------------------------------------------------------------------------

_DB_FIX_NAMES = ["s::ch0_band_shaft", "s::ch0_log_rms"]


def _db_fix_run(run_name: str, *, band_value: float, log_rms_value: float) -> ad._RunFeatures:
    """A single-mode ('turbine'), noise-free run: every one of its 10 windows
    holds the SAME (band_value, log_rms_value) pair, so the cross-day
    `shift_abs` between two calls of this helper is EXACTLY the difference of
    the values passed in -- no bootstrap/median noise to obscure the family-
    factor boundary math."""
    n = 10
    features = np.column_stack([np.full(n, band_value), np.full(n, log_rms_value)])
    return ad._RunFeatures(
        run_name=run_name, features=features,
        gt_states=np.array(["turbine"] * n, dtype=object),
        segment_ids=np.zeros(n, dtype=np.int64),
        feature_names=list(_DB_FIX_NAMES), has_gt=True,
    )


def test_feature_stability_band_shift_04_log10_is_4db_drifting() -> None:
    runs = [
        _db_fix_run("dayA", band_value=0.0, log_rms_value=0.0),
        _db_fix_run("dayB", band_value=0.4, log_rms_value=0.0),
    ]
    table = ad._feature_stability_table(
        runs, n_boot=20, seed=1, cutoff=3.0, min_mode_windows=5, variant="audio",
    )
    band_rows = table[table["feature"] == "s::ch0_band_shaft"]
    assert band_rows["shift_db"].iloc[0] == pytest.approx(4.0)  # 0.4 * 10
    assert (band_rows["classification"] == "drifting").all()


def test_feature_stability_band_shift_02_log10_is_2db_slow() -> None:
    runs = [
        _db_fix_run("dayA", band_value=0.0, log_rms_value=0.0),
        _db_fix_run("dayB", band_value=0.2, log_rms_value=0.0),
    ]
    table = ad._feature_stability_table(
        runs, n_boot=20, seed=1, cutoff=3.0, min_mode_windows=5, variant="audio",
    )
    band_rows = table[table["feature"] == "s::ch0_band_shaft"]
    assert band_rows["shift_db"].iloc[0] == pytest.approx(2.0)  # 0.2 * 10
    assert (band_rows["classification"] == "slow").all()


def test_feature_stability_log_rms_shift_02_log10_is_4db_drifting() -> None:
    # SAME 0.2 log10-unit magnitude as the band 'slow' case above, but
    # log_rms's own x20 family factor puts it over the 3.0 dB cutoff -- the
    # two families must NOT share one raw-unit comparison (the BLOCKER bug).
    runs = [
        _db_fix_run("dayA", band_value=0.0, log_rms_value=0.0),
        _db_fix_run("dayB", band_value=0.0, log_rms_value=0.2),
    ]
    table = ad._feature_stability_table(
        runs, n_boot=20, seed=1, cutoff=3.0, min_mode_windows=5, variant="audio",
    )
    log_rms_rows = table[table["feature"] == "s::ch0_log_rms"]
    assert log_rms_rows["shift_db"].iloc[0] == pytest.approx(4.0)  # 0.2 * 20
    assert (log_rms_rows["classification"] == "drifting").all()


# ---------------------------------------------------------------------------
# T8-review item 2: classification is LEVEL-columns-only (shape columns keep
# their dot-interval rows but read "n/a"); fusion/embedding variants skip the
# dB classification entirely (warned), since their level-NAMED columns hold
# z-score/embedding values, not log10 ones.
# ---------------------------------------------------------------------------

_SHAPE_FIX_NAMES = ["s::ch0_band_shaft", "s::ch0_spectral_centroid"]


def _shape_fix_run(run_name: str, *, band_value: float, shape_value: float) -> ad._RunFeatures:
    n = 10
    features = np.column_stack([np.full(n, band_value), np.full(n, shape_value)])
    return ad._RunFeatures(
        run_name=run_name, features=features,
        gt_states=np.array(["turbine"] * n, dtype=object),
        segment_ids=np.zeros(n, dtype=np.int64),
        feature_names=list(_SHAPE_FIX_NAMES), has_gt=True,
    )


def test_feature_stability_shape_column_is_n_a_but_keeps_dot_interval_rows() -> None:
    runs = [
        _shape_fix_run("dayA", band_value=0.0, shape_value=100.0),
        _shape_fix_run("dayB", band_value=0.4, shape_value=500.0),
    ]
    table = ad._feature_stability_table(
        runs, n_boot=20, seed=1, cutoff=3.0, min_mode_windows=5, variant="audio",
    )
    shape_rows = table[table["feature"] == "s::ch0_spectral_centroid"]
    assert len(shape_rows) == 2  # one dot-interval row per day, never dropped
    assert (shape_rows["classification"] == "n/a").all()
    assert shape_rows["shift_db"].isna().all()
    # the LEVEL column in the same table is unaffected -- still classified.
    band_rows = table[table["feature"] == "s::ch0_band_shaft"]
    assert (band_rows["classification"] == "drifting").all()


def test_feature_stability_fusion_variant_skips_db_classification(caplog) -> None:
    runs = [
        _db_fix_run("dayA", band_value=0.0, log_rms_value=0.0),
        _db_fix_run("dayB", band_value=0.4, log_rms_value=0.0),
    ]
    with caplog.at_level(logging.WARNING):
        table = ad._feature_stability_table(
            runs, n_boot=20, seed=1, cutoff=3.0, min_mode_windows=5, variant="fusion",
        )
    assert (table["classification"] == "n/a").all()
    assert table["shift_db"].isna().all()
    assert any(
        "fusion" in r.getMessage() and "z-score" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_feature_stability_embedding_variant_skips_db_classification(caplog) -> None:
    names = ["beats_0", "beats_1"]  # no _log_rms/_band_/_octave_ token at all

    def _embed_run(run_name: str, value: float) -> ad._RunFeatures:
        n = 10
        return ad._RunFeatures(
            run_name=run_name, features=np.column_stack([np.full(n, value), np.zeros(n)]),
            gt_states=np.array(["turbine"] * n, dtype=object),
            segment_ids=np.zeros(n, dtype=np.int64), feature_names=list(names), has_gt=True,
        )

    runs = [_embed_run("dayA", 0.0), _embed_run("dayB", 5.0)]
    with caplog.at_level(logging.WARNING):
        table = ad._feature_stability_table(
            runs, n_boot=20, seed=1, cutoff=3.0, min_mode_windows=5, variant="audio-beats",
        )
    assert (table["classification"] == "n/a").all()
    assert any(
        "embedding" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


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
# T8-review item 3: era-step's `--variant` is raw-scale-only (audio/
# vibration) -- fusion's per-run z-score (A1.1) has no meaningful log10 level
# to plot here.
# ---------------------------------------------------------------------------


def test_era_step_refuses_fusion_variant(tmp_path, capsys) -> None:
    code = ad.main(
        [
            "era-step", "--runs", "dayA", "--variant", "fusion",
            "--out", str(tmp_path / "out"),
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "fusion" in err
    assert "audio" in err and "vibration" in err


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
    # T9-review item 2 (superseded by the "nearest NON-CONTAINING octave"
    # semantics below): 6.25 Hz sits outside every candidate's own span, so
    # the plain nearest-by-Hz result is unchanged by the new rule.
    assert ad._nearest_octave_hz(6.25, [31.5, 63.0, 125.0]) == 31.5
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


# ---------------------------------------------------------------------------
# T9-review polish (2026-07-21 fix-loop, post-a05344e): item 1 variant-aware
# unit labeling on mode-signatures/tonal-table (title + axis/colorbar +
# digest caveat), item 2 tonal-table's nearest NON-CONTAINING octave floor,
# item 3 a per-leaf parse guard on pillar3-figure's discovery, item 4 two
# digest-prose nits (fold in).
# ---------------------------------------------------------------------------

# --- item 1: `_feature_unit_label` (shared pure helper) --------------------

_FUSION_STYLE_NAMES = [
    "RAWGeneratorMic__0::ch0_band_shaft", "RAWGeneratorMic__0::ch0_octave_31",
    "RAWTurbineVib__3::ch0_band_shaft", "RAWTurbineVib__3::ch0_octave_31",
]  # fuse() concatenates audio+vib column NAMES unchanged -- only the VALUES
   # are per-run z-scored (A1.1), so the names alone can't tell fusion apart
   # from a raw-scale variant; the classifier keys on *variant*, not names.


def test_feature_unit_label_fusion_reads_zscore_not_log10() -> None:
    label = ad._feature_unit_label("fusion", _FUSION_STYLE_NAMES)
    assert "z-score" in label
    assert "log10" not in label


def test_feature_unit_label_embedding_variant_reads_embedding_units() -> None:
    label = ad._feature_unit_label("audio-beats", ["beats_0", "beats_1"])
    assert "embedding units" in label
    assert "log10" not in label


def test_feature_unit_label_raw_scale_variant_reads_log10() -> None:
    label = ad._feature_unit_label("audio", _FUSION_STYLE_NAMES)
    assert "log10" in label


# --- item 1: mode-signatures / tonal-table wiring (variant in the title;
# RED: fusion's axis/title reads 'z-score' and never claims 'log10') -------


def test_plot_mode_signatures_fusion_variant_axis_and_title_read_zscore(tmp_path) -> None:
    features = np.vstack([np.zeros((5, 2)), np.full((5, 2), 3.0)])
    gt = np.array((["turbine"] * 5) + (["pump"] * 5), dtype=object)
    table = ad._mode_profile(features, gt, level_cols=[0, 1])
    table = table.copy()
    table["feature"] = [_FUSION_STYLE_NAMES[int(c)] for c in table["column"]]
    ax = ad._plot_mode_signatures(
        table, tmp_path / "fusion.png", "fusion", _FUSION_STYLE_NAMES, top_n=5,
    )
    assert "fusion" in ax.get_title(loc="left")
    assert "z-score" in ax.get_xlabel()
    assert "log10" not in ax.get_xlabel()


def test_plot_mode_signatures_raw_scale_variant_axis_reads_log10(tmp_path) -> None:
    features = np.vstack([np.zeros((5, 2)), np.full((5, 2), 3.0)])
    gt = np.array((["turbine"] * 5) + (["pump"] * 5), dtype=object)
    table = ad._mode_profile(features, gt, level_cols=[0, 1])
    table = table.copy()
    table["feature"] = [_FUSION_STYLE_NAMES[int(c)] for c in table["column"]]
    ax = ad._plot_mode_signatures(
        table, tmp_path / "audio.png", "audio", _FUSION_STYLE_NAMES, top_n=5,
    )
    assert "audio" in ax.get_title(loc="left")
    assert "log10" in ax.get_xlabel()


def test_plot_tonal_table_fusion_variant_title_and_colorbar_read_zscore(tmp_path) -> None:
    rf = _tonal_run("dayA", shaft=-30.0, floor=-45.0)
    table = ad._tonal_table([rf])
    ax = ad._plot_tonal_table(table, tmp_path / "fusion.png", "fusion", _TONAL_NAMES)
    assert "fusion" in ax.get_title(loc="left")
    cbar_label = ax.figure.axes[-1].get_ylabel()
    assert "z-score" in cbar_label
    assert "log10" not in cbar_label


def test_plot_tonal_table_title_says_nearest_non_containing_octave(tmp_path) -> None:
    rf = _tonal_run("dayA", shaft=-30.0, floor=-45.0)
    table = ad._tonal_table([rf])
    ax = ad._plot_tonal_table(table, tmp_path / "audio.png", "audio", _TONAL_NAMES)
    assert "nearest non-containing octave" in ax.get_title(loc="left")
    assert "log10" in ax.figure.axes[-1].get_ylabel()


# --- item 1: digest carries the same caveat, one sentence per section -----


def test_digest_mode_signatures_section_carries_the_zscore_embedding_caveat(tmp_path) -> None:
    text = ad._render_digest(tmp_path / "analysis-days")
    section = text.split("## mode-signatures")[1].split("## tonal-table")[0]
    assert "z-score" in section
    assert "embedding units" in section


def test_digest_tonal_table_section_carries_the_zscore_embedding_caveat(tmp_path) -> None:
    text = ad._render_digest(tmp_path / "analysis-days")
    section = text.split("## tonal-table")[1].split("## pillar3-figure")[0]
    assert "z-score" in section
    assert "embedding units" in section


# --- item 2: tonal-table's nearest NON-CONTAINING octave floor -------------


def test_nearest_octave_hz_prefers_nearest_by_absolute_distance_when_tone_free() -> None:
    # shaft (6.25 Hz): outside every candidate's own span -- unaffected by
    # the new rule, still the plain nearest-by-Hz.
    assert ad._nearest_octave_hz(6.25, [31.5, 63.0, 125.0]) == 31.5


def test_nearest_octave_hz_skips_a_candidate_whose_own_span_contains_the_target() -> None:
    # blade_pass (43.75 Hz): octave 31.5's own span (~[22.3, 44.5]) CONTAINS
    # it -- disqualified; octave 63.0's span (~[44.5, 89.1]) does not.
    assert ad._nearest_octave_hz(43.75, [31.5, 63.0, 125.0]) == 63.0


def test_nearest_octave_hz_skips_an_exact_center_match_too() -> None:
    # guide_vane_pass (125.0 Hz) IS a candidate center -- its own span
    # trivially contains itself, disqualified. Remaining: 63.0 (dist 62) is
    # nearer than 250.0 (dist 125).
    assert ad._nearest_octave_hz(125.0, [31.5, 63.0, 125.0, 250.0]) == 63.0


def test_nearest_octave_hz_falls_back_to_a_containing_candidate_if_none_is_tone_free(
    caplog,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert ad._nearest_octave_hz(125.0, [125.0]) == 125.0
    assert any(
        "125" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    )


def test_nearest_octave_hz_empty_available_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        ad._nearest_octave_hz(6.25, [])


_TONAL_BLADE_NAMES = [
    "RAWGeneratorMic__0::ch0_band_blade_pass",
    "RAWGeneratorMic__0::ch0_octave_31",
    "RAWGeneratorMic__0::ch0_octave_63",
]


def _tonal_blade_run(
    run_name: str, *, blade: float, floor31: float, floor63: float
) -> ad._RunFeatures:
    n = 20
    features = np.zeros((n, 3), dtype=np.float64)
    features[:, 0] = blade
    features[:, 1] = floor31
    features[:, 2] = floor63
    return ad._RunFeatures(
        run_name=run_name, features=features,
        gt_states=np.array(["turbine"] * n, dtype=object),
        segment_ids=np.arange(n) // 5, feature_names=list(_TONAL_BLADE_NAMES), has_gt=True,
    )


def test_tonal_table_blade_pass_skips_the_containing_octave_31() -> None:
    # old behaviour would pick octave 31 (nearest by |Hz|, 12.25) and read
    # floor31=-99.0 (contrast=79.0); the new rule disqualifies it (span
    # contains 43.75) and picks octave 63 instead (floor63=-35.0, contrast=15.0).
    rf = _tonal_blade_run("dayA", blade=-20.0, floor31=-99.0, floor63=-35.0)
    table = ad._tonal_table([rf])
    row = table[table["band"] == "blade_pass"].iloc[0]
    assert row["octave_floor_hz"] == pytest.approx(63.0)
    assert row["octave_floor"] == pytest.approx(-35.0)
    assert row["tonal_contrast"] == pytest.approx(15.0)  # -20.0 - (-35.0)


# --- item 3: `_discover_pillar3_leaves` per-leaf parse guard ---------------


def test_discover_pillar3_leaves_skips_a_corrupt_csv_with_warning(tmp_path, caplog) -> None:
    root = tmp_path / "results" / "pillar3"
    _write_event_eval(
        root / "080726-pu_strikes" / "audio-a0.05",
        n_events=10.0, n_detected=10.0, event_tpr=1.0, realized_window_far=0.06,
    )
    corrupt_leaf = root / "080726-pu_strikes" / "fusion-a0.01"
    corrupt_leaf.mkdir(parents=True)
    # wrong schema entirely (no 'row_type' column at all) -- KeyError, one of
    # the guarded exception types; a plausible real-world truncated/corrupt write.
    pd.DataFrame([{"unexpected": 1}]).to_csv(corrupt_leaf / "event_eval.csv", index=False)
    with caplog.at_level(logging.WARNING):
        table = ad._discover_pillar3_leaves(root)
    assert list(table["representation"]) == ["audio"]  # only the good leaf renders
    assert any(
        "fusion-a0.01" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


# --- item 4: digest prose nits (fold in) ------------------------------------


def test_digest_fusion_finding_points_below_not_above(tmp_path) -> None:
    text = ad._render_digest(tmp_path / "analysis-days")
    assert "figure above" not in text


def test_digest_pillar3_section_annotates_fusion_snorm_as_session_norm_side_arm(
    tmp_path,
) -> None:
    text = ad._render_digest(tmp_path / "analysis-days")
    section = text.split("## pillar3-figure")[1]
    assert "fusion-snorm" in section
    assert "session-norm" in section
    assert "base representation" in section  # explicitly disclaims this framing
