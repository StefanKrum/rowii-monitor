"""Smoke tests for `scripts/run_step2.py` (Step-2 Task S6): synthetic fixture tree, no
real ROWII data anywhere -- mirrors `tests/test_cli_smoke.py`'s established pattern for
`scripts/run_step1.py`.

Fixture: one SCADA-covered "day" is `_N_SEGMENTS` separate burst files per stream (needed
so `rowii.anomaly.references.split_by_segments` has enough segments to produce a
non-degenerate calibration/fit/conformal/scoring split) covering a single, uniform
"turbine" SCADA state throughout. Sizing was verified empirically (scratch script, not
committed -- same practice as `rowii/anomaly/sweep.py`'s own module docstring) to give
the DEFAULT detected-cluster sweep (`SweepConfig.min_ref=20`) at least one non-excluded
label with real top-k candidates under both conditioning modes, so `candidates.md`
sections and register content are non-trivial to assert against, not just "the file
exists".
"""
from __future__ import annotations

import logging
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rowii.anomaly.references import split_by_segments
from rowii.config import Config, load_config
from rowii.pipeline import PreparedRun
from rowii.signals.windows import WindowGrid
from tests.fixtures.gantner_builder import build_gantner_file

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Synthetic fixture tree
# ---------------------------------------------------------------------------

_MIC_RATE_HZ = 200.0
_VIB_RATE_HZ = 100.0
_SCADA_RATE_HZ = 10.0
_N_SEGMENTS = 10
_SEG_SECONDS = 60
_T0_NS = 1_750_000_000_000_000_000  # arbitrary but fixed UTC epoch, ns


def _build_day_tree(day_dir: Path, *, day_label: str, t0_ns: int = _T0_NS) -> None:
    """One SCADA-covered day tree (`<day_dir>/<day_label> Messung/{TU,Betriebsdaten}`):
    `_N_SEGMENTS` contiguous `_SEG_SECONDS`-long burst files per stream (2 mic + 2 vib,
    each its own on-disk file -> its own `PreparedRun.segment_ids` value), and a single
    Betriebsdaten hour whose GT state is uniformly "turbine" for the whole span (nominal
    speed/power, flow_tu > flow_pu -- `rowii.scada.labels._base_state`).
    """
    meas = day_dir / f"{day_label} Messung"
    tu = meas / "TU"
    bd = meas / "Betriebsdaten"
    tu.mkdir(parents=True, exist_ok=True)
    bd.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    seg_ns = int(_SEG_SECONDS * 1e9)
    for seg in range(_N_SEGMENTS):
        seg_t0 = t0_ns + seg * seg_ns
        ts = f"2026-06-25_06-{seg:02d}-00_000000"  # _N_SEGMENTS=10 < 60 -> always valid MM
        for stream_name in ("RAWGeneratorMic__0", "RAWTurbineMic__1"):
            n = int(_MIC_RATE_HZ * _SEG_SECONDS)
            data = rng.normal(0.0, 0.5, size=(n, 4)).astype(np.float32)
            build_gantner_file(
                tu / f"{stream_name}_{ts}.dat", ["ch0", "ch1", "ch2", "ch3"], data,
                t0_ns=seg_t0, rate_hz=_MIC_RATE_HZ,
            )
        for stream_name in ("RAWGeneratorVib__2", "RAWTurbineVib__3"):
            n = int(_VIB_RATE_HZ * _SEG_SECONDS)
            data = rng.normal(0.0, 0.2, size=(n, 2)).astype(np.float32)
            build_gantner_file(
                tu / f"{stream_name}_{ts}.dat", ["chX", "chY"], data,
                t0_ns=seg_t0, rate_hz=_VIB_RATE_HZ,
            )

    total_s = _N_SEGMENTS * _SEG_SECONDS
    n_scada = int(_SCADA_RATE_HZ * total_s)
    power = np.full(n_scada, 10.0, dtype=np.float32)
    speed = np.full(n_scada, 378.832, dtype=np.float32)
    guide_vane = np.full(n_scada, 50.0, dtype=np.float32)
    flow_tu = np.full(n_scada, 5.0, dtype=np.float32)
    flow_pu = np.zeros(n_scada, dtype=np.float32)
    reactive = np.zeros(n_scada, dtype=np.float32)
    ks_valve = np.full(n_scada, 3.0, dtype=np.float32)
    scada_data = np.stack(
        [power, speed, guide_vane, flow_tu, flow_pu, reactive, ks_valve], axis=1
    )
    build_gantner_file(
        bd / "2026-06-25_06-00-00.dat",
        [
            "1_P_Ist", "1_Drehzahl UPM", "1_Leitapparat Stell.", "Durchfluss TU",
            "Durchfluss PU", "1_Q_Ist", "1_KS Stellung",
        ],
        scada_data, t0_ns=t0_ns, rate_hz=_SCADA_RATE_HZ,
    )


def _build_one_day_root(root: Path) -> Path:
    """Single-tree (legacy) data_root -- one SCADA-covered run named "tu"."""
    _build_day_tree(root, day_label="20260625")
    return root


def _build_two_day_root(root: Path) -> Path:
    """Parent-root (multi-day) data_root -- two SCADA-covered runs, "000001-tu" and
    "000002-tu", each a self-contained day tree with its own Betriebsdaten."""
    _build_day_tree(root / "illwerke-000001", day_label="20260625")
    _build_day_tree(root / "illwerke-000002", day_label="20260701")
    return root


def _build_day_tree_too_sparse(day_dir: Path, *, day_label: str, t0_ns: int = _T0_NS) -> None:
    """A day tree with only two 1-second bursts per stream, 60 s apart, instead of
    `_N_SEGMENTS` contiguous ones -- the resulting grid is dominated by the gap between
    them (>5% of windows invalid), reproducing the `RuntimeError`
    `rowii.pipeline.compute_validity_mask` raises for a real, genuinely-short
    "stray file" run that is still SCADA-covered and still discovered as one run (no
    >15-min gap to split on) -- e.g. the real `010726-tu1-afternoon` run (Task S7,
    2026-07-09), which crashed `--protocol cross-day`'s `_run_cross_day` before its
    `prepare_run` call was guarded, since that loop had no `try/except` at all around a
    call the module docstring already documents can raise for other reasons.
    """
    meas = day_dir / f"{day_label} Messung"
    tu = meas / "TU"
    bd = meas / "Betriebsdaten"
    tu.mkdir(parents=True, exist_ok=True)
    bd.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(3)
    burst_s = 1
    gap_s = 60
    for i, offset_s in enumerate((0, burst_s + gap_s)):
        seg_t0 = t0_ns + int(offset_s * 1e9)
        ts = f"2026-06-25_07-{i:02d}-00_000000"
        for stream_name in ("RAWGeneratorMic__0", "RAWTurbineMic__1"):
            n = int(_MIC_RATE_HZ * burst_s)
            data = rng.normal(0.0, 0.5, size=(n, 4)).astype(np.float32)
            build_gantner_file(
                tu / f"{stream_name}_{ts}.dat", ["ch0", "ch1", "ch2", "ch3"], data,
                t0_ns=seg_t0, rate_hz=_MIC_RATE_HZ,
            )
        for stream_name in ("RAWGeneratorVib__2", "RAWTurbineVib__3"):
            n = int(_VIB_RATE_HZ * burst_s)
            data = rng.normal(0.0, 0.2, size=(n, 2)).astype(np.float32)
            build_gantner_file(
                tu / f"{stream_name}_{ts}.dat", ["chX", "chY"], data,
                t0_ns=seg_t0, rate_hz=_VIB_RATE_HZ,
            )

    total_s = burst_s + gap_s + burst_s
    n_scada = int(_SCADA_RATE_HZ * total_s)
    power = np.full(n_scada, 10.0, dtype=np.float32)
    speed = np.full(n_scada, 378.832, dtype=np.float32)
    guide_vane = np.full(n_scada, 50.0, dtype=np.float32)
    flow_tu = np.full(n_scada, 5.0, dtype=np.float32)
    flow_pu = np.zeros(n_scada, dtype=np.float32)
    reactive = np.zeros(n_scada, dtype=np.float32)
    ks_valve = np.full(n_scada, 3.0, dtype=np.float32)
    scada_data = np.stack(
        [power, speed, guide_vane, flow_tu, flow_pu, reactive, ks_valve], axis=1
    )
    build_gantner_file(
        bd / "2026-06-25_07-00-00.dat",
        [
            "1_P_Ist", "1_Drehzahl UPM", "1_Leitapparat Stell.", "Durchfluss TU",
            "Durchfluss PU", "1_Q_Ist", "1_KS Stellung",
        ],
        scada_data, t0_ns=t0_ns, rate_hz=_SCADA_RATE_HZ,
    )


def _build_three_day_root(root: Path) -> Path:
    """Parent-root data_root -- two NORMAL SCADA-covered runs ("000001-tu",
    "000002-tu") plus one run whose own `prepare_run` raises `RuntimeError`
    ("000003-tu", `_build_day_tree_too_sparse`), for cross-day's prepare-failure
    skip-not-crash regression."""
    _build_day_tree(root / "illwerke-000001", day_label="20260625")
    _build_day_tree(root / "illwerke-000002", day_label="20260701")
    _build_day_tree_too_sparse(root / "illwerke-000003", day_label="20260702")
    return root


# ---------------------------------------------------------------------------
# 1. --help exits 0
# ---------------------------------------------------------------------------


def test_run_step2_help_exits_zero(capsys) -> None:
    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--protocol", "--run", "--variant", "--scorer", "--conditioning",
        "--alpha", "--top-k", "--labels", "--states", "--no-cache",
    ):
        assert flag in out, f"missing {flag!r} in --help output"


# ---------------------------------------------------------------------------
# 2. within-day miniature e2e: fusion, detected labels, both conditionings x knn
# ---------------------------------------------------------------------------


def test_within_day_detected_labels_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_one_day_root(tmp_path / "data")

    import run_step2

    exit_code = run_step2.main(
        [
            "--protocol", "within-day", "--variant", "fusion", "--labels", "detected",
            "--conditioning", "all", "--scorer", "knn",
        ]
    )
    assert exit_code == 0

    results_root = tmp_path / "results"
    for conditioning in ("per-state", "pooled"):
        combo_dir = (
            results_root / "step2" / "within-day" / "tu" / "fusion-detected"
            / f"{conditioning}-knn"
        )
        far_table_path = combo_dir / "far_table.csv"
        assert far_table_path.is_file(), f"missing {far_table_path}"
        far_table = pd.read_csv(far_table_path)
        assert len(far_table) > 0
        assert list(far_table.columns) == [
            "label", "n_calibration", "n_scored", "n_alarms", "realized_far",
            "nominal_alpha", "achievable_alpha_floor", "low_confidence", "threshold",
            "excluded",
        ]

        assert (combo_dir / "scores.parquet").is_file()
        scores = pd.read_parquet(combo_dir / "scores.parquet")
        assert list(scores.columns) == ["window", "label", "score", "p_value", "alarm"]

        candidates_path = combo_dir / "candidates.md"
        assert candidates_path.is_file()
        candidates_text = candidates_path.read_text()
        assert "## Label" in candidates_text, (
            f"expected at least one top-k label section in {conditioning}'s "
            f"candidates.md, got:\n{candidates_text}"
        )
        assert "assessment" in candidates_text

    summary_path = results_root / "step2" / "summary.csv"
    assert summary_path.is_file()
    summary = pd.read_csv(summary_path)
    assert len(summary) == 2  # per-state + pooled
    assert set(summary["conditioning"]) == {"per-state", "pooled"}
    assert (summary["run"] == "tu").all()
    assert (summary["variant"] == "fusion").all()
    assert (summary["labels"] == "detected").all()
    assert (summary["scorer"] == "knn").all()
    assert (summary["per_label_count"] > 0).all()

    register_path = results_root / "step2" / "candidate_register.md"
    assert register_path.is_file()
    register_text = register_path.read_text()
    assert "External candidates" in register_text
    assert "Fuelldüse" in register_text
    assert "deck-v3 p.16" in register_text
    assert "operator-confirmed normal (partner)" in register_text
    assert "tu / fusion-detected / per-state-knn" in register_text
    assert "tu / fusion-detected / pooled-knn" in register_text
    assert "own sweep" in register_text
    assert "unreviewed" in register_text


# ---------------------------------------------------------------------------
# 3. gt-labels mode works on the fixture (SCADA channels present)
# ---------------------------------------------------------------------------


def test_within_day_gt_labels_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_one_day_root(tmp_path / "data")

    import run_step2

    exit_code = run_step2.main(
        [
            "--variant", "fusion", "--labels", "gt", "--conditioning", "per-state",
            "--scorer", "knn",
        ]
    )
    assert exit_code == 0

    combo_dir = (
        tmp_path / "results" / "step2" / "within-day" / "tu" / "fusion-gt"
        / "per-state-knn"
    )
    far_table = pd.read_csv(combo_dir / "far_table.csv")
    # The whole synthetic run is a single, uniform "turbine" GT state (fixture
    # docstring) -- must appear as a real (non-excluded) label, not "unknown".
    assert "turbine" in set(far_table["label"])
    assert "unknown" not in set(far_table["label"])
    row = far_table.set_index("label").loc["turbine"]
    assert not bool(row["excluded"])
    assert row["n_scored"] > 0

    candidates_text = (combo_dir / "candidates.md").read_text()
    assert "## Label `turbine`" in candidates_text

    summary = pd.read_csv(tmp_path / "results" / "step2" / "summary.csv")
    assert (summary["labels"] == "gt").all()
    assert summary.iloc[0]["per_label_count"] == 1


def test_within_day_gt_labels_skips_run_without_scada(tmp_path, monkeypatch) -> None:
    """A run with no Betriebsdaten at all must be skipped (logged, not crashed) in
    gt-labels mode -- `_run_within_day_for_run`'s explicit guard."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    data_root = tmp_path / "data"
    _build_day_tree(data_root, day_label="20260625")
    # Remove the Betriebsdaten file entirely -- run "tu" now has zero SCADA coverage,
    # but --run explicitly names it (bypassing the SCADA-covered default filter).
    bd_dir = data_root / "20260625 Messung" / "Betriebsdaten"
    for f in bd_dir.iterdir():
        f.unlink()

    import run_step2

    exit_code = run_step2.main(
        ["--run", "tu", "--variant", "fusion", "--labels", "gt", "--scorer", "knn"]
    )
    assert exit_code == 0
    assert not (tmp_path / "results" / "step2" / "within-day").exists()


def test_within_day_skips_run_that_fails_to_prepare(tmp_path, monkeypatch, caplog) -> None:
    """A SCADA-covered run whose `prepare_run` raises `RuntimeError` (too few valid
    windows for this variant, e.g. a real "two stray files" run like
    `010726-tu1-afternoon`) must be logged and skipped -- not crash the whole
    invocation -- exactly like a `run_sweep` `ValueError` already is (module
    docstring). Before this fix, `_run_within_day_for_run` called `prepare_run` with no
    `try/except` at all."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    data_root = tmp_path / "data"
    _build_day_tree_too_sparse(data_root, day_label="20260625")

    import run_step2

    with caplog.at_level(logging.WARNING):
        exit_code = run_step2.main(
            [
                "--run", "tu", "--variant", "fusion", "--labels", "detected",
                "--scorer", "knn", "--conditioning", "all",
            ]
        )
    assert exit_code == 0
    assert not (tmp_path / "results" / "step2" / "within-day").exists()
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("tu" in w and "prepare_run" in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# 4. cross-day with two fixture days -> pooled matrix rows
# ---------------------------------------------------------------------------


def test_cross_day_pooled_matrix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_two_day_root(tmp_path / "data")

    import run_step2

    exit_code = run_step2.main(
        [
            "--protocol", "cross-day", "--variant", "fusion", "--scorer", "knn",
            "--labels", "detected",
        ]
    )
    assert exit_code == 0

    results_root = tmp_path / "results"
    forward_dir = (
        results_root / "step2" / "cross-day" / "fusion-detected"
        / "000001-tu__to__000002-tu" / "knn-pooled"
    )
    backward_dir = (
        results_root / "step2" / "cross-day" / "fusion-detected"
        / "000002-tu__to__000001-tu" / "knn-pooled"
    )
    for out_dir in (forward_dir, backward_dir):
        far_table_path = out_dir / "far_table.csv"
        assert far_table_path.is_file(), f"missing {far_table_path}"
        far_table = pd.read_csv(far_table_path)
        assert list(far_table["label"]) == ["pooled"]
        assert far_table.iloc[0]["n_scored"] > 0
        assert (out_dir / "scores.parquet").is_file()
        assert (out_dir / "candidates.md").is_file()

    summary = pd.read_csv(results_root / "step2" / "summary.csv")
    cross_rows = summary[summary["notes"] == "cross-day pooled"]
    assert len(cross_rows) == 2
    assert set(cross_rows["run"]) == {
        "000001-tu__to__000002-tu", "000002-tu__to__000001-tu",
    }
    assert (cross_rows["conditioning"] == "pooled").all()
    assert (cross_rows["per_label_count"] == 1).all()

    register_text = (results_root / "step2" / "candidate_register.md").read_text()
    assert "000001-tu__to__000002-tu / fusion-detected / pooled-knn" in register_text


def test_cross_day_skips_pairs_within_the_same_day(tmp_path, monkeypatch) -> None:
    """Two SCADA-covered runs sharing one day tree (same `Run.day_root`) must never be
    paired as a "cross-day" -- `_run_cross_day`'s `day_root` equality guard."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    data_root = tmp_path / "data"
    _build_day_tree(data_root, day_label="20260625")
    # A second session folder ("PU") under the SAME day tree -> a second run sharing
    # this one day's Betriebsdaten/day_root.
    meas = data_root / "20260625 Messung"
    pu = meas / "PU"
    pu.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for stream_name, rate, n_ch in (
        ("RAWGeneratorMic__0", _MIC_RATE_HZ, 4), ("RAWTurbineMic__1", _MIC_RATE_HZ, 4),
        ("RAWGeneratorVib__2", _VIB_RATE_HZ, 2), ("RAWTurbineVib__3", _VIB_RATE_HZ, 2),
    ):
        n = int(rate * _SEG_SECONDS)
        data = rng.normal(0.0, 0.5, size=(n, n_ch)).astype(np.float32)
        build_gantner_file(
            pu / f"{stream_name}_2026-06-25_09-00-00_000000.dat",
            [f"ch{i}" for i in range(n_ch)], data,
            t0_ns=_T0_NS + int(3 * 3600 * 1e9), rate_hz=rate,
        )

    import run_step2

    from rowii.config import load_config
    from rowii.io.dataset import discover

    cfg = load_config()
    index = discover(data_root)
    assert sorted(r.name for r in index.runs) == ["pu", "tu"]
    assert index.runs[0].day_root == index.runs[1].day_root

    n_written = run_step2._run_cross_day(
        "fusion", cfg, index, ("knn",), "detected", 0.05, 20, use_cache=True,
    )
    assert n_written == 0
    assert not (tmp_path / "results" / "step2" / "cross-day").exists()


def test_cross_day_skips_run_that_fails_to_prepare(tmp_path, monkeypatch, caplog) -> None:
    """A SCADA-covered day whose `prepare_run` raises `RuntimeError` must be excluded
    from cross-day (every pair touching it skipped), while pairs between the OTHER,
    healthy days still get written -- reproduces the real `--protocol cross-day
    --variant fusion --scorer knn` crash Task S7 (2026-07-09) hit against
    `010726-tu1-afternoon` (a real, SCADA-covered, single-run "two stray files" day):
    before this fix, `_run_cross_day`'s `prepared_by_run` prewarm loop had no
    `try/except` around `prepare_run` at all, so ONE bad day crashed the entire
    invocation, losing every other day's matrix cell too.
    """
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_three_day_root(tmp_path / "data")

    import run_step2

    with caplog.at_level(logging.WARNING):
        exit_code = run_step2.main(
            [
                "--protocol", "cross-day", "--variant", "fusion", "--scorer", "knn",
                "--labels", "detected",
            ]
        )
    assert exit_code == 0

    results_root = tmp_path / "results"
    summary = pd.read_csv(results_root / "step2" / "summary.csv")
    cross_rows = summary[summary["notes"] == "cross-day pooled"]
    # Only the healthy pair (both directions) -- every pair touching "000003-tu" (the
    # too-sparse day) is skipped, not just silently missing one direction.
    assert set(cross_rows["run"]) == {
        "000001-tu__to__000002-tu", "000002-tu__to__000001-tu",
    }
    assert not any("000003" in run_pair for run_pair in summary["run"])

    forward_dir = (
        results_root / "step2" / "cross-day" / "fusion-detected"
        / "000001-tu__to__000002-tu" / "knn-pooled"
    )
    assert forward_dir.is_dir()
    assert not (results_root / "step2" / "cross-day" / "fusion-detected"
                / "000003-tu__to__000001-tu").exists()
    assert not (results_root / "step2" / "cross-day" / "fusion-detected"
                / "000001-tu__to__000003-tu").exists()

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("000003-tu" in w and "prepare_run" in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# 5. determinism of summary values across two invocations
# ---------------------------------------------------------------------------


def test_within_day_summary_deterministic_across_invocations(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_one_day_root(tmp_path / "data")

    import run_step2

    argv = [
        "--variant", "fusion", "--labels", "detected", "--conditioning", "per-state",
        "--scorer", "knn",
    ]
    assert run_step2.main(argv) == 0
    assert run_step2.main(argv) == 0

    summary = pd.read_csv(tmp_path / "results" / "step2" / "summary.csv")
    # Append-only: two invocations of one combo -> two rows, not deduplicated.
    assert len(summary) == 2

    first, second = summary.iloc[0], summary.iloc[1]
    for col in summary.columns:
        v1, v2 = first[col], second[col]
        if isinstance(v1, float) and math.isnan(v1):
            assert isinstance(v2, float) and math.isnan(v2), f"{col}: {v1!r} != {v2!r}"
        else:
            assert v1 == v2, f"{col}: {v1!r} != {v2!r}"


# ---------------------------------------------------------------------------
# 6. far_table.csv / scores.parquet label-dtype consistency (round-trip merge)
# ---------------------------------------------------------------------------


def test_per_state_far_table_and_scores_label_dtypes_merge_cleanly(tmp_path, monkeypatch) -> None:
    """`conditioning="per-state"` detected-labels sweeps carry `run_sweep`'s own
    aggregate `label="pooled"` row (`rowii.anomaly.sweep` module docstring point 4)
    alongside int cluster-id rows -- an OBJECT `far_table["label"]` column mixing
    Python `int` and `str` that `to_csv`/`read_csv` round-trips as all-string (pandas
    cannot partially infer a numeric dtype once any one value fails conversion), while
    `scores.parquet`'s own label column never contains the "pooled" row and so
    preserves the ORIGINAL int64 dtype unchanged. Before this fix, `pd.merge(scores,
    far_table, on="label")` on the two re-loaded artifacts raised a `ValueError` for a
    label-dtype mismatch (pandas: "You are trying to merge on int64 and object/str
    columns") even though both files describe exactly the same labels (S6 review
    finding).
    """
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_one_day_root(tmp_path / "data")

    import run_step2

    from rowii.config import load_config
    from rowii.io.dataset import discover

    cfg = load_config()
    index = discover(tmp_path / "data")
    run = next(r for r in index.runs if r.name == "tu")

    n_written = run_step2._run_within_day_for_run(
        run, "fusion", cfg, index, ("knn",), ("per-state",), "detected", 0.05, 20,
        use_cache=True,
    )
    assert n_written == 1

    combo_dir = (
        tmp_path / "results" / "step2" / "within-day" / "tu" / "fusion-detected"
        / "per-state-knn"
    )
    far_table = pd.read_csv(combo_dir / "far_table.csv")
    scores = pd.read_parquet(combo_dir / "scores.parquet")

    # Sanity: the fixture must genuinely exercise the mixed-source aggregate row and a
    # non-empty scores table, else the regression this test guards against can't occur.
    assert "pooled" in set(far_table["label"])
    assert len(scores) > 0

    # (pandas >= 2.x may back a string column with its own dedicated `StringDtype`
    # rather than legacy `object` -- `is_string_dtype` recognizes both.)
    assert pd.api.types.is_string_dtype(far_table["label"])
    assert all(isinstance(v, str) for v in far_table["label"])
    assert pd.api.types.is_string_dtype(scores["label"])
    assert all(isinstance(v, str) for v in scores["label"])

    merged = pd.merge(scores, far_table, on="label", how="inner", suffixes=("", "_far"))
    assert len(merged) == len(scores)


# ---------------------------------------------------------------------------
# 7. summary.csv crash-safety: corrupt-file recovery + atomic write
# ---------------------------------------------------------------------------


def test_append_summary_row_recovers_from_unbalanced_quote(tmp_path, monkeypatch, caplog) -> None:
    """A `summary.csv` left behind by a crashed/killed prior invocation mid-write can be
    malformed CSV (e.g. an unbalanced quote from a partially flushed field) --
    `pd.read_csv` then raises `pandas.errors.ParserError`. Before this fix,
    `_append_summary_row` let that exception propagate uncaught, losing every prior
    invocation's rows and crashing the CURRENT one on a shared, append-only artifact
    (S6 review finding). The corrupt file must be quarantined, never deleted.
    """
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    import run_step2

    step2_dir = tmp_path / "results" / "step2"
    step2_dir.mkdir(parents=True)
    summary_path = step2_dir / "summary.csv"
    corrupt_content = (
        'run,variant,labels,conditioning,scorer,alpha\n"tu,fusion,detected,pooled,knn,0.05\n'
    )
    summary_path.write_text(corrupt_content)

    row = run_step2._SummaryRow(
        run="tu", protocol="within-day", variant="fusion", labels="detected",
        conditioning="pooled", scorer="knn", alpha=0.05, per_label_count=1,
        pooled_realized_far=0.1, mean_per_state_far=0.1, n_low_confidence=0, notes="",
    )
    with caplog.at_level(logging.WARNING):
        run_step2._append_summary_row(tmp_path / "results", row)

    corrupt_files = list(step2_dir.glob("summary.csv.corrupt-*"))
    assert len(corrupt_files) == 1, corrupt_files
    assert corrupt_files[0].read_text() == corrupt_content  # never deleted, byte-identical
    assert not (step2_dir / "summary.csv.tmp").exists()  # tmp write cleaned up via os.replace

    new_summary = pd.read_csv(summary_path)
    assert list(new_summary.columns) == list(run_step2._SUMMARY_COLUMNS)
    assert len(new_summary) == 1
    assert new_summary.iloc[0]["run"] == "tu"

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(str(summary_path) in w for w in warnings), warnings


def test_append_summary_row_recovers_from_truncated_header(tmp_path, monkeypatch, caplog) -> None:
    """A `summary.csv` truncated mid-write can lose part of the HEADER line itself (as
    opposed to a data row) -- `pd.read_csv` then parses SILENTLY (no exception raised),
    just with the wrong, truncated set of columns, since pandas has nothing of its own
    to compare against. `_append_summary_row` must catch this via an explicit
    columns-match check rather than relying on `pd.read_csv` to raise (S6 review
    finding).
    """
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    import run_step2

    step2_dir = tmp_path / "results" / "step2"
    step2_dir.mkdir(parents=True)
    summary_path = step2_dir / "summary.csv"
    truncated_content = "run,variant,labels,condit"  # header itself cut off mid-write
    summary_path.write_text(truncated_content)

    # Confirm the premise: this parses without raising, just with the wrong columns --
    # else this test would not actually exercise the "parses silently" case.
    silently_parsed = pd.read_csv(summary_path)
    assert list(silently_parsed.columns) != list(run_step2._SUMMARY_COLUMNS)

    row = run_step2._SummaryRow(
        run="tu", protocol="within-day", variant="fusion", labels="detected",
        conditioning="pooled", scorer="knn", alpha=0.05, per_label_count=1,
        pooled_realized_far=0.1, mean_per_state_far=0.1, n_low_confidence=0, notes="",
    )
    with caplog.at_level(logging.WARNING):
        run_step2._append_summary_row(tmp_path / "results", row)

    corrupt_files = list(step2_dir.glob("summary.csv.corrupt-*"))
    assert len(corrupt_files) == 1, corrupt_files
    assert corrupt_files[0].read_text() == truncated_content  # never deleted

    new_summary = pd.read_csv(summary_path)
    assert list(new_summary.columns) == list(run_step2._SUMMARY_COLUMNS)
    assert len(new_summary) == 1

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(str(summary_path) in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# 8. candidate_register.md crash-safety: truncated-header repair
# ---------------------------------------------------------------------------


def test_append_candidate_register_repairs_truncated_header(tmp_path, monkeypatch, caplog) -> None:
    """A `candidate_register.md` left behind by a crashed/killed prior invocation can
    have its HEADER itself cut off mid-write (OS writes are not atomic at arbitrary
    byte offsets) -- `path.exists()` is still `True`, so the old `not path.exists()`
    header-once guard never notices and would silently append a new section onto a
    header-less/garbled file. `_append_candidate_register` must instead check whether
    the file actually STARTS WITH the header's first line, and repair (quarantine +
    rewrite fresh) if not (S6 review finding).
    """
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    import run_step2

    step2_dir = tmp_path / "results" / "step2"
    step2_dir.mkdir(parents=True)
    register_path = step2_dir / "candidate_register.md"
    first_line = run_step2._REGISTER_HEADER.splitlines()[0]
    # Truncated mid-write of the header LINE itself (OS writes are not atomic at
    # arbitrary byte offsets) -- the realistic crash scenario the first-line check
    # targets; a truncation landing anywhere past the first line is out of this
    # check's scope (helper docstring's documented gap).
    truncated_content = first_line[: len(first_line) // 2]
    register_path.write_text(truncated_content)

    candidates = pd.DataFrame(columns=["window", "label", "score", "p_value", "rank"])
    grid = run_step2.WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=1)

    with caplog.at_level(logging.WARNING):
        run_step2._append_candidate_register(
            tmp_path / "results", "tu", "fusion", "detected", "per-state", "knn", 0.05,
            candidates, grid, None,
        )

    corrupt_files = list(step2_dir.glob("candidate_register.md.corrupt-*"))
    assert len(corrupt_files) == 1, corrupt_files
    assert corrupt_files[0].read_text() == truncated_content  # never deleted
    assert not (step2_dir / "candidate_register.md.tmp").exists()

    text = register_path.read_text()
    assert text.count("Fuelldüse") == 1  # header rewritten fresh, exactly once
    assert "tu / fusion-detected / per-state-knn" in text

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(str(register_path) in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# 9. cross-day-per-state protocol (Task 3, package-2 spec D2)
# ---------------------------------------------------------------------------


class TestCrossDayPerStateSweep:
    """Unit coverage of `run_step2._cross_day_per_state_sweep` directly, against
    hand-built `PreparedRun`s -- mirrors `tests/test_sweep.py`'s own `_prepared_run`
    fixture-construction style, since this sweep composes the same public row builders
    `run_sweep` does (Task 3 Step 1's rename). The CLI-level guard test at the bottom
    goes through `run_step2.main` on this file's usual synthetic-tree fixtures instead,
    since it exercises argument parsing, not the sweep itself.
    """

    def _make_prepared(
        self,
        seed: int,
        n_segments: int = 8,
        seg_len: int = 40,
        order: tuple[int, ...] = (0, 1),
    ) -> PreparedRun:
        """Two well-separated 'states' (feature values 0 and 5), alternating by
        segment so `split_by_segments` has material on both sides for both labels."""
        rng = np.random.default_rng(seed)
        feats: list[np.ndarray] = []
        seg_ids: list[np.ndarray] = []
        for s in range(n_segments):
            value = 5.0 * order[s % len(order)]
            feats.append(rng.normal(value, 0.1, (seg_len, 2)))
            seg_ids.append(np.full(seg_len, s, dtype=np.int64))
        features = np.vstack(feats)
        n = len(features)
        return PreparedRun(
            features=features,
            grid=WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=n),
            valid_mask=np.ones(n, dtype=bool),
            feature_names=["f0", "f1"],
            segment_ids=np.concatenate(seg_ids),
        )

    def _rowii_cfg(self) -> Config:
        cfg = load_config()  # follow the file's existing config-construction pattern
        return replace(cfg, detect=replace(cfg.detect, n_states=2, min_dwell_s=3))

    def test_per_state_rows_and_far_control_on_shifted_day(self) -> None:
        prepared_a = self._make_prepared(seed=0)
        prepared_b = self._make_prepared(seed=1)  # same distribution, new draws

        import run_step2

        result, labels_b = run_step2._cross_day_per_state_sweep(
            prepared_a, prepared_a.valid_mask, prepared_b, prepared_b.valid_mask,
            self._rowii_cfg(), "knn", alpha=0.10, top_k=5,
        )
        far = result.far_table
        real_rows = far[(far["label"] != "pooled") & (~far["excluded"])]
        assert len(real_rows) == 2  # both fit-day states got references + scoring
        # Exchangeable B => realized FAR near alpha for every state (loose gate;
        # per-rep FAR is Beta-distributed, not exact)
        assert (real_rows["realized_far"] < 0.35).all()
        # aggregate row present, labeled like run_sweep's own convention
        assert (far["label"] == "pooled").sum() == 1
        assert labels_b.shape == (prepared_b.features.shape[0],)

    def test_day_b_windows_keyed_by_predicted_state(self) -> None:
        prepared_a = self._make_prepared(seed=0, order=(0, 1))
        # Day B: 6 of 8 segments are state "5.0", 2 are state "0.0" -- proportions
        # must show up in per-label n_scored via PREDICTED labels
        prepared_b = self._make_prepared(seed=2, order=(1, 1, 1, 0))

        import run_step2

        result, labels_b = run_step2._cross_day_per_state_sweep(
            prepared_a, prepared_a.valid_mask, prepared_b, prepared_b.valid_mask,
            self._rowii_cfg(), "knn", alpha=0.10, top_k=5,
        )
        far = result.far_table
        real_rows = far[(far["label"] != "pooled") & (~far["excluded"])]
        counts = dict(zip(real_rows["label"], real_rows["n_scored"], strict=True))
        assert max(counts.values()) >= 2.5 * min(counts.values())
        # Runtime-honest (spec D2): day B's predicted labels live in day A's own
        # cluster-id space (here {0, 1}, n_states=2) -- never a day-B-only concept.
        assert set(np.unique(labels_b[labels_b >= 0])) <= {0, 1}

    def test_gt_labels_mode_rejected(self, tmp_path, monkeypatch, capsys) -> None:
        """CLI-level guard (spec D2: the transfer runtime path is detected-labels
        only) -- `--protocol cross-day-per-state` combined with `--labels gt` must be
        rejected before any run is ever prepared. `parser.error(...)` is the simplest,
        clearest rejection (orchestrator resolution 2): argparse writes a usage message
        to stderr and raises `SystemExit(2)`, the same exit-code contract this file's
        own `test_run_step2_help_exits_zero` already asserts for `--help` (exit 0).
        """
        monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
        monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
        (tmp_path / "data").mkdir()  # must exist for discover() -- empty is fine, the
        # guard fires before any run is ever looked up

        import run_step2

        with pytest.raises(SystemExit) as exc_info:
            run_step2.main(
                ["--protocol", "cross-day-per-state", "--labels", "gt", "--variant", "fusion"]
            )
        assert exc_info.value.code == 2

        err = capsys.readouterr().err
        assert "cross-day-per-state" in err
        assert "detected-labels only" in err


def test_cross_day_per_state_end_to_end_and_summary_backfill(tmp_path, monkeypatch) -> None:
    """CLI-level end-to-end run of `--protocol cross-day-per-state`, on the SAME
    two-day synthetic tree `test_cross_day_pooled_matrix` uses (Task 3 Step 5): the
    usual three per-combo files plus the GT-diagnostic `far_by_true_state.csv` exist
    for both pair directions, `summary.csv` gains real `"cross-day-per-state"` rows,
    AND a `summary.csv` already on disk in the OLD (pre-package-2, no `protocol`
    column) schema is backfilled -- not quarantined -- when this run's own append
    reads it back (`_read_summary_csv_or_none`'s backfill path).
    """
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_two_day_root(tmp_path / "data")

    import run_step2

    # Pre-seed a legacy (pre-package-2) summary.csv: one within-day-shaped row (bare
    # run name) and one cross-day-shaped row (`"__to__"` pair encoding), so this run's
    # own append exercises BOTH branches of `_infer_legacy_protocol`.
    results_root = tmp_path / "results"
    step2_dir = results_root / "step2"
    step2_dir.mkdir(parents=True)
    legacy_columns = list(run_step2._SUMMARY_COLUMNS_LEGACY)
    legacy_rows = pd.DataFrame(
        [
            {
                "run": "tu", "variant": "fusion", "labels": "detected",
                "conditioning": "pooled", "scorer": "knn", "alpha": 0.05,
                "per_label_count": 1, "pooled_realized_far": 0.05,
                "mean_per_state_far": 0.05, "n_low_confidence": 0, "notes": "",
            },
            {
                "run": "000001-tu__to__000002-tu", "variant": "fusion",
                "labels": "detected", "conditioning": "pooled", "scorer": "knn",
                "alpha": 0.05, "per_label_count": 1, "pooled_realized_far": 0.07,
                "mean_per_state_far": 0.07, "n_low_confidence": 0,
                "notes": "cross-day pooled",
            },
        ],
        columns=legacy_columns,
    )
    legacy_rows.to_csv(step2_dir / "summary.csv", index=False)

    exit_code = run_step2.main(
        ["--protocol", "cross-day-per-state", "--variant", "fusion", "--scorer", "knn"]
    )
    assert exit_code == 0

    forward_dir = (
        results_root / "step2" / "cross-day-per-state" / "000001-tu--to--000002-tu"
        / "fusion-knn"
    )
    backward_dir = (
        results_root / "step2" / "cross-day-per-state" / "000002-tu--to--000001-tu"
        / "fusion-knn"
    )
    for out_dir in (forward_dir, backward_dir):
        far_table_path = out_dir / "far_table.csv"
        assert far_table_path.is_file(), f"missing {far_table_path}"
        far_table = pd.read_csv(far_table_path)
        assert len(far_table) > 0

        assert (out_dir / "scores.parquet").is_file()
        assert (out_dir / "candidates.md").is_file()

        far_by_true_state_path = out_dir / "far_by_true_state.csv"
        assert far_by_true_state_path.is_file(), f"missing {far_by_true_state_path}"
        far_by_true_state = pd.read_csv(far_by_true_state_path)
        assert list(far_by_true_state.columns) == [
            "true_state", "n_scored", "n_alarms", "realized_far",
        ]

    summary = pd.read_csv(results_root / "step2" / "summary.csv")
    assert "protocol" in summary.columns
    new_rows = summary[summary["protocol"] == "cross-day-per-state"]
    assert len(new_rows) == 2  # forward + backward pair
    assert set(new_rows["run"]) == {
        "000001-tu--to--000002-tu", "000002-tu--to--000001-tu",
    }
    assert (new_rows["labels"] == "detected").all()
    assert (new_rows["conditioning"] == "per-state").all()
    assert (new_rows["notes"] == "cross-day-per-state transfer").all()

    # Backfill path (Step 5): the two PRE-EXISTING legacy rows survive this run's
    # append, unchanged apart from the newly-backfilled "protocol" value.
    old_rows = summary[summary["protocol"] != "cross-day-per-state"].set_index(
        "run", drop=False
    )
    assert len(old_rows) == 2
    assert old_rows.loc["tu", "protocol"] == "within-day"
    assert old_rows.loc["000001-tu__to__000002-tu", "protocol"] == "cross-day"
    legacy_by_run = legacy_rows.set_index("run", drop=False)
    for run_name in ("tu", "000001-tu__to__000002-tu"):
        for col in legacy_columns:
            actual = old_rows.loc[run_name, col]
            expected = legacy_by_run.loc[run_name, col]
            # CSV round-trip quirk (unrelated to the backfill logic under test): an
            # empty-string field (e.g. the "tu" row's own blank "notes") reads back as
            # NaN, not "" -- both count as "unchanged" here.
            if pd.isna(actual) and (expected == "" or pd.isna(expected)):
                continue
            assert actual == expected, (
                f"backfilled row for {run_name!r} changed column {col!r}: "
                f"{actual!r} != {expected!r}"
            )

    register_text = (results_root / "step2" / "candidate_register.md").read_text()
    assert "000001-tu--to--000002-tu / fusion-detected / per-state-knn" in register_text


# ---------------------------------------------------------------------------
# 10. --run scoping: comma lists + cross-day pair filtering (Task 3 follow-up)
# ---------------------------------------------------------------------------


def test_cross_day_per_state_run_list_produces_exactly_the_listed_pairs(
    tmp_path, monkeypatch
) -> None:
    """A 2-name `--run` list for `--protocol cross-day-per-state` produces exactly the
    2 ordered pairs among the named runs -- no other pair directory exists."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_two_day_root(tmp_path / "data")

    import run_step2

    exit_code = run_step2.main(
        [
            "--protocol", "cross-day-per-state", "--variant", "fusion",
            "--scorer", "knn", "--run", "000001-tu,000002-tu",
        ]
    )
    assert exit_code == 0

    root = tmp_path / "results" / "step2" / "cross-day-per-state"
    assert sorted(p.name for p in root.iterdir()) == [
        "000001-tu--to--000002-tu", "000002-tu--to--000001-tu",
    ]
    for pair_dir_name in ("000001-tu--to--000002-tu", "000002-tu--to--000001-tu"):
        assert (root / pair_dir_name / "fusion-knn" / "far_table.csv").is_file()


def test_cross_day_run_list_excludes_unlisted_day_before_prepare(
    tmp_path, monkeypatch, caplog
) -> None:
    """The `--run` list restricts the cross-day PAIR SET itself, not just which outputs
    get asserted: on the three-day root, "000003-tu" is SCADA-covered but too sparse to
    prepare -- under `--run all` its `prepare_run` failure warning is exactly what
    `test_cross_day_skips_run_that_fails_to_prepare` asserts EXISTS, so its absence
    here proves the unlisted day was filtered out BEFORE `prepare_run` was ever
    attempted, not merely skipped after failing."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_three_day_root(tmp_path / "data")

    import run_step2

    with caplog.at_level(logging.WARNING):
        exit_code = run_step2.main(
            [
                "--protocol", "cross-day", "--variant", "fusion", "--scorer", "knn",
                "--labels", "detected", "--run", "000001-tu,000002-tu",
            ]
        )
    assert exit_code == 0

    cross_root = tmp_path / "results" / "step2" / "cross-day" / "fusion-detected"
    assert sorted(p.name for p in cross_root.iterdir()) == [
        "000001-tu__to__000002-tu", "000002-tu__to__000001-tu",
    ]
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("000003" in w for w in warnings), warnings

    summary = pd.read_csv(tmp_path / "results" / "step2" / "summary.csv")
    assert not any("000003" in run_pair for run_pair in summary["run"])


@pytest.mark.parametrize("protocol", ["cross-day", "cross-day-per-state"])
def test_cross_day_single_run_name_rejected(tmp_path, monkeypatch, capsys, protocol) -> None:
    """A 1-name `--run` list for a cross-day protocol has no pairs to sweep -- must be
    rejected up front (exit 2, `parser.error`) rather than silently writing nothing."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    (tmp_path / "data").mkdir()  # defensive: never reached if the guard fires first

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(
            ["--protocol", protocol, "--variant", "fusion", "--run", "000001-tu"]
        )
    assert exc_info.value.code == 2

    err = capsys.readouterr().err
    assert "no cross-day pairs" in err


def test_unknown_run_name_exits_2_listing_available(tmp_path, monkeypatch, capsys) -> None:
    """An unknown name anywhere in the `--run` list is a hard usage error (exit 2,
    naming every discovered run) -- mirrors `scripts/warm_cache.py`'s own unknown-run
    precedent; before this change, within-day logged a warning and exited 0 having
    processed nothing."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_two_day_root(tmp_path / "data")

    import run_step2

    exit_code = run_step2.main(
        ["--run", "000001-tu,bogus", "--variant", "fusion", "--scorer", "knn"]
    )
    assert exit_code == 2

    err = capsys.readouterr().err
    assert "bogus" in err
    assert "000001-tu" in err
    assert "000002-tu" in err
    assert not (tmp_path / "results" / "step2").exists()  # nothing was written


def test_within_day_run_list_processes_each_named_run(tmp_path, monkeypatch) -> None:
    """A comma-separated `--run` list for within-day processes each named run in
    sequence -- the natural extension of the pre-existing single-name behavior."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_two_day_root(tmp_path / "data")

    import run_step2

    exit_code = run_step2.main(
        [
            "--run", "000001-tu,000002-tu", "--variant", "fusion",
            "--labels", "detected", "--conditioning", "per-state", "--scorer", "knn",
        ]
    )
    assert exit_code == 0

    for run_name in ("000001-tu", "000002-tu"):
        combo_dir = (
            tmp_path / "results" / "step2" / "within-day" / run_name
            / "fusion-detected" / "per-state-knn"
        )
        assert (combo_dir / "far_table.csv").is_file(), f"missing outputs for {run_name}"

    summary = pd.read_csv(tmp_path / "results" / "step2" / "summary.csv")
    assert list(summary["run"]) == ["000001-tu", "000002-tu"]  # in the given order


# ---------------------------------------------------------------------------
# 11. --score-fusion view (Task 5, spec D5)
# ---------------------------------------------------------------------------


def test_score_fusion_view_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_one_day_root(tmp_path / "data")

    import run_step2

    exit_code = run_step2.main(
        [
            "--protocol", "within-day", "--variant", "fusion", "--labels", "detected",
            "--conditioning", "per-state", "--scorer", "knn", "--score-fusion",
        ]
    )
    assert exit_code == 0

    combo_dir = (
        tmp_path / "results" / "step2" / "within-day" / "tu" / "fusion-detected"
        / "per-state-knn"
    )
    sf_path = combo_dir / "far_table_scorefusion.csv"
    assert sf_path.is_file(), f"missing {sf_path}"
    sf_table = pd.read_csv(sf_path)
    assert list(sf_table.columns) == [
        "rule", "label", "n_calibration", "n_scored", "n_alarms", "realized_far",
        "low_confidence",
    ]
    assert set(sf_table["rule"]) == {"fisher", "tippett", "audio-only", "vib-only"}
    # every label appears exactly once per rule
    for rule in ("fisher", "tippett", "audio-only", "vib-only"):
        assert (sf_table["rule"] == rule).sum() == sf_table["label"].nunique()

    # the fixture was sized (file docstring, top of file) to give run_sweep's default
    # min_ref=20 at least one non-excluded label -- score-fusion reuses the SAME
    # split/min_ref, so at least one row must be a real (non-low-confidence, actually
    # scored) row, not just NaN placeholders everywhere.
    real_rows = sf_table[~sf_table["low_confidence"]]
    assert len(real_rows) > 0
    assert (real_rows["n_scored"] > 0).any()

    notes_path = combo_dir / "scorefusion_notes.md"
    assert notes_path.is_file(), f"missing {notes_path}"
    notes_text = notes_path.read_text()
    assert "Guarantee restoration" in notes_text
    assert "Honesty notes" in notes_text


def test_score_fusion_requires_within_day_and_fusion_variant(
    tmp_path, monkeypatch, capsys
) -> None:
    """CLI-level guard (orchestrator resolution 4): `--score-fusion` combined with a
    non-`fusion` variant or a non-`within-day` protocol must be rejected before any
    run is ever prepared -- mirrors `test_gt_labels_mode_rejected`'s own
    `parser.error(...)` precedent for `--protocol cross-day-per-state --labels gt`."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    (tmp_path / "data").mkdir()  # guard fires before any run is ever looked up

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(["--score-fusion", "--variant", "audio"])
    assert exc_info.value.code == 2
    # Specific enough to only match main()'s own semantic parser.error (orchestrator
    # resolution 4), not argparse's generic "unrecognized arguments" rejection.
    assert "requires --protocol within-day and --variant fusion" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc_info2:
        run_step2.main(
            ["--score-fusion", "--variant", "fusion", "--protocol", "cross-day"]
        )
    assert exc_info2.value.code == 2
    assert "requires --protocol within-day and --variant fusion" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 12. --states K conditioning-granularity flag (within-day, package-3 Task 6)
# ---------------------------------------------------------------------------


def test_states_flag_conditions_on_k_and_suffixes_out_dir(tmp_path, monkeypatch) -> None:
    """`--states 3` overrides the default k=4 detector granularity for within-day
    detected-label sweeps: the combo out-dir gains a `-k3` suffix on its variant-labels
    segment (so it never collides with -- or overwrites -- the default-K layout), the
    resulting far_table has at most 3 distinct (non-aggregate) detected-state rows, and
    the summary.csv row carries the same suffixed `variant` string (task brief binding
    detail 3)."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_one_day_root(tmp_path / "data")

    import run_step2

    exit_code = run_step2.main(
        [
            "--protocol", "within-day", "--variant", "fusion", "--labels", "detected",
            "--conditioning", "per-state", "--scorer", "knn", "--states", "3",
        ]
    )
    assert exit_code == 0

    results_root = tmp_path / "results"
    combo_dir = (
        results_root / "step2" / "within-day" / "tu" / "fusion-detected-k3"
        / "per-state-knn"
    )
    far_table_path = combo_dir / "far_table.csv"
    assert far_table_path.is_file(), f"missing {far_table_path}"
    far_table = pd.read_csv(far_table_path)
    non_pooled = far_table[far_table["label"] != "pooled"]
    assert len(non_pooled) <= 3

    default_dir = results_root / "step2" / "within-day" / "tu" / "fusion-detected"
    assert not default_dir.exists(), (
        "--states 3 must not write into (or collide with) the default-K layout"
    )

    summary = pd.read_csv(results_root / "step2" / "summary.csv")
    assert (summary["variant"] == "fusion-k3").all()
    assert (summary["labels"] == "detected").all()


def test_states_below_two_rejected(tmp_path, monkeypatch, capsys) -> None:
    """`--states 1` leaves nothing to condition on (need >= 2 clusters) -- rejected up
    front, same `parser.error` pattern as `test_score_fusion_requires_within_day_and_
    fusion_variant`."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    (tmp_path / "data").mkdir()  # guard fires before any run is ever looked up

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(["--states", "1"])
    assert exc_info.value.code == 2
    # Specific enough to only match main()'s own semantic parser.error, not argparse's
    # generic "unrecognized arguments" rejection (which would also exit 2 and also
    # contain the substring "--states" if the flag did not exist at all).
    assert "must be >= 2" in capsys.readouterr().err


@pytest.mark.parametrize("protocol", ["cross-day", "cross-day-per-state"])
def test_states_requires_within_day_protocol(
    tmp_path, monkeypatch, capsys, protocol
) -> None:
    """`--states` conditions the WITHIN-DAY detected-label detector only -- neither
    cross-day protocol ever calls `_detected_labels_and_detector` with a caller-
    supplied `k` (module docstring's cross-day/cross-day-per-state sections), so
    combining `--states` with either is a usage error, same `parser.error` pattern as
    `test_cross_day_single_run_name_rejected`."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    (tmp_path / "data").mkdir()  # guard fires before any run is ever looked up

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(["--protocol", protocol, "--variant", "fusion", "--states", "3"])
    assert exc_info.value.code == 2
    # Specific enough to only match main()'s own semantic parser.error, not argparse's
    # generic "unrecognized arguments" rejection (same reasoning as
    # test_states_below_two_rejected above).
    assert "requires --protocol within-day" in capsys.readouterr().err


def test_states_flag_passes_k_to_fitted_detector_fit(tmp_path, monkeypatch) -> None:
    """`--states K` must actually reach `FittedDetector.fit`'s own `k` parameter --
    the real mechanism that changes how many detected sub-clusters the sweep
    conditions on. Spies on `FittedDetector.fit` (delegating to the real
    implementation, so the sweep still runs correctly) rather than relying only on
    `far_table` row counts, which could coincidentally satisfy the test above even if
    `k` were silently dropped somewhere along the way."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_one_day_root(tmp_path / "data")

    import run_step2

    captured_k: list[int | None] = []
    original_fit = run_step2.FittedDetector.fit

    def _spy_fit(
        features: np.ndarray, grid: WindowGrid, cfg: object, clusterer: str = "kmeans",
        k: int | None = None,
    ) -> object:
        captured_k.append(k)
        return original_fit(features, grid, cfg, clusterer=clusterer, k=k)

    monkeypatch.setattr(run_step2.FittedDetector, "fit", _spy_fit)

    exit_code = run_step2.main(
        [
            "--protocol", "within-day", "--variant", "fusion", "--labels", "detected",
            "--conditioning", "pooled", "--scorer", "knn", "--states", "3",
        ]
    )
    assert exit_code == 0
    assert captured_k == [3]


def test_states_default_produces_unsuffixed_dir(tmp_path, monkeypatch) -> None:
    """No `--states` (the default, `None` -> `cfg.detect.n_states`) must reproduce the
    pre-Task-6 unsuffixed out-dir and summary `variant` string byte-for-byte -- the
    binding byte-compatibility requirement (package-3 Task 6 brief)."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_one_day_root(tmp_path / "data")

    import run_step2

    exit_code = run_step2.main(
        [
            "--protocol", "within-day", "--variant", "fusion", "--labels", "detected",
            "--conditioning", "pooled", "--scorer", "knn",
        ]
    )
    assert exit_code == 0

    results_root = tmp_path / "results"
    combo_dir = (
        results_root / "step2" / "within-day" / "tu" / "fusion-detected" / "pooled-knn"
    )
    assert (combo_dir / "far_table.csv").is_file()

    summary = pd.read_csv(results_root / "step2" / "summary.csv")
    assert (summary["variant"] == "fusion").all()


def test_states_register_sections_disambiguate_k(tmp_path, monkeypatch) -> None:
    """Two invocations of the SAME (run, variant, labels, conditioning, scorer) combo
    at different conditioning granularities (default K, then `--states 3`) must append
    DISTINCT section headers to the append-only candidate register -- otherwise a
    granularity sweep (T7: K=4/8/12 on one run/variant) produces colliding,
    indistinguishable sections in the human-facing review artifact (coordinator
    resolution to Task 6 concern 2). The section identity carries the same `-k<K>`
    suffix convention as the combo out-dir (`_within_day_out_dir`), applied to its
    `<variant>-<labels>` part, and ONLY when `--states` is given -- a default run's
    header stays byte-compatible with the pre-Task-6 register."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_one_day_root(tmp_path / "data")

    import run_step2

    base_args = [
        "--protocol", "within-day", "--variant", "fusion", "--labels", "detected",
        "--conditioning", "pooled", "--scorer", "knn",
    ]
    assert run_step2.main(base_args) == 0
    assert run_step2.main([*base_args, "--states", "3"]) == 0

    register_text = (
        tmp_path / "results" / "step2" / "candidate_register.md"
    ).read_text()
    assert "### tu / fusion-detected / pooled-knn" in register_text
    assert "### tu / fusion-detected-k3 / pooled-knn" in register_text
    # exactly the two combos' own sections, nothing colliding or duplicated
    assert register_text.count("### ") == 2


def test_states_with_gt_labels_rejected(tmp_path, monkeypatch, capsys) -> None:
    """`--states` conditions the DETECTED-labels detector only -- a gt-labels sweep
    never fits a detector at all, so combining `--states` with `--labels gt` would be
    a silent no-op inviting misread results (coordinator resolution to Task 6 concern
    4); rejected up front instead, same `parser.error` pattern as the flag's other two
    guards."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    (tmp_path / "data").mkdir()  # guard fires before any run is ever looked up

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(["--labels", "gt", "--states", "3"])
    assert exc_info.value.code == 2
    # Specific to main()'s own semantic parser.error for THIS flag combination -- the
    # cross-day-per-state gt guard's message also contains "detected-labels only" but
    # cannot fire here (--protocol defaults to within-day).
    assert "--states is detected-labels only" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 13. --ensemble majority-vote view (Task 4, design chapter's committed ensemble:
# OC-SVM + Isolation Forest + LSTM-AE, decision-level majority vote)
# ---------------------------------------------------------------------------


class _MeanDistanceStubScorer:
    """Torch-free stand-in for `LstmAeScorer` (monkeypatched into
    `run_step2._ENSEMBLE_MEMBER_FACTORIES["lstmae"]`): Euclidean distance to the
    fit-time reference's own per-feature mean plays the same "higher = more
    anomalous" reconstruction-error role a real autoencoder would, without any
    actual training or torch import -- deterministic, cheap, and satisfies the
    `Scorer` protocol structurally (`name`/`fit`/`score`)."""

    name = "lstmae"

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None

    def fit(self, reference: np.ndarray) -> _MeanDistanceStubScorer:
        self._mean = reference.mean(axis=0)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        assert self._mean is not None
        return np.linalg.norm(x - self._mean, axis=1)


class _AlarmSetScorer:
    """Test-only stub `Scorer` with FULL, PRECISE control over which windows it
    alarms on: column 0 of every feature row is that window's own absolute index
    (embedded by `_make_ensemble_prepared_pair`, the marker this class reads), and
    `.score()` returns `1.0` for a marker in `alarm_windows`, else `0.0`. The test
    never places an `alarm_windows` marker on a fit/conformal-side window (only ever
    on scoring-side windows, disjoint from calibration by `split_by_segments`'s own
    segment-level partition), so every conformal-side score is `0.0`, `calibrate`'s
    threshold lands at exactly `0.0`, and `score > threshold` reproduces `marker in
    alarm_windows` exactly -- deterministic, no fitting or torch involved.
    """

    name = "stub"

    def __init__(self, alarm_windows: frozenset[int]) -> None:
        self._alarm_windows = alarm_windows

    def fit(self, reference: np.ndarray) -> _AlarmSetScorer:
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        markers = x[:, 0].astype(np.int64)
        return np.array(
            [1.0 if int(m) in self._alarm_windows else 0.0 for m in markers],
            dtype=np.float64,
        )


def _make_ensemble_prepared_pair(
    n_segments: int = 8, seg_len: int = 40
) -> tuple[PreparedRun, PreparedRun, np.ndarray]:
    """A single-label (`label=0` everywhere), grid/segment_ids/valid_mask-sharing
    `PreparedRun` pair for direct `run_step2._run_ensemble_view` unit tests -- mirrors
    `TestCrossDayPerStateSweep._make_prepared`'s hand-built-fixture style. Feature
    column 0 is each window's own absolute index (the marker `_AlarmSetScorer` reads);
    the rest is irrelevant filler. `prepared_variant`/`prepared_logmel` differ only in
    filler-column width -- production `--variant`/`logmel` PreparedRuns differ in real
    feature CONTENT, never in grid/segment_ids/valid_mask, under the `--ensemble`
    grid-equality precondition this pair satisfies by sharing one `grid` object.
    """
    n = n_segments * seg_len
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=n)
    segment_ids = np.concatenate(
        [np.full(seg_len, s, dtype=np.int64) for s in range(n_segments)]
    )
    valid_mask = np.ones(n, dtype=bool)
    marker = np.arange(n, dtype=np.float64)
    variant_features = np.stack([marker, np.zeros(n)], axis=1)
    logmel_features = np.stack([marker, np.zeros(n), np.zeros(n)], axis=1)
    prepared_variant = PreparedRun(
        features=variant_features, grid=grid, valid_mask=valid_mask,
        feature_names=["marker", "f1"], segment_ids=segment_ids,
    )
    prepared_logmel = PreparedRun(
        features=logmel_features, grid=grid, valid_mask=valid_mask,
        feature_names=["marker", "f1", "f2"], segment_ids=segment_ids,
    )
    labels = np.zeros(n, dtype=np.int64)
    return prepared_variant, prepared_logmel, labels


def _make_prepared_pair_with_outliers(
    top_split_seed: int,
    n_segments: int = 12,
    seg_len: int = 30,
    n_outliers: int = 5,
    seed: int = 0,
) -> tuple[PreparedRun, PreparedRun, np.ndarray]:
    """A single-label `PreparedRun` pair with `n_outliers` windows on the TOP-LEVEL
    split's SCORING side (seeded with *top_split_seed*, matching `run_step2.
    _ENSEMBLE_SEED` so the caller and this builder agree on which windows end up
    where) pushed far from the N(0, 0.1) background (+5.0 mean shift) -- built so
    real `OcSvmScorer`/`IsolationForestScorer` members and a realistic
    reconstruction-distance stub all converge on catching the SAME obvious outliers,
    making the majority-vote max-alarms property (task brief binding detail 6b) a
    check on genuine agreement rather than pure-noise small-sample luck. Fit/
    conformal windows (the nested split's own calibration side) are left untouched --
    outliers land ONLY on the top-level scoring side, so every member's own
    calibration stays uncontaminated.
    """
    n = n_segments * seg_len
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=n)
    segment_ids = np.concatenate(
        [np.full(seg_len, s, dtype=np.int64) for s in range(n_segments)]
    )
    valid_mask = np.ones(n, dtype=bool)

    rng = np.random.default_rng(seed)
    base_variant = rng.normal(0.0, 0.1, size=(n, 4))
    base_logmel = rng.normal(0.0, 0.1, size=(n, 6))

    top = split_by_segments(segment_ids, valid_mask, 0.5, top_split_seed)
    outlier_windows = top.scoring_windows[:n_outliers]
    assert outlier_windows.size == n_outliers, (
        "fixture too small for the requested outlier count -- grow n_segments/seg_len"
    )

    variant_features = base_variant.copy()
    variant_features[outlier_windows] += 5.0
    logmel_features = base_logmel.copy()
    logmel_features[outlier_windows] += 5.0

    prepared_variant = PreparedRun(
        features=variant_features, grid=grid, valid_mask=valid_mask,
        feature_names=["a", "b", "c", "d"], segment_ids=segment_ids,
    )
    prepared_logmel = PreparedRun(
        features=logmel_features, grid=grid, valid_mask=valid_mask,
        feature_names=["p", "q", "r", "s", "t", "u"], segment_ids=segment_ids,
    )
    labels = np.zeros(n, dtype=np.int64)
    return prepared_variant, prepared_logmel, labels


def _patch_ensemble_members_ocsvm_iforest_stub_lstmae(monkeypatch, run_step2) -> None:
    """Shared monkeypatch for every `--ensemble` test below: real `OcSvmScorer`/
    `IsolationForestScorer` (fast, torch-free already) plus `_MeanDistanceStubScorer`
    standing in for the torch-dependent `LstmAeScorer` -- task brief's own seam
    (`_ENSEMBLE_MEMBER_FACTORIES`, module-level dict, replaced wholesale)."""
    monkeypatch.setattr(
        run_step2, "_ENSEMBLE_MEMBER_FACTORIES",
        {
            "ocsvm": run_step2.OcSvmScorer,
            "iforest": run_step2.IsolationForestScorer,
            "lstmae": _MeanDistanceStubScorer,
        },
    )


def test_ensemble_view_end_to_end(tmp_path, monkeypatch) -> None:
    """(task brief binding detail 6a) `--ensemble` on the existing synthetic fixture:
    exit 0, `far_table_ensemble.csv` exists with all four `member` values, and the
    honesty note is present. Also exercises binding detail 4's "write ONCE, not per
    combo" placement decision: `--conditioning all` sweeps 2 combos, but exactly one
    `ensemble/` directory is written, not one per combo dir.
    """
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_one_day_root(tmp_path / "data")

    import run_step2

    _patch_ensemble_members_ocsvm_iforest_stub_lstmae(monkeypatch, run_step2)

    exit_code = run_step2.main(
        [
            "--protocol", "within-day", "--variant", "fusion", "--labels", "detected",
            "--conditioning", "all", "--scorer", "knn", "--ensemble",
        ]
    )
    assert exit_code == 0

    results_root = tmp_path / "results"
    ensemble_dir = (
        results_root / "step2" / "within-day" / "tu" / "fusion-detected" / "ensemble"
    )
    far_table_path = ensemble_dir / "far_table_ensemble.csv"
    assert far_table_path.is_file(), f"missing {far_table_path}"
    far_table = pd.read_csv(far_table_path)
    assert list(far_table.columns) == [
        "member", "label", "n_calibration", "n_scored", "n_alarms", "realized_far",
        "low_confidence",
    ]
    assert set(far_table["member"]) == {"ocsvm", "iforest", "lstmae", "ENSEMBLE"}

    # Fixture sized (file docstring) to give run_sweep's default min_ref=20 a real
    # non-excluded label -- the ensemble view shares that same split/min_ref, so at
    # least one row must be real (not low-confidence, actually scored).
    real_rows = far_table[~far_table["low_confidence"]]
    assert len(real_rows) > 0
    assert (real_rows["n_scored"] > 0).any()

    notes_path = ensemble_dir / "ensemble_notes.md"
    assert notes_path.is_file(), f"missing {notes_path}"
    notes_text = notes_path.read_text()
    assert "marginal" in notes_text
    assert "empirical" in notes_text.lower()
    assert "majority" in notes_text.lower()

    # Conditioning/scorer-independent placement: --conditioning all wrote 2 normal
    # sweep combos, but the ensemble view must exist exactly once, at the base
    # variant-labels level, not duplicated into either combo dir.
    per_state_dir = (
        results_root / "step2" / "within-day" / "tu" / "fusion-detected"
        / "per-state-knn"
    )
    pooled_dir = (
        results_root / "step2" / "within-day" / "tu" / "fusion-detected" / "pooled-knn"
    )
    assert per_state_dir.is_dir()
    assert pooled_dir.is_dir()
    assert not (per_state_dir / "far_table_ensemble.csv").exists()
    assert not (pooled_dir / "far_table_ensemble.csv").exists()


def test_ensemble_alarms_never_exceed_the_most_alarm_prone_member(monkeypatch) -> None:
    """(task brief binding detail 6b) For every state, the ENSEMBLE (>= 2-of-3) row's
    `n_alarms` must not exceed the largest individual member's `n_alarms` -- checked on
    the controlled outlier fixture (`_make_prepared_pair_with_outliers`) where real
    OC-SVM/Isolation-Forest members and the reconstruction-distance stub all converge
    on the same handful of obvious anomalies."""
    import run_step2

    _patch_ensemble_members_ocsvm_iforest_stub_lstmae(monkeypatch, run_step2)
    prepared_variant, prepared_logmel, labels = _make_prepared_pair_with_outliers(
        run_step2._ENSEMBLE_SEED
    )

    far_table = run_step2._run_ensemble_view(
        prepared_variant, prepared_logmel, labels, alpha=0.05, run_name="test",
    )

    for label, sub in far_table.groupby("label"):
        by_member = sub.set_index("member")["n_alarms"]
        member_values = [
            by_member[m] for m in ("ocsvm", "iforest", "lstmae") if pd.notna(by_member[m])
        ]
        if not member_values:
            continue
        ensemble_alarms = by_member["ENSEMBLE"]
        if pd.isna(ensemble_alarms):
            continue
        assert ensemble_alarms <= max(member_values), (label, by_member.to_dict())


def test_ensemble_disjoint_member_alarms_never_reach_majority(monkeypatch) -> None:
    """(task brief binding detail 6c) Constructed disagreement: three members with
    pairwise DISJOINT alarm sets never reach the >= 2-of-3 majority anywhere ->
    ENSEMBLE `n_alarms` == 0."""
    import run_step2

    prepared_variant, prepared_logmel, labels = _make_ensemble_prepared_pair()
    top = split_by_segments(
        prepared_variant.segment_ids, prepared_variant.valid_mask, 0.5,
        run_step2._ENSEMBLE_SEED,
    )
    scoring_windows = top.scoring_windows
    assert scoring_windows.size >= 6  # enough to carve 3 non-empty disjoint groups
    groups = np.array_split(scoring_windows, 3)

    monkeypatch.setattr(
        run_step2, "_ENSEMBLE_MEMBER_FACTORIES",
        {
            "ocsvm": lambda: _AlarmSetScorer(frozenset(groups[0].tolist())),
            "iforest": lambda: _AlarmSetScorer(frozenset(groups[1].tolist())),
            "lstmae": lambda: _AlarmSetScorer(frozenset(groups[2].tolist())),
        },
    )

    far_table = run_step2._run_ensemble_view(
        prepared_variant, prepared_logmel, labels, alpha=0.05, run_name="test",
    )
    ensemble_row = far_table[far_table["member"] == "ENSEMBLE"].iloc[0]
    assert ensemble_row["n_alarms"] == 0.0


def test_ensemble_two_member_agreement_equals_shared_alarm_set(monkeypatch) -> None:
    """(task brief binding detail 6d) Constructed agreement: two members alarm on the
    EXACT SAME window set, the third on a disjoint set -> ENSEMBLE `n_alarms` equals
    that shared set's size exactly (every shared window gets 2 votes regardless of the
    third member; no other window ever reaches 2)."""
    import run_step2

    prepared_variant, prepared_logmel, labels = _make_ensemble_prepared_pair()
    top = split_by_segments(
        prepared_variant.segment_ids, prepared_variant.valid_mask, 0.5,
        run_step2._ENSEMBLE_SEED,
    )
    scoring_windows = top.scoring_windows
    assert scoring_windows.size >= 4
    half = scoring_windows.size // 2
    shared = frozenset(scoring_windows[:half].tolist())
    rest = frozenset(scoring_windows[half:].tolist())

    monkeypatch.setattr(
        run_step2, "_ENSEMBLE_MEMBER_FACTORIES",
        {
            "ocsvm": lambda: _AlarmSetScorer(shared),
            "iforest": lambda: _AlarmSetScorer(shared),
            "lstmae": lambda: _AlarmSetScorer(rest),
        },
    )

    far_table = run_step2._run_ensemble_view(
        prepared_variant, prepared_logmel, labels, alpha=0.05, run_name="test",
    )
    ensemble_row = far_table[far_table["member"] == "ENSEMBLE"].iloc[0]
    assert ensemble_row["n_alarms"] == len(shared)


def test_ensemble_deciding_pair_includes_lstmae_vote(monkeypatch) -> None:
    """(review follow-up M2, mutant-killing) `iforest` + `lstmae` share one alarm set
    while `ocsvm` alarms a DISJOINT set -> ENSEMBLE `n_alarms` equals the shared set's
    size exactly: LSTM-AE's vote must be the DECIDING one in a 2-of-3 majority. Kills
    the `ocsvm & iforest`-only mutant (`ensemble_alarms` computed from just those two
    members, LSTM-AE's vote dropped), which the disjoint test (every pairwise AND
    empty -> 0 == expected) and the two-member-agreement test (ocsvm/iforest ARE its
    agreeing pair -> their AND == expected) above both fail to distinguish from the
    real >= 2-of-3 rule -- verified by applying that exact mutant in scratch: the two
    prior tests pass under it, this one fails (0 != len(shared)); see the task-4
    report's follow-up section for the captured run."""
    import run_step2

    prepared_variant, prepared_logmel, labels = _make_ensemble_prepared_pair()
    top = split_by_segments(
        prepared_variant.segment_ids, prepared_variant.valid_mask, 0.5,
        run_step2._ENSEMBLE_SEED,
    )
    scoring_windows = top.scoring_windows
    assert scoring_windows.size >= 4
    half = scoring_windows.size // 2
    shared = frozenset(scoring_windows[:half].tolist())
    rest = frozenset(scoring_windows[half:].tolist())

    monkeypatch.setattr(
        run_step2, "_ENSEMBLE_MEMBER_FACTORIES",
        {
            "ocsvm": lambda: _AlarmSetScorer(rest),
            "iforest": lambda: _AlarmSetScorer(shared),
            "lstmae": lambda: _AlarmSetScorer(shared),
        },
    )

    far_table = run_step2._run_ensemble_view(
        prepared_variant, prepared_logmel, labels, alpha=0.05, run_name="test",
    )
    ensemble_row = far_table[far_table["member"] == "ENSEMBLE"].iloc[0]
    assert ensemble_row["n_alarms"] == len(shared)


@pytest.mark.parametrize("protocol", ["cross-day", "cross-day-per-state"])
def test_ensemble_requires_within_day_protocol(
    tmp_path, monkeypatch, capsys, protocol
) -> None:
    """(task brief binding detail 6e) `--ensemble` mirrors `--states`' own
    within-day-only guard: any other protocol is a usage error, rejected before any
    run is ever prepared."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    (tmp_path / "data").mkdir()  # guard fires before any run is ever looked up

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(["--protocol", protocol, "--variant", "fusion", "--ensemble"])
    assert exc_info.value.code == 2
    assert "requires --protocol within-day" in capsys.readouterr().err


def test_ensemble_with_gt_labels_rejected(tmp_path, monkeypatch, capsys) -> None:
    """(task brief binding detail 6e) `--ensemble` mirrors `--states`' own
    detected-labels-only guard."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    (tmp_path / "data").mkdir()  # guard fires before any run is ever looked up

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(["--labels", "gt", "--ensemble"])
    assert exc_info.value.code == 2
    assert "--ensemble is detected-labels only" in capsys.readouterr().err


@pytest.mark.parametrize("variant", ["vibration", "logmel"])
def test_ensemble_requires_mic_primary_variant(
    tmp_path, monkeypatch, capsys, variant
) -> None:
    """(review follow-up M1) `--ensemble` accepts only mic-primary sweep variants
    (audio, audio-beats, fusion, fusion-beats): the ensemble's three-way split is
    drawn from the SWEEP variant's own `segment_ids` and reused to index the logmel
    features the LSTM-AE member consumes -- only variants whose primary stream is the
    generator mic share recording-segment boundaries with logmel BY CONSTRUCTION, so
    for `vibration` the grid guard can pass by coincidence while the LSTM-AE member's
    split-conformal calibration silently loses its leakage-safety basis (reviewer
    finding). Rejected up front, exit 2, with the leakage reason in the message --
    same `parser.error` pattern as the flag's other two guards. `logmel` itself is
    also rejected (all three members would fit one identical feature matrix -- not
    the design chapter's committed ensemble contrast)."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    (tmp_path / "data").mkdir()  # guard fires before any run is ever looked up

    import run_step2

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(["--variant", variant, "--ensemble"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--ensemble requires a mic-primary --variant" in err
    assert "leakage" in err


def test_ensemble_grid_mismatch_guard_exits_2(tmp_path, monkeypatch, capsys) -> None:
    """(task brief binding detail 6f) `--ensemble`'s grid-alignment guard (binding
    detail 3): a `logmel` `PreparedRun` whose grid does not match the sweep variant's
    own grid must abort the whole invocation (exit 2, clear stderr message) instead of
    silently indexing two prepared runs at mismatched window positions. `prepare_run`
    is monkeypatched to shift ONLY the logmel call's returned grid."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    _build_one_day_root(tmp_path / "data")

    import run_step2

    real_prepare_run = run_step2.prepare_run

    def _shifted_logmel_prepare_run(run, variant, cfg, **kwargs):
        prepared = real_prepare_run(run, variant, cfg, **kwargs)
        if variant != "logmel":
            return prepared
        shifted_grid = WindowGrid(
            t0_ns=prepared.grid.t0_ns, window_ns=prepared.grid.window_ns,
            n_windows=prepared.grid.n_windows - 1,
        )
        return replace(prepared, grid=shifted_grid)

    monkeypatch.setattr(run_step2, "prepare_run", _shifted_logmel_prepare_run)
    _patch_ensemble_members_ocsvm_iforest_stub_lstmae(monkeypatch, run_step2)

    with pytest.raises(SystemExit) as exc_info:
        run_step2.main(
            [
                "--protocol", "within-day", "--variant", "fusion", "--labels", "detected",
                "--conditioning", "pooled", "--scorer", "knn", "--ensemble",
            ]
        )
    assert exc_info.value.code == 2
    assert "grid mismatch" in capsys.readouterr().err.lower()
