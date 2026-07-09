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

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
        "--alpha", "--top-k", "--labels", "--no-cache",
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
