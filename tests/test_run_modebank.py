"""CLI tests for scripts/run_modebank.py (Package-8 D1): monkeypatched prepare/discover
seams with hand-built PreparedRuns + a fake per-run GT map, verifying artifact shapes,
the {unknown,transition} ARI mask, the supervised/unsupervised tags, --smooth =
duration-filter-only (A1.3), the A3.8-style day-group/duplicate/pool-member guards, and
the mandatory low_confidence_modes surfacing (adversarial-review binding, T2 finding 1:
a low_confidence bank member's +inf threshold can never contribute a no_mode_fits
rejection, so the rate under-fires for it -- both the written metrics.json and a WARNING
line must name it).

Style-2 fixtures throughout (no synthetic Gantner trees), mirroring
`tests/test_step2_pooled_cli.py`'s established monkeypatch seam (`discover`/
`prepare_run` patched, no real data root ever touched) plus `tests/test_monitor_cli.py`'s
direct `load_config` monkeypatch (config built by hand, ambient env/.env never read).
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from rowii.config import Config, DetectConfig  # noqa: E402
from rowii.io.dataset import BurstFile, RecordingIndex, Run  # noqa: E402
from rowii.pipeline import PreparedRun  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402
from rowii.state.modebank import ModeBank  # noqa: E402

_W = 1_000_000_000
_NAMES = [f"ch0_octave_{i}" for i in range(4)]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _prepared(t0: int, seed: int, n_seg: int = 8, seg: int = 30) -> tuple[PreparedRun, np.ndarray]:
    """Contiguous *seg*-window segments alternating GT mode ('turbine'/'pump') --
    mirrors `tests/test_modebank.py`'s two-mode blob shape, laid out as segments so
    `build_pool`'s leakage-safe splits have real segment boundaries to draw."""
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
    """The plan's own seam (module docstring): `Run`s with NO burst files
    (`files={}`) -- callers that need the REAL `_run_day_groups` (which requires
    burst files, see its own docstring) monkeypatch it away; the ones below that
    exercise it for real use `_fake_run` instead (has a real dated burst file)."""
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


def _fake_run(name: str, date: str) -> Run:
    """A discovery-shaped `Run` whose single burst file's NAME carries *date* --
    exactly what the real (unmocked) `_run_day_groups` parses. Mirrors
    `tests/test_step2_pooled_cli.py`'s own `_fake_run` verbatim."""
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


def _out_dir(tmp_path: Path, test_run: str = "testC", variant: str = "fusion",
             family: str = "gaussian") -> Path:
    return tmp_path / "results" / "step2" / "modebank" / test_run / f"{variant}-{family}"


# ---------------------------------------------------------------------------
# 1. End-to-end: exit 0, metrics.json with supervised/unsupervised tags,
#    confusion.csv, assignments.parquet, notes.md (plan's own RED test)
# ---------------------------------------------------------------------------


def test_run_modebank_writes_metrics_with_supervised_and_unsupervised_tags(
    tmp_path, monkeypatch
) -> None:
    import run_modebank as rm

    pf1, g1 = _prepared(0, 1)
    pf2, g2 = _prepared(0, 2)  # second fit day
    pt, gt = _prepared(9_000_000_000, 3)
    prepared = {"fitA": pf1, "fitB": pf2, "testC": pt}
    gts = {"fitA": g1, "fitB": g2, "testC": gt}
    _install(monkeypatch, rm, tmp_path / "results", prepared, gts)
    monkeypatch.setattr(rm, "_run_day_groups", lambda run: {run.name})  # force disjoint

    code = rm.main([
        "--fit-runs", "fitA,fitB", "--test-run", "testC",
        "--variant", "fusion", "--family", "gaussian", "--alpha", "0.05", "--min-ref", "10",
    ])
    assert code == 0

    out = _out_dir(tmp_path)
    import json

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["bank"]["tag"] == "supervised"
    assert metrics["p7_pooled"]["tag"] == "unsupervised"
    assert 0.0 <= metrics["bank"]["ari"] <= 1.0
    assert "accuracy" in metrics["bank"] and "accuracy" not in metrics["p7_pooled"]
    assert 0.0 <= metrics["bank"]["no_mode_fits_rate"] <= 1.0
    # Mandatory addition (T2 finding 1): the field is ALWAYS present, empty when
    # every surviving member calibrated with enough data (this fixture's case).
    assert metrics["bank"]["low_confidence_modes"] == []
    assert (out / "confusion.csv").is_file()

    assign = pd.read_parquet(out / "assignments.parquet")
    assert set(assign.columns) >= {"window", "gt_state", "assigned", "no_mode_fits"}
    # "scattered to full grid" (plan interface text): one row per window of the
    # test run's own grid, not just its valid subset.
    assert len(assign) == pt.grid.n_windows

    notes = (out / "notes.md").read_text()
    assert "fitA" in notes and "fitB" in notes and "testC" in notes
    assert "inspired by the partner" in notes


# ---------------------------------------------------------------------------
# 2. _masked_ari drops BOTH unknown and transition (A1.5) -- plan's own RED test
# ---------------------------------------------------------------------------


def test_ari_mask_excludes_unknown_and_transition() -> None:
    """The masked-ARI helper drops BOTH labels (A1.5), unlike eval.metrics.evaluate."""
    import run_modebank as rm

    gt = np.array(["turbine", "unknown", "transition", "pump", "pump"], dtype=object)
    pred = np.array(["turbine", "pump", "turbine", "pump", "pump"], dtype=object)
    ari, n = rm._masked_ari(gt, pred)
    assert n == 3  # only turbine/pump/pump counted
    assert ari == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. --smooth = duration_filter ONLY (A1.3) -- plan's own RED test
# ---------------------------------------------------------------------------


def test_smooth_uses_duration_filter_only() -> None:
    import run_modebank as rm

    # a single 1-window flicker between two long runs is removed by duration_filter,
    # and the smoothing path must NOT call any HMM re-estimation.
    labels = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.int64)
    out = rm._smooth_ids(labels, min_dwell=3)
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# 4. MANDATORY ADDITION (adversarial-review binding, T2 finding 1):
#    low_confidence_modes surfaced in the metrics dict + a WARNING is logged,
#    pinned with a tiny-calibration fixture (mirrors tests/test_modebank.py's
#    own low-confidence construction).
# ---------------------------------------------------------------------------


def test_low_confidence_modes_surfaced_in_metrics_and_warning(caplog) -> None:
    import run_modebank as rm

    fit_f, fit_l = _prepared(0, 0)
    fit_f = fit_f.features
    cal_f, cal_l = _prepared(0, 1)
    cal_f = cal_f.features
    # 'pump' keeps its full fit-side reference (120 windows, ample) but its
    # calibration side is shrunk to 5 rows (< 19 = ceil(1/alpha)-1 at alpha=0.05)
    # -- still >= 1, so 'pump' survives as a member, but calibrates
    # low_confidence=True (rowii.anomaly.conformal.calibrate).
    pump_idx = np.flatnonzero(cal_l == "pump")
    keep = np.concatenate([np.flatnonzero(cal_l == "turbine"), pump_idx[:5]])
    cal_f, cal_l = cal_f[keep], cal_l[keep]

    bank = ModeBank.fit(
        fit_f, fit_l, cal_f, cal_l, family="gaussian", alpha=0.05, feature_names=_NAMES,
    )
    assert bank.low_confidence_modes == ("pump",)

    metrics = rm._bank_metrics(bank, ari=1.0, n_masked=10, accuracy=1.0, no_mode_fits_rate=0.0)
    assert metrics["low_confidence_modes"] == ["pump"]

    with caplog.at_level(logging.WARNING):
        rm._warn_low_confidence(bank)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("pump" in m and "no_mode_fits" in m for m in warnings), warnings


# ---------------------------------------------------------------------------
# 5. Argument-shape guards (pure, no data touched -- parser.error/SystemExit,
#    mirroring scripts/run_step2.py's cross-day-pooled A3.1/duplicate guards)
# ---------------------------------------------------------------------------


def test_duplicate_fit_run_names_exits_2(capsys) -> None:
    import run_modebank as rm

    with pytest.raises(SystemExit) as exc_info:
        rm.main([
            "--fit-runs", "fitA,fitA", "--test-run", "testC",
            "--variant", "fusion", "--family", "gaussian",
        ])
    assert exc_info.value.code == 2
    assert "duplicate" in capsys.readouterr().err


def test_test_run_listed_in_fit_runs_exits_2(capsys) -> None:
    import run_modebank as rm

    with pytest.raises(SystemExit) as exc_info:
        rm.main([
            "--fit-runs", "fitA,testC", "--test-run", "testC",
            "--variant", "fusion", "--family", "gaussian",
        ])
    assert exc_info.value.code == 2
    assert "testC" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 6. Data-dependent guards (discover/prepare_run seam, return 2 -- mirrors
#    cross-day-pooled's own "loud failure" posture)
# ---------------------------------------------------------------------------


def test_unknown_run_names_exit_2(tmp_path, monkeypatch, capsys) -> None:
    import run_modebank as rm

    pf1, g1 = _prepared(0, 1)
    pt, gt = _prepared(9_000_000_000, 3)
    _install(
        monkeypatch, rm, tmp_path / "results",
        {"fitA": pf1, "testC": pt}, {"fitA": g1, "testC": gt},
    )

    code = rm.main([
        "--fit-runs", "fitA,nope", "--test-run", "testC",
        "--variant", "fusion", "--family", "gaussian",
    ])
    assert code == 2
    err = capsys.readouterr().err
    assert "nope" in err


def test_day_group_overlap_between_fit_and_test_exits_2(tmp_path, monkeypatch, capsys) -> None:
    """The A3.8-style day-group guard (plan Task 3 Binding note, duplicated from
    scripts/run_step2.py's `_run_day_groups`) fires on the REAL (unmocked)
    day-group computation this time -- fitA and testC share one calendar day."""
    import run_modebank as rm

    pf1, g1 = _prepared(0, 1)
    pt, gt = _prepared(9_000_000_000, 3)
    prepared = {"fitA": pf1, "testC": pt}
    gts = {"fitA": g1, "testC": gt}
    runs = [_fake_run("fitA", "2026-07-01"), _fake_run("testC", "2026-07-01")]
    monkeypatch.setattr(
        rm, "discover",
        lambda dr: RecordingIndex(runs=runs, betriebsdaten=[], betriebsdaten_by_day={}),
    )
    monkeypatch.setattr(
        rm, "load_config",
        lambda: Config(data_root=Path("/d"), results_root=tmp_path / "results"),
    )
    monkeypatch.setattr(
        rm, "prepare_run",
        lambda run, variant, cfg, *, use_cache: prepared[run.name],
    )
    monkeypatch.setattr(
        rm, "_run_gt_states",
        lambda prepared_run, run, index, cfg: gts[run.name],
    )

    code = rm.main([
        "--fit-runs", "fitA", "--test-run", "testC",
        "--variant", "fusion", "--family", "gaussian",
    ])
    assert code == 2
    err = capsys.readouterr().err
    assert "fitA" in err
    assert "2026-07-01" in err
