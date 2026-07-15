"""Tests for `scripts/apply_detector.py` (package-2 Task 9, D2 stretch): CLI-level
tests against a monkeypatched `discover`/`load_config`/`prepare_run` seam, feeding
hand-built `PreparedRun`s directly -- no real ROWII data or synthetic gantner tree
anywhere (orchestrator resolution 6: mirrors `tests/test_scarcity_cli.py`'s
established monkeypatch style, not `tests/test_step2_cli.py`'s synthetic-tree
fixture).

This script (unlike `run_step2_scarcity.py`) also touches SCADA/GT for the FIT day
(to compute the cluster-id -> mode-name majority mapping), so two extra module-level
seams are monkeypatched too: `_betriebsdaten_for_grid` (bypasses real Betriebsdaten
`.dat` header reads) and `gt_labels` (bypasses the SCADA rule engine entirely --
that engine has its own dedicated tests in `tests/test_gt_labels.py`; this file only
needs full control over which GT state name lands on which synthetic block).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from rowii.config import load_config
from rowii.io.dataset import RecordingIndex, Run
from rowii.pipeline import PreparedRun
from rowii.signals.windows import WindowGrid

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

_FIT_RUN = "fit-day"
_APPLY_RUN = "apply-day"
_FIT_T0_NS = 0
_APPLY_T0_NS = 500_000_000_000
"""Deliberately different from `_FIT_T0_NS` (orchestrator resolution 6: "give the
apply grid a different t0_ns to pin" that `segments.csv`'s UTC bounds come from the
APPLY grid, not the fit grid)."""

_N_SEGMENTS = 8
_SEG_LEN = 30
"""8 segments x 30 windows = 240 windows per state -- comfortably enough for
`StickyHmmSmoother`/`duration_filter` (min_dwell_s=3) to recover two clean,
well-separated states (mirrors `tests/test_scarcity_cli.py`'s `_two_state_prepared`
sizing, which is already verified reliable in this exact codebase)."""


# ---------------------------------------------------------------------------
# Hand-built fixtures (no gantner tree -- orchestrator resolution 6)
# ---------------------------------------------------------------------------


def _fake_index(fit_has_betriebsdaten: bool) -> RecordingIndex:
    fit_day_root = Path("/fake-fit-day-root")
    apply_day_root = Path("/fake-apply-day-root")
    runs = [
        Run(name=_FIT_RUN, files={}, day_root=fit_day_root),
        Run(name=_APPLY_RUN, files={}, day_root=apply_day_root),
    ]
    betriebsdaten_by_day = (
        {fit_day_root: [Path("/fake-betriebsdaten/2026-07-01_00-00-00.dat")]}
        if fit_has_betriebsdaten
        else {}
    )
    return RecordingIndex(runs=runs, betriebsdaten=[], betriebsdaten_by_day=betriebsdaten_by_day)


def _two_state_prepared(t0_ns: int, seed: int) -> PreparedRun:
    """Two well-separated 'states' (feature values 0 and 20), alternating by
    segment -- `_N_SEGMENTS` segments of `_SEG_LEN` windows each."""
    rng = np.random.default_rng(seed)
    feats: list[np.ndarray] = []
    seg_ids: list[np.ndarray] = []
    for s in range(_N_SEGMENTS):
        value = 20.0 * (s % 2)
        feats.append(rng.normal(value, 0.1, (_SEG_LEN, 2)))
        seg_ids.append(np.full(_SEG_LEN, s, dtype=np.int64))
    features = np.vstack(feats)
    n = len(features)
    return PreparedRun(
        features=features,
        grid=WindowGrid(t0_ns=t0_ns, window_ns=1_000_000_000, n_windows=n),
        valid_mask=np.ones(n, dtype=bool),
        feature_names=["f0", "f1"],
        segment_ids=np.concatenate(seg_ids),
    )


def _fake_gt_for_two_state() -> pd.DataFrame:
    """Fixed GT state per window, aligned block-for-block with `_two_state_prepared`'s
    own even/odd segment pattern: even segments (feature value 0.0) -> "standstill",
    odd segments (feature value 20.0) -> "turbine". Bypasses the real SCADA rule
    engine entirely (`gt_labels` is monkeypatched to return this) so the test does not
    depend on `GtRules` threshold tuning -- see module docstring."""
    state = np.array(
        [
            "standstill" if s % 2 == 0 else "turbine"
            for s in range(_N_SEGMENTS)
            for _ in range(_SEG_LEN)
        ],
        dtype=object,
    )
    n = _N_SEGMENTS * _SEG_LEN
    return pd.DataFrame({"state": state, "load_bin": np.full(n, -1, dtype=np.int64)})


def _two_state_cfg(results_root: Path):
    cfg = load_config()
    cfg = replace(cfg, results_root=results_root, data_root=Path("/fake-data-root"))
    return replace(cfg, detect=replace(cfg.detect, n_states=2, min_dwell_s=3))


def _fake_prepare_run(fit_prepared: PreparedRun, apply_prepared: PreparedRun):
    def _inner(run, variant, cfg, *, use_cache):
        if run.name == _FIT_RUN:
            return fit_prepared
        if run.name == _APPLY_RUN:
            return apply_prepared
        raise AssertionError(f"unexpected run name {run.name!r}")

    return _inner


def _install_common_monkeypatches(
    monkeypatch, apply_detector, results_root: Path, *, fit_has_betriebsdaten: bool
):
    fit_prepared = _two_state_prepared(_FIT_T0_NS, seed=0)
    apply_prepared = _two_state_prepared(_APPLY_T0_NS, seed=1)

    monkeypatch.setattr(
        apply_detector, "discover", lambda data_root: _fake_index(fit_has_betriebsdaten)
    )
    monkeypatch.setattr(apply_detector, "load_config", lambda: _two_state_cfg(results_root))
    monkeypatch.setattr(
        apply_detector, "prepare_run", _fake_prepare_run(fit_prepared, apply_prepared)
    )
    monkeypatch.setattr(
        apply_detector, "_betriebsdaten_for_grid",
        (lambda betriebsdaten, grid: list(betriebsdaten)) if fit_has_betriebsdaten
        else (lambda betriebsdaten, grid: []),
    )
    monkeypatch.setattr(
        apply_detector, "load_scada_window_means", lambda files, grid: pd.DataFrame()
    )
    monkeypatch.setattr(
        apply_detector, "gt_labels", lambda scada, rules, *, window_s: _fake_gt_for_two_state()
    )
    return fit_prepared, apply_prepared


# ---------------------------------------------------------------------------
# 1. End-to-end: segments.csv carries fit-day cluster ids + apply-grid UTC bounds,
#    timeline.md carries the qualitative-only banner (brief Step 1)
# ---------------------------------------------------------------------------


def test_segments_carry_fit_day_ids_and_apply_grid_utc_bounds_and_timeline_has_banner(
    tmp_path, monkeypatch,
) -> None:
    import apply_detector

    results_root = tmp_path / "results"
    _install_common_monkeypatches(
        monkeypatch, apply_detector, results_root, fit_has_betriebsdaten=True
    )

    exit_code = apply_detector.main(
        ["--fit-run", _FIT_RUN, "--apply-run", _APPLY_RUN, "--variant", "fusion"]
    )
    assert exit_code == 0

    out_dir = (
        results_root / "step2" / "transfer" / f"{_FIT_RUN}--to--{_APPLY_RUN}" / "fusion"
    )
    segments_path = out_dir / "segments.csv"
    timeline_path = out_dir / "timeline.md"
    assert segments_path.is_file()
    assert timeline_path.is_file()

    segments = pd.read_csv(segments_path, parse_dates=["start_utc", "end_utc"])
    assert list(segments.columns) == [
        "start_utc", "end_utc", "duration_s", "cluster_id", "mapped_mode",
    ]
    assert len(segments) > 0

    # cluster ids come from the FIT day's own label space: exactly {0, 1}, the two
    # states FittedDetector.fit found on the (2-state) fit day -- never some other
    # id space independently re-derived from the apply day.
    assert set(segments["cluster_id"].unique().tolist()) <= {0, 1}

    # UTC bounds are derived from the APPLY grid (t0_ns=_APPLY_T0_NS), NOT the fit
    # grid (t0_ns=_FIT_T0_NS) -- the whole point of the different-t0_ns fixture setup.
    expected_apply_start = pd.Timestamp(_APPLY_T0_NS, unit="ns", tz="UTC")
    assert segments["start_utc"].min() == expected_apply_start
    assert segments["start_utc"].min() != pd.Timestamp(_FIT_T0_NS, unit="ns", tz="UTC")

    # Every mapped_mode is one of the two fit-day GT state names (the fit day DOES
    # have SCADA here) -- never blank, since the fit day's own eval windows cover
    # both clusters.
    assert set(segments["mapped_mode"].unique().tolist()) <= {"standstill", "turbine"}
    assert "" not in segments["mapped_mode"].unique().tolist()

    timeline_text = timeline_path.read_text()
    assert f"labels are transferred from {_FIT_RUN}" in timeline_text
    assert "no SCADA ground truth" in timeline_text
    assert "qualitative only" in timeline_text
    # No accuracy/ARI/F1 or other metric wording anywhere in the qualitative report.
    for forbidden in ("ARI", "accuracy", "F1", "macro-F1"):
        assert forbidden not in timeline_text


# ---------------------------------------------------------------------------
# 2. Fit day lacking SCADA -> every mapped_mode is "" (orchestrator resolution 2)
# ---------------------------------------------------------------------------


def test_fit_day_without_scada_leaves_every_mapped_mode_empty(tmp_path, monkeypatch) -> None:
    import apply_detector

    results_root = tmp_path / "results"
    _install_common_monkeypatches(
        monkeypatch, apply_detector, results_root, fit_has_betriebsdaten=False
    )

    exit_code = apply_detector.main(
        ["--fit-run", _FIT_RUN, "--apply-run", _APPLY_RUN, "--variant", "fusion"]
    )
    assert exit_code == 0

    out_dir = (
        results_root / "step2" / "transfer" / f"{_FIT_RUN}--to--{_APPLY_RUN}" / "fusion"
    )
    segments = pd.read_csv(out_dir / "segments.csv")
    assert len(segments) > 0
    # pandas reads an all-empty-string column back as NaN (float) unless told
    # otherwise -- coerce back to "" for the comparison, the on-disk/in-memory value
    # this script itself writes is always the empty string, never NaN.
    mapped = segments["mapped_mode"].fillna("").tolist()
    assert all(m == "" for m in mapped)


# ---------------------------------------------------------------------------
# 3. Unknown run name: exit 2, names the available runs (established CLI convention)
# ---------------------------------------------------------------------------


def test_unknown_run_name_exits_2_and_lists_available_runs(tmp_path, monkeypatch, capsys) -> None:
    import apply_detector

    monkeypatch.setattr(apply_detector, "discover", lambda data_root: _fake_index(True))

    def _boom_prepare_run(run, variant, cfg, *, use_cache):
        raise AssertionError("prepare_run must not be called for an unknown run name")

    monkeypatch.setattr(apply_detector, "prepare_run", _boom_prepare_run)

    exit_code = apply_detector.main(
        ["--fit-run", "totally-bogus-run", "--apply-run", _APPLY_RUN]
    )

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "totally-bogus-run" in err
    assert _FIT_RUN in err
    assert _APPLY_RUN in err
