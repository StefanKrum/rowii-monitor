"""Tests for `scripts/run_step2_scarcity.py` (Step-2 package 2, Task 5): CLI-level
tests against a monkeypatched `discover`/`load_config`/`prepare_run` seam, feeding a
hand-built `PreparedRun` directly -- no real ROWII data or synthetic gantner tree
anywhere (orchestrator resolution 6: mirrors `tests/test_warm_cache.py`'s established
monkeypatch style, not `tests/test_step2_cli.py`'s synthetic-tree fixture).
"""
from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rowii.anomaly.scarcity import _CURVE_COLUMNS, _SEGMENT_CURVE_COLUMNS
from rowii.config import load_config
from rowii.io.dataset import RecordingIndex, Run
from rowii.pipeline import PreparedRun
from rowii.signals.windows import WindowGrid

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Hand-built fixtures (no gantner tree -- orchestrator resolution 6)
# ---------------------------------------------------------------------------


def _fake_index(run_names: list[str]) -> RecordingIndex:
    runs = [Run(name=name, files={}, day_root=Path("/fake-day-root")) for name in run_names]
    return RecordingIndex(runs=runs, betriebsdaten=[], betriebsdaten_by_day={})


def _two_state_prepared(seed: int = 0, n_segments: int = 16, seg_len: int = 40) -> PreparedRun:
    """Two well-separated, well-populated 'states' (feature values 0 and 20),
    alternating by segment -- 8 segments/320 windows per label, comfortably above
    `_MIN_REF=20` on every side of the nested (top seed 7, nested seed 8) split
    (empirically verified: both labels clear the fit/conformal/scoring floor).
    """
    rng = np.random.default_rng(seed)
    feats: list[np.ndarray] = []
    seg_ids: list[np.ndarray] = []
    for s in range(n_segments):
        value = 20.0 * (s % 2)
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


def _three_state_one_starved_prepared(seed: int = 0) -> PreparedRun:
    """Two well-populated states (0, 1; 8 segments x 40 windows each) plus a
    deliberately STARVED third state (2; two 6-window segments, 12 windows total --
    nowhere near `_MIN_REF=20` on any split side) so it must end up "not curvable"."""
    rng = np.random.default_rng(seed)
    feats: list[np.ndarray] = []
    seg_ids: list[np.ndarray] = []
    seg_counter = 0
    for _label, value in ((0, 0.0), (1, 40.0)):
        for _ in range(8):
            feats.append(rng.normal(value, 0.1, (40, 2)))
            seg_ids.append(np.full(40, seg_counter, dtype=np.int64))
            seg_counter += 1
    for _ in range(2):
        feats.append(rng.normal(80.0, 0.1, (6, 2)))
        seg_ids.append(np.full(6, seg_counter, dtype=np.int64))
        seg_counter += 1
    features = np.vstack(feats)
    n = len(features)
    return PreparedRun(
        features=features,
        grid=WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=n),
        valid_mask=np.ones(n, dtype=bool),
        feature_names=["f0", "f1"],
        segment_ids=np.concatenate(seg_ids),
    )


def _two_state_cfg():
    cfg = load_config()
    return replace(cfg, detect=replace(cfg.detect, n_states=2, min_dwell_s=3))


def _three_state_cfg():
    cfg = load_config()
    return replace(cfg, detect=replace(cfg.detect, n_states=3, min_dwell_s=1))


# ---------------------------------------------------------------------------
# 1. --help exits 0 and documents every flag
# ---------------------------------------------------------------------------


def test_help_exits_zero_and_documents_every_flag(capsys) -> None:
    import run_step2_scarcity

    with pytest.raises(SystemExit) as exc_info:
        run_step2_scarcity.main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--runs", "--variants", "--scorers", "--alpha", "--reps", "--secondary-reps",
        "--secondary", "--out", "--data-root", "--results-root",
    ):
        assert flag in out, f"missing {flag!r} in --help output"


def test_build_parser_defaults_match_the_documented_defaults() -> None:
    import run_step2_scarcity

    args = run_step2_scarcity.build_parser().parse_args([])
    assert args.runs == ["010726-tu_ph_tu", "290626-tu"]
    assert args.variants == ["fusion", "audio-beats"]
    assert args.scorers == "knn"
    assert args.alpha == pytest.approx(0.05)
    assert args.reps == 50
    assert args.secondary_reps == 20
    assert args.secondary is False
    assert args.out == Path("results/step2/scarcity")
    assert args.data_root is None
    assert args.results_root is None


# ---------------------------------------------------------------------------
# 2. End-to-end: primary curve + --secondary, all files present and well-formed
# ---------------------------------------------------------------------------


def test_end_to_end_primary_and_secondary_curves(tmp_path, monkeypatch) -> None:
    import run_step2_scarcity

    prepared = _two_state_prepared()
    monkeypatch.setattr(run_step2_scarcity, "discover", lambda data_root: _fake_index(["run-a"]))
    monkeypatch.setattr(run_step2_scarcity, "load_config", _two_state_cfg)
    monkeypatch.setattr(
        run_step2_scarcity, "prepare_run",
        lambda run, variant, cfg, *, use_cache: prepared,
    )

    out_dir = tmp_path / "scarcity"
    exit_code = run_step2_scarcity.main(
        [
            "--runs", "run-a", "--variants", "fusion", "--scorers", "knn",
            "--reps", "10", "--secondary", "--secondary-reps", "5",
            "--out", str(out_dir),
        ]
    )
    assert exit_code == 0

    combo_dir = out_dir / "run-a--fusion-knn"

    curve_path = combo_dir / "curve.csv"
    assert curve_path.is_file()
    curve = pd.read_csv(curve_path)
    assert list(curve.columns) == list(_CURVE_COLUMNS)
    assert len(curve) > 0
    assert set(curve["label"].unique()) == {0, 1}

    png_path = combo_dir / "curve_by_state.png"
    assert png_path.is_file()
    assert png_path.stat().st_size > 0

    summary_path = out_dir / "summary.md"
    assert summary_path.is_file()
    summary_text = summary_path.read_text()
    # Every curvable state must be named (headline table), not just present in the CSV.
    assert "| 0 |" in summary_text
    assert "| 1 |" in summary_text
    assert "Not curvable states" in summary_text
    assert "Honesty notes" in summary_text
    assert "Scoring-side sampling noise" in summary_text

    seg_curve_path = combo_dir / "segment_curve.csv"
    assert seg_curve_path.is_file()
    seg_curve = pd.read_csv(seg_curve_path)
    assert list(seg_curve.columns) == list(_SEGMENT_CURVE_COLUMNS)
    assert len(seg_curve) > 0

    seg_png_path = combo_dir / "segment_curve.png"
    assert seg_png_path.is_file()
    assert seg_png_path.stat().st_size > 0

    assert "Secondary (segment accumulation)" in summary_text
    assert "Minutes-per-mode headline" in summary_text


def test_without_secondary_flag_no_segment_curve_files(tmp_path, monkeypatch) -> None:
    import run_step2_scarcity

    prepared = _two_state_prepared()
    monkeypatch.setattr(run_step2_scarcity, "discover", lambda data_root: _fake_index(["run-a"]))
    monkeypatch.setattr(run_step2_scarcity, "load_config", _two_state_cfg)
    monkeypatch.setattr(
        run_step2_scarcity, "prepare_run",
        lambda run, variant, cfg, *, use_cache: prepared,
    )

    out_dir = tmp_path / "scarcity"
    exit_code = run_step2_scarcity.main(
        ["--runs", "run-a", "--variants", "fusion", "--scorers", "knn", "--reps", "5",
         "--out", str(out_dir)]
    )
    assert exit_code == 0

    combo_dir = out_dir / "run-a--fusion-knn"
    assert (combo_dir / "curve.csv").is_file()
    assert not (combo_dir / "segment_curve.csv").exists()
    assert not (combo_dir / "segment_curve.png").exists()

    summary_text = (out_dir / "summary.md").read_text()
    assert "Secondary" not in summary_text


# ---------------------------------------------------------------------------
# 3. Not-curvable states are named, never silently dropped (orchestrator resolution 4)
# ---------------------------------------------------------------------------


def test_starved_state_is_named_not_curvable_not_silently_dropped(tmp_path, monkeypatch) -> None:
    import run_step2_scarcity

    prepared = _three_state_one_starved_prepared()
    monkeypatch.setattr(run_step2_scarcity, "discover", lambda data_root: _fake_index(["run-a"]))
    monkeypatch.setattr(run_step2_scarcity, "load_config", _three_state_cfg)
    monkeypatch.setattr(
        run_step2_scarcity, "prepare_run",
        lambda run, variant, cfg, *, use_cache: prepared,
    )

    out_dir = tmp_path / "scarcity"
    exit_code = run_step2_scarcity.main(
        ["--runs", "run-a", "--variants", "fusion", "--scorers", "knn", "--reps", "5",
         "--out", str(out_dir)]
    )
    assert exit_code == 0

    curve = pd.read_csv(out_dir / "run-a--fusion-knn" / "curve.csv")
    curvable_labels = set(curve["label"].unique())
    assert 2 not in curvable_labels  # the starved state never got a scarcity_curve call

    summary_text = (out_dir / "summary.md").read_text()
    # The starved state must still be NAMED somewhere (the "not curvable" table),
    # with a reason and a count -- never just absent from the report.
    assert "| 2 | excluded by min_ref" in summary_text


# ---------------------------------------------------------------------------
# 4. prepare_run RuntimeError: logged + skipped, not fatal to the whole invocation
# ---------------------------------------------------------------------------


def test_prepare_run_runtime_error_is_skipped_not_fatal(tmp_path, monkeypatch, caplog) -> None:
    import run_step2_scarcity

    good_prepared = _two_state_prepared()
    monkeypatch.setattr(
        run_step2_scarcity, "discover", lambda data_root: _fake_index(["run-bad", "run-good"])
    )
    monkeypatch.setattr(run_step2_scarcity, "load_config", _two_state_cfg)

    def _fake_prepare_run(run, variant, cfg, *, use_cache):
        if run.name == "run-bad":
            raise RuntimeError("92.5% of grid windows are invalid")
        return good_prepared

    monkeypatch.setattr(run_step2_scarcity, "prepare_run", _fake_prepare_run)

    out_dir = tmp_path / "scarcity"
    with caplog.at_level(logging.WARNING):
        exit_code = run_step2_scarcity.main(
            [
                "--runs", "run-bad", "run-good", "--variants", "fusion", "--scorers", "knn",
                "--reps", "5", "--out", str(out_dir),
            ]
        )
    assert exit_code == 0
    assert not (out_dir / "run-bad--fusion-knn").exists()
    assert (out_dir / "run-good--fusion-knn" / "curve.csv").is_file()

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("run-bad" in w and "prepare_run" in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# 5. Unknown run name: exit 2, names the available runs
# ---------------------------------------------------------------------------


def test_unknown_run_name_exits_2_and_lists_available_runs(tmp_path, monkeypatch, capsys) -> None:
    import run_step2_scarcity

    monkeypatch.setattr(
        run_step2_scarcity, "discover", lambda data_root: _fake_index(["run-a", "run-b"])
    )

    def _boom_prepare_run(run, variant, cfg, *, use_cache):
        raise AssertionError("prepare_run must not be called for an unknown run name")

    monkeypatch.setattr(run_step2_scarcity, "prepare_run", _boom_prepare_run)

    exit_code = run_step2_scarcity.main(
        ["--runs", "run-a", "totally-bogus-run", "--out", str(tmp_path / "scarcity")]
    )

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "totally-bogus-run" in err
    assert "run-a" in err
    assert "run-b" in err
