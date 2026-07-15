"""Tests for `rowii.anomaly.overlap` (Task 7, package-2 design spec `docs/
superpowers/specs/2026-07-15-step2-scarcity-crossday-beats-design.md` D4): pure
unit coverage of the candidate-overlap primitives (`top_candidates`, `to_utc_ns`,
`match_by_time`, `jaccard`), plus one script-level smoke test for `scripts/
analyze_step2.py` -- no real `results/`/`data/` anywhere, a monkeypatched grid
lookup stands in for `rowii.io.dataset.discover`/`rowii.pipeline.prepare_run`
(mirrors `tests/test_warm_cache.py`'s established "fake the one I/O seam" pattern
rather than a synthetic Gantner data tree).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rowii.anomaly.overlap import GLOBAL_LABEL, jaccard, match_by_time, to_utc_ns, top_candidates
from rowii.signals.windows import WindowGrid

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# match_by_time (brief Step 1 fixtures, verbatim)
# ---------------------------------------------------------------------------


class TestMatchByTime:
    def _cands(self, times_s, label="2"):
        return pd.DataFrame({
            "t_utc_ns": (np.array(times_s) * 1_000_000_000).astype(np.int64),
            "label": label, "p_value": 0.01,
        })

    def test_matches_within_tolerance_only(self):
        a, b = self._cands([100, 200, 300]), self._cands([101, 250, 304])
        m = match_by_time(a, b, tol_s=5.0)
        assert len(m) == 2
        assert sorted(m["dt_s"].abs().round(0).tolist()) == [1.0, 4.0]

    def test_greedy_one_to_one(self):
        a, b = self._cands([100, 102]), self._cands([101])
        m = match_by_time(a, b, tol_s=5.0)
        assert len(m) == 1
        assert abs(m["dt_s"].iloc[0]) == pytest.approx(1.0)

    def test_columns_and_empty_inputs(self):
        """Column contract (brief interface spec) + the degenerate empty-set case
        (no real sweep ever produces zero candidates for a non-excluded label, but
        `top_candidates` on an all-excluded scores frame legitimately can)."""
        a, b = self._cands([100]), self._cands([])
        m = match_by_time(a, b, tol_s=5.0)
        assert list(m.columns) == [
            "t_utc_ns_a", "t_utc_ns_b", "dt_s", "label_a", "label_b",
            "p_value_a", "p_value_b",
        ]
        assert len(m) == 0

    def test_matched_rows_carry_the_source_labels_and_p_values(self):
        a = pd.DataFrame({"t_utc_ns": [100_000_000_000], "label": ["x"], "p_value": [0.02]})
        b = pd.DataFrame({"t_utc_ns": [101_000_000_000], "label": ["y"], "p_value": [0.03]})
        m = match_by_time(a, b, tol_s=5.0)
        assert len(m) == 1
        row = m.iloc[0]
        assert row["label_a"] == "x"
        assert row["label_b"] == "y"
        assert row["p_value_a"] == pytest.approx(0.02)
        assert row["p_value_b"] == pytest.approx(0.03)
        assert row["dt_s"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# jaccard (brief Step 1 fixture, verbatim)
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_values(self):
        assert jaccard(3, 3, 2) == pytest.approx(0.5)
        assert jaccard(0, 0, 0) == 0.0


# ---------------------------------------------------------------------------
# top_candidates (brief Step 1 fixture, verbatim, plus top_k/columns coverage)
# ---------------------------------------------------------------------------


class TestTopCandidates:
    def test_per_label_and_global_ordering(self):
        scores = pd.DataFrame({
            "window": [0, 1, 2, 3], "label": ["0", "0", "1", "1"],
            "score": [5.0, 1.0, 9.0, 2.0], "p_value": [0.01, 0.5, 0.01, 0.3],
            "alarm": [True, False, True, False],
        })
        top = top_candidates(scores, top_k=1)
        per_label = top[top["label"] != "__any__"]
        assert set(per_label["window"]) == {0, 2}
        global_rows = top[top["label"] == "__any__"]
        assert global_rows["window"].iloc[0] == 2  # p tied at 0.01 -> higher score wins

    def test_top_k_larger_than_group_size_returns_all_rows_ranked(self):
        scores = pd.DataFrame({
            "window": [0, 1], "label": ["0", "0"],
            "score": [5.0, 1.0], "p_value": [0.01, 0.5], "alarm": [True, False],
        })
        top = top_candidates(scores, top_k=10)
        per_label = top[top["label"] == "0"].reset_index(drop=True)
        assert list(per_label["window"]) == [0, 1]  # min(top_k, len(group))
        assert list(per_label["rank"]) == [1, 2]

    def test_columns_and_global_label_constant(self):
        scores = pd.DataFrame({
            "window": [0], "label": ["0"], "score": [1.0], "p_value": [0.1], "alarm": [False],
        })
        top = top_candidates(scores, top_k=1)
        assert list(top.columns) == ["window", "label", "score", "p_value", "rank"]
        assert GLOBAL_LABEL == "__any__"
        assert set(top["label"]) == {"0", GLOBAL_LABEL}


# ---------------------------------------------------------------------------
# to_utc_ns
# ---------------------------------------------------------------------------


class TestToUtcNs:
    def test_adds_t_utc_ns_column(self):
        candidates = pd.DataFrame({"window": [0, 1, 5], "label": ["a", "a", "b"]})
        out = to_utc_ns(candidates, t0_ns=1_000_000_000, window_ns=2_000_000_000)
        assert list(out["t_utc_ns"]) == [1_000_000_000, 3_000_000_000, 11_000_000_000]
        assert out["t_utc_ns"].dtype == np.int64

    def test_does_not_mutate_input(self):
        candidates = pd.DataFrame({"window": [0], "label": ["a"]})
        to_utc_ns(candidates, t0_ns=0, window_ns=1_000_000_000)
        assert "t_utc_ns" not in candidates.columns


# ---------------------------------------------------------------------------
# Script-level smoke test for scripts/analyze_step2.py (Step 4)
# ---------------------------------------------------------------------------

_T0_NS = 1_750_000_000_000_000_000  # arbitrary but fixed UTC epoch, ns


def _write_scores_parquet(path: Path, **columns) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_parquet(path, engine="pyarrow", index=False)


def test_analyze_step2_writes_overlap_report_and_needs_listening_check(
    monkeypatch, tmp_path
) -> None:
    """End-to-end smoke test: two hand-written `scores.parquet` files (no real
    `results/`/`data/` anywhere) under the within-day combo layout, plus a
    monkeypatched grid lookup (`analyze_step2._grid_for_combo`, the one seam that
    would otherwise touch `rowii.io.dataset.discover`/`rowii.pipeline.prepare_run`)
    -- one `--check-utc` timestamp that hits both combos and one that misses both.

    ComboA ("fusion-knn") windows 10/500/900 map to a DIFFERENT grid than comboB's
    ("audio-beats-knn") windows 503/901 -- window 500 (comboA) and window 503
    (comboB) are 3s apart in absolute UTC time (within the 5s default tolerance)
    despite their raw indices disagreeing by 3, proving the match goes through UTC
    time, not raw window equality (module docstring's central point).
    """
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))

    results_root = tmp_path / "results"
    run = "test-day"
    combo_a, combo_b = "fusion-knn", "audio-beats-knn"
    window_ns = 1_000_000_000
    grid_a = WindowGrid(t0_ns=_T0_NS, window_ns=window_ns, n_windows=1000)
    grid_b = WindowGrid(t0_ns=_T0_NS, window_ns=window_ns, n_windows=1000)

    _write_scores_parquet(
        results_root / "step2" / "within-day" / run / "fusion-detected" / "per-state-knn"
        / "scores.parquet",
        window=[10, 500, 900], label=["0", "1", "0"], score=[9.0, 8.0, 1.0],
        p_value=[0.01, 0.02, 0.5], alarm=[True, True, False],
    )
    _write_scores_parquet(
        results_root / "step2" / "within-day" / run / "audio-beats-detected" / "per-state-knn"
        / "scores.parquet",
        window=[503, 901], label=["0", "1"], score=[7.0, 1.0],
        p_value=[0.01, 0.6], alarm=[True, False],
    )

    import analyze_step2

    grids = {"fusion": grid_a, "audio-beats": grid_b}

    def _fake_grid_for_combo(day_name, variant, cfg):
        assert day_name == run  # bare within-day run name -> day B is itself
        return grids[variant]

    monkeypatch.setattr(analyze_step2, "_grid_for_combo", _fake_grid_for_combo)

    hit_utc = pd.Timestamp(_T0_NS + 500 * window_ns, unit="ns", tz="UTC").isoformat()
    miss_utc = pd.Timestamp(_T0_NS, unit="ns", tz="UTC").isoformat()

    exit_code = analyze_step2.main([
        "--results-root", str(results_root),
        "--runs", run,
        "--pairs", f"{combo_a}:{combo_b}",
        "--top-k", "2",
        "--check-utc", hit_utc, miss_utc,
    ])
    assert exit_code == 0

    overlap_dir = results_root / "step2" / "overlap"
    report_path = overlap_dir / f"{run}--{combo_a}--vs--{combo_b}.md"
    assert report_path.is_file()
    report_text = report_path.read_text()
    assert f"{combo_a}: 2 candidate(s)" in report_text
    assert f"{combo_b}: 2 candidate(s)" in report_text
    assert "matched: 1" in report_text
    assert "Jaccard: 0.333" in report_text
    assert "## Matches" in report_text
    assert "No matches within tolerance" not in report_text
    assert f"Unmatched ({combo_a} only)" in report_text
    assert f"Unmatched ({combo_b} only)" in report_text

    needs_listening_path = overlap_dir / "needs_listening_check.md"
    assert needs_listening_path.is_file()
    listening_text = needs_listening_path.read_text()
    assert combo_a in listening_text
    assert combo_b in listening_text
    assert f"| {hit_utc} | {run} | {combo_a} | hit |" in listening_text
    assert f"| {hit_utc} | {run} | {combo_b} | hit |" in listening_text
    assert f"| {miss_utc} | {run} | {combo_a} | miss |" in listening_text
    assert f"| {miss_utc} | {run} | {combo_b} | miss |" in listening_text


def test_analyze_step2_partner_failure_does_not_drop_the_successful_combo(
    monkeypatch, tmp_path
) -> None:
    """A pair where only ONE side has a `scores.parquet` on disk: the pair's own
    overlap report must be skipped (exit code 1, no report file), but the side that
    DID load must still show up in `needs_listening_check.md` -- a combo that
    loaded fine must not be dropped just because its pair partner is missing
    (`_ensure_combo_candidates`'s docstring)."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))

    results_root = tmp_path / "results"
    run = "test-day"
    combo_a, combo_b = "fusion-knn", "audio-beats-knn"  # comboB's scores.parquet is never written
    window_ns = 1_000_000_000
    grid_a = WindowGrid(t0_ns=_T0_NS, window_ns=window_ns, n_windows=1000)

    _write_scores_parquet(
        results_root / "step2" / "within-day" / run / "fusion-detected" / "per-state-knn"
        / "scores.parquet",
        window=[10], label=["0"], score=[9.0], p_value=[0.01], alarm=[True],
    )

    import analyze_step2

    monkeypatch.setattr(analyze_step2, "_grid_for_combo", lambda day_name, variant, cfg: grid_a)

    hit_utc = pd.Timestamp(_T0_NS + 10 * window_ns, unit="ns", tz="UTC").isoformat()

    exit_code = analyze_step2.main([
        "--results-root", str(results_root),
        "--runs", run,
        "--pairs", f"{combo_a}:{combo_b}",
        "--top-k", "5",
        "--check-utc", hit_utc,
    ])
    assert exit_code == 1  # the pair was skipped -- comboB has no data

    overlap_dir = results_root / "step2" / "overlap"
    assert not (overlap_dir / f"{run}--{combo_a}--vs--{combo_b}.md").is_file()

    listening_text = (overlap_dir / "needs_listening_check.md").read_text()
    assert f"| {hit_utc} | {run} | {combo_a} | hit |" in listening_text
    assert combo_b not in listening_text  # never successfully analyzed -- no row at all
