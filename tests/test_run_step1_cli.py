"""Tests for `scripts/run_step1.py`'s summary accumulator: re-running a combo must
replace its `summary.csv` row in place (identity: `_SUMMARY_KEY_COLUMNS`), never
append a duplicate. The 2026-08-18 completeness audit traced duplicated
`results/analysis/overview.md` rows to exactly such a rerun (a mid-grid abort made
`300626-pu`'s audio combos run twice), so plain append is a correctness bug."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _result(run_step1, **overrides):
    base = dict(
        run="300626-pu",
        variant="audio",
        clusterer="kmeans",
        k=4,
        n_windows=100,
        n_valid=95,
        n_eval=90,
        ari=0.5,
        macro_f1=0.5,
        boundary_median_abs_s=10.0,
        silhouette=0.3,
        state_ari=0.8,
        state_accuracy=0.9,
        state_macro_f1=0.7,
        notes="",
    )
    base.update(overrides)
    return run_step1.ComboResult(**base)


def test_append_summary_row_replaces_same_combo_row(tmp_path) -> None:
    import run_step1

    root = tmp_path / "results"
    run_step1._append_summary_row(root, _result(run_step1, state_ari=0.1))
    run_step1._append_summary_row(root, _result(run_step1, state_ari=0.9))

    df = pd.read_csv(root / "summary.csv")
    assert len(df) == 1, "rerun of the same combo must upsert, not append"
    assert df.loc[0, "state_ari"] == 0.9, "the LATEST rerun's row must win"


def test_append_summary_row_keeps_distinct_identities(tmp_path) -> None:
    import run_step1

    root = tmp_path / "results"
    run_step1._append_summary_row(root, _result(run_step1))
    run_step1._append_summary_row(root, _result(run_step1, clusterer="gmm"))
    run_step1._append_summary_row(root, _result(run_step1, k=6, notes="k-sweep"))
    run_step1._append_summary_row(root, _result(run_step1, k=6, notes="k-sweep"))

    df = pd.read_csv(root / "summary.csv")
    # default-k kmeans + default-k gmm + ONE k-sweep row (its own rerun upserted)
    assert len(df) == 3
    assert not (root / "summary.csv.tmp").exists(), "tmp sibling must be replaced away"
