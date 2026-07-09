"""Unit tests for `scripts/analyze_step1.py`'s pure functions (synthetic fixtures
only -- no real ROWII_DATA_ROOT needed). A handful of `@pytest.mark.data` tests at
the bottom cross-check the real-data recompute path (`load_run_variant_eval`)
against `results/summary.csv`'s already-recorded baseline numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_step1 as a  # noqa: E402

from rowii.config import load_config  # noqa: E402
from rowii.io.dataset import discover  # noqa: E402

# ---------------------------------------------------------------------------
# boundary_bucket
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("distance_s", "expected"),
    [
        (0.0, "0-5"),
        (4.999, "0-5"),
        (5.0, "5-15"),
        (14.999, "5-15"),
        (15.0, "15-30"),
        (29.999, "15-30"),
        (30.0, "30-60"),
        (59.999, "30-60"),
        (60.0, ">60"),
        (1000.0, ">60"),
    ],
)
def test_boundary_bucket_edges(distance_s: float, expected: str) -> None:
    assert a.boundary_bucket(distance_s) == expected


def test_boundary_bucket_nan_is_not_applicable() -> None:
    assert a.boundary_bucket(float("nan")) == "n/a"


# ---------------------------------------------------------------------------
# gt_boundary_distances_s
# ---------------------------------------------------------------------------


def test_gt_boundary_distances_single_change_point() -> None:
    # standstill x5, turbine x5 -> one change point at index 5 (the new run's own
    # first window, matching rowii.eval.metrics' "states[i] != states[i - 1]"
    # convention) -- index 5 itself is distance 0, everything else counts outward.
    states = ["standstill"] * 5 + ["turbine"] * 5
    distances = a.gt_boundary_distances_s(states, window_s=1.0)
    expected = np.array([5, 4, 3, 2, 1, 0, 1, 2, 3, 4], dtype=np.float64)
    np.testing.assert_array_equal(distances, expected)


def test_gt_boundary_distances_scales_with_window_s() -> None:
    states = ["standstill"] * 3 + ["turbine"] * 3
    distances = a.gt_boundary_distances_s(states, window_s=2.0)
    # Change point at index 3; in windows: [3, 2, 1, 0, 1, 2] -> x2s per window.
    np.testing.assert_array_equal(distances, np.array([6, 4, 2, 0, 2, 4], dtype=np.float64))


def test_gt_boundary_distances_ignores_changes_touching_unknown() -> None:
    # unknown -> turbine is not a countable change (mirrors rowii.eval.metrics'
    # _state_change_indices); only the turbine -> standstill change at index 6
    # counts.
    states = ["unknown"] * 3 + ["turbine"] * 3 + ["standstill"] * 3
    distances = a.gt_boundary_distances_s(states, window_s=1.0)
    # Nearest (only) real change point is index 6.
    expected = np.array(
        [6, 5, 4, 3, 2, 1, 0, 1, 2], dtype=np.float64
    )
    np.testing.assert_array_equal(distances, expected)


def test_gt_boundary_distances_all_nan_when_no_changes_exist() -> None:
    states = ["turbine"] * 5
    distances = a.gt_boundary_distances_s(states, window_s=1.0)
    assert len(distances) == 5
    assert np.all(np.isnan(distances))


# ---------------------------------------------------------------------------
# build_eval_window_table
# ---------------------------------------------------------------------------


def test_build_eval_window_table_marks_correct_and_excludes_unknown_gt() -> None:
    gt_states = ["standstill", "standstill", "turbine", "turbine", "unknown"]
    cluster = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    state_mapping = {0: "standstill", 1: "turbine"}

    table = a.build_eval_window_table(gt_states, cluster, state_mapping, window_s=1.0)

    # The "unknown" GT row is dropped entirely.
    assert len(table) == 4
    assert set(table.columns) >= {
        "gt_state", "cluster", "pred_state", "boundary_distance_s", "correct", "bucket",
    }
    assert table["correct"].all()


def test_build_eval_window_table_flags_a_genuine_error() -> None:
    gt_states = ["standstill", "turbine"]
    cluster = np.array([0, 0], dtype=np.int64)
    state_mapping = {0: "standstill"}  # cluster 0 majority-maps to standstill only

    table = a.build_eval_window_table(gt_states, cluster, state_mapping, window_s=1.0)

    assert list(table["correct"]) == [True, False]
    assert list(table["pred_state"]) == ["standstill", "standstill"]


def test_build_eval_window_table_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        a.build_eval_window_table(
            ["turbine", "turbine"], np.array([0], dtype=np.int64), {0: "turbine"}, window_s=1.0
        )


# ---------------------------------------------------------------------------
# error_anatomy_histogram
# ---------------------------------------------------------------------------


def _window_table(
    gt_states: list[str], pred_states: list[str], distances: list[float]
) -> pd.DataFrame:
    df = pd.DataFrame(
        {"gt_state": gt_states, "pred_state": pred_states, "boundary_distance_s": distances}
    )
    df["correct"] = df["gt_state"] == df["pred_state"]
    df["bucket"] = df["boundary_distance_s"].map(a.boundary_bucket)
    return df


def test_error_anatomy_histogram_counts_and_rates() -> None:
    # bucket 0-5: distances 1, 2, 3 (indices 0, 1, 3) -- 2 errors (idx 0, 1), 1
    #   correct (idx 3, "c" predicted correctly).
    # bucket 5-15: empty.
    # bucket 15-30: distance 20 (idx 2) -- 1 error.
    # bucket 30-60: distance 45 (idx 4) -- 1 error.
    # bucket >60: distance 70 (idx 5) -- correct, 0 errors (but n_eval_windows=1).
    table = _window_table(
        gt_states=["a", "a", "b", "c", "c", "d"],
        pred_states=["x", "x", "y", "c", "z", "d"],
        distances=[1.0, 2.0, 20.0, 3.0, 45.0, 70.0],
    )

    hist = a.error_anatomy_histogram(table)
    by_bucket = hist.set_index("bucket_s")

    assert by_bucket.loc["0-5", "n_eval_windows"] == 3
    assert by_bucket.loc["0-5", "n_errors"] == 2
    assert by_bucket.loc["0-5", "error_rate"] == pytest.approx(2 / 3)

    assert by_bucket.loc["5-15", "n_eval_windows"] == 0
    assert np.isnan(by_bucket.loc["5-15", "error_rate"])

    assert by_bucket.loc["15-30", "n_errors"] == 1
    assert by_bucket.loc["15-30", "error_rate"] == pytest.approx(1.0)

    assert by_bucket.loc["30-60", "n_errors"] == 1
    assert by_bucket.loc["30-60", "error_rate"] == pytest.approx(1.0)

    assert by_bucket.loc[">60", "n_eval_windows"] == 1
    assert by_bucket.loc[">60", "n_errors"] == 0
    assert by_bucket.loc[">60", "error_rate"] == 0.0

    # Total errors = 2 + 1 + 1 = 4 (a bucket with zero errors, incl. the empty
    # 5-15 bucket, contributes a defined 0.0 share, not NaN -- only a GLOBALLY
    # error-free table would make every share NaN, see the next test).
    assert by_bucket.loc["5-15", "pct_of_all_errors"] == pytest.approx(0.0)
    assert hist["pct_of_all_errors"].sum() == pytest.approx(1.0)
    assert by_bucket.loc["0-5", "pct_of_all_errors"] == pytest.approx(2 / 4)
    assert by_bucket.loc["15-30", "pct_of_all_errors"] == pytest.approx(1 / 4)
    assert by_bucket.loc["30-60", "pct_of_all_errors"] == pytest.approx(1 / 4)
    assert by_bucket.loc[">60", "pct_of_all_errors"] == pytest.approx(0.0)


def test_error_anatomy_histogram_handles_zero_errors_without_bad_percentages() -> None:
    table = _window_table(["a", "b"], ["a", "b"], [1.0, 2.0])

    hist = a.error_anatomy_histogram(table)

    assert hist["n_errors"].sum() == 0
    assert hist["pct_of_all_errors"].isna().all()


def test_error_anatomy_histogram_excludes_na_bucket_from_denominator() -> None:
    # A run with literally no GT changes: every window buckets to "n/a" and must
    # not appear in (or silently distort) the five named-bucket rows.
    table = _window_table(["a", "a"], ["a", "b"], [float("nan"), float("nan")])

    hist = a.error_anatomy_histogram(table)

    assert set(hist["bucket_s"]) == set(a._BUCKET_ORDER)
    assert hist["n_eval_windows"].sum() == 0
    assert hist["n_errors"].sum() == 0


# ---------------------------------------------------------------------------
# per_state_precision_recall
# ---------------------------------------------------------------------------


def test_per_state_precision_recall_perfect_prediction() -> None:
    table = _window_table(
        gt_states=["a", "a", "b", "b"], pred_states=["a", "a", "b", "b"], distances=[0, 0, 0, 0]
    )

    pr = a.per_state_precision_recall(table)

    assert set(pr["state"]) == {"a", "b"}
    assert (pr["precision"] == 1.0).all()
    assert (pr["recall"] == 1.0).all()
    assert list(pr.set_index("state").loc[["a", "b"], "support"]) == [2, 2]


def test_per_state_precision_recall_with_confusion() -> None:
    # GT: a,a,a,b,b. Pred: a,a,b,b,b.
    # state a: TP=2, n_gt=3 -> recall 2/3; n_pred=2 -> precision 2/2=1.0
    # state b: TP=2, n_gt=2 -> recall 2/2=1.0; n_pred=3 -> precision 2/3
    table = _window_table(
        gt_states=["a", "a", "a", "b", "b"],
        pred_states=["a", "a", "b", "b", "b"],
        distances=[0, 0, 0, 0, 0],
    )

    pr = a.per_state_precision_recall(table).set_index("state")

    assert pr.loc["a", "recall"] == pytest.approx(2 / 3)
    assert pr.loc["a", "precision"] == pytest.approx(1.0)
    assert pr.loc["b", "recall"] == pytest.approx(1.0)
    assert pr.loc["b", "precision"] == pytest.approx(2 / 3)


def test_per_state_precision_recall_nan_for_never_predicted_state() -> None:
    table = _window_table(gt_states=["a", "a"], pred_states=["b", "b"], distances=[0, 0])

    pr = a.per_state_precision_recall(table, states=["a", "b"]).set_index("state")

    assert pr.loc["a", "recall"] == 0.0  # predicted wrong every time, but n_gt > 0
    assert not np.isnan(pr.loc["b", "precision"])  # b was predicted; precision is defined
    assert pr.loc["b", "precision"] == 0.0  # never correct when predicted
    assert np.isnan(pr.loc["b", "recall"])  # state b never occurs in GT -> undefined


# ---------------------------------------------------------------------------
# recall_matrix
# ---------------------------------------------------------------------------


def test_recall_matrix_rows_states_cols_variants() -> None:
    audio = _window_table(["standstill", "turbine"], ["standstill", "turbine"], [0, 0])
    vibration = _window_table(["standstill", "turbine"], ["turbine", "turbine"], [0, 0])

    matrix = a.recall_matrix({"audio": audio, "vibration": vibration})

    assert list(matrix.columns) == ["audio", "vibration"]
    assert set(matrix.index) == {"standstill", "turbine"}
    assert matrix.loc["standstill", "audio"] == pytest.approx(1.0)
    assert matrix.loc["standstill", "vibration"] == pytest.approx(0.0)
    assert matrix.loc["turbine", "vibration"] == pytest.approx(1.0)


def test_recall_matrix_missing_state_in_one_variant_is_nan() -> None:
    audio = _window_table(["standstill", "turbine"], ["standstill", "turbine"], [0, 0])
    vibration = _window_table(["turbine", "turbine"], ["turbine", "turbine"], [0, 0])

    matrix = a.recall_matrix(
        {"audio": audio, "vibration": vibration}, states=["standstill", "turbine"]
    )

    assert np.isnan(matrix.loc["standstill", "vibration"])
    assert matrix.loc["standstill", "audio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# sensitivity_deltas / sensitivity_verdict
# ---------------------------------------------------------------------------


def test_sensitivity_deltas_computes_signed_differences() -> None:
    observations = [
        a.SensitivityObservation("power_eps_mw", 0.5, 1.0, state_ari=0.90, state_accuracy=0.95),
        a.SensitivityObservation("power_eps_mw", 2.0, 4.0, state_ari=0.93, state_accuracy=0.97),
    ]

    deltas = a.sensitivity_deltas(
        baseline_state_ari=0.92, baseline_state_accuracy=0.96, observations=observations
    )

    assert deltas.loc[0, "delta_state_ari"] == pytest.approx(0.90 - 0.92)
    assert deltas.loc[1, "delta_state_ari"] == pytest.approx(0.93 - 0.92)
    assert deltas.loc[0, "delta_state_accuracy"] == pytest.approx(0.95 - 0.96)


def test_sensitivity_verdict_robust_when_all_deltas_within_tolerance() -> None:
    observations = [
        a.SensitivityObservation("power_eps_mw", 0.5, 1.0, state_ari=0.905, state_accuracy=0.955),
        a.SensitivityObservation("power_eps_mw", 2.0, 4.0, state_ari=0.915, state_accuracy=0.965),
    ]
    deltas = a.sensitivity_deltas(0.91, 0.96, observations)

    verdict = a.sensitivity_verdict(deltas, tolerance=0.02)

    assert "**robust**" in verdict
    assert "not robust" not in verdict


def test_sensitivity_verdict_not_robust_when_a_delta_exceeds_tolerance() -> None:
    observations = [
        a.SensitivityObservation(
            "ph_min_dwell_s", 0.5, 300.0, state_ari=0.60, state_accuracy=0.80
        ),
        a.SensitivityObservation(
            "ph_min_dwell_s", 2.0, 1200.0, state_ari=0.91, state_accuracy=0.96
        ),
    ]
    deltas = a.sensitivity_deltas(0.91, 0.96, observations)

    verdict = a.sensitivity_verdict(deltas, tolerance=0.02)

    assert "**not robust**" in verdict
    assert "ph_min_dwell_s" in verdict


def test_sensitivity_verdict_empty_observations() -> None:
    deltas = a.sensitivity_deltas(0.9, 0.9, [])
    assert "No perturbations" in a.sensitivity_verdict(deltas)


# ---------------------------------------------------------------------------
# compare_duration_s
# ---------------------------------------------------------------------------


def test_compare_duration_s_positive_delta() -> None:
    cmp_ = a.compare_duration_s("ph hold", ours_s=2400.0, theirs_s=2220.0)  # 40min vs 37min

    assert cmp_.delta_s == pytest.approx(180.0)
    assert cmp_.delta_pct == pytest.approx(180.0 / 2220.0 * 100.0)


def test_compare_duration_s_negative_delta() -> None:
    cmp_ = a.compare_duration_s("ph hold", ours_s=2000.0, theirs_s=2220.0)

    assert cmp_.delta_s == pytest.approx(-220.0)
    assert cmp_.delta_pct < 0


def test_compare_duration_s_zero_reference_yields_nan_pct() -> None:
    cmp_ = a.compare_duration_s("edge case", ours_s=10.0, theirs_s=0.0)
    assert np.isnan(cmp_.delta_pct)


# ---------------------------------------------------------------------------
# format_overview_table / primary_grid_rows
# ---------------------------------------------------------------------------


def _synthetic_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run": "010726-tu_ph_tu", "variant": "fusion", "clusterer": "kmeans", "k": 4,
                "n_windows": 100, "n_valid": 99, "n_eval": 90,
                "ari": 0.5, "macro_f1": 0.4, "boundary_median_abs_s": 12.0, "silhouette": 0.3,
                "state_ari": 0.93, "state_accuracy": 0.97, "state_macro_f1": 0.39, "notes": "",
            },
            {
                "run": "010726-tu_ph_tu", "variant": "fusion", "clusterer": "kmeans", "k": 3,
                "n_windows": 100, "n_valid": 99, "n_eval": 90,
                "ari": 0.45, "macro_f1": 0.35, "boundary_median_abs_s": 40.5, "silhouette": 0.46,
                "state_ari": 0.929, "state_accuracy": 0.97, "state_macro_f1": 0.39,
                "notes": "k-sweep",
            },
            {
                "run": "270626-x", "variant": "fusion", "clusterer": "kmeans", "k": 4,
                "n_windows": 18716, "n_valid": 17946, "n_eval": 0,
                "ari": np.nan, "macro_f1": np.nan, "boundary_median_abs_s": np.nan,
                "silhouette": 0.58, "state_ari": np.nan, "state_accuracy": np.nan,
                "state_macro_f1": np.nan, "notes": "no SCADA coverage",
            },
        ]
    )


def test_primary_grid_rows_excludes_k_sweep_and_no_scada_rows() -> None:
    summary = _synthetic_summary()

    rows = a.primary_grid_rows(summary)

    assert len(rows) == 1
    assert rows.iloc[0]["run"] == "010726-tu_ph_tu"
    assert rows.iloc[0]["notes"] == ""


def test_format_overview_table_renders_nan_as_na_and_rounds_floats() -> None:
    summary = _synthetic_summary()

    table = a.format_overview_table(summary)

    assert table.loc[0, "state_ari"] == "0.930"
    no_scada_row = table[table["run"] == "270626-x"].iloc[0]
    assert no_scada_row["state_ari"] == "n/a"
    assert no_scada_row["ari"] == "n/a"


# ---------------------------------------------------------------------------
# _dataframe_to_markdown
# ---------------------------------------------------------------------------


def test_dataframe_to_markdown_renders_header_separator_and_rows() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    text = a._dataframe_to_markdown(df)

    lines = text.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| 1 | x |"
    assert lines[3] == "| 2 | y |"


def test_dataframe_to_markdown_with_index_label() -> None:
    df = pd.DataFrame({"audio": [0.9]}, index=["turbine"])

    text = a._dataframe_to_markdown(df, index_label="state")

    lines = text.splitlines()
    assert lines[0] == "| state | audio |"
    assert lines[2] == "| turbine | 0.9 |"


# ---------------------------------------------------------------------------
# _coalesced_state_sequence / _longest_run_of_state / _majority_mapped_segments
# ---------------------------------------------------------------------------


def test_majority_mapped_segments_adds_state_column() -> None:
    segments = pd.DataFrame({"cluster": [0, 1, 0], "duration_s": [10.0, 20.0, 30.0]})

    out = a._majority_mapped_segments(segments, {0: "turbine", 1: "phase-shifter"})

    assert list(out["state"]) == ["turbine", "phase-shifter", "turbine"]


def test_majority_mapped_segments_unmapped_cluster_is_unknown() -> None:
    segments = pd.DataFrame({"cluster": [-1], "duration_s": [1.0]})

    out = a._majority_mapped_segments(segments, {0: "turbine"})

    assert list(out["state"]) == ["unknown"]


def test_coalesced_state_sequence_collapses_adjacent_duplicates() -> None:
    segments = pd.DataFrame({"state": ["turbine", "turbine", "phase-shifter", "turbine"]})

    assert a._coalesced_state_sequence(segments) == ["turbine", "phase-shifter", "turbine"]


def test_longest_run_of_state_coalesces_across_cluster_boundaries() -> None:
    # Two consecutive segments both mapped to "phase-shifter" (different raw
    # cluster ids) must count as ONE contiguous run for duration purposes.
    segments = pd.DataFrame(
        {
            "cluster": [0, 2, 1],
            "state": ["turbine", "phase-shifter", "phase-shifter"],
            "duration_s": [100.0, 30.0, 45.0],
        }
    )

    longest = a._longest_run_of_state(segments, "phase-shifter")

    assert longest == pytest.approx(75.0)


def test_longest_run_of_state_returns_none_when_state_absent() -> None:
    segments = pd.DataFrame({"cluster": [0], "state": ["turbine"], "duration_s": [10.0]})

    assert a._longest_run_of_state(segments, "phase-shifter") is None


# ---------------------------------------------------------------------------
# state_holds / _drop_brief_unknown_segments
# ---------------------------------------------------------------------------


def _segments_df(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Build a segments-with-state frame from (state, start_iso, end_iso) tuples;
    duration_s derived from the timestamps (mirrors `to_segments`' invariant)."""
    starts = pd.to_datetime([r[1] for r in rows], utc=True)
    ends = pd.to_datetime([r[2] for r in rows], utc=True)
    return pd.DataFrame(
        {
            "state": [r[0] for r in rows],
            "start_utc": starts,
            "end_utc": ends,
            "duration_s": [(e - s).total_seconds() for s, e in zip(starts, ends, strict=True)],
        }
    )


def test_state_holds_bridges_sub_tolerance_slivers_into_one_hold() -> None:
    # Replicates the real 290626-tu fragmentation shape: PH fragments separated by
    # 1-second unknown slivers at burst-file boundaries. All four fragments must
    # merge into ONE hold; envelope includes the slivers, summed excludes them.
    segments = _segments_df(
        [
            ("phase-shifter", "2026-06-29T02:30:57", "2026-06-29T02:41:59"),  # 662 s
            ("unknown", "2026-06-29T02:41:59", "2026-06-29T02:42:00"),  # 1 s sliver
            ("phase-shifter", "2026-06-29T02:42:00", "2026-06-29T02:53:59"),  # 719 s
            ("unknown", "2026-06-29T02:53:59", "2026-06-29T02:54:00"),  # 1 s sliver
            ("phase-shifter", "2026-06-29T02:54:00", "2026-06-29T03:05:59"),  # 719 s
            ("unknown", "2026-06-29T03:05:59", "2026-06-29T03:06:00"),  # 1 s sliver
            ("phase-shifter", "2026-06-29T03:06:00", "2026-06-29T03:10:22"),  # 262 s
            ("turbine", "2026-06-29T03:10:22", "2026-06-29T06:00:00"),
        ]
    )

    holds = a.state_holds(segments, "phase-shifter", gap_tolerance_s=60.0)

    assert len(holds) == 1
    hold = holds[0]
    assert hold.n_fragments == 4
    assert hold.envelope_s == pytest.approx(2365.0)  # 02:30:57 -> 03:10:22, slivers included
    assert hold.summed_s == pytest.approx(662.0 + 719.0 + 719.0 + 262.0)  # 2362, excluded


def test_state_holds_does_not_bridge_gaps_above_tolerance() -> None:
    segments = _segments_df(
        [
            ("phase-shifter", "2026-06-29T02:00:00", "2026-06-29T02:10:00"),
            ("turbine", "2026-06-29T02:10:00", "2026-06-29T05:00:00"),  # ~3 h apart
            ("phase-shifter", "2026-06-29T05:00:00", "2026-06-29T05:05:00"),
        ]
    )

    holds = a.state_holds(segments, "phase-shifter", gap_tolerance_s=60.0)

    assert len(holds) == 2
    assert holds[0].summed_s == pytest.approx(600.0)
    assert holds[0].n_fragments == 1
    assert holds[1].summed_s == pytest.approx(300.0)


def test_state_holds_gap_exactly_at_tolerance_is_bridged() -> None:
    # Boundary semantics: gap <= tolerance bridges (docstring contract).
    segments = _segments_df(
        [
            ("phase-shifter", "2026-06-29T02:00:00", "2026-06-29T02:10:00"),
            ("unknown", "2026-06-29T02:10:00", "2026-06-29T02:11:00"),  # exactly 60 s
            ("phase-shifter", "2026-06-29T02:11:00", "2026-06-29T02:20:00"),
        ]
    )

    holds = a.state_holds(segments, "phase-shifter", gap_tolerance_s=60.0)

    assert len(holds) == 1
    assert holds[0].n_fragments == 2
    assert holds[0].envelope_s == pytest.approx(1200.0)
    assert holds[0].summed_s == pytest.approx(1140.0)


def test_state_holds_returns_empty_for_absent_state() -> None:
    segments = _segments_df([("turbine", "2026-06-29T02:00:00", "2026-06-29T03:00:00")])

    assert a.state_holds(segments, "phase-shifter", gap_tolerance_s=60.0) == []


def test_state_holds_accepts_string_timestamps_from_raw_csv() -> None:
    # segments.csv read WITHOUT parse_dates yields string timestamps -- state_holds
    # must parse them itself rather than crash on str - str arithmetic.
    segments = pd.DataFrame(
        {
            "state": ["phase-shifter", "phase-shifter"],
            "start_utc": ["2026-06-29 02:00:00+00:00", "2026-06-29 02:10:01+00:00"],
            "end_utc": ["2026-06-29 02:10:00+00:00", "2026-06-29 02:20:00+00:00"],
            "duration_s": [600.0, 599.0],
        }
    )

    holds = a.state_holds(segments, "phase-shifter", gap_tolerance_s=60.0)

    assert len(holds) == 1
    assert holds[0].summed_s == pytest.approx(1199.0)


def test_drop_brief_unknown_segments_removes_only_sub_tolerance_unknowns() -> None:
    segments = _segments_df(
        [
            ("phase-shifter", "2026-06-29T02:00:00", "2026-06-29T02:10:00"),
            ("unknown", "2026-06-29T02:10:00", "2026-06-29T02:10:01"),  # 1 s sliver -> drop
            ("turbine", "2026-06-29T02:10:01", "2026-06-29T03:00:00"),
            ("unknown", "2026-06-29T03:00:00", "2026-06-29T03:30:00"),  # 30 min genuine -> keep
            ("turbine", "2026-06-29T03:30:00", "2026-06-29T04:00:00"),
        ]
    )

    out = a._drop_brief_unknown_segments(segments, max_duration_s=60.0)

    assert list(out["state"]) == ["phase-shifter", "turbine", "unknown", "turbine"]


def test_drop_brief_unknown_segments_never_drops_known_states() -> None:
    # A brief KNOWN-state segment (e.g. a 10 s turbine blip) is real detector
    # output and must survive regardless of duration.
    segments = _segments_df(
        [
            ("phase-shifter", "2026-06-29T02:00:00", "2026-06-29T02:10:00"),
            ("turbine", "2026-06-29T02:10:00", "2026-06-29T02:10:10"),  # 10 s, kept
            ("phase-shifter", "2026-06-29T02:10:10", "2026-06-29T02:20:00"),
        ]
    )

    out = a._drop_brief_unknown_segments(segments, max_duration_s=60.0)

    assert list(out["state"]) == ["phase-shifter", "turbine", "phase-shifter"]


# ---------------------------------------------------------------------------
# _baseline_gt_value / _perturbed_gt_rules
# ---------------------------------------------------------------------------


def test_perturbed_gt_rules_scales_only_the_named_factor() -> None:
    from rowii.config import GtRules

    baseline = GtRules()

    perturbed = a._perturbed_gt_rules(baseline, "power_eps_mw", 0.5)

    assert perturbed.power_eps_mw == pytest.approx(baseline.power_eps_mw * 0.5)
    assert perturbed.ph_min_dwell_s == baseline.ph_min_dwell_s
    assert perturbed.transition_buffer_s == baseline.transition_buffer_s


def test_perturbed_gt_rules_matches_task_specified_absolute_values() -> None:
    from rowii.config import GtRules

    baseline = GtRules()
    assert baseline.power_eps_mw == pytest.approx(2.0)
    assert baseline.ph_min_dwell_s == pytest.approx(600.0)
    assert baseline.transition_buffer_s == pytest.approx(10.0)

    expected = {
        ("power_eps_mw", 0.5): 1.0,
        ("power_eps_mw", 2.0): 4.0,
        ("ph_min_dwell_s", 0.5): 300.0,
        ("ph_min_dwell_s", 2.0): 1200.0,
        ("transition_buffer_s", 0.5): 5.0,
        ("transition_buffer_s", 2.0): 20.0,
    }
    for (factor, multiplier), value in expected.items():
        perturbed = a._perturbed_gt_rules(baseline, factor, multiplier)
        assert a._baseline_gt_value(perturbed, factor) == pytest.approx(value)


def test_baseline_gt_value_reads_the_named_field() -> None:
    from rowii.config import GtRules

    rules = GtRules()
    assert a._baseline_gt_value(rules, "power_eps_mw") == rules.power_eps_mw
    assert a._baseline_gt_value(rules, "ph_min_dwell_s") == rules.ph_min_dwell_s
    assert a._baseline_gt_value(rules, "transition_buffer_s") == rules.transition_buffer_s


def test_unknown_sensitivity_factor_raises() -> None:
    from rowii.config import GtRules

    with pytest.raises(ValueError, match="unknown sensitivity factor"):
        a._perturbed_gt_rules(GtRules(), "not_a_field", 1.0)
    with pytest.raises(ValueError, match="unknown sensitivity factor"):
        a._baseline_gt_value(GtRules(), "not_a_field")


# ---------------------------------------------------------------------------
# plot smoke tests (matches tests/test_report.py's own "file exists and is
# reasonably sized" convention -- no pixel-content assertions).
# ---------------------------------------------------------------------------


def test_plot_overview_writes_a_nonempty_png(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "run": ["010726-tu_ph_tu", "290626-tu"],
            "variant": ["fusion", "fusion"],
            "state_accuracy": [0.9728, 0.9512],
            "state_ari": [0.9296, 0.8941],
        }
    )

    a.plot_overview(rows, tmp_path / "overview.png")

    assert (tmp_path / "overview.png").stat().st_size > 1024


def test_plot_error_histogram_writes_a_nonempty_png(tmp_path: Path) -> None:
    table = _window_table(
        gt_states=["a", "a", "b"], pred_states=["x", "a", "y"], distances=[1.0, 20.0, 45.0]
    )
    hist = a.error_anatomy_histogram(table)

    a.plot_error_histogram(hist, tmp_path / "error_vs_boundary.png")

    assert (tmp_path / "error_vs_boundary.png").stat().st_size > 1024


# ---------------------------------------------------------------------------
# Real-data integration checks (mirrors tests/test_real_data.py's own
# skip-if-no-data convention): the recompute path must reproduce
# results/summary.csv's already-recorded baseline for at least one combination.
# ---------------------------------------------------------------------------

_DATA_ROOT = load_config().data_root
_HAS_DATA_ROOT = _DATA_ROOT.is_dir()
_RESULTS_ROOT = load_config().results_root
_TU_PH_TU_FUSION_DIR = _RESULTS_ROOT / "010726-tu_ph_tu" / "fusion-kmeans"
_HAS_TU_PH_TU_FUSION = (_TU_PH_TU_FUSION_DIR / "frame_labels.parquet").exists()

skip_reason = "ROWII_DATA_ROOT is unset or does not point at an existing directory"
skip_reason_artifacts = "results/010726-tu_ph_tu/fusion-kmeans artifacts not present"


@pytest.mark.data
def test_recomputed_gt_reproduces_summary_csv_baseline_for_tu_ph_tu_fusion() -> None:
    if not _HAS_DATA_ROOT:
        pytest.skip(skip_reason)
    if not _HAS_TU_PH_TU_FUSION:
        pytest.skip(skip_reason_artifacts)

    cfg = load_config()
    index = discover(cfg.data_root)
    summary = pd.read_csv(cfg.results_root / "summary.csv")
    recorded = summary[
        (summary["run"] == "010726-tu_ph_tu")
        & (summary["variant"] == "fusion")
        & (summary["clusterer"] == "kmeans")
        & (summary["notes"].fillna("") == "")
    ].iloc[0]

    rv = a.load_run_variant_eval("010726-tu_ph_tu", "fusion", "kmeans", cfg, index)

    assert rv.ev.state_ari == pytest.approx(float(recorded["state_ari"]), abs=1e-6)
    assert rv.ev.state_accuracy == pytest.approx(float(recorded["state_accuracy"]), abs=1e-6)
    assert rv.ev.n_eval_windows == int(recorded["n_eval"])


@pytest.mark.data
def test_load_run_variant_eval_perturbed_gt_rules_changes_nothing_but_gt() -> None:
    if not _HAS_DATA_ROOT:
        pytest.skip(skip_reason)
    if not _HAS_TU_PH_TU_FUSION:
        pytest.skip(skip_reason_artifacts)


    cfg = load_config()
    index = discover(cfg.data_root)

    baseline = a.load_run_variant_eval("010726-tu_ph_tu", "fusion", "kmeans", cfg, index)
    perturbed_rules = a._perturbed_gt_rules(cfg.gt, "power_eps_mw", 2.0)
    perturbed = a.load_run_variant_eval(
        "010726-tu_ph_tu", "fusion", "kmeans", cfg, index, gt_rules=perturbed_rules
    )

    # Predictions (cluster ids) are identical -- only GT rules changed.
    np.testing.assert_array_equal(baseline.cluster, perturbed.cluster)
