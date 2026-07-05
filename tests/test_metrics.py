import numpy as np
import pandas as pd
import pytest

from rowii.eval.metrics import EvalResult, evaluate
from rowii.signals.windows import WindowGrid


def _grid(n_windows: int, window_s: float = 1.0) -> WindowGrid:
    return WindowGrid(t0_ns=0, window_ns=round(window_s * 1e9), n_windows=n_windows)


def _gt(states: list[str]) -> pd.DataFrame:
    n = len(states)
    return pd.DataFrame(
        {"state": states, "load_bin": np.full(n, -1, dtype=np.int64)},
        index=pd.RangeIndex(n),
    )


def test_perfect_prediction_with_permuted_cluster_ids_reaches_ari_and_f1_of_one() -> None:
    # 3 blocks of 10 windows, GT states standstill/turbine/pump in order. Predicted
    # cluster ids are an arbitrary PERMUTATION of {0, 1, 2} relative to GT block order
    # (block 0 -> cluster 2, block 1 -> cluster 0, block 2 -> cluster 1): Hungarian
    # matching must recover this permutation and map cluster ids back onto GT names
    # exactly, so the metrics are unaffected by the arbitrary cluster numbering.
    states = ["standstill"] * 10 + ["turbine"] * 10 + ["pump"] * 10
    gt = _gt(states)
    pred = np.array([2] * 10 + [0] * 10 + [1] * 10, dtype=np.int64)
    grid = _grid(30)

    result = evaluate(pred, gt, grid)

    assert isinstance(result, EvalResult)
    assert result.ari == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.mapping == {2: "standstill", 0: "turbine", 1: "pump"}
    assert result.n_eval_windows == 30
    assert result.boundary_median_abs_s == pytest.approx(0.0)


def test_perfect_prediction_confusion_matrix_is_diagonal() -> None:
    states = ["standstill"] * 5 + ["turbine"] * 5
    gt = _gt(states)
    pred = np.array([7] * 5 + [3] * 5, dtype=np.int64)
    grid = _grid(10)

    result = evaluate(pred, gt, grid)

    # rows = GT states, cols = mapped predicted states; a perfect prediction puts
    # every count on the diagonal and nothing anywhere else.
    assert result.confusion.loc["standstill", "standstill"] == 5
    assert result.confusion.loc["turbine", "turbine"] == 5
    assert result.confusion.to_numpy().sum() == 10
    non_diagonal_cells = [
        result.confusion.loc[row, col]
        for row in result.confusion.index
        for col in result.confusion.columns
        if row != col
    ]
    assert sum(non_diagonal_cells) == 0


def test_unknown_windows_are_dropped_before_all_metrics() -> None:
    # 10 known windows (perfect match) + 5 "unknown" windows where the prediction is
    # garbage (a cluster id never used elsewhere) -- the unknown windows must not
    # appear in n_eval_windows nor influence ari/macro_f1 at all.
    states = ["standstill"] * 5 + ["turbine"] * 5 + ["unknown"] * 5
    gt = _gt(states)
    pred = np.array([0] * 5 + [1] * 5 + [99] * 5, dtype=np.int64)
    grid = _grid(15)

    result = evaluate(pred, gt, grid)

    assert result.n_eval_windows == 10
    assert result.ari == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(1.0)
    # Garbage-only cluster id 99 must not leak into the mapping (it has no eval-window
    # presence at all, since every window it covers was dropped as unknown).
    assert 99 not in result.mapping
    assert result.confusion.to_numpy().sum() == 10


def test_k_greater_than_states_extra_cluster_maps_to_its_best_fit_state() -> None:
    # 2 GT states (standstill, turbine), 3 predicted clusters. Cluster 0 and 1 each
    # perfectly cover one GT state (Hungarian pairs them 1:1); cluster 2 is an "extra"
    # split of the turbine block (no GT state left unmatched to pair it with) and must
    # fall back to mapping onto whichever GT state its own column maximizes -- here
    # entirely turbine windows, so cluster 2 -> "turbine". Splitting one GT block into
    # two predicted clusters is a genuine over-segmentation that ARI is SUPPOSED to
    # penalize (it compares raw cluster ids, not mapped names) -- macro_f1, computed
    # AFTER mapping folds clusters 1 and 2 back onto the single "turbine" label, is
    # the metric that stays perfect here, which is what this test actually pins down.
    states = ["standstill"] * 10 + ["turbine"] * 6 + ["turbine"] * 4
    gt = _gt(states)
    pred = np.array([0] * 10 + [1] * 6 + [2] * 4, dtype=np.int64)
    grid = _grid(20)

    result = evaluate(pred, gt, grid)

    assert result.mapping[0] == "standstill"
    assert result.mapping[1] == "turbine"
    assert result.mapping[2] == "turbine"
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.ari < 1.0


def test_boundary_is_none_when_gt_has_no_state_changes() -> None:
    gt = _gt(["standstill"] * 10)
    pred = np.array([0] * 5 + [1] * 5, dtype=np.int64)
    grid = _grid(10)

    result = evaluate(pred, gt, grid)

    assert result.boundary_median_abs_s is None


def test_boundary_is_none_when_predicted_has_no_state_changes_but_gt_does() -> None:
    gt = _gt(["standstill"] * 5 + ["turbine"] * 5)
    pred = np.array([0] * 10, dtype=np.int64)  # single cluster, no predicted change
    grid = _grid(10)

    result = evaluate(pred, gt, grid)

    assert result.boundary_median_abs_s is None


def test_boundary_median_equals_shift_amount_for_uniformly_shifted_prediction() -> None:
    # GT changes state at window 20 (standstill -> turbine) and window 40
    # (turbine -> pump). The predicted cluster sequence reproduces the exact same
    # two-change shape but shifted 3 windows later; at window_s=2.0s, a 3-window
    # shift is 6.0s, and since both boundaries shift identically the median (of the
    # two |Δt| values, both exactly 6.0s) must equal exactly window_s * shift = 6.0.
    window_s = 2.0
    shift = 3
    n = 60
    states = ["standstill"] * 20 + ["turbine"] * 20 + ["pump"] * 20
    gt = _gt(states)
    pred = np.array([0] * (20 + shift) + [1] * 20 + [2] * (20 - shift), dtype=np.int64)
    grid = _grid(n, window_s=window_s)

    result = evaluate(pred, gt, grid)

    assert result.boundary_median_abs_s == pytest.approx(window_s * shift)


def test_boundary_excludes_gt_changes_into_or_out_of_unknown() -> None:
    # GT: standstill(10) -> unknown(10) -> turbine(10). The two GT "changes" at window
    # 10 (standstill->unknown) and window 20 (unknown->turbine) both touch "unknown"
    # and must NOT be counted as boundary events at all -- so with a predicted
    # sequence that has genuine changes at both those same window indices, the
    # boundary metric must still report None (zero countable GT changes), not measure
    # a spurious near-zero deviation against them.
    states = ["standstill"] * 10 + ["unknown"] * 10 + ["turbine"] * 10
    gt = _gt(states)
    pred = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)
    grid = _grid(30)

    result = evaluate(pred, gt, grid)

    assert result.boundary_median_abs_s is None


def test_mapping_keys_are_python_ints_and_values_are_state_strings() -> None:
    states = ["standstill"] * 4 + ["turbine"] * 4
    gt = _gt(states)
    pred = np.array([5] * 4 + [6] * 4, dtype=np.int64)
    grid = _grid(8)

    result = evaluate(pred, gt, grid)

    for key, value in result.mapping.items():
        assert isinstance(key, int)
        assert isinstance(value, str)


def test_confusion_is_dataframe_with_gt_states_as_rows() -> None:
    states = ["standstill"] * 3 + ["turbine"] * 3 + ["pump"] * 3
    gt = _gt(states)
    pred = np.array([0] * 3 + [1] * 3 + [2] * 3, dtype=np.int64)
    grid = _grid(9)

    result = evaluate(pred, gt, grid)

    assert isinstance(result.confusion, pd.DataFrame)
    assert set(result.confusion.index) == {"standstill", "turbine", "pump"}


def test_imperfect_prediction_yields_macro_f1_below_one() -> None:
    # 20 windows: GT is standstill(10) then turbine(10). Predicted cluster 0 covers
    # windows 0-11 (2 misclassified turbine windows as cluster 0 -> mapped
    # "standstill"), cluster 1 covers windows 12-19. Hungarian still maps 0
    # -> standstill (8 correct + 2 wrong) and 1 -> turbine (8 correct), so overall
    # accuracy is imperfect and macro_f1 must be strictly below 1.0.
    states = ["standstill"] * 10 + ["turbine"] * 10
    gt = _gt(states)
    pred = np.array([0] * 12 + [1] * 8, dtype=np.int64)
    grid = _grid(20)

    result = evaluate(pred, gt, grid)

    assert result.macro_f1 < 1.0
    assert result.ari < 1.0
