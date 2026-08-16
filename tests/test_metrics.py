import numpy as np
import pandas as pd
import pytest

from rowii.eval.metrics import EvalResult, evaluate, load_alignment
from rowii.signals.windows import WindowGrid


def _grid(n_windows: int, window_s: float = 1.0) -> WindowGrid:
    return WindowGrid(t0_ns=0, window_ns=round(window_s * 1e9), n_windows=n_windows)


def _gt(states: list[str]) -> pd.DataFrame:
    n = len(states)
    return pd.DataFrame(
        {"state": states, "load_bin": np.full(n, -1, dtype=np.int64)},
        index=pd.RangeIndex(n),
    )


def _gt_with_load_bin(states: list[str], load_bin: list[int]) -> pd.DataFrame:
    assert len(states) == len(load_bin)
    return pd.DataFrame(
        {"state": states, "load_bin": np.array(load_bin, dtype=np.int64)},
        index=pd.RangeIndex(len(states)),
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


def test_evaluate_raises_clear_error_when_all_gt_windows_are_unknown() -> None:
    """Evaluate must raise a clear ValueError when no GT windows have a known state."""
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=4)
    pred = np.array([0, 0, 1, 1], dtype=np.int64)
    gt = pd.DataFrame({"state": ["unknown"] * 4, "load_bin": [-1] * 4})
    with pytest.raises(ValueError, match="known ground-truth state"):
        evaluate(pred, gt, grid)


# ---------------------------------------------------------------------------
# Mode-level (state) metrics -- primary view
#
# The strict Hungarian mapping is a 1:1 correspondence: two predicted clusters
# that are both genuinely a turbine-load sub-cluster (a legitimate partition
# refinement -- "load sub-structure appears as extra
# clusters ... or reported as sub-clusters") get Hungarian-assigned to TWO
# DIFFERENT GT states when there are exactly as many clusters as states, which
# then penalizes ARI/macro_f1 as if the split were a genuine misclassification.
# The state-level view collapses every cluster onto its OWN majority GT state
# (independently, no 1:1 constraint) before scoring, so genuine sub-clusters of
# the same state count as fully correct.
# ---------------------------------------------------------------------------


def test_state_level_two_clusters_both_majority_turbine_count_as_turbine() -> None:
    # 2 GT states (standstill, turbine). Cluster 0 is PURELY turbine (14 windows).
    # Cluster 1 is turbine-majority but has one contaminating standstill window (1
    # standstill + 15 turbine) -- both clusters are unambiguously turbine-majority,
    # so state_mapping (independent per-cluster majority vote) correctly resolves
    # BOTH to "turbine". Hungarian's STRICT 1:1 assignment, however, is scored by
    # TOTAL matched windows: pairing (cluster0->turbine, cluster1->standstill)
    # matches 14 + 1 = 15 windows, while the "intuitive" pairing
    # (cluster0->turbine, cluster1->turbine is impossible under 1:1 -- the only
    # other valid pairing is cluster0->standstill, cluster1->turbine, matching
    # 0 + 15 = 15) -- Hungarian is indifferent on raw count here but the mapping it
    # picks forces cluster1, 15/16 of which is genuinely turbine, onto a single
    # GT state that only fits its 1 contaminating window -- this is the exact
    # failure mode being tested here: a legitimate load sub-cluster loses its
    # "turbine" identity to satisfy a global 1:1 constraint it has no reason to
    # respect. state_macro_f1 must score strictly higher than the strict
    # macro_f1 on this exact scenario (see module-level docstring above).
    states = ["turbine"] * 14 + ["standstill"] * 1 + ["turbine"] * 15
    gt = _gt(states)
    pred = np.array([0] * 14 + [1] * 1 + [1] * 15, dtype=np.int64)
    grid = _grid(30)

    result = evaluate(pred, gt, grid)

    assert result.state_mapping[0] == "turbine"
    assert result.state_mapping[1] == "turbine"
    # The strict, Hungarian-based macro_f1 must be strictly worse in this exact
    # scenario -- pinning down that the new state-level view is a genuine
    # improvement here, not just an alias for the strict one.
    assert result.state_macro_f1 > result.macro_f1


def test_state_level_degenerate_all_one_state_scores_perfectly() -> None:
    # Degenerate case: every GT window (and every cluster) is the same single
    # state. Both accuracy and macro-F1 must resolve to a clean 1.0, not NaN or a
    # divide-by-zero artifact, and state_ari (a single-partition-vs-itself
    # comparison) must also be well-defined (ARI is 1.0 by convention for
    # identical partitions, including the trivial single-cluster case per
    # sklearn's own documented behaviour).
    states = ["turbine"] * 12
    gt = _gt(states)
    pred = np.array([0] * 7 + [1] * 5, dtype=np.int64)  # 2 clusters, both pure turbine
    grid = _grid(12)

    result = evaluate(pred, gt, grid)

    assert result.state_mapping == {0: "turbine", 1: "turbine"}
    assert result.state_accuracy == pytest.approx(1.0)
    assert result.state_macro_f1 == pytest.approx(1.0)
    assert result.state_ari == pytest.approx(1.0)


def test_state_ari_between_gt_states_and_majority_collapsed_predictions() -> None:
    # state_ari collapses cluster ids onto their majority GT state name FIRST,
    # then computes ARI between that collapsed label sequence and the raw GT
    # state sequence -- unlike the strict `ari` field (raw cluster ids vs GT), a
    # perfect partition-preserving split (every cluster pure, one-to-many onto
    # GT states) must score exactly 1.0 here even though the strict `ari` is
    # penalized for the same over-segmentation.
    states = ["standstill"] * 10 + ["turbine"] * 10 + ["turbine"] * 10
    gt = _gt(states)
    pred = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int64)
    grid = _grid(30)

    result = evaluate(pred, gt, grid)

    assert result.state_ari == pytest.approx(1.0)
    assert result.ari < 1.0


def test_state_mapping_uses_majority_vote_not_hungarian_1to1() -> None:
    # A single cluster (0) spans BOTH standstill and turbine windows, with
    # turbine as the majority (7 vs 3) -- state_mapping must resolve it to
    # "turbine" (plain majority vote), independent of any other cluster's
    # assignment (no 1:1 constraint to violate, unlike Hungarian).
    states = ["standstill"] * 3 + ["turbine"] * 7
    gt = _gt(states)
    pred = np.array([0] * 10, dtype=np.int64)
    grid = _grid(10)

    result = evaluate(pred, gt, grid)

    assert result.state_mapping == {0: "turbine"}


def test_state_mapping_only_covers_eval_window_clusters_like_strict_mapping() -> None:
    # Mirrors test_unknown_windows_are_dropped_before_all_metrics' garbage-cluster check,
    # but for state_mapping: a cluster confined entirely to "unknown" windows must not
    # appear in state_mapping either (state_mapping is computed on eval windows only,
    # same restriction _hungarian_mapping already applies).
    states = ["standstill"] * 5 + ["turbine"] * 5 + ["unknown"] * 5
    gt = _gt(states)
    pred = np.array([0] * 5 + [1] * 5 + [99] * 5, dtype=np.int64)
    grid = _grid(15)

    result = evaluate(pred, gt, grid)

    assert 99 not in result.state_mapping


# ---------------------------------------------------------------------------
# state_confusion: majority-mapped confusion matrix, the state-level (primary)
# counterpart to the strict/Hungarian-mapped `confusion` (report-clarity fix).
# ---------------------------------------------------------------------------


def test_state_confusion_is_dataframe_with_gt_states_as_rows() -> None:
    states = ["standstill"] * 3 + ["turbine"] * 3 + ["pump"] * 3
    gt = _gt(states)
    pred = np.array([0] * 3 + [1] * 3 + [2] * 3, dtype=np.int64)
    grid = _grid(9)

    result = evaluate(pred, gt, grid)

    assert isinstance(result.state_confusion, pd.DataFrame)
    assert set(result.state_confusion.index) == {"standstill", "turbine", "pump"}


def test_perfect_prediction_state_confusion_matrix_is_diagonal() -> None:
    # Majority and Hungarian mappings coincide on a perfect, unambiguous prediction
    # (every cluster pure), so state_confusion must be diagonal exactly like confusion.
    states = ["standstill"] * 5 + ["turbine"] * 5
    gt = _gt(states)
    pred = np.array([7] * 5 + [3] * 5, dtype=np.int64)
    grid = _grid(10)

    result = evaluate(pred, gt, grid)

    assert result.state_confusion.loc["standstill", "standstill"] == 5
    assert result.state_confusion.loc["turbine", "turbine"] == 5
    assert result.state_confusion.to_numpy().sum() == 10
    non_diagonal_cells = [
        result.state_confusion.loc[row, col]
        for row in result.state_confusion.index
        for col in result.state_confusion.columns
        if row != col
    ]
    assert sum(non_diagonal_cells) == 0


def test_state_confusion_diverges_from_confusion_when_majority_disagrees_hungarian() -> None:
    # Reproduces the exact real-world divergence these two fields exist to make
    # legible (see `results/010726-tu_ph_tu/vibration-kmeans/report.md`, cluster 3):
    # one predicted cluster's OWN majority GT state differs from the GT state the
    # global 1:1 Hungarian assignment forces onto it.
    #
    # Cluster 0 is pure "transition" (20 windows) -- unambiguous either way.
    # Cluster 1 mixes 10 "standstill" + 12 "transition" windows: taken alone, its
    # own majority is "transition" (12 > 10). But Hungarian's 1:1 assignment
    # maximizes the GLOBAL matched-window count across BOTH clusters at once, and
    # pairing (standstill -> cluster1, transition -> cluster0) matches 10 + 20 = 30
    # windows, beating the alternative (standstill -> cluster0, transition ->
    # cluster1)'s 0 + 12 = 12 -- so Hungarian forces cluster1 onto "standstill" even
    # though "transition" is its own majority.
    states = ["transition"] * 20 + ["standstill"] * 10 + ["transition"] * 12
    gt = _gt(states)
    pred = np.array([0] * 20 + [1] * 10 + [1] * 12, dtype=np.int64)
    grid = _grid(42)

    result = evaluate(pred, gt, grid)

    # Premise: the two schemes must actually disagree on cluster 1, or this test
    # would not be pinning down a real divergence.
    assert result.mapping[1] == "standstill"
    assert result.state_mapping[1] == "transition"

    # `confusion` (strict/Hungarian): cluster 1 is named "standstill" throughout --
    # its own 10 GT-standstill windows land on the diagonal, but its 12 GT-transition
    # windows are misattributed to the "standstill" predicted column.
    assert result.confusion.loc["standstill", "standstill"] == 10
    assert result.confusion.loc["transition", "standstill"] == 12
    assert result.confusion.loc["transition", "transition"] == 20
    assert result.confusion.to_numpy().sum() == 42

    # `state_confusion` (majority): cluster 1 is named "transition" instead -- its
    # 10 GT-standstill windows now land under the "transition" predicted column
    # (the divergent cluster's windows following majority, not Hungarian), and
    # "standstill" never appears as a predicted column at all, because NEITHER
    # cluster's own majority is "standstill".
    assert "standstill" not in result.state_confusion.columns
    assert result.state_confusion.loc["standstill", "transition"] == 10
    assert result.state_confusion.loc["transition", "transition"] == 32
    assert result.state_confusion.to_numpy().sum() == 42


# ---------------------------------------------------------------------------
# load_alignment: sub-cluster vs load-bin analysis
# ---------------------------------------------------------------------------


def test_load_alignment_clusters_matching_load_bins_exactly_score_ari_one() -> None:
    # 3 load bins within the turbine state, 3 predicted clusters that reproduce the
    # bins exactly under an arbitrary id permutation -- ARI must be exactly 1.0.
    states = ["turbine"] * 30
    load_bin = [0] * 10 + [1] * 10 + [2] * 10
    gt = _gt_with_load_bin(states, load_bin)
    pred = np.array([7] * 10 + [3] * 10 + [9] * 10, dtype=np.int64)

    result = load_alignment(pred, gt)

    assert result is not None
    assert result.attrs["ari"] == pytest.approx(1.0)


def test_load_alignment_fewer_than_two_load_bins_returns_none() -> None:
    # A single load bin within the turbine state gives nothing to align clusters
    # against -- must return None rather than a degenerate ARI.
    states = ["turbine"] * 10
    load_bin = [0] * 10
    gt = _gt_with_load_bin(states, load_bin)
    pred = np.array([0] * 5 + [1] * 5, dtype=np.int64)

    result = load_alignment(pred, gt)

    assert result is None


def test_load_alignment_excludes_unknown_windows() -> None:
    # Unknown windows (no SCADA coverage) must never enter the crosstab or the ARI --
    # here they carry a garbage load_bin/cluster id that would corrupt the alignment
    # if not excluded.
    states = ["unknown"] * 5 + ["turbine"] * 20
    load_bin = [-1] * 5 + [0] * 10 + [1] * 10
    gt = _gt_with_load_bin(states, load_bin)
    pred = np.array([99] * 5 + [0] * 10 + [1] * 10, dtype=np.int64)

    result = load_alignment(pred, gt)

    assert result is not None
    assert result.attrs["ari"] == pytest.approx(1.0)
    assert result.to_numpy().sum() == 20
    assert 99 not in result.index


def test_load_alignment_restricts_to_turbine_state_only() -> None:
    # Standstill/transition windows (load_bin == -1 by construction, per gt_labels)
    # must not appear in the crosstab even though they are eval windows -- only
    # "turbine" windows are in scope.
    states = ["standstill"] * 5 + ["turbine"] * 10 + ["transition"] * 5
    load_bin = [-1] * 5 + ([0] * 5 + [1] * 5) + [-1] * 5
    gt = _gt_with_load_bin(states, load_bin)
    pred = np.array([9] * 5 + [0] * 5 + [1] * 5 + [9] * 5, dtype=np.int64)

    result = load_alignment(pred, gt)

    assert result is not None
    assert result.to_numpy().sum() == 10
    assert 9 not in result.index


def test_load_alignment_falls_back_to_pump_when_no_turbine_windows() -> None:
    # No turbine windows at all in this run (e.g. a pump-only recording) -- the
    # brief's documented fallback restricts to "pump" instead.
    states = ["pump"] * 20
    load_bin = [0] * 10 + [1] * 10
    gt = _gt_with_load_bin(states, load_bin)
    pred = np.array([5] * 10 + [6] * 10, dtype=np.int64)

    result = load_alignment(pred, gt)

    assert result is not None
    assert result.attrs["ari"] == pytest.approx(1.0)
    assert result.to_numpy().sum() == 20


def test_load_alignment_returns_none_when_no_turbine_or_pump_windows_at_all() -> None:
    # Neither turbine nor pump windows present (e.g. an all-standstill run) -- there
    # is no subset to analyze at all, so this must return None like the
    # too-few-load-bins case, not raise.
    states = ["standstill"] * 10
    load_bin = [-1] * 10
    gt = _gt_with_load_bin(states, load_bin)
    pred = np.array([0] * 5 + [1] * 5, dtype=np.int64)

    result = load_alignment(pred, gt)

    assert result is None


def test_load_alignment_crosstab_shape_and_index_names() -> None:
    # The crosstab's rows/cols must be cluster id x load_bin (order not asserted
    # beyond containing exactly the right labels), and the returned object must be
    # a real DataFrame so callers can render it directly.
    states = ["turbine"] * 20
    load_bin = [0] * 10 + [1] * 10
    gt = _gt_with_load_bin(states, load_bin)
    pred = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    result = load_alignment(pred, gt)

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert set(result.index) == {0, 1}
    assert set(result.columns) == {0, 1}
    assert "ari" in result.attrs


def test_derive_state_names_maps_clean_two_mode() -> None:
    from rowii.eval.metrics import derive_state_names
    gt = np.array(["turbine"] * 30 + ["pump"] * 30, dtype=object)
    pred = np.array([0] * 30 + [1] * 30, dtype=np.int64)
    names = derive_state_names(gt, pred, fitted_ids=[0, 1])
    assert names == {0: "turbine", 1: "pump"}


def test_derive_state_names_masks_both_unknown_and_transition() -> None:
    """BOTH masked before the vote -- narrower than evaluate's unknown-only."""
    from rowii.eval.metrics import derive_state_names
    gt = np.array(["turbine", "transition", "unknown", "turbine", "turbine"], dtype=object)
    pred = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    names = derive_state_names(gt, pred, fitted_ids=[0])
    # only the 3 turbine windows count; cluster 0's plurality is 3/3 -> turbine.
    assert names == {0: "turbine"}


def test_derive_state_names_fallback_cluster_absent_from_masked_pred() -> None:
    from rowii.eval.metrics import derive_state_names
    # cluster 1 appears ONLY on transition/unknown windows -> zero GT-known -> fallback.
    gt = np.array(["turbine", "turbine", "transition", "unknown"], dtype=object)
    pred = np.array([0, 0, 1, 1], dtype=np.int64)
    names = derive_state_names(gt, pred, fitted_ids=[0, 1])
    assert names == {0: "turbine", 1: "cluster-1"}


def test_derive_state_names_fallback_sub_50pct_plurality() -> None:
    from rowii.eval.metrics import derive_state_names
    # cluster 0's masked windows split 2 turbine / 3 pump -> winner (pump) = 3/5 = 60% >= 50%.
    gt_ok = np.array(["turbine", "turbine", "pump", "pump", "pump"], dtype=object)
    pred = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    assert derive_state_names(gt_ok, pred, [0]) == {0: "pump"}
    # now a true <50% plurality: 2 turbine / 2 pump / 1 standstill -> max 2/5 = 40% -> fallback.
    gt_tie = np.array(["turbine", "turbine", "pump", "pump", "standstill"], dtype=object)
    assert derive_state_names(gt_tie, pred, [0]) == {0: "cluster-0"}


def test_derive_state_names_exact_50pct_plurality_keeps_name() -> None:
    """Boundary pin: `frac >= min_plurality` uses `>=`, so an
    EXACT tie at the default 50% plurality KEEPS the majority name rather than
    falling back -- the complement of test_derive_state_names_fallback_sub_50pct_
    plurality's 40%-fallback / 60%-keep cases, neither of which touches the
    boundary itself."""
    from rowii.eval.metrics import derive_state_names
    # cluster 0's masked windows split EXACTLY 2 turbine / 2 pump. _majority_mapping
    # breaks the tie via pandas idxmax's first-in-(sorted)-index order -> "pump"
    # (state_names sorted alphabetically: "pump" < "turbine"), whose own plurality
    # is exactly 2/4 = 0.5 == min_plurality.
    gt = np.array(["turbine", "turbine", "pump", "pump"], dtype=object)
    pred = np.array([0, 0, 0, 0], dtype=np.int64)
    assert derive_state_names(gt, pred, [0]) == {0: "pump"}


def test_derive_state_names_fills_over_all_fitted_ids() -> None:
    from rowii.eval.metrics import derive_state_names
    gt = np.array(["turbine"] * 10, dtype=object)
    pred = np.array([0] * 10, dtype=np.int64)
    # fitted id 2 never appears in pred -> filled as its bare name.
    names = derive_state_names(gt, pred, fitted_ids=[0, 1, 2])
    assert names == {0: "turbine", 1: "cluster-1", 2: "cluster-2"}


def test_derive_state_names_all_fallback_when_no_gt_known() -> None:
    from rowii.eval.metrics import derive_state_names
    gt = np.array(["unknown", "transition", "unknown"], dtype=object)
    pred = np.array([0, 1, 0], dtype=np.int64)
    assert derive_state_names(gt, pred, [0, 1]) == {0: "cluster-0", 1: "cluster-1"}
