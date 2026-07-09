"""Tests for `rowii.anomaly.sweep`: mode-conditioned conformal sweep orchestration
(`run_sweep`) -- synthetic end-to-end coverage per the Step-2 plan (`docs/superpowers/
plans/2026-07-09-step2-first-package.md` Task S5). No real data.

Every numeric bound/seed used below was verified empirically against the real
implementation before being hardcoded here (scratch scripts, not committed -- see
the S5 review record for the full derivations), matching this
package's own established practice (`test_conformal.py`'s module docstring; `task-s4-
report.md`'s "Verification performed" section) for statistical/segment-split
constructions whose outcome cannot be derived by inspection alone.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rowii.anomaly.references import split_by_segments
from rowii.anomaly.sweep import (
    SweepConfig,
    SweepResult,
    _assert_three_way_disjoint,
    _scores_and_candidates,
    run_sweep,
)
from rowii.pipeline import PreparedRun
from rowii.signals.windows import WindowGrid

# ---------------------------------------------------------------------------
# Shared fixture plumbing
# ---------------------------------------------------------------------------


def _prepared_run(features: np.ndarray, segment_ids: np.ndarray) -> PreparedRun:
    """A `PreparedRun` with every window valid -- `run_sweep` only reads `features`,
    `valid_mask`, `segment_ids` (module docstring), so `grid`/`feature_names` are
    filled with cheap placeholders."""
    valid_mask = np.ones(features.shape[0], dtype=bool)
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=features.shape[0])
    return PreparedRun(
        features=features,
        grid=grid,
        valid_mask=valid_mask,
        feature_names=[f"f{i}" for i in range(features.shape[1])],
        segment_ids=segment_ids,
    )


# ---------------------------------------------------------------------------
# Fixture 1: 3-label well-separated stream + 10 injected far-out windows in label 1
# (items 1, 2, 4, 5)
# ---------------------------------------------------------------------------

_N_FEATURES = 6
_N_SEGMENTS_PER_LABEL = 20
_WINDOWS_PER_SEGMENT = 25
_N_INJECTED = 10


def _three_label_run_with_injected_outliers(
    data_seed: int = 0,
) -> tuple[PreparedRun, np.ndarray, np.ndarray]:
    """3 well-separated int-labeled Gaussian clusters (0/1/2), `_N_SEGMENTS_PER_LABEL`
    segments of `_WINDOWS_PER_SEGMENT` windows each, PLUS one extra dedicated segment of
    `_N_INJECTED` far-out windows appended to label 1 -- offset from label 1's own mean
    along feature index 4, an axis none of the three labels' own clusters otherwise use,
    so they are unambiguous outliers under kNN cosine (the default scorer) rather than
    just a bigger deviation along label 1's own existing axis. Returns `(prepared,
    labels, injected_window_indices)` -- `injected_window_indices` are the absolute row
    indices of the 10 injected windows in `prepared.features`.
    """
    rng = np.random.default_rng(data_seed)
    means = {
        0: np.zeros(_N_FEATURES),
        1: np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        2: np.array([0.0, 30.0, 0.0, 0.0, 0.0, 0.0]),
    }
    feats: list[np.ndarray] = []
    labels: list[int] = []
    segment_ids: list[int] = []
    seg = 0
    for label in (0, 1, 2):
        for _ in range(_N_SEGMENTS_PER_LABEL):
            block = means[label] + rng.normal(scale=0.5, size=(_WINDOWS_PER_SEGMENT, _N_FEATURES))
            feats.append(block)
            labels.extend([label] * _WINDOWS_PER_SEGMENT)
            segment_ids.extend([seg] * _WINDOWS_PER_SEGMENT)
            seg += 1

    injected_offset = np.array([0.0, 0.0, 0.0, 0.0, 25.0, 0.0])
    injected_window_start = sum(a.shape[0] for a in feats)
    injected_block = means[1] + injected_offset + rng.normal(
        scale=0.5, size=(_N_INJECTED, _N_FEATURES)
    )
    feats.append(injected_block)
    labels.extend([1] * _N_INJECTED)
    segment_ids.extend([seg] * _N_INJECTED)
    injected_windows = np.arange(injected_window_start, injected_window_start + _N_INJECTED)

    features = np.vstack(feats)
    labels_arr = np.array(labels, dtype=np.int64)
    segment_ids_arr = np.array(segment_ids, dtype=np.int64)
    return _prepared_run(features, segment_ids_arr), labels_arr, injected_windows


# ---------------------------------------------------------------------------
# Item 1 + 2: injected outliers dominate candidates; non-injected FAR still holds
# ---------------------------------------------------------------------------


def test_injected_outliers_dominate_candidates_and_far_holds_on_non_injected() -> None:
    # seed=8 verified (scratch script) to put all 10 injected windows on the SCORING
    # side of label 1 -- several other seeds instead route the injected segment to
    # calibration, which is a legitimate split outcome but not what this test needs.
    # Also deliberately NOT seed=0: at seed=0 every injected window already happens to
    # rank in the top 10 even with the WRONG tie-break (ascending window index instead
    # of descending score, see `_scores_and_candidates`), so it would not catch a
    # regression of that fix; seed=8 has four non-injected windows exactly tied with
    # the injected ones at the achievable-minimum p-value, and only ranks correctly
    # with the score-descending tie-break -- see task report for the mutation-testing
    # evidence.
    prepared, labels, injected = _three_label_run_with_injected_outliers()
    cfg = SweepConfig(seed=8)

    result = run_sweep(prepared, labels, cfg)

    # (1) injected windows occupy the top of label 1's candidate ranking.
    candidates_1 = result.candidates[result.candidates["label"] == 1]
    assert set(injected.tolist()).issubset(set(candidates_1["window"].tolist())), (
        "all 10 injected windows must appear among label 1's top-k candidates at all"
    )
    top12 = set(candidates_1[candidates_1["rank"] <= 12]["window"].tolist())
    assert set(injected.tolist()).issubset(top12)

    n_calibration_1 = result.far_table.set_index("label").loc[1, "n_calibration"]
    injected_p_values = (
        result.scores.set_index("window").loc[injected.tolist(), "p_value"].to_numpy()
    )
    assert (injected_p_values <= (11.0 / (n_calibration_1 + 1)) + 1e-9).all()
    # Tighter, mechanism-precise check: every injected score exceeds the ENTIRE
    # conformal-part calibration set by construction (offset +25 along an axis none of
    # the reference ever varies in), so `p_values`' formula collapses every one of them
    # to the single achievable minimum `1/(n_cal+1)` exactly (conformal.py module
    # docstring) -- not merely "small".
    np.testing.assert_allclose(injected_p_values, 1.0 / (n_calibration_1 + 1))

    # (2) realized FAR on the REST of label 1's scoring windows (excluding the 10
    # injected ones) stays within a generous mean-level band of alpha. The injected
    # windows sit only on the scoring side, never calibration (verified above via the
    # exact-minimum p-value check -- if any had leaked into calibration, `n_calibration`
    # would be affected but the injected windows themselves could not also appear in
    # `result.scores`), so calibration itself is uncontaminated; this checks the
    # ordinary in-distribution population's FAR is still what split-conformal promises.
    scores_1 = result.scores[result.scores["label"] == 1]
    non_injected = scores_1[~scores_1["window"].isin(injected.tolist())]
    n_scored = len(non_injected)
    non_injected_far = non_injected["alarm"].mean()

    band = (
        cfg.alpha
        + 3 * math.sqrt(cfg.alpha * (1 - cfg.alpha) / n_scored)
        + 1.0 / (n_calibration_1 + 1)
    )
    assert 0.0 <= non_injected_far <= band, (
        f"non-injected FAR {non_injected_far:.4f} outside generous band [0, {band:.4f}]"
    )


def test_candidate_p_value_ties_are_broken_by_descending_score_hand_computed() -> None:
    """Direct, hand-computed complement to the seed=8 end-to-end case above: conformal
    p-values collapse to the SAME achievable minimum for every score that exceeds the
    whole calibration set (`p_values` module docstring), so ties at that minimum are
    common, not a corner case, whenever more than a couple of windows are genuinely
    extreme. `_scores_and_candidates` must rank ties by descending raw score (more
    extreme first), not by window index."""
    windows = np.array([10, 20, 30, 40])
    scores = np.array([5.0, 1.0, 3.0, 100.0])
    p_vals = np.array([0.02, 0.02, 0.02, 0.02])  # all four tied
    alarms = np.array([True, True, True, True])

    _score_rows, candidate_rows = _scores_and_candidates(
        label=0, windows=windows, scores=scores, p_vals=p_vals, alarms=alarms, top_k=2
    )

    assert [c.window for c in candidate_rows] == [40, 10]  # scores 100.0, 5.0 -- NOT window order
    assert [c.rank for c in candidate_rows] == [1, 2]


# ---------------------------------------------------------------------------
# Item 4: fit/conformal/scoring windows are pairwise disjoint (no self-scoring leak)
# ---------------------------------------------------------------------------


def test_assert_three_way_disjoint_raises_on_any_pairwise_overlap() -> None:
    """Direct unit test of the internal disjointness guard (module docstring point 2,
    "kNN self-scoring hazard") -- mirrors `test_pipeline.py`'s established precedent of
    testing a `_private` helper directly (`pipeline._extract_stream_features`)."""
    _assert_three_way_disjoint(
        np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])
    )  # disjoint -- must not raise

    with pytest.raises(AssertionError):
        _assert_three_way_disjoint(np.array([1, 2]), np.array([2, 3]), np.array([4, 5]))
    with pytest.raises(AssertionError):
        _assert_three_way_disjoint(np.array([1, 2]), np.array([3, 4]), np.array([1, 5]))
    with pytest.raises(AssertionError):
        _assert_three_way_disjoint(np.array([1, 2]), np.array([3, 4]), np.array([4, 5]))


def test_fit_conformal_scoring_windows_are_pairwise_disjoint_end_to_end() -> None:
    """End-to-end complement to the direct guard test above: independently recomputes
    `run_sweep`'s own two-level split (module docstring points 1-2) and checks the
    ACTUAL produced `result.scores` windows against it -- catches a bug in HOW
    `run_sweep` calls `split_by_segments` (wrong mask, wrong seed offset, ...), not just
    a bug in the disjointness check itself.
    """
    prepared, labels, _ = _three_label_run_with_injected_outliers()
    cfg = SweepConfig(seed=3)

    result = run_sweep(prepared, labels, cfg)

    top_split = split_by_segments(
        prepared.segment_ids, prepared.valid_mask, cfg.calibration_frac, cfg.seed
    )
    calib_mask = np.zeros(prepared.features.shape[0], dtype=bool)
    calib_mask[top_split.calibration_windows] = True
    nested_split = split_by_segments(prepared.segment_ids, calib_mask, 0.5, cfg.seed + 1)
    fit_windows = set(nested_split.calibration_windows.tolist())
    conformal_windows = set(nested_split.scoring_windows.tolist())
    scoring_windows = set(top_split.scoring_windows.tolist())

    assert fit_windows.isdisjoint(conformal_windows)
    assert fit_windows.isdisjoint(scoring_windows)
    assert conformal_windows.isdisjoint(scoring_windows)

    scored_windows = set(result.scores["window"].tolist())
    assert scored_windows.issubset(scoring_windows)
    assert scored_windows.isdisjoint(fit_windows)
    assert scored_windows.isdisjoint(conformal_windows)


# ---------------------------------------------------------------------------
# Item 5: determinism
# ---------------------------------------------------------------------------


def test_run_sweep_is_deterministic_for_the_same_config() -> None:
    prepared, labels, _ = _three_label_run_with_injected_outliers()

    # Two SEPARATE (but equal-valued) SweepConfig instances -- not the same object --
    # so this cannot pass by accident via e.g. identity-based caching.
    result_a = run_sweep(prepared, labels, SweepConfig(seed=2))
    result_b = run_sweep(prepared, labels, SweepConfig(seed=2))

    pd.testing.assert_frame_equal(result_a.far_table, result_b.far_table)
    pd.testing.assert_frame_equal(result_a.scores, result_b.scores)
    pd.testing.assert_frame_equal(result_a.candidates, result_b.candidates)


# ---------------------------------------------------------------------------
# Fixture 2: two well-separated str-labeled clusters with mismatched feature scale
# (item 3)
# ---------------------------------------------------------------------------

_N_SEGMENTS_TWO_LABEL = 40
_WINDOWS_PER_SEGMENT_TWO_LABEL = 25


def _two_label_run_with_scale_mismatch(data_seed: int = 0) -> tuple[PreparedRun, np.ndarray]:
    """Two well-separated (different-mean) str-labeled Gaussian clusters: `"A"` is LOOSE
    (std=1.0), `"B"` is TIGHT (std=0.02, 50x smaller). See `sweep.py`'s module docstring
    "IMPORTANT deviation" note for why "A loose / B tight" -- not the dispatch's literal
    "tight label A" -- is the construction that actually inflates a label's pooled FAR:
    split conformal's per-label guarantee is scorer-agnostic under `conditioning=
    "per-state"` (a label's threshold only ever sees that SAME label's own conformal
    scores), so ONLY `conditioning="pooled"` (a threshold shared across labels) can move
    a label's realized FAR off alpha at all, and pooling label A in with a population
    whose scores are SYSTEMATICALLY SMALLER pulls the shared threshold DOWN below what
    A's own per-state threshold would require -- inflating A's realized FAR. Verified
    (scratch script) across 15 independent seeds and both same-mean/separated-mean
    variants: pooled FAR(A) > per-state FAR(A) in every single one.
    """
    rng = np.random.default_rng(data_seed)
    feats: list[np.ndarray] = []
    labels: list[str] = []
    segment_ids: list[int] = []
    seg = 0
    for label, mean0, std in (("A", 5.0, 1.0), ("B", 20.0, 0.02)):
        mean = np.array([mean0] + [0.0] * (_N_FEATURES - 1))
        for _ in range(_N_SEGMENTS_TWO_LABEL):
            block = mean + rng.normal(scale=std, size=(_WINDOWS_PER_SEGMENT_TWO_LABEL, _N_FEATURES))
            feats.append(block)
            labels.extend([label] * _WINDOWS_PER_SEGMENT_TWO_LABEL)
            segment_ids.extend([seg] * _WINDOWS_PER_SEGMENT_TWO_LABEL)
            seg += 1
    features = np.vstack(feats)
    labels_arr = np.array(labels)
    segment_ids_arr = np.array(segment_ids, dtype=np.int64)
    return _prepared_run(features, segment_ids_arr), labels_arr


def test_pooled_conditioning_inflates_far_for_the_loose_label_versus_per_state() -> None:
    prepared, labels = _two_label_run_with_scale_mismatch()

    # cfg.seed defaults to 7 -- verified (scratch script) to reproduce the effect
    # cleanly; not cherry-picked (holds for every one of 15 independently tested seeds).
    per_state = run_sweep(prepared, labels, SweepConfig(conditioning="per-state"))
    pooled = run_sweep(prepared, labels, SweepConfig(conditioning="pooled"))

    far_per_state_a = per_state.far_table.set_index("label").loc["A", "realized_far"]
    far_pooled_a = pooled.far_table.set_index("label").loc["A", "realized_far"]
    n_scored_a = per_state.far_table.set_index("label").loc["A", "n_scored"]
    n_calibration_a = per_state.far_table.set_index("label").loc["A", "n_calibration"]

    # per-state: label A's threshold is calibrated on ONLY label A's own conformal-part
    # scores -- split conformal's guarantee holds regardless of label B's presence at
    # all, so this must stay within the same generous mean-level band used elsewhere.
    band = (
        0.05 + 3 * math.sqrt(0.05 * 0.95 / n_scored_a) + 1.0 / (n_calibration_a + 1)
    )
    assert far_per_state_a <= band

    # pooled: label A's threshold is calibrated on the MIXED (A+B) conformal scores;
    # since B's scores are systematically smaller (tight, 50x less spread), the shared
    # threshold sits below what A's own alpha-quantile would require -- realized FAR for
    # A is pulled strictly above both alpha and the per-state figure (directional
    # comparison, the actually-required assertion -- verified robust across 15 seeds,
    # unlike the exact inflation MAGNITUDE, which is seed-dependent).
    assert far_pooled_a > far_per_state_a
    assert far_pooled_a > 0.05


# ---------------------------------------------------------------------------
# Item 6: labels dtype validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_labels",
    [
        np.array([1.0, 2.0, 3.0]),
        np.array([True, False, True]),
        np.array([1 + 2j, 3 + 4j]),
        np.array([0, "a", 1.5], dtype=object),
    ],
    ids=["float", "bool", "complex", "mixed-object-not-all-str"],
)
def test_run_sweep_raises_on_invalid_labels_dtype(bad_labels: np.ndarray) -> None:
    prepared, _labels, _injected = _three_label_run_with_injected_outliers()

    with pytest.raises(ValueError, match="integer or string"):
        run_sweep(prepared, bad_labels, SweepConfig())


def test_run_sweep_accepts_int_and_str_labels() -> None:
    """Positive counterpart to the dtype-rejection tests: genuine int (detected cluster
    id) and str (GT state name) labels must both be ACCEPTED end-to-end -- the design's
    dual `reference_labels: detected | gt` mode (spec `docs/superpowers/specs/
    2026-07-09-step2-mode-conditioned-ad-design.md` §2)."""
    prepared, labels_int, _injected = _three_label_run_with_injected_outliers()

    result_int = run_sweep(prepared, labels_int, SweepConfig())
    assert isinstance(result_int, SweepResult)
    assert set(result_int.far_table["label"]) >= {0, 1, 2}

    str_labels = np.array(["standstill", "turbine", "pump"])[labels_int]
    result_str = run_sweep(prepared, str_labels, SweepConfig())
    assert set(result_str.far_table["label"]) >= {"standstill", "turbine", "pump"}


def test_run_sweep_raises_on_invalid_conditioning() -> None:
    prepared, labels, _injected = _three_label_run_with_injected_outliers()
    bad_cfg = SweepConfig(conditioning="bogus")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="conditioning"):
        run_sweep(prepared, labels, bad_cfg)


def test_run_sweep_raises_on_invalid_scorer_name() -> None:
    prepared, labels, _injected = _three_label_run_with_injected_outliers()
    bad_cfg = SweepConfig(scorer="bogus")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="scorer"):
        run_sweep(prepared, labels, bad_cfg)


# ---------------------------------------------------------------------------
# Item 1: nested-split crash with clear diagnostics
# ---------------------------------------------------------------------------


def test_nested_split_failure_on_too_few_calibration_segments() -> None:
    """A run with only 2 total segments will place one on each side of the top-level
    split, leaving only 1 segment on the calibration side. The nested 50/50 split then
    fails (cannot split 1 segment into 2 non-empty parts). Must raise a clear ValueError
    naming the nested split context and actual counts."""
    rng = np.random.default_rng(0)
    n_features = 5
    feats: list[np.ndarray] = []
    labels: list[int] = []
    segment_ids: list[int] = []
    # Only 2 segments total -- top-level split puts one on each side
    for seg in range(2):
        block = rng.normal(scale=0.5, size=(25, n_features))
        feats.append(block)
        labels.extend([0] * 25)
        segment_ids.extend([seg] * 25)

    features = np.vstack(feats)
    labels_arr = np.array(labels, dtype=np.int64)
    segment_ids_arr = np.array(segment_ids, dtype=np.int64)
    prepared = _prepared_run(features, segment_ids_arr)

    cfg = SweepConfig(calibration_frac=0.5, seed=0)
    with pytest.raises(ValueError, match="nested fit/conformal split failed"):
        run_sweep(prepared, labels_arr, cfg)


def test_nested_split_error_includes_actual_counts() -> None:
    """The nested split error message must include the actual number of calibration
    segments and windows, to help diagnose the problem."""
    rng = np.random.default_rng(0)
    n_features = 5
    feats: list[np.ndarray] = []
    labels: list[int] = []
    segment_ids: list[int] = []
    # Only 2 segments total -- top-level split puts one on each side
    for seg in range(2):
        block = rng.normal(scale=0.5, size=(25, n_features))
        feats.append(block)
        labels.extend([0] * 25)
        segment_ids.extend([seg] * 25)

    features = np.vstack(feats)
    labels_arr = np.array(labels, dtype=np.int64)
    segment_ids_arr = np.array(segment_ids, dtype=np.int64)
    prepared = _prepared_run(features, segment_ids_arr)

    cfg = SweepConfig(calibration_frac=0.5, seed=0)
    with pytest.raises(ValueError, match="1 segment.*25 windows"):
        run_sweep(prepared, labels_arr, cfg)


def test_top_level_split_error_and_nested_split_error_differ() -> None:
    """The top-level split failure (single segment) and nested split failure
    (2-segment top-level split leaving 1 on calibration side) must raise DIFFERENT error
    messages, so a caller can distinguish between the two problems."""
    rng = np.random.default_rng(0)
    n_features = 5

    # Test 1: Single segment (cannot split top-level, different error)
    feats_1 = [rng.normal(scale=0.5, size=(100, n_features))]
    labels_1 = [0] * 100
    segment_ids_1 = [0] * 100

    features_1 = np.array(feats_1[0])
    labels_arr_1 = np.array(labels_1, dtype=np.int64)
    segment_ids_arr_1 = np.array(segment_ids_1, dtype=np.int64)
    prepared_1 = _prepared_run(features_1, segment_ids_arr_1)

    cfg = SweepConfig()
    try:
        run_sweep(prepared_1, labels_arr_1, cfg)
        pytest.fail("Expected ValueError on single-segment run")
    except ValueError as e:
        top_level_msg = str(e)
        assert "nested" not in top_level_msg.lower(), (
            "top-level split error should NOT mention 'nested'"
        )

    # Test 2: Two segments total (top-level splits them 1-1, nested fails on calibration)
    feats_2: list[np.ndarray] = []
    labels_2: list[int] = []
    segment_ids_2: list[int] = []
    for seg in range(2):
        block = rng.normal(scale=0.5, size=(25, n_features))
        feats_2.append(block)
        labels_2.extend([0] * 25)
        segment_ids_2.extend([seg] * 25)

    features_2 = np.vstack(feats_2)
    labels_arr_2 = np.array(labels_2, dtype=np.int64)
    segment_ids_arr_2 = np.array(segment_ids_2, dtype=np.int64)
    prepared_2 = _prepared_run(features_2, segment_ids_arr_2)

    try:
        run_sweep(prepared_2, labels_arr_2, cfg)
        pytest.fail("Expected ValueError on nested-split-fails run")
    except ValueError as e:
        nested_msg = str(e)
        assert "nested" in nested_msg.lower(), (
            "nested split error should mention 'nested'"
        )

    assert top_level_msg != nested_msg, (
        "top-level and nested split errors should have different messages"
    )


# ---------------------------------------------------------------------------
# Item 2: eager alpha validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_run_sweep_raises_on_invalid_alpha(bad_alpha: float) -> None:
    prepared, labels, _injected = _three_label_run_with_injected_outliers()
    bad_cfg = SweepConfig(alpha=bad_alpha)

    with pytest.raises(ValueError, match="alpha.*0.*1"):
        run_sweep(prepared, labels, bad_cfg)


# ---------------------------------------------------------------------------
# Item 3: top_k validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_top_k", [0, -1, -5])
def test_run_sweep_raises_on_invalid_top_k(bad_top_k: int) -> None:
    prepared, labels, _injected = _three_label_run_with_injected_outliers()
    bad_cfg = SweepConfig(top_k=bad_top_k)

    with pytest.raises(ValueError, match="top_k.*>= 1"):
        run_sweep(prepared, labels, bad_cfg)


# ---------------------------------------------------------------------------
# Item 4: pooled-row convention (0 for counts, NaN for aggregates when all excluded)
# ---------------------------------------------------------------------------


def test_pooled_row_uses_zero_for_counts_when_all_labels_excluded() -> None:
    """When all labels are excluded (min_ref too high), the pooled aggregate row must
    use 0 (not NaN) for n_calibration/n_scored/n_alarms, and NaN only for
    realized_far/threshold."""
    rng = np.random.default_rng(0)
    n_features = 5
    feats: list[np.ndarray] = []
    labels: list[int] = []
    segment_ids: list[int] = []
    # Only 12 segments total -> 300 windows. With 25% in fit-part -> 75 windows fit.
    # Setting min_ref=100 excludes all (75 < 100).
    for seg in range(12):
        block = rng.normal(scale=0.5, size=(25, n_features))
        feats.append(block)
        labels.extend([0] * 25)
        segment_ids.extend([seg] * 25)

    features = np.vstack(feats)
    labels_arr = np.array(labels, dtype=np.int64)
    segment_ids_arr = np.array(segment_ids, dtype=np.int64)
    prepared = _prepared_run(features, segment_ids_arr)

    cfg = SweepConfig(min_ref=100, seed=0)

    result = run_sweep(prepared, labels_arr, cfg)

    pooled_row = result.far_table.set_index("label").loc["pooled"]
    # All labels excluded -> no contributing rows -> counts should be 0, not NaN
    assert pooled_row["n_calibration"] == 0.0, "n_calibration should be 0 when all excluded"
    assert pooled_row["n_scored"] == 0.0, "n_scored should be 0 when all excluded"
    assert pooled_row["n_alarms"] == 0.0, "n_alarms should be 0 when all excluded"
    # But realized FAR and threshold are still NaN
    assert math.isnan(pooled_row["realized_far"]), "realized_far should be NaN"
    assert math.isnan(pooled_row["threshold"]), "threshold should be NaN"


# ---------------------------------------------------------------------------
# Fixture 3 + item 7: excluded-label row (total count < min_ref, ANY seed)
# ---------------------------------------------------------------------------

_RARE_LABEL_TOTAL = 15
"""< the default `min_ref=20` -- guarantees label 1 is excluded regardless of how the
top-level or nested split happens to fall: `build_references`'s `min_ref` check counts
label 1's presence within `fit_windows` alone, a SUBSET of its (at most 15) total
windows, so that count can never reach 20 no matter the split outcome. Deliberately
seed-independent, unlike the empty-scoring-side and no-conformal-data fixtures below
(which need a SPECIFIC split outcome, not just an upper bound on a count)."""


def _run_with_one_rare_label() -> tuple[PreparedRun, np.ndarray]:
    rng = np.random.default_rng(0)
    n_features = 5
    feats: list[np.ndarray] = []
    labels: list[int] = []
    segment_ids: list[int] = []
    seg = 0
    for _ in range(20):  # label 0: dominant, 20 segments x 25 = 500 windows.
        block = rng.normal(scale=0.5, size=(25, n_features))
        feats.append(block)
        labels.extend([0] * 25)
        segment_ids.extend([seg] * 25)
        seg += 1
    block = np.array([50.0] + [0.0] * (n_features - 1)) + rng.normal(
        scale=0.5, size=(_RARE_LABEL_TOTAL, n_features)
    )
    feats.append(block)
    labels.extend([1] * _RARE_LABEL_TOTAL)
    segment_ids.extend([seg] * _RARE_LABEL_TOTAL)

    features = np.vstack(feats)
    labels_arr = np.array(labels, dtype=np.int64)
    segment_ids_arr = np.array(segment_ids, dtype=np.int64)
    return _prepared_run(features, segment_ids_arr), labels_arr


def test_excluded_label_row_has_nan_metrics_and_excluded_flag() -> None:
    prepared, labels = _run_with_one_rare_label()
    cfg = SweepConfig()  # seed=7, min_ref=20 defaults

    result = run_sweep(prepared, labels, cfg)

    row = result.far_table.set_index("label").loc[1]
    assert row["excluded"]
    for col in (
        "n_calibration",
        "n_scored",
        "n_alarms",
        "realized_far",
        "achievable_alpha_floor",
        "threshold",
    ):
        assert math.isnan(row[col]), f"{col} should be NaN for an excluded label, got {row[col]!r}"
    assert row["low_confidence"]
    assert row["nominal_alpha"] == cfg.alpha
    # An excluded label contributes no rows to scores/candidates -- no scorer was ever
    # fit for it, so there is nothing to score.
    assert (result.scores["label"] == 1).sum() == 0
    assert (result.candidates["label"] == 1).sum() == 0


# ---------------------------------------------------------------------------
# Fixture 4 + item 8: empty-scoring-side row (label entirely on the calibration side)
# ---------------------------------------------------------------------------


def _run_with_calibration_only_label() -> tuple[PreparedRun, np.ndarray]:
    """label 0 dominates (30 segments x 20 windows); label 1 (4 segments x 15 = 60
    windows, comfortably >= min_ref=20 if most land in calibration) is constructed so
    that, at `seed=10` specifically (verified empirically), ALL of its windows land on
    the top-level CALIBRATION side and further split into a real fit-part (>= min_ref)
    and conformal-part (>= 1), leaving zero for the scoring side."""
    rng = np.random.default_rng(0)
    n_features = 5
    feats: list[np.ndarray] = []
    labels: list[int] = []
    segment_ids: list[int] = []
    seg = 0
    for _ in range(30):
        block = rng.normal(scale=0.5, size=(20, n_features))
        feats.append(block)
        labels.extend([0] * 20)
        segment_ids.extend([seg] * 20)
        seg += 1
    for _ in range(4):
        block = np.array([50.0] + [0.0] * (n_features - 1)) + rng.normal(
            scale=0.5, size=(15, n_features)
        )
        feats.append(block)
        labels.extend([1] * 15)
        segment_ids.extend([seg] * 15)
        seg += 1

    features = np.vstack(feats)
    labels_arr = np.array(labels, dtype=np.int64)
    segment_ids_arr = np.array(segment_ids, dtype=np.int64)
    return _prepared_run(features, segment_ids_arr), labels_arr


def test_empty_scoring_side_row_has_zero_n_scored_and_nan_far() -> None:
    prepared, labels = _run_with_calibration_only_label()
    cfg = SweepConfig(seed=10)  # verified empirically -- see fixture docstring

    result = run_sweep(prepared, labels, cfg)

    row = result.far_table.set_index("label").loc[1]
    assert not row["excluded"]
    assert row["n_scored"] == 0
    assert row["n_alarms"] == 0
    assert math.isnan(row["realized_far"])
    # A real reference AND threshold WERE computed (unlike the excluded-label case).
    assert row["n_calibration"] > 0
    assert not math.isnan(row["threshold"])
    assert not row["low_confidence"]

    assert (result.scores["label"] == 1).sum() == 0
    assert (result.candidates["label"] == 1).sum() == 0


# ---------------------------------------------------------------------------
# Bonus: label passes min_ref on the fit-part but has zero conformal-part windows
# (defensive edge case beyond the dispatch's 8 numbered items -- see `sweep.py`'s
# `_no_conformal_data_row` docstring)
# ---------------------------------------------------------------------------


def _run_with_label_prone_to_zero_conformal_windows() -> tuple[PreparedRun, np.ndarray]:
    rng = np.random.default_rng(0)
    n_features = 5
    feats: list[np.ndarray] = []
    labels: list[int] = []
    segment_ids: list[int] = []
    seg = 0
    for _ in range(20):
        block = rng.normal(scale=0.5, size=(25, n_features))
        feats.append(block)
        labels.extend([0] * 25)
        segment_ids.extend([seg] * 25)
        seg += 1
    for _ in range(3):
        block = np.array([50.0] + [0.0] * (n_features - 1)) + rng.normal(
            scale=0.5, size=(15, n_features)
        )
        feats.append(block)
        labels.extend([1] * 15)
        segment_ids.extend([seg] * 15)
        seg += 1

    features = np.vstack(feats)
    labels_arr = np.array(labels, dtype=np.int64)
    segment_ids_arr = np.array(segment_ids, dtype=np.int64)
    return _prepared_run(features, segment_ids_arr), labels_arr


def test_label_with_reference_but_zero_conformal_windows_does_not_crash() -> None:
    """A label can clear `min_ref` on the fit-part yet end up with zero conformal-part
    windows if its calibration-side presence is concentrated in few segments that all
    land on the fit side of the nested split -- must produce a NaN-metric row (like
    exclusion), not raise `calibrate`'s "at least 1 element" `ValueError`."""
    prepared, labels = _run_with_label_prone_to_zero_conformal_windows()
    cfg = SweepConfig(seed=0)  # verified empirically -- see fixture docstring

    result = run_sweep(prepared, labels, cfg)

    row = result.far_table.set_index("label").loc[1]
    assert not row["excluded"]  # min_ref WAS cleared -- a real reference exists
    assert math.isnan(row["n_calibration"])
    assert math.isnan(row["realized_far"])
    assert row["low_confidence"]
    assert (result.scores["label"] == 1).sum() == 0


# ---------------------------------------------------------------------------
# Schema / contract sanity (columns, and the "pooled" aggregate row's conditioning)
# ---------------------------------------------------------------------------


def test_far_table_and_scores_and_candidates_have_the_contracted_columns() -> None:
    prepared, labels, _injected = _three_label_run_with_injected_outliers()

    result = run_sweep(prepared, labels, SweepConfig())

    assert list(result.far_table.columns) == [
        "label",
        "n_calibration",
        "n_scored",
        "n_alarms",
        "realized_far",
        "nominal_alpha",
        "achievable_alpha_floor",
        "low_confidence",
        "threshold",
        "excluded",
    ]
    assert list(result.scores.columns) == ["window", "label", "score", "p_value", "alarm"]
    assert list(result.candidates.columns) == ["window", "label", "score", "p_value", "rank"]


def test_aggregate_pooled_row_present_only_for_per_state_conditioning() -> None:
    prepared, labels, _injected = _three_label_run_with_injected_outliers()

    per_state = run_sweep(prepared, labels, SweepConfig(conditioning="per-state"))
    pooled = run_sweep(prepared, labels, SweepConfig(conditioning="pooled"))

    assert "pooled" in set(per_state.far_table["label"])
    assert set(per_state.far_table["label"]) == {0, 1, 2, "pooled"}
    assert set(pooled.far_table["label"]) == {0, 1, 2}


def test_sweep_config_and_result_are_frozen() -> None:
    cfg = SweepConfig()
    with pytest.raises(AttributeError):
        cfg.alpha = 0.1  # type: ignore[misc]

    prepared, labels, _injected = _three_label_run_with_injected_outliers()
    result = run_sweep(prepared, labels, cfg)
    with pytest.raises(AttributeError):
        result.far_table = pd.DataFrame()  # type: ignore[misc]
