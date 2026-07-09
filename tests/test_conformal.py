"""Tests for `rowii.anomaly.conformal`: split-conformal calibration (`calibrate`) and
conformal p-values (`p_values`). Synthetic-only, per the Step-2 plan
(`docs/superpowers/plans/2026-07-09-step2-first-package.md` Task S4) -- no real data.

The Monte Carlo validity tests below are grounded in the EXACT finite-sample theory of
split conformal (Angelopoulos & Bates 2022, "A Gentle Introduction to Conformal
Prediction and Distribution-Free Uncertainty Quantification", arXiv:2107.07511,
Appendix "Concentration Properties of the Empirical Coverage" -- cross-derived
independently from order-statistics theory before consulting the paper; both agree).
For `n` calibration scores and threshold rank `idx = ceil((n + 1) * (1 - alpha))`, the
realised false-alarm rate of a SINGLE calibration draw is a
`Beta(n + 1 - idx, idx)`-distributed random variable, not merely "close to alpha", and
the empirical FAR over `n_test` fresh test draws against that one calibration set is
`Beta-Binomial(n_test, n + 1 - idx, idx)`. Only the MEAN over many independent
calibration draws concentrates tightly around `(n + 1 - idx) / (n + 1)` (itself always
within `1 / (n + 1)` of `alpha` -- the guarantee this module implements); a SINGLE
repetition's FAR can deviate from `alpha` far more than a naive
`sqrt(alpha * (1 - alpha) / n_test)` binomial estimate would suggest, especially at
small `n` (`n=19, alpha=0.05` forces `idx = n`, i.e. the threshold IS the sample
maximum, the highest-variance order statistic there is).

This was verified empirically before writing the assertions below (a 300-seed sweep,
scratch script, not committed): a first-draft per-repetition bound of the form
`alpha + 3 * sqrt(alpha * (1 - alpha) / 1000) + 1 / (n + 1)` is regularly violated by a
CORRECT implementation at n=19 (observed per-repetition FAR up to ~0.4 against that
draft bound's ~0.12) -- so the bounds below use the exact Beta / Beta-Binomial
distributions instead of that approximation. See `.superpowers/sdd/task-s4-report.md`
for the full derivation and the counter-evidence.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import betabinom

from rowii.anomaly.conformal import ConformalThreshold, calibrate, p_values

_N_REPS = 200
_N_TEST = 1000
_ALPHA = 0.05
_MEAN_BAND_Z = 4.0
"""Sigma multiplier for the two-sided Monte Carlo bands below. 3.0 (matching the
dispatch's own "3 sigma" convention) was tried first and empirically fails ~1/300
seeds even for a provably correct implementation (a true 3-sigma band is expected to
miss ~0.3% of the time); 4.0 had zero failures over an independent 300-seed sweep at
every tested n, so is used throughout for a non-flaky yet still statistically
meaningful (not vacuous) two-sided check."""
_PER_REP_TAIL_PROB = 1e-6 / 3
"""Per-(n, repetition) tail probability for the Beta-Binomial per-repetition bound,
Bonferroni-corrected over the 3 tested `n` values so the whole validity test's
false-failure probability from this check alone stays far below 1e-3."""


# ---------------------------------------------------------------------------
# Exact-theory helpers for the Monte Carlo tests (independent of the module under
# test: none of these import `rowii.anomaly.conformal`'s private `_threshold_index`).
# ---------------------------------------------------------------------------


def _expected_idx(n: int, level: float) -> int:
    """Independent recomputation of the calibration order-statistic rank `calibrate`
    and `p_values` are built on: `ceil((n + 1) * (1 - level))`. Exact for every
    `n`/`level` combination used below (verified none sit near an integer boundary
    from the wrong side, unlike the `alpha=1/3` case exercised in
    `test_calibrate_threshold_index_robust_to_floating_point_representation_error`)."""
    return math.ceil((n + 1) * (1 - level))


def _exact_exceed_prob(n: int, level: float) -> float:
    """Exact `P(a fresh exchangeable score exceeds the level-calibrated threshold)`,
    marginalised over BOTH the calibration draw and the test draw:
    `(n + 1 - idx) / (n + 1)`, always within `1 / (n + 1)` of `level` (module
    docstring). Thresholding at `level` and testing `p_values(...) <= level` are the
    same event, so this also doubles as the exact expected value of
    `P(p_values(...) <= level)`, used by the p-value uniformity test below."""
    idx = _expected_idx(n, level)
    return (n + 1 - idx) / (n + 1)


def _mean_se(n: int, level: float, n_test: int, n_reps: int) -> float:
    """Exact standard error of the MEAN exceed-rate over `n_reps` independent
    (calibration, `n_test` test scores) repetitions (Angelopoulos & Bates
    arXiv:2107.07511 Appendix "Concentration Properties of the Empirical Coverage",
    translated from their coverage notation -- their `l` is `n + 1 - idx` here; the
    product `idx * (n + 1 - idx)` is symmetric under that relabelling, so the
    substitution is unambiguous)."""
    idx = _expected_idx(n, level)
    numerator = idx * (n + 1 - idx) * (n + n_test + 1)
    denominator = n_test * n_reps * (n + 1) ** 2 * (n + 2)
    return math.sqrt(numerator / denominator)


def _per_rep_upper_bound(n: int, level: float, n_test: int, tail_prob: float) -> float:
    """Exact upper bound on a SINGLE repetition's empirical exceed-rate: the
    `1 - tail_prob` quantile of `Beta-Binomial(n_test, n + 1 - idx, idx)` (the exact
    distribution of the exceed-count out of `n_test` test draws for one fixed
    calibration draw), divided by `n_test`."""
    idx = _expected_idx(n, level)
    return float(betabinom(n_test, n + 1 - idx, idx).ppf(1.0 - tail_prob) / n_test)


# ---------------------------------------------------------------------------
# Validity simulation (item 1): FAR within exact-theory bands + p-value uniformity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [19, 50, 500])
def test_conformal_validity_empirical_far_matches_exact_theory(n: int) -> None:
    """calibrate on `n` normal (Exponential(1) -- split conformal is
    distribution-free, so an arbitrary continuous distribution suffices) scores,
    score 1000 fresh draws from the same distribution, repeat 200 times. Checks:
      1. `calibrate`'s advertised guarantee, `|exact FAR - alpha| <= 1/(n+1)`, holds
         for the exact expected FAR (a deterministic fact about `idx`, sanity-checking
         the Monte Carlo oracle itself before using it).
      2. The MEAN empirical FAR over the 200 repetitions falls within the
         `_MEAN_BAND_Z`-sigma band of that exact expected FAR (two-sided validity).
      3. EVERY one of the 200 repetitions' empirical FAR stays under the
         Beta-Binomial-exact per-repetition bound -- wide by design (module
         docstring), but still catches a genuinely broken calibration: a
         wrong-direction or otherwise miscalibrated threshold blows through it by an
         order of magnitude (verified via mutation testing, see task report).
    """
    exact_mean_far = _exact_exceed_prob(n, _ALPHA)
    assert abs(exact_mean_far - _ALPHA) <= 1.0 / (n + 1) + 1e-12

    se = _mean_se(n, _ALPHA, _N_TEST, _N_REPS)
    per_rep_upper = _per_rep_upper_bound(n, _ALPHA, _N_TEST, _PER_REP_TAIL_PROB)

    rng = np.random.default_rng(0)
    fars = np.empty(_N_REPS)
    for rep in range(_N_REPS):
        calibration_scores = rng.exponential(size=n)
        test_scores = rng.exponential(size=_N_TEST)
        threshold = calibrate(calibration_scores, _ALPHA).threshold
        fars[rep] = np.mean(test_scores > threshold)

    band_lo, band_hi = exact_mean_far - _MEAN_BAND_Z * se, exact_mean_far + _MEAN_BAND_Z * se
    mean_far = fars.mean()
    assert band_lo <= mean_far <= band_hi, (
        f"n={n}: mean FAR {mean_far:.5f} outside [{band_lo:.5f}, {band_hi:.5f}]"
    )
    assert fars.max() <= per_rep_upper, (
        f"n={n}: worst single-repetition FAR {fars.max():.5f} exceeds the "
        f"Beta-Binomial bound {per_rep_upper:.5f}"
    )


@pytest.mark.parametrize("n", [19, 50, 500])
def test_p_values_are_super_uniform(n: int) -> None:
    """`p_values`' super-uniformity guarantee, `P(p <= t) <= t` for continuous scores
    (module docstring): pool p-values from 200 independent (calibration, 1000 test
    scores) repetitions and check the empirical `P(p <= t)` stays within the
    `_MEAN_BAND_Z`-sigma statistical margin of `t`, for `t` in
    `{0.01, 0.05, 0.1, 0.5}` (a KS-style check at several points of the p-value CDF at
    once). Thresholding at level `t` and testing `p <= t` are the same event, so this
    reuses `_exact_exceed_prob`/`_mean_se` exactly as the FAR validity test above,
    varying `t` in place of a single fixed `alpha`.
    """
    rng = np.random.default_rng(1)
    pooled_p_values = []
    for _ in range(_N_REPS):
        calibration_scores = rng.exponential(size=n)
        test_scores = rng.exponential(size=_N_TEST)
        pooled_p_values.append(p_values(test_scores, calibration_scores))
    pooled = np.concatenate(pooled_p_values)

    for t in [0.01, 0.05, 0.1, 0.5]:
        se = _mean_se(n, t, _N_TEST, _N_REPS)
        bound = t + _MEAN_BAND_Z * se
        empirical = float(np.mean(pooled <= t))
        assert empirical <= bound, (
            f"n={n} t={t}: empirical P(p<=t)={empirical:.5f} exceeds super-uniform "
            f"bound {bound:.5f}"
        )


# ---------------------------------------------------------------------------
# Exact boundary: n=19 vs n=18 at alpha=0.05, achievable_alpha_floor (item 2)
# ---------------------------------------------------------------------------


def test_calibrate_n19_alpha05_is_exactly_at_the_achievable_floor() -> None:
    # idx = ceil(20 * 0.95) = ceil(19.0) = 19 == n -> threshold exists: the sample
    # maximum (the top of 19 calibration scores).
    calibration_scores = np.arange(1.0, 20.0)  # 1..19

    result = calibrate(calibration_scores, 0.05)

    assert result == ConformalThreshold(
        threshold=19.0,
        alpha=0.05,
        n_calibration=19,
        achievable_alpha_floor=pytest.approx(1.0 / 20.0),
        low_confidence=False,
    )


def test_calibrate_n18_alpha05_falls_below_the_achievable_floor() -> None:
    # idx = ceil(19 * 0.95) = ceil(18.05) = 19 > n=18 -> no order statistic exists.
    calibration_scores = np.arange(1.0, 19.0)  # 1..18

    result = calibrate(calibration_scores, 0.05)

    assert math.isinf(result.threshold) and result.threshold > 0
    assert result.low_confidence is True
    assert result.n_calibration == 18
    assert result.achievable_alpha_floor == pytest.approx(1.0 / 19.0)


@pytest.mark.parametrize("n", [1, 4, 19, 99, 1000])
def test_achievable_alpha_floor_is_exactly_one_over_n_plus_one(n: int) -> None:
    calibration_scores = np.arange(1.0, n + 1.0)

    result = calibrate(calibration_scores, 0.5)

    assert result.achievable_alpha_floor == 1.0 / (n + 1)


# ---------------------------------------------------------------------------
# Hand-computed 5-score case (item 3)
# ---------------------------------------------------------------------------


def test_calibrate_hand_computed_five_score_case() -> None:
    # idx = ceil(6 * 0.5) = ceil(3.0) = 3 <= n=5 -> threshold = 3rd smallest = 3.0.
    calibration_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    result = calibrate(calibration_scores, 0.5)

    assert result.threshold == 3.0
    assert result.low_confidence is False
    assert result.n_calibration == 5


def test_p_values_hand_computed_matches_dispatch_example() -> None:
    calibration_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scores = np.array([2.5, 6.0])

    result = p_values(scores, calibration_scores)

    # score=2.5: #{cal >= 2.5} = |{3,4,5}| = 3 -> (1+3)/6.
    # score=6.0: #{cal >= 6.0} = 0           -> (1+0)/6.
    np.testing.assert_allclose(result, [4.0 / 6.0, 1.0 / 6.0])


# ---------------------------------------------------------------------------
# Tie handling: brute-force cross-check on random small cases (item 4)
# ---------------------------------------------------------------------------


def _brute_force_p_values(scores: np.ndarray, calibration_scores: np.ndarray) -> np.ndarray:
    """Direct, unvectorized reimplementation of `p_values`'s own definition (a plain
    Python loop, no `searchsorted`) -- an independent oracle for the fuzz test below.
    `>=` (not `>`) matches the "ties count as conservative" contract exactly."""
    n = calibration_scores.shape[0]
    return np.array(
        [(1.0 + sum(1 for c in calibration_scores if c >= s)) / (n + 1) for s in scores]
    )


def test_p_values_ties_match_brute_force_on_random_small_cases() -> None:
    # A small discrete value set (0..5) forces frequent exact ties between
    # `calibration_scores` and `scores`, unlike the continuous distributions used
    # elsewhere in this file -- exactly what exercises the ">=" (not ">") tie rule.
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(1, 11))
        m = int(rng.integers(1, 11))
        calibration_scores = rng.integers(0, 6, size=n).astype(np.float64)
        scores = rng.integers(0, 6, size=m).astype(np.float64)

        result = p_values(scores, calibration_scores)
        expected = _brute_force_p_values(scores, calibration_scores)

        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Numerical robustness beyond the literal dispatch (see task report for rationale)
# ---------------------------------------------------------------------------


def test_calibrate_threshold_index_robust_to_floating_point_representation_error() -> None:
    # (n+1)*(1-alpha) = 30 * (2/3) = 20 EXACTLY in real arithmetic, but `1 - 1/3` is
    # not exactly representable in binary floating point, so the raw double product
    # comes out as 20.000000000000004 -- a naive `math.ceil` would round that UP to
    # 21, silently off by one. idx must come out as 20 (threshold = 20th smallest).
    calibration_scores = np.arange(1.0, 30.0)  # 1..29

    result = calibrate(calibration_scores, 1.0 / 3.0)

    assert result.threshold == 20.0
    assert result.low_confidence is False


def test_calibrate_extreme_alpha_near_one_does_not_crash() -> None:
    # alpha this close to 1 is not a realistic false-alarm target, but `calibrate`
    # must not silently misbehave: an over-aggressive floating-point tolerance
    # adjustment could otherwise push the computed index to <= 0, and Python's
    # negative-indexing semantics would then silently return the WRONG (maximum, via
    # `sorted_scores[-1]`) element instead of the correct one.
    calibration_scores = np.array([1.0, 2.0, 3.0])

    result = calibrate(calibration_scores, 1.0 - 1e-12)

    assert result.threshold == 1.0  # idx clamped to 1 -> smallest calibration score
    assert result.low_confidence is False


# ---------------------------------------------------------------------------
# Error paths (item 5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1, math.nan])
def test_calibrate_raises_on_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        calibrate(np.array([1.0, 2.0, 3.0]), alpha)


def test_calibrate_raises_on_empty_calibration_scores() -> None:
    with pytest.raises(ValueError, match="at least 1 element"):
        calibrate(np.empty(0), 0.05)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_calibrate_raises_on_non_finite_calibration_score(bad_value: float) -> None:
    calibration_scores = np.array([1.0, 2.0, bad_value, 4.0])

    with pytest.raises(ValueError, match="finite"):
        calibrate(calibration_scores, 0.05)


def test_p_values_raises_on_empty_calibration_scores() -> None:
    with pytest.raises(ValueError, match="at least 1 element"):
        p_values(np.array([1.0]), np.empty(0))


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_p_values_raises_on_non_finite_calibration_score(bad_value: float) -> None:
    calibration_scores = np.array([1.0, bad_value, 3.0])

    with pytest.raises(ValueError, match="finite"):
        p_values(np.array([2.0]), calibration_scores)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_p_values_raises_on_non_finite_query_score(bad_value: float) -> None:
    scores = np.array([1.0, bad_value])

    with pytest.raises(ValueError, match="finite"):
        p_values(scores, np.array([1.0, 2.0, 3.0]))
