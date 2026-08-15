"""Unit tests for `rowii.anomaly.fusion`: score-level fusion of the `fusion` variant's
audio and vibration branches via p-value combination. Synthetic-only -- no real data.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from rowii.anomaly.conformal import calibrate, loo_p_values, p_values
from rowii.anomaly.fusion import fisher_statistic, split_branch_columns, tippett_statistic

# ---------------------------------------------------------------------------
# split_branch_columns
# ---------------------------------------------------------------------------


class TestSplitBranchColumns:
    def test_realistic_names_split_disjoint_and_exhaustive(self) -> None:
        names = [
            "RAWGeneratorMic__0::ch0_log_rms",
            "RAWGeneratorMic__0::ch1_kurtosis",
            "RAWTurbineMic__1::ch0_log_rms",
            "RAWGeneratorVib__2::ch1_kurtosis",
            "RAWGeneratorVib__2::chX_rms",
            "RAWTurbineVib__3::chY_rms",
        ]
        audio_idx, vib_idx = split_branch_columns(names)

        assert audio_idx.dtype == np.int64
        assert vib_idx.dtype == np.int64
        assert audio_idx.tolist() == [0, 1, 2]
        assert vib_idx.tolist() == [3, 4, 5]
        # Disjoint + exhaustive: every original index appears in exactly one branch.
        assert set(audio_idx.tolist()) & set(vib_idx.tolist()) == set()
        assert set(audio_idx.tolist()) | set(vib_idx.tolist()) == set(range(len(names)))

    def test_raises_on_empty_audio_branch(self) -> None:
        names = ["RAWGeneratorVib__2::ch0", "RAWTurbineVib__3::ch1"]
        with pytest.raises(ValueError, match="audio branch is empty"):
            split_branch_columns(names)

    def test_raises_on_empty_vibration_branch(self) -> None:
        names = ["RAWGeneratorMic__0::ch0", "RAWTurbineMic__1::ch1"]
        with pytest.raises(ValueError, match="vibration branch is empty"):
            split_branch_columns(names)

    def test_raises_on_missing_separator(self) -> None:
        names = ["RAWGeneratorMic__0::ch0", "RAWGeneratorVib__2::ch1", "no_separator_here"]
        with pytest.raises(ValueError, match="no_separator_here"):
            split_branch_columns(names)

    def test_raises_on_ambiguous_stream_prefix(self) -> None:
        names = [
            "RAWGeneratorMic__0::ch0", "RAWGeneratorVib__2::ch1", "RAWSomethingElse__9::ch2",
        ]
        with pytest.raises(ValueError, match="RAWSomethingElse__9"):
            split_branch_columns(names)

    def test_raises_on_prefix_matching_both_mic_and_vib(self) -> None:
        # A stream prefix containing BOTH markers cannot be assigned to exactly one
        # branch -- must raise (naming the offender), never silently pick a side.
        names = [
            "RAWGeneratorMic__0::ch0", "RAWGeneratorVib__2::ch1", "RAWMicVibCombo__9::ch2",
        ]
        with pytest.raises(ValueError, match="RAWMicVibCombo__9") as exc_info:
            split_branch_columns(names)
        assert "both" in str(exc_info.value)

    def test_raises_names_every_offender_not_just_the_first(self) -> None:
        names = ["missing_sep_1", "RAWGeneratorMic__0::ch0", "missing_sep_2"]
        with pytest.raises(ValueError, match="missing_sep_1") as exc_info:
            split_branch_columns(names)
        assert "missing_sep_2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# fisher_statistic
# ---------------------------------------------------------------------------


class TestFisherStatistic:
    def test_known_value(self) -> None:
        # -2 * (ln 0.05 + ln 0.05) ~= 11.9829 (orchestrator resolution 2's own
        # worked example).
        stat = fisher_statistic(np.array([0.05]), np.array([0.05]))
        assert stat[0] == pytest.approx(11.9829, abs=1e-3)

    def test_smaller_p_values_give_larger_statistic(self) -> None:
        p_a = np.array([0.5, 0.5, 0.01])
        p_v = np.array([0.5, 0.01, 0.01])
        stat = fisher_statistic(p_a, p_v)
        assert stat[0] < stat[1] < stat[2]

    def test_dtype_is_float64(self) -> None:
        stat = fisher_statistic(np.array([0.2]), np.array([0.3]))
        assert stat.dtype == np.float64

    def test_raises_on_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            fisher_statistic(np.array([0.1, 0.2]), np.array([0.1]))


# ---------------------------------------------------------------------------
# tippett_statistic
# ---------------------------------------------------------------------------


class TestTippettStatistic:
    def test_known_value(self) -> None:
        stat = tippett_statistic(np.array([0.3]), np.array([0.05]))
        assert stat[0] == pytest.approx(0.95)

    def test_smaller_min_p_gives_larger_statistic(self) -> None:
        p_a = np.array([0.5, 0.5, 0.01])
        p_v = np.array([0.5, 0.2, 0.5])
        stat = tippett_statistic(p_a, p_v)
        assert stat[0] < stat[1] < stat[2]

    def test_order_equivalent_to_textbook_min_p_rule(self) -> None:
        """Orchestrator resolution 1: ranking by `tippett_statistic` DESCENDING
        (higher = more anomalous, this module's convention) must reproduce EXACTLY
        the same window order as ranking by the textbook Tippett rule `min(p_a, p_v)`
        ASCENDING (smaller = more anomalous, the classical convention)."""
        rng = np.random.default_rng(0)
        p_a = rng.uniform(1e-3, 1.0, 200)
        p_v = rng.uniform(1e-3, 1.0, 200)
        stat = tippett_statistic(p_a, p_v)
        order_by_statistic = np.argsort(-stat)
        order_by_textbook_rule = np.argsort(np.minimum(p_a, p_v))
        np.testing.assert_array_equal(order_by_statistic, order_by_textbook_rule)

    def test_raises_on_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            tippett_statistic(np.array([0.1, 0.2]), np.array([0.1]))


# ---------------------------------------------------------------------------
# FAR-validity test across dependence regimes (centerpiece, orchestrator
# resolution 5; sharpened + multi-regime per the 2026-07-15 review fix)
# ---------------------------------------------------------------------------

_VALIDITY_REGIMES = ("independent", "corr", "anti", "identical")
"""Branch-dependence regimes the validity test sweeps: independent draws, shared
latent + per-branch noise (rho ~ 0.78), near-deterministic anti-correlation, and
exactly identical branches -- the point being that split-conformal re-calibration of
the combined statistic must hold the FAR at every one of these, since the guarantee
never assumed anything about how the two branches relate to each other."""


def _draw_branch_pair(
    rng: np.random.Generator, n: int, regime: str
) -> tuple[np.ndarray, np.ndarray]:
    """`(audio, vib)` raw-score arrays of length *n* under one `_VALIDITY_REGIMES`
    dependence regime (all rows exchangeable within each branch -- a pure null)."""
    if regime == "independent":
        return rng.normal(size=n), rng.normal(size=n)
    if regime == "corr":
        latent = rng.normal(0.0, 1.0, n)
        return latent + rng.normal(0.0, 0.5, n), latent + rng.normal(0.0, 0.5, n)
    if regime == "anti":
        a = rng.normal(size=n)
        return a, -a + rng.normal(0.0, 0.01, n)
    if regime == "identical":
        a = rng.normal(size=n)
        return a, a.copy()
    raise ValueError(f"unknown regime {regime!r}")


def _combined_far_over_reps(
    statistic_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    regime: str,
    n_calibration: int,
    calibration_p_fn: Callable[[np.ndarray], np.ndarray],
    n_scoring: int = 1000,
    n_reps: int = 500,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """`(mean realized FAR, standard error of that mean)` of the full score-fusion
    path for one combination rule over *n_reps* seeded repetitions: per rep, draw one
    regime's branch pair, compute CALIBRATION-side branch p-values via
    *calibration_p_fn* and SCORING-side branch p-values via `p_values(scoring,
    calibration)`, combine both sides with *statistic_fn* (`fisher_statistic` or
    `tippett_statistic`), `calibrate` the calibration-side combined statistic, and
    count scoring-side alarms.

    *calibration_p_fn* is the calibration-side construction under test: the fixed
    code passes `loo_p_values` (leave-one-out, each point evaluated against the other
    n-1 -- the same footing a scoring point gets); the review-time mutant check
    passed the DEFECTIVE self-referential `lambda s: p_values(s, s)` (each point in
    its own reference) through the Fisher path to demonstrate the Fisher test catches
    exactly that defect (scratch run, see the task-5 fix report: the mutant fails 6
    of the 8 (regime, n) cases, worst mean FAR ~0.10 at alpha=0.05 for
    anti-correlated branches at n=39). For `tippett_statistic` the choice of
    *calibration_p_fn* is decision-neutral -- a min-rule statistic is a strictly
    monotone transform of the calibration-side min-rank either way, so LOO and
    self-referential produce bit-identical alarms (verified, review round 2) -- which
    is exactly why Tippett's residual excess under positive correlation is intrinsic
    rather than fixable by the LOO switch (see the class docstring below).
    """
    fars = np.empty(n_reps)
    for rep in range(n_reps):
        rng = np.random.default_rng(rep)
        audio, vib = _draw_branch_pair(rng, n_calibration + n_scoring, regime)
        audio_conf, audio_score = audio[:n_calibration], audio[n_calibration:]
        vib_conf, vib_score = vib[:n_calibration], vib[n_calibration:]

        p_a_conf = calibration_p_fn(audio_conf)
        p_v_conf = calibration_p_fn(vib_conf)
        p_a_score = p_values(audio_score, audio_conf)
        p_v_score = p_values(vib_score, vib_conf)

        threshold = calibrate(statistic_fn(p_a_conf, p_v_conf), alpha)
        combined_score = statistic_fn(p_a_score, p_v_score)
        fars[rep] = float(np.mean(combined_score > threshold.threshold))
    return float(fars.mean()), float(fars.std(ddof=1) / np.sqrt(n_reps))


class TestFarValidityAcrossDependenceRegimes:
    """Centerpiece: the Fisher-combined, LOO-calibrated, RE-CALIBRATED statistic
    holds the mean realized FAR at (or conservatively below) alpha under EVERY
    branch-dependence regime -- the classical Fisher chi2(4) null assumes independent
    p-values and is invalid here, but the orchestration never uses it: the combined
    statistic is thresholded by `rowii.anomaly.conformal.calibrate` on held-out
    conformal-side values instead (`rowii.anomaly.fusion` module docstring's
    "Statistical note").

    The calibration side MUST use `loo_p_values` (leave-one-out), not
    `p_values(conf, conf)`: the self-referential form puts each calibration point in
    its own reference (its p-value can never go below 2/(n+1), while a scoring point
    reaches 1/(n+1)), so the combined statistic is NOT one fixed transform applied to
    both sides and its calibration/scoring exchangeability breaks -- measured
    anti-conservative up to mean FAR ~0.10 at alpha=0.05 (n=39, anti-correlated;
    review finding 2026-07-15, reproduced in this test's own mutant check, task-5 fix
    report). The LOO form leaves a one-unit granularity residual (LOO min p = 1/n vs
    scoring 1/(n+1)) whose direction is conservative -- hence the one-sided sharp
    bound for Fisher.

    TIPPETT gets a narrower pin, not the same clean claim (review round 2): a
    min-rule statistic is a strictly monotone transform of the calibration-side
    min-rank under ANY shared-reference p construction (LOO vs self-referential is
    bit-identical in alarms), so its residual excess under POSITIVELY correlated
    branches is intrinsic to the calibration set doubling as the p-value reference --
    measured +0.007 absolute at alpha=0.05, n=39, decaying roughly like 1/n (0.0518
    at n=159, 0.0512 at n=319); independent/anti-correlated/identical stay within
    alpha + 3*SE. The Tippett test below asserts the strict bound where validity
    holds and PINS the documented excess where it does not (`rowii.anomaly.fusion`
    module docstring's Statistical note carries the full story).
    """

    _TIPPETT_CORR_EXCESS_BAND = {39: 0.010, 159: 0.005}
    """Absolute FAR excess allowed for Tippett under the positive-correlation regime
    -- these bands PIN the known, documented intrinsic excess (measured +0.0074 at
    n=39, +0.0017 at n=159 with this harness's exact seeds): a regression ABOVE them
    is a NEW defect; a drop back to within alpha + 3*SE means someone fixed Tippett
    properly (a dedicated p-reference split) and this carve-out should be deleted."""

    @pytest.mark.parametrize("n_calibration", [39, 159])
    @pytest.mark.parametrize("regime", list(_VALIDITY_REGIMES))
    def test_fisher_mean_realized_far_at_most_alpha_plus_three_se(
        self, regime: str, n_calibration: int
    ) -> None:
        alpha = 0.05
        mean_far, se = _combined_far_over_reps(
            fisher_statistic, regime, n_calibration, loo_p_values, alpha=alpha
        )
        # Rep-count sanity: 500 reps must push the Monte-Carlo SE of the mean to
        # ~0.002 or below (per-rep FAR sd is ~0.035 at n=39, ~0.019 at n=159, both
        # Beta-threshold-noise dominated), else the bound below would be too loose
        # to call sharp.
        assert se <= 2.0e-3, f"{regime}/n={n_calibration}: SE {se:.5f} too large"
        assert mean_far <= alpha + 3.0 * se, (
            f"{regime}/n={n_calibration}: mean realized FAR {mean_far:.4f} exceeds "
            f"alpha + 3*SE = {alpha + 3.0 * se:.4f} -- FAR validity broken"
        )

    @pytest.mark.parametrize("n_calibration", [39, 159])
    @pytest.mark.parametrize("regime", list(_VALIDITY_REGIMES))
    def test_tippett_mean_realized_far_within_documented_bounds(
        self, regime: str, n_calibration: int
    ) -> None:
        alpha = 0.05
        mean_far, se = _combined_far_over_reps(
            tippett_statistic, regime, n_calibration, loo_p_values, alpha=alpha
        )
        assert se <= 2.0e-3, f"{regime}/n={n_calibration}: SE {se:.5f} too large"
        if regime == "corr":
            # NOT a validity claim: under shared-latent positive correlation Tippett
            # is mildly anti-conservative BY CONSTRUCTION (class docstring; the
            # strict alpha + 3*SE bound genuinely fails here -- 0.0574 vs 0.0553 at
            # n=39 with these exact seeds). These bands pin the known, documented
            # intrinsic excess: a failure above them is a NEW defect; a pass back
            # under alpha + 3*SE means it was properly fixed (dedicated p-reference
            # split) and this branch should be collapsed into the one below.
            bound = alpha + self._TIPPETT_CORR_EXCESS_BAND[n_calibration]
            reason = "documented intrinsic Tippett excess regressed"
        else:
            bound = alpha + 3.0 * se
            reason = "FAR validity broken"
        assert mean_far <= bound, (
            f"tippett {regime}/n={n_calibration}: mean realized FAR {mean_far:.4f} "
            f"exceeds {bound:.4f} -- {reason}"
        )

    def test_regime_constructions_have_the_claimed_dependence(self) -> None:
        """Premise check: the regimes really produce the branch dependence their
        names claim -- otherwise a passing validity sweep would not distinguish
        "guarantee holds despite dependence" from "the branches were accidentally
        independent everywhere"."""
        rng = np.random.default_rng(0)
        a, v = _draw_branch_pair(rng, 5000, "corr")
        assert float(np.corrcoef(a, v)[0, 1]) > 0.5
        a, v = _draw_branch_pair(np.random.default_rng(0), 5000, "anti")
        assert float(np.corrcoef(a, v)[0, 1]) < -0.9
        a, v = _draw_branch_pair(np.random.default_rng(0), 5000, "identical")
        np.testing.assert_array_equal(a, v)
