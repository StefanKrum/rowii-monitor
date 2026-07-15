"""Unit tests for `rowii.anomaly.fusion`: score-level fusion of the `fusion` variant's
audio and vibration branches via p-value combination (design spec `docs/superpowers/
specs/2026-07-15-step2-package3-baselines-design.md` D5, plan `docs/superpowers/plans/
2026-07-15-step2-package3-baselines.md` Task 5). Synthetic-only -- no real data.
"""
from __future__ import annotations

import numpy as np
import pytest

from rowii.anomaly.conformal import calibrate, p_values
from rowii.anomaly.fusion import fisher_statistic, split_branch_columns, tippett_statistic
from rowii.anomaly.scarcity import beta_band

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
# Guarantee-restored test (centerpiece, orchestrator resolution 5)
# ---------------------------------------------------------------------------


class TestGuaranteeRestoredUnderCorrelatedBranches:
    """Fisher-combined, RE-CALIBRATED statistic controls the realized FAR even though
    the two branches are CORRELATED (shared latent cause + independent per-branch
    noise) -- the classical Fisher chi2(4) null assumes independent p-values and would
    NOT be valid here (verified separately, scratch script, not committed: the naive
    chi2(4) threshold's mean FAR drifts to ~0.08-0.09 at this correlation, nearly
    double the nominal alpha=0.05). Neither `fisher_statistic` nor the orchestration
    ever compares against that classical null, though -- the combined statistic is
    just a deterministic score transform, RE-CALIBRATED via `rowii.anomaly.conformal.
    calibrate` on its own held-out conformal-side values, so the standard split-
    conformal guarantee applies to it exactly like it would to any other exchangeable
    score (`rowii.anomaly.fusion` module docstring's "Statistical note")."""

    def test_mean_realized_far_within_exact_beta_band(self) -> None:
        n_calibration = 159
        n_scoring = 1000
        n_reps = 50
        alpha = 0.05

        fars = np.empty(n_reps)
        sample_corr: float | None = None
        for rep in range(n_reps):
            rng = np.random.default_rng(rep)
            n_total = n_calibration + n_scoring
            latent = rng.normal(0.0, 1.0, n_total)
            # Shared latent + independent per-branch noise -> audio and vibration
            # scores are substantially correlated (~0.78, verified in the scratch
            # check), not independent -- this is the whole point of the test.
            audio_scores = latent + rng.normal(0.0, 0.5, n_total)
            vib_scores = latent + rng.normal(0.0, 0.5, n_total)
            if rep == 0:
                sample_corr = float(np.corrcoef(audio_scores, vib_scores)[0, 1])

            audio_conf = audio_scores[:n_calibration]
            audio_score_side = audio_scores[n_calibration:]
            vib_conf = vib_scores[:n_calibration]
            vib_score_side = vib_scores[n_calibration:]

            # Conformal-side p-values are computed against the SAME conformal set
            # (self-referential, n+1 denominator) so the combined conformal-side
            # statistic has the same shape/role as any other calibration-score
            # array `calibrate` expects; scoring-side p-values use the identical
            # fixed conformal set as their reference -- exactly the orchestration's
            # own "branch p-values for conformal-side AND scoring-side windows"
            # (orchestrator resolution 4).
            p_a_conf = p_values(audio_conf, audio_conf)
            p_v_conf = p_values(vib_conf, vib_conf)
            p_a_score = p_values(audio_score_side, audio_conf)
            p_v_score = p_values(vib_score_side, vib_conf)

            combined_conf = fisher_statistic(p_a_conf, p_v_conf)
            combined_score = fisher_statistic(p_a_score, p_v_score)

            threshold = calibrate(combined_conf, alpha)
            fars[rep] = float(np.mean(combined_score > threshold.threshold))

        # Sanity: the branches really are correlated -- otherwise this test would
        # not distinguish "guarantee restored despite dependence" from "guarantee
        # holds trivially because the branches happen to be independent".
        assert sample_corr is not None and sample_corr > 0.5, sample_corr

        band = beta_band(n_calibration, alpha)
        assert band is not None  # n_calibration=159 >> 1/alpha - 1 = 19
        lo, hi = band
        mean_far = float(fars.mean())
        # Scoring-side binomial slack: n_scoring=1000 keeps each repetition's own
        # sampling noise small (std ~ sqrt(alpha*(1-alpha)/n_scoring) ~= 0.0069),
        # shrunk further by averaging n_reps=50 of them -- a fixed 0.02 margin on
        # each side comfortably covers that residual noise (mirrors
        # tests/test_scarcity.py's own loose validity-test tolerance style, e.g.
        # `test_mean_far_tracks_alpha_above_floor`'s hand-picked `[0.02, 0.09]`).
        slack = 0.02
        assert (lo - slack) <= mean_far <= (hi + slack), (
            f"mean realized FAR {mean_far:.4f} outside beta_band[{lo:.4f}, {hi:.4f}] "
            f"(+/- {slack} slack)"
        )
