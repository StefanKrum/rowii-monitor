"""Unit tests for rowii.anomaly.scarcity (package-2 spec D3, primary + secondary curves)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rowii.anomaly.conformal import threshold_index
from rowii.anomaly.scarcity import (
    ScarcityConfig,
    SegmentAccumulationConfig,
    beta_band,
    scarcity_curve,
    segment_accumulation_curve,
)
from rowii.anomaly.scorers import KnnScorer


def _pool(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n), rng.normal(0, 1, 2000)  # conformal, scoring


class TestScarcityCurve:
    def test_deterministic_given_seeds(self):
        conformal, scoring = _pool()
        cfg = ScarcityConfig(budgets=(10, 39), n_reps=5)
        a = scarcity_curve(conformal, scoring, 2, cfg)
        b = scarcity_curve(conformal, scoring, 2, cfg)
        pd.testing.assert_frame_equal(a, b)

    def test_below_floor_never_alarms_and_flags(self):
        conformal, scoring = _pool()
        cfg = ScarcityConfig(budgets=(5,), n_reps=3, alpha=0.05, include_full_pool=False)
        df = scarcity_curve(conformal, scoring, "turbine", cfg)
        assert df["low_confidence"].all()
        assert (df["threshold"] == np.inf).all()
        assert (df["realized_far"] == 0.0).all()

    def test_saturation_flag_and_achieved_n(self):
        conformal, scoring = _pool(n=50)
        cfg = ScarcityConfig(budgets=(79,), n_reps=2, include_full_pool=False)
        df = scarcity_curve(conformal, scoring, 2, cfg)
        assert df["saturated"].all()
        assert (df["achieved_n"] == 50).all()

    def test_full_pool_row_appended(self):
        conformal, scoring = _pool(n=400)
        cfg = ScarcityConfig(budgets=(19,), n_reps=2, include_full_pool=True)
        df = scarcity_curve(conformal, scoring, 2, cfg)
        assert set(df["budget"]) == {19, 400}

    def test_mean_far_tracks_alpha_above_floor(self):
        # Exchangeable normal scores: mean realized FAR across reps must sit near
        # alpha once n clears the floor (loose gate — per-rep FAR is Beta-spread).
        conformal, scoring = _pool(n=1000, seed=1)
        cfg = ScarcityConfig(budgets=(159,), n_reps=50, alpha=0.05, include_full_pool=False)
        df = scarcity_curve(conformal, scoring, 2, cfg)
        assert 0.02 <= df["realized_far"].mean() <= 0.09


class TestBetaBand:
    def test_none_below_floor(self):
        assert beta_band(5, 0.05) is None

    def test_band_brackets_alpha_at_large_n(self):
        lo, hi = beta_band(999, 0.05)
        assert lo < 0.05 < hi
        assert hi - lo < 0.03  # tight at n=999

    def test_band_widens_at_small_n(self):
        lo_small, hi_small = beta_band(19, 0.05)
        lo_big, hi_big = beta_band(999, 0.05)
        assert (hi_small - lo_small) > (hi_big - lo_big)

    def test_matches_threshold_index_parameters(self):
        n, alpha = 99, 0.05
        idx = threshold_index(n, alpha)
        from scipy.stats import beta as beta_dist
        lo, hi = beta_band(n, alpha)
        assert lo == pytest.approx(beta_dist.ppf(0.05, n + 1 - idx, idx))
        assert hi == pytest.approx(beta_dist.ppf(0.95, n + 1 - idx, idx))


class TestSegmentAccumulation:
    def _run(self, n_segments=12, seg_len=60, seed=0):
        rng = np.random.default_rng(seed)
        feats, segs, labels = [], [], []
        for s in range(n_segments):
            value = 5.0 * (s % 2)
            feats.append(rng.normal(value, 0.1, (seg_len, 2)))
            segs.append(np.full(seg_len, s, dtype=np.int64))
            labels.append(np.full(seg_len, s % 2, dtype=np.int64))
        return (np.vstack(feats), np.concatenate(labels), np.concatenate(segs),
                np.ones(n_segments * seg_len, dtype=bool))

    def test_scoring_segments_never_in_prefix(self):
        features, labels, segment_ids, valid = self._run()
        scoring_windows = np.flatnonzero(np.isin(segment_ids, [10, 11]))
        cfg = SegmentAccumulationConfig(n_reps=3)
        df = segment_accumulation_curve(
            features, labels, segment_ids, valid,
            lambda: KnnScorer(), scoring_windows, cfg,
        )
        # every row's n_scored equals the fixed scoring-side count for its label
        for label in (0, 1):
            expected = int((labels[scoring_windows] == label).sum())
            got = df[df["label"] == label]["n_scored"].unique()
            assert list(got) == [expected]

    def test_deterministic_and_monotone_minutes(self):
        features, labels, segment_ids, valid = self._run()
        scoring_windows = np.flatnonzero(np.isin(segment_ids, [10, 11]))
        cfg = SegmentAccumulationConfig(n_reps=2)
        a = segment_accumulation_curve(features, labels, segment_ids, valid,
                                       lambda: KnnScorer(), scoring_windows, cfg)
        b = segment_accumulation_curve(features, labels, segment_ids, valid,
                                       lambda: KnnScorer(), scoring_windows, cfg)
        pd.testing.assert_frame_equal(a, b)
        one_rep = a[(a["rep"] == 0) & (a["label"] == 0)].sort_values("n_segments")
        assert one_rep["minutes"].is_monotonic_increasing

    def test_starved_prefix_flags_low_confidence(self):
        features, labels, segment_ids, valid = self._run(n_segments=6, seg_len=8)
        scoring_windows = np.flatnonzero(np.isin(segment_ids, [4, 5]))
        cfg = SegmentAccumulationConfig(n_reps=2, min_ref=20)
        df = segment_accumulation_curve(features, labels, segment_ids, valid,
                                        lambda: KnnScorer(), scoring_windows, cfg)
        starved = df[df["n_segments"] == 2]
        assert starved["low_confidence"].all()
        assert (starved["realized_far"] == 0.0).all()
