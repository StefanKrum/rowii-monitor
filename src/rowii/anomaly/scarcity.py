"""Calibration-scarcity curves for Step-2 (package-2 spec D3, primary curve).

Answers the partner's "enough data per mode" question quantitatively: how does the
realized false-alarm rate (and its spread) behave as the per-mode conformal
calibration set shrinks? The module is deliberately free of scorer dependencies --
it operates on PRECOMPUTED score arrays (the scorer is fitted once on the full
fit-side reference and both score arrays computed once; only the threshold is
recomputed per subsample), which makes a 50-repetition sweep over 8 budgets a
sub-second operation per state. The deployment-view variant that shrinks the
reference set too (segment accumulation) lives with the CLI (spec D3 secondary).

Per-repetition realized FAR at calibration size n is Beta-distributed -- see the
S-package derivation in tests/test_conformal.py's validity suite -- so `beta_band`
overlays the EXACT `Beta(n + 1 - idx, idx)` quantiles (idx = threshold_index(n,
alpha)), not a binomial approximation. Scoring-side sampling noise adds on top of
that band; reports must say so (spec D3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta_dist

from rowii.anomaly.conformal import calibrate, threshold_index

_CURVE_COLUMNS = (
    "label", "budget", "achieved_n", "saturated", "rep",
    "threshold", "low_confidence", "n_scored", "n_alarms", "realized_far",
)


@dataclass(frozen=True)
class ScarcityConfig:
    """One `scarcity_curve` call's parameters -- see module docstring."""

    budgets: tuple[int, ...] = (5, 10, 19, 39, 79, 159, 319)
    """Requested per-state calibration sizes; 19 is the alpha=0.05 achievability
    floor (n >= 1/alpha - 1) and belongs in every default sweep."""
    n_reps: int = 50
    """Repetitions per budget; rep r draws with `numpy.random.default_rng(r)`."""
    alpha: float = 0.05
    include_full_pool: bool = True
    """Append the full conformal pool as a final budget when it is not already in
    `budgets` -- the 'all available data' anchor point of the curve."""


def scarcity_curve(
    conformal_scores: np.ndarray,
    scoring_scores: np.ndarray,
    label: int | str,
    cfg: ScarcityConfig,
) -> pd.DataFrame:
    """Realized-FAR-vs-calibration-size table for ONE state's precomputed scores.

    Args:
        conformal_scores: `(n_pool,)` finite calibration scores of this state's
            held-out normal windows (full pool; subsampled per budget x rep).
        scoring_scores: `(m,)` finite scores of this state's FIXED scoring windows
            (never subsampled -- spec D3: scoring split fixed across repetitions).
        label: State label carried into the output rows (int cluster id or str).
        cfg: See `ScarcityConfig`.

    Returns:
        DataFrame with columns `label, budget, achieved_n, saturated, rep,
        threshold, low_confidence, n_scored, n_alarms, realized_far` -- one row per
        budget x rep. A saturated budget (requested > pool) draws the whole pool
        (identical across reps, still emitted per rep for uniform aggregation).

    Raises:
        ValueError: propagated from `calibrate` on non-finite/empty inputs.
    """
    n_pool = int(conformal_scores.shape[0])
    m = int(scoring_scores.shape[0])
    budgets = list(cfg.budgets)
    if cfg.include_full_pool and n_pool not in budgets:
        budgets.append(n_pool)

    rows: list[dict[str, object]] = []
    for budget in budgets:
        achieved = min(budget, n_pool)
        saturated = budget > n_pool
        for rep in range(cfg.n_reps):
            if achieved < n_pool:
                rng = np.random.default_rng(rep)
                drawn = rng.choice(conformal_scores, size=achieved, replace=False)
            else:
                drawn = conformal_scores
            th = calibrate(drawn, cfg.alpha)
            n_alarms = int((scoring_scores > th.threshold).sum())
            rows.append({
                "label": label, "budget": budget, "achieved_n": achieved,
                "saturated": saturated, "rep": rep, "threshold": th.threshold,
                "low_confidence": th.low_confidence, "n_scored": m,
                "n_alarms": n_alarms,
                "realized_far": n_alarms / m if m else float("nan"),
            })
    return pd.DataFrame(rows, columns=list(_CURVE_COLUMNS))


def beta_band(
    n: int, alpha: float, q_lo: float = 0.05, q_hi: float = 0.95
) -> tuple[float, float] | None:
    """Exact per-repetition-FAR quantile band at calibration size *n* -- the
    `(q_lo, q_hi)` quantiles of `Beta(n + 1 - idx, idx)` with
    `idx = threshold_index(n, alpha)`; `None` when the threshold order statistic
    does not exist (`idx > n`, below the achievability floor)."""
    idx = threshold_index(n, alpha)
    if idx > n:
        return None
    lo = float(_beta_dist.ppf(q_lo, n + 1 - idx, idx))
    hi = float(_beta_dist.ppf(q_hi, n + 1 - idx, idx))
    return lo, hi
