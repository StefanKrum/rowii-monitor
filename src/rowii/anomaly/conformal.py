"""Split-conformal calibration for Step-2 per-mode anomaly thresholds (design spec
`docs/superpowers/specs/2026-07-09-step2-mode-conditioned-ad-design.md` §2-3, design
chapter §3.5 "Mode conditioning and thresholds", plan `docs/superpowers/plans/
2026-07-09-step2-first-package.md` Task S4).

`calibrate` turns one mode's held-out normal calibration scores (from
`rowii.anomaly.scorers`, one scalar per window, higher = more anomalous) into a single
decision threshold with a distribution-free finite-sample false-alarm guarantee: for
exchangeable normal data, `P(score > threshold) <= alpha`, and, when scores are almost
surely distinct (continuous score distribution), `P(score > threshold) >= alpha -
1/(n + 1)` -- so the realised false-alarm rate is pinned within `1/(n + 1)` of the
nominal target `alpha` (Angelopoulos & Bates 2022, "A Gentle Introduction to Conformal
Prediction and Distribution-Free Uncertainty Quantification", arXiv:2107.07511; design
§3.5). With `n` calibration scores sorted ascending `s_(1) <= ... <= s_(n)`, the
threshold is the `idx`-th smallest, `idx = ceil((n + 1) * (1 - alpha))` (1-based). That
order statistic exists only while `idx <= n`, equivalently `n >= 1/alpha - 1`
(`ConformalThreshold.achievable_alpha_floor`, `1 / (n + 1)`, is the smallest alpha ANY
threshold could certify at this `n`, regardless of whether THIS call's `alpha` cleared
it). Below that floor, `calibrate` sets `threshold = +inf` and `low_confidence = True`
rather than silently emitting a threshold with no valid guarantee behind it -- a mode
that has not accumulated enough held-out normal data should never alarm, not alarm
under a false promise.

`p_values` reports, for a batch of new scores, the standard conformal p-value against
the same calibration set: `p_i = (1 + #{j : calibration_scores[j] >= scores[i]}) /
(n + 1)`, vectorized via `np.searchsorted` on the sorted calibration scores (a tie
counts toward the `>=` set, the conservative direction). These p-values are
super-uniform under the same exchangeability assumption -- `P(p <= t) <= t` for every
`t` in `[0, 1]` -- so `p_values` is designed to double as the ranking key for Step-2's
outlier sweep (spec §2 candidate register, Task S5): smaller p-value = more anomalous,
comparable across scorers and modes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Tolerance subtracted from `(n + 1) * (1 - alpha)` before `math.ceil`, guarding
# against floating-point representation error nudging a mathematically-exact integer
# boundary a few ulps ABOVE that integer -- e.g. alpha=1/3, n=29: the exact value is
# 20, but double-precision arithmetic rounds `(1 - 1/3)` down first, and the product
# comes out as 20.000000000000004, which `math.ceil` would otherwise send to 21,
# silently off by one in a threshold-defining index. An undershoot (raw slightly BELOW
# an intended integer) needs no such correction: `math.ceil` already rounds it up to
# the correct integer on its own. Safe at the calibration-set sizes this module is
# used with (at most a few times 10**5 windows, see `references.build_references`):
# the product's absolute floating-point error there is many orders of magnitude below
# 1e-9, so this can only ever correct representation noise, never mask a genuinely
# fractional value close to (but not at) an integer.
_CEIL_TOLERANCE = 1e-9


def threshold_index(n: int, alpha: float) -> int:
    """1-based rank `idx = ceil((n + 1) * (1 - alpha))` of the calibration order
    statistic `calibrate` needs -- see module docstring. Clamped to >= 1: for any
    `alpha` in `(0, 1)` the true (unperturbed) value is always > 0, so the only way
    the tolerance-adjusted computation could reach <= 0 is `_CEIL_TOLERANCE` itself
    swallowing a legitimately tiny positive value (alpha within `1e-9 / (n + 1)` of 1,
    an extreme setting no realistic false-alarm target approaches) -- the clamp maps
    that case back to the correct answer, 1, instead of an invalid non-positive index.

    `calibrate`'s `low_confidence` flag is derived from comparing this same `idx` to
    `n`, not recomputed via a separate `n < 1 / alpha - 1` floating-point comparison,
    so the threshold and the flag can never disagree at a boundary case.
    """
    raw = (n + 1) * (1.0 - alpha) - _CEIL_TOLERANCE
    return max(1, math.ceil(raw))


def _raise_if_non_finite(values: np.ndarray, name: str) -> None:
    """Shared precondition for `calibrate` and `p_values`: every element finite.

    Raises:
        ValueError: naming the offending count and first offending index.
    """
    non_finite = ~np.isfinite(values)
    if non_finite.any():
        bad = np.flatnonzero(non_finite)
        raise ValueError(
            f"{name} contains {int(non_finite.sum())} non-finite value(s) (first "
            f"offending index {int(bad[0])}) -- must be all-finite"
        )


@dataclass(frozen=True)
class ConformalThreshold:
    """Per-mode split-conformal decision threshold, from `calibrate`."""

    threshold: float
    """Flag a window as anomalous iff its score exceeds this value. `+inf` iff
    `low_confidence` (see below) -- a mode with too little calibration data to certify
    `alpha` never alarms rather than alarming under a false guarantee."""
    alpha: float
    """Nominal false-alarm rate this threshold targets (the `alpha` `calibrate` was
    called with)."""
    n_calibration: int
    """Number of calibration scores `calibrate` was given (`n`)."""
    achievable_alpha_floor: float
    """`1 / (n + 1)` -- the smallest `alpha` this many calibration scores could ever
    certify, independent of whether THIS call's `alpha` cleared that floor."""
    low_confidence: bool
    """True iff `n_calibration < 1 / alpha - 1`, equivalently the `idx`-th calibration
    order statistic does not exist (`idx > n`) -- `threshold` is `+inf` in that case."""


def calibrate(calibration_scores: np.ndarray, alpha: float) -> ConformalThreshold:
    """Split-conformal threshold from one mode's held-out normal calibration scores.

    Args:
        calibration_scores: `(n,)` finite anomaly scores of normal (in-mode) held-out
            windows, higher = more anomalous (e.g. `rowii.anomaly.scorers.KnnScorer`
            output) -- at least 1 element.
        alpha: Nominal false-alarm rate target, in `(0, 1)`.

    Returns:
        A `ConformalThreshold` (see field docs): `threshold` is the
        `ceil((n + 1) * (1 - alpha))`-th smallest calibration score (1-based) when
        that order statistic exists (`n >= 1 / alpha - 1`), else `+inf` with
        `low_confidence=True`.

    Raises:
        ValueError: if `alpha` is not in `(0, 1)`; if `calibration_scores` has fewer
            than 1 element; if `calibration_scores` has a non-finite value.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    n = calibration_scores.shape[0]
    if n < 1:
        raise ValueError(f"calibration_scores must have at least 1 element, got {n}")
    _raise_if_non_finite(calibration_scores, "calibration_scores")

    idx = threshold_index(n, alpha)
    if idx <= n:
        threshold = float(np.sort(calibration_scores)[idx - 1])
        low_confidence = False
    else:
        threshold = math.inf
        low_confidence = True

    return ConformalThreshold(
        threshold=threshold,
        alpha=alpha,
        n_calibration=n,
        achievable_alpha_floor=1.0 / (n + 1),
        low_confidence=low_confidence,
    )


def p_values(scores: np.ndarray, calibration_scores: np.ndarray) -> np.ndarray:
    """Standard conformal p-value of each of `scores` against `calibration_scores`.

    `p_i = (1 + #{j : calibration_scores[j] >= scores[i]}) / (n + 1)`, vectorized via
    `np.searchsorted` on the sorted calibration scores. A tie (`calibration_scores[j]
    == scores[i]`) counts toward the `>=` set -- the conservative direction, matching
    `calibrate`'s own order-statistic convention and preserving the super-uniform
    guarantee `P(p <= t) <= t` under exact ties.

    Args:
        scores: `(m,)` finite scores to evaluate, higher = more anomalous. May be
            empty.
        calibration_scores: `(n,)` finite calibration scores, at least 1 element --
            the same set `calibrate` would be given for this mode.

    Returns:
        `(m,)` float64 p-values in `(0, 1]`; smaller = more anomalous relative to the
        calibration set.

    Raises:
        ValueError: if `calibration_scores` has fewer than 1 element; if either array
            has a non-finite value.
    """
    n = calibration_scores.shape[0]
    if n < 1:
        raise ValueError(f"calibration_scores must have at least 1 element, got {n}")
    _raise_if_non_finite(calibration_scores, "calibration_scores")
    _raise_if_non_finite(scores, "scores")

    sorted_calibration = np.sort(calibration_scores)
    less_than_count: np.ndarray = np.searchsorted(sorted_calibration, scores, side="left")
    at_least_as_extreme = n - less_than_count
    result: np.ndarray = (1.0 + at_least_as_extreme) / (n + 1)
    return result


def loo_p_values(calibration_scores: np.ndarray) -> np.ndarray:
    """Leave-one-out conformal p-value of each calibration score against the OTHER
    calibration scores: `p_i = #{j : calibration_scores[j] >= calibration_scores[i]}
    / n`.

    Derivation: the standard conformal p-value of point `i` against a reference of
    the other `n - 1` points is `p_i = (1 + #{j != i : s_j >= s_i}) / ((n - 1) + 1)`
    (the `p_values` formula with the reference shrunk by one). Since `s_i >= s_i`
    always, the self-excluded count is `#{j != i : s_j >= s_i} = #{all j : s_j >=
    s_i} - 1`, so the `+1` numerator correction folds the self-exclusion away:
    `p_i = #{all j : s_j >= s_i} / n` -- computable directly from the FULL array via
    the same sorted/`searchsorted` machinery as `p_values`, no per-`i` loop. A tie
    counts toward the `>=` set, the conservative direction, matching `p_values`' own
    tie convention.

    Why this exists (score-fusion review fix, 2026-07-15): `scripts/run_step2.py`'s
    score-level fusion view calibrates a COMBINED statistic of two branches'
    per-window p-values (`rowii.anomaly.fusion`), which requires the calibration-side
    windows' p-values to be computed on the SAME footing as the scoring-side
    windows' (one fixed transform applied to both sides, else calibration/scoring
    exchangeability of the combined statistic breaks). The naive
    `p_values(calibration_scores, calibration_scores)` puts each point in its OWN
    reference (its self-match forces `p >= 2 / (n + 1)`, while a scoring point can
    reach `1 / (n + 1)`) -- for a single branch that mismatch is a monotone
    transform and cancels, but COMBINED across two branches it is anti-conservative:
    measured mean realized FAR up to ~0.10 at alpha=0.05, n=39, anti-correlated
    branches (review simulation, 2026-07-15, scratch scripts not committed). This
    leave-one-out form evaluates each calibration point against a reference that
    excludes it, exactly like a scoring point's reference excludes the scoring point.

    Residual one-unit granularity: the LOO reference has `n - 1` points where the
    scoring reference has `n`, so the smallest achievable LOO p-value is `1 / n`
    versus `1 / (n + 1)` scoring-side. At any given raw score `s` with tie-inclusive
    count `c = #{j : s_j >= s}`, the LOO p-value `c / n` is <= the scoring-side
    `(1 + c) / (n + 1)` (equivalent to `c <= n`, always true), so calibration-side
    combined statistics are weakly INFLATED relative to a perfectly-shared
    transform and the calibrated threshold sits weakly higher -- the residual can
    only SUPPRESS scoring-side alarms relative to that ideal, never add them
    (conservative direction). Validated empirically FOR THE FISHER-combined
    statistic: mean realized FAR at or below alpha within Monte-Carlo precision
    (one-sided `alpha + 3*SE` bound, SE <= 0.002) across dependence regimes --
    independent, shared-latent-correlated (rho ~ 0.78), anti-correlated, and
    identical branches -- at n in {39, 159} (`tests/test_fusion.py`'s multi-regime
    validity test) and additionally at n=319 in the review-time simulation. For the
    TIPPETT (min-rule) statistic this LOO rescaling is decision-neutral: it is a
    strictly monotone transform of the calibration-side min-rank, so every alarm
    decision is bit-identical to the self-referential construction's -- and Tippett
    retains a small intrinsic excess under positively correlated branches (measured
    ~+0.007 at n=39, decaying ~1/n) that NO calibration-side p-value construction
    sharing the calibration set as reference can remove; see `rowii.anomaly.fusion`'s
    module docstring for the scoped claim and the deliberate no-third-split
    trade-off.

    Args:
        calibration_scores: `(n,)` finite calibration scores, at least 1 element,
            higher = more anomalous -- the same set `calibrate`/`p_values` would be
            given. (`n = 1` degrades gracefully: the LOO reference is empty and the
            single p-value is `(1 + 0) / 1 = 1`.)

    Returns:
        `(n,)` float64 leave-one-out p-values in `(0, 1]` (specifically in
        `[1/n, 1]`); smaller = more anomalous relative to the rest of the set.

    Raises:
        ValueError: if `calibration_scores` has fewer than 1 element or a non-finite
            value.
    """
    n = calibration_scores.shape[0]
    if n < 1:
        raise ValueError(f"calibration_scores must have at least 1 element, got {n}")
    _raise_if_non_finite(calibration_scores, "calibration_scores")

    sorted_calibration = np.sort(calibration_scores)
    less_than_count: np.ndarray = np.searchsorted(
        sorted_calibration, calibration_scores, side="left"
    )
    at_least_as_extreme = n - less_than_count
    result: np.ndarray = at_least_as_extreme / n
    return result
