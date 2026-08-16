"""Calibration-scarcity curves for Step-2.

Answers the partner's "enough data per mode" question quantitatively: how does the
realized false-alarm rate (and its spread) behave as the per-mode conformal
calibration set shrinks? `scarcity_curve` (the PRIMARY curve) is deliberately free
of scorer dependencies -- it operates on PRECOMPUTED score arrays (the scorer is
fitted once on the full fit-side reference and both score arrays computed once;
only the threshold is recomputed per subsample), which makes a 50-repetition sweep
over 8 budgets a sub-second operation per state. `segment_accumulation_curve` (the
SECONDARY, deployment-view curve) breaks that scorer-free
pattern on purpose: it shrinks the FIT/reference side too, not just the
calibration size, so it must refit a fresh scorer at every checkpoint --
"how many more recording MINUTES until this mode is curvable" needs a bigger
reference, and a bigger reference is exactly what more recording minutes buys.
Either way this module stays free of any `PreparedRun`/pipeline dependency; the
CLI (`scripts/run_step2_scarcity.py`) supplies real prepared-run arrays to both
functions.

Per-repetition realized FAR at calibration size n is Beta-distributed -- see the
S-package derivation in tests/test_conformal.py's validity suite -- so `beta_band`
overlays the EXACT `Beta(n + 1 - idx, idx)` quantiles (idx = threshold_index(n,
alpha)), not a binomial approximation. Scoring-side sampling noise adds on top of
that band; reports must say so.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta_dist

from rowii.anomaly.conformal import calibrate, threshold_index

if TYPE_CHECKING:
    # Only needed for the `segment_accumulation_curve` type annotation below --
    # `from __future__ import annotations` makes every annotation a lazily-evaluated
    # string, so this import never runs at module-import time and never becomes a
    # real dependency edge; at runtime `segment_accumulation_curve` accepts any
    # object with `fit(reference) -> Self` / `score(x) -> np.ndarray` (duck typing,
    # matching every other caller of `Scorer` in this package).
    from rowii.anomaly.scorers import Scorer

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
            (never subsampled -- scoring split fixed across repetitions).
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


_SEGMENT_CURVE_COLUMNS = (
    "label", "n_segments", "n_fit", "n_conformal", "minutes", "rep",
    "low_confidence", "n_scored", "n_alarms", "realized_far",
)


@dataclass(frozen=True)
class SegmentAccumulationConfig:
    """One `segment_accumulation_curve` call's parameters -- see that function's
    docstring for the exact accumulation mechanics."""

    prefixes: tuple[int, ...] | None = None
    """Segment-count checkpoints to evaluate; `None` (default) means every even
    count `2, 4, 6, ...` up to the number of available NON-SCORING segments."""
    n_reps: int = 20
    """Repetitions; repetition `r` shuffles the non-scoring segments with
    `numpy.random.default_rng(r)`."""
    alpha: float = 0.05
    min_ref: int = 20
    """Minimum FIT-side windows a label needs at a given checkpoint to get a real
    fitted scorer/threshold; below this (or zero conformal-side windows) the row
    is emitted `low_confidence=True` (`+inf`-threshold semantics, mirroring
    `rowii.anomaly.conformal.calibrate`'s own below-the-floor convention) instead
    -- see `segment_accumulation_curve`."""


def segment_accumulation_curve(
    features: np.ndarray,
    labels: np.ndarray,
    segment_ids: np.ndarray,
    valid_mask: np.ndarray,
    scorer_factory: Callable[[], Scorer],
    scoring_windows: np.ndarray,
    cfg: SegmentAccumulationConfig,
) -> pd.DataFrame:
    """Deployment-view scarcity curve: "how many more
    recording MINUTES per mode until calibration is achievable and stable" --
    unlike `scarcity_curve` (which only resamples the CALIBRATION SIZE out of an
    already-fixed, already-fit reference pool), this shrinks the FIT/reference side
    too, refitting *scorer_factory* from scratch at every checkpoint (module
    docstring).

    Per repetition `r` (`numpy.random.default_rng(r)`): the run's NON-SCORING
    segments (every valid, real segment id -- `segment_ids != -1` -- not present
    among *scoring_windows*' own segments) are permuted, then walked as growing
    prefixes (`cfg.prefixes`, default every even count `2, 4, 6, ...` up to all of
    them). Within one prefix, the EVEN-positioned segments (in permuted order:
    index 0, 2, 4, ...) feed the per-label FIT reference and the ODD-positioned
    segments feed per-label CONFORMAL calibration -- both sides grow together as
    the prefix grows, and a segment, once included at some prefix length, stays on
    the SAME side (fit or conformal) at every longer prefix in that same
    repetition (`order[:n_seg]` is itself always a prefix of `order[:n_seg + 2]`).

    A label needs `>= cfg.min_ref` fit-side windows AND `>= 1` conformal-side
    window at a checkpoint to get a real fitted scorer/threshold; below that floor
    the row is emitted with `low_confidence=True` WITHOUT ever fitting anything --
    `realized_far=0.0`, `n_alarms=0` (a mode with too little data never alarms
    rather than alarming under a false guarantee, the same convention `rowii.
    anomaly.conformal.calibrate` uses for its own `+inf`-threshold rows).

    Args:
        features: `(W, F)` finite feature matrix (only rows selected by the fit/
            conformal/scoring masks below are ever read).
        labels: `(W,)` per-window labels aligned with *features* (int cluster ids
            or GT state strings).
        segment_ids: `(W,)` per-window source-segment id, `PreparedRun.segment_ids`
            convention (`-1` marks a window no segment covers).
        valid_mask: `(W,)` per-window validity, `PreparedRun.valid_mask` convention.
        scorer_factory: Zero-argument callable returning a FRESH, unfitted scorer
            (e.g. `lambda: KnnScorer()`) -- called once per (label, checkpoint,
            rep) that clears the `min_ref`/conformal floor, so every fit is
            independent (no state leaks across checkpoints or reps).
        scoring_windows: `(m,)` FIXED window indices held out from every prefix for
            good (scoring split fixed across repetitions) -- never
            resampled, and their own segments never enter `non_scoring_segments`.
        cfg: See `SegmentAccumulationConfig`.

    Returns:
        DataFrame with columns `label, n_segments, n_fit, n_conformal, minutes,
        rep, low_confidence, n_scored, n_alarms, realized_far` -- one row per
        (label, checkpoint, rep), for every label present in
        `labels[scoring_windows]`. `minutes` is the SAME value across every label
        at one (rep, checkpoint) pair -- it counts TOTAL valid windows drawn into
        that checkpoint's combined fit+conformal pool, across every label at once
        (per-state counts are coupled through shared segments -- stated
        openly), divided by 60 (1-second windows).
    """
    non_scoring_segments = np.setdiff1d(
        np.unique(segment_ids[valid_mask & (segment_ids != -1)]),
        np.unique(segment_ids[scoring_windows]),
    )
    prefixes = cfg.prefixes or tuple(range(2, len(non_scoring_segments) + 1, 2))
    scoring_labels = sorted(np.unique(labels[scoring_windows]).tolist())

    rows: list[dict[str, object]] = []
    for rep in range(cfg.n_reps):
        order = np.random.default_rng(rep).permutation(non_scoring_segments)
        for n_seg in prefixes:
            prefix = order[:n_seg]
            fit_segs, conf_segs = prefix[0::2], prefix[1::2]
            fit_mask = valid_mask & np.isin(segment_ids, fit_segs)
            conf_mask = valid_mask & np.isin(segment_ids, conf_segs)
            minutes = float((fit_mask | conf_mask).sum()) / 60.0
            for label in scoring_labels:
                fit_w = np.flatnonzero(fit_mask & (labels == label))
                conf_w = np.flatnonzero(conf_mask & (labels == label))
                score_w = scoring_windows[labels[scoring_windows] == label]
                n_scored = int(score_w.shape[0])
                if fit_w.shape[0] < cfg.min_ref or conf_w.shape[0] < 1:
                    rows.append({
                        "label": label, "n_segments": n_seg,
                        "n_fit": int(fit_w.shape[0]),
                        "n_conformal": int(conf_w.shape[0]),
                        "minutes": minutes, "rep": rep, "low_confidence": True,
                        "n_scored": n_scored, "n_alarms": 0, "realized_far": 0.0,
                    })
                    continue
                scorer = scorer_factory().fit(features[fit_w])
                th = calibrate(scorer.score(features[conf_w]), cfg.alpha)
                alarms = int((scorer.score(features[score_w]) > th.threshold).sum())
                rows.append({
                    "label": label, "n_segments": n_seg,
                    "n_fit": int(fit_w.shape[0]), "n_conformal": int(conf_w.shape[0]),
                    "minutes": minutes, "rep": rep, "low_confidence": th.low_confidence,
                    "n_scored": n_scored, "n_alarms": alarms,
                    "realized_far": alarms / n_scored,
                })
    return pd.DataFrame(rows, columns=list(_SEGMENT_CURVE_COLUMNS))
