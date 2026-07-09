"""Sweep orchestration: per-state (or pooled) conformal anomaly scoring over a single
prepared run, producing a FAR table, per-window scores, and top-K candidates. Step-2
mode-conditioned scoring (design spec `docs/superpowers/specs/
2026-07-09-step2-mode-conditioned-ad-design.md` §2-4, plan `docs/superpowers/plans/
2026-07-09-step2-first-package.md` Task S5).

`run_sweep` composes the three already-built primitives (`rowii.anomaly.references`,
`rowii.anomaly.scorers`, `rowii.anomaly.conformal`) into ONE deterministic, leakage-safe
pass over a `PreparedRun`:

1. **Top-level split** (`split_by_segments`, `cfg.seed`): partitions the run's valid
   windows into a CALIBRATION side and a SCORING side, whole 12-min segments at a time
   (never straddling a segment -- `references.split_by_segments`' own guarantee).
2. **Nested fit/conformal split** (`split_by_segments` again, `cfg.seed + 1`, applied to
   the calibration side ONLY -- via a `valid_mask` restricted to `calibration_windows`,
   the same "mask restricted to a subset" trick `split_by_segments` itself uses
   internally for `valid_mask`). Standard split-conformal needs the reference a scorer is
   FIT on to be disjoint from the scores `calibrate()` treats as exchangeable normal
   draws: fitting `KnnScorer`/`MahalanobisScorer` on a window and then calibrating a
   threshold on that SAME window's own score is a self-scoring leak (kNN k=1 especially
   -- a window's distance to itself is always the global minimum, zero). This splits
   calibration windows into a FIT-part (the reference matrices, via
   `references.build_references`) and a CONFORMAL-part (`calibrate`'s input) --
   three-way disjoint from the scoring side by construction, defensively re-asserted by
   `_assert_three_way_disjoint` before any scoring happens (mirrors
   `references.build_references`'s own "trust but verify" non-finite assert).
3. **Reference + threshold** (`cfg.conditioning`):
   - `"per-state"`: one scorer PER LABEL, fit on that label's own fit-part rows
     (`build_references`), and one threshold PER LABEL, calibrated on that label's own
     conformal-part scores. A label with fewer than `cfg.min_ref` fit-part windows (or
     none at all) is EXCLUDED: no reference, no threshold, every FAR-table metric NaN
     (`_excluded_row`).
   - `"pooled"`: ONE scorer fit on `references.pooled` (every label's fit-part rows
     together) and ONE threshold calibrated on the WHOLE conformal-part's scores (every
     label's conformal windows pooled together, unlike per-state) -- `min_ref`/exclusion
     do not apply (the pooled reference never depends on any single label's own window
     count).
4. **Scoring**: every label's own scoring-part windows are scored against whichever
   scorer applies (per-label or the shared pooled one), flagged `alarm = score >
   threshold`, and ranked by `p_values` (against that same conformal-part scores) for
   the top-`cfg.top_k` candidate register.

`far_table` always carries one row per label seen anywhere in this sweep's calibration
or scoring windows (`excluded=True` rows included, metrics NaN). `conditioning=
"per-state"` additionally appends one aggregate `label="pooled"` row summarising the
realized FAR across every non-excluded label's alarms combined (`_aggregate_pooled_row`)
-- a single number answering "how well did the state-conditioned regime do overall",
distinct from the `conditioning="pooled"` MODE (which never adds this extra row: every
per-label row it emits already shares one scorer/threshold, so a further roll-up would
be redundant, per the dispatch's "+ one 'pooled' row WHEN per-state" wording).

IMPORTANT deviation from the dispatch's own worked intuition for the conditioning-
comparison test, flagged prominently here and in the task report (`.superpowers/sdd/
task-s5-report.md`): the dispatch's test item 3 describes "label B with 10x feature
scale: pooled FAR for tight label A inflates". Verified both theoretically and
empirically (scratch script, not committed) that this direction is backwards. Split
conformal's guarantee is scorer-agnostic as long as a label's threshold is calibrated on
THAT SAME label's own conformal-part scores -- true for `conditioning="per-state"`
regardless of which reference produced the scores, so per-state FAR provably cannot be
moved off alpha by another label's scale at all (the reference is just a fixed function
from the calibration/scoring split's point of view; exchangeability of that one label's
own conformal-part/scoring-part is untouched). Under `conditioning="pooled"`, pooling
label A with a LOOSER-scoring label B can only ever push the SHARED threshold UP
(stochastic dominance: B's systematically larger scores occupy the top of the pooled
conformal-score distribution, so the `ceil((n+1)(1-alpha))`-th order statistic lands
within B's own range) -- which DEFLATES, not inflates, A's realized FAR. The reverse
construction -- pool a LOOSE label with a TIGHTER one -- reliably inflates the LOOSE
label's realized FAR under pooled conditioning (verified: default `KnnScorer` cosine,
10 independent seeds, pooled FAR consistently ~2-3x the per-state FAR, both same-mean
and separated-mean constructions). `test_sweep.py`'s conditioning-comparison test uses
this verified direction, with variable names matching the OBSERVED effect rather than
the dispatch's literal "tight label A" framing; see the task report for the full
derivation and numeric evidence.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from rowii.anomaly.conformal import ConformalThreshold, calibrate, p_values
from rowii.anomaly.references import build_references, split_by_segments
from rowii.anomaly.scorers import KnnScorer, MahalanobisScorer, Scorer
from rowii.pipeline import PreparedRun

_FAR_TABLE_COLUMNS = [
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
_SCORES_COLUMNS = ["window", "label", "score", "p_value", "alarm"]
_CANDIDATES_COLUMNS = ["window", "label", "score", "p_value", "rank"]

_POOLED_ROW_LABEL = "pooled"
"""FAR-table label for the per-state-mode aggregate row (module docstring point 4).
Collides with a real label only if a caller names an actual state "pooled" -- not a
name any of this project's cluster ids (ints) or GT state strings ("standstill",
"turbine", "pump", ...) ever use."""


@dataclass(frozen=True)
class SweepConfig:
    """Configuration for one `run_sweep` call -- see module docstring for semantics."""

    alpha: float = 0.05
    calibration_frac: float = 0.5
    seed: int = 7
    min_ref: int = 20
    top_k: int = 20
    conditioning: Literal["per-state", "pooled"] = "per-state"
    scorer: Literal["knn", "mahalanobis"] = "knn"


@dataclass(frozen=True)
class SweepResult:
    """Output of `run_sweep` -- see module docstring for the three DataFrames' exact
    row/column semantics."""

    far_table: pd.DataFrame
    """One row per label seen in this sweep's calibration/scoring windows, + one
    aggregate `"pooled"` row when `conditioning="per-state"`. Columns:
    `label, n_calibration, n_scored, n_alarms, realized_far, nominal_alpha,
    achievable_alpha_floor, low_confidence, threshold, excluded`."""
    scores: pd.DataFrame
    """One row per (label, scored window) actually scored (excluded labels and empty
    scoring sides contribute none). Columns: `window, label, score, p_value, alarm`."""
    candidates: pd.DataFrame
    """Top `cfg.top_k` lowest-p-value scored windows PER LABEL (fewer if a label has
    fewer than `top_k` scored windows). Columns: `window, label, score, p_value, rank`
    (`rank` 1-based, ascending p-value, ties broken by ascending window index)."""


@dataclass
class _FarRow:
    """Internal, mutable row builder for `SweepResult.far_table` -- one instance per
    label (+ one aggregate instance), converted to a plain dict via `dataclasses.asdict`
    at the very end. `n_calibration`/`n_scored`/`n_alarms` are floats (not ints) purely
    so a NaN "not attempted" row (`_excluded_row`/`_no_conformal_data_row`) can share
    the same field types as a real row -- pandas upcasts the whole column to float64
    once any row contributes a NaN there regardless, so this changes nothing about the
    final DataFrame's dtype."""

    label: int | str
    n_calibration: float
    n_scored: float
    n_alarms: float
    realized_far: float
    nominal_alpha: float
    achievable_alpha_floor: float
    low_confidence: bool
    threshold: float
    excluded: bool


@dataclass
class _ScoreRow:
    window: int
    label: int | str
    score: float
    p_value: float
    alarm: bool


@dataclass
class _CandidateRow:
    window: int
    label: int | str
    score: float
    p_value: float
    rank: int


def _validate_labels(labels: np.ndarray) -> None:
    """`labels` must be an integer dtype (detected cluster ids) or a string/object-of-str
    dtype (GT state names) -- `rowii.anomaly.references.build_references` trusts this
    without checking (module docstring: "labels is deliberately generic"), so `run_sweep`,
    as the layer that accepts raw labels from a caller, validates it explicitly (S2
    review follow-up).

    Raises:
        ValueError: if `labels.dtype` is neither integer nor string/object-of-str.
    """
    kind = labels.dtype.kind
    if kind in "iu":
        return
    if kind in "US":
        return
    if kind == "O" and all(isinstance(v, str) for v in labels):
        return
    raise ValueError(
        f"labels must have an integer or string/object-of-str dtype, got {labels.dtype} "
        f"(dtype.kind={kind!r})"
    )


def _make_scorer(name: str) -> Scorer:
    """A fresh, unfitted scorer instance for `cfg.scorer`, at the exact constructor
    defaults named in the consumed interface (`KnnScorer(k=1, metric="cosine",
    chunk_size=4096)`, `MahalanobisScorer(shrinkage=0.1)`).

    Raises:
        ValueError: if `name` is neither `"knn"` nor `"mahalanobis"` -- a runtime-only
            guard, since `SweepConfig.scorer`'s `Literal` type does not stop an
            arbitrary string reaching here at runtime (matches `KnnScorer.__init__`'s
            own `metric` validation and `pipeline._streams_for_variant`'s `variant`
            validation).
    """
    if name == "knn":
        return KnnScorer()
    if name == "mahalanobis":
        return MahalanobisScorer()
    raise ValueError(f"cfg.scorer must be 'knn' or 'mahalanobis', got {name!r}")


def _assert_three_way_disjoint(
    fit_windows: np.ndarray, conformal_windows: np.ndarray, scoring_windows: np.ndarray
) -> None:
    """Defensive re-check that the fit/conformal/scoring window-index arrays share no
    element (module docstring point 2, the kNN self-scoring hazard) -- structurally
    guaranteed by two correct `split_by_segments` calls, but re-asserted here the same
    way `references.build_references` re-asserts its own upstream-guaranteed invariant
    (all-finite drawn features): trust but verify, cheap at this module's realistic
    window counts (at most a few times `10**5`, per `conformal.py`'s own sizing note).
    """
    fit_set = set(fit_windows.tolist())
    conformal_set = set(conformal_windows.tolist())
    scoring_set = set(scoring_windows.tolist())
    assert fit_set.isdisjoint(conformal_set), (
        "run_sweep: fit-part and conformal-part windows overlap -- self-scoring leak"
    )
    assert fit_set.isdisjoint(scoring_set), (
        "run_sweep: fit-part and scoring windows overlap -- self-scoring leak"
    )
    assert conformal_set.isdisjoint(scoring_set), (
        "run_sweep: conformal-part and scoring windows overlap -- calibration/scoring leak"
    )


def _excluded_row(label: int | str, cfg: SweepConfig) -> _FarRow:
    """A label with fewer than `cfg.min_ref` fit-part windows (or none at all -- both
    read the same way, see `run_sweep`'s `label not in references.references` check):
    no reference was ever fit, so every downstream metric is NaN ("labels excluded by
    min_ref appear with NaN metrics + excluded flag", binding dispatch semantics).
    `low_confidence=True` rather than NaN: unlike the float metric columns, a NaN would
    force this column to `object` dtype (pandas has no float bool), and "no reliable
    calibration exists for this label at all" is unambiguously the maximally
    not-confident case, keeping the column clean `bool` throughout.
    """
    return _FarRow(
        label=label,
        n_calibration=math.nan,
        n_scored=math.nan,
        n_alarms=math.nan,
        realized_far=math.nan,
        nominal_alpha=cfg.alpha,
        achievable_alpha_floor=math.nan,
        low_confidence=True,
        threshold=math.nan,
        excluded=True,
    )


def _no_conformal_data_row(label: int | str, cfg: SweepConfig) -> _FarRow:
    """Same NaN-metrics shape as `_excluded_row`, for the narrower edge case of a label
    that DID clear `cfg.min_ref` on the fit-part (a real reference exists) yet has zero
    windows on the conformal-part (so `calibrate` has nothing to calibrate on) -- an
    unlikely but reachable segment-granularity corner: the nested fit/conformal split
    (`run_sweep`) operates on ALL calibration-side segments together, not per label, so
    a label whose calibration presence is concentrated in very few segments can have
    all of them land on the fit side. `excluded=False` since `min_ref` was not the
    reason (a real reference DOES exist -- just nothing to calibrate a threshold with).
    """
    row = _excluded_row(label, cfg)
    row.excluded = False
    return row


def _empty_scoring_row(
    label: int | str, cfg: SweepConfig, threshold_result: ConformalThreshold
) -> _FarRow:
    """A label with a real reference AND a real calibrated threshold, but zero windows
    on the scoring side: `n_scored=n_alarms=0`, `realized_far=NaN` (0/0 undefined) --
    binding dispatch semantics ("Empty scoring side for a label -> row with n_scored=0,
    NaN far (no crash)"). Every other field reports the real, successfully-calibrated
    threshold (informative on its own, even with nothing yet scored against it).
    """
    return _FarRow(
        label=label,
        n_calibration=float(threshold_result.n_calibration),
        n_scored=0.0,
        n_alarms=0.0,
        realized_far=math.nan,
        nominal_alpha=cfg.alpha,
        achievable_alpha_floor=threshold_result.achievable_alpha_floor,
        low_confidence=threshold_result.low_confidence,
        threshold=threshold_result.threshold,
        excluded=False,
    )


def _scored_row(
    label: int | str,
    cfg: SweepConfig,
    threshold_result: ConformalThreshold,
    n_scored: int,
    n_alarms: int,
) -> _FarRow:
    """A label with a real reference, threshold, and >= 1 scored window --
    `realized_far = n_alarms / n_scored` per the dispatch's literal formula."""
    return _FarRow(
        label=label,
        n_calibration=float(threshold_result.n_calibration),
        n_scored=float(n_scored),
        n_alarms=float(n_alarms),
        realized_far=n_alarms / n_scored,
        nominal_alpha=cfg.alpha,
        achievable_alpha_floor=threshold_result.achievable_alpha_floor,
        low_confidence=threshold_result.low_confidence,
        threshold=threshold_result.threshold,
        excluded=False,
    )


def _aggregate_pooled_row(rows: list[_FarRow], cfg: SweepConfig) -> _FarRow:
    """The extra `label="pooled"` row `run_sweep` appends when `conditioning=
    "per-state"` (module docstring point 4): realized FAR treating every non-excluded
    label's already-computed alarms/scored-counts as one combined bucket --
    `n_scored`/`n_alarms`/`n_calibration` are plain sums across every row that has real
    (non-NaN) data (`_excluded_row`/`_no_conformal_data_row` rows contribute nothing,
    identified via `math.isnan(r.n_scored)` rather than `r.excluded` alone since the
    latter is False for `_no_conformal_data_row` despite it also carrying no real
    counts). `threshold`/`achievable_alpha_floor` are NaN here: per-state conditioning
    calibrates a DIFFERENT threshold per label, so no single scalar threshold value
    describes this aggregate row. `low_confidence` is True iff ANY constituent label
    (excluded or not) was low-confidence -- a conservative "is there a state anywhere
    in this sweep we should not trust" flag.
    """
    contributing = [r for r in rows if not math.isnan(r.n_scored)]
    total_scored = sum(int(r.n_scored) for r in contributing)
    total_alarms = sum(int(r.n_alarms) for r in contributing)
    total_calibration = sum(int(r.n_calibration) for r in contributing)
    any_low_confidence = any(r.low_confidence for r in rows)
    return _FarRow(
        label=_POOLED_ROW_LABEL,
        n_calibration=float(total_calibration) if contributing else math.nan,
        n_scored=float(total_scored),
        n_alarms=float(total_alarms),
        realized_far=(total_alarms / total_scored) if total_scored > 0 else math.nan,
        nominal_alpha=cfg.alpha,
        achievable_alpha_floor=math.nan,
        low_confidence=any_low_confidence,
        threshold=math.nan,
        excluded=False,
    )


def _scores_and_candidates(
    label: int | str,
    windows: np.ndarray,
    scores: np.ndarray,
    p_vals: np.ndarray,
    alarms: np.ndarray,
    top_k: int,
) -> tuple[list[_ScoreRow], list[_CandidateRow]]:
    """Every scored window's `_ScoreRow`, plus the `min(top_k, len(windows))`
    lowest-p-value windows' `_CandidateRow`s (rank 1-based ascending).

    Tie-break order: p-value ascending, then SCORE DESCENDING, then window ascending
    (only to guarantee a fully deterministic order in the astronomically unlikely case
    of an exact score tie too). The score tie-break is not just for determinism -- it is
    load-bearing: `p_values`' definition (`conformal.py` module docstring) means every
    score that exceeds the ENTIRE calibration set collapses to the SAME minimal
    achievable p-value `1/(n+1)`, regardless of how far past the calibration maximum it
    sits (conformal p-values encode RANK relative to calibration, not raw magnitude) --
    with as few as `n_calibration` genuinely-extreme windows this is not a corner case,
    it is the expected outcome for any label with more than a handful of far-out
    windows. Breaking such ties by ascending window index (tried first, see task
    report) silently let ordinary in-distribution windows that happened to have a small
    index outrank genuinely extreme injected outliers -- observed directly on a
    synthetic fixture (`.superpowers/sdd/task-s5-report.md`). Breaking by descending
    score instead surfaces the more extreme reading first, matching what a human
    reviewing the candidate register would want. `np.lexsort`'s primary key is its LAST
    argument, so `(windows, -scores, p_vals)` sorts by `p_vals` first, `-scores`
    (i.e. `scores` descending) second, `windows` last.
    """
    score_rows = [
        _ScoreRow(window=int(w), label=label, score=float(s), p_value=float(p), alarm=bool(a))
        for w, s, p, a in zip(
            windows.tolist(), scores.tolist(), p_vals.tolist(), alarms.tolist(), strict=True
        )
    ]
    top_order = np.lexsort((windows, -scores, p_vals))[:top_k]
    candidate_rows = [
        _CandidateRow(
            window=int(windows[i]),
            label=label,
            score=float(scores[i]),
            p_value=float(p_vals[i]),
            rank=rank,
        )
        for rank, i in enumerate(top_order.tolist(), start=1)
    ]
    return score_rows, candidate_rows


def run_sweep(prepared: PreparedRun, labels: np.ndarray, cfg: SweepConfig) -> SweepResult:
    """Mode-conditioned conformal anomaly sweep over one `PreparedRun` -- see module
    docstring for the full split/reference/threshold/scoring algorithm.

    Args:
        prepared: A `rowii.pipeline.prepare_run` output (or an equivalent hand-built
            `PreparedRun` in tests) -- `features`, `valid_mask`, `segment_ids` are read;
            `grid`/`feature_names` are not used by this function.
        labels: Per-window labels aligned with `prepared.features`, shape `(W,)` --
            either an integer dtype (Step-1 detected cluster ids) or a string/
            object-of-str dtype (GT state names), see `_validate_labels`.
        cfg: Sweep configuration (`SweepConfig`).

    Returns:
        A `SweepResult` (see its own field docs).

    Raises:
        ValueError: if `labels` has neither an integer nor a string/object-of-str
            dtype; if `cfg.conditioning`/`cfg.scorer` is not one of the two literal
            values each accepts (a runtime-only guard -- see `_make_scorer`); if
            `prepared.features.shape[0] != labels.shape[0]` (surfaced by
            `references.build_references`); if `split_by_segments` cannot produce a
            non-empty two-way split at either the top level or the nested fit/conformal
            level (e.g. too few segments, or a `cfg.calibration_frac` that empties one
            side -- see `split_by_segments`' own `Raises`).
    """
    _validate_labels(labels)
    if cfg.conditioning not in ("per-state", "pooled"):
        raise ValueError(
            f"cfg.conditioning must be 'per-state' or 'pooled', got {cfg.conditioning!r}"
        )

    n_windows = prepared.features.shape[0]

    top_split = split_by_segments(
        prepared.segment_ids, prepared.valid_mask, cfg.calibration_frac, cfg.seed
    )
    calibration_windows = top_split.calibration_windows
    scoring_windows = top_split.scoring_windows

    calib_mask = np.zeros(n_windows, dtype=bool)
    calib_mask[calibration_windows] = True
    nested_split = split_by_segments(prepared.segment_ids, calib_mask, 0.5, cfg.seed + 1)
    fit_windows = nested_split.calibration_windows
    conformal_windows = nested_split.scoring_windows

    _assert_three_way_disjoint(fit_windows, conformal_windows, scoring_windows)

    references = build_references(prepared.features, labels, fit_windows, min_ref=cfg.min_ref)

    all_windows = np.concatenate([calibration_windows, scoring_windows])
    all_labels = sorted(np.unique(labels[all_windows]).tolist())

    far_rows: list[_FarRow] = []
    score_rows: list[_ScoreRow] = []
    candidate_rows: list[_CandidateRow] = []

    pooled_scorer: Scorer | None = None
    pooled_conformal_scores: np.ndarray | None = None
    pooled_threshold: ConformalThreshold | None = None
    if cfg.conditioning == "pooled":
        pooled_scorer = _make_scorer(cfg.scorer).fit(references.pooled)
        pooled_conformal_scores = pooled_scorer.score(prepared.features[conformal_windows])
        pooled_threshold = calibrate(pooled_conformal_scores, cfg.alpha)

    for label in all_labels:
        scorer: Scorer
        label_conformal_scores: np.ndarray
        threshold_result: ConformalThreshold

        if cfg.conditioning == "per-state":
            if label not in references.references:
                far_rows.append(_excluded_row(label, cfg))
                continue
            scorer = _make_scorer(cfg.scorer).fit(references.references[label])
            label_conformal_windows = conformal_windows[labels[conformal_windows] == label]
            if label_conformal_windows.shape[0] == 0:
                far_rows.append(_no_conformal_data_row(label, cfg))
                continue
            label_conformal_scores = scorer.score(prepared.features[label_conformal_windows])
            threshold_result = calibrate(label_conformal_scores, cfg.alpha)
        else:
            assert pooled_scorer is not None
            assert pooled_conformal_scores is not None
            assert pooled_threshold is not None
            scorer = pooled_scorer
            label_conformal_scores = pooled_conformal_scores
            threshold_result = pooled_threshold

        label_scoring_windows = scoring_windows[labels[scoring_windows] == label]
        if label_scoring_windows.shape[0] == 0:
            far_rows.append(_empty_scoring_row(label, cfg, threshold_result))
            continue

        label_scores = scorer.score(prepared.features[label_scoring_windows])
        label_p_values = p_values(label_scores, label_conformal_scores)
        label_alarms = label_scores > threshold_result.threshold
        n_scored = int(label_scoring_windows.shape[0])
        n_alarms = int(label_alarms.sum())

        far_rows.append(_scored_row(label, cfg, threshold_result, n_scored, n_alarms))
        new_scores, new_candidates = _scores_and_candidates(
            label, label_scoring_windows, label_scores, label_p_values, label_alarms, cfg.top_k
        )
        score_rows.extend(new_scores)
        candidate_rows.extend(new_candidates)

    if cfg.conditioning == "per-state":
        far_rows.append(_aggregate_pooled_row(far_rows, cfg))

    far_table = pd.DataFrame([asdict(r) for r in far_rows], columns=_FAR_TABLE_COLUMNS)
    scores_df = pd.DataFrame([asdict(r) for r in score_rows], columns=_SCORES_COLUMNS)
    candidates_df = pd.DataFrame([asdict(r) for r in candidate_rows], columns=_CANDIDATES_COLUMNS)

    return SweepResult(far_table=far_table, scores=scores_df, candidates=candidates_df)
