"""`MonitorSnapshot`: one pickle-free artifact bundling everything the `monitor` CLI
needs -- fitted state detector + per-state scoring references + conformal thresholds
(Step-2 package-6 design spec `docs/superpowers/specs/2026-07-16-step2-package6-
runtime-pillar3-design.md` D1 + amendment A1, plan `docs/superpowers/plans/
2026-07-16-step2-package6-runtime-pillar3.md` Task 1).

**Why numpy-only, `allow_pickle=False` everywhere.** The snapshot is the one artifact
in this repo designed to be moved between machines and loaded months later at the
plant -- pickle's arbitrary-code-on-load hazard is exactly wrong for it, and every
other artifact in this repo (pipeline feature cache, TF-C corpora caches) already
follows the same `np.savez` + `np.load(..., allow_pickle=False)` convention. That
constraint is WHY the runtime scorer set is `{"knn", "mahalanobis"}` and nothing
else: both scorers' fitted state is pure numpy (`KnnScorer` stores an L2-normalized
copy of its reference matrix; `MahalanobisScorer` a per-feature mean + shrunk
variance), and both `fit()` calls are deterministic normalization, NOT training -- so
the snapshot stores the RAW reference matrix per label and simply refits at use time,
bit-identically. The sklearn-backed baselines (OC-SVM/IF/LOF) and the torch AEs would
require pickling their fitted estimators; they are sweep-only comparison poles (spec
D2 non-goals), and `save_snapshot`/`fit_snapshot` refuse them with the whitelist
named.

**Detector half.** `mean`/`std` (fit-day standardization), the sticky HMM's four
parameter arrays as PLAIN arrays, `fitted_ids`, and the `DetectConfig` scalars needed
to rebuild `StickyHmmSmoother`/`FittedDetector`. The sklearn clusterer is NOT stored:
`FittedDetector.apply` never calls it -- Viterbi labels both the fit and apply paths
(documented behavior, `rowii.state.detect`), so the HMM alone reproduces `apply`
exactly. Three reconstruction subtleties, all binding (amendment A1):

- **hmmlearn covars trap (A1.1, measured):** for `covariance_type="diag"` the
  `covars_` GETTER returns full `(k, F, F)` matrices while the SETTER demands
  `(k, F)` diagonals. The snapshot stores `np.diagonal(covars_, axis1=1, axis2=2)`;
  reconstruction assigns that `(k, F)` array back. Viterbi parity of the
  reconstructed model was verified empirically before implementation (identical
  `predict`, non-contiguous label ids).
- **k<=1 degenerate detector (A1.2):** `StickyHmmSmoother.fit_decode` with one
  unique init id leaves `last_model_ = None` and `decode` returns the single fitted
  id for every window. The snapshot always carries `fitted_ids`; the four HMM arrays
  are `None` in memory and ABSENT from the npz exactly in this case, and
  `to_detector` rebuilds the degenerate smoother faithfully (no HMM object at all).
- **component/id invariant:** `fit_decode` builds its id<->component mapping from
  `np.unique(init_labels)` order, so HMM component `i` corresponds to
  `fitted_ids[i]` by construction. Rather than persisting a redundant mapping that
  could drift from the array, extraction ASSERTS the invariant against the live
  smoother (raising `RuntimeError` if the upstream construction ever changes) and
  reconstruction rebuilds `_component_to_id = {i: int(fitted_ids[i])}`.

Reconstruction never calls `fit`: `GaussianHMM(..., params="mc", init_params="")` is
instantiated (the same EM-restriction flags `StickyHmmSmoother.fit_decode` uses, so a
hypothetical later `fit` could still not silently re-estimate the sticky transmat)
and the four arrays are assigned directly.

**Scoring half.** Per surviving label: the raw reference matrix (fit-side windows of
that label), the calibration-score array (the one piece no existing dataclass
captures -- required so the monitor can compute `p_values` against the SAME
calibration set the threshold came from), and the full `ConformalThreshold`.
`fit_snapshot` mirrors `run_sweep`'s split discipline 1:1 (A1.6, verified against
`run_sweep`'s actual code): top split (`calibration_frac`, `seed`) over the run's
valid windows -> nested split of the calibration half (0.5, `seed + 1`) ->
references from the nested FIT side (`build_references`, `min_ref` floor), one
threshold per label calibrated on that label's own nested CONFORMAL-side scores.
A label whose reference exists but whose conformal side is empty is DROPPED from the
snapshot with a warning -- the monitor cannot alarm on a state without a calibrated
threshold, and silently keeping an un-thresholded reference would invite exactly the
false-promise alarms `rowii.anomaly.conformal`'s low-confidence machinery exists to
prevent. (A label excluded by `min_ref` is likewise absent; `build_references`
already logs that case.)

**Provenance / geometry guard.** `variant` + `feature_names` are stored so the
monitor can refuse a mismatched recording LOUDLY at apply time (a snapshot fitted on
`fusion` must never silently score `audio-beats` features -- hmmlearn's `decode`
would not catch the width mismatch itself, see `FittedDetector.apply`'s width-check
rationale). `checkpoints` records every checkpoint path configured in the fitting
environment as plain strings -- provenance only, never validated at load (the
features are already extracted; the snapshot never re-runs a featurizer).

**On-disk layout** (`save_snapshot`): one `.npz` with detector arrays under their own
keys, per-label arrays under `ref__<label>` / `cal__<label>`, and ONE `meta` member
-- a single JSON string in a 1-element str array (the pipeline-cache convention) --
carrying every scalar/string/threshold field. `format_version` gates loading: a
future incompatible layout bumps `SNAPSHOT_FORMAT_VERSION` and old readers fail with
both versions named instead of misreading. A human-readable `<stem>.json` sidecar
duplicates the metadata for quick inspection (`cat`-able provenance next to the
binary artifact); `load_snapshot` never reads it -- the npz is self-contained.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np
from hmmlearn.hmm import GaussianHMM

from rowii.anomaly.conformal import ConformalThreshold, calibrate
from rowii.anomaly.references import build_references, split_by_segments
from rowii.anomaly.scorers import KnnScorer, MahalanobisScorer, Scorer
from rowii.anomaly.sweep import SweepConfig, _assert_three_way_disjoint
from rowii.config import Config
from rowii.pipeline import PreparedRun
from rowii.signals.windows import WindowGrid
from rowii.state.detect import FittedDetector
from rowii.state.smooth import StickyHmmSmoother

logger = logging.getLogger(__name__)

SNAPSHOT_FORMAT_VERSION: int = 1
"""Bump on any incompatible change to the npz layout or meta schema -- `load_snapshot`
refuses a mismatched file naming both versions."""

_RUNTIME_SCORERS: tuple[str, ...] = ("knn", "mahalanobis")
"""The pickle-free runtime scorer whitelist (module docstring) -- everything else is
sweep-only."""

_INVALID_LABEL = -1
"""Sentinel for invalid windows in the full-length label array `fit_snapshot`
returns -- same value as the step2/apply_detector scripts' own `_INVALID_LABEL`
(defined here independently: src modules never import script internals, and scripts
never import each other's)."""

_ClustererName = Literal["kmeans", "gmm"]


@dataclass(frozen=True)
class MonitorSnapshot:
    """Everything the `monitor` CLI needs, as plain data -- see module docstring for
    the design rationale, persistence rules, and reconstruction subtleties."""

    mean: np.ndarray
    """(F,) fit-day per-column feature means (`FittedDetector.mean`)."""
    std: np.ndarray
    """(F,) fit-day per-column feature stds (`FittedDetector.std`)."""
    fitted_ids: np.ndarray
    """(k_eff,) int64 -- the smoother's fitted label ids in `np.unique` order; HMM
    component `i` corresponds to `fitted_ids[i]` (asserted at extraction, module
    docstring). Always present, even for the k<=1 degenerate detector."""
    hmm_startprob: np.ndarray | None
    """(k_eff,) float64, or `None` for the degenerate (single-state) detector --
    all four `hmm_*` fields are `None` together (A1.2)."""
    hmm_transmat: np.ndarray | None
    """(k_eff, k_eff) float64 sticky transition matrix, or `None` (see above)."""
    hmm_means: np.ndarray | None
    """(k_eff, F) float64 per-component emission means, or `None` (see above)."""
    hmm_covars_diag: np.ndarray | None
    """(k_eff, F) float64 per-component emission-covariance DIAGONALS -- stored via
    `np.diagonal(covars_, axis1=1, axis2=2)` because hmmlearn's diag-covariance
    getter/setter shapes disagree (A1.1, module docstring). `None` when degenerate."""
    min_dwell_s: float
    """`FittedDetector.min_dwell_s` -- duration-filter parameter at fit time."""
    k: int
    """Number of clusters REQUESTED at fit time (`FittedDetector.k`). May exceed
    `len(fitted_ids)` if the clusterer collapsed states; the HMM's own component
    count is always `len(fitted_ids)`."""
    self_transition: float
    """`StickyHmmSmoother.self_transition` at fit time (rebuild parameter)."""
    random_seed: int
    """`StickyHmmSmoother.random_seed` at fit time (rebuild parameter)."""
    references: dict[int, np.ndarray]
    """Label -> (Ni, F) RAW reference matrix (fit-side windows of that label) --
    the runtime scorer refits from this deterministically (module docstring)."""
    calibration_scores: dict[int, np.ndarray]
    """Label -> (Nc,) conformal-side calibration scores the threshold was computed
    from -- the monitor's `p_values` reference set."""
    thresholds: dict[int, ConformalThreshold]
    """Label -> the calibrated split-conformal threshold. Keys of `references`/
    `calibration_scores`/`thresholds` are always identical (enforced at save)."""
    scorer: str
    """Runtime scorer name, one of `_RUNTIME_SCORERS` (enforced at fit AND save)."""
    alpha: float
    """Nominal per-state false-alarm target the thresholds were calibrated at."""
    min_ref: int
    """`SweepConfig.min_ref` at fit time -- reference floor for per-label entries."""
    calibration_frac: float
    """`SweepConfig.calibration_frac` at fit time -- the monitor reuses it (with
    `seed`) to split a NEW run for threshold recalibration (spec D2)."""
    seed: int
    """`SweepConfig.seed` at fit time (top split seed; nested split used seed+1)."""
    variant: str
    """Feature variant the snapshot was fitted on (e.g. `"fusion"`) -- geometry
    guard, together with `feature_names`."""
    feature_names: list[str]
    """Fit-day feature column names -- the monitor refuses a new recording whose
    prepared `feature_names` differ (module docstring)."""
    fit_run: str
    """Name of the run the snapshot was fitted on (provenance)."""
    checkpoints: dict[str, str]
    """Config-field-name -> checkpoint path (str) for every checkpoint configured in
    the fitting environment -- provenance only, never validated at load."""
    created_at: str
    """ISO-8601 UTC timestamp of `fit_snapshot` (provenance)."""
    format_version: int
    """`SNAPSHOT_FORMAT_VERSION` at creation -- checked by `load_snapshot`."""


def _runtime_scorer(name: str) -> Scorer:
    """A fresh runtime scorer instance for *name*, at the EXACT constructor defaults
    `rowii.anomaly.sweep._make_scorer` uses for the same names (`KnnScorer(k=1,
    metric="cosine")`, `MahalanobisScorer(shrinkage=0.1)`) -- calibration scores
    stored in the snapshot and monitor-time scores must come from identically
    configured scorers or the conformal guarantee silently breaks.

    Raises:
        ValueError: if *name* is not in `_RUNTIME_SCORERS`, naming the whitelist and
            why it exists (no pickle -- module docstring).
    """
    if name == "knn":
        return KnnScorer(k=1, metric="cosine")
    if name == "mahalanobis":
        return MahalanobisScorer()
    raise ValueError(
        f"scorer {name!r} is not a runtime scorer: the snapshot whitelist is "
        f"{_RUNTIME_SCORERS} (pure-numpy fitted state, refit deterministically from "
        f"the stored reference -- no pickle in the snapshot path; every other scorer "
        f"is sweep-only, spec D1)"
    )


def scorer_for_label(snapshot: MonitorSnapshot, label: int) -> Scorer:
    """The fitted runtime scorer for *label*, refit from the snapshot's stored raw
    reference -- deterministic normalization, not training (module docstring), so
    scores are bit-identical across save/load round trips.

    Args:
        snapshot: The snapshot to draw the reference (and scorer name) from.
        label: A state label present in `snapshot.references`.

    Returns:
        A fitted `Scorer` (higher = more anomalous, the shared contract).

    Raises:
        KeyError: if *label* has no reference in the snapshot -- the message names
            the labels that do (callers gate unknown states BEFORE scoring, spec
            A1.3, so reaching this is a caller bug worth a loud name-carrying error).
        ValueError: if `snapshot.scorer` is not a runtime scorer (only possible for
            a hand-built snapshot -- `fit_snapshot`/`save_snapshot` both refuse).
    """
    if label not in snapshot.references:
        raise KeyError(
            f"label {label} has no reference in this snapshot (known labels: "
            f"{sorted(snapshot.references)})"
        )
    return _runtime_scorer(snapshot.scorer).fit(snapshot.references[label])


def _hmm_arrays(
    smoother: StickyHmmSmoother, fitted_ids: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extract `(startprob, transmat, means, covars_diag)` from a fitted smoother,
    or four `None`s for the k<=1 degenerate case (A1.2) -- asserting the
    component/id invariant the reconstruction in `to_detector` relies on (module
    docstring).

    Raises:
        RuntimeError: if the smoother's live `_component_to_id` mapping is not
            `{i: fitted_ids[i]}`, or its component count is not `len(fitted_ids)` --
            both impossible under `StickyHmmSmoother.fit_decode`'s current
            `np.unique`-ordered construction, so a failure here means that upstream
            construction changed and this module's persistence format must be
            revisited BEFORE any snapshot is written (trust but verify, the
            `run_sweep._assert_three_way_disjoint` style).
    """
    model = smoother.last_model_
    if model is None:
        return None, None, None, None
    if int(model.n_components) != len(fitted_ids):
        raise RuntimeError(
            f"HMM component count ({int(model.n_components)}) != len(fitted_ids) "
            f"({len(fitted_ids)}) -- StickyHmmSmoother's construction changed; the "
            f"snapshot format cannot represent this"
        )
    expected = {i: int(fitted_ids[i]) for i in range(len(fitted_ids))}
    actual = smoother._component_to_id
    if actual is None or {int(c): int(i) for c, i in actual.items()} != expected:
        raise RuntimeError(
            f"component_to_id invariant violated: expected {expected} (component i "
            f"== fitted_ids[i], np.unique order), got {actual} -- "
            f"StickyHmmSmoother's construction changed; the snapshot format cannot "
            f"represent this"
        )
    # A1.1: the diag-covariance GETTER returns (k, F, F) full matrices; store the
    # (k, F) diagonals the SETTER will demand at reconstruction.
    covars_diag = np.ascontiguousarray(np.diagonal(model.covars_, axis1=1, axis2=2))
    return (
        np.asarray(model.startprob_, dtype=np.float64),
        np.asarray(model.transmat_, dtype=np.float64),
        np.asarray(model.means_, dtype=np.float64),
        np.asarray(covars_diag, dtype=np.float64),
    )


def fit_snapshot(
    prepared: PreparedRun,
    rowii_cfg: Config,
    sweep_cfg: SweepConfig,
    *,
    variant: str,
    fit_run: str,
    clusterer: str = "kmeans",
    k: int | None = None,
) -> tuple[MonitorSnapshot, np.ndarray]:
    """Fit a detector on *prepared* and assemble the full `MonitorSnapshot` --
    detector half from `FittedDetector.fit` on the VALID rows, scoring half via
    `run_sweep`'s exact split discipline (A1.6, module docstring).

    Detector fitting mirrors `scripts/apply_detector.py::_fit_detector_and_mapping`'s
    valid-grid pattern: fit on `features[valid_mask]` against a grid of
    `valid_mask.sum()` windows, then scatter labels back to full length with
    `_INVALID_LABEL` on invalid windows. State labels for the scoring half are these
    DETECTED labels (run-time realism -- the monitor will only ever have detected
    states on a new day; spec §4).

    Args:
        prepared: `rowii.pipeline.prepare_run` output for the fit run (or an
            equivalent hand-built `PreparedRun` in tests).
        rowii_cfg: Project config -- `detect` drives the detector fit; the
            checkpoint-path fields are recorded as provenance (`checkpoints`).
        sweep_cfg: Split/threshold parameters (`alpha`, `calibration_frac`, `seed`,
            `min_ref`, `scorer`) -- `conditioning`/`top_k` are sweep-only and
            ignored here (the snapshot is inherently per-state: one reference and
            one threshold per label).
        variant: Feature variant *prepared* was built with (geometry guard field).
        fit_run: Run name (provenance field).
        clusterer: `"kmeans"` or `"gmm"` -- `FittedDetector.fit`'s own choices.
        k: Cluster-count override; `None` uses `rowii_cfg.detect.n_states`.

    Returns:
        `(snapshot, full_labels)` -- the snapshot, and the full-length detected
        label array (shape `(W,)`, int64, `_INVALID_LABEL` on invalid windows) so
        callers can report the fit day's own timeline without re-deriving it.

    Raises:
        ValueError: if `sweep_cfg.scorer` is not a runtime scorer (checked FIRST --
            a snapshot that could never be saved must never be built); if
            *clusterer* is unknown; if either `split_by_segments` call cannot
            produce a non-empty split (the nested failure is re-raised with the
            same run-too-short context `run_sweep` adds).
    """
    _runtime_scorer(sweep_cfg.scorer)  # whitelist gate before any heavy work
    if clusterer not in ("kmeans", "gmm"):
        raise ValueError(f"unknown clusterer {clusterer!r}: expected 'kmeans' or 'gmm'")

    # -- Detector half: fit on VALID rows, scatter labels back to full length.
    valid_mask = prepared.valid_mask
    features_valid = prepared.features[valid_mask]
    valid_grid = WindowGrid(
        t0_ns=prepared.grid.t0_ns,
        window_ns=prepared.grid.window_ns,
        n_windows=int(valid_mask.sum()),
    )
    detector, det_valid = FittedDetector.fit(
        features_valid,
        valid_grid,
        rowii_cfg.detect,
        clusterer=cast(_ClustererName, clusterer),
        k=k,
    )
    n_windows = prepared.features.shape[0]
    full_labels = np.full(n_windows, _INVALID_LABEL, dtype=np.int64)
    full_labels[valid_mask] = det_valid.frame_labels

    smoother = detector.smoother
    assert smoother._fitted_ids is not None  # FittedDetector.fit always fits it
    fitted_ids = np.asarray(smoother._fitted_ids, dtype=np.int64)
    startprob, transmat, means, covars_diag = _hmm_arrays(smoother, fitted_ids)

    # -- Scoring half: run_sweep's exact top/nested split (A1.6).
    top = split_by_segments(
        prepared.segment_ids, prepared.valid_mask, sweep_cfg.calibration_frac, sweep_cfg.seed
    )
    calib_mask = np.zeros(n_windows, dtype=bool)
    calib_mask[top.calibration_windows] = True
    try:
        nested = split_by_segments(prepared.segment_ids, calib_mask, 0.5, sweep_cfg.seed + 1)
    except ValueError as e:
        calib_segments = np.unique(prepared.segment_ids[top.calibration_windows])
        raise ValueError(
            f"nested fit/conformal split failed: calibration side has too few segments "
            f"({len(calib_segments)} segments, {len(top.calibration_windows)} windows); "
            f"the run is too short/sparse for a three-way split — need >= 2 "
            f"calibration-side segments (got {len(calib_segments)}). Consider a longer "
            f"run or different calibration_frac."
        ) from e
    fit_windows = nested.calibration_windows
    conformal_windows = nested.scoring_windows
    _assert_three_way_disjoint(fit_windows, conformal_windows, top.scoring_windows)

    reference_set = build_references(
        prepared.features, full_labels, fit_windows, min_ref=sweep_cfg.min_ref
    )

    references: dict[int, np.ndarray] = {}
    calibration_scores: dict[int, np.ndarray] = {}
    thresholds: dict[int, ConformalThreshold] = {}
    for label in (int(v) for v in np.unique(det_valid.frame_labels).tolist()):
        reference = reference_set.references.get(label)
        if reference is None:
            # Zero fit-side windows, or below min_ref (build_references already
            # warned for the latter) -- either way there is nothing to score against.
            logger.warning(
                "fit_snapshot: label %d has no usable reference (zero or fewer than "
                "min_ref=%d fit-side windows) -- dropped from the snapshot",
                label,
                sweep_cfg.min_ref,
            )
            continue
        label_conformal = conformal_windows[full_labels[conformal_windows] == label]
        if label_conformal.shape[0] == 0:
            logger.warning(
                "fit_snapshot: label %d has a reference (%d fit-side windows) but "
                "ZERO conformal-side windows -- dropped from the snapshot (no "
                "threshold can be calibrated, and the monitor must never alarm on a "
                "state without one)",
                label,
                reference.shape[0],
            )
            continue
        scores = _runtime_scorer(sweep_cfg.scorer).fit(reference).score(
            prepared.features[label_conformal]
        )
        references[label] = reference
        calibration_scores[label] = scores
        thresholds[label] = calibrate(scores, sweep_cfg.alpha)

    if not thresholds:
        logger.warning(
            "fit_snapshot: NO label survived the reference/conformal requirements -- "
            "the snapshot carries a detector but an empty scoring half (the monitor "
            "will label states yet never alarm)"
        )

    checkpoints = {
        name: str(path)
        for name, path in (
            ("beats_checkpoint", rowii_cfg.beats_checkpoint),
            ("tfc_audio_checkpoint", rowii_cfg.tfc_audio_checkpoint),
            ("tfc_vib_checkpoint", rowii_cfg.tfc_vib_checkpoint),
            ("student_checkpoint", rowii_cfg.student_checkpoint),
            ("beats_int8_checkpoint", rowii_cfg.beats_int8_checkpoint),
            ("xattn_checkpoint", rowii_cfg.xattn_checkpoint),
        )
        if path is not None
    }

    snapshot = MonitorSnapshot(
        mean=np.asarray(detector.mean, dtype=np.float64),
        std=np.asarray(detector.std, dtype=np.float64),
        fitted_ids=fitted_ids,
        hmm_startprob=startprob,
        hmm_transmat=transmat,
        hmm_means=means,
        hmm_covars_diag=covars_diag,
        min_dwell_s=detector.min_dwell_s,
        k=detector.k,
        self_transition=smoother.self_transition,
        random_seed=smoother.random_seed,
        references=references,
        calibration_scores=calibration_scores,
        thresholds=thresholds,
        scorer=sweep_cfg.scorer,
        alpha=sweep_cfg.alpha,
        min_ref=sweep_cfg.min_ref,
        calibration_frac=sweep_cfg.calibration_frac,
        seed=sweep_cfg.seed,
        variant=variant,
        feature_names=list(prepared.feature_names),
        fit_run=fit_run,
        checkpoints=checkpoints,
        created_at=datetime.now(UTC).isoformat(),
        format_version=SNAPSHOT_FORMAT_VERSION,
    )
    return snapshot, full_labels


def to_detector(snapshot: MonitorSnapshot) -> FittedDetector:
    """Rebuild the `FittedDetector` from *snapshot* -- assignment only, NEVER a
    `fit`/EM call anywhere (the whole point of the fit/apply split: the fit day's
    emission model must reach the apply day unchanged).

    The `GaussianHMM` is constructed with the same flags `StickyHmmSmoother.
    fit_decode` uses (`params="mc"`, `init_params=""`, the fit-time `random_state`)
    and its component count is `len(fitted_ids)` -- the HMM's own k, which
    `_hmm_arrays` asserted equals the stored arrays' leading dimension; the
    snapshot's `k` field (clusters REQUESTED) is carried onto the `FittedDetector`
    unchanged but never sizes the HMM. Covars are assigned from the stored `(k, F)`
    DIAGONALS (A1.1). For the degenerate single-state snapshot (`hmm_startprob is
    None`) the smoother is rebuilt with `last_model_ = None` and `decode` reproduces
    the constant `fitted_ids[0]` labeling (A1.2).
    """
    smoother = StickyHmmSmoother(
        self_transition=snapshot.self_transition, random_seed=snapshot.random_seed
    )
    fitted_ids = np.asarray(snapshot.fitted_ids, dtype=np.int64)
    smoother._fitted_ids = fitted_ids
    if snapshot.hmm_startprob is not None:
        assert snapshot.hmm_transmat is not None  # all-or-none by format (A1.2)
        assert snapshot.hmm_means is not None
        assert snapshot.hmm_covars_diag is not None
        model = GaussianHMM(
            n_components=len(fitted_ids),
            covariance_type="diag",
            params="mc",
            init_params="",
            random_state=snapshot.random_seed,
        )
        model.startprob_ = snapshot.hmm_startprob
        model.transmat_ = snapshot.hmm_transmat
        model.means_ = snapshot.hmm_means
        model.covars_ = snapshot.hmm_covars_diag  # setter takes (k, F) diagonals
        smoother.last_model_ = model
        smoother._component_to_id = {
            i: int(fitted_ids[i]) for i in range(len(fitted_ids))
        }
    return FittedDetector(
        mean=snapshot.mean,
        std=snapshot.std,
        smoother=smoother,
        min_dwell_s=snapshot.min_dwell_s,
        k=snapshot.k,
    )


def _meta_dict(snapshot: MonitorSnapshot) -> dict[str, object]:
    """Every non-array field as one JSON-ready dict -- the npz `meta` member and the
    `.json` sidecar share it verbatim. `labels` pins which `ref__`/`cal__` members
    exist (JSON keys are strings, so threshold keys are stringified label ids and
    `labels` preserves the real int values). A low-confidence threshold's `+inf`
    serializes as JSON `Infinity` -- a Python-`json` extension to strict JSON,
    round-tripped exactly by `json.loads` (the only reader of the npz meta member;
    the sidecar is for humans)."""
    return {
        "format_version": snapshot.format_version,
        "min_dwell_s": snapshot.min_dwell_s,
        "k": snapshot.k,
        "self_transition": snapshot.self_transition,
        "random_seed": snapshot.random_seed,
        "scorer": snapshot.scorer,
        "alpha": snapshot.alpha,
        "min_ref": snapshot.min_ref,
        "calibration_frac": snapshot.calibration_frac,
        "seed": snapshot.seed,
        "variant": snapshot.variant,
        "feature_names": snapshot.feature_names,
        "fit_run": snapshot.fit_run,
        "checkpoints": snapshot.checkpoints,
        "created_at": snapshot.created_at,
        "labels": sorted(snapshot.thresholds),
        "thresholds": {
            str(label): {
                "threshold": t.threshold,
                "alpha": t.alpha,
                "n_calibration": t.n_calibration,
                "achievable_alpha_floor": t.achievable_alpha_floor,
                "low_confidence": t.low_confidence,
            }
            for label, t in snapshot.thresholds.items()
        },
    }


def save_snapshot(path: Path, snapshot: MonitorSnapshot) -> None:
    """Persist *snapshot* to *path* (npz, written through an open file handle so the
    name is used EXACTLY as given -- `np.savez`'s implicit `.npz`-appending never
    applies) plus a human-readable `<stem>.json` metadata sidecar.

    Args:
        path: Target npz path (conventionally `*.npz`); parents are created.
        snapshot: The snapshot to persist.

    Raises:
        ValueError: if `snapshot.scorer` is not in the runtime whitelist (module
            docstring -- a non-numpy scorer cannot round-trip without pickle); or if
            the `references`/`calibration_scores`/`thresholds` key sets differ (a
            hand-modified snapshot -- the on-disk format reconstructs all three
            dicts from ONE label list, so a mismatch would corrupt silently).
    """
    if snapshot.scorer not in _RUNTIME_SCORERS:
        raise ValueError(
            f"cannot save snapshot with scorer {snapshot.scorer!r}: the runtime "
            f"whitelist is {_RUNTIME_SCORERS} (pure-numpy fitted state only -- no "
            f"pickle in the snapshot path, spec D1)"
        )
    if not (
        set(snapshot.references)
        == set(snapshot.calibration_scores)
        == set(snapshot.thresholds)
    ):
        raise ValueError(
            f"snapshot label sets disagree: references={sorted(snapshot.references)}, "
            f"calibration_scores={sorted(snapshot.calibration_scores)}, "
            f"thresholds={sorted(snapshot.thresholds)} -- refusing to write a "
            f"corrupt artifact"
        )

    arrays: dict[str, np.ndarray] = {
        "mean": np.asarray(snapshot.mean, dtype=np.float64),
        "std": np.asarray(snapshot.std, dtype=np.float64),
        "fitted_ids": np.asarray(snapshot.fitted_ids, dtype=np.int64),
    }
    if snapshot.hmm_startprob is not None:
        assert snapshot.hmm_transmat is not None  # all-or-none by format (A1.2)
        assert snapshot.hmm_means is not None
        assert snapshot.hmm_covars_diag is not None
        arrays["hmm_startprob"] = np.asarray(snapshot.hmm_startprob, dtype=np.float64)
        arrays["hmm_transmat"] = np.asarray(snapshot.hmm_transmat, dtype=np.float64)
        arrays["hmm_means"] = np.asarray(snapshot.hmm_means, dtype=np.float64)
        arrays["hmm_covars_diag"] = np.asarray(snapshot.hmm_covars_diag, dtype=np.float64)
    for label in sorted(snapshot.thresholds):
        arrays[f"ref__{label}"] = np.asarray(snapshot.references[label], dtype=np.float64)
        arrays[f"cal__{label}"] = np.asarray(
            snapshot.calibration_scores[label], dtype=np.float64
        )
    meta = _meta_dict(snapshot)
    arrays["meta"] = np.array([json.dumps(meta)], dtype=str)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        # allow_pickle=False at WRITE time too: refuses any object-dtype array
        # outright rather than trusting the dtype coercions above.
        np.savez(f, allow_pickle=False, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def load_snapshot(path: Path) -> MonitorSnapshot:
    """Load a `MonitorSnapshot` from *path* -- `allow_pickle=False`, format-version
    gated (checked FIRST, before any other member is interpreted).

    Args:
        path: An npz written by `save_snapshot`.

    Returns:
        The reconstructed snapshot; array bytes are exactly as saved, so
        `to_detector`/`scorer_for_label` reproduce apply/score output bitwise (the
        round-trip parity tests pin this).

    Raises:
        ValueError: if the file has no `meta` member (not a snapshot), or if its
            `format_version` differs from `SNAPSHOT_FORMAT_VERSION` (both versions
            named -- never misread an incompatible layout).
    """
    with np.load(path, allow_pickle=False) as data:
        if "meta" not in data.files:
            raise ValueError(
                f"{path} is not a MonitorSnapshot: no 'meta' member in the npz"
            )
        meta = json.loads(str(data["meta"][0]))
        version = int(meta["format_version"])
        if version != SNAPSHOT_FORMAT_VERSION:
            raise ValueError(
                f"snapshot {path} has format_version {version}, but this reader "
                f"supports only {SNAPSHOT_FORMAT_VERSION} -- refusing to misread an "
                f"incompatible layout"
            )

        labels = [int(label) for label in meta["labels"]]
        references = {label: data[f"ref__{label}"] for label in labels}
        calibration_scores = {label: data[f"cal__{label}"] for label in labels}
        thresholds = {
            label: ConformalThreshold(
                threshold=float(meta["thresholds"][str(label)]["threshold"]),
                alpha=float(meta["thresholds"][str(label)]["alpha"]),
                n_calibration=int(meta["thresholds"][str(label)]["n_calibration"]),
                achievable_alpha_floor=float(
                    meta["thresholds"][str(label)]["achievable_alpha_floor"]
                ),
                low_confidence=bool(meta["thresholds"][str(label)]["low_confidence"]),
            )
            for label in labels
        }
        has_hmm = "hmm_startprob" in data.files

        return MonitorSnapshot(
            mean=data["mean"],
            std=data["std"],
            fitted_ids=data["fitted_ids"],
            hmm_startprob=data["hmm_startprob"] if has_hmm else None,
            hmm_transmat=data["hmm_transmat"] if has_hmm else None,
            hmm_means=data["hmm_means"] if has_hmm else None,
            hmm_covars_diag=data["hmm_covars_diag"] if has_hmm else None,
            min_dwell_s=float(meta["min_dwell_s"]),
            k=int(meta["k"]),
            self_transition=float(meta["self_transition"]),
            random_seed=int(meta["random_seed"]),
            references=references,
            calibration_scores=calibration_scores,
            thresholds=thresholds,
            scorer=str(meta["scorer"]),
            alpha=float(meta["alpha"]),
            min_ref=int(meta["min_ref"]),
            calibration_frac=float(meta["calibration_frac"]),
            seed=int(meta["seed"]),
            variant=str(meta["variant"]),
            feature_names=[str(name) for name in meta["feature_names"]],
            fit_run=str(meta["fit_run"]),
            checkpoints={str(k_): str(v) for k_, v in meta["checkpoints"].items()},
            created_at=str(meta["created_at"]),
            format_version=version,
        )
