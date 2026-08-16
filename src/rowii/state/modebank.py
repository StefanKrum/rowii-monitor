"""Per-mode model bank (Stefan's idea): SCADA ground-truth modes on the FIT days
train one small model per mode
(three families: diagonal Gaussian/Mahalanobis on standardized features, per-mode
cosine-kNN on RAW features, 2-component diagonal GMM on standardized features).
At apply time the bank runs LABEL-FREE: argmin distance / argmax likelihood
assigns each window a mode, and a per-mode split-conformal threshold (`rowii.
anomaly.conformal.calibrate`) on that mode's own calibration-side scores flags a
window rejected by EVERY member as `no_mode_fits` (the "keins passt" novelty
signal, reported as a rate, never a detector without controlled-event evidence).

Standardization: `gaussian`/`gmm` standardize with the GLOBAL pool-FIT-side
mean/std (`rowii.signals.features.zscore_stats` over every surviving mode's fit
rows pooled together, stored on the bank) -- a single shared scale, not a
per-mode one, so per-mode Mahalanobis/GMM distances stay comparable to each
other. `knn` scores raw (unstandardized) features directly, mirroring the
existing Step-2 `KnnScorer` cosine contract.

GT `unknown` AND `transition` windows are excluded from bank TRAINING on both
the fit and calibration sides -- narrower than `rowii.eval.metrics.
evaluate`'s `unknown`-only mask, which callers must account for when comparing
ARI against the unsupervised clusterer arm.

A mode surviving to become a bank member needs BOTH a fit-side reference of at
least `min_ref` windows AND at least one calibration-side window to calibrate a
threshold on; a mode failing either floor is dropped with a `coverage_warnings`-
style (`rowii.anomaly.pools.coverage_warnings`) WARNING log and recorded in
`dropped_modes` -- its GT windows then mis-assign to a surviving mode or get
rejected by every surviving member, which IS the measured, reported behaviour,
not a silent gap.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture

from rowii.anomaly.conformal import ConformalThreshold, calibrate
from rowii.anomaly.scorers import KnnScorer, MahalanobisScorer, Scorer
from rowii.signals.features import apply_zscore, zscore_stats

logger = logging.getLogger(__name__)

_FAMILIES = ("gaussian", "knn", "gmm")
"""The three bank-member families: diagonal Gaussian/Mahalanobis, per-mode
cosine-kNN, and diagonal 2-component GMM -- see module docstring for the
standardization each one scores under."""

_EXCLUDED_GT = ("unknown", "transition")
"""GT state labels never trained on and never a bank member -- excluded
from BOTH the fit and calibration sides before anything else in `ModeBank.fit`."""


class GmmModeScorer:
    """Per-mode 2-component diagonal GMM scorer on the shared `Scorer` contract:
    `score = -GaussianMixture.score_samples(x)`, i.e. negative
    log-likelihood under the fitted mixture -- higher = more anomalous = less
    likely, matching every other scorer in this package. Polarity is fixed by
    construction here (mirrors `rowii.anomaly.scorers`'s OcSvm/IsolationForest/Lof
    scorers' explicit sign-flip convention), never auto-detected.
    """

    name: str = "gmm"

    def __init__(self, n_components: int = 2, random_seed: int = 7) -> None:
        """Args:
            n_components: Number of mixture components (2).
            random_seed: `GaussianMixture`'s own `random_state`, for deterministic
                EM initialization given the same reference.
        """
        self.n_components = n_components
        self.random_seed = random_seed
        self._model = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            random_state=random_seed,
            reg_covar=1e-6,  # sklearn's own default, pinned explicitly for provenance
        )

    def fit(self, reference: np.ndarray) -> GmmModeScorer:
        """Fit the mixture on *reference* (one mode's standardized fit rows).

        Args:
            reference: `(N, F)` finite matrix of one mode's normal reference rows.

        Returns:
            self.
        """
        self._model.fit(reference)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """`(W, F)` windows -> `(W,)` float64 scores, higher = more anomalous
        (`-score_samples(x)`, see class docstring)."""
        return np.asarray(-self._model.score_samples(x), dtype=np.float64)


@dataclass(frozen=True)
class ModeAssignment:
    """Label-free per-window output of `ModeBank.assign`."""

    labels: np.ndarray
    """`(W,)` object array of assigned mode names (argmin/argmax member per
    window); `""` for every window when the bank has no surviving members."""
    scores: np.ndarray
    """`(W, M)` float64 anomaly scores, one column per `modes[j]`, in `modes`
    order; `(W, 0)` when the bank has no surviving members."""
    modes: list[str]
    """The bank's surviving mode names, in the same order as `scores`'s columns
    (a copy of `ModeBank.modes` at assignment time)."""
    no_mode_fits: np.ndarray
    """`(W,)` bool: True iff the window's score exceeded EVERY member's own
    conformal threshold -- rejected by the whole bank ("keins passt"); all False
    when the bank has no surviving members (nothing to be rejected BY)."""


@dataclass(frozen=True)
class ModeBank:
    """A fitted per-mode model bank: one scorer + one conformal threshold per
    surviving GT mode, plus the (optional) global standardization the
    `gaussian`/`gmm` families were fit under. See module docstring for the full
    design rationale; build via `ModeBank.fit`, use via `ModeBank.assign`.
    """

    family: str
    """Which of `_FAMILIES` this bank's members were built with."""
    modes: list[str]
    """Surviving mode names, sorted -- the dict-iteration order every other
    mode-keyed field and `ModeAssignment.scores`'s columns follow."""
    members: dict[str, Scorer]
    """Mode name -> its fitted scorer (`MahalanobisScorer`/`KnnScorer`/
    `GmmModeScorer` per `family`)."""
    mean: np.ndarray | None
    """`(F,)` global pool-fit-side feature mean (`gaussian`/`gmm` only); `None`
    for `knn`, which scores raw features (see module docstring)."""
    std: np.ndarray | None
    """`(F,)` global pool-fit-side feature std, paired with `mean`; `None` for
    `knn`."""
    thresholds: dict[str, ConformalThreshold]
    """Mode name -> its split-conformal rejection threshold, calibrated on that
    mode's own calibration-side scores."""
    calibration_scores: dict[str, np.ndarray]
    """Mode name -> the calibration-side scores `thresholds[mode]` was calibrated
    from (kept for downstream p-value/diagnostic use, e.g. `run_modebank_chain`)."""
    dropped_modes: dict[str, int]
    """Mode name -> its fit-window count, for every GT mode present in the fit
    pool that did NOT survive to become a bank member (below `min_ref`, or zero
    calibration windows)."""
    low_confidence_modes: tuple[str, ...]
    """Sorted names of surviving members (present in `modes`) whose
    `thresholds[mode].low_confidence` is True -- too few calibration scores for
    `alpha` (`rowii.anomaly.conformal.calibrate`), so `threshold` is `+inf` and
    the member can NEVER contribute a rejection to `assign`'s whole-bank
    `no_mode_fits` AND-conjunction (this used to be silent --
    see `fit`'s WARNING log and `assign`'s docstring caveat). Empty when every
    surviving member calibrated with enough data."""
    feature_names: list[str]
    """Column names of the feature matrices this bank was fit on, in order --
    `assign`'s width check is against `len(feature_names)`."""
    alpha: float
    """Nominal false-alarm rate every member's threshold was calibrated at."""
    min_ref: int
    """Minimum fit-side window count a mode needed to become a bank member."""
    k: int
    """`knn` family's neighbour count (unused by `gaussian`/`gmm`, kept for
    provenance/reporting)."""
    gmm_components: int
    """`gmm` family's component count (unused by `gaussian`/`knn`, kept for
    provenance/reporting)."""

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        """Apply this bank's fit-time transform: `apply_zscore` with the stored
        global mean/std for `gaussian`/`gmm`, identity for `knn`."""
        if self.mean is None or self.std is None:
            return x
        return apply_zscore(x, self.mean, self.std)

    @classmethod
    def fit(
        cls,
        fit_features: np.ndarray,
        fit_labels: np.ndarray,
        calib_features: np.ndarray,
        calib_labels: np.ndarray,
        *,
        family: str,
        alpha: float,
        feature_names: list[str],
        min_ref: int = 20,
        k: int = 5,
        gmm_components: int = 2,
        random_seed: int = 7,
    ) -> ModeBank:
        """Fit one scorer + one conformal threshold per GT mode present in
        *fit_features*/*fit_labels*.

        Chain: exclude `_EXCLUDED_GT` from BOTH sides; for `gaussian`/`gmm`,
        compute the GLOBAL pool-fit-side `zscore_stats` over every surviving
        row (module docstring); per mode with `>= min_ref` fit rows AND `>= 1`
        calibration row, fit that mode's scorer on its own (transformed) fit
        rows and `calibrate` a threshold on its own (transformed) calibration
        scores; a mode failing either floor is dropped (`dropped_modes` +
        WARNING log) instead of becoming a member.

        Args:
            fit_features: `(N, F)` finite fit-side feature matrix (pool-fit
                rows across every fit day/run).
            fit_labels: `(N,)` object array of fit-side GT mode names, aligned
                row-for-row with *fit_features*.
            calib_features: `(M, F)` finite calibration-side feature matrix.
            calib_labels: `(M,)` object array of calibration-side GT mode
                names, aligned row-for-row with *calib_features*.
            family: One of `_FAMILIES` (`"gaussian"`, `"knn"`, `"gmm"`).
            alpha: Nominal false-alarm rate every member's threshold targets,
                passed through to `calibrate` unchanged.
            feature_names: Column names of *fit_features*/*calib_features`, in
                order -- stored for `assign`'s width check.
            min_ref: Minimum fit-side row count a mode needs to become a bank
                member (default: 20).
            k: `KnnScorer`'s `k` (`knn` family only).
            gmm_components: `GmmModeScorer`'s component count (`gmm` family
                only).
            random_seed: `GmmModeScorer`'s `GaussianMixture` `random_state`
                (`gmm` family only) -- deterministic EM initialization.

        Returns:
            A `ModeBank` with one member per surviving mode (possibly empty, if
            no mode clears both floors).

        Raises:
            ValueError: if *family* is not one of `_FAMILIES`; if *family* is
                `"knn"` and *min_ref* < *k* (every mode clearing the fit-side
                reference floor must have enough rows for `KnnScorer`'s own
                `k`-vs-reference-size check to never fire -- caught up front
                here with both values named, instead of surfacing later as a
                `KnnScorer.fit` error for whichever mode happens to sit closest
                to the floor); if *fit_features*/*calib_features* is not 2-D,
                its column count does not match `len(feature_names)`, or its
                row count does not match its paired labels array (loud
                geometry -- a caller assembling the pool/GT arrays
                independently is the most likely source of a silent
                misalignment otherwise).
        """
        if family not in _FAMILIES:
            raise ValueError(f"family must be one of {_FAMILIES}, got {family!r}")
        if family == "knn" and min_ref < k:
            raise ValueError(
                f"family='knn' requires min_ref >= k (a mode clearing the fit-side "
                f"reference floor must have enough rows for its own kNN scorer), got "
                f"min_ref={min_ref} and k={k}"
            )
        if fit_features.ndim != 2 or calib_features.ndim != 2:
            raise ValueError(
                f"fit_features/calib_features must be 2-D, got shapes "
                f"{fit_features.shape} / {calib_features.shape}"
            )
        if fit_features.shape[1] != len(feature_names) or calib_features.shape[1] != len(
            feature_names
        ):
            raise ValueError(
                f"fit_features/calib_features must have {len(feature_names)} column(s) "
                f"(len(feature_names)), got {fit_features.shape[1]} / "
                f"{calib_features.shape[1]}"
            )
        if fit_features.shape[0] != fit_labels.shape[0]:
            raise ValueError(
                f"fit_features row count ({fit_features.shape[0]}) must equal "
                f"fit_labels row count ({fit_labels.shape[0]})"
            )
        if calib_features.shape[0] != calib_labels.shape[0]:
            raise ValueError(
                f"calib_features row count ({calib_features.shape[0]}) must equal "
                f"calib_labels row count ({calib_labels.shape[0]})"
            )

        ff = np.asarray(fit_features, dtype=np.float64)
        cf = np.asarray(calib_features, dtype=np.float64)
        fl = np.asarray(fit_labels, dtype=object)
        cl = np.asarray(calib_labels, dtype=object)

        fit_ok = ~np.isin(fl, _EXCLUDED_GT)
        ff, fl = ff[fit_ok], fl[fit_ok]
        cal_ok = ~np.isin(cl, _EXCLUDED_GT)
        cf, cl = cf[cal_ok], cl[cal_ok]

        mean: np.ndarray | None = None
        std: np.ndarray | None = None
        if family in ("gaussian", "gmm"):
            mean, std = zscore_stats(ff)

        def _tf(x: np.ndarray) -> np.ndarray:
            """Fit-time transform: standardize (global fit-side mean/std) for
            `gaussian`/`gmm`, identity for `knn` (module docstring)."""
            if mean is None or std is None:
                return x
            return apply_zscore(x, mean, std)

        members: dict[str, Scorer] = {}
        thresholds: dict[str, ConformalThreshold] = {}
        cal_scores: dict[str, np.ndarray] = {}
        dropped: dict[str, int] = {}

        for mode in sorted(set(fl.tolist())):
            rows = ff[fl == mode]
            n = int(rows.shape[0])
            if n < min_ref:
                dropped[mode] = n
                logger.warning(
                    "modebank: mode %r has %d fit window(s) (< min_ref=%d) -- "
                    "dropped from the bank",
                    mode,
                    n,
                    min_ref,
                )
                continue

            cal_rows = cf[cl == mode]
            if cal_rows.shape[0] == 0:
                dropped[mode] = n
                logger.warning(
                    "modebank: mode %r has a %d-window fit reference but ZERO "
                    "calibration window(s) -- dropped from the bank",
                    mode,
                    n,
                )
                continue

            scorer: Scorer
            if family == "gaussian":
                scorer = MahalanobisScorer().fit(_tf(rows))
            elif family == "knn":
                scorer = KnnScorer(k=k, metric="cosine").fit(rows)
            else:
                scorer = GmmModeScorer(gmm_components, random_seed).fit(_tf(rows))

            scores = scorer.score(_tf(cal_rows))
            members[mode] = scorer
            cal_scores[mode] = scores
            thresholds[mode] = calibrate(scores, alpha)

        if not members:
            logger.warning(
                "modebank: no mode survived the min_ref=%d/calibration floors -- "
                "empty bank",
                min_ref,
            )

        low_confidence_modes = tuple(
            sorted(mode for mode, t in thresholds.items() if t.low_confidence)
        )
        if low_confidence_modes:
            # `rejected &= scores[:, j] > threshold` in `assign`
            # means a member whose threshold is +inf (low_confidence) can NEVER
            # contribute a rejection -- it silently makes the whole-bank
            # no_mode_fits signal under-fire for every window instead of raising
            # or abstaining. Surfacing the affected modes here (WARNING + field)
            # is the fix in scope; the conjunction itself is unchanged (see
            # `assign`'s docstring).
            logger.warning(
                "modebank: %d surviving mode(s) %s calibrated with low_confidence=True "
                "(fewer calibration windows than alpha=%s needs) -- threshold is +inf "
                "for each, so these members can NEVER contribute a rejection and "
                "whole-bank no_mode_fits under-fires while they survive (see "
                "ModeBank.low_confidence_modes)",
                len(low_confidence_modes),
                low_confidence_modes,
                alpha,
            )

        return cls(
            family=family,
            modes=sorted(members),
            members=members,
            mean=mean,
            std=std,
            thresholds=thresholds,
            calibration_scores=cal_scores,
            dropped_modes=dropped,
            low_confidence_modes=low_confidence_modes,
            feature_names=list(feature_names),
            alpha=alpha,
            min_ref=min_ref,
            k=k,
            gmm_components=gmm_components,
        )

    def assign(self, features: np.ndarray) -> ModeAssignment:
        """Label-free per-window mode assignment + whole-bank rejection.

        Every surviving member scores every window (`_standardize`d first for
        `gaussian`/`gmm`); the argmin-scoring member's name is the assigned
        label, and a window is `no_mode_fits` iff its score exceeds every
        member's OWN conformal threshold (`self.thresholds[mode].threshold`) --
        rejected as an outlier by all of them simultaneously, not merely by its
        closest match.

        Caveat: a member listed in `self.low_confidence_modes`
        has `self.thresholds[mode].threshold == +inf` (too few calibration scores
        for `alpha`, see `rowii.anomaly.conformal.calibrate`), so `scores[:, j] >
        threshold` is `False` for every finite score -- that member can NEVER
        contribute a rejection to the AND-conjunction above. `no_mode_fits`
        therefore UNDER-fires (misses windows a fully-calibrated bank would have
        flagged) for as long as ANY surviving member is low-confidence, silently
        turning "not enough data to judge this mode" into an apparent "some mode
        fits". This is the deliberately conservative veto semantics -- abstain
        rather than alarm on too little calibration data -- not a bug to route
        around here; any downstream reporting of `no_mode_fits` MUST also surface
        `self.low_confidence_modes` alongside it.

        Args:
            features: `(W, F)` finite feature matrix, `F == len(feature_names)`.

        Returns:
            A `ModeAssignment`. When the bank has no surviving members (empty
            `self.modes`), `labels` is `""` for every window, `scores` is
            `(W, 0)`, and `no_mode_fits` is all `False` (there are no members to
            reject a window, so "rejected by all members" is vacuously false).

        Raises:
            ValueError: if *features* is not 2-D or its column count does not
                match `len(self.feature_names)` (loud geometry, snapshot
                posture -- matches `FittedDetector.apply`'s width check).
        """
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError(
                f"assign: features must be 2-D with {len(self.feature_names)} "
                f"column(s), got {x.shape}"
            )
        w = x.shape[0]
        if not self.modes:
            return ModeAssignment(
                labels=np.full(w, "", dtype=object),
                scores=np.zeros((w, 0), dtype=np.float64),
                modes=[],
                no_mode_fits=np.zeros(w, dtype=bool),
            )

        xt = self._standardize(x)
        columns = [self.members[mode].score(xt) for mode in self.modes]
        scores = np.column_stack(columns)
        best = np.argmin(scores, axis=1)
        labels = np.array([self.modes[j] for j in best], dtype=object)

        rejected = np.ones(w, dtype=bool)
        for j, mode in enumerate(self.modes):
            rejected &= scores[:, j] > self.thresholds[mode].threshold

        return ModeAssignment(
            labels=labels, scores=scores, modes=list(self.modes), no_mode_fits=rejected
        )
