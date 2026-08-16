"""End-to-end state-detection orchestration: features -> segments.

`run_detection` is a thin convenience wrapper for callers that only need a
single day's detection result and do not care about the fitted intermediate
state: it delegates to `FittedDetector.fit` and returns just the
`DetectionResult` half of its output (Step-1's original entry point, kept for
backward compatibility -- its numeric output is unchanged bit-for-bit).

`FittedDetector` splits the SAME pipeline into a
`fit` half (learn on one day: `zscore_stats`/`apply_zscore` standardisation,
`KMeansClusterer`/`GmmClusterer` initial clustering, `StickyHmmSmoother.
fit_decode` smoothing) and an `apply` half (label ANY day -- including a
DIFFERENT day's features -- with the fit day's parameters: `apply_zscore`
with the fit day's mean/std, `StickyHmmSmoother.decode` with the fit day's
HMM, no refit or EM anywhere). This split is what lets a detector trained on
one day's recording be reused, unchanged, to label a different day's
recording -- the prerequisite for Step-2's cross-day comparisons.

Both paths share duration-based flicker removal (`duration_filter`) and
segment-table export (`to_segments`) as their common tail
(`FittedDetector._finish`). No new numerical logic lives here -- this module
only wires the existing, independently-tested stages together in the fixed
order the pipeline requires.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from rowii.config import Config, DetectConfig
from rowii.signals.features import apply_zscore, zscore_stats
from rowii.signals.windows import WindowGrid
from rowii.state.cluster import GmmClusterer, KMeansClusterer
from rowii.state.segments import duration_filter, to_segments
from rowii.state.smooth import StickyHmmSmoother, _init_means_covars, _sticky_transmat


@dataclass(frozen=True)
class DetectionResult:
    """Output of `run_detection`: per-window labels, the segment table, and k."""

    frame_labels: np.ndarray
    """Per-window state labels after smoothing + duration filtering, shape (W,), int64."""
    segments: pd.DataFrame
    """`to_segments` output: one row per maximal run of `frame_labels`."""
    k: int
    """Number of clusters actually used (either `k` if given, else `cfg.n_states`)."""


@dataclass(frozen=True)
class FittedDetector:
    """The Step-1 detection chain, split into fit (learn on day A) and apply
    (label any day with day-A parameters). Everything the
    chain learns is captured here; `apply` never re-estimates anything:
    standardization uses the FIT day's per-column mean/std (`run_detection`
    z-scores per run, so transfer MUST carry these), labeling is Viterbi decode
    with the FIT day's HMM. Not serialized to disk in this package (a
    future runtime-prototype serialization point).

    Two usage caveats (final whole-branch review, 2026-07-15): the per-window
    labeler on BOTH the fit and apply paths is `StickyHmmSmoother.decode`
    (Viterbi with fit-day emissions) — `KMeansClusterer.predict` /
    `GmmClusterer.predict` are reserved transfer primitives that `apply`
    deliberately does NOT call, since Viterbi discards any init assignment and
    decode-only is what makes `apply(fit_features) == fit` hold. And although
    this dataclass is `frozen=True`, `smoother` is a mutable object: freezing
    blocks field REASSIGNMENT only — never call `fit_decode` on a
    `FittedDetector`'s smoother directly, or the captured emission model is
    silently re-estimated in place.
    """

    mean: np.ndarray
    """(F,) fit-day per-column feature means (`zscore_stats`)."""
    std: np.ndarray
    """(F,) fit-day per-column feature stds (`zscore_stats`)."""
    smoother: StickyHmmSmoother
    """Fitted sticky HMM (holds the emission model + fit-time label-id mapping);
    `decode` is the per-window labeler on both fit and apply days."""
    min_dwell_s: float
    """`DetectConfig.min_dwell_s` at fit time (duration filter parameter)."""
    k: int
    """Number of clusters requested at fit time (mirrors `DetectionResult.k`)."""

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        grid: WindowGrid,
        cfg: DetectConfig,
        clusterer: Literal["kmeans", "gmm"] = "kmeans",
        k: int | None = None,
    ) -> tuple[FittedDetector, DetectionResult]:
        """Run the full Step-1 chain on *features* and capture the fitted state.

        Identical chain, order, and parameters as the historical `run_detection`
        (which now delegates here) -- existing e2e/real-data results are the
        regression gate for that equivalence.

        Args:
            features: Per-window feature matrix, shape (W, F), to fit on.
            grid: The `WindowGrid` *features* was extracted against.
            cfg: Detection parameters (`n_states`, `self_transition`,
                `min_dwell_s`, `random_seed`).
            clusterer: Which clustering algorithm to use: `"kmeans"` or `"gmm"`.
            k: Number of clusters to request, overriding `cfg.n_states` when given.

        Returns:
            A `(FittedDetector, DetectionResult)` pair: the fitted detector
            (for later `apply` calls on other days) and this fit day's own
            `DetectionResult`, identical to what `run_detection` would return.

        Raises:
            ValueError: same conditions as `run_detection` (shape mismatch,
                unknown clusterer).
        """
        if features.shape[0] != grid.n_windows:
            raise ValueError(
                f"features.shape[0] ({features.shape[0]}) must equal grid.n_windows "
                f"({grid.n_windows})"
            )
        n_clusters = k if k is not None else cfg.n_states

        mean, std = zscore_stats(features)
        z = apply_zscore(features, mean, std)

        if clusterer == "kmeans":
            init_labels = KMeansClusterer(
                n_clusters=n_clusters, random_seed=cfg.random_seed
            ).fit_predict(z)
        elif clusterer == "gmm":
            init_labels = GmmClusterer(
                n_clusters=n_clusters, random_seed=cfg.random_seed
            ).fit_predict(z)
        else:
            raise ValueError(
                f"unknown clusterer {clusterer!r}: expected 'kmeans' or 'gmm'"
            )

        smoother = StickyHmmSmoother(
            self_transition=cfg.self_transition, random_seed=cfg.random_seed
        )
        smoothed = smoother.fit_decode(z, init_labels)

        detector = cls(
            mean=mean, std=std, smoother=smoother,
            min_dwell_s=cfg.min_dwell_s, k=n_clusters,
        )
        result = detector._finish(smoothed, grid)
        return detector, result

    @classmethod
    def fit_pooled(
        cls,
        pooled_features: np.ndarray,
        cfg: Config,
        *,
        k: int,
        clusterer: str = "kmeans",
    ) -> FittedDetector:
        """Fit a detector on POOLED multi-run features: pooled KMeans emissions,
        per-run Viterbi decode -- by design, "pump owns a
        cluster".

        Chain: `zscore_stats`/`apply_zscore` over the POOLED matrix (the returned
        detector standardizes every later run with these pooled statistics),
        `KMeansClusterer(k, cfg.detect.random_seed)` on the pooled z-scored
        features (label ids = KMeans cluster ids `0..k-1`), per-cluster emission
        means/covars via `_init_means_covars` on the pooled labels, and a
        `GaussianHMM` assembled EXACTLY like `StickyHmmSmoother.fit_decode`'s
        construction (same flags, `_sticky_transmat` transition matrix, uniform
        startprob) -- but with **no `model.fit` call anywhere**.

        **The no-EM decision (binding).** `fit_decode`'s Baum-Welch
        refinement is a single-run operation: running it over CONCATENATED runs
        would let EM treat the artificial jump from the last window of one run to
        the first window of the next as a real observed transition -- a cross-run
        EM chain over boundaries that never existed in time. Pooled emissions
        therefore stay at their per-cluster moment estimates, and decoding is per
        run ONLY: each `apply(features, grid)` call Viterbi-decodes ONE run's
        windows against the fixed pooled emission model. For the same reason,
        callers must never concatenate several runs into a single `apply` call.

        **k is not chosen here.** k is selected at execution time by a
        GT-state-ARI sweep (k in {4, 5, 6} on the pool days) and reported; this
        method only fits the k it is given.

        **Row order matters.** sklearn's k-means++
        seeding is NOT guaranteed bit-identical under row permutation even with a
        fixed `random_state`: two calls on the SAME array are bit-identical, but
        a permuted copy may yield a different (equally valid) clustering with
        permuted cluster ids. Callers MUST therefore pass *pooled_features* in a
        deterministic row order. `rowii.anomaly.pools.PoolResult.features`'
        stacked order is deterministic given the same prepared-dict insertion
        order -- pass exactly that array. (This method takes the plain ARRAY, not
        the `PoolResult`, so this module stays free of anomaly-package imports.)

        Because label ids are `0..k-1` by construction, `_fitted_ids` is
        `arange(k)` and `_component_to_id` the identity -- exactly the
        component/id invariant `rowii.runtime.snapshot._hmm_arrays` asserts, so
        the returned detector round-trips through the EXISTING snapshot
        extraction unchanged. `k == 1` mirrors `fit_decode`'s degenerate path
        (`last_model_ is None`; `apply` labels every window 0).

        Args:
            pooled_features: (N, F) stacked feature rows of ALL pool runs' fit
                sides, in deterministic row order (see above).
            cfg: Project config -- `cfg.detect` supplies `random_seed`,
                `self_transition`, and `min_dwell_s`.
            k: Number of pooled clusters to fit (>= 1).
            clusterer: Only `"kmeans"` is supported: a
                GMM's `fit_predict` cannot guarantee every component owns >= 1
                window, which the `0..k-1` label-id invariant above requires.

        Returns:
            A plain `FittedDetector` whose `apply` performs per-run Viterbi
            decoding with the pooled emission model.

        Raises:
            ValueError: if *clusterer* is not `"kmeans"`, *k* < 1, or
                *pooled_features* is not 2-D.
            RuntimeError: if KMeans returns labels other than exactly `0..k-1`.
                This CAN happen (observed on sklearn 1.9): when the pooled
                data has fewer effectively-distinct clusters than *k* --
                duplicate or near-constant feature rows (extended idle/steady
                stretches), or a *k* above the natural cluster count -- KMeans
                may leave label ids unassigned. Guarded loudly here because a
                silent gap would violate the `_fitted_ids = arange(k)` identity
                invariant the snapshot extraction relies on; the execution's
                k-sweep treats this error as "k too large for this pool".
        """
        if clusterer != "kmeans":
            raise ValueError(
                f"unknown clusterer {clusterer!r}: fit_pooled supports 'kmeans' "
                f"only (A3.4 -- see the docstring's clusterer note)"
            )
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if pooled_features.ndim != 2:
            raise ValueError(
                f"pooled_features must be 2-D (N, F), got shape "
                f"{pooled_features.shape}"
            )

        detect_cfg = cfg.detect
        mean, std = zscore_stats(pooled_features)
        z = apply_zscore(pooled_features, mean, std)

        smoother = StickyHmmSmoother(
            self_transition=detect_cfg.self_transition,
            random_seed=detect_cfg.random_seed,
        )

        if k == 1:
            # fit_decode's own k<=1 degenerate contract: no HMM
            # at all; `decode` labels every window with the single fitted id 0.
            smoother._fitted_ids = np.arange(1, dtype=np.int64)
            return cls(
                mean=mean, std=std, smoother=smoother,
                min_dwell_s=detect_cfg.min_dwell_s, k=k,
            )

        labels = KMeansClusterer(
            n_clusters=k, random_seed=detect_cfg.random_seed
        ).fit_predict(z)
        if not np.array_equal(np.unique(labels), np.arange(k)):
            raise RuntimeError(
                f"pooled KMeans did not assign every cluster id 0..{k - 1} "
                f"(got {np.unique(labels).tolist()}) -- the 0..k-1 label-id "
                f"invariant this construction relies on is broken"
            )

        z64 = np.asarray(z, dtype=np.float64)
        transmat = _sticky_transmat(k, detect_cfg.self_transition)
        startprob = np.full(k, 1.0 / k)
        means, covars = _init_means_covars(z64, labels, k, z64.shape[1])

        model = GaussianHMM(
            n_components=k,
            covariance_type="diag",
            params="mc",
            init_params="",
            random_state=detect_cfg.random_seed,
        )
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model.covars_ = covars
        # NO model.fit(...) here -- assignment only (docstring rationale).
        # The one fit-side effect the assignment path must replicate: hmmlearn's
        # `covars_` GETTER (used by the snapshot extraction, `_hmm_arrays`) needs
        # `n_features`, which `fit`/`_check` would set. `predict` sets it too,
        # but extraction must not depend on an `apply` having happened first.
        model.n_features = z64.shape[1]

        smoother.last_model_ = model
        smoother._fitted_ids = np.arange(k, dtype=np.int64)
        smoother._component_to_id = {i: i for i in range(k)}

        return cls(
            mean=mean, std=std, smoother=smoother,
            min_dwell_s=detect_cfg.min_dwell_s, k=k,
        )

    def apply(self, features: np.ndarray, grid: WindowGrid) -> DetectionResult:
        """Label *features* with the fit-day parameters: fit-day standardization ->
        fit-day HMM Viterbi decode -> duration filter -> segments. No refit, no EM.

        Args:
            features: Per-window feature matrix, shape (W, F), to label. May be
                a DIFFERENT day's features than the one this detector was fit
                on -- that is the whole point of the fit/apply split.
            grid: The `WindowGrid` *features* was extracted against.

        Returns:
            A `DetectionResult` for *features*, using this detector's fit-day
            mean/std, HMM, and `min_dwell_s`.

        Raises:
            ValueError: if `features.shape[0] != grid.n_windows`, or if
                `features.shape[1] != self.mean.shape[0]` (the fit-day feature
                count). The latter is checked explicitly because `apply` is
                exactly the entry point that receives ANOTHER day's feature
                array: `apply_zscore` would fail on a mismatched column count
                with an unrelated broadcasting error, and hmmlearn's `decode`
                does not validate feature width at all -- it would silently
                decode garbage against the fit-day HMM's emission model rather
                than raising, which must fail loudly instead.
        """
        if features.shape[0] != grid.n_windows:
            raise ValueError(
                f"features.shape[0] ({features.shape[0]}) must equal grid.n_windows "
                f"({grid.n_windows})"
            )
        if features.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"features.shape[1] ({features.shape[1]}) must equal the fit-day "
                f"feature count ({self.mean.shape[0]})"
            )
        z = apply_zscore(features, self.mean, self.std)
        smoothed = self.smoother.decode(z)
        return self._finish(smoothed, grid)

    def _finish(self, smoothed: np.ndarray, grid: WindowGrid) -> DetectionResult:
        """Shared tail of fit/apply: duration filter + segments."""
        window_s = grid.window_ns / 1e9
        min_dwell = max(1, round(self.min_dwell_s / window_s))
        filtered = duration_filter(smoothed, min_dwell=min_dwell)
        segments = to_segments(filtered, grid)
        return DetectionResult(frame_labels=filtered, segments=segments, k=self.k)


def run_detection(
    features: np.ndarray,
    grid: WindowGrid,
    cfg: DetectConfig,
    clusterer: Literal["kmeans", "gmm"] = "kmeans",
    k: int | None = None,
) -> DetectionResult:
    """Run the full state-detection pipeline: zscore -> cluster -> smooth -> filter -> segments.

    Chain (fixed order, matching the Step-1 pipeline spec):
        1. `zscore(features)` -- per-column standardisation.
        2. `clusterer(k or cfg.n_states, cfg.random_seed).fit_predict(...)` --
           initial per-window cluster labels.
        3. `StickyHmmSmoother(cfg.self_transition, cfg.random_seed).fit_decode(...)` --
           Viterbi-smoothed labels using a fixed high-self-transition HMM.
        4. `duration_filter(labels, min_dwell=max(1, round(cfg.min_dwell_s / window_s)))`
           where `window_s = grid.window_ns / 1e9` -- merges runs shorter than
           `min_dwell` windows into a neighbour.
        5. `to_segments(labels, grid)` -- per-run segment table with UTC boundaries.

    Args:
        features: Per-window feature matrix, shape (W, F).
        grid: The `WindowGrid` the features were extracted against. Used both
            for the min_dwell-in-windows conversion (via `window_ns`) and for
            `to_segments`'s UTC boundaries.
        cfg: Detection parameters (`n_states`, `self_transition`, `min_dwell_s`,
            `random_seed`).
        clusterer: Which clustering algorithm to use: `"kmeans"` or `"gmm"`.
        k: Number of clusters to request, overriding `cfg.n_states` when given.
            Not part of the original Step-1 plan signature; added because the
            CLI (`--k` / `--k-sweep`) needs to sweep cluster counts
            without constructing a new `DetectConfig` per value. When `None`
            (the default), behaviour is identical to the original plan: `k`
            defaults to `cfg.n_states`.

    Returns:
        A `DetectionResult` with the smoothed+filtered `frame_labels`, the
        `segments` table, and the `k` actually used.

    Raises:
        ValueError: if `features.shape[0] != grid.n_windows`, or if
            `clusterer` is not `"kmeans"` or `"gmm"`.
    """
    _, result = FittedDetector.fit(features, grid, cfg, clusterer=clusterer, k=k)
    return result
