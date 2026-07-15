"""End-to-end state-detection orchestration: features -> segments.

`run_detection` is a thin convenience wrapper for callers that only need a
single day's detection result and do not care about the fitted intermediate
state: it delegates to `FittedDetector.fit` and returns just the
`DetectionResult` half of its output (Step-1's original entry point, kept for
backward compatibility -- its numeric output is unchanged bit-for-bit).

`FittedDetector` (package-2 spec D1, `docs/superpowers/specs/2026-07-15-
step2-scarcity-crossday-beats-design.md`) splits the SAME pipeline into a
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

from rowii.config import DetectConfig
from rowii.signals.features import apply_zscore, zscore_stats
from rowii.signals.windows import WindowGrid
from rowii.state.cluster import GmmClusterer, KMeansClusterer
from rowii.state.segments import duration_filter, to_segments
from rowii.state.smooth import StickyHmmSmoother


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
    (label any day with day-A parameters) -- package-2 spec D1. Everything the
    chain learns is captured here; `apply` never re-estimates anything:
    standardization uses the FIT day's per-column mean/std (`run_detection`
    z-scores per run, so transfer MUST carry these), labeling is Viterbi decode
    with the FIT day's HMM. Not serialized to disk in this package (spec D1:
    future runtime-prototype serialization point).
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
            CLI (Task 12, `--k` / `--k-sweep`) needs to sweep cluster counts
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
