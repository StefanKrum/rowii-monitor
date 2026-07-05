"""End-to-end state-detection orchestration: features -> segments.

`run_detection` chains the pipeline stages built in prior tasks into a single
call: z-score normalisation (`rowii.signals.features.zscore`), clustering
(`KMeansClusterer` or `GmmClusterer`), sticky-HMM smoothing
(`StickyHmmSmoother`), duration-based flicker removal (`duration_filter`), and
segment-table export (`to_segments`). No new numerical logic lives here --
this module only wires the existing, independently-tested stages together in
the fixed order the pipeline requires.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from rowii.config import DetectConfig
from rowii.signals.features import zscore
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
    if features.shape[0] != grid.n_windows:
        raise ValueError(
            f"features.shape[0] ({features.shape[0]}) must equal grid.n_windows "
            f"({grid.n_windows})"
        )

    n_clusters = k if k is not None else cfg.n_states

    z = zscore(features)

    if clusterer == "kmeans":
        init_labels = KMeansClusterer(
            n_clusters=n_clusters, random_seed=cfg.random_seed
        ).fit_predict(z)
    elif clusterer == "gmm":
        init_labels = GmmClusterer(
            n_clusters=n_clusters, random_seed=cfg.random_seed
        ).fit_predict(z)
    else:
        raise ValueError(f"unknown clusterer {clusterer!r}: expected 'kmeans' or 'gmm'")

    smoothed = StickyHmmSmoother(
        self_transition=cfg.self_transition, random_seed=cfg.random_seed
    ).fit_decode(z, init_labels)

    window_s = grid.window_ns / 1e9
    min_dwell = max(1, round(cfg.min_dwell_s / window_s))
    filtered = duration_filter(smoothed, min_dwell=min_dwell)

    segments = to_segments(filtered, grid)

    return DetectionResult(frame_labels=filtered, segments=segments, k=n_clusters)
