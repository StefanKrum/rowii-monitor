"""Label-free per-session robust normalization.

The runtime-prototype evidence decomposed the frozen-threshold failure into two
causes; the GLOBAL one (250526's recording-chain shift: every state's score median
0.86-2.51x threshold) is exactly what a per-session affine renormalization can
absorb. `fit_session_stats` estimates a robust center/scale (per-column median and
MAD * 1.4826, the normal-consistency factor) from the FIRST `norm_minutes` of a
run's VALID windows -- observable on a new day without any labels -- and
`apply_session_norm` maps that run's feature matrix into the session-normalized
space. The scale is floored at 1e-8, the house silent-window divide-by-zero
convention (`rowii.adapt.target_windows._standardize_1d` / `rowii.tfc.wrapper.
_standardize`).

**A3.5 boundary (BINDING).** The state DETECTOR always consumes RAW features --
detected labels are norm-invariant by construction (the detector carries its own
fit-day standardization). Session normalization applies to the SCORING space only:
references, calibration scores, and scoring windows, each transformed with their
OWN session's stats (the fit day's references with the fit day's stats -- stored in
the `MonitorSnapshot` since format v2 -- and a monitored/test run's windows with
that run's own first-N stats). Comparability of scores/thresholds across the raw
and normalized spaces is FAR-level only (A3.5): a normalized-space threshold is
never compared to a raw-space score.

**Pooled-snapshot stats (`fit_pool_stats`, plan Task 4 design decision).** A pooled
artifact has no single fit day, so no first-N prefix defines its reference-side
stats. The committed choice: center/scale over the POOLED fit matrix as a whole,
carrying `norm_minutes == 0.0` as an explicit sentinel ("pool-global stats, not a
first-N prefix"). The monitor applies them exactly like fit-day stats -- the
reference-side transform -- and, seeing the sentinel, falls back to its default
first-N window for the monitored run's own stats. `fit_session_stats` refuses
`norm_minutes <= 0` so the sentinel can never be produced by the prefix path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rowii.signals.windows import WindowGrid

_MAD_TO_SIGMA = 1.4826
"""Normal-consistency factor: for Gaussian data, `MAD * 1.4826` estimates sigma --
so the normalized space is comparable to a z-score space while staying robust to
the (unlabeled!) anomalies the first minutes might contain."""

_SCALE_FLOOR = 1e-8
"""The house divide-by-zero floor (`rowii.adapt.target_windows._standardize_1d`'s
convention), applied AFTER the 1.4826 scaling: a feature column constant over the
estimation rows gets scale 1e-8, never 0."""


@dataclass(frozen=True)
class SessionStats:
    """One session's robust normalization parameters, from `fit_session_stats` (or
    `fit_pool_stats` for the pooled sentinel variant)."""

    center: np.ndarray
    """(F,) float64 -- per-column median over the estimation rows."""
    scale: np.ndarray
    """(F,) float64 -- per-column `MAD * 1.4826`, floored at 1e-8."""
    n_windows: int
    """Number of rows the stats were estimated from (valid first-N windows, or the
    whole pooled matrix for `fit_pool_stats`)."""
    norm_minutes: float
    """The first-N prefix length in minutes -- or `0.0`, the POOL-GLOBAL sentinel
    (module docstring): these stats came from a pooled matrix, not a prefix."""


def _center_scale(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column median + floored `MAD * 1.4826` of *rows* (float64)."""
    center = np.median(rows, axis=0).astype(np.float64)
    mad = np.median(np.abs(rows - center), axis=0)
    scale = np.maximum(mad * _MAD_TO_SIGMA, _SCALE_FLOOR).astype(np.float64)
    return center, scale


def fit_session_stats(
    features: np.ndarray,
    valid_mask: np.ndarray,
    grid: WindowGrid,
    *,
    norm_minutes: float,
) -> SessionStats:
    """Robust per-column stats from the first *norm_minutes* of a run's VALID
    windows -- the label-free, deployment-realistic estimator (spec D3): the first
    minutes of a new day are observable without any labels or SCADA, so these
    stats are exactly what a deployed monitor could compute before scoring.

    Membership is by window START: window `i` qualifies iff its grid start offset
    `i * grid.window_ns` is strictly below `norm_minutes * 60 * 1e9` ns AND
    `valid_mask[i]` is True.

    **State-mix confound (A3.5, documented caveat).** The first N minutes of a run
    can be -- and measurably are -- a single operating state: 290626's first 20
    minutes are 100% one state. The stats are therefore state-composition-
    dependent: a run that opens in turbine operation yields different center/scale
    than one opening at standstill, and that difference propagates into every
    normalized score. The N-sweep over `norm_minutes` in {5, 20, 60} (spec A2.2)
    is the designated sensitivity probe for this confound, and the rolling-
    recalibration caveat's sibling applies too: if the first minutes contain a
    fault, normalization absorbs part of it.

    Args:
        features: (W, F) per-window feature matrix (a `PreparedRun.features`).
        valid_mask: (W,) bool validity mask aligned with *features*.
        grid: The window grid *features* is aligned against (start offsets).
        norm_minutes: Prefix length in minutes; must be > 0 (`0.0` is reserved as
            `fit_pool_stats`' pool-global sentinel, module docstring).

    Returns:
        A `SessionStats` with `n_windows` = the number of qualifying rows and
        `norm_minutes` echoed as passed.

    Raises:
        ValueError: if `norm_minutes <= 0`; if *features*/*valid_mask*/*grid*
            disagree on the window count; or if ZERO windows qualify (the whole
            prefix invalid, or the run starts after it) -- stats from nothing
            would be a silent lie, and the caller must know this run/prefix
            combination cannot be normalized.
    """
    if not norm_minutes > 0.0:
        raise ValueError(
            f"norm_minutes must be > 0, got {norm_minutes!r} (0.0 is reserved as "
            f"fit_pool_stats' pool-global sentinel and never a valid prefix length)"
        )
    n_windows = features.shape[0]
    if valid_mask.shape[0] != n_windows or grid.n_windows != n_windows:
        raise ValueError(
            f"features/valid_mask/grid disagree on the window count: "
            f"features has {n_windows} row(s), valid_mask {valid_mask.shape[0]}, "
            f"grid {grid.n_windows}"
        )
    cutoff_ns = int(round(norm_minutes * 60.0 * 1e9))
    offsets = np.arange(n_windows, dtype=np.int64) * np.int64(grid.window_ns)
    qualifying = (offsets < cutoff_ns) & valid_mask
    rows = features[qualifying]
    if rows.shape[0] == 0:
        raise ValueError(
            f"zero valid windows start within the first {norm_minutes:g} minute(s) "
            f"of the run -- cannot fit session stats (prefix all-invalid or shorter "
            f"than one window)"
        )
    center, scale = _center_scale(np.asarray(rows, dtype=np.float64))
    return SessionStats(
        center=center,
        scale=scale,
        n_windows=int(rows.shape[0]),
        norm_minutes=float(norm_minutes),
    )


def fit_pool_stats(rows: np.ndarray) -> SessionStats:
    """Pool-global robust stats over an already-stacked matrix -- the pooled-
    snapshot variant (module docstring): a pooled artifact has no single fit day,
    so its reference-side stats are the center/scale of the RAW pooled fit matrix
    as a whole, marked with the `norm_minutes == 0.0` sentinel. The monitor uses
    them exactly like fit-day stats for the reference-side transform and falls
    back to its default first-N prefix for the monitored run's own stats.

    Args:
        rows: (N, F) stacked feature rows (e.g. `PoolResult.features`).

    Returns:
        A `SessionStats` with `n_windows == N` and `norm_minutes == 0.0`.

    Raises:
        ValueError: if *rows* has zero rows, or contains non-finite values (a
            pooled fit matrix is finite by `build_pool`'s own assertion -- a
            non-finite input here means the caller bypassed it).
    """
    if rows.shape[0] == 0:
        raise ValueError("fit_pool_stats: zero rows -- cannot fit pool-global stats")
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(
            "fit_pool_stats: rows contain non-finite values -- pool-global stats "
            "require the finite pooled matrix build_pool guarantees"
        )
    center, scale = _center_scale(matrix)
    return SessionStats(
        center=center, scale=scale, n_windows=int(matrix.shape[0]), norm_minutes=0.0
    )


def apply_session_norm(features: np.ndarray, stats: SessionStats) -> np.ndarray:
    """Map *features* into the session-normalized space: `(X - center) / scale`,
    as a fresh float64 array (never a view -- callers keep their raw matrix for
    the detector, A3.5). NaN rows (invalid windows) pass through as NaN.

    Raises:
        ValueError: if the feature width does not match the stats' width (the
            same loud-geometry posture as the snapshot's feature_names guard).
    """
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != stats.center.shape[0]:
        raise ValueError(
            f"apply_session_norm: features must be 2-D with "
            f"{stats.center.shape[0]} column(s) (the stats' width), got shape "
            f"{arr.shape}"
        )
    result: np.ndarray = (arr - stats.center) / stats.scale
    return result
