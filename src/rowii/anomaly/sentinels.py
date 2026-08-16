"""Two label-free drift sentinels for the once-calibrated replay. Both fire on
a monitored day using thresholds derived
ONLY from the commissioning (B1) CONFORMAL side, so both are label-free at
runtime. s1 reuses the mode bank's `no_mode_fits` rate; s2 is a per-stream
level-step on the RAW mic caches (fusion's z-scored columns are excluded
upstream -- callers read the raw `audio`/`vibration` caches for s2, never
fusion's stored columns). The sentinel idea echoes the partner's drift
monitoring (Rodrigues & Zhang, 2026); every number here is computed from OUR
caches -- no partner constant is read or asserted.

**Firewall.** `97.5` (the bootstrap percentile), `1000` (the
bootstrap replicate count `B`), and `3` (the `s2_fires` sigma-equivalent factor
`k`) are named, derived-from-nothing-partner-published standard-statistics
constants -- never asserted against a partner figure anywhere in this module or
its tests. The firing decision lives entirely in the stored log10 level domain;
a dB conversion (`analyze_days._level_db_factor`) is a DRIVER-side reporting
nicety only, never the firing criterion -- this module stays dB-free
and never imports a sibling script.

**s2 MAD scaling (pinned).** `s2_anchor_mad`'s `mad` is the
`_MAD_TO_SIGMA`-scaled (`1.4826x`) MAD over the per-`segment_ids`-block medians,
the SAME normal-consistency precedent `rowii.anomaly.normalize`'s
`SessionNormalizer` uses (`_center_scale`: `MAD * 1.4826` estimates sigma for
Gaussian data). Scaling the MAD makes `k * mad` in `s2_fires` read as the
standard k-sigma-equivalent robust criterion rather than a bare multiple of the
raw MAD -- resolving an open RAW-vs-scaled question flagged during design.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from rowii.anomaly.levelrecal import level_columns
from rowii.pipeline import stream_columns

logger = logging.getLogger(__name__)

_MAD_TO_SIGMA = 1.4826
"""Normal-consistency MAD->sigma factor -- the SAME named constant and precedent
as `rowii.anomaly.normalize._MAD_TO_SIGMA` (`SessionNormalizer`,
`fit_session_stats`/`_center_scale`): for Gaussian data, `MAD * 1.4826` estimates
sigma. Applied in `s2_anchor_mad` so `s2_fires`'s `k * mad` reads as the standard
k-sigma-equivalent robust criterion."""

_SCALE_FLOOR = 1e-8
"""The house divide-by-zero floor -- the SAME named constant, value, and
precedent as `rowii.anomaly.normalize._SCALE_FLOOR` (`_center_scale`: `MAD *
1.4826` floored at `1e-8` so a degenerate/constant estimation window never
yields a raw zero scale). Applied in `s2_anchor_mad` so a
single-block (or otherwise MAD-degenerate) commissioning anchor can never
yield `mad == 0.0`, which would make `s2_fires`'s `k * mad` margin zero and
turn the sentinel into a hair-trigger that fires on any nonzero deviation."""


def level_series(rows: np.ndarray, feature_names: list[str], streams: Sequence[str]) -> np.ndarray:
    """`(W,)` per-window mean of the intersection of *streams*' columns and the
    LEVEL columns of *feature_names* (the `analyze_days._levels_by_stream` rule,
    reimplemented here in `src/` since scripts never lend their internals to a
    module -- `rowii.pipeline.stream_columns` ∩ `rowii.anomaly.levelrecal.
    level_columns`). A stream absent from *feature_names* is skipped
    (`stream_columns`'s `ValueError`, caught) rather than raising -- s2 reads
    whichever of the two mic (or two vibration) streams are actually present in
    the RAW cache; a PARTIAL absence (some, not all, of *streams* missing) is
    logged via `logger.warning` naming the skipped stream(s), never fully
    silent; only a total absence of stream∩level columns is
    an error (see Raises).

    Args:
        rows: `(W, F)` feature matrix, `F == len(feature_names)` -- pass VALID
            rows only (the `column_medians` convention); this function does not
            itself filter by `valid_mask`. Enforced (mirroring
            `rowii.anomaly.levelrecal.column_medians`'s own geometry posture):
            see Raises.
        feature_names: Column names aligned with `rows`' columns.
        streams: Stream name(s) to average over (e.g. `("RAWGeneratorMic__0",
            "RAWTurbineMic__1")` for s2's mic-level series, or a single
            `("RAWGeneratorVib__2",)` for the vibration cross-check).

    Returns:
        `(W,)` float64 -- per-row mean over the sorted, de-duplicated column
        indices that are BOTH in one of `streams`' blocks AND a level column.

    Raises:
        ValueError: the stream∩level intersection is empty across ALL of
            *streams* -- an embedding variant (`level_columns` returns `[]`), or
            none of *streams* is present in *feature_names* at all. s2 must read
            a RAW mic/vibration cache, never a representation where this is
            structurally impossible. Also raised when `rows` is not 2-D
            with `len(feature_names)` columns (loud geometry, the levelrecal
            posture) -- a shape mismatch must never silently mis-slice.
    """
    level = set(level_columns(feature_names))
    cols: list[int] = []
    skipped: list[str] = []
    for stream in streams:
        try:
            cols.extend(int(c) for c in stream_columns(feature_names, stream) if int(c) in level)
        except ValueError:
            skipped.append(stream)
    if skipped and len(skipped) < len(streams):
        # PARTIAL absence only -- a total absence falls through to the loud
        # ValueError below instead (never both).
        logger.warning(
            "level_series: stream(s) %s not present in feature_names -- skipped "
            "(reading whichever of streams=%s are actually present, A1.1)",
            skipped, list(streams),
        )
    if not cols:
        raise ValueError(
            f"level_series: no stream∩level column found across streams={list(streams)!r} "
            "(an embedding variant, or none of these streams is present in feature_names) "
            "-- s2 must read a RAW mic/vibration cache (A1.1)"
        )
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(feature_names):
        raise ValueError(
            f"level_series: rows must be 2-D with {len(feature_names)} column(s) "
            f"(len(feature_names)), got shape {arr.shape}"
        )
    return arr[:, sorted(set(cols))].mean(axis=1)


def _block_medians(values: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    """Per-`segment_ids`-block median of *values*, one entry per distinct block
    (in `np.unique` order) -- `s2_anchor_mad`'s `m`, the per-block medians the MAD
    is computed over."""
    v = np.asarray(values, dtype=np.float64)
    return np.array([float(np.median(v[segment_ids == s])) for s in np.unique(segment_ids)])


def _bootstrap_rate_pct(
    values: np.ndarray, segment_ids: np.ndarray, pct: float, n_boot: int, seed: int
) -> float:
    """The *pct*-th percentile of *n_boot* `segment_ids`-block bootstrap resamples
    of `mean(values)` -- s1's threshold statistic (blocks = `segment_ids`,
    NEVER wall-clock, the SAME rule `scripts/analyze_days.py::_block_bootstrap_ci`
    uses for its own feature-stability CI, reimplemented here since `src/` never
    imports a script). Each replicate draws `len(groups)` blocks WITH replacement
    from the `len(groups)` distinct blocks and pools them before taking the mean
    -- resampling blocks, not individual rows, is what block bootstrap means; a
    single-block input is therefore degenerate (every replicate resamples that
    one block), matching `_block_bootstrap_ci`'s documented single-segment
    behaviour."""
    rng = np.random.default_rng(seed)
    groups = [np.asarray(values)[segment_ids == s] for s in np.unique(segment_ids)]
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        boots[b] = float(np.mean(np.concatenate([groups[i] for i in pick])))
    return float(np.percentile(boots, pct))


def s1_threshold(
    no_mode_fits: np.ndarray, segment_ids: np.ndarray, *, n_boot: int = 1000, seed: int = 7
) -> float:
    """The s1 firing threshold: the 97.5th percentile of `n_boot=1000`
    `segment_ids`-block bootstrap resamples of `mean(no_mode_fits)` on the B1
    CONFORMAL side, seeded `rng(seed=7)`. `97.5`/`1000` are named,
    derived-from-nothing-partner-published standard-statistics constants (the
    firewall) -- never asserted against a partner figure."""
    return _bootstrap_rate_pct(
        np.asarray(no_mode_fits, dtype=np.float64), np.asarray(segment_ids), 97.5, n_boot, seed
    )


def s1_fires(day_rate: float, threshold: float) -> bool:
    """`True` iff a monitored day's `no_mode_fits` rate exceeds `s1_threshold`'s
    B1-derived band (strict `>`, so the threshold value itself never fires)."""
    return bool(day_rate > threshold)


def s2_anchor_mad(level_values: np.ndarray, segment_ids: np.ndarray) -> tuple[float, float]:
    """The s2 anchor/band: `anchor = median(level_values)`; `mad =
    max(1.4826 * median(|m - median(m)|), 1e-8)` over the per-`segment_ids`-block
    medians `m` (`_block_medians`) -- the SAME `_MAD_TO_SIGMA` normal-consistency
    scaling AND the SAME `1e-8` divide-by-zero floor (`_SCALE_FLOOR`) as
    `rowii.anomaly.normalize`'s `SessionNormalizer` (`_center_scale`), so `k *
    mad` in `s2_fires` reads as the standard k-sigma-equivalent robust criterion
    rather than a bare multiple of the raw MAD, and a degenerate (e.g.
    single-block) anchor can never collapse that margin to zero and hair-trigger
    on any nonzero deviation. Both statistics are computed on the B1 CONFORMAL
    side (same held-out side as s1, same anchor discipline) -- the caller's
    responsibility, this function is agnostic to which side its inputs came
    from."""
    v = np.asarray(level_values, dtype=np.float64)
    block_med = _block_medians(v, np.asarray(segment_ids))
    raw_mad = float(np.median(np.abs(block_med - np.median(block_med))))
    return float(np.median(v)), max(_MAD_TO_SIGMA * raw_mad, _SCALE_FLOOR)


def s2_fires(day_median: float, anchor: float, mad: float, *, k: float = 3.0) -> bool:
    """`True` iff `|day_median - anchor| > k * mad` (`k=3.0`, the
    standard robust-outlier criterion, derived from nothing partner-published --
    NOT a fixed-dB-headroom constant, which does not exist in this
    module)."""
    return bool(abs(day_median - anchor) > k * mad)


def s2_attribution(mic_fires: bool, vib_fires: bool) -> str:
    """`"instrumentation"` when the mic sentinel fires but the RAWGeneratorVib__2
    cross-check does not (mic-steps-vib-flat signature);
    `"machine"` otherwise (both fire, or mic alone is silent). Labels the
    trigger's CAUSE -- the overall s2 verdict is `mic_fires` alone; this
    function never vetoes it."""
    return "instrumentation" if (mic_fires and not vib_fires) else "machine"
