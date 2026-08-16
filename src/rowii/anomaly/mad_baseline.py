"""Hand-set-threshold "MAD baseline": a median-plus-k-MAD band-energy detector.

**Purpose.** Quantify, in OUR OWN harness on OUR OWN data, what a field-standard
fixed-threshold practice achieves, so the thesis can compare it against the
calibrated (conformal, mode-conditioned) system in one table
(`scripts/run_mad_baseline.py`'s own module docstring has the full experiment
design). This module holds only the PURE math: the commissioning statistics, the
k <-> threshold <-> flag-rate triangle, and the log-mel -> high-band-score
derivation. No I/O, no run discovery, no GT loading -- see the CLI script for
orchestration.

**Attribution (TYPE, not COPY).** The method type -- a global median + k * MAD
band-energy threshold over a high-frequency microphone band -- is inspired by
the partner project's own transient-detection practice ("Bruno's transient
detector (band-MAD)"; co-authored account:
Zhang, Krummenacher et al., "Multi-Modal Acoustic-Vibration Anomaly Detection in
Pumped-Storage Turbines", Viennahydro 2026 draft). This module is an INDEPENDENT
implementation: every threshold, every score, and every number it produces is
computed from THIS repo's own caches -- no partner code, constant, or number is
read or asserted anywhere here (the same firewall `rowii.anomaly.sentinels`
states for its own partner-inspired drift sentinels).

**MAD scaling (pinned, consistent with the rest of this package).** `median_mad`
returns `(median, 1.4826 * MAD)` -- the SAME `_MAD_TO_SIGMA` normal-consistency
scaling and `1e-8` divide-by-zero floor `rowii.anomaly.sentinels.s2_anchor_mad`
uses for its own level sentinel. This baseline's threshold is the MODE-AGNOSTIC,
NON-BLOCK, one-sided analogue: `median + k * mad` over every valid window of the
commissioning pool directly (no segment-block bootstrap, no per-mode
conditioning) -- deliberately simple, matching what a practitioner sets by hand
at commissioning time, not the conformal/sentinel machinery elsewhere in this
package.

**Score: log high-band (5-20 kHz) microphone energy.** No existing handcrafted
`rowii.signals.features.AudioFeaturizer` column matches a 5-20 kHz band cleanly:
its highest octave band (`ch{N}_octave_8000`) only spans ~5.66-11.3 kHz (one
octave, capped by the fixed `MACHINE_HZ`/octave-center list). This module
instead derives the score from the CACHED `"logmel"` variant
(`rowii.signals.logmel.LogmelFeaturizer`, primary generator-mic stream only,
its own size-bound design decision, reused as-is): `high_band_mel_bins` selects
the mel filterbank's columns whose CENTER frequency falls in `[5000, 20000)` Hz
(24 of 64 bins at the plant's real 50 kHz mic rate, centers ~5.12-19.92 kHz --
the closest achievable match to a literal 5-20 kHz band from this
representation); `band_energy_score` undoes the cache's own
`log10(power + floor)` encoding, sums those bins per frame (band power), and
averages over frames -- a whole-window POWER estimate, mirroring
`AudioFeaturizer`'s own `log_rms` (a whole-window average, never a peak/max
statistic) -- before re-applying `log10(... + floor)` for the final per-window
score. Caveat restated in the CLI's `summary.md`: this is a WINDOW-level
analogue (1-s aggregate energy); the partner's own detector runs on raw impulse
resolution, so a fair comparison is scoped to what a 1-s-window monitoring
harness can see, not sub-window transient shape.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

_MAD_TO_SIGMA = 1.4826
"""Normal-consistency MAD->sigma factor -- the SAME named constant and
precedent as `rowii.anomaly.sentinels._MAD_TO_SIGMA` /
`rowii.anomaly.normalize._MAD_TO_SIGMA`: for Gaussian data, `MAD * 1.4826`
estimates sigma."""

_SCALE_FLOOR = 1e-8
"""The house divide-by-zero floor -- the SAME named constant, value, and
precedent as `rowii.anomaly.sentinels._SCALE_FLOOR` /
`rowii.anomaly.normalize._SCALE_FLOOR`: a degenerate (e.g. constant) score
population can never yield `mad == 0.0`, which would make `k * mad` a
zero-margin hair-trigger."""

_LOG_FLOOR = 1e-12
"""Log10 floor for the final band-energy score -- matches
`rowii.signals.features._LOG_FLOOR` (this module's score lives in the same
"handcrafted audio level feature" family, just derived from the log-mel cache
instead of a dedicated Welch PSD pass)."""

_BAND_LO_HZ = 5000.0
_BAND_HI_HZ = 20000.0
"""The target high band (module docstring): `[_BAND_LO_HZ, _BAND_HI_HZ)` Hz."""

_LOGMEL_FMIN_HZ = 20.0
"""`rowii.signals.logmel.LogmelFeaturizer`'s own default `fmin_hz` -- mirrored
here (not imported: a single float literal, the module-sibling-duplication
precedent this repo already uses for small, stable constants, e.g.
`rowii.anomaly.sentinels._MIC_STREAMS`) so `high_band_mel_bins` reproduces the
EXACT mel filterbank geometry the cache was built with."""

_LOGMEL_PRIMARY_STREAM = "RAWGeneratorMic__0"
"""The one stream `rowii.pipeline`'s `"logmel"` variant ever featurizes (its
own private `_LOGMEL_STREAMS` constant -- primary/generator mic only,
size bound) -- duplicated here since `rowii.pipeline` does not export
it (same module-sibling-duplication precedent as `_LOGMEL_FMIN_HZ`)."""

_LOGMEL_LOCAL_NAME_RE = re.compile(r"^logmel_f(\d+)_m(\d+)$")


# ---------------------------------------------------------------------------
# Commissioning statistics + the k <-> threshold <-> flag-rate triangle
# ---------------------------------------------------------------------------


def median_mad(scores: np.ndarray) -> tuple[float, float]:
    """`(median, scaled MAD)` of *scores* -- the commissioning statistics this
    baseline's threshold is built from (module docstring: `1.4826 *
    median(|x - median(x)|)`, floored at `1e-8`).

    Args:
        scores: `(N,)` float array, the commissioning-pool per-window scores
            (VALID windows only is the caller's responsibility).

    Returns:
        `(median, mad)`, both Python floats; `mad` is already 1.4826-scaled
        and floor-protected -- pass it straight to `threshold_from_k`.

    Raises:
        ValueError: *scores* is empty.
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.shape[0] == 0:
        raise ValueError("median_mad: scores must be non-empty")
    median = float(np.median(x))
    raw_mad = float(np.median(np.abs(x - median)))
    return median, max(_MAD_TO_SIGMA * raw_mad, _SCALE_FLOOR)


def threshold_from_k(median: float, mad: float, k: float) -> float:
    """`median + k * mad` -- the MAD baseline's threshold (module docstring).
    *mad* is already 1.4826-scaled (`median_mad`'s own return contract), so
    this is literally `median + k * 1.4826 * MAD_raw`."""
    return float(median + k * mad)


def flag_rate(scores: np.ndarray, threshold: float) -> float:
    """Fraction of *scores* STRICTLY greater than *threshold* -- the "flagged"
    convention this whole module uses (mirrors `rowii.anomaly.sentinels.
    s2_fires`'s own strict `>`, so the threshold value itself never flags).

    Raises:
        ValueError: *scores* is empty.
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.shape[0] == 0:
        raise ValueError("flag_rate: scores must be non-empty")
    return float(np.mean(x > threshold))


def k_for_target_rate(
    scores: np.ndarray, median: float, mad: float, target_rate: float
) -> tuple[float, float]:
    """The `k` whose `threshold_from_k(median, mad, k)` flags as close to
    `target_rate` of *scores* as the data's own order statistics allow --
    `k_1pct`'s derivation ("the k that yields exactly 1 percent flagged
    windows on the commissioning pool").

    Flagging (`flag_rate`, strict `>`) is a non-increasing step function of
    the threshold: the achievable flag COUNTS are exactly `{0, 1, ..., N}`
    where `N = len(scores)`, so `target_n = round(target_rate * N)` is
    realized EXACTLY (for data with no tie straddling the cut, the common
    case for a continuous acoustic-energy score) by placing the threshold at
    the `target_n`-th largest score: every one of the `target_n` strictly
    larger scores is flagged, that score itself and everything smaller is
    not. No numerical root-finding is needed.

    Args:
        scores: `(N,)` float array, the SAME commissioning-pool population
            *median*/*mad* were derived from (`median_mad`).
        median: `median_mad`'s first return value.
        mad: `median_mad`'s second return value (already 1.4826-scaled).
        target_rate: Target flagged fraction, in `[0, 1]`.

    Returns:
        `(k, realized_rate)` -- `realized_rate` is `flag_rate` evaluated on
        `threshold_from_k(median, mad, k)`, i.e. the TRUE outcome of applying
        the returned `k` (never the caller's bare request) -- identical to
        `target_rate` unless *scores* has an exact tie straddling the cut
        (an aspirational-but-checked "exactly", never a silent approximation).

    Raises:
        ValueError: *scores* is empty, or *target_rate* is outside `[0, 1]`.
    """
    x = np.asarray(scores, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        raise ValueError("k_for_target_rate: scores must be non-empty")
    if not (0.0 <= target_rate <= 1.0):
        raise ValueError(
            f"k_for_target_rate: target_rate must be in [0, 1], got {target_rate!r}"
        )

    target_n = int(round(target_rate * n))
    sorted_desc = np.sort(x)[::-1]
    if target_n <= 0 or target_n >= n:
        # A boundary target (0% or 100%) needs a threshold strictly outside
        # the data range -- but the eventual k = (threshold - median) / mad ->
        # threshold_from_k(median, mad, k) round-trip only preserves absolute
        # precision down to about `max(|median|, mad) * 2e-16` (float64
        # rounding), so an absolute epsilon (e.g. `np.nextafter`) anchored at
        # the data itself can silently collapse back to the boundary value
        # once re-expressed through k. `eps` is instead scaled to the larger
        # of the data's own magnitude and `mad`, comfortably above that
        # rounding floor yet negligible next to the data.
        scale = max(abs(median), mad, float(np.abs(x).max()), 1.0)
        eps = scale * 1e-9
        threshold = (
            float(sorted_desc[0]) + eps if target_n <= 0  # above the max -> 0 flagged
            else float(sorted_desc[-1]) - eps               # below the min -> n flagged
        )
    else:
        # The target_n-th largest score: exactly target_n scores are STRICTLY
        # greater than it (assuming no tie at this exact boundary) -- placed
        # EXACTLY, no epsilon, since this is a genuine order statistic of the
        # data, not an out-of-range sentinel.
        threshold = float(sorted_desc[target_n])

    k = (threshold - median) / mad
    realized = flag_rate(x, threshold_from_k(median, mad, k))
    return float(k), float(realized)


# ---------------------------------------------------------------------------
# Log-mel -> high-band score
# ---------------------------------------------------------------------------


def logmel_geometry(feature_names: Sequence[str]) -> tuple[int, int]:
    """`(n_frames, n_mels)` parsed from a `"logmel"`-variant `PreparedRun`'s
    own `feature_names` (`"RAWGeneratorMic__0::logmel_f{f}_m{m}"`,
    `rowii.pipeline._assemble_feature_names`'s stream-prefix convention over
    `rowii.signals.logmel.LogmelFeaturizer.feature_names`'s frame-major
    contract) -- derived from the run's OWN cached columns rather than a
    hardcoded 49x64 pair, so a future window-length/mel-count/sample-rate
    change is caught here instead of silently mis-reshaping downstream.

    Args:
        feature_names: A `"logmel"`-variant `PreparedRun.feature_names` list,
            in column order.

    Returns:
        `(n_frames, n_mels)`.

    Raises:
        ValueError: *feature_names* is empty; any name does not start with
            `"RAWGeneratorMic__0::"` or does not match
            `"logmel_f<f>_m<m>"` after that prefix; or the parsed `(f, m)`
            pairs do not form a COMPLETE, frame-major
            `range(n_frames) x range(n_mels)` grid in that exact order --
            this function refuses a partial or reordered column set rather
            than guessing a shape.
    """
    if not feature_names:
        raise ValueError("logmel_geometry: feature_names must be non-empty")

    prefix = f"{_LOGMEL_PRIMARY_STREAM}::"
    pairs: list[tuple[int, int]] = []
    for name in feature_names:
        if not name.startswith(prefix):
            raise ValueError(
                f"logmel_geometry: {name!r} does not start with {prefix!r} -- "
                f"expected the 'logmel' variant's own single-stream "
                f"({_LOGMEL_PRIMARY_STREAM!r}) column contract"
            )
        match = _LOGMEL_LOCAL_NAME_RE.match(name[len(prefix) :])
        if match is None:
            raise ValueError(
                f"logmel_geometry: {name!r} does not match "
                f"'{prefix}logmel_f<f>_m<m>'"
            )
        pairs.append((int(match.group(1)), int(match.group(2))))

    n_frames = max(f for f, _ in pairs) + 1
    n_mels = max(m for _, m in pairs) + 1
    expected = [(f, m) for f in range(n_frames) for m in range(n_mels)]
    if pairs != expected:
        raise ValueError(
            f"logmel_geometry: feature_names are not a complete frame-major "
            f"{n_frames}x{n_mels} grid (parsed {len(pairs)} column(s), "
            f"expected {len(expected)} in 'for f in range(n_frames) for m in "
            f"range(n_mels)' order)"
        )
    return n_frames, n_mels


def _mel(hz: float) -> float:
    """Mirrors `rowii.signals.logmel._mel` (private, module-sibling
    duplication -- a 1-line HTK mel-scale formula, not worth a cross-module
    import of a leading-underscore name)."""
    return float(2595.0 * np.log10(1.0 + hz / 700.0))


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Mirrors `rowii.signals.logmel._mel_to_hz` (see `_mel`'s docstring)."""
    return np.asarray(700.0 * (10.0 ** (mel / 2595.0) - 1.0), dtype=np.float64)


def high_band_mel_bins(
    n_mels: int,
    rate_hz: float,
    *,
    lo_hz: float = _BAND_LO_HZ,
    hi_hz: float = _BAND_HI_HZ,
    fmin_hz: float = _LOGMEL_FMIN_HZ,
) -> np.ndarray:
    """Ascending indices of the `LogmelFeaturizer` mel bins whose CENTER
    frequency falls in `[lo_hz, hi_hz)` -- reproduces the exact mel
    filterbank geometry `rowii.signals.logmel._mel_filterbank` builds
    (`_mel`/`_mel_to_hz` above, mirrored from that module): `n_mels + 2`
    edges evenly spaced in MEL scale between `mel(fmin_hz)` and
    `mel(rate_hz / 2)`; filter *m*'s center is edge `m + 1`.

    At the plant's real geometry (50 kHz mic rate, 64 mels, `fmin_hz=20`)
    this resolves to 24 bins spanning ~5.12-19.92 kHz -- the closest
    achievable match to a literal 5-20 kHz band from this representation
    (module docstring).

    Args:
        n_mels: Mel-bin count of the log-mel geometry (`logmel_geometry`).
        rate_hz: The stream's sample rate (determines `fmax_hz = rate_hz / 2`).
        lo_hz: Band lower edge, inclusive.
        hi_hz: Band upper edge, exclusive.
        fmin_hz: The filterbank's own lower edge (must match the geometry the
            cache was built with -- `LogmelFeaturizer`'s default).

    Returns:
        `(B,)` int64 ascending mel-bin indices, `0 <= index < n_mels`.

    Raises:
        ValueError: no mel bin's center falls in `[lo_hz, hi_hz)` at this
            `(n_mels, rate_hz, fmin_hz)` combination -- must fail loudly,
            never silently score on an empty band.
    """
    fmax_hz = rate_hz / 2.0
    mel_edges = np.linspace(_mel(fmin_hz), _mel(fmax_hz), n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)
    centers = hz_edges[1:-1]
    bins = np.flatnonzero((centers >= lo_hz) & (centers < hi_hz))
    if bins.size == 0:
        raise ValueError(
            f"high_band_mel_bins: no mel bin center falls in [{lo_hz}, {hi_hz}) "
            f"Hz at n_mels={n_mels}, rate_hz={rate_hz}, fmin_hz={fmin_hz} -- "
            f"cannot extract a high-band score from this log-mel geometry"
        )
    return np.asarray(bins, dtype=np.int64)


def band_energy_score(
    features: np.ndarray, n_frames: int, n_mels: int, target_bins: np.ndarray
) -> np.ndarray:
    """`(W,)` log10 high-band energy per window, derived from a cached
    `"logmel"`-variant feature matrix `(W, n_frames * n_mels)` (frame-major
    `logmel_f{f}_m{m}` columns, module docstring on why this baseline reads
    the log-mel cache rather than a single named handcrafted feature).

    `LogmelFeaturizer.transform` stores `log10(mel_power + 1e-10)` per
    `(frame, mel)` cell; this function undoes that (`10 ** cell`, the
    `1e-10` added back in is negligible for any non-silent window) to
    recover LINEAR mel power, sums the *target_bins* columns per frame (band
    power), averages over frames -- a whole-window POWER estimate, mirroring
    `AudioFeaturizer`'s own `log_rms` (a whole-window average, never a
    peak/max statistic) -- then re-applies `log10(... + 1e-12)` for the
    final per-window score.

    Args:
        features: `(W, n_frames * n_mels)` float64 -- one run's `"logmel"`-
            variant `PreparedRun.features` (or any row subset of it; VALID
            rows only is the caller's responsibility, this function scores
            every row it is given).
        n_frames: Frame count of the log-mel geometry (`logmel_geometry`).
        n_mels: Mel-bin count of the log-mel geometry (`logmel_geometry`).
        target_bins: Ascending mel-bin indices to sum (`high_band_mel_bins`).

    Returns:
        `(W,)` float64.

    Raises:
        ValueError: `features.shape[1] != n_frames * n_mels` (geometry
            mismatch) -- a caller must never silently mis-reshape.
    """
    x = np.asarray(features, dtype=np.float64)
    expected_width = n_frames * n_mels
    if x.ndim != 2 or x.shape[1] != expected_width:
        raise ValueError(
            f"band_energy_score: features has shape {x.shape}, expected "
            f"(W, n_frames * n_mels) = (W, {expected_width}) for "
            f"n_frames={n_frames}, n_mels={n_mels}"
        )
    patches = x.reshape(x.shape[0], n_frames, n_mels)
    linear = np.power(10.0, patches[:, :, target_bins])
    band_power_per_frame = linear.sum(axis=2)
    band_power = band_power_per_frame.mean(axis=1)
    return np.asarray(np.log10(band_power + _LOG_FLOOR), dtype=np.float64)
