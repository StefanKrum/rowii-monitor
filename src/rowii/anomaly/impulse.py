"""Band-energy impulse detector: candidate-kit register criterion #3.

**Why a third path.** The existing two candidate-kit paths (`scripts/candidate_
kit.py`'s own module docstring) are both LEARNED-EMBEDDING detectors scored on a
1 s window grid (fusion's sustained-episode path, audio-beats' single-window
transient path). Per-strike evaluation against the seconds-level 08.07 ground
truth (`research/notes/analysis_2026-08-15_perstrike_latency.md` §4) found a
documented complementarity: the two PU landmark strike events (07/13) carry ZERO
marks because the annotator could not locate them under pump noise by ear, and
BOTH embedding paths also miss them -- yet a raw 5-20 kHz band-energy search on
the SAME audio, at a coarser but purpose-built sub-window resolution, recovers
the annotator's own ST-landmark triplets almost exactly (z = 7-17, offsets
0.1-0.3 s) even though it still finds no PU-landmark triplet (three independent
methods -- annotator, embeddings, band-energy search -- agree those two strikes
sit below the airborne noise floor under pump operation; an honest scope
boundary, not a detector failure). The embedding paths are scored once per 1 s
window and therefore structurally cannot resolve two impulses closer together
than about a window width; this module operates at 5 ms resolution and finds
exactly the sub-window impulses the other two paths cannot see by construction.
This is the validated approach (session-scratchpad exploration, `knack_own_
search.py`/`find_pu_landmarks.py`, 2026-08-15/16), ported here as a proper,
tested module.

**Method.** Per mic stream: split the (native-rate, ~50 kHz) audio into 10 ms
frames (`FRAME_S`) hopped every 5 ms (`HOP_S`, 50% overlap), Hann-windowed;
`band_frame_energy` sums the FFT power spectrum over bins whose frequency falls
in `[BAND_LO_HZ, BAND_HI_HZ)` = [5, 20) kHz per frame and returns the log10 of
that sum (`loge`) -- a literal band-energy measurement, unlike `rowii.anomaly.
mad_baseline`'s own high-band score (that module derives an approximate 1 s
window-level analogue from the CACHED log-mel representation for a fair
apples-to-apples comparison against a monitoring harness; this module computes
the band energy directly from raw samples at native sub-window resolution,
exactly because resolving SUB-window impulses is the entire point of this third
path). `mad_z_score` then rolling-median-detrends `loge` over a 1 s window
(`MED_WIN_S`, so a genuine but slow level change -- a load ramp, a mode change --
is tracked out rather than mistaken for an impulse) and rescales the residual to
a MAD z-score (`_MAD_TO_SIGMA = 1.4826`, the SAME normal-consistency scaling
`rowii.anomaly.sentinels`/`rowii.anomaly.normalize`/`rowii.anomaly.mad_baseline`
already use). `pick_peaks` greedily keeps the most extreme frame first and
suppresses any other candidate within `MIN_SEP_S` = 0.25 s of an already-kept
peak (so one physical impulse, whose energy typically spans several consecutive
10 ms frames, yields one peak, not a cluster) -- the same "most extreme first,
one claim per neighbourhood" idea as `scripts/candidate_kit.py`'s own
`_greedy_suppress`, just operating on a 1-D frame axis instead of a 2-D time
span. `detect_impulses` composes all three steps.

**Threshold (`Z_REGISTER_THRESHOLD` = 6.0), principled and validated.** Run
against the ST-landmark strikes (`docs/groundtruth/080726_strikes_seconds_st.
csv`, the seconds-level per-strike ground truth an independent human annotator
produced by ear) -- see `tests/test_candidate_kit.py`'s own `@pytest.mark.data`
validation test, which re-derives these numbers directly from real data rather
than trusting a one-off exploratory claim -- this method recovers the
annotator's own marked triplets at z approx 7-17, within 0.1-0.3 s. Background
(non-impulse) frames across the same clips sit at z approx 2-5. A threshold of
6.0 therefore sits COMFORTABLY below the validated detection range (>= 7) while
staying clear of the observed background ceiling (<= 5) -- conservative in the
sense that asks BOTH "would it have caught every validated strike" (yes, all
land at z >= 7) AND "would it reject ordinary background" (yes, background
tops out around z = 5), rather than splitting the difference. It is
deliberately more conservative than the z = 4.0 floor the exploratory
session-scratchpad search used while still hunting for a first candidate signal
in a 5-20 minute window (an exploration floor, never intended as a register
threshold): z = 4.0 is well inside the observed background range and would
flood the register with sensor/background noise over full multi-hour sessions.
`Z_REGISTER_THRESHOLD` is a module-level default, not a hardcoded literal
inside `pick_peaks`/`detect_impulses` -- every call site takes `z_min` as a
keyword argument (module docstring's "configurable").

**Both-mic coincidence (module-external, `scripts/candidate_kit.py`).** This
module deliberately stops at "peaks on ONE stream" -- pairing peaks across the
generator/turbine mic streams into register-eligible coincident events is
`scripts/candidate_kit.py`'s own `match_coincident_peaks`/`build_impulse_pairs`
job (needs `datetime`/session bookkeeping this module intentionally has no
knowledge of; this module stays a pure, offset-seconds-only signal-processing
library, testable with plain numpy arrays, mirroring `rowii.anomaly.mad_
baseline`'s own no-I/O, no-datetime convention). The RATIONALE for requiring
both-mic coincidence lives there too, but restated briefly: a genuine
plant-wide mechanical impulse (a strike, a click) radiates through the machine
and reaches BOTH microphones at essentially the same instant, while a
sensor-local artifact (a cable knock, a connector transient, electrical
pickup at one preamp) shows up on only one channel -- exactly the pattern the
confirmed, human-annotated strikes exhibit (both mics, offsets 0.1-0.3 s of
each other) and exactly what separates them from a single-channel outlier.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter

BAND_LO_HZ = 5000.0
BAND_HI_HZ = 20000.0
"""The target band, module docstring: `[BAND_LO_HZ, BAND_HI_HZ)` Hz -- a
LITERAL 5-20 kHz band computed directly from raw samples (unlike `rowii.
anomaly.mad_baseline.high_band_mel_bins`'s log-mel-filterbank approximation of
the same nominal band; see this module's own docstring for why the two differ
on purpose)."""

FRAME_S = 0.010
"""10 ms analysis frame (module docstring) -- short enough that a millisecond-
scale broadband impulse does not smear across a wide window, mirroring
`scripts/annotation_kit.py`'s own `_SPEC_NPERSEG` rationale for the same class
of signal at a coarser (2.56 ms) resolution."""
HOP_S = 0.005
"""5 ms hop (50% overlap at `FRAME_S` = 10 ms) -- dense enough that an
impulse's energy is never split across two under-lapping frames."""
MED_WIN_S = 1.0
"""1 s rolling-median detrend window (module docstring) -- long enough to
track a genuine slow level change (a load ramp, a mode change) out without
itself reacting to a single impulse frame."""
MIN_SEP_S = 0.25
"""Minimum time separation `pick_peaks` enforces between two accepted peaks
(module docstring) -- one physical impulse should yield one peak, not a
cluster of adjacent frames all crossing the threshold together."""

Z_REGISTER_THRESHOLD = 6.0
"""Default register-inclusion threshold (module docstring: validated against
the ST-landmark strikes, z = 7-17 there vs. z <= 5 background) -- every
`z_min`-accepting function defaults to this, but it is always a keyword
argument, never hardcoded inline, so a caller (or a future re-validation) can
override it."""

_MAD_TO_SIGMA = 1.4826
"""Normal-consistency MAD -> sigma scaling -- the SAME named value and
precedent as `rowii.anomaly.sentinels._MAD_TO_SIGMA` / `rowii.anomaly.
normalize._MAD_TO_SIGMA` / `rowii.anomaly.mad_baseline._MAD_TO_SIGMA`."""
_MAD_FLOOR = 1e-12
"""Divide-by-zero floor for the MAD scale (module docstring's ported
algorithm) -- mirrors `rowii.anomaly.mad_baseline._LOG_FLOOR`'s value/role; a
degenerate (all-equal) residual population can never yield `mad == 0.0`."""
_ENERGY_FLOOR = 1e-20
"""Log10 floor for the raw band-power sum, applied BEFORE the log (ported
algorithm) -- prevents `log10(0)` for an exactly-silent frame; distinct from,
and much smaller than, `_MAD_FLOOR` (that one floors the z-score's own
denominator, this one floors the log argument two stages earlier)."""


@dataclass(frozen=True)
class ImpulsePeak:
    """One picked peak on a single mic stream, relative to the analysed clip's
    own sample 0 -- `scripts/candidate_kit.py` converts `time_offset_s` to an
    absolute UTC timestamp using the clip's own `StreamClip.covered_start_utc`
    (never the requested window start -- see that module's own docstring on
    why the two can legitimately differ)."""

    time_offset_s: float
    z: float


def frame_count(n_samples: int, frame_len: int, hop_len: int) -> int:
    """Number of frames `band_frame_energy` produces for *n_samples* input
    samples at *frame_len*/*hop_len* -- frame *i* (0-indexed) covers samples
    `[i * hop_len, i * hop_len + frame_len)`; the last frame that still fits
    entirely within *n_samples* determines the count. Faithfully mirrors the
    validated session-scratchpad search's own arithmetic (module docstring),
    just factored out and independently tested.

    Returns:
        `0` if *n_samples* is shorter than one whole frame.
    """
    if n_samples < frame_len:
        return 0
    return (n_samples - frame_len) // hop_len + 1


def band_frame_energy(
    samples: np.ndarray,
    rate_hz: float,
    *,
    lo_hz: float = BAND_LO_HZ,
    hi_hz: float = BAND_HI_HZ,
    frame_s: float = FRAME_S,
    hop_s: float = HOP_S,
) -> np.ndarray:
    """`(n_frames,)` log10 band-energy per frame -- Hann-windowed FFT power,
    summed over bins whose frequency falls in `[lo_hz, hi_hz)`, per `frame_s`-
    long frame hopped every `hop_s` (module docstring). Vectorized (one batched
    `np.fft.rfft` call over all frames via `sliding_window_view`, plus one
    boolean-mask reduction) rather than a per-frame Python loop -- numerically
    equivalent to that loop (same segment slicing, same window, same summed
    bins; `real**2 + imag**2` rather than `abs(...)**2` only skips a redundant
    sqrt-then-square round trip) but the vectorized form is what makes running
    this search over full multi-hour sessions (`scripts/candidate_kit.py`'s own
    chunked driver) tractable.

    Args:
        samples: `(N,)` raw audio samples, native rate, one channel.
        rate_hz: Sample rate of *samples*, Hz.
        lo_hz: Band lower edge, inclusive.
        hi_hz: Band upper edge, exclusive.
        frame_s: Frame length, seconds.
        hop_s: Hop length, seconds.

    Returns:
        `(frame_count(len(samples), frame_len, hop_len),)` float64 array.
        Empty if *samples* is shorter than one whole frame.

    Raises:
        ValueError: no FFT bin at this `(rate_hz, frame_s)` combination falls
            in `[lo_hz, hi_hz)` -- e.g. *rate_hz* is too low for the requested
            band (Nyquist below *lo_hz*). Must fail loudly, never silently
            score on an empty band.
    """
    x = np.asarray(samples, dtype=np.float64)
    frame_len = int(frame_s * rate_hz)
    hop_len = int(hop_s * rate_hz)
    n_frames = frame_count(x.shape[0], frame_len, hop_len)

    freqs = np.fft.rfftfreq(frame_len, d=1.0 / rate_hz)
    band_mask = (freqs >= lo_hz) & (freqs < hi_hz)
    if not band_mask.any():
        raise ValueError(
            f"band_frame_energy: no FFT bin falls in [{lo_hz}, {hi_hz}) Hz at "
            f"frame_len={frame_len} samples, rate_hz={rate_hz} -- cannot compute a "
            f"band-energy score for this geometry"
        )
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)

    window = np.hanning(frame_len)
    frames = sliding_window_view(x, frame_len)[::hop_len][:n_frames]
    windowed = frames * window
    spec = np.fft.rfft(windowed, axis=1)
    band_power = (spec.real**2 + spec.imag**2)[:, band_mask].sum(axis=1)
    return np.asarray(np.log10(band_power + _ENERGY_FLOOR), dtype=np.float64)


def mad_z_score(
    loge: np.ndarray, *, hop_s: float = HOP_S, med_win_s: float = MED_WIN_S
) -> np.ndarray:
    """`(n_frames,)` MAD z-score of *loge* after a rolling-median detrend
    (module docstring). `scipy.ndimage.median_filter(loge, size=mw,
    mode="nearest")` computes, for every index, the median over a window of
    `mw` samples CENTRED on it with edge-value replication past the array's own
    boundary -- exactly the ported algorithm's manual `np.pad(..., mode="edge")`
    + per-index windowed median, just vectorized (C-implemented, no per-index
    Python loop -- the same tractability motivation as `band_frame_energy`'s
    own vectorization).

    Args:
        loge: `(N,)` float array, one session/stream's own `band_frame_energy`
            output (or any per-frame log-energy series).
        hop_s: Seconds per frame -- converts *med_win_s* to a frame count.
        med_win_s: Rolling-median window length, seconds (module docstring:
            1 s default, `MED_WIN_S`).

    Returns:
        `(N,)` float64 z-score array; empty if *loge* is empty.
    """
    x = np.asarray(loge, dtype=np.float64)
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    mw = max(3, int(med_win_s / hop_s) | 1)
    med = median_filter(x, size=mw, mode="nearest")
    resid = x - med
    resid_median = float(np.median(resid))
    mad = float(np.median(np.abs(resid - resid_median))) * _MAD_TO_SIGMA + _MAD_FLOOR
    return np.asarray((resid - resid_median) / mad, dtype=np.float64)


def pick_peaks(
    z: np.ndarray,
    *,
    hop_s: float = HOP_S,
    min_sep_s: float = MIN_SEP_S,
    z_min: float = Z_REGISTER_THRESHOLD,
) -> list[ImpulsePeak]:
    """Peaks in *z* at or above *z_min*, greedily picked most-extreme-first
    with non-max suppression within *min_sep_s* of any already-picked peak
    (module docstring) -- visits frame indices in DESCENDING z order (so a
    strong peak always wins over a weaker one nearby, regardless of which
    comes first in time -- mirrors `scripts/candidate_kit.py`'s own
    `_greedy_suppress`, "most extreme first" over a different axis), and stops
    as soon as it reaches a frame below *z_min* (the remaining, lower-z frames
    can never qualify either). Faithfully mirrors the validated
    session-scratchpad search's own peak-picking arithmetic (module docstring).

    Args:
        z: `(N,)` MAD z-score array (`mad_z_score`'s own output shape).
        hop_s: Seconds per frame -- converts a frame index to
            `ImpulsePeak.time_offset_s` and *min_sep_s* to a frame count.
        min_sep_s: Minimum time separation between two accepted peaks,
            seconds (module docstring: 0.25 s default, `MIN_SEP_S`).
        z_min: Minimum z-score to accept a peak (module docstring: 6.0
            default, `Z_REGISTER_THRESHOLD`, always passed explicitly rather
            than assumed).

    Returns:
        `ImpulsePeak`s sorted by `time_offset_s` ascending; empty if *z* is
        empty or nothing reaches *z_min*.
    """
    x = np.asarray(z, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return []

    sep = max(1, int(min_sep_s / hop_s))
    taken = np.zeros(n, dtype=bool)
    order = np.argsort(x)[::-1]
    peaks: list[ImpulsePeak] = []
    for i in order:
        zi = float(x[i])
        if zi < z_min:
            break
        if taken[max(0, i - sep) : i + sep].any():
            continue
        taken[i] = True
        peaks.append(ImpulsePeak(time_offset_s=float(i) * hop_s, z=zi))
    return sorted(peaks, key=lambda p: p.time_offset_s)


def detect_impulses(
    samples: np.ndarray,
    rate_hz: float,
    *,
    lo_hz: float = BAND_LO_HZ,
    hi_hz: float = BAND_HI_HZ,
    frame_s: float = FRAME_S,
    hop_s: float = HOP_S,
    med_win_s: float = MED_WIN_S,
    min_sep_s: float = MIN_SEP_S,
    z_min: float = Z_REGISTER_THRESHOLD,
) -> list[ImpulsePeak]:
    """`band_frame_energy` -> `mad_z_score` -> `pick_peaks`, composed with one
    consistent parameter set (module docstring) -- the single entry point
    `scripts/candidate_kit.py`'s chunked per-session driver calls once per
    audio block per mic stream.
    """
    loge = band_frame_energy(
        samples, rate_hz, lo_hz=lo_hz, hi_hz=hi_hz, frame_s=frame_s, hop_s=hop_s
    )
    z = mad_z_score(loge, hop_s=hop_s, med_win_s=med_win_s)
    return pick_peaks(z, hop_s=hop_s, min_sep_s=min_sep_s, z_min=z_min)
