"""Build real-audio assets for the interactive live demo (`docs/demo/demo_live.html`)
and the single-screen control-room dashboard (`docs/demo/demo_dashboard.html`): short
WAV clips cut from the 080726 controlled-event campaign day, plus the
manifest describing them, plus (second/third subcommand) the pipeline-overview
figures and the self-contained demo page itself, plus (fourth subcommand) the
dashboard's own once-calibrated state/score/alarm data export.

Four subcommands:

    extract-clips     Read real 080726 data (via the existing `rowii.io.dataset.
                       discover`/`rowii.io.gantner.read_gantner` seams) and write
                       `docs/demo/assets/*.wav` + `docs/demo/assets/manifest.json`.
    render-figures    Render the four Section-1 pipeline PNGs (waveform, spectrogram,
                       feature-bars, score-histogram; see `render_figures`) from one
                       already-extracted clip WAV + the real fusion feature cache +
                       the existing demo's `demo-data` trace, and write them into
                       `docs/demo/assets/figures/*.png`.
    build-html        Template-inject the clips (as base64 data-URIs + waveform
                       sparklines), the four `render-figures` PNGs, and the EXISTING
                       `demo_080726_pu.html` state/score/report data model into
                       `docs/demo/demo_live_template.html`, producing
                       `docs/demo/demo_live.html`. Never touches real data -- pure
                       file/template assembly, so this step is fast and reproducible
                       from `docs/demo/assets/` alone (`render-figures`' PNG output
                       included, since those are themselves tracked files under
                       `assets/`, not a live real-data read).
    build-dashboard   Template-inject BOTH 080726 sessions' once-calibrated state/
                       score/alarm timelines (`results/step2/once-calibrated/`, its
                       `state_name`/`near_transition` columns included),
                       ground-truth strike events (`docs/groundtruth/`), and the
                       SAME already-extracted clips into `docs/demo/
                       demo_dashboard_template.html`, producing `docs/demo/
                       demo_dashboard.html` (see `build_dashboard_data`). Also pure
                       file/artifact assembly -- reads already-computed CSV/parquet/
                       markdown reports and already-extracted WAVs only, no
                       `ROWII_DATA_ROOT`, no model run.

Clip selection (six clips total):

    - One STATE clip per cluster id (`>= 0`) that actually appears in
      `results/pillar3/080726-pu_strikes/audio-beats-a0.01/segments.csv`: the LONGEST
      segment of that cluster, 10 s cut from its middle (`center_window`), nudged away
      from any hammer strike (`dodge_collision` against `docs/groundtruth/
      080726_events_pu.csv`, +/-10 s pad) if the two happen to collide.
    - Three STRIKE clips, 10 s from the logged event start: the pump-operation
      'plate-tur_0' strike (run 080726-pu_strikes), the standstill 'plate-gen_0'
      strike, and the standstill vane-sweep (the latter two from run
      080726-st_strikes) -- see `docs/groundtruth/080726_events_{pu,st}.csv`.

Audio contract: mono = channel 0 ("GenMic0") of the `RAWGeneratorMic__0` stream,
native 50 kHz, resampled to 16 kHz (`scipy.signal.resample_poly`, mirroring
`rowii.tfc.corpora`'s/`rowii.adapt.target_windows`'s own resampling convention) and
peak-normalized to -1 dBFS (`peak_normalize`) before 16-bit PCM WAV write -- otherwise
a quiet ambient-noise state clip and a sharp hammer strike would sit at wildly
different playback levels.

Memory discipline (this task's own instruction): `RAWGeneratorMic__0` burst files in
this campaign are ~0.8-1 GB each (`read_gantner` loads a file fully into memory --
`rowii.pipeline`'s own module docstring notes the same order of magnitude for its
chunked feature extraction). `_extract_mono_clip` loads exactly ONE burst file at a
time, slices out the ~10 s of samples it needs, and frees the full array
(`del`/`gc.collect()`) before returning -- never two burst files resident at once, and
never any stream other than `RAWGeneratorMic__0` (the three sibling streams'
per-file HEADERS are still read, cheaply, by `rowii.io.dataset.run_utc_offset_ns`,
which by design pools every stream's headers for its offset estimate -- see that
function's own docstring).

Structure (per this task's own instruction): the WINDOW ARITHMETIC (segment ->
10 s-centered window, collision check/dodge against strike events, UTC -> in-file
sample-index, burst-file selection) and the PEAK NORMALIZATION are pure functions with
no disk I/O, unit-tested with synthetic fixtures in `tests/test_make_demo_assets.py`.
Everything that touches `ROWII_DATA_ROOT`, writes a WAV, or reads/writes HTML lives in
this module too but is exercised by actually running the CLI against real data, not by
a `pytest -m "not data"` unit test.
"""
from __future__ import annotations

import argparse
import base64
import bisect
import gc
import html
import io
import json
import logging
import math
import re
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPTS_DIR.parent / "src"
for _extra_path in (str(_SCRIPTS_DIR), str(_SRC_DIR)):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

from rowii.config import Config, load_config  # noqa: E402
from rowii.io.dataset import (  # noqa: E402
    BurstFile,
    RecordingIndex,
    Run,
    discover,
    run_utc_offset_ns,
)
from rowii.io.gantner import read_gantner  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = _SCRIPTS_DIR.parent
DEFAULT_ASSETS_DIR = REPO_ROOT / "docs" / "demo" / "assets"
DEFAULT_TEMPLATE = REPO_ROOT / "docs" / "demo" / "demo_live_template.html"
DEFAULT_SOURCE_DEMO = REPO_ROOT / "docs" / "demo" / "demo_080726_pu.html"
DEFAULT_OUT_HTML = REPO_ROOT / "docs" / "demo" / "demo_live.html"

PU_RUN_NAME = "080726-pu_strikes"
ST_RUN_NAME = "080726-st_strikes"
PU_SEGMENTS_CSV = (
    REPO_ROOT / "results" / "pillar3" / PU_RUN_NAME / "audio-beats-a0.01" / "segments.csv"
)
PU_EVENTS_CSV = REPO_ROOT / "docs" / "groundtruth" / "080726_events_pu.csv"
ST_EVENTS_CSV = REPO_ROOT / "docs" / "groundtruth" / "080726_events_st.csv"

ONCE_CALIBRATED_DIR = REPO_ROOT / "results" / "step2" / "once-calibrated"
"""The "run once per instrumentation era, decide frozen-vs-recalibrate per
sentinel verdict" replay output (`scripts/run_once_calibrated.py`) -- the ONLY
existing artifact family that carries the two fields the control-room dashboard's
state tile needs natively (`state_name`, `near_transition`, both `scripts/
monitor.py`'s own columns of `alarms.parquet`), so `build_dashboard_data` reads
its per-run `monitor/<run>/<mode>/` outputs directly rather than the older,
KMeans-clustered `results/pillar3/` family `render_figures`/`extract_clips` above
read from (that family predates `scripts/run_once_calibrated.py`'s named states
and has no `state_name`/`near_transition` column at all)."""
DASHBOARD_REPRESENTATION = "audio-beats"
"""Mic-only (BEATs-embedding) scorer -- matches this module's own six demo clips,
all cut from the `RAWGeneratorMic__0` mono stream (module docstring), so the score
track the dashboard plots is scoring the SAME signal the live waveform/FFT panels
visualize."""
DASHBOARD_VIB_REPRESENTATION = "vibration"
"""Read ONLY for the Alarm-Feed's "which stream" cross-check (`_load_alarm_intervals`
+ `has_collision` in `_load_session`) -- an independent accelerometer-only scorer
run against the SAME grid, never mixed into the primary (mic) score/alarm/state
trace itself."""
DASHBOARD_THRESHOLD_MODE = "recalibrate"
"""Of once-calibrated's two threshold arms (`frozen` keeps the very first
calibration forever; `recalibrate` refits thresholds -- never the state/score
references themselves -- on each run's own calibration-side windows), `recalibrate`
is `audio-beats.json`'s own "cross-day evidence: the only recipe whose
realized false-alarm rate stayed at its nominal alpha" -- the operationally
recommended regime, not `frozen`'s deliberately-naive stress test (which alarms on
most of a held-out day almost by design, to prove the sentinel's drift detection
works; not a representative "how the system runs" demo)."""
DASHBOARD_EVENT_TOLERANCE_S = 5.0
"""Matches `scripts/eval_events.py`'s own `tolerance_s` (see any `event_notes.md`
"## Inputs" section) -- an alarm episode is tagged with a ground-truth strike using
the EXACT SAME pad the event-level evaluation already scored these runs against,
not a dashboard-invented number."""
_SESSION_LABEL = {
    ST_RUN_NAME: "080726 – standstill, with Schonhammer strikes",
    PU_RUN_NAME: "080726 – pump trial, with Schonhammer strikes",
}
DASHBOARD_DEFAULT_SESSION = ST_RUN_NAME
"""`080726-st_strikes` opens first: a compact ~24 min session whose own two demo
clips (module docstring's six-clip list) are BOTH hammer strikes, so pressing the
main Play button starts real audio (and, within that first clip's own 10 s, a real
alarm) almost immediately -- `080726-pu_strikes`, in contrast, is a ~3.6 h day whose
own first clip (a normal-state one) sits ~29 min in. Both sessions stay one click
away via the session selector."""
DEFAULT_DASHBOARD_TEMPLATE = REPO_ROOT / "docs" / "demo" / "demo_dashboard_template.html"
DEFAULT_DASHBOARD_OUT = REPO_ROOT / "docs" / "demo" / "demo_dashboard.html"

CLIP_DURATION_S = 10.0
COLLISION_PAD_S = 10.0
"""Padding (seconds, both sides) around every hammer-strike event a STATE clip's
window must clear -- task instruction: "±10s" against `080726_events_pu.csv`."""
TARGET_SAMPLE_RATE_HZ = 16_000
TARGET_DBFS = -1.0
MONO_STREAM = "RAWGeneratorMic__0"
MONO_CHANNEL_INDEX = 0
_EXPECTED_SOURCE_RATE_HZ = 50_000.0
_SOURCE_RATE_TOLERANCE_HZ = 5.0
"""`_resample_to_target` refuses to run (loud failure) if a burst file's OWN measured
rate strays further than this from the expected 50 kHz mic rate (verified against the
real 080726 header, `read_header(...).sample_rate_hz == 50000.000`) -- a fixed
`resample_poly` ratio silently mis-pitches audio resampled at the WRONG assumed source
rate, so a real deviation this large must raise, not produce a subtly wrong clip."""


# ---------------------------------------------------------------------------
# Pure helpers (no disk I/O) -- unit-tested in tests/test_make_demo_assets.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """One row of a Step-1 `segments.csv` table (`rowii.state.segments.to_segments`),
    already parsed into timezone-aware true-UTC datetimes."""

    start_utc: datetime
    end_utc: datetime
    cluster_id: int

    @property
    def duration_s(self) -> float:
        return (self.end_utc - self.start_utc).total_seconds()


def longest_segment_per_cluster(segments: Sequence[Segment]) -> dict[int, Segment]:
    """The longest `Segment` per `cluster_id >= 0`, ties broken by the EARLIEST
    `start_utc`. `cluster_id < 0` (the detector's invalid/transition sentinel,
    `rowii.state.segments.to_segments`) never wins -- it is never a real operating
    state and must never be picked as a "state" demo clip source.
    """
    best: dict[int, Segment] = {}
    for seg in segments:
        if seg.cluster_id < 0:
            continue
        incumbent = best.get(seg.cluster_id)
        if (
            incumbent is None
            or seg.duration_s > incumbent.duration_s
            or (seg.duration_s == incumbent.duration_s and seg.start_utc < incumbent.start_utc)
        ):
            best[seg.cluster_id] = seg
    return best


def center_window(
    segment_start: datetime, segment_end: datetime, duration_s: float
) -> tuple[datetime, datetime]:
    """A `duration_s`-long window centered on `[segment_start, segment_end)`'s
    midpoint. If the segment itself is shorter than `duration_s` (real case: cluster
    0's longest segment in the 080726-pu_strikes segments.csv is only 6 s) the window
    necessarily extends beyond the segment's own bounds -- there is no way to fit a
    longer clip inside a shorter segment. Deliberate, documented trade-off; see
    `dodge_collision`'s own docstring for how a collision is still handled in that
    case (there is no room left to dodge INTO).
    """
    mid = segment_start + (segment_end - segment_start) / 2
    half = timedelta(seconds=duration_s / 2)
    return mid - half, mid + half


def has_collision(
    window_start: datetime,
    window_end: datetime,
    events: Sequence[tuple[datetime, datetime]],
    pad_s: float,
) -> bool:
    """True iff `[window_start, window_end)` overlaps any *events* interval, each
    padded by `pad_s` seconds on both sides. Half-open on both the window and the
    padded events -- touching exactly at a boundary is NOT a collision, matching
    `rowii.eval.events`' own inclusive-start/exclusive-end convention elsewhere in
    this codebase.
    """
    pad = timedelta(seconds=pad_s)
    return any(
        window_start < e_end + pad and window_end > e_start - pad for e_start, e_end in events
    )


def dodge_collision(
    window_start: datetime,
    window_end: datetime,
    allowed_start: datetime,
    allowed_end: datetime,
    events: Sequence[tuple[datetime, datetime]],
    pad_s: float,
    step_s: float = 1.0,
) -> tuple[datetime, datetime]:
    """If `[window_start, window_end)` collides with any padded *events* interval
    (`has_collision`), search outward in `step_s` increments (alternating
    later/earlier at each step) for the smallest shift that both clears every
    collision AND keeps the window fully inside `[allowed_start, allowed_end]` --
    "weiche innerhalb des Segments aus": a caller passes the source STATE segment's
    own bounds as the allowed range, so a dodged clip never leaves the segment it was
    chosen to represent.

    Returns the window UNCHANGED when: it does not collide in the first place; the
    allowed range is narrower than the window itself (a segment shorter than the clip
    duration has no slack to shift into at all -- `center_window`'s own short-segment
    case); or the search exhausts the whole allowed range without finding a clear
    placement. This is a documented best-effort fallback, not a silent failure -- on
    the real 080726 data this script runs against, no chosen clip actually reaches it
    (every real collision case has room to clear), but the contract must still be
    well-defined for the general case.

    Args:
        window_start: Candidate window start (typically `center_window`'s output).
        window_end: Candidate window end.
        allowed_start: Left bound the window may never cross.
        allowed_end: Right bound the window may never cross.
        events: `(start, end)` intervals to avoid.
        pad_s: Symmetric padding added to every event before the collision check.
        step_s: Search step size in seconds.

    Returns:
        A `(start, end)` window of the SAME duration as the input.
    """
    if not has_collision(window_start, window_end, events, pad_s):
        return window_start, window_end

    duration = window_end - window_start
    max_shift = (allowed_end - allowed_start) - duration
    if max_shift.total_seconds() < 0:
        return window_start, window_end

    step = timedelta(seconds=step_s)
    offset = step
    while offset <= max_shift:
        for direction in (1, -1):
            shifted_start = window_start + direction * offset
            shifted_end = window_end + direction * offset
            if shifted_start < allowed_start or shifted_end > allowed_end:
                continue
            if not has_collision(shifted_start, shifted_end, events, pad_s):
                return shifted_start, shifted_end
        offset += step
    return window_start, window_end


def find_burst_file(files: Sequence[BurstFile], target_utc: datetime) -> BurstFile:
    """The single burst file among *files* whose filename-hint bucket covers
    *target_utc*: the LATEST file (by `start_utc_hint`) that still starts at or before
    *target_utc*. Pure w.r.t. disk -- compares only each `BurstFile`'s own
    `start_utc_hint` (`rowii.io.dataset`'s local -> UTC filename parse), never opens a
    file. The bucket's upper bound is implicit (the next file's own start, or
    unbounded for the last file) rather than checked here: every caller in this script
    already knows its target instant falls comfortably inside a single ~12-minute
    burst file, never within the DAQ's short inter-file gap -- `utc_window_to_
    sample_range`'s `searchsorted` against that file's OWN real per-sample timestamps
    is the actual membership test.

    Raises:
        ValueError: if *files* is empty, or *target_utc* is before the earliest
            file's `start_utc_hint` (no file could possibly cover it).
    """
    ordered = sorted(files, key=lambda f: f.start_utc_hint)
    if not ordered or target_utc < ordered[0].start_utc_hint:
        earliest = ordered[0].start_utc_hint if ordered else "n/a (no files)"
        raise ValueError(f"no burst file covers {target_utc} (earliest file starts {earliest})")
    candidate = ordered[0]
    for bf in ordered[1:]:
        if bf.start_utc_hint > target_utc:
            break
        candidate = bf
    return candidate


def utc_window_to_sample_range(
    window_start_utc: datetime,
    window_end_utc: datetime,
    sample_timestamps_ns: np.ndarray,
) -> tuple[int, int]:
    """Convert a true-UTC time window into a half-open `[start, end)` sample-index
    range against ONE burst file's own per-sample true-UTC timestamps
    (`rowii.io.gantner.GantnerFile.timestamps_ns`, already shifted onto the true-UTC
    axis by the run's derived `rowii.io.dataset.run_utc_offset_ns` -- see
    `_extract_mono_clip`). Uses `np.searchsorted` against the REAL (jittery) per-frame
    clock rather than a nominal-rate arithmetic estimate -- exact regardless of any
    small deviation between the header's rate estimate and the file's true
    instantaneous rate (mirrors `rowii.signals.windows.window_slices`'s own
    per-sample-timestamp approach, not a nominal-rate shortcut). `searchsorted`'s own
    contract already clamps both indices to `[0, len(sample_timestamps_ns)]` -- a
    window reaching outside the file's own span simply gets whatever overlap exists.

    Both endpoints are converted via `round(dt.timestamp() * 1e9)` (matching `rowii.
    io.dataset._hint_utc_ns`'s own documented rounding rationale: negligible
    sub-nanosecond float noise at 2020s-scale timestamps) and compared as `uint64`
    against *sample_timestamps_ns* -- deliberately NOT a bare Python-int comparison,
    which numpy can silently upcast to float64 for a uint64 array (losing precision
    far below these ~1e18 ns values; the exact bug class `rowii.pipeline._shift_ts_ns`
    exists to avoid).

    Raises:
        ValueError: if `window_end_utc <= window_start_utc`.
    """
    if window_end_utc <= window_start_utc:
        raise ValueError(
            f"window_end_utc ({window_end_utc}) must be after window_start_utc "
            f"({window_start_utc})"
        )
    start_ns = np.uint64(round(window_start_utc.timestamp() * 1e9))
    end_ns = np.uint64(round(window_end_utc.timestamp() * 1e9))
    start_idx = int(np.searchsorted(sample_timestamps_ns, start_ns, side="left"))
    end_idx = int(np.searchsorted(sample_timestamps_ns, end_ns, side="left"))
    return start_idx, end_idx


def peak_normalize(samples: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    """Scale *samples* so its absolute peak sits at *target_dbfs* dBFS relative to
    full scale (1.0) -- e.g. -1 dBFS ~= 0.891. Silence (all-zero, or empty) is
    returned unchanged (float64-cast): there is no peak to scale by, and dividing by
    zero would produce NaN/inf. Every other input comes back genuinely
    peak-normalized, same shape, float64.
    """
    x = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0.0:
        return x
    target_linear = 10.0 ** (target_dbfs / 20.0)
    return np.asarray(x * (target_linear / peak), dtype=np.float64)


def quantile_threshold(scores: Sequence[float], alpha: float) -> float:
    """The (1 - alpha) quantile of *scores* -- the illustrative, POOLED-across-the-
    whole-day conformal-style threshold line drawn on the Section-1 score-histogram
    figure (`render_figures`). This is NOT the literal per-state
    operational threshold: the real pipeline (`monitor_notes.md` for e.g. this demo's
    own `audio-beats-a0.01` run) fits one threshold PER detected state on that state's
    own calibration-side windows -- 0.0498 for state 1 vs. 0.0679 for state 3 on this
    exact run/day. Pooling every scored window across the day into one line is a
    deliberate simplification for the pipeline-OVERVIEW figure, not a reproduction of
    that per-state calibration (the figure/caption say so explicitly).

    Args:
        scores: Scored-window scores, any order (not required to be pre-sorted).
        alpha: Miscoverage level, strictly between 0 and 1 (e.g. 0.01).

    Raises:
        ValueError: if *scores* is empty, or *alpha* is not in the open interval
            (0, 1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    values = np.asarray(list(scores), dtype=np.float64)
    if values.size == 0:
        raise ValueError("scores must be non-empty")
    return float(np.quantile(values, 1.0 - alpha))


def nearest_sorted_index(times: Sequence[float], target: float) -> int:
    """The index into ascending-sorted *times* whose value is closest to *target* --
    `bisect.bisect_left` alone only gives the FIRST index `>= target` (the insertion
    point), which is wrong whenever the PRECEDING entry is actually nearer; this
    compares both straddling neighbors and picks the closer one (a tie -- equal
    distance either side -- keeps the EARLIER index, `bisect_left`'s own convention).

    This is the reference implementation for this demo's two call sites
    that both need "closest entry to a given instant in a monotonically increasing
    but IRREGULARLY spaced array of times": (1) here, in `render_figures`, to pick
    which real `demo-data` trace row backs the Section-1 feature-bars figure's example
    window (the fusion cache is a dense uniform 1 Hz grid, but the TRACE array is
    sparse -- it excludes calibration-consumed/unscored windows -- so a naive
    round-to-nearest-second can land on a window with no matching trace entry).
    (2) Hand-ported (kept algorithmically identical on purpose) into the live-replay
    JavaScript in `demo_live_template.html`'s playhead, which runs the same lookup
    against `D.trace` at animation-frame rate (~60 fps) to drive the live panel --
    JS has no test runner in this project, so THIS function, tested here, is the
    verified spec that port is checked against by hand, not a separate
    implementation.

    Raises:
        ValueError: if *times* is empty.
    """
    if len(times) == 0:
        raise ValueError("times must be non-empty")
    i = bisect.bisect_left(times, target)
    if i == 0:
        return 0
    if i == len(times):
        return len(times) - 1
    before, after = times[i - 1], times[i]
    return i - 1 if (target - before) <= (after - target) else i


def column_zscore_stats(
    features: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean/std of *features*, computed over only the `valid_mask`-true
    ROWS -- the day-level reference distribution `top_k_abs_z_indices` scores one
    window's own row against for the Section-1 "feature bars" figure
    (`render_figures`). `ddof=0` (population std, numpy's own default and every other
    z-score in this codebase).

    Raises:
        ValueError: if *features*/`valid_mask` disagree on row count, or no row is
            valid at all (mean/std would be undefined).
    """
    if features.shape[0] != valid_mask.shape[0]:
        raise ValueError(
            f"features has {features.shape[0]} rows but valid_mask has "
            f"{valid_mask.shape[0]}"
        )
    if not valid_mask.any():
        raise ValueError("valid_mask has no True entries -- no rows to compute stats over")
    valid_rows = features[valid_mask]
    return valid_rows.mean(axis=0), valid_rows.std(axis=0)


def top_k_abs_z_indices(row: np.ndarray, mean: np.ndarray, std: np.ndarray, k: int) -> list[int]:
    """Indices of the *k* columns of *row* with the largest |z|-score
    `(row - mean) / std`, sorted DESCENDING by |z| -- the pure selection logic behind
    the Section-1 "feature bars" figure (which columns of one real 243-feature
    fusion window deviate most from `column_zscore_stats`' day-level reference).
    Columns with `std == 0` (constant across every valid window that day -- none
    occur in the real `080726-pu_strikes` fusion cache, but the contract must still be
    well-defined) get z == 0, i.e. they can never be selected unless *k* exceeds the
    number of non-constant columns.

    Raises:
        ValueError: if `row`/`mean`/`std` do not all share one shape, or `k <= 0`.
    """
    if not (row.shape == mean.shape == std.shape):
        raise ValueError(
            f"row/mean/std must share one shape, got {row.shape}/{mean.shape}/{std.shape}"
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    safe_std = np.where(std == 0.0, 1.0, std)
    z = np.where(std == 0.0, 0.0, (row - mean) / safe_std)
    order = np.argsort(-np.abs(z))
    return [int(i) for i in order[: min(k, len(row))]]


_FEATURE_NAME_RE = re.compile(r"^RAW(\w+?)__(\d+)::ch(\d+)_(.+)$")
_STREAM_ABBREV = {"Generator": "Gen", "Turbine": "Tur"}


def shorten_feature_name(name: str) -> str:
    """A short, chart-label-friendly form of a real `feature_names` entry from the
    fusion cache, e.g. `"RAWGeneratorMic__0::ch0_log_rms"` -> `"GenMic0.ch0.log_rms"`
    -- pure string transform for the Section-1 "feature bars" figure's bar labels
    (`render_figures`). Strips the `RAW` prefix, abbreviates the `Generator`/`Turbine`
    stream-name component (`_STREAM_ABBREV`), and swaps the `::`/`ch<i>_` separators
    for a compact dotted form. A name that does not match the expected
    `RAW<Stream>__<n>::ch<i>_<feature>` shape (the real cache's own inventory, module
    docstring) is returned UNCHANGED -- defensive fallback, this must never raise.
    """
    m = _FEATURE_NAME_RE.match(name)
    if m is None:
        return name
    stream, stream_idx, ch, feat = m.groups()
    for long, short in _STREAM_ABBREV.items():
        stream = stream.replace(long, short)
    return f"{stream}{stream_idx}.ch{ch}.{feat}"


def extract_window_samples(
    pcm: np.ndarray, sample_rate_hz: int, start_s: float, duration_s: float
) -> np.ndarray:
    """The `[start_s, start_s + duration_s)` slice of *pcm* (samples, any dtype --
    typically the int16 array `scipy.io.wavfile.read` returns for one of this demo's
    own clip WAVs), converted to `float64`. The pure "which samples feed the figure"
    contract behind the Section-1 waveform/spectrogram renderers
    (`render_figures`), kept separate from those two so the slice arithmetic is
    unit-testable without matplotlib (mirrors this module's existing IO-touching vs.
    pure-helper split, module docstring).

    Raises:
        ValueError: if the requested window falls even partially outside `[0,
            len(pcm))` at *sample_rate_hz* -- a caller bug (an out-of-range constant
            in `render_figures`), not a real runtime condition, so this fails loudly
            rather than silently clamping/padding.
    """
    start_idx = round(start_s * sample_rate_hz)
    end_idx = round((start_s + duration_s) * sample_rate_hz)
    if start_idx < 0 or end_idx > len(pcm) or end_idx <= start_idx:
        raise ValueError(
            f"window [{start_s}, {start_s + duration_s}) s -> samples "
            f"[{start_idx}, {end_idx}) is out of range for pcm of length {len(pcm)} "
            f"at {sample_rate_hz} Hz"
        )
    return np.asarray(pcm[start_idx:end_idx], dtype=np.float64)


# ---------------------------------------------------------------------------
# Control-room dashboard helpers (feat/demo-dashboard) -- pure, no disk I/O,
# unit-tested in tests/test_make_demo_assets.py. `build_dashboard_data` (IO-touching
# section below) is the only caller.
# ---------------------------------------------------------------------------


def matching_event_kind(
    window_start: datetime,
    window_end: datetime,
    events: Sequence[tuple[datetime, datetime, str]],
    pad_s: float,
) -> str | None:
    """The `kind` of the FIRST *events* entry (in list order) whose *pad_s*-padded
    interval overlaps `[window_start, window_end)` -- `has_collision` generalized to
    report WHICH event matched (that function only reports whether ANY did), for
    tagging one alarm episode with the ground-truth strike it most plausibly
    belongs to on the dashboard's Alarm-Feed cards. Same half-open-window/symmetric-
    padding convention as `has_collision` (this module) and `rowii.eval.events`
    elsewhere in this codebase. `None` if no event overlaps.
    """
    pad = timedelta(seconds=pad_s)
    for e_start, e_end, kind in events:
        if window_start < e_end + pad and window_end > e_start - pad:
            return kind
    return None


_STATE_NAME = {
    "standstill": "Standstill",
    "turbine": "Turbine operation",
    "pump": "Pump operation",
    "phase-shifter": "Phase-shifter operation",
    "invalid": "Transition / invalid",
}


def state_display_name(name: str) -> str:
    """Human-readable display label for one of `rowii.scada.labels._KNOWN_STATES`
    (`"standstill"`, `"turbine"`, `"pump"`, `"phase-shifter"`) plus this codebase's
    own `"invalid"` sentinel (`scripts/monitor.py`'s `_state_name_for`) -- the
    dashboard's state-badge subtitle. Falls back to a labelled passthrough for
    `derive_state_names`' own `cluster-<id>` naming fallback (a cluster without a
    >=50% ground-truth plurality winner at commissioning time) -- not expected on
    the real 080726 data this dashboard embeds (every state occurring there has a
    plurality winner), but must stay well-defined rather than raise.
    """
    return _STATE_NAME.get(name, f"State ({name})")


def parse_markdown_table(text: str, header_prefix: str) -> tuple[list[str], list[dict[str, str]]]:
    """The first GitHub-flavoured-markdown table in *text* whose header row (a
    `"| col1 | col2 | ... |"` line) starts with *header_prefix*, parsed into
    `(column_names, rows)` -- each row a `{column_name: raw_cell_string}` dict (cell
    text stripped), in table order. The mandatory `|---|---|`-style alignment row
    right after the header is always skipped, unconditionally (every real table this
    parses -- `scripts/monitor.py`'s and `scripts/eval_events.py`'s own markdown
    writers -- always emits exactly one). Parsing then stops at the first following
    line that is no longer a well-formed `|`-delimited row of the SAME column count
    (blank line, prose, a later heading, ...) -- so a short table embedded in a
    longer report never accidentally swallows unrelated text below it.

    One parser backs every dashboard field pulled from either report -- see
    `parse_state_table`/`parse_event_summary_table`, its two typed call sites,
    rather than a bespoke regex per table.

    Raises:
        ValueError: if no header line starts with *header_prefix*.
    """
    lines = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith(header_prefix)), None
    )
    if header_idx is None:
        raise ValueError(f"no markdown table header starting with {header_prefix!r} found")
    columns = [c.strip() for c in lines[header_idx].strip().strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for ln in lines[header_idx + 2 :]:
        stripped = ln.strip()
        if not stripped.startswith("|"):
            break
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) != len(columns):
            break
        rows.append(dict(zip(columns, cells, strict=True)))
    return columns, rows


_STATE_CELL_RE = re.compile(r"^(-?\d+)\s*\(([^)]*)\)$")


def parse_state_table(monitor_notes_text: str) -> dict[int, dict[str, Any]]:
    """`scripts/monitor.py`'s `monitor_notes.md` "## Per-state results" table (e.g.
    `results/step2/once-calibrated/audio-beats/monitor/<run>/recalibrate/
    monitor_notes.md`), parsed into `{state_id: {"name": str, "threshold": float |
    None, "low_confidence": bool}}` -- the dashboard's per-state naming + live
    threshold-line source (`build_dashboard_data`). `threshold` is `math.inf` for
    the table's literal `"inf"` cell (state has a certified never-alarm threshold,
    too few calibration windows -- `low_confidence` is then always `True`) and
    `None` for `"n/a"` (state never occurs on this run at all -- zero calibration-
    side windows, `monitor.py`'s `no_conformal_data` status; such a state also never
    appears in `alarms.parquet`, so the dashboard never needs a value for it, but
    the table still carries the row).

    Raises:
        ValueError: if the table is missing (`parse_markdown_table`), or a `state`
            cell is not the expected `"<id> (<name>)"` shape.
    """
    _, rows = parse_markdown_table(monitor_notes_text, "| state ")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        m = _STATE_CELL_RE.match(row["state"])
        if m is None:
            raise ValueError(f"unexpected state-table 'state' cell: {row['state']!r}")
        state_id = int(m.group(1))
        threshold_raw = row["threshold"]
        if threshold_raw == "inf":
            threshold: float | None = math.inf
        elif threshold_raw == "n/a":
            threshold = None
        else:
            threshold = float(threshold_raw)
        result[state_id] = {
            "name": m.group(2),
            "threshold": threshold,
            "low_confidence": row["low_confidence"] == "True",
        }
    return result


def parse_event_summary_table(event_notes_text: str) -> dict[str, float]:
    """`scripts/eval_events.py`'s `event_notes.md` "## Summary" table, parsed into
    `{"n_events": int, "n_detected": int, "event_tpr": float}` -- the dashboard's
    "X/Y strikes detected" header stat. Deliberately only these three fields (the
    table also carries false-alarm-rate columns not surfaced on the dashboard).

    Raises:
        ValueError: if the table is missing (`parse_markdown_table`) or has no data
            row.
    """
    _, rows = parse_markdown_table(event_notes_text, "| n_events ")
    if not rows:
        raise ValueError("event summary table has no data row")
    row = rows[0]
    return {
        "n_events": int(row["n_events"]),
        "n_detected": int(row["n_detected"]),
        "event_tpr": float(row["event_tpr"]),
    }


# ---------------------------------------------------------------------------
# IO-touching helpers (real files: CSVs, burst .dat files, WAV write)
# ---------------------------------------------------------------------------


def _shift_ts_ns(ts_ns: np.ndarray, offset_ns: int) -> np.ndarray:
    """`ts_ns + offset_ns`, staying in `uint64` throughout -- mirrors (duplicated, not
    imported: small self-contained utility, house convention, see e.g. `rowii.
    pipeline._shift_ts_ns`'s own docstring) the fix for a real bug class in this
    codebase: mixing `int64`/Python-int and `uint64` numpy arrays silently upcasts
    BOTH to `float64`, losing precision far below these ~1e18 ns timestamps.
    """
    if offset_ns >= 0:
        return ts_ns + np.uint64(offset_ns)
    return ts_ns - np.uint64(-offset_ns)


def _load_segments_csv(path: Path) -> list[Segment]:
    """`Segment` rows from a Step-1 `segments.csv` (`start_utc,end_utc,duration_s,
    cluster_id`), sorted as written (chronological, `rowii.state.segments.
    to_segments`'s own output order)."""
    df = pd.read_csv(path)
    starts = pd.to_datetime(df["start_utc"], utc=True)
    ends = pd.to_datetime(df["end_utc"], utc=True)
    with warnings.catch_warnings():
        # Real segments.csv rows carry sub-microsecond (ns) precision (e.g.
        # "...11:46:03.410000115+00:00"); `datetime` only has microsecond
        # resolution, so pandas warns about discarding the residual ~100 ns on
        # every `to_pydatetime()` call. Negligible next to a 10 s clip window --
        # silenced here specifically (not globally), not worked around.
        warnings.filterwarnings(
            "ignore", message="Discarding nonzero nanoseconds", category=UserWarning
        )
        return [
            Segment(start_utc=s.to_pydatetime(), end_utc=e.to_pydatetime(), cluster_id=int(c))
            for s, e, c in zip(starts, ends, df["cluster_id"], strict=True)
        ]


def _load_events_csv(path: Path) -> pd.DataFrame:
    """A `docs/groundtruth/*_events_*.csv` table (`start_utc,end_utc,kind`, leading
    `#`-comment lines), with `start_utc`/`end_utc` parsed to timezone-aware UTC."""
    df = pd.read_csv(path, comment="#")
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
    df["end_utc"] = pd.to_datetime(df["end_utc"], utc=True)
    return df


def _event_by_kind(events: pd.DataFrame, kind: str) -> tuple[datetime, datetime]:
    rows = events[events["kind"] == kind]
    if rows.empty:
        known = sorted(events["kind"].unique().tolist())
        raise ValueError(f"no event with kind={kind!r} in events table (have: {known})")
    row = rows.iloc[0]
    start_utc: datetime = row["start_utc"].to_pydatetime()
    end_utc: datetime = row["end_utc"].to_pydatetime()
    return start_utc, end_utc


def _get_run(index: RecordingIndex, name: str) -> Run:
    for run in index.runs:
        if run.name == name:
            return run
    available = ", ".join(sorted(r.name for r in index.runs))
    raise SystemExit(f"make_demo_assets: run {name!r} not discovered; available: {available}")


def _extract_mono_clip(
    files: Sequence[BurstFile],
    offset_ns: int,
    window_start_utc: datetime,
    window_end_utc: datetime,
) -> tuple[np.ndarray, float]:
    """Load exactly ONE `RAWGeneratorMic__0` burst file (`find_burst_file`'s pick) and
    slice out channel 0 (mono) of `[window_start_utc, window_end_utc)`. Returns the
    raw (still at the file's native rate, NOT yet resampled/normalized) float64
    samples plus that file's own measured sample rate.

    Memory discipline (module docstring): the full `GantnerFile` (~0.8-1 GB for this
    campaign's `RAWGeneratorMic__0`) is dereferenced and garbage-collected before this
    function returns -- only the tiny (~10 s) slice survives.

    Raises:
        ValueError: if the requested window has no overlap with the selected file at
            all (e.g. a caller bug, or a window that crosses a burst-file boundary --
            not expected for any of this script's own 10 s clips against ~12-minute
            files, see the module docstring).
    """
    bf = find_burst_file(files, window_start_utc)
    gf = read_gantner(bf.path)
    true_utc_ns = _shift_ts_ns(gf.timestamps_ns, offset_ns)
    start_idx, end_idx = utc_window_to_sample_range(window_start_utc, window_end_utc, true_utc_ns)
    if end_idx <= start_idx:
        del gf, true_utc_ns
        gc.collect()
        raise ValueError(
            f"[{window_start_utc}, {window_end_utc}) has no overlap with {bf.path.name}"
        )
    clip = gf.data[start_idx:end_idx, MONO_CHANNEL_INDEX].astype(np.float64).copy()
    rate_hz = gf.header.sample_rate_hz
    del gf, true_utc_ns
    gc.collect()
    return clip, rate_hz


def _resample_to_target(samples: np.ndarray, source_rate_hz: float) -> np.ndarray:
    if abs(source_rate_hz - _EXPECTED_SOURCE_RATE_HZ) > _SOURCE_RATE_TOLERANCE_HZ:
        raise ValueError(
            f"RAWGeneratorMic__0 sample rate {source_rate_hz} Hz is far from the "
            f"expected {_EXPECTED_SOURCE_RATE_HZ} Hz -- refusing to resample with a "
            f"fixed ratio (module docstring)"
        )
    resampled: np.ndarray = resample_poly(samples, TARGET_SAMPLE_RATE_HZ, round(source_rate_hz))
    return resampled.astype(np.float64)


def _write_clip_wav(path: Path, samples_float: np.ndarray) -> None:
    normalized = peak_normalize(samples_float, TARGET_DBFS)
    pcm16 = np.clip(
        normalized * np.iinfo(np.int16).max, np.iinfo(np.int16).min, np.iinfo(np.int16).max
    ).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, TARGET_SAMPLE_RATE_HZ, pcm16)


# ---------------------------------------------------------------------------
# Clip assembly (extract-clips)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipMeta:
    """One `manifest.json` entry."""

    file: str
    kind: str
    """`"state"` or `"strike"`."""
    label: str
    """`cluster_id` (state) or the GT `kind` string (strike)."""
    start_utc: datetime
    duration_s: float
    source_run: str
    description: str
    """One English sentence."""

    def to_json(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "kind": self.kind,
            "label": self.label,
            "start_utc": self.start_utc.isoformat(),
            "duration_s": self.duration_s,
            "source_run": self.source_run,
            "description": self.description,
        }


@dataclass(frozen=True)
class _StrikeTarget:
    filename: str
    run: Run
    offset_ns: int
    events: pd.DataFrame
    gt_kind: str
    description: str


def _build_state_clips(pu_run: Run, out_dir: Path) -> list[ClipMeta]:
    segments = _load_segments_csv(PU_SEGMENTS_CSV)
    events_df = _load_events_csv(PU_EVENTS_CSV)
    event_intervals = list(
        zip(
            (t.to_pydatetime() for t in events_df["start_utc"]),
            (t.to_pydatetime() for t in events_df["end_utc"]),
            strict=True,
        )
    )
    offset_ns = run_utc_offset_ns(pu_run)
    longest = longest_segment_per_cluster(segments)

    clips: list[ClipMeta] = []
    for cluster_id in sorted(longest):
        seg = longest[cluster_id]
        w_start, w_end = center_window(seg.start_utc, seg.end_utc, CLIP_DURATION_S)
        w_start, w_end = dodge_collision(
            w_start, w_end, seg.start_utc, seg.end_utc, event_intervals, COLLISION_PAD_S
        )

        samples, rate_hz = _extract_mono_clip(pu_run.files[MONO_STREAM], offset_ns, w_start, w_end)
        resampled = _resample_to_target(samples, rate_hz)
        filename = f"state_cluster{cluster_id}.wav"
        _write_clip_wav(out_dir / filename, resampled)

        n_windows = int(round(sum(s.duration_s for s in segments if s.cluster_id == cluster_id)))
        description = (
            f"Unsupervised-detected state {cluster_id} ({n_windows} windows on this "
            f"measurement day, 1-s grid) – 10 s from the middle of this state's longest "
            f"contiguous segment, checked against the strike ground truth."
        )
        clips.append(
            ClipMeta(
                file=filename,
                kind="state",
                label=str(cluster_id),
                start_utc=w_start,
                duration_s=CLIP_DURATION_S,
                source_run=pu_run.name,
                description=description,
            )
        )
        logger.info(
            "make_demo_assets: state cluster %d -> %s (start %s)", cluster_id, filename, w_start
        )
    return clips


def _build_strike_clips(pu_run: Run, st_run: Run, out_dir: Path) -> list[ClipMeta]:
    pu_events = _load_events_csv(PU_EVENTS_CSV)
    st_events = _load_events_csv(ST_EVENTS_CSV)

    targets = [
        _StrikeTarget(
            filename="strike_pump_plate_tur_0.wav",
            run=pu_run,
            offset_ns=run_utc_offset_ns(pu_run),
            events=pu_events,
            gt_kind="plate-tur_0",
            description=(
                'Schonhammer strike "plate-tur_0" (reference plate, '
                "turbine side, 0°) during pump operation – SCADA-confirmed at "
                "approx. −279 MW / −377.8 rpm."
            ),
        ),
        _StrikeTarget(
            filename="strike_standstill_plate_gen_0.wav",
            run=st_run,
            offset_ns=run_utc_offset_ns(st_run),
            events=st_events,
            gt_kind="plate-gen_0",
            description=(
                'Schonhammer strike "plate-gen_0" (reference plate, '
                "generator side, 0°) at standstill (calibration session)."
            ),
        ),
        _StrikeTarget(
            filename="strike_standstill_vane_sweep.wav",
            run=st_run,
            offset_ns=run_utc_offset_ns(st_run),
            events=st_events,
            gt_kind="vane-sweep",
            description=(
                "Vane sweep – structure-borne excitation at the guide-vane cover, "
                "first 10 s of the roughly three-minute sweep session (standstill)."
            ),
        ),
    ]

    clips: list[ClipMeta] = []
    for t in targets:
        event_start, _event_end = _event_by_kind(t.events, t.gt_kind)
        w_start = event_start
        w_end = w_start + timedelta(seconds=CLIP_DURATION_S)

        samples, rate_hz = _extract_mono_clip(t.run.files[MONO_STREAM], t.offset_ns, w_start, w_end)
        resampled = _resample_to_target(samples, rate_hz)
        _write_clip_wav(out_dir / t.filename, resampled)

        clips.append(
            ClipMeta(
                file=t.filename,
                kind="strike",
                label=t.gt_kind,
                start_utc=w_start,
                duration_s=CLIP_DURATION_S,
                source_run=t.run.name,
                description=t.description,
            )
        )
        logger.info(
            "make_demo_assets: strike %r -> %s (start %s)", t.gt_kind, t.filename, w_start
        )
    return clips


def extract_clips(cfg: Config, out_dir: Path) -> list[ClipMeta]:
    """Discover the 080726 runs, cut all six clips, write their WAVs into *out_dir*
    plus `<out_dir>/manifest.json`. Returns the same clip list that was written."""
    index = discover(cfg.data_root)
    pu_run = _get_run(index, PU_RUN_NAME)
    st_run = _get_run(index, ST_RUN_NAME)

    out_dir.mkdir(parents=True, exist_ok=True)
    clips: list[ClipMeta] = []
    clips.extend(_build_state_clips(pu_run, out_dir))
    clips.extend(_build_strike_clips(pu_run, st_run, out_dir))

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"clips": [c.to_json() for c in clips]}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("make_demo_assets: wrote %s (%d clips)", manifest_path, len(clips))
    return clips


# ---------------------------------------------------------------------------
# HTML build (build-html) -- no real ROWII data, only docs/demo/assets + the
# existing demo_080726_pu.html
# ---------------------------------------------------------------------------

_CLIP_KIND_LABEL = {"state": "State", "strike": "Strike"}
_DEMO_DATA_RE = re.compile(
    r'<script id="demo-data" type="application/json">(.*?)</script>', re.DOTALL
)
_ANCHOR_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _extract_demo_data_block(source_demo_path: Path) -> str:
    """The raw JSON text inside `<script id="demo-data">...</script>` of an EXISTING
    demo page (`demo_080726_pu.html`) -- copied byte-for-byte, never re-serialized
    (task instruction: "do NOT regenerate the data"), so `demo_live.html` shows
    EXACTLY the same Step-1/Step-2/conformal outputs that page does.

    Raises:
        ValueError: if *source_demo_path* has no `<script id="demo-data">` block, or
            its content is not valid JSON (fail fast rather than embed garbage).
    """
    text = source_demo_path.read_text(encoding="utf-8")
    m = _DEMO_DATA_RE.search(text)
    if m is None:
        raise ValueError(f'{source_demo_path} has no <script id="demo-data"> block')
    raw = m.group(1)
    json.loads(raw)  # fail fast if the source page's own data block is not valid JSON
    return raw


def _render_sparkline_png_base64(
    samples_pcm: np.ndarray, *, width_px: int = 220, height_px: int = 36
) -> str:
    """A tiny min/max-envelope waveform sparkline (no axes/ticks/padding),
    base64-encoded PNG -- a cosmetic complement to each `<audio>` player in the
    "Listen" section. matplotlib is imported LOCALLY (not at module level): it is a
    required project dependency (`pyproject.toml`), but only this one, optional-in-
    spirit, cosmetic function needs it, so keeping the import scoped here avoids
    forcing every other `make_demo_assets` entry point through matplotlib's "Agg"
    backend setup. Not a pure helper (renders bytes) and not unit-tested -- see the
    module docstring.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mono = samples_pcm.astype(np.float64)
    n_buckets = min(width_px, max(1, mono.size))
    edges = np.linspace(0, mono.size, n_buckets + 1).astype(np.int64)
    mins = np.zeros(n_buckets)
    maxs = np.zeros(n_buckets)
    for i in range(n_buckets):
        chunk = mono[edges[i] : edges[i + 1]]
        if chunk.size:
            mins[i] = chunk.min()
            maxs[i] = chunk.max()

    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.fill_between(np.arange(n_buckets), mins, maxs, color="#5b6a8c", linewidth=0)
    ax.set_ylim(-32768, 32767)
    ax.axis("off")
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _clip_card_html(clip: dict[str, Any], wav_bytes: bytes, pcm: np.ndarray) -> str:
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    sparkline_b64 = _render_sparkline_png_base64(pcm)
    kind = str(clip["kind"])
    label_raw = str(clip["label"])
    label = html.escape(label_raw)
    description = html.escape(str(clip["description"]))
    start_utc = html.escape(str(clip["start_utc"]))
    source_run = html.escape(str(clip["source_run"]))
    duration_s = html.escape(str(clip["duration_s"]))
    anchor_id = f"clip-{kind}-{_ANCHOR_SANITIZE_RE.sub('_', label_raw)}"
    kind_label = _CLIP_KIND_LABEL.get(kind, kind)
    kind_attr = html.escape(kind)
    # `data-start-utc`/`data-duration-s`: the
    # live-replay JS reads these to align this clip's own audio.currentTime onto the
    # shared `demo-data` trace time axis (`(new Date(data-start-utc) - t0) / 1000`)
    # when its "Follow live" button is clicked -- kept as machine-readable
    # attributes rather than re-parsing the already-escaped, human-formatted
    # `.clip-time` text.
    return (
        f'<div class="clip" id="{anchor_id}" data-kind="{kind_attr}" data-label="{label}" '
        f'data-run="{source_run}" data-start-utc="{start_utc}" data-duration-s="{duration_s}">\n'
        f'  <div class="clip-head"><span class="badge">{kind_label} {label}</span>'
        f'<span class="clip-time">{start_utc} · {source_run}</span></div>\n'
        f'  <div class="clip-wave-wrap">\n'
        f'    <img class="clip-wave" src="data:image/png;base64,{sparkline_b64}" alt="Waveform">\n'
        f'    <canvas class="clip-live-canvas" hidden></canvas>\n'
        f'    <div class="clip-live-cursor" hidden></div>\n'
        f"  </div>\n"
        f'  <audio controls preload="none" src="data:audio/wav;base64,{audio_b64}"></audio>\n'
        f'  <p class="clip-desc">{description}</p>\n'
        f'  <button class="clip-live-btn" type="button">&#9654; Follow live</button>\n'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Figure rendering (render-figures) -- real fusion-cache/WAV/demo-data reads,
# matplotlib. The real-data-touching counterpart of `extract_clips` (module
# docstring): writes four PNGs into `docs/demo/assets/figures/`, tracked in git like
# the clip WAVs, so `build_html` below can stay real-data-free (it only ever reads
# already-rendered PNG bytes off disk, the same kind of "assets/ only" read as its
# existing clip-WAV reads -- never touches the fusion cache or `ROWII_DATA_ROOT`
# itself). Not unit-tested (matplotlib rendering, disk IO) -- see the module
# docstring's own pure-helper/IO-touching split; the DATA feeding each figure
# (`top_k_abs_z_indices`, `column_zscore_stats`, `quantile_threshold`,
# `nearest_sorted_index`, `extract_window_samples`) is the separately tested part.
# ---------------------------------------------------------------------------

DEFAULT_FUSION_CACHE = REPO_ROOT / "results" / "cache" / f"{PU_RUN_NAME}--fusion.npz"
"""Only present in a checkout with real (gitignored) `results/` -- this task's own
worktree has none (module docstring's own `extract_clips`/`ROWII_DATA_ROOT` parallel:
a fresh checkout/worktree never has either). Pass `--fusion-cache` explicitly from
one that does not."""
DEFAULT_FIGURES_DIR = DEFAULT_ASSETS_DIR / "figures"
DEFAULT_FIGURE_STATE_CLIP = "state_cluster1.wav"
"""State 1 (pump operation, SCADA-verified 98.6% pure on this run -- see
`demo_live_template.html`'s own live-panel legend) -- the cleanest single-mode state
clip to anchor the Section-1 waveform/spectrogram/feature-bars figures to one real,
nameable moment."""
FIGURE_WINDOW_OFFSET_S = 4.0
"""Seconds into the 10 s state clip where the illustrated 1 s window starts -- clear
of both edges (`center_window`'s own dodge/collision slack sits at the clip
boundaries, module docstring)."""
FIGURE_WINDOW_DURATION_S = 1.0
TOP_K_FEATURES = 10

_FIG_BG = "#0f1420"  # --bg
_FIG_INK = "#e8ecf4"  # --ink
_FIG_MUTED = "#8b96ad"  # --muted
_FIG_GRID = "#2a3348"  # --grid
_FIG_ACCENT = "#4c8dff"  # --s0
_FIG_ALARM = "#ff5c5c"  # --alarm
"""Mirror `demo_live_template.html`'s own `:root` CSS custom properties so the four
figures blend into the dark page instead of looking like a foreign export."""


def _pyplot() -> ModuleType:
    """Local, Agg-backend-safe `matplotlib.pyplot` import shared by the four figure
    renderers below -- same rationale, and the same three-line sequence, as the
    pre-existing `_render_sparkline_png_base64` (that function's own docstring):
    matplotlib stays a required dependency (`pyproject.toml`) but only these cosmetic
    renderers need it, so the import (and Agg backend selection) stays scoped to
    callers that actually render a figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _strip_axes_frame(ax: Any, *, keep: str = "bottom") -> None:
    """Hide every spine except *keep* (or all of them, `keep=""`) and color the
    survivor `_FIG_GRID` -- the "minimal axes" look shared by all four
    figures below."""
    for side in ("top", "right", "left", "bottom"):
        spine = ax.spines[side]
        if side == keep:
            spine.set_color(_FIG_GRID)
        else:
            spine.set_visible(False)


def _render_waveform_png_bytes(
    samples: np.ndarray, sample_rate_hz: int, *, width_px: int = 320, height_px: int = 110
) -> bytes:
    """A minimal-axis 1 s waveform figure (amplitude vs. time) for the Section-1
    "raw signal" pipe-step tile -- real samples (one of this demo's own
    `docs/demo/assets/*.wav` clips), not synthetic. Unlike the pre-existing per-clip
    sparkline (`_render_sparkline_png_base64`, a bare axis-less envelope), this keeps
    a small time axis, distinguishing these four "real mini graphics" from that
    purely cosmetic sparkline.
    """
    plt = _pyplot()
    t = np.arange(len(samples)) / sample_rate_hz
    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax = fig.add_axes((0.10, 0.24, 0.88, 0.72))
    ax.plot(t, samples, color=_FIG_ACCENT, linewidth=0.6)
    ax.set_xlim(0, t[-1] if len(t) else 1.0)
    ax.set_yticks([])
    ax.set_xlabel("s", color=_FIG_MUTED, fontsize=8, labelpad=2)
    ax.tick_params(axis="x", colors=_FIG_MUTED, labelsize=7, length=2)
    _strip_axes_frame(ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return buf.getvalue()


def _render_spectrogram_png_bytes(
    samples: np.ndarray, sample_rate_hz: int, *, width_px: int = 320, height_px: int = 110
) -> bytes:
    """A minimal-axis spectrogram of the SAME 1 s window `_render_waveform_png_bytes`
    plots, for the Section-1 "1-s window" pipe-step tile -- `Axes.specgram` (Task
    A instruction: "matplotlib specgram is enough"), not a reproduction of this
    codebase's real feature/BEATs frontend (`rowii.features`/`rowii.encoders.ssl`
    elsewhere) -- a cosmetic pipeline-overview figure only.
    """
    plt = _pyplot()
    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax = fig.add_axes((0.11, 0.24, 0.86, 0.72))
    nfft = 512
    ax.specgram(samples, NFFT=nfft, Fs=sample_rate_hz, noverlap=nfft // 2, cmap="magma")
    ax.set_xlabel("s", color=_FIG_MUTED, fontsize=8, labelpad=2)
    ax.set_ylabel("Hz", color=_FIG_MUTED, fontsize=8, labelpad=2)
    ax.tick_params(colors=_FIG_MUTED, labelsize=7, length=2)
    _strip_axes_frame(ax, keep="")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return buf.getvalue()


def _render_feature_bars_png_bytes(
    names: Sequence[str], z_values: Sequence[float], *, width_px: int = 420, height_px: int = 190
) -> bytes:
    """Horizontal top-|z| feature-bar chart for the Section-1 "Features" pipe-step
    tile -- *names*/*z_values* are already selected+ordered by `top_k_abs_z_indices`
    (this function only draws; that selection is the separately unit-tested pure
    contract). Bar color encodes SIGN (this window's value above/below that column's
    own day-mean, `column_zscore_stats`); bar length encodes |z|. Wider than the other
    three figures (`width_px`): the longest real `shorten_feature_name` output (e.g.
    `"GenVib2.ch2.band_guide_vane_pass"`, ~33 chars) needs the extra room even at a
    small font, or it clips against the left edge of the canvas.
    """
    plt = _pyplot()
    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax = fig.add_axes((0.53, 0.05, 0.44, 0.88))
    y = np.arange(len(names))
    colors = [_FIG_ALARM if v >= 0 else _FIG_ACCENT for v in z_values]
    ax.barh(y, z_values, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=_FIG_INK, fontsize=6.3)
    ax.invert_yaxis()
    ax.axvline(0, color=_FIG_GRID, linewidth=0.8)
    ax.set_xlabel("z", color=_FIG_MUTED, fontsize=8, labelpad=2)
    ax.tick_params(axis="x", colors=_FIG_MUTED, labelsize=7, length=2)
    ax.tick_params(axis="y", length=0)
    _strip_axes_frame(ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return buf.getvalue()


def _render_histogram_png_bytes(
    scores: Sequence[float],
    threshold: float,
    alpha: float,
    *,
    width_px: int = 320,
    height_px: int = 190,
) -> bytes:
    """Score histogram + the pooled `quantile_threshold` line for the Section-1
    "split-conformal threshold" pipe-step tile -- see that helper's own docstring for
    why the single line is a pooled-across-the-day illustrative simplification, not
    the literal per-state operational threshold.
    """
    plt = _pyplot()
    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax = fig.add_axes((0.09, 0.24, 0.88, 0.72))
    ax.hist(scores, bins=30, color=_FIG_ACCENT, alpha=0.85)
    ax.axvline(threshold, color=_FIG_ALARM, linewidth=1.1)
    # Anchored to the AXES corner (`ax.transAxes`), not the threshold's own data
    # x-position: the pooled quantile can land anywhere in [min(scores), max(scores)]
    # depending on the day's own score distribution, and a data-coordinate label can
    # run past the canvas edge when the threshold sits close to the right tail (as it
    # does on the real 080726-pu_strikes/audio-beats run this demo embeds).
    ax.text(
        0.97,
        0.94,
        f"Conformal threshold (α={alpha:g})\n≈ {threshold:.3f}",
        transform=ax.transAxes,
        color=_FIG_ALARM,
        fontsize=6.3,
        va="top",
        ha="right",
    )
    ax.set_xlabel("Score", color=_FIG_MUTED, fontsize=8, labelpad=2)
    ax.tick_params(axis="x", colors=_FIG_MUTED, labelsize=7, length=2)
    ax.set_yticks([])
    _strip_axes_frame(ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return buf.getvalue()


def render_figures(
    assets_dir: Path,
    fusion_cache_path: Path,
    source_demo_path: Path,
    state_clip_name: str,
    out_dir: Path,
) -> dict[str, Path]:
    """Render the four Section-1 pipeline-overview PNGs (waveform, spectrogram,
    feature-bars, score-histogram) and write them into *out_dir*. Reads exactly one
    already-extracted clip WAV from *assets_dir* (no new burst-file access -- Task
    A #1: "no fresh burst access needed"), the fusion feature cache (real data,
    *fusion_cache_path* -- see `DEFAULT_FUSION_CACHE`'s own docstring on why this
    needs to be passed explicitly from a checkout that actually has it), and the
    `demo-data` trace embedded in *source_demo_path* (the same source `build_html`
    already copies its own data block from, byte for byte).

    The SAME real moment backs three of the four figures: `state_clip_name`'s own
    start time + `FIGURE_WINDOW_OFFSET_S` picks an absolute instant; the closest
    ACTUALLY-SCORED `demo-data` trace row to that instant (`nearest_sorted_index` --
    the fusion cache's own grid is dense/uniform, but the trace is sparse, excluding
    calibration-consumed/unscored windows) fixes the exact second used for both the
    waveform/spectrogram (from the WAV) and the feature-bars window (the fusion
    cache's row for that same second) -- "this exact moment's raw signal AND its 243
    features", not two arbitrary, unrelated windows.

    Returns the four PNG paths, keyed by figure name (`"waveform"`, `"spectrogram"`,
    `"features"`, `"histogram"`).

    Raises:
        ValueError: if *state_clip_name* is not in *assets_dir*'s manifest, or the
            trace-matched second maps to an out-of-range or `valid_mask`-false fusion
            cache row (should not happen on real data -- the trace only ever contains
            scored, hence valid, windows -- but this fails loudly rather than
            silently drawing a chart from garbage).
    """
    manifest = json.loads((assets_dir / "manifest.json").read_text(encoding="utf-8"))
    clip_entries = {c["file"]: c for c in manifest["clips"]}
    if state_clip_name not in clip_entries:
        available = ", ".join(sorted(clip_entries))
        raise ValueError(f"{state_clip_name!r} not in manifest.json (have: {available})")
    clip_meta = clip_entries[state_clip_name]
    clip_start_utc = datetime.fromisoformat(str(clip_meta["start_utc"]))

    rate_hz, pcm = wavfile.read(assets_dir / state_clip_name)
    window_samples = extract_window_samples(
        np.asarray(pcm), int(rate_hz), FIGURE_WINDOW_OFFSET_S, FIGURE_WINDOW_DURATION_S
    )

    demo_data = json.loads(_extract_demo_data_block(source_demo_path))
    t0_utc = datetime.fromisoformat(str(demo_data["t0_utc"]))
    alpha = float(demo_data["alpha"])
    trace = demo_data["trace"]  # [[t_s, score, state, alarm, p], ...], sorted by t_s
    trace_times = [float(row[0]) for row in trace]
    scores = [float(row[1]) for row in trace]

    target_s = (clip_start_utc - t0_utc).total_seconds() + FIGURE_WINDOW_OFFSET_S
    trace_idx = nearest_sorted_index(trace_times, target_s)
    feature_window_s = round(trace_times[trace_idx])

    with np.load(fusion_cache_path, allow_pickle=True) as npz:
        features = np.asarray(npz["features"])
        valid_mask = np.asarray(npz["valid_mask"])
        feature_names = [str(n) for n in npz["feature_names"]]
        grid_t0_ns = int(npz["grid_t0_ns"][0])

    grid_t0_utc = datetime.fromtimestamp(grid_t0_ns / 1e9, tz=UTC)
    fusion_row_idx = feature_window_s + round((t0_utc - grid_t0_utc).total_seconds())
    if not (0 <= fusion_row_idx < features.shape[0]):
        raise ValueError(
            f"feature window at t0+{feature_window_s}s (fusion cache row "
            f"{fusion_row_idx}) is outside the fusion cache's own "
            f"{features.shape[0]} rows"
        )
    if not valid_mask[fusion_row_idx]:
        raise ValueError(
            f"fusion cache row {fusion_row_idx} (t0+{feature_window_s}s) is marked "
            f"invalid -- the demo-data trace point it was matched to should always "
            f"be a genuinely scored (hence valid) window"
        )

    mean, std = column_zscore_stats(features, valid_mask)
    row = features[fusion_row_idx]
    top_idx = top_k_abs_z_indices(row, mean, std, TOP_K_FEATURES)
    bar_names = [shorten_feature_name(feature_names[i]) for i in top_idx]
    bar_z = [float((row[i] - mean[i]) / std[i]) if std[i] != 0.0 else 0.0 for i in top_idx]

    threshold = quantile_threshold(scores, alpha)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_bytes_by_name = {
        "waveform": _render_waveform_png_bytes(window_samples, int(rate_hz)),
        "spectrogram": _render_spectrogram_png_bytes(window_samples, int(rate_hz)),
        "features": _render_feature_bars_png_bytes(bar_names, bar_z),
        "histogram": _render_histogram_png_bytes(scores, threshold, alpha),
    }
    paths: dict[str, Path] = {}
    for name, png_bytes in png_bytes_by_name.items():
        path = out_dir / f"{name}.png"
        path.write_bytes(png_bytes)
        paths[name] = path
        logger.info("make_demo_assets: wrote %s (%d bytes)", path, len(png_bytes))
    return paths


def build_html(
    assets_dir: Path, template_path: Path, source_demo_path: Path, figures_dir: Path, out_path: Path
) -> Path:
    """Render `demo_live.html` from *template_path* by injecting the clip player
    cards (from *assets_dir*'s `manifest.json` + WAVs), the `demo_080726_pu.html`
    data block (from *source_demo_path*), and the four Section-1 pipeline PNGs (from
    *figures_dir* -- `render_figures`' own output) -- reproducible from
    `docs/demo/assets/` alone, no real ROWII data access (reading an already-rendered
    PNG off disk is the same kind of "assets/ only" read as the existing clip-WAV
    reads just above, not a new real-data touch).
    """
    manifest = json.loads((assets_dir / "manifest.json").read_text(encoding="utf-8"))
    cards = []
    for clip in manifest["clips"]:
        wav_path = assets_dir / clip["file"]
        wav_bytes = wav_path.read_bytes()
        _rate, pcm = wavfile.read(wav_path)
        cards.append(_clip_card_html(clip, wav_bytes, pcm))
    clips_html = "\n".join(cards)

    data_block = _extract_demo_data_block(source_demo_path)

    template = template_path.read_text(encoding="utf-8")
    figure_placeholders = {
        "waveform": "__FIG_WAVEFORM__",
        "spectrogram": "__FIG_SPECTROGRAM__",
        "features": "__FIG_FEATURES__",
        "histogram": "__FIG_HISTOGRAM__",
    }
    required_placeholders = ["__CLIPS_HTML__", "__DATA__", *figure_placeholders.values()]
    missing = [p for p in required_placeholders if p not in template]
    if missing:
        raise ValueError(f"{template_path} is missing placeholder(s): {', '.join(missing)}")

    rendered = template.replace("__CLIPS_HTML__", clips_html).replace("__DATA__", data_block)
    for fig_name, placeholder in figure_placeholders.items():
        png_path = figures_dir / f"{fig_name}.png"
        if not png_path.exists():
            raise FileNotFoundError(
                f"{png_path} not found -- run `render-figures` before `build-html`"
            )
        fig_b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
        rendered = rendered.replace(placeholder, fig_b64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    logger.info("make_demo_assets: wrote %s (%d clip(s))", out_path, len(manifest["clips"]))
    return out_path


# ---------------------------------------------------------------------------
# Control-room dashboard assembly (build-dashboard) -- real results/ CSV+parquet+
# markdown reads (once-calibrated monitor/eval_events reports, docs/groundtruth
# events) plus the already-extracted docs/demo/assets/ clips; never touches
# ROWII_DATA_ROOT or a model. Mirrors build_html's own "real artifacts in, one
# self-contained HTML out" shape, so the two subcommands stay symmetric, but reads
# a different (newer) artifact family -- see ONCE_CALIBRATED_DIR's own
# docstring for why. Not unit-tested (real CSV/parquet/markdown disk reads,
# `json.dumps` assembly) -- the DATA feeding it (`parse_state_table`,
# `parse_event_summary_table`, `matching_event_kind`, `state_display_name`) is the
# separately-tested part, same pure/IO-touching split as this module's other two
# subcommands (module docstring).
# ---------------------------------------------------------------------------


def _monitor_dir(run: str, representation: str, mode: str) -> Path:
    return ONCE_CALIBRATED_DIR / representation / "monitor" / run / mode


def _eval_events_dir(run: str, representation: str, mode: str) -> Path:
    return ONCE_CALIBRATED_DIR / representation / "eval_events" / run / mode


def _to_pydatetime_quiet(ts: pd.Timestamp) -> datetime:
    """`ts.to_pydatetime()`, silencing pandas' "Discarding nonzero nanoseconds"
    UserWarning -- every real timestamp this dashboard parses carries a fixed
    sub-microsecond DAQ offset (e.g. `...11:46:03.410000115+00:00`); `datetime` only
    has microsecond resolution, so the truncation is expected and harmless
    (negligible next to this dashboard's 1 s window grid). Same rationale as
    `_load_segments_csv`'s own local suppression above, applied per call site here
    instead of one large indented block (this module's several dashboard call
    sites each convert only a handful of timestamps, not a bulk column)."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Discarding nonzero nanoseconds", category=UserWarning
        )
        return cast(datetime, ts.to_pydatetime())


def _load_alarm_intervals(alarm_segments_csv: Path) -> list[tuple[datetime, datetime]]:
    """A `monitor.py`-written `alarm_segments.csv` (`start_utc,end_utc,duration_s`),
    parsed into `(start, end)` UTC-datetime pairs -- the shape `has_collision`/
    `matching_event_kind` (this module's own pure helpers) both consume, reused
    here for the Alarm-Feed's cross-representation "did vibration ALSO alarm near
    this episode" check (`_load_session`)."""
    df = pd.read_csv(alarm_segments_csv)
    starts = pd.to_datetime(df["start_utc"], utc=True)
    ends = pd.to_datetime(df["end_utc"], utc=True)
    return [
        (_to_pydatetime_quiet(s), _to_pydatetime_quiet(e))
        for s, e in zip(starts, ends, strict=True)
    ]


def _load_session(run: str, events_csv: Path) -> dict[str, Any]:
    """One `sessions[<run>]` entry of the dashboard payload: the once-calibrated/
    `DASHBOARD_THRESHOLD_MODE` state timeline, per-window score/alarm/near-
    transition trace, ground-truth strike events, and per-episode Alarm-Feed cards
    for *run* -- everything the control-room UI needs to replay this ONE session,
    time-relative to that session's own first window (`t0_utc`, matching
    `demo_live_template.html`'s existing `t0`/`t_s` convention)."""
    mdir = _monitor_dir(run, DASHBOARD_REPRESENTATION, DASHBOARD_THRESHOLD_MODE)

    segments_df = pd.read_csv(mdir / "segments.csv")
    segments_df["start_utc"] = pd.to_datetime(segments_df["start_utc"], utc=True)
    segments_df["end_utc"] = pd.to_datetime(segments_df["end_utc"], utc=True)
    t0 = segments_df["start_utc"].min()
    t0_ns = int(t0.value)
    duration_s = (int(segments_df["end_utc"].max().value) - t0_ns) / 1e9

    state_table = parse_state_table((mdir / "monitor_notes.md").read_text(encoding="utf-8"))

    alarms_df = pd.read_parquet(mdir / "alarms.parquet")
    scored = alarms_df.loc[alarms_df["role"] == "scored"].sort_values("t_utc_ns")
    if scored.empty:
        raise ValueError(f"{run}: no scored windows in {mdir / 'alarms.parquet'}")

    states: dict[str, Any] = {}
    for raw_state_id in sorted(scored["state"].unique().tolist()):
        state_id = int(raw_state_id)
        info = state_table.get(state_id)
        if info is None:
            raise ValueError(
                f"{run}: state {state_id} has scored windows but no row in "
                f"{mdir / 'monitor_notes.md'}'s per-state table"
            )
        threshold = info["threshold"]
        states[str(state_id)] = {
            "name": info["name"],
            "name_label": state_display_name(info["name"]),
            "threshold": (
                None if threshold is None or math.isinf(threshold) else round(threshold, 6)
            ),
            "low_confidence": info["low_confidence"],
        }

    segments = [
        {
            "start_s": round((int(row.start_utc.value) - t0_ns) / 1e9, 3),
            "end_s": round((int(row.end_utc.value) - t0_ns) / 1e9, 3),
            "state": int(row.cluster_id),
        }
        for row in segments_df.itertuples()
    ]

    trace = [
        [
            round((int(t_ns) - t0_ns) / 1e9, 3),
            round(float(score), 6),
            int(state),
            bool(alarm),
            round(float(p_value), 6),
            bool(near_transition),
        ]
        for t_ns, score, state, alarm, p_value, near_transition in zip(
            scored["t_utc_ns"],
            scored["score"],
            scored["state"],
            scored["alarm"],
            scored["p_value"],
            scored["near_transition"],
            strict=True,
        )
    ]

    events_df = _load_events_csv(events_csv)
    events = [
        {
            "start_s": round((int(row.start_utc.value) - t0_ns) / 1e9, 3),
            "end_s": round((int(row.end_utc.value) - t0_ns) / 1e9, 3),
            "kind": str(row.kind),
        }
        for row in events_df.itertuples()
    ]
    event_tuples = [
        (_to_pydatetime_quiet(row.start_utc), _to_pydatetime_quiet(row.end_utc), str(row.kind))
        for row in events_df.itertuples()
    ]

    vib_dir = _monitor_dir(run, DASHBOARD_VIB_REPRESENTATION, DASHBOARD_THRESHOLD_MODE)
    vib_intervals = _load_alarm_intervals(vib_dir / "alarm_segments.csv")

    alarm_seg_df = pd.read_csv(mdir / "alarm_segments.csv")
    alarm_seg_df["start_utc"] = pd.to_datetime(alarm_seg_df["start_utc"], utc=True)
    alarm_seg_df["end_utc"] = pd.to_datetime(alarm_seg_df["end_utc"], utc=True)

    scored_t_ns = scored["t_utc_ns"].to_numpy()
    scored_score = scored["score"].to_numpy()
    scored_p = scored["p_value"].to_numpy()
    scored_state = scored["state"].to_numpy()
    scored_near = scored["near_transition"].to_numpy()

    episodes = []
    for row in alarm_seg_df.itertuples():
        start_ns = int(row.start_utc.value)
        end_ns = int(row.end_utc.value)
        # Half-open [start_ns, end_ns) over WINDOW STARTS -- same convention
        # `alarm_segments.csv`'s own start/end already encode (module docstring's
        # `_extract_mono_clip`/`utc_window_to_sample_range` note the same "window
        # STARTS" convention elsewhere in this module).
        lo = int(np.searchsorted(scored_t_ns, start_ns, side="left"))
        hi = int(np.searchsorted(scored_t_ns, end_ns, side="left"))
        if hi <= lo:
            raise ValueError(
                f"{run}: alarm episode [{row.start_utc}, {row.end_utc}) matches no "
                f"scored window -- alarm_segments.csv and alarms.parquet should "
                f"always agree (same monitor.py run)"
            )
        window_states = scored_state[lo:hi]
        state_id = int(pd.Series(window_states).mode().iloc[0])
        threshold = states.get(str(state_id), {}).get("threshold")
        peak_score = float(scored_score[lo:hi].max())
        ep_start_dt = _to_pydatetime_quiet(row.start_utc)
        ep_end_dt = _to_pydatetime_quiet(row.end_utc)
        episodes.append(
            {
                "start_s": round((start_ns - t0_ns) / 1e9, 3),
                "end_s": round((end_ns - t0_ns) / 1e9, 3),
                "n": int(hi - lo),
                "state": state_id,
                "peak_score": round(peak_score, 6),
                "min_p": round(float(scored_p[lo:hi].min()), 6),
                "threshold": threshold,
                "ratio": round(peak_score / threshold, 2) if threshold else None,
                "near_transition": bool(scored_near[lo:hi].any()),
                "gt": matching_event_kind(
                    ep_start_dt, ep_end_dt, event_tuples, DASHBOARD_EVENT_TOLERANCE_S
                ),
                "vib_coincident": has_collision(ep_start_dt, ep_end_dt, vib_intervals, pad_s=0.0),
            }
        )

    event_notes_path = (
        _eval_events_dir(run, DASHBOARD_REPRESENTATION, DASHBOARD_THRESHOLD_MODE) / "event_notes.md"
    )
    event_summary = parse_event_summary_table(event_notes_path.read_text(encoding="utf-8"))

    return {
        "run": run,
        "label": _SESSION_LABEL[run],
        "t0_utc": _to_pydatetime_quiet(t0).isoformat(),
        "duration_s": round(duration_s, 3),
        "representation": DASHBOARD_REPRESENTATION,
        "threshold_mode": DASHBOARD_THRESHOLD_MODE,
        "states": states,
        "segments": segments,
        "trace": trace,
        "events": events,
        "episodes": episodes,
        "event_summary": event_summary,
    }


def build_dashboard_data(assets_dir: Path) -> dict[str, Any]:
    """The full control-room dashboard payload: both sessions (`_load_session`) plus
    every clip in *assets_dir*'s `manifest.json` (module docstring's existing six --
    reused byte-for-byte as WAV+sparkline, never re-extracted), each clip's
    `start_s` now expressed relative to ITS OWN session's `t0_utc` so the replay
    bar can seek any session's playhead straight to it.
    """
    sessions = {
        run: _load_session(run, events_csv)
        for run, events_csv in ((ST_RUN_NAME, ST_EVENTS_CSV), (PU_RUN_NAME, PU_EVENTS_CSV))
    }

    manifest = json.loads((assets_dir / "manifest.json").read_text(encoding="utf-8"))
    clips = []
    for clip in manifest["clips"]:
        run = str(clip["source_run"])
        if run not in sessions:
            raise ValueError(f"clip {clip['file']!r}: unknown source_run {run!r}")
        wav_path = assets_dir / str(clip["file"])
        wav_bytes = wav_path.read_bytes()
        _rate, pcm = wavfile.read(wav_path)
        start_utc_dt = datetime.fromisoformat(str(clip["start_utc"]))
        t0_dt = datetime.fromisoformat(sessions[run]["t0_utc"])
        clips.append(
            {
                "file": clip["file"],
                "kind": clip["kind"],
                "label": clip["label"],
                "description": clip["description"],
                "run": run,
                "start_s": round((start_utc_dt - t0_dt).total_seconds(), 3),
                "duration_s": clip["duration_s"],
                "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
                "sparkline_b64": _render_sparkline_png_base64(pcm),
            }
        )
    clips.sort(key=lambda c: (str(c["run"]), float(c["start_s"])))

    return {"sessions": sessions, "clips": clips, "default_session": DASHBOARD_DEFAULT_SESSION}


def build_dashboard(assets_dir: Path, template_path: Path, out_path: Path) -> Path:
    """Render `demo_dashboard.html` from *template_path* by injecting
    `build_dashboard_data`'s payload as one JSON blob -- reproducible from
    `docs/demo/assets/` + the once-calibrated/groundtruth artifacts alone, exactly
    like `build_html`'s own contract (this module's other HTML-assembling
    subcommand)."""
    data = build_dashboard_data(assets_dir)
    payload = json.dumps(data, ensure_ascii=False)

    template = template_path.read_text(encoding="utf-8")
    placeholder = "__DASHBOARD_DATA__"
    if placeholder not in template:
        raise ValueError(f"{template_path} is missing placeholder {placeholder}")
    rendered = template.replace(placeholder, payload)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    logger.info(
        "make_demo_assets: wrote %s (%d session(s), %d clip(s))",
        out_path,
        len(data["sessions"]),
        len(data["clips"]),
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the interactive live-audio demo's assets: real 080726 WAV clips + "
            "manifest (extract-clips), then the self-contained demo_live.html "
            "(build-html)."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser(
        "extract-clips",
        help="Cut the six demo WAV clips from real 080726 data and write manifest.json.",
    )
    extract.add_argument(
        "--out-dir", type=Path, default=DEFAULT_ASSETS_DIR,
        help=f"Output directory for WAVs + manifest.json (default: {DEFAULT_ASSETS_DIR}).",
    )

    render = sub.add_parser(
        "render-figures",
        help="Render the four Section-1 pipeline PNGs (waveform, spectrogram, "
        "feature-bars, score-histogram) from real assets + the fusion cache.",
    )
    render.add_argument(
        "--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR,
        help=f"Directory holding the WAVs + manifest.json (default: {DEFAULT_ASSETS_DIR}).",
    )
    render.add_argument(
        "--fusion-cache", type=Path, default=DEFAULT_FUSION_CACHE,
        help=f"Fusion feature cache .npz for the {PU_RUN_NAME} run (default: "
             f"{DEFAULT_FUSION_CACHE} -- only present in a checkout with real, "
             f"gitignored results/; pass explicitly from a fresh worktree).",
    )
    render.add_argument(
        "--source-demo", type=Path, default=DEFAULT_SOURCE_DEMO,
        help=f"Existing demo page to read the <script id=demo-data> trace/alpha from "
             f"(default: {DEFAULT_SOURCE_DEMO}).",
    )
    render.add_argument(
        "--state-clip", type=str, default=DEFAULT_FIGURE_STATE_CLIP,
        help="Which assets-dir state-clip WAV backs the waveform/spectrogram/"
             f"feature-window figures (default: {DEFAULT_FIGURE_STATE_CLIP}).",
    )
    render.add_argument(
        "--out-dir", type=Path, default=DEFAULT_FIGURES_DIR,
        help=f"Output directory for the four PNGs (default: {DEFAULT_FIGURES_DIR}).",
    )

    build = sub.add_parser(
        "build-html", help="Render demo_live.html from the template + assets + manifest."
    )
    build.add_argument(
        "--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR,
        help=f"Directory holding the WAVs + manifest.json (default: {DEFAULT_ASSETS_DIR}).",
    )
    build.add_argument(
        "--template", type=Path, default=DEFAULT_TEMPLATE,
        help=f"HTML template with __CLIPS_HTML__/__DATA__/__FIG_*__ placeholders "
             f"(default: {DEFAULT_TEMPLATE}).",
    )
    build.add_argument(
        "--source-demo", type=Path, default=DEFAULT_SOURCE_DEMO,
        help=f"Existing demo page to copy the <script id=demo-data> block from "
             f"(default: {DEFAULT_SOURCE_DEMO}).",
    )
    build.add_argument(
        "--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR,
        help=f"Directory holding the four render-figures PNGs (default: "
             f"{DEFAULT_FIGURES_DIR}).",
    )
    build.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_HTML,
        help=f"Output HTML path (default: {DEFAULT_OUT_HTML}).",
    )

    dashboard = sub.add_parser(
        "build-dashboard",
        help="Render demo_dashboard.html (control-room view) from the once-calibrated "
        "monitor reports + groundtruth events + the existing demo clips.",
    )
    dashboard.add_argument(
        "--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR,
        help=f"Directory holding the WAVs + manifest.json (default: {DEFAULT_ASSETS_DIR}).",
    )
    dashboard.add_argument(
        "--template", type=Path, default=DEFAULT_DASHBOARD_TEMPLATE,
        help=f"HTML template with a __DASHBOARD_DATA__ placeholder (default: "
             f"{DEFAULT_DASHBOARD_TEMPLATE}).",
    )
    dashboard.add_argument(
        "--out", type=Path, default=DEFAULT_DASHBOARD_OUT,
        help=f"Output HTML path (default: {DEFAULT_DASHBOARD_OUT}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "extract-clips":
        cfg = load_config()
        clips = extract_clips(cfg, args.out_dir)
        print(f"make_demo_assets: wrote {len(clips)} clip(s) to {args.out_dir}")
        return 0

    if args.command == "render-figures":
        paths = render_figures(
            args.assets_dir, args.fusion_cache, args.source_demo, args.state_clip, args.out_dir
        )
        print(f"make_demo_assets: wrote {len(paths)} figure(s) to {args.out_dir}")
        return 0

    if args.command == "build-dashboard":
        out_path = build_dashboard(args.assets_dir, args.template, args.out)
        print(f"make_demo_assets: wrote {out_path}")
        return 0

    out_path = build_html(
        args.assets_dir, args.template, args.source_demo, args.figures_dir, args.out
    )
    print(f"make_demo_assets: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
