"""Build real-audio assets for the interactive live demo (`docs/demo/demo_live.html`):
short WAV clips cut from the 080726 induced-Schonhammer-strike campaign day, plus the
manifest describing them, plus (second subcommand) the self-contained demo page itself.

Two subcommands:

    extract-clips   Read real 080726 data (via the existing `rowii.io.dataset.discover`
                     / `rowii.io.gantner.read_gantner` seams) and write
                     `docs/demo/assets/*.wav` + `docs/demo/assets/manifest.json`.
    build-html       Template-inject the clips (as base64 data-URIs + waveform
                     sparklines) and the EXISTING `demo_080726_pu.html` state/score/
                     report data model into `docs/demo/demo_live_template.html`,
                     producing `docs/demo/demo_live.html`. Never touches real data --
                     pure file/template assembly, so this half is fast and reproducible
                     from `docs/demo/assets/` alone.

Clip selection (six clips total):

    - One STATE clip per cluster id (`>= 0`) that actually appears in
      `results/pillar3/080726-pu_strikes/audio-beats-a0.01/segments.csv`: the LONGEST
      segment of that cluster, 10 s cut from its middle (`center_window`), nudged away
      from any induced strike (`dodge_collision` against `docs/groundtruth/
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
import gc
import html
import json
import logging
import re
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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

CLIP_DURATION_S = 10.0
COLLISION_PAD_S = 10.0
"""Padding (seconds, both sides) around every induced-strike event a STATE clip's
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
    """One German sentence."""

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
            f"Unüberwacht erkannter Zustand {cluster_id} ({n_windows} Fenster an diesem "
            f"Messtag, 1-s-Raster) – 10 s aus der Mitte des längsten zusammenhängenden "
            f"Segments dieses Zustands, gegen die Schlag-Ground-Truth geprüft."
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
                'Induzierter Schonhammer-Schlag "plate-tur_0" (Referenzplatte '
                "Turbinenseite, 0°) während des Pumpbetriebs – SCADA-bestätigt bei "
                "ca. −279 MW / −377.8 U/min."
            ),
        ),
        _StrikeTarget(
            filename="strike_standstill_plate_gen_0.wav",
            run=st_run,
            offset_ns=run_utc_offset_ns(st_run),
            events=st_events,
            gt_kind="plate-gen_0",
            description=(
                'Induzierter Schonhammer-Schlag "plate-gen_0" (Referenzplatte '
                "Generatorseite, 0°) im Stillstand (Kalibrierungs-Session)."
            ),
        ),
        _StrikeTarget(
            filename="strike_standstill_vane_sweep.wav",
            run=st_run,
            offset_ns=run_utc_offset_ns(st_run),
            events=st_events,
            gt_kind="vane-sweep",
            description=(
                "Vane-Sweep – strukturübertragene Anregung am Leitschaufeldeckel, "
                "erste 10 s der rund dreiminütigen Sweep-Session (Stillstand)."
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

_CLIP_KIND_LABEL_DE = {"state": "Zustand", "strike": "Schlag"}
_DEMO_DATA_RE = re.compile(
    r'<script id="demo-data" type="application/json">(.*?)</script>', re.DOTALL
)
_ANCHOR_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _extract_demo_data_block(source_demo_path: Path) -> str:
    """The raw JSON text inside `<script id="demo-data">...</script>` of an EXISTING
    demo page (`demo_080726_pu.html`) -- copied byte-for-byte, never re-serialized
    (task instruction: "regeneriere die Daten NICHT"), so `demo_live.html` shows
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
    "Anhören" section. matplotlib is imported LOCALLY (not at module level): it is a
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
    anchor_id = f"clip-{kind}-{_ANCHOR_SANITIZE_RE.sub('_', label_raw)}"
    kind_de = _CLIP_KIND_LABEL_DE.get(kind, kind)
    kind_attr = html.escape(kind)
    return (
        f'<div class="clip" id="{anchor_id}" data-kind="{kind_attr}" data-label="{label}" '
        f'data-run="{source_run}">\n'
        f'  <div class="clip-head"><span class="badge">{kind_de} {label}</span>'
        f'<span class="clip-time">{start_utc} · {source_run}</span></div>\n'
        f'  <img class="clip-wave" src="data:image/png;base64,{sparkline_b64}" alt="Waveform">\n'
        f'  <audio controls preload="none" src="data:audio/wav;base64,{audio_b64}"></audio>\n'
        f'  <p class="clip-desc">{description}</p>\n'
        f"</div>"
    )


def build_html(
    assets_dir: Path, template_path: Path, source_demo_path: Path, out_path: Path
) -> Path:
    """Render `demo_live.html` from *template_path* by injecting the clip player
    cards (from *assets_dir*'s `manifest.json` + WAVs) and the `demo_080726_pu.html`
    data block (from *source_demo_path*) -- reproducible from `docs/demo/assets/`
    alone, no real ROWII data access.
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
    if "__CLIPS_HTML__" not in template or "__DATA__" not in template:
        raise ValueError(f"{template_path} is missing the __CLIPS_HTML__ or __DATA__ placeholder")
    rendered = template.replace("__CLIPS_HTML__", clips_html).replace("__DATA__", data_block)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    logger.info("make_demo_assets: wrote %s (%d clip(s))", out_path, len(manifest["clips"]))
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

    build = sub.add_parser(
        "build-html", help="Render demo_live.html from the template + assets + manifest."
    )
    build.add_argument(
        "--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR,
        help=f"Directory holding the WAVs + manifest.json (default: {DEFAULT_ASSETS_DIR}).",
    )
    build.add_argument(
        "--template", type=Path, default=DEFAULT_TEMPLATE,
        help=f"HTML template with __CLIPS_HTML__/__DATA__ placeholders "
             f"(default: {DEFAULT_TEMPLATE}).",
    )
    build.add_argument(
        "--source-demo", type=Path, default=DEFAULT_SOURCE_DEMO,
        help=f"Existing demo page to copy the <script id=demo-data> block from "
             f"(default: {DEFAULT_SOURCE_DEMO}).",
    )
    build.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_HTML,
        help=f"Output HTML path (default: {DEFAULT_OUT_HTML}).",
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

    out_path = build_html(args.assets_dir, args.template, args.source_demo, args.out)
    print(f"make_demo_assets: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
