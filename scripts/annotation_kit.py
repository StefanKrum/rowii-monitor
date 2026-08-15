"""Strike-annotation kit for the 08.07.2026 induced-Schonhammer-strike campaign:
upgrades the existing MINUTE-level ground truth (`docs/groundtruth/080726_events_
{st,pu}.csv`, 13 events each) to SECONDS-level per-strike UTC timestamps -- needed
for per-strike TPR and first-alarm-latency evaluation, neither of which is
computable from a minute-wide interval alone.

Two subcommands:

    build      For every ground-truth event: cut a +/-15 s-padded WAV clip from
               BOTH mic streams (`RAWGeneratorMic__0`/`RAWTurbineMic__1`, channel
               0 each -- `make_demo_assets.MONO_CHANNEL_INDEX`'s own convention),
               peak-normalized and resampled exactly like `make_demo_assets`'
               demo clips (`_resample_to_target`/`_write_clip_wav`, reused
               unchanged); render one 0-20 kHz log-magnitude spectrogram PNG per
               stream from the RAW (native ~50 kHz, NOT resampled) samples --
               resampling to the 16 kHz WAV rate would Nyquist-limit the view to
               8 kHz, well under the requested 20 kHz band; and write a
               per-session `annotation_template_<session>.csv` (offset/
               confidence/notes columns left empty) plus a self-contained
               `index.html` for Stefan to look/listen and fill the CSV from.
    compile    Turn a FILLED `annotation_template_<session>.csv` into absolute-
               UTC per-strike rows (`session,event_id,kind,strike_no,strike_utc,
               confidence,notes`), validating every offset (>= 0, <= snippet
               duration, strictly increasing per event) before writing anything.

Boundary-crossing windows (this task's own generalization of `make_demo_assets.
_extract_mono_clip`, which is hardcoded to ONE stream and ONE burst file): this
campaign's burst files are ~12 min; the +/-15 s-padded windows here range ~90 s
(single-strike events) to ~3.5 min (the vane-sweep event) and DO straddle a
burst-file boundary on real data -- verified against the actual 080726 delivery:
the ST vane-sweep window and three PU plate-strike windows each cross exactly one
boundary. `files_covering_window`/`extract_stream_clip` stitch however many
adjacent burst files a window needs, loading at most one file's full array into
memory at a time (`del`/`gc.collect()` before the next -- same discipline as
`make_demo_assets._extract_mono_clip`'s own docstring). A window reaching past
the earliest/latest available file (or a genuine gap between two files) never
raises: the clip covers whatever overlap exists and the shortfall is recorded
(`StreamClip.clamped`/`.note`), surfaced into the template's own `notes` column
so Stefan sees it before annotating that event.

Pure logic (window arithmetic, burst-file selection, offset<->UTC conversion,
template validation) is unit-tested with synthetic fixtures in
`tests/test_annotation_kit.py`, including `extract_stream_clip`'s own
boundary-stitch/clamp behaviour against synthetic Gantner files (`tests/fixtures/
gantner_builder.py`) -- a deliberate, documented extension of `make_demo_assets`'
own "pure vs IO-touching" test split (see that module's docstring): the
boundary-crossing stitch is exactly the new, highest-risk logic this task adds,
and synthetic `.dat` files make it fully testable without `ROWII_DATA_ROOT`. Real
reads against `ROWII_DATA_ROOT`, WAV writes, spectrogram PNG rendering and HTML
assembly are exercised by actually running `build` against the real campaign
data instead, not by a unit test.
"""
from __future__ import annotations

import argparse
import gc
import html
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPTS_DIR.parent / "src"
for _extra_path in (str(_SCRIPTS_DIR), str(_SRC_DIR)):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

import make_demo_assets  # noqa: E402

from rowii.config import Config, load_config  # noqa: E402
from rowii.io.dataset import BurstFile, RecordingIndex, discover, run_utc_offset_ns  # noqa: E402
from rowii.io.gantner import read_gantner  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = _SCRIPTS_DIR.parent
DEFAULT_KIT_DIR = REPO_ROOT / "results" / "annotation-kit" / "080726"

SNIPPET_PAD_S = 15.0
"""Seconds padded on EACH side of a ground-truth event's own [start, end) span to
get the annotation snippet window -- task spec's value, not independently chosen.
A 60 s single-strike event -> a 90 s snippet; the 180 s vane-sweep event -> a
210 s (3.5 min) snippet."""

TUR_STREAM = "RAWTurbineMic__1"
"""Turbine-side mic stream name (`rowii.io.dataset._STREAMS`). The generator-side
counterpart is `make_demo_assets.MONO_STREAM` ("RAWGeneratorMic__0") -- reused
directly rather than redefined here."""

_STREAM_NAME_BY_KEY: dict[str, str] = {"gen": make_demo_assets.MONO_STREAM, "tur": TUR_STREAM}
_STREAM_LABEL: dict[str, str] = {
    "gen": "Generator microphone (channel 0)",
    "tur": "Turbine microphone (channel 0)",
}


@dataclass(frozen=True)
class _SessionConfig:
    run_name: str
    events_csv: Path
    label: str


_SESSION_CONFIG: dict[str, _SessionConfig] = {
    "st": _SessionConfig(
        run_name=make_demo_assets.ST_RUN_NAME,
        events_csv=make_demo_assets.ST_EVENTS_CSV,
        label="ST -- standstill / calibration",
    ),
    "pu": _SessionConfig(
        run_name=make_demo_assets.PU_RUN_NAME,
        events_csv=make_demo_assets.PU_EVENTS_CSV,
        label="PU -- pump operation",
    ),
}

TEMPLATE_CSV_COLUMNS = (
    "session", "event_id", "kind", "snippet_start_utc", "snippet_end_utc",
    "gen_wav", "tur_wav", "expected_strikes",
    "strike1_offset_s", "strike2_offset_s", "strike3_offset_s", "extra_offsets_s",
    "confidence", "notes",
)
"""`annotation_template_<session>.csv`'s exact column contract, in this order."""

COMPILED_CSV_COLUMNS = (
    "session", "event_id", "kind", "strike_no", "strike_utc", "confidence", "notes",
)
"""`080726_strikes_seconds_<session>.csv`'s exact column contract, in this order."""

_SPEC_NPERSEG = 128
"""STFT window length, samples -- 2.56 ms at the native ~50 kHz rate. Short on
purpose (task instruction): a millisecond-scale broadband hammer-strike impulse
would smear across a wider window; this keeps the rendered vertical line thin
enough to read a second-level offset off by eye."""
_SPEC_NOVERLAP = 96
"""75% overlap (hop = 32 samples = 0.64 ms) -- dense enough in time that the
impulse's energy is never split between two under-lapping frames."""
_SPEC_FMAX_HZ = 20_000.0
_SPEC_DYNAMIC_RANGE_DB = 70.0
_SPEC_EPS = 1e-9
_SPEC_TICK_STEP_S = 5.0
_SPEC_PX_PER_S = 14.0
_SPEC_MIN_WIDTH_PX = 1000
_SPEC_MAX_WIDTH_PX = 3000
_SPEC_HEIGHT_PX = 480

_FLAT_PX_PER_S = 20.0
"""Pixel width per second for the borderless `*_flat.png` the interactive
index.html reads -- fixed (unlike `_SPEC_PX_PER_S`, no min/max clamp) so the
click<->time mapping is EXACTLY linear across an image's full width, for every
event regardless of duration: a 90s single-strike snippet renders at 1800 px,
the 210s vane-sweep at 4200 px."""
_FLAT_HEIGHT_PX = 360


# ---------------------------------------------------------------------------
# Pure: snippet window + burst-file boundary selection
# ---------------------------------------------------------------------------


def snippet_window(
    event_start_utc: datetime, event_end_utc: datetime, pad_s: float = SNIPPET_PAD_S
) -> tuple[datetime, datetime]:
    """`(event_start_utc - pad_s, event_end_utc + pad_s)` -- the annotation-clip
    window around one ground-truth event. A 60 s single-strike event
    (`event_end_utc - event_start_utc == 60s`) yields a 90 s window; the 180 s
    vane-sweep event yields a 210 s (3.5 min) window.

    Raises:
        ValueError: if `event_end_utc < event_start_utc` (a malformed event).
    """
    if event_end_utc < event_start_utc:
        raise ValueError(
            f"event_end_utc ({event_end_utc}) is before event_start_utc ({event_start_utc})"
        )
    pad = timedelta(seconds=pad_s)
    return event_start_utc - pad, event_end_utc + pad


def files_covering_window(
    files: Sequence[BurstFile], window_start_utc: datetime, window_end_utc: datetime
) -> list[BurstFile]:
    """Every burst file (ordered by `start_utc_hint`, ascending) whose bucket can
    overlap `[window_start_utc, window_end_utc)`: the file `make_demo_assets.
    find_burst_file` would pick for `window_start_utc` (its own bucket rule --
    the LATEST file starting at or before the target -- generalized here to
    simply fall back to the EARLIEST file when the window starts before every
    file's own start, i.e. "clamp to what's available" rather than
    `find_burst_file`'s own "raise if nothing covers it" contract), PLUS every
    subsequent file (in start-time order) that itself still starts before
    `window_end_utc`. This is the pure (no disk I/O -- filename-hint metadata
    only) selection step behind `extract_stream_clip`'s stitching; whether each
    selected file's REAL per-sample timestamps actually contain any of the
    window is decided later, per file, by `make_demo_assets.
    utc_window_to_sample_range`.

    Deliberately permissive at both ends (never raises for a window that
    reaches outside every available file): near a recording's own start/end, or
    around a dropped burst file, the caller (`extract_stream_clip`) is
    responsible for detecting and reporting the resulting gap -- this function
    only ever answers "which files are worth opening".

    Raises:
        ValueError: if *files* is empty, or `window_end_utc <= window_start_utc`.
    """
    if window_end_utc <= window_start_utc:
        raise ValueError(
            f"window_end_utc ({window_end_utc}) must be after window_start_utc "
            f"({window_start_utc})"
        )
    ordered = sorted(files, key=lambda f: f.start_utc_hint)
    if not ordered:
        raise ValueError("files must be non-empty")

    start_index = 0
    for i, bf in enumerate(ordered):
        if bf.start_utc_hint <= window_start_utc:
            start_index = i
        else:
            break
    selected = [ordered[start_index]]
    for bf in ordered[start_index + 1 :]:
        if bf.start_utc_hint >= window_end_utc:
            break
        selected.append(bf)
    return selected


# ---------------------------------------------------------------------------
# IO-touching: multi-file stream extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamClip:
    """Result of `extract_stream_clip`: possibly-stitched samples for one
    stream/channel over a requested UTC window."""

    samples: np.ndarray
    """float64, native (un-resampled) rate, already sliced to the requested
    channel and concatenated in chronological order across however many burst
    files were needed."""
    rate_hz: float
    covered_start_utc: datetime
    covered_end_utc: datetime
    clamped: bool
    """True iff the actually-covered span is narrower than the requested window
    by more than ~2 sample periods (missing neighboring burst file, or a
    genuine gap between two available files)."""
    note: str
    """Explanation when `clamped` is True; `""` otherwise."""


def extract_stream_clip(
    files: Sequence[BurstFile],
    offset_ns: int,
    channel_index: int,
    window_start_utc: datetime,
    window_end_utc: datetime,
) -> StreamClip:
    """Load and stitch every burst file needed to cover `[window_start_utc,
    window_end_utc)` for one stream/channel -- generalizes `make_demo_assets.
    _extract_mono_clip` along the two axes this task's windows need: (1) ANY
    channel of ANY stream (that function hardcodes `RAWGeneratorMic__0` channel
    0), and (2) a window that CROSSES a burst-file boundary (this campaign's
    files are ~12 min; a +/-15 s-padded strike window is ~90 s-3.5 min and DOES
    straddle a boundary on real 080726 data -- see module docstring). Loads at
    most one burst file's full array into memory at a time (`del`/
    `gc.collect()` before the next), mirroring `_extract_mono_clip`'s own
    memory discipline.

    A window reaching outside the AVAILABLE files -- near a recording's own
    start/end, or a genuine gap between two burst files -- never raises AS LONG
    AS AT LEAST ONE SAMPLE overlaps the window: the returned clip covers
    whatever overlap exists, `clamped=True`, and `note` explains why (task
    instruction: "clamp the window and record that in the metadata"). Only a
    window with NO overlap at all against every candidate file still raises
    (same contract as `_extract_mono_clip`'s own "no overlap" case) -- there is
    nothing to clamp to, and a silently empty clip would be actively
    misleading.

    Raises:
        ValueError: if *files* is empty, `window_end_utc <= window_start_utc`
            (both via `files_covering_window`), or the window has zero overlap
            with every candidate burst file.
    """
    selected = files_covering_window(files, window_start_utc, window_end_utc)

    chunks: list[np.ndarray] = []
    rate_hz = 0.0
    covered_start_utc: datetime | None = None
    covered_end_utc: datetime | None = None
    for bf in selected:
        gf = read_gantner(bf.path)
        true_utc_ns = make_demo_assets._shift_ts_ns(gf.timestamps_ns, offset_ns)
        start_idx, end_idx = make_demo_assets.utc_window_to_sample_range(
            window_start_utc, window_end_utc, true_utc_ns
        )
        if end_idx > start_idx:
            chunk = gf.data[start_idx:end_idx, channel_index].astype(np.float64).copy()
            chunk_start = datetime.fromtimestamp(int(true_utc_ns[start_idx]) / 1e9, tz=UTC)
            chunk_end = datetime.fromtimestamp(int(true_utc_ns[end_idx - 1]) / 1e9, tz=UTC)
            chunks.append(chunk)
            rate_hz = gf.header.sample_rate_hz
            covered_start_utc = (
                chunk_start if covered_start_utc is None else min(covered_start_utc, chunk_start)
            )
            covered_end_utc = (
                chunk_end if covered_end_utc is None else max(covered_end_utc, chunk_end)
            )
        del gf, true_utc_ns
        gc.collect()

    if not chunks or covered_start_utc is None or covered_end_utc is None:
        raise ValueError(
            f"[{window_start_utc}, {window_end_utc}) has no overlap with any of the "
            f"{len(selected)} candidate burst file(s) for this stream"
        )
    samples = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    tolerance = timedelta(seconds=2.0 / rate_hz) if rate_hz > 0 else timedelta(0)
    gap_notes: list[str] = []
    if covered_start_utc - window_start_utc > tolerance:
        gap_notes.append(
            f"start clamped by {(covered_start_utc - window_start_utc).total_seconds():.3f}s "
            "(no earlier data available)"
        )
    if window_end_utc - covered_end_utc > tolerance:
        gap_notes.append(
            f"end clamped by {(window_end_utc - covered_end_utc).total_seconds():.3f}s "
            "(no later data available)"
        )

    return StreamClip(
        samples=samples,
        rate_hz=rate_hz,
        covered_start_utc=covered_start_utc,
        covered_end_utc=covered_end_utc,
        clamped=bool(gap_notes),
        note="; ".join(gap_notes),
    )


# ---------------------------------------------------------------------------
# Spectrogram rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SpectrogramData:
    """Shared STFT-to-dB result behind BOTH `render_spectrogram_png` (labeled,
    for humans) and `render_flat_spectrogram_png` (borderless, for the
    interactive UI) -- computed via `_compute_spectrogram_db` ONCE per
    (samples, rate_hz), so "same STFT params" is a code-sharing fact, not just
    a documentation claim."""

    f_khz: np.ndarray
    sxx_db: np.ndarray
    vmin: float
    vmax: float
    duration_s: float


def _compute_spectrogram_db(samples: np.ndarray, rate_hz: float) -> _SpectrogramData:
    """Log-magnitude (dB) spectrogram of *samples* (native, UN-resampled rate --
    resampling to the 16 kHz WAV rate would Nyquist-limit the view to 8 kHz,
    under the requested 0-20 kHz band), 0-20 kHz band only. Short-window/
    high-overlap STFT (`_SPEC_NPERSEG`/`_SPEC_NOVERLAP`) so a millisecond-scale
    broadband hammer strike renders as a sharp vertical line rather than a
    smeared blob.
    """
    normalized = make_demo_assets.peak_normalize(samples, make_demo_assets.TARGET_DBFS)
    f_hz, _t_s, sxx = signal.spectrogram(
        normalized.astype(np.float32), fs=rate_hz, window="hann",
        nperseg=_SPEC_NPERSEG, noverlap=_SPEC_NOVERLAP, mode="magnitude",
    )
    in_band = f_hz <= _SPEC_FMAX_HZ
    f_khz = f_hz[in_band] / 1000.0
    sxx_db = 20.0 * np.log10(np.maximum(sxx[in_band, :], _SPEC_EPS))
    vmax = float(np.max(sxx_db))
    vmin = vmax - _SPEC_DYNAMIC_RANGE_DB
    duration_s = len(samples) / rate_hz
    return _SpectrogramData(f_khz=f_khz, sxx_db=sxx_db, vmin=vmin, vmax=vmax, duration_s=duration_s)


def render_spectrogram_png(
    path: Path,
    samples: np.ndarray,
    rate_hz: float,
    *,
    kind: str,
    stream_label: str,
    snippet_start_utc: datetime,
) -> None:
    """Log-magnitude spectrogram PNG of *samples*, WITH axes/ticks/title -- the
    human-readable companion to `render_flat_spectrogram_png`'s borderless
    version (`_compute_spectrogram_db` computes the shared data once; only the
    presentation differs). X-axis is SECONDS SINCE *snippet_start_utc* (0 at
    the very first sample, ticked every `_SPEC_TICK_STEP_S` s) --
    *snippet_start_utc* itself only appears in the title, for cross-reference
    against the CSV template's own `snippet_start_utc` column.
    """
    spec = _compute_spectrogram_db(samples, rate_hz)
    width_px = int(
        np.clip(round(_SPEC_PX_PER_S * spec.duration_s), _SPEC_MIN_WIDTH_PX, _SPEC_MAX_WIDTH_PX)
    )
    plt = make_demo_assets._pyplot()
    fig = plt.figure(figsize=(width_px / 100, _SPEC_HEIGHT_PX / 100), dpi=100)
    ax = fig.add_axes((0.055, 0.15, 0.93, 0.74))
    ax.imshow(
        spec.sxx_db, aspect="auto", origin="lower", cmap="magma",
        extent=(0.0, spec.duration_s, float(spec.f_khz[0]), float(spec.f_khz[-1])),
        vmin=spec.vmin, vmax=spec.vmax,
    )
    ax.set_xlim(0.0, spec.duration_s)
    ticks = np.arange(0.0, spec.duration_s + 1e-6, _SPEC_TICK_STEP_S)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tv:g}" for tv in ticks])
    ax.grid(axis="x", color="white", alpha=0.25, linewidth=0.6)
    ax.set_xlabel("Time since snippet start (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(
        f"{kind} -- {stream_label}\nSnippet start (UTC): {snippet_start_utc.isoformat()}",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png")
    plt.close(fig)


def flat_spectrogram_width_px(duration_s: float) -> int:
    """Pixel width of a `*_flat.png` for a snippet of *duration_s* seconds:
    exactly `_FLAT_PX_PER_S` px/s, rounded to the nearest pixel -- deliberately
    NO min/max clamp (unlike `render_spectrogram_png`'s labeled version),
    because the interactive UI's click-to-time mapping (`offset_s = x_px /
    image_width_px * duration_s`) depends on this proportionality holding
    EXACTLY for every event: `flat_spectrogram_width_px(90.0) == 1800`,
    `flat_spectrogram_width_px(210.0) == 4200` (the vane-sweep).
    """
    return round(_FLAT_PX_PER_S * duration_s)


def render_flat_spectrogram_png(path: Path, samples: np.ndarray, rate_hz: float) -> None:
    """Borderless log-magnitude spectrogram PNG: exactly the plot area, no
    axes/ticks/title/margins (`fig.add_axes((0, 0, 1, 1))` + `ax.axis("off")`)
    -- the property the interactive index.html UI relies on: because the axes
    fill the ENTIRE canvas with zero margin, a click at client-pixel x maps
    LINEARLY to `x / image_client_width_px * duration_s` seconds over the
    FULL image width, with no offset/margin correction needed. Uses the SAME
    STFT parameters as `render_spectrogram_png` (`_compute_spectrogram_db`,
    called once and shared) so impulses are exactly as visible in both; only
    the presentation and pixel width (`flat_spectrogram_width_px`, no
    min/max clamp) differ.
    """
    spec = _compute_spectrogram_db(samples, rate_hz)
    width_px = max(1, flat_spectrogram_width_px(spec.duration_s))
    plt = make_demo_assets._pyplot()
    fig = plt.figure(figsize=(width_px / 100, _FLAT_HEIGHT_PX / 100), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(
        spec.sxx_db, aspect="auto", origin="lower", cmap="magma",
        extent=(0.0, spec.duration_s, float(spec.f_khz[0]), float(spec.f_khz[-1])),
        vmin=spec.vmin, vmax=spec.vmax,
    )
    ax.set_xlim(0.0, spec.duration_s)
    ax.set_ylim(float(spec.f_khz[0]), float(spec.f_khz[-1]))
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Template CSV (build side)
# ---------------------------------------------------------------------------

_TEMPLATE_HEADER_COMMENT = (
    "# Strike-annotation template (seconds-level; offsets are relative to "
    "snippet_start_utc). Full workflow: index.html in this same directory. "
    "extra_offsets_s is ';'-separated (e.g. \"12.3;12.9\"). Compile with: "
    "python scripts/annotation_kit.py compile --template <this file> --out "
    "docs/groundtruth/080726_strikes_seconds_<session>.csv --date YYYY-MM-DD\n"
)


@dataclass(frozen=True)
class TemplateRow:
    """One `annotation_template_<session>.csv` row. Everything EXCEPT the four
    annotation fields (`strike1_offset_s`/`strike2_offset_s`/`strike3_offset_s`/
    `extra_offsets_s`), `confidence`, and `notes` is prefilled by
    `build_session`; those are left empty for Stefan -- UNLESS `notes_prefill`
    carries a system warning (a clamped snippet, see `extract_stream_clip`),
    which is written into `notes` up front so it is not missed.
    """

    session: str
    event_id: str
    kind: str
    snippet_start_utc: datetime
    snippet_end_utc: datetime
    gen_wav: str
    tur_wav: str
    expected_strikes: str
    notes_prefill: str = ""

    def to_row(self) -> dict[str, str]:
        return {
            "session": self.session,
            "event_id": self.event_id,
            "kind": self.kind,
            "snippet_start_utc": self.snippet_start_utc.isoformat(),
            "snippet_end_utc": self.snippet_end_utc.isoformat(),
            "gen_wav": self.gen_wav,
            "tur_wav": self.tur_wav,
            "expected_strikes": self.expected_strikes,
            "strike1_offset_s": "",
            "strike2_offset_s": "",
            "strike3_offset_s": "",
            "extra_offsets_s": "",
            "confidence": "",
            "notes": self.notes_prefill,
        }


def write_template_csv(rows: Sequence[TemplateRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([r.to_row() for r in rows], columns=list(TEMPLATE_CSV_COLUMNS))
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(_TEMPLATE_HEADER_COMMENT)
        frame.to_csv(fh, index=False)
    return path


@dataclass(frozen=True)
class SessionBuildResult:
    session: str
    rows: list[TemplateRow]
    template_path: Path
    n_wav: int
    n_png: int
    n_flat_png: int
    clamp_notes: list[str]
    """`"<event_id>: <stream>: <note>"` entries -- every event where at least
    one stream's snippet got clamped, for the CLI's own summary warning."""


def build_session(index: RecordingIndex, session: str, out_dir: Path) -> SessionBuildResult:
    """Build one session's WAVs/PNGs/template under `<out_dir>/<session>/` +
    `<out_dir>/annotation_template_<session>.csv`. Does NOT touch `index.html`
    (see `render_index_html`, called once by `build_kit` after every requested
    session is done).
    """
    session_cfg = _SESSION_CONFIG[session]
    run = make_demo_assets._get_run(index, session_cfg.run_name)
    events = make_demo_assets._load_events_csv(session_cfg.events_csv)
    offset_ns = run_utc_offset_ns(run)

    session_dir = out_dir / session
    session_dir.mkdir(parents=True, exist_ok=True)

    rows: list[TemplateRow] = []
    clamp_notes: list[str] = []
    for i, ev in enumerate(events.itertuples(index=False), start=1):
        event_id = f"{i:02d}"
        kind = str(ev.kind)
        event_start: datetime = ev.start_utc.to_pydatetime()
        event_end: datetime = ev.end_utc.to_pydatetime()
        win_start, win_end = snippet_window(event_start, event_end)

        wav_names: dict[str, str] = {}
        event_clamp_notes: list[str] = []
        for stream_key, stream_name in _STREAM_NAME_BY_KEY.items():
            files = run.files.get(stream_name, [])
            if not files:
                raise ValueError(
                    f"run {run.name!r} has no {stream_name!r} files (session {session!r})"
                )
            clip = extract_stream_clip(
                files, offset_ns, make_demo_assets.MONO_CHANNEL_INDEX, win_start, win_end
            )
            base = f"event_{event_id}_{kind}__{stream_key}"
            resampled = make_demo_assets._resample_to_target(clip.samples, clip.rate_hz)
            make_demo_assets._write_clip_wav(session_dir / f"{base}.wav", resampled)
            render_spectrogram_png(
                session_dir / f"{base}.png", clip.samples, clip.rate_hz,
                kind=kind, stream_label=_STREAM_LABEL[stream_key], snippet_start_utc=win_start,
            )
            render_flat_spectrogram_png(
                session_dir / f"{base}_flat.png", clip.samples, clip.rate_hz
            )
            wav_names[stream_key] = f"{session}/{base}.wav"
            if clip.clamped:
                event_clamp_notes.append(f"{stream_key}: {clip.note}")
            logger.info(
                "annotation_kit: %s event %s (%s) %s -> %s",
                session, event_id, kind, stream_key, wav_names[stream_key],
            )

        if event_clamp_notes:
            clamp_notes.append(f"{event_id}: {'; '.join(event_clamp_notes)}")

        rows.append(
            TemplateRow(
                session=session, event_id=event_id, kind=kind,
                snippet_start_utc=win_start, snippet_end_utc=win_end,
                gen_wav=wav_names["gen"], tur_wav=wav_names["tur"],
                expected_strikes="sweep" if kind == "vane-sweep" else "3",
                notes_prefill="; ".join(event_clamp_notes),
            )
        )

    template_path = write_template_csv(rows, out_dir / f"annotation_template_{session}.csv")
    return SessionBuildResult(
        session=session, rows=rows, template_path=template_path,
        n_wav=2 * len(rows), n_png=2 * len(rows), n_flat_png=2 * len(rows),
        clamp_notes=clamp_notes,
    )


def build_kit(cfg: Config, sessions: Sequence[str], out_dir: Path) -> list[SessionBuildResult]:
    """Build every session in *sessions* (each `"st"` or `"pu"`) under
    *out_dir*, then (re)render `events_meta.json`, the interactive `index.html`,
    and the read-only `index_static.html` from every `annotation_template_
    *.csv` the directory now holds (`events_meta_from_templates`/
    `render_interactive_index_html`/`render_static_index_html`'s own shared
    idempotent-rebuild contract: a partial `build --session st` run now,
    followed by `build --session pu` later, still yields complete, correct
    outputs covering BOTH sessions).
    """
    index = discover(cfg.data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [build_session(index, session, out_dir) for session in sessions]
    write_events_meta_json(out_dir)
    render_interactive_index_html(out_dir)
    render_static_index_html(out_dir)
    return results


# ---------------------------------------------------------------------------
# index.html
# ---------------------------------------------------------------------------

_INDEX_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 2rem auto; max-width: 1100px; padding: 0 1rem; line-height: 1.5; }
h1 { font-size: 1.5rem; }
h2 { font-size: 1.2rem; margin-top: 2.5rem; border-bottom: 1px solid #8888;
     padding-bottom: 0.3rem; }
code { background: #80808022; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.92em; }
.instructions { background: #80808014; border: 1px solid #8888; border-radius: 8px;
                padding: 1rem 1.4rem; margin-bottom: 2rem; }
.instructions ol { padding-left: 1.3rem; }
.hint { color: #808080; font-size: 0.9rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1rem; }
th, td { border: 1px solid #8886; padding: 0.5rem; text-align: left; vertical-align: top; }
th { background: #80808022; }
td.media { min-width: 260px; }
td.media img { max-width: 100%; height: auto; display: block; margin-bottom: 0.35rem;
               border: 1px solid #8886; }
audio { width: 100%; }
"""

_INSTRUCTIONS_HTML = """<section class="instructions">
<h2>Workflow</h2>
<ol>
<li><strong>Look at the spectrogram:</strong> two spectrograms per event
(generator/turbine microphone, 0&ndash;20 kHz, time axis in seconds since snippet start,
5-s grid). A hammer strike is broadband and appears as a sharp vertical line
&ndash; typically 3 lines per event.</li>
<li><strong>Confirm by ear:</strong> use the players below to listen to the
read-off timestamps and confirm or correct them.</li>
<li><strong>Fill in the CSV:</strong> in <code>annotation_template_&lt;session&gt;.csv</code>
fill <code>strike1_offset_s</code> / <code>strike2_offset_s</code> /
<code>strike3_offset_s</code> per event with the read-off second offsets (relative to
<code>snippet_start_utc</code>, NOT the original minute). Heard more than 3 strikes?
Add further offsets <em>semicolon-separated</em> in <code>extra_offsets_s</code> (e.g.
<code>12.3;12.9</code>). Fill <code>confidence</code> (e.g. high/medium/low) and
<code>notes</code> freely.</li>
</ol>
<p><strong>Special case <code>vane-sweep</code>:</strong> not 3 discrete strikes, but a roughly
3-minute structure-borne excitation at the guide-vane cover. Enter sweep start/end as free text in
<code>notes</code>; individually audible strikes can optionally additionally be entered in
<code>extra_offsets_s</code>.</p>
<p>Then compile:<br><code>python scripts/annotation_kit.py compile --template
results/annotation-kit/080726/annotation_template_&lt;session&gt;.csv --out
docs/groundtruth/080726_strikes_seconds_&lt;session&gt;.csv --date YYYY-MM-DD</code></p>
<p class="hint">If <code>notes</code> already carries a system note (e.g. "end clamped by ..."):
that only concerns the edge of the snippet (missing neighboring file), not the strike itself.</p>
<p class="hint">Adjacent events are only ~1 minute apart, so their
&plusmn;15-s windows overlap at the edge: right at the start or end of a snippet
a second, similar-looking strike cluster can appear that actually
belongs to the NEIGHBORING event. For this event, count the strikes near the
snippet middle (roughly t=15s to snippet duration minus 15s, the actual
protocol minute window).</p>
</section>
"""


def _cell_to_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _wav_to_png(wav_rel_path: str) -> str:
    return wav_rel_path.removesuffix(".wav") + ".png"


def _wav_to_flat_png(wav_rel_path: str) -> str:
    return wav_rel_path.removesuffix(".wav") + "_flat.png"


def _event_row_html(row: Mapping[str, str]) -> str:
    event_id, kind = row["event_id"], row["kind"]
    gen_wav, tur_wav = row["gen_wav"], row["tur_wav"]
    gen_png, tur_png = _wav_to_png(gen_wav), _wav_to_png(tur_wav)
    return (
        "<tr>"
        f"<td>{html.escape(event_id)}</td>"
        f"<td>{html.escape(kind)}</td>"
        f"<td>{html.escape(row['snippet_start_utc'])}<br>&ndash;<br>"
        f"{html.escape(row['snippet_end_utc'])}</td>"
        f'<td class="media"><img src="{html.escape(gen_png)}" '
        f'alt="Spectrogram generator, event {html.escape(event_id)}">'
        f'<audio controls src="{html.escape(gen_wav)}"></audio></td>'
        f'<td class="media"><img src="{html.escape(tur_png)}" '
        f'alt="Spectrogram turbine, event {html.escape(event_id)}">'
        f'<audio controls src="{html.escape(tur_wav)}"></audio></td>'
        "</tr>\n"
    )


def _session_section_html(session: str, df: pd.DataFrame) -> str:
    cfg = _SESSION_CONFIG.get(session)
    label = cfg.label if cfg is not None else session
    rows_html = "".join(
        _event_row_html({str(k): _cell_to_str(v) for k, v in record.items()})
        for record in df.to_dict(orient="records")
    )
    return (
        f"<section>\n<h2>Session {html.escape(session)} &ndash; {html.escape(label)}</h2>\n"
        f'<p class="hint">Template: <code>annotation_template_{html.escape(session)}.csv</code>'
        f" ({len(df)} Event(s))</p>\n"
        "<table>\n<thead><tr><th>Event</th><th>Kind</th><th>UTC minute (snippet)</th>"
        "<th>Generator microphone</th><th>Turbine microphone</th></tr></thead>\n"
        f"<tbody>\n{rows_html}</tbody>\n</table>\n</section>\n"
    )


def render_static_index_html(out_dir: Path) -> Path:
    """(Re)generate `<out_dir>/index_static.html` -- the ORIGINAL read-only,
    one-big-table overview (superseded as Stefan's actual workflow entrypoint
    by `render_interactive_index_html`'s `index.html`, kept around as a plain
    look/listen reference) -- from every `annotation_template_<session>.csv`
    CURRENTLY present in *out_dir* -- NOT just the session(s) the triggering
    `build` call itself just (re)wrote -- so a partial rebuild (e.g. `build
    --session st` run again later) still produces a complete, correct index
    reflecting the kit directory's actual contents. Self-contained (inline CSS
    only, no external resources); every `<audio>`/`<img>` reference is a path
    RELATIVE to *out_dir* (never a base64 embed -- 104 real clips would make an
    embedded page impractically large).
    """
    template_paths = sorted(out_dir.glob("annotation_template_*.csv"))
    sections = []
    for template_path in template_paths:
        session = template_path.stem.removeprefix("annotation_template_")
        df = read_template_csv(template_path)
        sections.append(_session_section_html(session, df))

    doc = (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        "<title>080726 Strike Annotation Kit (static)</title>\n"
        f"<style>{_INDEX_CSS}</style>\n</head>\n<body>\n"
        "<h1>08.07.2026 &ndash; Schonhammer strike annotation (static overview)</h1>\n"
        '<p class="hint">Interactive annotation tool: <code>index.html</code> in '
        "this folder. This page is a read-only overview only.</p>\n"
        f"{_INSTRUCTIONS_HTML}\n{''.join(sections)}\n</body>\n</html>\n"
    )
    out_path = out_dir / "index_static.html"
    out_path.write_text(doc, encoding="utf-8")
    logger.info("annotation_kit: wrote %s (%d session section(s))", out_path, len(sections))
    return out_path


# ---------------------------------------------------------------------------
# Interactive index.html -- Stefan's actual per-strike annotation workflow:
# click-to-mark on a borderless flat spectrogram, audio playback with a
# millisecond-precision clock + live playhead, per-marker chips, autosave to
# localStorage, per-session CSV export/import. See `render_interactive_index_
# html`'s own docstring for why the events data is EMBEDDED rather than
# `fetch()`-ed (file:// CORS).
# ---------------------------------------------------------------------------

_INTERACTIVE_CSS = """
:root {
  color-scheme: light dark;
  --border: #8888;
  --muted: #808080;
  --accent: #3b82f6;
  --marker: #f59e0b;
  --playhead: #ef4444;
  --bg-soft: #80808014;
  --bg-soft-2: #80808022;
  --ok: #16a34a;
  --err: #dc2626;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 2rem auto; max-width: 1400px; padding: 0 1rem; line-height: 1.5; }
h1 { font-size: 1.5rem; }
h2 { font-size: 1.2rem; margin-top: 2.5rem; border-bottom: 1px solid var(--border);
     padding-bottom: 0.3rem; }
code { background: var(--bg-soft-2); padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.92em; }
.instructions { background: var(--bg-soft); border: 1px solid var(--border); border-radius: 8px;
                padding: 0.8rem 1.2rem; margin-bottom: 1.5rem; font-size: 0.93rem; }
.instructions p { margin: 0.4em 0; }
.hint { color: var(--muted); font-size: 0.85rem; margin: 0.3em 0; }

.session-section { margin-top: 2rem; }
.session-toolbar { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1rem; }
.session-status { color: var(--muted); font-size: 0.85rem; }

.event-card { border: 1px solid var(--border); border-radius: 8px; padding: 0.9rem 1.1rem;
              margin-bottom: 1.2rem; outline-offset: 2px; }
.event-card:focus, .event-card:focus-visible { outline: 2px solid var(--accent); }

.card-head { display: flex; justify-content: space-between; align-items: baseline;
             gap: 1rem; flex-wrap: wrap; }
.card-title { font-weight: 600; }
.marker-count { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 0.9rem; }
.marker-count.count-complete { color: var(--ok); font-weight: 600; }

.controls-row { display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap;
                margin: 0.5rem 0; }
.btn-group { display: inline-flex; border: 1px solid var(--border); border-radius: 6px;
             overflow: hidden; }
.btn-group button { border: none; background: transparent; padding: 0.25rem 0.6rem; cursor: pointer;
                     font-size: 0.85rem; color: inherit; border-right: 1px solid var(--border); }
.btn-group button:last-child { border-right: none; }
.btn-group button.active { background: var(--accent); color: white; }
.time-display { font-variant-numeric: tabular-nums; font-family: ui-monospace, Menlo, monospace;
                 font-size: 0.9rem; }
.utc-live { color: var(--muted); font-size: 0.8rem; font-family: ui-monospace, Menlo, monospace; }

.spectro-scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: 4px; }
.spectro-wrapper { position: relative; display: block; }
.spectro-img { display: block; width: 100%; height: 100%; }
.overlay-canvas { position: absolute; top: 0; left: 0; cursor: crosshair; display: block; }

audio.event-audio { width: 100%; margin: 0.5rem 0; }

.marker-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.5rem 0;
                 min-height: 1.6rem; }
.marker-chip { border: 1px solid var(--marker); color: var(--marker); background: transparent;
                border-radius: 12px; padding: 0.1rem 0.55rem; font-size: 0.78rem; cursor: pointer;
                font-family: ui-monospace, Menlo, monospace; }
.marker-chip:hover { background: var(--marker); color: #1a1200; }

.meta-row { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin: 0.4rem 0; }
.field-label { display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem;
               color: var(--muted); }
.notes-label { flex: 1 1 260px; }
.notes-input { flex: 1; padding: 0.25rem 0.5rem; border: 1px solid var(--border);
               border-radius: 4px; background: transparent; color: inherit; font-size: 0.85rem; }
.confidence-select { padding: 0.2rem 0.4rem; border: 1px solid var(--border); border-radius: 4px;
                      background: transparent; color: inherit; }

.card-footer { display: flex; justify-content: flex-end; }
.save-indicator { color: var(--muted); font-size: 0.78rem; }
.save-indicator.save-error { color: var(--err); }

.sweep-hint { font-style: italic; }

button, input, select { font: inherit; }
"""

_INTERACTIVE_INSTRUCTIONS_HTML = """<section class="instructions">
<p><strong>Click</strong> in the spectrogram sets a strike marker at the clicked position
and jumps there &ndash; <strong>Shift-click</strong> only jumps there (no marker). The
<strong>M</strong> key marks at the current playback position, <strong>space bar</strong>
plays/pauses (the card must be focused, e.g. after a click into the spectrogram). Arrow keys
<strong>&larr;/&rarr;</strong> jump &plusmn;0.5s (with Shift &plusmn;0.05s).</p>
<p>Switching channels (Gen/Tur) keeps the playback position, zoom 1x/4x stretches the time axis
for more precise clicking. Changes are saved locally as you go (shown as
&ldquo;saved&rdquo;) &ndash; at the end of each session click
<strong>&ldquo;Export CSV&rdquo;</strong> to save the markers as a file
(later restorable via <strong>&ldquo;Import CSV&rdquo;</strong>).</p>
</section>
"""

_INTERACTIVE_JS = r"""
(function () {
"use strict";

var STORAGE_PREFIX = "strike-annot:v1:";
var ARROW_STEP_S = 0.5;
var ARROW_STEP_FINE_S = 0.05;
var ZOOM_LEVELS = [1, 4];
var FLAT_PX_PER_S = 20;
var FLAT_HEIGHT_PX = 360;
var MARKS_CSV_HEADER = [
  "session", "event_id", "kind", "strike_no", "offset_s", "strike_utc", "confidence", "notes"
];

var metaEl = document.getElementById("events-meta-data");
var labelsEl = document.getElementById("session-labels-data");
var EVENTS_META = JSON.parse(metaEl.textContent);
var SESSION_LABELS = JSON.parse(labelsEl.textContent);

function flatWidthPx(durationS) {
  return Math.round(FLAT_PX_PER_S * durationS);
}

function storageKey(session, eventId) {
  return STORAGE_PREFIX + session + ":" + eventId;
}

function defaultState() {
  return { markers: [], confidence: "", notes: "" };
}

function loadState(session, eventId) {
  var raw = null;
  try {
    raw = localStorage.getItem(storageKey(session, eventId));
  } catch (e) {
    raw = null;
  }
  if (!raw) return defaultState();
  try {
    var parsed = JSON.parse(raw);
    var markers = Array.isArray(parsed.markers) ? parsed.markers.slice() : [];
    markers = markers.map(Number).filter(function (v) { return isFinite(v); });
    markers.sort(function (a, b) { return a - b; });
    return {
      markers: markers,
      confidence: typeof parsed.confidence === "string" ? parsed.confidence : "",
      notes: typeof parsed.notes === "string" ? parsed.notes : "",
    };
  } catch (e) {
    return defaultState();
  }
}

function saveState(card) {
  try {
    var key = storageKey(card.meta.session, card.meta.event_id);
    localStorage.setItem(key, JSON.stringify(card.state));
    card.saveStatus = "saved";
  } catch (e) {
    card.saveStatus = "error";
  }
  updateSaveIndicator(card);
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function pad2(n) {
  return (n < 10 ? "0" : "") + n;
}

function pad3(n) {
  if (n < 10) return "00" + n;
  if (n < 100) return "0" + n;
  return "" + n;
}

function formatClock(t) {
  if (!isFinite(t) || t < 0) t = 0;
  var m = Math.floor(t / 60);
  var s = Math.floor(t % 60);
  var ms = Math.round((t - Math.floor(t)) * 1000);
  if (ms === 1000) { ms = 0; s += 1; }
  if (s === 60) { s = 0; m += 1; }
  return pad2(m) + ":" + pad2(s) + "." + pad3(ms);
}

function offsetToAbsoluteUtcIso(snippetStartIso, offsetS) {
  var start = new Date(snippetStartIso);
  var abs = new Date(start.getTime() + offsetS * 1000);
  return abs.toISOString();
}

function nowClock() {
  var d = new Date();
  return pad2(d.getHours()) + ":" + pad2(d.getMinutes()) + ":" + pad2(d.getSeconds());
}

function csvField(value) {
  var s = value === null || value === undefined ? "" : String(value);
  if (/[",\r\n]/.test(s)) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function parseCsv(text) {
  var rows = [];
  var row = [];
  var field = "";
  var inQuotes = false;
  for (var i = 0; i < text.length; i++) {
    var c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else { inQuotes = false; }
      } else {
        field += c;
      }
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === ",") {
      row.push(field); field = "";
    } else if (c === "\n") {
      row.push(field); rows.push(row); row = []; field = "";
    } else if (c === "\r") {
      // skip -- \n (or end of input) terminates the row
    } else {
      field += c;
    }
  }
  if (field.length > 0 || row.length > 0) { row.push(field); rows.push(row); }
  if (rows.length === 0) return [];
  var header = rows[0];
  var out = [];
  for (var r = 1; r < rows.length; r++) {
    var cells = rows[r];
    if (cells.length === 1 && cells[0] === "") continue;
    var obj = {};
    for (var h = 0; h < header.length; h++) {
      obj[header[h]] = cells[h] !== undefined ? cells[h] : "";
    }
    out.push(obj);
  }
  return out;
}

function downloadTextFile(filename, text) {
  var blob = new Blob([text], { type: "text/csv" });
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function el(tag, attrs, children) {
  var node = document.createElement(tag);
  if (attrs) {
    Object.keys(attrs).forEach(function (k) {
      if (k === "class") node.className = attrs[k];
      else if (k === "text") node.textContent = attrs[k];
      else node.setAttribute(k, attrs[k]);
    });
  }
  (children || []).forEach(function (c) { node.appendChild(c); });
  return node;
}

var CARDS = {};
var CARDS_BY_SESSION = {};

function cardKey(session, eventId) {
  return session + ":" + eventId;
}

function buildCard(meta) {
  var restoredState = loadState(meta.session, meta.event_id);
  var hasRestoredData = restoredState.markers.length > 0 || !!restoredState.confidence ||
    !!restoredState.notes;
  var card = {
    meta: meta,
    channel: "gen",
    zoomIndex: 0,
    state: restoredState,
    saveStatus: hasRestoredData ? "restored" : "unsaved",
    naturalWidth: 0,
    naturalHeight: 0,
  };

  var head = el("div", { class: "card-head" });
  var titleSpan = el("span", { class: "card-title" });
  titleSpan.textContent = "Event " + meta.event_id + " — " + meta.kind;
  var countSpan = el("span", { class: "marker-count" });
  head.appendChild(titleSpan);
  head.appendChild(countSpan);

  var utcSpan = el("div", { class: "hint" });
  utcSpan.textContent = "Snippet start (UTC): " + meta.snippet_start_utc +
    "  ·  Duration: " + meta.duration_s.toFixed(1) + "s";

  var controls = el("div", { class: "controls-row" });
  var chanGroup = el("span", { class: "btn-group" });
  var genBtn = el("button", { type: "button", text: "Gen" });
  var turBtn = el("button", { type: "button", text: "Tur" });
  chanGroup.appendChild(genBtn);
  chanGroup.appendChild(turBtn);

  var zoomGroup = el("span", { class: "btn-group" });
  var zoom1Btn = el("button", { type: "button", text: "1x" });
  var zoom4Btn = el("button", { type: "button", text: "4x" });
  zoomGroup.appendChild(zoom1Btn);
  zoomGroup.appendChild(zoom4Btn);

  var rateGroup = el("span", { class: "btn-group" });
  var rate05Btn = el("button", { type: "button", text: "0.5x" });
  var rate1Btn = el("button", { type: "button", text: "1x" });
  rateGroup.appendChild(rate05Btn);
  rateGroup.appendChild(rate1Btn);

  var timeDisplay = el("span", { class: "time-display" });
  var utcLive = el("span", { class: "utc-live" });

  controls.appendChild(chanGroup);
  controls.appendChild(zoomGroup);
  controls.appendChild(rateGroup);
  controls.appendChild(timeDisplay);
  controls.appendChild(utcLive);

  var scroll = el("div", { class: "spectro-scroll" });
  var wrapper = el("div", { class: "spectro-wrapper" });
  var img = el("img", { class: "spectro-img", alt: "Spectrogram " + meta.event_id });
  var canvas = el("canvas", { class: "overlay-canvas" });
  wrapper.appendChild(img);
  wrapper.appendChild(canvas);
  scroll.appendChild(wrapper);

  var audio = el("audio", { class: "event-audio", preload: "metadata", controls: "" });

  var chips = el("div", { class: "marker-chips" });

  var metaRow = el("div", { class: "meta-row" });
  var confLabel = el("label", { class: "field-label", text: "Confidence" });
  var confSelect = el("select", { class: "confidence-select" });
  ["", "high", "medium", "low"].forEach(function (v) {
    confSelect.appendChild(el("option", { value: v, text: v === "" ? "–" : v }));
  });
  confLabel.appendChild(confSelect);

  var notesLabel = el("label", { class: "field-label notes-label", text: "Notes" });
  var notesInput = el("input", { type: "text", class: "notes-input", placeholder: "free text..." });
  notesLabel.appendChild(notesInput);

  metaRow.appendChild(confLabel);
  metaRow.appendChild(notesLabel);

  var saveIndicator = el("span", { class: "save-indicator" });
  var footer = el("div", { class: "card-footer" }, [saveIndicator]);

  var cardEl = el("div", { class: "event-card", tabindex: "0" },
    [head, utcSpan, controls, scroll, audio, chips, metaRow, footer]);

  if (meta.kind === "vane-sweep") {
    var sweepHint = el("p", { class: "hint sweep-hint" });
    sweepHint.textContent = "Special case vane-sweep: individual strikes are optional; " +
      "sweep start/end can be set as normal markers.";
    cardEl.appendChild(sweepHint);
  }

  card.el = cardEl;
  card.audioEl = audio;
  card.imgEl = img;
  card.canvasEl = canvas;
  card.wrapperEl = wrapper;
  card.countEl = countSpan;
  card.timeDisplayEl = timeDisplay;
  card.utcLiveEl = utcLive;
  card.chipsEl = chips;
  card.confSelect = confSelect;
  card.notesInput = notesInput;
  card.saveIndicatorEl = saveIndicator;
  card.genBtn = genBtn;
  card.turBtn = turBtn;
  card.zoom1Btn = zoom1Btn;
  card.zoom4Btn = zoom4Btn;
  card.rate05Btn = rate05Btn;
  card.rate1Btn = rate1Btn;

  wireCard(card);
  confSelect.value = card.state.confidence;
  notesInput.value = card.state.notes;
  setChannel(card, "gen");
  setZoom(card, 0);
  updateRateButtons(card);
  updateSaveIndicator(card);
  renderCard(card);

  return card;
}

function wireCard(card) {
  card.genBtn.addEventListener("click", function () { setChannel(card, "gen"); card.el.focus(); });
  card.turBtn.addEventListener("click", function () { setChannel(card, "tur"); card.el.focus(); });
  card.zoom1Btn.addEventListener("click", function () { setZoom(card, 0); card.el.focus(); });
  card.zoom4Btn.addEventListener("click", function () { setZoom(card, 1); card.el.focus(); });
  card.rate05Btn.addEventListener("click", function () { setRate(card, 0.5); card.el.focus(); });
  card.rate1Btn.addEventListener("click", function () { setRate(card, 1.0); card.el.focus(); });

  card.canvasEl.addEventListener("click", function (e) {
    card.el.focus();
    var rect = card.canvasEl.getBoundingClientRect();
    var xPx = e.clientX - rect.left;
    var frac = clamp(rect.width > 0 ? xPx / rect.width : 0, 0, 1);
    var offsetS = frac * card.meta.duration_s;
    if (e.shiftKey) {
      seekTo(card, offsetS);
    } else {
      addMarker(card, offsetS);
      seekTo(card, offsetS);
    }
  });

  card.el.addEventListener("keydown", function (e) {
    var tag = e.target && e.target.tagName;
    var skipTags = ["SELECT", "INPUT", "TEXTAREA", "BUTTON", "AUDIO"];
    if (skipTags.indexOf(tag) !== -1) return;
    if (e.key === " " || e.key === "Spacebar") {
      e.preventDefault();
      togglePlay(card);
    } else if (e.key === "m" || e.key === "M") {
      addMarker(card, card.audioEl.currentTime);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      seekTo(card, card.audioEl.currentTime - (e.shiftKey ? ARROW_STEP_FINE_S : ARROW_STEP_S));
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      seekTo(card, card.audioEl.currentTime + (e.shiftKey ? ARROW_STEP_FINE_S : ARROW_STEP_S));
    }
  });

  card.audioEl.addEventListener("play", function () { renderCard(card); });
  card.audioEl.addEventListener("pause", function () { renderCard(card); });
  card.audioEl.addEventListener("seeked", function () { renderCard(card); });
  card.imgEl.addEventListener("load", function () {
    card.naturalWidth = card.imgEl.naturalWidth || flatWidthPx(card.meta.duration_s);
    card.naturalHeight = card.imgEl.naturalHeight || FLAT_HEIGHT_PX;
    applyZoom(card);
  });

  card.confSelect.addEventListener("change", function () {
    card.state.confidence = card.confSelect.value;
    saveState(card);
  });
  card.notesInput.addEventListener("input", function () {
    card.state.notes = card.notesInput.value;
    saveState(card);
  });
}

function setChannel(card, channel) {
  var wasPlaying = !card.audioEl.paused;
  var t = card.audioEl.currentTime || 0;
  card.channel = channel;
  var wavPath = channel === "gen" ? card.meta.gen_wav : card.meta.tur_wav;
  var pngPath = channel === "gen" ? card.meta.gen_flat_png : card.meta.tur_flat_png;

  function onLoaded() {
    card.audioEl.removeEventListener("loadedmetadata", onLoaded);
    var dur = isFinite(card.audioEl.duration) && card.audioEl.duration > 0
      ? card.audioEl.duration : card.meta.duration_s;
    card.audioEl.currentTime = clamp(t, 0, dur);
    if (wasPlaying) { card.audioEl.play().catch(function () {}); }
    renderCard(card);
  }
  card.audioEl.addEventListener("loadedmetadata", onLoaded);
  card.audioEl.src = wavPath;
  card.audioEl.load();
  card.imgEl.src = pngPath;
  updateChannelButtons(card);
}

function updateChannelButtons(card) {
  card.genBtn.classList.toggle("active", card.channel === "gen");
  card.turBtn.classList.toggle("active", card.channel === "tur");
}

function setZoom(card, zoomIndex) {
  card.zoomIndex = zoomIndex;
  card.zoom1Btn.classList.toggle("active", zoomIndex === 0);
  card.zoom4Btn.classList.toggle("active", zoomIndex === 1);
  applyZoom(card);
}

function applyZoom(card) {
  var factor = ZOOM_LEVELS[card.zoomIndex] || 1;
  var w = (card.naturalWidth || flatWidthPx(card.meta.duration_s)) * factor;
  var h = card.naturalHeight || FLAT_HEIGHT_PX;
  card.wrapperEl.style.width = w + "px";
  card.wrapperEl.style.height = h + "px";
  card.canvasEl.width = w;
  card.canvasEl.height = h;
  drawOverlay(card);
}

function setRate(card, rate) {
  card.audioEl.playbackRate = rate;
  updateRateButtons(card);
}

function updateRateButtons(card) {
  var rate = card.audioEl.playbackRate || 1;
  card.rate05Btn.classList.toggle("active", Math.abs(rate - 0.5) < 0.001);
  card.rate1Btn.classList.toggle("active", Math.abs(rate - 1.0) < 0.001);
}

function togglePlay(card) {
  if (card.audioEl.paused) { card.audioEl.play().catch(function () {}); }
  else { card.audioEl.pause(); }
}

function seekTo(card, t) {
  var dur = isFinite(card.audioEl.duration) && card.audioEl.duration > 0
    ? card.audioEl.duration : card.meta.duration_s;
  card.audioEl.currentTime = clamp(t, 0, dur);
  renderCard(card);
}

function addMarker(card, offsetS) {
  var dur = card.meta.duration_s;
  var off = clamp(Math.round(offsetS * 1000) / 1000, 0, dur);
  card.state.markers.push(off);
  card.state.markers.sort(function (a, b) { return a - b; });
  saveState(card);
  renderCard(card);
}

function deleteMarker(card, index) {
  card.state.markers.splice(index, 1);
  saveState(card);
  renderCard(card);
}

function renderCard(card) {
  renderChips(card);
  renderCount(card);
  drawOverlay(card);
  updateTimeDisplay(card);
}

function renderChips(card) {
  card.chipsEl.innerHTML = "";
  card.state.markers.forEach(function (off, i) {
    var chip = el("button", { type: "button", class: "marker-chip", title: "Click to delete" });
    chip.textContent = "#" + (i + 1) + " " + off.toFixed(3) + "s";
    chip.addEventListener("click", function () { deleteMarker(card, i); });
    card.chipsEl.appendChild(chip);
  });
}

function renderCount(card) {
  var n = card.state.markers.length;
  if (card.meta.expected_strikes === "sweep") {
    card.countEl.textContent = n + " markers (sweep)";
    card.countEl.classList.remove("count-complete");
  } else {
    card.countEl.textContent = n + "/" + card.meta.expected_strikes;
    var complete = String(n) === String(card.meta.expected_strikes);
    card.countEl.classList.toggle("count-complete", complete);
  }
}

function updateTimeDisplay(card) {
  var t = card.audioEl.currentTime || 0;
  card.timeDisplayEl.textContent = formatClock(t) + " / " + formatClock(card.meta.duration_s);
  card.utcLiveEl.textContent = offsetToAbsoluteUtcIso(card.meta.snippet_start_utc, t);
}

function updateSaveIndicator(card) {
  card.saveIndicatorEl.classList.remove("save-error");
  if (card.saveStatus === "saved") {
    card.saveIndicatorEl.textContent = "saved " + nowClock();
  } else if (card.saveStatus === "restored") {
    card.saveIndicatorEl.textContent = "restored (previously saved)";
  } else if (card.saveStatus === "error") {
    card.saveIndicatorEl.textContent = "save failed (localStorage unavailable)";
    card.saveIndicatorEl.classList.add("save-error");
  } else {
    card.saveIndicatorEl.textContent = "";
  }
}

function drawOverlay(card) {
  var ctx = card.canvasEl.getContext("2d");
  var w = card.canvasEl.width;
  var h = card.canvasEl.height;
  ctx.clearRect(0, 0, w, h);
  var dur = card.meta.duration_s;
  if (!dur || dur <= 0) return;

  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.font = "11px monospace";
  ctx.lineWidth = 1;
  for (var t = 0; t <= dur + 1e-6; t += 5) {
    var x = (t / dur) * w;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, h);
    ctx.stroke();
    ctx.fillText(Math.round(t) + "s", x + 3, 12);
  }

  card.state.markers.forEach(function (off, i) {
    var x = (off / dur) * w;
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.fillStyle = "#f59e0b";
    ctx.fillRect(x, h - 16, 18, 16);
    ctx.fillStyle = "#1a1200";
    ctx.fillText("" + (i + 1), x + 3, h - 4);
  });

  var t2 = card.audioEl.currentTime || 0;
  if (t2 >= 0 && t2 <= dur) {
    var xph = (t2 / dur) * w;
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xph, 0);
    ctx.lineTo(xph, h);
    ctx.stroke();
  }
}

function tick() {
  Object.keys(CARDS).forEach(function (key) {
    var card = CARDS[key];
    if (!card.audioEl.paused && !card.audioEl.ended) {
      drawOverlay(card);
      updateTimeDisplay(card);
    }
  });
  requestAnimationFrame(tick);
}

function exportSessionCsv(sessionKey) {
  var cards = CARDS_BY_SESSION[sessionKey] || [];
  var lines = [MARKS_CSV_HEADER.join(",")];
  cards.forEach(function (card) {
    var offsets = card.state.markers.slice().sort(function (a, b) { return a - b; });
    offsets.forEach(function (off, i) {
      var utcIso = offsetToAbsoluteUtcIso(card.meta.snippet_start_utc, off);
      var row = [
        card.meta.session, card.meta.event_id, card.meta.kind,
        String(i + 1), off.toFixed(3), utcIso,
        card.state.confidence || "", card.state.notes || "",
      ];
      lines.push(row.map(csvField).join(","));
    });
  });
  return lines.join("\r\n") + "\r\n";
}

function importSessionCsvText(sessionKey, text) {
  var records = parseCsv(text);
  var bySessionEvent = {};
  records.forEach(function (r) {
    if (r.session !== sessionKey) return;
    var k = r.event_id;
    if (!bySessionEvent[k]) bySessionEvent[k] = [];
    bySessionEvent[k].push(r);
  });
  var cards = CARDS_BY_SESSION[sessionKey] || [];
  var touched = 0;
  cards.forEach(function (card) {
    var rows = bySessionEvent[card.meta.event_id];
    if (!rows || rows.length === 0) return;
    rows.sort(function (a, b) { return Number(a.strike_no) - Number(b.strike_no); });
    card.state.markers = rows.map(function (r) { return Number(r.offset_s); });
    card.state.confidence = rows[0].confidence || "";
    card.state.notes = rows[0].notes || "";
    card.confSelect.value = card.state.confidence;
    card.notesInput.value = card.state.notes;
    saveState(card);
    renderCard(card);
    touched++;
  });
  return touched;
}

function buildApp() {
  var app = document.getElementById("app");
  app.innerHTML = "";

  var bySession = {};
  var sessionOrder = [];
  EVENTS_META.forEach(function (meta) {
    if (!bySession[meta.session]) { bySession[meta.session] = []; sessionOrder.push(meta.session); }
    bySession[meta.session].push(meta);
  });
  sessionOrder.sort();

  sessionOrder.forEach(function (sessionKey) {
    var metas = bySession[sessionKey];
    var section = el("section", { class: "session-section" });
    var label = SESSION_LABELS[sessionKey] || sessionKey;
    section.appendChild(el("h2", { text: "Session " + sessionKey + " – " + label }));

    var toolbar = el("div", { class: "session-toolbar" });
    var exportBtn = el("button", { type: "button", class: "export-btn", text: "Export CSV" });
    var importBtn = el("button", { type: "button", class: "import-btn", text: "Import CSV" });
    var importInput = el("input", { type: "file", accept: ".csv,text/csv", class: "import-input" });
    importInput.style.display = "none";
    var status = el("span", { class: "session-status" });
    toolbar.appendChild(exportBtn);
    toolbar.appendChild(importBtn);
    toolbar.appendChild(importInput);
    toolbar.appendChild(status);
    section.appendChild(toolbar);

    exportBtn.addEventListener("click", function () {
      var csvText = exportSessionCsv(sessionKey);
      downloadTextFile("annotation_marks_" + sessionKey + ".csv", csvText);
      status.textContent = "exported " + nowClock();
    });
    importBtn.addEventListener("click", function () { importInput.click(); });
    importInput.addEventListener("change", function () {
      var file = importInput.files && importInput.files[0];
      if (!file) return;
      var reader = new FileReader();
      reader.onload = function () {
        var n = importSessionCsvText(sessionKey, String(reader.result));
        status.textContent = "imported: " + n + " event(s) updated (" + nowClock() + ")";
      };
      reader.readAsText(file);
      importInput.value = "";
    });

    CARDS_BY_SESSION[sessionKey] = [];
    metas.forEach(function (meta) {
      var card = buildCard(meta);
      CARDS[cardKey(meta.session, meta.event_id)] = card;
      CARDS_BY_SESSION[sessionKey].push(card);
      section.appendChild(card.el);
    });

    app.appendChild(section);
  });
}

buildApp();
requestAnimationFrame(tick);

window.StrikeAnnotator = {
  getState: function (session, eventId) {
    var card = CARDS[cardKey(session, eventId)];
    if (!card) return null;
    return {
      markers: card.state.markers.slice(),
      confidence: card.state.confidence,
      notes: card.state.notes,
    };
  },
  getMeta: function (session, eventId) {
    var card = CARDS[cardKey(session, eventId)];
    return card ? card.meta : null;
  },
  addMarkerAtOffset: function (session, eventId, offsetS) {
    var card = CARDS[cardKey(session, eventId)];
    if (!card) return null;
    addMarker(card, offsetS);
    return card.state.markers.slice();
  },
  clickAtFraction: function (session, eventId, fraction, shiftKey) {
    var card = CARDS[cardKey(session, eventId)];
    if (!card) return null;
    var rect = card.canvasEl.getBoundingClientRect();
    var evt = new MouseEvent("click", {
      clientX: rect.left + fraction * rect.width,
      clientY: rect.top + rect.height / 2,
      bubbles: true, cancelable: true, shiftKey: !!shiftKey,
    });
    card.canvasEl.dispatchEvent(evt);
    return card.state.markers.slice();
  },
  focusCard: function (session, eventId) {
    var card = CARDS[cardKey(session, eventId)];
    if (card) card.el.focus();
  },
  getPlayheadPx: function (session, eventId) {
    var card = CARDS[cardKey(session, eventId)];
    if (!card) return null;
    var t = card.audioEl.currentTime || 0;
    return (t / card.meta.duration_s) * card.canvasEl.width;
  },
  exportSessionCsv: exportSessionCsv,
  importSessionCsvText: importSessionCsvText,
  cardKey: cardKey,
  cards: CARDS,
};
})();
"""


def _json_script_safe(value: object) -> str:
    """`json.dumps(value)` with `</` escaped to `<\\/` -- defends the `<script
    type="application/json">` blocks `render_interactive_index_html` embeds
    this into against a literal `</script>` substring inside the data
    prematurely closing the tag (defensive: nothing currently written here can
    contain one, but cheap to guard regardless).
    """
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def render_interactive_index_html(out_dir: Path) -> Path:
    """(Re)generate the INTERACTIVE `<out_dir>/index.html` -- Stefan's actual
    per-strike annotation workflow entrypoint (audio + borderless flat
    spectrogram + click-to-mark + autosave; see the "Interactive index.html"
    section banner above). Fully self-contained (inline CSS/JS, no external
    resources, no `fetch()` calls): the ONLY data Python injects is
    `events_meta_from_templates`'s own list PLUS each session's label,
    both embedded as `<script type="application/json">` blocks and read by
    the inline JS at page-load time via `JSON.parse` -- deliberately NOT
    `fetch()`-ed from the sibling `events_meta.json` file, because Chrome's
    file:// origin blocks `fetch()`/XHR of local files with a CORS error
    while ordinary `<img>`/`<audio src>` resource loads are NOT subject to
    that restriction; embedding sidesteps the failure mode entirely so the
    SAME page works identically opened directly (`file://.../index.html`) or
    served (`python3 -m http.server`). Every card's spectrogram image/audio
    element still references the real WAV/PNG files by RELATIVE path (same
    "never base64-embed" rationale as `render_static_index_html`).

    The OLD read-only overview now lives at `index_static.html`
    (`render_static_index_html`) -- this function's own output REPLACES
    `index.html` as the real workflow entrypoint (`build_kit` calls both).
    """
    metas = events_meta_from_templates(out_dir)
    sessions = sorted({m.session for m in metas})
    labels = {s: (_SESSION_CONFIG[s].label if s in _SESSION_CONFIG else s) for s in sessions}
    events_json = _json_script_safe([m.to_dict() for m in metas])
    labels_json = _json_script_safe(labels)

    doc = (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        "<title>080726 Strike Annotation (interactive)</title>\n"
        f"<style>{_INTERACTIVE_CSS}</style>\n</head>\n<body>\n"
        "<h1>08.07.2026 &ndash; Schonhammer strike annotation (interactive)</h1>\n"
        f"{_INTERACTIVE_INSTRUCTIONS_HTML}\n"
        '<div id="app">Loading events&hellip;</div>\n'
        f'<script id="events-meta-data" type="application/json">{events_json}</script>\n'
        f'<script id="session-labels-data" type="application/json">{labels_json}</script>\n'
        f"<script>{_INTERACTIVE_JS}</script>\n"
        "</body>\n</html>\n"
    )
    out_path = out_dir / "index.html"
    out_path.write_text(doc, encoding="utf-8")
    logger.info("annotation_kit: wrote %s (%d event(s), interactive)", out_path, len(metas))
    return out_path


# ---------------------------------------------------------------------------
# events_meta.json -- the interactive index.html's own data source (embedded
# inline at build time, see `render_interactive_index_html`) AND `compile-
# marks`' source of per-event `duration_s`/`snippet_start_utc` truth (see
# `load_events_meta`). Derived entirely from the `annotation_template_
# <session>.csv` files already on disk -- no new data, just a re-shaped,
# machine-friendly (JSON, not the template's wide per-event-row CSV) view of
# the SAME fields, plus the two flat-PNG paths (`_wav_to_flat_png`, derived,
# never a separate column).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventMeta:
    """One event's worth of everything the interactive UI (or `compile-marks`)
    needs WITHOUT re-deriving it from a template CSV: `session`/`event_id`/
    `kind`/`expected_strikes` identify and describe the event; `snippet_start_
    utc`/`duration_s` are the offset<->UTC conversion's own two inputs;
    `gen_wav`/`tur_wav`/`gen_flat_png`/`tur_flat_png` are paths RELATIVE to the
    kit root (same convention as `TemplateRow.gen_wav`/`.tur_wav`)."""

    session: str
    event_id: str
    kind: str
    snippet_start_utc: datetime
    duration_s: float
    gen_wav: str
    tur_wav: str
    gen_flat_png: str
    tur_flat_png: str
    expected_strikes: str

    def to_dict(self) -> dict[str, str | float]:
        return {
            "session": self.session,
            "event_id": self.event_id,
            "kind": self.kind,
            "snippet_start_utc": self.snippet_start_utc.isoformat(),
            "duration_s": self.duration_s,
            "gen_wav": self.gen_wav,
            "tur_wav": self.tur_wav,
            "gen_flat_png": self.gen_flat_png,
            "tur_flat_png": self.tur_flat_png,
            "expected_strikes": self.expected_strikes,
        }


def _event_meta_from_row(row: Mapping[str, str]) -> EventMeta:
    """One `EventMeta` from an `annotation_template_<session>.csv` row (as
    normalized by `read_template_csv`/`_cell_to_str` -- every cell a plain
    `str`). `duration_s` is derived from the row's own `snippet_start_utc`/
    `snippet_end_utc` pair, NOT stored separately anywhere.

    Raises:
        ValueError: if `snippet_start_utc`/`snippet_end_utc` do not parse as
            ISO-8601 timestamps.
    """
    event_id = row["event_id"]
    try:
        snippet_start = datetime.fromisoformat(row["snippet_start_utc"])
        snippet_end = datetime.fromisoformat(row["snippet_end_utc"])
    except ValueError as exc:
        raise ValueError(f"event {event_id}: malformed snippet timestamp: {exc}") from exc
    return EventMeta(
        session=row["session"], event_id=event_id, kind=row["kind"],
        snippet_start_utc=snippet_start,
        duration_s=(snippet_end - snippet_start).total_seconds(),
        gen_wav=row["gen_wav"], tur_wav=row["tur_wav"],
        gen_flat_png=_wav_to_flat_png(row["gen_wav"]),
        tur_flat_png=_wav_to_flat_png(row["tur_wav"]),
        expected_strikes=row["expected_strikes"],
    )


def events_meta_from_templates(out_dir: Path) -> list[EventMeta]:
    """Every event currently described by an `annotation_template_<session>.csv`
    in *out_dir*, as `EventMeta` -- mirrors `render_static_index_html`'s own
    "rebuild from whatever's on disk" contract: reads every template CSV
    currently present (sorted by filename, i.e. session), not just the
    session(s) the triggering `build` call itself just wrote.
    """
    metas: list[EventMeta] = []
    for template_path in sorted(out_dir.glob("annotation_template_*.csv")):
        df = read_template_csv(template_path)
        for record in df.to_dict(orient="records"):
            row = {str(k): _cell_to_str(v) for k, v in record.items()}
            metas.append(_event_meta_from_row(row))
    return metas


def write_events_meta_json(out_dir: Path) -> Path:
    """(Re)write `<out_dir>/events_meta.json`: a JSON array of every current
    `events_meta_from_templates(out_dir)` entry, in that same (session-then-
    event-id) order. `ensure_ascii=False` -- every field here is already plain
    ASCII (session codes, zero-padded ids, kind slugs, ISO timestamps, POSIX
    relative paths), so this only matters if that ever changes; kept for
    consistency with the rest of the kit's UTF-8 (`encoding="utf-8"`) files.
    """
    metas = events_meta_from_templates(out_dir)
    path = out_dir / "events_meta.json"
    path.write_text(
        json.dumps([m.to_dict() for m in metas], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("annotation_kit: wrote %s (%d event(s))", path, len(metas))
    return path


def load_events_meta(path: Path) -> dict[tuple[str, str], EventMeta]:
    """Read an `events_meta.json` (as written by `write_events_meta_json`) into
    a `{(session, event_id): EventMeta}` lookup -- `compile_marks`' own source
    of per-event `duration_s`/`snippet_start_utc` truth (an exported
    `annotation_marks_<session>.csv` carries neither).

    Raises:
        ValueError: if *path* does not exist, is not valid JSON, is not a JSON
            array, or any entry is missing/mistyped a required field.
    """
    if not path.is_file():
        raise ValueError(f"events_meta.json not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"events_meta.json is not valid JSON ({path}): {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"events_meta.json must be a JSON array ({path})")

    lookup: dict[tuple[str, str], EventMeta] = {}
    for i, entry in enumerate(raw):
        try:
            meta = EventMeta(
                session=str(entry["session"]), event_id=str(entry["event_id"]),
                kind=str(entry["kind"]),
                snippet_start_utc=datetime.fromisoformat(str(entry["snippet_start_utc"])),
                duration_s=float(entry["duration_s"]),
                gen_wav=str(entry["gen_wav"]), tur_wav=str(entry["tur_wav"]),
                gen_flat_png=str(entry["gen_flat_png"]), tur_flat_png=str(entry["tur_flat_png"]),
                expected_strikes=str(entry["expected_strikes"]),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"events_meta.json entry #{i} ({path}): {exc}") from exc
        lookup[(meta.session, meta.event_id)] = meta
    return lookup


# ---------------------------------------------------------------------------
# compile side
# ---------------------------------------------------------------------------


def parse_extra_offsets(raw: str) -> list[float]:
    """Parse the `extra_offsets_s` template cell: `;`-separated seconds-offsets
    (e.g. `"12.3;12.9"`), in the order given. Empty/whitespace-only -> `[]`.
    Empty tokens between two `;` (a trailing `;`, or `;;`) are skipped rather
    than rejected -- a harmless typo, not a data error.

    Raises:
        ValueError: if a non-empty token does not parse as a float.
    """
    text = raw.strip()
    if not text:
        return []
    offsets: list[float] = []
    for raw_token in text.split(";"):
        token = raw_token.strip()
        if not token:
            continue
        try:
            offsets.append(float(token))
        except ValueError as exc:
            raise ValueError(f"extra_offsets_s: {token!r} is not a number (in {raw!r})") from exc
    return offsets


_PRIMARY_OFFSET_COLUMNS = ("strike1_offset_s", "strike2_offset_s", "strike3_offset_s")


def collect_offsets(row: Mapping[str, str]) -> list[float]:
    """Every offset a filled template row provides, in the CHRONOLOGICAL entry
    order `validate_offsets` checks: `strike1_offset_s`, `strike2_offset_s`,
    `strike3_offset_s` (each only if non-empty -- a gap, e.g. strike1 empty but
    strike2/3 filled, is allowed: it just means 2 strikes were confidently
    heard, not 3), THEN `extra_offsets_s`'s own parsed values appended in the
    order given. Deliberately NOT re-sorted by value: an offset typed into the
    wrong column is caught by `validate_offsets`'s monotonic check as an
    ordering violation, rather than silently reordered into a plausible-looking
    but wrong sequence.

    Raises:
        ValueError: if a non-empty offset cell does not parse as a float.
    """
    offsets: list[float] = []
    for col in _PRIMARY_OFFSET_COLUMNS:
        text = row[col].strip()
        if not text:
            continue
        try:
            offsets.append(float(text))
        except ValueError as exc:
            raise ValueError(f"{col}: {text!r} is not a number") from exc
    offsets.extend(parse_extra_offsets(row["extra_offsets_s"]))
    return offsets


def validate_offsets(offsets: Sequence[float], snippet_duration_s: float, *, event_id: str) -> None:
    """Raise ValueError unless every offset in *offsets* lies in
    `[0, snippet_duration_s]` AND the sequence is STRICTLY increasing (in the
    order given -- `collect_offsets`'s own column-then-extras order, i.e. the
    order Stefan is expected to enter strikes in as he hears them
    chronologically). Two strikes sharing the same offset is treated as an
    entry mistake, not a real simultaneous double-strike -- rejected, not
    silently accepted. Assumes *offsets* is already known non-empty (an
    unannotated/no-strikes-heard event should skip validation entirely, not
    call this with `[]`).
    """
    for i, off in enumerate(offsets, start=1):
        if not (0.0 <= off <= snippet_duration_s):
            raise ValueError(
                f"event {event_id}: offset #{i} = {off}s is out of range "
                f"[0, {snippet_duration_s}s]"
            )
    for i in range(1, len(offsets)):
        if offsets[i] <= offsets[i - 1]:
            raise ValueError(
                f"event {event_id}: offsets must be strictly increasing -- entry #{i + 1} = "
                f"{offsets[i]}s is not after entry #{i} = {offsets[i - 1]}s"
            )


@dataclass(frozen=True)
class CompiledStrike:
    session: str
    event_id: str
    kind: str
    strike_no: int
    strike_utc: datetime
    confidence: str
    notes: str


def compile_row(row: Mapping[str, str]) -> list[CompiledStrike]:
    """Every strike a single (filled) template row yields, as absolute-UTC
    `CompiledStrike`s -- `[]` if the row carries no offsets at all (an event
    Stefan has not annotated yet, or genuinely heard zero strikes; not an
    error).

    Raises:
        ValueError: if `snippet_start_utc`/`snippet_end_utc` do not parse as
            ISO-8601 timestamps, or (via `validate_offsets`) an offset is out
            of range or not strictly increasing.
    """
    event_id = row["event_id"]
    try:
        snippet_start = datetime.fromisoformat(row["snippet_start_utc"])
        snippet_end = datetime.fromisoformat(row["snippet_end_utc"])
    except ValueError as exc:
        raise ValueError(f"event {event_id}: malformed snippet timestamp: {exc}") from exc
    duration_s = (snippet_end - snippet_start).total_seconds()

    offsets = collect_offsets(row)
    if not offsets:
        return []
    validate_offsets(offsets, duration_s, event_id=event_id)

    return [
        CompiledStrike(
            session=row["session"], event_id=event_id, kind=row["kind"],
            strike_no=i, strike_utc=snippet_start + timedelta(seconds=off),
            confidence=row["confidence"], notes=row["notes"],
        )
        for i, off in enumerate(offsets, start=1)
    ]


def compile_template(df: pd.DataFrame) -> list[CompiledStrike]:
    """Every strike every (filled) row of *df* (an `annotation_template_
    <session>.csv`, as read by `read_template_csv`) yields, in row order.

    Raises:
        ValueError: if a required column is missing, the template mixes more
            than one `session` value, or any row fails `compile_row`'s own
            validation.
    """
    missing = [c for c in TEMPLATE_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"template is missing column(s): {', '.join(missing)}")

    sessions = {str(v).strip() for v in df["session"]} - {""}
    if len(sessions) > 1:
        raise ValueError(f"template mixes multiple sessions: {sorted(sessions)}")

    strikes: list[CompiledStrike] = []
    for record in df.to_dict(orient="records"):
        row = {str(k): _cell_to_str(v) for k, v in record.items()}
        strikes.extend(compile_row(row))
    return strikes


def read_template_csv(path: Path) -> pd.DataFrame:
    """Read an `annotation_template_<session>.csv` (or a Stefan-filled copy of
    one): skips `#`-comment lines (mirrors `make_demo_assets._load_events_csv`'s
    own convention) and reads every column as plain `str` with empty cells kept
    as `""` (never promoted to a pandas NaN/float) -- `compile_row`/
    `collect_offsets` depend on this to distinguish "not filled in" from a
    genuine numeric value without also having to special-case NaN.
    """
    return pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)


def write_compiled_csv(
    strikes: Sequence[CompiledStrike],
    out_path: Path,
    *,
    source_path: Path,
    compiled_date: date,
    provenance: str,
    command_name: str,
) -> Path:
    """Write *strikes* to *out_path* as a `docs/groundtruth`-style CSV: a `#`-
    prefixed provenance comment (*provenance* -- differs between `compile`'s
    "manual template annotation" and `compile-marks`' "interactive per-strike
    annotation", see each subcommand's own call site in `main`), the source
    file (*source_path* -- a filled template for `compile`, an exported marks
    CSV for `compile-marks`), and the *compiled_date* passed via `--date`
    (never `date.today()`, so identical inputs always produce byte-identical
    output), followed by the `COMPILED_CSV_COLUMNS` header and one row per
    strike, in *strikes*'s own order. *command_name* is the exact subcommand
    invocation quoted back into the "Compiled:" line, so the comment always
    names the RIGHT command for how the file was actually produced.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comment = (
        f"# {out_path.name} -- seconds-level per-strike ground truth for the 08.07.2026 "
        "induced-Schonhammer-strike campaign (upgrades docs/groundtruth/080726_events_"
        "{st,pu}.csv from minute-level to per-strike UTC timestamps).\n"
        f"# Provenance: {provenance}\n"
        f"# Source: {source_path}\n"
        f"# Compiled: {compiled_date.isoformat()} (scripts/annotation_kit.py {command_name}, "
        "--date kept explicit for reproducibility).\n"
    )
    frame = pd.DataFrame(
        [
            {
                "session": s.session, "event_id": s.event_id, "kind": s.kind,
                "strike_no": s.strike_no, "strike_utc": s.strike_utc.isoformat(),
                "confidence": s.confidence, "notes": s.notes,
            }
            for s in strikes
        ],
        columns=list(COMPILED_CSV_COLUMNS),
    )
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(comment)
        frame.to_csv(fh, index=False)
    return out_path


# ---------------------------------------------------------------------------
# compile-marks side -- the interactive index.html's "Export CSV" output
# (`annotation_marks_<session>.csv`, one row per STRIKE already, unlike the
# legacy per-EVENT template) compiled into the same `COMPILED_CSV_COLUMNS`
# ground-truth shape `compile_template` produces, via `load_events_meta` for
# the per-event `duration_s`/`snippet_start_utc` the marks CSV itself does not
# carry.
# ---------------------------------------------------------------------------

MARKS_CSV_COLUMNS = (
    "session", "event_id", "kind", "strike_no", "offset_s", "strike_utc", "confidence", "notes",
)
"""`annotation_marks_<session>.csv`'s exact column contract (index.html's
"Export CSV" button, `exportSessionCsv` in `_INTERACTIVE_JS`), in this order."""


def read_marks_csv(path: Path) -> pd.DataFrame:
    """Read an exported `annotation_marks_<session>.csv`: same `#`-comment-
    skipping, all-`str`, empty-cells-stay-empty convention as
    `read_template_csv` (the browser export never writes a `#` header today,
    but skipping one is harmless if a hand-edited copy ever adds one).
    """
    return pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)


def compile_marks(
    df: pd.DataFrame, events: Mapping[tuple[str, str], EventMeta]
) -> list[CompiledStrike]:
    """Every strike in *df* (an exported `annotation_marks_<session>.csv`, as
    read by `read_marks_csv`) as absolute-UTC `CompiledStrike`s, validated
    against *events* (`load_events_meta`'s own lookup).

    For each event_id present in *df*: rows are ordered by their own
    `strike_no` (so a hand-reordered file is still compiled in the INTENDED
    chronological order, not raw row order), offsets are re-validated with
    `validate_offsets` (REUSED unchanged from the legacy `compile` path --
    same `[0, duration_s]`-range + strictly-increasing contract) against the
    event's `duration_s` from *events*, and the output `strike_no`/`kind` are
    always freshly assigned (1..N by validated chronological order / the
    event's own `EventMeta.kind`) rather than trusted from the input row --
    the marks CSV's `strike_no`/`kind` columns are for human readability only.

    Raises:
        ValueError: if a required column is missing, *df* mixes more than one
            `session`, a row's `event_id` has no matching entry in *events*,
            `strike_no`/`offset_s` do not parse, or (via `validate_offsets`) an
            offset is out of range or not strictly increasing for its event.
    """
    missing = [c for c in MARKS_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"marks CSV is missing column(s): {', '.join(missing)}")

    sessions = {str(v).strip() for v in df["session"]} - {""}
    if len(sessions) > 1:
        raise ValueError(f"marks CSV mixes multiple sessions: {sorted(sessions)}")

    rows = [{str(k): _cell_to_str(v) for k, v in r.items()} for r in df.to_dict(orient="records")]
    by_event: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_event.setdefault(row["event_id"], []).append(row)

    strikes: list[CompiledStrike] = []
    for event_id, event_rows in by_event.items():
        session = event_rows[0]["session"]
        key = (session, event_id)
        if key not in events:
            raise ValueError(
                f"event {event_id}: no matching entry in events_meta.json (session {session!r})"
            )
        meta = events[key]

        try:
            ordered = sorted(event_rows, key=lambda r: int(r["strike_no"]))
        except ValueError as exc:
            raise ValueError(f"event {event_id}: non-integer strike_no: {exc}") from exc

        offsets: list[float] = []
        for r in ordered:
            try:
                offsets.append(float(r["offset_s"]))
            except ValueError as exc:
                raise ValueError(
                    f"event {event_id}: offset_s {r['offset_s']!r} is not a number"
                ) from exc
        validate_offsets(offsets, meta.duration_s, event_id=event_id)

        strikes.extend(
            CompiledStrike(
                session=session, event_id=event_id, kind=meta.kind,
                strike_no=i, strike_utc=meta.snippet_start_utc + timedelta(seconds=off),
                confidence=r["confidence"], notes=r["notes"],
            )
            for i, (r, off) in enumerate(zip(ordered, offsets, strict=True), start=1)
        )

    strikes.sort(key=lambda s: (s.event_id, s.strike_no))
    return strikes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build (build) or compile (compile) the 080726 strike-annotation kit: "
            "seconds-level per-strike ground truth from the minute-level "
            "docs/groundtruth/080726_events_{st,pu}.csv."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build",
        help="Cut +/-15s-padded WAV clips + spectrogram PNGs for every 080726 strike "
        "event and write an annotation_template_<session>.csv + index.html.",
    )
    build.add_argument(
        "--session", choices=["st", "pu", "both"], default="both",
        help="Which session(s) to build (default: both).",
    )
    build.add_argument(
        "--out", type=Path, default=DEFAULT_KIT_DIR,
        help=f"Kit output directory (default: {DEFAULT_KIT_DIR}).",
    )

    compile_p = sub.add_parser(
        "compile",
        help="LEGACY: convert a FILLED annotation_template_<session>.csv (hand-edited "
        "offset columns) into absolute-UTC per-strike ground truth.",
    )
    compile_p.add_argument(
        "--template", type=Path, required=True,
        help="Path to the filled annotation_template_<session>.csv.",
    )
    compile_p.add_argument(
        "--out", type=Path, required=True,
        help="Output CSV path, e.g. docs/groundtruth/080726_strikes_seconds_<session>.csv.",
    )
    compile_p.add_argument(
        "--date", type=date.fromisoformat, required=True,
        help="Compile date (YYYY-MM-DD), written into the output's provenance comment -- "
        "passed explicitly (never today's real date) so the same inputs always produce "
        "byte-identical output.",
    )

    compile_marks_p = sub.add_parser(
        "compile-marks",
        help="Convert an EXPORTED annotation_marks_<session>.csv (index.html's interactive "
        "per-strike 'Export CSV' button) into absolute-UTC per-strike ground truth.",
    )
    compile_marks_p.add_argument(
        "--csv", type=Path, required=True,
        help="Path to the exported annotation_marks_<session>.csv.",
    )
    compile_marks_p.add_argument(
        "--out", type=Path, required=True,
        help="Output CSV path, e.g. docs/groundtruth/080726_strikes_seconds_<session>.csv.",
    )
    compile_marks_p.add_argument(
        "--date", type=date.fromisoformat, required=True,
        help="Compile date (YYYY-MM-DD), written into the output's provenance comment -- "
        "passed explicitly (never today's real date) so the same inputs always produce "
        "byte-identical output.",
    )
    compile_marks_p.add_argument(
        "--events-meta", type=Path, default=DEFAULT_KIT_DIR / "events_meta.json",
        help="Path to events_meta.json (default: "
        f"{DEFAULT_KIT_DIR / 'events_meta.json'}), for per-event duration_s/"
        "snippet_start_utc (offset range validation + offset->UTC conversion).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        sessions = ["st", "pu"] if args.session == "both" else [args.session]
        cfg = load_config()
        try:
            results = build_kit(cfg, sessions, args.out)
        except ValueError as exc:
            print(f"annotation_kit: {exc}", file=sys.stderr)
            return 2
        n_events = sum(len(r.rows) for r in results)
        n_wav = sum(r.n_wav for r in results)
        n_png = sum(r.n_png for r in results)
        n_flat_png = sum(r.n_flat_png for r in results)
        for r in results:
            for note in r.clamp_notes:
                print(f"annotation_kit: WARNING {r.session} event {note}")
        print(
            f"annotation_kit: wrote {n_events} event(s), {n_wav} wav(s), {n_png} labeled "
            f"png(s), {n_flat_png} flat png(s), events_meta.json, index.html "
            f"(interactive), index_static.html -> {args.out}"
        )
        return 0

    if args.command == "compile":
        if not args.template.is_file():
            print(f"annotation_kit: template not found: {args.template}", file=sys.stderr)
            return 2
        try:
            df = read_template_csv(args.template)
            strikes = compile_template(df)
        except ValueError as exc:
            print(f"annotation_kit: {exc}", file=sys.stderr)
            return 2
        write_compiled_csv(
            strikes, args.out, source_path=args.template, compiled_date=args.date,
            provenance=(
                "manual audio/spectrogram annotation against results/annotation-kit/080726/ "
                "(scripts/annotation_kit.py build), hand-filled into the per-event template."
            ),
            command_name="compile --template <template>",
        )
        print(
            f"annotation_kit: wrote {len(strikes)} strike(s) from {len(df)} event row(s) "
            f"-> {args.out}"
        )
        return 0

    assert args.command == "compile-marks"
    if not args.csv.is_file():
        print(f"annotation_kit: marks CSV not found: {args.csv}", file=sys.stderr)
        return 2
    try:
        events = load_events_meta(args.events_meta)
        df = read_marks_csv(args.csv)
        strikes = compile_marks(df, events)
    except ValueError as exc:
        print(f"annotation_kit: {exc}", file=sys.stderr)
        return 2
    write_compiled_csv(
        strikes, args.out, source_path=args.csv, compiled_date=args.date,
        provenance=(
            "interactive per-strike annotation (scripts/annotation_kit.py build's "
            "index.html) by the author."
        ),
        command_name="compile-marks --csv <marks csv>",
    )
    print(
        f"annotation_kit: wrote {len(strikes)} strike(s) from {len(df)} mark row(s) "
        f"-> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
