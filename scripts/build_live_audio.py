"""Build the full-session replay audio for `docs/site/live.html`: BOTH mic streams
(`RAWGeneratorMic__0`/`RAWTurbineMic__1`, channel 0 -- the same "channel 0 of each
stream" convention as `make_demo_assets.MONO_CHANNEL_INDEX`) for RUN, resampled to
16 kHz mono and compressed to AAC-in-.m4a via macOS's built-in `afconvert` (this
repo has no `ffmpeg` dependency -- `afconvert` ships with every macOS install).

Unlike `annotation_kit.py`'s +/-15 s-padded strike snippets, this covers the
ENTIRE ~4.4 h session in one run: the raw burst files are read in BOUNDED ~5 min
blocks (`annotation_kit.extract_stream_clip`, driven by this module's own
`plan_extraction_blocks`), each block is resampled and written directly into a
streaming WAV (never more than one block -- and, inside `extract_stream_clip`,
one burst file -- resident at once, the same memory discipline `annotation_kit`'s
own module docstring describes), then the finished WAV is handed to `afconvert`
for the AAC encode and deleted. A small sidecar JSON records each stream's exact
extracted start UTC + duration, so `build_live_replay.py` (which never touches
`ROWII_DATA_ROOT`) can fold "keep an `<audio>` element in sync with the replay
clock" into the payload it already assembles from precomputed artifacts alone.

Level: peak-normalized, mirroring `make_demo_assets.peak_normalize`'s target-dBFS
convention -- but a TRUE whole-session peak would require either reading every
burst file twice (once to find the peak, once to write) or buffering the entire
resampled stream on disk first; neither is warranted for what is, acoustically, a
near-stationary machine-noise recording (this session has no hammer strikes --
`build_live_replay.py`'s own module docstring). Instead the peak is ESTIMATED from
`PROBE_COUNT` short probes spread evenly across the session (covering standstill/
transition/loaded operation alike, not just the opening minutes), with
`PEAK_TARGET_DBFS` giving extra headroom (vs. `make_demo_assets.TARGET_DBFS`'s
curated-clip -1 dBFS) against under-sampling a rarer, louder transient, and a hard
int16 clip as the final safety net -- the same clip `make_demo_assets.
_write_clip_wav` already applies unconditionally. A handful of inaudibly clipped
samples at a rare, larger-than-probed transient is an acceptable trade for not
reading ~40 GB of raw burst files twice.

Pure logic (session-bounds/block-window planning, probe-instant spacing, the
replay-playhead <-> audio-offset mapping `assets/live.js` mirrors client-side, and
the transport-speed <-> playbackRate/mute mapping) is unit-tested with no
`ROWII_DATA_ROOT` dependency in `tests/test_build_live_audio.py`. Everything that
opens a burst file, writes a WAV, or shells out to `afconvert` is exercised by
actually running this script against real data instead (mirrors `annotation_kit.
py`'s own "pure vs IO-touching" split).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import subprocess
import sys
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPTS_DIR.parent / "src"
for _extra_path in (str(_SCRIPTS_DIR), str(_SRC_DIR)):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

import annotation_kit as ak  # noqa: E402
import make_demo_assets as mda  # noqa: E402

from rowii.config import load_config  # noqa: E402
from rowii.io.dataset import BurstFile, discover, run_utc_offset_ns  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = _SCRIPTS_DIR.parent
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "site" / "assets" / "live"

RUN = "290626-tu"
STREAM_BY_KEY: dict[str, str] = {"gen": "RAWGeneratorMic__0", "tur": "RAWTurbineMic__1"}
STREAM_LABEL: dict[str, str] = {"gen": "Generator mic", "tur": "Turbine mic"}

TARGET_SAMPLE_RATE_HZ = mda.TARGET_SAMPLE_RATE_HZ
"""16 kHz -- reuses `make_demo_assets`'s own constant rather than redefining it, so
the two scripts can never quietly drift apart on the WAV sample rate."""

AUDIO_BLOCK_S = 300.0
"""~5 min per extraction block (task instruction). Deliberately shorter than a
burst file (~12 min): bounds peak memory to a handful of resampled blocks' worth
of samples at a time, never the whole ~4.4 h session. The trade-off (spelled out,
not accidental): a block grid this fine re-reads some burst files more than once
(most 12-minute files straddle 2-3 successive 5-minute blocks), roughly 2-3x more
raw-file IO than a file-aligned grid would need -- accepted for the simpler, more
obviously-correct "one fixed step size, no burst-file-boundary bookkeeping" plan."""
START_MARGIN_S = 60.0
"""Seconds the search window starts BEFORE the earliest burst file's own
filename-hint start -- hints are LOCAL-time filename parses (`rowii.io.dataset`'s
own "hint" language), not exact; starting the search a bit early costs nothing
(`extract_stream_clip` simply clamps to whatever real data exists) and avoids
ever missing genuine leading samples because the hint slightly over-estimated the
true first timestamp."""
OUTER_MARGIN_S = 900.0
"""Seconds the search window extends PAST the latest burst file's own
filename-hint start -- matches `rowii.io.dataset._GAP_THRESHOLD` (15 min), the
same "how long a within-run gap can plausibly be" bound that module already uses
to decide whether two burst files belong to the same run at all. Comfortably
covers that last file's own ~12 min tail plus a full gap-threshold's slack."""

PROBE_COUNT = 5
PROBE_S = 45.0
PEAK_TARGET_DBFS = -3.0
"""Target peak level (dBFS) for the GAIN estimated from probes -- 2 dB more
headroom than `make_demo_assets.TARGET_DBFS` (-1.0), because that figure
peak-normalizes an EXHAUSTIVELY-read ~10 s clip while this one scales an entire
~4.4 h stream from a `PROBE_COUNT`-point sample (module docstring)."""

AAC_BITRATE_BPS = 40_000
"""~40 kbps mono -- inside the task's requested 32-48 kbps band; for a ~4.4 h
session this lands the compressed file in the "roughly 50-80 MB" range the task
itself anticipates."""


# ---------------------------------------------------------------------------
# Pure: session bounds, block-window planning, probe-instant spacing
# ---------------------------------------------------------------------------


def session_bounds(
    file_start_hints: Sequence[datetime],
    *,
    start_margin_s: float = START_MARGIN_S,
    outer_margin_s: float = OUTER_MARGIN_S,
) -> tuple[datetime, datetime]:
    """`(search_start_utc, search_end_utc)` -- the outer UTC span this module
    searches for real audio: from `start_margin_s` before the earliest burst
    file's own filename-hint start to `outer_margin_s` after the latest one's.
    Shared by `plan_extraction_blocks` (the block-window grid) and `main` (the
    gain-probing span) so both cover EXACTLY the same span -- one derivation, not
    two that could quietly disagree.

    Raises:
        ValueError: if *file_start_hints* is empty.
    """
    if not file_start_hints:
        raise ValueError("file_start_hints must be non-empty")
    return (
        min(file_start_hints) - timedelta(seconds=start_margin_s),
        max(file_start_hints) + timedelta(seconds=outer_margin_s),
    )


def plan_extraction_blocks(
    file_start_hints: Sequence[datetime],
    *,
    block_s: float = AUDIO_BLOCK_S,
    start_margin_s: float = START_MARGIN_S,
    outer_margin_s: float = OUTER_MARGIN_S,
) -> list[tuple[datetime, datetime]]:
    """The full, FIXED sequence of `(block_start, block_end)` windows to attempt
    for one stream, stepping by `block_s` across `session_bounds`'s own span. A
    PLAN only: whether a given window actually yields any samples is a runtime
    fact decided by `extract_stream_clip` (a window landing in a genuine
    within-run gap, or past the true end of real data, simply yields nothing --
    `extract_full_stream` skips it rather than treating a single empty block as
    proof the whole rest of the session is exhausted), so this never needs to
    know about real per-sample timestamps, only the cheap filename-hint metadata
    already available at discovery time.

    Raises:
        ValueError: if *file_start_hints* is empty (via `session_bounds`).
    """
    start, end_bound = session_bounds(
        file_start_hints, start_margin_s=start_margin_s, outer_margin_s=outer_margin_s
    )
    step = timedelta(seconds=block_s)
    windows: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end_bound:
        windows.append((cursor, cursor + step))
        cursor += step
    return windows


def probe_instants(
    session_start_utc: datetime, session_end_utc: datetime, n: int = PROBE_COUNT
) -> list[datetime]:
    """*n* evenly spaced UTC instants across `[session_start_utc,
    session_end_utc]` (both ends included), for peak-level probing spread across
    the whole session rather than just its opening minutes -- standstill,
    transition, and loaded operation can have genuinely different acoustic
    levels (module docstring).

    Raises:
        ValueError: if *n* < 2, or `session_end_utc <= session_start_utc`.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if session_end_utc <= session_start_utc:
        raise ValueError("session_end_utc must be after session_start_utc")
    span_s = (session_end_utc - session_start_utc).total_seconds()
    return [session_start_utc + timedelta(seconds=span_s * i / (n - 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# Pure: the replay-playhead <-> audio-offset mapping `assets/live.js` mirrors,
# and the transport-speed <-> playbackRate/mute mapping.
# ---------------------------------------------------------------------------


def audio_offset_s(playhead_s: float, t0_utc: str, audio_start_utc: str) -> float:
    """Seconds into an extracted audio stream corresponding to replay playhead
    *playhead_s* (seconds since the replay's own `t0_utc`) -- the authoritative
    reference for the mapping `assets/live.js` applies client-side to set
    `audio.currentTime`: `(t0_utc + playhead_s) - audio_start_utc`, in seconds.
    Can come out negative (the requested instant is before this stream's own
    extracted audio begins) or exceed the stream's own duration (after it ends)
    -- the caller decides what to do with an out-of-range result (`live.js`
    clamps and, at the extremes, pauses/mutes; `tests/test_build_live_audio.py`'s
    coverage checks assert the in-bounds case holds for the real payload), this
    function only computes the mapping.
    """
    t0 = datetime.fromisoformat(t0_utc)
    audio_start = datetime.fromisoformat(audio_start_utc)
    return playhead_s + (t0 - audio_start).total_seconds()


MUTE_AT_OR_ABOVE_SPEED = 16.0
"""`HTMLMediaElement.playbackRate` is reliable across evergreen browsers up to
~4x; the replay transport's 16x step is explicitly out of that range (task
instruction: "browsers do not do 16x audio") -- muted rather than played at a
wrong/choppy rate."""


def audio_playback_for_speed(speed: float) -> tuple[float, bool]:
    """`(playback_rate, muted)` for the replay transport's *speed* multiplier
    (the 1x/4x/16x buttons) -- passthrough below `MUTE_AT_OR_ABOVE_SPEED`, muted
    (with `playback_rate` pinned to 1.0 -- moot once muted, but still a valid
    positive rate) at or above it. Mirrored by hand in `assets/live.js` (no
    shared runtime between Python and the browser); this is the pinned reference
    `tests/test_build_live_audio.py` checks against.

    Raises:
        ValueError: if *speed* <= 0.
    """
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if speed >= MUTE_AT_OR_ABOVE_SPEED:
        return 1.0, True
    return speed, False


# ---------------------------------------------------------------------------
# IO-touching: gain estimation, full-stream extraction, AAC encode
# ---------------------------------------------------------------------------


def _native_peak(
    files: Sequence[BurstFile],
    offset_ns: int,
    channel_index: int,
    instant: datetime,
    probe_s: float,
) -> float:
    """Peak absolute value (native units, un-resampled) of a *probe_s*-long
    window starting at *instant* -- `0.0` if the window has no overlap with real
    data at all (a probe instant landing exactly in a gap; `estimate_gain`
    tolerates this by simply not counting that probe)."""
    try:
        clip = ak.extract_stream_clip(
            files, offset_ns, channel_index, instant, instant + timedelta(seconds=probe_s)
        )
    except ValueError:
        return 0.0
    if clip.samples.size == 0:
        return 0.0
    return float(np.max(np.abs(clip.samples)))


def estimate_gain(
    files: Sequence[BurstFile],
    offset_ns: int,
    channel_index: int,
    session_start_utc: datetime,
    session_end_utc: datetime,
    *,
    target_dbfs: float = PEAK_TARGET_DBFS,
    probe_count: int = PROBE_COUNT,
    probe_s: float = PROBE_S,
) -> tuple[float, float]:
    """`(gain, probed_peak)` -- *gain* scales native-unit samples so the largest
    PROBED peak lands at *target_dbfs*; *probed_peak* is that raw peak (native
    units, Pa for these mic streams), returned only for the build log. `gain =
    1.0` if every probe was silent/empty -- nothing to scale by, and a silent
    stream cannot clip regardless of gain.
    """
    instants = probe_instants(session_start_utc, session_end_utc, probe_count)
    peak = max(_native_peak(files, offset_ns, channel_index, t, probe_s) for t in instants)
    if peak <= 0.0:
        return 1.0, 0.0
    target_linear = 10.0 ** (target_dbfs / 20.0)
    return target_linear / peak, peak


@dataclass(frozen=True)
class ExtractionResult:
    """Everything `main` needs to log and write into the sidecar JSON for one
    stream's extraction."""

    start_utc: datetime
    duration_s: float
    n_samples: int
    peak_dbfs: float | None
    """Actual (post-gain) peak reached anywhere in the stream, in dBFS --
    `None` only if the entire stream came out silent (in which case
    `estimate_gain` already returned `gain=1.0`, so this would then also read
    `None` for a genuinely all-zero recording)."""
    clipped_sample_count: int


def extract_full_stream(
    files: Sequence[BurstFile],
    offset_ns: int,
    channel_index: int,
    gain: float,
    out_wav_path: Path,
    *,
    block_s: float = AUDIO_BLOCK_S,
    max_blocks: int | None = None,
) -> ExtractionResult:
    """Extract, resample (native rate -> `TARGET_SAMPLE_RATE_HZ`), scale by
    *gain*, and stream-write the ENTIRE stream to *out_wav_path* as 16-bit PCM
    mono WAV, one `plan_extraction_blocks` window at a time (`wave.Wave_write.
    writeframes` -- never the whole session resident as one array). A block that
    raises `ValueError` (no overlap -- past the true end of data, or inside a
    within-run gap) is logged and skipped, NOT treated as "the session is over":
    only `plan_extraction_blocks`'s own fixed, bounded grid decides when the
    scan stops.

    Args:
        max_blocks: if set, stop after this many BLOCKS THAT YIELDED SAMPLES
            (not attempted windows) -- a debug/smoke-test knob only, unused in
            a normal full build.

    Raises:
        ValueError: if every block was empty (nothing to write at all -- almost
            certainly a wrong *files*/*offset_ns*, not a genuine silent day).
    """
    hints = [f.start_utc_hint for f in files]
    windows = plan_extraction_blocks(hints, block_s=block_s)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    start_utc: datetime | None = None
    end_utc: datetime | None = None
    n_samples = 0
    peak_actual = 0.0
    clipped = 0
    int16_max = float(np.iinfo(np.int16).max)

    with wave.open(str(out_wav_path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(TARGET_SAMPLE_RATE_HZ)
        n_yielded = 0
        for i, (w_start, w_end) in enumerate(windows):
            try:
                clip = ak.extract_stream_clip(files, offset_ns, channel_index, w_start, w_end)
            except ValueError as exc:
                logger.debug(
                    "build_live_audio: %s block %d/%d [%s, %s) empty (%s), skipping",
                    out_wav_path.stem, i + 1, len(windows), w_start, w_end, exc,
                )
                continue
            resampled = mda._resample_to_target(clip.samples, clip.rate_hz)
            if resampled.size == 0:
                continue
            scaled = resampled * gain
            peak_actual = max(peak_actual, float(np.max(np.abs(scaled))))
            clipped += int(np.count_nonzero(np.abs(scaled) > 1.0))
            pcm16 = np.clip(scaled * int16_max, -int16_max - 1, int16_max).astype(np.int16)
            wav_out.writeframes(pcm16.tobytes())
            n_samples += int(pcm16.size)
            if start_utc is None:
                start_utc = clip.covered_start_utc
            end_utc = clip.covered_end_utc
            n_yielded += 1
            logger.info(
                "build_live_audio: %s block %d/%d [%s, %s) -> %d samples",
                out_wav_path.stem, i + 1, len(windows),
                clip.covered_start_utc, clip.covered_end_utc, pcm16.size,
            )
            if max_blocks is not None and n_yielded >= max_blocks:
                logger.info(
                    "build_live_audio: %s stopping early (--max-blocks=%d)",
                    out_wav_path.stem, max_blocks,
                )
                break

    if start_utc is None or end_utc is None or n_samples == 0:
        raise ValueError(f"no audio extracted for {out_wav_path.name} -- every block was empty")

    duration_s = n_samples / TARGET_SAMPLE_RATE_HZ
    peak_dbfs = 20.0 * math.log10(peak_actual) if peak_actual > 0.0 else None
    return ExtractionResult(
        start_utc=start_utc,
        duration_s=duration_s,
        n_samples=n_samples,
        peak_dbfs=peak_dbfs,
        clipped_sample_count=clipped,
    )


def encode_to_m4a(wav_path: Path, out_path: Path, *, bitrate_bps: int = AAC_BITRATE_BPS) -> Path:
    """Shell out to macOS's built-in `afconvert` (this repo has no `ffmpeg`
    dependency): mono AAC-in-.m4a, CONSTANT bit rate (`-s 0`) so the OUTPUT SIZE
    stays predictable regardless of how acoustically "busy" a given stream is --
    VBR/VBR_constrained would let a quieter/simpler stream come out smaller and a
    busier one larger, undermining the "roughly the same size for both mic
    streams" expectation the task itself states.

    Raises:
        RuntimeError: if `afconvert` exits non-zero or does not produce
            *out_path*.
    """
    if out_path.exists():
        out_path.unlink()
    cmd = [
        "afconvert",
        "-f", "m4af",
        "-d", f"aac@{TARGET_SAMPLE_RATE_HZ}",
        "-c", "1",
        "-b", str(bitrate_bps),
        "-s", "0",  # 0 = CBR (see docstring)
        str(wav_path),
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"afconvert failed (exit {proc.returncode}) for {wav_path} -> {out_path}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=RUN, help=f"Run name to extract (default: {RUN!r}).")
    parser.add_argument(
        "--streams", nargs="+", choices=sorted(STREAM_BY_KEY), default=list(STREAM_BY_KEY),
        help="Which stream(s) to extract (default: both).",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--scratch-dir", type=Path, default=None,
        help="Where temporary WAVs are written before the AAC encode "
        "(default: <out-dir>/.scratch, removed after use unless --keep-wav).",
    )
    parser.add_argument(
        "--keep-wav", action="store_true",
        help="Keep the intermediate WAV file(s) instead of deleting them after afconvert "
        "(debugging).",
    )
    parser.add_argument("--bitrate-bps", type=int, default=AAC_BITRATE_BPS)
    parser.add_argument(
        "--max-blocks", type=int, default=None,
        help="Stop each stream after this many non-empty ~5 min blocks (debug/smoke-test only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    cfg = load_config()
    index = discover(cfg.data_root)
    run = mda._get_run(index, args.run)
    offset_ns = run_utc_offset_ns(run)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir: Path = args.scratch_dir or (out_dir / ".scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    streams_meta: dict[str, Any] = {}
    for key in args.streams:
        stream_name = STREAM_BY_KEY[key]
        files = run.files.get(stream_name, [])
        if not files:
            raise SystemExit(f"build_live_audio: run {args.run!r} has no {stream_name!r} files")
        hints = [f.start_utc_hint for f in files]
        session_start, session_end = session_bounds(hints)

        logger.info(
            "build_live_audio: %s (%s) -- estimating gain from %d probes across %s .. %s",
            key, stream_name, PROBE_COUNT, session_start.isoformat(), session_end.isoformat(),
        )
        gain, probed_peak = estimate_gain(
            files, offset_ns, mda.MONO_CHANNEL_INDEX, session_start, session_end
        )
        logger.info(
            "build_live_audio: %s gain=%.4f (probed native-unit peak %.6g)", key, gain, probed_peak
        )

        wav_path = scratch_dir / f"{args.run}_{key}.wav"
        result = extract_full_stream(
            files, offset_ns, mda.MONO_CHANNEL_INDEX, gain, wav_path,
            max_blocks=args.max_blocks,
        )
        peak_str = f"{result.peak_dbfs:.2f} dBFS" if result.peak_dbfs is not None else "silent"
        logger.info(
            "build_live_audio: %s extracted %.1f min starting %s -- peak %s, %d clipped sample(s)",
            key, result.duration_s / 60.0, result.start_utc.isoformat(),
            peak_str, result.clipped_sample_count,
        )

        m4a_path = out_dir / f"{args.run}_{key}.m4a"
        encode_to_m4a(wav_path, m4a_path, bitrate_bps=args.bitrate_bps)
        size_bytes = m4a_path.stat().st_size
        logger.info(
            "build_live_audio: %s -> %s (%.2f MB)", key, m4a_path, size_bytes / 1_000_000.0
        )

        if not args.keep_wav:
            wav_path.unlink()

        streams_meta[key] = {
            "stream": stream_name,
            "label": STREAM_LABEL[key],
            "file": f"assets/live/{m4a_path.name}",
            "start_utc": result.start_utc.isoformat(),
            "duration_s": round(result.duration_s, 3),
            "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
            "bytes": size_bytes,
        }

    if not args.keep_wav:
        # not empty (a --streams subset run alongside kept WAVs) -- fine, leave it
        with contextlib.suppress(OSError):
            scratch_dir.rmdir()

    meta_path = out_dir / f"{args.run}_audio_meta.json"
    meta = {"run": args.run, "generated_at": datetime.now(UTC).isoformat(), "streams": streams_meta}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("build_live_audio: wrote %s", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
