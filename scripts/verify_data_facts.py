"""D4 data verifications (Package-8, spec D4 + A1.6). Three cheap scripted probes on
OUR own files, each independently attributed where it echoes the partner's data
work (Rodrigues & Zhang 2026): (1) generator-mic level anomaly at plate-strike
minutes -- CHANNEL-ANONYMOUS (no azimuth->channel map exists in-repo; A1.6); (2)
RAWTurbineVib__3 ch0 per-file DATA-VARIANCE liveness across days (std<1e-9, the
VibFeaturizer dead-channel criterion, A1.6); (3) SCADA timebase probe: locate the
080726 changeover in Betriebsdaten rpm/power and compare it against the audio-UTC
state timeline (13:05:28 UTC reference). No partner number is read by this script.

Deliberately independent of `rowii.pipeline`/`rowii.signals.features`: these are
small, standalone sanity probes over raw files (mirrors `scripts/verify_parameters.
py`'s own bypass of the full pipeline for the same reason), not a rehearsal of the
detection pipeline -- see each subcommand's own helper docstring for its local,
duplicated grid-building logic (the no-sibling-script-import rule: a script must
never import a SIBLING script's internals; `src/rowii/` modules are imported
normally).
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.config import load_config  # noqa: E402
from rowii.io.dataset import (  # noqa: E402
    RecordingIndex,
    Run,
    betriebsdaten_utc_offset_ns,
    discover,
    run_utc_offset_ns,
)
from rowii.io.gantner import read_gantner, read_header  # noqa: E402
from rowii.scada.labels import load_scada_window_means  # noqa: E402
from rowii.signals.windows import WindowGrid, window_slices  # noqa: E402

_DEAD_STD = 1e-9
_REFERENCE_UTC_DEFAULT = "2026-07-08T13:05:28+00:00"
_DAY_ROOT_DEFAULT = "illwerke-080726"
_VIB_STREAM_DEFAULT = "RAWTurbineVib__3"
_GEN_MIC_STREAM_DEFAULT = "RAWGeneratorMic__0"
_DEFAULT_WINDOW_S = 1.0
_DEFAULT_STRIKE_KIND = "plate-gen_90"
_LEVEL_FLOOR = 1e-12
"""Same floor as `rowii.signals.features`' `_log10_floor` (VERIFIED, plan Global
Constraints) -- this script computes its own simple log10(rms) level rather than
importing that private helper, but anchors the floor to the same verified value so
a level printed here is directly comparable in scale to the pipeline's own level
features."""


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested, tests/test_verify_data_facts.py) -- the load-bearing
# logic of every subcommand below.
# ---------------------------------------------------------------------------


def block_is_dead(block: np.ndarray) -> bool:
    """`VibFeaturizer`'s own dead-channel criterion (`rowii.signals.features`,
    `_DEAD_CHANNEL_STD = 1e-9`): std of *block*, cast to float64 first regardless of
    the input dtype (a channel that is exactly constant in float32 is exactly
    constant in float64 too -- widening float32->float64 is exact per-element)."""
    return bool(float(np.asarray(block, dtype=np.float64).std()) < _DEAD_STD)


def locate_changeover(ts_ns: np.ndarray, values: np.ndarray) -> int:
    """Index `i >= 1` of the largest `|values[i] - values[i-1]|` first-difference
    over finite neighbours (a non-finite neighbour can never win the argmax). Ties
    resolve to the leftmost (smallest-index) maximal jump -- `np.argmax`'s own
    first-occurrence tie-break. *ts_ns* is accepted (and used by callers to look up
    the changeover's real timestamp) but is not itself part of the jump math."""
    v = np.asarray(values, dtype=np.float64)
    if v.shape[0] < 2:
        raise ValueError("locate_changeover needs at least 2 samples")
    diffs = np.abs(np.diff(v))
    diffs[~np.isfinite(diffs)] = -np.inf
    return int(np.argmax(diffs)) + 1


def channel_level_profile(levels: np.ndarray, strike_mask: np.ndarray) -> np.ndarray:
    """`(C,)` per-channel median level over the *strike_mask* rows of a `(W, C)`
    level matrix -- the ring's per-channel signature at the strike minutes."""
    rows = np.asarray(levels, dtype=np.float64)[np.asarray(strike_mask, dtype=bool)]
    profile: np.ndarray = np.median(rows, axis=0).astype(np.float64)
    return profile


def outlier_channel(profile: np.ndarray) -> int:
    """`argmax(|profile - median(profile)|)`: the ring-outlier channel INDEX only --
    channel-anonymous, no azimuth claim (A1.6: no azimuth->channel map is in-repo)."""
    p = np.asarray(profile, dtype=np.float64)
    return int(np.argmax(np.abs(p - np.median(p))))


# ---------------------------------------------------------------------------
# Shared small helpers (duplicated from sibling scripts' own private helpers per
# the no-sibling-script-import rule; each docstring names its mirror).
# ---------------------------------------------------------------------------


def _shift_ts_ns(ts_ns: np.ndarray, offset_ns: int) -> np.ndarray:
    """`ts_ns + offset_ns`, staying in uint64 throughout -- mirrors
    `rowii.scada.labels._shift_ts_ns` (private there, duplicated here): mixing
    int64 and uint64 numpy arrays silently upcasts BOTH to float64, losing
    precision far below these ~1e18 ns timestamps."""
    if offset_ns >= 0:
        return ts_ns + np.uint64(offset_ns)
    return ts_ns - np.uint64(-offset_ns)


def _ns_to_utc(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, tz=UTC)


def _day_betriebsdaten(index: RecordingIndex, day_root_name: str) -> list[Path]:
    """Betriebsdaten files for the day root named *day_root_name* (e.g.
    `"illwerke-080726"`), or `[]` if no such day root was discovered."""
    for root, files in index.betriebsdaten_by_day.items():
        if root.name == day_root_name:
            return list(files)
    return []


def _day_window_grid(betriebsdaten: list[Path], window_s: float) -> WindowGrid:
    """Whole-day grid spanning one Betriebsdaten file set's own coverage (first
    file's start to last file's end), offset onto true UTC by the set's OWN
    derived offset (`betriebsdaten_utc_offset_ns`) -- mirrors `scripts/
    verify_parameters.py`'s `_build_whole_day_grid` (private there, duplicated
    here): a standalone grid builder bypassing `rowii.pipeline.build_run_grid`
    entirely, since no audio run is involved."""
    sorted_files = sorted(betriebsdaten)
    first_h = read_header(sorted_files[0])
    last_h = read_header(sorted_files[-1])
    offset_ns = betriebsdaten_utc_offset_ns(betriebsdaten)
    first_start_ns = first_h.t0_ns + offset_ns
    last_end_ns = last_h.t0_ns + offset_ns + round(last_h.n_frames / last_h.sample_rate_hz * 1e9)
    window_ns = round(window_s * 1e9)
    n_windows = (last_end_ns - first_start_ns) // window_ns
    return WindowGrid(t0_ns=first_start_ns, window_ns=window_ns, n_windows=n_windows)


def _run_window_grid_and_levels(
    run: Run, stream: str, window_s: float
) -> tuple[WindowGrid, np.ndarray]:
    """Pool *stream*'s burst files of *run* into one true-UTC window grid plus a
    `(n_windows, n_channels)` per-window per-channel level matrix
    (`log10(max(rms, floor))`, this script's own simple level -- not the pipeline's
    `_log10_floor` feature, but anchored to the same floor, module docstring).

    Frame timestamps are placed on the true-UTC axis via the run's OWN derived
    DAQ-clock-quirk offset (`rowii.io.dataset.run_utc_offset_ns`) -- required for
    any comparison against a ground-truth CSV, which is written in true UTC.
    """
    files = sorted(run.files.get(stream, []), key=lambda f: f.start_utc_hint)
    if not files:
        raise ValueError(f"run {run.name!r} has no {stream!r} files")

    offset_ns = run_utc_offset_ns(run)
    ts_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    for bf in files:
        gf = read_gantner(bf.path)
        ts_parts.append(_shift_ts_ns(gf.timestamps_ns, offset_ns))
        data_parts.append(gf.data)
    ts_ns = np.concatenate(ts_parts)
    data = np.concatenate(data_parts, axis=0)

    window_ns = round(window_s * 1e9)
    n_windows = int((int(ts_ns[-1]) - int(ts_ns[0])) // window_ns)
    if n_windows == 0:
        raise ValueError(f"run {run.name!r} stream {stream!r} spans less than one window")
    grid = WindowGrid(t0_ns=int(ts_ns[0]), window_ns=window_ns, n_windows=n_windows)

    slices = window_slices(ts_ns, grid)
    n_ch = data.shape[1]
    levels = np.full((n_windows, n_ch), np.nan, dtype=np.float64)
    for i, sl in enumerate(slices):
        if sl.stop > sl.start:
            block = data[sl].astype(np.float64)
            rms = np.sqrt(np.mean(np.square(block), axis=0))
            levels[i] = np.log10(np.maximum(rms, _LEVEL_FLOOR))
    return grid, levels


def _load_strike_intervals(path: Path, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a ground-truth strike CSV (`start_utc,end_utc,kind`; `#` comment lines
    skipped -- the `docs/groundtruth` contract) into `(starts_ns, ends_ns)` int64
    UTC-epoch arrays, filtered to rows whose `kind` column equals *kind*. Mirrors
    `scripts/monitor.py`'s `_load_exclusion_intervals` (private there, duplicated
    here per the no-sibling-script-import rule), extended with the `kind` filter.

    Raises:
        ValueError: missing columns, no rows of *kind*, or naive (non-tz-aware)
            timestamps.
    """
    frame = pd.read_csv(path, comment="#")
    missing = [c for c in ("start_utc", "end_utc", "kind") if c not in frame.columns]
    if missing:
        raise ValueError(
            f"strike events file {path} is missing required column(s) {missing} -- "
            f"got columns {list(frame.columns)}"
        )
    frame = frame[frame["kind"] == kind]
    if frame.empty:
        raise ValueError(f"strike events file {path} has no rows with kind={kind!r}")
    bounds: list[np.ndarray] = []
    for col in ("start_utc", "end_utc"):
        parsed = pd.to_datetime(frame[col])
        if getattr(parsed.dt, "tz", None) is None:
            raise ValueError(
                f"strike events file {path} column {col!r} has naive timestamps -- "
                f"tz-aware ISO-8601 required (the eval_events/groundtruth contract)"
            )
        bounds.append(parsed.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy())
    starts_ns, ends_ns = bounds
    return starts_ns, ends_ns


def _strike_mask(
    window_starts_ns: np.ndarray, window_ns: int, starts_ns: np.ndarray, ends_ns: np.ndarray
) -> np.ndarray:
    """True for every window whose CENTER falls inside `[starts_ns[k], ends_ns[k])`
    for any *k* (left-closed/right-open, matching `eval_events.py`'s interval
    convention, module docstring note in the groundtruth CSVs)."""
    centers_ns = window_starts_ns.astype(np.int64) + window_ns // 2
    mask = np.zeros(window_starts_ns.shape[0], dtype=bool)
    for start_ns, end_ns in zip(starts_ns, ends_ns, strict=True):
        mask |= (centers_ns >= start_ns) & (centers_ns < end_ns)
    return mask


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _vib_ch0_liveness_rows(
    index: RecordingIndex, stream: str, all_files: bool
) -> list[dict[str, object]]:
    """Per-(day root, run, sampled file) ch0 liveness rows. Default samples only the
    EARLIEST file of each run (D4.2's own objective is "which days carry live ch0",
    a per-run/per-day question -- burst files within one run share the same physical
    wiring, so checking every file of a multi-hundred-MB stream is redundant for that
    question); `--all-files` widens the sample to every file for full per-file
    thoroughness (D4.2's literal "per-file" wording)."""
    rows: list[dict[str, object]] = []
    for run in index.runs:
        files = sorted(run.files.get(stream, []), key=lambda f: f.start_utc_hint)
        if not files:
            continue
        targets = files if all_files else files[:1]
        for bf in targets:
            gf = read_gantner(bf.path)
            if gf.data.shape[1] == 0:
                continue
            ch0 = gf.data[:, 0]
            std = float(np.asarray(ch0, dtype=np.float64).std())
            rows.append(
                {
                    "day_root": run.day_root.name,
                    "run": run.name,
                    "file": bf.path.name,
                    "std": std,
                    "live": not block_is_dead(ch0),
                }
            )
    return rows


def _run_vib_ch0_liveness(args: argparse.Namespace) -> int:
    cfg = load_config()
    index = discover(cfg.data_root)
    rows = _vib_ch0_liveness_rows(index, str(args.stream), bool(args.all_files))
    if not rows:
        print(
            f"vib-ch0-liveness: no {args.stream!r} files discovered under {cfg.data_root}",
            file=sys.stderr,
        )
        return 1

    print(f"{'day root':<20} {'run':<28} {'file':<48} {'std':>12}  status")
    by_day: dict[str, bool] = {}
    for row in rows:
        print(
            f"{row['day_root']!s:<20} {row['run']!s:<28} {row['file']!s:<48} "
            f"{row['std']:>12.3e}  {'LIVE' if row['live'] else 'DEAD'}"
        )
        day_root = str(row["day_root"])
        by_day[day_root] = by_day.get(day_root, False) or bool(row["live"])

    print()
    print("Per-day-root verdict (ch0 live in >= 1 sampled file), era timeline order:")
    for day_root in sorted(by_day):
        print(f"  {day_root}: {'LIVE' if by_day[day_root] else 'DEAD'}")
    return 0


def _run_scada_timebase(args: argparse.Namespace) -> int:
    cfg = load_config()
    index = discover(cfg.data_root)
    betriebsdaten = _day_betriebsdaten(index, str(args.day_root))
    if not betriebsdaten:
        print(
            f"scada-timebase: no Betriebsdaten found for day root {args.day_root!r} "
            f"under {cfg.data_root}",
            file=sys.stderr,
        )
        return 1

    try:
        reference = datetime.fromisoformat(str(args.reference_utc))
    except ValueError as exc:
        print(
            f"scada-timebase: cannot parse --reference-utc {args.reference_utc!r}: {exc}",
            file=sys.stderr,
        )
        return 2
    if reference.tzinfo is None:
        print(
            f"scada-timebase: --reference-utc must be tz-aware, got naive "
            f"{args.reference_utc!r}",
            file=sys.stderr,
        )
        return 2

    betriebsdaten = sorted(betriebsdaten)
    grid = _day_window_grid(betriebsdaten, cfg.window.window_s)
    scada = load_scada_window_means(betriebsdaten, grid)
    window_starts_ns = grid.edges_ns()[:-1]

    speed = scada["speed"].to_numpy(dtype=np.float64)
    power = scada["power"].to_numpy(dtype=np.float64)

    speed_idx = locate_changeover(window_starts_ns, speed)
    power_idx = locate_changeover(window_starts_ns, power)
    speed_ts = _ns_to_utc(int(window_starts_ns[speed_idx]))
    power_ts = _ns_to_utc(int(window_starts_ns[power_idx]))

    print(f"scada-timebase: day root {args.day_root} ({len(betriebsdaten)} Betriebsdaten file(s))")
    print(f"  speed (rpm) changeover:  {speed_ts.isoformat()}  (window {speed_idx})")
    print(f"  power (MW)  changeover:  {power_ts.isoformat()}  (window {power_idx})")
    print(f"  reference (audio-UTC state timeline): {reference.isoformat()}")
    print(f"  delta speed - reference: {(speed_ts - reference).total_seconds():+.3f} s")
    print(f"  delta power - reference: {(power_ts - reference).total_seconds():+.3f} s")
    return 0


def _run_gen_mic_profile(args: argparse.Namespace) -> int:
    cfg = load_config()
    index = discover(cfg.data_root)
    by_name = {r.name: r for r in index.runs}
    if str(args.run) not in by_name:
        available = ", ".join(sorted(by_name)) or "(none discovered)"
        print(
            f"gen-mic-profile: unknown run {args.run!r}; available runs: {available}",
            file=sys.stderr,
        )
        return 2
    run = by_name[str(args.run)]

    try:
        starts_ns, ends_ns = _load_strike_intervals(Path(args.events), str(args.kind))
    except ValueError as exc:
        print(f"gen-mic-profile: {exc}", file=sys.stderr)
        return 2

    try:
        grid, levels = _run_window_grid_and_levels(run, str(args.stream), float(args.window_s))
    except ValueError as exc:
        print(f"gen-mic-profile: {exc}", file=sys.stderr)
        return 2

    window_starts_ns = grid.edges_ns()[:-1]
    strike_mask = _strike_mask(window_starts_ns, grid.window_ns, starts_ns, ends_ns)
    if not bool(strike_mask.any()):
        print(
            f"gen-mic-profile: zero windows of run {args.run!r} fall inside "
            f"kind={args.kind!r} strike interval(s) from {args.events}",
            file=sys.stderr,
        )
        return 1

    profile = channel_level_profile(levels, strike_mask)
    outlier = outlier_channel(profile)

    print(
        f"gen-mic-profile: run={args.run} stream={args.stream} kind={args.kind!r} "
        f"({int(strike_mask.sum())} strike window(s))"
    )
    for ch, level in enumerate(profile):
        marker = "  <-- outlier" if ch == outlier else ""
        print(f"  ch{ch}: {level:+.3f} (log10 rms){marker}")
    print(f"channel-anonymous outlier index: {outlier} (no azimuth mapping asserted -- A1.6)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "D4 data verifications (Package-8, spec D4 + A1.6): three small scripted "
            "probes on our own recorded files -- vib-ch0-liveness, scada-timebase, "
            "gen-mic-profile (module docstring)."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_vib = sub.add_parser(
        "vib-ch0-liveness",
        help="RAWTurbineVib__3 ch0 per-file data-variance liveness scan across days.",
    )
    p_vib.add_argument(
        "--stream", default=_VIB_STREAM_DEFAULT,
        help=f"Burst stream name to scan (default {_VIB_STREAM_DEFAULT!r}).",
    )
    p_vib.add_argument(
        "--all-files", action="store_true",
        help="Check every file of every run instead of only the earliest per run.",
    )

    p_scada = sub.add_parser(
        "scada-timebase",
        help="Locate the day's changeover in Betriebsdaten rpm/power vs a reference UTC.",
    )
    p_scada.add_argument(
        "--day-root", default=_DAY_ROOT_DEFAULT,
        help=f"Day root name to probe (default {_DAY_ROOT_DEFAULT!r}).",
    )
    p_scada.add_argument(
        "--reference-utc", default=_REFERENCE_UTC_DEFAULT,
        help=f"tz-aware ISO-8601 reference timestamp, e.g. the audio-UTC state "
             f"timeline's own changeover (default {_REFERENCE_UTC_DEFAULT!r}).",
    )

    p_mic = sub.add_parser(
        "gen-mic-profile",
        help="Channel-anonymous generator-mic level profile at plate-strike minutes.",
    )
    p_mic.add_argument("--run", required=True, help="Discovered run name to probe.")
    p_mic.add_argument(
        "--events", type=Path, required=True,
        help="Ground-truth strike CSV (start_utc,end_utc,kind; '#' comments; "
             "docs/groundtruth contract).",
    )
    p_mic.add_argument(
        "--kind", default=_DEFAULT_STRIKE_KIND,
        help=f"Strike 'kind' value to select (default {_DEFAULT_STRIKE_KIND!r}).",
    )
    p_mic.add_argument(
        "--stream", default=_GEN_MIC_STREAM_DEFAULT,
        help=f"Burst stream name to profile (default {_GEN_MIC_STREAM_DEFAULT!r}).",
    )
    p_mic.add_argument(
        "--window-s", type=float, default=_DEFAULT_WINDOW_S,
        help=f"Detection window duration in seconds (default {_DEFAULT_WINDOW_S:g}).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "vib-ch0-liveness":
        return _run_vib_ch0_liveness(args)
    if args.command == "scada-timebase":
        return _run_scada_timebase(args)
    if args.command == "gen-mic-profile":
        return _run_gen_mic_profile(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
