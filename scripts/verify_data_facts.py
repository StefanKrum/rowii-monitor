"""Data verification CLI. Three cheap scripted probes on
OUR own files, each independently attributed where it echoes the partner's data
work (Rodrigues & Zhang 2026): (1) generator-mic level anomaly at plate-strike
minutes -- CHANNEL-ANONYMOUS (no azimuth->channel map exists in-repo); (2)
RAWTurbineVib__3 ch0 per-file DATA-VARIANCE liveness across days (std<1e-9, the
VibFeaturizer dead-channel criterion); (3) SCADA timebase probe: locate the
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
from collections.abc import Iterable
from datetime import UTC, date, datetime
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
_SEARCH_RADIUS_MIN_DEFAULT = 30.0
"""Default `--search-radius-min` for `scada-timebase`'s reference-windowed changeover
search -- generous enough to comfortably contain DAQ-clock-quirk
and ramp-duration slack around the reference, while still excluding a same-channel
decoy step half an hour or more away (e.g. 080726's pump-start step vs. its
pump->phase-shifter changeover, ~52 min apart)."""
_TOP_K_STEPS = 3
"""Number of largest whole-day steps `scada-timebase` prints per channel alongside
its windowed changeover pick -- lets a human see the
alternatives the windowed argmax itself cannot show; not independently tunable via
any CLI flag."""
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


def locate_changeover(
    ts_ns: np.ndarray,
    values: np.ndarray,
    *,
    reference_index: int | None = None,
    search_radius: int | None = None,
) -> int:
    """Index `i >= 1` of the largest `|values[i] - values[i-1]|` first-difference
    over finite neighbours (a non-finite neighbour can never win the argmax). Ties
    resolve to the leftmost (smallest-index) maximal jump -- `np.argmax`'s own
    first-occurrence tie-break. *ts_ns* is accepted (and used by callers to look up
    the changeover's real timestamp) but is not itself part of the jump math.

    When *reference_index* and *search_radius* are both given, the argmax search is
    restricted to value-indices `i` with `abs(i - reference_index) <= search_radius`
    (whole-series search otherwise) -- needed because a day can carry two near-equal
    steps in the SAME channel (e.g. 080726 power: the pump-start step AND the
    pump->phase-shifter changeover itself), where the unrestricted global argmax can
    pick the wrong one and ties resolve leftmost. The two
    parameters must be given together or both omitted: there is no sensible reading
    of "centred at an unknown reference" or "reference with an unlimited radius", so
    a partial pair raises rather than silently guessing which was meant.

    Raises:
        ValueError: fewer than 2 samples; exactly one of *reference_index* /
            *search_radius* given; *reference_index* outside `[0, len(values))`;
            negative *search_radius*; no value-index falls within the requested
            window at all; or every first-difference under consideration
            (whole-series, or within the window) is non-finite (NaN/inf).
    """
    v = np.asarray(values, dtype=np.float64)
    if v.shape[0] < 2:
        raise ValueError("locate_changeover needs at least 2 samples")
    if (reference_index is None) != (search_radius is None):
        raise ValueError(
            "locate_changeover requires reference_index and search_radius together "
            "-- both None for a whole-series search, or both given for a windowed "
            f"search (got reference_index={reference_index!r}, "
            f"search_radius={search_radius!r})"
        )

    diffs = np.abs(np.diff(v))
    diffs[~np.isfinite(diffs)] = -np.inf

    if reference_index is not None and search_radius is not None:
        if not 0 <= reference_index < v.shape[0]:
            raise ValueError(
                f"reference_index={reference_index} out of range for "
                f"{v.shape[0]} sample(s)"
            )
        if search_radius < 0:
            raise ValueError(f"search_radius must be >= 0, got {search_radius}")
        value_indices = np.arange(1, v.shape[0])
        window_mask = np.abs(value_indices - reference_index) <= search_radius
        if not bool(window_mask.any()):
            raise ValueError(
                f"no candidate index within +/-{search_radius} of "
                f"reference_index={reference_index} ({v.shape[0]} sample(s) total)"
            )
        windowed = np.where(window_mask, diffs, -np.inf)
        if not bool(np.isfinite(windowed).any()):
            raise ValueError(
                f"no finite consecutive-window difference within "
                f"+/-{search_radius} of reference_index={reference_index} -- every "
                f"first-difference in that window is non-finite (NaN/inf)"
            )
        return int(np.argmax(windowed)) + 1

    if not bool(np.isfinite(diffs).any()):
        raise ValueError(
            "no finite consecutive-window difference -- every first-difference in "
            "the series is non-finite (NaN/inf)"
        )
    return int(np.argmax(diffs)) + 1


def top_k_steps(values: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Up to *k* value-indices `i >= 1` with the largest finite `|values[i] -
    values[i-1]|` first-difference, magnitude-descending (`np.argsort`'s stable sort
    keeps equal-magnitude ties in ascending-index order, mirroring
    `locate_changeover`'s own leftmost tie-break). Non-finite differences are
    dropped outright rather than sentinel-substituted. Gives a human the runner-up
    candidates `locate_changeover`'s single argmax search cannot show
    -- `scada-timebase` prints this alongside its windowed pick. Returns
    fewer than *k* pairs if fewer than *k* finite differences exist, `[]` if none
    do."""
    v = np.asarray(values, dtype=np.float64)
    diffs = np.abs(np.diff(v))
    finite = np.flatnonzero(np.isfinite(diffs))
    if finite.size == 0:
        return []
    order = finite[np.argsort(-diffs[finite], kind="stable")]
    top = order[: max(k, 0)]
    return [(int(i) + 1, float(diffs[i])) for i in top]


def channel_level_profile(levels: np.ndarray, strike_mask: np.ndarray) -> np.ndarray:
    """`(C,)` per-channel median level over the *strike_mask* rows of a `(W, C)`
    level matrix -- the ring's per-channel signature at the strike minutes.
    `np.nanmedian`, not `np.median`: a strike window with zero audio samples
    (`_run_window_grid_and_levels`'s all-NaN fill for an empty window) must not
    poison an otherwise-healthy channel's median across the whole ring. The
    caller (`_run_gen_mic_profile`) separately counts and warns
    about how many strike rows carried such a non-finite value.

    Raises:
        ValueError: some channel has no finite value across any strike row (an
            empty *strike_mask* selection included) -- `nanmedian` would otherwise
            silently emit a RuntimeWarning and return NaN for that channel.
    """
    rows = np.asarray(levels, dtype=np.float64)[np.asarray(strike_mask, dtype=bool)]
    all_nan_channels = np.all(np.isnan(rows), axis=0)
    if bool(all_nan_channels.any()):
        bad = np.flatnonzero(all_nan_channels).tolist()
        raise ValueError(
            f"channel_level_profile: channel(s) {bad} have no finite value across "
            f"any of the {rows.shape[0]} strike row(s)"
        )
    profile: np.ndarray = np.nanmedian(rows, axis=0).astype(np.float64)
    return profile


def outlier_channel(profile: np.ndarray) -> int:
    """`argmax(|profile - median(profile)|)`: the ring-outlier channel INDEX only --
    channel-anonymous, no azimuth claim (A1.6: no azimuth->channel map is in-repo)."""
    p = np.asarray(profile, dtype=np.float64)
    return int(np.argmax(np.abs(p - np.median(p))))


def sorted_day_roots(day_roots: Iterable[str]) -> list[str]:
    """*day_roots* (names like `"illwerke-DDMMYY"`, the day-root token convention
    `rowii.io.dataset._dayid_for` also uses) sorted CHRONOLOGICALLY by their
    embedded `DDMMYY` date, not alphabetically -- alphabetical order scrambles the
    era timeline (e.g. `"illwerke-010726"`, July 1st, would sort before
    `"illwerke-250526"`, May 25th)."""

    def _date(day_root: str) -> date:
        token = day_root.rsplit("-", 1)[-1]
        return datetime.strptime(token, "%d%m%y").date()

    return sorted(day_roots, key=_date)


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


def _utc_dt_to_ns(dt: datetime) -> int:
    """*dt* (tz-aware) as nanoseconds since the Unix epoch -- mirrors `rowii.io.
    dataset._hint_utc_ns` (private there, duplicated here per the no-sibling-
    script-import rule's own spirit: this script avoids reaching into a src
    module's private helper just as it avoids reaching into a sibling script's)."""
    return round(dt.timestamp() * 1e9)


def _reference_window_index(grid: WindowGrid, reference_ns: int) -> int:
    """*reference_ns*'s window index on *grid* (`floor((reference_ns - t0_ns) /
    window_ns)`), clamped into `[0, grid.n_windows - 1]` -- a reference outside the
    day's own coverage still yields the nearest in-range window rather than an
    out-of-bounds index, so `locate_changeover`'s reference-windowed search (T1
    review finding 1) always has a valid centre to search around."""
    raw_index = (reference_ns - grid.t0_ns) // grid.window_ns
    return int(min(max(raw_index, 0), grid.n_windows - 1))


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
    """True for every window whose window-center-in-[start,end) for any *k*
    (left-closed/right-open) -- equivalent to `eval_events.py`'s own start-based
    convention (module docstring: window START `t` inside `[start_ns, end_ns)`) ON
    MINUTE-ALIGNED INTERVALS, where every interval boundary already coincides with
    a window edge on this grid, so a center check and a start check can never
    disagree. This is NOT a claim of exact equivalence in general -- a boundary
    that falls strictly inside one window could flip the two conventions' verdict
    for that window. The groundtruth strike CSVs this script reads (module
    docstring) are whole-minute aligned, so that condition holds here."""
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
    thoroughness (D4.2's literal "per-file" wording). The earliest-file default is
    ASYMMETRIC: it can only produce a false DEAD verdict (never a false LIVE one)
    for a channel that only goes live partway through a run -- a live earliest file
    already proves the wiring was live at least once."""
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
    for day_root in sorted_day_roots(by_day):
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

    search_radius_min = float(args.search_radius_min)
    if not np.isfinite(search_radius_min) or search_radius_min < 0:
        print(
            f"scada-timebase: --search-radius-min must be a finite number >= 0, "
            f"got {args.search_radius_min!r}",
            file=sys.stderr,
        )
        return 2

    betriebsdaten = sorted(betriebsdaten)
    grid = _day_window_grid(betriebsdaten, cfg.window.window_s)
    scada = load_scada_window_means(betriebsdaten, grid)
    window_starts_ns = grid.edges_ns()[:-1]

    speed = scada["speed"].to_numpy(dtype=np.float64)
    power = scada["power"].to_numpy(dtype=np.float64)

    reference_index = _reference_window_index(grid, _utc_dt_to_ns(reference))
    window_s = grid.window_ns / 1e9
    search_radius = round(search_radius_min * 60.0 / window_s)

    try:
        speed_idx = locate_changeover(
            window_starts_ns, speed,
            reference_index=reference_index, search_radius=search_radius,
        )
        power_idx = locate_changeover(
            window_starts_ns, power,
            reference_index=reference_index, search_radius=search_radius,
        )
    except ValueError as exc:
        print(f"scada-timebase: {exc}", file=sys.stderr)
        return 2
    speed_ts = _ns_to_utc(int(window_starts_ns[speed_idx]))
    power_ts = _ns_to_utc(int(window_starts_ns[power_idx]))

    print(f"scada-timebase: day root {args.day_root} ({len(betriebsdaten)} Betriebsdaten file(s))")
    print(
        f"  search window: reference window {reference_index} +/- {search_radius} "
        f"windows ({search_radius_min:g} min)"
    )
    print(f"  speed (rpm) changeover:  {speed_ts.isoformat()}  (window {speed_idx})")
    print(f"  power (MW)  changeover:  {power_ts.isoformat()}  (window {power_idx})")
    print(f"  reference (audio-UTC state timeline): {reference.isoformat()}")
    print(f"  delta speed - reference: {(speed_ts - reference).total_seconds():+.3f} s")
    print(f"  delta power - reference: {(power_ts - reference).total_seconds():+.3f} s")

    print()
    print(
        f"  top-{_TOP_K_STEPS} largest whole-day steps per channel (alternatives "
        f"the windowed pick above cannot show):"
    )
    for label, series in (("speed (rpm)", speed), ("power (MW)", power)):
        print(f"    {label}:")
        candidates = top_k_steps(series, _TOP_K_STEPS)
        if not candidates:
            print("      (no finite first-difference in the whole series)")
        for rank, (idx, magnitude) in enumerate(candidates, start=1):
            step_ts = _ns_to_utc(int(window_starts_ns[idx]))
            print(f"      {rank}. {step_ts.isoformat()}  (window {idx})  |delta|={magnitude:.6g}")
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

    n_nonfinite = int(np.any(~np.isfinite(levels[strike_mask]), axis=1).sum())
    if n_nonfinite:
        print(
            f"gen-mic-profile: WARNING {n_nonfinite} of {int(strike_mask.sum())} "
            f"strike window(s) have a non-finite level in >= 1 channel (likely a "
            f"zero-sample window); channel_level_profile's nanmedian excludes them "
            f"per-channel rather than letting them poison the profile",
            file=sys.stderr,
        )
    try:
        profile = channel_level_profile(levels, strike_mask)
    except ValueError as exc:
        print(f"gen-mic-profile: {exc}", file=sys.stderr)
        return 2
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
        help=(
            "Check every file of every run instead of only the earliest per run. "
            "The earliest-file default is ASYMMETRIC: it can only under-report "
            "liveness (false DEAD), never over-report it (false LIVE), for a "
            "channel that only goes live partway through a run."
        ),
    )

    p_scada = sub.add_parser(
        "scada-timebase",
        help="Locate the day's changeover in Betriebsdaten rpm/power vs a reference UTC.",
        description=(
            "Locate the day's changeover in Betriebsdaten rpm/power vs a reference "
            "UTC timestamp (e.g. the audio-UTC state timeline's own changeover). "
            "NOTE: SPEED is structurally BLIND to a pump->phase-shifter changeover "
            "-- speed holds through it (e.g. 080726's ~13:05 UTC changeover holds "
            "at approx. -377 rpm) -- POWER is the diagnostic channel for that "
            "changeover, not speed. A day can also carry two near-equal steps in "
            "the SAME channel (e.g. 080726 power: the pump-start step AND the "
            "pump->phase-shifter changeover itself), where an unrestricted "
            "whole-day argmax can pick the wrong one -- the search is therefore "
            "windowed around --reference-utc (+/- --search-radius-min), and the "
            "whole day's top steps per channel are printed alongside the windowed "
            "pick so a human can see the alternatives."
        ),
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
    p_scada.add_argument(
        "--search-radius-min", type=float, default=_SEARCH_RADIUS_MIN_DEFAULT,
        help=(
            f"Restrict the changeover argmax search to +/- this many minutes "
            f"around --reference-utc's own window (default "
            f"{_SEARCH_RADIUS_MIN_DEFAULT:g}); keeps a same-channel decoy step "
            f"(e.g. pump start) from winning the argmax over the true changeover "
            f"on days with two near-equal steps."
        ),
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
