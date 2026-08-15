"""Recording discovery: filenames-only index of measurement-day trees.

Discovery never opens a file -- it works purely on directory names and filename
timestamps (`pathlib` + `re` + `zoneinfo`), so it is fast and testable with empty
files. Filename timestamps are LOCAL Europe/Vienna time; they are converted to
UTC as a *hint* only -- per the format's authoritative source, the UDBF frame
timestamps recovered by `rowii.io.gantner` are the ground truth for actual
alignment.

`discover(data_root)` accepts two shapes of `data_root`:

- **Single-tree (legacy, backward compatible):** `data_root` itself directly
  contains a `"* Messung"` directory (e.g. `<data_root>/20260626 Messung/TU`).
  Run names carry no day prefix -- the exact pre-addendum behaviour.
- **Parent root (multi-day):** `data_root` contains several `illwerke-<dayid>`
  subdirectories, each itself a day tree with its own `"* Messung"` directory
  (e.g. `<data_root>/illwerke-010726/20260701 Messung/TU1`). Every run
  discovered under such a day tree is prefixed `<dayid>-`, where `dayid` is the
  token after the last `-` in the day tree's own directory name.

The distinguishing test is structural, not a hardcoded root-name pattern: for a
given `"* Messung"` directory, its immediate PARENT is the "day root". If that
day root IS `data_root` itself, this is the single-tree/legacy case (no
prefix). If the day root is some deeper descendant of `data_root` (an
`illwerke-*` directory found by scanning), every run under it gets that day
root's dayid prefix. This means a `data_root` can even mix both shapes without
special-casing -- each `"* Messung"` directory is resolved independently.

A **session** is any direct subfolder of a `"* Messung"` directory that
contains at least one burst-pattern file (no more hardcoded `("TU", "PU")`
folder-name whitelist -- operators name session folders however they like,
e.g. real folder names `TU1`, `TU2`, `TU_PH_TU`, `PU_PH_PU_PH_PU_PH`). A
`Betriebsdaten` folder (by name) is still recognised and handled separately,
per day tree -- see `RecordingIndex.betriebsdaten_by_day`.
"""
from __future__ import annotations

import logging
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from rowii.io.gantner import read_header

logger = logging.getLogger(__name__)

_LOCAL_TZ = ZoneInfo("Europe/Vienna")
_GAP_THRESHOLD = timedelta(minutes=15)
_BETRIEBSDATEN_DIR_NAME = "Betriebsdaten"
_MESSUNG_DIR_SUFFIX = " Messung"

_STREAMS = (
    "RAWGeneratorMic__0",
    "RAWTurbineMic__1",
    "RAWGeneratorVib__2",
    "RAWTurbineVib__3",
)
_BURST_RE = re.compile(
    r"^(?P<stream>" + "|".join(_STREAMS) + r")"
    r"_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_(?P<frac>\d{6})\.dat$"
)
_BETRIEBSDATEN_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})-00-00(?:_\d+)?\.dat$"
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class BurstFile:
    path: Path
    stream: str
    start_utc_hint: datetime


@dataclass(frozen=True)
class Run:
    name: str
    files: dict[str, list[BurstFile]]
    day_root: Path
    """The day tree's root directory (the immediate parent of this run's `"*
    Messung"` directory). Equals `data_root` itself in single-tree/legacy mode;
    an `illwerke-<dayid>` directory in parent-root/multi-day mode. Used to look
    up exactly this run's own day's Betriebsdaten via
    `RecordingIndex.betriebsdaten_by_day` -- a run must never be evaluated
    against a DIFFERENT day's SCADA data."""


@dataclass(frozen=True)
class RecordingIndex:
    runs: list[Run]
    betriebsdaten: list[Path]
    """Every discovered Betriebsdaten file, pooled across all day trees and
    sorted by (date, hour) -- kept for backward compatibility with callers that
    only need "all SCADA files anywhere under data_root" (e.g. a single-tree
    root only ever has one day, so this is already correctly scoped there).
    Multi-day callers that must respect per-run day scoping should use
    `betriebsdaten_by_day` instead (see `Run.day_root`)."""
    betriebsdaten_by_day: dict[Path, list[Path]]
    """Betriebsdaten files, grouped by day root (see `Run.day_root`). A day
    tree with no `Betriebsdaten` folder at all (e.g. the 27.06 delivery) simply
    has no key here -- callers must not fall back to a different day's files or
    to the flat `betriebsdaten` list when a run's own day has none."""


class _NoBurstMatch(ValueError):
    """Filename does not match the burst pattern at all (e.g. a stray non-.dat sibling)."""


class _UnparseableBurstDate(ValueError):
    """Filename matches the burst pattern's shape but its date/time is not a real timestamp."""


def _parse_burst_filename(path: Path) -> tuple[str, datetime]:
    """Parse `<stream>_YYYY-MM-DD_HH-MM-SS_ffffff.dat`; local CEST -> UTC."""
    m = _BURST_RE.match(path.name)
    if m is None:
        raise _NoBurstMatch(f"{path.name}: does not match the burst filename pattern")
    try:
        local_dt = datetime.strptime(
            f"{m['date']}_{m['time']}", "%Y-%m-%d_%H-%M-%S"
        ).replace(tzinfo=_LOCAL_TZ)
    except ValueError as exc:
        raise _UnparseableBurstDate(
            f"{path.name}: matches the burst filename pattern but has an invalid date/time"
        ) from exc
    return m["stream"], local_dt.astimezone(ZoneInfo("UTC"))


def _sanitize(name: str) -> str:
    """Lowercase *name*, collapsing every run of non-alphanumeric characters to `_`.

    Applied to session folder names when building a run's base name (e.g.
    `"TU_PH_TU"` -> `"tu_ph_tu"`, already idempotent since it only has
    alphanumerics and underscores; `"PU_PH_PU_PH_PU_PH"` -> unchanged but
    lowercased). Leading/trailing separators are stripped so a folder name
    that happens to start or end with a non-alnum character does not leave a
    stray leading/trailing underscore in the run name.
    """
    return _NON_ALNUM_RE.sub("_", name.lower()).strip("_")


def _split_on_gaps(files: list[BurstFile]) -> list[list[BurstFile]]:
    """Split a time-sorted pooled file list wherever a gap > 15 min occurs.

    *files* is expected to already be the POOLED sequence across ALL streams
    of a burst folder (see `_discover_session_folder`), not a single stream's
    files. This assumes the DAQ exports every stream's files synchronously, so
    a genuine session boundary shows up as a gap in the pooled timeline too. A
    gap that exists in only one stream -- while other streams have a file in
    between that bridges it in the pooled sequence -- does NOT produce a run
    boundary here; it is a masked per-stream gap, detected and logged
    separately (see `_warn_on_masked_stream_gaps`), not treated as a split
    point.
    """
    if not files:
        return []
    groups: list[list[BurstFile]] = [[files[0]]]
    for prev, cur in zip(files, files[1:], strict=False):
        if cur.start_utc_hint - prev.start_utc_hint > _GAP_THRESHOLD:
            groups.append([])
        groups[-1].append(cur)
    return groups


def _group_name(base_name: str, group_index: int, n_groups: int) -> str:
    if n_groups == 1:
        return base_name
    if n_groups == 2:
        return f"{base_name}-{'morning' if group_index == 0 else 'afternoon'}"
    return f"{base_name}-{group_index + 1}"


def _warn_on_masked_stream_gaps(run: Run) -> None:
    """Warn when a stream has an internal gap that the pooled split missed.

    `_split_on_gaps` pools all streams together before splitting, so a run
    boundary is only detected where the POOLED timeline has a gap. If one
    stream drops a file while a sibling stream keeps filling every slot, the
    pooled sequence never shows the gap and the run stays intact -- correct
    per the synchronous-DAQ assumption, but worth flagging in case a stream
    genuinely lost data mid-run. Checked per-stream, per-run, after grouping.
    """
    for stream, bursts in run.files.items():
        for prev, cur in zip(bursts, bursts[1:], strict=False):
            gap = cur.start_utc_hint - prev.start_utc_hint
            if gap > _GAP_THRESHOLD:
                logger.warning(
                    "Run %s: stream %s has a %.1f-min gap between %s and %s "
                    "that other streams bridged (no run boundary was created)",
                    run.name, stream, gap.total_seconds() / 60,
                    prev.path.name, cur.path.name,
                )


def _peek_has_burst_files(folder: Path) -> bool:
    """True iff *folder* directly contains at least one burst-pattern file.

    Used to decide whether a `"* Messung"` subfolder is a session (replaces
    the old hardcoded `("TU", "PU")` name whitelist) -- any non-`Betriebsdaten`
    subfolder qualifies as long as it actually holds burst files, regardless
    of what the operator named it.
    """
    for candidate in folder.iterdir():
        if candidate.is_file() and _BURST_RE.match(candidate.name):
            return True
    return False


def _discover_session_folder(folder: Path, base_name: str, day_root: Path) -> list[Run]:
    """Discover runs in one session folder by pooling all streams and gap-splitting.

    Files of ALL streams present in *folder* are pooled together and sorted by
    filename timestamp before gap-splitting (see `_split_on_gaps`). This
    assumes the DAQ exports every stream's burst files synchronously, so a
    real session boundary appears as a gap in every stream at once. A gap
    inside a single stream that other streams' files bridge in the pooled
    timeline does NOT create a run boundary here -- it is detected and logged
    as a warning instead (see `_warn_on_masked_stream_gaps`), not silently
    absorbed into the run.

    *base_name* is the (already day-prefixed, if applicable) run name stem;
    *day_root* is stamped onto every resulting `Run` so it can look up its own
    day's Betriebsdaten later (see `Run.day_root`).
    """
    all_files: list[BurstFile] = []
    for candidate in folder.iterdir():
        if not candidate.is_file():
            continue
        try:
            stream, start_utc = _parse_burst_filename(candidate)
        except _UnparseableBurstDate:
            logger.warning(
                "Skipping %s: matches the burst filename pattern but its date/time "
                "could not be parsed",
                candidate.name,
            )
            continue
        except _NoBurstMatch:
            continue
        all_files.append(BurstFile(path=candidate, stream=stream, start_utc_hint=start_utc))
    all_files.sort(key=lambda f: f.start_utc_hint)

    groups = _split_on_gaps(all_files)
    runs: list[Run] = []
    for i, group in enumerate(groups):
        files_by_stream: dict[str, list[BurstFile]] = {}
        for burst in group:
            files_by_stream.setdefault(burst.stream, []).append(burst)
        for bursts in files_by_stream.values():
            bursts.sort(key=lambda f: f.start_utc_hint)
        run = Run(
            name=_group_name(base_name, i, len(groups)),
            files=files_by_stream,
            day_root=day_root,
        )
        _warn_on_masked_stream_gaps(run)
        runs.append(run)
    return runs


def _discover_betriebsdaten(folder: Path) -> list[Path]:
    by_hour: dict[tuple[str, str], Path] = {}
    for candidate in folder.iterdir():
        if not candidate.is_file():
            continue
        m = _BETRIEBSDATEN_RE.match(candidate.name)
        if m is None:
            continue
        key = (m["date"], m["hour"])
        incumbent = by_hour.get(key)
        if incumbent is None:
            by_hour[key] = candidate
            continue
        # Path.iterdir() gives no ordering guarantee, so "incumbent" vs.
        # "candidate" here is an arbitrary arrival order -- the tie-break rule
        # must not depend on it. Rule: larger size wins; if sizes are exactly
        # equal, the lexicographically smaller filename wins (deterministic
        # regardless of which one iterdir() happened to yield first).
        incumbent_size, candidate_size = incumbent.stat().st_size, candidate.stat().st_size
        if incumbent_size != candidate_size:
            winner, loser = (
                (incumbent, candidate)
                if incumbent_size > candidate_size
                else (candidate, incumbent)
            )
        else:
            winner, loser = (
                (incumbent, candidate)
                if incumbent.name < candidate.name
                else (candidate, incumbent)
            )
        logger.warning(
            "Betriebsdaten duplicate for %s %s:00: keeping %s (%d bytes), dropping %s (%d bytes)",
            key[0], key[1], winner.name, winner.stat().st_size,
            loser.name, loser.stat().st_size,
        )
        by_hour[key] = winner

    return sorted(by_hour.values(), key=lambda p: (p.stem[:10], p.stem[11:13]))


def _dayid_for(day_root: Path, data_root: Path) -> str | None:
    """The day-prefix token for *day_root*, or `None` in single-tree/legacy mode.

    `None` exactly when `day_root == data_root` (no `illwerke-<dayid>` layer
    exists between them) -- the single-tree/backward-compat case, which must
    keep producing entirely unprefixed run names. Otherwise the token after
    the last `-` in `day_root.name` (e.g. `"illwerke-010726"` -> `"010726"`;
    falls back to the whole name if it has no `-` at all).
    """
    if day_root == data_root:
        return None
    return day_root.name.rsplit("-", 1)[-1]


def _discover_messung_dir(messung_dir: Path, data_root: Path) -> tuple[list[Run], list[Path]]:
    """Runs + Betriebsdaten files found directly under one `"* Messung"` directory."""
    day_root = messung_dir.parent
    dayid = _dayid_for(day_root, data_root)

    runs: list[Run] = []
    betriebsdaten: list[Path] = []
    for candidate in sorted(messung_dir.iterdir()):
        if not candidate.is_dir():
            continue
        if candidate.name == _BETRIEBSDATEN_DIR_NAME:
            betriebsdaten.extend(_discover_betriebsdaten(candidate))
            continue
        if not _peek_has_burst_files(candidate):
            continue
        sanitized = _sanitize(candidate.name)
        base_name = f"{dayid}-{sanitized}" if dayid is not None else sanitized
        runs.extend(_discover_session_folder(candidate, base_name, day_root))

    return runs, betriebsdaten


def discover(data_root: Path) -> RecordingIndex:
    """Index every run and Betriebsdaten file under *data_root* by name only.

    See the module docstring for the single-tree vs. parent-root distinction
    and exactly how run names are derived in each case.
    """
    runs: list[Run] = []
    betriebsdaten_by_day: dict[Path, list[Path]] = {}
    for folder in sorted(data_root.rglob("*")):
        if not folder.is_dir() or not folder.name.endswith(_MESSUNG_DIR_SUFFIX):
            continue
        messung_runs, messung_betriebsdaten = _discover_messung_dir(folder, data_root)
        runs.extend(messung_runs)
        if messung_betriebsdaten:
            day_root = folder.parent
            betriebsdaten_by_day.setdefault(day_root, []).extend(messung_betriebsdaten)

    pooled_betriebsdaten = sorted(
        (p for files in betriebsdaten_by_day.values() for p in files),
        key=lambda p: (p.stem[:10], p.stem[11:13]),
    )
    return RecordingIndex(
        runs=runs,
        betriebsdaten=pooled_betriebsdaten,
        betriebsdaten_by_day=betriebsdaten_by_day,
    )


# ---------------------------------------------------------------------------
# DAQ epoch-2000 clock quirk (Task 10): derived per-run / per-Betriebsdaten-file-set
# offset that maps the raw on-disk time axis onto true UTC. Every Gantner UDBF file
# in this dataset (audio, vibration, AND Betriebsdaten) carries binary frame
# timestamps from a DAQ clock that counts seconds since 2000-01-01 LOCAL time but
# declares them as Unix (1970) nanoseconds -- `GantnerHeader.t0_ns`, read naively,
# therefore decodes to a wall-clock-of-day that matches the file's true local time
# exactly, under a fixed, wrong epoch YEAR (see `run_utc_offset_ns`'s own docstring
# for the worked numeric example). The offset is always derived, never hardcoded,
# since it depends on which local UTC offset (CEST/CET) was in effect when the
# recording was made -- `_median_utc_offset_ns` below is the one place that
# median/round/plausibility-gate/per-file-warning algorithm lives, shared by
# `run_utc_offset_ns` (a burst `Run`, D1) and `betriebsdaten_utc_offset_ns` (a flat
# Betriebsdaten file list, D3).
# ---------------------------------------------------------------------------

_HOUR_NS = 3_600_000_000_000
"""One hour in nanoseconds -- the DAQ-clock-quirk offset (epoch-2000-as-1970 shift
minus a whole local UTC offset) is always an exact multiple of this."""
_OFFSET_PLAUSIBILITY_GATE_NS = _HOUR_NS
"""Below this magnitude, a derived offset is treated as ordinary DAQ jitter on an
ALREADY-correct clock (e.g. a future dataset), not a genuine quirky-clock shift --
`_median_utc_offset_ns` returns 0 rather than inventing a tiny rounded-to-an-hour
offset for data that was never quirky to begin with."""
_OFFSET_WARN_TOLERANCE_NS = 2_000_000_000
"""Max per-file deviation from the derived, rounded-to-the-hour offset before
`_median_utc_offset_ns` logs a WARNING for that file. Filename-second-truncation
scatter is documented at 0-900 ms (task-10-brief.md); 2 s leaves a comfortable
margin above that while still catching a genuinely mismatched file."""


def _hint_utc_ns(hint: datetime) -> int:
    """*hint* (a tz-aware `datetime`, e.g. `BurstFile.start_utc_hint`) as nanoseconds
    since the Unix epoch. `round()`, not truncation, of the sub-nanosecond float
    product -- see `_median_utc_offset_ns`'s own docstring for why the residual
    float-precision noise this can leave (up to a few hundred ns at 2020s-scale
    timestamps) never affects the final derived offset."""
    return round(hint.timestamp() * 1e9)


def _median_utc_offset_ns(
    entries: Sequence[tuple[Path, int, datetime]], *, context: str
) -> int:
    """Shared derivation behind `run_utc_offset_ns` and `betriebsdaten_utc_offset_ns`:
    the median raw offset (`hint_utc_ns - header_t0_ns`) over *entries*, rounded to
    the NEAREST HOUR, 0 below the 1-hour plausibility gate (`_OFFSET_PLAUSIBILITY_
    GATE_NS`). Logs one WARNING per entry whose own raw offset deviates from the
    rounded value by more than `_OFFSET_WARN_TOLERANCE_NS`, naming *context* (the
    run, or the Betriebsdaten file set) and that entry's file.

    The nearest-hour rounding is not cosmetic: the true offset (epoch-2000 shift
    minus a whole local UTC offset, both exact numbers of seconds divisible by
    3600) IS always an exact multiple of one hour, so rounding the noisy median
    recovers it exactly regardless of any sub-second scatter in the inputs
    (filename-second truncation) or sub-microsecond float-precision noise from
    `_hint_utc_ns`'s `datetime.timestamp() * 1e9` product (at 2020s-scale
    timestamps this product needs ~61 bits to represent exactly, a few bits past
    float64's 52-bit mantissa -- utterly negligible next to the half-an-hour of
    slack the rounding step tolerates).

    Args:
        entries: `(path, header_t0_ns, start_utc_hint)` triples -- *path* is used
            only for the per-entry deviation warning's message.
        context: Human-readable label for the warning message (e.g. `f"run
            {run.name!r}"` or `"Betriebsdaten set"`).

    Returns:
        The derived offset in nanoseconds: 0 if *entries* is empty or the median
        raw offset's magnitude is below the 1-hour plausibility gate, else the
        rounded-to-the-hour value.
    """
    if not entries:
        return 0
    raw_offsets = [_hint_utc_ns(hint) - t0_ns for _, t0_ns, hint in entries]
    median_offset = int(statistics.median(raw_offsets))
    if abs(median_offset) < _OFFSET_PLAUSIBILITY_GATE_NS:
        return 0
    rounded = round(median_offset / _HOUR_NS) * _HOUR_NS
    for (path, _t0_ns, _hint), raw_offset in zip(entries, raw_offsets, strict=True):
        deviation_ns = raw_offset - rounded
        if abs(deviation_ns) > _OFFSET_WARN_TOLERANCE_NS:
            logger.warning(
                "%s: file %s raw UTC offset %.3fs deviates from the derived, "
                "rounded-to-the-hour offset %.3fs by %.3fs (> 2s tolerance)",
                context, path.name, raw_offset / 1e9, rounded / 1e9, deviation_ns / 1e9,
            )
    return rounded


def run_utc_offset_ns(run: Run) -> int:
    """Nanosecond offset that maps this run's raw DAQ time axis onto true UTC
    (documented DAQ quirk: clock counts from 2000-01-01 local time but labels
    the count as Unix nanoseconds). Derived, never hardcoded: median over all
    of the run's files of (start_utc_hint − header.t0_ns), rounded to the
    NEAREST HOUR (the true offset is epoch-2000-shift minus the local UTC
    offset, always a whole hour; sub-second scatter comes from
    filename-second truncation). Logs one WARNING naming the run and the file
    if any file's raw offset deviates from the rounded value by more than 2 s.

    Example (verified against the real June-2026 delivery): a file whose
    `header.t0_ns` decodes (naively, as Unix nanoseconds) to
    `1996-06-27T06:41:03Z` and whose filename-derived `start_utc_hint` is
    `2026-06-27T04:41:03Z` (the true UTC instant -- local Europe/Vienna
    wall-clock digits `06:41:03` on 2026-06-27, CEST = UTC+2) yields an
    offset of `946_677_600` seconds (`946_684_800` s epoch-2000 shift minus
    `7_200` s CEST) -- exactly a whole number of hours (262,966), so the
    rounding step is a no-op here; winter (CET, UTC+1) data would instead
    give `946_681_200` s (`946_684_800` s minus `3_600` s).

    Header reads go via `read_header` (cheap -- first ~1000 frames only, never
    a full-file read); every file across every stream of *run* contributes to
    the median (not just one variant's own streams), for the largest robust
    sample. A run with a CORRECT clock (a future dataset) comes out as offset
    0 via the plausibility gate (`_median_utc_offset_ns`): if the median raw
    offset magnitude is below 1 hour, this returns 0 rather than inventing a
    spurious rounded-to-an-hour shift for data that was never quirky.

    Args:
        run: The `Run` (burst files by stream) to derive the offset for.

    Returns:
        The derived offset in nanoseconds (0 for an empty run, or one with no
        genuine quirky-clock shift).
    """
    entries = [
        (bf.path, read_header(bf.path).t0_ns, bf.start_utc_hint)
        for files in run.files.values()
        for bf in files
    ]
    return _median_utc_offset_ns(entries, context=f"run {run.name!r}")


def _betriebsdaten_utc_hint(path: Path) -> datetime | None:
    """*path*'s own filename-derived local-time-of-day, converted to UTC (mirrors
    `_parse_burst_filename`'s local -> UTC conversion, applied to the Betriebsdaten
    filename convention instead: `_BETRIEBSDATEN_RE`, whole-hour local time only).
    `None` if *path*'s name does not match that pattern at all.
    """
    m = _BETRIEBSDATEN_RE.match(path.name)
    if m is None:
        return None
    local_dt = datetime.strptime(
        f"{m['date']}_{m['hour']}-00-00", "%Y-%m-%d_%H-%M-%S"
    ).replace(tzinfo=_LOCAL_TZ)
    return local_dt.astimezone(ZoneInfo("UTC"))


def betriebsdaten_utc_offset_ns(files: Sequence[Path]) -> int:
    """Nanosecond offset that maps a Betriebsdaten file set's raw DAQ time axis onto
    true UTC -- mirrors `run_utc_offset_ns` (D1: same DAQ-clock quirk, same
    median/round-to-the-hour/plausibility-gate derivation, see that function's own
    docstring for the numeric worked example and full rationale) applied to the
    SCADA Betriebsdaten file set instead of a burst `Run`.

    Betriebsdaten filenames carry no pre-parsed `BurstFile.start_utc_hint` (unlike
    burst files), so this function derives that hint itself, per file
    (`_betriebsdaten_utc_hint`, `_BETRIEBSDATEN_RE`'s local Europe/Vienna
    wall-clock convention), before delegating to the shared `_median_utc_offset_ns`.
    The SCADA clock is independent hardware from the audio/vibration DAQ, so this
    function's own derivation must NEVER be skipped in favour of blindly reusing
    `run_utc_offset_ns`'s result for the corresponding audio run -- the two are
    expected to agree closely but are derived independently; a caller that has
    both values is responsible for cross-checking them (e.g.
    `rowii.scada.labels.load_scada_window_means`'s `audio_run_offset_ns`
    parameter warns if they disagree by more than 2 s).

    Files whose name does not match the Betriebsdaten filename pattern at all are
    skipped (no hint can be derived for them) -- every real Betriebsdaten file
    matches by construction (`_discover_betriebsdaten`), so this only matters for a
    caller passing arbitrarily-named files (e.g. a synthetic test fixture that was
    never trying to model the DAQ-clock quirk in the first place), for which this
    function is then a safe no-op (returns 0, same as the plausibility gate's own
    "nothing quirky here" outcome).

    Args:
        files: Betriebsdaten file paths (typically the day-scoped subset already
            selected for one run's grid, or a whole day's files).

    Returns:
        The derived offset in nanoseconds (0 if *files* is empty, none of its
        names match the Betriebsdaten filename pattern, or there is no genuine
        quirky-clock shift to derive).
    """
    entries: list[tuple[Path, int, datetime]] = []
    for path in files:
        hint = _betriebsdaten_utc_hint(path)
        if hint is None:
            continue
        entries.append((path, read_header(path).t0_ns, hint))
    return _median_utc_offset_ns(entries, context="Betriebsdaten set")
