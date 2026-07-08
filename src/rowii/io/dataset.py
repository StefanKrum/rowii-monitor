"""Recording discovery: filenames-only index of measurement-day trees.

Discovery never opens a file -- it works purely on directory names and filename
timestamps (`pathlib` + `re` + `zoneinfo`), so it is fast and testable with empty
files. Filename timestamps are LOCAL Europe/Vienna time; they are converted to
UTC as a *hint* only -- per the format's authoritative source, the UDBF frame
timestamps recovered by `rowii.io.gantner` are the ground truth for actual
alignment.

`discover(data_root)` accepts two shapes of `data_root` (spec: docs/superpowers/
specs/2026-07-07-step1-multiday-phase-shifter-addendum.md §2):

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
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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
