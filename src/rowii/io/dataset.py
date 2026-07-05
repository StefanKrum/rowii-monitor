"""Recording discovery: filenames-only index of TU/PU bursts and Betriebsdaten files.

Discovery never opens a file -- it works purely on directory names and filename
timestamps (`pathlib` + `re` + `zoneinfo`), so it is fast and testable with empty
files. Filename timestamps are LOCAL Europe/Vienna time; they are converted to
UTC as a *hint* only -- per the format's authoritative source, the UDBF frame
timestamps recovered by `rowii.io.gantner` are the ground truth for actual
alignment.
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


@dataclass(frozen=True)
class BurstFile:
    path: Path
    stream: str
    start_utc_hint: datetime


@dataclass(frozen=True)
class Run:
    name: str
    files: dict[str, list[BurstFile]]


@dataclass(frozen=True)
class RecordingIndex:
    runs: list[Run]
    betriebsdaten: list[Path]


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


def _split_on_gaps(files: list[BurstFile]) -> list[list[BurstFile]]:
    """Split a time-sorted pooled file list wherever a gap > 15 min occurs.

    *files* is expected to already be the POOLED sequence across ALL streams
    of a burst folder (see `_discover_burst_folder`), not a single stream's
    files. This assumes the DAQ exports every stream's files synchronously,
    so a genuine session boundary shows up as a gap in the pooled timeline
    too. A gap that exists in only one stream -- while other streams have a
    file in between that bridges it in the pooled sequence -- does NOT
    produce a run boundary here; it is a masked per-stream gap, detected and
    logged separately (see `_warn_on_masked_stream_gaps`), not treated as a
    split point.
    """
    if not files:
        return []
    groups: list[list[BurstFile]] = [[files[0]]]
    for prev, cur in zip(files, files[1:], strict=False):
        if cur.start_utc_hint - prev.start_utc_hint > _GAP_THRESHOLD:
            groups.append([])
        groups[-1].append(cur)
    return groups


def _group_name(folder_name: str, group_index: int, n_groups: int) -> str:
    folder_lower = folder_name.lower()
    if n_groups == 1:
        return folder_lower
    if n_groups == 2:
        return f"{folder_lower}-{'morning' if group_index == 0 else 'afternoon'}"
    return f"{folder_lower}-{group_index + 1}"


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


def _discover_burst_folder(folder: Path) -> list[Run]:
    """Discover runs in one `TU`/`PU` folder by pooling all streams and gap-splitting.

    Files of ALL streams present in *folder* are pooled together and sorted by
    filename timestamp before gap-splitting (see `_split_on_gaps`). This
    assumes the DAQ exports every stream's burst files synchronously, so a
    real session boundary appears as a gap in every stream at once. A gap
    inside a single stream that other streams' files bridge in the pooled
    timeline does NOT create a run boundary here -- it is detected and logged
    as a warning instead (see `_warn_on_masked_stream_gaps`), not silently
    absorbed into the run.
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
        run = Run(name=_group_name(folder.name, i, len(groups)), files=files_by_stream)
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


def discover(data_root: Path) -> RecordingIndex:
    """Index runs (TU/PU bursts) and Betriebsdaten files under *data_root* by name only."""
    runs: list[Run] = []
    betriebsdaten: list[Path] = []
    for folder in sorted(data_root.rglob("*")):
        if not folder.is_dir():
            continue
        if folder.name in ("TU", "PU"):
            runs.extend(_discover_burst_folder(folder))
        elif folder.name == "Betriebsdaten":
            betriebsdaten.extend(_discover_betriebsdaten(folder))
    return RecordingIndex(runs=runs, betriebsdaten=betriebsdaten)
