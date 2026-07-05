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


def _parse_burst_filename(path: Path) -> tuple[str, datetime]:
    """Parse `<stream>_YYYY-MM-DD_HH-MM-SS_ffffff.dat`; local CEST -> UTC."""
    m = _BURST_RE.match(path.name)
    if m is None:
        raise ValueError(f"{path.name}: does not match the burst filename pattern")
    local_dt = datetime.strptime(
        f"{m['date']}_{m['time']}", "%Y-%m-%d_%H-%M-%S"
    ).replace(tzinfo=_LOCAL_TZ)
    return m["stream"], local_dt.astimezone(ZoneInfo("UTC"))


def _split_on_gaps(files: list[BurstFile]) -> list[list[BurstFile]]:
    """Split a time-sorted pooled file list wherever a gap > 15 min occurs."""
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


def _discover_burst_folder(folder: Path) -> list[Run]:
    all_files: list[BurstFile] = []
    for candidate in folder.iterdir():
        if not candidate.is_file():
            continue
        try:
            stream, start_utc = _parse_burst_filename(candidate)
        except ValueError:
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
        runs.append(Run(name=_group_name(folder.name, i, len(groups)), files=files_by_stream))
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
        winner, loser = (
            (incumbent, candidate)
            if incumbent.stat().st_size >= candidate.stat().st_size
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
