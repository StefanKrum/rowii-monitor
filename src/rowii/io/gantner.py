"""Reader for the Gantner 'UniversalDataBinFile' container used at Rodundwerk II.

Layout (verified on the June-2026 delivery, see plan header): version-prefixed magic
string, JSON metadata, channel descriptor block (name [unit] uuid per channel),
a run of 0x2a padding, then frames of uint64 ns-timestamp + one float32 per channel.
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MAGIC = b"UniversalDataBinFile - GANTNER instruments"
_UUID_RE = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_NAME_RE = re.compile(rb"([ -~]{1,60})\x00")
_HEADER_SCAN = 64 * 1024


class GantnerFormatError(Exception):
    """Raised when a .dat file does not match the expected container layout."""


@dataclass(frozen=True)
class GantnerHeader:
    source_name: str
    channel_names: list[str]
    channel_units: list[str]
    t0_ns: int
    sample_rate_hz: float
    n_frames: int


@dataclass(frozen=True)
class GantnerFile:
    header: GantnerHeader
    timestamps_ns: np.ndarray
    data: np.ndarray


def _find_json(buf: bytes) -> tuple[dict[str, str], int]:
    start = buf.find(b"{")
    if start < 0:
        raise GantnerFormatError("no JSON metadata found in header region")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(buf)):
        c = buf[i]
        if in_str:
            if esc:
                esc = False
            elif c == 0x5C:
                esc = True
            elif c == 0x22:
                in_str = False
        elif c == 0x22:
            in_str = True
        elif c == 0x7B:
            depth += 1
        elif c == 0x7D:
            depth -= 1
            if depth == 0:
                return json.loads(buf[start : i + 1]), i + 1
    raise GantnerFormatError("unterminated JSON metadata")


def _parse_descriptor(buf: bytes, json_end: int) -> tuple[list[str], list[str], int]:
    pad = re.search(rb"\x2a{8,}", buf[json_end:])
    if pad is None:
        raise GantnerFormatError("channel-descriptor padding (0x2a run) not found")
    desc = buf[json_end : json_end + pad.start()]
    names: list[str] = []
    units: list[str] = []
    cursor = 0
    for m in _UUID_RE.finditer(desc):
        # exclude the 2-byte little-endian length prefix of the UUID token itself: its
        # low byte (uuid string length is always 36 = 0x24 = '$') is printable ASCII and
        # would otherwise be mistaken for a 1-character name/unit token.
        tokens = [
            t.group(1).decode("ascii") for t in _NAME_RE.finditer(desc[cursor : m.start() - 2])
        ]
        tokens = [t for t in tokens if not _UUID_RE.fullmatch(t.encode())]
        if not tokens:
            raise GantnerFormatError("channel descriptor without a name token")
        names.append(tokens[0])
        units.append(tokens[1] if len(tokens) > 1 else "")
        cursor = m.end()
    if not names:
        raise GantnerFormatError("no channels found in descriptor block")
    return names, units, json_end + pad.end()


def _parse_header_region(path: Path) -> tuple[dict[str, str], list[str], list[str], int]:
    with path.open("rb") as fh:
        head = fh.read(_HEADER_SCAN)
    if len(head) < 4 or struct.unpack(">H", head[:2])[0] != 0x006B or _MAGIC not in head[:200]:
        raise GantnerFormatError(f"{path.name}: bad magic / not a Gantner container")
    meta, json_end = _find_json(head)
    names, units, data_off = _parse_descriptor(head, json_end)
    return meta, names, units, data_off


def read_gantner(path: Path) -> GantnerFile:
    meta, names, units, data_off = _parse_header_region(path)
    n_ch = len(names)
    frame_size = 8 + 4 * n_ch
    body = np.fromfile(path, dtype=np.uint8, offset=data_off)
    n_frames = body.size // frame_size
    if n_frames == 0:
        raise GantnerFormatError(f"{path.name}: no complete frames")
    frames = body[: n_frames * frame_size].reshape(n_frames, frame_size)
    ts = frames[:, :8].copy().view("<u8").reshape(-1)
    data = frames[:, 8:].copy().view("<f4").reshape(n_frames, n_ch)
    dt = np.diff(ts.astype(np.int64))
    rate = 1e9 / float(np.median(dt)) if dt.size else 0.0
    header = GantnerHeader(
        source_name=str(meta.get("SourceName", path.stem)),
        channel_names=names,
        channel_units=units,
        t0_ns=int(ts[0]),
        sample_rate_hz=rate,
        n_frames=n_frames,
    )
    return GantnerFile(header=header, timestamps_ns=ts, data=data)


def read_header(path: Path) -> GantnerHeader:
    """Header + rate estimate from the first 1000 frames only (no full read)."""
    meta, names, units, data_off = _parse_header_region(path)
    n_ch = len(names)
    frame_size = 8 + 4 * n_ch
    size = path.stat().st_size
    n_frames = (size - data_off) // frame_size
    with path.open("rb") as fh:
        fh.seek(data_off)
        probe = fh.read(frame_size * min(1000, max(n_frames, 1)))
    arr = np.frombuffer(probe, dtype=np.uint8)
    k = arr.size // frame_size
    ts = arr[: k * frame_size].reshape(k, frame_size)[:, :8].copy().view("<u8").reshape(-1)
    dt = np.diff(ts.astype(np.int64))
    rate = 1e9 / float(np.median(dt)) if dt.size else 0.0
    return GantnerHeader(str(meta.get("SourceName", path.stem)), names, units,
                         int(ts[0]) if k else 0, rate, int(n_frames))
