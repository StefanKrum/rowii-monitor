"""Reader for the Gantner 'UniversalDataBinFile' container used at Rodundwerk II.

Layout (verified on the June-2026 delivery, see plan header): version-prefixed magic
string, JSON metadata, channel descriptor block (name [unit] uuid per channel),
a run of 0x2a padding, then frames of uint64 ns-timestamp + one float32 per channel.
"""
from __future__ import annotations

import json
import logging
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MAGIC = b"UniversalDataBinFile - GANTNER instruments"
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_HEADER_SCAN = 64 * 1024
_MAX_TOKEN_LEN = 200


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


def _scan_tokens(desc: bytes) -> list[str]:
    """Validating scan-tokenizer over the channel-descriptor region.

    Every real token is length-prefixed: ``<u16 LE length L><L printable-ASCII
    bytes>\\x00``. A plain printable-ASCII regex (the previous approach) cannot
    distinguish a token's own 2-byte length prefix from real token bytes: whenever a
    token's length L falls in [32, 126], the prefix's low byte (L & 0xFF) is itself a
    printable ASCII character, and a regex scan mis-parses it as a spurious 1-character
    token, silently corrupting the following name/unit.

    This scanner is structural instead: at each byte offset it validates that a
    plausible length-prefixed token actually starts there (length in range, all payload
    bytes printable, NUL terminator present) before accepting it. Unknown filler bytes
    between tokens (observed as a small per-channel descriptor blob in the container's
    reverse-engineered binary layout) fail validation and are skipped one byte at a time.
    """
    tokens: list[str] = []
    i = 0
    n = len(desc)
    while i + 2 <= n:
        length = struct.unpack_from("<H", desc, i)[0]
        end = i + 2 + length
        if 1 <= length <= _MAX_TOKEN_LEN and end < n:
            payload = desc[i + 2 : end]
            if all(0x20 <= b <= 0x7E for b in payload) and desc[end] == 0x00:
                tokens.append(payload.decode("ascii"))
                i = end + 1
                continue
        i += 1
    return tokens


def _parse_descriptor(buf: bytes, json_end: int) -> tuple[list[str], list[str], int]:
    pad = re.search(rb"\x2a{8,}", buf[json_end:])
    if pad is None:
        raise GantnerFormatError("channel-descriptor padding (0x2a run) not found")
    desc = buf[json_end : json_end + pad.start()]
    names: list[str] = []
    units: list[str] = []
    pending: list[str] = []
    for token in _scan_tokens(desc):
        if _UUID_RE.fullmatch(token):
            if not pending:
                raise GantnerFormatError("channel descriptor without a name token")
            names.append(pending[0])
            units.append(pending[1] if len(pending) > 1 else "")
            if len(pending) > 2:
                logger.warning(
                    "channel descriptor has %d name/unit tokens before its UUID "
                    "(expected at most 2: name, unit); using the first two, "
                    "ignoring: %r",
                    len(pending),
                    pending[2:],
                )
            pending = []
        else:
            pending.append(token)
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
