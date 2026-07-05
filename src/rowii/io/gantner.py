"""Reader for the Gantner 'UniversalDataBinFile' container used at Rodundwerk II.

Layout: version-prefixed magic string, JSON metadata, channel descriptor block
(name [unit] uuid per channel, each length-prefixed -- see `_valid_token_at`),
a run of 0x2a padding, then frames of uint64 ns-timestamp + one float32 per
channel. Verified against real files (Betriebsdaten `DeviceAppVersion` V2.17,
TU vibration streams V2.18) during Task 13: earlier revisions of this reader
had only ever been exercised against the synthetic test fixture
(`tests/fixtures/gantner_builder.py`), which -- until Task 13's fix -- encoded
a subtly different (and wrong) token length-prefix convention than the real
delivery; see `_valid_token_at`'s docstring for the corrected semantic.
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


def _valid_token_at(desc: bytes, i: int) -> tuple[str, int] | None:
    """If a length-prefixed token validates starting at *i*, return `(payload, end)` with
    `end` the index one past its NUL terminator; else `None`.

    Length semantic verified against the real June-2026 delivery (Task 13): the u16
    length prefix counts the payload bytes PLUS the terminating NUL itself (`length ==
    len(payload) + 1`), not the payload alone -- so payload is `desc[i+2 : i+2+length-1]`
    and the terminator is the single byte at `i+2+length-1`.
    """
    n = len(desc)
    if i + 2 > n:
        return None
    length = struct.unpack_from("<H", desc, i)[0]
    end = i + 2 + length
    if not (2 <= length <= _MAX_TOKEN_LEN + 1 and end <= n):
        return None
    payload = desc[i + 2 : end - 1]
    if all(0x20 <= b <= 0x7E for b in payload) and desc[end - 1] == 0x00:
        return payload.decode("ascii"), end
    return None


def _scan_tokens(desc: bytes) -> list[str]:
    """Validating scan-tokenizer over the channel-descriptor region.

    Every real token is length-prefixed: ``<u16 LE length L><L-1 printable-ASCII
    bytes>\\x00`` -- *L* counts the terminating NUL as part of the length (see
    `_valid_token_at`). A plain printable-ASCII regex cannot distinguish a token's own
    2-byte length prefix from real token bytes: whenever a token's length L falls in
    [32, 126], the prefix's low byte (L & 0xFF) is itself a printable ASCII character,
    and a regex scan mis-parses it as a spurious 1-character token, silently corrupting
    the following name/unit.

    This scanner is structural instead: at each byte offset it validates that a
    plausible length-prefixed token actually starts there (length in range, all payload
    bytes printable, NUL terminator present) before accepting it. Unknown filler bytes
    between tokens (observed as a small per-channel descriptor blob in the container's
    reverse-engineered binary layout) fail validation and are skipped one byte at a time.

    Maximal munch: the real per-channel record layout has a short run of fixed
    non-token bytes right before each token's true length prefix (Task 13 finding --
    e.g. `2b 00 02 00` before a UUID token's own `<len><uuid>\\x00`). Those bytes can,
    read from one byte earlier than the genuine token's start, coincidentally validate
    as a SHORT token themselves (short length, one printable payload byte, NUL
    terminator) -- e.g. a `length=2` token whose single payload byte happens to be the
    genuine token's own length-prefix low byte. Accepting such a short match would
    silently swallow the position where the real, longer token starts and drop it
    entirely. So whenever a candidate token validates at `i` with end `end`, this
    scanner first checks every later position `j` in `(i, end)` for a LONGER valid
    token (one whose own `end` extends past the candidate's `end`) and prefers that
    one instead of accepting the shorter match -- the ambiguity can only arise within
    the short candidate's own byte span, since a `length` field is itself only 2 bytes
    wide, so this lookahead never needs to extend past the current candidate's `end`.
    """
    tokens: list[str] = []
    i = 0
    n = len(desc)
    while i + 2 <= n:
        candidate = _valid_token_at(desc, i)
        if candidate is not None:
            _, cand_end = candidate
            better = None
            for j in range(i + 1, cand_end):
                alt = _valid_token_at(desc, j)
                if alt is not None and alt[1] > cand_end:
                    better = (j, alt)
                    break
            if better is None:
                tokens.append(candidate[0])
                i = cand_end
                continue
            j, (payload, end) = better
            tokens.append(payload)
            i = end
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
