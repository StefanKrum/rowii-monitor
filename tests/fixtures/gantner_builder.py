"""Build synthetic Gantner-container .dat files for tests (format doc: plan header)."""
from __future__ import annotations

import json
import struct
import uuid
from pathlib import Path

import numpy as np

MAGIC = b"UniversalDataBinFile - GANTNER instruments"


_PAD_BYTE = 0x2A  # '*' — the container's own padding-run sentinel (see gantner._parse_descriptor)


def _filler(n: int, salt: int) -> bytes:
    """Seeded pseudo-random non-printable filler bytes for exercising a scan-tokenizer's
    reject-and-advance path. Values come from a simple congruential formula, then any byte
    that would land on the padding sentinel (0x2a) is nudged aside so a filler run can never
    be mistaken for the real padding run the reader searches for.
    """
    out = bytearray()
    for i in range(n):
        v = (7 * (i + salt) + 3) % 256
        if v == _PAD_BYTE:
            v = (v + 1) % 256
        out.append(v)
    return bytes(out)


def build_gantner_file(
    path: Path,
    channel_names: list[str],
    data: np.ndarray,               # (T, C) float32
    t0_ns: int = 1_750_000_000_000_000_000,
    rate_hz: float = 100.0,
    units: list[str] | None = None,
    corrupt_padding: bool = False,
    *,
    filler_bytes: int = 0,
) -> Path:
    t = np.arange(data.shape[0], dtype=np.uint64) * np.uint64(int(1e9 / rate_hz)) + np.uint64(t0_ns)
    buf = bytearray()
    buf += struct.pack(">H", 0x006B) + struct.pack(">H", len(MAGIC)) + MAGIC
    meta = {"_id": "0.309", "SourceName": "TestStream", "MeasName": "unit-test"}
    buf += json.dumps(meta).encode()
    for i, name in enumerate(channel_names):
        buf += struct.pack("<H", len(name)) + name.encode() + b"\x00"
        unit = (units or [""] * len(channel_names))[i]
        if unit:
            buf += struct.pack("<H", len(unit)) + unit.encode() + b"\x00"
        if filler_bytes > 0:
            buf += _filler(filler_bytes, salt=i)
        u = str(uuid.uuid4()).encode()
        buf += struct.pack("<H", len(u)) + u + b"\x00"
    buf += b"\x2a" * (23 if not corrupt_padding else 2)
    frames = bytearray()
    for k in range(data.shape[0]):
        frames += struct.pack("<Q", int(t[k]))
        frames += data[k].astype("<f4").tobytes()
    buf += frames
    path.write_bytes(bytes(buf))
    return path
