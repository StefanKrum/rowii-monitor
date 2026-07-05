# Step 1 — Operating-State Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unsupervised operating-state detection on the Rodundwerk II June-25 recordings (TU + PU), validated against SCADA ground truth, reported per input variant (audio-handcrafted / audio-beats / vibration / fusion×2) × clusterer (KMeans/GMM).

**Architecture:** One Gantner-container reader feeds a per-run 1-s window grid; pluggable featurizers produce per-window vectors; KMeans/GMM → sticky HMM → duration filter produce segments; SCADA-derived rule labels ground the evaluation (ARI, Hungarian-matched F1, boundary deltas). Spec: `docs/superpowers/specs/2026-07-05-step1-state-detection-design.md`.

**Tech Stack:** Python 3.12, numpy/scipy/scikit-learn/hmmlearn/pandas/matplotlib/openpyxl/python-dotenv; optional extra `[beats]`: torch/torchaudio. pytest + ruff + mypy. Repo root = `repos/rowii-monitor`.

## Global Constraints

- Fresh code only — no code copied from `hydropower-anomaly` or `pshp-ssl-transfer` (format knowledge documented below is fair game).
- SCADA is never an input to detection — only `scada/` → GT labels for eval.
- BEATs: audio branch only, frozen, behind extra `[beats]`; NO further representations.
- No real data in unit tests; real-data tests behind `@pytest.mark.data`, skipped if `ROWII_DATA_ROOT` unset.
- `hmmlearn` GaussianHMM with `params="mc"` (never re-estimate the sticky transmat — proven lesson).
- Conventional commits, no AI attribution, line length 100 (ruff), mypy clean.
- All timestamps UTC; UDBF frame timestamps (ns since Unix epoch) are authoritative, filenames are local-time hints only.

## Gantner container format (reference for Task 2; verified on real files 2026-07-05)

```
uint16 BE version (0x006B = v1.07)
uint16 BE strlen (=43) + b"UniversalDataBinFile - GANTNER instruments"
... bytes until first '{' ...
JSON metadata object (balanced-brace scan; keys incl. SourceName, MeasName)
channel descriptor block: per channel, tokens appear in order
    name (printable ASCII, \x00-terminated, length-prefixed)
    [optional unit token, e.g. "m/s2", "MVar", "m3/s"]
    UUID string "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
padding: run of >= 8 bytes 0x2a
frames until EOF: uint64 LE timestamp_ns, then n_channels x float32 LE
```
Channel count is recovered by counting UUIDs in the descriptor region; the frame size
(8 + 4·C) must tile the body (partial tail frame allowed and dropped). Known channel
sets: `RAWGeneratorMic__0` → GenMic0/90/180/270 (~50 kHz); `RAWTurbineMic__1` →
TurMic* (~50 kHz, 5 ch incl. bottom); `RAW*Vib__{2,3}` → {Gen,Tur}Vib{0,180}{X,Y,Z}
(~10 kHz, m/s²; *Vib0* channels are dead); Betriebsdaten → ~30 named channels @ 10 Hz
incl. `1_P_Ist`, `1_Drehzahl_Ist`, `1_Leitapparat Stell.`, `Durchfluss TU`,
`Durchfluss PU`, `1_Q_Ist`.

## File Structure

```
pyproject.toml  README.md  .gitignore  .env.example
src/rowii/__init__.py
src/rowii/config.py                 # env + dataclasses (Task 1)
src/rowii/io/__init__.py
src/rowii/io/gantner.py             # container reader (Task 2)
src/rowii/io/dataset.py             # recording discovery (Task 3)
src/rowii/signals/__init__.py
src/rowii/signals/windows.py        # window grid + slicing (Task 4)
src/rowii/signals/features.py       # Featurizer protocol + handcrafted (Task 6)
src/rowii/signals/beats.py          # BeatsFeaturizer, extra [beats] (Task 14)
src/rowii/scada/__init__.py
src/rowii/scada/labels.py           # channels + GT rules (Task 5)
src/rowii/state/__init__.py
src/rowii/state/cluster.py          # KMeans/GMM (Task 7)
src/rowii/state/smooth.py           # sticky HMM (Task 8)
src/rowii/state/segments.py         # duration filter + export (Task 9)
src/rowii/state/detect.py           # orchestration (Task 10)
src/rowii/eval/__init__.py
src/rowii/eval/metrics.py           # ARI/F1/boundaries (Task 11)
src/rowii/eval/report.py            # report.md + timeline.png (Task 11)
scripts/copy_data.py                # selective copy (Task 12)
scripts/run_step1.py                # CLI (Task 12)
tests/fixtures/gantner_builder.py   # synthetic .dat writer for tests (Task 2)
tests/test_*.py                     # one per module
```

---

### Task 1: Repo scaffold + config + GitHub

**Files:** Create: `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`, `src/rowii/__init__.py`, `src/rowii/config.py`, `tests/test_config.py`

**Interfaces — Produces:** `rowii.config.load_config(env: Mapping[str, str] | None = None) -> Config`; `Config` frozen dataclass with fields `data_root: Path`, `results_root: Path`, `window: WindowConfig(window_s: float = 1.0)`, `gt: GtRules` (fields in Task 5), `detect: DetectConfig(n_states: int = 4, self_transition: float = 0.98, min_dwell_s: float = 5.0, random_seed: int = 7)`, `beats_checkpoint: Path | None`.

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "rowii-monitor"
version = "0.1.0"
description = "Acoustic + vibration condition monitoring for the Rodundwerk II pump-turbine (HSG master thesis implementation)"
requires-python = ">=3.12"
dependencies = [
  "numpy>=1.26", "scipy>=1.12", "scikit-learn>=1.4", "hmmlearn>=0.3.2",
  "pandas>=2.2", "matplotlib>=3.8", "openpyxl>=3.1", "python-dotenv>=1.0",
  "pyarrow>=15",
]

[project.optional-dependencies]
beats = ["torch>=2.2", "torchaudio>=2.2"]
dev = ["pytest>=8", "ruff>=0.4", "mypy>=1.10"]

[tool.setuptools]
package-dir = {"" = "src"}
[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["data: needs ROWII_DATA_ROOT with real recordings"]
```

- [ ] **Step 2: Write .gitignore (`data/`, `results/`, `.env`, `__pycache__/`, `*.egg-info`, `.venv/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `.DS_Store`), `.env.example` (`ROWII_DATA_ROOT=~/AI Workspace/master-thesis/data/illwerke-250526`, `ROWII_RESULTS_ROOT=./results`, `ROWII_BEATS_CHECKPOINT=`), README.md (project purpose, install `pip install -e ".[dev]"`, data layout per spec §3, quickstart commands from Task 12).**

- [ ] **Step 3: Write the failing config test**

```python
# tests/test_config.py
from pathlib import Path
from rowii.config import load_config

def test_defaults_without_env() -> None:
    cfg = load_config(env={})
    assert cfg.window.window_s == 1.0
    assert cfg.detect.n_states == 4
    assert cfg.detect.self_transition == 0.98
    assert cfg.beats_checkpoint is None

def test_env_overrides() -> None:
    cfg = load_config(env={"ROWII_DATA_ROOT": "/tmp/x", "ROWII_BEATS_CHECKPOINT": "/tmp/b.pt"})
    assert cfg.data_root == Path("/tmp/x")
    assert cfg.beats_checkpoint == Path("/tmp/b.pt")
```

- [ ] **Step 4: Run `pytest tests/test_config.py -v` → FAIL (module missing)**
- [ ] **Step 5: Implement `src/rowii/config.py`**

```python
"""Central configuration: environment-driven paths + tunable parameter dataclasses."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class WindowConfig:
    window_s: float = 1.0


@dataclass(frozen=True)
class GtRules:
    speed_nominal_rpm: float = 375.0       # 8-pole 50 Hz machine: 375 rpm hypothesis;
    speed_eps_frac: float = 0.05           # validated against data in Task 5/13
    power_eps_mw: float = 2.0
    ramp_mw_per_s: float = 1.0
    transition_buffer_s: float = 10.0
    n_load_bins: int = 3


@dataclass(frozen=True)
class DetectConfig:
    n_states: int = 4
    self_transition: float = 0.98
    min_dwell_s: float = 5.0
    random_seed: int = 7


@dataclass(frozen=True)
class Config:
    data_root: Path
    results_root: Path
    window: WindowConfig = field(default_factory=WindowConfig)
    gt: GtRules = field(default_factory=GtRules)
    detect: DetectConfig = field(default_factory=DetectConfig)
    beats_checkpoint: Path | None = None


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Build a Config from (in order) explicit *env*, process env, and .env file."""
    file_env = {k: v for k, v in dotenv_values(".env").items() if v is not None}
    merged: dict[str, str] = {**file_env, **os.environ}
    if env is not None:
        merged = dict(env)
    ckpt = merged.get("ROWII_BEATS_CHECKPOINT") or None
    return Config(
        data_root=Path(merged.get("ROWII_DATA_ROOT", "data/illwerke-250526")).expanduser(),
        results_root=Path(merged.get("ROWII_RESULTS_ROOT", "results")).expanduser(),
        beats_checkpoint=Path(ckpt).expanduser() if ckpt else None,
    )
```

- [ ] **Step 6: `pip install -e ".[dev]"`; `pytest tests/test_config.py -v` → PASS; `ruff check . && mypy src` → clean**
- [ ] **Step 7: Commit `chore: scaffold rowii-monitor (pyproject, config, tooling)`**
- [ ] **Step 8: Create private GitHub repo and push: `gh repo create StefanKrum/rowii-monitor --private --source . --push`. Verify `gh repo view StefanKrum/rowii-monitor --json name` succeeds.**

### Task 2: Gantner container reader

**Files:** Create: `src/rowii/io/__init__.py`, `src/rowii/io/gantner.py`, `tests/fixtures/__init__.py`, `tests/fixtures/gantner_builder.py`, `tests/test_gantner.py`

**Interfaces — Produces:**
```python
class GantnerFormatError(Exception): ...
@dataclass(frozen=True)
class GantnerHeader:
    source_name: str            # JSON "SourceName", e.g. "RAWGeneratorMic"
    channel_names: list[str]
    channel_units: list[str]    # "" where absent
    t0_ns: int                  # first frame timestamp (UTC ns)
    sample_rate_hz: float       # median of 1e9/diff(timestamps)
    n_frames: int
@dataclass(frozen=True)
class GantnerFile:
    header: GantnerHeader
    timestamps_ns: np.ndarray   # shape (T,), uint64
    data: np.ndarray            # shape (T, C), float32
def read_gantner(path: Path) -> GantnerFile
def read_header(path: Path) -> GantnerHeader   # cheap: header + first/last frames only
```

- [ ] **Step 1: Write the synthetic builder fixture (test-only writer mirroring the format block above)**

```python
# tests/fixtures/gantner_builder.py
"""Build synthetic Gantner-container .dat files for tests (format doc: plan header)."""
from __future__ import annotations

import json
import struct
import uuid
from pathlib import Path

import numpy as np

MAGIC = b"UniversalDataBinFile - GANTNER instruments"


def build_gantner_file(
    path: Path,
    channel_names: list[str],
    data: np.ndarray,               # (T, C) float32
    t0_ns: int = 1_750_000_000_000_000_000,
    rate_hz: float = 100.0,
    units: list[str] | None = None,
    corrupt_padding: bool = False,
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
```

- [ ] **Step 2: Write failing tests (round-trip, header fields, dead-channel passthrough, format errors)**

```python
# tests/test_gantner.py
import numpy as np
import pytest
from rowii.io.gantner import GantnerFormatError, read_gantner, read_header
from tests.fixtures.gantner_builder import build_gantner_file

def test_roundtrip_reads_names_units_rate_and_data(tmp_path) -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(500, 3)).astype(np.float32)
    p = build_gantner_file(tmp_path / "t.dat", ["ChA", "ChB", "ChC"], data,
                           rate_hz=100.0, units=["Pa", "", "m/s2"])
    f = read_gantner(p)
    assert f.header.channel_names == ["ChA", "ChB", "ChC"]
    assert f.header.channel_units == ["Pa", "", "m/s2"]
    assert f.header.n_frames == 500
    assert abs(f.header.sample_rate_hz - 100.0) < 0.5
    np.testing.assert_allclose(f.data, data, rtol=1e-6)
    assert f.timestamps_ns[0] == f.header.t0_ns

def test_partial_tail_frame_is_dropped(tmp_path) -> None:
    data = np.zeros((10, 2), dtype=np.float32)
    p = build_gantner_file(tmp_path / "t.dat", ["A", "B"], data)
    raw = p.read_bytes()
    p.write_bytes(raw[:-5])                    # cut into the last frame
    assert read_gantner(p).header.n_frames == 9

def test_missing_magic_raises(tmp_path) -> None:
    p = tmp_path / "bad.dat"
    p.write_bytes(b"\x00\x00not a gantner file" * 10)
    with pytest.raises(GantnerFormatError, match="magic"):
        read_gantner(p)

def test_missing_padding_raises(tmp_path) -> None:
    data = np.zeros((5, 1), dtype=np.float32)
    p = build_gantner_file(tmp_path / "t.dat", ["A"], data, corrupt_padding=True)
    with pytest.raises(GantnerFormatError, match="padding"):
        read_gantner(p)

def test_read_header_is_cheap_and_consistent(tmp_path) -> None:
    data = np.zeros((1000, 2), dtype=np.float32)
    p = build_gantner_file(tmp_path / "t.dat", ["A", "B"], data, rate_hz=50.0)
    h = read_header(p)
    assert h.n_frames == 1000 and abs(h.sample_rate_hz - 50.0) < 0.5
```

- [ ] **Step 3: Run → FAIL (module missing)**
- [ ] **Step 4: Implement `src/rowii/io/gantner.py`**

```python
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
_NAME_RE = re.compile(rb"([ -~]{2,60})\x00")
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
        tokens = [t.group(1).decode("ascii") for t in _NAME_RE.finditer(desc[cursor : m.start()])]
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
```

- [ ] **Step 5: `pytest tests/test_gantner.py -v` → PASS; lint/type clean**
- [ ] **Step 6: Commit `feat: Gantner container reader with synthetic-fixture tests`**

### Task 3: Recording discovery

**Files:** Create: `src/rowii/io/dataset.py`, `tests/test_dataset.py`

**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class BurstFile:
    path: Path; stream: str          # "RAWGeneratorMic__0" | "RAWTurbineMic__1" | "RAWGeneratorVib__2" | "RAWTurbineVib__3"
    start_utc_hint: datetime         # parsed from filename (local CEST -> UTC)
@dataclass(frozen=True)
class Run:
    name: str                        # "tu" | "pu-morning" | "pu-afternoon"
    files: dict[str, list[BurstFile]]  # stream -> time-sorted files
@dataclass(frozen=True)
class RecordingIndex:
    runs: list[Run]
    betriebsdaten: list[Path]        # time-sorted, overlap-reconciled
def discover(data_root: Path) -> RecordingIndex
```
Grouping rule: files of one folder (TU/, PU/) sorted by filename time; a gap > 15 min between consecutive files of the same stream splits a folder into separate runs (PU morning vs afternoon). Filename pattern: `<stream>_YYYY-MM-DD_HH-MM-SS_ffffff.dat`, local time Europe/Vienna → UTC via `zoneinfo`. Betriebsdaten: pattern `YYYY-MM-DD_HH-00-00.dat`; duplicates for the same hour resolved by preferring the larger file (MANIFEST overlap caveat), dropped ones logged via `logging.warning`.

- [ ] **Step 1: Failing tests: build a fake `data_root` tree with empty files following the real naming (TU 2 segments × 2 streams; PU morning 09:08/09:20 + afternoon 13:56 → expect 3 runs with correct split; two Betriebsdaten files for the same hour with different sizes → larger wins).**
- [ ] **Step 2: Run → FAIL. Step 3: Implement (pure `pathlib` + `re` + `zoneinfo`; no file reads — discovery works on names alone so it stays fast and testable with empty files).**
- [ ] **Step 4: Tests PASS; lint/type clean. Step 5: Commit `feat: recording discovery (runs, streams, betriebsdaten reconciliation)`.**

### Task 4: Window grid + slicing

**Files:** Create: `src/rowii/signals/__init__.py`, `src/rowii/signals/windows.py`, `tests/test_windows.py`

**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class WindowGrid:
    t0_ns: int; window_ns: int; n_windows: int
    def edges_ns(self) -> np.ndarray            # (n_windows+1,)
def common_grid(headers: Sequence[GantnerHeader], window_s: float) -> WindowGrid
    # spans the INTERSECTION of [t0, t_end] across streams, aligned to whole windows
def window_slices(ts_ns: np.ndarray, grid: WindowGrid) -> list[slice]
    # per window the sample slice; empty slice where a stream has a gap
def coverage(ts_ns: np.ndarray, grid: WindowGrid, rate_hz: float) -> np.ndarray
    # fraction of expected samples present per window (0..1)
```
Rule from spec §6: windows with coverage < 0.8 in any used stream are dropped from features; if > 5 % of a run's windows drop → raise `RuntimeError`.

- [ ] **Step 1: Failing tests: two synthetic headers with offset t0 → grid covers intersection; slices reconstruct correct sample counts at 100 Hz; a gap in ts produces coverage < 1 for exactly the gapped windows.**
- [ ] **Step 2..5: Red → implement → green → lint → commit `feat: UTC window grid with per-stream slicing and coverage`.**

### Task 5: SCADA ground truth

**Files:** Create: `src/rowii/scada/__init__.py`, `src/rowii/scada/labels.py`, `tests/test_gt_labels.py`

**Interfaces — Produces:**
```python
GT_CHANNELS = {"power": "1_P_Ist", "speed": "1_Drehzahl_Ist",
               "guide_vane": "1_Leitapparat Stell.",
               "flow_tu": "Durchfluss TU", "flow_pu": "Durchfluss PU"}
STATES = ("standstill", "turbine", "pump", "transition")
def load_scada_window_means(files: list[Path], grid: WindowGrid,
                            channels: Mapping[str, str] = GT_CHANNELS) -> pd.DataFrame
    # index = window id; columns = channel keys; NaN where no coverage
def gt_labels(scada: pd.DataFrame, rules: GtRules) -> pd.DataFrame
    # columns: state (str, "unknown" where NaN), load_bin (int, -1 outside turbine/pump)
```
Rules (exact, all thresholds from `GtRules`): let `P` = power MW (sign per plant convention; if pump power is logged positive, `flow_pu > flow_eps` forces pump), `n` = speed rpm.
`standstill` ⇔ |n| < 0.05·n_nom and |P| < P_eps; `turbine` ⇔ n ≥ 0.95·n_nom and P > +P_eps (or `flow_tu` dominant); `pump` ⇔ n ≥ 0.95·n_nom and (P < −P_eps or `flow_pu` dominant); everything else, plus any window within `transition_buffer_s` of a state change, plus |dP/dt| > ramp threshold → `transition`. `load_bin` = quantile bin of P within turbine (resp. pump) windows, `n_load_bins` bins.

- [ ] **Step 1: Failing tests: constructed SCADA frame walking standstill → ramp → turbine plateaus (two load levels) → ramp → standstill; assert exact expected label sequence incl. buffer windows and load bins; pump case via negative P and via flow_pu; NaN → "unknown".**
- [ ] **Step 2..5: Red → implement → green → commit `feat: SCADA window means + rule-based ground-truth labels`.**

### Task 6: Handcrafted featurizers

**Files:** Create: `src/rowii/signals/features.py`, `tests/test_features.py`

**Interfaces — Produces:**
```python
class Featurizer(Protocol):
    name: str
    def feature_names(self) -> list[str]: ...
    def transform(self, windows: np.ndarray, rate_hz: float) -> np.ndarray
        # windows: (W, S, C) float32  ->  (W, F) float64
MACHINE_HZ = {"shaft": 6.25, "blade_pass": 43.75, "guide_vane_pass": 125.0}
class AudioFeaturizer:   # per channel: log-RMS; band energy at each MACHINE_HZ ± 10 %;
                         # log-energy in octave bands 31.5..8000 Hz; spectral centroid + 95 % rolloff
class VibFeaturizer:     # per live channel: log-RMS, kurtosis, MACHINE_HZ band energies,
                         # octave bands up to 4 kHz; dead channels (std < 1e-9) dropped with warning
def zscore(x: np.ndarray) -> np.ndarray            # per-feature, guards zero std
def fuse(a: np.ndarray, b: np.ndarray) -> np.ndarray  # z-scored concat
```
Spectra via `scipy.signal.welch` (nperseg=4096 at 50 kHz, 2048 at 10 kHz), band energy = mean PSD in band, log10 with 1e-12 floor.

- [ ] **Step 1: Failing tests: a synthetic 44 Hz sine at 50 kHz → blade-pass band energy ≫ neighbour bands; white noise vs sine → centroid ordering; dead-channel drop emits warning and shrinks feature count; zscore returns zero-mean unit-std and survives constant columns; fuse concatenates shapes.**
- [ ] **Step 2..5: Red → implement → green → commit `feat: handcrafted audio/vibration featurizers with machine-frequency bands`.**

### Task 7: Clusterers

**Files:** Create: `src/rowii/state/__init__.py`, `src/rowii/state/cluster.py`, `tests/test_cluster.py`

**Interfaces — Produces:**
```python
class KMeansClusterer:
    def __init__(self, n_clusters: int, random_seed: int) -> None: ...
    def fit_predict(self, x: np.ndarray) -> np.ndarray   # (W,F) -> (W,) int labels
class GmmClusterer:  # same signature, full covariance
```

- [ ] **Steps: failing test on 3 separated Gaussian blobs (both clusterers recover 3 groups, ARI == 1.0 vs construction); implement thin wrappers around sklearn with fixed seeds; green; commit `feat: kmeans/gmm clusterers`.**

### Task 8: Sticky HMM smoother

**Files:** Create: `src/rowii/state/smooth.py`, `tests/test_smooth.py`

**Interfaces — Produces:**
```python
class StickyHmmSmoother:
    def __init__(self, self_transition: float = 0.98, random_seed: int = 7) -> None: ...
    def fit_decode(self, features: np.ndarray, init_labels: np.ndarray) -> np.ndarray
        # GaussianHMM(n_components=k, covariance_type="diag", params="mc", init_params="")
        # transmat fixed sticky; startprob uniform; means/covs init from init_labels groups;
        # returns Viterbi state sequence aligned to init label ids
```

- [ ] **Step 1: Failing test: 3-state synthetic features where init_labels contain injected 1-frame flips → decoded sequence removes ≥ 90 % of flips and preserves true boundaries within ±1 frame; a second test asserts `model.transmat_` unchanged after fit (params="mc" honoured).**
- [ ] **Step 2..5: Red → implement → green → commit `feat: sticky HMM smoothing (fixed transmat, params=mc)`.**

### Task 9: Duration filter + segments

**Files:** Create: `src/rowii/state/segments.py`, `tests/test_segments.py`

**Interfaces — Produces:**
```python
def duration_filter(labels: np.ndarray, min_dwell: int) -> np.ndarray
    # merges runs shorter than min_dwell into the neighbouring run (longer side wins)
def to_segments(labels: np.ndarray, grid: WindowGrid) -> pd.DataFrame
    # columns: start_utc, end_utc, duration_s, cluster
```

- [ ] **Steps: failing tests (flicker run of 3 merges into surrounding state; alternating pattern resolves deterministically; segment table timestamps match grid edges); implement; green; commit `feat: duration filter and segment export`.**

### Task 10: Detection orchestration + synthetic end-to-end

**Files:** Create: `src/rowii/state/detect.py`, `tests/test_detect_e2e.py`

**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class DetectionResult:
    frame_labels: np.ndarray; segments: pd.DataFrame; k: int
def run_detection(features: np.ndarray, grid: WindowGrid, cfg: DetectConfig,
                  clusterer: Literal["kmeans", "gmm"] = "kmeans") -> DetectionResult
```
Chain: zscore → clusterer(k=cfg.n_states) → StickyHmmSmoother → duration_filter(min_dwell = min_dwell_s / window_s) → to_segments.

- [ ] **Step 1: Failing e2e test: synthetic 3-state feature stream (600 windows, distinct means, 5 % label noise pre-injected through weak cluster separation) → `adjusted_rand_score(truth, result.frame_labels) == 1.0` after smoothing and segments count == 3.**
- [ ] **Step 2..5: Red → implement → green → commit `feat: detection pipeline orchestration with synthetic e2e test`.**

### Task 11: Evaluation metrics + report

**Files:** Create: `src/rowii/eval/__init__.py`, `src/rowii/eval/metrics.py`, `src/rowii/eval/report.py`, `tests/test_metrics.py`

**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class EvalResult:
    ari: float; macro_f1: float; confusion: pd.DataFrame
    boundary_median_abs_s: float | None; n_eval_windows: int; mapping: dict[int, str]
def evaluate(pred: np.ndarray, gt: pd.DataFrame, grid: WindowGrid) -> EvalResult
    # excludes gt.state == "unknown"; Hungarian matching (scipy.optimize.linear_sum_assignment
    # on the contingency table) maps cluster ids -> GT states before F1/confusion;
    # boundary metric: for each GT state change, |Δt| to nearest predicted change
def write_report(out_dir: Path, run: str, variant: str, det: DetectionResult,
                 ev: EvalResult, scada: pd.DataFrame) -> None
    # report.md (metrics table, mapping, dropped/unknown counts) +
    # timeline.png (3 stacked panels: power curve, GT states, predicted states) +
    # segments.csv + frame_labels.parquet
```

- [ ] **Steps: failing tests (perfect prediction → ari=1, f1=1, boundary 0; permuted cluster ids → same scores via Hungarian; unknown windows excluded from counts); implement; green; commit `feat: evaluation metrics (ARI, Hungarian F1, boundaries) and run report`.**

### Task 12: Data copy script + CLI

**Files:** Create: `scripts/copy_data.py`, `scripts/run_step1.py`, `tests/test_cli_smoke.py`

**copy_data.py:** argparse `--source` (default `~/Downloads/illwerke-250526-analysis`) `--dest` (default from config) `--dry-run`. Copies exactly: `20260626 Messung/TU/*.dat`, `20260626 Messung/PU/*.dat`, `20260626 Messung/Betriebsdaten/2026-06-25_*.dat`, `Sensor_Anordnung_15062026.xlsx`, `MANIFEST.md`, `ROWII_Leistung_*.jpg`, `20260626 Messung/ROWII_Leistung.jpg`. Skips files already present with equal size; writes `copy_manifest.json` (per file: relpath, bytes); prints total copied/skipped; refuses if free disk < 1.2 × required (`shutil.disk_usage`).

**run_step1.py:** argparse `--run {tu,pu-morning,pu-afternoon,all}` `--variant {audio,audio-beats,vibration,fusion,fusion-beats,all}` `--clusterer {kmeans,gmm,all}` `--k INT` (default cfg.n_states). Flow per (run, variant, clusterer): discover → grid over the run's used streams → featurize (audio streams: both mic files' channels; vibration: both vib streams' live channels; beats variants import `rowii.signals.beats` lazily with an actionable ImportError pointing to `pip install -e ".[beats]"`) → run_detection → load SCADA GT (skip eval with a logged notice when no Betriebsdaten coverage, e.g. pu-afternoon) → evaluate → write_report to `results/<run>/<variant>-<clusterer>/`. Ends by writing `results/summary.csv` (one row per executed combination).

- [ ] **Steps: failing smoke test (invoke `copy_data.py --dry-run` on a fake tree → correct file list printed, nothing copied; `run_step1.py --help` exits 0); implement both scripts; green; commit `feat: selective data copy and step-1 CLI`.**

### Task 13: Real-data smoke + first TU results  *(checkpoint with Stefan)*

**Files:** Create: `tests/test_real_data.py`. Modify: `README.md` (results section stub).

- [ ] **Step 1: `@pytest.mark.data` test: discover real `ROWII_DATA_ROOT` → expect run "tu" with 4 streams × 12 files; `read_header` of one mic file reports ~50 kHz and 4+ channels; one Betriebsdaten file contains all `GT_CHANNELS` names.**
- [ ] **Step 2: Run `python scripts/copy_data.py` (real copy ~35 GB), then `pytest -m data -v` → PASS.**
- [ ] **Step 3: `python scripts/run_step1.py --run tu --variant audio --clusterer kmeans` then `--variant fusion`; inspect `results/tu/*/report.md` + timeline.png; sanity: detected standstill/turbine phases match the deck's power timeline; record ARI in README results table.**
- [ ] **Step 4: Fix whatever reality breaks (channel-name mismatches, sign conventions, speed nominal) in config defaults — each fix with its own test + commit.**
- [ ] **Step 5: Commit `feat: real-data smoke tests + first TU results` and STOP for review with Stefan (acceptance gate: fusion ARI ≥ 0.9 on TU).**

### Task 14: BeatsFeaturizer (extra `[beats]`)

**Files:** Create: `src/rowii/signals/beats.py`, `tests/test_beats.py`

**Interfaces — Produces:**
```python
def best_device() -> "torch.device"     # ROWII_FORCE_CPU env > mps > cuda > cpu
class BeatsFeaturizer:
    name = "beats"
    def __init__(self, checkpoint: Path, device: "torch.device | None" = None,
                 encoder: "BeatsEncoderProtocol | None" = None) -> None: ...
    def feature_names(self) -> list[str]        # ["beats_0", ...]
    def transform(self, windows: np.ndarray, rate_hz: float) -> np.ndarray
        # mono-mix channels -> resample to 16 kHz (torchaudio) -> 128-mel fbank
        # (torchaudio.compliance.kaldi.fbank, 25 ms / 10 ms, BEATs normalisation)
        # -> frozen encoder -> mean-pool tokens -> (W, D) float64
class BeatsEncoderProtocol(Protocol):
    def extract(self, fbank: "torch.Tensor") -> "torch.Tensor": ...
```
Checkpoint loading targets the official BEATs iter3 checkpoint file (path from `Config.beats_checkpoint`); the encoder wrapper is written fresh against the published BEATs module API. Unit tests use a stub encoder injected via the `encoder` parameter (returns deterministic tensors), so CI needs no torch weights; a `@pytest.mark.data` test runs the real checkpoint on 10 s of synthetic audio and asserts output shape (W, 768) and finiteness.

- [ ] **Steps: failing stub-based tests (shape, mono-mix of multi-channel input, device selection honours ROWII_FORCE_CPU); implement; green; extend `run_step1.py` variants audio-beats/fusion-beats (already wired in Task 12 CLI enum — implement the lazy import body now); run full grid `--run all --variant all --clusterer all`; update README results table; commit `feat: frozen-BEATs featurizer behind [beats] extra + full step-1 grid`.**

---

## Self-Review (plan vs spec)

- Spec §3 copy plan → Task 12; §4 modules → Tasks 1–11 + 14 (beats.py added under signals/); §5 grid → Tasks 12–14; §6 error handling → Tasks 2 (format errors), 3 (overlaps), 4 (coverage/5 % rule), 6 (dead channels), 12 (pu-afternoon GT skip); §7 testing → every task carries tests, real-data tier in Task 13; §8 deliverables/acceptance → Tasks 11–13 (ARI ≥ 0.9 gate in Task 13), GitHub push in Task 1.
- Type consistency: `GantnerHeader` consumed by Tasks 3/4/13; `WindowGrid` by 5/9/10/11; `Featurizer.transform(windows,(W,S,C))` by 6/14; `DetectionResult` by 11/12. Names match.
- No placeholders: every code step carries real code or exact content; Tasks 3–12 tests described with concrete cases and exact assertions where short, full code where the logic is subtle (reader, config).
