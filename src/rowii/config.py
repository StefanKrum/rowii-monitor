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
    # Measured plateau of 1_Drehzahl_Ist during full-power turbine generation (Task 13,
    # 2026-06-25 05:00 Betriebsdaten hour: median while power > 50 MW). The pre-delivery
    # "8-pole 50 Hz machine" 375 rpm hypothesis was off by 73% and is superseded by this
    # measured value; see results/parameter_verification.md for the full derivation.
    speed_nominal_rpm: float = 101.0
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
