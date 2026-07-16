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
    # Measured plateau of "1_Drehzahl UPM" (GT_CHANNELS["speed"]) during full-power
    # turbine generation (Task 13b, 2026-06-25 05:00 Betriebsdaten hour: median while
    # power > 50 MW). Task 13 originally measured this off "1_Drehzahl_Ist" instead
    # (~101 rpm) -- that channel is NOT rpm (a percent-of-nominal-ish quantity, ~3.75x
    # smaller than the true rpm channel on the same file); Task 13b corrected
    # GT_CHANNELS["speed"] to the genuine rpm channel and remeasured, landing almost
    # exactly on the pre-delivery "8-pole 50 Hz machine" 375 rpm hypothesis Task 13 had
    # discarded. See results/parameter_verification.md's Revision 2026-07-07 section
    # for the full derivation.
    speed_nominal_rpm: float = 378.832
    speed_eps_frac: float = 0.05           # validated against data in Task 5/13
    power_eps_mw: float = 2.0
    ramp_mw_per_s: float = 1.0
    transition_buffer_s: float = 10.0
    n_load_bins: int = 3
    # Phase-shifter GT (multi-day/phase-shifter addendum, spec §3): a contiguous run
    # of nominal-speed, near-zero-power windows must dwell at least this long before
    # being promoted from "transition" (unloaded spinning during a start/stop ramp) to
    # "phase-shifter" (a genuine, sustained synchronous-condenser operating mode).
    # 600s (10 min) is the addendum's own conservative default -- shorter unloaded-
    # spinning runs are ramp artifacts, not phase-shifter operation.
    ph_min_dwell_s: float = 600.0
    # DEDICATED power threshold for the PH candidate check (deliberately separate from
    # power_eps_mw, which is calibrated for standstill/loaded turbine-pump
    # discrimination): measured directly from ~98 min of confirmed 2026-07-01
    # phase-shifter operation (TU_PH_TU day, results/parameter_verification.md's
    # "Phase-shifter channels, 2026-07-08" section) -- active power sits at a stable
    # ~-3.5 MW median (range -4.3 to -2.9 MW, real motoring/idling losses, not noise),
    # well outside power_eps_mw=2.0. 5.0 MW is a round threshold comfortably above the
    # measured 4.34 MW max magnitude. Steady turbine operation sits far above this band,
    # but brief ramp transients can dip low or negative (observed min −41 MW on 2026-07-01);
    # such transients are excluded from PH promotion by the 600 s dwell requirement
    # (ph_min_dwell_s). Reusing the general power_eps_mw here would have silently made PH
    # promotion unreachable on real data (the spec's own literal text ties the PH rule to
    # power_eps_mw; this dedicated field is a deliberate, documented deviation to make
    # the rule actually fire on real data without loosening standstill/turbine/pump
    # discrimination elsewhere).
    ph_power_eps_mw: float = 5.0
    # Optional conjunctive gate: "1_KS Stellung" (spherical inlet valve position,
    # GT_CHANNELS["ks_valve"]) must also read <= ks_closed_max for a candidate run to
    # be promoted. VERIFIED and ENABLED (2026-07-08, scripts/verify_parameters.py's
    # "Phase-shifter channels" section, run against the 2026-07-01 delivery,
    # results/parameter_verification.md): per-GT-state ks_valve distribution (computed
    # WITHOUT this gate, i.e. purely from the speed+power+dwell rule) shows a clean,
    # wide separation -- phase-shifter median 3.208 (p95 3.213, n=6148) and standstill
    # median 3.015 (p95 3.112, n=36938) vs. turbine median 104.278 (p5 104.274,
    # n=35719) and pump median 104.277 (p5 104.276, n=4309). Provenance: the ~3=closed/
    # ~104=open hypothesis itself originates from the partner's (Bruno's) SCADA channel
    # audit, relayed pre-verification -- this confirmation is independently derived
    # from OUR OWN 2026-07-01 data, not carried over from that audit's numbers.
    ph_requires_ks_closed: bool = True
    ks_closed_max: float = 10.0
    """Threshold on GT_CHANNELS["ks_valve"] ("1_KS Stellung") below which the valve
    counts as closed, used only when `ph_requires_ks_closed` is True. 10.0 sits
    comfortably above the measured closed-state p95 (phase-shifter 3.213, standstill
    3.112) with a wide margin below the measured open-state p5 (~104.27) -- see
    `ph_requires_ks_closed`'s docstring for the full measurement provenance."""


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
    tfc_audio_checkpoint: Path | None = None
    """Frozen TF-C audio-branch checkpoint (`ROWII_TFC_AUDIO_CHECKPOINT`, package-4
    spec D4) -- mirrors `beats_checkpoint`'s own env-driven pattern exactly, but as
    two independent fields (`tfc_audio_checkpoint`/`tfc_vib_checkpoint`) rather than
    one: unlike BEATs (audio-branch only), TF-C is pre-trained separately per branch
    (MIMII for audio, CWRU/Paderborn bearing vibration for vibration), so the two
    checkpoints are unrelated files and must be settable independently. Consumed by
    `rowii.pipeline._featurizer_for_stream`'s `"audio-tfc"` dispatch."""
    tfc_vib_checkpoint: Path | None = None
    """Frozen TF-C vibration-branch checkpoint (`ROWII_TFC_VIB_CHECKPOINT`) -- see
    `tfc_audio_checkpoint`'s docstring. Consumed by `rowii.pipeline.
    _featurizer_for_stream`'s `"vibration-tfc"` dispatch."""


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Build a Config from (in order) explicit *env*, process env, and .env file."""
    file_env = {k: v for k, v in dotenv_values(".env").items() if v is not None}
    merged: dict[str, str] = {**file_env, **os.environ}
    if env is not None:
        merged = dict(env)
    ckpt = merged.get("ROWII_BEATS_CHECKPOINT") or None
    tfc_audio_ckpt = merged.get("ROWII_TFC_AUDIO_CHECKPOINT") or None
    tfc_vib_ckpt = merged.get("ROWII_TFC_VIB_CHECKPOINT") or None
    return Config(
        data_root=Path(merged.get("ROWII_DATA_ROOT", "data")).expanduser(),
        results_root=Path(merged.get("ROWII_RESULTS_ROOT", "results")).expanduser(),
        beats_checkpoint=Path(ckpt).expanduser() if ckpt else None,
        tfc_audio_checkpoint=Path(tfc_audio_ckpt).expanduser() if tfc_audio_ckpt else None,
        tfc_vib_checkpoint=Path(tfc_vib_ckpt).expanduser() if tfc_vib_ckpt else None,
    )
