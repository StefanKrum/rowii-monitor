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
    # turbine generation (2026-06-25 05:00 Betriebsdaten hour: median while
    # power > 50 MW). This channel was originally measured off "1_Drehzahl_Ist" instead
    # (~101 rpm) -- that channel is NOT rpm (a percent-of-nominal-ish quantity, ~3.75x
    # smaller than the true rpm channel on the same file); GT_CHANNELS["speed"] was
    # corrected to the genuine rpm channel and remeasured, landing almost
    # exactly on the pre-delivery "8-pole 50 Hz machine" 375 rpm hypothesis that had
    # previously been discarded. See results/parameter_verification.md's Revision 2026-07-07 section
    # for the full derivation.
    speed_nominal_rpm: float = 378.832
    speed_eps_frac: float = 0.05           # validated against real data
    power_eps_mw: float = 2.0
    ramp_mw_per_s: float = 1.0
    transition_buffer_s: float = 10.0
    n_load_bins: int = 3
    # Phase-shifter GT (multi-day/phase-shifter addendum): a contiguous run
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
    max_invalid_fraction: float = 0.05
    """Hard-fail ceiling for `rowii.pipeline.compute_validity_mask`'s invalid-window
    fraction (`ROWII_MAX_INVALID_FRACTION`). 0.05 is the spec rule; raise it only
    per-invocation for a delivery with documented stream gaps (e.g. `300626-tu`,
    two partner-side 12-min single-chunk gaps -- see that delivery's MANIFEST.md).
    Invalid windows stay masked out of every downstream stage regardless of this
    ceiling; deliberately NOT part of `_cache_fingerprint`'s payload (it changes
    which runs are accepted, never what a featurizer computes)."""
    beats_checkpoint: Path | None = None
    tfc_audio_checkpoint: Path | None = None
    """Frozen TF-C audio-branch checkpoint (`ROWII_TFC_AUDIO_CHECKPOINT`)
    -- mirrors `beats_checkpoint`'s own env-driven pattern exactly, but as
    two independent fields (`tfc_audio_checkpoint`/`tfc_vib_checkpoint`) rather than
    one: unlike BEATs (audio-branch only), TF-C is pre-trained separately per branch
    (MIMII for audio, CWRU/Paderborn bearing vibration for vibration), so the two
    checkpoints are unrelated files and must be settable independently. Consumed by
    `rowii.pipeline._featurizer_for_stream`'s `"audio-tfc"` dispatch."""
    tfc_vib_checkpoint: Path | None = None
    """Frozen TF-C vibration-branch checkpoint (`ROWII_TFC_VIB_CHECKPOINT`) -- see
    `tfc_audio_checkpoint`'s docstring. Consumed by `rowii.pipeline.
    _featurizer_for_stream`'s `"vibration-tfc"` dispatch."""
    student_checkpoint: Path | None = None
    """Distilled BEATs-student checkpoint (`ROWII_STUDENT_CHECKPOINT`)
    -- backs the `"audio-student"` variant's `rowii.adapt.
    student.StudentFeaturizer`. Unlike TF-C's two independent branch
    checkpoints, the student has only ONE (audio-only, distilled from BEATs'
    own audio-branch teacher embeddings via `scripts/distill_beats.py`) --
    mirrors `beats_checkpoint`'s own single-field env-driven pattern. Consumed
    by `rowii.pipeline._featurizer_for_stream`'s `"audio-student"` dispatch."""
    beats_int8_checkpoint: Path | None = None
    """Post-training INT8-quantized BEATs checkpoint (`ROWII_BEATS_INT8_
    CHECKPOINT`) -- a `scripts/quantize_beats.py`-
    produced module pickle, NOT the `{"cfg","model"}` state-dict format
    `beats_checkpoint` points at. Independent of `beats_checkpoint` (both may
    be set together: the int8 file was quantized FROM the fp32 one, but
    `rowii.signals.beats.BeatsFeaturizer`'s int8 branch never reads
    `beats_checkpoint` at all once this is set -- mirrors `student_checkpoint`'s
    own single-field env-driven pattern otherwise). Consumed by `rowii.pipeline.
    _featurizer_for_stream`'s beats-variant dispatch (`"audio-beats"`/
    `"fusion-beats"`) as `BeatsFeaturizer`'s `int8_model_path` constructor arg."""
    xattn_checkpoint: Path | None = None
    """Cross-attention fusion-head checkpoint (`ROWII_XATTN_CHECKPOINT`)
    -- a `scripts/train_xattn.py`-produced
    `{"cfg","model","run","vib_dim","epochs"}` checkpoint backing
    `scripts/run_step2.py`'s `--xattn-fusion` view (the design chapter's third
    fusion level). Single-field env-driven pattern like `student_checkpoint`;
    consumed only by the run_step2 view (no pipeline variant of its own -- the
    joint embedding is computed from the fusion cache's vibration columns plus
    the audio-beats cache at view time)."""


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Build a Config from (in order) explicit *env*, process env, and .env file."""
    file_env = {k: v for k, v in dotenv_values(".env").items() if v is not None}
    merged: dict[str, str] = {**file_env, **os.environ}
    if env is not None:
        merged = dict(env)
    ckpt = merged.get("ROWII_BEATS_CHECKPOINT") or None
    tfc_audio_ckpt = merged.get("ROWII_TFC_AUDIO_CHECKPOINT") or None
    tfc_vib_ckpt = merged.get("ROWII_TFC_VIB_CHECKPOINT") or None
    student_ckpt = merged.get("ROWII_STUDENT_CHECKPOINT") or None
    beats_int8_ckpt = merged.get("ROWII_BEATS_INT8_CHECKPOINT") or None
    xattn_ckpt = merged.get("ROWII_XATTN_CHECKPOINT") or None
    max_invalid = merged.get("ROWII_MAX_INVALID_FRACTION") or None
    return Config(
        data_root=Path(merged.get("ROWII_DATA_ROOT", "data")).expanduser(),
        results_root=Path(merged.get("ROWII_RESULTS_ROOT", "results")).expanduser(),
        max_invalid_fraction=float(max_invalid) if max_invalid else 0.05,
        beats_checkpoint=Path(ckpt).expanduser() if ckpt else None,
        tfc_audio_checkpoint=Path(tfc_audio_ckpt).expanduser() if tfc_audio_ckpt else None,
        tfc_vib_checkpoint=Path(tfc_vib_ckpt).expanduser() if tfc_vib_ckpt else None,
        student_checkpoint=Path(student_ckpt).expanduser() if student_ckpt else None,
        beats_int8_checkpoint=Path(beats_int8_ckpt).expanduser() if beats_int8_ckpt else None,
        xattn_checkpoint=Path(xattn_ckpt).expanduser() if xattn_ckpt else None,
    )
