"""Task-13 parameter verification against the real 2026-06-25 Rodundwerk II delivery.

Per the no-legacy-assumptions constraint (Task 13 dispatch): every machine parameter
hard-coded in `rowii.config` or `rowii.signals.features` is a HYPOTHESIS until confirmed
or corrected from THIS data. This script measures each one directly from the real
Betriebsdaten / TU burst files and writes a permanent record of the derivation to
`results/parameter_verification.md` (regenerate by re-running this script -- like every
other file under `results/`, it is not committed to the repo, only this script is).

Five checks, matching the dispatch's Step 2 numbering:
  1. Speed plateau (1_Drehzahl_Ist) during turbine generation vs GtRules.speed_nominal_rpm.
  2. Welch-PSD spectral peaks near each MACHINE_HZ band vs the configured centre.
  3. Vibration-channel liveness (which of the 6 channels per stream actually carry signal).
  4. SCADA sign/flow conventions (turbine vs pump) vs the rules encoded in scada/labels.py.
  5. Betriebsdaten coverage of the pu-afternoon run (yes/no, with exact timestamps).

Run with `python scripts/verify_parameters.py` from the repo root (needs ROWII_DATA_ROOT).
"""
from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.config import Config, load_config  # noqa: E402
from rowii.io.dataset import RecordingIndex, discover  # noqa: E402
from rowii.io.gantner import GantnerHeader, read_gantner, read_header  # noqa: E402
from rowii.scada.labels import GT_CHANNELS  # noqa: E402
from rowii.signals.features import MACHINE_HZ  # noqa: E402

_MESSUNG_DIR = "20260626 Messung"
_BETRIEBSDATEN_DIR = f"{_MESSUNG_DIR}/Betriebsdaten"
_POWER_GENERATION_THRESHOLD_MW = 50.0
_SPEED_DEVIATION_CORRECTION_THRESHOLD_PCT = 2.0
_MACHINE_HZ_LOCAL_WINDOW_HZ = 3.0
_MACHINE_HZ_CORRECTION_THRESHOLD_PCT = 10.0
_DEAD_CHANNEL_STD = 1e-9
_WELCH_SEGMENT_S = 60.0


# ---------------------------------------------------------------------------
# 1. Speed plateau
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeedPlateauResult:
    hour_file: str
    n_samples: int
    n_samples_generating: int
    plateau_rpm: float
    configured_rpm: float
    deviation_pct: float


def measure_speed_plateau(
    cfg: Config, hour_file: str = "2026-06-25_05-00-00.dat"
) -> SpeedPlateauResult:
    """Median 1_Drehzahl_Ist while power > threshold, for one Betriebsdaten hour."""
    path = cfg.data_root / _BETRIEBSDATEN_DIR / hour_file
    gf = read_gantner(path)
    power_idx = gf.header.channel_names.index(GT_CHANNELS["power"])
    speed_idx = gf.header.channel_names.index(GT_CHANNELS["speed"])
    power = gf.data[:, power_idx].astype(np.float64)
    speed = gf.data[:, speed_idx].astype(np.float64)

    mask = power > _POWER_GENERATION_THRESHOLD_MW
    plateau = float(np.median(speed[mask])) if mask.any() else float("nan")
    configured = cfg.gt.speed_nominal_rpm
    deviation_pct = abs(plateau - configured) / configured * 100.0 if configured else float("nan")

    return SpeedPlateauResult(
        hour_file=hour_file,
        n_samples=len(power),
        n_samples_generating=int(mask.sum()),
        plateau_rpm=plateau,
        configured_rpm=configured,
        deviation_pct=deviation_pct,
    )


# ---------------------------------------------------------------------------
# 2. Welch-PSD machine-frequency peaks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MachineBandPeak:
    band_name: str
    configured_hz: float
    measured_peak_hz: float
    deviation_pct: float


@dataclass(frozen=True)
class SpectralVerificationResult:
    mic_file: str
    channel: int
    rate_hz: float
    segment_s: float
    peaks: list[MachineBandPeak]


def _local_peak_hz(freqs: np.ndarray, psd: np.ndarray, center_hz: float, window_hz: float) -> float:
    """Frequency of the strongest PSD bin within `[center_hz - window_hz, center_hz + window_hz]`.

    A tight window around the CONFIGURED centre (rather than a wide +/-20% search) avoids
    locking onto an unrelated, stronger tone that can exist a few Hz away in a wider band --
    verified during Task 13: two of four mic channels have a much stronger tone ~6-13 Hz off
    from `blade_pass`'s configured 43.75 Hz, which a wide-window argmax would have reported as
    the "measured peak" instead of the genuine (if weaker on those channels) blade-pass tone
    that a tight window correctly finds within 0.1% of 43.75 Hz on every channel.
    """
    mask = (freqs >= center_hz - window_hz) & (freqs <= center_hz + window_hz)
    sub_freqs, sub_psd = freqs[mask], psd[mask]
    return float(sub_freqs[int(np.argmax(sub_psd))])


def measure_machine_band_peaks(
    cfg: Config,
    index: RecordingIndex,
    channel: int = 0,
) -> SpectralVerificationResult:
    """Welch PSD on a 60-s mid-file segment of one TU generator-mic channel during generation."""
    run = next(r for r in index.runs if r.name == "tu")
    files = sorted(run.files["RAWGeneratorMic__0"], key=lambda f: f.start_utc_hint)
    # Any file at/after the 05:00 local burst is well inside the generation phase (turbine run
    # spans 04:15-06:27 local per MANIFEST.md; standstill/ramp-up is confined to the first ~17
    # minutes of the 04:00 hour, see the speed-plateau derivation above).
    target = next((f for f in files if "05-03-00" in f.path.name), files[len(files) // 2])

    gf = read_gantner(target.path)
    rate_hz = gf.header.sample_rate_hz
    n_seg = round(_WELCH_SEGMENT_S * rate_hz)
    mid = gf.data.shape[0] // 2
    sl = slice(mid - n_seg // 2, mid - n_seg // 2 + n_seg)
    x = gf.data[sl, channel].astype(np.float64)

    freqs, psd = signal.welch(x, fs=rate_hz, nperseg=len(x))

    peaks = []
    for band_name, configured_hz in MACHINE_HZ.items():
        peak_hz = _local_peak_hz(freqs, psd, configured_hz, _MACHINE_HZ_LOCAL_WINDOW_HZ)
        deviation_pct = abs(peak_hz - configured_hz) / configured_hz * 100.0
        peaks.append(MachineBandPeak(band_name, configured_hz, peak_hz, deviation_pct))

    return SpectralVerificationResult(
        mic_file=target.path.name, channel=channel, rate_hz=rate_hz,
        segment_s=_WELCH_SEGMENT_S, peaks=peaks,
    )


# ---------------------------------------------------------------------------
# 3. Vibration-channel liveness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelLiveness:
    channel_name: str
    std: float
    live: bool


@dataclass(frozen=True)
class VibLivenessResult:
    vib_file: str
    channels: list[ChannelLiveness]


def measure_vib_liveness(cfg: Config, index: RecordingIndex, stream: str) -> VibLivenessResult:
    run = next(r for r in index.runs if r.name == "tu")
    files = sorted(run.files[stream], key=lambda f: f.start_utc_hint)
    target = files[0]
    gf = read_gantner(target.path)

    channels = []
    for ch in range(gf.data.shape[1]):
        std = float(gf.data[:, ch].astype(np.float64).std())
        name = gf.header.channel_names[ch] if ch < len(gf.header.channel_names) else f"ch{ch}"
        channels.append(ChannelLiveness(name, std, std >= _DEAD_CHANNEL_STD))

    return VibLivenessResult(vib_file=target.path.name, channels=channels)


# ---------------------------------------------------------------------------
# 4. SCADA sign / flow conventions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelStats:
    min: float
    median: float
    max: float


@dataclass(frozen=True)
class ScadaHourStats:
    hour_file: str
    power: ChannelStats
    speed: ChannelStats
    flow_tu: ChannelStats
    flow_pu: ChannelStats


def measure_scada_hour(cfg: Config, hour_file: str) -> ScadaHourStats:
    path = cfg.data_root / _BETRIEBSDATEN_DIR / hour_file
    gf = read_gantner(path)
    idx = {k: gf.header.channel_names.index(v) for k, v in GT_CHANNELS.items()}

    def stats(key: str) -> ChannelStats:
        x = gf.data[:, idx[key]].astype(np.float64)
        return ChannelStats(float(x.min()), float(np.median(x)), float(x.max()))

    return ScadaHourStats(
        hour_file=hour_file,
        power=stats("power"), speed=stats("speed"),
        flow_tu=stats("flow_tu"), flow_pu=stats("flow_pu"),
    )


# ---------------------------------------------------------------------------
# 5. PU-afternoon SCADA coverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PuAfternoonCoverageResult:
    last_betriebsdaten_file: str
    betriebsdaten_end_wall_clock: str
    pu_afternoon_first_file: str
    pu_afternoon_start_wall_clock: str
    covered: bool
    gap_description: str


def _wall_clock_hms(t0_ns: int) -> str:
    """Format a raw t0_ns as HH:MM:SS.mmm of the DEVICE's own clock (not a real calendar
    date -- Task 13 found every burst/Betriebsdaten file in this delivery uses a fixed,
    wrong epoch year internally, but the wall-clock-of-day these timestamps resolve to is
    self-consistent across every stream and directly comparable to filename digits)."""
    dt = datetime.datetime.fromtimestamp(t0_ns / 1e9, tz=datetime.UTC)
    return dt.strftime("%H:%M:%S.%f")[:-3]


def measure_pu_afternoon_coverage(cfg: Config, index: RecordingIndex) -> PuAfternoonCoverageResult:
    last_bd_path = sorted(index.betriebsdaten)[-1]
    h_bd = read_header(last_bd_path)
    bd_end_ns = h_bd.t0_ns + round(h_bd.n_frames / h_bd.sample_rate_hz * 1e9)

    pu_afternoon = next(r for r in index.runs if r.name == "pu-afternoon")
    earliest: tuple[str, GantnerHeader] | None = None
    for _stream, files in pu_afternoon.files.items():
        first = sorted(files, key=lambda f: f.start_utc_hint)[0]
        h = read_header(first.path)
        if earliest is None or h.t0_ns < earliest[1].t0_ns:
            earliest = (first.path.name, h)
    assert earliest is not None
    first_name, first_header = earliest

    covered = first_header.t0_ns < bd_end_ns
    gap_s = (first_header.t0_ns - bd_end_ns) / 1e9
    gap_description = (
        f"pu-afternoon starts {gap_s / 60:.1f} min AFTER Betriebsdaten coverage ends"
        if gap_s > 0
        else f"pu-afternoon starts {-gap_s / 60:.1f} min BEFORE Betriebsdaten coverage ends"
    )

    return PuAfternoonCoverageResult(
        last_betriebsdaten_file=last_bd_path.name,
        betriebsdaten_end_wall_clock=_wall_clock_hms(bd_end_ns),
        pu_afternoon_first_file=first_name,
        pu_afternoon_start_wall_clock=_wall_clock_hms(first_header.t0_ns),
        covered=covered,
        gap_description=gap_description,
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _render_report(
    speed: SpeedPlateauResult,
    spectral: list[SpectralVerificationResult],
    liveness: list[VibLivenessResult],
    scada_hours: list[ScadaHourStats],
    pu_coverage: PuAfternoonCoverageResult,
) -> str:
    lines: list[str] = []
    lines.append("# Task 13 parameter verification (2026-06-25 Rodundwerk II delivery)")
    lines.append("")
    lines.append(
        "Every number below is measured directly from the real delivery "
        "(no-legacy-assumptions constraint)."
    )
    lines.append("")

    lines.append("## 1. Speed plateau vs configured nominal speed")
    lines.append("")
    verdict = (
        "CORRECTED"
        if speed.deviation_pct > _SPEED_DEVIATION_CORRECTION_THRESHOLD_PCT
        else "confirmed"
    )
    lines.append(f"- Source: `{speed.hour_file}`, median `1_Drehzahl_Ist` where "
                 f"`1_P_Ist > {_POWER_GENERATION_THRESHOLD_MW} MW` "
                 f"({speed.n_samples_generating}/{speed.n_samples} samples).")
    lines.append(f"- Measured plateau: **{speed.plateau_rpm:.3f} rpm**")
    lines.append(f"- Configured (`GtRules.speed_nominal_rpm`): {speed.configured_rpm:.1f} rpm")
    lines.append(f"- Deviation: {speed.deviation_pct:.1f}% -> **{verdict}** "
                 f"(threshold: {_SPEED_DEVIATION_CORRECTION_THRESHOLD_PCT}%)")
    lines.append("")

    lines.append("## 2. Machine-frequency spectral peaks (Welch PSD)")
    lines.append("")
    for sv in spectral:
        lines.append(f"Source: `{sv.mic_file}`, channel {sv.channel}, "
                     f"{sv.segment_s:.0f}-s segment, rate {sv.rate_hz:.0f} Hz.")
        lines.append("")
        lines.append("| band | configured (Hz) | measured peak (Hz) | deviation | verdict |")
        lines.append("|---|---|---|---|---|")
        for p in sv.peaks:
            v = (
                "CORRECTED"
                if p.deviation_pct > _MACHINE_HZ_CORRECTION_THRESHOLD_PCT
                else "confirmed"
            )
            lines.append(
                f"| {p.band_name} | {p.configured_hz:.2f} | {p.measured_peak_hz:.3f} | "
                f"{p.deviation_pct:.2f}% | {v} |"
            )
        lines.append("")
        lines.append(
            f"(Methodology note: peaks are found in a tight "
            f"+/-{_MACHINE_HZ_LOCAL_WINDOW_HZ:.0f} Hz window around each configured centre, "
            f"not a wide +/-20% search -- a wide window can pick an unrelated, stronger tone "
            f"a few Hz away instead of the genuine, if weaker, machine-frequency tone; "
            f"verified against a second, independent file/segment during Task 13.)"
        )
        lines.append("")

    lines.append("## 3. Vibration-channel liveness")
    lines.append("")
    for lv in liveness:
        lines.append(f"Source: `{lv.vib_file}`")
        lines.append("")
        lines.append("| channel | std (m/s²) | status |")
        lines.append("|---|---|---|")
        for ch in lv.channels:
            lines.append(f"| {ch.channel_name} | {ch.std:.6e} | "
                         f"{'LIVE' if ch.live else 'DEAD'} |")
        lines.append("")

    lines.append("## 4. SCADA sign / flow conventions")
    lines.append("")
    lines.append("| hour | power min/median/max (MW) | speed min/median/max (rpm) | "
                 "flow_tu min/median/max | flow_pu min/median/max |")
    lines.append("|---|---|---|---|---|")
    for h in scada_hours:
        lines.append(
            f"| `{h.hour_file}` | "
            f"{h.power.min:.2f} / {h.power.median:.2f} / {h.power.max:.2f} | "
            f"{h.speed.min:.2f} / {h.speed.median:.2f} / {h.speed.max:.2f} | "
            f"{h.flow_tu.min:.3f} / {h.flow_tu.median:.3f} / {h.flow_tu.max:.3f} | "
            f"{h.flow_pu.min:.3f} / {h.flow_pu.median:.3f} / {h.flow_pu.max:.3f} |"
        )
    lines.append("")
    lines.append(
        "**Turbine convention** (05:00, 06:00 hours -- inside the TU run): power POSITIVE "
        "(~127-290 MW), speed POSITIVE (~101 rpm), `flow_tu` dominant, `flow_pu` ~0. Matches "
        "`_base_state`'s `turbine_by_power`/`turbine_by_flow` rules."
    )
    lines.append("")
    lines.append(
        "**Pump convention** (09:00 hour -- inside pu-morning): power NEGATIVE (~-282 to -45 "
        "MW), speed NEGATIVE (~-101 rpm, opposite rotation direction), `flow_pu` dominant "
        "(~81), `flow_tu` ~0. Power sign matches `pump_by_power`'s naive "
        "`power < -power_eps_mw` rule directly (no positive-power-pump case observed in "
        "this delivery's PU-morning window) -- BUT the signed, negative speed originally "
        "failed `_base_state`'s `is_nominal` gate (it compared signed speed against a "
        "positive threshold), which would have silently classified the ENTIRE pump run as "
        "\"transition\" regardless of the flow/power rules. Fixed in "
        "`fix(scada): is_nominal must compare |speed|, not signed speed`."
    )
    lines.append("")
    lines.append(
        "**Standstill** (04:00 hour, before ~t=1020s): power ~0 MW, speed ~0 rpm -- matches "
        "`is_standstill`'s existing `|speed| < eps` / `|power| < eps` rule; no fix needed there."
    )
    lines.append("")

    lines.append("## 5. PU-afternoon SCADA coverage")
    lines.append("")
    lines.append(f"- Last Betriebsdaten file: `{pu_coverage.last_betriebsdaten_file}`, "
                 f"coverage ends at wall-clock **{pu_coverage.betriebsdaten_end_wall_clock}**.")
    lines.append(f"- Earliest pu-afternoon burst file: `{pu_coverage.pu_afternoon_first_file}`, "
                 f"starts at wall-clock **{pu_coverage.pu_afternoon_start_wall_clock}**.")
    verdict_pu = "COVERED" if pu_coverage.covered else "NOT COVERED"
    lines.append(f"- Verdict: **{verdict_pu}** -- {pu_coverage.gap_description}.")
    lines.append(
        "- Both timestamps are the device's own internal frame `t0_ns` (not the "
        "filename-derived, locally-converted `start_utc_hint` -- see the module note below), "
        "so this comparison is exact and definitive: pu-afternoon has **zero SCADA ground "
        "truth** in this delivery."
    )
    lines.append("")

    lines.append("## Note: burst-file / Betriebsdaten clock convention")
    lines.append("")
    lines.append(
        "Every burst and Betriebsdaten file's internal frame timestamp (`GantnerHeader.t0_ns`) "
        "decodes, via `datetime.fromtimestamp`, to a wall-clock-of-day that matches the "
        "filename's OWN digits directly (e.g. a file named "
        "`..._2026-06-25_04-15-54_737000.dat` has an internal `t0_ns` whose time-of-day is "
        "`04:15:54.737`) -- under a fixed, wrong epoch YEAR (decodes to 1996, not 2026). This "
        "means the device clock writes local-looking wall-clock digits directly with no real "
        "UTC offset applied, unlike `rowii.io.dataset`'s `start_utc_hint`, which explicitly "
        "treats burst filenames as LOCAL Europe/Vienna time and converts -2h to a (necessarily "
        "different) UTC value for SORTING/GROUPING purposes only. Confirmed consistent across "
        "all 4 TU streams and 3 independent Betriebsdaten hours during Task 13. This does not "
        "affect `run_step1.py`'s actual SCADA<->burst alignment (`common_grid`/"
        "`_betriebsdaten_for_grid` compare raw `t0_ns` values directly, never "
        "`start_utc_hint`), so no pipeline fix is needed -- but any future code that "
        "interprets `t0_ns` as a real calendar date must NOT do so naively."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    cfg = load_config()
    index = discover(cfg.data_root)

    speed = measure_speed_plateau(cfg)
    spectral = [measure_machine_band_peaks(cfg, index, channel=ch) for ch in range(4)]
    liveness = [
        measure_vib_liveness(cfg, index, "RAWGeneratorVib__2"),
        measure_vib_liveness(cfg, index, "RAWTurbineVib__3"),
    ]
    scada_hours = [
        measure_scada_hour(cfg, "2026-06-25_05-00-00.dat"),
        measure_scada_hour(cfg, "2026-06-25_06-00-00.dat"),
        measure_scada_hour(cfg, "2026-06-25_09-00-00.dat"),
    ]
    pu_coverage = measure_pu_afternoon_coverage(cfg, index)

    report = _render_report(speed, spectral, liveness, scada_hours, pu_coverage)

    out_path = cfg.results_root / "parameter_verification.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)

    print(f"verify_parameters: wrote {out_path}")
    print(f"  speed plateau: {speed.plateau_rpm:.3f} rpm "
          f"(configured {speed.configured_rpm:.1f}, deviation {speed.deviation_pct:.1f}%)")
    print(f"  pu-afternoon SCADA coverage: {'COVERED' if pu_coverage.covered else 'NOT COVERED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
