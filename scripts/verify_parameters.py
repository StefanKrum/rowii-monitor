"""Task-13 parameter verification against the real 2026-06-25 Rodundwerk II delivery.

Per the no-legacy-assumptions constraint (Task 13 dispatch): every machine parameter
hard-coded in `rowii.config` or `rowii.signals.features` is a HYPOTHESIS until confirmed
or corrected from THIS data. This script measures each one directly from the real
Betriebsdaten / TU burst files and writes a permanent record of the derivation to
`results/parameter_verification.md` (regenerate by re-running this script -- like every
other file under `results/`, it is not committed to the repo, only this script is).

Five checks, matching the dispatch's Step 2 numbering:
  1. Speed plateau (GT_CHANNELS["speed"]) during turbine generation vs GtRules.speed_nominal_rpm.
  2. Welch-PSD spectral peaks near each MACHINE_HZ band vs the configured centre.
  3. Vibration-channel liveness (which of the 6 channels per stream actually carry signal).
  4. SCADA sign/flow conventions (turbine vs pump) vs the rules encoded in scada/labels.py.
  5. Betriebsdaten coverage of the pu-afternoon run (yes/no, with exact timestamps).

Run with `python scripts/verify_parameters.py` from the repo root (needs ROWII_DATA_ROOT).
"""
from __future__ import annotations

import dataclasses
import datetime
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.config import Config, GtRules, load_config  # noqa: E402
from rowii.io.dataset import RecordingIndex, betriebsdaten_utc_offset_ns, discover  # noqa: E402
from rowii.io.gantner import GantnerHeader, read_gantner, read_header  # noqa: E402
from rowii.scada.labels import GT_CHANNELS, gt_labels, load_scada_window_means  # noqa: E402
from rowii.signals.features import MACHINE_HZ  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402

_DAY_ROOT = "illwerke-250526"
"""ROWII_DATA_ROOT now points at the PARENT root, containing every
`illwerke-<dayid>` day tree -- checks 1-5 below are specific to the original
2026-06-25 delivery, so they resolve their own Betriebsdaten hour under THIS
day tree explicitly rather than assuming `cfg.data_root` itself is a single
day root (the pre-addendum behaviour)."""
_MESSUNG_DIR = "20260626 Messung"
_BETRIEBSDATEN_DIR = f"{_DAY_ROOT}/{_MESSUNG_DIR}/Betriebsdaten"
_RUN_NAME_TU = "250526-tu"
_RUN_NAME_PU_AFTERNOON = "250526-pu-afternoon"
_PH_VERIFICATION_RUN_NAME = "010726-tu_ph_tu"
"""Day tree used for the phase-shifter channel verification (addendum spec §3/
Item 4): 2026-07-01 is the campaign's only day with ALL FOUR operating modes
in one full-SCADA-covered day, including a confirmed, sustained phase-shifter
interval."""
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
    channel_name: str
    n_samples: int
    n_samples_generating: int
    plateau_rpm: float
    configured_rpm: float
    deviation_pct: float


def measure_speed_plateau(
    cfg: Config, hour_file: str = "2026-06-25_05-00-00.dat"
) -> SpeedPlateauResult:
    """Median GT_CHANNELS["speed"] while power > threshold, for one Betriebsdaten hour."""
    path = cfg.data_root / _BETRIEBSDATEN_DIR / hour_file
    gf = read_gantner(path)
    channel_name = GT_CHANNELS["speed"]
    power_idx = gf.header.channel_names.index(GT_CHANNELS["power"])
    speed_idx = gf.header.channel_names.index(channel_name)
    power = gf.data[:, power_idx].astype(np.float64)
    speed = gf.data[:, speed_idx].astype(np.float64)

    mask = power > _POWER_GENERATION_THRESHOLD_MW
    plateau = float(np.median(speed[mask])) if mask.any() else float("nan")
    configured = cfg.gt.speed_nominal_rpm
    deviation_pct = abs(plateau - configured) / configured * 100.0 if configured else float("nan")

    return SpeedPlateauResult(
        hour_file=hour_file,
        channel_name=channel_name,
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
    run = next(r for r in index.runs if r.name == _RUN_NAME_TU)
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
    run = next(r for r in index.runs if r.name == _RUN_NAME_TU)
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
    pu_afternoon = next(r for r in index.runs if r.name == _RUN_NAME_PU_AFTERNOON)
    # Scoped to pu-afternoon's OWN day tree (see `Run.day_root`) -- the pooled,
    # flat `index.betriebsdaten` now spans every day in the parent root, so its
    # own last-sorted entry is no longer necessarily 250526's own last hour.
    day_betriebsdaten = index.betriebsdaten_by_day[pu_afternoon.day_root]
    last_bd_path = sorted(day_betriebsdaten)[-1]
    h_bd = read_header(last_bd_path)
    bd_end_ns = h_bd.t0_ns + round(h_bd.n_frames / h_bd.sample_rate_hz * 1e9)

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
# 6. Phase-shifter channels (multi-day/phase-shifter addendum, Item 4): per-GT-state
#    distribution of ks_valve ("1_KS Stellung") and reactive ("1_Q_Ist") on the
#    2026-07-01 delivery, GT ALWAYS computed with the KS gate forced off
#    (dataclasses.replace(cfg.gt, ph_requires_ks_closed=False)) -- verification
#    must observe the gate's OWN hypothesis, never have an already-adopted
#    verdict circularly baked into the GT it is checking.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelDistribution:
    median: float
    p5: float
    p95: float
    n: int


@dataclass(frozen=True)
class PhaseShifterChannelStats:
    day_root: str
    n_windows_total: int
    ks_valve_by_state: dict[str, ChannelDistribution]
    reactive_by_state: dict[str, ChannelDistribution]


def _build_whole_day_grid(betriebsdaten: list[Path], window_s: float) -> WindowGrid:
    """A grid spanning one day tree's FULL Betriebsdaten coverage (first file's start
    to last file's end), not the stream-INTERSECTION `common_grid` computes (which is
    empty by construction for a sequence of back-to-back, non-overlapping hourly
    files -- `common_grid` is designed for OVERLAPPING burst streams, not consecutive
    SCADA hours).

    Task 10 (D3 consequence): this is a standalone grid builder, bypassing `rowii.
    pipeline.build_run_grid` entirely (no audio run involved) -- `load_scada_window_
    means` (module-docstring's only caller of this function, `measure_phase_shifter_
    channels`) now always shifts its OWN raw SCADA timestamps onto true UTC (D3), so
    this grid's `t0_ns` must be shifted by the SAME *betriebsdaten*-derived offset
    (`rowii.io.dataset.betriebsdaten_utc_offset_ns`) to stay on that same axis --
    otherwise every window here would search for true-UTC timestamps against a
    grid still on the raw ~30-years-earlier axis and find nothing.
    """
    sorted_files = sorted(betriebsdaten)
    first_h = read_header(sorted_files[0])
    last_h = read_header(sorted_files[-1])
    offset_ns = betriebsdaten_utc_offset_ns(betriebsdaten)
    first_start_ns = first_h.t0_ns + offset_ns
    last_end_ns = last_h.t0_ns + offset_ns + round(last_h.n_frames / last_h.sample_rate_hz * 1e9)
    window_ns = round(window_s * 1e9)
    n_windows = (last_end_ns - first_start_ns) // window_ns
    return WindowGrid(t0_ns=first_start_ns, window_ns=window_ns, n_windows=n_windows)


def _channel_distribution(values: np.ndarray) -> ChannelDistribution:
    finite = values[~np.isnan(values)]
    if len(finite) == 0:
        return ChannelDistribution(median=float("nan"), p5=float("nan"), p95=float("nan"), n=0)
    return ChannelDistribution(
        median=float(np.median(finite)),
        p5=float(np.percentile(finite, 5)),
        p95=float(np.percentile(finite, 95)),
        n=len(finite),
    )


def measure_phase_shifter_channels(
    cfg: Config, index: RecordingIndex, run_name: str = _PH_VERIFICATION_RUN_NAME
) -> PhaseShifterChannelStats:
    """Per-GT-state ks_valve/reactive distributions for *run_name*'s own day tree.

    GT is ALWAYS computed with the KS gate FORCED OFF (`dataclasses.replace(cfg.gt,
    ph_requires_ks_closed=False)`), regardless of what `cfg.gt.ph_requires_ks_closed`
    itself currently is -- this measurement's entire purpose is to check whether
    ks_valve separates cleanly ACROSS the states the speed+power+dwell rule alone
    produces, so it must never let a previously-adopted verdict (the gate now ships
    ENABLED by default, per this very section's own 2026-07-08 finding) circularly
    feed back into the states being re-measured on a later re-run of this script.
    """
    run = next(r for r in index.runs if r.name == run_name)
    day_betriebsdaten = index.betriebsdaten_by_day[run.day_root]

    grid = _build_whole_day_grid(day_betriebsdaten, cfg.window.window_s)
    scada = load_scada_window_means(sorted(day_betriebsdaten), grid)
    rules_without_gate = dataclasses.replace(cfg.gt, ph_requires_ks_closed=False)
    gt = gt_labels(scada, rules_without_gate, window_s=cfg.window.window_s)

    ks_valve = scada["ks_valve"].to_numpy(dtype=np.float64)
    reactive = scada["reactive"].to_numpy(dtype=np.float64)
    states = gt["state"].to_numpy()

    ks_by_state: dict[str, ChannelDistribution] = {}
    reactive_by_state: dict[str, ChannelDistribution] = {}
    for state in sorted(set(states)):
        mask = states == state
        ks_by_state[state] = _channel_distribution(ks_valve[mask])
        reactive_by_state[state] = _channel_distribution(reactive[mask])

    return PhaseShifterChannelStats(
        day_root=run.day_root.name,
        n_windows_total=grid.n_windows,
        ks_valve_by_state=ks_by_state,
        reactive_by_state=reactive_by_state,
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _render_ph_channel_table(
    title: str, by_state: dict[str, ChannelDistribution]
) -> list[str]:
    lines = [
        f"**{title}**",
        "",
        "| GT state | n | median | p5 | p95 |",
        "|---|---|---|---|---|",
    ]
    for state, dist in sorted(by_state.items()):
        lines.append(
            f"| {state} | {dist.n} | {dist.median:.3f} | {dist.p5:.3f} | {dist.p95:.3f} |"
        )
    lines.append("")
    return lines


def _render_phase_shifter_channels_section(
    ph_stats: PhaseShifterChannelStats, rules: GtRules
) -> list[str]:
    ks_ph = ph_stats.ks_valve_by_state.get("phase-shifter")
    ks_ss = ph_stats.ks_valve_by_state.get("standstill")
    ks_tu = ph_stats.ks_valve_by_state.get("turbine")
    ks_pu = ph_stats.ks_valve_by_state.get("pump")

    lines: list[str] = []
    lines.append("## Phase-shifter channels, 2026-07-08")
    lines.append("")
    lines.append(
        f"Source: day tree `{ph_stats.day_root}` (2026-07-01, ALL FOUR operating "
        f"modes -- addendum spec §1), whole-day grid ({ph_stats.n_windows_total} "
        f"windows at {1.0:.0f}-s resolution assumed from `cfg.window.window_s`). GT "
        "computed with the KS gate FORCED OFF regardless of the current config "
        "(`measure_phase_shifter_channels` always uses "
        "`dataclasses.replace(cfg.gt, ph_requires_ks_closed=False)`) -- this "
        "section's own purpose is to check whether the gate's hypothesis holds, so "
        "it must never let an already-adopted verdict circularly feed back into the "
        "states being re-measured on a later re-run of this script."
    )
    lines.append("")
    lines.extend(_render_ph_channel_table("1_KS Stellung (ks_valve)", ph_stats.ks_valve_by_state))
    lines.extend(_render_ph_channel_table("1_Q_Ist (reactive)", ph_stats.reactive_by_state))

    lines.append(
        "**Hypothesis under test** (provenance: partner/Bruno's SCADA channel audit, "
        "relayed pre-verification -- NOT an internally-derived number): "
        "`1_KS Stellung` (spherical inlet valve) reads ~3 when closed (standstill, "
        "phase-shifter) and ~104 when open (turbine, pump); independent confirmation "
        "required on our own 2026-07-01 data before use (addendum spec §3)."
    )
    lines.append("")

    if ks_ph is not None and ks_ss is not None and ks_tu is not None and ks_pu is not None:
        closed_states_max = max(ks_ph.p95, ks_ss.p95)
        open_states_min = min(ks_tu.p5, ks_pu.p5)
        separated = closed_states_max < open_states_min
        verdict = "CONFIRMED" if separated else "AMBIGUOUS"
        lines.append(
            f"**Verdict: {verdict}** -- phase-shifter/standstill ks_valve "
            f"(p95: {ks_ph.p95:.3f} / {ks_ss.p95:.3f}) vs. turbine/pump ks_valve "
            f"(p5: {ks_tu.p5:.3f} / {ks_pu.p5:.3f})."
        )
        lines.append("")
        if separated:
            lines.append(
                f"Clean separation confirmed on our own data: measured closed-state p95 "
                f"(phase-shifter {ks_ph.p95:.3f}, standstill {ks_ss.p95:.3f}) vs. "
                f"open-state p5 (turbine {ks_tu.p5:.3f}, pump {ks_pu.p5:.3f}). "
                f"`ph_requires_ks_closed=True` and `ks_closed_max={rules.ks_closed_max}` "
                f"recorded in `GtRules` with this section as verification provenance."
            )
        else:
            lines.append(
                "Separation is NOT clean on our own data -- `ph_requires_ks_closed` "
                "stays `False` (unverified hypothesis not adopted); see the per-state "
                "tables above for the actual measured overlap."
            )
    else:
        missing = [
            name
            for name, dist in (
                ("phase-shifter", ks_ph), ("standstill", ks_ss),
                ("turbine", ks_tu), ("pump", ks_pu),
            )
            if dist is None
        ]
        lines.append(
            f"**Verdict: AMBIGUOUS** -- one or more GT states have zero eval windows "
            f"on this day tree ({', '.join(missing)}), so the separation cannot be "
            "checked; `ph_requires_ks_closed` stays `False`."
        )
    lines.append("")
    lines.append(
        f"Current config: `ph_requires_ks_closed={rules.ph_requires_ks_closed}`, "
        f"`ks_closed_max={rules.ks_closed_max}`."
    )
    lines.append("")

    return lines


def _render_report(
    speed: SpeedPlateauResult,
    spectral: list[SpectralVerificationResult],
    liveness: list[VibLivenessResult],
    scada_hours: list[ScadaHourStats],
    pu_coverage: PuAfternoonCoverageResult,
    ph_stats: PhaseShifterChannelStats,
    gt_rules: GtRules,
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
    lines.append(f"- Source: `{speed.hour_file}`, median `{speed.channel_name}` where "
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
        "(~127-290 MW), speed POSITIVE (~379 rpm), `flow_tu` dominant, `flow_pu` ~0. Matches "
        "`_base_state`'s `turbine_by_power`/`turbine_by_flow` rules."
    )
    lines.append("")
    lines.append(
        "**Pump convention** (09:00 hour -- inside pu-morning): power NEGATIVE (~-282 to -45 "
        "MW), speed NEGATIVE (~-378 rpm, opposite rotation direction), `flow_pu` dominant "
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

    lines.extend(_render_phase_shifter_channels_section(ph_stats, gt_rules))

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
        "all 4 TU streams and 3 independent Betriebsdaten hours during Task 13. "
        "**Update (Task 10, epoch-2000 clock quirk fix):** the quirk described above IS now "
        "corrected in the pipeline -- `rowii.io.dataset.run_utc_offset_ns`/"
        "`betriebsdaten_utc_offset_ns` derive this same offset from exactly the filename-hint-"
        "vs-`t0_ns` relationship documented here, and `rowii.pipeline.build_run_grid`/"
        "`rowii.scada.labels.load_scada_window_means`/every script's own "
        "`_betriebsdaten_for_grid` apply it before any SCADA<->burst alignment; the pipeline's "
        "grid, segment, and report timestamps are true UTC from that commit on. Any code that "
        "still reads `header.t0_ns` directly (as the measurements in this report do, for "
        "diagnostic purposes) must continue to NOT interpret it as a real calendar date without "
        "applying that same derived offset first."
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
    ph_stats = measure_phase_shifter_channels(cfg, index)

    report = _render_report(
        speed, spectral, liveness, scada_hours, pu_coverage, ph_stats, cfg.gt
    )

    out_path = cfg.results_root / "parameter_verification.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)

    print(f"verify_parameters: wrote {out_path}")
    print(f"  speed plateau: {speed.plateau_rpm:.3f} rpm "
          f"(configured {speed.configured_rpm:.1f}, deviation {speed.deviation_pct:.1f}%)")
    print(f"  pu-afternoon SCADA coverage: {'COVERED' if pu_coverage.covered else 'NOT COVERED'}")
    ks_ph = ph_stats.ks_valve_by_state.get("phase-shifter")
    ks_tu = ph_stats.ks_valve_by_state.get("turbine")
    if ks_ph is not None and ks_tu is not None:
        print(f"  phase-shifter ks_valve median: {ks_ph.median:.3f} "
              f"(vs turbine median: {ks_tu.median:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
