"""Campaign-overview figure: one row per measurement day, showing (a) the
SCADA operating-state timeline over the day, (b) the signed active-power curve,
and (c) every audio recording session as a bracket above the row, edge-styled
by its role in the evaluation (commissioning pool / held-out monitored /
sentinel-only / controlled events / excluded). This is the visual form of the
per-day data-usage map: which modes ran when, what was used for what, and at
which loads. Reads only discover() metadata, Betriebsdaten and the pinned
usage constants; writes graphics for the thesis campaign section."""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from rowii.config import load_config
from rowii.io.dataset import betriebsdaten_utc_offset_ns, discover
from rowii.io.gantner import read_gantner
from rowii.scada.labels import gt_labels, load_scada_window_means
from rowii.signals.windows import WindowGrid

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/analysis-days/campaign")

DAYS = [
    ("250526", "25 June", "A"),
    ("270626", "27 June", "A"),
    ("290626", "29 June", "B"),
    ("300626", "30 June", "B"),
    ("010726", "1 July", "B"),
    ("080726", "8 July", "C"),
]

# Session role map -- mirrors run_once_calibrated._REPLAY/_B1_FIT_RUNS,
# candidate_kit's coverage extension, and the documented exclusions.
ROLE: dict[str, str] = {
    "250526-tu": "monitored",
    "250526-pu-morning": "monitored",
    "250526-pu-afternoon": "excluded",
    "270626-pu_ph_pu_ph_pu_ph-1": "sentinel-only",
    "270626-pu_ph_pu_ph_pu_ph-2": "excluded",
    "270626-pu_ph_pu_ph_pu_ph-3": "excluded",
    "290626-tu": "monitored",
    "290626-pu": "monitored",
    "300626-tu": "monitored",
    "300626-pu": "monitored",
    "010726-tu_ph_tu": "pool",
    "010726-pu": "pool",
    "010726-tu1-morning": "pool",
    "010726-tu1-afternoon": "excluded",
    "010726-tu2": "pool",
    "080726-pu_strikes": "events",
    "080726-st_strikes": "events",
}
ROLE_STYLE = {
    "pool": dict(color="#228833", label="commissioning pool (calibration)"),
    "monitored": dict(color="#111111", label="monitored, held out"),
    "sentinel-only": dict(color="#666666", label="sentinel-only (no process data)"),
    "events": dict(color="#cc3311", label="controlled events"),
    "excluded": dict(color="#bbbbbb", label="excluded"),
}
CAL_COLOR = "#228833"   # windows used for calibration (pool, or on-day recal block)
SCORED_COLOR = "#111111"  # windows scored (tested), held out
TRIGGER_LOG = Path("results/step2/once-calibrated/fusion/fusion_trigger_log.csv")
MONITOR_ROOT = Path("results/step2/once-calibrated/fusion/monitor")
STATE_COLOR = {
    "standstill": "#e6e6e6",
    "turbine": "#88bbdd",
    "pump": "#f4b183",
    "phase-shifter": "#c5a3d9",
    "transition": "#999999",
}
SHORT = {
    "250526-tu": "TU", "250526-pu-morning": "PU (morning)", "250526-pu-afternoon": "PU (afternoon)",
    "270626-pu_ph_pu_ph_pu_ph-1": "PU+PS #1", "270626-pu_ph_pu_ph_pu_ph-2": "#2",
    "270626-pu_ph_pu_ph_pu_ph-3": "#3",
    "290626-tu": "TU", "290626-pu": "PU",
    "300626-tu": "TU", "300626-pu": "PU",
    "010726-tu_ph_tu": "TU+PS", "010726-pu": "PU", "010726-tu1-morning": "TU1",
    "010726-tu1-afternoon": "TU1 (aft.)", "010726-tu2": "TU2",
    "080726-pu_strikes": "PU, strikes", "080726-st_strikes": "ST, strikes",
}

cfg = load_config()
idx = discover(cfg.data_root)


def day_midnight_utc(day: str) -> datetime:
    """Midnight (UTC) of the day's ACTUAL calendar date, derived from the
    recordings themselves rather than the folder token: `illwerke-250526` is a
    documented misnomer (the recordings are 25 June, not 25 May), so parsing
    the token would shift that row a month off the axis."""
    starts = [
        bf.start_utc_hint
        for run in idx.runs if run.name.startswith(day)
        for files in run.files.values() for bf in files
    ]
    d = min(starts).astimezone(UTC).date()
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def scada_timeline(day: str):
    """(hours, state_str_array, power_array) at 1-s resolution, or None."""
    files = next(
        (v for k, v in idx.betriebsdaten_by_day.items() if day in str(k)), None
    )
    if not files:
        return None
    off = betriebsdaten_utc_offset_ns(files)
    t0 = read_gantner(files[0]).header.t0_ns + off
    t_end = read_gantner(files[-1]).header.t0_ns + off + 3_600_000_000_000
    n = int((t_end - t0) // 1_000_000_000)
    grid = WindowGrid(t0_ns=int(t0), window_ns=1_000_000_000, n_windows=n)
    scada = load_scada_window_means(files, grid)
    gt = gt_labels(scada, cfg.gt, window_s=1.0)
    mid = day_midnight_utc(day)
    hours = (t0 - int(mid.timestamp() * 1e9)) / 3.6e12 + np.arange(n) / 3600.0
    return hours, gt["state"].to_numpy(), scada["power"].to_numpy()


def segments(hours: np.ndarray, states: np.ndarray):
    """Contiguous (start_h, width_h, state) runs."""
    out = []
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            out.append((hours[start], hours[i - 1] - hours[start] + 1 / 3600, states[start]))
            start = i
    return out


def session_extents(day: str):
    out = []
    for run in idx.runs:
        if not run.name.startswith(day):
            continue
        starts = [bf.start_utc_hint for files in run.files.values() for bf in files]
        if not starts:
            continue
        mid = day_midnight_utc(day)
        h0 = (min(starts) - mid).total_seconds() / 3600.0
        h1 = (max(starts) - mid).total_seconds() / 3600.0 + 0.2  # last chunk ~12 min
        out.append((run.name, h0, h1))
    return sorted(out, key=lambda t: t[1])


import pandas as pd  # noqa: E402

_decisions = (
    pd.read_csv(TRIGGER_LOG).set_index("run")["decision"].to_dict()
    if TRIGGER_LOG.exists() else {}
)


def usage_segments(name: str, day: str, h0: float, h1: float):
    """Per-window usage of one session as (start_h, end_h, color) segments,
    read off the replayed once-per-era arm's own alarms.parquet role column:
    green = the window's data went into calibration (the whole commissioning
    pool, or a recalibrate-decision day's consumed on-day block), black/red =
    the window was scored (tested). Falls back to one full-extent segment
    where no replay artifact exists (excluded / sentinel-only sessions)."""
    role = ROLE[name]
    if role == "pool":
        return [(h0, h1, CAL_COLOR)]
    if role in ("excluded", "sentinel-only"):
        return [(h0, h1, ROLE_STYLE[role]["color"])]
    scored_color = ROLE_STYLE["events"]["color"] if role == "events" else SCORED_COLOR
    arm = "frozen" if _decisions.get(name) == "frozen" else "recalibrate"
    path = MONITOR_ROOT / name / arm / "alarms.parquet"
    if not path.exists():
        return [(h0, h1, scored_color)]
    df = pd.read_parquet(path)[["t_utc_ns", "role"]]
    df = df[df["role"].isin(["consumed_for_calibration", "scored"])]
    mid_ns = int(day_midnight_utc(day).timestamp() * 1e9)
    hours = (df["t_utc_ns"].to_numpy() - mid_ns) / 3.6e12
    colors = np.where(
        df["role"].to_numpy() == "consumed_for_calibration", CAL_COLOR, scored_color
    )
    segs = []
    start = 0
    for i in range(1, len(colors) + 1):
        if i == len(colors) or colors[i] != colors[start] or hours[i] - hours[i - 1] > 0.05:
            segs.append((hours[start], hours[i - 1] + 1 / 3600, colors[start]))
            start = i
    return segs


plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7.5, "axes.linewidth": 0.6,
    "xtick.labelsize": 7, "ytick.labelsize": 6.5,
})
fig, axes = plt.subplots(len(DAYS), 1, figsize=(7.0, 7.6), sharex=True)

for ax, (day, label, era) in zip(axes, DAYS, strict=True):
    tl = scada_timeline(day)
    if tl is not None:
        hours, states, power = tl
        for h0, w, st in segments(hours, states):
            if st in STATE_COLOR:
                ax.axvspan(h0, h0 + w, color=STATE_COLOR[st], lw=0, zorder=0)
        step = 30
        ax.plot(hours[::step], power[::step], color="#222222", lw=0.7, zorder=3)
    else:
        ax.text(12, 0, "no process export", ha="center", va="center",
                fontsize=7.5, style="italic", color="#888888")
    for name, h0, h1 in session_extents(day):
        style = ROLE_STYLE[ROLE[name]]
        ls = ":" if ROLE[name] == "excluded" else "-"
        for s0, s1, col in usage_segments(name, day, h0, h1):
            ax.plot([s0, s1], [352, 352], color=col, lw=3.2, linestyle=ls,
                    solid_capstyle="butt", zorder=4, clip_on=False)
        ax.text((h0 + h1) / 2, 385, SHORT[name], ha="center", va="bottom",
                fontsize=6.2, color=style["color"], clip_on=False)
    ax.set_ylim(-320, 340)
    ax.set_yticks([-280, 280])
    ax.axhline(0, color="#cccccc", lw=0.4, zorder=1)
    ax.text(-0.075, 0.5, f"{label}\n(era {era})", transform=ax.transAxes,
            ha="right", va="center", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

axes[-1].set_xlim(0, 24)
axes[-1].set_xticks(range(0, 25, 2))
axes[-1].set_xlabel("hour of day (UTC)")
fig.text(0.006, 0.5, "active power (MW; negative = pump)", rotation=90,
         va="center", fontsize=7.5)

state_handles = [Patch(color=c, label=s) for s, c in STATE_COLOR.items()]
role_handles = [
    Line2D([], [], color=CAL_COLOR, lw=3.2,
           label="used for calibration (pool / on-day block)"),
    Line2D([], [], color=SCORED_COLOR, lw=3.2, label="scored (tested), held out"),
    Line2D([], [], color=ROLE_STYLE["events"]["color"], lw=3.2,
           label="scored, controlled events"),
    Line2D([], [], color=ROLE_STYLE["sentinel-only"]["color"], lw=3.2,
           label="sentinel-only (no process data)"),
    Line2D([], [], color=ROLE_STYLE["excluded"]["color"], lw=3.2, linestyle=":",
           label="excluded"),
]
fig.legend(handles=state_handles + role_handles, loc="lower center",
           ncol=4, fontsize=6.4, frameon=False, bbox_to_anchor=(0.53, -0.005))
fig.subplots_adjust(left=0.14, right=0.99, top=0.955, bottom=0.12, hspace=0.42)

OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    p = OUT_DIR / f"campaign.{ext}"
    fig.savefig(p, dpi=200)
    print(f"wrote {p}")
