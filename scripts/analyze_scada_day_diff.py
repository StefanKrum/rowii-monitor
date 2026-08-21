"""What differs in the SCADA data between 29.06 / 30.06 / 01.07 turbine
operation -- globally and at matched power (220-245 MW)?"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from rowii.config import load_config
from rowii.io.dataset import discover
from rowii.pipeline import prepare_run
from rowii.scada.labels import gt_labels, load_scada_window_means

CHANNELS = {
    "power": "1_P_Ist",
    "speed": "1_Drehzahl UPM",
    "head_mws": "Oberwasserdruck_mWs",
    "ow_pegel": "Oberwasserpegel",
    "uw_pegel": "UW_Pegel_Rodund",
    "vane": "1_Leitapparat Stell.",
    "flow_tu": "Durchfluss TU",
    "spiraldruck": "1_Spiraldruck",
    "laufraddruck": "1_Laufraddruck",
    "q_mvar": "1_Q_Ist",
    "erregerstrom": "1_Erregerstrom",
}
RUNS = ["290626-tu", "300626-tu", "010726-tu1-morning", "010726-tu2", "010726-tu_ph_tu"]

cfg = load_config()
idx = discover(cfg.data_root)
runs = {r.name: r for r in idx.runs}


def bd_for(name: str) -> list[Path]:
    day = name.split("-")[0]
    for k in idx.betriebsdaten_by_day:
        if day in str(k):
            return idx.betriebsdaten_by_day[k]
    raise KeyError(name)


frames = {}
for name in RUNS:
    prep = prepare_run(run=runs[name], variant="fusion", cfg=cfg)
    df = load_scada_window_means(bd_for(name), prep.grid, CHANNELS)
    gt = gt_labels(
        load_scada_window_means(bd_for(name), prep.grid), cfg.gt,
        window_s=cfg.window.window_s,
    )
    df["state"] = gt["state"].to_numpy()
    frames[name] = df[df["state"] == "turbine"]

cols = [c for c in CHANNELS if c != "power"] + ["power"]
print("=== Turbinen-Fenster: Median pro Tag (alle Lasten) ===")
rows = []
for name, df in frames.items():
    rows.append({"run": name, "n": len(df), **{c: round(float(df[c].median()), 2) for c in cols}})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== Bei GLEICHER Leistung (220-245 MW): Median pro Tag ===")
rows = []
for name, df in frames.items():
    m = df[(df.power >= 220) & (df.power < 245)]
    if len(m) < 30:
        rows.append({"run": name, "n": len(m)})
        continue
    rows.append({"run": name, "n": len(m), **{c: round(float(m[c].median()), 2) for c in cols}})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== Leistungsverteilung Turbine (Perzentile) ===")
for name, df in frames.items():
    q = np.percentile(df.power.dropna(), [5, 25, 50, 75, 95])
    print(
        f"{name:22} P5={q[0]:6.1f}  P25={q[1]:6.1f}  Median={q[2]:6.1f}"
        f"  P75={q[3]:6.1f}  P95={q[4]:6.1f}"
    )
