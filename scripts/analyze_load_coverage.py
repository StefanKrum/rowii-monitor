"""Load-coverage analysis (Stefan's diagnostic-sentinel idea, exploratory):
do the flagged 30 June turbine windows sit in load bands the commissioning
pool never supplied? And could a load-stratified era-B pool (29 June + 1 July)
have covered them?"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from rowii.config import load_config
from rowii.io.dataset import discover, run_utc_offset_ns
from rowii.pipeline import _streams_for_variant, build_run_grid
from rowii.scada.labels import load_scada_window_means

cfg = load_config()
idx = discover(cfg.data_root)
runs = {r.name: r for r in idx.runs}
bd_keys = list(idx.betriebsdaten_by_day.keys())
print("betriebsdaten_by_day keys:", bd_keys)

POOL = ["010726-pu", "010726-tu1-morning", "010726-tu2", "010726-tu_ph_tu"]
TEST = "300626-tu"
ALT = "290626-tu"
EPS = cfg.gt.power_eps_mw if hasattr(cfg.gt, "power_eps_mw") else 1.0
print("power_eps_mw:", EPS)


def bd_for(run_name: str) -> list[Path]:
    day = run_name.split("-")[0]
    for k in bd_keys:
        if day in str(k):
            return idx.betriebsdaten_by_day[k]
    raise KeyError(run_name)


def window_power(run_name: str) -> pd.DataFrame:
    run = runs[run_name]
    offset = run_utc_offset_ns(run)
    grid = build_run_grid(
        run, _streams_for_variant("fusion"), cfg.window.window_s, offset_ns=offset
    )
    df = load_scada_window_means(bd_for(run_name), grid)
    df = df.reset_index(drop=True)
    df["window"] = np.arange(len(df))
    return df


BIN_MW = 5.0

pool_pow = pd.concat([window_power(r) for r in POOL], ignore_index=True)
pool_tu = pool_pow[pool_pow["power"] > EPS]["power"]

test_df = window_power(TEST)
alarms = pd.read_parquet(
    "results/step2/once-calibrated/fusion/monitor/300626-tu/frozen/alarms.parquet"
)[["window", "alarm", "role", "state"]]
test = test_df.merge(alarms, on="window", how="inner")
test_tu = test[(test["power"] > EPS) & (test["role"] == "scored")].copy()

alt_pow = window_power(ALT)
alt_tu = alt_pow[alt_pow["power"] > EPS]["power"]

lo = np.floor(min(pool_tu.min(), test_tu["power"].min(), alt_tu.min()) / BIN_MW) * BIN_MW
hi = np.ceil(max(pool_tu.max(), test_tu["power"].max(), alt_tu.max()) / BIN_MW) * BIN_MW
bins = np.arange(lo, hi + BIN_MW, BIN_MW)

pool_n, _ = np.histogram(pool_tu, bins=bins)
alt_n, _ = np.histogram(alt_tu, bins=bins)
test_tu["bin"] = pd.cut(test_tu["power"], bins=bins, labels=False)

rows = []
for b in range(len(bins) - 1):
    sub = test_tu[test_tu["bin"] == b]
    if len(sub) == 0 and pool_n[b] == 0 and alt_n[b] == 0:
        continue
    rows.append({
        "band_mw": f"{bins[b]:.0f}-{bins[b+1]:.0f}",
        "pool_n": int(pool_n[b]),
        "n_2906": int(alt_n[b]),
        "n_3006": int(len(sub)),
        "flag_rate_3006": round(float(sub["alarm"].mean()), 3) if len(sub) else np.nan,
    })
tab = pd.DataFrame(rows)
print(tab.to_string(index=False))

FLOOR = 19
test_tu["covered"] = test_tu["bin"].map(lambda b: pool_n[int(b)] >= FLOOR if pd.notna(b) else False)
cov = test_tu[test_tu["covered"]]
unc = test_tu[~test_tu["covered"]]
print(f"\n30.06-tu turbine windows (scored): {len(test_tu)}")
print(
    f"  in pool-covered load bands (>= {FLOOR} pool windows): {len(cov)}"
    f"  -> flag rate {cov['alarm'].mean():.3f}"
)
print(f"  in bands NOT covered: {len(unc)}  -> flag rate {unc['alarm'].mean():.3f}")
unc_bins = sorted(test_tu[~test_tu['covered']]['bin'].dropna().unique())
alt_could = sum(alt_n[int(b)] >= FLOOR for b in unc_bins)
print(
    f"  uncovered bands: {len(unc_bins)},"
    f" of which coverable by 29.06-tu (>= {FLOOR}): {alt_could}"
)
