"""Load-band-stratified per-cell calibration (exploratory): keep the operating
modes, but give turbine windows per-(mode x absolute 5-MW band) references and
conformal thresholds fitted on the commissioning pool, then score 300626-tu in
its own cells. Anchor check: mode-only cells must land near the pooled
protocol's 0.376 frozen reading for the same pool/day, else this harness is
not trusted. Uses only package primitives (prepare_run caches, KnnScorer,
split_by_segments, conformal.calibrate, gt_labels)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from rowii.anomaly.conformal import calibrate
from rowii.anomaly.references import split_by_segments
from rowii.anomaly.scorers import KnnScorer
from rowii.config import load_config
from rowii.io.dataset import discover
from rowii.pipeline import prepare_run
from rowii.scada.labels import gt_labels, load_scada_window_means

POOL = ("010726-tu_ph_tu", "010726-pu", "010726-tu1-morning", "010726-tu2")
TEST = "300626-tu"
ALPHA = 0.05
BIN_MW = 5.0
SEED = 7

cfg = load_config()
idx = discover(cfg.data_root)
runs = {r.name: r for r in idx.runs}


def bd_for(name: str) -> list[Path]:
    day = name.split("-")[0]
    for k in idx.betriebsdaten_by_day:
        if day in str(k):
            return idx.betriebsdaten_by_day[k]
    raise KeyError(name)


def per_run(name: str) -> dict[str, np.ndarray]:
    prep = prepare_run(run=runs[name], variant="fusion", cfg=cfg)
    scada = load_scada_window_means(bd_for(name), prep.grid)
    gt = gt_labels(scada, cfg.gt, window_s=cfg.window.window_s)
    m = prep.valid_mask
    return {
        "F": prep.features[m],
        "seg": prep.segment_ids[m],
        "state": gt["state"].to_numpy()[m],
        "power": scada["power"].to_numpy()[m],
    }


def cells(state: np.ndarray, power: np.ndarray, banded: bool) -> np.ndarray:
    band = np.floor(np.nan_to_num(power, nan=-1e9) / BIN_MW).astype(int)
    return np.array([
        f"{s}@{b}" if (banded and s == "turbine") else str(s)
        for s, b in zip(state, band, strict=True)
    ])


pool = [per_run(n) for n in POOL]
test = per_run(TEST)
F_pool = np.vstack([p["F"] for p in pool])
seg_pool = np.concatenate([p["seg"] + 100_000 * i for i, p in enumerate(pool)])
state_pool = np.concatenate([p["state"] for p in pool])
power_pool = np.concatenate([p["power"] for p in pool])


def evaluate(banded: bool) -> None:
    cp = cells(state_pool, power_pool, banded)
    ct = cells(test["state"], test["power"], banded)
    scored = test["state"] != "transition"
    judged = alarms = unjudged = 0
    rows = []
    for c in np.unique(ct[scored]):
        if c.startswith("transition") or c.startswith("unknown"):
            continue
        ti = np.where((ct == c) & scored)[0]
        pi = np.where(cp == c)[0]
        verdict: str | float | None = None
        try:
            split = split_by_segments(seg_pool[pi], np.ones(len(pi), bool), 0.5, SEED)
            refs = F_pool[pi][split.scoring_windows]
            calib = F_pool[pi][split.calibration_windows]
            scorer = KnnScorer().fit(refs)
            th = calibrate(scorer.score(calib), ALPHA)
            if th.low_confidence:
                raise ValueError("conformal floor")
            s = scorer.score(test["F"][ti])
            a = int((s > th.threshold).sum())
            judged += len(ti)
            alarms += a
            verdict = round(a / len(ti), 3)
        except (ValueError, IndexError) as exc:  # too few pool windows / floor
            unjudged += len(ti)
            verdict = f"unjudged ({exc})"
        rows.append((c, len(pi), len(ti), verdict))
    label = "banded (turbine @ 5-MW cells)" if banded else "mode-only (anchor)"
    print(f"\n=== {label} ===")
    for r in sorted(rows):
        print(f"  {r[0]:<18} pool={r[1]:>6} test={r[2]:>5} -> {r[3]}")
    total = judged + unjudged
    far = alarms / judged if judged else float("nan")
    print(f"  FAR on judged: {far:.3f} | judged {judged}/{total} "
          f"({unjudged} unjudged)")


evaluate(banded=False)
evaluate(banded=True)
