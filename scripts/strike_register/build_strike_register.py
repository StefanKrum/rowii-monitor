"""Canonical 180-strike register for 08.07.2026: one row per PROTOCOL strike.

Fills the protocol's 90 slots per session (12 positions x 3 + 18 vanes x 3)
from two independent sources on the true-UTC axis:
  - Stefan's per-strike annotation (docs/groundtruth/080726_strikes_seconds_*.csv)
  - the reproduced partner detector (OUTPUT_ROOT/{raw,protocol}_*.csv, written
    by repro_bruno_strikes.py)

Slot filling:
  positions : the detector's protocol view (pick_triplet, <=3 per position)
              matched to marks within +/-0.3 s; unmatched marks fill remaining
              slots (annotated-only), unmatched detector picks fill as
              detector-only; still-empty slots = unresolved.
  vanes     : 18 groups from the 17 largest gaps between consecutive sweep
              marks (protocol: vane-by-vane starting at mic 0 deg). Per vane
              up to 3 intentional strikes, detector-matched marks preferred;
              vanes with <3 resolvable strikes leave unresolved slots. Extra
              marks beyond 3 per vane are counted as bounces, not slotted.

Output: OUTPUT_ROOT/strikes_register_{st,pu}.csv + console tally.
Usage:  .venv/bin/python scripts/strike_register/build_strike_register.py [ST PU]
"""
from __future__ import annotations

import csv
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import GROUNDTRUTH, OUTPUT_ROOT  # noqa: E402

HERE = OUTPUT_ROOT
GT = GROUNDTRUTH

MATCH_TOL_S = 0.3
POSITIONS = {
    "ST": ["landmark-C_EG", "plate-gen_0", "plate-gen_90", "plate-gen_180",
           "plate-gen_270", "plate-tur_bottom", "landmark-B_11TG",
           "landmark-A_kugelschieber", "plate-tur_0", "plate-tur_90",
           "plate-tur_180", "plate-tur_270"],
    "PU": ["plate-gen_0", "plate-gen_90", "plate-gen_180", "plate-gen_270",
           "plate-tur_bottom", "landmark-B_11TG", "landmark-A_kugelschieber",
           "plate-tur_0", "plate-tur_90", "plate-tur_180", "plate-tur_270",
           "landmark-C_EG"],
}
#: detector label -> GT kind
DET2KIND = {"C_EG": "landmark-C_EG", "B_11TG": "landmark-B_11TG",
            "A_kugel": "landmark-A_kugelschieber",
            **{f"gen_{a}": f"plate-gen_{a}" for a in (0, 90, 180, 270)},
            **{f"tur_{a}": f"plate-tur_{a}" for a in (0, 90, 180, 270)},
            "tur_bottom": "plate-tur_bottom"}


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="milliseconds")


def load_marks(session: str) -> list[dict[str, Any]]:
    lines = [line for line in (GT / f"080726_strikes_seconds_{session.lower()}.csv").open()
             if not line.startswith("#")]
    out: list[dict[str, Any]] = []
    for r in csv.DictReader(lines):
        out.append({"kind": r["kind"], "no": int(r["strike_no"]),
                    "t": datetime.fromisoformat(r["strike_utc"]).timestamp()})
    return out


def load_detector(session: str, which: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in csv.DictReader((HERE / f"{which}_{session.lower()}.csv").open()):
        out.append({"label": r["label"], "t": datetime.fromisoformat(r["t_utc"]).timestamp(),
                    "k": float(r["k"])})
    return out


def pair(
    marks: list[dict[str, Any]], dets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Unique greedy 1:1 pairing within MATCH_TOL_S; returns list of slots
    sorted by time: {'t','source','det_k','mark_no'}."""
    cand = sorted((abs(m["t"] - d["t"]), i, j) for i, m in enumerate(marks)
                  for j, d in enumerate(dets) if abs(m["t"] - d["t"]) <= MATCH_TOL_S)
    um: set[int] = set()
    ud: set[int] = set()
    slots: list[dict[str, Any]] = []
    for _, i, j in cand:
        if i in um or j in ud:
            continue
        um.add(i)
        ud.add(j)
        slots.append({"t": marks[i]["t"], "source": "both",
                      "det_k": dets[j]["k"], "mark_no": marks[i]["no"]})
    for i, m in enumerate(marks):
        if i not in um:
            slots.append({"t": m["t"], "source": "annotated-only",
                          "det_k": None, "mark_no": m["no"]})
    for j, d in enumerate(dets):
        if j not in ud:
            slots.append({"t": d["t"], "source": "detector-only",
                          "det_k": d["k"], "mark_no": None})
    return sorted(slots, key=lambda s: s["t"])


def top3(slots: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep <=3 intentional strikes: prefer both/detector-backed, then order
    of occurrence; return (kept sorted by time, n_extra)."""
    if len(slots) <= 3:
        return slots, 0
    rank = {"both": 0, "detector-only": 1, "annotated-only": 2}
    kept = sorted(slots, key=lambda s: (rank[s["source"]], -(s["det_k"] or 0.0)))[:3]
    return sorted(kept, key=lambda s: s["t"]), len(slots) - 3


def vane_groups(marks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """18 groups from the 17 largest inter-mark gaps (order preserved)."""
    ts = sorted(m["t"] for m in marks)
    gaps = np.diff(ts)
    cut_idx = set(np.argsort(gaps)[-17:])
    groups: list[list[dict[str, Any]]] = []
    cur = [marks[0]]
    marks_sorted = sorted(marks, key=lambda m: m["t"])
    cur = [marks_sorted[0]]
    for gi, m in enumerate(marks_sorted[1:]):
        if gi in cut_idx:
            groups.append(cur)
            cur = []
        cur.append(m)
    groups.append(cur)
    return groups


def run(session: str) -> list[list[Any]]:
    marks = load_marks(session)
    det_proto = load_detector(session, "protocol")
    det_raw = load_detector(session, "raw")
    rows: list[list[Any]] = []
    tally = {"both": 0, "annotated-only": 0, "detector-only": 0,
             "unresolved": 0, "extras": 0}

    def emit(slot_name: str, kept: list[dict[str, Any]], n_extra: int) -> None:
        tally["extras"] += n_extra
        for no in (1, 2, 3):
            if no <= len(kept):
                s = kept[no - 1]
                tally[s["source"]] += 1
                rows.append([slot_name, no, iso(s["t"]), s["source"],
                             f"{s['det_k']:.0f}" if s["det_k"] else "",
                             s["mark_no"] if s["mark_no"] is not None else ""])
            else:
                tally["unresolved"] += 1
                rows.append([slot_name, no, "", "unresolved", "", ""])

    # -- 12 positions ---------------------------------------------------
    for kind in POSITIONS[session]:
        mk = [m for m in marks if m["kind"] == kind]
        det_label = next(k for k, v in DET2KIND.items() if v == kind)
        dt = [d for d in det_proto if d["label"] == det_label]
        kept, n_extra = top3(pair(mk, dt))
        emit(kind, kept, n_extra)

    # -- 18 vanes -------------------------------------------------------
    sweep_marks = [m for m in marks if m["kind"] == "vane-sweep"]
    sweep_dets = [d for d in det_raw if d["label"].startswith("vane")]
    for vi, grp in enumerate(vane_groups(sweep_marks), 1):
        near = [d for d in sweep_dets
                if grp[0]["t"] - 1.0 <= d["t"] <= grp[-1]["t"] + 1.0]
        kept, n_extra = top3(pair(grp, near))
        emit(f"vane_{vi:02d}", kept, n_extra)

    out = HERE / f"strikes_register_{session.lower()}.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slot", "strike_no", "t_utc", "source", "det_k", "gt_mark_no"])
        w.writerows(rows)
    resolved = 90 - tally["unresolved"]
    print(f"[{session}] 90 protocol slots -> resolved {resolved} "
          f"(both {tally['both']}, annotated-only {tally['annotated-only']}, "
          f"detector-only {tally['detector-only']}), unresolved {tally['unresolved']}, "
          f"bounces/extras not slotted: {tally['extras']}")
    unresolved = [r for r in rows if r[3] == "unresolved"]
    if unresolved:
        from collections import Counter
        print(f"   unresolved slots: {dict(Counter(r[0] for r in unresolved))}")
    print(f"-> {out}")
    return rows


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["ST", "PU"]):
        run(s)
