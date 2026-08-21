"""Consolidate the strike registers: merge the bearing-based ST vane assignment
and the two template-confirmed PU landmark singles INTO strikes_register_{st,pu}.csv.

Closes the gap found during packaging (2026-08-19): the registers carried the
17-largest-gaps vane segmentation (9 ST sweep slots unresolved) and left the two
PU landmark anchors blank, while the bearing assignment (bearing_st_sweep.csv)
and the template finds (research note §11) already resolved them. After this:
ST 88 measured + 2 rhythm-inferred, PU 86 measured + 4 rhythm-inferred = 174 + 6.

ST vane slots are rebuilt from the 18 bearing groups (time order = protocol
order, starting at mic 0 deg): per group up to 3 slots, preferring
detector-backed impulses, then the strongest (partner's convention); mark-only
impulses fill remaining slots as annotated-only. vane_18 keeps its
rhythm-inferred #2/#3 rows from the matched-filter verdicts.
Usage: .venv/bin/python scripts/strike_register/consolidate_register.py
"""
from __future__ import annotations

import csv
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import OUTPUT_ROOT  # noqa: E402

HERE = OUTPUT_ROOT


def ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def iso_ms(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="milliseconds")


def consolidate_st() -> list[dict[str, str]]:
    reg = list(csv.DictReader((HERE / "strikes_register_st.csv").open()))
    bearing = list(csv.DictReader((HERE / "bearing_st_sweep.csv").open()))
    raw = [r for r in csv.DictReader((HERE / "raw_st.csv").open())
           if r["label"].startswith("vane")]
    det_times = [(ts(r["t_utc"]), float(r["k"])) for r in raw]

    def det_match(t: float) -> tuple[float, float] | None:
        best = min(det_times, key=lambda d: abs(d[0] - t), default=None)
        return best if best and abs(best[0] - t) <= 0.3 else None

    groups: dict[int, list[dict[str, Any]]] = {}
    for r in bearing:
        m: dict[str, Any] = {"t": ts(r["t_utc"]), "mark_no": r["gt_mark_no"]}
        d = det_match(m["t"])
        m["det_k"] = d[1] if d else None
        m["source"] = ("both" if d and m["mark_no"] else
                       "detector-only" if d else "annotated-only")
        groups.setdefault(int(r["vane_group"]), []).append(m)

    rank = {"both": 0, "detector-only": 1, "annotated-only": 2}
    rows = [r for r in reg if not r["slot"].startswith("vane")]  # keep positions
    inferred = [r for r in reg if r["slot"].startswith("vane")
                and r["source"] == "rhythm-inferred"]
    for vi in range(1, 19):
        grp = sorted(groups.get(vi, []), key=lambda m: m["t"])
        kept = sorted(grp, key=lambda m: (rank[m["source"]], -(m["det_k"] or 0.0)))[:3]
        kept = sorted(kept, key=lambda m: m["t"])
        for no in (1, 2, 3):
            if no <= len(kept):
                m = kept[no - 1]
                rows.append({"slot": f"vane_{vi:02d}", "strike_no": str(no),
                             "t_utc": iso_ms(m["t"]), "source": m["source"],
                             "det_k": f"{m['det_k']:.0f}" if m["det_k"] else "",
                             "gt_mark_no": m["mark_no"] or ""})
            elif vi == 18 and inferred:
                r = dict(inferred.pop(0))
                r["strike_no"] = str(no)
                rows.append(r)
            else:
                rows.append({"slot": f"vane_{vi:02d}", "strike_no": str(no),
                             "t_utc": "", "source": "unresolved",
                             "det_k": "", "gt_mark_no": ""})
    return rows


def consolidate_pu() -> list[dict[str, str]]:
    reg = list(csv.DictReader((HERE / "strikes_register_pu.csv").open()))
    finds = {("landmark-A_kugelschieber", "1"):
             ("2026-07-08T12:49:46.216+00:00", "template-confirmed"),
             ("landmark-C_EG", "1"):
             ("2026-07-08T13:01:19.670+00:00", "template-confirmed")}
    for r in reg:
        key = (r["slot"], r["strike_no"])
        if key in finds and not r["t_utc"]:
            r["t_utc"], r["source"] = finds[key]
    return reg


def write(session: str, rows: list[dict[str, str]]) -> dict[str, int]:
    path = HERE / f"strikes_register_{session}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["slot", "strike_no", "t_utc",
                                           "source", "det_k", "gt_mark_no"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    tally = Counter(r["source"] for r in rows)
    measured = sum(v for k, v in tally.items()
                   if k not in ("unresolved", "rhythm-inferred"))
    print(f"[{session.upper()}] {len(rows)} slots: {dict(tally)} "
          f"-> measured {measured}")
    return tally


if __name__ == "__main__":
    write("st", consolidate_st())
    write("pu", consolidate_pu())
