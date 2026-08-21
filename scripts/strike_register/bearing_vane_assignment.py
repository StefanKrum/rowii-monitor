"""Bearing for the ST vane-sweep impulses -> 18-vane assignment.

Two bearing methods, both CALIBRATED on the 8 ring plates (known azimuths,
detector peak times from the reproduced partner detector):

  level : band-energy amplitude bearing -- 5-20 kHz energy per mic over
          80 ms from peak-10 ms; azimuth = energy^0.5-weighted circular
          mean over the ring mics (near-field: the struck plate sits AT a
          mic position, so amplitude carries the direction). Self-contained,
          the default, and the only method required for this script to run.
  srp   : the partner's SRP-PHAT beamformer (OPTIONAL, read-only import from
          the measurement partner's `hydropower-anomaly` package -- not a
          dependency of this repo). 0.13 s window from peak-10 ms, surfaces
          of both rings averaged, circular mean = azimuth. Enabled only when
          ROWII_BRUNO_SRC points at that package's `src/` directory; if
          unset (or the import fails) this script prints a note and falls
          back to level mode -- it never hard-requires the partner's code.

When both are available, the method with the smaller worst pooled
calibration error bears the sweep; grouping walks marks in time order,
starting a new vane on an azimuth jump > AZ_JUMP_DEG or a time gap > GAP_MAX_S.

Output: OUTPUT_ROOT/bearing_st_sweep.csv + console report.
Usage:  .venv/bin/python scripts/strike_register/bearing_vane_assignment.py
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import GROUNDTRUTH, OUTPUT_ROOT  # noqa: E402
from repro_bruno_strikes import SESSION_DIR, Stream  # noqa: E402

GT_DIR = GROUNDTRUTH

# -- optional SRP-PHAT beamformer (measurement partner's package) ----------
# level mode (below) is self-contained and used by default. srp mode
# additionally needs the partner's `hydropower-anomaly` package: point
# ROWII_BRUNO_SRC at its `src/` directory (and, if the repo's own pydantic/
# PyYAML versions conflict with it, ROWII_BRUNO_PYDEPS at an isolated
# checkout of those deps) to enable it. Neither variable is required.
SRP_AVAILABLE = False
_SRP_IMPORT_ERROR: str | None = None
_bruno_src = os.environ.get("ROWII_BRUNO_SRC")
if _bruno_src:
    _bruno_pydeps = os.environ.get("ROWII_BRUNO_PYDEPS")
    if _bruno_pydeps:
        sys.path.insert(0, _bruno_pydeps)
    sys.path.insert(0, _bruno_src)
    try:
        from hydropower_anomaly.daq_format.datasegment import DataSegment
        from hydropower_anomaly.features.beamforming import SRPPHATBeamformer
        SRP_AVAILABLE = True
    except ImportError as exc:
        _SRP_IMPORT_ERROR = f"import failed ({exc})"
else:
    _SRP_IMPORT_ERROR = "ROWII_BRUNO_SRC not set"

WALL_FROM_UTC = 7200.0
PRE_S = 0.01
SRP_WIN_S = 0.12
LEVEL_WIN_S = 0.08
BAND = (5000, 20000)
AZ_JUMP_DEG = 14.0
GAP_MAX_S = 30.0
RING_ANGLES = np.array([0.0, 90.0, 180.0, 270.0])

EXPECTED = {"gen_0": 0.0, "gen_90": 90.0, "gen_180": 180.0, "gen_270": 270.0,
            "tur_0": 0.0, "tur_90": 90.0, "tur_180": 180.0, "tur_270": 270.0}


def circ_diff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def circ_mean(deg: np.ndarray, w: np.ndarray) -> float:
    z = (w * np.exp(1j * np.radians(deg))).sum()
    return float(np.degrees(np.angle(z)) % 360.0)


class Bearer:
    def __init__(self) -> None:
        self.bf: Any = SRPPHATBeamformer(n_azimuth=72) if SRP_AVAILABLE else None
        self.angles = np.radians(np.linspace(0, 360, 72, endpoint=False))
        self.gen = Stream(SESSION_DIR["ST"], "RAWGeneratorMic__0")
        self.tur = Stream(SESSION_DIR["ST"], "RAWTurbineMic__1")
        self.sos: np.ndarray | None = None

    def window(self, t_wall: float, dur: float) -> tuple[np.ndarray, int] | None:
        g = next(iter(self.gen.chunks(t_wall, dur + 1.0)), None)
        t = next(iter(self.tur.chunks(t_wall, dur + 1.0)), None)
        if g is None or t is None:
            return None
        (g0, sr, gd), (t0, _, td) = g, t
        n = int(dur * sr)
        gi, ti = int(round((t_wall - g0) * sr)), int(round((t_wall - t0) * sr))
        if gi < 0 or ti < 0 or gi + n > gd.shape[1] or ti + n > td.shape[1]:
            return None
        return np.vstack([gd[:, gi:gi + n], td[:, ti:ti + n]]), sr

    def srp_surface(self, t_utc: float) -> np.ndarray | None:
        if not SRP_AVAILABLE:
            raise RuntimeError(
                "SRP mode unavailable: the measurement partner's package is "
                f"not importable ({_SRP_IMPORT_ERROR}). Set ROWII_BRUNO_SRC "
                "to enable it, or use level mode (the default).")
        win = self.window(t_utc + WALL_FROM_UTC - PRE_S, SRP_WIN_S)
        if win is None:
            return None
        x, sr = win
        seg = DataSegment.from_arrays(
            x, np.zeros((12, int(SRP_WIN_S * 10000))),
            datetime.fromtimestamp(t_utc + WALL_FROM_UTC - PRE_S, UTC), mic_sr=sr)
        f = self.bf.extract(seg).features
        return np.mean([np.asarray(f[f"srp_phat_{lvl}"], dtype=float)
                        for lvl in ("generator", "turbine")], axis=0)

    def srp_azimuth(self, t_utc: float) -> float | None:
        s = self.srp_surface(t_utc)
        return None if s is None else circ_mean(np.degrees(self.angles), s)

    def level_azimuth(self, t_utc: float) -> float | None:
        """Energy-weighted circular mean over the 8 ring mics (both rings)."""
        win = self.window(t_utc + WALL_FROM_UTC - PRE_S, LEVEL_WIN_S)
        if win is None:
            return None
        x, sr = win
        if self.sos is None:
            self.sos = butter(4, list(BAND), btype="band", fs=sr, output="sos")
        e = np.array([(sosfilt(self.sos, x[ch]) ** 2).sum() for ch in range(9)])
        ring_e = e[[0, 1, 2, 3]] + e[[4, 5, 6, 7]]      # pool the two rings per angle
        return circ_mean(RING_ANGLES, np.sqrt(ring_e))


def load_position_times() -> dict[str, list[float]]:
    """Detector peak times (protocol view) for the 8 ring plates."""
    out: dict[str, list[float]] = {}
    for r in csv.DictReader((OUTPUT_ROOT / "protocol_st.csv").open()):
        if r["label"] in EXPECTED:
            out.setdefault(r["label"], []).append(
                datetime.fromisoformat(r["t_utc"]).timestamp())
    return out


def load_sweep() -> list[dict[str, Any]]:
    """Sweep impulses: detector raw times (peak-referenced); GT mark_no via
    +/-0.3 s pairing for traceability (unmatched detector impulses keep '')."""
    dets: list[dict[str, Any]] = [
        {"t": datetime.fromisoformat(r["t_utc"]).timestamp(), "det": True,
         "k": float(r["k"])}
        for r in csv.DictReader((OUTPUT_ROOT / "raw_st.csv").open())
        if r["label"].startswith("vane")]
    gt = GT_DIR / "080726_strikes_seconds_st.csv"
    marks: list[dict[str, Any]] = []
    for r in csv.DictReader([line for line in gt.open() if not line.startswith("#")]):
        if r["kind"] == "vane-sweep":
            marks.append({"no": r["strike_no"],
                          "t": datetime.fromisoformat(r["strike_utc"]).timestamp()})
    cand = sorted((abs(m["t"] - d["t"]), i, j) for i, m in enumerate(marks)
                  for j, d in enumerate(dets) if abs(m["t"] - d["t"]) <= 0.3)
    um: set[int] = set()
    ud: set[int] = set()
    for _, i, j in cand:
        if i in um or j in ud:
            continue
        um.add(i)
        ud.add(j)
        dets[j]["no"] = marks[i]["no"]
    # marks the detector missed still deserve a bearing (mark time as onset)
    extra: list[dict[str, Any]] = [
        {"t": m["t"], "no": m["no"], "det": False, "k": 0.0}
        for i, m in enumerate(marks) if i not in um]
    return sorted(dets + extra, key=lambda d: d["t"])


def main() -> None:
    b = Bearer()
    pos_times = load_position_times()
    print("== calibration on the 8 ring plates (detector peak times) ==")
    methods = [("level", b.level_azimuth)]
    if SRP_AVAILABLE:
        methods.insert(0, ("srp", b.srp_azimuth))
    else:
        print(f"(SRP mode skipped: {_SRP_IMPORT_ERROR} -- "
              "level mode is self-contained and used by default)")
    hdr = "  ".join(f"{name:>7s}" for name, _ in methods)
    print(f"{'plate':9s} {hdr}   (expected)")
    worst = {name: 0.0 for name, _ in methods}
    for label, expected in EXPECTED.items():
        errs: dict[str, float] = {}
        for name, fn in methods:
            azs: list[float | None] = [fn(t) for t in pos_times.get(label, [])]
            azs = [a for a in azs if a is not None]
            if not azs:
                errs[name] = float("nan")
                continue
            pooled = circ_mean(np.array(azs), np.ones(len(azs)))
            errs[name] = circ_diff(pooled, expected)
            worst[name] = max(worst[name], abs(errs[name]))
        row = "  ".join(f"{errs[name]:+7.1f}" for name, _ in methods)
        print(f"{label:9s} {row}   ({expected:.0f})")
    method = min(worst, key=lambda k: worst[k])
    summary = " | ".join(f"{name} {worst[name]:.1f}" for name, _ in methods)
    print(f"worst pooled error: {summary} -> using '{method}'")
    if worst[method] > 30.0:
        print("ABORT: no bearing method reaches +/-30 deg on the calibration plates")
        return

    fn = dict(methods)[method]
    sweep = load_sweep()
    for m in sweep:
        m["az"] = fn(m["t"])
    ok = [m for m in sweep if m["az"] is not None]
    print(f"\n== sweep: {len(ok)}/{len(sweep)} impulses beared ({method}) ==")

    # -- stage 1: LOUD impulses only (k >= K_MIN: bearing is stable) form
    #    time bursts (gap >= 1.5 s); azimuth pooled per burst ---------------
    K_MIN = 500.0
    loud = [m for m in ok if m.get("k", 0.0) >= K_MIN]
    quiet = [m for m in ok if m.get("k", 0.0) < K_MIN]
    bursts: list[list[dict[str, Any]]] = []
    for m in loud:
        if bursts and m["t"] - bursts[-1][-1]["t"] < 1.5:
            bursts[-1].append(m)
        else:
            bursts.append([m])

    def burst_az(grp: list[dict[str, Any]]) -> float:
        return circ_mean(np.array([x["az"] for x in grp]), np.ones(len(grp)))

    # -- stage 2: merge consecutive bursts of the SAME vane (bounce pauses):
    #    same azimuth within AZ_MERGE_DEG and gap below GAP_MAX_S ----------
    AZ_MERGE_DEG = 11.0
    groups: list[list[dict[str, Any]]] = [list(bursts[0])]
    for grp in bursts[1:]:
        prev = groups[-1]
        if (abs(circ_diff(burst_az(grp), burst_az(prev))) <= AZ_MERGE_DEG
                and grp[0]["t"] - prev[-1]["t"] <= GAP_MAX_S):
            prev.extend(grp)
        else:
            groups.append(list(grp))
    # -- stage 3: quiet impulses join the nearest group in TIME (<=20 s),
    #    they never define groups (their bearing is noise) -----------------
    n_orphan = 0
    for m in quiet:
        best: list[dict[str, Any]] | None = None
        best_dt = 20.0
        for grp in groups:
            dt = min(abs(m["t"] - grp[0]["t"]), abs(m["t"] - grp[-1]["t"]))
            if dt < best_dt:
                best, best_dt = grp, dt
        if best is None:
            n_orphan += 1
        else:
            best.append(m)
            best.sort(key=lambda x: x["t"])
    print(f"loud impulses (k>={K_MIN:.0f}): {len(loud)}, quiet/mark-only: {len(quiet)} "
          f"({n_orphan} orphans, listed as noise)")
    print(f"loud time bursts: {len(bursts)} -> after same-azimuth merge "
          f"(<= {AZ_MERGE_DEG} deg): {len(groups)} vane groups (target 18); "
          f"sizes {[len(g) for g in groups]}")
    with (OUTPUT_ROOT / "bearing_st_sweep.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["vane_group", "gt_mark_no", "t_utc", "azimuth_deg", "method"])
        for vi, grp in enumerate(groups, 1):
            for m in grp:
                w.writerow([vi, m.get("no", ""),
                            datetime.fromtimestamp(m["t"], UTC)
                            .isoformat(timespec="milliseconds"),
                            f"{m['az']:.1f}", method])
    prev_az: float | None = None
    for vi, grp in enumerate(groups, 1):
        az = circ_mean(np.array([m["az"] for m in grp]), np.ones(len(grp)))
        step = f" step={circ_diff(az, prev_az):+6.1f}" if prev_az is not None else ""
        prev_az = az
        t0 = datetime.fromtimestamp(grp[0]["t"], UTC).strftime("%H:%M:%S")
        print(f"  vane {vi:2d}: n={len(grp)} t={t0} az={az:6.1f}{step}")
    print("-> bearing_st_sweep.csv")


if __name__ == "__main__":
    main()
