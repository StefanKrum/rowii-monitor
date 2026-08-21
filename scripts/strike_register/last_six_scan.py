"""Pre-registered matched-filter scan for the last 6 of 180 hammer strikes
(2026-07-08 campaign): PU A_kugel #2/#3, PU C_EG #2/#3, ST vane_18 #2/#3.

Method (fixed BEFORE looking at the target windows' scan output):
  1. Session templates exactly as confirm_missing_strikes.build_template does
     (ST: from sweep 'both' strikes; PU: from all 'both'); reuses that
     module's Session/score/envelope_shape.
  2. Scan statistic = template score (score(envelope_shape(t))) on a sliding
     25 ms grid across each pre-registered window. No energy threshold; the
     template correlation IS the detector.
  3. Calibration = the IDENTICAL scan procedure (same window width, same
     25 ms grid, max over the window) run on >=100 strike-free reference
     draws from logged gap minutes of the same session (ST: wall 12:16 /
     12:27; PU: wall 14:53 / 14:58-15:00 -- identical ranges to
     confirm_missing_strikes.negatives()). This is the look-elsewhere-
     corrected null of SCAN MAXIMA. A target peak counts FOUND only if it
     beats the null max; WEAK if it beats the null p99 but not the max;
     otherwise NOT FOUND.
  4. Corroborating (non-gating) evidence per peak: local energy z (MAD
     z-score, same formula as low_threshold_detect, read continuously
     rather than via its grouping step), rhythm offset from the anchor
     modulo the 0.53-0.92 s lattice, and (ST only) level-bearing.
  5. FOUND -> register row updated (source="matched-filter"). WEAK/NOT
     FOUND -> register row source set to "rhythm-inferred", t_utc left
     empty (register has no notes column); the estimate + full evidence go
     to the last_six_verdicts.csv sidecar instead.

Windows and anchors are pre-registered constants below and are NOT widened
after seeing scan output.

Usage: .venv/bin/python scripts/strike_register/last_six_scan.py
"""
from __future__ import annotations

import csv
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bearing_vane_assignment import Bearer  # noqa: E402
from confirm_missing_strikes import Session, build_template, score  # noqa: E402
from paths import OUTPUT_ROOT  # noqa: E402
from repro_bruno_strikes import BAND, EDGE_GUARD_S, ENV_MS  # noqa: E402, F401

# ----------------------------------------------------------------------
# Pre-registered constants (fixed before this script was run on the data)
# ----------------------------------------------------------------------
GRID_STEP_S = 0.025
N_NULL = 150                      # >= 100 required by the pre-registration
RNG_SEED = 20260708

RHYTHM_LO_S, RHYTHM_HI_S, RHYTHM_NOMINAL_S = 0.53, 0.92, 0.75
EXCLUDE_ANCHOR_S = 0.30           # mask radius around the known anchor peak
NMS_RADIUS_S = 0.30               # non-max-suppression radius between peaks
Z_MATCH = 3.0                     # low_threshold_detect-equivalent z, for characterization only
BEARING_EXPECTED_DEG = 275.0
BEARING_TOL_DEG = 25.0

TARGETS: list[dict[str, Any]] = [
    dict(session="pu", slot="landmark-A_kugelschieber", label="PU A_kugel",
         anchor_iso="2026-07-08T12:49:46.216+00:00",
         win_lo_iso="2026-07-08T12:49:43.4+00:00",
         win_hi_iso="2026-07-08T12:49:49.0+00:00"),
    dict(session="pu", slot="landmark-C_EG", label="PU C_EG",
         anchor_iso="2026-07-08T13:01:19.670+00:00",
         win_lo_iso="2026-07-08T13:01:16.9+00:00",
         win_hi_iso="2026-07-08T13:01:22.5+00:00"),
    dict(session="st", slot="vane_18", label="ST vane_18",
         anchor_iso="2026-07-08T10:26:41.144+00:00",
         win_lo_iso="2026-07-08T10:26:38.6+00:00",
         win_hi_iso="2026-07-08T10:26:43.7+00:00",
         bearing=True),
]

# Strike-free reference gap ranges (true-UTC epoch seconds), identical to
# confirm_missing_strikes.negatives() -- already vetted to sit clear of
# every logged protocol / vane-sweep minute of that session.
WALL_FROM_UTC = 7200.0


def _wall(hhmm: str) -> float:
    return datetime.fromisoformat(f"2026-07-08T{hhmm}:00").replace(
        tzinfo=UTC).timestamp()


GAP_RANGES = {
    "st": [(_wall("12:26") + 40 - WALL_FROM_UTC, _wall("12:27") + 55 - WALL_FROM_UTC),
           (_wall("12:16") + 5 - WALL_FROM_UTC, _wall("12:16") + 55 - WALL_FROM_UTC)],
    "pu": [(_wall("14:53") + 5 - WALL_FROM_UTC, _wall("14:53") + 55 - WALL_FROM_UTC),
           (_wall("14:58") + 5 - WALL_FROM_UTC, _wall("14:59") + 55 - WALL_FROM_UTC)],
}

REGISTER_PATH = {"st": OUTPUT_ROOT / "strikes_register_st.csv",
                  "pu": OUTPUT_ROOT / "strikes_register_pu.csv"}


def iso_to_ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def ts_to_iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="milliseconds")


# ----------------------------------------------------------------------
# Scan primitives
# ----------------------------------------------------------------------
def scan_grid(
    ses: Session,
    t_lo: float,
    t_hi: float,
    env_t: np.ndarray,
    spec_t: np.ndarray,
    step: float = GRID_STEP_S,
) -> tuple[list[float], list[float | None]]:
    n = int(round((t_hi - t_lo) / step)) + 1
    times = [t_lo + i * step for i in range(n)]
    scores: list[float | None] = []
    for t in times:
        e, s = ses.envelope_shape(t)
        scores.append(score(e, s, env_t, spec_t))
    return times, scores


def scan_max(
    ses: Session,
    t_lo: float,
    t_hi: float,
    env_t: np.ndarray,
    spec_t: np.ndarray,
    step: float = GRID_STEP_S,
) -> float:
    _, scores = scan_grid(ses, t_lo, t_hi, env_t, spec_t, step)
    valid = [s for s in scores if s is not None]
    return max(valid) if valid else float("-inf")


def null_distribution(ses: Session, session: str, width_s: float, env_t: np.ndarray,
                       spec_t: np.ndarray,
                       n: int, rng: np.random.Generator) -> np.ndarray:
    ranges = GAP_RANGES[session]
    weights = np.array([hi - lo for lo, hi in ranges], dtype=float)
    weights /= weights.sum()
    samples: list[float] = []
    for _ in range(n):
        gi = rng.choice(len(ranges), p=weights)
        lo, hi = ranges[gi]
        start_hi = hi - width_s
        start = float(rng.uniform(lo, start_hi)) if start_hi > lo else lo
        m = scan_max(ses, start, start + width_s, env_t, spec_t)
        if np.isfinite(m):
            samples.append(m)
    return np.array(samples)


def find_peaks(
    times: list[float],
    scores: list[float | None],
    exclude_t: float,
    exclude_radius: float,
    suppress_radius: float,
    top_n: int = 4,
) -> list[tuple[float, float]]:
    """Non-max suppression over the scored grid. Excludes a radius around
    the known anchor time, then greedily picks the highest-scoring
    remaining points, suppressing points within suppress_radius of a pick."""
    idx = [i for i, t in enumerate(times)
           if scores[i] is not None and abs(t - exclude_t) > exclude_radius]
    idx.sort(key=lambda i: cast(float, scores[i]), reverse=True)
    picked: list[tuple[float, float]] = []
    for i in idx:
        t = times[i]
        if any(abs(t - pt) <= suppress_radius for pt, _ in picked):
            continue
        picked.append((t, cast(float, scores[i])))
        if len(picked) >= top_n:
            break
    return picked


# ----------------------------------------------------------------------
# Corroborating evidence (reported, never gates)
# ----------------------------------------------------------------------
def z_profile(
    ses: Session, t_lo: float, dur: float
) -> tuple[np.ndarray | None, float | None]:
    """Same envelope + per-frame MAD z-score formula as
    confirm_missing_strikes.low_threshold_detect, returned as a continuous
    array (not grouped) so any query time has a well-defined z."""
    x = ses.read(t_lo, dur)
    if x is None:
        return None, None
    e = ses.band(x) ** 2
    w = int(ses.sr * ENV_MS / 1000)
    n = e.shape[1] // w
    env = e[:, : n * w].reshape(e.shape[0], n, w).mean(2)
    med = np.median(env, axis=1, keepdims=True)
    mad = np.median(np.abs(env - med), axis=1, keepdims=True) + 1e-30
    k = ((env - med) / mad).max(0)
    guard = int(round(EDGE_GUARD_S * 1000 / ENV_MS))
    if guard:
        k[:guard] = 0.0
    dt = ENV_MS / 1000
    return k, dt


def energy_z_at(
    k_arr: np.ndarray | None, dt: float | None, t_lo: float, t_query: float | None
) -> float:
    if k_arr is None or t_query is None:
        return float("nan")
    i = int(round((t_query - t_lo) / cast(float, dt)))
    i = max(0, min(len(k_arr) - 1, i))
    return float(k_arr[i])


def rhythm_fit(offset_s: float | None) -> tuple[int | None, float | None, bool]:
    """Best-fitting lattice multiple k in {1,2} with per-step spacing inside
    [RHYTHM_LO_S, RHYTHM_HI_S]. Returns (k, per_step, consistent)."""
    if offset_s is None:
        return None, None, False
    best: tuple[int, float, float] | None = None
    for k in (1, 2):
        per_step = abs(offset_s) / k
        if RHYTHM_LO_S <= per_step <= RHYTHM_HI_S:
            resid = abs(per_step - RHYTHM_NOMINAL_S)
            if best is None or resid < best[2]:
                best = (k, per_step, resid)
    if best is None:
        return None, None, False
    k, per_step, _ = best
    return k, per_step, True


def snap_to_lattice(
    offset_s: float | None, nominal: float = RHYTHM_NOMINAL_S, max_k: int = 2
) -> float | None:
    """Fallback prediction for WEAK/NOT-FOUND slots: keep the empirical
    peak's side/rough magnitude, snapped to the nearest nominal step."""
    if offset_s is None:
        return None
    sign = 1.0 if offset_s >= 0 else -1.0
    k = max(1, min(max_k, round(abs(offset_s) / nominal)))
    return sign * k * nominal


def circ_diff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------
# Per-target processing
# ----------------------------------------------------------------------
def process_target(
    tgt: dict[str, Any],
    ses: Session,
    env_t: np.ndarray,
    spec_t: np.ndarray,
    null_samples: np.ndarray,
    bearer: Bearer | None,
) -> dict[str, Any]:
    win_lo, win_hi = iso_to_ts(tgt["win_lo_iso"]), iso_to_ts(tgt["win_hi_iso"])
    anchor = iso_to_ts(tgt["anchor_iso"])
    width = win_hi - win_lo

    times, scores = scan_grid(ses, win_lo, win_hi, env_t, spec_t)
    anchor_e, anchor_s = ses.envelope_shape(anchor)
    anchor_score = score(anchor_e, anchor_s, env_t, spec_t)

    peaks = find_peaks(times, scores, exclude_t=anchor, exclude_radius=EXCLUDE_ANCHOR_S,
                        suppress_radius=NMS_RADIUS_S, top_n=5)
    k_arr, dt = z_profile(ses, win_lo, width)

    null_max = float(null_samples.max())
    null_p99 = float(np.percentile(null_samples, 99))

    slots: list[dict[str, Any]] = []
    for rank in range(2):
        t_peak: float | None
        offset: float | None
        k_fit: int | None
        per_step: float | None
        bearing: float | None
        if rank < len(peaks):
            t_peak, sc = peaks[rank]
            offset = t_peak - anchor
            verdict = ("FOUND" if sc > null_max else
                       "WEAK" if sc > null_p99 else "NOT FOUND")
            k_fit, per_step, consistent = rhythm_fit(offset)
            ez = energy_z_at(k_arr, dt, win_lo, t_peak)
            bearing = bearer.level_azimuth(t_peak) if (bearer and tgt.get("bearing")) else None
            predicted_t = t_peak if verdict == "FOUND" else anchor + cast(
                float, snap_to_lattice(offset))
        else:
            # defensive fallback -- should not trigger in practice (a ~200+
            # point grid always yields >=2 local maxima outside the anchor
            # exclusion zone); if it ever does, fall back to a generic
            # nominal-lattice guess rather than leaving the slot empty.
            t_peak, sc, verdict = None, float("-inf"), "NOT FOUND"
            offset = k_fit = per_step = None
            consistent, ez, bearing = False, float("nan"), None
            predicted_t = anchor + (rank + 1) * RHYTHM_NOMINAL_S
        slots.append(dict(t_peak=t_peak, scan_peak_score=sc, verdict=verdict, offset=offset,
                           k_fit=k_fit, per_step=per_step, rhythm_consistent=consistent,
                           energy_z=ez, bearing=bearing, predicted_t=predicted_t))

    # order by whichever time we actually have (peak time, else predicted)
    def sort_key(s: dict[str, Any]) -> float:
        return cast(float, s["t_peak"] if s["t_peak"] is not None else s["predicted_t"])
    slots.sort(key=sort_key)

    return dict(tgt=tgt, win_lo=win_lo, win_hi=win_hi, anchor=anchor,
                anchor_score=anchor_score, peaks=peaks, null_max=null_max,
                null_p99=null_p99, null_n=len(null_samples), slots=slots)


# ----------------------------------------------------------------------
# Register I/O
# ----------------------------------------------------------------------
def load_rows(session: str) -> tuple[Sequence[str] | None, list[dict[str, str]]]:
    with REGISTER_PATH[session].open(newline="") as fh:
        r = csv.DictReader(fh)
        fieldnames = r.fieldnames
        rows = list(r)
    return fieldnames, rows


def write_rows(session: str, fieldnames: Sequence[str], rows: list[dict[str, str]]) -> None:
    with REGISTER_PATH[session].open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def apply_register_update(
    fieldnames: Sequence[str] | None,
    rows: list[dict[str, str]],
    slot: str,
    strike_no: int,
    verdict: str,
    t_utc_iso: str,
) -> None:
    updated = False
    for row in rows:
        if row["slot"] == slot and row["strike_no"] == str(strike_no):
            assert row["source"] == "unresolved", (
                f"expected unresolved row for {slot}#{strike_no}, got source={row['source']!r}")
            if verdict == "FOUND":
                row["t_utc"] = t_utc_iso
                row["source"] = "matched-filter"
                row["det_k"] = ""
                row["gt_mark_no"] = ""
            else:
                row["t_utc"] = ""
                row["source"] = "rhythm-inferred"
                row["det_k"] = ""
                row["gt_mark_no"] = ""
            updated = True
            break
    if not updated:
        raise KeyError(f"no unresolved row found for {slot}#{strike_no}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    sessions: dict[str, Session] = {}
    templates: dict[str, tuple[Any, ...]] = {}
    for s in ("st", "pu"):
        ses = Session(s.upper())
        fieldnames, rows = load_rows(s)
        kinds = ("vane",) if s == "st" else None
        env_t, spec_t, selfs = build_template(ses, rows, kinds_filter=kinds)
        sessions[s] = ses
        templates[s] = (env_t, spec_t, selfs, fieldnames, rows)
        print(f"[{s.upper()}] template from {len(selfs)} 'both' strikes "
              f"(self-score p5={np.percentile(selfs, 5):.3f}, median={np.median(selfs):.3f})")

    bearer = Bearer()

    # one null distribution per (session, window width) pair -- PU A_kugel
    # and PU C_EG share width 5.6 s, so PU needs only one null run.
    null_cache: dict[tuple[str, float], np.ndarray] = {}
    print(f"\nbuilding null distributions (n={N_NULL} draws each, 25 ms grid, "
          f"identical scan procedure)...")
    for tgt in TARGETS:
        s = tgt["session"]
        width = round(iso_to_ts(tgt["win_hi_iso"]) - iso_to_ts(tgt["win_lo_iso"]), 3)
        key = (s, width)
        if key in null_cache:
            continue
        ses = sessions[s]
        env_t, spec_t, *_ = templates[s]
        samples = null_distribution(ses, s, width, env_t, spec_t, N_NULL, rng)
        null_cache[key] = samples
        print(f"  [{s.upper()} width={width:.3f}s] n_valid={len(samples)}  "
              f"mean={samples.mean():.3f} sd={samples.std():.3f}  "
              f"p99={np.percentile(samples, 99):.3f}  max={samples.max():.3f}")

    results: list[dict[str, Any]] = []
    print("\nscanning pre-registered target windows...")
    for tgt in TARGETS:
        s = tgt["session"]
        ses = sessions[s]
        env_t, spec_t, *_ = templates[s]
        width = round(iso_to_ts(tgt["win_hi_iso"]) - iso_to_ts(tgt["win_lo_iso"]), 3)
        null_samples = null_cache[(s, width)]
        res = process_target(tgt, ses, env_t, spec_t, null_samples, bearer)
        results.append(res)

        print(f"\n-- {tgt['label']} ({tgt['slot']}) window "
              f"[{tgt['win_lo_iso']} .. {tgt['win_hi_iso']}]  width={width:.3f}s")
        print(f"   anchor {tgt['anchor_iso']}  anchor_score={res['anchor_score']:.3f}  "
              f"(sanity check; template match at the known strike)")
        print(f"   null: max={res['null_max']:.3f}  p99={res['null_p99']:.3f}  n={res['null_n']}")
        print(f"   top non-anchor peaks (post-NMS, {NMS_RADIUS_S}s radius, "
              f"{EXCLUDE_ANCHOR_S}s anchor exclusion):")
        for t, sc in res["peaks"]:
            off = t - res["anchor"]
            print(f"     t={ts_to_iso(t)}  score={sc:+.3f}  offset={off:+.3f}s")
        for i, sl in enumerate(res["slots"], start=2):
            tp = ts_to_iso(sl["t_peak"]) if sl["t_peak"] is not None else "(none)"
            if sl["offset"] is not None:
                print(f"   slot #{i}: peak={tp}  score={sl['scan_peak_score']:+.3f}  "
                      f"verdict={sl['verdict']:9s}  z={sl['energy_z']:.1f}  "
                      f"offset={sl['offset']:+.3f}s")
            else:
                print(f"   slot #{i}: peak={tp}  verdict={sl['verdict']}  "
                      f"(no distinguishable non-anchor peak in window)")

    # -- update registers (FOUND only get a real t_utc; others -> rhythm-inferred)
    print("\napplying register updates...")
    touched = {"st": False, "pu": False}
    for res in results:
        s = res["tgt"]["session"]
        _, _, _, fieldnames, rows = templates[s]
        slot = res["tgt"]["slot"]
        for i, sl in enumerate(res["slots"], start=2):
            t_iso = ts_to_iso(sl["t_peak"]) if sl["verdict"] == "FOUND" else ""
            apply_register_update(fieldnames, rows, slot, i, sl["verdict"], t_iso)
            touched[s] = True
            print(f"   {slot}#{i}: verdict={sl['verdict']} -> "
                  f"source={'matched-filter' if sl['verdict']=='FOUND' else 'rhythm-inferred'}"
                  + (f", t_utc={t_iso}" if t_iso else ", t_utc left empty"))

    for s in ("st", "pu"):
        if touched[s]:
            _, _, _, fieldnames, rows = templates[s]
            write_rows(s, fieldnames, rows)
            print(f"-> wrote {REGISTER_PATH[s].name}")

    # -- sidecar verdicts CSV
    sidecar = OUTPUT_ROOT / "last_six_verdicts.csv"
    with sidecar.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slot", "verdict", "predicted_t_utc", "scan_peak_score",
                    "null_max", "null_p99", "energy_z", "bearing_deg", "rhythm_offset_s"])
        for res in results:
            slot = res["tgt"]["slot"]
            for i, sl in enumerate(res["slots"], start=2):
                pred = ts_to_iso(sl["predicted_t"]) if sl["predicted_t"] is not None else ""
                bdeg = f"{sl['bearing']:.1f}" if sl["bearing"] is not None else ""
                off = f"{sl['offset']:+.3f}" if sl["offset"] is not None else ""
                sc = f"{sl['scan_peak_score']:.3f}" if np.isfinite(sl["scan_peak_score"]) else ""
                w.writerow([f"{slot}#{i}", sl["verdict"], pred, sc,
                            f"{res['null_max']:.3f}", f"{res['null_p99']:.3f}",
                            f"{sl['energy_z']:.1f}" if np.isfinite(sl["energy_z"]) else "",
                            bdeg, off])
    print(f"-> wrote {sidecar.name}")

    # -- compact final table
    print("\n" + "=" * 78)
    print("FINAL VERDICT TABLE")
    print("=" * 78)
    hdr = (f"{'slot':28s} {'verdict':10s} {'time (found/predicted)':30s} "
           f"{'score':>7s} {'rhythm':>8s} {'bearing':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for res in results:
        slot = res["tgt"]["label"]
        for i, sl in enumerate(res["slots"], start=2):
            t_show = sl["t_peak"] if sl["t_peak"] is not None else sl["predicted_t"]
            t_str = ts_to_iso(t_show) if t_show is not None else "n/a"
            rh = (f"k={sl['k_fit']} {sl['per_step']:.2f}s" if sl["rhythm_consistent"]
                  else "no fit")
            if sl["bearing"] is not None:
                bok = abs(circ_diff(sl["bearing"], BEARING_EXPECTED_DEG)) <= BEARING_TOL_DEG
                bz = f"{sl['bearing']:.0f}°{'ok' if bok else 'off'}"
            else:
                bz = "-"
            sc = f"{sl['scan_peak_score']:+.3f}" if np.isfinite(sl["scan_peak_score"]) else "n/a"
            print(f"{slot + ' #' + str(i):28s} {sl['verdict']:10s} {t_str:30s} "
                  f"{sc:>7s} {rh:>8s} {bz:>8s}")


if __name__ == "__main__":
    main()
