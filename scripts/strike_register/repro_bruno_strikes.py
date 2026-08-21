"""Reproduce the partner's 08.07.2026 strike hit-timestamps from OUR raw files.

Method transcribed 1:1 from sensing-group/hydropower-anomaly
`tools/run_080726_strike_analysis.py` + `tools/crosscheck_080726_groundtruth.py`
(read-only reference clone, 2026-08-18):
  detector  : 5-20 kHz band, 4th-order Butterworth SOS, energy envelope in
              2 ms frames, per-channel MAD z-score, max over the 9 mics,
              threshold k > 30, impulse groups merged within 0.25 s
  positions : per logged minute keep the 3 strongest impulses inside one
              10 s span (`pick_triplet`) = the protocol's "3 Schlaege"
  vane sweep: every impulse in the 3-min window (-10 s / +20 s pad),
              clusters separated by gaps > 1.5 s = one guide vane each

Time axes (pinned by docs/groundtruth + Arne's official log, 2026-08-18):
  wall  = CEST wall clock = DAQ-embedded timestamps (ns since 2000-01-01,
          epoch-corrected) = filename times = partner's PROTOCOL minutes
  utc   = wall - 2 h   (the ground-truth CSVs' axis)
  Arne's log = wall + 2 h (DAQ display double-converted; -2 h recovers wall)

Validation target: the per-minute counts in
data/illwerke-080726/crosscheck-notes-v1.md (partner's scan, 2026-07-13).

Outputs -> OUTPUT_ROOT (env ROWII_STRIKE_OUT, default
<repo>/results/strike-register/)/{raw,protocol}_{st,pu}.csv + console report.
"""
from __future__ import annotations

import csv
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_ROOT, GROUNDTRUTH, OUTPUT_ROOT, ensure_rowii_importable  # noqa: E402

ensure_rowii_importable()
from rowii.io.gantner import _parse_header_region, read_header  # noqa: E402

DATA = DATA_ROOT
GT_DIR = GROUNDTRUTH
OUT = OUTPUT_ROOT
DAY = "2026-07-08"

GANTNER_EPOCH_S = 946_684_800.0          # 2000-01-01T00:00:00Z as unix seconds
WALL_TO_UTC_S = -7200.0                  # CEST wall clock -> true UTC

SESSION_DIR = {"ST": DATA / "ST_STRIKES", "PU": DATA / "PU_STRIKES"}

#: partner's PROTOCOL (wall-clock minutes), now confirmed minute-by-minute by
#: Arne's official log (Sensor_Anordnung_15062026.xlsx, log times - 2 h).
PROTOCOL = {
    "ST": [
        ("C_EG", "12:15"), ("gen_0", "12:17"), ("gen_90", "12:18"),
        ("gen_180", "12:19"), ("gen_270", "12:20"), ("tur_bottom", "12:21"),
        ("B_11TG", "12:22"), ("A_kugel", "12:23"), ("tur_0", "12:28"),
        ("tur_90", "12:29"), ("tur_180", "12:30"), ("tur_270", "12:31"),
    ],
    "PU": [
        ("gen_0", "14:43"), ("gen_90", "14:44"), ("gen_180", "14:45"),
        ("gen_270", "14:46"), ("tur_bottom", "14:47"), ("B_11TG", "14:48"),
        ("A_kugel", "14:49"), ("tur_0", "14:54"), ("tur_90", "14:55"),
        ("tur_180", "14:56"), ("tur_270", "14:57"), ("C_EG", "15:01"),
    ],
}
VANE_SWEEP = {"ST": ("12:24", 3), "PU": ("14:50", 3)}

#: partner's crosscheck-notes-v1.md per-minute counts (validation reference)
REF_COUNTS = {
    "ST": {"C_EG": 3, "gen_0": 3, "gen_90": 3, "gen_180": 3, "gen_270": 3,
           "tur_bottom": 2, "B_11TG": 3, "A_kugel": 2, "tur_0": 3,
           "tur_90": 1, "tur_180": 7, "tur_270": 6},
    "PU": {"gen_0": 4, "gen_90": 4, "gen_180": 4, "gen_270": 3,
           "tur_bottom": 4, "B_11TG": 3, "A_kugel": 1, "tur_0": 4,
           "tur_90": 4, "tur_180": 4, "tur_270": 4, "C_EG": 1},
}
REF_SWEEP = {"ST": [17, 33, 18], "PU": [21, 24, 10]}

BAND = (5000, 20000)
ENV_MS = 2.0
MERGE_S = 0.25
K_THRESH = 30.0
#: causal sosfilt rings in at every chunk start; under pump noise that
#: transient exceeds k=30 and fakes an "impulse" at exactly frame 0 (verified:
#: it moves with the read-window start). Guard: first 50 ms of each chunk are
#: excluded from peak search. Set to 0 to reproduce the partner's exact counts
#: (his minute-scan carries the same artifact: the "4th impulses" in PU and
#: the lone A_kugel/C_EG "detections" are edge transients).
EDGE_GUARD_S = 0.05


def wall(hhmm: str) -> float:
    """Wall-clock minute -> epoch seconds on the wall axis (naive-as-UTC)."""
    return datetime.fromisoformat(f"{DAY}T{hhmm}:00").replace(
        tzinfo=UTC).timestamp()


def wall_to_utc_iso(t_wall: float) -> str:
    return datetime.fromtimestamp(t_wall + WALL_TO_UTC_S, UTC).isoformat(
        timespec="milliseconds")


class Stream:
    """Seek-reader over one mic stream's burst files (embedded-timestamp based)."""

    def __init__(self, session_dir: Path, stub: str) -> None:
        self.files: list[dict[str, Any]] = []
        for p in sorted(session_dir.glob(f"{stub}_*.dat")):
            h = read_header(p)
            _, names, _, data_off = _parse_header_region(p)
            t0_wall = h.t0_ns / 1e9 + GANTNER_EPOCH_S
            sr = int(round(h.sample_rate_hz))
            self.files.append({"path": p, "t0": t0_wall, "sr": sr,
                               "n": h.n_frames, "off": data_off,
                               "n_ch": len(names), "names": names})

    def chunks(self, t_wall: float, dur: float) -> Iterator[tuple[float, int, np.ndarray]]:
        """Yield (t_start_wall, sr, data[ch, n]) for every file overlap >= 1 s."""
        t_end = t_wall + dur
        for f in self.files:
            f_end = f["t0"] + f["n"] / f["sr"]
            lo, hi = max(t_wall, f["t0"]), min(t_end, f_end)
            if hi - lo < 1.0:
                continue
            sr, n_ch = f["sr"], f["n_ch"]
            frame = 8 + 4 * n_ch
            i0 = max(0, int(round((lo - f["t0"]) * sr)))
            n = min(f["n"] - i0, int(round((hi - lo) * sr)))
            if n <= 0:
                continue
            arr = np.fromfile(f["path"], dtype=np.uint8,
                              offset=f["off"] + i0 * frame, count=n * frame)
            k = arr.size // frame
            data = (arr[: k * frame].reshape(k, frame)[:, 8:].copy()
                    .view("<f4").reshape(k, n_ch).T)
            yield f["t0"] + i0 / sr, sr, data


def detect_strikes(x: np.ndarray, sr: int, k_thresh: float) -> list[dict[str, float]]:
    """Partner's detector, transcribed verbatim (see module docstring)."""
    sos = butter(4, list(BAND), btype="band", fs=sr, output="sos")
    w = int(sr * ENV_MS / 1000)
    ks: list[np.ndarray] = []
    for ch in range(x.shape[0]):
        e = sosfilt(sos, x[ch]) ** 2
        n = len(e) // w
        env = e[: n * w].reshape(n, w).mean(1)
        med = np.median(env)
        mad = np.median(np.abs(env - med)) + 1e-30
        ks.append((env - med) / mad)
    k = np.array(ks)
    kmax = k.max(0)
    guard = int(round(EDGE_GUARD_S * 1000 / ENV_MS))
    if guard:
        kmax[:guard] = 0.0
    pk = np.where(kmax > k_thresh)[0]
    dt = ENV_MS / 1000
    groups: list[list[np.intp]] = []
    for p in pk:
        if groups and (p - groups[-1][-1]) * dt < MERGE_S:
            groups[-1].append(p)
        else:
            groups.append([p])
    out: list[dict[str, float]] = []
    for g in groups:
        i = g[int(np.argmax(kmax[g]))]
        out.append({"t_rel": float(i * dt), "k": float(kmax[i]),
                    "ch": int(k[:, i].argmax())})
    return out


def pick_triplet(strikes: list[dict[str, float]]) -> list[dict[str, float]]:
    """Partner's 3-strongest-within-10 s selection, transcribed verbatim."""
    if len(strikes) <= 3:
        return sorted(strikes, key=lambda s: s["t_wall"])
    best: list[dict[str, float]] | None = None
    best_score = -1.0
    ts = [s["t_wall"] for s in strikes]
    for i in range(len(strikes)):
        grp = [s for s in strikes if abs(s["t_wall"] - ts[i]) <= 5.0]
        grp = sorted(grp, key=lambda s: -s["k"])[:3]
        score = sum(s["k"] for s in grp)
        if score > best_score:
            best, best_score = grp, score
    return sorted(cast("list[dict[str, float]]", best), key=lambda s: s["t_wall"])


def detect_window(
    gen: Stream, tur: Stream, t_wall: float, dur: float
) -> list[dict[str, float]]:
    """Detect impulses in [t_wall, t_wall+dur) across the 9 mics, absolute times."""
    tur_chunks = list(tur.chunks(t_wall, dur))
    out: list[dict[str, float]] = []
    for g_t0, sr, g_data in gen.chunks(t_wall, dur):
        match: tuple[float, int, np.ndarray] | None = None
        for t_t0, t_sr, t_data in tur_chunks:
            if abs(t_t0 - g_t0) < 5.0:
                match = (t_t0, t_sr, t_data)
                break
        if match is None:
            continue
        t_t0, _, t_data = match
        # align both streams on the later start, trim to common length
        start = max(g_t0, t_t0)
        gi, ti = int(round((start - g_t0) * sr)), int(round((start - t_t0) * sr))
        n = min(g_data.shape[1] - gi, t_data.shape[1] - ti)
        if n < sr:
            continue
        x = np.vstack([g_data[:, gi:gi + n], t_data[:, ti:ti + n]])
        for s in detect_strikes(x, sr, K_THRESH):
            s["t_wall"] = start + s["t_rel"]
            out.append(s)
    out.sort(key=lambda s: s["t_wall"])
    dedup: list[dict[str, float]] = []
    for s in out:
        if dedup and s["t_wall"] - dedup[-1]["t_wall"] < MERGE_S:
            continue
        dedup.append(s)
    return dedup


def cluster_vanes(
    strikes: list[dict[str, float]], gap_s: float = 1.5
) -> list[list[dict[str, float]]]:
    vanes: list[list[dict[str, float]]] = []
    for s in strikes:
        if vanes and s["t_wall"] - vanes[-1][-1]["t_wall"] <= gap_s:
            vanes[-1].append(s)
        else:
            vanes.append([s])
    return vanes


def load_gt_marks(session: str) -> list[dict[str, Any]]:
    path = GT_DIR / f"080726_strikes_seconds_{session.lower()}.csv"
    marks: list[dict[str, Any]] = []
    with path.open() as fh:
        rows = [r for r in fh if not r.startswith("#")]
    for rec in csv.DictReader(rows):
        t = datetime.fromisoformat(rec["strike_utc"]).timestamp()
        marks.append({"t_utc": t, "kind": rec["kind"], "no": rec["strike_no"]})
    return marks


def match_report(
    detections: list[dict[str, float]], marks: list[dict[str, Any]], tol: float
) -> tuple[int, int, list[float]]:
    """Fraction of detections with a GT mark within tol (and vice versa)."""
    det_t = np.array([d["t_wall"] + WALL_TO_UTC_S for d in detections])
    mark_t = np.array([m["t_utc"] for m in marks])
    if not len(det_t) or not len(mark_t):
        return 0, 0, []
    d_hit = sum(np.min(np.abs(mark_t - t)) <= tol for t in det_t)
    m_hit = sum(np.min(np.abs(det_t - t)) <= tol for t in mark_t)
    dts = [float(np.min(np.abs(mark_t - t))) for t in det_t]
    return d_hit, m_hit, dts


def run_session(session: str) -> bool:
    print(f"\n================ session {session} ({SESSION_DIR[session].name}) ================")
    gen = Stream(SESSION_DIR[session], "RAWGeneratorMic__0")
    tur = Stream(SESSION_DIR[session], "RAWTurbineMic__1")
    print(f"files: gen={len(gen.files)} tur={len(tur.files)}  "
          f"(sr={gen.files[0]['sr']} Hz, ch={gen.files[0]['n_ch']}+{tur.files[0]['n_ch']})")

    raw_rows: list[tuple[Any, ...]] = []
    proto_rows: list[tuple[Any, ...]] = []
    # -- positions ------------------------------------------------------
    print(f"{'minute':6s} {'label':11s} {'#det':>4s} {'ref':>4s} {'max_k':>10s}  triplet(kept)")
    ok = True
    for label, hhmm in PROTOCOL[session]:
        det = detect_window(gen, tur, wall(hhmm), 60.0)
        for s in det:
            raw_rows.append((label, s))
        kept = pick_triplet(det)
        for i, s in enumerate(kept, 1):
            proto_rows.append((label, i, s))
        ref = REF_COUNTS[session][label]
        mark = "==" if len(det) == ref else "!!"
        if len(det) != ref:
            ok = False
        mk = f"{max((s['k'] for s in det), default=0):.3g}"
        print(f"{hhmm:6s} {label:11s} {len(det):4d} {ref:4d} {mk:>10s}  "
              f"{len(kept)} kept  {mark}")
    # -- vane sweep -----------------------------------------------------
    start, n_min = VANE_SWEEP[session]
    t0 = wall(start) - 10.0
    dur = n_min * 60.0 + 30.0
    det = detect_window(gen, tur, t0, dur)
    per_min: dict[str, int] = {}
    for s in det:
        m = datetime.fromtimestamp(s["t_wall"], UTC).strftime("%H:%M")
        per_min[m] = per_min.get(m, 0) + 1
    sweep_mins = [datetime.fromisoformat(f"{DAY}T{start}:00") + timedelta(minutes=i)
                  for i in range(n_min)]
    counts = [per_min.get(m.strftime("%H:%M"), 0) for m in sweep_mins]
    print(f"sweep  {start}+{n_min}min: per-minute {counts} (ref {REF_SWEEP[session]}), "
          f"total {len(det)}")
    vanes = cluster_vanes([s for s in det])
    print(f"sweep  clusters (gap>1.5s): {len(vanes)} vanes, sizes "
          f"{[len(v) for v in vanes]}")
    for vi, grp in enumerate(vanes):
        for s in grp:
            raw_rows.append((f"vane_{vi:02d}", s))
        kept = sorted(sorted(grp, key=lambda s: -s["k"])[:3], key=lambda s: s["t_wall"])
        for i, s in enumerate(kept, 1):
            proto_rows.append((f"vane_{vi:02d}", i, s))

    # -- write csvs -----------------------------------------------------
    for name, rows in (("raw", raw_rows), ("protocol", proto_rows)):
        path = OUT / f"{name}_{session.lower()}.csv"
        with path.open("w", newline="") as fh:
            wcsv = csv.writer(fh)
            if name == "raw":
                wcsv.writerow(["label", "t_utc", "t_wall", "k", "loudest_ch"])
                for label, s in rows:
                    wcsv.writerow([label, wall_to_utc_iso(s["t_wall"]),
                                   datetime.fromtimestamp(s["t_wall"], UTC)
                                   .strftime("%H:%M:%S.%f")[:-3],
                                   f"{s['k']:.1f}", s["ch"]])
            else:
                wcsv.writerow(["label", "strike_no", "t_utc", "t_wall", "k", "loudest_ch"])
                for label, i, s in rows:
                    wcsv.writerow([label, i, wall_to_utc_iso(s["t_wall"]),
                                   datetime.fromtimestamp(s["t_wall"], UTC)
                                   .strftime("%H:%M:%S.%f")[:-3],
                                   f"{s['k']:.1f}", s["ch"]])
        print(f"-> {path}  ({len(rows)} rows)")

    # -- against our annotated GT --------------------------------------
    marks = load_gt_marks(session)
    dets = [s for _, s in raw_rows]
    protos = [s for _, _, s in proto_rows]
    for tag, dset in (("raw", dets), ("protocol", protos)):
        for tol in (0.3, 1.0):
            d_hit, m_hit, dts = match_report(dset, marks, tol)
            med = np.median(dts) if dts else float("nan")
            print(f"GT match [{tag:8s}] tol +/-{tol}s: detector->marks "
                  f"{d_hit}/{len(dset)}, marks->detector {m_hit}/{len(marks)}"
                  f"  (median |dt| {med:.3f}s)")
    n_pos = sum(1 for label, *_ in proto_rows if not label.startswith("vane"))
    n_vane = len(proto_rows) - n_pos
    print(f"protocol view: {n_pos} position strikes + {n_vane} vane strikes "
          f"= {len(proto_rows)}  (protocol target 90)")
    return ok


def main() -> None:
    sessions = sys.argv[1:] or ["ST", "PU"]
    OUT.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        run_session(session)


if __name__ == "__main__":
    main()
