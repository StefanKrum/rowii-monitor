"""Rhythm-guided low-threshold search + template confirmation (steps a+b).

Template (per session, from the double-confirmed 'both' strikes): the median
normalized 5-20 kHz envelope shape (-20..+180 ms around the peak) and median
normalized log band spectrum (40 ms), on each strike's loudest channel.
Candidates score cosine similarity against both; negatives are drawn from
strike-free gap seconds of the same session (noise-calibrated).

Searches:
  ST vane_18 : +/-6 s around the lone 275-deg impulse (10:26:41.144 UTC),
               detector threshold lowered to z>=5, rhythm check 0.4-1.2 s.
  PU A_kugel / C_EG : full logged minute, z>=4.5, triplet rhythm constraint.
  Hardening : every annotated-only mark is scored against the template.

Verdict per candidate: energy z, envelope cos, spectrum cos, and whether it
clears the negative distribution (score > max of 200 noise draws).
Usage: .venv/bin/python scripts/strike_register/confirm_missing_strikes.py
"""
from __future__ import annotations

import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import GROUNDTRUTH, OUTPUT_ROOT  # noqa: E402
from repro_bruno_strikes import (  # noqa: E402
    BAND,
    EDGE_GUARD_S,
    ENV_MS,
    MERGE_S,
    SESSION_DIR,
    Stream,
    wall,
)

GT = GROUNDTRUTH
WALL_FROM_UTC = 7200.0
ENV_PRE_S, ENV_POST_S = 0.02, 0.18
SPEC_WIN_S = 0.04
RNG = np.random.default_rng(80726)


class Session:
    def __init__(self, session: str):
        self.name = session
        self.gen = Stream(SESSION_DIR[session], "RAWGeneratorMic__0")
        self.tur = Stream(SESSION_DIR[session], "RAWTurbineMic__1")
        self.sos = None
        self.sr = self.gen.files[0]["sr"]

    def read(self, t_utc: float, dur: float):
        """9-channel window at true-UTC start (None if not fully readable)."""
        t_wall = t_utc + WALL_FROM_UTC
        g = next(iter(self.gen.chunks(t_wall, dur + 1.0)), None)
        t = next(iter(self.tur.chunks(t_wall, dur + 1.0)), None)
        if g is None or t is None:
            return None
        (g0, sr, gd), (t0, _, td) = g, t
        n = int(dur * sr)
        gi, ti = int(round((t_wall - g0) * sr)), int(round((t_wall - t0) * sr))
        if gi < 0 or ti < 0 or gi + n > gd.shape[1] or ti + n > td.shape[1]:
            return None
        return np.vstack([gd[:, gi:gi + n], td[:, ti:ti + n]])

    def band(self, x: np.ndarray) -> np.ndarray:
        if self.sos is None:
            self.sos = butter(4, list(BAND), btype="band", fs=self.sr, output="sos")
        return sosfilt(self.sos, x)

    def envelope_shape(self, t_utc: float):
        """Normalized envelope (2 ms frames) around the candidate's own peak.

        The peak is searched only within +/-40 ms of *t_utc* so a louder
        unrelated transient elsewhere in the read window cannot steal it
        (that failure produced NaN scores for quiet candidates)."""
        lead = 0.25
        x = self.read(t_utc - lead, lead + ENV_POST_S + 0.25)
        if x is None:
            return None, None
        xb = self.band(x)
        e = xb ** 2
        w = int(self.sr * ENV_MS / 1000)
        n = e.shape[1] // w
        env = e[:, : n * w].reshape(e.shape[0], n, w).mean(2)
        c0 = int(lead * 1000 / ENV_MS)                      # frame of t_utc
        half = int(0.04 * 1000 / ENV_MS)
        lo, hi = max(0, c0 - half), min(env.shape[1], c0 + half)
        core = env[:, lo:hi]
        ch = int(core.max(1).argmax())
        p = lo + int(core[ch].argmax())
        pre, post = int(ENV_PRE_S * 1000 / ENV_MS), int(ENV_POST_S * 1000 / ENV_MS)
        if p - pre < 0 or p + post > env.shape[1]:
            return None, None
        shape = env[ch, p - pre:p + post].astype(float)
        m = shape.max()
        if m <= 0:
            return None, None
        # spectrum: 40 ms starting 5 ms before the peak SAMPLE, same channel
        s0 = max(0, p * w - int(0.005 * self.sr))
        seg = xb[ch, s0: s0 + int(SPEC_WIN_S * self.sr)]
        spec = None
        if len(seg) > 1024:
            f = np.fft.rfftfreq(len(seg), 1 / self.sr)
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            spec = np.log10(spec[(f >= 5000) & (f <= 24000)] + 1e-12)
            spec = spec - spec.mean()
        return shape / m, spec


def cos(a, b):
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def score(e, s, env_t, spec_t):
    """min(envelope cos, spectrum cos); envelope-only when no spectrum."""
    ce, cs = cos(e, env_t), cos(s, spec_t)
    if ce is None:
        return None
    return ce if cs is None else min(ce, cs)


def low_threshold_detect(ses: Session, t_utc0: float, dur: float, z: float):
    """detect_strikes logic at low threshold on a true-UTC window."""
    x = ses.read(t_utc0, dur)
    if x is None:
        return []
    e = ses.band(x) ** 2
    w = int(ses.sr * ENV_MS / 1000)
    n = e.shape[1] // w
    env = e[:, : n * w].reshape(e.shape[0], n, w).mean(2)
    med = np.median(env, axis=1, keepdims=True)
    mad = np.median(np.abs(env - med), axis=1, keepdims=True) + 1e-30
    k = ((env - med) / mad).max(0)
    k[: int(round(EDGE_GUARD_S * 1000 / ENV_MS))] = 0.0
    pk = np.where(k > z)[0]
    dt = ENV_MS / 1000
    groups, out = [], []
    for p in pk:
        if groups and (p - groups[-1][-1]) * dt < MERGE_S:
            groups[-1].append(p)
        else:
            groups.append([p])
    for g in groups:
        i = g[int(np.argmax(k[g]))]
        out.append({"t": t_utc0 + i * dt, "z": float(k[i])})
    return out


def load_register(session: str):
    return list(csv.DictReader((OUTPUT_ROOT / f"strikes_register_{session}.csv").open()))


def build_template(ses: Session, rows, kinds_filter=None):
    envs, specs, self_scores = [], [], []
    for r in rows:
        if r["source"] != "both" or not r["t_utc"]:
            continue
        if kinds_filter and not any(r["slot"].startswith(k) for k in kinds_filter):
            continue
        e, s = ses.envelope_shape(datetime.fromisoformat(r["t_utc"]).timestamp())
        if e is not None:
            envs.append(e)
            specs.append(s)
    env_t = np.median(np.vstack(envs), axis=0)
    sp = [s for s in specs if s is not None]
    spec_t = np.median(np.vstack([s[:min(map(len, sp))] for s in sp]), axis=0)
    for e, s in zip(envs, specs, strict=True):
        sc = score(e, s, env_t, spec_t)
        if sc is not None:
            self_scores.append(sc)
    return env_t, spec_t, np.array(self_scores)


def negatives(ses: Session, gap_ranges, env_t, spec_t, n=200):
    scores = []
    for _ in range(n):
        lo, hi = gap_ranges[RNG.integers(len(gap_ranges))]
        t = float(RNG.uniform(lo, hi))
        e, s = ses.envelope_shape(t)
        sc = score(e, s, env_t, spec_t)
        if sc is not None:
            scores.append(sc)
    return np.array(scores)


def report(tag, cands, ses, env_t, spec_t, neg_max, pos_p5):
    print(f"-- {tag}")
    if not cands:
        print("   no candidates above threshold")
    for c in cands:
        e, s = ses.envelope_shape(c["t"])
        sc = score(e, s, env_t, spec_t)
        if sc is None:
            tt = datetime.fromtimestamp(c["t"], UTC).strftime("%H:%M:%S.%f")[:-3]
            print(f"   {tt}  unreadable")
            continue
        verdict = ("CONFIRMED" if sc > max(neg_max, 0.0) and sc >= pos_p5 * 0.9
                   else "above-noise" if sc > neg_max else "rejected")
        tt = datetime.fromtimestamp(c["t"], UTC).strftime("%H:%M:%S.%f")[:-3]
        print(f"   {tt}  z={c.get('z', float('nan')):6.1f}  score={sc:+.3f}  {verdict}")


def main():
    for session in ("st", "pu"):
        ses = Session(session.upper())
        reg = load_register(session)
        kinds = ("vane",) if session == "st" else None
        env_t, spec_t, selfs = build_template(ses, reg, kinds_filter=kinds)
        pos_p5 = float(np.percentile(selfs, 5))
        # strike-free gaps (true UTC): between-position silence
        gap_ranges_by_session = {
            "st": [(wall("12:26") + 40 - WALL_FROM_UTC, wall("12:27") + 55 - WALL_FROM_UTC),
                   (wall("12:16") + 5 - WALL_FROM_UTC, wall("12:16") + 55 - WALL_FROM_UTC)],
            "pu": [(wall("14:53") + 5 - WALL_FROM_UTC, wall("14:53") + 55 - WALL_FROM_UTC),
                   (wall("14:58") + 5 - WALL_FROM_UTC, wall("14:59") + 55 - WALL_FROM_UTC)],
        }
        gaps = gap_ranges_by_session[session]
        neg = negatives(ses, gaps, env_t, spec_t)
        neg_max = float(neg.max())
        print(f"===== {session.upper()}: template from {len(selfs)} 'both' strikes; "
              f"self-score p5={pos_p5:.3f}, noise max={neg_max:.3f} (n={len(neg)})")

        if session == "st":
            anchor = datetime.fromisoformat("2026-07-08T10:26:41.144+00:00").timestamp()
            cands = [c for c in low_threshold_detect(ses, anchor - 6.0, 12.0, z=5.0)
                     if abs(c["t"] - anchor) > 0.3]
            report("vane_18 window (+/-6 s, z>=5, rhythm 0.4-1.2 s noted per dt)", cands,
                   ses, env_t, spec_t, neg_max, pos_p5)
            for c in cands:
                print(f"      dt_to_anchor={c['t'] - anchor:+.2f}s")
        else:
            for label, hhmm in (("A_kugel", "14:49"), ("C_EG", "15:01")):
                t0 = wall(hhmm) - WALL_FROM_UTC
                cands = low_threshold_detect(ses, t0, 60.0, z=4.5)
                report(f"{label} minute (z>=4.5)", cands, ses, env_t, spec_t, neg_max, pos_p5)

        # hardening: annotated-only slots
        ann = [r for r in reg if r["source"] == "annotated-only"]
        print(f"-- hardening {len(ann)} annotated-only slots")
        for r in ann:
            t = datetime.fromisoformat(r["t_utc"]).timestamp()
            e, s = ses.envelope_shape(t)
            sc = score(e, s, env_t, spec_t)
            if sc is None:
                print(f"   {r['slot']}#{r['strike_no']}  unreadable")
                continue
            verdict = ("CONFIRMED" if sc > max(neg_max, 0.0) and sc >= pos_p5 * 0.9
                       else "above-noise" if sc > neg_max else "NOT confirmed")
            slot_id = f"{r['slot']}#{r['strike_no']}"
            print(f"   {slot_id} {r['t_utc'][11:23]}  score={sc:+.3f}  {verdict}")


if __name__ == "__main__":
    main()
