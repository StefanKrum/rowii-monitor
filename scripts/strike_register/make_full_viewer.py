"""Full campaign listening viewer: EVERY 08.07.2026 event as one v2-style card.

26 cards (2 sessions x [12 position minutes + 1 vane sweep]), assets as
RELATIVE files (too big to embed): per card the SNR-best channel(s), original
+ 4.5 kHz high-pass WAV, borderless spectrogram PNG per channel.

Markers per card, from the canonical register + verdicts sidecar:
  fat green   = measured strike (both / annotated-only / detector-only)
  fat orange  = statistical candidate (unconfirmed, "K?")
  dashed orange + shaded zone = rhythm-predicted position of a missing strike
  thin green  = extra impulses (bounces) without a protocol slot
Usage: .venv/bin/python scripts/strike_register/make_full_viewer.py
Output: OUTPUT_ROOT/listening/full/{viewer_full.html, assets/}
"""
from __future__ import annotations

import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, spectrogram

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from confirm_missing_strikes import Session  # noqa: E402
from paths import OUTPUT_ROOT  # noqa: E402

OUT = OUTPUT_ROOT / "listening" / "full"
ASSETS = OUT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

MIC = ["GenMic0", "GenMic90", "GenMic180", "GenMic270",
       "TurMic0", "TurMic90", "TurMic180", "TurMic270", "TurMicBottom"]
ZONE_HALF_S = 0.45


def utc(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def wall_min_to_utc(hhmm: str) -> float:
    return utc(f"2026-07-08T{hhmm}:00+00:00") - 7200.0


#: (session, register slot prefix, wall minute, channels) — channels from the
#: per-event loudest-detection table (2026-08-19); PU landmarks from the
#: anchor-SNR measurement (TurMic180 / TurMic90 best).
POS = [
    ("st", "landmark-C_EG", "12:15", ["GenMic270"]),
    ("st", "plate-gen_0", "12:17", ["GenMic0"]),
    ("st", "plate-gen_90", "12:18", ["TurMic90"]),
    ("st", "plate-gen_180", "12:19", ["GenMic180"]),
    ("st", "plate-gen_270", "12:20", ["TurMic270"]),
    ("st", "plate-tur_bottom", "12:21", ["TurMicBottom"]),
    ("st", "landmark-B_11TG", "12:22", ["TurMic270"]),
    ("st", "landmark-A_kugelschieber", "12:23", ["TurMic0"]),
    ("st", "plate-tur_0", "12:28", ["TurMic0"]),
    ("st", "plate-tur_90", "12:29", ["TurMic90"]),
    ("st", "plate-tur_180", "12:30", ["TurMic180"]),
    ("st", "plate-tur_270", "12:31", ["TurMic270"]),
    ("pu", "plate-gen_0", "14:43", ["GenMic0"]),
    ("pu", "plate-gen_90", "14:44", ["GenMic90"]),
    ("pu", "plate-gen_180", "14:45", ["TurMic180"]),
    ("pu", "plate-gen_270", "14:46", ["TurMic270"]),
    ("pu", "plate-tur_bottom", "14:47", ["TurMicBottom"]),
    ("pu", "landmark-B_11TG", "14:48", ["TurMicBottom"]),
    ("pu", "plate-tur_0", "14:54", ["TurMic0"]),
    ("pu", "plate-tur_90", "14:55", ["TurMic90"]),
    ("pu", "plate-tur_180", "14:56", ["TurMic180"]),
    ("pu", "plate-tur_270", "14:57", ["TurMic270"]),
]
SWEEP = [("st", "vane", "12:24", ["TurMic270", "TurMic90"], 210.0),
         ("pu", "vane", "14:50", ["TurMic0", "TurMic90"], 210.0)]

BRIEFS = {
    "pu_extA_landmarkA_kugelsch": (
        "GESUCHT: 3 Schonhammer-Schläge auf Blech, 13. TG neben dem Kugelschieber — unter "
        "Pumpenlärm, HP-Modus nutzen! Muster: 3er-Gruppe im ~0,75-s-Takt (Gesamtspanne "
        "~1,5–2,3 s). Beste Schätzung: um Clip-Sekunde 100,5 / 101,2 / 102,7 (K1 = "
        "statistischer Kandidat, unbestätigt). Die Karte deckt jetzt die GESAMTE Zeit ab, "
        "in der die Schläge physisch möglich sind (Ende B-Schläge bis Sweep-Beginn) — auch "
        "frei durchhören. Referenzklang ohne Lärm: Karte 'landmark-A_kugelschieber' in "
        "Session ST. Jeden gehörten Schlag mit M markieren."),
    "pu_extC_landmarkC_EG": (
        "GESUCHT: 3 Schonhammer-Schläge auf Blech im Erdgeschoss (weitester Weg, leiseste "
        "Station) — unter Pumpenlärm, HP-Modus! Muster: 3er-Gruppe im ~0,75-s-Takt. Beste "
        "Schätzung: um Clip-Sekunde 78,2 / 79,7 / 81,2 (K1 unbestätigt). Auch ±3 s drumherum "
        "und den Rest der Minute freihören. Referenzklang: Karte 'landmark-C_EG' in Session "
        "ST. Mit M markieren."),
    "st_focus_vane_18": (
        "GESUCHT: 2 der 3 Schläge auf die Blechabdeckung der 18. (letzten) Leitschaufel — "
        "Stillstand, gute Hörbarkeit. 1 Schlag ist gesichert (grün bei ~17,1 s). Muster: "
        "3er-Gruppe im ~0,75-s-Takt — die 2 fehlenden direkt davor (orange Zonen ~15,6/16,4 s) "
        "ODER bis ~3 s danach. NEU: der Rand-Scan fand 2 unbestätigte Nachzügler-Kandidaten "
        "K-a? (~37,1 s, schwach) und K-b? (~61,8 s, deutlich, z=30) — bitte beide anhören: "
        "Schlag, Hammer ablegen, oder anderes Geräusch? Klang-Referenz: die blauen "
        "Schaufel-17-Schläge links. Mit M markieren."),
    "st_1224_vane": (
        "Hier ist alles grün ausser GANZ AM ENDE (~Sekunde 164–167): die 2 fehlenden "
        "vane_18-Schläge. Bequemer suchen auf der Karte 'FOKUS: vane_18' weiter unten."),
}

#: edge-scan candidates (2026-08-19) outside all previous scan windows
EXTRA_ORANGE = {
    "st_focus_vane_18": [("K-a?", "2026-07-08T10:27:01.144+00:00"),
                         ("K-b?", "2026-07-08T10:27:25.824+00:00")],
}

PREDICTED = {  # slot -> predicted utc (last_six_verdicts sidecar)
    ("pu", "landmark-A_kugelschieber", "2"): utc("2026-07-08T12:49:45.466+00:00"),
    ("pu", "landmark-A_kugelschieber", "3"): utc("2026-07-08T12:49:47.716+00:00"),
    ("pu", "landmark-C_EG", "2"): utc("2026-07-08T13:01:18.170+00:00"),
    ("pu", "landmark-C_EG", "3"): utc("2026-07-08T13:01:21.170+00:00"),
    ("st", "vane_18", "2"): utc("2026-07-08T10:26:39.644+00:00"),
    ("st", "vane_18", "3"): utc("2026-07-08T10:26:40.394+00:00"),
}


def _collect(stream, t_wall: float, dur: float, sr: int) -> np.ndarray | None:
    """Concatenate one stream's chunks across burst-file boundaries; files may
    overlap by seconds (the earlier file wins, cursor-deduplicated). Seam
    error is a handful of samples — irrelevant for listening."""
    n_target = int(dur * sr)
    parts, cursor = [], t_wall
    for c0, c_sr, data in sorted(stream.chunks(t_wall, dur), key=lambda c: c[0]):
        c_end = c0 + data.shape[1] / c_sr
        if c_end <= cursor + 1e-4:
            continue
        off = max(0, int(round((cursor - c0) * c_sr)))
        parts.append(data[:, off:])
        cursor = c0 + data.shape[1] / c_sr
    if not parts:
        return None
    x = np.concatenate(parts, axis=1)
    return x[:, :n_target] if x.shape[1] >= sr else None


def read_stitched(ses: Session, t0: float, dur: float):
    t_wall = t0 + 7200.0
    g = _collect(ses.gen, t_wall, dur, ses.sr)
    t = _collect(ses.tur, t_wall, dur, ses.sr)
    if g is None or t is None:
        return None
    n = min(g.shape[1], t.shape[1])
    return np.vstack([g[:, :n], t[:, :n]])


def load_register(ses):
    return list(csv.DictReader((OUTPUT_ROOT / f"strikes_register_{ses}.csv").open()))


def load_raw(ses):
    return [{"t": utc(r["t_utc"]), "label": r["label"]}
            for r in csv.DictReader((OUTPUT_ROOT / f"raw_{ses}.csv").open())]


def png_for(sig, sr, path, width_in):
    f, t, S = spectrogram(sig, fs=sr, nperseg=1024, noverlap=512)
    db = 10 * np.log10(S + 1e-14)
    fig = plt.figure(figsize=(width_in, 3.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.pcolormesh(t, f / 1000, db, shading="auto", cmap="magma",
                  vmin=np.percentile(db, 55), vmax=np.percentile(db, 99.9))
    ax.set(ylim=(0, 25), xlim=(t[0], t[-1]))
    ax.axis("off")
    fig.savefig(path, format="png")
    plt.close(fig)


def wav_for(sig, sr, path):
    wavfile.write(path, sr, (0.9 * sig / (np.max(np.abs(sig)) + 1e-12) * 32767)
                  .astype(np.int16))


def main():
    sessions = {"st": Session("ST"), "pu": Session("PU")}
    regs = {s: load_register(s) for s in ("st", "pu")}
    raws = {s: load_raw(s) for s in ("st", "pu")}
    hp = butter(4, 4500, btype="high", fs=50000, output="sos")

    ALL9 = ["TurMic270", "GenMic270", "TurMic90", "TurMic180", "GenMic0",
            "GenMic90", "GenMic180", "TurMic0", "TurMicBottom"]
    cards = []
    jobs = ([(s, sl, m, chs, 70.0, False, None) for s, sl, m, chs in POS]
            + [(s, sl, m, chs, d, True, None) for s, sl, m, chs, d in SWEEP]
            + [("st", "vane_18", "focus", ALL9, 75.0, False,
                utc("2026-07-08T10:26:24.0+00:00")),
               ("pu", "landmark-A_kugelschieber", "extA", ALL9, 130.0, False,
                utc("2026-07-08T12:48:05.0+00:00")),
               ("pu", "landmark-C_EG", "extC", ALL9, 150.0, False,
                utc("2026-07-08T13:00:00.0+00:00"))])
    for ses_key, slot_prefix, wall_min, chs, dur, is_sweep, t0_override in jobs:
        ses = sessions[ses_key]
        t0 = t0_override if t0_override else wall_min_to_utc(wall_min) - 5.0
        cid = f"{ses_key}_{wall_min.replace(':', '')}_{slot_prefix.replace('-', '')[:18]}"
        need = [ch for ch in chs if not (ASSETS / f"{cid}_{ch}_orig.wav").is_file()]
        x = read_stitched(ses, t0, dur) if need else True
        if x is None:
            print(f"SKIP {ses_key} {slot_prefix} (unreadable)")
            continue
        for ch in need:
            sig = x[MIC.index(ch)].astype(np.float64)
            png_for(sig, ses.sr, ASSETS / f"{cid}_{ch}.png",
                    width_in=min(60, max(14, dur / 4.5)))
            wav_for(sosfilt(hp, sig), ses.sr, ASSETS / f"{cid}_{ch}_hp.wav")
            wav_for(sig, ses.sr, ASSETS / f"{cid}_{ch}_orig.wav")
        # markers
        greens, oranges, zones, thin = [], [], [], []
        for r in regs[ses_key]:
            in_card = (r["slot"].startswith("vane") if is_sweep
                       else r["slot"] == slot_prefix)
            if not in_card:
                continue
            if r["t_utc"]:
                trel = utc(r["t_utc"]) - t0
                if 0 <= trel <= dur:
                    lab = (r["slot"].replace("vane_", "v") + "#" + r["strike_no"]
                           if is_sweep else "✓" + r["strike_no"])
                    if r["source"] == "candidate-statistical":
                        oranges.append((f"K{r['strike_no']}?", trel, False))
                    else:
                        greens.append((lab, trel))
            else:
                key = (ses_key, r["slot"], r["strike_no"])
                if key in PREDICTED:
                    trel = PREDICTED[key] - t0
                    if 0 <= trel <= dur:
                        oranges.append((f"#{r['strike_no']}?", trel, True))
                        zones.append(trel)
        marked = [t for _, t in greens] + [t for _, t, _ in oranges]
        slot_ts = [utc(r["t_utc"]) for r in regs[ses_key] if r["t_utc"]]
        neigh = []
        for d in raws[ses_key]:
            trel = d["t"] - t0
            match_card = (d["label"].startswith("vane") if is_sweep
                          else True)
            if 0 <= trel <= dur and match_card and (
                    not marked or min(abs(trel - m) for m in marked) > 0.05):
                if min((abs(d["t"] - st) for st in slot_ts), default=9) <= 0.3:
                    neigh.append(trel)   # a measured slot of ANOTHER card
                else:
                    thin.append(trel)
        title = ("Leitschaufel-Sweep (18 Schaufeln × 3)" if is_sweep
                 else "FOKUS: vane_18 — letzte Schaufel (mit Schaufel 17 als Referenz links)"
                 if wall_min == "focus"
                 else "landmark-A_kugelschieber — VOLLE plausible Spanne "
                      "(Ende B-Schläge bis Sweep-Beginn)"
                 if wall_min == "extA"
                 else "landmark-C_EG — VOLLE plausible Spanne (Abstieg bis nach Protokollminute)"
                 if wall_min == "extC" else slot_prefix)
        cards.append(dict(cid=cid, ses=ses_key.upper(), title=title, chs=chs,
                          dur=dur, is_sweep=is_sweep, t0=t0,
                          start=datetime.fromtimestamp(t0, UTC)
                          .strftime("%H:%M:%S"),
                          greens=greens, oranges=oranges, zones=zones, thin=thin,
                          neigh=neigh))
        print(f"card {cid}: {len(greens)} measured, {len(oranges)} orange, "
              f"{len(thin)} context")

    # ---- html ----------------------------------------------------------
    head = """<!doctype html><html><head><meta charset="utf-8">
<title>Full Strike Viewer — alle Events 08.07.2026</title><style>
body{background:#14161a;color:#e8e8e8;font:14px/1.45 -apple-system,Helvetica,Arial;
margin:24px auto;max-width:1280px}
h1{font-size:19px} h2.ses{font-size:16px;color:#9fc1ff;margin-top:30px}
.card{background:#1d2026;border:1px solid #2c313a;border-radius:10px;
padding:12px 16px;margin:14px 0}
.card h3{font-size:14px;margin:0 0 2px} .meta{color:#9aa3b0;font-size:12px;margin-bottom:6px}
.wrap{position:relative;user-select:none;overflow-x:auto}
.inner{position:relative;display:inline-block;min-width:100%}
.inner img{display:block;border-radius:4px;cursor:crosshair;width:100%;height:auto}
.card.sweep .inner img{width:auto;height:300px;max-width:none}
.mk{position:absolute;top:0;bottom:0;width:0;border-left:2px solid} .mk.c{border-color:#2ecc71}
.mk.o{border-color:#f5a623} .mk.od{border-color:#f5a623;border-left-style:dashed}
.mk.x{border-left:1px solid rgba(46,204,113,.4)}
.mk.nb{border-left:1px solid rgba(120,160,255,.55)}
.mk.user{border-left:2px solid #ffd84d} .mk.user span{color:#ffd84d}
.brief{background:rgba(245,166,35,.10);border:1px solid rgba(245,166,35,.5);border-radius:6px;
padding:7px 10px;margin:0 0 8px;font-size:12.5px;color:#ffd9a0}
.brief b{color:#f5a623}
#exportbar{position:sticky;top:0;background:#14161a;padding:8px 0;z-index:9;
border-bottom:1px solid #2c313a}
#exportbox{width:100%;height:110px;background:#101215;color:#ffd84d;
border:1px solid #3a4150;border-radius:6px;font:11px/1.4 monospace;
display:none;margin-top:6px}
.mk span{position:absolute;top:2px;left:2px;font-size:9px;background:rgba(0,0,0,.6);
padding:0 3px;border-radius:3px;white-space:nowrap}
.zone{position:absolute;top:0;bottom:0;background:rgba(245,166,35,.16);
border-left:1px dashed rgba(245,166,35,.6);border-right:1px dashed rgba(245,166,35,.6)}
.ph{position:absolute;top:0;bottom:0;width:0;border-left:2px solid #ff5d5d;pointer-events:none}
.ctrl{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}
button,select{background:#2b303a;color:#e8e8e8;border:1px solid #3a4150;border-radius:6px;
padding:4px 10px;font-size:12px;cursor:pointer}
button:hover{background:#39404e} button.on{background:#3d5a3f;border-color:#2ecc71}
button.chb.on{background:#3a4a63;border-color:#6aa1ff}
audio{width:250px;height:30px} .lg{color:#9aa3b0;font-size:12px;margin:4px 0 10px}
</style></head><body>
<h1>Full Strike Viewer — alle Events, 08.07.2026</h1>
<div class="lg"><b style="color:#2ecc71">grün</b> = gemessener Schlag ·
<b style="color:#f5a623">orange K?</b> = unbestätigter statistischer Kandidat ·
<b style="color:#f5a623">orange gestrichelt + Zone</b> = fehlender Schlag,
rhythmisch vorhergesagter Suchbereich ·
<b style="color:rgba(46,204,113,.7)">dünn grün</b> = Zusatzimpulse/Preller
(echt, aber ohne Protokoll-Slot) ·
<b style="color:rgba(120,160,255,.9)">dünn blau</b> = Schlag der NACHBAR-Position
(bereits im Register, eigene Karte) ·
Klick ins Bild = springen · ⟲ = Loop ±1.25 s · Sweep-Karten sind horizontal scrollbar</div>
<div id="exportbar"><b style="color:#ffd84d">Deine Marken:</b>
Taste <b>M</b> = Marke am roten Cursor der zuletzt bedienten Karte ·
<b>X</b> = letzte Marke dieser Karte löschen ·
<button id="exp">Marken exportieren</button> <button id="clr">alle löschen</button>
<span id="mcount"></span><textarea id="exportbox" readonly></textarea></div>
"""
    body, last_ses = [], None
    for c in cards:
        if c["ses"] != last_ses:
            body.append(f'<h2 class="ses">Session {c["ses"]} '
                        f'({"Stillstand" if c["ses"] == "ST" else "Pumpbetrieb"})</h2>')
            last_ses = c["ses"]
        pw = 100.0 / c["dur"]
        marks = []
        for t in c["thin"]:
            marks.append(f'<div class="mk x" style="left:{t*pw:.3f}%"></div>')
        for t in c.get("neigh", []):
            marks.append(f'<div class="mk nb" style="left:{t*pw:.3f}%"></div>')
        for z in c["zones"]:
            marks.append(f'<div class="zone" style="left:{(z-ZONE_HALF_S)*pw:.3f}%;'
                         f'width:{2*ZONE_HALF_S*pw:.3f}%"></div>')
        btns = []
        for lab, t in c["greens"]:
            marks.append(f'<div class="mk c" style="left:{t*pw:.3f}%"><span>{lab}</span></div>')
            if not c["is_sweep"]:
                btns.append(f'<button class="jmp" data-t="{t:.3f}">▶ {lab}</button>'
                            f'<button class="loop" data-t="{t:.3f}">⟲</button>')
        for lab, tt_iso in EXTRA_ORANGE.get(c["cid"], []):
            base_t0 = (utc("2026-07-08T10:26:24.0+00:00")
                       if c["cid"] == "st_focus_vane_18" else 0)
            trel_x = utc(tt_iso) - base_t0
            c["oranges"].append((lab, trel_x, False))
            c["oranges"].sort(key=lambda o: o[1])
        for lab, t, dashed in c["oranges"]:
            cls = "od" if dashed else "o"
            marks.append(f'<div class="mk {cls}" style="left:{t*pw:.3f}%"><span>{lab}</span></div>')
            btns.append(f'<button class="jmp" data-t="{t:.3f}">▶ {lab}</button>'
                        f'<button class="loop" data-t="{t:.3f}">⟲</button>')
        chbtns = "".join(
            f'<button class="chb{" on" if i == 0 else ""}" data-ch="{ch}">{ch}</button>'
            for i, ch in enumerate(c["chs"]))
        sweep_cls = " sweep" if c["is_sweep"] else ""
        brief = BRIEFS.get(c["cid"])
        brief_html = f'<div class="brief"><b>🔍 Suchauftrag:</b> {brief}</div>' if brief else ""
        body.append(f"""<div class="card{sweep_cls}" data-cid="{c['cid']}"
data-dur="{c['dur']}" data-t0="{c['t0']:.3f}" data-chs='{",".join(c["chs"])}'>
<h3>{c['title']}</h3>
<div class="meta">Start {c['start']} UTC · {c['dur']:.0f} s · HP 4.5 kHz aktiv</div>
{brief_html}
<div class="wrap"><div class="inner"><img loading="lazy">
{''.join(marks)}<div class="ph" style="left:0%"></div></div></div>
<div class="ctrl">{chbtns}<audio controls preload="none"></audio>
<button class="src on">HP 4.5k</button>
<select class="spd"><option value="1">1×</option>
<option value="0.75">0.75×</option><option value="0.5">0.5×</option></select>
{''.join(btns)}</div></div>""")

    js = """<script>
const MKEY='strikeUserMarks_v1';
const BLOBS={};
function toBlob(url){ if(!BLOBS[url])
  BLOBS[url]=fetch(url).then(r=>{if(!r.ok)throw 0;return r.blob();})
    .then(b=>URL.createObjectURL(b)).catch(()=>url);
 return BLOBS[url];}
let USER=JSON.parse(localStorage.getItem(MKEY)||'{}'), LAST=null;
const saveU=()=>{localStorage.setItem(MKEY,JSON.stringify(USER));updCount();};
const updCount=()=>{const n=Object.values(USER).reduce((a,b)=>a+b.length,0);
  document.getElementById('mcount').textContent=' '+n+' Marken gesetzt';};
function renderUser(card){
 card.querySelectorAll('.mk.user').forEach(e=>e.remove());
 const dur=+card.dataset.dur, inner=card.querySelector('.inner');
 (USER[card.dataset.cid]||[]).forEach((t,i)=>{
   const d=document.createElement('div'); d.className='mk user';
   d.style.left=(t/dur*100)+'%'; d.innerHTML='<span>M'+(i+1)+' '+t.toFixed(2)+'s</span>';
   inner.appendChild(d);});}
document.addEventListener('keydown',e=>{
 if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;
 if(!LAST)return;
 const cid=LAST.dataset.cid, a=LAST.querySelector('audio');
 if(e.key==='m'||e.key==='M'){(USER[cid]=USER[cid]||[]).push(+a.currentTime.toFixed(3));
   USER[cid].sort((x,y)=>x-y); saveU(); renderUser(LAST);}
 if(e.key==='x'||e.key==='X'){if(USER[cid]&&USER[cid].length){USER[cid].pop();saveU();renderUser(LAST);}}});
document.getElementById('exp').onclick=()=>{
 let rows=['card,t_rel_s,t_utc_iso'];
 document.querySelectorAll('.card').forEach(card=>{
   const t0=+card.dataset.t0;
   (USER[card.dataset.cid]||[]).forEach(t=>{
     rows.push(card.dataset.cid+','+t.toFixed(3)+','+new Date((t0+t)*1000).toISOString());});});
 const box=document.getElementById('exportbox'); box.style.display='block';
 box.value=rows.join('\\n'); box.select();
 try{navigator.clipboard.writeText(box.value);}catch(_){}};
document.getElementById('clr').onclick=()=>{
 if(confirm('Alle eigenen Marken löschen?')){USER={};saveU();
  document.querySelectorAll('.card').forEach(renderUser);}};
document.querySelectorAll('.card').forEach(card=>{
 const dur=+card.dataset.dur, cid=card.dataset.cid, a=card.querySelector('audio'),
   inner=card.querySelector('.inner'), img=card.querySelector('img'),
   ph=card.querySelector('.ph'), srcBtn=card.querySelector('.src');
 let loop=null, ch=card.dataset.chs.split(',')[0];
 card.addEventListener('click',()=>{LAST=card;});
 card.querySelector('audio').addEventListener('play',()=>{LAST=card;});
 renderUser(card);
 let curUrl='';
 const upgrade=()=>{const u=curUrl;
   toBlob(u).then(b=>{if(curUrl!==u||a.src.endsWith(b)||b===u)return;
     const t=a.currentTime,pl=!a.paused; a.src=b;
     a.addEventListener('loadedmetadata',()=>{a.currentTime=t;if(pl)a.play();},{once:true});
     a.load();});};
 const load=(keep)=>{const t=a.currentTime,pl=!a.paused,hpOn=srcBtn.classList.contains('on');
   img.src='assets/'+cid+'_'+ch+'.png';
   curUrl='assets/'+cid+'_'+ch+(hpOn?'_hp.wav':'_orig.wav');
   a.src=curUrl; a.load(); upgrade._armed=false;
   if(keep){a.addEventListener('loadedmetadata',()=>{a.currentTime=t;if(pl)a.play();},{once:true});
     upgrade();}};
 a.addEventListener('play',()=>{if(!upgrade._armed){upgrade._armed=true;upgrade();}});
 load(false);
 card.querySelectorAll('.chb').forEach(b=>b.onclick=()=>{
   card.querySelectorAll('.chb').forEach(x=>x.classList.remove('on'));
   b.classList.add('on'); ch=b.dataset.ch; load(true);});
 srcBtn.onclick=()=>{srcBtn.classList.toggle('on');
   srcBtn.textContent=srcBtn.classList.contains('on')?'HP 4.5k':'Original'; load(true);};
 card.querySelector('.spd').onchange=e=>a.playbackRate=+e.target.value;
 inner.onclick=e=>{const r=inner.getBoundingClientRect();
   a.currentTime=(e.clientX-r.left)/r.width*dur; if(a.paused)a.play();};
 card.querySelectorAll('.jmp').forEach(b=>b.onclick=()=>{loop=null;
   card.querySelectorAll('.loop').forEach(x=>x.classList.remove('on'));
   a.currentTime=Math.max(0,+b.dataset.t-1.5); a.play();});
 card.querySelectorAll('.loop').forEach(b=>b.onclick=()=>{
   const on=!b.classList.contains('on');
   card.querySelectorAll('.loop').forEach(x=>x.classList.remove('on'));
   if(on){b.classList.add('on'); loop=[Math.max(0,+b.dataset.t-1.25),+b.dataset.t+1.25];
     a.currentTime=loop[0]; a.play();} else loop=null;});
 a.addEventListener('timeupdate',()=>{if(loop&&a.currentTime>loop[1])a.currentTime=loop[0];});
 const wrapEl=card.querySelector('.wrap');
 (function tick(){ph.style.left=(a.currentTime/dur*100)+'%';
   if(!a.paused&&inner.offsetWidth>wrapEl.clientWidth){
     const px=a.currentTime/dur*inner.offsetWidth;
     if(px<wrapEl.scrollLeft+40||px>wrapEl.scrollLeft+wrapEl.clientWidth-80)
       wrapEl.scrollLeft=Math.max(0,px-wrapEl.clientWidth*0.3);}
   requestAnimationFrame(tick);})();
});
updCount();
</script></body></html>"""
    (OUT / "viewer_full.html").write_text(head + "".join(body) + js)
    total = sum(f.stat().st_size for f in ASSETS.glob("*")) / 1e6
    print(f"-> {OUT / 'viewer_full.html'} ({len(cards)} cards, assets {total:.0f} MB)")


if __name__ == "__main__":
    main()
