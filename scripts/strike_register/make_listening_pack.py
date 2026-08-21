"""Listening pack for the 6 open strike slots: WAV clips + annotated spectrograms.

For each search region: the loudest mic channel, original + 4.5 kHz high-pass
WAV (pump noise is low-frequency; the high-pass makes strikes audible), and an
STFT spectrogram (0-25 kHz) with the confirmed anchor (green) and the
rhythm-predicted positions (orange dashed). Two ST reference clips of confirmed
landmark triplets show what a strike looks/sounds like without pump noise.
Usage: .venv/bin/python scripts/strike_register/make_listening_pack.py
Output: OUTPUT_ROOT/listening/
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

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

OUT = OUTPUT_ROOT / "listening"
OUT.mkdir(parents=True, exist_ok=True)

MIC_NAMES = ["GenMic0", "GenMic90", "GenMic180", "GenMic270",
             "TurMic0", "TurMic90", "TurMic180", "TurMic270", "TurMicBottom"]


def utc(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


#: name, session, clip start, clip dur, confirmed times (green), predicted (orange)
CLIPS = [
    ("PU_A_kugelschieber_search", "PU", utc("2026-07-08T12:49:44.0+00:00"), 5.4,
     [utc("2026-07-08T12:49:46.216+00:00")],
     [utc("2026-07-08T12:49:45.466+00:00"), utc("2026-07-08T12:49:47.716+00:00")]),
    ("PU_C_EG_search", "PU", utc("2026-07-08T13:01:16.7+00:00"), 6.0,
     [utc("2026-07-08T13:01:19.670+00:00")],
     [utc("2026-07-08T13:01:18.170+00:00"), utc("2026-07-08T13:01:21.170+00:00")]),
    ("ST_vane18_search", "ST", utc("2026-07-08T10:26:38.6+00:00"), 4.2,
     [utc("2026-07-08T10:26:41.144+00:00")],
     [utc("2026-07-08T10:26:39.644+00:00"), utc("2026-07-08T10:26:40.394+00:00")]),
    ("ST_A_kugelschieber_REFERENCE", "ST", utc("2026-07-08T10:23:14.9+00:00"), 3.6,
     [utc("2026-07-08T10:23:15.915+00:00"), utc("2026-07-08T10:23:16.582+00:00"),
      utc("2026-07-08T10:23:17.242+00:00")], []),
    ("ST_C_EG_REFERENCE", "ST", utc("2026-07-08T10:15:01.2+00:00"), 4.0,
     [utc("2026-07-08T10:15:02.243+00:00"), utc("2026-07-08T10:15:03.041+00:00"),
      utc("2026-07-08T10:15:04.018+00:00")], []),
]


def wav_norm(x: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, (0.9 * x / (np.max(np.abs(x)) + 1e-12) * 32767).astype(np.int16))


def main() -> None:
    sessions = {"ST": Session("ST"), "PU": Session("PU")}
    hp: np.ndarray | None = None
    index = ["# Listening pack — open strike slots (2026-08-19)\n",
             "green solid = confirmed strike · orange dashed = rhythm-predicted (NOT detected)\n"]
    for name, ses_name, t0, dur, confirmed, predicted in CLIPS:
        ses = sessions[ses_name]
        x = ses.read(t0, dur)
        if x is None:
            print(f"{name}: window unreadable")
            continue
        sr = ses.sr
        band = ses.band(x)
        ch = int((band ** 2).sum(axis=1).argmax())
        sig = x[ch].astype(np.float64)
        if hp is None:
            hp = butter(4, 4500, btype="high", fs=sr, output="sos")
        sig_hp = sosfilt(hp, sig)
        wavfile.write(OUT / f"{name}_{MIC_NAMES[ch]}_orig.wav", sr, wav_norm(sig))
        wavfile.write(OUT / f"{name}_{MIC_NAMES[ch]}_hp4k5.wav", sr, wav_norm(sig_hp))

        f, t, S = spectrogram(sig, fs=sr, nperseg=1024, noverlap=768)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        db = 10 * np.log10(S + 1e-14)
        ax.pcolormesh(t, f / 1000, db, shading="gouraud", cmap="magma",
                      vmin=np.percentile(db, 55), vmax=np.percentile(db, 99.9))
        for tt in confirmed:
            ax.axvline(tt - t0, color="#2ecc71", lw=1.6,
                       label="confirmed strike" if tt == confirmed[0] else None)
        for i, tt in enumerate(predicted):
            ax.axvline(tt - t0, color="#f39c12", lw=1.6, ls="--",
                       label="predicted (not detected)" if i == 0 else None)
        start_iso = datetime.fromtimestamp(t0, UTC).strftime("%H:%M:%S.%f")[:-3]
        ax.set(title=f"{name}  ·  {MIC_NAMES[ch]}  ·  clip start {start_iso} UTC",
               xlabel="seconds into clip", ylabel="kHz", ylim=(0, 25))
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        fig.tight_layout()
        fig.savefig(OUT / f"{name}.png", dpi=150)
        plt.close(fig)
        marks = " ".join(f"{tt - t0:.2f}s" for tt in confirmed + predicted)
        index.append(f"- **{name}** ({MIC_NAMES[ch]}): start {start_iso} UTC, "
                     f"marks at {marks} into the clip")
        print(f"{name}: ch={MIC_NAMES[ch]}  wav+png written")
    (OUT / "INDEX.md").write_text("\n".join(index) + "\n")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
