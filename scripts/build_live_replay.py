"""Build `docs/site/live.html`: a real, native (not iframed) control-room replay of
one recorded session (`RUN`, 290626-tu) from already-computed real artifacts --
NEVER touches `ROWII_DATA_ROOT`/raw Gantner burst files. Everything the page needs
(state timeline, p-value stream, alarm feed, sentinel verdict, SCADA line, per-stream
level series, a downsampled log-mel strip, a downsampled feature snapshot matrix) is
precomputed here into ONE embedded JSON payload, injected into `docs/site/
live_template.html` at a single `__LIVE_DATA_JSON__` token, producing `docs/site/
live.html`. Reproducible from `results/` + `docs/site/live_template.html` alone.

Representation/regime choice (documented, not arbitrary): the primary state/score/
alarm arm is **fusion, frozen thresholds** -- `candidate_kit.REGIME_BY_SESSION
["290626-tu"] == "frozen"`, i.e. the ACTUAL once-calibrated decision the label-free
drift sentinel made for this session (s1_rate=0.0767 sits just under its own
threshold 0.0805 -- `research/notes/analysis_2026-07-22_p9_once_calibrated_named_
transitions.md`'s own documented "Beinahe-Miss"/near-miss case), not a
counterfactual. fusion (not audio-beats) is the headline arm because its realized
FAR under frozen thresholds (7.5%, common-window basis) sits close to the nominal
5% budget -- audio-beats' OWN frozen-mode FAR balloons to 23% on this exact day
(the synthesis note's own "one expensive case"), which would flood the alert feed
with ~1900 episodes and misrepresent the system's typical behaviour. The sentinel
gauge itself is representation-independent by construction (s1 always reads the
audio-beats mode bank, `run_once_calibrated.py`'s own docstring) so its numbers are
identical either way.

Alert-feed source: `results/candidate-kit/candidates.csv` rows for this session
(already the two-path SUSTAINED/TRANSIENT classification, already SCADA-attached,
already carrying a human criterion sentence) -- NOT the raw 275 frozen-mode alarm
episodes, which lack a why-line entirely. See `scripts/candidate_kit.py`'s own
module docstring for the selection rule.

Time alignment: every per-second series in this payload (RMS levels, log-mel strip,
feature snapshot, SCADA) is aligned to the PRIMARY (fusion monitor) timeline by
ARRAY INDEX, not by re-matching nanosecond timestamps. Real per-cache `grid_t0_ns`
metadata disagrees by up to ~100 ms between caches (one legacy-format cache pair,
`audio.npz`/`fusion.npz`, even carries a stale pre-UTC-offset-fix value entirely a
different order of magnitude -- verified during this task, not a documented
pre-existing fact) despite every cache sharing the SAME `grid_n_windows`/
`grid_window_ns` (1.0 s) and being built from the same run's audio-stream offset.
For a 1 Hz visual replay this sub-second/metadata-only disagreement is immaterial;
index `i` is treated as "second `i` of the replay" throughout.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPTS_DIR.parent / "src"
for _extra_path in (str(_SCRIPTS_DIR), str(_SRC_DIR)):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

import annotation_kit as ak  # noqa: E402
import candidate_kit as ck  # noqa: E402
import make_demo_assets as mda  # noqa: E402
import run_step1 as rs1  # noqa: E402
import site_common as sc  # noqa: E402

from rowii.config import load_config  # noqa: E402
from rowii.io.dataset import discover, run_utc_offset_ns  # noqa: E402
from rowii.pipeline import build_run_grid  # noqa: E402
from rowii.scada.labels import gt_labels, load_scada_window_means  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = _SCRIPTS_DIR.parent
RESULTS_ROOT = REPO_ROOT / "results"
CACHE_DIR = RESULTS_ROOT / "cache"
DEFAULT_TEMPLATE = REPO_ROOT / "docs" / "site" / "live_template.html"
DEFAULT_OUT = REPO_ROOT / "docs" / "site" / "live.html"

RUN = "290626-tu"
REPRESENTATION = "fusion"
REGIME = ck.REGIME_BY_SESSION[RUN]
"""`"frozen"` -- the real once-calibrated decision for this session (module
docstring), read from the same lookup `candidate_kit.py` itself uses, not
re-hardcoded."""
SENTINEL_REPRESENTATION = "audio-beats"
"""s1/s2 always read the audio-beats mode bank/raw caches, representation-independent
by construction (`run_once_calibrated.py`)."""
UNIT_NAME = "ROWII Machine 1 — Rodundwerk II"

MONITOR_DIR = RESULTS_ROOT / "step2" / "once-calibrated" / REPRESENTATION / "monitor" / RUN / REGIME
SENTINEL_JSON = (
    RESULTS_ROOT / "step2" / "once-calibrated" / SENTINEL_REPRESENTATION
    / f"{SENTINEL_REPRESENTATION}.json"
)
CANDIDATES_CSV = RESULTS_ROOT / "candidate-kit" / "candidates.csv"
CANDIDATES_META = RESULTS_ROOT / "candidate-kit" / "candidates_meta.json"

FEATURE_SNAPSHOT_STRIDE_S = 4
"""Feature-snapshot heatmap cadence: one column every N seconds of replay time --
231 raw dims (135 audio + 96 vibration) at 1 Hz for the whole ~4.4 h day would be
~14 MB of binary alone; every `FEATURE_SNAPSHOT_STRIDE_S`-th window keeps the panel
genuinely live (a new column every few seconds of simulated time) inside the page's
size budget."""
LOGMEL_N_MELS = 64
LOGMEL_STRIDE_S = 1
"""Log-mel strip cadence: one column per second (mean-pooled over each window's own
49 STFT frames -- module docstring's "downsampled" instruction) -- ~15.6k columns x
64 mels, well under a megabyte once PNG-compressed (spectrograms compress well)."""


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _f32_b64(arr: np.ndarray) -> str:
    """A 2D array, row-major float32, base64-encoded -- ~40% smaller on the wire
    than a JSON array of text numbers, decoded client-side via `atob` +
    `Float32Array`."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def _round_list(arr: np.ndarray, ndigits: int) -> list[float]:
    return [round(float(v), ndigits) for v in arr]


def _png_b64(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Primary timeline: fusion/frozen monitor (state, score, p-value, alarm, segments)
# ---------------------------------------------------------------------------


def load_primary_timeline() -> dict[str, Any]:
    segments_df = pd.read_csv(MONITOR_DIR / "segments.csv")
    segments_df["start_utc"] = pd.to_datetime(segments_df["start_utc"], utc=True)
    segments_df["end_utc"] = pd.to_datetime(segments_df["end_utc"], utc=True)
    t0 = segments_df["start_utc"].min()
    t0_ns = int(t0.value)
    duration_s = (int(segments_df["end_utc"].max().value) - t0_ns) / 1e9

    notes_text = (MONITOR_DIR / "monitor_notes.md").read_text(encoding="utf-8")
    state_table = mda.parse_state_table(notes_text)
    states = {
        str(sid): {
            "name": info["name"],
            "name_label": mda.state_display_name(info["name"]),
            "threshold": (
                None if info["threshold"] is None or math.isinf(info["threshold"])
                else round(info["threshold"], 6)
            ),
            "low_confidence": info["low_confidence"],
        }
        for sid, info in state_table.items()
    }

    segments = [
        {
            "start_s": round((int(r.start_utc.value) - t0_ns) / 1e9, 3),
            "end_s": round((int(r.end_utc.value) - t0_ns) / 1e9, 3),
            "state": int(r.cluster_id),
            "state_name": str(r.mapped_mode),
        }
        for r in segments_df.itertuples()
    ]

    alarms_df = pd.read_parquet(MONITOR_DIR / "alarms.parquet")
    scored = alarms_df.loc[alarms_df["role"] == "scored"].sort_values("t_utc_ns")
    if scored.empty:
        raise ValueError(f"{RUN}: no scored windows in {MONITOR_DIR / 'alarms.parquet'}")
    t_s = ((scored["t_utc_ns"].to_numpy() - t0_ns) / 1e9).round(3)

    trace = {
        "t_s": t_s.tolist(),
        "p_value": _round_list(scored["p_value"].to_numpy(), 6),
        "score": _round_list(scored["score"].to_numpy(), 6),
        "state": [int(v) for v in scored["state"].to_numpy()],
        "state_name": [str(v) for v in scored["state_name"].to_numpy()],
        "alarm": [bool(v) for v in scored["alarm"].to_numpy()],
        "near_transition": [bool(v) for v in scored["near_transition"].to_numpy()],
    }

    alarm_seg_df = pd.read_csv(MONITOR_DIR / "alarm_segments.csv")
    n_alarm_episodes = len(alarm_seg_df)
    n_scored = len(scored)
    n_alarmed_windows = int(scored["alarm"].sum())

    # 290626-tu has no induced events (no eval_events dir for this session);
    # realized FAR/budget instead comes from the regimes table (load_sentinel
    # below), matching every other non-event day's own reporting convention
    # (run_once_calibrated.py).

    return {
        "t0_ns": t0_ns,
        "t0_utc": t0.isoformat(),
        "duration_s": round(duration_s, 3),
        "representation": REPRESENTATION,
        "regime": REGIME,
        "states": states,
        "segments": segments,
        "trace": trace,
        "n_scored": n_scored,
        "n_alarmed_windows": n_alarmed_windows,
        "n_alarm_episodes": n_alarm_episodes,
    }


def load_sentinel() -> dict[str, Any]:
    payload = json.loads(SENTINEL_JSON.read_text(encoding="utf-8"))
    trig = next(t for t in payload["trigger_log"] if t["run"] == RUN)
    fusion_json_path = (
        RESULTS_ROOT / "step2" / "once-calibrated" / REPRESENTATION / f"{REPRESENTATION}.json"
    )
    fusion_json = json.loads(
        fusion_json_path.read_text(
            encoding="utf-8"
        )
    )
    regime = next(r for r in fusion_json["regimes"] if r["run"] == RUN)
    return {
        "era": trig["era"],
        "s1_rate": round(trig["s1_rate"], 6),
        "s1_threshold": round(trig["s1_threshold"], 6),
        "s1_fired": bool(trig["s1_fired"]),
        "s2_fired": bool(trig["s2_fired"]),
        "s2_attribution": trig["s2_attribution"],
        "decision": trig["decision"],
        "nominal_alpha": 0.05,
        "realized_far": round(regime["once_triggered_far"], 6),
        "always_frozen_far": round(regime["always_frozen_far"], 6),
        "always_recalibrate_far": round(regime["always_recalibrate_far"], 6),
        "far_basis": regime["far_basis"],
    }


# ---------------------------------------------------------------------------
# Per-stream RMS level series (sensor panel ring pulsing)
# ---------------------------------------------------------------------------


def load_levels() -> dict[str, Any]:
    """One 1 Hz level series per raw stream -- mic streams use channel 0's
    `log_rms` (the pipeline's own established "channel 0 of each stream"
    convention, `make_demo_assets.MONO_CHANNEL_INDEX`); vibration streams use the
    MEAN `log_rms` over every channel the handcrafted cache actually carries for
    that stream (no established single-channel convention exists for vibration,
    module docstring's own honesty note on the ring's real position<->channel
    mapping never being verified)."""
    audio = np.load(CACHE_DIR / f"{RUN}--audio.npz", allow_pickle=True)
    vibration = np.load(CACHE_DIR / f"{RUN}--vibration.npz", allow_pickle=True)

    def _column(d: Any, name: str) -> np.ndarray:
        idx = int(np.where(d["feature_names"] == name)[0][0])
        return np.asarray(d["features"][:, idx], dtype=np.float64)

    def _mean_columns(d: Any, prefix: str, suffix: str) -> np.ndarray:
        names = list(d["feature_names"])
        cols = [i for i, n in enumerate(names) if n.startswith(prefix) and n.endswith(suffix)]
        if not cols:
            raise ValueError(f"no columns matching {prefix}*{suffix} in cache")
        return np.asarray(d["features"][:, cols], dtype=np.float64).mean(axis=1)

    gen_mic = _column(audio, "RAWGeneratorMic__0::ch0_log_rms")
    tur_mic = _column(audio, "RAWTurbineMic__1::ch0_log_rms")
    gen_vib = _mean_columns(vibration, "RAWGeneratorVib__2::", "_log_rms")
    tur_vib = _mean_columns(vibration, "RAWTurbineVib__3::", "_log_rms")
    valid = np.asarray(audio["valid_mask"], dtype=bool)

    def _norm01(x: np.ndarray) -> np.ndarray:
        finite = x[valid]
        lo, hi = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
        span = max(hi - lo, 1e-9)
        out = np.clip((x - lo) / span, 0.0, 1.0)
        out[~valid] = 0.0
        return out

    return {
        "n": int(len(gen_mic)),
        "gen_mic": _round_list(_norm01(gen_mic), 4),
        "tur_mic": _round_list(_norm01(tur_mic), 4),
        "gen_vib": _round_list(_norm01(gen_vib), 4),
        "tur_vib": _round_list(_norm01(tur_vib), 4),
        "valid": [bool(v) for v in valid],
    }


# ---------------------------------------------------------------------------
# SCADA line (P, n)
# ---------------------------------------------------------------------------


def load_scada(index: Any, cfg: Any) -> dict[str, Any]:
    session_scada = ck.load_session_scada(index, RUN, cfg)

    # Load bin (Step 1 stage's "load bin" readout): re-derives `rowii.scada.labels.
    # gt_labels`'s own `load_bin` column (quantile-binned power, turbine/pump windows
    # only, n_load_bins=3 by default) -- `load_session_scada` itself only keeps
    # `state`, not `load_bin`, so this mirrors that function's own internals
    # (`rs1`/`build_run_grid`/`load_scada_window_means`/`gt_labels`, imported
    # directly here -- not via `candidate_kit`'s own namespace, which mypy's
    # `no_implicit_reexport` correctly refuses to treat as a public API) rather
    # than duplicating a second real-data read.
    run = mda._get_run(index, RUN)
    betriebsdaten = index.betriebsdaten_by_day.get(run.day_root, [])
    offset_ns = run_utc_offset_ns(run)
    grid = build_run_grid(run, rs1._AUDIO_STREAMS, cfg.window.window_s, offset_ns=offset_ns)
    matched_files = rs1._betriebsdaten_for_grid(betriebsdaten, grid)
    scada_means = load_scada_window_means(matched_files, grid, audio_run_offset_ns=offset_ns)
    gt = gt_labels(scada_means, cfg.gt, window_s=cfg.window.window_s)
    load_bin = gt["load_bin"].to_numpy()

    def _round_or_none(v: float | None, digits: int) -> float | None:
        return None if v is None or math.isnan(v) else round(float(v), digits)

    return {
        "has_scada": session_scada.has_scada,
        "n": len(session_scada.power_mw),
        "power_mw": [_round_or_none(v, 3) for v in session_scada.power_mw],
        "speed_rpm": [_round_or_none(v, 2) for v in session_scada.speed_rpm],
        "scada_state": session_scada.state,
        "load_bin": [int(v) for v in load_bin],
        "n_load_bins": int(cfg.gt.n_load_bins),
    }


# ---------------------------------------------------------------------------
# Log-mel scrolling strip
# ---------------------------------------------------------------------------


def load_logmel_strip() -> dict[str, Any]:
    """A `(LOGMEL_N_MELS, n_windows)` grayscale-then-colormapped PNG: each
    window's own `(49 frames, 64 mels)` patch (`RAWGeneratorMic__0` only --
    `rowii.signals.logmel`'s own frame/mel layout, verified against this cache's
    real `feature_names`) mean-pooled over its 49 frames down to one column,
    columns stacked in grid order (`LOGMEL_STRIDE_S` apart). An invalid window
    holds the previous valid column (a flat "no new data" carry-forward, never a
    fabricated value) so the strip has no visual gap for the ~0.1% of windows
    with no usable audio."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(CACHE_DIR / f"{RUN}--logmel.npz", allow_pickle=True)
    features = np.asarray(d["features"], dtype=np.float32)
    valid = np.asarray(d["valid_mask"], dtype=bool)
    n_windows = features.shape[0]
    patch = features.reshape(n_windows, 49, LOGMEL_N_MELS)
    per_window_mel = patch.mean(axis=1)  # (n_windows, 64)

    cols = []
    last = np.zeros(LOGMEL_N_MELS, dtype=np.float32)
    for i in range(0, n_windows, LOGMEL_STRIDE_S):
        if valid[i]:
            last = per_window_mel[i]
        cols.append(last)
    image = np.stack(cols, axis=1)  # (64, n_cols)

    width_px = image.shape[1]
    height_px = LOGMEL_N_MELS
    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(image, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
    ax.axis("off")
    png_b64 = _png_b64(fig)
    plt.close(fig)

    return {
        "png_b64": png_b64,
        "width_px": int(width_px),
        "height_px": int(height_px),
        "stride_s": LOGMEL_STRIDE_S,
        "n_mels": LOGMEL_N_MELS,
    }


# ---------------------------------------------------------------------------
# Feature snapshot (fusion: 135-d audio + 96-d vibration; BEATs summary)
# ---------------------------------------------------------------------------


def load_feature_snapshot() -> dict[str, Any]:
    audio = np.load(CACHE_DIR / f"{RUN}--audio.npz", allow_pickle=True)
    vibration = np.load(CACHE_DIR / f"{RUN}--vibration.npz", allow_pickle=True)
    beats = np.load(CACHE_DIR / f"{RUN}--audio-beats.npz", allow_pickle=True)

    audio_feats = np.asarray(audio["features"], dtype=np.float64)
    vib_feats = np.asarray(vibration["features"], dtype=np.float64)
    valid = np.asarray(audio["valid_mask"], dtype=bool)
    n = audio_feats.shape[0]

    idx = np.arange(0, n, FEATURE_SNAPSHOT_STRIDE_S)

    def _zscore(x: np.ndarray) -> np.ndarray:
        ref = x[valid]
        mean = ref.mean(axis=0)
        std = ref.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
        z = (x - mean) / std
        return np.clip(z, -4.0, 4.0)

    audio_z = _zscore(audio_feats)[idx]
    vib_z = _zscore(vib_feats)[idx]

    beats_feats = np.asarray(beats["features"], dtype=np.float64)
    beats_norm = np.linalg.norm(beats_feats, axis=1)
    beats_valid = np.asarray(beats["valid_mask"], dtype=bool)
    ref = beats_norm[beats_valid]
    lo, hi = float(np.percentile(ref, 2)), float(np.percentile(ref, 98))
    span = max(hi - lo, 1e-9)
    beats_norm01 = np.clip((beats_norm - lo) / span, 0.0, 1.0)
    beats_norm01[~beats_valid] = 0.0

    return {
        "t_idx": [int(i) for i in idx],
        "stride_s": FEATURE_SNAPSHOT_STRIDE_S,
        "n_t": int(len(idx)),
        "n_audio": int(audio_feats.shape[1]),
        "n_vibration": int(vib_feats.shape[1]),
        "audio_names": [mda.shorten_feature_name(n) for n in audio["feature_names"]],
        "vibration_names": [mda.shorten_feature_name(n) for n in vibration["feature_names"]],
        "audio_b64": _f32_b64(audio_z),
        "vibration_b64": _f32_b64(vib_z),
        "n_beats": int(beats_feats.shape[1]),
        "beats_norm01": _round_list(beats_norm01[idx], 4),
    }


# ---------------------------------------------------------------------------
# Alert feed (candidate register, this session only)
# ---------------------------------------------------------------------------


def load_alerts(t0_ns: int) -> list[dict[str, Any]]:
    meta = json.loads(CANDIDATES_META.read_text(encoding="utf-8"))
    rows = [m for m in meta if m["session"] == RUN]
    rows.sort(key=lambda r: str(r["start_utc"]))
    out = []
    for r in rows:
        start_ns = int(datetime.fromisoformat(str(r["start_utc"])).timestamp() * 1e9)
        out.append(
            {
                "candidate_id": r["candidate_id"],
                "klass": r["class"],
                "start_s": round((start_ns - t0_ns) / 1e9, 3),
                "duration_s": r["duration_s"],
                "min_p": r["min_p"],
                "state_name": r["state_name"],
                "near_transition": bool(r["near_transition"]),
                "scada_state": r["scada_state"],
                "scada_transition": bool(r["scada_transition"]),
                "mode_mismatch": bool(r["mode_mismatch"]),
                "criterion_text": r["criterion_text"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_payload() -> dict[str, Any]:
    cfg = load_config()
    index = discover(cfg.data_root)

    primary = load_primary_timeline()
    t0_ns = primary.pop("t0_ns")

    payload: dict[str, Any] = {
        "run": RUN,
        "unit_name": UNIT_NAME,
        "label": "29.06.2026 — turbine operation",
        "generated_at": datetime.now(UTC).isoformat(),
        **primary,
        "sentinel": load_sentinel(),
        "levels": load_levels(),
        "scada": load_scada(index, cfg),
        "logmel": load_logmel_strip(),
        "features": load_feature_snapshot(),
        "alerts": load_alerts(t0_ns),
        "rings": {
            "generator": sc.render_ring_svg(
                "GENERATOR", sc.GENERATOR_MARKERS, size=196, interactive=False
            ),
            "turbine": sc.render_ring_svg(
                "TURBINE", sc.TURBINE_MARKERS, size=196, interactive=False
            ),
        },
    }
    return payload


def render_live_html(payload: dict[str, Any], template_path: Path, out_path: Path) -> Path:
    template = template_path.read_text(encoding="utf-8")
    data_json = ak._json_script_safe(payload)
    if "__LIVE_DATA_JSON__" not in template:
        raise ValueError(f"{template_path} has no __LIVE_DATA_JSON__ token")
    doc = template.replace("__LIVE_DATA_JSON__", data_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    payload = build_payload()
    out_path = render_live_html(payload, args.template, args.out)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"build_live_replay: wrote {out_path} ({size_mb:.2f} MB) -- {len(payload['segments'])} "
        f"segments, {len(payload['trace']['t_s'])} scored windows, {len(payload['alerts'])} alerts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
