"""Build one `docs/site/live*.html` control-room replay page PER recorded session
in `LIVE_SESSIONS` (real, native -- not iframed) from already-computed real
artifacts -- NEVER touches `ROWII_DATA_ROOT`/raw Gantner burst files. Everything a
page needs (state timeline, p-value stream, alarm feed, sentinel verdict, SCADA
line, per-stream level series, a downsampled log-mel strip, a downsampled feature
snapshot matrix) is precomputed into ONE embedded JSON payload per session,
injected into `docs/site/live_template.html` at a single `__LIVE_DATA_JSON__`
token, producing `docs/site/<LIVE_SESSIONS[run]["out"]>`. Reproducible from
`results/` + `docs/site/live_template.html` alone.

Two-pass build (`main`): pass 1 builds every requested run's payload (one
`RunContext` per run, `make_run_context`); pass 2 computes the cross-session nav
switcher (`pages_nav` -- a ribbon + tick summary per session, derived from each
run's OWN already-built payload) and injects it into every payload as
`payload["sessions_nav"]` before rendering -- so a page's nav always reflects
exactly the set of pages THIS invocation actually built (`--run`, repeatable;
default: every `LIVE_SESSIONS` entry). A `preflight` gate runs before pass 1: if
any requested run is missing a required real-data input, the build aborts naming
the missing path and the script that produces it -- never a partially-built page.

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
identical either way. fusion is the headline REPRESENTATION for every session in
`LIVE_SESSIONS`, not just 290626-tu -- only REGIME (frozen vs. recalibrate) varies
per session, read from the same `REGIME_BY_SESSION` lookup `candidate_kit.py`
itself uses.

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
pre-existing fact) despite every cache sharing the SAME `grid_window_ns` (1.0 s) and
being built from the same run's audio-stream offset. `grid_n_windows` itself is NOT
always shared, corrected here after this task's own full 4-session rebuild found a
real counter-case: each cache variant is its own intersection across only ITS OWN
streams (`rowii.pipeline.build_run_grid`), so a variant built from fewer/differently
-gapped streams can genuinely run longer or shorter than another --
`270626-pu_ph_pu_ph_pu_ph-1`'s `audio.npz` (18716 windows) vs. its `vibration.npz`
(19436 windows): the audio variant's own common grid is 720 windows (12 min)
shorter than vibration's for this run; per-variant grids intersect only their own
streams' first/last-file spans (`common_grid`'s `max(t0)`/`min(t_end)`,
`rowii.signals.windows.common_grid`) -- a mid-recording gap cannot be the cause
(verified against `_synthesize_run_header`: only each stream's FIRST/LAST file
feeds the grid). Both still share the
identical `grid_t0_ns`, so index `i` means the same real second in either one;
`load_levels`/`load_feature_snapshot` truncate every array they combine down to the
shortest cache's own length before indexing (`_truncate_to_common_length`) rather
than assuming agreement. For a 1 Hz visual replay the sub-second/metadata-only
`grid_t0_ns` disagreement above, and any truncated tail beyond the primary (fusion)
timeline's own `duration_s`, are both immaterial; index `i` is treated as "second `i`
of the replay" throughout.
"""
from __future__ import annotations

import base64
import dataclasses
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
import build_live_audio as lva  # noqa: E402
import candidate_kit as ck  # noqa: E402
import make_demo_assets as mda  # noqa: E402
import run_step1 as rs1  # noqa: E402
import site_common as sc  # noqa: E402
from feature_labels import humanize_feature_name  # noqa: E402

from rowii.config import load_config  # noqa: E402
from rowii.io.dataset import discover, run_utc_offset_ns  # noqa: E402
from rowii.pipeline import build_run_grid  # noqa: E402
from rowii.scada.labels import gt_labels, load_scada_window_means  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = _SCRIPTS_DIR.parent
RESULTS_ROOT = REPO_ROOT / "results"
CACHE_DIR = RESULTS_ROOT / "cache"
DEFAULT_TEMPLATE = REPO_ROOT / "docs" / "site" / "live_template.html"

REPRESENTATION = "fusion"
"""The primary state/score/alarm arm for every replay page (module docstring) --
NOT per-session: every `LIVE_SESSIONS` entry uses fusion as its headline
representation, only `RunContext.regime` (frozen vs. recalibrate) varies by
session."""
SENTINEL_REPRESENTATION = "audio-beats"
"""s1/s2 always read the audio-beats mode bank/raw caches, representation-independent
by construction (`run_once_calibrated.py`)."""
UNIT_NAME = "ROWII Machine 1 — Rodundwerk II"
AUDIO_DIR = REPO_ROOT / "docs" / "site" / "assets" / "live"
"""Written by `scripts/build_live_audio.py` (the one step in this page's build
that DOES touch `ROWII_DATA_ROOT` -- run rarely/manually, its two `.m4a` outputs
committed like any other `docs/site/assets/` file). Reading its small, already-
committed sidecar JSON (`RunContext.audio_meta_json`, `load_audio` below) keeps
`build_live_replay.py` itself honest about its own module docstring's "NEVER
touches ROWII_DATA_ROOT" -- it folds in already-extracted audio metadata, never
raw sensor data."""

LIVE_SESSIONS: dict[str, dict[str, object]] = {
    "290626-tu": {"out": "live.html", "display_name": "Turbine day",
                  "blurb": "quiet reference day", "events": False},
    "080726-pu_strikes": {"out": "live-080726-pu-strikes.html", "display_name": "Hammer-strike day",
                          "blurb": "controlled tap tests at each sensor", "events": True},
    "010726-tu1-morning": {"out": "live-010726-tu1-morning.html", "display_name": "Morning session",
                           "blurb": "turbine start-up in the morning", "events": False},
    "270626-pu_ph_pu_ph_pu_ph-1": {"out": "live-270626-cycles.html", "display_name": "Cycling day",
                                   "blurb": "pump ⇄ phase-shifter cycles", "events": False},
}
"""Registry of every recorded session this build script can turn into a `live*
.html` replay page (`"out"` is the filename under `docs/site/`) -- SINGLE SOURCE
OF TRUTH for `session_summary`, `make_run_context`, `pages_nav`, and `main`'s own
`--run` validation/default. `main` builds every entry here by default, or exactly
the `--run`-selected subset; `preflight` gates each requested run on its own
real-data inputs actually being present (module docstring) before ANY page is
written."""


@dataclasses.dataclass(frozen=True)
class RunContext:
    """Everything a `load_*`/`build_payload` function needs for ONE session's
    build -- replaces the single hardcoded `RUN`-derived module constants this
    file used before it built more than `live.html`. Built once per requested run
    by `make_run_context` and threaded explicitly through every function below
    (never read off a module global), so `main` can build any subset of
    `LIVE_SESSIONS` in one process without cross-run state leaking."""

    run: str
    representation: str
    regime: str
    sentinel_representation: str
    unit_name: str
    monitor_dir: Path
    """`results/step2/once-calibrated/<representation>/monitor/<run>/<regime>/`
    for a normally-monitored session, or `results/monitor-ext/<run>/
    <representation>/` (no `<regime>` path segment there) for a
    `candidate_kit._MONITOR_EXT_SESSIONS` coverage-extension session --
    `candidate_kit._alarms_path_for`'s own branch, reused here (not re-derived)
    so this can never quietly drift from the SAME lookup `candidate_kit.py`'s own
    `select_session` uses to find each session's `alarms.parquet`."""
    sentinel_json: Path
    fusion_json: Path
    """`results/step2/once-calibrated/<representation>/<representation>.json` --
    the SECOND file `load_sentinel` conditionally opens (its `regimes` table),
    factored out here (not a local variable inside that function, as it was
    before this field existed) so `preflight` can check its existence too,
    from the SAME formula, without duplicating the path-join."""
    candidates_csv: Path
    candidates_meta: Path
    audio_dir: Path
    audio_meta_json: Path


def make_run_context(run: str) -> RunContext:
    """`RunContext` for *run* -- `run` should be a `LIVE_SESSIONS` key (`main`
    validates this before calling; an unregistered run instead fails naturally
    inside `ck.REGIME_BY_SESSION[run]` with a `KeyError`)."""
    regime = ck.REGIME_BY_SESSION[run]
    monitor_dir = ck._alarms_path_for(RESULTS_ROOT, REPRESENTATION, run, regime).parent
    return RunContext(
        run=run,
        representation=REPRESENTATION,
        regime=regime,
        sentinel_representation=SENTINEL_REPRESENTATION,
        unit_name=UNIT_NAME,
        monitor_dir=monitor_dir,
        sentinel_json=(
            RESULTS_ROOT / "step2" / "once-calibrated" / SENTINEL_REPRESENTATION
            / f"{SENTINEL_REPRESENTATION}.json"
        ),
        fusion_json=(
            RESULTS_ROOT / "step2" / "once-calibrated" / REPRESENTATION / f"{REPRESENTATION}.json"
        ),
        candidates_csv=RESULTS_ROOT / "candidate-kit" / "candidates.csv",
        candidates_meta=RESULTS_ROOT / "candidate-kit" / "candidates_meta.json",
        audio_dir=AUDIO_DIR,
        audio_meta_json=AUDIO_DIR / f"{run}_audio_meta.json",
    )


AUDIO_COVERAGE_SLACK_S = 1.0
"""Seconds of tolerance `_check_audio_covers_replay` allows the extracted audio to
fall short of the replay's own `[t0_utc, t0_utc + duration_s]` span by, before
failing the build loudly -- covers only float/ISO round-trip noise, not a genuine
extraction gap (the real audio in practice overshoots this span on both ends, see
that function's own docstring)."""

FEATURE_SNAPSHOT_STRIDE_S = 4
"""Feature-snapshot heatmap cadence: one column every N seconds of replay time --
231 raw dims (135 audio + 96 vibration) at 1 Hz for a whole ~4.4 h day would be
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


def humanize_names(raw_names: list[str]) -> list[str]:
    """Human-readable labels for a list of raw handcrafted feature names
    (`feature_labels.humanize_feature_name`, Task 5) -- used for the feature
    snapshot panel's `audio_names`/`vib_names` (`load_feature_snapshot` below)."""
    return [humanize_feature_name(n) for n in raw_names]


def _truncate_to_common_length(*arrays: np.ndarray, label: str) -> tuple[np.ndarray, ...]:
    """Every *array* truncated (axis 0 only -- a 2D feature matrix keeps every
    column) to the SHORTEST one's own length -- a no-op when every array already
    agrees, which is the common case (`290626-tu`/`080726-pu_strikes`/
    `010726-tu1-morning`: `audio.npz` and `vibration.npz` share one
    `grid_n_windows`).

    Exists because that agreement is NOT guaranteed (module docstring's own
    "Time alignment" note was wrong to claim it always holds -- found running
    this exact function's callers for real, not a documented pre-existing
    fact): each cache variant is its own intersection across only ITS OWN
    streams (`rowii.pipeline.build_run_grid`), so a variant built from
    fewer/differently-gapped streams can genuinely run longer or shorter than
    another. `270626-pu_ph_pu_ph_pu_ph-1` is the one real `LIVE_SESSIONS` case:
    `audio.npz`'s 2-mic intersection is 18716 windows, `vibration.npz`'s
    2-accelerometer intersection is 19436 windows -- the audio variant's own
    common grid is 720 windows (12 min) shorter than vibration's for this run;
    per-variant grids intersect only their own streams' first/last-file spans
    (`rowii.signals.windows.common_grid`); a mid-recording gap cannot be the
    cause. Both caches still share the identical `grid_t0_ns`/1.0 s
    `grid_window_ns`, so index `i` means the same real second in either one;
    truncating the longer array down to the shorter one's length is therefore
    EXACT alignment over the shared prefix, not a lossy approximation, and the
    discarded vibration tail is beyond the primary (fusion) timeline's own
    `duration_s` anyway -- the replay's own playhead never reaches it. Used by
    `load_levels`/`load_feature_snapshot`, both of which combine an
    `audio.npz`-derived `valid_mask`/length with `vibration.npz`-derived
    arrays.

    *label* identifies the caller for the `logger.warning` this emits WHEN
    truncation actually discards data (the common length is shorter than the
    longest input) -- e.g. `f"load_levels[{ctx.run}]"`. Silent (no warning) in
    the equal-length common case; reviewer finding (round 1) was that a real
    mismatch like the 270626 case above was previously absorbed with no log
    trace at all.
    """
    lengths = [a.shape[0] for a in arrays]
    n = min(lengths)
    if n < max(lengths):
        discarded = [length - n for length in lengths]
        logger.warning(
            "%s: _truncate_to_common_length discarding data -- input lengths %s, "
            "common length %d, discarded per array %s",
            label, lengths, n, discarded,
        )
    return tuple(a[:n] for a in arrays)


def session_summary(run: str, *, duration_s: float, n_episodes: int) -> dict[str, object]:
    """One `LIVE_SESSIONS` entry's UI-facing summary (`payload["session"]`,
    `build_payload` below) -- `date_label` is derived from the session id's own
    `DDMMYY` prefix (`run[:6]`) via `datetime.strptime`, never a hardcoded
    weekday table, so it can never drift out of sync with the calendar."""
    meta = LIVE_SESSIONS[run]
    day = datetime.strptime(run[:6], "%d%m%y").replace(tzinfo=UTC)
    date_label = day.strftime("%a · %d %b %Y").upper()
    return {
        "id": run,
        "display_name": meta["display_name"],
        "blurb": meta["blurb"],
        "events": bool(meta["events"]),
        "date_label": date_label,
        "duration_s": float(duration_s),
        "n_episodes": int(n_episodes),
    }


# ---------------------------------------------------------------------------
# Primary timeline: fusion/frozen monitor (state, score, p-value, alarm, segments)
# ---------------------------------------------------------------------------


def load_primary_timeline(ctx: RunContext) -> dict[str, Any]:
    segments_df = pd.read_csv(ctx.monitor_dir / "segments.csv")
    segments_df["start_utc"] = pd.to_datetime(segments_df["start_utc"], utc=True)
    segments_df["end_utc"] = pd.to_datetime(segments_df["end_utc"], utc=True)
    t0 = segments_df["start_utc"].min()
    t0_ns = int(t0.value)
    duration_s = (int(segments_df["end_utc"].max().value) - t0_ns) / 1e9

    notes_text = (ctx.monitor_dir / "monitor_notes.md").read_text(encoding="utf-8")
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

    alarms_df = pd.read_parquet(ctx.monitor_dir / "alarms.parquet")
    scored = alarms_df.loc[alarms_df["role"] == "scored"].sort_values("t_utc_ns")
    if scored.empty:
        raise ValueError(f"{ctx.run}: no scored windows in {ctx.monitor_dir / 'alarms.parquet'}")
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

    alarm_seg_df = pd.read_csv(ctx.monitor_dir / "alarm_segments.csv")
    n_alarm_episodes = len(alarm_seg_df)
    n_scored = len(scored)
    n_alarmed_windows = int(scored["alarm"].sum())

    # Some LIVE_SESSIONS entries (e.g. 080726-pu_strikes) have controlled acoustic
    # events with their own eval_events/-based FAR computation elsewhere
    # (scripts/eval_events.py); this function never reads that directory --
    # realized FAR/budget for the SENTINEL gauge always comes from the regimes
    # table instead (load_sentinel below), the same reporting convention
    # run_once_calibrated.py itself uses for every day it monitors.

    return {
        "t0_ns": t0_ns,
        "t0_utc": t0.isoformat(),
        "duration_s": round(duration_s, 3),
        "representation": ctx.representation,
        "regime": ctx.regime,
        "states": states,
        "segments": segments,
        "trace": trace,
        "n_scored": n_scored,
        "n_alarmed_windows": n_alarmed_windows,
        "n_alarm_episodes": n_alarm_episodes,
    }


def sentinel_payload(
    trig: dict[str, Any] | None, regime: dict[str, Any] | None
) -> dict[str, Any]:
    """Map one `trigger_log` row (*trig*) and one `regimes` row (*regime*) --
    either or both may be `None`, since not every `LIVE_SESSIONS` entry was
    scored by the once-calibrated sentinel driver -- to `payload["sentinel"]`'s
    own three-shape contract (review round 2 correction: `decision` is REAL
    recorded sentinel data, carried through verbatim whenever *trig* exists --
    never discarded just because *regime* happens to be missing).

    - both present -> the original full shape, plus `"available": "full"`.
    - *trig* only (*regime* is `None`) -> exactly the fields the trigger row
      itself carries (era, s1_rate, s1_threshold, s1_fired, s2_fired,
      s2_attribution, decision -- verbatim, since it's the sentinel's own
      real recorded decision, not something *regime* derives) plus
      `"available": "trigger_only"` and an explanatory `"note"`. No FAR field
      (`realized_far` etc.) is present: those are only ever derivable from
      *regime* -- what's genuinely missing here is the regime REPLAY/FAR
      bookkeeping, never the decision.
    - neither present (*trig* is `None`) -> `{"available": "none", "note":
      ...}` and NOTHING else -- no fabricated rates of any kind.

    Pure (no file I/O) so it's unit-testable directly against hand-built rows,
    independent of `load_sentinel`'s own two-JSON-file read."""
    if trig is None:
        return {
            "available": "none",
            "note": (
                "session not scored by the once-calibrated sentinel driver; alarms come "
                "from the frozen-threshold monitoring extension"
            ),
        }
    trig_fields = {
        "era": trig["era"],
        "s1_rate": round(trig["s1_rate"], 6),
        "s1_threshold": round(trig["s1_threshold"], 6),
        "s1_fired": bool(trig["s1_fired"]),
        "s2_fired": bool(trig["s2_fired"]),
        "s2_attribution": trig["s2_attribution"],
        "decision": trig["decision"],
    }
    if regime is None:
        return {
            **trig_fields,
            "available": "trigger_only",
            "note": (
                "sentinel decision recorded; no regime replay (FAR evaluation) exists for "
                "this session"
            ),
        }
    return {
        **trig_fields,
        "nominal_alpha": 0.05,
        "realized_far": round(regime["once_triggered_far"], 6),
        "always_frozen_far": round(regime["always_frozen_far"], 6),
        "always_recalibrate_far": round(regime["always_recalibrate_far"], 6),
        "far_basis": regime["far_basis"],
        "available": "full",
    }


def load_sentinel(ctx: RunContext) -> dict[str, Any]:
    """`sentinel_payload` fed from *ctx*'s real two files -- `regime` is only
    looked up (a second file read) when a `trig` row exists at all, since with
    no `trig` row `sentinel_payload` returns `"none"` regardless of *regime*."""
    payload = json.loads(ctx.sentinel_json.read_text(encoding="utf-8"))
    trig = next((t for t in payload["trigger_log"] if t["run"] == ctx.run), None)
    regime = None
    if trig is not None:
        fusion_json = json.loads(ctx.fusion_json.read_text(encoding="utf-8"))
        regime = next((r for r in fusion_json["regimes"] if r["run"] == ctx.run), None)
    return sentinel_payload(trig, regime)


# ---------------------------------------------------------------------------
# Per-stream RMS level series (sensor panel ring pulsing)
# ---------------------------------------------------------------------------


def load_levels(ctx: RunContext) -> dict[str, Any]:
    """One 1 Hz level series per raw stream -- mic streams use channel 0's
    `log_rms` (the pipeline's own established "channel 0 of each stream"
    convention, `make_demo_assets.MONO_CHANNEL_INDEX`); vibration streams use the
    MEAN `log_rms` over every channel the handcrafted cache actually carries for
    that stream (no established single-channel convention exists for vibration,
    module docstring's own honesty note on the ring's real position<->channel
    mapping never being verified)."""
    audio = np.load(CACHE_DIR / f"{ctx.run}--audio.npz", allow_pickle=True)
    vibration = np.load(CACHE_DIR / f"{ctx.run}--vibration.npz", allow_pickle=True)

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
    # audio.npz and vibration.npz are not guaranteed to share one grid_n_windows
    # (_truncate_to_common_length's own docstring: 270626-pu_ph_pu_ph_pu_ph-1 is
    # the real case where they don't) -- truncate every series to the shortest
    # one's length before `valid` is used to index any of them.
    gen_mic, tur_mic, gen_vib, tur_vib, valid = _truncate_to_common_length(
        gen_mic, tur_mic, gen_vib, tur_vib, valid, label=f"load_levels[{ctx.run}]"
    )
    gen_mic = np.asarray(gen_mic, dtype=np.float64)
    tur_mic = np.asarray(tur_mic, dtype=np.float64)
    gen_vib = np.asarray(gen_vib, dtype=np.float64)
    tur_vib = np.asarray(tur_vib, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

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


def load_scada(ctx: RunContext, index: Any, cfg: Any) -> dict[str, Any]:
    session_scada = ck.load_session_scada(index, ctx.run, cfg)

    # Load bin (Step 1 stage's "load bin" readout): re-derives `rowii.scada.labels.
    # gt_labels`'s own `load_bin` column (quantile-binned power, turbine/pump windows
    # only, n_load_bins=3 by default) -- `load_session_scada` itself only keeps
    # `state`, not `load_bin`, so this mirrors that function's own internals
    # (`rs1`/`build_run_grid`/`load_scada_window_means`/`gt_labels`, imported
    # directly here -- not via `candidate_kit`'s own namespace, which mypy's
    # `no_implicit_reexport` correctly refuses to treat as a public API) rather
    # than duplicating a second real-data read.
    run = mda._get_run(index, ctx.run)
    betriebsdaten = index.betriebsdaten_by_day.get(run.day_root, [])
    offset_ns = run_utc_offset_ns(run)
    grid = build_run_grid(run, rs1._AUDIO_STREAMS, cfg.window.window_s, offset_ns=offset_ns)
    matched_files = rs1._betriebsdaten_for_grid(betriebsdaten, grid)
    scada_means = load_scada_window_means(matched_files, grid, audio_run_offset_ns=offset_ns)
    gt = gt_labels(scada_means, cfg.gt, window_s=cfg.window.window_s)
    load_bin = gt["load_bin"].to_numpy()

    def _round_or_none(v: float | None, digits: int) -> float | None:
        return None if v is None or math.isnan(v) else round(float(v), digits)

    # flow_net_m3s/ks_valve (SCADA CONTEXT PANEL's 3rd/4th mini-axis, v2 site
    # redesign): the SAME 1 Hz resampling `candidate_kit.build_extended_readout_
    # series` already does for review.html's live readout, called here on
    # *session_scada*'s own full-run span (`window_start_utc[0]`, `n` windows --
    # window_s == 1.0s in this pipeline, so this reproduces `power_mw`/`speed_rpm`'s
    # own index alignment exactly) rather than reimplementing its NaN-to-null
    # resampling logic a second time. power_mw/speed_rpm above are discarded here
    # since they're already extracted, unresampled, straight from *session_scada*.
    n = len(session_scada.power_mw)
    _, _, flow_net_m3s, ks_valve = ck.build_extended_readout_series(
        session_scada, session_scada.window_start_utc[0], n
    )

    return {
        "has_scada": session_scada.has_scada,
        "n": n,
        "power_mw": [_round_or_none(v, 3) for v in session_scada.power_mw],
        "speed_rpm": [_round_or_none(v, 2) for v in session_scada.speed_rpm],
        "flow_net_m3s": [_round_or_none(v, 3) for v in flow_net_m3s],
        "ks_valve": [_round_or_none(v, 2) for v in ks_valve],
        "scada_state": session_scada.state,
        "load_bin": [int(v) for v in load_bin],
        "n_load_bins": int(cfg.gt.n_load_bins),
    }


# ---------------------------------------------------------------------------
# Log-mel scrolling strip
# ---------------------------------------------------------------------------


def load_logmel_strip(ctx: RunContext) -> dict[str, Any]:
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

    d = np.load(CACHE_DIR / f"{ctx.run}--logmel.npz", allow_pickle=True)
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


def load_feature_snapshot(ctx: RunContext) -> dict[str, Any]:
    audio = np.load(CACHE_DIR / f"{ctx.run}--audio.npz", allow_pickle=True)
    vibration = np.load(CACHE_DIR / f"{ctx.run}--vibration.npz", allow_pickle=True)
    beats = np.load(CACHE_DIR / f"{ctx.run}--audio-beats.npz", allow_pickle=True)

    audio_feats = np.asarray(audio["features"], dtype=np.float64)
    vib_feats = np.asarray(vibration["features"], dtype=np.float64)
    valid = np.asarray(audio["valid_mask"], dtype=bool)
    # See _truncate_to_common_length's own docstring / load_levels's identical
    # comment: audio.npz and vibration.npz do not always share one
    # grid_n_windows (270626-pu_ph_pu_ph_pu_ph-1 is the real counter-case).
    audio_feats, vib_feats, valid = _truncate_to_common_length(
        audio_feats, vib_feats, valid, label=f"load_feature_snapshot[{ctx.run}]"
    )
    audio_feats = np.asarray(audio_feats, dtype=np.float64)
    vib_feats = np.asarray(vib_feats, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
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
        "audio_names": humanize_names([str(n) for n in audio["feature_names"]]),
        "vib_names": humanize_names([str(n) for n in vibration["feature_names"]]),
        "audio_b64": _f32_b64(audio_z),
        "vibration_b64": _f32_b64(vib_z),
        "n_beats": int(beats_feats.shape[1]),
        "beats_norm01": _round_list(beats_norm01[idx], 4),
    }


# ---------------------------------------------------------------------------
# Alert feed (candidate register, this session only)
# ---------------------------------------------------------------------------


def load_alerts(ctx: RunContext, t0_ns: int) -> list[dict[str, Any]]:
    meta = json.loads(ctx.candidates_meta.read_text(encoding="utf-8"))
    rows = [m for m in meta if m["session"] == ctx.run]
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
# Live audio (both mic streams, scripts/build_live_audio.py's own output)
# ---------------------------------------------------------------------------


def load_audio(ctx: RunContext) -> dict[str, Any]:
    """The `{"gen": {...}, "tur": {...}}` sidecar `scripts/build_live_audio.py`
    already wrote under `ctx.audio_dir`, unchanged except for dropping its own
    `"stream"` key (`assets/live.js` never needs the raw stream name, only the
    `"gen"`/`"tur"` key it's already nested under and the human-readable
    `"label"`).

    Raises:
        FileNotFoundError: if `ctx.audio_meta_json` does not exist -- run
            `.venv/bin/python scripts/build_live_audio.py --run <run>` once
            (requires `ROWII_DATA_ROOT`) before building this page; its two
            `.m4a` outputs plus this sidecar are then committed like any other
            `docs/site/assets/` file, so every subsequent `build_live_replay.py`
            run (which never touches `ROWII_DATA_ROOT` itself) picks them up for
            free.
    """
    if not ctx.audio_meta_json.exists():
        raise FileNotFoundError(
            f"{ctx.audio_meta_json} not found -- run `.venv/bin/python "
            f"scripts/build_live_audio.py --run {ctx.run}` first (see that script's own "
            "module docstring)"
        )
    payload = json.loads(ctx.audio_meta_json.read_text(encoding="utf-8"))
    streams = payload["streams"]
    return {key: {k: v for k, v in meta.items() if k != "stream"} for key, meta in streams.items()}


def _check_audio_covers_replay(
    ctx: RunContext, t0_utc: str, duration_s: float, audio: dict[str, Any]
) -> None:
    """Fail loudly (not silently ship a broken sync) if either stream's extracted
    audio does not actually span the replay's own `[t0_utc, t0_utc + duration_s]`
    timeline -- the exact mapping `assets/live.js` applies client-side
    (`build_live_audio.audio_offset_s`, the pinned reference `tests/
    test_build_live_audio.py` checks) would then request a `currentTime` before 0
    or past the `<audio>` element's own duration at some point during playback.
    `AUDIO_COVERAGE_SLACK_S` tolerates only float/ISO round-trip noise -- on the
    real 290626-tu extraction the audio comfortably outlasts the replay on the
    start side and ends within a few MILLISECONDS of it on the end side (both
    are ultimately derived from the same recording session's own real burst-file
    boundaries), which is exactly why `assets/live.js`'s own `END_EPSILON_S`
    safety clamp exists on the client side too.

    Raises:
        ValueError: if a stream's audio starts too late or ends too early to
            cover the full replay span.
    """
    for key, meta in audio.items():
        start_off = lva.audio_offset_s(0.0, t0_utc, str(meta["start_utc"]))
        end_off = lva.audio_offset_s(duration_s, t0_utc, str(meta["start_utc"]))
        if start_off < -AUDIO_COVERAGE_SLACK_S:
            raise ValueError(
                f"{ctx.run}: audio[{key!r}] starts {-start_off:.3f}s after the replay's own "
                "t0_utc -- the opening seconds of the replay would have no audio"
            )
        if end_off > float(meta["duration_s"]) + AUDIO_COVERAGE_SLACK_S:
            raise ValueError(
                f"{ctx.run}: audio[{key!r}] ends {end_off - float(meta['duration_s']):.3f}s "
                "before the replay's own t0_utc + duration_s -- the closing seconds of the "
                "replay would have no audio"
            )


# ---------------------------------------------------------------------------
# Preflight -- required real-data inputs for one run, checked BEFORE any page
# in the requested build is written (no partial builds).
# ---------------------------------------------------------------------------


def preflight(ctx: RunContext) -> list[str]:
    """Human-readable problems blocking *ctx*'s build (empty list if *ctx.run* is
    ready) -- each entry names the missing path and the script that produces it,
    so `main`'s abort message is immediately actionable. Checks `ctx.sentinel_json`/
    `ctx.fusion_json` exist as FILES, same as every other input here (an absent
    file would otherwise bypass this function's promise entirely and crash
    mid-build with a bare `FileNotFoundError`) -- but deliberately does NOT check
    whether either file actually HAS a `trigger_log`/`regimes` ROW for this run.
    Row absence is real and expected (270626-pu_ph_pu_ph_pu_ph-1 has a
    trigger_log row but no regimes row; 010726-tu1-morning has neither -- neither
    was ever scored/monitored by the once-calibrated driver), and is NOT a
    preflight abort: `load_sentinel`/`sentinel_payload` degrade it honestly at
    build time instead (`payload["sentinel"]["available"]` in `{"full",
    "trigger_only", "none"}`), since which of a session's real rows exist is a
    data-production fact this function has no basis to gate the whole build on."""
    problems: list[str] = []

    for variant in ("audio", "vibration", "logmel", "audio-beats"):
        cache_path = CACHE_DIR / f"{ctx.run}--{variant}.npz"
        if not cache_path.is_file():
            problems.append(
                f"{ctx.run!r}: missing {cache_path} -- run `scripts/warm_cache.py "
                f"--runs {ctx.run} --variants {variant}` first"
            )

    for name in ("segments.csv", "monitor_notes.md", "alarms.parquet", "alarm_segments.csv"):
        monitor_path = ctx.monitor_dir / name
        if not monitor_path.is_file():
            producer = (
                "scripts/monitor.py --snapshot models/adapted/monitor_pool_b1_fusion_named.npz "
                "--thresholds recalibrate (coverage-extension session -- see "
                "candidate_kit._MONITOR_EXT_SESSIONS)"
                if ctx.run in ck._MONITOR_EXT_SESSIONS
                else "scripts/run_once_calibrated.py"
            )
            problems.append(f"{ctx.run!r}: missing {monitor_path} -- run `{producer}` first")

    for sentinel_path in (ctx.sentinel_json, ctx.fusion_json):
        if not sentinel_path.is_file():
            problems.append(
                f"{ctx.run!r}: missing {sentinel_path} -- run "
                "`scripts/run_once_calibrated.py` first"
            )

    if not ctx.candidates_meta.is_file():
        problems.append(
            f"{ctx.run!r}: missing {ctx.candidates_meta} -- run "
            "`scripts/candidate_kit.py select` first"
        )
    else:
        meta = json.loads(ctx.candidates_meta.read_text(encoding="utf-8"))
        if not any(row["session"] == ctx.run for row in meta):
            problems.append(
                f"{ctx.run!r}: no rows for this session in {ctx.candidates_meta} -- run "
                "`scripts/candidate_kit.py select` first"
            )

    for suffix in ("gen", "tur"):
        audio_path = ctx.audio_dir / f"{ctx.run}_{suffix}.m4a"
        if not audio_path.is_file():
            problems.append(
                f"{ctx.run!r}: missing {audio_path} -- run `.venv/bin/python "
                f"scripts/build_live_audio.py --run {ctx.run}` first"
            )
    if not ctx.audio_meta_json.is_file():
        problems.append(
            f"{ctx.run!r}: missing {ctx.audio_meta_json} -- run `.venv/bin/python "
            f"scripts/build_live_audio.py --run {ctx.run}` first"
        )

    return problems


# ---------------------------------------------------------------------------
# Cross-session nav switcher
# ---------------------------------------------------------------------------


def pages_nav(
    summaries: dict[str, dict[str, Any]], timelines: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """The cross-session switcher payload for every session THIS build actually
    produced: `summaries[run]` is that run's own `payload["session"]`
    (`session_summary`'s output); `timelines[run]` is `{"segments": payload
    ["segments"], "tick_s": [a["start_s"] for a in payload["alerts"]]}` -- built
    by `main` from each run's already-assembled payload (segments/tick times are
    never recomputed here). Iterates `LIVE_SESSIONS` (not `summaries`) so nav
    entries always come out in registry order; a `LIVE_SESSIONS` id absent from
    *summaries* (a `--run`-subset build that didn't include it) is simply skipped
    -- never a dangling link to a page this invocation didn't write."""
    nav = []
    for run, meta in LIVE_SESSIONS.items():
        if run not in summaries:
            continue
        s = summaries[run]
        tl = timelines[run]
        dur = float(s["duration_s"]) or 1.0
        ribbon = [
            {"start_frac": round(seg["start_s"] / dur, 4),
             "end_frac": round(seg["end_s"] / dur, 4),
             "state": seg["state_name"]}
            for seg in tl["segments"]
        ]
        ticks = [round(t / dur, 4) for t in tl["tick_s"]]
        nav.append({"id": run, "href": meta["out"], "display_name": s["display_name"],
                    "date_label": s["date_label"], "duration_s": s["duration_s"],
                    "n_episodes": s["n_episodes"], "events": s["events"],
                    "blurb": s["blurb"], "ribbon": ribbon, "ticks": ticks})
    return nav


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_payload(ctx: RunContext, index: Any, cfg: Any) -> dict[str, Any]:
    primary = load_primary_timeline(ctx)
    t0_ns = primary.pop("t0_ns")
    audio = load_audio(ctx)
    _check_audio_covers_replay(ctx, primary["t0_utc"], primary["duration_s"], audio)
    alerts = load_alerts(ctx, t0_ns)

    # "label" has no reader in docs/site/assets/live.js today (superseded by the
    # structured "session" object below) -- kept only for payload back-compat,
    # so it must still be a genuinely per-run string rather than 290626-tu's own
    # hardcoded date, derived the SAME way `session_summary`'s own `date_label`
    # is (`run[:6]` -> `datetime.strptime`), not re-hardcoded per session.
    day = datetime.strptime(ctx.run[:6], "%d%m%y").replace(tzinfo=UTC)
    display_name = str(LIVE_SESSIONS[ctx.run]["display_name"])
    label = f"{day.strftime('%d.%m.%Y')} — {display_name.lower()}"

    payload: dict[str, Any] = {
        "run": ctx.run,
        "unit_name": ctx.unit_name,
        "label": label,
        "generated_at": datetime.now(UTC).isoformat(),
        **primary,
        "sentinel": load_sentinel(ctx),
        "levels": load_levels(ctx),
        "scada": load_scada(ctx, index, cfg),
        "logmel": load_logmel_strip(ctx),
        "features": load_feature_snapshot(ctx),
        "alerts": alerts,
        "audio": audio,
        "session": session_summary(
            ctx.run, duration_s=primary["duration_s"], n_episodes=len(alerts)
        ),
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
    parser.add_argument(
        "--run", action="append", metavar="RUN",
        help="Session to build (repeatable; default: every LIVE_SESSIONS entry). "
        f"Valid ids: {', '.join(LIVE_SESSIONS)}.",
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    args = parser.parse_args(argv)

    runs: list[str] = args.run if args.run else list(LIVE_SESSIONS)
    unknown = [r for r in runs if r not in LIVE_SESSIONS]
    if unknown:
        raise SystemExit(
            f"build_live_replay: unknown --run {unknown!r} -- valid ids: "
            f"{', '.join(LIVE_SESSIONS)}"
        )

    contexts = {run: make_run_context(run) for run in runs}

    # Preflight EVERY requested run before building ANY of them -- "no partial
    # pages" means the check for run N must not happen only after runs 1..N-1
    # were already written.
    problems: dict[str, list[str]] = {}
    for run, ctx in contexts.items():
        probs = preflight(ctx)
        if probs:
            problems[run] = probs
    if problems:
        lines = ["build_live_replay: preflight failed -- fix these before building "
                  "(no pages written):"]
        for run, probs in problems.items():
            lines.append(f"  {run}:")
            lines.extend(f"    - {p}" for p in probs)
        raise SystemExit("\n".join(lines))

    cfg = load_config()
    index = discover(cfg.data_root)

    # Pass 1: build every requested run's payload.
    payloads: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    timelines: dict[str, dict[str, Any]] = {}
    for run, ctx in contexts.items():
        payload = build_payload(ctx, index, cfg)
        payloads[run] = payload
        summaries[run] = payload["session"]
        timelines[run] = {
            "segments": payload["segments"],
            "tick_s": [a["start_s"] for a in payload["alerts"]],
        }

    # Pass 2: inject the cross-session nav (built from pass 1's own payloads) into
    # every payload, then render.
    nav = pages_nav(summaries, timelines)
    for run, payload in payloads.items():
        payload["sessions_nav"] = nav
        out_path = REPO_ROOT / "docs" / "site" / str(LIVE_SESSIONS[run]["out"])
        render_live_html(payload, args.template, out_path)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(
            f"build_live_replay: wrote {out_path} ({size_mb:.2f} MB) -- "
            f"{len(payload['segments'])} segments, {len(payload['trace']['t_s'])} scored windows, "
            f"{len(payload['alerts'])} alerts"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
