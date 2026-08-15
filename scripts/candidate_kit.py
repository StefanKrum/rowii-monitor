"""Candidate-review kit: the deliverable handover artifact for windows where the
models suspect an anomaly on days WITHOUT induced anomalies -- the "candidate
register explicitly not ground truth" ruling (Jinyuan, 2026-08-04) resolved into a
separate, VERSION-CONTROLLED qualitative listening study rather than a treated-as-
labels register. The old `results/step2/candidate_register.md` lost its own manual
assessment tags because `results/` is gitignored (see `.gitignore`); this kit's
`compile` subcommand writes the reviewed subset to `docs/assessments/` instead,
which IS tracked.

Three subcommands:

    select     Build `results/candidate-kit/candidates.csv` (all 11 sessions, one row
               per candidate) from the once-calibrated FUSION monitor's `alarms.
               parquet` (SUSTAINED class: episodes of >=3 CONTIGUOUS `role="scored"`
               windows at p<0.01, grouped by strict window-index adjacency -- a
               skipped/invalid grid window never bridges two episodes) plus the
               once-calibrated AUDIO-BEATS monitor's `alarms.parquet` for the SAME
               (day, regime) pair (TRANSIENT class: single p<0.001 scored windows,
               deduplicated within +/-5s of a more extreme neighbour). Both classes
               are merged and extremity-overlap-resolved (the more extreme of two
               overlapping candidates wins; `suppress_overlaps`/`dedupe_by_radius`
               share one greedy primitive, `_greedy_suppress`), then capped at the
               top 25 most extreme (lowest min p-value) per session. Three of the 11
               sessions (`270626-pu_ph_pu_ph_pu_ph-1`, `010726-tu1-morning`,
               `010726-tu2` -- `_MONITOR_EXT_SESSIONS`, coverage extension, Stefan's
               decision 2026-08-16) read their `alarms.parquet` from
               `results/monitor-ext/<session>/<representation>/` instead (produced by
               running `scripts/monitor.py --thresholds recalibrate` directly against
               the same once-calibrated fusion/audio-beats snapshots, since none of
               the three was ever a monitored day in `run_once_calibrated.py`'s own
               `_REPLAY` set) -- the once-calibrated tree itself is untouched.
               `010726-tu1-morning`/`010726-tu2` are additionally tagged
               `in_sample=True` (`IN_SAMPLE_SESSIONS`): both are literal members of
               the scoring snapshot's own fit pool (`models/adapted/monitor_pool_b1_
               {fusion,audio_beats}_named.npz`'s `fit_run`), so their candidates carry
               that caveat; `270626-pu_ph_pu_ph_pu_ph-1` is genuinely held-out despite
               having no SCADA either. `270626-pu_ph_pu_ph_pu_ph-1` has no
               Betriebsdaten at all, so every one of its candidates resolves to
               `scada_state="unknown"`/`scada_transition=False` via
               `scada_majority_state`'s existing no-overlapping-window fallback --
               the SAME code path any out-of-SCADA-range span already took, no special
               case needed. The two 08.07.2026 sessions additionally drop any raw
               candidate within +/-30s of a known
               induced strike (`docs/groundtruth/080726_strikes_seconds_*.csv` UNION
               `080726_events_*.csv` -- the latter is needed too: e.g. the real PU
               event_id 07/landmark-A_kugelschieber minute has zero compiled
               per-strike rows yet, so the strikes CSV alone would miss it) BEFORE
               dedup/merge/cap, so an excluded window can never consume a top-25 slot
               or out-compete a genuine non-event candidate during overlap
               resolution. Each kept candidate's SCADA-derived operating mode is then
               attached (`load_session_scada`/`scada_majority_state`: the rule-based
               `rowii.scada.labels.gt_labels` state majority-voted over the
               candidate's own `[start_utc, end_utc)` span, plus whether that span
               touches a SCADA `"transition"` window) as `scada_state`/
               `scada_transition` -- independent of, and not necessarily equal to,
               the DETECTED `state_name` column (known clusterer-naming limitation:
               it reliably isolates turbine/standstill but can fold pump into a
               turbine-named cluster, and near-transition windows are unreliable) --
               any disagreement between the two is itself surfaced, never hidden
               (`scada_detector_mismatch`). `candidate_id` assignment is STABLE
               across re-`select` runs (`pin_stable_ids`): a candidate whose
               (session, start_utc) matches a row already published in the CURRENT
               `candidates.csv` keeps that exact id even if its rank shifts, so a
               reviewer's `localStorage` assessment (keyed by `candidate_id`, see
               `build` below) is never silently stranded by a re-`select`.
    build      For every candidate in candidates.csv: REUSE its assets untouched
               (`_reuse_existing_assets`) when a previous `candidates_meta.json`
               entry for the SAME `candidate_id` proves the span/extremity are
               unchanged and every file is still present -- extracting a WAV clip
               from a multi-gigabyte burst file is expensive, and re-rendering an
               unchanged candidate would only reproduce the identical bytes.
               Otherwise render assets exactly like `annotation_kit.py`
               (`extract_stream_clip`, `_compute_spectrogram_db`/
               `render_spectrogram_png`/`render_flat_spectrogram_png`, `make_demo_
               assets._resample_to_target`/`_write_clip_wav` -- all imported, never
               duplicated) for BOTH mic streams over an asset window (candidate span
               +/-10s, clamped to [20s, 60s] via `asset_window`), plus a self-contained
               interactive `results/candidate-kit/index.html`: a LEGEND box
               (`_CANDIDATE_LEGEND_HTML`) up front explains the two classes, the
               SCADA-vs-detector mode split, and the assessment vocabulary in plain
               reviewer language -- this kit is a HANDOVER artifact read by plant
               experts, not only by the author -- followed by per-session cards
               (sorted by extremity), each showing "Mode (SCADA)" prominently and
               "Mode (Detector)" secondarily (a warning chip when they disagree),
               the exact trigger criterion rendered as a human sentence
               (`criterion_sentence`), two independent playback lanes (generator/
               turbine mic, each its own flat-spectrogram-plus-canvas-playhead --
               click seeks, space plays/pauses, mirrors `annotation_kit._INTERACTIVE_
               JS`'s own click/keyboard/rAF-tick pattern), and an ASSESSMENT control
               (4-way radio + free-text note, autosaved to localStorage keyed by
               `candidate_id`, restored on reload). Per-session and all-sessions CSV
               export (columns: session, candidate_id, class, start_utc, duration_s,
               min_p, state_name, near_transition, scada_state, scada_transition,
               assessment, note).
    compile    Read an EXPORTED assessments CSV (index.html's "Export CSV" buttons),
               validate every row's `candidate_id` against the reference
               candidates.csv and every non-empty `assessment` against the fixed
               4-value vocabulary (raises BEFORE writing anything on the first bad
               row -- mirrors `annotation_kit.compile_template`'s all-validate-then-
               write contract), re-derive every descriptive column (class/start_utc/
               duration_s/min_p/state_name/near_transition/scada_state/
               scada_transition) from the TRUSTED candidates.csv rather than the
               export (only `assessment`/`note` are genuinely export-only fields --
               mirrors `annotation_kit.compile_marks`' own "strike_no/kind are for
               human readability only" precedent), skip unreviewed rows (empty
               assessment -- mirrors `compile_row`'s empty-offsets skip), and write
               the reviewed subset to `docs/assessments/candidate_assessments_
               <date>.csv` with a provenance header stating this is a manual
               listening review by the author, NOT ground truth, and NOT part of the
               expert handover deliverable -- the handover product is `build`'s own
               review tool (`index.html`) with every assessment left EMPTY for the
               plant experts themselves to fill in (`write_assessments_csv`'s header,
               handover-semantics clarification, Stefan's decision 2026-08-16).

Pure logic (episode grouping, transient extraction, dedup/overlap suppression,
strike-exclusion interval loading + overlap check, asset-window sizing, top-N
capping/id assignment, SCADA-majority-vote lookup, SCADA/detector mismatch,
criterion-sentence rendering, stable-id pinning, compile validation) is unit-tested
with synthetic fixtures in `tests/test_candidate_kit.py`. Real parquet reads against
`results/step2/once-calibrated/`, real Betriebsdaten reads (`load_session_scada`),
real WAV writes, spectrogram rendering, and `index.html` assembly are exercised by
actually running the CLI against real data instead, not by a test here (same "pure
vs IO-touching" split `annotation_kit.py`'s own module docstring documents).
"""
from __future__ import annotations

import argparse
import bisect
import dataclasses
import json
import logging
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
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
import make_demo_assets as mda  # noqa: E402
import run_step1 as rs1  # noqa: E402

from rowii.config import Config, load_config  # noqa: E402
from rowii.io.dataset import RecordingIndex, discover, run_utc_offset_ns  # noqa: E402
from rowii.pipeline import build_run_grid  # noqa: E402
from rowii.scada.labels import gt_labels, load_scada_window_means  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = _SCRIPTS_DIR.parent
DEFAULT_KIT_DIR = REPO_ROOT / "results" / "candidate-kit"
DEFAULT_CANDIDATES_CSV = DEFAULT_KIT_DIR / "candidates.csv"
DEFAULT_ASSESSMENTS_DIR = REPO_ROOT / "docs" / "assessments"

# ---------------------------------------------------------------------------
# Session scope + once-calibrated regime per session (2026-08-15 snapshot of the
# trigger-log decision -- `results/step2/once-calibrated/{fusion,audio-beats}_
# trigger_log.csv`'s own `decision` column, identical machine attribution on both
# representations; `run_once_calibrated.py` module docstring/D1). 080726-st_strikes
# carries no OWN trigger-log row (only 080726-pu_strikes does, era C) -- confirmed by
# reading `run_once_calibrated.py`'s pillar-3 section (`era_c_decision`, lines
# ~1034-1080): ST reuses the SAME day-level decision as PU, never computed
# separately. Hardcoded (not re-derived at runtime) mirrors `annotation_kit.
# _SESSION_CONFIG`'s own precedent; re-verify against the two `*_trigger_log.csv`
# files if `run_once_calibrated.py` is ever re-run with different data.
# ---------------------------------------------------------------------------

REGIME_BY_SESSION: dict[str, str] = {
    "250526-tu": "recalibrate",
    "250526-pu-morning": "recalibrate",
    "290626-tu": "frozen",
    "290626-pu": "recalibrate",
    "010726-tu_ph_tu": "frozen",
    "010726-pu": "frozen",
    "080726-pu_strikes": "recalibrate",
    "080726-st_strikes": "recalibrate",
    # --- Coverage extension (Stefan's decision 2026-08-16): the handover register
    # must show model-suspected anomalies from every recorded day, including the
    # three sessions `run_once_calibrated.py` never scored as a monitored day
    # (`_MONITOR_EXT_SESSIONS`'s docstring below has the full "why"). Their regime
    # is a DELIBERATE choice here, not read off a `*_trigger_log.csv` `decision`
    # column -- no such row exists for any of the three (270626 is that driver's
    # SENTINEL-ONLY entry, no Betriebsdaten -> no monitor.py call at all;
    # 010726-tu1-morning/010726-tu2 are `_B1_FIT_RUNS` commissioning-pool members,
    # never `_REPLAY` entries). "recalibrate" is `scripts/monitor.py`'s own default
    # mode and needs no era-level FAR judgement to justify it, unlike "frozen"
    # (package-2's cross-day evidence: frozen cross-day thresholds did NOT hold
    # their nominal FAR).
    "270626-pu_ph_pu_ph_pu_ph-1": "recalibrate",
    "010726-tu1-morning": "recalibrate",
    "010726-tu2": "recalibrate",
}
SESSIONS: tuple[str, ...] = tuple(REGIME_BY_SESSION)

_MONITOR_EXT_SESSIONS: frozenset[str] = frozenset(
    {"270626-pu_ph_pu_ph_pu_ph-1", "010726-tu1-morning", "010726-tu2"}
)
"""Sessions whose `alarms.parquet` lives under `results/monitor-ext/<session>/
<representation>/` (`scripts/monitor.py --snapshot models/adapted/monitor_pool_b1_
{fusion,audio_beats}_named.npz --thresholds recalibrate`, run directly against the
SAME once-calibrated snapshots) rather than under the pinned once-calibrated tree
(`results/step2/once-calibrated/<representation>/monitor/<session>/<regime>/`,
`run_once_calibrated.py`'s own `_REPLAY`/`_B1_FIT_RUNS` output) -- none of these
three was ever a MONITORED day in that driver (270626 is its sentinel-only row, no
Betriebsdaten at all; 010726-tu1-morning/010726-tu2 are commissioning-pool-only
`_B1_FIT_RUNS` members, never `_REPLAY` entries), so their recalibrate-mode alarms
were produced by this coverage extension instead, into a SEPARATE output tree that
never touches (or requires re-running) the existing once-calibrated trees. See
`_alarms_path_for`."""

IN_SAMPLE_SESSIONS: frozenset[str] = frozenset({"010726-tu1-morning", "010726-tu2"})
"""Sessions whose candidates carry the `in_sample=True` caveat (`Candidate.
in_sample`): both are literal members of the fusion/audio-beats scoring snapshot's
OWN fit pool (`models/adapted/monitor_pool_b1_{fusion,audio_beats}_named.npz`'s
`fit_run` = `"pool:010726-tu_ph_tu,010726-pu,010726-tu1-morning,010726-tu2"`), so a
candidate flagged here comes from data the detector was itself fit on -- a genuinely
different evidentiary status than a held-out day. `270626-pu_ph_pu_ph_pu_ph-1` is
NOT in this set: it has no SCADA either, but it is not part of the snapshot's fit
pool, so it stays held-out."""

_STRIKE_EXCLUSION_FILES: dict[str, tuple[Path, Path]] = {
    "080726-pu_strikes": (
        REPO_ROOT / "docs" / "groundtruth" / "080726_strikes_seconds_pu.csv",
        mda.PU_EVENTS_CSV,
    ),
    "080726-st_strikes": (
        REPO_ROOT / "docs" / "groundtruth" / "080726_strikes_seconds_st.csv",
        mda.ST_EVENTS_CSV,
    ),
}

FUSION_SUSTAINED_ALPHA = 0.01
FUSION_SUSTAINED_MIN_DURATION_S = 3.0
AUDIOBEATS_TRANSIENT_ALPHA = 0.001
TRANSIENT_DEDUP_RADIUS_S = 5.0
STRIKE_EXCLUSION_PAD_S = 30.0
CANDIDATE_CAP_PER_SESSION = 25

ASSET_PAD_S = 10.0
ASSET_MIN_DURATION_S = 20.0
ASSET_MAX_DURATION_S = 60.0

ASSESSMENT_VALUES: tuple[str, ...] = (
    "plausible anomaly", "operational/explained", "artifact/sensor", "no finding",
    "unclear",
)

_ALARMS_COLUMNS = ["window", "t_utc_ns", "p_value", "role", "near_transition", "state_name"]


# ---------------------------------------------------------------------------
# Candidate + pure grouping/extraction logic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One candidate window, either a SUSTAINED fusion episode or a single TRANSIENT
    audio-beats window. `candidate_id`/`rank` start empty/0 and are filled in by
    `assign_ids` once the session's final (post-cap) extremity order is known --
    every candidate-building/filtering step before that stays agnostic of id/rank."""

    session: str
    klass: str
    """`"sustained"` or `"transient"`."""
    start_utc: datetime
    end_utc: datetime
    duration_s: float
    min_p: float
    """The episode's (or, for a transient, the single window's) lowest p-value --
    the extremity key used everywhere: dedup, overlap suppression, cap, ranking."""
    state_name: str
    near_transition: bool
    n_windows: int
    modality: str
    """`"fusion"` or `"audio-beats"` -- which monitor produced this candidate."""
    regime: str
    """`"frozen"` or `"recalibrate"` -- the once-calibrated threshold arm actually
    used for this session (`REGIME_BY_SESSION`)."""
    alarms_path: str
    """Provenance: the `alarms.parquet` path this candidate was read from, relative
    to `Config.results_root` when possible."""
    candidate_id: str = ""
    rank: int = 0
    scada_state: str = ""
    """SCADA-derived operating mode, majority-voted (`scada_majority_state`) over
    THIS candidate's own `[start_utc, end_utc)` span from `rowii.scada.labels.
    gt_labels`'s rule-based state series -- filled in by `select_session` after id
    assignment, empty for a `Candidate` built directly by `build_sustained_episodes`/
    `build_transient_candidates` (mirrors `candidate_id`/`rank`'s own
    fill-in-later pattern). This is the TRUSTED operating mode: `state_name` above
    is only what the unsupervised detector's cluster naming guessed, which can be
    wrong (module docstring's known clusterer-naming limitation)."""
    scada_transition: bool = False
    """Whether ANY SCADA window overlapping this candidate's span carries GT state
    `"transition"` (`rowii.scada.labels.gt_labels`'s unified ramp-rate/transition-
    buffer state) -- the SCADA-side analogue of `near_transition` above (which is
    the DETECTOR's own signal). See `scada_majority_state`."""
    in_sample: bool = False
    """`True` iff this candidate's session is a member of the scoring snapshot's OWN
    fit pool (`IN_SAMPLE_SESSIONS`) -- filled in by `select_session` alongside
    `scada_state`/`scada_transition`, `False` for every session that is genuinely
    held-out. A reviewer-relevant caveat distinct from `scada_state`: a candidate can
    be in-sample regardless of its SCADA coverage (or lack of it)."""


def _utc_from_ns(t_utc_ns: int) -> datetime:
    """`datetime.fromtimestamp(ns/1e9, tz=UTC)` -- same idiom `annotation_kit.
    extract_stream_clip` already uses for the identical `int64`-nanosecond ->
    timezone-aware-`datetime` conversion."""
    return datetime.fromtimestamp(t_utc_ns / 1e9, tz=UTC)


def build_sustained_episodes(
    df: pd.DataFrame,
    *,
    window_s: float,
    alpha: float,
    min_duration_s: float,
    session: str,
    modality: str,
    regime: str,
    alarms_path: str,
) -> list[Candidate]:
    """Group `role=="scored"` rows with `p_value < alpha` into maximal runs of
    STRICTLY contiguous `window` indices (`window[i+1] == window[i] + 1`) -- a
    skipped/invalid grid window (real gaps of 2 occur in `alarms.parquet`, one per
    invalid window never scored) never bridges two episodes into one, matching the
    conservative reading of "consecutive". `duration_s` covers the full span each
    window represents: `(t_utc[last] - t_utc[first]) + window_s`, so e.g. 3
    consecutive 1.0s windows yield exactly a 3.0s episode. Episodes shorter than
    `min_duration_s` are dropped. `state_name`/`near_transition` are NOT simply the
    first row's: `state_name` comes from the row with the LOWEST p_value (the
    episode's own extremum -- the state active exactly when the anomaly was most
    extreme, well-defined even if a transition occurs mid-episode), and
    `near_transition` is `True` if ANY row in the episode carries it (the more
    informative, conservative reading for a reviewer -- "a transition might explain
    part of this").

    `role != "scored"` rows are excluded explicitly (not merely relied upon via
    `p_value` being `NaN` for `consumed_for_calibration`/`unknown_state` rows,
    though that is also true) -- alarming a calibration-side window would be
    circular (it helped fit the very threshold that would flag it).
    """
    mask = (df["role"] == "scored") & (df["p_value"] < alpha)
    sub = df.loc[mask].sort_values("window").reset_index(drop=True)
    if sub.empty:
        return []

    window = sub["window"].to_numpy()
    breaks = np.flatnonzero(np.diff(window) != 1) + 1
    groups = np.split(np.arange(len(sub)), breaks)

    out: list[Candidate] = []
    for group_idx in groups:
        rows = sub.iloc[group_idx]
        start_utc = _utc_from_ns(int(rows["t_utc_ns"].iloc[0]))
        end_utc = _utc_from_ns(int(rows["t_utc_ns"].iloc[-1])) + timedelta(seconds=window_s)
        duration_s = (end_utc - start_utc).total_seconds()
        if duration_s < min_duration_s:
            continue
        extremum = rows.loc[rows["p_value"].idxmin()]
        out.append(
            Candidate(
                session=session, klass="sustained", start_utc=start_utc, end_utc=end_utc,
                duration_s=duration_s, min_p=float(rows["p_value"].min()),
                state_name=str(extremum["state_name"]),
                near_transition=bool(rows["near_transition"].any()),
                n_windows=len(rows), modality=modality, regime=regime, alarms_path=alarms_path,
            )
        )
    return out


def build_transient_candidates(
    df: pd.DataFrame,
    *,
    window_s: float,
    alpha: float,
    session: str,
    modality: str,
    regime: str,
    alarms_path: str,
) -> list[Candidate]:
    """One `Candidate` per `role=="scored"` row with `p_value < alpha` -- no
    grouping (each window is its own candidate; `dedupe_by_radius` handles
    near-duplicate suppression separately, since transient windows are compared
    against a FIXED time radius rather than pure index-adjacency)."""
    mask = (df["role"] == "scored") & (df["p_value"] < alpha)
    sub = df.loc[mask]

    out: list[Candidate] = []
    for row in sub.itertuples(index=False):
        start_utc = _utc_from_ns(int(row.t_utc_ns))
        out.append(
            Candidate(
                session=session, klass="transient", start_utc=start_utc,
                end_utc=start_utc + timedelta(seconds=window_s), duration_s=window_s,
                min_p=float(row.p_value), state_name=str(row.state_name),
                near_transition=bool(row.near_transition), n_windows=1,
                modality=modality, regime=regime, alarms_path=alarms_path,
            )
        )
    return out


def _spans_overlap(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    """Half-open interval overlap (codebase-wide convention, e.g. `make_demo_assets.
    has_collision`): touching exactly at a boundary is NOT an overlap."""
    return a_start < b_end and a_end > b_start


def _greedy_suppress(
    spans: Sequence[tuple[datetime, datetime]], keys: Sequence[tuple[float, datetime]]
) -> list[int]:
    """Indices to keep: visit *spans* in ascending *keys* order (most extreme
    first), accept an index unless it overlaps (`_spans_overlap`) a span already
    accepted. The one greedy interval-suppression primitive behind both
    `suppress_overlaps` (natural spans) and `dedupe_by_radius` (spans pre-widened by
    the dedup radius) -- "keep the more extreme of two colliding candidates" is the
    same rule either way, only how the span is computed differs."""
    order = sorted(range(len(spans)), key=lambda i: keys[i])
    kept: list[int] = []
    for i in order:
        a_start, a_end = spans[i]
        if not any(_spans_overlap(a_start, a_end, spans[j][0], spans[j][1]) for j in kept):
            kept.append(i)
    return kept


def suppress_overlaps(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Drop every candidate whose `[start_utc, end_utc)` span overlaps a MORE
    extreme (lower `min_p`) already-kept candidate's span -- "merge, drop
    candidates overlapping each other, keep the more extreme" applied across BOTH
    classes at once (a transient inside a sustained episode's span, or vice versa,
    is exactly the case this resolves). Returns the kept candidates in their
    original relative input order (callers needing extremity order re-sort; several
    do, e.g. `cap_top_n`/`assign_ids`)."""
    spans = [(c.start_utc, c.end_utc) for c in candidates]
    keys = [(c.min_p, c.start_utc) for c in candidates]
    keep = set(_greedy_suppress(spans, keys))
    return [c for i, c in enumerate(candidates) if i in keep]


def dedupe_by_radius(candidates: Sequence[Candidate], radius_s: float) -> list[Candidate]:
    """Drop every candidate whose START TIME is within *radius_s* seconds of a MORE
    extreme already-kept candidate's start time (either direction) -- transient
    candidates are near-instantaneous (single `window_s`-long windows), so this
    compares alarm TIMES, not spans: each candidate's start is widened to
    `[start - radius_s/2, start + radius_s/2)` (half the radius on each side, NOT
    the full radius -- two spans widened by the FULL radius on each side would
    collide at up to `2*radius_s` apart, not `radius_s`) and the shared
    `_greedy_suppress`/`_spans_overlap` primitive is reused: two such half-open
    spans overlap iff `abs(start_a - start_b) < radius_s` exactly, so a gap of
    exactly *radius_s* is NOT a collision (half-open convention, matches
    `suppress_overlaps`). Output candidates keep their ORIGINAL (unwidened)
    start/end."""
    half = timedelta(seconds=radius_s / 2)
    widened_spans = [(c.start_utc - half, c.start_utc + half) for c in candidates]
    keys = [(c.min_p, c.start_utc) for c in candidates]
    keep = set(_greedy_suppress(widened_spans, keys))
    return [c for i, c in enumerate(candidates) if i in keep]


def overlaps_any(
    start: datetime, end: datetime, intervals: Sequence[tuple[datetime, datetime]]
) -> bool:
    """`True` iff `[start, end)` overlaps (half-open, `_spans_overlap`) any interval
    in *intervals* -- the strike-exclusion mask's own membership test."""
    return any(_spans_overlap(start, end, s, e) for s, e in intervals)


def load_strike_exclusion_intervals(
    strikes_csv: Path, events_csv: Path, *, pad_s: float
) -> list[tuple[datetime, datetime]]:
    """Every `[strike_utc - pad_s, strike_utc + pad_s]` interval from *strikes_csv*
    (a compiled `080726_strikes_seconds_<session>.csv`, `annotation_kit.
    COMPILED_CSV_COLUMNS` shape) UNION every `[event.start_utc - pad_s, event.end_utc
    + pad_s]` interval from *events_csv* (`make_demo_assets._load_events_csv`'s
    minute-level shape) -- the events file is not redundant padding: on real data
    `080726_strikes_seconds_pu.csv` has ZERO compiled rows for event_id 07
    (landmark-A_kugelschieber, 12:49-12:50) and 13 (landmark-C_EG, 13:01-13:02), so
    the strikes file alone would leave those two known-induced minutes unmasked.

    `format="ISO8601"` is required (not the plain `pd.to_datetime` default): real
    `strike_utc` values mix microsecond-precision (`...58.212000+00:00`, the vast
    majority) and whole-second (`...12:50:58+00:00`, exactly one row each in the
    real PU/ST files -- a strike landing exactly on a whole second) formatting
    WITHIN the same column; pandas' single-format auto-inference locks onto
    whichever format the first rows show and then raises on the other -- ISO8601
    mode parses each value on its own.
    """
    pad = timedelta(seconds=pad_s)
    intervals: list[tuple[datetime, datetime]] = []

    strikes = pd.read_csv(strikes_csv, comment="#")
    strike_times = pd.to_datetime(strikes["strike_utc"], utc=True, format="ISO8601")
    for raw in strike_times:
        t = raw.to_pydatetime()
        intervals.append((t - pad, t + pad))

    events = mda._load_events_csv(events_csv)
    for row in events.itertuples(index=False):
        intervals.append((row.start_utc.to_pydatetime() - pad, row.end_utc.to_pydatetime() + pad))

    return intervals


def asset_window(
    start_utc: datetime, end_utc: datetime, *, pad_s: float, min_s: float, max_s: float
) -> tuple[datetime, datetime]:
    """`[start_utc - pad_s, end_utc + pad_s]`, then symmetrically expanded to
    *min_s* (around the padded window's own midpoint) if still shorter, or
    symmetrically clipped to *max_s* if longer -- "window = candidate span +/-10s
    (min 20s, max 60s)". Mirrors `make_demo_assets.center_window`'s own
    midpoint-preserving resize pattern."""
    pad = timedelta(seconds=pad_s)
    padded_start, padded_end = start_utc - pad, end_utc + pad
    duration_s = (padded_end - padded_start).total_seconds()
    if duration_s >= min_s and duration_s <= max_s:
        return padded_start, padded_end

    mid = padded_start + (padded_end - padded_start) / 2
    target_s = min_s if duration_s < min_s else max_s
    half = timedelta(seconds=target_s / 2)
    return mid - half, mid + half


def cap_top_n(candidates: Sequence[Candidate], n: int) -> list[Candidate]:
    """The *n* most extreme (lowest `min_p`) candidates, ties broken by earliest
    `start_utc`."""
    ordered = sorted(candidates, key=lambda c: (c.min_p, c.start_utc))
    return ordered[:n]


def assign_ids(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Assign `candidate_id` (`"<session>-<rank:02d>"`) and `rank` (1 = most
    extreme) over *candidates* re-sorted by extremity -- independent of whatever
    order the caller passed in, so this step is safe to call on any candidate list.
    Stable only within one `select` run over unchanged input data; a re-`select` on
    the same alarms.parquet files reproduces the identical ids deterministically.
    See `pin_stable_ids` for the cross-run stability guarantee `select_session`
    layers on top of this."""
    ordered = sorted(candidates, key=lambda c: (c.min_p, c.start_utc))
    return [
        dataclasses.replace(c, candidate_id=f"{c.session}-{rank:02d}", rank=rank)
        for rank, c in enumerate(ordered, start=1)
    ]


# ---------------------------------------------------------------------------
# Stable candidate_id pinning across re-`select` runs
# ---------------------------------------------------------------------------


def load_prior_candidate_ids(path: Path) -> dict[tuple[str, str], str]:
    """`{(session, start_utc ISO string): candidate_id}` from an existing
    `candidates.csv`, or `{}` if *path* does not exist yet (the very first
    `select` run -- nothing published yet to pin against). The key is a real-world
    candidate window's own natural identity (module docstring's `select` section):
    the SAME (session, start_utc) pair re-appearing across two `select` runs is,
    by construction, the SAME underlying alarms.parquet window(s), whatever rank
    it lands on this time -- see `pin_stable_ids`.
    """
    if not path.is_file():
        return {}
    df = read_candidates_csv(path)
    return {
        (str(row["session"]), str(row["start_utc"])): str(row["candidate_id"])
        for row in df.to_dict(orient="records")
    }


def pin_stable_ids(
    candidates: Sequence[Candidate], prior_ids: Mapping[tuple[str, str], str]
) -> list[Candidate]:
    """Re-key one session's *candidates* (already `assign_ids`-ranked) so a
    candidate whose (session, start_utc) matches *prior_ids* keeps that EXACT
    earlier `candidate_id`, instead of the fresh rank-derived one `assign_ids`
    gave it -- `index.html`'s reviewer assessments live in `localStorage` keyed by
    `candidate_id` (module docstring), so an id drifting across a re-`select`
    would silently strand already-saved review work even though nothing about the
    actual review changed. `rank` is always left as the CURRENT run's own
    extremity order, independent of id pinning -- it is a display-sort field, not
    an identity.

    A genuinely NEW candidate (no (session, start_utc) match) gets the next free
    `<session>-NN` slot -- but "free" means never used ANYWHERE in *prior_ids* for
    this session, not merely not currently taken: reusing a slot that a now-
    vanished candidate used to own would let a brand new, unrelated candidate
    silently "inherit" whatever assessment a reviewer had saved against that old
    slot in `localStorage`. In the common case -- no prior file yet, or an
    unchanged candidate set -- every id `prior_ids` could possibly supply is
    already exactly the one `assign_ids` would give anyway, so this is a no-op.
    """
    if not candidates:
        return []
    session = candidates[0].session
    ever_used_this_session = {v for (s, _start), v in prior_ids.items() if s == session}

    pinned_id_by_index: dict[int, str] = {}
    for i, c in enumerate(candidates):
        prior_id = prior_ids.get((c.session, c.start_utc.isoformat()))
        if prior_id is not None:
            pinned_id_by_index[i] = prior_id

    taken = set(pinned_id_by_index.values()) | ever_used_this_session
    out: list[Candidate] = []
    next_n = 1
    for i, c in enumerate(candidates):
        if i in pinned_id_by_index:
            out.append(dataclasses.replace(c, candidate_id=pinned_id_by_index[i]))
            continue
        while f"{session}-{next_n:02d}" in taken:
            next_n += 1
        candidate_id = f"{session}-{next_n:02d}"
        taken.add(candidate_id)
        next_n += 1
        out.append(dataclasses.replace(c, candidate_id=candidate_id))
    return out


# ---------------------------------------------------------------------------
# SCADA operating-mode lookup (majority vote over a candidate's own span)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionScada:
    """One session's full-run SCADA/GT window series: `window_start_utc[i]` is the
    start of the window whose rule-based state is `state[i]` (same length, both
    ascending in time) -- `scada_majority_state`'s own input shape. Produced by
    `load_session_scada`.

    `power_mw`/`speed_rpm`/`flow_net_m3s` (SCADA CONTEXT PANEL, Stefan's decision
    2026-08-16) are the SAME per-window means `load_session_scada` already computes
    (`rowii.scada.labels.load_scada_window_means`'s own `"power"`/`"speed"` columns,
    plus `"flow_tu"`/`"flow_pu"` combined into one SIGNED net-flow series --
    `flow_tu - flow_pu`, positive during turbine operation, negative during pump
    operation, mirroring `speed`'s own turbine-positive/pump-negative sign
    convention; NaN only where BOTH source channels are NaN, never where just one
    is, since the inactive path's true value is ~0 m3/s, not unknown) -- carried
    alongside `state` rather than recomputed, so `build`'s SCADA strip/live-readout
    never re-reads Betriebsdaten. `has_scada` is `False` only for a session with NO
    Betriebsdaten files at all for its own day (`270626-pu_ph_pu_ph_pu_ph-1`, the
    ONLY such session) -- every list is then empty and every candidate's
    `scada_state` already resolves to `"unknown"` via `scada_majority_state`'s own
    no-overlap fallback (module docstring); `has_scada` exists so `build` can
    render the neutral "no SCADA available for this day" strip placeholder
    explicitly, instead of silently plotting three empty (all-NaN) mini-axes."""

    window_start_utc: list[datetime]
    window_s: float
    state: list[str]
    has_scada: bool
    power_mw: list[float]
    speed_rpm: list[float]
    flow_net_m3s: list[float]


def scada_majority_state(
    window_start_utc: Sequence[datetime],
    window_s: float,
    state: Sequence[str],
    start_utc: datetime,
    end_utc: datetime,
) -> tuple[str, bool]:
    """The majority SCADA `state` (`rowii.scada.labels.gt_labels`'s own state
    vocabulary: standstill/turbine/pump/phase-shifter/transition/unknown) among
    every window `i` whose `[window_start_utc[i], window_start_utc[i] + window_s)`
    overlaps `[start_utc, end_utc)` (half-open, `_spans_overlap`'s own convention),
    plus whether ANY overlapping window's state is `"transition"`. Ties in the
    majority vote are broken by whichever tied state occurs EARLIEST in
    *window_start_utc* (`collections.Counter.most_common()` is a stable sort over
    each key's first-insertion order, and *state* is walked in *window_start_utc*'s
    own ascending order here -- no separate tie-break code needed). Returns
    `("unknown", False)` if no window overlaps the span at all (e.g. a candidate
    span outside SCADA's own recorded range).

    *window_start_utc*/*state* must be the same length and already sorted
    ascending by *window_start_utc* -- `load_session_scada`'s own contract
    (`WindowGrid.edges_ns()` is monotonic by construction).
    """
    window_span = timedelta(seconds=window_s)
    overlapping = [
        s
        for ws, s in zip(window_start_utc, state, strict=True)
        if _spans_overlap(ws, ws + window_span, start_utc, end_utc)
    ]
    if not overlapping:
        return "unknown", False
    counts: Counter[str] = Counter(overlapping)
    majority = counts.most_common(1)[0][0]
    touches_transition = "transition" in counts
    return majority, touches_transition


def scada_detector_mismatch(scada_state: str, detected_state_name: str) -> bool:
    """Whether the SCADA-derived operating mode and the detector's OWN
    `state_name` disagree -- reviewer-relevant per se (module docstring: the
    clusterer reliably isolates turbine/standstill but can fold pump into a
    turbine-named cluster). A straight inequality, except `scada_state ==
    "unknown"` (no SCADA coverage for this span -- nothing to compare the
    detector against) never counts as a mismatch.
    """
    if scada_state == "unknown":
        return False
    return scada_state != detected_state_name


def criterion_sentence(candidate: Candidate) -> str:
    """Human-readable English sentence stating exactly which selection criterion
    (module docstring's `select` section) *candidate* satisfied -- built from
    already-known fields (`klass`/`n_windows`/`duration_s`/`min_p`) and the SAME
    threshold constants `select_session` itself filters on, so the sentence can
    never drift out of sync with the actual selection rule.

    Raises:
        ValueError: if `candidate.klass` is neither `"sustained"` nor
            `"transient"` (the only two classes `select` ever produces).
    """
    if candidate.klass == "sustained":
        return (
            f"Triggered by a sustained anomaly (fusion path): "
            f"{candidate.n_windows} consecutive windows with "
            f"p < {FUSION_SUSTAINED_ALPHA:g} against the operating mode's normal "
            f"model, {candidate.duration_s:.1f} s total (minimum duration "
            f"{FUSION_SUSTAINED_MIN_DURATION_S:g} s) -- lowest value "
            f"p = {candidate.min_p:.3e}."
        )
    if candidate.klass == "transient":
        return (
            f"Triggered by a single short impulse (BEATs path): one "
            f"window with p < {AUDIOBEATS_TRANSIENT_ALPHA:g} against the normal "
            f"model (p = {candidate.min_p:.3e})."
        )
    raise ValueError(f"unknown candidate class {candidate.klass!r}")


def load_session_scada(index: RecordingIndex, session: str, cfg: Config) -> SessionScada:
    """Real SCADA read + `rowii.scada.labels.gt_labels` rule evaluation for
    *session*'s FULL run span -- never truncated to one candidate's own window.
    `gt_labels`'s phase-shifter promotion/ramp/transition-buffer rules all look
    OUTSIDE a single window for context (a 600s dwell run, a +/-1-window ramp
    derivative, +/-`transition_buffer_s` around a state change) -- computing GT
    over anything narrower than the whole run risks truncating that context and
    mis-labeling a window near the truncation edge. Meant to be called ONCE per
    session (`select_all`) and reused across every one of that session's own
    candidates (`scada_majority_state`).

    The grid is built from the AUDIO streams only (`run_step1._AUDIO_STREAMS`),
    deliberately the WIDEST (largest-time-range) intersection any once-calibrated
    monitor's own grid could have used: `_streams_for_variant("audio-beats")` IS
    this exact 2-stream grid, and `"fusion"`'s is the SAME 2 streams additionally
    intersected with both vibration streams -- a SUBSET of this time range. So
    this grid is guaranteed to cover every candidate this kit ever selects,
    regardless of which monitor (fusion or audio-beats) produced it.

    Deliberately calls `load_scada_window_means`/`gt_labels` directly rather than
    reusing `run_step1.load_run_gt` -- that helper's own last step overwrites
    every window outside an audio-FEATURE validity mask to `"unknown"`; that mask
    reflects audio/vibration recording coverage, which has nothing to do with what
    SCADA itself knows about the plant's operating mode at that instant, and this
    kit specifically wants the undiluted SCADA-only answer.
    """
    run = mda._get_run(index, session)
    betriebsdaten = index.betriebsdaten_by_day.get(run.day_root, [])
    offset_ns = run_utc_offset_ns(run)
    grid = build_run_grid(run, rs1._AUDIO_STREAMS, cfg.window.window_s, offset_ns=offset_ns)
    matched_files = rs1._betriebsdaten_for_grid(betriebsdaten, grid)
    scada = load_scada_window_means(matched_files, grid, audio_run_offset_ns=offset_ns)
    gt = gt_labels(scada, cfg.gt, window_s=cfg.window.window_s)

    window_start_utc = [_utc_from_ns(int(t)) for t in grid.edges_ns()[:-1]]
    flow_tu = scada["flow_tu"].to_numpy(dtype=np.float64)
    flow_pu = scada["flow_pu"].to_numpy(dtype=np.float64)
    both_flows_nan = np.isnan(flow_tu) & np.isnan(flow_pu)
    flow_net = np.where(
        both_flows_nan, np.nan, np.nan_to_num(flow_tu) - np.nan_to_num(flow_pu)
    )
    return SessionScada(
        window_start_utc=window_start_utc, window_s=cfg.window.window_s,
        state=[str(s) for s in gt["state"].tolist()],
        has_scada=bool(matched_files),
        power_mw=scada["power"].tolist(),
        speed_rpm=scada["speed"].tolist(),
        flow_net_m3s=flow_net.tolist(),
    )


# ---------------------------------------------------------------------------
# SCADA context panel (build, Stefan's decision 2026-08-16): fixed state colors,
# window-series slicing, 1 Hz downsampling for the strip/ribbon/live-readout.
# ---------------------------------------------------------------------------

_STATE_COLORS: dict[str, str] = {
    "standstill": "#4b5563",
    "turbine": "#0072b2",
    "pump": "#d55e00",
    "phase-shifter": "#cc79a7",
    "transition": "#f0e442",
    "unknown": "#cbd5e1",
}
"""Fixed color per state name -- the closed `rowii.scada.labels.STATES` vocabulary
(standstill/turbine/pump/phase-shifter/transition) plus `scada_majority_state`'s
own `"unknown"` fallback (module docstring: "unknown = grey"). A colorblind-safe
qualitative palette (Okabe & Ito, 2008), deliberately DISTINCT from `_CANDIDATE_CSS`'s
own `--sustained`/`--transient` class-badge colors (amber/purple) so the ribbon's
state colors are never confused with the class badge above it. ONE shared mapping
for BOTH ribbon rows (SCADA state and detected state) -- a detector `state_name`
that carries a named mapping (`rowii.eval.metrics.derive_state_names`) uses the
SAME vocabulary and therefore the SAME color as the SCADA row."""

_STATE_OTHER_COLOR = "#111827"
"""Fallback swatch for any state name outside `_STATE_COLORS` -- in practice only
ever a detector `cluster-<id>` fallback name (`derive_state_names`'s own
un-named-cluster case; the SCADA vocabulary above is closed and never produces
one). ONE fixed color for every such name, so the ribbon/legend never needs a new
entry per un-named cluster id."""

STATE_LEGEND: tuple[tuple[str, str], ...] = (
    ("standstill", _STATE_COLORS["standstill"]),
    ("turbine", _STATE_COLORS["turbine"]),
    ("pump", _STATE_COLORS["pump"]),
    ("phase-shifter", _STATE_COLORS["phase-shifter"]),
    ("transition", _STATE_COLORS["transition"]),
    ("unknown", _STATE_COLORS["unknown"]),
    ("other (unnamed cluster)", _STATE_OTHER_COLOR),
)
"""`(English name, hex color)` pairs in display order -- `render_index_html`'s
single, page-top state-color legend (module docstring build/2: "a small legend
once at the page top")."""


def state_color(state_name: str) -> str:
    """Fixed hex color for *state_name* -- `_STATE_COLORS` for the closed SCADA
    vocabulary (also covers every NAMED detected state, since a named detector
    state reuses that same vocabulary), else `_STATE_OTHER_COLOR` (an un-named
    `cluster-<id>` detector fallback)."""
    return _STATE_COLORS.get(state_name, _STATE_OTHER_COLOR)


def slice_window_series[T](
    window_start_utc: Sequence[datetime],
    window_s: float,
    values: Sequence[T],
    start_utc: datetime,
    end_utc: datetime,
) -> tuple[list[datetime], list[T]]:
    """The subsequence of `(window_start_utc[i], values[i])` pairs whose own window
    `[window_start_utc[i], window_start_utc[i] + window_s)` overlaps `[start_utc,
    end_utc)` (half-open, `_spans_overlap`'s own convention) -- a `bisect`
    pre-filter (*window_start_utc* must be sorted ascending, `load_session_scada`'s
    own contract) so `resample_channel_to_seconds`/`resample_states_to_seconds`
    below never re-scan an entire session's window series (tens of thousands of
    rows) for one candidate's 20-60 s asset window: O(log n) to locate the slice
    bounds, regardless of how large *window_start_utc* is.
    """
    span = timedelta(seconds=window_s)
    lo = bisect.bisect_right(window_start_utc, start_utc - span)
    hi = max(bisect.bisect_left(window_start_utc, end_utc), lo)
    return list(window_start_utc[lo:hi]), list(values[lo:hi])


def resample_channel_to_seconds(
    window_start_utc: Sequence[datetime],
    window_s: float,
    values: Sequence[float],
    asset_start_utc: datetime,
    n_seconds: int,
) -> list[float | None]:
    """The LIVE READOUT's own 1 Hz series (module docstring build/3): mean of
    *values* (NaN ignored) over each whole second `[asset_start_utc + i,
    asset_start_utc + i + 1)` for `i` in `range(n_seconds)`. A second with no
    overlapping window, or where every overlapping value is NaN, resamples to
    `None` (JSON `null`) rather than a fabricated number -- the JS readout blanks
    out for that second instead of showing stale/wrong data. Slices first
    (`slice_window_series`) so this never re-scans a whole session's series.
    """
    asset_end_utc = asset_start_utc + timedelta(seconds=n_seconds)
    starts, vals = slice_window_series(
        window_start_utc, window_s, values, asset_start_utc, asset_end_utc
    )
    span = timedelta(seconds=window_s)
    out: list[float | None] = []
    for i in range(n_seconds):
        sec_start = asset_start_utc + timedelta(seconds=i)
        sec_end = sec_start + timedelta(seconds=1)
        overlapping = [
            float(v)
            for ws, v in zip(starts, vals, strict=True)
            if _spans_overlap(ws, ws + span, sec_start, sec_end)
        ]
        finite = [v for v in overlapping if not math.isnan(v)]
        out.append(sum(finite) / len(finite) if finite else None)
    return out


def resample_states_to_seconds(
    window_start_utc: Sequence[datetime],
    window_s: float,
    states: Sequence[str],
    asset_start_utc: datetime,
    n_seconds: int,
) -> list[str]:
    """The STATE RIBBON's own per-second row (module docstring build/2): the
    majority state (`scada_majority_state`, reused unchanged -- works identically
    for SCADA states or detector `state_name` values, since both are just a state
    label per window) over each whole second `[asset_start_utc + i, asset_start_utc
    + i + 1)`. `"unknown"` (`scada_majority_state`'s own no-overlap fallback, the
    ribbon's grey cell) for a second with no overlapping window. Slices first
    (`slice_window_series`), same rationale as `resample_channel_to_seconds`.
    """
    asset_end_utc = asset_start_utc + timedelta(seconds=n_seconds)
    starts, vals = slice_window_series(
        window_start_utc, window_s, states, asset_start_utc, asset_end_utc
    )
    out: list[str] = []
    for i in range(n_seconds):
        sec_start = asset_start_utc + timedelta(seconds=i)
        sec_end = sec_start + timedelta(seconds=1)
        state, _touches = scada_majority_state(starts, window_s, vals, sec_start, sec_end)
        out.append(state)
    return out


def build_readout_series(
    session_scada: SessionScada, asset_start_utc: datetime, n_seconds: int
) -> tuple[list[float | None], list[float | None]]:
    """The LIVE READOUT's own `(power_mw, speed_rpm)` 1 Hz pair for one candidate's
    asset window (module docstring build/3) -- `resample_channel_to_seconds` over
    *session_scada*'s own `power_mw`/`speed_rpm` channels, EXCEPT when
    *session_scada.has_scada* is False (the 270626-pu_ph_pu_ph_pu_ph-1
    no-Betriebsdaten placeholder path): both series are then `[None] * n_seconds`
    without attempting to resample anything (there is nothing to resample --
    `session_scada.power_mw`/`speed_rpm` are empty for that session), so the UI's
    readout blanks out rather than showing a misleading all-NaN-derived value.
    """
    if not session_scada.has_scada:
        return [None] * n_seconds, [None] * n_seconds
    power = resample_channel_to_seconds(
        session_scada.window_start_utc, session_scada.window_s, session_scada.power_mw,
        asset_start_utc, n_seconds,
    )
    speed = resample_channel_to_seconds(
        session_scada.window_start_utc, session_scada.window_s, session_scada.speed_rpm,
        asset_start_utc, n_seconds,
    )
    return power, speed


# ---------------------------------------------------------------------------
# select: per-session pipeline + candidates.csv
# ---------------------------------------------------------------------------

CANDIDATES_CSV_COLUMNS = (
    "session", "candidate_id", "rank", "class", "start_utc", "end_utc", "duration_s",
    "min_p", "state_name", "near_transition", "scada_state", "scada_transition",
    "n_windows", "modality", "regime", "alarms_path", "in_sample",
)


@dataclass(frozen=True)
class SelectResult:
    session: str
    regime: str
    n_sustained_raw: int
    n_transient_raw: int
    n_excluded_strike: int
    n_after_merge: int
    kept: list[Candidate]


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _read_alarms(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"alarms.parquet not found: {path}")
    return pd.read_parquet(path, columns=_ALARMS_COLUMNS)


def _once_calibrated_alarms_path(
    results_root: Path, representation: str, session: str, regime: str
) -> Path:
    return (
        results_root / "step2" / "once-calibrated" / representation / "monitor"
        / session / regime / "alarms.parquet"
    )


def _monitor_ext_alarms_path(results_root: Path, representation: str, session: str) -> Path:
    """`results/monitor-ext/<session>/<representation>/alarms.parquet` --
    `_MONITOR_EXT_SESSIONS`'s own output layout (coverage extension, Stefan's
    decision 2026-08-16): a single `--thresholds recalibrate` monitor.py pass per
    (session, representation), so unlike `_once_calibrated_alarms_path` there is no
    `<regime>/` path segment to select between."""
    return results_root / "monitor-ext" / session / representation / "alarms.parquet"


def _alarms_path_for(results_root: Path, representation: str, session: str, regime: str) -> Path:
    """The `alarms.parquet` path for *session*/*representation*: `_monitor_ext_
    alarms_path` for a `_MONITOR_EXT_SESSIONS` member, else the unchanged
    `_once_calibrated_alarms_path` -- the ONLY call site that needs to know the two
    trees exist; every other session keeps reading exactly the path it always has."""
    if session in _MONITOR_EXT_SESSIONS:
        return _monitor_ext_alarms_path(results_root, representation, session)
    return _once_calibrated_alarms_path(results_root, representation, session, regime)


def _attach_scada(candidate: Candidate, session_scada: SessionScada) -> Candidate:
    """`candidate` with `scada_state`/`scada_transition` filled in from
    `scada_majority_state` over `session_scada`'s full-run window series."""
    state, touches_transition = scada_majority_state(
        session_scada.window_start_utc, session_scada.window_s, session_scada.state,
        candidate.start_utc, candidate.end_utc,
    )
    return dataclasses.replace(candidate, scada_state=state, scada_transition=touches_transition)


def select_session(
    results_root: Path,
    session: str,
    regime: str,
    *,
    window_s: float,
    cap: int,
    session_scada: SessionScada,
    strike_intervals: Sequence[tuple[datetime, datetime]] | None = None,
    prior_ids: Mapping[tuple[str, str], str] | None = None,
    in_sample: bool = False,
) -> SelectResult:
    """Build one session's final candidate list: raw sustained (fusion) + raw
    transient (audio-beats) -> [strike-exclusion, 080726 only] -> transient dedup ->
    merge + overlap suppression -> cap -> id assignment -> stable-id pinning
    (`pin_stable_ids`) -> SCADA operating-mode lookup (`_attach_scada`) -> in-sample
    tagging (`in_sample`, `IN_SAMPLE_SESSIONS`). See the module docstring's `select`
    section for the full rationale of this exact order (exclusion BEFORE
    dedup/merge/cap, so an excluded window never consumes a top-25 slot)."""
    fusion_path = _alarms_path_for(results_root, "fusion", session, regime)
    beats_path = _alarms_path_for(results_root, "audio-beats", session, regime)

    fusion_df = _read_alarms(fusion_path)
    beats_df = _read_alarms(beats_path)

    sustained = build_sustained_episodes(
        fusion_df, window_s=window_s, alpha=FUSION_SUSTAINED_ALPHA,
        min_duration_s=FUSION_SUSTAINED_MIN_DURATION_S, session=session, modality="fusion",
        regime=regime, alarms_path=_relpath(fusion_path, results_root),
    )
    transient = build_transient_candidates(
        beats_df, window_s=window_s, alpha=AUDIOBEATS_TRANSIENT_ALPHA, session=session,
        modality="audio-beats", regime=regime, alarms_path=_relpath(beats_path, results_root),
    )
    n_sustained_raw = len(sustained)
    n_transient_raw = len(transient)

    n_excluded = 0
    if strike_intervals:
        n_raw_total = len(sustained) + len(transient)
        sustained = [
            c for c in sustained if not overlaps_any(c.start_utc, c.end_utc, strike_intervals)
        ]
        transient = [
            c for c in transient if not overlaps_any(c.start_utc, c.end_utc, strike_intervals)
        ]
        n_excluded = n_raw_total - (len(sustained) + len(transient))

    transient = dedupe_by_radius(transient, TRANSIENT_DEDUP_RADIUS_S)
    merged = suppress_overlaps([*sustained, *transient])
    n_after_merge = len(merged)

    kept = pin_stable_ids(assign_ids(cap_top_n(merged, cap)), prior_ids or {})
    kept = [
        dataclasses.replace(_attach_scada(c, session_scada), in_sample=in_sample) for c in kept
    ]

    return SelectResult(
        session=session, regime=regime, n_sustained_raw=n_sustained_raw,
        n_transient_raw=n_transient_raw, n_excluded_strike=n_excluded,
        n_after_merge=n_after_merge, kept=kept,
    )


def _candidate_to_row(c: Candidate) -> dict[str, object]:
    return {
        "session": c.session, "candidate_id": c.candidate_id, "rank": c.rank, "class": c.klass,
        "start_utc": c.start_utc.isoformat(), "end_utc": c.end_utc.isoformat(),
        "duration_s": round(c.duration_s, 3), "min_p": c.min_p, "state_name": c.state_name,
        "near_transition": c.near_transition, "scada_state": c.scada_state,
        "scada_transition": c.scada_transition, "n_windows": c.n_windows, "modality": c.modality,
        "regime": c.regime, "alarms_path": c.alarms_path, "in_sample": c.in_sample,
    }


def select_all(
    cfg: Config, out_csv: Path, *, cap: int = CANDIDATE_CAP_PER_SESSION
) -> list[SelectResult]:
    """Run `select_session` over every `SESSIONS` entry and write the combined
    `candidates.csv`. Returns the per-session `SelectResult`s (found-vs-kept
    counts) for the CLI to log.

    Loads `prior_ids` from *out_csv* BEFORE overwriting it -- `pin_stable_ids`'s
    published-id stability guarantee (module docstring) only works if the "prior"
    snapshot is read before this call's own write. Also `discover`s `cfg.data_root`
    once and loads each session's `SessionScada` (`load_session_scada`) once,
    reused across every one of that session's own candidates -- neither is
    per-candidate work.
    """
    window_s = cfg.window.window_s
    index = discover(cfg.data_root)
    prior_ids = load_prior_candidate_ids(out_csv)
    results: list[SelectResult] = []
    rows: list[dict[str, object]] = []

    for session in SESSIONS:
        regime = REGIME_BY_SESSION[session]
        strike_intervals = None
        exclusion_files = _STRIKE_EXCLUSION_FILES.get(session)
        if exclusion_files is not None:
            strikes_csv, events_csv = exclusion_files
            strike_intervals = load_strike_exclusion_intervals(
                strikes_csv, events_csv, pad_s=STRIKE_EXCLUSION_PAD_S
            )
        session_scada = load_session_scada(index, session, cfg)
        result = select_session(
            cfg.results_root, session, regime, window_s=window_s, cap=cap,
            session_scada=session_scada, strike_intervals=strike_intervals, prior_ids=prior_ids,
            in_sample=session in IN_SAMPLE_SESSIONS,
        )
        results.append(result)
        rows.extend(_candidate_to_row(c) for c in result.kept)
        logger.info(
            "candidate_kit: select %-20s regime=%-11s sustained_raw=%3d transient_raw=%3d "
            "excluded_strike=%3d after_merge=%3d kept=%3d",
            session, regime, result.n_sustained_raw, result.n_transient_raw,
            result.n_excluded_strike, result.n_after_merge, len(result.kept),
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=list(CANDIDATES_CSV_COLUMNS))
    frame.to_csv(out_csv, index=False)
    logger.info("candidate_kit: wrote %d candidate(s) -> %s", len(rows), out_csv)
    return results


# ---------------------------------------------------------------------------
# build: assets + interactive index.html
# ---------------------------------------------------------------------------


def read_candidates_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"candidates CSV not found: {path}")
    return pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)


def _candidate_from_row(row: Mapping[str, str]) -> Candidate:
    try:
        return Candidate(
            session=row["session"], klass=row["class"],
            start_utc=datetime.fromisoformat(row["start_utc"]),
            end_utc=datetime.fromisoformat(row["end_utc"]),
            duration_s=float(row["duration_s"]), min_p=float(row["min_p"]),
            state_name=row["state_name"], near_transition=row["near_transition"] == "True",
            n_windows=int(row["n_windows"]), modality=row["modality"], regime=row["regime"],
            alarms_path=row["alarms_path"], candidate_id=row["candidate_id"], rank=int(row["rank"]),
            scada_state=row["scada_state"], scada_transition=row["scada_transition"] == "True",
            # .get(...) with a default (not row["in_sample"]): a candidates.csv
            # written before this column existed must still load -- in_sample
            # defaults to False exactly like Candidate's own field default.
            in_sample=row.get("in_sample", "False") == "True",
        )
    except (KeyError, ValueError) as exc:
        cid = row.get("candidate_id", "?")
        raise ValueError(f"malformed candidates.csv row {cid!r}: {exc}") from exc


def candidates_from_csv(path: Path) -> list[Candidate]:
    df = read_candidates_csv(path)
    return [_candidate_from_row(r) for r in df.to_dict(orient="records")]


@dataclass(frozen=True)
class CandidateAssetResult:
    candidate: Candidate
    asset_start_utc: datetime
    asset_duration_s: float
    gen_wav: str
    tur_wav: str
    gen_png: str
    tur_png: str
    gen_flat_png: str
    tur_flat_png: str
    scada_png: str
    """SCADA CONTEXT PANEL (Stefan's decision 2026-08-16): the combined strip
    (power/speed/flow mini-axes, or the neutral no-Betriebsdaten placeholder) plus
    the two-row state ribbon, one PNG per candidate -- `render_scada_strip_png`.
    Unlike `gen_wav`/`tur_wav`/the four spectrogram PNGs above, this is NEVER
    reused across a `build` re-run (`build_all`'s own docstring): it is
    cheap-to-recompute from already-loaded SCADA/alarms data, never audio-derived,
    so it is always freshly rendered alongside `candidates_meta.json`/`index.html`
    themselves."""
    power_mw_1hz: list[float | None]
    speed_rpm_1hz: list[float | None]
    """The LIVE READOUT's own 1 Hz `(power, speed)` series over the asset window
    (`build_readout_series`) -- `None` entries (JSON `null`) where SCADA coverage
    is missing, and BOTH lists empty when `n_seconds == 0` (never happens in
    practice, `asset_duration_s` is always >= `ASSET_MIN_DURATION_S`) or
    all-`None` for a `has_scada=False` session (270626-pu_ph_pu_ph_pu_ph-1)."""


@dataclass(frozen=True)
class _AudioAssets:
    """The WAV/spectrogram half of one candidate's rendered assets -- exactly what
    `_reuse_existing_assets` can reconstruct without touching audio data, and what
    `render_candidate_assets` produces when it can't. Kept separate from the full
    `CandidateAssetResult` because the SCADA strip (`CandidateAssetResult.
    scada_png`/`power_mw_1hz`/`speed_rpm_1hz`) is NEVER part of the reuse decision
    (see `CandidateAssetResult.scada_png`'s own docstring) -- `build_all` always
    combines one of these with a freshly rendered SCADA strip into the final
    `CandidateAssetResult`."""

    asset_start_utc: datetime
    asset_duration_s: float
    gen_wav: str
    tur_wav: str
    gen_png: str
    tur_png: str
    gen_flat_png: str
    tur_flat_png: str


def _load_prior_meta_by_id(out_dir: Path) -> dict[str, dict[str, object]]:
    """`candidate_id -> meta dict` from an existing `<out_dir>/candidates_meta.json`,
    or `{}` if none exists yet, or it fails to parse -- a missing/corrupt prior file
    just means "nothing to reuse", never a hard error (mirrors `prepare_run`'s own
    cache-miss-is-not-an-error treatment of `results/cache/*.npz`)."""
    path = out_dir / "candidates_meta.json"
    if not path.is_file():
        return {}
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        str(e["candidate_id"]): e for e in entries if isinstance(e, dict) and "candidate_id" in e
    }


def _reuse_existing_assets(
    candidate: Candidate, prior_meta: Mapping[str, object] | None, out_dir: Path
) -> _AudioAssets | None:
    """An `_AudioAssets` reconstructed WITHOUT touching a single audio sample, if
    *prior_meta* (this candidate's own entry in a previous `build`'s `candidates_
    meta.json`, `_load_prior_meta_by_id`) proves the assets already on disk were
    rendered for the EXACT SAME candidate span/extremity -- `start_utc`/
    `duration_s`/`min_p`/`class` all matching -- AND every one of the six expected
    files is still present. Extracting a WAV clip from a multi-gigabyte burst file
    is expensive; a candidate whose identity (module docstring: pinned by
    `pin_stable_ids`) AND whose own span/extremity are BOTH unchanged since the
    last `build` would only ever re-produce the identical bytes. (The SCADA strip
    PNG is deliberately NOT part of this reuse check -- `CandidateAssetResult.
    scada_png`'s own docstring.)

    Returns `None` (render for real) otherwise: no prior entry, a mismatched one
    (rare -- an id kept its identity via `pin_stable_ids` but its underlying span
    actually moved, e.g. a re-`select` on updated alarms.parquet data), or a
    missing file (an interrupted previous `build`).
    """
    if prior_meta is None:
        return None
    if (
        str(prior_meta.get("start_utc")) != candidate.start_utc.isoformat()
        or str(prior_meta.get("class")) != candidate.klass
        or float(prior_meta.get("duration_s", "nan")) != candidate.duration_s  # type: ignore[arg-type]
        or float(prior_meta.get("min_p", "nan")) != candidate.min_p  # type: ignore[arg-type]
    ):
        return None
    try:
        gen_wav, tur_wav = str(prior_meta["gen_wav"]), str(prior_meta["tur_wav"])
        gen_flat_png = str(prior_meta["gen_flat_png"])
        tur_flat_png = str(prior_meta["tur_flat_png"])
        asset_start_utc = datetime.fromisoformat(str(prior_meta["asset_start_utc"]))
        asset_duration_s = float(prior_meta["asset_duration_s"])  # type: ignore[arg-type]
    except (KeyError, ValueError):
        return None

    # gen_png/tur_png (the tall, axis-labelled spectrogram) are never stored in
    # candidates_meta.json (index.html only ever needs the flat ones) -- their path
    # follows `render_candidate_assets`' own deterministic naming instead.
    base = f"{candidate.session}/{candidate.candidate_id}"
    gen_png, tur_png = f"{base}_gen.png", f"{base}_tur.png"

    all_rel = (gen_wav, tur_wav, gen_png, tur_png, gen_flat_png, tur_flat_png)
    if not all((out_dir / rel).is_file() for rel in all_rel):
        return None

    return _AudioAssets(
        asset_start_utc=asset_start_utc, asset_duration_s=asset_duration_s,
        gen_wav=gen_wav, tur_wav=tur_wav, gen_png=gen_png, tur_png=tur_png,
        gen_flat_png=gen_flat_png, tur_flat_png=tur_flat_png,
    )


def render_candidate_assets(
    index: RecordingIndex, candidate: Candidate, out_dir: Path
) -> _AudioAssets:
    """Render one candidate's WAV/PNG assets under `<out_dir>/<session>/`, exactly
    like `annotation_kit.build_session`'s own per-event loop -- `extract_stream_
    clip`/`render_spectrogram_png`/`render_flat_spectrogram_png`/`_resample_to_
    target`/`_write_clip_wav` are the SAME imported functions, only the source
    window (`asset_window` around the candidate's own span, not a ground-truth
    event) and output naming (`<candidate_id>_<stream>.*`, not `event_<id>_<kind>__
    <stream>.*`) differ."""
    run = mda._get_run(index, candidate.session)
    offset_ns = run_utc_offset_ns(run)
    asset_start, asset_end = asset_window(
        candidate.start_utc, candidate.end_utc,
        pad_s=ASSET_PAD_S, min_s=ASSET_MIN_DURATION_S, max_s=ASSET_MAX_DURATION_S,
    )

    session_dir = out_dir / candidate.session
    session_dir.mkdir(parents=True, exist_ok=True)

    wav_rel: dict[str, str] = {}
    png_rel: dict[str, str] = {}
    flat_png_rel: dict[str, str] = {}
    for stream_key, stream_name in ak._STREAM_NAME_BY_KEY.items():
        files = run.files.get(stream_name, [])
        if not files:
            raise ValueError(f"run {run.name!r} has no {stream_name!r} files")
        clip = ak.extract_stream_clip(
            files, offset_ns, mda.MONO_CHANNEL_INDEX, asset_start, asset_end
        )
        base = f"{candidate.candidate_id}_{stream_key}"
        resampled = mda._resample_to_target(clip.samples, clip.rate_hz)
        mda._write_clip_wav(session_dir / f"{base}.wav", resampled)
        ak.render_spectrogram_png(
            session_dir / f"{base}.png", clip.samples, clip.rate_hz,
            kind=f"{candidate.klass} candidate {candidate.candidate_id}",
            stream_label=ak._STREAM_LABEL[stream_key], snippet_start_utc=asset_start,
        )
        ak.render_flat_spectrogram_png(session_dir / f"{base}_flat.png", clip.samples, clip.rate_hz)
        wav_rel[stream_key] = f"{candidate.session}/{base}.wav"
        png_rel[stream_key] = f"{candidate.session}/{base}.png"
        flat_png_rel[stream_key] = f"{candidate.session}/{base}_flat.png"
        logger.info(
            "candidate_kit: build %s %s (%s) -> %s", candidate.session, candidate.candidate_id,
            stream_key, wav_rel[stream_key],
        )

    return _AudioAssets(
        asset_start_utc=asset_start,
        asset_duration_s=(asset_end - asset_start).total_seconds(),
        gen_wav=wav_rel["gen"], tur_wav=wav_rel["tur"],
        gen_png=png_rel["gen"], tur_png=png_rel["tur"],
        gen_flat_png=flat_png_rel["gen"], tur_flat_png=flat_png_rel["tur"],
    )


# ---------------------------------------------------------------------------
# SCADA context panel: strip PNG (power/speed/flow or placeholder) + state ribbon
# ---------------------------------------------------------------------------

_SCADA_ACCENT_HEX = "#3b82f6"
"""Candidate-span shading on every mini-axis/the ribbon -- matches `_CANDIDATE_CSS`'s
`--accent` so the strip reads as part of the same visual language as the rest of
the card."""
_SCADA_MARKER_HEX = "#dc2626"
"""Anomaly-start dashed line -- matches `_CANDIDATE_CSS`'s `--err` (distinct from
`--playhead`'s `#ef4444`, which the LIVE playhead canvas overlay uses, so a
static "where the anomaly starts" marker is never confused with the moving
"where playback is now" line even when both happen to fall at the same x)."""
_SCADA_BG_HEX = "#f8f9fb"
_SCADA_PLACEHOLDER_BG_HEX = "#eef0f4"
_SCADA_PLACEHOLDER_TEXT = "No SCADA available for this day"

_SCADA_MINI_AXIS_HEIGHT_PX = 54
_SCADA_N_MINI_AXES = 3
_SCADA_RIBBON_ROW_HEIGHT_PX = 16
_SCADA_RIBBON_N_ROWS = 2
_SCADA_STRIP_HEIGHT_PX = _SCADA_MINI_AXIS_HEIGHT_PX * _SCADA_N_MINI_AXES
_SCADA_RIBBON_HEIGHT_PX = _SCADA_RIBBON_ROW_HEIGHT_PX * _SCADA_RIBBON_N_ROWS
_SCADA_TOTAL_HEIGHT_PX = _SCADA_STRIP_HEIGHT_PX + _SCADA_RIBBON_HEIGHT_PX

_SCADA_CHANNELS_META: tuple[tuple[str, str], ...] = (
    ("P [MW]", "#2563eb"),
    ("n [rpm]", "#059669"),
    ("Q [m3/s]", "#7c3aed"),
)
"""`(small English axis label, line color)` for the three stacked mini-axes, in
display order -- active power, shaft speed, net flow (`SessionScada.power_mw`/
`speed_rpm`/`flow_net_m3s`, module docstring build/1). `P`/`n` match the LIVE
READOUT's own variable names (module docstring build/3)."""


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    """`"#rrggbb"` -> `(r, g, b)` floats in `[0, 1]` -- `matplotlib.imshow`'s own
    expected float-RGB range, for the state-ribbon raster (`_draw_state_ribbon`)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return r, g, b


def _draw_scada_mini_axis(
    ax: Any,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    duration_s: float,
    color: str,
    label: str,
    span_s: tuple[float, float],
    marker_s: float,
) -> None:
    """One power/speed/flow mini-axis: full horizontal bleed (the caller's
    `fig.add_axes` rect already spans x=[0,1] of the figure) so the x-axis stays
    pixel-linear with the flat spectrogram -- no matplotlib x-margin, no x-ticks.
    "axis labels small" (module docstring build/1) is a small in-plot text label
    (*label*) rather than a conventional matplotlib y-axis label, which would
    need left-margin space that breaks the pixel-linear width contract. A zero
    reference line is drawn when 0 falls within the (padded) value range -- power
    and speed are both signed (turbine positive / pump negative, `SessionScada`'s
    own docstring)."""
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0.0, duration_s)
    ax.margins(x=0)

    finite = ys[~np.isnan(ys)] if ys.size else ys
    if finite.size:
        lo, hi = float(np.min(finite)), float(np.max(finite))
        pad = max((hi - lo) * 0.15, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
        if lo - pad <= 0.0 <= hi + pad:
            ax.axhline(0.0, color="#9ca3af", linewidth=0.5, zorder=1)
        ax.plot(xs, ys, color=color, linewidth=1.1, zorder=3)
    else:
        ax.set_ylim(0.0, 1.0)
        ax.text(
            0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center",
            fontsize=7, color="#9ca3af",
        )

    span_start_s, span_end_s = span_s
    ax.axvspan(span_start_s, span_end_s, color=_SCADA_ACCENT_HEX, alpha=0.15, zorder=0, linewidth=0)
    ax.axvline(marker_s, color=_SCADA_MARKER_HEX, linewidth=1.2, linestyle="--", zorder=4)
    ax.text(
        0.006, 0.9, label, transform=ax.transAxes, fontsize=7, va="top", ha="left",
        color="#374151", zorder=5,
    )


def _draw_state_ribbon(
    ax: Any,
    scada_row: Sequence[str],
    detected_row: Sequence[str],
    *,
    duration_s: float,
    span_s: tuple[float, float],
    marker_s: float,
) -> None:
    """The two-row per-second state ribbon (module docstring build/2): row 0 (top)
    is the SCADA state, row 1 (bottom) the detected state, each per-second cell
    colored by `state_color` (the ONE shared legend, `STATE_LEGEND`). Rendered as
    a small RGB raster (`imshow`, `interpolation="nearest"` for crisp per-second
    cell edges) stretched across `extent=(0, duration_s, 0, 2)` -- full
    horizontal bleed, the SAME pixel-linear x-mapping as the mini-axes above and
    the flat spectrogram."""
    n = max(len(scada_row), len(detected_row), 1)
    rgb = np.zeros((2, n, 3), dtype=np.float64)
    for col in range(n):
        top = scada_row[col] if col < len(scada_row) else "unknown"
        bottom = detected_row[col] if col < len(detected_row) else "unknown"
        rgb[0, col] = _hex_to_rgb01(state_color(top))
        rgb[1, col] = _hex_to_rgb01(state_color(bottom))

    ax.set_xlim(0.0, duration_s)
    ax.set_ylim(0.0, 2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(x=0)
    ax.imshow(
        rgb, extent=(0.0, duration_s, 0.0, 2.0), origin="upper", aspect="auto",
        interpolation="nearest", zorder=0,
    )
    ax.axhline(1.0, color="white", linewidth=0.6, zorder=1)
    span_start_s, span_end_s = span_s
    ax.axvspan(
        span_start_s, span_end_s, fill=False, edgecolor=_SCADA_ACCENT_HEX,
        linewidth=1.2, zorder=2,
    )
    ax.axvline(marker_s, color=_SCADA_MARKER_HEX, linewidth=1.2, linestyle="--", zorder=3)
    ax.text(
        0.006, 0.97, "SCADA", transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
        color="white", zorder=4,
    )
    ax.text(
        0.006, 0.47, "Detected", transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
        color="white", zorder=4,
    )


def render_scada_strip_png(
    path: Path,
    *,
    candidate: Candidate,
    asset_start_utc: datetime,
    asset_duration_s: float,
    n_seconds: int,
    session_scada: SessionScada,
    detected_window_start_utc: Sequence[datetime],
    detected_window_s: float,
    detected_states: Sequence[str],
) -> None:
    """The SCADA CONTEXT PANEL's single combined PNG for one candidate (Stefan's
    decision 2026-08-16, module docstring build/1-2): three stacked mini-axes
    (active power, shaft speed, net flow) OR the neutral no-Betriebsdaten
    placeholder (`session_scada.has_scada is False`, e.g.
    `270626-pu_ph_pu_ph_pu_ph-1`), followed by a two-row per-second state ribbon
    (SCADA state, detected state) -- ONE image so a single overlay canvas
    (index.html's own JS) can draw ONE playhead line through both at once.

    EXACTLY `ak.flat_spectrogram_width_px(asset_duration_s)` pixels wide -- the
    SAME `ak._FLAT_PX_PER_S` linear px-per-second mapping as `gen_flat_png`/
    `tur_flat_png` (both rendered over the SAME *asset_duration_s*), so the
    playhead stays in sync across the spectrogram lanes and this strip. The
    candidate's own `[start_utc, end_utc)` span is shaded and its start marked
    (dashed) on every mini-axis and the ribbon alike, in the same colors
    `_CANDIDATE_CSS` already uses for `--accent`/`--err`.
    """
    width_px = max(1, ak.flat_spectrogram_width_px(asset_duration_s))
    asset_end_utc = asset_start_utc + timedelta(seconds=asset_duration_s)
    span_s = (
        (candidate.start_utc - asset_start_utc).total_seconds(),
        (candidate.end_utc - asset_start_utc).total_seconds(),
    )
    marker_s = span_s[0]

    plt = mda._pyplot()
    fig = plt.figure(figsize=(width_px / 100, _SCADA_TOTAL_HEIGHT_PX / 100), dpi=100)
    fig.patch.set_facecolor(_SCADA_BG_HEX)

    ribbon_frac = _SCADA_RIBBON_HEIGHT_PX / _SCADA_TOTAL_HEIGHT_PX
    strip_frac = 1.0 - ribbon_frac

    if session_scada.has_scada:
        mini_frac = strip_frac / _SCADA_N_MINI_AXES
        channel_values = (
            session_scada.power_mw, session_scada.speed_rpm, session_scada.flow_net_m3s,
        )
        for i, ((label, color), values) in enumerate(
            zip(_SCADA_CHANNELS_META, channel_values, strict=True)
        ):
            y0 = ribbon_frac + strip_frac - (i + 1) * mini_frac
            ax = fig.add_axes((0.0, y0, 1.0, mini_frac))
            starts, vals = slice_window_series(
                session_scada.window_start_utc, session_scada.window_s, values,
                asset_start_utc, asset_end_utc,
            )
            xs = np.array([(s - asset_start_utc).total_seconds() for s in starts])
            ys = np.array(vals, dtype=np.float64)
            _draw_scada_mini_axis(
                ax, xs, ys, duration_s=asset_duration_s, color=color, label=label,
                span_s=span_s, marker_s=marker_s,
            )
    else:
        ax = fig.add_axes((0.0, ribbon_frac, 1.0, strip_frac))
        ax.set_xlim(0.0, asset_duration_s)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")
        ax.set_facecolor(_SCADA_PLACEHOLDER_BG_HEX)
        ax.text(
            0.5, 0.5, _SCADA_PLACEHOLDER_TEXT, transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color="#6b7280",
        )

    ribbon_ax = fig.add_axes((0.0, 0.0, 1.0, ribbon_frac))
    scada_row = resample_states_to_seconds(
        session_scada.window_start_utc, session_scada.window_s, session_scada.state,
        asset_start_utc, n_seconds,
    )
    detected_row = resample_states_to_seconds(
        detected_window_start_utc, detected_window_s, detected_states, asset_start_utc, n_seconds,
    )
    _draw_state_ribbon(
        ribbon_ax, scada_row, detected_row, duration_s=asset_duration_s, span_s=span_s,
        marker_s=marker_s,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)


def _asset_result_to_meta(r: CandidateAssetResult) -> dict[str, object]:
    c = r.candidate
    return {
        "session": c.session, "candidate_id": c.candidate_id, "rank": c.rank, "class": c.klass,
        "start_utc": c.start_utc.isoformat(), "duration_s": c.duration_s, "min_p": c.min_p,
        "state_name": c.state_name, "near_transition": c.near_transition,
        "scada_state": c.scada_state, "scada_transition": c.scada_transition,
        "in_sample": c.in_sample,
        "mode_mismatch": scada_detector_mismatch(c.scada_state, c.state_name),
        "criterion_text": criterion_sentence(c),
        "asset_start_utc": r.asset_start_utc.isoformat(), "asset_duration_s": r.asset_duration_s,
        "gen_wav": r.gen_wav, "tur_wav": r.tur_wav,
        "gen_flat_png": r.gen_flat_png, "tur_flat_png": r.tur_flat_png,
        "scada_png": r.scada_png,
        "power_mw_1hz": r.power_mw_1hz, "speed_rpm_1hz": r.speed_rpm_1hz,
    }


def write_candidates_meta_json(results: Sequence[CandidateAssetResult], out_dir: Path) -> Path:
    """`<out_dir>/candidates_meta.json` -- a standalone, machine-readable dump of
    every rendered candidate's UI-facing fields (mirrors `annotation_kit.
    write_events_meta_json`'s own precedent); NOT read by `index.html` itself
    (embedded inline instead, `file://` CORS -- see `render_index_html`'s
    docstring), kept for external tooling/inspection."""
    path = out_dir / "candidates_meta.json"
    payload = json.dumps(
        [_asset_result_to_meta(r) for r in results], indent=2, ensure_ascii=False
    )
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def _load_detected_state_series(path: Path) -> tuple[list[datetime], list[str]]:
    """`(window_start_utc, state_name)` from one `alarms.parquet` file, sorted
    ascending by time -- the STATE RIBBON's own "detected state" row source
    (module docstring build/2), read via the SAME `_read_alarms` `select` already
    uses. Explicit `sort_values` (defensive, mirrors `build_sustained_episodes`'
    own `.sort_values("window")`): `slice_window_series`'s `bisect` pre-filter
    requires ascending order, and nothing about `_read_alarms`'s contract
    guarantees the on-disk row order."""
    df = _read_alarms(path).sort_values("t_utc_ns")
    starts = [_utc_from_ns(int(t)) for t in df["t_utc_ns"]]
    states = [str(s) for s in df["state_name"]]
    return starts, states


def build_all(cfg: Config, candidates_csv: Path, out_dir: Path) -> list[CandidateAssetResult]:
    """Render (or reuse, `_reuse_existing_assets`) every candidate's WAV/spectrogram
    assets, ALWAYS (re)render its SCADA strip PNG (`render_scada_strip_png`) and
    1 Hz live-readout series (`build_readout_series`), then (re)write `candidates_
    meta.json`/`index.html` -- the SCADA context panel is cheap-to-recompute,
    never audio-derived (`CandidateAssetResult.scada_png`'s own docstring), so it
    is never gated behind the WAV/spectrogram reuse check, mirroring `scada_state`/
    `mode_mismatch`/`criterion_text`'s own pre-existing "always fresh" treatment.

    `discover(cfg.data_root)` is no longer deferred (unlike before the SCADA
    context panel existed): every session's `SessionScada` (`load_session_scada`,
    loaded once per session and reused across that session's own candidates,
    `session_scada_by_session`) needs `index` regardless of whether any candidate
    in it needs real WAV rendering. `discover` itself is a cheap filesystem index
    (no audio decoding), so this is not a meaningful cost. Each unique
    `alarms.parquet` (`Candidate.alarms_path`) is likewise read and time-converted
    only ONCE (`detected_state_by_alarms_path`), reused across every candidate
    that shares it (typically ~25 per session).
    """
    candidates = candidates_from_csv(candidates_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    prior_meta_by_id = _load_prior_meta_by_id(out_dir)
    index = discover(cfg.data_root)

    session_scada_by_session: dict[str, SessionScada] = {}
    detected_state_by_alarms_path: dict[str, tuple[list[datetime], list[str]]] = {}

    results: list[CandidateAssetResult] = []
    n_reused = 0
    for c in candidates:
        audio = _reuse_existing_assets(c, prior_meta_by_id.get(c.candidate_id), out_dir)
        if audio is not None:
            n_reused += 1
        else:
            audio = render_candidate_assets(index, c, out_dir)

        if c.session not in session_scada_by_session:
            session_scada_by_session[c.session] = load_session_scada(index, c.session, cfg)
        session_scada = session_scada_by_session[c.session]

        if c.alarms_path not in detected_state_by_alarms_path:
            detected_state_by_alarms_path[c.alarms_path] = _load_detected_state_series(
                cfg.results_root / c.alarms_path
            )
        detected_starts, detected_states = detected_state_by_alarms_path[c.alarms_path]

        n_seconds = max(1, math.ceil(audio.asset_duration_s))
        scada_png_rel = f"{c.session}/{c.candidate_id}_scada.png"
        render_scada_strip_png(
            out_dir / scada_png_rel, candidate=c, asset_start_utc=audio.asset_start_utc,
            asset_duration_s=audio.asset_duration_s, n_seconds=n_seconds,
            session_scada=session_scada, detected_window_start_utc=detected_starts,
            detected_window_s=cfg.window.window_s, detected_states=detected_states,
        )
        power_1hz, speed_1hz = build_readout_series(session_scada, audio.asset_start_utc, n_seconds)

        results.append(
            CandidateAssetResult(
                candidate=c, asset_start_utc=audio.asset_start_utc,
                asset_duration_s=audio.asset_duration_s,
                gen_wav=audio.gen_wav, tur_wav=audio.tur_wav,
                gen_png=audio.gen_png, tur_png=audio.tur_png,
                gen_flat_png=audio.gen_flat_png, tur_flat_png=audio.tur_flat_png,
                scada_png=scada_png_rel, power_mw_1hz=power_1hz, speed_rpm_1hz=speed_1hz,
            )
        )

    logger.info(
        "candidate_kit: build reused existing WAV/spectrogram assets for %d/%d candidate(s), "
        "rendered %d fresh (SCADA strip always freshly rendered for all %d)",
        n_reused, len(candidates), len(candidates) - n_reused, len(candidates),
    )
    write_candidates_meta_json(results, out_dir)
    render_index_html(results, out_dir)
    return results


# ---------------------------------------------------------------------------
# Interactive index.html
# ---------------------------------------------------------------------------

_SESSION_LABEL: dict[str, str] = {
    "250526-tu": "25.05. – turbine operation",
    "250526-pu-morning": "25.05. – pump operation (morning)",
    "290626-tu": "29.06. – turbine operation",
    "290626-pu": "29.06. – pump operation",
    "010726-tu_ph_tu": "01.07. – turbine/phase-shifter/turbine",
    "010726-pu": "01.07. – pump operation",
    "080726-pu_strikes": "08.07. – pump trial (Schonhammer campaign, events excluded)",
    "080726-st_strikes": "08.07. – standstill (Schonhammer campaign, events excluded)",
    "270626-pu_ph_pu_ph_pu_ph-1": "27.06. – pump/phase-shifter cycling (no SCADA, held-out)",
    "010726-tu1-morning": "01.07. – turbine operation (morning, in-sample fit-pool day)",
    "010726-tu2": "01.07. – turbine operation (session 2, in-sample fit-pool day)",
}

_CANDIDATE_CSS = """
:root {
  color-scheme: light dark;
  --border: #8888;
  --muted: #808080;
  --accent: #3b82f6;
  --playhead: #ef4444;
  --bg-soft: #80808014;
  --bg-soft-2: #80808022;
  --ok: #16a34a;
  --err: #dc2626;
  --sustained: #f59e0b;
  --transient: #a855f7;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 2rem auto; max-width: 1400px; padding: 0 1rem; line-height: 1.5; }
h1 { font-size: 1.5rem; }
h2 { font-size: 1.2rem; margin-top: 2.5rem; border-bottom: 1px solid var(--border);
     padding-bottom: 0.3rem; }
code { background: var(--bg-soft-2); padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.92em; }
.instructions { background: var(--bg-soft); border: 1px solid var(--border); border-radius: 8px;
                padding: 0.8rem 1.2rem; margin-bottom: 1.5rem; font-size: 0.93rem; }
.instructions p { margin: 0.4em 0; }
.legend { background: var(--bg-soft); border: 1px solid var(--border); border-radius: 8px;
          padding: 0.8rem 1.2rem; margin-bottom: 1.5rem; font-size: 0.93rem; }
.legend h2 { margin: 0 0 0.5rem; border: none; padding: 0; font-size: 1.05rem; }
.legend p { margin: 0.5em 0; }
.legend .class-term { font-weight: 600; }
.hint { color: var(--muted); font-size: 0.85rem; margin: 0.3em 0; }
.top-toolbar { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1rem; }
.session-section { margin-top: 2rem; }
.session-toolbar { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1rem; }
.session-status { color: var(--muted); font-size: 0.85rem; }

.candidate-card { border: 1px solid var(--border); border-radius: 8px; padding: 0.9rem 1.1rem;
                   margin-bottom: 1.2rem; outline-offset: 2px; }
.card-head { display: flex; justify-content: space-between; align-items: baseline;
             gap: 1rem; flex-wrap: wrap; }
.card-title { font-weight: 600; }
.class-badge { border-radius: 10px; padding: 0.05rem 0.55rem; font-size: 0.78rem; color: white; }
.class-badge.sustained { background: var(--sustained); }
.class-badge.transient { background: var(--transient); }
.near-transition-badge { border: 1px solid var(--err); color: var(--err); border-radius: 10px;
                          padding: 0.05rem 0.55rem; font-size: 0.78rem; }
.meta-line { color: var(--muted); font-size: 0.85rem; margin: 0.3em 0; }
.meta-line b { color: inherit; font-weight: 600; }

.mode-row { display: flex; align-items: baseline; gap: 0.9rem; flex-wrap: wrap;
            margin: 0.6rem 0 0.15rem; }
.mode-scada { font-size: 1rem; font-weight: 700; }
.mode-detector { font-size: 0.85rem; color: var(--muted); }
.mismatch-badge { background: var(--err); color: white; border-radius: 10px;
                   padding: 0.05rem 0.55rem; font-size: 0.78rem; }
.scada-transition-badge { border: 1px solid var(--muted); color: var(--muted); border-radius: 10px;
                           padding: 0.05rem 0.55rem; font-size: 0.78rem; }
.in-sample-badge { border: 1px solid var(--muted); color: var(--muted); border-radius: 10px;
                    padding: 0.05rem 0.55rem; font-size: 0.78rem; }
.criterion-line { color: var(--muted); font-size: 0.85rem; margin: 0.2em 0 0.6em;
                   font-style: italic; }

.lanes { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 0.6rem 0; }
@media (max-width: 800px) { .lanes { grid-template-columns: 1fr; } }
.lane { border: 1px solid var(--border); border-radius: 6px; padding: 0.5rem; outline-offset: 2px; }
.lane:focus, .lane:focus-visible { outline: 2px solid var(--accent); }
.lane-title { font-size: 0.8rem; color: var(--muted); margin-bottom: 0.3rem; }
.spectro-scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: 4px; }
.spectro-wrapper { position: relative; display: block; }
.spectro-img { display: block; width: 100%; height: 100%; }
.overlay-canvas { position: absolute; top: 0; left: 0; cursor: crosshair; display: block; }
audio.lane-audio { width: 100%; margin: 0.4rem 0; }
.lane-time { font-variant-numeric: tabular-nums; font-family: ui-monospace, Menlo, monospace;
             font-size: 0.82rem; color: var(--muted); }

.scada-block { margin: 0.7rem 0 0.3rem; }
.scada-title { font-size: 0.8rem; color: var(--muted); margin-bottom: 0.3rem; }
.scada-scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: 4px; }
.scada-wrapper { position: relative; display: block; }
.scada-img { display: block; width: 100%; height: 100%; }
.scada-overlay-canvas { position: absolute; top: 0; left: 0; display: block; pointer-events: none; }
.scada-readout { font-variant-numeric: tabular-nums; font-family: ui-monospace, Menlo, monospace;
                  font-size: 0.82rem; color: var(--muted); margin-top: 0.3rem; }
.scada-readout b { color: inherit; font-weight: 600; }

.state-legend { display: flex; gap: 0.9rem; flex-wrap: wrap; margin-top: 0.5rem;
                 font-size: 0.82rem; }
.state-legend-item { display: inline-flex; align-items: center; gap: 0.35rem; }
.state-swatch { width: 0.85em; height: 0.85em; border-radius: 3px; border: 1px solid var(--border);
                 display: inline-block; flex: none; }

.assessment-row { display: flex; gap: 1rem; align-items: flex-start; flex-wrap: wrap;
                  margin: 0.6rem 0; }
.assessment-options { display: flex; gap: 0.8rem; flex-wrap: wrap; font-size: 0.85rem; }
.assessment-options label { display: inline-flex; align-items: center; gap: 0.3rem;
                             cursor: pointer; }
.note-label { flex: 1 1 260px; display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem;
              color: var(--muted); }
.note-input { flex: 1; padding: 0.25rem 0.5rem; border: 1px solid var(--border);
              border-radius: 4px; background: transparent; color: inherit; font-size: 0.85rem; }

.card-footer { display: flex; justify-content: flex-end; }
.save-indicator { color: var(--muted); font-size: 0.78rem; }
.save-indicator.save-error { color: var(--err); }

button, input, select { font: inherit; }
"""

_CANDIDATE_LEGEND_HTML = """<section class="legend">
<h2>What do the categories mean?</h2>
<p><span class="class-term">sustained</span> (fusion path): several seconds
contiguously anomalous relative to the normal behavior of the respective
operating mode &ndash; at least 3 consecutive windows with
p&nbsp;&lt;&nbsp;0.01 (&gt;=3&nbsp;s). Typical for acoustic anomalies
of this kind: a changed timbre, rubbing, flow noise.</p>
<p><span class="class-term">transient</span> (BEATs path): a single short impulse,
extremely unlikely under the learned normal behavior &ndash; a single
window with p&nbsp;&lt;&nbsp;0.001. Typical: a click, a strike, a switching noise.</p>
<p>Each card also states the <strong>exact trigger reason</strong>
(class, window count/duration, minimum p-value) as a sentence below the mode line.</p>
<p><strong>Mode (SCADA)</strong> is the operating mode derived rule-based from the
operating data (speed, power, flow) for the candidate's time window &ndash;
the more reliable figure. <strong>Mode (Detector)</strong> is the mode
recognized by the unsupervised state cluster; this can, e.g., mislabel
pump-operation windows as &ldquo;turbine&rdquo; (a known limitation
of the cluster naming), and is generally unreliable near operating transitions.
When the two disagree, a warning
appears &ndash; when in doubt, SCADA governs. The &ldquo;SCADA:
transition/ramp&rdquo; note additionally flags that the time window itself touches a
ramp-up/ramp-down or a power ramp, i.e. the SCADA figure is the
majority of a non-uniform set of windows.</p>
<p>A &ldquo;<strong>in-sample (fit-pool day)</strong>&rdquo; badge marks a candidate
whose recording day was itself used to fit the detector/scoring model &ndash; its
evidentiary status is weaker than a genuinely held-out day, shown here rather than
left implicit.</p>
<p><strong>SCADA context panel:</strong> below the two audio lanes, each card also
shows the actual operating data (active power, shaft speed, net flow) over the
same snippet, with the candidate window shaded and its start marked (dashed) --
so you can judge what the machine was actually doing, not just what the model
flagged. Underneath it, a two-row <strong>state ribbon</strong> shows the
SCADA-derived state (top row) and the detector's own state (bottom row) for every
second of the snippet, colored per the legend below. While a lane plays, the
current power/speed reading is shown next to the panel's own time display,
following the playhead. 27.06. has no Betriebsdaten at all, so its panel shows a
placeholder instead of power/speed/flow (the state ribbon's SCADA row is then
uniformly grey/&ldquo;unknown&rdquo;, correctly, since there is nothing to derive
a state from).</p>
<div class="state-legend">__STATE_LEGEND_HTML__</div>
<p><strong>Assessment:</strong> <em>plausible anomaly</em> = sounds like a genuine
anomaly &middot; <em>operational/explained</em> = plausibly explained by normal
operation (e.g. a switching event) &middot; <em>artifact/sensor</em> =
likely a measurement/sensor artifact &middot; <em>no finding</em> = listened,
nothing noteworthy &middot; <em>unclear</em> = not
confidently assessable. An honest &ldquo;no finding&rdquo; or
&ldquo;unclear&rdquo; is a full, welcome result &ndash; this register is
explicitly <strong>not ground truth</strong>, but a qualitative listening
review.</p>
</section>
"""

_CANDIDATE_INSTRUCTIONS_HTML = """<section class="instructions">
<p><strong>Listen &amp; look:</strong> clicking a spectrogram jumps to that
position and continues playback from there; the space bar plays/pauses the
respective lane (the lane must be focused, e.g. after a click). Two independent
lanes per candidate (generator/turbine microphone).</p>
<p><strong>Assess:</strong> choose one assessment per candidate (plausible
anomaly / operational-explained / artifact-sensor / unclear) and optionally enter a note
&ndash; saved locally as you go (shown as &ldquo;saved&rdquo;),
persists across reloads.</p>
<p><strong>Export:</strong> at the end of each session click <strong>&ldquo;Export
CSV&rdquo;</strong>, or <strong>&ldquo;Export all
sessions&rdquo;</strong> above &ndash; this file is the input for
<code>candidate_kit.py compile</code>.</p>
</section>
"""


def _render_state_legend_html() -> str:
    """The state-color legend swatches (module docstring build/2: "a small legend
    once at the page top") -- built from `STATE_LEGEND` (the SAME source
    `state_color` reads for both ribbon rows), so the legend can never drift out
    of sync with the ribbon's own actual colors."""
    return "".join(
        f'<span class="state-legend-item"><span class="state-swatch" '
        f'style="background:{color}"></span>{name}</span>'
        for name, color in STATE_LEGEND
    )


_CANDIDATE_JS = r"""
(function () {
"use strict";

var STORAGE_PREFIX = "candidate-review:v1:";
var FLAT_PX_PER_S = __FLAT_PX_PER_S__;
var FLAT_HEIGHT_PX = __FLAT_HEIGHT_PX__;
var SCADA_HEIGHT_PX = __SCADA_HEIGHT_PX__;
var ASSESSMENT_VALUES = __ASSESSMENT_VALUES_JSON__;
var ARROW_STEP_S = 0.5;
var ARROW_STEP_FINE_S = 0.05;
var EXPORT_CSV_HEADER = [
  "session", "candidate_id", "class", "start_utc", "duration_s", "min_p", "state_name",
  "near_transition", "scada_state", "scada_transition", "assessment", "note"
];

var metaEl = document.getElementById("candidates-meta-data");
var labelsEl = document.getElementById("session-labels-data");
var CANDIDATES_META = JSON.parse(metaEl.textContent);
var SESSION_LABELS = JSON.parse(labelsEl.textContent);

function flatWidthPx(durationS) {
  return Math.round(FLAT_PX_PER_S * durationS);
}

function storageKey(candidateId) {
  return STORAGE_PREFIX + candidateId;
}

function defaultState() {
  return { assessment: "", note: "" };
}

function loadState(candidateId) {
  var raw = null;
  try {
    raw = localStorage.getItem(storageKey(candidateId));
  } catch (e) {
    raw = null;
  }
  if (!raw) return defaultState();
  try {
    var parsed = JSON.parse(raw);
    return {
      assessment: typeof parsed.assessment === "string" ? parsed.assessment : "",
      note: typeof parsed.notes === "string" ? parsed.notes
          : (typeof parsed.note === "string" ? parsed.note : ""),
    };
  } catch (e) {
    return defaultState();
  }
}

function saveState(card) {
  try {
    localStorage.setItem(storageKey(card.meta.candidate_id), JSON.stringify(card.state));
    card.saveStatus = "saved";
  } catch (e) {
    card.saveStatus = "error";
  }
  updateSaveIndicator(card);
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function pad2(n) {
  return (n < 10 ? "0" : "") + n;
}

function pad3(n) {
  if (n < 10) return "00" + n;
  if (n < 100) return "0" + n;
  return "" + n;
}

function formatClock(t) {
  if (!isFinite(t) || t < 0) t = 0;
  var m = Math.floor(t / 60);
  var s = Math.floor(t % 60);
  var ms = Math.round((t - Math.floor(t)) * 1000);
  if (ms === 1000) { ms = 0; s += 1; }
  if (s === 60) { s = 0; m += 1; }
  return pad2(m) + ":" + pad2(s) + "." + pad3(ms);
}

function nowClock() {
  var d = new Date();
  return pad2(d.getHours()) + ":" + pad2(d.getMinutes()) + ":" + pad2(d.getSeconds());
}

function csvField(value) {
  var s = value === null || value === undefined ? "" : String(value);
  if (/[",\r\n]/.test(s)) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function downloadTextFile(filename, text) {
  var blob = new Blob([text], { type: "text/csv" });
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function el(tag, attrs, children) {
  var node = document.createElement(tag);
  if (attrs) {
    Object.keys(attrs).forEach(function (k) {
      if (k === "class") node.className = attrs[k];
      else if (k === "text") node.textContent = attrs[k];
      else node.setAttribute(k, attrs[k]);
    });
  }
  (children || []).forEach(function (c) { node.appendChild(c); });
  return node;
}

var CARDS = {};
var CARDS_BY_SESSION = {};
var LANES = [];

function buildLane(card, key, wavPath, flatPngPath, label) {
  var lane = { card: card, key: key, naturalWidth: 0, naturalHeight: 0 };

  var wrap = el("div", { class: "lane", tabindex: "0" });
  wrap.appendChild(el("div", { class: "lane-title", text: label }));

  var scroll = el("div", { class: "spectro-scroll" });
  var wrapper = el("div", { class: "spectro-wrapper" });
  var img = el("img", { class: "spectro-img", alt: label + " spectrogram" });
  var canvas = el("canvas", { class: "overlay-canvas" });
  wrapper.appendChild(img);
  wrapper.appendChild(canvas);
  scroll.appendChild(wrapper);
  wrap.appendChild(scroll);

  var audio = el("audio", { class: "lane-audio", preload: "metadata", controls: "", src: wavPath });
  wrap.appendChild(audio);

  var timeDisplay = el("div", { class: "lane-time" });
  wrap.appendChild(timeDisplay);

  lane.el = wrap;
  lane.imgEl = img;
  lane.canvasEl = canvas;
  lane.wrapperEl = wrapper;
  lane.audioEl = audio;
  lane.timeDisplayEl = timeDisplay;

  canvas.addEventListener("click", function (e) {
    wrap.focus();
    var rect = canvas.getBoundingClientRect();
    var xPx = e.clientX - rect.left;
    var frac = clamp(rect.width > 0 ? xPx / rect.width : 0, 0, 1);
    seekLane(lane, frac * card.meta.asset_duration_s);
  });

  wrap.addEventListener("keydown", function (e) {
    if (e.key === " " || e.key === "Spacebar") {
      e.preventDefault();
      toggleLanePlay(lane);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      seekLane(lane, lane.audioEl.currentTime - (e.shiftKey ? ARROW_STEP_FINE_S : ARROW_STEP_S));
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      seekLane(lane, lane.audioEl.currentTime + (e.shiftKey ? ARROW_STEP_FINE_S : ARROW_STEP_S));
    }
  });

  audio.addEventListener("play", function () { card.activeLaneKey = lane.key; renderLane(lane); });
  audio.addEventListener("pause", function () { renderLane(lane); });
  audio.addEventListener("seeked", function () { renderLane(lane); });
  img.addEventListener("load", function () {
    lane.naturalWidth = img.naturalWidth || flatWidthPx(card.meta.asset_duration_s);
    lane.naturalHeight = img.naturalHeight || FLAT_HEIGHT_PX;
    var w = lane.naturalWidth, h = lane.naturalHeight;
    wrapper.style.width = w + "px";
    wrapper.style.height = h + "px";
    canvas.width = w;
    canvas.height = h;
    drawLaneOverlay(lane);
  });
  img.src = flatPngPath;

  LANES.push(lane);
  return lane;
}

function toggleLanePlay(lane) {
  if (lane.audioEl.paused) { lane.audioEl.play().catch(function () {}); }
  else { lane.audioEl.pause(); }
}

function seekLane(lane, t) {
  var dur = isFinite(lane.audioEl.duration) && lane.audioEl.duration > 0
    ? lane.audioEl.duration : lane.card.meta.asset_duration_s;
  lane.audioEl.currentTime = clamp(t, 0, dur);
  lane.card.activeLaneKey = lane.key;
  renderLane(lane);
}

function renderLane(lane) {
  drawLaneOverlay(lane);
  updateLaneTimeDisplay(lane);
  if (lane.card.activeLaneKey === lane.key) {
    renderScadaBlock(lane.card);
  }
}

function drawLaneOverlay(lane) {
  var ctx = lane.canvasEl.getContext("2d");
  var w = lane.canvasEl.width;
  var h = lane.canvasEl.height;
  ctx.clearRect(0, 0, w, h);
  var dur = lane.card.meta.asset_duration_s;
  if (!dur || dur <= 0) return;

  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.font = "11px monospace";
  ctx.lineWidth = 1;
  for (var t = 0; t <= dur + 1e-6; t += 5) {
    var x = (t / dur) * w;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, h);
    ctx.stroke();
    ctx.fillText(Math.round(t) + "s", x + 3, 12);
  }

  var t2 = lane.audioEl.currentTime || 0;
  if (t2 >= 0 && t2 <= dur) {
    var xph = (t2 / dur) * w;
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xph, 0);
    ctx.lineTo(xph, h);
    ctx.stroke();
  }
}

function updateLaneTimeDisplay(lane) {
  var t = lane.audioEl.currentTime || 0;
  var total = lane.card.meta.asset_duration_s;
  lane.timeDisplayEl.textContent = formatClock(t) + " / " + formatClock(total);
}

// ---------------------------------------------------------------------------
// SCADA context panel: strip+ribbon image, shared playhead, live P/n readout.
// The panel is ONE per card (not per mic lane, unlike the spectrogram lanes
// above) -- its playhead/readout follow whichever lane is "active"
// (card.activeLaneKey, set on that lane's own play/seek, defaulted to the
// first lane at card-build time), since SCADA context does not depend on
// which microphone is playing.
// ---------------------------------------------------------------------------

function buildScadaBlock(card, meta) {
  var scada = { card: card, naturalWidth: 0, naturalHeight: 0 };

  var wrap = el("div", { class: "scada-block" });
  wrap.appendChild(
    el("div", { class: "scada-title", text: "Operating data (SCADA context)" })
  );

  var scroll = el("div", { class: "scada-scroll" });
  var wrapper = el("div", { class: "scada-wrapper" });
  var img = el("img", { class: "scada-img", alt: "SCADA strip and state ribbon" });
  var canvas = el("canvas", { class: "scada-overlay-canvas" });
  wrapper.appendChild(img);
  wrapper.appendChild(canvas);
  scroll.appendChild(wrapper);
  wrap.appendChild(scroll);

  var readout = el("div", { class: "scada-readout" });
  wrap.appendChild(readout);

  scada.el = wrap;
  scada.imgEl = img;
  scada.canvasEl = canvas;
  scada.wrapperEl = wrapper;
  scada.readoutEl = readout;

  img.addEventListener("load", function () {
    scada.naturalWidth = img.naturalWidth || flatWidthPx(meta.asset_duration_s);
    scada.naturalHeight = img.naturalHeight || SCADA_HEIGHT_PX;
    var w = scada.naturalWidth, h = scada.naturalHeight;
    wrapper.style.width = w + "px";
    wrapper.style.height = h + "px";
    canvas.width = w;
    canvas.height = h;
    renderScadaBlock(card);
  });
  img.src = meta.scada_png;

  return scada;
}

function drawScadaOverlay(scada, t) {
  var ctx = scada.canvasEl.getContext("2d");
  var w = scada.canvasEl.width;
  var h = scada.canvasEl.height;
  ctx.clearRect(0, 0, w, h);
  var dur = scada.card.meta.asset_duration_s;
  if (!dur || dur <= 0 || !w) return;

  var x = clamp((t / dur) * w, 0, w);
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, h);
  ctx.stroke();
}

function formatReading(value, unit, digits) {
  return (value === null || value === undefined) ? "–" : value.toFixed(digits) + " " + unit;
}

function updateScadaReadout(scada, meta, t) {
  var total = meta.asset_duration_s;
  var secIdx = Math.floor(clamp(t, 0, Math.max(total - 1e-6, 0)));
  var powerSeries = meta.power_mw_1hz || [];
  var speedSeries = meta.speed_rpm_1hz || [];
  var power = secIdx < powerSeries.length ? powerSeries[secIdx] : null;
  var speed = secIdx < speedSeries.length ? speedSeries[secIdx] : null;
  scada.readoutEl.textContent =
    formatClock(t) + " / " + formatClock(total) +
    "  ·  P: " + formatReading(power, "MW", 1) +
    "  ·  n: " + formatReading(speed, "rpm", 0);
}

function renderScadaBlock(card) {
  var scada = card.scada;
  if (!scada) return;
  var laneKey = card.activeLaneKey || (card.lanes[0] && card.lanes[0].key);
  var lane = card.lanes.filter(function (l) { return l.key === laneKey; })[0];
  var t = lane ? (lane.audioEl.currentTime || 0) : 0;
  drawScadaOverlay(scada, t);
  updateScadaReadout(scada, card.meta, t);
}

function tick() {
  LANES.forEach(function (lane) {
    if (!lane.audioEl.paused && !lane.audioEl.ended) {
      renderLane(lane);
    }
  });
  requestAnimationFrame(tick);
}

function buildCard(meta) {
  var restoredState = loadState(meta.candidate_id);
  var hasRestoredData = !!restoredState.assessment || !!restoredState.note;
  var initialStatus = hasRestoredData ? "restored" : "unsaved";
  var card = { meta: meta, state: restoredState, saveStatus: initialStatus };

  var head = el("div", { class: "card-head" });
  var titleSpan = el("span", { class: "card-title" });
  titleSpan.textContent = meta.candidate_id + " — " + meta.start_utc;
  var classBadge = el("span", { class: "class-badge " + meta.class, text: meta.class });
  head.appendChild(titleSpan);
  head.appendChild(classBadge);
  if (meta.near_transition) {
    head.appendChild(el("span", { class: "near-transition-badge", text: "near transition" }));
  }
  if (meta.in_sample) {
    head.appendChild(el("span", { class: "in-sample-badge", text: "in-sample (fit-pool day)" }));
  }
  card.el = el("div", { class: "candidate-card" }, [head]);

  var modeRow = el("div", { class: "mode-row" });
  var scadaModeSpan = el("span", { class: "mode-scada" });
  scadaModeSpan.innerHTML = "Mode (SCADA): " + meta.scada_state;
  modeRow.appendChild(scadaModeSpan);
  var detectorModeSpan = el("span", { class: "mode-detector" });
  detectorModeSpan.innerHTML = "Mode (Detector): " + meta.state_name;
  modeRow.appendChild(detectorModeSpan);
  if (meta.mode_mismatch) {
    modeRow.appendChild(
      el("span", { class: "mismatch-badge", text: "⚠ SCADA/detector disagree" })
    );
  }
  if (meta.scada_transition) {
    modeRow.appendChild(
      el("span", { class: "scada-transition-badge", text: "SCADA: transition/ramp" })
    );
  }
  card.el.appendChild(modeRow);

  var metaLine = el("div", { class: "meta-line" });
  metaLine.innerHTML =
    "<b>Duration:</b> " + meta.duration_s.toFixed(1) + "s &middot; " +
    "<b>min p:</b> " + meta.min_p.toExponential(3) + " &middot; " +
    "<b>Asset window:</b> " + meta.asset_duration_s.toFixed(1) + "s starting " +
    meta.asset_start_utc;
  card.el.appendChild(metaLine);

  var criterionLine = el("div", { class: "criterion-line" });
  criterionLine.textContent = meta.criterion_text;
  card.el.appendChild(criterionLine);

  var lanes = el("div", { class: "lanes" });
  var genLane = buildLane(card, "gen", meta.gen_wav, meta.gen_flat_png, "Generator microphone");
  var turLane = buildLane(card, "tur", meta.tur_wav, meta.tur_flat_png, "Turbine microphone");
  lanes.appendChild(genLane.el);
  lanes.appendChild(turLane.el);
  card.el.appendChild(lanes);
  card.lanes = [genLane, turLane];
  card.activeLaneKey = genLane.key;

  card.scada = buildScadaBlock(card, meta);
  card.el.appendChild(card.scada.el);

  var assessmentRow = el("div", { class: "assessment-row" });
  var options = el("div", { class: "assessment-options" });
  var radioName = "assessment-" + meta.candidate_id;
  var radios = [];
  ASSESSMENT_VALUES.forEach(function (value) {
    var id = radioName + "-" + value.replace(/[^a-z0-9]+/gi, "-");
    var radio = el("input", { type: "radio", name: radioName, value: value, id: id });
    if (restoredState.assessment === value) radio.checked = true;
    radio.addEventListener("change", function () {
      card.state.assessment = value;
      saveState(card);
    });
    radios.push(radio);
    var label = el("label", { for: id });
    label.appendChild(radio);
    label.appendChild(document.createTextNode(value));
    options.appendChild(label);
  });
  assessmentRow.appendChild(options);

  var noteLabel = el("label", { class: "note-label", text: "Note" });
  var noteInput = el("input", { type: "text", class: "note-input", placeholder: "free text..." });
  noteInput.value = restoredState.note;
  noteInput.addEventListener("input", function () {
    card.state.note = noteInput.value;
    saveState(card);
  });
  noteLabel.appendChild(noteInput);
  assessmentRow.appendChild(noteLabel);
  card.el.appendChild(assessmentRow);

  var saveIndicator = el("span", { class: "save-indicator" });
  card.el.appendChild(el("div", { class: "card-footer" }, [saveIndicator]));
  card.saveIndicatorEl = saveIndicator;
  card.radios = radios;
  card.noteInput = noteInput;

  updateSaveIndicator(card);
  return card;
}

function updateSaveIndicator(card) {
  card.saveIndicatorEl.classList.remove("save-error");
  if (card.saveStatus === "saved") {
    card.saveIndicatorEl.textContent = "saved " + nowClock();
  } else if (card.saveStatus === "restored") {
    card.saveIndicatorEl.textContent = "restored (previously saved)";
  } else if (card.saveStatus === "error") {
    card.saveIndicatorEl.textContent = "save failed (localStorage unavailable)";
    card.saveIndicatorEl.classList.add("save-error");
  } else {
    card.saveIndicatorEl.textContent = "";
  }
}

function cardsToCsv(cards) {
  var lines = [EXPORT_CSV_HEADER.join(",")];
  cards.forEach(function (card) {
    var m = card.meta;
    var row = [
      m.session, m.candidate_id, m.class, m.start_utc, m.duration_s, m.min_p, m.state_name,
      m.near_transition, m.scada_state, m.scada_transition,
      card.state.assessment || "", card.state.note || "",
    ];
    lines.push(row.map(csvField).join(","));
  });
  return lines.join("\r\n") + "\r\n";
}

function exportCsv(sessionKeyOrNull) {
  var cards;
  if (sessionKeyOrNull) {
    cards = CARDS_BY_SESSION[sessionKeyOrNull] || [];
  } else {
    cards = Object.keys(CARDS).map(function (k) { return CARDS[k]; });
  }
  var text = cardsToCsv(cards);
  var filename = sessionKeyOrNull
    ? "candidate_assessments_" + sessionKeyOrNull + ".csv"
    : "candidate_assessments_all.csv";
  downloadTextFile(filename, text);
  return text;
}

function buildApp() {
  var app = document.getElementById("app");
  app.innerHTML = "";

  var bySession = {};
  var sessionOrder = [];
  CANDIDATES_META.forEach(function (meta) {
    if (!bySession[meta.session]) { bySession[meta.session] = []; sessionOrder.push(meta.session); }
    bySession[meta.session].push(meta);
  });
  sessionOrder.sort();

  sessionOrder.forEach(function (sessionKey) {
    var metas = bySession[sessionKey].slice().sort(function (a, b) { return a.rank - b.rank; });
    var section = el("section", { class: "session-section", id: "session-" + sessionKey });
    var label = SESSION_LABELS[sessionKey] || sessionKey;
    section.appendChild(el("h2", { text: "Session " + sessionKey + " – " + label }));

    var toolbar = el("div", { class: "session-toolbar" });
    var exportBtn = el("button", { type: "button", text: "Export CSV" });
    var status = el("span", { class: "session-status" });
    toolbar.appendChild(exportBtn);
    toolbar.appendChild(status);
    section.appendChild(toolbar);

    exportBtn.addEventListener("click", function () {
      exportCsv(sessionKey);
      status.textContent = "exported " + nowClock();
    });

    CARDS_BY_SESSION[sessionKey] = [];
    metas.forEach(function (meta) {
      var card = buildCard(meta);
      CARDS[meta.candidate_id] = card;
      CARDS_BY_SESSION[sessionKey].push(card);
      section.appendChild(card.el);
    });

    app.appendChild(section);
  });
}

buildApp();
requestAnimationFrame(tick);

var exportAllBtn = document.getElementById("export-all-btn");
if (exportAllBtn) {
  exportAllBtn.addEventListener("click", function () {
    exportCsv(null);
    document.getElementById("export-all-status").textContent = "exported " + nowClock();
  });
}

window.CandidateKit = {
  getState: function (candidateId) {
    var card = CARDS[candidateId];
    return card ? { assessment: card.state.assessment, note: card.state.note } : null;
  },
  setAssessment: function (candidateId, value) {
    var card = CARDS[candidateId];
    if (!card) return null;
    card.state.assessment = value;
    card.radios.forEach(function (r) { r.checked = r.value === value; });
    saveState(card);
    return card.state.assessment;
  },
  clickLaneAtFraction: function (candidateId, laneKey, fraction) {
    var card = CARDS[candidateId];
    if (!card) return null;
    var lane = card.lanes.filter(function (l) { return l.key === laneKey; })[0];
    if (!lane) return null;
    var rect = lane.canvasEl.getBoundingClientRect();
    var evt = new MouseEvent("click", {
      clientX: rect.left + fraction * rect.width, clientY: rect.top + rect.height / 2,
      bubbles: true, cancelable: true,
    });
    lane.canvasEl.dispatchEvent(evt);
    return lane.audioEl.currentTime;
  },
  getLanePlayheadPx: function (candidateId, laneKey) {
    var card = CARDS[candidateId];
    if (!card) return null;
    var lane = card.lanes.filter(function (l) { return l.key === laneKey; })[0];
    if (!lane) return null;
    var t = lane.audioEl.currentTime || 0;
    return (t / card.meta.asset_duration_s) * lane.canvasEl.width;
  },
  exportCsv: exportCsv,
  cardKey: function (candidateId) { return candidateId; },
  cards: CARDS,
};
})();
"""


def render_index_html(results: Sequence[CandidateAssetResult], out_dir: Path) -> Path:
    """(Re)generate the interactive `<out_dir>/index.html`. Fully self-contained
    (inline CSS/JS, no external resources, no `fetch()`): the candidate metadata is
    embedded as a `<script type="application/json">` block and read via
    `JSON.parse` at load time -- same `file://` CORS rationale as `annotation_kit.
    render_interactive_index_html`'s own docstring. Every card's spectrogram image/
    audio element references the real PNG/WAV by a RELATIVE path (never base64)."""
    metas = [_asset_result_to_meta(r) for r in results]
    sessions = sorted({str(m["session"]) for m in metas})
    labels = {s: _SESSION_LABEL.get(s, s) for s in sessions}
    metas_json = ak._json_script_safe(metas)
    labels_json = ak._json_script_safe(labels)

    legend_html = _CANDIDATE_LEGEND_HTML.replace(
        "__STATE_LEGEND_HTML__", _render_state_legend_html()
    )

    # Plain token .replace() rather than `%`-style formatting: the JS body itself
    # contains literal `%` (the modulo operator, `t % 60`), which `%`-formatting
    # would misparse as a conversion spec -- tokens sidestep that whole class of
    # bug regardless of what the JS body ever grows to contain.
    js = (
        _CANDIDATE_JS
        .replace("__FLAT_PX_PER_S__", repr(ak._FLAT_PX_PER_S))
        .replace("__FLAT_HEIGHT_PX__", repr(ak._FLAT_HEIGHT_PX))
        .replace("__SCADA_HEIGHT_PX__", repr(_SCADA_TOTAL_HEIGHT_PX))
        .replace("__ASSESSMENT_VALUES_JSON__", ak._json_script_safe(list(ASSESSMENT_VALUES)))
    )

    doc = (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        "<title>Candidate Review Kit</title>\n"
        f"<style>{_CANDIDATE_CSS}</style>\n</head>\n<body>\n"
        "<h1>Candidate Review Kit – unverified anomaly candidates</h1>\n"
        f"{legend_html}\n"
        f"{_CANDIDATE_INSTRUCTIONS_HTML}\n"
        '<div class="top-toolbar"><button type="button" id="export-all-btn">Export all '
        'sessions</button><span class="session-status" id="export-all-status"></span></div>\n'
        '<div id="app">Loading candidates&hellip;</div>\n'
        f'<script id="candidates-meta-data" type="application/json">{metas_json}</script>\n'
        f'<script id="session-labels-data" type="application/json">{labels_json}</script>\n'
        f"<script>{js}</script>\n"
        "</body>\n</html>\n"
    )
    out_path = out_dir / "index.html"
    out_path.write_text(doc, encoding="utf-8")
    logger.info("candidate_kit: wrote %s (%d candidate(s), interactive)", out_path, len(metas))
    return out_path


# ---------------------------------------------------------------------------
# compile: assessments -> docs/assessments/
# ---------------------------------------------------------------------------

_ASSESSMENTS_CSV_COLUMNS = (
    "session", "candidate_id", "class", "start_utc", "duration_s", "min_p",
    "state_name", "near_transition", "scada_state", "scada_transition", "assessment", "note",
)


def read_assessments_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"assessments CSV not found: {path}")
    return pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)


def validate_and_merge_assessments(
    assessments_df: pd.DataFrame, candidates_by_id: Mapping[str, Candidate]
) -> list[dict[str, object]]:
    """Validate every row of an exported assessments CSV against *candidates_by_id*
    (`candidate_id` must be known; a non-empty `assessment` must be one of
    `ASSESSMENT_VALUES`) and RAISE on the first violation before returning
    anything -- mirrors `annotation_kit.compile_template`'s all-validate-then-write
    contract (a bad row anywhere must fail the whole call). Rows with an EMPTY
    assessment (not yet reviewed) are silently skipped from the output (mirrors
    `compile_row`'s empty-offsets skip) -- they carry no reviewed information.
    Every OUTPUT column except `assessment`/`note` is re-derived from the trusted
    *candidates_by_id* entry, never trusted from the export (mirrors `compile_
    marks`' "strike_no/kind are for human readability only" precedent).

    Every re-derived column now includes `scada_state`/`scada_transition` (added
    alongside `state_name`/`near_transition` -- module docstring's `compile`
    section) so the tracked, provenance-stamped review file carries the same
    SCADA-vs-detector context a reviewer saw in `index.html`, not just the
    detector's own (possibly wrong) `state_name`.

    Raises:
        ValueError: if a required column is missing, a row's `candidate_id` has no
            match in *candidates_by_id*, or a non-empty `assessment` is not in
            `ASSESSMENT_VALUES`.
    """
    required = ("candidate_id", "assessment", "note")
    missing = [c for c in required if c not in assessments_df.columns]
    if missing:
        raise ValueError(f"assessments CSV is missing column(s): {', '.join(missing)}")

    records = assessments_df.to_dict(orient="records")
    rows = [{str(k): ak._cell_to_str(v) for k, v in r.items()} for r in records]

    for row in rows:
        cid = row["candidate_id"]
        if cid not in candidates_by_id:
            raise ValueError(f"unknown candidate_id {cid!r} (not in the reference candidates CSV)")
        assessment = row["assessment"].strip()
        if assessment and assessment not in ASSESSMENT_VALUES:
            raise ValueError(
                f"candidate {cid!r}: assessment {assessment!r} is not one of "
                f"{list(ASSESSMENT_VALUES)}"
            )

    out: list[dict[str, object]] = []
    for row in rows:
        assessment = row["assessment"].strip()
        if not assessment:
            continue
        c = candidates_by_id[row["candidate_id"]]
        out.append(
            {
                "session": c.session, "candidate_id": c.candidate_id, "class": c.klass,
                "start_utc": c.start_utc.isoformat(), "duration_s": round(c.duration_s, 3),
                "min_p": c.min_p, "state_name": c.state_name, "near_transition": c.near_transition,
                "scada_state": c.scada_state, "scada_transition": c.scada_transition,
                "assessment": assessment, "note": row["note"],
            }
        )
    return out


def write_assessments_csv(
    rows: Sequence[Mapping[str, object]], out_path: Path, *, source_path: Path, compiled_date: date
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comment = (
        f"# {out_path.name} -- candidate-review assessments: manual listening review by the "
        "author against scripts/candidate_kit.py build's index.html; register explicitly not "
        "ground truth.\n"
        "# Provenance: this register is explicitly not ground truth (Jinyuan ruling, "
        "2026-08-04) -- a candidate is a model-flagged window on a day WITHOUT induced "
        "anomalies; 'assessment' records the author's own qualitative listening judgement, "
        "nothing more.\n"
        "# Handover scope: this file is the author's OWN qualitative listening pass (a "
        "thesis artifact / audit trail) -- it is NOT part of the expert handover deliverable. "
        "The handover product is scripts/candidate_kit.py build's review tool "
        "(results/candidate-kit/index.html) with every assessment left EMPTY for the plant "
        "experts themselves to fill in (Stefan's decision 2026-08-16).\n"
        f"# Source: {source_path}\n"
        f"# Compiled: {compiled_date.isoformat()} (scripts/candidate_kit.py compile, --date kept "
        "explicit for reproducibility).\n"
    )
    frame = pd.DataFrame(list(rows), columns=list(_ASSESSMENTS_CSV_COLUMNS))
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(comment)
        frame.to_csv(fh, index=False)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Candidate-review kit: select/build/compile a listening-review deliverable for "
            "model-flagged windows on days WITHOUT induced anomalies (register explicitly not "
            "ground truth)."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    select_p = sub.add_parser(
        "select",
        help="Build results/candidate-kit/candidates.csv from the once-calibrated "
             "monitors plus results/monitor-ext/ for the coverage-extension sessions "
             "(_MONITOR_EXT_SESSIONS).",
    )
    select_p.add_argument(
        "--out", type=Path, default=DEFAULT_CANDIDATES_CSV, help="Output CSV path."
    )
    select_p.add_argument(
        "--cap", type=int, default=CANDIDATE_CAP_PER_SESSION,
        help="Max candidates kept per session."
    )

    build_p = sub.add_parser("build", help="Render WAV/PNG assets + the interactive index.html.")
    build_p.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_CSV)
    build_p.add_argument("--out", type=Path, default=DEFAULT_KIT_DIR)

    compile_p = sub.add_parser(
        "compile", help="Validate an exported assessments CSV and write docs/assessments/."
    )
    compile_p.add_argument("--assessments", type=Path, required=True)
    compile_p.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_CSV)
    compile_p.add_argument("--out", type=Path, required=True)
    compile_p.add_argument("--date", type=date.fromisoformat, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "select":
        cfg = load_config()
        try:
            results = select_all(cfg, args.out, cap=args.cap)
        except ValueError as exc:
            print(f"candidate_kit: {exc}", file=sys.stderr)
            return 2
        total_kept = sum(len(r.kept) for r in results)
        print(f"candidate_kit: select wrote {total_kept} candidate(s) -> {args.out}")
        return 0

    if args.command == "build":
        if not args.candidates.is_file():
            print(f"candidate_kit: candidates CSV not found: {args.candidates}", file=sys.stderr)
            return 2
        cfg = load_config()
        try:
            build_results = build_all(cfg, args.candidates, args.out)
        except ValueError as exc:
            print(f"candidate_kit: {exc}", file=sys.stderr)
            return 2
        print(
            f"candidate_kit: build wrote {len(build_results)} candidate(s) assets, index.html, "
            f"candidates_meta.json -> {args.out}"
        )
        return 0

    assert args.command == "compile"
    if not args.assessments.is_file():
        print(f"candidate_kit: assessments CSV not found: {args.assessments}", file=sys.stderr)
        return 2
    if not args.candidates.is_file():
        print(f"candidate_kit: candidates CSV not found: {args.candidates}", file=sys.stderr)
        return 2
    try:
        candidates_by_id = {c.candidate_id: c for c in candidates_from_csv(args.candidates)}
        assessments_df = read_assessments_csv(args.assessments)
        rows = validate_and_merge_assessments(assessments_df, candidates_by_id)
    except ValueError as exc:
        print(f"candidate_kit: {exc}", file=sys.stderr)
        return 2
    write_assessments_csv(rows, args.out, source_path=args.assessments, compiled_date=args.date)
    print(
        f"candidate_kit: compile wrote {len(rows)} assessed candidate(s) from "
        f"{len(assessments_df)} exported row(s) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
