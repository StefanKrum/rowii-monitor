"""Runtime monitor CLI: apply a persisted `MonitorSnapshot` (fitted detector +
per-state references + conformal thresholds, `rowii.runtime.snapshot`) to a
NEW recording, emitting a state timeline plus per-window conformal alarm
verdicts -- the "runs at the plant" requirement as a batch CLI over recorded
files (the chosen deployment model: no streaming, no retraining, no refit of
anything).

Pipeline: `prepare_run` (feature cache honored unless `--no-cache`) -> geometry
guard (the snapshot's `feature_names` are the SCORING CONTRACT: the prepared run
must contain every snapshot column -- else exit 2 naming the variant, both
widths, and the missing columns -- and is projected, by name, onto exactly those
columns in snapshot order; extra prepared columns, e.g. a channel dead at fit
time but live on the monitored day (the 080726 TurbineVib-ch0 case), are dropped
with a WARNING naming each one -- the trained model never saw them, so they must
never influence scoring; a snapshot fitted on `fusion` must likewise never
silently score some other variant's features) -> `to_detector(snapshot).apply` on
the valid windows (fit-day standardization + fit-day HMM Viterbi decode, scattered
back to the full grid with `-1` on invalid windows) -> per-state scoring against
the snapshot's OWN fit-day references, thresholded in one of two modes:

- `--thresholds recalibrate` (DEFAULT): split the new run's valid windows by
  segments at the snapshot's own `(calibration_frac, seed)`, recompute each
  snapshot-known state's conformal threshold on the calibration-side windows of
  that state at `--alpha` (default: the snapshot's fit-time alpha), and emit
  verdicts for the SCORING-side windows only. This operationalizes the
  central cross-day finding: "transfer detector + references, recalibrate
  thresholds per day" is the only recipe whose realized false-alarm rate held its
  nominal alpha. References are NEVER refit here -- recalibrate mode touches
  thresholds only. The calibration-side windows themselves are consumed
  by the threshold fit and are NEVER alarmed (`role="consumed_for_calibration"` --
  the calibration-bias rule: alarming the very windows a threshold was fitted
  on would break the conformal exchangeability argument in the anti-conservative
  direction).
- `--thresholds frozen`: apply the fit day's stored thresholds unchanged to every
  valid window of a snapshot-known state. Reported with an explicit
  distribution-shift warning in the notes: in the cross-day evidence,
  frozen cross-day thresholds did NOT hold their nominal FAR.
- `--thresholds rolling --rolling-minutes M` (default M=60): the SAME roles/split
  as recalibrate mode -- rolling
  replaces HOW each scored window's threshold is derived, not WHICH windows get
  verdicts. Per (state, scored window): the threshold is `calibrate(...)` over the
  scores of the SAME state's calibration-side ("consumed") windows whose window
  START lies within `[t_w - M minutes, t_w)`, whenever that trailing count reaches
  the conformal floor `ceil(1/alpha) - 1`; below the floor the window FALLS BACK to
  the snapshot's stored fit-day frozen threshold. p-values follow the same branch
  (against the trailing set when rolling, against the fit day's stored calibration
  scores when falling back). Every scored row is flagged per window in the
  `threshold_source` column (`rolling` / `fit_day_fallback`) -- never a SILENT
  fallback (the recalibrate-mode rule forbids silent frozen fallbacks; here
  the fallback is the designed, per-window-flagged behavior, and per-state
  trailing-coverage statistics are a MANDATORY notes output). Consumed
  windows' scores stay recorded in alarms.parquet, so every rolling threshold is
  auditable from the artifact alone. Explicit limitation: a slowly developing
  fault inflates the trailing calibration scores and can be ABSORBED by a rolling
  threshold -- the notes carry the double-reference honesty note (read rolling
  verdicts side by side with fit-day-referenced verdicts). This is a batch
  ablation over recorded files, not a production stream.

Per-state semantics on the new run (binding): a valid window's detected
state can be absent from the snapshot (no reference/threshold survived fitting) --
such windows get `role="unknown_state"`, are never alarmed, and are counted in the
notes. In recalibrate mode a snapshot-known state with ZERO calibration-side
windows on the new run cannot be recalibrated -- ALL its windows get
`role="no_conformal_data"`, no verdicts, and a notes row (never a silent fallback
to the frozen fit-day threshold, which would smuggle exactly the un-recalibrated
behavior the mode exists to avoid).

`--session-norm`: label-free
per-session robust normalization of the SCORING space. The snapshot must carry
session stats (format v2, `rowii.anomaly.normalize.SessionStats`) -- a v1 file or
any snapshot fitted without stats is REFUSED with exit 2 (never a silent raw-space
fallback). When active: the DETECTOR still consumes RAW features (labels are
norm-invariant by construction); the scoring path
transforms (a) the snapshot's stored RAW references with the SNAPSHOT's stats
(`scorer_for_label(..., session_stats=...)` -- pool-global stats for a pooled
snapshot, `norm_minutes == 0.0` sentinel) and (b) the monitored run's features
with the MONITORED run's OWN first-N stats (N = the snapshot's stored
`norm_minutes`; the pool-global sentinel falls back to the default of
`_DEFAULT_NORM_MINUTES`), so scores, thresholds and p-values all live in the
session-normalized space. Roles and alarm logic are otherwise unchanged.
Comparability with raw-space numbers is FAR-level only. The reverse
mismatch -- a stats-bearing snapshot run WITHOUT `--session-norm` -- only logs a
WARNING: the stored references are raw, so recalibrate mode stays fully coherent;
frozen mode would compare raw-space scores against the snapshot's
normalized-space thresholds, which the warning names.

`--level-recal`: the
shape-preserving, feature-native counterpart to `--session-norm` -- an additive
offset in the log10 domain (`rowii.anomaly.levelrecal`), applied ONLY to the
monitored run's LEVEL columns (`*_log_rms`/`*_band_*`/`*_octave_*`); shape
columns (`*_spectral_centroid`/`*_rolloff95`/`*_kurtosis`) are untouched.
Mutually exclusive with `--session-norm` (fit-path exclusivity, exit 2).
The snapshot must carry `level_recal_medians` (format v2, the pooled FIT side's
own per-column median at `run_step2 --save-snapshot --level-recal` fit time)
-- a snapshot without them is REFUSED with exit 2 (never a silent
raw-space fallback, mirroring `--session-norm`'s stats-less refusal). Unlike
`--session-norm`, this applies AFTER the snapshot-contract geometry projection:
the run-side median is computed over the monitored run's own
label-free first-`_DEFAULT_NORM_MINUTES` (20) minutes of valid windows,
name-keyed against the PROJECTED `feature_names` (mirrors `scripts/
run_step2.py`'s `_first_n_minutes_rows`, duplicated locally -- the
script-sibling rule). The DETECTOR still consumes RAW features (labels are
norm-invariant, the same boundary `--session-norm` uses); references stay
the snapshot's RAW references -- only the monitored run's scoring features
shift. A zero-level-column snapshot feature contract (an embedding variant)
raises inside `rowii.anomaly.levelrecal` and is caught here as exit 2.

Named states end to end and transition visibility: the snapshot's OPTIONAL
commissioning-time `state_names` member
(`dict[int, str]`, `rowii.eval.metrics.derive_state_names`, `rowii.runtime.snapshot`)
surfaces as `state_name` on EVERY `alarms.parquet` row (ALWAYS present -- fallback
`cluster-<id>` when the snapshot carries no names) and as an OPTIONAL
`mapped_mode` column on `segments.csv` (present only when the snapshot carries names
-- inventing a mapping the artifact cannot back would be a claim, not a fact).
`timeline.md` and `monitor_notes.md`'s per-state table name states the same way when
a name exists. `near_transition` (`alarms.parquet`, ALWAYS present) is True for
every VALID window within ≈W seconds, measured along the VALID subsequence (not
wall-clock -- an invalid gap contributes no distance), of a detected-state CHANGE
found in that same subsequence (an invalid gap never counts as a change itself
either); `W` defaults to the snapshot's own `min_dwell_s` -- the same dwell scale
the `min_dwell` sweep grounds. The OPTIONAL `--suppress-transition-alarms`
flag (default OFF) then withholds alarms on near-transition SCORED windows
(`alarm -> False`); `score`/`p_value`/`near_transition`/`role` stay recorded, so
every suppressed alarm remains fully auditable from `alarms.parquet` alone, and the
suppressed count is reported in `monitor_notes.md`. Suppression is an explicit
OPERATOR choice this package makes visible, never a default behavior.

Outputs under `--out` (default `results/monitor/<run>/`): `segments.csv` (+ an
OPTIONAL `mapped_mode` column) + `timeline.md` (the state half,
`scripts/apply_detector.py`'s conventions incl. scatter-back over `valid_mask`),
`alarms.parquet` (one row per VALID window: `window, t_utc_ns, state, score,
p_value, alarm, low_confidence, role, threshold_source, near_transition,
state_name` -- `threshold_source` is per-window `rolling`/`fit_day_fallback` on
rolling-mode scored rows, `""` on rolling-mode rows without a verdict, and the
constant mode name in the other modes, so the column exists uniformly;
`near_transition`/`state_name` are ALWAYS present regardless of mode),
`alarm_segments.csv` (maximal alarm runs: `start_utc, end_utc, duration_s`
-- reflects `--suppress-transition-alarms` when active, since it forces `alarm`
values BEFORE this table is built), and `monitor_notes.md` (snapshot provenance,
mode, per-state table, window accounting incl. the suppressed-alarm count when
suppression ran, and the standing honesty framing: NO fault labels exist -- alarms
are candidates for operator review, never verified detections).

Bootstrapping (config/env via `rowii.config.load_config`, run discovery via
`rowii.io.dataset.discover`, unknown-run exit 2 listing every discovered run)
mirrors `scripts/warm_cache.py`'s skeleton. Small helpers shared with sibling
scripts are DUPLICATED, not imported -- a script must not depend on a SIBLING
script's internals (`scripts/apply_detector.py`'s module docstring states the
rule); `src/rowii/` modules are imported normally.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.anomaly.conformal import calibrate, p_values, threshold_index  # noqa: E402
from rowii.anomaly.levelrecal import (  # noqa: E402
    apply_level_recal,
    column_medians,
    level_recal_offsets,
)
from rowii.anomaly.normalize import (  # noqa: E402
    SessionStats,
    apply_session_norm,
    fit_session_stats,
)
from rowii.anomaly.overlap import to_utc_ns  # noqa: E402
from rowii.anomaly.references import split_by_segments  # noqa: E402
from rowii.config import load_config  # noqa: E402
from rowii.io.dataset import Run, discover  # noqa: E402
from rowii.pipeline import PreparedRun, prepare_run  # noqa: E402
from rowii.runtime.snapshot import (  # noqa: E402
    MonitorSnapshot,
    load_snapshot,
    scorer_for_label,
    to_detector,
)
from rowii.signals.windows import WindowGrid  # noqa: E402
from rowii.state.detect import FittedDetector  # noqa: E402
from rowii.state.segments import to_segments  # noqa: E402

logger = logging.getLogger(__name__)

_INVALID_LABEL = -1
"""Sentinel for invalid windows in a full-length label array -- mirrors `scripts/
run_step2.py`'s/`scripts/apply_detector.py`'s own `_INVALID_LABEL` (duplicated, not
imported -- module docstring)."""

_THRESHOLD_MODES: tuple[str, ...] = ("recalibrate", "frozen", "rolling")

_DEFAULT_ROLLING_MINUTES = 60.0
"""`--thresholds rolling`'s default trailing-window length M in minutes -- the
binding default (its motivating probe on 290626-tu: at M=20 only 46.8% of scored
windows reach the conformal floor, states 2/3 at 0%; at M=60 still 53.8%)."""
_MAX_ROLLING_MINUTES = 5_256_000.0
"""Upper guard for `--rolling-minutes` (~10 years): far beyond any real use, and
values above ~1.5e8 minutes overflow the int64 nanosecond arithmetic in
`_trailing_bounds` (guarded to avoid a raw OverflowError instead of the
CLI's clean exit-2 contract)."""

_DEFAULT_NORM_MINUTES = 20.0
"""`--session-norm` fallback prefix length for the MONITORED run's own stats when
the snapshot's stored `norm_minutes` is the pool-global sentinel (0.0); a
first-N snapshot's stored value is used verbatim instead. Also `--level-recal`'s
(unconfigurable) first-N-minutes prefix length for the monitored run's own
level-column median (`_first_n_minutes_rows`) -- mirrors `scripts/
run_step2.py`'s own reuse of the identical constant/value for the same purpose."""

_DEFAULT_EXCLUDE_TOLERANCE_S = 5.0
"""`--exclude-calibration-events` default interval padding in seconds -- the same
value `scripts/eval_events.py` uses as its default matching tolerance, so the set
of windows banned from calibration is (by default) exactly the set the event
evaluation could credit as a detection."""

_ALARM_COLUMNS: tuple[str, ...] = (
    "window", "t_utc_ns", "state", "score", "p_value", "alarm", "low_confidence", "role",
    "threshold_source", "near_transition", "state_name",
)
"""alarms.parquet's exact column contract, in this order: `threshold_source` was
appended after the base columns; `near_transition` then `state_name` were
appended at the END. BOTH
new columns are ALWAYS present, never conditional on the snapshot carrying
`state_names`. Downstream readers that select columns by name
(`rowii.eval.events.evaluate_events`, `scripts/eval_events.py`) ignore the addition
by construction."""

SOURCE_ROLLING = "rolling"
"""`threshold_source` value: the window's threshold/p-value came from its own
trailing same-state calibration set (rolling mode only)."""
SOURCE_FIT_DAY_FALLBACK = "fit_day_fallback"
"""`threshold_source` value: the trailing count sat below the conformal floor, so
the snapshot's stored fit-day threshold/calibration scores were applied (rolling
mode only -- the per-window-flagged fallback)."""

ROLE_SCORED = "scored"
"""A window with a real verdict: its state's threshold was applied to its score."""
ROLE_CONSUMED = "consumed_for_calibration"
"""Recalibrate mode only: a calibration-side window whose score parameterized its
state's recalibrated threshold -- never alarmed (calibration-bias rule)."""
ROLE_UNKNOWN_STATE = "unknown_state"
"""Detected state has no reference/threshold in the snapshot -- never alarmed."""
ROLE_NO_CONFORMAL_DATA = "no_conformal_data"
"""Recalibrate mode only: the state is snapshot-known but has zero calibration-side
windows on this run, so no per-day threshold exists -- never alarmed."""

_FROZEN_SHIFT_WARNING = (
    "**Distribution-shift warning (frozen thresholds):** the fit day's conformal "
    "thresholds were applied UNCHANGED to this recording. In package-2's cross-day "
    "evidence, frozen cross-day thresholds did NOT hold their nominal false-alarm "
    "rate -- realized FAR drifted far from alpha across days -- so the alarm counts "
    "below carry NO per-day false-alarm guarantee. `--thresholds recalibrate` (the "
    "default) is the recipe whose FAR held."
)
"""The notes' verbatim frozen-mode warning (reported with an explicit
distribution-shift warning; the binding phrase is "did NOT hold")."""


def _first_n_minutes_rows(prepared: PreparedRun, norm_minutes: float) -> np.ndarray:
    """*prepared*'s own first *norm_minutes* of VALID windows, float64 -- the
    `--level-recal` run-side anchor source (the monitored run's own
    first-N-minutes windows, label-free).

    DUPLICATED from `scripts/run_step2.py::_first_n_minutes_rows` (the
    script-sibling rule, module docstring: a script must not depend on a
    SIBLING script's internals) -- mirrors `rowii.anomaly.normalize.
    fit_session_stats`'s identical window-membership rule (window START offset
    < `norm_minutes * 60s` AND `valid_mask`), duplicated rather than imported
    because level-recal needs the qualifying ROWS themselves
    (`rowii.anomaly.levelrecal.column_medians`' input), not a fitted
    `SessionStats` center/scale.

    Raises:
        ValueError: if zero windows qualify (empty prefix / all-invalid) --
            the same failure mode as `fit_session_stats`, loud rather than
            silently medianing over nothing.
    """
    n_windows = prepared.features.shape[0]
    cutoff_ns = int(round(norm_minutes * 60.0 * 1e9))
    window_offsets = np.arange(n_windows, dtype=np.int64) * np.int64(prepared.grid.window_ns)
    qualifying = (window_offsets < cutoff_ns) & prepared.valid_mask
    rows = prepared.features[qualifying]
    if rows.shape[0] == 0:
        raise ValueError(
            f"zero valid windows start within the first {norm_minutes:g} minute(s) "
            f"of the run -- cannot compute the level-recal run-side median"
        )
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a persisted MonitorSnapshot (rowii.runtime.snapshot: fitted "
            "detector + per-state references + conformal thresholds) to a NEW "
            "recording: state timeline (segments.csv + timeline.md) plus per-window "
            "conformal alarm verdicts (alarms.parquet + alarm_segments.csv + "
            "monitor_notes.md). Alarms are CANDIDATES for operator review -- no "
            "fault labels exist (module docstring)."
        )
    )
    parser.add_argument(
        "--snapshot", type=Path, required=True,
        help="Path to a MonitorSnapshot .npz written by rowii.runtime.snapshot."
             "save_snapshot (it determines the feature variant to prepare).",
    )
    parser.add_argument(
        "--run", required=True,
        help="Discovered run name to monitor (unknown names exit 2 listing every "
             "available run).",
    )
    parser.add_argument(
        "--thresholds", choices=_THRESHOLD_MODES, default="recalibrate",
        help="Threshold mode: 'recalibrate' (default) re-derives each state's "
             "conformal threshold on this run's own calibration-side windows "
             "(package-2's cross-day recipe -- the only one whose FAR held); "
             "'frozen' applies the fit-day thresholds unchanged (reported with a "
             "distribution-shift warning -- package-2 evidence: frozen cross-day "
             "thresholds did NOT hold their FAR); 'rolling' (package-7 spec "
             "D7/A3.2, batch ablation) recalibrates each scored window's threshold "
             "on the trailing --rolling-minutes of same-state calibration-side "
             "windows, falling back per window to the fit-day frozen threshold "
             "below the conformal floor ceil(1/alpha)-1 -- flagged in "
             "alarms.parquet's threshold_source column.",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Recalibrate-/rolling-mode nominal false-alarm target in (0, 1); "
             "default: the snapshot's own fit-time alpha. Ignored (with a warning) "
             "in frozen mode -- frozen thresholds are applied exactly as stored.",
    )
    parser.add_argument(
        "--rolling-minutes", type=float, default=None,
        help=f"Rolling-mode trailing window length M in minutes (default "
             f"{_DEFAULT_ROLLING_MINUTES:g}, spec A3.2). Only valid with "
             f"--thresholds rolling; must be > 0.",
    )
    parser.add_argument(
        "--session-norm", action="store_true",
        help="Score in the label-free session-normalized space (package-7 spec "
             "D3/A3.5): the snapshot's stored references are transformed with the "
             "SNAPSHOT's session stats and this run's windows with THIS run's own "
             "first-N median/MAD stats (N from the snapshot; pool-global sentinel "
             "falls back to 20 min). Requires a snapshot that stores session stats "
             "(format v2) -- a v1/no-stats snapshot is refused (exit 2). The "
             "detector always consumes RAW features; only scoring is normalized.",
    )
    parser.add_argument(
        "--level-recal", action="store_true",
        help="Score with the monitored run's LEVEL columns (*_log_rms/*_band_*/"
             "*_octave_*) additively recentred onto the snapshot's stored anchor "
             "medians (package-8 spec D2/A1.4/A1.10/A1.11); shape columns are "
             "untouched. Applies AFTER the snapshot-contract geometry projection, "
             "name-keyed against the projected feature_names. Requires a snapshot "
             "that stores level-recal medians (format v2) -- a snapshot without "
             "them is refused (exit 2). Mutually exclusive with --session-norm. "
             "The detector always consumes RAW features; only scoring shifts.",
    )
    parser.add_argument(
        "--exclude-calibration-events", type=Path, default=None,
        help="CSV of tz-aware event intervals (columns start_utc,end_utc; '#' "
             "comment lines skipped -- the docs/groundtruth contract) whose "
             "windows are BANNED from the calibration side and moved to the "
             "scoring side instead (spec A2.3.3: an induced-event day must "
             "calibrate on event-free windows -- without this, event minutes "
             "landing in calibration segments are consumed, never alarmed, and "
             "contaminate the thresholds they feed). Applies to recalibrate and "
             "rolling modes; frozen mode draws no calibration from the monitored "
             "run, so there the flag only warns.",
    )
    parser.add_argument(
        "--exclude-tolerance-s", type=float, default=None,
        help=f"Padding in seconds applied to BOTH ends of every excluded event "
             f"interval (default {_DEFAULT_EXCLUDE_TOLERANCE_S:g}, mirroring "
             f"eval_events' matching tolerance). Only valid together with "
             f"--exclude-calibration-events.",
    )
    parser.add_argument(
        "--suppress-transition-alarms", action="store_true",
        help="Withhold alarms on windows within the near-transition band (spec "
             "D3c ablation, default OFF): forces alarm=False on SCORED windows "
             "where near_transition is True (≈±min_dwell_s seconds, measured "
             "along the valid subsequence, of a detected-state change). "
             "score/p_value/near_transition/role stay recorded in "
             "alarms.parquet for a full audit trail; the suppressed count is "
             "reported in monitor_notes.md. An explicit operator choice -- this "
             "package makes the trade-off visible, it does not adopt it.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Bypass rowii.pipeline.prepare_run's on-disk feature cache and "
             "recompute features for the monitored run.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory (created if missing); default: "
             "<results_root>/monitor/<run>/.",
    )
    return parser


# ---------------------------------------------------------------------------
# Detected labels (snapshot's fit-day id space, full grid length)
# ---------------------------------------------------------------------------


def _apply_detector_labels(prepared: PreparedRun, detector: FittedDetector) -> np.ndarray:
    """Per-window labels for *prepared* in the SNAPSHOT's fit-day id space via
    `FittedDetector.apply` (fit-day standardization + fit-day HMM decode, no refit),
    `_INVALID_LABEL` on invalid windows -- duplicated from `scripts/run_step2.py`'s
    own private helper of the same name (a script must not depend on a SIBLING
    script's internals -- module docstring)."""
    valid_mask = prepared.valid_mask
    features_valid = prepared.features[valid_mask]
    n_valid = int(valid_mask.sum())
    valid_grid = WindowGrid(
        t0_ns=prepared.grid.t0_ns, window_ns=prepared.grid.window_ns, n_windows=n_valid
    )
    det = detector.apply(features_valid, valid_grid)
    full_labels = np.full(prepared.features.shape[0], _INVALID_LABEL, dtype=np.int64)
    full_labels[valid_mask] = det.frame_labels
    return full_labels


# ---------------------------------------------------------------------------
# Per-window verdicts (both threshold modes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StateRow:
    """One snapshot-known state's row of `monitor_notes.md`'s per-state table."""

    state: int
    n_windows: int
    """Valid windows detected as this state on the monitored run (all roles)."""
    n_scored: int
    n_alarms: int
    n_consumed: int
    threshold: float
    """The threshold actually applied (recalibrated or frozen); NaN when none could
    be (the `no_conformal_data` path)."""
    low_confidence: bool | None
    """The applied threshold's low-confidence flag; None when no threshold exists
    -- also None in rolling mode, where no single per-state threshold exists."""
    status: str
    """`"scored"` or `"no_conformal_data"` (per-state, not per-window)."""
    n_rolling: int = 0
    """Rolling mode only: scored windows whose threshold came from their own
    trailing calibration set (`threshold_source == "rolling"`); 0 elsewhere."""
    n_fallback: int = 0
    """Rolling mode only: scored windows below the conformal floor that fell back
    to the fit-day threshold (`threshold_source == "fit_day_fallback"`); 0
    elsewhere. `n_rolling + n_fallback == n_scored` in rolling mode."""


@dataclass(frozen=True)
class _Verdicts:
    """Full-grid per-window verdict arrays + notes bookkeeping. Invalid windows keep
    the defaults everywhere (NaN score/p_value, False alarm, empty role) and are
    excluded from alarms.parquet; `_assert_roles_complete` guarantees every VALID
    window received exactly one role."""

    score: np.ndarray
    """(W,) float64 -- NaN where no score was computed (invalid/unknown/no-data)."""
    p_value: np.ndarray
    """(W,) float64 -- NaN everywhere except scored windows (verdict-only)."""
    alarm: np.ndarray
    """(W,) bool -- True only on scored windows over their state's threshold."""
    low_confidence: np.ndarray
    """(W,) bool -- True only on scored windows whose applied threshold is
    low-confidence (threshold=+inf, can never alarm)."""
    role: np.ndarray
    """(W,) object -- one of the ROLE_* strings on valid windows, "" on invalid."""
    threshold_source: np.ndarray
    """(W,) object -- alarms.parquet's `threshold_source` column:
    per-window `SOURCE_ROLLING`/`SOURCE_FIT_DAY_FALLBACK` on rolling-mode scored
    windows ("" on rolling-mode windows without a verdict), the constant mode name
    everywhere in the other modes."""
    state_rows: list[_StateRow]
    unknown_counts: dict[int, int]
    """Detected-but-not-snapshot-known state id -> valid-window count."""


def _empty_verdict_arrays(
    n: int, threshold_source_fill: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score = np.full(n, math.nan, dtype=np.float64)
    p_value = np.full(n, math.nan, dtype=np.float64)
    alarm = np.zeros(n, dtype=bool)
    low_confidence = np.zeros(n, dtype=bool)
    role = np.full(n, "", dtype=object)
    threshold_source = np.full(n, threshold_source_fill, dtype=object)
    return score, p_value, alarm, low_confidence, role, threshold_source


def _mark_unknown_states(
    prepared: PreparedRun, snapshot: MonitorSnapshot, labels: np.ndarray, role: np.ndarray
) -> dict[int, int]:
    """Tag every valid window whose detected state has no snapshot threshold with
    `ROLE_UNKNOWN_STATE` (in place) -- returns `{state id: window count}` for the
    notes (such windows get no verdict and are counted, never alarmed)."""
    unknown: dict[int, int] = {}
    for label in (int(v) for v in np.unique(labels[prepared.valid_mask]).tolist()):
        if label in snapshot.thresholds:
            continue
        idx = np.flatnonzero(prepared.valid_mask & (labels == label))
        role[idx] = ROLE_UNKNOWN_STATE
        unknown[label] = int(idx.size)
        logger.warning(
            "monitor: %d valid window(s) have detected state %d, which the snapshot "
            "has no reference/threshold for -- role=%s, never alarmed",
            int(idx.size), label, ROLE_UNKNOWN_STATE,
        )
    return unknown


def _frozen_verdicts(
    prepared: PreparedRun,
    snapshot: MonitorSnapshot,
    labels: np.ndarray,
    *,
    scoring_features: np.ndarray | None = None,
    session_stats: SessionStats | None = None,
) -> _Verdicts:
    """Frozen mode: every valid window of a snapshot-known state gets a
    verdict against the fit day's STORED threshold; p-values against the fit day's
    stored calibration scores (the same set the threshold came from).

    Under `--session-norm` the caller passes *scoring_features* (this
    run's session-normalized matrix -- `None` means score the raw features) and
    *session_stats* (the SNAPSHOT's stats, transforming the stored references via
    `scorer_for_label`); the stored thresholds/calibration scores then live in the
    snapshot's own normalized space, comparable at FAR level only."""
    features = prepared.features if scoring_features is None else scoring_features
    score, p_value, alarm, low_confidence, role, threshold_source = _empty_verdict_arrays(
        prepared.features.shape[0], "frozen"
    )
    state_rows: list[_StateRow] = []
    for label in sorted(snapshot.thresholds):
        threshold = snapshot.thresholds[label]
        idx = np.flatnonzero(prepared.valid_mask & (labels == label))
        n_alarms = 0
        if idx.size:
            scores = scorer_for_label(
                snapshot, label, session_stats=session_stats
            ).score(features[idx])
            score[idx] = scores
            p_value[idx] = p_values(scores, snapshot.calibration_scores[label])
            alarms = scores > threshold.threshold
            alarm[idx] = alarms
            low_confidence[idx] = threshold.low_confidence
            role[idx] = ROLE_SCORED
            n_alarms = int(alarms.sum())
        state_rows.append(
            _StateRow(
                state=label, n_windows=int(idx.size), n_scored=int(idx.size),
                n_alarms=n_alarms, n_consumed=0, threshold=threshold.threshold,
                low_confidence=threshold.low_confidence, status=ROLE_SCORED,
            )
        )
    unknown_counts = _mark_unknown_states(prepared, snapshot, labels, role)
    return _Verdicts(
        score=score, p_value=p_value, alarm=alarm, low_confidence=low_confidence,
        role=role, threshold_source=threshold_source, state_rows=state_rows,
        unknown_counts=unknown_counts,
    )


def _load_exclusion_intervals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an `--exclude-calibration-events` CSV into `(starts_ns, ends_ns)`
    int64 UTC-epoch arrays -- the SAME file contract as `scripts/eval_events.py`'s
    `--events` input (`start_utc`/`end_utc` tz-aware ISO-8601, `#` comment lines
    skipped), so one ground-truth file drives both the calibration ban here and
    the detection credit there.

    Raises:
        ValueError: missing columns, no rows, naive (or unparseable) timestamps,
            or an interval whose end is not after its start -- the caller turns
            this into the CLI's exit-2 message.
    """
    frame = pd.read_csv(path, comment="#")
    missing = [c for c in ("start_utc", "end_utc") if c not in frame.columns]
    if missing:
        raise ValueError(
            f"exclusion events file {path} is missing required column(s) {missing} "
            f"-- got columns {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError(f"exclusion events file {path} contains no event rows")
    bounds: list[np.ndarray] = []
    for col in ("start_utc", "end_utc"):
        parsed = pd.to_datetime(frame[col])
        if getattr(parsed.dt, "tz", None) is None:
            raise ValueError(
                f"exclusion events file {path} column {col!r} has naive "
                f"timestamps -- tz-aware ISO-8601 required (the eval_events "
                f"contract; a naive reading silently guesses the timebase, the "
                f"exact failure mode the 080726 ground-truth correction fixed)"
            )
        # as_unit("ns") before the int64 view: pandas >= 2 may parse second-
        # resolution strings into datetime64[s], whose int64 view is epoch
        # SECONDS -- a silent 1e9 scale error.
        bounds.append(
            parsed.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()
        )
    starts_ns, ends_ns = bounds
    if bool((ends_ns <= starts_ns).any()):
        raise ValueError(
            f"exclusion events file {path} has interval(s) with end_utc <= start_utc"
        )
    return starts_ns, ends_ns


def _exclusion_mask(
    grid: WindowGrid, starts_ns: np.ndarray, ends_ns: np.ndarray, tolerance_s: float
) -> np.ndarray:
    """(W,) bool -- True where a grid window's `[start, end)` span overlaps any
    event interval expanded by *tolerance_s* on both ends (half-open overlap
    test, mirroring `evaluate_events`' inclusive-start/exclusive-end reading)."""
    tol_ns = int(round(tolerance_s * 1e9))
    w_start = grid.t0_ns + np.arange(grid.n_windows, dtype=np.int64) * grid.window_ns
    w_end = w_start + grid.window_ns
    mask = np.zeros(grid.n_windows, dtype=bool)
    for s, e in zip(starts_ns.tolist(), ends_ns.tolist(), strict=True):
        mask |= (w_start < e + tol_ns) & (w_end > s - tol_ns)
    return mask


def _apply_calibration_exclusion(
    cal_windows: np.ndarray, scoring_windows: np.ndarray, exclusion_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Calibration-side windows inside excluded event intervals are
    moved to the SCORING side -- they must never feed a threshold (an induced
    event calibrated into the conformal set both inflates the threshold and, via
    the consumed-never-alarmed rule, becomes structurally undetectable),
    but they remain fully evaluable against the event-free thresholds. Scoring-
    side event windows are untouched (already scored). Returns the adjusted
    `(cal_windows, scoring_windows, n_moved)`."""
    in_events = exclusion_mask[cal_windows]
    excluded = cal_windows[in_events]
    if excluded.size == 0:
        return cal_windows, scoring_windows, 0
    return (
        cal_windows[~in_events],
        np.sort(np.concatenate([scoring_windows, excluded])),
        int(excluded.size),
    )


def _recalibrate_verdicts(
    prepared: PreparedRun,
    snapshot: MonitorSnapshot,
    labels: np.ndarray,
    alpha: float,
    *,
    scoring_features: np.ndarray | None = None,
    session_stats: SessionStats | None = None,
    exclusion_mask: np.ndarray | None = None,
) -> _Verdicts:
    """Recalibrate mode (DEFAULT): top split of the new run's valid
    windows at the snapshot's own `(calibration_frac, seed)`; per snapshot-known
    state, a fresh conformal threshold from the calibration-side windows of that
    state (references stay the SNAPSHOT's -- thresholds only), verdicts for the
    scoring-side windows only. Calibration-side windows are consumed, never alarmed
    (calibration-bias rule); a state with zero calibration-side windows takes the
    `no_conformal_data` path for ALL its windows.

    Under `--session-norm`: *scoring_features*/*session_stats* exactly as
    in `_frozen_verdicts` -- here the recalibrated thresholds AND p-values are
    computed from this run's own scores in the normalized space end to end, so the
    mode stays fully coherent (no cross-space comparison anywhere)."""
    features = prepared.features if scoring_features is None else scoring_features
    score, p_value, alarm, low_confidence, role, threshold_source = _empty_verdict_arrays(
        prepared.features.shape[0], "recalibrate"
    )
    top = split_by_segments(
        prepared.segment_ids, prepared.valid_mask, snapshot.calibration_frac, snapshot.seed
    )
    cal_windows, scoring_windows = top.calibration_windows, top.scoring_windows
    if exclusion_mask is not None:
        cal_windows, scoring_windows, n_excluded = _apply_calibration_exclusion(
            cal_windows, scoring_windows, exclusion_mask
        )
        if n_excluded:
            logger.info(
                "monitor: %d calibration-side window(s) overlap excluded event "
                "intervals -- moved to the scoring side (spec A2.3.3: event "
                "windows must never calibrate)",
                n_excluded,
            )

    state_rows: list[_StateRow] = []
    for label in sorted(snapshot.thresholds):
        label_all = np.flatnonzero(prepared.valid_mask & (labels == label))
        label_cal = cal_windows[labels[cal_windows] == label]
        label_scr = scoring_windows[labels[scoring_windows] == label]

        if label_cal.size == 0:
            # No per-day threshold can be calibrated -> no verdicts at all for
            # this state (falling back to the frozen fit-day threshold here would
            # silently reintroduce exactly the behavior this mode exists to avoid).
            role[label_all] = ROLE_NO_CONFORMAL_DATA
            logger.warning(
                "monitor: state %d has ZERO calibration-side windows on this run -- "
                "all %d of its window(s) get role=%s, no verdicts",
                label, int(label_all.size), ROLE_NO_CONFORMAL_DATA,
            )
            state_rows.append(
                _StateRow(
                    state=label, n_windows=int(label_all.size), n_scored=0,
                    n_alarms=0, n_consumed=0, threshold=math.nan,
                    low_confidence=None, status=ROLE_NO_CONFORMAL_DATA,
                )
            )
            continue

        scorer = scorer_for_label(snapshot, label, session_stats=session_stats)
        cal_scores = scorer.score(features[label_cal])
        # Deliberately NO `min_ref` gate here (resolved as a
        # documented reading of "identical to sweeps"): `run_sweep` gates
        # the FIT side with `min_ref` (reference quality -- enforced for this
        # snapshot at BUILD time by `fit_snapshot`) and governs the conformal/
        # calibration side ONLY by `calibrate`'s own achievable-alpha floor
        # (`low_confidence`), zero-count aside. A tiny calibration side therefore
        # yields a VALID conformal threshold whose sample count is visible in the
        # notes' per-state `n_consumed` column and whose confidence is carried by
        # `low_confidence` -- exactly the sweeps' semantics, not a second gate.
        threshold = calibrate(cal_scores, alpha)
        # Calibration-bias rule: scores are recorded for provenance (they ARE
        # the threshold's empirical distribution), but no p-value, never an alarm.
        score[label_cal] = cal_scores
        role[label_cal] = ROLE_CONSUMED

        n_alarms = 0
        if label_scr.size:
            scores = scorer.score(features[label_scr])
            score[label_scr] = scores
            p_value[label_scr] = p_values(scores, cal_scores)
            alarms = scores > threshold.threshold
            alarm[label_scr] = alarms
            low_confidence[label_scr] = threshold.low_confidence
            role[label_scr] = ROLE_SCORED
            n_alarms = int(alarms.sum())
        state_rows.append(
            _StateRow(
                state=label, n_windows=int(label_all.size), n_scored=int(label_scr.size),
                n_alarms=n_alarms, n_consumed=int(label_cal.size),
                threshold=threshold.threshold, low_confidence=threshold.low_confidence,
                status=ROLE_SCORED,
            )
        )

    unknown_counts = _mark_unknown_states(prepared, snapshot, labels, role)
    return _Verdicts(
        score=score, p_value=p_value, alarm=alarm, low_confidence=low_confidence,
        role=role, threshold_source=threshold_source, state_rows=state_rows,
        unknown_counts=unknown_counts,
    )



def _trailing_bounds(
    cal_t: np.ndarray, scr_t: np.ndarray, m_ns: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per scored-window start time `t`, the [lo, hi) index range of SORTED
    calibration start times inside the half-open trailing interval
    `[t - m_ns, t)` -- INCLUSIVE at the lower edge, EXCLUSIVE at the upper.

    Factored out because the upper-edge exclusivity is
    structurally unreachable through the CLI (calibration and scoring windows
    are segment-disjoint, so `cal_t == scr_t` never occurs in real runs) --
    only a direct unit test on synthetic arrays can pin it, and a future
    relaxation of that disjointness elsewhere must not silently flip this
    boundary.
    """
    lo = np.searchsorted(cal_t, scr_t - m_ns, side="left")
    hi = np.searchsorted(cal_t, scr_t, side="left")
    return lo, hi

def _rolling_floor(alpha: float) -> int:
    """Smallest trailing calibration count at which `calibrate(..., alpha)` yields a
    REAL (finite, non-low-confidence) threshold -- mathematically `ceil(1/alpha) - 1`
    (the conformal floor, equivalently `n >= 1/alpha - 1`). Derived by
    probing `threshold_index` around the closed-form candidate instead of a separate
    floating-point `ceil(1/alpha)` so the rolling gate and `calibrate`'s own
    `low_confidence` boundary can NEVER disagree (the same boundary-consistency rule
    `threshold_index`'s docstring pins for `calibrate` itself). The achievability
    condition `threshold_index(n, alpha) <= n` is monotone in `n` ( `<=>`
    `(n + 1) * alpha >= 1` up to the shared ceil tolerance), so the two local
    adjustment loops terminate after at most a step each in practice."""
    candidate = max(1, math.ceil(1.0 / alpha) - 1)
    while threshold_index(candidate, alpha) > candidate:
        candidate += 1
    while candidate > 1 and threshold_index(candidate - 1, alpha) <= candidate - 1:
        candidate -= 1
    return candidate


def _rolling_verdicts(
    prepared: PreparedRun,
    snapshot: MonitorSnapshot,
    labels: np.ndarray,
    alpha: float,
    *,
    rolling_minutes: float,
    scoring_features: np.ndarray | None = None,
    session_stats: SessionStats | None = None,
    exclusion_mask: np.ndarray | None = None,
) -> _Verdicts:
    """Rolling mode: the SAME top split and roles as
    recalibrate mode (calibration side consumed and never alarmed, scoring side
    scored -- rolling changes HOW each threshold is derived, not WHICH windows get
    verdicts). Per scored window `w` of state `s`: the trailing set is the scores
    of `s`'s calibration-side windows whose window START lies in
    `[t_w - rolling_minutes, t_w)`; when its count reaches the conformal floor
    `ceil(1/alpha) - 1` (`_rolling_floor`) the threshold is `calibrate(trailing,
    alpha)` and the p-value is computed against that SAME trailing set
    (`threshold_source = SOURCE_ROLLING`); below the floor the window falls back to
    the snapshot's STORED fit-day threshold and stored calibration scores
    (`SOURCE_FIT_DAY_FALLBACK`) -- an explicit, per-window-flagged fallback, never
    a silent one (the rule against silent frozen fallbacks is exactly why the
    flag column exists). Consumed windows' scores are recorded (no p-value, never
    an alarm), which doubles as the audit trail: every rolling threshold is
    recomputable from alarms.parquet alone.

    Under `--session-norm`: *scoring_features*/*session_stats* exactly as
    in `_recalibrate_verdicts` -- trailing calibration scores, rolling thresholds
    and rolling p-values all live in the session-normalized space; the FALLBACK
    branch compares against the snapshot's stored (fit-space) thresholds exactly
    like frozen mode, with frozen mode's comparability caveats."""
    features = prepared.features if scoring_features is None else scoring_features
    score, p_value, alarm, low_confidence, role, threshold_source = _empty_verdict_arrays(
        prepared.features.shape[0], ""
    )
    top = split_by_segments(
        prepared.segment_ids, prepared.valid_mask, snapshot.calibration_frac, snapshot.seed
    )
    cal_windows, scoring_windows = top.calibration_windows, top.scoring_windows
    if exclusion_mask is not None:
        cal_windows, scoring_windows, n_excluded = _apply_calibration_exclusion(
            cal_windows, scoring_windows, exclusion_mask
        )
        if n_excluded:
            logger.info(
                "monitor: %d calibration-side window(s) overlap excluded event "
                "intervals -- moved to the scoring side (spec A2.3.3: event "
                "windows must never calibrate; their rolling thresholds come from "
                "event-free trailing sets, or the flagged fit-day fallback)",
                n_excluded,
            )
    m_ns = int(round(rolling_minutes * 60.0 * 1e9))
    floor = _rolling_floor(alpha)

    state_rows: list[_StateRow] = []
    for label in sorted(snapshot.thresholds):
        frozen = snapshot.thresholds[label]
        label_all = np.flatnonzero(prepared.valid_mask & (labels == label))
        # Ascending window index == ascending start time (uniform grid), as
        # np.searchsorted below requires.
        label_cal = np.sort(cal_windows[labels[cal_windows] == label])
        label_scr = scoring_windows[labels[scoring_windows] == label]

        scorer = scorer_for_label(snapshot, label, session_stats=session_stats)
        if label_cal.size:
            cal_scores = scorer.score(features[label_cal])
            score[label_cal] = cal_scores
            role[label_cal] = ROLE_CONSUMED
        else:
            # No same-state calibration windows on this run: every scored window's
            # trailing count is 0 < floor -> all take the flagged fallback branch
            # (the rolling analogue of recalibrate mode's no_conformal_data state,
            # made explicit per window instead of withholding verdicts).
            cal_scores = np.empty(0, dtype=np.float64)

        n_alarms = n_rolling = n_fallback = 0
        if label_scr.size:
            scr_scores = scorer.score(features[label_scr])
            score[label_scr] = scr_scores
            role[label_scr] = ROLE_SCORED

            cal_t = prepared.grid.t0_ns + label_cal.astype(np.int64) * prepared.grid.window_ns
            scr_t = prepared.grid.t0_ns + label_scr.astype(np.int64) * prepared.grid.window_ns
            lo, hi = _trailing_bounds(cal_t, scr_t, m_ns)
            rolling_mask = (hi - lo) >= floor

            fallback_idx = label_scr[~rolling_mask]
            if fallback_idx.size:
                fb_scores = scr_scores[~rolling_mask]
                p_value[fallback_idx] = p_values(
                    fb_scores, snapshot.calibration_scores[label]
                )
                alarm[fallback_idx] = fb_scores > frozen.threshold
                low_confidence[fallback_idx] = frozen.low_confidence
                threshold_source[fallback_idx] = SOURCE_FIT_DAY_FALLBACK
                n_fallback = int(fallback_idx.size)

            for pos in np.flatnonzero(rolling_mask).tolist():
                w = int(label_scr[pos])
                trailing = cal_scores[int(lo[pos]) : int(hi[pos])]
                rolled = calibrate(trailing, alpha)
                p_value[w] = p_values(scr_scores[pos : pos + 1], trailing)[0]
                alarm[w] = bool(scr_scores[pos] > rolled.threshold)
                low_confidence[w] = rolled.low_confidence  # False: count >= floor
                threshold_source[w] = SOURCE_ROLLING
                n_rolling += 1

            n_alarms = int(alarm[label_scr].sum())

        state_rows.append(
            _StateRow(
                state=label, n_windows=int(label_all.size), n_scored=int(label_scr.size),
                n_alarms=n_alarms, n_consumed=int(label_cal.size),
                # No single per-state threshold exists in rolling mode -- the
                # per-state picture lives in the notes' coverage table.
                threshold=math.nan, low_confidence=None, status=ROLE_SCORED,
                n_rolling=n_rolling, n_fallback=n_fallback,
            )
        )

    unknown_counts = _mark_unknown_states(prepared, snapshot, labels, role)
    return _Verdicts(
        score=score, p_value=p_value, alarm=alarm, low_confidence=low_confidence,
        role=role, threshold_source=threshold_source, state_rows=state_rows,
        unknown_counts=unknown_counts,
    )


def _assert_roles_complete(role: np.ndarray, valid_mask: np.ndarray) -> None:
    """Every valid window must have received exactly one role. A valid window can
    only miss one if it sits outside every real segment (`segment_ids == -1`), which
    the validity rule makes impossible (zero primary-stream coverage fails the
    >= 0.8 coverage requirement) -- so a failure here means an upstream invariant
    changed and the verdicts above are incomplete (trust but verify, the
    `run_sweep._assert_three_way_disjoint` style).

    Raises:
        RuntimeError: naming the offending window count.
    """
    missing = int((role[valid_mask] == "").sum())
    if missing:
        raise RuntimeError(
            f"{missing} valid window(s) received no role -- the valid-windows-"
            f"always-carry-a-real-segment-id invariant no longer holds; refusing to "
            f"write incomplete verdicts"
        )


# ---------------------------------------------------------------------------
# Transitions + named states
# ---------------------------------------------------------------------------


def _near_transition_mask(
    labels: np.ndarray, valid_mask: np.ndarray, window_ns: int, w_seconds: float
) -> np.ndarray:
    """(W,) bool, True for every VALID window within `+-round(w_seconds/window_s)`
    STEPS -- counted along the VALID subsequence, not raw grid index -- of a
    detected-state CHANGE found in that same valid subsequence.

    Invalid windows are filtered FIRST; a change is any label difference between
    consecutive entries of the compacted valid-only label sequence, so a
    valid->invalid->valid blip is NEVER a change (the invalid run in between never
    participates in the comparison). Distance is then measured in valid-sequence
    STEPS rather than wall-clock grid steps: this mirrors `FittedDetector._finish`'s
    own `min_dwell` filter, which duration-filters the SAME compacted valid
    sequence the Viterbi decoder actually saw (`_apply_detector_labels` decodes
    `features[valid_mask]` against a grid of exactly `valid_mask.sum()` windows) --
    counting through an invalid gap as if it were real elapsed time would put
    `near_transition` on a different clock than the dwell filter it is meant to
    match. Invalid windows are always False (never flagged)."""
    out = np.zeros(labels.shape[0], dtype=bool)
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size < 2:
        return out
    valid_labels = labels[valid_idx]
    # Position (within the compacted valid subsequence) of every new-state onset.
    change_positions = np.flatnonzero(valid_labels[1:] != valid_labels[:-1]) + 1
    if change_positions.size == 0:
        return out
    # max(1, ...) floor: literal parity with FittedDetector._finish's own
    # `max(1, round(min_dwell_s / window_s))` -- a w_seconds small enough that
    # round() truncates to 0 steps must still flag the immediate neighbours of a
    # transition, not degenerate to "only the boundary window itself".
    w_windows = max(1, int(round(w_seconds / (window_ns / 1e9))))
    positions = np.arange(valid_idx.shape[0])
    near = np.zeros(valid_idx.shape[0], dtype=bool)
    for cp in change_positions.tolist():
        near |= np.abs(positions - cp) <= w_windows
    out[valid_idx[near]] = True
    return out


def _state_name_for(state_id: int, state_names: dict[int, str] | None) -> str:
    """The naming convention shared by every surfacing site: `"invalid"` for
    `_INVALID_LABEL`; the snapshot's own name when *state_names* carries one for
    *state_id*; else the bare `cluster-<id>` fallback (matching
    `rowii.eval.metrics.derive_state_names`' own fallback)."""
    if state_id == _INVALID_LABEL:
        return "invalid"
    if state_names is not None and state_id in state_names:
        return state_names[state_id]
    return f"cluster-{state_id}"


def _apply_transition_suppression(
    alarm: np.ndarray, near_transition: np.ndarray, role: np.ndarray
) -> int:
    """`--suppress-transition-alarms`: in place, force
    `alarm=False` on every SCORED window that is both currently an alarm and
    `near_transition` -- returns the count actually flipped
    (`suppressed_by_transition`, the notes' reported number). `score`/`p_value`/
    `role`/`near_transition` are left untouched (the full-audit-trail rule: a
    suppressed alarm must remain reconstructable from `alarms.parquet` alone).
    In-place array mutation is the established codebase idiom for `_Verdicts`
    (`_mark_unknown_states`); only SCORED windows are eligible -- consumed/
    unknown-state/no-conformal-data windows are never alarmed in the first place."""
    target = near_transition & (role == ROLE_SCORED) & alarm
    n = int(target.sum())
    alarm[target] = False
    return n


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _alarms_frame(
    prepared: PreparedRun,
    labels: np.ndarray,
    verdicts: _Verdicts,
    near_transition: np.ndarray,
    state_names: dict[int, str] | None,
) -> pd.DataFrame:
    """One row per VALID window, ascending, exactly `_ALARM_COLUMNS`. `t_utc_ns` is
    the window's grid left edge (`rowii.anomaly.overlap.to_utc_ns`'s own
    `t0_ns + window * window_ns` convention, reused directly -- it lives in `src`,
    not in a sibling script). `near_transition`/`state_name` are ALWAYS present
    -- the latter via `_state_name_for`, so an unmapped state still gets its
    `cluster-<id>` fallback name rather than a missing/empty cell."""
    idx = np.flatnonzero(prepared.valid_mask).astype(np.int64)
    frame = pd.DataFrame(
        {
            "window": idx,
            "state": labels[idx],
            "score": verdicts.score[idx],
            "p_value": verdicts.p_value[idx],
            "alarm": verdicts.alarm[idx],
            "low_confidence": verdicts.low_confidence[idx],
            "role": verdicts.role[idx].astype(str),
            "threshold_source": verdicts.threshold_source[idx].astype(str),
            "near_transition": near_transition[idx],
            "state_name": np.array(
                [_state_name_for(int(labels[i]), state_names) for i in idx], dtype=object
            ),
        }
    )
    frame = to_utc_ns(frame, prepared.grid.t0_ns, prepared.grid.window_ns)
    return frame[list(_ALARM_COLUMNS)]


def _alarm_segments(alarm: np.ndarray, grid: WindowGrid) -> pd.DataFrame:
    """Maximal alarm runs as `start_utc, end_utc, duration_s` -- `to_segments` over
    the full-grid 0/1 alarm indicator (invalid/unscored windows are 0 by
    construction), keeping only the alarm (`cluster == 1`) rows."""
    indicator = alarm.astype(np.int64)
    segments = to_segments(indicator, grid)
    out = segments.loc[segments["cluster"] == 1, ["start_utc", "end_utc", "duration_s"]]
    return out.reset_index(drop=True)


def _state_segments(
    labels: np.ndarray, grid: WindowGrid, state_names: dict[int, str] | None
) -> pd.DataFrame:
    """`segments.csv`'s table: `to_segments` over the full-length label array
    (`_INVALID_LABEL` segments included, visibly), `cluster` renamed to
    `cluster_id` -- `scripts/apply_detector.py`'s convention. `mapped_mode`
    is added ONLY when *state_names* is not None -- a nameless snapshot
    still carries no cluster-id -> mode-name mapping, and inventing one here would
    be a claim the artifact cannot back; this keeps the column conditional, unlike
    `alarms.parquet`'s ALWAYS-present `state_name`."""
    segments = to_segments(labels, grid).rename(columns={"cluster": "cluster_id"})
    if state_names is not None:
        segments["mapped_mode"] = [
            _state_name_for(int(cluster_id), state_names)
            for cluster_id in segments["cluster_id"]
        ]
    return segments


def _timeline_markdown(
    run_name: str, snapshot: MonitorSnapshot, segments: pd.DataFrame
) -> str:
    known = ", ".join(str(label) for label in sorted(snapshot.thresholds)) or "(none)"
    lines = [
        f"# State timeline: {run_name} ({snapshot.variant}), "
        f"labeled by the snapshot of {snapshot.fit_run}",
        "",
        f"**Labels come from {snapshot.fit_run}'s persisted detector (fit-day "
        "standardisation + fit-day HMM decode, no refit) applied unchanged to this "
        "recording. This day has no ground truth and NO fault labels -- the timeline "
        "is qualitative context for the alarm outputs, never an evaluation.** "
        f"States the snapshot can alarm on: {known}.",
        "",
        "## Segments",
        "",
    ]
    if segments.empty:
        lines.append("(no segments)")
    for _, row in segments.iterrows():
        cluster_id = int(row["cluster_id"])
        if cluster_id == _INVALID_LABEL:
            desc = "invalid windows (no usable features)"
        elif cluster_id in snapshot.thresholds:
            if snapshot.state_names is not None:
                desc = f"state {cluster_id} ({_state_name_for(cluster_id, snapshot.state_names)})"
            else:
                desc = f"state {cluster_id}"
        else:
            desc = f"state {cluster_id} (unknown to the snapshot -- never alarmed)"
        lines.append(
            f"- {row['start_utc'].isoformat()} -> {row['end_utc'].isoformat()} "
            f"({row['duration_s']:.1f}s): {desc}"
        )
    lines.append("")
    return "\n".join(lines)


def _session_norm_lines(snapshot_stats: SessionStats, run_stats: SessionStats) -> list[str]:
    """The notes' `--session-norm` section: names the mode, BOTH stats'
    n_windows, the pool-global sentinel where it applies, and the caveats that must
    travel with every session-normalized number."""
    if snapshot_stats.norm_minutes > 0.0:
        snapshot_desc = (
            f"n_windows={snapshot_stats.n_windows}, "
            f"norm_minutes={snapshot_stats.norm_minutes:g} (fit-day first-N stats)"
        )
    else:
        snapshot_desc = (
            f"n_windows={snapshot_stats.n_windows}, norm_minutes=0 (pool-global "
            f"stats over the pooled fit matrix -- `rowii.anomaly.normalize."
            f"fit_pool_stats`' sentinel)"
        )
    return [
        "",
        "## Session normalization (--session-norm, spec D3/A3.5)",
        "",
        "Scoring happened in the session-normalized space: the snapshot's stored "
        "RAW references were transformed with the SNAPSHOT's stats, this run's "
        "windows with THIS run's own label-free first-N median/MAD stats -- "
        "scores, thresholds and p-values above all live in that space. Detected "
        "state labels come from RAW features (A3.5 binding boundary: labels are "
        "norm-invariant by construction). Comparability with raw-space monitor "
        "numbers is FAR-level only.",
        "",
        f"- snapshot reference-side stats: {snapshot_desc}",
        f"- monitored-run stats: n_windows={run_stats.n_windows}, "
        f"norm_minutes={run_stats.norm_minutes:g} (fit on this run's first "
        f"minutes, no labels)",
        "- caveats: if the first minutes contain a fault, normalization absorbs "
        "part of it; the first minutes' STATE MIX parameterizes the stats "
        "(state-mix confound -- the norm-minutes sweep is the sensitivity probe).",
    ]


def _level_recal_lines(offsets: dict[str, float]) -> list[str]:
    """The notes' `--level-recal` section: names the
    mechanism, the anchor source, and the per-column offset table -- mirrors
    `scripts/run_step2.py`'s own `level_recal_note_lines` block (duplicated
    text, not imported -- the script-sibling rule)."""
    return [
        "",
        "## Level-only recalibration (--level-recal, spec D2/A1.4/A1.10/A1.11)",
        "",
        "Scoring happened with this run's LEVEL columns (`*_log_rms`/`*_band_*`/"
        "`*_octave_*`, log10-scaled by construction -- VERIFIED fact, "
        "`rowii.anomaly.levelrecal` module docstring) additively recentred onto "
        "the SNAPSHOT's stored anchor medians (`level_recal_medians`, fit-time "
        "anchor, A1.4); shape columns (`*_spectral_centroid`/`*_rolloff95`/"
        "`*_kurtosis`) pass through untouched by construction. The recalibration "
        "is applied AFTER the snapshot-contract projection (A1.11), name-keyed "
        "against the projected feature_names. Offset = `run_median - "
        "reference_median` per column, the run-side median drawn from this "
        f"run's own label-free first-{_DEFAULT_NORM_MINUTES:g}-minute prefix. "
        "The DETECTOR consumed RAW features (labels are norm-invariant, the "
        "same A3.5 boundary --session-norm uses); references stay the "
        "snapshot's RAW references -- only this run's scoring features shift.",
        "",
        "This is our own, independent, feature-native test of the partner's "
        "channel-recalibration idea (Rodrigues & Zhang, 2026): every offset "
        "below is computed from OUR OWN caches; no partner dB figure is "
        "imported here (D2, spec A1.1/A1.9).",
        "",
        "| level column | offset (log10 units, run - reference) |",
        "|---|---|",
        *(f"| {name} | {offset:.6g} |" for name, offset in sorted(offsets.items())),
        "",
    ]


def _rolling_coverage_lines(
    state_rows: list[_StateRow],
    snapshot: MonitorSnapshot,
    rolling_minutes: float,
    floor: int,
) -> list[str]:
    """The notes' MANDATORY rolling-mode block: per-state trailing-
    coverage table (fraction of scored windows whose trailing calibration count
    reached the conformal floor at this M), the motivating measurement as the
    rationale, and the double-reference honesty note."""
    lines = [
        "",
        f"## Rolling trailing coverage (M = {rolling_minutes:g} min, conformal "
        f"floor = {floor} window(s); MANDATORY output, spec A3.2)",
        "",
        "| state | n_scored | n_rolling | n_fallback | rolling_coverage "
        "| fit_day_fallback_threshold |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in state_rows:
        coverage = f"{row.n_rolling / row.n_scored:.4f}" if row.n_scored else "n/a"
        fallback = snapshot.thresholds[row.state].threshold
        fallback_str = "inf" if math.isinf(fallback) else f"{fallback:.6g}"
        lines.append(
            f"| {row.state} | {row.n_scored} | {row.n_rolling} | {row.n_fallback} "
            f"| {coverage} | {fallback_str} |"
        )
    lines += [
        "",
        "Rationale (A3.2's motivating measurement, spec-review probe on 290626-tu): "
        "at M=20 only 46.8% of scored windows reach the conformal floor (states 2/3: "
        "0%), and at M=60 still only 53.8% -- trailing coverage varies sharply with "
        "state mix and M, so this per-state table is a mandatory output and the "
        "fit-day fallback is what keeps every scored window verdict-bearing.",
        "",
        "**Double-reference honesty note (spec D7):** a slowly developing fault "
        "inflates the trailing calibration scores themselves, so a rolling threshold "
        "can absorb it. Rolling verdicts must therefore be read side by side with "
        "fit-day-referenced verdicts -- the `threshold_source` column is that "
        f"visibility: every `{SOURCE_FIT_DAY_FALLBACK}` row IS a fit-day verdict, "
        "and a separate frozen/recalibrate pass over the same recording supplies "
        "the full fit-day comparison.",
    ]
    return lines


def _notes_markdown(
    run_name: str,
    snapshot_path: Path,
    snapshot: MonitorSnapshot,
    mode: str,
    alpha_used: float,
    verdicts: _Verdicts,
    prepared: PreparedRun,
    *,
    session_stats: SessionStats | None = None,
    run_stats: SessionStats | None = None,
    level_recal_offsets_used: dict[str, float] | None = None,
    rolling_minutes: float | None = None,
    exclusion_note: str | None = None,
    suppressed_by_transition: int | None = None,
) -> str:
    n_windows = int(prepared.features.shape[0])
    n_valid = int(prepared.valid_mask.sum())
    role_valid = verdicts.role[prepared.valid_mask]
    n_scored = int((role_valid == ROLE_SCORED).sum())
    n_consumed = int((role_valid == ROLE_CONSUMED).sum())
    n_unknown = int((role_valid == ROLE_UNKNOWN_STATE).sum())
    n_no_conformal = int((role_valid == ROLE_NO_CONFORMAL_DATA).sum())

    lines = [
        f"# Monitor report: {run_name} ({snapshot.variant}, thresholds mode: {mode})",
        "",
        "**NO fault labels exist for this recording (or any recording in this "
        "project so far) -- every alarm below is a CANDIDATE for operator review, "
        "never a verified detection (spec §4 honesty rule). Alarm rates measure "
        "statistical outlierness against the snapshot's fit-day normal references, "
        "nothing more.**",
        "",
        "## Snapshot provenance",
        "",
        f"- snapshot file: `{snapshot_path}`",
        f"- fit_run: {snapshot.fit_run}",
        f"- variant: {snapshot.variant}",
        f"- scorer: {snapshot.scorer}",
        f"- nominal alpha at fit time: {snapshot.alpha}",
        f"- alpha used by this pass: {alpha_used}"
        + (" (frozen mode: fit-day thresholds applied unchanged)" if mode == "frozen" else ""),
        f"- created_at: {snapshot.created_at}",
        f"- fit split: calibration_frac={snapshot.calibration_frac}, "
        f"seed={snapshot.seed}, min_ref={snapshot.min_ref}",
    ]
    if exclusion_note is not None:
        lines.append(exclusion_note)
    if snapshot.checkpoints:
        lines.append("- checkpoints:")
        lines.extend(
            f"  - {name}: `{path}`" for name, path in sorted(snapshot.checkpoints.items())
        )
    else:
        lines.append("- checkpoints: (none recorded)")

    lines += ["", f"## Threshold mode: {mode}", ""]
    if mode == "frozen":
        lines.append(_FROZEN_SHIFT_WARNING)
    elif mode == "rolling":
        assert rolling_minutes is not None  # main() always supplies it in this mode
        floor = _rolling_floor(alpha_used)
        lines.append(
            "References stay the snapshot's fit-day references. Per SCORED window, "
            f"the threshold was recalibrated from the trailing {rolling_minutes:g} "
            "minutes' consumed calibration-side windows of the SAME state (top "
            f"split: calibration_frac={snapshot.calibration_frac}, "
            f"seed={snapshot.seed}) at alpha={alpha_used} whenever at least "
            f"ceil(1/alpha)-1 = {floor} such windows exist; below that conformal "
            "floor the window FELL BACK to the snapshot's stored fit-day threshold "
            f"(calibrated at fit-time alpha={snapshot.alpha}), flagged per window "
            f"in alarms.parquet's `threshold_source` column (`{SOURCE_ROLLING}` / "
            f"`{SOURCE_FIT_DAY_FALLBACK}` -- never a silent fallback). p-values "
            "follow the same branch: against the trailing set when rolling, "
            "against the fit day's stored calibration scores when falling back. "
            "Calibration-side windows are consumed by the trailing sets and are "
            f"NEVER alarmed (role `{ROLE_CONSUMED}` -- calibration-bias rule, "
            "A1.3); their recorded scores make every rolling threshold auditable "
            "from alarms.parquet alone. Batch ablation over recorded files, not a "
            "production stream (spec D7)."
        )
    else:
        lines.append(
            "References stay the snapshot's fit-day references; ONLY the per-state "
            "thresholds were recalibrated on this run's own calibration-side windows "
            f"(top split: calibration_frac={snapshot.calibration_frac}, "
            f"seed={snapshot.seed}) at alpha={alpha_used}. Package-2's cross-day "
            "evidence: this transfer-plus-recalibrate recipe is the only one whose "
            "realized false-alarm rate stayed at its nominal alpha. Calibration-side "
            "windows are consumed by the threshold fit and are NEVER alarmed "
            f"(role `{ROLE_CONSUMED}` -- calibration-bias rule, A1.3)."
        )

    if session_stats is not None and run_stats is not None:
        lines += _session_norm_lines(session_stats, run_stats)
    if level_recal_offsets_used is not None:
        lines += _level_recal_lines(level_recal_offsets_used)

    lines += [
        "",
        "## Per-state results",
        "",
        "| state | n_windows | n_scored | n_alarms | alarm_rate | low_confidence "
        "| threshold | n_consumed | status |",
        "|---:|---:|---:|---:|---:|:--|---:|---:|:--|",
    ]
    for row in verdicts.state_rows:
        rate = f"{row.n_alarms / row.n_scored:.4f}" if row.n_scored else "n/a"
        low_conf = "n/a" if row.low_confidence is None else str(row.low_confidence)
        threshold = "n/a" if math.isnan(row.threshold) else f"{row.threshold:.6g}"
        state_cell = (
            f"{row.state} ({_state_name_for(row.state, snapshot.state_names)})"
            if snapshot.state_names is not None
            else str(row.state)
        )
        lines.append(
            f"| {state_cell} | {row.n_windows} | {row.n_scored} | {row.n_alarms} "
            f"| {rate} | {low_conf} | {threshold} | {row.n_consumed} | {row.status} |"
        )
    if mode == "rolling":
        assert rolling_minutes is not None
        lines += [
            "",
            "In rolling mode no single per-state threshold exists (the `threshold` "
            "column above is n/a by design) -- the per-window picture is in "
            "alarms.parquet's `threshold_source` column and the coverage table "
            "below.",
        ]
        lines += _rolling_coverage_lines(
            verdicts.state_rows, snapshot, rolling_minutes, _rolling_floor(alpha_used)
        )
    lines += [
        "",
        "A state with `low_confidence=True` has too few calibration windows to "
        "certify its alpha (its threshold is +inf) and can therefore never alarm -- "
        "deliberate: a state below the conformal floor must not alarm under a false "
        "guarantee (`rowii.anomaly.conformal`).",
        "",
        "## Window accounting",
        "",
        f"- grid windows: {n_windows} ({n_valid} valid, {n_windows - n_valid} "
        "invalid -- invalid windows are never scored and carry no row in "
        "alarms.parquet)",
        f"- {ROLE_SCORED}: {n_scored}",
        f"- {ROLE_CONSUMED}: {n_consumed}"
        + (
            " (recalibrate-mode calibration side; never alarmed)"
            if mode == "recalibrate"
            else " (rolling-mode calibration side -- the trailing sets; never alarmed)"
            if mode == "rolling"
            else ""
        ),
        f"- {ROLE_UNKNOWN_STATE}: {n_unknown} (detected state has no snapshot "
        "reference/threshold; never alarmed)",
        f"- {ROLE_NO_CONFORMAL_DATA}: {n_no_conformal} (snapshot-known state with "
        "zero calibration-side windows on this run; never alarmed)",
    ]
    if suppressed_by_transition is not None:
        lines.append(
            f"- suppressed_by_transition: {suppressed_by_transition} (scored "
            "alarm(s) withheld by --suppress-transition-alarms because their "
            "window sits within the near-transition band, spec D3c; score/"
            "p_value/near_transition/role stay recorded above for a full audit "
            "trail -- every suppressed alarm remains reconstructable from "
            "alarms.parquet alone)"
        )
    if verdicts.unknown_counts:
        lines.append("- unknown detected states:")
        lines.extend(
            f"  - state {label}: {count} window(s)"
            for label, count in sorted(verdicts.unknown_counts.items())
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.alpha is not None and not (0.0 < args.alpha < 1.0):
        print(f"monitor: --alpha must be in (0, 1), got {args.alpha!r}", file=sys.stderr)
        return 2
    if args.rolling_minutes is not None and args.thresholds != "rolling":
        print(
            f"monitor: --rolling-minutes is only valid with --thresholds rolling "
            f"(got --thresholds {args.thresholds})",
            file=sys.stderr,
        )
        return 2
    if args.rolling_minutes is not None and not (
        math.isfinite(args.rolling_minutes)
        and 0.0 < args.rolling_minutes <= _MAX_ROLLING_MINUTES
    ):
        print(
            f"monitor: --rolling-minutes must be a positive number of minutes "
            f"<= {_MAX_ROLLING_MINUTES:g} (about a decade -- larger values overflow "
            f"the int64 ns arithmetic), got "
            f"{args.rolling_minutes!r}",
            file=sys.stderr,
        )
        return 2
    if args.level_recal and args.session_norm:
        # Fit-path exclusivity: session-normalization and level-only
        # recalibration are different scoring-space transforms and are never
        # both active in one pass.
        print(
            "monitor: --level-recal is mutually exclusive with --session-norm "
            "(A1.10 fit-path exclusivity)",
            file=sys.stderr,
        )
        return 2
    if not args.snapshot.is_file():
        print(f"monitor: snapshot file not found: {args.snapshot}", file=sys.stderr)
        return 2
    try:
        snapshot = load_snapshot(args.snapshot)
    except ValueError as exc:
        print(f"monitor: cannot load snapshot {args.snapshot}: {exc}", file=sys.stderr)
        return 2

    if args.session_norm and snapshot.session_stats is None:
        # Binding: never a silent raw-space fallback under --session-norm.
        print(
            f"monitor: --session-norm requires a snapshot that stores session-"
            f"normalization stats (format v2 with session stats), but "
            f"{args.snapshot} has none -- it is a format-v1 file or was fitted "
            f"without session normalization. Refit it with stats (scripts/"
            f"run_step2.py --protocol cross-day-pooled --session-norm "
            f"--save-snapshot) or drop --session-norm.",
            file=sys.stderr,
        )
        return 2
    if not args.session_norm and snapshot.session_stats is not None:
        logger.warning(
            "monitor: snapshot %s stores session-normalization stats (fitted under "
            "--session-norm) but --session-norm is OFF -- its stored FROZEN "
            "thresholds/calibration scores live in the session-normalized space and "
            "would be compared against RAW-space scores; this affects frozen mode "
            "AND rolling mode's fit_day_fallback branch (which reuses exactly those "
            "stored thresholds/scores); recalibrate mode and rolling's rolling "
            "branch are unaffected (thresholds recomputed on this run's raw scores "
            "against the raw references)",
            args.snapshot,
        )

    if args.level_recal and snapshot.level_recal_medians is None:
        # Binding: never a silent raw-space fallback under --level-recal.
        print(
            f"monitor: --level-recal requires a snapshot that stores level-"
            f"recal reference medians (format v2 with level_recal_medians), "
            f"but {args.snapshot} has none -- it is a format-v1 file, was "
            f"fitted without --level-recal, or was fitted under --session-norm "
            f"(session_stats and level_recal_medians are mutually exclusive by "
            f"fit path, A1.10). Refit it with medians (scripts/run_step2.py "
            f"--protocol cross-day-pooled --level-recal --save-snapshot) or "
            f"drop --level-recal.",
            file=sys.stderr,
        )
        return 2

    cfg = load_config()
    index = discover(cfg.data_root)
    by_name: dict[str, Run] = {r.name: r for r in index.runs}
    if args.run not in by_name:
        available = ", ".join(sorted(by_name)) or "(none discovered)"
        print(
            f"monitor: unknown run name: {args.run}; available runs: {available}",
            file=sys.stderr,
        )
        return 2
    run = by_name[args.run]

    prepared = prepare_run(run, snapshot.variant, cfg, use_cache=not args.no_cache)

    if list(prepared.feature_names) != list(snapshot.feature_names):
        # The snapshot's trained columns are the scoring contract (module
        # docstring): project the prepared run onto them BY NAME, refuse only
        # when a trained column is absent from the prepared run.
        prepared_pos = {name: i for i, name in enumerate(prepared.feature_names)}
        missing = [name for name in snapshot.feature_names if name not in prepared_pos]
        if missing:
            print(
                f"monitor: feature geometry mismatch for run {args.run!r}: the "
                f"snapshot was fitted on variant {snapshot.variant!r} with "
                f"{len(snapshot.feature_names)} feature column(s), but the prepared "
                f"run ({len(prepared.feature_names)} column(s)) is missing "
                f"{len(missing)} of them -- refusing to score (missing columns: "
                f"{missing})",
                file=sys.stderr,
            )
            return 2
        snapshot_names = set(snapshot.feature_names)
        extras = [name for name in prepared.feature_names if name not in snapshot_names]
        logger.warning(
            "monitor: projecting run %r onto the snapshot's %d-column feature "
            "contract -- dropping %d prepared column(s) the %r snapshot never saw "
            "at fit time (channel-availability drift; a trained model cannot score "
            "features it was not fitted on): %s. valid_mask is kept from the full "
            "prepared run (conservative: a window invalid only because of a "
            "dropped extra column stays excluded).",
            args.run, len(snapshot.feature_names), len(extras), snapshot.variant,
            extras,
        )
        column_index = np.array(
            [prepared_pos[name] for name in snapshot.feature_names], dtype=np.intp
        )
        prepared = dataclasses.replace(
            prepared,
            features=prepared.features[:, column_index],
            feature_names=list(snapshot.feature_names),
        )

    if not bool(prepared.valid_mask.any()):
        # Hardening: an all-invalid run would otherwise surface as a raw
        # sklearn ValueError inside the detector apply -- refuse cleanly instead.
        print(
            f"monitor: monitored run {args.run!r} has zero valid windows (valid_mask "
            f"is all False) -- nothing can be labeled or scored; refusing before "
            f"the detector apply",
            file=sys.stderr,
        )
        return 2

    detector = to_detector(snapshot)
    labels = _apply_detector_labels(prepared, detector)  # ALWAYS raw features

    session_stats: SessionStats | None = None
    run_stats: SessionStats | None = None
    scoring_features: np.ndarray | None = None
    level_recal_offsets_used: dict[str, float] | None = None
    if args.session_norm:
        session_stats = snapshot.session_stats
        assert session_stats is not None  # guarded right after load_snapshot
        norm_minutes = session_stats.norm_minutes
        if norm_minutes <= 0.0:
            logger.info(
                "monitor: snapshot stats are pool-global (norm_minutes=0 sentinel) "
                "-- fitting the monitored run's own stats over the default first "
                "%g minutes",
                _DEFAULT_NORM_MINUTES,
            )
            norm_minutes = _DEFAULT_NORM_MINUTES
        try:
            run_stats = fit_session_stats(
                prepared.features, prepared.valid_mask, prepared.grid,
                norm_minutes=norm_minutes,
            )
        except ValueError as exc:
            print(
                f"monitor: --session-norm cannot fit run {args.run!r}'s first-"
                f"{norm_minutes:g}-minute stats: {exc}",
                file=sys.stderr,
            )
            return 2
        scoring_features = apply_session_norm(prepared.features, run_stats)
    elif args.level_recal:
        # Applies AFTER the geometry-guard projection above, so
        # prepared.feature_names already equals snapshot.feature_names here --
        # the run-side median is name-keyed against the PROJECTED columns.
        # session_stats stays None (references scored RAW, the query is
        # aligned onto them, mirrors the detector-RAW boundary).
        assert snapshot.level_recal_medians is not None  # guarded after load_snapshot
        try:
            first_n_rows = _first_n_minutes_rows(prepared, _DEFAULT_NORM_MINUTES)
            run_median = column_medians(first_n_rows, snapshot.feature_names)
            level_recal_offsets_used = level_recal_offsets(
                run_median, snapshot.level_recal_medians
            )
            scoring_features = apply_level_recal(
                prepared.features, snapshot.feature_names, level_recal_offsets_used
            )
        except ValueError as exc:
            # Covers the zero-level-column refusal (an embedding snapshot's
            # feature contract) and an empty first-N/anchor overlap alike --
            # every ValueError from the levelrecal module surfaces as exit 2.
            print(
                f"monitor: --level-recal cannot compute offsets for run "
                f"{args.run!r}: {exc}",
                file=sys.stderr,
            )
            return 2

    exclusion_mask: np.ndarray | None = None
    exclusion_note: str | None = None
    if args.exclude_tolerance_s is not None and args.exclude_calibration_events is None:
        print(
            "monitor: --exclude-tolerance-s is only valid together with "
            "--exclude-calibration-events",
            file=sys.stderr,
        )
        return 2
    if args.exclude_calibration_events is not None:
        try:
            starts_ns, ends_ns = _load_exclusion_intervals(args.exclude_calibration_events)
        except ValueError as exc:
            print(f"monitor: {exc}", file=sys.stderr)
            return 2
        if args.thresholds == "frozen":
            logger.warning(
                "monitor: --exclude-calibration-events has no effect in frozen "
                "mode -- frozen draws no calibration from the monitored run "
                "(thresholds are applied exactly as stored), so there is nothing "
                "to exclude",
            )
        else:
            exclude_tolerance_s = (
                args.exclude_tolerance_s
                if args.exclude_tolerance_s is not None
                else _DEFAULT_EXCLUDE_TOLERANCE_S
            )
            exclusion_mask = _exclusion_mask(
                prepared.grid, starts_ns, ends_ns, exclude_tolerance_s
            )
            exclusion_note = (
                f"- calibration exclusion (spec A2.3.3): {int(exclusion_mask.sum())} "
                f"window(s) inside {len(starts_ns)} event interval(s) from "
                f"`{args.exclude_calibration_events}` (excluded with a "
                f"±{exclude_tolerance_s:g} s pad) are banned from the calibration "
                f"side and scored against the event-free thresholds instead"
            )

    rolling_minutes: float | None = None
    if args.thresholds == "frozen":
        if args.alpha is not None:
            logger.warning(
                "monitor: --alpha %s ignored -- frozen mode applies the snapshot's "
                "stored thresholds exactly as calibrated at fit time (alpha=%s)",
                args.alpha, snapshot.alpha,
            )
        alpha_used = snapshot.alpha
        verdicts = _frozen_verdicts(
            prepared, snapshot, labels,
            scoring_features=scoring_features, session_stats=session_stats,
        )
    elif args.thresholds == "rolling":
        alpha_used = args.alpha if args.alpha is not None else snapshot.alpha
        rolling_minutes = (
            args.rolling_minutes
            if args.rolling_minutes is not None
            else _DEFAULT_ROLLING_MINUTES
        )
        verdicts = _rolling_verdicts(
            prepared, snapshot, labels, alpha_used, rolling_minutes=rolling_minutes,
            scoring_features=scoring_features, session_stats=session_stats,
            exclusion_mask=exclusion_mask,
        )
    else:
        alpha_used = args.alpha if args.alpha is not None else snapshot.alpha
        verdicts = _recalibrate_verdicts(
            prepared, snapshot, labels, alpha_used,
            scoring_features=scoring_features, session_stats=session_stats,
            exclusion_mask=exclusion_mask,
        )

    _assert_roles_complete(verdicts.role, prepared.valid_mask)

    # near_transition is computed for every run
    # (ALWAYS-present column); --suppress-transition-alarms then optionally
    # forces alarm=False on near-transition SCORED windows BEFORE any output is
    # built, so segments/alarms/alarm_segments/notes all see the suppressed set.
    near_transition = _near_transition_mask(
        labels, prepared.valid_mask, prepared.grid.window_ns, snapshot.min_dwell_s
    )
    n_suppressed = (
        _apply_transition_suppression(verdicts.alarm, near_transition, verdicts.role)
        if args.suppress_transition_alarms
        else None
    )

    out_dir = args.out if args.out is not None else cfg.results_root / "monitor" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = _state_segments(labels, prepared.grid, snapshot.state_names)
    segments.to_csv(out_dir / "segments.csv", index=False)
    (out_dir / "timeline.md").write_text(_timeline_markdown(args.run, snapshot, segments))

    alarms = _alarms_frame(prepared, labels, verdicts, near_transition, snapshot.state_names)
    alarms.to_parquet(out_dir / "alarms.parquet", engine="pyarrow", index=False)

    _alarm_segments(verdicts.alarm, prepared.grid).to_csv(
        out_dir / "alarm_segments.csv", index=False
    )

    (out_dir / "monitor_notes.md").write_text(
        _notes_markdown(
            args.run, args.snapshot, snapshot, args.thresholds, alpha_used,
            verdicts, prepared,
            session_stats=session_stats, run_stats=run_stats,
            level_recal_offsets_used=level_recal_offsets_used,
            rolling_minutes=rolling_minutes, exclusion_note=exclusion_note,
            suppressed_by_transition=n_suppressed,
        )
    )

    print(
        f"monitor: {args.run} ({snapshot.variant}, {args.thresholds}): "
        f"{int(verdicts.alarm.sum())} alarm window(s) over "
        f"{int(prepared.valid_mask.sum())} valid window(s) -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
