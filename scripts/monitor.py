"""Runtime monitor CLI (Step-2 package-6 design spec `docs/superpowers/specs/
2026-07-16-step2-package6-runtime-pillar3-design.md` D2 + amendment A1.3, plan
`docs/superpowers/plans/2026-07-16-step2-package6-runtime-pillar3.md` Task 2):
apply a persisted `MonitorSnapshot` (fitted detector + per-state references +
conformal thresholds, `rowii.runtime.snapshot`) to a NEW recording, emitting a
state timeline plus per-window conformal alarm verdicts -- the design chapter's
"runs at the plant" requirement as a batch CLI over recorded files (the deployment
model the spec commits to; no streaming, no retraining, no refit of anything).

Pipeline: `prepare_run` (feature cache honored unless `--no-cache`) -> geometry
guard (the prepared run's `feature_names` must EQUAL the snapshot's, else exit 2
naming the variant and both widths -- a snapshot fitted on `fusion` must never
silently score some other variant's features) -> `to_detector(snapshot).apply` on
the valid windows (fit-day standardization + fit-day HMM Viterbi decode, scattered
back to the full grid with `-1` on invalid windows) -> per-state scoring against
the snapshot's OWN fit-day references, thresholded in one of two modes:

- `--thresholds recalibrate` (DEFAULT): split the new run's valid windows by
  segments at the snapshot's own `(calibration_frac, seed)`, recompute each
  snapshot-known state's conformal threshold on the calibration-side windows of
  that state at `--alpha` (default: the snapshot's fit-time alpha), and emit
  verdicts for the SCORING-side windows only. This operationalizes package-2's
  central cross-day finding: "transfer detector + references, recalibrate
  thresholds per day" is the only recipe whose realized false-alarm rate held its
  nominal alpha. References are NEVER refit here -- recalibrate mode touches
  thresholds only (spec §4). The calibration-side windows themselves are consumed
  by the threshold fit and are NEVER alarmed (`role="consumed_for_calibration"` --
  the calibration-bias rule, A1.3: alarming the very windows a threshold was fitted
  on would break the conformal exchangeability argument in the anti-conservative
  direction).
- `--thresholds frozen`: apply the fit day's stored thresholds unchanged to every
  valid window of a snapshot-known state. Reported with an explicit
  distribution-shift warning in the notes: in package-2's cross-day evidence,
  frozen cross-day thresholds did NOT hold their nominal FAR.

Per-state semantics on the new run (A1.3, binding): a valid window's detected
state can be absent from the snapshot (no reference/threshold survived fitting) --
such windows get `role="unknown_state"`, are never alarmed, and are counted in the
notes. In recalibrate mode a snapshot-known state with ZERO calibration-side
windows on the new run cannot be recalibrated -- ALL its windows get
`role="no_conformal_data"`, no verdicts, and a notes row (never a silent fallback
to the frozen fit-day threshold, which would smuggle exactly the un-recalibrated
behavior the mode exists to avoid).

Outputs under `--out` (default `results/monitor/<run>/`): `segments.csv` +
`timeline.md` (the state half, `scripts/apply_detector.py`'s conventions incl.
scatter-back over `valid_mask`), `alarms.parquet` (one row per VALID window:
`window, t_utc_ns, state, score, p_value, alarm, low_confidence, role`),
`alarm_segments.csv` (maximal alarm runs: `start_utc, end_utc, duration_s`), and
`monitor_notes.md` (snapshot provenance, mode, per-state table, window accounting,
and the standing honesty framing: NO fault labels exist -- alarms are candidates
for operator review, never verified detections; spec §4).

Bootstrapping (config/env via `rowii.config.load_config`, run discovery via
`rowii.io.dataset.discover`, unknown-run exit 2 listing every discovered run)
mirrors `scripts/warm_cache.py`'s skeleton. Small helpers shared with sibling
scripts are DUPLICATED, not imported -- a script must not depend on a SIBLING
script's internals (`scripts/apply_detector.py`'s module docstring states the
rule); `src/rowii/` modules are imported normally.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.anomaly.conformal import calibrate, p_values  # noqa: E402
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

_THRESHOLD_MODES: tuple[str, ...] = ("recalibrate", "frozen")

_ALARM_COLUMNS: tuple[str, ...] = (
    "window", "t_utc_ns", "state", "score", "p_value", "alarm", "low_confidence", "role",
)
"""alarms.parquet's exact column contract (spec D2 / plan Task 2), in this order."""

ROLE_SCORED = "scored"
"""A window with a real verdict: its state's threshold was applied to its score."""
ROLE_CONSUMED = "consumed_for_calibration"
"""Recalibrate mode only: a calibration-side window whose score parameterized its
state's recalibrated threshold -- never alarmed (calibration-bias rule, A1.3)."""
ROLE_UNKNOWN_STATE = "unknown_state"
"""Detected state has no reference/threshold in the snapshot -- never alarmed."""
ROLE_NO_CONFORMAL_DATA = "no_conformal_data"
"""Recalibrate mode only: the state is snapshot-known but has zero calibration-side
windows on this run, so no per-day threshold exists -- never alarmed (A1.3)."""

_FROZEN_SHIFT_WARNING = (
    "**Distribution-shift warning (frozen thresholds):** the fit day's conformal "
    "thresholds were applied UNCHANGED to this recording. In package-2's cross-day "
    "evidence, frozen cross-day thresholds did NOT hold their nominal false-alarm "
    "rate -- realized FAR drifted far from alpha across days -- so the alarm counts "
    "below carry NO per-day false-alarm guarantee. `--thresholds recalibrate` (the "
    "default) is the recipe whose FAR held."
)
"""The notes' verbatim frozen-mode warning (spec D2: reported with an explicit
distribution-shift warning; the binding phrase is "did NOT hold")."""


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
             "thresholds did NOT hold their FAR).",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Recalibrate-mode nominal false-alarm target in (0, 1); default: the "
             "snapshot's own fit-time alpha. Ignored (with a warning) in frozen "
             "mode -- frozen thresholds are applied exactly as stored.",
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
    """The applied threshold's low-confidence flag; None when no threshold exists."""
    status: str
    """`"scored"` or `"no_conformal_data"` (per-state, not per-window)."""


@dataclass(frozen=True)
class _Verdicts:
    """Full-grid per-window verdict arrays + notes bookkeeping. Invalid windows keep
    the defaults everywhere (NaN score/p_value, False alarm, empty role) and are
    excluded from alarms.parquet; `_assert_roles_complete` guarantees every VALID
    window received exactly one role."""

    score: np.ndarray
    """(W,) float64 -- NaN where no score was computed (invalid/unknown/no-data)."""
    p_value: np.ndarray
    """(W,) float64 -- NaN everywhere except scored windows (verdict-only, A1.3)."""
    alarm: np.ndarray
    """(W,) bool -- True only on scored windows over their state's threshold."""
    low_confidence: np.ndarray
    """(W,) bool -- True only on scored windows whose applied threshold is
    low-confidence (threshold=+inf, can never alarm)."""
    role: np.ndarray
    """(W,) object -- one of the ROLE_* strings on valid windows, "" on invalid."""
    state_rows: list[_StateRow]
    unknown_counts: dict[int, int]
    """Detected-but-not-snapshot-known state id -> valid-window count."""


def _empty_verdict_arrays(
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score = np.full(n, math.nan, dtype=np.float64)
    p_value = np.full(n, math.nan, dtype=np.float64)
    alarm = np.zeros(n, dtype=bool)
    low_confidence = np.zeros(n, dtype=bool)
    role = np.full(n, "", dtype=object)
    return score, p_value, alarm, low_confidence, role


def _mark_unknown_states(
    prepared: PreparedRun, snapshot: MonitorSnapshot, labels: np.ndarray, role: np.ndarray
) -> dict[int, int]:
    """Tag every valid window whose detected state has no snapshot threshold with
    `ROLE_UNKNOWN_STATE` (in place) -- returns `{state id: window count}` for the
    notes (A1.3: such windows get no verdict and are counted, never alarmed)."""
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
    prepared: PreparedRun, snapshot: MonitorSnapshot, labels: np.ndarray
) -> _Verdicts:
    """Frozen mode (spec D2): every valid window of a snapshot-known state gets a
    verdict against the fit day's STORED threshold; p-values against the fit day's
    stored calibration scores (the same set the threshold came from)."""
    score, p_value, alarm, low_confidence, role = _empty_verdict_arrays(
        prepared.features.shape[0]
    )
    state_rows: list[_StateRow] = []
    for label in sorted(snapshot.thresholds):
        threshold = snapshot.thresholds[label]
        idx = np.flatnonzero(prepared.valid_mask & (labels == label))
        n_alarms = 0
        if idx.size:
            scores = scorer_for_label(snapshot, label).score(prepared.features[idx])
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
        role=role, state_rows=state_rows, unknown_counts=unknown_counts,
    )


def _recalibrate_verdicts(
    prepared: PreparedRun, snapshot: MonitorSnapshot, labels: np.ndarray, alpha: float
) -> _Verdicts:
    """Recalibrate mode (DEFAULT, spec D2 + A1.3): top split of the new run's valid
    windows at the snapshot's own `(calibration_frac, seed)`; per snapshot-known
    state, a fresh conformal threshold from the calibration-side windows of that
    state (references stay the SNAPSHOT's -- thresholds only), verdicts for the
    scoring-side windows only. Calibration-side windows are consumed, never alarmed
    (calibration-bias rule); a state with zero calibration-side windows takes the
    `no_conformal_data` path for ALL its windows."""
    score, p_value, alarm, low_confidence, role = _empty_verdict_arrays(
        prepared.features.shape[0]
    )
    top = split_by_segments(
        prepared.segment_ids, prepared.valid_mask, snapshot.calibration_frac, snapshot.seed
    )
    cal_windows, scoring_windows = top.calibration_windows, top.scoring_windows

    state_rows: list[_StateRow] = []
    for label in sorted(snapshot.thresholds):
        label_all = np.flatnonzero(prepared.valid_mask & (labels == label))
        label_cal = cal_windows[labels[cal_windows] == label]
        label_scr = scoring_windows[labels[scoring_windows] == label]

        if label_cal.size == 0:
            # A1.3: no per-day threshold can be calibrated -> no verdicts at all for
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

        scorer = scorer_for_label(snapshot, label)
        cal_scores = scorer.score(prepared.features[label_cal])
        # Deliberately NO `min_ref` gate here (T2-review finding, resolved as a
        # documented reading of A1.3's "identical to sweeps"): `run_sweep` gates
        # the FIT side with `min_ref` (reference quality -- enforced for this
        # snapshot at BUILD time by `fit_snapshot`) and governs the conformal/
        # calibration side ONLY by `calibrate`'s own achievable-alpha floor
        # (`low_confidence`), zero-count aside. A tiny calibration side therefore
        # yields a VALID conformal threshold whose sample count is visible in the
        # notes' per-state `n_consumed` column and whose confidence is carried by
        # `low_confidence` -- exactly the sweeps' semantics, not a second gate.
        threshold = calibrate(cal_scores, alpha)
        # Calibration-bias rule (A1.3): scores are recorded for provenance (they ARE
        # the threshold's empirical distribution), but no p-value, never an alarm.
        score[label_cal] = cal_scores
        role[label_cal] = ROLE_CONSUMED

        n_alarms = 0
        if label_scr.size:
            scores = scorer.score(prepared.features[label_scr])
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
        role=role, state_rows=state_rows, unknown_counts=unknown_counts,
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
# Outputs
# ---------------------------------------------------------------------------


def _alarms_frame(
    prepared: PreparedRun, labels: np.ndarray, verdicts: _Verdicts
) -> pd.DataFrame:
    """One row per VALID window, ascending, exactly `_ALARM_COLUMNS`. `t_utc_ns` is
    the window's grid left edge (`rowii.anomaly.overlap.to_utc_ns`'s own
    `t0_ns + window * window_ns` convention, reused directly -- it lives in `src`,
    not in a sibling script)."""
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


def _state_segments(labels: np.ndarray, grid: WindowGrid) -> pd.DataFrame:
    """`segments.csv`'s table: `to_segments` over the full-length label array
    (`_INVALID_LABEL` segments included, visibly), `cluster` renamed to
    `cluster_id` -- `scripts/apply_detector.py`'s convention (minus its
    `mapped_mode`: the snapshot carries no cluster-id -> mode-name mapping, and
    inventing one here would be a claim the artifact cannot back)."""
    return to_segments(labels, grid).rename(columns={"cluster": "cluster_id"})


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
            desc = f"state {cluster_id}"
        else:
            desc = f"state {cluster_id} (unknown to the snapshot -- never alarmed)"
        lines.append(
            f"- {row['start_utc'].isoformat()} -> {row['end_utc'].isoformat()} "
            f"({row['duration_s']:.1f}s): {desc}"
        )
    lines.append("")
    return "\n".join(lines)


def _notes_markdown(
    run_name: str,
    snapshot_path: Path,
    snapshot: MonitorSnapshot,
    mode: str,
    alpha_used: float,
    verdicts: _Verdicts,
    prepared: PreparedRun,
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
        lines.append(
            f"| {row.state} | {row.n_windows} | {row.n_scored} | {row.n_alarms} "
            f"| {rate} | {low_conf} | {threshold} | {row.n_consumed} | {row.status} |"
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
        + (" (recalibrate-mode calibration side; never alarmed)" if mode == "recalibrate" else ""),
        f"- {ROLE_UNKNOWN_STATE}: {n_unknown} (detected state has no snapshot "
        "reference/threshold; never alarmed)",
        f"- {ROLE_NO_CONFORMAL_DATA}: {n_no_conformal} (snapshot-known state with "
        "zero calibration-side windows on this run; never alarmed)",
    ]
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
    if not args.snapshot.is_file():
        print(f"monitor: snapshot file not found: {args.snapshot}", file=sys.stderr)
        return 2
    try:
        snapshot = load_snapshot(args.snapshot)
    except ValueError as exc:
        print(f"monitor: cannot load snapshot {args.snapshot}: {exc}", file=sys.stderr)
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
        print(
            f"monitor: feature geometry mismatch for run {args.run!r}: the snapshot "
            f"was fitted on variant {snapshot.variant!r} with "
            f"{len(snapshot.feature_names)} feature column(s), but the prepared run "
            f"has {len(prepared.feature_names)} -- refusing to score "
            f"(snapshot columns: {snapshot.feature_names}; prepared columns: "
            f"{prepared.feature_names})",
            file=sys.stderr,
        )
        return 2

    detector = to_detector(snapshot)
    labels = _apply_detector_labels(prepared, detector)

    if args.thresholds == "frozen":
        if args.alpha is not None:
            logger.warning(
                "monitor: --alpha %s ignored -- frozen mode applies the snapshot's "
                "stored thresholds exactly as calibrated at fit time (alpha=%s)",
                args.alpha, snapshot.alpha,
            )
        alpha_used = snapshot.alpha
        verdicts = _frozen_verdicts(prepared, snapshot, labels)
    else:
        alpha_used = args.alpha if args.alpha is not None else snapshot.alpha
        verdicts = _recalibrate_verdicts(prepared, snapshot, labels, alpha_used)

    _assert_roles_complete(verdicts.role, prepared.valid_mask)

    out_dir = args.out if args.out is not None else cfg.results_root / "monitor" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = _state_segments(labels, prepared.grid)
    segments.to_csv(out_dir / "segments.csv", index=False)
    (out_dir / "timeline.md").write_text(_timeline_markdown(args.run, snapshot, segments))

    alarms = _alarms_frame(prepared, labels, verdicts)
    alarms.to_parquet(out_dir / "alarms.parquet", engine="pyarrow", index=False)

    _alarm_segments(verdicts.alarm, prepared.grid).to_csv(
        out_dir / "alarm_segments.csv", index=False
    )

    (out_dir / "monitor_notes.md").write_text(
        _notes_markdown(
            args.run, args.snapshot, snapshot, args.thresholds, alpha_used,
            verdicts, prepared,
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
