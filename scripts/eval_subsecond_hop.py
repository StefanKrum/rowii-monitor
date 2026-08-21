"""Sub-second-hop re-scoring readout: the COARSE (1.0 s hop) alarms already on
disk vs. the FINE (0.25 s hop) alarms produced by re-running the SAME pipeline
with `scripts/monitor.py --hop-s 0.25`.

Motivation. The monitor scores non-overlapping 1 s windows, but the 2026-07-08
hammer strikes land ~0.75 s apart, so pairs of strikes share a scoring window:
no window on the coarse grid holds one of those strikes ALONE, and a single
alarm can never be attributed to a single impulse. A 0.25 s hop keeps the window
DURATION at 1 s and only starts windows more often, which is enough for every
such pair to get windows containing exactly one of the two.

This script never runs the monitor and never recomputes an alarm -- it reads the
two alarms.parquet tables plus the seconds-level ground truth
(`docs/groundtruth/080726_strikes_seconds_{st,pu}.csv`) and reports:

- **Detection sweep** (`rowii.eval.per_strike.sweep_strike_detection`): TPR at
  impulse and physical granularity for every (tolerance, alpha) in
  `_TOLERANCES_S` x `_ALPHAS`, per (session, grid). Detection is always
  re-thresholded from `p_value`, never read off the baked-in `alarm` column
  (that module's own contract).
- **Single-strike windows**: how many marks are covered by an ALARMING scored
  window whose 1 s span contains ONLY that mark. This is the claim the fine hop
  exists to test, and it is a CONTAINMENT question (does the window's span hold
  exactly one mark), deliberately not the same criterion as
  `mark_detected`'s symmetric start-timestamp tolerance.

- **Shared-window recovery**: restricted to exactly the marks the coarse grid
  could never isolate (no single-strike window at all), how many gain one on the
  fine grid and how many of those windows actually alarm.

Outputs (`results/rescoring-subsecond/`): `per_strike_detection.csv`,
`single_strike_windows.csv`, `shared_window_recovery.csv`. `REPORT.md` is
written by hand alongside them.

Usage:
    cd repos/rowii-monitor && .venv/bin/python scripts/eval_subsecond_hop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rowii.eval.events import ROLE_SCORED  # noqa: E402
from rowii.eval.per_strike import sweep_strike_detection  # noqa: E402

_SESSIONS: tuple[str, ...] = ("st", "pu")
_ALPHAS: tuple[float, ...] = (0.01, 0.05)
_TOLERANCES_S: tuple[float, ...] = (0.25, 0.5, 1.0)

_WINDOW_S = 1.0
"""Window DURATION of BOTH grids (`rowii.config.WindowConfig.window_s`) -- the
hop changes only how often a window starts, never how long it is, so one
constant covers the coarse and the fine table alike."""
_NS_PER_S = 1_000_000_000

_COARSE_GRID = "coarse-1.00s"
_FINE_GRID = "fine-0.25s"

_OUT_DIR = _REPO_ROOT / "results" / "rescoring-subsecond"
_GROUNDTRUTH_DIR = _REPO_ROOT / "docs" / "groundtruth"


def _coarse_alarms_path(session: str) -> Path:
    """The committed once-calibrated RECALIBRATE alarms for one session -- read
    only, never rewritten by this study."""
    return (
        _REPO_ROOT / "results" / "step2" / "once-calibrated" / "audio-beats" / "monitor"
        / f"080726-{session}_strikes" / "recalibrate" / "alarms.parquet"
    )


def _fine_alarms_path(session: str) -> Path:
    return _OUT_DIR / f"080726-{session}_strikes" / "alarms.parquet"


def _marks(session: str) -> pd.DataFrame:
    """The seconds-level per-strike ground truth, `#`-comment lines skipped (the
    docs/groundtruth contract)."""
    return pd.read_csv(_GROUNDTRUTH_DIR / f"080726_strikes_seconds_{session}.csv", comment="#")


def _mark_ns(marks: pd.DataFrame) -> np.ndarray:
    """Sorted int64 UTC NANOseconds of every mark's `strike_utc`.

    The explicit `datetime64[ns, UTC]` cast is load-bearing: `to_datetime` on
    these CSVs lands on `datetime64[us, UTC]` (pandas 2.x keeps the source's own
    microsecond resolution), and casting THAT straight to `int64` silently
    yields MICROseconds -- a 1000x-off axis that makes every window/mark
    comparison below vacuously empty rather than loudly wrong.
    """
    ts = pd.to_datetime(marks["strike_utc"], format="ISO8601", utc=True)
    return np.sort(ts.astype("datetime64[ns, UTC]").astype("int64").to_numpy())


def _marks_per_window(starts_ns: np.ndarray, mark_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(count, first_mark_idx)` per window: how many marks fall inside the
    window's own `[t_utc_ns, t_utc_ns + window_s)` span, and the index (into the
    SORTED *mark_ns*) of the earliest of them. `first_mark_idx` is meaningful
    only where `count > 0`."""
    window_ns = int(round(_WINDOW_S * _NS_PER_S))
    lo = np.searchsorted(mark_ns, starts_ns, side="left")
    hi = np.searchsorted(mark_ns, starts_ns + window_ns, side="left")
    return hi - lo, lo


def _single_strike_stats(
    alarms: pd.DataFrame, mark_ns: np.ndarray, *, alpha: float
) -> dict[str, int]:
    """Single-strike-window accounting for one (alarms table, alpha).

    A window is a SINGLE-STRIKE window for mark *m* iff its 1 s span contains
    *m* and no other mark of the session. Only SCORED windows count (the role
    contract every consumer of these tables shares -- a calibration-consumed
    window carries no verdict), and a window alarms iff `p_value < alpha`,
    re-thresholded here rather than read off the stored `alarm` column.
    """
    scored = alarms[alarms["role"] == ROLE_SCORED]
    starts = scored["t_utc_ns"].to_numpy(dtype=np.int64)
    alarming = scored["p_value"].to_numpy(dtype=np.float64) < alpha

    counts, first_idx = _marks_per_window(starts, mark_ns)
    covered = counts > 0
    single = counts == 1

    marks_with_single_window = set(first_idx[single].tolist())
    marks_detected_single = set(first_idx[single & alarming].tolist())
    marks_covered: set[int] = set()
    for w in np.flatnonzero(covered).tolist():
        marks_covered.update(range(int(first_idx[w]), int(first_idx[w] + counts[w])))

    return {
        "n_marks": int(mark_ns.size),
        "n_scored_windows": int(starts.size),
        "n_alarming_windows": int(alarming.sum()),
        "n_windows_covering_a_mark": int(covered.sum()),
        "n_windows_with_exactly_one_mark": int(single.sum()),
        "n_windows_with_two_or_more_marks": int((counts >= 2).sum()),
        "n_marks_covered_by_any_window": len(marks_covered),
        "n_marks_with_a_single_strike_window": len(marks_with_single_window),
        "n_marks_detected_via_single_strike_window": len(marks_detected_single),
    }


def _single_strike_mark_sets(
    alarms: pd.DataFrame, mark_ns: np.ndarray, *, alpha: float
) -> tuple[set[int], set[int]]:
    """`(marks with a single-strike SCORED window, marks with a single-strike
    ALARMING window)` -- the index sets `_single_strike_stats` only counts."""
    scored = alarms[alarms["role"] == ROLE_SCORED]
    starts = scored["t_utc_ns"].to_numpy(dtype=np.int64)
    alarming = scored["p_value"].to_numpy(dtype=np.float64) < alpha
    counts, first_idx = _marks_per_window(starts, mark_ns)
    single = counts == 1
    return set(first_idx[single].tolist()), set(first_idx[single & alarming].tolist())


def _shared_pair_recovery(
    coarse: pd.DataFrame, fine: pd.DataFrame, mark_ns: np.ndarray, *, alpha: float
) -> dict[str, int]:
    """What the fine hop buys for exactly the marks the coarse grid could never
    isolate: marks with NO single-strike window on the 1 s grid, how many of them
    gain one on the 0.25 s grid, and how many of those windows actually alarm."""
    coarse_single, _ = _single_strike_mark_sets(coarse, mark_ns, alpha=alpha)
    fine_single, fine_alarming = _single_strike_mark_sets(fine, mark_ns, alpha=alpha)
    shared = set(range(int(mark_ns.size))) - coarse_single
    return {
        "n_marks": int(mark_ns.size),
        "n_marks_without_a_coarse_single_strike_window": len(shared),
        "n_recovered_single_strike_window_on_fine_grid": len(shared & fine_single),
        "n_recovered_and_alarming_on_fine_grid": len(shared & fine_alarming),
    }


def _detection_rows(
    session: str, grid: str, alarms: pd.DataFrame, marks: pd.DataFrame
) -> pd.DataFrame:
    sweep = sweep_strike_detection(
        alarms, marks, tolerances_s=_TOLERANCES_S, alphas=_ALPHAS
    )
    sweep.insert(0, "grid", grid)
    sweep.insert(0, "session", session)
    return sweep


def main() -> int:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    detection_parts: list[pd.DataFrame] = []
    single_rows: list[dict[str, object]] = []
    recovery_rows: list[dict[str, object]] = []

    for session in _SESSIONS:
        marks = _marks(session)
        mark_ns = _mark_ns(marks)
        tables: dict[str, pd.DataFrame] = {}
        for grid, path in (
            (_COARSE_GRID, _coarse_alarms_path(session)),
            (_FINE_GRID, _fine_alarms_path(session)),
        ):
            if not path.is_file():
                print(f"eval_subsecond_hop: missing {path} -- skipped", file=sys.stderr)
                continue
            alarms = pd.read_parquet(path, engine="pyarrow")
            tables[grid] = alarms
            detection_parts.append(_detection_rows(session, grid, alarms, marks))
            for alpha in _ALPHAS:
                row: dict[str, object] = {
                    "session": session, "grid": grid, "alpha": alpha,
                    "alarms_path": str(path.relative_to(_REPO_ROOT)),
                }
                row.update(_single_strike_stats(alarms, mark_ns, alpha=alpha))
                single_rows.append(row)

        if _COARSE_GRID in tables and _FINE_GRID in tables:
            for alpha in _ALPHAS:
                rec: dict[str, object] = {"session": session, "alpha": alpha}
                rec.update(
                    _shared_pair_recovery(
                        tables[_COARSE_GRID], tables[_FINE_GRID], mark_ns, alpha=alpha
                    )
                )
                recovery_rows.append(rec)

    detection = pd.concat(detection_parts, ignore_index=True)
    detection.to_csv(_OUT_DIR / "per_strike_detection.csv", index=False)
    single = pd.DataFrame(single_rows)
    single.to_csv(_OUT_DIR / "single_strike_windows.csv", index=False)
    recovery = pd.DataFrame(recovery_rows)
    recovery.to_csv(_OUT_DIR / "shared_window_recovery.csv", index=False)

    overall = detection[
        (detection["kind_group"] == "ALL") & (detection["granularity"] == "impulse")
    ]
    print(overall.to_string(index=False))
    print()
    print(single.drop(columns=["alarms_path"]).to_string(index=False))
    print()
    print(recovery.to_string(index=False))
    for name in ("per_strike_detection.csv", "single_strike_windows.csv",
                 "shared_window_recovery.csv"):
        print(f"wrote {_OUT_DIR / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
