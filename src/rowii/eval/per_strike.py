"""Per-strike detection and first-alarm latency: seconds-level hammer-strike
ground truth vs. per-window alarms.

Companion to `rowii.eval.events` (event/interval-level TPR + latency): this
module works at the SINGLE-INSTANT ("mark") level, against the seconds-level
ground truth compiled by `scripts/annotation_kit.py compile-marks`
(`docs/groundtruth/080726_strikes_seconds_{st,pu}.csv` -- columns `session,
event_id, kind, strike_no, strike_utc, confidence, notes`; one row per
individual hammer-strike/rebound impulse the annotator marked, NOT one row per
labeled interval).

Input contracts:

- **alarms**: one row per window, the `scripts/monitor.py` / `scripts/
  run_step2.py` / pillar-3 `alarms.parquet` shape -- required columns
  `t_utc_ns` (int64 window-START UTC nanoseconds) and `p_value` (float64). If
  a `role` column is present, rows with `role != "scored"` (`rowii.eval.
  events.ROLE_SCORED`) are dropped FIRST, mirroring `rowii.eval.events`'
  own role-filtering contract -- calibration-consumed / unknown-state /
  no-conformal-data windows carry no verdict.
- **marks**: one row per strike-impulse mark -- required columns `session`,
  `event_id`, `kind`, `strike_no` (1-based within its event, chronological),
  `strike_utc` (timezone-AWARE ISO-8601 or an already-parsed timestamp; naive
  values raise `ValueError`, same refusal as `rowii.eval.events`).

Semantics (every edge pinned by `tests/test_per_strike.py`):

- **Detection** (`mark_detected`): a mark counts as detected at level *alpha*
  and tolerance *T* iff at least one SCORED window with `p_value < alpha` has
  a window-START timestamp within `T` seconds of the mark, on EITHER side
  (`|t_window_start - t_mark| <= T`). Windows are compared by their START
  timestamp only (this project's `t_utc_ns` convention throughout, e.g.
  `rowii.eval.events`) -- window duration is never used to widen the match.
  `p_value < alpha` is evaluated directly against the stored per-window
  p-value, NEVER against a `alarm` column baked in at one fixed alpha (a
  `monitor.py`/pillar-3 alarms.parquet's own `alarm` column reflects whatever
  alpha it was written at; this module re-thresholds from `p_value` so any
  alpha can be swept from one alarms table).
- **Double-impulse dedup** (`deduplicate_marks`): consecutive marks of the
  SAME event (`session`, `event_id`) whose gap to the PREVIOUS mark is
  STRICTLY LESS than `gap_s` (default 1.5 s) are folded into one "physical
  strike" -- a gap of EXACTLY `gap_s` starts a new physical strike ("closer
  than", not "at most"). The physical strike's `strike_utc`/`strike_no` are
  the FIRST raw mark's (the group's earliest mark identifies the strike);
  `n_impulses` counts the raw marks folded in, `last_strike_no` is the last
  raw mark's `strike_no` (traceability). Grouping never crosses an event
  boundary, however close two marks from different events are in time.
- **First-alarm latency** (`evaluate_event_latency` / `evaluate_strike_
  latency`): for a reference instant (an event's onset -- its EARLIEST raw
  mark -- or a deduplicated physical strike), latency is the first SCORED,
  `p_value < alpha` window's start AT-OR-AFTER (`>=`) the reference, minus the
  reference, in seconds (always >= 0, unlike `rowii.eval.events`' symmetric
  tolerance pad -- there is no "before onset" case here). A reference with no
  qualifying window within `search_horizon_s` afterward is `missed` (`latency_
  s` is NaN); the two callers use different horizons (60 s per event, 5 s per
  physical strike) per the campaign's own strike cadence.
- **Kind groups**: `kind_group` maps every observed `kind` string onto the
  fixed four-way taxonomy `plate-gen` / `plate-tur` / `landmark` / `vane-
  sweep` (an unrecognized `kind` raises loudly -- exhaustive by construction
  against the campaign's fixed strike vocabulary). `sweep_strike_detection`'s
  tidy output always carries exactly five kind_group rows per (granularity,
  tolerance_s, alpha) combination -- the four groups plus `"ALL"` -- with
  `n_marks=0`/`tpr=NaN` for a group absent from the given input, so every
  sweep produces the SAME row shape regardless of which representation or
  session is being processed (a stable schema across the full sweep grid).
- **Binary / pre-thresholded alarm streams** (`mark_detected_binary`,
  `sweep_strike_detection_binary`, `evaluate_strike_latency_binary`): a
  companion path for alarms tables that carry an already-thresholded bool
  column (`t_utc_ns` + a named bool column, e.g. `scripts/run_mad_baseline.
  py`'s own `_alarms_frame` shape) instead of a continuous `p_value` to
  re-threshold. The fixed-threshold MAD baseline is the motivating case: its
  threshold is commissioned ONCE on a disjoint pool and applied once
  (`score > threshold`), so there is no p-value to sweep and no `alpha`
  argument to take. These functions mirror `mark_detected`/
  `sweep_strike_detection`/`evaluate_strike_latency` EXACTLY -- both paths
  share the same private `_match_within_tolerance`/`_first_alarm_latency`
  matching/latency arithmetic -- with the p_value/alpha axis replaced by a
  caller-named boolean `alarm_column`. This is a genuine second code path,
  not a `p_value` encoding trick: a fake `p_value` of `0.0`/`1.0` for alarm/
  no-alarm would make the EXISTING alpha-based functions run, but `alpha`
  would then be purely decorative (identical result for every `alpha` in
  `(0, 1]`) -- indistinguishable from a real p-value at the call site, and
  this module refuses that shortcut. There is no `evaluate_event_latency_
  binary` (no caller needs event-level latency for a pre-thresholded stream
  yet) -- add one the same way if one does.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rowii.eval.events import ROLE_SCORED

_NS_PER_S = 1_000_000_000

_KIND_GROUP_PLATE_GEN = "plate-gen"
_KIND_GROUP_PLATE_TUR = "plate-tur"
_KIND_GROUP_LANDMARK = "landmark"
_KIND_GROUP_VANE_SWEEP = "vane-sweep"
_KIND_GROUPS_ORDER: tuple[str, ...] = (
    _KIND_GROUP_PLATE_GEN, _KIND_GROUP_PLATE_TUR, _KIND_GROUP_LANDMARK, _KIND_GROUP_VANE_SWEEP,
)
"""The four-way strike taxonomy, in the campaign's own reporting order (module
docstring; task order: "plate-gen, plate-tur, landmark, vane-sweep")."""

_ALL_KIND_GROUP = "ALL"
"""Sentinel `kind_group` value for the overall (all-groups-pooled) row that
`sweep_strike_detection` appends alongside the four per-group rows."""

_REQUIRED_MARK_COLUMNS: tuple[str, ...] = ("session", "event_id", "kind", "strike_no", "strike_utc")

_DEDUP_COLUMNS: tuple[str, ...] = (
    "session", "event_id", "kind", "strike_no", "strike_utc", "n_impulses", "last_strike_no",
)
_EVENT_ONSET_COLUMNS: tuple[str, ...] = ("session", "event_id", "kind", "kind_group", "onset_utc")
_GAP_COLUMNS: tuple[str, ...] = (
    "session", "event_id", "kind", "kind_group", "to_strike_no", "gap_s",
)
_MARKS_PER_EVENT_COLUMNS: tuple[str, ...] = ("session", "event_id", "kind", "kind_group", "n_marks")
_SWEEP_COLUMNS: tuple[str, ...] = (
    "granularity", "tolerance_s", "alpha", "kind_group", "n_marks", "n_detected", "tpr",
)
_GRANULARITY_IMPULSE = "impulse"
_GRANULARITY_PHYSICAL = "physical"


def kind_group(kind: str) -> str:
    """*kind* (a marks-table `kind` value, e.g. `"plate-gen_90"`, `"landmark-
    B_11TG"`, `"vane-sweep"`) mapped onto the fixed four-way taxonomy (module
    docstring). Matching is by PREFIX for `plate-gen`/`plate-tur`/`landmark`
    (the campaign's own `<group>_<position>` naming) and exact for `vane-
    sweep` (no position suffix). Raises `ValueError` for any other string --
    exhaustive against the campaign's fixed strike-class vocabulary
    (`docs/groundtruth/080726_events_{st,pu}.csv`'s own "Strike classes"
    comment), so a typo'd or new `kind` fails loudly instead of silently
    landing in the wrong group.
    """
    if kind == _KIND_GROUP_VANE_SWEEP:
        return _KIND_GROUP_VANE_SWEEP
    if kind.startswith("plate-gen"):
        return _KIND_GROUP_PLATE_GEN
    if kind.startswith("plate-tur"):
        return _KIND_GROUP_PLATE_TUR
    if kind.startswith("landmark"):
        return _KIND_GROUP_LANDMARK
    raise ValueError(
        f"unrecognized strike kind {kind!r}; expected 'vane-sweep' or a "
        f"'plate-gen_*' / 'plate-tur_*' / 'landmark-*' variant"
    )


def _parse_utc_ns(values: pd.Series, column: str) -> np.ndarray:
    """*values* (ISO-8601 strings or already-parsed timestamps) as int64 UTC
    nanoseconds. Row-by-row via `pd.Timestamp` (never the vectorized `pd.
    to_datetime(series)` fast path, which infers ONE strptime format from the
    column and then raises on any row that doesn't match it byte-for-byte --
    the real ground-truth CSVs mix `...ss.ffffff+00:00` and bare `...ss+00:00`
    rows, e.g. `docs/groundtruth/080726_strikes_seconds_st.csv` event 09
    strike_no 13). Naive or missing entries raise `ValueError` (mirrors
    `rowii.eval.events._parse_utc_ns` by contract, kept as an independent
    copy so this module has no non-`rowii.eval.events`-constant coupling to
    that module's internals).
    """
    out = np.empty(len(values), dtype=np.int64)
    for i, raw in enumerate(values):
        try:
            ts = pd.Timestamp(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{column!r} row {i}: cannot parse {raw!r} as a timestamp"
            ) from exc
        if pd.isna(ts):
            raise ValueError(f"{column!r} row {i}: missing timestamp")
        if ts.tzinfo is None:
            raise ValueError(
                f"{column!r} row {i}: naive timestamp {raw!r} has no UTC offset -- "
                f"write ISO-8601 with an explicit offset (e.g. "
                f"'2026-07-08T10:15:02.243000+00:00' or a trailing 'Z')"
            )
        out[i] = ts.tz_convert("UTC").value
    return out


def _validate_alarms(alarms: pd.DataFrame) -> None:
    missing = [c for c in ("t_utc_ns", "p_value") if c not in alarms.columns]
    if missing:
        raise ValueError(
            f"alarms is missing required column(s) {missing}; expected the "
            f"monitor/pillar-3 alarms.parquet shape with 't_utc_ns' (int64 window "
            f"starts) and 'p_value' (float64) -- got columns {list(alarms.columns)}"
        )
    if not pd.api.types.is_integer_dtype(alarms["t_utc_ns"]):
        raise ValueError(
            f"alarms['t_utc_ns'] must be int64 UTC nanoseconds (window starts), "
            f"got dtype {alarms['t_utc_ns'].dtype}"
        )
    if not pd.api.types.is_float_dtype(alarms["p_value"]):
        raise ValueError(
            f"alarms['p_value'] must be float64, got dtype {alarms['p_value'].dtype}"
        )


def _validate_marks(marks: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_MARK_COLUMNS if c not in marks.columns]
    if missing:
        raise ValueError(
            f"marks is missing required column(s) {missing}; expected the "
            f"seconds-level strike ground-truth shape (docs/groundtruth/"
            f"080726_strikes_seconds_{{st,pu}}.csv) with columns "
            f"{_REQUIRED_MARK_COLUMNS} -- got columns {list(marks.columns)}"
        )


def _scored_alarming_times(alarms: pd.DataFrame, alpha: float) -> np.ndarray:
    """Sorted int64 UTC-ns window-START timestamps of every SCORED window with
    `p_value < alpha` in *alarms* (role filter per the module docstring)."""
    scored = alarms[alarms["role"] == ROLE_SCORED] if "role" in alarms.columns else alarms
    t = scored["t_utc_ns"].to_numpy(dtype=np.int64)
    p = scored["p_value"].to_numpy(dtype=np.float64)
    return np.sort(t[p < alpha])


def _check_tolerance_s(tolerance_s: float) -> None:
    if not (math.isfinite(tolerance_s) and tolerance_s >= 0):
        raise ValueError(f"tolerance_s must be >= 0 seconds, got {tolerance_s!r}")


def _check_alpha(alpha: float) -> None:
    if not (math.isfinite(alpha) and 0 < alpha <= 1):
        raise ValueError(f"alpha must satisfy 0 < alpha <= 1, got {alpha!r}")


def _match_within_tolerance(
    alarming_t: np.ndarray, mark_ns: np.ndarray, tol_ns: int
) -> np.ndarray:
    """Bool array aligned with *mark_ns* (module docstring's "Detection"
    bullet): `True` where the nearest entry of the SORTED *alarming_t*, on
    either side, lies within `tol_ns`. The shared tolerance-matching core of
    `mark_detected` (p_value/alpha path) and `mark_detected_binary`
    (pre-thresholded path) -- factored out so both run the EXACT same
    matching arithmetic regardless of how *alarming_t* was sourced."""
    detected = np.zeros(mark_ns.shape, dtype=bool)
    if alarming_t.size and mark_ns.size:
        idx_right = np.clip(np.searchsorted(alarming_t, mark_ns), 0, alarming_t.size - 1)
        idx_left = np.clip(idx_right - 1, 0, alarming_t.size - 1)
        dist_right = np.abs(alarming_t[idx_right] - mark_ns)
        dist_left = np.abs(alarming_t[idx_left] - mark_ns)
        detected = (dist_right <= tol_ns) | (dist_left <= tol_ns)
    return detected


def mark_detected(
    alarms: pd.DataFrame, marks: pd.DataFrame, *, tolerance_s: float, alpha: float
) -> np.ndarray:
    """Bool array aligned with *marks*' row order (never sorted or
    reindexed): `True` where at least one scored, `p_value < alpha` window's
    START timestamp lies within `tolerance_s` seconds of that mark's
    `strike_utc`, on either side (module docstring).

    Args:
        alarms: One row per window -- `t_utc_ns` (int64 UTC window-start ns),
            `p_value` (float64); rows with a `role` other than `"scored"` are
            dropped first (if a `role` column exists).
        marks: One row per strike-impulse mark -- `strike_utc` (tz-aware).
        tolerance_s: Half-width of the matching window in seconds (>= 0).
        alpha: Detection threshold; a window detects iff `p_value < alpha`
            (strict), so `0 < alpha <= 1`.

    Returns:
        `np.ndarray[bool]`, `len(marks)` entries, in input order.

    Raises:
        ValueError: On missing/mistyped alarms or marks columns, naive/
            unparseable `strike_utc`, `tolerance_s < 0`, or `alpha` outside
            `(0, 1]`.
    """
    _validate_alarms(alarms)
    _validate_marks(marks)
    _check_tolerance_s(tolerance_s)
    _check_alpha(alpha)

    alarming_t = _scored_alarming_times(alarms, alpha)
    mark_ns = _parse_utc_ns(marks["strike_utc"], "strike_utc")
    tol_ns = int(round(tolerance_s * _NS_PER_S))
    return _match_within_tolerance(alarming_t, mark_ns, tol_ns)


def deduplicate_marks(marks: pd.DataFrame, *, gap_s: float = 1.5) -> pd.DataFrame:
    """One row per "physical strike": consecutive marks of the SAME
    `(session, event_id)` whose gap to the previous mark is strictly less
    than `gap_s` seconds are folded together (module docstring's double-
    impulse rule). Columns: `session, event_id, kind, strike_no, strike_utc,
    n_impulses, last_strike_no` -- `strike_no`/`strike_utc` are the group's
    FIRST raw mark's ("first mark = the strike time"); `kind` is that same
    first mark's (constant within an event by construction); `n_impulses`
    is the count of raw marks folded in; `last_strike_no` is the group's
    LAST raw mark's `strike_no` (traceability back to the raw table).

    Args:
        marks: The raw (impulse-level) marks table (`_REQUIRED_MARK_COLUMNS`).
        gap_s: Grouping threshold in seconds (> 0); a gap of EXACTLY `gap_s`
            starts a new physical strike ("closer than", module docstring).

    Raises:
        ValueError: Missing/mistyped marks columns, naive/unparseable
            `strike_utc`, or non-positive `gap_s`.
    """
    _validate_marks(marks)
    if not (math.isfinite(gap_s) and gap_s > 0):
        raise ValueError(f"gap_s must be a positive number of seconds, got {gap_s!r}")

    working = marks.copy()
    working["_ns"] = _parse_utc_ns(working["strike_utc"], "strike_utc")
    working = working.sort_values(["session", "event_id", "_ns"], kind="stable").reset_index(
        drop=True
    )
    gap_ns = int(round(gap_s * _NS_PER_S))

    rows: list[dict[str, object]] = []
    for _, group in working.groupby(["session", "event_id"], sort=False):
        group = group.reset_index(drop=True)
        ns = group["_ns"].to_numpy(dtype=np.int64)
        new_strike = np.ones(len(group), dtype=bool)
        if len(group) > 1:
            new_strike[1:] = (ns[1:] - ns[:-1]) >= gap_ns
        strike_id = np.cumsum(new_strike) - 1
        for sid in np.unique(strike_id):
            members = group[strike_id == sid]
            first = members.iloc[0]
            rows.append(
                {
                    "session": first["session"],
                    "event_id": first["event_id"],
                    "kind": first["kind"],
                    "strike_no": first["strike_no"],
                    "strike_utc": pd.Timestamp(first["_ns"], tz="UTC", unit="ns"),
                    "n_impulses": int(len(members)),
                    "last_strike_no": members.iloc[-1]["strike_no"],
                }
            )
    return pd.DataFrame(rows, columns=list(_DEDUP_COLUMNS))


def _kind_group_summary(kinds: pd.Series, detected: np.ndarray) -> pd.DataFrame:
    """Tidy `(kind_group, n_marks, n_detected, tpr)` frame: one row per entry
    of `_KIND_GROUPS_ORDER` PLUS one `"ALL"` row -- always five rows,
    regardless of which groups actually occur in *kinds* (module docstring:
    a stable schema across the whole sweep). `tpr` is NaN when `n_marks == 0`
    (vacuous, mirrors `rowii.eval.events`'s TPR convention -- never 1.0)."""
    groups = kinds.map(kind_group).to_numpy()
    rows = []
    for group in (*_KIND_GROUPS_ORDER, _ALL_KIND_GROUP):
        mask = np.ones(groups.shape, dtype=bool) if group == _ALL_KIND_GROUP else groups == group
        n = int(mask.sum())
        n_detected = int(detected[mask].sum())
        rows.append(
            {
                "kind_group": group,
                "n_marks": n,
                "n_detected": n_detected,
                "tpr": n_detected / n if n else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=["kind_group", "n_marks", "n_detected", "tpr"])


def sweep_strike_detection(
    alarms: pd.DataFrame,
    marks: pd.DataFrame,
    *,
    tolerances_s: Sequence[float],
    alphas: Sequence[float],
    gap_s: float = 1.5,
) -> pd.DataFrame:
    """Tidy long-format per-strike detection sweep for ONE (representation,
    session, alarms-regime) combination already selected by the caller: every
    combination of `granularity in {"impulse", "physical"}` (raw marks vs.
    `deduplicate_marks(marks, gap_s=gap_s)`), `tolerance_s in tolerances_s`,
    `alpha in alphas`, and kind_group (`_kind_group_summary`'s five rows).

    Columns: `granularity, tolerance_s, alpha, kind_group, n_marks,
    n_detected, tpr`; row count is
    `2 * len(tolerances_s) * len(alphas) * 5`.

    Raises:
        ValueError: Any `mark_detected`/`deduplicate_marks` validation error
            (empty *tolerances_s*/*alphas* produces zero rows, not an error).
    """
    _validate_marks(marks)
    physical = deduplicate_marks(marks, gap_s=gap_s)

    rows = []
    for granularity, variant in (
        (_GRANULARITY_IMPULSE, marks),
        (_GRANULARITY_PHYSICAL, physical),
    ):
        for tolerance_s in tolerances_s:
            for alpha in alphas:
                detected = mark_detected(alarms, variant, tolerance_s=tolerance_s, alpha=alpha)
                summary = _kind_group_summary(variant["kind"], detected)
                summary.insert(0, "alpha", alpha)
                summary.insert(0, "tolerance_s", tolerance_s)
                summary.insert(0, "granularity", granularity)
                rows.append(summary)

    if not rows:
        return pd.DataFrame(columns=list(_SWEEP_COLUMNS))
    return pd.concat(rows, ignore_index=True).reindex(columns=list(_SWEEP_COLUMNS))


def _event_onsets(marks: pd.DataFrame) -> pd.DataFrame:
    """One row per `(session, event_id)`: `kind`, `kind_group`, `onset_utc` =
    the EARLIEST raw mark's `strike_utc` in that event (an event's onset for
    `evaluate_event_latency`)."""
    _validate_marks(marks)
    working = marks.copy()
    working["_ns"] = _parse_utc_ns(working["strike_utc"], "strike_utc")

    rows = []
    for (session, event_id), group in working.groupby(["session", "event_id"]):
        onset_idx = group["_ns"].idxmin()
        onset_row = group.loc[onset_idx]
        rows.append(
            {
                "session": session,
                "event_id": event_id,
                "kind": onset_row["kind"],
                "kind_group": kind_group(str(onset_row["kind"])),
                "onset_utc": pd.Timestamp(onset_row["_ns"], tz="UTC", unit="ns"),
            }
        )
    return pd.DataFrame(rows, columns=list(_EVENT_ONSET_COLUMNS))


def _check_search_horizon_s(search_horizon_s: float) -> None:
    if not (math.isfinite(search_horizon_s) and search_horizon_s > 0):
        raise ValueError(
            f"search_horizon_s must be a positive number of seconds, got {search_horizon_s!r}"
        )


def _first_alarm_latency(
    alarming_t: np.ndarray, ref_ns: np.ndarray, horizon_ns: int
) -> tuple[np.ndarray, np.ndarray]:
    """`(latency_s, missed)` for each entry of *ref_ns* against the SORTED
    *alarming_t* (module docstring's "First-alarm latency" bullet): for each
    reference instant, the first entry of *alarming_t* AT-OR-AFTER it
    (`>= ref`); `latency_s` is that entry minus `ref` in seconds (always
    >= 0, NaN when missed); `missed` is `True` when no such entry falls
    within `horizon_ns`. The shared core of `_add_first_alarm_latency`
    (p_value/alpha path) and the binary-alarm-stream latency functions --
    factored out so both run the EXACT same latency arithmetic regardless of
    how *alarming_t* was sourced."""
    latency_s = np.full(ref_ns.shape, np.nan, dtype=np.float64)
    missed = np.ones(ref_ns.shape, dtype=bool)
    if alarming_t.size:
        idx = np.searchsorted(alarming_t, ref_ns, side="left")
        within = idx < alarming_t.size
        first_at_or_after = np.where(within, alarming_t[np.clip(idx, 0, alarming_t.size - 1)], 0)
        delta_ns = first_at_or_after - ref_ns
        in_horizon = within & (delta_ns <= horizon_ns)
        latency_s = np.where(in_horizon, delta_ns / _NS_PER_S, np.nan)
        missed = ~in_horizon
    return latency_s, missed


def _attach_latency(
    reference: pd.DataFrame, latency_s: np.ndarray, missed: np.ndarray
) -> pd.DataFrame:
    """*reference* (a copy) with `latency_s` (float, NaN when missed) and
    `missed` (bool) columns appended -- the shared tail of
    `_add_first_alarm_latency` and the binary-alarm-stream latency
    functions."""
    out = reference.copy()
    out["latency_s"] = latency_s
    out["missed"] = missed
    return out


def _add_first_alarm_latency(
    reference: pd.DataFrame,
    alarms: pd.DataFrame,
    *,
    ref_col: str,
    alpha: float,
    search_horizon_s: float,
) -> pd.DataFrame:
    """*reference* (a copy) with two columns appended: `latency_s` (float,
    NaN when missed) and `missed` (bool). For each row's `ref_col` timestamp,
    finds the first scored, `p_value < alpha` window AT-OR-AFTER it
    (`>= ref`); `latency_s` is that window's start minus `ref`, in seconds
    (always >= 0); a row is `missed` (no such window within
    `search_horizon_s` seconds afterward) when `latency_s` would be NaN.
    """
    _validate_alarms(alarms)
    _check_alpha(alpha)
    _check_search_horizon_s(search_horizon_s)

    alarming_t = _scored_alarming_times(alarms, alpha)
    ref_ns = _parse_utc_ns(reference[ref_col], ref_col)
    horizon_ns = int(round(search_horizon_s * _NS_PER_S))
    latency_s, missed = _first_alarm_latency(alarming_t, ref_ns, horizon_ns)
    return _attach_latency(reference, latency_s, missed)


def evaluate_event_latency(
    alarms: pd.DataFrame,
    marks: pd.DataFrame,
    *,
    alpha: float = 0.05,
    search_horizon_s: float = 60.0,
) -> pd.DataFrame:
    """Per-EVENT first-alarm latency: one row per `(session, event_id)`,
    reference instant = that event's onset (its earliest raw mark,
    `_event_onsets`). Columns: `session, event_id, kind, kind_group,
    onset_utc, latency_s, missed` (module docstring's latency semantics;
    default alpha=0.05, search_horizon_s=60 s, this module's primary reporting default).

    Raises:
        ValueError: Any `_event_onsets`/`_add_first_alarm_latency` validation
            error (bad columns, naive timestamps, invalid alpha/horizon).
    """
    onsets = _event_onsets(marks)
    return _add_first_alarm_latency(
        onsets, alarms, ref_col="onset_utc", alpha=alpha, search_horizon_s=search_horizon_s
    )


def evaluate_strike_latency(
    alarms: pd.DataFrame,
    marks: pd.DataFrame,
    *,
    alpha: float = 0.05,
    search_horizon_s: float = 5.0,
    gap_s: float = 1.5,
) -> pd.DataFrame:
    """Per-PHYSICAL-STRIKE first-alarm latency: one row per deduplicated
    physical strike (`deduplicate_marks(marks, gap_s=gap_s)`). Columns:
    `session, event_id, kind, strike_no, strike_utc, n_impulses,
    last_strike_no, latency_s, missed` (default alpha=0.05,
    search_horizon_s=5 s -- much shorter than
    the per-event horizon, matched to individual-strike cadence rather than
    a whole event's span).

    Raises:
        ValueError: Any `deduplicate_marks`/`_add_first_alarm_latency`
            validation error.
    """
    physical = deduplicate_marks(marks, gap_s=gap_s)
    return _add_first_alarm_latency(
        physical, alarms, ref_col="strike_utc", alpha=alpha, search_horizon_s=search_horizon_s
    )


# ---------------------------------------------------------------------------
# Binary / pre-thresholded alarm streams -- the adapter path for alarms
# tables that carry an already-thresholded bool column instead of a
# continuous p_value (module docstring's "Binary / pre-thresholded alarm
# streams" bullet; motivating case: scripts/run_mad_baseline.py's fixed
# median+k*MAD threshold, commissioned once and applied once, never swept).
# ---------------------------------------------------------------------------


def _validate_alarms_binary(alarms: pd.DataFrame, alarm_column: str) -> None:
    missing = [c for c in ("t_utc_ns", alarm_column) if c not in alarms.columns]
    if missing:
        raise ValueError(
            f"alarms is missing required column(s) {missing}; expected a "
            f"pre-thresholded binary alarm stream with 't_utc_ns' (int64 "
            f"window starts) and {alarm_column!r} (bool) -- got columns "
            f"{list(alarms.columns)}"
        )
    if not pd.api.types.is_integer_dtype(alarms["t_utc_ns"]):
        raise ValueError(
            f"alarms['t_utc_ns'] must be int64 UTC nanoseconds (window starts), "
            f"got dtype {alarms['t_utc_ns'].dtype}"
        )
    if not pd.api.types.is_bool_dtype(alarms[alarm_column]):
        raise ValueError(
            f"alarms[{alarm_column!r}] must be bool, got dtype "
            f"{alarms[alarm_column].dtype}"
        )


def _binary_alarming_times(alarms: pd.DataFrame, alarm_column: str) -> np.ndarray:
    """Sorted int64 UTC-ns window-START timestamps of every SCORED window
    whose *alarm_column* is `True` -- the binary-alarm-stream analogue of
    `_scored_alarming_times` (same `role` filter, module docstring)."""
    scored = alarms[alarms["role"] == ROLE_SCORED] if "role" in alarms.columns else alarms
    t = scored["t_utc_ns"].to_numpy(dtype=np.int64)
    flagged = scored[alarm_column].to_numpy(dtype=bool)
    return np.sort(t[flagged])


def mark_detected_binary(
    alarms: pd.DataFrame,
    marks: pd.DataFrame,
    *,
    tolerance_s: float,
    alarm_column: str = "alarm",
) -> np.ndarray:
    """`mark_detected`'s binary-alarm-stream counterpart (module docstring):
    a mark counts as detected iff at least one SCORED window with
    `alarms[alarm_column] == True` has a window-START timestamp within
    `tolerance_s` seconds of the mark, on either side -- the IDENTICAL
    tolerance matching `mark_detected` uses (`_match_within_tolerance`); the
    only difference is how the alarming-window timestamps are sourced (an
    already-thresholded bool column instead of `p_value < alpha`).

    Args:
        alarms: One row per window -- `t_utc_ns` (int64 UTC window-start ns),
            *alarm_column* (bool, already thresholded upstream); rows with a
            `role` other than `"scored"` are dropped first (if a `role`
            column exists).
        marks: One row per strike-impulse mark -- `strike_utc` (tz-aware).
        tolerance_s: Half-width of the matching window in seconds (>= 0).
        alarm_column: Name of the pre-thresholded bool column in *alarms*.

    Returns:
        `np.ndarray[bool]`, `len(marks)` entries, in input order.

    Raises:
        ValueError: On missing/mistyped alarms or marks columns, naive/
            unparseable `strike_utc`, or `tolerance_s < 0`.
    """
    _validate_alarms_binary(alarms, alarm_column)
    _validate_marks(marks)
    _check_tolerance_s(tolerance_s)

    alarming_t = _binary_alarming_times(alarms, alarm_column)
    mark_ns = _parse_utc_ns(marks["strike_utc"], "strike_utc")
    tol_ns = int(round(tolerance_s * _NS_PER_S))
    return _match_within_tolerance(alarming_t, mark_ns, tol_ns)


_SWEEP_COLUMNS_BINARY: tuple[str, ...] = (
    "granularity", "tolerance_s", "kind_group", "n_marks", "n_detected", "tpr",
)


def sweep_strike_detection_binary(
    alarms: pd.DataFrame,
    marks: pd.DataFrame,
    *,
    tolerances_s: Sequence[float],
    gap_s: float = 1.5,
    alarm_column: str = "alarm",
) -> pd.DataFrame:
    """`sweep_strike_detection`'s binary-alarm-stream counterpart (module
    docstring): every combination of `granularity in {"impulse", "physical"}`
    and `tolerance_s in tolerances_s`, for the ONE pre-thresholded
    *alarm_column* -- no `alpha` axis (the alarm decision is already fixed
    upstream, e.g. the MAD baseline's once-commissioned `score > threshold`).
    A caller comparing several fixed thresholds (e.g. `k1pct` vs `k5`) calls
    this once per threshold, against its own alarm column, and tags the
    result itself (this function has no notion of "threshold identity").

    Columns: `granularity, tolerance_s, kind_group, n_marks, n_detected,
    tpr`; row count is `2 * len(tolerances_s) * 5`.

    Raises:
        ValueError: Any `mark_detected_binary`/`deduplicate_marks` validation
            error (empty *tolerances_s* produces zero rows, not an error).
    """
    _validate_marks(marks)
    physical = deduplicate_marks(marks, gap_s=gap_s)

    rows = []
    for granularity, variant in (
        (_GRANULARITY_IMPULSE, marks),
        (_GRANULARITY_PHYSICAL, physical),
    ):
        for tolerance_s in tolerances_s:
            detected = mark_detected_binary(
                alarms, variant, tolerance_s=tolerance_s, alarm_column=alarm_column
            )
            summary = _kind_group_summary(variant["kind"], detected)
            summary.insert(0, "tolerance_s", tolerance_s)
            summary.insert(0, "granularity", granularity)
            rows.append(summary)

    if not rows:
        return pd.DataFrame(columns=list(_SWEEP_COLUMNS_BINARY))
    return pd.concat(rows, ignore_index=True).reindex(columns=list(_SWEEP_COLUMNS_BINARY))


def evaluate_strike_latency_binary(
    alarms: pd.DataFrame,
    marks: pd.DataFrame,
    *,
    search_horizon_s: float = 5.0,
    gap_s: float = 1.5,
    alarm_column: str = "alarm",
) -> pd.DataFrame:
    """`evaluate_strike_latency`'s binary-alarm-stream counterpart (module
    docstring): per-PHYSICAL-STRIKE first-alarm latency against the ONE
    pre-thresholded *alarm_column*, no `alpha`. Columns: `session, event_id,
    kind, strike_no, strike_utc, n_impulses, last_strike_no, latency_s,
    missed` (default `search_horizon_s=5 s`, matching `evaluate_strike_
    latency`'s own default).

    Raises:
        ValueError: Any `deduplicate_marks`/`_validate_alarms_binary`
            validation error.
    """
    _validate_alarms_binary(alarms, alarm_column)
    _check_search_horizon_s(search_horizon_s)

    physical = deduplicate_marks(marks, gap_s=gap_s)
    alarming_t = _binary_alarming_times(alarms, alarm_column)
    ref_ns = _parse_utc_ns(physical["strike_utc"], "strike_utc")
    horizon_ns = int(round(search_horizon_s * _NS_PER_S))
    latency_s, missed = _first_alarm_latency(alarming_t, ref_ns, horizon_ns)
    return _attach_latency(physical, latency_s, missed)


@dataclass(frozen=True)
class LatencySummary:
    """Scalar latency summary of one `evaluate_event_latency`/
    `evaluate_strike_latency` result (or any `latency_s`/`missed` pair) --
    `summarize_latency`'s return type."""

    n_total: int
    """Total reference instants (events or physical strikes) summarized."""
    n_detected: int
    """References with a qualifying alarm inside their search horizon."""
    n_missed: int
    """`n_total - n_detected`."""
    median_s: float
    """Median `latency_s` over DETECTED references only; NaN when
    `n_detected == 0` (a missed reference has no latency value to average)."""
    iqr_low_s: float
    """25th percentile of `latency_s` over detected references; NaN when
    `n_detected == 0`."""
    iqr_high_s: float
    """75th percentile of `latency_s` over detected references; NaN when
    `n_detected == 0`."""


def summarize_latency(
    latency_s: np.ndarray | pd.Series, missed: np.ndarray | pd.Series
) -> LatencySummary:
    """`LatencySummary` of the *latency_s*/*missed* pair (`evaluate_event_
    latency`/`evaluate_strike_latency`'s own columns, or any array pair of
    the same shape) -- median/IQR computed over the DETECTED (non-missed)
    entries only (module docstring's `LatencySummary` field docs).

    Raises:
        ValueError: *latency_s* and *missed* have different shapes.
    """
    latency_arr = np.asarray(latency_s, dtype=np.float64)
    missed_arr = np.asarray(missed, dtype=bool)
    if latency_arr.shape != missed_arr.shape:
        raise ValueError(
            f"latency_s and missed must have the same shape, got "
            f"{latency_arr.shape} vs {missed_arr.shape}"
        )

    n_total = int(latency_arr.shape[0])
    n_missed = int(missed_arr.sum())
    n_detected = n_total - n_missed
    detected_latencies = latency_arr[~missed_arr]

    if n_detected:
        median_s = float(np.median(detected_latencies))
        iqr_low_s, iqr_high_s = (float(q) for q in np.percentile(detected_latencies, [25, 75]))
    else:
        median_s = iqr_low_s = iqr_high_s = float("nan")

    return LatencySummary(
        n_total=n_total,
        n_detected=n_detected,
        n_missed=n_missed,
        median_s=median_s,
        iqr_low_s=iqr_low_s,
        iqr_high_s=iqr_high_s,
    )


def inter_mark_gaps(marks: pd.DataFrame) -> pd.DataFrame:
    """One row per consecutive raw-mark PAIR within the same `(session,
    event_id)`, in chronological order: `session, event_id, kind, kind_group,
    to_strike_no` (the LATER mark's `strike_no` -- identifies "the gap before
    mark #to_strike_no"), `gap_s` (time from the previous mark to this one).
    An event with a single mark contributes zero rows. Quantifies the
    within-event impulse spacing (module docstring; `deduplicate_marks` uses
    the same consecutive-gap quantity against a 1.5 s threshold).

    Raises:
        ValueError: Missing/mistyped marks columns or naive/unparseable
            `strike_utc`.
    """
    _validate_marks(marks)
    working = marks.copy()
    working["_ns"] = _parse_utc_ns(working["strike_utc"], "strike_utc")
    working = working.sort_values(["session", "event_id", "_ns"], kind="stable")

    rows = []
    for (session, event_id), group in working.groupby(["session", "event_id"], sort=False):
        ns = group["_ns"].to_numpy(dtype=np.int64)
        strike_no = group["strike_no"].to_numpy()
        kind = str(group["kind"].iloc[0])
        for i in range(1, len(group)):
            rows.append(
                {
                    "session": session,
                    "event_id": event_id,
                    "kind": kind,
                    "kind_group": kind_group(kind),
                    "to_strike_no": strike_no[i],
                    "gap_s": (ns[i] - ns[i - 1]) / _NS_PER_S,
                }
            )
    return pd.DataFrame(rows, columns=list(_GAP_COLUMNS))


def marks_per_event(marks: pd.DataFrame) -> pd.DataFrame:
    """One row per `(session, event_id)`: `session, event_id, kind,
    kind_group, n_marks` (raw-mark count -- the campaign's protocol expected
    3 per plate/landmark position; module docstring notes most plate
    positions actually carry 6, some carry more).

    Raises:
        ValueError: Missing marks columns (`_REQUIRED_MARK_COLUMNS`).
    """
    _validate_marks(marks)
    rows = []
    for (session, event_id), group in marks.groupby(["session", "event_id"]):
        kind = str(group["kind"].iloc[0])
        rows.append(
            {
                "session": session,
                "event_id": event_id,
                "kind": kind,
                "kind_group": kind_group(kind),
                "n_marks": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=list(_MARKS_PER_EVENT_COLUMNS))
