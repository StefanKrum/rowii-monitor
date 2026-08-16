"""Pillar-3 event-level evaluation: labeled fault intervals vs per-window alarms.

PREPARED-ONLY harness: no fault labels exist
yet for any ROWII recording -- the controlled-event measurement campaign is pending.
This module is the campaign-data interface, ready the day labeled fault intervals
land; until then every invocation runs on synthetic intervals and is a workflow
DEMO, never a detection result (honesty rule).

Input contract (the campaign-data interface):

- **alarms**: one row per window, the `scripts/monitor.py` / `scripts/
  run_step2.py` `alarms.parquet` shape -- required columns `t_utc_ns` (int64
  window-START UTC nanoseconds, `rowii.anomaly.overlap.to_utc_ns`'s left-edge
  convention) and `alarm` (bool). If a `role` column is present, rows with
  `role != "scored"` are dropped FIRST (`scripts/monitor.py`'s role vocabulary:
  calibration-consumed / unknown-state / no-conformal-data windows carry no
  verdict, so they can neither detect an event nor count as a false alarm --
  only scored windows enter any numerator or denominator below).
- **events**: one row per labeled fault interval (`events.csv`), required
  columns `start_utc`/`end_utc` as timezone-AWARE ISO-8601 timestamps (naive
  values raise `ValueError`: a campaign log without an explicit UTC offset is
  ambiguous by one-to-two hours around CET/CEST, exactly the error this loud
  refusal prevents); optional `kind` column, defaulting to `"fault"`.

Semantics (every edge pinned by `tests/test_events.py`):

- **Membership**: window start `t` lies inside event `e` iff
  `start_ns - tol_ns <= t < end_ns + tol_ns` -- inclusive start, EXCLUSIVE end,
  because timestamps are window STARTS: a window starting exactly at `end_utc`
  lies entirely after the event.
- An event is **detected** iff at least one ALARMING window lies inside its
  tolerance-padded interval; `event_tpr = n_detected / n_events`, NaN when
  there are zero events (vacuous -- deliberately never reported as 1.0).
- **Detection latency** = first alarming window start inside the padded
  interval minus `start_utc`, in seconds. NEGATIVE when that window lies in
  the tolerance pad before the logged start -- kept, not clamped: an alarm
  shortly before the logged onset is early warning relative to the label, and
  clamping would silently overstate the label's precision. NaN for missed
  events.
- **False alarms** are alarming windows outside EVERY padded event.
  `realized_window_far` divides them by the number of non-event scored
  windows; `false_alarm_rate_per_hour` divides them by the non-event covered
  duration (`n_non_event_scored_windows * window_s / 3600`). Both are NaN when
  no scored window lies outside the padded events (denominator 0).
- **Overlapping events** are evaluated independently: one alarming window may
  detect several events (membership is per event, never first-match).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROLE_SCORED = "scored"
"""The one `role` value whose rows carry a verdict. Mirrors `scripts/monitor.py`'s
`ROLE_SCORED` by value (a library module must not import from a script; the role
vocabulary itself is pinned by that script's alarms.parquet contract)."""

_PER_EVENT_COLUMNS = ["start_utc", "end_utc", "kind", "detected", "latency_s"]
_DEFAULT_KIND = "fault"
_NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class EventEvalResult:
    """Event-level detection metrics of one alarms table against one fault-events
    table (`evaluate_events` output; module docstring for every definition)."""

    per_event: pd.DataFrame
    """One row per input event, in input order: columns `start_utc, end_utc`
    (timezone-aware UTC), `kind`, `detected` (bool), `latency_s` (float; NaN for
    missed events, possibly NEGATIVE inside the tolerance pad)."""
    n_events: int
    """Number of labeled fault intervals in the events table."""
    n_detected: int
    """Events with at least one alarming window inside their padded interval."""
    event_tpr: float
    """`n_detected / n_events`; NaN when `n_events == 0` (vacuous, never 1.0)."""
    false_alarm_windows: int
    """Alarming scored windows outside EVERY tolerance-padded event."""
    false_alarm_rate_per_hour: float
    """`false_alarm_windows / (n_non_event_scored_windows * window_s / 3600)`;
    NaN when no scored window lies outside the padded events."""
    realized_window_far: float
    """`false_alarm_windows / n_non_event_scored_windows`; NaN when no scored
    window lies outside the padded events."""
    tolerance_s: float
    """The tolerance pad (seconds) this result was computed with, echoed for
    provenance -- latency and FAR numbers are only comparable at equal pads."""

    def to_frame(self) -> pd.DataFrame:
        """This result's scalar fields as ONE summary row (declaration order) --
        the CSV/notes-facing view; `per_event` stays its own table."""
        return pd.DataFrame(
            [
                {
                    "n_events": self.n_events,
                    "n_detected": self.n_detected,
                    "event_tpr": self.event_tpr,
                    "false_alarm_windows": self.false_alarm_windows,
                    "false_alarm_rate_per_hour": self.false_alarm_rate_per_hour,
                    "realized_window_far": self.realized_window_far,
                    "tolerance_s": self.tolerance_s,
                }
            ]
        )


def _parse_utc_ns(values: pd.Series, column: str) -> np.ndarray:
    """*values* (ISO-8601 strings or already-parsed timestamps) as int64 UTC
    nanoseconds. Naive or missing entries raise `ValueError` (module docstring:
    a fault log without an explicit UTC offset is ambiguous, refuse loudly)."""
    out = np.empty(len(values), dtype=np.int64)
    for i, raw in enumerate(values):
        try:
            ts = pd.Timestamp(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"events[{column!r}] row {i}: cannot parse {raw!r} as a timestamp"
            ) from exc
        if pd.isna(ts):
            raise ValueError(f"events[{column!r}] row {i}: missing timestamp")
        if ts.tzinfo is None:
            raise ValueError(
                f"events[{column!r}] row {i}: naive timestamp {raw!r} has no UTC "
                f"offset -- write ISO-8601 with an explicit offset (e.g. "
                f"'2026-07-16T14:30:00+00:00' or a trailing 'Z'); fault intervals "
                f"must be timezone-aware (spec D3)"
            )
        out[i] = ts.tz_convert("UTC").value
    return out


def _validate_alarms(alarms: pd.DataFrame) -> None:
    missing = [c for c in ("t_utc_ns", "alarm") if c not in alarms.columns]
    if missing:
        raise ValueError(
            f"alarms is missing required column(s) {missing}; expected the "
            f"monitor/sweep alarms.parquet shape with 't_utc_ns' (int64 window "
            f"starts) and 'alarm' (bool) -- got columns {list(alarms.columns)}"
        )
    if not pd.api.types.is_integer_dtype(alarms["t_utc_ns"]):
        raise ValueError(
            f"alarms['t_utc_ns'] must be int64 UTC nanoseconds (window starts), "
            f"got dtype {alarms['t_utc_ns'].dtype}"
        )
    if not pd.api.types.is_bool_dtype(alarms["alarm"]):
        raise ValueError(
            f"alarms['alarm'] must be bool, got dtype {alarms['alarm'].dtype}"
        )


def evaluate_events(
    alarms: pd.DataFrame,
    events: pd.DataFrame,
    *,
    window_s: float,
    tolerance_s: float = 0.0,
) -> EventEvalResult:
    """Per-event TPR, detection latency, and false-alarm rates of *alarms*
    against the labeled fault intervals in *events* (all conventions in the
    module docstring; every edge pinned by `tests/test_events.py`).

    Args:
        alarms: One row per window -- required columns `t_utc_ns` (int64 UTC
            window-start nanoseconds) and `alarm` (bool); rows with a `role`
            column value other than `"scored"` are dropped first.
        events: One row per fault interval -- required columns `start_utc` /
            `end_utc` (timezone-AWARE ISO-8601), optional `kind` (defaults to
            `"fault"`, also per missing cell).
        window_s: Window length in seconds of the alarms grid; converts the
            non-event window count into covered hours for
            `false_alarm_rate_per_hour`.
        tolerance_s: Pad (seconds, >= 0) applied to BOTH sides of every event
            interval before membership tests.

    Returns:
        An `EventEvalResult` (see its field docs for exact definitions,
        including which quantities are NaN in degenerate inputs).

    Raises:
        ValueError: On missing/mistyped required columns, naive or unparseable
            event timestamps, `end_utc` preceding `start_utc`, non-positive
            *window_s*, or negative *tolerance_s*.
    """
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError(f"window_s must be a positive number of seconds, got {window_s!r}")
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError(f"tolerance_s must be >= 0 seconds, got {tolerance_s!r}")

    _validate_alarms(alarms)
    missing_events = [c for c in ("start_utc", "end_utc") if c not in events.columns]
    if missing_events:
        raise ValueError(
            f"events is missing required column(s) {missing_events}; expected "
            f"'start_utc' and 'end_utc' (tz-aware ISO-8601) -- got columns "
            f"{list(events.columns)}"
        )

    scored = alarms[alarms["role"] == ROLE_SCORED] if "role" in alarms.columns else alarms
    t = scored["t_utc_ns"].to_numpy(dtype=np.int64)
    alarm = scored["alarm"].to_numpy(dtype=bool)

    start_ns = _parse_utc_ns(events["start_utc"], "start_utc")
    end_ns = _parse_utc_ns(events["end_utc"], "end_utc")
    swapped = end_ns < start_ns
    if bool(swapped.any()):
        row = int(np.argmax(swapped))
        raise ValueError(
            f"events row {row}: end_utc precedes start_utc -- fault intervals must "
            f"satisfy start_utc <= end_utc (swapped columns are a data-entry error, "
            f"not a missed event)"
        )
    if "kind" in events.columns:
        kind = [_DEFAULT_KIND if pd.isna(k) else str(k) for k in events["kind"]]
    else:
        kind = [_DEFAULT_KIND] * len(start_ns)

    tol_ns = int(round(tolerance_s * _NS_PER_S))
    n_events = int(start_ns.shape[0])

    # Membership is evaluated PER EVENT (module docstring: overlapping events
    # are independent; one alarming window may detect several).
    in_any_event = np.zeros(t.shape[0], dtype=bool)
    detected = np.zeros(n_events, dtype=bool)
    latency_s = np.full(n_events, np.nan, dtype=np.float64)
    for i in range(n_events):
        member = (t >= start_ns[i] - tol_ns) & (t < end_ns[i] + tol_ns)
        in_any_event |= member
        alarming_ts = t[member & alarm]
        if alarming_ts.size:
            detected[i] = True
            latency_s[i] = (int(alarming_ts.min()) - int(start_ns[i])) / _NS_PER_S

    n_detected = int(detected.sum())
    event_tpr = n_detected / n_events if n_events else float("nan")

    false_alarm_windows = int((alarm & ~in_any_event).sum())
    n_non_event = int((~in_any_event).sum())
    realized_window_far = (
        false_alarm_windows / n_non_event if n_non_event else float("nan")
    )
    false_alarm_rate_per_hour = (
        false_alarm_windows / (n_non_event * window_s / 3600.0)
        if n_non_event
        else float("nan")
    )

    per_event = pd.DataFrame(
        {
            "start_utc": pd.to_datetime(start_ns, utc=True),
            "end_utc": pd.to_datetime(end_ns, utc=True),
            "kind": pd.Series(kind, dtype=object),
            "detected": detected,
            "latency_s": latency_s,
        },
        columns=_PER_EVENT_COLUMNS,
    )

    return EventEvalResult(
        per_event=per_event,
        n_events=n_events,
        n_detected=n_detected,
        event_tpr=float(event_tpr),
        false_alarm_windows=false_alarm_windows,
        false_alarm_rate_per_hour=float(false_alarm_rate_per_hour),
        realized_window_far=float(realized_window_far),
        tolerance_s=float(tolerance_s),
    )
