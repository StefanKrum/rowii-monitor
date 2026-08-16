"""Pillar-3 event-level evaluation CLI (PREPARED-ONLY): labeled fault intervals
(`events.csv`) vs a monitor/sweep alarms table (`alarms.parquet`) -> per-event
TPR, detection latency, and false-alarm rates.

Campaign day = `scripts/monitor.py` (snapshot + new recording -> alarms.parquet)
then THIS script (alarms.parquet + the campaign's labeled fault intervals ->
`event_eval.csv` + `event_notes.md`), nothing else. Until the induced-fault
campaign delivers real labels, every invocation runs on synthetic intervals and
is a workflow DEMO -- `event_notes.md` restates this on every run (honesty
rule).

All semantics live in `rowii.eval.events.evaluate_events` (its module docstring:
inclusive-start/exclusive-end membership over window-START timestamps, tolerance
padding with the negative-latency convention, role filtering to scored windows,
non-event FAR denominators); this script only parses the two files, calls it,
and writes the two outputs. `event_eval.csv` is tidy: ONE `row_type="summary"`
row (the `EventEvalResult` scalars) followed by one `row_type="event"` row per
fault interval -- each row family leaves the other family's columns empty.

Exit codes: 0 on success; 2 on usage errors (missing input files, malformed or
naive event timestamps, invalid tolerance/window values).
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.eval.events import ROLE_SCORED, EventEvalResult, evaluate_events  # noqa: E402

logger = logging.getLogger(__name__)

_ROW_TYPE_SUMMARY = "summary"
_ROW_TYPE_EVENT = "event"

_CSV_COLUMNS: tuple[str, ...] = (
    "row_type",
    "n_events", "n_detected", "event_tpr", "false_alarm_windows",
    "false_alarm_rate_per_hour", "realized_window_far", "tolerance_s",
    "start_utc", "end_utc", "kind", "detected", "latency_s",
)
"""event_eval.csv's exact column contract, in this order (module docstring:
summary row first, then per-event rows; the two row families share no columns
beyond `row_type`)."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Event-level evaluation of a per-window alarms table against labeled "
            "fault intervals: per-event TPR, detection latency, and false-alarm "
            "rates (rowii.eval.events.evaluate_events). PREPARED-ONLY: until the "
            "induced-fault campaign delivers real fault labels, every run is a "
            "workflow demo (module docstring)."
        )
    )
    parser.add_argument(
        "--alarms", type=Path, required=True,
        help="Path to an alarms parquet (scripts/monitor.py or scripts/run_step2.py "
             "schema: t_utc_ns int64 window starts + alarm bool; rows whose 'role' "
             "is not 'scored' are dropped before every metric).",
    )
    parser.add_argument(
        "--events", type=Path, required=True,
        help="Path to a fault-events CSV: start_utc,end_utc[,kind] with tz-AWARE "
             "ISO-8601 timestamps (naive timestamps are a usage error, exit 2).",
    )
    parser.add_argument(
        "--tolerance-s", type=float, default=0.0,
        help="Tolerance pad in seconds applied to BOTH sides of every event "
             "interval before membership tests (default: 0.0).",
    )
    parser.add_argument(
        "--window-s", type=float, default=1.0,
        help="Window length in seconds of the alarms grid -- converts the "
             "non-event window count into covered hours for "
             "false_alarm_rate_per_hour (default: 1.0, this repo's standard grid).",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output directory (created if missing) for event_eval.csv + "
             "event_notes.md.",
    )
    return parser


def _event_eval_frame(result: EventEvalResult) -> pd.DataFrame:
    """The tidy `event_eval.csv` frame: the summary row first, then one row per
    event, columns per `_CSV_COLUMNS` (each row family leaves the other's
    columns empty)."""
    summary = result.to_frame()
    summary.insert(0, "row_type", _ROW_TYPE_SUMMARY)
    per_event = result.per_event.copy()
    per_event.insert(0, "row_type", _ROW_TYPE_EVENT)
    combined = pd.concat([summary, per_event], ignore_index=True)
    return combined.reindex(columns=list(_CSV_COLUMNS))


def _notes_markdown(
    alarms_path: Path,
    events_path: Path,
    result: EventEvalResult,
    window_s: float,
    n_alarm_rows: int,
    n_scored_rows: int,
) -> str:
    def fmt(value: float) -> str:
        return "n/a" if math.isnan(value) else f"{value:.6g}"

    lines = [
        f"# Event-level evaluation: {events_path.name} vs {alarms_path.name}",
        "",
        "**PREPARED-ONLY harness (spec §4 honesty rule): NO fault labels exist yet "
        "for any recording in this project -- this harness is PREPARED for the "
        "induced-fault campaign. Any run on synthetic or hand-crafted event "
        "intervals is a workflow demo of the campaign-day pipeline "
        "(scripts/monitor.py -> scripts/eval_events.py), never a detection "
        "result.**",
        "",
        "## Inputs",
        "",
        f"- alarms: `{alarms_path}` ({n_alarm_rows} row(s), {n_scored_rows} scored "
        "-- rows with any other `role` carry no verdict and are dropped before "
        "every metric)",
        f"- events: `{events_path}` ({result.n_events} fault interval(s))",
        f"- window_s: {window_s:g} (seconds per window; timestamps are window "
        "STARTS)",
        f"- tolerance_s: {result.tolerance_s:g} (pad applied to BOTH sides of "
        "every event interval)",
        "",
        "## Conventions (spec D3)",
        "",
        "- Membership: window start `t` lies inside event `e` iff "
        "`start - tolerance <= t < end + tolerance` -- inclusive start, exclusive "
        "end (timestamps are window starts, so a window starting exactly at `end` "
        "lies entirely after the event).",
        "- An event is detected iff >= 1 ALARMING window lies inside its padded "
        "interval; `event_tpr = n_detected / n_events`, NaN when there are zero "
        "events (vacuous -- never reported as 1.0).",
        "- Latency = first alarming window start inside the padded interval minus "
        "the event start (seconds); NEGATIVE when that window lies in the "
        "tolerance pad before the logged start (kept, not clamped); NaN for "
        "missed events.",
        "- False alarms = alarming windows outside EVERY padded event; "
        "`realized_window_far` divides by the non-event scored windows, "
        "`false_alarm_rate_per_hour` by the non-event covered duration "
        "(`n_non_event_scored_windows * window_s / 3600`); both are NaN when no "
        "scored window lies outside the padded events.",
        "- Overlapping events are evaluated independently: one alarming window "
        "may detect several events.",
        "",
        "## Summary",
        "",
        "| n_events | n_detected | event_tpr | false_alarm_windows "
        "| false_alarm_rate_per_hour | realized_window_far | tolerance_s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {result.n_events} | {result.n_detected} | {fmt(result.event_tpr)} "
        f"| {result.false_alarm_windows} | {fmt(result.false_alarm_rate_per_hour)} "
        f"| {fmt(result.realized_window_far)} | {result.tolerance_s:g} |",
        "",
        "## Per-event results",
        "",
    ]
    if result.n_events == 0:
        lines.append("(no events in the input table)")
    else:
        lines += [
            "| start_utc | end_utc | kind | detected | latency_s |",
            "|:--|:--|:--|:--|---:|",
        ]
        for _, row in result.per_event.iterrows():
            latency = (
                "n/a (missed)"
                if math.isnan(row["latency_s"])
                else f"{row['latency_s']:.3f}"
            )
            lines.append(
                f"| {row['start_utc'].isoformat()} | {row['end_utc'].isoformat()} "
                f"| {row['kind']} | {bool(row['detected'])} | {latency} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.alarms.is_file():
        print(f"eval_events: alarms file not found: {args.alarms}", file=sys.stderr)
        return 2
    if not args.events.is_file():
        print(f"eval_events: events file not found: {args.events}", file=sys.stderr)
        return 2

    try:
        alarms = pd.read_parquet(args.alarms, engine="pyarrow")
        # comment="#": the ground-truth CSVs (docs/groundtruth/080726_events_*.csv)
        # open with # provenance lines before the header row.
        events = pd.read_csv(args.events, comment="#")
        result = evaluate_events(
            alarms, events, window_s=args.window_s, tolerance_s=args.tolerance_s
        )
    except ValueError as exc:
        # Covers evaluate_events' own contract errors (naive timestamps, missing
        # columns, invalid tolerance/window values) AND pandas/pyarrow read errors,
        # which subclass ValueError (pd.errors.EmptyDataError, pyarrow ArrowInvalid).
        print(f"eval_events: {exc}", file=sys.stderr)
        return 2

    n_alarm_rows = int(len(alarms))
    n_scored_rows = (
        int((alarms["role"] == ROLE_SCORED).sum())
        if "role" in alarms.columns
        else n_alarm_rows
    )

    args.out.mkdir(parents=True, exist_ok=True)
    _event_eval_frame(result).to_csv(args.out / "event_eval.csv", index=False)
    (args.out / "event_notes.md").write_text(
        _notes_markdown(
            args.alarms, args.events, result, args.window_s, n_alarm_rows,
            n_scored_rows,
        )
    )

    print(
        f"eval_events: {result.n_detected}/{result.n_events} event(s) detected, "
        f"{result.false_alarm_windows} false-alarm window(s) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
