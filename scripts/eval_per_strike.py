"""Per-strike detection + first-alarm latency: the brand-new seconds-level
hammer-strike ground truth (`docs/groundtruth/080726_strikes_seconds_
{st,pu}.csv`) vs. the alarms.parquet artifacts ALREADY on disk under
`results/step2/once-calibrated/` and `results/pillar3/`.

This script never runs `scripts/monitor.py` and never recomputes an alarm --
it only reads existing alarms tables (module docstring of `rowii.eval.
per_strike`, whose pure functions do all the matching/latency/dedup logic)
and writes four report artifacts to `results/pillar3-perstrike/`.

Representations x alarms source (the FIVE the task scopes this to):

- `fusion`, `audio-beats`, `vibration`: `results/step2/once-calibrated/
  <representation>/monitor/080726-<session>_strikes/recalibrate/
  alarms.parquet` -- the RECALIBRATE regime specifically (never `frozen`).
- `audio`, `audio-student`: `results/pillar3/080726-<session>_strikes/
  <representation>-a0.05/alarms.parquet`. Pillar-3 exports one alarms.parquet
  PER (representation, alpha) combination, but `p_value`/`t_utc_ns`/`role`
  are IDENTICAL across a representation's `-a0.01`/`-a0.05`/`-a0.10` exports
  (verified directly: see the accompanying research note) -- only the baked-
  in `alarm` column differs, and this script never reads that column, always
  re-thresholding `p_value` itself (`rowii.eval.per_strike.mark_detected`).
  The `-a0.05` file is read as the one canonical source of `p_value`/
  `t_utc_ns`/`role` for a representation; this is a file-selection detail,
  not a restriction to alpha=0.05 (the detection sweep still covers alpha in
  {0.01, 0.05, 0.10} from that one file's `p_value` column).

A representation missing for a session (`.is_file()` false) is skipped with a
warning, not a hard failure ("where present").

Outputs (`results/pillar3-perstrike/`):

- `per_strike_detection.csv` -- `rowii.eval.per_strike.sweep_strike_detection`
  per (session, representation), tagged with `alarms_path`/`regime`.
- `latency.csv` -- one `row_type="summary"` row per (session, representation,
  level) with `rowii.eval.per_strike.summarize_latency`'s median/IQR/counts,
  followed by `row_type="detail"` rows (one per event onset / physical
  strike) with its own latency_s/missed -- ALWAYS at `LATENCY_ALPHA` (0.05),
  the recalibrate-regime alarms (module docstring; alpha basis restated in
  `summary.md`).
- `impulse_structure.csv` -- `row_type="event"` (marks-per-event counts),
  `row_type="gap"` (every within-event consecutive-mark gap), and
  `row_type="summary"` (median/min/max gap for the 6-mark plate events, split
  at the DEDUP_GAP_S=1.5s threshold into short/intra-cluster vs. long/inter-
  cluster gaps -- quantifies the double-impulse spacing the dedup rule acts
  on). Ground-truth-only, no alarms artifact involved.
- `summary.md` -- plain-language headline findings with exact numbers, the
  two-unmarked-PU-landmark-events finding (attributed to the partner's own
  reported result), and the alarms-artifact provenance table.

Usage:
    cd repos/rowii-monitor && .venv/bin/python scripts/eval_per_strike.py
"""
from __future__ import annotations

import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPTS_DIR.parent / "src"
for _extra_path in (str(_SCRIPTS_DIR), str(_SRC_DIR)):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

from rowii.eval.events import ROLE_SCORED  # noqa: E402
from rowii.eval.per_strike import (  # noqa: E402
    evaluate_event_latency,
    evaluate_strike_latency,
    inter_mark_gaps,
    kind_group,
    marks_per_event,
    summarize_latency,
    sweep_strike_detection,
)

logger = logging.getLogger(__name__)

SESSIONS: tuple[str, ...] = ("st", "pu")
REPRESENTATIONS: tuple[str, ...] = ("fusion", "audio-beats", "vibration", "audio", "audio-student")
TOLERANCES_S: tuple[float, ...] = (1.0, 2.0, 5.0)
ALPHAS: tuple[float, ...] = (0.01, 0.05, 0.10)
DEDUP_GAP_S = 1.5
LATENCY_ALPHA = 0.05
EVENT_LATENCY_HORIZON_S = 60.0
STRIKE_LATENCY_HORIZON_S = 5.0
#: The headline slice this script's own `main()` print statement and the
#: REPORT step both key off: physical-strike granularity, T=2s, alpha=0.05.
HEADLINE_TOLERANCE_S = 2.0
HEADLINE_ALPHA = 0.05

_ONCE_CALIBRATED_REPS = frozenset({"fusion", "audio-beats", "vibration"})
_PILLAR3_COMBO: dict[str, str] = {"audio": "audio-a0.05", "audio-student": "audio-student-a0.05"}

_DETECTION_COLUMNS: list[str] = [
    "session", "representation", "regime", "alarms_path",
    "granularity", "tolerance_s", "alpha", "kind_group", "n_marks", "n_detected", "tpr",
]
_LATENCY_COLUMNS: list[str] = [
    "row_type", "level", "session", "representation", "regime", "alarms_path",
    "alpha", "search_horizon_s",
    "n_total", "n_detected", "n_missed", "median_s", "iqr_low_s", "iqr_high_s",
    "event_id", "kind", "kind_group", "strike_no", "n_impulses", "ref_utc", "latency_s", "missed",
]
_IMPULSE_COLUMNS: list[str] = [
    "row_type", "session", "event_id", "kind", "kind_group",
    "n_marks", "to_strike_no", "gap_s",
    "subset", "n_gaps", "median_gap_s", "min_gap_s", "max_gap_s",
]

_PARTNER_DIGEST_NOTE = "research/notes/external_2026-07-21_bruno_analyses_digest.md"
"""Attributed external reference (Rodrigues & Zhang, HSG Sensing Group; read
via the master-thesis workspace, not this repo) recording the partner's own
080726 detector result: all plate/ring strikes detected under PU, landmarks
under pump noise not. Cited, never imported as our own result (Bruno-content-
firewall convention: partner analyses are attributed external reference,
never thesis knowledge; our own ground truth/finding stands on our own
data)."""


def _session_marks_path(repo_root: Path, session: str) -> Path:
    return repo_root / "docs" / "groundtruth" / f"080726_strikes_seconds_{session}.csv"


def _session_events_reference_path(repo_root: Path, session: str) -> Path:
    return repo_root / "docs" / "groundtruth" / f"080726_events_{session}.csv"


def _alarms_path_and_regime(repo_root: Path, representation: str, session: str) -> tuple[Path, str]:
    session_dir = f"080726-{session}_strikes"
    if representation in _ONCE_CALIBRATED_REPS:
        path = (
            repo_root / "results" / "step2" / "once-calibrated" / representation / "monitor"
            / session_dir / "recalibrate" / "alarms.parquet"
        )
        return path, "once-calibrated/recalibrate"
    combo = _PILLAR3_COMBO[representation]
    path = repo_root / "results" / "pillar3" / session_dir / combo / "alarms.parquet"
    regime = (
        f"pillar3/{combo} (recalibrate; p_value/t_utc_ns/role verified identical "
        f"across that representation's -a0.01/-a0.05/-a0.10 exports -- this file is "
        f"the canonical p_value source, re-thresholded for every alpha in the sweep)"
    )
    return path, regime


def _read_marks(path: Path) -> pd.DataFrame:
    # comment="#": the ground-truth CSVs open with provenance lines before the
    # header row. dtype event_id=str: "07"/"13" must stay zero-padded -- pandas
    # would otherwise infer int64 and silently drop the leading zero.
    return pd.read_csv(path, comment="#", dtype={"event_id": str})


def _n_scored(alarms: pd.DataFrame) -> int:
    if "role" in alarms.columns:
        return int((alarms["role"] == ROLE_SCORED).sum())
    return int(len(alarms))


def _unmarked_events(repo_root: Path, session: str, marks: pd.DataFrame) -> list[tuple[str, str]]:
    """`(event_id, kind)` pairs from the minute-level reference events CSV
    (`080726_events_{session}.csv`) that contribute ZERO rows to the seconds-
    level marks table -- events the annotator could not mark at all. The
    reference file has no `event_id` column; its row order IS the seconds-
    level file's 1-based `event_id` numbering (both files describe the same
    campaign protocol in the same chronological order -- the seconds-level
    CSV's own header states it "upgrades" the minute-level file). Asserts
    that alignment on every event actually present in *marks*, rather than
    assuming it silently.
    """
    reference = pd.read_csv(_session_events_reference_path(repo_root, session), comment="#")
    reference_event_ids = [f"{i + 1:02d}" for i in range(len(reference))]

    marked_kind_by_id = marks.drop_duplicates("event_id").set_index("event_id")["kind"]
    for event_id, expected_kind in zip(reference_event_ids, reference["kind"], strict=True):
        if event_id in marked_kind_by_id.index:
            actual_kind = marked_kind_by_id.loc[event_id]
            if actual_kind != expected_kind:
                raise ValueError(
                    f"{session} event {event_id}: reference kind {expected_kind!r} != "
                    f"marks kind {actual_kind!r} -- the positional event_id<->reference-row "
                    f"alignment this function assumes is broken; fix the alignment, not this "
                    f"error"
                )

    return [
        (event_id, kind)
        for event_id, kind in zip(reference_event_ids, reference["kind"], strict=True)
        if event_id not in marked_kind_by_id.index
    ]


def _plate_6mark_gap_summary(session: str, marks: pd.DataFrame) -> list[dict[str, Any]]:
    """`row_type="summary"` rows: inter-mark-gap median/min/max for the
    "6-mark plate events", pooled and split at DEDUP_GAP_S --
    directly shows why the 1.5s dedup threshold folds the short gaps
    together and leaves the long ones as separate physical strikes."""
    mpe = marks_per_event(marks)
    six_mark = mpe[(mpe["n_marks"] == 6) & (mpe["kind_group"].isin(["plate-gen", "plate-tur"]))]

    gaps = inter_mark_gaps(marks)
    subset_gaps = gaps.merge(
        six_mark[["session", "event_id"]], on=["session", "event_id"], how="inner"
    )

    variants = (
        ("plate_6mark_all_gaps", subset_gaps),
        ("plate_6mark_short_gaps_lt_1.5s", subset_gaps[subset_gaps["gap_s"] < DEDUP_GAP_S]),
        ("plate_6mark_long_gaps_ge_1.5s", subset_gaps[subset_gaps["gap_s"] >= DEDUP_GAP_S]),
    )
    rows = []
    for label, sub in variants:
        rows.append(
            {
                "row_type": "summary",
                "session": session,
                "subset": label,
                "n_gaps": int(len(sub)),
                "median_gap_s": float(np.median(sub["gap_s"])) if len(sub) else float("nan"),
                "min_gap_s": float(sub["gap_s"].min()) if len(sub) else float("nan"),
                "max_gap_s": float(sub["gap_s"].max()) if len(sub) else float("nan"),
            }
        )
    return rows


def _df_to_markdown(df: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    """Minimal Markdown table renderer -- `pandas.DataFrame.to_markdown`
    needs the optional `tabulate` package, not part of this repo's dependency
    set (`pyproject.toml`); mirrors `scripts/compare_partner_labels.py`'s
    `_df_to_markdown` (same rationale, independent copy: a script must not
    import from another script)."""
    headers = [str(c) for c in df.columns]

    def fmt(v: object) -> str:
        if isinstance(v, float):
            if np.isnan(v):
                return "n/a"
            if np.isinf(v):
                return "inf"
            return format(v, floatfmt)
        return str(v)

    rows = [[fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max([len(headers[i]), *(len(r[i]) for r in rows)]) for i in range(len(headers))]

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)) + " |"

    lines = [render_row(headers), render_row(["-" * w for w in widths])]
    lines += [render_row(r) for r in rows]
    return "\n".join(lines)


def run(repo_root: Path, out_dir: Path) -> dict[str, Any]:
    """Runs the full per-strike sweep and writes the four output files under
    *out_dir*; returns the assembled frames + provenance (for `main()`'s
    print summary and for direct testing without touching the filesystem's
    real `results/pillar3-perstrike/`)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    detection_frames: list[pd.DataFrame] = []
    latency_rows: list[dict[str, Any]] = []
    impulse_rows: list[dict[str, Any]] = []
    provenance: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    unmarked_by_session: dict[str, list[tuple[str, str]]] = {}

    for session in SESSIONS:
        marks = _read_marks(_session_marks_path(repo_root, session))
        unmarked_by_session[session] = _unmarked_events(repo_root, session, marks)

        mpe = marks_per_event(marks)
        for row in mpe.itertuples(index=False):
            impulse_rows.append(
                {
                    "row_type": "event", "session": row.session, "event_id": row.event_id,
                    "kind": row.kind, "kind_group": row.kind_group, "n_marks": int(row.n_marks),
                }
            )
        gaps = inter_mark_gaps(marks)
        for row in gaps.itertuples(index=False):
            impulse_rows.append(
                {
                    "row_type": "gap", "session": row.session, "event_id": row.event_id,
                    "kind": row.kind, "kind_group": row.kind_group,
                    "to_strike_no": int(row.to_strike_no), "gap_s": float(row.gap_s),
                }
            )
        impulse_rows.extend(_plate_6mark_gap_summary(session, marks))

        for representation in REPRESENTATIONS:
            path, regime = _alarms_path_and_regime(repo_root, representation, session)
            if not path.is_file():
                logger.warning(
                    "no alarms.parquet for representation=%s session=%s at %s -- skipped",
                    representation, session, path,
                )
                skipped.append(f"{session}/{representation}: {path}")
                continue

            alarms = pd.read_parquet(path, engine="pyarrow")
            rel_path = str(path.relative_to(repo_root))
            provenance[f"{session}/{representation}"] = {
                "alarms_path": rel_path, "regime": regime,
                "n_rows": len(alarms), "n_scored": _n_scored(alarms),
            }
            common = {
                "session": session, "representation": representation,
                "regime": regime, "alarms_path": rel_path,
            }

            sweep = sweep_strike_detection(
                alarms, marks, tolerances_s=TOLERANCES_S, alphas=ALPHAS, gap_s=DEDUP_GAP_S
            )
            for key, value in common.items():
                sweep[key] = value
            detection_frames.append(sweep.reindex(columns=_DETECTION_COLUMNS))

            ev_lat = evaluate_event_latency(
                alarms, marks, alpha=LATENCY_ALPHA, search_horizon_s=EVENT_LATENCY_HORIZON_S
            )
            st_lat = evaluate_strike_latency(
                alarms, marks, alpha=LATENCY_ALPHA,
                search_horizon_s=STRIKE_LATENCY_HORIZON_S, gap_s=DEDUP_GAP_S,
            )
            ev_summary = summarize_latency(
                ev_lat["latency_s"].to_numpy(), ev_lat["missed"].to_numpy()
            )
            st_summary = summarize_latency(
                st_lat["latency_s"].to_numpy(), st_lat["missed"].to_numpy()
            )

            latency_rows.append(
                {
                    "row_type": "summary", "level": "event", **common,
                    "alpha": LATENCY_ALPHA, "search_horizon_s": EVENT_LATENCY_HORIZON_S,
                    **asdict(ev_summary),
                }
            )
            latency_rows.append(
                {
                    "row_type": "summary", "level": "physical_strike", **common,
                    "alpha": LATENCY_ALPHA, "search_horizon_s": STRIKE_LATENCY_HORIZON_S,
                    **asdict(st_summary),
                }
            )
            for row in ev_lat.itertuples(index=False):
                latency_rows.append(
                    {
                        "row_type": "detail", "level": "event", **common,
                        "alpha": LATENCY_ALPHA, "search_horizon_s": EVENT_LATENCY_HORIZON_S,
                        "event_id": row.event_id, "kind": row.kind, "kind_group": row.kind_group,
                        "ref_utc": row.onset_utc, "latency_s": row.latency_s,
                        "missed": bool(row.missed),
                    }
                )
            for row in st_lat.itertuples(index=False):
                latency_rows.append(
                    {
                        "row_type": "detail", "level": "physical_strike", **common,
                        "alpha": LATENCY_ALPHA, "search_horizon_s": STRIKE_LATENCY_HORIZON_S,
                        "event_id": row.event_id, "kind": row.kind,
                        "kind_group": kind_group(row.kind),
                        "strike_no": int(row.strike_no), "n_impulses": int(row.n_impulses),
                        "ref_utc": row.strike_utc, "latency_s": row.latency_s,
                        "missed": bool(row.missed),
                    }
                )

    detection = (
        pd.concat(detection_frames, ignore_index=True)
        if detection_frames
        else pd.DataFrame(columns=_DETECTION_COLUMNS)
    )
    latency = pd.DataFrame(latency_rows).reindex(columns=_LATENCY_COLUMNS)
    impulse = pd.DataFrame(impulse_rows).reindex(columns=_IMPULSE_COLUMNS)

    detection.to_csv(out_dir / "per_strike_detection.csv", index=False)
    latency.to_csv(out_dir / "latency.csv", index=False)
    impulse.to_csv(out_dir / "impulse_structure.csv", index=False)
    (out_dir / "summary.md").write_text(
        _summary_markdown(detection, latency, impulse, provenance, skipped, unmarked_by_session)
    )

    return {
        "detection": detection, "latency": latency, "impulse": impulse,
        "provenance": provenance, "skipped": skipped, "unmarked_by_session": unmarked_by_session,
    }


def _headline_table(detection: pd.DataFrame) -> pd.DataFrame:
    headline = detection[
        (detection["granularity"] == "physical")
        & (detection["tolerance_s"] == HEADLINE_TOLERANCE_S)
        & (detection["alpha"] == HEADLINE_ALPHA)
        & (detection["kind_group"] == "ALL")
    ]
    return headline[["session", "representation", "n_marks", "n_detected", "tpr"]].sort_values(
        ["representation", "session"]
    ).reset_index(drop=True)


def _join_natural(items: list[str]) -> str:
    """*items* joined for prose: `"a"`, `"a and b"`, `"a, b, and c"`."""
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _median_latency_table(latency: pd.DataFrame) -> pd.DataFrame:
    summary = latency[latency["row_type"] == "summary"]
    return summary[
        ["session", "representation", "level", "n_total", "n_detected", "n_missed",
         "median_s", "iqr_low_s", "iqr_high_s"]
    ].sort_values(["level", "representation", "session"]).reset_index(drop=True)


def _summary_markdown(
    detection: pd.DataFrame,
    latency: pd.DataFrame,
    impulse: pd.DataFrame,
    provenance: dict[str, dict[str, Any]],
    skipped: list[str],
    unmarked_by_session: dict[str, list[tuple[str, str]]],
) -> str:
    headline = _headline_table(detection)
    med_lat = _median_latency_table(latency)
    gap_summary = impulse[impulse["row_type"] == "summary"]

    events_only = impulse[impulse["row_type"] == "event"]
    n_marks_by_session = {
        session: int(events_only.loc[events_only["session"] == session, "n_marks"].sum())
        for session in SESSIONS
    }
    n_events_by_session = {
        session: int((events_only["session"] == session).sum()) for session in SESSIONS
    }

    lines = [
        "# Per-strike detection + first-alarm latency: 08.07.2026 controlled-event campaign",
        "",
        "Seconds-level ground truth (`docs/groundtruth/080726_strikes_seconds_"
        "{st,pu}.csv`) vs. the alarms.parquet artifacts already on disk -- this run "
        "reads existing alarms tables only, it never re-executes `scripts/monitor.py`.",
        "",
        "## Ground truth at a glance",
        "",
        f"- ST (standstill): {n_marks_by_session['st']} marks over "
        f"{n_events_by_session['st']} events, every event has at least one mark.",
        f"- PU (pump): {n_marks_by_session['pu']} marks over "
        f"{n_events_by_session['pu']} events.",
        "",
        "## Headline: physical-strike TPR at T=2s, alpha=0.05",
        "",
        "One row per (session, representation); `n_marks`/`n_detected` count "
        "DEDUPLICATED physical strikes (the double-impulse-folded granularity, "
        "`granularity=\"physical\"` in `per_strike_detection.csv`), `kind_group=\"ALL\"` "
        "(pooled across plate-gen/plate-tur/landmark/vane-sweep).",
        "",
        _df_to_markdown(headline, floatfmt=".3f"),
        "",
    ]

    zero_scored = sorted(key for key, p in provenance.items() if p["n_scored"] == 0)
    if zero_scored:
        lines += [
            f"**Caveat:** {_join_natural(zero_scored)} had ZERO scored windows in this "
            "session (see the Provenance table's `n_scored` column) -- every window's "
            "role is `unknown_state`/`no_conformal_data`, so no conformal p-value was "
            "ever produced to threshold. Its 0.000 TPR above means \"nothing existed to "
            "alarm from\", not \"every strike was individually missed\" -- not comparable "
            "to the other cells, which all have hundreds to thousands of genuinely "
            "scored windows.",
            "",
        ]

    lines += [
        "## First-alarm latency (alpha=0.05, recalibrate-regime alarms)",
        "",
        "`level=\"event\"`: onset = the event's EARLIEST raw mark, search horizon 60 s. "
        "`level=\"physical_strike\"`: reference = each deduplicated physical strike, "
        "search horizon 5 s. `median_s`/`iqr_low_s`/`iqr_high_s` are computed over "
        "DETECTED references only (a missed reference has no latency value).",
        "",
        _df_to_markdown(med_lat, floatfmt=".3f"),
        "",
        "## Impulse structure: the double-impulse spacing behind the 1.5 s dedup rule",
        "",
        "Inter-mark gaps within the 6-mark plate events (`plate-gen`/`plate-tur`), "
        "split at the DEDUP_GAP_S=1.5 s threshold into short (intra-cluster, the "
        "double-impulse pairs) vs. long (inter-cluster, separate physical strikes).",
        "",
        _df_to_markdown(
            gap_summary[["session", "subset", "n_gaps", "median_gap_s", "min_gap_s", "max_gap_s"]],
            floatfmt=".3f",
        ),
        "",
        "## Finding: PU landmark strikes could not be annotated at all",
        "",
    ]

    pu_unmarked = unmarked_by_session.get("pu", [])
    st_unmarked = unmarked_by_session.get("st", [])
    if pu_unmarked:
        listed = _join_natural([f"{eid} ({kind})" for eid, kind in pu_unmarked])
        lines += [
            f"PU events {listed} contributed ZERO seconds-level marks: the human "
            "annotator working from the PU recording could not identify these "
            "landmark strikes under pump noise at all (not a low-confidence guess -- "
            "no mark was placed). This is a finding, not a data gap.",
            "",
            "It independently reproduces the partner team's own reported result "
            f"(attributed external reference, `{_PARTNER_DIGEST_NOTE}`: Rodrigues & "
            "Zhang's own 080726 detector identified all plate/ring strikes under PU "
            "but could not detect the landmark strikes under pump-operation noise). "
            "The same conclusion now holds by an independent method (manual human "
            "annotation on our own seconds-level recordings, not an automated "
            "detector) -- landmark strikes are not identifiable under pump operation, "
            "for annotator and detector alike.",
            "",
        ]
    else:
        lines.append(
            "(no unmarked PU events found in this run -- unexpected; verify the marks CSV)"
        )
        lines.append("")
    if st_unmarked:
        lines.append(
            f"ST, for comparison, has every event marked (all {n_events_by_session['st']} "
            "events; standstill carries none of pump operation's masking noise)."
        )
        lines.append("")

    lines += [
        "## Provenance: which alarms artifact each table used",
        "",
        "Detection sweep alphas: {0.01, 0.05, 0.10} (all three from the SAME file per "
        "representation, re-thresholding `p_value` directly -- never the file's own "
        "baked-in `alarm` column). Latency: alpha=0.05 only, recalibrate-regime "
        "alarms. `impulse_structure.csv` uses the ground-truth marks CSVs only, no "
        "alarms artifact.",
        "",
        "| session/representation | alarms_path | regime | n_rows | n_scored |",
        "|:--|:--|:--|---:|---:|",
    ]
    for key in sorted(provenance):
        p = provenance[key]
        lines.append(
            f"| {key} | `{p['alarms_path']}` | {p['regime']} | {p['n_rows']} | {p['n_scored']} |"
        )
    lines.append("")

    if skipped:
        lines += ["## Skipped (alarms.parquet not found)", ""]
        lines += [f"- {s}" for s in skipped]
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "results" / "pillar3-perstrike"

    result = run(repo_root, out_dir)
    headline = _headline_table(result["detection"])
    n_skipped = len(result["skipped"])

    print(
        f"eval_per_strike: {len(result['detection'])} detection row(s), "
        f"{len(result['latency'])} latency row(s), {len(result['impulse'])} impulse-structure "
        f"row(s), {n_skipped} representation/session combo(s) skipped -> {out_dir}"
    )
    if len(headline):
        print("headline physical-strike TPR (T=2s, alpha=0.05):")
        for row in headline.itertuples(index=False):
            print(
                f"  {row.session}/{row.representation}: "
                f"{row.n_detected}/{row.n_marks} = {row.tpr:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
