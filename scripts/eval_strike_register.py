"""Definitive per-strike evaluation: the final 180-strike protocol verification
register (`docs/groundtruth/verification-080726/strikes_register_{st,pu}.csv`)
against every alarms.parquet representation already on disk.

This is a NEW canonical ground-truth basis for per-strike evaluation --
distinct from the seconds-level annotator marks (`docs/groundtruth/
080726_strikes_seconds_{st,pu}.csv`, still what `scripts/eval_per_strike.py`
reads). The register is a second, independent verification pass: it pairs
the annotator's by-ear marks against a reproduction of the measurement
partner's own impulse-detection method run on our own raw audio, consolidates
both into one 90-protocol-slot-per-session table (12 fixed positions x3 +
18 guide-vane covers x3), and resolves the slots neither pass could measure
with a final ear-cued listening pass (`docs/groundtruth/verification-080726/
README.md`).

**Primary evaluation set: ALL180 -- every one of the 90 protocol slots per
session, pooled** (`source` in {both, annotated-only, detector-only} UNION
`source` starting with "ear-cued"). Every slot carries a `t_utc`, measured or
cued, so pooling gives full 90/90 coverage. `detection_all180.csv`,
`latency_all180.csv`, `per_kind_headline.csv`, and every table in
`REPORT.md` run on this pooled basis.

**Secondary: the measured-only / ear-cued-only split**, kept because it falls
out of the exact same computation for free and answers a different question
(is a cued timestamp behaving differently from a genuinely measured one) --
`detection_by_source.csv` (tagged `source_set` in {measured, ear-cued}), plus
one measured-only and one ear-cued-only table in `REPORT.md` for reference.
Measured: ST 88/90, PU 84/90. Ear-cued: ST 2/90 (vane_18 strikes 2-3 only),
PU 6/90 (landmark-A_kugelschieber + landmark-C_EG, all 3 strikes each -- the
two positions unreadable under pump noise for annotator and detector alike).

The register's rows are already one-per-INTENDED-strike (unlike the seconds-
level marks CSV, which records every acoustic impulse including rebounds).
`rowii.eval.per_strike.sweep_strike_detection`'s `deduplicate_marks` "physical
strike" folding (default 1.5 s gap) is therefore NOT applied to the detection
sweep here -- most same-slot triples land under 1.5 s apart and would
wrongly fold three deliberate, separately-timed strikes into one. Only the
`granularity="impulse"` (raw marks, no folding) rows are kept. Latency is the
one exception: `evaluate_strike_latency` is used at its documented default
(`gap_s=1.5`), so the per-slot triples DO fold there, exactly as the module's
own physical-strike latency semantics intend.

Six representations, six alarms.parquet sources (all read-only, all already
on disk -- this script never runs `scripts/monitor.py` and never writes under
`results/step2/`, `results/pillar3/`, or `results/rescoring-subsecond/`):

- `fusion`, `audio-beats`, `vibration`: `results/step2/once-calibrated/
  <representation>/monitor/080726-<session>_strikes/recalibrate/
  alarms.parquet`.
- `audio`, `audio-student`: `results/pillar3/080726-<session>_strikes/
  <representation>-a0.05/alarms.parquet`. Pillar-3 exports one alarms.parquet
  PER (representation, alpha), but `p_value`/`t_utc_ns`/`role` are IDENTICAL
  across a representation's -a0.01/-a0.05/-a0.10 exports (re-verified here,
  same finding `scripts/eval_per_strike.py` already documents) -- the -a0.05
  file is read as the one canonical `p_value` source, re-thresholded for
  every alpha in the sweep.
- `audio-beats-fine`: `results/rescoring-subsecond/080726-<session>_strikes/
  alarms.parquet` -- the SAME audio-beats snapshot/calibration, scored on a
  0.25 s hop instead of 1.0 s (`results/rescoring-subsecond/REPORT.md`).

Detection is always re-thresholded from `p_value` directly
(`rowii.eval.per_strike.mark_detected`'s own contract), never read off a
file's baked-in `alarm` column.

Outputs -> `results/strike-register-eval/`: `detection_all180.csv`,
`latency_all180.csv`, `per_kind_headline.csv`, `detection_by_source.csv`,
`REPORT.md`.

Usage:
    cd repos/rowii-monitor && .venv/bin/python scripts/eval_strike_register.py
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rowii.eval.events import ROLE_SCORED  # noqa: E402
from rowii.eval.per_strike import (  # noqa: E402
    evaluate_strike_latency,
    summarize_latency,
    sweep_strike_detection,
)

SESSIONS: tuple[str, ...] = ("st", "pu")
REPRESENTATIONS: tuple[str, ...] = (
    "fusion", "audio-beats", "vibration", "audio", "audio-student", "audio-beats-fine",
)
TOLERANCES_S: tuple[float, ...] = (0.5, 1.0, 2.0)
ALPHAS: tuple[float, ...] = (0.01, 0.05, 0.10)
HEADLINE_ALPHA = 0.05
HEADLINE_TOLERANCE_S = 1.0
LATENCY_ALPHA = 0.05
#: `rowii.eval.per_strike.evaluate_strike_latency`'s own documented defaults,
#: spelled out explicitly here so the report is self-describing even if the
#: module's defaults ever change.
LATENCY_SEARCH_HORIZON_S = 5.0
LATENCY_GAP_S = 1.5

_ONCE_CALIBRATED_REPS = frozenset({"fusion", "audio-beats", "vibration"})
_PILLAR3_COMBO: dict[str, str] = {"audio": "audio-a0.05", "audio-student": "audio-student-a0.05"}

_REGISTER_DIR = _REPO_ROOT / "docs" / "groundtruth" / "verification-080726"
_OUT_DIR = _REPO_ROOT / "results" / "strike-register-eval"

_MEASURED_SOURCES = frozenset({"both", "annotated-only", "detector-only"})
_SOURCE_SET_MEASURED = "measured"
_SOURCE_SET_EARCUED = "ear-cued"

_PROTOCOL_SLOTS_N = 90
_EXPECTED_MEASURED_N: dict[str, int] = {"st": 88, "pu": 84}
_EXPECTED_EARCUED_N: dict[str, int] = {"st": 2, "pu": 6}

#: Old marks-basis cross-check (`results/rescoring-subsecond/REPORT.md`):
#: audio-beats, COARSE (1.0 s hop) grid, impulse granularity, alpha=0.05,
#: tolerance=1.0 s, against the seconds-level marks CSV -- 118 ST / 87 PU raw
#: marks, NOT this register's 90/90 protocol strikes (different basis,
#: first-impulse vs. first-recovered-strike times, informational only).
_CROSSCHECK_OLD_TPR: dict[str, tuple[int, int]] = {"st": (117, 118), "pu": (87, 87)}
#: Hard sanity floor for the ALL180 (pooled) basis: audio-beats coarse,
#: alpha=0.05, tolerance=1.0s must detect at least this many of the 90
#: protocol strikes per session (90 minus at most the 8 ear-cued strikes
#: across both sessions) -- below this, re-inspect parsing before accepting.
_CROSSCHECK_FLOOR_N = 82

_KIND_BUCKETS: dict[str, tuple[str, ...]] = {
    "plates": ("plate-gen", "plate-tur"),
    "landmarks": ("landmark",),
    "vane-sweep": ("vane-sweep",),
}

_WINDOW_S = 1.0
_NS_PER_S = 1_000_000_000


# ---------------------------------------------------------------------------
# Ground truth: register -> marks
# ---------------------------------------------------------------------------


def _alarms_path(representation: str, session: str) -> Path:
    session_dir = f"080726-{session}_strikes"
    if representation in _ONCE_CALIBRATED_REPS:
        return (
            _REPO_ROOT / "results" / "step2" / "once-calibrated" / representation / "monitor"
            / session_dir / "recalibrate" / "alarms.parquet"
        )
    if representation == "audio-beats-fine":
        return _REPO_ROOT / "results" / "rescoring-subsecond" / session_dir / "alarms.parquet"
    combo = _PILLAR3_COMBO[representation]
    return _REPO_ROOT / "results" / "pillar3" / session_dir / combo / "alarms.parquet"


def _kind_from_slot(slot: str) -> str:
    """Task contract: a slot starting with "vane" collapses to the fixed
    `"vane-sweep"` kind (`rowii.eval.per_strike.kind_group`'s exact-match
    branch); `plate-*`/`landmark-*` slots are already spelled exactly as
    that module's prefix-matched `kind` vocabulary expects -- kept verbatim.
    """
    return "vane-sweep" if slot.startswith("vane") else slot


def _load_register(session: str) -> pd.DataFrame:
    path = _REGISTER_DIR / f"strikes_register_{session}.csv"
    register = pd.read_csv(path, comment="#")
    register["t_utc"] = pd.to_datetime(register["t_utc"], utc=True, format="ISO8601")
    return register


def _to_marks(register: pd.DataFrame, session: str) -> pd.DataFrame:
    """*register* rows (already filtered to one of the two disjoint sets) as
    a `rowii.eval.per_strike._REQUIRED_MARK_COLUMNS`-satisfying marks table:
    `event_id` = slot (task contract), `kind` via `_kind_from_slot`."""
    register = register.reset_index(drop=True)
    marks = pd.DataFrame(index=register.index)
    marks["session"] = session
    marks["event_id"] = register["slot"]
    marks["kind"] = register["slot"].map(_kind_from_slot)
    marks["strike_no"] = register["strike_no"].astype(np.int64)
    marks["strike_utc"] = register["t_utc"]
    return marks


def _split_register(session: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """`(all180_marks, measured_marks, earcued_marks)` for *session* -- every
    one of the 90 protocol slots is classified into exactly one of the two
    disjoint secondary sets (asserted below); `all180_marks` is their union,
    the primary evaluation set (nothing dropped or double-counted)."""
    register = _load_register(session)
    is_measured = register["source"].isin(_MEASURED_SOURCES)
    is_earcued = register["source"].str.startswith("ear-cued")

    unclassified = register[~(is_measured | is_earcued)]
    if len(unclassified):
        raise ValueError(
            f"{session}: {len(unclassified)} register row(s) with an unrecognized "
            f"source value: {sorted(unclassified['source'].unique())!r}"
        )
    if (is_measured & is_earcued).any():
        raise ValueError(f"{session}: a register row matched both measured and ear-cued")
    if len(register) != _PROTOCOL_SLOTS_N:
        raise ValueError(
            f"{session}: register has {len(register)} rows, expected "
            f"{_PROTOCOL_SLOTS_N} protocol slots"
        )

    measured = _to_marks(register[is_measured], session)
    earcued = _to_marks(register[is_earcued], session)
    all180 = _to_marks(register, session)
    return all180, measured, earcued


# ---------------------------------------------------------------------------
# Detection sweep (impulse granularity only -- see module docstring)
# ---------------------------------------------------------------------------


def _detection_sweep(alarms: pd.DataFrame, marks: pd.DataFrame) -> pd.DataFrame:
    """Impulse-granularity slice of `sweep_strike_detection` across the full
    `TOLERANCES_S` x `ALPHAS` x kind_group grid. `granularity="physical"`
    rows are computed by `sweep_strike_detection` regardless (harmless, just
    unused here -- see module docstring for why "physical" folding is wrong
    for this register) and dropped.
    """
    sweep = sweep_strike_detection(alarms, marks, tolerances_s=TOLERANCES_S, alphas=ALPHAS)
    return sweep[sweep["granularity"] == "impulse"].reset_index(drop=True)


def _all_kind_rows(sweep: pd.DataFrame, *, session: str, representation: str) -> pd.DataFrame:
    """`kind_group="ALL"` rows of *sweep*, tidied to: `session, representation,
    alpha, tolerance_s, n, n_detected, tpr`."""
    all_rows = sweep[sweep["kind_group"] == "ALL"].copy()
    all_rows.insert(0, "representation", representation)
    all_rows.insert(0, "session", session)
    all_rows = all_rows.rename(columns={"n_marks": "n"})
    return all_rows[["session", "representation", "alpha", "tolerance_s", "n", "n_detected", "tpr"]]


def _kind_bucket_rows(sweep: pd.DataFrame, *, session: str, representation: str) -> pd.DataFrame:
    """Headline-setting (`HEADLINE_ALPHA`, `HEADLINE_TOLERANCE_S`) per-kind
    breakdown, `sweep_strike_detection`'s four `kind_group`s combined into the
    task's three buckets (`_KIND_BUCKETS`: plates = plate-gen + plate-tur,
    landmarks, vane-sweep)."""
    headline = sweep[
        (sweep["alpha"] == HEADLINE_ALPHA)
        & (sweep["tolerance_s"] == HEADLINE_TOLERANCE_S)
        & (sweep["kind_group"] != "ALL")
    ]
    rows = []
    for bucket, groups in _KIND_BUCKETS.items():
        subset = headline[headline["kind_group"].isin(groups)]
        n = int(subset["n_marks"].sum())
        n_detected = int(subset["n_detected"].sum())
        rows.append(
            {
                "session": session,
                "representation": representation,
                "kind_bucket": bucket,
                "n": n,
                "n_detected": n_detected,
                "tpr": n_detected / n if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single-strike-window accounting (fine vs. coarse audio-beats, task item 5)
# ---------------------------------------------------------------------------


def _mark_ns_sorted(marks: pd.DataFrame) -> np.ndarray:
    ts = marks["strike_utc"]
    return np.sort(ts.astype("datetime64[ns, UTC]").astype("int64").to_numpy())


def _single_strike_window_stats(
    alarms: pd.DataFrame, mark_ns: np.ndarray, *, alpha: float
) -> dict[str, int]:
    """Single-strike-window accounting (task item 5): a SCORED window is a
    single-strike window for a mark iff its `[t_utc_ns, t_utc_ns + 1s)` span
    contains that mark and no other mark of *mark_ns*; a mark counts as
    "detected via a single-strike window" iff ANY such window also alarms
    (`p_value < alpha`). Independent copy of `scripts/eval_subsecond_hop.py`'s
    `_marks_per_window`/`_single_strike_stats` logic (this repo's convention:
    a script never imports another script, only the shared `rowii.eval`
    library) -- containment is deliberately NOT `mark_detected`'s symmetric
    start-timestamp tolerance, so it is not something `rowii.eval.per_strike`
    exposes as a reusable function.
    """
    scored = alarms[alarms["role"] == ROLE_SCORED] if "role" in alarms.columns else alarms
    starts = scored["t_utc_ns"].to_numpy(dtype=np.int64)
    alarming = scored["p_value"].to_numpy(dtype=np.float64) < alpha

    window_ns = int(round(_WINDOW_S * _NS_PER_S))
    lo = np.searchsorted(mark_ns, starts, side="left")
    hi = np.searchsorted(mark_ns, starts + window_ns, side="left")
    counts = hi - lo
    single = counts == 1

    single_idx = np.unique(lo[single])
    alarming_single_idx = np.unique(lo[single & alarming])
    return {
        "n_marks": int(mark_ns.size),
        "n_marks_with_single_strike_window": int(single_idx.size),
        "n_marks_detected_via_single_strike_window": int(alarming_single_idx.size),
    }


# ---------------------------------------------------------------------------
# Markdown rendering (independent copy of the `_df_to_markdown` pattern used
# by `scripts/eval_per_strike.py` / `scripts/compare_partner_labels.py` --
# this repo's convention: a script never imports another script)
# ---------------------------------------------------------------------------


def _df_to_markdown(df: pd.DataFrame, *, floatfmt: str = ".3f") -> str:
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


def _fmt_frac(n_detected: int, n: int) -> str:
    tpr = n_detected / n if n else float("nan")
    return f"{tpr:.3f} ({n_detected}/{n})"


def _fmt_cell(cell: pd.DataFrame) -> str:
    """`_fmt_frac` of *cell*'s single row (`n_detected`, `n`), or `"n/a"` if
    *cell* is empty -- the common per-cell formatter for every pivoted
    report table below."""
    if not len(cell):
        return "n/a"
    row = cell.iloc[0]
    return _fmt_frac(int(row["n_detected"]), int(row["n"]))


def _pivot_by_session(detection: pd.DataFrame, *, alpha: float, tolerance_s: float) -> pd.DataFrame:
    """*detection* (tidy: session, representation, alpha, tolerance_s, n,
    n_detected, tpr) sliced to (*alpha*, *tolerance_s*) and pivoted to one row
    per representation, one column per session (`tpr (n_detected/n)`)."""
    headline = detection[(detection["alpha"] == alpha) & (detection["tolerance_s"] == tolerance_s)]
    rows = []
    for representation in REPRESENTATIONS:
        row: dict[str, Any] = {"representation": representation}
        for session in SESSIONS:
            cell = headline[
                (headline["representation"] == representation) & (headline["session"] == session)
            ]
            row[session.upper()] = _fmt_cell(cell)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(repo_root: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    all180_marks: dict[str, pd.DataFrame] = {}
    measured_marks: dict[str, pd.DataFrame] = {}
    earcued_marks: dict[str, pd.DataFrame] = {}
    for session in SESSIONS:
        all180, measured, earcued = _split_register(session)
        all180_marks[session] = all180
        measured_marks[session] = measured
        earcued_marks[session] = earcued

    alarms_cache: dict[tuple[str, str], pd.DataFrame] = {}
    provenance: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []

    detection_all180_parts: list[pd.DataFrame] = []
    detection_by_source_parts: list[pd.DataFrame] = []
    per_kind_parts: list[pd.DataFrame] = []
    latency_rows: list[dict[str, Any]] = []

    for session in SESSIONS:
        for representation in REPRESENTATIONS:
            path = _alarms_path(representation, session)
            if not path.is_file():
                skipped.append(f"{session}/{representation}: {path}")
                continue

            alarms = pd.read_parquet(path, engine="pyarrow")
            alarms_cache[(representation, session)] = alarms
            has_role = "role" in alarms.columns
            n_scored = int((alarms["role"] == ROLE_SCORED).sum()) if has_role else len(alarms)
            provenance[f"{session}/{representation}"] = {
                "alarms_path": str(path.relative_to(repo_root)),
                "n_rows": len(alarms),
                "n_scored": n_scored,
            }

            all180_sweep = _detection_sweep(alarms, all180_marks[session])
            detection_all180_parts.append(
                _all_kind_rows(all180_sweep, session=session, representation=representation)
            )
            per_kind_parts.append(
                _kind_bucket_rows(all180_sweep, session=session, representation=representation)
            )

            for source_set, marks in (
                (_SOURCE_SET_MEASURED, measured_marks[session]),
                (_SOURCE_SET_EARCUED, earcued_marks[session]),
            ):
                sweep = _detection_sweep(alarms, marks)
                tagged = _all_kind_rows(sweep, session=session, representation=representation)
                tagged.insert(2, "source_set", source_set)
                detection_by_source_parts.append(tagged)

            lat = evaluate_strike_latency(
                alarms, all180_marks[session],
                alpha=LATENCY_ALPHA,
                search_horizon_s=LATENCY_SEARCH_HORIZON_S,
                gap_s=LATENCY_GAP_S,
            )
            summary = summarize_latency(lat["latency_s"].to_numpy(), lat["missed"].to_numpy())
            latency_rows.append(
                {
                    "session": session,
                    "representation": representation,
                    "alpha": LATENCY_ALPHA,
                    "search_horizon_s": LATENCY_SEARCH_HORIZON_S,
                    "gap_s": LATENCY_GAP_S,
                    **asdict(summary),
                }
            )

    detection_all180 = pd.concat(detection_all180_parts, ignore_index=True)
    detection_by_source = pd.concat(detection_by_source_parts, ignore_index=True)
    per_kind_headline = pd.concat(per_kind_parts, ignore_index=True)
    latency_all180 = pd.DataFrame(latency_rows)

    # Fine-vs-coarse single-strike-window accounting (audio-beats only, ALL180 marks).
    single_strike_rows: list[dict[str, Any]] = []
    _FINE_COARSE_GRIDS = (("audio-beats", "coarse-1.00s"), ("audio-beats-fine", "fine-0.25s"))
    for session in SESSIONS:
        mark_ns = _mark_ns_sorted(all180_marks[session])
        for representation, grid in _FINE_COARSE_GRIDS:
            key = (representation, session)
            if key not in alarms_cache:
                continue
            stats = _single_strike_window_stats(alarms_cache[key], mark_ns, alpha=HEADLINE_ALPHA)
            row = {
                "session": session, "representation": representation,
                "grid": grid, "alpha": HEADLINE_ALPHA, **stats,
            }
            single_strike_rows.append(row)
    single_strike = pd.DataFrame(single_strike_rows)

    detection_all180.to_csv(out_dir / "detection_all180.csv", index=False)
    latency_all180.to_csv(out_dir / "latency_all180.csv", index=False)
    per_kind_headline.to_csv(out_dir / "per_kind_headline.csv", index=False)
    detection_by_source.to_csv(out_dir / "detection_by_source.csv", index=False)

    return {
        "all180_marks": all180_marks,
        "measured_marks": measured_marks,
        "earcued_marks": earcued_marks,
        "detection_all180": detection_all180,
        "detection_by_source": detection_by_source,
        "latency_all180": latency_all180,
        "per_kind_headline": per_kind_headline,
        "single_strike": single_strike,
        "provenance": provenance,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------


def _alpha_compare_table(detection_all180: pd.DataFrame) -> pd.DataFrame:
    reps = ("audio-beats", "audio-student")
    rows = []
    for representation in reps:
        row: dict[str, Any] = {"representation": representation}
        for session in SESSIONS:
            for alpha, label in ((HEADLINE_ALPHA, "a0.05"), (0.01, "a0.01")):
                cell = detection_all180[
                    (detection_all180["representation"] == representation)
                    & (detection_all180["session"] == session)
                    & (detection_all180["alpha"] == alpha)
                    & (detection_all180["tolerance_s"] == HEADLINE_TOLERANCE_S)
                ]
                col = f"{session.upper()} {label}"
                row[col] = _fmt_cell(cell)
        rows.append(row)
    return pd.DataFrame(rows)


def _detected_over_total(row: pd.Series) -> str:
    return f"{int(row['n_detected'])}/{int(row['n_total'])}"


def _latency_table(latency: pd.DataFrame) -> pd.DataFrame:
    out = latency.copy()
    out["detected/total"] = out.apply(_detected_over_total, axis=1)
    out["median_s"] = out["median_s"].astype(float)
    out["iqr_s"] = out.apply(
        lambda r: (
            "n/a" if pd.isna(r["iqr_low_s"]) else f"[{r['iqr_low_s']:.3f}, {r['iqr_high_s']:.3f}]"
        ),
        axis=1,
    )
    return out[["session", "representation", "detected/total", "median_s", "iqr_s"]].sort_values(
        ["representation", "session"]
    ).reset_index(drop=True)


def _per_kind_table(per_kind_headline: pd.DataFrame, session: str) -> pd.DataFrame:
    sub = per_kind_headline[per_kind_headline["session"] == session]
    rows = []
    for representation in REPRESENTATIONS:
        row: dict[str, Any] = {"representation": representation}
        for bucket in _KIND_BUCKETS:
            cell = sub[(sub["representation"] == representation) & (sub["kind_bucket"] == bucket)]
            row[bucket] = _fmt_cell(cell)
        rows.append(row)
    return pd.DataFrame(rows)


_SINGLE_STRIKE_COLUMN_RENAME: dict[str, str] = {
    "n_marks_with_single_strike_window": "n_with_single_window",
    "n_marks_detected_via_single_strike_window": "n_detected_via_single_window",
}
_SINGLE_STRIKE_COLUMNS: list[str] = [
    "session", "representation", "grid", "n_marks",
    "n_with_single_window", "n_detected_via_single_window",
]


def _single_strike_table(single_strike: pd.DataFrame) -> pd.DataFrame:
    renamed = single_strike.rename(columns=_SINGLE_STRIKE_COLUMN_RENAME)
    return renamed[_SINGLE_STRIKE_COLUMNS]


def _crosscheck(detection_all180: pd.DataFrame) -> tuple[list[str], bool]:
    lines = []
    all_pass = True
    for session in SESSIONS:
        cell = detection_all180[
            (detection_all180["representation"] == "audio-beats")
            & (detection_all180["session"] == session)
            & (detection_all180["alpha"] == HEADLINE_ALPHA)
            & (detection_all180["tolerance_s"] == HEADLINE_TOLERANCE_S)
        ].iloc[0]
        n_detected, n = int(cell["n_detected"]), int(cell["n"])
        floor_pass = n_detected >= _CROSSCHECK_FLOOR_N
        all_pass = all_pass and floor_pass
        old_num, old_den = _CROSSCHECK_OLD_TPR[session]
        lines.append(
            f"- {session.upper()} ALL180: audio-beats coarse, alpha=0.05, tol=1.0s detects "
            f"{n_detected}/{n} ({n_detected / n:.4f}) -- hard floor >= {_CROSSCHECK_FLOOR_N}/90: "
            f"{'PASS' if floor_pass else 'FAIL -- INVESTIGATE PARSING'}. Informational-only "
            f"comparison to the old, different-basis marks-level result: {old_num}/{old_den} "
            f"({old_num / old_den:.4f}) on 118/87 raw acoustic-impulse marks (not this "
            "register's 90 protocol strikes -- proximity expected, not equality)."
        )
    return lines, all_pass


def _write_report(result: dict[str, Any], out_dir: Path) -> None:
    detection_all180 = result["detection_all180"]
    detection_by_source = result["detection_by_source"]
    latency_all180 = result["latency_all180"]
    per_kind_headline = result["per_kind_headline"]
    single_strike = result["single_strike"]
    provenance = result["provenance"]
    skipped = result["skipped"]
    all180_marks = result["all180_marks"]
    measured_marks = result["measured_marks"]
    earcued_marks = result["earcued_marks"]

    is_measured = detection_by_source["source_set"] == _SOURCE_SET_MEASURED
    is_earcued = detection_by_source["source_set"] == _SOURCE_SET_EARCUED
    measured_only = detection_by_source[is_measured].drop(columns="source_set")
    earcued_only = detection_by_source[is_earcued].drop(columns="source_set")

    def _headline(df: pd.DataFrame) -> pd.DataFrame:
        return _pivot_by_session(df, alpha=HEADLINE_ALPHA, tolerance_s=HEADLINE_TOLERANCE_S)

    headline_all180 = _headline(detection_all180)
    headline_measured = _headline(measured_only)
    headline_earcued = _headline(earcued_only)

    crosscheck_lines, crosscheck_pass = _crosscheck(detection_all180)

    lines: list[str] = [
        "# Definitive per-strike evaluation: 08.07.2026 final 180-strike protocol register",
        "",
        "Ground truth: `docs/groundtruth/verification-080726/strikes_register_{st,pu}.csv`. "
        "**Primary evaluation set is ALL180: every one of the 90 protocol slots per session, "
        "pooled** (measured + ear-cued together -- every slot carries a `t_utc`). The "
        "measured-only / ear-cued-only split is kept as secondary context. Detection is "
        "always re-thresholded from `p_value` directly "
        "(`rowii.eval.per_strike.mark_detected`), never read off a file's own baked-in "
        "`alarm` column.",
        "",
        "## Ground truth coverage",
        "",
        "| session | protocol slots (ALL180, primary) | measured (secondary) | "
        "ear-cued (secondary) |",
        "|:--|--:|--:|--:|",
    ]
    for session in SESSIONS:
        lines.append(
            f"| {session.upper()} | {len(all180_marks[session])} / {_PROTOCOL_SLOTS_N} | "
            f"{len(measured_marks[session])} (expected {_EXPECTED_MEASURED_N[session]}) | "
            f"{len(earcued_marks[session])} (expected {_EXPECTED_EARCUED_N[session]}) |"
        )
    lines += [
        "",
        "Every one of the 180 protocol slots (90 ST + 90 PU) is classified into exactly one "
        "of the measured/ear-cued sets (`_split_register` asserts this); ALL180 is their "
        "union, so ALL180 n=90 for every session regardless of alpha/tolerance/representation.",
        "",
        "## (a) HEADLINE: TPR at alpha=0.05, tolerance=1.0s, ALL180 (primary, n=90/90)",
        "",
        "`n_detected/n` per cell; n=90 for both sessions and every representation.",
        "",
        _df_to_markdown(headline_all180),
        "",
        "### For reference: measured-only and ear-cued-only at the same setting (secondary)",
        "",
        f"Measured-only (ST n={_EXPECTED_MEASURED_N['st']}, PU n={_EXPECTED_MEASURED_N['pu']}):",
        "",
        _df_to_markdown(headline_measured),
        "",
        f"Ear-cued-only (ST n={_EXPECTED_EARCUED_N['st']}, PU n={_EXPECTED_EARCUED_N['pu']}) -- "
        "cued, not measured, timestamps; never pooled into a \"measured\" claim:",
        "",
        _df_to_markdown(headline_earcued),
        "",
        "## (b) alpha=0.01 vs alpha=0.05 for audio-beats and audio-student (tol=1.0s, ALL180)",
        "",
        _df_to_markdown(_alpha_compare_table(detection_all180)),
        "",
        "## (c) First-alarm latency, ALL180, alpha=0.05",
        "",
        f"`rowii.eval.per_strike.evaluate_strike_latency` at its documented defaults "
        f"(search_horizon_s={LATENCY_SEARCH_HORIZON_S}, gap_s={LATENCY_GAP_S} -- physical-"
        "strike granularity: same-slot strikes under the 1.5s gap fold into one reference "
        "instant for this table only, per the module's own physical-strike latency "
        "semantics; the detection tables never fold). `median_s`/`iqr_s` over DETECTED "
        "references only.",
        "",
        _df_to_markdown(_latency_table(latency_all180)),
        "",
        "## (d) Per-kind breakdown at the headline setting (alpha=0.05, tol=1.0s, ALL180)",
        "",
        "`plates` pools the 9 plate positions (4 plate-gen + 5 plate-tur, n=27); `landmarks` "
        "(n=9) and `vane-sweep` (n=54) are `rowii.eval.per_strike.kind_group`'s own groups. "
        "On the ALL180 basis these n's are uniform across ST and PU (27+9+54=90) -- the "
        "measured-only split is where the ear-cued exclusions would make PU landmarks n=3 "
        "and ST vane-sweep n=52 instead; see `detection_by_source.csv`. Cell = "
        "`tpr (n_detected/n)`.",
        "",
        "### ST",
        "",
        _df_to_markdown(_per_kind_table(per_kind_headline, "st")),
        "",
        "### PU",
        "",
        _df_to_markdown(_per_kind_table(per_kind_headline, "pu")),
        "",
        f"## (e) Ear-cued set, reported apart (ST n={_EXPECTED_EARCUED_N['st']}, "
        f"PU n={_EXPECTED_EARCUED_N['pu']}), alpha=0.05, tol=1.0s",
        "",
        "Repeated from (a)'s reference block for visibility. ST ear-cued = vane_18 strikes "
        "2-3 only. PU ear-cued = landmark-A_kugelschieber and landmark-C_EG, all 3 strikes "
        "each (the two positions unreadable under pump noise for both the annotator and the "
        "detector reproduction alike). These are cued, not measured, timestamps -- included "
        "in the ALL180 primary basis, but never claimed as an independent measurement.",
        "",
        _df_to_markdown(headline_earcued),
        "",
        "## (f) Fine (0.25s hop) vs. coarse (1.0s hop) audio-beats, ALL180",
        "",
        "Headline TPR (alpha=0.05, tol=1.0s), both grids, from the same table as (a):",
        "",
        _df_to_markdown(
            headline_all180[
                headline_all180["representation"].isin(["audio-beats", "audio-beats-fine"])
            ]
        ),
        "",
        "Single-strike-window accounting (alpha=0.05, ALL180 marks): a SCORED window is a "
        "single-strike window for a mark iff its 1s span `[start, start+1s)` contains that "
        "mark and no other ALL180 mark of the session; a mark is \"detected via\" one iff "
        "ANY such window also alarms. On the fine grid, windows overlap, so \"ANY\" is "
        "evaluated across all of them.",
        "",
        _df_to_markdown(_single_strike_table(single_strike)),
        "",
        "## Cross-check (sanity gate on the ALL180 basis)",
        "",
    ]
    lines += crosscheck_lines
    lines += ["", "## Sanity notes", ""]

    lines.append(
        f"- **Cross-check gate:** {'PASSED' if crosscheck_pass else 'FAILED'} for both "
        f"sessions (floor {_CROSSCHECK_FLOOR_N}/90)."
    )

    zero_scored = sorted(key for key, p in provenance.items() if p["n_scored"] == 0)
    if zero_scored:
        lines.append(
            f"- **Zero scored windows:** {', '.join(zero_scored)} -- every window's role is "
            "`unknown_state`/`no_conformal_data` in this session, so no conformal p-value "
            "was ever produced to threshold. Its 0.000 TPR in every table above means "
            "\"nothing existed to alarm from\", not \"every strike was individually missed\" "
            "-- a known, expected degeneracy (ST/vibration: the standstill session never "
            "leaves the single calibrated state, so vibration's per-state conformal "
            "calibration has no second state to score against)."
        )
    else:
        lines.append("- No representation/session combination had zero scored windows.")

    if skipped:
        lines += ["", "**Skipped (alarms.parquet not found):**", ""]
        lines += [f"- {s}" for s in skipped]

    lines += [
        "", "## Provenance", "",
        "| session/representation | alarms_path | n_rows | n_scored |",
        "|:--|:--|--:|--:|",
    ]
    for key in sorted(provenance):
        p = provenance[key]
        lines.append(f"| {key} | `{p['alarms_path']}` | {p['n_rows']} | {p['n_scored']} |")
    lines.append("")

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    result = run(_REPO_ROOT, _OUT_DIR)
    _write_report(result, _OUT_DIR)

    print(f"eval_strike_register: wrote outputs to {_OUT_DIR}")
    for session in SESSIONS:
        print(
            f"  {session}: ALL180 n={len(result['all180_marks'][session])}, "
            f"measured n={len(result['measured_marks'][session])} "
            f"(expected {_EXPECTED_MEASURED_N[session]}), "
            f"ear-cued n={len(result['earcued_marks'][session])} "
            f"(expected {_EXPECTED_EARCUED_N[session]})"
        )
    if result["skipped"]:
        print(f"  skipped {len(result['skipped'])} representation/session combo(s)")
    print("headline TPR (alpha=0.05, tol=1.0s, ALL180 n=90/90):")
    headline = _pivot_by_session(
        result["detection_all180"], alpha=HEADLINE_ALPHA, tolerance_s=HEADLINE_TOLERANCE_S
    )
    print(headline.to_string(index=False))
    _, crosscheck_pass = _crosscheck(result["detection_all180"])
    print(f"cross-check gate: {'PASS' if crosscheck_pass else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
