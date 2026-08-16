"""Per-strike evaluation of the MAD fixed-threshold baseline (`scripts/
run_mad_baseline.py`) against the seconds-level hammer-strike ground truth
(`docs/groundtruth/080726_strikes_seconds_{st,pu}.csv`), using the SAME
per-strike harness (`rowii.eval.per_strike`) `scripts/eval_per_strike.py` uses
for the representation monitors -- so the MAD baseline's numbers land in a
directly comparable shape next to `results/pillar3-perstrike/`.

**Why a companion script, not an extension of `eval_per_strike.py`.**
`eval_per_strike.py` reads already-committed `alarms.parquet` artifacts (one
per representation) and re-thresholds their `p_value` column through `rowii.
eval.per_strike`'s ALPHA-SWEEP functions (`mark_detected`, `sweep_strike_
detection`, `evaluate_strike_latency`). The MAD baseline has no such artifact
and no `p_value`: `scripts/run_mad_baseline.py` commissions ONE fixed
threshold per k on the B1 pool and applies it once (`score > threshold`) -- a
pre-thresholded BINARY decision, never a continuous p-value to sweep.
Encoding that decision as a fake `p_value` (`0.0` for "alarm", `1.0` for "no
alarm") would make `rowii.eval.per_strike`'s existing alpha-based functions
run, but the `alpha` argument would then be purely decorative (any `alpha` in
`(0, 1]` gives the identical result) -- indistinguishable from a genuine
p-value at the call site. This script instead uses the binary-alarm-stream
adapter path `rowii.eval.per_strike` exposes for exactly this case
(`mark_detected_binary` / `sweep_strike_detection_binary` /
`evaluate_strike_latency_binary` -- see that module's docstring, "Binary /
pre-thresholded alarm streams"), which shares every tolerance-matching and
first-alarm-latency computation with the p_value path (both call the same
private `_match_within_tolerance` / `_first_alarm_latency` helpers) but never
manufactures a p-value.

**Threshold provenance -- reused, not recomputed.** The B1-pool commissioning
statistics and the `k1pct` / `k5` thresholds are read directly from the
already-committed `results/mad-baseline/mad_baseline.json` (written by
`scripts/run_mad_baseline.py`'s own run) rather than re-derived: this script
only recomputes the per-window SCORE for the two 08.07 sessions
(`rowii.anomaly.mad_baseline.band_energy_score`, the SAME function `run_mad_
baseline.py` calls -- never reimplemented), then applies the two already-
committed thresholds to it. The log-mel geometry, target mel bins, and mic
rate are cross-checked against the JSON's own recorded values before scoring
(the same run/variant/config combination `rowii.pipeline.prepare_run`'s cache
fingerprint already keys on), so any drift between this run and the committed
one is caught loudly, never silently absorbed.

Outputs (`results/mad-baseline/perstrike/`):

- `alarms_pu.csv` / `alarms_st.csv` -- the EXPORTED per-window MAD alarm
  stream, one row per valid window: `t_utc_ns` (int64 window-start UTC ns),
  `score` (float64, `band_energy_score`'s own output), `alarm_k1pct` /
  `alarm_k5` (bool, `score > threshold` at each of the two reported
  thresholds -- `scripts/run_mad_baseline.py`'s own `_alarms_frame` shape,
  extended with both thresholds as sibling columns). This is the schema
  `mark_detected_binary` etc. consume directly (`alarm_column="alarm_k1pct"`
  / `"alarm_k5"`), and the artifact a later run can re-evaluate without
  recomputing the score.
- `per_strike_detection.csv` -- `rowii.eval.per_strike.
  sweep_strike_detection_binary` per (session, k_label): the SAME
  `tolerances_s` grid `eval_per_strike.py` uses (1, 2, 5 s), both
  granularities (impulse, physical), all five kind_group rows.
- `latency.csv` -- `rowii.eval.per_strike.evaluate_strike_latency_binary` +
  `summarize_latency` per (session, k_label): one `row_type="summary"` row
  (5 s search horizon, 1.5 s dedup gap -- the SAME physical-strike constants
  `eval_per_strike.py` uses) followed by one `row_type="detail"` row per
  physical strike.
- `summary.md` -- short summary: detected/total per session (k1pct and k5),
  median first-alarm latency of detected strikes, threshold provenance.

Usage:
    cd repos/rowii-monitor && .venv/bin/python scripts/eval_mad_per_strike.py
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.anomaly.mad_baseline import (  # noqa: E402
    band_energy_score,
    high_band_mel_bins,
    logmel_geometry,
)
from rowii.config import Config, load_config  # noqa: E402
from rowii.eval.per_strike import (  # noqa: E402
    evaluate_strike_latency_binary,
    kind_group,
    summarize_latency,
    sweep_strike_detection_binary,
)
from rowii.io.dataset import Run, discover  # noqa: E402
from rowii.io.gantner import read_header  # noqa: E402
from rowii.pipeline import PreparedRun, prepare_run  # noqa: E402

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GROUNDTRUTH_DIR = _REPO_ROOT / "docs" / "groundtruth"
_BASELINE_DIR = _REPO_ROOT / "results" / "mad-baseline"
_BASELINE_JSON_PATH = _BASELINE_DIR / "mad_baseline.json"

_VARIANT = "logmel"
_LOGMEL_PRIMARY_STREAM = "RAWGeneratorMic__0"
"""Mirrors `scripts/run_mad_baseline.py`'s own `_LOGMEL_PRIMARY_STREAM`
(script-sibling duplication precedent -- a script never imports another
script's internals, module docstrings across this repo's `scripts/` all
state this rule)."""

_K_LABELS: tuple[str, ...] = ("k1pct", "k5")
"""The two thresholds this script reports (task scope): `k1pct` (tuned once
to flag ~1% of the B1 pool -- the "careful practitioner" variant) and `k5`
(the loosest member of the conventional grid `{5, 10, 20, 30, 40}`, reported
for contrast -- `scripts/run_mad_baseline.py`'s own module docstring)."""

_SESSIONS: dict[str, tuple[str, Path]] = {
    "pu": ("080726-pu_strikes", _GROUNDTRUTH_DIR / "080726_strikes_seconds_pu.csv"),
    "st": ("080726-st_strikes", _GROUNDTRUTH_DIR / "080726_strikes_seconds_st.csv"),
}

_TOLERANCES_S: tuple[float, ...] = (1.0, 2.0, 5.0)
"""SAME grid as `scripts/eval_per_strike.py`'s own `TOLERANCES_S`."""
_DEDUP_GAP_S = 1.5
"""SAME double-impulse dedup gap as `scripts/eval_per_strike.py`."""
_STRIKE_LATENCY_HORIZON_S = 5.0
"""SAME physical-strike search horizon as `scripts/eval_per_strike.py`'s own
`STRIKE_LATENCY_HORIZON_S`."""
_HEADLINE_TOLERANCE_S = 2.0
"""SAME headline tolerance as `scripts/eval_per_strike.py`'s own
`HEADLINE_TOLERANCE_S` -- keeps the two scripts' headline numbers directly
comparable."""

_DETECTION_COLUMNS: list[str] = [
    "session", "k_label", "threshold", "granularity", "tolerance_s",
    "kind_group", "n_marks", "n_detected", "tpr",
]
_LATENCY_COLUMNS: list[str] = [
    "row_type", "session", "k_label", "threshold", "search_horizon_s",
    "n_total", "n_detected", "n_missed", "median_s", "iqr_low_s", "iqr_high_s",
    "event_id", "kind", "kind_group", "strike_no", "n_impulses", "ref_utc",
    "latency_s", "missed",
]


def _mic_rate_hz(run: Run) -> float:
    """*run*'s primary generator-mic sample rate -- mirrors `scripts/
    run_mad_baseline.py`'s own `_mic_rate_hz` (script-sibling duplication)."""
    files = sorted(run.files[_LOGMEL_PRIMARY_STREAM], key=lambda f: f.start_utc_hint)
    return read_header(files[0].path).sample_rate_hz


def _prepare_and_score_checked(
    run_name: str,
    runs_by_name: dict[str, Run],
    cfg: Config,
    *,
    expected_geometry: tuple[int, int],
    expected_bins: np.ndarray,
    expected_rate_hz: float,
) -> tuple[PreparedRun, np.ndarray]:
    """*run_name*'s `"logmel"`-variant `PreparedRun` and its full-grid
    band-energy score (`band_energy_score`, reused unchanged from `rowii.
    anomaly.mad_baseline` -- never reimplemented), after asserting its
    log-mel geometry / target mel bins / mic rate match the values the
    already-committed `mad_baseline.json` thresholds were derived under
    (module docstring's provenance paragraph).

    Raises:
        ValueError: *run_name*'s geometry/bins/rate do not match the
            committed thresholds' own -- the score would not mean the same
            physical quantity, so applying those thresholds to it would be
            meaningless.
    """
    run = runs_by_name[run_name]
    prepared = prepare_run(run, _VARIANT, cfg, use_cache=True)
    n_frames, n_mels = logmel_geometry(prepared.feature_names)
    if (n_frames, n_mels) != expected_geometry:
        raise ValueError(
            f"{run_name}: logmel geometry {(n_frames, n_mels)} != committed "
            f"{expected_geometry} ({_BASELINE_JSON_PATH}) -- the score would not mean "
            f"the same physical quantity as the committed thresholds"
        )
    rate_hz = _mic_rate_hz(run)
    if rate_hz != expected_rate_hz:
        raise ValueError(
            f"{run_name}: mic rate {rate_hz} != committed {expected_rate_hz} "
            f"({_BASELINE_JSON_PATH})"
        )
    target_bins = high_band_mel_bins(n_mels, rate_hz)
    if not np.array_equal(target_bins, expected_bins):
        raise ValueError(
            f"{run_name}: target mel bins {target_bins.tolist()} != committed "
            f"{expected_bins.tolist()} ({_BASELINE_JSON_PATH})"
        )
    score = band_energy_score(prepared.features, n_frames, n_mels, target_bins)
    return prepared, score


def _df_to_markdown(df: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    """Minimal Markdown table renderer -- mirrors `scripts/eval_per_strike.
    py`'s own `_df_to_markdown` (independent copy: a script must not import
    from another script; `pandas.DataFrame.to_markdown` needs the optional
    `tabulate` package, not part of this repo's dependency set)."""
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


def _headline_table(detection: pd.DataFrame) -> pd.DataFrame:
    """Physical-strike detection at `_HEADLINE_TOLERANCE_S`, `kind_group`
    pooled ("ALL") -- one row per (session, k_label), the number this
    script's `main()` print summary and `summary.md` both key off."""
    headline = detection[
        (detection["granularity"] == "physical")
        & (detection["tolerance_s"] == _HEADLINE_TOLERANCE_S)
        & (detection["kind_group"] == "ALL")
    ]
    cols = ["session", "k_label", "threshold", "n_marks", "n_detected", "tpr"]
    return headline[cols].sort_values(["k_label", "session"]).reset_index(drop=True)


def _latency_summary_table(latency: pd.DataFrame) -> pd.DataFrame:
    summary = latency[latency["row_type"] == "summary"]
    return summary[
        ["session", "k_label", "n_total", "n_detected", "n_missed",
         "median_s", "iqr_low_s", "iqr_high_s"]
    ].sort_values(["k_label", "session"]).reset_index(drop=True)


def _summary_markdown(
    detection: pd.DataFrame,
    latency: pd.DataFrame,
    *,
    thresholds: dict[str, float],
    provenance: dict[str, Any],
    n_scored_by_session: dict[str, int],
) -> str:
    headline = _headline_table(detection)
    lat_summary = _latency_summary_table(latency)

    lines = [
        "# MAD baseline: per-strike detection + first-alarm latency",
        "",
        "The fixed-threshold MAD baseline (`scripts/run_mad_baseline.py`, "
        "`summary.md` in the parent directory for the full method + "
        "event-level results) evaluated against the seconds-level hammer-"
        "strike ground truth (`docs/groundtruth/080726_strikes_seconds_"
        "{st,pu}.csv`), through the SAME per-strike harness "
        "(`rowii.eval.per_strike`) `scripts/eval_per_strike.py` uses for "
        "the representation monitors -- see `results/pillar3-perstrike/` "
        "for their own numbers in the same shape. This baseline's alarm "
        "decision is a fixed threshold commissioned once on the B1 pool "
        "(never conformal, never recalibrated), so there is no `alpha` "
        "axis here -- `rowii.eval.per_strike`'s binary-alarm-stream "
        "adapter functions (`sweep_strike_detection_binary`, "
        "`evaluate_strike_latency_binary`) are used in place of the "
        "p_value/alpha functions the representation monitors go through.",
        "",
        "## Threshold provenance",
        "",
        "Reused verbatim from the already-committed "
        "`results/mad-baseline/mad_baseline.json` -- NOT recomputed here. "
        "This script only recomputes the per-window score for the two "
        "08.07 sessions (`rowii.anomaly.mad_baseline.band_energy_score`, "
        "the same function `scripts/run_mad_baseline.py` calls) and "
        "cross-checks the log-mel geometry / target mel bins / mic rate "
        "against the JSON before applying its thresholds.",
        "",
        f"- B1 commissioning pool: {', '.join(provenance['b1_runs'])} "
        f"({provenance['b1_n_windows']} pooled valid windows; "
        f"median={provenance['b1_median']:.4g}, "
        f"scaled MAD={provenance['b1_mad']:.4g}).",
        f"- `k1pct` = {provenance['k1pct']:.4g} (tuned once to flag "
        f"~{provenance['k1pct_target_rate']:.0%} of the B1 pool; realized "
        f"{provenance['k1pct_realized_rate_on_b1']:.4%}) -> "
        f"threshold={thresholds['k1pct']:.6g}.",
        f"- `k5` = 5.0 (loosest member of the conventional grid "
        f"{{5, 10, 20, 30, 40}}) -> threshold={thresholds['k5']:.6g}.",
        f"- Scored windows (every valid window -- the B1 threshold never "
        f"draws calibration from any 080726 window, so none are excluded): "
        f"pu={n_scored_by_session.get('pu', 0)}, "
        f"st={n_scored_by_session.get('st', 0)}.",
        "",
        f"## Detection: physical strikes within T={_HEADLINE_TOLERANCE_S:g}s",
        "",
        "`n_marks`/`n_detected` count DEDUPLICATED physical strikes "
        "(`granularity=\"physical\"` in `per_strike_detection.csv`), "
        "`kind_group=\"ALL\"` (pooled across plate-gen/plate-tur/landmark/"
        "vane-sweep). Full sweep (both granularities, "
        f"tolerances {_TOLERANCES_S}, all kind_group rows): "
        "`per_strike_detection.csv`.",
        "",
        _df_to_markdown(headline, floatfmt=".3f"),
        "",
        "## First-alarm latency (physical strikes, detected only)",
        "",
        f"Search horizon {_STRIKE_LATENCY_HORIZON_S:g} s, dedup gap "
        f"{_DEDUP_GAP_S:g} s -- the SAME constants `scripts/"
        "eval_per_strike.py` uses. `median_s`/`iqr_low_s`/`iqr_high_s` are "
        "computed over DETECTED strikes only (a missed strike has no "
        "latency value). Full detail (one row per physical strike): "
        "`latency.csv`.",
        "",
        _df_to_markdown(lat_summary, floatfmt=".3f"),
        "",
        "## Files",
        "",
        "- `alarms_pu.csv` / `alarms_st.csv` -- exported per-window MAD "
        "alarm stream (`t_utc_ns`, `score`, `alarm_k1pct`, `alarm_k5`).",
        "- `per_strike_detection.csv` -- full detection sweep.",
        "- `latency.csv` -- full latency table (summary + per-strike detail).",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not _BASELINE_JSON_PATH.is_file():
        print(
            f"eval_mad_per_strike: {_BASELINE_JSON_PATH} not found -- run "
            f"scripts/run_mad_baseline.py first (this script reuses its "
            f"committed thresholds, never recomputes them)",
            file=sys.stderr,
        )
        return 2
    provenance: dict[str, Any] = json.loads(_BASELINE_JSON_PATH.read_text())
    thresholds: dict[str, float] = {
        label: float(provenance["thresholds"][label]) for label in _K_LABELS
    }
    expected_geometry = (int(provenance["logmel_n_frames"]), int(provenance["logmel_n_mels"]))
    expected_bins = np.array(provenance["target_mel_bins"], dtype=np.int64)
    expected_rate_hz = float(provenance["mic_rate_hz"])

    cfg = load_config()
    index = discover(cfg.data_root)
    runs_by_name = {r.name: r for r in index.runs}

    unknown = [run_name for run_name, _ in _SESSIONS.values() if run_name not in runs_by_name]
    if unknown:
        print(f"eval_mad_per_strike: unknown run name(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    out_dir = _BASELINE_DIR / "perstrike"
    out_dir.mkdir(parents=True, exist_ok=True)

    detection_frames: list[pd.DataFrame] = []
    latency_rows: list[dict[str, Any]] = []
    n_scored_by_session: dict[str, int] = {}

    for session, (run_name, marks_path) in _SESSIONS.items():
        try:
            prepared, score = _prepare_and_score_checked(
                run_name, runs_by_name, cfg,
                expected_geometry=expected_geometry, expected_bins=expected_bins,
                expected_rate_hz=expected_rate_hz,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"eval_mad_per_strike: {exc}", file=sys.stderr)
            return 2

        valid = prepared.valid_mask
        t_ns = prepared.grid.edges_ns()[:-1][valid].astype(np.int64)
        score_valid = score[valid]
        n_scored_by_session[session] = int(valid.sum())

        alarms_df = pd.DataFrame({"t_utc_ns": t_ns, "score": score_valid})
        for k_label in _K_LABELS:
            alarms_df[f"alarm_{k_label}"] = score_valid > thresholds[k_label]
        alarms_df.to_csv(out_dir / f"alarms_{session}.csv", index=False)

        marks = pd.read_csv(marks_path, comment="#", dtype={"event_id": str})

        for k_label in _K_LABELS:
            alarm_column = f"alarm_{k_label}"
            common = {"session": session, "k_label": k_label, "threshold": thresholds[k_label]}

            sweep = sweep_strike_detection_binary(
                alarms_df, marks, tolerances_s=_TOLERANCES_S, gap_s=_DEDUP_GAP_S,
                alarm_column=alarm_column,
            )
            for key, value in common.items():
                sweep[key] = value
            detection_frames.append(sweep.reindex(columns=_DETECTION_COLUMNS))

            st_lat = evaluate_strike_latency_binary(
                alarms_df, marks, search_horizon_s=_STRIKE_LATENCY_HORIZON_S,
                gap_s=_DEDUP_GAP_S, alarm_column=alarm_column,
            )
            lat_summary = summarize_latency(
                st_lat["latency_s"].to_numpy(), st_lat["missed"].to_numpy()
            )
            latency_rows.append(
                {
                    "row_type": "summary", **common,
                    "search_horizon_s": _STRIKE_LATENCY_HORIZON_S, **asdict(lat_summary),
                }
            )
            for row in st_lat.itertuples(index=False):
                latency_rows.append(
                    {
                        "row_type": "detail", **common,
                        "search_horizon_s": _STRIKE_LATENCY_HORIZON_S,
                        "event_id": row.event_id, "kind": row.kind,
                        "kind_group": kind_group(str(row.kind)),
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

    detection.to_csv(out_dir / "per_strike_detection.csv", index=False)
    latency.to_csv(out_dir / "latency.csv", index=False)
    (out_dir / "summary.md").write_text(
        _summary_markdown(
            detection, latency, thresholds=thresholds, provenance=provenance,
            n_scored_by_session=n_scored_by_session,
        )
    )

    headline = _headline_table(detection)
    print(
        f"eval_mad_per_strike: {len(detection)} detection row(s), {len(latency)} latency "
        f"row(s) -> {out_dir}"
    )
    print(f"headline physical-strike detection (T={_HEADLINE_TOLERANCE_S:g}s):")
    for row in headline.itertuples(index=False):
        print(f"  {row.session}/{row.k_label}: {row.n_detected}/{row.n_marks} = {row.tpr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
