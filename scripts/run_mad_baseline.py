"""MAD baseline CLI: a hand-set-threshold median-plus-k-MAD band-energy
detector, run in OUR OWN harness on OUR OWN data as a comparison point for the
thesis' calibrated (conformal, mode-conditioned) system.

**Purpose.** Quantify what a field-standard fixed-threshold practice achieves
here, so the thesis can put it in the same table as the once-calibrated fusion
system (`results/step2/once-calibrated/`, `results/pillar3/`) -- NOT to
outperform it (a global, mode-blind, hand-set threshold is not expected to).
`rowii.anomaly.mad_baseline`'s own module docstring covers the PURE math
(commissioning statistics, the k <-> threshold <-> flag-rate triangle, the
log-mel -> high-band-score derivation) in full; this script is orchestration
only: discover runs, `prepare_run(..., "logmel", ...)`, commission a threshold
on the B1 pool, evaluate three experiments, write results.

**Attribution.** The method TYPE -- a global median + k * MAD band-energy
threshold over a high-frequency microphone band -- is inspired by the partner
project's own transient-detection practice ("Bruno's transient detector
(band-MAD)"; co-authored account: Zhang, Krummenacher et al., "Multi-Modal
Acoustic-Vibration Anomaly Detection in Pumped-Storage Turbines", Viennahydro
2026 draft). This module is an
INDEPENDENT implementation: every threshold, score, and reported number is
computed from THIS repo's own caches -- no partner code or number is read or
asserted anywhere in this script or in `rowii.anomaly.mad_baseline`.

**Commissioning (B1 pool).** `_B1_RUNS` is the SAME four-run era-B
commissioning pool `scripts/run_once_calibrated.py` uses as `_B1_FIT_RUNS`
("the once-calibrated system"). Unlike that system's conformal machinery, this
baseline pools EVERY valid window of the four runs directly (no fit/conformal
split -- deliberately simple, matching what a practitioner sets by hand from a
commissioning recording) and derives ONE mode-agnostic median/MAD pair; `k` is
then hand-set from a grid `{5, 10, 20, 30, 40}`, plus one additionally-derived
`k_1pct` (the k whose threshold flags exactly ~1% of the B1 pool --
"tuned once to a target", the variant a careful practitioner would use).
Strikes never enter the threshold fit by construction: the B1 pool is four
010726 runs, and the strike day (080726) is never one of them.

**Experiment 1 -- flag-rate drift + mode-dependence (`flag_rates.csv`).** For
every k (grid + `k_1pct`) and every one of 8 sessions (the 010726 pair
in-sample, the rest held-out across three DAQ-configuration eras -- see
`rowii-monitor/README.md`'s "Data layout" table), the realized flag rate is
reported both aggregated ("(all)") and broken down per SCADA-derived GT
operating mode (`rowii.scada.labels.gt_labels`) -- showing (a) drift of the
rate across eras (a mode-BLIND threshold has no way to track the era-6-29
microphone level step also visible in `scripts/analyze_days.py`'s `era-step`
view and independently reported by the partner, Rodrigues & Zhang 2026) and
(b) mode-dependence within a single day (a global threshold necessarily
over- or under-fires on whichever mode's level differs most from the B1
pool's own mode mix).

**Experiment 2 -- event detection on 080726 (`events_pu.csv`/`events_st.csv`).**
Per-event TPR and FAR outside events (`rowii.eval.events.evaluate_events`,
+/-5 s tolerance -- the SAME definition and tolerance
`scripts/eval_events.py`/the README's pillar-3 section use), one row per k
(grid + `k_1pct`), for the pumping-strikes and standstill-strikes sessions
separately. No calibration-exclusion logic is needed here (unlike
`scripts/monitor.py --exclude-calibration-events`): the B1 threshold never
sees any 080726 window at all, so every 080726 valid window is simply scored.

**Experiment 3 -- comparison row.** The once-calibrated fusion system's own
numbers are NEVER rerun here -- `_read_once_calibrated_comparison` only READS
its already-committed `results/step2/once-calibrated/fusion/{fusion_regimes.csv,
fusion.json}` and `results/pillar3/` artifacts, citing their paths (and a few
headline numbers) in `summary.md` so the thesis table can be assembled from
both sources side by side.

**Honesty caveat (restated in `summary.md`).** This is a WINDOW-level analogue:
the score is a 1-s-window aggregate (mean band POWER over the window, never a
peak/max statistic -- `rowii.anomaly.mad_baseline.band_energy_score`'s own
docstring), scored on the SAME 1-s monitoring grid the rest of this repo uses.
The partner's own detector runs on raw impulse resolution; this baseline is
the fair analogue INSIDE a 1-s-window monitoring harness, not a claim that
window-level and impulse-level detectors are directly comparable in general.

Outputs (`--out`, default `<results_root>/mad-baseline/`): `flag_rates.csv`
(experiment 1), `events_pu.csv` / `events_st.csv` (experiment 2), `summary.md`
(method, attribution, k grid, headline findings in plain language with exact
numbers, the comparison row, the window-level caveat), and `mad_baseline.json`
(every threshold/statistic above, machine-readable, for full traceability --
mirrors `scripts/run_once_calibrated.py`'s own sidecar convention).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.anomaly.mad_baseline import (  # noqa: E402
    band_energy_score,
    high_band_mel_bins,
    k_for_target_rate,
    logmel_geometry,
    median_mad,
    threshold_from_k,
)
from rowii.config import Config, load_config  # noqa: E402
from rowii.eval.events import EventEvalResult, evaluate_events  # noqa: E402
from rowii.io.dataset import (  # noqa: E402
    RecordingIndex,
    Run,
    betriebsdaten_utc_offset_ns,
    discover,
    run_utc_offset_ns,
)
from rowii.io.gantner import read_header  # noqa: E402
from rowii.pipeline import PreparedRun, prepare_run  # noqa: E402
from rowii.scada.labels import gt_labels, load_scada_window_means  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GROUNDTRUTH_DIR = _REPO_ROOT / "docs" / "groundtruth"

_VARIANT = "logmel"
_LOGMEL_PRIMARY_STREAM = "RAWGeneratorMic__0"
"""Mirrors `rowii.anomaly.mad_baseline._LOGMEL_PRIMARY_STREAM` (itself a
mirror of `rowii.pipeline`'s own private `_LOGMEL_STREAMS`) -- duplicated
here, script-sibling/module-sibling precedent throughout this repo (e.g.
`rowii.anomaly.sentinels._MIC_STREAMS`), since none of those are exported."""

_B1_RUNS: tuple[str, ...] = (
    "010726-pu", "010726-tu1-morning", "010726-tu2", "010726-tu_ph_tu",
)
"""The commissioning pool -- textually IDENTICAL to
`scripts/run_once_calibrated.py`'s `_B1_FIT_RUNS` (era B, "the once-calibrated
system"'s own commissioning pool; module docstring)."""

_EVAL_RUNS: tuple[str, ...] = (
    "250526-tu", "250526-pu-morning", "290626-tu", "290626-pu",
    "010726-tu_ph_tu", "010726-pu", "080726-pu_strikes", "080726-st_strikes",
)
"""Experiment 1's 8 sessions -- the SAME held-out run set
`scripts/run_once_calibrated.py`'s `_REPLAY` reports on (minus its
Betriebsdaten-free `270626` sentinel-only row, which this baseline has no
sentinel to evaluate), plus `080726-st_strikes` explicitly (that driver only
uses it for its own pillar-3 retention check, never a `_REPLAY` row)."""

_IN_SAMPLE_RUNS = frozenset({"010726-tu_ph_tu", "010726-pu"})
"""These two are ALSO `_B1_RUNS` members: their flag-rate rows are in-sample
by construction (the SAME two runs `run_once_calibrated.py` tags
`"in-sample"`), never held-out evidence."""

_ERA_BY_DAY: dict[str, str] = {
    "250526": "A", "270626": "A", "290626": "B", "010726": "B", "080726": "C",
}
"""DAQ-configuration era per day-root (README "Data layout" table /
`scripts/run_once_calibrated.py`'s `_ReplayEntry.era` tags)."""

_K_GRID: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0, 40.0)
_K1PCT_TARGET_RATE = 0.01
_K1PCT_LABEL = "k1pct"
_MODE_ALL = "(all)"

_EVENT_TOLERANCE_S = 5.0
"""+/-5 s membership tolerance -- the README pillar-3 section's own
convention, `scripts/run_once_calibrated.py`'s `_PILLAR3_TOLERANCE_S`."""

_EVENTS_BY_RUN: dict[str, Path] = {
    "080726-pu_strikes": _GROUNDTRUTH_DIR / "080726_events_pu.csv",
    "080726-st_strikes": _GROUNDTRUTH_DIR / "080726_events_st.csv",
}

_ONCE_CALIBRATED_FUSION_DIR = _REPO_ROOT / "results" / "step2" / "once-calibrated" / "fusion"
_PILLAR3_DIR = _REPO_ROOT / "results" / "pillar3"


# ---------------------------------------------------------------------------
# Duplicated script-sibling helpers (a script never imports another script's
# internals -- module docstrings across this repo's scripts/ all state this
# rule). `_betriebsdaten_for_grid` is verbatim from `scripts/
# run_once_calibrated.py` / `scripts/run_step1.py`; `_run_gt_state` is adapted
# from `run_once_calibrated.py`'s `_run_gt_states` (same Betriebsdaten-match +
# load + label recipe, but returns the state Series alone -- this baseline
# has no bank/fit-vs-conformal split to align it against).
# ---------------------------------------------------------------------------


def _betriebsdaten_for_grid(betriebsdaten: list[Path], grid: WindowGrid) -> list[Path]:
    """Betriebsdaten files whose hourly span intersects *grid*'s UTC time range."""
    grid_end_ns = int(grid.edges_ns()[-1])
    offset_ns = betriebsdaten_utc_offset_ns(betriebsdaten)
    matched = []
    for path in betriebsdaten:
        header = read_header(path)
        file_start_ns = header.t0_ns + offset_ns
        file_end_ns = file_start_ns + round(header.n_frames / header.sample_rate_hz * 1e9)
        if file_start_ns < grid_end_ns and file_end_ns > grid.t0_ns:
            matched.append(path)
    return sorted(matched)


def _run_gt_state(
    run: Run, prepared: PreparedRun, index: RecordingIndex, cfg: Config
) -> np.ndarray:
    """`(W,)` object array of GT state strings (`rowii.scada.labels.STATES`,
    plus `"unknown"`) over *run*'s FULL grid -- the per-mode breakdown key for
    experiment 1. Every one of the 8 `_EVAL_RUNS` has Betriebsdaten coverage
    (verified against the discovered index before this is ever called).

    Raises:
        ValueError: *run*'s day has no Betriebsdaten coverage overlapping its
            own grid.
    """
    day_betriebsdaten = index.betriebsdaten_by_day.get(run.day_root, [])
    matched = (
        _betriebsdaten_for_grid(day_betriebsdaten, prepared.grid) if day_betriebsdaten else []
    )
    if not matched:
        raise ValueError(
            f"run {run.name!r} has no Betriebsdaten coverage overlapping its "
            f"own grid -- experiment 1's per-mode breakdown needs GT on every "
            f"_EVAL_RUNS member"
        )
    scada = load_scada_window_means(
        matched, prepared.grid, audio_run_offset_ns=run_utc_offset_ns(run)
    )
    labels: np.ndarray = gt_labels(scada, cfg.gt, window_s=cfg.window.window_s)[
        "state"
    ].to_numpy()
    return labels


# ---------------------------------------------------------------------------
# Score derivation
# ---------------------------------------------------------------------------


def _mic_rate_hz(run: Run) -> float:
    """*run*'s primary generator-mic sample rate (cheap header-only read,
    never assumed -- even though it is, in practice, a fixed 50 kHz hardware
    property across this whole campaign, verified per-run here rather than
    hardcoded) -- `high_band_mel_bins` needs it to reproduce the exact mel
    filterbank geometry the `"logmel"` cache was built with."""
    files = sorted(run.files[_LOGMEL_PRIMARY_STREAM], key=lambda f: f.start_utc_hint)
    return read_header(files[0].path).sample_rate_hz


def _prepare_and_score(
    run_name: str, runs_by_name: dict[str, Run], cfg: Config, *, use_cache: bool
) -> tuple[PreparedRun, np.ndarray, tuple[int, int], np.ndarray]:
    """One run's `"logmel"`-variant `PreparedRun`, its full-grid band-energy
    score, and the `(n_frames, n_mels)` geometry + target mel-bin indices the
    score was computed with (returned so the caller can assert every run
    shares an IDENTICAL geometry/band before pooling scores across them --
    `rowii.anomaly.mad_baseline` module docstring: the score must mean the
    same physical quantity on every day for a single global threshold to be
    meaningful at all)."""
    run = runs_by_name[run_name]
    prepared = prepare_run(run, _VARIANT, cfg, use_cache=use_cache)
    n_frames, n_mels = logmel_geometry(prepared.feature_names)
    rate_hz = _mic_rate_hz(run)
    target_bins = high_band_mel_bins(n_mels, rate_hz)
    score = band_energy_score(prepared.features, n_frames, n_mels, target_bins)
    return prepared, score, (n_frames, n_mels), target_bins


# ---------------------------------------------------------------------------
# Experiment 1 -- flag rates per day and per mode
# ---------------------------------------------------------------------------


def _flag_rate_rows(
    run_name: str,
    prepared: PreparedRun,
    score: np.ndarray,
    states: np.ndarray,
    thresholds: dict[str, float],
    k_values: dict[str, float],
) -> list[dict[str, object]]:
    """One `flag_rates.csv` row per `(k_label, mode)` for *run_name* -- an
    `"(all)"` aggregate row plus one row per distinct GT state actually
    present among its valid windows (module docstring, experiment 1)."""
    valid = prepared.valid_mask
    scores_valid = score[valid]
    states_valid = states[valid]
    day = run_name.split("-", 1)[0]
    era = _ERA_BY_DAY.get(day, "?")
    in_sample = run_name in _IN_SAMPLE_RUNS

    rows: list[dict[str, object]] = []
    for k_label, threshold in thresholds.items():
        flagged = scores_valid > threshold
        rows.append({
            "run": run_name, "day": day, "era": era, "in_sample": in_sample,
            "k_label": k_label, "k": k_values[k_label], "threshold": threshold,
            "mode": _MODE_ALL, "n_windows": int(scores_valid.shape[0]),
            "n_flagged": int(flagged.sum()), "flag_rate": float(flagged.mean()),
        })
        for state in sorted(str(s) for s in set(states_valid.tolist())):
            mask = states_valid == state
            n = int(mask.sum())
            n_flag = int((flagged & mask).sum())
            rows.append({
                "run": run_name, "day": day, "era": era, "in_sample": in_sample,
                "k_label": k_label, "k": k_values[k_label], "threshold": threshold,
                "mode": state, "n_windows": n, "n_flagged": n_flag,
                "flag_rate": (n_flag / n) if n else float("nan"),
            })
    return rows


# ---------------------------------------------------------------------------
# Experiment 2 -- event detection on 080726
# ---------------------------------------------------------------------------


def _alarms_frame(prepared: PreparedRun, score: np.ndarray, threshold: float) -> pd.DataFrame:
    """One `t_utc_ns`/`alarm` row per VALID window of *prepared* -- the
    `rowii.eval.events.evaluate_events` alarms contract. No `role` column: the
    B1 threshold never draws calibration from any 080726 window (module
    docstring), so every valid window here is simply scored, none consumed."""
    valid = prepared.valid_mask
    t_ns = prepared.grid.edges_ns()[:-1][valid].astype(np.int64)
    alarm = score[valid] > threshold
    return pd.DataFrame({"t_utc_ns": t_ns, "alarm": alarm})


def _event_rows(
    run_name: str,
    prepared: PreparedRun,
    score: np.ndarray,
    events_df: pd.DataFrame,
    thresholds: dict[str, float],
    k_values: dict[str, float],
    *,
    window_s: float,
) -> list[dict[str, object]]:
    """One `events_{pu,st}.csv` row per k (grid + `k_1pct`) for *run_name*."""
    n_scored = int(prepared.valid_mask.sum())
    rows: list[dict[str, object]] = []
    for k_label, threshold in thresholds.items():
        alarms = _alarms_frame(prepared, score, threshold)
        result: EventEvalResult = evaluate_events(
            alarms, events_df, window_s=window_s, tolerance_s=_EVENT_TOLERANCE_S
        )
        rows.append({
            "run": run_name, "k_label": k_label, "k": k_values[k_label],
            "threshold": threshold, "n_scored_windows": n_scored,
            "n_events": result.n_events, "n_detected": result.n_detected,
            "event_tpr": result.event_tpr,
            "false_alarm_windows": result.false_alarm_windows,
            "realized_window_far": result.realized_window_far,
            "false_alarm_rate_per_hour": result.false_alarm_rate_per_hour,
            "tolerance_s": result.tolerance_s,
        })
    return rows


# ---------------------------------------------------------------------------
# Experiment 3 -- comparison row (READ existing artifacts, never rerun them)
# ---------------------------------------------------------------------------


def _read_once_calibrated_comparison() -> dict[str, object]:
    """A read-only digest of the once-calibrated fusion system's already-
    committed numbers (`results/step2/once-calibrated/fusion/`,
    `results/pillar3/`) -- module docstring, experiment 3: NEVER rerun,
    always cited by path."""
    regimes_path = _ONCE_CALIBRATED_FUSION_DIR / "fusion_regimes.csv"
    json_path = _ONCE_CALIBRATED_FUSION_DIR / "fusion.json"
    comparison: dict[str, object] = {
        "regimes_csv_path": str(regimes_path.relative_to(_REPO_ROOT)),
        "sidecar_json_path": str(json_path.relative_to(_REPO_ROOT)),
        "pillar3_dir_path": str(_PILLAR3_DIR.relative_to(_REPO_ROOT)),
    }
    if regimes_path.is_file():
        comparison["regimes"] = pd.read_csv(regimes_path).to_dict(orient="records")
    else:
        logger.warning(
            "run_mad_baseline: %s not found -- comparison row will cite the path "
            "without numbers", regimes_path,
        )
    if json_path.is_file():
        sidecar = json.loads(json_path.read_text())
        comparison["pillar3"] = sidecar.get("pillar3")
    else:
        logger.warning(
            "run_mad_baseline: %s not found -- pillar-3 comparison numbers omitted",
            json_path,
        )
    return comparison


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------


def _fmt(value: float, digits: int = 4) -> str:
    return "n/a" if (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}g}"


def _write_summary(
    out_path: Path,
    *,
    b1_n_windows: int,
    median: float,
    mad: float,
    k1pct: float,
    k1pct_realized: float,
    thresholds: dict[str, float],
    flag_rates_df: pd.DataFrame,
    events_pu_df: pd.DataFrame,
    events_st_df: pd.DataFrame,
    comparison: dict[str, object],
    n_frames: int,
    n_mels: int,
    target_bins: np.ndarray,
    rate_hz: float,
) -> None:
    lines: list[str] = [
        "# MAD baseline: hand-set-threshold band-energy detector",
        "",
        "## Method",
        "",
        "A median-plus-k-MAD band-energy detector: per 1-s window, the score "
        f"is `log10(mean-over-frames(sum-of-target-mel-bins-power) + 1e-12)`, "
        f"i.e. the log high-band microphone energy in **[5000, 20000) Hz** "
        f"({target_bins.size} of {n_mels} mel bins at this plant's real "
        f"{rate_hz:g} Hz mic rate / {n_frames}-frame log-mel geometry, "
        "derived from the cached `logmel` variant, primary generator-mic "
        "stream only -- `rowii.anomaly.mad_baseline` module docstring has "
        "the full derivation and the reasoning for reading the log-mel cache "
        "rather than a single named handcrafted feature, none of which "
        "matches a 5-20 kHz band cleanly). The threshold is `median + k * "
        "1.4826 * MAD` of that score, computed ONCE from the B1 commissioning "
        f"pool ({', '.join(_B1_RUNS)}; {b1_n_windows} pooled valid windows; "
        f"median={median:.4g}, scaled MAD={mad:.4g}), mode-agnostic (one "
        "global threshold -- that is the point of this baseline, in contrast "
        "to the mode-conditioned calibrated system).",
        "",
        "**Attribution.** The method TYPE is inspired by the partner "
        "project's own transient-detection practice (\"Bruno's transient "
        "detector (band-MAD)\"; co-authored account: Zhang, Krummenacher "
        "et al., \"Multi-Modal Acoustic-Vibration Anomaly Detection in "
        "Pumped-Storage Turbines\", Viennahydro 2026 draft). This is an "
        "INDEPENDENT implementation -- every threshold, score, and number "
        "below is computed from this repo's own caches; no partner code or "
        "number is read or asserted anywhere in it.",
        "",
        f"**k grid:** {{{', '.join(str(int(k)) for k in _K_GRID)}}}. "
        f"**k_1pct** (the k tuned once to flag exactly ~1% of the B1 pool): "
        f"**{k1pct:.4g}** (realized on B1: {k1pct_realized:.4%}).",
        "",
        "**Window-level caveat.** This is a WINDOW-level analogue: the score "
        "is a 1-s aggregate (mean band power over the window, never a "
        "peak/max statistic). The partner's own detector runs on raw impulse "
        "resolution; this baseline is the fair analogue inside a 1-s-window "
        "monitoring harness, not a claim that window- and impulse-level "
        "detectors are directly comparable in general.",
        "",
        "## Experiment 1 -- flag-rate drift and mode-dependence",
        "",
        "Full table: `flag_rates.csv` (one row per run x k x mode, plus an "
        "`\"(all)\"` aggregate row per run x k). Aggregate (`\"(all)\"`) "
        "flag rate per run, by k:",
        "",
    ]

    agg = flag_rates_df[flag_rates_df["mode"] == _MODE_ALL]
    k_labels = list(thresholds.keys())
    lines.append("| run | day | era | in-sample | " + " | ".join(k_labels) + " |")
    lines.append("|:--|:--:|:--:|:--:|" + "---:|" * len(k_labels))
    for run_name in _EVAL_RUNS:
        row = agg[agg["run"] == run_name].set_index("k_label")
        day = str(row["day"].iloc[0])
        era = str(row["era"].iloc[0])
        in_sample = "yes" if bool(row["in_sample"].iloc[0]) else ""
        cells = " | ".join(f"{row.loc[k, 'flag_rate']:.2%}" for k in k_labels)
        lines.append(f"| {run_name} | {day} | {era} | {in_sample} | {cells} |")
    lines += ["", "**Mode-dependence** (k=k1pct, per-mode flag rate; a "
              "mode-blind threshold has no reason to hold ~1% on every mode "
              "at once):", ""]
    per_mode_k1 = flag_rates_df[
        (flag_rates_df["k_label"] == _K1PCT_LABEL) & (flag_rates_df["mode"] != _MODE_ALL)
    ]
    lines.append("| run | mode | n_windows | flag_rate |")
    lines.append("|:--|:--|--:|--:|")
    for _, r in per_mode_k1.sort_values(["run", "mode"]).iterrows():
        lines.append(f"| {r['run']} | {r['mode']} | {int(r['n_windows'])} | {r['flag_rate']:.2%} |")

    lines += ["", "## Experiment 2 -- event detection on 080726", "",
              "Per-k event-level TPR and window-FAR outside events "
              f"(+/-{_EVENT_TOLERANCE_S:g} s tolerance, "
              "`rowii.eval.events.evaluate_events`, the SAME definition "
              "`scripts/eval_events.py` uses). Full tables: `events_pu.csv` "
              "(pumping strikes), `events_st.csv` (standstill strikes).", ""]
    for name, df in (("PU (pumping)", events_pu_df), ("ST (standstill)", events_st_df)):
        lines += [f"**{name}:**", "",
                  "| k | event_tpr | n_detected/n_events | realized_window_far "
                  "| false_alarm_rate_per_hour |",
                  "|:--|--:|:--:|--:|--:|"]
        for _, r in df.iterrows():
            lines.append(
                f"| {r['k_label']} | {_fmt(r['event_tpr'], 3)} "
                f"| {int(r['n_detected'])}/{int(r['n_events'])} "
                f"| {_fmt(r['realized_window_far'], 3)} "
                f"| {_fmt(r['false_alarm_rate_per_hour'], 4)} |"
            )
        lines.append("")

    lines += ["## Experiment 3 -- comparison with the calibrated system", "",
              "The once-calibrated fusion system's numbers are NOT rerun "
              "here -- cited from its own already-committed artifacts:", "",
              f"- Regimes table: `{comparison.get('regimes_csv_path')}`",
              f"- Sidecar (incl. pillar-3 TPR): `{comparison.get('sidecar_json_path')}`",
              f"- Pillar-3 per-alpha artifacts: `{comparison.get('pillar3_dir_path')}`", ""]
    regimes = comparison.get("regimes")
    if isinstance(regimes, list) and regimes:
        lines += ["Once-calibrated fusion `once_triggered_far` per run (for the "
                   "same sessions this baseline evaluates):", "",
                   "| run | once_triggered_far | decision |", "|:--|--:|:--:|"]
        wanted = set(_EVAL_RUNS)
        for row in regimes:
            if row.get("run") in wanted:
                lines.append(
                    f"| {row['run']} | {_fmt(float(row['once_triggered_far']), 3)} "
                    f"| {row.get('decision')} |"
                )
        lines.append("")
    pillar3 = comparison.get("pillar3")
    if isinstance(pillar3, dict):
        lines += ["Once-calibrated fusion pillar-3 event TPR (once+triggered "
                   "regime):", "",
                   f"- PU: {_fmt(float(pillar3['pu']['once_triggered_event_tpr']), 3)}",
                   f"- ST: {_fmt(float(pillar3['st']['once_triggered_event_tpr']), 3)}", ""]

    lines += [
        "## Files",
        "",
        "- `flag_rates.csv` -- experiment 1, full table.",
        "- `events_pu.csv` / `events_st.csv` -- experiment 2, full tables.",
        "- `mad_baseline.json` -- every threshold/statistic above, machine-readable.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# argparse + main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hand-set-threshold MAD baseline: commission a mode-agnostic "
            "median+k*MAD high-band (5-20 kHz) microphone-energy threshold "
            "on the B1 pool, then evaluate flag-rate drift/mode-dependence "
            "(experiment 1) and event detection on 080726 (experiment 2), "
            "citing the once-calibrated system's own numbers for comparison "
            "(experiment 3). Module docstring has the full method + "
            "attribution."
        )
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output directory (default: <results_root>/mad-baseline/).",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable rowii.pipeline.prepare_run's on-disk feature cache.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config()
    index = discover(cfg.data_root)
    runs_by_name = {r.name: r for r in index.runs}

    all_run_names = list(dict.fromkeys([*_B1_RUNS, *_EVAL_RUNS]))
    unknown = [n for n in all_run_names if n not in runs_by_name]
    if unknown:
        available = ", ".join(sorted(runs_by_name)) or "(none discovered)"
        print(
            f"run_mad_baseline: unknown run name(s): {', '.join(unknown)}; "
            f"available runs: {available}",
            file=sys.stderr,
        )
        return 2

    prepared_by_run: dict[str, PreparedRun] = {}
    score_by_run: dict[str, np.ndarray] = {}
    geometry_by_run: dict[str, tuple[int, int]] = {}
    bins_by_run: dict[str, np.ndarray] = {}
    for name in all_run_names:
        logger.info("run_mad_baseline: preparing %s (%s)", name, _VARIANT)
        try:
            prepared, score, geometry, bins = _prepare_and_score(
                name, runs_by_name, cfg, use_cache=not args.no_cache
            )
        except (RuntimeError, ValueError) as exc:
            print(f"run_mad_baseline: failed to prepare/score {name!r}: {exc}", file=sys.stderr)
            return 2
        prepared_by_run[name] = prepared
        score_by_run[name] = score
        geometry_by_run[name] = geometry
        bins_by_run[name] = bins

    reference = all_run_names[0]
    ref_geometry, ref_bins = geometry_by_run[reference], bins_by_run[reference]
    for name in all_run_names[1:]:
        if geometry_by_run[name] != ref_geometry or not np.array_equal(bins_by_run[name], ref_bins):
            print(
                f"run_mad_baseline: log-mel geometry/target-band mismatch between "
                f"{reference!r} ({ref_geometry}, bins={ref_bins.tolist()}) and "
                f"{name!r} ({geometry_by_run[name]}, bins={bins_by_run[name].tolist()}) "
                f"-- the score would not mean the same physical quantity on both days",
                file=sys.stderr,
            )
            return 2
    n_frames, n_mels = ref_geometry
    rate_hz = _mic_rate_hz(runs_by_name[reference])

    # --- Commission the threshold on the B1 pool (module docstring) --------
    b1_scores = np.concatenate(
        [score_by_run[name][prepared_by_run[name].valid_mask] for name in _B1_RUNS]
    )
    median, mad = median_mad(b1_scores)
    thresholds: dict[str, float] = {
        f"k{int(k)}": threshold_from_k(median, mad, k) for k in _K_GRID
    }
    k_values: dict[str, float] = {f"k{int(k)}": k for k in _K_GRID}
    k1pct, k1pct_realized = k_for_target_rate(b1_scores, median, mad, _K1PCT_TARGET_RATE)
    thresholds[_K1PCT_LABEL] = threshold_from_k(median, mad, k1pct)
    k_values[_K1PCT_LABEL] = k1pct

    # --- GT state labels for experiment 1's 8 sessions ----------------------
    gt_state_by_run: dict[str, np.ndarray] = {}
    for name in _EVAL_RUNS:
        try:
            gt_state_by_run[name] = _run_gt_state(
                runs_by_name[name], prepared_by_run[name], index, cfg
            )
        except ValueError as exc:
            print(f"run_mad_baseline: {exc}", file=sys.stderr)
            return 2

    # --- Experiment 1: flag rates -------------------------------------------
    flag_rows: list[dict[str, object]] = []
    for name in _EVAL_RUNS:
        flag_rows.extend(
            _flag_rate_rows(
                name, prepared_by_run[name], score_by_run[name], gt_state_by_run[name],
                thresholds, k_values,
            )
        )
    flag_rates_df = pd.DataFrame(flag_rows)

    # --- Experiment 2: event detection on 080726 ----------------------------
    events_df_by_session: dict[str, pd.DataFrame] = {}
    for run_name, events_path in _EVENTS_BY_RUN.items():
        events_gt = pd.read_csv(events_path, comment="#")
        rows = _event_rows(
            run_name, prepared_by_run[run_name], score_by_run[run_name], events_gt,
            thresholds, k_values, window_s=cfg.window.window_s,
        )
        events_df_by_session[run_name] = pd.DataFrame(rows)

    # --- Experiment 3: comparison (read-only) -------------------------------
    comparison = _read_once_calibrated_comparison()

    # --- Write outputs -------------------------------------------------------
    out_dir = Path(args.out) if args.out is not None else cfg.results_root / "mad-baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    flag_rates_df.to_csv(out_dir / "flag_rates.csv", index=False)
    events_df_by_session["080726-pu_strikes"].to_csv(out_dir / "events_pu.csv", index=False)
    events_df_by_session["080726-st_strikes"].to_csv(out_dir / "events_st.csv", index=False)

    sidecar = {
        "b1_runs": list(_B1_RUNS),
        "eval_runs": list(_EVAL_RUNS),
        "in_sample_runs": sorted(_IN_SAMPLE_RUNS),
        "k_grid": list(_K_GRID),
        "k1pct_target_rate": _K1PCT_TARGET_RATE,
        "k1pct": k1pct,
        "k1pct_realized_rate_on_b1": k1pct_realized,
        "b1_n_windows": int(b1_scores.shape[0]),
        "b1_median": median,
        "b1_mad": mad,
        "thresholds": thresholds,
        "band_lo_hz": 5000.0,
        "band_hi_hz": 20000.0,
        "logmel_n_frames": n_frames,
        "logmel_n_mels": n_mels,
        "mic_rate_hz": rate_hz,
        "target_mel_bins": ref_bins.tolist(),
        "event_tolerance_s": _EVENT_TOLERANCE_S,
        "once_calibrated_comparison": comparison,
        "provenance_note": (
            "Hand-set-threshold MAD baseline (scripts/run_mad_baseline.py, "
            "rowii.anomaly.mad_baseline). Method TYPE inspired by the "
            "partner project's own transient-detection practice ('Bruno's "
            "transient detector (band-MAD)'); every "
            "threshold, score, and number here is computed from this repo's "
            "own caches -- no partner code or number is read or asserted "
            "anywhere. Window-level analogue (1-s aggregate band power, "
            "never a peak/max statistic) -- the partner's own detector runs "
            "on raw impulse resolution."
        ),
    }
    (out_dir / "mad_baseline.json").write_text(json.dumps(sidecar, indent=2) + "\n")

    _write_summary(
        out_dir / "summary.md",
        b1_n_windows=int(b1_scores.shape[0]), median=median, mad=mad,
        k1pct=k1pct, k1pct_realized=k1pct_realized, thresholds=thresholds,
        flag_rates_df=flag_rates_df,
        events_pu_df=events_df_by_session["080726-pu_strikes"],
        events_st_df=events_df_by_session["080726-st_strikes"],
        comparison=comparison, n_frames=n_frames, n_mels=n_mels,
        target_bins=ref_bins, rate_hz=rate_hz,
    )

    print(f"run_mad_baseline: done -- k1pct={k1pct:.4g} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
