"""Operating-mode LABEL comparison: our SCADA GT rules vs. the partner's (Bruno's).

Self-serve cross-check (no partner export available): re-implements the partner's
own DOCUMENTED SCADA-rule methodology from his shared read-only reference repo
(`repos/hydropower-anomaly`, working tree 101 commits stale -- content read via
`git show cf92b0f:<path>`, never checked out or modified) and applies it, verbatim,
to OUR Betriebsdaten. Firewall: no partner-computed RESULT, label, or score is
imported at any point -- only his documented RULE (thresholds + control flow) is
re-implemented, as an attributed external-methodology comparison. Our own
`rowii.scada.labels.gt_labels` stays primary; this script never feeds the
partner's labels back into our pipeline, only writes them to `results/
crosscheck-labels/` for side-by-side inspection.

Investigation summary (2026-08-12; see the accompanying research note,
`research/notes/analysis_2026-08-12_partner_label_agreement.md`, for the full
write-up): the partner repo contains explicit SCADA THRESHOLD RULES (not a
classifier) for the ground-truth label itself. Two variants exist, from
different points in the repo's history, applied to different campaigns:

Variant B (`partner_mode_b*`, PRIMARY comparison below) --
`src/hydropower_anomaly/ingestion/scada.py`, `ScadaTimeline._classify`
(module created 2026-07-03; git show cf92b0f:src/hydropower_anomaly/
ingestion/scada.py, lines ~136-190). Duplicated inline by
`tools/run_010726_extract.py::load_scada_day` (git show cf92b0f:tools/
run_010726_extract.py, lines ~90-98) to label `features_*.csv`'s "state"
column, which `tools/run_010726_analysis.py` (`load_july`) consumes directly;
`tools/run_290626_day_analysis.py` imports `ScadaTimeline` itself (`from
hydropower_anomaly.ingestion.scada import ScadaTimeline`, then `tl.
transitions()`). This is the rule that actually produced the state labels
behind the two day-analysis scripts named in the task, and it is what drove
the 2026-06-29 and 2026-07-01 campaigns -- 4 of our 5 shared measurement days.
Rule (rpm unsigned magnitude; power_mw as read from "1_P_Ist", no sign
correction applied anywhere in the partner's own loader):
    |rpm| < 50 rpm          -> "ST"  (standstill)
    power_mw > +5 MW        -> "TU"  (turbine)
    power_mw < -5 MW        -> "PU"  (pump)
    otherwise               -> "PH"  (phase-shifter candidate)
then any contiguous "PH" run shorter than `ph_sustained_s` (default 600 s) is
reclassified "TRANS" (transition) -- a spinning-unloaded ramp segment, not
genuine phase-shifter operation.

Variant A (`partner_mode_a*`, SECONDARY/historical, 250526 only below) --
`src/hydropower_anomaly/state/scada_estimator.py`, `estimate_operating_point`
+ `detect_transition_mask` (module created 2026-06-15, unmodified since --
`git log cf92b0f -- src/hydropower_anomaly/state/scada_estimator.py` shows a
single commit; git show cf92b0f:src/hydropower_anomaly/state/
scada_estimator.py, `ModeThresholds` defaults lines 36-56, `estimate_
operating_point` lines 59-104, `detect_transition_mask`/`estimate_
operating_track` lines 115-213). Cited as "production `state.scada_estimator.
estimate_operating_point`" in `research/regime-discovery-results-v1.md`
(dated 2026-06-25) and used by `tools/{run_ablation,run_all_cluster,
run_experiments,run_tierB}.py` -- all four load SCADA from the 2026-06-25
Betriebsdaten only (e.g. `run_tierB.py`'s `load_scada` reads
`2026-06-25_{h}-00-00.dat`). Rule (rpm signed: + turbine dir., - pump dir.;
power_mw signed to match, per the docstring):
    |rpm| < 10 rpm                          -> "ST"
    power_mw < -20 MW                       -> "PU"
    rpm > 0 and vane_pct > 55%              -> "TU_full"
    rpm > 0 and 5% < vane_pct <= 55%        -> "TU_partial"
    rpm > 0 (else)                          -> "PH"
    otherwise                               -> "transition"
then any window whose |d(rpm)/dt| > 50 rpm/s OR |d(vane)/dt| > 5 %pt/s
(`np.gradient`, matching the partner's own implementation) is forced to
"transition" regardless of the static classification.

Both variants are re-implemented here as pure, deterministic functions of SCADA
window means (never fitted to our data, never a classifier) and applied with
the partner's OWN thresholds, unchanged.

ADAPTATIONS (both variants, required to run a rule written for a continuous
per-second SCADA timeline against our windowed grid; not a change to either
rule's logic):
  1. NaN guard: the partner's own timeline has no gaps (`ScadaTimeline`/
     `estimate_operating_point` never see a missing sample). Our `WindowGrid`
     can include windows with no SCADA sample at all (grid edges, coverage
     gaps) -- `load_scada_window_means` returns NaN there. Both functions
     below label a NaN window "unknown", mirroring `rowii.scada.labels.
     _base_state`'s own `known` guard.
  2. Contiguous-run duration: `ScadaTimeline._classify`'s own PH-run-length
     test uses the run's actual elapsed wall time; reproduced here as
     `n_windows_in_run * window_s`, algebraically identical for a uniform
     `WindowGrid` (`rowii.signals.windows.WindowGrid` guarantees uniform
     window spacing).

Usage:
    cd repos/rowii-monitor && .venv/bin/python scripts/compare_partner_labels.py
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPTS_DIR.parent / "src"
for _extra_path in (str(_SCRIPTS_DIR), str(_SRC_DIR)):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

import run_step1  # noqa: E402 -- reuses build_run_grid/_betriebsdaten_for_grid (repo convention, see scripts/analyze_step1.py)

from rowii.config import Config, load_config  # noqa: E402
from rowii.io.dataset import RecordingIndex, Run, discover, run_utc_offset_ns  # noqa: E402
from rowii.scada.labels import STATES, gt_labels, load_scada_window_means  # noqa: E402

logger = logging.getLogger(__name__)

#: Our own GT vocabulary (`rowii.scada.labels.STATES`) plus "unknown" -- both
#: partner variants are mapped onto this exact set, so a plain `==` comparison
#: against `our_state` is always meaningful.
STATE_ORDER: tuple[str, ...] = (*STATES, "unknown")

#: The task's shared-day run list: every (day, session) pair with independently
#: confirmed SCADA (Betriebsdaten) coverage on our side (README's coverage
#: table + `rowii.io.dataset.discover` betriebsdaten-file counts). 270626 is
#: excluded (never exported); 250526-pu-afternoon is excluded (export ends
#: 10:00 UTC, before that session starts) -- "250526 only where SCADA coverage
#: exists" (by design).
PRIMARY_RUNS: tuple[str, ...] = (
    "250526-tu",
    "250526-pu-morning",
    "290626-tu",
    "290626-pu",
    "010726-tu_ph_tu",
    "010726-pu",
    "080726-st_strikes",
    "080726-pu_strikes",
)

#: Variant A is only run where it is the partner's own historically-documented
#: rule (the 2026-06-25 burst) -- see module docstring.
VARIANT_A_RUNS: tuple[str, ...] = ("250526-tu", "250526-pu-morning")


# ---------------------------------------------------------------------------
# Variant B -- ingestion/scada.py::ScadaTimeline._classify
# ---------------------------------------------------------------------------

_B_TO_OURS: dict[str, str] = {
    "ST": "standstill", "TU": "turbine", "PU": "pump",
    "PH": "phase-shifter", "TRANS": "transition",
}


def _remap(labels: np.ndarray, mapping: dict[str, str]) -> np.ndarray:
    """Vectorized string remap; values absent from *mapping* (e.g. "unknown")
    pass through unchanged."""
    out = labels.astype("<U16").copy()
    for src, dst in mapping.items():
        out[labels == src] = dst
    return out


def _contiguous_runs(labels: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of every maximal contiguous run of equal *labels*."""
    n = len(labels)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def partner_mode_b_raw(
    rpm: np.ndarray,
    power_mw: np.ndarray,
    *,
    standstill_rpm: float = 50.0,
    turbine_power_mw: float = 5.0,
    pump_power_mw: float = -5.0,
    ph_sustained_s: float = 600.0,
    window_s: float = 1.0,
) -> np.ndarray:
    """Faithful re-implementation of `ScadaTimeline._classify`, in the partner's
    own vocabulary {"ST","TU","PU","PH","TRANS","unknown"}.

    See module docstring (Variant B) for the file+line attribution and the
    exact rule; parameter names mirror the partner's own local variables/
    literals one-for-one. `window_s` has no partner-side equivalent -- it
    converts the PH-run-length test from seconds to a window count
    (module docstring, ADAPTATIONS §2).
    """
    if rpm.shape != power_mw.shape:
        raise ValueError(f"rpm shape {rpm.shape} != power_mw shape {power_mw.shape}")

    known = ~(np.isnan(rpm) | np.isnan(power_mw))
    raw = np.where(
        np.abs(rpm) < standstill_rpm, "ST",
        np.where(
            power_mw > turbine_power_mw, "TU",
            np.where(power_mw < pump_power_mw, "PU", "PH"),
        ),
    )
    state = raw.astype("<U16")
    state[~known] = "unknown"

    min_windows = ph_sustained_s / window_s
    for start, stop in _contiguous_runs(state):
        if state[start] == "PH" and (stop - start) < min_windows:
            state[start:stop] = "TRANS"

    return state


def partner_mode_b(
    rpm: np.ndarray,
    power_mw: np.ndarray,
    *,
    standstill_rpm: float = 50.0,
    turbine_power_mw: float = 5.0,
    pump_power_mw: float = -5.0,
    ph_sustained_s: float = 600.0,
    window_s: float = 1.0,
) -> np.ndarray:
    """`partner_mode_b_raw`, mapped onto our own GT vocabulary (`STATE_ORDER`)."""
    raw = partner_mode_b_raw(
        rpm, power_mw,
        standstill_rpm=standstill_rpm, turbine_power_mw=turbine_power_mw,
        pump_power_mw=pump_power_mw, ph_sustained_s=ph_sustained_s, window_s=window_s,
    )
    return _remap(raw, _B_TO_OURS)


# ---------------------------------------------------------------------------
# Variant A -- state/scada_estimator.py::estimate_operating_point +
# detect_transition_mask
# ---------------------------------------------------------------------------

_A_TO_OURS: dict[str, str] = {
    "ST": "standstill", "TU_full": "turbine", "TU_partial": "turbine",
    "PU": "pump", "PH": "phase-shifter", "transition": "transition",
}


def partner_mode_a_raw(
    rpm: np.ndarray,
    power_mw: np.ndarray,
    vane_pct: np.ndarray,
    *,
    standstill_rpm: float = 10.0,
    pump_power_mw: float = -20.0,
    full_load_vane_pct: float = 55.0,
    partial_load_vane_pct: float = 5.0,
    transition_drpm_per_s: float = 50.0,
    transition_dvane_per_s: float = 5.0,
    sample_rate_hz: float = 1.0,
) -> np.ndarray:
    """Faithful re-implementation of `estimate_operating_point` +
    `detect_transition_mask`, in the partner's own vocabulary
    {"ST","TU_full","TU_partial","PU","PH","transition","unknown"}.

    See module docstring (Variant A) for the file+line attribution.
    `partial_load_vane_pct` names the partner's own inline literal `5.0`
    (not a `ModeThresholds` field in the original -- only `full_load_vane_pct`
    is) for clarity/testability; its default reproduces that literal exactly.
    Static per-window classification first (branch order matches the
    original's if/elif chain exactly, including that PU is checked before
    rpm sign), then any window whose rpm or vane slew exceeds the transition
    thresholds is overwritten to "transition", mirroring `estimate_
    operating_track`'s own two-stage structure.
    """
    if not (rpm.shape == power_mw.shape == vane_pct.shape):
        raise ValueError(
            f"rpm/power_mw/vane_pct shape mismatch: "
            f"{rpm.shape}, {power_mw.shape}, {vane_pct.shape}"
        )

    known = ~(np.isnan(rpm) | np.isnan(power_mw) | np.isnan(vane_pct))

    is_standstill = np.abs(rpm) < standstill_rpm
    is_pump = ~is_standstill & (power_mw < pump_power_mw)
    is_tu_full = ~is_standstill & ~is_pump & (rpm > 0) & (vane_pct > full_load_vane_pct)
    is_tu_partial = (
        ~is_standstill & ~is_pump & (rpm > 0)
        & (vane_pct > partial_load_vane_pct) & (vane_pct <= full_load_vane_pct)
    )
    is_ph = ~is_standstill & ~is_pump & (rpm > 0) & (vane_pct <= partial_load_vane_pct)

    # Default "transition" matches estimate_operating_point's final `else`
    # branch (not standstill/pump, and rpm <= 0).
    state = np.full(rpm.shape, "transition", dtype="<U16")
    state[is_standstill] = "ST"
    state[is_pump] = "PU"
    state[is_tu_full] = "TU_full"
    state[is_tu_partial] = "TU_partial"
    state[is_ph] = "PH"

    with np.errstate(invalid="ignore"):
        drpm_dt = np.abs(np.gradient(rpm)) * sample_rate_hz
        dvane_dt = np.abs(np.gradient(vane_pct)) * sample_rate_hz
    slew = (drpm_dt > transition_drpm_per_s) | (dvane_dt > transition_dvane_per_s)
    state[slew & known] = "transition"

    state[~known] = "unknown"
    return state


def partner_mode_a(
    rpm: np.ndarray,
    power_mw: np.ndarray,
    vane_pct: np.ndarray,
    *,
    standstill_rpm: float = 10.0,
    pump_power_mw: float = -20.0,
    full_load_vane_pct: float = 55.0,
    partial_load_vane_pct: float = 5.0,
    transition_drpm_per_s: float = 50.0,
    transition_dvane_per_s: float = 5.0,
    sample_rate_hz: float = 1.0,
) -> np.ndarray:
    """`partner_mode_a_raw`, collapsing the turbine full/partial-load split
    (our own GT does not distinguish load bands within "turbine" -- `rowii.
    scada.labels.STATES`) and mapped onto our GT vocabulary (`STATE_ORDER`)."""
    raw = partner_mode_a_raw(
        rpm, power_mw, vane_pct,
        standstill_rpm=standstill_rpm, pump_power_mw=pump_power_mw,
        full_load_vane_pct=full_load_vane_pct, partial_load_vane_pct=partial_load_vane_pct,
        transition_drpm_per_s=transition_drpm_per_s, transition_dvane_per_s=transition_dvane_per_s,
        sample_rate_hz=sample_rate_hz,
    )
    return _remap(raw, _A_TO_OURS)


# ---------------------------------------------------------------------------
# Disagreement-location analysis
# ---------------------------------------------------------------------------


def _distance_to_change_s(state: np.ndarray, window_s: float) -> np.ndarray:
    """Per-window distance (seconds) to the nearest state-change boundary in
    *state* (0.0 at the change-point windows themselves; `inf` if *state* never
    changes). Used to test whether our-vs-partner disagreements cluster near
    OUR OWN GT's transition zones (buffer/ramp-handling differences) or are
    spread through steady-state regions (threshold-value differences)."""
    n = len(state)
    change = np.zeros(n, dtype=bool)
    change[1:] = state[1:] != state[:-1]
    change_idx = np.flatnonzero(change)
    idx = np.arange(n)
    if change_idx.size == 0:
        return np.full(n, np.inf)

    pos = np.searchsorted(change_idx, idx)
    prev_idx = change_idx[np.maximum(pos - 1, 0)]
    next_idx = change_idx[np.minimum(pos, change_idx.size - 1)]
    big = np.iinfo(np.int64).max
    dist_prev = np.where(pos > 0, idx - prev_idx, big)
    dist_next = np.where(pos < change_idx.size, next_idx - idx, big)
    dist_windows = np.minimum(dist_prev, dist_next).astype(np.float64)
    return dist_windows * window_s


# ---------------------------------------------------------------------------
# Driver: load our SCADA + GT for a named run, apply the partner rule(s)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunWindowData:
    run_name: str
    t_utc: pd.DatetimeIndex
    rpm: np.ndarray
    power_mw: np.ndarray
    vane_pct: np.ndarray
    our_state: np.ndarray
    our_load_bin: np.ndarray


def _run_window_data(run_name: str, cfg: Config, index: RecordingIndex) -> RunWindowData | None:
    """Grid + SCADA window means + OUR GT for *run_name*, or None if that run
    has zero Betriebsdaten coverage over its own audio grid.

    Reuses `rowii.pipeline.build_run_grid` (via `run_step1`, repo convention --
    see `scripts/analyze_step1.py`) and `rowii.scada.labels.{load_scada_
    window_means,gt_labels}` exactly as `scripts/run_step1.py::load_run_gt`
    does -- but WITHOUT that function's audio-validity "unknown" overwrite:
    this is a pure SCADA-rule-vs-SCADA-rule comparison, independent of
    whether the audio pipeline could extract features for a given window.
    """
    matches = [r for r in index.runs if r.name == run_name]
    if not matches:
        raise ValueError(
            f"run {run_name!r} not found (available: "
            f"{sorted(r.name for r in index.runs)})"
        )
    run: Run = matches[0]
    betriebsdaten = index.betriebsdaten_by_day.get(run.day_root, [])
    streams = tuple(sorted(run.files.keys()))
    grid = run_step1.build_run_grid(run, streams=streams, window_s=cfg.window.window_s)
    matched_bd = run_step1._betriebsdaten_for_grid(betriebsdaten, grid)
    if not matched_bd:
        logger.warning(
            "run %s: zero Betriebsdaten files overlap its grid -- no SCADA "
            "coverage, skipped", run_name,
        )
        return None

    scada = load_scada_window_means(
        matched_bd, grid, audio_run_offset_ns=run_utc_offset_ns(run)
    )
    gt = gt_labels(scada, cfg.gt, window_s=cfg.window.window_s)

    edges_ns = grid.edges_ns()[:-1].astype(np.int64)
    t_utc = pd.to_datetime(edges_ns, unit="ns", utc=True)

    return RunWindowData(
        run_name=run_name,
        t_utc=t_utc,
        rpm=scada["speed"].to_numpy(dtype=np.float64),
        power_mw=scada["power"].to_numpy(dtype=np.float64),
        vane_pct=scada["guide_vane"].to_numpy(dtype=np.float64),
        our_state=gt["state"].to_numpy(),
        our_load_bin=gt["load_bin"].to_numpy(),
    )


def compare_run(run_name: str, cfg: Config, index: RecordingIndex) -> pd.DataFrame | None:
    """Per-window comparison table for *run_name*: our GT vs. partner Variant B."""
    data = _run_window_data(run_name, cfg, index)
    if data is None:
        return None

    partner_b = partner_mode_b(data.rpm, data.power_mw, window_s=cfg.window.window_s)
    known_both = (data.our_state != "unknown") & (partner_b != "unknown")

    return pd.DataFrame({
        "run": run_name,
        "window_idx": np.arange(len(data.our_state)),
        "t_utc": data.t_utc,
        "rpm": data.rpm,
        "power_mw": data.power_mw,
        "vane_pct": data.vane_pct,
        "our_state": data.our_state,
        "our_load_bin": data.our_load_bin,
        "partner_b_state": partner_b,
        "known_both": known_both,
        "agree": known_both & (data.our_state == partner_b),
        "dist_to_our_change_s": _distance_to_change_s(data.our_state, cfg.window.window_s),
    })


def compare_run_variant_a(run_name: str, cfg: Config, index: RecordingIndex) -> pd.DataFrame | None:
    """Per-window comparison table for *run_name*: our GT vs. partner Variant A
    (secondary/historical -- see module docstring)."""
    data = _run_window_data(run_name, cfg, index)
    if data is None:
        return None

    sample_rate_hz = 1.0 / cfg.window.window_s
    raw_a = partner_mode_a_raw(
        data.rpm, data.power_mw, data.vane_pct, sample_rate_hz=sample_rate_hz
    )
    mapped_a = _remap(raw_a, _A_TO_OURS)
    known_both = (data.our_state != "unknown") & (mapped_a != "unknown")

    return pd.DataFrame({
        "run": run_name,
        "window_idx": np.arange(len(data.our_state)),
        "t_utc": data.t_utc,
        "rpm": data.rpm,
        "power_mw": data.power_mw,
        "vane_pct": data.vane_pct,
        "our_state": data.our_state,
        "our_load_bin": data.our_load_bin,
        "partner_a_state_raw": raw_a,
        "partner_a_state": mapped_a,
        "known_both": known_both,
        "agree": known_both & (data.our_state == mapped_a),
        "dist_to_our_change_s": _distance_to_change_s(data.our_state, cfg.window.window_s),
    })


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _per_run_agreement(window_labels: pd.DataFrame, partner_col: str) -> pd.DataFrame:
    rows = []
    for run_name, g in window_labels.groupby("run", sort=False):
        known = g[g["known_both"]]
        rows.append({
            "run": run_name,
            "n_windows": len(g),
            "n_known_both": len(known),
            "n_agree": int(known["agree"].sum()),
            "agreement_rate": float(known["agree"].mean()) if len(known) else float("nan"),
            "our_states_present": ",".join(
                sorted(g.loc[g["our_state"] != "unknown", "our_state"].unique())
            ),
            "partner_states_present": ",".join(
                sorted(g.loc[g[partner_col] != "unknown", partner_col].unique())
            ),
        })
    return pd.DataFrame(rows)


def _pump_power_sign_check(window_labels: pd.DataFrame) -> pd.DataFrame:
    """For every window OUR GT calls "pump", what sign does power_mw actually
    have? Directly tests whether the partner's power-sign-only pump rule
    (Variant B: `power_mw < -5 -> PU`) is compatible with our own data, or
    whether the plant's pump-power sign convention (flagged in `rowii.scada.
    labels`'s own module docstring: "Pump power may be logged POSITIVE at
    this plant") actually triggers here."""
    rows = []
    for run_name, g in window_labels.groupby("run", sort=False):
        pump = g[g["our_state"] == "pump"]
        if len(pump) == 0:
            continue
        rows.append({
            "run": run_name,
            "n_our_pump_windows": len(pump),
            "n_power_negative": int((pump["power_mw"] < 0).sum()),
            "n_power_positive": int((pump["power_mw"] > 0).sum()),
            "frac_power_positive": float((pump["power_mw"] > 0).mean()),
            "power_mw_median": float(pump["power_mw"].median()),
        })
    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame, *, index: bool = False, floatfmt: str = ".4f") -> str:
    """Minimal Markdown table renderer.

    `pandas.DataFrame.to_markdown` needs the optional `tabulate` package,
    which is not part of this repo's dependency set (`pyproject.toml`) --
    adding it for one script's report formatting is not worth a new
    dependency, so this hand-rolls the (small) subset needed here."""
    work = df.reset_index() if index else df
    headers = [str(c) for c in work.columns]

    def fmt(v: object) -> str:
        if isinstance(v, float):
            if np.isnan(v):
                return "nan"
            if np.isinf(v):
                return "inf"
            return format(v, floatfmt)
        return str(v)

    rows = [[fmt(v) for v in row] for row in work.itertuples(index=False, name=None)]
    widths = [
        max([len(headers[i]), *(len(r[i]) for r in rows)]) for i in range(len(headers))
    ]

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)) + " |"

    lines = [render_row(headers), render_row(["-" * w for w in widths])]
    lines += [render_row(r) for r in rows]
    return "\n".join(lines)


def _write_summary_md(
    out_dir: Path,
    window_labels: pd.DataFrame,
    confusion: pd.DataFrame,
    per_run: pd.DataFrame,
    pump_sign: pd.DataFrame,
    boundary_buckets: pd.DataFrame,
    variant_a_confusion: pd.DataFrame | None,
    variant_a_per_run: pd.DataFrame | None,
) -> None:
    known = window_labels[window_labels["known_both"]]
    overall_rate = float(known["agree"].mean()) if len(known) else float("nan")

    lines = [
        "# Operating-mode label comparison: ours vs. the partner's (Bruno's) SCADA rule",
        "",
        "**Partner methodology re-implemented from his read-only reference repo, "
        "attributed by file + line; no partner-computed label, score, or result is "
        "imported into our pipeline or claimed as ours. Every number in this file is "
        "computed fresh from OUR OWN Betriebsdaten by `scripts/compare_partner_labels.py`. "
        "Our own `rowii.scada.labels.gt_labels` stays the primary ground truth "
        "throughout.**",
        "",
        "Primary comparison: partner Variant B (`ScadaTimeline._classify`, "
        "`repos/hydropower-anomaly` git show cf92b0f:src/hydropower_anomaly/"
        "ingestion/scada.py) -- the rule that actually produced the state labels "
        "behind `tools/run_290626_day_analysis.py` and `tools/run_010726_extract.py` "
        "(290626/010726 campaigns). Secondary/historical: partner Variant A "
        "(`estimate_operating_point`, git show cf92b0f:src/hydropower_anomaly/state/"
        "scada_estimator.py), the rule actually used for the 2026-06-25 (250526) "
        "burst only -- reported separately below, on 250526 runs only.",
        "",
        "## Overall agreement (Variant B, all shared days)",
        "",
        f"- Windows compared (both sides known): {len(known)} / {len(window_labels)}",
        f"- Overall agreement rate: {overall_rate:.4f}",
        "",
        "## Per-run agreement (Variant B)",
        "",
        _df_to_markdown(per_run),
        "",
        "## Confusion matrix (Variant B; rows = our GT, columns = partner; "
        "known-to-both windows only)",
        "",
        _df_to_markdown(confusion, index=True, floatfmt=".0f"),
        "",
        "## Disagreement location: distance to OUR own nearest GT state change",
        "",
        "Tests whether disagreements cluster near our GT's own transition buffer "
        "(`GtRules.transition_buffer_s`, ramp handling) or are spread through "
        "steady-state regions (threshold-value / sign-convention differences).",
        "",
        _df_to_markdown(boundary_buckets, floatfmt=".4f"),
        "",
        "## Pump-power sign check (our GT \"pump\" windows only)",
        "",
        "The partner's Variant B rule classifies purely on `power_mw` sign "
        "(`power_mw < -5 -> PU`), with no flow-based override. Our own GT rule "
        "(`rowii.scada.labels._base_state`) instead treats flow dominance as "
        "primary and power sign as a fallback, because its own module docstring "
        "warns pump power \"may be logged POSITIVE at this plant\". This table "
        "checks, per run, what sign `power_mw` actually has on every window OUR "
        "GT calls \"pump\" -- a negative median here means the partner's power-sign "
        "rule and ours agree on convention for that run; a positive median would "
        "explain a systematic pump<->turbine confusion.",
        "",
        (
            _df_to_markdown(pump_sign, floatfmt=".2f")
            if len(pump_sign)
            else "(no pump windows in our GT on any compared run)"
        ),
        "",
    ]

    if variant_a_confusion is not None and variant_a_per_run is not None:
        lines += [
            "## Appendix: Variant A (historical, 250526 only)",
            "",
            "`estimate_operating_point` + `detect_transition_mask` "
            "(`state/scada_estimator.py`), the rule the partner's own tools "
            "actually used for the 2026-06-25 burst -- applied here to both "
            "250526 runs with SCADA coverage, for direct comparison against "
            "Variant B on the same days.",
            "",
            "### Per-run agreement (Variant A)",
            "",
            _df_to_markdown(variant_a_per_run),
            "",
            "### Confusion matrix (Variant A; rows = our GT, columns = partner)",
            "",
            _df_to_markdown(variant_a_confusion, index=True, floatfmt=".0f"),
            "",
        ]

    lines += [
        "## Attribution",
        "",
        "Partner rule source: `repos/hydropower-anomaly` (StefanKrum workspace "
        "read-only clone), working tree 101 commits stale relative to its own "
        "`main` -- every threshold and code excerpt referenced above was read via "
        "`git show cf92b0f:<path>`, never from a checkout, and the repo was never "
        "modified. See `research/notes/analysis_2026-08-12_partner_label_agreement.md` "
        "for the full narrative writeup and interpretation.",
        "",
    ]

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _boundary_buckets(window_labels: pd.DataFrame) -> pd.DataFrame:
    disagree = window_labels[window_labels["known_both"] & ~window_labels["agree"]]
    edges = [0.0, 10.0, 30.0, 60.0, 300.0, np.inf]
    bucket_labels = ["[0,10)s", "[10,30)s", "[30,60)s", "[60,300)s", "[300,inf)s"]
    binned = pd.cut(
        disagree["dist_to_our_change_s"], bins=edges, labels=bucket_labels,
        right=False, include_lowest=True,
    )
    counts = binned.value_counts().reindex(bucket_labels, fill_value=0)
    total = len(disagree)
    frac = (counts / total).to_numpy() if total else np.zeros(len(bucket_labels))
    return pd.DataFrame({
        "distance_to_our_gt_change_bucket": bucket_labels,
        "n_disagreements": counts.to_numpy(),
        "frac_of_disagreements": frac,
    })


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = load_config()
    index = discover(cfg.data_root)
    out_dir = cfg.results_root / "crosscheck-labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for run_name in PRIMARY_RUNS:
        df = compare_run(run_name, cfg, index)
        if df is not None:
            frames.append(df)
    if not frames:
        raise RuntimeError("no run in PRIMARY_RUNS produced SCADA-covered windows")
    window_labels = pd.concat(frames, ignore_index=True)
    window_labels.to_csv(out_dir / "window_labels.csv", index=False)

    known = window_labels[window_labels["known_both"]]
    confusion = pd.crosstab(known["our_state"], known["partner_b_state"])
    confusion.to_csv(out_dir / "confusion_matrix.csv")

    per_run = _per_run_agreement(window_labels, "partner_b_state")
    per_run.to_csv(out_dir / "per_run_agreement.csv", index=False)

    disagreements = window_labels[window_labels["known_both"] & ~window_labels["agree"]]
    disagreements.to_csv(out_dir / "disagreements.csv", index=False)

    pump_sign = _pump_power_sign_check(window_labels)
    pump_sign.to_csv(out_dir / "pump_power_sign_check.csv", index=False)

    boundary_buckets = _boundary_buckets(window_labels)
    boundary_buckets.to_csv(out_dir / "disagreement_boundary_distance.csv", index=False)

    variant_a_confusion: pd.DataFrame | None = None
    variant_a_per_run: pd.DataFrame | None = None
    a_frames = []
    for run_name in VARIANT_A_RUNS:
        df = compare_run_variant_a(run_name, cfg, index)
        if df is not None:
            a_frames.append(df)
    if a_frames:
        variant_a_labels = pd.concat(a_frames, ignore_index=True)
        variant_a_labels.to_csv(out_dir / "variant_a_250526_window_labels.csv", index=False)
        a_known = variant_a_labels[variant_a_labels["known_both"]]
        variant_a_confusion = pd.crosstab(a_known["our_state"], a_known["partner_a_state"])
        variant_a_confusion.to_csv(out_dir / "variant_a_250526_confusion.csv")
        variant_a_per_run = _per_run_agreement(variant_a_labels, "partner_a_state")
        variant_a_per_run.to_csv(out_dir / "variant_a_250526_per_run_agreement.csv", index=False)

    _write_summary_md(
        out_dir, window_labels, confusion, per_run, pump_sign, boundary_buckets,
        variant_a_confusion, variant_a_per_run,
    )

    overall_rate = float(known["agree"].mean()) if len(known) else float("nan")
    print(
        f"compare_partner_labels: {len(frames)} run(s), {len(window_labels)} window(s), "
        f"overall agreement (Variant B) = {overall_rate:.4f} -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
