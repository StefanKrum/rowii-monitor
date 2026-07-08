"""Rule-based operating-state ground truth derived from Betriebsdaten (SCADA) channels.

SCADA is never fed to the detector at run time; this module only produces the labels used
to evaluate unsupervised state detection (spec: docs/superpowers/specs/2026-07-05-step1-
state-detection-design.md, extended by docs/superpowers/specs/2026-07-07-step1-multiday-
phase-shifter-addendum.md §3 for the "phase-shifter" state). Two stages: (1)
`load_scada_window_means` reduces raw Betriebsdaten samples to one row per detection
window, (2) `gt_labels` turns those window means into a discrete state + load bin per the
plant's operating rules.

Rule ordering inside `gt_labels` (addendum spec §3, exact order matters):
`_base_state` -> `_apply_ph_promotion` -> `_apply_ramp` -> `_apply_transition_buffer`.
Phase-shifter promotion runs BEFORE the ramp rule so the ramp rule can explicitly protect
already-promoted windows (a phase-shifter run's own near-zero power should rarely trip the
ramp threshold, but the rule must still never demote a promoted window even if it does);
the transition buffer runs LAST and is deliberately unaware of phase-shifter as anything
special -- it treats a standstill/turbine/pump/phase-shifter boundary uniformly, so only a
promoted run's own EDGE windows (within the buffer radius of a boundary) can ever become
"transition" again, never its interior (in practice never possible in practice at any
sane parameter setting, since `ph_min_dwell_s` is two orders of magnitude larger than
`transition_buffer_s`).
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from rowii.config import GtRules
from rowii.io.gantner import read_gantner
from rowii.signals.windows import WindowGrid, window_slices

logger = logging.getLogger(__name__)

GT_CHANNELS: Mapping[str, str] = {
    "power": "1_P_Ist",
    # "1_Drehzahl UPM" ("Umdrehungen Pro Minute" = rpm) is the genuine rpm channel --
    # verified against the real 2026-06-25 delivery (Task 13b): its plateau during
    # full-power turbine generation is 378.832 rpm, vs. "1_Drehzahl_Ist"'s ~101 on the
    # SAME file (ratio ~3.75x, consistent with a percent-of-nominal-ish quantity, not
    # rpm). Task 13 originally wired GT_CHANNELS["speed"] to "1_Drehzahl_Ist" and
    # measured ITS plateau (~101 rpm), silently taking the wrong channel's number as
    # the machine's nominal speed -- see results/parameter_verification.md's Revision
    # 2026-07-07 section for the corrected derivation.
    "speed": "1_Drehzahl UPM",
    "guide_vane": "1_Leitapparat Stell.",
    "flow_tu": "Durchfluss TU",
    "flow_pu": "Durchfluss PU",
    # Multi-day/phase-shifter addendum (spec §3): loaded for verification/reporting --
    # the phase-shifter promotion rule itself uses speed+power+dwell (± the ks_valve
    # gate below when enabled), never reactive power directly.
    "reactive": "1_Q_Ist",
    # Spherical inlet valve position -- optional conjunctive gate for phase-shifter
    # promotion (see `GtRules.ph_requires_ks_closed`). Verified present (exact name) in
    # every SCADA-bearing day's Betriebsdaten during the addendum's own data audit.
    "ks_valve": "1_KS Stellung",
}

STATES: tuple[str, ...] = ("standstill", "turbine", "pump", "transition", "phase-shifter")

_KNOWN_STATES = frozenset({"standstill", "turbine", "pump", "phase-shifter"})
_PH_STATE = "phase-shifter"
_TRANSITION_STATE = "transition"


def load_scada_window_means(
    files: list[Path],
    grid: WindowGrid,
    channels: Mapping[str, str] = GT_CHANNELS,
) -> pd.DataFrame:
    """Per-window mean of each GT channel across one or more (hourly) Betriebsdaten files.

    Each file is read once; *files* are assumed already time-sorted (hourly Betriebsdaten
    files from `dataset.discover`) and are concatenated in the given order before slicing
    into *grid*'s windows. Windows with zero SCADA samples get NaN in every column.
    """
    ts_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    keys = list(channels.keys())
    for path in files:
        gf = read_gantner(path)
        available = gf.header.channel_names
        col_idx = []
        for key in keys:
            name = channels[key]
            try:
                col_idx.append(available.index(name))
            except ValueError as exc:
                raise KeyError(
                    f"channel {name!r} (for {key!r}) not found in {path.name}; "
                    f"available channels: {available}"
                ) from exc
        ts_parts.append(gf.timestamps_ns)
        data_parts.append(gf.data[:, col_idx])

    ts_ns = np.concatenate(ts_parts) if ts_parts else np.array([], dtype=np.uint64)
    data = (
        np.concatenate(data_parts, axis=0)
        if data_parts
        else np.zeros((0, len(keys)), dtype=np.float32)
    )

    slices = window_slices(ts_ns, grid)
    means = np.full((grid.n_windows, len(keys)), np.nan, dtype=np.float64)
    for i, sl in enumerate(slices):
        if sl.stop > sl.start:
            means[i] = data[sl].mean(axis=0)

    return pd.DataFrame(means, columns=keys, index=pd.RangeIndex(grid.n_windows))


def _base_state(scada: pd.DataFrame, rules: GtRules) -> pd.Series:
    power = scada["power"].to_numpy(dtype=np.float64)
    speed = scada["speed"].to_numpy(dtype=np.float64)
    flow_tu = scada["flow_tu"].to_numpy(dtype=np.float64)
    flow_pu = scada["flow_pu"].to_numpy(dtype=np.float64)

    known = ~(np.isnan(power) | np.isnan(speed))
    n_nom = rules.speed_nominal_rpm
    is_standstill = known & (np.abs(speed) < rules.speed_eps_frac * n_nom) & (
        np.abs(power) < rules.power_eps_mw
    )
    # "1_Drehzahl UPM" (GT_CHANNELS["speed"]) is SIGNED by rotation direction at this
    # plant (verified Task 13/13b, real 2026-06-25 data): positive during turbine
    # operation, negative during pump operation (a reversible pump-turbine spins the
    # opposite way in each mode) -- the nominal-speed gate must compare the MAGNITUDE
    # against the threshold, exactly like is_standstill already does above, or every
    # pump-mode window fails this check and falls through to "transition" regardless
    # of how the flow/power rules below would otherwise resolve it.
    is_nominal = known & (np.abs(speed) >= (1.0 - rules.speed_eps_frac) * n_nom)

    # Pump power may be logged POSITIVE at this plant (plant-specific SCADA convention) --
    # flow dominance at nominal speed is the ground truth's way of overriding a misleading
    # power sign: whichever flow path (TU vs PU) is actually carrying water wins over the
    # naive P > 0 => turbine / P < 0 => pump heuristic.
    pump_by_flow = is_nominal & (flow_pu > flow_tu)
    turbine_by_flow = is_nominal & (flow_tu > flow_pu)
    turbine_by_power = is_nominal & (power > rules.power_eps_mw) & ~pump_by_flow
    pump_by_power = is_nominal & (power < -rules.power_eps_mw) & ~turbine_by_flow

    state = np.full(len(scada), "transition", dtype=object)
    state[~known] = "unknown"
    state[known & is_standstill] = "standstill"
    state[known & is_nominal & (turbine_by_flow | turbine_by_power)] = "turbine"
    state[known & is_nominal & (pump_by_flow | pump_by_power)] = "pump"
    return pd.Series(state, index=scada.index)


def _ph_candidate_mask(scada: pd.DataFrame, rules: GtRules) -> np.ndarray:
    """Windows satisfying the phase-shifter BASE criteria: nominal speed, |P| <= eps.

    This is intentionally computed directly from `power`/`speed` rather than as
    "`_base_state` says transition" -- a sub-nominal ramp-up/down window also falls
    through `_base_state` to `"transition"`, but must NOT be a phase-shifter
    candidate (only the unloaded-but-AT-nominal-speed sub-case is a genuine phase-
    shifter candidate; addendum spec §3).

    Uses `rules.ph_power_eps_mw` (a DEDICATED threshold), not `rules.power_eps_mw` --
    see `GtRules.ph_power_eps_mw`'s own docstring for why: real phase-shifter idling
    power (measured on 2026-07-01) sits at a stable ~3.5 MW magnitude, well outside
    the tighter power_eps_mw used for standstill/loaded turbine-pump discrimination.
    """
    power = scada["power"].to_numpy(dtype=np.float64)
    speed = scada["speed"].to_numpy(dtype=np.float64)
    known = ~(np.isnan(power) | np.isnan(speed))
    n_nom = rules.speed_nominal_rpm
    is_nominal = np.abs(speed) >= (1.0 - rules.speed_eps_frac) * n_nom
    is_unloaded = np.abs(power) <= rules.ph_power_eps_mw
    result: np.ndarray = known & is_nominal & is_unloaded
    return result


def _ph_ks_gate_mask(scada: pd.DataFrame, rules: GtRules) -> np.ndarray:
    """True where the optional KS-valve gate does NOT block promotion.

    Always all-True when `rules.ph_requires_ks_closed` is False (the gate is
    disabled). When enabled: a window passes if `ks_valve <= ks_closed_max`, OR if
    `ks_valve` is NaN for that window -- a NaN reading means the gate cannot be
    evaluated, so it is IGNORED for that window (falls back to promoting purely on
    speed/power/dwell) rather than silently treated as either closed or open; a
    warning is logged once per call when this fallback is exercised at all.
    """
    if not rules.ph_requires_ks_closed:
        return np.ones(len(scada), dtype=bool)

    ks_valve = scada["ks_valve"].to_numpy(dtype=np.float64)
    is_nan = np.isnan(ks_valve)
    if is_nan.any():
        logger.warning(
            "ph_requires_ks_closed is True but ks_valve (%r) is NaN for %d/%d window(s); "
            "the KS gate is ignored for those windows (falls back to speed/power/dwell "
            "criteria alone)",
            GT_CHANNELS["ks_valve"], int(is_nan.sum()), len(scada),
        )
    result: np.ndarray = is_nan | (ks_valve <= rules.ks_closed_max)
    return result


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of every maximal contiguous run of True in *mask*."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _apply_ph_promotion(
    state: pd.Series, scada: pd.DataFrame, rules: GtRules, window_s: float
) -> pd.Series:
    """Promote contiguous nominal-speed/unloaded runs >= `ph_min_dwell_s` to phase-shifter.

    Runs on the OUTPUT of `_base_state`. Because `ph_power_eps_mw` is deliberately
    WIDER than `power_eps_mw` (see `GtRules.ph_power_eps_mw`), a qualifying window can
    have been base-classified as "transition" (the |P| <= power_eps_mw fallthrough) OR
    as "pump" (real-data finding, 2026-07-01: `_base_state`'s `pump_by_power` fires on
    ANY power < -power_eps_mw regardless of actual water flow -- a motoring window with
    zero flow_pu/flow_tu but power in the (power_eps_mw, ph_power_eps_mw] range is
    genuinely NOT pumping, just spinning at nominal speed drawing a small amount of
    grid power, which `_base_state` alone cannot distinguish from real pumping).
    Promotion intentionally overrides either base classification for a long enough
    dwell -- this is the correction mechanism, not a bug. Shorter candidate runs
    (below the dwell threshold) are left completely untouched (whatever `_base_state`
    decided stands), matching a start/stop ramp's unloaded-spinning phase.
    """
    candidate = _ph_candidate_mask(scada, rules) & _ph_ks_gate_mask(scada, rules)
    min_windows = rules.ph_min_dwell_s / window_s

    out = state.to_numpy(copy=True)
    for start, stop in _contiguous_true_runs(candidate):
        if (stop - start) >= min_windows:
            out[start:stop] = _PH_STATE
    return pd.Series(out, index=state.index)


def _apply_ramp(
    state: pd.Series, scada: pd.DataFrame, rules: GtRules, window_s: float
) -> pd.Series:
    power = scada["power"].to_numpy(dtype=np.float64)
    n = len(power)
    dpdt = np.full(n, np.nan)
    # Centered difference over window means; windows are uniform, so dt = 2 * window_s
    # between the two neighbors used (Task-4 WindowGrid guarantees uniform window_ns).
    for i in range(1, n - 1):
        if not (np.isnan(power[i - 1]) or np.isnan(power[i + 1])):
            dpdt[i] = (power[i + 1] - power[i - 1]) / (2.0 * window_s)

    out = state.to_numpy(copy=True)
    ramp_hit = ~np.isnan(dpdt) & (np.abs(dpdt) > rules.ramp_mw_per_s)
    # Phase-shifter windows are never demoted by the ramp rule (addendum spec §3:
    # "ramp rule ... never demotes PH interiors") -- a promoted run's own near-zero
    # power rarely trips this threshold anyway, but this guard makes the invariant
    # explicit rather than relying on that coincidence.
    out[(out != "unknown") & (out != _PH_STATE) & ramp_hit] = _TRANSITION_STATE
    return pd.Series(out, index=state.index)


def _apply_transition_buffer(
    state: pd.Series, base_state: pd.Series, rules: GtRules, window_s: float
) -> pd.Series:
    base = base_state.to_numpy()
    n = len(base)
    # round(), not floor(): a buffer_s that is a near-exact multiple of window_s (subject to
    # float error) should resolve to that multiple, not one window short of it. A buffer
    # shorter than half a window genuinely covers zero whole windows -- 0 is correct there,
    # not an arbitrary floor of 1.
    buffer_windows = round(rules.transition_buffer_s / window_s)

    change_indices = [
        i
        for i in range(1, n)
        if base[i - 1] in _KNOWN_STATES and base[i] in _KNOWN_STATES and base[i - 1] != base[i]
    ]

    out = state.to_numpy(copy=True)
    for ci in change_indices:
        lo, hi = max(0, ci - buffer_windows), min(n, ci + buffer_windows)
        for j in range(lo, hi):
            if out[j] != "unknown":
                out[j] = "transition"
    return pd.Series(out, index=state.index)


def _load_bins(power: pd.Series, n_bins: int) -> pd.Series:
    """Quantile-bin *power* into *n_bins* bins (0..k-1), collapsing duplicate edges.

    `pd.qcut(..., duplicates="drop")` yields fewer than *n_bins* bins when *power* has
    fewer distinct values than requested quantiles -- this is expected and still a valid,
    ordered binning (documented in the module + brief). In the degenerate case of a SINGLE
    distinct value, `qcut` cannot form any edge at all and returns NaN for every row; that
    single homogeneous load level is mapped to bin 0 instead of leaving it unbinned.
    """
    if power.nunique() <= 1:
        return pd.Series(np.zeros(len(power), dtype=np.int64), index=power.index)
    binned = pd.qcut(power, q=n_bins, labels=False, duplicates="drop")
    return binned.astype(np.int64)


def gt_labels(scada: pd.DataFrame, rules: GtRules, *, window_s: float) -> pd.DataFrame:
    """Rule-based state + load-bin labels for each window of *scada*.

    *window_s* is the detection window duration (seconds) used to build *scada* via
    `load_scada_window_means` -- required to convert `GtRules.transition_buffer_s` into a
    window count and to compute dP/dt for the ramp rule; it is not itself a `GtRules`
    field (that dataclass is a fixed, already-committed interface) nor derivable from
    *scada* alone (its index carries no timestamps).

    Rule ordering (addendum spec §3, see module docstring): base -> PH promotion ->
    ramp -> transition buffer. The transition buffer's own change-point detection uses
    the POST-promotion base (`ph_promoted`, not the pre-promotion `base`) -- a
    phase-shifter run's own edges are a genuine base-state boundary (e.g.
    standstill<->phase-shifter) that the buffer must be able to see and react to;
    the pre-promotion base would still read "transition" there, which is not in
    `_KNOWN_STATES` and would never register as a change point at all.
    """
    base = _base_state(scada, rules)
    ph_promoted = _apply_ph_promotion(base, scada, rules, window_s)
    state = _apply_ramp(ph_promoted, scada, rules, window_s)
    state = _apply_transition_buffer(state, ph_promoted, rules, window_s)

    load_bin = pd.Series(np.full(len(scada), -1, dtype=np.int64), index=scada.index)
    for st in ("turbine", "pump"):
        mask = state == st
        if mask.any():
            load_bin.loc[mask] = _load_bins(scada.loc[mask, "power"], rules.n_load_bins)

    return pd.DataFrame({"state": state, "load_bin": load_bin}, index=scada.index)
