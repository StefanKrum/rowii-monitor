"""Rule-based operating-state ground truth derived from Betriebsdaten (SCADA) channels.

SCADA is never fed to the detector at run time; this module only produces the labels used
to evaluate unsupervised state detection (spec: docs/superpowers/specs/2026-07-05-step1-
state-detection-design.md). Two stages: (1) `load_scada_window_means` reduces raw
Betriebsdaten samples to one row per detection window, (2) `gt_labels` turns those window
means into a discrete state + load bin per the plant's operating rules.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from rowii.config import GtRules
from rowii.io.gantner import read_gantner
from rowii.signals.windows import WindowGrid, window_slices

GT_CHANNELS: Mapping[str, str] = {
    "power": "1_P_Ist",
    "speed": "1_Drehzahl_Ist",
    "guide_vane": "1_Leitapparat Stell.",
    "flow_tu": "Durchfluss TU",
    "flow_pu": "Durchfluss PU",
}

STATES: tuple[str, ...] = ("standstill", "turbine", "pump", "transition")

_KNOWN_STATES = frozenset({"standstill", "turbine", "pump"})


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
    is_nominal = known & (speed >= (1.0 - rules.speed_eps_frac) * n_nom)

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
    out[(out != "unknown") & ramp_hit] = "transition"
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
    """
    base = _base_state(scada, rules)
    state = _apply_ramp(base, scada, rules, window_s)
    state = _apply_transition_buffer(state, base, rules, window_s)

    load_bin = pd.Series(np.full(len(scada), -1, dtype=np.int64), index=scada.index)
    for st in ("turbine", "pump"):
        mask = state == st
        if mask.any():
            load_bin.loc[mask] = _load_bins(scada.loc[mask, "power"], rules.n_load_bins)

    return pd.DataFrame({"state": state, "load_bin": load_bin}, index=scada.index)
