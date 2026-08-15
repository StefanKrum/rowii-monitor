"""Unit tests for the pure rule logic in `scripts/compare_partner_labels.py`.

Every expected value below is hand-derived directly from the partner's (Bruno's)
source, read read-only via `git show cf92b0f:<path>` in
`repos/hydropower-anomaly` (working tree is 101 commits stale -- see that
script's module docstring for the full file+line attribution). These tests are
the spec: they encode what the partner's rule DOES, not what a first-draft
re-implementation happened to produce.

Real-data driver tests (`compare_run`) are `@pytest.mark.data`, mirroring
`tests/test_real_data.py`'s own skip-if-no-`ROWII_DATA_ROOT` convention.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import compare_partner_labels as cpl  # noqa: E402

from rowii.config import load_config  # noqa: E402
from rowii.io.dataset import discover  # noqa: E402

_DATA_ROOT = load_config().data_root
_HAS_DATA_ROOT = _DATA_ROOT.is_dir()


# ---------------------------------------------------------------------------
# Variant B -- ingestion/scada.py::ScadaTimeline._classify (partner, 2026-07-03)
# Used by tools/run_290626_day_analysis.py (imports ScadaTimeline directly) and
# tools/run_010726_extract.py::load_scada_day (inline duplicate) -- the rule
# actually driving the 290626/010726 day-analysis scripts named in the task.
# ---------------------------------------------------------------------------


def test_partner_mode_b_raw_standstill_below_rpm_threshold() -> None:
    rpm = np.array([5.0, -5.0, 49.9])
    power = np.array([0.0, 0.0, 0.0])

    result = cpl.partner_mode_b_raw(rpm, power)

    assert list(result) == ["ST", "ST", "ST"]


def test_partner_mode_b_raw_rpm_exactly_at_standstill_threshold_is_not_standstill() -> None:
    # Original test: `np.abs(self._rpm) < 50` -- strict inequality.
    # ph_sustained_s=0.0 isolates the static threshold being tested here from
    # the (separately tested) PH-run dwell reclassification below.
    rpm = np.array([50.0])
    power = np.array([0.0])  # -5 <= 0 <= 5 -> falls through to "PH"

    result = cpl.partner_mode_b_raw(rpm, power, ph_sustained_s=0.0)

    assert list(result) == ["PH"]


def test_partner_mode_b_raw_turbine_above_power_threshold() -> None:
    rpm = np.array([300.0])
    power = np.array([10.0])

    result = cpl.partner_mode_b_raw(rpm, power)

    assert list(result) == ["TU"]


def test_partner_mode_b_raw_power_exactly_at_turbine_threshold_is_not_turbine() -> None:
    rpm = np.array([300.0])
    power = np.array([5.0])  # strict `> 5`, so 5.0 exactly falls through to PH

    result = cpl.partner_mode_b_raw(rpm, power, ph_sustained_s=0.0)

    assert list(result) == ["PH"]


def test_partner_mode_b_raw_pump_below_negative_power_threshold_regardless_of_rpm_sign() -> None:
    # The partner's rule has NO rpm-sign gate for TU vs PU (unlike variant A) --
    # power alone decides once the window is not standstill. Positive (turbine-
    # direction) rpm with power < -5 MW is still classified "PU", faithfully.
    rpm = np.array([300.0])
    power = np.array([-10.0])

    result = cpl.partner_mode_b_raw(rpm, power)

    assert list(result) == ["PU"]


def test_partner_mode_b_raw_short_ph_run_is_reclassified_transition() -> None:
    # ph_sustained_s=3.0, window_s=1.0 -> 3-window dwell threshold. A 2-window
    # PH-candidate run is strictly shorter, so it is demoted to "TRANS".
    rpm = np.array([300.0, 300.0])
    power = np.array([0.0, 0.0])

    result = cpl.partner_mode_b_raw(rpm, power, ph_sustained_s=3.0, window_s=1.0)

    assert list(result) == ["TRANS", "TRANS"]


def test_partner_mode_b_raw_ph_run_at_exactly_the_sustained_threshold_stays_ph() -> None:
    # Original test: `(self._ep[j-1] - self._ep[i]) < ph_sustained_s` -- strict
    # less-than, so a run of EXACTLY the threshold duration is NOT demoted.
    rpm = np.array([300.0, 300.0, 300.0])
    power = np.array([0.0, 0.0, 0.0])

    result = cpl.partner_mode_b_raw(rpm, power, ph_sustained_s=3.0, window_s=1.0)

    assert list(result) == ["PH", "PH", "PH"]


def test_partner_mode_b_raw_unknown_gap_breaks_ph_run_contiguity() -> None:
    # Two 2-window PH-candidate runs separated by one unknown (NaN) window, dwell
    # threshold 3 windows: each sub-run is independently too short (2 < 3) and
    # must be demoted on its own -- the gap must not let them "average out" or be
    # treated as one contiguous run.
    rpm = np.array([300.0, 300.0, np.nan, 300.0, 300.0])
    power = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    result = cpl.partner_mode_b_raw(rpm, power, ph_sustained_s=3.0, window_s=1.0)

    assert list(result) == ["TRANS", "TRANS", "unknown", "TRANS", "TRANS"]


def test_partner_mode_b_raw_nan_rpm_or_power_yields_unknown() -> None:
    rpm = np.array([np.nan, 300.0])
    power = np.array([0.0, np.nan])

    result = cpl.partner_mode_b_raw(rpm, power)

    assert list(result) == ["unknown", "unknown"]


def test_partner_mode_b_raw_shape_mismatch_raises_value_error() -> None:
    with pytest.raises(ValueError, match="shape"):
        cpl.partner_mode_b_raw(np.array([1.0, 2.0]), np.array([1.0]))


def test_partner_mode_b_maps_onto_our_vocabulary() -> None:
    # ST/TU/PU/PH/TRANS -> standstill/turbine/pump/phase-shifter/transition
    # (rowii.scada.labels.STATES); "unknown" passes through unchanged.
    rpm = np.array([5.0, 300.0, 300.0, np.nan])
    power = np.array([0.0, 10.0, -10.0, 0.0])

    result = cpl.partner_mode_b(rpm, power, ph_sustained_s=3.0, window_s=1.0)

    assert list(result) == ["standstill", "turbine", "pump", "unknown"]


def test_partner_mode_b_long_ph_run_maps_to_phase_shifter() -> None:
    rpm = np.array([300.0, 300.0, 300.0])
    power = np.array([0.0, 0.0, 0.0])

    result = cpl.partner_mode_b(rpm, power, ph_sustained_s=3.0, window_s=1.0)

    assert list(result) == ["phase-shifter", "phase-shifter", "phase-shifter"]


# ---------------------------------------------------------------------------
# Variant A -- state/scada_estimator.py::estimate_operating_point +
# detect_transition_mask (partner, 2026-06-15, unmodified since). Used by
# tools/{run_ablation,run_all_cluster,run_experiments,run_tierB}.py, all of
# which load only the 2026-06-25 (250526) Betriebsdaten.
# ---------------------------------------------------------------------------


def test_partner_mode_a_raw_standstill_below_rpm_threshold() -> None:
    rpm = np.array([5.0, -5.0, 9.9])
    power = np.array([0.0, 0.0, 0.0])
    vane = np.array([0.0, 0.0, 0.0])

    result = cpl.partner_mode_a_raw(rpm, power, vane)

    assert list(result) == ["ST", "ST", "ST"]


# Every remaining static-classification test below uses TWO identical
# consecutive samples (rather than one): `detect_transition_mask`'s
# `np.gradient` call needs >= 2 elements (like the partner's own
# implementation, this re-implementation does not special-case shorter
# input -- real runs are always thousands of windows), and two IDENTICAL
# samples keep the rpm/vane slew at exactly zero so only the static
# threshold under test is exercised.


def test_partner_mode_a_raw_rpm_exactly_at_standstill_threshold_is_not_standstill() -> None:
    rpm = np.array([10.0, 10.0])  # strict `< 10`
    power = np.array([0.0, 0.0])
    vane = np.array([0.0, 0.0])

    result = cpl.partner_mode_a_raw(rpm, power, vane)

    # not standstill, not pump, rpm not > 0 is false here (10 > 0 is True) and
    # vane (0.0) is not > 5.0, so this falls to the PH branch.
    assert list(result) == ["PH", "PH"]


def test_partner_mode_a_raw_pump_below_negative_power_threshold_regardless_of_rpm_sign() -> None:
    # Partner's own branch order checks power BEFORE rpm sign (only standstill is
    # checked first) -- a positive (turbine-direction) rpm with power < -20 MW is
    # still "PU", faithfully reproducing the original's literal order.
    rpm = np.array([300.0, 300.0])
    power = np.array([-25.0, -25.0])
    vane = np.array([10.0, 10.0])

    result = cpl.partner_mode_a_raw(rpm, power, vane)

    assert list(result) == ["PU", "PU"]


def test_partner_mode_a_raw_turbine_full_load_above_vane_threshold() -> None:
    rpm = np.array([300.0, 300.0])
    power = np.array([100.0, 100.0])
    vane = np.array([60.0, 60.0])  # > 55.0

    result = cpl.partner_mode_a_raw(rpm, power, vane)

    assert list(result) == ["TU_full", "TU_full"]


def test_partner_mode_a_raw_vane_exactly_at_full_load_threshold_is_partial() -> None:
    rpm = np.array([300.0, 300.0])
    power = np.array([100.0, 100.0])
    vane = np.array([55.0, 55.0])  # strict `> 55.0`, so 55.0 exactly is TU_partial

    result = cpl.partner_mode_a_raw(rpm, power, vane)

    assert list(result) == ["TU_partial", "TU_partial"]


def test_partner_mode_a_raw_turbine_partial_load_between_vane_thresholds() -> None:
    rpm = np.array([300.0, 300.0])
    power = np.array([50.0, 50.0])
    vane = np.array([30.0, 30.0])  # 5.0 < 30.0 <= 55.0

    result = cpl.partner_mode_a_raw(rpm, power, vane)

    assert list(result) == ["TU_partial", "TU_partial"]


def test_partner_mode_a_raw_vane_exactly_at_partial_load_threshold_is_phase_shifter() -> None:
    rpm = np.array([300.0, 300.0])
    power = np.array([0.0, 0.0])
    vane = np.array([5.0, 5.0])  # strict `> 5.0`, so 5.0 exactly does not qualify TU_partial

    result = cpl.partner_mode_a_raw(rpm, power, vane)

    assert list(result) == ["PH", "PH"]


def test_partner_mode_a_raw_negative_rpm_not_standstill_not_pump_falls_to_transition() -> None:
    # rpm magnitude above the standstill threshold but negative (pump-direction
    # ramp), power not negative enough for "PU": none of the rpm>0 branches fire,
    # so the original's final `else` branch applies.
    rpm = np.array([-15.0, -15.0])
    power = np.array([0.0, 0.0])
    vane = np.array([50.0, 50.0])

    result = cpl.partner_mode_a_raw(rpm, power, vane)

    assert list(result) == ["transition", "transition"]


def test_partner_mode_a_raw_rpm_slew_overrides_static_classification() -> None:
    # rpm = [300, 300, 450]: np.gradient (spacing=1) = [0, 75, 150]. Threshold
    # 50 rpm/s -> indices 1 and 2 are forced to "transition" regardless of their
    # static class (both would otherwise be "TU_full": rpm>0, vane=60>55).
    rpm = np.array([300.0, 300.0, 450.0])
    power = np.array([100.0, 100.0, 100.0])
    vane = np.array([60.0, 60.0, 60.0])

    result = cpl.partner_mode_a_raw(rpm, power, vane, sample_rate_hz=1.0)

    assert list(result) == ["TU_full", "transition", "transition"]


def test_partner_mode_a_raw_vane_slew_overrides_static_classification() -> None:
    # vane = [60, 60, 90]: np.gradient = [0, 15, 30]. Threshold 5 %pt/s ->
    # indices 1 and 2 exceed it and are forced to "transition".
    rpm = np.array([300.0, 300.0, 300.0])
    power = np.array([100.0, 100.0, 100.0])
    vane = np.array([60.0, 60.0, 90.0])

    result = cpl.partner_mode_a_raw(rpm, power, vane, sample_rate_hz=1.0)

    assert list(result) == ["TU_full", "transition", "transition"]


def test_partner_mode_a_raw_nan_input_yields_unknown_and_does_not_leak_into_neighbors() -> None:
    rpm = np.array([300.0, np.nan, 300.0])
    power = np.array([100.0, 100.0, 100.0])
    vane = np.array([60.0, 60.0, 60.0])

    result = cpl.partner_mode_a_raw(rpm, power, vane, sample_rate_hz=1.0)

    assert list(result) == ["TU_full", "unknown", "TU_full"]


def test_partner_mode_a_raw_shape_mismatch_raises_value_error() -> None:
    with pytest.raises(ValueError, match="shape"):
        cpl.partner_mode_a_raw(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0])
        )


def test_partner_mode_a_collapses_tu_full_and_tu_partial_to_turbine() -> None:
    # TU_full/TU_partial both collapse to "turbine" -- our own GT does not
    # distinguish load bands within "turbine" (rowii.scada.labels.STATES).
    # Two separate constant-value calls (rather than one concatenated array)
    # so no unrealistic same-window jump between regimes trips the (separately
    # tested) slew-based transition override.
    full_load = cpl.partner_mode_a(
        np.array([300.0, 300.0]), np.array([100.0, 100.0]), np.array([60.0, 60.0]),
    )
    partial_load = cpl.partner_mode_a(
        np.array([300.0, 300.0]), np.array([50.0, 50.0]), np.array([30.0, 30.0]),
    )

    assert list(full_load) == ["turbine", "turbine"]
    assert list(partial_load) == ["turbine", "turbine"]


def test_partner_mode_a_maps_standstill_pump_phase_shifter_and_unknown() -> None:
    standstill = cpl.partner_mode_a(
        np.array([5.0, 5.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]),
    )
    pump = cpl.partner_mode_a(
        np.array([300.0, 300.0]), np.array([-25.0, -25.0]), np.array([10.0, 10.0]),
    )
    phase_shifter = cpl.partner_mode_a(
        np.array([300.0, 300.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]),
    )
    unknown = cpl.partner_mode_a(
        np.array([np.nan, np.nan]), np.array([0.0, 0.0]), np.array([0.0, 0.0]),
    )

    assert list(standstill) == ["standstill", "standstill"]
    assert list(pump) == ["pump", "pump"]
    assert list(phase_shifter) == ["phase-shifter", "phase-shifter"]
    assert list(unknown) == ["unknown", "unknown"]


# ---------------------------------------------------------------------------
# _distance_to_change_s -- boundary-distance helper for the disagreement
# analysis ("do disagreements cluster near OUR GT's own transition zones?").
# ---------------------------------------------------------------------------


def test_distance_to_change_s_single_change_point() -> None:
    state = np.array(["a", "a", "a", "b", "b"])

    result = cpl._distance_to_change_s(state, window_s=1.0)

    assert list(result) == [3.0, 2.0, 1.0, 0.0, 1.0]


def test_distance_to_change_s_no_changes_returns_inf() -> None:
    state = np.array(["a", "a", "a"])

    result = cpl._distance_to_change_s(state, window_s=1.0)

    assert list(result) == [np.inf, np.inf, np.inf]


def test_distance_to_change_s_scales_with_window_s() -> None:
    state = np.array(["a", "a", "a", "b", "b"])

    result = cpl._distance_to_change_s(state, window_s=5.0)

    assert list(result) == [15.0, 10.0, 5.0, 0.0, 5.0]


def test_distance_to_change_s_multiple_changes_picks_nearest() -> None:
    state = np.array(["a", "b", "a", "a", "a", "c"])

    result = cpl._distance_to_change_s(state, window_s=1.0)

    assert list(result) == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# Real-data driver smoke test.
# ---------------------------------------------------------------------------

pytestmark_data = pytest.mark.data


@pytest.mark.data
def test_compare_run_smoke_on_a_small_real_run() -> None:
    if not _HAS_DATA_ROOT:
        pytest.skip("ROWII_DATA_ROOT is unset or does not point at an existing directory")
    cfg = load_config()
    index = discover(cfg.data_root)

    df = cpl.compare_run("080726-st_strikes", cfg, index)

    assert df is not None
    assert len(df) > 0
    assert set(df["run"]) == {"080726-st_strikes"}
    for col in (
        "window_idx", "t_utc", "rpm", "power_mw", "vane_pct",
        "our_state", "our_load_bin", "partner_b_state", "known_both", "agree",
        "dist_to_our_change_s",
    ):
        assert col in df.columns, f"missing column {col!r}"
    assert df["our_state"].isin([*cpl.STATE_ORDER]).all()
    assert df["partner_b_state"].isin([*cpl.STATE_ORDER]).all()
