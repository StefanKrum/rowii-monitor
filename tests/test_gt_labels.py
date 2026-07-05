import dataclasses

import numpy as np
import pandas as pd
import pytest

from rowii.config import GtRules
from rowii.io.gantner import GantnerFormatError, read_gantner
from rowii.scada.labels import GT_CHANNELS, STATES, gt_labels, load_scada_window_means
from rowii.signals.windows import WindowGrid
from tests.fixtures.gantner_builder import build_gantner_file

GT_CHANNEL_NAMES = [
    "1_P_Ist",
    "1_Drehzahl_Ist",
    "1_Leitapparat Stell.",
    "Durchfluss TU",
    "Durchfluss PU",
]


def test_gt_channels_constant_matches_the_five_real_betriebsdaten_channel_names() -> None:
    assert dict(GT_CHANNELS) == {
        "power": "1_P_Ist",
        "speed": "1_Drehzahl_Ist",
        "guide_vane": "1_Leitapparat Stell.",
        "flow_tu": "Durchfluss TU",
        "flow_pu": "Durchfluss PU",
    }


def test_states_constant_matches_the_four_documented_states() -> None:
    assert STATES == ("standstill", "turbine", "pump", "transition")


# ---------------------------------------------------------------------------
# load_scada_window_means
# ---------------------------------------------------------------------------


def test_load_scada_window_means_computes_per_window_column_means(tmp_path) -> None:
    # 10 Hz, 2 windows of 1s -> 10 samples/window. Window 0: power ramps 0..9 (mean 4.5);
    # window 1: constant power=100. Other GT channels held constant for simplicity.
    rate_hz = 10.0
    n_samples = 20
    power = np.concatenate([np.arange(10, dtype=np.float32), np.full(10, 100.0, dtype=np.float32)])
    speed = np.full(n_samples, 375.0, dtype=np.float32)
    guide_vane = np.full(n_samples, 50.0, dtype=np.float32)
    flow_tu = np.full(n_samples, 5.0, dtype=np.float32)
    flow_pu = np.full(n_samples, 0.0, dtype=np.float32)
    data = np.stack([power, speed, guide_vane, flow_tu, flow_pu], axis=1)
    p = build_gantner_file(
        tmp_path / "bd.dat", GT_CHANNEL_NAMES, data, t0_ns=0, rate_hz=rate_hz
    )
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=2)

    scada = load_scada_window_means([p], grid)

    assert list(scada.columns) == ["power", "speed", "guide_vane", "flow_tu", "flow_pu"]
    assert list(scada.index) == [0, 1]
    assert scada.loc[0, "power"] == pytest.approx(4.5)
    assert scada.loc[1, "power"] == pytest.approx(100.0)
    assert scada.loc[0, "speed"] == pytest.approx(375.0)
    assert scada.loc[0, "guide_vane"] == pytest.approx(50.0)
    assert scada.loc[0, "flow_tu"] == pytest.approx(5.0)
    assert scada.loc[0, "flow_pu"] == pytest.approx(0.0)


def test_load_scada_window_means_concatenates_hourly_files_in_order(tmp_path) -> None:
    # Two hourly files, back to back in time; a window straddling neither file alone should
    # still average correctly once concatenated (values differ per file so a wrong file order
    # or missing concat would be caught by the exact mean).
    rate_hz = 10.0
    window_ns = 1_000_000_000
    data1 = np.zeros((10, 5), dtype=np.float32)
    data1[:, 0] = 1.0  # power = 1 for the whole first file (t = [0, 1s))
    data2 = np.zeros((10, 5), dtype=np.float32)
    data2[:, 0] = 3.0  # power = 3 for the whole second file (t = [1s, 2s))
    p1 = build_gantner_file(tmp_path / "h1.dat", GT_CHANNEL_NAMES, data1, t0_ns=0, rate_hz=rate_hz)
    p2 = build_gantner_file(
        tmp_path / "h2.dat", GT_CHANNEL_NAMES, data2, t0_ns=window_ns, rate_hz=rate_hz
    )
    grid = WindowGrid(t0_ns=0, window_ns=window_ns, n_windows=2)

    scada = load_scada_window_means([p1, p2], grid)

    assert scada.loc[0, "power"] == pytest.approx(1.0)
    assert scada.loc[1, "power"] == pytest.approx(3.0)


def test_load_scada_window_means_nan_for_window_with_no_scada_samples(tmp_path) -> None:
    # Grid spans 3 windows but the SCADA file only covers window 0 -> windows 1, 2 must be NaN
    # in every column (zero-coverage), not zero or dropped.
    rate_hz = 10.0
    window_ns = 1_000_000_000
    data = np.ones((10, 5), dtype=np.float32) * 7.0
    p = build_gantner_file(tmp_path / "bd.dat", GT_CHANNEL_NAMES, data, t0_ns=0, rate_hz=rate_hz)
    grid = WindowGrid(t0_ns=0, window_ns=window_ns, n_windows=3)

    scada = load_scada_window_means([p], grid)

    assert scada.loc[0, "power"] == pytest.approx(7.0)
    assert scada.loc[1].isna().all()
    assert scada.loc[2].isna().all()
    assert list(scada.index) == [0, 1, 2]


def test_load_scada_window_means_missing_channel_raises_key_error_listing_available(
    tmp_path,
) -> None:
    rate_hz = 10.0
    data = np.zeros((10, 3), dtype=np.float32)
    other_names = ["1_P_Ist", "1_Drehzahl_Ist", "SomeOtherChannel"]
    p = build_gantner_file(tmp_path / "bd.dat", other_names, data, t0_ns=0, rate_hz=rate_hz)
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=1)

    with pytest.raises(KeyError) as exc_info:
        load_scada_window_means([p], grid)

    message = str(exc_info.value)
    assert "1_Leitapparat Stell." in message
    for name in other_names:
        assert name in message


def test_gt_channel_names_with_spaces_and_dots_round_trip_through_gantner_reader(tmp_path) -> None:
    # Regression coverage for the reader's name tokenizer: two of the five real GT channel
    # names contain a space ("1_Leitapparat Stell.") -- confirm the fixture/reader pair
    # preserves them exactly end to end (build -> read -> select by exact name).
    rate_hz = 10.0
    data = np.arange(50, dtype=np.float32).reshape(10, 5)
    p = build_gantner_file(tmp_path / "bd.dat", GT_CHANNEL_NAMES, data, t0_ns=0, rate_hz=rate_hz)
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=1)

    scada = load_scada_window_means([p], grid)

    assert not scada.isna().any().any()


# ---------------------------------------------------------------------------
# gt_labels
# ---------------------------------------------------------------------------

RULES = GtRules()  # nominal=375.0, speed_eps_frac=0.05, power_eps_mw=2.0, ramp=1.0 MW/s,
# transition_buffer_s=10.0, n_load_bins=3
WINDOW_S = 5.0  # matches the scenarios' hand-derivation (buffer=10s -> 2-window radius)


def _scada(power, speed, guide_vane=None, flow_tu=None, flow_pu=None) -> pd.DataFrame:
    n = len(power)
    return pd.DataFrame(
        {
            "power": power,
            "speed": speed,
            "guide_vane": guide_vane if guide_vane is not None else [0.0] * n,
            "flow_tu": flow_tu if flow_tu is not None else [0.0] * n,
            "flow_pu": flow_pu if flow_pu is not None else [0.0] * n,
        }
    )


def test_gt_labels_full_walk_standstill_ramp_turbine_plateaus_ramp_standstill() -> None:
    # Window-by-window intent (window_s = 5.0s so transition_buffer_s=10.0s -> 2-window radius):
    #  0, 1: standstill (n=0, P=0)
    #  2:    ramp-up, sub-nominal speed (n=200, P=0)      -> transition (base rule)
    #  3:    ramp-up tail, nominal but unloaded (n=375, P=0) -> transition (spinning, unloaded)
    #  4-7:  turbine plateau A (n=375, P=10)               -> turbine, load_bin=0
    #  8-11: turbine plateau B (n=375, P=40)               -> turbine, load_bin=1 (distinct level)
    #  12:   ramp-down tail, nominal speed but unloaded (n=375, P=0) -> transition
    #  13:   ramp-down, sub-nominal speed (n=200, P=0)         -> transition (base rule)
    #  14,15: standstill (n=0, P=0)
    #
    # The abrupt power jump 10->40 between windows 7/8 (and 40->0 between windows 11/12)
    # produces |dP/dt| = 3.0 and 4.0 MW/s respectively (> ramp_mw_per_s=1.0), which flips
    # windows 7, 8, and 11 from their base state ("turbine") to "transition" via the ramp
    # rule -- this is what makes plateau A (windows 4-6, all P=10) and plateau B (windows
    # 9-10, all P=40) land in genuinely distinct load bins once the ramp-affected windows
    # are excluded from the per-state quantile binning.
    power = [0, 0, 0, 0, 10, 10, 10, 10, 40, 40, 40, 40, 0, 0, 0, 0]
    speed = [0, 0, 200, 375, 375, 375, 375, 375, 375, 375, 375, 375, 375, 200, 0, 0]
    scada = _scada(power, speed)

    result = gt_labels(scada, RULES, window_s=WINDOW_S)

    expected_state = [
        "standstill", "standstill",
        "transition", "transition",
        "turbine", "turbine", "turbine", "transition",
        "transition", "turbine", "turbine", "transition",
        "transition", "transition",
        "standstill", "standstill",
    ]
    assert list(result["state"]) == expected_state
    assert set(result["state"]).issubset(set(STATES))

    expected_load_bin = [-1, -1, -1, -1, 0, 0, 0, -1, -1, 1, 1, -1, -1, -1, -1, -1]
    assert list(result["load_bin"]) == expected_load_bin
    assert result["load_bin"].dtype == np.int64
    # The core "two distinct load levels" invariant the brief asks for: plateau A's bin(s)
    # and plateau B's bin(s) must be disjoint and ordered by power.
    plateau_a_bins = set(result.loc[[4, 5, 6], "load_bin"])
    plateau_b_bins = set(result.loc[[9, 10], "load_bin"])
    assert plateau_a_bins.isdisjoint(plateau_b_bins)
    assert max(plateau_a_bins) < min(plateau_b_bins)


def test_gt_labels_abrupt_state_change_gets_buffer_transition_without_ramp() -> None:
    # Turbine at a small, steady power (P=3, just above power_eps_mw=2.0) trips DIRECTLY to
    # standstill with no intervening window. dP/dt across the trip is only 0.3 MW/s (well
    # under ramp_mw_per_s=1.0), so the ramp rule alone would NOT flag anything here -- the
    # surrounding "transition" labels can only come from the buffer rule reacting to the
    # turbine->standstill base-state change.
    power = [3, 3, 3, 3, 0, 0, 0, 0]
    speed = [375, 375, 375, 375, 0, 0, 0, 0]
    scada = _scada(power, speed)

    result = gt_labels(scada, RULES, window_s=WINDOW_S)

    # Base-state change is at index 4 (turbine -> standstill); buffer radius = 2 windows
    # (10.0s / 5.0s) on each side -> indices [2, 3, 4, 5] become "transition".
    assert list(result["state"]) == [
        "turbine", "turbine",
        "transition", "transition", "transition", "transition",
        "standstill", "standstill",
    ]


def test_gt_labels_buffer_does_not_overwrite_unknown() -> None:
    # A genuine turbine->standstill change is detected at index 2. Its buffer radius (2
    # windows) reaches back to index 0, which is NaN (both power and speed) and must stay
    # "unknown" -- the buffer rule must skip it rather than relabeling it "transition".
    power = [np.nan, 3, 0, 0, 0]
    speed = [np.nan, 375, 0, 0, 0]
    scada = _scada(power, speed)

    result = gt_labels(scada, RULES, window_s=WINDOW_S)

    assert list(result["state"]) == [
        "unknown", "transition", "transition", "transition", "standstill",
    ]
    assert list(result["load_bin"]) == [-1, -1, -1, -1, -1]


def test_gt_labels_transition_buffer_shorter_than_half_a_window_adds_no_buffer() -> None:
    # transition_buffer_s=2.0 with window_s=5.0 -> round(2.0/5.0)=0 whole windows: the
    # configured buffer radius covers less than half a window, which genuinely resolves to
    # a zero-window buffer (not an arbitrary floor of 1) -- the abrupt trip's own base-rule
    # boundary is the only transition produced, with no buffer spread on either side.
    tight_rules = dataclasses.replace(RULES, transition_buffer_s=2.0)
    power = [3, 3, 3, 3, 0, 0, 0, 0]
    speed = [375, 375, 375, 375, 0, 0, 0, 0]
    scada = _scada(power, speed)

    result = gt_labels(scada, tight_rules, window_s=WINDOW_S)

    # No ramp trigger either (dP/dt magnitude is only 0.3 MW/s, see the sibling buffer-only
    # test), so with a zero-window buffer the base state passes through unchanged.
    assert list(result["state"]) == [
        "turbine", "turbine", "turbine", "turbine",
        "standstill", "standstill", "standstill", "standstill",
    ]


def test_gt_labels_pump_via_negative_power() -> None:
    power = [-5, -5, -5]
    speed = [375, 375, 375]
    scada = _scada(power, speed)

    result = gt_labels(scada, RULES, window_s=WINDOW_S)

    assert list(result["state"]) == ["pump", "pump", "pump"]
    assert list(result["load_bin"]) == [0, 0, 0]


def test_gt_labels_pump_mode_with_negative_speed_is_still_nominal() -> None:
    # Task 13 real-data finding (Rodundwerk II, 2026-06-25 09:00 pump run): 1_Drehzahl_Ist is
    # SIGNED by rotation direction at this plant -- positive during turbine operation, negative
    # during pump operation (a reversible pump-turbine spins the opposite way in each mode).
    # The base-state "nominal speed" check must use |speed|, exactly like the standstill check
    # already does two lines above it in _base_state -- a negative-but-nominal-magnitude speed
    # must not silently fail the nominal-speed gate and fall through to "transition" for the
    # entire pump run (the bug this test guards against: is_nominal previously compared signed
    # speed directly against a positive threshold, so it was never true when speed < 0).
    #
    # Uses an explicit speed_nominal_rpm (independent of RULES' own default) so this test's
    # pass/fail is decoupled from whatever nominal-speed value GtRules ships with -- the
    # magnitude/sign behaviour under test is orthogonal to that number.
    rules = dataclasses.replace(RULES, speed_nominal_rpm=101.0)
    power = [-281.0, -281.0, -281.0]
    speed = [-100.8, -100.8, -100.8]  # measured PU-morning plateau magnitude, negated
    scada = _scada(power, speed)

    result = gt_labels(scada, rules, window_s=WINDOW_S)

    assert list(result["state"]) == ["pump", "pump", "pump"]


def test_gt_labels_pump_via_flow_pu_dominance_with_positive_power() -> None:
    # Pump power may be logged POSITIVE at this plant -- that is exactly why flow dominance
    # (flow_pu > flow_tu at nominal speed) overrides the naive power-sign rule: without the
    # override this window would be misclassified "turbine" (P=+5 > power_eps_mw=2.0).
    power = [5, 5, 5]
    speed = [375, 375, 375]
    flow_tu = [2, 2, 2]
    flow_pu = [10, 10, 10]
    scada = _scada(power, speed, flow_tu=flow_tu, flow_pu=flow_pu)

    result = gt_labels(scada, RULES, window_s=WINDOW_S)

    assert list(result["state"]) == ["pump", "pump", "pump"]


def test_gt_labels_flow_tu_dominance_overrides_negative_power_to_turbine() -> None:
    # Mirror case: flow_tu dominance at nominal speed forces "turbine" even if power alone
    # (were it negative) would suggest "pump".
    power = [-5, -5, -5]
    speed = [375, 375, 375]
    flow_tu = [10, 10, 10]
    flow_pu = [2, 2, 2]
    scada = _scada(power, speed, flow_tu=flow_tu, flow_pu=flow_pu)

    result = gt_labels(scada, RULES, window_s=WINDOW_S)

    assert list(result["state"]) == ["turbine", "turbine", "turbine"]


def test_gt_labels_nan_power_or_speed_yields_unknown() -> None:
    power = [np.nan, 0.0, np.nan]
    speed = [375.0, np.nan, np.nan]
    scada = _scada(power, speed)

    result = gt_labels(scada, RULES, window_s=WINDOW_S)

    assert list(result["state"]) == ["unknown", "unknown", "unknown"]
    assert list(result["load_bin"]) == [-1, -1, -1]


def test_gt_labels_load_bin_degenerate_single_value_group_still_gets_bin_zero() -> None:
    # A turbine group where every remaining window has the EXACT same power cannot form real
    # quantile edges (pd.qcut on a single distinct value returns all-NaN internally) -- the
    # single homogeneous load level must still resolve to a valid bin (0), not NaN/-1.
    power = [3, 3]
    speed = [375, 375]
    scada = _scada(power, speed)

    result = gt_labels(scada, RULES, window_s=WINDOW_S)

    assert list(result["state"]) == ["turbine", "turbine"]
    assert list(result["load_bin"]) == [0, 0]
    assert result["load_bin"].dtype == np.int64


def test_gt_labels_standstill_base_rule() -> None:
    power = [0.5, -1.5, 0.0]
    speed = [10.0, -5.0, 18.0]  # all |speed| < 0.05*375 = 18.75
    scada = _scada(power, speed)

    result = gt_labels(scada, RULES, window_s=WINDOW_S)

    assert list(result["state"]) == ["standstill", "standstill", "standstill"]
    assert list(result["load_bin"]) == [-1, -1, -1]


def test_gt_labels_power_exactly_at_eps_boundary_is_not_turbine() -> None:
    # Brief's rule is a STRICT inequality (P > +P_eps); a window sitting exactly at the
    # threshold must not qualify as loaded turbine operation.
    scada = _scada([RULES.power_eps_mw], [375.0])

    result = gt_labels(scada, RULES, window_s=WINDOW_S)

    assert list(result["state"]) == ["transition"]


def test_gt_labels_speed_exactly_at_nominal_threshold_counts_as_nominal() -> None:
    # Brief's rule is an INCLUSIVE inequality (n >= 0.95*n_nom); a window sitting exactly at
    # the threshold must already count as running at nominal speed.
    threshold_rpm = (1.0 - RULES.speed_eps_frac) * RULES.speed_nominal_rpm
    scada = _scada([10.0], [threshold_rpm])

    result = gt_labels(scada, RULES, window_s=WINDOW_S)

    assert list(result["state"]) == ["turbine"]


def test_gt_labels_output_index_matches_input_index() -> None:
    scada = _scada([0, 0], [0, 0])
    scada.index = pd.Index([5, 9])

    result = gt_labels(scada, RULES, window_s=WINDOW_S)

    assert list(result.index) == [5, 9]


def test_gantner_name_tokenizer_handles_space_containing_channel_names(tmp_path) -> None:
    # Direct regression test on the reader itself (not just via load_scada_window_means):
    # "1_Leitapparat Stell." contains an internal space -- the reader's _NAME_RE must not
    # split or truncate it at the space.
    data = np.zeros((5, 1), dtype=np.float32)
    p = build_gantner_file(tmp_path / "t.dat", ["1_Leitapparat Stell."], data, rate_hz=10.0)

    f = read_gantner(p)

    assert f.header.channel_names == ["1_Leitapparat Stell."]


def test_load_scada_window_means_smoke_does_not_raise_gantner_format_error(tmp_path) -> None:
    # Guards against silently swallowing a real reader error inside load_scada_window_means.
    data = np.zeros((10, 5), dtype=np.float32)
    p = build_gantner_file(tmp_path / "bd.dat", GT_CHANNEL_NAMES, data, t0_ns=0, rate_hz=10.0)
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=1)
    try:
        load_scada_window_means([p], grid)
    except GantnerFormatError:
        pytest.fail("load_scada_window_means raised GantnerFormatError on a well-formed file")
