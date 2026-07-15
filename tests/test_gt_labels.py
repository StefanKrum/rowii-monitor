import dataclasses
import logging
from datetime import UTC, datetime

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
    "1_Drehzahl UPM",
    "1_Leitapparat Stell.",
    "Durchfluss TU",
    "Durchfluss PU",
    "1_Q_Ist",
    "1_KS Stellung",
]


def test_gt_channels_constant_matches_the_seven_real_betriebsdaten_channel_names() -> None:
    # Task 13b real-data finding: "1_Drehzahl_Ist" is NOT rpm -- it is a
    # percent-of-nominal-ish quantity (measured ratio ~3.75x smaller than the true rpm
    # channel on the same file). "1_Drehzahl UPM" ("Umdrehungen Pro Minute" = rpm) is
    # the genuine rpm channel and is what GtRules.speed_nominal_rpm must be compared
    # against. "reactive"/"ks_valve" added by the multi-day/phase-shifter addendum
    # (spec §3) -- both verified present (real names, exact spelling) in every
    # SCADA-bearing day's Betriebsdaten during Task 13b/addendum work.
    assert dict(GT_CHANNELS) == {
        "power": "1_P_Ist",
        "speed": "1_Drehzahl UPM",
        "guide_vane": "1_Leitapparat Stell.",
        "flow_tu": "Durchfluss TU",
        "flow_pu": "Durchfluss PU",
        "reactive": "1_Q_Ist",
        "ks_valve": "1_KS Stellung",
    }


def test_states_constant_matches_the_five_documented_states() -> None:
    assert STATES == ("standstill", "turbine", "pump", "transition", "phase-shifter")


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
    reactive = np.full(n_samples, 20.0, dtype=np.float32)
    ks_valve = np.full(n_samples, 3.0, dtype=np.float32)
    data = np.stack([power, speed, guide_vane, flow_tu, flow_pu, reactive, ks_valve], axis=1)
    p = build_gantner_file(
        tmp_path / "bd.dat", GT_CHANNEL_NAMES, data, t0_ns=0, rate_hz=rate_hz
    )
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=2)

    scada = load_scada_window_means([p], grid)

    assert list(scada.columns) == [
        "power", "speed", "guide_vane", "flow_tu", "flow_pu", "reactive", "ks_valve",
    ]
    assert list(scada.index) == [0, 1]
    assert scada.loc[0, "power"] == pytest.approx(4.5)
    assert scada.loc[1, "power"] == pytest.approx(100.0)
    assert scada.loc[0, "speed"] == pytest.approx(375.0)
    assert scada.loc[0, "guide_vane"] == pytest.approx(50.0)
    assert scada.loc[0, "flow_tu"] == pytest.approx(5.0)
    assert scada.loc[0, "flow_pu"] == pytest.approx(0.0)
    assert scada.loc[0, "reactive"] == pytest.approx(20.0)
    assert scada.loc[0, "ks_valve"] == pytest.approx(3.0)


def test_load_scada_window_means_concatenates_hourly_files_in_order(tmp_path) -> None:
    # Two hourly files, back to back in time; a window straddling neither file alone should
    # still average correctly once concatenated (values differ per file so a wrong file order
    # or missing concat would be caught by the exact mean).
    rate_hz = 10.0
    window_ns = 1_000_000_000
    data1 = np.zeros((10, 7), dtype=np.float32)
    data1[:, 0] = 1.0  # power = 1 for the whole first file (t = [0, 1s))
    data2 = np.zeros((10, 7), dtype=np.float32)
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
    data = np.ones((10, 7), dtype=np.float32) * 7.0
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
    # "1_Leitapparat Stell." (guide_vane) is the FIRST GT_CHANNELS entry missing from
    # this file, in GT_CHANNELS' own key order (power, speed, guide_vane, flow_tu,
    # flow_pu, reactive, ks_valve) -- "power" and "speed" are both present under their
    # real names, so the KeyError must specifically name guide_vane, not fire early on
    # an unrelated missing channel (this file is also missing flow_tu/flow_pu/reactive/
    # ks_valve, but guide_vane's the one that raises first).
    rate_hz = 10.0
    data = np.zeros((10, 3), dtype=np.float32)
    other_names = ["1_P_Ist", "1_Drehzahl UPM", "SomeOtherChannel"]
    p = build_gantner_file(tmp_path / "bd.dat", other_names, data, t0_ns=0, rate_hz=rate_hz)
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=1)

    with pytest.raises(KeyError) as exc_info:
        load_scada_window_means([p], grid)

    message = str(exc_info.value)
    assert "1_Leitapparat Stell." in message
    for name in other_names:
        assert name in message


def test_gt_channel_names_with_spaces_and_dots_round_trip_through_gantner_reader(tmp_path) -> None:
    # Regression coverage for the reader's name tokenizer: two of the seven real GT channel
    # names contain a space ("1_Leitapparat Stell.", "1_KS Stellung") -- confirm the
    # fixture/reader pair preserves them exactly end to end (build -> read -> select by
    # exact name).
    rate_hz = 10.0
    data = np.arange(70, dtype=np.float32).reshape(10, 7)
    p = build_gantner_file(tmp_path / "bd.dat", GT_CHANNEL_NAMES, data, t0_ns=0, rate_hz=rate_hz)
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=1)

    scada = load_scada_window_means([p], grid)

    assert not scada.isna().any().any()


# ---------------------------------------------------------------------------
# load_scada_window_means: DAQ epoch-2000 clock quirk (Task 10, D3) -- the raw
# Betriebsdaten timestamps carry the SAME quirk as burst files (module docstring
# of rowii.io.dataset); load_scada_window_means must shift them onto true UTC
# before slicing against a (by then already true-UTC, D2) grid.
# ---------------------------------------------------------------------------


def _naive_unix_decode_ns(dt: datetime) -> int:
    """Nanoseconds since the Unix epoch that *dt* decodes to when its own digits are
    read naively AS IF it already were a Unix timestamp -- mirrors `tests.
    test_dataset._naive_unix_decode_ns` (duplicated, not imported -- each test
    module builds its own fixtures, matching this codebase's "no cross-script
    dependency" convention extended to tests)."""
    return int((dt - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds()) * 10**9


# CEST (UTC+2) worked example, identical constant to tests/test_dataset.py's own
# (task-10-brief.md's numeric example): local hour 06 on 2026-06-27 -> true UTC
# 04:00:00Z; offset = 946_684_800 s epoch-2000 shift - 7_200 s CEST = 946_677_600 s.
_CEST_OFFSET_NS = 946_677_600 * 10**9


def test_load_scada_window_means_shifts_raw_scada_axis_onto_true_utc(tmp_path) -> None:
    # Betriebsdaten filename encodes local hour 06 (CEST) -> true UTC start
    # 2026-06-27T04:00:00Z; the file's own raw header.t0_ns instead carries the
    # QUIRKY axis (decodes naively to 1996-06-27T06:00:00Z). A grid built on the
    # TRUE UTC axis (as `rowii.pipeline.build_run_grid` now produces, D2) must still
    # capture every sample once load_scada_window_means derives and applies the
    # SCADA file's own offset -- without the fix, the window would see zero samples
    # (raw ts ~30 years earlier than the grid) and come out all-NaN instead.
    rate_hz = 10.0
    data = np.full((10, 7), 7.0, dtype=np.float32)
    raw_t0_ns = _naive_unix_decode_ns(datetime(1996, 6, 27, 6, 0, 0, tzinfo=UTC))
    p = build_gantner_file(
        tmp_path / "2026-06-27_06-00-00.dat", GT_CHANNEL_NAMES, data,
        t0_ns=raw_t0_ns, rate_hz=rate_hz,
    )
    true_utc_t0_ns = _naive_unix_decode_ns(datetime(2026, 6, 27, 4, 0, 0, tzinfo=UTC))
    grid = WindowGrid(t0_ns=true_utc_t0_ns, window_ns=1_000_000_000, n_windows=1)

    scada = load_scada_window_means([p], grid)

    assert scada.loc[0, "power"] == pytest.approx(7.0)
    assert not scada.isna().any().any()


def test_load_scada_window_means_files_not_matching_pattern_stay_unshifted(tmp_path) -> None:
    # Backward-compat guard: every OTHER test in this file (and elsewhere) uses
    # arbitrarily-named files ("bd.dat", "h1.dat", ...) that were never meant to
    # model the DAQ-clock quirk -- load_scada_window_means must still align them
    # against a raw (unshifted) grid exactly as before (offset 0, the existing
    # behaviour), not silently reinterpret their timestamps.
    rate_hz = 10.0
    data = np.full((10, 7), 7.0, dtype=np.float32)
    p = build_gantner_file(tmp_path / "bd.dat", GT_CHANNEL_NAMES, data, t0_ns=0, rate_hz=rate_hz)
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=1)

    scada = load_scada_window_means([p], grid)

    assert scada.loc[0, "power"] == pytest.approx(7.0)


def test_load_scada_window_means_warns_when_audio_offset_disagrees(tmp_path, caplog) -> None:
    # The SCADA-derived offset must never blindly copy the audio run's own offset
    # (D3) -- but the two ARE expected to agree closely for the same day, so a
    # caller that has both must be warned when they do not. Winter-vs-summer
    # offsets (946_681_200 s vs 946_677_600 s, a 3600 s disagreement) stand in for
    # "something is wrong" here.
    rate_hz = 10.0
    data = np.full((10, 7), 7.0, dtype=np.float32)
    raw_t0_ns = _naive_unix_decode_ns(datetime(1996, 6, 27, 6, 0, 0, tzinfo=UTC))
    p = build_gantner_file(
        tmp_path / "2026-06-27_06-00-00.dat", GT_CHANNEL_NAMES, data,
        t0_ns=raw_t0_ns, rate_hz=rate_hz,
    )
    true_utc_t0_ns = _naive_unix_decode_ns(datetime(2026, 6, 27, 4, 0, 0, tzinfo=UTC))
    grid = WindowGrid(t0_ns=true_utc_t0_ns, window_ns=1_000_000_000, n_windows=1)
    winter_offset_ns = 946_681_200 * 10**9

    with caplog.at_level(logging.WARNING):
        load_scada_window_means([p], grid, audio_run_offset_ns=winter_offset_ns)

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("disagree" in w.lower() for w in warnings), warnings


def test_load_scada_window_means_no_warning_when_audio_offset_agrees(tmp_path, caplog) -> None:
    rate_hz = 10.0
    data = np.full((10, 7), 7.0, dtype=np.float32)
    raw_t0_ns = _naive_unix_decode_ns(datetime(1996, 6, 27, 6, 0, 0, tzinfo=UTC))
    p = build_gantner_file(
        tmp_path / "2026-06-27_06-00-00.dat", GT_CHANNEL_NAMES, data,
        t0_ns=raw_t0_ns, rate_hz=rate_hz,
    )
    true_utc_t0_ns = _naive_unix_decode_ns(datetime(2026, 6, 27, 4, 0, 0, tzinfo=UTC))
    grid = WindowGrid(t0_ns=true_utc_t0_ns, window_ns=1_000_000_000, n_windows=1)

    with caplog.at_level(logging.WARNING):
        load_scada_window_means([p], grid, audio_run_offset_ns=_CEST_OFFSET_NS)

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("disagree" in w.lower() for w in warnings), warnings


# ---------------------------------------------------------------------------
# gt_labels
# ---------------------------------------------------------------------------

RULES = GtRules()  # nominal=378.832 (measured, Task 13b), speed_eps_frac=0.05, power_eps_mw=2.0,
# ramp=1.0 MW/s, transition_buffer_s=10.0, n_load_bins=3
WINDOW_S = 5.0  # matches the scenarios' hand-derivation (buffer=10s -> 2-window radius)


def _scada(
    power, speed, guide_vane=None, flow_tu=None, flow_pu=None, reactive=None, ks_valve=None
) -> pd.DataFrame:
    n = len(power)
    return pd.DataFrame(
        {
            "power": power,
            "speed": speed,
            "guide_vane": guide_vane if guide_vane is not None else [0.0] * n,
            "flow_tu": flow_tu if flow_tu is not None else [0.0] * n,
            "flow_pu": flow_pu if flow_pu is not None else [0.0] * n,
            "reactive": reactive if reactive is not None else [0.0] * n,
            "ks_valve": ks_valve if ks_valve is not None else [np.nan] * n,
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
    # Task 13/13b real-data finding (Rodundwerk II, 2026-06-25 09:00 pump run):
    # "1_Drehzahl UPM" (the genuine rpm channel -- see GT_CHANNELS) is SIGNED by
    # rotation direction at this plant -- positive during turbine operation, negative
    # during pump operation (a reversible pump-turbine spins the opposite way in each
    # mode). The base-state "nominal speed" check must use |speed|, exactly like the
    # standstill check already does two lines above it in _base_state -- a
    # negative-but-nominal-magnitude speed must not silently fail the nominal-speed
    # gate and fall through to "transition" for the entire pump run (the bug this
    # test guards against: is_nominal previously compared signed speed directly
    # against a positive threshold, so it was never true when speed < 0).
    #
    # Uses an explicit speed_nominal_rpm (independent of RULES' own default) so this test's
    # pass/fail is decoupled from whatever nominal-speed value GtRules ships with -- the
    # magnitude/sign behaviour under test is orthogonal to that number.
    rules = dataclasses.replace(RULES, speed_nominal_rpm=378.832)
    power = [-281.0, -281.0, -281.0]
    speed = [-377.9, -377.9, -377.9]  # measured PU-morning UPM plateau magnitude, negated
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
    # Speeds derived from RULES.speed_nominal_rpm (not a hardcoded magic number) so this test
    # keeps exercising "clearly below the standstill epsilon" regardless of what
    # speed_nominal_rpm is currently set to (Task 13 changed it from an unverified 375 to a
    # measured-but-wrong-channel 101.0; Task 13b corrected the channel itself to "1_Drehzahl
    # UPM" and remeasured the true value as 378.832 -- a hardcoded sub-threshold value here
    # would have silently stopped being sub-threshold and turned this into a false green
    # across either transition).
    standstill_eps = RULES.speed_eps_frac * RULES.speed_nominal_rpm
    power = [0.5, -1.5, 0.0]
    speed = [0.2 * standstill_eps, -0.4 * standstill_eps, 0.8 * standstill_eps]
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
    data = np.zeros((10, 7), dtype=np.float32)
    p = build_gantner_file(tmp_path / "bd.dat", GT_CHANNEL_NAMES, data, t0_ns=0, rate_hz=10.0)
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=1)
    try:
        load_scada_window_means([p], grid)
    except GantnerFormatError:
        pytest.fail("load_scada_window_means raised GantnerFormatError on a well-formed file")


# ---------------------------------------------------------------------------
# Phase-shifter ground truth (addendum spec §3)
# ---------------------------------------------------------------------------

# 1-minute windows here (not the 5.0s WINDOW_S above) so the real
# ph_min_dwell_s=600.0 default and 15-/5-min test scenarios stay hand-writable
# (10 windows for the dwell threshold, 15/5 windows for the two scenarios)
# without artificially shrinking the dwell threshold itself.
PH_WINDOW_S = 60.0
# transition_buffer_s=60.0 (not RULES' own 10.0 default) so the buffer radius
# is exactly 1 whole window at this window_s -- RULES' own 10.0s default would
# round to 0 windows here (round(10.0/60.0) == 0), producing no buffer edge at
# all and defeating the "transition edges" half of these scenarios.
PH_RULES = dataclasses.replace(RULES, transition_buffer_s=60.0)


def test_gt_labels_15min_unloaded_spinning_run_becomes_phase_shifter_with_transition_edges() -> (
    None
):
    # 2 standstill -> 15 nominal-speed/unloaded (candidate PH run, well over the
    # 10-window dwell threshold at window_s=60.0) -> 2 standstill. The transition
    # buffer's existing, unchanged, SYMMETRIC radius (1 window at
    # transition_buffer_s=60.0 -> round(60.0/60.0)=1) touches 1 window on EACH SIDE
    # of the standstill<->phase-shifter change point (matching the pre-existing
    # buffer contract already exercised by
    # test_gt_labels_abrupt_state_change_gets_buffer_transition_without_ramp) --
    # index 1 (the standstill window immediately before the change) and index 2
    # (the PH run's own first window) both become "transition"; symmetrically at
    # the far end. The 13 windows strictly inside those buffered edges stay
    # "phase-shifter".
    n_nom = RULES.speed_nominal_rpm
    power = [0.0] * 2 + [0.0] * 15 + [0.0] * 2
    speed = [0.0] * 2 + [n_nom] * 15 + [0.0] * 2
    scada = _scada(power, speed)

    result = gt_labels(scada, PH_RULES, window_s=PH_WINDOW_S)

    expected_state = (
        ["standstill", "transition", "transition"]
        + ["phase-shifter"] * 13
        + ["transition", "transition", "standstill"]
    )
    assert list(result["state"]) == expected_state
    assert set(result["state"]).issubset(set(STATES))


def test_gt_labels_5min_unloaded_spinning_run_stays_transition() -> None:
    # Same shape as the 15-min scenario but only 5 candidate windows -- well
    # under the 10-window (600s / 60s) dwell threshold, so the whole run must
    # remain "transition" (never promoted to phase-shifter).
    n_nom = RULES.speed_nominal_rpm
    power = [0.0] * 2 + [0.0] * 5 + [0.0] * 2
    speed = [0.0] * 2 + [n_nom] * 5 + [0.0] * 2
    scada = _scada(power, speed)

    result = gt_labels(scada, PH_RULES, window_s=PH_WINDOW_S)

    assert list(result["state"]) == (
        ["standstill", "standstill"] + ["transition"] * 5 + ["standstill", "standstill"]
    )
    assert "phase-shifter" not in set(result["state"])


def test_gt_labels_ph_promotion_disabled_ks_gate_ignores_ks_value() -> None:
    # ph_requires_ks_closed=False (explicitly disabled here -- the shipped default
    # flipped to True on 2026-07-08 once verify_parameters.py confirmed the
    # separation on real 2026-07-01 data, see GtRules.ph_requires_ks_closed's own
    # docstring): with the gate off, a qualifying 15-window run is promoted to
    # phase-shifter regardless of the KS valve reading, even when it reads fully
    # OPEN (well above ks_closed_max) throughout.
    rules = dataclasses.replace(PH_RULES, ph_requires_ks_closed=False)
    n_nom = RULES.speed_nominal_rpm
    n = 15
    power = [0.0] * n
    speed = [n_nom] * n
    ks_open = [rules.ks_closed_max * 10.0] * n  # far above the closed threshold
    scada = _scada(power, speed, ks_valve=ks_open)
    assert rules.ph_requires_ks_closed is False

    result = gt_labels(scada, rules, window_s=PH_WINDOW_S)

    assert set(result["state"]) == {"phase-shifter"}


def test_gt_labels_ph_promotion_enabled_ks_gate_blocks_open_valve() -> None:
    # ph_requires_ks_closed=True + KS reading OPEN throughout (above
    # ks_closed_max): the speed/power/dwell criteria alone are satisfied, but
    # the conjunctive KS gate must block promotion -- the run stays
    # "transition" exactly as the under-dwell scenario does.
    rules = dataclasses.replace(PH_RULES, ph_requires_ks_closed=True)
    n_nom = RULES.speed_nominal_rpm
    n = 15
    power = [0.0] * n
    speed = [n_nom] * n
    ks_open = [rules.ks_closed_max * 10.0] * n

    scada = _scada(power, speed, ks_valve=ks_open)
    result = gt_labels(scada, rules, window_s=PH_WINDOW_S)

    assert set(result["state"]) == {"transition"}


def test_gt_labels_ph_promotion_enabled_ks_gate_allows_closed_valve() -> None:
    # ph_requires_ks_closed=True + KS reading CLOSED throughout (at/under
    # ks_closed_max): the conjunctive gate is satisfied, so the run is promoted
    # exactly as in the gate-disabled case.
    rules = dataclasses.replace(PH_RULES, ph_requires_ks_closed=True)
    n_nom = RULES.speed_nominal_rpm
    n = 15
    power = [0.0] * n
    speed = [n_nom] * n
    ks_closed = [rules.ks_closed_max * 0.5] * n

    scada = _scada(power, speed, ks_valve=ks_closed)
    result = gt_labels(scada, rules, window_s=PH_WINDOW_S)

    assert set(result["state"]) == {"phase-shifter"}


def test_gt_labels_ph_promotion_enabled_nan_ks_falls_back_to_promoting_with_warning(
    caplog,
) -> None:
    # ph_requires_ks_closed=True but the KS channel is entirely NaN for this
    # run (e.g. a day tree without that channel populated) -- the gate cannot
    # be evaluated and must be IGNORED (not silently treated as "closed" nor
    # as "open"): the run is promoted purely on speed/power/dwell, same as the
    # gate-disabled case, but a warning documents that the gate was skipped.
    rules = dataclasses.replace(PH_RULES, ph_requires_ks_closed=True)
    n_nom = RULES.speed_nominal_rpm
    n = 15
    power = [0.0] * n
    speed = [n_nom] * n
    ks_nan = [np.nan] * n

    scada = _scada(power, speed, ks_valve=ks_nan)

    with caplog.at_level(logging.WARNING):
        result = gt_labels(scada, rules, window_s=PH_WINDOW_S)

    assert set(result["state"]) == {"phase-shifter"}
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("ks" in msg.lower() for msg in warnings), (
        f"expected a KS-gate warning, got: {warnings}"
    )


def test_gt_labels_ph_promotion_uses_dedicated_ph_power_eps_not_the_general_one() -> None:
    # Real-data finding (2026-07-01 TU_PH_TU day, ~98 min of confirmed phase-shifter
    # operation measured directly): active power sits at a stable ~-3.5 MW median
    # (range -4.3 to -2.9 MW) during genuine PH operation -- well OUTSIDE the general
    # power_eps_mw=2.0 threshold (calibrated for standstill/loaded turbine-pump
    # discrimination, not PH idling losses). ph_power_eps_mw is a DEDICATED,
    # separately-configurable threshold specifically for the PH candidate check, so
    # a run at this realistic ~3.5 MW magnitude is still promoted while the general
    # power_eps_mw (used everywhere else in _base_state) stays untouched.
    n_nom = RULES.speed_nominal_rpm
    n = 15
    power = [-3.5] * n  # exceeds RULES.power_eps_mw (2.0) but not ph_power_eps_mw
    speed = [n_nom] * n
    scada = _scada(power, speed)
    assert abs(power[0]) > RULES.power_eps_mw
    assert abs(power[0]) <= PH_RULES.ph_power_eps_mw

    result = gt_labels(scada, PH_RULES, window_s=PH_WINDOW_S)

    assert set(result["state"]) == {"phase-shifter"}


def test_gt_labels_ramp_rule_never_demotes_phase_shifter_interior() -> None:
    # A tiny power blip inside an otherwise-qualifying PH run (dP/dt exceeding
    # ramp_mw_per_s at one interior window) must NOT flip that window back to
    # "transition" -- the ramp rule must explicitly skip windows the PH
    # promotion already claimed, per the spec's stage-ordering requirement
    # ("ramp rule ... never demotes PH interiors").
    n_nom = RULES.speed_nominal_rpm
    n = 15
    power = [0.0] * n
    # A brief, single-window power blip (still within power_eps_mw so the base
    # state stays a PH candidate) large enough to trip the ramp rule via its
    # centered-difference neighbors -- windows 6,7,8 form a 0 -> ~1.9 -> 0
    # spike whose centered dP/dt at window 7 is nonzero relative to its
    # immediate neighbors, but the RAW values everywhere stay under
    # power_eps_mw=2.0 so the whole 15-window run is still a valid PH
    # candidate at the base-state stage.
    blip_idx = 7
    power[blip_idx] = 1.9
    speed = [n_nom] * n
    scada = _scada(power, speed)

    result = gt_labels(scada, PH_RULES, window_s=PH_WINDOW_S)

    assert set(result["state"]) == {"phase-shifter"}, (
        f"expected the whole run to stay phase-shifter despite the interior power "
        f"blip, got: {list(result['state'])}"
    )
