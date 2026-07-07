from pathlib import Path

from rowii.config import load_config


def test_defaults_without_env() -> None:
    cfg = load_config(env={})
    assert cfg.window.window_s == 1.0
    assert cfg.detect.n_states == 4
    assert cfg.detect.self_transition == 0.98
    assert cfg.beats_checkpoint is None


def test_gt_rules_speed_nominal_rpm_matches_measured_plateau() -> None:
    # Task 13b (2026-06-25 real data): Task 13 measured the plateau off the WRONG
    # channel ("1_Drehzahl_Ist", ~101 rpm) -- that channel is a percent-of-nominal-ish
    # quantity, not rpm. GT_CHANNELS["speed"] now points at "1_Drehzahl UPM" (the
    # genuine rpm channel; confirmed ~3.75x "1_Drehzahl_Ist" on the same file), whose
    # measured plateau during full-power turbine generation (05:00 Betriebsdaten hour,
    # median while power > 50 MW) is 378.832 rpm -- almost exactly the pre-delivery
    # "8-pole 50 Hz machine" 375 rpm hypothesis Task 13 had discarded. Pinned here so a
    # future edit cannot silently regress this to an unverified value again.
    cfg = load_config(env={})
    assert cfg.gt.speed_nominal_rpm == 378.832


def test_env_overrides() -> None:
    cfg = load_config(env={"ROWII_DATA_ROOT": "/tmp/x", "ROWII_BEATS_CHECKPOINT": "/tmp/b.pt"})
    assert cfg.data_root == Path("/tmp/x")
    assert cfg.beats_checkpoint == Path("/tmp/b.pt")
