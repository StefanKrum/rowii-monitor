from pathlib import Path

from rowii.config import load_config


def test_defaults_without_env() -> None:
    cfg = load_config(env={})
    assert cfg.window.window_s == 1.0
    assert cfg.detect.n_states == 4
    assert cfg.detect.self_transition == 0.98
    assert cfg.beats_checkpoint is None


def test_gt_rules_speed_nominal_rpm_matches_measured_plateau() -> None:
    # Task 13 (2026-06-25 real data): the pre-delivery 375 rpm hypothesis (documented as
    # "8-pole 50 Hz machine" reasoning) does not hold -- the measured 1_Drehzahl_Ist plateau
    # during full-power turbine generation (05:00 Betriebsdaten hour, median while
    # power > 50 MW) is ~101.0 rpm, a 73% deviation from 375. Pinned here so a future edit
    # cannot silently regress this to an unverified value again.
    cfg = load_config(env={})
    assert cfg.gt.speed_nominal_rpm == 101.0


def test_env_overrides() -> None:
    cfg = load_config(env={"ROWII_DATA_ROOT": "/tmp/x", "ROWII_BEATS_CHECKPOINT": "/tmp/b.pt"})
    assert cfg.data_root == Path("/tmp/x")
    assert cfg.beats_checkpoint == Path("/tmp/b.pt")
