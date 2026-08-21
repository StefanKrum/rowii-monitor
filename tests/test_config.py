from pathlib import Path

from rowii.config import load_config


def test_defaults_without_env() -> None:
    cfg = load_config(env={})
    assert cfg.window.window_s == 1.0
    assert cfg.window.hop_s is None
    assert cfg.detect.n_states == 4
    assert cfg.detect.self_transition == 0.98
    assert cfg.beats_checkpoint is None
    assert cfg.tfc_audio_checkpoint is None
    assert cfg.tfc_vib_checkpoint is None
    assert cfg.student_checkpoint is None
    assert cfg.beats_int8_checkpoint is None


def test_gt_rules_speed_nominal_rpm_matches_measured_plateau() -> None:
    # 2026-06-25 real data: an earlier pass measured the plateau off the WRONG
    # channel ("1_Drehzahl_Ist", ~101 rpm) -- that channel is a percent-of-nominal-ish
    # quantity, not rpm. GT_CHANNELS["speed"] now points at "1_Drehzahl UPM" (the
    # genuine rpm channel; confirmed ~3.75x "1_Drehzahl_Ist" on the same file), whose
    # measured plateau during full-power turbine generation (05:00 Betriebsdaten hour,
    # median while power > 50 MW) is 378.832 rpm -- almost exactly the pre-delivery
    # "8-pole 50 Hz machine" 375 rpm hypothesis that earlier pass had discarded. Pinned
    # here so a future edit cannot silently regress this to an unverified value again.
    cfg = load_config(env={})
    assert cfg.gt.speed_nominal_rpm == 378.832


def test_env_overrides() -> None:
    cfg = load_config(env={"ROWII_DATA_ROOT": "/tmp/x", "ROWII_BEATS_CHECKPOINT": "/tmp/b.pt"})
    assert cfg.data_root == Path("/tmp/x")
    assert cfg.beats_checkpoint == Path("/tmp/b.pt")


def test_tfc_checkpoint_env_overrides() -> None:
    # Mirrors beats_checkpoint's own env-driven pattern exactly:
    # two independent checkpoints (audio vs vibration branch), each its own env var.
    cfg = load_config(
        env={
            "ROWII_DATA_ROOT": "/tmp/x",
            "ROWII_TFC_AUDIO_CHECKPOINT": "/tmp/tfc_audio.pt",
            "ROWII_TFC_VIB_CHECKPOINT": "/tmp/tfc_vib.pt",
        }
    )
    assert cfg.tfc_audio_checkpoint == Path("/tmp/tfc_audio.pt")
    assert cfg.tfc_vib_checkpoint == Path("/tmp/tfc_vib.pt")


def test_tfc_checkpoint_env_overrides_independently() -> None:
    # Setting only ONE of the two must not affect the other -- they are independent
    # fields, not a shared/derived pair.
    cfg = load_config(
        env={"ROWII_DATA_ROOT": "/tmp/x", "ROWII_TFC_AUDIO_CHECKPOINT": "/tmp/tfc_audio.pt"}
    )
    assert cfg.tfc_audio_checkpoint == Path("/tmp/tfc_audio.pt")
    assert cfg.tfc_vib_checkpoint is None


def test_student_checkpoint_env_override() -> None:
    # Mirrors beats_checkpoint's own single-field env-driven pattern:
    # unlike TF-C, the student has only ONE (audio-only) checkpoint.
    cfg = load_config(
        env={"ROWII_DATA_ROOT": "/tmp/x", "ROWII_STUDENT_CHECKPOINT": "/tmp/student.pt"}
    )
    assert cfg.student_checkpoint == Path("/tmp/student.pt")


def test_beats_int8_checkpoint_env_override() -> None:
    # Mirrors student_checkpoint's own single-field env-driven pattern:
    # the INT8-quantized BEATs module pickle (scripts/quantize_beats.py's
    # output), independent of beats_checkpoint (the fp32 source it was quantized
    # FROM -- both may be set together, see
    # test_beats_int8_checkpoint_env_overrides_independently_of_beats_checkpoint below).
    cfg = load_config(
        env={"ROWII_DATA_ROOT": "/tmp/x", "ROWII_BEATS_INT8_CHECKPOINT": "/tmp/beats_int8.pt"}
    )
    assert cfg.beats_int8_checkpoint == Path("/tmp/beats_int8.pt")


def test_beats_int8_checkpoint_env_overrides_independently_of_beats_checkpoint() -> None:
    cfg = load_config(
        env={
            "ROWII_DATA_ROOT": "/tmp/x",
            "ROWII_BEATS_CHECKPOINT": "/tmp/beats.pt",
            "ROWII_BEATS_INT8_CHECKPOINT": "/tmp/beats_int8.pt",
        }
    )
    assert cfg.beats_checkpoint == Path("/tmp/beats.pt")
    assert cfg.beats_int8_checkpoint == Path("/tmp/beats_int8.pt")


def test_window_hop_s_env_override() -> None:
    # The window DURATION stays window_s; ROWII_WINDOW_HOP_S only changes the
    # spacing between consecutive window starts (unset -> hop == window_s).
    cfg = load_config(env={"ROWII_DATA_ROOT": "/tmp/x", "ROWII_WINDOW_HOP_S": "0.25"})
    assert cfg.window.window_s == 1.0
    assert cfg.window.hop_s == 0.25
