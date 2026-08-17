import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from feature_labels import humanize_feature_name


def test_mic_channel_features() -> None:
    assert humanize_feature_name("RAWGeneratorMic__0::ch0_log_rms") == "Generator mic 1 · loudness (log RMS)"
    assert humanize_feature_name("RAWGeneratorMic__0::ch2_octave_2000") == "Generator mic 3 · octave band 2 kHz"
    assert humanize_feature_name("RAWTurbineMic__1::ch0_octave_500") == "Turbine mic 1 · octave band 500 Hz"
    assert humanize_feature_name("RAWGeneratorMic__0::ch3_spectral_centroid") == "Generator mic 4 · spectral centroid"
    assert humanize_feature_name("RAWTurbineMic__1::ch0_rolloff95") == "Turbine mic 1 · spectral rolloff (95 %)"


def test_machine_bands() -> None:
    assert humanize_feature_name("RAWGeneratorMic__0::ch1_band_shaft") == "Generator mic 2 · shaft band"
    assert humanize_feature_name("RAWGeneratorMic__0::ch0_band_blade_pass") == "Generator mic 1 · blade-pass band"
    assert humanize_feature_name("RAWGeneratorMic__0::ch0_band_guide_vane_pass") == "Generator mic 1 · guide-vane-pass band"


def test_vibration_features() -> None:
    assert humanize_feature_name("RAWGeneratorVib__2::ch1_log_rms") == "Generator vibration ch 2 · level (log RMS)"
    assert humanize_feature_name("RAWTurbineVib__3::ch0_kurtosis") == "Turbine vibration ch 1 · impulsiveness (kurtosis)"
    assert humanize_feature_name("RAWTurbineVib__3::ch2_band_shaft") == "Turbine vibration ch 3 · shaft band"


def test_unknown_falls_back_to_raw() -> None:
    assert humanize_feature_name("weird") == "weird"
    assert humanize_feature_name("RAWGeneratorMic__0::ch0_totally_new") == "Generator mic 1 · totally new"
