"""Human-readable labels for handcrafted feature names (spec D9).

Raw names come from the feature caches (`results/cache/<run>--{audio,vibration}
.npz`, key `feature_names`) in the form `"<stream>::ch<N>_<feature>"`, e.g.
`RAWGeneratorMic__0::ch0_octave_2000`. The site shows these as
`"Generator mic 1 · octave band 2 kHz"`. Unknown parts fall back to a cleaned
version of the raw text — never raise for a new feature name.
"""
from __future__ import annotations

import re

_STREAM_LABELS: dict[str, tuple[str, str]] = {
    # stream-name substring -> (display base, channel noun)
    "GeneratorMic": ("Generator mic", ""),
    "TurbineMic": ("Turbine mic", ""),
    "GeneratorVib": ("Generator vibration", "ch "),
    "TurbineVib": ("Turbine vibration", "ch "),
}

_FEATURE_LABELS: dict[str, str] = {
    "log_rms": "loudness (log RMS)",
    "spectral_centroid": "spectral centroid",
    "rolloff95": "spectral rolloff (95 %)",
    "kurtosis": "impulsiveness (kurtosis)",
    "band_shaft": "shaft band",
    "band_blade_pass": "blade-pass band",
    "band_guide_vane_pass": "guide-vane-pass band",
}

_VIB_FEATURE_OVERRIDES: dict[str, str] = {
    "log_rms": "level (log RMS)",
}

_OCTAVE_RE = re.compile(r"^octave_(\d+)$")
_CH_RE = re.compile(r"^ch(\d+)_(.+)$")


def _octave_label(center_hz: int) -> str:
    if center_hz >= 1000:
        khz = center_hz / 1000.0
        num = f"{khz:.1f}".rstrip("0").rstrip(".")
        return f"octave band {num} kHz"
    return f"octave band {center_hz} Hz"


def _band_label(feature: str, *, vib: bool) -> str:
    m = _OCTAVE_RE.match(feature)
    if m:
        return _octave_label(int(m.group(1)))
    if vib and feature in _VIB_FEATURE_OVERRIDES:
        return _VIB_FEATURE_OVERRIDES[feature]
    if feature in _FEATURE_LABELS:
        return _FEATURE_LABELS[feature]
    if feature.startswith("band_"):
        return feature.removeprefix("band_").replace("_", "-") + " band"
    return feature.replace("_", " ")


def humanize_feature_name(raw: str) -> str:
    if "::" not in raw:
        return raw
    stream, feat = raw.split("::", 1)
    m = _CH_RE.match(feat)
    if not m:
        return raw
    ch_index = int(m.group(1)) + 1  # 1-based for humans
    feature = m.group(2)
    for key, (base, ch_noun) in _STREAM_LABELS.items():
        if key in stream:
            vib = "Vib" in key
            return f"{base} {ch_noun}{ch_index} · {_band_label(feature, vib=vib)}"
    return f"{stream} ch {ch_index} · {_band_label(feature, vib=False)}"
