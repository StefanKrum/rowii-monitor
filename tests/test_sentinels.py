"""Tests for rowii.anomaly.sentinels (Package-9 D1, A1.1): the level-series
stream∩level extraction (raises for embeddings), the seeded segment-block s1
bootstrap threshold + firing, and the s2 anchor/MAD math + mic-out/vib-in
attribution. Deterministic, no real data, no partner number as an expected value.

The s2 MAD is 1.4826-scaled (the SAME `_MAD_TO_SIGMA` normal-consistency factor as
`rowii.anomaly.normalize`'s `SessionNormalizer`/`_center_scale`) -- the Task-5
Interfaces bullet's resolution of the plan's own self-review item 5 (RAW vs scaled
MAD), pinned explicitly by `test_s2_anchor_mad_uses_1_4826_scaling_like_session_
normalizer` below so a future regression to raw MAD fails loudly.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from rowii.anomaly.sentinels import (
    level_series,
    s1_fires,
    s1_threshold,
    s2_anchor_mad,
    s2_attribution,
    s2_fires,
)

_MIC = ("RAWGeneratorMic__0", "RAWTurbineMic__1")


def _names() -> list[str]:
    return [
        "RAWGeneratorMic__0::ch0_log_rms", "RAWGeneratorMic__0::ch0_spectral_centroid",
        "RAWTurbineMic__1::ch0_octave_125", "RAWTurbineMic__1::ch0_rolloff95",
    ]


def test_level_series_averages_stream_level_columns() -> None:
    names = _names()
    rows = np.zeros((5, 4), dtype=np.float64)
    rows[:, 0] = -40.0  # mic0 log_rms (level)
    rows[:, 2] = -30.0  # mic1 octave (level); cols 1,3 are shape -> ignored
    series = level_series(rows, names, _MIC)
    np.testing.assert_allclose(series, np.full(5, -35.0))  # mean of the two level cols


def test_level_series_raises_for_embedding_names() -> None:
    embed = [f"RAWGeneratorMic__0::beats_{i}" for i in range(8)]
    with pytest.raises(ValueError, match="level"):
        level_series(np.zeros((3, 8)), embed, _MIC)


def test_level_series_raises_for_geometry_mismatch() -> None:
    """P9 hardening T5a: width guard (mirrors `rowii.anomaly.levelrecal`'s
    `column_medians`/`apply_level_recal` geometry posture: `ndim != 2 or
    shape[1] != len(feature_names)` -> ValueError). Without it, an oversized
    rows array silently succeeds -- the selected columns (0, 2) happen to
    still be in bounds for a 6-wide array -- instead of failing loudly on the
    geometry mismatch itself."""
    names = _names()  # 4 columns
    with pytest.raises(ValueError, match="2-D"):
        level_series(np.zeros((5, 6)), names, _MIC)  # 6 != len(names)==4


def test_level_series_warns_when_some_but_not_all_streams_absent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P9 hardening T5b: a PARTIAL stream absence (one of the two `_MIC`
    streams present, the other entirely missing from feature_names) must be
    visible via `logger.warning` naming the skipped stream -- only a TOTAL
    absence across every requested stream stays a silent-to-a-log,
    hard-ValueError condition (`test_level_series_raises_for_embedding_names`
    above)."""
    names = [
        "RAWGeneratorMic__0::ch0_log_rms", "RAWGeneratorMic__0::ch0_spectral_centroid",
    ]  # RAWTurbineMic__1 (_MIC's second stream) is entirely absent
    rows = np.zeros((5, 2), dtype=np.float64)
    rows[:, 0] = -40.0
    with caplog.at_level(logging.WARNING):
        series = level_series(rows, names, _MIC)
    np.testing.assert_allclose(series, np.full(5, -40.0))
    assert any(
        "RAWTurbineMic__1" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    )


def test_s1_threshold_is_deterministic_and_fires_above() -> None:
    rng = np.random.default_rng(0)
    seg = np.repeat(np.arange(20), 10)
    no_fit = (rng.random(200) < 0.05)  # ~5% baseline rejection
    thr = s1_threshold(no_fit, seg, n_boot=1000, seed=7)
    assert thr == s1_threshold(no_fit, seg, n_boot=1000, seed=7)  # seeded -> identical
    assert 0.0 <= thr <= 1.0
    assert s1_fires(0.5, thr) is True     # a 50% day-rate is well above the ~5% band
    assert s1_fires(float(no_fit.mean()), thr) is False  # the baseline itself does not fire


def test_s2_anchor_mad_and_fire() -> None:
    seg = np.repeat(np.arange(6), 10)
    level = np.concatenate([np.full(10, -40.0 + 0.01 * b) for b in range(6)])
    anchor, mad = s2_anchor_mad(level, seg)
    assert anchor == pytest.approx(np.median(level), abs=1e-6)
    assert mad >= 0.0
    assert s2_fires(anchor + 10.0, anchor, mad, k=3.0) is True    # a big step fires
    assert s2_fires(anchor + 0.001, anchor, mad, k=3.0) is False  # within the band


def test_s2_anchor_mad_uses_1_4826_scaling_like_session_normalizer() -> None:
    """A1.1 pin (T5 Interfaces, resolving the plan's own self-review item 5): mad is
    the 1.4826-scaled MAD over the per-segment-block medians, not the raw MAD --
    the same `_MAD_TO_SIGMA` precedent as `rowii.anomaly.normalize._center_scale`."""
    seg = np.repeat(np.arange(6), 10)
    level = np.concatenate([np.full(10, -40.0 + 0.01 * b) for b in range(6)])
    _anchor, mad = s2_anchor_mad(level, seg)
    block_medians = np.array([-40.0 + 0.01 * b for b in range(6)])
    raw_mad = float(np.median(np.abs(block_medians - np.median(block_medians))))
    assert mad == pytest.approx(1.4826 * raw_mad, rel=1e-9)


def test_s2_anchor_mad_floors_degenerate_single_block_mad() -> None:
    """P9 hardening T5c: a single-block anchor has `raw_mad == 0` exactly (one
    block medians to nothing to spread across) -- without a floor this makes
    `mad == 0.0`, a zero-margin hair-trigger where `s2_fires` fires on ANY
    nonzero deviation. Floored at 1e-8 (mirrors `rowii.anomaly.normalize.
    _center_scale`'s `_SCALE_FLOOR` precedent) so a degenerate commissioning
    block can never produce that hair-trigger."""
    seg = np.zeros(10, dtype=np.int64)  # single block -> raw_mad == 0.0 exactly
    level = np.full(10, -40.0)
    anchor, mad = s2_anchor_mad(level, seg)
    assert anchor == pytest.approx(-40.0)
    assert mad == pytest.approx(1e-8)
    assert s2_fires(anchor + 1e-9, anchor, mad, k=3.0) is False  # tiny deviation: no fire


def test_s2_attribution_mic_out_vib_in_is_instrumentation() -> None:
    assert s2_attribution(mic_fires=True, vib_fires=False) == "instrumentation"
    assert s2_attribution(mic_fires=True, vib_fires=True) == "machine"
