"""Tests for `rowii.anomaly.levelrecal` (Package-8 D2 core, spec §3.D2 +
A1.1/A1.4/A1.9): level vs shape column selection (the verified log10 fact from
`rowii.signals.features`), offset golden math (offset = run - reference,
`apply_level_recal` subtracts it to recentre the run onto the reference), shape
columns untouched, and the empty-level-set guard (embedding/fusion variants).

`_NAMES` below mirrors the REAL `"<stream>::<local_name>"` naming produced by
`rowii.pipeline._assemble_feature_names` (e.g. `"RAWGeneratorMic__0::ch0_log_rms"`)
-- substring matching must work regardless of the stream prefix.
"""
from __future__ import annotations

import numpy as np
import pytest

from rowii.anomaly.levelrecal import (
    apply_level_recal,
    column_medians,
    level_columns,
    level_recal_offsets,
)

_NAMES = [
    "RAWGeneratorMic__0::ch0_log_rms",
    "RAWGeneratorMic__0::ch0_band_shaft",
    "RAWGeneratorMic__0::ch0_octave_125",
    "RAWGeneratorMic__0::ch0_spectral_centroid",  # shape
    "RAWGeneratorMic__0::ch0_rolloff95",  # shape
    "RAWTurbineVib__3::ch0_kurtosis",  # shape
]


def test_level_columns_selects_only_log_scaled_features() -> None:
    # log_rms, band, octave -- NOT centroid/rolloff/kurtosis.
    assert level_columns(_NAMES) == [0, 1, 2]


def test_offsets_and_apply_align_run_to_reference() -> None:
    rng = np.random.default_rng(0)
    feats = rng.normal(0.0, 0.1, (200, 6))
    feats[:, :3] += np.array([-40.0, -35.0, -30.0])  # a +level run
    run_med = column_medians(feats, _NAMES)
    ref_med = {n: run_med[n] - 2.0 for n in run_med}  # reference sits 2 (log10 units) below
    offsets = level_recal_offsets(run_med, ref_med)
    assert all(abs(v - 2.0) < 1e-9 for v in offsets.values())  # run - reference = +2 everywhere
    out = apply_level_recal(feats, _NAMES, offsets)
    # level columns recentred onto the reference; shape columns bit-identical.
    np.testing.assert_allclose(
        np.median(out[:, :3], axis=0),
        [run_med[_NAMES[i]] - 2.0 for i in range(3)],
        atol=1e-9,
    )
    np.testing.assert_array_equal(out[:, 3:], feats[:, 3:])


def test_empty_level_set_raises_for_embedding_variant() -> None:
    embedding_names = [f"RAWGeneratorMic__0::dim{i}" for i in range(8)]  # no level pattern
    assert level_columns(embedding_names) == []
    with pytest.raises(ValueError, match="no level column"):
        column_medians(np.zeros((5, 8)), embedding_names)
    with pytest.raises(ValueError, match="no level column"):
        apply_level_recal(np.zeros((5, 8)), embedding_names, {})


def test_docstring_records_the_verified_log_scale_fact() -> None:
    import rowii.anomaly.levelrecal as lr

    assert lr.__doc__ is not None
    assert "log10" in lr.__doc__ and "spectral_centroid" in lr.__doc__
