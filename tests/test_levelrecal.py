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

import logging

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


# ---------------------------------------------------------------------------
# T5-review finding 1 (BLOCKER): the empty-set guard is an EMBEDDING-variant
# refusal, not a fusion refusal -- fuse() z-scores VALUES only (module
# docstring), so fusion's feature NAMES retain the level tokens and pattern-match
# level_columns exactly like audio/vibration. These two tests pin the REAL
# behaviour of both sides of that corrected claim.
# ---------------------------------------------------------------------------


def test_fusion_style_names_are_matched_by_level_columns() -> None:
    """`fuse()` (module docstring) z-scores VALUES only and never renames or drops
    a column, so a `"fusion"` variant's `feature_names` is literally the audio
    stream's names followed by the vibration stream's names -- level tokens
    included. `_NAMES` above is exactly that shape (mic-prefixed level/shape
    columns plus a vib-prefixed shape column), so `level_columns`/`column_medians`
    treat it like any other audio/vibration mix -- the empty-set guard does NOT
    fire for fusion. Excluding fusion from `--level-recal` is therefore a
    VALUE-domain decision made by callers (`run_step2 --level-recal` requires
    `--variant audio` or `vibration`), not something this name-based guard can or
    should encode."""
    assert level_columns(_NAMES) == [0, 1, 2]
    medians = column_medians(np.ones((4, len(_NAMES))), _NAMES)
    assert set(medians) == {_NAMES[0], _NAMES[1], _NAMES[2]}


def test_beats_embedding_names_raise_the_empty_set_guard() -> None:
    """Real `BeatsFeaturizer.feature_names()` output (`rowii.signals.beats`) --
    unlike fusion (previous test), an embedding variant's names carry no level
    token at all, so this IS the empty-set guard's actual refusal case (A1.9)."""
    embedding_names = ["beats_0", "beats_1"]
    assert level_columns(embedding_names) == []
    with pytest.raises(ValueError, match="no level column"):
        column_medians(np.zeros((3, 2)), embedding_names)


# ---------------------------------------------------------------------------
# T5-review finding 2 (minor): level_recal_offsets now logs a warning naming the
# dropped-key count (+ up to 3 examples) whenever the run/reference key sets
# differ, instead of dropping the non-overlapping keys silently.
# ---------------------------------------------------------------------------


def test_offsets_logs_no_warning_when_key_sets_match(
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_median = {"a": 1.0, "b": 2.0}
    reference_median = {"a": 0.0, "b": 0.0}
    with caplog.at_level(logging.WARNING, logger="rowii.anomaly.levelrecal"):
        level_recal_offsets(run_median, reference_median)
    assert caplog.records == []


def test_offsets_warns_with_dropped_count_and_up_to_three_examples(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 5 run-only keys (r0..r4) plus one key shared with the reference -- 5 dropped
    # total (symmetric difference), only the first 3 in sorted order get named.
    run_median = {f"r{i}": float(i) for i in range(5)} | {"shared": 9.0}
    reference_median = {"shared": 0.0}
    with caplog.at_level(logging.WARNING, logger="rowii.anomaly.levelrecal"):
        offsets = level_recal_offsets(run_median, reference_median)
    assert offsets == {"shared": 9.0}  # drop semantics unchanged -- still silent-safe
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    message = record.getMessage()
    assert "5" in message  # dropped-key count
    assert "r0" in message and "r1" in message and "r2" in message
    assert "r3" not in message and "r4" not in message  # capped at 3 examples


# ---------------------------------------------------------------------------
# T5-review finding 3 (minor): column_medians documents its no-NaN precondition
# and now enforces it (loud, matching this module's posture) -- scoped to the
# LEVEL columns only, since those are the only ones it reads.
# ---------------------------------------------------------------------------


def test_column_medians_raises_on_nan_in_level_column() -> None:
    rows = np.ones((4, len(_NAMES)))
    rows[2, 0] = np.nan  # NaN inside a level column (ch0_log_rms)
    with pytest.raises(ValueError, match="non-finite"):
        column_medians(rows, _NAMES)


def test_column_medians_raises_on_inf_in_level_column() -> None:
    rows = np.ones((4, len(_NAMES)))
    rows[1, 2] = np.inf  # +inf inside a level column (ch0_octave_125)
    with pytest.raises(ValueError, match="non-finite"):
        column_medians(rows, _NAMES)


def test_column_medians_ignores_non_finite_values_outside_level_columns() -> None:
    # NaN in a SHAPE column (spectral_centroid, index 3) must not affect a
    # function that only ever reads the LEVEL columns.
    rows = np.ones((4, len(_NAMES)))
    rows[0, 3] = np.nan
    medians = column_medians(rows, _NAMES)
    assert set(medians) == {_NAMES[0], _NAMES[1], _NAMES[2]}


# ---------------------------------------------------------------------------
# T5-review finding 4 (minor): guards that already existed but had no direct
# test -- apply_level_recal's unknown-offset-key guard, both functions' geometry
# guards, and apply_level_recal's float64-copy/non-mutation contract.
# ---------------------------------------------------------------------------


def test_apply_level_recal_raises_on_unknown_offset_key() -> None:
    feats = np.ones((3, len(_NAMES)))
    with pytest.raises(ValueError, match="not level columns"):
        apply_level_recal(feats, _NAMES, {"not_a_column": 1.0})


def test_apply_level_recal_raises_when_offset_key_is_a_shape_column() -> None:
    feats = np.ones((3, len(_NAMES)))
    # _NAMES[3] is ch0_spectral_centroid -- a shape column, never a valid offset key.
    with pytest.raises(ValueError, match="not level columns"):
        apply_level_recal(feats, _NAMES, {_NAMES[3]: 1.0})


def test_column_medians_raises_on_geometry_mismatch() -> None:
    rows = np.ones((4, len(_NAMES) - 1))  # one column short of feature_names
    with pytest.raises(ValueError, match="must be 2-D"):
        column_medians(rows, _NAMES)


def test_apply_level_recal_raises_on_geometry_mismatch() -> None:
    feats = np.ones((4, len(_NAMES) + 1))  # one column too many
    with pytest.raises(ValueError, match="must be 2-D"):
        apply_level_recal(feats, _NAMES, {})


def test_apply_level_recal_output_is_float64_and_does_not_mutate_input() -> None:
    feats = np.ones((3, len(_NAMES)), dtype=np.float32)
    original = feats.copy()
    out = apply_level_recal(feats, _NAMES, {_NAMES[0]: 1.0})
    assert out.dtype == np.float64
    np.testing.assert_array_equal(feats, original)  # input array untouched
    assert not np.shares_memory(out, feats)


def test_apply_level_recal_does_not_mutate_already_float64_input() -> None:
    # float64 input takes np.asarray's no-copy fast path internally -- the
    # explicit .copy() before writing must still protect it from mutation.
    feats = np.ones((3, len(_NAMES)), dtype=np.float64)
    original = feats.copy()
    out = apply_level_recal(feats, _NAMES, {_NAMES[0]: 5.0})
    np.testing.assert_array_equal(feats, original)
    assert not np.shares_memory(out, feats)
    assert out[0, 0] == feats[0, 0] - 5.0
