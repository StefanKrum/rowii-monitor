"""Level-only, shape-preserving channel recalibration (Package-8 D2, spec
`docs/superpowers/specs/2026-07-21-step2-package8-modebank-recal-explain.md` §3.D2
as amended by A1.1/A1.4/A1.9, plan `docs/superpowers/plans/2026-07-21-step2-
package8-modebank-explain.md` Task 5).

**VERIFIED FACT (`rowii.signals.features`, re-confirmed against the source before
writing the offset math below).** `AudioFeaturizer`/`VibFeaturizer` emit `*_log_rms`,
`*_band_<name>`, and `*_octave_<center>` via `_log10_floor(·)` -- base-10 log with a
`1e-12` floor -- so these are LEVEL features in the log10 domain. `*_spectral_centroid`
and `*_rolloff95` (audio) and `*_kurtosis` (vibration) are RAW units (Hz / a
dimensionless moment), never log-scaled -- these are SHAPE features and this module
never touches them. `_assemble_feature_names` (`rowii.pipeline`) prefixes each
featurizer's local names with `"<stream>::"` (e.g. `"RAWGeneratorMic__0::ch0_log_rms"`)
without renaming the suffix, so plain substring matching on the local-name tokens below
is stream-prefix-agnostic.

**Offset direction (pinned).** An offset is `run_median - reference_median` per
level column, name-keyed (`level_recal_offsets`): positive when the run sits ABOVE the
reference. `apply_level_recal` SUBTRACTS the offset from the run's level columns,
recentring the run's level distribution onto the reference's -- i.e. this module always
aligns "the run being recalibrated" onto "the anchor", never the other way round. Since
the offset is a difference of medians of the STORED log10 features, it is
self-consistently in log10 units by construction; a partner "dB" figure is never our
unit and is never imported here (D2 computes its own offsets from our own caches).
Which side is "run" and which is "reference" is a caller decision -- `run_step2
--level-recal` anchors on the pooled-fit median (A1.4), `monitor --level-recal` anchors
on the snapshot-stored median -- this module's four functions are surface-agnostic:
they operate on features + names in, offsets/recalibrated features out.

**Fusion is out of scope by design (A1.1).** `fuse()` (`rowii.signals.features`)
z-scores each stream PER RUN before concatenating, so fusion's stored features are
dimensionless per-run z-scores with no meaningful level columns -- and a broadband
level step is already removed by that per-run standardization. `level_columns` returns
an empty list for such a variant (and for any embedding variant, e.g. audio-beats),
and `column_medians`/`apply_level_recal` both refuse (`ValueError`) on an empty level
set rather than silently operating on nothing -- this refusal path IS the A1.9
fusion/embedding guard; D2's variants are `audio` and `vibration` only.
"""
from __future__ import annotations

import numpy as np

_LEVEL_SUBSTRINGS = ("_log_rms", "_band_", "_octave_")
"""Local-name substrings of the log10-scaled level features (module docstring,
VERIFIED against `rowii.signals.features`): `*_log_rms`, `*_band_<name>`,
`*_octave_<center>`."""

_SHAPE_SUBSTRINGS = ("_spectral_centroid", "_rolloff95", "_kurtosis")
"""Local-name substrings of the RAW-unit shape features that this module never
touches: `*_spectral_centroid`, `*_rolloff95` (audio), `*_kurtosis` (vibration)."""


def level_columns(feature_names: list[str]) -> list[int]:
    """Indices of `feature_names` whose local name contains a level-feature token
    (`_LEVEL_SUBSTRINGS`) -- i.e. the log10-scaled `*_log_rms`/`*_band_*`/`*_octave_*`
    columns, in their original column order.

    Args:
        feature_names: `PreparedRun.feature_names`-style column names (may carry a
            `"<stream>::"` prefix, matched by substring so the prefix is irrelevant).

    Returns:
        Ascending column indices of the level columns. Empty for a variant with no
        level columns at all (fusion's z-scored features, embedding variants such as
        audio-beats) -- a legitimate, non-error result; only the callers that need at
        least one level column (`column_medians`, `apply_level_recal`) raise.
    """
    return [
        i
        for i, name in enumerate(feature_names)
        if any(token in name for token in _LEVEL_SUBSTRINGS)
    ]


def column_medians(rows: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Per-column median of the LEVEL columns of `rows`, keyed by feature name.

    Args:
        rows: `(N, F)` feature matrix, `F == len(feature_names)`.
        feature_names: Column names aligned with `rows`' columns.

    Returns:
        `{feature_names[c]: median} for c in level_columns(feature_names)` -- one
        entry per level column, name-keyed so it survives a column reorder/subset
        between the run this was computed from and wherever it is later consumed
        (`level_recal_offsets`).

    Raises:
        ValueError: if `feature_names` selects zero level columns (A1.9 -- fusion's
            z-scored features or an embedding variant have no meaningful level
            column; level-recal is undefined for them); if `rows` is not 2-D with
            `len(feature_names)` columns (loud geometry, the snapshot posture).
    """
    cols = level_columns(feature_names)
    if not cols:
        raise ValueError(
            "column_medians: no level column (_log_rms/_band_/_octave_) in "
            "feature_names -- level-recal is undefined for this variant (fusion "
            "z-scores / embeddings)"
        )
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(feature_names):
        raise ValueError(
            f"column_medians: rows must be 2-D with {len(feature_names)} column(s) "
            f"(len(feature_names)), got shape {arr.shape}"
        )
    medians = np.median(arr[:, cols], axis=0)
    return {feature_names[c]: float(m) for c, m in zip(cols, medians, strict=True)}


def level_recal_offsets(
    run_median: dict[str, float], reference_median: dict[str, float]
) -> dict[str, float]:
    """Per-column additive offset aligning a run's level columns onto a reference's.

    Args:
        run_median: Name-keyed level-column medians of the run to be recalibrated
            (`column_medians` on that run's own rows).
        reference_median: Name-keyed level-column medians of the anchor/reference
            (e.g. the pooled-fit median in `run_step2 --level-recal`, or the
            snapshot-stored median in `monitor --level-recal`, A1.4).

    Returns:
        `{k: run_median[k] - reference_median[k]}` for every key SHARED by both
        dicts (sorted for determinism) -- positive when the run sits above the
        reference. A key present in only one side is silently dropped rather than
        raising, so a partial overlap (e.g. `vibration`'s live-channel set differing
        across runs, `rowii.signals.features.VibFeaturizer`) degrades to "no offset
        computed for that column" instead of a hard failure; `apply_level_recal`
        then leaves any such column un-recentred.

    Raises:
        ValueError: if `run_median` and `reference_median` share no key at all --
            an offset computed from nothing would silently do nothing while looking
            like it succeeded.
    """
    keys = sorted(set(run_median) & set(reference_median))
    if not keys:
        raise ValueError(
            "level_recal_offsets: run_median and reference_median share no level "
            "column -- cannot compute any offset"
        )
    return {k: run_median[k] - reference_median[k] for k in keys}


def apply_level_recal(
    features: np.ndarray, feature_names: list[str], offsets: dict[str, float]
) -> np.ndarray:
    """Recentre `features`' level columns onto the reference by subtracting `offsets`;
    every other column (shape features, and any level column absent from `offsets`)
    passes through bit-identical.

    Args:
        features: `(N, F)` feature matrix, `F == len(feature_names)`.
        feature_names: Column names aligned with `features`' columns.
        offsets: Name-keyed additive offsets from `level_recal_offsets`, one entry
            per column to recentre; every key must be one of `feature_names`' level
            columns (see Raises).

    Returns:
        A fresh `(N, F)` float64 array (never a view of `features`): for each
        `name -> offset` in `offsets`, that column is `features[:, idx] - offset`;
        every other column is copied unchanged.

    Raises:
        ValueError: if `feature_names` selects zero level columns (A1.9, the same
            fusion/embedding refusal as `column_medians`); if `features` is not 2-D
            with `len(feature_names)` columns (loud geometry); if `offsets` names a
            column that is not one of `feature_names`' level columns (a shape column
            must never receive an offset, and an unknown name would otherwise raise
            a bare `KeyError` deep inside the loop below).
    """
    cols = level_columns(feature_names)
    if not cols:
        raise ValueError(
            "apply_level_recal: no level column (_log_rms/_band_/_octave_) in "
            "feature_names -- level-recal is undefined for this variant (fusion "
            "z-scores / embeddings)"
        )
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(feature_names):
        raise ValueError(
            f"apply_level_recal: features must be 2-D with {len(feature_names)} "
            f"column(s) (len(feature_names)), got shape {arr.shape}"
        )
    level_names = {feature_names[c] for c in cols}
    unknown = sorted(set(offsets) - level_names)
    if unknown:
        raise ValueError(
            f"apply_level_recal: offsets name {len(unknown)} column(s) that are not "
            f"level columns of feature_names ({unknown!r}) -- shape columns must "
            f"never receive a level offset"
        )
    out: np.ndarray = arr.copy()
    index_by_name = {name: i for i, name in enumerate(feature_names)}
    for name, offset in offsets.items():
        out[:, index_by_name[name]] -= offset
    return out
