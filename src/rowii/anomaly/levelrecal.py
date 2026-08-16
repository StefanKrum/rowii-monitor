"""Level-only, shape-preserving channel recalibration.

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
unit and is never imported here (this module computes its own offsets from our own caches).
Which side is "run" and which is "reference" is a caller decision -- `run_step2
--level-recal` anchors on the pooled-fit median, `monitor --level-recal` anchors
on the snapshot-stored median -- this module's four functions are surface-agnostic:
they operate on features + names in, offsets/recalibrated features out.

**Fusion is out of scope by design -- refused upstream, not by this
module's empty-set guard.** `fuse()` (`rowii.signals.features`) is
`np.hstack([zscore(a), zscore(b)])`: it z-scores VALUES per run but never
renames or drops a column, so a `"fusion"` variant's `feature_names` is
literally its audio stream's names followed by its vibration stream's names
(`_assemble_feature_names`, `rowii.pipeline`), level tokens included --
`level_columns` returns the SAME non-empty set for `"fusion"` as for
`"audio"`/`"vibration"`, and `column_medians`/`apply_level_recal` do NOT
raise on it. The empty-set guard below is the EMBEDDING-variant refusal path
instead: `"audio-beats"`, `"audio-tfc"`, `"audio-student"`,
`"vibration-tfc"`, and `"logmel"` are single-stream variants whose one
featurizer (`BeatsFeaturizer`/`TfcFeaturizer`/`StudentFeaturizer`/
`LogmelFeaturizer`) names every column `beats_<i>`/`tfc_e<i>`/`student_e<i>`/
`logmel_f<i>_m<j>` -- no level token survives, so `level_columns` is empty
and `column_medians`/`apply_level_recal` correctly refuse. Fusion is
excluded a level up, in the VALUE domain rather than the name domain: its
stored features are per-run z-scores, so an additive log10-unit offset --
meaningful on a raw level feature -- has no physical meaning once applied to
a z-score, even though fusion's columns still pattern-match a level name.
That exclusion is enforced by the caller, not this module: `run_step2
--level-recal` requires `--variant audio` or `vibration` and errors
otherwise (commit d791d13); `monitor --level-recal` (not
yet built) is designed to inherit the same restriction transitively, since a
fusion-fit snapshot could never carry `level_recal_medians` once `run_step2
--save-snapshot --level-recal` itself already refused fusion at fit time.
Only `audio` and `vibration` are supported; this module stays agnostic
to which variant it is fed and cannot by itself stop a caller from
misapplying it to fusion's z-scored columns.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

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
        Ascending column indices of the level columns. Empty for an EMBEDDING
        variant (e.g. `audio-beats`/`audio-tfc`/`logmel` -- names like
        `beats_<i>`/`tfc_e<i>`/`logmel_f<i>_m<j>` carry no level token). NON-empty
        for fusion: its names retain the audio/vibration level tokens even though
        its stored VALUES are per-run z-scores (module docstring). Either result is
        legitimate and non-error; only the callers that need at least one level
        column (`column_medians`, `apply_level_recal`) raise.
    """
    return [
        i
        for i, name in enumerate(feature_names)
        if any(token in name for token in _LEVEL_SUBSTRINGS)
    ]


def column_medians(rows: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Per-column median of the LEVEL columns of `rows`, keyed by feature name.

    Args:
        rows: `(N, F)` feature matrix, `F == len(feature_names)` -- pass valid
            (non-NaN) rows only. `PreparedRun.features` is NaN on invalid windows
            (`rowii.pipeline._extract_stream_features`'s per-window NaN-fill,
            `compute_validity_mask`'s precondition); callers filter by `valid_mask`
            (or an equivalent valid-rows selection, e.g. `run_step2`'s
            `_first_n_minutes_rows`) before calling this. Enforced below for the
            LEVEL columns only -- the ones this function actually reads.
        feature_names: Column names aligned with `rows`' columns.

    Returns:
        `{feature_names[c]: median} for c in level_columns(feature_names)` -- one
        entry per level column, name-keyed so it survives a column reorder/subset
        between the run this was computed from and wherever it is later consumed
        (`level_recal_offsets`).

    Raises:
        ValueError: if `feature_names` selects zero level columns -- an EMBEDDING
            variant (`beats_<i>`/`tfc_e<i>`/`student_e<i>`/`logmel_f<i>_m<j>` names
            carry no level token at all); NOT fusion, whose names retain the
            level tokens (module docstring) and are excluded by callers on
            value-domain grounds instead. Also raises if `rows` is not 2-D with
            `len(feature_names)` columns (loud geometry, the snapshot posture), or
            if `rows`' level columns hold a non-finite (NaN/Inf) value (the Args
            precondition above, enforced rather than silently propagated into a
            NaN/garbage median).
    """
    cols = level_columns(feature_names)
    if not cols:
        raise ValueError(
            "column_medians: no level column (_log_rms/_band_/_octave_) in "
            "feature_names -- level-recal is undefined for this (embedding) variant"
        )
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(feature_names):
        raise ValueError(
            f"column_medians: rows must be 2-D with {len(feature_names)} column(s) "
            f"(len(feature_names)), got shape {arr.shape}"
        )
    level_values = arr[:, cols]
    if not np.isfinite(level_values).all():
        raise ValueError(
            "column_medians: rows' level column(s) contain a non-finite (NaN/Inf) "
            "value -- pass valid (non-NaN) rows only (PreparedRun.features is NaN "
            "on invalid windows; filter by valid_mask first)"
        )
    medians = np.median(level_values, axis=0)
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
            snapshot-stored median in `monitor --level-recal`).

    Returns:
        `{k: run_median[k] - reference_median[k]}` for every key SHARED by both
        dicts (sorted for determinism) -- positive when the run sits above the
        reference. A key present in only one side is silently dropped rather than
        raising, so a partial overlap (e.g. `vibration`'s live-channel set differing
        across runs, `rowii.signals.features.VibFeaturizer`) degrades to "no offset
        computed for that column" instead of a hard failure; `apply_level_recal`
        then leaves any such column un-recentred. A `logger.warning` (module
        `"rowii.anomaly.levelrecal"`) names the dropped-key count and up to 3
        example keys whenever the two key sets differ at all, so the drop stays
        silent only in the sense of "not an exception" -- it is always logged.

    Raises:
        ValueError: if `run_median` and `reference_median` share no key at all --
            an offset computed from nothing would silently do nothing while looking
            like it succeeded.
    """
    run_keys = set(run_median)
    reference_keys = set(reference_median)
    keys = sorted(run_keys & reference_keys)
    if not keys:
        raise ValueError(
            "level_recal_offsets: run_median and reference_median share no level "
            "column -- cannot compute any offset"
        )
    dropped = sorted(run_keys ^ reference_keys)
    if dropped:
        logger.warning(
            "level_recal_offsets: run_median/reference_median key sets differ -- "
            "%d level column(s) dropped (no offset computed for them): %s%s",
            len(dropped),
            dropped[:3],
            ", ..." if len(dropped) > 3 else "",
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
        ValueError: if `feature_names` selects zero level columns (the same
            EMBEDDING refusal as `column_medians` -- not fusion, whose names retain
            the level tokens, module docstring); if `features` is not 2-D with
            `len(feature_names)` columns (loud geometry); if `offsets` names a
            column that is not one of `feature_names`' level columns (a shape column
            must never receive an offset, and an unknown name would otherwise raise
            a bare `KeyError` deep inside the loop below).
    """
    cols = level_columns(feature_names)
    if not cols:
        raise ValueError(
            "apply_level_recal: no level column (_log_rms/_band_/_octave_) in "
            "feature_names -- level-recal is undefined for this (embedding) variant"
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
