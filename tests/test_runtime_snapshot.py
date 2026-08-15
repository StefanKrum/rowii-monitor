"""Tests for `rowii.runtime.snapshot`: pickle-free `MonitorSnapshot` round-trip (bitwise
apply/score parity), format guards (version mismatch, runtime-scorer whitelist), the k<=1 degenerate
detector (A1.2), and split parity with `run_sweep`'s exact top/nested discipline
(A1.6).

Fixture sizing note (empirically verified before hardcoding, matching
`test_sweep.py`'s established practice for segment-split constructions whose outcome
cannot be derived by inspection): the two-state fixture here uses TEN alternating
segments, not `test_apply_detector.py`'s eight -- at `SweepConfig` defaults (top-split
seed 7, nested seed 8) the 8-segment layout routes every odd-segment calibration
window to the nested FIT side, leaving that label with zero conformal-side windows,
which `fit_snapshot` (correctly, per the binding drop rule) removes from the snapshot.
The 8-segment layout is therefore used HERE only by the dedicated drop-path test;
every round-trip test needs BOTH labels to survive into the snapshot, which the
10-segment layout provides at these exact seeds (verified counts: fit-side >= min_ref
and conformal-side >= 1 for both labels).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest

from rowii.anomaly.conformal import ConformalThreshold, calibrate
from rowii.anomaly.normalize import SessionStats, apply_session_norm
from rowii.anomaly.references import split_by_segments
from rowii.anomaly.scorers import KnnScorer
from rowii.anomaly.sweep import SweepConfig
from rowii.config import Config, DetectConfig
from rowii.pipeline import PreparedRun
from rowii.runtime.snapshot import (
    SNAPSHOT_FORMAT_VERSION,
    MonitorSnapshot,
    fit_snapshot,
    fit_snapshot_from_parts,
    load_snapshot,
    save_snapshot,
    scorer_for_label,
    to_detector,
)
from rowii.signals.windows import WindowGrid
from rowii.state.detect import FittedDetector

_N_SEGMENTS = 10
"""Round-trip fixture segment count -- see module docstring for why not 8."""
_SEG_LEN = 30


# ---------------------------------------------------------------------------
# Hand-built fixtures (mirrors `tests/test_apply_detector.py::_two_state_prepared`)
# ---------------------------------------------------------------------------


def _two_state_prepared(n_segments: int = _N_SEGMENTS, seed: int = 0) -> PreparedRun:
    """Two well-separated 'states' (feature values 0 and 20), alternating by
    segment -- *n_segments* segments of `_SEG_LEN` windows each, every window valid."""
    rng = np.random.default_rng(seed)
    feats: list[np.ndarray] = []
    seg_ids: list[np.ndarray] = []
    for s in range(n_segments):
        value = 20.0 * (s % 2)
        feats.append(rng.normal(value, 0.1, (_SEG_LEN, 2)))
        seg_ids.append(np.full(_SEG_LEN, s, dtype=np.int64))
    features = np.vstack(feats)
    n = len(features)
    return PreparedRun(
        features=features,
        grid=WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=n),
        valid_mask=np.ones(n, dtype=bool),
        feature_names=["f0", "f1"],
        segment_ids=np.concatenate(seg_ids),
    )


def _one_state_prepared() -> PreparedRun:
    """A single tight blob -- with `k=1` the detector's smoother stays degenerate
    (`last_model_ is None`, amendment A1.2)."""
    rng = np.random.default_rng(3)
    n = _N_SEGMENTS * _SEG_LEN
    features = rng.normal(5.0, 0.1, (n, 2))
    return PreparedRun(
        features=features,
        grid=WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=n),
        valid_mask=np.ones(n, dtype=bool),
        feature_names=["f0", "f1"],
        segment_ids=np.repeat(np.arange(_N_SEGMENTS, dtype=np.int64), _SEG_LEN),
    )


def _cfg() -> Config:
    """Constructed directly (not `load_config()`) so ambient env vars / .env files
    can never leak into these library-level tests."""
    return Config(
        data_root=Path("/unused"),
        results_root=Path("/unused"),
        detect=DetectConfig(n_states=2, min_dwell_s=3.0),
    )


def _fit(
    prepared: PreparedRun, sweep_cfg: SweepConfig | None = None, k: int | None = None
) -> tuple[MonitorSnapshot, np.ndarray]:
    return fit_snapshot(
        prepared,
        _cfg(),
        sweep_cfg if sweep_cfg is not None else SweepConfig(),
        variant="fusion",
        fit_run="fit-day",
        k=k,
    )


def _valid_grid(prepared: PreparedRun) -> WindowGrid:
    """Grid over the VALID rows only -- the exact valid-grid construction
    `fit_snapshot` mirrors from `scripts/apply_detector.py::_fit_detector_and_mapping`."""
    return WindowGrid(
        t0_ns=prepared.grid.t0_ns,
        window_ns=prepared.grid.window_ns,
        n_windows=int(prepared.valid_mask.sum()),
    )


def _rewrite_meta(path: Path, **overrides: object) -> None:
    """Rewrite the npz's `meta` member in place with *overrides* merged in --
    the corruption seam for the version-mismatch test."""
    with np.load(path, allow_pickle=False) as data:
        members = {name: data[name] for name in data.files}
    meta = json.loads(str(members["meta"][0]))
    meta.update(overrides)
    members["meta"] = np.array([json.dumps(meta)], dtype=str)
    with open(path, "wb") as f:
        np.savez(f, allow_pickle=False, **members)


# ---------------------------------------------------------------------------
# 1. Round-trip apply + score parity (the central D1 guarantee)
# ---------------------------------------------------------------------------


def test_fit_snapshot_round_trip_apply_parity(tmp_path: Path) -> None:
    prepared = _two_state_prepared()
    snapshot, full_labels = _fit(prepared)

    # Both states must actually be in the snapshot for this test to mean anything
    # (fixture sizing, module docstring).
    assert len(snapshot.thresholds) == 2
    assert set(snapshot.references) == set(snapshot.calibration_scores) == set(
        snapshot.thresholds
    )

    path = tmp_path / "snap.npz"
    save_snapshot(path, snapshot)
    loaded = load_snapshot(path)

    features_valid = prepared.features[prepared.valid_mask]
    grid = _valid_grid(prepared)

    result_original = to_detector(snapshot).apply(features_valid, grid)
    result_loaded = to_detector(loaded).apply(features_valid, grid)
    assert np.array_equal(result_loaded.frame_labels, result_original.frame_labels)

    # ... and both equal the fit-time detector's own labels: `full_labels` IS the
    # fit-time output (FittedDetector.apply(fit_features) == fit, decode-only
    # property, `rowii.state.detect` docstring), scattered over valid_mask.
    assert np.array_equal(result_original.frame_labels, full_labels[prepared.valid_mask])

    # Per-label scorer parity, bitwise: identical reference bytes -> identical
    # deterministic normalization -> identical scores.
    for label in snapshot.thresholds:
        scores_original = scorer_for_label(snapshot, label).score(features_valid)
        scores_loaded = scorer_for_label(loaded, label).score(features_valid)
        assert np.array_equal(scores_original, scores_loaded)
        assert np.array_equal(
            snapshot.calibration_scores[label], loaded.calibration_scores[label]
        )
        assert loaded.thresholds[label] == snapshot.thresholds[label]


# ---------------------------------------------------------------------------
# 2. No pickle anywhere in the artifact
# ---------------------------------------------------------------------------


def test_snapshot_npz_has_no_pickle(tmp_path: Path) -> None:
    prepared = _two_state_prepared()
    snapshot, _ = _fit(prepared)
    path = tmp_path / "snap.npz"
    save_snapshot(path, snapshot)

    # `allow_pickle=False` load succeeding at all is the guarantee; every member
    # additionally has a plain (non-object) dtype.
    with np.load(path, allow_pickle=False) as data:
        assert all(data[name].dtype != object for name in data.files)
        meta = json.loads(str(data["meta"][0]))

    assert meta["format_version"] == SNAPSHOT_FORMAT_VERSION == 2  # v2 since package-7 Task 4
    assert meta["scorer"] == "knn"
    assert meta["variant"] == "fusion"
    assert meta["fit_run"] == "fit-day"
    assert set(meta["thresholds"]) == {str(label) for label in snapshot.thresholds}

    # The human-readable sidecar carries the same metadata.
    sidecar = path.with_suffix(".json")
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text())["format_version"] == 2


# ---------------------------------------------------------------------------
# 3. Format-version guard
# ---------------------------------------------------------------------------


def test_version_mismatch_raises(tmp_path: Path) -> None:
    prepared = _two_state_prepared()
    snapshot, _ = _fit(prepared)
    path = tmp_path / "snap.npz"
    save_snapshot(path, snapshot)

    _rewrite_meta(path, format_version=99)

    with pytest.raises(ValueError) as exc_info:
        load_snapshot(path)
    message = str(exc_info.value)
    assert "99" in message
    assert "1" in message


# ---------------------------------------------------------------------------
# 4. Geometry-guard fields round-trip (the monitor's refusal basis, Task 2)
# ---------------------------------------------------------------------------


def test_geometry_fields_present(tmp_path: Path) -> None:
    prepared = _two_state_prepared()
    snapshot, _ = _fit(prepared)
    path = tmp_path / "snap.npz"
    save_snapshot(path, snapshot)
    loaded = load_snapshot(path)

    assert loaded.feature_names == prepared.feature_names
    assert loaded.variant == "fusion"
    assert loaded.fit_run == "fit-day"
    assert loaded.mean.shape == (len(prepared.feature_names),)
    assert loaded.std.shape == (len(prepared.feature_names),)


# ---------------------------------------------------------------------------
# 5. Runtime-scorer whitelist refusal (save AND fit)
# ---------------------------------------------------------------------------


def test_save_refuses_non_runtime_scorer(tmp_path: Path) -> None:
    prepared = _two_state_prepared()
    snapshot, _ = _fit(prepared)

    bad = dataclasses.replace(snapshot, scorer="ocsvm")
    with pytest.raises(ValueError) as exc_info:
        save_snapshot(tmp_path / "bad.npz", bad)
    message = str(exc_info.value)
    assert "ocsvm" in message
    assert "knn" in message and "mahalanobis" in message

    # `fit_snapshot` refuses the same scorers up front -- a snapshot that could
    # never be saved must never be built.
    with pytest.raises(ValueError, match="ocsvm"):
        _fit(prepared, sweep_cfg=SweepConfig(scorer="ocsvm"))


# ---------------------------------------------------------------------------
# 6. k<=1 degenerate detector (amendment A1.2)
# ---------------------------------------------------------------------------


def test_degenerate_single_state_round_trip(tmp_path: Path) -> None:
    prepared = _one_state_prepared()
    snapshot, full_labels = _fit(prepared, k=1)

    assert snapshot.hmm_startprob is None
    assert snapshot.hmm_transmat is None
    assert snapshot.hmm_means is None
    assert snapshot.hmm_covars_diag is None
    assert len(snapshot.fitted_ids) == 1

    path = tmp_path / "snap.npz"
    save_snapshot(path, snapshot)

    # The degenerate snapshot must not even CARRY hmm members on disk.
    with np.load(path, allow_pickle=False) as data:
        assert not any(name.startswith("hmm_") for name in data.files)

    loaded = load_snapshot(path)
    features_valid = prepared.features[prepared.valid_mask]
    result = to_detector(loaded).apply(features_valid, _valid_grid(prepared))
    single_id = int(snapshot.fitted_ids[0])
    assert np.array_equal(
        result.frame_labels, np.full(len(features_valid), single_id, dtype=np.int64)
    )
    assert np.array_equal(full_labels[prepared.valid_mask], result.frame_labels)

    # The degenerate snapshot's SCORING half must work too --
    # one threshold, and a scorer that produces finite scores for that label.
    assert set(loaded.thresholds) == {single_id}
    scores = scorer_for_label(loaded, single_id).score(features_valid)
    assert np.isfinite(scores).all()


# ---------------------------------------------------------------------------
# 7. Split parity with run_sweep (amendment A1.6): hand-run of the same splits
# ---------------------------------------------------------------------------


def test_split_parity_with_run_sweep() -> None:
    prepared = _two_state_prepared()
    sweep_cfg = SweepConfig()
    snapshot, full_labels = _fit(prepared, sweep_cfg=sweep_cfg)

    # Hand-run the EXACT split sequence `run_sweep` performs (top split at
    # (calibration_frac, seed), nested split of the calibration half at (0.5, seed+1)).
    n = prepared.features.shape[0]
    top = split_by_segments(
        prepared.segment_ids, prepared.valid_mask, sweep_cfg.calibration_frac, sweep_cfg.seed
    )
    calib_mask = np.zeros(n, dtype=bool)
    calib_mask[top.calibration_windows] = True
    nested = split_by_segments(prepared.segment_ids, calib_mask, 0.5, sweep_cfg.seed + 1)
    fit_windows = nested.calibration_windows
    conformal_windows = nested.scoring_windows

    for label, threshold in snapshot.thresholds.items():
        n_fit = int((full_labels[fit_windows] == label).sum())
        n_conformal = int((full_labels[conformal_windows] == label).sum())
        assert snapshot.references[label].shape[0] == n_fit
        assert snapshot.calibration_scores[label].shape[0] == n_conformal
        assert threshold.n_calibration == n_conformal

        # Full threshold parity: refit the runtime scorer on the hand-run fit side
        # and recalibrate on the hand-run conformal side -- must agree exactly.
        reference = prepared.features[fit_windows][full_labels[fit_windows] == label]
        label_conformal = conformal_windows[full_labels[conformal_windows] == label]
        scores = KnnScorer(k=1, metric="cosine").fit(reference).score(
            prepared.features[label_conformal]
        )
        assert threshold == calibrate(scores, sweep_cfg.alpha)


# ---------------------------------------------------------------------------
# 8. Every ConformalThreshold field survives save/load exactly (incl. +inf)
# ---------------------------------------------------------------------------


def test_threshold_round_trip_fields(tmp_path: Path) -> None:
    prepared = _two_state_prepared()
    snapshot, _ = _fit(prepared)

    # Overwrite one label's threshold with a crafted low-confidence one so the
    # +inf / low_confidence=True corner round-trips too (a mode below the conformal
    # floor stores threshold=inf -- `rowii.anomaly.conformal.calibrate`).
    some_label = next(iter(snapshot.thresholds))
    crafted = ConformalThreshold(
        threshold=math.inf,
        alpha=0.001,
        n_calibration=3,
        achievable_alpha_floor=0.25,
        low_confidence=True,
    )
    snapshot = dataclasses.replace(
        snapshot, thresholds={**snapshot.thresholds, some_label: crafted}
    )

    path = tmp_path / "snap.npz"
    save_snapshot(path, snapshot)
    loaded = load_snapshot(path)

    assert set(loaded.thresholds) == set(snapshot.thresholds)
    for label, expected in snapshot.thresholds.items():
        got = loaded.thresholds[label]
        assert got == expected
        assert got.threshold == expected.threshold
        assert got.alpha == expected.alpha
        assert got.n_calibration == expected.n_calibration
        assert isinstance(got.n_calibration, int)
        assert got.achievable_alpha_floor == expected.achievable_alpha_floor
        assert got.low_confidence is expected.low_confidence


# ---------------------------------------------------------------------------
# 9. Zero-conformal-window labels are dropped with a warning (binding drop rule)
# ---------------------------------------------------------------------------


def test_label_without_conformal_windows_dropped_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The 8-segment layout at SweepConfig defaults leaves one label with a real
    # reference but ZERO conformal-side windows (module docstring) -- exactly the
    # drop path under test.
    prepared = _two_state_prepared(n_segments=8)
    with caplog.at_level(logging.WARNING):
        snapshot, _ = _fit(prepared)

    assert len(snapshot.fitted_ids) == 2  # the detector half still knows both states
    assert len(snapshot.thresholds) == 1  # ... but only one state can alarm
    assert set(snapshot.references) == set(snapshot.calibration_scores) == set(
        snapshot.thresholds
    )
    assert "dropped from the snapshot" in caplog.text
    assert "conformal" in caplog.text


# ---------------------------------------------------------------------------
# 10. Mahalanobis is the second whitelisted runtime scorer -- full round trip
# ---------------------------------------------------------------------------


def test_mahalanobis_scorer_round_trip(tmp_path: Path) -> None:
    prepared = _two_state_prepared()
    snapshot, _ = _fit(prepared, sweep_cfg=SweepConfig(scorer="mahalanobis"))
    assert snapshot.scorer == "mahalanobis"

    path = tmp_path / "snap.npz"
    save_snapshot(path, snapshot)
    loaded = load_snapshot(path)

    features_valid = prepared.features[prepared.valid_mask]
    for label in snapshot.thresholds:
        scores_original = scorer_for_label(snapshot, label).score(features_valid)
        scores_loaded = scorer_for_label(loaded, label).score(features_valid)
        assert np.array_equal(scores_original, scores_loaded)


def test_scorer_for_label_unknown_label_raises() -> None:
    prepared = _two_state_prepared()
    snapshot, _ = _fit(prepared)
    missing = max(snapshot.references) + 100
    with pytest.raises(KeyError) as exc_info:
        scorer_for_label(snapshot, missing)
    assert str(missing) in str(exc_info.value)


# ---------------------------------------------------------------------------
# 8. Hardening: covars pin, partial valid_mask, truncated archive
# ---------------------------------------------------------------------------


def test_covars_diagonals_round_trip_bit_exact(tmp_path: Path) -> None:
    """Pins a fixed-covariance guarantee directly: the reconstructed GaussianHMM consumes the
    STORED (k, F) diagonals bit-exactly (the well-separated
    fixture's Viterbi labels cannot pin this -- covariance never decides a label
    there -- so the array itself is asserted, plus disk-mutation propagation)."""
    prepared = _two_state_prepared()
    snapshot, _ = _fit(prepared)
    path = tmp_path / "snap.npz"
    save_snapshot(path, snapshot)

    loaded = load_snapshot(path)
    model = to_detector(loaded).smoother.last_model_
    assert model is not None
    assert snapshot.hmm_covars_diag is not None
    np.testing.assert_array_equal(model._covars_, snapshot.hmm_covars_diag)

    # Mutate the stored diagonals on disk: the reconstructed model must carry
    # the MUTATED values (proves the stored array is consumed, not recomputed).
    with np.load(path, allow_pickle=False) as data:
        members = {name: data[name] for name in data.files}
    members["hmm_covars_diag"] = members["hmm_covars_diag"] * 100.0 + 3.0
    with open(path, "wb") as f:
        np.savez(f, allow_pickle=False, **members)
    mutated = to_detector(load_snapshot(path)).smoother.last_model_
    assert mutated is not None
    np.testing.assert_array_equal(mutated._covars_, members["hmm_covars_diag"])


def test_partial_valid_mask_fit_and_round_trip(tmp_path: Path) -> None:
    """The operationally-critical path the original fixtures skipped:
    invalid windows must come back as -1 in full_labels, never enter
    any reference matrix, and round-trip apply parity must hold on valid rows."""
    base = _two_state_prepared()
    valid_mask = base.valid_mask.copy()
    invalid_rows = np.array([3, 17, 44, 90, 121, 160, 201])
    valid_mask[invalid_rows] = False
    features = base.features.copy()
    features[invalid_rows] = np.nan
    prepared = dataclasses.replace(base, features=features, valid_mask=valid_mask)

    snapshot, full_labels = _fit(prepared)

    assert np.all(full_labels[invalid_rows] == -1)
    assert np.all(full_labels[valid_mask] != -1)
    for ref in snapshot.references.values():
        assert np.isfinite(ref).all()

    path = tmp_path / "snap.npz"
    save_snapshot(path, snapshot)
    loaded = load_snapshot(path)
    features_valid = prepared.features[valid_mask]
    grid = _valid_grid(prepared)
    orig = to_detector(snapshot).apply(features_valid, grid).frame_labels
    round_tripped = to_detector(loaded).apply(features_valid, grid).frame_labels
    np.testing.assert_array_equal(orig, round_tripped)
    np.testing.assert_array_equal(full_labels[valid_mask], orig)


def test_truncated_archive_raises_snapshot_level_value_error(tmp_path: Path) -> None:
    prepared = _two_state_prepared()
    snapshot, _ = _fit(prepared)
    path = tmp_path / "snap.npz"
    save_snapshot(path, snapshot)

    with np.load(path, allow_pickle=False) as data:
        members = {name: data[name] for name in data.files}
    dropped = next(name for name in members if name.startswith("ref__"))
    del members[dropped]
    with open(path, "wb") as f:
        np.savez(f, allow_pickle=False, **members)

    with pytest.raises(ValueError, match="corrupt or truncated"):
        load_snapshot(path)


# ---------------------------------------------------------------------------
# 11. fit_snapshot_from_parts (package-7 Task 3, spec A3.11) + save_snapshot
#     provenance kwarg -- the pooled-artifact assembly path
# ---------------------------------------------------------------------------


def _detector_and_parts() -> tuple[
    PreparedRun,
    FittedDetector,
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, ConformalThreshold],
]:
    """A fitted detector plus hand-derived scoring parts -- what a pooled caller
    (`scripts/run_step2.py`'s cross-day-pooled protocol) hands to
    `fit_snapshot_from_parts` after deriving references/scores/thresholds itself."""
    prepared = _two_state_prepared()
    detector, det = FittedDetector.fit(prepared.features, prepared.grid, _cfg().detect)
    labels = det.frame_labels
    references: dict[int, np.ndarray] = {}
    calibration_scores: dict[int, np.ndarray] = {}
    thresholds: dict[int, ConformalThreshold] = {}
    for label in sorted(int(v) for v in np.unique(labels)):
        idx = np.flatnonzero(labels == label)
        reference = prepared.features[idx[:40]]
        scores = KnnScorer(k=1, metric="cosine").fit(reference).score(
            prepared.features[idx[40:80]]
        )
        references[label] = reference
        calibration_scores[label] = scores
        thresholds[label] = calibrate(scores, 0.05)
    return prepared, detector, references, calibration_scores, thresholds


def _from_parts(
    detector: FittedDetector,
    references: dict[int, np.ndarray],
    calibration_scores: dict[int, np.ndarray],
    thresholds: dict[int, ConformalThreshold],
    **overrides: object,
) -> MonitorSnapshot:
    kwargs: dict = {
        "scorer": "knn",
        "alpha": 0.05,
        "min_ref": 20,
        "calibration_frac": 0.5,
        "seed": 7,
        "variant": "fusion",
        "fit_run": "pool:day-a,day-b",
        "feature_names": ["f0", "f1"],
        "checkpoints": {},
    }
    kwargs.update(overrides)
    return fit_snapshot_from_parts(
        detector, references, calibration_scores, thresholds, **kwargs
    )


def test_from_parts_round_trip_apply_and_scorer_parity(tmp_path: Path) -> None:
    prepared, detector, references, cal_scores, thresholds = _detector_and_parts()
    snapshot = _from_parts(detector, references, cal_scores, thresholds)

    assert snapshot.fit_run == "pool:day-a,day-b"
    assert snapshot.format_version == SNAPSHOT_FORMAT_VERSION
    assert set(snapshot.thresholds) == set(references)

    path = tmp_path / "parts.npz"
    save_snapshot(path, snapshot)
    loaded = load_snapshot(path)

    # Detector parity: the round-tripped from-parts snapshot labels exactly like
    # the live detector it was assembled from.
    np.testing.assert_array_equal(
        to_detector(loaded).apply(prepared.features, prepared.grid).frame_labels,
        detector.apply(prepared.features, prepared.grid).frame_labels,
    )
    # Scorer parity: per-label runtime scorers refit from the stored references
    # score bitwise like scorers fit directly on the parts.
    for label, reference in references.items():
        np.testing.assert_array_equal(
            scorer_for_label(loaded, label).score(prepared.features),
            KnnScorer(k=1, metric="cosine").fit(reference).score(prepared.features),
        )
        assert loaded.thresholds[label] == thresholds[label]
        np.testing.assert_array_equal(
            loaded.calibration_scores[label], cal_scores[label]
        )


def test_from_parts_key_set_mismatch_raises() -> None:
    _prepared, detector, references, cal_scores, thresholds = _detector_and_parts()
    some_label = next(iter(cal_scores))
    broken = {k: v for k, v in cal_scores.items() if k != some_label}
    with pytest.raises(ValueError, match="label sets disagree"):
        _from_parts(detector, references, broken, thresholds)

    broken_thresholds = {k: v for k, v in thresholds.items() if k != some_label}
    with pytest.raises(ValueError, match="label sets disagree"):
        _from_parts(detector, references, cal_scores, broken_thresholds)


def test_from_parts_scorer_whitelist() -> None:
    _prepared, detector, references, cal_scores, thresholds = _detector_and_parts()
    with pytest.raises(ValueError) as exc_info:
        _from_parts(detector, references, cal_scores, thresholds, scorer="ocsvm")
    message = str(exc_info.value)
    assert "ocsvm" in message
    assert "knn" in message and "mahalanobis" in message


def test_from_parts_reference_geometry_refusals() -> None:
    _prepared, detector, references, cal_scores, thresholds = _detector_and_parts()
    some_label = next(iter(references))

    empty = {**references, some_label: np.empty((0, 2))}
    with pytest.raises(ValueError, match="non-empty"):
        _from_parts(detector, empty, cal_scores, thresholds)

    one_d = {**references, some_label: np.ones(5)}
    with pytest.raises(ValueError, match="2-D"):
        _from_parts(detector, one_d, cal_scores, thresholds)

    wide = {**references, some_label: np.ones((5, 3))}
    with pytest.raises(ValueError, match="feature_names"):
        _from_parts(detector, wide, cal_scores, thresholds)


def test_save_snapshot_provenance_round_trip(tmp_path: Path) -> None:
    prepared = _two_state_prepared()
    snapshot, _ = _fit(prepared)
    provenance = {
        "protocol": "cross-day-pooled",
        "fit_runs": ["day-a", "day-b"],
        "pool_members": {"day-a": {"n_windows": 30}, "day-b": {"n_windows": 25}},
    }

    path = tmp_path / "prov.npz"
    save_snapshot(path, snapshot, provenance=provenance)

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"][0]))
    assert meta["provenance"] == provenance
    assert meta["format_version"] == SNAPSHOT_FORMAT_VERSION == 2  # additive metadata
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["provenance"] == provenance

    # Provenance never affects loading -- the scoring/detector halves round-trip
    # exactly as without it.
    loaded = load_snapshot(path)
    assert loaded.thresholds == snapshot.thresholds
    assert loaded.fit_run == snapshot.fit_run

    # Absent provenance (the default) -> byte-level baseline unchanged: no
    # "provenance" key anywhere.
    bare_path = tmp_path / "bare.npz"
    save_snapshot(bare_path, snapshot)
    with np.load(bare_path, allow_pickle=False) as data:
        bare_meta = json.loads(str(data["meta"][0]))
    assert "provenance" not in bare_meta
    assert "provenance" not in json.loads(bare_path.with_suffix(".json").read_text())


# ---------------------------------------------------------------------------
# 12. Format v2: session stats (package-7 Task 4, spec D3/A3.5) -- round trips
#     with and without stats, v1-FILE compatibility, geometry refusal
# ---------------------------------------------------------------------------


def _session_stats_2f() -> SessionStats:
    return SessionStats(
        center=np.array([0.5, -1.25]),
        scale=np.array([2.0, 0.75]),
        n_windows=42,
        norm_minutes=20.0,
    )


def test_v2_round_trip_with_session_stats(tmp_path: Path) -> None:
    prepared, detector, references, cal_scores, thresholds = _detector_and_parts()
    stats = _session_stats_2f()
    snapshot = _from_parts(
        detector, references, cal_scores, thresholds, session_stats=stats
    )
    assert snapshot.format_version == SNAPSHOT_FORMAT_VERSION == 2
    assert snapshot.session_stats is stats

    path = tmp_path / "v2_stats.npz"
    save_snapshot(path, snapshot)

    with np.load(path, allow_pickle=False) as data:
        assert "session_center" in data.files
        assert "session_scale" in data.files
        meta = json.loads(str(data["meta"][0]))
    assert meta["format_version"] == 2
    assert meta["session_stats"] == {"n_windows": 42, "norm_minutes": 20.0}

    loaded = load_snapshot(path)
    assert loaded.session_stats is not None
    np.testing.assert_array_equal(loaded.session_stats.center, stats.center)
    np.testing.assert_array_equal(loaded.session_stats.scale, stats.scale)
    assert loaded.session_stats.n_windows == 42
    assert loaded.session_stats.norm_minutes == 20.0
    # The scoring/detector halves are untouched by the stats members.
    assert loaded.thresholds == snapshot.thresholds
    np.testing.assert_array_equal(
        to_detector(loaded).apply(prepared.features, prepared.grid).frame_labels,
        detector.apply(prepared.features, prepared.grid).frame_labels,
    )


def test_v2_without_stats_carries_no_session_members(tmp_path: Path) -> None:
    prepared = _two_state_prepared()
    snapshot, _ = _fit(prepared)
    assert snapshot.session_stats is None  # fit_snapshot never fits session stats
    assert snapshot.level_recal_medians is None  # nor level-recal medians (P8 T7)
    assert snapshot.format_version == 2  # NEW saves are v2 either way (Task 4)

    path = tmp_path / "v2_bare.npz"
    save_snapshot(path, snapshot)
    with np.load(path, allow_pickle=False) as data:
        assert not any(name.startswith("session_") for name in data.files)
        meta = json.loads(str(data["meta"][0]))
    assert meta["format_version"] == 2
    assert "session_stats" not in meta
    assert "level_recal_medians" not in meta

    loaded = load_snapshot(path)
    assert loaded.session_stats is None
    assert loaded.level_recal_medians is None
    assert loaded.format_version == 2


def test_v1_file_loads_with_session_stats_none(tmp_path: Path) -> None:
    """A pre-package-7 snapshot FILE (format_version 1, no session members) must
    keep loading -- the reader accepts {1, 2}; refusing v1 is the MONITOR's job,
    and only under --session-norm (tests/test_monitor_cli.py)."""
    prepared = _two_state_prepared()
    snapshot, _ = _fit(prepared)
    path = tmp_path / "v1_file.npz"
    save_snapshot(path, snapshot)
    _rewrite_meta(path, format_version=1)  # simulate the pre-v2 on-disk artifact

    loaded = load_snapshot(path)
    assert loaded.format_version == 1
    assert loaded.session_stats is None
    assert loaded.thresholds == snapshot.thresholds
    features_valid = prepared.features[prepared.valid_mask]
    for label in snapshot.thresholds:
        np.testing.assert_array_equal(
            scorer_for_label(loaded, label).score(features_valid),
            scorer_for_label(snapshot, label).score(features_valid),
        )


def test_from_parts_session_stats_geometry_refusal() -> None:
    _prepared, detector, references, cal_scores, thresholds = _detector_and_parts()
    three_wide = SessionStats(
        center=np.zeros(3), scale=np.ones(3), n_windows=5, norm_minutes=20.0
    )
    with pytest.raises(ValueError, match="session"):
        _from_parts(
            detector, references, cal_scores, thresholds, session_stats=three_wide
        )


def test_scorer_for_label_session_stats_transforms_the_reference() -> None:
    """`scorer_for_label(..., session_stats=...)` must fit on the session-normalized
    reference (the monitor's reference-side transform seam, D3/A3.5) -- parity with
    a hand-transformed KnnScorer, bitwise."""
    prepared = _two_state_prepared()
    snapshot, _ = _fit(prepared)
    stats = _session_stats_2f()
    query = prepared.features[prepared.valid_mask][:50]
    for label, reference in snapshot.references.items():
        expected = (
            KnnScorer(k=1, metric="cosine")
            .fit(apply_session_norm(reference, stats))
            .score(query)
        )
        got = scorer_for_label(snapshot, label, session_stats=stats).score(query)
        np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# 13. Format v2 (cont'd): level-recal medians (package-8 Task 7, spec
#     D2/A1.4/A1.10/A1.11) -- OPTIONAL v2 member, NO version bump, mutually
#     exclusive with session_stats by fit path, name-keyed geometry guard.
# ---------------------------------------------------------------------------


def test_snapshot_round_trips_level_recal_medians(tmp_path: Path) -> None:
    _prepared, detector, references, cal_scores, thresholds = _detector_and_parts()
    medians = {"f0": -40.0, "f1": -35.0}  # feature_names = ["f0", "f1"] (_from_parts default)
    snapshot = _from_parts(
        detector, references, cal_scores, thresholds, level_recal_medians=medians
    )
    assert snapshot.format_version == SNAPSHOT_FORMAT_VERSION == 2  # no version bump (A1.10)
    assert snapshot.level_recal_medians == medians
    assert snapshot.session_stats is None

    path = tmp_path / "v2_level_recal.npz"
    save_snapshot(path, snapshot)

    with np.load(path, allow_pickle=False) as data:
        # No new npz array -- the small dict lives entirely in the meta JSON
        # (module docstring / A1.10: "no new npz array").
        assert not any(name.startswith("session_") for name in data.files)
        meta = json.loads(str(data["meta"][0]))
    assert meta["level_recal_medians"] == medians

    got = load_snapshot(path)
    assert got.level_recal_medians == medians
    assert got.session_stats is None
    # The scoring/detector halves are untouched by the medians member.
    assert got.thresholds == snapshot.thresholds


def test_level_recal_and_session_stats_are_mutually_exclusive() -> None:
    """A1.10: session_stats and level_recal_medians are two different fit paths
    (session-normalization vs level-only recalibration) and must never both be
    stored in one snapshot."""
    _prepared, detector, references, cal_scores, thresholds = _detector_and_parts()
    with pytest.raises(ValueError, match="mutually exclusive"):
        _from_parts(
            detector, references, cal_scores, thresholds,
            session_stats=_session_stats_2f(),
            level_recal_medians={"f0": 1.0, "f1": -1.0},
        )


def test_from_parts_level_recal_medians_geometry_refusal() -> None:
    """Every level_recal_medians key must be one of feature_names (A1.10's
    geometry guard, the same posture as the session-stats width check) -- a
    stray key would silently promise an anchor the snapshot cannot back."""
    _prepared, detector, references, cal_scores, thresholds = _detector_and_parts()
    with pytest.raises(ValueError, match="feature_names"):
        _from_parts(
            detector, references, cal_scores, thresholds,
            level_recal_medians={"f0": 1.0, "not_a_real_column": 2.0},
        )


# ---------------------------------------------------------------------------
# 14. Format v2 (cont'd): state_names (Step-2 package-9 Task 2, spec D2/A1.8) --
#     OPTIONAL v2 member, NO version bump, keyed over FITTED ids (not the
#     threshold-label subset), NO mutual-exclusivity with level_recal_medians.
# ---------------------------------------------------------------------------


def test_v2_round_trip_with_state_names(tmp_path: Path) -> None:
    prepared, detector, references, cal_scores, thresholds = _detector_and_parts()
    fitted = [int(i) for i in np.asarray(detector.smoother._fitted_ids)]
    names = {fitted[0]: "turbine"}  # a subset of fitted ids is legal (orthogonality)
    snapshot = _from_parts(detector, references, cal_scores, thresholds, state_names=names)
    assert snapshot.state_names == names
    assert snapshot.session_stats is None
    path = tmp_path / "names.npz"
    save_snapshot(path, snapshot)
    meta = json.loads((path.with_suffix(".json")).read_text())
    assert meta["state_names"] == {str(fitted[0]): "turbine"}
    loaded = load_snapshot(path)
    assert loaded.state_names == names  # int keys restored
    assert loaded.format_version == SNAPSHOT_FORMAT_VERSION  # no bump


def test_state_names_keys_must_be_fitted_ids(tmp_path: Path) -> None:
    _p, detector, references, cal_scores, thresholds = _detector_and_parts()
    bad = 9999  # never a fitted id
    with pytest.raises(ValueError, match="state_names"):
        _from_parts(detector, references, cal_scores, thresholds, state_names={bad: "x"})


def test_state_names_coexists_with_level_recal_medians(tmp_path: Path) -> None:
    """NO mutual-exclusivity (A1.5): state_names is a naming layer, not a transform."""
    _p, detector, references, cal_scores, thresholds = _detector_and_parts()
    fitted = [int(i) for i in np.asarray(detector.smoother._fitted_ids)]
    snap = _from_parts(
        detector, references, cal_scores, thresholds,
        state_names={fitted[0]: "turbine"},
        level_recal_medians={"f0": -40.0},
    )
    assert snap.state_names == {fitted[0]: "turbine"}
    assert snap.level_recal_medians == {"f0": -40.0}


def test_state_names_coexists_with_session_stats(tmp_path: Path) -> None:
    """NO mutual-exclusivity (A1.5): state_names is a naming layer, not a
    transform -- mirrors test_state_names_coexists_with_level_recal_medians for
    the OTHER v2 scoring-space member state_names must coexist with (P9
    hardening T2a)."""
    _p, detector, references, cal_scores, thresholds = _detector_and_parts()
    fitted = [int(i) for i in np.asarray(detector.smoother._fitted_ids)]
    stats = _session_stats_2f()
    snap = _from_parts(
        detector, references, cal_scores, thresholds,
        state_names={fitted[0]: "turbine"},
        session_stats=stats,
    )
    assert snap.state_names == {fitted[0]: "turbine"}
    assert snap.session_stats is stats
