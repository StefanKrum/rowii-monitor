"""Tests for `rowii.runtime.snapshot` (Step-2 package-6 Task 1, design spec
`docs/superpowers/specs/2026-07-16-step2-package6-runtime-pillar3-design.md` D1 +
amendment A1): pickle-free `MonitorSnapshot` round-trip (bitwise apply/score parity),
format guards (version mismatch, runtime-scorer whitelist), the k<=1 degenerate
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
from rowii.anomaly.references import split_by_segments
from rowii.anomaly.scorers import KnnScorer
from rowii.anomaly.sweep import SweepConfig
from rowii.config import Config, DetectConfig
from rowii.pipeline import PreparedRun
from rowii.runtime.snapshot import (
    SNAPSHOT_FORMAT_VERSION,
    MonitorSnapshot,
    fit_snapshot,
    load_snapshot,
    save_snapshot,
    scorer_for_label,
    to_detector,
)
from rowii.signals.windows import WindowGrid

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

    assert meta["format_version"] == SNAPSHOT_FORMAT_VERSION == 1
    assert meta["scorer"] == "knn"
    assert meta["variant"] == "fusion"
    assert meta["fit_run"] == "fit-day"
    assert set(meta["thresholds"]) == {str(label) for label in snapshot.thresholds}

    # The human-readable sidecar carries the same metadata.
    sidecar = path.with_suffix(".json")
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text())["format_version"] == 1


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
