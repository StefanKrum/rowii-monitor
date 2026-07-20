"""Tests for `FittedDetector.fit_pooled` (Step-2 package-7 Task 2, design spec
`docs/superpowers/specs/2026-07-18-step2-package7-robustness-design.md` Amendment
A3.4): pooled KMeans emissions with per-run Viterbi decode and NO cross-run EM
chain.

Fixture: two synthetic runs with a DISJOINT extra mode -- run A carries gaussian
blobs 1+2, run B carries blobs 2+3 -- so blob 3 exists ONLY in run B. The central
A3.4 property under test ("pump owns a cluster"): a detector fit on the POOLED
features must give blob 3 its own label id, and `apply` on run B must label blob-3
windows with that id while blob 2 keeps ONE shared id across both runs. Blobs are
tight (std 0.3) at centers >= 12 apart, laid out in contiguous 30-window blocks so
neither the sticky Viterbi decode nor the 3-window duration filter can flip the
expected labels -- every ARI assertion below is exact (1.0) by construction.

Row-order tests (T1-review forward finding): sklearn's k-means++ seeding is not
guaranteed bit-identical under row permutation even with a fixed `random_state`.
Two calls on the SAME array must be bit-identical; a permuted copy may yield a
different but EQUALLY VALID clustering -- asserted as equal ARI vs the fixture's
ground truth, deliberately NOT as equal labels.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from rowii.anomaly.conformal import calibrate
from rowii.config import Config, DetectConfig
from rowii.runtime.snapshot import (
    SNAPSHOT_FORMAT_VERSION,
    MonitorSnapshot,
    _hmm_arrays,
    load_snapshot,
    save_snapshot,
    to_detector,
)
from rowii.signals.features import apply_zscore, zscore_stats
from rowii.signals.windows import WindowGrid
from rowii.state.cluster import KMeansClusterer
from rowii.state.detect import FittedDetector
from rowii.state.smooth import _init_means_covars, _sticky_transmat

_BLOCK = 30
"""Windows per contiguous blob block -- 10x the 3-window duration-filter floor at
`min_dwell_s=3.0` / 1-second windows, so filtering can never merge a block away."""

_CENTERS: dict[int, tuple[float, float]] = {
    1: (0.0, 0.0),
    2: (12.0, 12.0),
    3: (-12.0, 18.0),
}
"""Blob id -> 2-D center. Pairwise separation >= 12 at blob std 0.3: KMeans(k=3,
n_init=10) recovers exactly this partition from any init, which is what makes the
permutation test's "equally valid" assertion exact."""


class _PooledFixture(NamedTuple):
    feat_a: np.ndarray
    gt_a: np.ndarray
    grid_a: WindowGrid
    feat_b: np.ndarray
    gt_b: np.ndarray
    grid_b: WindowGrid
    pooled: np.ndarray
    """`vstack([feat_a, feat_b])` -- the deterministic stacked row order
    `PoolResult.features` would deliver for insertion order (run A, run B)."""


def _run(blob_ids: list[int], seed: int) -> tuple[np.ndarray, np.ndarray, WindowGrid]:
    """One synthetic run: contiguous `_BLOCK`-window blocks, one per entry of
    *blob_ids*, plus the ground-truth blob id per window."""
    rng = np.random.default_rng(seed)
    feats: list[np.ndarray] = []
    gt: list[np.ndarray] = []
    for blob_id in blob_ids:
        feats.append(rng.normal(_CENTERS[blob_id], 0.3, (_BLOCK, 2)))
        gt.append(np.full(_BLOCK, blob_id, dtype=np.int64))
    features = np.vstack(feats)
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=len(features))
    return features, np.concatenate(gt), grid


def _fixture() -> _PooledFixture:
    feat_a, gt_a, grid_a = _run([1, 2, 1, 2], seed=0)
    feat_b, gt_b, grid_b = _run([2, 3, 2, 3], seed=1)
    return _PooledFixture(
        feat_a, gt_a, grid_a, feat_b, gt_b, grid_b, np.vstack([feat_a, feat_b])
    )


def _cfg() -> Config:
    """Constructed directly (not `load_config()`) so ambient env vars / .env files
    can never leak into these library-level tests (`test_runtime_snapshot` pattern)."""
    return Config(
        data_root=Path("/unused"),
        results_root=Path("/unused"),
        detect=DetectConfig(n_states=2, min_dwell_s=3.0),
    )


def _single_label(frame_labels: np.ndarray, mask: np.ndarray) -> int:
    """The one label on the masked windows -- fails if the windows disagree."""
    values = np.unique(frame_labels[mask])
    assert len(values) == 1, f"expected one label on mask, got {values.tolist()}"
    return int(values[0])


# ---------------------------------------------------------------------------
# 1. The A3.4 property: pooled fit recovers all 3 modes; pump owns a cluster
# ---------------------------------------------------------------------------


def test_fit_pooled_recovers_disjoint_modes_pump_owns_a_cluster() -> None:
    fx = _fixture()
    detector = FittedDetector.fit_pooled(fx.pooled, _cfg(), k=3)

    assert detector.k == 3
    assert detector.smoother._fitted_ids is not None
    np.testing.assert_array_equal(
        detector.smoother._fitted_ids, np.arange(3, dtype=np.int64)
    )
    assert detector.smoother._component_to_id == {0: 0, 1: 1, 2: 2}

    result_a = detector.apply(fx.feat_a, fx.grid_a)
    result_b = detector.apply(fx.feat_b, fx.grid_b)

    c1 = _single_label(result_a.frame_labels, fx.gt_a == 1)
    c2_on_a = _single_label(result_a.frame_labels, fx.gt_a == 2)
    c2_on_b = _single_label(result_b.frame_labels, fx.gt_b == 2)
    c3 = _single_label(result_b.frame_labels, fx.gt_b == 3)

    # The shared mode (blob 2) carries ONE pooled id on both runs ...
    assert c2_on_a == c2_on_b
    # ... and the run-B-only mode (blob 3, "the pump") owns its own third id.
    assert {c1, c2_on_a, c3} == {0, 1, 2}


# ---------------------------------------------------------------------------
# 2. z-score statistics come from the POOLED features
# ---------------------------------------------------------------------------


def test_zscore_stats_equal_pooled_stats() -> None:
    fx = _fixture()
    detector = FittedDetector.fit_pooled(fx.pooled, _cfg(), k=3)
    mean, std = zscore_stats(fx.pooled)
    np.testing.assert_array_equal(detector.mean, mean)
    np.testing.assert_array_equal(detector.std, std)


# ---------------------------------------------------------------------------
# 3. NO EM anywhere (A3.4): transmat is the sticky prior bitwise, and the
#    emissions are EXACTLY the pooled per-cluster moment estimates
# ---------------------------------------------------------------------------


def test_no_em_transmat_and_emissions_bitwise() -> None:
    fx = _fixture()
    cfg = _cfg()
    detector = FittedDetector.fit_pooled(fx.pooled, cfg, k=3)
    model = detector.smoother.last_model_
    assert model is not None

    np.testing.assert_array_equal(
        model.transmat_, _sticky_transmat(3, cfg.detect.self_transition)
    )
    np.testing.assert_array_equal(model.startprob_, np.full(3, 1.0 / 3))

    # The SHARP no-EM probe: `params="mc"` would leave the transmat fixed even
    # under `model.fit`, so the transmat check alone cannot detect EM -- but EM
    # would move means/covars away from the `_init_means_covars` moment
    # estimates. Recompute those from the pooled KMeans labels and demand
    # bitwise equality.
    mean, std = zscore_stats(fx.pooled)
    z = apply_zscore(fx.pooled, mean, std)
    labels = KMeansClusterer(
        n_clusters=3, random_seed=cfg.detect.random_seed
    ).fit_predict(z)
    z64 = np.asarray(z, dtype=np.float64)
    means, covars = _init_means_covars(z64, labels, 3, z64.shape[1])
    np.testing.assert_array_equal(model.means_, means)
    np.testing.assert_array_equal(model._covars_, covars)


# ---------------------------------------------------------------------------
# 4. Snapshot round trip through the EXISTING extraction path (apply parity)
# ---------------------------------------------------------------------------


def test_snapshot_round_trip_apply_parity(tmp_path: Path) -> None:
    fx = _fixture()
    detector = FittedDetector.fit_pooled(fx.pooled, _cfg(), k=3)
    smoother = detector.smoother
    assert smoother._fitted_ids is not None
    fitted_ids = np.asarray(smoother._fitted_ids, dtype=np.int64)

    # This call IS the component/id invariant verification: `_hmm_arrays`
    # raises RuntimeError unless component i == fitted_ids[i] -- which
    # fit_pooled's 0..k-1 label-id construction must satisfy.
    startprob, transmat, means, covars_diag = _hmm_arrays(smoother, fitted_ids)
    assert startprob is not None

    # Minimal-but-valid scoring half: apply parity is the property under test;
    # references/scores/thresholds only have to satisfy save_snapshot's
    # key-set and whitelist invariants.
    cal_scores = np.linspace(0.1, 1.0, 40)
    threshold = calibrate(cal_scores, 0.05)
    label_ids = [0, 1, 2]
    snapshot = MonitorSnapshot(
        mean=detector.mean,
        std=detector.std,
        fitted_ids=fitted_ids,
        hmm_startprob=startprob,
        hmm_transmat=transmat,
        hmm_means=means,
        hmm_covars_diag=covars_diag,
        min_dwell_s=detector.min_dwell_s,
        k=detector.k,
        self_transition=smoother.self_transition,
        random_seed=smoother.random_seed,
        references={i: fx.pooled[i * 5 : (i + 1) * 5] for i in label_ids},
        calibration_scores=dict.fromkeys(label_ids, cal_scores),
        thresholds=dict.fromkeys(label_ids, threshold),
        scorer="knn",
        alpha=0.05,
        min_ref=5,
        calibration_frac=0.5,
        seed=7,
        variant="fusion",
        feature_names=["f0", "f1"],
        fit_run="pool:run-a,run-b",
        checkpoints={},
        created_at=datetime.now(UTC).isoformat(),
        format_version=SNAPSHOT_FORMAT_VERSION,
    )

    path = tmp_path / "pooled_snap.npz"
    save_snapshot(path, snapshot)
    rebuilt = to_detector(load_snapshot(path))

    for feats, grid in ((fx.feat_a, fx.grid_a), (fx.feat_b, fx.grid_b)):
        np.testing.assert_array_equal(
            rebuilt.apply(feats, grid).frame_labels,
            detector.apply(feats, grid).frame_labels,
        )


# ---------------------------------------------------------------------------
# 5. Degenerate k=1 path (mirrors fit_decode's k<=1 contract, snapshot A1.2)
# ---------------------------------------------------------------------------


def test_degenerate_k1() -> None:
    fx = _fixture()
    detector = FittedDetector.fit_pooled(fx.pooled, _cfg(), k=1)

    assert detector.k == 1
    smoother = detector.smoother
    assert smoother.last_model_ is None
    assert smoother._fitted_ids is not None
    np.testing.assert_array_equal(smoother._fitted_ids, np.arange(1, dtype=np.int64))

    result = detector.apply(fx.feat_b, fx.grid_b)
    np.testing.assert_array_equal(
        result.frame_labels, np.zeros(len(fx.feat_b), dtype=np.int64)
    )
    assert result.k == 1

    # z-stats are still the pooled stats even on the degenerate path.
    mean, std = zscore_stats(fx.pooled)
    np.testing.assert_array_equal(detector.mean, mean)
    np.testing.assert_array_equal(detector.std, std)

    # Snapshot-representable in the A1.2 degenerate form: four Nones.
    assert _hmm_arrays(smoother, np.asarray(smoother._fitted_ids)) == (
        None,
        None,
        None,
        None,
    )


# ---------------------------------------------------------------------------
# 6. Row-order determinism: SAME array -> bit-identical detector
# ---------------------------------------------------------------------------


def test_same_array_is_bit_identical() -> None:
    fx = _fixture()
    cfg = _cfg()
    d1 = FittedDetector.fit_pooled(fx.pooled, cfg, k=3)
    d2 = FittedDetector.fit_pooled(fx.pooled.copy(), cfg, k=3)

    np.testing.assert_array_equal(d1.mean, d2.mean)
    np.testing.assert_array_equal(d1.std, d2.std)
    m1, m2 = d1.smoother.last_model_, d2.smoother.last_model_
    assert m1 is not None and m2 is not None
    np.testing.assert_array_equal(m1.startprob_, m2.startprob_)
    np.testing.assert_array_equal(m1.transmat_, m2.transmat_)
    np.testing.assert_array_equal(m1.means_, m2.means_)
    np.testing.assert_array_equal(m1._covars_, m2._covars_)
    np.testing.assert_array_equal(
        d1.apply(fx.feat_b, fx.grid_b).frame_labels,
        d2.apply(fx.feat_b, fx.grid_b).frame_labels,
    )


# ---------------------------------------------------------------------------
# 7. Row-order sensitivity, documented honestly: a permuted copy may relabel
#    clusters but must remain EQUALLY VALID (same ARI vs ground truth)
# ---------------------------------------------------------------------------


def test_permuted_rows_equally_valid_not_necessarily_identical() -> None:
    fx = _fixture()
    cfg = _cfg()
    detector = FittedDetector.fit_pooled(fx.pooled, cfg, k=3)
    rng = np.random.default_rng(123)
    permuted = fx.pooled[rng.permutation(len(fx.pooled))]
    detector_perm = FittedDetector.fit_pooled(permuted, cfg, k=3)

    labels = detector.apply(fx.feat_b, fx.grid_b).frame_labels
    labels_perm = detector_perm.apply(fx.feat_b, fx.grid_b).frame_labels

    # ARI is label-permutation invariant: both detectors must recover run B's
    # ground-truth partition perfectly, whatever cluster ids they hand out.
    # Deliberately NO assertion that labels == labels_perm (the docstring's
    # row-order caveat: k-means++ seeding consumes the rng in row order).
    ari = adjusted_rand_score(fx.gt_b, labels)
    ari_perm = adjusted_rand_score(fx.gt_b, labels_perm)
    assert ari == 1.0
    assert ari_perm == ari


# ---------------------------------------------------------------------------
# 8. Contract refusals
# ---------------------------------------------------------------------------


def test_unknown_clusterer_raises() -> None:
    fx = _fixture()
    with pytest.raises(ValueError, match="kmeans"):
        FittedDetector.fit_pooled(fx.pooled, _cfg(), k=3, clusterer="gmm")


def test_non_2d_features_raise() -> None:
    with pytest.raises(ValueError, match="2-D"):
        FittedDetector.fit_pooled(np.zeros(10), _cfg(), k=2)


def test_k_below_one_raises() -> None:
    fx = _fixture()
    with pytest.raises(ValueError, match="k"):
        FittedDetector.fit_pooled(fx.pooled, _cfg(), k=0)
