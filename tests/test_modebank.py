"""Tests for rowii.state.modebank (Package-8 D1 core, A1.5): per-family fit/score
shapes, argmin assignment, conformal rejection (incl. all-rejected), min_ref
floor + dropped modes, unknown/transition exclusion, global-standardization
storage, empty-bank degeneracy. Deterministic seeded blobs, no real data."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from rowii.state.modebank import _EXCLUDED_GT, ModeBank

_NAMES = [f"ch0_octave_{i}" for i in range(4)]  # 4 level-ish columns


def _two_mode(seed: int, n_per: int = 60):
    """Two well-separated GT modes ('turbine'@0, 'pump'@10) over 4 features."""
    rng = np.random.default_rng(seed)
    feats = np.vstack([rng.normal(0.0, 0.2, (n_per, 4)), rng.normal(10.0, 0.2, (n_per, 4))])
    labels = np.array(["turbine"] * n_per + ["pump"] * n_per, dtype=object)
    return feats, labels


@pytest.mark.parametrize("family", ["gaussian", "knn", "gmm"])
def test_fit_assign_recovers_two_modes(family: str) -> None:
    fit_f, fit_l = _two_mode(0)
    cal_f, cal_l = _two_mode(1)
    bank = ModeBank.fit(fit_f, fit_l, cal_f, cal_l, family=family, alpha=0.05, feature_names=_NAMES)
    assert set(bank.modes) == {"turbine", "pump"}
    query_f, query_l = _two_mode(2, n_per=40)
    a = bank.assign(query_f)
    assert a.labels.shape == (80,)
    assert a.scores.shape == (80, 2)
    # argmin assignment: >= 95% of windows land on their true GT mode.
    assert float(np.mean(a.labels == query_l)) >= 0.95


def test_gaussian_gmm_store_global_standardization_knn_does_not() -> None:
    fit_f, fit_l = _two_mode(0)
    cal_f, cal_l = _two_mode(1)
    g = ModeBank.fit(
        fit_f, fit_l, cal_f, cal_l, family="gaussian", alpha=0.05, feature_names=_NAMES
    )
    assert g.mean is not None and g.std is not None
    np.testing.assert_allclose(g.mean, fit_f[~np.isin(fit_l, _EXCLUDED_GT)].mean(axis=0))
    knn = ModeBank.fit(fit_f, fit_l, cal_f, cal_l, family="knn", alpha=0.05, feature_names=_NAMES)
    assert knn.mean is None and knn.std is None


def test_unknown_and_transition_excluded_from_training(caplog) -> None:
    fit_f, fit_l = _two_mode(0)
    fit_l = fit_l.copy()
    fit_l[:5] = "unknown"
    fit_l[5:10] = "transition"
    cal_f, cal_l = _two_mode(1)
    bank = ModeBank.fit(
        fit_f, fit_l, cal_f, cal_l, family="gaussian", alpha=0.05, feature_names=_NAMES
    )
    assert "unknown" not in bank.modes and "transition" not in bank.modes


def test_below_min_ref_mode_is_dropped_with_warning(caplog) -> None:
    fit_f, fit_l = _two_mode(0)
    # shrink 'pump' to 3 fit windows (< min_ref=20): dropped.
    keep = np.concatenate([np.arange(60), np.arange(60, 63)])
    fit_f, fit_l = fit_f[keep], fit_l[keep]
    cal_f, cal_l = _two_mode(1)
    with caplog.at_level(logging.WARNING):
        bank = ModeBank.fit(
            fit_f, fit_l, cal_f, cal_l, family="knn", alpha=0.05, feature_names=_NAMES
        )
    assert bank.modes == ["turbine"]
    assert bank.dropped_modes.get("pump") == 3
    assert any("pump" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def test_no_mode_fits_flags_a_far_out_window() -> None:
    fit_f, fit_l = _two_mode(0)
    cal_f, cal_l = _two_mode(1)
    bank = ModeBank.fit(
        fit_f, fit_l, cal_f, cal_l, family="gaussian", alpha=0.05, feature_names=_NAMES
    )
    # one window from each true mode (fits) + one wildly out-of-distribution window.
    query = np.vstack([np.zeros((1, 4)), np.full((1, 4), 10.0), np.full((1, 4), 500.0)])
    a = bank.assign(query)
    assert a.no_mode_fits[0] == False  # noqa: E712 -- fits 'turbine'
    assert a.no_mode_fits[1] == False  # noqa: E712 -- fits 'pump'
    assert a.no_mode_fits[2] == True   # noqa: E712 -- rejected by BOTH members


def test_unknown_family_and_width_mismatch_raise() -> None:
    fit_f, fit_l = _two_mode(0)
    cal_f, cal_l = _two_mode(1)
    with pytest.raises(ValueError, match="family"):
        ModeBank.fit(fit_f, fit_l, cal_f, cal_l, family="forest", alpha=0.05, feature_names=_NAMES)
    bank = ModeBank.fit(fit_f, fit_l, cal_f, cal_l, family="knn", alpha=0.05, feature_names=_NAMES)
    with pytest.raises(ValueError, match="width|column"):
        bank.assign(np.zeros((5, 3)))
