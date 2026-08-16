"""Tests for rowii.state.modebank: per-family fit/score
shapes, argmin assignment, conformal rejection (incl. all-rejected), min_ref
floor + dropped modes, unknown/transition exclusion (fit- AND calibration-side
invariance), low-confidence member visibility, global-standardization storage,
empty-bank degeneracy, knn min_ref-vs-k guard. Deterministic seeded blobs, no
real data."""
from __future__ import annotations

import logging
import math

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


def _excluded_rows(seed: int, n: int, label: str, center: float):
    """`n` rows GT-labeled *label*, tightly clustered around *center* -- for the
    unknown/transition-exclusion tests below: DISTINCT from `_two_mode`'s turbine
    (~0) and pump (~10) clusters AND from each other's *center*, so a broken
    `_EXCLUDED_GT` filter would place them as their OWN distinguishable bank
    member/threshold rather than coincidentally blending into a real mode."""
    rng = np.random.default_rng(seed)
    feats = rng.normal(center, 0.2, (n, 4))
    labels = np.array([label] * n, dtype=object)
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


def test_unknown_and_transition_excluded_from_training() -> None:
    # >= min_ref (25) rows per excluded label on BOTH sides -- neither the
    # fit-side min_ref=20 floor nor the calibration-side "zero rows" floor
    # would drop 'unknown'/'transition' on their own at this size, so only
    # `_EXCLUDED_GT` filtering can (a mutation removing it would let both
    # become real bank members with real thresholds, unlike the previous
    # 5-row version of this test, which the min_ref floor alone defeated).
    fit_f, fit_l = _two_mode(0)
    cal_f, cal_l = _two_mode(1)
    unk_fit_f, unk_fit_l = _excluded_rows(10, 25, "unknown", -500.0)
    tra_fit_f, tra_fit_l = _excluded_rows(11, 25, "transition", 500.0)
    unk_cal_f, unk_cal_l = _excluded_rows(12, 15, "unknown", -500.0)
    tra_cal_f, tra_cal_l = _excluded_rows(13, 15, "transition", 500.0)
    fit_f = np.vstack([fit_f, unk_fit_f, tra_fit_f])
    fit_l = np.concatenate([fit_l, unk_fit_l, tra_fit_l])
    cal_f = np.vstack([cal_f, unk_cal_f, tra_cal_f])
    cal_l = np.concatenate([cal_l, unk_cal_l, tra_cal_l])

    bank = ModeBank.fit(
        fit_f, fit_l, cal_f, cal_l, family="gaussian", alpha=0.05, feature_names=_NAMES
    )
    assert "unknown" not in bank.modes and "transition" not in bank.modes
    assert "unknown" not in bank.thresholds and "transition" not in bank.thresholds


def test_unknown_transition_calibration_side_is_invariant() -> None:
    # Identical FIT side for both banks (turbine/pump + >= min_ref excluded-
    # label rows, so 'unknown'/'transition' clear the FIT floor); only the
    # CALIBRATION side differs. If calibration-side exclusion were broken,
    # `bank_with_extra`'s 'unknown'/'transition' would gain non-empty
    # calibration support (unlike `bank_base`, whose calibration side never
    # mentions those labels at all) and surface as real members/thresholds --
    # so "no transition/unknown threshold in bank_with_extra" is the
    # mutation-catching assertion; the byte-identical turbine/pump thresholds
    # confirm the extra rows leave the real members completely untouched.
    fit_f, fit_l = _two_mode(0)
    unk_fit_f, unk_fit_l = _excluded_rows(10, 25, "unknown", -500.0)
    tra_fit_f, tra_fit_l = _excluded_rows(11, 25, "transition", 500.0)
    fit_f = np.vstack([fit_f, unk_fit_f, tra_fit_f])
    fit_l = np.concatenate([fit_l, unk_fit_l, tra_fit_l])

    cal_f, cal_l = _two_mode(1)
    unk_cal_f, unk_cal_l = _excluded_rows(12, 15, "unknown", -500.0)
    tra_cal_f, tra_cal_l = _excluded_rows(13, 15, "transition", 500.0)
    cal_f_extra = np.vstack([cal_f, unk_cal_f, tra_cal_f])
    cal_l_extra = np.concatenate([cal_l, unk_cal_l, tra_cal_l])

    bank_base = ModeBank.fit(
        fit_f, fit_l, cal_f, cal_l, family="gaussian", alpha=0.05, feature_names=_NAMES
    )
    bank_with_extra = ModeBank.fit(
        fit_f,
        fit_l,
        cal_f_extra,
        cal_l_extra,
        family="gaussian",
        alpha=0.05,
        feature_names=_NAMES,
    )

    assert "transition" not in bank_base.thresholds and "unknown" not in bank_base.thresholds
    assert (
        "transition" not in bank_with_extra.thresholds
        and "unknown" not in bank_with_extra.thresholds
    )
    base_thresholds = np.array([bank_base.thresholds[m].threshold for m in ("turbine", "pump")])
    extra_thresholds = np.array(
        [bank_with_extra.thresholds[m].threshold for m in ("turbine", "pump")]
    )
    np.testing.assert_array_equal(base_thresholds, extra_thresholds)


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


def test_low_confidence_member_is_warned_recorded_and_never_rejects(caplog) -> None:
    # 'turbine' keeps its full calibration side (60, ample); 'pump' is shrunk
    # to 5 (< 19 = ceil(1/alpha)-1 at alpha=0.05) -- still >= 1, so 'pump'
    # survives as a member, but `rowii.anomaly.conformal.calibrate` gives it
    # threshold=+inf, low_confidence=True.
    fit_f, fit_l = _two_mode(0)
    cal_f, cal_l = _two_mode(1)
    keep = np.concatenate([np.arange(60), np.arange(60, 65)])
    cal_f, cal_l = cal_f[keep], cal_l[keep]

    with caplog.at_level(logging.WARNING):
        bank = ModeBank.fit(
            fit_f, fit_l, cal_f, cal_l, family="gaussian", alpha=0.05, feature_names=_NAMES
        )

    assert set(bank.modes) == {"turbine", "pump"}
    assert bank.thresholds["pump"].low_confidence is True
    assert bank.thresholds["pump"].threshold == math.inf
    assert bank.thresholds["turbine"].low_confidence is False
    assert bank.low_confidence_modes == ("pump",)
    assert any("pump" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)

    # Pin the documented conservative-veto behaviour: even a window that is
    # wildly outside BOTH modes is not flagged, because 'pump's +inf
    # threshold can never be exceeded and no_mode_fits needs EVERY member to
    # reject (the fix makes the cause visible; it does not change this).
    query = np.full((1, 4), 5000.0)
    a = bank.assign(query)
    assert a.no_mode_fits[0] == False  # noqa: E712


def test_empty_bank(caplog) -> None:
    fit_f, fit_l = _two_mode(0)
    cal_f, cal_l = _two_mode(1)
    with caplog.at_level(logging.WARNING):
        bank = ModeBank.fit(
            fit_f,
            fit_l,
            cal_f,
            cal_l,
            family="gaussian",
            alpha=0.05,
            feature_names=_NAMES,
            min_ref=1_000_000,
        )

    assert bank.modes == []
    assert bank.low_confidence_modes == ()
    assert any(
        "empty bank" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    )

    a = bank.assign(np.zeros((5, 4)))
    assert a.labels.tolist() == [""] * 5
    assert a.scores.shape == (5, 0)
    assert a.scores.dtype == np.float64
    assert a.no_mode_fits.tolist() == [False] * 5


def test_knn_family_requires_min_ref_at_least_k() -> None:
    fit_f, fit_l = _two_mode(0)
    cal_f, cal_l = _two_mode(1)
    with pytest.raises(ValueError, match="min_ref"):
        ModeBank.fit(
            fit_f,
            fit_l,
            cal_f,
            cal_l,
            family="knn",
            alpha=0.05,
            feature_names=_NAMES,
            min_ref=3,
            k=5,
        )
