"""Tests for `rowii.anomaly.normalize`: label-free per-session robust normalization --
median/MAD*1.4826 stats from the first `norm_minutes` of a run's VALID windows, scale floored at the
house 1e-8 (`rowii.adapt.target_windows._standardize_1d`'s convention), applied to
the SCORING space only (the detector always consumes RAW features -- wiring tested
in `tests/test_monitor_cli.py` / `tests/test_step2_pooled_cli.py`; this file pins
the stats themselves).

The first-N membership rule is by window START: window `i` qualifies iff its grid
start offset `i * window_ns` is strictly below `norm_minutes * 60 * 1e9` ns, AND
the window is valid. `fit_pool_stats` is the pooled-snapshot companion: pool-global
center/scale over an ALREADY-STACKED
matrix, carrying the `norm_minutes == 0.0` sentinel that marks stats NOT derived
from any first-N prefix.
"""
from __future__ import annotations

import numpy as np
import pytest

from rowii.anomaly.normalize import (
    SessionStats,
    apply_session_norm,
    fit_pool_stats,
    fit_session_stats,
)
from rowii.signals.windows import WindowGrid

_MINUTE_NS = 60_000_000_000
"""One-minute windows so `norm_minutes` maps 1:1 onto window indices."""


def _grid(n_windows: int, *, t0_ns: int = 5_000_000_000) -> WindowGrid:
    """A non-zero t0 on purpose: the first-N rule is about OFFSETS from t0, and a
    zero t0 could not distinguish `t0 + i * window_ns < t0 + cutoff` from the buggy
    absolute-time reading `t0 + i * window_ns < cutoff`."""
    return WindowGrid(t0_ns=t0_ns, window_ns=_MINUTE_NS, n_windows=n_windows)


# ---------------------------------------------------------------------------
# 1. Median/MAD correctness + the 1e-8 scale floor
# ---------------------------------------------------------------------------


def test_center_is_per_column_median_and_scale_is_mad_scaled() -> None:
    # Column 0: median 3, MAD 1 (abs devs [2, 1, 0, 1, 97] -- the 100 outlier must
    # NOT drag the robust stats). Column 1: median 10, MAD 2.
    features = np.array(
        [
            [1.0, 8.0],
            [2.0, 9.0],
            [3.0, 10.0],
            [4.0, 12.0],
            [100.0, 13.0],
        ]
    )
    grid = _grid(5)
    stats = fit_session_stats(
        features, np.ones(5, dtype=bool), grid, norm_minutes=60.0
    )
    np.testing.assert_array_equal(stats.center, np.array([3.0, 10.0]))
    np.testing.assert_array_equal(stats.scale, np.array([1.0 * 1.4826, 2.0 * 1.4826]))
    assert stats.n_windows == 5
    assert stats.norm_minutes == 60.0
    assert stats.center.dtype == np.float64
    assert stats.scale.dtype == np.float64


def test_scale_floor_1e_8_on_constant_column() -> None:
    features = np.column_stack([np.full(6, 7.0), np.arange(6, dtype=np.float64)])
    stats = fit_session_stats(
        features, np.ones(6, dtype=bool), _grid(6), norm_minutes=60.0
    )
    # Constant column: MAD 0 -> the house 1e-8 floor, never a zero divisor.
    assert stats.scale[0] == 1e-8
    assert stats.scale[1] > 1e-8
    normed = apply_session_norm(features, stats)
    assert np.isfinite(normed).all()


# ---------------------------------------------------------------------------
# 2. First-N membership: window START offsets, valid rows only
# ---------------------------------------------------------------------------


def test_first_n_boundary_is_window_start_exclusive() -> None:
    # 10 one-minute windows, norm_minutes=5: starts at offsets 0..4 min qualify;
    # window 5 STARTS exactly at the 5-minute mark and must be excluded (strict <).
    features = np.arange(10, dtype=np.float64).reshape(10, 1)
    features[5:] = 1e6  # any leakage of rows >= 5 would wreck the median
    stats = fit_session_stats(
        features, np.ones(10, dtype=bool), _grid(10), norm_minutes=5.0
    )
    assert stats.n_windows == 5
    assert stats.center[0] == 2.0  # median of rows 0..4 only


def test_only_valid_rows_enter_the_stats() -> None:
    features = np.arange(10, dtype=np.float64).reshape(10, 1)
    valid = np.ones(10, dtype=bool)
    valid[2] = False
    features[2] = np.nan  # invalid rows are NaN in real PreparedRuns
    stats = fit_session_stats(features, valid, _grid(10), norm_minutes=5.0)
    assert stats.n_windows == 4  # rows {0, 1, 3, 4}
    assert stats.center[0] == 2.0  # median of [0, 1, 3, 4]
    assert np.isfinite(stats.center).all() and np.isfinite(stats.scale).all()


def test_zero_qualifying_rows_raises_value_error() -> None:
    features = np.arange(10, dtype=np.float64).reshape(10, 1)
    valid = np.ones(10, dtype=bool)
    valid[:5] = False  # the whole first-5-minute prefix is invalid
    with pytest.raises(ValueError, match="zero"):
        fit_session_stats(features, valid, _grid(10), norm_minutes=5.0)


@pytest.mark.parametrize("bad_minutes", [0.0, -3.0])
def test_nonpositive_norm_minutes_raises(bad_minutes: float) -> None:
    # 0.0 is the POOL-GLOBAL sentinel (`fit_pool_stats`), never a valid prefix
    # length here -- refusing it keeps the sentinel unambiguous.
    features = np.ones((4, 2))
    with pytest.raises(ValueError, match="norm_minutes"):
        fit_session_stats(
            features, np.ones(4, dtype=bool), _grid(4), norm_minutes=bad_minutes
        )


def test_shape_mismatch_raises() -> None:
    features = np.ones((4, 2))
    with pytest.raises(ValueError, match="4"):
        fit_session_stats(
            features, np.ones(5, dtype=bool), _grid(4), norm_minutes=5.0
        )
    with pytest.raises(ValueError, match="6"):
        fit_session_stats(
            features, np.ones(4, dtype=bool), _grid(6), norm_minutes=5.0
        )


# ---------------------------------------------------------------------------
# 3. Docstring pins (cheap doc contract)
# ---------------------------------------------------------------------------


def test_docstring_documents_state_mix_confound_and_deployment_rationale() -> None:
    doc = fit_session_stats.__doc__
    assert doc is not None
    # The state-mix confound (measured instance named) + the N-sweep as
    # the sensitivity probe + the deployment-realism rationale must be IN the
    # docstring -- callers must not be able to miss the caveat.
    assert "confound" in doc
    assert "290626" in doc
    assert "N-sweep" in doc or "A2.2" in doc
    assert "label" in doc  # label-free / deployment-realistic rationale


# ---------------------------------------------------------------------------
# 4. apply_session_norm: float64 copy, exact arithmetic, geometry guard
# ---------------------------------------------------------------------------


def test_apply_session_norm_is_float64_copy_with_exact_arithmetic() -> None:
    stats = SessionStats(
        center=np.array([1.0, -2.0]),
        scale=np.array([2.0, 0.5]),
        n_windows=9,
        norm_minutes=20.0,
    )
    features = np.array([[3.0, 0.0], [1.0, -2.0]], dtype=np.float32)
    out = apply_session_norm(features, stats)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, np.array([[1.0, 4.0], [0.0, 0.0]]))
    out[0, 0] = 999.0  # a COPY: mutating the output must not touch the input
    assert features[0, 0] == np.float32(3.0)


def test_apply_session_norm_width_mismatch_raises() -> None:
    stats = SessionStats(
        center=np.zeros(2), scale=np.ones(2), n_windows=3, norm_minutes=20.0
    )
    with pytest.raises(ValueError, match="2"):
        apply_session_norm(np.ones((4, 3)), stats)


# ---------------------------------------------------------------------------
# 5. fit_pool_stats: the pooled-snapshot sentinel constructor
# ---------------------------------------------------------------------------


def test_fit_pool_stats_sentinel_and_values() -> None:
    rows = np.array([[1.0, 8.0], [2.0, 9.0], [3.0, 10.0], [4.0, 12.0], [100.0, 13.0]])
    stats = fit_pool_stats(rows)
    assert stats.norm_minutes == 0.0  # the pool-global sentinel
    assert stats.n_windows == 5
    np.testing.assert_array_equal(stats.center, np.array([3.0, 10.0]))
    np.testing.assert_array_equal(stats.scale, np.array([1.0 * 1.4826, 2.0 * 1.4826]))


def test_fit_pool_stats_refuses_empty_and_non_finite() -> None:
    with pytest.raises(ValueError, match="zero"):
        fit_pool_stats(np.empty((0, 2)))
    bad = np.ones((3, 2))
    bad[1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_pool_stats(bad)
