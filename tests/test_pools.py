"""Tests for `rowii.anomaly.pools`: multi-run training pools with leakage-safe sides
and A4.1/A4.2 coverage tables (package-7 plan `docs/superpowers/plans/
2026-07-18-step2-package7-robustness.md` Task 1, design spec D1 + A3.7 + A4.1/A4.2).
No real data -- hand-built `PreparedRun`s throughout (mirrors `tests/test_sweep.py`'s
fixture style).

The split-parity tests re-derive every expected window set INDEPENDENTLY by calling
`split_by_segments` by hand with the exact `run_sweep` convention (top split at
`(calibration_frac, seed)`, nested split of the calibration side at `(0.5, seed + 1)`)
-- deliberately NOT the `_cross_day_per_state_sweep` top-split-as-fit convention the
spec's A3.7 WARNING forbids copying. A non-default `SweepConfig(calibration_frac=0.4,
seed=11)` is used so an implementation that hardcodes the defaults (0.5 / 7) fails the
parity tests instead of passing by coincidence.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

import rowii.anomaly.pools as pools_module
from rowii.anomaly.pools import (
    PoolResult,
    build_pool,
    coverage_table,
    coverage_warnings,
)
from rowii.anomaly.references import SegmentSplit, split_by_segments
from rowii.anomaly.sweep import SweepConfig
from rowii.pipeline import PreparedRun
from rowii.signals.windows import WindowGrid

# ---------------------------------------------------------------------------
# Shared fixture plumbing (mirrors tests/test_sweep.py::_prepared_run)
# ---------------------------------------------------------------------------

_CFG = SweepConfig(calibration_frac=0.4, seed=11)
"""Deliberately non-default `calibration_frac`/`seed` (defaults are 0.5/7) -- see
module docstring."""

_SIDES = ("calibration", "fit", "conformal")


def _prepared_run(
    features: np.ndarray, segment_ids: np.ndarray, valid_mask: np.ndarray | None = None
) -> PreparedRun:
    """A `PreparedRun` whose `grid`/`feature_names` are cheap placeholders --
    `build_pool` only reads `features`, `valid_mask`, `segment_ids`."""
    if valid_mask is None:
        valid_mask = np.ones(features.shape[0], dtype=bool)
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=features.shape[0])
    return PreparedRun(
        features=features,
        grid=grid,
        valid_mask=valid_mask,
        feature_names=[f"f{i}" for i in range(features.shape[1])],
        segment_ids=segment_ids,
    )


def _pool_run(n_segments: int, windows_per_segment: int, seed: int, n_features: int = 3):
    """Random-featured run of `n_segments` x `windows_per_segment` windows -- random
    (seeded) features make every row unique, so the bitwise row-alignment test below
    can only pass when `run_index`/`window_index` genuinely map back to the right
    source rows."""
    rng = np.random.default_rng(seed)
    n = n_segments * windows_per_segment
    features = rng.normal(size=(n, n_features))
    segment_ids = np.repeat(np.arange(n_segments, dtype=np.int64), windows_per_segment)
    return _prepared_run(features, segment_ids)


def _three_run_pool() -> dict[str, PreparedRun]:
    """Three runs of unequal length; run-c additionally carries two INVALID windows and
    one uncovered (`segment_id=-1`) window, so the per-run splits must honour
    `valid_mask`/the `-1` sentinel exactly like `split_by_segments` itself does."""
    run_a = _pool_run(6, 10, seed=0)
    run_b = _pool_run(8, 6, seed=1)
    run_c = _pool_run(5, 9, seed=2)
    valid_c = run_c.valid_mask.copy()
    valid_c[[3, 17]] = False
    segment_ids_c = run_c.segment_ids.copy()
    segment_ids_c[40] = -1
    run_c = _prepared_run(run_c.features, segment_ids_c, valid_c)
    return {"run-a": run_a, "run-b": run_b, "run-c": run_c}


def _hand_top_split(run: PreparedRun, cfg: SweepConfig) -> SegmentSplit:
    """The run_sweep TOP split, re-derived independently of `build_pool`."""
    return split_by_segments(run.segment_ids, run.valid_mask, cfg.calibration_frac, cfg.seed)


def _hand_nested_split(run: PreparedRun, cfg: SweepConfig) -> SegmentSplit:
    """The run_sweep NESTED fit/conformal split of the calibration side (0.5,
    seed + 1), re-derived independently of `build_pool`."""
    top = _hand_top_split(run, cfg)
    calibration_mask = np.zeros(run.valid_mask.shape[0], dtype=bool)
    calibration_mask[top.calibration_windows] = True
    return split_by_segments(run.segment_ids, calibration_mask, 0.5, cfg.seed + 1)


def _member_by_run(pool: PoolResult, run_name: str):
    matches = [m for m in pool.members if m.run_name == run_name]
    assert len(matches) == 1, f"expected exactly one member for {run_name!r}"
    return matches[0]


# ---------------------------------------------------------------------------
# 1+2. Per-run split parity vs hand-run split_by_segments (all three sides)
# ---------------------------------------------------------------------------


def test_calibration_side_matches_hand_run_top_split_per_run() -> None:
    prepared = _three_run_pool()

    pool = build_pool(prepared, "calibration", _CFG)

    assert pool.side == "calibration"
    assert [m.run_name for m in pool.members] == list(prepared)
    for run_name, run in prepared.items():
        expected = _hand_top_split(run, _CFG).calibration_windows
        member = _member_by_run(pool, run_name)
        np.testing.assert_array_equal(member.windows, expected)
        assert member.n_windows == expected.shape[0]


def test_fit_and_conformal_sides_match_hand_run_nested_split_per_run() -> None:
    prepared = _three_run_pool()

    fit_pool = build_pool(prepared, "fit", _CFG)
    conformal_pool = build_pool(prepared, "conformal", _CFG)

    for run_name, run in prepared.items():
        nested = _hand_nested_split(run, _CFG)
        np.testing.assert_array_equal(
            _member_by_run(fit_pool, run_name).windows, nested.calibration_windows
        )
        np.testing.assert_array_equal(
            _member_by_run(conformal_pool, run_name).windows, nested.scoring_windows
        )


# ---------------------------------------------------------------------------
# 3. Side semantics: calibration >= fit u conformal, fit n conformal = {} per run
# ---------------------------------------------------------------------------


def test_fit_and_conformal_partition_the_calibration_side_per_run() -> None:
    prepared = _three_run_pool()

    calibration_pool = build_pool(prepared, "calibration", _CFG)
    fit_pool = build_pool(prepared, "fit", _CFG)
    conformal_pool = build_pool(prepared, "conformal", _CFG)

    for run_name in prepared:
        calibration = set(_member_by_run(calibration_pool, run_name).windows.tolist())
        fit = set(_member_by_run(fit_pool, run_name).windows.tolist())
        conformal = set(_member_by_run(conformal_pool, run_name).windows.tolist())
        assert fit, f"{run_name}: fit side must be non-empty in this fixture"
        assert conformal, f"{run_name}: conformal side must be non-empty in this fixture"
        assert fit.isdisjoint(conformal), f"{run_name}: fit/conformal overlap"
        assert (fit | conformal) <= calibration, (
            f"{run_name}: fit u conformal must stay inside the calibration side"
        )
        # The nested split partitions the calibration side exactly (every calibration
        # window has a real segment and is valid, so none can be dropped).
        assert (fit | conformal) == calibration


# ---------------------------------------------------------------------------
# 4. Stacked features/run_index/window_index bitwise alignment
# ---------------------------------------------------------------------------


def test_stacked_rows_align_bitwise_with_source_runs() -> None:
    prepared = _three_run_pool()

    pool = build_pool(prepared, "fit", _CFG)

    total = sum(m.n_windows for m in pool.members)
    assert pool.features.shape == (total, 3)
    assert pool.run_index.shape == (total,)
    assert pool.window_index.shape == (total,)

    # Row order is members order: run_index/window_index are exactly the members'
    # own windows, concatenated in members order.
    np.testing.assert_array_equal(
        pool.run_index,
        np.concatenate(
            [np.full(m.n_windows, i, dtype=np.int64) for i, m in enumerate(pool.members)]
        ),
    )
    np.testing.assert_array_equal(
        pool.window_index, np.concatenate([m.windows for m in pool.members])
    )

    # Every stacked row maps back bitwise to prepared[run].features[window].
    for i in range(total):
        member = pool.members[int(pool.run_index[i])]
        window = int(pool.window_index[i])
        source_row = prepared[member.run_name].features[window]
        assert np.array_equal(pool.features[i], source_row), (
            f"row {i} does not map back to {member.run_name!r} window {window}"
        )


# ---------------------------------------------------------------------------
# 5. Leakage probe: NO run's scoring windows in ANY side
# ---------------------------------------------------------------------------


def test_no_scoring_windows_of_any_run_in_any_side() -> None:
    prepared = _three_run_pool()

    for side in _SIDES:
        pool = build_pool(prepared, side, _CFG)
        for run_name, run in prepared.items():
            scoring = set(_hand_top_split(run, _CFG).scoring_windows.tolist())
            member_windows = set(_member_by_run(pool, run_name).windows.tolist())
            assert member_windows.isdisjoint(scoring), (
                f"{side!r} side of {run_name!r} contains scoring windows -- leakage"
            )


# ---------------------------------------------------------------------------
# 6. Provenance counts
# ---------------------------------------------------------------------------


def test_provenance_counts_match_members() -> None:
    prepared = _three_run_pool()

    pool = build_pool(prepared, "conformal", _CFG)

    assert set(pool.provenance) == set(prepared)
    for member in pool.members:
        assert pool.provenance[member.run_name] == {"n_windows": member.n_windows}
    assert sum(v["n_windows"] for v in pool.provenance.values()) == pool.features.shape[0]


# ---------------------------------------------------------------------------
# 7. Pooled features are float64 COPIES (binding: never views into the source runs)
# ---------------------------------------------------------------------------


def test_pool_features_are_float64_copies() -> None:
    prepared = _three_run_pool()
    original = prepared["run-a"].features.copy()

    pool = build_pool(prepared, "calibration", _CFG)

    assert pool.features.dtype == np.float64
    pool.features[:] = -999.0
    np.testing.assert_array_equal(prepared["run-a"].features, original)


# ---------------------------------------------------------------------------
# 8+9. Empty side for a run -> member with n_windows 0 + warning, never a crash
# ---------------------------------------------------------------------------


def test_single_segment_run_yields_empty_member_and_warning(caplog) -> None:
    prepared = {"good": _pool_run(6, 10, seed=0), "tiny": _pool_run(1, 12, seed=3)}

    with caplog.at_level(logging.WARNING):
        pool = build_pool(prepared, "calibration", _CFG)

    tiny = _member_by_run(pool, "tiny")
    assert tiny.n_windows == 0
    assert tiny.windows.shape == (0,)
    assert tiny.windows.dtype == np.int64
    assert pool.provenance["tiny"] == {"n_windows": 0}
    # Stacked arrays carry ONLY the good run's rows.
    good = _member_by_run(pool, "good")
    assert pool.features.shape[0] == good.n_windows
    assert set(pool.run_index.tolist()) == {pool.members.index(good)}

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("tiny" in w and "empty" in w for w in warnings)


def test_two_segment_run_empty_on_nested_sides_but_not_calibration(caplog) -> None:
    # 2 segments: the top split works (one segment per side), but the nested
    # fit/conformal split of the single-segment calibration side cannot -- so the
    # emptiness is SIDE-specific, not run-specific.
    prepared = {"two-seg": _pool_run(2, 10, seed=4)}

    calibration_pool = build_pool(prepared, "calibration", _CFG)
    assert _member_by_run(calibration_pool, "two-seg").n_windows > 0

    with caplog.at_level(logging.WARNING):
        fit_pool = build_pool(prepared, "fit", _CFG)

    assert _member_by_run(fit_pool, "two-seg").n_windows == 0
    assert fit_pool.features.shape[0] == 0
    assert fit_pool.features.ndim == 2
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("two-seg" in w and "empty" in w for w in warnings)


# ---------------------------------------------------------------------------
# 10+11. Input validation
# ---------------------------------------------------------------------------


def test_unknown_side_and_empty_prepared_raise() -> None:
    prepared = {"run-a": _pool_run(6, 10, seed=0)}

    with pytest.raises(ValueError, match="side"):
        build_pool(prepared, "scoring", _CFG)
    with pytest.raises(ValueError, match="at least one run"):
        build_pool({}, "calibration", _CFG)


def test_feature_dim_mismatch_raises() -> None:
    prepared = {
        "run-a": _pool_run(6, 10, seed=0, n_features=3),
        "run-b": _pool_run(6, 10, seed=1, n_features=4),
    }

    with pytest.raises(ValueError, match="feature dimensionality"):
        build_pool(prepared, "calibration", _CFG)


# ---------------------------------------------------------------------------
# 12+13. coverage_table: schema + counts, int labels and composite string labels
# ---------------------------------------------------------------------------


def test_coverage_table_schema_and_counts_int_labels() -> None:
    prepared = {"run-a": _pool_run(2, 4, seed=0), "run-b": _pool_run(2, 3, seed=1)}
    labels_a = np.array([0, 0, 1, 1, 2, 2, 2, 0], dtype=np.int64)
    labels_b = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    windows_a = np.array([0, 1, 2, 4, 5, 6], dtype=np.int64)  # labels 0,0,1,2,2,2
    windows_b = np.array([0, 3], dtype=np.int64)  # labels 1,0

    table = coverage_table(
        prepared,
        {"run-a": windows_a, "run-b": windows_b},
        {"run-a": labels_a, "run-b": labels_b},
    )

    assert list(table.columns) == ["run", "label", "n_windows"]
    rows = {(r, label): n for r, label, n in table.itertuples(index=False)}
    assert rows == {
        ("run-a", 0): 2,
        ("run-a", 1): 1,
        ("run-a", 2): 3,
        ("run-b", 0): 1,
        ("run-b", 1): 1,
    }
    assert table["n_windows"].dtype == np.int64


def test_coverage_table_composite_state_load_bin_labels() -> None:
    # Composite "state|load_bin" strings (A4.2's state x load-bin cells) are just
    # another label array -- the machinery is labels-agnostic.
    prepared = {"run-a": _pool_run(2, 3, seed=0)}
    labels = np.array(
        ["turbine|0", "turbine|2", "turbine|0", "pump|1", "standstill|-1", "pump|1"],
        dtype=object,
    )
    windows = np.array([0, 1, 2, 3, 5], dtype=np.int64)

    table = coverage_table(prepared, {"run-a": windows}, {"run-a": labels})

    rows = {(r, label): n for r, label, n in table.itertuples(index=False)}
    assert rows == {
        ("run-a", "turbine|0"): 2,
        ("run-a", "turbine|2"): 1,
        ("run-a", "pump|1"): 2,
    }


def test_coverage_table_validates_runs_and_label_alignment() -> None:
    prepared = {"run-a": _pool_run(2, 3, seed=0)}
    windows = np.array([0, 1], dtype=np.int64)
    good_labels = np.zeros(6, dtype=np.int64)

    with pytest.raises(ValueError, match="ghost"):
        coverage_table(prepared, {"ghost": windows}, {"ghost": good_labels})
    with pytest.raises(ValueError, match="labels_per_run"):
        coverage_table(prepared, {"run-a": windows}, {})
    with pytest.raises(ValueError, match="full window grid"):
        coverage_table(prepared, {"run-a": windows}, {"run-a": np.zeros(4, dtype=np.int64)})
    with pytest.raises(ValueError, match="outside"):
        coverage_table(
            prepared, {"run-a": np.array([0, 99], dtype=np.int64)}, {"run-a": good_labels}
        )


# ---------------------------------------------------------------------------
# 14. coverage_warnings fires exactly on eval-not-train cells (A4.1/A4.2)
# ---------------------------------------------------------------------------


def test_coverage_warnings_fire_exactly_on_eval_not_train_cells(caplog) -> None:
    train = pd.DataFrame(
        [
            {"run": "fit-1", "label": "turbine|0", "n_windows": 5},
            {"run": "fit-2", "label": "turbine|0", "n_windows": 2},
            {"run": "fit-1", "label": "pump|1", "n_windows": 3},
            # An explicit zero-count row is NOT coverage -- must still warn.
            {"run": "fit-1", "label": "phase-shifter|-1", "n_windows": 0},
        ],
        columns=["run", "label", "n_windows"],
    )
    eval_ = pd.DataFrame(
        [
            {"run": "test-1", "label": "turbine|0", "n_windows": 4},  # covered: silent
            {"run": "test-1", "label": "pump|2", "n_windows": 7},  # eval-only: warns
            {"run": "test-1", "label": "standstill|-1", "n_windows": 2},  # eval-only: warns
            {"run": "test-1", "label": "phase-shifter|-1", "n_windows": 3},  # zero train: warns
        ],
        columns=["run", "label", "n_windows"],
    )

    with caplog.at_level(logging.WARNING):
        warnings = coverage_warnings(train, eval_)

    assert len(warnings) == 3
    joined = "\n".join(warnings)
    for expected_label in ("pump|2", "standstill|-1", "phase-shifter|-1"):
        assert expected_label in joined
    assert "turbine|0" not in joined
    # `pump|1` is train-only (never evaluated) -- not an eval-not-train cell.
    assert "pump|1" not in joined
    assert all("zero training coverage" in w for w in warnings)
    # The warnings are also LOGGED (spec A4.2: "the pool builder logs a warning").
    logged = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(logged) == 3


def test_coverage_warnings_silent_when_everything_covered() -> None:
    train = pd.DataFrame(
        [{"run": "fit-1", "label": 0, "n_windows": 10}], columns=["run", "label", "n_windows"]
    )
    eval_ = pd.DataFrame(
        [{"run": "test-1", "label": 0, "n_windows": 6}], columns=["run", "label", "n_windows"]
    )

    assert coverage_warnings(train, eval_) == []


# ---------------------------------------------------------------------------
# 15. A3.7 WARNING is pinned in the module docstring (cheap doc pin, same pattern
#     as the plan's Task-4 docstring pin)
# ---------------------------------------------------------------------------


def test_module_docstring_pins_a3_7_warning() -> None:
    doc = pools_module.__doc__
    assert doc is not None
    assert "_cross_day_per_state_sweep" in doc
    assert "TOP split" in doc
    assert "run_sweep" in doc


def test_coverage_table_empty_selection_keeps_int64_dtype() -> None:
    """T1-review MEDIUM: an EMPTY selection is a mainline A4.1 output; pandas
    would infer `object` for n_windows from an empty row list, and pd.concat
    would then degrade every aggregated coverage table downstream."""
    empty = coverage_table({}, {}, {})
    assert list(empty.columns) == ["run", "label", "n_windows"]
    assert empty["n_windows"].dtype == np.int64

    prepared = {
        "run-a": _prepared_run(
            features=np.arange(12, dtype=np.float64).reshape(6, 2),
            segment_ids=np.array([0, 0, 0, 1, 1, 1]),
        )
    }
    real = coverage_table(
        prepared,
        {"run-a": np.array([0, 1, 2])},
        {"run-a": np.array([0, 0, 1, 1, 1, 0])},
    )
    combined = pd.concat([empty, real], ignore_index=True)
    assert combined["n_windows"].dtype == np.int64
