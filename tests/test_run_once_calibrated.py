"""Tests for scripts/run_once_calibrated.py: pure regime-
selection + trigger-log + FAR-reading helpers on synthetic parquet/verdict inputs,
and the 270626 sentinel-only path -- monitor/eval_events subprocesses are behind
monkeypatched seams, so no real monitor run happens in a unit test.

RED tests written directly against the design's own acceptance criteria:
`test_read_realized_far_over_scored_only`, `test_far_on_common_window_set`,
`test_regime_far_and_trigger_verdict`, `test_sentinel_only_day_has_no_far_row`.
This file extends them with: `_scoring_windows` (declared in the plan's own
Interfaces text but not spelled out as a RED test there), `_pool_block_ids` (the
cross-run block-id uniqueness fix this driver needs -- no existing caller in the
repo pools multiple runs' `segment_ids` into one block statistic,
`scripts/analyze_days.py::_block_bootstrap_ci` is single-run by construction),
and an end-to-end `main()` CLI run over the FULL pinned `_REPLAY`/`_B1_FIT_RUNS`/
pillar-3 run set on hand-built `PreparedRun`s (mirrors `tests/
test_run_modebank.py`'s own `_prepared`/`_install` seam shape -- the real
`build_pool` + `ModeBank.fit` + `rowii.anomaly.sentinels` commissioning path runs
for real on synthetic blobs; `_run_monitor`/`_run_eval_events` are monkeypatched
away, so no real subprocess and no real data anywhere in this file).

Hardening-pass additions (fixes filed and closed as a batch):
`test_read_realized_window_far_reads_summary_row` (the event-free FAR
reader) and `test_run_monitor_builds_expected_argv_frozen_recalibrate_and_
event_free` (patches `subprocess.run` itself -- never the `_run_monitor`
seam) are new. `test_run_once_calibrated_replay`'s `_fake_run_monitor` now
marks a STRICT SUBSET of recalibrate windows scored (exercises the
common-window subsetting for real) and its 080726-pu_strikes assertions +
the sidecar's `s1` block assertions are extended (the event-free `far_basis`
headline + still-present-and-distinct raw secondary; the echoed
`pct`/`n_boot`/`seed`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from rowii.anomaly.pools import PoolMember, PoolResult  # noqa: E402
from rowii.config import Config  # noqa: E402
from rowii.io.dataset import RecordingIndex, Run  # noqa: E402
from rowii.pipeline import PreparedRun  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402

# ---------------------------------------------------------------------------
# Plan's own RED tests (verbatim)
# ---------------------------------------------------------------------------


def _alarms(tmp_path: Path, window, alarm, role) -> Path:
    p = tmp_path / "alarms.parquet"
    pd.DataFrame({"window": window, "alarm": alarm, "role": role}).to_parquet(p, index=False)
    return p


def test_read_realized_far_over_scored_only(tmp_path: Path) -> None:
    import run_once_calibrated as roc
    path = _alarms(tmp_path, [0, 1, 2, 3], [True, False, True, True],
                   ["scored", "scored", "consumed_for_calibration", "scored"])
    # scored windows: 0(True),1(False),3(True) -> 2/3.
    assert roc._read_realized_far(path) == pytest.approx(2 / 3)


def test_far_on_common_window_set(tmp_path: Path) -> None:
    import run_once_calibrated as roc
    path = _alarms(tmp_path, [0, 1, 2, 3], [True, True, False, True],
                   ["scored", "scored", "scored", "scored"])
    # Subset the frozen arm onto the recalibrate scoring split {1, 3}.
    assert roc._far_on_windows(path, np.array([1, 3])) == pytest.approx(1.0)


def test_regime_far_and_trigger_verdict() -> None:
    import run_once_calibrated as roc
    assert roc._trigger_verdict(s1_fired=False, s2_fired=False) is False
    assert roc._trigger_verdict(s1_fired=True, s2_fired=False) is True
    assert roc._regime_far(0.9, 0.05, triggered=True) == 0.05   # fired -> recalibrate
    assert roc._regime_far(0.9, 0.05, triggered=False) == 0.9   # quiet -> frozen (NOT sticky)


def test_sentinel_only_day_has_no_far_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import run_once_calibrated as roc
    row = roc._trigger_log_row(
        day="270626", era="A", tags=("sentinel-only",),
        s1_rate=0.4, s1_threshold=0.1, low_confidence_modes=(),
        s2_mic=-30.0, s2_vib=-50.0, anchor=-40.0, mad=0.5,
        attribution="instrumentation", decision="recalibrate",
    )
    assert row["day"] == "270626" and "sentinel-only" in row["tags"]
    assert "far" not in row  # sentinel-only: no FAR/GT


# ---------------------------------------------------------------------------
# `_scoring_windows` -- declared in the plan's own Interfaces text ("Pure
# helpers (unit-tested -- the load-bearing logic)") but not spelled out as its
# own RED test there; the recalibrate arm's own scoring-split population, the
# common population `_far_on_windows` subsets every OTHER arm onto.
# ---------------------------------------------------------------------------


def test_scoring_windows_selects_scored_role_only(tmp_path: Path) -> None:
    import run_once_calibrated as roc
    path = _alarms(tmp_path, [10, 11, 12, 13], [True, False, True, False],
                   ["scored", "consumed_for_calibration", "scored", "unknown_state"])
    np.testing.assert_array_equal(roc._scoring_windows(path), np.array([10, 12]))


# ---------------------------------------------------------------------------
# `_pool_block_ids`: the cross-run block-id uniqueness fix (module docstring) --
# hand-built PoolResult/PreparedRun, no real data.
# ---------------------------------------------------------------------------


def test_pool_block_ids_unique_per_run_and_local_segment() -> None:
    """Two runs, each with its OWN `segment_ids` restarting at 0: run "b"'s
    local segment 0 must NOT be merged into the SAME block as run "a"'s own
    local segment 0 -- the exact collision `_pool_block_ids` exists to avoid."""
    import run_once_calibrated as roc

    prep_a = PreparedRun(
        features=np.zeros((4, 1)), grid=WindowGrid(0, 1_000_000_000, 4),
        valid_mask=np.ones(4, dtype=bool), feature_names=["f0"],
        segment_ids=np.array([0, 0, 1, 1], dtype=np.int64),
    )
    prep_b = PreparedRun(
        features=np.zeros((4, 1)), grid=WindowGrid(0, 1_000_000_000, 4),
        valid_mask=np.ones(4, dtype=bool), feature_names=["f0"],
        segment_ids=np.array([0, 0, 1, 1], dtype=np.int64),
    )
    pool = PoolResult(
        side="conformal",
        members=[
            PoolMember(run_name="a", windows=np.array([0, 1, 2, 3]), n_windows=4),
            PoolMember(run_name="b", windows=np.array([0, 1, 2, 3]), n_windows=4),
        ],
        features=np.zeros((8, 1)),
        run_index=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
        window_index=np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64),
        provenance={},
    )
    block_ids = roc._pool_block_ids(pool, {"a": prep_a, "b": prep_b})
    assert block_ids.dtype == np.int64
    # 4 distinct (run, local-segment) pairs -> exactly 4 distinct block ids.
    assert len(set(block_ids.tolist())) == 4
    # Rows sharing the SAME (run, local-segment) pair get the SAME block id...
    assert block_ids[0] == block_ids[1]
    assert block_ids[4] == block_ids[5]
    # ...but run a's local segment 0 and run b's local segment 0 do NOT collide.
    assert block_ids[0] != block_ids[4]
    assert block_ids[2] != block_ids[6]


# ---------------------------------------------------------------------------
# `_read_realized_window_far` -- the event-free FAR reading a
# regimes row for an EVENT-BEARING `_REPLAY` entry must use (never the raw
# `alarms.parquet` scored-window mean, which silently counts the
# controlled-event windows too -- they correctly alarm, inflating the raw
# reading ~2.5-3x for 080726).
# ---------------------------------------------------------------------------


def _event_eval_csv(tmp_path: Path, *, realized_window_far: float) -> Path:
    p = tmp_path / "event_eval.csv"
    pd.DataFrame([{
        "row_type": "summary", "n_events": 2, "n_detected": 2, "event_tpr": 1.0,
        "false_alarm_windows": 3, "false_alarm_rate_per_hour": 1.5,
        "realized_window_far": realized_window_far, "tolerance_s": 5.0,
    }]).to_csv(p, index=False)
    return p


def test_read_realized_window_far_reads_summary_row(tmp_path: Path) -> None:
    import run_once_calibrated as roc
    path = _event_eval_csv(tmp_path, realized_window_far=0.1234)
    assert roc._read_realized_window_far(path) == pytest.approx(0.1234)


# ---------------------------------------------------------------------------
# `_run_monitor`'s constructed subprocess.run argv -- patches
# `subprocess.run` itself (NOT the `_run_monitor` seam used everywhere else in
# this file), so the real argv-building logic inside `_run_monitor` is
# exercised. Asserts full LIST equality (never a joined string), covering a
# frozen call (no --alpha), a plain recalibrate call (--alpha present, no
# exclusion), and an 080726-style call (--alpha AND --exclude-calibration-
# events + the events CSV path together, matching the module's own _REPLAY
# entry for 080726-pu_strikes).
# ---------------------------------------------------------------------------


def test_run_monitor_builds_expected_argv_frozen_recalibrate_and_event_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import run_once_calibrated as roc

    captured: list[list[str]] = []

    def _fake_subprocess_run(cmd: list[str], *, check: bool, cwd: Path) -> None:
        captured.append(cmd)
        assert check is True
        assert cwd == roc._REPO_ROOT

    monkeypatch.setattr(roc.subprocess, "run", _fake_subprocess_run)

    snapshot_path = tmp_path / "snap.npz"
    monitor_py = str(roc._SCRIPTS_DIR / "monitor.py")

    frozen_out = tmp_path / "frozen"
    roc._run_monitor(snapshot_path, "290626-tu", "frozen", frozen_out)
    assert captured[-1] == [
        sys.executable, monitor_py,
        "--snapshot", str(snapshot_path),
        "--run", "290626-tu",
        "--thresholds", "frozen",
        "--out", str(frozen_out),
    ]

    recal_out = tmp_path / "recal"
    roc._run_monitor(snapshot_path, "290626-tu", "recalibrate", recal_out, alpha=0.01)
    assert captured[-1] == [
        sys.executable, monitor_py,
        "--snapshot", str(snapshot_path),
        "--run", "290626-tu",
        "--thresholds", "recalibrate",
        "--out", str(recal_out),
        "--alpha", "0.01",
    ]

    events_csv = tmp_path / "080726_events_pu.csv"
    event_out = tmp_path / "event_free"
    roc._run_monitor(
        snapshot_path, "080726-pu_strikes", "recalibrate", event_out,
        alpha=0.01, event_free=events_csv,
    )
    assert captured[-1] == [
        sys.executable, monitor_py,
        "--snapshot", str(snapshot_path),
        "--run", "080726-pu_strikes",
        "--thresholds", "recalibrate",
        "--out", str(event_out),
        "--alpha", "0.01",
        "--exclude-calibration-events", str(events_csv),
    ]


# ---------------------------------------------------------------------------
# End-to-end `main()`: monkeypatched discover/load_config/prepare_run/
# _run_gt_states/_run_monitor/_run_eval_events. REAL build_pool + ModeBank.fit
# + rowii.anomaly.sentinels run on hand-built PreparedRuns covering every
# pinned run name (mirrors tests/test_run_modebank.py's _prepared/_install
# seam shape). No real data, no real subprocess.
# ---------------------------------------------------------------------------

_W = 1_000_000_000
_BEATS_NAMES = [f"beats_{i}" for i in range(4)]
_AUDIO_NAMES = [
    "RAWGeneratorMic__0::ch0_log_rms", "RAWGeneratorMic__0::ch0_spectral_centroid",
    "RAWTurbineMic__1::ch0_octave_125", "RAWTurbineMic__1::ch0_rolloff95",
]
_VIB_NAMES = ["RAWGeneratorVib__2::ch0_log_rms", "RAWGeneratorVib__2::ch0_spectral_centroid"]

_B1_RUNS = ("010726-pu", "010726-tu1-morning", "010726-tu2", "010726-tu_ph_tu")
_NORMAL_RUNS = (
    "250526-tu", "250526-pu-morning", "290626-tu", "290626-pu",
    # 300626 (era B, held-out; _REPLAY addition 2026-08-18): same fixture profile
    # as the other in-era normal days -- bank recognizes every window, sentinels
    # stay quiet.
    "300626-tu", "300626-pu",
)
_DRIFTED_RUNS = ("270626-pu_ph_pu_ph_pu_ph-1", "080726-pu_strikes")
_OTHER_RUNS = ("080726-st_strikes",)  # pillar-3 only, not a _REPLAY entry


def _blob_prepared(
    t0: int, seed: int, names: list[str], locs: list[float], seg_len: int = 10
) -> PreparedRun:
    """`len(locs)` contiguous *seg_len*-window segments, each a tight blob at
    its OWN `locs[i]` value across every column (uniform columns -- sufficient
    for `level_series`, which only needs the LEVEL-column subset's plain MEAN,
    identical to every other column here by construction). NOT used for the
    audio-beats variant (see `_beats_blob` below): `ModeBank`'s `knn` family
    scores COSINE distance (`ModeBank.fit`'s own module docstring), under which
    a uniform-column vector's DIRECTION is degenerate near the origin -- see
    `_beats_blob`'s docstring."""
    rng = np.random.default_rng(seed)
    blocks = [
        np.full((seg_len, len(names)), loc) + rng.normal(0.0, 0.05, (seg_len, len(names)))
        for loc in locs
    ]
    feats = np.vstack(blocks)
    n = feats.shape[0]
    seg_ids = np.concatenate(
        [np.full(seg_len, i, dtype=np.int64) for i in range(len(locs))]
    )
    return PreparedRun(
        features=feats, grid=WindowGrid(t0, _W, n), valid_mask=np.ones(n, dtype=bool),
        feature_names=list(names), segment_ids=seg_ids,
    )


_TURBINE_DIR = np.array([5.0, 0.0, 0.0, 0.0])
_PUMP_DIR = np.array([0.0, 5.0, 0.0, 0.0])
_DRIFTED_DIR = np.array([0.0, 0.0, 0.0, -5.0])
"""Audio-beats blob directions, ORTHOGONAL to each other by construction (dot
product 0): `ModeBank`'s `knn` family scores COSINE distance, so genuine
separation needs DIFFERENT DIRECTIONS, not merely different magnitudes along a
shared direction (a uniform-column vector's direction is ill-defined near the
origin, which is exactly what made an earlier draft of this fixture
accidentally drop the 'pump' mode from the bank -- see `_bank_fit_prepared`'s
docstring)."""


def _beats_blob(t0: int, seed: int, directions: list[np.ndarray], seg_len: int) -> PreparedRun:
    """`len(directions)` contiguous *seg_len*-window segments, each a tight
    blob around its OWN direction vector (tiled + small noise) -- the
    audio-beats-variant counterpart of `_blob_prepared`, using DIRECTIONAL
    vectors instead of a uniform scalar broadcast across every column, so
    `ModeBank`'s cosine-distance `knn` family sees genuinely separated blobs."""
    rng = np.random.default_rng(seed)
    blocks = [
        np.tile(direction, (seg_len, 1)) + rng.normal(0.0, 0.05, (seg_len, len(direction)))
        for direction in directions
    ]
    feats = np.vstack(blocks)
    n = feats.shape[0]
    seg_ids = np.concatenate(
        [np.full(seg_len, i, dtype=np.int64) for i in range(len(directions))]
    )
    return PreparedRun(
        features=feats, grid=WindowGrid(t0, _W, n), valid_mask=np.ones(n, dtype=bool),
        feature_names=list(_BEATS_NAMES), segment_ids=seg_ids,
    )


_BANK_FIT_MODE_PATTERN = (
    "turbine", "turbine", "turbine", "pump", "turbine", "pump", "pump", "pump",
)
"""8-segment mode pattern for `_bank_fit_prepared`, deliberately NOT a simple
alternation. `rowii.anomaly.references.split_by_segments` shuffles segment IDS
with a FIXED seed (`SweepConfig`'s own `seed`/`seed+1` defaults), and every B1
run here shares the IDENTICAL 8-segment layout -- so `build_pool`'s top and
nested splits land on the EXACT SAME segment INDICES for every run (verified:
fit segments = {0, 7}, conformal segments = {2, 6} for this exact 8x30-window
layout under `SweepConfig(alpha=0.01)`'s defaults). A plain alternating
pattern puts segments 2 AND 6 in the SAME mode (both even indices), so every
run's pooled CONFORMAL side would contain ONLY that one mode and the other
mode would be dropped from the bank entirely ("ZERO calibration window(s)",
`ModeBank.fit`'s own warning) -- this pattern instead assigns index 0 != index
7 and index 2 != index 6, so both modes clear `_BANK_MIN_REF` on BOTH the
pooled fit AND conformal sides. Still balanced (4 turbine / 4 pump segments)."""


def _bank_fit_prepared(t0: int, seed: int) -> tuple[PreparedRun, np.ndarray]:
    """A B1 commissioning run's audio-beats side: 8 contiguous 30-window
    segments in `_BANK_FIT_MODE_PATTERN` order -- ample rows per mode (120)
    for `_BANK_MIN_REF=20` pooled across the four B1 runs, mirroring `tests/
    test_run_modebank.py`'s own `_prepared` shape (same segment/window
    counts), but with `_BANK_FIT_MODE_PATTERN`'s non-alternating mode order
    (see that constant's own docstring) and DIRECTIONAL (not scalar) blobs
    (`_beats_blob`)."""
    directions = [
        _TURBINE_DIR if mode == "turbine" else _PUMP_DIR for mode in _BANK_FIT_MODE_PATTERN
    ]
    prepared = _beats_blob(t0, seed, directions, seg_len=30)
    gt = np.array(
        [mode for mode in _BANK_FIT_MODE_PATTERN for _ in range(30)], dtype=object,
    )
    return prepared, gt


def _install(
    monkeypatch: pytest.MonkeyPatch,
    mod: object,
    results_root: Path,
    prepared: dict[tuple[str, str], PreparedRun],
    gt_by_run: dict[str, np.ndarray],
) -> None:
    """*prepared* is keyed by `(run_name, variant)`; *gt_by_run* by run_name
    (B1 fit runs only -- `_run_gt_states` is never called for any `_REPLAY`-only
    run, module docstring)."""
    run_names = sorted({name for name, _variant in prepared})
    runs = [Run(name=n, files={}, day_root=Path(f"/d/{n}")) for n in run_names]
    monkeypatch.setattr(
        mod, "discover",
        lambda dr: RecordingIndex(runs=runs, betriebsdaten=[], betriebsdaten_by_day={}),
    )
    monkeypatch.setattr(
        mod, "load_config",
        lambda: Config(data_root=Path("/d"), results_root=results_root),
    )
    monkeypatch.setattr(
        mod, "prepare_run",
        lambda run, variant, cfg, *, use_cache: prepared[(run.name, variant)],
    )
    monkeypatch.setattr(
        mod, "_run_gt_states",
        lambda prepared_run, run, index, cfg: gt_by_run[run.name],
    )


def _fake_run_monitor(
    snapshot_path: Path, run: str, mode: str, out_dir: Path, *, alpha=None, event_free=None
) -> Path:
    """Deterministic fake alarms.parquet: FROZEN alarms at a HIGH rate over its
    FULL 20-window population (12/20 = 0.6), RECALIBRATE at a LOW rate over a
    STRICT SUBSET of scored windows -- windows 0-3 are `consumed_for_
    calibration` (NOT scored), windows 4-19 are `scored` with exactly 1 alarm
    (1/16 = 0.0625) -- mirrors the central finding (frozen cross-day FAR
    does not hold) so the 3-regime arithmetic is meaningfully checkable, not
    merely plumbing, AND exercises the common-window subsetting for real
    (regression coverage): the OLD fixture marked every window `role="scored"` on
    BOTH arms, making `_far_on_windows` a no-op end-to-end (window sets
    always matched) -- a real subsetting bug would have gone uncaught by this
    test. Applies uniformly to every run name (including `080726-pu_strikes`,
    an EVENT-BEARING entry) since this fixture never varies by run; the
    event-bearing regime FAR no longer reads this parquet directly
    (`_fake_run_eval_events` below), so the subset change is invisible to
    it except via the UNCHANGED frozen-arm secondary reading."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 20
    if mode == "frozen":
        role = np.array(["scored"] * n, dtype=object)
        alarm = np.array([True] * 12 + [False] * (n - 12), dtype=bool)
    else:
        role = np.array(
            ["consumed_for_calibration"] * 4 + ["scored"] * (n - 4), dtype=object
        )
        alarm = np.array([False] * 4 + [True] + [False] * (n - 5), dtype=bool)
    df = pd.DataFrame({
        "window": np.arange(n, dtype=np.int64),
        "t_utc_ns": np.arange(n, dtype=np.int64) * _W,
        "alarm": alarm,
        "role": role,
    })
    path = out_dir / "alarms.parquet"
    df.to_parquet(path, index=False)
    return path


def _fake_run_eval_events(
    alarms_path: Path, events_path: Path, out_dir: Path, *, tolerance_s: float
) -> Path:
    """Deterministic fake event_eval.csv: TPR 1.0 + realized_window_far 0.2
    when fed the RECALIBRATE alarms, TPR 0.5 + realized_window_far 0.9 when
    fed the FROZEN alarms (inferred from `_fake_run_monitor`'s own out_dir
    convention, both ends controlled by this same test file) -- the
    once+triggered/frozen TPR contrast this module wants illustrated, AND
    `realized_window_far` values DELIBERATELY distinct from
    `_fake_run_monitor`'s own raw parquet-mean FARs (0.6 frozen / 0.0625
    recalibrate) so a regimes-row assertion that accidentally reads the raw
    parquet mean instead of this eval_events summary is caught rather than
    passing by coincidence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    is_recalibrate = "recalibrate" in alarms_path.parts
    tpr = 1.0 if is_recalibrate else 0.5
    far = 0.2 if is_recalibrate else 0.9
    df = pd.DataFrame([{
        "row_type": "summary", "n_events": 13, "n_detected": round(tpr * 13),
        "event_tpr": tpr, "false_alarm_windows": 0, "false_alarm_rate_per_hour": 0.0,
        "realized_window_far": far, "tolerance_s": tolerance_s,
    }])
    path = out_dir / "event_eval.csv"
    df.to_csv(path, index=False)
    return path


def _build_prepared_and_gt() -> tuple[dict[tuple[str, str], PreparedRun], dict[str, np.ndarray]]:
    prepared: dict[tuple[str, str], PreparedRun] = {}
    gt_by_run: dict[str, np.ndarray] = {}

    for i, name in enumerate(_B1_RUNS):
        beats, gt = _bank_fit_prepared(i * 100 * _W, seed=i)
        prepared[(name, "audio-beats")] = beats
        gt_by_run[name] = gt
        prepared[(name, "audio")] = _blob_prepared(
            i * 100 * _W, seed=100 + i, names=_AUDIO_NAMES, locs=[-40.0] * 6
        )
        prepared[(name, "vibration")] = _blob_prepared(
            i * 100 * _W, seed=200 + i, names=_VIB_NAMES, locs=[-40.0] * 6
        )

    for i, name in enumerate(_NORMAL_RUNS):
        t0 = (1000 + i * 10) * _W
        # A mix of BOTH commissioning directions -> the bank recognizes every
        # window comfortably (no_mode_fits stays low).
        prepared[(name, "audio-beats")] = _beats_blob(
            t0, seed=300 + i, directions=[_TURBINE_DIR, _PUMP_DIR], seg_len=10
        )
        prepared[(name, "audio")] = _blob_prepared(
            t0, seed=400 + i, names=_AUDIO_NAMES, locs=[-40.0] * 3
        )
        prepared[(name, "vibration")] = _blob_prepared(
            t0, seed=500 + i, names=_VIB_NAMES, locs=[-40.0] * 3
        )

    for i, name in enumerate(_DRIFTED_RUNS):
        t0 = (2000 + i * 10) * _W
        # ORTHOGONAL to BOTH commissioning directions -> the whole bank
        # rejects every window (no_mode_fits ~ 1.0), firing s1 robustly.
        prepared[(name, "audio-beats")] = _beats_blob(
            t0, seed=600 + i, directions=[_DRIFTED_DIR], seg_len=30
        )
        # Mic level 80 units off the -40 B1 anchor -> fires s2 (mic).
        prepared[(name, "audio")] = _blob_prepared(
            t0, seed=700 + i, names=_AUDIO_NAMES, locs=[40.0] * 3
        )
        # Vibration STAYS at the B1 anchor -> the vib cross-check does NOT
        # fire -> s2_attribution == "instrumentation" (the real mic-steps-
        # vib-flat signature, per the README).
        prepared[(name, "vibration")] = _blob_prepared(
            t0, seed=800 + i, names=_VIB_NAMES, locs=[-40.0] * 3
        )

    for i, name in enumerate(_OTHER_RUNS):
        t0 = (3000 + i * 10) * _W
        prepared[(name, "audio-beats")] = _beats_blob(
            t0, seed=900 + i, directions=[_TURBINE_DIR, _PUMP_DIR], seg_len=10
        )
        prepared[(name, "audio")] = _blob_prepared(
            t0, seed=910 + i, names=_AUDIO_NAMES, locs=[-40.0] * 3
        )
        prepared[(name, "vibration")] = _blob_prepared(
            t0, seed=920 + i, names=_VIB_NAMES, locs=[-40.0] * 3
        )

    return prepared, gt_by_run


def test_run_once_calibrated_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import run_once_calibrated as roc

    prepared, gt_by_run = _build_prepared_and_gt()
    _install(monkeypatch, roc, tmp_path / "results", prepared, gt_by_run)
    monkeypatch.setattr(roc, "_import_beats_or_exit", lambda: None)
    monkeypatch.setattr(roc, "_run_monitor", _fake_run_monitor)
    monkeypatch.setattr(roc, "_run_eval_events", _fake_run_eval_events)

    snapshot_path = tmp_path / "b1-fusion.npz"
    snapshot_path.write_bytes(b"not a real snapshot -- load_snapshot is monkeypatched below")

    class _FakeSnapshot:
        variant = "fusion"

    monkeypatch.setattr(roc, "load_snapshot", lambda path: _FakeSnapshot())

    code = roc.main([
        "--representation", "fusion",
        "--snapshot", str(snapshot_path),
        "--bank-fit-runs", ",".join(_B1_RUNS),
        "--alpha", "0.01",
    ])
    assert code == 0

    out_dir = tmp_path / "results" / "step2" / "once-calibrated" / "fusion"
    sidecar = json.loads((out_dir / "fusion.json").read_text())

    trigger_log = sidecar["trigger_log"]
    assert len(trigger_log) == 10  # every _REPLAY entry, sentinel-only included
    by_run = {row["run"]: row for row in trigger_log}

    # Normal-blob days never fire either sentinel -> frozen.
    assert by_run["290626-tu"]["decision"] == "frozen"
    assert by_run["290626-tu"]["s1_fired"] is False
    assert by_run["290626-tu"]["s2_fired"] is False

    # 270626 (sentinel-only, drifted) fires s1 (drifted audio-beats blob) AND
    # s2 (drifted mic) -> recalibrate decision, "instrumentation" attribution
    # (vib stayed flat) -- but NO far/GT row.
    sentinel_row = by_run["270626-pu_ph_pu_ph_pu_ph-1"]
    assert sentinel_row["decision"] == "recalibrate"
    assert sentinel_row["s1_fired"] is True
    assert sentinel_row["s2_fired"] is True
    assert sentinel_row["s2_attribution"] == "instrumentation"
    assert "far" not in sentinel_row

    # era-A boundary caught: at least one era-A row triggered.
    assert sidecar["boundary_caught_era_a"] is True

    # era C (080726-pu_strikes, drifted) triggers too.
    assert sidecar["era_c_triggered"] is True
    assert by_run["080726-pu_strikes"]["decision"] == "recalibrate"

    # 010726 rows are tagged in-sample and STILL get a full FAR row (normal
    # blob, same distribution as the B1 commissioning pool itself).
    assert list(by_run["010726-pu"]["tags"]) == ["in-sample"]
    assert list(by_run["010726-tu_ph_tu"]["tags"]) == ["in-sample"]

    regimes = sidecar["regimes"]
    assert len(regimes) == 9  # every _REPLAY entry EXCEPT the sentinel-only one
    regimes_by_run = {row["run"]: row for row in regimes}
    normal = regimes_by_run["290626-tu"]
    # Fixed fake alarm rates (module-level _fake_run_monitor): frozen 12/20 =
    # 0.6 over its full population; recalibrate marks windows 0-3 consumed
    # (NOT scored) and 1/16 scored windows alarming = 0.0625 -- a STRICT
    # SUBSET, so the common-window subsetting is exercised for real
    # (regression coverage): subsetting the frozen arm onto the recalibrate arm's own
    # scoring split {4..19} keeps only frozen's windows 4-11 alarming (8/16 =
    # 0.5), genuinely DIFFERENT from the frozen arm's own full-population
    # reading (0.6) -- the OLD all-scored fixture made these equal by
    # construction and could not have caught a subsetting bug.
    assert normal["far_basis"] == roc._FAR_BASIS_COMMON_WINDOW
    assert normal["always_frozen_far"] == pytest.approx(0.5)
    assert normal["always_recalibrate_far"] == pytest.approx(0.0625)
    assert normal["frozen_far_full_population"] == pytest.approx(0.6)
    assert normal["always_frozen_far"] != normal["frozen_far_full_population"]
    # Untriggered -> once+triggered reads the FROZEN (common-window) arm (NOT
    # sticky recalibrate).
    assert normal["once_triggered_far"] == pytest.approx(0.5)

    strikes = regimes_by_run["080726-pu_strikes"]
    # 080726 is EVENT-BEARING -- its regime FAR must be sourced
    # from eval_events' own event-free `realized_window_far` (the fake
    # _fake_run_eval_events summary: 0.9 frozen / 0.2 recalibrate), NEVER the
    # raw alarms.parquet scored-window mean (which would read 0.6/0.0625 here,
    # `_fake_run_monitor` -- the exact values `normal` reads above, proving
    # these two rows are NOT accidentally sharing one computation path).
    assert strikes["far_basis"] == roc._FAR_BASIS_EVENT_FREE
    assert strikes["always_frozen_far"] == pytest.approx(0.9)
    assert strikes["always_recalibrate_far"] == pytest.approx(0.2)
    # Triggered -> once+triggered reads the RECALIBRATE (event-free) arm.
    assert strikes["once_triggered_far"] == pytest.approx(0.2)
    # (b) The secondary raw (event-INCLUSIVE) scored-window FAR of the frozen
    # arm stays visible and numerically DISTINCT from the event-free headline
    # -- the two readings are never conflated.
    assert strikes["frozen_far_full_population"] == pytest.approx(0.6)
    assert strikes["frozen_far_full_population"] != strikes["always_frozen_far"]

    # Pillar-3: once+triggered (recalibrate, since era C triggers) reads TPR
    # 1.0 for BOTH sessions; the frozen-arm contrast reads 0.5 (module
    # docstring: both numbers reported, never only the favorable one).
    pillar3 = sidecar["pillar3"]
    assert pillar3["pu"]["once_triggered_event_tpr"] == pytest.approx(1.0)
    assert pillar3["pu"]["frozen_event_tpr"] == pytest.approx(0.5)
    assert pillar3["st"]["once_triggered_event_tpr"] == pytest.approx(1.0)
    assert pillar3["st"]["frozen_event_tpr"] == pytest.approx(0.5)

    # s1/s2 threshold derivations are persisted (sidecar JSON requirement).
    assert sidecar["s1"]["family"] == "knn"
    assert 0.0 <= sidecar["s1"]["threshold"] <= 1.0
    # pct/n_boot/seed are echoed EXPLICITLY (never left implicit
    # in s1_threshold's own defaults) so the sidecar fully documents the
    # bootstrap derivation without a reader having to open sentinels.py.
    assert sidecar["s1"]["pct"] == 97.5
    assert sidecar["s1"]["n_boot"] == 1000
    assert sidecar["s1"]["seed"] == 7
    assert sidecar["s2"]["mic_streams"] == list(roc._MIC_STREAMS)

    # CSV artifacts exist alongside the JSON sidecar.
    trigger_csv = pd.read_csv(out_dir / "fusion_trigger_log.csv")
    assert len(trigger_csv) == 10
    regimes_csv = pd.read_csv(out_dir / "fusion_regimes.csv")
    assert len(regimes_csv) == 9


def test_representation_mismatch_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import run_once_calibrated as roc

    snapshot_path = tmp_path / "snap.npz"
    snapshot_path.write_bytes(b"stub")

    class _FakeSnapshot:
        variant = "vibration"  # deliberately does NOT match --representation

    monkeypatch.setattr(roc, "load_snapshot", lambda path: _FakeSnapshot())

    code = roc.main([
        "--representation", "fusion",
        "--snapshot", str(snapshot_path),
        "--bank-fit-runs", ",".join(_B1_RUNS),
    ])
    assert code == 2


def test_duplicate_bank_fit_run_names_exits_2() -> None:
    import run_once_calibrated as roc

    with pytest.raises(SystemExit) as exc_info:
        roc.main([
            "--representation", "fusion",
            "--snapshot", "/does/not/matter.npz",
            "--bank-fit-runs", "a,a",
        ])
    assert exc_info.value.code == 2
