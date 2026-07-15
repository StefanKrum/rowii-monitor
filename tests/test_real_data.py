"""Real-data smoke guards against the full Illwerke campaign delivery.

Every test here is `@pytest.mark.data`: skipped unless `ROWII_DATA_ROOT` (via
`rowii.config.load_config`, the same `.env` + process-env resolution the rest
of the pipeline uses) points at a directory that actually exists (a stale
value or a missing `.env` must skip, not fail with a FileNotFoundError from
deep inside `discover`).

These are integration guards, not unit tests of new logic -- their purpose is
to catch the moment reality (channel names, sample rates, channel counts, day
tree layout) drifts from what the rest of the pipeline hard-codes or assumes.
Task 13's no-legacy-assumptions constraint: every number asserted below is a
property of THIS data, not carried over from an earlier delivery or
exploratory deck.

`ROWII_DATA_ROOT` now points at the PARENT root (`data/`, containing every
`illwerke-<dayid>` day tree -- spec: docs/superpowers/specs/2026-07-07-step1-
multiday-phase-shifter-addendum.md §2), so every run name discovered here
carries its day-id prefix (e.g. `"250526-tu"`, `"010726-tu_ph_tu"`) -- this
file's own single-tree/legacy naming used unprefixed names before the
addendum; that backward-compat behaviour is covered separately by
`tests/test_dataset.py`'s synthetic fixtures, not against real data here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rowii.config import load_config
from rowii.io.dataset import discover
from rowii.io.gantner import read_header
from rowii.pipeline import prepare_run
from rowii.scada.labels import GT_CHANNELS
from rowii.signals.windows import WindowGrid
from rowii.state.detect import FittedDetector

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

_DATA_ROOT = load_config().data_root
_HAS_DATA_ROOT = _DATA_ROOT.is_dir()
_RESULTS_ROOT = load_config().results_root

pytestmark = pytest.mark.data

skip_reason = (
    "ROWII_DATA_ROOT is unset or does not point at an existing directory"
)


@pytest.fixture(scope="module")
def data_root() -> Path:
    if not _HAS_DATA_ROOT:
        pytest.skip(skip_reason)
    return _DATA_ROOT


def test_discover_finds_250526_tu_run_with_four_streams_of_twelve_files_each(
    data_root: Path,
) -> None:
    index = discover(data_root)
    tu_runs = [r for r in index.runs if r.name == "250526-tu"]
    assert len(tu_runs) == 1, (
        f"expected exactly one '250526-tu' run, found {len(tu_runs)} "
        f"(all runs: {sorted(r.name for r in index.runs)})"
    )

    run = tu_runs[0]
    expected_streams = {
        "RAWGeneratorMic__0",
        "RAWTurbineMic__1",
        "RAWGeneratorVib__2",
        "RAWTurbineVib__3",
    }
    assert set(run.files.keys()) == expected_streams
    for stream in expected_streams:
        n_files = len(run.files[stream])
        assert n_files == 12, f"stream {stream}: expected 12 files, found {n_files}"


def test_discover_finds_at_least_one_pu_run(data_root: Path) -> None:
    index = discover(data_root)
    pu_runs = [r for r in index.runs if "pu" in r.name]
    assert len(pu_runs) >= 1, "expected at least one '*pu*' run in the discovered index"


def test_discover_finds_every_day_tree_with_its_dayid_prefix(data_root: Path) -> None:
    # Spec-named headline runs, verified directly against the campaign's own
    # four day trees (each `illwerke-<dayid>` gets its own prefix -- see
    # `rowii.io.dataset` module docstring): 25.06 (existing, legacy TU/PU
    # session names), 29.06 (TU includes the ~37-min PH hold, still one
    # continuous run -- no burst-file gap), 01.07 (ALL FOUR modes, including
    # the day's own headline `TU_PH_TU` session). 27.06 has no Betriebsdaten
    # at all (see the dedicated test below) but its burst session must still
    # be discoverable.
    index = discover(data_root)
    names = {r.name for r in index.runs}

    assert "250526-tu" in names
    assert "290626-tu" in names
    assert "010726-tu_ph_tu" in names
    assert any(n.startswith("270626-pu_ph_pu_ph_pu_ph") for n in names), (
        f"expected a 270626-pu_ph_pu_ph_pu_ph* run, found: {sorted(names)}"
    )


def test_discover_scopes_betriebsdaten_per_day_tree_on_real_data(data_root: Path) -> None:
    # 27.06 has zero Betriebsdaten (spec §1: "none (photo only)") while the
    # other three days each have their own full day's worth -- a day tree with
    # no SCADA at all must have no entry (or an empty list), never silently
    # inherit a sibling day's files.
    index = discover(data_root)

    day_270626 = next(
        (r.day_root for r in index.runs if r.name.startswith("270626-")), None
    )
    assert day_270626 is not None, "expected at least one 270626-* run"
    assert index.betriebsdaten_by_day.get(day_270626, []) == []

    for prefix in ("250526-", "290626-", "010726-"):
        day_root = next(
            (r.day_root for r in index.runs if r.name.startswith(prefix)), None
        )
        assert day_root is not None, f"expected at least one {prefix}* run"
        assert index.betriebsdaten_by_day.get(day_root, []), (
            f"expected non-empty Betriebsdaten for day tree {day_root} ({prefix}*)"
        )


def test_tu_mic_file_header_reports_plausible_audio_rate_and_channels(data_root: Path) -> None:
    index = discover(data_root)
    run = next(r for r in index.runs if r.name == "250526-tu")
    first_mic_file = sorted(
        run.files["RAWGeneratorMic__0"], key=lambda f: f.start_utc_hint
    )[0]

    header = read_header(first_mic_file.path)

    assert 45_000 <= header.sample_rate_hz <= 55_000, (
        f"mic sample rate {header.sample_rate_hz} Hz outside the expected ~50 kHz range"
    )
    assert len(header.channel_names) >= 4, (
        f"mic file has {len(header.channel_names)} channels, expected >= 4"
    )


def test_tu_vib_file_header_reports_plausible_vibration_rate_and_six_channels(
    data_root: Path,
) -> None:
    index = discover(data_root)
    run = next(r for r in index.runs if r.name == "250526-tu")
    first_vib_file = sorted(
        run.files["RAWGeneratorVib__2"], key=lambda f: f.start_utc_hint
    )[0]

    header = read_header(first_vib_file.path)

    assert 9_000 <= header.sample_rate_hz <= 11_000, (
        f"vib sample rate {header.sample_rate_hz} Hz outside the expected ~10 kHz range"
    )
    assert len(header.channel_names) == 6, (
        f"vib file has {len(header.channel_names)} channels, expected exactly 6"
    )


def test_betriebsdaten_file_contains_all_gt_channel_names(data_root: Path) -> None:
    index = discover(data_root)
    assert index.betriebsdaten, "expected at least one discovered Betriebsdaten file"

    target = next(
        (p for p in index.betriebsdaten if p.name == "2026-06-25_05-00-00.dat"),
        index.betriebsdaten[0],
    )

    header = read_header(target)

    for key, channel_name in GT_CHANNELS.items():
        assert channel_name in header.channel_names, (
            f"GT_CHANNELS[{key!r}] = {channel_name!r} not found in "
            f"{target.name}; available channels: {header.channel_names}"
        )


def test_fitted_detector_apply_equals_fit_on_cached_run(data_root: Path) -> None:
    """Same-day apply == fit labels on a real cached PreparedRun (spec D1 gate).

    No dedicated real-run fixture exists yet in this file, so this builds the
    `PreparedRun` inline via `prepare_run(..., use_cache=True)`, exactly like
    every other test here calls `discover(data_root)` fresh -- for the
    010726-tu_ph_tu fusion variant, which already has an on-disk cache entry
    (`results/cache/010726-tu_ph_tu--fusion.npz`), so this stays fast (cache
    hit, no raw file re-read). Compacts to valid windows exactly like
    `scripts/run_step2.py::_detected_labels`, then checks that
    `FittedDetector.apply` reproduces `FittedDetector.fit`'s own labels when
    given the SAME features -- the same-day-equivalence contract already
    covered on synthetic data by `tests/test_detect_e2e.py::TestFittedDetector`,
    proven here end-to-end on a real multi-stream recording.
    """
    cfg = load_config()
    index = discover(data_root)
    run = next(r for r in index.runs if r.name == "010726-tu_ph_tu")
    prepared = prepare_run(run, "fusion", cfg, use_cache=True)

    valid = prepared.valid_mask
    feats = prepared.features[valid]
    grid = WindowGrid(
        t0_ns=prepared.grid.t0_ns,
        window_ns=prepared.grid.window_ns,
        n_windows=int(valid.sum()),
    )
    det, fit_result = FittedDetector.fit(feats, grid, cfg.detect, clusterer="kmeans")
    applied = det.apply(feats, grid)
    np.testing.assert_array_equal(applied.frame_labels, fit_result.frame_labels)


# ---------------------------------------------------------------------------
# Task 10 (D6b/c): true-UTC time axis, the invariance regression on real data --
# labels/scores/FAR/GT-eval-window-counts must be bit-for-bit unaffected by moving
# the whole pipeline from the raw DAQ axis onto true UTC. Both tests are cache-hit
# fast (no raw file re-read): `250526-tu--fusion.npz`/`250526-tu--audio.npz` are
# already on disk with a RAW-axis `grid_t0_ns` (written before this task), so
# `prepare_run`'s cache-hit path (D4) exercises the override for real here, not
# just on the synthetic fixture in `tests/test_pipeline.py`.
# ---------------------------------------------------------------------------

_FAR_TABLE_PATH = (
    _RESULTS_ROOT / "step2" / "within-day" / "250526-tu" / "fusion-detected"
    / "per-state-knn" / "far_table.csv"
)
_HAS_FAR_TABLE = _FAR_TABLE_PATH.is_file()
skip_reason_far_table = f"{_FAR_TABLE_PATH} not present"


def test_within_day_far_table_matches_persisted_after_true_utc_fix(data_root: Path) -> None:
    """Re-run the within-day sweep combo 250526-tu / fusion / detected-labels /
    per-state / knn IN MEMORY via `rowii.anomaly.sweep.run_sweep` (the exact
    combo `scripts/run_step2.py`'s default CLI arguments produce, reproduced here
    via its own private `_detected_labels`) and assert the resulting `far_table` is
    numerically identical to the ALREADY-PERSISTED `results/step2/within-day/
    250526-tu/fusion-detected/per-state-knn/far_table.csv` (note: the directory
    name is `<conditioning>-<scorer>` = "per-state-knn", not the brief's literal
    "knn-per-state" -- see the task report). `far_table` carries no time/UTC
    column at all (label, counts, FAR, threshold, ...) -- window INDICES and
    VALUES, never absolute timestamps -- so this is a direct, real-data proof of
    D6's bit-identity guarantee: the grid this sweep runs against is now true-UTC
    (D2/D4), yet the FAR table matches a file generated back when the same grid
    was still on the raw DAQ axis, numeral for numeral.
    """
    if not _HAS_FAR_TABLE:
        pytest.skip(skip_reason_far_table)

    import run_step2

    from rowii.anomaly.sweep import SweepConfig, run_sweep

    cfg = load_config()
    index = discover(data_root)
    run = next(r for r in index.runs if r.name == "250526-tu")

    prepared = prepare_run(run, "fusion", cfg, use_cache=True)
    labels = run_step2._detected_labels(prepared, cfg)
    sweep_cfg = SweepConfig(alpha=0.05, top_k=20, conditioning="per-state", scorer="knn")
    result = run_sweep(prepared, labels, sweep_cfg)

    far_table = result.far_table.copy()
    far_table["label"] = far_table["label"].astype(str)  # matches _write_sweep_outputs
    persisted = pd.read_csv(_FAR_TABLE_PATH)

    # check_dtype=False: a CSV round-trip infers plain numeric dtypes for columns
    # that are float64-with-NaN in memory (pandas' own read_csv type inference,
    # unrelated to this task) -- values (NaN-equal by assert_frame_equal's default)
    # are what D6 actually claims invariant, not incidental dtype.
    pd.testing.assert_frame_equal(
        far_table.reset_index(drop=True), persisted.reset_index(drop=True), check_dtype=False
    )


def test_250526_tu_gt_eval_window_counts_match_historical_readme_values(
    data_root: Path,
) -> None:
    """Recompute GT labels for 250526-tu (audio variant) on the NEW true-UTC axis
    (`scripts/run_step1.py::load_run_gt`, now D3-fixed) and assert the per-state
    eval-window (non-"unknown") counts equal README.md's historical Step-1-grid
    values (122 standstill / 403 transition / 7750 turbine = 8275 total,
    "Step-1 grid results" section) -- proving SCADA (D3, its own independently
    derived offset) and the audio grid (D2, the run's own offset) moved together
    under this run's true-UTC shift. A genuine misalignment (e.g. SCADA selected
    from the wrong file set, or one offset silently diverging from the other)
    would shift these counts, not just their rendered times -- this is the
    real-data equivalent of `tests/test_sweep.py`'s/`tests/test_detect_e2e.py`'s
    synthetic D6a invariance tests, specifically for the SCADA/GT side D6a alone
    does not cover.
    """
    import run_step1

    cfg = load_config()
    index = discover(data_root)
    run = next(r for r in index.runs if r.name == "250526-tu")
    betriebsdaten = index.betriebsdaten_by_day.get(run.day_root, [])

    prepared = prepare_run(run, "audio", cfg, use_cache=True)
    _scada, gt = run_step1.load_run_gt(
        run, betriebsdaten, prepared.grid, cfg, prepared.valid_mask
    )

    counts = gt["state"].value_counts()
    assert int(counts.get("standstill", 0)) == 122
    assert int(counts.get("transition", 0)) == 403
    assert int(counts.get("turbine", 0)) == 7750
    assert int((gt["state"] != "unknown").sum()) == 8275
