# Step-2 Package 9: Once-Calibrated Operation, Named States, Transition/Dwell Handling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Operationalise Stefan's three deployment-ergonomics asks as refinements of the shipped best system, each honest about its own limits: (D1) calibrate ONCE per instrumentation era and recalibrate only when a label-free drift sentinel fires — a retrospective, day-granular SIMULATION, never an online claim; (D2) name detected states end to end by reusing the existing majority-vote primitive, persisted as an OPTIONAL snapshot member; (D3) treat transitions and minimum dwell as a first-class step — a SCADA transition taxonomy, a data-grounded `min_dwell` sweep, and monitor-level `near_transition` visibility with an optional suppression ablation. No partner number enters any computation; every claim is computed from our own caches and reported honestly, negatives included.

**Architecture:** One tiny wrapper `rowii.eval.metrics.derive_state_names` (reusing `_majority_mapping`) backs a new OPTIONAL format-v2 snapshot member `state_names` (mirroring the `level_recal_medians` pattern EXACTLY in `rowii.runtime.snapshot`), wired at `scripts/run_step2.py`'s `--save-snapshot` block. `scripts/monitor.py` surfaces names (`state_name` alarms column, `mapped_mode` segments column, named timeline/notes) AND transition visibility (`near_transition` alarms column + optional `--suppress-transition-alarms` ablation). `scripts/analyze_days.py` gains a `transitions` subcommand (D3a taxonomy on our own `gt_labels`); a new small driver `scripts/sweep_min_dwell.py` grounds the dwell default (D3b). One greenfield src module `rowii.anomaly.sentinels` (the package's real cost centre) provides two label-free drift sentinels, driven by `scripts/run_once_calibrated.py`, the D1 replay driver. Everything composes existing machinery (`ModeBank`, `levelrecal`, `FittedDetector.fit_pooled`, `fit_snapshot_from_parts`, `evaluate`, `eval_events`) — no new encoders, no streaming, no retraining at monitor time.

**Tech Stack:** unchanged (numpy, pandas, scipy, sklearn `adjusted_rand_score`, hmmlearn; matplotlib Agg for figures; `pyarrow` for the alarms parquet). No new dependency.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-22-step2-package9-once-naming-transitions.md` **§3 (D1–D3) + Amendment A1 (all findings A1.1–A1.8 adopted)** — the amendment OVERRIDES §3 where they conflict (esp. A1.1 pinned sentinel thresholds + firewall-safe MAD, A1.2 pinned replay run set, A1.3 `_ALARM_COLUMNS` test updates, A1.4 `state_name` ALWAYS present + END column order, A1.5 the `derive_state_names` wrapper, A1.6 FAR common-population, A1.7 snapshot-build `--test-run`, A1.8 the underspecified-point pins).
- Gates per task (ALL must be green before commit): `.venv/bin/python -m pytest tests/ -q -m "not data"`, `.venv/bin/ruff check .`, `.venv/bin/mypy src scripts` (must print `Success: no issues`).
- **Tests FIRST** (RED before GREEN) for every task. Deterministic seeded fixtures; NO real data in tests; CLI-level tests use the established monkeypatch seams (`tests/test_monitor_cli.py`'s monkeypatched `discover`/`load_config`/`prepare_run` + `dataclasses.replace` on the real snapshot; `tests/test_step2_pooled_cli.py`'s `_install_fakes`; `tests/test_run_modebank.py`/`tests/test_analyze_days.py`'s monkeypatched `_run_*` seams; `tests/test_runtime_snapshot.py`'s `_detector_and_parts`/`_from_parts`).
- **Implementer verification MUST use the temp-file + `&&` pattern**, never `pytest | tail` (a pipe hides the exit code): e.g. `.venv/bin/python -m pytest tests/test_metrics.py -q > /tmp/p9_t1.txt 2>&1 && tail -5 /tmp/p9_t1.txt`.
- **No `Co-Authored-By` lines** in any commit (Stefan is sole author).
- **Firewall (BINDING TEST RULE, inherited A1.8):** NO partner-derived numeric constant may appear as an expected value in `src/`, `scripts/`, or test fixtures. Every D1 sentinel threshold is derived from the commissioning (B1) pool ALONE (A1.1); the standard-statistics constants (`97.5` percentile, `B = 1000` bootstrap, the factor `3` in `3·MAD`) are named, derived-from-nothing-partner-published constants, never asserted against a partner figure; the `3.0` dB stability cutoff stays P8's named constant and is not reintroduced anywhere in P9. The `080726` pillar-3 ground truth comes from OUR OWN `docs/groundtruth/080726_events_{pu,st}.csv`, never a partner detection count.
- **Attribution (spec §4):** every analysis type echoing the partner's work (D1 sentinel idea, D3a taxonomy) carries a one-line attribution in the script docstring AND the digest; all numbers computed from OUR caches; no partner JSON/number is read by any code.
- **Model policy (Stefan, spec §6):** implementation/tests/readers on **sonnet**; adversarial per-task and whole-branch/spec reviews on **opus**; **fable** only if a review blocks twice. Implementer dispatches MUST forbid Agent-tool use (no self-orchestration / branch races).
- Scripts NEVER import a sibling script's internals (duplicate-with-rationale, the repo rule); `src/rowii/` modules are imported normally.

### Verified seam facts (recorded here so every interface below is real)

- **`_majority_mapping` (VERIFIED, `rowii.eval.metrics`, line 138):** `_majority_mapping(gt_states: pd.Series, pred: np.ndarray) -> dict[int, str]` — each cluster id present in `pred` maps INDEPENDENTLY to the majority GT state among its own windows, ties broken by `pandas.Series.idxmax` first-occurrence order. `evaluate`'s mask drops ONLY `gt.state == "unknown"` (line 202, `_UNKNOWN`); D2 needs BOTH `{unknown, transition}` masked (A1.5), so `derive_state_names` masks itself and reuses `_majority_mapping` on the masked pair — `_majority_mapping` stays UNCHANGED. `scripts/apply_detector.py` already calls `_majority_mapping` for its `mapped_mode` column (its own fallback is `""`; D2 chooses `cluster-<id>` instead, A1.5).
- **Snapshot optional-member pattern (VERIFIED, `rowii.runtime.snapshot`):** `MonitorSnapshot` is `@dataclass(frozen=True)` with TWO optional format-v2 members already present, each defaulting to `None`: `session_stats` (line 230, backed by two npz arrays + a `meta` scalar entry) and `level_recal_medians` (line 238, a `dict[str, float]` living ENTIRELY in the `meta` JSON — no npz array). `state_names: dict[int, str] | None` follows `level_recal_medians` EXACTLY: meta-only, present only when carried, NO version bump (`SNAPSHOT_FORMAT_VERSION` stays 2; `_SUPPORTED_FORMAT_VERSIONS == (1, 2)`). `_meta_dict` (796) adds entries conditionally; `load_snapshot` (932) reads them back (`level_recal_medians` via `{str(k): float(v)}`; `state_names` via `{int(k): str(v)}` since keys are label ids). `fit_snapshot_from_parts` (551) computes `fitted_ids = np.asarray(smoother._fitted_ids, dtype=np.int64)` at line 712 — the `state_names` key-validation must run AFTER that line. `session_stats`/`level_recal_medians` are mutually exclusive (line 688); `state_names` has NO mutual-exclusivity (it is a naming layer, coexists with either).
- **run_step2 `--save-snapshot` block (VERIFIED, `scripts/run_step2.py` 3961–4044):** builds the snapshot via `fit_snapshot_from_parts(detector, snapshot_references, snapshot_cal_scores, snapshot_thresholds, ..., session_stats=snapshot_session_stats, level_recal_medians=level_recal_anchor_used)`. In scope at that block: `detector` (its `smoother._fitted_ids == arange(k)`), `pool_fit` + `pool_fit_labels` (3711/3763, int64 detected labels per pooled fit row), `scada_by_run` (3854, `dict[str, DataFrame|None]`), `missing_fit_scada` (3860, the A1.8 GT-skip seam — non-empty iff any fit run lacks Betriebsdaten), `fit_run_names`, `cfg`. Helpers: `_gt_state_labels(scada, cfg) -> np.ndarray` (868, object GT state strings), `_pool_row_labels(pool, labels_per_run) -> np.ndarray` (3328, **int64** — needs an object-dtype sibling for GT strings, exactly as `scripts/run_modebank.py::_pool_gt_labels` (240) already established).
- **monitor alarm/segments/timeline/notes seams (VERIFIED, `scripts/monitor.py`):** `_ALARM_COLUMNS` (204) is a 9-tuple ending `"threshold_source"`; `_alarms_frame` (933) builds the frame explicitly and returns `frame[list(_ALARM_COLUMNS)]`; `_state_segments` (967) drops `mapped_mode` for lack of a mapping (`cluster` → `cluster_id`); `_timeline_markdown` (1000) writes `desc = f"state {cluster_id}"`; `_notes_markdown` (1234) renders `| {row.state} | …`; `main` writes outputs at 1580–1599. `_Verdicts` (436) is frozen but its `.alarm`/`.role` are mutable numpy arrays (in-place assignment is the codebase idiom, e.g. `_mark_unknown_states` 487). `_INVALID_LABEL == -1`. `snapshot.min_dwell_s` (185) is the W scale (A1.8, T3).
- **test_monitor_cli asserts (VERIFIED):** local `_ALARM_COLUMNS` list at line 58; `assert list(alarms.columns) == _ALARM_COLUMNS` at 198/949/1092 (fixed by editing the local list); `assert list(segments.columns) == ["start_utc","end_utc","duration_s","cluster_id"]` at 238 STAYS (the default fixture's snapshot has NO `state_names`, so `mapped_mode` is absent — D2(c) is conditional, unlike the always-present `state_name`); `alarm_segments.csv` cols at 467 are unrelated. `dataclasses.replace` is imported (25); `_two_state_prepared`/`_fake_index` build the fixtures.
- **detector/dwell (VERIFIED, `rowii.state.detect`):** `FittedDetector.fit_pooled(pooled_features, cfg, *, k, clusterer="kmeans") -> FittedDetector`; `_finish` (350) computes `min_dwell = max(1, round(self.min_dwell_s / window_s))` then `duration_filter`. `Config`/`DetectConfig` are frozen (`rowii.config`, `min_dwell_s` default 5.0) → sweep via `dataclasses.replace`.
- **scada transition machinery (VERIFIED, `rowii.scada.labels`):** `STATES == ("standstill","turbine","pump","transition","phase-shifter")`; `gt_labels(scada, rules, *, window_s)["state"]` yields the state sequence including `"transition"` runs (from `_apply_ramp` + `_apply_transition_buffer`); `_contiguous_true_runs(mask)` (270) gives `[start, stop)` pairs; `GT_CHANNELS["power"] == "1_P_Ist"`; the ramp uses centred `dP/dt = (power[i+1]-power[i-1])/(2*window_s)` (321). Known states = `set(STATES) - {"transition"}` (`"unknown"` is not in `STATES`).
- **analyze_days seams (VERIFIED, `scripts/analyze_days.py`):** argparse subparsers via `sub.add_parser("name", …)` + `main` `if args.command == "…"` dispatch (2052/2199); `_run_features_and_gt(run_name, variant, cfg, index, *, use_cache) -> _RunFeatures(run_name, features, gt_states, segment_ids, feature_names, has_gt)` is the monkeypatch seam (323); `_run_scada_or_none(prepared, run, index)` (280); `_level_db_factor` (×20 `_log_rms` / ×10 `_band_`/`_octave_`, 407); `_block_bootstrap_ci(values, segment_ids, n_boot, seed)` (431); `_MIC_STREAMS == ("RAWGeneratorMic__0","RAWTurbineMic__1")`, `_VIB_STREAMS == ("RAWGeneratorVib__2","RAWTurbineVib__3")` (193); `_stream_level_columns`/`_levels_by_stream` (1018/1032).
- **modebank / sentinel deps (VERIFIED):** `ModeBank.fit(fit_features, fit_labels, calib_features, calib_labels, *, family, alpha, feature_names, min_ref=20, k=5, …) -> ModeBank`; `ModeBank.assign(features).no_mode_fits` is `(W,)` bool; `ModeBank.low_confidence_modes` is `tuple[str, ...]` (the under-fire caveat). `rowii.pipeline.stream_columns(feature_names, stream) -> np.ndarray`; `rowii.anomaly.levelrecal.level_columns(feature_names) -> list[int]`; `rowii.io.dataset.run_utc_offset_ns(run)`.
- **Branch:** `feat/step2-package9-once-naming-transitions` at `291b6c5` (base main `5c92b90`, post-P8).

---

### Task 1: `rowii.eval.metrics.derive_state_names` — commissioning-time name map (D2, A1.5)

**Files:** Modify `src/rowii/eval/metrics.py` (add ONE function + a module constant; `_majority_mapping` UNCHANGED) · Test extend `tests/test_metrics.py`

**Interfaces:**
- Consumes (in-module): `_majority_mapping(gt_states: pd.Series, pred: np.ndarray) -> dict[int, str]`, `_UNKNOWN`; adds `_TRANSITION = "transition"` (mirrors `_UNKNOWN`).
- Produces: `derive_state_names(gt_states: np.ndarray, pred: np.ndarray, fitted_ids: Iterable[int], *, min_plurality: float = 0.5) -> dict[int, str]` — (1) mask `{unknown, transition}` from BOTH arrays BEFORE the vote (A1.5, narrower than `evaluate`'s `unknown`-only mask); (2) inner vote = `_majority_mapping(pd.Series(gt_masked), pred_masked)`; (3) per `cid` in `fitted_ids`, keep the winner iff the cluster is PRESENT in the masked pred (equivalently: has ≥1 GT-known window) AND its winner covers `>= min_plurality` of its masked windows, else fall back to `f"cluster-{cid}"`; (4) fill over ALL `fitted_ids`. Returns `{int(cid): name}`.
- Binding: home is `metrics.py` (it wraps `_majority_mapping`, lives beside it, adds no dependency — the correct home per the code); `_majority_mapping` is not modified (A1.5).

- [ ] RED extend `tests/test_metrics.py` (append; `derive_state_names` + the three fallback conditions + fill-over-fitted_ids + the transition-mask delta vs `evaluate`):
```python
def test_derive_state_names_maps_clean_two_mode() -> None:
    from rowii.eval.metrics import derive_state_names
    gt = np.array(["turbine"] * 30 + ["pump"] * 30, dtype=object)
    pred = np.array([0] * 30 + [1] * 30, dtype=np.int64)
    names = derive_state_names(gt, pred, fitted_ids=[0, 1])
    assert names == {0: "turbine", 1: "pump"}


def test_derive_state_names_masks_both_unknown_and_transition() -> None:
    """A1.5: BOTH masked before the vote -- narrower than evaluate's unknown-only."""
    from rowii.eval.metrics import derive_state_names
    gt = np.array(["turbine", "transition", "unknown", "turbine", "turbine"], dtype=object)
    pred = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    names = derive_state_names(gt, pred, fitted_ids=[0])
    # only the 3 turbine windows count; cluster 0's plurality is 3/3 -> turbine.
    assert names == {0: "turbine"}


def test_derive_state_names_fallback_cluster_absent_from_masked_pred() -> None:
    from rowii.eval.metrics import derive_state_names
    # cluster 1 appears ONLY on transition/unknown windows -> zero GT-known -> fallback.
    gt = np.array(["turbine", "turbine", "transition", "unknown"], dtype=object)
    pred = np.array([0, 0, 1, 1], dtype=np.int64)
    names = derive_state_names(gt, pred, fitted_ids=[0, 1])
    assert names == {0: "turbine", 1: "cluster-1"}


def test_derive_state_names_fallback_sub_50pct_plurality() -> None:
    from rowii.eval.metrics import derive_state_names
    # cluster 0's masked windows split 2 turbine / 3 pump -> winner (pump) = 3/5 = 60% >= 50%.
    gt_ok = np.array(["turbine", "turbine", "pump", "pump", "pump"], dtype=object)
    pred = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    assert derive_state_names(gt_ok, pred, [0]) == {0: "pump"}
    # now a true <50% plurality: 2 turbine / 2 pump / 1 standstill -> max 2/5 = 40% -> fallback.
    gt_tie = np.array(["turbine", "turbine", "pump", "pump", "standstill"], dtype=object)
    assert derive_state_names(gt_tie, pred, [0]) == {0: "cluster-0"}


def test_derive_state_names_fills_over_all_fitted_ids() -> None:
    from rowii.eval.metrics import derive_state_names
    gt = np.array(["turbine"] * 10, dtype=object)
    pred = np.array([0] * 10, dtype=np.int64)
    # fitted id 2 never appears in pred -> filled as its bare name.
    names = derive_state_names(gt, pred, fitted_ids=[0, 1, 2])
    assert names == {0: "turbine", 1: "cluster-1", 2: "cluster-2"}


def test_derive_state_names_all_fallback_when_no_gt_known() -> None:
    from rowii.eval.metrics import derive_state_names
    gt = np.array(["unknown", "transition", "unknown"], dtype=object)
    pred = np.array([0, 1, 0], dtype=np.int64)
    assert derive_state_names(gt, pred, [0, 1]) == {0: "cluster-0", 1: "cluster-1"}
```
- [ ] Run RED: `.venv/bin/python -m pytest tests/test_metrics.py -q -k derive_state_names > /tmp/p9_t1.txt 2>&1 && tail -15 /tmp/p9_t1.txt` → `ImportError`/`AttributeError` (function absent).
- [ ] GREEN `src/rowii/eval/metrics.py` (add beside `_majority_mapping`; real, no placeholder):
```python
_TRANSITION = "transition"
_NAMING_EXCLUDED = (_UNKNOWN, _TRANSITION)


def derive_state_names(
    gt_states: np.ndarray,
    pred: np.ndarray,
    fitted_ids: Iterable[int],
    *,
    min_plurality: float = 0.5,
) -> dict[int, str]:
    """Cluster id -> operating-mode name, the commissioning-time map D2 persists as
    an optional snapshot member (spec D2(a) + A1.5). Reuses `_majority_mapping`; the
    {unknown, transition} windows are masked BEFORE the vote (A1.5, narrower than
    `evaluate`'s unknown-only mask). A cluster keeps its majority name only when it
    is present in the masked prediction AND its winner covers >= `min_plurality` of
    its masked windows; otherwise -- absent from the masked pred, zero GT-known
    windows, or a sub-plurality winner -- it falls back to the bare `cluster-<id>`
    name (English, matching the repo's English-only artifact rule and the already-
    English `labels.STATES` strings). The map is filled over ALL `fitted_ids` (a
    cluster can carry a timeline name even without an alarming threshold)."""
    gt_arr = np.asarray(gt_states, dtype=object)
    pred_arr = np.asarray(pred)
    keep = ~np.isin(gt_arr, _NAMING_EXCLUDED)
    gt_m, pred_m = gt_arr[keep], pred_arr[keep]
    mapping = _majority_mapping(pd.Series(gt_m), pred_m) if gt_m.size else {}
    names: dict[int, str] = {}
    for raw in fitted_ids:
        cid = int(raw)
        in_cluster = pred_m == cid
        n_c = int(in_cluster.sum())
        if n_c == 0 or cid not in mapping:
            names[cid] = f"cluster-{cid}"
            continue
        winner = mapping[cid]
        frac = float(np.mean(gt_m[in_cluster] == winner))
        names[cid] = winner if frac >= min_plurality else f"cluster-{cid}"
    return names
```
  Add `from collections.abc import Iterable` to the imports (top of `metrics.py`).
- [ ] Run GREEN gates (all three) with temp-file+&&; expect pass + `Success: no issues`.
- [ ] Commit `feat: derive_state_names -- commissioning name map reusing _majority_mapping (P9 D2/A1.5)`.

---

### Task 2: snapshot v2 `state_names` member + run_step2 `--save-snapshot` wiring (D2, A1.8)

**Files:** Modify `src/rowii/runtime/snapshot.py` (field + `_meta_dict` + `load_snapshot` + `fit_snapshot_from_parts` kwarg/validation), `scripts/run_step2.py` (object-dtype pool gather + state_names computation in the `--save-snapshot` block) · Tests extend `tests/test_runtime_snapshot.py`, `tests/test_step2_pooled_cli.py`

**Interfaces:**
- `MonitorSnapshot` gains `state_names: dict[int, str] | None = None` (OPTIONAL v2 member, NO version bump — mirrors `level_recal_medians`: meta-only, no npz array). `_meta_dict` adds `"state_names": {str(k): v}` exactly when present; `load_snapshot` reads it back as `{int(k): str(v)}`; `save_snapshot` needs no new array member. Keyed over the `fitted_ids` id space, NOT the threshold-label subset — orthogonal to `references`/`thresholds`, and the key-agreement invariant does NOT apply to it.
- `fit_snapshot_from_parts(..., state_names: dict[int, str] | None = None)` — validates every key is one of the detector's fitted ids (the geometry-guard posture; validation runs AFTER `fitted_ids` is computed at line 712). **NO mutual-exclusivity** against `session_stats`/`level_recal_medians` (a naming layer, coexists with either).
- `scripts/run_step2.py`: `_pool_row_gt_labels(pool: PoolResult, gt_by_run: dict[str, np.ndarray]) -> np.ndarray` — object-dtype sibling of `_pool_row_labels` (duplicated from `scripts/run_modebank.py::_pool_gt_labels`, script-sibling rule). In the `--save-snapshot` block: `state_names = None` iff `missing_fit_scada` (A1.8 GT-skip seam), else `derive_state_names(_pool_row_gt_labels(pool_fit, {n: _gt_state_labels(scada_by_run[n], cfg) for n in fit_run_names}), pool_fit_labels, [int(i) for i in detector.smoother._fitted_ids])`, passed as `state_names=` to `fit_snapshot_from_parts`. Import `derive_state_names` from `rowii.eval.metrics`.

- [ ] RED extend `tests/test_runtime_snapshot.py` (mirror `test_v2_round_trip_with_session_stats` / `_from_parts`):
```python
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
```
- [ ] RED extend `tests/test_step2_pooled_cli.py` (the default `_default_index()` has NO Betriebsdaten -> the A1.8 GT-skip path -> `state_names is None`; plus a direct unit test of the object-dtype gather):
```python
def test_save_snapshot_state_names_none_without_gt(tmp_path, monkeypatch) -> None:
    """Default fixture has no Betriebsdaten (missing_fit_scada) -> state_names=None (A1.8)."""
    import run_step2
    from rowii.runtime.snapshot import load_snapshot
    prepared = _pooled_prepared()
    _install_fakes(monkeypatch, tmp_path, prepared, _default_index())
    snap_path = _out_dir(tmp_path) / "snap.npz"
    assert run_step2.main([*_BASE_ARGS, "--save-snapshot", str(snap_path)]) == 0
    assert load_snapshot(snap_path).state_names is None


def test_pool_row_gt_labels_object_dtype_alignment() -> None:
    import run_step2
    prepared = _pooled_prepared()
    hand = _hand_pipeline(prepared)  # existing helper -> pooled fit sides
    gt_by_run = {name: np.array(["turbine"] * len(p.features), dtype=object)
                 for name, p in prepared.items()}
    out = run_step2._pool_row_gt_labels(hand.pool_fit, gt_by_run)
    assert out.dtype == object
    assert out.shape[0] == hand.pool_fit.features.shape[0]
    assert set(out.tolist()) == {"turbine"}
```
- [ ] Run RED with temp-file+&&; expect failures (field/kwarg/helper absent).
- [ ] GREEN `src/rowii/runtime/snapshot.py`:
  - Add the field after `level_recal_medians` (docstring notes: keyed over fitted ids, meta-only, NO version bump, NO mutual-exclusivity): `state_names: dict[int, str] | None = None`.
  - `fit_snapshot_from_parts`: add the kwarg `state_names: dict[int, str] | None = None`; AFTER `fitted_ids = np.asarray(smoother._fitted_ids, dtype=np.int64)` (line ~712) insert:
```python
    if state_names is not None:
        fitted_set = {int(i) for i in fitted_ids.tolist()}
        stray = sorted(int(k) for k in state_names if int(k) not in fitted_set)
        if stray:
            raise ValueError(
                f"state_names key(s) {stray} are not fitted label ids "
                f"{sorted(fitted_set)} -- a stored name must key onto a real "
                f"detected state (D2/A1.5); state_names is orthogonal to the "
                f"threshold-label subset but still lives in the fitted-id space"
            )
```
    and pass `state_names=(dict(state_names) if state_names is not None else None)` into the `MonitorSnapshot(...)` constructor.
  - `_meta_dict`: append `if snapshot.state_names is not None: meta["state_names"] = {str(k): v for k, v in snapshot.state_names.items()}`.
  - `load_snapshot`: after the `level_recal_medians` reconstruction add `raw_names = meta.get("state_names"); state_names = ({int(k): str(v) for k, v in raw_names.items()} if raw_names is not None else None)`, and pass `state_names=state_names` to the constructor.
- [ ] GREEN `scripts/run_step2.py`:
  - Add `_pool_row_gt_labels` right after `_pool_row_labels` (object-dtype, duplicated from run_modebank's `_pool_gt_labels`):
```python
def _pool_row_gt_labels(pool: PoolResult, gt_by_run: dict[str, np.ndarray]) -> np.ndarray:
    """Per stacked pool row, the GT mode-name STRING of its source window -- the
    object-dtype sibling of `_pool_row_labels` (duplicated from
    `scripts/run_modebank.py::_pool_gt_labels`, script-sibling rule). Feeds
    `derive_state_names` at snapshot save (D2)."""
    out = np.empty(pool.features.shape[0], dtype=object)
    for member_idx, member in enumerate(pool.members):
        mask = pool.run_index == member_idx
        out[mask] = gt_by_run[member.run_name][pool.window_index[mask]]
    return out
```
  - Import `from rowii.eval.metrics import derive_state_names`.
  - In the `--save-snapshot` block (~3961), BEFORE the `fit_snapshot_from_parts(...)` call compute (docstring the A1.8 rule: any fit run lacking Betriebsdaten -> None):
```python
        state_names: dict[int, str] | None = None
        if not missing_fit_scada:
            gt_state_by_run = {
                name: _gt_state_labels(scada_by_run[name], cfg) for name in fit_run_names
            }
            pool_fit_gt = _pool_row_gt_labels(pool_fit, gt_state_by_run)
            fitted_ids = [int(i) for i in np.asarray(detector.smoother._fitted_ids)]
            state_names = derive_state_names(pool_fit_gt, pool_fit_labels, fitted_ids)
```
    and add `state_names=state_names,` to the `fit_snapshot_from_parts(...)` kwargs.
- [ ] Run GREEN gates; expect pass + `Success: no issues`.
- [ ] Commit `feat: snapshot v2 state_names member + run_step2 --save-snapshot wiring (P9 D2/A1.8)`.

---

### Task 3: monitor — named states + `near_transition` + optional suppression (D2 surfacing + D3c, A1.4)

> Merges spec §6 steps 2 (D2 surfacing) and 4 (D3c): both edit the SAME monitor seams (`_ALARM_COLUMNS`, `_alarms_frame`, the output block) and the SAME `test_monitor_cli.py` asserts, so doing them once avoids editing `_ALARM_COLUMNS` and its exact-equality tests twice (a resolved ambiguity, flagged for the orchestrator).

**Files:** Modify `scripts/monitor.py` · Test extend `tests/test_monitor_cli.py`

**Interfaces:**
- `_ALARM_COLUMNS` gains `"near_transition"` then `"state_name"` at the END (A1.4 order) → 11-tuple. BOTH always present (A1.4: `state_name` fallback `cluster-<id>`; `near_transition` is a detected-state property, no snapshot dependency). The "exactly as today" sentence is retracted for these two columns.
- `_near_transition_mask(labels: np.ndarray, valid_mask: np.ndarray, window_ns: int, w_seconds: float) -> np.ndarray` — `(W,)` bool, True for every VALID window within `±round(w_seconds/window_s)` grid steps of a detected-state CHANGE found in the VALID subsequence (A1.8: filter invalid/-1 FIRST, find changes in the valid subsequence, map the new-state onset back to its grid index; a valid→invalid→valid blip is NOT a change). Invalid windows always False. `W` default at the call site = `snapshot.min_dwell_s`.
- `_state_name_for(state_id: int, state_names: dict[int, str] | None) -> str` — `"invalid"` for `_INVALID_LABEL`; else `state_names[id]` when present, else `f"cluster-{id}"` (the A1.4 fallback / D2(a) `cluster-<id>` convention).
- `_apply_transition_suppression(alarm: np.ndarray, near_transition: np.ndarray, role: np.ndarray) -> int` — in place: force `alarm=False` on `near_transition & (role == ROLE_SCORED) & alarm`; return the count actually suppressed (`suppressed_by_transition`). `score`/`p_value`/`near_transition`/`role` are untouched (full audit trail).
- `_alarms_frame(prepared, labels, verdicts, near_transition, state_names)` — appends `near_transition[idx]` (bool) and per-row `_state_name_for(int(labels[i]), state_names)`; returns `frame[list(_ALARM_COLUMNS)]`.
- `_state_segments(labels, grid, state_names)` — when `state_names` is not None, add a `mapped_mode` column (`_state_name_for` per `cluster_id`); ABSENT otherwise (D2(c) conditional — keeps the line-238 default-fixture assert green).
- `_timeline_markdown` / `_notes_markdown` — both ALREADY receive `snapshot`, so they read `snapshot.state_names` directly (NO new `state_names` param): name states when present (`state {id} ({name})`), bare id otherwise. `_notes_markdown` gains ONE new param `suppressed_by_transition: int | None` and reports the count when suppression ran.
- `build_parser`: `--suppress-transition-alarms` (`store_true`, default OFF).

- [ ] RED — first EDIT the local `_ALARM_COLUMNS` list in `tests/test_monitor_cli.py` (line 58) so 198/949/1092 track the new contract (A1.3):
```python
_ALARM_COLUMNS = [
    "window", "t_utc_ns", "state", "score", "p_value", "alarm", "low_confidence", "role",
    "threshold_source", "near_transition", "state_name",
]
```
- [ ] RED add new tests to `tests/test_monitor_cli.py`:
```python
def test_near_transition_mask_marks_boundary_windows_valid_subseq() -> None:
    import monitor
    # labels: run of 0s, an invalid gap (-1), run of 1s. The 0->1 change is at the
    # FIRST valid 1 (index 6); invalid window 3 is NOT a change (A1.8).
    labels = np.array([0, 0, 0, -1, 1, 1, 1, 1], dtype=np.int64)
    valid = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    mask = monitor._near_transition_mask(labels, valid, window_ns=1_000_000_000, w_seconds=1.0)
    assert mask[3] == False  # invalid window never flagged   # noqa: E712
    assert mask[4] == True   # first valid 1 (boundary onset) # noqa: E712
    assert mask[2] == True   # last valid 0, within 1 window of the boundary onset
    assert mask[7] == False  # 3 valid windows past the boundary -> outside +-1


def test_apply_transition_suppression_forces_false_and_counts() -> None:
    import monitor
    alarm = np.array([True, True, False, True], dtype=bool)
    near = np.array([True, False, True, True], dtype=bool)
    role = np.array(["scored", "scored", "scored", "consumed_for_calibration"], dtype=object)
    n = monitor._apply_transition_suppression(alarm, near, role)
    assert n == 1                                        # only window 0: near & scored & was-True
    assert alarm.tolist() == [False, True, False, True]  # window 3 (consumed) untouched


def test_state_name_column_always_present_fallback_cluster_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monitor
    # snapshot WITHOUT state_names -> every alarm row's state_name is cluster-<id> (A1.4).
    snapshot_path, _snapshot = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out"
    assert monitor.main(
        ["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN, "--out", str(out_dir)]
    ) == 0
    alarms = pd.read_parquet(out_dir / "alarms.parquet")
    assert list(alarms.columns) == _ALARM_COLUMNS  # near_transition + state_name appended
    assert alarms["state_name"].str.startswith("cluster-").all()
    assert alarms["near_transition"].dtype == bool


def test_named_snapshot_surfaces_state_name_and_mapped_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monitor
    from dataclasses import replace
    _path, snapshot = _make_snapshot(tmp_path)
    labels = sorted(snapshot.thresholds)  # both fixture labels survive (module docstring)
    named = replace(snapshot, state_names={labels[0]: "turbine", labels[1]: "pump"})
    named_path = tmp_path / "named.npz"
    save_snapshot(named_path, named)  # save_snapshot does not re-validate state_names
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out"
    assert monitor.main(
        ["--snapshot", str(named_path), "--run", _MONITOR_RUN, "--out", str(out_dir)]
    ) == 0
    alarms = pd.read_parquet(out_dir / "alarms.parquet")
    assert set(alarms["state_name"].unique()) <= {"turbine", "pump"}
    segments = pd.read_csv(out_dir / "segments.csv")
    assert "mapped_mode" in segments.columns  # D2(c): present because state_names present
    timeline = (out_dir / "timeline.md").read_text()
    assert "(turbine)" in timeline or "(pump)" in timeline


def test_suppress_transition_alarms_invariant_and_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monitor
    snapshot_path, _snapshot = _make_snapshot(tmp_path)
    mon_prepared = _two_state_prepared(_MON_T0_NS, seed=1)
    _install_common_monkeypatches(monkeypatch, monitor, tmp_path / "results", mon_prepared)
    out_dir = tmp_path / "out"
    assert monitor.main(
        ["--snapshot", str(snapshot_path), "--run", _MONITOR_RUN, "--out", str(out_dir),
         "--suppress-transition-alarms"]
    ) == 0
    alarms = pd.read_parquet(out_dir / "alarms.parquet")
    scored = alarms[alarms["role"] == "scored"]
    # invariant: no scored near_transition window remains an alarm; audit columns retained.
    assert not bool((scored["alarm"] & scored["near_transition"]).any())
    assert scored["score"].notna().any()
    assert "suppressed_by_transition" in (out_dir / "monitor_notes.md").read_text()
```
  (`_make_snapshot`/`_install_common_monkeypatches`/`_two_state_prepared`/`_MON_T0_NS`/`_MONITOR_RUN` are the file's existing helpers; `save_snapshot`/`replace`/`np` are already imported. The pure `_apply_transition_suppression` test is the non-vacuous suppression check; the CLI test is the integration invariant.)
- [ ] Run RED with temp-file+&&; expect failures (columns/helpers/flag absent).
- [ ] GREEN `scripts/monitor.py` (real sketches):
```python
_ALARM_COLUMNS: tuple[str, ...] = (
    "window", "t_utc_ns", "state", "score", "p_value", "alarm", "low_confidence", "role",
    "threshold_source", "near_transition", "state_name",
)  # A1.4: near_transition then state_name appended at the END, both ALWAYS present.


def _near_transition_mask(
    labels: np.ndarray, valid_mask: np.ndarray, window_ns: int, w_seconds: float
) -> np.ndarray:
    out = np.zeros(labels.shape[0], dtype=bool)
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size < 2:
        return out
    valid_labels = labels[valid_idx]
    change_pos = np.flatnonzero(valid_labels[1:] != valid_labels[:-1]) + 1  # onset in valid subseq
    if change_pos.size == 0:
        return out
    boundary_grid = valid_idx[change_pos]  # grid index of each new-state onset (A1.8 map-back)
    w_windows = int(round(w_seconds / (window_ns / 1e9)))
    for b in boundary_grid:
        out[valid_idx[np.abs(valid_idx - b) <= w_windows]] = True  # only VALID windows flagged
    return out


def _state_name_for(state_id: int, state_names: dict[int, str] | None) -> str:
    if state_id == _INVALID_LABEL:
        return "invalid"
    if state_names is not None and state_id in state_names:
        return state_names[state_id]
    return f"cluster-{state_id}"


def _apply_transition_suppression(
    alarm: np.ndarray, near_transition: np.ndarray, role: np.ndarray
) -> int:
    target = near_transition & (role == ROLE_SCORED) & alarm
    n = int(target.sum())
    alarm[target] = False  # in-place (the _mark_unknown_states idiom); role/score untouched
    return n
```
  - `_alarms_frame(prepared, labels, verdicts, near_transition, state_names)`: add `"near_transition": near_transition[idx]` and `"state_name": np.array([_state_name_for(int(labels[i]), state_names) for i in idx], dtype=object)` into the frame dict, keep the `return frame[list(_ALARM_COLUMNS)]`.
  - `_state_segments(labels, grid, state_names)`: after the rename, `if state_names is not None: seg["mapped_mode"] = [_state_name_for(int(c), state_names) for c in seg["cluster_id"]]`.
  - `_timeline_markdown` (reads `snapshot.state_names` internally): when present render `desc = f"state {cluster_id} ({_state_name_for(cluster_id, snapshot.state_names)})"` in the known-state branch.
  - `_notes_markdown` (reads `snapshot.state_names` internally): gains ONE param `suppressed_by_transition: int | None`; render the per-state row's `state` cell as `{row.state} ({name})` when named; append a `- suppressed_by_transition: N (…)` line under Window accounting when suppression ran.
  - `main`: after `_assert_roles_complete`, `near_transition = _near_transition_mask(labels, prepared.valid_mask, prepared.grid.window_ns, snapshot.min_dwell_s)`; `n_suppressed = _apply_transition_suppression(verdicts.alarm, near_transition, verdicts.role) if args.suppress_transition_alarms else None`; pass `near_transition` + `snapshot.state_names` to `_alarms_frame`; `snapshot.state_names` to `_state_segments`; `n_suppressed` to `_notes_markdown` (`_timeline_markdown`/`_notes_markdown` already receive `snapshot`). Add the `--suppress-transition-alarms` flag to `build_parser`.
- [ ] Run GREEN gates; expect pass + `Success: no issues`.
- [ ] Commit `feat: monitor named states (state_name/mapped_mode) + near_transition + suppression ablation (P9 D2/D3c/A1.4)`.

---

### Task 4: `analyze_days transitions` taxonomy + `scripts/sweep_min_dwell.py` (D3a + D3b, A1.8)

**Files:** Modify `scripts/analyze_days.py` (one subcommand + a seam + pure helpers) · Create `scripts/sweep_min_dwell.py` · Tests extend `tests/test_analyze_days.py`, create `tests/test_sweep_min_dwell.py`

**Interfaces (D3a `transitions`):**
- Seam `_run_states_and_power(run_name, cfg, index) -> tuple[np.ndarray, np.ndarray, float]` — full-length `(gt_states, power, window_s)` via `_run_scada_or_none` + `gt_labels`'s `state` column + `scada["power"]`; monkeypatched in tests (mirrors `_run_features_and_gt`). Raises `ValueError` for a run with no SCADA (the taxonomy needs GT).
- Pure helpers (unit-tested):
  - `_transition_segments(states: np.ndarray) -> list[tuple[str | None, str | None, int, int]]` — for each contiguous `"transition"` run `[start, stop)`, its `(from_state, to_state)` bracketing KNOWN states (`states[start-1]` / `states[stop]` when in `set(STATES) - {"transition"}`, else `None`).
  - `_transition_class(from_state, to_state) -> str` — `f"{from_state}->{to_state}"` when BOTH bracket, else `"unbracketed"` (A1.8 explicit category).
  - `_transition_taxonomy(states, power, window_s) -> pd.DataFrame` — one row per CLASS: `n_segments`, dwell stats (`dwell_s` mean/median/min/max from `(stop-start)*window_s`), ramp stats (`median_abs_dpdt` via the centred `dP/dt` over each run — the `_apply_ramp` formula).
- `transitions` subcommand: `--runs` (GT-bearing days incl. `080726`'s changeover), `--out`; writes `results/analysis-days/transitions/<...>.{png,csv}` + a digest paragraph. Docstring + digest carry the our-own-analysis attribution; NO partner numeric as an expected value.

**Interfaces (D3b `scripts/sweep_min_dwell.py`):**
- Consumes: `rowii.anomaly.pools.build_pool`, `FittedDetector.fit_pooled`, `rowii.eval.metrics.evaluate` (`.state_ari`, the SAME majority-mapped metric as the P7 k-selection), `rowii.state.segments.duration_filter`, `dataclasses.replace` on `Config`/`DetectConfig`. Duplicates run_step2's `_load_run_scada`/`_gt_state_labels`-style GT via a monkeypatch seam.
- Produces: CLI `sweep_min_dwell.py --fit-runs <csv> --test-run <name> --variant <v> [--min-dwells 5,10,20] [--k 4]`. Per `min_dwell_s ∈ {5,10,20}` s: refit `fit_pooled` on the pool (Config rebuilt via `replace`), `apply` to the held-out run, `evaluate(pred_valid, gt_valid, grid).state_ari` — a `state_ari` table across the six rotations, **detector arm only** (the bank is dwell-free except under `--smooth`; not swept). Plus one Step-2 chain FAR spot-check at the three values (one rotation, e.g. B1→`290626-tu` fusion recalibrate). Pure helper `_min_dwell_windows(min_dwell_s, window_s) -> int == max(1, round(min_dwell_s/window_s))` (the `_finish` formula, duplicated for the sweep's own reporting/testing). Output `results/step2/min-dwell-sweep/<test_run>/<variant>.{csv,json}`; NO `DetectConfig` default is changed unless the data argues for it (reported either way).

- [ ] RED extend `tests/test_analyze_days.py` (pure-helper math on synthetic `states`/`power`):
```python
def test_transition_segments_bracketing_and_unbracketed() -> None:
    import analyze_days as ad
    states = np.array(
        ["standstill", "transition", "transition", "turbine", "transition", "unknown"],
        dtype=object,
    )
    segs = ad._transition_segments(states)
    assert segs[0] == ("standstill", "turbine", 1, 3)   # bracketed
    assert segs[1] == ("turbine", None, 4, 5)           # to_state is unknown -> None
    assert ad._transition_class(*segs[0][:2]) == "standstill->turbine"
    assert ad._transition_class(*segs[1][:2]) == "unbracketed"


def test_transition_taxonomy_counts_dwell_and_ramp() -> None:
    import analyze_days as ad
    states = np.array(
        ["standstill", "transition", "transition", "turbine", "turbine",
         "transition", "standstill"], dtype=object,
    )
    power = np.array([0.0, 5.0, 20.0, 60.0, 60.0, 20.0, 0.0], dtype=np.float64)
    tax = ad._transition_taxonomy(states, power, window_s=1.0)
    row = tax.set_index("transition_class")
    assert int(row.loc["standstill->turbine", "n_segments"]) == 1
    assert float(row.loc["standstill->turbine", "dwell_s_median"]) == 2.0  # 2 windows @ 1s
    assert float(row.loc["standstill->turbine", "median_abs_dpdt"]) > 0.0
```
- [ ] RED create `tests/test_sweep_min_dwell.py` (window-count conversion + a monkeypatched CLI ARI table):
```python
"""Tests for scripts/sweep_min_dwell.py (Package-9 D3b): the min_dwell->window-count
conversion (5/10/20; duration_filter no-op at min_dwell<=1) + a monkeypatched
fit_pooled/GT-seam CLI writing a state_ari-by-min_dwell table -- no real data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def test_min_dwell_windows_conversion() -> None:
    import sweep_min_dwell as sd
    assert sd._min_dwell_windows(5.0, 1.0) == 5
    assert sd._min_dwell_windows(10.0, 1.0) == 10
    assert sd._min_dwell_windows(20.0, 1.0) == 20
    assert sd._min_dwell_windows(0.5, 1.0) == 1  # floored at 1 -> duration_filter no-op


def test_duration_filter_noop_at_min_dwell_one() -> None:
    from rowii.state.segments import duration_filter
    labels = np.array([0, 1, 0, 0, 0], dtype=np.int64)
    np.testing.assert_array_equal(duration_filter(labels, min_dwell=1), labels)
```
- [ ] Run RED with temp-file+&&; expect ImportError/AttributeError.
- [ ] GREEN `scripts/analyze_days.py` (real sketches; `transitions` subparser + `main` dispatch branch + digest attribution line):
```python
_KNOWN_GT_STATES = tuple(s for s in STATES if s != "transition")  # from rowii.scada.labels


def _transition_segments(states: np.ndarray) -> list[tuple[str | None, str | None, int, int]]:
    st = np.asarray(states, dtype=object)
    out: list[tuple[str | None, str | None, int, int]] = []
    for start, stop in _contiguous_true_runs(st == "transition"):  # rowii.scada.labels helper
        prev = st[start - 1] if start > 0 else None
        nxt = st[stop] if stop < st.shape[0] else None
        frm = str(prev) if prev in _KNOWN_GT_STATES else None
        to = str(nxt) if nxt in _KNOWN_GT_STATES else None
        out.append((frm, to, int(start), int(stop)))
    return out


def _transition_class(from_state: str | None, to_state: str | None) -> str:
    if from_state is None or to_state is None:
        return "unbracketed"
    return f"{from_state}->{to_state}"


def _run_abs_dpdt(power: np.ndarray, start: int, stop: int, window_s: float) -> float:
    p = np.asarray(power, dtype=np.float64)
    vals = [abs((p[i + 1] - p[i - 1]) / (2.0 * window_s))
            for i in range(max(1, start), min(p.shape[0] - 1, stop))
            if np.isfinite(p[i - 1]) and np.isfinite(p[i + 1])]
    return float(np.median(vals)) if vals else float("nan")
```
  - `_transition_taxonomy` groups `_transition_segments` by `_transition_class`, aggregating `n_segments`, dwell stats over `(stop-start)*window_s`, and `median_abs_dpdt` over per-segment `_run_abs_dpdt`. `_run_transitions(args)` calls `_run_states_and_power` per run, concatenates class stats across runs, writes PNG (bar of `n_segments` per class + a dwell/ramp CSV) and a digest line ("D3a SCADA transition taxonomy from OUR OWN gt_labels; analysis type our own, no partner constant"). Import `from rowii.scada.labels import STATES, _contiguous_true_runs`.
- [ ] GREEN `scripts/sweep_min_dwell.py` (real sketch):
```python
def _min_dwell_windows(min_dwell_s: float, window_s: float) -> int:
    return max(1, round(min_dwell_s / window_s))  # the FittedDetector._finish formula

def _sweep_state_ari(pool_fit_features, prepared_test, gt_test_valid, cfg, *, k, min_dwells):
    from dataclasses import replace
    out = {}
    for d in min_dwells:
        swept = replace(cfg, detect=replace(cfg.detect, min_dwell_s=float(d)))
        detector = FittedDetector.fit_pooled(pool_fit_features, swept, k=k)
        valid = prepared_test.valid_mask
        grid = WindowGrid(prepared_test.grid.t0_ns, prepared_test.grid.window_ns, int(valid.sum()))
        pred = detector.apply(prepared_test.features[valid], grid).frame_labels
        gt = pd.DataFrame({"state": gt_test_valid})
        out[d] = float(evaluate(pred, gt, grid).state_ari)
    return out
```
    The CLI wires `build_pool(prepared_fit, "fit", sweep_cfg)` → `pool_fit.features`, a monkeypatched `_run_gt_states`-style seam for `gt_test_valid`, and writes the `state_ari`-by-`min_dwell` table + the one recalibrate FAR spot-check (reusing `run_step2`'s FAR-table math via duplication, script-sibling rule). Attribution/`min_dwell` verdict recorded in the JSON.
- [ ] Run GREEN gates; expect pass + `Success: no issues`.
- [ ] Commit `feat: analyze_days transitions taxonomy (P9 D3a)` then `feat: sweep_min_dwell -- state_ari-by-min_dwell + FAR spot-check (P9 D3b)` (two test-then-feat pairs within the task).

---

### Task 5: `src/rowii/anomaly/sentinels.py` — two label-free drift sentinels (D1, A1.1)

**Files:** Create `src/rowii/anomaly/sentinels.py` · Test `tests/test_sentinels.py`

**Interfaces:**
- Consumes: `rowii.pipeline.stream_columns`, `rowii.anomaly.levelrecal.level_columns`.
- Produces:
  - `level_series(rows: np.ndarray, feature_names: list[str], streams: Sequence[str]) -> np.ndarray` — `(W,)` per-window mean of the intersection of `stream_columns(streams)` and `level_columns(feature_names)` (the `analyze_days._levels_by_stream` rule, in src); raises `ValueError` when that intersection is empty (an embedding variant, or the streams absent — s2 reads the RAW `audio`/`vibration` caches, never fusion's z-scored columns, A1.1).
  - **s1 (mode-bank rejection rate):** `s1_threshold(no_mode_fits: np.ndarray, segment_ids: np.ndarray, *, n_boot: int = 1000, seed: int = 7) -> float` — the **97.5th percentile of B=1000 `segment_ids`-block bootstrap resamples of `mean(no_mode_fits)`** on the B1 CONFORMAL side (A1.1, seeded `rng(seed)`); `s1_fires(day_rate: float, threshold: float) -> bool == day_rate > threshold`. The bank's `low_confidence_modes` are surfaced ALONGSIDE the rate by the driver (T6), because a low-confidence member makes `no_mode_fits` UNDER-fire (`ModeBank.assign` caveat).
  - **s2 (per-stream level-step):** `s2_anchor_mad(level_values: np.ndarray, segment_ids: np.ndarray) -> tuple[float, float]` — `anchor = median(level_values)`; `mad = 1.4826 * median(|m - median(m)|)` over the per-`segment_ids`-block medians `m` (A1.1: MAD over the B1 CONFORMAL-side per-block medians of the mic-level columns; the 1.4826 consistency scaling follows the repo's own `SessionNormalizer` precedent in `rowii.anomaly.normalize`, so `3·mad` reads as the standard 3-sigma-equivalent robust criterion — orchestrator decision 2026-07-22, resolving the RAW-vs-scaled flag). `s2_fires(day_median: float, anchor: float, mad: float, *, k: float = 3.0) -> bool == abs(day_median - anchor) > k * mad`. `s2_attribution(mic_fires: bool, vib_fires: bool) -> str` — `"instrumentation"` when `mic_fires and not vib_fires` (P8's mic-steps-vib-flat signature, A1.8: **RAWGeneratorVib__2 only** for the vib cross-check), else `"machine"`; the cross-check labels the CAUSE, never vetoes (overall s2 fire == `mic_fires`).
  - Internal: `_block_medians(values, segment_ids) -> np.ndarray`; `_bootstrap_rate_pct(values, segment_ids, pct, n_boot, seed) -> float`.
- Binding: firewall-safe — every threshold is B1-derived; `97.5`/`1000`/`3` are named standard-statistics constants, never partner figures. The firing decision is in the stored log10 level domain (A1.1); the dB conversion is a DRIVER reporting nicety (T6, duplicating `_level_db_factor`), never the decision.

- [ ] RED `tests/test_sentinels.py`:
```python
"""Tests for rowii.anomaly.sentinels (Package-9 D1, A1.1): the level-series
stream∩level extraction (raises for embeddings), the seeded segment-block s1
bootstrap threshold + firing, and the s2 anchor/MAD math + mic-out/vib-in
attribution. Deterministic, no real data, no partner number as an expected value."""
from __future__ import annotations

import numpy as np
import pytest

from rowii.anomaly.sentinels import (
    level_series, s1_fires, s1_threshold, s2_anchor_mad, s2_attribution, s2_fires,
)

_MIC = ("RAWGeneratorMic__0", "RAWTurbineMic__1")


def _names() -> list[str]:
    return [
        "RAWGeneratorMic__0::ch0_log_rms", "RAWGeneratorMic__0::ch0_spectral_centroid",
        "RAWTurbineMic__1::ch0_octave_125", "RAWTurbineMic__1::ch0_rolloff95",
    ]


def test_level_series_averages_stream_level_columns() -> None:
    names = _names()
    rows = np.zeros((5, 4), dtype=np.float64)
    rows[:, 0] = -40.0  # mic0 log_rms (level)
    rows[:, 2] = -30.0  # mic1 octave (level); cols 1,3 are shape -> ignored
    series = level_series(rows, names, _MIC)
    np.testing.assert_allclose(series, np.full(5, -35.0))  # mean of the two level cols


def test_level_series_raises_for_embedding_names() -> None:
    embed = [f"RAWGeneratorMic__0::beats_{i}" for i in range(8)]
    with pytest.raises(ValueError, match="level"):
        level_series(np.zeros((3, 8)), embed, _MIC)


def test_s1_threshold_is_deterministic_and_fires_above() -> None:
    rng = np.random.default_rng(0)
    seg = np.repeat(np.arange(20), 10)
    no_fit = (rng.random(200) < 0.05)  # ~5% baseline rejection
    thr = s1_threshold(no_fit, seg, n_boot=1000, seed=7)
    assert thr == s1_threshold(no_fit, seg, n_boot=1000, seed=7)  # seeded -> identical
    assert 0.0 <= thr <= 1.0
    assert s1_fires(0.5, thr) is True     # a 50% day-rate is well above the ~5% band
    assert s1_fires(float(no_fit.mean()), thr) is False  # the baseline itself does not fire


def test_s2_anchor_mad_and_fire() -> None:
    seg = np.repeat(np.arange(6), 10)
    level = np.concatenate([np.full(10, -40.0 + 0.01 * b) for b in range(6)])
    anchor, mad = s2_anchor_mad(level, seg)
    assert anchor == pytest.approx(np.median(level), abs=1e-6)
    assert mad >= 0.0
    assert s2_fires(anchor + 10.0, anchor, mad, k=3.0) is True    # a big step fires
    assert s2_fires(anchor + 0.001, anchor, mad, k=3.0) is False  # within the band


def test_s2_attribution_mic_out_vib_in_is_instrumentation() -> None:
    assert s2_attribution(mic_fires=True, vib_fires=False) == "instrumentation"
    assert s2_attribution(mic_fires=True, vib_fires=True) == "machine"
```
- [ ] Run RED with temp-file+&&; expect ImportError.
- [ ] GREEN `src/rowii/anomaly/sentinels.py` (module docstring carries the D1 attribution + the A1.1 firewall/RAW-MAD note; real sketches):
```python
"""Two label-free drift sentinels for the once-calibrated D1 replay (Package-9,
spec §3.D1 + A1.1). Both fire on a monitored day using thresholds derived ONLY
from the commissioning (B1) CONFORMAL side, so both are label-free at runtime.
s1 reuses the P8 mode bank's `no_mode_fits` rate; s2 is a per-stream level-step
on the RAW mic caches (fusion's z-scored columns are excluded upstream, A1.1).
The sentinel idea echoes the partner's drift monitoring (Rodrigues & Zhang,
2026); every number here is computed from OUR caches -- no partner constant is
read or asserted."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rowii.anomaly.levelrecal import level_columns
from rowii.pipeline import stream_columns


def level_series(rows: np.ndarray, feature_names: list[str], streams: Sequence[str]) -> np.ndarray:
    lvl = set(level_columns(feature_names))
    cols: list[int] = []
    for stream in streams:
        try:
            cols.extend(int(c) for c in stream_columns(feature_names, stream) if int(c) in lvl)
        except ValueError:
            continue
    if not cols:
        raise ValueError(
            "level_series: no stream∩level column (embedding variant, or streams "
            "absent) -- s2 must read a RAW mic/vibration cache (A1.1)"
        )
    return np.asarray(rows, dtype=np.float64)[:, sorted(set(cols))].mean(axis=1)


def _block_medians(values: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    return np.array([float(np.median(v[segment_ids == s])) for s in np.unique(segment_ids)])


def _bootstrap_rate_pct(values, segment_ids, pct, n_boot, seed) -> float:
    rng = np.random.default_rng(seed)
    groups = [np.asarray(values)[segment_ids == s] for s in np.unique(segment_ids)]
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        boots[b] = float(np.mean(np.concatenate([groups[i] for i in pick])))
    return float(np.percentile(boots, pct))


def s1_threshold(no_mode_fits, segment_ids, *, n_boot: int = 1000, seed: int = 7) -> float:
    return _bootstrap_rate_pct(np.asarray(no_mode_fits, dtype=np.float64),
                               np.asarray(segment_ids), 97.5, n_boot, seed)


def s1_fires(day_rate: float, threshold: float) -> bool:
    return bool(day_rate > threshold)


def s2_anchor_mad(level_values: np.ndarray, segment_ids: np.ndarray) -> tuple[float, float]:
    v = np.asarray(level_values, dtype=np.float64)
    block_med = _block_medians(v, np.asarray(segment_ids))
    mad = float(np.median(np.abs(block_med - np.median(block_med))))  # RAW MAD (A1.1)
    return float(np.median(v)), mad


def s2_fires(day_median: float, anchor: float, mad: float, *, k: float = 3.0) -> bool:
    return bool(abs(day_median - anchor) > k * mad)


def s2_attribution(mic_fires: bool, vib_fires: bool) -> str:
    return "instrumentation" if (mic_fires and not vib_fires) else "machine"
```
- [ ] Run GREEN gates; expect pass + `Success: no issues`.
- [ ] Commit `feat: drift sentinels -- s1 no_mode_fits bootstrap threshold + s2 level-step MAD (P9 D1/A1.1)`.

---

### Task 6: `scripts/run_once_calibrated.py` — the D1 replay driver (D1, A1.2/A1.6)

**Files:** Create `scripts/run_once_calibrated.py` · Test `tests/test_run_once_calibrated.py`

**Interfaces:**
- Consumes: `rowii.anomaly.sentinels` (T5), `rowii.state.modebank.ModeBank` (s1 canonical bank = **audio-beats**), `rowii.runtime.snapshot.load_snapshot`, `rowii.pipeline.prepare_run`, `rowii.io.dataset.discover`, `rowii.config.load_config`. Runs `monitor.py`/`eval_events.py` as SUBPROCESSES (script-sibling rule — never import a sibling script's internals); those calls sit behind monkeypatch seams so unit tests never launch a real monitor run (spec §5).
- Pinned constants (A1.2 — no run enumeration by day root anywhere):
  - `_B1_FIT_RUNS = ("010726-pu", "010726-tu1-morning", "010726-tu2", "010726-tu_ph_tu")` (era B, the P7 pool-B1).
  - `_REPLAY` chronological (A1.8: `270626` at its true position): `250526` (era A: `250526-tu`, `250526-pu-morning` — **`250526-pu-afternoon` EXCLUDED**, only its fusion cache exists), `270626` (era A, SENTINEL-ONLY: `270626-pu_ph_pu_ph_pu_ph-1`, no Betriebsdaten → no FAR/GT row), `290626` (era B, clean held-out: `290626-tu`, `290626-pu`), `010726` (era B, IN-SAMPLE: `010726-tu_ph_tu`, `010726-pu` — B1 members, tagged non-held-out, D1 honesty 3), `080726` (era C: `080726-pu_strikes`; `080726-st_strikes` for the event-retention check ONLY).
- Produces: CLI `run_once_calibrated.py --representation <fusion|vibration|audio-beats> --snapshot <B1_far_snapshot.npz> --bank-fit-runs <B1 csv> [--alpha 0.01] [--out DIR]` (one representation per invocation, the run_step2 one-arm rule). Per (day, representation) it:
  1. evaluates s1 (audio-beats bank `no_mode_fits` day-rate vs `s1_threshold`) and s2 (raw-`audio` mic `level_series` day-median vs the B1 `s2_anchor_mad` band; raw-`vibration` **RAWGeneratorVib__2** cross-check) — the SAME `s1 ∨ s2` verdict gates all three arms;
  2. runs `monitor.py --thresholds frozen` and `--thresholds recalibrate` into temp out-dirs, reads each realized FAR from `alarms.parquet` (`mean(alarm)` over `role == "scored"`); for `080726` the FAR is the event-free window-FAR (`--exclude-calibration-events docs/groundtruth/080726_events_{pu,st}.csv`, P7 pillar-3 rule);
  3. reports three regimes — **always-frozen**, **always-recalibrate**, **once+triggered** (frozen FAR if neither sentinel fired that day, else recalibrate FAR — **PER-DAY, NOT sticky**: D1 is deliberately day-granular, no persistent recalibrated-baseline state machine; spec §2 OUT-of-scope + D1 honesty 1);
  4. reports every arm's FAR on the **common recalibrate scoring-split window set** (A1.6 headline) with the full-population frozen FAR as a secondary labelled column;
  5. writes a **trigger log** (per day: s1 rate vs threshold + `low_confidence_modes`; s2 mic/vib medians vs band + `s2_attribution`; the frozen/recalibrate decision) answering explicitly whether the 2026-06-29 boundary is caught (a sentinel firing on the era-A days monitored with the era-B snapshot) and whether era-C `080726` triggers; and a `080726` pillar-3 TPR-RETAINED readout under once+triggered (recalibrate → strikes remain detectable) via `eval_events` at α=0.01.
- Pure helpers (unit-tested — the load-bearing logic): `_read_realized_far(alarms_path) -> float`; `_scoring_windows(alarms_path) -> np.ndarray`; `_far_on_windows(alarms_path, window_set) -> float` (A1.6 common-population); `_trigger_verdict(s1_fired, s2_fired) -> bool == s1 or s2`; `_regime_far(frozen_far, recal_far, triggered) -> float == recal_far if triggered else frozen_far`; `_trigger_log_row(day, era, tags, s1_rate, s1_threshold, low_confidence_modes, s2_mic, s2_vib, anchor, mad, attribution, decision) -> dict`.

- [ ] RED `tests/test_run_once_calibrated.py` (pure helpers on synthetic per-day FAR/verdict inputs + a monkeypatched-seam CLI; no real monitor run):
```python
"""Tests for scripts/run_once_calibrated.py (Package-9 D1, A1.2/A1.6): pure regime-
selection + trigger-log + FAR-reading helpers on synthetic parquet/verdict inputs,
and the 270626 sentinel-only path -- monitor/eval_events subprocesses are behind
monkeypatched seams, so no real monitor run happens in a unit test (spec §5)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


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
    # A1.6: subset the frozen arm onto the recalibrate scoring split {1, 3}.
    assert roc._far_on_windows(path, np.array([1, 3])) == pytest.approx(1.0)


def test_regime_far_and_trigger_verdict() -> None:
    import run_once_calibrated as roc
    assert roc._trigger_verdict(s1_fired=False, s2_fired=False) is False
    assert roc._trigger_verdict(s1_fired=True, s2_fired=False) is True
    assert roc._regime_far(0.9, 0.05, triggered=True) == 0.05   # fired -> recalibrate
    assert roc._regime_far(0.9, 0.05, triggered=False) == 0.9   # quiet -> frozen (NOT sticky)


def test_sentinel_only_day_has_no_far_row(tmp_path: Path, monkeypatch) -> None:
    import run_once_calibrated as roc
    row = roc._trigger_log_row(
        day="270626", era="A", tags=("sentinel-only",),
        s1_rate=0.4, s1_threshold=0.1, low_confidence_modes=(),
        s2_mic=-30.0, s2_vib=-50.0, anchor=-40.0, mad=0.5,
        attribution="instrumentation", decision="recalibrate",
    )
    assert row["day"] == "270626" and "sentinel-only" in row["tags"]
    assert "far" not in row  # sentinel-only: no FAR/GT (A1.2)
```
- [ ] Run RED with temp-file+&&; expect ImportError/AttributeError.
- [ ] GREEN `scripts/run_once_calibrated.py` (module docstring: retrospective/day-granular/NO online claim, "once" scoped to per-era, the `010726` in-sample + `080726` event-free-FAR tags, the D1 attribution). Real sketches:
```python
def _read_realized_far(alarms_path: Path) -> float:
    df = pd.read_parquet(alarms_path)
    scored = df[df["role"] == "scored"]
    return float(scored["alarm"].mean()) if len(scored) else float("nan")

def _scoring_windows(alarms_path: Path) -> np.ndarray:
    df = pd.read_parquet(alarms_path)
    return df.loc[df["role"] == "scored", "window"].to_numpy()

def _far_on_windows(alarms_path: Path, window_set: np.ndarray) -> float:
    df = pd.read_parquet(alarms_path)
    sub = df[(df["role"] == "scored") & df["window"].isin(set(int(w) for w in window_set))]
    return float(sub["alarm"].mean()) if len(sub) else float("nan")

def _trigger_verdict(*, s1_fired: bool, s2_fired: bool) -> bool:
    return bool(s1_fired or s2_fired)

def _regime_far(frozen_far: float, recal_far: float, *, triggered: bool) -> float:
    return recal_far if triggered else frozen_far  # per-day, NOT sticky (spec D1)
```
  - `_run_monitor(snapshot, run, mode, out_dir, *, event_free=None) -> Path` subprocess-invokes `monitor.py` (`--thresholds <mode>`, `--exclude-calibration-events` on `080726`), returns the `alarms.parquet` path; monkeypatched in tests. The main loop iterates `_REPLAY`, evaluates s1/s2 via `rowii.anomaly.sentinels` (bank fit once on B1 via `ModeBank.fit`; s1 threshold from the B1 conformal side; s2 anchor/MAD from the B1 conformal mic `level_series`), assembles the three-regime table (A1.6 common-population primary), the trigger log, and the `080726` pillar-3 retention (subprocess `eval_events.py` at α=0.01, with/without once+triggered recalibration); writes a JSON sidecar with all threshold derivations + provenance.
- [ ] Run GREEN gates; expect pass + `Success: no issues`.
- [ ] Commit `feat: run_once_calibrated -- B1-frozen replay, s1∨s2 trigger, 3-regime FAR + pillar-3 retention (P9 D1/A1.2/A1.6)`.

---

### Task 7: Execution + synthesis (orchestrator-executed, NO code)

> Marked orchestrator-executed: no source changes. All commands run from `repos/rowii-monitor/` with the project `.venv`. Every synthesis number MUST be artifact-verified; negative results reported plainly. NEVER touch the real data root beyond the read-only cache-warming / prepare_run paths the scripts already use.

- [ ] **Warm caches (BLOCKING).** Ensure `audio`, `vibration`, `audio-beats`, `fusion` caches exist for every pinned run (A1.2): `250526-tu`, `250526-pu-morning`, `270626-pu_ph_pu_ph_pu_ph-1`, `290626-tu`, `290626-pu`, `010726-tu_ph_tu`, `010726-pu`, the four B1 fit runs (`010726-pu`, `010726-tu1-morning`, `010726-tu2`, `010726-tu_ph_tu`), `080726-pu_strikes`, `080726-st_strikes`. Warm any missing (`scripts/warm_cache.py --variants audio vibration audio-beats fusion --runs …`); confirm each `results/cache/<run>--<variant>.npz` before proceeding.
- [ ] **Build the three B1 snapshots (A1.7 -- required `--test-run 290626-tu`):** for `variant ∈ {fusion, vibration, audio-beats}`:
  - `.venv/bin/python scripts/run_step2.py --protocol cross-day-pooled --fit-runs 010726-pu,010726-tu1-morning,010726-tu2,010726-tu_ph_tu --test-run 290626-tu --variant <variant> --scorer knn --alpha 0.01 --save-snapshot results/step2/snapshots/b1-<variant>.npz` → verify the saved snapshot's `state_names` is populated (GT present on the B1 010726 runs).
- [ ] **D1 replay** (per representation): `.venv/bin/python scripts/run_once_calibrated.py --representation <variant> --snapshot results/step2/snapshots/b1-<variant>.npz --bank-fit-runs 010726-pu,010726-tu1-morning,010726-tu2,010726-tu_ph_tu --alpha 0.01` → the three-regime FAR table (A1.6 common-population primary), the trigger log (does a sentinel fire on the era-A days monitored with the era-B snapshot? does era-C `080726` trigger?), and the `080726` pillar-3 TPR-retained readout. Record whether the 2026-06-29 boundary is CAUGHT (the D1 success criterion).
- [ ] **D3 drivers:**
  - `.venv/bin/python scripts/analyze_days.py transitions --runs 250526-tu,290626-tu,010726-tu_ph_tu,080726-pu_strikes` → the SCADA transition taxonomy (per-class count/dwell/ramp, incl. `unbracketed`).
  - `.venv/bin/python scripts/sweep_min_dwell.py --fit-runs <B1> --test-run 290626-tu --variant fusion --k 4` (× the six P7/P8 rotations) → `state_ari` at `min_dwell ∈ {5,10,20}` s + the one recalibrate FAR spot-check; record the `min_dwell` verdict (change `DetectConfig.min_dwell_s` default ONLY if the data argues for it, reported either way).
- [ ] **D2/D3c monitor spot-checks:** run `monitor.py` on `290626-tu` with `b1-fusion.npz` and confirm `alarms.parquet` carries `near_transition` + `state_name` (named), `segments.csv` carries `mapped_mode`, `timeline.md` shows `state N (name)`; re-run with `--suppress-transition-alarms` and confirm the `suppressed_by_transition` count + the audit-trail columns.
- [ ] **Synthesis:** README package-9 section (once-calibrated FAR-regime table + trigger log + the caught-boundary statement; named-state example; transition taxonomy + `min_dwell` verdict; suppression ablation delta on the `080726` pump/standstill strikes) — all numbers artifact-verified, negatives plain; master-thesis research note (figures inline, in `master-thesis/research/notes/`); memory update. Every partner-inspired analysis carries its attribution; no partner number appears as an expected value.
- [ ] **Final whole-branch review (opus)** — named focuses: `derive_state_names` mask/fallbacks (A1.5); snapshot `state_names` round-trip + fitted-id validation + NO mutual-exclusivity (A1.5); the `_ALARM_COLUMNS` exact-equality updates + `state_name` ALWAYS present + `mapped_mode` conditional (A1.3/A1.4); `near_transition` valid-subsequence map-back (A1.8); the sentinel thresholds fit-day-ONLY + firewall (A1.1); the D1 day-granular (NOT sticky) regime + A1.6 common-population + `080726` event-free FAR + pillar-3 retention; the transition taxonomy `unbracketed` category (A1.8). → fix loop → **PR #12** → merge.

---

## Self-review (at write time)

**Spec coverage.** D1 → T5 (sentinels s1/s2 + vib cross-check) + T6 (replay driver: three regimes, trigger log, A1.6 common-population, `080726` event-free FAR + pillar-3 retention) + T7 (snapshot builds, replay runs, synthesis). D2 → T1 (`derive_state_names`) + T2 (snapshot `state_names` member + run_step2 wiring) + T3 (monitor `state_name`/`mapped_mode`/named timeline+notes) + T7 (spot-check). D3 → T4 (D3a `transitions` taxonomy + D3b `sweep_min_dwell`) + T3 (D3c `near_transition` + `--suppress-transition-alarms`) + T7 (drivers). Amendments: A1.1 → T5 pinned thresholds (97.5-pct bootstrap, 3·MAD, firewall-safe); A1.2 → T6 pinned run set (`250526-pu-afternoon` excluded, `270626` sentinel-only at true position); A1.3 → T3 `_ALARM_COLUMNS` exact-equality test updates; A1.4 → T3 `state_name` ALWAYS present + END order (`near_transition`, then `state_name`); A1.5 → T1 wrapper (mask both, three fallbacks, fill over fitted_ids) + T2 keys⊆fitted-ids + NO mutual-exclusivity; A1.6 → T6 common recalibrate-scoring-split FAR primary; A1.7 → T7 snapshot builds carry `--test-run 290626-tu`; A1.8 → T2 GT-skip `None` seam, T4 `unbracketed` category, T3 `near_transition` invalid-first map-back, T5 vib `RAWGeneratorVib__2`-only, T4 swept-dwell transplantation, T6 `270626` true position.

**Type/name consistency.** `derive_state_names(gt, pred, fitted_ids) -> dict[int, str]` (T1) is consumed by run_step2's save-snapshot block (T2) and validated key-by-key against `detector.smoother._fitted_ids` in `fit_snapshot_from_parts` (T2). `MonitorSnapshot.state_names: dict[int,str] | None` (T2) is read by monitor's `_state_name_for`/`_state_segments`/`_timeline_markdown`/`_notes_markdown`/`_alarms_frame` (T3) and by the D1 driver's named-snapshot load (T6). `_pool_row_gt_labels` (T2) mirrors `run_modebank._pool_gt_labels` exactly (object dtype). `rowii.anomaly.sentinels`' `s1_threshold`/`s1_fires`/`s2_anchor_mad`/`s2_fires`/`s2_attribution`/`level_series` (T5) are consumed identically by the driver (T6). `_min_dwell_windows == max(1, round(min_dwell_s/window_s))` (T4) matches `FittedDetector._finish` (verified). No placeholders, no TBD — every module/CLI has a real code sketch and complete RED tests.

**Resolved ambiguities (flagged for the orchestrator).**
1. `derive_state_names` home is `rowii.eval.metrics` (not a new module): it wraps `_majority_mapping`, lives beside it, adds only `from collections.abc import Iterable` — the correct home per the code.
2. T3 merges spec §6 steps 2 (D2 surfacing) and 4 (D3c) into ONE monitor task because both edit `_ALARM_COLUMNS` + `_alarms_frame` + the exact-equality tests; splitting would touch them twice.
3. `mapped_mode` on `segments.csv` is CONDITIONAL on `snapshot.state_names` (D2(c)), so the line-238 default-fixture assert stays green; `state_name` on `alarms.parquet` is UNCONDITIONAL (A1.4, `cluster-<id>` fallback). Verified against the exact asserts.
4. D1 once+triggered is PER-DAY, NOT sticky — `_regime_far(frozen, recal, triggered)` picks per day from that day's `s1 ∨ s2` verdict; the persistent recalibrated-baseline state machine is explicitly OUT of scope (spec §2, D1 honesty 1). Pinned in `_regime_far` + its test.
5. s2 firing is the A1.1 log10-domain aggregate `3·MAD` rule using **RAW MAD** (`median(|m - median(m)|)`) — the literal `3·MAD` reading; the 1.4826-scaled variant is the alternative — flagged for the reviewer. The `_level_db_factor` dB conversion is a driver-side reporting nicety (T6), never the firing criterion, and the sentinel src module stays dB-free (no cross-script import).
6. D3b is a NEW `scripts/sweep_min_dwell.py` (spec §3.D3(b) names it) and D3a is an `analyze_days` subcommand — grouped under T4 with two test-then-feat commit pairs.
7. The D1 driver invokes `monitor.py`/`eval_events.py` as SUBPROCESSES (script-sibling rule); their FAR/TPR reads sit behind monkeypatch seams, so no real monitor run happens in a unit test (spec §5).
