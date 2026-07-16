# Step-2 Package 6: Runtime Prototype + Pillar-3 Readiness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persistable monitor snapshot (detector + per-state scorer + conformal thresholds) with a `monitor` CLI, a prepared-only event-level evaluation harness, and the detection-performance scarcity curve on the MIMII proxy — the LAST package of the thesis code roadmap.

**Architecture:** New `src/rowii/runtime/` (snapshot), new `src/rowii/eval/events.py`, a labeled MIMII iterator beside the package-4 unlabeled one in `src/rowii/tfc/corpora.py`, three new CLIs (`monitor.py`, `eval_events.py`, `scarcity_detection.py`). Everything reuses the existing fit/apply/score/conformal machinery; NO new modeling.

**Tech Stack:** numpy (`np.savez`, `allow_pickle=False`), hmmlearn GaussianHMM reconstruction, pandas, sklearn.metrics (AUC/pAUC), matplotlib (figure), existing featurizer classes.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-16-step2-package6-runtime-pillar3-design.md` **including Amendment A1** (binding: covars-diag storage, k<=1 degenerate case, monitor per-state semantics, clip-level metric coherence, standardization + width caveats, split parity).
- NO pickle anywhere in the snapshot path (`allow_pickle=False` on every load). Runtime scorer whitelist: `{"knn", "mahalanobis"}` — `save_snapshot` refuses others with a clear message.
- Scripts must not import sibling scripts' internals — duplicate small helpers with the established rationale comment (warm_cache.py convention).
- Torch import discipline: lazy imports outside dedicated model modules.
- Gates after every task: `.venv/bin/python -m pytest tests/ -q -m "not data"`, `.venv/bin/ruff check .`, `.venv/bin/mypy src scripts` (strict, must stay at "no issues").
- Commit style: `test:` then `feat:` (TDD pairs) or combined `feat:` with tests, matching repo history. No Co-Authored-By lines.
- Every user-facing output (notes/README rows) restates the honesty framings from spec §4.

---

### Task 1: `MonitorSnapshot` — dataclass, save/load, detector reconstruction, `fit_snapshot`

**Files:**
- Create: `src/rowii/runtime/__init__.py` (empty docstring module)
- Create: `src/rowii/runtime/snapshot.py`
- Test: `tests/test_runtime_snapshot.py`

**Interfaces:**
- Consumes: `FittedDetector`/`DetectionResult` (`rowii.state.detect`), `StickyHmmSmoother` (`rowii.state.smooth`), `split_by_segments`/`build_references` (`rowii.anomaly.references`), `calibrate`/`ConformalThreshold` (`rowii.anomaly.conformal`), `KnnScorer`/`MahalanobisScorer` (`rowii.anomaly.scorers`), `SweepConfig` (`rowii.anomaly.sweep`), `PreparedRun` (`rowii.pipeline`), `Config`/`DetectConfig` (`rowii.config`).
- Produces (used by Task 2):
  - `SNAPSHOT_FORMAT_VERSION: int = 1`
  - `@dataclass(frozen=True) class MonitorSnapshot` with fields:
    `mean, std: np.ndarray` · `fitted_ids: np.ndarray` · `hmm_startprob, hmm_transmat, hmm_means, hmm_covars_diag: np.ndarray | None` (all None iff k<=1; covars stored as `(k, F)` DIAGONALS per A1.1) · `min_dwell_s: float` · `k: int` · `self_transition: float` · `random_seed: int` · `references: dict[int, np.ndarray]` · `calibration_scores: dict[int, np.ndarray]` · `thresholds: dict[int, ConformalThreshold]` · `scorer: str` · `alpha: float` · `min_ref: int` · `calibration_frac: float` · `seed: int` · `variant: str` · `feature_names: list[str]` · `fit_run: str` · `checkpoints: dict[str, str]` · `created_at: str` · `format_version: int`
  - `fit_snapshot(prepared: PreparedRun, rowii_cfg: Config, sweep_cfg: SweepConfig, *, variant: str, fit_run: str, clusterer: str = "kmeans", k: int | None = None) -> tuple[MonitorSnapshot, np.ndarray]` — returns the snapshot and the FULL-length detected labels (`-1` on invalid windows).
  - `to_detector(snapshot: MonitorSnapshot) -> FittedDetector`
  - `scorer_for_label(snapshot: MonitorSnapshot, label: int) -> Scorer`
  - `save_snapshot(path: Path, snapshot: MonitorSnapshot) -> None` (npz + `<stem>.json` sidecar)
  - `load_snapshot(path: Path) -> MonitorSnapshot`

**Implementation core (binding):**

- npz layout: per-label arrays under `ref__<label>` / `cal__<label>`; detector arrays under their own keys; ONE `meta` member = `np.array([json.dumps({...})], dtype=str)` carrying every scalar/string/threshold field (thresholds as `{label: {threshold, alpha, n_calibration, achievable_alpha_floor, low_confidence}}`). Load: `np.load(path, allow_pickle=False)`; `format_version != SNAPSHOT_FORMAT_VERSION` → `ValueError` naming both versions.
- `fit_snapshot` split parity (A1.6, mirrors `run_sweep` code exactly): detector fit on VALID rows (mirror `scripts/apply_detector.py::_fit_detector_and_mapping`'s valid-grid pattern), labels scattered back (`-1` invalid); `top = split_by_segments(prepared.segment_ids, prepared.valid_mask, sweep_cfg.calibration_frac, sweep_cfg.seed)`; `nested = split_by_segments(prepared.segment_ids, calib_mask, 0.5, sweep_cfg.seed + 1)`; `references = build_references(features, labels, nested.calibration_windows, min_ref=sweep_cfg.min_ref)`; per label with conformal windows of that label: `scores = scorer.fit(ref).score(...)`, `calibrate(scores, sweep_cfg.alpha)`. Labels with no conformal windows are DROPPED from the snapshot with a logged warning (monitor cannot alarm without a threshold).
- HMM extraction: from `detector.smoother.last_model_` — store `startprob_`, `transmat_`, `means_`, `np.diagonal(covars_, axis1=1, axis2=2)` (A1.1); `fitted_ids` + `component_to_id` (store as the fitted_ids array itself — component i maps to `fitted_ids[i]`; VERIFY this invariant against `StickyHmmSmoother.fit_decode`'s `id_to_component` construction: `unique_ids` sorted by `np.unique` → component idx == position in `fitted_ids`. Assert it at extraction: `component_to_id == {i: fitted_ids[i]}` else raise).
- `to_detector`: rebuild `StickyHmmSmoother(self_transition, random_seed)`; if k>=2: `GaussianHMM(n_components=k, covariance_type="diag", params="mc", init_params="", random_state=random_seed)`, assign the four arrays (covars from stored diagonals), set `smoother.last_model_`, `smoother._fitted_ids`, `smoother._component_to_id = {i: int(fitted_ids[i])}`; else (k<=1): `last_model_ = None`, `_fitted_ids` set. Wrap in `FittedDetector(mean, std, smoother, min_dwell_s, k)`.
- `scorer_for_label`: `KnnScorer(k=1, metric="cosine").fit(references[label])` for "knn" (fit is deterministic normalization — spec D1); `MahalanobisScorer().fit(...)` for "mahalanobis"; anything else `ValueError`.
- `save_snapshot` refuses `snapshot.scorer not in ("knn", "mahalanobis")`.

- [ ] **Step 1 (RED):** `tests/test_runtime_snapshot.py` — build a synthetic two-state `PreparedRun` (reuse the pattern of `tests/test_apply_detector.py::_two_state_prepared`: 8 segments × 30 windows, two well-separated Gaussian blobs, `segment_ids`, all-valid mask). Tests:
  - `test_fit_snapshot_round_trip_apply_parity`: `fit_snapshot` → `save_snapshot` → `load_snapshot` → `to_detector(loaded).apply(features_valid, grid)` labels ARE `np.array_equal` to `to_detector(original).apply(...)` labels AND to the fit-time detector's own apply; per-label `scorer_for_label(loaded, l).score(X)` equals the original's bitwise.
  - `test_snapshot_npz_has_no_pickle`: `np.load(path, allow_pickle=False)` succeeds (implicitly) and `json.loads(meta)` carries `format_version == 1`, `scorer`, `variant`, `fit_run`, thresholds dict.
  - `test_version_mismatch_raises`: rewrite meta with `format_version: 99` → `load_snapshot` raises `ValueError` naming 99 and 1.
  - `test_geometry_fields_present`: `loaded.feature_names == prepared.feature_names`.
  - `test_save_refuses_non_runtime_scorer`: snapshot with `scorer="ocsvm"` → `ValueError` mentioning the whitelist.
  - `test_degenerate_single_state_round_trip` (A1.2): one-blob prepared (single label) → snapshot has `hmm_startprob is None`, round-trip `to_detector(...).apply` returns the single fitted id everywhere.
  - `test_split_parity_with_run_sweep` (A1.6): monkeypatch nothing — assert `fit_snapshot`'s per-label `n_calibration` values equal a hand-run of the same splits (`split_by_segments` twice + `calibrate`) on the same labels.
  - `test_threshold_round_trip_fields`: every `ConformalThreshold` field survives save/load exactly.
- [ ] **Step 2:** run new tests → FAIL (module missing).
- [ ] **Step 3 (GREEN):** implement `src/rowii/runtime/snapshot.py` per the binding core above.
- [ ] **Step 4:** full gates (pytest not-data, ruff, mypy) → green.
- [ ] **Step 5:** commit `feat: MonitorSnapshot — pickle-free runtime bundle (detector + per-state scorer + thresholds)`.

### Task 2: `scripts/monitor.py` — snapshot + new recording → timeline + alarms

**Files:**
- Create: `scripts/monitor.py`
- Test: `tests/test_monitor_cli.py`

**Interfaces:**
- Consumes: Task 1's `load_snapshot`, `to_detector`, `scorer_for_label`; `prepare_run`, `discover`, `load_config`; `p_values`, `calibrate` (`rowii.anomaly.conformal`); `split_by_segments`; `to_segments` (`rowii.state.segments`); `to_utc_ns` (`rowii.anomaly.overlap`).
- Produces: CLI `monitor.py --snapshot <path> --run <name> [--thresholds recalibrate|frozen] [--alpha F] [--no-cache] [--out DIR]` (default out `results/monitor/<run>/`); output files `segments.csv`, `timeline.md`, `alarms.parquet` (columns: `window, t_utc_ns, state, score, p_value, alarm, low_confidence, role`), `alarm_segments.csv`, `monitor_notes.md`. Exit 0 success, 2 usage/geometry errors.

**Binding semantics (spec D2 + A1.3):**
- Geometry guard AFTER `prepare_run`: `prepared.feature_names != snapshot.feature_names` → stderr message naming variant + both widths, exit 2.
- Detected labels: `to_detector(snapshot)`; apply on valid rows, scatter to full length with `-1` (mirror `run_step2._apply_detector_labels` inline with the sibling-script rationale comment).
- `frozen`: every valid window of a snapshot-known state gets `role="scored"`, alarm = `score > snapshot.thresholds[label].threshold`, `p_value = p_values(scores, snapshot.calibration_scores[label])`; states on the new run absent from the snapshot → counted in notes as `unknown_state_windows` (no verdict, `role="unknown_state"`, `alarm=False`, `p_value=NaN`). Notes carry the distribution-shift warning verbatim: frozen cross-day thresholds did NOT hold their FAR in package-2 evidence.
- `recalibrate` (DEFAULT): `top = split_by_segments(prepared.segment_ids, prepared.valid_mask, snapshot.calibration_frac, snapshot.seed)`; per snapshot-known label: calibration-side windows of that label → `calibrate(scores, alpha)` (CLI `--alpha` overrides, default snapshot.alpha); scoring-side windows of that label → verdicts (`role="scored"`); calibration-side windows → `role="consumed_for_calibration"`, `alarm=False`, `p_value=NaN`. A label with zero calibration-side windows → all its windows `role="no_conformal_data"`, no verdicts, notes row. `low_confidence` from the recalibrated `ConformalThreshold`.
- `alarm_segments.csv`: `to_segments` over the full-grid indicator `alarm.astype(int64)` (invalid/unscored = 0), keep only `cluster == 1` rows, rename to `start_utc,end_utc,duration_s`.
- `monitor_notes.md`: snapshot provenance block (fit_run, variant, scorer, alpha, created_at, checkpoints), mode, per-state table (n_scored, n_alarms, realized alarm rate, low_confidence), consumed/unknown counts, and the standing honesty framing (alarms = candidates, no fault labels).

- [ ] **Step 1 (RED):** `tests/test_monitor_cli.py` in the `tests/test_apply_detector.py` monkeypatch style: fake index with a fit run + a monitor run, `_two_state_prepared`-style PreparedRuns (different seeds), monkeypatch `monitor.discover`, `monitor.load_config`, `monitor.prepare_run`. Build a real snapshot via Task 1's `fit_snapshot` + `save_snapshot` into tmp_path. Tests:
  - `test_recalibrate_mode_end_to_end`: exit 0; alarms.parquet exists with ALL contracted columns; every scored row's state is a snapshot label; `role` values ⊆ {scored, consumed_for_calibration}; realized alarm rate over scored windows of each state ≤ ~3×alpha (sanity, same distribution); notes name mode "recalibrate".
  - `test_frozen_mode_flags_shift_warning`: `--thresholds frozen` → notes contain "did NOT hold" (the package-2 warning); all valid known-state windows have `role="scored"`.
  - `test_geometry_mismatch_exits_2`: snapshot fitted on 4-col features, monitored prepared has 6 cols → exit 2, stderr names both.
  - `test_unknown_state_windows_counted`: craft monitor-run labels so one state id is absent from the snapshot (fit run with 2 states, monitor prepared drawing a third blob far away → detector maps it to an existing id; instead force it: delete one label from the snapshot's references/thresholds dicts before save) → those windows get `role="unknown_state"`, notes count them.
  - `test_alarm_segments_schema`: alarm_segments.csv columns exactly `start_utc,end_utc,duration_s`.
  - `test_help_documents_every_flag`.
- [ ] **Step 2:** run → FAIL (script missing).
- [ ] **Step 3 (GREEN):** implement `scripts/monitor.py` (CLI skeleton: `warm_cache.py`; output patterns: `apply_detector.py`).
- [ ] **Step 4:** full gates green.
- [ ] **Step 5:** commit `feat: monitor CLI — snapshot + new recording -> state timeline + conformal alarms`.

### Task 3: Pillar-3 event-level harness — `rowii/eval/events.py` + `scripts/eval_events.py`

**Files:**
- Create: `src/rowii/eval/events.py`
- Create: `scripts/eval_events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: nothing new (pandas/numpy only).
- Produces:
  - `@dataclass(frozen=True) class EventEvalResult`: `per_event: pd.DataFrame` (columns `start_utc,end_utc,kind,detected,latency_s`), `n_events: int`, `n_detected: int`, `event_tpr: float` (NaN when `n_events == 0`), `false_alarm_windows: int`, `false_alarm_rate_per_hour: float`, `realized_window_far: float`, `tolerance_s: float`; method `to_frame() -> pd.DataFrame` (one summary row).
  - `evaluate_events(alarms: pd.DataFrame, events: pd.DataFrame, *, window_s: float, tolerance_s: float = 0.0) -> EventEvalResult`.
  - CLI: `eval_events.py --alarms <parquet> --events <csv> [--tolerance-s F] [--window-s F] --out DIR` → `event_eval.csv` (summary + per-event rows), `event_notes.md`.

**Binding semantics (spec D3):**
- `alarms` requires columns `t_utc_ns` (int64) + `alarm` (bool); rows with `role` present and != "scored" are dropped first (only scored windows count for FAR).
- `events` requires tz-AWARE ISO-8601 `start_utc,end_utc` (naive timestamps → `ValueError` telling the user to add an offset), optional `kind` (default "fault").
- Membership: window START `t` detected for event e ⇔ `start_ns - tol_ns <= t < end_ns + tol_ns` (inclusive start, exclusive end; window starts are the timestamps — spec D3).
- Latency: `(first alarming t inside e) - start_ns` in seconds (may be negative within the tolerance pad — kept, documented); NaN if missed.
- False alarms: alarm rows outside EVERY padded event; `realized_window_far = false_alarm_windows / n_non_event_scored_windows` (NaN if zero); rate/hour = `false_alarm_windows / (n_non_event_scored_windows * window_s / 3600)`.
- Edge battery: overlapping events (a window may detect BOTH), zero events (TPR NaN, all alarms false alarms), zero alarms (TPR 0.0 when events exist), event fully outside the alarms' time span (missed, latency NaN).

- [ ] **Step 1 (RED):** `tests/test_events.py` — synthetic frames (hand-built ns timestamps at window_s=1.0): one test per edge above plus `test_latency_first_alarm_only`, `test_boundary_inclusive_start_exclusive_end` (alarm exactly at `start` → detected; exactly at `end` → NOT), `test_naive_timestamps_raise`, plus a CLI smoke test writing parquet+csv into tmp_path and asserting `event_eval.csv` + notes exist with the summary row.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3 (GREEN):** implement module + CLI.
- [ ] **Step 4:** gates green.
- [ ] **Step 5:** commit `feat: pillar-3 event-level evaluation harness (per-event TPR, latency, FAR) — prepared-only`.

### Task 4: Labeled MIMII clip iterator

**Files:**
- Modify: `src/rowii/tfc/corpora.py` (append; do not touch the unlabeled iterator)
- Test: `tests/test_corpora_labeled.py`

**Interfaces:**
- Consumes: existing private helpers `_wav_to_mono_float`, `_cut_windows`, `_resample_windows`, `_standardize` (same module).
- Produces:
  - `@dataclass(frozen=True) class LabeledClip`: `windows: np.ndarray` ((W,S) float32, per-window standardized), `label: int` (0 normal / 1 abnormal), `machine_id: str`, `path: str`.
  - `iter_labeled_clips_wav_dir(root: Path, *, window_s: float = 1.0, target_hz: int = 16_000, limit_clips_per_class: int | None = None, machine_ids: Sequence[str] | None = None) -> Iterator[LabeledClip]`.

**Binding semantics (spec D4 + A1.5):** walk `root/**/id_*/{normal,abnormal}/*.wav` sorted (deterministic); label from the parent directory name; `machine_id` = the `id_*` directory name; per-(machine_id, class) cap `limit_clips_per_class` applied in sorted order with a `logger.info` of kept/total counts (no silent truncation); 16 kHz default target (A1.5: BEATs-native; the 8 kHz unlabeled default was a TF-C pretraining choice); windows standardized per window (P4 convention, documented in the docstring including the removed-level-cue caveat).

- [ ] **Step 1 (RED):** `tests/test_corpora_labeled.py` — build a tmp synthetic wav tree (reuse the wav-writing approach of the existing corpora tests — check `tests/test_corpora.py` for its `_write_wav` helper and mirror it): 2 machine ids × {normal: 3 clips, abnormal: 2 clips} of 2 s @ 16 kHz. Tests: both classes yielded with correct labels/ids/counts; windows shape (2, 16000) float32; per-window mean≈0/std≈1; determinism (two runs → identical order and bytes); `limit_clips_per_class=1` keeps exactly one per (id, class) and logs counts; `machine_ids=["id_00"]` filters.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3 (GREEN):** implement.
- [ ] **Step 4:** gates green.
- [ ] **Step 5:** commit `feat: labeled MIMII clip iterator (normal+abnormal, machine ids) for pillar-3 proxy work`.

### Task 5: `scripts/scarcity_detection.py` — detection-performance scarcity harness

**Files:**
- Create: `scripts/scarcity_detection.py`
- Test: `tests/test_scarcity_detection_cli.py`

**Interfaces:**
- Consumes: Task 4's iterator; featurizer classes DIRECTLY (`rowii.signals.features.AudioFeaturizer`, `rowii.signals.logmel.LogmelFeaturizer`, `rowii.signals.beats.BeatsFeaturizer`, `rowii.tfc.wrapper.TfcFeaturizer`, `rowii.adapt.student.StudentFeaturizer`) on `(n, S, 1)` windows @ 16 kHz; `KnnScorer`; `calibrate`; `sklearn.metrics.roc_auc_score`.
- Produces: CLI `scarcity_detection.py --root data/public/mimii/pump_0db [--representations csv] [--fractions csv] [--seeds csv] [--machine-ids csv] [--limit-clips-per-class N] [--alpha F] --out results/scarcity-detection` → `scarcity_detection.csv`, `scarcity_detection.md`, `scarcity_curve.png`, embedding cache under `<out>/cache/`.

**Binding protocol (spec D4 + A1.4):**
- Representations map to featurizers with the benchmark harness's skip-with-log semantics for missing checkpoint envs (`beats`/`tfc`/`student`); `handcrafted`/`logmel` always available.
- Per (representation × machine_id): extract features ONCE over all clips (normal capped by `--limit-clips-per-class`, abnormal capped likewise), cache npz `(features per clip stacked + clip index arrays)` keyed by a sha256 fingerprint of (representation, checkpoint paths, window_s, target_hz, caps, sorted clip relpaths+sizes); `allow_pickle=False`.
- Split protocol (clip-level, leakage rule): normal clips → 30% TEST (seeded ONCE with seed 7, shared by every fraction/seed cell) / 70% TRAIN-POOL. Per (fraction × seed): draw `ceil(fraction * n_pool)` pool clips (seeded), split the DRAW 80/20 into FIT clips / CAL clips (clip-level); kNN(k=1, cosine) fit on fit-clip windows; clip score = MEAN over the clip's window scores; conformal threshold = `calibrate(cal_clip_scores, alpha)` (CLIP-level per A1.4); metrics on test-normal + ALL abnormal clips: `auc_clip`, `pauc_clip = roc_auc_score(..., max_fpr=0.1)` (STANDARDIZED/McClish — named in CSV header comment + md), `tpr_at_alpha` (abnormal clips over threshold), `realized_normal_clip_far`, `auc_window` (secondary, window-level).
- CSV row per (representation, machine_id, fraction, seed) with all counts (`n_fit_clips, n_cal_clips, n_test_normal_clips, n_abnormal_clips`); degenerate draws (fit or cal empty at tiny fractions) → row with NaN metrics + `degenerate=True`, logged.
- Figure: matplotlib (`Agg`), 2 panels (`auc_clip`, `tpr_at_alpha`) vs fraction (log-x), one line per representation (mean over machine_ids × seeds), shaded min–max band over seeds; title names the corpus + caps.
- `--dry-run` prints the cell matrix + caps and exits 0 (warm_cache convention).

- [ ] **Step 1 (RED):** `tests/test_scarcity_detection_cli.py` — synthetic wav tree (Task 4's builder pattern; make abnormal clips LOUD-noise so handcrafted separates them), run `main(["--root", ..., "--representations", "handcrafted", "--fractions", "0.5,1.0", "--seeds", "7", "--limit-clips-per-class", "4", "--out", ...])`. Tests: exit 0; CSV schema exact; `n_fit_clips` monotone in fraction; `auc_clip` ∈ [0,1] and > 0.5 for the separable synthetic; determinism (second run → identical CSV bytes, cache HIT logged); figure + md exist; unknown representation → exit 2; missing checkpoint env for `beats` → skipped with log, CSV has no beats rows; `--dry-run` writes nothing.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3 (GREEN):** implement.
- [ ] **Step 4:** gates green.
- [ ] **Step 5:** commit `feat: detection-performance scarcity harness (MIMII proxy, clip-level protocol, conformal TPR)`.

### Task 6: Execution + synthesis (no new code; fixes only if execution finds defects)

- [ ] Fit + save snapshot: `010726-tu_ph_tu`, variant `fusion`, knn, per-state, alpha 0.05, detected labels → `models/adapted/monitor_010726_fusion.npz` (small fit CLI usage lives in `monitor.py`? NO — fitting stays a python snippet in the execution script calling `fit_snapshot`/`save_snapshot`, logged; the monitor CLI consumes only).
- [ ] Round-trip verification on the fit run (bitwise apply + score parity — rerun Task 1's parity path against the REAL artifact, via a logged python snippet).
- [ ] `monitor.py` on `250526-tu` + `290626-tu`, both `--thresholds` modes; cross-check recalibrate per-state alarm rates against package-2's cross-day recalibrated FAR tables (agreement expected; note any deviation honestly).
- [ ] `eval_events.py` HARNESS DEMO: synthetic `events.csv` over `290626-tu`'s real alarms.parquet — labeled a demo in every output.
- [ ] MIMII scarcity run: all representations available per checkpoint envs (beats/tfc/student set), machine ids all four, caps default → CSV + figure.
- [ ] README package-6 section + master-thesis research note (numbers verified against artifacts; honesty framings).
- [ ] Final whole-branch review (adversarial, named focuses: snapshot security/geometry, monitor semantics vs A1.3, scarcity protocol leakage, README number verification) → fix loop → gates.
- [ ] PR #9 → merge. Roadmap check-off in the ledger.

## Self-review (done at write time)

- Spec coverage: D1→T1, D2→T2, D3→T3, D4→T4+T5, D5→T6, A1 items bound inside T1/T2/T5 steps. ✓
- No placeholders; interfaces typed; names consistent across tasks (`MonitorSnapshot`, `fit_snapshot`, `to_detector`, `scorer_for_label`, `evaluate_events`, `iter_labeled_clips_wav_dir`, `LabeledClip`). ✓
- Type consistency: Task 2 consumes exactly Task 1's produced signatures; Task 5 consumes Task 4's `LabeledClip`. ✓
