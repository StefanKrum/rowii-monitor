# Step-2 Package 7: Robustness & Best System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Multi-day/multi-mode pools with honest held-out-day evaluation, session normalization, rolling thresholds, TF-C-PSHP continued pretraining, vib-corpus v2, pool-adapted encoders/snapshot — and the first REAL pillar-3 run on the 080726 induced-strike day.

**Architecture:** New `pools.py` + `normalize.py` in `rowii/anomaly/`, `FittedDetector.fit_pooled` in `rowii/state/detect.py`, `fit_snapshot_from_parts` in `rowii/runtime/snapshot.py`, protocol/CLI extensions in `run_step2.py`/`monitor.py`/`pretrain_tfc.py`/`download_corpora.py`/`adapt_beats.py`/`distill_beats.py`. Everything composes existing machinery (split_by_segments, build_references, calibrate, gt_labels).

**Tech Stack:** unchanged (numpy, sklearn KMeans, hmmlearn, pandas; torch lazily).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-18-step2-package7-robustness-design.md` **D1–D7 + Amendments A2, A3 (ALL 14 adopted resolutions), A4 (incl. A4.5)** — the amendments OVERRIDE the original D-sections where they conflict (esp. A3.1 pool-member evaluation ban, A3.4 fit_pooled, A3.2 rolling fallback, A3.7 conformal pool side, A3.11 APIs).
- Leakage rules: per-run top split (seed 7) everywhere; pool sides never touch any run's scoring windows; pool artifacts NEVER evaluated on pool-member runs (A3.1); 250526-* and 270626-* never pooled.
- Bruno firewall (A4.3): campaign ground truth = data with attribution; partner results never our findings.
- Gates per task: `.venv/bin/python -m pytest tests/ -q -m "not data"`, `.venv/bin/ruff check .`, `.venv/bin/mypy src scripts` ("Success: no issues").
- Commit style: single `feat:`/`test:` commits, NO Co-Authored-By. Scripts never import sibling scripts' internals (duplicate-with-rationale).
- Coverage tables (A4.1/A4.2): every pool and every evaluation reports windows per GT state × load bin; zero-coverage cells present in evaluation but absent in training log a WARNING.

---

### Task 1: `rowii/anomaly/pools.py` — multi-run pools + coverage tables

**Files:** Create `src/rowii/anomaly/pools.py` · Test `tests/test_pools.py`

**Interfaces:**
- Consumes: `split_by_segments` (references.py), `PreparedRun` (pipeline), `gt_labels`/`load_scada_window_means` (scada.labels, optional — coverage only), `SweepConfig`.
- Produces (used by T2/T3/T6/T8):
  - `@dataclass(frozen=True) class PoolMember`: `run_name: str`, `windows: np.ndarray` (indices into that run's grid), `n_windows: int`.
  - `@dataclass(frozen=True) class PoolResult`: `side: str` (`"calibration" | "fit" | "conformal"`), `members: list[PoolMember]`, `features: np.ndarray` (stacked rows, row order = members order), `run_index: np.ndarray` (per stacked row → member idx), `window_index: np.ndarray` (per stacked row → source window), `provenance: dict[str, dict[str, int]]` (run → {"n_windows": …}).
  - `build_pool(prepared: dict[str, PreparedRun], side: Literal["calibration","fit","conformal"], sweep_cfg: SweepConfig) -> PoolResult` — per run: top split (`calibration_frac`, `seed`); `"calibration"` = top calibration side; `"fit"`/`"conformal"` = nested split (0.5, seed+1) of the calibration side (run_sweep convention — A3.7 WARNING in the docstring: do NOT copy `_cross_day_per_state_sweep`'s top-split-as-fit semantics).
  - `coverage_table(prepared: dict[str, PreparedRun], windows_per_run: dict[str, np.ndarray], labels_per_run: dict[str, np.ndarray]) -> pd.DataFrame` — rows (run, label, n_windows); plus `coverage_warnings(train_table, eval_table) -> list[str]` (A4.1/A4.2 zero-coverage warnings; label here is ANY per-window label array — detected state or GT state or "state|load_bin" composite strings, caller's choice).
- Binding: labels-agnostic (pool machinery never derives labels itself); pooled `features` are float64 copies; empty side for a run → member with n_windows 0 + warning, never a crash.

- [x] RED tests: per-run split parity (hand-run `split_by_segments` per run, compare member windows exactly); side semantics (calibration ⊃ fit ∪ conformal, fit ∩ conformal = ∅ per run); stacked `features`/`run_index`/`window_index` alignment (probe row i maps back to `prepared[run].features[window]` bitwise); scoring windows of every run NEVER in any side (leakage probe); provenance counts; coverage_table schema + composite-label rows; coverage_warnings fires exactly on eval-not-train cells; empty-side warning.
- [x] GREEN implementation → gates → commit `feat: multi-run pools with leakage-safe sides and coverage tables (P7 D1/A3.7/A4.1-2)`.

### Task 2: `FittedDetector.fit_pooled` — pooled KMeans, per-run decode

**Files:** Modify `src/rowii/state/detect.py` (add classmethod; do not touch `fit`/`apply`) · Test `tests/test_detect_pooled.py`

**Interfaces:**
- Consumes: T1's `PoolResult` (fit side), `KMeansClusterer` (state.cluster), `StickyHmmSmoother` internals pattern (smooth.py: `_sticky_transmat`, `_init_means_covars` — import, don't reimplement), `Config.detect`.
- Produces (used by T3): `FittedDetector.fit_pooled(cls, pooled_features: np.ndarray, cfg: Config, *, k: int, clusterer: str = "kmeans") -> FittedDetector` — z-score stats from POOLED features; KMeans (seed = cfg.detect.random_seed) on pooled z-scored features → labels; emissions via `_init_means_covars` on pooled labels; GaussianHMM assembled EXACTLY like `StickyHmmSmoother.fit_decode`'s construction but **NO `model.fit` call** (A3.4: no cross-run EM chain); smoother fields set so the returned detector's `apply` does per-run Viterbi with the pooled emission model. Label ids = KMeans cluster ids 0..k-1.
- Binding: returned object is a normal `FittedDetector` (snapshot-serializable via the EXISTING extraction path — verify `_hmm_arrays`' component_to_id invariant holds: ids are 0..k-1 → identity ✓); docstring states the no-EM decision + that k is chosen by GT-ARI sweep at execution (A3.4).

- [x] RED tests: two synthetic runs with DISJOINT extra mode (run A: blobs 1+2; run B: blobs 2+3) → fit_pooled(k=3) recovers 3 clusters and `apply` on run B labels blob-3 windows with their own id (the pump-owns-a-cluster property); z-stats = pooled stats; NO EM (transmat bitwise == `_sticky_transmat(k, cfg)`); apply parity after snapshot save/load round trip (reuse `test_runtime_snapshot` helpers); degenerate k=1 path.
- [x] GREEN → gates → commit `feat: FittedDetector.fit_pooled — pooled KMeans emissions, per-run Viterbi (A3.4)`.

### Task 3: `--protocol cross-day-pooled` + `fit_snapshot_from_parts`

**Files:** Modify `scripts/run_step2.py` (new protocol + guards), `src/rowii/runtime/snapshot.py` (new constructor) · Tests `tests/test_step2_pooled_cli.py`, extend `tests/test_runtime_snapshot.py`

**Interfaces:**
- Consumes: T1 `build_pool`/`coverage_table`, T2 `fit_pooled`, existing `calibrate`/`p_values`/`build_references`/row builders.
- Produces:
  - `fit_snapshot_from_parts(detector: FittedDetector, references: dict[int, np.ndarray], calibration_scores: dict[int, np.ndarray], thresholds: dict[int, ConformalThreshold], *, scorer: str, alpha: float, min_ref: int, calibration_frac: float, seed: int, variant: str, fit_run: str, feature_names: list[str], checkpoints: dict[str, str]) -> MonitorSnapshot` (pure assembly + validation: key-set equality, scorer whitelist, geometry; `fit_run` carries a `"pool:"`-prefixed comma list; per-run provenance goes in the npz sidecar via a new optional `provenance` kwarg on `save_snapshot`).
  - CLI: `run_step2.py --protocol cross-day-pooled --fit-runs <csv> --test-run <name> [--k N|auto] [--alpha F] [--session-norm] …` → per-state + aggregate FAR tables for BOTH threshold modes (frozen thresholds from the pool's CONFORMAL side per A3.7; recalibrate from the test run's calibration side), coverage tables (detected-state AND GT-state × load-bin when Betriebsdaten exist), notes with baselines comparison hooks. Day-group disjointness guard (calendar day of first file; `parser.error` on overlap — A3.8). Detected labels on every run come from the ONE pooled detector.
- [x] RED: synthetic 3-run fixture (2 fit + 1 test, style-2 monkeypatch); tests: disjointness guard exit 2 (incl. sibling-run same-day case 010726-tu1 vs 010726-tu2 style names); frozen thresholds bitwise == calibrate(pool conformal scores, alpha); recalibrate == calibrate(test-run calibration side); pool-member-as-test refused (A3.1 guard!); coverage tables written; snapshot_from_parts round trip + refusal tests; α override.
- [x] GREEN → gates → commit `feat: cross-day-pooled protocol + fit_snapshot_from_parts (A3.1/A3.7/A3.8)`.

### Task 4: `rowii/anomaly/normalize.py` + `--session-norm` + snapshot v2

**Files:** Create `src/rowii/anomaly/normalize.py` · Modify `scripts/run_step2.py`, `scripts/monitor.py`, `src/rowii/runtime/snapshot.py` (format v2) · Tests `tests/test_normalize.py` + extensions

**Interfaces:**
- `@dataclass(frozen=True) class SessionStats`: `center: np.ndarray`, `scale: np.ndarray` (median / MAD·1.4826, floor 1e-8), `n_windows: int`, `norm_minutes: float`.
- `fit_session_stats(features: np.ndarray, valid_mask: np.ndarray, grid: WindowGrid, *, norm_minutes: float) -> SessionStats` (first-N-minutes valid rows); `apply_session_norm(features, stats) -> np.ndarray`.
- Wiring (A3.5 binding): detector ALWAYS consumes RAW features; normalization applies to the SCORING space only (references, calibration scores, scoring windows — all transformed with their OWN run's stats; fit-day references with the fit day's stats). Snapshot: `SNAPSHOT_FORMAT_VERSION = 2` when `session_stats` stored; `load_snapshot` accepts v1 (no stats) and v2; monitor `--session-norm` REFUSES a v1 snapshot (clear message) and fits the monitored run's stats from its first N minutes.
- [x] RED: median/MAD correctness + floor; first-N boundary (window starts, valid-only); detector-labels invariance under norm (bitwise); scoring-space application sites (probe: references transformed with fit stats, test windows with test stats — construct a shift that norm removes and assert FAR recovers on the synthetic fixture); v1-refusal + v2 round trip; state-mix confound documented in docstring (assert docstring mentions it — cheap doc pin).
- [x] GREEN → gates → commit `feat: label-free session normalization (median/MAD, scoring-space only) + snapshot v2 (D3/A3.5)`.

### Task 5: monitor `--thresholds rolling`

**Files:** Modify `scripts/monitor.py` · Test extend `tests/test_monitor_cli.py`

**Interfaces:** `--thresholds rolling --rolling-minutes M` (default 60): per (state, scored window): trailing same-state calibration-role windows within M minutes; threshold = `calibrate(those, alpha)` when count ≥ ceil(1/alpha)−1 else fit-day frozen threshold; new alarms.parquet column `threshold_source ∈ {rolling, fit_day_fallback}` (other modes: constant `mode` value); notes gain per-state trailing-coverage stats (fraction of scored windows with rolling coverage, per M) citing the A3.2 motivating numbers.
- [x] RED: synthetic run where early windows force fallback and later ones roll (assert per-window threshold_source pattern + bitwise thresholds both branches); coverage stats in notes; column contract update everywhere (events harness role-filter unaffected — assert eval_events still consumes the parquet).
- [x] GREEN → gates → commit `feat: rolling thresholds with conformal-floor fallback + coverage stats (D7/A3.2)`.

### Task 6: `pretrain_tfc.py --corpus pshp-pool --continue-from` + materialized pool windows

**Files:** Modify `scripts/pretrain_tfc.py` · Test extend `tests/test_pretrain_tfc.py`

**Interfaces:** new corpus key `pshp-pool`: windows via `iter_target_windows(run, cfg, target_hz=8000)` over `--pool-runs <csv>` (default = canonical pool), MATERIALIZED once to `<out>/pshp_pool_windows.npz` (members: `windows` float32 (N,8000), `run_names`, `per_run_counts`; `allow_pickle=False`; reused on rerun via sha256 of run names + per-run window counts + target_hz); `--continue-from PATH` initializes the model from that checkpoint's state dict; checkpoint + sidecar gain `continued_from: str | None` and `pool_runs`; `_CHECKPOINT_NAMES["pshp-pool"] = "tfc_audio_pshp.pt"`, scratch control via `--out-name tfc_audio_pshp_scratch.pt` override flag.
- [x] RED: corpus-key wiring (KeyError paths gone); materialize-once (second call: no iter_target_windows invocation — monkeypatch-count); `--continue-from` lineage in sidecar + init actually loaded (probe: weights equal source before training step); out-name override.
- [x] GREEN → gates → commit `feat: TF-C continued pretraining on PSHP pool (materialized windows, lineage) (D4/A3.9)`.

### Task 7: Paderborn K003–K006 + `tfc_vib_v2`

**Files:** Modify `scripts/download_corpora.py` · Test extend `tests/test_download_corpora.py`

**Interfaces:** K003–K006 entries with `groups.uni-paderborn.de/kat/BearingDataCenter/K00x.rar` URLs + `_SHA256_TBD` sentinel (live-HEAD verification at execution per house procedure — A3.10; NOT Zenodo); extraction reuses the existing rar path; `pretrain_tfc --corpus bearings` output name becomes overridable (`--out-name`, default unchanged `tfc_vib.pt`; execution uses `tfc_vib_v2.pt`).
- [x] RED: manifest entries present with correct host; dry-run lists new files; out-name override test (shared with T6's flag — implement once in T6, this task only VERIFIES it applies to bearings).
- [x] GREEN → gates → commit `feat: Paderborn K003-K006 downloads + vib checkpoint naming (D5/A3.10)`.

### Task 8: `adapt_beats`/`distill_beats` `--runs` round-robin

**Files:** Modify `scripts/adapt_beats.py`, `scripts/distill_beats.py`, `src/rowii/adapt/target_windows.py` (multi-run iterator helper) · Tests extend `tests/test_adapt_beats.py`, `tests/test_student.py`, `tests/test_target_windows.py`

**Interfaces:** `iter_target_windows_multi(runs: list[Run], cfg: Config, *, max_windows: int, target_hz: int, seed: int) -> Iterator[np.ndarray]` — ROUND-ROBIN across per-run iterators (A3.11: budget is TOTAL; a sequential chain would train almost entirely on run 1), per-run splits unchanged; exhausted runs drop out of the rotation. CLIs accept `--runs a,b,c` (singular `--run` stays, mutually exclusive); sidecars record per-run window counts. distill_beats pools calibration windows across runs for BOTH student inputs and teacher targets (per-run caches loaded, per-run splits, stacked; grid-alignment check per run).
- [x] RED: round-robin order pinned on unequal-length fake iterators (a,b,c,a,c,a…); total budget honored; per-run counts in sidecar; mutual-exclusion parser error; distill multi-run stacking parity vs single-run path on a 1-run list (bitwise).
- [x] GREEN → gates → commit `feat: multi-run round-robin adaptation/distillation pools (D6/A3.11)`.

### Task 9: Execution A — baselines, rotations, norm ablation (spec §8.2–8.4)

- [x] Monitor baselines: P6 snapshot on 290626-pu + 250526-pu-morning (both modes).
- [x] k-selection: fit_pooled k ∈ {4,5,6} on canonical pool → GT-ARI per pool day → pick + record.
- [x] Rotations (both modes × α grid {0.01,0.05,0.10}): fit {010726-*} → test 290626-{tu,pu}; fit {290626-*} → test 010726-{tu_ph_tu,pu}; cross-config probes → 250526-tu, 250526-pu-morning; coverage tables archived.
- [x] Session-norm ablation on the same rotations (N ∈ {5,20,60}) + monitor `--session-norm` on the cross-config days.
- [x] H2/H3 verdict vs single-day baselines; ledger.

### Task 10: Execution B — TF-C-PSHP, vib v2, pool adaptation, rolling, pillar-3 (spec §8.5–8.8)

- [x] Materialize pool windows → `tfc_audio_pshp.pt` + scratch control → Step-1 ARI (all GT days) + rotation rows (checkpoint-swap archival procedure).
- [x] Downloads K003–K006 (live sha256 transcription) → `tfc_vib_v2.pt` (window count vs 808) → vibration evidence side by side.
- [x] Pool-adapted LoRA + student (`--runs` canonical pool) → held-out evaluations only (A3.1) → pool snapshot via from_parts → monitor held-out days incl. pump.
- [x] Rolling ablation M ∈ {30,60,120} on monitor runs; coverage stats.
- [x] **Pillar-3 on 080726**: warm caches done → pooled snapshot (pool WITHOUT 080726) → monitor 080726-pu_strikes + st_strikes (recalibrate primary; frozen as cross-era row; session-norm arm) → `eval_events.py` vs `docs/groundtruth/080726_events_{st,pu}.csv` (α grid; per-class breakdown plate/vane/landmark) → per-event TPR, latency, FAR. Bruno-firewall framing in every output.
- [x] Ledger + hypothesis verdicts H4/H5/H6.

### Task 11: Synthesis + audit + mockup + PR (spec §8.9, A2.4, A2.5)

- [x] README package-7 section + master-thesis research note (universality axis per A2.1 in every table; all numbers artifact-verified).
- [x] Design-vs-implementation audit note (A2.4): blueprint/design claims vs code, item-by-item table.
- [x] Demo mockup (A2.5): static visual of the live pipeline view built around 080726 (state timeline, per-state score trace, alarm markers, candidate report) — for Stefan's review.
- [x] Final whole-branch review (opus; named focuses: pools leakage, fit_pooled correctness, A3.1 guard, rolling fallback, pillar-3 number verification) → fix loop → PR #10 → merge.

## Self-review (at write time)

Spec coverage: D1→T1, A3.4→T2, D2/A3.1/A3.7/A3.8→T3, D3/A3.5→T4, D7/A3.2→T5, D4/A3.9→T6, D5/A3.10→T7, D6/A3.11→T8, §8→T9-T11, A2.1/A2.2 grids→T9/T10 params, A2.3→T10 pillar-3, A2.4/A2.5→T11, A4.1/A4.2→T1 coverage + T3 wiring, A4.3→T10/T11 framing, A4.5 estimator-vs-final→T3 protocol + T11 narrative. Interfaces type-consistent (PoolResult in T1 consumed by T2/T3/T6/T8 as written; SessionStats only in T4; threshold_source only in T5). No placeholders.

## Plan amendment (2026-07-21, Stefan): representation axis pinned for execution

Stefan: "ich muss in der Thesis ja alle vergleichen". The comparison axis is
explicit, not implied:
- **T9 rotations** run per representation over at least {fusion, audio-beats,
  audio-tfc} (all off warm caches; audio-student added where the pool-student
  checkpoint from T10 exists at execution time).
- **T10 pillar-3 (080726)** runs the event evaluation per representation over
  {handcrafted audio, fusion, audio-beats, audio-tfc, audio-student} × α grid —
  the same five-way comparison shape as the P6 scarcity curve, now at event
  level on real induced events.
- The final "best system" recommendation in T11's synthesis is REQUIRED to be
  stated as the outcome of these comparisons ("so funktioniert es am besten,
  weil …"), never as an assumption; frozen BEATs is the universal BASELINE row
  in every table (A2.1), not a pre-decided winner.
