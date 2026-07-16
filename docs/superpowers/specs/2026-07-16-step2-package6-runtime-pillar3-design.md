# Step-2 Package 6: Runtime Prototype + Pillar-3 Readiness — Design Spec

**Date:** 2026-07-16 · **Branch:** `feat/step2-package6-runtime` · **Base:** main after PR #8
**Roadmap:** `master-thesis/research/notes/roadmap_2026-07-15_code_completion.md` (package 6, the LAST package)

## 1. Problem

Three roadmap items close the thesis code base:

1. **Runtime prototype** — the design chapter's "runs at the plant" requirement made
   concrete: persist a fitted detector + per-state scorer + conformal thresholds as ONE
   artifact, and a `monitor` CLI that takes that artifact plus a NEW recording and emits
   a state timeline + alarms. Today `FittedDetector` is explicitly "not serialized —
   future runtime-prototype serialization point" (`src/rowii/state/detect.py:61`), and
   the per-state `{label -> (fitted scorer, ConformalThreshold, calibration scores)}`
   bundle exists only inline in `scripts/run_step2.py::_cross_day_per_state_sweep`.
2. **Pillar-3 event-level evaluation harness** (PREPARED-ONLY) — labeled fault
   intervals → per-event TPR at fixed FAR + detection latency. No fault labels exist
   yet (induced-fault campaign pending); the harness must be ready the day data lands.
3. **Detection-performance scarcity curve** — the design's central figure (performance
   vs fraction of target-normal training data, per representation) needs LABELED
   anomalies, so it runs NOW on a public proxy (MIMII pump, which ships `abnormal/`
   clips the package-4 loader deliberately excludes) and reruns on PSHP when the
   campaign delivers.

## 2. Scope / non-goals

In scope: `src/rowii/runtime/` (snapshot save/load/fit), `scripts/monitor.py`,
`src/rowii/eval/events.py` + `scripts/eval_events.py`, a labeled MIMII iterator beside
the package-4 unlabeled one, `scripts/scarcity_detection.py`, execution + synthesis.

Non-goals: any new modeling; PSHP fault labels (campaign); dashboards/streaming
ingestion (batch CLI over recorded files is the deployment model the design commits
to); serializing sweep-only scorers (see D2); retraining/adaptation at monitor time.

## 3. Design decisions

### D1 — `MonitorSnapshot`: one artifact, numpy-only, `allow_pickle=False`

New module `src/rowii/runtime/snapshot.py`. A frozen dataclass bundling everything
`monitor` needs:

- **Detector half** (from `FittedDetector` + `StickyHmmSmoother`): `mean`, `std`
  (F,), HMM params as PLAIN ARRAYS (`startprob`, `transmat`, `means`, `covars`),
  `fitted_ids`, `component_to_id` (as two aligned int arrays), `min_dwell_s`, `k`.
  The sklearn clusterer is NOT stored — `FittedDetector.apply` never calls it
  (Viterbi labels both paths; documented behavior).
- **Scoring half**, per state label: reference matrix (Ni,F), calibration-score array
  (the piece no existing dataclass captures — required for `p_values` at monitor
  time), and the `ConformalThreshold` scalar fields. Plus `alpha`, `min_ref`,
  `scorer name`, `conditioning`.
- **Provenance / geometry guard**: `variant`, `feature_names` (validated against the
  new recording's prepared features at load-apply time — a snapshot fitted on
  `fusion` must refuse `audio-beats` features loudly), fit `run` name, `seed`,
  `calibration_frac`, relevant checkpoint paths as strings, `created_at`,
  `snapshot_format_version = 1`.

Persistence: `np.savez` + JSON-in-array pattern (pipeline cache convention,
`allow_pickle=False` on load) — per-label arrays under prefixed keys
(`ref__<label>`, `cal__<label>`), scalars/strings in a single JSON metadata string
array member. NO pickle anywhere: sklearn-backed scorers (OC-SVM/IF/LOF) and torch
AEs are SWEEP-ONLY comparison poles; the runtime scorer set is `{knn, mahalanobis}`
whose fitted state is pure numpy (`KnnScorer._reference` L2-normalized cosine /
`MahalanobisScorer._mean`+`_var_shrunk`). `save_snapshot` refuses other scorers with
a clear message; the docstring states why (no pickle = no arbitrary-code-on-load,
consistent with every other artifact in this repo). kNN is stored as the RAW
reference matrix + k + metric and refit at load (`KnnScorer.fit` is
deterministic normalization, not training).

`fit_snapshot(prepared, labels, rowii_cfg, sweep_cfg, detector) -> MonitorSnapshot`
mirrors `run_sweep`'s within-day assembly exactly: top split (seed, calibration_frac)
on the fit run, references from the fit side of the calibration half, thresholds from
the conformal side — the SAME leakage discipline as every sweep. HMM extraction:
rebuild `GaussianHMM` at load with `params=""`/no fit call, assigning the stored
arrays (mirrors the hmm-construction pattern in `StickyHmmSmoother`); round-trip
equality is test-pinned (identical Viterbi labels before/after save/load).

### D2 — `scripts/monitor.py`: snapshot + new recording → timeline + alarms

CLI (template: `warm_cache.py` skeleton + `apply_detector.py` output patterns):

```
monitor.py --snapshot <path> --run <name> [--thresholds recalibrate|frozen]
           [--alpha 0.05] [--no-cache] [--out results/monitor/<run>/]
```

Pipeline: `prepare_run` (cache honored unless `--no-cache`) → geometry guard against
snapshot `feature_names` → `FittedDetector.apply` (reconstructed from snapshot) →
per-state scoring:

- `--thresholds recalibrate` (DEFAULT): split the NEW run's windows by segments
  (snapshot seed), recalibrate per-state thresholds on the calibration side, alarm on
  the scoring side — package-2's central cross-day finding operationalized ("transfer
  detector + references, recalibrate thresholds per day" is the only recipe whose FAR
  held). References stay the SNAPSHOT's (fit-day) references; only thresholds are
  recalibrated.
- `--thresholds frozen`: apply fit-day thresholds unchanged to ALL new windows —
  reported with an explicit distribution-shift warning in the notes (package-2
  evidence: cross-day FAR does NOT hold frozen).

Outputs under `--out`: `segments.csv` + `timeline.md` (state half, apply_detector
conventions incl. scatter-back over `valid_mask`), `alarms.parquet` (window,
`t_utc_ns`, state, score, p_value, alarm, low_confidence), `alarm_segments.csv`
(maximal alarm runs via `to_segments` on the alarm indicator), `monitor_notes.md`
(snapshot provenance, mode, per-state counts/realized alarm rates, and the standing
honesty framing: NO fault labels exist — alarms are candidates for review, not
detections; states below `min_ref` fit windows are low-confidence-gated like every
sweep). Per-state gating semantics reuse the sweep row builders' conventions.

### D3 — Pillar-3 event-level harness (prepared-only): `src/rowii/eval/events.py`

Input contract (the campaign-data interface, documented in the module docstring):
a fault-events table `events.csv` with `start_utc,end_utc,kind` (ISO-8601 UTC), and
an alarms table (the `monitor`/sweep `alarms.parquet` schema).

`evaluate_events(alarms, events, *, grid | t_utc_ns, tolerance_s=0.0, far_alpha)` →
`EventEvalResult` (frozen dataclass + `to_frame()`):

- **Per-event detection**: event detected ⇔ ≥1 alarm window inside
  `[start - tolerance, end + tolerance]`; event-level TPR = detected/total.
- **Detection latency**: first alarm timestamp − event start (s), per event; median +
  per-event column (NaN for missed).
- **False alarms**: alarm windows OUTSIDE every (tolerance-padded) event, as count,
  per-hour rate over the covered non-event duration, and realized window FAR — the
  "at fixed FAR" side of the roadmap's "per-event TPR at fixed FAR".
- Edge semantics pinned by tests: overlapping events, events fully outside the
  recording, zero alarms, zero events (vacuous TPR = NaN not 1.0), alarms exactly on
  boundaries (inclusive start, exclusive end — timestamps are window STARTS).

`scripts/eval_events.py`: thin CLI — `--alarms <parquet> --events <csv>
[--tolerance-s N] --out <dir>` → `event_eval.csv` + `event_notes.md`. Campaign day =
`monitor.py` then `eval_events.py`, nothing else. Unit-tested on synthetic intervals
only (prepared-only, per roadmap: 🅿).

### D4 — Labeled MIMII iterator + detection-scarcity harness

**Loader** (beside the package-4 unlabeled one in `src/rowii/tfc/corpora.py`):
`iter_labeled_clips_wav_dir(root, *, window_s=1.0, target_hz=16_000, limit_clips_per_class=None)`
yielding per CLIP: `(windows (W,S) float32, label ∈ {0 normal, 1 abnormal},
machine_id str, clip_path str)` — reusing `_wav_to_mono_float`/`_cut_windows`/
`_resample_windows`/`_standardize` verbatim; `exclude_substring` logic replaced by
explicit `{normal,abnormal}` directory walking (both classes, labels from the path).
Target 16 kHz (BEATs-native; every featurizer resamples from `rate_hz` anyway —
documented; the 8 kHz default of the unlabeled iterator was a TF-C pretraining
choice, not a corpus property).

**Harness** `scripts/scarcity_detection.py` — the design's central figure on the
proxy:

- Representations: `handcrafted | logmel | beats | tfc | student` via the DIRECT
  featurizer classes (`AudioFeaturizer`, `LogmelFeaturizer`, `BeatsFeaturizer`,
  `TfcFeaturizer`, `StudentFeaturizer`) on `(n, S, 1)` windows at 16 kHz — checkpoint
  envs gate availability with the benchmark harness's skip-with-log semantics.
- Protocol per (representation × machine id × fraction × seed): clip-level
  train/test split of NORMAL clips (never window-level — leakage rule; windows of
  one clip stay together), fractions `{0.05, 0.1, 0.25, 0.5, 1.0}` of training-normal
  clips (seeded draw), kNN (k=1 cosine) fit on training-normal windows; score
  held-out normal + ALL abnormal windows; metrics: AUC, pAUC@FPR≤0.1, and
  TPR@(conformal threshold at α=0.05 calibrated on a held-out slice of
  training-normal) — clip-level scores = mean over the clip's windows (MIMII's
  standard clip-level evaluation), window-level AUC also reported.
- Embedding cache per (representation × machine id) as an npz under
  `results/scarcity-detection/cache/` (fingerprint: corpus manifest sha + featurizer
  checkpoint paths + window params) so fractions × seeds reuse ONE extraction pass —
  BEATs on ~13k windows is the long pole, extract once.
- Outputs: `scarcity_detection.csv` (all cells), `scarcity_detection.md`, and the
  figure `scarcity_curve.png` (matplotlib, mean ± min/max over seeds per fraction,
  one panel per metric, lines per representation).
- Caps documented + logged (no silent truncation): `--limit-clips-per-class`
  default bounds extraction (normal clips per id) with the exact counts printed.

### D5 — Execution (evidence, real artifacts)

1. Fit snapshot on `010726-tu_ph_tu` (variant `fusion`, knn, per-state, α=0.05,
   detected labels) → `models/adapted/monitor_010726_fusion.npz`.
2. Round-trip verification (bitwise label + score equality pre/post save-load) on the
   fit run, then `monitor.py` on `250526-tu` and `290626-tu` in BOTH threshold modes;
   cross-check recalibrate-mode per-state alarm rates against the package-2 cross-day
   recalibrated FAR tables (must agree — same machinery, now through the snapshot).
3. `eval_events.py` demonstration on synthetic intervals over a real day's alarms —
   labeled clearly as a HARNESS DEMO (no real fault labels), verifying the campaign-day
   workflow end to end.
4. MIMII proxy scarcity run (all available representations; beats/tfc/student per
   checkpoint availability) → curve + verdict on "does SSL help MORE under scarcity"
   ON THE PROXY (framed as proxy evidence, machine-id domain, not PSHP).
5. README package-6 section + master-thesis research note + final whole-branch
   review → PR #9 → merge. Roadmap check-off: every ⬜ code item closed.

## 3b. Amendment A1 (pre-implementation adversarial review, 2026-07-16)

Verified against the real code/library before any implementation; these bind:

1. **hmmlearn covars trap (D1, measured):** for `covariance_type="diag"` the
   `covars_` GETTER returns full `(k, F, F)` matrices while the SETTER demands
   `(k, F)` diagonals (probed: direct set fails with "'diag' covars must have
   shape (n_components, n_dim)"). The snapshot stores
   `np.diagonal(covars_, axis1=1, axis2=2)`; reconstruction assigns that. Viterbi
   parity of the reconstructed model was verified empirically (identical
   `predict` on 120 windows, non-contiguous label ids).
2. **k<=1 degenerate detector (D1):** `StickyHmmSmoother.fit_decode` with one
   unique init id leaves `last_model_ = None` and `decode` returns the single
   fitted id for every window. The snapshot format carries `fitted_ids` always
   and HMM arrays only when `k >= 2`; load reconstructs the degenerate smoother
   faithfully (no HMM). Test-pinned.
3. **Monitor per-state semantics on the new run (D2):** states can be absent or
   thin on the monitored day. Recalibrate mode follows the sweep row
   conventions exactly: a state with no snapshot reference → excluded row; a
   state with zero calibration-side windows on the new run → no-conformal-data
   row; below-`min_ref` gating identical to sweeps. In recalibrate mode alarms
   are emitted for SCORING-side windows only (the new run's calibration-side
   windows are consumed by threshold calibration and are reported as
   `consumed_for_calibration`, never alarmed — calibration bias rule). In
   frozen mode every valid window gets a verdict.
4. **Scarcity metric coherence (D4):** the conformal threshold for TPR@alpha is
   calibrated on CLIP-LEVEL scores (mean over windows) of held-out
   training-normal clips, and applied to clip-level scores of test clips —
   never window-calibrated/clip-applied. Reported metrics per cell: clip-level
   AUC, clip-level pAUC (`sklearn.roc_auc_score(max_fpr=0.1)`, the
   STANDARDIZED/McClish partial AUC — definition named in every output), clip
   TPR@(conformal alpha=0.05), window-level AUC (secondary). No pAUC utility
   exists in this repo yet; this harness introduces it with the definition
   pinned by a test against a hand-computed case.
5. **MIMII windows are per-window standardized** (P4 corpus convention,
   `_standardize`) for every representation — removes clip-gain confounds AND
   absolute-level anomaly cues; documented as the conservative choice in all
   outputs. `AudioFeaturizer`'s output width varies with `rate_hz` (bands
   capped at Nyquist) — self-consistent within the 16 kHz corpus, stated in the
   notes. The student on MIMII measures TRANSFERABILITY of a PSHP-distilled
   encoder (framing restated in outputs).
6. **run_sweep split parity confirmed:** top split (`calibration_frac`, seed) →
   nested split of the calibration half (0.5, seed+1) → references from fit
   windows, thresholds from conformal windows — `fit_snapshot` mirrors this
   1:1 (verified against `run_sweep`'s actual code, not its docstring).

## 4. Leakage & honesty rules (restated once, enforced everywhere)

- Snapshot fitting follows the sweeps' split discipline; the monitor never refits
  references on the monitored run (recalibrate mode touches THRESHOLDS only).
- MIMII splits are CLIP-level; abnormal clips never touch fitting or calibration.
- Every output restates: PSHP alarms = candidates (no fault labels); MIMII results =
  public-proxy evidence; event-eval = prepared harness awaiting campaign data.
- Detected-state labels come from the snapshot's own detector on the new run —
  per-encoder label non-alignment caveats do not arise (one detector, one label
  space, the snapshot's).

## 5. Testing

Per module: snapshot round-trip (bitwise apply-equality, geometry refusal, version
mismatch, scorer whitelist refusal), monitor CLI via the `test_apply_detector.py`
monkeypatch style (fake index + two-state PreparedRun, both threshold modes, gating,
notes content), events edge-case battery (synthetic), labeled-MIMII iterator on
synthetic wav trees (tmp-path, both classes, determinism, clip integrity), scarcity
harness on a synthetic corpus with the handcrafted featurizer (fractions monotone
n_train, csv schema, seed determinism, skip-on-missing-checkpoint). Gates as always:
pytest `-m "not data"`, ruff, mypy strict.
