# Step 2 — Package 2: Per-State Cross-Day, Calibration-Scarcity Curve, BEATs Scoring Evidence

**Date:** 2026-07-15 · **Thesis anchors:** Design chapter §3.5 (scoring + mode
conditioning + conformal), §3.7 (evaluation design, scarcity axis). The thesis chapter
is the design authority; this spec scopes the SECOND Step-2 package.
**Precondition (met):** S-package merged on main (5a98896 lineage): `anomaly/` module,
within-day + pooled cross-day protocols, candidate register, Beta-exact conformal
validity tests. Feature cache (S1) live under `results/cache/`.

## 1. Questions this package answers

1. **Cross-day, per-state:** the S-package showed pooled cross-day conformal is
   structurally unusable (median realized FAR 0.60 at α=0.05). Open cell: does
   PER-STATE conditioning under a runtime-honest transfer protocol restore FAR
   control across days?
2. **Scarcity ("enough data per mode" — partner's open question):** calibration
   windows per state were the binding constraint (29.06: 3 of 4 states starved). How
   many windows/minutes per mode until (a) the conformal guarantee is achievable and
   (b) realized FAR and thresholds stabilize?
3. **BEATs for scoring:** Step-1 verdict was "BEATs loses state detection, wins load
   alignment". Anomaly scoring is a different task — does the frozen-BEATs
   representation beat handcrafted features as the scoring representation?

## 2. Binding design decisions

### D1 — `FittedDetector` (fit/apply split; enables cross-day transfer)

New class in `src/rowii/state/detect.py`:

- `FittedDetector.fit(features, grid, cfg, clusterer, k)` captures everything the
  Step-1 chain learns on the fit day: per-column zscore mean/std (**the fit day's —
  `run_detection` currently z-scores per run, so transfer MUST carry these**), the
  fitted clusterer (KMeans centroids / GMM), the fitted sticky HMM (means/covars
  learned with `params="mc"`; transmat/startprob stay fixed as today), min-dwell
  parameters.
- `.apply(features, grid) -> DetectionResult` on any day: standardize with the
  **fit-day** mean/std (never recompute), cluster-assign via `predict` (nearest
  centroid / GMM posterior argmax), Viterbi **decode with fit-day HMM parameters (no
  refit, no EM)**, duration filter, segments.
- `run_detection` is re-expressed as fit-then-apply on the same day. Regression gate:
  KMeans path reproduces existing results **bit-identically** on at least one real
  cached run (fixed seed); GMM path label-identical or the delta explained and
  documented (sklearn `fit_predict` vs `fit().predict()` subtlety).
- **No disk serialization in this package.** `FittedDetector` is deliberately the
  future serialization point for the runtime prototype. README gains a short
  "Compute reuse" section documenting the once-only story: BEATs checkpoint stored
  once (no pre-training on our side, frozen weights); embedding extraction cached
  once per run × variant (`results/cache/`, sha256-fingerprinted); clusterers/scorers
  refit in seconds by design (deterministic, seeded) so persisting them buys nothing
  yet.

### D2 — Protocol `cross-day-per-state` (runtime-honest)

- **Calibration day A:** leakage-safe three-way segment split exactly as in the
  S-package; detector fitted on A (detected-labels path only — no GT anywhere in the
  runtime path); per-state references and per-state conformal thresholds from A.
- **Scoring day B:** `FittedDetector.apply` → predicted states for B's windows;
  each B window scored against A's reference of its predicted state; A's per-state
  thresholds applied. Conditioning key = **A's cluster id** (this dissolves the
  cross-day label-alignment problem: one model, one label space). A's majority
  mapping to mode names is used for REPORTING ONLY.
- Pairs: all ordered pairs of the S-package cross-day run set (250526-tu, 290626-tu,
  010726-tu_ph_tu) — format-comparable with the S-package pooled matrix.
- Grid: variants {fusion, audio, audio-beats} × scorers {knn, mahalanobis} ×
  conditioning {per-state, **pooled recomputed under the same transfer protocol**}
  (so the per-state-vs-pooled delta isolates conditioning, not protocol).
- Views: realized FAR per pair × grid cell × state; GT-diagnostic view (FAR by true
  state); flags exactly as in S-package (state never predicted on B; n_cal below the
  α floor → low-confidence). Side-by-side table vs the S-package pooled cross-day
  numbers.
- **Stretch (qualitative only):** apply the 01.07-fitted detector to
  `270626-pu_ph_pu_ph_pu_ph` → first state timeline for the no-SCADA day; narrative
  cross-check against the partner's photo-hybrid description (report-only, no values
  adopted, no metrics claimed).

### D3 — Calibration-scarcity protocol

- Days: `010726-tu_ph_tu` (primary; richest per-state counts), `290626-tu`
  (secondary; shows the starved regime). Within-day protocol, scoring split FIXED
  across repetitions.
- **Primary curve (windows per mode):** per-state subsampling at the WINDOW level
  from the (already leakage-separated) calibration pool. Budgets
  n_cal ∈ {5, 10, 19, 39, 79, 159, 319, all} with the α=0.05 achievability floor
  (n ≥ 19) highlighted; budgets above a state's pool → saturated flag, no resampling
  beyond the pool. R = 50 repetitions, `numpy.random.default_rng(seed)` with seeds
  0..49.
- **Secondary curve (recording minutes per mode):** segment-level accumulation —
  randomly ordered calibration segments accumulated until the target window count is
  reached (12-min recordings never split; deployment-realistic "record N more
  minutes" view; per-state counts coupled through shared segments — stated openly).
  Smaller grid is fine.
- Metrics per state × budget × variant × scorer: mean realized FAR + empirical 5/95
  percentiles across reps, **exact Beta band** overlay (per-rep FAR ~ Beta, the
  S-package derivation — NOT binomial), threshold dispersion (std/IQR), fraction of
  reps in low-confidence/+inf.
- Grid control: variants {fusion, audio-beats}, scorer knn primary (mahalanobis
  secondary on the primary day only). Outputs under `results/step2/scarcity/`:
  tables (csv) + per-state figures (png, matplotlib) + `summary.md`.

### D4 — BEATs Step-2 execution

- CLI wiring for `audio-beats` / `fusion-beats` already exists in `run_step2` but has
  never executed (summary.csv holds only fusion/audio rows). Work = **cache warm-up →
  execution → analysis**, plus fixing any latent bugs the first real execution
  surfaces.
- **Warm-up first, in background, before other sweeps need it:** `prepare_run(...,
  use_cache=True)` for exactly the runs the package needs — {250526-tu, 290626-tu,
  010726-tu_ph_tu, 270626-pu_ph_pu_ph_pu_ph (stretch)} × {audio-beats, fusion-beats}.
  No speculative extraction beyond that list.
- Sweeps: within-day on the three SCADA days × beats variants × both scorers × both
  conditionings + candidates; cross-day-per-state (D2) and scarcity (D3) include
  audio-beats per their grids.
- Analysis: FAR parity vs handcrafted; candidate top-K overlap vs handcrafted
  candidates (UTC time-window intersection + Jaccard + qualitative table); explicit
  check whether BEATs independently surfaces the 4 needs-listening windows; register
  update with provenance tier `our-sweep-beats`.

### D5 — Reporting plumbing

- `summary.csv` gains a `protocol` column; existing rows stay valid (absent value ↔
  `within-day`); writers always emit it going forward.
- Every new report keeps the S-package honesty notes (detected-label conditioning
  inherits Step-1 error; low-confidence flags; no partner values adopted).

## 3. Non-goals (later packages)

LoRA/full fine-tuning, TF-C, from-scratch AEs, fusion cross-attention,
distillation/quantization, detector disk-serialization, quantitative 27.06 metrics,
photo-label digitization, the human listening pass.

## 4. Acceptance

- **Unit tests:** FittedDetector same-day equivalence (KMeans bit-identical
  regression on a real cached run or high-fidelity synthetic; documented GMM
  stance); apply-uses-fit-stats test (B standardized with A's mean/std — direct
  assertion); HMM-not-refitted-on-apply test (fit-day params unchanged / no EM call);
  subsampler tests (leakage-safety, seeded reproducibility, saturation flag, floor
  flagging); Beta-band reuse from S-package test utilities.
- **Real-data artifacts:** cross-day per-state FAR tables (all pairs × grid),
  scarcity tables + figures (01.07 primary, 29.06 secondary), BEATs within-day FAR
  tables + candidate overlap + updated register, stretch 27.06 timeline if reached.
- ruff + mypy clean; all existing tests stay green; conventional commits on branch
  `feat/step2-scarcity-crossday-beats`; adversarial per-task review + final
  whole-branch review; PR opened at the end — merge decision stays with Stefan.
