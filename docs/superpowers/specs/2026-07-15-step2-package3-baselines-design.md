# Step-2 Package 3: Baselines & Scoring Completeness — Design Spec

**Date:** 2026-07-15 · **Thesis anchors:** Design chapter § Anomaly scoring (classical
baselines retained on the same representations; reconstruction paradigm; the
OC-SVM + IF + LSTM-AE majority ensemble as an explicit design commitment; sub-cluster
back-end as reference configuration) and § Multimodal detection (score-level fusion —
the second of three fusion levels; cross-attention stays package 5).
**Precondition (met):** package 2 merged on main (14beb56): `FittedDetector`, scarcity
machinery, BEATs caches warm, true-UTC axis.

## 1. Questions this package answers

1. **Where do classical one-class detectors and reconstruction scorers stand** against
   the kNN/Mahalanobis reference on the SAME representations, splits, and conformal
   harness? (The design retains them as baselines; nobody has run them here yet.)
2. **Does the design-committed majority ensemble (OC-SVM + IF + LSTM-AE) suppress
   false alarms** relative to its members, as the chapter claims from first
   principles?
3. **Does score-level fusion** (audio branch + vibration branch combined at the score
   stage) add anything over feature-level fusion?
4. **Does conditioning granularity matter?** Per-state (k=4) vs sub-cluster (k=8, 12)
   detected-label conditioning — quantifying the "detected beats GT" sub-cluster
   mechanism from package 1.

## 2. Binding design decisions

### D1 — Classical one-class scorers (`src/rowii/anomaly/scorers.py` additions)

Three new classes implementing the existing `Scorer` protocol (`name`,
`fit(reference) -> Scorer`, `score(x) -> (W,) float64`, higher = more anomalous):

- `OcSvmScorer(nu=0.1, gamma="scale")` — sklearn OneClassSVM (RBF);
  `score = -decision_function(x)`.
- `IsolationForestScorer(n_estimators=200, random_seed=7)` —
  `score = -score_samples(x)`.
- `LofScorer(n_neighbors=20)` — LocalOutlierFactor with `novelty=True`;
  `score = -score_samples(x)`.

**Polarity is set EXPLICITLY per scorer with a docstring note** (v1 lesson: polarity
auto-detection is unreliable for degenerate models; none here). All share
`_check_reference`. `n_neighbors`/`nu` guards: reference smaller than the parameter
raises the sklearn error untouched but `fit` documents it; the sweep's `min_ref=20`
floor makes tiny references structurally rare.

### D2 — Reconstruction scorers (torch; the `[beats]` extra provides torch)

New module `src/rowii/anomaly/recon.py` (torch imported lazily inside methods, same
pattern as the BEATs wrapper; tests run on CPU with fixed `torch.manual_seed`):

- `MlpAeScorer(hidden=(128, 32), epochs=200, lr=1e-3, batch_size=256, seed=7)` —
  feature-vector autoencoder on ANY variant's `(N, F)` matrix; score = per-window
  reconstruction MSE.
- `LstmAeScorer(hidden=64, epochs=100, ...)` and `ConvAeScorer(channels=(16, 32),
  epochs=100, ...)` — consume the NEW `logmel` variant (D3): each window's flattened
  log-mel patch is reshaped internally to `(frames=49, mels=64)` (LSTM: frames as
  timesteps) / `(1, 64, 49)` (Conv). The window-internal TIME axis is the sequence —
  no cross-window contiguity is needed, so the `Scorer` protocol holds unchanged.
- Device: reuse the repo's existing device selection used by the BEATs path; CPU
  fallback always works. Training determinism: seeded; MPS nondeterminism caveat
  documented (tests assert on CPU).

### D3 — New `logmel` audio variant (input for Conv/LSTM AEs)

`pipeline._streams_for_variant` gains `"logmel"` (audio streams only); featurizer:
per 1-s window, 64-band log-mel over 25 ms frames with ~20 ms hop → 49 frames,
flattened to F = 3136 float64 columns, named `"<stream>::logmel_f<frame>_m<mel>"`.
Cached like every variant (`results/cache/<run>--logmel.npz`, ~0.7 GB/run for the
three SCADA days). Uses the primary mic stream only (`RAWGeneratorMic__0`) to bound
size — documented; the multi-mic extension is future work.

### D4 — Majority ensemble (decision level; the design's own formulation)

The chapter commits to majority voting of OC-SVM + IF + LSTM-AE ("requiring agreement
before an alarm suppresses false alarms"). Voting is a DECISION-level rule, so it
lives in the sweep orchestration, not in a `Scorer`:

- Each member is fitted per state on the same reference, gets its OWN per-state
  split-conformal threshold at the same alpha, alarms independently.
- Ensemble alarm = >= 2 of 3 members alarm. New evaluation view
  `far_table_ensemble.csv`: per state, each member's realized FAR + the ensemble's,
  same schema plus a `member` column.
- No conformal guarantee is claimed FOR the ensemble decision (the members hold their
  own marginal guarantees; the intersection bound is reported: ensemble FAR <= sum of
  any two members' alphas is NOT tight — report empirical only, honesty note in the
  writer). LSTM-AE member runs on `logmel`; OC-SVM/IF on the sweep's variant features
  — the ensemble is therefore evaluated on runs where BOTH variants share the grid
  (guaranteed: logmel uses the same primary-mic stream and window grid construction;
  the orchestration asserts `n_windows` equality and window alignment via `t0_ns`).

### D5 — Score-level fusion (p-value combination; principled and conformal-native)

Feature-level fusion exists (package 1). Score-level fusion combines the AUDIO and
VIBRATION branches at the score stage using p-value combination:

- Within the `fusion` variant's PreparedRun, the column split audio-vs-vibration is
  recovered from `feature_names` stream prefixes (`*Mic*` vs `*Vib*`) — robust, no new
  extraction.
- Per state: fit the base scorer (kNN default; Mahalanobis variant) per branch on the
  branch's columns; compute per-branch conformal p-values against the branch's own
  calibration scores; combine per window with **Fisher's method**
  (`chi2 = -2 (ln p_a + ln p_v)`, higher = more anomalous) and, as the max-rule
  contrast, **Tippett** (`score = -min(p_a, p_v)` equivalent ordering); the combined
  statistic is then calibrated with the SAME split-conformal machinery on the
  calibration side — the final FAR guarantee is restored by construction.
- Orchestration-level (like D4): new view `far_table_scorefusion.csv` with rows per
  state × rule ∈ {fisher, tippett} + the two single-branch baselines.

### D6 — Conditioning-granularity sweep (sub-cluster mechanism, quantified)

`run_step2.py` gains `--states K` (within-day only; default = config n_states = 4):
detected labels come from `FittedDetector.fit(..., k=K)`. Execute K ∈ {4, 8, 12} on
010726-tu_ph_tu (fusion, kNN + Mahalanobis, per-state): FAR table + candidate top-K
stability across K (overlap of top-20 sets between K values via the existing
`match_by_time`). Honest note: larger K shrinks per-state calibration pools (floor
flags expected at K=12 — that IS the finding: granularity vs achievability
trade-off).

### D7 — Execution & evidence (within-day, 3 SCADA days; all cache-hit or one-off)

- Classical scorers: fusion + audio-beats variants × per-state/pooled × 3 days.
- Recon scorers: MLP-AE on fusion + audio-beats; LSTM/Conv-AE on logmel; per-state
  conditioning; 010726 + 250526 (290626 starved — expected mostly excluded, still
  run, flags reported).
- Ensemble + score fusion + granularity per D4-D6.
- Comparison table vs the package-1/2 kNN/Mahalanobis numbers; README section +
  research note; summary.csv gains `scorer` values for the new names (schema
  unchanged — scorer column already exists).

## 3. Non-goals (later packages)

LoRA/full fine-tune/KD/INT8 (package 5), TF-C pre-training (package 4),
cross-attention fusion head (package 5), detector serialization/monitor CLI
(package 6), any labeled-fault evaluation.

## 4. Acceptance

- Unit tests per scorer (synthetic separable data: anomalies score higher; polarity
  pinned with constructed outliers; determinism with fixed seeds [CPU]; protocol
  conformance incl. `_check_reference` reuse); logmel featurizer tests (shape, names,
  cache round-trip); ensemble view test (member disagreement suppresses ensemble
  alarms on constructed scores); Fisher/Tippett combination tests (known p-values →
  known ordering; guarantee restored = calibrated combined statistic holds FAR on
  exchangeable synthetic data); `--states K` test (labels from k=K detector,
  conditioning rows match K).
- Real-data artifacts per D7; all existing tests green; ruff/mypy clean; conventional
  commits on branch `feat/step2-package3-baselines`; per-task adversarial review +
  final whole-branch review; PR at the end (merge decision Stefan's).
- No partner values adopted anywhere.
