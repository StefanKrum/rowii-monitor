# Step 2 — Mode-Conditioned Anomaly Detection: Design Spec (first work package)

**Date:** 2026-07-09 · **Thesis anchor:** Design chapter §3.4 (representation/adaptation),
§3.5 (scoring + mode conditioning + conformal), §3.7 (evaluation design). The thesis
chapter IS the design authority; this spec scopes the FIRST implementable package.
**Precondition (met):** Step 1 merged on main — per-window operating-state labels and
segments for all four measurement days (25.06 / 27.06 / 29.06 / 01.07).

## 1. Goal of this package

Build the Step-2 scoring skeleton exactly as the thesis reference pipeline defines it,
and produce the first two evidence artifacts that need no anomaly labels:

- **Pillar 1 (FAR control):** does the per-mode conformal threshold hold its nominal
  false-alarm rate on held-out normal data, per state and per day?
- **Candidate register (pillar-2/4 groundwork):** an outlier sweep over all four days
  that surfaces the highest-scoring windows per run × state as ANOMALY CANDIDATES with
  provenance — including an explicit cross-reference against externally flagged events
  (partner's pre-start filling-valve observation), verified independently, never adopted.

## 2. Scope (this package) and non-goals

In scope:
- Per-state normal references built from Step-1 detected labels (not GT — run-time
  realism; GT states used only in evaluation views), with a config switch
  `reference_labels: detected | gt` for diagnostics.
- Scorers (design §3.5): kNN embedding distance (k=1, cosine; design-cited default) and
  Mahalanobis (diag shrinkage) — each on BOTH representations already produced by Step 1:
  handcrafted features and frozen-BEATs embeddings (audio branch; `[beats]` extra).
- Mode conditioning axis: per-state reference vs mode-agnostic pooled reference (the
  design's central conditioning comparison).
- Per-mode split-conformal calibration (α default 0.05): threshold = ⌈(n+1)(1−α)⌉/n
  calibration-score quantile; report per state the calibration n, the achievable α floor
  (n ≥ 1/α − 1 rule), and mark states below the floor as low-confidence (design §3.5).
- Leakage-safe splits: calibration and scoring never share a 12-min recording segment;
  two protocols: (a) within-day blocked split, (b) cross-day (calibrate day A → score
  day B) — the cross-day FA-rate table is deliberately format-compatible with the
  partner's published cross-day matrix for side-by-side METHOD comparison (each side
  computed by its own code; no values adopted).
- Outlier sweep output: `results/step2/<protocol>/<run>/<variant>/` scores.parquet +
  conformal p-values + `candidates.md` (top-K windows per state, K=20, with UTC time,
  state, score, p-value, SCADA context columns [P, rpm, KS, Q] and a blank
  `assessment` column for the human pass).
- Candidate register: `results/step2/candidate_register.md` — every candidate carries
  source (our sweep / partner slide / operator), timestamps, evidence status
  (unreviewed / normal-but-rare / operationally-explained / unexplained).

Non-goals (later packages): LoRA/full fine-tuning, TF-C pre-training, from-scratch AEs,
fusion cross-attention, distillation/quantization, the data-scarcity curve, and any
labeled-fault evaluation (pillars 3–4 beyond register groundwork).

## 3. Architecture (new module `src/rowii/anomaly/`)

```
src/rowii/anomaly/
├── references.py   # build_references(features, state_labels, splits) ->
│                   #   per-state reference sets (+ pooled); leakage-safe segment splits
├── scorers.py      # KnnScorer (k=1 cosine), MahalanobisScorer (diag + shrinkage);
│                   #   fit(reference) / score(windows) -> float scores
├── conformal.py    # calibrate(scores, alpha) -> threshold + achievable-alpha floor;
│                   #   p_values(scores, calibration_scores)
└── sweep.py        # orchestration: per run × variant × conditioning × scorer ->
                    #   scores, FAR table, candidates
scripts/run_step2.py  # CLI: --protocol within-day|cross-day --variant ... --scorer ...
                      #   --conditioning per-state|pooled --alpha 0.05
```
Reuses Step-1 modules untouched (featurizers, grid, discovery, detection for labels).

## 4. Evaluation views (no anomaly labels exist)

1. **FAR table (primary):** realized FAR on held-out normal vs nominal α, per state ×
   day × variant × conditioning × scorer + binomial CI; the conditioning comparison
   (per-state vs pooled) is the design's first Step-2 claim.
2. **Cross-day FA matrix:** calibrate A → score B for all SCADA-day pairs (25.06,
   29.06, 01.07), per variant; partner-format-compatible table.
3. **Candidate ranking:** top-K per state; overlap check with partner's flagged
   pre-start events (time-window intersection, report only).
4. Honest confounder note in every report: detected-label conditioning inherits Step-1
   errors (quantified: state-accuracy 0.95–0.975); the `reference_labels: gt` switch
   isolates that effect diagnostically.

## 5. Acceptance

- Unit-tested (synthetic) conformal validity: empirical FAR within binomial CI of α
  across 100 synthetic repetitions; kNN/Mahalanobis sanity on constructed outliers.
- On real data: FAR table produced for all three SCADA days; per-state calibration
  sizes reported with low-confidence flags; candidate register populated with ≥ top-20
  per state for 01.07; partner cross-reference section present.
- All existing tests stay green; ruff/mypy clean; conventional commits; no partner
  values adopted anywhere.
