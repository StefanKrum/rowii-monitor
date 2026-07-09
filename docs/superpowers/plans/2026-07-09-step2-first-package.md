# Step 2 First Package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mode-conditioned anomaly scoring skeleton (references → kNN/Mahalanobis → per-mode conformal) with the first two label-free evidence artifacts: per-state FAR tables (within-day + cross-day) and the four-day anomaly-candidate register.

**Architecture:** New `src/rowii/anomaly/` package consuming Step-1 outputs. Shared run-preparation is promoted from `scripts/run_step1.py` into `src/rowii/pipeline.py` (behavior-preserving refactor, feature caching added). References are keyed by DETECTED cluster ids (run-time realism; k groups may include load sub-clusters — by design §3.5); GT state names enter only evaluation views via the existing majority mapping. Spec: `docs/superpowers/specs/2026-07-09-step2-mode-conditioned-ad-design.md`.

**Tech Stack:** unchanged (Python 3.12, numpy/scipy/sklearn/pandas; `[beats]` extra for BEATs variants). Branch: `feat/step2-skeleton`.

## Global Constraints

- HARD for every implementer dispatch: direct implementation only, NEVER use the Agent tool / spawn subagents / write plan documents.
- Step-1 modules' behavior unchanged (refactor task carries regression protection: full suite green, CLI outputs byte-stable on the miniature e2e fixture).
- No partner (Bruno) values adopted anywhere; cross-references labeled as such.
- Leakage rule: calibration and scoring windows never share a 12-min recording segment.
- Conformal exactness: threshold index = ceil((n+1)*(1-alpha))/n quantile; achievable-alpha floor n >= 1/alpha - 1 reported per state; low-confidence flag below floor.
- No real data in unit tests (`@pytest.mark.data` tier for real runs); mypy strict + ruff 100; conventional commits; NEVER Co-Authored-By.

## Tasks (summary level; each dispatched with full interface blocks as established)

### Task S1: pipeline extraction refactor + feature cache
- Move `_prepare_run_features`/grid/mask logic from `scripts/run_step1.py` into `src/rowii/pipeline.py` as `prepare_run(run, variant, cfg, *, use_cache=True) -> PreparedRun(features, grid, valid_mask, stream_meta)`; script becomes a thin caller (imports from rowii.pipeline).
- Feature cache: `results/cache/<run>--<variant>.npz` (features float64, mask, grid params) + config-fingerprint sidecar; `--no-cache` flag. Cache write/read unit-tested with synthetic fixture tree.
- Regression: full suite green; miniature e2e produces identical summary row values pre/post refactor (assert against recorded values in the test).

### Task S2: references.py
- `SegmentSplit`: leakage-safe partition of a run's windows by 12-min source segment (from stream_meta), `split_for_calibration(frac, seed)` → calib/score window index sets, never splitting a segment.
- `build_references(features, labels, split) -> dict[label, np.ndarray]` (+ `pooled` key). labels = int cluster ids (detected) or str states (gt mode). Min-size guard (< min_ref windows → state flagged, excluded from per-state scoring, logged).
- Tests: segment integrity (no segment straddles), determinism, min-size exclusion.

### Task S3: scorers.py
- `KnnScorer(k=1, metric="cosine")`: fit(reference) stores normalized matrix; score(x) = 1 - max cosine similarity (distance form). Batched matmul; memory cap via chunking.
- `MahalanobisScorer(shrinkage=0.1)`: diag covariance + shrinkage toward global variance; score = sqrt Mahalanobis.
- Tests: constructed outlier scores >> inlier scores (both scorers, both metrics families); cosine equivalence to sklearn pairwise on small case; chunking equals unchunked.

### Task S4: conformal.py
- `calibrate(cal_scores, alpha) -> ConformalThreshold(threshold, n, achievable_alpha_floor, low_confidence)`; `p_values(scores, cal_scores)` = (1 + #{cal >= s}) / (n + 1).
- Validity test: 100 synthetic repetitions (seeded), empirical FAR within two-sided binomial CI of alpha for n in {19, 50, 500}; floor rule exact at n=19/alpha=0.05; low_confidence flips below.

### Task S5: sweep.py + synthetic e2e
- `run_sweep(prepared, det_labels, cfg2) -> SweepResult(far_table, scores_df, candidates_df)` over conditioning {per-state, pooled} × scorer {knn, mahalanobis}; per-state FAR on held-out normal + candidates (top-K by p-value per state).
- Synthetic e2e: 3-cluster stream with injected far-out windows → injected windows dominate candidates; FAR within CI; pooled vs per-state produce different FAR under state-dependent score scales (constructed).

### Task S6: run_step2.py CLI + smoke
- argparse: `--protocol within-day|cross-day`, `--run/--variant/--scorer/--conditioning/--alpha/--top-k`, cross-day pairs auto from SCADA days. Outputs per spec §2 paths + `candidate_register.md` merge logic (append-only, provenance columns).
- Smoke tests on synthetic fixture tree (as Step-1 CLI tests do).

### Task S7 (checkpoint): real-data first evidence
- Within-day FAR tables for 250526/290626/010726 (fusion + audio, kNN, per-state vs pooled, alpha 0.05); cross-day matrix (3×3 SCADA days, fusion-kNN); candidates top-20/state for 010726; register incl. partner cross-reference section (time-window overlap vs partner's pre-start filling-valve slides — comparison only).
- README "Step 2 first evidence" section; honest notes (detected-label conditioning; calibration sizes per state with low-confidence flags).
- STOP for review with Stefan (numbers + register).

## Verification per task: RED→GREEN evidence, full suite, ruff, mypy src scripts; reviewer per task as established (READ-ONLY constraint, scratch copies).
