# Step-2 Package 7: Robustness & Best System — Design Spec

**Date:** 2026-07-18 · **Branch:** `feat/step2-package7-robustness` · **Base:** main after PR #9
**Origin:** Stefan's scientific-ambition directive (2026-07-18): "wir wollen herausfinden,
ob man das so noch verbessern kann … möglichst das Beste, wissenschaftlich korrekt" —
plus four concrete challenges: adaptation never saw all operating modes; extend the
vibration pretraining corpus; TF-C finetuning on final PSHP data IS in scope; and the
frozen-threshold fragility (0.776/0.522 cross-day) demands a robustness investigation.

## 1. Problem

Package-6 closed the original roadmap, but its own evidence exposes the next question:
frozen thresholds fail cross-day for two DISTINGUISHABLE reasons (measured 2026-07-18
from the monitor artifacts): a GLOBAL shift on the cross-CONFIG day (250526 — recorded
**2026-06-25** per the raw filenames; the run label is an inherited misnomer — all
states' score medians 0.86–2.51× threshold, a recording-chain signature) and a
STATE-SPECIFIC drift on the same-config pair (290626: states 1/3 hold 0.09/0.03 ≈ α,
state 0 breaks at 0.73 and state 2 at 0.39). PROVENANCE (spec-review finding, verified
against Gantner headers): the config split is NOT the 15.06 sensor installation —
250526 AND 270626 carry `MeasName: 2026-06-15` while 290626 AND 010726 both carry
`MeasName: 2026-06-29`; the empirical "same-config pair" is the 29.06-config era.
"cross-setup" reads "cross-config" throughout this spec; the README data-layout
section's 15.06-as-final wording is corrected in this package. Meanwhile every
adapted artifact (LoRA/FT/student/snapshot) was fit on ONE day (010726-tu_ph_tu)
that contains NO pump operation, although same-config pump runs exist (290626-pu
~360 min, 010726-pu ~84 min; 270626-pu_ph_pu_ph_pu_ph-1 ~312 min is OLD-config).

## 2. Scope

Five experiment families, each with its own falsifiable hypothesis; plus the corpus/
tooling work they need. Non-goals: streaming/online monitor rewrite (rolling
recalibration is DESIGNED and discussed, implemented only as a batch ablation);
new encoder architectures; any PSHP fault-label claims (campaign still pending).

## 3. Design decisions

### D1 — Multi-run/multi-mode training pools (foundation for D2/D4/D6)

New module `src/rowii/anomaly/pools.py`:
`build_pool(runs: list[PreparedRun], run_names: list[str], side: Literal["calibration","fit"], sweep_cfg) -> PoolResult`
— per run: the SAME `split_by_segments` top split (seed 7) every sweep uses; the pool
collects ONLY the requested side's windows, tagged with (run_name, window). Leakage
rule unchanged and now CROSS-RUN: scoring sides of every run remain untouched by any
pooled fitting/calibration; a pooled artifact evaluated on run R uses R's scoring side
only. `PoolResult` carries per-run provenance (n windows per run per label) for the
sidecars. The canonical FINAL-SETUP pool: `{010726-tu_ph_tu, 290626-tu, 290626-pu,
010726-pu}` calibration sides (TU+PH+PU coverage; 270626-1 excluded from the
canonical pool — no Betriebsdaten/GT for its day, kept available via CLI flag with a
documented caveat; 250526-* NEVER pooled — it is the held-out cross-SETUP probe).

### D2 — Leave-one-day-out multi-day references (the "day as a dataset point" experiment)

Extend `scripts/run_step2.py` with `--protocol cross-day-pooled`: references and
detector from a POOL of fit days, thresholds in BOTH modes on the held-out day:
- Rotations over final-setup GT days {250526 excluded — see D1}: fit {290626-tu+pu,
  010726-*} → test 250526-tu (cross-setup probe, reported separately), and fit
  {010726-*} → test 290626-*, fit {290626-*} → test 010726-* (same-setup rotations).
- Detector: fitted on the RICHEST pool day (010726-tu_ph_tu) and APPLIED to every
  other run (P2's transfer recipe; a pool-concatenated HMM fit would stitch
  discontinuous days into one Viterbi chain — rejected, documented). References:
  pooled over the pool days' nested-fit sides, per detected state.
- **Hypothesis H2 (falsifiable):** pooled references reduce the FROZEN-mode FAR excess
  on the held-out same-setup day vs the single-day baseline (P6 monitor numbers are
  the baseline: 0.522 pooled alarm rate on 290626). Recalibrate mode must stay ≈ α
  (sanity). Report per-state + aggregate, both modes, vs the single-day rows.

### D3 — Session normalization (deployment-realistic, label-free)

`src/rowii/anomaly/normalize.py`: `SessionNormalizer.fit(embeddings_first_n) /
transform(embeddings)` — per-run z-score of the feature matrix estimated from the
FIRST `--norm-minutes` (default 20) of the run's windows (valid rows only), applied
to every window of that run before scoring/calibration. Deployment-realistic: the
first minutes of a new day are observable without labels; caveat documented (if the
first minutes contain a fault, normalization absorbs part of it — mirrors the rolling-
recalibration caveat; the fit-day references are normalized with the FIT day's own
first-N stats, symmetric). Wired as `--session-norm` into the within-day/cross-day/
pooled protocols and the monitor CLI (snapshot stores the fit-day stats).
- **Hypothesis H3:** session norm shrinks the GLOBAL (cross-setup, 250526) frozen
  excess substantially more than the state-specific (290626) one — matching the
  2026-07-18 two-cause decomposition.

### D4 — TF-C continued pretraining on PSHP (the finetune experiment)

`scripts/pretrain_tfc.py --continue-from models/pretrained/tfc/tfc_audio.pt
--corpus pshp-pool` : continued contrastive (NT-Xent) pretraining on the D1 canonical
pool's raw audio windows (calibration sides only, resampled to TF-C's 8 kHz) →
`tfc_audio_pshp.pt`. Same for a from-scratch control `--corpus pshp-pool` without
`--continue-from` (`tfc_audio_pshp_scratch.pt`) to separate "industrial init helps"
from "PSHP data suffices". Evaluation: Step-1 state ARI on all GT days + within-day
FAR + the D2 pooled protocol, as variant `audio-tfc` under swapped
`ROWII_TFC_AUDIO_CHECKPOINT` (result dirs archived with suffixes, P5 procedure).
- **Hypothesis H4:** PSHP-continued TF-C improves state-ARI/calibration on PSHP over
  MIMII-TF-C; the P5 expectation (same-condition tool) predicts limited cross-day
  gain — either outcome is publishable.

### D5 — Vibration corpus extension + vib-TF-C v2

`scripts/download_corpora.py` grows the remaining Paderborn HEALTHY sets (K003–K006;
same Zenodo scheme as K001/K002 — URLs/sha256 verified at implementation time, the
established live-HEAD procedure). OPTIONAL second corpus (IMS or XJTU-SY) only if
format verification at implementation time shows a loader is < 1 day of work —
decided in the plan, never silently. Re-pretrain `tfc_vib_v2.pt` (window count
reported vs the 808 floor), re-run the vibration-tfc evidence (Step-1 vib ARI,
within-day FAR) with both checkpoints side by side.
- **Hypothesis H5:** lifting the data floor improves vib-TF-C's state separation;
  if it does NOT, the floor caveat is retired and replaced by a stronger negative.

### D6 — Multi-mode adaptation + student + snapshot v2

Re-run the P5 adaptation family on the D1 canonical pool instead of single-day:
`adapt_beats --runs <pool>` (target_windows extended to multi-run iteration,
per-run splits), `distill_beats --runs <pool>`, and `fit_snapshot` on pooled
references (D2 machinery) → `monitor_pool_fusion.npz`. Evaluate exactly like P5/P6
(sweeps under swapped checkpoints; monitor on 250526-tu + both 290626 runs incl.
PUMP-day monitoring, which the single-day snapshot could never do honestly).
- **Hypothesis H6:** pool-adapted LoRA keeps its on-day gains without the off-day
  degradation (more diverse adaptation data); the pool snapshot monitors pump days
  with per-state coverage the 010726 snapshot lacks.

### D7 — Rolling recalibration: batch ablation + design discussion (no streaming code)

`--thresholds rolling --rolling-minutes M` in the monitor: thresholds recalibrated
from the trailing M minutes' calibration-role windows, evaluated in the SAME batch
artifacts (roles extended with window-index provenance). Explicitly documented
limitation: a slowly developing fault can be absorbed — mitigated in the notes by
reporting BOTH rolling and fit-day-frozen verdicts side by side (double-reference
principle). This is an ablation + design contribution, not a production stream.

## 4. Evaluation & honesty rules

- 250526 is the CROSS-SETUP probe: never pooled, never adapted on; every 250526
  result labeled cross-setup. Same-setup conclusions cite 290626↔010726 only.
- All pooled/adapted artifacts carry per-run provenance sidecars; every FAR claim
  names its mode (frozen/recalibrate/rolling ± session-norm).
- No detection claims on PSHP (unchanged); candidates framing unchanged.
- Compute stays one-off + cached: pool feature needs = existing caches (all pool
  runs already cached for fusion/audio-beats from earlier packages; missing
  variants warmed once).

## 5. Testing

Per the house standard: pools (leakage: scoring windows never in any pool — probe
tests), normalize (stat correctness, first-N boundary, snapshot round-trip of stats),
cross-day-pooled protocol (synthetic two-day fixtures, rotation correctness,
baseline-comparison table schema), pretrain --continue-from (checkpoint lineage in
sidecar), corpora additions (synthetic trees), multi-run target_windows (per-run
split independence), rolling thresholds (window provenance, absorption caveat in
notes). Gates: pytest -m "not data", ruff, mypy strict.

## 6. Amendment A2 (2026-07-18, Stefan's directives + the illwerke-080726 delivery)

### A2.1 — Universality is the frame, not plant-specific maximization

Stefan (2026-07-18): the RQ targets scarce-data transfer to SIMILAR machines — the
system must not be over-fitted to ROWII. Binding consequences:
- Every improvement in this package is REPORTED along two axes: (a) performance on
  ROWII, (b) what it needs from a NEW machine (hours of normal data, labels: none).
  A method that wins on ROWII but requires plant-specific tuning is framed as such.
- The frozen-BEATs + conformal path stays the UNIVERSAL baseline in every table
  (it needs only ~20 min of calibration audio on a new machine); adapted/pooled
  variants are measured AGAINST it as "what plant-specific data buys".
- D4/D6 adaptation experiments get an explicit transfer readout where possible
  (e.g., pool-adapted encoder evaluated on the held-out cross-setup day = a
  proxy for "different but similar machine").

### A2.2 — Parameter comparisons are part of the deliverable

Robustness knobs are SWEPT, not point-estimated, and reported as comparison tables
(the research narrative "so funktioniert es am besten, weil ..."):
- alpha ∈ {0.01, 0.05, 0.10} on the headline protocols (calibration-size floors
  reported alongside: 1/alpha − 1).
- session-norm minutes N ∈ {5, 20, 60} (D3).
- rolling window M ∈ {30, 60, 120} minutes (D7), with per-state conformal-floor
  feasibility reported per M.
- pool composition ablation (D2): single-day vs 2-day pool vs full pool.

### A2.3 — illwerke-080726: the induced-event day joins the evaluation (TOP priority)

New delivery (verified by Bruno's crosscheck + skeptical audit, docs in
`~/Downloads/illwerke-080726-analysis/`): 2026-07-08, machine 1, Schonhammer
strikes at 24 logged positions + 2 guide-vane sweeps, in TWO sessions — standstill
(`Kalibrierung_ST`, 12:15–12:31 UTC) and PUMP operation at −279 MW
(`Kunstliche_Anomalie`, 14:43–15:01 UTC inside a 13:46–17:10 recording that is
otherwise strike-free pumping, incl. a real pump→phase-shifter changeover
~15:04–15:06 and one unlogged 17:15 transient), full-day SCADA. Same GANTNER
format/streams as every other day.

Binding integration (replaces D2's synthetic-demo status of the events harness):
1. **Ingest** as `illwerke-080726/20260708 Messung/{PU_STRIKES, ST_STRIKES,
   Betriebsdaten}` → runs `080726-pu_strikes` / `080726-st_strikes`; warm caches.
2. **events.csv from the VERIFIED minute table** (crosscheck-notes-v1.md): one
   event per logged strike minute (POS/VANE rows, UTC), committed under
   `docs/groundtruth/080726_events_{st,pu}.csv` with provenance header; the
   changeover window recorded as a separate `kind=machine-transient` entry.
3. **First REAL pillar-3 run:** monitor (pool snapshot, recalibrate mode,
   calibration drawn from strike-free segments) → `eval_events.py` → per-event
   TPR, first-alarm latency, FAR outside events — at the A2.2 alpha grid, for the
   headline representations. Honesty framing fixed in advance: induced impulsive
   events are a DETECTABILITY validation (Bruno's note: "Not a fault"), machine 1
   only, airborne + structure-borne classes reported separately where the mapping
   allows; window-level dilution (ms-scale strikes in 1-s windows) is an expected
   effect and reported, not hidden. Bruno's transient detector (band-MAD) is cited
   as the specialist reference point; ours is the generalist — complementary, not
   competing.
4. **080726 strike-free pumping hours** additionally join the D1 pool question as
   an ablation arm (they are machine-1 final-setup normals; ~3 h).

### A2.4 — Design-vs-implementation audit (new deliverable)

Stefan: "unsicher, wie sehr unser Design erfüllt ist, ob wir abweichen". A
systematic audit document: design chapter / methodology blueprint claims vs what
the code does, item by item (fulfilled / deviates-with-rationale / open), with
file references — delivered as `research/notes/` audit note + a thesis-ready
deviation table. Every KNOWN deviation already documented in specs (e.g. proxy
objective, ph_power_eps_mw) is collected, not re-litigated.

### A2.5 — Pipeline demo mockup (separate deliverable, not gated on D1–D7)

A visual mockup of the live-pipeline demo (state timeline + per-state anomaly
score trace + alarm markers + candidate report with timestamps), designed around
the 080726 day (real states incl. the changeover, real strike events) — first as
a static mockup for Stefan's review, then (separate decision) an implementation.
Purpose is dual: partner-facing demo AND Stefan's own understanding of the
pipeline. Bruno's demo (localization focus) is the sibling, not the template.

## 7. Amendment A3 (adversarial spec review, 2026-07-18 — all findings adopted)

The review (1 BLOCKER, 4 HIGH, 6 MEDIUM, 3 LOW) verified its findings by probes
against code, disk artifacts, and raw Gantner headers. Binding resolutions:

1. **[BLOCKER] Pool artifacts are NEVER evaluated on pool-member runs.** All D6
   evaluation runs come from held-out day groups via the D2 rotations (fit
   {010726-*} → monitor 290626-*, fit {290626-*} → monitor 010726-*, plus the
   cross-config probes). An optional within-pool demo requires excluding that
   run's windows from references AND thresholds and labels every output
   "within-pool (in-sample)". Rationale: recalibrate mode draws the new run's
   calibration side, which CONTAINS pool reference windows (kNN self-scores ≈ 0
   → deflated thresholds); frozen mode scores the reference windows themselves.
2. **[HIGH] D7 rolling gets floors + fallback:** default M=60 min; per (state,
   window): rolling threshold when the trailing same-state calibration count
   ≥ 1/α−1, else FALL BACK to the fit-day frozen threshold, flagged per window
   (`threshold_source ∈ {rolling, fit_day_fallback}`); per-state trailing-
   coverage statistics are a MANDATORY output. Motivating measurement (review
   probe on 290626-tu): at M=20 only 46.8% of scored windows reach the floor
   (states 2/3: 0%); at M=60 still 53.8%.
3. **[HIGH] Provenance corrected** (see the rewritten §1). 270626-1 carries the
   OLD config: its pool CLI flag caveat now names BOTH missing GT and the config
   mismatch; it is excluded from every "same-config" claim. README data-layout
   correction is a package deliverable. All prior notes' "25.05." date errors
   corrected in master-thesis (done 2026-07-18).
4. **[HIGH] Pooled detector so pump owns a cluster:** new
   `FittedDetector.fit_pooled(features_per_run, cfg, k)` — KMeans centers fit on
   the POOLED nested-fit features of all pool runs; emissions from pooled
   per-cluster stats (the existing `_init_means_covars` construction); sticky
   transmat fixed; NO cross-run EM chain — decoding is per run (Viterbi within
   each run only). k selected by GT-state-ARI over k ∈ {4, 5, 6} on the pool
   days, reported. Mahalanobis on pooled (possibly multi-modal) references
   carries an explicit caveat; kNN is the primary pooled scorer. H6 reworded:
   "pump windows are scored against pump-containing references under a detector
   whose label space includes pump", not "per-state coverage".
5. **[HIGH] D3 boundary pinned:** the DETECTOR always consumes RAW features
   (labels are norm-invariant by construction); session norm applies to the
   SCORING space only. Stats are median/MAD with the house 1e-8 floor
   (state-mix confound documented: 290626's first 20 min are 100% one state —
   the N-sweep in A2.2 is the sensitivity probe). Snapshots storing norm stats
   bump `SNAPSHOT_FORMAT_VERSION` to 2; v1 snapshots are REFUSED under
   `--session-norm` with a clear message. P6 comparability is FAR-level only.
6. **[MEDIUM] Compute owned in §4-replacement:** pool warming is REAL work
   (290626-pu/010726-pu have NO audio-beats/audio caches; ~30–60 min + ~0.7 GB);
   D6 swapped-checkpoint evaluation re-extracts per checkpoint per run (~1–2.5 h,
   the package's dominant compute); TF-C pretraining pair ~1.5–2.5 h; total
   ~5–9 h wall-clock, sequenced in §8.
7. **[MEDIUM] `build_pool` sides:** `Literal["calibration", "fit", "conformal"]`
   — pooled FROZEN thresholds calibrate on the pool's nested-CONFORMAL scores
   (run_sweep/snapshot convention). WARNING pinned in the module docstring:
   `_cross_day_per_state_sweep` uses the TOP split as fit/conformal — do NOT
   copy its split semantics for pooled work.
8. **[MEDIUM] Protocol renamed** "held-out-day-group evaluation" (LODO-style;
   n=2 same-config rotations — the honest count). Pools explicitly include
   010726-tu1-morning/010726-tu2 (GT + caches exist). `--protocol
   cross-day-pooled` gets a parser guard: fit day-groups and the test day-group
   must be disjoint (day-group = calendar day of the run's first file). The
   missing single-day baseline for 290626-pu (one monitor run with the P6
   snapshot) is added to §8 so H2 has a comparator on every test run.
9. **[MEDIUM] D4 tooling made explicit:** `pretrain_tfc.py` gains
   `--corpus pshp-pool` (windows via `iter_target_windows(run, cfg,
   target_hz=8000)` per pool run — the P5 leakage rule for free) and
   `--continue-from PATH` (loads the checkpoint state dict as init; sidecar +
   checkpoint carry a `continued_from` lineage field). Pool windows are
   MATERIALIZED ONCE to an npz (~1.2 GB) so the two pretrainings don't stream
   ~40 GB of Gantner twice; `_CHECKPOINT_NAMES`/`_MANIFEST_DIRS` gain the new
   corpus keys.
10. **[MEDIUM] D5 corrected:** Paderborn downloads come from
    `groups.uni-paderborn.de/kat/BearingDataCenter/` (NOT Zenodo — only MIMII
    is Zenodo); K003–K006 extend the existing `_SHA256_TBD` + live-HEAD
    procedure; `tfc_vib_v2.pt` gets an explicit output-name mechanism.
11. **[MEDIUM] D6 APIs named:** new `fit_snapshot_from_parts(detector,
    references, calibration_scores, thresholds, *, provenance...)` beside the
    single-run `fit_snapshot` (unchanged); `adapt_beats`/`distill_beats` gain
    `--runs` (plural); `--max-windows` is the TOTAL budget drawn ROUND-ROBIN
    across runs (a sequential chain would train almost entirely on run 1);
    per-run window counts recorded in the sidecars.
12. **[LOW] §1 corrected** (state 2 breaks too — done in the rewrite).
13. **[LOW] 250526-pu-morning joins the evaluation** as the cross-config PUMP
    probe (GT + fusion cache exist; near-free).
14. **[LOW] §8 execution/synthesis added below; run names written in full.**

## 8. Execution & synthesis (order)

1. Ingest illwerke-080726 (A2.3), warm pool caches (audio, audio-beats for
   290626-pu/010726-pu/010726-tu1-morning/010726-tu2 + 080726 runs), transcribe
   `docs/groundtruth/080726_events_{st,pu}.csv`, check 080726's `MeasName`
   config era and record it in the README coverage section.
2. Missing baselines: monitor 290626-pu + 250526-pu-morning with the P6
   single-day snapshot (both modes).
3. D1/D2: pooled references + held-out-day-group rotations (both modes,
   α grid) vs single-day baselines.
4. D3: session-norm ablation on the same rotations (N grid).
5. D4: materialize pool windows → tfc_audio_pshp + scratch control → Step-1
   ARI + rotations.
6. D5: downloads → tfc_vib_v2 → vibration evidence side by side.
7. D6: pool-adapted LoRA/student + `fit_snapshot_from_parts` pool snapshot →
   held-out evaluations; D7 rolling ablation (M grid) on the monitor runs.
8. A2.3 pillar-3 run on 080726 (events harness, α grid, headline
   representations); A2.4 design audit; A2.5 demo mockup.
9. README package-7 section + master-thesis note (universality axis per A2.1 in
   every table) → final whole-branch review → PR #10 → merge.
