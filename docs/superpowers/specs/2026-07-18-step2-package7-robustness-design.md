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
from the monitor artifacts): a GLOBAL shift on the cross-setup day (250526, all states'
score medians 0.86–2.51× threshold — recording-chain signature; the day predates the
final 15.06 setup) and a STATE-SPECIFIC drift on the same-setup pair (290626: states
1/3 hold 0.09/0.03 ≈ α, only state 0 breaks at 0.73). Every mitigation so far is
per-day recalibration. Meanwhile every adapted artifact (LoRA/FT/student/snapshot) was
fit on ONE day (010726-tu_ph_tu) that contains NO pump operation, although final-setup
pump runs exist (290626-pu ~360 min, 010726-pu ~84 min, 270626-1 ~312 min).

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
