# Step-2 Package 8 — Mode-Model-Bank, Level-Only Recalibration, Explainable Results

Date: 2026-07-21. Base: main 5da8b35 (post-P7). Branch: `feat/step2-package8-modebank-explain`.

## 1. Problem & directives

Stefan (2026-07-21): (a) *"wieso nicht vier modelle bei state detection auf den
jeweiligen state trainieren und dann schauen welches anschlägt"* — build the
per-mode model bank as a real comparison arm; (b) results must become
understandable — *"solche analysen wie bruno es hat zu machen um auch die
ergebnisse verständlich zu machen"* — produce explanatory analyses/figures on
OUR data; (c) *"nicht zu schnell auf eine methode festfahren"* — keep exploring
scientifically sound alternatives.

New knowledge (attributed external reference, Rodrigues & Zhang 2026, digest:
`master-thesis/research/notes/external_2026-07-21_bruno_analyses_digest.md`):
their data-driven epoch discovery — a **+1.6…+2.6 dB microphone-only broadband
level step between 2026-06-27 and 2026-06-29, vibration flat** — lands exactly
on OUR independently logged Gantner MeasName era boundary (2026-06-29). Their
per-microphone level-offset self-recalibration recovers cross-epoch one-class
transfer (their PU 27→29: 100%→2% flag rate). Modes separate within-day
essentially perfectly; cross-day mode ID reaches ~94% with vibration features
under matched budget. All of these are HYPOTHESES for us to test independently
on our own pipeline — never numbers to import.

## 2. Scope (in / out)

IN: D1 mode-model-bank (Step-1 alternative + Step-2 chain probe), D2 level-only
channel recalibration, D3 explainability analysis suite + figures, D4 data
verifications (Gen-90° mic anomaly, TurbineVib__3 ch0 header scan, SCADA
timebase probe). OUT: localization of any kind (partner's paper territory);
importing partner numbers; new encoders; streaming.

## 3. Design decisions

### D1 — Per-mode model bank (Stefan's idea) as a Step-1 alternative

- **Labels for training:** SCADA ground-truth mode labels (`rowii.scada.labels.
  gt_labels`) on the FIT days only. Deployment story: SCADA exists during the
  supervised commissioning window; on monitored days the bank runs label-free
  and GT is used for EVALUATION only. This is a deliberate contrast to the
  unsupervised KMeans+HMM default (which needs no SCADA ever) — both arms stay
  in the thesis with this trade-off stated.
- **Bank members (one per GT mode present in the fit pool):** three model
  families, each a separate variant arm:
  (a) `gaussian` — diagonal Gaussian per mode on standardized features
  (Mahalanobis distance = negative score);
  (b) `knn` — per-mode kNN distance (mean cosine distance to k=5 nearest
  same-mode fit windows) — mirrors the existing Step-2 scorer;
  (c) `gmm` — per-mode GMM (m=2 components, diag) log-likelihood.
- **Assignment rule:** per window, argmin distance / argmax likelihood over the
  bank → mode label. **Rejection arm ("which one fires"):** per-mode conformal
  threshold on that mode's own calibration-side scores (same split machinery as
  Step-2, alpha grid {0.01, 0.05}); a window rejected by ALL bank members is
  flagged `no_mode_fits` — Stefan's "keins passt" novelty signal, reported as a
  rate per run (NOT claimed as a detector without induced-event evidence).
- **Optional smoothing:** the existing sticky-HMM/duration-filter smoothing
  applied on top of argmax labels (same `min_dwell_s`), as a `--smooth` flag —
  so the comparison vs KMeans+HMM isolates the LABELING mechanism, not the
  smoothing.
- **Representations:** fusion, audio-beats, vibration (the vibration arm tests
  the partner-reported vibration advantage independently).
- **Evaluation:** on the SAME six held-out rotations as P7 (fit pool trains the
  bank, held-out day evaluated): mode-accuracy vs GT + ARI (frame-level, valid
  windows, GT != unknown), side by side with pooled KMeans+HMM (k=4) from P7.
  Plus one Step-2 chain probe: per-mode references built from bank labels
  (fusion, B1→290626-tu, alpha 0.05) vs the P7 detected-state chain — does
  better state assignment translate into better FAR control?
- **Pillar-3 spot:** bank labels on 080726-pu_strikes (fusion + vibration),
  recalibrate + event-free calibration — TPR/FAR vs the P7 clustered-state
  chain (one alpha, 0.01, the strict operating point).

### D2 — Level-only channel recalibration (shape-preserving)

- **Mechanism:** additive offsets in the log/dB domain applied ONLY to
  level-type features — `*_log_rms`, `*_band_*`, `*_octave_*` — per feature
  column; shape features (`*_spectral_centroid`, `*_rolloff95`, `*_kurtosis`,
  ratios) untouched. Offset per column = median over the monitored run's first
  N minutes (N=20 default, label-free) minus the same statistic over the
  snapshot's reference windows (stored in the snapshot at fit time). This is
  the level-only, shape-preserving counterpart to P7's falsified full
  session-norm; the state-mix confound is expected to shrink because level
  features are far less mode-dependent than the full vector — measured, not
  assumed.
- **Surface:** `run_step2 --level-recal` (cross-day-pooled) and
  `monitor --level-recal`; mutually exclusive with `--session-norm`; snapshot
  stores the reference medians for the level columns (format extension,
  backward compatible: absent field = feature unavailable, refuse with exit 2).
- **Evaluation:** the six P7 rotations, FROZEN mode with level-recal vs raw
  frozen vs recalibrate (the question: does level-recal rescue frozen transfer
  across the era boundary — our independent test of the partner's
  recalibration idea, on our features and our protocol). Alpha 0.05; the two
  cross-era rotations are the headline cells. Plus pillar-3 side arm (fusion,
  both sessions, alpha 0.01/0.05) with event-free calibration.
- **Honesty:** if it fails like session-norm, that is the reported result;
  the comparison table shows raw-frozen | level-recal-frozen | session-norm-
  frozen | recalibrate on identical cells.

### D3 — Explainability analysis suite (`scripts/analyze_days.py`)

One CLI producing publication-grade figures + a markdown digest from EXISTING
caches/artifacts (no new sweeps). Analyses (each an independent subcommand,
all with per-figure PNG + underlying CSV):
1. `--rotations-heatmap`: day×day flag-rate heatmaps (frozen | recalibrate |
   level-recal once D2 exists) from the P7/P8 rotation artifacts — the visual
   replacement for the FAR tables Stefan found unreadable.
2. `--feature-stability`: per-feature cross-day shift in dB (level features) /
   raw units (shape features) with 12-minute-block bootstrap CIs, per mode
   (GT), classified slow (<3 dB) vs drifting (>=3 dB); output: sorted table +
   dot-interval figure. Independent replication of the stability-classes
   analysis type on OUR 231 fusion features across all five days.
3. `--era-step`: THE explanatory figure — per-day per-stream median level
   (mic streams vs vib streams, log_rms + key bands) across 25.06→08.07 in
   matched GT modes, with the MeasName era boundaries marked. Expected (to be
   measured): mic-only step at 2026-06-29, vibration flat — our own,
   independently computed account of WHY frozen thresholds failed, consistent
   with (and attributed alongside) the partner's finding.
4. `--mode-signatures`: per-GT-mode band/octave profile (median + IQR) per day
   — the "modes are separable" picture on our features.
5. `--tonal-table`: per mode×day the shaft/blade-pass/guide-vane band energies
   relative to neighboring octave floor (an SNR-like contrast, defined from
   our features; NOT the partner's exact metric) — machine-fingerprint table.
6. `--pillar3-figure`: TPR-vs-alpha grouped bars per representation for both
   080726 sessions from the P7 event_eval artifacts.
Digest: `results/analysis-days/README.md` linking every figure with a 2-3
sentence plain-language reading (German optional flag `--lang de` NOT built —
English only, thesis language).

### D4 — Data verifications (small, scripted, logged in the note)

1. Gen-90° mic anomaly: per-channel level comparison at plate-strike minutes
   on 080726 (our features) — does gen ch at 90° behave anomalously vs its
   ring? Documented outcome either way.
2. `RAWTurbineVib__3` ch0: header/dead-channel scan across all five days
   (which days carry live ch0), pinned to the era timeline.
3. SCADA timebase probe: locate the 080726 changeover in Betriebsdaten
   timestamps and compare against the audio-UTC state timeline (13:05:28 UTC)
   → states the Betriebsdaten timebase with evidence.

## 4. Evaluation & honesty rules

- GT mode labels: training only on fit days (D1); evaluation everywhere;
  never as runtime input on monitored days.
- Splits/leakage: unchanged P7 machinery (segment splits, pool-member ban,
  event-free calibration on strike days).
- Attribution: every analysis type inspired by the partner's work carries a
  one-line attribution in the script docstring and the digest; all numbers are
  computed from our caches; no partner JSON/number is read by any code.
- No localization computations of any kind.
- Frozen BEATs remains the universal baseline row where representations are
  compared (D1 uses audio-beats as one of three arms — satisfied).
- Every synthesis number artifact-verified; negative results reported plainly.

## 5. Testing

House rules (pytest, ruff, mypy --strict; tests first; temp-file+&& in every
implementer verification). New test files per module; CLI-level tests with the
established monkeypatch seams; deterministic fixtures (seeded); no real data
in tests. Mode-bank: unit tests per family (fit/score shapes, argmax,
rejection, all-rejected), split-parity test vs Step-2 splits, smoothing-flag
equivalence test (smoothing off = raw argmax). Level-recal: column-selection
test (level vs shape names), offset math golden test, snapshot round-trip,
mutual-exclusion + missing-field refusals. analyze_days: per-subcommand
artifact-shape tests on synthetic inputs; era-step column math unit test.

## 6. Execution order

1. D4 verifications (cheap, inform everything).
2. D1 bank (module + CLI + tests) → rotations × 3 families × 3 representations
   → chain probe → pillar-3 spot.
3. D2 level-recal (module + CLI + tests) → 6 rotations frozen comparison →
   pillar-3 side arm.
4. D3 analyze_days subcommands → full figure set from existing + new artifacts.
5. Synthesis: README package-8 section + master-thesis note (figures inline),
   memory update, final whole-branch review → PR #11 → merge.

Model policy for subagents (Stefan 2026-07-21): implementation/tests/readers on
sonnet; adversarial spec/whole-branch reviews on opus; fable only if a review
blocks twice.
