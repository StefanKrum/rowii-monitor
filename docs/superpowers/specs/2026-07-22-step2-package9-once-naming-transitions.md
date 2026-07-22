# Step-2 Package 9 — Once-Calibrated Operation, Named States, Transition/Dwell Handling

Date: 2026-07-22. Base: main 5c92b90 (post-P8). Branch:
`feat/step2-package9-once-naming-transitions`.

## 1. Problem & directives

Stefan (2026-07-22), three deployment-ergonomics asks, each a design decision:

- (a, → D1) *"ziel wäre es dass die kalibrierung auch da nur einmal und nicht
  jeden tag einmal gemacht werden muss, quasi nur einmal für eine neue anlage"*
  — the P7/P8 recipe recalibrates thresholds **every** monitored day; Stefan
  wants calibration to happen **once per plant**, not once per day.
- (b, → D2) *"kann man da die zustände auch benennen?"* — the timeline/alarms
  report detected states as bare cluster ids (`state 2`); name them.
- (c, → D3) *"in der praxis macht es ja nicht sinn dass ein modus nur irgendwie
  eine sekunde ist"* — one-window state flips are physically meaningless; handle
  transitions and minimum dwell as a first-class step.

These are refinements of the shipped best system (P7 §"best-system statement"),
NOT new methods. The honest tension in (a) is already answered by P7/P8's central
result — **frozen cross-day/cross-era thresholds do NOT hold their FAR** (P7 D1/D2:
frozen 0.075–1.000 vs recalibrate 0.03–0.10 across six rotations; README §package-7).
So "calibrate once" cannot mean "freeze forever and hope". This package operationalizes
the achievable version: **calibrate once, run frozen, and recalibrate only when a
label-free drift sentinel says the instrumentation changed** — which, on this campaign,
means once per DAQ-config era (the 2026-06-29 `MeasName` boundary; README §"SCADA
coverage", P8 D3 era-step). A caught era boundary is the success criterion, not a
failure of the "once" goal.

No partner numbers enter any computation (A1.8 firewall inherited); every claim is
computed from our own caches/artifacts and reported honestly, negative results
included.

## 2. Scope (in / out)

IN: D1 once-calibrated + drift-triggered recalibration (a retrospective, day-granular
SIMULATION on the recorded days, with two label-free drift sentinels and a driver
script); D2 named states end to end (optional snapshot member + monitor/analyze
surfacing, reusing the existing majority-vote primitive); D3 transition & dwell
handling (a SCADA transition taxonomy, a `min_dwell` sweep to ground the default, and
monitor-level `near_transition` visibility with an optional suppression ablation).

OUT: online/streaming operation (this stays a batch replay over recorded files —
package-6's deployment model is unchanged); any localization (partner's paper
territory); importing partner numeric constants; new encoders, scorers, or clusterers;
retraining or refitting the detector at monitor time; a persistent recalibrated-baseline
state machine (D1 is deliberately day-granular, see D1 honesty).

## 3. Design decisions

### D1 — Once-calibrated operation with drift-triggered recalibration (simulation)

**Framing (binding honesty).** A retrospective, day-granular SIMULATION on the recorded
days — NOT an online detector. The commissioning artifact is the **pool-B1 snapshot**
(the four `010726` runs, era B; P7 §setup), fit once via
`run_step2 --protocol cross-day-pooled --fit-runs <B1> --save-snapshot` for each of the
three FAR-scored representations (`fusion`, `vibration`, `audio-beats`). It is then held
FROZEN while the driver replays the recorded days in chronological order and, per day,
decides label-free whether to stay frozen or recalibrate.

**Two label-free drift sentinels**, both evaluated per monitored day, both with thresholds
derived from the commissioning (B1) pool ONLY:

- **s1 — mode-bank rejection rate (`no_mode_fits`).** Reuse the P8 bank
  (`rowii.state.modebank.ModeBank.fit`/`assign`; runner `scripts/run_modebank.py`, which
  already emits `no_mode_fits_rate`). Canonical sentinel bank = **audio-beats**: P8 found
  it is the era-drift-robust representation whose columns are never refused by the
  contract guard across eras (README §package-8 D1: era-C zero-shot `no_mode_fits` = 19.5%
  computable where fusion/vibration columns are refused). At commissioning, fit the bank
  on B1's SCADA-labelled windows and record a fit-day BASELINE rate; on each monitored day
  compute `assign(...).no_mode_fits` label-free and fire s1 when the day rate exceeds
  `baseline + margin`. Baseline + margin are BOTH measured from B1 alone: baseline = the
  pooled `no_mode_fits` rate over B1's held-out (conformal-side) windows; margin = the
  upper edge of a `segment_ids`-block bootstrap band of that rate on B1 (the same
  bootstrap-block rule P8 D3 uses: the 12-min recording segment, never wall-clock). MUST
  surface `ModeBank.low_confidence_modes` alongside the rate: a low-confidence member's
  `+inf` threshold makes `no_mode_fits` UNDER-fire (modebank.py `assign` caveat, T2
  finding), so a low rate is only trustworthy when `low_confidence_modes` is empty. Tag:
  s1 is SUPERVISED at commissioning (SCADA modes), label-free at runtime.

- **s2 — per-stream level-step sentinel.** Operationalize the P8 era-step logic
  (`scripts/analyze_days.py::_levels_by_stream`/`_era_step_row`,
  `rowii.anomaly.levelrecal.level_columns`/`column_medians`). Per monitored day compute the
  median level of the **microphone** streams' level columns (`RAWGeneratorMic__0`/
  `RAWTurbineMic__1`, via `rowii.pipeline.stream_columns` ∩ `level_columns`, averaged per
  window then medianed) from the **raw** `audio` cache, and compare to the B1 anchor median.
  Fire s2 when the shift, converted to dB with the P8 per-family factor
  (`analyze_days._level_db_factor`: ×20 for `_log_rms`, ×10 for `_band_`/`_octave_`),
  leaves the B1 within-era band. The band (the margin) is derived from B1 ONLY — the
  per-run/per-`segment_ids` scatter of the mic-level median within the commissioning pool
  plus a fixed log10-domain headroom; NO partner dB figure and NO cross-era day set the
  threshold (A1.8). **Vibration-flat cross-check:** compute the same for the vibration
  streams (`RAWGeneratorVib__2`/`RAWTurbineVib__3`) from the raw `vibration` cache; if mic
  is OUTSIDE its band while vibration is WITHIN its band, attribute the change to
  instrumentation/era (P8's signature: microphones step at 2026-06-29, vibration flat) and
  still trigger — the cross-check labels the trigger's CAUSE (instrumentation vs machine),
  it does not veto it.

  s2 is a DAY-level instrumentation signal computed from raw single-stream caches, so it is
  representation-agnostic: fusion features are per-run z-scores (`fuse()`, A1.1 —
  `analyze_days` era-step refuses fusion for exactly this reason and reads one run of EACH
  raw variant), so s2 for the fusion FAR arm is read from the raw audio/vibration caches,
  never from fusion's stored columns. The SAME day-level trigger verdict (s1 ∨ s2) gates
  the frozen/recalibrate choice for all three FAR arms.

**Deliverable — `scripts/run_once_calibrated.py`.** Replays the days chronologically —
`250526` (era A) → `290626` (era B) → `010726` (era B) → `080726` (era C) — for
`fusion` + `vibration` + `audio-beats`, and additionally evaluates the sentinels on
`270626` (era A) as a SENTINEL-ONLY row (no SCADA, so no FAR/ARI — a label-free
corroboration of the era-A trigger). Per (day, representation) it runs `monitor.py` in
`--thresholds frozen` and `--thresholds recalibrate` (default), reads each realized
aggregate FAR from the produced `alarms.parquet` (mean `alarm` over `role == "scored"`
windows), and reports three regimes:

  (i) **always-frozen** — the frozen FAR;
  (ii) **always-recalibrate** — the recalibrate FAR (the P7/P8 reference recipe);
  (iii) **once+triggered** — frozen FAR if neither sentinel fired that day, else the
       recalibrate FAR;

plus a **trigger log** (per day: s1 rate vs baseline+margin and `low_confidence_modes`;
s2 mic/vib dB shift vs band and the instrumentation-vs-machine attribution; the resulting
frozen/recalibrate decision) that answers explicitly **whether the 2026-06-29 era boundary
is caught** — i.e. whether a sentinel fires on the era-A days (`250526`, `270626`) monitored
with the era-B snapshot, and whether era-C `080726` triggers.

For `080726` (the induced-strike day, era C) the FAR is the event-free window-FAR:
`monitor.py --exclude-calibration-events docs/groundtruth/080726_events_{pu,st}.csv`
(P7 pillar-3 rule, A2.3.3), and the driver additionally reports the pillar-3 TPR RETAINED
under the once+triggered regime — the story that ties D1 to the headline: the sentinel
firing on `080726` → recalibrate → strikes remain detectable, whereas staying frozen
across the era gives the trivially-broken cross-era snapshot (P7: window-FAR 0.62/0.96,
useless).

**Honesty (binding).** (1) Retrospective, day-granular, no online claim — the chronological
order drives WHEN sentinels fire in the report, not a causal runtime decision; sentinels
are label-free at runtime, thresholds are fit-day-only. (2) The "once" claim is scoped to
"once per instrumentation era": a caught era boundary IS success. (3) `010726` is IN the
B1 fit pool — monitoring it frozen/once is IN-SAMPLE and is tagged non-held-out (mirrors
P7's tainted `audio-student` row); `290626` is the clean held-out same-cfg day. (4) The s2
margin is estimated from the single commissioning era-B day family (B1) and may
under-estimate true within-era day-to-day drift; s1 corroboration and the vib-flat
cross-check guard against a spurious mic-only trigger. (5) On `080726` s1 (the P8 19.5%
`no_mode_fits`, plus the vibration column-count drift the projection/contract guard already
logs) is the PRIMARY era-C trigger; a mic-level step on era C is not asserted (P8 did not
measure one).

### D2 — Named states end to end

**(a) Clusterer path — a commissioning-time name map, persisted as an optional snapshot
member.** The map `cluster id → mode name` is the majority vote of SCADA GT modes over the
FIT-side windows of each fit-day cluster. This primitive ALREADY EXISTS:
`rowii.eval.metrics._majority_mapping` (used by `scripts/apply_detector.py`'s `mapped_mode`
reporting column). D2 reuses it — no new mapping logic. Ties / no clear GT majority fall
back to the bare id-derived name `cluster-<id>` (English, per the repo's English-only
artifact rule and because the SCADA state strings — `standstill`/`turbine`/`pump`/
`phase-shifter` — are already English; `labels.STATES`).

Persist as a NEW optional format-v2 member `state_names: dict[int, str] | None` on
`MonitorSnapshot`, following the `level_recal_medians` pattern EXACTLY
(`rowii.runtime.snapshot`): it is a small dict, so it lives entirely in the `meta` JSON
(no npz array member), present only when a snapshot carries names; `load_snapshot`
reconstructs it (int-keyed, JSON keys stringified like `thresholds`); `save_snapshot`
round-trips it. Keyed over ALL fit-day cluster ids (the `fitted_ids` id space), NOT the
threshold-label subset — a cluster can carry a name for the timeline even if it has no
alarming threshold, so `state_names` is orthogonal to `references`/`thresholds` and their
key-agreement invariant does not apply to it. **Mutual-exclusivity: NONE** — `state_names`
is a naming layer, not a scoring-space transform, so it coexists with `session_stats` OR
`level_recal_medians` freely (unlike those two, which remain mutually exclusive with each
other). `fit_snapshot_from_parts` gains a `state_names` kwarg; `run_step2 --save-snapshot`
computes it from the pooled fit-side GT labels via `_majority_mapping` and passes it.

**(b) Bank path — names are native.** `ModeBank.assign` already emits mode-string labels
(`turbine`/`pump`/…); no map is needed. D2's surfacing just uses those strings directly.

**(c) Surfacing.** When `state_names` is present, `monitor.py` restores a `mapped_mode`
column to `segments.csv` (the `apply_detector.py` convention it deliberately dropped for
lack of a mapping — snapshot.py/monitor.py docstrings), names states in `timeline.md`
(`state 2 (turbine)`) and the `monitor_notes.md` per-state table, and appends an optional
`state_name` column to `alarms.parquet` (at the END of `_ALARM_COLUMNS` — backward
compatible; `eval_events`/`evaluate_events` select by name and ignore the addition, the
established precedent). Absent `state_names`, every output falls back to bare ids exactly
as today. `analyze_days` figures already label by GT mode strings (era-step, mode-signatures);
the only addition is that any detected-cluster axis uses the snapshot's names when a
snapshot is in scope. **Strictly no GT at monitor time** — the map is fitted once at
snapshot BUILD (commissioning), never re-derived on the monitored day.

### D3 — Transition & dwell handling

**(a) SCADA transition taxonomy — an `analyze_days` subcommand.** A new `transitions`
subcommand characterizing OUR OWN `gt_labels` transition windows (`rowii.scada.labels`:
the `"transition"` state from `_apply_ramp` + `_apply_transition_buffer`) across the
GT-bearing days (`250526`/`290626`/`010726`, + `080726`'s changeover). Per transition CLASS
— the `(from_state → to_state)` pair of KNOWN states bracketing each contiguous transition
run — report: count of transition segments, dwell stats (run duration in seconds), and ramp
stats (the `_apply_ramp` centered `dP/dt` over the run, from the SCADA power channel). This
is a descriptive, our-own analysis type; per the A1.8 documentation discipline the
subcommand's docstring and the digest state the analysis-type provenance and NO partner
numeric constant appears as an expected value. Output follows the other `analyze_days`
subcommands (per-figure PNG + underlying CSV + a digest paragraph).

**(b) `min_dwell` sweep — grounding the default in data.** `DetectConfig.min_dwell_s`
default is 5.0 (`config.py`); via `FittedDetector._finish`'s
`min_dwell = max(1, round(min_dwell_s / window_s))` this is 5 windows at `window_s = 1.0`,
and `duration_filter` (`rowii.state.segments`) merges every detected run shorter than that
into a neighbour — this is precisely the mechanism that removes Stefan's "1-second modes"
at Step-1. A small driver `scripts/sweep_min_dwell.py` sweeps `min_dwell_s ∈ {5, 10, 20}` s
(→ 5/10/20 windows) and reports Step-1 `state_ari` vs GT (`rowii.eval.metrics`'
majority-mapped `state_ari`, the SAME metric as the P7 k-selection, so results are directly
comparable) over the six P7/P8 held-out rotations, **detector arm only** — the KMeans+HMM+
`duration_filter` clusterer. The bank arm is dwell-free by construction (`ModeBank` applies
no duration filter) EXCEPT under `--smooth`, which is the duration-filter ONLY
(P8 A1.3: `--smooth` = `min_dwell` duration filter, never the EM smoother); the sweep
therefore does not touch the bank. Plus one **Step-2 chain FAR spot-check** at the three
`min_dwell` values (one rotation, e.g. B1→`290626-tu` fusion recalibrate) so the default is
grounded in downstream FAR, not just Step-1 ARI. The sweep constructs `Config`s with each
`min_dwell_s`, refits the pooled detector, and scores — self-contained; no `DetectConfig`
default is changed unless the data argues for it (reported either way).

**(c) Monitor transition visibility + optional suppression ablation.** `monitor.py` gains a
`near_transition` boolean column on `alarms.parquet` (appended at the END of
`_ALARM_COLUMNS`, backward compatible): True for every valid window within ±W seconds of a
detected-state CHANGE (a boundary in the decoded `labels` array among valid windows),
`W` default = the snapshot's `min_dwell_s` (the same dwell scale D3b sweeps). An OPTIONAL
`--suppress-transition-alarms` flag (default OFF) then withholds alarms on `near_transition`
windows (`alarm → False` there), reporting in `monitor_notes.md` the suppressed-alarm count;
`score`/`p_value`/`near_transition` stay in the parquet so the pre-suppression alarm is
always recoverable (full audit trail). Its effect is reported as an ABLATION: the risk case
is **strikes near the changeover** — hammer strikes do NOT flip the detected state (sticky
HMM `self_transition = 0.98` + `duration_filter` remove any sub-`min_dwell` flip), so
`near_transition` windows sit at REAL operational boundaries (ramps), and the ONLY strike-loss
risk is strikes coinciding with the `080726` pump→phase-shifter changeover (~13:05 UTC, P7).
The ablation re-runs `eval_events` on `080726` with and without suppression and reports the
pillar-3 TPR delta (did any of the 13 pump/standstill strikes get lost?). Suppression is
explicitly an OPERATOR choice, default OFF; the package makes the trade-off visible, it does
not adopt suppression.

## 4. Evaluation & honesty rules

- **Inherited from P7/P8 unchanged:** segment-blocked calibration/scoring splits, the
  pool-member/leakage bans, event-free calibration on the strike day
  (`--exclude-calibration-events`, A2.3.3), the A1.8 firewall (no partner-derived numeric
  constant as an expected value anywhere in `src`/`scripts`/tests; all numbers computed from
  our caches), supervised/unsupervised tags on every cell where the bank is used (D1 s1, D2b),
  and artifact-verified synthesis (every reported number traceable to a committed artifact;
  negative results reported plainly).
- **ARI/masking:** D3b's Step-1 `state_ari` uses the same majority-mapped metric and mask
  convention as the P7 k-selection, so the `min_dwell` sweep is read alongside the k-sweep.
- **New — sentinel thresholds are fit-day-only (D1).** Both s1's `baseline + margin` and s2's
  within-era band are derived from the commissioning (B1) pool ALONE; no monitored day and no
  partner figure sets a threshold. Sentinels are label-free at runtime.
- **New — "once-calibrated" is scoped to "per instrumentation era".** A caught era boundary
  is the success criterion; the report states which days triggered and why. The `010726`
  in-sample tag and the `080726` event-free-FAR rule are stated on every affected cell.

## 5. Testing

House rules: pytest with `-m "not data"` for unit tests (no real data), tests written first,
`ruff` + `mypy --strict` clean, every implementer verification a temp-file `&&` chain,
deterministic seeded fixtures, CLI-level tests via the established monkeypatch seams, and NO
partner numeric as an expected value. Per decision:

- **D1 sentinels:** unit tests for s1 (rate vs baseline+margin decision; `low_confidence_modes`
  surfaced with the rate; under-fire caveat asserted) and s2 (mic/vib median math, dB-factor
  conversion via `_level_db_factor`, band membership, the mic-out/vib-in → instrumentation
  attribution, fusion reads raw caches not fused columns). Driver: pure-helper tests (regime
  selection frozen-vs-recalibrate-vs-triggered; trigger-log assembly; the `270626` sentinel-only
  path) on synthetic per-day FAR/verdict inputs — no real monitor run in unit tests.
- **D2 snapshot member:** `state_names` save/load round-trip (int keys, `meta`-only, absent →
  `None`); `fit_snapshot_from_parts` kwarg + geometry (keys need NOT match the threshold-label
  set — the orthogonality); mutual-exclusivity is NOT enforced against `session_stats`/
  `level_recal_medians` (assert it coexists). Monitor surfacing: `mapped_mode` in segments,
  named timeline/notes, `state_name` appended to alarms; fallback to ids when absent.
- **D3:** `transitions` taxonomy shape tests on synthetic `gt_labels` sequences (class keys,
  dwell/ramp stats); `min_dwell` sweep helper tests (window-count conversion 5/10/20,
  `duration_filter` no-op at `min_dwell ≤ 1`); `near_transition` mask math (±W boundary
  windows, `W = min_dwell_s`) and `--suppress-transition-alarms` (alarms flip to False on
  near-transition windows only, count reported, parquet audit columns retained).

## 6. Execution order & model policy

1. **D2 snapshot member** (`state_names`): `rowii.runtime.snapshot` v2 optional member +
   `fit_snapshot_from_parts` kwarg + `run_step2 --save-snapshot` wiring via `_majority_mapping`;
   tests. (Cheap, unblocks the named snapshots D1's driver consumes.)
2. **D2 surfacing** in `monitor.py` (segments `mapped_mode`, named timeline/notes, `state_name`
   on alarms); tests.
3. **D3b `min_dwell` sweep** + **D3a `transitions` taxonomy** (`scripts/sweep_min_dwell.py`,
   `analyze_days transitions`) — grounds the dwell default in data before it is used elsewhere.
4. **D3c** monitor `near_transition` column + `--suppress-transition-alarms` ablation; tests.
5. **D1 drift sentinels** (`rowii.runtime.drift`: s1 no_mode_fits sentinel, s2 level-step
   sentinel + vib cross-check) reusing `ModeBank`, `levelrecal`, `analyze_days._level_db_factor`;
   tests.
6. **D1 driver** (`scripts/run_once_calibrated.py`): build the three B1 snapshots, replay the
   days, three FAR regimes + trigger log + `080726` event-free FAR & pillar-3 retention; tests.
7. **Synthesis:** README package-9 section (once-calibrated FAR-regime table + trigger log,
   named-state example, transition taxonomy + `min_dwell` verdict, suppression ablation) +
   master-thesis note (figures inline) + memory update + final whole-branch review → PR #12 →
   merge.

Model policy for subagents (Stefan, P8): implementation/tests/readers on **sonnet**;
adversarial per-task and whole-branch/spec reviews on **opus**; **fable** only if a review
blocks twice. Implementer dispatches MUST forbid Agent-tool use (no self-orchestration /
branch races).

**Scope note.** Small package: six code tasks (steps 1–6) + synthesis. The real new cost is
D1 (two sentinels + the replay driver); D2 rides an existing mapping primitive and the
established optional-snapshot-member pattern; D3 is one monitor column + two analysis drivers.
