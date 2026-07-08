# rowii-monitor

Acoustic + vibration condition monitoring for the Rodundwerk II pump-turbine
(HSG master thesis implementation).

Step 1 of this repo detects the operating state of the machine per 1-second
window from audio and vibration recordings, unsupervised, and validates the
detected states against SCADA-derived ground truth. Results are reported per
input modality (audio-only / vibration-only / fusion), audio featurizer
(handcrafted / frozen BEATs embeddings), and clusterer (KMeans / GMM).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional BEATs-embedding support (adds `torch`/`torchaudio`):

```bash
pip install -e ".[beats]"
```

Copy `.env.example` to `.env` and adjust the paths for your machine:

```bash
cp .env.example .env
```

## Data layout

Sensor data is never committed to this repo. `ROWII_DATA_ROOT` (env var or
`.env`) points at a local **parent root** containing one subdirectory per
measurement day (`illwerke-<dayid>/`), each itself a full day tree:

```
<ROWII_DATA_ROOT>/
├── illwerke-250526/
│   └── 20260626 Messung/
│       ├── TU/                     # 48 files, 25 GB — turbine run, 2026-06-25 04:15-06:27 UTC,
│       │                           #   12-min segments x 4 streams
│       ├── PU/                     # 21 files, 9.6 GB — pump runs, 2026-06-25 morning + afternoon
│       └── Betriebsdaten/          # hourly SCADA, 2026-06-25 00:00-11:00, ~30 channels @ 10 Hz
├── illwerke-270626/
│   └── 20260627 Messung/
│       └── PU_PH_PU_PH_PU_PH/      # ~4h alternating pump / phase-shifter; NO Betriebsdaten
├── illwerke-290626/
│   └── 20260629 Messung/           # TU (incl. a ~37-min phase-shifter hold), PU, full-day SCADA
└── illwerke-010726/
    └── 20260701 Messung/           # PU, TU1, TU2, TU_PH_TU (all 4 operating modes), full-day SCADA
```

`rowii.io.dataset.discover` also accepts a single day tree directly (e.g.
`ROWII_DATA_ROOT=.../illwerke-250526`) for backward compatibility — run names
then have no day-id prefix, matching the pre-multi-day behaviour exactly. Under
the parent-root layout above, every discovered run is prefixed with its own
day's 6-digit id (`250526-tu`, `010726-tu_ph_tu`, `270626-pu_ph_pu_ph_pu_ph`,
...); each run's SCADA ground truth is scoped to its own day tree only (a run
never sees a different day's Betriebsdaten).

Each burst contributes four streams: `RAWGeneratorMic__0` / `RAWTurbineMic__1`
(~50 kHz microphone rings) and `RAWGeneratorVib__2` / `RAWTurbineVib__3`
(~10 kHz tri-axial accelerometers). All files are Gantner UDBF v1.07 binaries.
Session folder names (`TU`, `PU`, `TU_PH_TU`, ...) are operator hints only —
ground truth always comes from SCADA (or `unknown` on the one day with none).

## Quickstart

Copy the required subset of a source tree into `ROWII_DATA_ROOT`:

```bash
python scripts/copy_data.py --source ~/Downloads/illwerke-250526-analysis
```

Run state detection for one recording / input variant / clusterer (run names
are day-prefixed under a parent `ROWII_DATA_ROOT`, see "Data layout" above):

```bash
python scripts/run_step1.py --run 250526-tu --variant audio --clusterer kmeans
python scripts/run_step1.py --run 250526-tu --variant fusion --clusterer kmeans
```

`audio-beats` / `fusion-beats` need the `[beats]` extra installed and
`ROWII_BEATS_CHECKPOINT` set to a BEATs `.pt` checkpoint (e.g.
`BEATs_iter3_plus_AS2M.pt`):

```bash
python scripts/run_step1.py --run 250526-tu --variant audio-beats --clusterer kmeans
python scripts/run_step1.py --run 250526-tu --variant fusion-beats --clusterer kmeans
```

Run the full grid (all recordings x all variants x both clusterers):

```bash
python scripts/run_step1.py --run all --variant all --clusterer all
```

Results are written to `results/<run>/<variant>-<clusterer>/` (`segments.csv`,
`frame_labels.parquet`, `report.md`, `timeline.png`), with an aggregate
`results/summary.csv` across all executed combinations.

## Development

```bash
pytest tests/ -q            # unit tests (no real data required)
pytest -m data -v           # real-data smoke tests (needs ROWII_DATA_ROOT)
ruff check .
mypy src scripts
```

## Step-1 grid results (TU + PU-morning + PU-afternoon, 2026-06-25)

The complete Step-1 grid ran against the real June-25 Rodundwerk II delivery
(35 GB: 48 TU + 8 PU-morning + 13 PU-afternoon burst files across 4 streams,
12 Betriebsdaten hours): all three recordings x {audio, vibration, fusion} x
{KMeans, GMM}, plus TU and PU-morning x {audio-beats, fusion-beats} x both
clusterers, plus the TU fusion KMeans k-sweep — `results/summary.csv` was
deleted and fully regenerated from scratch (30 rows: 26 combinations + 4
k-sweep rows, each combination exactly once). All KMeans numbers reproduce
the earlier Task-13/13b/14 values bit-for-bit (fixed `random_seed = 7`).
Parameter verification (`scripts/verify_parameters.py`,
`results/parameter_verification.md`) measured every machine-parameter
hypothesis directly from this data rather than carrying over pre-delivery
guesses. **Measured nominal speed: 378.832 rpm** (`GT_CHANNELS["speed"]` =
`"1_Drehzahl UPM"`, the genuine rpm channel -- Task 13 originally measured
this off the wrong channel, `"1_Drehzahl_Ist"`, and got ~101 rpm; see
`results/parameter_verification.md`'s Revision 2026-07-07 section for the
full derivation). The three `MACHINE_HZ` spectral centres were confirmed
as-is, and one plant-specific sign convention had a genuine bug (pump-mode
speed sign, fixed in `is_nominal`).

**State-level metrics** (primary view: each cluster maps independently to
its majority GT state, so legitimate load-level sub-clusters within one
operating mode are not penalized as confusion -- see
`rowii.eval.metrics` module docstring). TU has 8275 eval windows
(122 standstill / 403 transition / 7750 turbine):

| run | variant | clusterer | state ARI | state accuracy | state macro-F1 |
|---|---|---|---|---|---|
| tu | audio | kmeans | 0.684 | 0.960 | 0.721 |
| tu | audio | gmm | 0.691 | 0.954 | 0.509 |
| tu | vibration | kmeans | 0.153 | 0.942 | 0.509 |
| tu | vibration | gmm | 0.683 | 0.955 | 0.501 |
| tu | fusion | kmeans | 0.687 | 0.959 | 0.705 |
| tu | fusion | gmm | 0.704 | 0.956 | 0.513 |
| tu | audio-beats | kmeans | 0.000 | 0.937 | 0.322 |
| tu | audio-beats | gmm | 0.000 | 0.937 | 0.322 |
| tu | fusion-beats | kmeans | 0.000 | 0.937 | 0.322 |
| tu | fusion-beats | gmm | 0.000 | 0.937 | 0.322 |
| pu-morning | *all five variants* | kmeans & gmm | 1.000 | 1.000 | 1.000 |

PU-morning's 1.000 rows are a degenerate case, not a detector triumph: this
recording never leaves the pump state (all 719-1439 eval windows are GT
`"pump"`), so every cluster's majority vote trivially resolves to
`"pump"` and every state-level metric collapses to its
identical-partition value by convention -- for every variant and BOTH
clusterers alike. The strict-metrics view shows the same runs'
`ARI = 0.000` (single-GT-class ARI is degenerate), which is the more honest
read of PU-morning's actual information content. Note also that TU state
accuracy has a floor of 0.937 from turbine prevalence alone (an
always-turbine labeling scores 7750/8275 = 0.937 -- exactly what all four
BEATs rows land on), so accuracy is nearly uninformative here; state ARI
and state macro-F1 carry the signal.

**GMM vs. KMeans.** GMM edges out KMeans on TU state ARI for every
handcrafted variant (audio 0.691 vs. 0.684, vibration 0.683 vs. 0.153,
fusion 0.704 vs. 0.687 -- the vibration jump is the single largest change
anywhere in the grid), but the mechanism deserves suspicion before
celebration: **no TU GMM run allocates any majority-standstill cluster.**
In all three handcrafted GMM runs, all 122 standstill windows fall into the
cluster whose majority is `transition`, so state-level standstill recall is
0/122 and standstill F1 is 0 -- which is why every TU GMM row's state
macro-F1 sits at ~0.50-0.51 while KMeans audio/fusion reach 0.72/0.70.
GMM's higher ARI reflects a cleaner two-way turbine vs. non-turbine split
(its small mixed cluster absorbs most standstill+transition mass together),
not a better three-mode recovery. KMeans remains the better operating choice
when catching standstill matters (audio-kmeans: 118/122 in the strict view);
GMM's vibration result mostly shows that vibration's KMeans weakness was
partly a clusterer artifact, not purely a modality limit. GMM also does
nothing for BEATs: all four TU beats x gmm rows keep state ARI at exactly
0.000 with the same degenerate all-turbine majority mapping as kmeans.

**Strict (1:1 Hungarian) metrics** -- secondary, kept for continuity with
Task 13's original numbers and as an over-segmentation diagnostic (a large
state-level vs. strict gap means the detector's extra clusters are
sub-modes, not confusion):

| run | variant | clusterer | k | ARI | macro-F1 | boundary \|Δt\| (s) | silhouette |
|---|---|---|---|---|---|---|---|
| tu | audio | kmeans | 4 | 0.144 | 0.721 | 40 | 0.470 |
| tu | audio | gmm | 4 | 0.073 | 0.451 | 56.5 | 0.295 |
| tu | vibration | kmeans | 4 | 0.128 | 0.658 | 135 | 0.316 |
| tu | vibration | gmm | 4 | 0.125 | 0.444 | 43 | 0.161 |
| tu | fusion | kmeans | 4 | 0.153 | 0.705 | 58 | 0.399 |
| tu | fusion | gmm | 4 | 0.072 | 0.455 | 57 | 0.219 |
| tu | audio-beats | kmeans | 4 | 0.064 | 0.430 | 39.5 | 0.162 |
| tu | audio-beats | gmm | 4 | 0.012 | 0.312 | 129.5 | 0.148 |
| tu | fusion-beats | kmeans | 4 | 0.007 | 0.333 | 37 | 0.171 |
| tu | fusion-beats | gmm | 4 | 0.028 | 0.348 | 186.5 | 0.169 |

On the strict view GMM is a regression nearly across the board (audio 0.073
vs. 0.144, fusion 0.072 vs. 0.153), with one bright spot: vibration-gmm's
boundary deviation (43 s) is a third of vibration-kmeans' 135 s.
PU-morning's strict rows are uniformly degenerate for both clusterers
(ARI 0.000, macro-F1 1.000, boundary None, silhouettes 0.011-0.045).

**PU-afternoon: processed, but excluded from GT metrics.** The afternoon
pump recording's streams span 13:44-14:40 UTC while the delivered
Betriebsdaten hours end at 12:00 UTC -- zero SCADA overlap, so all six
pu-afternoon combinations (audio/vibration/fusion x kmeans/gmm; 1439
windows for audio/fusion grids, 2159 for the longer vibration-only grid)
ran the full detection pipeline and then took the documented reduced-report
path: `report.md` + `segments.csv` + `frame_labels.parquet` written,
`summary.csv` rows carry `notes = "no SCADA coverage"` with every GT metric
empty (`n_eval = 0`). This is exactly the spec's "processed but excluded
from GT metrics" contract, now evidenced end-to-end on real data.

TU fusion k-sweep (KMeans; state ARI the headline column, strict ARI/macro-F1
alongside for comparison; full detail per k in
`results/tu/fusion-kmeans-k<k>/report.md`):

| k | state ARI | strict ARI | strict macro-F1 | silhouette |
|---|---|---|---|---|
| 3 | 0.707 | 0.154 | 0.456 | 0.393 |
| 4 | 0.687 | 0.153 | 0.705 | 0.399 |
| 5 | 0.419 | 0.148 | 0.631 | 0.410 |
| 6 | 0.710 | 0.076 | 0.686 | 0.245 |

**Load-alignment verdict.** `rowii.eval.metrics.load_alignment` cross-tabs
predicted cluster id against SCADA `load_bin` on each run's turbine (TU) or
pump (PU-morning) eval windows -- "do sub-clusters track load levels?" is
answered *partially yes*, and this is where GMM genuinely helps. TU
ARI(load_bin, cluster), kmeans / gmm: audio 0.457 / 0.577, vibration
0.514 / 0.508, fusion 0.465 / 0.576, audio-beats 0.567 / 0.628,
fusion-beats 0.620 / 0.370. PU-morning: audio 0.259 / 0.259, vibration
0.204 / 0.155, fusion 0.259 / 0.259, audio-beats 0.155 / 0.121,
fusion-beats 0.160 / 0.140 (audio/fusion GMM converged to the same
partition as their KMeans init there). These are well above 0 (genuine,
non-random structure) but well below 1 -- the detector's turbine-phase
sub-clusters correlate with load level more than chance but do not cleanly
separate it. The grid's best load alignment is audio-beats-gmm at 0.628;
within KMeans both beats variants still beat every handcrafted variant on
TU, but the picture is no longer uniform once GMM enters: handcrafted
audio/fusion GMM (0.577/0.576) overtake audio-beats-kmeans, and
fusion-beats-gmm collapses to the grid's worst TU value (0.370).

**Does fusion clear 0.9 on state-level ARI?** No, on any run in the
"real detector recovering real structure" sense: TU fusion tops out at
0.704 (GMM, k=4) and 0.710 (KMeans k-sweep, k=6), both short of 0.9;
PU-morning's 1.000 is the degenerate single-GT-class case described above,
not a genuine 3-state recovery; PU-afternoon has no GT to evaluate against
at all.

**Honest reading.** State-level metrics change the story dramatically from
Task 13's strict-only view: TU fusion's strict ARI (0.153) looked like
near-total failure, but its state ARI (0.687) shows the detector recovers
the correct mode most of the time once load-level sub-clusters are
credited instead of penalized -- confirming Task 13's own qualitative
read (the timeline visually tracked the SCADA power curve) was closer to
the truth than the strict ARI number suggested. Audio and fusion remain
close on TU (state ARI 0.684 vs. 0.687 with KMeans; 0.691 vs. 0.704 with
GMM); vibration lags badly only under KMeans (0.153), and mostly for
clusterer rather than modality reasons (see the GMM paragraph above). The
standstill-recall weakness persists as the grid's dominant failure mode:
audio-kmeans catches 118/122 standstill windows, vibration/fusion-kmeans
catch 47/122, every GMM run catches 0/122 at the state level, and every
BEATs run's majority mapping never labels standstill at all. PU-morning's
perfect-looking numbers are an artifact of this delivery containing no
non-pump SCADA-covered windows in that run -- the load-alignment ARIs
(0.12-0.26) are PU-morning's only non-degenerate signal. Neither run
clears the ARI >= 0.9 acceptance gate in any metric family that isn't
degenerate.

**Does frozen-BEATs beat handcrafted audio?** No -- on every headline
metric, with either clusterer, it is worse. TU audio-beats' state ARI is
0.000 (both clusterers) versus handcrafted audio's 0.684/0.691, and its
strict ARI (0.064 kmeans, 0.012 gmm) and macro-F1 (0.430/0.312) trail
handcrafted audio (0.144/0.073, 0.721/0.451) too. All four TU beats rows
share the identical degenerate signature: every cluster's majority is
turbine, so state accuracy pins to the 0.937 always-turbine floor and
state macro-F1 to 0.322 (turbine F1 alone). The mechanism is visible in
the strict confusion matrices: audio-beats alone catches **0/122**
standstill windows -- KMeans and GMM alike find no separating structure
between "machine off" and "machine ramping" in raw BEATs embeddings. This
is consistent with BEATs' AudioSet pretraining objective: it was never
trained to represent "silence vs. quiet machinery," a distinction
handcrafted `log_rms` and machine-frequency band energies capture directly
by construction.

**Does fusion-beats fix the standstill weakness?** Only in the strict
(Hungarian) view, and the state-level view now reveals how thin that
result is: fusion-beats-kmeans' Hungarian-forced standstill cluster does
contain 118/122 standstill windows (matching handcrafted audio's row), but
that same cluster also contains 1921 turbine windows -- its majority is
turbine, so the state-level mapping labels nothing standstill and state
ARI is exactly 0.000. The earlier reading stands with sharper wording:
z-scored concatenation with BEATs changes what KMeans finds separable in
the fused space enough to concentrate standstill windows into one cluster,
but not enough to make that cluster standstill-dominated. Confirming a
precise mechanism would need per-branch ablations inside the fused feature
matrix, out of scope here.

BEATs' one genuine advantage across every metric measured here remains
load-alignment ARI (see verdict above): audio-beats-gmm posts the grid's
best TU value (0.628), though fusion-beats-gmm simultaneously posts its
worst (0.370), so the advantage is variant-dependent rather than uniform.
Runtime: each beats combo loads one frozen BEATs model (90.3M params) per
mic stream; on Apple Silicon MPS (`best_device()` default, no
`ROWII_FORCE_CPU` fallback needed) the two-combo steps (kmeans + gmm, one
full feature extraction each) took 7m12s (TU audio-beats), 9m49s (TU
fusion-beats), 46s (PU-morning audio-beats), and 59s (PU-morning
fusion-beats).

Genuine bugs surfaced and were fixed while getting real data through the
pipeline for the first time (see commit history, one fix per commit):

1. **Gantner reader**: the channel-descriptor token length prefix never
   actually matched real bytes -- it counts the payload PLUS its NUL
   terminator, not the payload alone, and a naive fix reintroduced a
   short-token ambiguity that silently dropped whole channels. Blocked
   `read_header`/`read_gantner` on every real file, TU and Betriebsdaten
   alike.
2. **`_extract_stream_features`**: real DAQ clocks jitter by ±1
   sample/window with no actual data gap; requiring an exact sample-count
   match to featurize a window left ~33% of real TU windows NaN, blowing
   through the 5%-invalid hard-fail guard before any variant could run.
3. **`zscore`**: a real run always has a handful of invalid (NaN) windows;
   plain `.std()` propagates NaN into a column's statistics, and the old
   zero-std guard (`std >= 1e-12`) is always False for NaN, silently
   zeroing the ENTIRE column -- not just the NaN row. Every stream in the
   TU run had at least one invalid window, so this zeroed all 231 fused
   columns and collapsed fusion's KMeans to a single cluster (ARI exactly
   0.0 before the fix).
4. **`is_nominal`**: pump-mode `"1_Drehzahl UPM"` (`GT_CHANNELS["speed"]`)
   is signed negative at this plant (opposite rotation direction from
   turbine mode), but the nominal-speed gate compared signed speed against
   a positive threshold, so it silently classified the entire pump run as
   "transition" regardless of `speed_nominal_rpm`.
5. **Speed channel** (Task 13b): `GT_CHANNELS["speed"]` was wired to
   `"1_Drehzahl_Ist"`, a channel that is NOT rpm (a percent-of-nominal-ish
   quantity, ~3.75x smaller than the genuine rpm channel on the same
   file) -- every downstream `_base_state` rule is dimensionless (fractions
   of whatever `speed_nominal_rpm` is configured as), so the pipeline ran
   end-to-end throughout Task 13 without any test catching the mismatch.
   Corrected to `"1_Drehzahl UPM"`; `speed_nominal_rpm` remeasured as
   378.832 rpm (was 101.0).

## Multi-day results (2026-07-08)

The full Illwerke campaign re-release (four day trees, see "Data layout")
went through the pipeline with the phase-shifter GT state enabled
(`ph_min_dwell_s = 600 s`, `ph_power_eps_mw = 5.0` measured from the real
2026-07-01 phase-shifter interval's stable ~-3.5 MW idling draw, and the
`1_KS Stellung` gate enabled after independent verification -- see
`results/parameter_verification.md`, "Phase-shifter channels, 2026-07-08":
phase-shifter/standstill KS median 3.2/3.0 vs. turbine/pump 104.28/104.28,
cleanly separated). All runs KMeans, `k = 4` unless noted, per-day SCADA
scoping (a run never sees another day's Betriebsdaten).

Run-name note: sessions split on >15-min burst gaps exactly as before
(`250526-pu` morning/afternoon precedent), so the operator session `TU1`
becomes `010726-tu1-morning` (04:14-07:26, 64 files) + `010726-tu1-afternoon`
(two stray 07:45/07:46 files), and `PU_PH_PU_PH_PU_PH` becomes
`270626-pu_ph_pu_ph_pu_ph-1` (06:41-12:05 bulk, 106 files) + `-2`/`-3`
(single-timestamp stragglers at 14:41 and 16:33). The substantive `-morning`
/ `-1` splits are what ran below.

**State-level metrics** (primary; majority cluster->state mapping):

| run | variant | k | state ARI | state accuracy | state macro-F1 | n_eval | load-align ARI |
|---|---|---|---|---|---|---|---|
| 010726-tu_ph_tu | audio | 4 | 0.929 | 0.973 | 0.394 | 27821 | 0.436 |
| 010726-tu_ph_tu | vibration | 4 | 0.941 | 0.975 | 0.454 | 28045 | 0.491 |
| 010726-tu_ph_tu | fusion | 4 | 0.930 | 0.973 | 0.394 | 27821 | 0.439 |
| 010726-tu1-morning | fusion | 4 | 0.749 | 0.971 | 0.509 | 11039 | 0.182 |
| 290626-tu | fusion | 4 | 0.894 | 0.951 | 0.509 | 15564 | 0.122 |
| 270626-pu_ph_pu_ph_pu_ph-1 | fusion | 4 | — | — | — | 0 | — |

`270626-pu_ph_pu_ph_pu_ph-1` (the day with no Betriebsdaten at all) took the
documented reduced-report path end-to-end: 18716 windows detected, GT
entirely `unknown`, `summary.csv` row carries `notes = "no SCADA coverage"`
with every GT metric empty -- the ~4 h alternating pump/phase-shifter session
is processed but unevaluable until SCADA (or the power-curve photo) provides
labels.

**The 4-state day (`010726-tu_ph_tu`, 15:45-24:00: standstill ->
phase-shifter 15:52-17:35 -> turbine generation).** GT eval windows: 20926
turbine / 6139 phase-shifter / 635 transition / 117 standstill / 4 pump
(pump is vestigial -- SCADA before the session start). State-level
(majority-mapped) confusion, fusion-KMeans k=4:

| GT \ predicted | phase-shifter | turbine |
|---|---|---|
| phase-shifter | 6139 | 0 |
| pump | 1 | 3 |
| standstill | 114 | 3 |
| transition | 101 | 534 |
| turbine | 0 | 20926 |

Audio's confusion is identical up to a couple of transition windows;
vibration differs in structure (its non-PH quiet cluster maps to
transition, catching all 117 standstill windows there and leaking 83
phase-shifter windows into it -- audio and fusion instead absorb standstill
INTO the phase-shifter cluster).

**Does the detector separate phase-shifter?** From **turbine: yes,
perfectly, in every variant** -- 6139/6139 phase-shifter windows in their
own cluster with zero turbine cross-leakage (audio and fusion; vibration
6056/6139 with the 83-window leak going to its transition cluster, still
zero turbine confusion). This is the cleanest state separation the detector
has produced on any real recording so far, and it drives the first >= 0.9
state ARI on a genuinely multi-state day (0.929-0.941 across variants vs.
0.687-0.704 on the June-25 TU day). From **standstill: no** -- audio and
fusion put 114/117 standstill windows into the phase-shifter-majority
cluster (both are "machine quiet-ish" regimes; at 117 vs. 6139 windows the
majority vote can never go standstill's way), and vibration collapses
standstill into its transition cluster instead. The honest caveat on the
headline: this day's class balance (75% turbine, 22% phase-shifter, ~2.7%
everything else) plus the huge acoustic distance between "generating
200-290 MW" and "motoring unloaded at -3.5 MW" make it an easier clustering
problem than June-25's standstill/transition/turbine discrimination; state
macro-F1 (0.39-0.45) is the number that keeps the standstill/transition
misses visible.

**k-sweep (fusion, KMeans; state ARI / strict ARI / silhouette):**

| k | state ARI | strict ARI | silhouette |
|---|---|---|---|
| 3 | 0.929 | 0.453 | 0.460 |
| 4 | 0.930 | 0.386 | 0.384 |
| 5 | 0.930 | 0.246 | 0.276 |
| 6 | 0.907 | 0.250 | 0.264 |

What does the sweep prefer on a 4-state day? Mode-level: nothing --
state ARI is flat at 0.929-0.930 for k = 3..5 (every k allocates exactly ONE
phase-shifter-majority cluster and spends the rest on turbine load
sub-structure; k=6 dips slightly). Internal cohesion (silhouette) prefers
k=3, and strict ARI decays monotonically past it -- consistent with the day
effectively containing two acoustically dominant regimes (phase-shifter vs.
turbine) plus load substructure, not four separable ones. There is no
evidence here that k must grow to match the nominal state count; the
detector's mode-level output is insensitive to k in the swept range.

**290626-tu -- the phase-shifter hold inside a TU session** (the hard
boundary case: a ~39-min converter-assisted-start PH hold before
generation, plus a shorter post-generation unloaded spin-down that also
exceeds the 600-s dwell; 3071 PH windows total in GT). Fusion-KMeans k=4:
state ARI 0.894, accuracy 0.951, macro-F1 0.509 -- and the PH hold is
separated **perfectly**: 3071/3071 phase-shifter windows in their own
cluster, zero leakage in either direction against turbine (11667/11667
turbine windows also perfectly clustered at the mode level). This day even
allocates a genuine majority-standstill cluster (66/148 standstill windows
caught; the other 82 fall into the PH cluster -- same standstill/PH
adjacency as on 01.07). The 0.894 headline is dragged below 0.9 by
transition scatter (266 transition windows in the PH cluster from the
ramp edges), not by any mode confusion.

**010726-tu1-morning -- plain 3-state control.** A TU-only session on the
same day/hardware as the 4-state headline run: state ARI 0.749 (accuracy
0.971, macro-F1 0.509), i.e. noticeably better than June-25 TU fusion
(0.687) but nowhere near the phase-shifter days -- reinforcing that the
0.9+ results above are driven by phase-shifter's acoustic distinctness, not
by a general leap in detector quality. Standstill recall at the state level
is again 0 (all 130 standstill windows in a transition-majority cluster).
Load alignment is modest everywhere this cycle (0.12-0.49), best on the
4-state day's vibration variant (0.491).

**Reality fixes shipped during these runs** (one commit each, see git
history): the CLI's `--run` argparse whitelist predated dynamic multi-day
run names and rejected every new run; and `VibFeaturizer`'s dead-channel
std, computed in float32, picked up pairwise-summation rounding noise
(~4.77e-07 for channels exactly constant at -7.0 on `RAWTurbineVib__3`,
2026-07-01) that flip-flopped the same dead channel live/dead across files
of one stream and crashed feature extraction with a width mismatch; the
std is now computed in float64, where a constant channel is exactly 0.0
for every batch shape.
