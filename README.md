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
│       ├── TU/                     # 48 files, 25 GB — turbine run, 2026-06-25 02:15-04:39 UTC,
│       │                           #   12-min segments x 4 streams
│       ├── PU/                     # 21 files, 9.6 GB — pump runs, 2026-06-25 morning + afternoon
│       └── Betriebsdaten/          # hourly SCADA (local-named files), coverage to 10:00 UTC, ~30 channels @ 10 Hz
├── illwerke-270626/
│   └── 20260627 Messung/
│       └── PU_PH_PU_PH_PU_PH/      # ~4h alternating pump / phase-shifter; NO Betriebsdaten
├── illwerke-290626/
│   └── 20260629 Messung/           # TU (incl. a ~37-min phase-shifter hold), PU, full-day SCADA
└── illwerke-010726/
    └── 20260701 Messung/           # PU, TU1, TU2, TU_PH_TU (all 4 operating modes), full-day SCADA
```

### SCADA coverage and permanent gaps (final — no historian re-export will be provided)

The sensor HARDWARE has been installed since **2026-06-15** (see
`Sensor_Anordnung_15062026.xlsx`) — but the Gantner stream CONFIGURATION was
re-saved on **2026-06-29**: the 250526 (recorded 2026-06-25; the run label is an
inherited misnomer) and 270626 deliveries carry `MeasName: 2026-06-15`, while
290626 and 010726 carry `MeasName: 2026-06-29`. Cross-day comparisons therefore
distinguish the two CONFIG eras, not the installation date (package-7 spec §1;
the 250526 frozen-threshold blow-up is the cross-config case). Delivered
coverage per window:

All times below are true UTC (derived from filename hints; corrected 2026-07-15
after the DAQ clock quirk fix — see the next subsection):

| Period | Audio + vibration | SCADA (Betriebsdaten) | Ground truth |
|---|---|---|---|
| 2026-06-25 TU (02:15–04:39 UTC) | ✓ | ✓ | ✓ |
| 2026-06-25 PU morning (06:56–07:44 UTC) | ✓ | ✓ | ✓ |
| 2026-06-25 PU afternoon (11:44–12:44 UTC) | ✓ | ✗ export ends 10:00 UTC | ✗ permanent |
| 2026-06-27 PU↔PH sessions (04:41–14:45 UTC, main session to 10:05) | ✓ | ✗ never exported | ✗ permanent |
| 2026-06-29 full day (TU 00:30–04:54, PU 07:40–13:40 UTC) | ✓ | ✓ 24 h | ✓ |
| 2026-07-01 full day (TU1 02:14–05:58, TU2 06:47–08:11, PU 10:52–12:16, TU_PH_TU 13:45–22:09 UTC) | ✓ | ✓ | ✓ |

Notes: (1) the gaps match the partner team's situation exactly — their 27.06
analyses rely on photo-derived hybrid labels and mark that day as an outlier;
(2) decision recorded 2026-07-14: Illwerke will NOT re-export the missing
historian hours, so these two windows permanently lack SCADA ground truth
(photo-derived approximate labels are the only possible fallback and would be
documented as a separate, lower-confidence label tier); (3) first fully
covered day: 2026-06-29; first day with all four operating modes: 2026-07-01;
(4) the Betriebsdaten export additionally contains SCADA-only history back to
2026-06-01 (577 hourly files before 25.06) — not usable for the pipeline
(no audio/vibration exists before 25.06), kept for provenance only.

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

### DAQ clock quirk (epoch-2000-as-1970, local-time clock)

Every Gantner UDBF file in this dataset — audio, vibration, AND the SCADA
Betriebsdaten `.dat` files — carries binary frame timestamps (`header.t0_ns`)
from a DAQ clock that counts **seconds since 2000-01-01 LOCAL time but declares
them as Unix (1970) nanoseconds**. Read naively, `header.t0_ns` therefore
decodes to a wall-clock-of-day that matches the file's true local time exactly,
under a fixed, wrong epoch year — e.g. a file whose true instant is
`2026-06-27T04:41:03Z` (local Europe/Vienna `06:41:03`, CEST = UTC+2) carries a
raw `header.t0_ns` that decodes to `1996-06-27T06:41:03Z`.

The pipeline derives, never hardcodes, the per-run and per-Betriebsdaten-file-set
offset that maps this raw axis onto true UTC
(`rowii.io.dataset.run_utc_offset_ns` / `betriebsdaten_utc_offset_ns`, full
numeric derivation in `run_utc_offset_ns`'s own docstring): the median, over
every file, of `(filename-hint UTC − header.t0_ns)`, rounded to the **nearest
hour** — the true offset is always an exact whole-hour value (epoch-2000 shift
minus a whole local UTC offset), so rounding recovers it exactly regardless of
sub-second filename-truncation scatter. A **plausibility gate** returns offset
`0` whenever the median raw offset is under 1 hour, so a future dataset
recorded on an already-correct clock is never given a spurious shift. The
audio/vibration run's own offset and the SCADA file set's own offset are always
derived independently (never one blindly copied onto the other) and are
cross-checked: a disagreement of more than 2 seconds is logged as a warning.

This offset is applied once, at the single upstream point each axis enters the
pipeline (`rowii.pipeline.build_run_grid` for audio/vibration,
`rowii.scada.labels.load_scada_window_means` for SCADA) — every window grid,
segment table, candidate timestamp, and report time is true UTC by
construction from there on; detected labels, scores, and false-alarm rates are
completely unaffected (a per-run constant shift moves only where a window is
*displayed* in time, never which samples fall in it). **All persisted times
are true UTC starting from this fix.** Every `results/step2` artifact was
regenerated on the corrected axis on 2026-07-15 (FARs/labels/scores are
translation-invariant and did not change — verified byte-identically), and the
"Data layout" tables above now carry true-UTC session spans. Step-1 artifacts
under `results/<run>/` and any timestamps quoted in the Step-1/multi-day README
sections below still render the raw axis (wall-clock numerals are correct local
time, the year/epoch is not); the raw-axis Step-2 originals are preserved under
`results/step2-rawaxis-archive/`.

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

## Step 2 first evidence (2026-07-09)

First real-data results from the Step-2 mode-conditioned anomaly-detection
skeleton (`scripts/run_step2.py`; design spec
`docs/superpowers/specs/2026-07-09-step2-mode-conditioned-ad-design.md`).
No anomaly labels exist yet -- these are the two label-free evidence
artifacts the spec defines: false-alarm-rate (FAR) control on held-out
normal windows, and an anomaly-candidate register. Handcrafted features,
kNN (k=1, cosine) scorer, split-conformal thresholds at nominal
alpha = 0.05, per-run blocked segment splits (calibration and scoring
never share a 12-min burst segment), Step-1 detected cluster labels
(k = 4 KMeans + HMM) as the mode-conditioning signal unless noted.
Commands:

```bash
python scripts/run_step2.py --protocol within-day --run <run> \
    --variant fusion|audio --scorer knn --conditioning all --alpha 0.05
python scripts/run_step2.py --run 010726-tu_ph_tu --variant fusion --labels gt ...       # diagnostics
python scripts/run_step2.py --run 010726-tu_ph_tu --variant fusion --scorer mahalanobis  # scorer contrast
python scripts/run_step2.py --protocol cross-day --variant fusion --scorer knn
```

Timestamp caveat (applies to every `utc_time` below and in
`results/step2/`): the DAQ writes local wall-clock digits under a mis-set
1996 epoch (see `results/parameter_verification.md`, "burst-file /
Betriebsdaten clock convention"). The offset is uniform across all
streams and SCADA of a day, so alignment, splits and segment logic are
unaffected -- but the displayed times are not true calendar UTC.
**This is now fixed at the source** (see "DAQ clock quirk" under "Data
layout" above) -- every `utc_time`/`start_utc`/`end_utc` from a
combination re-run after that fix is true UTC; the specific values quoted
below predate it and still carry the raw axis this caveat describes.

### Within-day FAR tables (detected labels, kNN, alpha = 0.05)

One row per detected state; `(aggregate)` pools every calibrated state's
alarms/windows. `n_cal` is the per-state conformal-calibration size
(low-conf = below the n >= 1/alpha - 1 floor, threshold +inf, never
alarms; excluded = fewer than 20 fit-part windows, no reference built).

| run | variant | conditioning | label | n_cal | n_scored | n_alarms | realized FAR | low-conf | excluded |
|---|---|---|---|---|---|---|---|---|---|
| 250526-tu | fusion | per-state | 0 | 517 | 1113 | 5 | 0.004 |  |  |
| 250526-tu | fusion | per-state | 1 | 914 | 2638 | 254 | 0.096 |  |  |
| 250526-tu | fusion | per-state | 2 | — | — | — | — | yes | yes |
| 250526-tu | fusion | per-state | 3 | 7 | 217 | 0 | 0.000 | yes |  |
| 250526-tu | fusion | per-state | **(aggregate)** | 1438 | 3968 | 259 | 0.065 | yes |  |
| 250526-tu | fusion | pooled | 0 | 1438 | 1113 | 0 | 0.000 |  |  |
| 250526-tu | fusion | pooled | 1 | 1438 | 2638 | 287 | 0.109 |  |  |
| 250526-tu | fusion | pooled | 2 | 1438 | 47 | 0 | 0.000 |  |  |
| 250526-tu | fusion | pooled | 3 | 1438 | 217 | 7 | 0.032 |  |  |
| 250526-tu | audio | per-state | 0 | 920 | 2672 | 90 | 0.034 |  |  |
| 250526-tu | audio | per-state | 1 | — | — | — | — | yes |  |
| 250526-tu | audio | per-state | 2 | 518 | 1115 | 11 | 0.010 |  |  |
| 250526-tu | audio | per-state | 3 | — | — | — | — | yes |  |
| 250526-tu | audio | per-state | **(aggregate)** | 1438 | 3787 | 101 | 0.027 | yes |  |
| 250526-tu | audio | pooled | 0 | 1438 | 2672 | 155 | 0.058 |  |  |
| 250526-tu | audio | pooled | 1 | 1438 | 155 | 42 | 0.271 |  |  |
| 250526-tu | audio | pooled | 2 | 1438 | 1115 | 0 | 0.000 |  |  |
| 250526-tu | audio | pooled | 3 | 1438 | 73 | 0 | 0.000 |  |  |
| 290626-tu | fusion | per-state | 0 | 2498 | 3127 | 205 | 0.066 |  |  |
| 290626-tu | fusion | per-state | 1 | — | — | — | — | yes | yes |
| 290626-tu | fusion | per-state | 2 | — | — | — | — | yes | yes |
| 290626-tu | fusion | per-state | 3 | — | — | — | — | yes | yes |
| 290626-tu | fusion | per-state | **(aggregate)** | 2498 | 3127 | 205 | 0.066 | yes |  |
| 290626-tu | fusion | pooled | 0 | 3595 | 3127 | 0 | 0.000 |  |  |
| 290626-tu | fusion | pooled | 1 | 3595 | 2438 | 270 | 0.111 |  |  |
| 290626-tu | fusion | pooled | 2 | 3595 | 2024 | 0 | 0.000 |  |  |
| 290626-tu | fusion | pooled | 3 | 3595 | 66 | 0 | 0.000 |  |  |
| 290626-tu | audio | per-state | 0 | 2520 | 3127 | 119 | 0.038 |  |  |
| 290626-tu | audio | per-state | 1 | — | — | — | — | yes | yes |
| 290626-tu | audio | per-state | 2 | — | — | — | — | yes | yes |
| 290626-tu | audio | per-state | 3 | — | — | — | — | yes | yes |
| 290626-tu | audio | per-state | **(aggregate)** | 2520 | 3127 | 119 | 0.038 | yes |  |
| 290626-tu | audio | pooled | 0 | 3595 | 3127 | 2 | 0.001 |  |  |
| 290626-tu | audio | pooled | 1 | 3595 | 2344 | 322 | 0.137 |  |  |
| 290626-tu | audio | pooled | 2 | 3595 | 2118 | 119 | 0.056 |  |  |
| 290626-tu | audio | pooled | 3 | 3595 | 66 | 7 | 0.106 |  |  |
| 010726-tu_ph_tu | fusion | per-state | 0 | 1678 | 7109 | 565 | 0.079 |  |  |
| 010726-tu_ph_tu | fusion | per-state | 1 | 2482 | 3595 | 168 | 0.047 |  |  |
| 010726-tu_ph_tu | fusion | per-state | 2 | 735 | 2010 | 86 | 0.043 |  |  |
| 010726-tu_ph_tu | fusion | per-state | 3 | 2280 | 1666 | 175 | 0.105 |  |  |
| 010726-tu_ph_tu | fusion | per-state | **(aggregate)** | 7175 | 14380 | 994 | 0.069 |  |  |
| 010726-tu_ph_tu | fusion | pooled | 0 | 7175 | 7109 | 2016 | 0.284 |  |  |
| 010726-tu_ph_tu | fusion | pooled | 1 | 7175 | 3595 | 27 | 0.008 |  |  |
| 010726-tu_ph_tu | fusion | pooled | 2 | 7175 | 2010 | 10 | 0.005 |  |  |
| 010726-tu_ph_tu | fusion | pooled | 3 | 7175 | 1666 | 0 | 0.000 |  |  |
| 010726-tu_ph_tu | audio | per-state | 0 | 735 | 2038 | 193 | 0.095 |  |  |
| 010726-tu_ph_tu | audio | per-state | 1 | 1691 | 7109 | 488 | 0.069 |  |  |
| 010726-tu_ph_tu | audio | per-state | 2 | 2477 | 3595 | 132 | 0.037 |  |  |
| 010726-tu_ph_tu | audio | per-state | 3 | 2272 | 1638 | 35 | 0.021 |  |  |
| 010726-tu_ph_tu | audio | per-state | **(aggregate)** | 7175 | 14380 | 848 | 0.059 |  |  |
| 010726-tu_ph_tu | audio | pooled | 0 | 7175 | 2038 | 1 | 0.000 |  |  |
| 010726-tu_ph_tu | audio | pooled | 1 | 7175 | 7109 | 1687 | 0.237 |  |  |
| 010726-tu_ph_tu | audio | pooled | 2 | 7175 | 3595 | 3 | 0.001 |  |  |
| 010726-tu_ph_tu | audio | pooled | 3 | 7175 | 1638 | 0 | 0.000 |  |  |

### Per-state vs. pooled: the design's first Step-2 claim

**Per-state conditioning holds FAR closer to nominal, and the effect is
per-state, not aggregate.** Across all six run x variant combos, every
state that per-state conditioning could calibrate realizes a FAR between
0.000 and 0.105 (nominal 0.05; worst case ~2.1x). Under a pooled
(mode-agnostic) reference the same states realize 0.000-0.284: false
alarms concentrate on one or two states per run (0.284 on the 01.07
turbine cluster, 0.271 / 0.237 / 0.137 / 0.111 / 0.109 elsewhere; on
290626-tu audio three states exceed nominal at 0.137/0.106/0.056) while
most remaining states are pushed far below nominal (many at 0.000-0.032)
by a threshold inflated past their own score range. Pooled AGGREGATES (0.035-0.143) can
nonetheless sit deceptively close to alpha because over- and
under-alarming states cancel -- the aggregate number hides exactly the
per-mode miscalibration the design predicts for mode-agnostic
thresholds. Two honest caveats: (1) per-state aggregates still land
1.2-1.4x above nominal on four of six combos (0.059-0.069) --
within-run drift makes blocked calibration/scoring splits only
approximately exchangeable; (2) on 290626-tu, three of four detected
states were excluded per-state (below `min_ref` = 20 fit windows):
that day's detected states are concentrated in few 12-min segments
(39-min PH hold, short standstill), so segment-granular splits starve
their references -- per-state conditioning needs either more data per
state or cross-day reference pooling (future package).

### Diagnostic: detected vs. GT labels (010726-tu_ph_tu, fusion, kNN)

The `--labels gt` switch isolates the detected-label confounder
(detected states inherit Step-1 errors; Step-1 state accuracy
0.95-0.975). Result -- the confounder currently cuts the OTHER way:

| labels | conditioning | aggregate FAR | per-state range | note |
|---|---|---|---|---|
| detected (k=4) | per-state | **0.069** | 0.043-0.105 | all 4 states calibrated |
| gt (5 states) | per-state | **0.188** | 0.059-0.237 | turbine 0.237; pump excluded; standstill n_cal=35, 0 scored |
| detected | pooled | 0.143 | 0.000-0.284 | |
| gt | pooled | 0.231 | 0.010-0.637 | transition 0.637 |

GT's single wide "turbine" state spans 91-292 MW of load; calibration
and scoring segments carry different load mixes, breaking within-state
exchangeability and inflating turbine FAR to 0.237. The finer detected
clusters (which split turbine load levels) are MORE acoustically
homogeneous than the coarse GT states, so Step-1 label noise does not
degrade FAR control here -- it improves it. Conditioning granularity,
not label perfection, is what mattered on this day.

### Scorer contrast: Mahalanobis vs. kNN (010726-tu_ph_tu, fusion, detected)

| scorer | conditioning | aggregate FAR | per-state range |
|---|---|---|---|
| kNN (k=1 cosine) | per-state | 0.069 | 0.043-0.105 |
| Mahalanobis (diag, shrinkage 0.1) | per-state | **0.043** | 0.000-0.148 |
| kNN | pooled | 0.143 | 0.000-0.284 |
| Mahalanobis | pooled | 0.040 | 0.000-0.161 |

Mahalanobis aggregates sit closer to nominal on this run, but its
per-state spread is wider (label 2: 0.148) and pooled conditioning shows
the same single-state alarm concentration (label 1: 0.161) -- scorer
choice does not rescue a mode-agnostic threshold.

### Cross-day pooled FA matrix (fusion, kNN, alpha = 0.05)

Calibrate a pooled threshold on the row run, score every valid window of
the column run. Rows/columns are RUNS (a measurement day contributes
multiple sessions); `n/a` = same day tree (not cross-day by
construction). `010726-tu1-afternoon` (SCADA-covered but only two stray
burst files; >5 % invalid windows) fails `prepare_run` and is a
documented exclusion -- every pair touching it is skipped.

| calib \ score | 010726-pu | 010726-tu1-morning | 010726-tu2 | 010726-tu_ph_tu | 250526-pu-aft | 250526-pu-morn | 250526-tu | 290626-pu | 290626-tu |
|---|---|---|---|---|---|---|---|---|---|
| 010726-pu | — | n/a | n/a | n/a | 0.999 | 1.000 | 0.600 | 0.178 | 0.236 |
| 010726-tu1-morning | n/a | — | n/a | n/a | 0.997 | 1.000 | 0.105 | 0.911 | 0.461 |
| 010726-tu2 | n/a | n/a | — | n/a | 0.998 | 1.000 | 0.287 | 0.911 | 0.603 |
| 010726-tu_ph_tu | n/a | n/a | n/a | — | 1.000 | 1.000 | 0.542 | 0.927 | 0.495 |
| 250526-pu-afternoon | 0.198 | 0.082 | 0.100 | 0.087 | — | n/a | n/a | 0.144 | 0.116 |
| 250526-pu-morning | 0.020 | 0.008 | 0.007 | 0.010 | n/a | — | n/a | 0.021 | 0.011 |
| 250526-tu | 0.903 | 0.133 | 0.256 | 0.144 | n/a | n/a | — | 0.924 | 0.616 |
| 290626-pu | 0.028 | 0.671 | 0.633 | 0.409 | 0.999 | 1.000 | 0.666 | — | n/a |
| 290626-tu | 0.667 | 0.810 | 0.715 | 0.090 | 1.000 | 1.000 | 0.647 | n/a | — |

Cross-day false-alarm inflation is severe and ubiquitous, exactly as the
domain-shift literature predicts: against nominal 0.05, a threshold
calibrated on one day realizes anywhere from 0.007 to 1.000 on another
day's windows (median cell 0.60), and even the best inflated cells
(290626-tu -> 010726-tu_ph_tu 0.090; 250526-pu-afternoon -> 01.07 runs
0.08-0.10) miss nominal by ~2x -- while the handful of BELOW-nominal
cells (0.007-0.028) are miscalibrated in the opposite, silent
direction, not evidence of transfer working. The 250526-pu-morning asymmetry (as calibration source:
0.007-0.021; as scoring target: ~1.000) shows the transfer failing in
both directions -- its own wide score distribution produces a threshold
nothing else exceeds, while its windows look alien to every other day's
reference. Caveat: these cells conflate genuine day-to-day acoustic
shift with operating-mode-mix differences (the pooled reference has no
state alignment across days -- a pump session scored against a
turbine-day reference inflates trivially); disentangling the two needs
the per-state cross-day alignment deferred to a later package.

### Candidate highlights (top-3 per headline run, fusion, per-state kNN)

Full top-20-per-state tables with SCADA context:
`results/step2/candidate_register.md` (374 of 1,060 within-day candidate
rows carry a mechanical `operationally-explained (SCADA: ...)`
assessment; everything else, and all cross-day sections, remain
**unreviewed** -- none of these has been listened to yet).

- **250526-tu**: (1)+(2) windows 6096/6097 (05:57:30 raw clock, p=0.001,
  the run's two highest scores) -- a ~90 MW upward load step (138 -> 225 MW
  in ~15 s); operationally-explained. (3) window 8021 (06:29:35, p=0.001)
  -- steady 182 MW, 30-60 s BEFORE the shutdown ramp begins, no visible
  SCADA event; the audio variant independently top-ranks the same
  pre-shutdown band (windows 8033/8034): unreviewed -- needs listening.
- **290626-tu**: (1) window 2954 (03:20:11) -- onset of a steep down-ramp
  (225 -> 118 MW within a minute); (2) window 2934 (03:19:51) -- last
  steady seconds immediately before that ramp; (3) window 3456 (03:28:33)
  -- up-ramp 155 -> 244 MW. All three operationally-explained (load
  steps); the most interesting UNEXPLAINED cluster on this run is steady
  273-275 MW around 04:43 (audio pooled ranks 4/7/8) -- needs listening.
- **010726-tu_ph_tu**: (1) window 24658 (22:36:13, global-best p=0.003)
  -- steady 115 MW generation, no SCADA event within +/-40 s: unreviewed
  -- needs listening. (2) window 25627 (22:52:22) -- start of a ~20 MW
  load reduction; operationally-explained. (3) windows 1927/1928
  (16:17:22) -- mid-phase-shifter hold, steady -3.5 MW motoring, KS
  closed, no SCADA event: unreviewed -- needs listening (acoustic
  novelty inside the PH hold is exactly the kind of event this register
  exists for).

**Partner cross-reference (pre-start filling-valve / Fuelldüse,
deck-v3 p.16 -- comparison only, no values adopted):** each headline
recording contains a 35-42 s pre-start standstill band. In four of six
fusion/audio sweeps that band fell on the calibration side (never
scored). Where it WAS scored, both sweeps flag it: 290626-tu fusion
pooled produces its single pre-start alarm at window 34 (02:31:31 raw
clock, p=0.024) -- the last second of true standstill, immediately
before shaft rotation begins (rpm 5.3 at w35) -- and 290626-tu audio
pooled independently ranks pre-start windows 27/30 (p=0.0017) in its
top-20. Timing-consistent with the partner's observation at
state-sequence level (the slide's own timestamp could not be
re-verified this session); evidence is weak-to-moderate (1-3 one-second
windows, pooled conditioning only) and stays **unreviewed** in the
register.

### Limitations

Detected-label conditioning inherits Step-1 errors (state accuracy
0.95-0.975) -- the gt-labels diagnostic bounds the effect and currently
shows detected labels HELPING FAR control (0.069 vs 0.188 aggregate on
01.07), but that is one run on one day, not a general finding.
Calibration sizes are very uneven (n_cal 7-7,175): one state runs at a
+inf threshold (250526-tu fusion label 3, n=7 < 19 = 1/alpha - 1), and
on 290626-tu three of four states could not be calibrated at all
(segment-blocked label concentration) -- the conformal guarantee's
1/(n+1) width spans 0.0001-0.125 across states. Everything above is a
single alpha (0.05), single split seed, and the cross-day matrix is
pooled-only (per-state cross-day needs label alignment across days;
documented spec simplification) at run granularity. All `utc_time`
values carry the DAQ's mis-set clock (see caveat at top). No candidate
has been auditioned yet -- "operationally-explained" marks are
mechanical SCADA-rule annotations, not human review.

## Step 2 package-2 evidence (2026-07-15): cross-day transfer, calibration scarcity, BEATs

All artifacts under `results/step2/` on the true-UTC axis (regenerated after the
DAQ-clock fix; FARs identical to the raw-axis originals in
`results/step2-rawaxis-archive/`). Full numeric digest with per-table source
paths: `.superpowers/sdd/results_digest.md`. Grid: ordered pairs of
{250526-tu, 290626-tu, 010726-tu_ph_tu} × {fusion, audio, audio-beats} ×
{kNN, Mahalanobis}; alpha = 0.05 throughout; detected labels (runtime path).

### Cross-day: no protocol controls per-state FAR; pooled "control" is cancellation

The new `cross-day-per-state` protocol transfers day A's fitted detector
(`FittedDetector`: fit-day standardisation + fit-day HMM decode, no refit) to
day B and scores each window against its PREDICTED state's day-A reference and
threshold — runtime-honest, no GT anywhere, and it dissolves the cross-day
label-alignment problem (one model, one label space).

Headline (fusion, kNN; aggregate FAR at alpha = 0.05):

| pair | per-state agg | pooled protocol | best state | worst state |
|---|---|---|---|---|
| 290626 -> 010726 | 0.138 | 0.090 | 0.023 (n=6282) | 0.268 (n=11002) |
| 250526 -> 010726 | 0.208 | 0.144 | 0.082 (n=13166) | 0.906 (n=2496) |
| 010726 -> 290626 | 0.462 | 0.495 | 0.021 (n=1056) | 0.618 (n=10320) |
| 290626 -> 250526 | 0.687 | 0.647 | 0.362 (n=138) | 0.692 (n=8087) |
| 010726 -> 250526 | 0.731 | 0.542 | 0.572 (n=250) | 1.000 (n=629) |
| 250526 -> 290626 | 0.507 | 0.616 | 0.055 (n=5258) | 0.970 (n=304) |

Findings (all 36 combos in the digest):

1. **No cross-day protocol reaches nominal FAR reliably.** Only 14 of 72
   aggregate FARs are <= 0.05, and 12 of those are Mahalanobis POOLED values
   whose apparent control is the cancellation mechanism already demonstrated
   within-day in package 1: e.g. fusion-Mahalanobis 290626->010726 reports a
   pooled FAR of 0.004 while the per-state view of the same transfer has state
   0 at 0.531 — a mode-aware monitor would flood one mode with alarms and stay
   silent elsewhere; the pooled number hides exactly that.
2. **Day shift is state- and direction-dependent.** 290626->010726 is the most
   transferable pair in every variant × scorer group, and individual states
   transfer with near-nominal FAR (fusion state 1: 0.023; audio-beats state 1:
   0.007; audio-beats-Mahalanobis state 3: 0.004) while sibling states of the
   same transfer fail badly (up to 1.000). Anything involving 25.06 as source
   or target transfers poorly.
3. **Consequence (the package's practical claim):** transfer the DETECTOR
   (Step-1 state detection survives the day change), but RECALIBRATE the
   per-state thresholds on the target day. The scarcity results below show
   that recalibration is cheap.

### Calibration scarcity: 19–159 windows per state, when curvable

Windows-per-mode curves (50 seeded reps per budget, exact Beta bands,
threshold-only fast path; `results/step2/scarcity*/`): on 010726-tu_ph_tu,
every curvable fusion/audio-beats state that stabilises at all does so between
**n = 19 and n = 159** calibration windows (band rule: 50-rep mean FAR in
[0.025, 0.10] and 95th percentile <= 0.10) — i.e. seconds-to-minutes of
per-state data, far below a day of recording. Failure modes are as informative
as the successes: fusion state 3 never stabilises under kNN (full-pool anchor
0.105) but does at n = 79 under Mahalanobis, which in turn loses state 2
(anchor 0.148); audio-beats state 0 never alarms under Mahalanobis at ANY
budget (FAR 0.000 with real thresholds — under-band, not "well controlled").
Neither scorer dominates. On the starved day (290626) only 1–2 states are
curvable at all (min_ref = 20 on the fit side), which is the quantitative form
of package 1's "calibration windows per state is the binding constraint".

The deployment-facing secondary curve (segment accumulation, "record N more
minutes") reaches band-stability for 2 of 4 states within 10–22 accumulated
12-min segments (~113–249 min of mixed-mode recording) and shows much wider
rep-spread than window-level sampling (segment draws are clumpy) — recording
time budgets per mode matter more than raw window counts.

### BEATs as scoring representation: no uniform win; a complementary lens

Within-day (detected labels): audio-beats posts the best cell on 250526
(per-state kNN FAR 0.007) and the best pooled-Mahalanobis cell on 010726
(0.020), but the worst cells on 290626 (0.130/0.112) — no representation wins
uniformly, mirroring Step-1's split verdict. The sharper finding is WHAT the
representations flag: the global top-20 candidate lists of fusion vs
audio-beats overlap at Jaccard <= 0.081 on every day (audio vs audio-beats
reaches 0.379 on 010726) — BEATs surfaces substantially different moments than
the physics features. Needs-listening cross-check: the 2026-07-01T14:17:22Z
candidate is confirmed by all three combos; the 2026-06-25T04:29:35Z candidate
by fusion (p = 0.0011) and audio (p = 0.0022) but NOT by audio-beats; the other
three checks by none (two have near-misses 14–23 s outside the 5 s tolerance,
listed in `results/step2/overlap/`).

### 27.06 gets its first state timeline (qualitative, transferred)

`scripts/apply_detector.py --fit-run 010726-tu_ph_tu --apply-run
270626-pu_ph_pu_ph_pu_ph-1` (fusion + audio-beats;
`results/step2/transfer/.../timeline.md`): both variants agree on the coarse
structure of the no-SCADA day — a phase-shifter-rich first ~100 min
(04:41–06:14 UTC, gap-tolerant PH blocks up to 42.2 min) followed by one
uninterrupted ~219 min non-PH stretch to 09:53 UTC; PH totals differ by only
77 s between variants, block boundaries by <= 19 s. Hard limitation, stated on
every artifact: the fit day has NO pump state, so pump windows can only land in
turbine-mapped clusters — the timeline is evidence of PH-vs-non-PH structure,
not of pump detection; labels are transferred, the day has no ground truth,
and no accuracy is claimed.

### Compute reuse (why re-running any of this is cheap)

Nothing expensive ever runs twice: the BEATs checkpoint is pre-trained by
Microsoft and stored once (no training on our side, encoder frozen); embedding
extraction is cached once per run × variant under `results/cache/`
(sha256-fingerprinted; the true-UTC fix reuses caches byte-identically, only
the grid anchor moves); clusterers, scorers, and conformal thresholds refit in
seconds by design (deterministic, seed 7), so the entire package-2 evidence
suite above regenerates from warm caches in ~15 minutes. `FittedDetector` is
the future serialisation point for the runtime prototype.

## Step 2 package-3 evidence (2026-07-16): baselines, reconstruction, ensemble, score fusion, granularity

All artifacts under `results/step2/within-day/`; full numeric digest with per-table
source paths: `.superpowers/sdd/results_digest_p3.md`. Within-day protocol, three
SCADA days, alpha = 0.05, detected labels; every scorer runs on the SAME splits and
conformal harness, so differences are attributable to the scorer alone.

### Classical one-class baselines: no scorer dominates

Across 12 (run × variant × conditioning) cells, Isolation Forest posts the best FAR
in 5, LOF in 3, OC-SVM in 2, kNN and Mahalanobis in 1 each — and kNN is the WORST in
6/12. Per-cell spreads run 0.015–0.116; the smaller spreads are within single-split
Beta scatter while the largest (010726 fusion pooled, 0.027–0.143) reflects the same
within-day exchangeability violations noted under score fusion below, which hit some
scorers harder than others. The design-cited kNN default is not an empirical
champion here; the honest reading is that all five scorers are competitive and
representation × day effects dominate scorer choice.
The starved-day pattern (290626: 3/4 states excluded) is scorer-independent — it is
a property of the split, confirmed identical across all five scorers.

### Reconstruction pole: strong on the rich day, fragile across days

MLP-AE (fusion features) is competitive everywhere it ran (aggregate 0.026–0.079).
The logmel autoencoders are the best performers on 010726 (LSTM-AE per-state 0.019,
Conv-AE 0.027, including one genuinely alarm-free state with a real threshold) but
blow up on the sparser days (250526 LSTM-AE aggregate 0.253 with states to 0.471) —
the from-scratch pole needs per-day data volume that only the rich day provides,
which is precisely the constraint the design's transfer poles exist to relax.

### Majority ensemble (design commitment): suppression holds partially

OC-SVM + IF + LSTM-AE, >=2-of-3 votes, each member on its own per-state conformal
threshold: the per-state "ensemble FAR <= best member" claim holds in 4 of 7
checkable states; at run-aggregate level the ensemble beats the best single member
on 2 of 3 days. Member windows are time-aligned within 26–97 ms on 250526/290626
(sub-window DAQ stream offsets, measured and documented in each run's
`ensemble_notes.md`); no distribution-free guarantee is claimed for the vote —
members keep their own marginal guarantees.

### Score-level fusion: no consistent edge over feature-level fusion

Fisher-combined branch p-values (the rule whose simulation-mean FAR guarantee is
verified, leave-one-out-calibrated) sit at/below alpha in 6/11 floor-clearing
states. Of the remaining five, one is within single-split Beta scatter; the other
four exceed alpha by far more than scatter explains (up to 0.139 vs a 99% Beta
bound of 0.069) — and the SAME states exceed under every rule including the
single-branch baselines, so these are genuine within-day exchangeability
violations (operating-point drift between held-out calibration and scoring
segments, the finite-sample limitation already documented in package 2), not a
Fisher artifact. Tippett (documented-excess max-rule contrast) behaves similarly
on a different subset. Neither rule consistently beats package-1's feature-level
fusion numbers — score fusion is a validity-preserving alternative, not an
accuracy upgrade, on this data.

### Conditioning granularity: sub-state structure is real, and it costs calibration

k = 4 → 8 → 12 detected states: achievable (calibrated) states grow 4 → 6 → 8 while
shrinking as a fraction of nominal (100% → 75% → 67%) — finer conditioning trades
achievability for resolution exactly as the conformal floor predicts. The top-20
candidate lists of k8 and k12 agree strongly with each other (Jaccard 0.667) but
barely with k4 (0.08–0.11): sub-cluster conditioning surfaces a consistent,
DIFFERENT candidate population than state-level conditioning — the quantified form
of package 1's "detected labels beat GT" sub-cluster mechanism.

## Step 2 package-4 evidence (2026-07-16): the TF-C industrial-pretraining pole

Compact TF-C (time+frequency 1-D CNN encoder pair, cross-view NT-Xent; a documented
simplification of Zhang et al. 2022) pre-trained ONCE offline: `tfc_audio.pt` on
37,490 normal 1-s windows of MIMII pump 0 dB (final loss 1.96 vs ~6.2 chance level)
and `tfc_vib.pt` on only **808 windows** of CWRU + Paderborn K001/K002 — the public
vibration corpora are short recordings, so every vibration-tfc number below carries
a DATA-FLOOR caveat: it is a floor on what vibration-native SSL could do, not a fair
test. Variants `audio-tfc` / `vibration-tfc`; digest:
`.superpowers/sdd/results_digest_p4.md`.

### Industrial pretraining transfers for STATE SEPARATION where general-audio failed

Step-1 state detection on 010726-tu_ph_tu (kmeans, k=4): audio-tfc state-ARI
**0.907** / accuracy 0.972 and vibration-tfc **0.920** / 0.973 — where frozen BEATs
historically collapsed to state-ARI ≈ 0.000 on this pipeline. Both trail the
handcrafted variants by only ~0.02 ARI (audio 0.929 / vibration 0.941 / fusion
0.930). The kind of pre-training data matters: machine-sound SSL preserves the
operating-mode structure that AudioSet SSL erased. Bonus: vibration-tfc posts the
best load-alignment ARI in the whole grid (0.697 vs handcrafted vibration's 0.491)
— sub-state load structure survives the transfer even from an 808-window corpus.

### Within-day scoring: competitive, no new champion

Aggregate FARs for both TF-C variants sit in the same band as fusion/audio-beats
(no uniform winner, consistent with package 3's verdict); on the starved day
(290626) audio-tfc calibrates 2 of 4 states — matching audio-beats and beating
fusion's 1 of 4. The per-state-vs-pooled cancellation mechanism replicates exactly
for both TF-C variants.

### Cross-day: still no free lunch, but a more CONSISTENT one

No audio-tfc cross-day per-state combo clears alpha = 0.05 (aggregates 0.020–0.385)
— industrial pretraining does not repeal the package-2 finding that thresholds must
be recalibrated per day. It does make day-pair difficulty SCORER-CONSISTENT (best
290626→250526 and worst 250526→010726 under both kNN and Mahalanobis, where
fusion's ordering flips by scorer) — the representation, not the scorer, now
carries the day-shift structure.

### Candidate overlap: a third, partially-BEATs-aligned lens

On the rich day audio-tfc's top-20 overlaps audio-beats at Jaccard 0.176 (vs
fusion's 0.081 with either); on both sparser days it overlaps NOTHING (exact zero
against fusion and audio-beats alike, with full candidate lists on both sides) —
three representations, three substantially different candidate populations.

Provenance: corpora under `data/public/` with sha256 manifests + licenses
(MIMII CC BY-SA 4.0, CWRU academic-free, Paderborn CC BY-NC 4.0); checkpoints
`models/pretrained/tfc/` (~1.9 MB each, seed 7, 40 epochs); one-off pre-training
~35 min total on MPS. The cache-fingerprint payload is now golden-pinned by tests —
a lesson from this package: a payload-shape change silently invalidated every
existing cache and was caught by the analyze_step2 beats guard, then fixed for
byte-identical backward compatibility.

## Step 2 package-5 evidence (2026-07-16): adaptation & compactness

Can the frozen general-audio encoder be ADAPTED to the plant, and how small can
scoring get? Four adaptation/compression paths off the same BEATs-iter3+ base
(90.4 M params, 361.5 MB), all trained/evaluated on real Rodundwerk II data, plus
a trained cross-attention fusion head and an end-to-end latency/size benchmark.
All FAR numbers below are within-day realized FAR at nominal alpha = 0.05
(per-state conformal, detected labels, kNN) — a CALIBRATION-health readout (the
data contains no confirmed faults), not detection performance. Detected states
are re-derived per encoder, so per-state rows are not label-aligned across
encoders; compare at pooled level.

The FAR columns are the per-state-conformal AGGREGATE (the `pooled` summary row
of `per-state-knn/far_table.csv`: total alarms / total scored across states) —
NOT the repo's "pooled" mode-agnostic conditioning, which is a different sweep.

| encoder | size | 250526 aggr. FAR | 290626 aggr. FAR | 010726 aggr. FAR (adaptation day) |
|---|---|---|---|---|
| frozen BEATs | 361.5 MB | **0.007** | 0.130 | 0.058 |
| + LoRA (r=8, q/v, 4 min MPS) | 361.3 MB merged | 0.069 | **0.136** | 0.028 |
| + full FT (2 epochs, 3 min) | 361.3 MB | 0.051 | 0.364 | **0.025** |
| distilled student (KD) | **0.78 MB** | 0.019 | 0.189 | 0.035 |
| + INT8 (dynamic) | 124.3 MB | — | — | 0.059 |

### Adaptation helps ON the adaptation day; full FT pays for it elsewhere

Both adapted encoders roughly HALVE the frozen encoder's aggregate FAR excess on the
day they were adapted on (0.058 → 0.028/0.025; 8,000 windows, native token-level
masked latent-MAE — a documented PROXY objective for BEATs' unreproducible
tokenizer pretraining, restated in every checkpoint sidecar). Off-day the picture
inverts: on 290626 full FT degrades to aggregate FAR 0.364 (7.3x nominal) while LoRA
stays at the frozen encoder's level (0.136 vs 0.130) — the classic
adapter-vs-full-FT robustness trade-off, reproduced on real plant data. On the
already-well-calibrated 250526 both adapted encoders are worse than frozen
(0.05–0.07 vs 0.007). Verdict: adaptation is a same-condition tool; nothing here
justifies replacing the frozen default across days. Step-1 state ARI on 010726
moves 0.9220 (frozen) → 0.9290/0.9291 (LoRA/FT) — small, consistent, not the
package-4 rescue story (BEATs is not degenerate on this day to begin with).

### A 0.78-MB student keeps most of the story (with one honest asterisk)

The KD student (192,643-param CNN on log-mel patches → the teacher's primary-mic
768-d embedding; MSE 0.0041 after 30 epochs / 44 s) is **464x smaller** and
**~40x faster on CPU** (0.5–0.8 ms/window vs ~30 ms) than its teacher, yet lands
state ARI 0.9187 on 010726 (teacher: 0.9220) and aggregate FARs in the frozen
encoder's range on all three days. Asterisk: it was distilled on 010726's
calibration side, so its 010726 numbers are in-domain for the distillation day
(still leakage-safe — calibration side only, never scoring windows; sidecar
restates this).

### INT8: a size tool, not a latency tool (here)

Dynamic INT8 quantization: 361.5 → 124.3 MB (2.91x). Embedding drift, measured
and PERSISTED by `quantize_beats.py --drift-run` (sidecar
`models/adapted/beats_int8.json`): on the 704 real windows of the run's first
burst file, mean cosine 0.9982, p5 0.9973, min 0.9960 (an earlier 2,000-window
cache-based measurement agreed to the third decimal). Aggregate FAR 0.059 vs
frozen 0.058 on the same day — parity. Latency does NOT improve (~30 ms/window either way): the
pipeline is preprocessing- and attention-bound, and the dynamically quantized
linear kernels return no net win on this workload/CPU. The benchmark's
`n_params` column counts only residual fp32 parameters for the int8 module —
use size-on-disk for compression claims.

### Cross-attention fusion head: calibrates cleanly where states are covered

A 4-head cross-attention head (audio-beats primary-mic 768-d slice as query,
fusion-cache vibration columns as key/value, CLIP-style InfoNCE on the
calibration side; seconds to train per day) yields per-state FARs of
0.025/0.054/0.033/0.052 on 010726 — tighter around nominal than the raw
feature-level fusion baseline on the same day (up to 0.105). On the two sparser
days most states fall below the fit gate and are honestly reported as
low-confidence NaN rows rather than numbers. Third fusion level demonstrated;
evidence limited to the rich day.

### End-to-end benchmark (incl. preprocessing), synthetic == real

`scripts/benchmark_inference.py`, median ms/window from raw 50 kHz windows,
real-window pass (`--source run:010726-tu_ph_tu`) reproduces the synthetic
numbers throughout:

| config | params | size | CPU b=1 | CPU b=256 | MPS b=256 |
|---|---|---|---|---|---|
| handcrafted | 0 | — | 3.0 ms | 2.9 ms | — |
| logmel | 0 | — | 0.48 ms | 0.23 ms | — |
| BEATs fp32 | 90.4 M | 361.5 MB | 30.1 ms | 33.3 ms | 31.4 ms |
| BEATs INT8 | (residual fp32: 4.9 M) | 124.3 MB | 29.8 ms | 30.0 ms | cpu-only |
| TF-C | 479,560 | 1.9 MB | 1.4 ms | 1.6 ms | 0.63 ms |
| student | 192,643 | 0.78 MB | 0.78 ms | 0.49 ms | 0.26 ms |

MPS does not help BEATs (preprocessing-bound); it helps the small encoders only
at batch. A Raspberry-class CPU deployment budget is comfortably met by
handcrafted/logmel/TF-C/student; fp32/int8 BEATs costs ~30 ms/window — still
30x real-time for 1-s windows, but the memory footprint is the harder constraint.

### Execution honesty & compute reuse

Three real-data defects were found BY this execution and fixed fundamentally
(commits `ec7f506`, `d5756c3`): the distillation target and the cross-attention
audio query both silently assumed a 768-column audio-beats cache (it is 1536:
both mic streams) — now a single shared primitive (`rowii.pipeline.
stream_columns`) slices the primary-mic block by feature-name prefix for all
three consumers; and the benchmark crashed on set-but-missing checkpoint envs
(now a logged skip). Feature caches are ONE file per (run, variant): checkpoint
swaps re-extract and overwrite (~3–10 min/day on MPS) — result dirs are archived
with suffixes (`audio-beats-{lora,ft,int8}-detected`), caches are not. All five
checkpoints under `models/adapted/` with JSON sidecars (objective caveat,
leakage note, seeds); adapters merge back into the standard `{"cfg","model"}`
format, so every downstream path loads them unchanged.

## Step 2 package-6 evidence (2026-07-17): runtime prototype + pillar-3 readiness

The LAST package of the code roadmap: the "runs at the plant" requirement made
concrete, the labeled-fault evaluation prepared for the induced-fault campaign,
and the design's central figure realized on a public proxy.

### One artifact runs the plant recipe: snapshot + monitor CLI

`rowii.runtime.MonitorSnapshot` persists the fitted detector (HMM as plain
arrays — no pickle anywhere, `allow_pickle=False` end to end), per-state
reference matrices, calibration scores, and conformal thresholds as ONE npz +
JSON sidecar (14.5 MB for the 010726 fusion fit, 4 states). Round-trip verified
BITWISE on 29,344 real windows (apply labels, per-state scores, thresholds).
`scripts/monitor.py <snapshot> <new run>` then emits the state timeline,
`alarms.parquet`, alarm segments, and provenance notes:

| monitored day | `--thresholds recalibrate` (default) | `--thresholds frozen` |
|---|---|---|
| 250526-tu | pooled alarm rate **0.022** | **0.776** (state 2: 100%) |
| 290626-tu | pooled alarm rate **0.052** | 0.522 |

Package-2's central finding ("transfer detector + references, recalibrate
thresholds per day") is now a one-command contrast at nominal alpha = 0.05; the
frozen mode's notes carry the distribution-shift warning verbatim. 290626
honestly reports `no_conformal_data` for one state (683 windows with no
calibration-side coverage). Alarms remain CANDIDATES — no fault labels exist.

### Pillar-3 event harness: ready for the campaign, demonstrated on synthetic intervals

`rowii.eval.events.evaluate_events` + `scripts/eval_events.py`: per-event TPR,
first-alarm latency, and false-alarm rate outside events, against a documented
`events.csv` contract (tz-aware ISO-8601). Campaign day = `monitor.py` then
`eval_events.py`. The committed demo (3 synthetic 5-min intervals over
290626's real alarms) is labeled a HARNESS DEMO in every output: 0/3 detected
(expected — alarms are FAR-level noise), 362 false-alarm windows = 194/h at
window FAR 0.054 ≈ alpha.

### Detection-performance scarcity on the MIMII proxy: SSL wins exactly when data is scarce

`scripts/scarcity_detection.py` (clip-level splits, leakage rule
review-verified by runtime instrumentation; conformal thresholds calibrated on
CLIP-level scores) on MIMII pump 0 dB, 4 machine ids, caps 300 normal clips/id,
90 test-normal + 100–143 abnormal clips per id, seeds 7/8/9. Clip-level AUC
(mean over ids × seeds):

| representation | 5% (8 fit clips) | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| frozen BEATs | **0.911** | 0.927 | 0.940 | 0.948 | 0.953 |
| TF-C (MIMII-pretrained) | 0.773 | 0.843 | 0.908 | 0.933 | **0.958** |
| student (0.78 MB, PSHP-distilled) | 0.733 | 0.791 | 0.840 | 0.869 | 0.890 |
| logmel | 0.708 | 0.726 | 0.720 | 0.721 | 0.722 |
| handcrafted | 0.659 | 0.693 | 0.751 | 0.766 | 0.771 |

The design's scarcity question answered on the proxy: frozen general-audio SSL
is nearly data-free (0.911 from EIGHT normal clips; per-seed range
[0.897, 0.924] does not overlap TF-C's [0.742, 0.795]), while industrial
pretraining (TF-C, pretrained on THIS corpus's normals) needs data and only
overtakes at the full set (0.958 vs 0.953). The PSHP-distilled student
transfers to a foreign machine at 0.89 AUC in 0.78 MB. TPR@alpha=0.05 is
exactly 0 below fraction 0.5 — CORRECT conformal behavior, not failure: the
calibration side holds < 19 clips there (11 at fraction 0.25), so the
distribution-free guarantee forces the threshold to +inf; at 0.5/1.0 the
realized normal-clip FAR is 0.03–0.06 ≈ alpha (per-representation means; per-cell spreads are wider) with TPR up to 0.71 (BEATs) /
0.75 (TF-C). Caveats restated from the harness outputs: public-proxy evidence
in the machine-id domain (never PSHP), per-window standardization erases
absolute-level cues, pAUC = standardized/McClish (sklearn max_fpr=0.1).

### Review honesty

Per-task adversarial reviews: T1 approved (zero functional defects; covars
bit-exactness, split parity, on-disk mutation propagation all probed), T2
fix-required → resolved (an --alpha mutation-test gap closed; the min_ref
question resolved as a documented sweeps-identical reading), T3 approved with
zero findings, T4/T5 fix-required → resolved (window_s-aware resampling,
machine-id collision refusal, mtime in the cache fingerprint, corpus-gone
honesty with exit 1). The scarcity harness's leakage-freedom and clip-level
calibration coherence were confirmed by the reviewer with adversarially
constructed divergence cases, not just re-reads.

## Step 2 package-7 evidence (2026-07-21): robustness & the best-system comparison

Does the pipeline hold up ACROSS days, configurations, and modes — and which
representation earns the best-system statement? Multi-day reference pools with
held-out-day-group rotations, session normalization, TF-C continued pretraining
on the plant's own audio, a 4x-larger vibration corpus, multi-day adaptation,
rolling recalibration, and the first induced-anomaly evaluation (080726 hammer
strikes). Universality tags per A2.1: [same-cfg] = fit and test days share a
DAQ config era (MeasName), [cross-cfg] = they do not, [cross-mode] = the
monitored mode is absent from the fit day.

### Setup: pools, k selection, and the 080726 day

Pool B1 = the four 010726 runs (era B), pool B2 = the two 290626 runs (era B),
ALL-B = both; 250526 (era A) and 080726 (era C) are never pooled. Pooled k
selected by mean GT-state ARI over the canonical pool: k=4 -> 0.671 vs k=5 ->
0.670, k=6 -> 0.668 (`results/step2/cross-day-pooled/k_selection.json`) — flat
in k, with a coverage-vs-granularity trade-off inside: PU days cluster at ARI
0.87–0.96, TU days at 0.39–0.46 (the pooled detector merges TU load levels).

**Ground-truth timebase lesson (deployment-real):** the 080726 strike protocol
logs LOCAL time (CEST); the first transcription read it as UTC, which silently
placed every ST event OUTSIDE the standstill recording and every PU event in
plain post-strike pumping — and still produced plausible-looking TPRs from
false-alarm coincidences. Two independent pins fixed the offset at exactly
−2 h: the protocol's pump->phase-shifter changeover (~15:04–15:06 local)
appears in OUR audio-UTC state timeline at 13:05:28 UTC, and the ST strike span
fits its recording only under −2 h. Ground truth corrected
(`docs/groundtruth/080726_events_*.csv` headers document the verification);
every event evaluation below uses the corrected times. Clock discipline is a
first-class deployment requirement.

### D1/D2 — pooled references travel; frozen thresholds don't

Fusion, pooled kNN (k=4), pooled FAR at alpha=0.05, frozen | recalibrate:

| rotation (fit -> test) | tag | frozen | recalibrate |
|---|---|---|---|
| B1 -> 290626-tu | same-cfg | 0.075 | 0.044 |
| B1 -> 290626-pu | same-cfg | 0.301 | 0.031 |
| B2 -> 010726-tu_ph_tu | same-cfg | 0.203 | 0.068 |
| B2 -> 010726-pu | same-cfg | 0.097 | 0.029 |
| ALL-B -> 250526-tu | cross-cfg | 0.280 | 0.103 |
| ALL-B -> 250526-pu-morning | cross-cfg | 1.000 | 0.091 |

Recalibrate holds within ~2x of alpha on EVERY rotation including cross-config;
frozen degrades 1.5–6x on same-config rotations and fails totally (FAR 1.0) on
the cross-config PU day. Same pattern at alpha 0.01/0.10 (per-alpha artifacts:
`fusion-pooled-a<alpha>/`). Versus the single-day P6 snapshot the pool helps
exactly where coverage was the problem: monitor 290626-pu with the single-day
010726 snapshot alarms at 0.932 frozen (cross-mode-starved) vs the pool's 0.301;
on 250526-pu-morning both fail frozen (1.0) — a config change breaks frozen
thresholds no matter how many days are pooled. The deployment recipe is pooled
references + per-day (or rolling) threshold recalibration, not frozen artifacts.

### Representation comparison on a held-out day (290626-tu, alpha=0.05)

| representation | frozen | recalibrate |
|---|---|---|
| fusion (handcrafted audio+vib) | **0.075** | **0.044** |
| audio-beats (frozen BEATs) | 0.233 | 0.068 |
| audio-tfc (MIMII TF-C) | 0.305 | 0.217 |
| audio (handcrafted) | 0.481 | 0.197 |
| audio-student (pooled distill)* | 0.360 | 0.154 |

Fusion wins cross-day FAR control decisively; frozen BEATs is second. (290626-pu
replicates the fusion-first ordering at 0.301|0.031 vs audio-beats 0.686|0.038.)
*The audio-student row is NOT held-out for the student itself: its distillation
pool contains 290626-tu's calibration side (checkpoint sidecar), so read it as
context only — the student's valid held-out evidence is pillar 3 below.

### D3 — session normalization: falsified as a frozen-transfer fix

First-20-min median/MAD normalization mostly BACKFIRES frozen (290626-tu 0.075
-> 0.300; 010726-pu 0.097 -> 1.000; only 290626-pu improves 0.301 -> 0.101):
the normalization window's STATE MIX, not the session offset, dominates the
statistics. N-sweep (5/20/60 min on 290626-tu): frozen 0.363 / 0.300 / 0.001 —
even the "good" N=60 case just overshoots into over-conservatism; recalibrate
sits at 0.024–0.026 for every N, i.e. normalization adds nothing once
thresholds recalibrate. Verdict: not part of the best system; kept as a
documented negative result.

### D4 — TF-C continued pretraining on PSHP audio: no free plant-tuning win

Continued pretraining (`tfc_audio_pshp`, 7 min on MPS) and a from-scratch
control (`tfc_audio_pshp_scratch`) on the canonical pool's unlabeled audio vs
the frozen MIMII checkpoint:

| checkpoint | Step-1 ARI 010726-tu_ph_tu | Step-1 ARI 290626-pu | rot 290626-tu frozen | recal |
|---|---|---|---|---|
| MIMII (frozen) | 0.907 | **0.954** | **0.305** | 0.217 |
| PSHP-continued | 0.907 | 0.952 | 0.478 | 0.139 |
| PSHP-scratch | 0.907 | 0.948 | 0.425 | **0.091** |

State structure is unchanged (Step-1 flat to 3 decimals on the rich day);
plant-audio pretraining reshapes SCORE distributions — frozen thresholds get
worse, recalibrated FAR gets better, and the scratch control beats the
continued model, so the gain is not MIMII-initialization carrying over. With
recalibration mandatory anyway (D1/D2), the added training buys no reliable
advantage: the frozen MIMII checkpoint stays the default.

### D5 — vibration corpus v2 (Paderborn K003–K006, sha256-pinned): flat

`tfc_vib_v2` (CWRU + Paderborn K001–K006) vs v1 (K001/K002 only), Step-1 state
ARI: 010726-tu_ph_tu 0.920 -> 0.926, 290626-tu 0.859 -> 0.860. 4x more healthy
bearing data moves nothing materially — the v1 "data floor" caveat was about
corpus KIND, not corpus SIZE. v1 stays the default checkpoint.

### D6 — multi-day adaptation: pools don't beat single-day adaptation

Within-day FAR on the never-pooled cross-config day 250526-tu (per-state kNN,
alpha=0.05, audio-beats family): frozen 0.007 (over-conservative), single-day
full-FT 0.051, single-day LoRA 0.069, pool-LoRA (4 runs, both modes) 0.072,
pooled student 0.069. Multi-mode pooling of the adaptation set does NOT improve
calibration transfer over single-day adaptation here — and nothing beats
full-FT's near-nominal 0.051. (Pool-adapted artifacts still matter for pillar-3
below, where the student is a top performer.)

### D7 — rolling recalibration: works, and M is forgiving

Monitor 290626-tu against the B1 pool snapshot, trailing-window thresholds:
alarm rate 0.0547 (M=30) / 0.0510 (M=60) / 0.0506 (M=120) at alpha=0.05, with
the fit-day-fallback share dropping from 41% (M=30) to 24% (M>=60). Rolling
approximates full recalibration without a second pass; M=60 is the default.

### Pillar 3 — induced hammer strikes (080726, era C): the headline

Corrected ground truth, ±5 s tolerance, event-level TPR | realized window-FAR;
pool-B1 snapshots, recalibrate mode WITH event-free calibration
(`--exclude-calibration-events`, spec A2.3.3): the ground-truth intervals
(±5 s) are banned from the calibration side and scored instead, so thresholds
are calibrated on strike-free windows and every event is evaluable. (The first
pass omitted this; strike minutes landing in calibration segments were consumed
— structurally undetectable — and contaminated the ST thresholds. The
whole-branch review caught it; the numbers below are the corrected evaluation.)
PU session = 13 strikes DURING PUMPING (−279 MW); ST session = 13 strikes at
standstill. All snapshots are fit on era-B days only -> every number below is
[cross-cfg] AND zero-shot on the strike day.

**PU (pumping), TPR at alpha 0.01 / 0.05 / 0.10 (window-FAR@0.05):**

| representation | 0.01 | 0.05 | 0.10 | FAR@0.05 |
|---|---|---|---|---|
| audio (handcrafted) | **0.85** | **0.92** | **1.00** | 0.063 |
| audio-beats (frozen BEATs) | **0.85** | 0.85 | 0.92 | 0.049 |
| audio-student (pooled distill) | 0.77 | **0.92** | **1.00** | 0.068 |
| fusion | 0.54 | **0.92** | **1.00** | **0.046** |
| audio-tfc | 0.08 | 0.23 | 0.69 | 0.076 |

**ST (standstill), TPR at alpha 0.01 / 0.05 / 0.10 (window-FAR@0.01):**

| representation | 0.01 | 0.05 | 0.10 | FAR@0.01 |
|---|---|---|---|---|
| audio-beats (frozen BEATs) | **1.00** | 1.00 | 1.00 | 0.003 |
| audio-student (pooled distill) | **1.00** | 1.00 | 1.00 | 0.003 |
| audio-tfc | 0.92 | 1.00 | 1.00 | **0.000** |
| fusion | 0.92 | 1.00 | 1.00 | 0.022 |
| audio (handcrafted) | 0.38 | 1.00 | 1.00 | 0.003 |

At standstill EVERY representation detects all 13 strikes from alpha 0.05 —
including the distant landmarks (EG, 11TG, Kugelschieber) — and frozen BEATs +
the student do it already at alpha 0.01 with window-FAR 0.003. During pumping,
three representations reach 13/13 at alpha 0.10; audio-beats detects 11/13 at
the strictest alpha 0.01 with window-FAR 0.008. Typical latencies 0.3–4 s
(minute-level ground truth; seconds pending).

Two planned side arms (fusion): the FROZEN cross-era row detects 13/13 in both
sessions only trivially — window-FAR 0.62 (PU) / 0.96 (ST), the cross-config
frozen failure alarming on most normal windows, useless as a detector; the
session-norm arm (stats-bearing snapshot) DEGRADES the raw recalibrate numbers
(PU 0.69 TPR at 0.094 FAR vs raw 0.92 at 0.046; ST 0.92 at 0.071) — consistent
with D3's falsification.

Readings: (1) frozen BEATs — zero plant-specific training — is the best
strict-alpha operating point in BOTH sessions (PU 11/13 at 0.008 FAR, ST 13/13
at 0.003 FAR), the strongest universality datum in the package; (2) the 0.8 MB
distilled student matches or beats its 361 MB teacher in every cell but
strict-alpha PU — the compact deployment path costs almost nothing on real
anomalies; (3) audio-tfc, the best STATE-structure transfer, is the weakest
pumping-transient detector — representation choice is task-dependent (though
even it reaches 13/13 at standstill from alpha 0.05); (4) fusion's vibration
channels add nothing over audio-only for these (airborne-dominated) strikes;
(5) evaluation methodology is itself a result: without event-free calibration
the SAME artifacts read as TPR ceilings of 0.38 (ST) / 0.77 (PU) — induced-
event days MUST ban event windows from calibration or they understate the
detector and contaminate its thresholds.

### The best-system statement (comparison-derived, A2.1/A4.5)

Per-state kNN + split conformal on FROZEN representations, pooled multi-day
references, per-day or rolling (M=60) recalibration with event-aware
calibration exclusion on induced-event days. Representation by target: fusion
for cross-day FAR control (0.03–0.10 across all six rotations), frozen BEATs
for strict-alpha event detection (11/13 pumping + 13/13 standstill at
alpha 0.01), handcrafted audio or the 0.8 MB student for maximum event recall
at alpha >= 0.05. Every plant-tuning attempt in this package — TF-C-PSHP
continued pretraining, session normalization, multi-day LoRA/student pools —
failed to beat its frozen/universal counterpart on held-out days: with scarce
data, the universal-encoder + calibration-layer architecture IS the best system
we can justify, which is the thesis' universality claim made empirical. The
final deployed artifact pools ALL available days (A4.5); the rotation numbers
above are its honest generalization estimate.

Honesty: no real machine faults exist in any recording — induced strikes are
surrogate transients (verified minute-level ground truth, seconds pending);
detection numbers reflect these sensors and this plant's noise, not
exhaustively tuned detectors; the audio-student rotation on 290626-tu is
pool-tainted for the STUDENT (its distillation pool contains that day's
calibration side) and is excluded from held-out claims — its pillar-3 numbers
(080726, never seen) are the valid ones. Execution surfaced and fixed FOUR
deployment-reality defects now under test: ground-truth CSV comment parsing,
channel-availability drift between fit and monitored days (monitor now projects
onto the snapshot's feature contract), DAQ stream-set grid skew between
audio-beats and logmel caches (distill now pairs by integer window shift), and
event-contaminated calibration on induced-event days (monitor now supports
`--exclude-calibration-events`; found by the final whole-branch review).

## Step 2 package-8 evidence (2026-07-21): mode-model-bank, level-recal, explainable results

Stefan's per-mode model-bank idea as a real Step-1 alternative, an independent
test of the partner-inspired level-only recalibration, an explainability figure
suite computed from our own artifacts, and three data verifications. Full
figures + plain-language digest: `results/analysis-days/README.md`.

### D4 — data verifications (gates, all passed)

- **SCADA timebase (080726): UTC, synchronous to +2 s.** The Betriebsdaten
  power changeover sits at 13:05:30 UTC vs our audio-UTC state-timeline
  reference 13:05:28 — every 080726 GT-matched analysis is un-gated. (Speed is
  structurally blind to pump->phase-shifter, as documented.)
- **`RAWTurbineVib__3` ch0: std exactly 0.0 on EVERY day through 01.07, live
  (std 3.1–3.5) on 08.07** — the channel was cabled between eras B and C; the
  documented cause of the 243-vs-231 column drift the P7 monitor projection
  handles and the bank's contract guard refuses.
- Gen-mic channel profile at the ST strikes recorded (channel-anonymous; no
  azimuth claim — the sensor-drawing mapping is not delivered).

### D1 — the mode-model bank beats the pooled clusterer almost everywhere

Six held-out rotations x 3 families x 3 representations; ARI on the identical
{unknown, transition}-masked windows for BOTH arms; the bank is SUPERVISED
(SCADA labels at commissioning time) vs the unsupervised KMeans+HMM — an
information advantage, stated as such, not a method win. Mean ARI:

| representation | bank gaussian | bank knn | bank gmm | P7 pooled clusterer |
|---|---|---|---|---|
| audio-beats | 0.471 | **0.883** | 0.755 | 0.606 |
| vibration | **0.827** | 0.688 | 0.644 | 0.614 |
| fusion | 0.454 | 0.408 | 0.454 | 0.361 |

Where it matters most — the cross-config era-A days the clusterer failed on —
the bank holds: 250526-tu best-family ARI 0.83–0.98 vs the clusterer's
0.05–0.06; 250526-pu-morning 1.00 vs 0.00 (fusion) / 1.00 vs 1.00 (vibration).
`--smooth` (duration-filter only) adds small consistent gains (up to +0.07).
The chain probe converts label quality into alarm quality: bank-gaussian
per-mode references drop the frozen pooled FAR on B1->290626-tu to **0.011 vs
0.075** for the P7 detected-state chain under split-parity. Era-C zero-shot
mode-ID readout (A1.7, audio-beats): knn accuracy **0.918** / ARI 0.71 on
080726-pu_strikes with a 19.5% `no_mode_fits` rate — the rejection signal
fires on era drift, exactly the "nothing fits" behavior the bank was built to
expose (gaussian degrades to 0.41 accuracy: Mahalanobis is era-shift-fragile).

### D2 — level-only recalibration: falsified on our pipeline, with the reason

The quad `raw-frozen | level-recal-frozen | session-norm-frozen | recalibrate`
on identical cells, six rotations, audio + vibration (fusion excluded by
design — its `fuse()` per-run z-score is ITSELF an implicit level
normalization, the A1.1 finding that plausibly explains fusion's cross-day FAR
advantage all along):

- **audio: level-recal is a no-op** (e.g. 290626-tu 0.481 -> 0.483; both
  250526 cells unchanged). Structural reason: our kNN scoring is cosine on
  L2-normalized rows — largely level-invariant by construction, so there is
  no level lever left to pull. The partner's 100%->2% recovery lives on
  level-sensitive one-class envelopes; our architecture had already removed
  that failure mode at the metric level.
- **vibration: level-recal BACKFIRES** (0.112 -> 0.800, 0.013 -> 0.929,
  0.010 -> 1.000): the label-free first-20-min offsets conflate the prefix's
  MODE MIX with a session gain — vibration levels are strongly
  mode-dependent, so the "session offset" is really a mode offset. The same
  confound that falsified session-norm in P7, now reproduced level-only.
- **Positive finding instead: raw-frozen VIBRATION survives the era boundary**
  (250526 cells 0.122 / 0.010 at alpha=0.05) — microphones step, vibration
  doesn't, independently reproduced in our pipeline. On the 080726 strike day
  the same raw-frozen vibration snapshot detects **11/13 pump-operation
  strikes at window-FAR 0.041, zero-shot across eras** (audio frozen is
  trivially broken there, FAR 0.69–1.0; level-recal wrecks the vibration arm
  to FAR 0.513; the recalibrate+level-recal control is redundant as
  predicted, 0.92 TPR @ 0.063). Standstill stays out of reach for an
  operation-pool frozen vibration snapshot (TPR 0.08) — the pool contains
  almost no standstill.

### D3 — the explainability suite (what the tables never showed)

`scripts/analyze_days.py` renders, from our own caches/artifacts: day x day
FAR heatmaps (frozen/recalibrate/level-recal leaves), per-feature cross-day
stability with segment-block bootstrap CIs (shifts converted to dB with the
per-family factor — x20 amplitude, x10 power — before the 3 dB comparability
cutoff), the era-step figure (audio levels step at the 2026-06-29 MeasName
boundary, vibration flat — our independent account of WHY frozen thresholds
failed, consistent with Rodrigues & Zhang 2026), per-mode band/octave
signatures, a tone-vs-nearest-tone-free-octave contrast table, and the
pillar-3 TPR bars. Every partner-inspired analysis type carries its
attribution; every number is computed from our artifacts.

### Verdicts

1. **Bank**: with commissioning-time SCADA labels, the per-mode bank is the
   better Step-1 on held-out days — especially across config eras — and its
   rejection rate doubles as an era-drift flag. The unsupervised clusterer
   remains the no-SCADA fallback; both stay in the thesis with the trade-off
   stated.
2. **Level-recal**: not adopted. No-op on cosine-scored audio, harmful under
   mode-mix on vibration; daily recalibration remains the transfer mechanism.
   The attempt yields two keepers: the fusion-self-normalization finding and
   the vibration-frozen era-robustness result.
3. **Explainability**: the heatmaps/era-step/stability figures replace the
   FAR tables as the primary communication artifacts.

Honesty: bank numbers are supervised-vs-unsupervised comparisons (information
advantage, tagged in every artifact); 080726 bank readout is era-C zero-shot
with the contract guard refusing the drifted fusion/vibration columns (by
design — audio-beats is the drift-free representation there); no partner
number enters any computation; ST vibration-frozen failure and the level-recal
falsification are reported as measured.
