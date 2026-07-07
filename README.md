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
`.env`) points at a local copy of the following tree, sourced from
`~/Downloads/illwerke-250526-analysis`:

```
<ROWII_DATA_ROOT>/
├── 20260626 Messung/
│   ├── TU/                     # 48 files, 25 GB — turbine run, 2026-06-25 04:15-06:27 UTC,
│   │                           #   12-min segments x 4 streams
│   ├── PU/                     # 21 files, 9.6 GB — pump runs, 2026-06-25 morning + afternoon
│   └── Betriebsdaten/          # hourly SCADA files, 2026-06-25 00:00-11:00, ~30 channels @ 10 Hz
├── Sensor_Anordnung_15062026.xlsx   # stream index -> physical sensor mapping
├── ROWII_Leistung_{PU,TU}.jpg       # power curve screenshots (backup ground truth)
└── MANIFEST.md                      # provenance notes
```

Each burst contributes four streams: `RAWGeneratorMic__0` / `RAWTurbineMic__1`
(~50 kHz microphone rings) and `RAWGeneratorVib__2` / `RAWTurbineVib__3`
(~10 kHz tri-axial accelerometers). All files are Gantner UDBF v1.07 binaries.

## Quickstart

Copy the required subset of a source tree into `ROWII_DATA_ROOT`:

```bash
python scripts/copy_data.py --source ~/Downloads/illwerke-250526-analysis
```

Run state detection for one recording / input variant / clusterer:

```bash
python scripts/run_step1.py --run tu --variant audio --clusterer kmeans
python scripts/run_step1.py --run tu --variant fusion --clusterer kmeans
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
mypy src
```

## First real results (TU + PU-morning, 2026-06-25)

Tasks 13/13b ran Step 1 against the real June-25 Rodundwerk II delivery (35
GB: 48 TU + 8 PU-morning burst files across 4 streams, 12 Betriebsdaten
hours). Parameter verification (`scripts/verify_parameters.py`,
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
`rowii.eval.metrics` module docstring):

| run | variant | state ARI | state accuracy | state macro-F1 |
|---|---|---|---|---|
| tu | audio | 0.684 | 0.960 | 0.721 |
| tu | vibration | 0.153 | 0.942 | 0.509 |
| tu | fusion | 0.687 | 0.959 | 0.705 |
| pu-morning | audio | 1.000 | 1.000 | 1.000 |
| pu-morning | vibration | 1.000 | 1.000 | 1.000 |
| pu-morning | fusion | 1.000 | 1.000 | 1.000 |

PU-morning's 1.000 row is a degenerate case, not a detector triumph: this
recording never leaves the pump state (all 719-1439 eval windows are GT
`"pump"`), so every cluster's majority vote trivially resolves to
`"pump"` and every state-level metric collapses to its
identical-partition value by convention. The strict-metrics table below
shows the same runs' `ARI = 0.000` (single-GT-class ARI is degenerate),
which is the more honest read of PU-morning's actual information content.

**Strict (1:1 Hungarian) metrics** -- secondary, kept for continuity with
Task 13's original numbers and as an over-segmentation diagnostic (a large
state-level vs. strict gap means the detector's extra clusters are
sub-modes, not confusion):

| run | variant | k | ARI | macro-F1 | boundary \|Δt\| (s) | silhouette |
|---|---|---|---|---|---|---|
| tu | audio | 4 | 0.144 | 0.721 | 40 | 0.470 |
| tu | vibration | 4 | 0.128 | 0.658 | 135 | 0.316 |
| tu | fusion | 4 | 0.153 | 0.705 | 58 | 0.399 |
| pu-morning | audio | 4 | 0.000 | 1.000 | None | 0.045 |
| pu-morning | vibration | 4 | 0.000 | 1.000 | None | 0.023 |
| pu-morning | fusion | 4 | 0.000 | 1.000 | None | 0.035 |

TU fusion k-sweep (state ARI now the headline column, strict ARI/macro-F1
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
answered *partially yes*: ARI(load_bin, cluster) is 0.457 (audio), 0.514
(vibration), 0.465 (fusion) on TU, and 0.259 (audio), 0.204 (vibration),
0.259 (fusion) on PU-morning. These are well above 0 (genuine, non-random
structure) but well below 1 (not a clean recovery of the 3 load bins) --
the detector's turbine-phase sub-clusters correlate with load level more
than chance but do not cleanly separate it, consistent with load level
being one of several factors (alongside noise, transient dynamics, and
whatever else drives the acoustic/vibration signature) the unsupervised
clustering is picking up on.

**Does fusion clear 0.9 on state-level ARI?** No, on either run in the
"real detector recovering real structure" sense: TU fusion state ARI is
0.687 (best k-sweep value 0.710 at k=6, still short of 0.9); PU-morning
fusion's 1.000 is the degenerate single-GT-class case described above, not
a genuine 3-state recovery.

**Honest reading.** State-level metrics change the story dramatically from
Task 13's strict-only view: TU fusion's strict ARI (0.153) looked like
near-total failure, but its state ARI (0.687) shows the detector recovers
the correct mode most of the time once load-level sub-clusters are
credited instead of penalized -- confirming Task 13's own qualitative
read (the timeline visually tracked the SCADA power curve) was closer to
the truth than the strict ARI number suggested. Audio and fusion are close
on TU (state ARI 0.684 vs. 0.687); vibration lags substantially (0.153),
consistent with Task 13's standstill-recall finding (vibration and fusion
both catch only 47/121 standstill windows vs. audio's 118/121) still being
the dominant weakness. PU-morning's perfect-looking numbers are an
artifact of this delivery containing no non-pump SCADA-covered windows in
that run, not evidence the detector solves a harder multi-state problem
there -- the load-alignment ARIs (0.2-0.26) are PU-morning's only
non-degenerate signal, and they show a similar "some real structure, not a
clean recovery" pattern to TU. Neither run clears the ARI >= 0.9
acceptance gate in any metric family that isn't degenerate.

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
