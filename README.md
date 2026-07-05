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

## First real results (TU, 2026-06-25)

Task 13 ran Step 1 against the real June-25 Rodundwerk II delivery for the
first time (35 GB: 48 TU burst files across 4 streams, 12 Betriebsdaten
hours). Parameter verification (`scripts/verify_parameters.py`,
`results/parameter_verification.md`) measured every machine-parameter
hypothesis directly from this data rather than carrying over pre-delivery
guesses -- one had to be corrected (nominal speed), one plant-specific sign
convention had a genuine bug (pump-mode speed sign), and the three
`MACHINE_HZ` spectral centres were confirmed as-is.

| combination | k | ARI | macro-F1 | boundary \|Δt\| (s) | silhouette |
|---|---|---|---|---|---|
| audio / kmeans | 4 | 0.144 | 0.723 | 40 | 0.470 |
| vibration / kmeans | 4 | 0.128 | 0.660 | 135 | 0.316 |
| fusion / kmeans | 4 | 0.153 | 0.706 | 58 | 0.399 |

Fusion k-sweep (silhouette + ARI per k, same run/clusterer):

| k | silhouette | ARI |
|---|---|---|
| 3 | 0.393 | 0.154 |
| 4 | 0.399 | 0.153 |
| 5 | 0.410 | 0.148 |
| 6 | 0.245 | 0.076 |

Silhouette peaks at k=5, but ARI against SCADA ground truth is essentially
flat across k=3-5 and drops sharply at k=6 -- the data does not prefer a
finer split than k≈4 once measured against the actual operating-state
labels, even though k=5's clusters are the most *internally* compact.
Macro-F1 tells a different story (`summary.csv`): 0.457 at k=3 vs.
0.63-0.71 at k=4-6 -- with only 3 clusters mapping 1:1 to the 3 GT states
and no spare cluster to specialize on the harder standstill/transition
boundary, k=3's minority-class F1 (which macro-F1 weights equally against
the dominant turbine class) is noticeably worse even though its ARI is
marginally the best of the sweep.

**Honest reading.** Fusion does **not** reach the ARI ≥ 0.9 acceptance
gate (best: 0.153 at k=3). The raw number understates how well the detector
actually tracks the machine, though: this TU run is ~95% turbine-phase
windows, and the confusion matrix (`results/tu/fusion-kmeans/report.md`)
shows KMeans splitting that long turbine phase into multiple sub-clusters
that all Hungarian-map back to "turbine" -- a partition *refinement*, not a
misclassification, but one ARI penalizes heavily under such a dominant
majority class. Visually
(`results/tu/fusion-kmeans/timeline.png`), the detected
standstill/turbine/transition phases line up closely with the real SCADA
power-curve steps, including several of the mid-run load-level changes
around hour 1.0-2.0 -- the qualitative match is considerably better than
the ARI number alone suggests, but it is not yet reliable enough to declare
the four-state detector production-ready on real data. Standstill recall is
the weakest link across all three variants (audio catches 118/121
standstill windows; vibration and fusion only 47/121), suggesting the
short standstill segments at the very start/end of the run are the hardest
part of this problem, not the long turbine plateau.

Four genuine bugs surfaced and were fixed while getting real data through
the pipeline for the first time (see commit history, one fix per commit):

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
4. **`is_nominal`** (relevant once PU runs are evaluated, not TU): pump-mode
   `1_Drehzahl_Ist` is signed negative at this plant (opposite rotation
   direction from turbine mode), but the nominal-speed gate compared
   signed speed against a positive threshold, so it silently classified
   the entire pump run as "transition" regardless of speed_nominal_rpm.
