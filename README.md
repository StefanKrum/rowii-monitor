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
