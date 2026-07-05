# Step 1 — Operating-State Detection: Design Spec

**Date:** 2026-07-05
**Repo:** `rowii-monitor` (whole thesis implementation; Step 1 is the first milestone)
**Thesis anchor:** Design chapter §3.3 (Step 1), Evaluation chapter Campaign 1.
**Approved decisions (Stefan, 2026-07-05):** approach C (feature-based unsupervised
pipeline with a pluggable featurizer interface); whole-thesis repo scope; copy only the
data actually used; write all code fresh (no code copied from `hydropower-anomaly` or
`pshp-ssl-transfer`; their designs serve as reference knowledge only).

## 1. Goal

Detect the operating state of the Rodundwerk II pump-turbine per time window from audio
and vibration, unsupervised, and validate the detected states against SCADA-derived
ground truth. Deliverable is Campaign 1 of the thesis evaluation: evidence that the
distinct operating states are recovered reliably from the sensor streams, per input
modality (audio-only / vibration-only / fusion).

**Positioning (from the thesis design):** Step 1 is a prerequisite capability, not a
research axis. The unsupervised pipeline is fixed and simple; the comparisons reported
are (a) input modality {audio, vibration, fusion}, (b) audio-branch featurizer
{handcrafted, frozen BEATs embeddings} (Stefan, 2026-07-05: include BEATs as the
proven-on-v1 reference, since the June data is the first real-plant recording), and
(c) a clusterer robustness check {KMeans, GMM}. Hard cap: no further representations
(no TF-C, SSAST, from-scratch AE) in Step 1 — the representation ablation proper
belongs to Step 2. BEATs serves the audio branch only and never ingests vibration
(thesis rule).

## 2. Non-goals

- No supervised state classifier (thesis TODO T5.3: interim milestone is unsupervised).
- No torch in the core install; BEATs (with torch/torchaudio) lives behind the optional
  extra `rowii-monitor[beats]`. No representation beyond handcrafted + frozen BEATs.
- No anomaly detection, no phase-shifter handling beyond a configurable state set
  (phase-shifter does not occur in the June-25 recordings).
- No SCADA input to the detector at run time (SCADA is ground truth only).
- No processing of the 2026-06-12 pre-flight sample or the 24 days of Betriebsdaten
  outside the June-25 window.

## 3. Data

**Source:** `~/Downloads/illwerke-250526-analysis` (canonical tree per its MANIFEST.md).
All sensor files are Gantner **UDBF v1.07** binaries.

| Set | Files | Size | Content |
|---|---|---|---|
| `20260626 Messung/TU/` | 48 | 25 GB | Turbine run 2026-06-25 04:15–06:27 UTC, 12-min segments × 4 streams |
| `20260626 Messung/PU/` | 21 | 9.6 GB | Pump runs 2026-06-25 (morning ~09:08–09:20, afternoon ~13:44–14:32 local) |
| `20260626 Messung/Betriebsdaten/` | 12 of 605 | ~55 MB | Hourly SCADA files for 2026-06-25 00:00–11:00 (~30 channels @ 10 Hz: speed, active power, guide-vane position, …) |
| `Sensor_Anordnung_15062026.xlsx` | 1 | — | stream index → physical sensor mapping |
| `ROWII_Leistung_{PU,TU}.jpg`, `MANIFEST.md` | 3 | — | power screenshots (backup GT), provenance |

Streams per burst: `RAWGeneratorMic__0`, `RAWTurbineMic__1` (~50 kHz, Pa, 9 mics on two
rings), `RAWGeneratorVib__2`, `RAWTurbineVib__3` (~10 kHz, m/s², tri-axial). Known data
reality (from the June-26 first-results analysis): 2 of 4 accelerometers deliver no data
(`GenVib0`, `TurVib0`); channel liveness must be detected, logged, and dead channels
excluded automatically.

**Copy plan:** `scripts/copy_data.py` copies exactly the table above to
`<DATA_ROOT>` (default `~/AI Workspace/master-thesis/data/illwerke-250526/`, ~35 GB,
disk verified: 493 GB free), preserving relative paths, verifying file count + sizes,
and writing a `copy_manifest.json` (source, per-file size, sha256 optional flag). The
repo never contains data; `DATA_ROOT` comes from an env var or `.env` (documented in
README), with a `data/` fallback inside the repo gitignored.

**Known coverage caveat:** Betriebsdaten end 11:00 on June 25. The PU afternoon bursts
(~13:44–14:32 local) therefore lack SCADA coverage; they are processed but excluded from
GT metrics and marked `gt=unknown` (the `ROWII_Leistung_PU.jpg` screenshot serves as a
qualitative cross-check only). Timestamp convention: filenames carry local time (CEST),
UDBF headers carry UTC; the UTC header timestamp is authoritative everywhere.

## 4. Architecture

```
src/rowii/
├── config.py             # dataclasses: PathsConfig (DATA_ROOT), WindowConfig,
│                         #   FeatureConfig, GtRules, DetectConfig; YAML override optional
├── io/
│   ├── udbf.py           # fresh UDBF v1.07 reader: parse_header() -> UdbfHeader
│   │                     #   (channels, names, units, sample_rate, t0_utc),
│   │                     #   read_udbf(path) -> UdbfFile(header, data[np.float32/…])
│   └── dataset.py        # discover_recordings(DATA_ROOT) -> RecordingIndex
│                         #   (TU/PU bursts grouped into contiguous runs by stream+time;
│                         #    Betriebsdaten hourly index with overlap reconciliation)
├── signals/
│   ├── windows.py        # common 1-s window grid per run (UTC-aligned);
│   │                     #   iter_windows(stream, grid) handles rate differences
│   └── features.py       # Featurizer protocol: (window) -> np.ndarray + feature names.
│                         #   AudioFeaturizer: log-RMS, band energies at shaft (6.25 Hz),
│                         #   blade-pass (43.75 Hz), guide-vane-pass (125 Hz) ± tolerance,
│                         #   octave-band log-energies, spectral centroid/rolloff, per ring.
│                         #   VibFeaturizer: per live axis RMS, band energies, kurtosis.
│                         #   FusionFeaturizer = concat(audio-branch, vib) after z-score.
│                         #   BeatsFeaturizer (extra [beats], audio only): 50 kHz -> 16 kHz
│                         #   resample, 128-d log-Mel fbank, frozen BEATs encoder, one
│                         #   pooled embedding per window; checkpoint path via config
│                         #   (BEATS_CHECKPOINT); fresh device helper (MPS > CUDA > CPU,
│                         #   env override), fresh-written per reuse policy.
├── scada/
│   ├── channels.py       # Betriebsdaten channel selection by name pattern
│   │                     #   (speed / active power / guide-vane; names resolved from
│   │                     #    UDBF header at first read, pinned in config thereafter)
│   └── labels.py         # rule-based GT per window: standstill | turbine | pump |
│                         #   transition (+ load bin sub-label within turbine/pump);
│                         #   rules: |n| < n_eps & |P| < P_eps -> standstill;
│                         #   P > +P_eps at nominal speed -> turbine; P < -P_eps -> pump;
│                         #   |dP/dt| or |dn/dt| above ramp threshold, or ±buffer around
│                         #   any state change -> transition. All thresholds in GtRules.
├── state/
│   ├── cluster.py        # KMeansClusterer, GmmClusterer (fit_predict interface)
│   ├── smooth.py         # StickyHmmSmoother (hmmlearn GaussianHMM, params="mc",
│   │                     #   self-transition prior fixed, transmat NOT re-estimated)
│   ├── segments.py       # duration filter (min dwell), frames -> segment table
│   └── detect.py         # run_detection(features, cfg) -> (frame_labels, segments)
└── eval/
    ├── metrics.py        # ARI, Hungarian-matched macro-F1 + confusion matrix,
    │                     #   boundary deviation stats (s) per transition
    └── report.py         # markdown report + timeline plot (detected vs GT vs power)
scripts/
├── copy_data.py          # selective copy per §3
└── run_step1.py          # CLI (argparse): --run tu|pu --modality audio|vibration|fusion
                          #   --clusterer kmeans|gmm; writes results/<run>/<variant>/
tests/                    # see §7
```

Cluster→state naming happens only in evaluation (Hungarian matching against GT); the
detector itself outputs anonymous cluster IDs plus segments, exactly as the thesis
frames the unsupervised interim milestone.

## 5. Pipeline defaults

1-s windows, no overlap (per-second features as in the June-26 first analysis).
Features z-scored per run. KMeans k = |expected states in run| (TU: standstill,
turbine, transition → k sweep 3–6 reported; load sub-structure appears as extra
clusters and is merged by the HMM/duration stage or reported as sub-clusters). Sticky
HMM self-transition 0.98 (config), duration filter min dwell 5 s (config). Run grid
per recording (TU, PU): variants {audio-handcrafted, audio-beats, vibration,
fusion-handcrafted, fusion-beats} × clusterer {kmeans, gmm} = 20 runs total; all fast
(BEATs inference on ~2.3 h audio runs in minutes on MPS). Milestone order: handcrafted
end-to-end first, BeatsFeaturizer second.

## 6. Error handling

- UDBF: wrong magic/version, truncated payload, checksum mismatch → typed exceptions
  (`UdbfFormatError`) with file path and offset; never silent.
- Dead/missing channels: near-zero variance or absent stream → warning + exclusion,
  recorded in the run report (expected: GenVib0/TurVib0).
- Betriebsdaten overlaps (restarted exports; MANIFEST caveat) → reconcile by UDBF start
  timestamp, prefer the longer file, log dropped duplicates.
- Windows without SCADA coverage → `gt=unknown`, excluded from metrics, counted in report.
- Rate drift/gaps inside a stream → window skipped + counted; hard fail if >5 % of a run.

## 7. Testing (TDD, per project standards)

- Unit tests per module. UDBF reader tests use synthetic fixture files built by a
  minimal test-only writer (`tests/fixtures/udbf_builder.py`) — no real data in tests.
- GT-rule tests: synthetic SCADA traces covering every rule branch incl. ramp buffer.
- Detection integration test: synthetic 3-state feature sequence with known boundaries
  → ARI 1.0 expected, duration filter removes injected flicker.
- Eval tests: Hungarian matching, boundary metrics on constructed cases.
- BeatsFeaturizer tests run against a stub encoder (protocol-level, no checkpoint, no
  torch needed in CI); real-checkpoint smoke test under `@pytest.mark.data`.
- A `@pytest.mark.data` tier runs a real-data smoke test locally (skipped when
  `DATA_ROOT` unset / in CI).
- Tooling: Python 3.12, pyproject (src layout), ruff (line length 100), mypy, pytest.
  Core dependencies: numpy, scipy, scikit-learn, hmmlearn, pandas, matplotlib, openpyxl
  (sensor xlsx), python-dotenv. Optional extra `[beats]`: torch, torchaudio.

## 8. Deliverables & acceptance

- `results/<run>/<variant>/`: segments.csv, frame_labels.parquet, report.md,
  timeline.png (detected states vs SCADA GT vs power curve).
- Campaign-1 summary table: ARI / macro-F1 / boundary median |Δt| per variant
  (modality × audio-branch featurizer) × clusterer × recording, so the handcrafted-vs-
  BEATs and audio-vs-vibration-vs-fusion questions are answered in one table.
- Acceptance: TU fusion variant reaches ARI ≥ 0.9 against SCADA GT on covered windows;
  all tests green; repo clones + installs + runs from README alone. No result from the
  v1 prototype counts as evidence for this data: nominal speed, machine frequencies,
  state count k (via sweep), channel liveness, and GT thresholds are all verified on
  the June-25 recordings themselves before results are reported.
- GitHub: private repo `StefanKrum/rowii-monitor`, pushed from day one, conventional
  commits, no AI attribution.

## 9. Thesis touchpoint (optional follow-up, not part of this implementation)

Evaluation chapter Campaign-1 sentence can later state the input-modality comparison
explicitly ("recovered reliably from audio, vibration, and their combination"). One-line
edit, deferred until results exist.
