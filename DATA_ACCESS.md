# Data access

## What is not in this repository

This repository never commits sensor data. The full pipeline was developed
and evaluated against audio, vibration, and SCADA recordings from the
Rodundwerk II pump-turbine, collected across six measurement days
(`illwerke-250526`, `illwerke-270626`, `illwerke-290626`, `illwerke-300626`,
`illwerke-010726`, `illwerke-080726`) — roughly 450 GB of raw microphone,
accelerometer, and SCADA (Betriebsdaten) data in total. The `illwerke-300626`
day was delivered on 2026-08-17/18, after the analysis was frozen, and is
replayed as a held-out post-freeze day (see `data/illwerke-300626/MANIFEST.md`
in the thesis workspace for its packaging and gap notes).

This dataset is **proprietary plant data owned by illwerke vkw AG** and is
**not redistributable**. It is not included in this repository, in its
release archives, or anywhere on GitHub, and it will not be added in the
future.

## Requesting access

Access to the underlying recordings can be requested on a reasonable basis
through the research partners at the University of St. Gallen (HSG) who
coordinated the measurement campaign with illwerke vkw AG. There is no
self-service download; approval and data transfer are handled case by case
by the partners, subject to their data-sharing agreement with illwerke vkw
AG.

## What works without the data

The repository is designed so that almost everything in it can be inspected,
run, and verified without ever obtaining the plant recordings:

- **The full test suite (1,547 tests) runs without any real data.** Every
  test either operates on small, synthetic, in-memory fixtures (hand-built
  arrays, mocked file structures, monkeypatched I/O seams) or is marked
  `@pytest.mark.data` and skips cleanly when `ROWII_DATA_ROOT` does not point
  at a real directory. `pytest tests/ -q` is the command used in CI-style
  checks and requires no data download or network access.
- **The public-data scarcity study is independently reproducible.** The
  detection-performance-vs-data-scarcity result reported for the thesis uses
  the public MIMII dataset (plus CWRU and Paderborn bearing corpora for the
  vibration-side pretraining), not the proprietary plant recordings.
  `scripts/download_corpora.py` fetches and verifies (sha256-pinned) these
  public corpora into `data/public/`, and `scripts/scarcity_detection.py`
  reproduces the scarcity curve from them end to end.
- **Every algorithm, CLI, and analysis script can be read and reasoned about
  in isolation** — the source under `src/rowii/` and `scripts/` has no
  hidden dependency on data contents beyond the documented layout below.

## Data layout expected by the code

Anyone who *does* obtain the recordings should point the `ROWII_DATA_ROOT`
environment variable (see `.env.example`; copy to `.env`) at a **parent
root** containing one subdirectory per measurement day:

```
<ROWII_DATA_ROOT>/
├── illwerke-250526/<date> Messung/{TU,PU,Betriebsdaten}/...
├── illwerke-270626/<date> Messung/PU_PH_PU_PH_PU_PH/...
├── illwerke-290626/<date> Messung/{TU,PU,Betriebsdaten}/...
├── illwerke-010726/<date> Messung/{PU,TU1,TU2,TU_PH_TU,Betriebsdaten}/...
└── illwerke-080726/<date> Messung/{PU,ST,Betriebsdaten}/...   # controlled-event campaign (hammer strikes)
```

Each day tree carries its own audio/vibration burst files (Gantner UDBF v1.07
binaries, four streams per burst: two microphones + two tri-axial
accelerometers) and, where delivered, hourly SCADA export files
(`Betriebsdaten/`). The full layout, per-day coverage, and known gaps are
documented in the "Data layout" section of `README.md`.

`rowii.io.dataset.discover` also accepts a single day tree directly (legacy,
backward-compatible single-day mode) for anyone who only has one delivery.

## Intake check for a new data delivery

Once `ROWII_DATA_ROOT` is set, run the verification CLI before anything else:

```bash
python scripts/verify_data_facts.py --help
```

`scripts/verify_data_facts.py` is the intake check: it runs small, targeted,
read-only probes directly against the raw files (generator-mic level
behaviour at known reference minutes, per-file channel-liveness checks, and
an SCADA-timebase cross-check against the audio clock) to confirm a new
delivery matches the assumptions the rest of the pipeline depends on, before
any downstream script is trusted to run on it. It never modifies the source
data and never depends on `rowii.pipeline`/`rowii.signals`, so it can be run
against a data tree in complete isolation.

Once the intake check passes, `scripts/copy_data.py` and `scripts/run_step1.py`
are the next commands in the reproduction sequence — see the "Reproducing
the thesis results" section of `README.md`.
