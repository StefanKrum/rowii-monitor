# Step-2 Package 4: TF-C Industrial-Pretraining Transfer Pole — Design Spec

**Date:** 2026-07-16 · **Thesis anchors:** Design chapter § Representation (the TF-C
pole: "pre-trained on public industrial sound such as MIMII, TF-C embodies the
industrial-pre-training hypothesis"; vibration branch: "TF-C applied natively to the
vibration signal, pre-trained on public vibration corpora, namely the CWRU and
Paderborn bearing-fault datasets") and § Deployment ("self-supervised pretraining is
a one-off offline step on public data").
**Precondition (met):** package 3 merged (main 897e3ec); scorer registry, feature
cache, conformal harness all live.

## 1. Questions this package answers

1. Does a representation pre-trained on PUBLIC INDUSTRIAL sound (MIMII) transfer to
   the pump-turbine better than handcrafted features or general-audio BEATs — the
   design's "does the KIND of pre-training data matter" question?
2. Does the same SSL objective, pre-trained natively on public VIBRATION corpora
   (CWRU + Paderborn), help the vibration branch — the design's complementary
   vibration-native question?

## 2. Binding design decisions

### D1 — Compact TF-C implementation (`src/rowii/tfc/`)

Time-Frequency Consistency (Zhang et al. 2022, NeurIPS): two encoders embed the
time-domain view and the frequency-domain view of the same 1-s window; the
contrastive objective pulls a window's time/freq embeddings together (consistency)
and apart from other windows' (NT-Xent), with augmentations per view (time:
jitter + scaling + masking; freq: magnitude perturbation + band masking). This
package ships a COMPACT, honest version — the pole must exist and be trained
reproducibly on public data within an MPS-hours budget; SOTA parity is a non-goal
and the compactness is documented wherever results are reported.

- Input geometry (both branches): 1-s windows resampled to **8 kHz** (8000 samples,
  mono). Time encoder: 1-D CNN (4 conv blocks stride 4, channels 32-64-128-128,
  BatchNorm+ReLU, global average pool) → 128-d. Freq view: `|rfft|` of the same
  window (4001 bins) → same CNN architecture (1-D over frequency) → 128-d.
  Projection heads (128→64) for the loss; the EMBEDDING handed downstream is the
  concatenated pooled encoder outputs (time ⊕ freq = 256-d, pre-projection, the
  paper's convention for downstream use).
- Module layout mirrors the BEATs precedent exactly: `src/rowii/tfc/model.py`
  (eager torch, plain nn.Modules), `src/rowii/tfc/wrapper.py` (lazy torch,
  checkpoint loading, `extract_embeddings(stack, rate_hz)` API, `best_device()`
  reuse), unit-tested with a stub encoder like `BeatsFeaturizer`'s tests.

### D2 — Public-corpus acquisition (`scripts/download_corpora.py`)

- **MIMII** (audio branch): Zenodo record for the 0 dB SNR "pump" machine type
  (topical match to the target machine) — normal segments only (SSL pre-training is
  unsupervised on normal-dominated data; abnormal files are excluded by filename
  convention and that exclusion is logged). ~10 GB zip, sha256-verified, extracted
  under `data/public/mimii/pump_0db/`.
- **CWRU** (vibration): the standard normal-baseline + drive-end 12 kHz files from
  the CWRU bearing data center mirrors (~200 MB), sha256-verified, under
  `data/public/cwru/`.
- **Paderborn (KAt)** (vibration): a scripted SUBSET (the healthy K001-K006 sets,
  ~2-4 GB) if the direct download works headlessly; otherwise the script prints
  precise manual-download instructions and the pre-train script accepts whatever
  subset is present — the design names CWRU AND Paderborn, so attempt both, but a
  documented CWRU-only fallback is acceptable for the vibration pole (recorded in
  the results notes if it happens).
- All downloads land under `data/public/` (gitignored via the existing data/ rule —
  verify), each with a `MANIFEST.json` (url, sha256, license note: MIMII CC BY-SA
  4.0, CWRU academic-free, Paderborn CC BY-NC 4.0), `--dry-run` prints the plan.

### D3 — Pre-training (`scripts/pretrain_tfc.py`)

- `--corpus mimii|bearings`, `--epochs 40`, `--batch-size 256`, `--lr 1e-3`,
  `--seed 7`, `--limit-clips N` (dev subsampling), `--out models/pretrained/tfc/`.
- Corpus loaders: stream WAV (MIMII, 16 kHz) / MAT (CWRU) / MAT-or-CSV (Paderborn)
  files, cut into 1-s windows, resample to 8 kHz (scipy.signal.resample_poly),
  per-window standardize. Loaders unit-tested on synthetic fixture files (tiny
  generated wav/mat), never on real downloads in CI.
- Training: NT-Xent + consistency term per the paper's structure (temperature 0.2);
  checkpoint = dict(cfg, model state, corpus manifest hash, epochs) saved as
  `tfc_audio.pt` / `tfc_vib.pt`. Deterministic seeding; MPS with CPU fallback;
  progress logging per epoch; expected budget ≤ ~2-4 h per corpus on MPS at the
  default settings (measured and logged).

### D4 — Pipeline integration: variants `audio-tfc` and `vibration-tfc`

- `pipeline._streams_for_variant`: `audio-tfc` → the audio streams (mirror `audio`),
  `vibration-tfc` → the vibration streams (mirror `vibration`).
- Featurizer `TfcFeaturizer` (in `src/rowii/tfc/wrapper.py`): resample each 1-s
  window's mono mix to 8 kHz, run the frozen encoders, emit 256-d embeddings;
  checkpoint paths via env `ROWII_TFC_AUDIO_CHECKPOINT` / `ROWII_TFC_VIB_CHECKPOINT`
  (config fields mirroring `beats_checkpoint`); guard scripts with a
  `_import_beats_or_exit`-style hint when torch or the checkpoint is missing.
- Cache: fingerprint must include the checkpoint path (mirror the BEATs rule);
  logmel-style exclusion from `run_step1 --variant all` does NOT apply — TF-C IS a
  legitimate state-detection candidate (256-d, well-conditioned for the GMM), so
  add both variants to `_CONCRETE_VARIANTS` in run_step1 AND to run_step2's choices.

### D5 — Evidence (execution)

- Step-1 state detection: 010726-tu_ph_tu × {audio-tfc, vibration-tfc} × kmeans
  (does industrial SSL transfer for state separation where BEATs failed?).
- Step-2 within-day: 3 SCADA days × {audio-tfc, vibration-tfc} × {knn, mahalanobis}
  × both conditionings — table vs handcrafted/BEATs/logmel-AE FARs.
- Cross-day per-state: the 6 day pairs × audio-tfc × knn (does industrial
  pre-training stabilize the day shift?).
- Candidate overlap: audio-tfc-knn vs fusion-knn and vs audio-beats-knn (which
  moments does the industrial representation flag?).
- README package-4 section + research note; compactness caveat stated wherever
  TF-C numbers appear.

## 3. Non-goals

Fine-tuning/LoRA of TF-C or BEATs (package 5), SOTA-parity TF-C, full-MIMII scale,
multi-machine-type MIMII pre-training, cross-attention fusion (package 5),
serialization (package 6).

## 4. Acceptance

- Unit tests: TF-C model shapes/loss (loss decreases on synthetic data; consistency
  term nonzero; determinism); wrapper contract incl. stub-encoder tests + poison
  probe (module imports without torch); corpus loaders on synthetic fixtures
  (windowing, resampling, normal-only filtering); featurizer dispatch + cache
  fingerprint inclusion; downloads NEVER run in tests.
- Real execution: both checkpoints trained (budgets logged); all D5 sweeps produced;
  digest + README + note.
- Gates green (full suite incl. data marks where cheap); conventional commits on
  `feat/step2-package4-tfc`; per-task adversarial review + final whole-branch
  review; PR at the end.
- No partner values adopted; licenses recorded in MANIFEST.json.
