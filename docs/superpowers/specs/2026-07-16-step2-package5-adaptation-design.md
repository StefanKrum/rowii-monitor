# Step-2 Package 5: Adaptation & Compactness — Design Spec

**Date:** 2026-07-16 · **Thesis anchors:** Design chapter § Adaptation (frozen default
vs LoRA vs full fine-tune as upper-capacity reference; KD toward deployment),
§ Multimodal detection (the third fusion level: "a lightweight cross-attention head
... operates on precomputed embeddings and stays small enough to train on the scarce
data available" — "within scope rather than deferred"), § Deployment and compactness
(hard on-premise constraint; "a distillation-free alternative is also tested, namely
post-training quantization ... to 8-bit"; "the preprocessing cost of feature
extraction belongs in the budget alongside inference").
**Precondition (met):** package 4 merged (main b6f471f); BEATs vendored + cached;
logmel variant; TF-C infra as precedent for torch training scripts.

## 1. Questions this package answers

1. Does parameter-efficient adaptation (LoRA) of the frozen BEATs encoder on target
   normal data improve Step-1/Step-2 behaviour, and how does it compare to the
   full-fine-tune upper-capacity reference? (Design: frozen is the default; the
   axis must be measured, not assumed.)
2. Can BEATs-level detection quality be shipped compactly — via a distilled student
   and via post-training INT8 quantization — and what does each cost in
   size/latency/memory INCLUDING preprocessing?
3. Does a lightweight cross-attention fusion head (third fusion level) beat
   feature-level and score-level fusion?

## 2. Binding design decisions

### D1 — Adaptation objective (shared by LoRA and full-FT): masked-patch reconstruction

BEATs' original pre-training objective (discrete-label distillation via its
tokenizer) is not reproducible here. The adaptation objective is therefore a
documented PROXY: mask 30% of the input fbank patches, reconstruct the masked
patches' fbank values with a small linear head on the encoder output, MSE on masked
positions only (MAE-style). Self-supervised, uses target NORMAL windows only, and is
stated as a proxy wherever adapted-model results appear. Module
`src/rowii/adapt/objective.py` (eager torch, `_recon_models.py` precedent).

### D2 — LoRA injection (`src/rowii/adapt/lora.py`)

Low-rank adapters (rank 8, alpha 16, dropout 0.0) injected into the QUERY and VALUE
projections of every self-attention block of the vendored BEATs encoder (the
literature-cited placement); base weights frozen; only adapter params train.
Implementation wraps the existing q/v `nn.Linear`s with a `LoraLinear` (W x + B A x
scaling alpha/r) — no vendored-code edits; injection walks named modules and
replaces by attribute path (tested against a small stand-in transformer with the
same module naming shape, plus a `@pytest.mark.data`-free structural test against
the real vendored class WITHOUT loading checkpoint weights). Merged export:
`merge_lora()` folds adapters back into plain Linears so the adapted encoder loads
through the EXISTING `load_beats_model` path (checkpoint stays format-compatible:
{"cfg", "model"} with BEATs' own keys) — no new featurizer code needed for scoring.

### D3 — Target-normal training windows (`src/rowii/adapt/target_windows.py`)

Iterator over a run's PRIMARY MIC stream gantner files, yielding 1-s windows
resampled to 16 kHz (BEATs' input rate; scipy resample_poly), LEAKAGE-AWARE: only
windows from the run's calibration-side segments (the same
`split_by_segments(segment_ids, valid, 0.5, seed=7)` top split every sweep uses —
adaptation never sees scoring-side segments; documented in every adapted result).
Reuses `read_gantner` + the established windowing; per-window standardize.

### D4 — Adaptation script (`scripts/adapt_beats.py`)

`--mode lora|full --run 010726-tu_ph_tu --epochs 5 (lora) / 2 (full) --batch-size 16
--lr 1e-4 (lora) / 1e-5 (full) --seed 7 --max-windows 8000 --out
models/adapted/`. Loads the vendored BEATs from ROWII_BEATS_CHECKPOINT, applies D2
(lora) or unfreezes all (full), trains D1's objective on D3's windows, saves
`beats_lora_<run>.pt` / `beats_ft_<run>.pt` as MERGED, format-compatible checkpoints
+ a sidecar json (mode, run, epochs, windows, final loss, elapsed). Budgets logged;
MPS with CPU fallback; deterministic seeding (MPS caveat documented as before).
Env integration: pointing `ROWII_BEATS_CHECKPOINT` at an adapted checkpoint turns
every existing `audio-beats`/`fusion-beats` variant into the adapted evaluation —
NO new variants needed (the cache fingerprint already keys on the checkpoint path;
golden-pin unaffected since the payload shape is unchanged). This is the package's
integration trick: adaptation evidence reuses the entire existing sweep machinery.

### D5 — Knowledge distillation (`scripts/distill_beats.py` + `src/rowii/adapt/student.py`)

Teacher = frozen-BEATs embeddings ALREADY CACHED per run (results/cache/
<run>--audio-beats.npz — zero teacher compute). Student = compact CNN on the
logmel variant's patches (reshape 49x64, ~0.5-1 M params, output 768-d), trained
with MSE against the teacher embeddings of the SAME windows (calibration-side only,
D3's leakage rule; the logmel cache provides the inputs — also zero extraction).
Checkpoint `student_<run>.pt` (own format: {"cfg","model","teacher_variant",
"run","epochs"}). New variant `audio-student` (featurizer mirrors TfcFeaturizer's
lazy pattern; env `ROWII_STUDENT_CHECKPOINT`; fingerprint scoped like tfc — WITH
golden-pin update done consciously per the banked lesson: the pin constant changes
in the same commit that documents the cache-migration decision [none needed: new
variant, no existing caches]).

### D6 — Post-training INT8 quantization (`scripts/quantize_beats.py`)

`torch.ao.quantization.quantize_dynamic` (Linear layers, qint8) on the frozen BEATs
encoder → `beats_int8.pt` (torch.save of the quantized module — NOT
state-dict-compatible; loaded by a dedicated path). Quantized inference is CPU-only
(documented; that IS the deployment target — the design's on-premise server has no
GPU). Evidence: embedding drift (cosine similarity to fp32 embeddings on real
cached windows), size on disk, latency (D7 harness), and one within-day FAR parity
sweep via a `ROWII_BEATS_INT8=1`-style featurizer branch... SIMPLER, binding:
`BeatsFeaturizer` gains an optional `quantized_model_path` constructor arg wired to
env `ROWII_BEATS_INT8_CHECKPOINT`; when set, it loads the quantized module (CPU)
instead of `load_beats_model`. Variant name stays `audio-beats`; the fingerprint
must include the int8 path when set (scoped; golden pins updated consciously —
same-commit documentation, no existing-cache migration needed since unset ⇒
payload unchanged).

### D7 — Compactness/latency harness (`scripts/benchmark_inference.py`)

Per featurizer configuration {handcrafted audio, logmel, BEATs fp32, BEATs INT8
(CPU), TF-C audio, student}: model size on disk, parameter count, peak RSS delta,
and per-window wall latency at batch 1 and batch 256 — measured END-TO-END from raw
50 kHz windows (i.e. INCLUDING resampling/fbank/logmel preprocessing, per the
design's explicit budget rule), on CPU and (where supported) MPS, N=200 windows of
real cached... raw windows come from one real gantner file (read-only) — a
`@pytest.mark.data` path; the script itself is unit-tested with synthetic windows.
Output: `results/benchmarks/inference.csv` + markdown table.

### D8 — Cross-attention fusion head (`src/rowii/fusionx/` + `scripts/train_xattn.py`)

Third fusion level (design: within scope). Head: single cross-attention block
(audio embedding 768-d [frozen BEATs] as query, vibration handcrafted features
as key/value after a linear lift to 128-d; 4 heads; output 128-d joint embedding;
~0.3 M params). Training objective (no labels available): symmetric InfoNCE
alignment — the audio and vibration views of the SAME window are the positive pair
(CLIP-style), calibration-side windows only (D3 rule), per-run. Scoring: kNN (k=1
cosine) on the joint embedding via the existing scorer machinery — integrated as an
orchestration view `--xattn-fusion` in run_step2 (within-day, fusion variant dirs,
mirroring `--score-fusion`'s placement pattern) with rows fisher-style per state +
the honest note that the head was trained on the calibration side (still disjoint
from scoring segments). Checkpoint `xattn_<run>.pt`, env `ROWII_XATTN_CHECKPOINT`.
Evidence: FAR table vs feature-fusion and score-fusion on the same days.

### D9 — Evidence (execution)

- Adapted encoders: Step-1 (010726, audio-beats variant under lora/ft checkpoints)
  + Step-2 within-day (3 days × {frozen, lora, ft} × knn) + candidate overlap
  frozen-vs-lora. Question: does adaptation fix BEATs' state-detection collapse the
  way industrial pretraining (TF-C) did?
- Student: same Step-1 + within-day sweeps (variant audio-student) + D7 row.
- INT8: embedding drift + D7 rows + one within-day FAR parity sweep (010726).
- Cross-attention: FAR vs feature/score fusion (3 days, knn).
- D7 compactness table = the deployment-chapter exhibit.
- README package-5 section + research note; proxy-objective and MPS-determinism
  caveats wherever adapted results appear.

## 3. Non-goals

TF-C adaptation (frozen by design), multi-run/pooled adaptation corpora, serving
infrastructure (package 6), NAS/architecture search, fp16 paths, source
localisation.

## 4. Acceptance

- Unit tests: LoRA injection (adapter shapes; base frozen; merge_lora equivalence:
  merged forward == unmerged forward within tolerance; injection on the stand-in
  transformer + structural test on the real vendored class), masked-patch objective
  (mask fraction; loss on masked positions only; decreases), target-window iterator
  (leakage: only calibration-side segments — mutant-style assertion; 16 kHz length),
  student (distillation loss decreases; featurizer contract 768-d), INT8 loader
  branch (fingerprint scoping + golden updates), xattn head (shapes; InfoNCE
  decreases; deterministic), benchmark harness (synthetic smoke).
- Real execution per D9; gates green; conventional commits on
  `feat/step2-package5-adaptation`; per-task adversarial review + final
  whole-branch review; PR at the end.
- Every adapted/distilled/quantized result carries its caveat block (proxy
  objective / data-floor / CPU-only as applicable). No partner values.

## Amendment A1 (2026-07-16, post-T3 review): D1 objective retargeted to native token-level MAE

The original D1 (frame-level masking with a frame-preserving encoder_forward) is
structurally incompatible with BEATs' native forward (patch_embedding conv
patchifies 98 fbank frames into 48 tokens); T3's frozen-random-linear bridge
satisfied D1's letter but trained the adapters on an input distribution decoupled
from the deployed inference path (reviewer-proven: bridge-trained deltas perturb
native embeddings measurably in an objective-irrelevant direction). D1 is
therefore AMENDED: the adaptation objective is **native token-level masked
reconstruction** — reuse the model's own preprocess → patch_embedding →
layer_norm → post_extract_proj (all frozen), mask ~30% of the resulting patch
tokens (zeroed rows, per-sample random via the seeded generator), run the
(LoRA-adapted or fully-trainable) encoder, and reconstruct the PRE-MASK token
embeddings with a Linear(encoder_dim, encoder_dim) head, MSE on masked positions
only (latent-target MAE; documented choice over pixel targets: no patch-pixel
bookkeeping, same self-supervision signal). The frame-level `masked_patch_loss`
remains in `rowii.adapt.objective` for non-patchifying encoders, with its caveat
extended to name this incompatibility. Everything else in T3 (CLI, mode prep,
optimizer scoping, merged export, verification, sidecar) stands unchanged.
