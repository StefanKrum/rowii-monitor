# Step-2 Package 5 Implementation Plan — Adaptation & Compactness

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the design chapter's adaptation axis (frozen vs LoRA vs full fine-tune of BEATs on target normals), ship the compactness pair (distilled student + post-training INT8), the third fusion level (cross-attention head), and the deployment latency/memory harness.

**Architecture:** LoRA wraps the vendored BEATs' `self_attn.q_proj`/`v_proj` Linears by attribute replacement (no vendor edits) and exports MERGED, format-compatible checkpoints so all adaptation evidence reuses the existing `audio-beats` machinery via `ROWII_BEATS_CHECKPOINT`. The masked-patch reconstruction objective is a documented proxy. The student distills from ALREADY-CACHED teacher embeddings against logmel-cache inputs (zero teacher compute). INT8 is a `BeatsFeaturizer` alternate-load branch (CPU). The cross-attention head trains CLIP-style on natural audio↔vibration window pairs and integrates as a run_step2 view like `--score-fusion`.

**Tech Stack:** torch (existing extra), torch.ao dynamic quantization, existing cache/sweep/conformal machinery, psutil-free RSS via `resource.getrusage`.

**Design authority:** `docs/superpowers/specs/2026-07-16-step2-package5-adaptation-design.md` (D1–D9). Read it first.

## Global Constraints

- Branch: `feat/step2-package5-adaptation` (from `main` = b6f471f).
- Gates after every task: `.venv/bin/python -m pytest tests/ -q -m "not data"`; `.venv/bin/ruff check .`; `.venv/bin/mypy src scripts` — all clean.
- Torch import discipline: eager only in dedicated model modules (`src/rowii/adapt/objective.py`, `src/rowii/adapt/lora.py`, `src/rowii/adapt/student.py` model part, `src/rowii/fusionx/model.py`), lazy everywhere else; poison-probe convention as established.
- Leakage rule (D3): every training/adaptation consumer uses ONLY calibration-side segments of the top split `split_by_segments(segment_ids, valid_mask, 0.5, seed=7)` — test-pinned per consumer.
- Cache-fingerprint golden pins: any payload change updates the pinned constants IN THE SAME COMMIT with an explicit cache-migration note (banked lesson).
- Adapted/distilled/quantized results always carry their caveat (proxy objective / CPU-only / trained-on-calibration-side).
- No real-data runs or checkpoint training inside implementer/reviewer tasks (synthetic only); execution is Task 8. Exception: none this package (unlike P4-T3).
- Conventional commits; explicit `git add`; no Co-Authored-By; no partner values.

---

### Task 1: LoRA injection + masked-patch objective (`src/rowii/adapt/`)

**Files:**
- Create: `src/rowii/adapt/__init__.py` (empty), `src/rowii/adapt/lora.py`, `src/rowii/adapt/objective.py`
- Test: `tests/test_adapt_lora.py`, `tests/test_adapt_objective.py`

**Interfaces:**
- Consumes: vendored BEATs structure (`encoder.layers[i].self_attn.q_proj` / `.v_proj`, plain `nn.Linear` — verified); `rowii.signals.beats_model.load_beats_model` (checkpoint format {"cfg", "model"}).
- Produces:
  - `lora.py` (eager torch): `class LoraLinear(torch.nn.Module)` wrapping a base Linear: `forward(x) = base(x) + (alpha/r) * B(A(x))`, `A: Linear(in, r, bias=False)` init kaiming-uniform, `B: Linear(r, out, bias=False)` init zeros (so injection starts as identity), base params frozen (`requires_grad_(False)`); `inject_lora(module: torch.nn.Module, r: int = 8, alpha: int = 16, target_names: tuple[str, ...] = ("q_proj", "v_proj")) -> int` — walks `named_modules()`, replaces every `nn.Linear` attribute whose NAME is in target_names AND whose parent path contains `self_attn`, returns count injected; `merge_lora(module) -> int` — folds `W += (alpha/r) * B.weight @ A.weight` into each base Linear and swaps the plain Linear back in (returns count merged); `lora_parameters(module) -> Iterator[nn.Parameter]` — only A/B params.
  - `objective.py` (eager torch): `masked_patch_loss(encoder_forward: Callable[[torch.Tensor], torch.Tensor], fbank: torch.Tensor, head: torch.nn.Linear, mask_frac: float = 0.3, generator: torch.Generator | None = None) -> torch.Tensor` — masks a random mask_frac of fbank TIME frames (zeroing them), runs the encoder on the masked input, projects encoder outputs through `head` back to fbank-frame dimension, MSE on the MASKED frames only. (Frame-level masking rather than BEATs' internal patch tokens: documented simplification — the encoder consumes fbank frames; masking at the frame level keeps the objective independent of vendored internals.)
- Tests (complete in the plan):

```python
# tests/test_adapt_lora.py — CPU-forced autouse fixture like tests/test_recon.py
class _TinyAttnModel(torch.nn.Module):
    """Stand-in mirroring the vendored naming shape: layers[i].self_attn.{q,k,v}_proj."""
    def __init__(self):
        super().__init__()
        attn = torch.nn.Module()
        attn.q_proj = torch.nn.Linear(8, 8)
        attn.k_proj = torch.nn.Linear(8, 8)
        attn.v_proj = torch.nn.Linear(8, 8)
        layer = torch.nn.Module()
        layer.self_attn = attn
        self.layers = torch.nn.ModuleList([layer])
        self.unrelated = torch.nn.Linear(8, 8)

    def forward(self, x):
        a = self.layers[0].self_attn
        return a.q_proj(x) + a.v_proj(x) + a.k_proj(x) + self.unrelated(x)

def test_inject_targets_only_qv_under_self_attn():
    m = _TinyAttnModel()
    n = inject_lora(m, r=2)
    assert n == 2  # q_proj + v_proj; k_proj and unrelated untouched
    assert isinstance(m.layers[0].self_attn.q_proj, LoraLinear)
    assert isinstance(m.layers[0].self_attn.k_proj, torch.nn.Linear)
    assert isinstance(m.unrelated, torch.nn.Linear)

def test_injection_starts_as_identity():
    torch.manual_seed(0)
    m = _TinyAttnModel()
    x = torch.randn(4, 8)
    before = m(x).detach().clone()
    inject_lora(m, r=2)
    torch.testing.assert_close(m(x), before)  # B init zeros

def test_base_frozen_adapters_trainable():
    m = _TinyAttnModel()
    inject_lora(m, r=2)
    q = m.layers[0].self_attn.q_proj
    assert not q.base.weight.requires_grad
    lora_named = {id(p) for p in lora_parameters(m)}
    assert id(q.lora_a.weight) in lora_named and id(q.lora_b.weight) in lora_named

def test_merge_restores_plain_linear_and_forward():
    torch.manual_seed(1)
    m = _TinyAttnModel()
    inject_lora(m, r=2)
    # push adapters off zero
    for p in lora_parameters(m):
        torch.nn.init.normal_(p, std=0.1)
    x = torch.randn(4, 8)
    unmerged = m(x).detach().clone()
    n = merge_lora(m)
    assert n == 2
    assert isinstance(m.layers[0].self_attn.q_proj, torch.nn.Linear)
    torch.testing.assert_close(m(x), unmerged, rtol=1e-5, atol=1e-6)

def test_structural_match_on_real_vendored_class():
    # constructs the real vendored TransformerEncoder config surface WITHOUT weights:
    # instantiate rowii.vendor.beats.BEATs.BEATs with its small default config if
    # feasible on CPU quickly (tiny encoder_layers via BEATsConfig override); assert
    # inject_lora count == 2 * encoder_layers. If instantiation needs >2s, mark slow
    # but keep it in the default suite (no data mark — no checkpoint load).
```

```python
# tests/test_adapt_objective.py
def test_loss_on_masked_frames_only():
    torch.manual_seed(0)
    fbank = torch.randn(2, 20, 16)          # (B, frames, mels)
    head = torch.nn.Linear(16, 16)
    calls = {}
    def encoder(x):
        calls["input"] = x.detach().clone()
        return x                              # identity encoder, dim-preserving
    gen = torch.Generator().manual_seed(7)
    loss = masked_patch_loss(encoder, fbank, head, mask_frac=0.3, generator=gen)
    assert loss.ndim == 0 and torch.isfinite(loss)
    masked = (calls["input"] == 0).all(dim=2)  # zeroed frames
    frac = masked.float().mean().item()
    assert 0.15 < frac < 0.45                  # ~0.3 of frames masked

def test_loss_decreases_with_training():
    torch.manual_seed(0)
    fbank = torch.randn(8, 20, 16)
    enc = torch.nn.Linear(16, 16)
    head = torch.nn.Linear(16, 16)
    opt = torch.optim.Adam([*enc.parameters(), *head.parameters()], lr=1e-2)
    gen = torch.Generator().manual_seed(7)
    first = masked_patch_loss(lambda x: enc(x), fbank, head, generator=gen).item()
    for _ in range(50):
        g = torch.Generator().manual_seed(7)
        opt.zero_grad()
        loss = masked_patch_loss(lambda x: enc(x), fbank, head, generator=g)
        loss.backward(); opt.step()
    g = torch.Generator().manual_seed(7)
    assert masked_patch_loss(lambda x: enc(x), fbank, head, generator=g).item() < first
```

- [ ] Steps: failing tests → verify → implement lora.py + objective.py (LoraLinear stores `base`, `lora_a`, `lora_b`, `scale = alpha / r`; inject walks `list(module.named_modules())` collecting (parent, attr) pairs first, then replaces; merge computes `base.weight.data += scale * (lora_b.weight @ lora_a.weight)`) → gates → commit `feat: LoRA injection and masked-patch adaptation objective`.

---

### Task 2: Leakage-aware target-window iterator

**Files:** Create `src/rowii/adapt/target_windows.py`; Test `tests/test_target_windows.py`.

**Interfaces:**
- Consumes: `rowii.io.gantner.read_gantner`, `rowii.io.dataset.Run` (files per stream), `rowii.anomaly.references.split_by_segments`, `rowii.pipeline` helpers for segment ids (READ how `PreparedRun.segment_ids` is derived; simplest faithful route: `prepare_run(run, "audio", cfg, use_cache=True)` gives valid_mask + segment_ids + grid — reuse it rather than re-deriving), resample helper precedent (`rowii.tfc.wrapper`).
- Produces: `iter_target_windows(run, cfg, *, target_hz: int = 16_000, seed: int = 7, max_windows: int | None = None) -> Iterator[np.ndarray]` — computes the top split (0.5, seed=7) on the audio variant's segment_ids/valid_mask; iterates ONLY calibration-side windows; for each, reads the corresponding 1-s slice from the primary mic stream files (via the grid's window edges and each file's true-UTC span), mono-mix, resample to 16 kHz (pad/trim + resample_poly), per-window standardize, yield float32 (16000,). Deterministic order (ascending window index); `max_windows` truncates. NOTE: reading raw slices per window re-opens gantner files — batch by FILE (group windows by containing file, read each file once). Tests: synthetic gantner tree via `tests/fixtures/gantner_builder` (the established fixture): leakage assertion (no yielded window index in the scoring side — mutant-style: monkeypatch split_by_segments to a known partition and assert exclusion), 16 kHz length, determinism, max_windows.

- [ ] Steps: failing tests → verify → implement → gates → commit `feat: leakage-aware target-normal window iterator for adaptation`.

---

### Task 3: Adaptation script (LoRA / full fine-tune, merged export)

**Files:** Create `scripts/adapt_beats.py`; Test `tests/test_adapt_beats.py`.

**Interfaces:**
- Consumes: T1 (`inject_lora`, `merge_lora`, `lora_parameters`, `masked_patch_loss`), T2 (`iter_target_windows`), `rowii.signals.beats_model.load_beats_model` + the vendored preprocessing (READ `BeatsFeaturizer`/vendored BEATs for how raw 16 kHz windows become fbank inputs — reuse its `extract_features`/fbank path; the objective's encoder_forward closes over the model's frame-level encoder outputs).
- Produces: `adapt_beats.py --mode lora|full --run <name> --epochs 5 --batch-size 16 --lr 1e-4 --seed 7 --max-windows 8000 --out models/adapted/` → `beats_lora_<run>.pt` / `beats_ft_<run>.pt` saved in the SAME {"cfg", "model"} format `load_beats_model` reads (merged first for lora; full-FT saves all weights) + sidecar `<name>.json` (mode, run, epochs, n_windows, final_loss, elapsed_s, proxy-objective note). Defaults per mode: lora lr 1e-4 epochs 5; full lr 1e-5 epochs 2 (applied when flags omitted; explicit flags win). Guards: torch/checkpoint via the established hint pattern.
- Test: monkeypatch `load_beats_model` to return a tiny stand-in (the T1 test's `_TinyAttnModel` extended with an `extract-features`-shaped callable) and `iter_target_windows` to synthetic windows; run main() with `--mode lora --epochs 1` → checkpoint file exists, loads via torch.load with {"cfg","model"} keys, sidecar json complete; `--mode full` path smoke; determinism of saved tensors across two same-seed runs; lora mode trains ONLY adapter+head params (spy on optimizer param count).

- [ ] Steps: failing tests → verify → implement → gates → commit `feat: BEATs adaptation script (LoRA and full fine-tune, merged export)`.

---

### Task 4: Distilled student (+ variant `audio-student`)

**Files:** Create `src/rowii/adapt/student.py`, `scripts/distill_beats.py`; Modify `src/rowii/config.py` (+ `.env.example`: `ROWII_STUDENT_CHECKPOINT`), `src/rowii/pipeline.py` (variant wiring + fingerprint scoped line + GOLDEN PIN update in the same commit), variant tuples in the five scripts; Test `tests/test_student.py`, pipeline/CLI test extensions.

**Interfaces:**
- Consumes: cached teacher embeddings (`results/cache/<run>--audio-beats.npz` — loaded in the SCRIPT via the public cache-load path with fingerprint check; test with synthetic npz), logmel cache as input source (same pattern), `split_by_segments` leakage rule.
- Produces: `StudentNet(n_mels=64, n_frames=49, out_dim=768, channels=(32, 64, 128))` (eager torch, Conv2d×3 stride 2 + GAP + Linear → 768); `StudentFeaturizer(checkpoint: Path | None, encoder=None)` mirroring TfcFeaturizer (input = logmel-STYLE flattened patches computed internally from raw windows: reuse `LogmelFeaturizer` for the front-end, then the student net; 768-d float64; names `student_e0..767`); `distill_beats.py --run <name> --epochs 30 --batch-size 256 --lr 1e-3 --seed 7 --out models/adapted/student_<run>.pt` — MSE(student(logmel_window), teacher_embedding) on calibration-side windows only; checkpoint {"cfg","model","teacher_variant","run","epochs"}. Variant `audio-student` (streams = audio's mic stream(s) — mirror audio-tfc's single-stream? NO: mirror `audio-beats`' stream set for grid comparability — READ what audio-beats uses and mirror), fingerprint gains a scoped `student_checkpoint=` line for this variant ONLY; the non-tfc golden pin must stay UNCHANGED (prove by keeping the test green) and a NEW golden pin for audio-student is added.
- Tests: distillation loss decreases (synthetic teacher = fixed random projection of inputs); featurizer contract (2-D/3-D, 768-d, names, stub encoder, torch-free import); variant dispatch + fingerprint scoping incl. goldens; guard tests.

- [ ] Steps: failing tests → verify → implement → gates → commit `feat: distilled BEATs student and audio-student variant`.

---

### Task 5: Post-training INT8 quantization branch

**Files:** Create `scripts/quantize_beats.py`; Modify `src/rowii/signals/beats.py` (`BeatsFeaturizer` alternate-load branch), `src/rowii/config.py` (+ `.env.example`: `ROWII_BEATS_INT8_CHECKPOINT`), `src/rowii/pipeline.py` (fingerprint: include the int8 path for beats variants ONLY WHEN SET — unset ⇒ payload byte-identical, non-tfc golden pin unchanged; add a set-case golden), tests.

**Interfaces:**
- Produces: `quantize_beats.py --out models/adapted/beats_int8.pt` — loads fp32 BEATs from env checkpoint, `torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)`, `torch.save` the module (documented: module pickle, not state-dict; CPU-only inference); prints size before/after. `BeatsFeaturizer`: when `cfg`-provided int8 path is set, load the quantized module (torch.load, map_location cpu), force CPU device, and document the branch; embedding drift helper `cosine_drift(a, b) -> float` in the script for the execution report.
- Tests: featurizer branch with a monkeypatched tiny quantized-like module (any nn.Module saved via torch.save loads through the branch); fingerprint conditional-inclusion tests + goldens; quantize script smoke with a tiny stand-in model (monkeypatched load) asserting the saved module reloads and the size log prints.

- [ ] Steps: failing tests → verify → implement → gates → commit `feat: post-training INT8 quantization path for BEATs (CPU deployment)`.

---

### Task 6: Cross-attention fusion head + `--xattn-fusion` view

**Files:** Create `src/rowii/fusionx/__init__.py`, `src/rowii/fusionx/model.py` (eager), `src/rowii/fusionx/wrapper.py` (lazy: config, load, joint-embedding computation), `scripts/train_xattn.py`; Modify `scripts/run_step2.py` (`--xattn-fusion` flag + view, mirroring `--score-fusion`'s structure and guards: within-day + variant fusion only + env `ROWII_XATTN_CHECKPOINT`), `src/rowii/config.py` + `.env.example`; Test `tests/test_fusionx.py`, `tests/test_step2_cli.py` extension.

**Interfaces:**
- Produces: `XattnConfig(audio_dim=768, vib_dim_lift=128, heads=4, out_dim=128, temperature=0.07)` (torch-free, wrapper); `XattnHead(cfg, vib_in_dim)` (model.py: vib lift Linear(vib_in_dim→128), `torch.nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)` with audio lifted 768→128 as a 1-token query and vib as 1-token key/value, residual + LayerNorm + Linear→out_dim); `info_nce(z_a, z_b, temperature) -> Tensor` (reuse the tfc_loss structure — import it: `from rowii.tfc.model import tfc_loss` IS the same symmetric InfoNCE; document the reuse); `train_xattn.py --run <name> --epochs 20 ...` trains on calibration-side windows: audio side = cached audio-beats embeddings, vib side = the fusion variant's VIB COLUMNS (via `rowii.anomaly.fusion.split_branch_columns` on the fusion cache — zero extraction; document the grid caveat: audio-beats vs fusion grids may differ sub-window — align by window index ONLY when `n_windows` equal AND |t0 delta| < window_ns, mirroring the ensemble tolerance, else exit); checkpoint `xattn_<run>.pt` {"cfg","model","run","vib_dim","epochs"}. run_step2 `--xattn-fusion`: per state, joint embeddings for fit/conformal/scoring windows → kNN scorer → conformal → `far_table_xattn.csv` (rule column: "xattn" + single-branch baselines already exist in score-fusion — do NOT duplicate them; columns `label, n_calibration, n_scored, n_alarms, realized_far, low_confidence`) + `xattn_notes.md` (trained-on-calibration-side caveat + proxy-free but scarce-data note).
- Tests: head shapes/determinism; InfoNCE reuse sanity (aligned < shuffled — one test, cheap); train script e2e with monkeypatched caches (synthetic npz for both variants, aligned grids) → checkpoint loads, loss decreased; CLI view smoke on the synthetic fixture with a stub head (monkeypatch load) → far_table_xattn.csv exists with expected columns + guards (protocol/variant/missing checkpoint → exit 2 or the established SystemExit style).

- [ ] Steps: failing tests → verify → implement → gates → commit `feat: cross-attention fusion head (third fusion level) with --xattn-fusion view`.

---

### Task 7: Inference benchmark harness

**Files:** Create `scripts/benchmark_inference.py`; Test `tests/test_benchmark_inference.py`.

**Interfaces:**
- Produces: `benchmark_inference.py --configs handcrafted,logmel,beats,beats-int8,tfc,student --n-windows 200 --batch-sizes 1,256 --devices cpu,mps --out results/benchmarks/` — per config: builds the featurizer exactly as the pipeline would (same constructors/checkpoints from cfg/env), generates OR loads raw 50 kHz windows (`--source synthetic` default for tests; `--source run:<name>` reads one real gantner file — data-marked path), measures wall latency per window (median over batches after 2 warmup batches), peak RSS delta (`resource.getrusage(RUSAGE_SELF).ru_maxrss` before/after, macOS bytes), model size on disk + parameter count (0 for handcrafted/logmel), END-TO-END including preprocessing. Skips gracefully (logged) any config whose checkpoint env is unset or device unsupported (int8+mps → skip with note). Output `inference.csv` (config, device, batch_size, n_params, size_mb, latency_ms_per_window, peak_rss_mb) + `inference.md` table.
- Tests: synthetic-source run over handcrafted+logmel on CPU → csv exists with those rows, latencies > 0, monotone sanity none required; unset-checkpoint configs skipped with log; unknown config exit 2.

- [ ] Steps: failing tests → verify → implement → gates → commit `feat: end-to-end inference benchmark harness (size, latency, memory incl. preprocessing)`.

---

### Task 8: Execution + synthesis (orchestrator-led)

- [ ] **Step 1:** adapt: `adapt_beats.py --mode lora --run 010726-tu_ph_tu` then `--mode full` (budgets logged; MPS).
- [ ] **Step 2:** distill: `distill_beats.py --run 010726-tu_ph_tu`; quantize: `quantize_beats.py`.
- [ ] **Step 3:** adapted evidence: with `ROWII_BEATS_CHECKPOINT=models/adapted/beats_lora_010726-tu_ph_tu.pt`: warm audio-beats caches for the 3 SCADA days (new fingerprint = new cache — deliberate), Step-1 010726 audio-beats, within-day 3 days × knn; repeat for the ft checkpoint; candidate overlap frozen-vs-lora (audio-beats-knn under both checkpoints — needs care: same combo name, different checkpoint ⇒ run sequentially into separate results copies or rename dirs; simplest: run lora sweeps with `--out`-free default then `mv results/step2/within-day/<run>/audio-beats-detected results/.../audio-beats-lora-detected` style archival BEFORE the ft pass — document the procedure in the log).
- [ ] **Step 4:** student evidence: `ROWII_STUDENT_CHECKPOINT` set → warm audio-student caches, Step-1 010726, within-day 3 days × knn.
- [ ] **Step 5:** INT8: embedding drift on 200 real cached windows; one within-day FAR parity sweep (010726, audio-beats under int8 env, CPU) into an archived dir as in step 3.
- [ ] **Step 6:** xattn: `train_xattn.py --run <each of 3 days>`; `run_step2 --xattn-fusion` per day.
- [ ] **Step 7:** benchmarks: synthetic + `--source run:010726-tu_ph_tu` real pass, cpu+mps.
- [ ] **Step 8:** digest agent → README package-5 section + research note + ledger; final whole-branch review; PR.

## Execution notes (orchestrator)

- Dependency order: T1 → T3; T2 → T3; T4, T5, T6, T7 independent after T1/T2 land (T4/T6 need only caches + established patterns). Suggested: T1, T2, T3, T4, T5, T6, T7, T8. One implementer at a time; HARD no-delegation clause; no real training in tasks.
- Step-3's checkpoint-swap evidence procedure produces MULTIPLE cache sets keyed by checkpoint path — disk cost ~3× audio-beats caches (~1.2 GB) — acceptable, note in log.

## Self-review (done at plan-writing time)

- Spec coverage: D1→T1, D2→T1, D3→T2, D4→T3, D5→T4, D6→T5, D7→T7, D8→T6, D9→T8. No gaps.
- Placeholder scan: T4/T5/T6/T7 carry complete interface contracts + test prescriptions with the novel code fully specified (T1 carries full reference tests; LoraLinear/merge math given in-line). The one open judgment (audio-student stream set) is delegated with an explicit READ instruction, not a TBD.
- Type consistency: checkpoint formats named per artifact; `tfc_loss` reuse for InfoNCE cross-referenced; fingerprint golden-pin rule restated where touched.
