# Step-2 Package 4 Implementation Plan — TF-C Industrial-Pretraining Transfer Pole

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the design chapter's industrial-pretraining pole: a compact TF-C encoder pair pre-trained on MIMII (audio) and CWRU/Paderborn (vibration), integrated as frozen-embedding variants `audio-tfc` / `vibration-tfc` in the existing sweep machinery.

**Architecture:** `src/rowii/tfc/model.py` holds eager-torch nn.Modules (time-CNN + freq-CNN encoders, projection heads, NT-Xent+consistency loss) mirroring the `_recon_models.py` precedent; `src/rowii/tfc/wrapper.py` is the lazy-torch featurizer + checkpoint loader mirroring `signals/beats.py`. Corpus download and pre-training are standalone scripts; downloads and real training never run in tests (synthetic fixtures only). Integration reuses the per-stream featurizer contract and the checkpoint-aware cache fingerprint rule established for BEATs.

**Tech Stack:** torch (existing `[beats]` extra), scipy.signal.resample_poly, existing pipeline/cache/conformal machinery, `curl`-based downloads with sha256 manifests.

**Design authority:** `docs/superpowers/specs/2026-07-16-step2-package4-tfc-design.md` (D1–D5). Read it first.

## Global Constraints

- Branch: `feat/step2-package4-tfc` (from `main` = 897e3ec).
- Gates after every task: `.venv/bin/python -m pytest tests/ -q -m "not data"` green; `.venv/bin/ruff check .` clean; `.venv/bin/mypy src scripts` clean.
- Torch stays lazy outside `src/rowii/tfc/model.py` (which is eager, like `_recon_models.py`); `src/rowii/tfc/wrapper.py` must import without torch (poison-probe-tested); missing torch/checkpoint → RuntimeError/SystemExit hints mirroring the beats guard.
- Embedding contract: 256-d float64 (time 128 ⊕ freq 128, pre-projection pooled encoder outputs); input resample target 8000 Hz; deterministic under fixed seeds (CPU).
- Downloads, real corpora, and real pre-training NEVER run inside implementer/reviewer tasks — synthetic fixtures only; execution is orchestrator-led (Task 5).
- Conventional commits; explicit `git add`; no Co-Authored-By. No partner values.
- Docstring density mirrors `signals/beats.py` / `anomaly/recon.py`.

---

### Task 1: TF-C model, loss, and wrapper (`src/rowii/tfc/`)

**Files:**
- Create: `src/rowii/tfc/__init__.py` (empty), `src/rowii/tfc/model.py`, `src/rowii/tfc/wrapper.py`
- Test: `tests/test_tfc_model.py`, `tests/test_tfc_wrapper.py`

**Interfaces:**
- Consumes: `rowii.signals.beats.best_device` (lazy, inside methods); the featurizer contract from `pipeline._extract_stream_features` (`transform(stack, rate_hz) -> (B, F) float64`, `feature_names() -> list[str]`; stack is `(B, S)` or `(B, S, C)` float — mono-mix `(B,S,C)` via `stack.mean(axis=2)` like `BeatsFeaturizer`/`LogmelFeaturizer`).
- Produces (Tasks 3/4 rely on these):
  - `model.py`: `TfcConfig(sample_rate_hz: int = 8000, n_samples: int = 8000, embed_dim: int = 128, proj_dim: int = 64, channels: tuple[int, ...] = (32, 64, 128, 128), temperature: float = 0.2)` (frozen dataclass, NO torch needed to import it — define in wrapper.py? NO: define `TfcConfig` in `wrapper.py` (torch-free) and have model.py import it from wrapper — wrapper must not import model at module level); `TfcModel(cfg)` (nn.Module: `.time_encoder`, `.freq_encoder`, `.time_proj`, `.freq_proj`; `forward(x_time, x_freq) -> tuple[h_t, h_f, z_t, z_f]` where h_* are (B,128) pooled encoder outputs and z_* are (B,64) projections); `tfc_loss(z_t, z_f, temperature) -> torch.Tensor` (scalar: NT-Xent across the batch treating (z_t_i, z_f_i) as the positive pair — the paper's consistency-through-contrast core; document that the full paper adds intra-view augmented pairs and this compact version uses the cross-view pair only, a documented simplification); `freq_view(x_time) -> torch.Tensor` (|rfft| of the time view, standardized per window).
  - `wrapper.py`: `TfcConfig` (torch-free dataclass, as above); `load_tfc_model(checkpoint: Path, device) -> "TfcModel"` (torch.load dict with keys `cfg`, `model`, `corpus_manifest_sha256`, `epochs`; rebuilds TfcConfig, load_state_dict strict); `TfcFeaturizer(checkpoint: Path | None, encoder: object | None = None)` — mirrors `BeatsFeaturizer`: injected `encoder` stub for tests (any object with `embed(batch_8khz: np.ndarray) -> np.ndarray` returning (B,256)); real path lazily imports torch + model, resamples each window's mono mix to 8 kHz via `scipy.signal.resample_poly(x, 8000, int(rate_hz))` (document: exact for integer rates; window lengths normalized to exactly 8000 samples by pad/trim), standardizes per window, batches through the frozen model (`h_t ⊕ h_f`), returns float64; `feature_names() -> ["tfc_e0" ... "tfc_e255"]`.

- [ ] **Step 1: Write the failing tests.**

`tests/test_tfc_model.py` (module-level `torch = pytest.importorskip("torch")`, autouse `ROWII_FORCE_CPU=1` fixture like tests/test_recon.py):

```python
"""TF-C model/loss unit tests (package-4 spec D1). CPU-forced, seeded."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rowii.tfc.model import TfcModel, freq_view, tfc_loss
from rowii.tfc.wrapper import TfcConfig


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setenv("ROWII_FORCE_CPU", "1")


def _batch(b=8, n=8000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, generator=g)


class TestTfcModel:
    def test_forward_shapes(self):
        cfg = TfcConfig()
        m = TfcModel(cfg)
        x = _batch()
        h_t, h_f, z_t, z_f = m(x, freq_view(x))
        assert h_t.shape == (8, 128) and h_f.shape == (8, 128)
        assert z_t.shape == (8, 64) and z_f.shape == (8, 64)

    def test_loss_decreases_with_training(self):
        torch.manual_seed(7)
        cfg = TfcConfig(channels=(8, 16), embed_dim=32, proj_dim=16)
        m = TfcModel(cfg)
        x = _batch(b=16, n=8000, seed=1)
        xf = freq_view(x)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        first = tfc_loss(*m(x, xf)[2:], cfg.temperature).item()
        for _ in range(30):
            opt.zero_grad()
            loss = tfc_loss(*m(x, xf)[2:], cfg.temperature)
            loss.backward()
            opt.step()
        last = tfc_loss(*m(x, xf)[2:], cfg.temperature).item()
        assert last < first

    def test_loss_prefers_aligned_pairs(self):
        # perfectly aligned projections -> lower loss than shuffled pairing
        torch.manual_seed(0)
        z = torch.nn.functional.normalize(torch.randn(16, 64), dim=1)
        aligned = tfc_loss(z, z, 0.2).item()
        perm = z[torch.randperm(16, generator=torch.Generator().manual_seed(1))]
        shuffled = tfc_loss(z, perm, 0.2).item()
        assert aligned < shuffled

    def test_freq_view_shape_and_determinism(self):
        x = _batch(b=3)
        f1, f2 = freq_view(x), freq_view(x)
        assert f1.shape == (3, 4001)
        assert torch.equal(f1, f2)
```

`tests/test_tfc_wrapper.py` (torch-free where possible; poison probe + stub encoder):

```python
"""TfcFeaturizer contract tests (package-4 spec D1/D4): stub encoder, no torch."""
from __future__ import annotations

import numpy as np
import pytest

from rowii.tfc.wrapper import TfcConfig, TfcFeaturizer


class _StubEncoder:
    def __init__(self):
        self.calls = []

    def embed(self, batch_8khz: np.ndarray) -> np.ndarray:
        self.calls.append(batch_8khz.shape)
        return np.tile(batch_8khz.mean(axis=1, keepdims=True), (1, 256))


class TestTfcFeaturizer:
    def test_transform_2d_shape_names_dtype(self):
        f = TfcFeaturizer(checkpoint=None, encoder=_StubEncoder())
        rng = np.random.default_rng(0)
        out = f.transform(rng.normal(0, 1, (3, 50_000)), 50_000.0)
        assert out.shape == (3, 256) and out.dtype == np.float64
        names = f.feature_names()
        assert names[0] == "tfc_e0" and names[-1] == "tfc_e255" and len(names) == 256

    def test_transform_3d_mono_mix(self):
        stub = _StubEncoder()
        f = TfcFeaturizer(checkpoint=None, encoder=stub)
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, (2, 10_000, 3))
        out = f.transform(x, 10_000.0)
        assert out.shape == (2, 256)
        assert stub.calls[0] == (2, 8000)  # resampled to 8 kHz

    def test_resample_normalizes_length(self):
        stub = _StubEncoder()
        f = TfcFeaturizer(checkpoint=None, encoder=stub)
        x = np.random.default_rng(2).normal(0, 1, (1, 50_004))  # jittered window
        f.transform(x, 50_000.0)
        assert stub.calls[0] == (1, 8000)

    def test_module_imports_without_torch(self):
        # wrapper.py itself must not import torch at module level; this test file
        # already imported it torch-free above. Assert the real-encoder path guards:
        f = TfcFeaturizer(checkpoint=None, encoder=None)
        with pytest.raises((RuntimeError, ValueError)):
            f.transform(np.zeros((1, 8000)), 8000.0)  # no checkpoint, no stub
```

Plus (in the same file) a `@pytest.mark.skipif(no torch)` round-trip: build a tiny real `TfcModel`, save a checkpoint dict via the SAME format `pretrain_tfc.py` will write (`{"cfg": asdict(cfg), "model": state_dict, "corpus_manifest_sha256": "test", "epochs": 1}`), `load_tfc_model` it, and check `TfcFeaturizer(checkpoint=path).transform` returns (B,256) float64 deterministically twice.

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_tfc_model.py tests/test_tfc_wrapper.py -q` → ModuleNotFoundError.

- [ ] **Step 3: Implement.** `model.py` (eager torch, plain modules — mirror `_recon_models.py`'s header comment about being the one eager-torch module of its package):

```python
"""Compact TF-C model + loss (package-4 spec D1). Eager torch module (the
`_recon_models.py` precedent): imported ONLY lazily from rowii.tfc.wrapper."""
from __future__ import annotations

import torch

from rowii.tfc.wrapper import TfcConfig


class _Cnn1d(torch.nn.Module):
    """Shared 1-D CNN encoder: conv(stride 4) blocks -> global average pool -> embed."""

    def __init__(self, cfg: TfcConfig) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_ch = 1
        for out_ch in cfg.channels:
            layers += [
                torch.nn.Conv1d(in_ch, out_ch, kernel_size=8, stride=4, padding=2),
                torch.nn.BatchNorm1d(out_ch),
                torch.nn.ReLU(),
            ]
            in_ch = out_ch
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(cfg.channels[-1], cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N) -> (B, embed_dim)
        h = self.body(x.unsqueeze(1))          # (B, C, N')
        h = h.mean(dim=2)                      # global average pool
        return self.head(h)


class TfcModel(torch.nn.Module):
    """Time encoder + frequency encoder + projection heads (spec D1)."""

    def __init__(self, cfg: TfcConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.time_encoder = _Cnn1d(cfg)
        self.freq_encoder = _Cnn1d(cfg)
        self.time_proj = torch.nn.Linear(cfg.embed_dim, cfg.proj_dim)
        self.freq_proj = torch.nn.Linear(cfg.embed_dim, cfg.proj_dim)

    def forward(
        self, x_time: torch.Tensor, x_freq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_t = self.time_encoder(x_time)
        h_f = self.freq_encoder(x_freq)
        return h_t, h_f, self.time_proj(h_t), self.freq_proj(h_f)


def freq_view(x_time: torch.Tensor) -> torch.Tensor:
    """|rfft| of each window, standardized per window (spec D1 frequency view)."""
    mag = torch.abs(torch.fft.rfft(x_time, dim=1))
    mean = mag.mean(dim=1, keepdim=True)
    std = mag.std(dim=1, keepdim=True).clamp_min(1e-8)
    return (mag - mean) / std


def tfc_loss(z_t: torch.Tensor, z_f: torch.Tensor, temperature: float) -> torch.Tensor:
    """Cross-view NT-Xent: (z_t_i, z_f_i) is the positive pair; every other
    projection in the batch (both views) is a negative. Compact form of the
    paper's time-frequency consistency objective (documented simplification:
    no intra-view augmented pairs)."""
    z_t = torch.nn.functional.normalize(z_t, dim=1)
    z_f = torch.nn.functional.normalize(z_f, dim=1)
    b = z_t.shape[0]
    z = torch.cat([z_t, z_f], dim=0)                      # (2B, D)
    sim = z @ z.T / temperature                           # (2B, 2B)
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat(
        [torch.arange(b, 2 * b), torch.arange(0, b)]
    ).to(z.device)
    return torch.nn.functional.cross_entropy(sim, targets)
```

`wrapper.py` (torch-free at module level; `TfcConfig` frozen dataclass here; `load_tfc_model` and `TfcFeaturizer` with lazy `import torch` + `from rowii.tfc import model as tfc_model` inside methods; resample via `scipy.signal.resample_poly` after pad/trim to `round(rate_hz)` samples; `_TORCH_HINT = "TF-C featurizer needs torch: pip install -e '.[beats]'"`; missing checkpoint AND missing stub → ValueError naming `ROWII_TFC_*_CHECKPOINT`). Batch through the model in chunks of 512 windows under `torch.no_grad()`, concatenating `h_t ⊕ h_f` → float64.

- [ ] **Step 4: Full gates.** — all green. **Step 5: Commit**

```bash
git add src/rowii/tfc/__init__.py src/rowii/tfc/model.py src/rowii/tfc/wrapper.py tests/test_tfc_model.py tests/test_tfc_wrapper.py
git commit -m "feat: compact TF-C model, loss, and frozen-embedding featurizer"
```

---

### Task 2: Corpus download script + corpus loaders

**Files:**
- Create: `scripts/download_corpora.py`, `src/rowii/tfc/corpora.py`
- Test: `tests/test_tfc_corpora.py`, `tests/test_download_corpora.py`

**Interfaces:**
- Consumes: nothing new (stdlib urllib/hashlib; scipy for MAT loading via `scipy.io.loadmat`; wave/soundfile? — use `scipy.io.wavfile` [existing dep] for WAV).
- Produces:
  - `corpora.py`: `iter_windows_wav_dir(root: Path, *, exclude_substring: str | None = "abnormal", window_s: float = 1.0, target_hz: int = 8000, limit_clips: int | None = None) -> Iterator[np.ndarray]` — walks `*.wav` recursively (sorted), skips paths containing the exclusion substring (logged count), cuts non-overlapping 1-s windows, resamples to 8 kHz, per-window standardize, yields float32 (8000,) arrays; `iter_windows_mat_dir(root, *, key_substring: str = "DE_time", native_hz: int | None = None, ...)` — same for CWRU/Paderborn `.mat` files (variable holding the signal chosen by key substring; native rate from a `--rate` map or per-file metadata; document CWRU DE 12 kHz convention).
  - `download_corpora.py`: `--corpus mimii|cwru|paderborn|all`, `--dest data/public`, `--dry-run`; per-corpus URL+sha256 tables as module constants (fill with the real Zenodo/CWRU/Paderborn URLs — the implementer researches the CURRENT canonical URLs via the dataset landing pages and records them with sha256 computed AFTER first download [manifest updated then]; --dry-run prints the plan without network); writes `MANIFEST.json` per corpus (url, sha256, license: MIMII CC BY-SA 4.0 / CWRU academic-free / Paderborn CC BY-NC 4.0, downloaded_at, bytes); unzip/unrar-free policy: MIMII zips extracted with `zipfile`, Paderborn rar → if `unar`/`unrar` absent, print manual instructions and continue (documented fallback per spec D2).
- Tests use SYNTHETIC fixtures only: tmp dirs with tiny generated wav (scipy.io.wavfile.write, 0.1-s files → windowing yields 0 windows unless file >= 1 s — generate 2.5-s files → 2 windows) and mat files (scipy.io.savemat with a `X097_DE_time`-style key); assertions: window count, 8 kHz length 8000, standardization (mean≈0), exclusion filtering counts, limit_clips; download script: --dry-run prints all corpora + never opens the network (monkeypatch urllib to raise), unknown corpus exits 2.

- [ ] Steps: failing tests → verify fail → implement → gates → commit `feat: public-corpus download script and TF-C corpus loaders`.

---

### Task 3: Pre-training script

**Files:**
- Create: `scripts/pretrain_tfc.py`
- Test: `tests/test_pretrain_tfc.py`

**Interfaces:**
- Consumes: Task 1's `TfcModel`/`tfc_loss`/`freq_view`/`TfcConfig`; Task 2's corpus iterators.
- Produces: `pretrain_tfc.py --corpus mimii|bearings --data-root data/public --epochs 40 --batch-size 256 --lr 1e-3 --seed 7 --limit-clips N --out models/pretrained/tfc/` → saves `tfc_audio.pt` (mimii) / `tfc_vib.pt` (bearings: CWRU + whatever Paderborn subset exists, concatenated) with the checkpoint dict format from Task 1's round-trip test; per-epoch loss logging; total-window and per-corpus counts logged; device via `best_device()`; deterministic seeding (torch.manual_seed + seeded shuffle Generator); windows materialized into a memory-mapped npy staging file when count > 200k (avoid RAM blowups) — simpler acceptable alternative: reservoir-subsample to `--max-windows 200000` (default) with a seeded RNG, documented; augmentations per spec D1: time view jitter (gaussian sigma 0.01) + random scaling (0.9-1.1) + random zero-mask (up to 10% of samples); freq view derived AFTER augmentation.
- Test: end-to-end on a synthetic corpus (tmp wav dir from Task 2's fixture builder), `--epochs 2 --batch-size 8 --limit-clips 4 --max-windows 64` on CPU → checkpoint file exists, loads via `load_tfc_model`, embeddings from `TfcFeaturizer` have shape (B,256); loss values logged and finite; determinism: two runs same seed → byte-identical state_dicts (compare a few tensors).

- [ ] Steps: failing test → verify fail → implement → gates → commit `feat: TF-C pre-training script (mimii audio / bearings vibration)`.

---

### Task 4: Pipeline + CLI integration (`audio-tfc`, `vibration-tfc`)

**Files:**
- Modify: `src/rowii/config.py` (fields `tfc_audio_checkpoint: Path | None`, `tfc_vib_checkpoint: Path | None` from env `ROWII_TFC_AUDIO_CHECKPOINT`/`ROWII_TFC_VIB_CHECKPOINT`, mirroring `beats_checkpoint`), `.env.example`
- Modify: `src/rowii/pipeline.py` (`_streams_for_variant`: `audio-tfc` → same streams as `audio`; `vibration-tfc` → same as `vibration`; `_featurizer_for_stream` returns `TfcFeaturizer(checkpoint=cfg.tfc_audio_checkpoint)` / `(cfg.tfc_vib_checkpoint)`; `_cache_fingerprint` includes the RELEVANT tfc checkpoint path unconditionally for tfc variants — read how beats_checkpoint is folded in and mirror exactly; segment_ids primary-stream docstring updated)
- Modify: variant tuples in ALL scripts (run_step1 `_CONCRETE_VARIANTS` INCLUDING both tfc variants this time — spec D4 says TF-C is a legitimate state-detection candidate — plus `_VARIANT_CHOICES` everywhere: run_step2, run_step2_scarcity, warm_cache, apply_detector, and the beats-import-style guard extended: tfc variants need torch + their checkpoint → `_import_tfc_or_exit(cfg, variant)` in each script that can hit them)
- Test: `tests/test_pipeline.py` (dispatch + fingerprint cases), `tests/test_cli_smoke.py` (variant expansion incl. tfc; guard messages), one `tests/test_tfc_wrapper.py` addition if needed.

**Interfaces:**
- Consumes: Task 1's `TfcFeaturizer`.
- Produces: variants `audio-tfc`/`vibration-tfc` usable by every CLI; cache entries `<run>--audio-tfc.npz` keyed on the checkpoint path.
- Tests: dispatch assertions mirroring logmel's (streams tuple, featurizer type via injected cfg paths); fingerprint changes when the tfc checkpoint path changes (mirror the beats fingerprint test — find it in test_pipeline.py); run_step1 `--variant all` NOW INCLUDES audio-tfc/vibration-tfc (assert), with the guard exiting cleanly when no checkpoint is configured (parser-level or prepare-level — mirror `_import_beats_or_exit`'s placement and test its message names the env var).

- [ ] Steps: failing tests → verify fail → implement → gates → commit `feat: audio-tfc and vibration-tfc variants (frozen TF-C embeddings) across the pipeline`.

---

### Task 5: Execution + synthesis (orchestrator-led)

No new code. Order:
- [ ] **Step 1:** `download_corpora.py --corpus all` (background; ~10-15 GB; manifest sha256s recorded on first download).
- [ ] **Step 2:** pre-train both checkpoints (background, MPS): `--corpus mimii` → tfc_audio.pt; `--corpus bearings` → tfc_vib.pt (budgets logged; expect ≤ 2-4 h each).
- [ ] **Step 3:** set env checkpoints; warm caches: `warm_cache.py --runs 250526-tu 290626-tu 010726-tu_ph_tu 270626-pu_ph_pu_ph_pu_ph-1 --variants audio-tfc vibration-tfc`.
- [ ] **Step 4:** Step-1 state detection: `run_step1.py --run 010726-tu_ph_tu --variant audio-tfc --clusterer kmeans` (+ vibration-tfc).
- [ ] **Step 5:** Step-2 within-day: 3 days × both tfc variants × knn+mahalanobis × both conditionings; cross-day-per-state: day pairs × audio-tfc × knn; overlap: audio-tfc-knn vs fusion-knn and vs audio-beats-knn.
- [ ] **Step 6:** digest agent → README package-4 section (with the compactness caveat) + research note in master-thesis + ledger; final whole-branch review; PR.

## Execution notes (orchestrator)

- Dependency order: T1 → T3 → (T4 after T1); T2 independent; downloads (T5 step 1) can start as soon as T2 lands — run them in the background WHILE T3/T4 are implemented.
- One implementer at a time; HARD no-delegation clause every dispatch; no downloads/real-training in implementer tasks.
- The implementer for T2 researches current canonical dataset URLs (MIMII Zenodo record 3384388 pump 0dB; CWRU bearing data center normal + 12k drive-end; Paderborn KAt main page) and encodes them as constants with `sha256: "TBD-first-download"` placeholders that the ORCHESTRATOR fills after the first verified download (the one sanctioned placeholder, marked exactly so).

## Self-review (done at plan-writing time)

- Spec coverage: D1→T1, D2→T2, D3→T3, D4→T4, D5→T5. No gaps.
- Placeholder scan: T2's sha256 "TBD-first-download" is explicitly sanctioned and assigned to the orchestrator; T2/T3 steps compressed to step-lists with complete interface contracts and test prescriptions (deliberate — their code is mechanical given the contracts; T1 carries full reference code for the novel parts).
- Type consistency: `TfcConfig` lives in wrapper.py (torch-free) and is imported by model.py — one direction, no cycle; checkpoint dict format defined once (T1 round-trip test) and reused by T3; `embed` stub contract matches wrapper's internal call.
