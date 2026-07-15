# Step-2 Package 3 Implementation Plan — Baselines & Scoring Completeness

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the design chapter's scoring commitments — classical one-class baselines, reconstruction scorers, the OC-SVM+IF+LSTM-AE majority ensemble, score-level fusion, and the conditioning-granularity sweep — on the existing conformal harness.

**Architecture:** Classical and reconstruction scorers slot into the existing `Scorer` protocol (`fit(reference)/score(x)`, higher = anomalous) and the existing scorer registry (`rowii.anomaly.sweep._make_scorer` + `run_step2` choices). A new `logmel` variant feeds the Conv/LSTM autoencoders (window-internal time axis as sequence — protocol unchanged). Ensemble voting and p-value score-fusion are DECISION/orchestration-level views in `run_step2.py`, not scorers. Granularity is a `--states K` pass-through to `FittedDetector.fit(k=K)`.

**Tech Stack:** sklearn (OneClassSVM, IsolationForest, LocalOutlierFactor), torch (via the existing `[beats]` extra; lazy imports; `best_device()` from `rowii.signals.beats`), scipy (chi2 for Fisher), existing pipeline/cache/conformal machinery.

**Design authority:** `docs/superpowers/specs/2026-07-15-step2-package3-baselines-design.md` (D1–D7). Read it before any task.

## Global Constraints

- Branch: `feat/step2-package3-baselines` (from `main` = 14beb56).
- Tests: `.venv/bin/python -m pytest tests/ -q -m "not data"` after every task (full suite with data marks only where a task says so); `.venv/bin/ruff check .` and `.venv/bin/mypy src scripts` clean before every commit.
- Conventional commits; explicit `git add` paths (never `-A`); no Co-Authored-By.
- Scorer polarity is EXPLICIT (higher = more anomalous) with a docstring note per scorer; never auto-detected.
- Torch: lazy import inside methods (module import must succeed without the extra — mirror `src/rowii/signals/beats.py`); training seeded (`torch.manual_seed`); tests force CPU via the existing `ROWII_FORCE_CPU` env understood by `best_device()`.
- No partner values adopted; honesty notes carried into every new writer.
- RNG: `numpy.random.default_rng(seed)` / `torch.manual_seed(seed)` only.
- Docstring style: match `src/rowii/anomaly/scorers.py`'s density (Google Args/Returns/Raises + spec cross-references).

---

### Task 1: Classical one-class scorers + registry extension

**Files:**
- Modify: `src/rowii/anomaly/scorers.py` (three new classes at the end)
- Modify: `src/rowii/anomaly/sweep.py` (`_make_scorer`, `SweepConfig.scorer` Literal)
- Modify: `scripts/run_step2.py` (`ScorerName` Literal at ~line 239, `_SCORER_CHOICES` at ~245, `_CONCRETE_SCORERS` at ~249, its local `_make_scorer` at ~430)
- Test: `tests/test_scorers.py` (extend), `tests/test_sweep.py` (one registry test)

**Interfaces:**
- Consumes: existing `Scorer` protocol, `_check_reference(reference)` helper (scorers.py), `SweepConfig`.
- Produces (Tasks 4/7 rely on these exact names): scorer registry names `"ocsvm"`, `"iforest"`, `"lof"`; classes `OcSvmScorer(nu: float = 0.1, gamma: str = "scale")`, `IsolationForestScorer(n_estimators: int = 200, random_seed: int = 7)`, `LofScorer(n_neighbors: int = 20)` — each with `name: str` class attr matching its registry name, `fit(reference) -> <Self>`, `score(x) -> np.ndarray` float64.
- `_CONCRETE_SCORERS` becomes `("knn", "mahalanobis", "ocsvm", "iforest", "lof")`; `"all"` resolves to all five.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_scorers.py` (mirror the file's existing synthetic-cluster fixtures; read them first and reuse the helper that builds a tight normal cluster + far outliers):

```python
class TestClassicalScorers:
    def _reference_and_outliers(self, seed=0):
        rng = np.random.default_rng(seed)
        reference = rng.normal(0.0, 0.1, (300, 8))
        inliers = rng.normal(0.0, 0.1, (50, 8))
        outliers = rng.normal(4.0, 0.1, (10, 8))
        return reference, inliers, outliers

    @pytest.mark.parametrize(
        "scorer_factory",
        [
            lambda: OcSvmScorer(),
            lambda: IsolationForestScorer(),
            lambda: LofScorer(),
        ],
    )
    def test_outliers_score_higher_than_inliers(self, scorer_factory):
        reference, inliers, outliers = self._reference_and_outliers()
        scorer = scorer_factory().fit(reference)
        s_in = scorer.score(inliers)
        s_out = scorer.score(outliers)
        assert s_in.dtype == np.float64 and s_in.shape == (50,)
        assert s_out.min() > s_in.max()  # explicit polarity: higher = anomalous

    def test_iforest_deterministic_given_seed(self):
        reference, inliers, _ = self._reference_and_outliers()
        a = IsolationForestScorer(random_seed=7).fit(reference).score(inliers)
        b = IsolationForestScorer(random_seed=7).fit(reference).score(inliers)
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize(
        "scorer_factory",
        [lambda: OcSvmScorer(), lambda: IsolationForestScorer(), lambda: LofScorer()],
    )
    def test_non_finite_reference_rejected(self, scorer_factory):
        bad = np.full((30, 4), np.nan)
        with pytest.raises(ValueError):
            scorer_factory().fit(bad)

    def test_names_match_registry(self):
        assert OcSvmScorer().name == "ocsvm"
        assert IsolationForestScorer().name == "iforest"
        assert LofScorer().name == "lof"
```

Append to `tests/test_sweep.py` (mirroring its existing `_make_scorer` coverage if present, else a minimal registry test):

```python
def test_make_scorer_knows_classical_names():
    from rowii.anomaly.sweep import _make_scorer
    for name, cls_name in [
        ("ocsvm", "OcSvmScorer"),
        ("iforest", "IsolationForestScorer"),
        ("lof", "LofScorer"),
    ]:
        scorer = _make_scorer(name)
        assert type(scorer).__name__ == cls_name
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_scorers.py -q -k Classical` → ImportError/NameError.

- [ ] **Step 3: Implement.** In `src/rowii/anomaly/scorers.py` append (docstrings in the file's established density — each carries the EXPLICIT polarity note "score = -<sklearn quantity>; higher = more anomalous; polarity is set here by construction, never auto-detected (v1 H2 lesson, spec D1)"):

```python
class OcSvmScorer:
    """One-class SVM baseline (RBF) on the shared Scorer contract (spec D1)."""

    name: str = "ocsvm"

    def __init__(self, nu: float = 0.1, gamma: str = "scale") -> None:
        from sklearn.svm import OneClassSVM

        self.nu = nu
        self.gamma = gamma
        self._model = OneClassSVM(nu=nu, gamma=gamma)

    def fit(self, reference: np.ndarray) -> "OcSvmScorer":
        _check_reference(reference)
        self._model.fit(reference)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(-self._model.decision_function(x), dtype=np.float64)


class IsolationForestScorer:
    """Isolation Forest baseline on the shared Scorer contract (spec D1)."""

    name: str = "iforest"

    def __init__(self, n_estimators: int = 200, random_seed: int = 7) -> None:
        from sklearn.ensemble import IsolationForest

        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self._model = IsolationForest(
            n_estimators=n_estimators, random_state=random_seed
        )

    def fit(self, reference: np.ndarray) -> "IsolationForestScorer":
        _check_reference(reference)
        self._model.fit(reference)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(-self._model.score_samples(x), dtype=np.float64)


class LofScorer:
    """Local Outlier Factor (novelty mode) baseline on the shared Scorer contract
    (spec D1)."""

    name: str = "lof"

    def __init__(self, n_neighbors: int = 20) -> None:
        from sklearn.neighbors import LocalOutlierFactor

        self.n_neighbors = n_neighbors
        self._model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)

    def fit(self, reference: np.ndarray) -> "LofScorer":
        _check_reference(reference)
        self._model.fit(reference)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(-self._model.score_samples(x), dtype=np.float64)
```

(sklearn top-level imports are fine here — sklearn is a core dependency; keep the constructor-local imports ONLY if scorers.py's existing style demands top-level otherwise move them to the module header with the existing sklearn import.)

Registry: in `sweep.py` extend `_make_scorer` with the three names (same pattern as the existing two branches) and widen `SweepConfig.scorer`'s `Literal["knn", "mahalanobis"]` to include `"ocsvm", "iforest", "lof"`. In `run_step2.py`: widen `ScorerName`, extend `_SCORER_CHOICES` (keep `"all"` last) and `_CONCRETE_SCORERS`, and extend the script's local `_make_scorer` identically.

- [ ] **Step 4: Suite + lints** — `.venv/bin/python -m pytest tests/ -q -m "not data" && .venv/bin/ruff check . && .venv/bin/mypy src scripts` → all pass.
- [ ] **Step 5: Commit**

```bash
git add src/rowii/anomaly/scorers.py src/rowii/anomaly/sweep.py scripts/run_step2.py tests/test_scorers.py tests/test_sweep.py
git commit -m "feat: classical one-class scorers (OC-SVM, IsolationForest, LOF) in the sweep registry"
```

---

### Task 2: `logmel` audio variant with cache support

**Files:**
- Create: `src/rowii/signals/logmel.py`
- Modify: `src/rowii/pipeline.py` (`_streams_for_variant` ~line 83, `_featurizer_for_stream` ~line 103; the variant literal lists it participates in)
- Modify: `scripts/run_step1.py` + `scripts/run_step2.py` + `scripts/run_step2_scarcity.py` + `scripts/warm_cache.py` + `scripts/apply_detector.py` variant choice tuples (add `"logmel"`)
- Test: `tests/test_logmel.py` (new), `tests/test_pipeline.py` (one cache round-trip case)

**Interfaces:**
- Consumes: the per-stream featurizer contract used by `_extract_stream_features` — read `_featurizer_for_stream` and the `AudioFeaturizer` class first: the contract is `transform(stack: np.ndarray, rate_hz: float) -> np.ndarray` over a `(B, n_samples)` window batch plus `feature_names() -> list[str]`.
- Produces: `LogmelFeaturizer(n_mels: int = 64, frame_s: float = 0.025, hop_s: float = 0.020)` with `transform` returning `(B, 64 * 49)` float64 (flattened `frames x mels`, frame-major) and `feature_names()` returning `[f"logmel_f{f}_m{m}" for f in range(49) for m in range(64)]`; variant name `"logmel"` mapping to streams `("RAWGeneratorMic__0",)` ONLY (spec D3: primary mic, size bound, documented).
- Mel filterbank: build with scipy-free NumPy (linear-spaced triangular filters on the mel scale, `mel = 2595 * log10(1 + hz/700)`, fmin 20 Hz, fmax rate/2), power spectrum via `np.fft.rfft` per frame with a Hann window, `log10(energy + 1e-10)`. Exactly 49 frames at 50 kHz/1 s (frame 1250 samples, hop 1000): `n_frames = 1 + (n_samples - frame_len) // hop_len` — assert and document; for other rates the count follows the same formula (do NOT hardcode 49 in the implementation, only in tests against 50 kHz).

- [ ] **Step 1: Failing tests** (`tests/test_logmel.py`):

```python
"""LogmelFeaturizer unit tests (package-3 spec D3)."""
from __future__ import annotations

import numpy as np

from rowii.signals.logmel import LogmelFeaturizer


class TestLogmelFeaturizer:
    def test_shape_and_names_at_50khz(self):
        rng = np.random.default_rng(0)
        stack = rng.normal(0, 0.1, (3, 50_000))
        f = LogmelFeaturizer()
        out = f.transform(stack, 50_000.0)
        assert out.shape == (3, 49 * 64)
        assert out.dtype == np.float64
        names = f.feature_names()
        assert len(names) == 49 * 64
        assert names[0] == "logmel_f0_m0" and names[64] == "logmel_f1_m0"

    def test_tone_lands_in_matching_mel_band(self):
        t = np.arange(50_000) / 50_000.0
        tone = np.sin(2 * np.pi * 1000.0 * t)[None, :]
        f = LogmelFeaturizer()
        out = f.transform(tone, 50_000.0).reshape(49, 64)
        band_energy = out.mean(axis=0)
        # the 1 kHz band must dominate quiet bands far away in mel space
        assert band_energy.argmax() not in (0, 63)
        assert band_energy.max() > band_energy.min() + 2.0  # log10 domain

    def test_deterministic(self):
        rng = np.random.default_rng(1)
        stack = rng.normal(0, 0.1, (2, 50_000))
        f = LogmelFeaturizer()
        np.testing.assert_array_equal(
            f.transform(stack, 50_000.0), f.transform(stack, 50_000.0)
        )
```

Plus in `tests/test_pipeline.py`: extend the existing variant-dispatch test(s) so `"logmel"` maps to `("RAWGeneratorMic__0",)` and `_featurizer_for_stream("RAWGeneratorMic__0", "logmel", cfg)` returns a `LogmelFeaturizer` (mirror the file's existing dispatch assertions for audio/vibration/beats), and add a cache round-trip case for a logmel-shaped feature matrix (reuse the file's existing cache write/load test pattern with F = 3136).

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_logmel.py -q` → ModuleNotFoundError.

- [ ] **Step 3: Implement** `src/rowii/signals/logmel.py`:

```python
"""Per-window log-mel featurizer for the `logmel` variant (package-3 spec D3).

Feeds the reconstruction scorers (`rowii.anomaly.recon`): each 1-second window
becomes a flattened (frames x mels) log-mel patch whose window-INTERNAL time
axis is the sequence the LSTM/Conv autoencoders consume -- no cross-window
contiguity is needed, so the `Scorer` protocol holds unchanged. Primary mic
stream only (size bound; spec D3). Pure NumPy (Hann window + rFFT + triangular
mel filterbank); no torch/librosa dependency.
"""
from __future__ import annotations

import numpy as np

_LOG_FLOOR = 1e-10


def _mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int, n_fft_bins: int, rate_hz: float, fmin_hz: float = 20.0) -> np.ndarray:
    """`(n_mels, n_fft_bins)` triangular filters, mel-spaced between fmin and rate/2."""
    fmax_hz = rate_hz / 2.0
    mel_edges = np.linspace(_mel(fmin_hz), _mel(fmax_hz), n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)
    fft_freqs = np.linspace(0.0, fmax_hz, n_fft_bins)
    bank = np.zeros((n_mels, n_fft_bins), dtype=np.float64)
    for m in range(n_mels):
        lo, mid, hi = hz_edges[m], hz_edges[m + 1], hz_edges[m + 2]
        rising = (fft_freqs - lo) / max(mid - lo, 1e-9)
        falling = (hi - fft_freqs) / max(hi - mid, 1e-9)
        bank[m] = np.clip(np.minimum(rising, falling), 0.0, None)
    return bank


class LogmelFeaturizer:
    """Flattened (frames x mels) log-mel patch per window -- see module docstring."""

    def __init__(self, n_mels: int = 64, frame_s: float = 0.025, hop_s: float = 0.020) -> None:
        self.n_mels = n_mels
        self.frame_s = frame_s
        self.hop_s = hop_s
        self._n_frames: int | None = None  # set on first transform, for feature_names

    def transform(self, stack: np.ndarray, rate_hz: float) -> np.ndarray:
        frame_len = int(round(self.frame_s * rate_hz))
        hop_len = int(round(self.hop_s * rate_hz))
        n_samples = stack.shape[1]
        n_frames = 1 + (n_samples - frame_len) // hop_len
        self._n_frames = n_frames

        window = np.hanning(frame_len)
        idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
        bank = _mel_filterbank(self.n_mels, frame_len // 2 + 1, rate_hz)

        out = np.empty((stack.shape[0], n_frames * self.n_mels), dtype=np.float64)
        for b in range(stack.shape[0]):
            frames = stack[b][idx] * window            # (n_frames, frame_len)
            power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
            mel_energy = power @ bank.T                # (n_frames, n_mels)
            out[b] = np.log10(mel_energy + _LOG_FLOOR).reshape(-1)
        return out

    def feature_names(self) -> list[str]:
        if self._n_frames is None:
            # 50 kHz / 1 s default geometry, matching the plant data
            self._n_frames = 1 + (50_000 - int(round(self.frame_s * 50_000))) // int(
                round(self.hop_s * 50_000)
            )
        return [
            f"logmel_f{f}_m{m}"
            for f in range(self._n_frames)
            for m in range(self.n_mels)
        ]
```

Wire into `pipeline.py`: `_streams_for_variant` gains `"logmel" -> ("RAWGeneratorMic__0",)`; `_featurizer_for_stream` returns `LogmelFeaturizer()` for the logmel variant (import at top — numpy-only, no laziness needed); check any variant-validation literals/error messages in pipeline.py and extend them. Add `"logmel"` to every script's variant choices tuple (grep `_CONCRETE_VARIANTS` / `_VARIANT_CHOICES`). NOTE: `feature_names()` before/after transform consistency and the `segment_ids` primary-stream doc (pipeline.py ~line 481) — logmel's primary stream is the mic, same as audio; verify the docstring list mentions logmel or update it.

- [ ] **Step 4: Suite + lints** — all pass. **Step 5: Commit**

```bash
git add src/rowii/signals/logmel.py src/rowii/pipeline.py scripts/run_step1.py scripts/run_step2.py scripts/run_step2_scarcity.py scripts/warm_cache.py scripts/apply_detector.py tests/test_logmel.py tests/test_pipeline.py
git commit -m "feat: logmel audio variant (64 mel x 49 frames per window) with cache support"
```

---

### Task 3: Reconstruction scorers (MLP-AE, LSTM-AE, Conv-AE)

**Files:**
- Create: `src/rowii/anomaly/recon.py`
- Modify: `src/rowii/anomaly/sweep.py` + `scripts/run_step2.py` (registry: names `"mlpae"`, `"lstmae"`, `"convae"`)
- Test: `tests/test_recon.py` (new; CPU-forced)

**Interfaces:**
- Consumes: `Scorer` protocol, `_check_reference` (import from `rowii.anomaly.scorers`), `best_device()` from `rowii.signals.beats` (env `ROWII_FORCE_CPU` honored), logmel geometry `(frames=49, mels=64)` from Task 2.
- Produces: `MlpAeScorer(hidden=(128, 32), epochs=200, lr=1e-3, batch_size=256, seed=7)`, `LstmAeScorer(hidden=64, epochs=100, lr=1e-3, batch_size=128, seed=7, n_mels=64)`, `ConvAeScorer(channels=(16, 32), epochs=100, lr=1e-3, batch_size=128, seed=7, n_mels=64)` — all with `name` attrs `"mlpae"/"lstmae"/"convae"`, `fit/score` per protocol; LSTM/Conv infer `n_frames = F // n_mels` from the input width and raise `ValueError` if `F % n_mels != 0` (guards against feeding a non-logmel variant).
- Torch is imported INSIDE `fit`/`score` (module import works without the extra); a missing torch raises `RuntimeError` with the install hint mirroring `_import_beats_or_exit`'s message.

- [ ] **Step 1: Failing tests** (`tests/test_recon.py`; module-level `pytest.importorskip("torch")` plus `monkeypatch.setenv("ROWII_FORCE_CPU", "1")` in a shared autouse fixture):

```python
"""Reconstruction-scorer tests (package-3 spec D2). CPU-forced, seeded."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rowii.anomaly.recon import ConvAeScorer, LstmAeScorer, MlpAeScorer


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setenv("ROWII_FORCE_CPU", "1")


def _vector_data(seed=0, f=32):
    rng = np.random.default_rng(seed)
    reference = rng.normal(0.0, 0.1, (400, f))
    inliers = rng.normal(0.0, 0.1, (40, f))
    outliers = rng.normal(3.0, 0.1, (10, f))
    return reference, inliers, outliers


def _patch_data(seed=0, frames=7, mels=8):
    rng = np.random.default_rng(seed)
    f = frames * mels
    reference = rng.normal(0.0, 0.1, (300, f))
    inliers = rng.normal(0.0, 0.1, (30, f))
    outliers = rng.normal(3.0, 0.1, (10, f))
    return reference, inliers, outliers


class TestMlpAe:
    def test_outliers_reconstruct_worse(self):
        reference, inliers, outliers = _vector_data()
        s = MlpAeScorer(hidden=(16, 4), epochs=60, seed=7).fit(reference)
        assert s.score(outliers).min() > s.score(inliers).max()

    def test_deterministic_given_seed(self):
        reference, inliers, _ = _vector_data()
        a = MlpAeScorer(hidden=(16, 4), epochs=10, seed=7).fit(reference).score(inliers)
        b = MlpAeScorer(hidden=(16, 4), epochs=10, seed=7).fit(reference).score(inliers)
        np.testing.assert_allclose(a, b)


class TestPatchAes:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: LstmAeScorer(hidden=8, epochs=40, seed=7, n_mels=8),
            lambda: ConvAeScorer(channels=(4, 8), epochs=40, seed=7, n_mels=8),
        ],
    )
    def test_outliers_reconstruct_worse(self, factory):
        reference, inliers, outliers = _patch_data()
        s = factory().fit(reference)
        assert s.score(outliers).min() > s.score(inliers).max()

    def test_non_divisible_width_rejected(self):
        with pytest.raises(ValueError, match="n_mels"):
            LstmAeScorer(n_mels=8).fit(np.zeros((30, 30)))
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError for `rowii.anomaly.recon`.

- [ ] **Step 3: Implement `src/rowii/anomaly/recon.py`.** Shared private helper `_train_autoencoder(model, reference_t, epochs, lr, batch_size, seed, device)` (Adam, MSELoss, shuffled batches via a seeded `torch.Generator`); scores = per-row reconstruction MSE as float64 numpy. Complete module skeleton the implementer fills mechanically:

```python
"""Reconstruction anomaly scorers (package-3 spec D2): MLP-AE on any feature
vector; LSTM-AE / Conv-AE on `logmel` patches, reshaped internally so the
window-internal time axis is the sequence (no cross-window contiguity; the
`Scorer` protocol holds unchanged). Score = per-window reconstruction MSE,
higher = more anomalous (explicit polarity by construction, spec D1 note).
Torch is imported lazily inside fit/score -- importing this module never
requires the `[beats]` extra; calling without torch raises RuntimeError with
the same install hint as scripts/run_step2.py's beats guard.
"""
from __future__ import annotations

import numpy as np

from rowii.anomaly.scorers import _check_reference

_TORCH_HINT = (
    "reconstruction scorers need torch: pip install -e '.[beats]'"
)


def _require_torch():  # -> module
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(_TORCH_HINT) from e
    return torch


def _device():
    from rowii.signals.beats import best_device

    return best_device()
```

Then the three classes; MLP-AE reference implementation the others mirror:

```python
class MlpAeScorer:
    name: str = "mlpae"

    def __init__(self, hidden=(128, 32), epochs=200, lr=1e-3, batch_size=256, seed=7):
        self.hidden = tuple(hidden)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self._model = None
        self._device = None

    def fit(self, reference: np.ndarray) -> "MlpAeScorer":
        _check_reference(reference)
        torch = _require_torch()
        torch.manual_seed(self.seed)
        device = _device()
        f = reference.shape[1]
        dims = [f, *self.hidden]
        enc = []
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [torch.nn.Linear(a, b), torch.nn.ReLU()]
        dec = []
        rdims = list(reversed(dims))
        for i, (a, b) in enumerate(zip(rdims[:-1], rdims[1:])):
            dec.append(torch.nn.Linear(a, b))
            if i < len(rdims) - 2:
                dec.append(torch.nn.ReLU())
        model = torch.nn.Sequential(*enc, *dec).to(device)
        x = torch.as_tensor(reference, dtype=torch.float32, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        gen = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(x), generator=gen)
            for start in range(0, len(x), self.batch_size):
                batch = x[perm[start : start + self.batch_size]]
                opt.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                opt.step()
        model.eval()
        self._model, self._device = model, device
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        torch = _require_torch()
        assert self._model is not None, "score() before fit()"
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float32, device=self._device)
            recon = self._model(t)
            mse = ((recon - t) ** 2).mean(dim=1)
        return np.asarray(mse.cpu().numpy(), dtype=np.float64)
```

`LstmAeScorer`: validate `F % n_mels == 0` in fit (ValueError mentioning `n_mels`); reshape `(N, frames, n_mels)`; encoder `torch.nn.LSTM(n_mels, hidden, batch_first=True)` → take last hidden state → repeat across frames → decoder LSTM(hidden, hidden) → `Linear(hidden, n_mels)`; MSE over the full patch. `ConvAeScorer`: reshape `(N, 1, n_mels, frames)`; encoder Conv2d(1→c1→c2, kernel 3, stride 2, padding 1) with ReLU; decoder mirrored ConvTranspose2d back to `(1, n_mels, frames)` (crop/pad to exact shape with `torch.nn.functional.interpolate` if the transpose shape is off by one — document); MSE over the patch. Same `_train_autoencoder`-style loop (factor the loop into the shared helper; all three use it).

Registry: add `"mlpae"/"lstmae"/"convae"` branches to both `_make_scorer`s and widen the Literals/choices as in Task 1.

- [ ] **Step 4: Suite + lints** (recon tests take ~1-2 min CPU) → pass. **Step 5: Commit**

```bash
git add src/rowii/anomaly/recon.py src/rowii/anomaly/sweep.py scripts/run_step2.py tests/test_recon.py
git commit -m "feat: reconstruction scorers (MLP-AE, LSTM-AE, Conv-AE) behind the torch extra"
```

---

### Task 4: Majority-ensemble evaluation view (design commitment)

**Files:**
- Modify: `scripts/run_step2.py` (new `--ensemble` flag, within-day only; view writer)
- Test: `tests/test_step2_cli.py` (extend)

**Interfaces:**
- Consumes: registry names `"ocsvm"`, `"iforest"`, `"lstmae"` (Tasks 1+3); `prepare_run` for BOTH the sweep variant and `logmel`; `split_by_segments` / `build_references` / `calibrate` exactly as `run_sweep` composes them; `far_row_*` public builders.
- Produces: with `--ensemble` (valid only with `--protocol within-day`), after the normal sweep outputs, an additional `far_table_ensemble.csv` in the combo dir with columns `member, label, n_calibration, n_scored, n_alarms, realized_far, low_confidence` where `member ∈ {ocsvm, iforest, lstmae, ENSEMBLE}` — members alarm via their own per-state conformal thresholds at the same alpha; `ENSEMBLE` rows count windows where >= 2 members alarm. OC-SVM/IF fit on the sweep variant's features; LSTM-AE on the logmel features of the SAME run.
- Grid-alignment guard: assert the logmel PreparedRun has identical `grid.t0_ns`, `grid.window_ns`, `grid.n_windows` to the variant's PreparedRun; on mismatch exit 2 with a clear message (structurally they match — same primary stream drives both grids — but assert, don't assume).
- Honesty note in the writer: "members hold their own marginal conformal guarantees; no distribution-free guarantee is claimed for the majority decision — empirical FARs only" (spec D4).

- [ ] **Step 1: Failing CLI test** — extend `tests/test_step2_cli.py` with an `--ensemble` smoke test on the existing synthetic fixture (monkeypatch torch-dependent LstmAeScorer with a lightweight stub via `monkeypatch.setattr(run_step2, "_ENSEMBLE_MEMBER_FACTORIES", ...)` — define that dict in the implementation exactly so tests can substitute members): assert exit 0; `far_table_ensemble.csv` exists; it has all four `member` values; for every state the ENSEMBLE row's `n_alarms <= max(member n_alarms)`; and a constructed-disagreement unit check: with stub members whose alarms are disjoint sets, ENSEMBLE alarms = 0.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.** `_ENSEMBLE_MEMBER_FACTORIES: dict[str, Callable[[], Scorer]] = {"ocsvm": OcSvmScorer, "iforest": IsolationForestScorer, "lstmae": LstmAeScorer}` at module level (test seam). New `_run_ensemble_view(prepared_variant, prepared_logmel, labels, cfg, alpha, out_dir)`: replicate `run_sweep`'s three-way split (same seeds 7/8) ONCE; per state with a min_ref reference: for each member, pick its feature matrix (logmel for lstmae, variant for the others), fit on the state's fit-side rows, calibrate on the state's conformal-side scores, alarm on the state's scoring-side windows; collect per-member rows + the >=2-of-3 ENSEMBLE row; write CSV + the honesty note as a sibling `ensemble_notes.md`. CLI: `--ensemble` flag; parser.error when combined with a non-within-day protocol; runs after `_write_sweep_outputs` inside the within-day loop, loading the logmel PreparedRun via `prepare_run(run, "logmel", cfg, use_cache=True)`.
- [ ] **Step 4: Suite + lints** → pass. **Step 5: Commit**

```bash
git add scripts/run_step2.py tests/test_step2_cli.py
git commit -m "feat: majority-ensemble evaluation view (OC-SVM + IF + LSTM-AE, decision level)"
```

---

### Task 5: Score-level fusion view (Fisher / Tippett p-value combination)

**Files:**
- Create: `src/rowii/anomaly/fusion.py`
- Modify: `scripts/run_step2.py` (`--score-fusion` flag, within-day + `--variant fusion` only)
- Test: `tests/test_fusion.py` (new), `tests/test_step2_cli.py` (one smoke case)

**Interfaces:**
- Consumes: `p_values` + `calibrate` from `rowii.anomaly.conformal`; `feature_names` stream prefixes from `PreparedRun` (audio columns = names whose stream prefix contains `"Mic"`, vibration = `"Vib"` — split on the `"::"` separator established by `_assemble_feature_names`).
- Produces (`src/rowii/anomaly/fusion.py`):
  - `split_branch_columns(feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]` — (audio_idx, vib_idx) int64 index arrays; raises ValueError if either branch is empty.
  - `fisher_statistic(p_a: np.ndarray, p_v: np.ndarray) -> np.ndarray` — `-2 * (log(p_a) + log(p_v))`, float64 (p-values are in (0,1] by conformal construction, no clipping needed; document).
  - `tippett_statistic(p_a, p_v) -> np.ndarray` — `-np.minimum(p_a, p_v)` … NOTE: for "higher = more anomalous" use `1.0 - np.minimum(p_a, p_v)`; pick ONE, document, and test the ordering.
  - Orchestration `_run_score_fusion_view(...)` in run_step2.py: per state — branch scorers (kNN default, `--score-fusion-scorer mahalanobis` optional) fit per branch on fit-side rows; branch p-values for BOTH conformal-side and scoring-side windows against the branch's conformal scores; combined statistic on conformal side → `calibrate` → threshold; combined statistic on scoring side → alarms; rows per state × rule ∈ {fisher, tippett} + per-branch single-branch baseline rows (audio-only, vib-only through the same conformal path) → `far_table_scorefusion.csv` (columns `rule, label, n_calibration, n_scored, n_alarms, realized_far, low_confidence`).
- Statistical note carried in the module docstring: combining the two branch p-values and RE-CALIBRATING the combined statistic on the held-out conformal side restores the distribution-free FAR guarantee regardless of branch dependence (the combination is just a score transform; conformal validity needs only exchangeability of the combined scores).

- [ ] **Step 1: Failing tests** (`tests/test_fusion.py`): `split_branch_columns` on realistic names (`"RAWGeneratorMic__0::ch0_log_rms"`, `"RAWGeneratorVib__2::ch1_kurtosis"`, …) returns disjoint exhaustive indices; empty-branch ValueError; `fisher_statistic` ordering (smaller ps → larger statistic; known value: p=(0.05,0.05) → −2(ln .05+ln .05) ≈ 11.98); tippett ordering; **guarantee-restored test**: synthetic exchangeable two-branch scores → full pipeline (branch p-values → fisher → calibrate on conformal half → alarm rate on scoring half) has realized FAR within the Beta band at n (reuse `beta_band` from `rowii.anomaly.scarcity` for the check bounds).
- [ ] **Step 2: verify failure.** **Step 3: implement** (fusion.py ~120 lines + orchestration; CLI guard: `--score-fusion` requires within-day + variant fusion, else parser.error). **Step 4: suite + lints.** **Step 5: Commit**

```bash
git add src/rowii/anomaly/fusion.py scripts/run_step2.py tests/test_fusion.py tests/test_step2_cli.py
git commit -m "feat: score-level fusion view (Fisher/Tippett p-value combination, re-calibrated)"
```

---

### Task 6: Conditioning-granularity flag (`--states K`)

**Files:**
- Modify: `scripts/run_step2.py` (`--states` int flag, within-day only; threads into `_detected_labels_and_detector` via `FittedDetector.fit(..., k=args.states)`)
- Test: `tests/test_step2_cli.py` (extend)

**Interfaces:**
- Consumes: `FittedDetector.fit(features, grid, cfg, clusterer, k)` (k param exists since package 2).
- Produces: `--states K` (default None = cfg.detect.n_states): detected labels come from a k=K detector; combo out-dir gains a `-k<K>` suffix ONLY when K is non-default (`fusion-detected-k8/`), so default runs stay byte-compatible with existing layouts; summary rows carry the same suffixed variant string. parser.error for K < 2 and for non-within-day protocols.

- [ ] **Step 1: Failing test**: within-day run with `--states 3` on the synthetic fixture → output dir `*-k3/` exists; far_table has <= 3 non-pooled state rows; `--states 1` → exit 2; cross-day + `--states` → exit 2. Also: labels really come from k: monkeypatch-spy `FittedDetector.fit` and assert `k=3` was passed.
- [ ] **Step 2: verify failure.** **Step 3: implement** (thread `k` through `_detected_labels_and_detector(prepared, cfg, k=None)` — extend that helper's signature with a defaulted param; `_within_day_out_dir` suffix). **Step 4: suite + lints.** **Step 5: Commit**

```bash
git add scripts/run_step2.py tests/test_step2_cli.py
git commit -m "feat: --states K conditioning-granularity flag (within-day)"
```

---

### Task 7: Execution + synthesis (orchestrator-led)

No new code. Real-data execution on warm caches (logmel extraction is the only one-off, ~10 min/run × 3):

- [ ] **Step 1:** warm logmel: `.venv/bin/python scripts/warm_cache.py --runs 250526-tu 290626-tu 010726-tu_ph_tu --variants logmel` (no torch needed — numpy featurizer).
- [ ] **Step 2:** classical scorers: for each of the 3 runs × variants {fusion, audio-beats} × `--scorer ocsvm`/`iforest`/`lof` × `--conditioning all`: `run_step2.py --protocol within-day`.
- [ ] **Step 3:** recon: 010726 + 250526 × {fusion: mlpae; logmel: lstmae, convae} × per-state (+ pooled on 010726); 290626 logmel lstmae (starved-regime flags expected, still run).
- [ ] **Step 4:** ensemble: 3 runs × fusion variant `--ensemble`.
- [ ] **Step 5:** score fusion: 3 runs × `--score-fusion` (knn; mahalanobis on 010726).
- [ ] **Step 6:** granularity: 010726 fusion × `--states 4/8/12` × knn+mahalanobis per-state; top-20 stability via `match_by_time` across K (small scratch analysis, results into the digest).
- [ ] **Step 7:** digest agent (same recipe as package 2) → README "Step-2 package 3 evidence" + research note in master-thesis + ledger; final whole-branch review; PR.

---

## Execution notes (orchestrator)

- Dependency order: T1 → T4 (needs ocsvm/iforest); T2 → T3 (logmel before patch-AEs) → T4 (lstmae member); T5 needs only conformal + package-1 infra; T6 independent. Suggested: T1, T2, T3, T5, T6, T4, T7.
- One implementer at a time; HARD no-delegation clause in every dispatch; no real-data runs inside implementer tasks (T7 is orchestrator-run).
- AE training budgets are deliberately small (spec: minutes per state); if real-data training is unexpectedly slow in T7, reduce epochs via constructor args in the execution commands, not code edits.

## Self-review (done at plan-writing time)

- Spec coverage: D1→T1, D2→T3, D3→T2, D4→T4, D5→T5, D6→T6, D7→T7. No gaps.
- Placeholder scan: T3's LSTM/Conv bodies are specified by contract + reference implementation to mirror (deliberate; the MLP body is complete and the mirroring is mechanical); T5's orchestration named with exact column contracts. No TBDs.
- Type consistency: registry names match class `name` attrs; `far_table_ensemble.csv`/`far_table_scorefusion.csv` column lists stated once each and reused; `--states` threading via the existing `k` param name.
