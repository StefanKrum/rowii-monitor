# Step-2 Package 8: Mode-Model-Bank, Level-Only Recalibration, Explainable Results — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build Stefan's per-mode model bank as a real Step-1 alternative + Step-2 chain probe (D1), a shape-preserving level-only channel recalibration on our own features and protocol (D2), an explainability analysis suite with publication-grade figures from existing caches (D3), and small scripted data verifications (D4) — every comparison honest, mapping-invariant, and firewalled from partner numbers.

**Architecture:** One greenfield module `src/rowii/state/modebank.py` (the package's real cost center) drives three CLIs (`run_modebank.py`, `run_modebank_chain.py`); one greenfield module `src/rowii/anomaly/levelrecal.py` backs `--level-recal` surfaces in `run_step2.py` and `monitor.py` (mirroring the P7 `--session-norm` wiring and snapshot-v2 pattern); `scripts/analyze_days.py` renders six figure families + a digest from existing/new artifacts; `scripts/verify_data_facts.py` runs the D4 probes. Everything composes existing machinery (`build_pool`, `split_by_segments`, `calibrate`, `gt_labels`, `FittedDetector.fit_pooled`, `fit_snapshot_from_parts`, `eval_events.py`) — no new encoders, no localization, no streaming.

**Tech Stack:** unchanged (numpy, sklearn KMeans/GaussianMixture/adjusted_rand_score, hmmlearn, pandas, scipy; matplotlib Agg for figures; torch lazily only for embedding variants).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-21-step2-package8-modebank-recal-explain.md` **§3 (D1–D4) + Amendment A1 (all 11 adopted findings A1.1–A1.11)** — the amendment OVERRIDES §3 where they conflict (esp. A1.1 fusion-excluded/D2-audio+vibration-only, A1.3 ARI-primary + `--smooth`=duration-filter-only, A1.4 offset sources + pillar-3 frozen mode, A1.5 bank internals, A1.6 D4 feasibility, A1.9 column-set cleanups, A1.10 snapshot policy, A1.11 remaining pins).
- Gates per task (ALL must be green before commit): `.venv/bin/python -m pytest tests/ -q -m "not data"`, `.venv/bin/ruff check .`, `.venv/bin/mypy src scripts` (must print `Success: no issues`).
- **Tests FIRST** (RED before GREEN) for every task. Deterministic seeded fixtures; NO real data in tests; CLI-level tests use the established monkeypatch seams (`tests/test_pools.py` hand-built `PreparedRun`s; `tests/test_monitor_cli.py` monkeypatched `discover`/`load_config`/`prepare_run`).
- **Implementer verification MUST use the temp-file + `&&` pattern**, never `pytest | tail` (a pipe hides the exit code): e.g. `.venv/bin/python -m pytest tests/test_modebank.py -q > /tmp/p8.txt 2>&1 && tail -5 /tmp/p8.txt`.
- **No `Co-Authored-By` lines** in any commit (Stefan is sole author).
- **Firewall (A1.8, BINDING TEST RULE):** NO partner-derived numeric constant may appear as an expected value in `src/`, `scripts/`, or test fixtures. The `3 dB` stability cutoff is a NAMED, adopted-for-comparability constant (label it "same cutoff as Rodrigues & Zhang (2026), adopted for comparability"), never asserted against a partner number; the CONTINUOUS shift distribution is the primary deliverable, the binary classification secondary.
- **Attribution (spec §4):** every analysis type inspired by the partner's work carries a one-line attribution in the script docstring AND the digest; all numbers are computed from OUR caches; no partner JSON/number is read by any code.
- **Model policy (Stefan 2026-07-21):** implementation/tests/readers on **sonnet**; adversarial spec/whole-branch reviews on **opus**; fable only if a review blocks twice.
- Scripts NEVER import a sibling script's internals (duplicate-with-rationale, the repo rule); `src/rowii/` modules are imported normally.

### Verified seam facts (recorded here so every interface below is real)

- **Feature log-scaling (VERIFIED in `src/rowii/signals/features.py`, load-bearing for D2/A1.1):** `*_log_rms`, `*_band_<name>`, `*_octave_<fc>` are ALL `_log10_floor(·)` = base-10 log with a `1e-12` floor (log_rms of RMS amplitude, band/octave of mean-PSD energy). `*_spectral_centroid` and `*_rolloff95` (audio) and `*_kurtosis` (vib) are RAW units (Hz / dimensionless), NOT log-scaled. So an additive offset on the level columns is an additive shift in the log10 domain (= a multiplicative gain in linear domain); the offset is computed directly as a difference of medians of the STORED log10 features, so it is self-consistently in log10 units — a partner "dB" figure is never our unit and is never imported (D2 computes its own offsets).
- **`fuse()` (VERIFIED):** `fuse(a,b) = np.hstack([zscore(a), zscore(b)])`, `zscore` = per-run mean/std standardization → fusion's stored features are dimensionless per-run z-scores. An additive dB offset is meaningless there and a broadband level step is ALREADY removed by fusion's own assembly ⇒ **D2 variants are `audio` and `vibration` only; `fusion` is excluded with a finding (A1.1).**
- **ARI mask (VERIFIED):** `rowii.eval.metrics.evaluate` drops only `gt.state == "unknown"`. Spec A1.5 requires masking BOTH `"unknown"` AND `"transition"` for the bank's ARI/accuracy, applied identically to the clusterer arm. ⇒ D1 computes ARI via `sklearn.metrics.adjusted_rand_score` directly on the `{unknown,transition}`-masked arrays (NOT via `evaluate`), and records the delta vs P7's `unknown`-only k-selection mask.
- **P7 pooled detector:** `results/step2/cross-day-pooled/k_selection.json` → `selected_k = 4`. No per-window P7 label artifact is persisted (cross-day-pooled writes only far/coverage tables) ⇒ D1's unsupervised comparison row RECOMPUTES the P7 pooled labels via `FittedDetector.fit_pooled(pool_fit.features, cfg, k=4).apply(...)` (deterministic), tagged unsupervised. `--p7-k` flag (default 4) exposes it.
- **Cache naming:** `results/cache/<run>--<variant>.npz`. Plain `vibration` caches DO NOT yet exist (only `vibration-tfc`) ⇒ T10 must warm `vibration` explicitly on every rotation/pool/080726 day.
- **Snapshot:** `MonitorSnapshot` is `@dataclass(frozen=True)` with optional v2 members already present (`session_stats`); `save_snapshot`/`load_snapshot` round-trip meta as JSON; `fit_snapshot_from_parts` is the pooled assembly path. Adding one more optional member follows the exact `session_stats` precedent (A1.10).

---

### Task 1: `scripts/verify_data_facts.py` — D4 data verifications (A1.6)

**Files:** Create `scripts/verify_data_facts.py` · Test `tests/test_verify_data_facts.py`

**Interfaces:**
- Consumes (src only): `rowii.io.gantner.read_gantner`/`read_header`, `rowii.io.dataset.discover`/`betriebsdaten_utc_offset_ns`, `rowii.scada.labels.load_scada_window_means`, `rowii.config.load_config`. IO seams are monkeypatched in tests.
- Produces (pure, unit-tested helpers — the load-bearing logic):
  - `block_is_dead(block: np.ndarray) -> bool` — `float64(block).std() < 1e-9` (the `VibFeaturizer` dead-channel criterion, in float64 as `features.py` documents).
  - `locate_changeover(ts_ns: np.ndarray, values: np.ndarray) -> int` — index `i>=1` of the largest `|values[i]-values[i-1]|` first-difference over finite neighbours (the regime step); ties → leftmost.
  - `channel_level_profile(levels: np.ndarray, strike_mask: np.ndarray) -> np.ndarray` — `(C,)` per-channel median level over the `strike_mask` rows of a `(W,C)` matrix.
  - `outlier_channel(profile: np.ndarray) -> int` — `argmax(|profile - median(profile)|)` (the ring-outlier index; channel-anonymous, no azimuth claim — A1.6).
  - subcommands `vib-ch0-liveness`, `scada-timebase`, `gen-mic-profile` on an argparse subparser.

- [ ] RED `tests/test_verify_data_facts.py`:
```python
"""Tests for scripts/verify_data_facts.py (Package-8 D4, A1.6): PURE-logic unit tests
on the variance criterion, the changeover locator (synthetic series), and the
channel-anonymous mic-profile math -- IO seams monkeypatched, no real data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import verify_data_facts as vdf  # noqa: E402


def test_block_is_dead_uses_float64_1e9_criterion() -> None:
    dead = np.full((500, 1), -7.0, dtype=np.float32)  # constant channel, float32 input
    assert vdf.block_is_dead(dead[:, 0]) is True
    live = np.random.default_rng(0).normal(size=500).astype(np.float32)
    assert vdf.block_is_dead(live) is False
    # A channel constant to float32 precision but re-cast: std must be exactly 0 in f64.
    near = np.full(500, 3.5, dtype=np.float32)
    assert vdf.block_is_dead(near) is True


def test_locate_changeover_finds_the_step_index() -> None:
    ts = np.arange(100, dtype=np.uint64) * np.uint64(1_000_000_000)
    values = np.concatenate([np.full(60, 0.0), np.full(40, 378.8)])  # step at index 60
    assert vdf.locate_changeover(ts, values) == 60
    # a noisy plateau then a jump: the JUMP wins over the noise.
    rng = np.random.default_rng(1)
    noisy = np.concatenate([rng.normal(0, 0.01, 60), rng.normal(100, 0.01, 40)])
    assert vdf.locate_changeover(ts, noisy) == 60


def test_channel_level_profile_and_outlier_are_channel_anonymous() -> None:
    # 4 mic channels; channel 2 sits ~6 dB above its ring at strike minutes.
    levels = np.zeros((200, 4), dtype=np.float64)
    levels[:, :] = -40.0
    strike = np.zeros(200, dtype=bool)
    strike[50:70] = True
    levels[strike, 0] = -40.0
    levels[strike, 1] = -40.2
    levels[strike, 2] = -34.0  # the outlier
    levels[strike, 3] = -39.8
    profile = vdf.channel_level_profile(levels, strike)
    assert profile.shape == (4,)
    assert vdf.outlier_channel(profile) == 2
    # NO azimuth asserted: the function returns an INDEX only.
    assert isinstance(vdf.outlier_channel(profile), int)


def test_locate_changeover_requires_two_points() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        vdf.locate_changeover(np.array([0], dtype=np.uint64), np.array([1.0]))
```
- [ ] Run RED: `.venv/bin/python -m pytest tests/test_verify_data_facts.py -q > /tmp/p8_t1.txt 2>&1 && tail -15 /tmp/p8_t1.txt` → import/collection error (module absent).
- [ ] GREEN `scripts/verify_data_facts.py` (module docstring carries the D4 attribution line + A1.6 no-azimuth note). Real sketch of the pure helpers + subcommand wiring:
```python
"""D4 data verifications (Package-8, spec D4 + A1.6). Three cheap scripted probes on
OUR own files, each independently attributed where it echoes the partner's data
work (Rodrigues & Zhang 2026): (1) generator-mic level anomaly at plate-strike
minutes -- CHANNEL-ANONYMOUS (no azimuth->channel map exists in-repo; A1.6); (2)
RAWTurbineVib__3 ch0 per-file DATA-VARIANCE liveness across days (std<1e-9, the
VibFeaturizer dead-channel criterion, A1.6); (3) SCADA timebase probe: locate the
080726 changeover in Betriebsdaten rpm/power and compare it against the audio-UTC
state timeline (13:05:28 UTC reference). No partner number is read by this script."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.config import load_config  # noqa: E402
from rowii.io.dataset import discover  # noqa: E402
from rowii.io.gantner import read_gantner  # noqa: E402

_DEAD_STD = 1e-9
_REFERENCE_UTC_DEFAULT = "2026-07-08T13:05:28+00:00"


def block_is_dead(block: np.ndarray) -> bool:
    return bool(float(np.asarray(block, dtype=np.float64).std()) < _DEAD_STD)


def locate_changeover(ts_ns: np.ndarray, values: np.ndarray) -> int:
    v = np.asarray(values, dtype=np.float64)
    if v.shape[0] < 2:
        raise ValueError("locate_changeover needs at least 2 samples")
    diffs = np.abs(np.diff(v))
    diffs[~np.isfinite(diffs)] = -np.inf
    return int(np.argmax(diffs)) + 1


def channel_level_profile(levels: np.ndarray, strike_mask: np.ndarray) -> np.ndarray:
    rows = np.asarray(levels, dtype=np.float64)[np.asarray(strike_mask, dtype=bool)]
    return np.median(rows, axis=0)


def outlier_channel(profile: np.ndarray) -> int:
    p = np.asarray(profile, dtype=np.float64)
    return int(np.argmax(np.abs(p - np.median(p))))
# ... argparse subparsers vib-ch0-liveness / scada-timebase / gen-mic-profile below,
#     each reading real files via read_gantner / discover / pd.read_csv(comment="#")
#     and calling the pure helpers; gen-mic-profile reads the ground-truth CSV with
#     the comment='#' contract (mirrors monitor._load_exclusion_intervals).
```
  - `scada-timebase` accepts `--reference-utc` (default `_REFERENCE_UTC_DEFAULT`), loads the 080726 Betriebsdaten via `read_gantner`, calls `locate_changeover` on the rpm then power channel, and prints both changeover timestamps + the signed delta vs `--reference-utc`.
  - `gen-mic-profile` reads a ground-truth CSV (`pd.read_csv(path, comment="#")`, `start_utc`/`end_utc` tz-aware), builds a strike-window mask over the run grid, computes `channel_level_profile` + `outlier_channel` across ALL generator-mic channels, prints the profile and the outlier index (no azimuth).
- [ ] Run GREEN gates (all three) with temp-file+&&; expect pass + `Success: no issues`.
- [ ] Commit `feat: verify_data_facts D4 probes (vib-ch0 liveness, SCADA timebase, mic profile) (P8 D4/A1.6)`.

---

### Task 2: `src/rowii/state/modebank.py` — per-mode model bank (D1 core, A1.5)

**Files:** Create `src/rowii/state/modebank.py` · Test `tests/test_modebank.py`

**Interfaces:**
- Consumes: `rowii.anomaly.scorers.KnnScorer`/`MahalanobisScorer`, `rowii.anomaly.conformal.calibrate`/`ConformalThreshold`, `rowii.signals.features.zscore_stats`/`apply_zscore`, `sklearn.mixture.GaussianMixture`.
- Produces:
  - `_FAMILIES = ("gaussian", "knn", "gmm")`; `_EXCLUDED_GT = ("unknown", "transition")` (A1.5).
  - `class GmmModeScorer` — `fit(reference)`/`score(x)` on the shared Scorer contract (higher = more anomalous = `-GaussianMixture(n_components, covariance_type="diag").score_samples(x)`), polarity by construction.
  - `@dataclass(frozen=True) class ModeAssignment`: `labels: np.ndarray` (W, object mode-name), `scores: np.ndarray` (W, M, per-mode anomaly score in `modes` order), `modes: list[str]`, `no_mode_fits: np.ndarray` (W, bool).
  - `@dataclass(frozen=True) class ModeBank`: `family: str`, `modes: list[str]` (sorted survivors), `members: dict[str, Scorer]` (per-mode fitted scorer), `mean: np.ndarray | None`/`std: np.ndarray | None` (pool-fit global standardization, gaussian/gmm only; `None` for knn), `thresholds: dict[str, ConformalThreshold]`, `calibration_scores: dict[str, np.ndarray]`, `dropped_modes: dict[str, int]` (mode → n fit windows, for below-floor/no-calib drops), `feature_names: list[str]`, `alpha: float`, `min_ref: int`, `k: int`, `gmm_components: int`.
  - `ModeBank.fit(fit_features, fit_labels, calib_features, calib_labels, *, family, alpha, feature_names, min_ref=20, k=5, gmm_components=2, random_seed=7) -> ModeBank` — exclude `_EXCLUDED_GT` from BOTH sides; global mean/std from `fit_features` (pool-fit-side global, gaussian/gmm; A1.5); per surviving mode (fit count `>= min_ref` AND `>= 1` calib window) fit the per-mode scorer (`MahalanobisScorer` on standardized rows for gaussian, `KnnScorer(k, "cosine")` on RAW rows for knn, `GmmModeScorer(gmm_components)` on standardized rows for gmm) and `calibrate` its threshold on that mode's own calib-side scores; a mode below `min_ref` OR with zero calib windows is dropped with a `coverage_warnings`-style WARNING and recorded in `dropped_modes`.
  - `ModeBank.assign(features) -> ModeAssignment` — per mode score `members[m].score(transform(x))` (`transform` = standardize for gaussian/gmm, identity for knn); `labels = modes[argmin]`; `no_mode_fits[w] = all(score[w,j] > thresholds[modes[j]].threshold for all j)` (rejected by every member — Stefan's "keins passt"). Empty bank → `labels=""`, `scores` (W,0), `no_mode_fits` all False.
- Binding: `family` NOT in `_FAMILIES` → `ValueError`; feature width mismatch → `ValueError` (loud geometry, snapshot posture).

- [ ] RED `tests/test_modebank.py` (full battery per spec §5):
```python
"""Tests for rowii.state.modebank (Package-8 D1 core, A1.5): per-family fit/score
shapes, argmin assignment, conformal rejection (incl. all-rejected), min_ref
floor + dropped modes, unknown/transition exclusion, global-standardization
storage, empty-bank degeneracy. Deterministic seeded blobs, no real data."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from rowii.state.modebank import ModeBank, _EXCLUDED_GT

_NAMES = [f"ch0_octave_{i}" for i in range(4)]  # 4 level-ish columns


def _two_mode(seed: int, n_per: int = 60):
    """Two well-separated GT modes ('turbine'@0, 'pump'@10) over 4 features."""
    rng = np.random.default_rng(seed)
    feats = np.vstack([rng.normal(0.0, 0.2, (n_per, 4)), rng.normal(10.0, 0.2, (n_per, 4))])
    labels = np.array(["turbine"] * n_per + ["pump"] * n_per, dtype=object)
    return feats, labels


@pytest.mark.parametrize("family", ["gaussian", "knn", "gmm"])
def test_fit_assign_recovers_two_modes(family: str) -> None:
    fit_f, fit_l = _two_mode(0)
    cal_f, cal_l = _two_mode(1)
    bank = ModeBank.fit(fit_f, fit_l, cal_f, cal_l, family=family, alpha=0.05, feature_names=_NAMES)
    assert set(bank.modes) == {"turbine", "pump"}
    query_f, query_l = _two_mode(2, n_per=40)
    a = bank.assign(query_f)
    assert a.labels.shape == (80,)
    assert a.scores.shape == (80, 2)
    # argmin assignment: >= 95% of windows land on their true GT mode.
    assert float(np.mean(a.labels == query_l)) >= 0.95


def test_gaussian_gmm_store_global_standardization_knn_does_not() -> None:
    fit_f, fit_l = _two_mode(0)
    cal_f, cal_l = _two_mode(1)
    g = ModeBank.fit(fit_f, fit_l, cal_f, cal_l, family="gaussian", alpha=0.05, feature_names=_NAMES)
    assert g.mean is not None and g.std is not None
    np.testing.assert_allclose(g.mean, fit_f[~np.isin(fit_l, _EXCLUDED_GT)].mean(axis=0))
    knn = ModeBank.fit(fit_f, fit_l, cal_f, cal_l, family="knn", alpha=0.05, feature_names=_NAMES)
    assert knn.mean is None and knn.std is None


def test_unknown_and_transition_excluded_from_training(caplog) -> None:
    fit_f, fit_l = _two_mode(0)
    fit_l = fit_l.copy()
    fit_l[:5] = "unknown"
    fit_l[5:10] = "transition"
    cal_f, cal_l = _two_mode(1)
    bank = ModeBank.fit(fit_f, fit_l, cal_f, cal_l, family="gaussian", alpha=0.05, feature_names=_NAMES)
    assert "unknown" not in bank.modes and "transition" not in bank.modes


def test_below_min_ref_mode_is_dropped_with_warning(caplog) -> None:
    fit_f, fit_l = _two_mode(0)
    # shrink 'pump' to 3 fit windows (< min_ref=20): dropped.
    keep = np.concatenate([np.arange(60), np.arange(60, 63)])
    fit_f, fit_l = fit_f[keep], fit_l[keep]
    cal_f, cal_l = _two_mode(1)
    with caplog.at_level(logging.WARNING):
        bank = ModeBank.fit(fit_f, fit_l, cal_f, cal_l, family="knn", alpha=0.05, feature_names=_NAMES)
    assert bank.modes == ["turbine"]
    assert bank.dropped_modes.get("pump") == 3
    assert any("pump" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def test_no_mode_fits_flags_a_far_out_window() -> None:
    fit_f, fit_l = _two_mode(0)
    cal_f, cal_l = _two_mode(1)
    bank = ModeBank.fit(fit_f, fit_l, cal_f, cal_l, family="gaussian", alpha=0.05, feature_names=_NAMES)
    # one window from each true mode (fits) + one wildly out-of-distribution window.
    query = np.vstack([np.zeros((1, 4)), np.full((1, 4), 10.0), np.full((1, 4), 500.0)])
    a = bank.assign(query)
    assert a.no_mode_fits[0] == False  # noqa: E712 -- fits 'turbine'
    assert a.no_mode_fits[1] == False  # fits 'pump'
    assert a.no_mode_fits[2] == True   # noqa: E712 -- rejected by BOTH members


def test_unknown_family_and_width_mismatch_raise() -> None:
    fit_f, fit_l = _two_mode(0)
    cal_f, cal_l = _two_mode(1)
    with pytest.raises(ValueError, match="family"):
        ModeBank.fit(fit_f, fit_l, cal_f, cal_l, family="forest", alpha=0.05, feature_names=_NAMES)
    bank = ModeBank.fit(fit_f, fit_l, cal_f, cal_l, family="knn", alpha=0.05, feature_names=_NAMES)
    with pytest.raises(ValueError, match="width|column"):
        bank.assign(np.zeros((5, 3)))
```
- [ ] Run RED with temp-file+&&; expect ImportError.
- [ ] GREEN `src/rowii/state/modebank.py` (real sketch — the standardization/argmin/reject core, no placeholders):
```python
"""Per-mode model bank (Stefan's idea, Package-8 D1 core, spec §3.D1 + A1.5): SCADA
GT modes on the FIT days train one small model per mode (three families: diagonal
Gaussian on standardized features, per-mode cosine-kNN on raw features, 2-component
diagonal GMM on standardized features). At apply time the bank runs LABEL-FREE:
argmin distance / argmax likelihood assigns each window a mode, and a per-mode
split-conformal threshold on that mode's own calibration-side scores flags a window
rejected by EVERY member as `no_mode_fits` (the "keins passt" novelty signal,
reported as a rate, never a detector without induced-event evidence). GT `unknown`
AND `transition` windows are excluded from training (A1.5)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture

from rowii.anomaly.conformal import ConformalThreshold, calibrate
from rowii.anomaly.scorers import KnnScorer, MahalanobisScorer, Scorer
from rowii.signals.features import apply_zscore, zscore_stats

logger = logging.getLogger(__name__)
_FAMILIES = ("gaussian", "knn", "gmm")
_EXCLUDED_GT = ("unknown", "transition")


class GmmModeScorer:
    name = "gmm"
    def __init__(self, n_components: int = 2, random_seed: int = 7) -> None:
        self._m = GaussianMixture(n_components=n_components, covariance_type="diag",
                                  random_state=random_seed)
    def fit(self, reference: np.ndarray) -> "GmmModeScorer":
        self._m.fit(reference); return self
    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(-self._m.score_samples(x), dtype=np.float64)


@dataclass(frozen=True)
class ModeAssignment:
    labels: np.ndarray
    scores: np.ndarray
    modes: list[str]
    no_mode_fits: np.ndarray


@dataclass(frozen=True)
class ModeBank:
    family: str
    modes: list[str]
    members: dict[str, Scorer]
    mean: np.ndarray | None
    std: np.ndarray | None
    thresholds: dict[str, ConformalThreshold]
    calibration_scores: dict[str, np.ndarray]
    dropped_modes: dict[str, int]
    feature_names: list[str]
    alpha: float
    min_ref: int
    k: int
    gmm_components: int

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self.family == "knn":
            return np.asarray(x, dtype=np.float64)
        assert self.mean is not None and self.std is not None
        return apply_zscore(np.asarray(x, dtype=np.float64), self.mean, self.std)

    @classmethod
    def fit(cls, fit_features, fit_labels, calib_features, calib_labels, *, family, alpha,
            feature_names, min_ref=20, k=5, gmm_components=2, random_seed=7) -> "ModeBank":
        if family not in _FAMILIES:
            raise ValueError(f"family must be one of {_FAMILIES}, got {family!r}")
        ff = np.asarray(fit_features, dtype=np.float64)
        cf = np.asarray(calib_features, dtype=np.float64)
        fl = np.asarray(fit_labels, dtype=object)
        cl = np.asarray(calib_labels, dtype=object)
        fit_ok = ~np.isin(fl, _EXCLUDED_GT)
        cal_ok = ~np.isin(cl, _EXCLUDED_GT)
        ff, fl, cf, cl = ff[fit_ok], fl[fit_ok], cf[cal_ok], cl[cal_ok]
        mean = std = None
        if family in ("gaussian", "gmm"):
            mean, std = zscore_stats(ff)
        def _tf(x):  # standardize per family
            return apply_zscore(x, mean, std) if family in ("gaussian", "gmm") else x
        members: dict[str, Scorer] = {}
        thresholds: dict[str, ConformalThreshold] = {}
        cal_scores: dict[str, np.ndarray] = {}
        dropped: dict[str, int] = {}
        for mode in sorted(set(fl.tolist())):
            rows = ff[fl == mode]; n = int(rows.shape[0])
            if n < min_ref:
                dropped[mode] = n
                logger.warning("modebank: mode %r has %d fit window(s) < min_ref=%d -- dropped", mode, n, min_ref)
                continue
            cal_rows = cf[cl == mode]
            if cal_rows.shape[0] == 0:
                dropped[mode] = n
                logger.warning("modebank: mode %r has a reference but ZERO calibration window(s) -- dropped", mode)
                continue
            if family == "gaussian":
                scorer: Scorer = MahalanobisScorer().fit(_tf(rows))
            elif family == "knn":
                scorer = KnnScorer(k=k, metric="cosine").fit(rows)
            else:
                scorer = GmmModeScorer(gmm_components, random_seed).fit(_tf(rows))
            scores = scorer.score(_tf(cal_rows))
            members[mode] = scorer
            cal_scores[mode] = scores
            thresholds[mode] = calibrate(scores, alpha)
        if not members:
            logger.warning("modebank: NO mode survived the min_ref/calibration floors -- empty bank")
        return cls(family=family, modes=sorted(members), members=members, mean=mean, std=std,
                   thresholds=thresholds, calibration_scores=cal_scores, dropped_modes=dropped,
                   feature_names=list(feature_names), alpha=alpha, min_ref=min_ref, k=k,
                   gmm_components=gmm_components)

    def assign(self, features: np.ndarray) -> ModeAssignment:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError(f"assign: features must be 2-D with {len(self.feature_names)} column(s), got {x.shape}")
        w = x.shape[0]
        if not self.modes:
            return ModeAssignment(np.full(w, "", dtype=object), np.zeros((w, 0)), [], np.zeros(w, dtype=bool))
        xt = self._standardize(x)
        cols = [self.members[m].score(xt) for m in self.modes]
        scores = np.column_stack(cols)
        labels = np.array([self.modes[j] for j in np.argmin(scores, axis=1)], dtype=object)
        rejected = np.ones(w, dtype=bool)
        for j, m in enumerate(self.modes):
            rejected &= scores[:, j] > self.thresholds[m].threshold
        return ModeAssignment(labels=labels, scores=scores, modes=list(self.modes), no_mode_fits=rejected)
```
- [ ] Run GREEN gates; expect pass + `Success: no issues`.
- [ ] Commit `feat: ModeBank per-mode model bank (gaussian/knn/gmm, conformal rejection) (P8 D1/A1.5)`.

---

### Task 3: `scripts/run_modebank.py` — bank rotations CLI (D1)

**Files:** Create `scripts/run_modebank.py` · Test `tests/test_run_modebank_cli.py`

**Interfaces:**
- Consumes: `ModeBank`, `rowii.anomaly.pools.build_pool`, `rowii.pipeline.prepare_run`, `rowii.io.dataset.discover`, `rowii.config.load_config`, `rowii.scada.labels.load_scada_window_means`/`gt_labels`, `rowii.state.detect.FittedDetector.fit_pooled`, `rowii.state.segments.duration_filter`, `sklearn.metrics.adjusted_rand_score`. Duplicates run_step2's `_load_run_scada`/`_run_day_groups`-style helpers (script-sibling rule) — mic-GT via `load_scada_window_means`+`gt_labels`.
- Produces: CLI `run_modebank.py --fit-runs <csv> --test-run <name> --variant <v> --family <gaussian|knn|gmm> [--alpha 0.05] [--k 5] [--smooth] [--p7-k 4] [--min-ref 20]`.
  - Fit: `build_pool(prepared_fit, "fit", sweep_cfg)` → pooled fit rows; GT mode label per pooled fit row via `_pool_gt_labels` (each member's GT string array indexed by the pool's `run_index`/`window_index`); `build_pool(prepared_fit, "conformal", sweep_cfg)` → calib rows + GT labels; `ModeBank.fit(fit_rows, fit_gt, calib_rows, calib_gt, family=..., alpha=..., feature_names=..., min_ref=..., k=...)`.
  - Evaluate held-out run: `assign(prepared_test.features[valid])`; predicted mode labels scattered to full grid; `--smooth` applies `duration_filter` ONLY (A1.3 — never `smooth.fit_decode`) after mapping mode-strings→dense ids and back.
  - **Metrics (A1.3/A1.5):** ARI PRIMARY = `adjusted_rand_score` on windows masked to GT `not in {"unknown","transition"}` (mask both arms identically); bank accuracy = fraction of masked windows with `assigned == gt_state` (supervised — the bank's modes ARE GT names, direct equality); confusion CSV (GT × assigned); `no_mode_fits` rate over valid windows.
  - **Comparison row (unsupervised):** `FittedDetector.fit_pooled(pool_fit.features, cfg, k=args.p7_k).apply(test_valid)` → cluster ids; ARI under the SAME `{unknown,transition}` mask; tagged `unsupervised` in the row (no accuracy — cluster ids have no GT identity).
  - Artifacts under `results/step2/modebank/<test_run>/<variant>-<family>/`: `metrics.json` (bank ARI/accuracy/no_mode_fits + P7-comparison ARI + supervised/unsupervised tags), `confusion.csv`, `assignments.parquet` (window, gt_state, assigned, no_mode_fits), `notes.md` (pool composition + attribution).
- Binding: fusion/vibration/audio-beats representations allowed (A1 keeps fusion for D1 — the bank operates on any variant's features, unlike D2); `--family all` NOT supported (one family per invocation, mirrors run_step2 one-scorer rule); day-group disjointness reused from run_step2's `_run_day_groups` pattern (duplicated).

- [ ] RED `tests/test_run_modebank_cli.py` (monkeypatch `discover`/`load_config`/`prepare_run` + a fake SCADA seam feeding hand-built GT, mirroring `test_monitor_cli.py`):
```python
"""CLI tests for scripts/run_modebank.py (Package-8 D1): monkeypatched prepare/discover
seams with hand-built PreparedRuns + a fake per-run GT map, verifying artifact shapes,
the {unknown,transition} ARI mask, the supervised/unsupervised tags, and --smooth =
duration-filter-only (A1.3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from rowii.config import Config, DetectConfig  # noqa: E402
from rowii.io.dataset import RecordingIndex, Run  # noqa: E402
from rowii.pipeline import PreparedRun  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402

_W = 1_000_000_000


def _prepared(t0, seed, n_seg=8, seg=30):
    rng = np.random.default_rng(seed)
    feats, ids, gt = [], [], []
    for s in range(n_seg):
        mode = s % 2  # alternate turbine / pump
        feats.append(rng.normal(0.0 if mode == 0 else 10.0, 0.2, (seg, 4)))
        ids.append(np.full(seg, s, dtype=np.int64))
        gt.append(np.array((["turbine"] if mode == 0 else ["pump"]) * seg, dtype=object))
    f = np.vstack(feats)
    p = PreparedRun(features=f, grid=WindowGrid(t0, _W, len(f)),
                    valid_mask=np.ones(len(f), dtype=bool),
                    feature_names=[f"ch0_octave_{i}" for i in range(4)],
                    segment_ids=np.concatenate(ids))
    return p, np.concatenate(gt)


def _install(monkeypatch, mod, results_root, prepared_by_run, gt_by_run):
    runs = [Run(name=n, files={}, day_root=Path(f"/d/{n}")) for n in prepared_by_run]
    monkeypatch.setattr(mod, "discover", lambda dr: RecordingIndex(runs=runs, betriebsdaten=[], betriebsdaten_by_day={}))
    monkeypatch.setattr(mod, "load_config", lambda: Config(data_root=Path("/d"), results_root=results_root,
                                                           detect=DetectConfig(n_states=2, min_dwell_s=3.0)))
    monkeypatch.setattr(mod, "prepare_run", lambda run, variant, cfg, *, use_cache: prepared_by_run[run.name])
    monkeypatch.setattr(mod, "_run_gt_states", lambda prepared, run, index, cfg: gt_by_run[run.name])


def test_run_modebank_writes_metrics_with_supervised_and_unsupervised_tags(tmp_path, monkeypatch):
    import run_modebank as rm
    pf1, g1 = _prepared(0, 1)
    pf2, g2 = _prepared(0, 2)          # second fit day (different calendar day names below)
    pt, gt = _prepared(9_000_000_000, 3)
    prepared = {"fitA": pf1, "fitB": pf2, "testC": pt}
    gts = {"fitA": g1, "fitB": g2, "testC": gt}
    _install(monkeypatch, rm, tmp_path / "results", prepared, gts)
    monkeypatch.setattr(rm, "_run_day_groups", lambda run: {run.name})  # force disjoint
    code = rm.main(["--fit-runs", "fitA,fitB", "--test-run", "testC",
                    "--variant", "fusion", "--family", "gaussian", "--alpha", "0.05", "--min-ref", "10"])
    assert code == 0
    out = tmp_path / "results" / "step2" / "modebank" / "testC" / "fusion-gaussian"
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["bank"]["tag"] == "supervised"
    assert metrics["p7_pooled"]["tag"] == "unsupervised"
    assert 0.0 <= metrics["bank"]["ari"] <= 1.0
    assert "accuracy" in metrics["bank"] and "accuracy" not in metrics["p7_pooled"]
    assert 0.0 <= metrics["bank"]["no_mode_fits_rate"] <= 1.0
    assert (out / "confusion.csv").is_file()
    assign = pd.read_parquet(out / "assignments.parquet")
    assert set(assign.columns) >= {"window", "gt_state", "assigned", "no_mode_fits"}


def test_ari_mask_excludes_unknown_and_transition(monkeypatch, tmp_path):
    """The masked-ARI helper drops BOTH labels (A1.5), unlike eval.metrics.evaluate."""
    import run_modebank as rm
    gt = np.array(["turbine", "unknown", "transition", "pump", "pump"], dtype=object)
    pred = np.array(["turbine", "pump", "turbine", "pump", "pump"], dtype=object)
    ari, n = rm._masked_ari(gt, pred)
    assert n == 3  # only turbine/pump/pump counted
    assert ari == pytest.approx(1.0)


def test_smooth_uses_duration_filter_only(monkeypatch):
    import run_modebank as rm
    # a single 1-window flicker between two long runs is removed by duration_filter,
    # and the smoothing path must NOT call any HMM re-estimation.
    labels = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.int64)
    out = rm._smooth_ids(labels, min_dwell=3)
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 0]
```
- [ ] Run RED with temp-file+&&; expect ImportError/collection failure.
- [ ] GREEN `scripts/run_modebank.py`. Key pure helpers (unit-tested above) + wiring:
```python
def _masked_ari(gt: np.ndarray, pred: np.ndarray) -> tuple[float, int]:
    from sklearn.metrics import adjusted_rand_score
    mask = ~np.isin(np.asarray(gt, dtype=object), ("unknown", "transition"))
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float(adjusted_rand_score(gt[mask], pred[mask])), n

def _smooth_ids(labels: np.ndarray, min_dwell: int) -> np.ndarray:
    from rowii.state.segments import duration_filter  # A1.3: duration filter ONLY
    return duration_filter(labels, min_dwell=min_dwell)
```
  - `_run_gt_states(prepared, run, index, cfg)` mirrors run_step2's `_load_run_scada`+`_gt_state_labels` (duplicated): SCADA means → `gt_labels(...)["state"].to_numpy()`; monkeypatched in tests.
  - `_pool_gt_labels(pool, gt_by_run)` mirrors run_step2's `_pool_row_labels` but stacks GT strings via `pool.run_index`/`pool.window_index`.
  - The P7-comparison ARI recomputes `FittedDetector.fit_pooled(pool_fit.features, cfg, k=args.p7_k).apply(...)` and calls `_masked_ari(gt_masked, cluster_ids)`; both arms masked identically.
  - `--smooth`: map assigned mode-strings → dense int ids (sorted), `duration_filter(ids, min_dwell=max(1, round(min_dwell_s/window_s)))`, map back to strings before scoring metrics.
  - `notes.md` carries the pool composition + one attribution line ("Per-mode model bank inspired by the partner's per-state modeling; all numbers computed from our caches").
- [ ] Run GREEN gates; expect pass + `Success: no issues`.
- [ ] Commit `feat: run_modebank rotations CLI -- bank ARI/accuracy vs P7 pooled (masked, tagged) (P8 D1/A1.3)`.

---

### Task 4: `scripts/run_modebank_chain.py` — bank Step-2 chain probe (D1)

**Files:** Create `scripts/run_modebank_chain.py` · Test `tests/test_run_modebank_chain.py`

**Interfaces:**
- Consumes: `ModeBank`, `rowii.anomaly.references.split_by_segments`/`build_references`, `rowii.anomaly.conformal.calibrate`/`p_values`, `rowii.anomaly.scorers.KnnScorer`, `build_pool`, `prepare_run`. Reuses the run_step2 `_FAR_TABLE_COLUMNS` schema (duplicated) so `far_table.csv` is comparable to the P7 chain.
- Produces: CLI `run_modebank_chain.py --fit-runs <csv> --test-run <name> --variant fusion --family <f> [--alpha 0.05] [--scorer knn]`.
  - Bank labels (T2) drive per-mode conditioning: on the fit pool, `build_pool("fit")`/`build_pool("conformal")` → per-mode references (fit side) keyed by BANK-assigned mode, per-mode conformal thresholds (conformal side). **Split parity (BINDING):** the held-out day's top split uses the SAME `split_by_segments(prepared_test.segment_ids, prepared_test.valid_mask, calibration_frac=0.5, seed=7)` call run_step2's cross-day-pooled makes — so the scoring window set is byte-identical to P7's chain and the FAR is apples-to-apples.
  - Score the held-out day's SCORING-side windows under their BANK-assigned mode against that mode's reference/threshold → `far_table.csv` (per-mode rows + aggregate `pooled` row), comparable to the P7 detected-state chain.
- Binding: this is the D1 "does better state assignment translate into better FAR control?" probe (fusion, alpha 0.05).

- [ ] RED `tests/test_run_modebank_chain.py`:
```python
"""Tests for scripts/run_modebank_chain.py (Package-8 D1 chain probe): split-parity
vs run_step2's top split (BINDING) + FAR math on synthetic two-mode data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from rowii.anomaly.references import split_by_segments  # noqa: E402


def test_top_split_parity_with_run_step2_convention():
    import run_modebank_chain as rc
    seg = np.repeat(np.arange(10, dtype=np.int64), 20)
    valid = np.ones(200, dtype=bool)
    got = rc._top_split(seg, valid)  # the chain's own call
    ref = split_by_segments(seg, valid, 0.5, 7)  # run_step2 cross-day-pooled convention
    np.testing.assert_array_equal(got.calibration_windows, ref.calibration_windows)
    np.testing.assert_array_equal(got.scoring_windows, ref.scoring_windows)


def test_far_math_on_synthetic_two_mode_scoring():
    import run_modebank_chain as rc
    # 8 scored windows, 2 flagged as alarms -> realized_far = 0.25.
    scores = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 9.0, 9.0])
    threshold = 1.0
    far, n_alarm, n = rc._far(scores, threshold)
    assert (n, n_alarm) == (8, 2)
    assert far == 0.25
```
- [ ] Run RED with temp-file+&&; expect ImportError.
- [ ] GREEN `scripts/run_modebank_chain.py`:
```python
_TOP_SEED = 7
_TOP_FRAC = 0.5

def _top_split(segment_ids, valid_mask):
    from rowii.anomaly.references import split_by_segments  # run_step2 cross-day-pooled parity
    return split_by_segments(segment_ids, valid_mask, _TOP_FRAC, _TOP_SEED)

def _far(scores, threshold):
    n = int(scores.shape[0]); n_alarm = int((scores > threshold).sum())
    return (n_alarm / n if n else float("nan")), n_alarm, n
```
  - Fit the bank (T2) on the pool GT; for the fit pool, assign bank modes to pooled FIT and CONFORMAL rows; `build_references(pool_fit_features, bank_modes_fit, all_fit_windows)`-style per-mode references keyed by mode string; `calibrate` per-mode thresholds on the conformal-side per-mode scores; assign bank modes to the held-out day's scoring windows and score each under its mode; assemble `far_table.csv` (reuse the run_step2 `far_row_*` semantics; duplicate the column tuple).
- [ ] Run GREEN gates; expect pass + `Success: no issues`.
- [ ] Commit `feat: run_modebank_chain -- bank-labeled Step-2 FAR chain (split-parity with P7) (P8 D1)`.

---

### Task 5: `src/rowii/anomaly/levelrecal.py` — level-only recalibration core (D2, A1.1/A1.4/A1.9)

**Files:** Create `src/rowii/anomaly/levelrecal.py` · Test `tests/test_levelrecal.py`

**Interfaces:**
- Produces:
  - `_LEVEL_SUBSTRINGS = ("_log_rms", "_band_", "_octave_")` (the VERIFIED log10 level columns); `_SHAPE_SUBSTRINGS = ("_spectral_centroid", "_rolloff95", "_kurtosis")` (raw-unit, never touched) — recorded in the module docstring as the verified `features.py` fact.
  - `level_columns(feature_names: list[str]) -> list[int]` — indices whose name contains any `_LEVEL_SUBSTRINGS` token (may be empty for embedding variants).
  - `column_medians(rows: np.ndarray, feature_names: list[str]) -> dict[str, float]` — name-keyed per-column median over the LEVEL columns of `rows`; raises `ValueError` on an empty level set (A1.9 guard).
  - `level_recal_offsets(run_median: dict[str, float], reference_median: dict[str, float]) -> dict[str, float]` — per shared level key `run_median[k] - reference_median[k]` (spec T5 wording); raises on empty intersection.
  - `apply_level_recal(features: np.ndarray, feature_names: list[str], offsets: dict[str, float]) -> np.ndarray` — fresh float64 copy with `out[:, j] -= offsets[name]` for each named level column; SHAPE columns untouched by construction; raises `ValueError` on an empty level set (A1.9). Aligns the run-being-recal'd's level onto the reference/anchor.
- Binding: additive in the log10 domain (docstring); `fuse` per-run z-score means fusion has no meaningful level columns — the empty-set guard is exactly the A1.9 embedding/fusion refusal path.

- [ ] RED `tests/test_levelrecal.py`:
```python
"""Tests for rowii.anomaly.levelrecal (Package-8 D2 core, A1.1/A1.4/A1.9): level vs
shape column selection (the verified log10 fact), offset golden math, shape columns
untouched, empty-set guard (embedding/fusion variants)."""
from __future__ import annotations

import numpy as np
import pytest

from rowii.anomaly.levelrecal import (
    apply_level_recal, column_medians, level_columns, level_recal_offsets,
)

_NAMES = [
    "RAWGeneratorMic__0::ch0_log_rms",
    "RAWGeneratorMic__0::ch0_band_shaft",
    "RAWGeneratorMic__0::ch0_octave_125",
    "RAWGeneratorMic__0::ch0_spectral_centroid",  # shape
    "RAWGeneratorMic__0::ch0_rolloff95",          # shape
    "RAWTurbineVib__3::ch0_kurtosis",             # shape
]


def test_level_columns_selects_only_log_scaled_features() -> None:
    assert level_columns(_NAMES) == [0, 1, 2]  # log_rms, band, octave -- NOT centroid/rolloff/kurtosis


def test_offsets_and_apply_align_run_to_reference() -> None:
    rng = np.random.default_rng(0)
    feats = rng.normal(0.0, 0.1, (200, 6))
    feats[:, :3] += np.array([-40.0, -35.0, -30.0])  # a +level run
    run_med = column_medians(feats, _NAMES)
    ref_med = {n: run_med[n] - 2.0 for n in run_med}  # reference sits 2 (log10 units) below
    offsets = level_recal_offsets(run_med, ref_med)
    assert all(abs(v - 2.0) < 1e-9 for v in offsets.values())  # run - reference = +2 everywhere
    out = apply_level_recal(feats, _NAMES, offsets)
    # level columns recentred onto the reference; shape columns bit-identical.
    np.testing.assert_allclose(np.median(out[:, :3], axis=0),
                               [run_med[_NAMES[i]] - 2.0 for i in range(3)], atol=1e-9)
    np.testing.assert_array_equal(out[:, 3:], feats[:, 3:])


def test_empty_level_set_raises_for_embedding_variant() -> None:
    embedding_names = [f"RAWGeneratorMic__0::dim{i}" for i in range(8)]  # no level pattern
    assert level_columns(embedding_names) == []
    with pytest.raises(ValueError, match="no level column"):
        column_medians(np.zeros((5, 8)), embedding_names)
    with pytest.raises(ValueError, match="no level column"):
        apply_level_recal(np.zeros((5, 8)), embedding_names, {})


def test_docstring_records_the_verified_log_scale_fact() -> None:
    import rowii.anomaly.levelrecal as lr
    assert lr.__doc__ is not None
    assert "log10" in lr.__doc__ and "spectral_centroid" in lr.__doc__
```
- [ ] Run RED with temp-file+&&; expect ImportError.
- [ ] GREEN `src/rowii/anomaly/levelrecal.py` (docstring records the verified fact; real sketch):
```python
"""Level-only, shape-preserving channel recalibration (Package-8 D2, spec §3.D2 +
A1.1/A1.4/A1.9). VERIFIED FACT (rowii.signals.features): `*_log_rms`, `*_band_*`,
`*_octave_*` are log10-scaled (1e-12 floor); `*_spectral_centroid`, `*_rolloff95`,
`*_kurtosis` are RAW units. So an additive offset on the LEVEL columns is a shift in
the log10 domain, computed directly as a median-difference of the stored log10
features -- never imported from a partner 'dB' number (D2 computes its own offsets).
Fusion's stored features are per-run z-scores (`fuse` z-scores each stream), so it has
no meaningful level columns -- the empty-set guard is exactly the A1.9 fusion/embedding
refusal path; D2's variants are `audio` and `vibration` only (A1.1)."""
from __future__ import annotations

import numpy as np

_LEVEL_SUBSTRINGS = ("_log_rms", "_band_", "_octave_")
_SHAPE_SUBSTRINGS = ("_spectral_centroid", "_rolloff95", "_kurtosis")


def level_columns(feature_names: list[str]) -> list[int]:
    return [i for i, n in enumerate(feature_names) if any(s in n for s in _LEVEL_SUBSTRINGS)]


def column_medians(rows: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    cols = level_columns(feature_names)
    if not cols:
        raise ValueError("no level column (_log_rms/_band_/_octave_) in feature_names -- "
                         "level-recal is undefined for this variant (fusion z-scores / embeddings)")
    med = np.median(np.asarray(rows, dtype=np.float64)[:, cols], axis=0)
    return {feature_names[c]: float(m) for c, m in zip(cols, med, strict=True)}


def level_recal_offsets(run_median: dict[str, float], reference_median: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(run_median) & set(reference_median))
    if not keys:
        raise ValueError("level_recal_offsets: run/reference medians share no level column")
    return {k: run_median[k] - reference_median[k] for k in keys}


def apply_level_recal(features: np.ndarray, feature_names: list[str], offsets: dict[str, float]) -> np.ndarray:
    cols = level_columns(feature_names)
    if not cols:
        raise ValueError("no level column to recalibrate -- refusing (A1.9)")
    out = np.asarray(features, dtype=np.float64).copy()
    idx = {n: i for i, n in enumerate(feature_names)}
    for name, off in offsets.items():
        out[:, idx[name]] -= off
    return out
```
- [ ] Run GREEN gates; expect pass + `Success: no issues`.
- [ ] Commit `feat: level-only shape-preserving recalibration core (log10 offsets, empty-set guard) (P8 D2/A1.1/A1.9)`.

---

### Task 6: `run_step2 --level-recal` — cross-day-pooled surface (D2, A1.4/A1.11)

**Files:** Modify `scripts/run_step2.py` (new flag + wiring in `main`/`_run_cross_day_pooled`) · Test extend `tests/test_step2_pooled_cli.py`

**Interfaces:**
- Consumes: T5 `level_columns`/`column_medians`/`level_recal_offsets`/`apply_level_recal`.
- Produces: `--level-recal` (store_true, cross-day-pooled ONLY). Variants `audio`/`vibration` only (fusion excluded, A1.1 → `parser.error` for any other variant). Mutually exclusive with `--session-norm` (`parser.error`). Anchor (reference) = per-column median over the POOLED FIT side raw features (A1.4). The TEST run is aligned: `run_median = column_medians(first-N valid rows, names)` (N = `_DEFAULT_NORM_MINUTES` = 20, label-free), `offsets = level_recal_offsets(run_median, anchor)`, `test_features_scoring = apply_level_recal(prepared_test.features, names, offsets)`; the pooled FIT/CONFORMAL rows stay RAW (they define the anchor — mirrors monitor, where only the monitored run shifts). Output leaf suffix `-lrecal` via `_cross_day_pooled_out_dir(..., level_recal=True)`.
- Binding: level-recal and session-norm are never both active (A1.10 fit-path exclusivity); `--norm-minutes` is a session-norm flag and stays rejected here.

- [ ] RED (extend `tests/test_step2_pooled_cli.py` with the established 3-run monkeypatch fixture; audio/vibration variant so level columns exist):
```python
def test_level_recal_writes_lrecal_leaf_and_refuses_fusion(tmp_path, monkeypatch):
    import run_step2
    # ... existing style-2 monkeypatch of discover/load_config/prepare_run with
    #     audio-variant PreparedRuns whose feature_names carry _octave_/_log_rms ...
    code = run_step2.main(["--protocol", "cross-day-pooled", "--fit-runs", "fitA,fitB",
                           "--test-run", "testC", "--variant", "audio", "--scorer", "knn",
                           "--level-recal"])
    assert code == 0
    assert (tmp_path / "results" / "step2" / "cross-day-pooled" / "testC" / "audio-pooled-lrecal"
            / "far_table_frozen.csv").is_file()

def test_level_recal_refuses_fusion_and_session_norm(monkeypatch, capsys):
    import run_step2
    with pytest.raises(SystemExit):
        run_step2.main(["--protocol", "cross-day-pooled", "--fit-runs", "fitA,fitB",
                        "--test-run", "testC", "--variant", "fusion", "--level-recal"])
    with pytest.raises(SystemExit):
        run_step2.main(["--protocol", "cross-day-pooled", "--fit-runs", "fitA,fitB",
                        "--test-run", "testC", "--variant", "audio", "--level-recal", "--session-norm"])
```
- [ ] Run RED; expect failures (flag/wiring absent).
- [ ] GREEN: add `--level-recal` to `build_parser`; in `main`, cross-day-pooled branch: `parser.error` when `--level-recal` with `--session-norm`, or with a variant not in `("audio", "vibration")`, or with `--protocol != cross-day-pooled` (add to the existing non-pooled rejection loop). Thread `level_recal: bool` into `_run_cross_day_pooled`; compute `anchor = column_medians(pool_fit.features, feature_names)` and `test_features_scoring = apply_level_recal(prepared_test.features, feature_names, level_recal_offsets(column_medians(first_n_rows, names), anchor))` in the scoring-space block (parallel to the `stats_by_run` block, mutually exclusive with it); pass `level_recal=True` to `_cross_day_pooled_out_dir` (append `-lrecal`); add a `notes.md` "Level-only recalibration (D2)" section (attribution + verified-log10 line, no partner number). `--save-snapshot` + `--level-recal` stores the anchor medians (T7).
- [ ] Run GREEN gates; expect pass + `Success: no issues`.
- [ ] Commit `feat: run_step2 --level-recal (cross-day-pooled, audio/vibration, -lrecal leaf) (P8 D2/A1.4)`.

---

### Task 7: `monitor --level-recal` + snapshot v2 `level_recal_medians` (D2, A1.10/A1.11)

**Files:** Modify `src/rowii/runtime/snapshot.py` (optional field + save/load + `fit_snapshot_from_parts` kwarg), `scripts/run_step2.py` (pass anchor into the pooled snapshot when `--level-recal`), `scripts/monitor.py` (`--level-recal`) · Tests extend `tests/test_runtime_snapshot.py`, `tests/test_monitor_cli.py`

**Interfaces:**
- `MonitorSnapshot` gains `level_recal_medians: dict[str, float] | None = None` (name-keyed level-column anchor medians; OPTIONAL v2 member, NO version bump — mirrors `session_stats`, A1.10). `_meta_dict` adds a `"level_recal_medians"` entry exactly when present; `load_snapshot` reads it back; `save_snapshot` needs no new npz array (small dict lives in `meta` JSON). **A1.10 exclusivity:** `session_stats` and `level_recal_medians` are never both set — `fit_snapshot_from_parts` raises `ValueError` if both are passed.
- `fit_snapshot_from_parts(..., level_recal_medians: dict[str, float] | None = None)` — validates every key is one of `feature_names` (geometry guard, like the session-stats width check) and mutual-exclusion with `session_stats`.
- `run_step2 _run_cross_day_pooled` with `--save-snapshot` + `--level-recal`: passes `level_recal_medians = column_medians(pool_fit.features, feature_names)` (the anchor) into `fit_snapshot_from_parts`; references stay RAW.
- `monitor --level-recal`: refuses (exit 2) when `snapshot.level_recal_medians is None` (A1.10); mutually exclusive with `--session-norm` (exit 2); applies AFTER the snapshot-contract projection (A1.11) — compute `run_median = column_medians(first-N monitored valid rows, snapshot.feature_names)`, `offsets = level_recal_offsets(run_median, snapshot.level_recal_medians)`, `scoring_features = apply_level_recal(prepared.features, snapshot.feature_names, offsets)`, `session_stats=None` (references scored RAW — the query is aligned onto them). Zero level columns (embedding snapshot) → the T5 guard raises → caught → exit 2 (A1.9). Detector consumes RAW features (labels norm-invariant, the A3.5 boundary reused).

- [ ] RED extend `tests/test_runtime_snapshot.py`:
```python
def test_snapshot_round_trips_level_recal_medians(tmp_path):
    from rowii.runtime.snapshot import fit_snapshot_from_parts, load_snapshot, save_snapshot
    # ... build a detector + one-label references/cal/thresholds via the existing helper ...
    medians = {"f0": -40.0, "f1": -35.0}  # feature_names = ["f0","f1"]
    snap = fit_snapshot_from_parts(detector, refs, cal, thr, scorer="knn", alpha=0.05,
                                   min_ref=20, calibration_frac=0.5, seed=7, variant="audio",
                                   fit_run="pool:a,b", feature_names=["f0", "f1"], checkpoints={},
                                   level_recal_medians=medians)
    path = tmp_path / "s.npz"; save_snapshot(path, snap)
    got = load_snapshot(path)
    assert got.level_recal_medians == medians
    assert got.session_stats is None

def test_level_recal_and_session_stats_are_mutually_exclusive(...):
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit_snapshot_from_parts(..., session_stats=some_stats, level_recal_medians={"f0": 1.0})
```
- [ ] RED extend `tests/test_monitor_cli.py` (mirror the session-norm test patterns):
```python
def test_monitor_level_recal_refuses_snapshot_without_medians(tmp_path, monkeypatch):
    # a snapshot with level_recal_medians=None + --level-recal -> exit 2 (A1.10).
    ...
    assert monitor.main([..., "--level-recal"]) == 2

def test_monitor_level_recal_applies_after_projection(tmp_path, monkeypatch):
    # snapshot carries level_recal_medians; monitored run has a +level shift on the
    # level columns; --level-recal recentres it; alarms.parquet columns unchanged.
    ...
    assert monitor.main([..., "--level-recal", "--out", str(out)]) == 0
    assert list(pd.read_parquet(out / "alarms.parquet").columns) == _ALARM_COLUMNS

def test_monitor_level_recal_and_session_norm_mutually_exclusive(...):
    assert monitor.main([..., "--level-recal", "--session-norm"]) == 2
```
- [ ] Run RED; expect failures.
- [ ] GREEN: add the snapshot field/save/load/kwarg + exclusivity guard; add `--level-recal` to monitor's `build_parser`; in monitor `main` after projection + after the `--session-norm` block, add the level-recal block (guards → exit 2; compute `scoring_features` via T5; leave `session_stats=None`); reuse the existing `scoring_features` plumbing through `_frozen_verdicts`/`_recalibrate_verdicts`/`_rolling_verdicts`; add a `_notes_markdown` level-recal section; wire `run_step2 --save-snapshot --level-recal` to pass the anchor.
- [ ] Run GREEN gates; expect pass + `Success: no issues`.
- [ ] Commit `feat: monitor --level-recal + snapshot v2 level_recal_medians (post-projection, session-stats-exclusive) (P8 D2/A1.10/A1.11)`.

---

### Task 8: `scripts/analyze_days.py` — rotations-heatmap, feature-stability, era-step (D3, A1.2/A1.8/A1.11)

**Files:** Create `scripts/analyze_days.py` · Test `tests/test_analyze_days.py`

**Interfaces:**
- Consumes: `matplotlib` (Agg, set before pyplot — the `analyze_step1.py` pattern), `pandas`, `numpy`, `rowii.pipeline.prepare_run`, `rowii.scada.labels.load_scada_window_means`/`gt_labels`, `rowii.config`/`rowii.io.dataset` (for the feature/era subcommands). No new sweeps — reads existing `results/step2/cross-day-pooled/**/far_table_*.csv` + warm caches.
- Produces argparse SUBCOMMANDS (spec writes `--rotations-heatmap`; implemented as `argparse` subparsers named `rotations-heatmap` etc., the "independent subcommand" the spec's prose commits to — see resolved ambiguity below). Each writes a PNG + underlying CSV under `results/analysis-days/<subcommand>/`.
  - `rotations-heatmap`: day×day flag-rate matrix per threshold mode (`frozen`/`recalibrate`/`-lrecal` leaves once D2 exists). Pure helper `_flag_rate_matrix(far_tables: dict[tuple[str,str], float]) -> pd.DataFrame` where the value is the aggregate `pooled`-row `realized_far`; `imshow` heatmap.
  - `feature-stability`: per-feature cross-day shift with **`segment_ids`-block bootstrap** (A1.11 — never wall-clock), per GT mode, on GT-bearing days only (A1.2). Continuous dot-interval figure PRIMARY; binary slow(<3 dB)/drifting(>=3 dB) classification SECONDARY with the attribution label "same cutoff as Rodrigues & Zhang (2026), adopted for comparability" (A1.8). Pure helpers `_block_bootstrap_ci(values, segment_ids, n_boot, seed) -> (lo, hi)` and `_classify_shift(shift_abs, cutoff=3.0) -> str`.
  - `era-step`: per-day per-stream median level (mic vs vib streams, `log_rms` + key bands) across 25.06→08.07 in matched GT modes, with MeasName era boundaries marked. **A1.2 gates:** GT-match on 25.06/29.06/01.07 (+080726 ONLY behind `--include-080726`, gated on the D4.3 timebase probe); 27.06 shown as an UN-MATCHED per-stream-median point flagged "no GT — era-A anchor by MeasName only". Pure helper `_era_step_row(levels_by_stream, gt_mode_mask) -> dict`.
- Binding: A1.8 test rule — no partner number as an expected value anywhere; the `3.0` cutoff is a named constant, tested only for the classification BOUNDARY behavior (a synthetic 2.9 → slow, 3.1 → drifting), never against a partner figure.

- [ ] RED `tests/test_analyze_days.py` (synthetic artifact trees + pure-helper unit tests):
```python
"""Tests for scripts/analyze_days.py (Package-8 D3): per-subcommand artifact-shape
tests on synthetic inputs + pure-helper math (flag-rate matrix, segment-block
bootstrap, 3dB classification boundary, era-step column math). A1.8: no partner
number appears as an expected value."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_days as ad  # noqa: E402


def test_flag_rate_matrix_reads_pooled_aggregate_row():
    far = {("010726-pu", "290626-tu"): 0.03, ("290626-tu", "010726-pu"): 0.51}
    m = ad._flag_rate_matrix(far)
    assert m.loc["010726-pu", "290626-tu"] == 0.03
    assert m.loc["290626-tu", "010726-pu"] == 0.51


def test_classify_shift_boundary_is_the_named_cutoff():
    assert ad._classify_shift(2.9, cutoff=3.0) == "slow"
    assert ad._classify_shift(3.1, cutoff=3.0) == "drifting"
    assert ad._classify_shift(3.0, cutoff=3.0) == "drifting"  # >= cutoff


def test_block_bootstrap_uses_segment_ids_not_wall_clock():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, 120)
    seg = np.repeat(np.arange(12), 10)  # 12 recording segments
    lo, hi = ad._block_bootstrap_ci(values, seg, n_boot=200, seed=1)
    assert lo < np.median(values) < hi
    # a degenerate single-segment array still returns a finite interval.
    lo1, hi1 = ad._block_bootstrap_ci(values, np.zeros(120, dtype=np.int64), n_boot=50, seed=1)
    assert np.isfinite(lo1) and np.isfinite(hi1)


def test_rotations_heatmap_subcommand_writes_png_and_csv(tmp_path, monkeypatch):
    # synthetic cross-day-pooled tree: two far tables with a 'pooled' aggregate row.
    root = tmp_path / "results" / "step2" / "cross-day-pooled"
    for test, fit, far in (("290626-tu", "010726-pu", 0.03), ("010726-pu", "290626-tu", 0.51)):
        d = root / test / "audio-pooled"; d.mkdir(parents=True)
        pd.DataFrame([{"label": "0", "realized_far": 0.0}, {"label": "pooled", "realized_far": far}]
                     ).to_csv(d / "far_table_frozen.csv", index=False)
        (d / "notes.md").write_text(f"- fit pool: {fit} (pool order = `--fit-runs` order)\n")
    out = tmp_path / "results" / "analysis-days"
    code = ad.main(["rotations-heatmap", "--root", str(root), "--out", str(out),
                    "--variant", "audio", "--mode", "frozen"])
    assert code == 0
    assert (out / "rotations-heatmap" / "audio-frozen.png").is_file()
    assert (out / "rotations-heatmap" / "audio-frozen.csv").is_file()
```
- [ ] Run RED with temp-file+&&; expect ImportError.
- [ ] GREEN `scripts/analyze_days.py` (Agg backend; docstring carries per-subcommand attribution lines). Pure helpers real:
```python
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

def _classify_shift(shift_abs: float, cutoff: float = 3.0) -> str:
    return "drifting" if shift_abs >= cutoff else "slow"  # cutoff named, adopted-for-comparability (A1.8)

def _block_bootstrap_ci(values, segment_ids, n_boot, seed):
    rng = np.random.default_rng(seed)
    seg_ids = np.unique(segment_ids)
    groups = [values[segment_ids == s] for s in seg_ids]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        boots[b] = float(np.median(np.concatenate([groups[i] for i in pick])))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def _flag_rate_matrix(far):  # {(fit_pool_key, test_run): pooled_realized_far}
    rows = sorted({k[0] for k in far}); cols = sorted({k[1] for k in far})
    m = pd.DataFrame(np.nan, index=rows, columns=cols)
    for (fit, test), v in far.items():
        m.loc[fit, test] = v
    return m
```
  - `rotations-heatmap` reads each leaf's `far_table_<mode>.csv`, extracts the `label == "pooled"` `realized_far`, parses the fit pool from `notes.md` ("fit pool:" line) and the test run from the parent dir → builds `_flag_rate_matrix`, `imshow` + annotate, save PNG + CSV.
  - `feature-stability` recomputes per-feature per-GT-mode day medians from warm caches (`prepare_run` + `load_scada_window_means`/`gt_labels`, masked GT-bearing days A1.2), `_block_bootstrap_ci` per feature (block = `segment_ids`), dot-interval `errorbar` (primary) + a `_classify_shift` column (secondary) → CSV + PNG.
  - `era-step` computes per-day per-stream median `log_rms`/band level in matched GT modes (25/29/01 + 080726 behind `--include-080726`), plots per-stream lines with era-boundary vlines, 27.06 as a distinct unmatched marker.
- [ ] Run GREEN gates; expect pass + `Success: no issues`.
- [ ] Commit `feat: analyze_days rotations-heatmap/feature-stability/era-step (segment-block bootstrap, GT gates) (P8 D3/A1.2/A1.11)`.

---

### Task 9: `analyze_days` mode-signatures, tonal-table, pillar3-figure + digest (D3, A1.8)

**Files:** Modify `scripts/analyze_days.py` (three subcommands + `digest`) · Test extend `tests/test_analyze_days.py`

**Interfaces:**
- `mode-signatures`: per-GT-mode band/octave profile (median + IQR) per day — the "modes are separable" picture on our features. Pure helper `_mode_profile(features, gt_states, level_cols) -> DataFrame` (median + IQR per mode).
- `tonal-table`: per mode×day the shaft/blade-pass/guide-vane band energies relative to the neighboring octave floor (an SNR-like contrast DEFINED FROM OUR FEATURES, NOT the partner's exact metric — A1.8 docstring note). Pure helper `_tonal_contrast(band_energy, octave_floor) -> float`.
- `pillar3-figure`: TPR-vs-alpha grouped bars per representation for both 080726 sessions, read from the P7 `results/pillar3/**` event_eval artifacts. Pure helper `_tpr_by_alpha(event_table) -> DataFrame`.
- `digest`: writes `results/analysis-days/README.md` linking every figure with a 2-3 sentence plain-language reading (English only — `--lang de` NOT built), **each partner-inspired analysis carrying its attribution line (A1.8)**; also documents the fusion per-run z-score FINDING (A1.1: an implicit session normalization that plausibly explains fusion's cross-day FAR advantage).

- [ ] RED extend `tests/test_analyze_days.py`:
```python
def test_tonal_contrast_is_band_minus_floor():
    assert ad._tonal_contrast(band_energy=-30.0, octave_floor=-45.0) == 15.0  # our own definition

def test_mode_signatures_subcommand_writes_artifacts(tmp_path, monkeypatch):
    # monkeypatch _run_features_and_gt to feed synthetic (features, gt_states, names)
    ...
    assert ad.main(["mode-signatures", "--runs", "290626-tu", "--variant", "audio",
                    "--out", str(out)]) == 0
    assert (out / "mode-signatures" / "290626-tu.png").is_file()

def test_digest_writes_readme_with_attribution_lines(tmp_path):
    out = tmp_path / "results" / "analysis-days"
    (out / "rotations-heatmap").mkdir(parents=True)
    (out / "rotations-heatmap" / "audio-frozen.png").write_bytes(b"x")
    assert ad.main(["digest", "--out", str(out)]) == 0
    readme = (out / "README.md").read_text()
    assert "Rodrigues & Zhang (2026)" in readme       # attribution present
    assert "z-score" in readme and "fusion" in readme  # A1.1 finding documented
```
- [ ] Run RED; expect failures.
- [ ] GREEN: add the three subcommands + `digest`; `_tonal_contrast(band, floor) = band - floor`; `mode-signatures`/`tonal-table` read warm caches + GT; `pillar3-figure` reads `results/pillar3/**` event tables → grouped bars; `digest` scans `results/analysis-days/*/*.png` and writes README with the plain-language readings + attribution + the fusion-z-score finding.
- [ ] Run GREEN gates; expect pass + `Success: no issues`.
- [ ] Commit `feat: analyze_days mode-signatures/tonal-table/pillar3-figure + digest (attribution, fusion-zscore finding) (P8 D3/A1.8)`.

---

### Task 10: Execution + synthesis (orchestrator-executed, NO code)

> Marked orchestrator-executed: no source changes. All commands run from `repos/rowii-monitor/` with the project `.venv`. Every synthesis number MUST be artifact-verified; negative results reported plainly.

- [ ] **Warm caches (BLOCKING — audio/vibration on every rotation/pool/080726 day).** Plain `vibration` caches do NOT exist yet (only `vibration-tfc`); `audio`/`audio-beats`/`fusion` exist for most run names but verify each. Rotation test days: `290626-tu`, `290626-pu`, `010726-tu_ph_tu`, `010726-pu`, `250526-tu`, `250526-pu-morning`. Fit-pool members: `010726-pu`, `010726-tu1-morning`, `010726-tu2`, `010726-tu_ph_tu`, `290626-tu`, `290626-pu`. Pillar-3: `080726-pu_strikes`, `080726-st_strikes`. Warm explicitly:
  - `.venv/bin/python scripts/warm_cache.py --variants vibration --runs 010726-pu 010726-tu1-morning 010726-tu2 010726-tu_ph_tu 290626-tu 290626-pu 250526-tu 250526-pu-morning 080726-pu_strikes 080726-st_strikes`
  - Verify `audio`/`audio-beats`/`fusion` caches present (warm any missing the same way); confirm each `results/cache/<run>--{audio,vibration,audio-beats,fusion}.npz` exists before proceeding.
- [ ] **D4 probes as GATES (run FIRST — A1.2/A1.6):**
  - `.venv/bin/python scripts/verify_data_facts.py scada-timebase --reference-utc 2026-07-08T13:05:28+00:00` → record the Betriebsdaten changeover timestamp + delta; **this result GATES every 080726 GT-matched analysis** (D1 pillar-3 spot, era-step 080726 point). If the timebase disagrees, 080726 GT-matching stays behind the gate (omit `--include-080726` / the pillar-3 spot's GT-conditioned rows).
  - `.venv/bin/python scripts/verify_data_facts.py vib-ch0-liveness --runs <all rotation/pool/080726 days>` → record which days carry a live `RAWTurbineVib__3` ch0, pinned to the era timeline.
  - `.venv/bin/python scripts/verify_data_facts.py gen-mic-profile --run 080726-st_strikes --events docs/groundtruth/080726_events_st.csv` → record the channel-anonymous outlier channel (no azimuth claim).
- [ ] **D1 bank rotations** (6 rotations × 3 families {gaussian,knn,gmm} × 3 representations {fusion, audio-beats, vibration}) — for each `(fit-pool, test)` rotation:
  - `.venv/bin/python scripts/run_modebank.py --fit-runs <pool> --test-run <test> --variant <rep> --family <fam> --alpha 0.05 [--smooth]` → ARI (primary, masked), bank accuracy (supervised-tagged), confusion, `no_mode_fits` rate, and the recomputed P7-pooled ARI (unsupervised-tagged) under the SAME mask.
- [ ] **D1 chain probe:** `.venv/bin/python scripts/run_modebank_chain.py --fit-runs <B1 pool> --test-run 290626-tu --variant fusion --family <best> --alpha 0.05` → `far_table.csv` comparable to the P7 detected-state chain (split-parity guaranteed).
- [ ] **D1 pillar-3 spot** (post-timebase-gate): bank labels on `080726-pu_strikes` (fusion + vibration), recalibrate + event-free calibration → chain → `.venv/bin/python scripts/eval_events.py --events docs/groundtruth/080726_events_pu.csv ...` at alpha 0.01 → TPR/FAR vs the P7 clustered-state chain. First report the bank's mode-ID accuracy on 080726 vs GT as a prerequisite readout (A1.7).
- [ ] **D2 level-recal — the six P7 rotations, FROZEN mode, variants audio + vibration** (A1.4: the two cross-era rotations are the headline cells): for each rotation, run the comparison quad on identical cells —
  - raw frozen: `run_step2 --protocol cross-day-pooled --fit-runs <pool> --test-run <test> --variant <audio|vibration> --scorer knn --thresholds ... --alpha 0.05` (existing far_table_frozen.csv)
  - level-recal frozen: add `--level-recal` (`-lrecal` leaf)
  - session-norm frozen: `--session-norm` (existing `-snorm` leaf, P7)
  - recalibrate (control): existing far_table_recalibrate.csv
  - Table: `raw-frozen | level-recal-frozen | session-norm-frozen | recalibrate` on identical cells.
- [ ] **D2 pillar-3 side arm** (A1.4): FROZEN mode WITHOUT `--exclude-calibration-events` (no-op in frozen), fusion excluded (A1.1) → run `audio`+`vibration` on both 080726 sessions, alpha 0.01/0.05, event-free calibration; recalibrate+level-recal run ONCE as a stated control row (expected-redundant, session-norm precedent).
- [ ] **D3 analyze_days — full figure set** from existing + new artifacts:
  - `.venv/bin/python scripts/analyze_days.py rotations-heatmap --variant <audio|vibration|fusion> --mode <frozen|recalibrate>` (+ `-lrecal` leaves once D2 rotations exist)
  - `.venv/bin/python scripts/analyze_days.py feature-stability --runs <GT-bearing days>` (segment-block bootstrap; continuous dot-interval primary)
  - `.venv/bin/python scripts/analyze_days.py era-step [--include-080726 (only if timebase gate passed)]`
  - `.venv/bin/python scripts/analyze_days.py mode-signatures --runs <days>`; `tonal-table --runs <days>`; `pillar3-figure`
  - `.venv/bin/python scripts/analyze_days.py digest` → `results/analysis-days/README.md`
- [ ] **Synthesis:** README package-8 section (all numbers artifact-verified; the "best system / best family" recommendation stated as the OUTCOME of the comparisons, never assumed; fusion remains the universal baseline row where representations are compared); master-thesis research note (figures inline, in `master-thesis/research/notes/`); memory update. Every partner-inspired analysis carries its attribution; the fusion-z-score finding (A1.1) is documented.
- [ ] **Final whole-branch review (opus)** — named focuses: bank internals correctness (standardization/argmin/reject, A1.5), the `{unknown,transition}` ARI mask on BOTH arms (A1.3/A1.5), split-parity of the chain probe, D2 log10-offset direction + fusion-exclusion (A1.1), snapshot v2 `level_recal_medians` round-trip + session-stats exclusivity (A1.10), the D4 timebase GATE actually gating the 080726 GT analyses (A1.2), and firewall (no partner number as an expected value, A1.8). → fix loop → **PR #11** → merge.

---

## Self-review (at write time)

**Spec coverage.** D1 → T2 (bank core) + T3 (rotations CLI: ARI/accuracy/no_mode_fits + P7 comparison) + T4 (chain probe) + T10 (pillar-3 spot). D2 → T5 (core) + T6 (run_step2) + T7 (monitor + snapshot) + T10 (rotations + side arm). D3 → T8 (rotations-heatmap/feature-stability/era-step) + T9 (mode-signatures/tonal-table/pillar3-figure/digest) + T10 (runs). D4 → T1 (three probes) + T10 (gates). Amendments: A1.1 → T5 docstring + T6 fusion-exclusion; A1.2 → T8 era-step/feature-stability GT gates + T10 timebase gate; A1.3 → T3 ARI-primary + `_smooth_ids`=duration-filter-only; A1.4 → T6/T7 offset sources + T10 pillar-3 frozen mode; A1.5 → T2 internals; A1.6 → T1 channel-anonymous + variance-liveness; A1.7 → T10 bank mode-ID prerequisite readout; A1.8 → firewall test rule + named `3.0` cutoff + digest attribution (T8/T9); A1.9 → T5 empty-set guard + T6 fusion refusal; A1.10 → T7 optional v2 member + session-stats exclusivity; A1.11 → T7 post-projection + T8 `segment_ids` bootstrap.

**Type/name consistency.** `ModeBank`/`ModeAssignment` (T2) consumed by T3/T4/T10; `ModeBank.assign` returns `(labels, scores, modes, no_mode_fits)` used identically in both CLIs. `level_columns`/`column_medians`/`level_recal_offsets`/`apply_level_recal` (T5) consumed by T6/T7 with the same signatures. `level_recal_medians` (T7) is the anchor produced by `column_medians(pool_fit.features, ...)` in run_step2 and consumed by monitor after projection. `_masked_ari` masks `{unknown,transition}` in T3 and reused-in-spirit by T8/T9's GT masking. `_FAR_TABLE_COLUMNS` schema reused (duplicated) in T4/T8. No placeholders, no TBD — every module/CLI has a real code sketch and complete RED tests.

**Resolved ambiguities (flagged for the orchestrator).**
1. Spec spells the D3 analyses as `--rotations-heatmap` flags but calls them "independent subcommand" — implemented as argparse SUBCOMMANDS (positional `rotations-heatmap` etc.), the cleaner reading of the spec's own "subcommand" prose.
2. No per-window P7 pooled-detector label artifact is persisted (cross-day-pooled writes only far/coverage tables) — the D1 unsupervised comparison RECOMPUTES P7 labels via `FittedDetector.fit_pooled(..., k=4).apply(...)` (deterministic; `--p7-k` default 4 from `k_selection.json`), rather than loading a nonexistent path.
3. `rowii.eval.metrics.evaluate` masks only `"unknown"`; A1.5 needs `{unknown,transition}` — D1 computes ARI via `adjusted_rand_score` on its OWN mask, both arms identical (delta vs P7 recorded).
4. Offset direction pinned unambiguously: `offset = run_median − reference/anchor`, `apply` SUBTRACTS → recentres the run-being-aligned ONTO the anchor (monitor: monitored run → stored fit anchor; run_step2: test run → pooled-fit anchor; pool/reference rows stay raw). Matches D2's stated mechanism and A1.4's anchor sources.
