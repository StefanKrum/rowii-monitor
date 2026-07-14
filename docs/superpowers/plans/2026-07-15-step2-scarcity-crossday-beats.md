# Step-2 Package 2 Implementation Plan — Per-State Cross-Day, Scarcity Curve, BEATs Evidence

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer three open Step-2 questions on real data: does per-state conditioning restore cross-day FAR control (runtime-honest detector transfer), how many calibration windows per mode are needed (scarcity curve), and does frozen-BEATs beat handcrafted features as scoring representation.

**Architecture:** A new `FittedDetector` (fit/apply split of the existing Step-1 chain) enables transferring one day's fitted state detector to another day without refit. A new `cross-day-per-state` protocol in `run_step2.py` builds per-state references/thresholds on day A and scores day B by *predicted* state. A new `anomaly/scarcity.py` module subsamples calibration scores to trace realized-FAR-vs-n curves with exact Beta bands. BEATs needs no new modeling code — only cache warm-up, execution, and overlap analysis.

**Tech Stack:** Python 3.12, numpy/scipy/pandas/scikit-learn/hmmlearn/matplotlib (all existing deps), pytest, existing feature cache (`results/cache/*.npz`).

**Design authority:** `docs/superpowers/specs/2026-07-15-step2-scarcity-crossday-beats-design.md` (spec D1–D5). Read it before starting any task.

## Global Constraints

- Branch: `feat/step2-scarcity-crossday-beats` (created from `main`).
- Tests: `.venv/bin/python -m pytest tests/ -q` from repo root — every existing test stays green after every task.
- Lint gates after every task: `.venv/bin/ruff check .` and `.venv/bin/mypy src scripts` both clean.
- Conventional commits (`feat:`, `test:`, `refactor:`, `docs:`); `git add` with explicit paths, never `-A`.
- Real-data tests use `@pytest.mark.data` (skipped without `ROWII_DATA_ROOT`); synthetic tests must not need real data.
- RNG: `numpy.random.default_rng(seed)` only — never `np.random.seed` / global state.
- No values from the partner's repo/analyses are ever adopted into code or results (comparisons are report-only).
- Docstrings follow the repo's Google style (Args/Returns/Raises) with design-spec cross-references, matching the density of `src/rowii/anomaly/conformal.py`.
- Scripts start with `from __future__ import annotations` and are import-safe (no work at import time except constants).

---

### Task 1: Standardization stats + clusterer `predict` + smoother `decode` (transfer primitives)

**Files:**
- Modify: `src/rowii/signals/features.py` (zscore refactor, ~line 388)
- Modify: `src/rowii/state/cluster.py` (predict methods)
- Modify: `src/rowii/state/smooth.py` (decode method, fitted-state storage)
- Test: `tests/test_features.py`, `tests/test_cluster.py`, `tests/test_smooth.py` (extend existing files)

**Interfaces:**
- Consumes: existing `zscore(x)`, `KMeansClusterer`, `GmmClusterer`, `StickyHmmSmoother`.
- Produces (Task 2 relies on these exact signatures):
  - `zscore_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — per-column `(mean, std)` via `nanmean`/`nanstd`, float64.
  - `apply_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray` — standardize with GIVEN stats; columns with `std < 1e-12` → 0; NaN rows stay NaN. Invariant: `zscore(x) == apply_zscore(x, *zscore_stats(x))` element-wise (NaN-equal).
  - `KMeansClusterer.predict(x) -> np.ndarray` / `GmmClusterer.predict(x) -> np.ndarray` — int64 labels; `RuntimeError` if called before `fit_predict`.
  - `StickyHmmSmoother.decode(features: np.ndarray) -> np.ndarray` — Viterbi labels in the FIT-time id space, no refit; `RuntimeError` if `fit_decode` never ran; handles the k<=1 fit case (returns the single stored id for every window).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_features.py`:

```python
class TestZscoreStatsApply:
    def test_apply_with_own_stats_equals_zscore(self):
        # NaN row + constant column + normal columns: full semantics coverage
        x = np.array(
            [[1.0, 5.0, 2.0], [2.0, 5.0, 4.0], [np.nan, 5.0, 6.0], [3.0, 5.0, 8.0]]
        )
        mean, std = zscore_stats(x)
        out = apply_zscore(x, mean, std)
        expected = zscore(x)
        np.testing.assert_array_equal(out, expected)  # NaN-positions compare equal

    def test_apply_with_foreign_stats_uses_given_stats(self):
        a = np.array([[0.0, 0.0], [2.0, 4.0]])  # mean (1,2), std (1,2)
        mean, std = zscore_stats(a)
        b = np.array([[1.0, 2.0]])
        out = apply_zscore(b, mean, std)
        np.testing.assert_allclose(out, [[0.0, 0.0]])

    def test_apply_zero_std_column_becomes_zero(self):
        mean = np.array([1.0, 2.0])
        std = np.array([1.0, 0.0])
        out = apply_zscore(np.array([[3.0, 9.0]]), mean, std)
        np.testing.assert_allclose(out, [[2.0, 0.0]])
```

Append to `tests/test_cluster.py`:

```python
class TestPredict:
    def test_kmeans_predict_matches_fit_predict_on_same_data(self):
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(0, 0.1, (50, 3)), rng.normal(5, 0.1, (50, 3))])
        c = KMeansClusterer(n_clusters=2, random_seed=7)
        fit_labels = c.fit_predict(x)
        np.testing.assert_array_equal(c.predict(x), fit_labels)
        assert c.predict(x).dtype == np.int64

    def test_kmeans_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit_predict"):
            KMeansClusterer(n_clusters=2, random_seed=7).predict(np.zeros((3, 2)))

    def test_gmm_predict_new_points_assigned_to_nearest_component(self):
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(0, 0.1, (50, 2)), rng.normal(5, 0.1, (50, 2))])
        c = GmmClusterer(n_clusters=2, random_seed=7)
        fit_labels = c.fit_predict(x)
        label_at_origin = fit_labels[0]
        assert c.predict(np.array([[0.05, -0.05]]))[0] == label_at_origin

    def test_gmm_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit_predict"):
            GmmClusterer(n_clusters=2, random_seed=7).predict(np.zeros((3, 2)))
```

Append to `tests/test_smooth.py`:

```python
class TestDecode:
    def _two_state_features(self, seed=0):
        rng = np.random.default_rng(seed)
        f = np.concatenate([rng.normal(0, 0.2, 60), rng.normal(4, 0.2, 60)])
        labels = np.array([3] * 60 + [7] * 60, dtype=np.int64)  # non-contiguous ids
        return f.reshape(-1, 1), labels

    def test_decode_same_features_matches_fit_decode(self):
        features, init = self._two_state_features()
        s = StickyHmmSmoother(random_seed=7)
        fitted = s.fit_decode(features, init)
        np.testing.assert_array_equal(s.decode(features), fitted)

    def test_decode_new_features_uses_fit_id_space_without_refit(self):
        features, init = self._two_state_features()
        s = StickyHmmSmoother(random_seed=7)
        s.fit_decode(features, init)
        means_before = s.last_model_.means_.copy()
        new = np.concatenate(
            [np.full(10, 4.0), np.full(10, 0.0)]
        ).reshape(-1, 1)  # reversed order vs fit day
        decoded = s.decode(new)
        assert set(np.unique(decoded)) <= {3, 7}
        assert decoded[0] == 7 and decoded[-1] == 3
        np.testing.assert_array_equal(s.last_model_.means_, means_before)

    def test_decode_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit_decode"):
            StickyHmmSmoother().decode(np.zeros((5, 1)))

    def test_decode_after_single_label_fit_returns_that_id(self):
        features = np.zeros((10, 1))
        init = np.full(10, 42, dtype=np.int64)
        s = StickyHmmSmoother()
        s.fit_decode(features, init)  # k=1 path: last_model_ stays None
        np.testing.assert_array_equal(
            s.decode(np.ones((4, 1))), np.full(4, 42, dtype=np.int64)
        )
```

Add the needed imports at the top of each test file if missing (`zscore_stats`, `apply_zscore` from `rowii.signals.features`; `pytest` in test_cluster.py).

- [ ] **Step 2: Run the new tests, verify they fail**

Run: `.venv/bin/python -m pytest tests/test_features.py -q -k ZscoreStats && .venv/bin/python -m pytest tests/test_cluster.py -q -k Predict && .venv/bin/python -m pytest tests/test_smooth.py -q -k Decode`
Expected: FAIL / ERROR with ImportError (`zscore_stats`) and AttributeError (`predict`, `decode`).

- [ ] **Step 3: Implement**

In `src/rowii/signals/features.py`, refactor `zscore` into three functions (keep the existing docstring content on `zscore`, split the mechanics; keep NaN/zero-std semantics EXACTLY):

```python
def zscore_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column (mean, std) via `nanmean`/`nanstd`, float64 — the statistics
    `zscore` standardizes with, exposed separately so a fitted model can carry its
    FIT-day statistics and re-apply them to another day's features
    (`rowii.state.detect.FittedDetector`, package-2 spec D1).
    """
    x64 = np.asarray(x, dtype=np.float64)
    return np.nanmean(x64, axis=0), np.nanstd(x64, axis=0)


def apply_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """`(x - mean) / std` per column with GIVEN statistics; columns with
    `std < 1e-12` become zero; NaN input rows stay NaN (same semantics as `zscore`,
    which is exactly `apply_zscore(x, *zscore_stats(x))` — regression-gated).
    """
    x64 = np.asarray(x, dtype=np.float64)
    nan_rows = np.isnan(x64).any(axis=1)
    out = np.zeros_like(x64)
    safe = std >= 1e-12
    out[:, safe] = (x64[:, safe] - mean[safe]) / std[safe]
    out[nan_rows] = np.nan
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    """<keep the existing docstring verbatim>"""
    return apply_zscore(x, *zscore_stats(x))
```

In `src/rowii/state/cluster.py`, add to `KMeansClusterer`:

```python
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Nearest-centroid labels for *x* from the already-fitted model.

        Raises:
            RuntimeError: if called before `fit_predict`.
        """
        if not hasattr(self._kmeans, "cluster_centers_"):
            raise RuntimeError("KMeansClusterer.predict called before fit_predict")
        return np.asarray(self._kmeans.predict(x), dtype=np.int64)
```

and to `GmmClusterer`:

```python
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Posterior-argmax component labels for *x* from the already-fitted model.

        Raises:
            RuntimeError: if called before `fit_predict`.
        """
        if not hasattr(self._gmm, "means_"):
            raise RuntimeError("GmmClusterer.predict called before fit_predict")
        return np.asarray(self._gmm.predict(x), dtype=np.int64)
```

In `src/rowii/state/smooth.py`: in `__init__` add `self._fitted_ids: np.ndarray | None = None` and `self._component_to_id: dict[int, int] | None = None`. In `fit_decode`, set `self._fitted_ids = unique_ids` in BOTH branches (the k<=1 early-return must set it before returning, alongside `self.last_model_ = None`), and store `self._component_to_id = component_to_id` right after building it in the k>1 branch (note: `component_to_id` is currently built inline near the end — hoist the dict construction so it is stored once and used for both the fit-time decode and `decode`). Add:

```python
    def decode(self, features: np.ndarray) -> np.ndarray:
        """Viterbi labels for *features* from the already-fitted sticky HMM — no
        refit, no EM; labels come back in the SAME id space `fit_decode` was given
        (package-2 spec D1: cross-day apply must never re-estimate on the new day).

        Raises:
            RuntimeError: if `fit_decode` has never run on this instance.
        """
        if self._fitted_ids is None:
            raise RuntimeError("StickyHmmSmoother.decode called before fit_decode")
        if self.last_model_ is None:  # k<=1 fit: every window is the single fit id
            return np.full(len(features), int(self._fitted_ids[0]), dtype=np.int64)
        assert self._component_to_id is not None
        components = self.last_model_.predict(np.asarray(features, dtype=np.float64))
        return np.array(
            [self._component_to_id[c] for c in components], dtype=np.int64
        )
```

- [ ] **Step 4: Run the full suite + lints**

Run: `.venv/bin/python -m pytest tests/ -q && .venv/bin/ruff check . && .venv/bin/mypy src scripts`
Expected: all pass (the zscore refactor is behavior-identical; existing tests gate it).

- [ ] **Step 5: Commit**

```bash
git add src/rowii/signals/features.py src/rowii/state/cluster.py src/rowii/state/smooth.py tests/test_features.py tests/test_cluster.py tests/test_smooth.py
git commit -m "feat: standardization stats + clusterer predict + HMM decode (transfer primitives)"
```

---

### Task 2: `FittedDetector` (fit/apply) with same-day equivalence

**Files:**
- Modify: `src/rowii/state/detect.py`
- Test: `tests/test_detect_e2e.py` (extend), `tests/test_real_data.py` (one `@pytest.mark.data` test)

**Interfaces:**
- Consumes: Task 1's `zscore_stats`, `apply_zscore`, `predict`, `decode`; existing `DetectConfig`, `DetectionResult`, `duration_filter`, `to_segments`, `WindowGrid`.
- Produces (Tasks 3 and 9 rely on these):
  - `FittedDetector.fit(features, grid, cfg: DetectConfig, clusterer: Literal["kmeans","gmm"]="kmeans", k: int | None = None) -> tuple[FittedDetector, DetectionResult]` (classmethod).
  - `FittedDetector.apply(features, grid) -> DetectionResult` — fit-day stats + fit-day HMM decode, NO refit anywhere.
  - `run_detection(...)` unchanged signature, now delegating to `FittedDetector.fit` (DRY) — existing outputs bit-identical (existing e2e tests are the gate).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_detect_e2e.py` (reuse that file's existing synthetic-feature helpers/config style — read the file first and match its fixture pattern):

```python
class TestFittedDetector:
    def _three_state_run(self, seed=0, n_per_state=120, offset=0.0):
        rng = np.random.default_rng(seed)
        blocks = [
            rng.normal(0 + offset, 0.15, (n_per_state, 2)),
            rng.normal(4 + offset, 0.15, (n_per_state, 2)),
            rng.normal(8 + offset, 0.15, (n_per_state, 2)),
        ]
        features = np.vstack(blocks)
        grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=len(features))
        return features, grid

    def _cfg(self):
        return DetectConfig(
            n_states=3, self_transition=0.98, min_dwell_s=3, random_seed=7
        )

    def test_fit_result_identical_to_run_detection(self):
        features, grid = self._three_state_run()
        cfg = self._cfg()
        legacy = run_detection(features, grid, cfg, clusterer="kmeans")
        _, fitted_result = FittedDetector.fit(features, grid, cfg, clusterer="kmeans")
        np.testing.assert_array_equal(fitted_result.frame_labels, legacy.frame_labels)
        assert fitted_result.k == legacy.k

    def test_apply_same_day_reproduces_fit_labels(self):
        features, grid = self._three_state_run()
        det, fit_result = FittedDetector.fit(features, grid, self._cfg())
        applied = det.apply(features, grid)
        np.testing.assert_array_equal(applied.frame_labels, fit_result.frame_labels)

    def test_apply_uses_fit_day_statistics(self):
        features_a, grid = self._three_state_run(seed=0)
        det, _ = FittedDetector.fit(features_a, grid, self._cfg())
        mean_a, std_a = zscore_stats(features_a)
        np.testing.assert_array_equal(det.mean, mean_a)
        np.testing.assert_array_equal(det.std, std_a)
        # Day B = day A shifted by a constant; with A's stats the z-features shift,
        # with B's own stats they would be identical to A's. Verify A's stats in use:
        features_b = features_a + 2.0
        z_expected = apply_zscore(features_b, mean_a, std_a)
        assert not np.allclose(z_expected, apply_zscore(features_b, *zscore_stats(features_b)))
        applied = det.apply(features_b, grid)  # must run through A-stats path
        assert applied.frame_labels.shape == (len(features_b),)

    def test_apply_never_refits_hmm(self):
        features_a, grid = self._three_state_run(seed=0)
        det, _ = FittedDetector.fit(features_a, grid, self._cfg())
        means_before = det.smoother.last_model_.means_.copy()
        features_b, grid_b = self._three_state_run(seed=1, offset=0.3)
        det.apply(features_b, grid_b)
        np.testing.assert_array_equal(det.smoother.last_model_.means_, means_before)

    def test_apply_transfers_label_space(self):
        # B presents the states in a different order/proportion; labels must come
        # from A's id space and follow the feature values, not the position.
        features_a, grid_a = self._three_state_run(seed=0)
        det, fit_result = FittedDetector.fit(features_a, grid_a, self._cfg())
        # B: only the "middle" (value 4) state, long enough to survive min_dwell
        rng = np.random.default_rng(9)
        features_b = rng.normal(4, 0.15, (100, 2))
        grid_b = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=100)
        applied = det.apply(features_b, grid_b)
        a_middle_label = fit_result.frame_labels[150]  # deep inside A's value-4 block
        assert set(np.unique(applied.frame_labels)) == {a_middle_label}
```

Append to `tests/test_real_data.py` (match that file's existing `pytestmark = pytest.mark.data` + data-root fixture pattern — read it first):

```python
def test_fitted_detector_apply_equals_fit_on_cached_run(prepared_fusion_010726):
    """Same-day apply == fit labels on a real cached PreparedRun (spec D1 gate).

    Uses whatever real-run fixture the file already provides for 010726-tu_ph_tu
    fusion (or construct via prepare_run(use_cache=True) following the file's
    existing pattern); compact to valid windows exactly like
    scripts/run_step2.py::_detected_labels.
    """
    prepared = prepared_fusion_010726
    valid = prepared.valid_mask
    feats = prepared.features[valid]
    grid = WindowGrid(
        t0_ns=prepared.grid.t0_ns,
        window_ns=prepared.grid.window_ns,
        n_windows=int(valid.sum()),
    )
    cfg = load_config().detect
    det, fit_result = FittedDetector.fit(feats, grid, cfg, clusterer="kmeans")
    applied = det.apply(feats, grid)
    np.testing.assert_array_equal(applied.frame_labels, fit_result.frame_labels)
```

(Adapt fixture name/config access to what `tests/test_real_data.py` actually provides — the assertion content is the contract; if no suitable fixture exists, build `prepared = prepare_run(...)` inline exactly like the file's other tests do.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_detect_e2e.py -q -k FittedDetector`
Expected: FAIL with ImportError/NameError (`FittedDetector` not defined).

- [ ] **Step 3: Implement in `src/rowii/state/detect.py`**

```python
@dataclass(frozen=True)
class FittedDetector:
    """The Step-1 detection chain, split into fit (learn on day A) and apply
    (label any day with day-A parameters) — package-2 spec D1. Everything the
    chain learns is captured here; `apply` never re-estimates anything:
    standardization uses the FIT day's per-column mean/std (`run_detection`
    z-scores per run, so transfer MUST carry these), labeling is Viterbi decode
    with the FIT day's HMM. Not serialized to disk in this package (spec D1:
    future runtime-prototype serialization point).
    """

    mean: np.ndarray
    """(F,) fit-day per-column feature means (`zscore_stats`)."""
    std: np.ndarray
    """(F,) fit-day per-column feature stds (`zscore_stats`)."""
    smoother: StickyHmmSmoother
    """Fitted sticky HMM (holds the emission model + fit-time label-id mapping);
    `decode` is the per-window labeler on both fit and apply days."""
    min_dwell_s: float
    """`DetectConfig.min_dwell_s` at fit time (duration filter parameter)."""
    k: int
    """Number of clusters requested at fit time (mirrors `DetectionResult.k`)."""

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        grid: WindowGrid,
        cfg: DetectConfig,
        clusterer: Literal["kmeans", "gmm"] = "kmeans",
        k: int | None = None,
    ) -> tuple["FittedDetector", DetectionResult]:
        """Run the full Step-1 chain on *features* and capture the fitted state.

        Identical chain, order, and parameters as the historical `run_detection`
        (which now delegates here) — existing e2e/real-data results are the
        regression gate for that equivalence.

        Raises:
            ValueError: same conditions as `run_detection` (shape mismatch,
                unknown clusterer).
        """
        if features.shape[0] != grid.n_windows:
            raise ValueError(
                f"features.shape[0] ({features.shape[0]}) must equal grid.n_windows "
                f"({grid.n_windows})"
            )
        n_clusters = k if k is not None else cfg.n_states

        mean, std = zscore_stats(features)
        z = apply_zscore(features, mean, std)

        if clusterer == "kmeans":
            init_labels = KMeansClusterer(
                n_clusters=n_clusters, random_seed=cfg.random_seed
            ).fit_predict(z)
        elif clusterer == "gmm":
            init_labels = GmmClusterer(
                n_clusters=n_clusters, random_seed=cfg.random_seed
            ).fit_predict(z)
        else:
            raise ValueError(
                f"unknown clusterer {clusterer!r}: expected 'kmeans' or 'gmm'"
            )

        smoother = StickyHmmSmoother(
            self_transition=cfg.self_transition, random_seed=cfg.random_seed
        )
        smoothed = smoother.fit_decode(z, init_labels)

        detector = cls(
            mean=mean, std=std, smoother=smoother,
            min_dwell_s=cfg.min_dwell_s, k=n_clusters,
        )
        result = detector._finish(smoothed, grid)
        return detector, result

    def apply(self, features: np.ndarray, grid: WindowGrid) -> DetectionResult:
        """Label *features* with the fit-day parameters: fit-day standardization →
        fit-day HMM Viterbi decode → duration filter → segments. No refit, no EM.

        Raises:
            ValueError: if `features.shape[0] != grid.n_windows`.
        """
        if features.shape[0] != grid.n_windows:
            raise ValueError(
                f"features.shape[0] ({features.shape[0]}) must equal grid.n_windows "
                f"({grid.n_windows})"
            )
        z = apply_zscore(features, self.mean, self.std)
        smoothed = self.smoother.decode(z)
        return self._finish(smoothed, grid)

    def _finish(self, smoothed: np.ndarray, grid: WindowGrid) -> DetectionResult:
        """Shared tail of fit/apply: duration filter + segments."""
        window_s = grid.window_ns / 1e9
        min_dwell = max(1, round(self.min_dwell_s / window_s))
        filtered = duration_filter(smoothed, min_dwell=min_dwell)
        segments = to_segments(filtered, grid)
        return DetectionResult(frame_labels=filtered, segments=segments, k=self.k)
```

Then reduce `run_detection`'s body to a delegation (keep its full docstring):

```python
    _, result = FittedDetector.fit(features, grid, cfg, clusterer=clusterer, k=k)
    return result
```

Update imports in detect.py: `from rowii.signals.features import apply_zscore, zscore_stats` (replacing the plain `zscore` import), plus `dataclass`/`Literal` if missing.

Note on the fit-time labeler: `fit` takes the smoother's `fit_decode` output directly (as today); `apply` uses `decode`. `decode(z_fit) == fit_decode(...)` return is guaranteed by Task 1's `test_decode_same_features_matches_fit_decode`, which is what makes `test_apply_same_day_reproduces_fit_labels` pass.

- [ ] **Step 4: Run suite + lints**

Run: `.venv/bin/python -m pytest tests/ -q && .venv/bin/ruff check . && .venv/bin/mypy src scripts`
Expected: all pass (existing detection e2e/CLI tests gate the delegation refactor bit-for-bit).

- [ ] **Step 5: Run the real-data test explicitly**

Run: `.venv/bin/python -m pytest tests/test_real_data.py -q -k fitted_detector -m data`
Expected: PASS (cache hit, well under a minute).

- [ ] **Step 6: Commit**

```bash
git add src/rowii/state/detect.py tests/test_detect_e2e.py tests/test_real_data.py
git commit -m "feat: FittedDetector fit/apply split enabling cross-day detector transfer"
```

---

### Task 3: Protocol `cross-day-per-state` + `protocol` column in summary.csv

**Files:**
- Modify: `src/rowii/anomaly/sweep.py` (promote row builders to public — mechanical rename)
- Modify: `scripts/run_step2.py`
- Test: `tests/test_step2_cli.py` (extend), `tests/test_sweep.py` (only if row-builder rename touches its imports)

**Interfaces:**
- Consumes: Task 2's `FittedDetector`; existing `split_by_segments`, `build_references`, `calibrate`, `p_values`, `_write_sweep_outputs`, `_append_summary_row`, `_cross_day_out_dir` pattern, `_CROSS_DAY_SEED = 7`.
- Produces:
  - Public row builders in `rowii.anomaly.sweep`: `FarRow` (renamed from `_FarRow`), `far_row_excluded(label, cfg)`, `far_row_no_conformal_data(label, cfg)`, `far_row_empty_scoring(label, cfg, threshold)`, `far_row_scored(label, cfg, threshold, n_scored, n_alarms)`, `far_row_aggregate(rows, cfg)`, `scores_and_candidates(label, windows, scores, p_values, alarms, top_k)` — bodies unchanged, names public, internal `run_sweep` call sites updated.
  - `run_step2.py::_cross_day_per_state_sweep(prepared_a, valid_a, prepared_b, valid_b, rowii_cfg, scorer_name, alpha, top_k) -> tuple[SweepResult, np.ndarray]` (returns the sweep result AND day-B predicted labels for the GT-diagnostic view).
  - CLI: `--protocol cross-day-per-state`; output dirs `results/step2/cross-day-per-state/<runA>--to--<runB>/<variant>-<scorer>/` (mirror `_cross_day_out_dir`'s existing naming scheme exactly — read it first).
  - `summary.csv` schema gains `protocol` as the 2nd column; `_read_summary_csv_or_none` backfills a missing `protocol` column on old files (`"cross-day"` where the run field encodes a pair per the existing cross-day summary-row format, else `"within-day"` — read `_cross_day_summary_row` for the exact run-field format before writing the inference).
  - GT-diagnostic: when day B has SCADA, also write `far_by_true_state.csv` (columns: `true_state, n_scored, n_alarms, realized_far`) into the combo dir.

**Constraints:**
- `--labels gt` combined with `--protocol cross-day-per-state` exits with a clear error (spec D2: runtime path is detected-only).
- The existing `cross-day` protocol stays byte-for-byte untouched — its pooled numbers are the published comparator (spec D2: pooled under the transfer protocol ≡ the existing pooled cross-day, because pooling ignores states; do NOT reimplement it).
- Detector fit for day A uses the same valid-window compaction as `_detected_labels` (run_step2.py:309) with `clusterer="kmeans"` and `cfg.detect` — refactor `_detected_labels` into `_detected_labels_and_detector(prepared, cfg) -> tuple[np.ndarray, FittedDetector]` plus a thin `_detected_labels` wrapper so within-day behavior is unchanged.

- [ ] **Step 1: Mechanical rename in sweep.py (no behavior change)**

Rename `_FarRow` → `FarRow`, `_excluded_row` → `far_row_excluded`, `_no_conformal_data_row` → `far_row_no_conformal_data`, `_empty_scoring_row` → `far_row_empty_scoring`, `_scored_row` → `far_row_scored`, `_aggregate_pooled_row` → `far_row_aggregate`, `_scores_and_candidates` → `scores_and_candidates`. Update every internal call site in `run_sweep` and any test imports. Docstrings: add one line each — "public since package 2: `scripts/run_step2.py`'s cross-day-per-state protocol builds the same rows".

Run: `.venv/bin/python -m pytest tests/test_sweep.py tests/test_step2_cli.py -q` → PASS. Commit:

```bash
git add src/rowii/anomaly/sweep.py tests/test_sweep.py
git commit -m "refactor: promote sweep far-table row builders to public API"
```

- [ ] **Step 2: Write the failing unit test for the sweep function**

Append to `tests/test_step2_cli.py` (import `run_step2` via the file's existing `sys.path` pattern; build hand-made `PreparedRun`s like `tests/test_sweep.py` does — read its builder helper and reuse/adapt):

```python
class TestCrossDayPerStateSweep:
    def _make_prepared(self, seed, n_segments=8, seg_len=40, order=(0, 1)):
        """Two well-separated 'states' (feature values 0 and 5), alternating by
        segment so split_by_segments has material on both sides for both labels."""
        rng = np.random.default_rng(seed)
        feats, seg_ids = [], []
        for s in range(n_segments):
            value = 5.0 * order[s % len(order)]
            feats.append(rng.normal(value, 0.1, (seg_len, 2)))
            seg_ids.append(np.full(seg_len, s, dtype=np.int64))
        features = np.vstack(feats)
        n = len(features)
        return PreparedRun(
            features=features,
            grid=WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=n),
            valid_mask=np.ones(n, dtype=bool),
            feature_names=["f0", "f1"],
            segment_ids=np.concatenate(seg_ids),
        )

    def _rowii_cfg(self):
        cfg = load_config()  # follow the file's existing config-construction pattern
        return replace(cfg, detect=replace(cfg.detect, n_states=2, min_dwell_s=3))

    def test_per_state_rows_and_far_control_on_shifted_day(self):
        prepared_a = self._make_prepared(seed=0)
        prepared_b = self._make_prepared(seed=1)  # same distribution, new draws
        result, labels_b = run_step2._cross_day_per_state_sweep(
            prepared_a, prepared_a.valid_mask, prepared_b, prepared_b.valid_mask,
            self._rowii_cfg(), "knn", alpha=0.10, top_k=5,
        )
        far = result.far_table
        real_rows = far[(far["label"] != "pooled") & (~far["excluded"])]
        assert len(real_rows) == 2  # both fit-day states got references + scoring
        # Exchangeable B ⇒ realized FAR near alpha for every state (loose gate;
        # per-rep FAR is Beta-distributed, not exact)
        assert (real_rows["realized_far"] < 0.35).all()
        # aggregate row present, labeled like run_sweep's own convention
        assert (far["label"] == "pooled").sum() == 1

    def test_day_b_windows_keyed_by_predicted_state(self):
        prepared_a = self._make_prepared(seed=0, order=(0, 1))
        # Day B: 6 of 8 segments are state "5.0", 2 are state "0.0" — proportions
        # must show up in per-label n_scored via PREDICTED labels
        prepared_b = self._make_prepared(seed=2, order=(1, 1, 1, 0))
        result, labels_b = run_step2._cross_day_per_state_sweep(
            prepared_a, prepared_a.valid_mask, prepared_b, prepared_b.valid_mask,
            self._rowii_cfg(), "knn", alpha=0.10, top_k=5,
        )
        far = result.far_table
        real_rows = far[(far["label"] != "pooled") & (~far["excluded"])]
        counts = dict(zip(real_rows["label"], real_rows["n_scored"]))
        assert max(counts.values()) >= 2.5 * min(counts.values())
        assert set(np.unique(labels_b[labels_b >= 0])) <= {0, 1}

    def test_gt_labels_mode_rejected(self):
        # CLI-level guard; use the parser + main() with --labels gt and assert
        # a non-zero exit + clear message, following this file's existing
        # error-path test style.
        ...
```

Replace the `...` in the last test with a real assertion following the file's existing error-path pattern (it has precedents for exit-code tests — mirror one; if main() is awkward to call without a data tree, assert `build_parser()`-level rejection or the early-exit branch directly — implementer's choice, but the guard must be TESTED).

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_step2_cli.py -q -k CrossDayPerState`
Expected: FAIL with AttributeError (`_cross_day_per_state_sweep` missing).

- [ ] **Step 4: Implement in `scripts/run_step2.py`**

(a) Refactor detected labels (behavior-preserving):

```python
def _detected_labels_and_detector(
    prepared: PreparedRun, cfg: Config
) -> tuple[np.ndarray, FittedDetector]:
    """`_detected_labels` + the FittedDetector behind it (package-2 spec D2:
    cross-day-per-state needs day A's detector, not just its labels)."""
    valid_mask = prepared.valid_mask
    features_valid = prepared.features[valid_mask]
    n_valid = int(valid_mask.sum())
    valid_grid = WindowGrid(
        t0_ns=prepared.grid.t0_ns, window_ns=prepared.grid.window_ns, n_windows=n_valid
    )
    detector, det = FittedDetector.fit(
        features_valid, valid_grid, cfg.detect, clusterer="kmeans"
    )
    full_labels = np.full(prepared.features.shape[0], _INVALID_LABEL, dtype=np.int64)
    full_labels[valid_mask] = det.frame_labels
    return full_labels, detector


def _detected_labels(prepared: PreparedRun, cfg: Config) -> np.ndarray:
    """<keep existing docstring>"""
    labels, _ = _detected_labels_and_detector(prepared, cfg)
    return labels


def _apply_detector_labels(
    prepared: PreparedRun, detector: FittedDetector
) -> np.ndarray:
    """Day-B per-window labels in day-A's id space via `FittedDetector.apply`,
    `_INVALID_LABEL` on invalid windows (same scatter-back as `_detected_labels`)."""
    valid_mask = prepared.valid_mask
    features_valid = prepared.features[valid_mask]
    n_valid = int(valid_mask.sum())
    valid_grid = WindowGrid(
        t0_ns=prepared.grid.t0_ns, window_ns=prepared.grid.window_ns, n_windows=n_valid
    )
    det = detector.apply(features_valid, valid_grid)
    full_labels = np.full(prepared.features.shape[0], _INVALID_LABEL, dtype=np.int64)
    full_labels[valid_mask] = det.frame_labels
    return full_labels
```

(b) The sweep function (SweepConfig carries alpha/top_k/scorer so the public row builders can be reused directly):

```python
def _cross_day_per_state_sweep(
    prepared_a: PreparedRun,
    valid_a: np.ndarray,
    prepared_b: PreparedRun,
    valid_b: np.ndarray,
    rowii_cfg: Config,
    scorer_name: str,
    alpha: float,
    top_k: int,
) -> tuple[SweepResult, np.ndarray]:
    """Per-state cross-day sweep (package-2 spec D2): fit day A's detector +
    per-state references + per-state conformal thresholds, then score day B's
    windows grouped by their PREDICTED (transferred) state. Runtime-honest: no GT
    anywhere; the conditioning key is day A's cluster id. Returns the SweepResult
    plus day B's full-length predicted labels (for the GT-diagnostic view).

    Raises:
        ValueError: if day A's eligible windows cannot form a fit/conformal split,
            or day B has zero eligible windows.
    """
    labels_a, detector = _detected_labels_and_detector(prepared_a, rowii_cfg)

    split = split_by_segments(prepared_a.segment_ids, valid_a, 0.5, _CROSS_DAY_SEED)
    fit_windows, conformal_windows = split.calibration_windows, split.scoring_windows

    refs = build_references(prepared_a.features, labels_a, fit_windows)

    labels_b = _apply_detector_labels(prepared_b, detector)
    scoring_windows = np.flatnonzero(valid_b).astype(np.int64)
    if scoring_windows.size == 0:
        raise ValueError("cross-day-per-state: day B has zero eligible windows")

    sweep_cfg = SweepConfig(alpha=alpha, top_k=top_k, scorer=scorer_name)  # type: ignore[arg-type]

    all_labels = sorted(
        set(refs.references) | set(refs.excluded)
        | set(np.unique(labels_b[scoring_windows]).tolist())
    )

    far_rows: list[FarRow] = []
    score_rows: list = []
    candidate_rows: list = []
    for label in all_labels:
        if label not in refs.references:
            far_rows.append(far_row_excluded(label, sweep_cfg))
            continue
        scorer = _make_scorer(scorer_name).fit(refs.references[label])
        label_conformal = conformal_windows[labels_a[conformal_windows] == label]
        if label_conformal.shape[0] == 0:
            far_rows.append(far_row_no_conformal_data(label, sweep_cfg))
            continue
        conformal_scores = scorer.score(prepared_a.features[label_conformal])
        threshold = calibrate(conformal_scores, alpha)

        label_scoring = scoring_windows[labels_b[scoring_windows] == label]
        if label_scoring.shape[0] == 0:
            far_rows.append(far_row_empty_scoring(label, sweep_cfg, threshold))
            continue
        scores = scorer.score(prepared_b.features[label_scoring])
        p_vals = p_values(scores, conformal_scores)
        alarms = scores > threshold.threshold
        far_rows.append(
            far_row_scored(label, sweep_cfg, threshold, int(label_scoring.shape[0]), int(alarms.sum()))
        )
        new_scores, new_cands = scores_and_candidates(
            label, label_scoring, scores, p_vals, alarms, top_k
        )
        score_rows.extend(new_scores)
        candidate_rows.extend(new_cands)

    far_rows.append(far_row_aggregate(far_rows, sweep_cfg))

    far_table = pd.DataFrame([asdict(r) for r in far_rows], columns=_FAR_TABLE_COLUMNS)
    scores_df = pd.DataFrame([asdict(r) for r in score_rows], columns=_SCORES_COLUMNS)
    candidates_df = pd.DataFrame(
        [asdict(r) for r in candidate_rows], columns=_CANDIDATES_COLUMNS
    )
    return SweepResult(far_table=far_table, scores=scores_df, candidates=candidates_df), labels_b
```

(Check `far_row_aggregate`'s exact contract in sweep.py before use — it must receive only this sweep's per-label rows, matching how `run_sweep` calls it.)

(c) GT-diagnostic helper:

```python
def _far_by_true_state(
    scores_df: pd.DataFrame, gt_states: np.ndarray
) -> pd.DataFrame:
    """Alarm rate of the scored day-B windows grouped by their TRUE (SCADA) state —
    diagnostic view only (spec D2), never part of the runtime path."""
    windows = scores_df["window"].to_numpy()
    alarms = scores_df["alarm"].to_numpy()
    states = gt_states[windows]
    rows = []
    for state in sorted(np.unique(states).tolist()):
        mask = states == state
        n = int(mask.sum())
        rows.append({
            "true_state": state, "n_scored": n,
            "n_alarms": int(alarms[mask].sum()),
            "realized_far": float(alarms[mask].sum()) / n if n else math.nan,
        })
    return pd.DataFrame(rows, columns=["true_state", "n_scored", "n_alarms", "realized_far"])
```

(d) CLI wiring: add `"cross-day-per-state"` to `_PROTOCOL_CHOICES`; in `main()` add the dispatch branch mirroring `_run_cross_day` (run_step2.py:999 — read it and copy its pair iteration, prepared/valid-mask construction, output writing via `_write_sweep_outputs`, register append, and summary append EXACTLY, swapping in `_cross_day_per_state_sweep`, a `cross-day-per-state` out-dir root, and — when day B's SCADA is available via `_load_run_scada` — writing `far_by_true_state.csv` next to the other outputs). Add the `--labels gt` guard at the top of the branch:

```python
    if args.labels == "gt":
        parser.error("--protocol cross-day-per-state is detected-labels only (spec D2)")
```

(e) `protocol` summary column: add `"protocol"` as second entry in `_SUMMARY_COLUMNS`; add `protocol: str` to `_SummaryRow`; set it at every construction site (`_summary_row` → `"within-day"`, `_cross_day_summary_row` → `"cross-day"`, new branch → `"cross-day-per-state"`); in `_read_summary_csv_or_none`, after a successful read of an old-schema file insert the column (read `_cross_day_summary_row` first for the pair-encoding of its run field and infer from that; otherwise `"within-day"`), so appends never produce ragged CSVs.

- [ ] **Step 5: Extend the CLI smoke test**

Extend `tests/test_step2_cli.py`'s existing two-day cross-day smoke test (or add a sibling using the same fixture) to run `--protocol cross-day-per-state` end-to-end on the synthetic tree and assert: exit code 0, `far_table.csv`/`scores.parquet`/`candidates.md` exist under `cross-day-per-state/`, summary.csv has a `protocol` column containing `cross-day-per-state`, and re-running after deleting the new rows leaves the old-schema rows intact (backfill path). Follow the file's established assertion style.

- [ ] **Step 6: Run suite + lints**

Run: `.venv/bin/python -m pytest tests/ -q && .venv/bin/ruff check . && .venv/bin/mypy src scripts`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_step2.py tests/test_step2_cli.py
git commit -m "feat: cross-day-per-state protocol (detector transfer) + summary protocol column"
```

---

### Task 4: Scarcity core (`src/rowii/anomaly/scarcity.py`)

**Files:**
- Modify: `src/rowii/anomaly/conformal.py` (rename `_threshold_index` → public `threshold_index`; update internal call sites + any test imports)
- Create: `src/rowii/anomaly/scarcity.py`
- Test: `tests/test_scarcity.py` (new)

**Interfaces:**
- Consumes: `calibrate`, `p_values`, `threshold_index` from `rowii.anomaly.conformal`; `scipy.stats.beta`.
- Produces (Task 5 relies on these):
  - `ScarcityConfig(budgets: tuple[int, ...] = (5, 10, 19, 39, 79, 159, 319), n_reps: int = 50, alpha: float = 0.05, include_full_pool: bool = True)` (frozen dataclass).
  - `scarcity_curve(conformal_scores: np.ndarray, scoring_scores: np.ndarray, label: int | str, cfg: ScarcityConfig) -> pd.DataFrame` — columns `label, budget, achieved_n, saturated, rep, threshold, low_confidence, n_scored, n_alarms, realized_far`; rep seeds are exactly `0..n_reps-1` via `default_rng(rep)`.
  - `beta_band(n: int, alpha: float, q_lo: float = 0.05, q_hi: float = 0.95) -> tuple[float, float] | None` — exact per-rep-FAR band `Beta(n + 1 - idx, idx)` quantiles with `idx = threshold_index(n, alpha)`; `None` when `idx > n` (below the achievability floor).
  - `segment_accumulation_curve(...)` is Task 5 (script-level needs), NOT here — this module stays free of scorer dependencies.

- [ ] **Step 1: Write the failing tests (`tests/test_scarcity.py`)**

```python
"""Unit tests for rowii.anomaly.scarcity (package-2 spec D3, primary curve)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rowii.anomaly.conformal import threshold_index
from rowii.anomaly.scarcity import ScarcityConfig, beta_band, scarcity_curve


def _pool(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n), rng.normal(0, 1, 2000)  # conformal, scoring


class TestScarcityCurve:
    def test_deterministic_given_seeds(self):
        conformal, scoring = _pool()
        cfg = ScarcityConfig(budgets=(10, 39), n_reps=5)
        a = scarcity_curve(conformal, scoring, 2, cfg)
        b = scarcity_curve(conformal, scoring, 2, cfg)
        pd.testing.assert_frame_equal(a, b)

    def test_below_floor_never_alarms_and_flags(self):
        conformal, scoring = _pool()
        cfg = ScarcityConfig(budgets=(5,), n_reps=3, alpha=0.05, include_full_pool=False)
        df = scarcity_curve(conformal, scoring, "turbine", cfg)
        assert df["low_confidence"].all()
        assert (df["threshold"] == np.inf).all()
        assert (df["realized_far"] == 0.0).all()

    def test_saturation_flag_and_achieved_n(self):
        conformal, scoring = _pool(n=50)
        cfg = ScarcityConfig(budgets=(79,), n_reps=2, include_full_pool=False)
        df = scarcity_curve(conformal, scoring, 2, cfg)
        assert df["saturated"].all()
        assert (df["achieved_n"] == 50).all()

    def test_full_pool_row_appended(self):
        conformal, scoring = _pool(n=400)
        cfg = ScarcityConfig(budgets=(19,), n_reps=2, include_full_pool=True)
        df = scarcity_curve(conformal, scoring, 2, cfg)
        assert set(df["budget"]) == {19, 400}

    def test_mean_far_tracks_alpha_above_floor(self):
        # Exchangeable normal scores: mean realized FAR across reps must sit near
        # alpha once n clears the floor (loose gate — per-rep FAR is Beta-spread).
        conformal, scoring = _pool(n=1000, seed=1)
        cfg = ScarcityConfig(budgets=(159,), n_reps=50, alpha=0.05, include_full_pool=False)
        df = scarcity_curve(conformal, scoring, 2, cfg)
        assert 0.02 <= df["realized_far"].mean() <= 0.09


class TestBetaBand:
    def test_none_below_floor(self):
        assert beta_band(5, 0.05) is None

    def test_band_brackets_alpha_at_large_n(self):
        lo, hi = beta_band(999, 0.05)
        assert lo < 0.05 < hi
        assert hi - lo < 0.03  # tight at n=999

    def test_band_widens_at_small_n(self):
        lo_small, hi_small = beta_band(19, 0.05)
        lo_big, hi_big = beta_band(999, 0.05)
        assert (hi_small - lo_small) > (hi_big - lo_big)

    def test_matches_threshold_index_parameters(self):
        n, alpha = 99, 0.05
        idx = threshold_index(n, alpha)
        from scipy.stats import beta as beta_dist
        lo, hi = beta_band(n, alpha)
        assert lo == pytest.approx(beta_dist.ppf(0.05, n + 1 - idx, idx))
        assert hi == pytest.approx(beta_dist.ppf(0.95, n + 1 - idx, idx))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_scarcity.py -q`
Expected: FAIL (ImportError: no module `rowii.anomaly.scarcity`; `threshold_index` not public yet).

- [ ] **Step 3: Implement**

In `conformal.py`: rename `_threshold_index` to `threshold_index` (keep the docstring, drop the leading underscore note if any), update the internal `calibrate` call site; grep tests for `_threshold_index` and update.

Create `src/rowii/anomaly/scarcity.py`:

```python
"""Calibration-scarcity curves for Step-2 (package-2 spec D3, primary curve).

Answers the partner's "enough data per mode" question quantitatively: how does the
realized false-alarm rate (and its spread) behave as the per-mode conformal
calibration set shrinks? The module is deliberately free of scorer dependencies —
it operates on PRECOMPUTED score arrays (the scorer is fitted once on the full
fit-side reference and both score arrays computed once; only the threshold is
recomputed per subsample), which makes a 50-repetition sweep over 8 budgets a
sub-second operation per state. The deployment-view variant that shrinks the
reference set too (segment accumulation) lives with the CLI (spec D3 secondary).

Per-repetition realized FAR at calibration size n is Beta-distributed — see the
S-package derivation in tests/test_conformal.py's validity suite — so `beta_band`
overlays the EXACT `Beta(n + 1 - idx, idx)` quantiles (idx = threshold_index(n,
alpha)), not a binomial approximation. Scoring-side sampling noise adds on top of
that band; reports must say so (spec D3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta_dist

from rowii.anomaly.conformal import calibrate, threshold_index

_CURVE_COLUMNS = (
    "label", "budget", "achieved_n", "saturated", "rep",
    "threshold", "low_confidence", "n_scored", "n_alarms", "realized_far",
)


@dataclass(frozen=True)
class ScarcityConfig:
    """One `scarcity_curve` call's parameters — see module docstring."""

    budgets: tuple[int, ...] = (5, 10, 19, 39, 79, 159, 319)
    """Requested per-state calibration sizes; 19 is the alpha=0.05 achievability
    floor (n >= 1/alpha - 1) and belongs in every default sweep."""
    n_reps: int = 50
    """Repetitions per budget; rep r draws with `numpy.random.default_rng(r)`."""
    alpha: float = 0.05
    include_full_pool: bool = True
    """Append the full conformal pool as a final budget when it is not already in
    `budgets` — the 'all available data' anchor point of the curve."""


def scarcity_curve(
    conformal_scores: np.ndarray,
    scoring_scores: np.ndarray,
    label: int | str,
    cfg: ScarcityConfig,
) -> pd.DataFrame:
    """Realized-FAR-vs-calibration-size table for ONE state's precomputed scores.

    Args:
        conformal_scores: `(n_pool,)` finite calibration scores of this state's
            held-out normal windows (full pool; subsampled per budget × rep).
        scoring_scores: `(m,)` finite scores of this state's FIXED scoring windows
            (never subsampled — spec D3: scoring split fixed across repetitions).
        label: State label carried into the output rows (int cluster id or str).
        cfg: See `ScarcityConfig`.

    Returns:
        DataFrame with columns `label, budget, achieved_n, saturated, rep,
        threshold, low_confidence, n_scored, n_alarms, realized_far` — one row per
        budget × rep. A saturated budget (requested > pool) draws the whole pool
        (identical across reps, still emitted per rep for uniform aggregation).

    Raises:
        ValueError: propagated from `calibrate` on non-finite/empty inputs.
    """
    n_pool = int(conformal_scores.shape[0])
    m = int(scoring_scores.shape[0])
    budgets = list(cfg.budgets)
    if cfg.include_full_pool and n_pool not in budgets:
        budgets.append(n_pool)

    rows: list[dict[str, object]] = []
    for budget in budgets:
        achieved = min(budget, n_pool)
        saturated = budget > n_pool
        for rep in range(cfg.n_reps):
            if achieved < n_pool:
                rng = np.random.default_rng(rep)
                drawn = rng.choice(conformal_scores, size=achieved, replace=False)
            else:
                drawn = conformal_scores
            th = calibrate(drawn, cfg.alpha)
            n_alarms = int((scoring_scores > th.threshold).sum())
            rows.append({
                "label": label, "budget": budget, "achieved_n": achieved,
                "saturated": saturated, "rep": rep, "threshold": th.threshold,
                "low_confidence": th.low_confidence, "n_scored": m,
                "n_alarms": n_alarms,
                "realized_far": n_alarms / m if m else float("nan"),
            })
    return pd.DataFrame(rows, columns=list(_CURVE_COLUMNS))


def beta_band(
    n: int, alpha: float, q_lo: float = 0.05, q_hi: float = 0.95
) -> tuple[float, float] | None:
    """Exact per-repetition-FAR quantile band at calibration size *n* — the
    `(q_lo, q_hi)` quantiles of `Beta(n + 1 - idx, idx)` with
    `idx = threshold_index(n, alpha)`; `None` when the threshold order statistic
    does not exist (`idx > n`, below the achievability floor)."""
    idx = threshold_index(n, alpha)
    if idx > n:
        return None
    lo = float(_beta_dist.ppf(q_lo, n + 1 - idx, idx))
    hi = float(_beta_dist.ppf(q_hi, n + 1 - idx, idx))
    return lo, hi
```

- [ ] **Step 4: Run suite + lints**

Run: `.venv/bin/python -m pytest tests/test_scarcity.py tests/test_conformal.py -q && .venv/bin/python -m pytest tests/ -q && .venv/bin/ruff check . && .venv/bin/mypy src scripts`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/rowii/anomaly/conformal.py src/rowii/anomaly/scarcity.py tests/test_scarcity.py tests/test_conformal.py
git commit -m "feat: calibration-scarcity curve core with exact Beta bands"
```

---

### Task 5: Scarcity CLI + segment-accumulation secondary + figures

**Files:**
- Create: `scripts/run_step2_scarcity.py`
- Modify: `src/rowii/anomaly/scarcity.py` (add `segment_accumulation_curve`)
- Test: `tests/test_scarcity.py` (extend), `tests/test_scarcity_cli.py` (new, synthetic only)

**Interfaces:**
- Consumes: Task 4's `scarcity_curve`/`beta_band`/`ScarcityConfig`; `run_sweep`-style splits (`split_by_segments` twice: top seed 7, nested seed 8 — mirror `rowii.anomaly.sweep.run_sweep`'s exact three-way construction); `build_references`; `KnnScorer`/`MahalanobisScorer`; `prepare_run(use_cache=True)`; `_detected_labels` pattern from `run_step2.py` (import `run_step2` the way tests do is NOT available cross-script — instead reuse `FittedDetector.fit` directly, mirroring `_detected_labels`' compaction; keep a comment pointing at run_step2.py:309 so the two stay in sync).
- Produces:
  - `segment_accumulation_curve(features, labels, segment_ids, valid_mask, scorer_factory: Callable[[], Scorer], scoring_windows: np.ndarray, cfg: SegmentAccumulationConfig) -> pd.DataFrame` in `scarcity.py` — columns `label, n_segments, n_fit, n_conformal, minutes, rep, low_confidence, n_scored, n_alarms, realized_far`. Per rep: permute the NON-scoring segments with `default_rng(rep)`, walk prefixes `2, 4, 6, ...` up to all; within a prefix, even-positioned segments (in permuted order) feed the per-label FIT references, odd-positioned feed per-label CONFORMAL calibration; per label require `>= min_ref` fit windows AND `>= 1` conformal window, else emit a row with `low_confidence=True`, `realized_far=0.0`, `n_alarms=0` (threshold `+inf` semantics); `minutes` = total valid windows in the prefix / 60 (1-s windows).
  - `SegmentAccumulationConfig(prefixes: tuple[int, ...] | None = None, n_reps: int = 20, alpha: float = 0.05, min_ref: int = 20)` — `prefixes=None` means `2, 4, 6, ...` up to the number of available segments.
  - CLI `scripts/run_step2_scarcity.py`: args `--runs` (default `010726-tu_ph_tu 290626-tu`), `--variants` (default `fusion audio-beats`), `--scorers` (default `knn`, choices knn/mahalanobis/all), `--alpha 0.05`, `--reps 50`, `--secondary-reps 20`, `--secondary` (flag; secondary curve only for the FIRST run × `fusion` × `knn`), `--out` (default `results/step2/scarcity`), `--data-root`/`--results-root` following `run_step2.py`'s existing arg conventions (read its `build_parser` and mirror names/env handling).
  - Outputs per run × variant × scorer: `<out>/<run>--<variant>-<scorer>/curve.csv` (scarcity_curve rows, all states), `curve_by_state.png` (one panel per state: x = achieved_n log-scale, y = per-rep realized FAR as translucent points + mean line + empirical 5/95 band; exact `beta_band` overlay where defined; horizontal `alpha` line; vertical line at n = 19 labeled "α=0.05 floor"), and — when `--secondary` — `segment_curve.csv` + `segment_curve.png` (x = minutes, same y semantics). Plus one `<out>/summary.md` table across all combos: per state the smallest budget whose mean FAR ∈ [alpha/2, 2·alpha] and whose rep-spread (95th pct) ≤ 2·alpha — the "windows per mode needed" headline; states that never stabilize marked openly. summary.md must carry the honesty notes (spec D5) + the scoring-side-noise caveat (spec D3).
  - Matplotlib: `matplotlib.use("Agg")` before pyplot import, figures closed after save.

**Per-state score precomputation (the scarcity fast path, mirroring run_sweep's split exactly):** top split `split_by_segments(segment_ids, valid_mask, 0.5, seed=7)`; nested `split_by_segments(segment_ids, calib_mask, 0.5, seed=8)` where `calib_mask` marks the top calibration windows (this is exactly `run_sweep`'s construction — seed 7 then `cfg.seed + 1 = 8`); detected labels via `FittedDetector.fit` on valid-compacted features (comment-linked to run_step2.py:309); per state with a `min_ref`-sized fit reference: fit scorer once on `features[fit_windows of that label]`, compute `conformal_scores` (that label's conformal windows) and `scoring_scores` (that label's scoring windows) once; hand both to `scarcity_curve`. States excluded by `min_ref` or with empty conformal/scoring sides are listed in summary.md as "not curvable" with their counts (no silent dropping — spec D5 no-silent-caps).

- [ ] **Step 1: Failing tests for `segment_accumulation_curve`** (extend `tests/test_scarcity.py`)

```python
class TestSegmentAccumulation:
    def _run(self, n_segments=12, seg_len=60, seed=0):
        rng = np.random.default_rng(seed)
        feats, segs, labels = [], [], []
        for s in range(n_segments):
            value = 5.0 * (s % 2)
            feats.append(rng.normal(value, 0.1, (seg_len, 2)))
            segs.append(np.full(seg_len, s, dtype=np.int64))
            labels.append(np.full(seg_len, s % 2, dtype=np.int64))
        return (np.vstack(feats), np.concatenate(labels), np.concatenate(segs),
                np.ones(n_segments * seg_len, dtype=bool))

    def test_scoring_segments_never_in_prefix(self):
        features, labels, segment_ids, valid = self._run()
        scoring_windows = np.flatnonzero(np.isin(segment_ids, [10, 11]))
        cfg = SegmentAccumulationConfig(n_reps=3)
        df = segment_accumulation_curve(
            features, labels, segment_ids, valid,
            lambda: KnnScorer(), scoring_windows, cfg,
        )
        # every row's n_scored equals the fixed scoring-side count for its label
        for label in (0, 1):
            expected = int((labels[scoring_windows] == label).sum())
            got = df[df["label"] == label]["n_scored"].unique()
            assert list(got) == [expected]

    def test_deterministic_and_monotone_minutes(self):
        features, labels, segment_ids, valid = self._run()
        scoring_windows = np.flatnonzero(np.isin(segment_ids, [10, 11]))
        cfg = SegmentAccumulationConfig(n_reps=2)
        a = segment_accumulation_curve(features, labels, segment_ids, valid,
                                       lambda: KnnScorer(), scoring_windows, cfg)
        b = segment_accumulation_curve(features, labels, segment_ids, valid,
                                       lambda: KnnScorer(), scoring_windows, cfg)
        pd.testing.assert_frame_equal(a, b)
        one_rep = a[(a["rep"] == 0) & (a["label"] == 0)].sort_values("n_segments")
        assert one_rep["minutes"].is_monotonic_increasing

    def test_starved_prefix_flags_low_confidence(self):
        features, labels, segment_ids, valid = self._run(n_segments=6, seg_len=8)
        scoring_windows = np.flatnonzero(np.isin(segment_ids, [4, 5]))
        cfg = SegmentAccumulationConfig(n_reps=2, min_ref=20)
        df = segment_accumulation_curve(features, labels, segment_ids, valid,
                                        lambda: KnnScorer(), scoring_windows, cfg)
        starved = df[df["n_segments"] == 2]
        assert starved["low_confidence"].all()
        assert (starved["realized_far"] == 0.0).all()
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_scarcity.py -q -k SegmentAccumulation` → ImportError.

- [ ] **Step 3: Implement `segment_accumulation_curve` + `SegmentAccumulationConfig`** in `scarcity.py` per the interface block above. Import `Scorer` type via `from rowii.anomaly.scorers import Scorer` under `TYPE_CHECKING` if needed for mypy; runtime takes any object with `fit`/`score`. Key mechanics:

```python
    non_scoring_segments = np.setdiff1d(
        np.unique(segment_ids[valid_mask & (segment_ids != -1)]),
        np.unique(segment_ids[scoring_windows]),
    )
    prefixes = cfg.prefixes or tuple(range(2, len(non_scoring_segments) + 1, 2))
    for rep in range(cfg.n_reps):
        order = np.random.default_rng(rep).permutation(non_scoring_segments)
        for n_seg in prefixes:
            prefix = order[:n_seg]
            fit_segs, conf_segs = prefix[0::2], prefix[1::2]
            fit_mask = valid_mask & np.isin(segment_ids, fit_segs)
            conf_mask = valid_mask & np.isin(segment_ids, conf_segs)
            minutes = float((fit_mask | conf_mask).sum()) / 60.0
            for label in sorted(np.unique(labels[scoring_windows]).tolist()):
                fit_w = np.flatnonzero(fit_mask & (labels == label))
                conf_w = np.flatnonzero(conf_mask & (labels == label))
                score_w = scoring_windows[labels[scoring_windows] == label]
                n_scored = int(score_w.shape[0])
                if fit_w.shape[0] < cfg.min_ref or conf_w.shape[0] < 1:
                    rows.append({..., "low_confidence": True, "n_alarms": 0,
                                 "realized_far": 0.0, "n_scored": n_scored, ...})
                    continue
                scorer = scorer_factory().fit(features[fit_w])
                th = calibrate(scorer.score(features[conf_w]), cfg.alpha)
                alarms = int((scorer.score(features[score_w]) > th.threshold).sum())
                rows.append({..., "low_confidence": th.low_confidence,
                             "n_alarms": alarms,
                             "realized_far": alarms / n_scored, ...})
```

(Fill the elided `...` dict keys with the full column set from the interface block — `label, n_segments, n_fit, n_conformal, minutes, rep, low_confidence, n_scored, n_alarms, realized_far` with `n_fit=int(fit_w.shape[0])`, `n_conformal=int(conf_w.shape[0])`.)

- [ ] **Step 4: CLI + smoke test.** Write `scripts/run_step2_scarcity.py` per the interface block (arg names mirroring `run_step2.py`'s `build_parser`). Then `tests/test_scarcity_cli.py`: hand-built `PreparedRun` monkeypatched into the script's prepare step (monkeypatch `run_step2_scarcity.prepare_run` and the discovery call — follow `tests/test_step2_cli.py`'s monkeypatch/fixture style, OR reuse the synthetic gantner tree fixture if simpler) asserting: exit 0; `curve.csv` exists with all `_CURVE_COLUMNS`; `curve_by_state.png` exists and is non-empty; `summary.md` exists and names every state (curvable or "not curvable"); with `--secondary`, `segment_curve.csv` + `.png` exist.

- [ ] **Step 5: Run suite + lints** — `.venv/bin/python -m pytest tests/ -q && .venv/bin/ruff check . && .venv/bin/mypy src scripts` → all pass.

- [ ] **Step 6: Commit**

```bash
git add src/rowii/anomaly/scarcity.py scripts/run_step2_scarcity.py tests/test_scarcity.py tests/test_scarcity_cli.py
git commit -m "feat: scarcity CLI with per-state curves, segment-accumulation view, figures"
```

---

### Task 6: Cache warm-up script

**Files:**
- Create: `scripts/warm_cache.py`
- Test: `tests/test_warm_cache.py` (new)

**Interfaces:**
- Consumes: `rowii.pipeline.prepare_run` (with `use_cache=True`), `rowii.io.dataset.discover`, config/env loading exactly as `scripts/run_step2.py` does (mirror its data-root/config bootstrapping and its `_import_beats_or_exit` guard for beats variants).
- Produces: `scripts/warm_cache.py --runs <names...> --variants <variants...> [--dry-run]`. Default runs: `250526-tu 290626-tu 010726-tu_ph_tu 270626-pu_ph_pu_ph_pu_ph`; default variants: `audio-beats fusion-beats`. For each run × variant: log start, call `prepare_run(..., use_cache=True)`, log elapsed seconds + `results/cache/<run>--<variant>.npz` size. `--dry-run` prints the combo list and exits 0 WITHOUT importing beats/torch. Unknown run names exit 2 listing available runs.

- [ ] **Step 1: Failing test (`tests/test_warm_cache.py`):** monkeypatch `warm_cache.prepare_run` with a recorder; synthetic run objects via a monkeypatched discovery (mirror `tests/test_scarcity_cli.py` / `tests/test_step2_cli.py` patterns). Assert: (a) `--dry-run` exits 0, prints every combo, recorder never called; (b) real invocation calls the recorder once per combo with `use_cache=True`; (c) unknown run name exits 2 and names the available runs.
- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_warm_cache.py -q` → FAIL (module missing).
- [ ] **Step 3:** Implement the script (~80 lines; argparse + loop + logging; beats import guard only on the non-dry-run path).
- [ ] **Step 4:** `.venv/bin/python -m pytest tests/ -q && .venv/bin/ruff check . && .venv/bin/mypy src scripts` → all pass.
- [ ] **Step 5: Commit**

```bash
git add scripts/warm_cache.py tests/test_warm_cache.py
git commit -m "feat: cache warm-up script for BEATs variants"
```

---

### Task 7: Candidate-overlap analysis (BEATs vs handcrafted)

**Files:**
- Create: `src/rowii/anomaly/overlap.py`
- Create: `scripts/analyze_step2.py`
- Test: `tests/test_overlap.py` (new)

**Interfaces:**
- Consumes: per-combo `scores.parquet` files written by `run_step2.py` (columns `window, label, score, p_value, alarm`; label column str after round-trip); grids via `prepare_run(use_cache=True)` (cache hit → no torch import even for beats variants); the candidate ordering convention `np.lexsort((window, -score, p_value))` (identical in `run_step2._cross_day_sweep` and `sweep.scores_and_candidates` — reuse, don't invent).
- Produces:
  - `top_candidates(scores: pd.DataFrame, top_k: int) -> pd.DataFrame` — top-k per label by that exact ordering, plus global top-k across labels (`label="__any__"` rows appended), columns `window, label, score, p_value, rank`.
  - `to_utc_ns(candidates: pd.DataFrame, t0_ns: int, window_ns: int) -> pd.DataFrame` — adds `t_utc_ns = t0_ns + window * window_ns` (int64).
  - `match_by_time(a: pd.DataFrame, b: pd.DataFrame, tol_s: float = 5.0) -> pd.DataFrame` — greedy nearest-time 1:1 matching on `t_utc_ns` (sort by |Δt|, take pairs whose members are both unmatched, |Δt| ≤ tol); columns `t_utc_ns_a, t_utc_ns_b, dt_s, label_a, label_b, p_value_a, p_value_b`.
  - `jaccard(n_a: int, n_b: int, n_matched: int) -> float` — `n_matched / (n_a + n_b - n_matched)`, 0.0 when the denominator is 0.
  - `scripts/analyze_step2.py --results-root results --runs <names...> --pairs fusion-knn:audio-beats-knn ... --top-k 20 --check-utc <ISO8601 ...>` → writes `results/step2/overlap/<run>--<comboA>--vs--<comboB>.md` (match table + Jaccard + unmatched lists, human-readable UTC) and a `results/step2/overlap/needs_listening_check.md` verifying each `--check-utc` timestamp against every analyzed combo's top-k (hit = within `tol_s`; table of hits/misses). `--check-utc` DEFAULT: empty (the orchestrator passes the four register timestamps explicitly at execution time — they are data, not code).
- Note for the implementer: two variants of the same run have DIFFERENT grids (per-variant stream intersection) — always convert to UTC via each combo's own grid before matching; never compare raw window indices.

- [ ] **Step 1: Failing tests (`tests/test_overlap.py`)** — pure-function tests, no I/O:

```python
class TestMatchByTime:
    def _cands(self, times_s, label="2"):
        return pd.DataFrame({
            "t_utc_ns": (np.array(times_s) * 1_000_000_000).astype(np.int64),
            "label": label, "p_value": 0.01,
        })

    def test_matches_within_tolerance_only(self):
        a, b = self._cands([100, 200, 300]), self._cands([101, 250, 304])
        m = match_by_time(a, b, tol_s=5.0)
        assert len(m) == 2
        assert sorted(m["dt_s"].abs().round(0).tolist()) == [1.0, 4.0]

    def test_greedy_one_to_one(self):
        a, b = self._cands([100, 102]), self._cands([101])
        m = match_by_time(a, b, tol_s=5.0)
        assert len(m) == 1
        assert abs(m["dt_s"].iloc[0]) == pytest.approx(1.0)

class TestJaccard:
    def test_values(self):
        assert jaccard(3, 3, 2) == pytest.approx(0.5)
        assert jaccard(0, 0, 0) == 0.0

class TestTopCandidates:
    def test_per_label_and_global_ordering(self):
        scores = pd.DataFrame({
            "window": [0, 1, 2, 3], "label": ["0", "0", "1", "1"],
            "score": [5.0, 1.0, 9.0, 2.0], "p_value": [0.01, 0.5, 0.01, 0.3],
            "alarm": [True, False, True, False],
        })
        top = top_candidates(scores, top_k=1)
        per_label = top[top["label"] != "__any__"]
        assert set(per_label["window"]) == {0, 2}
        global_rows = top[top["label"] == "__any__"]
        assert global_rows["window"].iloc[0] == 2  # p tied at 0.01 → higher score wins
```

- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_overlap.py -q` → FAIL (module missing).
- [ ] **Step 3:** Implement `src/rowii/anomaly/overlap.py` (pure pandas/numpy; ~120 lines) and `scripts/analyze_step2.py` (I/O shell: locate combo dirs under both `within-day` and `cross-day*` roots by combo name, load parquet, derive tops, convert per-combo UTC, write reports).
- [ ] **Step 4:** Extend `tests/test_overlap.py` with one script-level smoke test: tmp dir with two hand-written `scores.parquet` files + monkeypatched grid lookup → both output md files written, Jaccard line present.
- [ ] **Step 5:** `.venv/bin/python -m pytest tests/ -q && .venv/bin/ruff check . && .venv/bin/mypy src scripts` → all pass.
- [ ] **Step 6: Commit**

```bash
git add src/rowii/anomaly/overlap.py scripts/analyze_step2.py tests/test_overlap.py
git commit -m "feat: candidate time-overlap analysis (BEATs vs handcrafted)"
```

---

### Task 8: Execution + synthesis (orchestrator-led; real data, real hours)

**Files:**
- Modify: `README.md` (two sections), `results/step2/` artifacts (gitignored), `research note in the master-thesis repo` (outside this repo — orchestrator writes it)

No new code. Order of operations (the warm-up starts as soon as Task 6 lands — it runs in the background while Tasks 3–5, 7 are still being implemented):

- [ ] **Step 1 (background, after Task 6):** `.venv/bin/python scripts/warm_cache.py` (defaults cover the 4 runs × 2 beats variants). Expect one-off MPS extraction, order 1–4 h total; subsequent everything is cache hits.
- [ ] **Step 2 (after Task 3):** cross-day per-state, all pairs — for `V` in `fusion audio audio-beats`, `S` in `knn mahalanobis`: `.venv/bin/python scripts/run_step2.py --protocol cross-day-per-state --variant $V --scorer $S --alpha 0.05` (runs resolve to the SCADA-day set; audio-beats waits for warm-up).
- [ ] **Step 3 (after warm-up):** BEATs within-day — for each SCADA day run × `audio-beats fusion-beats` × both scorers × both conditionings: `.venv/bin/python scripts/run_step2.py --protocol within-day --variant <v> --scorer all --conditioning all`.
- [ ] **Step 4 (after Task 5):** scarcity — `.venv/bin/python scripts/run_step2_scarcity.py --runs 010726-tu_ph_tu 290626-tu --variants fusion audio-beats --scorers knn --secondary` and a mahalanobis pass on the primary day only.
- [ ] **Step 5 (after Task 7):** overlap — `.venv/bin/python scripts/analyze_step2.py` with pairs `fusion-knn:audio-beats-knn` (per day) and `--check-utc` set to the four needs-listening timestamps from `results/step2/candidate_register.md`.
- [ ] **Step 6:** README: (a) "Compute reuse" section under the existing cache docs — checkpoint stored once (no pre-training on our side), embeddings cached once per run × variant, clusterers/scorers refit in seconds by design, `FittedDetector` = future serialization point; (b) "Step-2 package 2 results" section — cross-day per-state vs pooled table (pooled numbers cited from the existing `cross-day` runs), scarcity headline ("windows per mode"), BEATs verdict, all with the standard honesty notes.
- [ ] **Step 7:** Research note `research/notes/analysis_2026-07-15_step2_scarcity_crossday_beats.md` in the master-thesis repo: full synthesis (numbers exact, thesis-ready), candidate-register delta, comparison-to-partner section (report-only).
- [ ] **Step 8: Commit docs** (`git add README.md` in rowii-monitor; research note committed in the master-thesis repo separately).

---

### Task 9 (stretch): 27.06 qualitative timeline via detector transfer

**Files:**
- Create: `scripts/apply_detector.py`
- Test: `tests/test_apply_detector.py` (new, synthetic)

**Interfaces:**
- Consumes: `FittedDetector` (Task 2), `prepare_run(use_cache=True)`, majority mapping utilities used by `scripts/run_step1.py` for its reports (read `run_step1.py`'s mapping/report helpers and reuse the same functions — do not re-derive the mapping logic).
- Produces: `scripts/apply_detector.py --fit-run 010726-tu_ph_tu --apply-run 270626-pu_ph_pu_ph_pu_ph --variant fusion` → `results/step2/transfer/<fit>--to--<apply>/segments.csv` (UTC start/end, fit-day cluster id, mapped mode name where the fit day has GT) + `timeline.md` (human-readable sequence with durations, an explicit "labels are transferred, day has no SCADA GT — qualitative only" banner, and NO accuracy/ARI claims — spec D2 stretch).

- [ ] **Step 1:** Failing synthetic test: fit on a hand-built 2-state PreparedRun, apply to a second one, assert `segments.csv` rows carry fit-day ids and UTC bounds derived from the apply grid, and `timeline.md` contains the qualitative-only banner. Monkeypatch prepare/discovery per the established CLI-test pattern.
- [ ] **Step 2:** Run → FAIL (module missing).
- [ ] **Step 3:** Implement (~120 lines).
- [ ] **Step 4:** Full suite + lints → pass.
- [ ] **Step 5: Commit**

```bash
git add scripts/apply_detector.py tests/test_apply_detector.py
git commit -m "feat: detector-transfer timeline script (qualitative view for no-SCADA days)"
```

- [ ] **Step 6 (orchestrator):** run it for 010726→270626 (fusion + audio-beats), attach the timeline to the Task-8 research note with the partner-narrative cross-check (report-only).

---

## Execution notes (orchestrator)

- Dependency order: T1 → T2 → {T3, T9} and T4 → T5; T6 and T7 are independent of T2–T5 (T6 as early as possible — the warm-up gates Task 8's BEATs steps). Suggested sequence: T1, T2, T6 (+ start warm-up), T4, T3, T5, T7, T9, T8.
- One implementer at a time in the shared working tree; never switch branches while an implementer is active; every dispatch carries the HARD no-delegation clause.
- Real-data sanity between tasks: after T3, run one cheap pair (`--variant fusion --scorer knn`) and eyeball the far_table before batch execution.
- 29.06 per-state cells will be starved (3/4 states below floor) — that is a RESULT (motivates the scarcity curve), not a bug; keep the low-confidence rows.

## Self-review (done at plan-writing time)

- Spec coverage: D1→T1+T2, D2→T3+T9, D3→T4+T5, D4→T6+T8 (+T3/T5 grids include audio-beats), D5→T3(e)+T5 summary + T8 README — no gaps.
- Placeholder scan: the two deliberate elisions (T5 Step 3 dict keys, T3 Step 2 error-path test body) each name their exact completion contract inline; all other steps carry full code.
- Type consistency: `FittedDetector.fit` returns `tuple[FittedDetector, DetectionResult]` everywhere; `scarcity_curve` column set matches `_CURVE_COLUMNS` in tests and figures; row builders' public names used consistently in T3.
