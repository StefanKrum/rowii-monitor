"""Step-2 calibration-scarcity CLI: per-state realized-FAR-vs-calibration-size curves
(package-2 spec D3 primary curve) plus the segment-accumulation deployment view (spec
D3 secondary curve), with figures and a cross-combo `summary.md` headline (plan Task
5, `docs/superpowers/plans/2026-07-15-step2-scarcity-crossday-beats.md`).

Answers the partner's "enough data per mode" question quantitatively, in two units:
how many calibration WINDOWS per state until the realized false-alarm rate is
achievable and stable (`rowii.anomaly.scarcity.scarcity_curve`, the primary curve),
and, with `--secondary`, how many recording MINUTES per state -- a deployment-
realistic view that shrinks the fit/reference side too, not just the calibration
size (`rowii.anomaly.scarcity.segment_accumulation_curve`).

**Per-state score precomputation (the scarcity fast path, `_precompute_state_scores`):**
mirrors `rowii.anomaly.sweep.run_sweep`'s exact three-way split -- top
`split_by_segments(segment_ids, valid_mask, 0.5, seed=7)`, nested
`split_by_segments(segment_ids, calib_mask, 0.5, seed=8)` on the top split's
calibration side (`cfg.seed` then `cfg.seed + 1`, `run_sweep`'s own construction) --
so the leakage-safety guarantee is identical: fit/conformal/scoring windows are
three-way disjoint. Detected labels come from `FittedDetector.fit` on valid-compacted
features, the SAME compaction `scripts/run_step2.py`'s `_detected_labels` (via its
`_detected_labels_and_detector`) uses -- reimplemented locally below
(`_detected_labels`) rather than imported, since a script must not depend on a
SIBLING script's internals (that module's own docstring). Per state with a
`min_ref`-sized fit reference (`rowii.anomaly.references.build_references`, same
default as `SweepConfig.min_ref`):
fit ONE scorer on that state's fit-part reference, score its own conformal-part and
(fixed) scoring-part windows ONCE -- only the calibration SIZE is resampled
downstream, inside `scarcity_curve` itself, which is what makes a 50-repetition sweep
over 8 budgets cheap per state. A state excluded by `min_ref`, or with an empty
conformal or scoring side, never gets a `scarcity_curve` call -- it is recorded in
`summary.md` as "not curvable" WITH its count instead of silently vanishing (spec D5
no-silent-caps, orchestrator resolution 4).

## Secondary curve: always the FIRST run x fusion x knn

`--secondary` computes exactly ONE `segment_accumulation_curve`, for `args.runs[0]` x
`"fusion"` x `"knn"` -- fixed, independent of whatever `--variants`/`--scorers` the
main (primary) sweep was actually given (Task 5 brief). It reuses the SAME
seed-7 top-level split as the primary curve would for that combo, so its `scoring_
windows` (the fixed, never-resampled held-out side, spec D3) are identical whether or
not that exact combo happened to run as part of the primary sweep too.

## Output layout

Per (run, variant, scorer) combo actually swept: `<out>/<run>--<variant>-<scorer>/
curve.csv` (`scarcity_curve` rows for every curvable state, concatenated) +
`curve_by_state.png` (one panel per curvable state -- see `_plot_curve_by_state`).
With `--secondary`, the FIXED combo above additionally gets `segment_curve.csv` +
`segment_curve.png` in that SAME `<out>/<run>--fusion-knn/` directory (created fresh
if that exact combo was not otherwise part of the primary sweep). One shared
`<out>/summary.md` covers every combo actually run this invocation: per state, the
smallest checkpoint whose mean realized FAR lands in `[alpha/2, 2*alpha]` and whose
95th-percentile realized FAR is also `<= 2*alpha` (the "windows/minutes per mode
needed" headline, `_stabilization_table`) -- a state that never clears both
conditions within the tested checkpoints is reported `stabilized=False` citing its
largest tested checkpoint, never silently omitted. `summary.md` also carries the
S-package honesty notes and the scoring-side-sampling-noise caveat verbatim
(`_HONESTY_NOTES`, spec D5 + spec D3).

A combo whose own `prepare_run` raises `RuntimeError` (run too short/sparse for the
requested variant) is logged and skipped, mirroring `scripts/run_step2.py`'s and
`scripts/warm_cache.py`'s identical "one bad combo must not crash the whole
invocation" principle.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import; headless-safe backend.

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import itertools  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.anomaly.references import build_references, split_by_segments  # noqa: E402
from rowii.anomaly.scarcity import (  # noqa: E402
    _CURVE_COLUMNS,
    ScarcityConfig,
    SegmentAccumulationConfig,
    beta_band,
    scarcity_curve,
    segment_accumulation_curve,
)
from rowii.anomaly.scorers import KnnScorer, MahalanobisScorer, Scorer  # noqa: E402
from rowii.config import Config, load_config  # noqa: E402
from rowii.io.dataset import RecordingIndex, Run, discover  # noqa: E402
from rowii.pipeline import (  # noqa: E402
    _BEATS_INSTALL_HINT,
    _STUDENT_INSTALL_HINT,
    _TFC_INSTALL_HINT,
    PreparedRun,
    _is_beats_variant,
    _is_student_variant,
    _is_tfc_variant,
    prepare_run,
)
from rowii.signals.windows import WindowGrid  # noqa: E402
from rowii.state.detect import FittedDetector  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RUNS: tuple[str, ...] = ("010726-tu_ph_tu", "290626-tu")
_DEFAULT_VARIANTS: tuple[str, ...] = ("fusion", "audio-beats")
_VARIANT_CHOICES: tuple[str, ...] = (
    "audio", "vibration", "fusion", "audio-beats", "fusion-beats",
    "audio-tfc", "vibration-tfc", "audio-student", "logmel",
)
"""Duplicated from `scripts/run_step2.py`'s own `_VARIANT_CHOICES` (and `scripts/
warm_cache.py`'s) -- scripts must not import from a sibling script, see this
module's own docstring."""
_SCORER_CHOICES: tuple[str, ...] = ("knn", "mahalanobis", "all")
_CONCRETE_SCORERS: tuple[str, ...] = ("knn", "mahalanobis")

_TOP_SEED = 7
"""Top-level calibration/scoring split seed -- `SweepConfig.seed`'s own default,
`rowii.anomaly.sweep.run_sweep`'s exact construction (module docstring)."""
_NESTED_SEED = 8
"""Nested fit/conformal split seed -- `_TOP_SEED + 1`, matching `run_sweep`'s own
`cfg.seed + 1` for the SAME nested split."""
_MIN_REF = 20
"""Minimum fit-part windows a state needs for a real reference -- matches `rowii.
anomaly.sweep.SweepConfig.min_ref`'s and `rowii.anomaly.references.build_references`'s
own default, so a state curvable here would also get a real per-state reference in a
`run_sweep` call over the same split."""

_SECONDARY_VARIANT = "fusion"
_SECONDARY_SCORER = "knn"
"""The secondary (segment-accumulation) curve is always computed for `args.runs[0]` x
`_SECONDARY_VARIANT` x `_SECONDARY_SCORER` -- fixed by the Task 5 brief, independent
of whatever `--variants`/`--scorers` the primary sweep was given."""

_ALPHA_FLOOR_N = 19
"""`1/0.05 - 1` -- the alpha=0.05 conformal achievability floor (`rowii.anomaly.
conformal.threshold_index`); always drawn as a fixed reference marker on the primary
figure regardless of the run's own `--alpha` (orchestrator resolution 5's literal
"vertical line at n = 19 labeled 'alpha=0.05 floor'")."""

_INVALID_LABEL = -1
"""Sentinel for invalid windows in the full-length detected-label array -- mirrors
`scripts/run_step2.py`'s own `_INVALID_LABEL` (not imported, see module docstring)."""


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibration-scarcity curves for Step-2 per-state conformal thresholds: "
            "how realized FAR (and its spread across repetitions) behaves as the "
            "per-state calibration set shrinks, at fixed calibration WINDOW-COUNT "
            "budgets (primary curve) and, with --secondary, at accumulated recording "
            "SEGMENTS/minutes (secondary, deployment view) -- package-2 spec D3."
        )
    )
    parser.add_argument(
        "--runs", nargs="+", default=list(_DEFAULT_RUNS), metavar="RUN",
        help=f"Run name(s) to sweep (default: {' '.join(_DEFAULT_RUNS)}).",
    )
    parser.add_argument(
        "--variants", nargs="+", default=list(_DEFAULT_VARIANTS), choices=_VARIANT_CHOICES,
        metavar="VARIANT",
        help=f"Variant(s) to sweep (default: {' '.join(_DEFAULT_VARIANTS)}).",
    )
    parser.add_argument(
        "--scorers", choices=_SCORER_CHOICES, default="knn",
        help="'knn' (default), 'mahalanobis', or 'all' (both).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Nominal false-alarm rate.")
    parser.add_argument(
        "--reps", type=int, default=50,
        help="Repetitions per budget for the primary curve (default: 50).",
    )
    parser.add_argument(
        "--secondary-reps", type=int, default=20,
        help="Repetitions per checkpoint for the secondary curve (default: 20).",
    )
    parser.add_argument(
        "--secondary", action="store_true",
        help=(
            "Also compute the segment-accumulation (deployment-minutes) curve for "
            "the FIRST --runs entry x fusion x knn only (module docstring) -- "
            "independent of whatever --variants/--scorers were requested."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/step2/scarcity"),
        help="Output root (default: results/step2/scarcity).",
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Override Config.data_root from load_config() (env ROWII_DATA_ROOT).",
    )
    parser.add_argument(
        "--results-root", type=Path, default=None,
        help=(
            "Override Config.results_root from load_config() (env "
            "ROWII_RESULTS_ROOT) -- only affects rowii.pipeline.prepare_run's "
            "feature-cache location, NOT this script's own --out."
        ),
    )
    return parser


def _resolve_scorers(choice: str) -> tuple[str, ...]:
    return _CONCRETE_SCORERS if choice == "all" else (choice,)


def _make_scorer(name: str) -> Scorer:
    """Mirrors `scripts/run_step2.py`'s own private `_make_scorer` (duplicated, not
    imported -- see module docstring)."""
    if name == "knn":
        return KnnScorer()
    if name == "mahalanobis":
        return MahalanobisScorer()
    raise ValueError(f"scorer must be 'knn' or 'mahalanobis', got {name!r}")


def _import_beats_or_exit() -> None:
    """Mirrors `scripts/run_step2.py`'s/`scripts/warm_cache.py`'s own private helper
    of the same name (duplicated, not imported -- see module docstring)."""
    try:
        import rowii.signals.beats  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"BEATs featurizer not available ({exc}); {_BEATS_INSTALL_HINT}"
        ) from exc


def _import_tfc_or_exit(cfg: Config, variant: str) -> None:
    """Mirrors `scripts/run_step1.py`'s own private helper of the same name
    (duplicated, not imported -- see module docstring). Extends the
    beats-import-guard pattern (package-4 spec D4): torch missing (checked first)
    -> SystemExit naming the shared `[beats]` extra; else the ONE checkpoint
    relevant to *variant* itself missing -> SystemExit naming its own env var."""
    try:
        import rowii.tfc.wrapper  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"TF-C featurizer not available ({exc}); {_TFC_INSTALL_HINT}"
        ) from exc
    if variant == "audio-tfc":
        checkpoint, env_var = cfg.tfc_audio_checkpoint, "ROWII_TFC_AUDIO_CHECKPOINT"
    else:
        checkpoint, env_var = cfg.tfc_vib_checkpoint, "ROWII_TFC_VIB_CHECKPOINT"
    if checkpoint is None:
        raise SystemExit(f"variant {variant!r} needs {env_var} set; {_TFC_INSTALL_HINT}")


def _import_student_or_exit(cfg: Config) -> None:
    """Mirrors `_import_tfc_or_exit` above (package-5 spec D5), simplified: the
    distilled student has only ONE checkpoint (unlike TF-C's two independent
    branches), so there is no variant-based checkpoint selection -- torch
    missing (checked first) -> SystemExit naming the shared `[beats]` extra;
    else `cfg.student_checkpoint` missing -> SystemExit naming
    ROWII_STUDENT_CHECKPOINT. Duplicated (not imported) -- see module
    docstring."""
    try:
        import rowii.adapt.student  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"Student featurizer not available ({exc}); {_STUDENT_INSTALL_HINT}"
        ) from exc
    if cfg.student_checkpoint is None:
        raise SystemExit(
            f"variant 'audio-student' needs ROWII_STUDENT_CHECKPOINT set; "
            f"{_STUDENT_INSTALL_HINT}"
        )


# ---------------------------------------------------------------------------
# Detected labels (mirrors _detected_labels_and_detector in scripts/run_step2.py)
# ---------------------------------------------------------------------------


def _detected_labels(prepared: PreparedRun, cfg: Config) -> np.ndarray:
    """Full-length `(W,)` int64 detected cluster-id labels, `_INVALID_LABEL` on
    invalid windows -- reimplements the exact valid-window compaction of
    `_detected_labels`/`_detected_labels_and_detector` in `scripts/run_step2.py`
    (duplicated, not imported, per this module's own "no sibling-script dependency"
    rule). Keep this in sync with that pair if either side changes.
    """
    valid_mask = prepared.valid_mask
    features_valid = prepared.features[valid_mask]
    n_valid = int(valid_mask.sum())
    valid_grid = WindowGrid(
        t0_ns=prepared.grid.t0_ns, window_ns=prepared.grid.window_ns, n_windows=n_valid
    )
    _detector, det = FittedDetector.fit(
        features_valid, valid_grid, cfg.detect, clusterer="kmeans"
    )
    full_labels = np.full(prepared.features.shape[0], _INVALID_LABEL, dtype=np.int64)
    full_labels[valid_mask] = det.frame_labels
    return full_labels


# ---------------------------------------------------------------------------
# Per-state score precomputation (module docstring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NotCurvable:
    """One state that could not be curved for one combo -- always reported in
    `summary.md` (spec D5 no-silent-caps), never just dropped."""

    label: int | str
    reason: str
    count: int


def _precompute_state_scores(
    prepared: PreparedRun, labels: np.ndarray, scorer_name: str,
) -> tuple[dict[int | str, tuple[np.ndarray, np.ndarray]], list[_NotCurvable]]:
    """Per-state `(conformal_scores, scoring_scores)` pairs -- the scarcity fast path
    (module docstring): mirrors `rowii.anomaly.sweep.run_sweep`'s exact three-way
    split so the leakage-safety guarantee is identical, then fits ONE scorer per
    state on that state's fit-part reference and scores its own conformal-part and
    scoring-part windows ONCE.

    Returns:
        `(scores_by_label, not_curvable)`: *scores_by_label* maps every state with a
        real `min_ref`-sized reference AND non-empty conformal/scoring sides to its
        own precomputed score pair; *not_curvable* lists every OTHER state seen
        anywhere in the top split's calibration/scoring windows, with the reason and
        the count that made it uncurvable.
    """
    top_split = split_by_segments(prepared.segment_ids, prepared.valid_mask, 0.5, _TOP_SEED)
    calibration_windows = top_split.calibration_windows
    scoring_windows = top_split.scoring_windows

    calib_mask = np.zeros(prepared.features.shape[0], dtype=bool)
    calib_mask[calibration_windows] = True
    nested_split = split_by_segments(prepared.segment_ids, calib_mask, 0.5, _NESTED_SEED)
    fit_windows = nested_split.calibration_windows
    conformal_windows = nested_split.scoring_windows

    refs = build_references(prepared.features, labels, fit_windows, min_ref=_MIN_REF)

    all_windows = np.concatenate([calibration_windows, scoring_windows])
    all_labels = sorted(np.unique(labels[all_windows]).tolist())

    scores_by_label: dict[int | str, tuple[np.ndarray, np.ndarray]] = {}
    not_curvable: list[_NotCurvable] = []
    for label in all_labels:
        if label not in refs.references:
            not_curvable.append(
                _NotCurvable(
                    label, f"excluded by min_ref={_MIN_REF} (fit-side windows)",
                    refs.excluded.get(label, 0),
                )
            )
            continue
        label_conformal = conformal_windows[labels[conformal_windows] == label]
        label_scoring = scoring_windows[labels[scoring_windows] == label]
        if label_conformal.shape[0] == 0 or label_scoring.shape[0] == 0:
            not_curvable.append(
                _NotCurvable(
                    label, "empty conformal-side or scoring-side windows",
                    min(int(label_conformal.shape[0]), int(label_scoring.shape[0])),
                )
            )
            continue
        scorer = _make_scorer(scorer_name).fit(refs.references[label])
        conformal_scores = scorer.score(prepared.features[label_conformal])
        scoring_scores = scorer.score(prepared.features[label_scoring])
        scores_by_label[label] = (conformal_scores, scoring_scores)
    return scores_by_label, not_curvable


# ---------------------------------------------------------------------------
# Primary combo: curve.csv + curve_by_state.png
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ComboResult:
    """One (run, variant, scorer) primary-curve combo's output, kept in memory for
    `summary.md` after `curve.csv`/`curve_by_state.png` are written."""

    run: str
    variant: str
    scorer: str
    curve: pd.DataFrame
    not_curvable: list[_NotCurvable]


def _combo_dir(out_root: Path, run_name: str, variant: str, scorer_name: str) -> Path:
    return out_root / f"{run_name}--{variant}-{scorer_name}"


def _run_primary_combo(
    out_root: Path, run_name: str, variant: str, scorer_name: str,
    prepared: PreparedRun, rowii_cfg: Config, alpha: float, n_reps: int,
) -> _ComboResult:
    """One (run, variant, scorer) primary curve: precompute per-state scores once
    (`_precompute_state_scores`), sweep every curvable state's calibration size
    (`scarcity_curve`), and write `curve.csv` + `curve_by_state.png`."""
    labels = _detected_labels(prepared, rowii_cfg)
    scores_by_label, not_curvable = _precompute_state_scores(prepared, labels, scorer_name)

    scarcity_cfg = ScarcityConfig(n_reps=n_reps, alpha=alpha)
    curves = [
        scarcity_curve(conformal, scoring, label, scarcity_cfg)
        for label, (conformal, scoring) in scores_by_label.items()
    ]
    curve = (
        pd.concat(curves, ignore_index=True) if curves
        else pd.DataFrame(columns=list(_CURVE_COLUMNS))
    )

    combo_dir = _combo_dir(out_root, run_name, variant, scorer_name)
    combo_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(combo_dir / "curve.csv", index=False)
    _plot_curve_by_state(combo_dir / "curve_by_state.png", curve, alpha)

    return _ComboResult(
        run=run_name, variant=variant, scorer=scorer_name, curve=curve,
        not_curvable=not_curvable,
    )


# ---------------------------------------------------------------------------
# Secondary combo: segment_curve.csv + segment_curve.png (module docstring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SecondaryResult:
    run: str
    curve: pd.DataFrame
    not_curvable: list[_NotCurvable]


def _run_secondary_combo(
    out_root: Path, run_name: str, prepared: PreparedRun, rowii_cfg: Config,
    alpha: float, n_reps: int,
) -> _SecondaryResult:
    """The ONE segment-accumulation curve `--secondary` asks for -- always
    `run_name` x `_SECONDARY_VARIANT` x `_SECONDARY_SCORER` (module docstring).
    Reuses the SAME seed-7 top-level split as a primary combo on this run would, so
    `scoring_windows` (the FIXED, never-resampled scoring side, spec D3) is
    identical either way; `segment_accumulation_curve` then only ever draws its
    growing fit/conformal prefixes from the calibration side's own segments.
    """
    labels = _detected_labels(prepared, rowii_cfg)
    top_split = split_by_segments(prepared.segment_ids, prepared.valid_mask, 0.5, _TOP_SEED)
    calibration_windows = top_split.calibration_windows
    scoring_windows = top_split.scoring_windows

    all_windows = np.concatenate([calibration_windows, scoring_windows])
    all_labels = sorted(np.unique(labels[all_windows]).tolist())
    scoring_labels = set(np.unique(labels[scoring_windows]).tolist())
    not_curvable = [
        _NotCurvable(
            label, "never appears among the fixed scoring-side windows",
            int((labels[calibration_windows] == label).sum()),
        )
        for label in all_labels if label not in scoring_labels
    ]

    seg_cfg = SegmentAccumulationConfig(n_reps=n_reps, alpha=alpha, min_ref=_MIN_REF)
    curve = segment_accumulation_curve(
        prepared.features, labels, prepared.segment_ids, prepared.valid_mask,
        lambda: _make_scorer(_SECONDARY_SCORER), scoring_windows, seg_cfg,
    )

    combo_dir = _combo_dir(out_root, run_name, _SECONDARY_VARIANT, _SECONDARY_SCORER)
    combo_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(combo_dir / "segment_curve.csv", index=False)
    _plot_segment_curve(combo_dir / "segment_curve.png", curve, alpha)

    return _SecondaryResult(run=run_name, curve=curve, not_curvable=not_curvable)


# ---------------------------------------------------------------------------
# Figures (orchestrator resolution 5)
# ---------------------------------------------------------------------------


def _empty_panel(ax: Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _plot_one_curve_panel(
    ax: Axes, state_curve: pd.DataFrame, alpha: float, *, x_col: str, title: str, log_x: bool,
) -> None:
    """Primary-figure per-state panel: x = *x_col* (`achieved_n`), translucent
    per-rep points + mean line + empirical 5/95 band, exact `beta_band` overlay at
    each distinct *x_col* value, horizontal alpha line, vertical `n = _ALPHA_FLOOR_N`
    marker (orchestrator resolution 5)."""
    x_raw = state_curve[x_col].to_numpy(dtype=float)
    far_raw = state_curve["realized_far"].to_numpy(dtype=float)
    ax.scatter(x_raw, far_raw, alpha=0.15, s=10, color="tab:blue", label="per-rep FAR")

    grouped = state_curve.groupby(x_col)["realized_far"]
    means = grouped.mean().sort_index()
    lo = grouped.quantile(0.05).sort_index()
    hi = grouped.quantile(0.95).sort_index()
    xs = means.index.to_numpy(dtype=float)

    ax.plot(xs, means.to_numpy(), color="tab:blue", marker="o", markersize=3,
            label="mean realized FAR")
    ax.fill_between(
        xs, lo.to_numpy(), hi.to_numpy(), color="tab:blue", alpha=0.15,
        label="empirical 5/95 band",
    )

    beta_lo = np.full(xs.shape, np.nan)
    beta_hi = np.full(xs.shape, np.nan)
    for i, n in enumerate(xs):
        band = beta_band(int(round(n)), alpha)
        if band is not None:
            beta_lo[i], beta_hi[i] = band
    ax.plot(xs, beta_lo, "--", color="tab:orange", linewidth=1, label="exact Beta 5/95 band")
    ax.plot(xs, beta_hi, "--", color="tab:orange", linewidth=1)

    ax.axhline(alpha, color="black", linewidth=1, linestyle=":", label=f"alpha={alpha:g}")
    ax.axvline(_ALPHA_FLOOR_N, color="gray", linewidth=1, linestyle="-.")
    ax.text(
        _ALPHA_FLOOR_N, ax.get_ylim()[1], " α=0.05 floor", rotation=90, fontsize="small",
        va="top", ha="left", color="0.3",
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_col)
    ax.set_ylabel("realized FAR")
    ax.set_title(title)
    ax.legend(fontsize="small", loc="upper right")


def _plot_curve_by_state(out_path: Path, curve: pd.DataFrame, alpha: float) -> None:
    """One panel per curvable state (module docstring / orchestrator resolution 5).
    Always writes a non-empty PNG, even when zero states are curvable."""
    labels = sorted(curve["label"].unique().tolist()) if len(curve) else []
    n_panels = max(len(labels), 1)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3.2 * n_panels), squeeze=False)

    if not labels:
        _empty_panel(axes[0][0], "no curvable states")
    else:
        for row, label in zip(axes, labels, strict=True):
            _plot_one_curve_panel(
                row[0], curve[curve["label"] == label], alpha,
                x_col="achieved_n", title=f"label {label}", log_x=True,
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_one_segment_panel(
    ax: Axes, state_curve: pd.DataFrame, alpha: float, label: object,
) -> None:
    """Secondary-figure per-state panel: x = minutes, same y-axis layers as
    `_plot_one_curve_panel` (translucent points, mean line, empirical 5/95 band,
    exact `beta_band` overlay, horizontal alpha line) -- grouped by `n_segments`
    (the deterministic per-repetition checkpoint) rather than by the continuous
    `minutes` column itself, since two repetitions reaching the SAME segment count
    can draw a slightly different total window count in general; the `beta_band`
    overlay uses each checkpoint's own MEAN achieved `n_conformal` (the actual
    calibration size a threshold was computed on at that checkpoint).
    """
    ax.scatter(
        state_curve["minutes"].to_numpy(dtype=float),
        state_curve["realized_far"].to_numpy(dtype=float),
        alpha=0.15, s=10, color="tab:green", label="per-rep FAR",
    )

    by_checkpoint = state_curve.groupby("n_segments")
    minutes_mean = by_checkpoint["minutes"].mean().sort_index()
    far_mean = by_checkpoint["realized_far"].mean().sort_index()
    far_lo = by_checkpoint["realized_far"].quantile(0.05).sort_index()
    far_hi = by_checkpoint["realized_far"].quantile(0.95).sort_index()
    n_conformal_mean = by_checkpoint["n_conformal"].mean().sort_index()

    xs = minutes_mean.to_numpy()
    ax.plot(xs, far_mean.to_numpy(), color="tab:green", marker="o", markersize=3,
            label="mean realized FAR")
    ax.fill_between(
        xs, far_lo.to_numpy(), far_hi.to_numpy(), color="tab:green", alpha=0.15,
        label="empirical 5/95 band",
    )

    beta_lo = np.full(xs.shape, np.nan)
    beta_hi = np.full(xs.shape, np.nan)
    for i, n in enumerate(n_conformal_mean.to_numpy()):
        band = beta_band(int(round(n)), alpha)
        if band is not None:
            beta_lo[i], beta_hi[i] = band
    ax.plot(
        xs, beta_lo, "--", color="tab:orange", linewidth=1,
        label="exact Beta 5/95 band (mean achieved n_conformal)",
    )
    ax.plot(xs, beta_hi, "--", color="tab:orange", linewidth=1)

    ax.axhline(alpha, color="black", linewidth=1, linestyle=":", label=f"alpha={alpha:g}")
    ax.set_xlabel("minutes")
    ax.set_ylabel("realized FAR")
    ax.set_title(f"label {label}")
    ax.legend(fontsize="small", loc="upper right")


def _plot_segment_curve(out_path: Path, curve: pd.DataFrame, alpha: float) -> None:
    """Secondary figure: x = minutes (module docstring / orchestrator resolution 5).
    Always writes a non-empty PNG, even when zero states are curvable."""
    labels = sorted(curve["label"].unique().tolist()) if len(curve) else []
    n_panels = max(len(labels), 1)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3.2 * n_panels), squeeze=False)

    if not labels:
        _empty_panel(axes[0][0], "no curvable states")
    else:
        for row, label in zip(axes, labels, strict=True):
            _plot_one_segment_panel(row[0], curve[curve["label"] == label], alpha, label)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# summary.md (orchestrator resolution 4: no state ever silently dropped)
# ---------------------------------------------------------------------------

_HONESTY_NOTES = """\
**Honesty notes (package-2 spec D5, carried over from the S-package):**

- Per-state conditioning here uses Step-1 DETECTED cluster labels, never ground
  truth -- any Step-1 detection error at a state boundary is inherited by this
  curve unchanged.
- A row with `low_confidence=True` never had a real calibrated threshold
  (`threshold=+inf` semantics): its `realized_far=0.0` means "never alarms", not
  "well controlled" -- do not read it as a good result.
- No value from the partner's own repo/analyses is adopted anywhere in this
  computation; any partner comparison is report-only, elsewhere.

**Scoring-side sampling noise (spec D3):** the empirical 5/95 band and the exact
Beta band both describe CALIBRATION-side variability only (repeating the draw at a
fixed calibration size). The SCORING split itself is one fixed sample per combo
(never resampled across repetitions, by design -- spec D3), so its own sampling
noise is not represented by either band; a narrow band at a given budget does not
mean the true FAR is known to that precision, only that resampling the calibration
set alone would not move the estimate much.
"""


def _not_curvable_section(not_curvable: list[_NotCurvable]) -> list[str]:
    lines = ["### Not curvable states", ""]
    if not not_curvable:
        lines.append("(none -- every detected state was curvable.)")
        lines.append("")
        return lines
    lines.append("| label | reason | count |")
    lines.append("|---|---|---|")
    for item in sorted(not_curvable, key=lambda nc: str(nc.label)):
        lines.append(f"| {item.label} | {item.reason} | {item.count} |")
    lines.append("")
    return lines


def _stabilization_table(
    curve: pd.DataFrame, alpha: float, x_col: str, extra_col: str,
) -> pd.DataFrame:
    """"Windows/minutes per mode needed" headline: for each label in *curve*, the
    smallest *x_col* checkpoint whose mean realized FAR across reps lands in
    `[alpha/2, 2*alpha]` AND whose 95th-percentile realized FAR (same reps) is also
    `<= 2*alpha` -- both conditions at the SAME checkpoint. A label with no such
    checkpoint among those actually tested is reported `stabilized=False`, citing
    the LARGEST tested checkpoint's own mean/95th-pct instead of silently omitting
    the state (spec D5 no-silent-caps).

    Args:
        curve: One combo's `curve`/`segment_curve` DataFrame (columns must include
            `label`, `realized_far`, *x_col*, *extra_col*).
        alpha: Nominal false-alarm rate the curve was generated with.
        x_col: The checkpoint column to search over (`"budget"` for the primary
            curve, `"n_segments"` for the secondary one).
        extra_col: A second, representative column to report alongside the chosen
            checkpoint (`"achieved_n"` for the primary curve, `"minutes"` for the
            secondary one) -- its MEAN at the chosen checkpoint.

    Returns:
        DataFrame with columns `label, stabilized, {x_col}, {extra_col}, mean_far,
        q95_far`, one row per distinct label in *curve*, sorted by label.
    """
    lower, upper = alpha / 2.0, 2.0 * alpha
    rows: list[dict[str, object]] = []
    for label in sorted(curve["label"].unique().tolist()):
        sub = curve[curve["label"] == label]
        grouped = sub.groupby(x_col)["realized_far"]
        means = grouped.mean().sort_index()
        q95 = grouped.quantile(0.95).sort_index()
        extra_means = sub.groupby(x_col)[extra_col].mean().sort_index()

        ok = means.index[(means >= lower) & (means <= upper) & (q95 <= upper)]
        stabilized = len(ok) > 0
        chosen = ok.min() if stabilized else means.index.max()
        rows.append({
            "label": label, "stabilized": stabilized, x_col: chosen,
            extra_col: float(extra_means.loc[chosen]),
            "mean_far": float(means.loc[chosen]), "q95_far": float(q95.loc[chosen]),
        })
    return pd.DataFrame(rows)


def _stabilization_markdown(table: pd.DataFrame, x_col: str, extra_col: str) -> list[str]:
    lines = [
        f"| label | stabilized | {x_col} | {extra_col} | mean FAR | 95th pct FAR |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        status = "yes" if r["stabilized"] else "NEVER (largest tested checkpoint shown)"
        lines.append(
            f"| {r['label']} | {status} | {r[x_col]:g} | {r[extra_col]:.3g} | "
            f"{r['mean_far']:.4f} | {r['q95_far']:.4f} |"
        )
    lines.append("")
    return lines


def _write_summary_md(
    out_path: Path, combos: list[_ComboResult], secondary: _SecondaryResult | None, alpha: float,
) -> None:
    lines = ["# Step-2 calibration-scarcity summary", "", _HONESTY_NOTES]
    for combo in combos:
        lines.append(f"## {combo.run} / {combo.variant} / {combo.scorer}")
        lines.append("")
        lines.extend(_not_curvable_section(combo.not_curvable))
        lines.append("### Windows-per-mode headline")
        lines.append("")
        if combo.curve.empty:
            lines.append('No curvable state for this combo -- see "not curvable" above.')
            lines.append("")
        else:
            table = _stabilization_table(combo.curve, alpha, "budget", "achieved_n")
            lines.extend(_stabilization_markdown(table, "budget", "achieved_n"))

    if secondary is not None:
        lines.append(
            f"## Secondary (segment accumulation): {secondary.run} / "
            f"{_SECONDARY_VARIANT} / {_SECONDARY_SCORER}"
        )
        lines.append("")
        lines.extend(_not_curvable_section(secondary.not_curvable))
        lines.append("### Minutes-per-mode headline")
        lines.append("")
        if secondary.curve.empty:
            lines.append(
                'No curvable state for the secondary curve -- see "not curvable" above.'
            )
            lines.append("")
        else:
            table = _stabilization_table(secondary.curve, alpha, "n_segments", "minutes")
            lines.extend(_stabilization_markdown(table, "n_segments", "minutes"))

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.data_root is not None:
        cfg = dataclasses.replace(cfg, data_root=args.data_root)
    if args.results_root is not None:
        cfg = dataclasses.replace(cfg, results_root=args.results_root)

    index: RecordingIndex = discover(cfg.data_root)
    by_name: dict[str, Run] = {r.name: r for r in index.runs}

    unknown = [name for name in args.runs if name not in by_name]
    if unknown:
        available = ", ".join(sorted(by_name)) or "(none discovered)"
        print(
            f"run_step2_scarcity: unknown run name(s): {', '.join(unknown)}; "
            f"available runs: {available}",
            file=sys.stderr,
        )
        return 2

    scorers = _resolve_scorers(args.scorers)
    out_root: Path = args.out
    prepared_cache: dict[tuple[str, str], PreparedRun] = {}

    def _prepared_for(run_name: str, variant: str) -> PreparedRun | None:
        key = (run_name, variant)
        if key in prepared_cache:
            return prepared_cache[key]
        if _is_beats_variant(variant):
            _import_beats_or_exit()
        if _is_tfc_variant(variant):
            _import_tfc_or_exit(cfg, variant)
        if _is_student_variant(variant):
            _import_student_or_exit(cfg)
        try:
            prepared = prepare_run(by_name[run_name], variant, cfg, use_cache=True)
        except RuntimeError as exc:
            logger.warning(
                "run_step2_scarcity: prepare_run failed for %s/%s (%s) -- run is too "
                "short/sparse for this variant, skipping",
                run_name, variant, exc,
            )
            return None
        prepared_cache[key] = prepared
        return prepared

    combos: list[_ComboResult] = []
    for run_name, variant, scorer_name in itertools.product(args.runs, args.variants, scorers):
        prepared = _prepared_for(run_name, variant)
        if prepared is None:
            continue
        combos.append(
            _run_primary_combo(
                out_root, run_name, variant, scorer_name, prepared, cfg, args.alpha, args.reps
            )
        )

    secondary: _SecondaryResult | None = None
    if args.secondary:
        first_run = args.runs[0]
        prepared = _prepared_for(first_run, _SECONDARY_VARIANT)
        if prepared is None:
            logger.warning(
                "run_step2_scarcity: --secondary requested but %s/%s could not be "
                "prepared -- secondary curve skipped",
                first_run, _SECONDARY_VARIANT,
            )
        else:
            secondary = _run_secondary_combo(
                out_root, first_run, prepared, cfg, args.alpha, args.secondary_reps
            )

    out_root.mkdir(parents=True, exist_ok=True)
    _write_summary_md(out_root / "summary.md", combos, secondary, args.alpha)

    print(
        f"run_step2_scarcity: wrote {len(combos)} primary combo(s)"
        + (" + secondary curve" if secondary is not None else "")
        + f" to {out_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
