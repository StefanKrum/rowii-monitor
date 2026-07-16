"""Detection-performance scarcity harness on the labeled MIMII proxy (package-6
pillar-3, design spec `docs/superpowers/specs/2026-07-16-step2-package6-runtime-
pillar3-design.md` D4 + amendment A1.4/A1.5, plan Task 5) -- the design chapter's
central figure (detection performance vs. fraction of target-normal training
data, per representation), run NOW on a public proxy because PSHP fault labels
do not exist yet: MIMII ships `abnormal/` clips, PSHP awaits the induced-fault
campaign. Every output therefore restates the honesty framing (spec section 4):
these are PUBLIC-PROXY results in the machine-id domain, never PSHP evidence.

Protocol per (representation x machine id), all splits CLIP-level (windows of
one clip stay together -- the leakage rule):

1. Features are extracted ONCE over all clips of the machine
   (`rowii.tfc.corpora.iter_labeled_clips_wav_dir`, caps applied and logged by
   the loader itself) by driving the featurizer classes DIRECTLY on
   `(n, S, 1)` float32 windows at 16 kHz, and cached (`_save_cache`'s docstring
   documents the npz members; `allow_pickle=False` on load, this repo's
   standing no-pickle rule).
2. NORMAL clips split 30% TEST / 70% TRAIN-POOL via ONE `seed=7` draw per
   machine, shared by every (fraction, seed) cell (`_test_pool_split`).
3. Per (fraction, seed) cell: draw `ceil(fraction * n_pool)` pool clips (rng =
   the cell's seed), split the draw 80/20 into FIT / CAL clips;
   `KnnScorer(k=1, cosine)` fit on the FIT clips' windows; clip score = MEAN
   over the clip's window scores (MIMII's standard clip-level evaluation);
   conformal threshold = `calibrate(cal_clip_scores, alpha)` on CLIP-LEVEL
   calibration scores -- amendment A1.4's coherence rule: never
   window-calibrated/clip-applied. Evaluated on TEST-normal clips + ALL
   abnormal clips (abnormal clips never touch fitting or calibration).
4. Metrics per cell: `auc_clip`; `pauc_clip` = `sklearn.metrics.roc_auc_score(
   ..., max_fpr=0.1)`, the STANDARDIZED (McClish 1989) partial AUC (definition
   named in the md and in a CSV header comment line -- amendment A1.4, pinned
   by a hand-computed test case); `tpr_at_alpha` (abnormal clips above the
   conformal threshold); `realized_normal_clip_far`; `auc_window` (secondary,
   window-level over the same test/abnormal windows). A draw whose fit or cal
   side is empty yields NaN metrics + `degenerate=True` + a log line -- never
   a crash.

Standardization caveat (amendment A1.5, restated in every output): the loader
per-window standardizes every window for EVERY representation -- removes
clip-gain confounds but also erases absolute-level anomaly cues (the
conservative choice). `AudioFeaturizer`'s output width varies with `rate_hz`
(bands capped at Nyquist) -- self-consistent within this 16 kHz corpus. The
student on MIMII measures TRANSFERABILITY of a PSHP-distilled encoder.

Availability: `handcrafted`/`logmel` always run; `beats`/`tfc`/`student` are
SKIPPED with a log line when their checkpoint env is unset or the file is
missing (`scripts/benchmark_inference.py`'s graceful-skip semantics -- never an
error), and their torch-dependent classes are imported lazily inside
`_build_featurizer` only on a cache MISS, so the numpy representations (and
every cache-hit rerun) work without torch installed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.anomaly.conformal import calibrate  # noqa: E402
from rowii.anomaly.scorers import KnnScorer  # noqa: E402
from rowii.config import Config, load_config  # noqa: E402
from rowii.tfc.corpora import iter_labeled_clips_wav_dir  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

_WINDOW_S = 1.0
_TARGET_HZ = 16_000
"""16 kHz corpus rate (amendment A1.5: BEATs-native; every featurizer resamples
from its `rate_hz` argument anyway) -- also the `rate_hz` handed to every
`transform` call, since the loader already resampled the windows."""
_PAUC_MAX_FPR = 0.1
_TEST_FRACTION = 0.3
_TEST_SPLIT_SEED = 7
"""The binding protocol's fixed seed for the ONE shared TEST draw per machine --
deliberately NOT tied to `--seeds`, so every (fraction, seed) cell of a machine
is evaluated against the SAME held-out normal clips."""
_FIT_FRACTION = 0.8

_KNOWN_REPRESENTATIONS = ("handcrafted", "logmel", "beats", "tfc", "student")

_CSV_COLUMNS = (
    "representation", "machine_id", "fraction", "seed",
    "n_fit_clips", "n_cal_clips", "n_test_normal_clips", "n_abnormal_clips",
    "auc_clip", "pauc_clip", "tpr_at_alpha", "realized_normal_clip_far",
    "auc_window", "degenerate",
)

_PAUC_DEFINITION = (
    "pauc_clip = sklearn.metrics.roc_auc_score(labels, clip_scores, max_fpr=0.1): the "
    "STANDARDIZED (McClish 1989) partial AUC over FPR <= 0.1 (0.5 = chance, 1.0 = perfect)."
)


class _Featurizer(Protocol):
    """The one method this harness needs from every representation's featurizer
    (positional-only parameters, so implementations naming the batch `stack`
    instead of `windows` -- `LogmelFeaturizer`/`TfcFeaturizer`/`StudentFeaturizer`
    -- still satisfy it)."""

    def transform(self, windows: np.ndarray, rate_hz: float, /) -> np.ndarray: ...


@dataclass(frozen=True)
class _Representation:
    """One AVAILABLE representation: its name plus the resolved checkpoint
    path(s) (env-var name -> path string; empty for the numpy representations)
    that both key the embedding cache fingerprint and tell `_build_featurizer`
    which constructor branch to take."""

    name: str
    checkpoints: dict[str, str]


def _resolve_representation(name: str, cfg: Config) -> tuple[_Representation | None, str]:
    """Availability gate (spec D4: `benchmark_inference`'s skip-with-log
    semantics): `(representation, "")` when *name* can run on this machine,
    `(None, reason)` when it must be skipped -- checkpoint env unset, or set but
    pointing at a missing file. Never an error. For `beats`, the fp32 checkpoint
    is preferred and the int8 one is a fallback (either alone suffices --
    `BeatsFeaturizer`'s int8 branch never reads the fp32 file)."""
    if name in ("handcrafted", "logmel"):
        return _Representation(name=name, checkpoints={}), ""
    candidates: list[tuple[str, Path | None]] = {
        "beats": [
            ("ROWII_BEATS_CHECKPOINT", cfg.beats_checkpoint),
            ("ROWII_BEATS_INT8_CHECKPOINT", cfg.beats_int8_checkpoint),
        ],
        "tfc": [("ROWII_TFC_AUDIO_CHECKPOINT", cfg.tfc_audio_checkpoint)],
        "student": [("ROWII_STUDENT_CHECKPOINT", cfg.student_checkpoint)],
    }[name]
    configured = [(env, path) for env, path in candidates if path is not None]
    if not configured:
        env_names = " / ".join(env for env, _ in candidates)
        return None, f"its checkpoint env is unset ({env_names})"
    usable = [(env, path) for env, path in configured if path.is_file()]
    if not usable:
        missing = ", ".join(str(path) for _, path in configured)
        return None, f"its checkpoint does not exist ({missing})"
    env, path = usable[0]
    return _Representation(name=name, checkpoints={env: str(path)}), ""


def _build_featurizer(rep: _Representation) -> _Featurizer:
    """The DIRECT featurizer instance for *rep* (spec D4: this harness drives
    the featurizer classes directly on `(n, S, 1)` windows at 16 kHz -- no
    pipeline variant). Torch-dependent classes are imported lazily HERE, never
    at module import (plan global constraint), so `handcrafted`/`logmel` runs
    -- and every all-cache-hit rerun -- need no torch install."""
    if rep.name == "handcrafted":
        from rowii.signals.features import AudioFeaturizer

        return AudioFeaturizer()
    if rep.name == "logmel":
        from rowii.signals.logmel import LogmelFeaturizer

        return LogmelFeaturizer()
    if rep.name == "beats":
        from rowii.signals.beats import BeatsFeaturizer

        if "ROWII_BEATS_INT8_CHECKPOINT" in rep.checkpoints:
            int8_path = Path(rep.checkpoints["ROWII_BEATS_INT8_CHECKPOINT"])
            return BeatsFeaturizer(int8_path, int8_model_path=int8_path)
        return BeatsFeaturizer(Path(rep.checkpoints["ROWII_BEATS_CHECKPOINT"]))
    if rep.name == "tfc":
        from rowii.tfc.wrapper import TfcFeaturizer

        return TfcFeaturizer(Path(rep.checkpoints["ROWII_TFC_AUDIO_CHECKPOINT"]))
    from rowii.adapt.student import StudentFeaturizer

    return StudentFeaturizer(Path(rep.checkpoints["ROWII_STUDENT_CHECKPOINT"]))


# ---------------------------------------------------------------------------
# Embedding cache (one extraction pass per representation x machine id)
# ---------------------------------------------------------------------------


def _clip_manifest(root: Path, machine_id: str, cap: int | None) -> list[tuple[str, int]]:
    """`(relpath, size_bytes)` for every clip `iter_labeled_clips_wav_dir` will
    read for *machine_id* under *cap*, in the iterator's own yield order (sorted
    machine dirs, `abnormal` before `normal`, sorted filenames, per-directory
    cap) -- computed from `stat()` alone so a cache HIT never touches audio."""
    machine_dirs = sorted(
        (d for d in root.rglob("id_*") if d.is_dir() and d.name == machine_id),
        key=lambda p: p.as_posix(),
    )
    entries: list[tuple[str, int]] = []
    for machine_dir in machine_dirs:
        for class_name in ("abnormal", "normal"):
            class_dir = machine_dir / class_name
            if not class_dir.is_dir():
                continue
            paths = sorted(class_dir.glob("*.wav"), key=lambda p: p.as_posix())
            if cap is not None:
                paths = paths[:cap]
            entries.extend((p.relative_to(root).as_posix(), p.stat().st_size) for p in paths)
    return entries


def _fingerprint(rep: _Representation, cap: int | None, manifest: list[tuple[str, int]]) -> str:
    """sha256 over everything that determines the extracted features (spec D4):
    representation + its resolved checkpoint paths, the fixed window geometry,
    the cap, and the sorted clip relpaths + byte sizes."""
    payload = json.dumps(
        {
            "representation": rep.name,
            "checkpoints": rep.checkpoints,
            "window_s": _WINDOW_S,
            "target_hz": _TARGET_HZ,
            "limit_clips_per_class": cap,
            "clips": manifest,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_features(
    root: Path, machine_id: str, cap: int | None, featurizer: _Featurizer
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """One extraction pass over *machine_id*'s clips: `(features (N, F) float64,
    clip_bounds (C+1,) int64, labels (C,) int64, relpaths)` -- clip `i`'s windows
    are `features[clip_bounds[i]:clip_bounds[i+1]]` (every yielded clip has >= 1
    window; the loader drops zero-window clips itself)."""
    blocks: list[np.ndarray] = []
    labels: list[int] = []
    relpaths: list[str] = []
    bounds: list[int] = [0]
    for clip in iter_labeled_clips_wav_dir(
        root,
        window_s=_WINDOW_S,
        target_hz=_TARGET_HZ,
        limit_clips_per_class=cap,
        machine_ids=[machine_id],
    ):
        feats = np.asarray(
            featurizer.transform(clip.windows[:, :, np.newaxis], float(_TARGET_HZ)),
            dtype=np.float64,
        )
        blocks.append(feats)
        bounds.append(bounds[-1] + feats.shape[0])
        labels.append(clip.label)
        relpaths.append(Path(clip.path).relative_to(root).as_posix())
    features = np.vstack(blocks) if blocks else np.empty((0, 0), dtype=np.float64)
    return (
        features,
        np.asarray(bounds, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        relpaths,
    )


def _save_cache(
    path: Path,
    rep: _Representation,
    machine_id: str,
    fingerprint: str,
    cap: int | None,
    features: np.ndarray,
    clip_bounds: np.ndarray,
    labels: np.ndarray,
    relpaths: list[str],
) -> None:
    """npz members (spec D4, pickle-free -- loaded with `allow_pickle=False`):
    `features` (N, F) float64 per-window features stacked in clip order;
    `clip_bounds` (C+1,) int64 boundaries (clip i = rows bounds[i]:bounds[i+1]);
    `labels` (C,) int64 (0 normal / 1 abnormal); `relpaths` (C,) unicode clip
    paths relative to the corpus root; `meta` -- one JSON string (the repo's
    JSON-in-array convention) carrying representation, machine_id, fingerprint,
    checkpoints, window_s, target_hz, limit_clips_per_class."""
    meta = json.dumps(
        {
            "representation": rep.name,
            "machine_id": machine_id,
            "fingerprint": fingerprint,
            "checkpoints": rep.checkpoints,
            "window_s": _WINDOW_S,
            "target_hz": _TARGET_HZ,
            "limit_clips_per_class": cap,
        },
        sort_keys=True,
    )
    np.savez(
        path,
        features=features,
        clip_bounds=clip_bounds,
        labels=labels,
        relpaths=np.asarray(relpaths, dtype=str),
        meta=np.asarray([meta], dtype=str),
    )


def _load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(features, clip_bounds, labels)` from `_save_cache`'s npz -- float64
    round-trips bit-exactly, so cached and fresh runs score identically."""
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["features"], dtype=np.float64),
            np.asarray(data["clip_bounds"], dtype=np.int64),
            np.asarray(data["labels"], dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# Split protocol + per-cell evaluation (spec D4 + amendment A1.4)
# ---------------------------------------------------------------------------


def _test_pool_split(normal_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ONE 30% TEST draw over a machine's normal clip indices, seeded with
    the fixed `_TEST_SPLIT_SEED` (7) and therefore identical for every
    (fraction, seed) cell of the machine -- the binding protocol's shared
    test-set rule. Returns `(test_idx, pool_idx)`, both sorted."""
    if normal_idx.size == 0:
        return normal_idx.copy(), normal_idx.copy()
    rng = np.random.default_rng(_TEST_SPLIT_SEED)
    perm = rng.permutation(normal_idx)
    n_test = min(normal_idx.size, max(1, round(_TEST_FRACTION * normal_idx.size)))
    return np.sort(perm[:n_test]), np.sort(perm[n_test:])


def _clip_scores(
    scorer: KnnScorer, features: np.ndarray, clip_bounds: np.ndarray, clip_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`(per-clip MEAN window score, concatenated per-window scores)` for the
    clips in *clip_idx* -- the clip-level score is the mean over the clip's own
    windows (MIMII's standard clip-level evaluation; binding protocol)."""
    per_clip = [
        scorer.score(features[clip_bounds[i] : clip_bounds[i + 1]]) for i in clip_idx
    ]
    clip_means = np.asarray([float(s.mean()) for s in per_clip], dtype=np.float64)
    window_scores = np.concatenate(per_clip) if per_clip else np.empty(0, dtype=np.float64)
    return clip_means, window_scores


def _standardized_pauc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """`sklearn.metrics.roc_auc_score(..., max_fpr=0.1)` -- the STANDARDIZED
    (McClish 1989) partial AUC over FPR <= 0.1: the raw partial area is rescaled
    so 0.5 = chance and 1.0 = perfect (amendment A1.4: the definition is named
    in every output and pinned by a hand-computed test case)."""
    return float(roc_auc_score(y_true, scores, max_fpr=_PAUC_MAX_FPR))


def _evaluate_cell(
    representation: str,
    machine_id: str,
    features: np.ndarray,
    clip_bounds: np.ndarray,
    test_normal: np.ndarray,
    pool: np.ndarray,
    abnormal_idx: np.ndarray,
    fraction: float,
    seed: int,
    alpha: float,
) -> dict[str, object]:
    """One CSV row (spec D4 + amendment A1.4 -- see module docstring for the
    full protocol). Degenerate draws (empty fit or cal side; also a machine
    with no test-normal or no abnormal clips, where no metric is defined)
    yield NaN metrics + `degenerate=True` + a log line, never a crash."""
    n_draw = math.ceil(fraction * pool.size)
    rng = np.random.default_rng(seed)
    draw = rng.permutation(pool)[:n_draw]
    n_fit = math.floor(_FIT_FRACTION * n_draw)
    fit_clips, cal_clips = draw[:n_fit], draw[n_fit:]

    row: dict[str, object] = {
        "representation": representation,
        "machine_id": machine_id,
        "fraction": fraction,
        "seed": seed,
        "n_fit_clips": int(fit_clips.size),
        "n_cal_clips": int(cal_clips.size),
        "n_test_normal_clips": int(test_normal.size),
        "n_abnormal_clips": int(abnormal_idx.size),
    }
    if min(fit_clips.size, cal_clips.size, test_normal.size, abnormal_idx.size) == 0:
        logger.info(
            "scarcity_detection: degenerate cell (%s, %s, fraction=%s, seed=%d) -- "
            "fit=%d cal=%d test_normal=%d abnormal=%d clip(s); NaN metrics",
            representation, machine_id, fraction, seed,
            fit_clips.size, cal_clips.size, test_normal.size, abnormal_idx.size,
        )
        row.update({
            "auc_clip": math.nan, "pauc_clip": math.nan, "tpr_at_alpha": math.nan,
            "realized_normal_clip_far": math.nan, "auc_window": math.nan,
            "degenerate": True,
        })
        return row

    fit_windows = np.vstack(
        [features[clip_bounds[i] : clip_bounds[i + 1]] for i in fit_clips]
    )
    scorer = KnnScorer(k=1, metric="cosine").fit(fit_windows)
    cal_clip_scores, _ = _clip_scores(scorer, features, clip_bounds, cal_clips)
    threshold = calibrate(cal_clip_scores, alpha)  # CLIP-level calibration (A1.4)

    test_clip_scores, test_window_scores = _clip_scores(
        scorer, features, clip_bounds, test_normal
    )
    abn_clip_scores, abn_window_scores = _clip_scores(
        scorer, features, clip_bounds, abnormal_idx
    )
    y_clip = np.concatenate(
        [np.zeros(test_clip_scores.size), np.ones(abn_clip_scores.size)]
    )
    s_clip = np.concatenate([test_clip_scores, abn_clip_scores])
    y_window = np.concatenate(
        [np.zeros(test_window_scores.size), np.ones(abn_window_scores.size)]
    )
    s_window = np.concatenate([test_window_scores, abn_window_scores])

    row.update({
        "auc_clip": round(float(roc_auc_score(y_clip, s_clip)), 6),
        "pauc_clip": round(_standardized_pauc(y_clip, s_clip), 6),
        "tpr_at_alpha": round(float(np.mean(abn_clip_scores > threshold.threshold)), 6),
        "realized_normal_clip_far": round(
            float(np.mean(test_clip_scores > threshold.threshold)), 6
        ),
        "auc_window": round(float(roc_auc_score(y_window, s_window)), 6),
        "degenerate": False,
    })
    return row


# ---------------------------------------------------------------------------
# Outputs (CSV + md + figure)
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as fh:
        fh.write(f"# {_PAUC_DEFINITION}\n")  # A1.4: definition named in a header comment
        writer = csv.DictWriter(fh, fieldnames=list(_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(
    path: Path,
    rows: list[dict[str, object]],
    *,
    corpus: str,
    cap: int | None,
    alpha: float,
    machine_ids: list[str],
    skipped: list[tuple[str, str]],
) -> None:
    lines = [
        "# Detection-performance scarcity curve (MIMII proxy)",
        "",
        f"**Corpus:** `{corpus}` -- public-proxy evidence in the machine-id domain, never "
        "PSHP evidence: the PSHP rerun of this harness awaits the induced-fault campaign's "
        "labels (spec section 4 honesty framing).",
        "",
        "## Protocol (spec D4 + amendment A1.4)",
        "",
        f"- Machine ids: {', '.join(machine_ids)}. `limit_clips_per_class={cap}` per "
        "(machine id, class), applied by the loader with kept/total counts logged -- no "
        "silent truncation.",
        "- All splits are CLIP-level (windows of one clip stay together -- the leakage "
        "rule); abnormal clips never touch fitting or calibration.",
        "- Per machine id: 30% of NORMAL clips -> TEST via ONE seed-7 draw, shared by "
        "every (fraction, seed) cell; the remaining 70% form the TRAIN-POOL.",
        "- Per (fraction, seed) cell: draw `ceil(fraction * n_pool)` pool clips (rng = "
        "cell seed), split the draw 80/20 into FIT / CAL clips; `KnnScorer(k=1, cosine)` "
        "fit on the FIT clips' windows; clip score = MEAN over the clip's window scores; "
        f"conformal threshold = `calibrate(cal_clip_scores, alpha={alpha:g})` on "
        "CLIP-level calibration scores -- never window-calibrated/clip-applied "
        "(amendment A1.4). Draws with an empty fit or cal side are flagged "
        "`degenerate` (NaN metrics), never a crash.",
        f"- `{_PAUC_DEFINITION}`",
        "- `tpr_at_alpha` = fraction of abnormal clips above the conformal threshold; "
        "`realized_normal_clip_far` = fraction of TEST-normal clips above it; "
        "`auc_window` = secondary window-level AUC over the same test/abnormal windows.",
        "",
        "## Standardization caveat (amendment A1.5)",
        "",
        "Every window is per-window standardized by the loader (the package-4 corpus "
        "convention) for EVERY representation: this removes clip-gain confounds but also "
        "erases absolute-level anomaly cues -- a fault manifesting only as an overall "
        "loudness change is invisible by construction (the conservative choice). "
        "`AudioFeaturizer`'s output width varies with `rate_hz` (bands capped at "
        "Nyquist) -- self-consistent within this 16 kHz corpus. The student on MIMII "
        "measures TRANSFERABILITY of a PSHP-distilled encoder.",
        "",
        "## Skipped representations",
        "",
    ]
    if skipped:
        lines += [f"- `{name}`: {reason}" for name, reason in skipped]
    else:
        lines.append("- (none)")
    lines += [
        "",
        "## Results",
        "",
        "| " + " | ".join(_CSV_COLUMNS) + " |",
        "|" + "|".join("---" for _ in _CSV_COLUMNS) + "|",
    ]
    lines += ["| " + " | ".join(str(r[c]) for c in _CSV_COLUMNS) + " |" for r in rows]
    path.write_text("\n".join(lines) + "\n")


def _empty_panel(ax: Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _plot_curve(
    out_path: Path, table: pd.DataFrame, *, corpus: str, cap: int | None, alpha: float
) -> None:
    """The scarcity figure (spec D4): 2 panels (`auc_clip`, `tpr_at_alpha`) vs
    fraction (log-x); per representation, the line is the mean over
    (machine_ids x seeds) and the shaded band spans min..max over seeds (each
    seed first averaged over machine ids). Degenerate cells are excluded.
    matplotlib is imported lazily here, with the headless Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), squeeze=False)
    panels = (
        ("auc_clip", "clip-level AUC"),
        ("tpr_at_alpha", f"clip TPR @ conformal alpha={alpha:g}"),
    )
    valid = table[~table["degenerate"].astype(bool)] if len(table) else table
    for ax, (column, ylabel) in zip(axes[0], panels, strict=True):
        if not len(valid):
            _empty_panel(ax, "no non-degenerate cells")
            continue
        for rep_name in sorted(valid["representation"].unique()):
            sub = valid[valid["representation"] == rep_name]
            fractions = np.asarray(sorted(sub["fraction"].unique()), dtype=float)
            means, lows, highs = [], [], []
            for frac in fractions:
                cell = sub[sub["fraction"] == frac]
                means.append(float(cell[column].mean()))
                per_seed = cell.groupby("seed")[column].mean()
                lows.append(float(per_seed.min()))
                highs.append(float(per_seed.max()))
            ax.plot(fractions, means, marker="o", label=rep_name)
            ax.fill_between(fractions, lows, highs, alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel("fraction of training-normal pool clips")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
    fig.suptitle(f"Detection scarcity curve -- {corpus} (limit_clips_per_class={cap})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Corpus root holding **/id_*/{normal,abnormal}/*.wav clips (MIMII layout).",
    )
    parser.add_argument(
        "--representations", default=",".join(_KNOWN_REPRESENTATIONS),
        help="Comma list of representations (unknown name: exit 2); beats/tfc/student "
             "are skipped with a log line when their checkpoint env is unset or the "
             "file is missing.",
    )
    parser.add_argument(
        "--fractions", default="0.05,0.1,0.25,0.5,1.0",
        help="Comma list of TRAIN-POOL fractions in (0, 1].",
    )
    parser.add_argument("--seeds", default="7,8,9", help="Comma list of per-cell draw seeds.")
    parser.add_argument(
        "--machine-ids", default=None,
        help="Comma list of id_* machine directories to evaluate (default: all "
             "discovered under --root; an unknown id: exit 2).",
    )
    parser.add_argument(
        "--limit-clips-per-class", type=int, default=300,
        help="Cap per (machine id, class), applied in sorted order by the loader with "
             "kept/total counts logged (default 300).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Conformal FAR target.")
    parser.add_argument("--out", type=Path, default=Path("results/scarcity-detection"))
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the cell matrix + caps and exit 0 -- writes nothing, never imports "
             "torch (warm_cache convention).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    representations = [r.strip() for r in args.representations.split(",") if r.strip()]
    unknown = [r for r in representations if r not in _KNOWN_REPRESENTATIONS]
    if unknown:
        parser.error(
            f"unknown representation(s) {unknown!r}; known: {list(_KNOWN_REPRESENTATIONS)}"
        )
    if not representations:
        parser.error("--representations must name at least one representation")
    try:
        fractions = [float(tok) for tok in args.fractions.split(",") if tok.strip()]
        seeds = [int(tok) for tok in args.seeds.split(",") if tok.strip()]
    except ValueError as exc:
        parser.error(f"--fractions/--seeds must be numeric comma lists ({exc})")
    if not fractions or any(not (0.0 < f <= 1.0) for f in fractions):
        parser.error(f"--fractions must be non-empty, each in (0, 1]; got {args.fractions!r}")
    if not seeds:
        parser.error("--seeds must name at least one seed")
    if not (0.0 < args.alpha < 1.0):
        parser.error(f"--alpha must be in (0, 1), got {args.alpha!r}")
    if not args.root.is_dir():
        parser.error(f"--root {args.root} is not a directory")

    discovered = sorted({d.name for d in args.root.rglob("id_*") if d.is_dir()})
    if args.machine_ids:
        wanted = [m.strip() for m in args.machine_ids.split(",") if m.strip()]
        missing = [m for m in wanted if m not in discovered]
        if missing:
            print(
                f"scarcity_detection: unknown machine id(s): {', '.join(missing)}; "
                f"discovered under {args.root}: {', '.join(discovered) or '(none)'}",
                file=sys.stderr,
            )
            return 2
        machine_ids = sorted(set(wanted))
    else:
        machine_ids = discovered
    if not machine_ids:
        print(
            f"scarcity_detection: no id_* machine directories under {args.root}",
            file=sys.stderr,
        )
        return 2

    cfg = load_config()
    active: list[_Representation] = []
    skipped: list[tuple[str, str]] = []
    for name in representations:
        rep, reason = _resolve_representation(name, cfg)
        if rep is None:
            logger.info("scarcity_detection: skipping %r -- %s", name, reason)
            skipped.append((name, reason))
        else:
            active.append(rep)

    if args.dry_run:
        print(f"scarcity_detection --dry-run: root={args.root}")
        print(f"  limit_clips_per_class={args.limit_clips_per_class}  alpha={args.alpha:g}")
        for name in representations:
            status = (
                "available" if any(r.name == name for r in active)
                else "SKIP -- " + next(reason for n, reason in skipped if n == name)
            )
            print(f"  representation {name}: {status}")
        print(f"  machine ids ({len(machine_ids)}): {', '.join(machine_ids)}")
        print(f"  fractions: {', '.join(str(f) for f in fractions)}")
        print(f"  seeds: {', '.join(str(s) for s in seeds)}")
        n_cells = len(active) * len(machine_ids) * len(fractions) * len(seeds)
        print(
            f"  cells: {len(active)} representation(s) x {len(machine_ids)} machine "
            f"id(s) x {len(fractions)} fraction(s) x {len(seeds)} seed(s) = {n_cells}"
        )
        return 0

    out: Path = args.out
    cache_dir = out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cap: int | None = args.limit_clips_per_class

    rows: list[dict[str, object]] = []
    for rep in active:
        featurizer: _Featurizer | None = None
        for machine_id in machine_ids:
            manifest = _clip_manifest(args.root, machine_id, cap)
            if not manifest:
                logger.warning(
                    "scarcity_detection: %s x %s -- no clips on disk; skipping machine",
                    rep.name, machine_id,
                )
                continue
            fingerprint = _fingerprint(rep, cap, manifest)
            cache_path = cache_dir / f"{rep.name}__{machine_id}__{fingerprint[:16]}.npz"
            if cache_path.is_file():
                features, clip_bounds, labels = _load_cache(cache_path)
                logger.info(
                    "scarcity_detection: cache HIT %s (%s x %s)",
                    cache_path.name, rep.name, machine_id,
                )
            else:
                if featurizer is None:
                    featurizer = _build_featurizer(rep)
                features, clip_bounds, labels, relpaths = _extract_features(
                    args.root, machine_id, cap, featurizer
                )
                _save_cache(
                    cache_path, rep, machine_id, fingerprint, cap,
                    features, clip_bounds, labels, relpaths,
                )
                logger.info(
                    "scarcity_detection: cache MISS -- extracted %d window(s) over %d "
                    "clip(s) -> %s", features.shape[0], labels.size, cache_path.name,
                )
            if labels.size == 0:
                logger.warning(
                    "scarcity_detection: %s x %s -- zero clips yielded (all shorter than "
                    "one window?); skipping machine", rep.name, machine_id,
                )
                continue

            normal_idx = np.flatnonzero(labels == 0)
            abnormal_idx = np.flatnonzero(labels == 1)
            test_normal, pool = _test_pool_split(normal_idx)
            for fraction in fractions:
                for seed in seeds:
                    rows.append(_evaluate_cell(
                        rep.name, machine_id, features, clip_bounds,
                        test_normal, pool, abnormal_idx, fraction, seed, args.alpha,
                    ))

    _write_csv(out / "scarcity_detection.csv", rows)
    table = pd.DataFrame(rows, columns=list(_CSV_COLUMNS))
    _plot_curve(
        out / "scarcity_curve.png", table,
        corpus=args.root.name or str(args.root), cap=cap, alpha=args.alpha,
    )
    _write_md(
        out / "scarcity_detection.md", rows,
        corpus=str(args.root), cap=cap, alpha=args.alpha,
        machine_ids=machine_ids, skipped=skipped,
    )
    logger.info(
        "scarcity_detection: wrote %s (%d rows), scarcity_curve.png, "
        "scarcity_detection.md", out / "scarcity_detection.csv", len(rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
