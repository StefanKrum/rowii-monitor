"""Step-2 Package 8: per-mode model bank rotations CLI (design spec
`docs/superpowers/specs/2026-07-21-step2-package8-modebank-recal-explain.md` §3.D1 +
Amendment A1.3/A1.5/A1.7, plan `docs/superpowers/plans/
2026-07-21-step2-package8-modebank-explain.md` Task 3): fits Stefan's per-mode model
bank (`rowii.state.modebank.ModeBank`) on a pool of `--fit-runs`' SCADA ground-truth
modes and evaluates it, label-free, on ONE held-out `--test-run` -- a Step-1
alternative arm, run through the SAME held-out-day-group rotation shape
`scripts/run_step2.py --protocol cross-day-pooled` uses (fit pool -> pooled
reference/detector, one held-out day scored), including its A3.8 day-group-
disjointness guard (`_run_day_groups`, duplicated here -- script-sibling rule;
`rowii.anomaly.pools`' own module docstring explains why scripts never import a
sibling script's internals).

Bank labels (spec D1) are SCADA ground truth on the fit days ONLY -- deployment
story: SCADA exists during the supervised commissioning window; the bank then runs
LABEL-FREE (`ModeBank.assign`) on every other day, and GT is used for EVALUATION
only, never as a runtime input. This is a deliberate INFORMATION-advantage contrast
to the unsupervised KMeans+HMM default (`rowii.state.detect.FittedDetector`, which
needs no SCADA ever) -- both arms are reported side by side in one `metrics.json`,
one row tagged `"supervised"` (the bank) and the other `"unsupervised"` (the P7
pooled comparator, RECOMPUTED here via `FittedDetector.fit_pooled` on this
rotation's own pooled fit features -- spec A1.3: "the bank's edge is an
INFORMATION advantage ..., not a method win").

Primary metric = ARI (spec A1.3, mapping-invariant), computed on windows masked to
GT `not in {"unknown", "transition"}` -- BOTH arms, identically (spec A1.5;
narrower than `rowii.eval.metrics.evaluate`'s `unknown`-only mask -- see
`_masked_ari`'s own docstring for the delta this creates vs P7's own k-selection
mask). Accuracy is reported for the bank ONLY: its modes ARE GT names, so
`assigned == gt_state` is a direct, meaningful equality; a KMeans cluster id has no
GT identity to compare against. `--smooth` applies `rowii.state.segments.
duration_filter` ONLY (never `rowii.state.smooth`'s HMM-EM re-estimation) so the
bank-vs-clusterer comparison isolates the LABELING mechanism, not the smoothing
(spec A1.3).

Low-confidence bank members (adversarial-review binding, T2 finding 1,
`rowii.state.modebank.ModeBank.low_confidence_modes`): a member whose conformal
threshold is `+inf` (too few calibration windows for `alpha`) can NEVER contribute
a rejection to `ModeAssignment.no_mode_fits`'s whole-bank AND-conjunction, so the
`no_mode_fits_rate` this rotation reports UNDER-FIRES for as long as such a member
survives -- "no_mode_fits low" must never be read as "no novelty" under that
condition. This CLI surfaces the affected mode names both in `metrics.json`
(`bank.low_confidence_modes`) and as a WARNING line (this module's own logger,
stderr-bound by `main`'s `logging.basicConfig` call like every other Step-2 CLI)
whenever the fitted bank has any (`_warn_low_confidence`).

`no_mode_fits_rate` and `ari`/`accuracy` sit on DIFFERENT footings, deliberately:
the rejection rate is computed on the bank's RAW pre-`--smooth` assignment over
EVERY valid window (`metrics.json`'s `bank.n_valid`, GT-independent), while
`ari`/`accuracy` score the (possibly `--smooth`ed) assigned labels restricted to
the `{unknown, transition}`-masked `n_masked` subset above -- the two rates'
denominators are therefore never directly comparable.

Artifacts under `results/step2/modebank/<test_run>/<variant>-<family>/`:
`metrics.json` (bank ARI/accuracy/no_mode_fits_rate/low_confidence_modes +
P7-comparison ARI, supervised/unsupervised tags), `confusion.csv` (GT state x
assigned-mode window counts, masked identically to the ARI), `assignments.parquet`
(window, valid, gt_state, assigned, no_mode_fits -- one row per window of the test
run's OWN grid, invalid windows carrying the `""`/`False` scatter-back sentinel,
mirroring `scripts/run_step2.py`'s `_INVALID_LABEL` convention), `notes.md` (pool
composition + the spec's D1 attribution line).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.anomaly.pools import PoolResult, build_pool  # noqa: E402
from rowii.anomaly.sweep import SweepConfig  # noqa: E402
from rowii.config import Config, load_config  # noqa: E402
from rowii.io.dataset import (  # noqa: E402
    RecordingIndex,
    Run,
    betriebsdaten_utc_offset_ns,
    discover,
    run_utc_offset_ns,
)
from rowii.io.gantner import read_header  # noqa: E402
from rowii.pipeline import PreparedRun, prepare_run  # noqa: E402
from rowii.scada.labels import gt_labels, load_scada_window_means  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402
from rowii.state.detect import FittedDetector  # noqa: E402
from rowii.state.modebank import ModeBank  # noqa: E402
from rowii.state.segments import duration_filter  # noqa: E402

logger = logging.getLogger(__name__)

_VARIANT_CHOICES: tuple[str, ...] = ("fusion", "vibration", "audio-beats")
"""D1's own representation list (spec §3.D1 "Representations" bullet) -- narrower
than `scripts/run_step2.py`'s full `_VARIANT_CHOICES`: the bank technically scores
any variant's features, but the package's evaluation plan commits to exactly these
three (the vibration arm tests the partner-reported vibration advantage
independently; fusion stays in scope for D1 -- only D2's level-only recalibration
excludes it, A1.1)."""
_FAMILY_CHOICES: tuple[str, ...] = ("gaussian", "knn", "gmm")

_EXCLUDED_GT = ("unknown", "transition")
"""Duplicated from `rowii.state.modebank._EXCLUDED_GT` (spec A1.5) rather than
importing a private module symbol: every ARI/accuracy/confusion computation in
this CLI must mask identically to what the bank itself trained on, so this tuple
is the single local source of truth for that mask -- if `ModeBank`'s own set ever
changes, this constant's docstring is the flag to update it too."""

_BEATS_INSTALL_HINT = (
    'install extra: pip install -e ".[beats]" and set ROWII_BEATS_CHECKPOINT'
)

_BURST_NAME_DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}_\d{6}\.dat$")
"""Duplicated from `scripts/run_step2.py`'s `_BURST_NAME_DATE_RE` (script-sibling
rule) -- the date group `_run_day_groups` needs."""


# ---------------------------------------------------------------------------
# Duplicated script-sibling helpers (rowii.anomaly.pools' module docstring: a
# script must not import another script's internals -- see each docstring below
# for which scripts/run_step2.py helper it mirrors)
# ---------------------------------------------------------------------------


def _import_beats_or_exit() -> None:
    """Duplicated from `scripts/run_step2.py` (script-sibling rule)."""
    try:
        import rowii.signals.beats  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"BEATs featurizer not available ({exc}); {_BEATS_INSTALL_HINT}"
        ) from exc


def _unknown_run_names(names: list[str], index: RecordingIndex) -> list[str]:
    """Duplicated from `scripts/run_step2.py`'s helper of the same name
    (script-sibling rule): names in *names* with no matching discovered run,
    de-duplicated, in the order first seen."""
    known = {r.name for r in index.runs}
    return list(dict.fromkeys(n for n in names if n not in known))


def _run_day_groups(run: Run) -> set[str]:
    """The A3.8-style day groups of *run*: the SET of calendar days
    (`"YYYY-MM-DD"`) parsed from every burst file's NAME across all streams --
    duplicated VERBATIM from `scripts/run_step2.py`'s helper of the same name
    (script-sibling rule; see that function's own docstring for the full
    midnight-crossing/sibling-run rationale for why this must be a SET, not just
    the first file's date).

    Raises:
        ValueError: if *run* has no burst files at all, or any file's name does
            not carry the `_YYYY-MM-DD_HH-MM-SS_ffffff.dat` timestamp.
    """
    bursts = [burst for stream_files in run.files.values() for burst in stream_files]
    if not bursts:
        raise ValueError(
            f"run {run.name!r} has no burst files -- cannot derive its day group"
        )
    days: set[str] = set()
    for burst in bursts:
        match = _BURST_NAME_DATE_RE.search(burst.path.name)
        if match is None:
            raise ValueError(
                f"run {run.name!r}: burst file {burst.path.name!r} does not carry "
                f"the expected _YYYY-MM-DD_HH-MM-SS_ffffff.dat timestamp"
            )
        days.add(match.group(1))
    if len(days) > 1:
        logger.info(
            "run_modebank: run %s spans %d calendar days (%s) -- its A3.8 day "
            "group is the full set",
            run.name, len(days), ", ".join(sorted(days)),
        )
    return days


def _betriebsdaten_for_grid(betriebsdaten: list[Path], grid: WindowGrid) -> list[Path]:
    """Betriebsdaten files whose hourly span intersects *grid*'s UTC time range --
    duplicated from `scripts/run_step2.py`'s helper of the same name
    (script-sibling rule); see that function's own docstring for the full D3
    timebase-shift rationale (candidate file headers are still on the raw DAQ
    axis and must be shifted before the intersection test)."""
    grid_end_ns = int(grid.edges_ns()[-1])
    offset_ns = betriebsdaten_utc_offset_ns(betriebsdaten)
    matched = []
    for path in betriebsdaten:
        header = read_header(path)
        file_start_ns = header.t0_ns + offset_ns
        file_end_ns = file_start_ns + round(header.n_frames / header.sample_rate_hz * 1e9)
        if file_start_ns < grid_end_ns and file_end_ns > grid.t0_ns:
            matched.append(path)
    return sorted(matched)


def _run_gt_states(
    prepared: PreparedRun, run: Run, index: RecordingIndex, cfg: Config
) -> np.ndarray:
    """Full-length `(W,)` object array of GT state strings for *run* -- mirrors
    `scripts/run_step2.py`'s `_load_run_scada` + `_gt_state_labels` collapsed into
    one call (duplicated, script-sibling rule): SCADA window means ->
    `rowii.scada.labels.gt_labels`'s `state` column.

    Unlike `run_step2.py`'s own `_load_run_scada` (which returns `None` for a run
    with no SCADA coverage and lets its own callers treat GT as an optional
    diagnostic), GT is NOT optional here: every fit run supplies bank TRAINING
    labels and the test run supplies bank EVALUATION labels (module docstring,
    spec D1) -- a run with no Betriebsdaten coverage overlapping its own grid
    cannot participate in a rotation at all, so this raises instead of returning
    a sentinel a caller could silently ignore.

    Raises:
        ValueError: if *run*'s day has no Betriebsdaten coverage overlapping its
            own grid.
    """
    day_betriebsdaten = index.betriebsdaten_by_day.get(run.day_root, [])
    matched = (
        _betriebsdaten_for_grid(day_betriebsdaten, prepared.grid) if day_betriebsdaten else []
    )
    if not matched:
        raise ValueError(
            f"run {run.name!r} has no Betriebsdaten coverage overlapping its own "
            f"grid -- the mode bank needs SCADA ground truth on every fit and "
            f"test run (spec D1)"
        )
    scada = load_scada_window_means(
        matched, prepared.grid, audio_run_offset_ns=run_utc_offset_ns(run)
    )
    labels: np.ndarray = gt_labels(scada, cfg.gt, window_s=cfg.window.window_s)[
        "state"
    ].to_numpy()
    return labels


def _pool_gt_labels(pool: PoolResult, gt_by_run: dict[str, np.ndarray]) -> np.ndarray:
    """Per stacked pool row, the GT mode-name STRING of its source window --
    mirrors `scripts/run_step2.py`'s `_pool_row_labels` (duplicated,
    script-sibling rule), generalized to an `object`-dtype output for GT state
    strings instead of `int64` detected-cluster ids. Pool rows are valid windows
    by construction, so every source window has a defined (possibly "unknown"/
    "transition") GT string."""
    out = np.empty(pool.features.shape[0], dtype=object)
    for member_idx, member in enumerate(pool.members):
        mask = pool.run_index == member_idx
        out[mask] = gt_by_run[member.run_name][pool.window_index[mask]]
    return out


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested directly)
# ---------------------------------------------------------------------------


def _masked_ari(gt: np.ndarray, pred: np.ndarray) -> tuple[float, int]:
    """Adjusted Rand Index between *gt* and *pred*, restricted to windows whose
    GT state is NOT in `_EXCLUDED_GT` (spec A1.5: BOTH `"unknown"` AND
    `"transition"` excluded -- narrower than `rowii.eval.metrics.evaluate`'s
    `unknown`-only mask, which callers must account for when comparing this
    against P7's own k-selection mask). The bank's own ARI and the P7-comparison
    row's ARI are both computed through this SAME helper, so the two arms are
    masked identically (spec A1.3's fair-comparison rule).

    Args:
        gt: `(W,)` object array of GT state strings.
        pred: `(W,)` array of predicted labels (bank mode strings, or P7 cluster
            ids) -- any dtype `sklearn.metrics.adjusted_rand_score` accepts.

    Returns:
        `(ari, n_masked)`; `ari` is `nan` when `n_masked == 0` (nothing to score,
        e.g. a run with zero non-excluded GT windows).
    """
    mask = ~np.isin(np.asarray(gt, dtype=object), _EXCLUDED_GT)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float(adjusted_rand_score(gt[mask], pred[mask])), n


def _masked_accuracy(gt: np.ndarray, assigned: np.ndarray) -> tuple[float, int]:
    """Bank accuracy (spec A1.3: bank-only, direct string equality since the
    bank's modes ARE GT names) under the SAME `_EXCLUDED_GT` mask `_masked_ari`
    uses. Returns `(accuracy, n_masked)`; `accuracy` is `nan` when
    `n_masked == 0`."""
    mask = ~np.isin(np.asarray(gt, dtype=object), _EXCLUDED_GT)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.mean(assigned[mask] == gt[mask])), n


def _smooth_ids(labels: np.ndarray, min_dwell: int) -> np.ndarray:
    """`--smooth`'s ONLY transform (spec A1.3, binding): `rowii.state.segments.
    duration_filter` on dense integer ids -- NEVER `rowii.state.smooth`'s HMM-EM
    re-estimation, which would make the bank-vs-clusterer comparison isolate two
    different smoothers' emission fits instead of the labeling mechanism alone.
    """
    return duration_filter(labels, min_dwell=min_dwell)


def _bank_metrics(
    bank: ModeBank,
    *,
    ari: float,
    n_masked: int,
    accuracy: float,
    no_mode_fits_rate: float,
    n_valid: int,
) -> dict[str, object]:
    """The `"bank"` sub-dict of `metrics.json` (the supervised arm) -- includes
    `low_confidence_modes` (adversarial-review binding, T2 finding 1, module
    docstring): a caller reading `no_mode_fits_rate` alone cannot tell whether it
    under-fires, so the affected mode names travel with every written artifact,
    not just the log line `_warn_low_confidence` emits at fit time. `n_valid` is
    `no_mode_fits_rate`'s OWN denominator (every valid window, GT-independent,
    computed on the bank's RAW pre-`--smooth` assignment -- module docstring's
    raw/smoothed, masked/unmasked coexistence note); it is NOT the same
    denominator as `ari`/`accuracy`'s `n_masked` above."""
    return {
        "tag": "supervised",
        "family": bank.family,
        "modes": list(bank.modes),
        "low_confidence_modes": list(bank.low_confidence_modes),
        "ari": ari,
        "n_masked": n_masked,
        "accuracy": accuracy,
        "n_valid": n_valid,
        "no_mode_fits_rate": no_mode_fits_rate,
    }


def _p7_metrics(k: int, ari: float, n_masked: int) -> dict[str, object]:
    """The `"p7_pooled"` sub-dict of `metrics.json` (the unsupervised comparator)
    -- no `accuracy` key: a KMeans cluster id has no GT identity to compare
    against (module docstring)."""
    return {"tag": "unsupervised", "k": k, "ari": ari, "n_masked": n_masked}


def _warn_low_confidence(bank: ModeBank) -> None:
    """Emit ONE WARNING (this module's own logger -- stderr-bound by `main`'s
    `logging.basicConfig`, like every other Step-2 CLI) naming every
    low-confidence surviving member, iff `bank.low_confidence_modes` is
    non-empty (module docstring, adversarial-review binding T2 finding 1): such
    a member's conformal threshold is `+inf` and can NEVER contribute a
    rejection to `ModeAssignment.no_mode_fits`'s whole-bank AND-conjunction, so
    the `no_mode_fits_rate` this rotation reports UNDER-FIRES for these modes --
    it must NOT be read as "no novelty" while any of them survive. A no-op when
    every surviving member calibrated with enough data.

    This CLI-level re-warn is intentional, not a duplicate left over from
    `ModeBank.fit`'s own internal low-confidence WARNING (`rowii.state.
    modebank`): it is artifact-adjacent surfacing, emitted right where this
    rotation's own artifacts are about to be written, for a signal the
    fit-time warning can already have scrolled away past in a long run's log.
    """
    if not bank.low_confidence_modes:
        return
    logger.warning(
        "run_modebank: %d bank member(s) %s calibrated low_confidence=True (too "
        "few calibration windows for alpha=%s) -- their threshold is +inf, so "
        "no_mode_fits_rate UNDER-FIRES for these modes and must NOT be read as "
        "'no novelty' (see ModeBank.low_confidence_modes)",
        len(bank.low_confidence_modes),
        bank.low_confidence_modes,
        bank.alpha,
    )


def _confusion_table(gt: np.ndarray, assigned: np.ndarray) -> pd.DataFrame:
    """GT state x assigned-mode window counts (crosstab), masked to the SAME
    `_EXCLUDED_GT` footing as `_masked_ari`/`_masked_accuracy` -- the confusion
    matrix a reader needs alongside the scalar ARI/accuracy to see WHICH modes
    get confused for which."""
    mask = ~np.isin(np.asarray(gt, dtype=object), _EXCLUDED_GT)
    table = pd.crosstab(
        pd.Series(gt[mask], name="gt_state"), pd.Series(assigned[mask], name="assigned")
    )
    return table.reset_index()


def _notes_markdown(
    fit_run_names: list[str],
    test_run: str,
    variant: str,
    family: str,
    alpha: float,
    bank: ModeBank,
    p7_k: int,
    smoothed: bool,
) -> str:
    """Pool composition + the spec's D1 attribution line (spec §4: "every
    analysis type inspired by the partner's work carries a one-line attribution
    in the script docstring AND the digest")."""
    lines = [
        f"# Mode-bank rotation: {test_run} ({variant}-{family})",
        "",
        f"- fit pool: {', '.join(fit_run_names)}",
        f"- held-out test run: {test_run}",
        f"- family: {family}, alpha: {alpha}, smoothed: {smoothed}",
        f"- bank modes: {', '.join(bank.modes) or '(none -- empty bank)'}",
    ]
    if bank.dropped_modes:
        dropped = ", ".join(f"{m} (n={n})" for m, n in sorted(bank.dropped_modes.items()))
        lines.append(f"- dropped modes (below min_ref or zero calibration): {dropped}")
    if bank.low_confidence_modes:
        lines.append(
            f"- LOW-CONFIDENCE member(s) (no_mode_fits_rate under-fires for "
            f"these -- see metrics.json): {', '.join(bank.low_confidence_modes)}"
        )
    lines += [
        f"- P7 pooled comparator: k={p7_k} (unsupervised, recomputed via "
        f"FittedDetector.fit_pooled on this rotation's own pooled fit features)",
        "",
        "Per-mode model bank inspired by the partner's per-state modeling; all "
        "numbers computed from our caches (spec D1 attribution).",
    ]
    return "\n".join(lines) + "\n"


def _out_dir(results_root: Path, test_run: str, variant: str, family: str) -> Path:
    """`results/step2/modebank/<test_run>/<variant>-<family>/` (plan's literal
    Task 3 layout) -- keyed by the HELD-OUT run, mirroring
    `scripts/run_step2.py`'s cross-day-pooled `_cross_day_pooled_out_dir`."""
    return results_root / "step2" / "modebank" / test_run / f"{variant}-{family}"


# ---------------------------------------------------------------------------
# argparse + main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the per-mode model bank (D1) on a pool of fit runs and evaluate "
            "it, label-free, on ONE held-out test run: bank ARI/accuracy vs the "
            "P7 pooled KMeans+HMM comparator, both masked to GT not in "
            "{unknown, transition} (A1.3/A1.5)."
        )
    )
    parser.add_argument(
        "--fit-runs", required=True,
        help=(
            "Comma-separated fit-run names, pool order matters "
            "(FittedDetector.fit_pooled's row-order note); the held-out "
            "--test-run must not appear here."
        ),
    )
    parser.add_argument("--test-run", required=True, help="The ONE held-out test run.")
    parser.add_argument("--variant", required=True, choices=_VARIANT_CHOICES)
    parser.add_argument("--family", required=True, choices=_FAMILY_CHOICES)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--k", type=int, default=5, help="knn family's neighbour count (unused otherwise)."
    )
    parser.add_argument("--min-ref", type=int, default=20)
    parser.add_argument(
        "--p7-k", type=int, default=4,
        help="Cluster count for the recomputed P7 pooled comparator (unsupervised row).",
    )
    parser.add_argument(
        "--smooth", action="store_true",
        help=(
            "Apply rowii.state.segments.duration_filter to the bank's argmin "
            "labels before scoring (A1.3: duration-filter ONLY, never the "
            "HMM-EM smoother)."
        ),
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable rowii.pipeline.prepare_run's on-disk feature cache.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    fit_run_names = [n.strip() for n in args.fit_runs.split(",") if n.strip()]
    if not fit_run_names:
        parser.error("--fit-runs got an empty run-name list")
    if len(set(fit_run_names)) != len(fit_run_names):
        parser.error(
            "--fit-runs contains duplicate run name(s) -- a run pools its sides once"
        )
    if args.test_run in fit_run_names:
        parser.error(
            f"--test-run {args.test_run!r} is listed in --fit-runs: the held-out "
            f"day must not be a pool member"
        )

    if args.variant == "audio-beats":
        _import_beats_or_exit()

    cfg = load_config()
    index = discover(cfg.data_root)

    unknown = _unknown_run_names([*fit_run_names, args.test_run], index)
    if unknown:
        available = ", ".join(sorted({r.name for r in index.runs})) or "(none discovered)"
        print(
            f"run_modebank: unknown run name(s): {', '.join(unknown)}; "
            f"available runs: {available}",
            file=sys.stderr,
        )
        return 2

    runs_by_name = {r.name: r for r in index.runs}
    fit_runs = [runs_by_name[name] for name in fit_run_names]
    test_run_obj = runs_by_name[args.test_run]

    try:
        test_days = _run_day_groups(test_run_obj)
        overlapping = sorted(
            {run.name for run in fit_runs if _run_day_groups(run) & test_days}
        )
    except ValueError as exc:
        print(f"run_modebank: {exc}", file=sys.stderr)
        return 2
    if overlapping:
        print(
            f"run_modebank: fit run(s) {', '.join(overlapping)} share a calendar "
            f"day with held-out test run {args.test_run!r} (day group(s) "
            f"{sorted(test_days)}) -- fit and test day groups must be disjoint "
            f"(A3.8-style guard)",
            file=sys.stderr,
        )
        return 2

    prepared_all: dict[str, PreparedRun] = {}
    for run in (*fit_runs, test_run_obj):
        try:
            prepared_all[run.name] = prepare_run(
                run, args.variant, cfg, use_cache=not args.no_cache
            )
        except RuntimeError as exc:
            print(
                f"run_modebank: prepare_run failed for run {run.name!r} ({exc})",
                file=sys.stderr,
            )
            return 2

    gt_by_run: dict[str, np.ndarray] = {}
    for run in (*fit_runs, test_run_obj):
        try:
            gt_by_run[run.name] = _run_gt_states(prepared_all[run.name], run, index, cfg)
        except ValueError as exc:
            print(f"run_modebank: {exc}", file=sys.stderr)
            return 2

    prepared_fit = {name: prepared_all[name] for name in fit_run_names}
    prepared_test = prepared_all[test_run_obj.name]

    # Test-run feature-contract guard (T3-review MEDIUM finding 1, mirrors
    # scripts/run_step2.py's `_run_cross_day_pooled` test-run-vs-fit-pool
    # geometry guard): `feature_names` here is the SAME "first fit run's
    # names" value ModeBank.fit is handed below as the pool's contract.
    # Checking full name equality (not just column COUNT) catches BOTH a
    # width mismatch -- otherwise an uncaught ValueError traceback out of
    # ModeBank.assign's own width check further down -- AND same-width
    # channel-name drift, which a width-only check would miss and which
    # would otherwise score silently against a positionally-misaligned
    # contract (the 080726 TurbineVib-ch0 case, rowii.anomaly.pools.
    # build_pool's own sibling guard against the same failure mode).
    feature_names = list(next(iter(prepared_fit.values())).feature_names)
    test_feature_names = list(prepared_test.feature_names)
    if test_feature_names != feature_names:
        if len(test_feature_names) != len(feature_names):
            detail = (
                f"{len(test_feature_names)} feature column(s) vs the fit "
                f"pool's {len(feature_names)} -- incompatible feature dims"
            )
        else:
            diverging = [
                (a, b)
                for a, b in zip(feature_names, test_feature_names, strict=True)
                if a != b
            ]
            detail = (
                f"same width ({len(feature_names)} column(s)) but disagrees "
                f"on feature names/order (first divergence(s): "
                f"{diverging[:3]}) -- channel-availability drift between "
                f"days can produce same-width runs whose columns MEAN "
                f"different channels"
            )
        print(
            f"run_modebank: test run {test_run_obj.name!r} feature contract "
            f"does not match the fit pool's ({detail}) -- refusing to score "
            f"a positionally-misaligned contract",
            file=sys.stderr,
        )
        return 2

    sweep_cfg = SweepConfig(alpha=args.alpha)
    pool_fit = build_pool(prepared_fit, "fit", sweep_cfg)
    pool_conformal = build_pool(prepared_fit, "conformal", sweep_cfg)
    if pool_fit.features.shape[0] == 0:
        print(
            "run_modebank: the pooled FIT side is empty (every fit run's splits "
            "were degenerate -- see the build_pool warnings above) -- nothing to "
            "fit the bank on",
            file=sys.stderr,
        )
        return 2

    fit_gt = _pool_gt_labels(pool_fit, gt_by_run)
    calib_gt = _pool_gt_labels(pool_conformal, gt_by_run)

    try:
        bank = ModeBank.fit(
            pool_fit.features, fit_gt, pool_conformal.features, calib_gt,
            family=args.family, alpha=args.alpha, feature_names=feature_names,
            min_ref=args.min_ref, k=args.k,
        )
    except ValueError as exc:
        print(f"run_modebank: ModeBank.fit failed ({exc})", file=sys.stderr)
        return 2
    _warn_low_confidence(bank)

    valid = prepared_test.valid_mask
    gt_full_test = gt_by_run[test_run_obj.name]
    gt_test_valid = gt_full_test[valid]
    assignment = bank.assign(prepared_test.features[valid])
    assigned_labels = assignment.labels

    if args.smooth and bank.modes:
        mode_to_id = {m: i for i, m in enumerate(bank.modes)}
        ids = np.array([mode_to_id[lbl] for lbl in assigned_labels], dtype=np.int64)
        window_s = prepared_test.grid.window_ns / 1e9
        min_dwell = max(1, round(cfg.detect.min_dwell_s / window_s))
        smoothed_ids = _smooth_ids(ids, min_dwell)
        assigned_labels = np.array([bank.modes[i] for i in smoothed_ids], dtype=object)

    ari, n_masked = _masked_ari(gt_test_valid, assigned_labels)
    accuracy, _n_acc = _masked_accuracy(gt_test_valid, assigned_labels)
    no_mode_fits_rate = (
        float(assignment.no_mode_fits.mean()) if assignment.no_mode_fits.size else float("nan")
    )

    try:
        p7_detector = FittedDetector.fit_pooled(pool_fit.features, cfg, k=args.p7_k)
    except (RuntimeError, ValueError) as exc:
        print(
            f"run_modebank: k too large for this pool: fit_pooled(k={args.p7_k}) on "
            f"{pool_fit.features.shape[0]} pooled fit window(s) failed ({exc}) -- "
            f"pick a smaller --p7-k",
            file=sys.stderr,
        )
        return 2
    n_valid_test = int(valid.sum())
    valid_grid = WindowGrid(
        t0_ns=prepared_test.grid.t0_ns, window_ns=prepared_test.grid.window_ns,
        n_windows=n_valid_test,
    )
    p7_labels = p7_detector.apply(prepared_test.features[valid], valid_grid).frame_labels
    p7_ari, p7_n_masked = _masked_ari(gt_test_valid, p7_labels)

    metrics: dict[str, dict[str, object]] = {
        "bank": _bank_metrics(
            bank, ari=ari, n_masked=n_masked, accuracy=accuracy,
            no_mode_fits_rate=no_mode_fits_rate, n_valid=n_valid_test,
        ),
        "p7_pooled": _p7_metrics(args.p7_k, p7_ari, p7_n_masked),
    }

    out_dir = _out_dir(cfg.results_root, test_run_obj.name, args.variant, args.family)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    confusion = _confusion_table(gt_test_valid, assigned_labels)
    confusion.to_csv(out_dir / "confusion.csv", index=False)

    # Scattered to the test run's FULL grid (plan interface text), invalid
    # windows carrying the ""/False scatter-back sentinel -- mirrors
    # scripts/run_step2.py's `_INVALID_LABEL` convention (module docstring).
    # `assigned` carries --smooth's duration-filtered labels when requested
    # (raw argmin otherwise); `no_mode_fits` stays the RAW pre-smooth
    # per-window conformal-rejection signal (module docstring's raw/smoothed
    # note) and is never itself smoothed.
    n_grid = prepared_test.grid.n_windows
    full_assigned = np.full(n_grid, "", dtype=object)
    full_no_mode_fits = np.zeros(n_grid, dtype=bool)
    full_assigned[valid] = assigned_labels
    full_no_mode_fits[valid] = assignment.no_mode_fits
    assignments = pd.DataFrame(
        {
            "window": np.arange(n_grid, dtype=np.int64),
            "valid": valid,
            "gt_state": gt_full_test,
            "assigned": full_assigned,
            "no_mode_fits": full_no_mode_fits,
        }
    )
    assignments.to_parquet(out_dir / "assignments.parquet", index=False)

    (out_dir / "notes.md").write_text(
        _notes_markdown(
            fit_run_names, test_run_obj.name, args.variant, args.family, args.alpha,
            bank, args.p7_k, args.smooth,
        )
    )

    print(
        f"run_modebank: wrote {args.family} bank rotation for test run "
        f"{test_run_obj.name!r} (fit pool: {', '.join(fit_run_names)}) to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
