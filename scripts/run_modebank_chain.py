"""Step-2 mode-bank CHAIN PROBE ("per-mode references built from bank labels
... vs the pooled-clusterer detected-state chain -- does better state
assignment translate into better FAR control?"): fits Stefan's per-mode model
bank (`rowii.state.modebank.ModeBank`) on a pool
of `--fit-runs`' SCADA ground-truth modes (`scripts/run_modebank.py`'s own fit/
evaluate machinery, duplicated here -- script-sibling rule), then runs a Step-2 kNN-style FAR
chain CONDITIONED ON THE BANK'S OWN LABEL-FREE MODE ASSIGNMENT: per-mode references on
the pooled fit side, per-mode conformal thresholds on the pooled conformal side, FAR
on the held-out test run's top-split SCORING side.

This is the "detected-labels only" convention `scripts/run_step2.py --protocol
cross-day-pooled` uses for its own (unsupervised) chain -- `rowii.state.detect.
FittedDetector`'s cluster ids there, `ModeBank.assign`'s mode names here -- so the
resulting `far_table.csv` is directly comparable, column-for-column
(`_FAR_TABLE_COLUMNS`, duplicated from `scripts/run_step2.py`) and split-for-split
(`_top_split`, BINDING split parity below), to that cross-day-pooled chain's own
`far_table_frozen.csv`. GROUND TRUTH IS NEVER READ FOR THE TEST RUN: unlike
`scripts/run_modebank.py` (which needs the test run's own GT to score bank
accuracy/ARI), this probe's whole question is answered by an EXTERNAL comparison of
two far_table.csv files -- SCADA is only ever loaded for `--fit-runs`, to train the
bank in the first place (`scripts/run_modebank.py`'s own deployment story: "SCADA
exists during the supervised commissioning window; ... the bank then runs LABEL-FREE
... GT is used for EVALUATION only" -- here there is no evaluation step at all, so no
GT load on the held-out day).

Split parity (BINDING): `_top_split` reproduces
`scripts/run_step2.py`'s cross-day-pooled top split
(`split_by_segments(prepared_test.segment_ids, prepared_test.valid_mask,
calibration_frac=0.5, seed=7)`, its `SweepConfig`'s own defaults) with two HARD-CODED
literals rather than reading them off this script's own `SweepConfig` -- so the
held-out day's scoring-window SET is byte-identical to the cross-day-pooled chain's
regardless of any future flag here that might otherwise perturb this script's own
`SweepConfig` construction. Only the SCORING side of that split is used
(frozen-threshold comparison only -- this chain probe is not a recalibrate/frozen
quad like the cross-day-pooled protocol; it produces exactly one `far_table.csv`).

Per-mode reference/threshold assembly (`_build_far_table`) mirrors
`scripts/run_step2.py`'s `_cross_day_pooled_tables` dispatch (`rowii.anomaly.sweep`'s
public `far_row_excluded`/`far_row_no_conformal_data`/`far_row_empty_scoring`/
`far_row_scored`/`far_row_aggregate`) with `rowii.anomaly.references.build_references`
standing in for that function's inline per-label boolean-mask loop: a bank-assigned
mode with fewer than `sweep_cfg.min_ref` pooled FIT-side rows is excluded outright
(no reference at all); a mode with a real reference but zero pooled CONFORMAL-side
rows gets a no-conformal-data row; a mode with a real threshold but zero SCORING-side
rows gets an empty-scoring row; otherwise a real scored row. A trailing `"pooled"`
aggregate row closes the table, identical convention to every other Step-2 FAR table
in this project.

Attribution: per-mode model bank inspired by the partner's per-state
modeling; all numbers in this script are computed from OUR OWN caches -- no partner
JSON/number is read anywhere in this module.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.anomaly.conformal import ConformalThreshold, calibrate  # noqa: E402
from rowii.anomaly.pools import PoolResult, build_pool  # noqa: E402
from rowii.anomaly.references import SegmentSplit, build_references, split_by_segments  # noqa: E402
from rowii.anomaly.scorers import KnnScorer, Scorer  # noqa: E402
from rowii.anomaly.sweep import (  # noqa: E402
    FarRow,
    SweepConfig,
    far_row_aggregate,
    far_row_empty_scoring,
    far_row_excluded,
    far_row_no_conformal_data,
    far_row_scored,
)
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
from rowii.state.modebank import ModeBank  # noqa: E402

logger = logging.getLogger(__name__)

_VARIANT_CHOICES: tuple[str, ...] = ("fusion", "vibration", "audio-beats")
"""Duplicated from `scripts/run_modebank.py`'s own tuple -- script-sibling rule.
The chain probe's OWN pinned binding runs `fusion` only; the other two stay
selectable for anticipated pillar-3-adjacent reuse, exactly as
`run_modebank.py`'s three-representation rotation matrix does."""
_FAMILY_CHOICES: tuple[str, ...] = ("gaussian", "knn", "gmm")
"""Duplicated from `rowii.state.modebank._FAMILIES` (never imported: private
symbol, matches `run_modebank.py`'s own `_FAMILY_CHOICES` duplication rationale)."""
_SCORER_CHOICES: tuple[str, ...] = ("knn",)
"""This CLI's `--scorer` choices -- narrower than `scripts/run_step2.py`'s full
eight-scorer `_SCORER_CHOICES` by design: this chain probe only ever needs
`rowii.anomaly.scorers.KnnScorer` (the same default `scripts/run_step2.py
--protocol cross-day-pooled` uses for its own cross-day-pooled chain),
so `_make_scorer` below implements exactly that one case. Kept as a real dispatch
(not a hard-coded `KnnScorer()` call) so a future scorer only needs a new `elif`
branch here plus a `_SCORER_CHOICES` entry, matching every other scorer-name
dispatch in this project (`rowii.anomaly.sweep._make_scorer`)."""

_FAR_TABLE_COLUMNS: tuple[str, ...] = (
    "label", "n_calibration", "n_scored", "n_alarms", "realized_far", "nominal_alpha",
    "achievable_alpha_floor", "low_confidence", "threshold", "excluded",
)
"""Duplicated from `scripts/run_step2.py`'s own `_FAR_TABLE_COLUMNS` (script-sibling
rule) -- matches `rowii.anomaly.sweep.SweepResult.far_table`'s documented column
contract, so this script's `far_table.csv` is directly comparable, column-for-column,
to the cross-day-pooled chain's own `far_table_frozen.csv` (module docstring)."""

_BEATS_INSTALL_HINT = (
    'install extra: pip install -e ".[beats]" and set ROWII_BEATS_CHECKPOINT'
)

_BURST_NAME_DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}_\d{6}\.dat$")
"""Duplicated from `scripts/run_step2.py`'s/`scripts/run_modebank.py`'s
`_BURST_NAME_DATE_RE` (script-sibling rule) -- the date group `_run_day_groups`
needs."""

_TOP_SEED = 7
_TOP_FRAC = 0.5
"""The held-out test run's top calibration/scoring split -- BINDING split parity
with `scripts/run_step2.py`'s cross-day-pooled protocol (module docstring).
That protocol's own `SweepConfig` construction (`SweepConfig(alpha=alpha,
top_k=top_k, scorer=scorer_name)`) never overrides `calibration_frac`/`seed`, so its
effective top split is always `(0.5, 7)` -- the two literals here, rather than this
script's own locally-constructed `SweepConfig`'s fields, so the held-out day's
scoring-window SET stays byte-identical to the cross-day-pooled chain's regardless of
any future flag added to THIS script that might otherwise perturb its own
`SweepConfig`.
Tripwire: `tests/test_run_modebank_chain.py::test_top_split_literals_match_sweepconfig_defaults`
pins these two literals against `SweepConfig`'s own defaults, so a future default
drift fails loudly there instead of silently rotting this parity claim."""


# ---------------------------------------------------------------------------
# Duplicated script-sibling helpers (rowii.anomaly.pools' module docstring: a
# script must not import another script's internals -- these mirror
# scripts/run_modebank.py's (itself mirroring scripts/run_step2.py's) helpers of
# the same name; see each one's own docstring for what differs here).
# ---------------------------------------------------------------------------


def _import_beats_or_exit() -> None:
    """Duplicated from `scripts/run_modebank.py`/`scripts/run_step2.py`
    (script-sibling rule)."""
    try:
        import rowii.signals.beats  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"BEATs featurizer not available ({exc}); {_BEATS_INSTALL_HINT}"
        ) from exc


def _unknown_run_names(names: list[str], index: RecordingIndex) -> list[str]:
    """Duplicated from `scripts/run_modebank.py`'s helper of the same name
    (script-sibling rule): names in *names* with no matching discovered run,
    de-duplicated, in the order first seen."""
    known = {r.name for r in index.runs}
    return list(dict.fromkeys(n for n in names if n not in known))


def _run_day_groups(run: Run) -> set[str]:
    """The day groups of *run* -- duplicated VERBATIM from
    `scripts/run_modebank.py`'s (originally `scripts/run_step2.py`'s) helper of the
    same name (script-sibling rule).

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
    return days


def _betriebsdaten_for_grid(betriebsdaten: list[Path], grid: WindowGrid) -> list[Path]:
    """Betriebsdaten files whose hourly span intersects *grid*'s UTC time range --
    duplicated from `scripts/run_modebank.py`'s helper of the same name
    (script-sibling rule)."""
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
    """Full-length `(W,)` object array of GT state strings for *run* -- duplicated
    from `scripts/run_modebank.py`'s helper of the same name (script-sibling rule):
    SCADA window means -> `rowii.scada.labels.gt_labels`'s `state` column.

    Unlike `run_modebank.py`, THIS script calls this helper for `--fit-runs` ONLY
    (module docstring): the held-out `--test-run` never needs GT here -- the chain
    probe's whole question ("does better state assignment translate into better FAR
    control?") is answered by an EXTERNAL comparison against a separately-produced
    cross-day-pooled `far_table_frozen.csv`, not by this script scoring against the
    test run's own ground truth.

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
            f"grid -- the mode bank needs SCADA ground truth on every fit run "
            f"(spec D1)"
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
    duplicated from `scripts/run_modebank.py`'s helper of the same name
    (script-sibling rule). Used ONLY to train the bank (`ModeBank.fit`'s
    `fit_labels`/`calib_labels`) -- never to condition the chain itself, which uses
    `ModeBank.assign`'s label-free output instead (module docstring)."""
    out = np.empty(pool.features.shape[0], dtype=object)
    for member_idx, member in enumerate(pool.members):
        mask = pool.run_index == member_idx
        out[mask] = gt_by_run[member.run_name][pool.window_index[mask]]
    return out


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested directly)
# ---------------------------------------------------------------------------


def _top_split(segment_ids: np.ndarray, valid_mask: np.ndarray) -> SegmentSplit:
    """The held-out test run's top calibration/scoring split -- see `_TOP_SEED`/
    `_TOP_FRAC`'s module-level docstring for the BINDING split-parity rationale.
    Only `.scoring_windows` is used downstream (frozen-threshold chain only).
    """
    return split_by_segments(segment_ids, valid_mask, _TOP_FRAC, _TOP_SEED)


def _far(scores: np.ndarray, threshold: float) -> tuple[float, int, int]:
    """`(realized_far, n_alarms, n_scored)` for one label's SCORING-side scores
    against its own calibrated *threshold* -- the shared alarm rule every Step-2
    FAR table in this project uses (`score > threshold`). `realized_far` is `nan`
    when *scores* is empty (never called that way here -- callers check
    `label_mask.any()` first and route an empty scoring side to
    `far_row_empty_scoring` instead, matching `scripts/run_step2.py`'s
    `_pooled_mode_row`)."""
    n = int(scores.shape[0])
    n_alarm = int((scores > threshold).sum())
    return (n_alarm / n if n else float("nan")), n_alarm, n


def _make_scorer(name: str) -> Scorer:
    """A fresh, unfitted scorer instance for *name* -- see `_SCORER_CHOICES`'s own
    docstring for why only `"knn"` is wired up.

    Raises:
        ValueError: if *name* is not `"knn"`.
    """
    if name == "knn":
        return KnnScorer()
    raise ValueError(f"scorer must be one of {_SCORER_CHOICES}, got {name!r}")


def _pooled_mode_row(
    label: str,
    sweep_cfg: SweepConfig,
    threshold_result: ConformalThreshold,
    label_scoring_scores: np.ndarray | None,
) -> FarRow:
    """One bank-assigned mode's row for a label that HAS a calibrated threshold --
    mirrors `scripts/run_step2.py`'s `_pooled_mode_row` (duplicated, script-sibling
    rule): `far_row_empty_scoring` when the held-out day's scoring side holds no
    window of this label (*label_scoring_scores* is `None` exactly then), else
    `_far` + `far_row_scored`."""
    if label_scoring_scores is None:
        return far_row_empty_scoring(label, sweep_cfg, threshold_result)
    _realized_far, n_alarms, n_scored = _far(label_scoring_scores, threshold_result.threshold)
    return far_row_scored(label, sweep_cfg, threshold_result, n_scored, n_alarms)


def _build_far_table(
    pool_fit_features: np.ndarray,
    pool_fit_labels: np.ndarray,
    pool_conformal_features: np.ndarray,
    pool_conformal_labels: np.ndarray,
    scoring_features: np.ndarray,
    scoring_labels: np.ndarray,
    sweep_cfg: SweepConfig,
) -> pd.DataFrame:
    """The chain's own per-(bank-assigned)-mode reference/threshold/FAR assembly,
    from ONE pooled-reference pass -- mirrors `scripts/run_step2.py`'s
    `_cross_day_pooled_tables` dispatch (module docstring), specialized to a single
    (frozen) threshold mode since this probe has no recalibrate arm.

    *_labels arrays are BANK-ASSIGNED mode strings (`ModeBank.assign(...).labels`),
    NEVER GT -- the "detected-labels only" convention this whole chain mirrors from
    `run_step2.py --protocol cross-day-pooled` (module docstring).

    Per label observed anywhere among the three inputs:

    - **reference**: the label's pooled FIT-side rows (`rowii.anomaly.references.
      build_references`, `min_ref=sweep_cfg.min_ref`); below that floor (or the
      label never appears on the fit side at all) -> `far_row_excluded`.
    - **threshold**: `calibrate` on the label's pooled CONFORMAL-side scores; zero
      such rows -> `far_row_no_conformal_data` (the reference above is real, just
      nothing to calibrate a threshold with).
    - **alarms**: the label's SCORING-side rows, scored once against that one
      threshold; zero such rows -> `far_row_empty_scoring`.

    A trailing `label="pooled"` aggregate row closes the table
    (`rowii.anomaly.sweep.far_row_aggregate`), the same convention every other
    Step-2 FAR table in this project uses.

    Args:
        pool_fit_features: `(Nf, F)` finite pooled FIT-side feature matrix.
        pool_fit_labels: `(Nf,)` bank-assigned mode strings, row-aligned with
            *pool_fit_features*.
        pool_conformal_features: `(Nc, F)` finite pooled CONFORMAL-side feature
            matrix.
        pool_conformal_labels: `(Nc,)` bank-assigned mode strings, row-aligned
            with *pool_conformal_features*.
        scoring_features: `(Ns, F)` finite held-out-day SCORING-side feature
            matrix (top-split scoring windows only).
        scoring_labels: `(Ns,)` bank-assigned mode strings, row-aligned with
            *scoring_features*.
        sweep_cfg: Supplies `alpha` (`calibrate`'s target), `min_ref` (the
            fit-side reference floor), and `scorer` (`_make_scorer`'s dispatch
            key).

    Returns:
        A DataFrame with `_FAR_TABLE_COLUMNS` columns, one row per observed label
        plus the trailing `"pooled"` aggregate row.
    """
    all_labels = sorted(
        {str(v) for v in np.unique(pool_fit_labels)}
        | {str(v) for v in np.unique(pool_conformal_labels)}
        | {str(v) for v in np.unique(scoring_labels)}
    )
    ref_set = build_references(
        pool_fit_features,
        pool_fit_labels,
        np.arange(pool_fit_features.shape[0], dtype=np.int64),
        min_ref=sweep_cfg.min_ref,
    )

    rows: list[FarRow] = []
    for label in all_labels:
        reference = ref_set.references.get(label)
        if reference is None:
            rows.append(far_row_excluded(label, sweep_cfg))
            continue
        scorer = _make_scorer(sweep_cfg.scorer).fit(reference)

        conformal_rows = pool_conformal_features[pool_conformal_labels == label]
        if conformal_rows.shape[0] == 0:
            rows.append(far_row_no_conformal_data(label, sweep_cfg))
            continue
        threshold = calibrate(scorer.score(conformal_rows), sweep_cfg.alpha)

        label_mask = scoring_labels == label
        label_scoring_scores = (
            scorer.score(scoring_features[label_mask]) if label_mask.any() else None
        )
        rows.append(_pooled_mode_row(label, sweep_cfg, threshold, label_scoring_scores))

    rows.append(far_row_aggregate(rows, sweep_cfg))
    return pd.DataFrame([asdict(r) for r in rows], columns=_FAR_TABLE_COLUMNS)


def _notes_markdown(
    fit_run_names: list[str],
    test_run: str,
    variant: str,
    family: str,
    alpha: float,
    scorer_name: str,
    bank: ModeBank,
) -> str:
    """Pool composition + the attribution line."""
    lines = [
        f"# Mode-bank Step-2 chain probe: {test_run} ({variant}-{family})",
        "",
        f"- fit pool: {', '.join(fit_run_names)}",
        f"- held-out test run: {test_run} (GT never loaded for this run -- "
        f"see module docstring)",
        f"- bank family: {family}, chain scorer: {scorer_name}, alpha: {alpha}",
        f"- bank modes (drive the chain's per-mode conditioning): "
        f"{', '.join(bank.modes) or '(none)'}",
        "- thresholds: pool-conformal (FROZEN regime, spec A3.7) -- compare "
        "against run_step2's far_table_frozen.csv for the same rotation, NOT "
        "far_table_recalibrate.csv (this probe has no recalibrate arm)",
    ]
    if bank.dropped_modes:
        dropped = ", ".join(f"{m} (n={n})" for m, n in sorted(bank.dropped_modes.items()))
        lines.append(
            f"- GT mode(s) dropped from the bank (below min_ref or zero "
            f"calibration): {dropped}"
        )
    lines += [
        "",
        "far_table.csv rows are conditioned on the bank's OWN label-free mode "
        "assignment (ModeBank.assign), never GT, on both the pooled fit/conformal "
        "sides and the held-out test run's top-split scoring side -- the same "
        "'detected-labels only' convention run_step2.py's cross-day-pooled "
        "protocol uses for its own (unsupervised) chain, with the bank standing "
        "in for the KMeans+HMM detector (spec D1: \"does better state assignment "
        "translate into better FAR control?\").",
        "",
        "Per-mode model bank inspired by the partner's per-state modeling; all "
        "numbers computed from our caches (spec D1 attribution).",
    ]
    return "\n".join(lines) + "\n"


def _out_dir(results_root: Path, test_run: str, variant: str, family: str) -> Path:
    """`results/step2/modebank-chain/<test_run>/<variant>-<family>/` -- a
    SEPARATE top-level leaf from `scripts/run_modebank.py`'s own
    `results/step2/modebank/...` (that CLI's `metrics.json`/`confusion.csv`/
    `assignments.parquet`/`notes.md` for the SAME (test_run, variant, family)
    combination would otherwise collide with this CLI's `far_table.csv`/
    `notes.md`), keyed by the HELD-OUT run like every other Step-2 protocol's
    out-dir helper."""
    return results_root / "step2" / "modebank-chain" / test_run / f"{variant}-{family}"


# ---------------------------------------------------------------------------
# argparse + main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step-2 chain probe (D1): fit the per-mode model bank on a pool of "
            "fit runs, then condition a Step-2 scorer's per-mode references and "
            "conformal thresholds on the bank's OWN label-free mode assignment "
            "(never GT) and score the held-out test run's top-split SCORING "
            "side -- far_table.csv, split-parity with run_step2.py's "
            "cross-day-pooled chain."
        )
    )
    parser.add_argument(
        "--fit-runs", required=True,
        help="Comma-separated fit-run names; the held-out --test-run must not appear here.",
    )
    parser.add_argument("--test-run", required=True, help="The ONE held-out test run.")
    parser.add_argument("--variant", required=True, choices=_VARIANT_CHOICES)
    parser.add_argument("--family", required=True, choices=_FAMILY_CHOICES)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--scorer", choices=_SCORER_CHOICES, default="knn")
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
            f"run_modebank_chain: unknown run name(s): {', '.join(unknown)}; "
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
        print(f"run_modebank_chain: {exc}", file=sys.stderr)
        return 2
    if overlapping:
        print(
            f"run_modebank_chain: fit run(s) {', '.join(overlapping)} share a "
            f"calendar day with held-out test run {args.test_run!r} (day group(s) "
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
                f"run_modebank_chain: prepare_run failed for run {run.name!r} ({exc})",
                file=sys.stderr,
            )
            return 2

    # GT is loaded for the FIT runs ONLY (module docstring/_run_gt_states): the
    # held-out test run needs no SCADA coverage at all for this probe.
    gt_by_run: dict[str, np.ndarray] = {}
    for run in fit_runs:
        try:
            gt_by_run[run.name] = _run_gt_states(prepared_all[run.name], run, index, cfg)
        except ValueError as exc:
            print(f"run_modebank_chain: {exc}", file=sys.stderr)
            return 2

    prepared_fit = {name: prepared_all[name] for name in fit_run_names}
    prepared_test = prepared_all[test_run_obj.name]

    sweep_cfg = SweepConfig(alpha=args.alpha, scorer=args.scorer)
    pool_fit = build_pool(prepared_fit, "fit", sweep_cfg)
    pool_conformal = build_pool(prepared_fit, "conformal", sweep_cfg)
    if pool_fit.features.shape[0] == 0:
        print(
            "run_modebank_chain: the pooled FIT side is empty (every fit run's "
            "splits were degenerate -- see the build_pool warnings above) -- "
            "nothing to fit the bank on",
            file=sys.stderr,
        )
        return 2

    fit_gt = _pool_gt_labels(pool_fit, gt_by_run)
    calib_gt = _pool_gt_labels(pool_conformal, gt_by_run)
    feature_names = list(next(iter(prepared_fit.values())).feature_names)

    try:
        bank = ModeBank.fit(
            pool_fit.features, fit_gt, pool_conformal.features, calib_gt,
            family=args.family, alpha=args.alpha, feature_names=feature_names,
        )
    except ValueError as exc:
        print(f"run_modebank_chain: ModeBank.fit failed ({exc})", file=sys.stderr)
        return 2

    if not bank.modes:
        print(
            "run_modebank_chain: the bank has no surviving mode (every GT mode "
            "failed the min_ref/calibration floor) -- nothing to condition the "
            "chain on",
            file=sys.stderr,
        )
        return 2

    try:
        top = _top_split(prepared_test.segment_ids, prepared_test.valid_mask)
    except ValueError as exc:
        print(
            f"run_modebank_chain: test run {test_run_obj.name!r} cannot form its "
            f"top calibration/scoring split ({exc})",
            file=sys.stderr,
        )
        return 2
    scoring_features = prepared_test.features[top.scoring_windows]

    # Bank-assigned (label-free) mode strings for every side the chain conditions
    # on -- NEVER GT (module docstring). ModeBank.assign has no temporal/ordering
    # dependency, so it can be called directly on each already-stacked/subset
    # matrix without a full-run-grid round trip.
    pool_fit_bank_labels = bank.assign(pool_fit.features).labels
    pool_conformal_bank_labels = bank.assign(pool_conformal.features).labels
    scoring_bank_labels = bank.assign(scoring_features).labels

    far_table = _build_far_table(
        pool_fit.features, pool_fit_bank_labels,
        pool_conformal.features, pool_conformal_bank_labels,
        scoring_features, scoring_bank_labels,
        sweep_cfg,
    )

    out_dir = _out_dir(cfg.results_root, test_run_obj.name, args.variant, args.family)
    out_dir.mkdir(parents=True, exist_ok=True)
    coerced = far_table.copy()
    coerced["label"] = coerced["label"].astype(str)  # _write_sweep_outputs convention
    coerced.to_csv(out_dir / "far_table.csv", index=False)

    (out_dir / "notes.md").write_text(
        _notes_markdown(
            fit_run_names, test_run_obj.name, args.variant, args.family, args.alpha,
            args.scorer, bank,
        )
    )

    print(
        f"run_modebank_chain: wrote bank-labeled FAR chain for test run "
        f"{test_run_obj.name!r} (fit pool: {', '.join(fit_run_names)}) to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
