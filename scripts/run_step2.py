"""Step-2 mode-conditioned anomaly-sweep CLI: prepare -> label -> sweep -> report.

Drives `rowii.anomaly.sweep.run_sweep` (Task S5) over one or more (run, variant,
labels, conditioning, scorer) combinations for three protocols (design spec
`docs/superpowers/specs/2026-07-09-step2-mode-conditioned-ad-design.md` §2-4, plan
`docs/superpowers/plans/2026-07-09-step2-first-package.md` Task S6, extended by
package 2's design spec `docs/superpowers/specs/2026-07-15-step2-scarcity-crossday-
beats-design.md` §D2 for the third protocol below):

- **within-day**: per selected run, prepare features (`rowii.pipeline.prepare_run`),
  attach labels, then run one sweep per (conditioning, scorer) pair the CLI was asked
  for.
- **cross-day**: calibrate a POOLED reference on one SCADA-covered run ("day A"),
  score every OTHER SCADA-covered run's ("day B") valid windows against it -- a
  cross-day false-alarm-rate matrix, format-compatible with the partner's own
  cross-day comparison table (no values adopted from either side).
- **cross-day-per-state** (package-2 Task 3, spec D2): same SCADA-covered day pairs as
  `cross-day`, but day A's detector is TRANSFERRED to day B (`rowii.state.detect.
  FittedDetector.apply`, no refit) instead of pooling, and day B's windows are scored
  under their own PREDICTED state against day A's per-state reference/threshold for
  that state -- package 2's answer to whether per-state conditioning restores the FAR
  control `cross-day`'s pooling loses (see the dedicated section below).

Every combo's outputs land under `results/step2/...` (see `_within_day_out_dir`/
`_cross_day_out_dir`/`_cross_day_per_state_out_dir`), plus two shared, append-only
artifacts every combo contributes a row/section to: `results/step2/summary.csv` and
`results/step2/candidate_register.md`.

## Labels (`--labels detected|gt`)

`detected` (the default, and the only run-time-realistic mode -- design spec §2: "Per-
state normal references built from Step-1 detected labels ... GT states used only in
evaluation views"): `rowii.state.detect.FittedDetector.fit` (`run_detection`'s own
delegate since package 2's Task 2) runs on this run's VALID windows only (mirrors
`scripts/run_step1.py`'s own `_detect_and_report`), then gets scattered back into a
full-length `(W,)` int64 array with the `-1` sentinel on invalid windows.
That sentinel never actually reaches `run_sweep`'s internals: `rowii.anomaly.
references.split_by_segments` (called from `run_sweep`) only ever draws window indices
from `PreparedRun.valid_mask == True` positions, which are EXACTLY the positions this
module filled with a real (>= 0) cluster id -- so `-1` is written for documentation/
debugging clarity only, never read by the sweep.

`gt` loads SCADA (`rowii.scada.labels.load_scada_window_means` + `gt_labels`) and uses
the state STRINGS directly as labels. Binding decision (documented here, not in the
spec's own literal text, which only says GT is for "diagnostics"): windows whose GT
state is `"unknown"` are excluded from the sweep by AND-ing `(state != "unknown")` into
a COPY of `PreparedRun.valid_mask` (`dataclasses.replace` -- the original `PreparedRun`
returned by `prepare_run`/its cache is never mutated), rather than assigning them some
placeholder label -- the same "exclude via mask, not via label value" mechanism
`detected` mode uses for its own invalid windows, so both modes share one exclusion
principle end to end.

## Cross-day: pooled-only (binding simplification, spec §2)

Detected cluster ids from two different days do not refer to the same physical state
(KMeans label 0 on day A is not comparable to label 0 on day B) and GT-state alignment
across days is out of scope for this package (spec: "per-state cross-day needs label
alignment across days ... out of scope; document"). Cross-day therefore NEVER calls
`run_sweep` (which is structurally single-run: one three-way split of ONE `PreparedRun`)
-- `_cross_day_sweep` below hand-builds the same `SweepResult` shape directly: fit a
scorer on day A's fit-part, calibrate its threshold on day A's conformal-part (both from
one `split_by_segments(day_a.segment_ids, day_a_valid_mask, 0.5, seed)` call -- day A's
own top-level calibration/scoring split collapses away since day A never contributes
scoring windows here), then score ALL of day B's valid windows against that one
threshold. The result's single `far_table` row is always labeled `"pooled"`.

`--labels` still applies to cross-day, but its effect is narrower than within-day: since
the pooled path never reads a label VALUE, `gt` mode only tightens which windows are
ELIGIBLE (excluding GT-`"unknown"` windows from both day A and day B, same mechanism as
within-day), while `detected` mode never runs `run_detection` at all for cross-day (there
is no per-label reference to build) and simply uses each day's own `valid_mask` unchanged.

## Cross-day-per-state: detector transfer (package-2 spec D2)

Per-state conditioning across two different days cannot simply reuse either day's own
detected cluster ids, for exactly the reason the section above gives as cross-day's own
reason for pooling only: KMeans label 0 fit independently on day A and day B has no
reason to mean the same physical state on both. `cross-day-per-state` dissolves this by
never fitting a SECOND detector at all: `_cross_day_per_state_sweep` fits day A's
detector ONCE (`_detected_labels_and_detector`, `FittedDetector.fit`) and TRANSFERS it
to day B (`_apply_detector_labels`, `FittedDetector.apply` -- fit-day standardization +
fit-day HMM Viterbi decode, no refit/EM anywhere, package-2 spec D1). Day B's windows
are then scored under their own PREDICTED (transferred) state, each against day A's
per-state reference and per-state conformal threshold for that SAME state id --
conditioning key = day A's cluster id, one model, one label space, so the cross-day
label-alignment problem this package's own spec calls out never arises. Day A's own
split collapses to fit/conformal only (`split_by_segments(day_a.segment_ids,
day_a_valid_mask, 0.5, seed)`, day A never contributes scoring windows here either,
exactly like `cross-day`'s own day A); day B contributes ALL of its own valid windows
to scoring, keyed by their predicted state.

This is the runtime-honest path (spec D2): `--labels gt` is rejected outright for this
protocol (`main`'s `parser.error` guard, checked before any run is even prepared)
rather than silently reinterpreted -- there is no meaningful "GT-conditioned" transfer
sweep to run, since GT state names are no more comparable across days than detected
cluster ids are, without the exact alignment problem this protocol exists to dissolve.
GT is used only as a DIAGNOSTIC overlay, never in the runtime path: when day B has
SCADA coverage, `far_by_true_state.csv` reports the SAME scored windows' realized FAR
grouped by their TRUE state instead of their predicted one (`_far_by_true_state`) -- a
side-by-side view of whether per-state conditioning controlled the false-alarm rate
both by what the detector predicted and by what actually happened physically, never
fed back into scoring itself.

The `pooled` cell of this protocol's own comparison grid (spec D2: "conditioning
{per-state, pooled recomputed under the same transfer protocol}") is intentionally
NEVER reimplemented here: pooling ignores per-state distinctions entirely, so a pooled
sweep under the transfer protocol is mathematically identical to `cross-day`'s own
existing pooled sweep (`_cross_day_sweep`) -- the existing `--protocol cross-day`
output IS that grid cell, unchanged, and doubles as the published pooled comparator
this protocol's per-state numbers are checked against.

## Output layout

- within-day: `results/step2/within-day/<run>/<variant>-<labels>/<conditioning>-<scorer>/`
  (`far_table.csv`, `scores.parquet`, `candidates.md`) -- exactly the spec's literal path.
  `--states K` (package-3 Task 6, within-day + detected-labels only) appends a `-k<K>`
  suffix to the `<variant>-<labels>` segment ONLY when K is non-default
  (`fusion-detected-k8/`), so a non-default conditioning-granularity run never collides
  with -- or overwrites -- the default-K layout; `summary.csv`'s own `variant` column
  and the combo's `candidate_register.md` section header carry the same suffix
  (`_within_day_out_dir`/`_summary_row`/`_register_section_markdown` -- the register
  would otherwise accumulate identical headers across a K-granularity sweep of one
  run/variant). `--ensemble` (Task 4, design chapter's committed majority-voting
  ensemble) writes `far_table_ensemble.csv` + `ensemble_notes.md` into a dedicated
  `<variant>-<labels>[-k<K>]/ensemble/` sibling directory, ONE level above the
  `<conditioning>-<scorer>/` combo dirs -- the view is conditioning/scorer-
  independent, so (unlike `--score-fusion`, which currently re-writes an identical
  copy into every combo dir) it is written exactly once per (run, `--labels`,
  `--states`) rather than duplicated (`_ensemble_out_dir`'s own docstring has the
  full placement rationale; the score-fusion redundancy is deliberately left as-is,
  noted for the final whole-branch review).
- cross-day: the spec only writes `results/step2/cross-day/<variant>/<dayA>__to__<dayB>/`
  literally, with no room for the `--labels`/`--scorer` axes a single invocation can still
  sweep over (`--scorer all`, `--labels gt`) without collision. Binding extension
  (documented here): `results/step2/cross-day/<variant>-<labels>/<dayA>__to__<dayB>/
  <scorer>-pooled/` -- same `<variant>-<labels>` convention as within-day, plus a
  `<scorer>-pooled` leaf (conditioning is always "pooled" for cross-day, so it is folded
  into the leaf name rather than kept as a separate segment).
- cross-day-per-state (package-2 Task 3, no literal-path precedent in either spec):
  `results/step2/cross-day-per-state/<dayA>--to--<dayB>/<variant>-<scorer>/`
  (`_cross_day_per_state_out_dir`) -- DOUBLE-DASH `--to--`, deliberately different from
  cross-day's `__to__`, so a pair directory's own name alone already distinguishes the
  two protocols' outputs on disk. `--labels` is not part of this path at all (always
  "detected" -- `gt` is rejected outright, see the section above) and conditioning is
  always "per-state" (no separate segment for it, mirroring how cross-day folds its own
  fixed "pooled" conditioning into its `<scorer>-pooled` leaf -- here there is no
  remaining leaf to fold it into beyond `<variant>-<scorer>` since labels/conditioning
  are both constants for this protocol). When day B has SCADA coverage,
  `far_by_true_state.csv` (columns: `true_state, n_scored, n_alarms, realized_far`) is
  written alongside the usual three files in the same combo dir (`_far_by_true_state`).
- `results/step2/summary.csv` (append-only, one row per combo actually written this
  invocation): `run, protocol, variant, labels, conditioning, scorer, alpha,
  per_label_count, pooled_realized_far, mean_per_state_far, n_low_confidence, notes`.
  `protocol` (package-2 addition, 2nd column) is `"within-day"`/`"cross-day"`/
  `"cross-day-per-state"`; `_read_summary_csv_or_none` backfills it onto a pre-package-2
  `summary.csv` that is missing the column entirely (`"cross-day"` when `run` already
  encodes a `_cross_day_summary_row`-style `"<dayA>__to__<dayB>"` pair, else
  `"within-day"` -- a file written before this package cannot contain a
  `cross-day-per-state` row, so that value is never inferred, only ever written
  explicitly by this package's own code going forward). `run` holds the pair string
  `"<dayA>__to__<dayB>"` for a cross-day row (`notes="cross-day pooled"`, spec's literal
  wording) or `"<dayA>--to--<dayB>"` (double-dash, matching the output-directory
  convention above) for a cross-day-per-state row. `per_label_count` is the spec's prose
  "per-label-count", snake_cased to match every other summary column.
  `pooled_realized_far` is the realized FAR treating every non-excluded label's
  alarms/scored-counts as one combined bucket (identical arithmetic to
  `rowii.anomaly.sweep`'s own per-state aggregate row, recomputed locally here for
  `conditioning="pooled"` sweeps too, which never get that row from `run_sweep` itself
  -- cross-day-per-state sweeps DO get it, from the now-public `far_row_aggregate`, so
  their summary row reads it back via `_summary_far_metrics` exactly like a within-day
  per-state sweep would). `mean_per_state_far` is the plain unweighted mean of each
  label's own `realized_far` (NaN/excluded labels dropped).
- `results/step2/candidate_register.md` (append-only): a static header (written once,
  `_REGISTER_HEADER`) naming this register's purpose plus the ONE external candidate this
  package's spec calls out by name -- the partner's pre-start filling-valve (Fuelldüse)
  observation, clearly source-labeled and never adopted as a value -- followed by one
  `### run / variant-labels / conditioning-scorer` section per combo actually written,
  reproducing that combo's own top-k-per-label table (`source="own sweep"`,
  `assessment="unreviewed"`, both fixed since a script cannot yet judge its own output).

A sweep failing with `ValueError` (e.g. `rowii.anomaly.references.split_by_segments`
finding a degenerate split -- too few segments, or a `calibration_frac` that would empty
one side; `_cross_day_per_state_sweep` raises the same way for the identical reasons,
plus "day B has zero eligible windows") is logged and SKIPPED for that one combo, not
fatal to the whole invocation -- mirrors `scripts/run_step1.py`'s own "no SCADA
coverage" defensive branch, generalized to every way a single combo can legitimately
have nothing to report. The same principle extends one level earlier: a run whose own
`rowii.pipeline.prepare_run` raises `RuntimeError` (too short/sparse for the requested
variant -- e.g. a real "two stray files" run like `010726-tu1-afternoon`, Task S7
real-data finding) is logged and excluded -- the whole run for within-day
(`_run_within_day_for_run`), or just that one day and every pair touching it for
cross-day (`_run_cross_day`) and cross-day-per-state (`_run_cross_day_per_state`,
identical exclusion mechanics), never the rest of the invocation.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import logging
import math
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.anomaly.conformal import (  # noqa: E402
    ConformalThreshold,
    calibrate,
    loo_p_values,
    p_values,
)
from rowii.anomaly.fusion import (  # noqa: E402
    fisher_statistic,
    split_branch_columns,
    tippett_statistic,
)
from rowii.anomaly.recon import ConvAeScorer, LstmAeScorer, MlpAeScorer  # noqa: E402
from rowii.anomaly.references import build_references, split_by_segments  # noqa: E402
from rowii.anomaly.scorers import (  # noqa: E402
    IsolationForestScorer,
    KnnScorer,
    LofScorer,
    MahalanobisScorer,
    OcSvmScorer,
    Scorer,
)
from rowii.anomaly.sweep import (  # noqa: E402
    FarRow,
    SweepConfig,
    SweepResult,
    _assert_three_way_disjoint,
    far_row_aggregate,
    far_row_empty_scoring,
    far_row_excluded,
    far_row_no_conformal_data,
    far_row_scored,
    run_sweep,
    scores_and_candidates,
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
from rowii.pipeline import (  # noqa: E402
    _BEATS_INSTALL_HINT,
    PreparedRun,
    _is_beats_variant,
    prepare_run,
)
from rowii.scada.labels import gt_labels, load_scada_window_means  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402
from rowii.state.detect import FittedDetector  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI enums
# ---------------------------------------------------------------------------

ConditioningName = Literal["per-state", "pooled"]
ScorerName = Literal[
    "knn", "mahalanobis", "ocsvm", "iforest", "lof", "mlpae", "lstmae", "convae"
]

_PROTOCOL_CHOICES: tuple[str, ...] = ("within-day", "cross-day", "cross-day-per-state")
_VARIANT_CHOICES: tuple[str, ...] = (
    "audio", "vibration", "fusion", "audio-beats", "fusion-beats", "logmel",
)
_SCORER_CHOICES: tuple[str, ...] = (
    "knn", "mahalanobis", "ocsvm", "iforest", "lof", "mlpae", "lstmae", "convae", "all",
)
_CONDITIONING_CHOICES: tuple[str, ...] = ("per-state", "pooled", "all")
_LABELS_CHOICES: tuple[str, ...] = ("detected", "gt")

_CONCRETE_SCORERS: tuple[ScorerName, ...] = (
    "knn", "mahalanobis", "ocsvm", "iforest", "lof", "mlpae", "lstmae", "convae",
)
_CONCRETE_CONDITIONINGS: tuple[ConditioningName, ...] = ("per-state", "pooled")

_INVALID_LABEL = -1
"""Sentinel written into `_detected_labels`'/`_detected_labels_and_detector`'s (and,
for day B, `_apply_detector_labels`'s) full-length label array on invalid windows --
never read by `run_sweep` or `_cross_day_per_state_sweep` (module docstring)."""

_POOLED_LABEL = "pooled"
"""FAR-table label for `rowii.anomaly.sweep`'s own per-state aggregate row (its private
`_POOLED_ROW_LABEL`, same value, not imported across modules -- also reused, via the
now-public `far_row_aggregate`, by `_cross_day_per_state_sweep`'s own aggregate row) and
this script's single cross-day row. Collides with a real label only if a state is
itself named "pooled" -- not a name any detected cluster id (int) or GT state string
uses."""


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Step-2 mode-conditioned anomaly sweep: per-state (or pooled) "
            "split-conformal scoring over one or more prepared runs, producing a FAR "
            "table, per-window scores, top-k anomaly candidates, and an append-only "
            "candidate register."
        )
    )
    parser.add_argument(
        "--protocol", choices=_PROTOCOL_CHOICES, default="within-day",
        help="'within-day': one blocked split per run. 'cross-day': calibrate on one "
             "SCADA-covered run, score every other one (pooled only). "
             "'cross-day-per-state': same day pairs, but day A's detector is "
             "TRANSFERRED to day B (FittedDetector.apply) and scoring is conditioned "
             "on each window's predicted state (detected-labels only, package-2 spec "
             "D2).",
    )
    parser.add_argument(
        "--run", default="all",
        help=(
            "Run name(s): 'all' (default) or a comma-separated list. within-day: "
            "process each named run in sequence ('all' = every SCADA-covered run "
            "discovered under ROWII_DATA_ROOT). cross-day/cross-day-per-state: 'all' "
            "sweeps every ordered pair of SCADA-covered runs; a list (>= 2 names) "
            "restricts the sweep to ordered pairs where BOTH days are in the list. "
            "An unknown name anywhere in the list exits 2 naming the available runs."
        ),
    )
    parser.add_argument("--variant", choices=_VARIANT_CHOICES, default="fusion")
    parser.add_argument("--scorer", choices=_SCORER_CHOICES, default="knn")
    parser.add_argument("--conditioning", choices=_CONDITIONING_CHOICES, default="all")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--labels", choices=_LABELS_CHOICES, default="detected",
        help="'detected' (default): Step-1 detected cluster ids. 'gt': SCADA-derived "
             "ground-truth state strings, for diagnostics (design spec §2).",
    )
    parser.add_argument(
        "--states", type=int, default=None,
        help=(
            "Conditioning granularity (package-3 Task 6): number of detected "
            "sub-clusters (k) the within-day sweep's own detector fits, overriding "
            "cfg.detect.n_states (default: 4). within-day + detected-labels only "
            "(parser.error for any other --protocol or for --labels gt, which never "
            "fits a detector); must be >= 2. A non-default value suffixes the combo "
            "out-dir, its summary.csv 'variant', and its candidate-register section "
            "header with '-k<K>', so it never collides with -- or overwrites -- the "
            "default-K layout."
        ),
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help=(
            "Disable rowii.pipeline.prepare_run's on-disk feature cache "
            "(results/cache/<run>--<variant>.npz): always recompute features and "
            "never write a cache entry for this invocation."
        ),
    )
    parser.add_argument(
        "--score-fusion", action="store_true",
        help=(
            "Also compute the score-level fusion view (Fisher/Tippett p-value "
            "combination of the audio and vibration branches, re-calibrated via "
            "split conformal, design spec D5): writes far_table_scorefusion.csv + "
            "scorefusion_notes.md into each combo dir alongside the normal sweep "
            "outputs. Only valid with --protocol within-day and --variant fusion."
        ),
    )
    parser.add_argument(
        "--score-fusion-scorer", choices=("knn", "mahalanobis"), default="knn",
        help=(
            "Per-branch scorer for --score-fusion (default: knn); fit "
            "independently on the audio branch's and the vibration branch's own "
            "columns."
        ),
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help=(
            "Also compute the majority-ensemble evaluation view (design chapter's "
            "committed ensemble: OC-SVM + Isolation Forest + LSTM-AE, decision-"
            "level -- an alarm requires >= 2 of the 3 members to agree): writes "
            "far_table_ensemble.csv + ensemble_notes.md into a dedicated "
            "'ensemble/' sibling directory alongside each run's normal sweep-combo "
            "dirs, once per (run, --labels, --states) -- conditioning/scorer-"
            "independent, unlike --score-fusion. Only valid with --protocol "
            "within-day and --labels detected."
        ),
    )
    return parser


def _resolve_choice(choice: str, all_value: str, concrete: tuple[str, ...]) -> tuple[str, ...]:
    return concrete if choice == all_value else (choice,)


def _resolve_scorers(choice: str) -> tuple[ScorerName, ...]:
    resolved = _resolve_choice(choice, "all", cast("tuple[str, ...]", _CONCRETE_SCORERS))
    return cast("tuple[ScorerName, ...]", resolved)


def _resolve_conditionings(choice: str) -> tuple[ConditioningName, ...]:
    resolved = _resolve_choice(choice, "all", cast("tuple[str, ...]", _CONCRETE_CONDITIONINGS))
    return cast("tuple[ConditioningName, ...]", resolved)


def _scada_covered_runs(index: RecordingIndex) -> list[Run]:
    """Runs whose own day tree has at least one Betriebsdaten file -- the within-day
    default run set (`--run all`) and cross-day's full "day" set alike. A run with no
    SCADA at all can still be swept WITHIN-DAY in `detected`-labels mode via an
    explicit `--run` name, but contributes no SCADA context/GT view, so it is excluded
    from this DEFAULT selection (and, being outside the cross-day protocols' "day"
    universe, from every cross-day pair set even when explicitly named).
    """
    return [r for r in index.runs if index.betriebsdaten_by_day.get(r.day_root)]


def _parse_run_names(run_arg: str) -> list[str] | None:
    """`--run`'s value -> `None` for the `"all"` sentinel, else the comma-split name
    list (whitespace-stripped, empty tokens dropped -- `"a, b"` and `"a,b"` parse
    identically; a single bare name is just a one-element list, preserving the
    pre-package-2 single-name calling convention)."""
    if run_arg == "all":
        return None
    return [name.strip() for name in run_arg.split(",") if name.strip()]


def _unknown_run_names(names: list[str], index: RecordingIndex) -> list[str]:
    """Names in *names* with no matching discovered run, de-duplicated, in the
    order first seen -- empty if every name resolves. Mirrors `scripts/warm_cache.py`'s
    own private helper of the same name (duplicated rather than imported -- one script
    must not depend on a SIBLING script's internals, `_betriebsdaten_for_grid`'s
    docstring)."""
    known = {r.name for r in index.runs}
    return list(dict.fromkeys(n for n in names if n not in known))


def _resolve_runs(run_names: list[str] | None, index: RecordingIndex) -> list[Run]:
    """`None` (`--run all`) -> every SCADA-covered run; else the named runs, in the
    given order. Names are already validated against the index by `main` (an unknown
    name exits 2 there, mirroring `scripts/warm_cache.py`'s precedent) before this is
    ever called, so every name resolves. A named run needs no SCADA coverage for
    within-day (`_scada_covered_runs`' own docstring), so resolution draws from ALL
    discovered runs, not the SCADA-covered subset."""
    if run_names is None:
        return _scada_covered_runs(index)
    return [r for name in run_names for r in index.runs if r.name == name]


def _import_beats_or_exit() -> None:
    try:
        import rowii.signals.beats  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"BEATs featurizer not available ({exc}); {_BEATS_INSTALL_HINT}"
        ) from exc


# ---------------------------------------------------------------------------
# SCADA context (used by both label modes' candidate/register output, and by gt mode
# for the labels themselves)
# ---------------------------------------------------------------------------


def _betriebsdaten_for_grid(betriebsdaten: list[Path], grid: WindowGrid) -> list[Path]:
    """Betriebsdaten files whose hourly span intersects the grid's UTC time range.

    Identical logic to `scripts/run_step1.py`'s own private helper of the same name
    (duplicated rather than imported -- this script must not depend on a SIBLING
    script's internals; `rowii.pipeline`'s own module docstring explains why shared
    Step-1/Step-2 logic lives in the `rowii` package instead, and this ~10-line
    time-overlap filter is small enough that duplicating it costs less than adding a
    script-to-script coupling would).

    Task 10 (D3 tracing finding): *grid* is true-UTC since `rowii.pipeline.
    build_run_grid` (D2), but each candidate file's own `header.t0_ns` (`read_
    header`, straight off disk) is still the raw DAQ axis -- shifted here by
    *betriebsdaten*'s own derived offset (`rowii.io.dataset.
    betriebsdaten_utc_offset_ns`) before the intersection test, mirroring `rowii.
    scada.labels.load_scada_window_means`'s identical D3 fix. BEFORE this task the
    comparison was RAW-vs-RAW (grid built on the pre-fix raw axis too) -- both
    sides shared the SAME axis by construction, so selection worked correctly by
    accident, not because either side was ever true UTC (see the task report for
    the full derivation).
    """
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


def _load_run_scada(prepared: PreparedRun, run: Run, index: RecordingIndex) -> pd.DataFrame | None:
    """Per-window SCADA means for *run*'s own day (`rowii.scada.labels.
    load_scada_window_means`), or `None` if this run's day has no Betriebsdaten at all
    (or none whose time range actually overlaps the grid) -- used both for candidate/
    register context columns (P/rpm/KS/Q, any label mode) and, by the caller, to derive
    `gt` labels.

    *run* also supplies the D3 audio-side cross-check (`run_utc_offset_ns(run)`,
    passed to `load_scada_window_means` as `audio_run_offset_ns` -- never used to
    derive the SCADA-side shift itself).
    """
    day_betriebsdaten = index.betriebsdaten_by_day.get(run.day_root, [])
    if not day_betriebsdaten:
        return None
    matched = _betriebsdaten_for_grid(day_betriebsdaten, prepared.grid)
    if not matched:
        return None
    return load_scada_window_means(
        matched, prepared.grid, audio_run_offset_ns=run_utc_offset_ns(run)
    )


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def _detected_labels_and_detector(
    prepared: PreparedRun, cfg: Config, k: int | None = None
) -> tuple[np.ndarray, FittedDetector]:
    """`_detected_labels` + the `FittedDetector` behind it (package-2 spec D2:
    `cross-day-per-state` needs day A's detector, not just its labels, to TRANSFER it
    to day B -- `_apply_detector_labels` below).

    `k` (package-3 Task 6, `--states`): overrides `cfg.detect.n_states`'s cluster
    count for THIS fit only, passed straight through to `FittedDetector.fit`'s own `k`
    parameter. `None` (every caller except a within-day run given `--states`) is a
    pure pass-through -- `FittedDetector.fit` already defaults `k=None` to
    `cfg.n_states` itself, so this adds no new branching, only a wider signature. Both
    existing callers (`_detected_labels` below, `_cross_day_per_state_sweep`) keep
    calling this with no `k` argument at all, so their behavior is unchanged.
    """
    valid_mask = prepared.valid_mask
    features_valid = prepared.features[valid_mask]
    n_valid = int(valid_mask.sum())
    valid_grid = WindowGrid(
        t0_ns=prepared.grid.t0_ns, window_ns=prepared.grid.window_ns, n_windows=n_valid
    )
    detector, det = FittedDetector.fit(
        features_valid, valid_grid, cfg.detect, clusterer="kmeans", k=k
    )
    full_labels = np.full(prepared.features.shape[0], _INVALID_LABEL, dtype=np.int64)
    full_labels[valid_mask] = det.frame_labels
    return full_labels, detector


def _detected_labels(prepared: PreparedRun, cfg: Config) -> np.ndarray:
    """Full-length `(W,)` int64 detected cluster-id labels, `_INVALID_LABEL` on invalid
    windows (module docstring) -- mirrors `scripts/run_step1.py`'s own
    `_detect_and_report` scatter-back pattern.
    """
    labels, _detector = _detected_labels_and_detector(prepared, cfg)
    return labels


def _apply_detector_labels(prepared: PreparedRun, detector: FittedDetector) -> np.ndarray:
    """Day-B per-window labels in day-A's id space via `FittedDetector.apply` (fit-day
    standardization + fit-day HMM decode, no refit -- package-2 spec D1),
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


def _gt_state_labels(scada: pd.DataFrame, cfg: Config) -> np.ndarray:
    """`(W,)` object array of GT state strings for *scada* (module docstring: `gt`
    labels mode)."""
    gt = gt_labels(scada, cfg.gt, window_s=cfg.window.window_s)
    labels: np.ndarray = gt["state"].to_numpy()
    return labels


# ---------------------------------------------------------------------------
# Scorer construction (cross-day only -- within-day delegates to run_sweep, which does
# its own internal dispatch)
# ---------------------------------------------------------------------------


def _make_scorer(name: str) -> Scorer:
    if name == "knn":
        return KnnScorer()
    if name == "mahalanobis":
        return MahalanobisScorer()
    if name == "ocsvm":
        return OcSvmScorer()
    if name == "iforest":
        return IsolationForestScorer()
    if name == "lof":
        return LofScorer()
    if name == "mlpae":
        return MlpAeScorer()
    if name == "lstmae":
        return LstmAeScorer()
    if name == "convae":
        return ConvAeScorer()
    raise ValueError(
        f"scorer must be 'knn', 'mahalanobis', 'ocsvm', 'iforest', 'lof', 'mlpae', "
        f"'lstmae', or 'convae', got {name!r}"
    )


# ---------------------------------------------------------------------------
# Cross-day sweep (hand-built SweepResult -- see module docstring)
# ---------------------------------------------------------------------------

_FAR_TABLE_COLUMNS: tuple[str, ...] = (
    "label", "n_calibration", "n_scored", "n_alarms", "realized_far", "nominal_alpha",
    "achievable_alpha_floor", "low_confidence", "threshold", "excluded",
)
"""Matches `rowii.anomaly.sweep.SweepResult.far_table`'s documented column contract --
redefined locally rather than importing the sweep module's own private column-name
tuple, since these DataFrames need to share exactly one schema with within-day's
`run_sweep` output (`_write_sweep_outputs`/`_summary_far_metrics` do not otherwise care
which protocol produced them)."""
_SCORES_COLUMNS: tuple[str, ...] = ("window", "label", "score", "p_value", "alarm")
_CANDIDATES_COLUMNS: tuple[str, ...] = ("window", "label", "score", "p_value", "rank")

_CROSS_DAY_SEED = 7
"""Seed for cross-day's single `split_by_segments` call -- matches `SweepConfig.seed`'s
own default (7), for the same reason: an arbitrary-but-fixed, documented choice."""


def _cross_day_sweep(
    prepared_a: PreparedRun,
    valid_a: np.ndarray,
    prepared_b: PreparedRun,
    valid_b: np.ndarray,
    scorer_name: str,
    alpha: float,
    top_k: int,
) -> SweepResult:
    """Fit + calibrate a POOLED scorer on day A's own fit/conformal split, then score
    ALL of day B's eligible windows against that one threshold (module docstring's
    "Cross-day: pooled-only" section) -- returns a `SweepResult` with exactly one
    `far_table` row (`label="pooled"`), so downstream writers never need to
    special-case which protocol produced it.

    Args:
        prepared_a: Day A's `PreparedRun` (calibration source).
        valid_a: Day A's EFFECTIVE valid mask (module docstring: `--labels gt` further
            restricts this beyond `prepared_a.valid_mask`; `detected` mode passes
            `prepared_a.valid_mask` unchanged).
        prepared_b: Day B's `PreparedRun` (scoring target).
        valid_b: Day B's effective valid mask, same convention as `valid_a`.
        scorer_name: Any scorer name `_make_scorer` accepts.
        alpha: Nominal false-alarm rate.
        top_k: Candidate register size.

    Raises:
        ValueError: if `split_by_segments` cannot produce a non-empty fit/conformal
            split of day A's eligible windows, or if day B has zero eligible windows to
            score.
    """
    fit_conformal_split = split_by_segments(prepared_a.segment_ids, valid_a, 0.5, _CROSS_DAY_SEED)
    fit_windows = fit_conformal_split.calibration_windows
    conformal_windows = fit_conformal_split.scoring_windows

    scorer = _make_scorer(scorer_name).fit(prepared_a.features[fit_windows])
    conformal_scores = scorer.score(prepared_a.features[conformal_windows])
    threshold: ConformalThreshold = calibrate(conformal_scores, alpha)

    scoring_windows = np.flatnonzero(valid_b).astype(np.int64)
    if scoring_windows.size == 0:
        raise ValueError("cross-day: day B has zero eligible (valid) windows to score")

    scores = scorer.score(prepared_b.features[scoring_windows])
    p_vals = p_values(scores, conformal_scores)
    alarms = scores > threshold.threshold
    n_scored = int(scoring_windows.shape[0])
    n_alarms = int(alarms.sum())

    far_table = pd.DataFrame(
        [{
            "label": _POOLED_LABEL,
            "n_calibration": float(threshold.n_calibration),
            "n_scored": float(n_scored),
            "n_alarms": float(n_alarms),
            "realized_far": (n_alarms / n_scored) if n_scored > 0 else math.nan,
            "nominal_alpha": alpha,
            "achievable_alpha_floor": threshold.achievable_alpha_floor,
            "low_confidence": threshold.low_confidence,
            "threshold": threshold.threshold,
            "excluded": False,
        }],
        columns=_FAR_TABLE_COLUMNS,
    )
    scores_df = pd.DataFrame(
        {
            "window": scoring_windows,
            "label": [_POOLED_LABEL] * n_scored,
            "score": scores,
            "p_value": p_vals,
            "alarm": alarms,
        },
        columns=_SCORES_COLUMNS,
    )

    # Tie-break identical to rowii.anomaly.sweep.scores_and_candidates: p-value
    # ascending, then score DESCENDING (surfaces the more extreme reading first among
    # windows tied at the achievable-minimum p-value), then window ascending.
    top_order = np.lexsort((scoring_windows, -scores, p_vals))[:top_k]
    candidates_df = pd.DataFrame(
        {
            "window": scoring_windows[top_order],
            "label": [_POOLED_LABEL] * len(top_order),
            "score": scores[top_order],
            "p_value": p_vals[top_order],
            "rank": np.arange(1, len(top_order) + 1, dtype=np.int64),
        },
        columns=_CANDIDATES_COLUMNS,
    )

    return SweepResult(far_table=far_table, scores=scores_df, candidates=candidates_df)


def _cross_day_valid_mask(
    prepared: PreparedRun, run: Run, index: RecordingIndex, cfg: Config, labels_mode: str
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """One day's effective valid mask + SCADA context for cross-day (module docstring:
    `--labels` narrows ELIGIBILITY only, never selects a per-label reference here).
    """
    scada = _load_run_scada(prepared, run, index)
    if labels_mode != "gt":
        return prepared.valid_mask, scada
    if scada is None:
        return np.zeros_like(prepared.valid_mask), None
    state = _gt_state_labels(scada, cfg)
    valid_mask: np.ndarray = prepared.valid_mask & (state != "unknown")
    return valid_mask, scada


# ---------------------------------------------------------------------------
# Cross-day-per-state sweep (package-2 Task 3, spec D2 -- see module docstring)
# ---------------------------------------------------------------------------


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
    """Per-state cross-day sweep under DETECTOR TRANSFER (package-2 spec D2): fit day
    A's detector, per-state references (from A's own fit-part), and per-state
    conformal thresholds (from A's own conformal-part) -- exactly `run_sweep`'s own
    per-state machinery, reused here via the now-public row builders (Task 3 Step 1)
    -- then score day B's windows grouped by their PREDICTED (transferred) state
    (`_apply_detector_labels`). Runtime-honest: no GT anywhere in this function; the
    conditioning key is day A's cluster id (module docstring). Returns the
    `SweepResult` plus day B's full-length predicted labels (`_INVALID_LABEL` on
    invalid windows, same convention as `_detected_labels`) for the GT-diagnostic view
    (`_far_by_true_state`).

    Day A's own split collapses to fit/conformal only (`split_by_segments(prepared_a.
    segment_ids, valid_a, 0.5, _CROSS_DAY_SEED)`, the SAME single call `_cross_day_
    sweep` makes for its own day A) -- day A never contributes scoring windows here
    either. Day B contributes ALL of its own eligible (`valid_b`) windows to scoring.

    Args:
        prepared_a: Day A's `PreparedRun` (detector fit + per-state reference source).
        valid_a: Day A's valid mask (always `prepared_a.valid_mask` unchanged -- this
            protocol is detected-labels only, module docstring; `main`'s `parser.error`
            guard rejects `--labels gt` before this function is ever called).
        prepared_b: Day B's `PreparedRun` (scoring target).
        valid_b: Day B's valid mask, same convention as `valid_a`.
        rowii_cfg: Project configuration -- `rowii_cfg.detect` parameterizes day A's
            `FittedDetector.fit` (named to avoid shadowing `sweep_cfg`, the per-sweep
            `SweepConfig` built below).
        scorer_name: Any scorer name `_make_scorer` accepts.
        alpha: Nominal false-alarm rate.
        top_k: Candidate register size.

    Raises:
        ValueError: if day A's eligible windows cannot form a non-empty fit/conformal
            split (`split_by_segments`), or day B has zero eligible windows to score.
    """
    labels_a, detector = _detected_labels_and_detector(prepared_a, rowii_cfg)

    split = split_by_segments(prepared_a.segment_ids, valid_a, 0.5, _CROSS_DAY_SEED)
    fit_windows, conformal_windows = split.calibration_windows, split.scoring_windows

    refs = build_references(prepared_a.features, labels_a, fit_windows)

    labels_b = _apply_detector_labels(prepared_b, detector)
    scoring_windows = np.flatnonzero(valid_b).astype(np.int64)
    if scoring_windows.size == 0:
        raise ValueError("cross-day-per-state: day B has zero eligible windows")

    # `scorer_name` is a plain `str` here (matching `_cross_day_sweep`'s own existing
    # parameter type, and `_make_scorer`'s runtime-validated signature below) rather
    # than the `ScorerName` Literal `SweepConfig.scorer` declares -- the mismatch is
    # real (an arbitrary string could reach this call), so the ignore is deliberate,
    # not a blanket suppression: `_make_scorer` already raises `ValueError` on an
    # unrecognised name wherever it is actually called with one, same as within-day.
    sweep_cfg = SweepConfig(alpha=alpha, top_k=top_k, scorer=scorer_name)  # type: ignore[arg-type]

    all_labels = sorted(
        set(refs.references) | set(refs.excluded)
        | set(np.unique(labels_b[scoring_windows]).tolist())
    )

    far_rows: list[FarRow] = []
    score_rows: list[Any] = []
    candidate_rows: list[Any] = []
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
            far_row_scored(
                label, sweep_cfg, threshold, int(label_scoring.shape[0]), int(alarms.sum())
            )
        )
        new_scores, new_cands = scores_and_candidates(
            label, label_scoring, scores, p_vals, alarms, top_k
        )
        score_rows.extend(new_scores)
        candidate_rows.extend(new_cands)

    # Same calling convention `run_sweep` itself uses (sweep.py's `far_row_aggregate`
    # docstring): `far_rows` here still holds ONLY this sweep's own per-label rows,
    # never a previously-appended aggregate (orchestrator resolution 3).
    far_rows.append(far_row_aggregate(far_rows, sweep_cfg))

    far_table = pd.DataFrame([asdict(r) for r in far_rows], columns=_FAR_TABLE_COLUMNS)
    scores_df = pd.DataFrame([asdict(r) for r in score_rows], columns=_SCORES_COLUMNS)
    candidates_df = pd.DataFrame(
        [asdict(r) for r in candidate_rows], columns=_CANDIDATES_COLUMNS
    )
    return SweepResult(far_table=far_table, scores=scores_df, candidates=candidates_df), labels_b


def _far_by_true_state(scores_df: pd.DataFrame, gt_states: np.ndarray) -> pd.DataFrame:
    """Alarm rate of `_cross_day_per_state_sweep`'s scored day-B windows, grouped by
    their TRUE (SCADA) state instead of the PREDICTED one -- the GT-diagnostic view
    (module docstring, spec D2): never part of the runtime path (no GT anywhere in
    `_cross_day_per_state_sweep` itself), a side-by-side comparison only.

    Args:
        scores_df: `_cross_day_per_state_sweep`'s own `SweepResult.scores` (raw, not
            the `_write_sweep_outputs`-coerced copy -- `window`/`alarm` are read at
            their original int/bool dtypes).
        gt_states: Day B's full-length `(W,)` GT state-name array (`_gt_state_labels`),
            aligned with `scores_df["window"]`'s indices (day B's OWN window space,
            never day A's).

    Returns:
        One row per distinct TRUE state seen among the scored windows, columns
        `true_state, n_scored, n_alarms, realized_far` -- `realized_far` is NaN for a
        state with zero scored windows (never occurs here in practice, since every row
        of *scores_df* already has some true state; kept for the same 0/0-safety
        `rowii.anomaly.sweep.far_row_scored`'s sibling rows use).
    """
    windows = scores_df["window"].to_numpy()
    alarms = scores_df["alarm"].to_numpy()
    states = gt_states[windows]
    rows: list[dict[str, object]] = []
    for state in sorted(np.unique(states).tolist()):
        mask = states == state
        n = int(mask.sum())
        rows.append({
            "true_state": state, "n_scored": n,
            "n_alarms": int(alarms[mask].sum()),
            "realized_far": float(alarms[mask].sum()) / n if n else math.nan,
        })
    return pd.DataFrame(rows, columns=["true_state", "n_scored", "n_alarms", "realized_far"])


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------


def _within_day_out_dir(
    results_root: Path, run_name: str, variant: str, labels_mode: str,
    conditioning: str, scorer: str, *, states: int | None = None,
) -> Path:
    """`results/step2/within-day/<run>/<variant>-<labels>[-k<states>]/<conditioning>-
    <scorer>/` -- `states` (package-3 Task 6, `--states`) appends a `-k<states>`
    suffix to the variant-labels segment ONLY when given (`fusion-detected-k8`), so a
    non-default conditioning-granularity run never collides with -- or overwrites --
    the default-K layout. `None` (every call without `--states`) reproduces the
    pre-Task-6 path byte-for-byte.
    """
    k_suffix = f"-k{states}" if states is not None else ""
    return (
        results_root / "step2" / "within-day" / run_name
        / f"{variant}-{labels_mode}{k_suffix}" / f"{conditioning}-{scorer}"
    )


def _cross_day_out_dir(
    results_root: Path, variant: str, labels_mode: str, day_a: str, day_b: str, scorer: str
) -> Path:
    return (
        results_root / "step2" / "cross-day" / f"{variant}-{labels_mode}"
        / f"{day_a}__to__{day_b}" / f"{scorer}-pooled"
    )


def _cross_day_per_state_out_dir(
    results_root: Path, variant: str, day_a: str, day_b: str, scorer: str
) -> Path:
    """`results/step2/cross-day-per-state/<day_a>--to--<day_b>/<variant>-<scorer>/`
    (module docstring's "Output layout" section) -- DOUBLE-DASH `--to--`, deliberately
    different from `_cross_day_out_dir`'s `__to__`, so a pair directory's own name
    alone already distinguishes the two protocols' outputs on disk. No `<labels_mode>`
    segment (always "detected", `--labels gt` is rejected outright) and no separate
    conditioning segment (always "per-state", the only remaining axis is `scorer`).
    """
    return (
        results_root / "step2" / "cross-day-per-state" / f"{day_a}--to--{day_b}"
        / f"{variant}-{scorer}"
    )


# ---------------------------------------------------------------------------
# far_table.csv / scores.parquet / candidates.md
# ---------------------------------------------------------------------------

_SCADA_CONTEXT_LABELS: tuple[str, ...] = ("P", "rpm", "KS", "Q")
_SCADA_SOURCE_CHANNELS: dict[str, str] = {
    "P": "power", "rpm": "speed", "KS": "ks_valve", "Q": "reactive",
}
"""Candidate-table context columns -> `rowii.scada.labels.load_scada_window_means`'s own
column names (design spec §2: "SCADA context columns [P, rpm, KS, Q]")."""


def _scada_context_row(scada: pd.DataFrame | None, window: int) -> dict[str, str]:
    """P/rpm/KS/Q values for one candidate window, formatted for markdown -- `"n/a"`
    both when this run has no Betriebsdaten at all (`scada is None`, spec's literal
    "else n/a") and when the specific value is itself NaN for that window.
    """
    if scada is None or window >= len(scada):
        return dict.fromkeys(_SCADA_CONTEXT_LABELS, "n/a")
    row = scada.iloc[window]
    result: dict[str, str] = {}
    for label, channel in _SCADA_SOURCE_CHANNELS.items():
        value = row[channel]
        result[label] = "n/a" if pd.isna(value) else f"{float(value):.3f}"
    return result


def _group_candidates_by_label(candidates: pd.DataFrame) -> list[tuple[object, pd.DataFrame]]:
    """`(label, sub_df)` pairs, one per distinct label in *candidates*, sorted by label
    (never a mix of int/str within one sweep -- `rowii.anomaly.sweep._validate_labels`),
    rows within each group ordered by ascending rank.
    """
    labels = sorted(candidates["label"].unique().tolist())
    return [
        (label, candidates[candidates["label"] == label].sort_values("rank"))
        for label in labels
    ]


def _candidates_markdown(
    candidates: pd.DataFrame, grid: WindowGrid, scada: pd.DataFrame | None, top_k: int
) -> str:
    """`candidates.md` body: top-`top_k` rows PER LABEL with UTC time, score, p-value,
    SCADA context, and a blank `assessment` column for the human review pass (design
    spec §2).
    """
    lines = [f"# Anomaly candidates (top-{top_k} per label)", ""]
    if candidates.empty:
        lines.append("No candidates (every label excluded, or zero scoring windows).")
        lines.append("")
        return "\n".join(lines)

    edges = grid.edges_ns()
    for label, sub in _group_candidates_by_label(candidates):
        lines.append(f"## Label `{label}`")
        lines.append("")
        lines.append(
            "| rank | window | utc_time | score | p_value | P | rpm | KS | Q | assessment |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            window = int(r["window"])
            ts = pd.Timestamp(int(edges[window]), unit="ns", tz="UTC")
            ctx = _scada_context_row(scada, window)
            lines.append(
                f"| {int(r['rank'])} | {window} | {ts.isoformat()} | "
                f"{float(r['score']):.6g} | {float(r['p_value']):.6g} | {ctx['P']} | "
                f"{ctx['rpm']} | {ctx['KS']} | {ctx['Q']} | |"
            )
        lines.append("")
    return "\n".join(lines)


def _write_sweep_outputs(
    out_dir: Path, result: SweepResult, grid: WindowGrid, scada: pd.DataFrame | None, top_k: int
) -> None:
    """Write `far_table.csv`, `scores.parquet`, `candidates.md` for one combo.

    Convention (S6 review finding): labels are strings in ALL THREE persisted
    artifacts, coerced right here before anything hits disk. `conditioning="per-
    state"` sweeps carry `run_sweep`'s own aggregate `label="pooled"` row (`rowii.
    anomaly.sweep` module docstring point 4) alongside int cluster-id rows, which makes
    the raw `far_table["label"]` column an OBJECT column mixing Python `int` and `str`
    -- `to_csv`/`read_csv` then round-trips that column as all-string (pandas cannot
    partially infer a numeric dtype once any one value fails conversion), while
    `scores["label"]` (whose rows never include the "pooled" aggregate) round-trips
    `scores.parquet` at its ORIGINAL int64 dtype, unchanged. Left alone, a caller who
    re-loads both files and does `pd.merge(scores, far_table, on="label")` hits a dtype
    mismatch even though the two files describe exactly the same labels. Coercing all
    three DataFrames' label columns to `str` before writing keeps every persisted
    artifact mutually consistent regardless of round-trip -- the "pooled" row makes
    mixed dtypes unavoidable otherwise, since a real label is an int (detected cluster
    id) or a str (GT state name) but "pooled" is always a str.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    far_table = result.far_table.copy()
    far_table["label"] = far_table["label"].astype(str)
    scores_df = result.scores.copy()
    scores_df["label"] = scores_df["label"].astype(str)
    candidates_df = result.candidates.copy()
    candidates_df["label"] = candidates_df["label"].astype(str)

    far_table.to_csv(out_dir / "far_table.csv", index=False)
    scores_df.to_parquet(out_dir / "scores.parquet", engine="pyarrow", index=False)
    (out_dir / "candidates.md").write_text(
        _candidates_markdown(candidates_df, grid, scada, top_k)
    )


# ---------------------------------------------------------------------------
# Candidate register (append-only)
# ---------------------------------------------------------------------------

_REGISTER_HEADER = """# Step-2 Anomaly Candidate Register

Append-only log of anomaly candidates surfaced by the Step-2 mode-conditioned sweep
(`scripts/run_step2.py`), plus externally reported candidates kept here for
cross-reference only. Every entry carries an explicit **source** (provenance) and an
**assessment** status. No externally sourced value (score, threshold, timing precision,
...) is ever adopted into our own computation -- external entries are comparison-only,
verified independently against our own sweeps where possible (design spec
`docs/superpowers/specs/2026-07-09-step2-mode-conditioned-ad-design.md` §1-2).

## External candidates (partner-reported, provenance labeled)

1. **Pre-start filling-valve (Fuelldüse) sound**, observed before machine start.
   - source: partner slide deck, deck-v3 p.16 (read-only reference)
   - assessment: operator-confirmed normal (partner); to cross-check in our sweeps

## Our sweeps

"""

_REGISTER_HEADER_FIRST_LINE = _REGISTER_HEADER.splitlines()[0]
"""`_REGISTER_HEADER`'s first line -- `_append_candidate_register`'s repair guard reads
just this one line back to confirm a previous header write actually completed (S6
review finding), rather than trusting `path.exists()` alone, which stays `True` even
for a file truncated mid-write of the header itself."""


# ---------------------------------------------------------------------------
# Shared crash-safety helper for append-only artifacts (candidate register +
# summary.csv, below)
# ---------------------------------------------------------------------------


def _quarantine_corrupt_file(path: Path, reason: str) -> None:
    """Rename *path* aside to `<name>.corrupt-<UTC-timestamp>` (NEVER delete) and log a
    warning -- shared recovery step for `_append_candidate_register`/
    `_append_summary_row` when a previously-written shared artifact turns out to be
    unreadable or fails its own schema/header check (crash mid-write from a killed
    prior invocation, disk full, ...). The caller treats *path* as absent afterwards,
    same as if this were the very first invocation to touch it.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    corrupt_path = path.with_name(f"{path.name}.corrupt-{timestamp}")
    logger.warning(
        "run_step2: %s appears corrupt (%s) -- moving aside to %s and continuing as if "
        "absent (never deleted)",
        path, reason, corrupt_path,
    )
    path.rename(corrupt_path)


def _register_path(results_root: Path) -> Path:
    return results_root / "step2" / "candidate_register.md"


def _register_section_markdown(
    run_name: str, variant: str, labels_mode: str, conditioning: str, scorer: str,
    alpha: float, candidates: pd.DataFrame, grid: WindowGrid, scada: pd.DataFrame | None,
    *, states: int | None = None,
) -> str:
    """One append-only register section for a (run, variant, combo) -- the same top-k-
    per-label content as that combo's own `candidates.md` (module docstring), with
    `source`/`assessment` provenance columns appended.

    `states` (package-3 Task 6 follow-up): the section header's `<variant>-<labels>`
    part carries the SAME `-k<states>` suffix `_within_day_out_dir` puts on the combo
    directory, and only under the same condition (`--states` given) -- without it, a
    conditioning-granularity sweep (K=4/8/12 on one run/variant, package-3 T7) would
    append several sections with IDENTICAL headers to this append-only, human-facing
    review artifact, distinguishable only by position. `None` reproduces the prior
    header byte-for-byte.
    """
    k_suffix = f"-k{states}" if states is not None else ""
    lines = [
        f"### {run_name} / {variant}-{labels_mode}{k_suffix} / {conditioning}-{scorer} "
        f"(alpha={alpha})",
        "",
    ]
    if candidates.empty:
        lines.append("No candidates (every label excluded, or zero scoring windows).")
        lines.append("")
        return "\n".join(lines) + "\n"

    edges = grid.edges_ns()
    lines.append(
        "| rank | label | window | utc_time | score | p_value | P | rpm | KS | Q | "
        "source | assessment |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, sub in _group_candidates_by_label(candidates):
        for _, r in sub.iterrows():
            window = int(r["window"])
            ts = pd.Timestamp(int(edges[window]), unit="ns", tz="UTC")
            ctx = _scada_context_row(scada, window)
            lines.append(
                f"| {int(r['rank'])} | {label} | {window} | {ts.isoformat()} | "
                f"{float(r['score']):.6g} | {float(r['p_value']):.6g} | {ctx['P']} | "
                f"{ctx['rpm']} | {ctx['KS']} | {ctx['Q']} | own sweep | unreviewed |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def _register_has_intact_header(path: Path) -> bool:
    """Whether *path* starts with `_REGISTER_HEADER_FIRST_LINE` -- cheap enough to check
    on every append (one `readline()`) and the signal `_append_candidate_register` uses
    instead of `path.exists()` alone: a header truncated mid-write (crash, disk full, a
    killed prior invocation) still makes `path.exists()` `True`, but its first line
    would then be a truncated prefix of (or otherwise not equal to) the real header's
    first line. Any read failure (missing file, undecodable bytes) also counts as "not
    intact" -- the caller's next step either way is to (re)write a fresh header.
    """
    try:
        with path.open(encoding="utf-8") as f:
            first_line = f.readline().rstrip("\n")
    except (OSError, UnicodeDecodeError):
        return False
    return first_line == _REGISTER_HEADER_FIRST_LINE


def _write_register_header(path: Path) -> None:
    """Atomic-ish header write: build the full header in a `.tmp` sibling, then
    `os.replace` it into place, so a crash mid-write never leaves a PARTIAL header
    under the real path (same tmp-then-replace pattern `_append_summary_row` uses for
    `summary.csv`, S6 review finding)."""
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(_REGISTER_HEADER, encoding="utf-8")
    os.replace(tmp_path, path)


def _append_candidate_register(
    results_root: Path, run_name: str, variant: str, labels_mode: str,
    conditioning: str, scorer: str, alpha: float, candidates: pd.DataFrame,
    grid: WindowGrid, scada: pd.DataFrame | None, *, states: int | None = None,
) -> None:
    """`states`: forwarded to `_register_section_markdown`'s section-header suffix
    (its docstring) -- only `_run_within_day_for_run` ever passes it; the cross-day
    call sites keep the default `None` (`--states` is rejected for their protocols
    up front, `main`'s parser.error guard)."""
    path = _register_path(results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _write_register_header(path)
    elif not _register_has_intact_header(path):
        _quarantine_corrupt_file(path, "candidate_register.md header missing or truncated")
        _write_register_header(path)
    section = _register_section_markdown(
        run_name, variant, labels_mode, conditioning, scorer, alpha, candidates, grid,
        scada, states=states,
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(section)


# ---------------------------------------------------------------------------
# summary.csv (append-only)
# ---------------------------------------------------------------------------

_SUMMARY_COLUMNS: tuple[str, ...] = (
    "run", "protocol", "variant", "labels", "conditioning", "scorer", "alpha",
    "per_label_count", "pooled_realized_far", "mean_per_state_far", "n_low_confidence",
    "notes",
)

_SUMMARY_COLUMNS_LEGACY: tuple[str, ...] = tuple(c for c in _SUMMARY_COLUMNS if c != "protocol")
"""`_SUMMARY_COLUMNS` before package 2 added `protocol` as the 2nd column --
`_read_summary_csv_or_none` still accepts a `summary.csv` written in this older shape
(backfilling `protocol` onto it, never quarantining it as corrupt) so appending to a
file left over from before this package never produces a ragged CSV."""


@dataclass(frozen=True)
class _SummaryRow:
    """One `results/step2/summary.csv` row -- see module docstring for column semantics."""

    run: str
    protocol: str
    variant: str
    labels: str
    conditioning: str
    scorer: str
    alpha: float
    per_label_count: int
    pooled_realized_far: float
    mean_per_state_far: float
    n_low_confidence: int
    notes: str


def _summary_far_metrics(
    far_table: pd.DataFrame, conditioning: str
) -> tuple[int, float, float, int]:
    """`(per_label_count, pooled_realized_far, mean_per_state_far, n_low_confidence)`
    from one WITHIN-DAY `run_sweep` call's `far_table` (module docstring) -- NOT used
    for cross-day, whose single-row far_table IS the pooled result already
    (`_cross_day_summary_row` reads it directly).

    `conditioning="per-state"`: `run_sweep` already appends an aggregate `label=
    "pooled"` row (`rowii.anomaly.sweep` module docstring point 4) -- read straight
    from it. `conditioning="pooled"`: no such row exists (every per-label row already
    shares one scorer/threshold), so it is recomputed here with the identical
    alarms-over-scored-count arithmetic, restricted to rows that were actually scored
    (`n_scored` not NaN).
    """
    is_aggregate = far_table["label"] == _POOLED_LABEL
    per_label = far_table[~is_aggregate]
    per_label_count = int(len(per_label))
    n_low_confidence = int(per_label["low_confidence"].sum())
    real_far = per_label["realized_far"].dropna()
    mean_per_state_far = float(real_far.mean()) if len(real_far) else math.nan

    if conditioning == "per-state":
        aggregate = far_table[is_aggregate]
        pooled_realized_far = (
            float(aggregate["realized_far"].iloc[0]) if len(aggregate) else math.nan
        )
    else:
        scored = per_label[per_label["n_scored"].notna()]
        total_scored = float(scored["n_scored"].sum())
        total_alarms = float(scored["n_alarms"].sum())
        pooled_realized_far = (total_alarms / total_scored) if total_scored > 0 else math.nan

    return per_label_count, pooled_realized_far, mean_per_state_far, n_low_confidence


def _summary_row(
    run_name: str, variant: str, labels_mode: str, conditioning: str, scorer: str,
    alpha: float, far_table: pd.DataFrame, *, states: int | None = None,
) -> _SummaryRow:
    """`states` (package-3 Task 6, `--states`): mirrors `_within_day_out_dir`'s own
    `-k<states>` suffix convention, applied to the `variant` column ONLY when given
    (`variant="fusion-k8"`) -- the same "no --states -> byte-compatible" guarantee, so
    a summary row stays traceable to (without literally re-deriving) its combo's own
    out-dir even though `variant`/`labels` remain two independent columns."""
    per_label_count, pooled_far, mean_far, n_low_conf = _summary_far_metrics(
        far_table, conditioning
    )
    k_suffix = f"-k{states}" if states is not None else ""
    return _SummaryRow(
        run=run_name, protocol="within-day", variant=f"{variant}{k_suffix}",
        labels=labels_mode, conditioning=conditioning, scorer=scorer, alpha=alpha,
        per_label_count=per_label_count, pooled_realized_far=pooled_far,
        mean_per_state_far=mean_far, n_low_confidence=n_low_conf, notes="",
    )


def _cross_day_summary_row(
    day_a: str, day_b: str, variant: str, labels_mode: str, scorer: str,
    alpha: float, far_table: pd.DataFrame,
) -> _SummaryRow:
    """Cross-day's summary row: `far_table` always has exactly one `"pooled"` row
    (`_cross_day_sweep`) -- read it directly rather than through
    `_summary_far_metrics` (which is built around within-day's "aggregate row on top
    of per-label rows" shape, not applicable here). `notes="cross-day pooled"` per the
    spec's literal wording.
    """
    row = far_table.iloc[0]
    far = float(row["realized_far"]) if pd.notna(row["realized_far"]) else math.nan
    return _SummaryRow(
        run=f"{day_a}__to__{day_b}", protocol="cross-day", variant=variant,
        labels=labels_mode, conditioning="pooled", scorer=scorer, alpha=alpha,
        per_label_count=1, pooled_realized_far=far, mean_per_state_far=far,
        n_low_confidence=int(bool(row["low_confidence"])), notes="cross-day pooled",
    )


def _cross_day_per_state_summary_row(
    day_a: str, day_b: str, variant: str, scorer: str, alpha: float, far_table: pd.DataFrame,
) -> _SummaryRow:
    """cross-day-per-state's summary row: unlike `_cross_day_summary_row`'s single
    pooled row, `_cross_day_per_state_sweep`'s `far_table` has the SAME shape a
    within-day `conditioning="per-state"` sweep produces -- one row per predicted
    state plus a `run_sweep`-style aggregate `"pooled"` row (from the now-public
    `far_row_aggregate`, Task 3 Step 1) -- so this reuses `_summary_far_metrics`
    directly rather than `_cross_day_summary_row`'s single-row shortcut.
    `labels`/`conditioning` are both constants for this protocol (`"detected"`/
    `"per-state"` -- `--labels gt` is rejected outright, module docstring).
    """
    per_label_count, pooled_far, mean_far, n_low_conf = _summary_far_metrics(
        far_table, "per-state"
    )
    return _SummaryRow(
        run=f"{day_a}--to--{day_b}", protocol="cross-day-per-state", variant=variant,
        labels="detected", conditioning="per-state", scorer=scorer, alpha=alpha,
        per_label_count=per_label_count, pooled_realized_far=pooled_far,
        mean_per_state_far=mean_far, n_low_confidence=n_low_conf,
        notes="cross-day-per-state transfer",
    )


def _infer_legacy_protocol(run_field: str) -> str:
    """Backfill inference for one legacy (pre-`protocol`-column) summary row's `run`
    field (`_read_summary_csv_or_none`): `"cross-day"` iff it matches
    `_cross_day_summary_row`'s own `f"{day_a}__to__{day_b}"` pair encoding -- the ONLY
    legacy row-builder that ever puts `"__to__"` into `run` (`_summary_row` always
    writes a single run's own bare name) -- else `"within-day"`. `cross-day-per-state`
    is never inferred here: that protocol was born together with the `protocol` column
    itself (this package), so a file lacking the column could never contain one of its
    rows in the first place.
    """
    return "cross-day" if "__to__" in run_field else "within-day"


def _read_summary_csv_or_none(summary_path: Path) -> pd.DataFrame | None:
    """`pd.read_csv(summary_path)`, or `None` if the file does not exist, fails to
    parse, or parses to neither a recognised column schema (S6 review finding:
    `summary.csv` is a shared, append-only artifact a killed prior invocation can leave
    mid-write). A corrupt file is quarantined via `_quarantine_corrupt_file` (never
    deleted, never silently overwritten); the caller then treats this exactly like
    "does not exist yet".

    Deliberately narrow on which parse failures count as "corrupt, recover":
    `pd.errors.ParserError` (malformed CSV token stream, e.g. an unbalanced quote),
    `pd.errors.EmptyDataError` (zero-byte file), `UnicodeDecodeError` (binary garbage /
    wrong encoding from a mid-write crash). A truncated write does not always raise one
    of these, though: losing only the LAST data row's trailing columns parses silently
    (pandas fills missing trailing fields with NaN), and losing part of the HEADER line
    itself instead changes which columns get parsed at all, again with no exception.
    Neither is caught by the exceptions above, so the parsed result's columns are also
    checked explicitly against TWO known schemas: `_SUMMARY_COLUMNS` (current) is
    returned as-is; `_SUMMARY_COLUMNS_LEGACY` (pre-package-2, missing `protocol`) is
    BACKFILLED -- `protocol` is inserted as the 2nd column, one value per row inferred
    from that row's own `run` field (`_infer_legacy_protocol`) -- and returned, never
    quarantined, so appending a fresh row (which always carries an explicit `protocol`)
    to an old-schema file never produces a ragged CSV. Any OTHER column set is treated
    as corrupt, same as before package 2.
    """
    if not summary_path.exists():
        return None
    try:
        existing = pd.read_csv(summary_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        _quarantine_corrupt_file(summary_path, f"{type(exc).__name__}: {exc}")
        return None
    if list(existing.columns) == list(_SUMMARY_COLUMNS_LEGACY):
        existing.insert(1, "protocol", existing["run"].map(_infer_legacy_protocol))
        return existing
    if list(existing.columns) != list(_SUMMARY_COLUMNS):
        _quarantine_corrupt_file(
            summary_path, f"unexpected columns {list(existing.columns)!r} (truncated write?)"
        )
        return None
    return existing


def _append_summary_row(results_root: Path, row: _SummaryRow) -> None:
    """Append one row to `summary.csv`, recovering from a corrupt prior write
    (`_read_summary_csv_or_none`) and writing crash-safely itself: the combined frame
    goes to a `summary.csv.tmp` sibling first, then `os.replace`s the real path, so a
    crash mid-write of THIS call can never leave a partially-written `summary.csv`
    behind either (S6 review finding)."""
    summary_path = results_root / "step2" / "summary.csv"
    row_df = pd.DataFrame([vars(row)], columns=_SUMMARY_COLUMNS)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_summary_csv_or_none(summary_path)
    combined = pd.concat([existing, row_df], ignore_index=True) if existing is not None else row_df
    tmp_path = summary_path.with_name(summary_path.name + ".tmp")
    combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, summary_path)


# ---------------------------------------------------------------------------
# Score-level fusion view (Task 5, spec D5) -- Fisher/Tippett p-value combination of
# the fusion variant's audio and vibration branches, re-calibrated via split conformal
# (rowii.anomaly.fusion's own module docstring carries the full statistical argument).
# ---------------------------------------------------------------------------

_SCORE_FUSION_SEED = 7
"""Top-level split seed for `_run_score_fusion_view` -- matches `SweepConfig.seed`'s
own default (7); the nested fit/conformal split below uses `_SCORE_FUSION_SEED + 1`
(=8), mirroring `run_sweep`'s own `cfg.seed + 1` convention exactly (orchestrator
resolution 4) so a `--score-fusion` run and a same-run `run_sweep` call partition one
`PreparedRun`'s windows identically."""

_SCORE_FUSION_MIN_REF = 20
"""Minimum fit-side windows a label needs before `_run_score_fusion_view` fits real
per-branch scorers for it -- matches `SweepConfig.min_ref`'s own default, and gates on
the SAME window COUNT `rowii.anomaly.references.build_references` gates on (both
branches share the same window set, just different feature columns)."""

_SCORE_FUSION_COLUMNS: tuple[str, ...] = (
    "rule", "label", "n_calibration", "n_scored", "n_alarms", "realized_far",
    "low_confidence",
)

_SCORE_FUSION_RULES: tuple[str, ...] = ("fisher", "tippett", "audio-only", "vib-only")
"""The four `far_table_scorefusion.csv` rules (orchestrator resolution 4): the two
COMBINED rules plus the two single-branch baselines, all evaluated through the exact
same p-value-then-recalibrate pipeline (`_score_fusion_statistic`) for a fair,
apples-to-apples comparison."""


@dataclass
class _ScoreFusionRow:
    """Mutable row builder for `far_table_scorefusion.csv` -- one instance per
    (label, rule) pair, mirroring `rowii.anomaly.sweep.FarRow`'s own "converted to a
    plain dict via `dataclasses.asdict` at the very end" pattern. Deliberately
    narrower than `FarRow`: this view carries no `threshold`/`nominal_alpha`/
    `achievable_alpha_floor`/`excluded` columns (orchestrator resolution 4's literal
    column list) -- each rule's own threshold lives on a different, incomparable
    scale (a Fisher chi-square-shaped statistic vs. a Tippett `[0, 1)` statistic), so
    comparing rules only ever needs the REALIZED outcome, not the threshold value
    that produced it."""

    rule: str
    label: int | str
    n_calibration: float
    n_scored: float
    n_alarms: float
    realized_far: float
    low_confidence: bool


def _score_fusion_statistic(rule: str, p_a: np.ndarray, p_v: np.ndarray) -> np.ndarray:
    """The one real-valued, higher-is-more-anomalous combined statistic for *rule*,
    from a label's own branch p-value arrays (*p_a*/*p_v*, aligned, same windows).

    `"audio-only"`/`"vib-only"` route through this SAME p-value machinery as
    `"fisher"`/`"tippett"` rather than the branch's raw score directly (orchestrator
    resolution 4's "through the same conformal path") -- `1.0 - p` is `rowii.anomaly.
    fusion.tippett_statistic` degenerated to a single branch (`tippett_statistic(p, p)
    == 1.0 - p`), so all four rules are, structurally, "some deterministic reduction
    of `(p_a, p_v)` to one number", differing only in how much of the OTHER branch's
    evidence they use -- both (fisher), the stronger one (tippett), or neither
    (audio-only/vib-only).

    Raises:
        ValueError: if `rule` is not one of `_SCORE_FUSION_RULES`.
    """
    if rule == "fisher":
        return fisher_statistic(p_a, p_v)
    if rule == "tippett":
        return tippett_statistic(p_a, p_v)
    if rule == "audio-only":
        audio_only: np.ndarray = 1.0 - p_a
        return audio_only
    if rule == "vib-only":
        vib_only: np.ndarray = 1.0 - p_v
        return vib_only
    raise ValueError(f"rule must be one of {_SCORE_FUSION_RULES!r}, got {rule!r}")


def _score_fusion_low_confidence_rows(label: int | str) -> list[_ScoreFusionRow]:
    """All four `_SCORE_FUSION_RULES` rows for *label*, NaN metrics + `low_confidence
    = True` -- `_score_fusion_rows_for_label`'s shared fallback when *label* cannot
    get a real per-branch reference or threshold at all (mirrors `rowii.anomaly.
    sweep.far_row_excluded`'s own "no reliable calibration exists, don't alarm under
    a false promise" convention, adapted to this view's narrower schema)."""
    return [
        _ScoreFusionRow(rule, label, math.nan, math.nan, math.nan, math.nan, True)
        for rule in _SCORE_FUSION_RULES
    ]


def _score_fusion_rows_for_label(
    label: int | str,
    prepared: PreparedRun,
    labels: np.ndarray,
    audio_idx: np.ndarray,
    vib_idx: np.ndarray,
    fit_windows: np.ndarray,
    conformal_windows: np.ndarray,
    scoring_windows: np.ndarray,
    alpha: float,
    scorer_name: str,
) -> list[_ScoreFusionRow]:
    """One `_ScoreFusionRow` per rule (`_SCORE_FUSION_RULES`) for *label* -- the four
    rules share every gating condition below (the same fit/conformal/scoring window
    sets per label), so a label that cannot get a real per-branch reference or
    threshold reports ALL FOUR rules as low-confidence (`_score_fusion_low_confidence_
    rows`) rather than partially succeeding.

    Args:
        label: The state/cluster id (or GT state string) to build rows for.
        prepared: The `fusion`(-beats)-variant `PreparedRun` being scored.
        labels: Per-window labels aligned with `prepared.features`.
        audio_idx: Audio-branch column indices (`rowii.anomaly.fusion.
            split_branch_columns`).
        vib_idx: Vibration-branch column indices, same source.
        fit_windows: This run's fit-side window indices (branch scorers are fit on
            *label*'s own subset of these).
        conformal_windows: This run's conformal-side window indices (branch p-values
            and the combined statistic's threshold are calibrated on *label*'s own
            subset of these).
        scoring_windows: This run's scoring-side window indices (alarms are raised on
            *label*'s own subset of these).
        alpha: Nominal false-alarm rate for `calibrate`.
        scorer_name: `"knn"` or `"mahalanobis"` -- fit independently per branch.

    Returns:
        Exactly `len(_SCORE_FUSION_RULES)` rows for *label*.
    """
    label_fit = fit_windows[labels[fit_windows] == label]
    if label_fit.shape[0] < _SCORE_FUSION_MIN_REF:
        return _score_fusion_low_confidence_rows(label)

    label_conformal = conformal_windows[labels[conformal_windows] == label]
    if label_conformal.shape[0] == 0:
        return _score_fusion_low_confidence_rows(label)

    fit_features = prepared.features[label_fit]
    scorer_a = _make_scorer(scorer_name).fit(fit_features[:, audio_idx])
    scorer_v = _make_scorer(scorer_name).fit(fit_features[:, vib_idx])

    conformal_features = prepared.features[label_conformal]
    conformal_scores_a = scorer_a.score(conformal_features[:, audio_idx])
    conformal_scores_v = scorer_v.score(conformal_features[:, vib_idx])
    # LEAVE-ONE-OUT, not p_values(x, x) (review fix, 2026-07-15): the calibration
    # side's p-values must be computed on the same footing as the scoring side's
    # (each window's p-value against a reference EXCLUDING that window), else the
    # combined statistic is not one fixed transform applied to both sides and its
    # calibration/scoring exchangeability breaks -- the self-referential form was
    # measured anti-conservative up to mean FAR ~0.10 at alpha=0.05 (see
    # `rowii.anomaly.conformal.loo_p_values`' docstring for the full derivation and
    # `tests/test_fusion.py`'s multi-regime validity test + its mutant check).
    p_a_conformal = loo_p_values(conformal_scores_a)
    p_v_conformal = loo_p_values(conformal_scores_v)

    label_scoring = scoring_windows[labels[scoring_windows] == label]
    n_scored = int(label_scoring.shape[0])

    if n_scored == 0:
        empty_rows: list[_ScoreFusionRow] = []
        for rule in _SCORE_FUSION_RULES:
            conformal_stat = _score_fusion_statistic(rule, p_a_conformal, p_v_conformal)
            threshold = calibrate(conformal_stat, alpha)
            empty_rows.append(_ScoreFusionRow(
                rule, label, float(threshold.n_calibration), 0.0, 0.0, math.nan,
                threshold.low_confidence,
            ))
        return empty_rows

    scoring_features = prepared.features[label_scoring]
    scoring_scores_a = scorer_a.score(scoring_features[:, audio_idx])
    scoring_scores_v = scorer_v.score(scoring_features[:, vib_idx])
    p_a_scoring = p_values(scoring_scores_a, conformal_scores_a)
    p_v_scoring = p_values(scoring_scores_v, conformal_scores_v)

    scored_rows: list[_ScoreFusionRow] = []
    for rule in _SCORE_FUSION_RULES:
        conformal_stat = _score_fusion_statistic(rule, p_a_conformal, p_v_conformal)
        scoring_stat = _score_fusion_statistic(rule, p_a_scoring, p_v_scoring)
        threshold = calibrate(conformal_stat, alpha)
        alarms = scoring_stat > threshold.threshold
        n_alarms = int(alarms.sum())
        scored_rows.append(_ScoreFusionRow(
            rule, label, float(threshold.n_calibration), float(n_scored),
            float(n_alarms), n_alarms / n_scored, threshold.low_confidence,
        ))
    return scored_rows


def _run_score_fusion_view(
    prepared: PreparedRun, labels: np.ndarray, alpha: float, scorer_name: str,
) -> pd.DataFrame:
    """Score-level fusion FAR table for one (run, labels_mode) -- Fisher/Tippett
    p-value combination of the `fusion` variant's audio and vibration branches,
    RE-CALIBRATED with split conformal (`rowii.anomaly.fusion` module docstring;
    design spec D5). Mirrors `rowii.anomaly.sweep.run_sweep`'s own three-way split
    exactly (`_SCORE_FUSION_SEED`=7 top-level, `_SCORE_FUSION_SEED + 1`=8 nested) so a
    `--score-fusion` run and a same-run `run_sweep` call partition *prepared*'s
    windows identically.

    Args:
        prepared: A `fusion`(-beats)-variant `PreparedRun` (caller-guaranteed by the
            `--score-fusion` CLI guard, `main`'s `parser.error` check).
        labels: Per-window labels aligned with `prepared.features`, same convention as
            `run_sweep`'s own `labels` argument.
        alpha: Nominal false-alarm rate target for `calibrate`, shared by every rule.
        scorer_name: `"knn"` or `"mahalanobis"` (`--score-fusion-scorer`) -- the SAME
            scorer type is fit independently on each branch's own columns.

    Returns:
        A DataFrame with columns `rule, label, n_calibration, n_scored, n_alarms,
        realized_far, low_confidence` -- `_SCORE_FUSION_RULES` (`"fisher"`,
        `"tippett"`, `"audio-only"`, `"vib-only"`) rows per label seen in the
        calibration or scoring windows.

    Raises:
        ValueError: propagated from `rowii.anomaly.fusion.split_branch_columns` if
            *prepared*'s feature names do not split cleanly into audio/vibration
            branches; from `split_by_segments` if the three-way split cannot be
            formed (too few segments -- identical failure mode to `run_sweep`
            itself); from `_make_scorer(...).fit(...)` if a branch's own fit-side
            reference is degenerate for *scorer_name* (e.g. a zero-norm row for
            `KnnScorer`'s cosine metric).
    """
    audio_idx, vib_idx = split_branch_columns(prepared.feature_names)

    top_split = split_by_segments(
        prepared.segment_ids, prepared.valid_mask, 0.5, _SCORE_FUSION_SEED
    )
    calibration_windows = top_split.calibration_windows
    scoring_windows = top_split.scoring_windows

    calib_mask = np.zeros(prepared.features.shape[0], dtype=bool)
    calib_mask[calibration_windows] = True
    nested_split = split_by_segments(
        prepared.segment_ids, calib_mask, 0.5, _SCORE_FUSION_SEED + 1
    )
    fit_windows = nested_split.calibration_windows
    conformal_windows = nested_split.scoring_windows

    # Same defensive re-check `run_sweep` makes on its own identically-constructed
    # split (`rowii.anomaly.sweep._assert_three_way_disjoint`'s docstring: trust but
    # verify, the kNN self-scoring hazard).
    _assert_three_way_disjoint(fit_windows, conformal_windows, scoring_windows)

    all_windows = np.concatenate([calibration_windows, scoring_windows])
    all_labels = sorted(np.unique(labels[all_windows]).tolist())

    rows: list[_ScoreFusionRow] = []
    for label in all_labels:
        rows.extend(_score_fusion_rows_for_label(
            label, prepared, labels, audio_idx, vib_idx,
            fit_windows, conformal_windows, scoring_windows, alpha, scorer_name,
        ))

    return pd.DataFrame([asdict(r) for r in rows], columns=list(_SCORE_FUSION_COLUMNS))


_SCORE_FUSION_NOTES = """# Score-level fusion notes

## Guarantee restoration

`far_table_scorefusion.csv`'s `fisher`/`tippett` rows combine the audio and vibration
branches' own conformal p-values into one statistic per window, then RE-CALIBRATE that
combined statistic with the same split-conformal machinery every other Step-2 view
uses (`rowii.anomaly.conformal.calibrate`, held out on the conformal-side windows).
The combination is used only as a deterministic score transform -- never compared
against the classical Fisher/Tippett null distributions, which assume the two branches
are independent. For the re-calibration itself to be valid, the calibration-side
branch p-values are LEAVE-ONE-OUT (`rowii.anomaly.conformal.loo_p_values`: each
calibration window evaluated against a reference excluding that window, the same
footing a scoring window gets); a naive self-referential construction breaks
calibration/scoring exchangeability of the combined statistic and was measured
anti-conservative (mean realized FAR up to ~0.10 at alpha=0.05, n=39, anti-correlated
branches -- review finding, 2026-07-15). With LOO calibration p-values,
exchangeability is restored up to a one-unit p-granularity difference (LOO minimum
p-value 1/n vs scoring-side 1/(n+1)) whose direction is conservative; validated
empirically FOR THE FISHER RULE at mean realized FAR <= alpha (within Monte-Carlo
precision, one-sided alpha + 3*SE bound) across independent, shared-latent-correlated
(rho ~ 0.78), anti-correlated, and identical branches at n in {39, 159}
(`tests/test_fusion.py`'s multi-regime validity test; additionally at n=319 in the
review-time simulation). See `rowii.anomaly.fusion`'s module docstring for the full
argument.

The `tippett` rows carry a NARROWER claim (review round 2): a min-rule combination
cannot be exactly calibrated when the calibration set doubles as the p-value
reference -- the LOO construction is decision-neutral for a min rule (bit-identical
alarms to the self-referential form), so this residual is intrinsic, not fixable by
the LOO switch. Under positively correlated branches Tippett's measured mean realized
FAR carries a small excess: +0.007 absolute at alpha=0.05, n=39, decaying roughly
like 1/n (0.0518 at n=159, 0.0512 at n=319); independent, anti-correlated, and
identical branches measure within alpha + 3*SE. A dedicated p-reference split would
restore exactness but is deliberately not adopted: per-state calibration pools are
the binding resource (package-2 scarcity results, spec D3 -- several states already
sit near the achievability floor, and a third split would push them under it).
Tippett rows are therefore the max-rule CONTRAST to Fisher, carrying this caveat --
not guaranteed-FAR rows; `tests/test_fusion.py` pins the documented excess so a
regression beyond it fails loudly.

## Honesty notes

- `audio-only`/`vib-only` rows go through the EXACT SAME p-value-then-re-calibrate
  pipeline as `fisher`/`tippett` (not a shortcut through the branch's raw score), so
  every row in this table is comparable on equal footing.
- Only the `fisher` rows (and, via the single-branch monotone-cancellation argument,
  the `audio-only`/`vib-only` rows) carry the restored distribution-free FAR claim;
  `tippett` rows carry the documented intrinsic excess under positive branch
  correlation instead (see "Guarantee restoration" above).
- No claim is made that Fisher dominates Tippett or vice versa in general: Fisher
  requires corroboration from both branches (a single very anomalous branch is damped
  by the other branch's own p-value), Tippett fires on the single most extreme branch
  alone -- which is better depends on how the underlying anomaly actually shows up
  across the two sensor modalities, an empirical question per dataset/finding, not a
  property of the method.
- A label whose fit-side window count falls below the same `min_ref=20` floor the rest
  of Step-2 uses (or has zero conformal-side windows) reports `low_confidence=True`
  with NaN metrics for all four rules and never contributes alarms -- the same "do not
  alarm under a false promise" convention as `rowii.anomaly.sweep.far_row_excluded`.
- This view only ever runs for `--variant fusion` under `--protocol within-day`
  (`--score-fusion`'s own CLI guard) -- it needs both audio and vibration feature
  columns on one grid, which only the `fusion`(-beats) variants provide.
"""


def _write_score_fusion_outputs(out_dir: Path, far_table: pd.DataFrame) -> None:
    """Write `far_table_scorefusion.csv` + `scorefusion_notes.md` into *out_dir* (an
    existing within-day combo directory, `_write_sweep_outputs`'s own `out_dir`) --
    `far_table["label"]` is coerced to `str` first, mirroring `_write_sweep_outputs`'s
    own label-dtype convention (detected cluster ids are ints; keeping the column
    homogeneous avoids the same round-trip hazard `test_per_state_far_table_and_
    scores_label_dtypes_merge_cleanly` guards against for the main far_table).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    table = far_table.copy()
    table["label"] = table["label"].astype(str)
    table.to_csv(out_dir / "far_table_scorefusion.csv", index=False)
    (out_dir / "scorefusion_notes.md").write_text(_SCORE_FUSION_NOTES)


# ---------------------------------------------------------------------------
# Majority-ensemble evaluation view (Task 4, design chapter's committed ensemble: a
# majority-voting ensemble of OC-SVM + Isolation Forest + LSTM-AE, "requiring
# agreement before an alarm suppresses false alarms"). Voting is DECISION-level (each
# member keeps its own scorer/reference/threshold; only the ALARM booleans are
# combined), so it lives here in orchestration, not as a `Scorer` itself.
# ---------------------------------------------------------------------------

_ENSEMBLE_MEMBER_FACTORIES: dict[str, Callable[[], Scorer]] = {
    "ocsvm": OcSvmScorer,
    "iforest": IsolationForestScorer,
    "lstmae": LstmAeScorer,
}
"""Module-level test seam (task brief): each factory returns a FRESH, unfitted
`Scorer` for one ensemble member. Tests monkeypatch this whole dict
(`monkeypatch.setattr(run_step2, "_ENSEMBLE_MEMBER_FACTORIES", {...})`) to substitute
a lightweight, torch-free stub for `"lstmae"` -- `LstmAeScorer.fit` needs torch
(`rowii.anomaly.recon` module docstring's lazy-import contract), which the CLI test
suite never requires. Iteration order (`ocsvm`, `iforest`, `lstmae`) is insertion
order (Python dict guarantee) and determines `far_table_ensemble.csv`'s row order
within each label."""

_ENSEMBLE_SEED = 7
"""Top-level split seed for `_run_ensemble_view` -- matches `SweepConfig.seed`'s own
default (7) and `_SCORE_FUSION_SEED`'s identical choice; the nested fit/conformal
split uses `_ENSEMBLE_SEED + 1` (=8), the same `run_sweep`-mirroring convention
`_run_score_fusion_view` already established (its own docstring)."""

_ENSEMBLE_MIN_REF = 20
"""Matches `SweepConfig.min_ref`'s own default and `_SCORE_FUSION_MIN_REF`'s
identical choice -- the minimum fit-side window count a state needs before any
ensemble member gets a real reference for it."""

_ENSEMBLE_VOTE_THRESHOLD = 2
"""Majority rule: an ENSEMBLE alarm requires at least this many of the 3 members
(`len(_ENSEMBLE_MEMBER_FACTORIES)`) to alarm on the same window (design chapter's
committed majority-voting ensemble: "requiring agreement before an alarm suppresses
false alarms")."""

_ENSEMBLE_LABEL = "ENSEMBLE"
"""`member` value for the >=2-of-3 majority-vote row -- uppercase by convention so it
never collides with a real (lowercase) member name from `_ENSEMBLE_MEMBER_FACTORIES`,
mirroring `_POOLED_LABEL`'s comparable role for `run_sweep`'s own aggregate row."""

_ENSEMBLE_COLUMNS: tuple[str, ...] = (
    "member", "label", "n_calibration", "n_scored", "n_alarms", "realized_far",
    "low_confidence",
)


@dataclass
class _EnsembleRow:
    """Mutable row builder for `far_table_ensemble.csv` -- one instance per (label,
    member) pair, plus one per (label, `_ENSEMBLE_LABEL`) majority-vote row, mirroring
    `_ScoreFusionRow`'s own "converted to a plain dict via `dataclasses.asdict` at the
    very end" pattern."""

    member: str
    label: int | str
    n_calibration: float
    n_scored: float
    n_alarms: float
    realized_far: float
    low_confidence: bool


def _ensemble_member_features(
    member_name: str, prepared_variant: PreparedRun, prepared_logmel: PreparedRun
) -> np.ndarray:
    """Which `PreparedRun`'s feature matrix *member_name* fits/scores on: the
    `logmel`-variant's for `"lstmae"` (`rowii.anomaly.recon.LstmAeScorer` needs a
    flattened logmel patch, its own module docstring), the sweep `--variant`'s own
    features for every other member (`"ocsvm"`/`"iforest"`, or any future addition to
    `_ENSEMBLE_MEMBER_FACTORIES`)."""
    if member_name == "lstmae":
        return prepared_logmel.features
    return prepared_variant.features


def _assert_ensemble_grids_match(
    run_name: str, prepared_variant: PreparedRun, prepared_logmel: PreparedRun
) -> None:
    """`--ensemble`'s grid-alignment guard (task brief binding detail 3): the sweep
    variant's and the `logmel` variant's `PreparedRun`s must describe the IDENTICAL
    window grid (`t0_ns`, `window_ns`, `n_windows`) for a shared window INDEX to mean
    the same physical time slot in both -- `_run_ensemble_view`'s single top-level
    split (below) is drawn once from the variant side and used to index BOTH prepared
    runs' feature matrices at the SAME window indices. Structurally, both
    preparations share the primary generator-mic stream for every audio-bearing sweep
    variant (`audio`, `audio-beats`, `fusion`, `fusion-beats`, `logmel` itself --
    `rowii.pipeline._streams_for_variant`), so this should always hold in practice --
    asserted here anyway (trust but verify, mirroring `rowii.anomaly.sweep.
    _assert_three_way_disjoint`'s identical stance on a different structurally-
    guaranteed invariant) rather than assumed, since `--variant vibration` shares NO
    stream with `logmel` at all and could plausibly desync.

    Raises:
        SystemExit: code 2, with a clear message on stderr (parser.error-style; the
            argparse `parser` object itself is out of scope this deep in the call
            stack, so this raises directly rather than routing back through it) -- a
            grid mismatch signals a structural inconsistency this view cannot safely
            paper over, not a per-run/per-combo condition to log-and-skip.
    """
    if prepared_variant.grid == prepared_logmel.grid:
        return
    va, la = prepared_variant.grid, prepared_logmel.grid
    print(
        f"run_step2: --ensemble grid mismatch for run {run_name!r}: sweep-variant "
        f"grid (t0_ns={va.t0_ns}, window_ns={va.window_ns}, n_windows={va.n_windows}) "
        f"!= logmel grid (t0_ns={la.t0_ns}, window_ns={la.window_ns}, "
        f"n_windows={la.n_windows}) -- the ensemble view indexes both PreparedRuns at "
        "the same window indices and needs one shared grid",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _ensemble_low_confidence_rows(label: int | str) -> list[_EnsembleRow]:
    """All 3 members + the `_ENSEMBLE_LABEL` row for *label*, NaN metrics +
    `low_confidence=True` -- mirrors `_score_fusion_low_confidence_rows`'s identical
    fallback for a label that cannot get a real per-member reference or threshold at
    all."""
    names = [*_ENSEMBLE_MEMBER_FACTORIES, _ENSEMBLE_LABEL]
    return [
        _EnsembleRow(name, label, math.nan, math.nan, math.nan, math.nan, True)
        for name in names
    ]


def _ensemble_rows_for_label(
    label: int | str,
    prepared_variant: PreparedRun,
    prepared_logmel: PreparedRun,
    labels: np.ndarray,
    fit_windows: np.ndarray,
    conformal_windows: np.ndarray,
    scoring_windows: np.ndarray,
    alpha: float,
) -> list[_EnsembleRow]:
    """One `_EnsembleRow` per member (`_ENSEMBLE_MEMBER_FACTORIES`) plus the
    majority-vote `_ENSEMBLE_LABEL` row, for *label* -- mirrors
    `_score_fusion_rows_for_label`'s own per-label gating/shape (same construction,
    this view's own members/features in place of that view's audio/vibration
    branches): a label whose fit-side window count is below `_ENSEMBLE_MIN_REF`, or
    that has zero conformal-side windows, reports every member AND the ENSEMBLE row as
    `low_confidence=True` with NaN metrics (`_ensemble_low_confidence_rows`) rather
    than partially succeeding.
    """
    label_fit = fit_windows[labels[fit_windows] == label]
    if label_fit.shape[0] < _ENSEMBLE_MIN_REF:
        return _ensemble_low_confidence_rows(label)

    label_conformal = conformal_windows[labels[conformal_windows] == label]
    if label_conformal.shape[0] == 0:
        return _ensemble_low_confidence_rows(label)

    label_scoring = scoring_windows[labels[scoring_windows] == label]
    n_scored = int(label_scoring.shape[0])
    shared_n_calibration = float(label_conformal.shape[0])

    member_rows: list[_EnsembleRow] = []
    member_alarms: dict[str, np.ndarray] = {}
    any_low_confidence = False

    for member_name, factory in _ENSEMBLE_MEMBER_FACTORIES.items():
        member_features = _ensemble_member_features(
            member_name, prepared_variant, prepared_logmel
        )
        scorer = factory().fit(member_features[label_fit])
        conformal_scores = scorer.score(member_features[label_conformal])
        threshold = calibrate(conformal_scores, alpha)
        any_low_confidence = any_low_confidence or threshold.low_confidence

        if n_scored == 0:
            # No alarm array recorded here -- the ensemble aggregate below branches
            # on this SAME `n_scored == 0` condition without ever reading
            # `member_alarms`, so populating it in this branch would be dead writes.
            member_rows.append(_EnsembleRow(
                member_name, label, float(threshold.n_calibration), 0.0, 0.0,
                math.nan, threshold.low_confidence,
            ))
            continue

        scores = scorer.score(member_features[label_scoring])
        alarms = scores > threshold.threshold
        member_alarms[member_name] = alarms
        n_alarms = int(alarms.sum())
        member_rows.append(_EnsembleRow(
            member_name, label, float(threshold.n_calibration), float(n_scored),
            float(n_alarms), n_alarms / n_scored, threshold.low_confidence,
        ))

    if n_scored == 0:
        ensemble_row = _EnsembleRow(
            _ENSEMBLE_LABEL, label, shared_n_calibration, 0.0, 0.0, math.nan,
            any_low_confidence,
        )
    else:
        votes = np.zeros(n_scored, dtype=np.int64)
        for member_name in _ENSEMBLE_MEMBER_FACTORIES:
            votes += member_alarms[member_name].astype(np.int64)
        ensemble_alarms = votes >= _ENSEMBLE_VOTE_THRESHOLD
        n_ensemble_alarms = int(ensemble_alarms.sum())
        ensemble_row = _EnsembleRow(
            _ENSEMBLE_LABEL, label, shared_n_calibration, float(n_scored),
            float(n_ensemble_alarms), n_ensemble_alarms / n_scored, any_low_confidence,
        )

    return [*member_rows, ensemble_row]


def _run_ensemble_view(
    prepared_variant: PreparedRun,
    prepared_logmel: PreparedRun,
    labels: np.ndarray,
    alpha: float,
    *,
    run_name: str,
) -> pd.DataFrame:
    """Majority-ensemble FAR table for one (run, labels_mode): OC-SVM + Isolation
    Forest + LSTM-AE (`_ENSEMBLE_MEMBER_FACTORIES`), each calibrated with its OWN
    per-state split-conformal threshold, plus an `_ENSEMBLE_LABEL` row per state
    counting windows where >= `_ENSEMBLE_VOTE_THRESHOLD` of the 3 members alarm
    (design chapter's committed majority-voting ensemble). Mirrors
    `_run_score_fusion_view`'s own three-way split construction exactly
    (`_ENSEMBLE_SEED`=7 top-level, `_ENSEMBLE_SEED + 1`=8 nested), so an `--ensemble`
    run and a same-run `run_sweep`/`--score-fusion` call all partition
    *prepared_variant*'s windows identically.

    Args:
        prepared_variant: The sweep `--variant`'s own `PreparedRun` -- drives the
            split (its own `segment_ids`/`valid_mask`) and supplies `ocsvm`/
            `iforest`'s feature matrix.
        prepared_logmel: The `logmel`-variant `PreparedRun` of the SAME run --
            supplies `lstmae`'s feature matrix. Must share *prepared_variant*'s exact
            grid (`_assert_ensemble_grids_match`, checked first).
        labels: Per-window labels aligned with *prepared_variant*'s (and, by the grid
            guard, *prepared_logmel*'s) windows -- same convention as `run_sweep`'s
            own `labels` argument.
        alpha: Nominal false-alarm rate target for `calibrate`, shared by every
            member.
        run_name: Named run, for the grid-mismatch guard's error message only.

    Returns:
        A DataFrame with columns `member, label, n_calibration, n_scored, n_alarms,
        realized_far, low_confidence` -- one row per (label, member) seen in the
        calibration or scoring windows, plus one `member="ENSEMBLE"` row per label.

    Raises:
        SystemExit: code 2, if *prepared_variant* and *prepared_logmel* do not share
            one grid (`_assert_ensemble_grids_match`).
        ValueError: propagated from `split_by_segments` if the three-way split cannot
            be formed (too few segments -- identical failure mode to `run_sweep`/
            `_run_score_fusion_view`).
    """
    _assert_ensemble_grids_match(run_name, prepared_variant, prepared_logmel)

    top_split = split_by_segments(
        prepared_variant.segment_ids, prepared_variant.valid_mask, 0.5, _ENSEMBLE_SEED
    )
    calibration_windows = top_split.calibration_windows
    scoring_windows = top_split.scoring_windows

    calib_mask = np.zeros(prepared_variant.features.shape[0], dtype=bool)
    calib_mask[calibration_windows] = True
    nested_split = split_by_segments(
        prepared_variant.segment_ids, calib_mask, 0.5, _ENSEMBLE_SEED + 1
    )
    fit_windows = nested_split.calibration_windows
    conformal_windows = nested_split.scoring_windows

    _assert_three_way_disjoint(fit_windows, conformal_windows, scoring_windows)

    all_windows = np.concatenate([calibration_windows, scoring_windows])
    all_labels = sorted(np.unique(labels[all_windows]).tolist())

    rows: list[_EnsembleRow] = []
    for label in all_labels:
        rows.extend(_ensemble_rows_for_label(
            label, prepared_variant, prepared_logmel, labels,
            fit_windows, conformal_windows, scoring_windows, alpha,
        ))

    return pd.DataFrame([asdict(r) for r in rows], columns=list(_ENSEMBLE_COLUMNS))


_ENSEMBLE_NOTES = """# Majority-ensemble notes

## What this view measures

`far_table_ensemble.csv` scores three independently-calibrated anomaly detectors --
`ocsvm` (`rowii.anomaly.scorers.OcSvmScorer`), `iforest` (`rowii.anomaly.scorers.
IsolationForestScorer`), and `lstmae` (`rowii.anomaly.recon.LstmAeScorer`, on the SAME
run's `logmel` features) -- against the SAME per-state split-conformal machinery every
other Step-2 view uses (`rowii.anomaly.conformal.calibrate`, one threshold per member
per state, held out on that state's own conformal-side windows). `ocsvm`/`iforest` fit
on the sweep `--variant`'s own features; `lstmae` fits on the `logmel` variant's
features of the SAME run (`rowii.anomaly.recon` module docstring: the reconstruction
scorers need a flattened logmel patch). The `ENSEMBLE` row per state counts windows
where AT LEAST 2 of the 3 members alarm -- the design chapter's committed
majority-voting ensemble ("requiring agreement before an alarm suppresses false
alarms").

## Honesty note

Each member holds its OWN marginal split-conformal guarantee, calibrated
independently at the same nominal alpha on its own per-state conformal-side scores --
exactly like every other per-member/per-branch Step-2 view. NO distribution-free
guarantee is claimed for the MAJORITY DECISION itself: combining three
independently-calibrated marginal guarantees into a joint statement about a ">= 2 of
3" vote is not a construction split conformal covers (unlike `--score-fusion`'s Fisher
rule, which re-calibrates ONE combined statistic through its own conformal threshold,
`rowii.anomaly.fusion` module docstring). `ENSEMBLE` rows report EMPIRICAL realized
FARs only -- exactly as measured on this run's own scoring-side windows, with no
finite-sample guarantee attached to the vote rule itself.

## Scope

- A state whose fit-side window count falls below the same `min_ref=20` floor the
  rest of Step-2 uses (or has zero conformal-side windows) reports every member AND
  the `ENSEMBLE` row as `low_confidence=True` with NaN metrics -- the same "do not
  alarm under a false promise" convention as `rowii.anomaly.sweep.far_row_excluded`.
- This view is conditioning/scorer-independent: it runs ONCE per (run, `--labels`,
  `--states`), not once per `--conditioning`/`--scorer` combination (unlike
  `--score-fusion`, which currently re-writes an identical copy into every combo dir
  -- see `_ensemble_out_dir`'s own docstring for the placement rationale).
- `--ensemble` requires `--protocol within-day` and `--labels detected` (the design
  chapter's committed ensemble is the runtime-realistic detected-label pipeline; GT
  states play no role in this view).
"""


def _write_ensemble_outputs(out_dir: Path, far_table: pd.DataFrame) -> None:
    """Write `far_table_ensemble.csv` + `ensemble_notes.md` into *out_dir* -- mirrors
    `_write_score_fusion_outputs`'s own label-dtype coercion (detected cluster ids are
    ints; keeping the column homogeneous avoids the same round-trip hazard
    `test_per_state_far_table_and_scores_label_dtypes_merge_cleanly` guards against
    for the main far_table)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table = far_table.copy()
    table["label"] = table["label"].astype(str)
    table.to_csv(out_dir / "far_table_ensemble.csv", index=False)
    (out_dir / "ensemble_notes.md").write_text(_ENSEMBLE_NOTES)


def _ensemble_out_dir(
    results_root: Path, run_name: str, variant: str, labels_mode: str, *,
    states: int | None = None,
) -> Path:
    """`results/step2/within-day/<run>/<variant>-<labels>[-k<states>]/ensemble/` --
    the ensemble view is conditioning/scorer-independent (task brief binding detail
    4), so unlike `--score-fusion` (which writes an IDENTICAL copy into EVERY
    `<conditioning>-<scorer>/` combo dir -- `_run_within_day_for_run`'s own
    docstring: "a deliberately simple... not a shared/deduplicated write") this view
    writes exactly ONCE per (run, variant, labels, states) into a dedicated sibling
    directory at the SAME level as every `<conditioning>-<scorer>/` combo dir, rather
    than duplicating into each one or arbitrarily picking one of them. The
    score-fusion redundancy itself is deliberately left as-is here (out of this
    task's scope; noted for the final whole-branch review). `states` mirrors
    `_within_day_out_dir`'s own `-k<states>` suffix convention, applied under the
    identical condition (only when `--states` is given), so a conditioning-
    granularity sweep's ensemble output stays disambiguated the same way its normal
    combo outputs already are.
    """
    k_suffix = f"-k{states}" if states is not None else ""
    return (
        results_root / "step2" / "within-day" / run_name
        / f"{variant}-{labels_mode}{k_suffix}" / "ensemble"
    )


def _run_and_write_ensemble_view(
    run: Run,
    variant: str,
    cfg: Config,
    sweep_prepared: PreparedRun,
    labels_mode: str,
    labels: np.ndarray,
    alpha: float,
    *,
    use_cache: bool,
    states: int | None = None,
) -> None:
    """`--ensemble`'s full per-run step (task brief binding detail 4): load the
    `logmel` `PreparedRun` for *run* ONCE (regardless of how many conditioning/scorer
    combos the caller's own sweep loop covers), compute `_run_ensemble_view`, and
    write its outputs into the dedicated `ensemble/` sibling directory
    (`_ensemble_out_dir`) -- called once per (run, variant, labels_mode, states) by
    `_run_within_day_for_run`, AFTER that run's own per-combo sweep loop (this view is
    conditioning/scorer-independent, module docstring). A failure at either stage
    (`prepare_run`'s `RuntimeError`, `_run_ensemble_view`'s `ValueError`) is logged
    and skipped -- never crashes the invocation or affects the normal sweep outputs
    already written for this run (same "strict addition, never a gate" principle
    `_run_within_day_for_run`'s own `score_fusion` docstring already documents for
    `--score-fusion`'s comparable failure mode). A grid mismatch
    (`_run_ensemble_view`'s own `SystemExit`) is NOT caught here -- it is a hard
    usage-level abort, propagated all the way up through `main`.
    """
    try:
        prepared_logmel = prepare_run(run, "logmel", cfg, use_cache=use_cache)
    except RuntimeError as exc:
        logger.warning(
            "run_step2: --ensemble's logmel prepare_run failed for run %r (%s) -- "
            "skipping the ensemble view (that run's normal sweep outputs, if any, "
            "are unaffected)",
            run.name, exc,
        )
        return

    try:
        far_table_ensemble = _run_ensemble_view(
            sweep_prepared, prepared_logmel, labels, alpha, run_name=run.name,
        )
    except ValueError as exc:
        logger.warning(
            "run_step2: --ensemble view failed for %s/%s-%s (%s) -- skipping (that "
            "run's normal sweep outputs, if any, are unaffected)",
            run.name, variant, labels_mode, exc,
        )
        return

    ensemble_dir = _ensemble_out_dir(
        cfg.results_root, run.name, variant, labels_mode, states=states
    )
    _write_ensemble_outputs(ensemble_dir, far_table_ensemble)


# ---------------------------------------------------------------------------
# within-day orchestration
# ---------------------------------------------------------------------------


def _run_within_day_for_run(
    run: Run,
    variant: str,
    cfg: Config,
    index: RecordingIndex,
    scorers: tuple[ScorerName, ...],
    conditionings: tuple[ConditioningName, ...],
    labels_mode: str,
    alpha: float,
    top_k: int,
    *,
    use_cache: bool,
    states: int | None = None,
    score_fusion: bool = False,
    score_fusion_scorer: str = "knn",
    ensemble: bool = False,
) -> int:
    """Prepare *run*, attach labels once, then run one sweep per (conditioning,
    scorer) pair, writing that combo's outputs/summary-row/register-section. Returns
    the number of combos actually written (a combo whose sweep raises `ValueError` --
    module docstring -- is logged and skipped, not counted).

    If `prepare_run` itself raises `RuntimeError` (a run too short/sparse for this
    variant -- e.g. a real "two stray files" run, more than 5% of its grid windows
    invalid, `rowii.pipeline.compute_validity_mask`), the WHOLE run is logged and
    skipped (0 combos written) -- the same "not fatal to the whole invocation"
    principle the module docstring already documents for a `run_sweep` `ValueError`,
    extended to this earlier failure mode (Task S7 real-data finding,
    `010726-tu1-afternoon`).

    `score_fusion` (`--score-fusion`, Task 5, spec D5; caller-guaranteed `variant ==
    "fusion"` via `main`'s own `parser.error` guard): after each combo's normal sweep
    outputs are written, ALSO run `_run_score_fusion_view` over the SAME `(sweep_
    prepared, labels)` and write `far_table_scorefusion.csv` + `scorefusion_notes.md`
    into that SAME combo's `out_dir`. The score-fusion view does not depend on
    *conditioning*/*scorer* at all (it has its own independent per-branch scorer,
    *score_fusion_scorer*), so requesting multiple conditionings/scorers together
    with `--score-fusion` writes an IDENTICAL copy of the view into each combo dir --
    a deliberately simple "recompute per combo iteration" choice, not a shared/
    deduplicated write. A score-fusion failure (`ValueError` -- e.g. `split_branch_
    columns` finding the variant's feature names do not split cleanly) is logged and
    skipped WITHOUT affecting that combo's own already-written normal sweep outputs
    or `n_written` count -- this view is a strict addition, never a gate on the base
    sweep.

    `states` (`--states`, package-3 Task 6, `None` by default): the conditioning
    granularity override for THIS run's own detector, forwarded to `_detected_labels_
    and_detector`'s `k` parameter -- only meaningful for `labels_mode == "detected"`,
    and `main`'s own parser.error guard rejects `--states` + `--labels gt` before
    this function is ever called (a `gt`-labels sweep never fits a detector at all;
    a direct programmatic call with both would just ignore `states` on the gt
    branch). Also forwarded to `_within_day_out_dir`/`_summary_row`/`_append_
    candidate_register` so a non-default value's combo outputs, summary row, and
    register section stay disambiguated from -- and never overwrite or collide with
    -- the default-K layout (see those functions' own docstrings).

    `ensemble` (`--ensemble`, Task 4, design chapter's committed majority-voting
    ensemble; caller-guaranteed `protocol == "within-day"` and `labels_mode ==
    "detected"` via `main`'s own `parser.error` guards, mirroring `--states`'):
    AFTER this run's own per-combo loop finishes (this view is conditioning/scorer-
    independent, unlike `score_fusion` which re-runs once per combo), load the
    `logmel` `PreparedRun` for THIS run ONCE, compute `_run_ensemble_view`, and write
    `far_table_ensemble.csv` + `ensemble_notes.md` into a dedicated `ensemble/`
    sibling directory (`_ensemble_out_dir`) -- see `_run_and_write_ensemble_view`'s
    own docstring for the full per-run step and its failure handling. Runs
    regardless of whether any combo in the loop above actually succeeded (`n_written`
    may be 0): the ensemble view never reads a combo's own `SweepResult`, only
    *sweep_prepared*/*labels* themselves.
    """
    if _is_beats_variant(variant):
        _import_beats_or_exit()

    try:
        prepared = prepare_run(run, variant, cfg, use_cache=use_cache)
    except RuntimeError as exc:
        logger.warning(
            "run_step2: prepare_run failed for run %r (%s) -- run is too short/sparse "
            "for this variant (module docstring's exclusion principle extended to "
            "prepare_run failures, not just run_sweep ValueErrors) -- skipping (0 "
            "combos written)",
            run.name, exc,
        )
        return 0
    scada = _load_run_scada(prepared, run, index)

    sweep_prepared: PreparedRun
    labels: np.ndarray
    if labels_mode == "gt":
        if scada is None:
            logger.warning(
                "run_step2: no SCADA coverage for run %r -- skipping gt-labels sweep",
                run.name,
            )
            return 0
        labels = _gt_state_labels(scada, cfg)
        valid_mask = prepared.valid_mask & (labels != "unknown")
        sweep_prepared = dataclasses.replace(prepared, valid_mask=valid_mask)
    else:
        labels, _detector = _detected_labels_and_detector(prepared, cfg, k=states)
        sweep_prepared = prepared

    n_written = 0
    for conditioning, scorer in itertools.product(conditionings, scorers):
        sweep_cfg = SweepConfig(alpha=alpha, top_k=top_k, conditioning=conditioning, scorer=scorer)
        try:
            result = run_sweep(sweep_prepared, labels, sweep_cfg)
        except ValueError as exc:
            logger.warning(
                "run_step2: sweep failed for %s/%s-%s/%s-%s (%s) -- skipping",
                run.name, variant, labels_mode, conditioning, scorer, exc,
            )
            continue

        out_dir = _within_day_out_dir(
            cfg.results_root, run.name, variant, labels_mode, conditioning, scorer,
            states=states,
        )
        _write_sweep_outputs(out_dir, result, sweep_prepared.grid, scada, top_k)
        _append_summary_row(
            cfg.results_root,
            _summary_row(
                run.name, variant, labels_mode, conditioning, scorer, alpha,
                result.far_table, states=states,
            ),
        )
        _append_candidate_register(
            cfg.results_root, run.name, variant, labels_mode, conditioning, scorer,
            alpha, result.candidates, sweep_prepared.grid, scada, states=states,
        )
        n_written += 1

        if score_fusion:
            try:
                sf_table = _run_score_fusion_view(
                    sweep_prepared, labels, alpha, score_fusion_scorer,
                )
            except ValueError as exc:
                logger.warning(
                    "run_step2: score-fusion view failed for %s/%s-%s/%s-%s (%s) -- "
                    "skipping (that combo's normal sweep outputs above are "
                    "unaffected)",
                    run.name, variant, labels_mode, conditioning, scorer, exc,
                )
            else:
                _write_score_fusion_outputs(out_dir, sf_table)

    if ensemble:
        _run_and_write_ensemble_view(
            run, variant, cfg, sweep_prepared, labels_mode, labels, alpha,
            use_cache=use_cache, states=states,
        )

    return n_written


# ---------------------------------------------------------------------------
# cross-day orchestration
# ---------------------------------------------------------------------------


def _run_cross_day(
    variant: str,
    cfg: Config,
    index: RecordingIndex,
    scorers: tuple[ScorerName, ...],
    labels_mode: str,
    alpha: float,
    top_k: int,
    *,
    use_cache: bool,
    run_names: list[str] | None = None,
) -> int:
    """Every ordered pair of SCADA-covered runs from DIFFERENT day trees (`Run.
    day_root`), same variant (trivially true -- one variant per invocation), skipping
    pairs whose feature dims are incompatible (module docstring). *run_names* (an
    explicit `--run` list; `None` = `--run all`, the pre-package-2 behavior unchanged)
    restricts the day set BEFORE anything is prepared: only pairs where BOTH day A and
    day B are named take part, so unlisted days never even reach `prepare_run` -- the
    point of the filter (Task 3 follow-up): scoping a sweep to specific pairs without
    triggering hours of cache-miss feature extraction for every other discovered day.
    Returns the number of (pair, scorer) combos actually written.

    A day whose own `prepare_run` raises `RuntimeError` (too short/sparse for this
    variant, e.g. a real "two stray files" run) is logged and excluded from
    `prepared_by_run` entirely -- every pair touching it is then skipped by the
    `prepared_by_run` membership check below, but pairs between the OTHER, healthy
    days are unaffected (Task S7 real-data finding: `010726-tu1-afternoon` crashed
    this function's `prepared_by_run` prewarm loop, which had no `try/except` at all,
    before this fix -- losing every OTHER day's matrix cell too, not just that one
    day's).
    """
    if _is_beats_variant(variant):
        _import_beats_or_exit()

    days = _scada_covered_runs(index)
    if run_names is not None:
        selected = set(run_names)
        days = [r for r in days if r.name in selected]
    if len(days) < 2:
        logger.warning(
            "run_step2: cross-day needs >= 2 SCADA-covered runs%s, found %d -- "
            "nothing to do",
            " matching --run" if run_names is not None else "",
            len(days),
        )
        return 0

    prepared_by_run: dict[str, tuple[PreparedRun, np.ndarray, pd.DataFrame | None]] = {}
    for run in days:
        try:
            prepared = prepare_run(run, variant, cfg, use_cache=use_cache)
        except RuntimeError as exc:
            logger.warning(
                "run_step2: prepare_run failed for run %r (%s) -- run is too "
                "short/sparse for this variant, excluding from cross-day (every pair "
                "touching it is skipped below, all OTHER days' pairs are unaffected)",
                run.name, exc,
            )
            continue
        valid_mask, scada = _cross_day_valid_mask(prepared, run, index, cfg, labels_mode)
        prepared_by_run[run.name] = (prepared, valid_mask, scada)

    n_written = 0
    for run_a, run_b in itertools.permutations(days, 2):
        if run_a.day_root == run_b.day_root:
            continue  # not genuinely "cross-day" -- same day tree, different session
        if run_a.name not in prepared_by_run or run_b.name not in prepared_by_run:
            continue  # one side failed prepare_run above -- pair has nothing to score

        prepared_a, valid_a, _scada_a = prepared_by_run[run_a.name]
        prepared_b, valid_b, scada_b = prepared_by_run[run_b.name]

        if prepared_a.features.shape[1] != prepared_b.features.shape[1]:
            logger.warning(
                "run_step2: skipping cross-day %s -> %s (incompatible feature dims %d vs %d)",
                run_a.name, run_b.name, prepared_a.features.shape[1], prepared_b.features.shape[1],
            )
            continue

        for scorer in scorers:
            try:
                result = _cross_day_sweep(
                    prepared_a, valid_a, prepared_b, valid_b, scorer, alpha, top_k
                )
            except ValueError as exc:
                logger.warning(
                    "run_step2: cross-day sweep failed for %s -> %s / %s (%s) -- skipping",
                    run_a.name, run_b.name, scorer, exc,
                )
                continue

            out_dir = _cross_day_out_dir(
                cfg.results_root, variant, labels_mode, run_a.name, run_b.name, scorer
            )
            _write_sweep_outputs(out_dir, result, prepared_b.grid, scada_b, top_k)
            _append_summary_row(
                cfg.results_root,
                _cross_day_summary_row(
                    run_a.name, run_b.name, variant, labels_mode, scorer, alpha,
                    result.far_table,
                ),
            )
            _append_candidate_register(
                cfg.results_root, f"{run_a.name}__to__{run_b.name}", variant, labels_mode,
                "pooled", scorer, alpha, result.candidates, prepared_b.grid, scada_b,
            )
            n_written += 1
    return n_written


# ---------------------------------------------------------------------------
# cross-day-per-state orchestration (package-2 Task 3, spec D2)
# ---------------------------------------------------------------------------


def _run_cross_day_per_state(
    variant: str,
    cfg: Config,
    index: RecordingIndex,
    scorers: tuple[ScorerName, ...],
    alpha: float,
    top_k: int,
    *,
    use_cache: bool,
    run_names: list[str] | None = None,
) -> int:
    """Every ordered pair of SCADA-covered runs from DIFFERENT day trees, same variant,
    scored under the per-state DETECTOR-TRANSFER protocol (`_cross_day_per_state_
    sweep`, module docstring). Mirrors `_run_cross_day`'s own orchestration closely
    (prepare-failure skip-not-crash, feature-dim compatibility check, same-day-root
    pairs skipped, and the same *run_names* pair-set filter -- see `_run_cross_day`'s
    docstring for its exact semantics and rationale), with two differences specific to
    this protocol: no `_cross_day_valid_mask` narrowing -- this protocol is
    detected-labels only (`main`'s `parser.error` guard rejects `--labels gt` before
    this function is ever called), so every day's own `prepared.valid_mask` is used
    unchanged -- and, when day B has SCADA coverage, an extra `far_by_true_state.csv`
    GT-diagnostic write alongside the usual three output files. Returns the number of
    (pair, scorer) combos actually written.
    """
    if _is_beats_variant(variant):
        _import_beats_or_exit()

    days = _scada_covered_runs(index)
    if run_names is not None:
        selected = set(run_names)
        days = [r for r in days if r.name in selected]
    if len(days) < 2:
        logger.warning(
            "run_step2: cross-day-per-state needs >= 2 SCADA-covered runs%s, found %d "
            "-- nothing to do",
            " matching --run" if run_names is not None else "",
            len(days),
        )
        return 0

    prepared_by_run: dict[str, tuple[PreparedRun, pd.DataFrame | None]] = {}
    for run in days:
        try:
            prepared = prepare_run(run, variant, cfg, use_cache=use_cache)
        except RuntimeError as exc:
            logger.warning(
                "run_step2: prepare_run failed for run %r (%s) -- run is too "
                "short/sparse for this variant, excluding from cross-day-per-state "
                "(every pair touching it is skipped below, all OTHER days' pairs are "
                "unaffected)",
                run.name, exc,
            )
            continue
        scada = _load_run_scada(prepared, run, index)
        prepared_by_run[run.name] = (prepared, scada)

    n_written = 0
    for run_a, run_b in itertools.permutations(days, 2):
        if run_a.day_root == run_b.day_root:
            continue  # not genuinely "cross-day" -- same day tree, different session
        if run_a.name not in prepared_by_run or run_b.name not in prepared_by_run:
            continue  # one side failed prepare_run above -- pair has nothing to score

        prepared_a, _scada_a = prepared_by_run[run_a.name]
        prepared_b, scada_b = prepared_by_run[run_b.name]

        if prepared_a.features.shape[1] != prepared_b.features.shape[1]:
            logger.warning(
                "run_step2: skipping cross-day-per-state %s -> %s (incompatible "
                "feature dims %d vs %d)",
                run_a.name, run_b.name,
                prepared_a.features.shape[1], prepared_b.features.shape[1],
            )
            continue

        for scorer in scorers:
            try:
                result, _labels_b = _cross_day_per_state_sweep(
                    prepared_a, prepared_a.valid_mask, prepared_b, prepared_b.valid_mask,
                    cfg, scorer, alpha, top_k,
                )
            except ValueError as exc:
                logger.warning(
                    "run_step2: cross-day-per-state sweep failed for %s -> %s / %s "
                    "(%s) -- skipping",
                    run_a.name, run_b.name, scorer, exc,
                )
                continue

            out_dir = _cross_day_per_state_out_dir(
                cfg.results_root, variant, run_a.name, run_b.name, scorer
            )
            _write_sweep_outputs(out_dir, result, prepared_b.grid, scada_b, top_k)
            if scada_b is not None:
                gt_states = _gt_state_labels(scada_b, cfg)
                far_by_true_state = _far_by_true_state(result.scores, gt_states)
                far_by_true_state.to_csv(out_dir / "far_by_true_state.csv", index=False)
            _append_summary_row(
                cfg.results_root,
                _cross_day_per_state_summary_row(
                    run_a.name, run_b.name, variant, scorer, alpha, result.far_table,
                ),
            )
            _append_candidate_register(
                cfg.results_root, f"{run_a.name}--to--{run_b.name}", variant, "detected",
                "per-state", scorer, alpha, result.candidates, prepared_b.grid, scada_b,
            )
            n_written += 1
    return n_written


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    # Pure argument-shape validation first (no config/index needed): every check here
    # is a usage error (`parser.error` -> exit 2) a user should hit BEFORE any data
    # tree is touched.
    run_names = _parse_run_names(args.run)
    if run_names is not None and not run_names:
        parser.error("--run got an empty run-name list")
    if args.protocol == "cross-day-per-state" and args.labels == "gt":
        parser.error("--protocol cross-day-per-state is detected-labels only (spec D2)")
    if (
        args.protocol in ("cross-day", "cross-day-per-state")
        and run_names is not None
        and len(run_names) < 2
    ):
        parser.error(
            f"--protocol {args.protocol} needs --run 'all' or >= 2 comma-separated "
            "run names (a single run has no cross-day pairs)"
        )
    if args.score_fusion and (args.protocol != "within-day" or args.variant != "fusion"):
        parser.error(
            "--score-fusion requires --protocol within-day and --variant fusion "
            f"(got --protocol {args.protocol!r} --variant {args.variant!r})"
        )
    if args.ensemble and args.protocol != "within-day":
        parser.error(
            "--ensemble requires --protocol within-day (the ensemble view "
            "conditions on the within-day sweep's own per-state references only) "
            f"-- got --protocol {args.protocol!r}"
        )
    if args.ensemble and args.labels == "gt":
        parser.error(
            "--ensemble is detected-labels only (the design chapter's committed "
            "ensemble is the runtime-realistic detected-label pipeline; GT states "
            "play no role in this view) -- got --labels 'gt'"
        )
    if args.states is not None and args.states < 2:
        parser.error(f"--states must be >= 2, got {args.states}")
    if args.states is not None and args.protocol != "within-day":
        parser.error(
            "--states requires --protocol within-day (conditions the within-day "
            f"sweep's own detector only) -- got --protocol {args.protocol!r}"
        )
    if args.states is not None and args.labels == "gt":
        parser.error(
            "--states is detected-labels only (a gt-labels sweep never fits a "
            "detector, so there is no cluster count to override -- rejecting rather "
            "than silently ignoring the flag) -- got --labels 'gt'"
        )

    cfg = load_config()
    index = discover(cfg.data_root)

    if run_names is not None:
        unknown = _unknown_run_names(run_names, index)
        if unknown:
            # Mirrors scripts/warm_cache.py's unknown-run precedent: hard usage error
            # (exit 2, listing every discovered run) rather than warn-and-skip -- an
            # explicitly named run silently matching nothing would defeat the point
            # of scoping the sweep in the first place.
            available = ", ".join(sorted({r.name for r in index.runs})) or "(none discovered)"
            print(
                f"run_step2: unknown run name(s): {', '.join(unknown)}; "
                f"available runs: {available}",
                file=sys.stderr,
            )
            return 2

    use_cache = not args.no_cache
    scorers = _resolve_scorers(args.scorer)

    if args.protocol == "within-day":
        conditionings = _resolve_conditionings(args.conditioning)
        runs = _resolve_runs(run_names, index)
        n_combos = 0
        for run in runs:
            n_combos += _run_within_day_for_run(
                run, args.variant, cfg, index, scorers, conditionings,
                args.labels, args.alpha, args.top_k, use_cache=use_cache,
                states=args.states,
                score_fusion=args.score_fusion,
                score_fusion_scorer=args.score_fusion_scorer,
                ensemble=args.ensemble,
            )
        print(
            f"run_step2: wrote {n_combos} within-day combo(s) across {len(runs)} run(s) "
            f"to {cfg.results_root / 'step2'}"
        )
    elif args.protocol == "cross-day":
        if args.conditioning not in ("all", "pooled"):
            logger.info(
                "run_step2: --conditioning=%r is ignored for --protocol cross-day "
                "(always pooled -- module docstring)",
                args.conditioning,
            )
        n_combos = _run_cross_day(
            args.variant, cfg, index, scorers, args.labels, args.alpha, args.top_k,
            use_cache=use_cache, run_names=run_names,
        )
        print(
            f"run_step2: wrote {n_combos} cross-day pair-combo(s) to "
            f"{cfg.results_root / 'step2'}"
        )
    else:
        if args.conditioning not in ("all", "per-state"):
            logger.info(
                "run_step2: --conditioning=%r is ignored for --protocol "
                "cross-day-per-state (always per-state -- module docstring; the "
                "pooled cell of this protocol's own grid is mathematically identical "
                "to --protocol cross-day, spec D2, so it is never reimplemented here)",
                args.conditioning,
            )
        n_combos = _run_cross_day_per_state(
            args.variant, cfg, index, scorers, args.alpha, args.top_k,
            use_cache=use_cache, run_names=run_names,
        )
        print(
            f"run_step2: wrote {n_combos} cross-day-per-state pair-combo(s) to "
            f"{cfg.results_root / 'step2'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
