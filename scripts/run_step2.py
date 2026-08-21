"""Step-2 mode-conditioned anomaly-sweep CLI: prepare -> label -> sweep -> report.

Drives `rowii.anomaly.sweep.run_sweep` over one or more (run, variant,
labels, conditioning, scorer) combinations for three protocols (extended for
the third protocol below):

- **within-day**: per selected run, prepare features (`rowii.pipeline.prepare_run`),
  attach labels, then run one sweep per (conditioning, scorer) pair the CLI was asked
  for.
- **cross-day**: calibrate a POOLED reference on one SCADA-covered run ("day A"),
  score every OTHER SCADA-covered run's ("day B") valid windows against it -- a
  cross-day false-alarm-rate matrix, format-compatible with the partner's own
  cross-day comparison table (no values adopted from either side).
- **cross-day-per-state**: same SCADA-covered day pairs as
  `cross-day`, but day A's detector is TRANSFERRED to day B (`rowii.state.detect.
  FittedDetector.apply`, no refit) instead of pooling, and day B's windows are scored
  under their own PREDICTED state against day A's per-state reference/threshold for
  that state -- answering whether per-state conditioning restores the FAR
  control `cross-day`'s pooling loses (see the dedicated section below).
- **cross-day-pooled**: held-out-day-group
  evaluation -- references/detector/frozen thresholds from a POOL of explicitly named
  fit runs (`--fit-runs`), evaluated on ONE held-out test run (`--test-run`) under
  BOTH threshold modes in one invocation (see the dedicated section below).

Every combo's outputs land under `results/step2/...` (see `_within_day_out_dir`/
`_cross_day_out_dir`/`_cross_day_per_state_out_dir`), plus two shared, append-only
artifacts every combo contributes a row/section to: `results/step2/summary.csv` and
`results/step2/candidate_register.md`.

## Labels (`--labels detected|gt`)

`detected` (the default, and the only run-time-realistic mode: per-state normal
references are built from Step-1's detected labels; GT states are used only in
evaluation views): `rowii.state.detect.FittedDetector.fit` (`run_detection`'s own
delegate) runs on this run's VALID windows only (mirrors
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

## Cross-day: pooled-only (binding simplification)

Detected cluster ids from two different days do not refer to the same physical state
(KMeans label 0 on day A is not comparable to label 0 on day B) and GT-state alignment
across days is out of scope for this package. Cross-day therefore NEVER calls
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

## Cross-day-per-state: detector transfer

Per-state conditioning across two different days cannot simply reuse either day's own
detected cluster ids, for exactly the reason the section above gives as cross-day's own
reason for pooling only: KMeans label 0 fit independently on day A and day B has no
reason to mean the same physical state on both. `cross-day-per-state` dissolves this by
never fitting a SECOND detector at all: `_cross_day_per_state_sweep` fits day A's
detector ONCE (`_detected_labels_and_detector`, `FittedDetector.fit`) and TRANSFERS it
to day B (`_apply_detector_labels`, `FittedDetector.apply` -- fit-day standardization +
fit-day HMM Viterbi decode, no refit/EM anywhere). Day B's windows
are then scored under their own PREDICTED (transferred) state, each against day A's
per-state reference and per-state conformal threshold for that SAME state id --
conditioning key = day A's cluster id, one model, one label space, so the cross-day
label-alignment problem this package's own spec calls out never arises. Day A's own
split collapses to fit/conformal only (`split_by_segments(day_a.segment_ids,
day_a_valid_mask, 0.5, seed)`, day A never contributes scoring windows here either,
exactly like `cross-day`'s own day A); day B contributes ALL of its own valid windows
to scoring, keyed by their predicted state.

This is the runtime-honest path: `--labels gt` is rejected outright for this
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

The `pooled` cell of this protocol's own comparison grid is intentionally
NEVER reimplemented here: pooling ignores per-state distinctions entirely, so a pooled
sweep under the transfer protocol is mathematically identical to `cross-day`'s own
existing pooled sweep (`_cross_day_sweep`) -- the existing `--protocol cross-day`
output IS that grid cell, unchanged, and doubles as the published pooled comparator
this protocol's per-state numbers are checked against.

## Cross-day-pooled: held-out-day-group evaluation

`--protocol cross-day-pooled --fit-runs <csv> --test-run <name> [--k N] [--alpha F]
[--save-snapshot PATH]` is the "day as a dataset point" protocol (LODO-style): the
artifact under test is fit on a POOL of days and evaluated on a day it has never seen.

- **One pooled detector, one label space**: `FittedDetector.fit_pooled` on the
  pooled nested-FIT features of every `--fit-runs` run (`rowii.anomaly.pools.
  build_pool(side="fit")`; `--k` clusters, default `_POOLED_DEFAULT_K`), then applied
  per run -- fit runs AND test run alike -- via `_apply_detector_labels` (per-run
  Viterbi, no cross-run EM chain). The cross-day label-alignment problem the two
  pair protocols above each dissolve in their own way is dissolved here by having
  only ONE model whose id space every run shares.
- **Pool sides, not top splits**: pooled references come from the pool's
  nested-FIT side, pooled FROZEN thresholds calibrate on the pool's nested-CONFORMAL
  side -- `run_sweep`'s exact nested-split convention per run, deliberately NOT
  `_cross_day_per_state_sweep`'s top-split-as-fit shortcut (see `rowii.anomaly.pools`'
  module-docstring WARNING for why that shortcut is wrong for pooled work).
- **Both threshold modes in one invocation**: `far_table_frozen.csv` (pool-conformal
  thresholds) and `far_table_recalibrate.csv` (thresholds recalibrated per
  state on the TEST run's own top-split calibration side -- `scripts/monitor.py`'s
  recalibrate recipe) share one scored window set: the test run's top-split SCORING
  side only. Every FAR number therefore names its mode by which file it lives in.
- **Pool-member evaluation ban**: the test run must not appear in
  `--fit-runs` (`parser.error`), and the fit day-GROUPS and the test
  day-group must be disjoint, where a day group is the DATE SET parsed from the
  run's burst-file names (`_run_day_groups`; catches sibling runs of one day
  like `010726-tu1`/`010726-tu2` AND midnight-crossing tails, which
  same-`day_root` checks alone would also catch but a differently-rooted
  re-ingest of the same day would not).
- **Coverage tables**: `coverage_train.csv` (pool calibration side) and
  `coverage_eval.csv` (test scoring side) count windows per detected-state label via
  `rowii.anomaly.pools.coverage_table`; when Betriebsdaten exist for the runs, GT
  `"state|load_bin"` composite tables (`coverage_train_gt.csv`/`coverage_eval_gt.
  csv`) are added, else skipped with a log line. `coverage_warnings` findings land
  in `notes.md` -- a mode evaluated but trained nowhere is never silent.
- **Optional pooled snapshot**: `--save-snapshot PATH` assembles the pooled
  `MonitorSnapshot` via `rowii.runtime.snapshot.fit_snapshot_from_parts` (frozen
  pool-conformal thresholds, `fit_run="pool:<csv>"`) and saves it with per-run pool
  provenance in the meta/sidecar.
- **Session normalization** (`--session-norm [--norm-minutes F]`): scoring moves into
  the label-free session-normalized space --
  every run's rows (pool references, pool conformal rows, the test run's
  calibration AND scoring windows) are transformed with THAT run's OWN first-N
  median/MAD stats (`rowii.anomaly.normalize.fit_session_stats`, N =
  `--norm-minutes`, default 20; per-run stats for pool members, BINDING). The
  pooled DETECTOR still fits/applies on RAW features (labels are
  norm-invariant). Outputs land in a `-snorm<N>` suffixed leaf
  (`<variant>-pooled-snorm20/`) so an N-sweep never overwrites the raw
  baseline. With `--save-snapshot`, the snapshot keeps RAW references (the field
  contract) plus POOL-GLOBAL stats over the raw pooled fit matrix
  (`fit_pool_stats`, `norm_minutes == 0.0` sentinel), and its stored conformal
  scores/thresholds are recomputed in that pool-global space so the artifact is
  SELF-CONSISTENT under the monitor's `--session-norm` reconstruction -- they
  deliberately differ from `far_table_frozen.csv`'s per-run-normalized thresholds
  (FAR-level-only comparability). DEFERRED: the within-day/cross-day
  `--session-norm` wiring is NOT implemented here -- cross-day-pooled only, to
  keep the change reviewable; the other protocols refuse the flag.
- **Level-only recalibration** (`--level-recal`): the shape-preserving,
  feature-native counterpart to
  `--session-norm` -- an additive offset in the log10 domain, applied ONLY to
  the TEST run's LEVEL columns (`*_log_rms`/`*_band_*`/`*_octave_*`,
  `rowii.anomaly.levelrecal`); shape columns (`*_spectral_centroid`/
  `*_rolloff95`/`*_kurtosis`) are untouched. The anchor (reference) is the
  per-column median over the POOLED FIT side's own RAW features; the
  run-side statistic is the TEST run's own label-free first-
  `_DEFAULT_NORM_MINUTES` (20) minutes of valid windows (mirrors
  `fit_session_stats`' first-N-minutes membership rule, duplicated locally --
  `_first_n_minutes_rows`). Only the TEST run's SCORING-space features shift;
  the pooled FIT/CONFORMAL rows stay RAW (they define the anchor -- mirrors
  `monitor.py`, where only the monitored run shifts).
  Requires `--variant audio` or `vibration` (fusion's per-run z-scored
  features have no meaningful level column -- any other variant is a
  `parser.error`); mutually exclusive with `--session-norm` (fit-path
  exclusivity); cross-day-pooled only, same precedent as `--session-norm`.
  Outputs land in a `-lrecal` suffixed leaf (`<variant>-pooled-lrecal/`). With
  `--save-snapshot`, the pooled snapshot
  ADDITIONALLY stores this SAME anchor as the OPTIONAL v2 member
  `level_recal_medians` (`rowii.runtime.snapshot.fit_snapshot_from_parts`'
  `level_recal_medians` kwarg; no version bump, mutually exclusive with
  `session_stats` by fit path) -- `monitor.py --level-recal` reads it back to
  align a NEW run onto exactly this anchor.
- **Failures are loud** (unlike the pair-matrix protocols' log-and-skip): this
  protocol names every run explicitly and produces exactly ONE combo, so any
  pool-level failure (a member that cannot prepare, k too large for the pool, a
  degenerate test split) exits 2 with the cause -- "log and skip" would always mean
  "silently wrote nothing", and a silently shrunken pool would corrupt provenance.
- Deliberately NO `summary.csv`/candidate-register rows here:
  rotation aggregation happens at execution time over the written
  far tables; the append-only artifacts keep their existing three-protocol schema.

## Output layout

- within-day: `results/step2/within-day/<run>/<variant>-<labels>/<conditioning>-<scorer>/`
  (`far_table.csv`, `scores.parquet`, `candidates.md`) -- exactly the spec's literal path.
  `--states K` (within-day + detected-labels only) appends a `-k<K>`
  suffix to the `<variant>-<labels>` segment ONLY when K is non-default
  (`fusion-detected-k8/`), so a non-default conditioning-granularity run never collides
  with -- or overwrites -- the default-K layout; `summary.csv`'s own `variant` column
  and the combo's `candidate_register.md` section header carry the same suffix
  (`_within_day_out_dir`/`_summary_row`/`_register_section_markdown` -- the register
  would otherwise accumulate identical headers across a K-granularity sweep of one
  run/variant). `--ensemble` (design chapter's committed majority-voting
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
- cross-day-per-state (no literal-path precedent in either spec):
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
- cross-day-pooled (layout pinned by the plan):
  `results/step2/cross-day-pooled/<test_run>/<variant>-pooled/` -- keyed by the
  HELD-OUT run (the pool is named inside `notes.md` and the snapshot provenance, not
  the path; a rotation writes one directory per test run). No scorer segment: the
  protocol runs exactly ONE scorer per invocation (`--scorer all` is rejected), so
  the `-pooled` leaf mirrors cross-day's fixed-conditioning folding without a scorer
  axis to fold in. Files: `far_table_frozen.csv`, `far_table_recalibrate.csv`
  (existing far-table schema, aggregate row included), `coverage_train.csv`,
  `coverage_eval.csv` (+ `coverage_{train,eval}_gt.csv` when SCADA exists),
  `notes.md`.
- `results/step2/summary.csv` (append-only, one row per combo actually written this
  invocation): `run, protocol, variant, labels, conditioning, scorer, alpha,
  per_label_count, pooled_realized_far, mean_per_state_far, n_low_confidence, notes`.
  `protocol` (2nd column) is `"within-day"`/`"cross-day"`/
  `"cross-day-per-state"`; `_read_summary_csv_or_none` backfills it onto an OLDER
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
variant -- e.g. a real "two stray files" run like `010726-tu1-afternoon`)
is logged and excluded -- the whole run for within-day
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
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch

    from rowii.fusionx.model import XattnHead

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
from rowii.anomaly.levelrecal import (  # noqa: E402
    apply_level_recal,
    column_medians,
    level_recal_offsets,
)
from rowii.anomaly.normalize import (  # noqa: E402
    SessionStats,
    apply_session_norm,
    fit_pool_stats,
    fit_session_stats,
)
from rowii.anomaly.pools import (  # noqa: E402
    PoolResult,
    build_pool,
    coverage_table,
    coverage_warnings,
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
from rowii.eval.metrics import derive_state_names  # noqa: E402
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
    _STUDENT_INSTALL_HINT,
    _TFC_INSTALL_HINT,
    PreparedRun,
    _is_beats_variant,
    _is_student_variant,
    _is_tfc_variant,
    _streams_for_variant,
    prepare_run,
    stream_columns,
)
from rowii.runtime.snapshot import fit_snapshot_from_parts, save_snapshot  # noqa: E402
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

_PROTOCOL_CHOICES: tuple[str, ...] = (
    "within-day", "cross-day", "cross-day-per-state", "cross-day-pooled",
)
_VARIANT_CHOICES: tuple[str, ...] = (
    "audio", "vibration", "fusion", "audio-beats", "fusion-beats",
    "audio-tfc", "vibration-tfc", "audio-student", "logmel",
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
             "D2). 'cross-day-pooled': held-out-day-group evaluation (package-7 spec "
             "D2/A3) -- pooled references/detector/frozen thresholds from --fit-runs, "
             "both threshold modes evaluated on the held-out --test-run.",
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
            "within-day, --labels detected, and a mic-primary --variant (audio, "
            "audio-beats, fusion, fusion-beats -- leakage safety for the LSTM-AE "
            "member's logmel split, see _ENSEMBLE_VARIANTS)."
        ),
    )
    parser.add_argument(
        "--xattn-fusion", action="store_true",
        help=(
            "Also compute the cross-attention fusion view (design chapter's third "
            "fusion level, package-5 spec D8): kNN on the trained head's joint "
            "audio-beats x vibration embedding, per-state conformal thresholds; "
            "writes far_table_xattn.csv + xattn_notes.md into a dedicated 'xattn/' "
            "sibling directory once per (run, --labels, --states). Requires "
            "--protocol within-day, --labels detected, --variant fusion, and "
            "ROWII_XATTN_CHECKPOINT pointing at a scripts/train_xattn.py checkpoint."
        ),
    )
    parser.add_argument(
        "--fit-runs", default=None,
        help=(
            "cross-day-pooled only: comma-separated FIT-run names, in pool order "
            "(row order matters for the pooled KMeans -- FittedDetector.fit_pooled's "
            "row-order note). Every listed run contributes its leakage-safe pool "
            "sides (rowii.anomaly.pools); the held-out --test-run must NOT be listed "
            "(spec A3.1) and must come from a different calendar day than every fit "
            "run (spec A3.8)."
        ),
    )
    parser.add_argument(
        "--test-run", default=None,
        help=(
            "cross-day-pooled only: the ONE held-out test run -- labeled by the "
            "pooled detector, thresholds applied in both modes, alarms evaluated on "
            "its own top-split scoring side only."
        ),
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help=(
            "cross-day-pooled only: pooled-detector cluster count for "
            "FittedDetector.fit_pooled (default: 5). Spec A3.4 selects the reported "
            "k by a GT-state-ARI sweep over {4, 5, 6} at execution time; a k too "
            "large for the pool exits 2 with a clear message."
        ),
    )
    parser.add_argument(
        "--save-snapshot", type=Path, default=None,
        help=(
            "cross-day-pooled only: also assemble the pooled MonitorSnapshot "
            "(rowii.runtime.snapshot.fit_snapshot_from_parts -- pooled references, "
            "FROZEN pool-conformal thresholds per spec A3.7, "
            "fit_run='pool:<fit-runs>') and save it to this path with per-run pool "
            "provenance in the meta/sidecar. Requires a runtime scorer "
            "(knn/mahalanobis)."
        ),
    )
    parser.add_argument(
        "--session-norm", action="store_true",
        help=(
            "cross-day-pooled only (package-7 Task 4, spec D3/A3.5; the plan's "
            "within-day/cross-day wiring is deferred): score in the label-free "
            "session-normalized space -- every run's rows are transformed with "
            "THAT run's own first-N median/MAD stats (per-run stats for pool "
            "members). The pooled detector still consumes RAW features. Outputs "
            "go to a '-snorm<N>' suffixed leaf; a --save-snapshot artifact "
            "stores RAW references + pool-global stats (norm_minutes=0 sentinel)."
        ),
    )
    parser.add_argument(
        "--norm-minutes", type=float, default=None,
        help=(
            "cross-day-pooled + --session-norm only: first-N prefix length in "
            "minutes for the per-run session stats (default: 20; spec A2.2 sweeps "
            "{5, 20, 60}). Must be > 0."
        ),
    )
    parser.add_argument(
        "--level-recal", action="store_true",
        help=(
            "cross-day-pooled only (package-8 Task 6, spec D2/A1.4/A1.9): "
            "shape-preserving level-only recalibration -- recentre the TEST "
            "run's LEVEL columns (*_log_rms/_band_*/_octave_*, log10-scaled) "
            "onto the pooled FIT side's own per-column median via an additive "
            "offset (rowii.anomaly.levelrecal), computed label-free from the "
            "TEST run's own first-20-minute prefix; shape columns are "
            "untouched. Requires --variant audio or vibration (fusion's "
            "per-run z-scored features have no meaningful level column, spec "
            "A1.1); mutually exclusive with --session-norm (A1.10). Outputs "
            "go to a '-lrecal' suffixed leaf."
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
    original single-name calling convention)."""
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


def _import_tfc_or_exit(cfg: Config, variant: str) -> None:
    """Mirrors `_import_beats_or_exit` above, extended: torch
    missing (checked first) -> SystemExit naming the shared `[beats]` extra; else
    the ONE checkpoint relevant to *variant* itself missing -> SystemExit naming
    its own env var. Duplicated (not imported) across every script that can reach
    a tfc variant -- see `scripts/warm_cache.py`'s own "a script must not depend
    on a sibling script's internals" rule."""
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
    """Mirrors `_import_tfc_or_exit` above, simplified: the
    distilled student has only ONE checkpoint (unlike TF-C's two independent
    branches), so there is no variant-based checkpoint selection -- torch
    missing (checked first) -> SystemExit naming the shared `[beats]` extra;
    else `cfg.student_checkpoint` missing -> SystemExit naming
    ROWII_STUDENT_CHECKPOINT. Duplicated (not imported) across every script
    that can reach `audio-student` -- see `scripts/warm_cache.py`'s own "a
    script must not depend on a sibling script's internals" rule."""
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

    *grid* is true-UTC since `rowii.pipeline.build_run_grid`, but each candidate
    file's own `header.t0_ns` (`read_header`, straight off disk) is still the raw
    DAQ axis -- shifted here by *betriebsdaten*'s own derived offset (`rowii.io.
    dataset.betriebsdaten_utc_offset_ns`) before the intersection test, mirroring
    `rowii.scada.labels.load_scada_window_means`'s identical DAQ-epoch-2000
    clock-offset fix. Before this fix, the comparison was RAW-vs-RAW (grid built
    on the pre-fix raw axis too) -- both sides shared the SAME axis by
    construction, so selection worked correctly by accident, not because either
    side was ever true UTC.
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

    *run* also supplies the audio-side DAQ-offset cross-check (`run_utc_offset_ns(run)`,
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
    """`_detected_labels` + the `FittedDetector` behind it (`cross-day-per-state`
    needs day A's detector, not just its labels, to TRANSFER it
    to day B -- `_apply_detector_labels` below).

    `k` (`--states`): overrides `cfg.detect.n_states`'s cluster
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
    standardization + fit-day HMM decode, no refit),
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
# Cross-day-per-state sweep (see module docstring)
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
    """Per-state cross-day sweep under DETECTOR TRANSFER: fit day
    A's detector, per-state references (from A's own fit-part), and per-state
    conformal thresholds (from A's own conformal-part) -- exactly `run_sweep`'s own
    per-state machinery, reused here via the now-public row builders
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
    # never a previously-appended aggregate.
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
    (module docstring): never part of the runtime path (no GT anywhere in
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
    <scorer>/` -- `states` (`--states`) appends a `-k<states>`
    suffix to the variant-labels segment ONLY when given (`fusion-detected-k8`), so a
    non-default conditioning-granularity run never collides with -- or overwrites --
    the default-K layout. `None` (every call without `--states`) reproduces the
    original (unsuffixed) path byte-for-byte.
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
column names."""


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
    SCADA context, and a blank `assessment` column for the human review pass.
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

    Convention: labels are strings in ALL THREE persisted
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
verified independently against our own sweeps where possible.

## External candidates (partner-reported, provenance labeled)

1. **Pre-start filling-valve (Fuelldüse) sound**, observed before machine start.
   - source: partner slide deck, deck-v3 p.16 (read-only reference)
   - assessment: operator-confirmed normal (partner); to cross-check in our sweeps

## Our sweeps

"""

_REGISTER_HEADER_FIRST_LINE = _REGISTER_HEADER.splitlines()[0]
"""`_REGISTER_HEADER`'s first line -- `_append_candidate_register`'s repair guard reads
just this one line back to confirm a previous header write actually completed,
rather than trusting `path.exists()` alone, which stays `True` even
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

    `states`: the section header's `<variant>-<labels>`
    part carries the SAME `-k<states>` suffix `_within_day_out_dir` puts on the combo
    directory, and only under the same condition (`--states` given) -- without it, a
    conditioning-granularity sweep (K=4/8/12 on one run/variant) would
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
    `summary.csv`)."""
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
"""`_SUMMARY_COLUMNS` before `protocol` was added as the 2nd column --
`_read_summary_csv_or_none` still accepts a `summary.csv` written in this older shape
(backfilling `protocol` onto it, never quarantining it as corrupt) so appending to a
file left over from before that change never produces a ragged CSV."""


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
    """`states` (`--states`): mirrors `_within_day_out_dir`'s own
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
    `far_row_aggregate`) -- so this reuses `_summary_far_metrics`
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
    parse, or parses to neither a recognised column schema (`summary.csv` is a
    shared, append-only artifact a killed prior invocation can leave
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
    returned as-is; `_SUMMARY_COLUMNS_LEGACY` (the older schema, missing `protocol`) is
    BACKFILLED -- `protocol` is inserted as the 2nd column, one value per row inferred
    from that row's own `run` field (`_infer_legacy_protocol`) -- and returned, never
    quarantined, so appending a fresh row (which always carries an explicit `protocol`)
    to an old-schema file never produces a ragged CSV. Any OTHER column set is treated
    as corrupt, same as before this schema change.
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


_SUMMARY_KEY_COLUMNS: tuple[str, ...] = (
    "run", "protocol", "variant", "labels", "conditioning", "scorer", "alpha", "notes"
)
"""Identity of one `summary.csv` row: every configuration axis, no metric column.
`_append_summary_row` REPLACES the existing row with the same identity instead of
blindly appending -- re-running a protocol combo (crash recovery, audit re-runs)
must update its row in place. Mirrors `run_step1._SUMMARY_KEY_COLUMNS` (the
2026-08-18 completeness audit traced duplicated Step-1 overview rows to exactly
such a rerun; this accumulator had the same blind-append shape)."""


def _append_summary_row(results_root: Path, row: _SummaryRow) -> None:
    """Upsert one row into `summary.csv` (identity: `_SUMMARY_KEY_COLUMNS`),
    recovering from a corrupt prior write (`_read_summary_csv_or_none`) and writing
    crash-safely itself: the combined frame goes to a `summary.csv.tmp` sibling
    first, then `os.replace`s the real path, so a crash mid-write of THIS call can
    never leave a partially-written `summary.csv` behind either."""
    summary_path = results_root / "step2" / "summary.csv"
    row_df = pd.DataFrame([vars(row)], columns=_SUMMARY_COLUMNS)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_summary_csv_or_none(summary_path)
    if existing is not None:
        key_cols = list(_SUMMARY_KEY_COLUMNS)
        new_key = tuple(row_df[key_cols].fillna("").astype(str).iloc[0])
        old_keys = existing[key_cols].fillna("").astype(str).apply(tuple, axis=1)
        combined = pd.concat([existing[old_keys != new_key], row_df], ignore_index=True)
    else:
        combined = row_df
    tmp_path = summary_path.with_name(summary_path.name + ".tmp")
    combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, summary_path)


# ---------------------------------------------------------------------------
# Score-level fusion view -- Fisher/Tippett p-value combination of
# the fusion variant's audio and vibration branches, re-calibrated via split conformal
# (rowii.anomaly.fusion's own module docstring carries the full statistical argument).
# ---------------------------------------------------------------------------

_SCORE_FUSION_SEED = 7
"""Top-level split seed for `_run_score_fusion_view` -- matches `SweepConfig.seed`'s
own default (7); the nested fit/conformal split below uses `_SCORE_FUSION_SEED + 1`
(=8), mirroring `run_sweep`'s own `cfg.seed + 1` convention exactly so a
`--score-fusion` run and a same-run `run_sweep` call partition one
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
"""The four `far_table_scorefusion.csv` rules: the two
COMBINED rules plus the two single-branch baselines, all evaluated through the exact
same p-value-then-recalibrate pipeline (`_score_fusion_statistic`) for a fair,
apples-to-apples comparison."""


@dataclass
class _ScoreFusionRow:
    """Mutable row builder for `far_table_scorefusion.csv` -- one instance per
    (label, rule) pair, mirroring `rowii.anomaly.sweep.FarRow`'s own "converted to a
    plain dict via `dataclasses.asdict` at the very end" pattern. Deliberately
    narrower than `FarRow`: this view carries no `threshold`/`nominal_alpha`/
    `achievable_alpha_floor`/`excluded` columns -- each rule's own threshold lives
    on a different, incomparable
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
    `"fisher"`/`"tippett"` rather than the branch's raw score directly -- `1.0 - p`
    is `rowii.anomaly.
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
    RE-CALIBRATED with split conformal (`rowii.anomaly.fusion` module docstring).
    Mirrors `rowii.anomaly.sweep.run_sweep`'s own three-way split
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
# Majority-ensemble evaluation view (design chapter's committed ensemble: a
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

_ENSEMBLE_VARIANTS: tuple[str, ...] = ("audio", "audio-beats", "fusion", "fusion-beats")
"""Sweep variants `--ensemble` accepts (review follow-up M1, leakage safety): exactly
the variants whose PRIMARY stream is the generator mic (`rowii.pipeline.
_streams_for_variant` puts `RAWGeneratorMic__0` first for all four -- the same, and
only, stream `logmel`'s own preparation reads). `_run_ensemble_view`'s three-way
split is drawn from the SWEEP variant's `segment_ids` and reused to index the logmel
features the LSTM-AE member consumes, so the split is leakage-safe for the LSTM-AE
member ONLY when the sweep variant's segment boundaries are logmel's own -- true by
construction for these four (shared primary stream), NOT for `vibration` (primary
stream `RAWGeneratorVib__2`, no structural relationship to the mic's file
boundaries): its grid can align with logmel's by coincidence, sailing past
`_check_ensemble_grid_alignment`, while a vibration-segment-safe split could still cut
one logmel recording segment across calibration and scoring -- silently voiding the
LSTM-AE member's split-conformal guarantee (reviewer finding). `logmel` itself is
also excluded, for a different reason: the committed ensemble contrasts OC-SVM/IF on
the sweep variant's OWN features against LSTM-AE on logmel -- under `--variant
logmel` all three members would fit one identical matrix, which is not the design
chapter's committed ensemble (and puts the classical scorers on a 3136-dim patch the
same dimensionality concern that scoped logmel out of `run_step1 --variant all`)."""

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


def _trim_prepared_tail(prepared: PreparedRun, n: int) -> PreparedRun:
    """Drop trailing windows so *prepared* carries exactly *n* -- used by the
    ensemble grid guard's off-by-one tolerance: the longer grid's extra TRAILING
    window has no partner index on the other grid, and trimming the tail leaves
    t0 (hence the sub-window index correspondence) untouched."""
    g = prepared.grid
    return dataclasses.replace(
        prepared,
        features=prepared.features[:n],
        valid_mask=prepared.valid_mask[:n],
        segment_ids=prepared.segment_ids[:n],
        grid=WindowGrid(t0_ns=g.t0_ns, window_ns=g.window_ns, n_windows=n),
    )


def _check_ensemble_grid_alignment(
    run_name: str, prepared_variant: PreparedRun, prepared_logmel: PreparedRun
) -> tuple[int, PreparedRun, PreparedRun, int]:
    """`--ensemble`'s grid-alignment guard (task brief binding detail 3, tolerance
    semantics per the real-data follow-up): the sweep variant's and the `logmel`
    variant's `PreparedRun`s must be STRUCTURALLY identical (`window_ns`,
    `n_windows` -- exact) and t0-aligned WITHIN ONE WINDOW for a shared window INDEX
    to refer to (near-)the-same physical time slot in both -- `_run_ensemble_view`'s
    single top-level split (below) is drawn once from the variant side and used to
    index BOTH prepared runs' feature matrices at the SAME window indices.

    Why tolerance, not equality (real-data finding, 2026-07-16): the DAQ's vib
    streams start tens of ms AFTER the mic streams, so a vibration-bearing sweep
    variant's grid -- built on the intersection of all its streams
    (`rowii.signals.windows.common_grid`) -- anchors its t0 slightly LATER than
    logmel's mic-only grid. Measured on the real days: 26 ms (250526-tu, fusion
    t0_ns ...763000116 vs logmel ...737000144) and 97 ms (290626-tu) on 1-s windows;
    010726 coincides exactly. Sub-window offsets are therefore physically inherent
    to this data -- the original exact-equality guard blocked the ensemble on 2 of
    the 3 real days. A sub-window offset keeps window i of both grids overlapping by
    at least `1 - offset/window_ns` (>= 97.4% / 90.3% on the measured days), which
    is acceptable for a DECISION-level evaluation view -- each member still votes on
    (almost exactly) the same second of plant operation -- so it is tolerated, but
    never silently: ONE warning logs the measured offset and overlap, and the same
    numbers are written into `ensemble_notes.md` (`_ensemble_alignment_note`). An
    offset of one full window or more breaks the index correspondence outright
    (window i refers to non-overlapping time slots) and still hard-aborts.

    Structurally, every variant the CLI admits to this view (`_ENSEMBLE_VARIANTS`,
    review follow-up M1) shares the primary generator-mic stream with `logmel`
    (`rowii.pipeline._streams_for_variant`), so the fatal branches should never fire
    in practice -- checked anyway (trust but verify, mirroring `rowii.anomaly.sweep.
    _assert_three_way_disjoint`'s identical stance) and kept as the backstop for a
    direct programmatic call that bypasses `main`'s guard. Note the guard's limits:
    grid alignment is necessary but NOT sufficient for the LSTM-AE member's leakage
    safety (a non-mic-primary variant's grid can align by coincidence while its
    segment boundaries do not) -- which is exactly why `--variant vibration` is
    rejected up front by `main`'s `_ENSEMBLE_VARIANTS` guard instead of being left
    to this check.

    A window-COUNT difference of exactly one is tolerated the same way (real-data
    follow-up 2, 2026-08-19, `300626-tu`): the same stream-start physics that
    produces the sub-window t0 offset can tip the window count by one when the
    earlier-starting mic-only grid gains an extra trailing window (measured: logmel
    9184 vs fusion 9183 on a 21 ms offset). The longer side's tail window has no
    partner index at all, so it is TRIMMED off (`_trim_prepared_tail`), logged, and
    stated in `ensemble_notes.md`; a count difference of two or more still
    hard-aborts.

    Returns:
        `(t0_offset_ns, prepared_variant, prepared_logmel, trimmed_windows)` -- the
        absolute t0 offset in ns (`0` when exactly aligned; always `< window_ns`),
        the two prepared runs (tail-trimmed copies when the off-by-one tolerance
        fired, the originals otherwise), and how many windows were trimmed (0 or
        1) -- threaded by the caller into `ensemble_notes.md`'s alignment line.

    Raises:
        SystemExit: code 2, with a clear message on stderr (parser.error-style; the
            argparse `parser` object itself is out of scope this deep in the call
            stack, so this raises directly rather than routing back through it), if
            `window_ns` differs, `n_windows` differ by two or more, or the t0
            offset is one full window or more -- each signals a structural
            inconsistency this view cannot safely paper over, not a
            per-run/per-combo condition to log-and-skip.
    """
    va, la = prepared_variant.grid, prepared_logmel.grid
    if va.window_ns != la.window_ns or abs(va.n_windows - la.n_windows) > 1:
        print(
            f"run_step2: --ensemble grid mismatch for run {run_name!r}: sweep-variant "
            f"grid (t0_ns={va.t0_ns}, window_ns={va.window_ns}, "
            f"n_windows={va.n_windows}) != logmel grid (t0_ns={la.t0_ns}, "
            f"window_ns={la.window_ns}, n_windows={la.n_windows}) -- window_ns must "
            "be identical and n_windows within one for the ensemble view to index "
            "both PreparedRuns at the same window positions",
            file=sys.stderr,
        )
        raise SystemExit(2)
    trimmed_windows = abs(va.n_windows - la.n_windows)
    if trimmed_windows:
        n = min(va.n_windows, la.n_windows)
        if va.n_windows > n:
            prepared_variant = _trim_prepared_tail(prepared_variant, n)
        else:
            prepared_logmel = _trim_prepared_tail(prepared_logmel, n)
        va, la = prepared_variant.grid, prepared_logmel.grid
    t0_offset_ns = abs(va.t0_ns - la.t0_ns)
    if t0_offset_ns >= va.window_ns:
        print(
            f"run_step2: --ensemble grids for run {run_name!r} are misaligned by >= "
            f"one window: |t0 offset| = {t0_offset_ns / 1e6:.1f} ms >= window "
            f"{va.window_ns / 1e6:.0f} ms (sweep-variant t0_ns={va.t0_ns}, logmel "
            f"t0_ns={la.t0_ns}) -- window index i would refer to non-overlapping "
            "time slots in the two PreparedRuns",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if t0_offset_ns > 0 or trimmed_windows:
        logger.warning(
            "run_step2: --ensemble grids for run %r are offset by %.1f ms on a "
            "%.0f ms window (minimum per-window overlap %.1f%%)%s -- proceeding: the "
            "LSTM-AE member votes on a window shifted by this sub-window DAQ "
            "stream-start offset relative to the classical members' window "
            "(documented in ensemble_notes.md)",
            run_name, t0_offset_ns / 1e6, va.window_ns / 1e6,
            (1.0 - t0_offset_ns / va.window_ns) * 100.0,
            (
                f"; trimmed {trimmed_windows} trailing window(s) off the longer "
                f"grid so both index the same {va.n_windows} windows"
                if trimmed_windows else ""
            ),
        )
    return t0_offset_ns, prepared_variant, prepared_logmel, trimmed_windows


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

    Emergent invariant (review follow-up L1): within one label, `low_confidence` is
    provably IDENTICAL across all three members -- `calibrate`'s flag depends only on
    the calibration COUNT and alpha (`rowii.anomaly.conformal.threshold_index`, never
    on the score values), and every member calibrates on the SAME `label_conformal`
    window set at the SAME alpha -- so partial low-confidence voting (some members
    trusted, others not, within one state) cannot occur, and the ENSEMBLE row's
    any-member `low_confidence` aggregate equals each member's own flag.
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
) -> tuple[pd.DataFrame, int, int]:
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
            supplies `lstmae`'s feature matrix. Must be structurally identical to
            *prepared_variant*'s grid (`window_ns`/`n_windows`) and t0-aligned within
            one window (`_check_ensemble_grid_alignment`, checked first -- sub-window
            DAQ stream-start offsets are tolerated and documented, see that guard's
            docstring for the real-data derivation).
        labels: Per-window labels aligned with *prepared_variant*'s (and, by the grid
            guard, *prepared_logmel*'s) windows -- same convention as `run_sweep`'s
            own `labels` argument.
        alpha: Nominal false-alarm rate target for `calibrate`, shared by every
            member.
        run_name: Named run, for the grid guard's messages only.

    Returns:
        `(far_table, t0_offset_ns, trimmed_windows)`: a DataFrame with columns
        `member, label, n_calibration, n_scored, n_alarms, realized_far,
        low_confidence` -- one row per (label, member) seen in the calibration or
        scoring windows, plus one `member="ENSEMBLE"` row per label -- the guard's
        measured absolute t0 offset between the two grids in ns (0 = exactly
        aligned; always < `window_ns`), and the guard's trailing-window trim count
        (0 or 1), both for `_write_ensemble_outputs`' notes line.

    Raises:
        SystemExit: code 2, if the two grids differ structurally (`window_ns`/
            `n_windows`) or are t0-misaligned by one full window or more
            (`_check_ensemble_grid_alignment`).
        ValueError: propagated from `split_by_segments` if the three-way split cannot
            be formed (too few segments -- identical failure mode to `run_sweep`/
            `_run_score_fusion_view`).
    """
    t0_offset_ns, prepared_variant, prepared_logmel, trimmed_windows = (
        _check_ensemble_grid_alignment(run_name, prepared_variant, prepared_logmel)
    )

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

    far_table = pd.DataFrame([asdict(r) for r in rows], columns=list(_ENSEMBLE_COLUMNS))
    return far_table, t0_offset_ns, trimmed_windows


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
- `--ensemble` requires `--protocol within-day`, `--labels detected` (the design
  chapter's committed ensemble is the runtime-realistic detected-label pipeline; GT
  states play no role in this view), and a mic-primary `--variant` (audio,
  audio-beats, fusion, fusion-beats): the three-way split is drawn from the sweep
  variant's own recording segments and reused for the LSTM-AE member's logmel
  features -- only mic-primary variants share segment boundaries with logmel by
  construction, so the split's leakage safety carries over to the LSTM-AE member
  (see `_ENSEMBLE_VARIANTS` in `scripts/run_step2.py` for the full rationale).
"""


def _ensemble_alignment_note(
    t0_offset_ns: int, window_ns: int, trimmed_windows: int = 0
) -> str:
    """The run-specific "Grid alignment" section appended to `_ENSEMBLE_NOTES` (the
    one run-dependent datum in an otherwise static notes file) -- real-data follow-up:
    a tolerated sub-window t0 offset between the sweep variant's and logmel's grids
    (`_check_ensemble_grid_alignment`'s docstring has the physical derivation) must be
    stated openly next to the FAR numbers it qualifies, not just logged. `t0_offset_ns`
    is the guard's return value, guaranteed `0 <= t0_offset_ns < window_ns`.
    """
    if t0_offset_ns == 0:
        return (
            "\n## Grid alignment (this run)\n\n"
            "Sweep-variant and logmel grids are exactly time-aligned (identical grid "
            "t0): every member votes on the identical time slot per window index.\n"
        )
    offset_ms = t0_offset_ns / 1e6
    overlap_pct = (1.0 - t0_offset_ns / window_ns) * 100.0
    return (
        "\n## Grid alignment (this run)\n\n"
        f"Member windows are time-aligned within {offset_ms:.1f} ms on a "
        f"{window_ns / 1e6:.0f} ms window (>= {overlap_pct:.1f}% per-window overlap): "
        f"the LSTM-AE member votes on a window shifted by {offset_ms:.1f} ms relative "
        "to the classical members' window -- a sub-window DAQ stream-start offset "
        "inherent to this data (the vib streams start tens of ms after the mic, so a "
        "vibration-bearing sweep variant's intersection grid anchors later than "
        "logmel's mic-only grid), acceptable for a decision-level evaluation view and "
        "stated openly here.\n"
        + (
            f"The same stream-start physics tipped the window count by one here: "
            f"{trimmed_windows} trailing window(s) of the longer grid, which have no "
            "partner index on the other grid, were trimmed before evaluation.\n"
            if trimmed_windows else ""
        )
    )


def _write_ensemble_outputs(
    out_dir: Path, far_table: pd.DataFrame, *, t0_offset_ns: int, window_ns: int,
    trimmed_windows: int = 0,
) -> None:
    """Write `far_table_ensemble.csv` + `ensemble_notes.md` into *out_dir* -- mirrors
    `_write_score_fusion_outputs`'s own label-dtype coercion (detected cluster ids are
    ints; keeping the column homogeneous avoids the same round-trip hazard
    `test_per_state_far_table_and_scores_label_dtypes_merge_cleanly` guards against
    for the main far_table). The notes get the run-specific grid-alignment line
    appended (`_ensemble_alignment_note`; *t0_offset_ns* is `_run_ensemble_view`'s
    measured value, *window_ns* the shared grid's window length)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table = far_table.copy()
    table["label"] = table["label"].astype(str)
    table.to_csv(out_dir / "far_table_ensemble.csv", index=False)
    (out_dir / "ensemble_notes.md").write_text(
        _ENSEMBLE_NOTES
        + _ensemble_alignment_note(t0_offset_ns, window_ns, trimmed_windows)
    )


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
        far_table_ensemble, t0_offset_ns, trimmed_windows = _run_ensemble_view(
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
    _write_ensemble_outputs(
        ensemble_dir, far_table_ensemble,
        t0_offset_ns=t0_offset_ns, window_ns=sweep_prepared.grid.window_ns,
        trimmed_windows=trimmed_windows,
    )




# ---------------------------------------------------------------------------
# --xattn-fusion view (the third fusion level)
# ---------------------------------------------------------------------------

_XATTN_SEED = 7
"""Top-level split seed for `_run_xattn_view` -- matches `SweepConfig.seed`'s own
default (7) and `_SCORE_FUSION_SEED`'s/`_ENSEMBLE_SEED`'s identical choice; the
nested fit/conformal split uses `_XATTN_SEED + 1` (=8), the same
`run_sweep`-mirroring convention (a view-local constant, not `_CROSS_DAY_SEED`:
this is a WITHIN-day view, the earlier reuse of the cross-day constant was a
final-review naming nit)."""

_XATTN_NOTES = """# Cross-attention fusion view -- notes

- The joint embedding is produced by a cross-attention head trained CLIP-style
  on the CALIBRATION side of this run's own top split (the audio-beats cache's
  PRIMARY-mic 768-d embedding slice vs the fusion cache's vibration columns of
  the SAME window as the positive pair) -- `scripts/train_xattn.py`; this view
  applies the identical `rowii.pipeline.stream_columns` slice at scoring time.
  The head therefore never saw a
  scoring-side segment, but it IS fitted on the same calibration data the
  conformal thresholds use; stated openly.
- Scoring is kNN (k=1, cosine) on the joint embedding with per-state
  split-conformal thresholds, the same machinery as every other view.
- Grid alignment between the fusion variant and the audio-beats cache follows
  the ensemble view's tolerance semantics (structural equality; sub-window t0
  offsets tolerated and recorded below).
"""


def _xattn_out_dir(
    results_root: Path, run_name: str, variant: str, labels_mode: str,
    *, states: int | None = None,
) -> Path:
    """`--xattn-fusion`'s dedicated sibling directory -- mirrors
    `_ensemble_out_dir`'s placement rationale (once per (run, labels, states),
    never per conditioning/scorer combo)."""
    suffix = f"-k{states}" if states is not None else ""
    return (
        results_root / "step2" / "within-day" / run_name
        / f"{variant}-{labels_mode}{suffix}" / "xattn"
    )


def _run_xattn_view(
    sweep_prepared: PreparedRun,
    prepared_audio: PreparedRun,
    labels: np.ndarray,
    alpha: float,
    head: XattnHead,
    device: torch.device,
    run_name: str,
) -> tuple[pd.DataFrame, int]:
    """FAR table for the cross-attention joint embedding, per state.

    Mirrors `run_sweep`'s three-way split exactly (top seed 7, nested seed 8) on
    the SWEEP variant's segment ids, restricted to windows valid in BOTH prepared
    runs (the fusion variant and the audio-beats cache have independently derived
    valid masks; a window NaN on either side must not enter any role). Per state
    with >= `SweepConfig.min_ref` fit windows and >= 1 conformal window: fit
    kNN (k=1, cosine) on the fit-side JOINT embeddings, calibrate on the
    conformal-side joint scores, alarm on the scoring side; states below the
    gate get a NaN row with `low_confidence=True`. Returns the table plus the
    measured |t0| offset between the two grids (ns) for the notes file.

    Raises:
        SystemExit: grid misalignment (structural mismatch or >= one window --
            `_check_ensemble_grid_alignment`'s semantics, reused verbatim).
        ValueError: degenerate splits (propagated from `split_by_segments`).
    """
    from rowii.fusionx.wrapper import joint_embeddings

    t0_offset_ns, sweep_prepared, prepared_audio, _trimmed = (
        _check_ensemble_grid_alignment(run_name, sweep_prepared, prepared_audio)
    )

    audio_idx, vib_idx = split_branch_columns(sweep_prepared.feature_names)
    del audio_idx  # the audio side comes from the audio-beats cache, not fusion
    vib_features = sweep_prepared.features[:, vib_idx]

    # The audio-beats cache concatenates BOTH mic streams (1536 columns); the
    # head's query side is defined on the PRIMARY-mic 768-d slice only -- the
    # SAME rowii.pipeline.stream_columns slice train_xattn.py trained on.
    primary_mic = _streams_for_variant("audio-beats")[0]
    audio_cols = stream_columns(prepared_audio.feature_names, primary_mic)
    if int(audio_cols.size) != head.cfg.audio_dim:
        raise ValueError(
            f"audio-beats primary-mic block ({primary_mic}) is {audio_cols.size} "
            f"column(s) wide but the xattn checkpoint's audio_dim is "
            f"{head.cfg.audio_dim} -- geometry mismatch"
        )
    audio_features = prepared_audio.features[:, audio_cols]

    valid_both = sweep_prepared.valid_mask & prepared_audio.valid_mask
    top = split_by_segments(sweep_prepared.segment_ids, valid_both, 0.5, _XATTN_SEED)
    calib_mask = np.zeros(sweep_prepared.features.shape[0], dtype=bool)
    calib_mask[top.calibration_windows] = True
    nested = split_by_segments(
        sweep_prepared.segment_ids, calib_mask, 0.5, _XATTN_SEED + 1
    )
    fit_w, conf_w = nested.calibration_windows, nested.scoring_windows
    scoring_w = top.scoring_windows

    def _joint(windows: np.ndarray) -> np.ndarray:
        return joint_embeddings(
            head, audio_features[windows], vib_features[windows], device
        )

    min_ref = SweepConfig().min_ref
    all_labels = sorted(
        np.unique(labels[np.concatenate([top.calibration_windows, scoring_w])]).tolist()
    )
    rows: list[dict[str, object]] = []
    for label in all_labels:
        fit_l = fit_w[labels[fit_w] == label]
        conf_l = conf_w[labels[conf_w] == label]
        score_l = scoring_w[labels[scoring_w] == label]
        if fit_l.shape[0] < min_ref or conf_l.shape[0] == 0 or score_l.shape[0] == 0:
            rows.append({
                "label": label, "n_calibration": math.nan, "n_scored": math.nan,
                "n_alarms": math.nan, "realized_far": math.nan,
                "low_confidence": True,
            })
            continue
        scorer = KnnScorer().fit(_joint(fit_l))
        conformal_scores = scorer.score(_joint(conf_l))
        threshold = calibrate(conformal_scores, alpha)
        alarms = int((scorer.score(_joint(score_l)) > threshold.threshold).sum())
        rows.append({
            "label": label,
            "n_calibration": float(threshold.n_calibration),
            "n_scored": float(score_l.shape[0]),
            "n_alarms": float(alarms),
            "realized_far": alarms / score_l.shape[0],
            "low_confidence": threshold.low_confidence,
        })

    columns = [
        "label", "n_calibration", "n_scored", "n_alarms", "realized_far",
        "low_confidence",
    ]
    return pd.DataFrame(rows, columns=columns), t0_offset_ns


def _run_and_write_xattn_view(
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
    """`--xattn-fusion`'s full per-run step, mirroring
    `_run_and_write_ensemble_view`'s structure: load the head (checkpoint from
    `cfg.xattn_checkpoint` -- `main` guarantees it is set) and the audio-beats
    `PreparedRun` ONCE per run, compute `_run_xattn_view`, write
    `far_table_xattn.csv` + `xattn_notes.md` into the dedicated `xattn/` sibling
    dir. `prepare_run` `RuntimeError` / view `ValueError` -> logged skip (strict
    addition, never a gate); grid-misalignment `SystemExit` propagates (hard
    usage abort), matching the ensemble view's contract exactly.
    """
    from rowii.fusionx.wrapper import load_xattn_head
    from rowii.signals.beats import best_device

    try:
        prepared_audio = prepare_run(run, "audio-beats", cfg, use_cache=use_cache)
    except RuntimeError as exc:
        logger.warning(
            "run_step2: --xattn-fusion's audio-beats prepare_run failed for run %r "
            "(%s) -- skipping the xattn view (normal sweep outputs unaffected)",
            run.name, exc,
        )
        return

    device = best_device()
    assert cfg.xattn_checkpoint is not None  # main()'s guard
    head = load_xattn_head(cfg.xattn_checkpoint, device)

    try:
        far_table, t0_offset_ns = _run_xattn_view(
            sweep_prepared, prepared_audio, labels, alpha, head, device, run.name
        )
    except ValueError as exc:
        logger.warning(
            "run_step2: --xattn-fusion view failed for %s/%s-%s (%s) -- skipping "
            "(normal sweep outputs unaffected)",
            run.name, variant, labels_mode, exc,
        )
        return

    out_dir = _xattn_out_dir(
        cfg.results_root, run.name, variant, labels_mode, states=states
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    far_table.to_csv(out_dir / "far_table_xattn.csv", index=False)
    notes = _XATTN_NOTES
    if t0_offset_ns:
        window_ns = sweep_prepared.grid.window_ns
        overlap = 1.0 - abs(t0_offset_ns) / window_ns
        notes += (
            f"\n- Measured grid t0 offset: {abs(t0_offset_ns) / 1e6:.1f} ms "
            f"(>= {overlap:.1%} window overlap between the two caches).\n"
        )
    (out_dir / "xattn_notes.md").write_text(notes)


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
    xattn_fusion: bool = False,
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
    extended to this earlier failure mode (`010726-tu1-afternoon`).

    `score_fusion` (`--score-fusion`; caller-guaranteed `variant ==
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

    `states` (`--states`, `None` by default): the conditioning
    granularity override for THIS run's own detector, forwarded to `_detected_labels_
    and_detector`'s `k` parameter -- only meaningful for `labels_mode == "detected"`,
    and `main`'s own parser.error guard rejects `--states` + `--labels gt` before
    this function is ever called (a `gt`-labels sweep never fits a detector at all;
    a direct programmatic call with both would just ignore `states` on the gt
    branch). Also forwarded to `_within_day_out_dir`/`_summary_row`/`_append_
    candidate_register` so a non-default value's combo outputs, summary row, and
    register section stay disambiguated from -- and never overwrite or collide with
    -- the default-K layout (see those functions' own docstrings).

    `ensemble` (`--ensemble`, design chapter's committed majority-voting
    ensemble; caller-guaranteed `protocol == "within-day"`, `labels_mode ==
    "detected"`, and a mic-primary *variant* (`_ENSEMBLE_VARIANTS`) via
    `main`'s own `parser.error` guards, mirroring `--states`'):
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
    if _is_tfc_variant(variant):
        _import_tfc_or_exit(cfg, variant)
    if _is_student_variant(variant):
        _import_student_or_exit(cfg)

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

    if xattn_fusion:
        _run_and_write_xattn_view(
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
    explicit `--run` list; `None` = `--run all`, the original behavior unchanged)
    restricts the day set BEFORE anything is prepared: only pairs where BOTH day A and
    day B are named take part, so unlisted days never even reach `prepare_run` -- the
    point of the filter: scoping a sweep to specific pairs without
    triggering hours of cache-miss feature extraction for every other discovered day.
    Returns the number of (pair, scorer) combos actually written.

    A day whose own `prepare_run` raises `RuntimeError` (too short/sparse for this
    variant, e.g. a real "two stray files" run) is logged and excluded from
    `prepared_by_run` entirely -- every pair touching it is then skipped by the
    `prepared_by_run` membership check below, but pairs between the OTHER, healthy
    days are unaffected (`010726-tu1-afternoon` crashed
    this function's `prepared_by_run` prewarm loop, which had no `try/except` at all,
    before this fix -- losing every OTHER day's matrix cell too, not just that one
    day's).
    """
    if _is_beats_variant(variant):
        _import_beats_or_exit()
    if _is_tfc_variant(variant):
        _import_tfc_or_exit(cfg, variant)
    if _is_student_variant(variant):
        _import_student_or_exit(cfg)

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
# cross-day-per-state orchestration
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
    if _is_tfc_variant(variant):
        _import_tfc_or_exit(cfg, variant)
    if _is_student_variant(variant):
        _import_student_or_exit(cfg)

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
# cross-day-pooled orchestration (see the module docstring's dedicated section)
# ---------------------------------------------------------------------------

_POOLED_DEFAULT_K = 5
"""`--k`'s default pooled cluster count. A GT-state-ARI sweep over {4, 5, 6} at
execution time picks the REPORTED k -- this default only
covers an invocation that does not say."""

_DEFAULT_NORM_MINUTES = 20.0
"""`--norm-minutes`' default; a sweep varies it over {5, 20, 60}. Duplicated in
`scripts/monitor.py` (same value, sibling scripts never import each other's
internals)."""

_BURST_NAME_DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}_\d{6}\.dat$")
"""The date portion of the burst filename pattern `<stream>_YYYY-MM-DD_HH-MM-SS_
ffffff.dat` -- a deliberately NARROWED duplicate of `rowii.io.dataset._BURST_RE`
(private; scripts do not import module internals, the same rule as
`_unknown_run_names`' script-sibling duplication) that keeps exactly the date group
the day-group guard needs."""


def _run_day_groups(run: Run) -> set[str]:
    """The day groups of *run*: the SET of calendar days (`"YYYY-MM-DD"`)
    parsed from EVERY burst file's NAME across all streams. The filename's LOCAL
    date is used as-is (not the UTC-converted hint): the spec's day groups are
    the recording days as the plant filesystem names them, and every guard
    comparison uses the same convention on both sides.

    A SET, not the first file's date: a recording that
    continues past local midnight without a >15-min gap stays ONE discovered
    run, so a first-file-only day group would report just the start date while
    most windows physically sit on the next calendar day -- silently bypassing
    the disjointness guard. No shipped run spans two dates today, but
    010726-tu_ph_tu's last file starts at 23:57 local -- three minutes from
    making this real. Multi-date runs are additionally logged, for visibility.

    Raises:
        ValueError: if *run* has no burst files at all, or any file's name does
            not carry the `_YYYY-MM-DD_HH-MM-SS_ffffff.dat` timestamp --
            impossible for a `rowii.io.dataset.discover` product (discovery only
            admits files matching the full burst pattern), so either means a
            hand-built `Run` violating the discovery contract.
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
            "run_step2: run %s spans %d calendar days (%s) -- its A3.8 day group "
            "is the full set",
            run.name, len(days), ", ".join(sorted(days)),
        )
    return days


def _cross_day_pooled_out_dir(
    results_root: Path,
    test_run: str,
    variant: str,
    *,
    alpha: float = 0.05,
    norm_minutes: float | None = None,
    level_recal: bool = False,
) -> Path:
    """`results/step2/cross-day-pooled/<test_run>/<variant>-pooled/` -- the plan's
    literal layout (module docstring's "Output layout" section has the full
    rationale: keyed by the HELD-OUT run, no scorer segment). A non-default
    *alpha* appends `-a<alpha>` (default 0.05, the CLI's own default): an
    alpha sweep and the standard run must never overwrite each other -- the
    2026-08-19 audit traced a contaminated leaf (an alpha-0.10 run silently
    replacing the 0.05 standard) to a refactor that had dropped exactly this
    suffix while the historical `-a0.0X` leaves still relied on it. The pooled
    k deliberately carries NO suffix (matching every historical leaf): it has
    no canonical constant default, and the alpha suffix already separates the
    sweep axis the contamination came from. A `--session-norm`
    run (*norm_minutes* not None) appends `-snorm<N>` (the `--states`/`-k<K>`
    suffix precedent) so an N-sweep and the raw baseline never overwrite
    each other. A `--level-recal` run (*level_recal* True)
    appends `-lrecal` -- mutually exclusive with *norm_minutes* by construction
    (the CLI guard, `main`), but this function itself stays a plain string
    composition, agnostic to which caller enforces that."""
    leaf = f"{variant}-pooled"
    if alpha != 0.05:
        leaf += f"-a{alpha:g}"
    if norm_minutes is not None:
        leaf += f"-snorm{norm_minutes:g}"
    if level_recal:
        leaf += "-lrecal"
    return results_root / "step2" / "cross-day-pooled" / test_run / leaf


def _pool_row_labels(pool: PoolResult, labels_per_run: dict[str, np.ndarray]) -> np.ndarray:
    """Per stacked pool row, the pooled-detector label of its SOURCE window --
    `labels_per_run[member][window]` via the pool's own `run_index`/`window_index`
    alignment (bitwise row-mapping property pinned by `tests/test_pools.py`). Pool
    rows are valid windows by construction, so no `_INVALID_LABEL` can appear."""
    out = np.empty(pool.features.shape[0], dtype=np.int64)
    for member_idx, member in enumerate(pool.members):
        mask = pool.run_index == member_idx
        out[mask] = labels_per_run[member.run_name][pool.window_index[mask]]
    return out


def _pool_row_gt_labels(pool: PoolResult, gt_by_run: dict[str, np.ndarray]) -> np.ndarray:
    """Per stacked pool row, the GT mode-name STRING of its source window -- the
    object-dtype sibling of `_pool_row_labels` (duplicated from
    `scripts/run_modebank.py::_pool_gt_labels`, script-sibling rule). Feeds
    `derive_state_names` at snapshot save."""
    out = np.empty(pool.features.shape[0], dtype=object)
    for member_idx, member in enumerate(pool.members):
        mask = pool.run_index == member_idx
        out[mask] = gt_by_run[member.run_name][pool.window_index[mask]]
    return out


def _session_norm_pool(
    pool: PoolResult, stats_by_run: dict[str, SessionStats]
) -> np.ndarray:
    """The pool's stacked feature rows in the session-normalized scoring space:
    each member's block transformed with ITS OWN run's first-N stats
    (per-run stats for pool members, BINDING -- pool-global stats here would
    let one run's session shift leak into every other run's normalized rows).
    Returns a fresh matrix; `pool.features` stays raw for the detector."""
    out = pool.features.copy()
    for member_idx, member in enumerate(pool.members):
        mask = pool.run_index == member_idx
        if mask.any():
            out[mask] = apply_session_norm(
                pool.features[mask], stats_by_run[member.run_name]
            )
    return out


def _first_n_minutes_rows(prepared: PreparedRun, norm_minutes: float) -> np.ndarray:
    """*prepared*'s own first *norm_minutes* of VALID windows, float64 -- the
    `--level-recal` run-side anchor source (the TEST run's own first-N-minutes
    windows, label-free).

    Mirrors `rowii.anomaly.normalize.fit_session_stats`'s identical window-
    membership rule (window START offset < `norm_minutes * 60s` AND
    `valid_mask`) -- duplicated here rather than imported, since level-recal
    needs the qualifying ROWS themselves (`rowii.anomaly.levelrecal.
    column_medians`' input), not a fitted `SessionStats` center/scale, and
    `rowii/anomaly/normalize.py` is out of this task's file scope.

    Raises:
        ValueError: if zero windows qualify (empty prefix / all-invalid) --
            the same failure mode as `fit_session_stats`, loud rather than
            silently medianing over nothing.
    """
    n_windows = prepared.features.shape[0]
    cutoff_ns = int(round(norm_minutes * 60.0 * 1e9))
    window_offsets = np.arange(n_windows, dtype=np.int64) * np.int64(prepared.grid.window_ns)
    qualifying = (window_offsets < cutoff_ns) & prepared.valid_mask
    rows = prepared.features[qualifying]
    if rows.shape[0] == 0:
        raise ValueError(
            f"zero valid windows start within the first {norm_minutes:g} minute(s) "
            f"of the run -- cannot compute the level-recal run-side median"
        )
    return np.asarray(rows, dtype=np.float64)


def _pooled_mode_row(
    label: int,
    sweep_cfg: SweepConfig,
    threshold: ConformalThreshold,
    label_scoring: np.ndarray,
    scoring_scores: np.ndarray | None,
) -> FarRow:
    """One threshold mode's row for a label that HAS a calibrated threshold:
    `far_row_empty_scoring` when the test run's scoring side holds no window of
    this label (*scoring_scores* is `None` exactly then), else `far_row_scored`
    with `alarm = score > threshold` -- the shared alarm rule of every Step-2
    path."""
    if scoring_scores is None:
        return far_row_empty_scoring(label, sweep_cfg, threshold)
    alarms = scoring_scores > threshold.threshold
    return far_row_scored(
        label, sweep_cfg, threshold, int(label_scoring.shape[0]), int(alarms.sum())
    )


def _cross_day_pooled_tables(
    pool_fit_features: np.ndarray,
    pool_fit_labels: np.ndarray,
    pool_conformal_features: np.ndarray,
    pool_conformal_labels: np.ndarray,
    test_features: np.ndarray,
    labels_test: np.ndarray,
    cal_windows: np.ndarray,
    scoring_windows: np.ndarray,
    sweep_cfg: SweepConfig,
    scorer_name: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, ConformalThreshold],
]:
    """Both threshold modes' FAR tables from ONE pooled-reference pass, plus the
    pooled snapshot parts. Space-agnostic: the caller passes the feature
    matrices in the SCORING space -- raw ones normally, session-normalized ones
    under `--session-norm` (pool rows per-run-normalized via `_session_norm_pool`,
    *test_features* with the test run's own stats) -- and every reference/score/
    threshold below simply lives in whatever space arrived here.

    Per label of the pooled-detector id space (ONE label space across every run --
    the property `_cross_day_per_state_sweep`'s transfer establishes for a pair,
    established here for the whole pool by the single pooled detector):

    - **reference**: the label's pooled nested-FIT-side rows; below
      `sweep_cfg.min_ref` -> `far_row_excluded` in BOTH tables (the same floor
      `build_references` applies for `run_sweep`).
    - **frozen threshold**: `calibrate` on the label's pooled
      nested-CONFORMAL-side scores -- never the top split.
    - **recalibrate threshold**: `calibrate` on the label's TEST-run
      calibration-side scores (`scripts/monitor.py`'s recalibrate recipe --
      references stay the POOL's, thresholds only).
    - zero conformal/calibration windows for one mode -> that mode's row is
      `far_row_no_conformal_data`; the other mode is unaffected.
    - **alarms**: the test run's top-split SCORING side only, scored ONCE per
      label and compared against each mode's own threshold.

    The returned `references`/`calibration_scores`/`thresholds` dicts are the
    `fit_snapshot_from_parts` inputs: a label enters ALL THREE iff it has a
    reference AND pooled conformal data (`fit_snapshot`'s own drop rule -- the
    snapshot's thresholds are the FROZEN pool-conformal ones), so their
    key sets are equal by construction.

    Both far tables end with the `run_sweep`-style aggregate `"pooled"` row
    (`far_row_aggregate` over that table's own per-label rows).
    """
    all_labels = sorted(
        {int(v) for v in np.unique(pool_fit_labels)}
        | {int(v) for v in np.unique(pool_conformal_labels)}
        | {int(v) for v in np.unique(labels_test[scoring_windows])}
    )

    frozen_rows: list[FarRow] = []
    recal_rows: list[FarRow] = []
    references: dict[int, np.ndarray] = {}
    calibration_scores: dict[int, np.ndarray] = {}
    thresholds: dict[int, ConformalThreshold] = {}

    for label in all_labels:
        reference = pool_fit_features[pool_fit_labels == label]
        if reference.shape[0] < sweep_cfg.min_ref:
            frozen_rows.append(far_row_excluded(label, sweep_cfg))
            recal_rows.append(far_row_excluded(label, sweep_cfg))
            continue
        scorer = _make_scorer(scorer_name).fit(reference)

        label_scoring = scoring_windows[labels_test[scoring_windows] == label]
        scoring_scores = (
            scorer.score(test_features[label_scoring])
            if label_scoring.shape[0]
            else None
        )

        pool_conformal_rows = pool_conformal_features[pool_conformal_labels == label]
        if pool_conformal_rows.shape[0] == 0:
            frozen_rows.append(far_row_no_conformal_data(label, sweep_cfg))
        else:
            conformal_scores = scorer.score(pool_conformal_rows)
            frozen_threshold = calibrate(conformal_scores, sweep_cfg.alpha)
            references[label] = reference
            calibration_scores[label] = conformal_scores
            thresholds[label] = frozen_threshold
            frozen_rows.append(
                _pooled_mode_row(label, sweep_cfg, frozen_threshold, label_scoring, scoring_scores)
            )

        label_cal = cal_windows[labels_test[cal_windows] == label]
        if label_cal.shape[0] == 0:
            recal_rows.append(far_row_no_conformal_data(label, sweep_cfg))
        else:
            recal_threshold = calibrate(
                scorer.score(test_features[label_cal]), sweep_cfg.alpha
            )
            recal_rows.append(
                _pooled_mode_row(label, sweep_cfg, recal_threshold, label_scoring, scoring_scores)
            )

    frozen_rows.append(far_row_aggregate(frozen_rows, sweep_cfg))
    recal_rows.append(far_row_aggregate(recal_rows, sweep_cfg))
    far_frozen = pd.DataFrame([asdict(r) for r in frozen_rows], columns=_FAR_TABLE_COLUMNS)
    far_recal = pd.DataFrame([asdict(r) for r in recal_rows], columns=_FAR_TABLE_COLUMNS)
    return far_frozen, far_recal, references, calibration_scores, thresholds


def _gt_composite_labels(scada: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Per-window GT `"state|load_bin"` composite labels (load-level
    coverage WITHIN states without new labels), e.g. `"turbine|2"`; off-state
    windows carry the `gt_labels` sentinel bin (`"standstill|-1"`)."""
    gt = gt_labels(scada, cfg.gt, window_s=cfg.window.window_s)
    composite: np.ndarray = (
        gt["state"].astype(str) + "|" + gt["load_bin"].astype(str)
    ).to_numpy()
    return composite


def _cross_day_pooled_notes(
    fit_run_names: list[str],
    test_run_name: str,
    variant: str,
    scorer_name: str,
    alpha: float,
    k: int,
    side_provenance: dict[str, dict[str, dict[str, int]]],
    coverage_warning_lines: list[str],
    *,
    session_norm_lines: list[str] | None = None,
    level_recal_lines: list[str] | None = None,
) -> str:
    """`notes.md` body: pool composition/provenance, threshold-mode semantics,
    the `--session-norm` section when active (*session_norm_lines*, built by
    `_run_cross_day_pooled` -- keyword-only with a `None` default so raw-baseline
    callers and the existing seam test stay untouched), the `--level-recal`
    section when active (*level_recal_lines*, same `None`-
    default keyword-only convention), coverage warnings (`"(none)"`
    sentinel when empty), the estimator-vs-final framing, and the honesty
    notes every pooled output must carry. Pure string assembly -- the
    warning-plumbing seam the tests exercise directly, since detected-label
    warnings cannot fire on a pool whose own detector defines the label space."""
    lines = [
        "# cross-day-pooled (held-out-day-group evaluation) -- notes",
        "",
        f"- fit pool: {', '.join(fit_run_names)} (pool order = `--fit-runs` order)",
        f"- held-out test run: {test_run_name}",
        f"- variant: {variant} | scorer: {scorer_name} | alpha: {alpha} | pooled k: {k}",
        "",
        "## Pool composition (per-run window counts)",
        "",
        "| run | calibration | fit | conformal |",
        "|---|---|---|---|",
    ]
    for run_name in fit_run_names:
        counts = [
            side_provenance.get(side, {}).get(run_name, {}).get("n_windows", 0)
            for side in ("calibration", "fit", "conformal")
        ]
        lines.append(f"| {run_name} | {counts[0]} | {counts[1]} | {counts[2]} |")
    lines += [
        "",
        "## Threshold modes",
        "",
        "- **frozen** (`far_table_frozen.csv`): per-state thresholds calibrated on the",
        "  POOL's nested-CONFORMAL side (spec A3.7) -- what a deployed pool artifact",
        "  would carry onto a new day unchanged.",
        "- **recalibrate** (`far_table_recalibrate.csv`): per-state thresholds",
        "  recalibrated on the TEST run's own calibration side (the monitor's",
        "  recalibrate recipe); references stay the pool's.",
        "- Both modes score the SAME window set: the test run's top-split SCORING",
        "  side only. Every FAR number names its mode by the file it lives in.",
    ]
    if session_norm_lines is not None:
        lines += ["", "## Session normalization (--session-norm, spec D3/A3.5)", ""]
        lines += session_norm_lines
    if level_recal_lines is not None:
        lines += ["", "## Level-only recalibration (--level-recal, spec D2/A1.4)", ""]
        lines += level_recal_lines
    lines += [
        "",
        "## Coverage warnings (A4.1/A4.2)",
        "",
    ]
    if coverage_warning_lines:
        lines += [f"- {warning}" for warning in coverage_warning_lines]
    else:
        lines.append("- (none)")
    lines += [
        "",
        "## Honesty notes",
        "",
        f"- Pool-member evaluation BAN (spec A3.1): the test run ({test_run_name}) is",
        "  NOT a pool member, and its day group is disjoint from every fit run's",
        "  (A3.8 parser guard) -- pooled artifacts are never evaluated on data they",
        "  were fit or calibrated on.",
        "- Estimator vs final system (spec A4.5): held-out-day-group rotations are",
        "  the honest ESTIMATOR -- train on some days, test on a day the system has",
        "  never seen, rotated so every day serves both roles (never both at once).",
        "  The FINAL deployed artifact pools EVERY available day's calibration side;",
        "  the rotations are how we know what that final system is worth, never a",
        "  restriction on what it may consume.",
        "- kNN is the primary pooled scorer; Mahalanobis on pooled (possibly",
        "  multi-modal) references carries an explicit caveat (spec A3.4): a single",
        "  Gaussian fit can straddle modes.",
        "- No fault-detection claims: FAR tables measure false-alarm behavior on",
        "  normal operation only (campaign pending; candidates framing unchanged).",
        "",
    ]
    return "\n".join(lines)


def _run_cross_day_pooled(
    fit_runs: list[Run],
    test_run: Run,
    variant: str,
    cfg: Config,
    index: RecordingIndex,
    scorer_name: str,
    alpha: float,
    top_k: int,
    *,
    k: int,
    use_cache: bool,
    save_snapshot_path: Path | None,
    session_norm: bool,
    norm_minutes: float,
    level_recal: bool,
) -> int:
    """The full cross-day-pooled pipeline for ONE (pool, test run) rotation --
    module docstring's dedicated section (incl. the `--session-norm` bullet:
    detector RAW, scoring space per-run session-normalized, `-snorm<N>` out-dir
    leaf, pooled-snapshot pool-global stats; and the `--level-recal` bullet:
    detector RAW, only the TEST run's scoring features shape-
    preservingly recentred onto the pooled FIT side's own median, `-lrecal`
    out-dir leaf). Returns a process exit code: 0 on success, 2 for any
    pool-level failure (loud-failure rationale there: exactly one combo,
    explicitly named runs, so log-and-skip would always mean "silently wrote
    nothing" and a silently shrunken pool would corrupt provenance -- a run
    whose first-N session-stats prefix is unusable under `--session-norm`, or
    whose first-N-minutes level median is unusable under `--level-recal`, is
    the same kind of hard error).
    """
    if _is_beats_variant(variant):
        _import_beats_or_exit()
    if _is_tfc_variant(variant):
        _import_tfc_or_exit(cfg, variant)
    if _is_student_variant(variant):
        _import_student_or_exit(cfg)

    fit_run_names = [run.name for run in fit_runs]
    prepared_all: dict[str, PreparedRun] = {}
    for run in (*fit_runs, test_run):
        try:
            prepared_all[run.name] = prepare_run(run, variant, cfg, use_cache=use_cache)
        except RuntimeError as exc:
            print(
                f"run_step2: prepare_run failed for run {run.name!r} ({exc}) -- "
                f"cross-day-pooled names every run explicitly, so a member that "
                f"cannot prepare is a hard error, not a skip",
                file=sys.stderr,
            )
            return 2
    prepared_fit = {name: prepared_all[name] for name in fit_run_names}
    prepared_test = prepared_all[test_run.name]

    fit_dim = next(iter(prepared_fit.values())).features.shape[1]
    if prepared_test.features.shape[1] != fit_dim:
        print(
            f"run_step2: test run {test_run.name!r} has {prepared_test.features.shape[1]} "
            f"feature column(s) but the fit pool has {fit_dim} -- incompatible "
            f"feature dims",
            file=sys.stderr,
        )
        return 2

    # --session-norm: one label-free first-N stats fit PER RUN -- fit
    # runs and test run alike, each on its OWN prefix (per-run stats for pool
    # members, binding). Fitted BEFORE any pooling so an unusable prefix
    # fails fast, loudly naming the run.
    stats_by_run: dict[str, SessionStats] | None = None
    if session_norm:
        stats_by_run = {}
        for name, prep in prepared_all.items():
            try:
                stats_by_run[name] = fit_session_stats(
                    prep.features, prep.valid_mask, prep.grid,
                    norm_minutes=norm_minutes,
                )
            except ValueError as exc:
                print(
                    f"run_step2: --session-norm cannot fit run {name!r}'s first-"
                    f"{norm_minutes:g}-minute stats ({exc}) -- every run in a "
                    f"session-normalized rotation needs a usable first-N prefix",
                    file=sys.stderr,
                )
                return 2

    # `scorer_name` is a plain `str` (same deliberate ignore as
    # `_cross_day_per_state_sweep`'s own SweepConfig construction).
    sweep_cfg = SweepConfig(alpha=alpha, top_k=top_k, scorer=scorer_name)  # type: ignore[arg-type]
    pool_calibration = build_pool(prepared_fit, "calibration", sweep_cfg)
    pool_fit = build_pool(prepared_fit, "fit", sweep_cfg)
    pool_conformal = build_pool(prepared_fit, "conformal", sweep_cfg)
    if pool_fit.features.shape[0] == 0:
        print(
            "run_step2: the pooled FIT side is empty (every fit run's splits were "
            "degenerate -- see the build_pool warnings above) -- nothing to fit on",
            file=sys.stderr,
        )
        return 2

    try:
        detector = FittedDetector.fit_pooled(pool_fit.features, cfg, k=k)
    except (RuntimeError, ValueError) as exc:
        # fit_pooled's RuntimeError (unassigned cluster ids) and sklearn's
        # n_samples < n_clusters ValueError are both "k too large for this pool"
        # from the CLI's point of view.
        print(
            f"run_step2: k too large for this pool: fit_pooled(k={k}) on "
            f"{pool_fit.features.shape[0]} pooled fit window(s) failed ({exc}) -- "
            f"pick a smaller --k (spec A3.4's execution sweep uses k in {{4, 5, 6}})",
            file=sys.stderr,
        )
        return 2

    labels_per_run = {
        name: _apply_detector_labels(prepared, detector)
        for name, prepared in prepared_all.items()
    }
    labels_test = labels_per_run[test_run.name]

    try:
        top = split_by_segments(
            prepared_test.segment_ids,
            prepared_test.valid_mask,
            sweep_cfg.calibration_frac,
            sweep_cfg.seed,
        )
    except ValueError as exc:
        print(
            f"run_step2: test run {test_run.name!r} cannot form its top "
            f"calibration/scoring split ({exc})",
            file=sys.stderr,
        )
        return 2
    cal_windows, scoring_windows = top.calibration_windows, top.scoring_windows

    # Scoring-space matrices: the DETECTOR above consumed raw
    # features; under --session-norm everything the SCORING path touches is
    # per-run session-normalized, and under --level-recal only the TEST run's
    # own scoring features are shape-preservingly recentred onto the pooled
    # FIT side's anchor -- the two are mutually exclusive
    # (parser-guarded in `main`), so at most one branch below ever applies.
    pool_fit_labels = _pool_row_labels(pool_fit, labels_per_run)
    pool_conformal_labels = _pool_row_labels(pool_conformal, labels_per_run)
    level_recal_offsets_used: dict[str, float] | None = None
    level_recal_anchor_used: dict[str, float] | None = None
    if stats_by_run is not None:
        pool_fit_scoring = _session_norm_pool(pool_fit, stats_by_run)
        pool_conformal_scoring = _session_norm_pool(pool_conformal, stats_by_run)
        test_features_scoring = apply_session_norm(
            prepared_test.features, stats_by_run[test_run.name]
        )
    elif level_recal:
        # The anchor = the per-column median over the POOLED FIT side's own
        # RAW features; the run-side statistic is the TEST run's own label-free
        # first-N-minutes prefix (_first_n_minutes_rows). The pooled FIT/
        # CONFORMAL rows stay RAW below -- only the TEST run shifts (mirrors
        # monitor.py).
        feature_names = list(next(iter(prepared_fit.values())).feature_names)
        try:
            anchor = column_medians(pool_fit.features, feature_names)
            test_first_n_rows = _first_n_minutes_rows(prepared_test, _DEFAULT_NORM_MINUTES)
            test_median = column_medians(test_first_n_rows, feature_names)
            level_recal_offsets_used = level_recal_offsets(test_median, anchor)
        except ValueError as exc:
            print(
                f"run_step2: --level-recal cannot compute offsets for test run "
                f"{test_run.name!r} ({exc})",
                file=sys.stderr,
            )
            return 2
        # The SAME anchor is stored as the pooled snapshot's
        # optional level_recal_medians member when --save-snapshot is also
        # given (below) -- the monitor aligns a new run's own first-N median
        # onto exactly this anchor, mirroring how far_table_{frozen,recalibrate}
        # .csv align the TEST run above.
        level_recal_anchor_used = anchor
        pool_fit_scoring = pool_fit.features
        pool_conformal_scoring = pool_conformal.features
        test_features_scoring = apply_level_recal(
            prepared_test.features, feature_names, level_recal_offsets_used
        )
    else:
        pool_fit_scoring = pool_fit.features
        pool_conformal_scoring = pool_conformal.features
        test_features_scoring = prepared_test.features

    far_frozen, far_recal, references, calibration_scores, thresholds = (
        _cross_day_pooled_tables(
            pool_fit_scoring,
            pool_fit_labels,
            pool_conformal_scoring,
            pool_conformal_labels,
            test_features_scoring,
            labels_test,
            cal_windows,
            scoring_windows,
            sweep_cfg,
            scorer_name,
        )
    )

    out_dir = _cross_day_pooled_out_dir(
        cfg.results_root, test_run.name, variant,
        alpha=alpha,
        norm_minutes=norm_minutes if session_norm else None,
        level_recal=level_recal,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in (
        ("far_table_frozen.csv", far_frozen),
        ("far_table_recalibrate.csv", far_recal),
    ):
        coerced = table.copy()
        coerced["label"] = coerced["label"].astype(str)  # _write_sweep_outputs convention
        coerced.to_csv(out_dir / filename, index=False)

    # Coverage tables: train = the pool's whole calibration side (fit u
    # conformal -- everything the pooled artifact consumed), eval = the test run's
    # scored windows; detected-state labels first, GT composite overlay when SCADA
    # exists.
    windows_train = {member.run_name: member.windows for member in pool_calibration.members}
    train_table = coverage_table(
        prepared_fit, windows_train, {name: labels_per_run[name] for name in prepared_fit}
    )
    eval_table = coverage_table(
        {test_run.name: prepared_test},
        {test_run.name: scoring_windows},
        {test_run.name: labels_test},
    )
    warning_lines = coverage_warnings(train_table, eval_table)
    train_table.to_csv(out_dir / "coverage_train.csv", index=False)
    eval_table.to_csv(out_dir / "coverage_eval.csv", index=False)

    scada_by_run: dict[str, pd.DataFrame | None] = {
        run.name: _load_run_scada(prepared_all[run.name], run, index)
        for run in (*fit_runs, test_run)
    }
    gt_train: pd.DataFrame | None = None
    gt_eval: pd.DataFrame | None = None
    missing_fit_scada = [name for name in fit_run_names if scada_by_run[name] is None]
    if missing_fit_scada:
        logger.info(
            "run_step2: skipping the GT state x load-bin TRAIN coverage table -- fit "
            "run(s) %s have no Betriebsdaten coverage",
            ", ".join(missing_fit_scada),
        )
    else:
        gt_labels_train: dict[str, np.ndarray] = {}
        for name in fit_run_names:
            scada_fit = scada_by_run[name]
            assert scada_fit is not None  # guarded by missing_fit_scada above
            gt_labels_train[name] = _gt_composite_labels(scada_fit, cfg)
        gt_train = coverage_table(prepared_fit, windows_train, gt_labels_train)
        gt_train.to_csv(out_dir / "coverage_train_gt.csv", index=False)
    scada_test = scada_by_run[test_run.name]
    if scada_test is None:
        logger.info(
            "run_step2: skipping the GT state x load-bin EVAL coverage table -- test "
            "run %s has no Betriebsdaten coverage",
            test_run.name,
        )
    else:
        gt_eval = coverage_table(
            {test_run.name: prepared_test},
            {test_run.name: scoring_windows},
            {test_run.name: _gt_composite_labels(scada_test, cfg)},
        )
        gt_eval.to_csv(out_dir / "coverage_eval_gt.csv", index=False)
    if gt_train is not None and gt_eval is not None:
        warning_lines.extend(coverage_warnings(gt_train, gt_eval))

    side_provenance = {
        "calibration": pool_calibration.provenance,
        "fit": pool_fit.provenance,
        "conformal": pool_conformal.provenance,
    }
    session_note_lines: list[str] | None = None
    if stats_by_run is not None:
        session_note_lines = [
            "Scoring happened in the label-free session-normalized space: every "
            "run's rows (pool references, pool conformal rows, the test run's "
            "calibration and scoring windows) were transformed with THAT run's "
            f"own first-{norm_minutes:g}-minute median/MAD stats (scale floored "
            "at 1e-8). The pooled DETECTOR consumed RAW features (A3.5 binding "
            "boundary: labels are norm-invariant). Comparability with raw-space "
            "FAR tables is FAR-level only (A3.5).",
            "",
            "| run | stats n_windows | norm_minutes |",
            "|---|---|---|",
            *(
                f"| {name} | {stats_by_run[name].n_windows} "
                f"| {stats_by_run[name].norm_minutes:g} |"
                for name in (*fit_run_names, test_run.name)
            ),
            "",
            "- caveats: if a run's first minutes contain a fault, normalization "
            "absorbs part of it; the first minutes' STATE MIX parameterizes the "
            "stats (state-mix confound -- the A2.2 norm-minutes sweep is the "
            "sensitivity probe).",
            "- scope: --session-norm is wired for cross-day-pooled only in this "
            "package; the within-day/cross-day wiring is DEFERRED (Task 4 scope "
            "decision).",
        ]
    level_recal_note_lines: list[str] | None = None
    if level_recal_offsets_used is not None:
        level_recal_note_lines = [
            "Scoring happened with the TEST run's LEVEL columns (`*_log_rms`/"
            "`*_band_*`/`*_octave_*`, log10-scaled by construction -- VERIFIED "
            "fact, `rowii.anomaly.levelrecal` module docstring) additively "
            "recentred onto the pooled FIT side's own per-column median "
            "(spec A1.4 anchor: `column_medians(pool_fit.features, ...)`); "
            "shape columns (`*_spectral_centroid`/`*_rolloff95`/`*_kurtosis`) "
            "pass through untouched by construction. Offset = `run_median - "
            "reference_median` per column, the run-side median drawn from the "
            f"TEST run's own label-free first-{_DEFAULT_NORM_MINUTES:g}-minute "
            "prefix. The pooled FIT/CONFORMAL rows stay RAW below (they define "
            "the anchor) -- only the TEST run's own scoring features shift.",
            "",
            "This is our own, independent, feature-native test of the "
            "partner's channel-recalibration idea (Rodrigues & Zhang, 2026): "
            "every offset below is computed from OUR OWN caches; no partner "
            "dB figure is imported here (D2, spec A1.1/A1.9).",
            "",
            "| level column | offset (log10 units, run - reference) |",
            "|---|---|",
            *(
                f"| {name} | {offset:.6g} |"
                for name, offset in sorted(level_recal_offsets_used.items())
            ),
            "",
        ]
    (out_dir / "notes.md").write_text(
        _cross_day_pooled_notes(
            fit_run_names, test_run.name, variant, scorer_name, alpha, k,
            side_provenance, warning_lines,
            session_norm_lines=session_note_lines,
            level_recal_lines=level_recal_note_lines,
        )
    )

    if save_snapshot_path is not None:
        # Same checkpoint-provenance construction as `fit_snapshot` (duplicated:
        # extracting a helper would touch that function's body, which this task
        # must leave behaviorally untouched).
        checkpoints = {
            name: str(path)
            for name, path in (
                ("beats_checkpoint", cfg.beats_checkpoint),
                ("tfc_audio_checkpoint", cfg.tfc_audio_checkpoint),
                ("tfc_vib_checkpoint", cfg.tfc_vib_checkpoint),
                ("student_checkpoint", cfg.student_checkpoint),
                ("beats_int8_checkpoint", cfg.beats_int8_checkpoint),
                ("xattn_checkpoint", cfg.xattn_checkpoint),
            )
            if path is not None
        }
        snapshot_references = references
        snapshot_cal_scores = calibration_scores
        snapshot_thresholds = thresholds
        snapshot_session_stats: SessionStats | None = None
        if stats_by_run is not None:
            # Pooled-snapshot session-norm semantics (documented in
            # fit_snapshot_from_parts' docstring): references stay
            # RAW (the MonitorSnapshot field contract), session_stats are
            # POOL-GLOBAL (center/scale of the raw pooled fit matrix,
            # norm_minutes=0.0 sentinel -- a pooled artifact has no single fit
            # day, hence no first-N prefix), and the stored conformal scores +
            # frozen thresholds are recomputed in that pool-global space so the
            # artifact is SELF-CONSISTENT under the monitor's --session-norm
            # reconstruction (stored stats transform the stored references into
            # exactly the space the scores were calibrated in). They deliberately
            # differ from far_table_frozen.csv's per-run-normalized thresholds --
            # FAR-level-only comparability.
            snapshot_session_stats = fit_pool_stats(pool_fit.features)
            snapshot_references = {}
            snapshot_cal_scores = {}
            snapshot_thresholds = {}
            for label in sorted(thresholds):
                raw_reference = pool_fit.features[pool_fit_labels == label]
                scorer = _make_scorer(scorer_name).fit(
                    apply_session_norm(raw_reference, snapshot_session_stats)
                )
                raw_conformal = pool_conformal.features[pool_conformal_labels == label]
                scores = scorer.score(
                    apply_session_norm(raw_conformal, snapshot_session_stats)
                )
                snapshot_references[label] = raw_reference
                snapshot_cal_scores[label] = scores
                snapshot_thresholds[label] = calibrate(scores, alpha)

        # The commissioning-time cluster-id -> operating-mode-name map,
        # derived from the POOLED FIT side's own detected labels + GT state
        # strings (the same pooled fit side the detector/references were fit
        # on). `None` when any fit run lacks Betriebsdaten (the GT-skip seam
        # `missing_fit_scada` already established above) -- a name derivation
        # needs GT for every fit run, not just a subset.
        state_names: dict[int, str] | None = None
        if not missing_fit_scada:
            gt_state_by_run: dict[str, np.ndarray] = {}
            for name in fit_run_names:
                scada_fit_gt = scada_by_run[name]
                assert scada_fit_gt is not None  # guarded by missing_fit_scada above
                gt_state_by_run[name] = _gt_state_labels(scada_fit_gt, cfg)
            pool_fit_gt = _pool_row_gt_labels(pool_fit, gt_state_by_run)
            state_fitted_ids = [int(i) for i in np.asarray(detector.smoother._fitted_ids)]
            state_names = derive_state_names(pool_fit_gt, pool_fit_labels, state_fitted_ids)

        snapshot = fit_snapshot_from_parts(
            detector,
            snapshot_references,
            snapshot_cal_scores,
            snapshot_thresholds,
            scorer=scorer_name,
            alpha=alpha,
            min_ref=sweep_cfg.min_ref,
            calibration_frac=sweep_cfg.calibration_frac,
            seed=sweep_cfg.seed,
            variant=variant,
            fit_run="pool:" + ",".join(fit_run_names),
            feature_names=list(next(iter(prepared_fit.values())).feature_names),
            checkpoints=checkpoints,
            session_stats=snapshot_session_stats,
            level_recal_medians=level_recal_anchor_used,
            state_names=state_names,
        )
        provenance: dict[str, object] = {
            "protocol": "cross-day-pooled",
            "fit_runs": fit_run_names,
            "held_out_test_run": test_run.name,
            "k": k,
            "pool_members": side_provenance,
        }
        if snapshot_session_stats is not None:
            provenance["session_norm"] = {
                "norm_minutes": norm_minutes,
                "pool_stats_n_windows": snapshot_session_stats.n_windows,
                "per_run_stats_n_windows": {
                    name: stats.n_windows
                    for name, stats in (stats_by_run or {}).items()
                },
            }
        save_snapshot(save_snapshot_path, snapshot, provenance=provenance)
        print(f"run_step2: saved pooled snapshot to {save_snapshot_path}")

    print(
        f"run_step2: wrote cross-day-pooled (held-out-day-group) tables for test run "
        f"{test_run.name!r} (fit pool: {', '.join(fit_run_names)}) to {out_dir}"
    )
    return 0


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
    fit_run_names = (
        [name.strip() for name in args.fit_runs.split(",") if name.strip()]
        if args.fit_runs is not None
        else None
    )
    if args.protocol == "cross-day-pooled":
        if args.fit_runs is None or args.test_run is None:
            parser.error(
                "--protocol cross-day-pooled requires both --fit-runs and --test-run "
                "(held-out-day-group evaluation, package-7 spec D2/A3.8)"
            )
        assert fit_run_names is not None
        if not fit_run_names:
            parser.error("--fit-runs got an empty run-name list")
        if len(set(fit_run_names)) != len(fit_run_names):
            parser.error(
                "--fit-runs contains duplicate run name(s) -- a run pools its sides "
                "once"
            )
        if run_names is not None:
            parser.error(
                "--protocol cross-day-pooled takes --fit-runs/--test-run, not --run "
                "(the fit/test roles must be explicit -- spec A3.1)"
            )
        if args.test_run in fit_run_names:
            parser.error(
                f"--test-run {args.test_run!r} is listed in --fit-runs: pool "
                f"artifacts are NEVER evaluated on pool-member runs (spec A3.1 -- "
                f"recalibrate mode would draw calibration windows the pool already "
                f"contains, frozen mode would score the reference windows themselves)"
            )
        if args.labels == "gt":
            parser.error(
                "--protocol cross-day-pooled is detected-labels only (the pooled "
                "detector defines the ONE shared label space; GT appears only in "
                "the coverage overlay)"
            )
        if args.scorer == "all":
            parser.error(
                "--protocol cross-day-pooled runs exactly one scorer per invocation "
                "(its output layout has no scorer axis) -- pick one"
            )
        if args.k is not None and args.k < 1:
            parser.error(f"--k must be >= 1, got {args.k}")
        if args.save_snapshot is not None and args.scorer not in ("knn", "mahalanobis"):
            parser.error(
                f"--save-snapshot requires a runtime scorer (knn or mahalanobis -- "
                f"the snapshot whitelist, rowii.runtime.snapshot), got --scorer "
                f"{args.scorer!r}"
            )
        if args.norm_minutes is not None and not args.session_norm:
            parser.error(
                "--norm-minutes requires --session-norm (it parameterizes the "
                "per-run session stats, spec D3)"
            )
        if args.norm_minutes is not None and args.norm_minutes <= 0:
            parser.error(f"--norm-minutes must be > 0, got {args.norm_minutes}")
        if args.level_recal and args.session_norm:
            parser.error(
                "--level-recal is mutually exclusive with --session-norm "
                "(package-8 spec D2/A1.4/A1.10 fit-path exclusivity)"
            )
        if args.level_recal and args.variant not in ("audio", "vibration"):
            parser.error(
                f"--level-recal requires --variant audio or vibration: fusion's "
                f"per-run z-scored features have no meaningful level column "
                f"(spec A1.1) -- got --variant {args.variant!r}"
            )
    else:
        for flag, value in (
            ("--fit-runs", args.fit_runs),
            ("--test-run", args.test_run),
            ("--k", args.k),
            ("--save-snapshot", args.save_snapshot),
            # store_true flag: only its True state is a usage error elsewhere.
            ("--session-norm", args.session_norm or None),
            ("--norm-minutes", args.norm_minutes),
            ("--level-recal", args.level_recal or None),
        ):
            if value is not None:
                parser.error(
                    f"{flag} requires --protocol cross-day-pooled -- got --protocol "
                    f"{args.protocol!r}"
                )
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
    if args.ensemble and args.variant not in _ENSEMBLE_VARIANTS:
        parser.error(
            "--ensemble requires a mic-primary --variant "
            f"({', '.join(_ENSEMBLE_VARIANTS)}): the ensemble's three-way split is "
            "drawn from the sweep variant's own segment_ids and reused to index the "
            "logmel features the LSTM-AE member consumes -- only mic-primary "
            "variants share recording-segment boundaries with logmel by "
            "construction, so any other variant would silently void the LSTM-AE "
            "member's leakage-safety (its grid can match logmel's by coincidence "
            "while its segment boundaries do not; see _ENSEMBLE_VARIANTS) "
            f"-- got --variant {args.variant!r}"
        )

    if args.xattn_fusion and args.protocol != "within-day":
        parser.error("--xattn-fusion is only valid with --protocol within-day")
    if args.xattn_fusion and args.labels == "gt":
        parser.error("--xattn-fusion is detected-labels only (spec D8)")
    if args.xattn_fusion and args.variant != "fusion":
        parser.error(
            "--xattn-fusion requires --variant fusion: the view reads the fusion "
            "cache's vibration columns (rowii.anomaly.fusion.split_branch_columns) "
            "as the head's vibration side"
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
    if args.xattn_fusion and cfg.xattn_checkpoint is None:
        parser.error(
            "--xattn-fusion needs ROWII_XATTN_CHECKPOINT set to a "
            "scripts/train_xattn.py checkpoint"
        )
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

    if args.protocol == "cross-day-pooled":
        assert fit_run_names is not None and args.test_run is not None  # guarded above
        unknown = _unknown_run_names([*fit_run_names, args.test_run], index)
        if unknown:
            # Same warm_cache-precedent hard error as the --run branch above.
            available = ", ".join(sorted({r.name for r in index.runs})) or "(none discovered)"
            print(
                f"run_step2: unknown run name(s): {', '.join(unknown)}; "
                f"available runs: {available}",
                file=sys.stderr,
            )
            return 2
        runs_by_name = {run.name: run for run in index.runs}
        fit_runs = [runs_by_name[name] for name in fit_run_names]
        test_run_obj = runs_by_name[args.test_run]
        # Day-group disjointness: day groups = the SET of calendar days each
        # run's burst-file names touch (_run_day_groups -- catches both the
        # sibling-runs-of-one-day case, e.g. 010726-tu1 vs 010726-tu2, AND a
        # midnight-crossing run whose tail shares a date with another run's day).
        test_days = _run_day_groups(test_run_obj)
        overlapping = sorted(
            {
                run.name
                for run in fit_runs
                if _run_day_groups(run) & test_days
            }
        )
        if overlapping:
            parser.error(
                f"--protocol cross-day-pooled requires disjoint day groups "
                f"(held-out-day-group evaluation, spec A3.8): test run "
                f"{args.test_run!r} shares calendar day(s) "
                f"{', '.join(sorted(test_days))} with fit run(s) "
                f"{', '.join(overlapping)}"
            )
        if args.conditioning not in ("all", "per-state"):
            logger.info(
                "run_step2: --conditioning=%r is ignored for --protocol "
                "cross-day-pooled (always per-state -- the pooled detector defines "
                "the label space, module docstring)",
                args.conditioning,
            )
        return _run_cross_day_pooled(
            fit_runs,
            test_run_obj,
            args.variant,
            cfg,
            index,
            args.scorer,
            args.alpha,
            args.top_k,
            k=args.k if args.k is not None else _POOLED_DEFAULT_K,
            use_cache=not args.no_cache,
            save_snapshot_path=args.save_snapshot,
            session_norm=args.session_norm,
            norm_minutes=(
                args.norm_minutes
                if args.norm_minutes is not None
                else _DEFAULT_NORM_MINUTES
            ),
            level_recal=args.level_recal,
        )

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
                xattn_fusion=args.xattn_fusion,
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
