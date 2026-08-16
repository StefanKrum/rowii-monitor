"""Replay driver -- "calibrate once, recalibrate only on a label-free drift
sentinel".

**Framing (binding honesty).** A retrospective, day-granular SIMULATION
over the recorded days -- NOT an online detector, NOT a persistent recalibrated-
baseline state machine (explicitly out of scope). The chronological order below
drives WHEN a sentinel fires in the report, not a causal runtime decision:
sentinels are label-free AT RUNTIME, but every threshold they fire against is
derived ONCE from the commissioning (B1) pool alone. "Once-calibrated" is
scoped to "once per instrumentation era" -- a caught era boundary (the
2026-06-29 `MeasName` changeover) IS the success criterion, not a failure of the
"once" goal.

**Pipeline.** One `--representation` (`fusion`/`vibration`/`audio-beats`) per
invocation (the `run_step2` one-arm rule) scores FAR through a pre-built B1
snapshot (`--snapshot`, built by `scripts/run_step2.py --protocol
cross-day-pooled --save-snapshot`). The two drift sentinels
(`rowii.anomaly.sentinels`) are REPRESENTATION-INDEPENDENT by design: s1
always reads the **audio-beats** mode bank ("the era-drift-robust
representation whose columns are never refused by the contract guard across
eras"); s2 always reads the RAW `audio`/`vibration` caches (never a
representation's own, possibly z-scored, columns). Both sentinels'
thresholds are commissioned ONCE, on `--bank-fit-runs` (the B1 pool, era B --
always exactly `_B1_FIT_RUNS`) via its CONFORMAL (held-out) side,
never a monitored day and never a partner figure (a strict no-partner-data
firewall). Per monitored
day this driver then: (1) evaluates s1 (`no_mode_fits` day-rate vs the B1
bootstrap threshold) and s2 (raw mic/vibration day-median vs the B1 anchor+MAD
band, with the `RAWGeneratorVib__2`-only vibration cross-check attributing the
cause) -- the SAME `s1 or s2` verdict gates all three FAR regimes; (2)
subprocess-invokes `scripts/monitor.py` in `--thresholds frozen` and
`--thresholds recalibrate` (script-sibling rule: a script never imports another
script's internals, so monitor.py runs as a real subprocess, behind the
`_run_monitor` seam tests monkeypatch away); (3) reports three FAR
regimes -- always-frozen, always-recalibrate, once+triggered -- for NON-event
days ALL on the common recalibrate scoring-split window population, with the
full-population frozen FAR as a labeled secondary column;
for the EVENT-BEARING day (`080726-pu_strikes`) the headline instead sources
all three arms from `scripts/eval_events.py`'s own event-free
`realized_window_far` (the common-window/raw-parquet reading
silently includes the induced-strike windows, which correctly alarm,
inflating it ~2.5-3x), labeled `far_basis="event-free per eval_events"`,
keeping the raw scored-window FAR as the SAME labeled
`frozen_far_full_population` secondary so the two readings are never
conflated.

**Pinned run set ("no run enumeration by day root anywhere").** `_REPLAY`
is the robustness study's rotation run set in true chronological order, including `270626`
(era A, between `250526` and `290626`) as a SENTINEL-ONLY row: it has no
Betriebsdaten, so it gets no monitor.py/FAR row at all, only a trigger-log
entry. `250526-pu-afternoon` is deliberately EXCLUDED (only its fusion
cache exists on disk). `010726-tu_ph_tu`/`010726-pu` are B1 pool members, so
their frozen/once monitoring is IN-SAMPLE and tagged `"in-sample"` --
computed exactly like every other day, just labeled for interpretation.
`080726-pu_strikes` (era C, the induced-strike day) is EVENT-BEARING
(`events_csv` set): `--exclude-calibration-events
docs/groundtruth/080726_events_pu.csv` (by design: calibration windows must
never contain labelled events) is
passed to every `monitor.py` call for it so the RECALIBRATE threshold is
never contaminated by a strike window (frozen mode ignores the flag with a
warning -- it draws no calibration from the monitored run at all, so there is
nothing to protect there). That calibration protection is ORTHOGONAL to the
regime FAR reported for this entry: the raw `alarms.parquet` scored-window
mean is NOT event-free (the induced-strike windows are still scored -- under
recalibrate they are moved from calibration onto the scoring side by monitor's
own `_apply_calibration_exclusion`, and under frozen they were always scored
regardless of the flag -- and they correctly alarm, inflating the raw reading
~2.5-3x). The genuinely event-free reading is
`scripts/eval_events.py`'s own `realized_window_far` (computed over
`role=="scored"` windows OUTSIDE every tolerance-padded strike interval,
`rowii.eval.events.evaluate_events`'s contract) -- sourced here for ALL THREE
regime arms and labeled `far_basis="event-free per eval_events"` in the
regimes table/sidecar, with the raw scored-window FAR of the frozen arm kept
visible as the SAME labeled `frozen_far_full_population` secondary so the two
readings are never conflated. `080726-st_strikes` is used ONLY for the
pillar-3 event-retention check below, never a regime/trigger-log row of
its own.

**Pillar-3 TPR-retained readout.** For BOTH `080726` sessions (PU pumping, ST
standstill) this driver reuses the SAME frozen/recalibrate alarms.parquet
already produced for the FAR table (`080726-pu_strikes`) -- and, for PU, the
SAME `eval_events.py` summary already computed there for the
event-free regime FAR (`event_eval_by_run`, never re-invoked as a second
subprocess call for the identical alarms/events pair) -- or produced once more
under the SAME era-C decision (`080726-st_strikes` -- this check alone scopes
it, so it inherits era C's own trigger verdict rather than being
independently sentineled: both sessions were recorded the same day under the
same instrumentation era, so re-deriving an independent sentinel verdict for ST
would not change anything this driver could act on differently) and feeds ST
through `scripts/eval_events.py` (subprocess, `_run_eval_events` seam) at the
README pillar-3 tolerance (5 s) and this driver's own `--alpha`. Reporting BOTH
the once+triggered (recalibrate) TPR and the frozen-arm TPR side by side is the
story this replay makes explicit: "the sentinel firing on 080726 -> recalibrate
-> strikes remain detectable, whereas staying frozen across the era gives the
trivially-broken cross-era snapshot" -- both numbers, never only the favorable
one.

**Attribution.** The two-sentinel drift-monitoring idea echoes the
partner's own drift monitoring (Rodrigues & Zhang, 2026); every threshold and
every reported number here is computed from OUR OWN caches/artifacts -- no
partner JSON or number is read by any code in this module (a strict
no-partner-data firewall, inherited from the pipeline's other modules).

**alpha unification.** `--alpha` (default 0.01, matching the B1
snapshot-build command) is used uniformly for THREE distinct conformal
calibrations that would otherwise silently disagree: the s1 bank's own
per-mode conformal threshold (`ModeBank.fit`'s `alpha`), `monitor.py`'s
recalibrate-mode FAR threshold (`--alpha`, frozen mode ignores it), and the
pillar-3 `eval_events.py` readout "at alpha 0.01" (via monitor.py's recalibrate
call feeding it, not eval_events.py itself, which has no alpha concept).

Outputs (`_out_dir`, default `<results_root>/step2/once-calibrated/
<representation>/`): `<representation>_trigger_log.csv` (one row per `_REPLAY`
entry, sentinel-only included), `<representation>_regimes.csv` (one row per
FAR-bearing entry: the three regimes + the labeled `frozen_far_full_population`
secondary + `far_basis` -- `"common-window"` for non-event days, `"event-free
per eval_events"` for 080726), and `<representation>.json`
(everything above plus the s1/s2 threshold derivations -- including s1's
`pct`/`n_boot`/`seed` bootstrap knobs -- the era-boundary-caught
verdict, and the pillar-3 readout) -- every number traceable to a committed
artifact, negative results included.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.anomaly.pools import PoolResult, build_pool  # noqa: E402
from rowii.anomaly.sentinels import (  # noqa: E402
    level_series,
    s1_fires,
    s1_threshold,
    s2_anchor_mad,
    s2_attribution,
    s2_fires,
)
from rowii.anomaly.sweep import SweepConfig  # noqa: E402
from rowii.config import Config, load_config  # noqa: E402
from rowii.eval.events import ROLE_SCORED  # noqa: E402
from rowii.io.dataset import (  # noqa: E402
    RecordingIndex,
    Run,
    betriebsdaten_utc_offset_ns,
    discover,
    run_utc_offset_ns,
)
from rowii.io.gantner import read_header  # noqa: E402
from rowii.pipeline import PreparedRun, prepare_run  # noqa: E402
from rowii.runtime.snapshot import MonitorSnapshot, load_snapshot  # noqa: E402
from rowii.scada.labels import gt_labels, load_scada_window_means  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402
from rowii.state.modebank import ModeBank  # noqa: E402

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_GROUNDTRUTH_DIR = _REPO_ROOT / "docs" / "groundtruth"

_REPRESENTATION_CHOICES: tuple[str, ...] = ("fusion", "vibration", "audio-beats")

_B1_FIT_RUNS: tuple[str, ...] = (
    "010726-pu", "010726-tu1-morning", "010726-tu2", "010726-tu_ph_tu",
)
"""The commissioning pool (era B) -- always passed exactly as this
comma-separated set via `--bank-fit-runs`;
kept here as a documented constant (not itself enforced -- `--bank-fit-runs`
stays a real CLI argument, matching every other pooled driver in this repo)."""

_MIC_STREAMS: tuple[str, ...] = ("RAWGeneratorMic__0", "RAWTurbineMic__1")
"""s2's microphone stream pair -- duplicated from `scripts/analyze_days.py`'s
own module constant of the same name (script-sibling rule): the era-step
signature stream set."""
_VIB_CROSSCHECK_STREAMS: tuple[str, ...] = ("RAWGeneratorVib__2",)
"""s2's vibration cross-check stream -- **RAWGeneratorVib__2 ONLY**:
`RAWTurbineVib__3` reads std ~ 0 through 2026-07-01 (cabled only between eras B
and C, per the README), so it is excluded here rather than averaged in
alongside a genuinely live channel."""

_BANK_FAMILY: Literal["gaussian", "knn", "gmm"] = "knn"
"""s1's canonical bank family, PINNED (not a CLI flag): the README's
six-rotation table shows audio-beats bank
`knn` as the best of the three families (mean ARI 0.882 vs gaussian 0.476 /
gmm 0.746), and it is the SAME family the era-C zero-shot readout cites
(README: "Era-C zero-shot mode-ID readout
(audio-beats): knn accuracy 0.918 ... with a 19.5% no_mode_fits rate") was built
with."""
_BANK_MIN_REF = 20
"""`ModeBank.fit`'s own default fit-side reference floor --
pinned here rather than exposed, matching `_BANK_FAMILY`."""
_BANK_K = 5
"""`ModeBank.fit`'s `knn` family neighbour count -- its own default, matching
`scripts/run_modebank.py --k`'s default."""

_S1_BOOTSTRAP_PCT = 97.5
"""`rowii.anomaly.sentinels.s1_threshold`'s own internal bootstrap percentile
-- NOT one of `s1_threshold`'s parameters (the percentile is a
fixed, named standard-statistics constant, not a caller-configurable
knob, so it cannot be "passed" to `s1_threshold`); kept here ONLY so the
sidecar can echo the exact value `s1_threshold` uses internally, making the
bootstrap derivation fully self-documenting without a reader having to
open `sentinels.py`."""
_S1_BOOTSTRAP_N_BOOT = 1000
"""Passed EXPLICITLY to `s1_threshold` -- equal to its own
default, but stated as a named constant here so the sidecar's echoed value can
never silently drift from the value actually used, even if `s1_threshold`'s
own default ever changes."""
_S1_BOOTSTRAP_SEED = 7
"""Passed EXPLICITLY to `s1_threshold` -- see
`_S1_BOOTSTRAP_N_BOOT`'s docstring; the same reproducibility argument."""

_PILLAR3_TOLERANCE_S = 5.0
"""`scripts/eval_events.py --tolerance-s` value for the pillar-3 readout below --
the README pillar-3 section's own "corrected ground truth, +-5 s tolerance"
convention (eval_events.py's OWN CLI default is 0.0; this driver always passes
the value explicitly, never relying on that default)."""

_EVENTS_PU = _GROUNDTRUTH_DIR / "080726_events_pu.csv"
_EVENTS_ST = _GROUNDTRUTH_DIR / "080726_events_st.csv"

_TAG_SENTINEL_ONLY = "sentinel-only"
_TAG_IN_SAMPLE = "in-sample"
_TAG_EVENT_FREE_FAR = "event-free-far"

_FAR_BASIS_COMMON_WINDOW = "common-window"
"""`far_basis` value for a NON-event `_REPLAY` entry: the frozen arm's
FAR is subset onto the recalibrate arm's own scoring-split window population
(`_far_on_windows`/`_scoring_windows`) before the two are compared."""
_FAR_BASIS_EVENT_FREE = "event-free per eval_events"
"""`far_basis` value for an EVENT-BEARING `_REPLAY` entry (`entry.events_csv`
is not `None`, currently only `080726-pu_strikes`): the raw
`alarms.parquet` scored-window mean silently includes the induced-event
windows too (they correctly alarm, inflating it ~2.5-3x for 080726) -- the
correct event-free reading is `scripts/eval_events.py`'s own
`realized_window_far` (`rowii.eval.events.EventEvalResult`, computed over
non-event scored windows only), sourced here for ALL THREE regime arms."""

_PILLAR3_ST_RUN = "080726-st_strikes"
"""era C's standstill strike session -- NOT a `_REPLAY` entry ("for the
event-retention check ONLY"); monitored separately, below, under the SAME
era-C once+triggered decision `080726-pu_strikes` already established (module
docstring's Pillar-3 section)."""


@dataclass(frozen=True)
class _ReplayEntry:
    """One `_REPLAY` row: a single monitored RUN (not a bare calendar day --
    several pinned days carry two runs each), the unit s1/s2/monitor.py all
    operate on."""

    day: str
    """Calendar day-root string (e.g. `"250526"`), for grouping/reporting only."""
    era: str
    """`"A"` / `"B"` / `"C"` -- the DAQ-configuration era."""
    run: str
    """The discovered run name monitor.py/the sentinels are evaluated against."""
    tags: tuple[str, ...]
    """`()`, or any of `_TAG_SENTINEL_ONLY` / `_TAG_IN_SAMPLE` /
    `_TAG_EVENT_FREE_FAR` -- see module docstring."""
    events_csv: Path | None = None
    """`--exclude-calibration-events` target for this entry's monitor.py calls,
    or `None` for every entry except `080726-pu_strikes`."""


_REPLAY: tuple[_ReplayEntry, ...] = (
    _ReplayEntry(day="250526", era="A", run="250526-tu", tags=()),
    _ReplayEntry(day="250526", era="A", run="250526-pu-morning", tags=()),
    _ReplayEntry(
        day="270626", era="A", run="270626-pu_ph_pu_ph_pu_ph-1", tags=(_TAG_SENTINEL_ONLY,)
    ),
    _ReplayEntry(day="290626", era="B", run="290626-tu", tags=()),
    _ReplayEntry(day="290626", era="B", run="290626-pu", tags=()),
    _ReplayEntry(day="010726", era="B", run="010726-tu_ph_tu", tags=(_TAG_IN_SAMPLE,)),
    _ReplayEntry(day="010726", era="B", run="010726-pu", tags=(_TAG_IN_SAMPLE,)),
    _ReplayEntry(
        day="080726", era="C", run="080726-pu_strikes",
        tags=(_TAG_EVENT_FREE_FAR,), events_csv=_EVENTS_PU,
    ),
)
"""The pinned replay set, in true chronological order: 250526 (era
A) -> 270626 (era A, sentinel-only, its true position between 250526 and
290626) -> 290626 (era B) -> 010726 (era B, in-sample) -> 080726 (era C,
event-free FAR). `250526-pu-afternoon` is deliberately absent (only its
fusion cache exists on disk)."""


# ---------------------------------------------------------------------------
# Duplicated script-sibling helpers (a script must not import another script's
# internals -- module docstrings across this repo's scripts/ all state this
# rule; each helper below names which sibling script it mirrors).
# ---------------------------------------------------------------------------


def _unknown_run_names(names: list[str], index: RecordingIndex) -> list[str]:
    """Duplicated from `scripts/run_step2.py`'s helper of the same name: names
    in *names* with no matching discovered run, de-duplicated, in the order
    first seen."""
    known = {r.name for r in index.runs}
    return list(dict.fromkeys(n for n in names if n not in known))


def _betriebsdaten_for_grid(betriebsdaten: list[Path], grid: WindowGrid) -> list[Path]:
    """Duplicated from `scripts/run_step2.py`'s helper of the same name:
    Betriebsdaten files whose hourly span intersects *grid*'s true-UTC time
    range."""
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
    """Duplicated from `scripts/run_modebank.py`'s helper of the same name: the
    full-length `(W,)` object array of GT state strings for *run*. Needed ONLY
    for `--bank-fit-runs` (s1's commissioning bank needs SCADA-labelled fit AND
    conformal sides) -- no `_REPLAY` entry ever calls this, since every
    sentinel/FAR evaluation on a monitored day is label-free by construction.

    Raises:
        ValueError: *run*'s day has no Betriebsdaten coverage overlapping its
            own grid.
    """
    day_betriebsdaten = index.betriebsdaten_by_day.get(run.day_root, [])
    matched = (
        _betriebsdaten_for_grid(day_betriebsdaten, prepared.grid) if day_betriebsdaten else []
    )
    if not matched:
        raise ValueError(
            f"run {run.name!r} has no Betriebsdaten coverage overlapping its "
            f"own grid -- the s1 bank needs SCADA ground truth on every "
            f"--bank-fit-runs member (spec D1)"
        )
    scada = load_scada_window_means(
        matched, prepared.grid, audio_run_offset_ns=run_utc_offset_ns(run)
    )
    labels: np.ndarray = gt_labels(scada, cfg.gt, window_s=cfg.window.window_s)[
        "state"
    ].to_numpy()
    return labels


def _pool_gt_labels(pool: PoolResult, gt_by_run: dict[str, np.ndarray]) -> np.ndarray:
    """Duplicated from `scripts/run_modebank.py`'s helper of the same name: per
    stacked pool row, the GT mode-name STRING of its source window (object
    dtype)."""
    out = np.empty(pool.features.shape[0], dtype=object)
    for member_idx, member in enumerate(pool.members):
        mask = pool.run_index == member_idx
        out[mask] = gt_by_run[member.run_name][pool.window_index[mask]]
    return out


def _import_beats_or_exit() -> None:
    """Duplicated from `scripts/run_step2.py`/`scripts/run_modebank.py`
    (script-sibling rule): s1's bank is ALWAYS audio-beats (module docstring),
    so this driver needs the BEATs featurizer unconditionally, unlike
    run_modebank.py's own conditional call."""
    try:
        import rowii.signals.beats  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"BEATs featurizer not available ({exc}); the s1 sentinel bank is "
            f"ALWAYS audio-beats (module docstring) and cannot run without it"
        ) from exc


def _pool_block_ids(pool: PoolResult, prepared: dict[str, PreparedRun]) -> np.ndarray:
    """Per stacked pool row, a GLOBALLY unique block id for the segment-block
    statistics `rowii.anomaly.sentinels.s1_threshold`/`s2_anchor_mad` need.

    `PreparedRun.segment_ids` numbers burst FILES *within one run*, restarting
    at 0 for every run (`rowii.pipeline`'s own module docstring). B1's
    commissioning pool stacks FOUR runs (`_B1_FIT_RUNS`), so pooling their raw
    `segment_ids` directly would silently merge same-numbered segments from
    DIFFERENT runs/days into one "block" -- exactly the failure a block
    bootstrap exists to avoid (a block must be one independent physical
    recording span, not merely a disjoint row range that happens to share a
    small integer with another run's own numbering). This assigns each
    distinct (run, local-segment) PAIR its own dense id via
    `np.unique(..., axis=0, return_inverse=True)`, so `s1_threshold`'s /
    `s2_anchor_mad`'s internal `np.unique(segment_ids)` grouping sees exactly
    the physical burst-file blocks the commissioning pool actually contains --
    one block per (run, burst file), never conflated across runs. No existing
    caller in this repo pools multiple runs' `segment_ids` into one block
    statistic (`scripts/analyze_days.py::_block_bootstrap_ci` is single-run by
    construction) -- this is a new requirement.
    """
    local = np.empty(pool.features.shape[0], dtype=np.int64)
    for member_idx, member in enumerate(pool.members):
        mask = pool.run_index == member_idx
        local[mask] = prepared[member.run_name].segment_ids[pool.window_index[mask]]
    pairs = np.stack([pool.run_index, local], axis=1)
    _, block_ids = np.unique(pairs, axis=0, return_inverse=True)
    return np.asarray(block_ids, dtype=np.int64)


# ---------------------------------------------------------------------------
# Sentinel commissioning (B1 CONFORMAL side ONLY) + per-day evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _S1Commission:
    bank: ModeBank
    threshold: float
    baseline_rate: float
    """B1 conformal-side pooled `no_mode_fits` rate (the day-rate this driver
    reports alongside `threshold` for every monitored day)."""


def _commission_s1(
    prepared_fit: dict[str, PreparedRun],
    gt_by_run: dict[str, np.ndarray],
    feature_names: list[str],
    *,
    alpha: float,
    sweep_cfg: SweepConfig,
) -> _S1Commission:
    """Fit the audio-beats bank on `--bank-fit-runs`' FIT side and derive s1's
    firing threshold from its own CONFORMAL side -- mirrors
    `scripts/run_modebank.py::main`'s own fit/conformal pooling recipe exactly.

    Raises:
        ValueError: `build_pool`'s empty-pool guard, or `ModeBank.fit`'s own
            geometry/family guards.
    """
    pool_fit = build_pool(prepared_fit, "fit", sweep_cfg)
    pool_conformal = build_pool(prepared_fit, "conformal", sweep_cfg)
    if pool_fit.features.shape[0] == 0 or pool_conformal.features.shape[0] == 0:
        raise ValueError(
            "_commission_s1: the pooled FIT or CONFORMAL side of --bank-fit-runs "
            "is empty (every fit run's splits were degenerate -- see the "
            "build_pool warnings above) -- nothing to commission s1 on"
        )
    fit_gt = _pool_gt_labels(pool_fit, gt_by_run)
    calib_gt = _pool_gt_labels(pool_conformal, gt_by_run)
    bank = ModeBank.fit(
        pool_fit.features, fit_gt, pool_conformal.features, calib_gt,
        family=_BANK_FAMILY, alpha=alpha, feature_names=feature_names,
        min_ref=_BANK_MIN_REF, k=_BANK_K,
    )
    if bank.low_confidence_modes:
        logger.warning(
            "run_once_calibrated: s1 bank has %d low_confidence member(s) %s -- "
            "no_mode_fits UNDER-fires for these (ModeBank.assign's own caveat); "
            "surfaced in the trigger log alongside every day's rate",
            len(bank.low_confidence_modes), bank.low_confidence_modes,
        )
    conformal_assignment = bank.assign(pool_conformal.features)
    block_ids = _pool_block_ids(pool_conformal, prepared_fit)
    threshold = s1_threshold(
        conformal_assignment.no_mode_fits, block_ids,
        n_boot=_S1_BOOTSTRAP_N_BOOT, seed=_S1_BOOTSTRAP_SEED,
    )  # Explicit, never s1_threshold's implicit defaults
    baseline_rate = (
        float(conformal_assignment.no_mode_fits.mean())
        if conformal_assignment.no_mode_fits.size
        else float("nan")
    )
    return _S1Commission(bank=bank, threshold=threshold, baseline_rate=baseline_rate)


@dataclass(frozen=True)
class _S2Commission:
    anchor: float
    mad: float


def _commission_s2(
    prepared_fit: dict[str, PreparedRun],
    feature_names: list[str],
    streams: tuple[str, ...],
    *,
    sweep_cfg: SweepConfig,
) -> _S2Commission:
    """B1 CONFORMAL-side (same held-out side as s1) anchor/MAD band for
    ONE stream set -- called once for the mic pair and once (separately) for
    the `RAWGeneratorVib__2` cross-check, each on its OWN raw
    audio/vibration `prepared_fit` (never a representation's own columns).

    Raises:
        ValueError: `build_pool`'s empty-pool guard.
    """
    pool_conformal = build_pool(prepared_fit, "conformal", sweep_cfg)
    if pool_conformal.features.shape[0] == 0:
        raise ValueError(
            "_commission_s2: the pooled CONFORMAL side of --bank-fit-runs is "
            "empty -- nothing to commission s2 on"
        )
    level_values = level_series(pool_conformal.features, feature_names, streams)
    block_ids = _pool_block_ids(pool_conformal, prepared_fit)
    anchor, mad = s2_anchor_mad(level_values, block_ids)
    return _S2Commission(anchor=anchor, mad=mad)


def _day_s1_rate(bank: ModeBank, prepared: PreparedRun) -> float:
    """One monitored day's label-free `no_mode_fits` rate under the
    commissioned bank (`bank.assign`, VALID windows only). `NaN` when the run
    has zero valid windows.

    Raises:
        ValueError: *prepared*'s audio-beats `feature_names` disagree with the
            bank's own fit-time contract (defensive -- audio-beats is not
            expected to drift, module docstring, but `bank.assign` scores
            POSITIONALLY and a silent misalignment must never happen).
    """
    valid = prepared.valid_mask
    if not bool(valid.any()):
        return float("nan")
    if list(prepared.feature_names) != bank.feature_names:
        raise ValueError(
            f"_day_s1_rate: prepared audio-beats feature_names "
            f"({len(prepared.feature_names)} column(s)) do not match the s1 "
            f"bank's own fit-time contract ({len(bank.feature_names)} "
            f"column(s)) -- audio-beats is the era-drift-free representation "
            f"(module docstring) and is not expected to drift; refusing a "
            f"positionally-misaligned score rather than silently mis-scoring"
        )
    assignment = bank.assign(prepared.features[valid])
    return float(assignment.no_mode_fits.mean()) if assignment.no_mode_fits.size else float("nan")


def _day_s2_median(prepared: PreparedRun, streams: tuple[str, ...]) -> float:
    """One monitored day's raw mic/vibration level MEDIAN (VALID windows only,
    `level_series` averaged per window then medianed across windows).
    `level_series` looks up columns BY NAME against *prepared*'s own
    `feature_names` (never the B1 pool's), so channel-availability drift
    between the commissioning pool and a monitored day never misaligns this
    call the way a positional score would -- only a TOTAL absence of
    stream-intersect-level columns on this run raises (`level_series`'s own
    contract). `NaN` when the run has zero valid windows."""
    valid = prepared.valid_mask
    if not bool(valid.any()):
        return float("nan")
    values = level_series(prepared.features[valid], prepared.feature_names, streams)
    return float(np.median(values))


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested directly, tests/test_run_once_calibrated.py) --
# the plan's own literal RED tests target these six.
# ---------------------------------------------------------------------------


def _read_realized_far(alarms_path: Path) -> float:
    """The realized aggregate FAR over `role == "scored"` windows of one
    monitor.py `alarms.parquet` -- the FULL-population reading (the labeled
    secondary column for the frozen arm; the primary reading for every other
    arm, which IS its own common population by construction)."""
    df = pd.read_parquet(alarms_path)
    scored = df[df["role"] == ROLE_SCORED]
    return float(scored["alarm"].mean()) if len(scored) else float("nan")


def _scoring_windows(alarms_path: Path) -> np.ndarray:
    """The `window` ids of every `role == "scored"` row of one `alarms.parquet`
    -- the recalibrate arm's own scoring-split population, the common
    population every OTHER arm's FAR is subset onto for the headline table."""
    df = pd.read_parquet(alarms_path)
    return np.asarray(df.loc[df["role"] == ROLE_SCORED, "window"].to_numpy())


def _far_on_windows(alarms_path: Path, window_set: np.ndarray) -> float:
    """The realized FAR of one `alarms.parquet`, restricted to `role ==
    "scored"` rows whose `window` id is in *window_set* (comparing the
    frozen arm's FAR against the recalibrate arm's on the SAME window
    population, since the two arms otherwise score different-sized/composed
    scoring sides by construction)."""
    df = pd.read_parquet(alarms_path)
    keep = {int(w) for w in window_set}
    sub = df[(df["role"] == ROLE_SCORED) & df["window"].isin(keep)]
    return float(sub["alarm"].mean()) if len(sub) else float("nan")


def _trigger_verdict(*, s1_fired: bool, s2_fired: bool) -> bool:
    """The day-level once+triggered decision: recalibrate iff EITHER sentinel
    fired ("the SAME day-level trigger verdict (s1 or s2) gates the
    frozen/recalibrate choice for all three FAR arms")."""
    return bool(s1_fired or s2_fired)


def _regime_far(frozen_far: float, recal_far: float, *, triggered: bool) -> float:
    """The once+triggered regime's FAR for one day: the recalibrate FAR if
    triggered, else the frozen FAR -- PER DAY, deliberately NOT sticky (no
    persistent recalibrated-baseline state machine -- out-of-scope)."""
    return recal_far if triggered else frozen_far


def _trigger_log_row(
    *,
    day: str,
    era: str,
    tags: tuple[str, ...],
    s1_rate: float,
    s1_threshold: float,
    low_confidence_modes: tuple[str, ...],
    s2_mic: float,
    s2_vib: float,
    anchor: float,
    mad: float,
    attribution: str,
    decision: str,
) -> dict[str, object]:
    """One trigger-log row: every sentinel diagnostic for one monitored day/run,
    self-consistently deriving `s1_fired`/`s2_fired` from the SAME
    `rowii.anomaly.sentinels.s1_fires`/`s2_fires` predicates the driver uses
    everywhere else (never re-implemented ad hoc here). NEVER carries a `"far"`
    key -- FAR fields are merged in by the caller for every entry EXCEPT
    sentinel-only days (`_TAG_SENTINEL_ONLY`: no Betriebsdaten -> no
    FAR/GT row), so this function's own output is identical either way and the
    "no far row" property holds by construction, not by a branch inside this
    function."""
    s1_fired = s1_fires(s1_rate, s1_threshold)
    s2_fired = s2_fires(s2_mic, anchor, mad)
    return {
        "day": day,
        "era": era,
        "tags": tags,
        "s1_rate": s1_rate,
        "s1_threshold": s1_threshold,
        "s1_fired": s1_fired,
        "low_confidence_modes": low_confidence_modes,
        "s2_mic_median": s2_mic,
        "s2_vib_median": s2_vib,
        "s2_anchor": anchor,
        "s2_mad": mad,
        "s2_fired": s2_fired,
        "s2_attribution": attribution,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Subprocess seams (script-sibling rule: monitor.py/eval_events.py run as REAL
# subprocesses, never imported -- monkeypatched in tests).
# ---------------------------------------------------------------------------


def _run_monitor(
    snapshot_path: Path,
    run: str,
    mode: str,
    out_dir: Path,
    *,
    alpha: float | None = None,
    event_free: Path | None = None,
) -> Path:
    """Subprocess-invoke `scripts/monitor.py` for one (run, threshold mode) and
    return its written `alarms.parquet` path. `alpha` is only ever passed for
    `mode == "recalibrate"` (frozen mode ignores `--alpha` with a warning, so
    this driver never triggers that warning). `event_free`, when given, is
    passed as `--exclude-calibration-events` regardless of mode -- harmless
    (frozen mode also only warns on it) and keeps the pillar-3 rule applied
    uniformly for the one induced-event day (module docstring).
    """
    cmd = [
        sys.executable, str(_SCRIPTS_DIR / "monitor.py"),
        "--snapshot", str(snapshot_path),
        "--run", run,
        "--thresholds", mode,
        "--out", str(out_dir),
    ]
    if mode != "frozen" and alpha is not None:
        cmd += ["--alpha", str(alpha)]
    if event_free is not None:
        cmd += ["--exclude-calibration-events", str(event_free)]
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)
    return out_dir / "alarms.parquet"


def _run_eval_events(
    alarms_path: Path, events_path: Path, out_dir: Path, *, tolerance_s: float
) -> Path:
    """Subprocess-invoke `scripts/eval_events.py` on one alarms/events pair and
    return its written `event_eval.csv` path."""
    cmd = [
        sys.executable, str(_SCRIPTS_DIR / "eval_events.py"),
        "--alarms", str(alarms_path),
        "--events", str(events_path),
        "--tolerance-s", str(tolerance_s),
        "--out", str(out_dir),
    ]
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)
    return out_dir / "event_eval.csv"


def _read_event_tpr(event_eval_csv: Path) -> float:
    """The pillar-3 headline number out of one `event_eval.csv`: the summary
    row's `event_tpr` (`scripts/eval_events.py`'s own `row_type == "summary"`
    contract)."""
    df = pd.read_csv(event_eval_csv)
    summary = df[df["row_type"] == "summary"]
    if summary.empty:
        return float("nan")
    return float(summary.iloc[0]["event_tpr"])


def _read_realized_window_far(event_eval_csv: Path) -> float:
    """The EVENT-FREE aggregate FAR out of one `event_eval.csv`'s summary row:
    `realized_window_far` (`rowii.eval.events.EventEvalResult`'s own contract
    -- false alarms restricted to `role == "scored"` windows OUTSIDE every
    tolerance-padded event interval, `scripts/eval_events.py`'s own
    `row_type == "summary"` contract). For an EVENT-BEARING `_REPLAY` entry
    this is the ONLY correct "event-free window-FAR" reading -- `_read_
    realized_far`/`_far_on_windows` over the raw `alarms.parquet` silently
    score the induced-event windows too (they correctly alarm, inflating the
    raw reading ~2.5-3x for 080726)."""
    df = pd.read_csv(event_eval_csv)
    summary = df[df["row_type"] == "summary"]
    if summary.empty:
        return float("nan")
    return float(summary.iloc[0]["realized_window_far"])


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------


def _out_dir(results_root: Path, representation: str) -> Path:
    """`results/step2/once-calibrated/<representation>/` -- mirrors
    `scripts/sweep_min_dwell.py`'s own `_out_dir` convention."""
    return results_root / "step2" / "once-calibrated" / representation


def _era_triggered(trigger_log: list[dict[str, object]], era: str) -> bool:
    """`True` iff ANY `_REPLAY` entry of *era* triggered a recalibrate decision
    -- the "is the era boundary caught" readout."""
    return any(
        row["era"] == era and row["decision"] == "recalibrate" for row in trigger_log
    )


# ---------------------------------------------------------------------------
# argparse + main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "D1: replay the pinned P9 run set chronologically against a B1 "
            "(era-B) snapshot, evaluate two label-free drift sentinels per day "
            "(s1 audio-beats mode-bank rejection, s2 raw mic/vibration level-"
            "step), and report three FAR regimes -- always-frozen, always-"
            "recalibrate, once+triggered -- on the common recalibrate scoring-"
            "split population (A1.6), plus a trigger log and the 080726 "
            "pillar-3 TPR-retained readout (spec §3.D1)."
        )
    )
    parser.add_argument(
        "--representation", required=True, choices=_REPRESENTATION_CHOICES,
        help="The ONE FAR-scored variant this invocation replays (the "
             "run_step2 one-arm rule) -- must match --snapshot's own fitted "
             "variant. The two sentinels are representation-INDEPENDENT "
             "(module docstring) and are evaluated regardless of this choice.",
    )
    parser.add_argument(
        "--snapshot", type=Path, required=True,
        help="The B1 (era-B) MonitorSnapshot .npz for --representation, built "
             "by scripts/run_step2.py --protocol cross-day-pooled --save-"
             "snapshot (plan Task 7).",
    )
    parser.add_argument(
        "--bank-fit-runs", required=True,
        help="Comma-separated commissioning pool for s1/s2 (Task 7 always "
             "passes _B1_FIT_RUNS's four runs); order matters "
             "(ModeBank.fit's pooling order).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01,
        help="Shared conformal alpha (default 0.01, matching Task 7's own "
             "snapshot-build command): the s1 bank's per-mode threshold, "
             "monitor.py's recalibrate-mode FAR threshold, and the pillar-3 "
             "eval_events readout (module docstring's alpha-unification note).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output root (default: <results_root>/step2/once-calibrated/"
             "<representation>/).",
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

    bank_fit_run_names = [n.strip() for n in str(args.bank_fit_runs).split(",") if n.strip()]
    if not bank_fit_run_names:
        parser.error("--bank-fit-runs got an empty run-name list")
    if len(set(bank_fit_run_names)) != len(bank_fit_run_names):
        parser.error("--bank-fit-runs contains duplicate run name(s)")

    if not args.snapshot.is_file():
        print(f"run_once_calibrated: snapshot file not found: {args.snapshot}", file=sys.stderr)
        return 2
    try:
        snapshot: MonitorSnapshot = load_snapshot(args.snapshot)
    except ValueError as exc:
        print(f"run_once_calibrated: cannot load snapshot {args.snapshot}: {exc}", file=sys.stderr)
        return 2
    if snapshot.variant != args.representation:
        print(
            f"run_once_calibrated: --representation {args.representation!r} does "
            f"not match --snapshot's own fitted variant {snapshot.variant!r} "
            f"({args.snapshot}) -- refusing a mismatched FAR arm",
            file=sys.stderr,
        )
        return 2

    _import_beats_or_exit()  # s1's bank is ALWAYS audio-beats (module docstring)

    cfg = load_config()
    index = discover(cfg.data_root)

    replay_run_names = [entry.run for entry in _REPLAY]
    all_needed = [*bank_fit_run_names, *replay_run_names, _PILLAR3_ST_RUN]
    unknown = _unknown_run_names(all_needed, index)
    if unknown:
        available = ", ".join(sorted({r.name for r in index.runs})) or "(none discovered)"
        print(
            f"run_once_calibrated: unknown run name(s): {', '.join(unknown)}; "
            f"available runs: {available}",
            file=sys.stderr,
        )
        return 2
    runs_by_name = {r.name: r for r in index.runs}

    sweep_cfg = SweepConfig(alpha=float(args.alpha))

    # --- Prepare every run needed for the sentinels (audio-beats/audio/vibration
    # ONLY -- monitor.py prepares --representation itself, internally, per run). ---
    sentinel_variants = ("audio-beats", "audio", "vibration")
    prepared_by_variant: dict[str, dict[str, PreparedRun]] = {v: {} for v in sentinel_variants}
    for run_name in dict.fromkeys(all_needed):  # de-duplicated, order preserved
        run_obj = runs_by_name[run_name]
        for variant in sentinel_variants:
            try:
                prepared_by_variant[variant][run_name] = prepare_run(
                    run_obj, variant, cfg, use_cache=not args.no_cache
                )
            except RuntimeError as exc:
                print(
                    f"run_once_calibrated: prepare_run failed for run "
                    f"{run_name!r} variant {variant!r} ({exc})",
                    file=sys.stderr,
                )
                return 2

    prepared_fit_beats = {n: prepared_by_variant["audio-beats"][n] for n in bank_fit_run_names}
    prepared_fit_audio = {n: prepared_by_variant["audio"][n] for n in bank_fit_run_names}
    prepared_fit_vib = {n: prepared_by_variant["vibration"][n] for n in bank_fit_run_names}

    gt_by_run: dict[str, np.ndarray] = {}
    for run_name in bank_fit_run_names:
        try:
            gt_by_run[run_name] = _run_gt_states(
                prepared_fit_beats[run_name], runs_by_name[run_name], index, cfg
            )
        except ValueError as exc:
            print(f"run_once_calibrated: {exc}", file=sys.stderr)
            return 2

    bank_feature_names = list(next(iter(prepared_fit_beats.values())).feature_names)
    try:
        s1 = _commission_s1(
            prepared_fit_beats, gt_by_run, bank_feature_names,
            alpha=float(args.alpha), sweep_cfg=sweep_cfg,
        )
        mic_feature_names = list(next(iter(prepared_fit_audio.values())).feature_names)
        s2_mic = _commission_s2(
            prepared_fit_audio, mic_feature_names, _MIC_STREAMS, sweep_cfg=sweep_cfg
        )
        vib_feature_names = list(next(iter(prepared_fit_vib.values())).feature_names)
        s2_vib = _commission_s2(
            prepared_fit_vib, vib_feature_names, _VIB_CROSSCHECK_STREAMS, sweep_cfg=sweep_cfg
        )
    except ValueError as exc:
        print(f"run_once_calibrated: sentinel commissioning failed: {exc}", file=sys.stderr)
        return 2

    out_root = Path(args.out) if args.out is not None else cfg.results_root
    out_dir = _out_dir(out_root, str(args.representation))
    out_dir.mkdir(parents=True, exist_ok=True)

    trigger_log: list[dict[str, object]] = []
    regimes: list[dict[str, object]] = []
    # run -> (frozen, recalibrate) event_eval.csv paths, EVENT-BEARING entries
    # only -- populated in the loop below and reused (never re-invoked) by the
    # pillar-3 section for 080726-pu_strikes.
    event_eval_by_run: dict[str, tuple[Path, Path]] = {}
    for entry in _REPLAY:
        prepared_beats = prepared_by_variant["audio-beats"][entry.run]
        prepared_audio = prepared_by_variant["audio"][entry.run]
        prepared_vib = prepared_by_variant["vibration"][entry.run]
        try:
            s1_rate = _day_s1_rate(s1.bank, prepared_beats)
            s2_mic_median = _day_s2_median(prepared_audio, _MIC_STREAMS)
            s2_vib_median = _day_s2_median(prepared_vib, _VIB_CROSSCHECK_STREAMS)
        except ValueError as exc:
            print(f"run_once_calibrated: {exc}", file=sys.stderr)
            return 2

        mic_fired = s2_fires(s2_mic_median, s2_mic.anchor, s2_mic.mad)
        vib_fired = s2_fires(s2_vib_median, s2_vib.anchor, s2_vib.mad)
        attribution = s2_attribution(mic_fires=mic_fired, vib_fires=vib_fired)
        s1_fired = s1_fires(s1_rate, s1.threshold)
        triggered = _trigger_verdict(s1_fired=s1_fired, s2_fired=mic_fired)
        decision = "recalibrate" if triggered else "frozen"

        row = _trigger_log_row(
            day=entry.day, era=entry.era, tags=entry.tags,
            s1_rate=s1_rate, s1_threshold=s1.threshold,
            low_confidence_modes=s1.bank.low_confidence_modes,
            s2_mic=s2_mic_median, s2_vib=s2_vib_median,
            anchor=s2_mic.anchor, mad=s2_mic.mad,
            attribution=attribution, decision=decision,
        )
        row["run"] = entry.run
        trigger_log.append(row)

        if _TAG_SENTINEL_ONLY in entry.tags:
            continue  # no Betriebsdaten -> no monitor.py/FAR row

        run_out = out_dir / "monitor" / entry.run
        try:
            frozen_alarms = _run_monitor(
                args.snapshot, entry.run, "frozen", run_out / "frozen",
                event_free=entry.events_csv,
            )
            recal_alarms = _run_monitor(
                args.snapshot, entry.run, "recalibrate", run_out / "recalibrate",
                alpha=float(args.alpha), event_free=entry.events_csv,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"run_once_calibrated: monitor.py failed for run {entry.run!r} "
                f"({exc})",
                file=sys.stderr,
            )
            return 2

        # The raw scored-window FAR of the frozen arm -- the SAME reading
        # regardless of entry type; for a NON-event day this IS its own
        # full-population secondary; for the EVENT-BEARING day the
        # "raw scored-window FAR" must stay visible too,
        # labeled and distinct from the event-free headline computed below.
        frozen_far_secondary = _read_realized_far(frozen_alarms)

        if entry.events_csv is not None:
            # EVENT-BEARING entry: the raw alarms.parquet
            # scored-window mean silently includes the induced-event windows
            # (they correctly alarm -- under recalibrate they were moved from
            # calibration onto the scoring side by monitor's own
            # `_apply_calibration_exclusion`; under frozen they were always
            # scored, flag or no flag), inflating it ~2.5-3x for 080726. The
            # correct event-free reading is eval_events' own
            # `realized_window_far`, sourced here for ALL THREE regime arms
            # (never the parquet mean). Cached in `event_eval_by_run` so the
            # pillar-3 section below reuses these exact CSVs instead of
            # re-invoking eval_events.py a second time for the same pair.
            try:
                frozen_eval = _run_eval_events(
                    frozen_alarms, entry.events_csv,
                    out_dir / "eval_events" / entry.run / "frozen",
                    tolerance_s=_PILLAR3_TOLERANCE_S,
                )
                recal_eval = _run_eval_events(
                    recal_alarms, entry.events_csv,
                    out_dir / "eval_events" / entry.run / "recalibrate",
                    tolerance_s=_PILLAR3_TOLERANCE_S,
                )
            except subprocess.CalledProcessError as exc:
                print(
                    f"run_once_calibrated: eval_events.py failed for run "
                    f"{entry.run!r} ({exc})",
                    file=sys.stderr,
                )
                return 2
            event_eval_by_run[entry.run] = (frozen_eval, recal_eval)
            frozen_far_headline = _read_realized_window_far(frozen_eval)
            recal_far_headline = _read_realized_window_far(recal_eval)
            far_basis = _FAR_BASIS_EVENT_FREE
        else:
            recal_far_headline = _read_realized_far(recal_alarms)
            scoring_windows = _scoring_windows(recal_alarms)
            frozen_far_headline = _far_on_windows(frozen_alarms, scoring_windows)
            far_basis = _FAR_BASIS_COMMON_WINDOW

        once_far = _regime_far(frozen_far_headline, recal_far_headline, triggered=triggered)

        regimes.append({
            "day": entry.day, "era": entry.era, "run": entry.run, "tags": entry.tags,
            "always_frozen_far": frozen_far_headline,
            "always_recalibrate_far": recal_far_headline,
            "once_triggered_far": once_far,
            "frozen_far_full_population": frozen_far_secondary,
            "far_basis": far_basis,
            "decision": decision,
        })

    # --- Pillar-3 TPR-retained readout (080726, PU + ST) -----------------------
    # PU's frozen/recalibrate eval_events summaries were already computed
    # above (event-bearing entry) -- reused here, never
    # re-invoked as a second subprocess call for the identical alarms/events
    # pair.
    pu_frozen_eval, pu_recal_eval = event_eval_by_run["080726-pu_strikes"]
    era_c_row = next(r for r in regimes if r["run"] == "080726-pu_strikes")
    era_c_decision = str(era_c_row["decision"])
    try:
        st_recal_alarms = _run_monitor(
            args.snapshot, _PILLAR3_ST_RUN, "recalibrate",
            out_dir / "monitor" / _PILLAR3_ST_RUN / "recalibrate",
            alpha=float(args.alpha), event_free=_EVENTS_ST,
        )
        st_frozen_alarms = _run_monitor(
            args.snapshot, _PILLAR3_ST_RUN, "frozen",
            out_dir / "monitor" / _PILLAR3_ST_RUN / "frozen",
            event_free=_EVENTS_ST,
        )
        st_recal_eval = _run_eval_events(
            st_recal_alarms, _EVENTS_ST,
            out_dir / "eval_events" / _PILLAR3_ST_RUN / "recalibrate",
            tolerance_s=_PILLAR3_TOLERANCE_S,
        )
        st_frozen_eval = _run_eval_events(
            st_frozen_alarms, _EVENTS_ST,
            out_dir / "eval_events" / _PILLAR3_ST_RUN / "frozen",
            tolerance_s=_PILLAR3_TOLERANCE_S,
        )
    except subprocess.CalledProcessError as exc:
        print(f"run_once_calibrated: pillar-3 evaluation failed ({exc})", file=sys.stderr)
        return 2

    pu_recal_tpr = _read_event_tpr(pu_recal_eval)
    pu_frozen_tpr = _read_event_tpr(pu_frozen_eval)
    st_recal_tpr = _read_event_tpr(st_recal_eval)
    st_frozen_tpr = _read_event_tpr(st_frozen_eval)
    pillar3: dict[str, dict[str, float]] = {
        "pu": {
            "once_triggered_event_tpr": (
                pu_recal_tpr if era_c_decision == "recalibrate" else pu_frozen_tpr
            ),
            "recalibrate_event_tpr": pu_recal_tpr,
            "frozen_event_tpr": pu_frozen_tpr,
        },
        "st": {
            "once_triggered_event_tpr": (
                st_recal_tpr if era_c_decision == "recalibrate" else st_frozen_tpr
            ),
            "recalibrate_event_tpr": st_recal_tpr,
            "frozen_event_tpr": st_frozen_tpr,
        },
    }

    boundary_caught = _era_triggered(trigger_log, "A")
    era_c_triggered = _era_triggered(trigger_log, "C")

    trigger_log_df = pd.DataFrame(trigger_log)
    trigger_log_df.to_csv(out_dir / f"{args.representation}_trigger_log.csv", index=False)
    regimes_df = pd.DataFrame(regimes)
    regimes_df.to_csv(out_dir / f"{args.representation}_regimes.csv", index=False)

    sidecar = {
        "representation": args.representation,
        "snapshot": str(args.snapshot),
        "bank_fit_runs": bank_fit_run_names,
        "alpha": args.alpha,
        "s1": {
            "family": _BANK_FAMILY, "k": _BANK_K, "min_ref": _BANK_MIN_REF,
            "baseline_rate": s1.baseline_rate, "threshold": s1.threshold,
            "pct": _S1_BOOTSTRAP_PCT, "n_boot": _S1_BOOTSTRAP_N_BOOT,
            "seed": _S1_BOOTSTRAP_SEED,
            "low_confidence_modes": list(s1.bank.low_confidence_modes),
        },
        "s2": {
            "mic_streams": list(_MIC_STREAMS), "vib_streams": list(_VIB_CROSSCHECK_STREAMS),
            "mic_anchor": s2_mic.anchor, "mic_mad": s2_mic.mad,
            "vib_anchor": s2_vib.anchor, "vib_mad": s2_vib.mad, "k": 3.0,
        },
        "trigger_log": trigger_log,
        "regimes": regimes,
        "boundary_caught_era_a": boundary_caught,
        "era_c_triggered": era_c_triggered,
        "pillar3": pillar3,
        "provenance_note": (
            "D1 replay driver: \"calibrate once, recalibrate only on a "
            "label-free drift sentinel\". The two-"
            "sentinel drift-monitoring idea echoes the partner's own drift "
            "monitoring (Rodrigues & Zhang, 2026); every threshold/number above "
            "is computed from our own caches (A1.1/A1.8 firewall) -- no partner "
            "JSON or number is read anywhere in this module. Retrospective, "
            "day-granular SIMULATION, never an online claim (D1 honesty 1). "
            "'once-calibrated' is scoped to 'once per instrumentation era' (D1 "
            "honesty 2); the 010726 rows are IN-SAMPLE (tags). 080726's regime "
            "FAR (far_basis='event-free per eval_events') is sourced from "
            "scripts/eval_events.py's own realized_window_far -- computed over "
            "role=='scored' windows OUTSIDE every tolerance-padded strike "
            "interval -- for all three regime arms, NEVER the raw "
            "alarms.parquet scored-window mean (which silently counts the "
            "induced-strike windows too, since they correctly alarm; that raw "
            "reading stays visible as the labeled frozen_far_full_population "
            "secondary field)."
        ),
    }
    (out_dir / f"{args.representation}.json").write_text(json.dumps(sidecar, indent=2) + "\n")

    print(
        f"run_once_calibrated: {args.representation} replay done -- era-A "
        f"boundary caught={boundary_caught}, era-C triggered={era_c_triggered} "
        f"-> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
