"""D3 explainability analysis suite (Package-8, spec `docs/superpowers/specs/
2026-07-21-step2-package8-modebank-recal-explain.md` §3.D3 + Amendment A1.2/A1.8/
A1.11, plan `docs/superpowers/plans/2026-07-21-step2-package8-modebank-explain.md`
Tasks 8+9): publication-grade figures + underlying CSVs from EXISTING artifacts/
warm caches -- no new sweeps, no partner JSON/number read by any code here.

This module ships all six of the D3 suite's analysis subcommands plus a
markdown digest. Five of the six share ONE read seam (`_run_features_and_gt`/
`_RunFeatures`) for everything that reads warm feature caches + GT;
`pillar3-figure` instead reads existing `results/pillar3/**/event_eval.csv`
artifacts directly (mirrors `rotations-heatmap`'s own direct-filesystem-read
style), and `digest` only reads THIS module's own output tree:

1. `rotations-heatmap` -- day x day pooled flag-rate heatmap, read straight from
   `results/step2/cross-day-pooled/<test_run>/<variant>-pooled<leaf_suffix>/
   far_table_<mode>.csv` (`--leaf-suffix`, default `""` -- e.g. `-a0.05` reaches
   an alpha-suffixed leaf instead of the plain one) (the `label == "pooled"`
   aggregate row) + that leaf's own `notes.md` "fit pool:" line -- the visual
   replacement for the FAR tables Stefan found unreadable (spec D3 point 1; no
   partner attribution -- this is Stefan's own usability motivation). Refuses to
   render below 2 discovered rotations (exit 2, listing what WAS found and
   hinting at `--leaf-suffix`) -- a 1-cell "matrix" is never silently plotted.
2. `feature-stability` -- per-feature cross-day shift (own stored units for the
   PRIMARY dot-interval figure: log10 for level columns, raw units for shape
   columns), per GT mode, on GT-bearing days only (A1.2), with 12-minute-block
   (`PreparedRun.segment_ids`, NEVER wall-clock, A1.11) bootstrap CIs. The
   CONTINUOUS per-day dot-interval figure is the PRIMARY deliverable; the binary
   slow(<3 dB)/drifting(>=3 dB) classification is SECONDARY and LEVEL-COLUMN-ONLY:
   each level column's own log10-domain shift is first converted to a genuine dB
   figure (`_level_db_factor` -- x20 for `_log_rms`, stored log10 RMS AMPLITUDE;
   x10 for `_band_`/`_octave_`, stored log10 mean Welch PSD POWER -- the standard
   amplitude/power dB relationship, computed entirely from our own stored
   features), THEN compared against a cutoff labeled "same cutoff as Rodrigues &
   Zhang (2026), adopted for comparability" (A1.8) -- an independent replication
   of their stability-classes ANALYSIS TYPE on our own features, never a claim
   that our shift value is numerically their dB figure. Shape columns (raw
   Hz/dimensionless units) keep their dot-interval row but are never classified
   (`"n/a"` -- a dB cutoff is meaningless in their own units); for the `fusion`
   variant, or any embedding variant (zero level columns), the dB classification
   is skipped entirely (logged warning) since those columns' shifts are per-run
   z-score / embedding units, never log10.
3. `era-step` -- per-day, per-STREAM-VARIANT (mic streams under `--variant
   audio`, vibration streams under `--variant vibration` -- `fusion` is refused,
   exit 2: `fuse()`'s per-run z-score, A1.1, has no meaningful raw-scale level to
   plot here; the mic-vs-vibration COMPOSITE view needs one run of EACH variant,
   each its own CSV/PNG) median level in GT-matched modes across the recording
   era, with 2026-06-27 (no Betriebsdaten at all) shown as an UN-MATCHED point
   flagged "no GT -- era-A anchor by MeasName only", and 2026-07-08 (080726)
   included ONLY behind the explicit `--include-080726` gate (A1.2 -- gated on
   the D4.3 SCADA-timebase probe, `scripts/verify_data_facts.py
   scada-timebase`). Consistent with, and attributed alongside, the partner's own
   independently reported mic-only broadband level step at the same era boundary
   (Rodrigues & Zhang, 2026) -- our own number, computed only from our own caches.
4. `mode-signatures` (Task 9) -- per-GT-mode band/octave profile (median +
   interquartile range) for ONE day at a time -- the "modes are separable"
   picture on our own features, restricted to the `_band_`/`_octave_`
   spectral-shape columns (narrower than `rowii.anomaly.levelrecal.
   level_columns`, which also folds in the single-scalar `_log_rms` loudness
   column -- not part of a "profile" shape). One PNG + CSV PER RUN (the
   comparison this figure makes is WITHIN one day, across modes, unlike
   feature-stability/era-step's ACROSS-day comparison). An independent,
   per-day replication of the partner's reported within-day mode
   separability (Rodrigues & Zhang, 2026), computed only from our own
   caches. The figure's own title names the *variant* it was rendered from
   (T9-review item 1, interpretation honesty), and its x-axis units follow
   `_feature_unit_label`: genuine log10 for a raw-scale variant, `fusion`'s
   own per-run z-score for `fusion` (A1.1 -- `fuse()`'s columns keep their
   `_band_`/`_octave_` name tokens even though the VALUES are z-scored, so
   the check is on *variant*, not column names), or that model's own
   embedding units for an embedding variant -- never a blanket "log10"
   claim regardless of variant.
5. `tonal-table` (Task 9) -- per (run, GT mode, physical stream) the three
   `rowii.signals.features.MACHINE_HZ` machine-tone band energies (shaft,
   blade-pass, guide-vane-pass) contrasted against their own NEAREST
   TONE-FREE OCTAVE FLOOR (`_nearest_octave_hz`, T9-review item 2: the
   octave center nearest by Hz among those actually present for that
   stream, EXCLUDING any candidate whose own `[center/sqrt(2),
   center*sqrt(2)]` span already contains the tone -- a "floor" that
   itself contains the tone would leak the tone's own energy into its own
   background reading, defeating the contrast's purpose; if every
   candidate's span contains the tone, all re-enter the pool with a
   logged warning rather than the tone silently dropping out) --
   `_tonal_contrast = band_energy - octave_floor`, an SNR-like contrast
   defined ENTIRELY from our own band/octave features, explicitly NOT the
   partner's exact metric (A1.8) -- it only shares the analysis TYPE (a
   machine-fingerprint table) with Rodrigues & Zhang (2026). Own stored
   units throughout: log10 for the audio/vibration variants (a genuine, if
   uncalibrated, level-ratio reading); for `fusion`, whose `fuse()` step
   per-run z-scores every column before concatenation (A1.1), the same
   subtraction is a difference of two INDEPENDENTLY-scaled z-scores, not a
   log10-domain ratio -- still an internally consistent RELATIVE reading,
   but not even loosely decibel-equivalent the way the audio/vibration
   case is; an embedding variant reads in that model's own embedding
   units instead (T9-review item 1). Both non-log10 cases are named in the
   figure's own title/colorbar (`_feature_unit_label`) and repeated once
   in the digest, never silently left as an implied log10 claim. This
   module does not exclude fusion here (unlike D2's corrective
   `--level-recal`, `tonal-table` is a descriptive figure, never an offset
   applied to downstream detection) -- the caveat is real and stated once,
   here.
6. `pillar3-figure` (Task 9) -- event-level TPR-vs-alpha grouped bars per
   representation, one panel per pillar-3 session, read straight from
   existing `results/pillar3/<session>/<representation>-a<alpha>/
   event_eval.csv` `row_type == "summary"` rows (`scripts/eval_events.py`'s
   own tidy-CSV contract) -- a leaf whose directory name carries no
   trailing `-a<alpha>` suffix (e.g. a `-frozen` leaf, outside the alpha
   grid) is not part of this comparison and is silently skipped. Our own
   numbers only; no partner attribution (like `rotations-heatmap`, this is
   Stefan's own comparison-readability motivation, not a replicated
   analysis type).

`digest` (Task 9) writes `results/analysis-days/README.md`: one section per
subcommand above, a 2-3 sentence plain-language reading of that chart type, a
markdown link to every PNG this module has actually written so far under
that subcommand's own directory, the A1.8 attribution line for every
partner-inspired analysis type (`feature-stability`, `era-step`,
`mode-signatures`, `tonal-table`), and a dedicated paragraph documenting the
A1.1 finding: `fuse()`'s built-in per-run z-score is an implicit session
normalization that plausibly explains fusion's own cross-day FAR advantage.
English only (`--lang de` not built, spec D3 -- thesis language).

Every subcommand writes a PNG (matplotlib, Agg backend) + its underlying CSV
under `results/analysis-days/<subcommand>/` (digest is the one exception: a
single `README.md`, no CSV); every run is deterministic (seeded bootstrap
where randomness is involved). Colors are the dataviz skill's validated
categorical/sequential steps (`references/palette.md`; duplicated here as
plain hex, mirroring `scripts/analyze_step1.py`'s own precedent comment) --
not re-imported from that sibling script (script-sibling rule: a script
never imports another script's internals; only `src/rowii/` modules are
imported normally).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import; headless-safe backend.

import argparse  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from collections.abc import Mapping, Sequence  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPTS_DIR.parent / "src"
for _extra_path in (str(_SCRIPTS_DIR), str(_SRC_DIR)):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

from rowii.anomaly.levelrecal import level_columns  # noqa: E402
from rowii.config import Config, load_config  # noqa: E402
from rowii.io.dataset import (  # noqa: E402
    RecordingIndex,
    Run,
    betriebsdaten_utc_offset_ns,
    discover,
    run_utc_offset_ns,
)
from rowii.io.gantner import read_header  # noqa: E402
from rowii.pipeline import PreparedRun, prepare_run, stream_columns  # noqa: E402
from rowii.scada.labels import gt_labels, load_scada_window_means  # noqa: E402
from rowii.signals.features import MACHINE_HZ  # noqa: E402
from rowii.signals.windows import WindowGrid  # noqa: E402

logger = logging.getLogger(__name__)

_ANALYSIS_DIR_NAME = "analysis-days"
_STABILITY_CUTOFF_DB = 3.0
"""The A1.8 named, adopted-for-comparability cutoff ("same cutoff as Rodrigues &
Zhang (2026), adopted for comparability"), in dB. Applied to THIS module's own
shift value AFTER converting a level column's own log10-domain shift to dB
(`_level_db_factor` -- x20 for `_log_rms`, x10 for `_band_`/`_octave_`, the
standard amplitude/power dB relationship), never against a partner-reported
number (the firewall's binding test rule); the label is a comparability choice
of CUTOFF, not a claim that our shift equals their reported figure. Shape
columns, and every column under the `fusion`/embedding variants, never receive
this classification at all (`"n/a"` -- see `_feature_stability_table`)."""
_EXCLUDED_GT = ("unknown", "transition")
"""Duplicated from `rowii.state.modebank._EXCLUDED_GT` / `scripts/run_modebank.py`
(spec A1.5, reused-in-spirit here per plan Task 8/9 self-review note): GT windows
excluded from every per-mode computation in this module."""

_MIC_STREAMS: tuple[str, ...] = ("RAWGeneratorMic__0", "RAWTurbineMic__1")
_VIB_STREAMS: tuple[str, ...] = ("RAWGeneratorVib__2", "RAWTurbineVib__3")
_ALL_STREAMS: tuple[str, ...] = _MIC_STREAMS + _VIB_STREAMS
"""The four physical burst streams (VERIFIED hardware stream-name literals,
mirroring `rowii.pipeline`'s own `_AUDIO_STREAMS`/`_VIB_STREAMS` tuples -- plain
data constants, not a private-API duplication)."""

_080726_TOKEN = "080726"
_UNMATCHED_NO_GT_NOTE = "no GT -- era-A anchor by MeasName only"
"""Exact spec wording (A1.2) for a day with zero Betriebsdaten coverage at all
(e.g. 2026-06-27) -- shown as an un-matched per-stream-median point."""

# ---------------------------------------------------------------------------
# Palette (dataviz skill's validated steps, `references/palette.md` -- duplicated
# plain hex, mirroring `scripts/analyze_step1.py`'s own precedent comment: every
# step here passed `node scripts/validate_palette.js` for both the categorical
# 4-slot all-pairs case (scatter/dot charts) and the sequential blue heatmap
# ramp).
# ---------------------------------------------------------------------------
_COLOR_SURFACE = "#fcfcfb"
_COLOR_TEXT_SECONDARY = "#52514e"
_COLOR_TEXT_MUTED = "#898781"
_COLOR_GRIDLINE = "#e1e0d9"
_COLOR_STEM = "#c3c2b7"

_CATEGORICAL_HEX: tuple[str, ...] = (
    "#2a78d6",  # slot 1 blue
    "#008300",  # slot 2 green
    "#e87ba4",  # slot 3 magenta
    "#eda100",  # slot 4 yellow
    "#1baf7a",  # slot 5 aqua
    "#eb6834",  # slot 6 orange
    "#4a3aa7",  # slot 7 violet
    "#e34948",  # slot 8 red
)
"""Fixed categorical hue ORDER (never cycled/reassigned) -- validated
`node scripts/validate_palette.js "<first 4>" --mode light` for the ALL-PAIRS
case this module's scatter/dot-interval charts need (worst adjacent CVD deltaE
16.3, normal-vision deltaE 19.6; magenta/yellow sit below 3:1 surface contrast,
so every use here also carries a direct label -- legend + axis ticks, never
color alone)."""

_SEQUENTIAL_BLUE_HEX: tuple[str, ...] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
"""Sequential blue ramp (steps 100..700, light->dark) for `rotations-heatmap`'s
magnitude encoding (a realized-FAR heatmap) -- the palette's own documented use
case for this exact chart form."""


def _sequential_blue_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("rowii_seq_blue", _SEQUENTIAL_BLUE_HEX)


# ---------------------------------------------------------------------------
# Shared script-sibling-duplicated helpers (script-sibling rule: this script
# must not import scripts/run_step2.py's or scripts/run_modebank.py's
# internals; each docstring below names its mirror).
# ---------------------------------------------------------------------------


def _unknown_run_names(names: list[str], index: RecordingIndex) -> list[str]:
    """Duplicated from `scripts/run_step2.py`/`scripts/run_modebank.py`'s helper
    of the same name: names in *names* with no matching discovered run,
    de-duplicated, in the order first seen."""
    known = {r.name for r in index.runs}
    return list(dict.fromkeys(n for n in names if n not in known))


def _betriebsdaten_for_grid(betriebsdaten: list[Path], grid: WindowGrid) -> list[Path]:
    """Duplicated from `scripts/run_step2.py`'s helper of the same name:
    Betriebsdaten files whose hourly span intersects *grid*'s true-UTC time
    range (shifted onto true UTC by their own derived offset before the
    intersection test)."""
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


def _run_scada_or_none(
    prepared: PreparedRun, run: Run, index: RecordingIndex
) -> pd.DataFrame | None:
    """Duplicated from `scripts/run_step2.py`'s `_load_run_scada`: per-window
    SCADA means for *run*'s own day, or `None` if this run's day has no
    Betriebsdaten coverage overlapping its own grid at all."""
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
# The shared read path: `_RunFeatures` + `_run_features_and_gt`. Every
# subcommand in this module (and, per the plan's own Task 9 note, `mode-
# signatures`/`tonal-table` later) goes through this ONE seam -- CLI tests
# monkeypatch it directly, bypassing `prepare_run`/`load_scada_window_means`/
# `gt_labels` entirely (mirrors how `tests/test_modebank.py` bypasses IO by
# testing `ModeBank` on hand-built arrays).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RunFeatures:
    """One run's VALID-window features + aligned GT state strings + segment ids
    (`rowii.pipeline.prepare_run` + `rowii.scada.labels.gt_labels`, restricted to
    `PreparedRun.valid_mask`). `has_gt` is False when the run's own day has ZERO
    Betriebsdaten coverage at all (e.g. 2026-06-27) -- distinct from a
    GT-bearing day whose windows individually read `"unknown"`/`"transition"`."""

    run_name: str
    features: np.ndarray
    gt_states: np.ndarray
    segment_ids: np.ndarray
    feature_names: list[str]
    has_gt: bool


def _run_features_and_gt(
    run_name: str, variant: str, cfg: Config, index: RecordingIndex, *, use_cache: bool = True
) -> _RunFeatures:
    """Resolve *run_name* against *index*, prepare its *variant* features
    (`rowii.pipeline.prepare_run`), and attach GT state strings
    (`rowii.scada.labels.gt_labels`) -- everything restricted to valid windows.

    Raises:
        KeyError: *run_name* is not a discovered run (callers validate names via
            `_unknown_run_names` first, so this should never fire in practice).
        RuntimeError: `prepare_run` itself raises (too short/sparse for the
            requested variant) -- propagated, callers catch it per-run.
    """
    runs_by_name = {r.name: r for r in index.runs}
    run = runs_by_name[run_name]
    prepared = prepare_run(run, variant, cfg, use_cache=use_cache)
    valid = prepared.valid_mask
    scada = _run_scada_or_none(prepared, run, index)
    if scada is None:
        gt_states = np.full(int(valid.sum()), "unknown", dtype=object)
        has_gt = False
    else:
        full_state = gt_labels(scada, cfg.gt, window_s=cfg.window.window_s)["state"].to_numpy()
        gt_states = full_state[valid]
        has_gt = True
    return _RunFeatures(
        run_name=run_name,
        features=prepared.features[valid],
        gt_states=gt_states,
        segment_ids=prepared.segment_ids[valid],
        feature_names=list(prepared.feature_names),
        has_gt=has_gt,
    )


def _resolve_run_names(runs_arg: str) -> list[str]:
    return [n.strip() for n in runs_arg.split(",") if n.strip()]


def _default_cross_day_pooled_root() -> Path:
    return load_config().results_root / "step2" / "cross-day-pooled"


def _default_pillar3_root() -> Path:
    return load_config().results_root / "pillar3"


def _resolve_out_root(out_arg: str | None, results_root: Path | None) -> Path:
    """`--out` if given, else `<results_root>/analysis-days` -- *results_root*
    is looked up via `load_config()` when `None` (deferred so a caller that
    already has a `Config` on hand, or a caller whose `--out` is always given
    in tests, never triggers an ambient `load_config()` read -- mirrors
    `tests/test_run_modebank.py`'s own "config built by hand, ambient env/.env
    never read" test-hygiene convention). Shared by all three runners below."""
    if out_arg is not None:
        return Path(out_arg)
    root = results_root if results_root is not None else load_config().results_root
    return root / _ANALYSIS_DIR_NAME


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested directly, tests/test_analyze_days.py)
# ---------------------------------------------------------------------------


def _classify_shift(shift_abs: float, cutoff: float = _STABILITY_CUTOFF_DB) -> str:
    """"drifting" if `shift_abs >= cutoff` else "slow" -- A1.8: *cutoff* is a
    NAMED, adopted-for-comparability constant, never asserted against a
    partner-reported number. A pure comparison: the caller decides what
    *shift_abs* means -- `feature-stability` passes a dB-converted level-column
    shift (`_level_db_factor`), never the raw log10 difference directly."""
    return "drifting" if shift_abs >= cutoff else "slow"


_LOG_RMS_DB_FACTOR = 20.0
"""`*_log_rms` stores `log10(RMS AMPLITUDE)` (`rowii.anomaly.levelrecal`'s own
VERIFIED-FACT module docstring) -- dB = 20*log10(amplitude ratio), the standard
amplitude-domain dB relationship."""
_BAND_OCTAVE_DB_FACTOR = 10.0
"""`*_band_*`/`*_octave_*` store `log10(mean Welch PSD)`, a POWER quantity
(`rowii.anomaly.levelrecal`'s own VERIFIED-FACT module docstring) -- dB =
10*log10(power ratio), the standard power-domain dB relationship."""


def _level_db_factor(feature_name: str) -> float:
    """The log10-domain-shift-to-dB multiplier for ONE level column, keyed by
    its own name token: `_LOG_RMS_DB_FACTOR` (20) for `*_log_rms` (stored log10
    RMS amplitude), `_BAND_OCTAVE_DB_FACTOR` (10) for `*_band_*`/`*_octave_*`
    (stored log10 mean Welch PSD power) -- the BLOCKER dB-unit-coherence fix
    (A1.8): the 3 dB comparability cutoff is meaningless applied to a raw log10
    shift without this conversion first.

    Raises:
        ValueError: *feature_name* carries neither token -- not a level column
            at all (callers restrict to `rowii.anomaly.levelrecal.
            level_columns` first; this is a caller-bug guard, not a real data
            case).
    """
    if "_log_rms" in feature_name:
        return _LOG_RMS_DB_FACTOR
    if "_band_" in feature_name or "_octave_" in feature_name:
        return _BAND_OCTAVE_DB_FACTOR
    raise ValueError(
        f"_level_db_factor: {feature_name!r} carries neither a _log_rms nor a "
        f"_band_/_octave_ token -- not a level column"
    )


def _block_bootstrap_ci(
    values: np.ndarray, segment_ids: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:
    """95% percentile bootstrap CI on `median(values)`, resampling whole
    `segment_ids` BLOCKS with replacement (A1.11: the 12-min recording segment,
    NEVER wall-clock/calendar time) -- a degenerate SINGLE-segment input still
    returns a FINITE interval, but a zero-WIDTH (degenerate) one: with only one
    block to draw from, every bootstrap replicate resamples that exact same
    block (there is no second block it could ever pick instead), so
    `median(values)` is identical on every draw regardless of how many DISTINCT
    values the lone block itself holds -- resampling blocks, not the individual
    values inside them, is what block bootstrap means. `n_segments < 2` is
    therefore a degenerate CI by construction (`lo == hi`); the figure shows a
    whisker-less dot for that (feature, mode, day)."""
    rng = np.random.default_rng(seed)
    seg_ids = np.unique(segment_ids)
    groups = [values[segment_ids == s] for s in seg_ids]
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        boots[b] = float(np.median(np.concatenate([groups[i] for i in pick])))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _flag_rate_matrix(far: Mapping[tuple[str, str], float]) -> pd.DataFrame:
    """`{(fit_pool_key, test_run): pooled_realized_far}` -> a (fit pool x test
    run) `DataFrame`, `NaN` where a rotation was never discovered."""
    rows = sorted({k[0] for k in far})
    cols = sorted({k[1] for k in far})
    m = pd.DataFrame(np.nan, index=rows, columns=cols)
    for (fit, test), v in far.items():
        m.loc[fit, test] = v
    return m


def _era_step_row(
    levels_by_stream: Mapping[str, np.ndarray], gt_mode_mask: np.ndarray
) -> dict[str, float]:
    """Per-stream median level over the *gt_mode_mask*-selected windows of one
    run/day -- one call produces one era-step row: `{stream: median_level}`.

    Raises:
        ValueError: *gt_mode_mask* selects zero windows -- the caller decides
            what to do about an empty match (era-step's own runner falls back
            to the all-valid-windows "unmatched" convention instead of ever
            calling this with an empty mask).
    """
    mask = np.asarray(gt_mode_mask, dtype=bool)
    if not bool(mask.any()):
        raise ValueError("_era_step_row: gt_mode_mask selects zero windows")
    return {
        stream: float(np.median(np.asarray(levels, dtype=np.float64)[mask]))
        for stream, levels in levels_by_stream.items()
    }


def _tonal_contrast(band_energy: float, octave_floor: float) -> float:
    """`band_energy - octave_floor` -- OUR OWN SNR-like contrast (Task 9,
    spec D3.5): how far a machine-tone band's own energy sits above
    (positive) or below (negative) its neighboring octave column's energy,
    in whatever units the two inputs already share (own stored log10 units
    for audio/vibration; per-run z-score units for fusion -- see module
    docstring). Explicitly NOT the partner's exact tonal metric (A1.8) --
    only the ANALYSIS TYPE (contrasting a narrowband tone against a nearby
    broadband floor) is shared."""
    return band_energy - octave_floor


def _feature_unit_label(variant: str, feature_names: Sequence[str]) -> str:
    """The units phrase for *variant*'s own stored feature values (T9-review
    item 1, LOW-MED interpretation honesty): shared by `mode-signatures`/
    `tonal-table` (their title/axis/colorbar text) and the digest's matching
    caveat sentences, so one variant reads identically everywhere this
    module talks about its units.

    `fusion` is checked FIRST, by *variant* NAME rather than column content:
    `fuse()` z-scores every stream per run before concatenating
    (`rowii.signals.features.fuse`, A1.1), but its level-named columns keep
    their `_band_`/`_octave_` name tokens -- a content-only check would
    misclassify them as genuine log10 columns. An embedding variant
    (`rowii.anomaly.levelrecal.level_columns` returns the empty set for
    *feature_names*, e.g. `audio-beats`/`audio-tfc`/`logmel`) is in that
    model's own embedding units, also never log10. Every other (raw-scale)
    variant keeps the genuine log10 reading."""
    if variant == "fusion":
        return (
            "per-run z-score (dimensionless) -- fuse()'s own per-run "
            "standardization, A1.1"
        )
    if not level_columns(list(feature_names)):
        return "embedding units (model-specific)"
    return "log10, own stored units -- not a calibrated acoustic-dB conversion"


def _mode_profile(
    features: np.ndarray, gt_states: np.ndarray, level_cols: Sequence[int]
) -> pd.DataFrame:
    """Tidy (mode, column, median, q25, q75, n_windows) table (Task 9, spec
    D3.4): per GT mode present in *gt_states* (`unknown`/`transition`
    excluded, `_EXCLUDED_GT`), per column INDEX in *level_cols*, that mode's
    own median + interquartile range over *features*' matching rows -- the
    "modes are separable" picture's underlying numbers. *column* is a bare
    INTEGER index, not a name: this helper is feature-name-agnostic
    (mirrors `rowii.anomaly.levelrecal.level_columns`'s own bare-index
    contract) -- callers with `feature_names` in scope join it back for
    display. A mode with zero matching rows contributes no row at all
    (nothing to summarise)."""
    x = np.asarray(features, dtype=np.float64)
    gt = np.asarray(gt_states, dtype=object)
    cols = list(level_cols)
    modes: set[str] = set(gt.tolist())
    modes -= set(_EXCLUDED_GT)

    rows: list[dict[str, object]] = []
    for mode in sorted(modes):
        mask = gt == mode
        n = int(mask.sum())
        if n == 0:
            continue
        sub = x[mask][:, cols]
        median = np.median(sub, axis=0)
        q25 = np.percentile(sub, 25, axis=0)
        q75 = np.percentile(sub, 75, axis=0)
        for k, col in enumerate(cols):
            rows.append(
                {
                    "mode": mode,
                    "column": int(col),
                    "median": float(median[k]),
                    "q25": float(q25[k]),
                    "q75": float(q75[k]),
                    "n_windows": n,
                }
            )
    columns = ["mode", "column", "median", "q25", "q75", "n_windows"]
    return pd.DataFrame(rows, columns=columns)


def _tpr_by_alpha(event_table: pd.DataFrame) -> pd.DataFrame:
    """(representation x alpha) `event_tpr` pivot from a tidy *event_table*
    (Task 9, spec D3.6) -- columns `representation`/`alpha`/`event_tpr`, one
    row per discovered pillar-3 leaf (mirrors `_flag_rate_matrix`'s own
    plain-pivot contract; callers filter to one `session` first). `NaN`
    where a (representation, alpha) combination was never discovered."""
    reps = sorted(event_table["representation"].unique())
    alphas = sorted(event_table["alpha"].unique())
    matrix = pd.DataFrame(np.nan, index=reps, columns=alphas)
    for _, row in event_table.iterrows():
        matrix.loc[row["representation"], row["alpha"]] = row["event_tpr"]
    return matrix


# ---------------------------------------------------------------------------
# Subcommand 1: rotations-heatmap
# ---------------------------------------------------------------------------

_FIT_POOL_RE = re.compile(r"^-\s*fit pool:\s*(.+?)\s*\(pool order", re.MULTILINE)


def _parse_fit_pool(notes_text: str) -> str:
    """The `"- fit pool: <names> (pool order ..."` line's *names* portion
    (`scripts/run_step2.py`'s `_cross_day_pooled_notes`'s own literal wording),
    verbatim (comma-joined names, whatever order that file recorded)."""
    match = _FIT_POOL_RE.search(notes_text)
    if match is None:
        raise ValueError("notes.md carries no '- fit pool: ... (pool order' line")
    return match.group(1).strip()


def _far_table_pooled_value(path: Path) -> float:
    """The `label == "pooled"` aggregate row's `realized_far` from a
    `far_table_<mode>.csv` (`rowii.anomaly.sweep`'s own aggregate-row
    convention, reused/duplicated by `scripts/run_step2.py`'s
    cross-day-pooled `_FAR_TABLE_COLUMNS` schema)."""
    table = pd.read_csv(path, dtype={"label": str})
    pooled = table[table["label"] == "pooled"]
    if pooled.empty:
        raise ValueError(f"{path}: no 'pooled' aggregate row")
    return float(pooled.iloc[0]["realized_far"])


def _discover_rotation_leaves(
    root: Path, variant: str, mode: str, leaf_suffix: str = ""
) -> dict[tuple[str, str], float]:
    """Walk `root/<test_run>/<variant>-pooled<leaf_suffix>/far_table_<mode>.csv`
    (`scripts/run_step2.py`'s `_cross_day_pooled_out_dir` leaf convention --
    plain `<variant>-pooled` when *leaf_suffix* is `""`, the default; a
    `--leaf-suffix` such as `-a0.05` instead discovers `<variant>-pooled-a0.05`
    leaves), pairing each discovered leaf with its own `notes.md` "fit pool:"
    line -> `{(fit_pool, test_run): pooled_far}`. A leaf missing either file is
    silently skipped (not every `<test_run>` subdirectory necessarily has this
    *variant*/*leaf_suffix* combination)."""
    far: dict[tuple[str, str], float] = {}
    if not root.is_dir():
        return far
    for test_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        leaf = test_dir / f"{variant}-pooled{leaf_suffix}"
        far_path = leaf / f"far_table_{mode}.csv"
        notes_path = leaf / "notes.md"
        if not far_path.is_file() or not notes_path.is_file():
            continue
        try:
            fit_pool = _parse_fit_pool(notes_path.read_text())
            pooled_far = _far_table_pooled_value(far_path)
        except ValueError as exc:
            logger.warning("rotations-heatmap: skipping %s (%s)", leaf, exc)
            continue
        far[(fit_pool, test_dir.name)] = pooled_far
    return far


def _plot_flag_rate_heatmap(matrix: pd.DataFrame, mode: str, out_path: Path) -> None:
    values = matrix.to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    vmax = float(finite.max()) if finite.size else 1.0

    # The row labels are fit-pool strings (potentially several comma-joined run
    # names) and the title is a fixed-length sentence -- both routinely wider
    # than a narrow (e.g. 3x3) matrix's own column count would otherwise size
    # the figure to, so the width floor is generous rather than column-driven.
    width = max(7.0, 0.9 * matrix.shape[1] + 3.0)
    height = max(3.0, 0.5 * matrix.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(_COLOR_SURFACE)
    ax.set_facecolor(_COLOR_SURFACE)

    im = ax.imshow(
        values, cmap=_sequential_blue_cmap(), vmin=0.0, vmax=max(vmax, 1e-9), aspect="auto"
    )
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(
        matrix.columns, rotation=45, ha="right", fontsize=8, color=_COLOR_TEXT_SECONDARY
    )
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8, color=_COLOR_TEXT_SECONDARY)
    ax.set_xlabel("held-out test run", color=_COLOR_TEXT_MUTED, fontsize=9)
    ax.set_ylabel("fit pool", color=_COLOR_TEXT_MUTED, fontsize=9)
    ax.set_title(
        f"Cross-day pooled flag rate -- {mode} thresholds", loc="left", fontsize=11, color="#0b0b0b"
    )

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                continue
            text_color = "#ffffff" if vmax > 0 and v > 0.6 * vmax else _COLOR_TEXT_SECONDARY
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5, color=text_color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("realized FAR (pooled aggregate row)", color=_COLOR_TEXT_MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _run_rotations_heatmap(args: argparse.Namespace) -> int:
    variant = str(args.variant)
    mode = str(args.mode)
    leaf_suffix = str(args.leaf_suffix)
    root = Path(args.root) if args.root is not None else _default_cross_day_pooled_root()

    far = _discover_rotation_leaves(root, variant, mode, leaf_suffix)
    if len(far) < 2:
        found = sorted(far)
        print(
            f"rotations-heatmap: only {len(far)} rotation(s) discovered under "
            f"{root} for variant {variant!r}, leaf "
            f"'<test_run>/{variant}-pooled{leaf_suffix}/far_table_{mode}.csv' "
            f"(need >= 2 to render a heatmap -- never rendering a 1-cell "
            f"'matrix' silently); found: {found}. If the leaves you expect "
            f"carry a different suffix (e.g. '-a0.05'), pass --leaf-suffix.",
            file=sys.stderr,
        )
        return 2

    matrix = _flag_rate_matrix(far)
    out_root = _resolve_out_root(args.out, None)
    out_dir = out_root / "rotations-heatmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{variant}-{mode}"
    matrix.to_csv(out_dir / f"{stem}.csv")
    _plot_flag_rate_heatmap(matrix, mode, out_dir / f"{stem}.png")
    print(
        f"rotations-heatmap: wrote {out_dir / (stem + '.csv')} and .png "
        f"({len(far)} rotation(s))"
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand 2: feature-stability
# ---------------------------------------------------------------------------

_STABILITY_N_BOOT_DEFAULT = 1000
_STABILITY_SEED_DEFAULT = 7
_STABILITY_MIN_MODE_WINDOWS_DEFAULT = 5
_STABILITY_TOP_N_DEFAULT = 20


def _feature_stability_table(
    runs: Sequence[_RunFeatures],
    *,
    n_boot: int,
    seed: int,
    cutoff: float,
    min_mode_windows: int,
    variant: str,
) -> pd.DataFrame:
    """Per (feature, mode, day) tidy table: that day's block-bootstrapped
    median + CI for *feature* restricted to GT mode *mode*, plus the
    (feature, mode)-level `shift_abs` (max day-median - min day-median across
    the qualifying days, OWN STORED UNITS -- log10 for level columns, raw for
    shape columns) and a `shift_db` column (the BLOCKER dB-unit-coherence fix,
    A1.8): for a LEVEL column (`rowii.anomaly.levelrecal.level_columns`) of a
    genuine raw-scale *variant*, `shift_abs` converted to a real dB figure via
    `_level_db_factor` (x20 `_log_rms` / x10 `_band_`/`_octave_`) -- THIS is
    what `_classify_shift` compares against *cutoff*, never the raw `shift_abs`
    directly. Repeated across that group's day-rows (tidy format, directly
    plottable/filterable). Sorted by `shift_abs` descending (worst offenders
    first, own-units magnitude -- unaffected by the dB fix, which only touches
    the classification/`shift_db`, not the ranking).

    `classification` is `"n/a"` (never "slow"/"drifting", `shift_db` is `NaN`)
    for:
    - every SHAPE column (own raw units, e.g. Hz or a dimensionless moment --
      a dB cutoff is meaningless there) in any variant, and
    - EVERY column, level-named or not, when *variant* is `"fusion"` (its
      level-named columns are `fuse()`'s own per-run z-scores, A1.1 -- not
      log10 values, so a dB conversion would be meaningless) or an embedding
      variant (`level_columns` returns the empty set, e.g.
      `audio-beats`/`audio-tfc`/`logmel` -- embedding units, not log10).
    Either fusion/embedding skip case logs one `logger.warning` naming the
    reason; the dot-interval rows themselves are still written in both cases
    (own-units median + CI stay meaningful and clearly axis-labeled even
    without a slow/drifting call).

    Only GT-bearing runs (`has_gt=True`) participate (A1.2); a mode needs >= 2
    qualifying days (each with >= *min_mode_windows* of that mode) to produce a
    shift at all -- modes/features that never reach that floor on any day pair
    are silently absent from the table (nothing to measure a shift from).

    Raises:
        ValueError: fewer than 2 GT-bearing runs, or the included runs disagree
            on `feature_names` (refusing to compare mismatched columns).
    """
    included = [r for r in runs if r.has_gt]
    if len(included) < 2:
        raise ValueError(
            f"feature-stability needs >= 2 GT-bearing run(s) to measure a "
            f"cross-day shift; got {len(included)} (A1.2: GT-bearing days only)"
        )
    feature_names = included[0].feature_names
    for r in included[1:]:
        if r.feature_names != feature_names:
            raise ValueError(
                f"feature-stability: run {r.run_name!r} feature_names disagree "
                f"with {included[0].run_name!r} -- refusing to compare "
                f"mismatched columns"
            )

    level_col_set = set(level_columns(feature_names))
    skip_reason: str | None = None
    if variant == "fusion":
        skip_reason = (
            "variant='fusion': fuse()'s per-run z-score (A1.1) makes every "
            "level-named column a z-score, not a log10 value -- dB "
            "classification skipped, every row reads 'n/a'"
        )
    elif not level_col_set:
        skip_reason = (
            f"variant={variant!r} has zero level column(s) (embedding "
            f"variant) -- shifts are embedding units, not log10 -- dB "
            f"classification skipped, every row reads 'n/a'"
        )
    if skip_reason is not None:
        logger.warning("feature-stability: %s", skip_reason)

    modes: set[str] = set()
    for r in included:
        modes |= set(np.unique(r.gt_states).tolist())
    modes -= set(_EXCLUDED_GT)

    rows: list[dict[str, object]] = []
    for mode in sorted(modes):
        qualifying = [(r, np.flatnonzero(r.gt_states == mode)) for r in included]
        qualifying = [(r, idx) for r, idx in qualifying if idx.size >= min_mode_windows]
        if len(qualifying) < 2:
            continue
        for j, feature in enumerate(feature_names):
            day_median: dict[str, float] = {}
            day_ci: dict[str, tuple[float, float]] = {}
            day_n: dict[str, int] = {}
            for r, idx in qualifying:
                values = r.features[idx, j]
                seg = r.segment_ids[idx]
                lo, hi = _block_bootstrap_ci(values, seg, n_boot, seed)
                day_median[r.run_name] = float(np.median(values))
                day_ci[r.run_name] = (lo, hi)
                day_n[r.run_name] = int(idx.size)
            shift_abs = float(max(day_median.values()) - min(day_median.values()))
            if skip_reason is None and j in level_col_set:
                shift_db: float = shift_abs * _level_db_factor(feature)
                classification = _classify_shift(abs(shift_db), cutoff)
            else:
                shift_db = float("nan")
                classification = "n/a"
            for run_name, median in day_median.items():
                lo, hi = day_ci[run_name]
                rows.append(
                    {
                        "feature": feature,
                        "mode": mode,
                        "day": run_name,
                        "n_windows": day_n[run_name],
                        "median": median,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "shift_abs": shift_abs,
                        "shift_db": shift_db,
                        "classification": classification,
                    }
                )

    columns = [
        "feature", "mode", "day", "n_windows", "median", "ci_lo", "ci_hi",
        "shift_abs", "shift_db", "classification",
    ]
    table = pd.DataFrame(rows, columns=columns)
    if not table.empty:
        table = table.sort_values(
            ["shift_abs", "feature", "mode", "day"], ascending=[False, True, True, True]
        ).reset_index(drop=True)
    return table


def _plot_feature_stability(
    table: pd.DataFrame, out_path: Path, top_n: int = _STABILITY_TOP_N_DEFAULT
) -> None:
    """Dot-interval (errorbar) figure -- one dot per (feature, mode, day),
    grouped into rows by (feature, mode), restricted to the top *top_n* groups
    by `shift_abs` (legibility; the full table is the CSV's job). PRIMARY
    deliverable (A1.8): the continuous per-day median + CI, not the binary
    classification, which lives only in the CSV's `classification` column."""
    if table.empty:
        fig, ax = plt.subplots(figsize=(8.5, 3.0))
        fig.patch.set_facecolor(_COLOR_SURFACE)
        ax.set_facecolor(_COLOR_SURFACE)
        ax.text(
            0.5, 0.5, "no (feature, mode) pair had >= 2 qualifying GT-bearing days",
            ha="center", va="center", color=_COLOR_TEXT_SECONDARY, fontsize=10,
        )
        ax.axis("off")
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return

    groups = (
        table[["feature", "mode", "shift_abs"]]
        .drop_duplicates()
        .sort_values("shift_abs", ascending=False)
        .head(top_n)
    )
    group_keys = list(zip(groups["feature"], groups["mode"], strict=True))
    days = sorted(table["day"].unique())
    day_color = {d: _CATEGORICAL_HEX[i % len(_CATEGORICAL_HEX)] for i, d in enumerate(days)}

    height = max(3.0, 0.42 * len(group_keys) + 1.6)
    fig, ax = plt.subplots(figsize=(8.5, height))
    fig.patch.set_facecolor(_COLOR_SURFACE)
    ax.set_facecolor(_COLOR_SURFACE)

    labels = [f"{feature} / {mode}" for feature, mode in group_keys]
    for row_i, (feature, mode) in enumerate(group_keys):
        sub = table[(table["feature"] == feature) & (table["mode"] == mode)]
        for _, r in sub.iterrows():
            color = day_color[str(r["day"])]
            median = float(r["median"])
            lo_err = max(0.0, median - float(r["ci_lo"]))
            hi_err = max(0.0, float(r["ci_hi"]) - median)
            ax.errorbar(
                median, row_i, xerr=[[lo_err], [hi_err]], fmt="o", color=color,
                ecolor=color, elinewidth=1.5, capsize=3, markersize=5, zorder=3,
            )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5, color=_COLOR_TEXT_SECONDARY)
    ax.invert_yaxis()
    ax.set_xlabel(
        "feature value (own stored units: log10 for level columns, raw for "
        "shape columns -- not a calibrated acoustic-dB conversion)",
        color=_COLOR_TEXT_MUTED, fontsize=8,
    )
    ax.set_title(
        f"Cross-day feature stability -- top {len(group_keys)} by shift "
        "(dot = day median, whisker = segment-block bootstrap CI)",
        loc="left", fontsize=10, color="#0b0b0b",
    )
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=day_color[d], label=d)
        for d in days
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(days), 4), frameon=False, fontsize=8,
    )
    ax.grid(axis="x", color=_COLOR_GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_COLOR_STEM)
    ax.tick_params(axis="both", colors=_COLOR_TEXT_MUTED, length=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _run_feature_stability(args: argparse.Namespace) -> int:
    run_names = _resolve_run_names(str(args.runs))
    if not run_names:
        print("feature-stability: --runs got an empty run-name list", file=sys.stderr)
        return 2
    variant = str(args.variant)

    cfg = load_config()
    index = discover(cfg.data_root)
    unknown = _unknown_run_names(run_names, index)
    if unknown:
        available = ", ".join(sorted({r.name for r in index.runs})) or "(none discovered)"
        print(
            f"feature-stability: unknown run name(s): {', '.join(unknown)}; "
            f"available runs: {available}",
            file=sys.stderr,
        )
        return 2

    runs_features: list[_RunFeatures] = []
    for name in run_names:
        try:
            rf = _run_features_and_gt(name, variant, cfg, index, use_cache=not args.no_cache)
        except RuntimeError as exc:
            print(f"feature-stability: prepare_run failed for {name!r} ({exc})", file=sys.stderr)
            return 2
        if not rf.has_gt:
            logger.warning(
                "feature-stability: run %r has no GT (Betriebsdaten) coverage at "
                "all -- excluded (A1.2: GT-bearing days only)", name,
            )
            continue
        runs_features.append(rf)

    try:
        table = _feature_stability_table(
            runs_features,
            n_boot=int(args.n_boot),
            seed=int(args.seed),
            cutoff=float(args.cutoff_db),
            min_mode_windows=int(args.min_mode_windows),
            variant=variant,
        )
    except ValueError as exc:
        print(f"feature-stability: {exc}", file=sys.stderr)
        return 2

    out_root = _resolve_out_root(args.out, cfg.results_root)
    out_dir = out_root / "feature-stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = variant
    table.to_csv(out_dir / f"{stem}.csv", index=False)
    _plot_feature_stability(table, out_dir / f"{stem}.png")
    print(
        f"feature-stability: wrote {out_dir / (stem + '.csv')} and .png "
        f"({len(runs_features)} GT-bearing day(s), {len(table)} (feature,mode,day) row(s))"
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand 3: era-step
# ---------------------------------------------------------------------------


def _is_080726(run_name: str) -> bool:
    return _080726_TOKEN in run_name


def _stream_level_columns(feature_names: list[str], stream: str) -> np.ndarray | None:
    """Column indices of *stream*'s own LEVEL columns (`rowii.anomaly.
    levelrecal.level_columns` intersected with `rowii.pipeline.stream_columns`),
    or `None` if *stream* is absent from *feature_names* (a variant that does
    not include this stream) or contributes zero level columns."""
    try:
        stream_cols = stream_columns(feature_names, stream)
    except ValueError:
        return None
    level_cols = set(level_columns(feature_names))
    cols = np.array([c for c in stream_cols if c in level_cols], dtype=np.int64)
    return cols if cols.size else None


def _levels_by_stream(features: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray]:
    """`{stream: (W,) level series}` for every one of `_ALL_STREAMS` present in
    *feature_names* with >= 1 level column -- the level series is that
    stream's own level columns averaged per window (one scalar per window)."""
    out: dict[str, np.ndarray] = {}
    for stream in _ALL_STREAMS:
        cols = _stream_level_columns(feature_names, stream)
        if cols is None:
            continue
        out[stream] = features[:, cols].mean(axis=1)
    return out


def _era_step_table(runs_features: Sequence[_RunFeatures], gt_mode: str) -> pd.DataFrame:
    """Tidy `(run, matched, note, stream, median_level, n_windows)` table, one
    row per (run, stream): a GT-bearing run with >= 1 window of *gt_mode*
    contributes a MATCHED row (that mode's own windows); a GT-bearing run with
    zero windows of *gt_mode*, or a run with no GT at all (`has_gt=False`,
    e.g. 2026-06-27), contributes an UNMATCHED row computed over ALL of that
    run's valid windows instead, flagged with a `note` explaining why (A1.2)."""
    rows: list[dict[str, object]] = []
    for rf in runs_features:
        levels_by_stream = _levels_by_stream(rf.features, rf.feature_names)
        if not levels_by_stream:
            logger.warning(
                "era-step: run %r has no mic/vib level column for any known "
                "stream under this variant -- skipped", rf.run_name,
            )
            continue
        matched: bool
        note: str
        if rf.has_gt:
            mask = rf.gt_states == gt_mode
            if not bool(mask.any()):
                logger.warning(
                    "era-step: run %r has GT but zero %r windows -- falling "
                    "back to all valid windows, flagged unmatched",
                    rf.run_name, gt_mode,
                )
                mask = np.ones(rf.features.shape[0], dtype=bool)
                matched, note = False, f"no {gt_mode!r} windows this day -- using all valid windows"
            else:
                matched, note = True, ""
        else:
            mask = np.ones(rf.features.shape[0], dtype=bool)
            matched, note = False, _UNMATCHED_NO_GT_NOTE
        row = _era_step_row(levels_by_stream, mask)
        for stream, median_level in row.items():
            rows.append(
                {
                    "run": rf.run_name,
                    "matched": matched,
                    "note": note,
                    "stream": stream,
                    "median_level": median_level,
                    "n_windows": int(mask.sum()),
                }
            )
    columns = ["run", "matched", "note", "stream", "median_level", "n_windows"]
    return pd.DataFrame(rows, columns=columns)


def _plot_era_step(
    table: pd.DataFrame,
    out_path: Path,
    run_order: Sequence[str],
    era_boundary_after: str | None,
    variant: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    fig.patch.set_facecolor(_COLOR_SURFACE)
    ax.set_facecolor(_COLOR_SURFACE)

    if table.empty:
        ax.text(
            0.5, 0.5, "no per-stream level data available", ha="center", va="center",
            color=_COLOR_TEXT_SECONDARY, fontsize=10,
        )
        ax.axis("off")
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return

    present_runs = [r for r in run_order if r in set(table["run"])]
    x_by_run = {r: i for i, r in enumerate(present_runs)}
    streams = sorted(table["stream"].unique())
    stream_color = {s: _CATEGORICAL_HEX[i % len(_CATEGORICAL_HEX)] for i, s in enumerate(streams)}

    for stream in streams:
        sub = table[table["stream"] == stream].copy()
        sub["x"] = sub["run"].map(x_by_run)
        sub = sub.sort_values("x")
        matched_rows = sub[sub["matched"]]
        unmatched_rows = sub[~sub["matched"]]
        color = stream_color[stream]
        ax.plot(sub["x"], sub["median_level"], color=color, linewidth=1.5, zorder=2)
        ax.scatter(
            matched_rows["x"], matched_rows["median_level"], color=color, s=45, zorder=3,
            label=stream,
        )
        ax.scatter(
            unmatched_rows["x"], unmatched_rows["median_level"], facecolors="none",
            edgecolors=color, marker="D", s=55, linewidths=1.5, zorder=3,
        )

    unmatched = table[~table["matched"]]
    for run_name, note in unmatched[["run", "note"]].drop_duplicates("run").itertuples(index=False):
        x = x_by_run.get(str(run_name))
        if x is None:
            continue
        top_y = float(unmatched.loc[unmatched["run"] == run_name, "median_level"].max())
        ax.annotate(
            str(note), (x, top_y), xytext=(0, 10), textcoords="offset points",
            ha="center", va="bottom", fontsize=7, color=_COLOR_TEXT_MUTED,
        )

    if era_boundary_after is not None and era_boundary_after in x_by_run:
        ax.axvline(
            x_by_run[era_boundary_after] + 0.5, color=_COLOR_STEM, linestyle="--", linewidth=1.2,
            zorder=1,
        )

    ax.set_xticks(range(len(present_runs)))
    ax.set_xticklabels(
        present_runs, rotation=30, ha="right", fontsize=8, color=_COLOR_TEXT_SECONDARY
    )
    ax.set_ylabel(
        f"median level (log10 units, {variant} -- raw-scale variant)",
        color=_COLOR_TEXT_MUTED, fontsize=9,
    )
    ax.set_title(
        "Per-day per-stream level (filled = GT-matched, hollow diamond = unmatched)",
        loc="left", fontsize=10.5, color="#0b0b0b",
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=min(len(streams), 4),
        frameon=False, fontsize=8,
    )
    ax.grid(axis="y", color=_COLOR_GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_COLOR_STEM)
    ax.spines["left"].set_color(_COLOR_STEM)
    ax.tick_params(axis="both", colors=_COLOR_TEXT_MUTED, length=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _run_era_step(args: argparse.Namespace) -> int:
    """era-step subcommand entrypoint. `--variant` is restricted to `audio`/
    `vibration` (exit 2 otherwise) -- `fusion` is refused: `fuse()`'s per-run
    z-score (A1.1) has no meaningful raw-scale level to plot here. ONE
    invocation therefore plots ONLY that variant's own stream(s) (the mic pair
    for `audio`, the vibration pair for `vibration`); the mic-vs-vibration
    COMPOSITE view the module docstring describes needs one run of EACH
    variant (`--variant audio` and `--variant vibration`, each its own
    `results/analysis-days/era-step/<variant>.{csv,png}`) -- this subcommand
    never overlays them into one figure itself."""
    run_names = _resolve_run_names(str(args.runs))
    if not run_names:
        print("era-step: --runs got an empty run-name list", file=sys.stderr)
        return 2
    variant = str(args.variant)
    if variant not in ("audio", "vibration"):
        print(
            f"era-step: --variant must be 'audio' or 'vibration' (got "
            f"{variant!r}) -- fusion's per-run z-scored features (fuse(), "
            f"A1.1) have no meaningful raw-scale level to plot; run era-step "
            f"once per stream variant (audio, then vibration) for the "
            f"mic-vs-vibration composite view",
            file=sys.stderr,
        )
        return 2
    gt_mode = str(args.gt_mode)
    include_080726 = bool(args.include_080726)

    excluded_080726 = [n for n in run_names if _is_080726(n) and not include_080726]
    if excluded_080726:
        logger.warning(
            "era-step: excluding 080726 run(s) %s -- pass --include-080726 "
            "(only after the D4.3 SCADA-timebase gate, A1.2) to include them",
            ", ".join(excluded_080726),
        )
    kept_run_names = [n for n in run_names if n not in excluded_080726]
    if not kept_run_names:
        print("era-step: every requested run was excluded (see warnings above)", file=sys.stderr)
        return 1

    cfg = load_config()
    index = discover(cfg.data_root)
    unknown = _unknown_run_names(kept_run_names, index)
    if unknown:
        available = ", ".join(sorted({r.name for r in index.runs})) or "(none discovered)"
        print(
            f"era-step: unknown run name(s): {', '.join(unknown)}; available "
            f"runs: {available}",
            file=sys.stderr,
        )
        return 2

    runs_features: list[_RunFeatures] = []
    for name in kept_run_names:
        try:
            rf = _run_features_and_gt(name, variant, cfg, index, use_cache=not args.no_cache)
        except RuntimeError as exc:
            print(f"era-step: prepare_run failed for {name!r} ({exc})", file=sys.stderr)
            return 2
        runs_features.append(rf)

    table = _era_step_table(runs_features, gt_mode)
    if table.empty:
        print("era-step: no per-stream level data could be computed for any run", file=sys.stderr)
        return 1

    out_root = _resolve_out_root(args.out, cfg.results_root)
    out_dir = out_root / "era-step"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = variant
    table.to_csv(out_dir / f"{stem}.csv", index=False)
    _plot_era_step(
        table, out_dir / f"{stem}.png", kept_run_names, args.era_boundary_after, variant
    )
    print(
        f"era-step: wrote {out_dir / (stem + '.csv')} and .png "
        f"({len(kept_run_names)} day(s), gt_mode={gt_mode!r})"
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand 4: mode-signatures (Task 9, spec D3.4)
# ---------------------------------------------------------------------------

_MODE_SIGNATURES_TOP_N_DEFAULT = 20


def _band_octave_columns(feature_names: list[str]) -> list[int]:
    """Indices carrying a `_band_` or `_octave_` name token -- the spectral
    "profile" `mode-signatures` compares across GT modes. Narrower than
    `rowii.anomaly.levelrecal.level_columns` (which also folds in the
    single-scalar `_log_rms` loudness column, not part of a band/octave
    profile shape)."""
    return [i for i, n in enumerate(feature_names) if "_band_" in n or "_octave_" in n]


def _plot_mode_signatures(
    table: pd.DataFrame,
    out_path: Path,
    variant: str,
    feature_names: Sequence[str],
    top_n: int = _MODE_SIGNATURES_TOP_N_DEFAULT,
) -> Axes:
    """Dot-interval (errorbar) figure, ONE day's own `_mode_profile` table --
    one row per (feature, mode), grouped by feature, colored by mode,
    restricted to the top *top_n* features by cross-mode separation
    (`max(median) - min(median)` over that feature's own mode rows;
    legibility, mirrors `_plot_feature_stability`'s `top_n` cap). Whiskers
    are the mode's own interquartile range (asymmetric q25/q75 bounds
    around the median). T9-review item 1: *variant* names the figure's own
    title, and the x-axis units follow `_feature_unit_label(variant,
    feature_names)` -- never a blanket "log10" claim for `fusion`/an
    embedding variant. Returns the rendered `Axes` (test seam)."""
    if table.empty:
        fig, ax = plt.subplots(figsize=(8.5, 3.0))
        fig.patch.set_facecolor(_COLOR_SURFACE)
        ax.set_facecolor(_COLOR_SURFACE)
        ax.text(
            0.5, 0.5, "no GT mode (excl. unknown/transition) had >= 1 window this day",
            ha="center", va="center", color=_COLOR_TEXT_SECONDARY, fontsize=10,
        )
        ax.set_title(f"Mode signatures ({variant})", loc="left", fontsize=10, color="#0b0b0b")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return ax

    separation = (
        table.groupby("feature")["median"]
        .agg(lambda s: float(s.max() - s.min()))
        .sort_values(ascending=False)
    )
    top_features = list(separation.head(top_n).index)
    modes = sorted(table["mode"].unique())
    mode_color = {m: _CATEGORICAL_HEX[i % len(_CATEGORICAL_HEX)] for i, m in enumerate(modes)}
    n_modes = max(len(modes), 1)

    height = max(3.0, 0.42 * len(top_features) + 1.6)
    fig, ax = plt.subplots(figsize=(8.5, height))
    fig.patch.set_facecolor(_COLOR_SURFACE)
    ax.set_facecolor(_COLOR_SURFACE)

    row_span = 0.6
    for row_i, feature in enumerate(top_features):
        sub = table[table["feature"] == feature]
        for k, mode in enumerate(modes):
            r = sub[sub["mode"] == mode]
            if r.empty:
                continue
            r0 = r.iloc[0]
            color = mode_color[mode]
            median = float(r0["median"])
            lo_err = max(0.0, median - float(r0["q25"]))
            hi_err = max(0.0, float(r0["q75"]) - median)
            y = row_i + (k - (n_modes - 1) / 2) * (row_span / n_modes)
            ax.errorbar(
                median, y, xerr=[[lo_err], [hi_err]], fmt="o", color=color,
                ecolor=color, elinewidth=1.5, capsize=3, markersize=5, zorder=3,
            )

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=7.5, color=_COLOR_TEXT_SECONDARY)
    ax.invert_yaxis()
    unit_label = _feature_unit_label(variant, feature_names)
    ax.set_xlabel(f"feature value ({unit_label})", color=_COLOR_TEXT_MUTED, fontsize=8)
    ax.set_title(
        f"Mode signatures ({variant}) -- top {len(top_features)} band/octave "
        "feature(s) by cross-mode separation (dot = mode median, whisker = IQR)",
        loc="left", fontsize=10, color="#0b0b0b",
    )
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=mode_color[m], label=m) for m in modes
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(modes), 4), frameon=False, fontsize=8,
    )
    ax.grid(axis="x", color=_COLOR_GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_COLOR_STEM)
    ax.tick_params(axis="both", colors=_COLOR_TEXT_MUTED, length=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return ax


def _run_mode_signatures(args: argparse.Namespace) -> int:
    run_names = _resolve_run_names(str(args.runs))
    if not run_names:
        print("mode-signatures: --runs got an empty run-name list", file=sys.stderr)
        return 2
    variant = str(args.variant)

    cfg = load_config()
    index = discover(cfg.data_root)
    unknown = _unknown_run_names(run_names, index)
    if unknown:
        available = ", ".join(sorted({r.name for r in index.runs})) or "(none discovered)"
        print(
            f"mode-signatures: unknown run name(s): {', '.join(unknown)}; "
            f"available runs: {available}",
            file=sys.stderr,
        )
        return 2

    out_root = _resolve_out_root(args.out, cfg.results_root)
    out_dir = out_root / "mode-signatures"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for name in run_names:
        try:
            rf = _run_features_and_gt(name, variant, cfg, index, use_cache=not args.no_cache)
        except RuntimeError as exc:
            print(f"mode-signatures: prepare_run failed for {name!r} ({exc})", file=sys.stderr)
            return 2
        if not rf.has_gt:
            logger.warning(
                "mode-signatures: run %r has no GT (Betriebsdaten) coverage at "
                "all -- skipped (A1.2: GT-bearing days only)", name,
            )
            continue
        cols = _band_octave_columns(rf.feature_names)
        if not cols:
            logger.warning(
                "mode-signatures: run %r / variant %r has zero band/octave "
                "column(s) -- skipped", name, variant,
            )
            continue
        table = _mode_profile(rf.features, rf.gt_states, cols)
        if table.empty:
            logger.warning(
                "mode-signatures: run %r has zero non-excluded GT mode with "
                ">= 1 window -- skipped", name,
            )
            continue
        table = table.copy()
        table["feature"] = [rf.feature_names[int(c)] for c in table["column"]]
        table = table[["mode", "feature", "column", "median", "q25", "q75", "n_windows"]]
        table.to_csv(out_dir / f"{name}.csv", index=False)
        _plot_mode_signatures(
            table, out_dir / f"{name}.png", variant, rf.feature_names, top_n=int(args.top_n)
        )
        written += 1

    if written == 0:
        print("mode-signatures: no run produced a profile (see warnings above)", file=sys.stderr)
        return 1
    print(f"mode-signatures: wrote {written} run(s) under {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand 5: tonal-table (Task 9, spec D3.5)
# ---------------------------------------------------------------------------

_OCTAVE_SUFFIX_RE = re.compile(r"_octave_(\d+)$")


def _stream_octave_centers(
    feature_names: list[str], stream_cols: np.ndarray
) -> dict[float, list[int]]:
    """`{octave_center_hz: [column indices]}` for *stream_cols* (already one
    run's one stream's own column indices) -- keyed by the center parsed
    straight from each column's own `_octave_<hz>` name suffix
    (`features.py`'s `AudioFeaturizer`/`VibFeaturizer` naming: `int(fc)`,
    e.g. `_octave_31` for the 31.5 Hz band) -- so the set of centers
    reflects whatever THIS run/stream actually produced (Nyquist truncation
    aware), never a hardcoded global list."""
    out: dict[float, list[int]] = {}
    for c in stream_cols:
        match = _OCTAVE_SUFFIX_RE.search(feature_names[int(c)])
        if match is None:
            continue
        out.setdefault(float(match.group(1)), []).append(int(c))
    return out


_OCTAVE_SPAN_SQRT2 = float(np.sqrt(2.0))
"""Octave half-bandwidth factor: a full-octave band centered at `fc` spans
`[fc / sqrt(2), fc * sqrt(2)]` (upper/lower ratio exactly 2, i.e. one
octave) -- VERIFIED to match `rowii.signals.features._octave_bands`'s own
`lo_hz, hi_hz = fc / sqrt(2), fc * sqrt(2)` span formula, duplicated here
(not imported: that helper is features.py-internal, leading-underscore --
this module already duplicates other VERIFIED facts from sibling
`src/rowii/` modules this way, e.g. `_LEVEL_SUBSTRINGS`/`_level_db_factor`)."""


def _octave_span_contains(center_hz: float, target_hz: float) -> bool:
    """Whether *target_hz* falls inside the full-octave span
    `[center_hz / sqrt(2), center_hz * sqrt(2)]` (`_OCTAVE_SPAN_SQRT2`)."""
    lo, hi = center_hz / _OCTAVE_SPAN_SQRT2, center_hz * _OCTAVE_SPAN_SQRT2
    return bool(lo <= target_hz <= hi)


def _nearest_octave_hz(target_hz: float, available_hz: Sequence[float]) -> float:
    """The octave CENTER in *available_hz* nearest to *target_hz*, restricted
    to centers whose OWN span (`_octave_span_contains`) does NOT already
    contain *target_hz* (T9-review item 2: a "floor" that contains the tone
    would let the background reading include the tone's own energy,
    defeating `tonal-table`'s contrast) -- absolute Hz distance among the
    tone-free survivors; an exact tie keeps the smaller center
    (deterministic). If EVERY candidate's own span contains *target_hz* (no
    tone-free floor exists among what this stream actually offers -- e.g. a
    single Nyquist-truncated octave column), every candidate re-enters the
    pool (logged) rather than the tone silently dropping out of the table.

    Raises:
        ValueError: *available_hz* is empty.
    """
    if not available_hz:
        raise ValueError("_nearest_octave_hz: available_hz is empty")
    tone_free = [hz for hz in available_hz if not _octave_span_contains(hz, target_hz)]
    if not tone_free:
        logger.warning(
            "_nearest_octave_hz: every candidate octave's own span contains "
            "%.3g Hz -- no tone-free floor among %s, falling back to the "
            "plain nearest center",
            target_hz, sorted(available_hz),
        )
        tone_free = list(available_hz)
    return min(tone_free, key=lambda hz: (abs(hz - target_hz), hz))


def _stream_machine_band_columns(
    feature_names: list[str], stream_cols: np.ndarray, band_name: str
) -> list[int]:
    """Indices of *stream_cols* whose name ends `_band_<band_name>` -- one
    `MACHINE_HZ` machine-tone column per live channel of that stream."""
    suffix = f"_band_{band_name}"
    return [int(c) for c in stream_cols if feature_names[int(c)].endswith(suffix)]


def _tonal_table(runs_features: Sequence[_RunFeatures]) -> pd.DataFrame:
    """Tidy (run, mode, stream, band, band_hz, octave_floor_hz, band_energy,
    octave_floor, tonal_contrast, n_windows) table: per GT-bearing run, per
    GT mode (`unknown`/`transition` excluded), per physical stream present,
    per `rowii.signals.features.MACHINE_HZ` band -- the tone's own median
    column energy (averaged across that stream's own live channels)
    contrasted (`_tonal_contrast`) against its NEAREST TONE-FREE OCTAVE
    FLOOR's median energy (`_nearest_octave_hz`, T9-review item 2: the
    octave center nearest by Hz among those ACTUALLY present for that
    stream, EXCLUDING any whose own span already contains the tone). A
    (stream, band) combination absent
    from a run (no matching column at all -- e.g. the band's upper edge
    exceeded that stream's Nyquist, or `VibFeaturizer` dropped every live
    channel) is silently omitted from that run's rows."""
    rows: list[dict[str, object]] = []
    for rf in runs_features:
        if not rf.has_gt:
            continue
        modes: set[str] = set(rf.gt_states.tolist())
        modes -= set(_EXCLUDED_GT)
        if not modes:
            continue
        for stream in _ALL_STREAMS:
            try:
                stream_cols = stream_columns(rf.feature_names, stream)
            except ValueError:
                continue
            octave_centers = _stream_octave_centers(rf.feature_names, stream_cols)
            if not octave_centers:
                continue
            for band_name, band_hz in MACHINE_HZ.items():
                band_cols = _stream_machine_band_columns(rf.feature_names, stream_cols, band_name)
                if not band_cols:
                    continue
                floor_hz = _nearest_octave_hz(band_hz, list(octave_centers))
                floor_cols = octave_centers[floor_hz]
                band_series = rf.features[:, band_cols].mean(axis=1)
                floor_series = rf.features[:, floor_cols].mean(axis=1)
                for mode in sorted(modes):
                    mask = rf.gt_states == mode
                    n = int(mask.sum())
                    if n == 0:
                        continue
                    band_energy = float(np.median(band_series[mask]))
                    octave_floor = float(np.median(floor_series[mask]))
                    rows.append(
                        {
                            "run": rf.run_name,
                            "mode": mode,
                            "stream": stream,
                            "band": band_name,
                            "band_hz": band_hz,
                            "octave_floor_hz": floor_hz,
                            "band_energy": band_energy,
                            "octave_floor": octave_floor,
                            "tonal_contrast": _tonal_contrast(band_energy, octave_floor),
                            "n_windows": n,
                        }
                    )
    columns = [
        "run", "mode", "stream", "band", "band_hz", "octave_floor_hz",
        "band_energy", "octave_floor", "tonal_contrast", "n_windows",
    ]
    return pd.DataFrame(rows, columns=columns)


def _plot_tonal_table(
    table: pd.DataFrame, out_path: Path, variant: str, feature_names: Sequence[str]
) -> Axes:
    """Heatmap, (run/mode x stream/band) -> `tonal_contrast` (own stored
    units; see module docstring's fusion z-score caveat), annotated with the
    numeric value per cell (mirrors `_plot_flag_rate_heatmap`'s layout, but
    SIGNED: `vmin`/`vmax` are the table's own min/max rather than a
    zero-anchored non-negative range). T9-review item 1: *variant* names
    the figure's own title, and the colorbar label follows
    `_feature_unit_label(variant, feature_names)`. Returns the rendered
    `Axes` (test seam; the colorbar itself lives on `ax.figure.axes[-1]`)."""
    if table.empty:
        fig, ax = plt.subplots(figsize=(8.5, 3.0))
        fig.patch.set_facecolor(_COLOR_SURFACE)
        ax.set_facecolor(_COLOR_SURFACE)
        ax.text(
            0.5, 0.5, "no (run, mode, stream, band) combination available",
            ha="center", va="center", color=_COLOR_TEXT_SECONDARY, fontsize=10,
        )
        ax.set_title(f"Tonal contrast ({variant})", loc="left", fontsize=10, color="#0b0b0b")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return ax

    pivot = table.copy()
    pivot["row"] = pivot["run"].astype(str) + " / " + pivot["mode"].astype(str)
    pivot["col"] = pivot["stream"].astype(str) + " " + pivot["band"].astype(str)
    rows = sorted(pivot["row"].unique())
    cols = sorted(pivot["col"].unique())
    matrix = pd.DataFrame(np.nan, index=rows, columns=cols)
    for _, r in pivot.iterrows():
        matrix.loc[r["row"], r["col"]] = r["tonal_contrast"]

    values = matrix.to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (-1.0, 1.0)
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    width = max(7.0, 0.9 * matrix.shape[1] + 3.0)
    height = max(3.0, 0.5 * matrix.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(_COLOR_SURFACE)
    ax.set_facecolor(_COLOR_SURFACE)

    im = ax.imshow(values, cmap=_sequential_blue_cmap(), vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(
        matrix.columns, rotation=45, ha="right", fontsize=7.5, color=_COLOR_TEXT_SECONDARY
    )
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=7.5, color=_COLOR_TEXT_SECONDARY)
    ax.set_xlabel("stream / machine band", color=_COLOR_TEXT_MUTED, fontsize=9)
    ax.set_ylabel("run / GT mode", color=_COLOR_TEXT_MUTED, fontsize=9)
    ax.set_title(
        f"Tonal contrast ({variant}) -- machine band vs. nearest "
        "non-containing octave floor (own units, our own SNR-like "
        "definition -- NOT the partner's metric)",
        loc="left", fontsize=10, color="#0b0b0b",
    )

    span = vmax - vmin
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                continue
            frac = (v - vmin) / span if span > 0 else 0.5
            text_color = "#ffffff" if frac > 0.6 else _COLOR_TEXT_SECONDARY
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    unit_label = _feature_unit_label(variant, feature_names)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        f"tonal contrast (band - nearest non-containing octave floor; {unit_label})",
        color=_COLOR_TEXT_MUTED, fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return ax


def _run_tonal_table(args: argparse.Namespace) -> int:
    run_names = _resolve_run_names(str(args.runs))
    if not run_names:
        print("tonal-table: --runs got an empty run-name list", file=sys.stderr)
        return 2
    variant = str(args.variant)

    cfg = load_config()
    index = discover(cfg.data_root)
    unknown = _unknown_run_names(run_names, index)
    if unknown:
        available = ", ".join(sorted({r.name for r in index.runs})) or "(none discovered)"
        print(
            f"tonal-table: unknown run name(s): {', '.join(unknown)}; "
            f"available runs: {available}",
            file=sys.stderr,
        )
        return 2

    runs_features: list[_RunFeatures] = []
    for name in run_names:
        try:
            rf = _run_features_and_gt(name, variant, cfg, index, use_cache=not args.no_cache)
        except RuntimeError as exc:
            print(f"tonal-table: prepare_run failed for {name!r} ({exc})", file=sys.stderr)
            return 2
        if not rf.has_gt:
            logger.warning(
                "tonal-table: run %r has no GT (Betriebsdaten) coverage at all "
                "-- excluded (A1.2: GT-bearing days only)", name,
            )
            continue
        runs_features.append(rf)

    table = _tonal_table(runs_features)
    if table.empty:
        print(
            "tonal-table: no (run, mode, stream, band) combination could be "
            "computed (see warnings above)",
            file=sys.stderr,
        )
        return 1

    out_root = _resolve_out_root(args.out, cfg.results_root)
    out_dir = out_root / "tonal-table"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = variant
    table.to_csv(out_dir / f"{stem}.csv", index=False)
    _plot_tonal_table(table, out_dir / f"{stem}.png", variant, runs_features[0].feature_names)
    print(
        f"tonal-table: wrote {out_dir / (stem + '.csv')} and .png "
        f"({len(runs_features)} GT-bearing day(s), {len(table)} row(s))"
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand 6: pillar3-figure (Task 9, spec D3.6)
# ---------------------------------------------------------------------------

_PILLAR3_LEAF_RE = re.compile(r"^(?P<rep>.+)-a(?P<alpha>\d+\.\d+)$")
_PILLAR3_COLUMNS: tuple[str, ...] = (
    "session", "representation", "alpha", "n_events", "n_detected",
    "event_tpr", "realized_window_far",
)


def _discover_pillar3_leaves(root: Path) -> pd.DataFrame:
    """Walk `root/<session>/<representation>-a<alpha>/event_eval.csv`
    (`scripts/eval_events.py`'s own tidy-CSV contract: ONE `row_type ==
    "summary"` row per leaf, columns `n_events`/`n_detected`/`event_tpr`/
    `false_alarm_windows`/`false_alarm_rate_per_hour`/`realized_window_far`/
    `tolerance_s`, verified against the real 080726 pillar-3 artifacts) into
    a tidy `_PILLAR3_COLUMNS` table -- one row per (session, representation,
    alpha). A leaf directory whose name carries no trailing `-a<alpha>`
    suffix (e.g. a `-frozen` leaf, outside the alpha grid this figure
    compares) is silently skipped, as is any leaf missing `event_eval.csv`
    or a `summary` row -- or whose CSV fails to parse or is missing an
    expected column at all (a corrupt/truncated write): T9-review item 3
    wraps the per-leaf read + summary extraction in a try/except
    (`ValueError`, `KeyError`, `OSError`, `pd.errors.ParserError`,
    `pd.errors.EmptyDataError`), logs a warning naming the leaf, and moves
    on -- matching `_discover_rotation_leaves`'s own skip-on-missing
    philosophy; one bad leaf never aborts discovery of the rest."""
    rows: list[dict[str, object]] = []
    if not root.is_dir():
        return pd.DataFrame(rows, columns=list(_PILLAR3_COLUMNS))
    for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for leaf_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
            match = _PILLAR3_LEAF_RE.match(leaf_dir.name)
            if match is None:
                continue
            csv_path = leaf_dir / "event_eval.csv"
            if not csv_path.is_file():
                continue
            try:
                table = pd.read_csv(csv_path)
                summary = table[table["row_type"] == "summary"]
                if summary.empty:
                    logger.warning(
                        "pillar3-figure: %s has no 'summary' row -- skipped", csv_path
                    )
                    continue
                r = summary.iloc[0]
                row: dict[str, object] = {
                    "session": session_dir.name,
                    "representation": match.group("rep"),
                    "alpha": float(match.group("alpha")),
                    "n_events": float(r["n_events"]),
                    "n_detected": float(r["n_detected"]),
                    "event_tpr": float(r["event_tpr"]),
                    "realized_window_far": float(r["realized_window_far"]),
                }
            except (
                ValueError,
                KeyError,
                OSError,
                pd.errors.ParserError,
                pd.errors.EmptyDataError,
            ) as exc:
                logger.warning("pillar3-figure: skipping %s (%s)", leaf_dir, exc)
                continue
            rows.append(row)
    return pd.DataFrame(rows, columns=list(_PILLAR3_COLUMNS))


def _plot_pillar3_figure(event_table: pd.DataFrame, out_path: Path) -> None:
    """One grouped-bar panel per pillar-3 session (`_tpr_by_alpha`'s own
    (representation x alpha) pivot), bars grouped by representation and
    coloured by alpha, sharing ONE alpha-color legend across every panel. A
    (representation, alpha) combination never discovered for a session draws
    no bar at all (not a zero-height bar -- a missing evaluation is not the
    same claim as a measured zero TPR)."""
    if event_table.empty:
        fig, ax = plt.subplots(figsize=(8.5, 3.0))
        fig.patch.set_facecolor(_COLOR_SURFACE)
        ax.set_facecolor(_COLOR_SURFACE)
        ax.text(
            0.5, 0.5, "no pillar-3 event_eval.csv summary row discovered",
            ha="center", va="center", color=_COLOR_TEXT_SECONDARY, fontsize=10,
        )
        ax.axis("off")
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return

    sessions = sorted(event_table["session"].unique())
    all_alphas = sorted(event_table["alpha"].unique())
    alpha_color = {
        a: _CATEGORICAL_HEX[i % len(_CATEGORICAL_HEX)] for i, a in enumerate(all_alphas)
    }

    fig, axes_2d = plt.subplots(
        1, len(sessions), figsize=(4.2 * len(sessions) + 1.5, 4.5), sharey=True, squeeze=False
    )
    fig.patch.set_facecolor(_COLOR_SURFACE)
    axes = list(axes_2d[0])

    for ax, session in zip(axes, sessions, strict=True):
        ax.set_facecolor(_COLOR_SURFACE)
        sub = event_table[event_table["session"] == session]
        matrix = _tpr_by_alpha(sub)
        reps = list(matrix.index)
        alphas = list(matrix.columns)
        n_alpha = max(len(alphas), 1)
        width = 0.8 / n_alpha
        x = np.arange(len(reps))
        for i, alpha in enumerate(alphas):
            color = alpha_color[alpha]
            offset = (i - (n_alpha - 1) / 2) * width
            for j, rep in enumerate(reps):
                v = matrix.loc[rep, alpha]
                if not np.isfinite(v):
                    continue
                ax.bar(float(x[j] + offset), float(v), width=width, color=color, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(reps, rotation=45, ha="right", fontsize=8, color=_COLOR_TEXT_SECONDARY)
        ax.set_title(str(session), loc="left", fontsize=10, color="#0b0b0b")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", color=_COLOR_GRIDLINE, linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(_COLOR_STEM)
        ax.spines["left"].set_color(_COLOR_STEM)
        ax.tick_params(axis="both", colors=_COLOR_TEXT_MUTED, length=0)

    axes[0].set_ylabel("event TPR", color=_COLOR_TEXT_MUTED, fontsize=9)
    handles = [Patch(facecolor=alpha_color[a], label=f"alpha={a:g}") for a in all_alphas]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(all_alphas), 6), frameon=False, fontsize=8,
    )
    fig.suptitle(
        "Pillar-3 event TPR by alpha, per representation (080726 strike sessions)",
        fontsize=11, color="#0b0b0b", x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _run_pillar3_figure(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root is not None else _default_pillar3_root()
    table = _discover_pillar3_leaves(root)
    if table.empty:
        print(
            f"pillar3-figure: no '<representation>-a<alpha>/event_eval.csv' "
            f"leaf found under {root}",
            file=sys.stderr,
        )
        return 1

    out_root = _resolve_out_root(args.out, None)
    out_dir = out_root / "pillar3-figure"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "pillar3.csv", index=False)
    _plot_pillar3_figure(table, out_dir / "pillar3.png")
    print(
        f"pillar3-figure: wrote {out_dir / 'pillar3.csv'} and .png "
        f"({len(table)} (session, representation, alpha) row(s))"
    )
    return 0


# ---------------------------------------------------------------------------
# digest (Task 9, spec D3 closing paragraph)
# ---------------------------------------------------------------------------

_FUSION_ZSCORE_FINDING = (
    "**Fusion's built-in per-run z-score (A1.1 finding).** `fuse()` z-scores "
    "each stream per run before concatenating (`rowii.signals.features.fuse`), "
    "so every fusion-variant feature column is already a dimensionless "
    "per-run z-score by construction. This is an implicit session "
    "normalization baked into the fusion representation itself, and it "
    "plausibly explains fusion's own cross-day FAR advantage over the "
    "raw-scale audio/vibration variants in the P7 rotations -- an "
    "accounting, not a claim that fusion 'solves' drift; the level-recal "
    "(D2) comparisons and the rotations-heatmap figure below test that "
    "hypothesis directly rather than assume it."
)

_FIGURE_READINGS: dict[str, tuple[str, str | None]] = {
    "rotations-heatmap": (
        "Each cell is one held-out rotation's pooled realized flag rate "
        "(darker blue = more windows flagged): read a row to see how one "
        "fit pool behaves across every held-out test day, and a column to "
        "see how one test day is treated by every fit pool. A plain "
        "readability replacement for the underlying far_table_*.csv files, "
        "not a new metric.",
        None,
    ),
    "feature-stability": (
        "One dot-and-whisker row per (feature, GT mode) pair still standing "
        "after ranking by cross-day shift: the dot is that day's median "
        "feature value (own stored units), the whisker its segment-block "
        "bootstrap 95% CI, and the colour is the day. The slow/drifting "
        "split in the CSV uses the named 3.0 cutoff purely for "
        "comparability with Rodrigues & Zhang (2026)'s own stability-classes "
        "analysis; the continuous shift distribution here is the primary "
        "reading.",
        "Rodrigues & Zhang (2026)",
    ),
    "era-step": (
        "One line per physical stream (mic vs. vibration), one point per "
        "day, tracing that stream's own median level across the recording "
        "era; a hollow diamond marks a day with no GT match for the "
        "requested mode (e.g. 2026-06-27, which has no Betriebsdaten at "
        "all). Consistent with -- and independently computed alongside -- "
        "Rodrigues & Zhang (2026)'s own reported mic-only broadband level "
        "step at the same era boundary; this figure never reads their "
        "number, only ours.",
        "Rodrigues & Zhang (2026)",
    ),
    "mode-signatures": (
        "One row per (band/octave feature, GT mode) on a single day, dot = "
        "that mode's own median, whisker = its interquartile range: modes "
        "whose whiskers do not overlap are separable on that feature alone. "
        "An independent, per-day replication of Rodrigues & Zhang (2026)'s "
        "reported within-day mode separability, computed only from our own "
        "features and caches. The figure's own title names the variant it "
        "was rendered from; for `fusion` the x-axis reads a per-run "
        "z-score (dimensionless) rather than log10, and for an embedding "
        "variant it reads that model's own embedding units instead.",
        "Rodrigues & Zhang (2026)",
    ),
    "tonal-table": (
        "A (run/mode x stream/band) heatmap of the shaft, blade-pass, and "
        "guide-vane machine-tone energies relative to their own nearest "
        "non-containing octave (the nearest octave band whose own span "
        "does not already contain the tone) -- positive means the tone "
        "reads above its local background, negative means it does not. An "
        "SNR-like contrast defined entirely from our own band/octave "
        "features (see the module docstring), NOT the partner's exact "
        "metric -- it only shares the analysis TYPE with Rodrigues & Zhang "
        "(2026)'s own machine-fingerprint table. The colorbar label "
        "follows the same per-variant units as mode-signatures: log10 for "
        "a raw-scale variant, a per-run z-score for `fusion`, or an "
        "embedding variant's own embedding units.",
        "Rodrigues & Zhang (2026)",
    ),
    "pillar3-figure": (
        "Grouped bars of event-level TPR by alpha for every representation "
        "evaluated on the 080726 strike sessions, one panel per session; "
        "taller bars mean more of the induced strike/sweep events were "
        "detected at that operating point. Read straight from our own "
        "results/pillar3/**/event_eval.csv summary rows -- no partner "
        "number involved. `fusion-snorm` is fusion's own session-norm "
        "side arm (a per-run first-N-minute recalibration layered on top "
        "of fusion, P7 D3), not a fourth base representation alongside "
        "audio/vibration/fusion/audio-beats.",
        None,
    ),
}


def _render_digest(out_root: Path) -> str:
    """The `results/analysis-days/README.md` markdown text: a short intro,
    the A1.1 fusion-z-score finding, then one section per `_FIGURE_READINGS`
    entry (its plain-language reading, an A1.8 attribution line when that
    analysis type is partner-inspired, and a markdown link to every PNG
    THIS module has actually written so far under `out_root/<name>/`)."""
    lines: list[str] = [
        "# Package-8 explainability digest (D3)",
        "",
        "Publication-grade figures + CSVs from `scripts/analyze_days.py`, "
        "read straight from existing artifacts/warm caches -- no new sweeps. "
        "English only (`--lang de` not built, spec D3). Every number below "
        "is computed from our own caches/artifacts; no partner JSON or "
        "number is read by any code in this repository.",
        "",
        _FUSION_ZSCORE_FINDING,
        "",
    ]
    for name, (reading, attribution) in _FIGURE_READINGS.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(reading)
        if attribution is not None:
            lines.append("")
            lines.append(
                f"*Analysis type inspired by {attribution}; all numbers on this "
                "page are computed from our own artifacts (spec A1.8).*"
            )
        lines.append("")
        figure_dir = out_root / name
        pngs = sorted(figure_dir.glob("*.png")) if figure_dir.is_dir() else []
        if pngs:
            lines.append("Figures:")
            lines.append("")
            for png in pngs:
                lines.append(f"- [{png.name}]({name}/{png.name})")
        else:
            lines.append("_No figures discovered yet under this subcommand's directory._")
        lines.append("")
    return "\n".join(lines)


def _run_digest(args: argparse.Namespace) -> int:
    out_root = _resolve_out_root(args.out, None)
    out_root.mkdir(parents=True, exist_ok=True)
    text = _render_digest(out_root)
    (out_root / "README.md").write_text(text)
    print(f"digest: wrote {out_root / 'README.md'}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "D3 explainability analysis suite (Package-8): publication-grade "
            "figures + underlying CSVs from existing artifacts/warm caches -- "
            "rotations-heatmap, feature-stability, era-step, mode-signatures, "
            "tonal-table, pillar3-figure, digest (module docstring)."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_heat = sub.add_parser(
        "rotations-heatmap",
        help="Day x day pooled flag-rate heatmap from cross-day-pooled far tables.",
    )
    p_heat.add_argument(
        "--root", default=None,
        help="cross-day-pooled root to scan (default: <results_root>/step2/cross-day-pooled).",
    )
    p_heat.add_argument(
        "--out", default=None,
        help="Output root (default: <results_root>/analysis-days).",
    )
    p_heat.add_argument("--variant", required=True, help="e.g. audio, vibration, fusion.")
    p_heat.add_argument(
        "--mode", choices=("frozen", "recalibrate"), default="frozen",
        help="Which far_table_<mode>.csv to read (default: frozen).",
    )
    p_heat.add_argument(
        "--leaf-suffix", default="",
        help=(
            "Leaf directory suffix appended after '<variant>-pooled' (default: "
            "'' -- the plain leaf; e.g. '-a0.05' discovers '<variant>-pooled-a0.05')."
        ),
    )

    p_stab = sub.add_parser(
        "feature-stability",
        help="Per-feature cross-day shift, per GT mode, with segment-block bootstrap CIs.",
    )
    p_stab.add_argument("--runs", required=True, help="Comma-separated run names.")
    p_stab.add_argument("--variant", required=True, help="e.g. audio, vibration, fusion.")
    p_stab.add_argument(
        "--out", default=None, help="Output root (default: <results_root>/analysis-days)."
    )
    p_stab.add_argument("--n-boot", type=int, default=_STABILITY_N_BOOT_DEFAULT)
    p_stab.add_argument("--seed", type=int, default=_STABILITY_SEED_DEFAULT)
    p_stab.add_argument(
        "--cutoff-db", type=float, default=_STABILITY_CUTOFF_DB,
        help=(
            f"slow/drifting classification cutoff IN DB, applied to LEVEL "
            f"columns only after the family-factor dB conversion (default "
            f"{_STABILITY_CUTOFF_DB:g}, the A1.8 named, adopted-for-comparability "
            f"constant -- see module docstring)."
        ),
    )
    p_stab.add_argument("--min-mode-windows", type=int, default=_STABILITY_MIN_MODE_WINDOWS_DEFAULT)
    p_stab.add_argument(
        "--no-cache", action="store_true",
        help="Disable rowii.pipeline.prepare_run's on-disk feature cache.",
    )

    p_era = sub.add_parser(
        "era-step",
        help="Per-day per-stream (mic vs vibration) median level across the recording era.",
    )
    p_era.add_argument(
        "--runs", required=True, help="Comma-separated run names, chronological order."
    )
    p_era.add_argument(
        "--variant", required=True,
        help=(
            "audio or vibration only -- fusion's per-run z-scored features "
            "(fuse(), A1.1) have no meaningful raw-scale level to plot; run "
            "era-step once per variant for the mic-vs-vibration composite view."
        ),
    )
    p_era.add_argument(
        "--gt-mode", default="turbine",
        help="GT mode to match on GT-bearing days (default: turbine).",
    )
    p_era.add_argument(
        "--include-080726", action="store_true",
        help=(
            "Include 080726 run(s) (A1.2 gate -- only pass this after the D4.3 "
            "SCADA-timebase probe, scripts/verify_data_facts.py scada-timebase, "
            "confirms the changeover)."
        ),
    )
    p_era.add_argument(
        "--era-boundary-after", default=None,
        help="Draw a vertical era-boundary line right after this run name's x-position.",
    )
    p_era.add_argument(
        "--out", default=None, help="Output root (default: <results_root>/analysis-days)."
    )
    p_era.add_argument(
        "--no-cache", action="store_true",
        help="Disable rowii.pipeline.prepare_run's on-disk feature cache.",
    )

    p_modesig = sub.add_parser(
        "mode-signatures",
        help="Per-GT-mode band/octave profile (median + IQR), one PNG+CSV per day.",
    )
    p_modesig.add_argument("--runs", required=True, help="Comma-separated run names.")
    p_modesig.add_argument("--variant", required=True, help="e.g. audio, vibration, fusion.")
    p_modesig.add_argument(
        "--out", default=None, help="Output root (default: <results_root>/analysis-days)."
    )
    p_modesig.add_argument("--top-n", type=int, default=_MODE_SIGNATURES_TOP_N_DEFAULT)
    p_modesig.add_argument(
        "--no-cache", action="store_true",
        help="Disable rowii.pipeline.prepare_run's on-disk feature cache.",
    )

    p_tonal = sub.add_parser(
        "tonal-table",
        help="Per mode x day machine-tone band energy relative to its neighboring octave floor.",
    )
    p_tonal.add_argument("--runs", required=True, help="Comma-separated run names.")
    p_tonal.add_argument("--variant", required=True, help="e.g. audio, vibration, fusion.")
    p_tonal.add_argument(
        "--out", default=None, help="Output root (default: <results_root>/analysis-days)."
    )
    p_tonal.add_argument(
        "--no-cache", action="store_true",
        help="Disable rowii.pipeline.prepare_run's on-disk feature cache.",
    )

    p_pillar3 = sub.add_parser(
        "pillar3-figure",
        help="TPR-vs-alpha grouped bars per representation, both 080726 pillar-3 sessions.",
    )
    p_pillar3.add_argument(
        "--root", default=None, help="pillar3 root to scan (default: <results_root>/pillar3)."
    )
    p_pillar3.add_argument(
        "--out", default=None, help="Output root (default: <results_root>/analysis-days)."
    )

    p_digest = sub.add_parser(
        "digest",
        help="Write results/analysis-days/README.md linking every figure with a reading.",
    )
    p_digest.add_argument(
        "--out", default=None,
        help=(
            "Analysis-days root to scan + write README.md into "
            "(default: <results_root>/analysis-days)."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "rotations-heatmap":
        return _run_rotations_heatmap(args)
    if args.command == "feature-stability":
        return _run_feature_stability(args)
    if args.command == "era-step":
        return _run_era_step(args)
    if args.command == "mode-signatures":
        return _run_mode_signatures(args)
    if args.command == "tonal-table":
        return _run_tonal_table(args)
    if args.command == "pillar3-figure":
        return _run_pillar3_figure(args)
    if args.command == "digest":
        return _run_digest(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
