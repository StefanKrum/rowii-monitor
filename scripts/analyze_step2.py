"""Candidate-overlap analysis CLI (Task 7, package-2 design spec `docs/superpowers/
specs/2026-07-15-step2-scarcity-crossday-beats-design.md` D4): do two Step-2 sweep
combos of the same run -- typically a BEATs variant and a handcrafted variant --
flag the SAME moments in time?

Drives `rowii.anomaly.overlap` (`top_candidates`, `to_utc_ns`, `match_by_time`,
`jaccard`) over already-written `results/step2/.../scores.parquet` files: this
script performs NO scoring itself, only reads `scripts/run_step2.py`'s persisted
output. Two combos of the same run generally do NOT share a window grid
(`rowii.pipeline.prepare_run` builds each variant's grid from the intersection of
that variant's OWN streams -- `rowii.anomaly.overlap` module docstring), so a
candidate's `window` is converted to an absolute UTC nanosecond timestamp via its
OWN combo's grid before the two combos' candidates are ever compared -- raw window
indices from different combos are never compared directly.

## Combo names

A "combo" is `"<variant>-<scorer>"` (e.g. `"fusion-knn"`, `"audio-beats-knn"`);
`_parse_combo` splits on the LAST hyphen, always exactly the variant/scorer
boundary since no variant string (`rowii.pipeline`'s five concrete variants) ends
in `"-knn"` or `"-mahalanobis"`. This script always reads the DETECTED-labels,
PER-STATE-conditioning sweep for a combo -- `scripts/run_step2.py`'s own
runtime-realistic default (module docstring: "detected ... the only run-time-
realistic mode"; package-2 design spec: pooled cross-day is "structurally
unusable") -- so a bare combo name is unambiguous without a separate labels/
conditioning axis.

## Run names and combo-directory lookup

`--runs` accepts the three kinds of name `scripts/run_step2.py`'s own
`summary.csv` `run` column already uses (that module's docstring, "Output layout"
section): a bare day name (within-day), `"<dayA>__to__<dayB>"` (cross-day), or
`"<dayA>--to--<dayB>"` (cross-day-per-state, double-dash). For a given run name +
combo, `_locate_combo_dir` tries the three protocols' own output-directory
conventions in that order (`_within_day_combo_dir`/`_cross_day_combo_dir`/
`_cross_day_per_state_combo_dir`, reproducing `scripts/run_step2.py`'s
`_within_day_out_dir`/`_cross_day_out_dir`/`_cross_day_per_state_out_dir` layouts
rather than importing them -- this script depends only on the ON-DISK layout those
functions define, mirroring `scripts/warm_cache.py`'s own "never import a sibling
script's internals" stance) and uses whichever directory actually has a
`scores.parquet`. Cross-day and cross-day-per-state combo dirs are always keyed to
day B: both protocols score EXCLUSIVELY day B's windows (`rowii.anomaly.sweep`/
`run_step2.py` -- day A only ever contributes calibration), so the grid used to
convert that combo's candidates to UTC is always day B's own `PreparedRun.grid`
regardless of which protocol produced the file (`_day_b_name`).

## Global top-k as "the combo's candidates"

`rowii.anomaly.overlap.top_candidates` returns per-label top-k rows AND one
across-all-labels ("global", `label=GLOBAL_LABEL`) top-k group in one DataFrame.
This script's notion of "a combo's candidate set" -- for both the pairwise overlap
report and the needs-listening cross-check -- is the GLOBAL group only: "do these
two combos flag the same moments" is naturally one ranked list per combo, and
matching the per-label rows too would double-count a window that is both its own
label's top-k AND the global top-k.

## Output

- `results/step2/overlap/<run>--<comboA>--vs--<comboB>.md` per `--pairs` entry x
  `--runs` entry (`_write_overlap_report`): candidate counts, Jaccard, a match
  table (human-readable UTC), and both combos' unmatched candidates.
- `results/step2/overlap/needs_listening_check.md` (`_write_needs_listening_
  report`), written once per invocation: each `--check-utc` timestamp against
  EVERY combo actually analyzed above (deduplicated by (run, combo), regardless of
  how many `--pairs` entries reference it) -- a hit/miss table plus, on a hit, the
  nearest candidate's own time offset/label/p-value.

A `--pairs` entry that cannot be located on disk (e.g. a BEATs sweep that has not
been run yet for that run) is logged and SKIPPED -- the remaining pairs still run,
same catch-and-skip principle `scripts/run_step2.py`'s own sweep orchestrators
apply to a single failing combo; the invocation exits 1 if any pair was skipped
this way. A beats-variant grid lookup with NO warm cache entry is different in
kind -- not "this data doesn't exist yet" but "this call is about to trigger an
hours-long extraction" -- so `_grid_for_combo` raises `SystemExit` instead
(uncaught here), refusing to proceed at all rather than silently either running the
extraction or masking it as a skipped pair.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.anomaly.overlap import (  # noqa: E402
    GLOBAL_LABEL,
    jaccard,
    match_by_time,
    to_utc_ns,
    top_candidates,
)
from rowii.config import Config, load_config  # noqa: E402
from rowii.io.dataset import discover, run_utc_offset_ns  # noqa: E402
from rowii.pipeline import (  # noqa: E402
    _cache_fingerprint,
    _cache_npz_path,
    _is_beats_variant,
    _load_cached_prepared_run,
    _streams_for_variant,
    prepare_run,
)
from rowii.signals.windows import WindowGrid  # noqa: E402

logger = logging.getLogger(__name__)

_SCORER_SUFFIXES: tuple[str, ...] = ("knn", "mahalanobis")
_DEFAULT_TOL_S = 5.0
_OVERLAP_DIR_NAME = "overlap"


# ---------------------------------------------------------------------------
# Combo names + on-disk lookup
# ---------------------------------------------------------------------------


def _parse_combo(combo: str) -> tuple[str, str]:
    """`"<variant>-<scorer>"` -> `(variant, scorer)`, splitting on the LAST hyphen
    (module docstring's "Combo names" section).

    Raises:
        ValueError: *combo* does not end in `"-knn"` or `"-mahalanobis"`.
    """
    for scorer in _SCORER_SUFFIXES:
        suffix = f"-{scorer}"
        if combo.endswith(suffix):
            return combo[: -len(suffix)], scorer
    raise ValueError(f"combo {combo!r} must end in one of {_SCORER_SUFFIXES!r}")


def _split_pair(pair: str) -> tuple[str, str]:
    """`"<comboA>:<comboB>"` -> `(comboA, comboB)`, splitting on the first colon.

    Raises:
        ValueError: *pair* has no colon.
    """
    if ":" not in pair:
        raise ValueError(f'--pairs entry {pair!r} must be "<comboA>:<comboB>"')
    combo_a, combo_b = pair.split(":", 1)
    return combo_a, combo_b


def _within_day_combo_dir(results_root: Path, run: str, variant: str, scorer: str) -> Path:
    """Mirrors `scripts/run_step2.py::_within_day_out_dir` for `labels="detected"`,
    `conditioning="per-state"` (module docstring)."""
    return (
        results_root / "step2" / "within-day" / run
        / f"{variant}-detected" / f"per-state-{scorer}"
    )


def _cross_day_combo_dir(results_root: Path, run: str, variant: str, scorer: str) -> Path:
    """Mirrors `scripts/run_step2.py::_cross_day_out_dir` for `labels="detected"`;
    *run* is the pair string `"<dayA>__to__<dayB>"`."""
    return (
        results_root / "step2" / "cross-day" / f"{variant}-detected"
        / run / f"{scorer}-pooled"
    )


def _cross_day_per_state_combo_dir(results_root: Path, run: str, variant: str, scorer: str) -> Path:
    """Mirrors `scripts/run_step2.py::_cross_day_per_state_out_dir`; *run* is the
    (double-dash) pair string `"<dayA>--to--<dayB>"`."""
    return results_root / "step2" / "cross-day-per-state" / run / f"{variant}-{scorer}"


def _locate_combo_dir(results_root: Path, run: str, combo: str) -> Path:
    """The combo directory, among within-day/cross-day/cross-day-per-state (that
    order), that actually has a `scores.parquet` for (*run*, *combo*) -- module
    docstring's "Run names and combo-directory lookup" section.

    Raises:
        FileNotFoundError: none of the three candidate directories has a
            `scores.parquet`.
    """
    variant, scorer = _parse_combo(combo)
    for candidate in (
        _within_day_combo_dir(results_root, run, variant, scorer),
        _cross_day_combo_dir(results_root, run, variant, scorer),
        _cross_day_per_state_combo_dir(results_root, run, variant, scorer),
    ):
        if (candidate / "scores.parquet").is_file():
            return candidate
    raise FileNotFoundError(
        f"no scores.parquet for run={run!r} combo={combo!r} under any of "
        f"within-day/cross-day/cross-day-per-state ({results_root / 'step2'})"
    )


def _day_b_name(run: str) -> str:
    """The day whose windows a combo's `scores.parquet` actually indexes: *run*
    itself for a within-day run name, else the LAST component of a cross-day(-per-
    state) pair string (module docstring -- both cross-day protocols score
    exclusively day B's windows, day A never contributes a scored window)."""
    for sep in ("__to__", "--to--"):
        if sep in run:
            return run.rsplit(sep, 1)[1]
    return run


# ---------------------------------------------------------------------------
# Grid lookup (the one seam a caller monkeypatches -- see module docstring)
# ---------------------------------------------------------------------------


def _grid_for_combo(day_name: str, variant: str, cfg: Config) -> WindowGrid:
    """*day_name*'s `WindowGrid` for *variant*, via `rowii.pipeline.prepare_run(...,
    use_cache=True)` -- discovers *day_name* under `cfg.data_root`
    (`rowii.io.dataset.discover`) and delegates the rest to `prepare_run`. With a
    warm cache (this package's Task 6) every call is a cache hit, so `prepare_run`
    never re-extracts features and, for a beats variant, never imports torch (the
    featurizer only runs on a MISS -- `rowii.pipeline.prepare_run`'s own module
    docstring).

    For a beats variant specifically, a miss would mean a from-scratch BEATs
    extraction over the whole run (`scripts/warm_cache.py`'s own module docstring:
    "expensive relative to handcrafted features", observed 334-784s per combo even
    on comparatively short runs) -- this is a lightweight analysis script, not a
    place to trigger that silently, so beats variants are checked against the
    on-disk cache FIRST, using `rowii.pipeline`'s own private cache primitives
    (the same cross-module-privates precedent `scripts/warm_cache.py` already sets
    for `_cache_npz_path`/`_is_beats_variant`), and refuse before ever reaching
    `prepare_run`'s extraction path.

    This whole function is the ONE seam `tests/test_overlap.py`'s script-level
    smoke test monkeypatches to avoid real data-root discovery, real cache I/O, and
    any torch import entirely -- production `main()` never calls `rowii.io.dataset.
    discover` or `prepare_run` directly itself, only through here.

    Raises:
        SystemExit: *day_name* was not discovered under `cfg.data_root`; or
            *variant* is a beats variant with no warm cache entry for (*day_name*,
            *variant*) matching the current fingerprint (missing file, or stale/
            mismatched content) -- message names the exact `scripts/warm_cache.py`
            invocation to fix it.
    """
    index = discover(cfg.data_root)
    by_name = {r.name: r for r in index.runs}
    run = by_name.get(day_name)
    if run is None:
        raise SystemExit(
            f"analyze_step2: run {day_name!r} not discovered under {cfg.data_root} "
            f"(available: {sorted(by_name)})"
        )
    if _is_beats_variant(variant):
        cache_path = _cache_npz_path(cfg.results_root, run.name, variant)
        fingerprint = _cache_fingerprint(run, variant, cfg)
        streams = _streams_for_variant(variant)
        offset_ns = run_utc_offset_ns(run)
        cached = _load_cached_prepared_run(
            cache_path, fingerprint, run, streams, cfg.window.window_s, offset_ns
        )
        if cached is None:
            raise SystemExit(
                f"analyze_step2: no warm cache for {run.name} x {variant} (expected "
                f"{cache_path} with a matching fingerprint) -- run `python scripts/"
                f"warm_cache.py --runs {run.name} --variants {variant}` first; this "
                "script refuses to trigger a from-scratch BEATs extraction silently."
            )
    return prepare_run(run, variant, cfg, use_cache=True).grid


# ---------------------------------------------------------------------------
# Per-combo candidates: load scores.parquet -> global top-k -> UTC
# ---------------------------------------------------------------------------


def _load_combo_candidates(cfg: Config, run: str, combo: str, top_k: int) -> pd.DataFrame:
    """*combo*'s global top-`top_k` candidates for *run*, with a `t_utc_ns` column
    (module docstring's "Global top-k as 'the combo's candidates'" section).

    Raises:
        FileNotFoundError: propagated from `_locate_combo_dir`.
        SystemExit: propagated from `_grid_for_combo` (beats cache miss / unknown
            run -- see its own docstring).
    """
    variant, _scorer = _parse_combo(combo)
    combo_dir = _locate_combo_dir(cfg.results_root, run, combo)
    scores = pd.read_parquet(combo_dir / "scores.parquet", engine="pyarrow")
    top = top_candidates(scores, top_k)
    global_top = top[top["label"] == GLOBAL_LABEL].reset_index(drop=True)
    grid = _grid_for_combo(_day_b_name(run), variant, cfg)
    return to_utc_ns(global_top, grid.t0_ns, grid.window_ns)


def _ensure_combo_candidates(
    cfg: Config,
    run: str,
    combo: str,
    top_k: int,
    cache: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame | None:
    """Cached `_load_combo_candidates(cfg, run, combo, top_k)`, keyed by (*run*,
    *combo*) in *cache* so a combo referenced by several `--pairs` entries is only
    ever loaded once. Returns `None` (never raises) on `FileNotFoundError` -- the
    "this data doesn't exist yet" case `main` treats as skip-and-continue (module
    docstring) -- so a combo that failed does not poison a PARTNER combo that
    loaded fine; `main` decides per-pair what a `None` means for that pair's
    report. `SystemExit` (a beats cache miss, a different KIND of failure -- module
    docstring) still propagates uncaught.
    """
    key = (run, combo)
    if key in cache:
        return cache[key]
    try:
        candidates = _load_combo_candidates(cfg, run, combo, top_k)
    except FileNotFoundError as exc:
        logger.warning("analyze_step2: no data for run=%r combo=%r: %s", run, combo, exc)
        return None
    cache[key] = candidates
    return candidates


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _fmt_utc(t_utc_ns: int) -> str:
    """`rowii.signals.windows.WindowGrid.edges_ns`/`scripts/run_step2.py::
    _candidates_markdown`'s own UTC rendering convention, reused here for
    consistency across every Step-2 markdown report."""
    return str(pd.Timestamp(int(t_utc_ns), unit="ns", tz="UTC").isoformat())


def _fmt_float(value: float, decimals: int = 6) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{decimals}g}"


def _unmatched_table(unmatched: pd.DataFrame) -> str:
    if unmatched.empty:
        return "None."
    lines = ["| utc | label | p_value |", "|---|---|---|"]
    for _, r in unmatched.iterrows():
        lines.append(f"| {_fmt_utc(r['t_utc_ns'])} | {r['label']} | {_fmt_float(r['p_value'])} |")
    return "\n".join(lines)


def _write_overlap_report(
    overlap_dir: Path,
    run: str,
    combo_a: str,
    combo_b: str,
    cands_a: pd.DataFrame,
    cands_b: pd.DataFrame,
    matches: pd.DataFrame,
) -> None:
    """`<run>--<comboA>--vs--<comboB>.md`: candidate counts, Jaccard, a match table
    (human-readable UTC), and both combos' unmatched candidates (orchestrator
    resolution, Task 7)."""
    jac = jaccard(len(cands_a), len(cands_b), len(matches))
    lines = [f"# Candidate overlap: {run} -- {combo_a} vs {combo_b}", ""]
    lines.append(
        f"Global top-k candidates (`rowii.anomaly.overlap.top_candidates`, "
        f"`label=\"{GLOBAL_LABEL}\"`), matched via UTC time "
        f"(`match_by_time`, tol_s={_DEFAULT_TOL_S})."
    )
    lines.append("")
    lines.append(f"- {combo_a}: {len(cands_a)} candidate(s)")
    lines.append(f"- {combo_b}: {len(cands_b)} candidate(s)")
    lines.append(f"- matched: {len(matches)}")
    lines.append(f"- Jaccard: {jac:.3f}")
    lines.append("")

    lines.append("## Matches")
    lines.append("")
    if matches.empty:
        lines.append("No matches within tolerance.")
    else:
        lines.append("| utc_a | utc_b | dt_s | label_a | label_b | p_value_a | p_value_b |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in matches.iterrows():
            lines.append(
                f"| {_fmt_utc(r['t_utc_ns_a'])} | {_fmt_utc(r['t_utc_ns_b'])} | "
                f"{r['dt_s']:.3f} | {r['label_a']} | {r['label_b']} | "
                f"{_fmt_float(r['p_value_a'])} | {_fmt_float(r['p_value_b'])} |"
            )
    lines.append("")

    unmatched_a = cands_a[~cands_a["t_utc_ns"].isin(matches["t_utc_ns_a"])]
    unmatched_b = cands_b[~cands_b["t_utc_ns"].isin(matches["t_utc_ns_b"])]
    lines.append(f"## Unmatched ({combo_a} only)")
    lines.append("")
    lines.append(_unmatched_table(unmatched_a))
    lines.append("")
    lines.append(f"## Unmatched ({combo_b} only)")
    lines.append("")
    lines.append(_unmatched_table(unmatched_b))
    lines.append("")

    path = overlap_dir / f"{run}--{combo_a}--vs--{combo_b}.md"
    path.write_text("\n".join(lines))


def _parse_check_utc(value: str) -> int:
    """ISO8601 string -> UTC nanoseconds since epoch (int64-range). A tz-naive
    string is assumed already UTC (localized, not converted); a tz-aware string is
    converted to UTC -- either way the result is directly comparable to a
    `to_utc_ns`-produced `t_utc_ns` column."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.value)


def _write_needs_listening_report(
    overlap_dir: Path,
    check_utc: list[str],
    analyzed: list[tuple[str, str]],
    candidates_by_combo: dict[tuple[str, str], pd.DataFrame],
) -> None:
    """`needs_listening_check.md`: each *check_utc* timestamp against EVERY
    *analyzed* (run, combo)'s own candidates (module docstring's "Output" section)
    -- a hit/miss table, hit rows also naming the nearest candidate's own time
    offset/label/p-value. Reuses `match_by_time` itself (a single "check" row
    against a combo's candidates is just a degenerate 1-vs-N nearest-time match)
    rather than a second ad hoc distance computation.
    """
    lines = ["# Needs-listening check", ""]
    lines.append(
        f"Each `--check-utc` timestamp checked against every analyzed combo's "
        f"global top-k candidates, tol_s={_DEFAULT_TOL_S} (`rowii.anomaly.overlap."
        f"match_by_time`)."
    )
    lines.append("")
    if not check_utc:
        lines.append("No --check-utc timestamps were provided.")
    elif not analyzed:
        lines.append("No combo was successfully analyzed.")
    else:
        lines.append("| check_utc | run | combo | hit | dt_s | label | p_value |")
        lines.append("|---|---|---|---|---|---|---|")
        for utc_str in check_utc:
            check_row = pd.DataFrame({
                "t_utc_ns": [_parse_check_utc(utc_str)],
                "label": ["__check__"],
                "p_value": [math.nan],
            })
            for run, combo in analyzed:
                cands = candidates_by_combo[(run, combo)]
                m = match_by_time(check_row, cands, tol_s=_DEFAULT_TOL_S)
                if m.empty:
                    lines.append(f"| {utc_str} | {run} | {combo} | miss | n/a | n/a | n/a |")
                else:
                    r = m.iloc[0]
                    lines.append(
                        f"| {utc_str} | {run} | {combo} | hit | {r['dt_s']:.3f} | "
                        f"{r['label_b']} | {_fmt_float(r['p_value_b'])} |"
                    )
    lines.append("")
    (overlap_dir / "needs_listening_check.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Candidate-overlap analysis: do two Step-2 sweep combos of the same run "
            "(typically a BEATs variant and a handcrafted variant) flag the same "
            "moments in time? Reads already-written scores.parquet files, matches "
            "each pair's top-k candidates via UTC time, and writes a match/Jaccard "
            "report per pair plus a needs-listening cross-check."
        )
    )
    parser.add_argument(
        "--results-root", default="results", metavar="DIR",
        help='Root results directory (default: "results") -- combos are read from '
             "<results-root>/step2/..., reports written to <results-root>/step2/overlap/.",
    )
    parser.add_argument(
        "--runs", nargs="+", required=True, metavar="RUN",
        help="Run name(s) -- a bare day name, or a cross-day(-per-state) pair "
             '("<dayA>__to__<dayB>" / "<dayA>--to--<dayB>").',
    )
    parser.add_argument(
        "--pairs", nargs="+", required=True, metavar="COMBO_A:COMBO_B",
        help='Combo pair(s) to compare, e.g. "fusion-knn:audio-beats-knn". Each '
             "pair is analyzed for every --runs entry.",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Global top-k candidates per combo (default: 20).",
    )
    parser.add_argument(
        "--check-utc", nargs="*", default=[], metavar="ISO8601",
        help="Timestamp(s) to cross-check against every analyzed combo's top-k "
             "(default: none).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate every --pairs entry COMPLETELY up front -- both the ":" split and
    # each half's combo suffix -- so a malformed combo is an argparse usage error
    # (exit 2) before any I/O, never a raw ValueError traceback midway through the
    # pair loop after earlier pairs already wrote reports (Task 7 review finding 1).
    try:
        pairs = [_split_pair(p) for p in args.pairs]
        for combo_a, combo_b in pairs:
            _parse_combo(combo_a)
            _parse_combo(combo_b)
    except ValueError as exc:
        parser.error(str(exc))

    cfg = dataclasses.replace(load_config(), results_root=Path(args.results_root))
    overlap_dir = cfg.results_root / "step2" / _OVERLAP_DIR_NAME
    overlap_dir.mkdir(parents=True, exist_ok=True)

    candidates_by_combo: dict[tuple[str, str], pd.DataFrame] = {}
    analyzed: list[tuple[str, str]] = []
    n_failed = 0

    for run in args.runs:
        for combo_a, combo_b in pairs:
            cands_a = _ensure_combo_candidates(cfg, run, combo_a, args.top_k, candidates_by_combo)
            cands_b = _ensure_combo_candidates(cfg, run, combo_b, args.top_k, candidates_by_combo)

            # A combo that loaded fine is registered for the needs-listening check
            # even if its PAIR PARTNER below failed -- the two are independent, see
            # `_ensure_combo_candidates`'s own docstring.
            for combo, cands in ((combo_a, cands_a), (combo_b, cands_b)):
                key = (run, combo)
                if cands is not None and key not in analyzed:
                    analyzed.append(key)

            if cands_a is None or cands_b is None:
                n_failed += 1
                logger.warning(
                    "analyze_step2: skipping overlap report for %s (%s vs %s): "
                    "missing scores.parquet for at least one side",
                    run, combo_a, combo_b,
                )
                continue

            matches = match_by_time(cands_a, cands_b, tol_s=_DEFAULT_TOL_S)
            _write_overlap_report(overlap_dir, run, combo_a, combo_b, cands_a, cands_b, matches)

    _write_needs_listening_report(overlap_dir, args.check_utc, analyzed, candidates_by_combo)

    if n_failed:
        print(f"analyze_step2: {n_failed} pair(s) skipped (see warnings above)")
        return 1
    print(f"analyze_step2: wrote overlap report(s) + needs_listening_check.md to {overlap_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
