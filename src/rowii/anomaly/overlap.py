"""Candidate-overlap analysis: do two Step-2 sweep combos flag the SAME moments in
time? (Task 7, package-2 design spec `docs/superpowers/specs/2026-07-15-step2-
scarcity-crossday-beats-design.md` D4: "candidate top-K overlap vs handcrafted
candidates (UTC time-window intersection + Jaccard + qualitative table)".)

The motivating comparison is a BEATs-based sweep (`audio-beats`/`fusion-beats`)
against a handcrafted-feature sweep (`audio`/`vibration`/`fusion`) of the SAME run,
but nothing here assumes that specific pairing -- these are plain pandas/numpy
primitives over two already-computed `SweepResult.scores`-shaped DataFrames
(`scripts/run_step2.py`'s persisted `scores.parquet`, columns `window, label,
score, p_value, alarm`, `label` a `str` after the parquet round-trip -- `_write_
sweep_outputs`' own docstring).

`rowii.pipeline.prepare_run` builds each (run, variant) combo's `WindowGrid` from
the INTERSECTION of that variant's own streams' `[t0, t_end)`
(`rowii.signals.windows.common_grid`) -- two variants of the same run essentially
never share the exact same grid (different streams, different coverage gaps), so a
raw `window` index means a DIFFERENT moment in each combo's own grid. Every
function below that compares two combos' candidates therefore operates on absolute
UTC nanosecond timestamps (`to_utc_ns`), never on `window` directly -- the caller
(`scripts/analyze_step2.py`) is responsible for converting each combo's own
candidates through its OWN grid before anything here ever sees them.

`top_candidates`' ranking -- p-value ascending, then score DESCENDING, then window
ascending -- is `np.lexsort((window, -score, p_value))`, the IDENTICAL convention
already used by `rowii.anomaly.sweep.scores_and_candidates` and `scripts/
run_step2.py::_cross_day_sweep` (both build one sweep's own top-k candidate
register this same way); this module reuses it rather than inventing a second
ranking rule for the same kind of table (orchestrator resolution, Task 7). See
`scores_and_candidates`'s own docstring for why the score-descending tie-break
matters (conformal p-values collapse to the same achievable minimum for every
window past the calibration maximum, so p-value alone under-ranks the most extreme
of a tied group).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GLOBAL_LABEL = "__any__"
"""`top_candidates`' label for its across-all-labels top-k rows (module docstring;
brief interface spec) -- collides with a real label only if a state were itself
named `"__any__"`, which no detected cluster id (int) or GT state string in this
project ever is (mirrors `rowii.anomaly.sweep._POOLED_ROW_LABEL`'s identical
non-collision argument for `"pooled"`)."""

_TOP_CANDIDATES_COLUMNS = ["window", "label", "score", "p_value", "rank"]
_MATCH_COLUMNS = [
    "t_utc_ns_a", "t_utc_ns_b", "dt_s", "label_a", "label_b", "p_value_a", "p_value_b",
]


def _top_k_rows(df: pd.DataFrame, top_k: int, label: object) -> pd.DataFrame:
    """The `min(top_k, len(df))` rows of *df* ranked by the module docstring's
    lexsort convention, tagged with *label* (a real per-label value, or
    `GLOBAL_LABEL` for the across-all-labels group) and 1-based ascending `rank`.
    `top_candidates` calls this once per label group and once more for the whole
    frame (the global group) -- the same "one call per label" shape `rowii.anomaly.
    sweep.run_sweep` uses around `scores_and_candidates` itself, just with the
    per-label split already done by the caller (`DataFrame.groupby`) rather than by
    a hand-rolled loop over label ids.
    """
    windows = df["window"].to_numpy()
    scores = df["score"].to_numpy()
    p_vals = df["p_value"].to_numpy()
    order = np.lexsort((windows, -scores, p_vals))[:top_k]
    return pd.DataFrame(
        {
            "window": windows[order],
            "label": label,
            "score": scores[order],
            "p_value": p_vals[order],
            "rank": np.arange(1, len(order) + 1, dtype=np.int64),
        }
    )


def top_candidates(scores: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Top-`top_k` candidates PER LABEL, plus one more top-`top_k` group ignoring
    label entirely (`label=GLOBAL_LABEL`) -- both ranked by the module docstring's
    lexsort convention. A label with fewer than `top_k` scored windows contributes
    all of them (`_top_k_rows`' own `min(top_k, len(df))`), matching `rowii.anomaly.
    sweep.scores_and_candidates`'s identical "fewer if a label has fewer than top_k"
    semantics.

    Args:
        scores: One row per scored window, columns `window, label, score, p_value`
            (`alarm`, if present, is ignored) -- typically `scripts/run_step2.py`'s
            persisted `scores.parquet`, loaded back with `label` as `str`, but any
            DataFrame with these four columns works (see `TestTopCandidates` in
            `tests/test_overlap.py` for the raw-label-dtype case).
        top_k: Candidates per label (and separately, per the global group) to keep.

    Returns:
        Columns `window, label, score, p_value, rank` (`rank` 1-based ascending
        within each label group AND within the global group -- ranks repeat across
        groups, they are not a single flat ranking over the whole result).
    """
    parts = [_top_k_rows(group, top_k, label) for label, group in scores.groupby("label")]
    parts.append(_top_k_rows(scores, top_k, GLOBAL_LABEL))
    return pd.concat(parts, ignore_index=True)[_TOP_CANDIDATES_COLUMNS]


def to_utc_ns(candidates: pd.DataFrame, t0_ns: int, window_ns: int) -> pd.DataFrame:
    """*candidates* (typically `top_candidates`' output) plus a `t_utc_ns` column:
    `t0_ns + window * window_ns`, int64 -- the SAME left-edge convention `rowii.
    signals.windows.WindowGrid.edges_ns`/`scripts/run_step2.py::_candidates_
    markdown` already use to turn a window index into an absolute UTC instant.

    *t0_ns*/*window_ns* must be the grid THIS candidate set's own combo was scored
    against (`rowii.pipeline.PreparedRun.grid`) -- module docstring: two combos of
    the same run generally do not share a grid, so mixing them here would silently
    misalign every downstream `match_by_time` call.

    Does not mutate *candidates*; returns a new DataFrame (its columns plus
    `t_utc_ns`).
    """
    out = candidates.copy()
    out["t_utc_ns"] = candidates["window"].to_numpy(dtype=np.int64) * np.int64(
        window_ns
    ) + np.int64(t0_ns)
    return out


def match_by_time(a: pd.DataFrame, b: pd.DataFrame, tol_s: float = 5.0) -> pd.DataFrame:
    """Greedy nearest-time 1:1 match between two candidate sets on `t_utc_ns`.

    Every cross pair `(a[i], b[j])` is considered, sorted by ascending `|dt|` (`dt =
    a[i].t_utc_ns - b[j].t_utc_ns`, converted to seconds); a pair is accepted, in
    that order, iff BOTH members are still unmatched and `|dt| <= tol_s` -- once an
    `a` row or a `b` row is matched it is never reconsidered (a genuine 1:1
    matching, not a nearest-neighbour assignment that could reuse a candidate on
    both sides). Because pairs are visited in ascending `|dt|` order, the first
    excess-tolerance pair encountered means every remaining pair also exceeds
    `tol_s`, so the scan stops there. An EXACT `|dt|` tie between competing pairs
    is itself deterministic: the sort is stable (`np.argsort(..., kind="stable")`)
    over the row-major `(a-row, b-row)` pair enumeration, so among tied pairs the
    one with the smaller `a` row index -- then the smaller `b` row index -- is
    accepted first, every run.

    Args:
        a: Candidate set A, at least columns `t_utc_ns` (int64), `label`,
            `p_value`. `window`/`score`/`rank`, if present, are ignored.
        b: Candidate set B, same column requirements as *a*.
        tol_s: Maximum absolute time difference, in seconds, for a pair to count as
            a match.

    Returns:
        One row per accepted match, columns `t_utc_ns_a, t_utc_ns_b, dt_s (= (a's
        t_utc_ns - b's t_utc_ns) / 1e9), label_a, label_b, p_value_a, p_value_b`.
        Empty (these columns, zero rows) if *a* or *b* is empty, or nothing matches
        within *tol_s*.
    """
    ta = a["t_utc_ns"].to_numpy(dtype=np.int64)
    tb = b["t_utc_ns"].to_numpy(dtype=np.int64)
    label_a = a["label"].to_numpy()
    label_b = b["label"].to_numpy()
    p_a = a["p_value"].to_numpy()
    p_b = b["p_value"].to_numpy()

    ia_mesh, ib_mesh = np.meshgrid(np.arange(ta.shape[0]), np.arange(tb.shape[0]), indexing="ij")
    ia = ia_mesh.ravel()
    ib = ib_mesh.ravel()
    dt_s = (ta[ia] - tb[ib]) / 1e9
    order = np.argsort(np.abs(dt_s), kind="stable")

    used_a = np.zeros(ta.shape[0], dtype=bool)
    used_b = np.zeros(tb.shape[0], dtype=bool)
    rows: list[dict[str, object]] = []
    for k in order:
        if abs(dt_s[k]) > tol_s:
            break  # ascending |dt| -- every later pair exceeds tol_s too
        i, j = int(ia[k]), int(ib[k])
        if used_a[i] or used_b[j]:
            continue
        used_a[i] = True
        used_b[j] = True
        rows.append(
            {
                "t_utc_ns_a": int(ta[i]),
                "t_utc_ns_b": int(tb[j]),
                "dt_s": float(dt_s[k]),
                "label_a": label_a[i],
                "label_b": label_b[j],
                "p_value_a": float(p_a[i]),
                "p_value_b": float(p_b[j]),
            }
        )
    return pd.DataFrame(rows, columns=_MATCH_COLUMNS)


def jaccard(n_a: int, n_b: int, n_matched: int) -> float:
    """`n_matched / (n_a + n_b - n_matched)` -- the Jaccard index of two candidate
    sets of size *n_a*/*n_b* sharing *n_matched* members (`len(match_by_time(...))`).
    `0.0`, not a `ZeroDivisionError`, when the denominator is 0 (both sets empty and
    nothing matched -- vacuously no overlap, not an error)."""
    denom = n_a + n_b - n_matched
    return n_matched / denom if denom > 0 else 0.0
