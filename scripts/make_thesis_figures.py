"""Thesis result figures (Chapter 3, Evaluation): five publication-style PDFs built
directly from the committed `results/` artifacts -- never a re-derived, invented or
partner number (the firewall). Every number plotted here is either printed in
`thesis/hsg-thesis/content/evaluation.tex`'s tables or prose, or read verbatim from a
committed artifact and named in the figure's own caption; this script only adds the
visual companion the chapter was missing.

One shared style helper (`_apply_style`) is used by all four figures: sans-serif,
small (~8-9pt) fonts, vector PDF output, no in-figure titles (the LaTeX caption carries
the title), and a grayscale-first encoding (bar shade/hatch, line style, marker shape)
with color used only as a sparing accent (era hue in F1/F2). Every axis carries its
unit or states that the quantity is a unitless fraction.

Figures
-------
F1 `f1_era_far.pdf`   -- E3 realized false-alarm rate under the three calibration
                         regimes (always-frozen / always-recalibrate / once-per-era),
                         per replayed session, log scale, nominal alpha line, sentinel-
                         fired markers under the axis.
                         Source: results/step2/once-calibrated/fusion/{fusion_regimes.csv,
                         fusion_trigger_log.csv, fusion.json}.
F2 `f2_sentinel.pdf`  -- E3 sentinel S1 (no_mode_fits day-rate) per session against its
                         fixed threshold, era-colored, near-miss annotated.
                         Source: results/step2/once-calibrated/fusion/{fusion_trigger_log.csv,
                         fusion.json}.
F3 `f3_scarcity.pdf`  -- MIMII clip-level AUC vs. fitting-budget fraction, one curve per
                         representation, markers at the five measured points.
                         Source: results/scarcity-detection/scarcity_detection.csv.
F4 `f4_latency.pdf`   -- Per-strike first-alarm latency distribution (physical-strike
                         level, alpha=0.05), standstill (ST) and pump (PU) sessions side
                         by side, detected/total counts annotated.
                         Source: results/pillar3-perstrike/latency.csv (row_type=detail).
F5 `f5_transfer.pdf`  -- E3 day-by-day transfer matrix: calibrated-on x evaluated-on
                         realized FAR (percent) for the three single-day turbine
                         rotations, era boundary drawn, own-day diagonal starred.
                         Sources: results/step2/cross-day/fusion-detected/
                         <src>__to__<dst>/knn-pooled/far_table.csv (pooled row) for the
                         six off-diagonal cells; results/step2/within-day/<run>/
                         fusion-detected/pooled-knn/far_table.csv (per-state rows
                         aggregated as sum(n_alarms)/sum(n_scored), the same aggregation
                         the sibling per-state tables carry as their own `pooled` row)
                         for the three own-day cells.

Usage
-----
    cd repos/rowii-monitor && .venv/bin/python scripts/make_thesis_figures.py \
        [--results-dir results] [--out results/thesis-figures]

The caller (not this script) copies the written PDFs into the thesis's graphics/
directory; this script only ever writes under `--out`.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import; headless-safe backend.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_DIR = DEFAULT_RESULTS_DIR / "thesis-figures"

# ---------------------------------------------------------------------------
# Shared style (one helper, used by all four figures)
# ---------------------------------------------------------------------------

# Era colors: three of the eight Okabe & Ito (2008) colorblind-safe hues that
# `scripts/candidate_kit.py`'s `_STATE_COLORS` does NOT already use for a plant
# state (that dict claims blue/vermillion/reddish-purple/yellow for
# turbine/pump/phase-shifter/transition) -- kept from the same accessible family
# without reusing a state's own color for an unrelated era axis.
ERA_COLOR: dict[str, str] = {"A": "#E69F00", "B": "#56B4E9", "C": "#009E73"}
ERA_HATCH: dict[str, str] = {"A": "", "B": "//", "C": "xx"}
"""Hatch pattern per era: the grayscale-safe primary encoding: era color is the
sparing accent on top of it, never the only cue."""


def _apply_style() -> None:
    """The one shared style helper: serif-free, ~8-9pt, vector-PDF-friendly, no
    figure titles (captions carry them in the LaTeX source)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.6,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#cccccc",
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,  # embed real (searchable) fonts, never Type-3 bitmaps
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# ---------------------------------------------------------------------------
# The eight replayed sessions (E3), in the SAME order and with the SAME era
# tags as Table~\ref{tab:rese3regimes} in content/evaluation.tex, so figure and
# table read as one artifact.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionMeta:
    run: str
    era: str
    label: str
    in_sample: bool = False


SESSIONS: tuple[SessionMeta, ...] = (
    SessionMeta("250526-tu", "A", "25 Jun\nTU"),
    SessionMeta("250526-pu-morning", "A", "25 Jun\nPU (morning)"),
    SessionMeta("270626-pu_ph_pu_ph_pu_ph-1", "A", "27 Jun\nPU+PS"),
    SessionMeta("290626-tu", "B", "29 Jun\nTU"),
    SessionMeta("290626-pu", "B", "29 Jun\nPU"),
    SessionMeta("300626-tu", "B", "30 Jun\nTU"),
    SessionMeta("300626-pu", "B", "30 Jun\nPU"),
    SessionMeta("010726-tu_ph_tu", "B", "1 Jul\nTU+PS", in_sample=True),
    SessionMeta("010726-pu", "B", "1 Jul\nPU", in_sample=True),
    SessionMeta("080726-pu_strikes", "C", "8 Jul\nPU, strikes"),
)
_ERA_GAP = 0.6
"""Extra x-axis spacing inserted between two adjacent sessions of different eras
(era-grouping for F1/F2), on top of the unit spacing within an era."""


def session_x_positions(sessions: tuple[SessionMeta, ...] = SESSIONS) -> np.ndarray:
    """x position per session: unit steps within an era, `_ERA_GAP` extra between
    two consecutive sessions of different eras. Pure function, unit-tested."""
    xs = []
    gap = 0.0
    prev_era: str | None = None
    for i, s in enumerate(sessions):
        if prev_era is not None and s.era != prev_era:
            gap += _ERA_GAP
        xs.append(i + gap)
        prev_era = s.era
    return np.asarray(xs, dtype=float)


def session_tick_label(s: SessionMeta) -> str:
    return s.label + ("*" if s.in_sample else "")


# ---------------------------------------------------------------------------
# F1: E3 realized FAR under the three calibration regimes
# ---------------------------------------------------------------------------

_REGIME_FIELD = {
    "frozen": "always_frozen_far",
    "recalibrate": "always_recalibrate_far",
    "once": "once_triggered_far",
}
_REGIME_LABEL = {
    "frozen": "always-frozen",
    "recalibrate": "always-recalibrate",
    "once": "once + triggered",
}
_REGIME_STYLE: dict[str, dict[str, str]] = {
    "frozen": {"color": "#dcdcdc", "edgecolor": "#262626", "hatch": ""},
    "recalibrate": {"color": "#8c8c8c", "edgecolor": "#262626", "hatch": "///"},
    "once": {"color": "#1a1a1a", "edgecolor": "#1a1a1a", "hatch": ""},
}


def load_fusion_regimes(results_dir: Path) -> pd.DataFrame:
    """One row per (day, run); index is `run`. Source: `fusion_regimes.csv`."""
    path = results_dir / "step2" / "once-calibrated" / "fusion" / "fusion_regimes.csv"
    df = pd.read_csv(path)
    return df.set_index("run")


def load_fusion_trigger_log(results_dir: Path) -> pd.DataFrame:
    """One row per (day, run); index is `run`. Source: `fusion_trigger_log.csv`."""
    path = results_dir / "step2" / "once-calibrated" / "fusion" / "fusion_trigger_log.csv"
    df = pd.read_csv(path)
    return df.set_index("run")


def load_fusion_meta(results_dir: Path) -> dict[str, Any]:
    path = results_dir / "step2" / "once-calibrated" / "fusion" / "fusion.json"
    meta: dict[str, Any] = json.loads(path.read_text())
    return meta


def make_f1_era_far(results_dir: Path, out_dir: Path) -> Path:
    regimes = load_fusion_regimes(results_dir)
    trigger = load_fusion_trigger_log(results_dir)
    meta = load_fusion_meta(results_dir)
    alpha = float(meta["alpha"])
    assert 0.0 < alpha < 1.0, f"unexpected nominal alpha: {alpha}"

    xpos = session_x_positions()
    fig, ax = plt.subplots(figsize=(6.3, 3.7))
    bw = 0.26

    fired_x: list[float] = []
    for i, s in enumerate(SESSIONS):
        if s.run in regimes.index:
            row = regimes.loc[s.run]
            for key, off in (("frozen", -bw), ("recalibrate", 0.0), ("once", bw)):
                val = float(row[_REGIME_FIELD[key]])
                style = _REGIME_STYLE[key]
                ax.bar(
                    xpos[i] + off,
                    val,
                    width=bw,
                    label=_REGIME_LABEL[key] if i == 0 else None,
                    zorder=3,
                    linewidth=0.7,
                    color=style["color"],
                    edgecolor=style["edgecolor"],
                    hatch=style["hatch"],
                )
        else:
            ax.text(
                xpos[i],
                0.10,
                "no process\ndata (n/a)",
                ha="center",
                va="center",
                fontsize=6.3,
                style="italic",
                color="#555555",
            )
        # A triangle marks "a label-free sentinel fired and triggered the
        # once-per-era recalibration" -- read off the decision column (any of
        # s1/s2/s3), not s1_fired alone: since the s3 alarm-rate watchdog,
        # 30 June turbine recalibrates with s1 AND s2 quiet.
        if s.run in trigger.index and str(trigger.loc[s.run, "decision"]) == "recalibrate":
            fired_x.append(float(xpos[i]))

    ax.set_yscale("log")
    ax.set_ylim(0.01, 1.6)
    ax.axhline(alpha, color="black", linewidth=0.9, linestyle="--", zorder=2)
    ax.text(
        xpos[-1] + bw + 0.55,
        alpha,
        f"α = {alpha:g}",
        va="center",
        ha="left",
        fontsize=7.5,
        zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )

    # Sentinel-fired row: a fixed POINT offset below the axis line (not an axes-
    # fraction guess), so it never collides with tick labels regardless of how many
    # lines a label wraps to; `ax.tick_params(pad=...)` below reserves the gap.
    if fired_x:
        marker_trans = offset_copy(ax.get_xaxis_transform(), fig=fig, x=0, y=-7, units="points")
        ax.plot(
            fired_x,
            [0.0] * len(fired_x),
            marker="^",
            linestyle="none",
            color="black",
            markersize=4.5,
            transform=marker_trans,
            clip_on=False,
            zorder=5,
            label="sentinel fired",
        )

    ax.set_xlim(xpos[0] - 0.6, xpos[-1] + 2.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels([session_tick_label(s) for s in SESSIONS])
    ax.tick_params(axis="x", pad=15, labelsize=7.0)
    for tick_label, s in zip(ax.get_xticklabels(), SESSIONS, strict=True):
        tick_label.set_color(ERA_COLOR[s.era])
    ax.set_ylabel("realized false-alarm rate\n(fraction of scored windows, log scale)")
    ax.grid(True, axis="y", which="major", alpha=0.55, zorder=0)
    ax.grid(True, axis="y", which="minor", alpha=0.18, zorder=0)

    # Era separators (vertical) between blocks; era letter as a sparing color accent
    # placed just above the axes (blended transform: x in data, y in axes fraction).
    prev_era: str | None = None
    for i, s in enumerate(SESSIONS):
        if prev_era is not None and s.era != prev_era:
            sep_x = (xpos[i] + xpos[i - 1]) / 2
            ax.axvline(sep_x, color="#999999", linewidth=0.6, linestyle=":", zorder=1)
        prev_era = s.era
    for era in ("A", "B", "C"):
        block = [i for i, s in enumerate(SESSIONS) if s.era == era]
        cx = float(xpos[block].mean())
        ax.text(
            cx,
            1.02,
            f"era {era}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=ERA_COLOR[era],
            fontweight="bold",
            clip_on=False,
        )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        fontsize=7.2,
        frameon=False,
    )

    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.90))
    out_path = out_dir / "f1_era_far.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# F2: sentinel S1 (no_mode_fits day-rate) vs. its fixed threshold
# ---------------------------------------------------------------------------


def make_f2_sentinel(results_dir: Path, out_dir: Path) -> Path:
    trigger = load_fusion_trigger_log(results_dir)
    meta = load_fusion_meta(results_dir)
    threshold = float(meta["s1"]["threshold"])
    assert 0.0 < threshold < 1.0, f"unexpected sentinel threshold: {threshold}"

    xpos = session_x_positions()
    fig, ax = plt.subplots(figsize=(6.3, 3.3))

    for i, s in enumerate(SESSIONS):
        assert s.run in trigger.index, f"session {s.run} missing from fusion_trigger_log.csv"
        rate = float(trigger.loc[s.run, "s1_rate"])
        ax.bar(
            xpos[i],
            rate,
            width=0.62,
            color=ERA_COLOR[s.era],
            alpha=0.55,
            edgecolor="#1a1a1a",
            linewidth=0.7,
            hatch=ERA_HATCH[s.era],
            zorder=3,
        )
        ax.text(xpos[i], rate + 0.014, f"{rate:.3f}", ha="center", va="bottom", fontsize=6.3)

    ax.axhline(threshold, color="black", linestyle="--", linewidth=0.9, zorder=2)
    ax.text(
        xpos[-1] + 0.75,
        threshold + 0.028,
        f"threshold = {threshold:.4f}",
        va="center",
        ha="left",
        fontsize=7.2,
        zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )

    # Near-miss annotation: 29 June turbine, 0.0767 against 0.0805 (Section 3.5.3 prose).
    nm_run = "290626-tu"
    nm_i = [i for i, s in enumerate(SESSIONS) if s.run == nm_run][0]
    nm_val = float(trigger.loc[nm_run, "s1_rate"])
    ax.annotate(
        "near miss",
        xy=(xpos[nm_i], nm_val),
        xytext=(xpos[nm_i] + 0.9, nm_val - 0.05),
        fontsize=6.6,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.7},
        zorder=6,
    )

    ax.set_xlim(xpos[0] - 0.6, xpos[-1] + 2.2)
    ax.set_ylim(0.0, 0.78)
    ax.set_xticks(xpos)
    ax.set_xticklabels([session_tick_label(s) for s in SESSIONS])
    ax.tick_params(axis="x", pad=20, labelsize=7.0)
    for tick_label, s in zip(ax.get_xticklabels(), SESSIONS, strict=True):
        tick_label.set_color(ERA_COLOR[s.era])

    # Sentinel-only day annotation: 27 June has no process export, so no FAR regime
    # exists for it (Figure F1); this sentinel rate is the only signal evaluated there.
    # Fixed POINT offset below the axis line, in the pad reserved above -- robust to
    # label line count, unlike a raw axes-fraction guess.
    so_run = "270626-pu_ph_pu_ph_pu_ph-1"
    so_i = [i for i, s in enumerate(SESSIONS) if s.run == so_run][0]
    so_trans = offset_copy(ax.get_xaxis_transform(), fig=fig, x=0, y=-40, units="points")
    ax.text(
        xpos[so_i],
        0.0,
        "sentinel-only day",
        transform=so_trans,
        ha="center",
        va="top",
        fontsize=6.0,
        style="italic",
        color="#555555",
        clip_on=False,
    )
    ax.set_ylabel("sentinel S1 rate\n(share of windows fitting no bank mode)")
    ax.grid(True, axis="y", alpha=0.4, zorder=0)

    handles = [
        Rectangle(
            (0, 0), 1, 1, facecolor=ERA_COLOR[e], alpha=0.55,
            edgecolor="#1a1a1a", hatch=ERA_HATCH[e],
        )
        for e in ("A", "B", "C")
    ]
    fig.legend(
        handles,
        ["era A", "era B", "era C"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        fontsize=7.2,
        frameon=False,
    )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.90))
    out_path = out_dir / "f2_sentinel.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# F3: MIMII clip-AUC vs. fitting-budget fraction, per representation
# ---------------------------------------------------------------------------

_SCARCITY_REPR_ORDER = ("beats", "tfc", "student", "logmel", "handcrafted")
_SCARCITY_REPR_LABEL = {
    "beats": "Frozen general-audio encoder (BEATs)",
    "tfc": "Industrially pretrained (TF-C)",
    "student": "Distilled student",
    "logmel": "Log-Mel baseline",
    "handcrafted": "Handcrafted features",
}
_SCARCITY_REPR_STYLE: dict[str, dict[str, object]] = {
    "beats": {"marker": "o", "linestyle": "-", "color": "#1a1a1a"},
    "tfc": {"marker": "s", "linestyle": "--", "color": "#3d3d3d"},
    "student": {"marker": "^", "linestyle": "-.", "color": "#5f5f5f"},
    "logmel": {"marker": "D", "linestyle": ":", "color": "#8a8a8a"},
    "handcrafted": {"marker": "x", "linestyle": "-", "color": "#a8a8a8"},
}
_SCARCITY_FRACTIONS = (0.05, 0.1, 0.25, 0.5, 1.0)


def load_scarcity_curve(results_dir: Path) -> pd.DataFrame:
    """Non-degenerate rows only. Source: `scarcity_detection.csv` (a leading `#`
    comment line documents the pauc_clip formula and is skipped by `comment="#"`)."""
    path = results_dir / "scarcity-detection" / "scarcity_detection.csv"
    df = pd.read_csv(path, comment="#")
    df = df[~df["degenerate"].astype(bool)].copy()
    assert set(df["representation"].unique()) >= set(_SCARCITY_REPR_ORDER), (
        "scarcity_detection.csv is missing an expected representation"
    )
    return df


def scarcity_curve_means(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Per representation: mean `auc_clip` over machine ids and seeds, indexed by
    `fraction` -- the same aggregation `scripts/scarcity_detection.py::_plot_curve`
    already uses for its own two-panel PNG (mean over machine_id x seed cells)."""
    out: dict[str, pd.Series] = {}
    for rep in _SCARCITY_REPR_ORDER:
        sub = df[df["representation"] == rep]
        out[rep] = sub.groupby("fraction")["auc_clip"].mean().sort_index()
    return out


def make_f3_scarcity(results_dir: Path, out_dir: Path) -> Path:
    df = load_scarcity_curve(results_dir)
    means = scarcity_curve_means(df)

    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    for rep in _SCARCITY_REPR_ORDER:
        agg = means[rep]
        assert list(agg.index) == list(_SCARCITY_FRACTIONS), (
            f"{rep}: unexpected fraction grid {list(agg.index)}"
        )
        style = _SCARCITY_REPR_STYLE[rep]
        ax.plot(
            agg.index,
            agg.to_numpy(),
            label=_SCARCITY_REPR_LABEL[rep],
            markersize=4.5,
            linewidth=1.15,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
        )
    ax.set_xscale("log")
    ax.set_xticks(list(_SCARCITY_FRACTIONS))
    ax.set_xticklabels(["5%", "10%", "25%", "50%", "100%"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("fitting budget: fraction of the available normal-clip pool (log scale)")
    ax.set_ylabel("clip-level AUC\n(0.5 = chance, 1.0 = perfect)")
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, alpha=0.35, zorder=0)
    ax.legend(loc="lower right", fontsize=7.0)

    fig.tight_layout()
    out_path = out_dir / "f3_scarcity.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# F4: per-strike first-alarm latency distribution, ST and PU
# ---------------------------------------------------------------------------

_LATENCY_REPR_ORDER = ("audio-beats", "audio-student", "audio", "fusion", "vibration")
_LATENCY_REPR_LABEL = {
    "audio-beats": "BEATs",
    "audio-student": "Student",
    "audio": "HC audio",
    "fusion": "HC fusion",
    "vibration": "Vibration",
}
_LATENCY_SESSION_LABEL = {"st": "standstill (ST)", "pu": "pump (PU)"}
_LATENCY_YLIM = (0.008, 8.0)
_LATENCY_LABEL_Y = 6.0


def load_latency_detail(results_dir: Path) -> pd.DataFrame:
    """Detail rows only, physical-strike level (the strike-count-matched
    granularity, alpha=0.05 by construction -- `latency.csv` has no other alpha).
    Source: `results/pillar3-perstrike/latency.csv`."""
    path = results_dir / "pillar3-perstrike" / "latency.csv"
    df = pd.read_csv(path)
    df = df[(df["row_type"] == "detail") & (df["level"] == "physical_strike")].copy()
    assert set(df["session"].unique()) == {"st", "pu"}
    assert set(df["representation"].unique()) >= set(_LATENCY_REPR_ORDER)
    return df


def latency_group_counts(
    df: pd.DataFrame, session: str, representation: str
) -> tuple[np.ndarray, int, int]:
    """`(detected_latencies_s, n_detected, n_total)` for one (session, representation)
    cell. Pure function, unit-tested against a tiny synthetic frame."""
    sub = df[(df["session"] == session) & (df["representation"] == representation)]
    n_total = len(sub)
    detected = sub[~sub["missed"].astype(bool)]["latency_s"].astype(float).to_numpy()
    return detected, len(detected), n_total


def make_f4_latency(results_dir: Path, out_dir: Path) -> Path:
    df = load_latency_detail(results_dir)
    rng = np.random.default_rng(7)

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.4), sharey=True)
    for ax, session in zip(axes, ("st", "pu"), strict=True):
        ax.text(
            0.04,
            0.965,
            _LATENCY_SESSION_LABEL[session],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
        )
        for i, rep in enumerate(_LATENCY_REPR_ORDER):
            detected, n_det, n_total = latency_group_counts(df, session, rep)
            if n_det:
                jitter = rng.uniform(-0.17, 0.17, size=n_det)
                ax.scatter(
                    np.full(n_det, i) + jitter,
                    detected,
                    s=7,
                    color="#2b2b2b",
                    alpha=0.55,
                    linewidths=0,
                    zorder=3,
                )
                med = float(np.median(detected))
                ax.plot([i - 0.24, i + 0.24], [med, med], color="black", linewidth=1.5, zorder=4)
            ax.text(
                i,
                _LATENCY_LABEL_Y,
                f"{n_det}/{n_total}",
                ha="center",
                va="bottom",
                fontsize=6.2,
            )
        ax.set_xlim(-0.6, len(_LATENCY_REPR_ORDER) - 0.4)
        ax.set_xticks(range(len(_LATENCY_REPR_ORDER)))
        ax.set_xticklabels([_LATENCY_REPR_LABEL[r] for r in _LATENCY_REPR_ORDER], fontsize=7.0)
        ax.set_yscale("log")
        ax.set_ylim(*_LATENCY_YLIM)
        ax.grid(True, axis="y", which="major", alpha=0.45, zorder=0)
        ax.grid(True, axis="y", which="minor", alpha=0.15, zorder=0)

    axes[0].set_ylabel("first-alarm latency (s, log scale)")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    out_path = out_dir / "f4_latency.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# F5: day-by-day transfer matrix (calibrated-on x evaluated-on realized FAR)
# ---------------------------------------------------------------------------

# The three single-day turbine rotations of the cross-day grid, in chronological
# order and with the SAME era tags the E3 table uses. The grid is defined over
# ordered pairs, so only the six off-diagonal cells are transfer measurements; the
# diagonal is the day's own within-day split and is starred, never read as transfer.
TRANSFER_DAYS: tuple[SessionMeta, ...] = (
    SessionMeta("250526-tu", "A", "25 Jun\nTU"),
    SessionMeta("290626-tu", "B", "29 Jun\nTU"),
    SessionMeta("300626-tu", "B", "30 Jun\nTU"),
    SessionMeta("010726-tu_ph_tu", "B", "1 Jul\nTU+PS"),
)
_TRANSFER_ALPHA = 0.05
"""Nominal budget of every cell; asserted against each artifact's own
`nominal_alpha` column rather than assumed."""


def load_transfer_cell(results_dir: Path, src: str, dst: str) -> tuple[float, int]:
    """Off-diagonal cell `(realized_far, n_scored)`: day `src`'s fitted detector
    applied unchanged to day `dst`. Source: the rotation's `far_table.csv` pooled row."""
    path = (
        results_dir
        / "step2"
        / "cross-day"
        / "fusion-detected"
        / f"{src}__to__{dst}"
        / "knn-pooled"
        / "far_table.csv"
    )
    df = pd.read_csv(path)
    row = df[df["label"].astype(str) == "pooled"]
    assert len(row) == 1, f"{path}: expected exactly one pooled row, found {len(row)}"
    rec = row.iloc[0]
    assert float(rec["nominal_alpha"]) == _TRANSFER_ALPHA, f"{path}: unexpected alpha"
    return float(rec["realized_far"]), int(rec["n_scored"])


def load_own_day_cell(results_dir: Path, run: str) -> tuple[float, int]:
    """Diagonal cell `(realized_far, n_scored)` for `run`: the day's own within-day
    pooled-kNN run, aggregated over its per-state rows as sum(alarms)/sum(scored).
    This is the same aggregation the sibling per-state artifacts write out as their
    own `pooled` row, so no new estimator is introduced -- only a sum of two committed
    integer columns. The cell is calibrated on the same day it is scored on (disjoint
    splits) and is therefore starred in the figure, never counted as a transfer."""
    path = (
        results_dir
        / "step2"
        / "within-day"
        / run
        / "fusion-detected"
        / "pooled-knn"
        / "far_table.csv"
    )
    df = pd.read_csv(path)
    per_state = df[df["label"].astype(str) != "pooled"]
    assert not per_state.empty, f"{path}: no per-state rows"
    assert (per_state["nominal_alpha"].astype(float) == _TRANSFER_ALPHA).all(), (
        f"{path}: unexpected alpha"
    )
    n_scored = int(per_state["n_scored"].sum())
    return float(per_state["n_alarms"].sum()) / n_scored, n_scored


def transfer_matrix(results_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """`(far, n_scored)` as (calibrated-on x evaluated-on) matrices over
    `TRANSFER_DAYS`. Pure aside from the reads; unit-tested on a synthetic tree."""
    n = len(TRANSFER_DAYS)
    far = np.zeros((n, n), dtype=float)
    scored = np.zeros((n, n), dtype=int)
    for i, src in enumerate(TRANSFER_DAYS):
        for j, dst in enumerate(TRANSFER_DAYS):
            if i == j:
                far[i, j], scored[i, j] = load_own_day_cell(results_dir, src.run)
            else:
                far[i, j], scored[i, j] = load_transfer_cell(results_dir, src.run, dst.run)
    return far, scored


def make_f5_transfer(results_dir: Path, out_dir: Path) -> Path:
    far, _scored = transfer_matrix(results_dir)
    n = len(TRANSFER_DAYS)

    fig, ax = plt.subplots(figsize=(5.1, 3.5))
    im = ax.imshow(far * 100.0, cmap="Greys", vmin=0.0, vmax=70.0, origin="upper")

    for i in range(n):
        for j in range(n):
            value = far[i, j] * 100.0
            own_day = i == j
            if own_day:
                # Hatch the diagonal so the "not a transfer measurement" reading
                # survives a grayscale print, where the star alone could be missed.
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        facecolor="none", edgecolor="#ffffff",
                        hatch="////", linewidth=0.0, zorder=2,
                    )
                )
            ax.text(
                j, i,
                f"{value:.1f}" + ("*" if own_day else ""),
                ha="center", va="center", zorder=3,
                fontsize=10.5, fontweight="bold",
                color="white" if value > 38.0 else "#1a1a1a",
            )

    # Era boundary: day 0 is era A, days 1-2 are era B. A cell that crosses this
    # line is a cross-era transfer; a cell inside a block is within-era.
    boundary = [i for i in range(1, n) if TRANSFER_DAYS[i].era != TRANSFER_DAYS[i - 1].era]
    for b in boundary:
        ax.axhline(b - 0.5, color="#b3282d", linewidth=1.8, zorder=4)
        ax.axvline(b - 0.5, color="#b3282d", linewidth=1.8, zorder=4)
    if boundary:
        ax.text(
            boundary[0] - 0.5, -0.62, "instrumentation-era boundary",
            ha="center", va="bottom", fontsize=6.5, color="#b3282d", clip_on=False,
        )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(
        [f"{s.label}\n(era {s.era})" for s in TRANSFER_DAYS], fontsize=7.2
    )
    ax.set_yticklabels(
        [f"{s.label} ({s.era})".replace("\n", " ") for s in TRANSFER_DAYS], fontsize=7.2
    )
    for tick_label, s in zip(ax.get_xticklabels(), TRANSFER_DAYS, strict=True):
        tick_label.set_color(ERA_COLOR[s.era])
    for tick_label, s in zip(ax.get_yticklabels(), TRANSFER_DAYS, strict=True):
        tick_label.set_color(ERA_COLOR[s.era])
    ax.set_xlabel("evaluated on")
    ax.set_ylabel("calibrated on")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0.5, -0.30, "* own-day cell: calibrated and scored on the same day, not a transfer",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=6.5, style="italic", color="#555555", clip_on=False,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("realized false-alarm rate (% of scored windows)", fontsize=7.5)
    alpha_pct = _TRANSFER_ALPHA * 100.0
    # The nominal budget rides on the colorbar as its own labelled tick, so every
    # cell can be read against it without a second annotation.
    cbar.set_ticks([0.0, alpha_pct, 20.0, 40.0, 60.0])
    cbar.set_ticklabels(["0", f"{alpha_pct:.0f} (α)", "20", "40", "60"])
    cbar.ax.tick_params(labelsize=7.0)
    cbar.ax.axhline(alpha_pct, color="#b3282d", linewidth=1.2)
    cbar.ax.get_yticklabels()[1].set_color("#b3282d")
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    out_path = out_dir / "f5_transfer.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def make_all(results_dir: Path, out_dir: Path) -> list[Path]:
    _apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        make_f1_era_far(results_dir, out_dir),
        make_f2_sentinel(results_dir, out_dir),
        make_f3_scarcity(results_dir, out_dir),
        make_f4_latency(results_dir, out_dir),
        make_f5_transfer(results_dir, out_dir),
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args(argv)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = _parse_args()
    paths = make_all(args.results_dir, args.out)
    for p in paths:
        size_kb = p.stat().st_size / 1024
        print(f"wrote {p} ({size_kb:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
