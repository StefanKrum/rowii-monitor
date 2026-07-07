"""Markdown + timeline-plot run report for one detection variant.

`write_report` is the terminal step of the Step-1 pipeline for a single (run, variant)
combination (spec §8 deliverables): it renders the `EvalResult` produced by
`rowii.eval.metrics.evaluate` into a human-readable `report.md`, a 3-panel
`timeline.png` (power curve / GT states / predicted mapped states, all in hours since
the grid start), and re-exports the machine-readable `segments.csv` /
`frame_labels.parquet` artifacts spec §8 lists alongside it. report.md now leads with a
state-level (mode) metrics block using majority cluster->state mapping, with the
original strict 1:1 Hungarian view retained below it as a secondary reference.

Note on the optional `gt` parameter: `EvalResult` (by design -- see
`rowii.eval.metrics`) carries only aggregate metrics (a GT-state x predicted-state
count crosstab, not a per-window array), because a crosstab cannot be inverted back
into an ordered per-window sequence. The timeline's GT-states panel therefore needs
the original `rowii.scada.labels.gt_labels` output directly; `write_report` accepts it
as an optional, keyword-only parameter (`gt: pd.DataFrame | None = None`), mirroring
the already-established, documented precedent for extending a plan-given signature
with an additive, default-preserving parameter (`rowii.state.detect.run_detection`'s
`k`). Every call using the base signature (`out_dir, run, variant, det, ev, scada`)
behaves identically whether or not this module ever added `gt`; passing it only
enables the GT panel to show real per-window data instead of a placeholder.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- must precede pyplot import; headless/CI-safe backend.

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from rowii.eval.metrics import EvalResult, load_alignment  # noqa: E402
from rowii.state.detect import DetectionResult  # noqa: E402

_STATE_COLORS = {
    "standstill": "tab:gray",
    "turbine": "tab:blue",
    "pump": "tab:orange",
    "transition": "tab:green",
    "unknown": "tab:red",
}
_FALLBACK_COLOR = "tab:purple"
_NO_GT_COLOR = "0.9"


def _state_color(state: str) -> str:
    return _STATE_COLORS.get(state, _FALLBACK_COLOR)


def _confusion_to_markdown(confusion: pd.DataFrame) -> str:
    header = "| GT \\ predicted | " + " | ".join(str(c) for c in confusion.columns) + " |"
    separator = "|---" * (len(confusion.columns) + 1) + "|"
    rows = [
        "| " + str(gt_state) + " | " + " | ".join(str(v) for v in row) + " |"
        for gt_state, row in confusion.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _mapping_to_markdown(mapping: dict[int, str]) -> str:
    header = "| cluster | state |"
    separator = "|---|---|"
    rows = [f"| {cluster_id} | {state} |" for cluster_id, state in sorted(mapping.items())]
    return "\n".join([header, separator, *rows])


def _state_mapping_to_markdown(mapping: dict[int, str]) -> str:
    header = "| cluster | state |"
    separator = "|---|---|"
    rows = [f"| {cluster_id} | {state} |" for cluster_id, state in sorted(mapping.items())]
    return "\n".join([header, separator, *rows])


def _load_alignment_crosstab_to_markdown(crosstab: pd.DataFrame) -> str:
    header = "| cluster \\ load_bin | " + " | ".join(str(c) for c in crosstab.columns) + " |"
    separator = "|---" * (len(crosstab.columns) + 1) + "|"
    rows = [
        "| " + str(cluster_id) + " | " + " | ".join(str(v) for v in row) + " |"
        for cluster_id, row in crosstab.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _load_alignment_section(det: DetectionResult, gt: pd.DataFrame | None) -> list[str]:
    """"Do sub-clusters track load levels?" section (Task 13b item 2).

    Restricted to the run's turbine (or pump-fallback) windows via
    `rowii.eval.metrics.load_alignment` -- `None` when `gt` was not supplied to
    `write_report`, or when `load_alignment` itself finds fewer than 2 distinct
    load bins to align clusters against (see its own docstring).
    """
    lines = [
        "## Do sub-clusters track load levels?",
        "",
        "Cross-tabulates predicted cluster id against SCADA-derived `load_bin` on "
        "this run's turbine (or pump, if no turbine windows exist) eval windows only "
        "-- a high alignment means the detector's sub-clusters (the ones "
        "`state_ari`/`state_macro_f1` above credit as \"still turbine\") correspond "
        "to genuine load-level structure, not noise.",
        "",
    ]
    alignment = load_alignment(det.frame_labels, gt) if gt is not None else None
    if alignment is None:
        lines.append("n/a (no GT provided, or fewer than 2 distinct load bins in this run).")
    else:
        lines.append(f"ARI(load_bin, cluster) = {alignment.attrs['ari']:.4f}")
        lines.append("")
        lines.append(_load_alignment_crosstab_to_markdown(alignment))
    lines.append("")
    return lines


def _report_markdown(
    run: str,
    variant: str,
    det: DetectionResult,
    ev: EvalResult,
    n_windows: int,
    gt: pd.DataFrame | None = None,
) -> str:
    n_dropped = n_windows - ev.n_eval_windows
    boundary_str = (
        f"{ev.boundary_median_abs_s:.4f}" if ev.boundary_median_abs_s is not None else "None"
    )
    lines = [
        f"# Run report: {run} / {variant}",
        "",
        "## State-level (mode) metrics — primary",
        "",
        "Clusters are mapped independently to their majority GT state (no 1:1 "
        "restriction), so legitimate load sub-clusters within one operating mode "
        "(e.g. two turbine-phase clusters at different load levels) both correctly "
        "collapse onto that mode instead of being penalized as a mismatch.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| state accuracy | {ev.state_accuracy:.4f} |",
        f"| state macro-F1 | {ev.state_macro_f1:.4f} |",
        f"| state ARI | {ev.state_ari:.4f} |",
        f"| n_eval_windows | {ev.n_eval_windows} |",
        "",
        "### Cluster -> state mapping (majority)",
        "",
        _state_mapping_to_markdown(ev.state_mapping),
        "",
        "## Strict (1:1 Hungarian) metrics — secondary",
        "",
        "| metric | value |",
        "|---|---|",
        f"| ARI | {ev.ari:.4f} |",
        f"| macro-F1 | {ev.macro_f1:.4f} |",
        f"| boundary median \\|Δt\\| (s) | {boundary_str} |",
        f"| n_eval_windows | {ev.n_eval_windows} |",
        f"| k (clusters) | {det.k} |",
        f"| unknown/dropped windows | {n_dropped} out of {n_windows} |",
        "",
        "### Cluster -> state mapping (Hungarian)",
        "",
        _mapping_to_markdown(ev.mapping),
        "",
        "## Confusion matrix (rows = GT state, cols = mapped predicted state)",
        "",
        _confusion_to_markdown(ev.confusion),
        "",
    ]
    lines.extend(_load_alignment_section(det, gt))
    return "\n".join(lines)


def _plot_state_panel(ax: Axes, hours: np.ndarray, states: list[str]) -> None:
    """Render *states* (one per window) as coloured axvspan regions over *hours*,
    with a legend listing every distinct state name that appears."""
    if len(states) == 0:
        return
    window_width = hours[1] - hours[0] if len(hours) > 1 else 1.0
    start = 0
    current = states[0]
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != current:
            ax.axvspan(
                hours[start], hours[i - 1] + window_width, color=_state_color(current), alpha=0.8
            )
            if i < len(states):
                start = i
                current = states[i]
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    legend_states = sorted(set(states))
    handles = [Rectangle((0, 0), 1, 1, color=_state_color(s), alpha=0.8) for s in legend_states]
    ax.legend(handles, legend_states, loc="upper right", fontsize="small", ncol=len(legend_states))


def _window_s_from_segments(det: DetectionResult) -> float:
    """Seconds per window, derived from `det.segments`' UTC boundaries.

    `to_segments` tiles the grid contiguously with no gaps (each row's `end_utc`
    equals the next row's `start_utc`), so the total wall-clock span divided by the
    window count is exact -- this is the only source of window duration available to
    `write_report` (neither `scada` nor `EvalResult` carries a `WindowGrid` or
    timestamps). Falls back to 1.0s (spec §5's Step-1 default) when there are no
    windows at all or no segment rows (nothing to derive from).
    """
    n_windows = len(det.frame_labels)
    if n_windows == 0 or len(det.segments) == 0:
        return 1.0
    span_s = float(
        (det.segments["end_utc"].iloc[-1] - det.segments["start_utc"].iloc[0]).total_seconds()
    )
    return span_s / n_windows


def _write_timeline_png(
    out_path: Path,
    det: DetectionResult,
    ev: EvalResult,
    scada: pd.DataFrame,
    gt: pd.DataFrame | None,
) -> None:
    n_windows = len(det.frame_labels)
    window_s = _window_s_from_segments(det)
    hours = np.arange(n_windows) * window_s / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(hours, scada["power"].to_numpy(), color="black", linewidth=0.8)
    axes[0].set_ylabel("power")
    axes[0].set_title("SCADA power")

    axes[1].set_title("GT states")
    if gt is not None:
        _plot_state_panel(axes[1], hours, list(gt["state"]))
    else:
        left = hours[0] if n_windows else 0
        right = hours[-1] if n_windows else 1
        axes[1].axvspan(left, right, color=_NO_GT_COLOR)
        axes[1].set_yticks([])
        axes[1].text(
            0.5,
            0.5,
            "no GT provided",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize="small",
            color="0.4",
        )

    predicted_states = [ev.state_mapping.get(int(c), "unknown") for c in det.frame_labels]
    _plot_state_panel(axes[2], hours, predicted_states)
    axes[2].set_title("Predicted (mapped) states")
    axes[2].set_xlabel("hours since grid start")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_report(
    out_dir: Path,
    run: str,
    variant: str,
    det: DetectionResult,
    ev: EvalResult,
    scada: pd.DataFrame,
    *,
    gt: pd.DataFrame | None = None,
) -> None:
    """Write `report.md`, `timeline.png`, `segments.csv`, `frame_labels.parquet` to *out_dir*.

    Args:
        out_dir: Destination directory; created (including parents) if missing.
        run: Recording identifier (e.g. `"tu"`, `"pu"`) -- report header only.
        variant: Feature/clusterer variant identifier -- report header only.
        det: The `DetectionResult` this report describes.
        ev: The `EvalResult` (`rowii.eval.metrics.evaluate` output) for `det` vs GT.
        scada: `rowii.scada.labels.load_scada_window_means` output, same window grid
            as `det`/`ev` -- used for the timeline's power-curve panel.
        gt: Optional `rowii.scada.labels.gt_labels` output (see module docstring for
            why this is needed and why it is not derivable from `ev` alone). When
            `None` (the default -- matching the exact base signature), the timeline's
            GT-states panel renders an explicit "no GT provided" placeholder instead
            of per-window data, and report.md's "Do sub-clusters track load levels?"
            section (Task 13b item 2) reports "n/a" instead of a real crosstab/ARI
            (both need `gt.load_bin`, which is not otherwise available here).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    n_windows = len(det.frame_labels)
    (out_dir / "report.md").write_text(_report_markdown(run, variant, det, ev, n_windows, gt))

    det.segments.to_csv(out_dir / "segments.csv", index=False)

    mapped_states = [ev.mapping.get(int(c), "unknown") for c in det.frame_labels]
    frame_df = pd.DataFrame(
        {
            "window": np.arange(n_windows, dtype=np.int64),
            "cluster": det.frame_labels,
            "mapped_state": mapped_states,
        }
    )
    frame_df.to_parquet(out_dir / "frame_labels.parquet", engine="pyarrow", index=False)

    _write_timeline_png(out_dir / "timeline.png", det, ev, scada, gt)
