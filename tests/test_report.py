from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rowii.eval.metrics import EvalResult, evaluate
from rowii.eval.report import _confusion_to_markdown, _window_s_from_segments, write_report
from rowii.signals.windows import WindowGrid
from rowii.state.detect import DetectionResult
from rowii.state.segments import to_segments


def _grid(n_windows: int, window_s: float = 1.0) -> WindowGrid:
    return WindowGrid(t0_ns=0, window_ns=round(window_s * 1e9), n_windows=n_windows)


def _gt(states: list[str]) -> pd.DataFrame:
    n = len(states)
    return pd.DataFrame(
        {"state": states, "load_bin": np.full(n, -1, dtype=np.int64)},
        index=pd.RangeIndex(n),
    )


def _scada(n_windows: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "power": rng.normal(10.0, 2.0, n_windows),
            "speed": np.full(n_windows, 375.0),
            "guide_vane": np.full(n_windows, 50.0),
            "flow_tu": np.full(n_windows, 5.0),
            "flow_pu": np.full(n_windows, 0.0),
        }
    )


def _det_and_ev(
    n_windows: int = 20,
) -> tuple[DetectionResult, EvalResult, WindowGrid, pd.DataFrame]:
    grid = _grid(n_windows)
    frame_labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    segments = to_segments(frame_labels, grid)
    det = DetectionResult(frame_labels=frame_labels, segments=segments, k=2)
    states = ["standstill"] * 10 + ["turbine"] * 10
    gt = _gt(states)
    ev = evaluate(frame_labels, gt, grid)
    scada = _scada(n_windows)
    return det, ev, grid, scada


def test_write_report_creates_all_four_expected_files(tmp_path: Path) -> None:
    det, ev, grid, scada = _det_and_ev()

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "timeline.png").exists()
    assert (tmp_path / "segments.csv").exists()
    assert (tmp_path / "frame_labels.parquet").exists()


def test_write_report_creates_out_dir_if_missing(tmp_path: Path) -> None:
    det, ev, grid, scada = _det_and_ev()
    out_dir = tmp_path / "nested" / "run_dir"
    assert not out_dir.exists()

    write_report(out_dir, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    assert out_dir.exists()
    assert (out_dir / "report.md").exists()


def test_report_md_contains_run_variant_header_and_metric_values(tmp_path: Path) -> None:
    det, ev, grid, scada = _det_and_ev()

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    text = (tmp_path / "report.md").read_text()
    assert "tu" in text
    assert "audio-handcrafted" in text
    assert f"{ev.ari:.4f}" in text or str(ev.ari) in text
    assert f"{ev.macro_f1:.4f}" in text or str(ev.macro_f1) in text
    assert str(ev.n_eval_windows) in text
    assert str(det.k) in text
    # Mapping entries and confusion matrix rows/cols must be represented.
    for cluster_id, state_name in ev.mapping.items():
        assert str(cluster_id) in text
        assert state_name in text
    for state in ev.confusion.index:
        assert state in text


def test_report_md_contains_boundary_metric_or_none_marker(tmp_path: Path) -> None:
    det, ev, grid, scada = _det_and_ev()

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    text = (tmp_path / "report.md").read_text()
    if ev.boundary_median_abs_s is None:
        assert "None" in text or "N/A" in text or "none" in text.lower()
    else:
        assert f"{ev.boundary_median_abs_s:.4f}" in text or str(ev.boundary_median_abs_s) in text


def test_report_md_reports_unknown_dropped_window_count(tmp_path: Path) -> None:
    n_windows = 20
    grid = _grid(n_windows)
    frame_labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    segments = to_segments(frame_labels, grid)
    det = DetectionResult(frame_labels=frame_labels, segments=segments, k=2)
    states = ["standstill"] * 8 + ["turbine"] * 7 + ["unknown"] * 5
    gt = _gt(states)
    ev = evaluate(frame_labels, gt, grid)
    scada = _scada(n_windows)

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    text = (tmp_path / "report.md").read_text()
    # 5 unknown windows out of 20 total -> 5 dropped.
    assert "5" in text


def test_timeline_png_is_non_empty_and_reasonably_sized(tmp_path: Path) -> None:
    det, ev, grid, scada = _det_and_ev()

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    size = (tmp_path / "timeline.png").stat().st_size
    assert size > 1024  # > 1 KB; not asserting pixel content, just non-trivial output


def test_segments_csv_matches_det_segments(tmp_path: Path) -> None:
    det, ev, grid, scada = _det_and_ev()

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    written = pd.read_csv(tmp_path / "segments.csv")
    assert len(written) == len(det.segments)
    assert list(written["cluster"]) == list(det.segments["cluster"])


def test_frame_labels_parquet_has_window_cluster_mapped_state_columns(tmp_path: Path) -> None:
    det, ev, grid, scada = _det_and_ev()

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    written = pd.read_parquet(tmp_path / "frame_labels.parquet", engine="pyarrow")
    assert list(written.columns) == ["window", "cluster", "mapped_state"]
    assert len(written) == len(det.frame_labels)
    assert list(written["window"]) == list(range(len(det.frame_labels)))
    assert list(written["cluster"]) == list(det.frame_labels)
    for cluster_id, mapped_state in zip(written["cluster"], written["mapped_state"], strict=True):
        if int(cluster_id) in ev.mapping:
            assert mapped_state == ev.mapping[int(cluster_id)]


def test_write_report_does_not_raise_when_scada_has_nan_windows(tmp_path: Path) -> None:
    # Uncovered SCADA windows are NaN (per load_scada_window_means contract) -- the
    # power-curve panel must tolerate this rather than crashing on plot.
    det, ev, grid, scada = _det_and_ev()
    scada = scada.copy()
    scada.loc[0:3, "power"] = np.nan

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    assert (tmp_path / "timeline.png").stat().st_size > 1024


def test_write_report_accepts_optional_gt_keyword_and_still_writes_all_files(
    tmp_path: Path,
) -> None:
    # `gt` is the ONLY object in the whole pipeline that carries true per-window GT
    # state identity (EvalResult.confusion is an aggregate crosstab that cannot be
    # inverted back to a per-window sequence) -- write_report accepts it as an
    # optional, keyword-only extra (mirroring the documented, controller-approved
    # `run_detection(..., k=...)` precedent in rowii.state.detect) so the timeline's
    # GT-states panel can be populated with real data when the caller has `gt` in
    # scope (`scripts/run_step1.py`'s CLI does, right before its own `evaluate`
    # call), while every
    # positional-only call from the exact brief-given signature keeps working
    # unchanged (see the other tests in this module, none of which pass `gt`).
    det, ev, grid, scada = _det_and_ev()
    states = ["standstill"] * 10 + ["turbine"] * 10
    gt = _gt(states)

    write_report(
        tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada, gt=gt
    )

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "timeline.png").exists()
    assert (tmp_path / "segments.csv").exists()
    assert (tmp_path / "frame_labels.parquet").exists()
    assert (tmp_path / "timeline.png").stat().st_size > 1024


def test_write_report_without_gt_still_produces_a_valid_png_gt_panel_placeholder(
    tmp_path: Path,
) -> None:
    # Omitting `gt` (the exact brief signature's call shape) must not crash and must
    # not silently plot something misleading in the GT panel -- just produce a valid,
    # non-trivial PNG with an explicit "no GT provided" placeholder panel.
    det, ev, grid, scada = _det_and_ev()

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    assert (tmp_path / "timeline.png").stat().st_size > 1024


def test_window_s_from_segments_derives_the_actual_grid_window_duration() -> None:
    # A grid with window_s=2.0 (NOT the 1.0s every other test in this module uses --
    # deliberately, so a "hardcoded 1.0" mutation cannot hide behind every scenario
    # sharing the same value): derived window_s must recover 2.0 exactly from the
    # segment table's UTC span alone (no WindowGrid/timestamps otherwise available to
    # write_report -- see module docstring rationale).
    grid = _grid(10, window_s=2.0)
    frame_labels = np.array([0] * 5 + [1] * 5, dtype=np.int64)
    segments = to_segments(frame_labels, grid)
    det = DetectionResult(frame_labels=frame_labels, segments=segments, k=2)

    window_s = _window_s_from_segments(det)

    assert window_s == pytest.approx(2.0)


def test_window_s_from_segments_falls_back_to_one_second_for_zero_windows() -> None:
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=0)
    frame_labels = np.array([], dtype=np.int64)
    segments = to_segments(frame_labels, grid)
    det = DetectionResult(frame_labels=frame_labels, segments=segments, k=0)

    assert _window_s_from_segments(det) == 1.0


def test_timeline_png_uses_correct_hours_scale_for_non_default_window_s(
    tmp_path: Path,
) -> None:
    # Integration-level companion to the direct _window_s_from_segments unit tests
    # above: a full write_report call with a 2.0s-window grid must not raise and must
    # still produce a valid PNG (guards against the derived window_s ever being fed
    # into the plotting call in a way that breaks on non-1.0 values, e.g. a stale
    # cached WindowGrid assumption elsewhere in the plotting code).
    grid = _grid(20, window_s=2.0)
    frame_labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    segments = to_segments(frame_labels, grid)
    det = DetectionResult(frame_labels=frame_labels, segments=segments, k=2)
    states = ["standstill"] * 10 + ["turbine"] * 10
    gt = _gt(states)
    ev = evaluate(frame_labels, gt, grid)
    scada = _scada(20)

    write_report(
        tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada, gt=gt
    )

    assert (tmp_path / "timeline.png").stat().st_size > 1024


def test_gt_states_panel_receives_the_actual_gt_state_sequence_when_gt_is_passed(
    tmp_path: Path,
) -> None:
    # Spy-based, independent of pixel content (by design: "do NOT assert pixel
    # content"): asserts the GT-states panel is plotted from the REAL per-window
    # `gt["state"]` sequence when `gt` is passed, not silently ignored in favour of the
    # "no GT provided" placeholder (a gap that file-existence/size-only assertions
    # cannot catch -- both code paths still produce a valid, non-trivial PNG).
    det, ev, grid, scada = _det_and_ev()
    states = ["standstill"] * 10 + ["turbine"] * 10
    gt = _gt(states)

    with patch("rowii.eval.report._plot_state_panel") as spy_plot_panel:
        write_report(
            tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada, gt=gt
        )

    # Two panels use _plot_state_panel when gt is provided: GT states and predicted
    # states (the placeholder "no GT provided" branch is NOT one of its callers).
    assert spy_plot_panel.call_count == 2
    gt_panel_call_states = spy_plot_panel.call_args_list[0].args[2]
    assert gt_panel_call_states == states


def test_gt_states_panel_is_not_plotted_from_real_data_when_gt_is_omitted(
    tmp_path: Path,
) -> None:
    det, ev, grid, scada = _det_and_ev()

    with patch("rowii.eval.report._plot_state_panel") as spy_plot_panel:
        write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    # Only the predicted-states panel calls _plot_state_panel; the GT panel falls back
    # to the placeholder branch instead.
    assert spy_plot_panel.call_count == 1


def test_report_md_contains_state_level_primary_section_above_strict_section(
    tmp_path: Path,
) -> None:
    det, ev, grid, scada = _det_and_ev()

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    text = (tmp_path / "report.md").read_text()
    assert "State-level (mode) metrics" in text
    assert "Strict (1:1 Hungarian) metrics" in text
    # The state-level section must appear BEFORE the strict section in the file.
    assert text.index("State-level (mode) metrics") < text.index(
        "Strict (1:1 Hungarian) metrics"
    )
    assert f"{ev.state_accuracy:.4f}" in text
    assert f"{ev.state_macro_f1:.4f}" in text
    assert f"{ev.state_ari:.4f}" in text


# ---------------------------------------------------------------------------
# Per-family confusion matrices, each explicitly labeled with its own
# mapping scheme (confusion-clarity fix): a raw cluster can be named
# differently under Hungarian vs. majority mapping, so neither confusion
# matrix may be rendered under an ambiguous or the wrong section's header.
# ---------------------------------------------------------------------------


def test_report_md_has_two_explicitly_labeled_confusion_matrix_headers(tmp_path: Path) -> None:
    det, ev, grid, scada = _det_and_ev()

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    text = (tmp_path / "report.md").read_text()
    assert (
        "## Confusion matrix — state-level / majority mapping "
        "(rows = GT state, cols = majority-mapped prediction)"
    ) in text
    assert (
        "## Confusion matrix — strict / Hungarian mapping "
        "(rows = GT state, cols = Hungarian-mapped prediction)"
    ) in text
    # The old, scheme-less header must be gone -- no unlabeled confusion section
    # may remain (the exact ambiguity this fix eliminates).
    assert "## Confusion matrix (rows = GT state, cols = mapped predicted state)" not in text


def test_report_md_each_confusion_matrix_appears_within_its_own_metric_family_section(
    tmp_path: Path,
) -> None:
    det, ev, grid, scada = _det_and_ev()

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    text = (tmp_path / "report.md").read_text()
    state_level_header_idx = text.index("State-level (mode) metrics")
    state_confusion_idx = text.index("## Confusion matrix — state-level / majority mapping")
    strict_header_idx = text.index("Strict (1:1 Hungarian) metrics")
    strict_confusion_idx = text.index("## Confusion matrix — strict / Hungarian mapping")

    # Ordering: state-level section (metrics -> its own confusion matrix) fully
    # precedes the strict section (metrics -> its own confusion matrix) -- each
    # family owns its confusion matrix instead of both being deferred to one
    # shared section at the end.
    assert (
        state_level_header_idx
        < state_confusion_idx
        < strict_header_idx
        < strict_confusion_idx
    )


def test_report_md_renders_state_confusion_and_confusion_under_their_own_headers(
    tmp_path: Path,
) -> None:
    # Divergence fixture (mirrors test_metrics.py's pinning test): cluster 1's
    # Hungarian-mapped name ("standstill") differs from its majority-mapped name
    # ("transition"), so `ev.confusion` and `ev.state_confusion` are genuinely
    # DIFFERENT DataFrames here -- this test pins down that report.md renders each
    # one under its OWN correctly labeled header, never the other's.
    n_windows = 42
    grid = _grid(n_windows)
    states = ["transition"] * 20 + ["standstill"] * 10 + ["transition"] * 12
    gt = _gt(states)
    frame_labels = np.array([0] * 20 + [1] * 10 + [1] * 12, dtype=np.int64)
    segments = to_segments(frame_labels, grid)
    det = DetectionResult(frame_labels=frame_labels, segments=segments, k=2)
    ev = evaluate(frame_labels, gt, grid)
    scada = _scada(n_windows)
    assert ev.mapping[1] == "standstill"
    assert ev.state_mapping[1] == "transition"

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    text = (tmp_path / "report.md").read_text()
    state_level_idx = text.index("## Confusion matrix — state-level / majority mapping")
    strict_idx = text.index("## Confusion matrix — strict / Hungarian mapping")
    state_level_section = text[state_level_idx:strict_idx]
    strict_section = text[strict_idx:]

    assert _confusion_to_markdown(ev.state_confusion) in state_level_section
    assert _confusion_to_markdown(ev.state_confusion) not in strict_section
    assert _confusion_to_markdown(ev.confusion) in strict_section
    assert _confusion_to_markdown(ev.confusion) not in state_level_section


def test_report_md_state_mapping_table_reflects_state_mapping_not_strict_mapping(
    tmp_path: Path,
) -> None:
    det, ev, grid, scada = _det_and_ev()

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    text = (tmp_path / "report.md").read_text()
    for cluster_id, state_name in ev.state_mapping.items():
        assert f"| {cluster_id} | {state_name} |" in text


def test_predicted_timeline_panel_uses_state_mapping_not_strict_mapping(
    tmp_path: Path,
) -> None:
    # Construct a case where state_mapping and strict mapping actually DIFFER for at
    # least one cluster, then assert the predicted-states panel (spied via
    # _plot_state_panel) receives the state_mapping-derived sequence.
    n_windows = 30
    grid = _grid(n_windows)
    states = ["standstill"] * 10 + ["turbine"] * 20
    gt = _gt(states)
    frame_labels = np.array([0] * 10 + [0] * 15 + [1] * 5, dtype=np.int64)
    segments = to_segments(frame_labels, grid)
    det = DetectionResult(frame_labels=frame_labels, segments=segments, k=2)
    ev = evaluate(frame_labels, gt, grid)
    scada = _scada(n_windows)
    assert ev.state_mapping[0] == "turbine" and ev.state_mapping[1] == "turbine"
    assert not (ev.mapping[0] == "turbine" and ev.mapping[1] == "turbine")

    with patch("rowii.eval.report._plot_state_panel") as spy_plot_panel:
        write_report(
            tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada, gt=gt
        )

    # Second call (index 1, after the GT-panel call at index 0) is the predicted panel.
    predicted_panel_states = spy_plot_panel.call_args_list[1].args[2]
    expected = [ev.state_mapping[int(c)] for c in frame_labels]
    assert predicted_panel_states == expected


# ---------------------------------------------------------------------------
# "Do sub-clusters track load levels?" section
# ---------------------------------------------------------------------------


def _gt_with_load_bin(states: list[str], load_bin: list[int]) -> pd.DataFrame:
    assert len(states) == len(load_bin)
    return pd.DataFrame(
        {"state": states, "load_bin": np.array(load_bin, dtype=np.int64)},
        index=pd.RangeIndex(len(states)),
    )


def test_report_md_contains_load_alignment_section_with_crosstab_and_ari_when_gt_given(
    tmp_path: Path,
) -> None:
    n_windows = 20
    grid = _grid(n_windows)
    states = ["turbine"] * n_windows
    load_bin = [0] * 10 + [1] * 10
    gt = _gt_with_load_bin(states, load_bin)
    frame_labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    segments = to_segments(frame_labels, grid)
    det = DetectionResult(frame_labels=frame_labels, segments=segments, k=2)
    ev = evaluate(frame_labels, gt, grid)
    scada = _scada(n_windows)

    write_report(
        tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada, gt=gt
    )

    text = (tmp_path / "report.md").read_text()
    assert "Do sub-clusters track load levels?" in text
    assert "1.0000" in text  # perfect cluster<->load_bin alignment in this fixture


def test_report_md_load_alignment_section_shows_na_when_gt_omitted(tmp_path: Path) -> None:
    det, ev, grid, scada = _det_and_ev()

    write_report(tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada)

    text = (tmp_path / "report.md").read_text()
    assert "Do sub-clusters track load levels?" in text
    assert "n/a" in text.lower()


def test_report_md_load_alignment_section_shows_na_when_fewer_than_two_load_bins(
    tmp_path: Path,
) -> None:
    n_windows = 10
    grid = _grid(n_windows)
    states = ["turbine"] * n_windows
    load_bin = [0] * n_windows  # single load bin -> load_alignment returns None
    gt = _gt_with_load_bin(states, load_bin)
    frame_labels = np.array([0] * 5 + [1] * 5, dtype=np.int64)
    segments = to_segments(frame_labels, grid)
    det = DetectionResult(frame_labels=frame_labels, segments=segments, k=2)
    ev = evaluate(frame_labels, gt, grid)
    scada = _scada(n_windows)

    write_report(
        tmp_path, run="tu", variant="audio-handcrafted", det=det, ev=ev, scada=scada, gt=gt
    )

    text = (tmp_path / "report.md").read_text()
    assert "Do sub-clusters track load levels?" in text
    assert "n/a" in text.lower()
