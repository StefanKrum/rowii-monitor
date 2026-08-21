import numpy as np
import pandas as pd
import pytest

from rowii.signals.windows import WindowGrid
from rowii.state.segments import duration_filter, to_segments


def test_flicker_run_merges_into_surrounding_state() -> None:
    # A long A run, a 3-window B flicker, another long A run. min_dwell=5 -> the
    # flicker (length 3 < 5) has two neighbours, both A and both length 10, so it
    # dissolves into a single A run spanning the whole array.
    labels = np.array([0] * 10 + [1] * 3 + [0] * 10, dtype=np.int64)

    result = duration_filter(labels, min_dwell=5)

    np.testing.assert_array_equal(result, np.zeros(23, dtype=np.int64))
    assert result.dtype == np.int64


def test_alternating_pattern_resolves_deterministically() -> None:
    # [0, 1, 0, 1, 0] with min_dwell=2: every run has length 1 (< 2), so every
    # iteration merges the shortest-then-leftmost run into its only-decidable
    # neighbour. Documented deterministic trace (leftmost-shortest-run-first,
    # left-tie-break):
    #   iter 1: runs [0:1,1:1,0:1,1:1,0:1] -> target=0 (only right neighbour, run 1)
    #           -> merges into value 1      -> [1:2, 0:1, 1:1, 0:1]
    #   iter 2: shortest run len=1, leftmost is the 0-run at position 1; neighbours
    #           are 1:2 (left) and 1:1 (right) -> left is longer -> absorbed left
    #           -> [1:4, 0:1]
    #   iter 3: single run 0:1 left, only neighbour is 1:4 -> absorbed
    #           -> [1:5]  (single run -> loop stops)
    # Net effect: the whole array collapses to a single run of cluster 1.
    labels = np.array([0, 1, 0, 1, 0], dtype=np.int64)

    result = duration_filter(labels, min_dwell=2)

    np.testing.assert_array_equal(result, np.array([1, 1, 1, 1, 1], dtype=np.int64))


def test_shortest_candidate_run_is_processed_before_longer_candidate_runs() -> None:
    # X(10) A(1) M(2) B(1) Z(10), min_dwell=3: three runs (A, M, B) are below
    # min_dwell. Processing the SHORTEST first (A and B tie at length 1, leftmost
    # -> A) gives a different final split than processing the longest candidate
    # (M, length 2) first -- this is what makes "shortest-first" a load-bearing
    # part of the determinism rule, not an arbitrary implementation detail.
    #
    # Shortest-first trace (the specified, correct order):
    #   iter 1: candidates A(len1,@1) M(len2,@2) B(len1,@3); shortest=1, tie A/B,
    #           leftmost -> A. Neighbours X(10) vs M(2): X longer -> A merges
    #           into X -> [X:11, M:2, B:1, Z:10]
    #   iter 2: candidates M(len2,@1) B(len1,@2); shortest=1 -> B. Neighbours
    #           M(2) vs Z(10): Z longer -> B merges into Z -> [X:11, M:2, Z:11]
    #   iter 3: only candidate M(len2,@1). Neighbours X(11) vs Z(11): tied ->
    #           LEFT wins -> M merges into X -> [X:13, Z:11]
    # Final: a single boundary at window 13, cluster 0 then cluster 4.
    labels = np.array([0] * 10 + [1] * 1 + [2] * 2 + [3] * 1 + [4] * 10, dtype=np.int64)

    result = duration_filter(labels, min_dwell=3)

    expected = np.array([0] * 13 + [4] * 11, dtype=np.int64)
    np.testing.assert_array_equal(result, expected)


def test_tie_break_uses_left_neighbour_when_neighbour_durations_are_equal() -> None:
    # A(3) B(1) C(3): the only sub-min_dwell run is the middle B; its neighbours
    # (A len 3, C len 3) are tied in length, so the LEFT neighbour (A) absorbs it.
    labels = np.array([0, 0, 0, 1, 2, 2, 2], dtype=np.int64)

    result = duration_filter(labels, min_dwell=2)

    np.testing.assert_array_equal(
        result, np.array([0, 0, 0, 0, 2, 2, 2], dtype=np.int64)
    )


def test_min_dwell_of_one_is_a_no_op_copy() -> None:
    labels = np.array([0, 1, 0, 1, 0], dtype=np.int64)

    result = duration_filter(labels, min_dwell=1)

    np.testing.assert_array_equal(result, labels)
    assert result is not labels  # copy, not the same array object
    assert result.dtype == np.int64


def test_min_dwell_of_zero_is_also_a_no_op_copy() -> None:
    labels = np.array([0, 1, 0, 1, 0], dtype=np.int64)

    result = duration_filter(labels, min_dwell=0)

    np.testing.assert_array_equal(result, labels)
    assert result.dtype == np.int64


def test_single_run_input_is_returned_unchanged() -> None:
    labels = np.full(8, 3, dtype=np.int64)

    result = duration_filter(labels, min_dwell=5)

    np.testing.assert_array_equal(result, labels)


def test_output_dtype_is_int64_even_for_int32_input() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    result = duration_filter(labels, min_dwell=2)

    assert result.dtype == np.int64
    assert len(result) == len(labels)


def test_to_segments_timestamps_match_grid_edges_exactly() -> None:
    # Grid: t0 = 1_000_000_000 ns, window = 2_000_000_000 ns (2s), 4 windows.
    # Labels: two segments, [0,0] then [1,1,1] -> wait, must match n_windows=4.
    t0_ns = 1_000_000_000
    window_ns = 2_000_000_000
    grid = WindowGrid(t0_ns=t0_ns, window_ns=window_ns, n_windows=4)
    labels = np.array([7, 7, 9, 9], dtype=np.int64)

    segments = to_segments(labels, grid)

    assert list(segments.columns) == ["start_utc", "end_utc", "duration_s", "cluster"]
    assert len(segments) == 2

    edges = grid.edges_ns()
    expected_start_0 = pd.Timestamp(int(edges[0]), unit="ns", tz="UTC")
    expected_end_0 = pd.Timestamp(int(edges[2]), unit="ns", tz="UTC")
    expected_start_1 = pd.Timestamp(int(edges[2]), unit="ns", tz="UTC")
    expected_end_1 = pd.Timestamp(int(edges[4]), unit="ns", tz="UTC")

    row0 = segments.iloc[0]
    row1 = segments.iloc[1]

    assert row0["start_utc"] == expected_start_0
    assert row0["end_utc"] == expected_end_0
    assert row0["cluster"] == 7
    assert row0["duration_s"] == pytest.approx(4.0)

    assert row1["start_utc"] == expected_start_1
    assert row1["end_utc"] == expected_end_1
    assert row1["cluster"] == 9
    assert row1["duration_s"] == pytest.approx(4.0)

    assert row0["start_utc"].tz is not None
    assert str(row0["start_utc"].tz) == "UTC"


def test_to_segments_dtypes() -> None:
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=3)
    labels = np.array([0, 0, 1], dtype=np.int64)

    segments = to_segments(labels, grid)

    assert isinstance(segments["start_utc"].iloc[0], pd.Timestamp)
    assert isinstance(segments["end_utc"].iloc[0], pd.Timestamp)
    assert segments["cluster"].dtype == np.int64
    assert segments["duration_s"].dtype == np.float64


def test_to_segments_all_one_label_yields_single_segment_covering_whole_grid() -> None:
    grid = WindowGrid(t0_ns=5_000_000_000, window_ns=500_000_000, n_windows=6)
    labels = np.full(6, 2, dtype=np.int64)

    segments = to_segments(labels, grid)

    assert len(segments) == 1
    edges = grid.edges_ns()
    assert segments.iloc[0]["start_utc"] == pd.Timestamp(int(edges[0]), unit="ns", tz="UTC")
    assert segments.iloc[0]["end_utc"] == pd.Timestamp(int(edges[6]), unit="ns", tz="UTC")
    assert segments.iloc[0]["cluster"] == 2
    assert segments.iloc[0]["duration_s"] == pytest.approx(3.0)


def test_to_segments_raises_on_length_mismatch() -> None:
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=5)
    labels = np.array([0, 0, 1], dtype=np.int64)  # length 3 != n_windows=5

    with pytest.raises(ValueError, match="length"):
        to_segments(labels, grid)


def test_to_segments_multiple_maximal_runs_row_count_and_order() -> None:
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=7)
    labels = np.array([0, 0, 1, 1, 1, 0, 2], dtype=np.int64)

    segments = to_segments(labels, grid)

    assert len(segments) == 4
    np.testing.assert_array_equal(segments["cluster"].to_numpy(), [0, 1, 0, 2])
    # Segment table is contiguous: each row's end == the next row's start.
    for i in range(len(segments) - 1):
        assert segments.iloc[i]["end_utc"] == segments.iloc[i + 1]["start_utc"]
    assert segments.iloc[0]["start_utc"] == pd.Timestamp(0, unit="ns", tz="UTC")
    assert segments.iloc[-1]["end_utc"] == pd.Timestamp(
        int(grid.edges_ns()[-1]), unit="ns", tz="UTC"
    )


def test_to_segments_on_an_overlapping_grid_spans_first_start_to_last_end() -> None:
    # 1 s windows every 0.25 s: a run of r windows starting at window i spans
    # [start(i), start(i + r - 1) + window_ns) -- the last window's END, which on
    # an overlapping grid is NOT the next window's start.
    grid = WindowGrid(
        t0_ns=0, window_ns=1_000_000_000, n_windows=5, hop_ns=250_000_000
    )
    labels = np.array([3, 3, 3, 8, 8], dtype=np.int64)

    segments = to_segments(labels, grid)

    assert len(segments) == 2
    row0, row1 = segments.iloc[0], segments.iloc[1]
    assert row0["start_utc"] == pd.Timestamp(0, unit="ns", tz="UTC")
    assert row0["end_utc"] == pd.Timestamp(1_500_000_000, unit="ns", tz="UTC")
    assert row0["duration_s"] == pytest.approx(1.5)
    assert row1["start_utc"] == pd.Timestamp(750_000_000, unit="ns", tz="UTC")
    assert row1["end_utc"] == pd.Timestamp(2_000_000_000, unit="ns", tz="UTC")
    assert row1["duration_s"] == pytest.approx(1.25)
