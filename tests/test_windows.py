import numpy as np
import pytest

from rowii.io.gantner import GantnerHeader
from rowii.signals.windows import WindowGrid, common_grid, coverage, window_slices


def _header(t0_ns: int, n_frames: int, rate_hz: float = 100.0) -> GantnerHeader:
    return GantnerHeader(
        source_name="s",
        channel_names=["A"],
        channel_units=[""],
        t0_ns=t0_ns,
        sample_rate_hz=rate_hz,
        n_frames=n_frames,
    )


def test_common_grid_covers_offset_t0_intersection() -> None:
    # Stream A: [0, 10s) at 100 Hz -> t_end = 10_000_000_000 ns
    # Stream B: starts 2.5s later, runs to 12.5s -> t_end = 12_500_000_000 ns
    # Intersection: [2_500_000_000, 10_000_000_000) ns = 7.5 s
    a = _header(t0_ns=0, n_frames=1000, rate_hz=100.0)
    b = _header(t0_ns=2_500_000_000, n_frames=1000, rate_hz=100.0)

    grid = common_grid([a, b], window_s=1.0)

    assert grid.t0_ns == 2_500_000_000
    assert grid.window_ns == 1_000_000_000
    # floor(7.5s / 1s) = 7 whole windows
    assert grid.n_windows == 7


def test_common_grid_raises_on_empty_intersection() -> None:
    # Stream A: [0, 1s); Stream B: [5s, 6s) -> disjoint, no intersection
    a = _header(t0_ns=0, n_frames=100, rate_hz=100.0)
    b = _header(t0_ns=5_000_000_000, n_frames=100, rate_hz=100.0)

    with pytest.raises(ValueError, match="intersection"):
        common_grid([a, b], window_s=1.0)


def test_common_grid_raises_when_intersection_shorter_than_one_window() -> None:
    # Intersection duration 0.5s < window_s=1.0 -> n_windows would be 0
    a = _header(t0_ns=0, n_frames=100, rate_hz=100.0)          # [0, 1s)
    b = _header(t0_ns=500_000_000, n_frames=100, rate_hz=100.0)  # [0.5s, 1.5s)

    with pytest.raises(ValueError):
        common_grid([a, b], window_s=1.0)


def test_edges_ns_shape_and_values() -> None:
    grid = WindowGrid(t0_ns=1_000, window_ns=1_000_000_000, n_windows=3)

    edges = grid.edges_ns()

    assert edges.dtype == np.uint64
    assert edges.shape == (4,)
    np.testing.assert_array_equal(
        edges, np.array([1_000, 1_000_001_000, 2_000_001_000, 3_000_001_000],
                         dtype=np.uint64),
    )


def test_window_slices_reconstruct_correct_sample_counts_at_100hz() -> None:
    # 5 windows of 1s at 100 Hz -> 100 samples/window, contiguous, no gaps.
    rate_hz = 100.0
    n_windows = 5
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=n_windows)
    n_samples = int(rate_hz * n_windows)
    ts_ns = (np.arange(n_samples, dtype=np.uint64) * np.uint64(round(1e9 / rate_hz)))

    slices = window_slices(ts_ns, grid)

    assert len(slices) == n_windows
    total = 0
    for sl in slices:
        count = sl.stop - sl.start
        assert count == 100
        total += count
    assert total == n_samples
    # Reconstructed samples equal the original array exactly (no overlap, no drop).
    reconstructed = np.concatenate([ts_ns[sl] for sl in slices])
    np.testing.assert_array_equal(reconstructed, ts_ns)


def test_window_slices_empty_window_yields_empty_slice() -> None:
    # ts only covers windows 0 and 2 of a 3-window grid; window 1 ([1e9, 2e9) ns) has
    # no samples at all.
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=3)
    ts_ns = np.array([0, 500_000_000, 2_000_000_000, 2_500_000_000], dtype=np.uint64)

    slices = window_slices(ts_ns, grid)

    assert len(slices) == 3
    assert slices[1].start == slices[1].stop  # empty: slice(i, i)
    assert ts_ns[slices[1]].size == 0
    assert ts_ns[slices[0]].size == 2
    assert ts_ns[slices[2]].size == 2


def test_coverage_full_for_gapless_stream() -> None:
    rate_hz = 100.0
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=4)
    ts_ns = np.arange(400, dtype=np.uint64) * np.uint64(round(1e9 / rate_hz))

    cov = coverage(ts_ns, grid, rate_hz)

    assert cov.shape == (4,)
    np.testing.assert_allclose(cov, np.ones(4), atol=1e-9)


def test_coverage_reduced_on_exactly_the_gapped_windows() -> None:
    # 5 windows at 100 Hz (100 samples/window each when full). Window index 2 has a
    # gap: only 40 of its 100 samples are present. Windows 0, 1, 3, 4 are full.
    rate_hz = 100.0
    window_ns = 1_000_000_000
    grid = WindowGrid(t0_ns=0, window_ns=window_ns, n_windows=5)
    dt_ns = round(1e9 / rate_hz)

    full_windows_ts = []
    for w in (0, 1, 3, 4):
        start = w * window_ns
        full_windows_ts.append(start + np.arange(100, dtype=np.int64) * dt_ns)
    gapped_window_ts = 2 * window_ns + np.arange(40, dtype=np.int64) * dt_ns
    ts_ns = np.sort(
        np.concatenate([*full_windows_ts, gapped_window_ts]).astype(np.uint64)
    )

    cov = coverage(ts_ns, grid, rate_hz)

    assert cov.shape == (5,)
    for w in (0, 1, 3, 4):
        assert cov[w] == pytest.approx(1.0, abs=1e-9)
    assert cov[2] == pytest.approx(0.40, abs=1e-9)
    # Exactly one window is below the 0.8 threshold mentioned in the brief; the
    # drop/hard-fail *policy* itself is out of scope for this module (caller concern).
    below_threshold = cov < 0.8
    assert below_threshold.tolist() == [False, False, True, False, False]


def test_coverage_is_clipped_to_one_when_extra_samples_present() -> None:
    # More samples in a window than the nominal rate implies (e.g. rate underestimate
    # or duplicate frames) must not push coverage above 1.0.
    rate_hz = 100.0
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=1)
    # 150 samples inside a single 1 s window at nominal 100 Hz -> raw ratio 1.5.
    ts_ns = np.linspace(0, 999_000_000, 150).astype(np.uint64)

    cov = coverage(ts_ns, grid, rate_hz)

    assert cov[0] == 1.0


# ---------------------------------------------------------------------------
# Sub-window hop (window_s stays the window DURATION; hop_s is the spacing
# between consecutive window STARTS). Everything below the first test is new
# behaviour; the first test is the invariance pin for the default (hop == window).
# ---------------------------------------------------------------------------


def test_common_grid_defaults_hop_to_the_window_length() -> None:
    # Omitting hop_s must reproduce the pre-hop grid exactly: same t0, same
    # window_ns, same n_windows, and a hop equal to the window (non-overlapping).
    a = _header(t0_ns=0, n_frames=1000, rate_hz=100.0)          # [0, 10s)
    b = _header(t0_ns=2_500_000_000, n_frames=1000, rate_hz=100.0)  # [2.5s, 12.5s)

    grid = common_grid([a, b], window_s=1.0)

    assert grid.t0_ns == 2_500_000_000
    assert grid.window_ns == 1_000_000_000
    assert grid.n_windows == 7
    assert grid.hop_ns == 1_000_000_000


def test_common_grid_sub_second_hop_packs_overlapping_windows() -> None:
    # One 3 s stream, 1 s windows every 0.25 s: the last window that still FITS
    # starts at 2.0 s, so (3.0 - 1.0) / 0.25 + 1 = 9 windows.
    a = _header(t0_ns=0, n_frames=300, rate_hz=100.0)  # [0, 3s)

    grid = common_grid([a], window_s=1.0, hop_s=0.25)

    assert grid.window_ns == 1_000_000_000
    assert grid.hop_ns == 250_000_000
    assert grid.n_windows == 9
    starts = grid.starts_ns()
    ends = grid.ends_ns()
    assert starts.dtype == np.uint64
    assert ends.dtype == np.uint64
    np.testing.assert_array_equal(
        starts, np.arange(9, dtype=np.uint64) * np.uint64(250_000_000)
    )
    np.testing.assert_array_equal(ends, starts + np.uint64(1_000_000_000))
    # Every window ends inside the stream -- no partial window is ever emitted.
    assert int(ends[-1]) == 3_000_000_000


def test_starts_and_ends_agree_with_edges_when_hop_equals_window() -> None:
    grid = WindowGrid(t0_ns=1_000, window_ns=1_000_000_000, n_windows=3)

    edges = grid.edges_ns()

    np.testing.assert_array_equal(grid.starts_ns(), edges[:-1])
    np.testing.assert_array_equal(grid.ends_ns(), edges[1:])


def test_edges_ns_refuses_an_overlapping_grid() -> None:
    # edges_ns()'s left-closed/right-open [edges[i], edges[i+1]) contract only
    # describes a tiling; on an overlapping grid it must fail loudly rather than
    # hand back hop-length pseudo-windows.
    grid = WindowGrid(
        t0_ns=0, window_ns=1_000_000_000, n_windows=3, hop_ns=250_000_000
    )

    with pytest.raises(ValueError, match="hop_ns"):
        grid.edges_ns()


def test_window_slices_overlap_when_hop_is_shorter_than_the_window() -> None:
    # 100 Hz, 1 s windows every 0.25 s: every window holds a FULL 100 samples and
    # consecutive windows share their last/first 75.
    rate_hz = 100.0
    grid = WindowGrid(
        t0_ns=0, window_ns=1_000_000_000, n_windows=5, hop_ns=250_000_000
    )
    ts_ns = np.arange(200, dtype=np.uint64) * np.uint64(round(1e9 / rate_hz))

    slices = window_slices(ts_ns, grid)

    assert len(slices) == 5
    for w, sl in enumerate(slices):
        assert sl.stop - sl.start == 100
        assert sl.start == 25 * w
    shared = set(range(slices[0].start, slices[0].stop)) & set(
        range(slices[1].start, slices[1].stop)
    )
    assert len(shared) == 75


def test_coverage_is_full_for_every_window_of_an_overlapping_grid() -> None:
    rate_hz = 100.0
    grid = WindowGrid(
        t0_ns=0, window_ns=1_000_000_000, n_windows=5, hop_ns=250_000_000
    )
    ts_ns = np.arange(200, dtype=np.uint64) * np.uint64(round(1e9 / rate_hz))

    cov = coverage(ts_ns, grid, rate_hz)

    assert cov.shape == (5,)
    np.testing.assert_allclose(cov, np.ones(5), atol=1e-9)


def test_hop_grid_window_starts_are_a_superset_of_the_coarse_grid_starts() -> None:
    # The fine grid must contain every coarse window verbatim (same start, same
    # duration) -- the fine run is the SAME grid, only denser.
    a = _header(t0_ns=0, n_frames=1000, rate_hz=100.0)  # [0, 10s)

    coarse = common_grid([a], window_s=1.0)
    fine = common_grid([a], window_s=1.0, hop_s=0.25)

    assert fine.window_ns == coarse.window_ns
    assert set(coarse.starts_ns().tolist()) <= set(fine.starts_ns().tolist())


def test_common_grid_rejects_a_non_positive_or_over_long_hop() -> None:
    a = _header(t0_ns=0, n_frames=1000, rate_hz=100.0)

    with pytest.raises(ValueError, match="hop_s"):
        common_grid([a], window_s=1.0, hop_s=0.0)
    with pytest.raises(ValueError, match="hop_s"):
        common_grid([a], window_s=1.0, hop_s=2.0)
