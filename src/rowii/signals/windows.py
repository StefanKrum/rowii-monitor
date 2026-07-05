"""Common UTC window grid across streams, plus per-stream slicing and coverage.

This module provides only the grid/slice/coverage primitives. It does NOT implement
any run-level policy: the 0.8-coverage drop rule and the >5% hard-fail rule (spec §6)
are the CALLER's responsibility (detect/CLI layer), not this module's.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rowii.io.gantner import GantnerHeader


@dataclass(frozen=True)
class WindowGrid:
    t0_ns: int
    window_ns: int
    n_windows: int

    def edges_ns(self) -> np.ndarray:
        """Window edge timestamps, shape (n_windows + 1,), dtype uint64.

        Windows are left-closed/right-open: window *i* covers
        ``[edges_ns()[i], edges_ns()[i + 1])``.
        """
        offsets = np.arange(self.n_windows + 1, dtype=np.uint64) * np.uint64(self.window_ns)
        return offsets + np.uint64(self.t0_ns)


def _stream_end_ns(header: GantnerHeader) -> int:
    return header.t0_ns + round(header.n_frames / header.sample_rate_hz * 1e9)


def common_grid(headers: Sequence[GantnerHeader], window_s: float) -> WindowGrid:
    """Build a window grid spanning the INTERSECTION of every stream's [t0, t_end).

    The grid's t0 is the exact intersection start (not rounded to a window boundary);
    windows tile forward from there. Raises ValueError if the intersection is empty
    or spans fewer than one whole window.
    """
    if not headers:
        raise ValueError("common_grid requires at least one header")

    start_ns = max(h.t0_ns for h in headers)
    end_ns = min(_stream_end_ns(h) for h in headers)
    if end_ns <= start_ns:
        raise ValueError(
            f"empty intersection across {len(headers)} stream(s): "
            f"start_ns={start_ns} >= end_ns={end_ns}"
        )

    window_ns = round(window_s * 1e9)
    n_windows = (end_ns - start_ns) // window_ns
    if n_windows == 0:
        raise ValueError(
            f"intersection duration ({end_ns - start_ns} ns) is shorter than one "
            f"window ({window_ns} ns)"
        )
    return WindowGrid(t0_ns=start_ns, window_ns=window_ns, n_windows=n_windows)


def window_slices(ts_ns: np.ndarray, grid: WindowGrid) -> list[slice]:
    """Per-window sample slice into *ts_ns* (sorted, uint64 timestamps of one stream).

    Windows are left-closed/right-open. A window with no samples in range yields
    slice(i, i) (empty).
    """
    edges = grid.edges_ns()
    idx = np.searchsorted(ts_ns, edges)
    return [slice(int(idx[i]), int(idx[i + 1])) for i in range(grid.n_windows)]


def coverage(ts_ns: np.ndarray, grid: WindowGrid, rate_hz: float) -> np.ndarray:
    """Fraction of expected samples present per window, clipped to [0, 1].

    Expected count per window is ``rate_hz * window_s``; the ratio is clipped so that
    duplicate/extra samples (or a rate underestimate) cannot push coverage above 1.0.
    """
    window_s = grid.window_ns / 1e9
    expected = rate_hz * window_s
    counts = np.array(
        [sl.stop - sl.start for sl in window_slices(ts_ns, grid)], dtype=np.float64
    )
    return np.clip(counts / expected, 0.0, 1.0)
