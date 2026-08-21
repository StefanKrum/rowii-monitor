"""Common UTC window grid across streams, plus per-stream slicing and coverage.

This module provides only the grid/slice/coverage primitives. It does NOT implement
any run-level policy: the 0.8-coverage drop rule and the >5% hard-fail rule
are the CALLER's responsibility (detect/CLI layer), not this module's.

A grid has two independent time scales: `window_ns` (how LONG a window is -- the
span of samples a featurizer sees) and `hop_ns` (how far apart consecutive window
STARTS sit). They coincide by default, which is the non-overlapping tiling this
project used before sub-window hops existed; `hop_ns < window_ns` produces an
OVERLAPPING grid whose windows are still `window_ns` long, only started more
often. Read the spacing through `WindowGrid.step_ns` (never through `window_ns`)
anywhere a window INDEX is turned into a timestamp -- `t0_ns + i * window_ns` is
correct only on a tiling and silently wrong on every other grid.
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
    hop_ns: int | None = None
    """Spacing between consecutive window STARTS in nanoseconds; `None` (the
    default) means "same as `window_ns`", i.e. the non-overlapping tiling every
    caller predating sub-window hops builds. Prefer reading `step_ns`, which
    resolves that default -- the raw field is `None`-typed only so a
    hand-constructed `WindowGrid(t0_ns=..., window_ns=..., n_windows=...)` keeps
    working verbatim. `common_grid` always sets it explicitly (never leaves it
    `None`), so two grids built the same way compare equal."""

    @property
    def step_ns(self) -> int:
        """`hop_ns` with its `None` default resolved to `window_ns` -- the
        spacing between consecutive window starts, always an `int`."""
        return self.window_ns if self.hop_ns is None else self.hop_ns

    @property
    def is_tiling(self) -> bool:
        """True iff windows tile without overlap (`step_ns == window_ns`) --
        the precondition `edges_ns` needs."""
        return self.step_ns == self.window_ns

    def starts_ns(self) -> np.ndarray:
        """Window START timestamps, shape (n_windows,), dtype uint64.

        Window *i* covers ``[starts_ns()[i], ends_ns()[i])`` (left-closed/
        right-open) on EVERY grid, overlapping or not.
        """
        offsets = np.arange(self.n_windows, dtype=np.uint64) * np.uint64(self.step_ns)
        return offsets + np.uint64(self.t0_ns)

    def ends_ns(self) -> np.ndarray:
        """Window END timestamps (exclusive), shape (n_windows,), dtype uint64 --
        `starts_ns() + window_ns`. On a tiling this equals the NEXT window's
        start; on an overlapping grid it does not."""
        return self.starts_ns() + np.uint64(self.window_ns)

    def edges_ns(self) -> np.ndarray:
        """Window edge timestamps, shape (n_windows + 1,), dtype uint64.

        Windows are left-closed/right-open: window *i* covers
        ``[edges_ns()[i], edges_ns()[i + 1])``.

        Raises:
            ValueError: on an OVERLAPPING grid (`hop_ns != window_ns`), where a
                single shared edge array cannot describe the windows at all --
                consecutive entries would be `hop_ns` apart, i.e. a different
                (shorter) window than the grid actually carries. Use
                `starts_ns()`/`ends_ns()`, which are defined on every grid.
        """
        if not self.is_tiling:
            raise ValueError(
                f"edges_ns() describes a non-overlapping tiling only, but this grid "
                f"has hop_ns={self.step_ns} != window_ns={self.window_ns} -- use "
                f"starts_ns()/ends_ns() instead"
            )
        offsets = np.arange(self.n_windows + 1, dtype=np.uint64) * np.uint64(self.window_ns)
        return offsets + np.uint64(self.t0_ns)


def _stream_end_ns(header: GantnerHeader) -> int:
    return header.t0_ns + round(header.n_frames / header.sample_rate_hz * 1e9)


def common_grid(
    headers: Sequence[GantnerHeader], window_s: float, hop_s: float | None = None
) -> WindowGrid:
    """Build a window grid spanning the INTERSECTION of every stream's [t0, t_end).

    The grid's t0 is the exact intersection start (not rounded to a window boundary);
    windows start every *hop_s* seconds from there, and only windows that fit ENTIRELY
    inside the intersection are emitted. Raises ValueError if the intersection is empty
    or spans fewer than one whole window.

    Args:
        headers: One header per stream; the grid spans their intersection.
        window_s: Window DURATION in seconds (how much data one window holds).
        hop_s: Spacing between consecutive window STARTS in seconds; `None`
            (the default) means `window_s`, i.e. the non-overlapping tiling.
            Must satisfy `0 < hop_s <= window_s` -- a hop LONGER than the window
            would skip samples entirely (this pipeline has no such use, and
            `coverage`'s "expected samples per window" accounting assumes every
            sample of the spanned interval belongs to some window).

    Raises:
        ValueError: empty intersection, an intersection shorter than one window,
            or a *hop_s* outside `(0, window_s]`.
    """
    if not headers:
        raise ValueError("common_grid requires at least one header")

    window_ns = round(window_s * 1e9)
    if hop_s is None:
        hop_ns = window_ns
    else:
        hop_ns = round(hop_s * 1e9)
        if hop_ns <= 0 or hop_ns > window_ns:
            raise ValueError(
                f"hop_s must satisfy 0 < hop_s <= window_s ({window_s!r}), got {hop_s!r}"
            )

    start_ns = max(h.t0_ns for h in headers)
    end_ns = min(_stream_end_ns(h) for h in headers)
    if end_ns <= start_ns:
        raise ValueError(
            f"empty intersection across {len(headers)} stream(s): "
            f"start_ns={start_ns} >= end_ns={end_ns}"
        )

    span_ns = end_ns - start_ns
    # (span - window) // hop + 1 is the count of window STARTS whose whole window
    # still fits inside the intersection. For hop == window it is identical to the
    # original `span // window` (both floor to the same integer for every span),
    # so the default grid is bit-identical to the pre-hop one.
    n_windows = (span_ns - window_ns) // hop_ns + 1 if span_ns >= window_ns else 0
    if n_windows == 0:
        raise ValueError(
            f"intersection duration ({span_ns} ns) is shorter than one "
            f"window ({window_ns} ns)"
        )
    return WindowGrid(
        t0_ns=start_ns, window_ns=window_ns, n_windows=n_windows, hop_ns=hop_ns
    )


def window_slices(ts_ns: np.ndarray, grid: WindowGrid) -> list[slice]:
    """Per-window sample slice into *ts_ns* (sorted, uint64 timestamps of one stream).

    Windows are left-closed/right-open. A window with no samples in range yields
    slice(i, i) (empty). On an OVERLAPPING grid consecutive slices genuinely share
    samples -- each window still holds one full `window_ns` worth of data.
    """
    lo = np.searchsorted(ts_ns, grid.starts_ns())
    hi = np.searchsorted(ts_ns, grid.ends_ns())
    return [slice(int(lo[i]), int(hi[i])) for i in range(grid.n_windows)]


def coverage(ts_ns: np.ndarray, grid: WindowGrid, rate_hz: float) -> np.ndarray:
    """Fraction of expected samples present per window, clipped to [0, 1].

    Expected count per window is ``rate_hz * window_s`` (the window DURATION, never
    the hop -- an overlapping grid's windows are just as long as a tiling's); the
    ratio is clipped so that duplicate/extra samples (or a rate underestimate)
    cannot push coverage above 1.0.
    """
    window_s = grid.window_ns / 1e9
    expected = rate_hz * window_s
    counts = np.array(
        [sl.stop - sl.start for sl in window_slices(ts_ns, grid)], dtype=np.float64
    )
    return np.clip(counts / expected, 0.0, 1.0)
