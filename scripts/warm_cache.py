"""Cache warm-up CLI: pre-populate `results/cache/<run>--<variant>.npz` for one or
more (run, variant) combinations (Step-2 package 2, design spec `docs/superpowers/
specs/2026-07-15-step2-scarcity-crossday-beats-design.md` D4, plan `docs/
superpowers/plans/2026-07-15-step2-scarcity-crossday-beats.md` Task 6).

BEATs variants (`audio-beats`, `fusion-beats`) run a frozen transformer over hours
of recorded audio -- expensive relative to handcrafted features. `rowii.pipeline.
prepare_run`'s on-disk cache (keyed by a content fingerprint; see that module's
docstring) means this extraction only needs to happen ONCE per (run, variant):
every later `scripts/run_step2.py` invocation against the same combo is then a
cache hit. This script exists so that one-off extraction can run in the
background (design spec D4: "cache warm-up first, in background, before other
sweeps need it") while the rest of this package's tasks are still being
implemented against cheap handcrafted variants -- the reason this task is
scheduled BEFORE those in execution order: its deliverable (a warm cache) gates a
long-running background computation that should start as early as possible.

Bootstrapping (config/env loading via `rowii.config.load_config`, run discovery
via `rowii.io.dataset.discover`, the beats-import guard) mirrors `scripts/
run_step2.py`'s own `main()`. Unlike that script, `--runs`/`--variants` here are
always EXPLICIT lists (no "all" sentinel): an unknown run name is a hard usage
error (exit 2, listing every discovered run) rather than a warn-and-skip, since
silently warming the wrong (or a typo'd) run's cache would defeat the point of
running this ahead of time. That check applies regardless of `--dry-run` (only
the beats-import guard itself is dry-run-exempt, per this module's own `main`).
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.config import load_config  # noqa: E402
from rowii.io.dataset import RecordingIndex, discover  # noqa: E402
from rowii.pipeline import (  # noqa: E402
    _BEATS_INSTALL_HINT,
    _cache_npz_path,
    _is_beats_variant,
    prepare_run,
)

logger = logging.getLogger(__name__)

_DEFAULT_RUNS: tuple[str, ...] = (
    # The 27.06 day tree gap-splits into THREE discovered runs (>15-min pauses between
    # its PU/PH blocks -- `rowii.io.dataset._split_on_gaps`/`_group_name`), so all three
    # are listed individually; a bare "270626-pu_ph_pu_ph_pu_ph" matches nothing.
    "250526-tu", "290626-tu", "010726-tu_ph_tu",
    "270626-pu_ph_pu_ph_pu_ph-1", "270626-pu_ph_pu_ph_pu_ph-2", "270626-pu_ph_pu_ph_pu_ph-3",
)
_DEFAULT_VARIANTS: tuple[str, ...] = ("audio-beats", "fusion-beats")
_VARIANT_CHOICES: tuple[str, ...] = (
    "audio", "vibration", "fusion", "audio-beats", "fusion-beats",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Warm rowii.pipeline.prepare_run's on-disk feature cache "
            "(results/cache/<run>--<variant>.npz) for one or more run x variant "
            "combinations, so BEATs' expensive feature extraction never blocks a "
            "later scripts/run_step2.py sweep (module docstring)."
        )
    )
    parser.add_argument(
        "--runs", nargs="+", default=list(_DEFAULT_RUNS), metavar="RUN",
        help=f"Run name(s) to warm (default: {' '.join(_DEFAULT_RUNS)}).",
    )
    parser.add_argument(
        "--variants", nargs="+", default=list(_DEFAULT_VARIANTS), choices=_VARIANT_CHOICES,
        metavar="VARIANT",
        help=f"Variant(s) to warm (default: {' '.join(_DEFAULT_VARIANTS)}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the run x variant combo list and exit 0 -- never imports "
             "beats/torch and never calls prepare_run.",
    )
    return parser


def _import_beats_or_exit() -> None:
    """Mirrors `scripts/run_step2.py`'s own private helper of the same name
    (duplicated rather than imported -- that module's own docstring explains why
    one script must not depend on a SIBLING script's internals)."""
    try:
        import rowii.signals.beats  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"BEATs featurizer not available ({exc}); {_BEATS_INSTALL_HINT}"
        ) from exc


def _unknown_run_names(names: list[str], index: RecordingIndex) -> list[str]:
    """Names in *names* with no matching discovered run, de-duplicated, in the
    order first seen -- empty if every name resolves."""
    known = {r.name for r in index.runs}
    return list(dict.fromkeys(n for n in names if n not in known))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config()
    index = discover(cfg.data_root)
    by_name = {r.name: r for r in index.runs}

    unknown = _unknown_run_names(args.runs, index)
    if unknown:
        available = ", ".join(sorted(by_name)) or "(none discovered)"
        print(
            f"warm_cache: unknown run name(s): {', '.join(unknown)}; "
            f"available runs: {available}",
            file=sys.stderr,
        )
        return 2

    combos = list(itertools.product(args.runs, args.variants))

    if args.dry_run:
        print(f"warm_cache --dry-run: {len(combos)} combo(s):")
        for run_name, variant in combos:
            print(f"  {run_name} x {variant}")
        return 0

    if any(_is_beats_variant(variant) for _, variant in combos):
        _import_beats_or_exit()

    for run_name, variant in combos:
        run = by_name[run_name]
        logger.info("warm_cache: starting %s x %s", run_name, variant)
        t0 = time.monotonic()
        prepare_run(run, variant, cfg, use_cache=True)
        elapsed_s = time.monotonic() - t0
        cache_path = _cache_npz_path(cfg.results_root, run_name, variant)
        size_bytes = cache_path.stat().st_size if cache_path.is_file() else -1
        logger.info(
            "warm_cache: %s x %s done in %.1fs -> %s (%d bytes)",
            run_name, variant, elapsed_s, cache_path, size_bytes,
        )

    print(f"warm_cache: warmed {len(combos)} combo(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
