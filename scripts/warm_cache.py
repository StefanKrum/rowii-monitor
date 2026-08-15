"""Cache warm-up CLI: pre-populate `results/cache/<run>--<variant>.npz` for one or
more (run, variant) combinations.

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

A combo whose own `prepare_run` raises `RuntimeError` (run too short/sparse for
the variant) or `KeyError` (a stream the variant needs is entirely absent from
the run) is logged as a WARNING and SKIPPED -- the remaining combos still run,
per the same catch-and-skip principle `scripts/run_step2.py`'s sweep
orchestrators already apply to `prepare_run` failures. The failure is still
surfaced in the exit code: 1 if ANY combo failed, 0 only when every combo
succeeded.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.config import Config, load_config  # noqa: E402
from rowii.io.dataset import RecordingIndex, discover  # noqa: E402
from rowii.pipeline import (  # noqa: E402
    _BEATS_INSTALL_HINT,
    _STUDENT_INSTALL_HINT,
    _TFC_INSTALL_HINT,
    _cache_npz_path,
    _is_beats_variant,
    _is_student_variant,
    _is_tfc_variant,
    prepare_run,
)

logger = logging.getLogger(__name__)

_DEFAULT_RUNS: tuple[str, ...] = (
    # The 27.06 day tree gap-splits into THREE discovered runs (>15-min pauses between
    # its PU/PH blocks -- `rowii.io.dataset._split_on_gaps`/`_group_name`); a bare
    # "270626-pu_ph_pu_ph_pu_ph" matches nothing. Only "-1" is worth warming: "-2" and
    # "-3" are negligible single-fragment ~12-min orphans, and "-2" has NO
    # RAWGeneratorMic__0 stream at all (real 2026-07-15 warm-up finding), so it is
    # structurally impossible for every audio-bearing variant (`build_run_grid`'s
    # `run.files[stream]` raises KeyError) -- deliberately excluded from the default.
    "250526-tu", "290626-tu", "010726-tu_ph_tu", "270626-pu_ph_pu_ph_pu_ph-1",
)
_DEFAULT_VARIANTS: tuple[str, ...] = ("audio-beats", "fusion-beats")
_VARIANT_CHOICES: tuple[str, ...] = (
    "audio", "vibration", "fusion", "audio-beats", "fusion-beats",
    "audio-tfc", "vibration-tfc", "audio-student", "logmel",
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


def _import_tfc_or_exit(cfg: Config, variant: str) -> None:
    """Mirrors `scripts/run_step1.py`'s own private helper of the same name
    (duplicated, not imported -- see `_import_beats_or_exit`'s docstring). Extends
    the beats-import-guard pattern (package-4 spec D4): torch missing (checked
    first) -> SystemExit naming the shared `[beats]` extra; else the ONE checkpoint
    relevant to *variant* itself missing -> SystemExit naming its own env var."""
    try:
        import rowii.tfc.wrapper  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"TF-C featurizer not available ({exc}); {_TFC_INSTALL_HINT}"
        ) from exc
    if variant == "audio-tfc":
        checkpoint, env_var = cfg.tfc_audio_checkpoint, "ROWII_TFC_AUDIO_CHECKPOINT"
    else:
        checkpoint, env_var = cfg.tfc_vib_checkpoint, "ROWII_TFC_VIB_CHECKPOINT"
    if checkpoint is None:
        raise SystemExit(f"variant {variant!r} needs {env_var} set; {_TFC_INSTALL_HINT}")


def _import_student_or_exit(cfg: Config) -> None:
    """Mirrors `_import_tfc_or_exit` above (package-5 spec D5), simplified: the
    distilled student has only ONE checkpoint (unlike TF-C's two independent
    branches), so there is no variant-based checkpoint selection -- torch
    missing (checked first) -> SystemExit naming the shared `[beats]` extra;
    else `cfg.student_checkpoint` missing -> SystemExit naming
    ROWII_STUDENT_CHECKPOINT. Duplicated (not imported) -- see
    `_import_beats_or_exit`'s docstring."""
    try:
        import rowii.adapt.student  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"Student featurizer not available ({exc}); {_STUDENT_INSTALL_HINT}"
        ) from exc
    if cfg.student_checkpoint is None:
        raise SystemExit(
            f"variant 'audio-student' needs ROWII_STUDENT_CHECKPOINT set; "
            f"{_STUDENT_INSTALL_HINT}"
        )


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
    for variant in sorted({v for _, v in combos if _is_tfc_variant(v)}):
        _import_tfc_or_exit(cfg, variant)
    if any(_is_student_variant(variant) for _, variant in combos):
        _import_student_or_exit(cfg)

    n_failed = 0
    for run_name, variant in combos:
        run = by_name[run_name]
        logger.info("warm_cache: starting %s x %s", run_name, variant)
        t0 = time.monotonic()
        try:
            prepare_run(run, variant, cfg, use_cache=True)
        except (RuntimeError, KeyError) as exc:
            # Same catch-and-skip principle as scripts/run_step2.py's sweep
            # orchestrators (its _run_within_day_for_run): ONE bad combo must never
            # kill the rest of the batch. RuntimeError = run too short/sparse for this
            # variant (rowii.pipeline.compute_validity_mask); KeyError = a stream the
            # variant needs is entirely absent from the run (build_run_grid's
            # run.files[stream] -- real 2026-07-15 case: 270626-pu_ph_pu_ph_pu_ph-2,
            # an orphan fragment with no RAWGeneratorMic__0 at all, killed the first
            # real warm-up after 8/12 combos before this guard existed).
            n_failed += 1
            logger.warning(
                "warm_cache: %s x %s FAILED (%s: %s) -- skipping, continuing with the "
                "next combo",
                run_name, variant, type(exc).__name__, exc,
            )
            continue
        elapsed_s = time.monotonic() - t0
        cache_path = _cache_npz_path(cfg.results_root, run_name, variant)
        size_bytes = cache_path.stat().st_size if cache_path.is_file() else -1
        logger.info(
            "warm_cache: %s x %s done in %.1fs -> %s (%d bytes)",
            run_name, variant, elapsed_s, cache_path, size_bytes,
        )

    if n_failed:
        print(
            f"warm_cache: warmed {len(combos) - n_failed}/{len(combos)} combo(s), "
            f"{n_failed} failed (see warnings above)"
        )
        return 1
    print(f"warm_cache: warmed {len(combos)} combo(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
