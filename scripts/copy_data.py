"""Selective copy of the June-25 Rodundwerk II delivery into `ROWII_DATA_ROOT`.

Copies exactly the files Step 1 needs from a source tree shaped like
`~/Downloads/illwerke-250526-analysis`, preserving relative paths:

    20260626 Messung/TU/*.dat
    20260626 Messung/PU/*.dat
    20260626 Messung/Betriebsdaten/2026-06-25_*.dat
    Sensor_Anordnung_15062026.xlsx
    MANIFEST.md
    ROWII_Leistung_PU.jpg
    ROWII_Leistung_TU.jpg
    20260626 Messung/ROWII_Leistung.jpg

The five top-level provenance/reference files (the `.xlsx`, `MANIFEST.md`, the
two `ROWII_Leistung_*.jpg` screenshots, and the Messung-scoped power screenshot)
are each individually optional: a missing one only logs a warning, since none of
them is required for `run_step1.py` to operate. The `.dat` glob groups (TU, PU,
Betriebsdaten) are the actual pipeline inputs and are copied file-by-file as
found (an empty glob for one group is not itself an error -- discovery/detect
downstream will fail loudly if a run's streams are genuinely absent).

Already-present destination files with an EQUAL size are skipped (idempotent
re-runs after a partial copy do not re-transfer ~35 GB). A destination file that
exists with a DIFFERENT size is treated as stale and re-copied -- this is not
just a resume/skip cache, so no hash comparison is needed for this repo's use
case (a single trusted source tree, copied once per machine).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

_DISK_SAFETY_FACTOR = 1.2

_MESSUNG_DIR = "20260626 Messung"
_DAT_GLOB_GROUPS = (
    f"{_MESSUNG_DIR}/TU",
    f"{_MESSUNG_DIR}/PU",
)
_BETRIEBSDATEN_DIR = f"{_MESSUNG_DIR}/Betriebsdaten"
_BETRIEBSDATEN_PREFIX = "2026-06-25_"

_OPTIONAL_TOP_LEVEL_FILES = (
    "Sensor_Anordnung_15062026.xlsx",
    "MANIFEST.md",
    "ROWII_Leistung_PU.jpg",
    "ROWII_Leistung_TU.jpg",
    f"{_MESSUNG_DIR}/ROWII_Leistung.jpg",
)


@dataclass(frozen=True)
class _PlannedFile:
    relpath: str
    """Path relative to --source / --dest, using '/' separators (manifest-stable)."""
    src: Path
    bytes: int


_ManifestRow = dict[str, str | int]
"""One `copy_manifest.json` entry: `{"relpath": str, "bytes": int}`."""


def _plan_dat_group(source: Path, group_reldir: str) -> list[_PlannedFile]:
    group_dir = source / group_reldir
    if not group_dir.is_dir():
        logger.warning("copy_data: %s not found under source, skipping this group", group_reldir)
        return []
    planned = []
    for p in sorted(group_dir.glob("*.dat")):
        rel = f"{group_reldir}/{p.name}"
        planned.append(_PlannedFile(relpath=rel, src=p, bytes=p.stat().st_size))
    return planned


def _plan_betriebsdaten(source: Path) -> list[_PlannedFile]:
    group_dir = source / _BETRIEBSDATEN_DIR
    if not group_dir.is_dir():
        logger.warning(
            "copy_data: %s not found under source, skipping Betriebsdaten", _BETRIEBSDATEN_DIR
        )
        return []
    planned = []
    for p in sorted(group_dir.glob(f"{_BETRIEBSDATEN_PREFIX}*.dat")):
        rel = f"{_BETRIEBSDATEN_DIR}/{p.name}"
        planned.append(_PlannedFile(relpath=rel, src=p, bytes=p.stat().st_size))
    return planned


def _plan_optional_top_level(source: Path) -> list[_PlannedFile]:
    planned = []
    for rel in _OPTIONAL_TOP_LEVEL_FILES:
        p = source / rel
        if not p.is_file():
            logger.warning("copy_data: optional file %s not found under source, skipping", rel)
            continue
        planned.append(_PlannedFile(relpath=rel, src=p, bytes=p.stat().st_size))
    return planned


def build_copy_plan(source: Path) -> list[_PlannedFile]:
    """Resolve the exact copy plan (spec §3) against a concrete *source* tree."""
    planned: list[_PlannedFile] = []
    for group in _DAT_GLOB_GROUPS:
        planned.extend(_plan_dat_group(source, group))
    planned.extend(_plan_betriebsdaten(source))
    planned.extend(_plan_optional_top_level(source))
    return planned


def _format_gb(n_bytes: int) -> str:
    return f"{n_bytes / 1e9:.3f} GB"


def _print_dry_run(planned: list[_PlannedFile]) -> None:
    print(f"copy_data --dry-run: {len(planned)} file(s) would be copied:")
    for pf in planned:
        print(f"  {pf.relpath}  ({pf.bytes} bytes)")
    total = sum(pf.bytes for pf in planned)
    print(f"Total: {total} bytes ({_format_gb(total)})")


def _needs_copy(pf: _PlannedFile, dest_root: Path) -> bool:
    dest_path = dest_root / pf.relpath
    if not dest_path.exists():
        return True
    return dest_path.stat().st_size != pf.bytes


def _has_enough_disk_space(dest_root: Path, remaining_bytes: int) -> bool:
    """True iff free disk at *dest_root* covers `_DISK_SAFETY_FACTOR` x *remaining_bytes*.

    Prints a diagnostic to stderr (free vs. required) when the check fails; the
    caller (`main`) turns that into the documented exit code 2. Kept as a plain
    predicate (no `SystemExit`) so it composes cleanly inside `_copy_files`,
    which itself must remain a pure helper callable from tests without needing
    to catch a control-flow exception raised several frames below `main`.
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest_root).free
    required = int(remaining_bytes * _DISK_SAFETY_FACTOR)
    if free < required:
        print(
            f"copy_data: refusing to copy -- {_format_gb(free)} free at {dest_root}, "
            f"but {_format_gb(required)} required "
            f"({_DISK_SAFETY_FACTOR}x the {_format_gb(remaining_bytes)} remaining to copy)",
            file=sys.stderr,
        )
        return False
    return True


class InsufficientDiskSpaceError(RuntimeError):
    """Raised by `_copy_files` when the destination lacks the required free space."""


def _copy_files(
    planned: list[_PlannedFile], dest_root: Path
) -> tuple[int, int, list[_ManifestRow]]:
    """Copy every file in *planned* that needs it. Returns (n_copied, n_skipped, manifest_rows).

    Raises:
        InsufficientDiskSpaceError: if free disk at *dest_root* is below
            `_DISK_SAFETY_FACTOR` x the bytes remaining to copy (`main` maps
            this to exit code 2, per the documented CLI contract).
    """
    to_copy = [pf for pf in planned if _needs_copy(pf, dest_root)]
    remaining_bytes = sum(pf.bytes for pf in to_copy)
    if not _has_enough_disk_space(dest_root, remaining_bytes):
        raise InsufficientDiskSpaceError(
            f"insufficient free disk space at {dest_root} for {remaining_bytes} remaining bytes"
        )

    n_copied = 0
    n_skipped = 0
    manifest_rows: list[_ManifestRow] = []
    for pf in planned:
        dest_path = dest_root / pf.relpath
        if _needs_copy(pf, dest_root):
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pf.src, dest_path)
            n_copied += 1
        else:
            n_skipped += 1
        manifest_rows.append({"relpath": pf.relpath, "bytes": pf.bytes})
    return n_copied, n_skipped, manifest_rows


def _write_manifest(dest_root: Path, manifest_rows: list[_ManifestRow]) -> None:
    manifest_path = dest_root / "copy_manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the Step-1 subset of a June-25 Rodundwerk II delivery tree into "
            "ROWII_DATA_ROOT (spec §3)."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("~/Downloads/illwerke-250526-analysis").expanduser(),
        help="Source tree root (default: ~/Downloads/illwerke-250526-analysis).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination root (default: Config.data_root from load_config()).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact file list and total size; copy nothing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    source: Path = args.source
    dest: Path = args.dest if args.dest is not None else load_config().data_root

    if not source.is_dir():
        print(f"copy_data: source directory not found: {source}", file=sys.stderr)
        return 2

    planned = build_copy_plan(source)

    if args.dry_run:
        _print_dry_run(planned)
        return 0

    try:
        n_copied, n_skipped, manifest_rows = _copy_files(planned, dest)
    except InsufficientDiskSpaceError:
        return 2
    _write_manifest(dest, manifest_rows)

    total_bytes = sum(pf.bytes for pf in planned)
    print(
        f"copy_data: {n_copied} copied, {n_skipped} skipped (already present), "
        f"{_format_gb(total_bytes)} total across {len(planned)} planned file(s) -> {dest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
