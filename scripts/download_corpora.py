"""Public-corpus acquisition CLI (package-4 spec D2, Task 2): downloads the
audio (MIMII) and vibration (CWRU, Paderborn KAt) corpora
`scripts/pretrain_tfc.py` (Task 3) pre-trains the TF-C encoders on, into
`--dest` (default `data/public/` -- gitignored via this repo's existing
blanket `data/` rule in `.gitignore`).

Network policy: `--dry-run` (recommended first call) prints the full
download plan -- every corpus's file list, URLs, licenses, and sha256
status -- and NEVER touches `urllib`; only a real (non-dry-run) invocation
opens a connection, and only for the corpus/corpora selected via `--corpus`.
Downloads never run in tests (`tests/test_download_corpora.py` monkeypatches
`urllib.request.urlopen`; its dry-run tests assert it is never even called).

sha256 policy: `_download_file` treats a declared hash two ways. A REAL
declared hash (every `_CORPUS_FILES` entry below -- mimii, cwru, paderborn
K001/K002 measured and transcribed after the first verified full download on
2026-07-16, paderborn K003-K006 likewise on 2026-07-21) is VERIFIED
against the freshly streamed file's computed hash -- on mismatch the corrupt
file is removed from disk, `ChecksumMismatchError` is raised, that corpus's
`MANIFEST.json` is never written, and `main` exits 1 with a message naming
the file (never a silent accept, never a traceback). The `_SHA256_TBD`
sentinel (the one sanctioned placeholder, for table additions that have not
yet been downloaded once -- currently none; the package-7 K003-K006 rows,
spec D5/A3.10, graduated to real hashes) is never "verified" against anything --
there is nothing yet to verify -- instead the computed hash is logged and
written into that corpus's `MANIFEST.json`, which is how a real sha256 first
becomes known, for a human to transcribe back into `_CORPUS_FILES` afterward
(exactly the flow that produced the hashes now in the table; the original
HEAD/metadata URL verification is documented in the Task-2 completion
report, `.superpowers/sdd/task-2-report.md`).

Corpus provenance (verified 2026-07-16, HEAD/metadata requests only, no
payload bytes fetched -- see the completion report for full detail):
  - mimii: Zenodo record 3384388 ("MIMII Dataset: Sound Dataset for
    Malfunctioning Industrial Machine Investigation and Inspection", Purohit
    et al. 2019, CC BY-SA 4.0), the 0 dB SNR "pump" machine type.
  - cwru: CWRU Bearing Data Center (engineering.case.edu/bearingdatacenter),
    Normal Baseline Data (4 loads) + a small 0.007"-fault-diameter subset of
    12k Drive End Bearing Fault Data (one file per fault location).
  - paderborn: Paderborn KAt-DataCenter
    (groups.uni-paderborn.de/kat/BearingDataCenter), the full healthy set
    K001-K006: K001-K002 since package-4 spec D2 (its named subset, verified
    2026-07-16); K003-K006 added by package-7 spec D5/A3.10 (same
    BearingDataCenter URL scheme -- NOT Zenodo -- sentinel-hashed until
    their first verified download).

Extraction: MIMII's zip is extracted via `zipfile` into
`<dest>/mimii/pump_0db/`. Paderborn's `.rar` files are extracted via `unar`
or `unrar` (whichever is found on PATH first) into a same-stem sibling
directory (`K001.rar` -> `K001/`); if NEITHER is on PATH, this script prints
precise manual-extraction instructions and continues -- a missing extractor
is not a download failure (spec D2's documented fallback). CWRU's `.mat`
files need no extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_SHA256_TBD = "TBD-first-download"
"""The one sanctioned sha256 placeholder (plan `docs/superpowers/plans/
2026-07-16-step2-package4-tfc.md`, Task-2 binding contract) -- see this
module's own docstring for how `_download_file` treats it."""

_MIMII_LICENSE = "CC BY-SA 4.0"
_CWRU_LICENSE = "academic-free"
_PADERBORN_LICENSE = "CC BY-NC 4.0"


@dataclass(frozen=True)
class _CorpusFile:
    url: str
    filename: str
    sha256: str
    license: str


_ManifestRow = dict[str, str | int]
"""One `MANIFEST.json` entry: url, filename, sha256 (always the COMPUTED
hash, never the `_SHA256_TBD` sentinel), license, downloaded_at (ISO 8601,
UTC), bytes."""


_CORPUS_FILES: dict[str, tuple[_CorpusFile, ...]] = {
    "mimii": (
        # Zenodo record 3384388, 0 dB SNR "pump" machine type (topical match
        # to this project's own pump-turbine target). Verified via the
        # Zenodo API + a HEAD request 2026-07-16: 7_869_431_302 bytes.
        _CorpusFile(
            url="https://zenodo.org/api/records/3384388/files/0_dB_pump.zip/content",
            filename="0_dB_pump.zip",
            sha256="1a2d416d2ad9d72f9ed3613ba78c623e170141d563c40db28322bcb9e56f8d91",
            license=_MIMII_LICENSE,
        ),
    ),
    "cwru": (
        # Normal Baseline Data (4 loads), engineering.case.edu/bearingdatacenter
        # -- verified via HEAD request 2026-07-16.
        _CorpusFile(  # Normal_0, 0 hp / 1797 rpm, 3_903_344 B
            url="https://engineering.case.edu/sites/default/files/97.mat",
            filename="97.mat",
            sha256="16bf48babcf1c7ac224bc1a81cd9eafdb27e42d5cf559761907e067e8eeadf3c",
            license=_CWRU_LICENSE,
        ),
        _CorpusFile(  # Normal_1, 1 hp / 1772 rpm, 7_742_720 B
            url="https://engineering.case.edu/sites/default/files/98.mat",
            filename="98.mat",
            sha256="37e6612c05e65c415dcfa2ab27a3fda648a5863160fa898b884a14743044e045",
            license=_CWRU_LICENSE,
        ),
        _CorpusFile(  # Normal_2, 2 hp / 1750 rpm, 15_503_928 B
            url="https://engineering.case.edu/sites/default/files/99.mat",
            filename="99.mat",
            sha256="4b97e6b5361f45efb6951dc3b1aebcdb3b89cb69d0f96d6f5c297dd9f45eee75",
            license=_CWRU_LICENSE,
        ),
        _CorpusFile(  # Normal_3, 3 hp / 1730 rpm, 7_770_624 B
            url="https://engineering.case.edu/sites/default/files/100.mat",
            filename="100.mat",
            sha256="88a5990cb541320e91505a1d72139e1993500ffe6e292a451011667f4138ca78",
            license=_CWRU_LICENSE,
        ),
        # 12k Drive End Bearing Fault Data, 0.007" fault diameter, load 0 hp
        # -- one file per fault location (a SMALL, diverse subset per spec
        # D2, not the full load x diameter x location matrix).
        _CorpusFile(  # IR007_0 (inner race), 2_910_768 B
            url="https://engineering.case.edu/sites/default/files/105.mat",
            filename="105.mat",
            sha256="f80b0ea04fd06b372a0eaec7c056543ea37e4bb4727a5b173d2a5bacd2aa9cab",
            license=_CWRU_LICENSE,
        ),
        _CorpusFile(  # B007_0 (ball), 2_942_112 B
            url="https://engineering.case.edu/sites/default/files/118.mat",
            filename="118.mat",
            sha256="b00628f8dd8d1d930af77fa465d1e5cdb385fe259489053f91f3680bda7f640e",
            license=_CWRU_LICENSE,
        ),
        _CorpusFile(  # OR007@6_0 (outer race @6:00), 2_928_192 B
            url="https://engineering.case.edu/sites/default/files/130.mat",
            filename="130.mat",
            sha256="35a095307d0971477049b343a1b5981dde465a58fb7f233ad89b035068c1717d",
            license=_CWRU_LICENSE,
        ),
    ),
    "paderborn": (
        # Healthy bearings K001-K002, groups.uni-paderborn.de/kat/BearingDataCenter
        # -- verified via HEAD request 2026-07-16.
        _CorpusFile(  # 173_881_721 B
            url="https://groups.uni-paderborn.de/kat/BearingDataCenter/K001.rar",
            filename="K001.rar",
            sha256="0f119ebdb28fb2f4d9fac1beb1319429f63f7ae1256c23c872f280f3560918e5",
            license=_PADERBORN_LICENSE,
        ),
        _CorpusFile(  # 161_981_588 B
            url="https://groups.uni-paderborn.de/kat/BearingDataCenter/K002.rar",
            filename="K002.rar",
            sha256="1040da8b74d169c4e8c8545afa335d7a3a320bcaf36a471250c3e434bc4caffd",
            license=_PADERBORN_LICENSE,
        ),
        # Healthy bearings K003-K006 (package-7 spec D5/A3.10): the REMAINING
        # Paderborn healthy set, same BearingDataCenter URL scheme as
        # K001/K002 above (NOT Zenodo -- only MIMII is Zenodo, A3.10). Hashes
        # measured and transcribed from the paderborn MANIFEST.json after the
        # first verified full download on 2026-07-21 (this module's own
        # docstring policy -- the same flow that produced K001/K002's hashes).
        # No downstream change is needed for these: `_extract_paderborn_rars`
        # iterates this tuple (so the new rars are extracted to
        # `K003/`..`K006/` automatically), and
        # `rowii.tfc.corpora.iter_windows_paderborn_dir` walks the paderborn
        # dir RECURSIVELY (`root.rglob("*.mat")`), picking the new extraction
        # dirs up with no loader change (verified 2026-07-21; pinned by
        # tests/test_download_corpora.py's synthetic-K003-tree test).
        _CorpusFile(  # 173_194_389 B
            url="https://groups.uni-paderborn.de/kat/BearingDataCenter/K003.rar",
            filename="K003.rar",
            sha256="d7d1edf6047fac088efe118198abd0856c5280dd8653496fbac9a5d877180f1e",
            license=_PADERBORN_LICENSE,
        ),
        _CorpusFile(  # 163_337_851 B
            url="https://groups.uni-paderborn.de/kat/BearingDataCenter/K004.rar",
            filename="K004.rar",
            sha256="97242542046cf3ce286c13e12d8df11f618a53365efe592750e2047256345333",
            license=_PADERBORN_LICENSE,
        ),
        _CorpusFile(  # 167_366_765 B
            url="https://groups.uni-paderborn.de/kat/BearingDataCenter/K005.rar",
            filename="K005.rar",
            sha256="2faf0ee4ee66055bcc066e5b6d9ed9fc44638b2c1ea397a279261103aa83caaa",
            license=_PADERBORN_LICENSE,
        ),
        _CorpusFile(  # 175_997_605 B
            url="https://groups.uni-paderborn.de/kat/BearingDataCenter/K006.rar",
            filename="K006.rar",
            sha256="250b8e0ef18082163ae4003cd7f50e6daa3061b3a095e226378910da55b78b94",
            license=_PADERBORN_LICENSE,
        ),
    ),
}

_READ_CHUNK_BYTES = 1 << 20  # 1 MiB per urllib .read() call
_PROGRESS_LOG_INTERVAL_BYTES = 100_000_000  # log every 100 MB streamed


class ChecksumMismatchError(RuntimeError):
    """A downloaded file's computed sha256 does not match its corpus-table
    declared value. Only possible for a REAL (non-sentinel) declared hash --
    see `_download_file`."""


def _format_gb(n_bytes: int) -> str:
    return f"{n_bytes / 1e9:.3f} GB"


def _stream_download(url: str, dest_path: Path) -> tuple[str, int]:
    """Stream *url* to *dest_path*, computing its sha256 as it goes.

    Resume-less: a plain `urllib.request.urlopen(url)` GET, no
    `Range`/`If-Range` request and no on-disk resume state -- an
    interrupted download must be restarted from scratch. Acceptable for
    this project's one-off, orchestrator-supervised corpus downloads (spec
    D2): simplicity over resumability. Progress is logged (INFO) every
    `_PROGRESS_LOG_INTERVAL_BYTES` (100 MB) streamed, so a multi-GB MIMII
    download shows liveness in a background log without flooding it.

    Returns:
        `(sha256_hexdigest, n_bytes_written)`.
    """
    hasher = hashlib.sha256()
    n_bytes = 0
    next_log_at = _PROGRESS_LOG_INTERVAL_BYTES
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as fh:
        while True:
            chunk = response.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            fh.write(chunk)
            hasher.update(chunk)
            n_bytes += len(chunk)
            if n_bytes >= next_log_at:
                logger.info(
                    "download_corpora: %s -- %s streamed", dest_path.name, _format_gb(n_bytes)
                )
                next_log_at += _PROGRESS_LOG_INTERVAL_BYTES
    return hasher.hexdigest(), n_bytes


def _download_file(spec: _CorpusFile, dest_dir: Path) -> _ManifestRow:
    """Download one corpus-table entry into *dest_dir*, returning its
    `MANIFEST.json` row (module docstring's sentinel-vs-real sha256 policy).

    Raises:
        ChecksumMismatchError: *spec.sha256* is a real (non-sentinel) hash
            and the downloaded file's computed sha256 does not match it. The
            mismatching file is REMOVED from disk before raising -- a
            corrupt/truncated/wrong download must never survive where a later
            run (or a human) could mistake it for a verified one.
    """
    dest_path = dest_dir / spec.filename
    computed_sha256, n_bytes = _stream_download(spec.url, dest_path)

    if spec.sha256 == _SHA256_TBD:
        logger.info(
            "download_corpora: %s sha256=%s (sentinel -- transcribe this into "
            "_CORPUS_FILES once verified)",
            spec.filename, computed_sha256,
        )
    elif computed_sha256 != spec.sha256:
        dest_path.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            f"{spec.filename}: sha256 mismatch -- expected {spec.sha256}, got {computed_sha256}"
        )

    return {
        "url": spec.url,
        "filename": spec.filename,
        "sha256": computed_sha256,
        "license": spec.license,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "bytes": n_bytes,
    }


def _write_manifest(corpus_dir: Path, rows: list[_ManifestRow]) -> None:
    manifest_path = corpus_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(rows, indent=2))


def _find_rar_extractor() -> str | None:
    """First of `unar`/`unrar` found on PATH, or `None` if neither is
    present (`_extract_paderborn_rars`'s fallback -- spec D2's documented
    rar-extraction policy)."""
    for name in ("unar", "unrar"):
        if shutil.which(name) is not None:
            return name
    return None


def _extract_mimii_zip(zip_path: Path, dest_root: Path) -> Path:
    """Extract *zip_path* into `dest_root/mimii/pump_0db/` (spec D2's fixed
    layout for the MIMII 0 dB pump zip), returning that directory."""
    extract_dir = dest_root / "mimii" / "pump_0db"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def _manual_paderborn_instructions(rar_path: Path, extract_dir: Path) -> str:
    return (
        f"download_corpora: neither 'unar' nor 'unrar' found on PATH -- cannot "
        f"extract {rar_path.name}. Install one (macOS: `brew install unar`; "
        f"Debian/Ubuntu: `apt install unar`) and re-run, or extract manually:\n"
        f"  unar -output-directory {extract_dir} {rar_path}\n"
        f"  # or: unrar x -y {rar_path} {extract_dir}/\n"
        f"{rar_path} is left in place, unextracted."
    )


def _extract_paderborn_rars(paderborn_dir: Path) -> None:
    """Extract every downloaded Paderborn `.rar` into a same-stem sibling
    directory (`K001.rar` -> `K001/`), via `unar`/`unrar` if either is on
    PATH. If neither is present, prints precise manual instructions
    (stderr) per file and LEAVES it unextracted -- the download step has
    already succeeded regardless (spec D2's documented fallback: a missing
    extractor is not a download failure).
    """
    extractor = _find_rar_extractor()
    for spec in _CORPUS_FILES["paderborn"]:
        rar_path = paderborn_dir / spec.filename
        extract_dir = paderborn_dir / Path(spec.filename).stem

        if extractor is None:
            print(_manual_paderborn_instructions(rar_path, extract_dir), file=sys.stderr)
            logger.warning(
                "download_corpora: no unar/unrar on PATH -- leaving %s unextracted", rar_path
            )
            continue

        extract_dir.mkdir(parents=True, exist_ok=True)
        if extractor == "unar":
            cmd = ["unar", "-quiet", "-output-directory", str(extract_dir), str(rar_path)]
        else:
            cmd = ["unrar", "x", "-y", str(rar_path), f"{extract_dir}/"]
        subprocess.run(cmd, check=True)
        logger.info("download_corpora: extracted %s -> %s (%s)", rar_path, extract_dir, extractor)


def _download_corpus(corpus: str, dest: Path) -> None:
    """Download every file in `_CORPUS_FILES[corpus]`, write that corpus's
    `MANIFEST.json`, then run the corpus-specific post-download step (MIMII:
    zipfile extraction; Paderborn: rar extraction if possible; CWRU: none,
    its `.mat` files are used as downloaded)."""
    corpus_dir = dest / corpus
    rows: list[_ManifestRow] = []
    for spec in _CORPUS_FILES[corpus]:
        logger.info("download_corpora: %s -> %s", spec.url, corpus_dir / spec.filename)
        rows.append(_download_file(spec, corpus_dir))
    _write_manifest(corpus_dir, rows)

    if corpus == "mimii":
        # Looped (rather than indexing the single current entry) so a future
        # second MIMII zip in the table is extracted too, not silently
        # ignored -- mirrors how the cwru/paderborn branches already iterate
        # `_CORPUS_FILES[corpus]` in full.
        for spec in _CORPUS_FILES["mimii"]:
            zip_path = corpus_dir / spec.filename
            extract_dir = _extract_mimii_zip(zip_path, dest)
            logger.info("download_corpora: extracted %s -> %s", spec.filename, extract_dir)
    elif corpus == "paderborn":
        _extract_paderborn_rars(corpus_dir)


def _print_dry_run(corpora: list[str], dest: Path) -> None:
    total_files = sum(len(_CORPUS_FILES[c]) for c in corpora)
    print(
        f"download_corpora --dry-run: {len(corpora)} corpus/corpora, "
        f"{total_files} file(s) -> {dest}"
    )
    for corpus in corpora:
        print(f"  {corpus}:")
        for spec in _CORPUS_FILES[corpus]:
            print(
                f"    {spec.filename}  <- {spec.url}  "
                f"(license={spec.license}, sha256={spec.sha256})"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the public TF-C pretraining corpora (MIMII pump 0 dB "
            "audio; CWRU + Paderborn KAt bearing vibration) into --dest, "
            "sha256-recorded per corpus in MANIFEST.json (package-4 spec D2)."
        )
    )
    parser.add_argument(
        "--corpus", choices=(*_CORPUS_FILES, "all"), default="all",
        help="Corpus to download (default: all).",
    )
    parser.add_argument(
        "--dest", type=Path, default=Path("data/public"),
        help="Destination root (default: data/public).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the full download plan (every corpus's files, URLs, sizes) "
             "and exit 0 -- never opens a network connection.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    corpora = list(_CORPUS_FILES) if args.corpus == "all" else [args.corpus]

    if args.dry_run:
        _print_dry_run(corpora, args.dest)
        return 0

    try:
        for corpus in corpora:
            _download_corpus(corpus, args.dest)
    except ChecksumMismatchError as exc:
        # Clean non-zero exit naming the file (the exception message carries
        # filename + both hashes), not a traceback: a checksum mismatch is an
        # expected operational failure (corrupted transfer, upstream file
        # replaced), and `_download_file` has already removed the corrupt
        # file; the failing corpus's MANIFEST.json is never written
        # (`_download_corpus` only writes it after every file succeeded).
        print(f"download_corpora: {exc}", file=sys.stderr)
        return 1

    print(f"download_corpora: done -- {len(corpora)} corpus/corpora -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
