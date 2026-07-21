"""Tests for `scripts/download_corpora.py` (package-4 spec D2, Task 2; grown
by package-7 spec D5/A3.10, Task 7 -- sections 7/8 below): CLI-level tests
with a monkeypatched `urllib.request.urlopen` -- NO real network access
anywhere in this file. `--dry-run` tests additionally assert `urlopen` is
never even CALLED, via a raising stub (module docstring's own network
policy: dry-run must never open a connection).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from rowii.tfc.corpora import iter_windows_paderborn_dir

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download_corpora  # noqa: E402
import pretrain_tfc  # noqa: E402


class _FakeResponse:
    """Minimal stand-in for `http.client.HTTPResponse`: a context manager
    with a chunked `.read(n)`, backed by in-memory *data*."""

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _raise_if_called(*args: object, **kwargs: object) -> None:
    raise AssertionError("urllib.request.urlopen must not be called here")


# ---------------------------------------------------------------------------
# 1. --help / parser basics
# ---------------------------------------------------------------------------


def test_help_exits_zero_and_documents_every_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        download_corpora.main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--corpus", "--dest", "--dry-run"):
        assert flag in out, f"missing {flag!r} in --help output"


def test_build_parser_defaults() -> None:
    args = download_corpora.build_parser().parse_args([])
    assert args.corpus == "all"
    assert args.dest == Path("data/public")
    assert args.dry_run is False


# ---------------------------------------------------------------------------
# 2. --dry-run: prints the plan, exits 0, NEVER touches the network
# ---------------------------------------------------------------------------


def test_dry_run_prints_all_corpora_and_never_touches_network(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(download_corpora.urllib.request, "urlopen", _raise_if_called)

    exit_code = download_corpora.main(["--dest", str(tmp_path), "--dry-run"])

    assert exit_code == 0
    out = capsys.readouterr().out
    for corpus in ("mimii", "cwru", "paderborn"):
        assert corpus in out
    for files in download_corpora._CORPUS_FILES.values():
        for spec in files:
            assert spec.filename in out
            assert spec.url in out


def test_dry_run_single_corpus_prints_only_that_corpus(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(download_corpora.urllib.request, "urlopen", _raise_if_called)

    exit_code = download_corpora.main(["--corpus", "cwru", "--dest", str(tmp_path), "--dry-run"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "cwru" in out
    assert "mimii" not in out
    assert "paderborn" not in out


# ---------------------------------------------------------------------------
# 3. Unknown corpus: argparse `choices=` -> SystemExit(2)
# ---------------------------------------------------------------------------


def test_unknown_corpus_exits_2(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        download_corpora.main(["--corpus", "bogus-corpus"])

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "bogus-corpus" in err


# ---------------------------------------------------------------------------
# 4. Sentinel sha256: "compute and print, then update manifest" -- never
#    "verify" (plan's Task-2 binding contract).
# ---------------------------------------------------------------------------


def test_download_file_with_sentinel_sha256_computes_and_returns_real_hash(
    monkeypatch, tmp_path
) -> None:
    payload = b"synthetic corpus bytes" * 1000
    monkeypatch.setattr(
        download_corpora.urllib.request, "urlopen", lambda url: _FakeResponse(payload)
    )

    spec = download_corpora._CorpusFile(
        url="https://example.invalid/fake.bin",
        filename="fake.bin",
        sha256=download_corpora._SHA256_TBD,
        license="TEST",
    )

    row = download_corpora._download_file(spec, tmp_path)

    expected = hashlib.sha256(payload).hexdigest()
    assert row["sha256"] == expected
    assert row["sha256"] != download_corpora._SHA256_TBD
    assert row["bytes"] == len(payload)
    assert row["url"] == spec.url
    assert row["license"] == "TEST"
    assert (tmp_path / "fake.bin").read_bytes() == payload


def test_real_declared_sha256_matching_passes(monkeypatch, tmp_path) -> None:
    payload = b"deterministic content"
    real_hash = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        download_corpora.urllib.request, "urlopen", lambda url: _FakeResponse(payload)
    )

    spec = download_corpora._CorpusFile(
        url="https://example.invalid/fake.bin",
        filename="fake.bin",
        sha256=real_hash,
        license="TEST",
    )

    row = download_corpora._download_file(spec, tmp_path)
    assert row["sha256"] == real_hash


def test_real_declared_sha256_mismatch_raises_and_removes_the_corrupt_file(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        download_corpora.urllib.request, "urlopen", lambda url: _FakeResponse(b"actual bytes")
    )

    spec = download_corpora._CorpusFile(
        url="https://example.invalid/fake.bin",
        filename="fake.bin",
        sha256="0" * 64,
        license="TEST",
    )

    with pytest.raises(download_corpora.ChecksumMismatchError):
        download_corpora._download_file(spec, tmp_path)

    # The mismatching (corrupt/truncated/wrong) file must not survive on disk
    # for a later run to mistake for a good download.
    assert not (tmp_path / "fake.bin").exists()


# ---------------------------------------------------------------------------
# 5. End-to-end (monkeypatched network) corpus download: manifest + extraction.
#
# These tests monkeypatch `_CORPUS_FILES`' per-corpus tuples with TEST-OWNED
# entries rather than downloading against the shipped table: the shipped
# table's sha256 values are the REAL, measured hashes of the actual multi-GB
# corpus files (transcribed after the first verified download), which a tiny
# synthetic payload can never match. Sentinel-sha entries exercise the
# compute-and-record flow; a real-sha entry whose hash IS the fake payload's
# exercises the verify flow -- either way the tests stay independent of
# whatever sha state the shipped table happens to be in.
# ---------------------------------------------------------------------------


def test_download_corpus_cwru_writes_manifest_with_computed_hashes(monkeypatch, tmp_path) -> None:
    payload = b"x" * 128
    monkeypatch.setattr(
        download_corpora.urllib.request, "urlopen", lambda url: _FakeResponse(payload)
    )
    monkeypatch.setitem(
        download_corpora._CORPUS_FILES,
        "cwru",
        (
            download_corpora._CorpusFile(
                url="https://example.invalid/a.mat",
                filename="a.mat",
                sha256=download_corpora._SHA256_TBD,
                license="academic-free",
            ),
            download_corpora._CorpusFile(
                url="https://example.invalid/b.mat",
                filename="b.mat",
                sha256=download_corpora._SHA256_TBD,
                license="academic-free",
            ),
        ),
    )

    download_corpora._download_corpus("cwru", tmp_path)

    manifest_path = tmp_path / "cwru" / "MANIFEST.json"
    assert manifest_path.is_file()
    rows = json.loads(manifest_path.read_text())
    assert len(rows) == 2
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    for row in rows:
        assert row["sha256"] == expected_sha256  # computed, never the sentinel
        assert row["license"] == "academic-free"
        assert row["bytes"] == 128
        assert "downloaded_at" in row


def test_download_corpus_mimii_extracts_zip_into_pump_0db(monkeypatch, tmp_path) -> None:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("pump/id_00/normal/00000000.wav", b"fake-wav-bytes")
    payload = zip_buf.getvalue()

    monkeypatch.setattr(
        download_corpora.urllib.request, "urlopen", lambda url: _FakeResponse(payload)
    )
    # Real-sha entry matching the fake payload: this test doubles as the
    # verify-flow SUCCESS case (declared == computed -> no error, manifest
    # written, extraction proceeds).
    monkeypatch.setitem(
        download_corpora._CORPUS_FILES,
        "mimii",
        (
            download_corpora._CorpusFile(
                url="https://example.invalid/0_dB_pump.zip",
                filename="0_dB_pump.zip",
                sha256=hashlib.sha256(payload).hexdigest(),
                license="CC BY-SA 4.0",
            ),
        ),
    )

    download_corpora._download_corpus("mimii", tmp_path)

    extracted = tmp_path / "mimii" / "pump_0db" / "pump" / "id_00" / "normal" / "00000000.wav"
    assert extracted.is_file()
    assert extracted.read_bytes() == b"fake-wav-bytes"


def test_download_corpus_paderborn_without_extractor_prints_manual_instructions_and_continues(
    monkeypatch, tmp_path, capsys, caplog
) -> None:
    monkeypatch.setattr(
        download_corpora.urllib.request, "urlopen", lambda url: _FakeResponse(b"fake-rar-bytes")
    )
    monkeypatch.setattr(download_corpora.shutil, "which", lambda name: None)
    monkeypatch.setitem(
        download_corpora._CORPUS_FILES,
        "paderborn",
        (
            download_corpora._CorpusFile(
                url="https://example.invalid/K001.rar",
                filename="K001.rar",
                sha256=download_corpora._SHA256_TBD,
                license="CC BY-NC 4.0",
            ),
        ),
    )

    with caplog.at_level(logging.WARNING):
        download_corpora._download_corpus("paderborn", tmp_path)  # must not raise

    manifest_path = tmp_path / "paderborn" / "MANIFEST.json"
    assert manifest_path.is_file()  # the download itself still succeeded
    assert (tmp_path / "paderborn" / "K001.rar").is_file()

    err = capsys.readouterr().err
    assert "unar" in err and "unrar" in err
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("K001.rar" in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# 6. Verify-flow FAILURE semantics (pinned): a recorded-sha mismatch aborts
#    the corpus with a clean non-zero exit naming the file, the partial
#    (corrupt) file is removed, and NO MANIFEST.json is written.
# ---------------------------------------------------------------------------


def test_sha_mismatch_exits_nonzero_removes_file_and_writes_no_manifest(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        download_corpora.urllib.request, "urlopen", lambda url: _FakeResponse(b"corrupted bytes")
    )
    monkeypatch.setitem(
        download_corpora._CORPUS_FILES,
        "cwru",
        (
            download_corpora._CorpusFile(
                url="https://example.invalid/a.mat",
                filename="a.mat",
                sha256="0" * 64,  # real (non-sentinel) hash that cannot match
                license="academic-free",
            ),
        ),
    )

    exit_code = download_corpora.main(["--corpus", "cwru", "--dest", str(tmp_path)])

    assert exit_code == 1  # clean non-zero exit, not a traceback
    err = capsys.readouterr().err
    assert "a.mat" in err
    assert "sha256 mismatch" in err
    assert not (tmp_path / "cwru" / "a.mat").exists()  # corrupt file removed
    assert not (tmp_path / "cwru" / "MANIFEST.json").exists()  # nothing recorded


# ---------------------------------------------------------------------------
# 7. Package-7 K003-K006 additions (spec D5/A3.10, Task 7): the REMAINING
#    Paderborn healthy bearings join the table -- same BearingDataCenter URL
#    scheme as K001/K002 (NOT Zenodo; only MIMII is Zenodo -- A3.10), each
#    carrying the `_SHA256_TBD` sentinel until its first verified download
#    (the execution task's live-HEAD + compute-and-transcribe procedure;
#    downloads never run in tests). Growing the table must never touch a
#    pre-existing entry -- the regression pin below holds every established
#    URL/sha256 byte-identical.
# ---------------------------------------------------------------------------

_BEARINGDATACENTER_BASE = "https://groups.uni-paderborn.de/kat/BearingDataCenter"


def _paderborn_entry(filename: str) -> download_corpora._CorpusFile:
    matches = [
        spec for spec in download_corpora._CORPUS_FILES["paderborn"] if spec.filename == filename
    ]
    assert len(matches) == 1, f"expected exactly one paderborn entry {filename!r}, got {matches}"
    return matches[0]


@pytest.mark.parametrize("stem", ["K003", "K004", "K005", "K006"])
def test_paderborn_k003_k006_use_bearingdatacenter_urls(stem: str) -> None:
    entry = _paderborn_entry(f"{stem}.rar")

    assert entry.url == f"{_BEARINGDATACENTER_BASE}/{stem}.rar"  # A3.10: NOT Zenodo
    assert entry.license == "CC BY-NC 4.0"


@pytest.mark.parametrize("stem", ["K003", "K004", "K005", "K006"])
def test_paderborn_k003_k006_carry_the_sha256_sentinel(stem: str) -> None:
    # Real hashes are transcribed only after the first verified download at
    # execution time -- never measured (or faked) in tests.
    assert _paderborn_entry(f"{stem}.rar").sha256 == download_corpora._SHA256_TBD


def test_dry_run_lists_k003_k006(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(download_corpora.urllib.request, "urlopen", _raise_if_called)

    exit_code = download_corpora.main(
        ["--corpus", "paderborn", "--dest", str(tmp_path), "--dry-run"]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    for stem in ("K001", "K002", "K003", "K004", "K005", "K006"):
        assert f"{stem}.rar" in out, f"{stem}.rar missing from the dry-run plan"


def test_established_entries_unchanged_by_k003_k006_addition() -> None:
    """Regression pin (Task 7): every pre-package-7 entry stays byte-identical
    (URL + measured sha256 transcribed literally) -- the table only GROWS."""
    files = download_corpora._CORPUS_FILES

    assert [(spec.url, spec.sha256) for spec in files["mimii"]] == [
        (
            "https://zenodo.org/api/records/3384388/files/0_dB_pump.zip/content",
            "1a2d416d2ad9d72f9ed3613ba78c623e170141d563c40db28322bcb9e56f8d91",
        ),
    ]
    assert [(spec.url, spec.sha256) for spec in files["cwru"]] == [
        (
            "https://engineering.case.edu/sites/default/files/97.mat",
            "16bf48babcf1c7ac224bc1a81cd9eafdb27e42d5cf559761907e067e8eeadf3c",
        ),
        (
            "https://engineering.case.edu/sites/default/files/98.mat",
            "37e6612c05e65c415dcfa2ab27a3fda648a5863160fa898b884a14743044e045",
        ),
        (
            "https://engineering.case.edu/sites/default/files/99.mat",
            "4b97e6b5361f45efb6951dc3b1aebcdb3b89cb69d0f96d6f5c297dd9f45eee75",
        ),
        (
            "https://engineering.case.edu/sites/default/files/100.mat",
            "88a5990cb541320e91505a1d72139e1993500ffe6e292a451011667f4138ca78",
        ),
        (
            "https://engineering.case.edu/sites/default/files/105.mat",
            "f80b0ea04fd06b372a0eaec7c056543ea37e4bb4727a5b173d2a5bacd2aa9cab",
        ),
        (
            "https://engineering.case.edu/sites/default/files/118.mat",
            "b00628f8dd8d1d930af77fa465d1e5cdb385fe259489053f91f3680bda7f640e",
        ),
        (
            "https://engineering.case.edu/sites/default/files/130.mat",
            "35a095307d0971477049b343a1b5981dde465a58fb7f233ad89b035068c1717d",
        ),
    ]
    assert (_paderborn_entry("K001.rar").url, _paderborn_entry("K001.rar").sha256) == (
        f"{_BEARINGDATACENTER_BASE}/K001.rar",
        "0f119ebdb28fb2f4d9fac1beb1319429f63f7ae1256c23c872f280f3560918e5",
    )
    assert (_paderborn_entry("K002.rar").url, _paderborn_entry("K002.rar").sha256) == (
        f"{_BEARINGDATACENTER_BASE}/K002.rar",
        "1040da8b74d169c4e8c8545afa335d7a3a320bcaf36a471250c3e434bc4caffd",
    )


# ---------------------------------------------------------------------------
# 8. Downstream contracts the K003-K006 growth relies on (Task 7):
#    (a) `rowii.tfc.corpora.iter_windows_paderborn_dir` walks its root
#        RECURSIVELY (`root.rglob("*.mat")`), so freshly extracted
#        `K003/`-`K006/` trees are picked up with NO loader change -- pinned
#        against a synthetic K003-style tree mirroring the real extraction
#        layout (`K003.rar` -> `paderborn/K003/K003/<file>.mat`, the nested
#        layout the K001/K002 rars produce on disk).
#    (b) `pretrain_tfc --corpus bearings --out-name tfc_vib_v2.pt` (D5's
#        re-pretrain name) parses; the `--out-name` flag itself landed in
#        Task 6, and main()'s honoring of it is covered end-to-end by
#        tests/test_pretrain_tfc.py::test_out_name_override_applies_to_public_corpora
#        over the SAME corpus-agnostic resolution
#        (`args.out_name or _CHECKPOINT_NAMES[args.corpus]`).
# ---------------------------------------------------------------------------

_PADERBORN_NATIVE_HZ = 64_000.0


def _write_paderborn_style_mat(path: Path, *, channels: dict[str, np.ndarray]) -> None:
    """Paderborn-layout `.mat` fixture: a `Y` struct ARRAY with `Name`/`Data`
    fields nested inside a root struct -- a deliberate mirror of
    tests/test_tfc_corpora.py's `_write_paderborn_style_mat` (duplicated, not
    imported: test modules here never import each other's helpers), whose own
    docstring records the real-file layout verification."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dt = np.dtype([("Name", "O"), ("Data", "O")])
    y = np.zeros((len(channels),), dtype=dt)
    for i, (name, data) in enumerate(channels.items()):
        y[i]["Name"] = name
        y[i]["Data"] = np.asarray(data, dtype=np.float64)
    savemat(path, {"root": {"Y": y}})


def test_k003_style_extraction_tree_is_picked_up_without_loader_changes(tmp_path) -> None:
    rng = np.random.default_rng(3)
    signal = rng.normal(0, 0.2, int(round(2.5 * _PADERBORN_NATIVE_HZ)))
    _write_paderborn_style_mat(
        tmp_path / "K003" / "K003" / "N15_M07_F10_K003_1.mat",
        channels={"force": np.zeros(100), "vibration_1": signal},
    )

    windows = list(iter_windows_paderborn_dir(tmp_path))

    assert len(windows) == 2  # 2.5 s @ nominal 64 kHz -> 2 whole 1-s windows
    assert all(w.shape == (8000,) for w in windows)


def test_pretrain_tfc_out_name_override_parses_for_bearings() -> None:
    parser = pretrain_tfc.build_parser()

    args = parser.parse_args(["--corpus", "bearings", "--out-name", "tfc_vib_v2.pt"])
    assert args.corpus == "bearings"
    assert args.out_name == "tfc_vib_v2.pt"

    default = parser.parse_args(["--corpus", "bearings"])
    assert default.out_name is None
    assert pretrain_tfc._CHECKPOINT_NAMES["bearings"] == "tfc_vib.pt"  # default name unchanged
