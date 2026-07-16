"""Tests for `scripts/download_corpora.py` (package-4 spec D2, Task 2):
CLI-level tests with a monkeypatched `urllib.request.urlopen` -- NO real
network access anywhere in this file. `--dry-run` tests additionally assert
`urlopen` is never even CALLED, via a raising stub (module docstring's own
network policy: dry-run must never open a connection).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download_corpora  # noqa: E402


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


def test_real_declared_sha256_mismatch_raises(monkeypatch, tmp_path) -> None:
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


# ---------------------------------------------------------------------------
# 5. End-to-end (monkeypatched network) corpus download: manifest + extraction
# ---------------------------------------------------------------------------


def test_download_corpus_cwru_writes_manifest_with_computed_hashes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        download_corpora.urllib.request, "urlopen", lambda url: _FakeResponse(b"x" * 128)
    )

    download_corpora._download_corpus("cwru", tmp_path)

    manifest_path = tmp_path / "cwru" / "MANIFEST.json"
    assert manifest_path.is_file()
    rows = json.loads(manifest_path.read_text())
    assert len(rows) == len(download_corpora._CORPUS_FILES["cwru"])
    for row in rows:
        assert row["sha256"] != download_corpora._SHA256_TBD
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

    with caplog.at_level(logging.WARNING):
        download_corpora._download_corpus("paderborn", tmp_path)  # must not raise

    manifest_path = tmp_path / "paderborn" / "MANIFEST.json"
    assert manifest_path.is_file()  # the download itself still succeeded
    assert (tmp_path / "paderborn" / "K001.rar").is_file()

    err = capsys.readouterr().err
    assert "unar" in err and "unrar" in err
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("K001.rar" in w for w in warnings), warnings
