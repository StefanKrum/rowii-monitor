"""Tests for `scripts/warm_cache.py` (Step-2 package 2, Task 6): CLI-level tests
against a synthetic, monkeypatched `discover` -- no real ROWII data or BEATs
checkpoint anywhere (mirrors `tests/test_step2_cli.py`'s established pattern),
plus a monkeypatched `prepare_run` recorder so a "real" (non-dry-run) invocation
never actually runs feature extraction.
"""
from __future__ import annotations

import builtins
import itertools
import logging
import sys
from pathlib import Path

import pytest

from rowii.io.dataset import RecordingIndex, Run

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _fake_index(run_names: list[str]) -> RecordingIndex:
    runs = [Run(name=name, files={}, day_root=Path("/fake-day-root")) for name in run_names]
    return RecordingIndex(runs=runs, betriebsdaten=[], betriebsdaten_by_day={})


# ---------------------------------------------------------------------------
# 1. --help exits 0 and documents every flag
# ---------------------------------------------------------------------------


def test_warm_cache_help_exits_zero(capsys) -> None:
    import warm_cache

    with pytest.raises(SystemExit) as exc_info:
        warm_cache.main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--runs", "--variants", "--dry-run"):
        assert flag in out, f"missing {flag!r} in --help output"


def test_build_parser_defaults_match_the_documented_defaults() -> None:
    import warm_cache

    args = warm_cache.build_parser().parse_args([])
    assert args.runs == list(warm_cache._DEFAULT_RUNS)
    assert args.variants == list(warm_cache._DEFAULT_VARIANTS)
    assert args.dry_run is False
    # 27.06 gap-splits into three discovered runs (-1/-2/-3) -- the pre-fix default
    # named the nonexistent unsuffixed run and correctly exited 2 on real data.
    assert list(warm_cache._DEFAULT_RUNS) == [
        "250526-tu", "290626-tu", "010726-tu_ph_tu",
        "270626-pu_ph_pu_ph_pu_ph-1", "270626-pu_ph_pu_ph_pu_ph-2",
        "270626-pu_ph_pu_ph_pu_ph-3",
    ]
    assert list(warm_cache._DEFAULT_VARIANTS) == ["audio-beats", "fusion-beats"]


# ---------------------------------------------------------------------------
# 2. --dry-run: prints every combo, exits 0, calls neither prepare_run nor the
#    beats-import guard
# ---------------------------------------------------------------------------


def test_dry_run_prints_every_combo_and_calls_neither_prepare_run_nor_beats_guard(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))

    import warm_cache

    monkeypatch.setattr(warm_cache, "discover", lambda data_root: _fake_index(["run-a", "run-b"]))

    def _boom_prepare_run(run, variant, cfg, *, use_cache):
        raise AssertionError("prepare_run must not be called under --dry-run")

    def _boom_import_beats_or_exit():
        raise AssertionError("_import_beats_or_exit must not be called under --dry-run")

    monkeypatch.setattr(warm_cache, "prepare_run", _boom_prepare_run)
    monkeypatch.setattr(warm_cache, "_import_beats_or_exit", _boom_import_beats_or_exit)

    exit_code = warm_cache.main(
        [
            "--runs", "run-a", "run-b",
            "--variants", "audio-beats", "fusion-beats",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    for run_name, variant in itertools.product(("run-a", "run-b"), ("audio-beats", "fusion-beats")):
        assert run_name in out
        assert variant in out
    assert "4" in out  # 2 runs x 2 variants = 4 combos, stated up front


# ---------------------------------------------------------------------------
# 3. Real invocation: calls the recorder once per combo with use_cache=True
# ---------------------------------------------------------------------------


def test_real_invocation_calls_prepare_run_once_per_combo_with_use_cache_true(
    monkeypatch, tmp_path, caplog
) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))

    import warm_cache

    monkeypatch.setattr(warm_cache, "discover", lambda data_root: _fake_index(["run-a", "run-b"]))
    monkeypatch.setattr(warm_cache, "_import_beats_or_exit", lambda: None)

    calls: list[tuple[str, str, bool]] = []

    def _fake_prepare_run(run, variant, cfg, *, use_cache):
        calls.append((run.name, variant, use_cache))

    monkeypatch.setattr(warm_cache, "prepare_run", _fake_prepare_run)

    with caplog.at_level(logging.INFO):
        exit_code = warm_cache.main(
            ["--runs", "run-a", "run-b", "--variants", "audio-beats", "fusion-beats"]
        )

    assert exit_code == 0
    assert calls == [
        ("run-a", "audio-beats", True),
        ("run-a", "fusion-beats", True),
        ("run-b", "audio-beats", True),
        ("run-b", "fusion-beats", True),
    ]
    messages = [r.message for r in caplog.records]
    assert any("run-a" in m and "audio-beats" in m for m in messages), messages


def test_real_invocation_never_touches_beats_guard_for_non_beats_variants(
    monkeypatch, tmp_path
) -> None:
    """Handcrafted variants (no BEATs involved) must not trigger the beats-import
    guard at all -- only combos that actually include a beats variant do."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))

    import warm_cache

    monkeypatch.setattr(warm_cache, "discover", lambda data_root: _fake_index(["run-a"]))

    def _boom_import_beats_or_exit():
        raise AssertionError("_import_beats_or_exit must not be called for non-beats variants")

    monkeypatch.setattr(warm_cache, "_import_beats_or_exit", _boom_import_beats_or_exit)
    monkeypatch.setattr(
        warm_cache, "prepare_run", lambda run, variant, cfg, *, use_cache: None
    )

    exit_code = warm_cache.main(["--runs", "run-a", "--variants", "fusion"])
    assert exit_code == 0


# ---------------------------------------------------------------------------
# 4. Unknown run name: exit 2, names the available runs
# ---------------------------------------------------------------------------


def test_unknown_run_name_exits_2_and_lists_available_runs(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))

    import warm_cache

    monkeypatch.setattr(warm_cache, "discover", lambda data_root: _fake_index(["run-a", "run-b"]))

    def _boom_prepare_run(run, variant, cfg, *, use_cache):
        raise AssertionError("prepare_run must not be called for an unknown run name")

    monkeypatch.setattr(warm_cache, "prepare_run", _boom_prepare_run)

    exit_code = warm_cache.main(["--runs", "run-a", "totally-bogus-run"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "totally-bogus-run" in err
    assert "run-a" in err
    assert "run-b" in err


def test_unknown_run_name_check_also_applies_under_dry_run(monkeypatch, tmp_path, capsys) -> None:
    """The exit-2 unknown-run-name contract is not exempted by --dry-run (only the
    beats-import guard is dry-run-exempt, per the brief's step 3)."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))

    import warm_cache

    monkeypatch.setattr(warm_cache, "discover", lambda data_root: _fake_index(["run-a"]))

    exit_code = warm_cache.main(["--runs", "totally-bogus-run", "--dry-run"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "totally-bogus-run" in err
    assert "run-a" in err


# ---------------------------------------------------------------------------
# 5. _import_beats_or_exit: SystemExit with the install hint when the extra is
#    missing (mirrors tests/test_cli_smoke.py's run_step1 equivalent)
# ---------------------------------------------------------------------------


def test_import_beats_or_exit_raises_systemexit_with_install_hint(monkeypatch) -> None:
    import warm_cache

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rowii.signals.beats" or name.startswith("rowii.signals.beats."):
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit) as exc_info:
        warm_cache._import_beats_or_exit()

    message = str(exc_info.value)
    assert "beats" in message
    assert 'pip install -e ".[beats]"' in message
    assert "ROWII_BEATS_CHECKPOINT" in message
