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
    # 27.06 gap-splits into three discovered runs (-1/-2/-3); only -1 is a real
    # session worth warming -- -2/-3 are negligible ~12-min orphan fragments, and -2
    # has no RAWGeneratorMic__0 stream at all (real 2026-07-15 warm-up finding), so
    # only -1 belongs in the default list.
    assert list(warm_cache._DEFAULT_RUNS) == [
        "250526-tu", "290626-tu", "010726-tu_ph_tu", "270626-pu_ph_pu_ph_pu_ph-1",
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
# 4. Per-combo error isolation: a failing combo is logged + skipped, the rest
#    of the batch still runs, and the invocation exits 1 (0 only when all
#    combos succeeded)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raised",
    [KeyError("RAWGeneratorMic__0"), RuntimeError("92.5% of 12 grid windows are invalid")],
    ids=["KeyError-missing-stream", "RuntimeError-too-sparse"],
)
def test_failing_combo_is_skipped_later_combos_still_run_and_exit_code_is_1(
    raised, monkeypatch, tmp_path, caplog
) -> None:
    """Real-data regression (2026-07-15 background warm-up): `270626-pu_ph_pu_ph_pu_ph-2`
    is an orphan fragment with NO RAWGeneratorMic__0 stream at all, so `prepare_run`
    raised `KeyError: 'RAWGeneratorMic__0'` (from `build_run_grid`'s
    `run.files[stream]`) and killed the whole batch after 8/12 combos. One bad combo
    must be logged and skipped -- every later combo still runs -- and the invocation
    must exit 1 so the failure is not silently swallowed either. RuntimeError (run too
    short/sparse for the variant, `rowii.pipeline.compute_validity_mask`) gets the
    identical treatment -- both halves of the catch tuple are exercised here."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))

    import warm_cache

    monkeypatch.setattr(
        warm_cache, "discover", lambda data_root: _fake_index(["run-a", "run-bad", "run-c"])
    )
    monkeypatch.setattr(warm_cache, "_import_beats_or_exit", lambda: None)

    calls: list[tuple[str, str]] = []

    def _fake_prepare_run(run, variant, cfg, *, use_cache):
        calls.append((run.name, variant))
        if run.name == "run-bad":
            raise raised

    monkeypatch.setattr(warm_cache, "prepare_run", _fake_prepare_run)

    with caplog.at_level(logging.WARNING):
        exit_code = warm_cache.main(
            ["--runs", "run-a", "run-bad", "run-c", "--variants", "audio-beats"]
        )

    assert exit_code == 1
    # Every combo was attempted -- the failing middle one did not abort the batch.
    assert calls == [
        ("run-a", "audio-beats"), ("run-bad", "audio-beats"), ("run-c", "audio-beats"),
    ]
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "run-bad" in w and "audio-beats" in w and type(raised).__name__ in w for w in warnings
    ), warnings


def test_all_combos_succeeding_still_exits_0(monkeypatch, tmp_path) -> None:
    """Contrast to the failure-isolation test above (same shape, no failing run):
    exit 1 is reserved for actual per-combo failures -- an all-success invocation
    keeps exiting 0."""
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))

    import warm_cache

    monkeypatch.setattr(
        warm_cache, "discover", lambda data_root: _fake_index(["run-a", "run-c"])
    )
    monkeypatch.setattr(warm_cache, "_import_beats_or_exit", lambda: None)
    monkeypatch.setattr(
        warm_cache, "prepare_run", lambda run, variant, cfg, *, use_cache: None
    )

    exit_code = warm_cache.main(["--runs", "run-a", "run-c", "--variants", "audio-beats"])

    assert exit_code == 0


# ---------------------------------------------------------------------------
# 5. Unknown run name: exit 2, names the available runs
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
# 6. _import_beats_or_exit: SystemExit with the install hint when the extra is
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
