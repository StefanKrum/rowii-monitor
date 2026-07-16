"""Smoke tests for scripts/benchmark_inference.py (package-5 spec D7).

Synthetic-only: the numpy configs (handcrafted, logmel) exercise the real
end-to-end measurement loop; torch configs are covered by the skip path
(checkpoint envs unset). Real-window and real-checkpoint measurement happens in
the orchestrated execution, never here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_inference  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    # chdir away from the repo root so load_config() cannot pick up the repo's
    # own .env file (deleting the env vars alone would not mask file values).
    monkeypatch.chdir(tmp_path)
    for var in (
        "ROWII_BEATS_CHECKPOINT", "ROWII_BEATS_INT8_CHECKPOINT",
        "ROWII_TFC_AUDIO_CHECKPOINT", "ROWII_STUDENT_CHECKPOINT",
        "ROWII_FORCE_CPU",
    ):
        monkeypatch.delenv(var, raising=False)


def test_synthetic_numpy_configs_produce_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    out = tmp_path / "bench"
    exit_code = benchmark_inference.main(
        [
            "--configs", "handcrafted,logmel", "--n-windows", "8",
            "--n-batches", "2", "--batch-sizes", "1,4", "--devices", "cpu",
            "--out", str(out),
        ]
    )
    assert exit_code == 0
    table = pd.read_csv(out / "inference.csv")
    assert list(table.columns) == list(benchmark_inference._CSV_COLUMNS)
    assert set(table["config"]) == {"handcrafted", "logmel"}
    assert len(table) == 4  # 2 configs x 2 batch sizes, cpu only
    assert (table["latency_ms_per_window"] > 0).all()
    assert (table["n_params"] == 0).all()
    assert (out / "inference.md").is_file()


def test_torch_configs_skipped_when_checkpoints_unset(
    tmp_path, monkeypatch, caplog
) -> None:
    import logging

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    out = tmp_path / "bench"
    exit_code = benchmark_inference.main(
        [
            "--configs", "handcrafted,beats,tfc,student,beats-int8",
            "--n-windows", "4", "--n-batches", "1", "--batch-sizes", "1",
            "--devices", "cpu", "--out", str(out),
        ]
    )
    assert exit_code == 0
    table = pd.read_csv(out / "inference.csv")
    assert set(table["config"]) == {"handcrafted"}
    skipped = [r.getMessage() for r in caplog.records if "skipping" in r.getMessage()]
    assert len(skipped) == 4  # all four torch configs skipped, each logged


def test_unknown_config_exits_2(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        benchmark_inference.main(
            ["--configs", "nonsense", "--out", str(tmp_path / "bench")]
        )
    assert exc_info.value.code == 2


def test_unknown_device_exits_2(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        benchmark_inference.main(
            [
                "--configs", "handcrafted", "--devices", "gpu",
                "--out", str(tmp_path / "bench"),
            ]
        )
    assert exc_info.value.code == 2
