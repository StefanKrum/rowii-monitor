"""Smoke tests for scripts/benchmark_inference.py.

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


def test_torch_config_with_missing_checkpoint_file_is_skipped(
    tmp_path, monkeypatch, caplog
) -> None:
    """A SET-but-nonexistent checkpoint env must skip like an unset one (with its
    own log line), not FileNotFoundError the whole benchmark table away -- this
    happened for real when the student checkpoint had not been produced yet."""
    import logging

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_STUDENT_CHECKPOINT", str(tmp_path / "missing.pt"))
    out = tmp_path / "bench"
    exit_code = benchmark_inference.main(
        [
            "--configs", "handcrafted,student", "--n-windows", "4",
            "--n-batches", "1", "--batch-sizes", "1", "--devices", "cpu",
            "--out", str(out),
        ]
    )
    assert exit_code == 0
    table = pd.read_csv(out / "inference.csv")
    assert set(table["config"]) == {"handcrafted"}
    skipped = [r.getMessage() for r in caplog.records if "does not exist" in r.getMessage()]
    assert len(skipped) == 1


def test_count_params_on_both_checkpoint_formats(tmp_path) -> None:
    """Pins the format-discriminating branch (final-review finding: previously
    exercised only in real execution): {"cfg","model"} state dicts sum tensor
    sizes; a module pickle (the int8 format) sums LIVE parameters -- for a
    dynamically quantized module that means residual fp32 params only, which
    is why the README discloses the int8 n_params as residual."""
    torch = pytest.importorskip("torch")

    sd_path = tmp_path / "sd.pt"
    torch.save(
        {"cfg": {}, "model": {"a.weight": torch.zeros(3, 4), "a.bias": torch.zeros(3)}},
        sd_path,
    )
    assert benchmark_inference._count_params("beats", sd_path) == 15

    mod_path = tmp_path / "mod.pt"
    torch.save(torch.nn.Linear(4, 2), mod_path)
    assert benchmark_inference._count_params("beats-int8", mod_path) == 10


def test_batch_size_larger_than_pool_is_tiled_to_full_batches(
    tmp_path, monkeypatch, caplog
) -> None:
    """A b=8 row measured from a 3-window pool must really run 8-window
    batches (final-review finding: the old cycling silently measured
    pool-sized batches under the requested-batch label)."""
    import logging

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    out = tmp_path / "bench"
    exit_code = benchmark_inference.main(
        [
            "--configs", "logmel", "--n-windows", "3", "--n-batches", "1",
            "--batch-sizes", "8", "--devices", "cpu", "--out", str(out),
        ]
    )
    assert exit_code == 0
    table = pd.read_csv(out / "inference.csv")
    assert len(table) == 1
    tiled = [r.getMessage() for r in caplog.records if "tiled" in r.getMessage()]
    assert len(tiled) == 1


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
