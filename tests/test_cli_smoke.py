"""Smoke tests for `scripts/copy_data.py` and `scripts/run_step1.py`.

These are CLI-level integration tests: no real ROWII data is used anywhere. Fake
source trees use empty files (copy_data only cares about names/sizes) or the
`gantner_builder` synthetic-fixture writer (run_step1 needs real bytes to read).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import copy_data  # noqa: E402

# ---------------------------------------------------------------------------
# Fake source-tree builder shared by the copy_data tests
# ---------------------------------------------------------------------------

_TOP_LEVEL_FILES = (
    "Sensor_Anordnung_15062026.xlsx",
    "MANIFEST.md",
    "ROWII_Leistung_PU.jpg",
    "ROWII_Leistung_TU.jpg",
)


def _touch(path: Path, size: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)


def _build_fake_source_tree(root: Path, *, sizes: dict[str, int] | None = None) -> Path:
    """Build a fake source tree with the real names, empty (or *sizes*-sized) files."""
    sizes = sizes or {}
    meas = root / "20260626 Messung"
    relpaths = [
        "20260626 Messung/TU/RAWGeneratorMic__0_2026-06-25_06-03-00_000000.dat",
        "20260626 Messung/TU/RAWTurbineMic__1_2026-06-25_06-03-00_000000.dat",
        "20260626 Messung/PU/RAWGeneratorMic__0_2026-06-25_09-08-00_000000.dat",
        "20260626 Messung/PU/RAWTurbineMic__1_2026-06-25_09-08-00_000000.dat",
        "20260626 Messung/Betriebsdaten/2026-06-25_08-00-00.dat",
        "20260626 Messung/Betriebsdaten/2026-06-25_09-00-00.dat",
        "20260626 Messung/ROWII_Leistung.jpg",
        *_TOP_LEVEL_FILES,
    ]
    for rel in relpaths:
        _touch(root / rel, size=sizes.get(rel, 128))
    # A decoy Betriebsdaten file OUTSIDE the 2026-06-25 date filter -- must never be copied.
    _touch(meas / "Betriebsdaten" / "2026-06-24_23-00-00.dat", size=128)
    # A decoy top-level file that is not in the copy plan -- must never be copied.
    _touch(root / "irrelevant_readme.txt", size=64)
    return root


# ---------------------------------------------------------------------------
# 1. copy_data --dry-run: correct file list printed, nothing copied, exit 0
# ---------------------------------------------------------------------------


def test_copy_data_dry_run_prints_file_list_and_copies_nothing(tmp_path, capsys) -> None:
    source = _build_fake_source_tree(tmp_path / "source")
    dest = tmp_path / "dest"

    exit_code = copy_data.main(["--source", str(source), "--dest", str(dest), "--dry-run"])

    assert exit_code == 0
    assert not dest.exists() or not any(dest.rglob("*"))

    out = capsys.readouterr().out
    expected_relpaths = [
        "20260626 Messung/TU/RAWGeneratorMic__0_2026-06-25_06-03-00_000000.dat",
        "20260626 Messung/TU/RAWTurbineMic__1_2026-06-25_06-03-00_000000.dat",
        "20260626 Messung/PU/RAWGeneratorMic__0_2026-06-25_09-08-00_000000.dat",
        "20260626 Messung/PU/RAWTurbineMic__1_2026-06-25_09-08-00_000000.dat",
        "20260626 Messung/Betriebsdaten/2026-06-25_08-00-00.dat",
        "20260626 Messung/Betriebsdaten/2026-06-25_09-00-00.dat",
        "20260626 Messung/ROWII_Leistung.jpg",
        *_TOP_LEVEL_FILES,
    ]
    for rel in expected_relpaths:
        assert rel in out, f"missing {rel!r} from dry-run output"
    # The out-of-range Betriebsdaten file and the irrelevant top-level file must be excluded.
    assert "2026-06-24_23-00-00.dat" not in out
    assert "irrelevant_readme.txt" not in out
    # Total size line: 8 * 128 (docs+ROWII_Leistung.jpg) - wait, see total below.
    total_bytes = 128 * len(expected_relpaths)
    assert str(total_bytes) in out or f"{total_bytes / 1e9:.3f}" in out or "GB" in out


# ---------------------------------------------------------------------------
# 2. copy_data real copy: files copied + manifest written; second run is idempotent
# ---------------------------------------------------------------------------


def test_copy_data_real_copy_writes_files_and_manifest(tmp_path, capsys) -> None:
    source = _build_fake_source_tree(tmp_path / "source")
    dest = tmp_path / "dest"

    exit_code = copy_data.main(["--source", str(source), "--dest", str(dest)])

    assert exit_code == 0
    expected_relpaths = [
        "20260626 Messung/TU/RAWGeneratorMic__0_2026-06-25_06-03-00_000000.dat",
        "20260626 Messung/TU/RAWTurbineMic__1_2026-06-25_06-03-00_000000.dat",
        "20260626 Messung/PU/RAWGeneratorMic__0_2026-06-25_09-08-00_000000.dat",
        "20260626 Messung/PU/RAWTurbineMic__1_2026-06-25_09-08-00_000000.dat",
        "20260626 Messung/Betriebsdaten/2026-06-25_08-00-00.dat",
        "20260626 Messung/Betriebsdaten/2026-06-25_09-00-00.dat",
        "20260626 Messung/ROWII_Leistung.jpg",
        *_TOP_LEVEL_FILES,
    ]
    for rel in expected_relpaths:
        dest_path = dest / rel
        assert dest_path.is_file(), f"expected copied file missing: {rel}"
        assert dest_path.stat().st_size == 128
    # Excluded files must never appear at dest.
    assert not (dest / "20260626 Messung" / "Betriebsdaten" / "2026-06-24_23-00-00.dat").exists()
    assert not (dest / "irrelevant_readme.txt").exists()

    manifest_path = dest / "copy_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    manifest_relpaths = {row["relpath"] for row in manifest}
    assert manifest_relpaths == set(expected_relpaths)
    for row in manifest:
        assert row["bytes"] == 128

    out = capsys.readouterr().out
    assert str(len(expected_relpaths)) in out  # n copied printed somewhere


def test_copy_data_second_run_skips_all_already_present_files(tmp_path, capsys) -> None:
    source = _build_fake_source_tree(tmp_path / "source")
    dest = tmp_path / "dest"
    copy_data.main(["--source", str(source), "--dest", str(dest)])
    capsys.readouterr()  # discard first-run output

    exit_code = copy_data.main(["--source", str(source), "--dest", str(dest)])

    assert exit_code == 0
    out = capsys.readouterr().out
    # 12 files total: 4 .dat groups (2 TU + 2 PU + 2 Betriebsdaten) + 5 top-level = 12
    assert "0 copied" in out
    assert "skipped" in out


def test_copy_data_refuses_when_free_disk_below_safety_margin(tmp_path, monkeypatch) -> None:
    source = _build_fake_source_tree(tmp_path / "source")
    dest = tmp_path / "dest"

    # Required-remaining bytes = 12 * 128 = 1536; 1.2x = 1843.2 -- report free space
    # just under that so the refusal path triggers deterministically regardless of
    # the tmp filesystem's real free space.
    fake_usage = shutil.disk_usage  # keep the real namedtuple type

    def fake_disk_usage(path):
        real = fake_usage(path)
        return real._replace(free=1000)

    monkeypatch.setattr(copy_data.shutil, "disk_usage", fake_disk_usage)

    exit_code = copy_data.main(["--source", str(source), "--dest", str(dest)])

    assert exit_code == 2


# ---------------------------------------------------------------------------
# 3. run_step1.py --help exits 0
# ---------------------------------------------------------------------------


def test_run_step1_help_exits_zero(capsys) -> None:
    import run_step1

    with pytest.raises(SystemExit) as exc_info:
        run_step1.main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--run" in out
    assert "--variant" in out
    assert "--clusterer" in out
    assert "--k" in out
    assert "--k-sweep" in out


# ---------------------------------------------------------------------------
# 4. beats variant without the extra installed -> SystemExit with install hint
# ---------------------------------------------------------------------------


def test_beats_variant_without_extra_raises_systemexit_with_install_hint(monkeypatch) -> None:
    import builtins

    import run_step1

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rowii.signals.beats" or name.startswith("rowii.signals.beats."):
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit) as exc_info:
        run_step1._import_beats_or_exit()

    message = str(exc_info.value)
    assert "beats" in message
    assert 'pip install -e ".[beats]"' in message
    assert "ROWII_BEATS_CHECKPOINT" in message


# ---------------------------------------------------------------------------
# 5. Miniature end-to-end: fusion/kmeans/k=2 on a tiny synthetic data_root
# ---------------------------------------------------------------------------

_E2E_MIC_RATE_HZ = 800.0
_E2E_VIB_RATE_HZ = 400.0
_E2E_SCADA_RATE_HZ = 10.0
_E2E_DURATION_S = 60
_E2E_T0_NS = 1_750_000_000_000_000_000  # arbitrary but fixed UTC epoch, ns


def _build_e2e_data_root(root: Path) -> Path:
    """A tiny synthetic data_root: 2 mic + 2 vib streams (~60 s) + one Betriebsdaten
    file with a clear standstill -> turbine step at t=30s, all sharing the same UTC
    time base (`_E2E_T0_NS`). Filenames use plausible (but otherwise arbitrary) local
    timestamps -- only `dataset.discover`'s filename PARSING needs to succeed; actual
    time alignment comes from each file's real UDBF header t0_ns.
    """
    from tests.fixtures.gantner_builder import build_gantner_file

    meas = root / "20260626 Messung"
    tu = meas / "TU"
    bd = meas / "Betriebsdaten"
    tu.mkdir(parents=True, exist_ok=True)
    bd.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    # Mic streams: 4 channels each, distinct means before/after t=30s so KMeans has a
    # genuine 2-cluster structure to recover (standstill = quiet/low-RMS, turbine =
    # loud/high-RMS).
    n_mic = int(_E2E_MIC_RATE_HZ * _E2E_DURATION_S)
    half_mic = n_mic // 2
    for stream_name in ("RAWGeneratorMic__0", "RAWTurbineMic__1"):
        quiet = rng.normal(0.0, 0.05, size=(half_mic, 4)).astype(np.float32)
        loud = rng.normal(0.0, 2.0, size=(n_mic - half_mic, 4)).astype(np.float32)
        data = np.vstack([quiet, loud])
        build_gantner_file(
            tu / f"{stream_name}_2026-06-25_06-00-00_000000.dat",
            ["ch0", "ch1", "ch2", "ch3"],
            data,
            t0_ns=_E2E_T0_NS,
            rate_hz=_E2E_MIC_RATE_HZ,
        )

    # Vib streams: 2 live channels each (avoid degenerate all-dead VibFeaturizer input),
    # same quiet/loud step.
    n_vib = int(_E2E_VIB_RATE_HZ * _E2E_DURATION_S)
    half_vib = n_vib // 2
    for stream_name in ("RAWGeneratorVib__2", "RAWTurbineVib__3"):
        quiet = rng.normal(0.0, 0.02, size=(half_vib, 2)).astype(np.float32)
        loud = rng.normal(0.0, 1.0, size=(n_vib - half_vib, 2)).astype(np.float32)
        data = np.vstack([quiet, loud])
        build_gantner_file(
            tu / f"{stream_name}_2026-06-25_06-00-00_000000.dat",
            ["chX", "chY"],
            data,
            t0_ns=_E2E_T0_NS,
            rate_hz=_E2E_VIB_RATE_HZ,
        )

    # Betriebsdaten: covers the same ~60 s span at 10 Hz, standstill (n=0, P=0) ->
    # turbine (n=375, P=10) step at the same t=30s boundary.
    n_scada = int(_E2E_SCADA_RATE_HZ * _E2E_DURATION_S)
    half_scada = n_scada // 2
    power = np.concatenate(
        [np.zeros(half_scada, dtype=np.float32), np.full(n_scada - half_scada, 10.0, np.float32)]
    )
    speed = np.concatenate(
        [np.zeros(half_scada, dtype=np.float32), np.full(n_scada - half_scada, 375.0, np.float32)]
    )
    guide_vane = np.full(n_scada, 50.0, dtype=np.float32)
    flow_tu = np.concatenate(
        [np.zeros(half_scada, dtype=np.float32), np.full(n_scada - half_scada, 5.0, np.float32)]
    )
    flow_pu = np.zeros(n_scada, dtype=np.float32)
    scada_data = np.stack([power, speed, guide_vane, flow_tu, flow_pu], axis=1)
    build_gantner_file(
        bd / "2026-06-25_08-00-00.dat",
        ["1_P_Ist", "1_Drehzahl_Ist", "1_Leitapparat Stell.", "Durchfluss TU", "Durchfluss PU"],
        scada_data,
        t0_ns=_E2E_T0_NS,
        rate_hz=_E2E_SCADA_RATE_HZ,
    )

    return root


def test_run_combo_fusion_kmeans_k2_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    data_root = _build_e2e_data_root(tmp_path / "data")

    import run_step1

    from rowii.config import load_config
    from rowii.io.dataset import discover

    cfg = load_config()
    index = discover(data_root)
    tu_runs = [r for r in index.runs if r.name == "tu"]
    assert len(tu_runs) == 1, f"expected exactly one 'tu' run, got {[r.name for r in index.runs]}"
    run = tu_runs[0]

    result = run_step1.run_combo(
        run, "fusion", "kmeans", cfg, index.betriebsdaten, cfg.results_root, k=2
    )

    assert result.run == "tu"
    assert result.variant == "fusion"
    assert result.clusterer == "kmeans"
    assert result.k == 2
    assert result.n_windows > 0
    assert result.ari is not None  # "any value" per the brief -- just must be present

    out_dir = cfg.results_root / "tu" / "fusion-kmeans"
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "segments.csv").is_file()
    assert (out_dir / "frame_labels.parquet").is_file()

    summary_path = cfg.results_root / "summary.csv"
    assert summary_path.is_file()
    summary = pd.read_csv(summary_path)
    assert len(summary) == 1
    assert summary.iloc[0]["run"] == "tu"
    assert summary.iloc[0]["variant"] == "fusion"
    assert summary.iloc[0]["clusterer"] == "kmeans"


def test_run_combo_k_sweep_writes_four_rows_with_silhouette_and_k_sweep_note(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    data_root = _build_e2e_data_root(tmp_path / "data")

    import run_step1

    from rowii.config import load_config
    from rowii.io.dataset import discover

    cfg = load_config()
    index = discover(data_root)
    run = next(r for r in index.runs if r.name == "tu")

    results = run_step1.run_combo_k_sweep(
        run, "audio", "kmeans", cfg, index.betriebsdaten, cfg.results_root
    )

    assert [r.k for r in results] == [3, 4, 5, 6]
    for r in results:
        assert r.notes == "k-sweep"
        assert r.silhouette is not None

    summary = pd.read_csv(cfg.results_root / "summary.csv")
    assert len(summary) == 4
    assert (summary["notes"] == "k-sweep").all()
    assert list(summary["k"]) == [3, 4, 5, 6]
