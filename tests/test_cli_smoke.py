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
    # Total size line: every fixture file is touched at the default 128 bytes
    # (_build_fake_source_tree's sizes=None), so the total is exactly 128 *
    # len(expected_relpaths) -- assert the precise byte count `_print_dry_run`
    # writes (`f"Total: {total} bytes (...)"`), not just that some size-shaped
    # substring appears.
    total_bytes = 128 * len(expected_relpaths)
    assert f"Total: {total_bytes} bytes" in out


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
    reactive = np.zeros(n_scada, dtype=np.float32)
    ks_valve = np.full(n_scada, 3.0, dtype=np.float32)
    scada_data = np.stack(
        [power, speed, guide_vane, flow_tu, flow_pu, reactive, ks_valve], axis=1
    )
    build_gantner_file(
        bd / "2026-06-25_08-00-00.dat",
        [
            "1_P_Ist", "1_Drehzahl UPM", "1_Leitapparat Stell.", "Durchfluss TU",
            "Durchfluss PU", "1_Q_Ist", "1_KS Stellung",
        ],
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
    # State-level (primary) metrics must be present and populated for a GT combo --
    # NaN would silently defeat their purpose as the summary's headline columns.
    for col in ("state_ari", "state_accuracy", "state_macro_f1"):
        assert col in summary.columns, f"missing column {col!r} in summary.csv"
        assert pd.notna(summary.iloc[0][col]), f"{col!r} is NaN for a GT combo"


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

    # Each k's report.md/timeline.png must land in its OWN directory, not overwrite a
    # shared one -- otherwise only the last k value's artifacts survive on disk.
    run_dir = cfg.results_root / "tu"
    k_dirs = sorted(p.name for p in run_dir.iterdir() if p.is_dir() and "-k" in p.name)
    assert len(set(k_dirs)) >= 2, f"expected at least 2 distinct per-k dirs, found {k_dirs}"
    for k in (3, 4, 5, 6):
        k_dir = run_dir / f"audio-kmeans-k{k}"
        assert k_dir.is_dir(), f"expected {k_dir} to exist"
        assert (k_dir / "report.md").is_file()
        assert (k_dir / "timeline.png").is_file()
        assert (k_dir / "segments.csv").is_file()
        assert (k_dir / "frame_labels.parquet").is_file()


# ---------------------------------------------------------------------------
# 6. Real-hardware clock jitter: +/-1 sample/window must not NaN out the window
# ---------------------------------------------------------------------------


def test_extract_stream_features_tolerates_one_sample_clock_jitter(tmp_path) -> None:
    # Task 13 real-data finding: real DAQ files do NOT tile perfectly into
    # `round(rate_hz * window_s)` samples per window -- natural clock jitter puts some
    # windows at expected_samples +/- 1 (measured on the real June-25 TU mic/vib streams:
    # ~33% of windows off by exactly 1 sample, with NO actual data gap). Before the fix,
    # `_extract_stream_features` required an EXACT sample-count match to featurize a
    # window, so every jittered window was silently left NaN -- at real jitter rates this
    # blows straight through the pipeline's 5%-invalid hard-fail threshold.
    #
    # A rate_hz that is not an exact integer nanosecond period (100.003 Hz here) makes
    # `build_gantner_file`'s own timestamp spacing accumulate exactly this kind of +/-1
    # rounding jitter across windows, without needing a special jittered-timestamp fixture.
    import run_step1

    from rowii.io.dataset import BurstFile
    from rowii.signals.features import AudioFeaturizer
    from rowii.signals.windows import WindowGrid
    from tests.fixtures.gantner_builder import build_gantner_file

    rate_hz = 100.003
    n_windows = 5
    n_samples = round(rate_hz * n_windows)  # ~500, spans exactly n_windows nominal seconds
    data = np.ones((n_samples, 2), dtype=np.float32)
    path = build_gantner_file(
        tmp_path / "jitter.dat", ["ChA", "ChB"], data, t0_ns=0, rate_hz=rate_hz,
    )
    burst = BurstFile(path=path, stream="RAWGeneratorMic__0", start_utc_hint=None)  # type: ignore[arg-type]
    grid = WindowGrid(t0_ns=0, window_ns=1_000_000_000, n_windows=n_windows)

    result = run_step1._extract_stream_features([burst], grid, AudioFeaturizer())

    nan_rows = np.isnan(result.features).any(axis=1)
    assert not nan_rows.any(), (
        f"expected every window to be featurized despite clock jitter, "
        f"but {nan_rows.sum()}/{n_windows} are NaN"
    )


# ---------------------------------------------------------------------------
# 7. Fusion assembly must not let a few invalid-window NaN rows zero out
#    every fused column for the WHOLE run
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 8. audio-beats / fusion-beats variants: real CLI wiring, stub BEATs encoder
# ---------------------------------------------------------------------------


def test_run_combo_audio_beats_kmeans_end_to_end_with_stub_encoder(tmp_path, monkeypatch) -> None:
    """Exercises the REAL `audio-beats` CLI wiring (streams -> BeatsFeaturizer ->
    hstack of the two mic streams' embeddings -> detect -> report), but with a
    stub `BeatsEncoderProtocol` injected so this test needs neither the real
    checkpoint nor real torch weights -- BEATs' own encoder correctness is
    already covered by `tests/test_beats.py`; this test only proves
    `run_step1.py` builds one `BeatsFeaturizer` per mic stream and assembles
    their outputs the same way handcrafted `audio` does.
    """
    pytest.importorskip("torch")
    import torch

    from rowii.signals import beats as beats_module

    _STUB_EMBED_DIM = 4
    call_count = {"n": 0}
    real_init = beats_module.BeatsFeaturizer.__init__

    class _StubEncoder:
        def extract(self, fbank: torch.Tensor) -> torch.Tensor:
            call_count["n"] += 1
            per_frame_mean = fbank.mean(dim=-1, keepdim=True)
            return per_frame_mean.expand(-1, _STUB_EMBED_DIM)

    def fake_init(self, checkpoint, device=None, encoder=None) -> None:
        real_init(self, checkpoint=checkpoint, device=torch.device("cpu"), encoder=_StubEncoder())

    monkeypatch.setattr(beats_module.BeatsFeaturizer, "__init__", fake_init)

    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", str(tmp_path / "fake_checkpoint.pt"))
    data_root = _build_e2e_data_root(tmp_path / "data")

    import run_step1

    from rowii.config import load_config
    from rowii.io.dataset import discover

    cfg = load_config()
    index = discover(data_root)
    run = next(r for r in index.runs if r.name == "tu")

    result = run_step1.run_combo(
        run, "audio-beats", "kmeans", cfg, index.betriebsdaten, cfg.results_root, k=2
    )

    # The stub encoder must actually have been exercised -- otherwise this test
    # would spuriously pass even if run_step1 silently fell back to the
    # handcrafted AudioFeaturizer (the pre-Task-14 bug: audio-beats/fusion-beats
    # were accepted by argparse and routed through _streams_for_variant/
    # assemble_variant_features, but _prepare_run_features's per-stream
    # featurizer selection never actually branched on the variant, so both
    # "beats" variants silently ran handcrafted features under the hood).
    assert call_count["n"] > 0, (
        "BeatsFeaturizer's encoder was never called -- audio-beats is not "
        "actually routing through BeatsFeaturizer"
    )

    assert result.variant == "audio-beats"
    assert result.k == 2
    assert result.n_windows > 0
    assert result.ari is not None

    out_dir = cfg.results_root / "tu" / "audio-beats-kmeans"
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "segments.csv").is_file()
    assert (out_dir / "frame_labels.parquet").is_file()

    # 2 mic streams x _STUB_EMBED_DIM each, hstacked -- handcrafted
    # AudioFeaturizer produces dozens of named spectral features per channel,
    # never exactly 2 * _STUB_EMBED_DIM, so this width is only reachable via
    # BeatsFeaturizer actually running on both streams.
    frame_labels = pd.read_parquet(out_dir / "frame_labels.parquet")
    assert "cluster" in frame_labels.columns  # sanity: real report, not a stub artifact


def test_run_combo_fusion_beats_kmeans_end_to_end_with_stub_encoder(tmp_path, monkeypatch) -> None:
    """Same as the `audio-beats` test above, for `fusion-beats`: BEATs
    embeddings on both mic streams `fuse()`d with handcrafted vibration
    features, exactly as `fusion` fuses handcrafted audio with handcrafted
    vibration."""
    pytest.importorskip("torch")
    import torch

    from rowii.signals import beats as beats_module

    call_count = {"n": 0}
    real_init = beats_module.BeatsFeaturizer.__init__

    class _StubEncoder:
        def extract(self, fbank: torch.Tensor) -> torch.Tensor:
            call_count["n"] += 1
            per_frame_mean = fbank.mean(dim=-1, keepdim=True)
            return per_frame_mean.expand(-1, 4)

    def fake_init(self, checkpoint, device=None, encoder=None) -> None:
        real_init(self, checkpoint=checkpoint, device=torch.device("cpu"), encoder=_StubEncoder())

    monkeypatch.setattr(beats_module.BeatsFeaturizer, "__init__", fake_init)

    monkeypatch.setenv("ROWII_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("ROWII_BEATS_CHECKPOINT", str(tmp_path / "fake_checkpoint.pt"))
    data_root = _build_e2e_data_root(tmp_path / "data")

    import run_step1

    from rowii.config import load_config
    from rowii.io.dataset import discover

    cfg = load_config()
    index = discover(data_root)
    run = next(r for r in index.runs if r.name == "tu")

    result = run_step1.run_combo(
        run, "fusion-beats", "kmeans", cfg, index.betriebsdaten, cfg.results_root, k=2
    )

    assert call_count["n"] > 0, (
        "BeatsFeaturizer's encoder was never called -- fusion-beats is not "
        "actually routing through BeatsFeaturizer"
    )

    assert result.variant == "fusion-beats"
    assert result.k == 2
    assert result.n_windows > 0
    assert result.ari is not None

    out_dir = cfg.results_root / "tu" / "fusion-beats-kmeans"
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "segments.csv").is_file()
    assert (out_dir / "frame_labels.parquet").is_file()


def test_assemble_variant_features_fusion_survives_a_few_nan_rows() -> None:
    # Task 13 real-data run (TU/fusion): a real run always has a handful of invalid
    # windows (coverage gaps at file boundaries, ~11/8286 on the June-25 TU run) whose
    # rows are NaN in every per-stream feature matrix -- this is normal and expected
    # (see _StreamFeatureResult.features' own docstring). `assemble_variant_features`'s
    # fusion path calls `fuse()` -> `zscore()` on the FULL, still-NaN-containing matrix
    # (valid-masking happens later, in _detect_and_report) -- `zscore`'s zero-std guard
    # (`std >= 1e-12`) is `False` for ANY column containing a NaN (NaN comparisons are
    # always False in IEEE-754), which zeroed out EVERY fused column for the ENTIRE run,
    # not just the invalid rows: real ARI on TU/fusion was silently 0.0 because of this
    # (sklearn KMeans then collapses to a single cluster on the resulting all-zero input).
    #
    # Reproduces the bug with a handful of NaN rows (not real data) and asserts the
    # non-NaN rows keep genuine, non-degenerate variance after fusion.
    from run_step1 import (
        _AUDIO_STREAMS,
        _VIB_STREAMS,
        _StreamFeatureResult,
        assemble_variant_features,
    )

    rng = np.random.default_rng(0)
    n_windows = 50
    stream_results = {}
    for stream in (*_AUDIO_STREAMS, *_VIB_STREAMS):
        features = rng.normal(size=(n_windows, 4))
        features[3] = np.nan  # one invalid window, matching real run's small NaN fraction
        stream_results[stream] = _StreamFeatureResult(
            features=features, coverage=np.ones(n_windows)
        )

    fused = assemble_variant_features("fusion", stream_results)

    valid_rows = ~np.isnan(fused).any(axis=1)
    assert valid_rows.sum() == n_windows - 1
    valid_std = fused[valid_rows].std(axis=0)
    assert (valid_std > 1e-6).all(), (
        "expected every fused column to retain genuine variance on the valid rows, "
        f"but {int((valid_std <= 1e-6).sum())}/{len(valid_std)} columns are degenerate"
    )


# ---------------------------------------------------------------------------
# 7. Multi-day parent root: Betriebsdaten must be scoped per day tree, not
#    pooled flat across the whole discovered index (addendum spec §2/§4).
# ---------------------------------------------------------------------------


def _build_one_day_tree(
    day_dir: Path, *, day_label: str, state: str
) -> None:
    """One day tree (`<day_dir>/<day_label> Messung/{TU,Betriebsdaten}`) with a
    SINGLE burst run and a SINGLE, internally-homogeneous Betriebsdaten hour
    whose GT state is entirely *state* ("standstill" or "turbine").

    Deliberately reuses the exact SAME `_E2E_T0_NS` epoch for every day tree
    built this way (unlike `_build_e2e_data_root`, which is free to use any
    fixed epoch since it only ever builds one day) -- this makes every day's
    burst/Betriebsdaten files overlap in absolute UTC time with every OTHER
    day's, so `_betriebsdaten_for_grid`'s time-overlap filter alone could never
    correctly disambiguate which Betriebsdaten file belongs to which run: the
    only thing that can keep the days from cross-contaminating each other's GT
    is `main()` actually scoping by `Run.day_root` via
    `RecordingIndex.betriebsdaten_by_day`, not a lucky non-overlapping-time
    coincidence.
    """
    from tests.fixtures.gantner_builder import build_gantner_file

    meas = day_dir / f"{day_label} Messung"
    tu = meas / "TU"
    bd = meas / "Betriebsdaten"
    tu.mkdir(parents=True, exist_ok=True)
    bd.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    n_mic = int(_E2E_MIC_RATE_HZ * _E2E_DURATION_S)
    for stream_name in ("RAWGeneratorMic__0", "RAWTurbineMic__1"):
        data = rng.normal(0.0, 0.5, size=(n_mic, 4)).astype(np.float32)
        build_gantner_file(
            tu / f"{stream_name}_2026-06-25_06-00-00_000000.dat",
            ["ch0", "ch1", "ch2", "ch3"],
            data,
            t0_ns=_E2E_T0_NS,
            rate_hz=_E2E_MIC_RATE_HZ,
        )
    n_vib = int(_E2E_VIB_RATE_HZ * _E2E_DURATION_S)
    for stream_name in ("RAWGeneratorVib__2", "RAWTurbineVib__3"):
        data = rng.normal(0.0, 0.2, size=(n_vib, 2)).astype(np.float32)
        build_gantner_file(
            tu / f"{stream_name}_2026-06-25_06-00-00_000000.dat",
            ["chX", "chY"],
            data,
            t0_ns=_E2E_T0_NS,
            rate_hz=_E2E_VIB_RATE_HZ,
        )

    n_scada = int(_E2E_SCADA_RATE_HZ * _E2E_DURATION_S)
    if state == "standstill":
        power = np.zeros(n_scada, dtype=np.float32)
        speed = np.zeros(n_scada, dtype=np.float32)
        flow_tu = np.zeros(n_scada, dtype=np.float32)
    else:
        power = np.full(n_scada, 10.0, dtype=np.float32)
        speed = np.full(n_scada, 375.0, dtype=np.float32)
        flow_tu = np.full(n_scada, 5.0, dtype=np.float32)
    guide_vane = np.full(n_scada, 50.0, dtype=np.float32)
    flow_pu = np.zeros(n_scada, dtype=np.float32)
    reactive = np.zeros(n_scada, dtype=np.float32)
    ks_valve = np.full(n_scada, 3.0, dtype=np.float32)
    scada_data = np.stack(
        [power, speed, guide_vane, flow_tu, flow_pu, reactive, ks_valve], axis=1
    )
    build_gantner_file(
        bd / "2026-06-25_08-00-00.dat",
        [
            "1_P_Ist", "1_Drehzahl UPM", "1_Leitapparat Stell.", "Durchfluss TU",
            "Durchfluss PU", "1_Q_Ist", "1_KS Stellung",
        ],
        scada_data,
        t0_ns=_E2E_T0_NS,
        rate_hz=_E2E_SCADA_RATE_HZ,
    )


def test_main_scopes_betriebsdaten_per_day_tree_not_pooled_across_days(
    tmp_path, monkeypatch
) -> None:
    # Two day trees under one PARENT root, same absolute UTC time base (see
    # `_build_one_day_tree`'s docstring) -- illwerke-000001 is all-standstill,
    # illwerke-000002 is all-turbine. If `main()` ever regresses to passing the
    # pooled flat `index.betriebsdaten` list to every run instead of each run's
    # own `index.betriebsdaten_by_day[run.day_root]`, `_betriebsdaten_for_grid`
    # would match BOTH Betriebsdaten files for BOTH runs (they overlap in
    # time), `load_scada_window_means` would average standstill (P=0) and
    # turbine (P=10) samples together into every window, and NEITHER run's
    # actual `report.md` confusion matrix would show a clean, homogeneous GT
    # state anymore -- it would show the same corrupted (turbine/transition)
    # mix for BOTH runs regardless of which day tree is which.
    data_root = tmp_path / "data"
    _build_one_day_tree(
        data_root / "illwerke-000001", day_label="20260601", state="standstill"
    )
    _build_one_day_tree(
        data_root / "illwerke-000002", day_label="20260602", state="turbine"
    )

    monkeypatch.setenv("ROWII_DATA_ROOT", str(data_root))
    monkeypatch.setenv("ROWII_RESULTS_ROOT", str(tmp_path / "results"))

    import run_step1

    exit_code = run_step1.main(["--run", "all", "--variant", "audio", "--clusterer", "kmeans"])
    assert exit_code == 0

    cfg_results_root = tmp_path / "results"
    summary = pd.read_csv(cfg_results_root / "summary.csv")
    assert sorted(summary["run"]) == ["000001-tu", "000002-tu"]

    # The confusion matrix's GT row labels are the ground-truth read directly
    # off the ACTUAL report.md main() wrote to disk -- the one artifact that
    # can only be correct if main()'s own call to run_combo used this run's
    # own day's Betriebsdaten, not a leaked pooled list.
    for run_name, expected_gt_state, forbidden_gt_state in (
        ("000001-tu", "standstill", "turbine"),
        ("000002-tu", "turbine", "standstill"),
    ):
        report_path = cfg_results_root / run_name / "audio-kmeans" / "report.md"
        assert report_path.is_file(), f"missing report for {run_name}"
        report_text = report_path.read_text()
        confusion_section = report_text.split("## Confusion matrix", 1)[1]
        assert f"| {expected_gt_state} |" in confusion_section, (
            f"{run_name}: expected GT row {expected_gt_state!r} in its own confusion "
            f"matrix, got:\n{confusion_section}"
        )
        assert f"| {forbidden_gt_state} |" not in confusion_section, (
            f"{run_name}: found sibling day's GT row {forbidden_gt_state!r} in its "
            f"confusion matrix -- Betriebsdaten leaked across day trees:\n"
            f"{confusion_section}"
        )
