"""Tests for `scripts/pretrain_tfc.py` (package-4 spec D3, Task 3): CLI-level
end-to-end tests against synthetic tmp corpus dirs (mirrors
`tests/test_tfc_corpora.py`'s/`tests/test_download_corpora.py`'s own
fixture-builder pattern) plus focused unit tests for the two pieces of new
logic this task adds that have no natural home in Task 1/2's own test files
(`_reservoir_sample`, `_augment_time_view`). No real corpus/download anywhere
in this file -- CPU-forced throughout (`ROWII_FORCE_CPU=1`), matching
`tests/test_tfc_model.py`'s/`tests/test_tfc_wrapper.py`'s own convention.
Whole module is torch-gated at collection time (`pretrain_tfc.py` has no
torch-free code path worth testing without it, unlike `rowii.tfc.wrapper`).
"""
from __future__ import annotations

import hashlib
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from scipy.io.wavfile import write as write_wav

torch = pytest.importorskip("torch")

from rowii.tfc.wrapper import TfcFeaturizer, load_tfc_model  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import pretrain_tfc  # noqa: E402


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setenv("ROWII_FORCE_CPU", "1")


def _write_noise_wav(path, *, duration_s: float, rate_hz: int = 16_000, seed: int = 0) -> None:
    """Mirrors `tests/test_tfc_corpora.py`'s own helper of the same name
    (Task 2) -- duplicated, not imported: test modules are not a shared
    library other test modules import from in this project."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(duration_s * rate_hz))
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, 0.2, n)
    int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    write_wav(path, rate_hz, int16)


def _mimii_style_corpus(root: Path, *, n_clips: int = 6, duration_s: float = 3.0) -> None:
    """A tmp `mimii/pump_0db/pump/id_00/{normal,abnormal}` tree mirroring the
    real MIMII layout `--corpus mimii` walks (`iter_windows_wav_dir`'s
    docstring): *n_clips* normal clips (enough windows to exercise batching
    across >1 batch at the tests' own `--batch-size`) plus one abnormal clip
    that must never be sampled from (`exclude_substring="abnormal"`).
    """
    base = root / "mimii" / "pump_0db" / "pump" / "id_00"
    for i in range(n_clips):
        _write_noise_wav(base / "normal" / f"{i:08d}.wav", duration_s=duration_s, seed=i)
    _write_noise_wav(base / "abnormal" / "00000000.wav", duration_s=duration_s, seed=999)


def _bearings_style_corpus(root: Path) -> None:
    """A tmp `cwru/` (flat `X097_DE_time`, 12 kHz) + `paderborn/` (nested
    `vibration_1`, 64 kHz) pair, mirroring the real layouts
    `iter_windows_mat_dir`/`iter_windows_paderborn_dir` each walk."""
    n_cwru = int(round(2.5 * 12_000.0))
    (root / "cwru").mkdir(parents=True)
    savemat(
        root / "cwru" / "97.mat",
        {"X097_DE_time": np.random.default_rng(0).normal(0, 0.2, (n_cwru, 1))},
    )

    n_pb = int(round(2.5 * 64_000.0))
    dt = np.dtype([("Name", "O"), ("Data", "O")])
    y = np.zeros((1,), dtype=dt)
    y[0]["Name"] = "vibration_1"
    y[0]["Data"] = np.random.default_rng(1).normal(0, 0.2, n_pb)
    (root / "paderborn").mkdir(parents=True)
    savemat(root / "paderborn" / "N15_M07_F04_K001_1.mat", {"root": {"Y": y}})


# ---------------------------------------------------------------------------
# 1. --help / parser basics
# ---------------------------------------------------------------------------


def test_help_exits_zero_and_documents_every_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        pretrain_tfc.main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for flag in (
        "--corpus", "--data-root", "--epochs", "--batch-size", "--lr",
        "--seed", "--limit-clips", "--max-windows", "--out",
    ):
        assert flag in out, f"missing {flag!r} in --help output"


def test_build_parser_defaults():
    args = pretrain_tfc.build_parser().parse_args(["--corpus", "mimii"])
    assert args.data_root == Path("data/public")
    assert args.epochs == 40
    assert args.batch_size == 256
    assert args.lr == pytest.approx(1e-3)
    assert args.seed == 7
    assert args.limit_clips is None
    assert args.max_windows == 200_000
    assert args.out == Path("models/pretrained/tfc/")


def test_corpus_flag_is_required():
    with pytest.raises(SystemExit) as exc_info:
        pretrain_tfc.build_parser().parse_args([])
    assert exc_info.value.code == 2


def test_unknown_corpus_exits_2(capsys):
    with pytest.raises(SystemExit) as exc_info:
        pretrain_tfc.main(["--corpus", "bogus-corpus"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "bogus-corpus" in err


# ---------------------------------------------------------------------------
# 2. torch-import guard (mirrors tests/test_warm_cache.py's
#    _import_beats_or_exit equivalent)
# ---------------------------------------------------------------------------


def test_import_torch_or_exit_raises_systemexit_with_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit) as exc_info:
        pretrain_tfc._import_torch_or_exit()

    message = str(exc_info.value)
    assert "torch" in message
    assert "pip install -e '.[beats]'" in message


# ---------------------------------------------------------------------------
# 3. End-to-end: --corpus mimii, tiny synthetic corpus, CPU (resolution 5's
#    literal parameters: --epochs 2 --batch-size 8 --limit-clips 4
#    --max-windows 64)
# ---------------------------------------------------------------------------


def test_end_to_end_mimii_checkpoint_loads_and_featurizes(tmp_path):
    data_root = tmp_path / "data"
    _mimii_style_corpus(data_root)
    out_dir = tmp_path / "out"

    exit_code = pretrain_tfc.main(
        [
            "--corpus", "mimii",
            "--data-root", str(data_root),
            "--epochs", "2",
            "--batch-size", "8",
            "--limit-clips", "4",
            "--max-windows", "64",
            "--out", str(out_dir),
        ]
    )

    assert exit_code == 0
    checkpoint_path = out_dir / "tfc_audio.pt"
    assert checkpoint_path.is_file()

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert state["epochs"] == 2
    assert state["corpus_manifest_sha256"] == "unknown"  # no MANIFEST.json in this fixture

    model = load_tfc_model(checkpoint_path, torch.device("cpu"))
    assert model.training is False

    featurizer = TfcFeaturizer(checkpoint=checkpoint_path)
    windows = np.random.default_rng(0).normal(0, 1, (3, 8000))
    embeddings = featurizer.transform(windows, 8000.0)
    assert embeddings.shape == (3, 256)
    assert embeddings.dtype == np.float64
    assert np.isfinite(embeddings).all()


def test_end_to_end_bearings_checkpoint_chains_cwru_and_paderborn(tmp_path):
    data_root = tmp_path / "data"
    _bearings_style_corpus(data_root)
    out_dir = tmp_path / "out"

    exit_code = pretrain_tfc.main(
        [
            "--corpus", "bearings",
            "--data-root", str(data_root),
            "--epochs", "1",
            "--batch-size", "4",
            "--max-windows", "32",
            "--out", str(out_dir),
        ]
    )

    assert exit_code == 0
    checkpoint_path = out_dir / "tfc_vib.pt"
    assert checkpoint_path.is_file()

    featurizer = TfcFeaturizer(checkpoint=checkpoint_path)
    embeddings = featurizer.transform(np.random.default_rng(2).normal(0, 1, (2, 8000)), 8000.0)
    assert embeddings.shape == (2, 256)
    assert np.isfinite(embeddings).all()


def test_end_to_end_records_manifest_sha256_when_present(tmp_path):
    data_root = tmp_path / "data"
    _mimii_style_corpus(data_root, n_clips=2, duration_s=2.0)
    manifest = data_root / "mimii" / "MANIFEST.json"
    manifest.write_text('[{"filename": "0_dB_pump.zip"}]')
    expected_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()

    out_dir = tmp_path / "out"
    exit_code = pretrain_tfc.main(
        [
            "--corpus", "mimii", "--data-root", str(data_root),
            "--epochs", "1", "--batch-size", "4", "--max-windows", "16",
            "--out", str(out_dir),
        ]
    )
    assert exit_code == 0

    state = torch.load(out_dir / "tfc_audio.pt", map_location="cpu", weights_only=False)
    assert state["corpus_manifest_sha256"] == expected_sha256


def test_no_windows_found_exits_nonzero(tmp_path, capsys):
    data_root = tmp_path / "empty_data"
    (data_root / "mimii" / "pump_0db").mkdir(parents=True)

    exit_code = pretrain_tfc.main(
        ["--corpus", "mimii", "--data-root", str(data_root), "--out", str(tmp_path / "out")]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "no windows" in err.lower()


# ---------------------------------------------------------------------------
# 4. Determinism: two full runs, same seed -> identical checkpoint state_dict
#    tensors (resolution 5's literal requirement)
# ---------------------------------------------------------------------------


def test_determinism_same_seed_produces_identical_state_dict(tmp_path):
    data_root = tmp_path / "data"
    _mimii_style_corpus(data_root)

    argv_base = [
        "--corpus", "mimii",
        "--data-root", str(data_root),
        "--epochs", "2",
        "--batch-size", "8",
        "--limit-clips", "4",
        "--max-windows", "64",
        "--seed", "7",
    ]
    out1, out2 = tmp_path / "out1", tmp_path / "out2"

    assert pretrain_tfc.main([*argv_base, "--out", str(out1)]) == 0
    assert pretrain_tfc.main([*argv_base, "--out", str(out2)]) == 0

    state1 = torch.load(out1 / "tfc_audio.pt", map_location="cpu", weights_only=False)["model"]
    state2 = torch.load(out2 / "tfc_audio.pt", map_location="cpu", weights_only=False)["model"]

    assert state1.keys() == state2.keys()
    compared = 0
    for key in state1:
        torch.testing.assert_close(state1[key], state2[key], rtol=0, atol=0, msg=f"key={key}")
        compared += 1
    assert compared > 0


def test_determinism_different_seed_produces_different_state_dict(tmp_path):
    data_root = tmp_path / "data"
    _mimii_style_corpus(data_root)

    argv_base = [
        "--corpus", "mimii",
        "--data-root", str(data_root),
        "--epochs", "2",
        "--batch-size", "8",
        "--limit-clips", "4",
        "--max-windows", "64",
    ]
    out1, out2 = tmp_path / "out1", tmp_path / "out2"

    assert pretrain_tfc.main([*argv_base, "--seed", "7", "--out", str(out1)]) == 0
    assert pretrain_tfc.main([*argv_base, "--seed", "8", "--out", str(out2)]) == 0

    state1 = torch.load(out1 / "tfc_audio.pt", map_location="cpu", weights_only=False)["model"]
    state2 = torch.load(out2 / "tfc_audio.pt", map_location="cpu", weights_only=False)["model"]

    any_different = any(not torch.equal(state1[k], state2[k]) for k in state1)
    assert any_different


# ---------------------------------------------------------------------------
# 5. Loss: finite (checked precisely on _train's own return value) and
#    logged (checked loosely via caplog on main())
# ---------------------------------------------------------------------------


def test_train_returns_one_finite_loss_per_epoch(tmp_path):
    from rowii.tfc.wrapper import TfcConfig

    windows = np.random.default_rng(0).normal(0, 1, (16, 8000)).astype(np.float32)
    cfg = TfcConfig(channels=(4, 8))

    model, epoch_losses = pretrain_tfc._train(
        windows, cfg, epochs=3, batch_size=4, lr=1e-3, seed=7,
    )

    assert len(epoch_losses) == 3
    assert all(math.isfinite(v) for v in epoch_losses)
    assert model.training is False  # left in eval mode, mirrors _train_autoencoder


def test_main_logs_a_loss_message_per_epoch(tmp_path, caplog):
    data_root = tmp_path / "data"
    _mimii_style_corpus(data_root, n_clips=2, duration_s=2.0)

    with caplog.at_level(logging.INFO):
        exit_code = pretrain_tfc.main(
            [
                "--corpus", "mimii", "--data-root", str(data_root),
                "--epochs", "3", "--batch-size", "4", "--max-windows", "16",
                "--out", str(tmp_path / "out"),
            ]
        )

    assert exit_code == 0
    loss_messages = [r.message for r in caplog.records if "loss" in r.message.lower()]
    assert len(loss_messages) >= 3


# ---------------------------------------------------------------------------
# 6. _reservoir_sample: standalone, torch-free streaming reservoir sample
# ---------------------------------------------------------------------------


class TestReservoirSample:
    def test_returns_all_items_when_stream_smaller_than_cap(self):
        windows = [np.full(4, float(i), dtype=np.float32) for i in range(5)]

        sample, n_seen = pretrain_tfc._reservoir_sample(iter(windows), 10, np.random.default_rng(0))

        assert n_seen == 5
        assert sample.shape == (5, 4)
        assert sorted(sample[:, 0].tolist()) == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_caps_at_max_windows_when_stream_larger(self):
        windows = [np.full(4, float(i), dtype=np.float32) for i in range(1000)]

        sample, n_seen = pretrain_tfc._reservoir_sample(iter(windows), 50, np.random.default_rng(0))

        assert n_seen == 1000
        assert sample.shape == (50, 4)
        indices = sample[:, 0].astype(int).tolist()
        assert all(0 <= v < 1000 for v in indices)
        assert len(set(indices)) == 50  # no duplicate original windows

    def test_deterministic_given_seed(self):
        def make_stream():
            return (np.full(4, float(i), dtype=np.float32) for i in range(200))

        sample1, _ = pretrain_tfc._reservoir_sample(make_stream(), 20, np.random.default_rng(3))
        sample2, _ = pretrain_tfc._reservoir_sample(make_stream(), 20, np.random.default_rng(3))

        np.testing.assert_array_equal(sample1, sample2)

    def test_different_seed_differs(self):
        def make_stream():
            return (np.full(4, float(i), dtype=np.float32) for i in range(200))

        sample1, _ = pretrain_tfc._reservoir_sample(make_stream(), 20, np.random.default_rng(1))
        sample2, _ = pretrain_tfc._reservoir_sample(make_stream(), 20, np.random.default_rng(2))

        assert not np.array_equal(sample1, sample2)

    def test_empty_stream_returns_empty(self):
        sample, n_seen = pretrain_tfc._reservoir_sample(iter([]), 10, np.random.default_rng(0))

        assert n_seen == 0
        assert sample.shape == (0, 0)

    def test_max_windows_zero_returns_empty(self):
        windows = [np.full(4, float(i), dtype=np.float32) for i in range(5)]

        sample, n_seen = pretrain_tfc._reservoir_sample(iter(windows), 0, np.random.default_rng(0))

        assert n_seen == 5
        assert sample.shape == (0, 4)


# ---------------------------------------------------------------------------
# 7. _augment_time_view: TIME-view augmentation (jitter + scale + zero-mask),
#    all draws from the caller's own torch.Generator
# ---------------------------------------------------------------------------


class TestAugmentTimeView:
    def test_shape_and_dtype_preserved(self):
        x = torch.zeros(4, 100)
        gen = torch.Generator().manual_seed(0)

        out = pretrain_tfc._augment_time_view(x, gen)

        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_deterministic_given_generator_seed(self):
        x = torch.randn(4, 100, generator=torch.Generator().manual_seed(0))

        out1 = pretrain_tfc._augment_time_view(x, torch.Generator().manual_seed(1))
        out2 = pretrain_tfc._augment_time_view(x, torch.Generator().manual_seed(1))

        torch.testing.assert_close(out1, out2, rtol=0, atol=0)

    def test_different_generator_seed_differs(self):
        x = torch.randn(4, 100, generator=torch.Generator().manual_seed(0))

        out1 = pretrain_tfc._augment_time_view(x, torch.Generator().manual_seed(1))
        out2 = pretrain_tfc._augment_time_view(x, torch.Generator().manual_seed(2))

        assert not torch.equal(out1, out2)

    def test_actually_changes_the_signal(self):
        x = torch.ones(4, 200)
        gen = torch.Generator().manual_seed(0)

        out = pretrain_tfc._augment_time_view(x, gen)

        assert not torch.equal(out, x)

    def test_zero_mask_fraction_stays_near_bound(self):
        # x is a nonzero constant, so any exact-0.0 output entry can only come
        # from the zero-mask (scale*1.0 + gaussian jitter is never exactly 0).
        x = torch.ones(8, 8000)
        gen = torch.Generator().manual_seed(0)

        out = pretrain_tfc._augment_time_view(x, gen)

        zero_frac = (out == 0.0).float().mean(dim=1)
        # generous slack around the 10% ceiling: the per-window threshold
        # itself is random in [0, 0.10], plus binomial sampling noise at
        # n_samples=8000.
        assert (zero_frac <= 0.15).all()


# ---------------------------------------------------------------------------
# 8. _corpus_manifest_sha256
# ---------------------------------------------------------------------------


class TestCorpusManifestSha256:
    def test_returns_unknown_when_no_manifest_present(self, tmp_path):
        assert pretrain_tfc._corpus_manifest_sha256("mimii", tmp_path) == "unknown"

    def test_hashes_mimii_manifest_when_present(self, tmp_path):
        manifest = tmp_path / "mimii" / "MANIFEST.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text('[{"filename": "x"}]')
        expected = hashlib.sha256(manifest.read_bytes()).hexdigest()

        assert pretrain_tfc._corpus_manifest_sha256("mimii", tmp_path) == expected

    def test_combines_cwru_and_paderborn_manifests_for_bearings(self, tmp_path):
        cwru = tmp_path / "cwru" / "MANIFEST.json"
        cwru.parent.mkdir(parents=True)
        cwru.write_text('[{"filename": "97.mat"}]')
        paderborn = tmp_path / "paderborn" / "MANIFEST.json"
        paderborn.parent.mkdir(parents=True)
        paderborn.write_text('[{"filename": "K001.rar"}]')
        expected = hashlib.sha256(cwru.read_bytes() + paderborn.read_bytes()).hexdigest()

        assert pretrain_tfc._corpus_manifest_sha256("bearings", tmp_path) == expected

    def test_bearings_uses_whichever_manifest_exists_when_one_is_missing(self, tmp_path):
        cwru = tmp_path / "cwru" / "MANIFEST.json"
        cwru.parent.mkdir(parents=True)
        cwru.write_text('[{"filename": "97.mat"}]')
        expected = hashlib.sha256(cwru.read_bytes()).hexdigest()

        assert pretrain_tfc._corpus_manifest_sha256("bearings", tmp_path) == expected
