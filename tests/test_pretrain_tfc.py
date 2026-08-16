"""Tests for `scripts/pretrain_tfc.py`: CLI-level
end-to-end tests against synthetic tmp corpus dirs (mirrors
`tests/test_tfc_corpora.py`'s/`tests/test_download_corpora.py`'s own
fixture-builder pattern) plus focused unit tests for the two pieces of new
logic here that have no natural home elsewhere
(`_reservoir_sample`, `_augment_time_view`). No real corpus/download anywhere
in this file -- CPU-forced throughout (`ROWII_FORCE_CPU=1`), matching
`tests/test_tfc_model.py`'s/`tests/test_tfc_wrapper.py`'s own convention.
Whole module is torch-gated at collection time (`pretrain_tfc.py` has no
torch-free code path worth testing without it, unlike `rowii.tfc.wrapper`).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

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
    -- duplicated, not imported: test modules are not a shared
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
# 3. End-to-end: --corpus mimii, tiny synthetic corpus, CPU
#    (--epochs 2 --batch-size 8 --limit-clips 4
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
#    tensors
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


# ---------------------------------------------------------------------------
# 9. --corpus pshp-pool + --continue-from + --out-name: materialize-once npz,
#    cache reuse, checkpoint/sidecar lineage, out-name override. All PSHP
#    plumbing (load_config /
#    discover / iter_target_windows) is monkeypatched -- no real plant data,
#    mirroring tests/test_adapt_beats.py's stubbed-target-windows convention.
# ---------------------------------------------------------------------------

_POOL_WINDOW_VALUES: dict[str, list[float]] = {
    "runA": [1.0, 2.0, 3.0],
    "runB": [10.0, 20.0],
}
"""Per-fake-run constant window values: each yielded window is
`np.full(8000, value)`, so npz row contents pin both the per-run counts AND
the sequential run-concatenation order."""


def _patch_pool_environment(monkeypatch, src_dir: Path, *, windows_per_run, calls):
    """Monkeypatch `pretrain_tfc.load_config` / `.discover` /
    `.iter_target_windows` (module-level names, the adapt_beats/warm_cache
    monkeypatchability precedent) with a synthetic pool: `discover` knows
    exactly *windows_per_run*'s run names, each carrying ONE real (stat-able)
    dummy mic burst file under *src_dir* -- the file-signature
    fingerprint stats these, so content-staleness tests can mutate them; the
    fake iterator yields 8000-sample float32 windows with that run's constant
    values and asserts the contract `target_hz=8000` while recording every
    extra kwarg into `calls["iter_kwargs"]` (the seed-pinning tripwire).
    *calls* counts invocations of both.
    """
    monkeypatch.setattr(
        pretrain_tfc, "load_config",
        lambda env=None: SimpleNamespace(data_root=Path("/pool-env-unused")),
    )

    def _run_with_file(name: str) -> SimpleNamespace:
        run_dir = src_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        burst_path = run_dir / (
            f"{pretrain_tfc._POOL_SIGNATURE_STREAM}_2026-07-08_06-00-00_000000.dat"
        )
        if not burst_path.exists():
            burst_path.write_bytes(name.encode() * 4)
        return SimpleNamespace(
            name=name,
            files={
                pretrain_tfc._POOL_SIGNATURE_STREAM: [
                    SimpleNamespace(path=burst_path)
                ]
            },
        )

    def fake_discover(data_root):
        calls["discover"] += 1
        return SimpleNamespace(
            runs=[_run_with_file(name) for name in windows_per_run]
        )

    def fake_iter_target_windows(run, cfg, *, target_hz, **kwargs):
        calls["iter"] += 1
        calls.setdefault("iter_kwargs", []).append(dict(kwargs))
        assert target_hz == 8000, "pshp-pool must request TF-C's own 8 kHz rate"
        values = windows_per_run[run.name]
        return iter([np.full(8000, v, dtype=np.float32) for v in values])

    monkeypatch.setattr(pretrain_tfc, "discover", fake_discover)
    monkeypatch.setattr(pretrain_tfc, "iter_target_windows", fake_iter_target_windows)


def _stub_train(monkeypatch, captured=None):
    """Replace `pretrain_tfc._train` with a fast stub returning a tiny real
    `TfcModel` (so `_save_checkpoint` still has a genuine state dict to
    persist) while recording every argument -- probing "the
    --continue-from init reaches training BEFORE any step runs"."""
    from rowii.tfc.model import TfcModel
    from rowii.tfc.wrapper import TfcConfig

    def fake_train(windows, cfg, *, epochs, batch_size, lr, seed, init_state=None):
        if captured is not None:
            captured.update(
                n_windows=int(windows.shape[0]), cfg=cfg, epochs=epochs,
                batch_size=batch_size, lr=lr, seed=seed, init_state=init_state,
            )
        torch.manual_seed(0)
        model = TfcModel(TfcConfig(channels=(4, 8)))
        model.eval()
        return model, [0.5] * max(epochs, 1)

    monkeypatch.setattr(pretrain_tfc, "_train", fake_train)


def _write_source_checkpoint(path: Path, *, channels=(4, 8), seed: int = 123):
    """Write a TF-C checkpoint (`load_tfc_model`'s documented
    dict) for --continue-from tests; returns its model state dict."""
    from rowii.tfc.model import TfcModel
    from rowii.tfc.wrapper import TfcConfig

    torch.manual_seed(seed)
    model = TfcModel(TfcConfig(channels=channels))
    state_dict = model.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": dataclasses.asdict(TfcConfig(channels=channels)),
            "model": state_dict,
            "corpus_manifest_sha256": "unknown",
            "epochs": 1,
        },
        path,
    )
    return state_dict


def test_parser_accepts_pshp_pool_and_new_flag_defaults():
    args = pretrain_tfc.build_parser().parse_args(["--corpus", "pshp-pool"])

    assert args.pool_runs == "010726-tu_ph_tu,290626-tu,290626-pu,010726-pu"
    assert args.continue_from is None
    assert args.out_name is None


def test_help_documents_pool_flags(capsys):
    with pytest.raises(SystemExit) as exc_info:
        pretrain_tfc.main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for needle in ("pshp-pool", "--pool-runs", "--continue-from", "--out-name"):
        assert needle in out, f"missing {needle!r} in --help output"


def test_checkpoint_names_and_manifest_dirs_gain_pool_keys(tmp_path):
    assert pretrain_tfc._CHECKPOINT_NAMES["pshp-pool"] == "tfc_audio_pshp.pt"
    # pshp-pool has NO public-corpus download manifest (provenance lives in the
    # repo's own data layout + the pool fingerprint) -- the lookup must degrade
    # gracefully, never KeyError.
    assert pretrain_tfc._MANIFEST_DIRS["pshp-pool"] == ()
    assert pretrain_tfc._corpus_manifest_sha256("pshp-pool", tmp_path) == "unknown"


def test_pshp_pool_end_to_end_materializes_npz_and_checkpoint(tmp_path, monkeypatch):
    calls = {"iter": 0, "discover": 0}
    _patch_pool_environment(
        monkeypatch, tmp_path / "pool-src", windows_per_run=_POOL_WINDOW_VALUES,
        calls=calls,
    )
    out_dir = tmp_path / "out"

    exit_code = pretrain_tfc.main(
        [
            "--corpus", "pshp-pool", "--pool-runs", "runA,runB",
            "--epochs", "1", "--batch-size", "4", "--out", str(out_dir),
        ]
    )

    assert exit_code == 0
    assert calls["iter"] == 2
    assert calls["discover"] == 1

    npz_path = out_dir / "pshp_pool_windows.npz"
    assert npz_path.is_file()
    with np.load(npz_path, allow_pickle=False) as npz:
        assert set(npz.files) == {"windows", "run_names", "per_run_counts", "fingerprint"}
        windows = npz["windows"]
        assert windows.dtype == np.float32
        assert windows.shape == (5, 8000)
        # sequential per-run concatenation, pool order: runA's 3 then runB's 2
        assert windows[:, 0].tolist() == [1.0, 2.0, 3.0, 10.0, 20.0]
        assert npz["run_names"].tolist() == ["runA", "runB"]
        assert npz["per_run_counts"].dtype == np.int64
        assert npz["per_run_counts"].tolist() == [3, 2]
        fingerprint = str(npz["fingerprint"])

    src_dir = tmp_path / "pool-src"
    expected_runs = [
        SimpleNamespace(
            name=name,
            files={
                pretrain_tfc._POOL_SIGNATURE_STREAM: [
                    SimpleNamespace(
                        path=src_dir / name / (
                            f"{pretrain_tfc._POOL_SIGNATURE_STREAM}"
                            "_2026-07-08_06-00-00_000000.dat"
                        )
                    )
                ]
            },
        )
        for name in ("runA", "runB")
    ]
    assert fingerprint == pretrain_tfc._pool_fingerprint(
        ["runA", "runB"], 8000, pretrain_tfc._pool_file_signatures(expected_runs)
    )
    assert len(fingerprint) == 64

    checkpoint_path = out_dir / "tfc_audio_pshp.pt"
    assert checkpoint_path.is_file()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert state["pool_runs"] == ["runA", "runB"]
    assert state["continued_from"] is None
    # the pool fingerprint plays the manifest role: real provenance, not "unknown"
    assert state["corpus_manifest_sha256"] == fingerprint

    model = load_tfc_model(checkpoint_path, torch.device("cpu"))
    assert model.training is False

    sidecar = json.loads(checkpoint_path.with_suffix(".json").read_text())
    assert sidecar["pool_runs"] == ["runA", "runB"]
    assert sidecar["continued_from"] is None
    assert sidecar["pool_fingerprint"] == fingerprint
    assert sidecar["pool_window_counts"] == {"runA": 3, "runB": 2}
    note = sidecar["note"].lower()
    assert "calibration" in note  # windows are calibration-side only (leakage rule)
    assert "universal" in note  # universality framing
    for run_name in ("runa", "runb"):
        assert run_name in note  # the note states the pool runs


def test_pshp_pool_second_invocation_reuses_npz_cache(tmp_path, monkeypatch, caplog):
    calls = {"iter": 0, "discover": 0}
    _patch_pool_environment(
        monkeypatch, tmp_path / "pool-src", windows_per_run=_POOL_WINDOW_VALUES,
        calls=calls,
    )
    _stub_train(monkeypatch)
    out_dir = tmp_path / "out"
    argv = [
        "--corpus", "pshp-pool", "--pool-runs", "runA,runB",
        "--epochs", "1", "--batch-size", "4", "--max-windows", "4", "--out", str(out_dir),
    ]

    assert pretrain_tfc.main(argv) == 0
    assert calls["iter"] == 2
    # the npz always materializes the FULL pool; --max-windows caps training only
    with np.load(out_dir / "pshp_pool_windows.npz", allow_pickle=False) as npz:
        assert npz["windows"].shape[0] == 5

    with caplog.at_level(logging.INFO):
        assert pretrain_tfc.main([*argv, "--out-name", "tfc_audio_pshp_scratch.pt"]) == 0

    assert calls["iter"] == 2  # iter_target_windows NOT called again on a HIT
    # Hardening: discovery + the stat-only file-signature pass DO run
    # on every invocation (freshness validation against the live corpus); only
    # window extraction is skipped on a HIT.
    assert calls["discover"] == 2
    assert any("cache hit" in record.message.lower() for record in caplog.records)
    # --out-name override: the scratch control shares the npz, not the filename
    assert (out_dir / "tfc_audio_pshp_scratch.pt").is_file()
    assert (out_dir / "tfc_audio_pshp_scratch.json").is_file()


def test_pshp_pool_different_pool_list_rematerializes(tmp_path, monkeypatch):
    calls = {"iter": 0, "discover": 0}
    _patch_pool_environment(
        monkeypatch, tmp_path / "pool-src", windows_per_run=_POOL_WINDOW_VALUES,
        calls=calls,
    )
    _stub_train(monkeypatch)
    out_dir = tmp_path / "out"
    base = ["--corpus", "pshp-pool", "--epochs", "1", "--batch-size", "4", "--out", str(out_dir)]

    assert pretrain_tfc.main([*base, "--pool-runs", "runA,runB"]) == 0
    with np.load(out_dir / "pshp_pool_windows.npz", allow_pickle=False) as npz:
        fingerprint_two_runs = str(npz["fingerprint"])
    assert calls["iter"] == 2

    assert pretrain_tfc.main([*base, "--pool-runs", "runA"]) == 0

    assert calls["iter"] == 3  # fingerprint mismatch -> runA re-materialized
    with np.load(out_dir / "pshp_pool_windows.npz", allow_pickle=False) as npz:
        assert npz["run_names"].tolist() == ["runA"]
        assert npz["per_run_counts"].tolist() == [3]
        assert npz["windows"].shape == (3, 8000)
        assert str(npz["fingerprint"]) != fingerprint_two_runs


def test_pshp_pool_unknown_run_exits_2(tmp_path, monkeypatch, capsys):
    calls = {"iter": 0, "discover": 0}
    _patch_pool_environment(
        monkeypatch, tmp_path / "pool-src", windows_per_run={"runA": [1.0]}, calls=calls
    )
    out_dir = tmp_path / "out"

    exit_code = pretrain_tfc.main(
        ["--corpus", "pshp-pool", "--pool-runs", "runA,bogus", "--out", str(out_dir)]
    )

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "bogus" in err
    assert "runA" in err  # available runs are listed
    assert calls["iter"] == 0  # no window was ever drawn
    assert not (out_dir / "pshp_pool_windows.npz").exists()


def test_out_name_override_applies_to_public_corpora(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    _mimii_style_corpus(data_root, n_clips=2, duration_s=2.0)
    _stub_train(monkeypatch)
    out_dir = tmp_path / "out"

    exit_code = pretrain_tfc.main(
        [
            "--corpus", "mimii", "--data-root", str(data_root),
            "--epochs", "1", "--batch-size", "4", "--max-windows", "16",
            "--out", str(out_dir), "--out-name", "custom_audio.pt",
        ]
    )

    assert exit_code == 0
    assert (out_dir / "custom_audio.pt").is_file()
    assert not (out_dir / "tfc_audio.pt").exists()


def test_public_corpus_checkpoint_and_sidecar_carry_null_lineage(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    _mimii_style_corpus(data_root, n_clips=2, duration_s=2.0)
    _stub_train(monkeypatch)
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
    assert state["continued_from"] is None
    assert state["pool_runs"] is None
    sidecar = json.loads((out_dir / "tfc_audio.json").read_text())
    assert sidecar["continued_from"] is None
    assert sidecar["pool_runs"] is None


def test_train_applies_continue_from_init_before_training():
    from rowii.tfc.model import TfcModel
    from rowii.tfc.wrapper import TfcConfig

    cfg = TfcConfig(channels=(4, 8))
    torch.manual_seed(321)
    source_state = TfcModel(cfg).state_dict()
    windows = np.random.default_rng(0).normal(0, 1, (8, 8000)).astype(np.float32)

    # epochs=0: no training step runs, so the returned weights ARE the init --
    # bitwise equality proves the state dict was actually loaded.
    model, epoch_losses = pretrain_tfc._train(
        windows, cfg, epochs=0, batch_size=4, lr=1e-3, seed=7, init_state=source_state,
    )
    assert epoch_losses == []
    loaded = model.state_dict()
    assert set(loaded.keys()) == set(source_state.keys())
    for key in source_state:
        torch.testing.assert_close(loaded[key], source_state[key], rtol=0, atol=0, msg=key)

    # negative control: one real epoch moves the weights away from the init
    trained, _ = pretrain_tfc._train(
        windows, cfg, epochs=1, batch_size=4, lr=1e-2, seed=7, init_state=source_state,
    )
    assert any(not torch.equal(trained.state_dict()[k], source_state[k]) for k in source_state)


def test_continue_from_cli_loads_source_state_and_records_lineage(tmp_path, monkeypatch):
    calls = {"iter": 0, "discover": 0}
    _patch_pool_environment(
        monkeypatch, tmp_path / "pool-src", windows_per_run=_POOL_WINDOW_VALUES,
        calls=calls,
    )
    captured = {}
    _stub_train(monkeypatch, captured)
    source_path = tmp_path / "src" / "tfc_audio.pt"
    source_state = _write_source_checkpoint(source_path, channels=(4, 8))
    out_dir = tmp_path / "out"

    exit_code = pretrain_tfc.main(
        [
            "--corpus", "pshp-pool", "--pool-runs", "runA,runB",
            "--epochs", "1", "--batch-size", "4", "--out", str(out_dir),
            "--continue-from", str(source_path),
        ]
    )

    assert exit_code == 0
    init_state = captured["init_state"]
    assert init_state is not None
    assert set(init_state.keys()) == set(source_state.keys())
    for key in source_state:
        torch.testing.assert_close(init_state[key], source_state[key], rtol=0, atol=0, msg=key)
    # the architecture follows the SOURCE checkpoint's own cfg, not defaults
    assert captured["cfg"].channels == (4, 8)

    state = torch.load(out_dir / "tfc_audio_pshp.pt", map_location="cpu", weights_only=False)
    assert state["continued_from"] == str(source_path.resolve())
    assert state["pool_runs"] == ["runA", "runB"]
    sidecar = json.loads((out_dir / "tfc_audio_pshp.json").read_text())
    assert sidecar["continued_from"] == str(source_path.resolve())
    assert sidecar["pool_runs"] == ["runA", "runB"]


def test_continue_from_missing_file_exits_2(tmp_path, capsys):
    data_root = tmp_path / "empty_data"
    (data_root / "mimii" / "pump_0db").mkdir(parents=True)

    exit_code = pretrain_tfc.main(
        [
            "--corpus", "mimii", "--data-root", str(data_root),
            "--continue-from", str(tmp_path / "missing.pt"),
            "--out", str(tmp_path / "out"),
        ]
    )

    # exit 2 (not the no-windows exit 1): the lineage check precedes any corpus walk
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "continue-from" in err
    assert "missing.pt" in err


# ---------------------------------------------------------------------------
# 10. Hardening: content staleness, seed tripwire, continue-from
#     failure modes, npz self-healing, lineage completeness
# ---------------------------------------------------------------------------


def test_pshp_pool_content_change_without_count_change_rematerializes(
    tmp_path, monkeypatch
):
    """A counts-only fingerprint previously served STALE windows after
    a same-structure re-ingest. With file signatures, mutating a source file's
    bytes+mtime must MISS and re-materialize."""
    import os
    import time

    calls = {"iter": 0, "discover": 0}
    _patch_pool_environment(
        monkeypatch, tmp_path / "pool-src", windows_per_run=_POOL_WINDOW_VALUES,
        calls=calls,
    )
    _stub_train(monkeypatch)
    out_dir = tmp_path / "out"
    argv = [
        "--corpus", "pshp-pool", "--pool-runs", "runA,runB",
        "--epochs", "1", "--batch-size", "4", "--out", str(out_dir),
    ]
    assert pretrain_tfc.main(argv) == 0
    assert calls["iter"] == 2

    burst = (
        tmp_path / "pool-src" / "runA"
        / f"{pretrain_tfc._POOL_SIGNATURE_STREAM}_2026-07-08_06-00-00_000000.dat"
    )
    data = bytearray(burst.read_bytes())
    data[0] ^= 0x01  # same size, different content
    burst.write_bytes(bytes(data))
    bumped = time.time() + 5
    os.utime(burst, (bumped, bumped))

    assert pretrain_tfc.main(argv) == 0
    assert calls["iter"] == 4  # re-materialized, not served stale


def test_pshp_pool_never_threads_a_seed_into_iter_target_windows(
    tmp_path, monkeypatch
):
    """Seed-pinning tripwire: threading --seed into
    iter_target_windows would shuffle scoring-side windows into pretraining
    (leakage). The iterator must be called WITHOUT any seed override."""
    calls = {"iter": 0, "discover": 0}
    _patch_pool_environment(
        monkeypatch, tmp_path / "pool-src", windows_per_run=_POOL_WINDOW_VALUES,
        calls=calls,
    )
    _stub_train(monkeypatch)
    assert pretrain_tfc.main(
        [
            "--corpus", "pshp-pool", "--pool-runs", "runA,runB",
            "--epochs", "1", "--batch-size", "4", "--seed", "99",
            "--out", str(tmp_path / "out"),
        ]
    ) == 0
    assert calls["iter_kwargs"], "iterator was never invoked"
    for kwargs in calls["iter_kwargs"]:
        assert "seed" not in kwargs, (
            "iter_target_windows must run at its own pinned split seed -- a "
            "threaded --seed would leak scoring-side windows into pretraining"
        )


def test_continue_from_garbage_file_exits_2_with_pointed_message(
    tmp_path, monkeypatch, capsys
):
    calls = {"iter": 0, "discover": 0}
    _patch_pool_environment(
        monkeypatch, tmp_path / "pool-src", windows_per_run=_POOL_WINDOW_VALUES,
        calls=calls,
    )
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"\x00\x01\x02 not a checkpoint")
    with pytest.raises(SystemExit) as exc_info:
        pretrain_tfc.main(
            [
                "--corpus", "pshp-pool", "--pool-runs", "runA,runB",
                "--epochs", "1", "--batch-size", "4",
                "--continue-from", str(garbage), "--out", str(tmp_path / "out"),
            ]
        )
    assert exc_info.value.code == 2
    assert "could not be read" in capsys.readouterr().err


def test_continue_from_missing_state_key_exits_2(tmp_path, monkeypatch, capsys):
    """strict=True regression pin: a checkpoint missing one
    parameter must fail LOUDLY -- under strict=False that parameter would
    silently keep its random init."""
    import torch

    from rowii.tfc.model import TfcModel
    from rowii.tfc.wrapper import TfcConfig

    calls = {"iter": 0, "discover": 0}
    _patch_pool_environment(
        monkeypatch, tmp_path / "pool-src", windows_per_run=_POOL_WINDOW_VALUES,
        calls=calls,
    )
    cfg = TfcConfig(channels=(4, 8))
    state = TfcModel(cfg).state_dict()
    dropped = next(iter(state))
    del state[dropped]
    source = tmp_path / "partial.pt"
    torch.save({"cfg": dataclasses.asdict(cfg), "model": state}, source)

    with pytest.raises(SystemExit) as exc_info:
        pretrain_tfc.main(
            [
                "--corpus", "pshp-pool", "--pool-runs", "runA,runB",
                "--epochs", "1", "--batch-size", "4",
                "--continue-from", str(source), "--out", str(tmp_path / "out"),
            ]
        )
    assert exc_info.value.code == 2
    assert "does not match" in capsys.readouterr().err


def test_pshp_pool_empty_pool_exits_1_and_writes_no_npz(tmp_path, monkeypatch, capsys):
    calls = {"iter": 0, "discover": 0}
    _patch_pool_environment(
        monkeypatch, tmp_path / "pool-src", windows_per_run={"runA": []}, calls=calls
    )
    out_dir = tmp_path / "out"
    exit_code = pretrain_tfc.main(
        [
            "--corpus", "pshp-pool", "--pool-runs", "runA",
            "--epochs", "1", "--batch-size", "4", "--out", str(out_dir),
        ]
    )
    assert exit_code == 1
    assert not (out_dir / pretrain_tfc._POOL_WINDOWS_FILENAME).exists()


def test_pshp_pool_malformed_npz_self_heals(tmp_path, monkeypatch, caplog):
    calls = {"iter": 0, "discover": 0}
    _patch_pool_environment(
        monkeypatch, tmp_path / "pool-src", windows_per_run=_POOL_WINDOW_VALUES,
        calls=calls,
    )
    _stub_train(monkeypatch)
    out_dir = tmp_path / "out"
    npz_path = out_dir / pretrain_tfc._POOL_WINDOWS_FILENAME
    out_dir.mkdir(parents=True)
    npz_path.write_bytes(b"this is not an npz archive")

    with caplog.at_level(logging.WARNING):
        exit_code = pretrain_tfc.main(
            [
                "--corpus", "pshp-pool", "--pool-runs", "runA,runB",
                "--epochs", "1", "--batch-size", "4", "--out", str(out_dir),
            ]
        )
    assert exit_code == 0
    assert calls["iter"] == 2  # re-materialized
    assert any("unreadable" in r.getMessage() for r in caplog.records)
    with np.load(npz_path, allow_pickle=False) as healed:
        assert healed["windows"].dtype == np.float32
        assert str(np.asarray(healed["fingerprint"]))  # real fingerprint stored


def test_bearings_checkpoint_carries_null_lineage(tmp_path, monkeypatch):
    _bearings_style_corpus(tmp_path)
    _stub_train(monkeypatch)
    out_dir = tmp_path / "out"
    assert pretrain_tfc.main(
        [
            "--corpus", "bearings", "--data-root", str(tmp_path),
            "--epochs", "1", "--batch-size", "4", "--out", str(out_dir),
        ]
    ) == 0
    import torch

    checkpoint = torch.load(out_dir / "tfc_vib.pt", map_location="cpu", weights_only=False)
    assert checkpoint["continued_from"] is None
    assert checkpoint["pool_runs"] is None
