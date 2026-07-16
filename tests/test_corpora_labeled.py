"""Labeled MIMII clip-iterator tests (package-6 pillar-3, plan Task 4):
synthetic wav trees only -- no real MIMII downloads anywhere in this file
(this repo's downloads-never-run-in-tests rule; `tests/test_tfc_corpora.py`
sets the synthetic-fixture convention this file mirrors, including its
`_write_noise_wav` int16 helper).
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write as write_wav

from rowii.tfc.corpora import LabeledClip, iter_labeled_clips_wav_dir

_RATE_HZ = 16_000


def _write_noise_wav(path, *, duration_s: float, rate_hz: int = _RATE_HZ, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(duration_s * rate_hz))
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, 0.2, n)
    int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    write_wav(path, rate_hz, int16)


def _build_mimii_style_tree(
    root,
    *,
    machine_ids=("id_00", "id_02"),
    n_normal: int = 3,
    n_abnormal: int = 2,
    duration_s: float = 2.0,
    rate_hz: int = _RATE_HZ,
) -> None:
    """`root/pump/id_*/{normal,abnormal}/<NNNNNNNN>.wav` -- MIMII's real
    layout (machine-type directory between root and the `id_*` machine
    dirs), so the iterator's recursive `**/id_*` matching is exercised, not
    just a flat `root/id_*`. Each clip gets a distinct noise seed so
    determinism checks compare genuinely different per-clip bytes."""
    seed = 0
    for mid in machine_ids:
        for class_name, n_clips in (("normal", n_normal), ("abnormal", n_abnormal)):
            for i in range(n_clips):
                _write_noise_wav(
                    root / "pump" / mid / class_name / f"{i:08d}.wav",
                    duration_s=duration_s,
                    rate_hz=rate_hz,
                    seed=seed,
                )
                seed += 1


class TestIterLabeledClipsWavDir:
    def test_both_classes_yielded_with_correct_labels_ids_and_counts(self, tmp_path):
        _build_mimii_style_tree(tmp_path)

        clips = list(iter_labeled_clips_wav_dir(tmp_path))

        assert len(clips) == 10  # 2 machines x (3 normal + 2 abnormal)
        assert all(isinstance(c, LabeledClip) for c in clips)
        counts = Counter((c.machine_id, c.label) for c in clips)
        assert counts == {
            ("id_00", 0): 3,
            ("id_00", 1): 2,
            ("id_02", 0): 3,
            ("id_02", 1): 2,
        }
        for c in clips:
            parent = Path(c.path).parent.name
            assert parent in ("normal", "abnormal")
            assert c.label == (1 if parent == "abnormal" else 0)
            assert c.path.endswith(".wav")

    def test_windows_shape_dtype_and_per_window_standardization(self, tmp_path):
        _build_mimii_style_tree(tmp_path, machine_ids=("id_00",), n_normal=1, n_abnormal=1)

        clips = list(iter_labeled_clips_wav_dir(tmp_path))

        assert len(clips) == 2
        for c in clips:
            assert c.windows.shape == (2, 16_000)  # 2 s @ 16 kHz -> two 1-s windows
            assert c.windows.dtype == np.float32
            means = c.windows.mean(axis=1, dtype=np.float64)
            stds = c.windows.astype(np.float64).std(axis=1)
            np.testing.assert_allclose(means, 0.0, atol=1e-5)
            np.testing.assert_allclose(stds, 1.0, atol=1e-5)

    def test_determinism_two_runs_identical_order_and_bytes(self, tmp_path):
        _build_mimii_style_tree(tmp_path)

        first = list(iter_labeled_clips_wav_dir(tmp_path))
        second = list(iter_labeled_clips_wav_dir(tmp_path))

        assert [(c.path, c.label, c.machine_id) for c in first] == [
            (c.path, c.label, c.machine_id) for c in second
        ]
        assert [c.windows.tobytes() for c in first] == [c.windows.tobytes() for c in second]
        # The documented order is the fully sorted POSIX-path order (abnormal
        # sorts before normal within each machine dir).
        assert [c.path for c in first] == sorted(c.path for c in first)

    def test_limit_clips_per_class_keeps_one_per_id_class_and_logs_counts(self, tmp_path, caplog):
        _build_mimii_style_tree(tmp_path)

        with caplog.at_level(logging.INFO):
            clips = list(iter_labeled_clips_wav_dir(tmp_path, limit_clips_per_class=1))

        counts = Counter((c.machine_id, c.label) for c in clips)
        assert counts == {
            ("id_00", 0): 1,
            ("id_00", 1): 1,
            ("id_02", 0): 1,
            ("id_02", 1): 1,
        }
        # The cap is applied in sorted order: the first file of each class.
        assert all(c.path.endswith("00000000.wav") for c in clips)
        infos = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert sum("kept 1 of 3" in m for m in infos) == 2, infos  # normal, per machine
        assert sum("kept 1 of 2" in m for m in infos) == 2, infos  # abnormal, per machine

    def test_machine_ids_filters_to_named_machines(self, tmp_path):
        _build_mimii_style_tree(tmp_path)

        clips = list(iter_labeled_clips_wav_dir(tmp_path, machine_ids=["id_00"]))

        assert len(clips) == 5  # 3 normal + 2 abnormal, id_00 only
        assert {c.machine_id for c in clips} == {"id_00"}

    def test_resamples_32khz_clips_to_16khz_windows(self, tmp_path):
        _build_mimii_style_tree(
            tmp_path, machine_ids=("id_00",), n_normal=1, n_abnormal=1, rate_hz=32_000
        )

        clips = list(iter_labeled_clips_wav_dir(tmp_path))

        assert len(clips) == 2
        for c in clips:
            assert c.windows.shape == (2, 16_000)  # 2 s @ 32 kHz -> two windows @ 16 kHz
            assert c.windows.dtype == np.float32

    def test_trailing_partial_window_is_dropped(self, tmp_path):
        _build_mimii_style_tree(
            tmp_path, machine_ids=("id_00",), n_normal=1, n_abnormal=0, duration_s=2.5
        )

        clips = list(iter_labeled_clips_wav_dir(tmp_path))

        assert len(clips) == 1  # abnormal/ never created -> silently absent
        assert clips[0].label == 0
        assert clips[0].windows.shape == (2, 16_000)  # 2.5 s -> 2 whole windows, tail dropped

    def test_zero_machine_dirs_logs_warning_and_yields_nothing(self, tmp_path, caplog):
        _write_noise_wav(tmp_path / "stray.wav", duration_s=2.0)  # wav outside any id_* dir

        with caplog.at_level(logging.WARNING):
            clips = list(iter_labeled_clips_wav_dir(tmp_path))

        assert clips == []
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("no id_*" in m for m in warnings), warnings
