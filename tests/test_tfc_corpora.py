"""Corpus window-iterator tests (package-4 spec D2, Task 2): synthetic wav/mat
fixtures only -- no real MIMII/CWRU/Paderborn downloads anywhere in this file
(those never run in tests; see `tests/test_download_corpora.py` for the
acquisition script's own synthetic-only tests).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy.io import savemat
from scipy.io.wavfile import write as write_wav

from rowii.tfc.corpora import iter_windows_mat_dir, iter_windows_wav_dir

_RATE_HZ = 16_000


def _write_noise_wav(path, *, duration_s: float, rate_hz: int = _RATE_HZ, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(duration_s * rate_hz))
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, 0.2, n)
    int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    write_wav(path, rate_hz, int16)


class TestIterWindowsWavDir:
    def test_2_5s_16khz_file_yields_two_standardized_8khz_windows(self, tmp_path):
        _write_noise_wav(tmp_path / "clip.wav", duration_s=2.5)

        windows = list(iter_windows_wav_dir(tmp_path))

        assert len(windows) == 2
        for w in windows:
            assert w.shape == (8000,)
            assert w.dtype == np.float32
            assert float(w.mean()) == pytest.approx(0.0, abs=1e-5)
            assert float(w.std()) == pytest.approx(1.0, abs=1e-5)

    def test_short_file_below_window_s_yields_nothing(self, tmp_path):
        _write_noise_wav(tmp_path / "tiny.wav", duration_s=0.1)

        assert list(iter_windows_wav_dir(tmp_path)) == []

    def test_exclude_substring_skips_abnormal_subdir_and_logs_count_once(self, tmp_path, caplog):
        _write_noise_wav(tmp_path / "normal" / "a.wav", duration_s=2.5, seed=1)
        _write_noise_wav(tmp_path / "abnormal" / "b.wav", duration_s=2.5, seed=2)
        _write_noise_wav(tmp_path / "abnormal" / "c.wav", duration_s=2.5, seed=3)

        with caplog.at_level(logging.INFO):
            windows = list(iter_windows_wav_dir(tmp_path))

        assert len(windows) == 2  # only normal/a.wav's 2 windows
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        matching = [m for m in info_messages if "excluded 2" in m]
        assert len(matching) == 1, info_messages  # logged exactly ONCE, at the end

    def test_exclude_substring_none_disables_filtering(self, tmp_path):
        _write_noise_wav(tmp_path / "abnormal" / "b.wav", duration_s=2.5)

        windows = list(iter_windows_wav_dir(tmp_path, exclude_substring=None))

        assert len(windows) == 2

    def test_limit_clips_caps_files_processed(self, tmp_path):
        for i in range(3):
            _write_noise_wav(tmp_path / f"clip_{i:03d}.wav", duration_s=2.5, seed=i)

        windows = list(iter_windows_wav_dir(tmp_path, limit_clips=2))

        assert len(windows) == 4  # 2 files x 2 windows

    def test_limit_clips_counts_too_short_files_too(self, tmp_path):
        _write_noise_wav(tmp_path / "a_tiny.wav", duration_s=0.1, seed=1)
        _write_noise_wav(tmp_path / "b_good.wav", duration_s=2.5, seed=2)

        # sorted order visits a_tiny.wav first; it consumes the whole budget
        # (yields nothing) so b_good.wav is never reached.
        windows = list(iter_windows_wav_dir(tmp_path, limit_clips=1))

        assert windows == []

    def test_stereo_wav_is_mono_mixed_via_channel_averaging(self, tmp_path):
        n = int(2.5 * _RATE_HZ)
        left = np.full(n, 10_000, dtype=np.int16)
        right = np.full(n, -10_000, dtype=np.int16)
        stereo = np.column_stack([left, right])
        write_wav(tmp_path / "stereo.wav", _RATE_HZ, stereo)

        windows = list(iter_windows_wav_dir(tmp_path))

        assert len(windows) == 2
        for w in windows:
            # channel-average is exactly 0 everywhere -> resample and
            # standardize both leave it at exactly 0.
            np.testing.assert_allclose(w, 0.0, atol=1e-5)

    def test_resamples_to_custom_target_hz(self, tmp_path):
        _write_noise_wav(tmp_path / "clip.wav", duration_s=2.5)

        windows = list(iter_windows_wav_dir(tmp_path, target_hz=4000))

        assert len(windows) == 2
        assert all(w.shape == (4000,) for w in windows)

    def test_sorted_recursive_walk_is_deterministic(self, tmp_path):
        for name in ("b.wav", "a.wav"):
            _write_noise_wav(tmp_path / name, duration_s=2.5)
        _write_noise_wav(tmp_path / "sub" / "c.wav", duration_s=2.5)

        first = [w.tobytes() for w in iter_windows_wav_dir(tmp_path)]
        second = [w.tobytes() for w in iter_windows_wav_dir(tmp_path)]

        assert first == second
        assert len(first) == 6  # 3 files x 2 windows


class TestIterWindowsMatDir:
    _NATIVE_HZ = 12_000.0

    def _signal(self, seed: int = 0, duration_s: float = 2.5, native_hz: float | None = None):
        native_hz = self._NATIVE_HZ if native_hz is None else native_hz
        n = int(round(duration_s * native_hz))
        rng = np.random.default_rng(seed)
        return rng.normal(0, 0.2, (n, 1))  # CWRU-style column vector

    def test_de_time_key_yields_standardized_8khz_windows(self, tmp_path):
        savemat(tmp_path / "097.mat", {"X097_DE_time": self._signal()})

        windows = list(iter_windows_mat_dir(tmp_path))

        assert len(windows) == 2
        for w in windows:
            assert w.shape == (8000,)
            assert w.dtype == np.float32
            assert float(w.mean()) == pytest.approx(0.0, abs=1e-5)
            assert float(w.std()) == pytest.approx(1.0, abs=1e-5)

    def test_missing_key_file_is_skipped_with_warning(self, tmp_path, caplog):
        savemat(tmp_path / "a_bad.mat", {"some_other_signal": self._signal(seed=1)})
        savemat(tmp_path / "b_good.mat", {"X097_DE_time": self._signal(seed=2)})

        with caplog.at_level(logging.WARNING):
            windows = list(iter_windows_mat_dir(tmp_path))

        assert len(windows) == 2  # only b_good.mat
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("a_bad.mat" in w for w in warnings), warnings

    def test_limit_clips_caps_files_processed(self, tmp_path):
        for i in range(3):
            savemat(tmp_path / f"f_{i:03d}.mat", {"X097_DE_time": self._signal(seed=i)})

        windows = list(iter_windows_mat_dir(tmp_path, limit_clips=2))

        assert len(windows) == 4

    def test_limit_clips_counts_skipped_missing_key_files_too(self, tmp_path):
        savemat(tmp_path / "a_bad.mat", {"nope": self._signal(seed=1)})
        savemat(tmp_path / "b_good.mat", {"X097_DE_time": self._signal(seed=2)})

        windows = list(iter_windows_mat_dir(tmp_path, limit_clips=1))

        assert windows == []

    def test_custom_key_substring_and_native_hz_paderborn_style(self, tmp_path):
        native_hz = 64_000.0
        savemat(
            tmp_path / "n15_m07_f10_k001_1.mat",
            {"vibration_1": self._signal(native_hz=native_hz).reshape(-1)},
        )

        windows = list(
            iter_windows_mat_dir(tmp_path, key_substring="vibration_1", native_hz=native_hz)
        )

        assert len(windows) == 2
        assert all(w.shape == (8000,) for w in windows)

    def test_short_signal_below_window_s_yields_nothing(self, tmp_path):
        savemat(tmp_path / "tiny.mat", {"X097_DE_time": self._signal(duration_s=0.1)})

        assert list(iter_windows_mat_dir(tmp_path)) == []
