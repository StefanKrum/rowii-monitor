"""Corpus window-iterator tests: synthetic wav/mat
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

from rowii.tfc.corpora import (
    iter_windows_mat_dir,
    iter_windows_paderborn_dir,
    iter_windows_wav_dir,
)

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


class TestIterWindowsPaderbornDir:
    """`iter_windows_paderborn_dir`: real
    Paderborn KAt `.mat` files hold a NESTED MATLAB struct -- unlike CWRU's
    flat top-level variable (`iter_windows_mat_dir`'s contract), confirmed
    against real files during a sanctioned read-only smoke check
    ("Paderborn real-file observation"):
    `loadmat(path, struct_as_record=False, squeeze_me=True)` on a
    real KAt file yields a root `mat_struct` (named after the file's own
    stem -- but this function never relies on that name, only on it being
    the sole non-dunder top-level key) with fields `Info`/`X`/`Y`/
    `Description`; `Y` is an array of per-channel `mat_struct`s, each with a
    `.Name` string and a `.Data` 1-D float64 array; the vibration channel is
    the entry whose `Name == "vibration_1"`, measured at ~64000.25 Hz across
    4 real files (matches the nominal `native_hz=64_000.0` this function
    hardcodes to within 0.0004%).

    `_write_paderborn_style_mat` below mimics exactly that layout (a `Y`
    struct ARRAY, not a scalar struct -- scipy's documented
    structured-object-dtype-array recipe, the only way `savemat` produces a
    MATLAB struct ARRAY rather than a single struct) -- verified against the
    real files' structure, not merely "plausible".
    """

    _NATIVE_HZ = 64_000.0

    def _vibration_signal(self, seed: int = 0, duration_s: float = 2.5) -> np.ndarray:
        n = int(round(duration_s * self._NATIVE_HZ))
        rng = np.random.default_rng(seed)
        return rng.normal(0, 0.2, n)

    def _write_paderborn_style_mat(self, path, *, channels: dict[str, np.ndarray]) -> None:
        """*channels*: `{name: 1-D data array}`, at least one entry -- written
        as a `Y` struct array with `Name`/`Data` fields, nested inside a
        root struct (variable name "root" here; real files use the file's
        own stem instead, deliberately different here to double-check this
        function never depends on that particular name).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        dt = np.dtype([("Name", "O"), ("Data", "O")])
        y = np.zeros((len(channels),), dtype=dt)
        for i, (name, data) in enumerate(channels.items()):
            y[i]["Name"] = name
            y[i]["Data"] = np.asarray(data, dtype=np.float64)
        savemat(path, {"root": {"Y": y}})

    def test_extracts_vibration_1_channel_at_64khz(self, tmp_path):
        self._write_paderborn_style_mat(
            tmp_path / "N15_M07_F04_K001_1.mat",
            channels={
                "force": np.zeros(100),
                "phase_current_1": np.zeros(100),
                "vibration_1": self._vibration_signal(),
            },
        )

        windows = list(iter_windows_paderborn_dir(tmp_path))

        assert len(windows) == 2  # 2.5 s @ nominal 64 kHz -> 2 whole 1-s windows
        for w in windows:
            assert w.shape == (8000,)
            assert w.dtype == np.float32
            assert float(w.mean()) == pytest.approx(0.0, abs=1e-5)
            assert float(w.std()) == pytest.approx(1.0, abs=1e-5)

    def test_single_channel_y_squeeze_edge_case_still_extracts(self, tmp_path):
        # scipy.io.loadmat's squeeze_me=True collapses a LENGTH-1 struct array
        # down to a bare scalar mat_struct (confirmed during this task's
        # real-layout research) -- a genuine edge case a single-vibration-
        # channel-only file would hit; the adapter must handle it via
        # np.atleast_1d, not assume Y is always already an ndarray.
        self._write_paderborn_style_mat(
            tmp_path / "single.mat", channels={"vibration_1": self._vibration_signal(seed=1)}
        )

        windows = list(iter_windows_paderborn_dir(tmp_path))

        assert len(windows) == 2

    def test_missing_y_field_is_skipped_with_warning(self, tmp_path, caplog):
        savemat(tmp_path / "bad.mat", {"root": {"NotY": {"foo": 1}}})

        with caplog.at_level(logging.WARNING):
            windows = list(iter_windows_paderborn_dir(tmp_path))

        assert windows == []
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("bad.mat" in w for w in warnings), warnings

    def test_no_vibration_1_channel_is_skipped_with_warning(self, tmp_path, caplog):
        self._write_paderborn_style_mat(
            tmp_path / "bad.mat",
            channels={"force": np.zeros(100), "torque": np.zeros(100)},
        )

        with caplog.at_level(logging.WARNING):
            windows = list(iter_windows_paderborn_dir(tmp_path))

        assert windows == []
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("bad.mat" in w for w in warnings), warnings

    def test_corrupt_file_is_skipped_with_warning_never_crashes(self, tmp_path, caplog):
        # A v7.3/HDF5-format .mat (loadmat can't parse those at all) or any
        # other genuinely malformed file must never crash the corpus build
        # (the explicit "NEVER crash" requirement) --
        # a plain non-mat byte blob exercises the same failure path.
        (tmp_path / "corrupt.mat").write_bytes(b"not a real mat file")

        with caplog.at_level(logging.WARNING):
            windows = list(iter_windows_paderborn_dir(tmp_path))  # must not raise

        assert windows == []
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("corrupt.mat" in w for w in warnings), warnings

    def test_limit_clips_caps_files_processed(self, tmp_path):
        for i in range(3):
            self._write_paderborn_style_mat(
                tmp_path / f"f_{i:03d}.mat",
                channels={"vibration_1": self._vibration_signal(seed=i)},
            )

        windows = list(iter_windows_paderborn_dir(tmp_path, limit_clips=2))

        assert len(windows) == 4

    def test_limit_clips_counts_skipped_files_too(self, tmp_path):
        savemat(tmp_path / "a_bad.mat", {"root": {"NotY": {"foo": 1}}})
        self._write_paderborn_style_mat(
            tmp_path / "b_good.mat", channels={"vibration_1": self._vibration_signal(seed=2)}
        )

        windows = list(iter_windows_paderborn_dir(tmp_path, limit_clips=1))

        assert windows == []  # a_bad.mat (sorted first) alone consumed the budget

    def test_short_signal_below_window_s_yields_nothing(self, tmp_path):
        self._write_paderborn_style_mat(
            tmp_path / "tiny.mat",
            channels={"vibration_1": self._vibration_signal(seed=3, duration_s=0.1)},
        )

        assert list(iter_windows_paderborn_dir(tmp_path)) == []

    def test_sorted_recursive_walk_is_deterministic(self, tmp_path):
        for name in ("b.mat", "a.mat"):
            self._write_paderborn_style_mat(
                tmp_path / name, channels={"vibration_1": self._vibration_signal()}
            )

        first = [w.tobytes() for w in iter_windows_paderborn_dir(tmp_path)]
        second = [w.tobytes() for w in iter_windows_paderborn_dir(tmp_path)]

        assert first == second
        assert len(first) == 4  # 2 files x 2 windows
