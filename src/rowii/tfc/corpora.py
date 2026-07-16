"""Public-corpus window iterators (package-4 spec D2, Task 2): turn a
directory tree of WAV (MIMII) or MAT (CWRU/Paderborn) files into the same
1-s, 8 kHz, per-window-standardized float32 windows `TfcFeaturizer`
(`rowii.tfc.wrapper`) and the pretraining script (`scripts/pretrain_tfc.py`,
Task 3) both expect -- these two functions are the ONLY place this project
reads a raw public corpus off disk.

Both iterators share one windowing/resample/standardize pipeline (private
helpers below): cut each clip into NON-OVERLAPPING `window_s`-second windows
at the clip's own native rate (a trailing partial window shorter than
`window_s` is dropped, never zero-padded -- this project has no need to
train on partial windows when whole ones are abundant), batch-resample the
whole clip's windows to `target_hz` in one `scipy.signal.resample_poly` call
(the same pad/trim-before-and-after-resample approach
`rowii.tfc.wrapper._resample_to_8khz` uses -- see `_resample_windows`'s
docstring for why that function is REIMPLEMENTED here rather than imported),
then per-window standardize (`_standardize`, the same mean-0/std-1,
1e-8-clamped convention as `rowii.tfc.wrapper._standardize` and
`rowii.tfc.model.freq_view`).

Both functions are plain generators: nothing is read from disk until the
caller actually iterates (a `for` loop / `list(...)` / `next()`), and only
ONE clip's windows are held in memory at a time -- important for MIMII's
real corpus size (tens of thousands of ~10-s clips per machine type), never
assumed by any test here (all tests use a handful of synthetic fixture
files, per this package's downloads-never-run-in-tests rule).
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from scipy.io import loadmat, wavfile
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


def _resample_windows(windows: np.ndarray, native_hz: float, target_hz: int) -> np.ndarray:
    """`(B, S)` float64 windows at *native_hz* -> `(B, target_hz)` float64,
    via `scipy.signal.resample_poly`.

    Reimplements (rather than imports) `rowii.tfc.wrapper._resample_to_8khz`'s
    pad/trim-before-and-after-resample approach: that function is hardcoded
    to an output rate of 8 kHz, whereas this module's callers
    (`iter_windows_wav_dir`/`iter_windows_mat_dir`) take `target_hz` as a
    caller-configurable parameter, so its body cannot be called directly --
    see that function's own docstring for the detailed rationale behind both
    pad/trim steps (BEFORE: makes `resample_poly`'s output-length formula
    collapse to exactly `target_hz` for an input of exactly
    `round(native_hz)` samples; AFTER: defensively guarantees that length
    regardless of FIR edge effects or a fractional *native_hz*). Every
    window in *windows* is assumed to already be exactly one `window_s`-long
    clip at *native_hz* (`_cut_windows`'s contract), so all rows share the
    same `target_in`/pad math.
    """
    target_in = int(round(native_hz))
    n_samples = windows.shape[1]
    if n_samples < target_in:
        padded = np.pad(windows, ((0, 0), (0, target_in - n_samples)))
    elif n_samples > target_in:
        padded = windows[:, :target_in]
    else:
        padded = windows

    resampled = resample_poly(padded, target_hz, target_in, axis=1)

    n_out = resampled.shape[1]
    if n_out < target_hz:
        resampled = np.pad(resampled, ((0, 0), (0, target_hz - n_out)))
    elif n_out > target_hz:
        resampled = resampled[:, :target_hz]
    return np.ascontiguousarray(resampled, dtype=np.float64)


def _standardize(batch: np.ndarray) -> np.ndarray:
    """Per-window (row) zero-mean/unit-std standardization, `1e-8`-floored
    std to avoid a divide-by-zero on a silent (constant) window -- the same
    convention as `rowii.tfc.wrapper._standardize` and
    `rowii.tfc.model.freq_view`."""
    mean = batch.mean(axis=1, keepdims=True)
    std = np.clip(batch.std(axis=1, keepdims=True), 1e-8, None)
    return (batch - mean) / std


def _cut_windows(signal: np.ndarray, native_hz: float, window_s: float) -> np.ndarray:
    """Non-overlapping `window_s`-second windows of 1-D *signal* at
    *native_hz*, as `(n_windows, window_len)`. A trailing partial window
    shorter than `window_s` is dropped (never zero-padded). Returns an empty
    `(0, window_len)` array -- never raises -- when *signal* is shorter than
    one window; callers simply see zero windows for that clip."""
    window_len = int(round(window_s * native_hz))
    if window_len <= 0:
        return np.empty((0, 0), dtype=signal.dtype)
    n_windows = signal.shape[0] // window_len
    if n_windows == 0:
        return np.empty((0, window_len), dtype=signal.dtype)
    trimmed = signal[: n_windows * window_len]
    return trimmed.reshape(n_windows, window_len)


def _wav_to_mono_float(raw: np.ndarray) -> np.ndarray:
    """`scipy.io.wavfile.read`'s raw samples -> float64 mono in ~[-1, 1].

    Integer PCM (the overwhelming majority of real WAV files, including
    MIMII's 16-bit recordings) is normalized by its dtype's maximum
    representable value (e.g. 32767 for int16); float PCM is already in
    that range and is passed through as float64 unchanged. Multi-channel
    (stereo) data is mono-mixed by averaging across channels -- the same
    rule `rowii.tfc.wrapper.TfcFeaturizer.transform` and
    `rowii.signals.beats.BeatsFeaturizer.transform` both use for `(..., C)`
    windows.
    """
    if np.issubdtype(raw.dtype, np.integer):
        mono = raw.astype(np.float64) / np.iinfo(raw.dtype).max
    else:
        mono = raw.astype(np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    return mono


def iter_windows_wav_dir(
    root: Path,
    *,
    exclude_substring: str | None = "abnormal",
    window_s: float = 1.0,
    target_hz: int = 8000,
    limit_clips: int | None = None,
) -> Iterator[np.ndarray]:
    """Recursively walk *root* for `*.wav` files (sorted, deterministic) and
    yield standardized, `target_hz`-resampled, non-overlapping `window_s`
    windows -- one `(target_hz,)` float32 array per window. Written for
    MIMII's own layout (spec D2: `data/public/mimii/pump_0db/<machine>/
    id_<NN>/{normal,abnormal}/*.wav`) but makes no MIMII-specific path
    assumptions beyond the `exclude_substring` filter, so it walks any WAV
    tree.

    Args:
        root: Directory to walk recursively for `*.wav` files.
        exclude_substring: Any file whose POSIX path contains this substring
            is skipped entirely -- MIMII's per-machine `abnormal/`
            subdirectory by default (self-supervised pretraining wants
            normal-dominated data; abnormal clips are excluded outright, not
            merely down-weighted). `None` disables filtering (every `*.wav`
            under *root* is used). The total number of files skipped this
            way is logged ONCE, as a single INFO message after the walk
            completes -- MIMII has thousands of `abnormal/` files, so a
            per-file message would flood the log.
        window_s: Window length in seconds, cut non-overlapping at each
            clip's own native sample rate; a trailing partial window shorter
            than `window_s` is dropped (never zero-padded).
        target_hz: Output sample rate every window is resampled to
            (`_resample_windows`). Default 8000 matches the TF-C model's
            fixed input rate (`rowii.tfc.wrapper._TFC_SAMPLE_RATE_HZ`).
        limit_clips: If given, stop after this many files (in sorted order,
            post-`exclude_substring`) have been OPENED -- a file that itself
            yields zero windows (shorter than `window_s`) still counts
            against this budget, since it was genuinely attempted; only
            excluded files are free. `None` processes every non-excluded
            file under *root*.

    Yields:
        `(target_hz,)` float32 arrays, per-window standardized (mean 0, std
        1, `1e-8`-floored).
    """
    excluded = 0
    processed = 0
    for path in sorted(root.rglob("*.wav"), key=lambda p: p.as_posix()):
        if exclude_substring is not None and exclude_substring in path.as_posix():
            excluded += 1
            continue
        if limit_clips is not None and processed >= limit_clips:
            break
        processed += 1

        native_hz, raw = wavfile.read(path)
        mono = _wav_to_mono_float(raw)
        windows = _cut_windows(mono, native_hz, window_s)
        if windows.shape[0] == 0:
            continue

        resampled = _resample_windows(windows, float(native_hz), target_hz)
        standardized = _standardize(resampled)
        for row in standardized:
            yield row.astype(np.float32)

    logger.info(
        "iter_windows_wav_dir: excluded %d file(s) under %s matching exclude_substring=%r "
        "(%d file(s) processed)",
        excluded, root, exclude_substring, processed,
    )


def iter_windows_mat_dir(
    root: Path,
    *,
    key_substring: str = "DE_time",
    native_hz: float = 12_000.0,
    window_s: float = 1.0,
    target_hz: int = 8000,
    limit_clips: int | None = None,
) -> Iterator[np.ndarray]:
    """Recursively walk *root* for `*.mat` files (sorted, deterministic) and
    yield standardized, `target_hz`-resampled, non-overlapping `window_s`
    windows from each file's chosen signal variable.

    CWRU convention (this function's defaults): the CWRU Bearing Data
    Center's `.mat` files store each accelerometer channel as a flat,
    top-level MATLAB variable named `X<file-number>_DE_time` (drive end),
    `_FE_time` (fan end), or `_BA_time` (base) -- e.g. `X097_DE_time` inside
    `97.mat`. The drive-end channel is this project's chosen vibration
    signal (`key_substring="DE_time"`) and is sampled at 12 kHz in both the
    "Normal Baseline Data" and "12k Drive End Bearing Fault Data" file sets
    this project downloads (`native_hz=12_000.0`; see
    `scripts/download_corpora.py`'s `cwru` table).

    Paderborn callers MUST override both defaults: Paderborn KAt `.mat`
    files are sampled at 64 kHz and do not use CWRU's flat `*_DE_time`
    naming (their convention nests each channel inside a struct variable
    named after the file itself, e.g. a `Y` field of named sub-channels --
    NOT a flat top-level array). This function only ever matches a FLAT,
    top-level variable name (see `Args` below), so a genuine Paderborn
    `.mat` file will hit this function's missing-key skip path as-is; the
    Task-2 completion report documents this gap for whoever wires Paderborn
    into `scripts/pretrain_tfc.py` (Task 3).

    Args:
        root: Directory to walk recursively for `*.mat` files.
        key_substring: A file is scanned (via `scipy.io.loadmat`) for the
            FIRST non-dunder top-level variable whose name contains this
            substring (iteration order = whatever `loadmat` returns, which
            is deterministic per file). A file with no matching variable is
            SKIPPED (not an error) with a WARNING logged naming that file.
        native_hz: The chosen signal's sample rate -- NOT read from the
            `.mat` file itself (neither CWRU's nor Paderborn's `.mat` files
            carry a per-file rate metadata field this loader can depend on);
            the caller supplies it directly, matching whichever corpus/file
            set *root* actually holds.
        window_s: See `iter_windows_wav_dir`.
        target_hz: See `iter_windows_wav_dir`.
        limit_clips: If given, stop after this many files (in sorted order)
            have been OPENED -- a file with no matching *key_substring*
            variable, or one shorter than one window, still counts against
            this budget, mirroring `iter_windows_wav_dir`'s identical rule.
            `None` processes every file under *root*.

    Yields:
        `(target_hz,)` float32 arrays, per-window standardized.
    """
    processed = 0
    for path in sorted(root.rglob("*.mat"), key=lambda p: p.as_posix()):
        if limit_clips is not None and processed >= limit_clips:
            break
        processed += 1

        data = loadmat(path)
        signal_name = next(
            (name for name in data if not name.startswith("__") and key_substring in name),
            None,
        )
        if signal_name is None:
            logger.warning(
                "iter_windows_mat_dir: no variable containing %r in %s -- skipping",
                key_substring, path,
            )
            continue

        signal = np.asarray(data[signal_name], dtype=np.float64).reshape(-1)
        windows = _cut_windows(signal, native_hz, window_s)
        if windows.shape[0] == 0:
            continue

        resampled = _resample_windows(windows, native_hz, target_hz)
        standardized = _standardize(resampled)
        for row in standardized:
            yield row.astype(np.float32)
