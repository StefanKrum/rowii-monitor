"""Public-corpus window iterators (package-4 spec D2 Task 2, extended by
Task 3's `iter_windows_paderborn_dir`): turn a directory tree of WAV (MIMII)
or MAT (CWRU/Paderborn) files into the same 1-s, 8 kHz,
per-window-standardized float32 windows `TfcFeaturizer` (`rowii.tfc.wrapper`)
and the pretraining script (`scripts/pretrain_tfc.py`, Task 3) both expect --
these three functions are the ONLY place this project reads a raw public
corpus off disk.

All three iterators share one windowing/resample/standardize pipeline
(private helpers below): cut each clip into NON-OVERLAPPING `window_s`-second
windows at the clip's own native rate (a trailing partial window shorter than
`window_s` is dropped, never zero-padded -- this project has no need to
train on partial windows when whole ones are abundant), batch-resample the
whole clip's windows to `target_hz` in one `scipy.signal.resample_poly` call
(the same pad/trim-before-and-after-resample approach
`rowii.tfc.wrapper._resample_to_8khz` uses -- see `_resample_windows`'s
docstring for why that function is REIMPLEMENTED here rather than imported),
then per-window standardize (`_standardize`, the same mean-0/std-1,
1e-8-clamped convention as `rowii.tfc.wrapper._standardize` and
`rowii.tfc.model.freq_view`). `iter_windows_paderborn_dir` differs from the
other two only in HOW it locates its signal on disk (a nested MATLAB struct,
not a flat `.wav`/`.mat` variable) -- see its own docstring.

All three functions are plain generators: nothing is read from disk until the
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


_PADERBORN_NATIVE_HZ = 64_000.0
"""Nominal sample rate of Paderborn KAt's `vibration_1`/`phase_current_*`
channels (the "HostService" raster). Confirmed both by community
documentation (Task 2's URL-research concerns) and, now, by this task's
sanctioned real-file smoke check (`.superpowers/sdd/task-3-report.md`):
`Data.shape[0] / Description.Measurement.Length` measured ~64000.25 Hz
across 4 real K001/K002 files -- within 0.0004% of this nominal constant, the
same "nominal, not measured-per-file" convention `iter_windows_mat_dir`
already uses for CWRU's 12 kHz (`native_hz`'s docstring there)."""

_PADERBORN_VIBRATION_CHANNEL = "vibration_1"
"""The `Y[i].Name` this function searches for (substring match, mirroring
`iter_windows_mat_dir`'s `key_substring` convention) -- Paderborn KAt's
vibration accelerometer channel, per orchestrator resolution 2 and confirmed
present (as exactly one of 7 named `Y` entries) in every real file this task
inspected."""


def _extract_paderborn_vibration(path: Path) -> np.ndarray | None:
    """Extract the `"vibration_1"` channel's raw float64 samples from one
    Paderborn KAt `.mat` file's NESTED struct layout (`iter_windows_paderborn_dir`'s
    module-level docstring section documents the confirmed real layout in
    full). Returns `None` -- NEVER raises -- for any file that does not match
    that layout in any of the ways checked below: a file that `loadmat`
    itself cannot parse at all (Task 2's completion report flags a known
    Paderborn v7.3/HDF5-format failure mode as a real possibility, on top of
    ordinary corruption/truncation), one whose root struct has no `Y` field,
    or one whose `Y` entries include no channel named `"vibration_1"` --
    orchestrator resolution 2's explicit "skip file with logged warning on
    failure -- NEVER crash the corpus build" requirement, which is why this
    function wraps the ENTIRE load-and-navigate sequence in one broad
    `except Exception`, deliberately broader than this module's other
    per-file error handling (`iter_windows_mat_dir` only guards the
    "missing key" case, not `loadmat` itself, since CWRU's `.mat` files are
    reliably flat and simple -- Paderborn's real-world failure modes are not
    yet as well-characterized, so this function trusts nothing beyond "some
    exception happened").

    Args:
        path: One `.mat` file.

    Returns:
        The channel's 1-D float64 samples, or `None` if extraction failed
        for any reason (a WARNING naming *path* and the reason is logged
        before returning `None`).
    """
    try:
        data = loadmat(str(path), struct_as_record=False, squeeze_me=True)
        top_keys = [name for name in data if not name.startswith("__")]
        if not top_keys:
            logger.warning(
                "iter_windows_paderborn_dir: no top-level variable in %s -- skipping", path
            )
            return None

        root = data[top_keys[0]]
        # squeeze_me=True collapses a length-1 struct ARRAY down to a bare
        # scalar mat_struct (confirmed during this task's real-layout
        # research) -- np.atleast_1d normalizes both shapes to a uniform,
        # indexable ndarray without disturbing the already-multi-element case.
        channels = np.atleast_1d(root.Y)
        for channel in channels:
            name = getattr(channel, "Name", None)
            if isinstance(name, str) and _PADERBORN_VIBRATION_CHANNEL in name:
                return np.asarray(channel.Data, dtype=np.float64).reshape(-1)
    except Exception as exc:  # noqa: BLE001 -- "never crash the corpus build" (see docstring)
        logger.warning(
            "iter_windows_paderborn_dir: failed to parse %s (%s: %s) -- skipping",
            path, type(exc).__name__, exc,
        )
        return None

    logger.warning(
        "iter_windows_paderborn_dir: no %r channel found in %s -- skipping",
        _PADERBORN_VIBRATION_CHANNEL, path,
    )
    return None


def iter_windows_paderborn_dir(
    root: Path,
    *,
    window_s: float = 1.0,
    target_hz: int = 8000,
    limit_clips: int | None = None,
) -> Iterator[np.ndarray]:
    """Recursively walk *root* for `*.mat` files (sorted, deterministic) and
    yield standardized, `target_hz`-resampled, non-overlapping `window_s`
    windows from each file's `"vibration_1"` channel -- the Paderborn KAt
    counterpart to `iter_windows_mat_dir` (CWRU), written separately rather
    than as another `iter_windows_mat_dir` parameterization because real
    Paderborn `.mat` files nest their channels inside a struct (a per-file
    `Y` field, itself a struct ARRAY with `Name`/`Data` sub-fields), not a
    flat top-level variable `iter_windows_mat_dir`'s `key_substring` search
    can match -- Task 2's own docstring/completion-report already flagged
    this gap; `_extract_paderborn_vibration` (this function's only real
    difference from `iter_windows_mat_dir`) closes it.

    Confirmed real layout (this task's sanctioned read-only smoke check
    against `data/public/paderborn/K001/K001/N15_M07_F04_K001_1.mat` and 3
    further real K001/K002 files, `.superpowers/sdd/task-3-report.md`):
    `scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)` yields
    a root `mat_struct` (named after the file's own stem -- never relied on
    here, only that it is the sole non-dunder top-level key) with fields
    `Info`/`X`/`Y`/`Description`; `root.Y` is a length-7 array of per-channel
    `mat_struct`s (`force`, `phase_current_1`, `phase_current_2`, `speed`,
    `temp_2_bearing_module`, `torque`, `vibration_1`, in that order across
    every real file inspected -- though this function searches by `Name`,
    never by a fixed index, so a different order or channel SET would still
    resolve correctly as long as SOME entry is literally named
    `"vibration_1"`); that channel's `.Data` is a 1-D float64 array sampled
    at a measured ~64000.25 Hz (`_PADERBORN_NATIVE_HZ`'s docstring).

    Any file that does not match this layout -- including one `loadmat`
    cannot parse at all (a plausible real failure mode per Task 2's
    completion report: some Paderborn `.mat` files are reportedly v7.3/HDF5
    format) -- is SKIPPED with a WARNING naming the file
    (`_extract_paderborn_vibration`), never raised: this function must NEVER
    crash the corpus build over one malformed file (orchestrator resolution
    2's explicit contract).

    Args:
        root: Directory to walk recursively for `*.mat` files.
        window_s: See `iter_windows_wav_dir`.
        target_hz: See `iter_windows_wav_dir`.
        limit_clips: If given, stop after this many files (in sorted order)
            have been OPENED -- a file that fails to parse, has no `Y`
            field, or has no `"vibration_1"` channel still counts against
            this budget, mirroring `iter_windows_mat_dir`'s identical rule
            (its own `limit_clips` docstring). `None` processes every file
            under *root*.

    Yields:
        `(target_hz,)` float32 arrays, per-window standardized.
    """
    processed = 0
    for path in sorted(root.rglob("*.mat"), key=lambda p: p.as_posix()):
        if limit_clips is not None and processed >= limit_clips:
            break
        processed += 1

        signal = _extract_paderborn_vibration(path)
        if signal is None:
            continue

        windows = _cut_windows(signal, _PADERBORN_NATIVE_HZ, window_s)
        if windows.shape[0] == 0:
            continue

        resampled = _resample_windows(windows, _PADERBORN_NATIVE_HZ, target_hz)
        standardized = _standardize(resampled)
        for row in standardized:
            yield row.astype(np.float32)
