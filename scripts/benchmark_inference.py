"""End-to-end inference benchmark harness.

Measures, per featurizer configuration, what a deployment budget needs
measured TOGETHER: model size on disk, parameter count, peak
RSS delta, and per-window wall latency at several batch sizes -- END-TO-END from
raw 50 kHz windows, i.e. INCLUDING each featurizer's own preprocessing
(resampling / fbank / log-mel), because "the preprocessing cost of feature
extraction belongs in the budget alongside inference" (this evaluation's own
Deployment and compactness).

Configurations map onto the production featurizers via
`rowii.pipeline._featurizer_for_stream` -- the EXACT constructor the sweep
pipeline uses, so a benchmark row measures the same object the pipeline runs:

- `handcrafted` -> variant `audio` (numpy; 0 params)
- `logmel`      -> variant `logmel` (numpy; 0 params)
- `beats`       -> variant `audio-beats` (fp32 BEATs; needs ROWII_BEATS_CHECKPOINT)
- `beats-int8`  -> variant `audio-beats` with ROWII_BEATS_INT8_CHECKPOINT set
                   (CPU-only by construction -- the quantized kernels have no
                   MPS/CUDA backend; an `mps` request is skipped with a note)
- `tfc`         -> variant `audio-tfc` (needs ROWII_TFC_AUDIO_CHECKPOINT)
- `student`     -> variant `audio-student` (needs ROWII_STUDENT_CHECKPOINT)

A configuration whose checkpoint env is unset is SKIPPED with a log line, never
an error -- the harness reports what is measurable on this machine (the
graceful-skip rule; silent omission would misread as "measured and absent").
Device control goes through the same `ROWII_FORCE_CPU` seam `best_device()`
honours everywhere else; numpy configs always report device `cpu`.

Latency protocol: per (config, device, batch_size): 2 warmup batches, then
`--n-batches` measured batches of fresh synthetic (or real, `--source
run:<name>`) raw windows; the row reports median wall time per WINDOW in ms.
Peak RSS is `resource.getrusage(RUSAGE_SELF).ru_maxrss` (bytes on macOS) delta
across construction + first measured batch -- an upper-bound style, monotonic
measure, documented as such in the output.

Real-window source (`--source run:<name>`) reads ONE burst file of the run's
primary mic stream read-only (a `@pytest.mark.data`-style path used only by the
orchestrated execution, never by unit tests).
"""
from __future__ import annotations

import argparse
import logging
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.config import Config, load_config  # noqa: E402
from rowii.pipeline import _featurizer_for_stream  # noqa: E402

logger = logging.getLogger(__name__)

_MIC_STREAM = "RAWGeneratorMic__0"
_RAW_RATE_HZ = 50_000.0
_RAW_SAMPLES = 50_000

_CONFIG_VARIANTS: dict[str, str] = {
    "handcrafted": "audio",
    "logmel": "logmel",
    "beats": "audio-beats",
    "beats-int8": "audio-beats",
    "tfc": "audio-tfc",
    "student": "audio-student",
}
_TORCH_CONFIGS = ("beats", "beats-int8", "tfc", "student")

_CSV_COLUMNS = (
    "config", "device", "batch_size", "n_params", "size_mb",
    "latency_ms_per_window", "peak_rss_mb",
)


def _checkpoint_for(config_name: str, cfg: Config) -> Path | None:
    """The on-disk artifact whose availability gates *config_name* (None for the
    numpy configs, which need no checkpoint)."""
    return {
        "handcrafted": None,
        "logmel": None,
        "beats": cfg.beats_checkpoint,
        "beats-int8": cfg.beats_int8_checkpoint,
        "tfc": cfg.tfc_audio_checkpoint,
        "student": cfg.student_checkpoint,
    }[config_name]


def _count_params(config_name: str, checkpoint: Path) -> int:
    """Parameter count from the checkpoint itself (state-dict formats sum tensor
    sizes; the int8 module pickle sums live parameters)."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        return int(sum(int(np.prod(t.shape)) for t in payload["model"].values()))
    return int(sum(p.numel() for p in payload.parameters()))


def _synthetic_windows(n: int, seed: int = 7) -> np.ndarray:
    """(n, 50000, 1) float32 -- 3-D per the per-stream featurizer contract (the
    handcrafted `AudioFeaturizer` requires an explicit channel axis; every other
    featurizer accepts 3-D and mono-mixes it)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.1, (n, _RAW_SAMPLES, 1)).astype(np.float32)


def _real_windows(run_name: str, n: int, cfg: Config) -> np.ndarray:
    """First *n* whole 1-s windows of the run's first primary-mic burst file
    (read-only; mono-mixed)."""
    from rowii.io.dataset import discover
    from rowii.io.gantner import read_gantner

    index = discover(cfg.data_root)
    run = next((r for r in index.runs if r.name == run_name), None)
    if run is None:
        raise SystemExit(f"benchmark_inference: unknown run {run_name!r}")
    burst = sorted(run.files[_MIC_STREAM], key=lambda f: f.start_utc_hint)[0]
    gf = read_gantner(burst.path)
    mono = gf.data.mean(axis=1) if gf.data.ndim == 2 else gf.data
    usable = min(n, mono.shape[0] // _RAW_SAMPLES)
    if usable < n:
        logger.warning(
            "benchmark_inference: burst file holds only %d whole windows "
            "(requested %d) -- using %d", usable, n, usable,
        )
    out = mono[: usable * _RAW_SAMPLES].reshape(usable, _RAW_SAMPLES, 1)
    return np.asarray(out, dtype=np.float32)


def _measure(
    config_name: str, cfg: Config, windows: np.ndarray, batch_size: int,
    n_batches: int,
) -> tuple[float, float]:
    """(median latency ms/window, peak RSS delta MB) for one configuration.

    Constructs the featurizer fresh (the RSS delta covers model load) and runs
    2 warmup batches before the measured ones. Batches cycle over *windows*.
    """
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    featurizer = _featurizer_for_stream(_MIC_STREAM, _CONFIG_VARIANTS[config_name], cfg)

    if windows.shape[0] < batch_size:
        # Tile up to the requested batch size so a "b=256" row really measures
        # 256-window batches (final-review finding: with the 200-window default
        # pool, the old cycling silently measured 200-window batches under a
        # b=256 label). Repeated windows are fine for a latency measurement --
        # the compute is identical.
        reps = -(-batch_size // windows.shape[0])
        windows = np.tile(windows, (reps, 1, 1))
        logger.info(
            "benchmark_inference: window pool smaller than batch size -- tiled "
            "%dx to %d windows so every batch truly holds %d",
            reps, windows.shape[0], batch_size,
        )

    def batch(i: int) -> np.ndarray:
        start = (i * batch_size) % max(1, windows.shape[0] - batch_size + 1)
        return windows[start : start + batch_size]

    for i in range(2):  # warmup (also triggers lazy model load)
        featurizer.transform(batch(i), _RAW_RATE_HZ)
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    per_window_ms: list[float] = []
    for i in range(n_batches):
        b = batch(i)
        t0 = time.perf_counter()
        featurizer.transform(b, _RAW_RATE_HZ)
        elapsed = time.perf_counter() - t0
        per_window_ms.append(1000.0 * elapsed / b.shape[0])
    return statistics.median(per_window_ms), (rss_after - rss_before) / 1e6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--configs", default="handcrafted,logmel,beats,beats-int8,tfc,student",
        help="Comma list of configurations (unknown name: exit 2).",
    )
    parser.add_argument("--n-windows", type=int, default=200)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--batch-sizes", default="1,256")
    parser.add_argument(
        "--devices", default="cpu",
        help="Comma list of cpu,mps -- device is applied via ROWII_FORCE_CPU.",
    )
    parser.add_argument(
        "--source", default="synthetic",
        help='"synthetic" (default) or "run:<name>" (reads one real burst file).',
    )
    parser.add_argument("--out", type=Path, default=Path("results/benchmarks"))
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in _CONFIG_VARIANTS]
    if unknown:
        parser.error(
            f"unknown config(s) {unknown!r}; known: {sorted(_CONFIG_VARIANTS)}"
        )
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    bad_devices = [d for d in devices if d not in ("cpu", "mps")]
    if bad_devices:
        parser.error(f"unknown device(s) {bad_devices!r}; known: cpu, mps")

    import os

    cfg = load_config()
    if args.source == "synthetic":
        windows = _synthetic_windows(args.n_windows)
    elif args.source.startswith("run:"):
        windows = _real_windows(args.source[4:], args.n_windows, cfg)
    else:
        parser.error(f'--source must be "synthetic" or "run:<name>", got {args.source!r}')

    rows: list[dict[str, object]] = []
    for config_name in configs:
        checkpoint = _checkpoint_for(config_name, cfg)
        needs_checkpoint = config_name in _TORCH_CONFIGS
        if needs_checkpoint and checkpoint is None:
            logger.info(
                "benchmark_inference: skipping %r -- its checkpoint env is unset",
                config_name,
            )
            continue
        if needs_checkpoint:
            assert checkpoint is not None  # the unset case was skipped above
            if not checkpoint.is_file():
                logger.info(
                    "benchmark_inference: skipping %r -- its checkpoint %s "
                    "does not exist",
                    config_name, checkpoint,
                )
                continue
            n_params = _count_params(config_name, checkpoint)
            size_mb = checkpoint.stat().st_size / 1e6
        else:
            n_params = 0
            size_mb = 0.0

        for device in devices:
            if device == "mps" and config_name in ("handcrafted", "logmel"):
                continue  # numpy configs have no device axis
            if device == "mps" and config_name == "beats-int8":
                logger.info(
                    "benchmark_inference: skipping beats-int8 on mps -- dynamic "
                    "quantized kernels are CPU-only"
                )
                continue
            if device == "cpu":
                os.environ["ROWII_FORCE_CPU"] = "1"
            else:
                os.environ.pop("ROWII_FORCE_CPU", None)
            for batch_size in batch_sizes:
                latency, rss = _measure(
                    config_name, cfg, windows, batch_size, args.n_batches
                )
                rows.append({
                    "config": config_name, "device": device,
                    "batch_size": batch_size, "n_params": n_params,
                    "size_mb": round(size_mb, 3),
                    "latency_ms_per_window": round(latency, 4),
                    "peak_rss_mb": round(rss, 1),
                })
                logger.info(
                    "benchmark_inference: %s/%s b=%d -> %.3f ms/window",
                    config_name, device, batch_size, latency,
                )
    os.environ.pop("ROWII_FORCE_CPU", None)

    args.out.mkdir(parents=True, exist_ok=True)
    import csv

    csv_path = args.out / "inference.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Inference benchmark (end-to-end incl. preprocessing)",
        "",
        "| " + " | ".join(_CSV_COLUMNS) + " |",
        "|" + "|".join("---" for _ in _CSV_COLUMNS) + "|",
    ]
    md_lines += [
        "| " + " | ".join(str(r[c]) for c in _CSV_COLUMNS) + " |" for r in rows
    ]
    md_lines += [
        "",
        "Peak RSS is a monotonic upper bound (ru_maxrss delta across model load "
        "+ first batch). Latency = median ms per window over the measured "
        "batches, end-to-end from raw 50 kHz windows.",
    ]
    (args.out / "inference.md").write_text("\n".join(md_lines) + "\n")
    logger.info("benchmark_inference: wrote %s (%d rows)", csv_path, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
