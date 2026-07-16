"""Post-training INT8 dynamic quantization CLI (Step-2 package-5 spec D6,
Task 5): loads the frozen fp32 BEATs encoder from `ROWII_BEATS_CHECKPOINT` and
applies `torch.ao.quantization.quantize_dynamic` (`nn.Linear` layers only,
`torch.qint8`) -- a WEIGHT-ONLY quantization scheme that needs no calibration
data (unlike static/observer-based quantization), so this script is a pure
load -> quantize -> save transform: no run, no windows, no training loop.

Usage: `quantize_beats.py --out models/adapted/beats_int8.pt` -> saves the
quantized module ITSELF (`torch.save(quantized_module, out)` -- a MODULE
pickle, NOT the `{"cfg","model"}` state-dict checkpoint `rowii.signals.
beats_model.load_beats_model` reads: `torch.ao.quantization.quantize_dynamic`'s
dynamically-quantized `nn.Linear` submodules pack their int8 weight plus a
separate scale/zero-point into an opaque `LinearPackedParams` object, not a
plain tensor a `state_dict` round trip alone can reconstruct -- see `rowii.
signals.beats_model.load_quantized_beats_model`'s own docstring, the dedicated
loader `BeatsFeaturizer`'s `int8_model_path` branch uses) + a sidecar
`<out-stem>.json` (`source_checkpoint`, `size_fp32_bytes`, `size_int8_bytes`,
`created_at`). CPU-only inference (documented here and at the loader):
dynamically quantized Linear kernels are a CPU-only PyTorch backend -- this
project's actual deployment target for the compact/quantized pole is an
on-premise server with no GPU (design spec D6).

`inplace=True` (READ THIS -- load-bearing, not a style choice, and NOT
documented anywhere in the design spec): `torch.ao.quantization.
quantize_dynamic`'s default (`inplace=False`) internally `copy.deepcopy()`s
the model before swapping `nn.Linear` leaves for quantized counterparts, but
the vendored `TransformerEncoder` (`rowii.vendor.beats.backbone.
TransformerEncoder.__init__`) wraps its positional conv in `torch.nn.utils.
weight_norm` -- a weight-normed module's `.weight` is a DERIVED, non-leaf
autograd tensor, and `Tensor.__deepcopy__` refuses to copy those (verified by
hand against the real vendored class: `RuntimeError: Only Tensors created
explicitly by the user (graph leaves) support the deepcopy protocol`).
`inplace=True` quantizes the loaded model IN PLACE instead, sidestepping
`deepcopy` entirely -- safe here since the model is freshly loaded above and
never used afterward except via the returned (same-object) quantized module.

`torch.backends.quantized.engine` (READ THIS TOO -- also verified by hand, also
undocumented in the design spec): defaults to `"none"`; `quantize_dynamic`
itself raises `RuntimeError` ("NoQEngine") unless this is set to one of
`torch.backends.quantized.supported_engines` FIRST -- see `rowii.signals.
beats_model.select_quantized_engine`, which this script calls before
quantizing (the SAME helper `BeatsFeaturizer`'s int8-loading branch calls
before `torch.load`-ing the result back).

Torch import discipline (plan's Global Constraints: eager only in dedicated
model modules, lazy everywhere else): every torch-touching name here is
imported lazily inside the function that needs it, INCLUDING `load_beats_
model`, which is a deliberate exception (mirrors `scripts/adapt_beats.py`'s/
`scripts/distill_beats.py`'s own module docstrings: a module-level import
specifically so `tests/test_quantize_beats.py` can `monkeypatch.setattr(
quantize_beats, "load_beats_model", ...)` -- a name only ever bound inside a
function body cannot be monkeypatched from outside it, since the function's
own `from ... import ...` statement would just re-resolve the REAL symbol on
every call).

`cosine_drift` (numpy-only, no torch): mean row-paired cosine similarity
between two `(N, D)` embedding matrices, exposed here for the execution phase
(design spec D9) -- comparing fp32 vs int8 `audio-beats` embeddings for the
SAME real cached windows, in the SAME order, is the INT8 embedding-drift
evidence the design calls for.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.config import Config, load_config  # noqa: E402
from rowii.pipeline import _BEATS_INSTALL_HINT  # noqa: E402
from rowii.signals.beats_model import load_beats_model  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("models/adapted/beats_int8.pt")

_QUANTIZED_NOTE = (
    "module pickle (torch.save of the quantized nn.Module itself), NOT state-dict "
    'compatible with load_beats_model\'s {"cfg","model"} format -- load via '
    "BeatsFeaturizer(int8_model_path=...) / rowii.signals.beats_model."
    "load_quantized_beats_model; CPU-only inference (dynamic-quantized Linear kernels "
    "are a CPU-only PyTorch backend, matching the on-premise deployment target, "
    "design spec D6)"
)


def cosine_drift(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity between ROW-paired vectors of *a* and *b*.

    Exposed for the execution phase (design spec D9): the INT8 quantization
    evidence compares fp32 vs int8 BEATs embeddings for the SAME real cached
    windows, in the SAME order -- e.g. `cosine_drift(fp32_embeddings,
    int8_embeddings)` on `results/cache/<run>--audio-beats.npz`'s own
    `features` array, loaded once per checkpoint env (fp32 then int8). `1.0`
    means every row points in exactly the same direction (up to a positive
    scale) as its pair; `0.0` means orthogonal on average; `-1.0` means
    opposite.

    Args:
        a: `(N, D)` float array.
        b: `(N, D)` float array, the SAME shape as *a* -- row i is *a*'s and
            *b*'s embedding of the SAME window.

    Returns:
        The MEAN of each row pair's cosine similarity across all N rows.

    Raises:
        ValueError: *a*/*b* have different shapes.
    """
    if a.shape != b.shape:
        raise ValueError(f"cosine_drift: shape mismatch a={a.shape} b={b.shape}")

    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    norm_a = np.linalg.norm(a64, axis=1)
    norm_b = np.linalg.norm(b64, axis=1)
    denom = norm_a * norm_b
    dot = np.sum(a64 * b64, axis=1)
    # A zero-norm row (a degenerate all-zero embedding) has no defined
    # direction to compare -- treated as similarity 0.0 for that row rather
    # than dividing by zero (which would silently poison the overall mean
    # with a NaN).
    safe_denom = np.where(denom > 0, denom, 1.0)
    similarities = np.where(denom > 0, dot / safe_denom, 0.0)
    return float(similarities.mean())


def _import_beats_or_exit() -> None:
    """Mirrors `scripts/adapt_beats.py`'s/`scripts/warm_cache.py`'s own
    `_import_beats_or_exit` (duplicated, not imported -- one script must not
    depend on a sibling script's internals, `warm_cache.py`'s own documented
    rationale). Runs early in `main()`, right after argument parsing --
    quantization is inherently a torch/beats operation."""
    try:
        import rowii.signals.beats  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"quantize_beats needs torch/beats ({exc}); {_BEATS_INSTALL_HINT}"
        ) from exc


def _require_beats_checkpoint(cfg: Config) -> Path:
    """*cfg*'s base BEATs checkpoint, or `SystemExit` with the established
    hint (mirrors `scripts/adapt_beats.py`'s `_require_beats_checkpoint`) if
    `ROWII_BEATS_CHECKPOINT` is unset."""
    checkpoint = cfg.beats_checkpoint
    if checkpoint is None:
        raise SystemExit(
            "quantize_beats needs a base checkpoint: set ROWII_BEATS_CHECKPOINT; "
            f"{_BEATS_INSTALL_HINT}"
        )
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-training INT8 dynamic quantization of the frozen BEATs encoder "
            "(nn.Linear layers, qint8; weight-only, no calibration data needed) -- "
            "saves the quantized MODULE (not a state dict) for BeatsFeaturizer's "
            "int8_model_path branch (package-5 spec D6, Task 5)."
        )
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output checkpoint path (default: {DEFAULT_OUT}); a sidecar "
             "<stem>.json is written alongside it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    _import_beats_or_exit()

    cfg = load_config()
    checkpoint = _require_beats_checkpoint(cfg)

    import torch

    from rowii.signals.beats_model import select_quantized_engine

    device = torch.device("cpu")
    model = load_beats_model(checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    size_fp32_bytes = checkpoint.stat().st_size
    logger.info(
        "quantize_beats: fp32 %s -- %d parameters, %.2f MB on disk",
        checkpoint, n_params, size_fp32_bytes / 1e6,
    )

    torch.backends.quantized.engine = select_quantized_engine()
    # inplace=True is LOAD-BEARING -- see this module's own docstring for why
    # the default (inplace=False, deepcopy-based) breaks on the vendored
    # weight-normed TransformerEncoder. quantize_dynamic itself is unannotated
    # (deprecated torch.ao API, no stubs) -- mirrors rowii.signals.beats_model.
    # load_beats_model's own BEATsConfig(...) # type: ignore[no-untyped-call].
    quantized = torch.ao.quantization.quantize_dynamic(  # type: ignore[no-untyped-call]
        model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized, args.out)
    size_int8_bytes = args.out.stat().st_size
    shrink_ratio = size_fp32_bytes / max(size_int8_bytes, 1)
    logger.info(
        "quantize_beats: int8 %s -- %.2f MB on disk (%.2fx smaller than fp32)",
        args.out, size_int8_bytes / 1e6, shrink_ratio,
    )

    sidecar = {
        "source_checkpoint": str(checkpoint),
        "size_fp32_bytes": size_fp32_bytes,
        "size_int8_bytes": size_int8_bytes,
        "created_at": datetime.now(UTC).isoformat(),
    }
    sidecar_path = args.out.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(
        f"quantize_beats: saved {args.out} ({size_int8_bytes / 1e6:.2f} MB, "
        f"{shrink_ratio:.2f}x smaller than fp32 {checkpoint} "
        f"({size_fp32_bytes / 1e6:.2f} MB, {n_params} parameters)); "
        f"sidecar {sidecar_path} -- {_QUANTIZED_NOTE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
