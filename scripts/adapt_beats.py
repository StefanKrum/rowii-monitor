"""BEATs adaptation CLI (Step-2 package-5 spec D4, Task 3): trains a LoRA-adapted
or fully fine-tuned copy of the vendored BEATs encoder on ONE run's leakage-safe
target-normal windows (`rowii.adapt.target_windows.iter_target_windows`, Task 2)
against the masked-frame reconstruction proxy objective
(`rowii.adapt.objective.masked_patch_loss`, Task 1), exporting a MERGED,
format-compatible checkpoint that reloads through the EXISTING
`rowii.signals.beats_model.load_beats_model` path -- D2's integration trick:
pointing `ROWII_BEATS_CHECKPOINT` at the saved file turns every existing
`audio-beats`/`fusion-beats` variant into the adapted evaluation, no new
featurizer code needed downstream.

Usage: `adapt_beats.py --mode lora|full --run <name> [--epochs] [--batch-size 16]
[--lr] --seed 7 --max-windows 8000 --out models/adapted/` -> `models/adapted/
beats_lora_<run>.pt` / `beats_ft_<run>.pt` + a sidecar `<same-stem>.json`.

The frame-preserving bridge (READ THIS FIRST -- the one non-obvious design
choice in this file): `masked_patch_loss`'s contract (`rowii.adapt.objective`'s
own docstring) requires `encoder_forward` to map a `(B, frames, mels)` fbank to
`(B, frames, D)` -- the SAME `frames` count in and out, since the loss indexes
`predicted[mask]`/`fbank[mask]` with a `(B, frames)` boolean mask built from the
INPUT fbank's own frame axis. The vendored BEATs' own `patch_embedding` (a
strided `Conv2d(kernel=input_patch_size, stride=input_patch_size)` over the
`(B, 1, frames, mels)` fbank "image", `rowii.vendor.beats.BEATs.BEATs.
extract_features`) does NOT preserve that axis: it downsamples BOTH the time
and mel dimensions and then FLATTENS the resulting time-patch and mel-patch
axes together into one token axis (`features.reshape(B, embed_dim, -1)`) -- for
the real `BEATs_iter3_plus_AS2M.pt` checkpoint (`input_patch_size=16`,
`n_mels=128`, ~98 fbank frames per 1-s window), that is `T = (98//16) * (128//16)
= 6 * 8 = 48` tokens, not 98. Feeding `masked_patch_loss` a closure built on
`patch_embedding` therefore does not just produce a worse proxy objective, it
raises at runtime (`predicted[mask]`'s mask shape would not match `predicted`'s
shape). `_build_frame_bridge` below stands in for `patch_embedding` +
`post_extract_proj` + `layer_norm` with a single frame-preserving, FROZEN,
randomly-initialized `nn.Linear(n_mels, encoder_embed_dim)` (`rowii.adapt.
objective`'s own docstring anticipates exactly this: "no dependency on how any
given encoder_forward closure internally patches/PROJECTS/pools its input") --
one token per input time frame, fed straight into `model.encoder` (the
vendored `TransformerEncoder`; self-attention and its `pos_conv`'s SAME-padded
depthwise conv both preserve sequence length by construction, verified by
reading `rowii/vendor/beats/backbone.py`). `model.encoder` is exactly "the
(adapted) encoder": it is the ONLY submodule LoRA touches (D2: q/v projections
under `encoder.layers[*].self_attn`), and in `--mode full` it is the ONLY
submodule this proxy objective's forward pass actually exercises -- an
EXTENSION of `masked_patch_loss`'s own already-documented caveat ("a
position-wise encoder's weight grads are exactly zero at masked positions; a
real attention encoder's are not"): `--mode full` marks every one of `model`'s
own parameters trainable (the literal contract), but `patch_embedding` /
`post_extract_proj` / the top-level `layer_norm` never receive gradient here
regardless of mode, because the frame-preserving bridge bypasses them by
design. The bridge itself is FROZEN (`requires_grad_(False)`) specifically so
it is never a THIRD component in either mode's optimizer (the task's own
contract: lora -> adapter params + head ONLY; full -> all(model) + head ONLY).

Torch import discipline (plan's Global Constraints: eager only in
`rowii.adapt.objective`/`rowii.adapt.lora`/`rowii.adapt.student`'s model part/
`rowii.fusionx.model`; lazy everywhere else): every torch-touching name here is
imported lazily inside the function that needs it, INCLUDING `load_beats_model`/
`iter_target_windows`/`discover`, which is a deliberate exception to "lazy
everywhere" -- `scripts/warm_cache.py` sets the precedent (`discover`/
`prepare_run` imported at module top specifically so `tests/test_warm_cache.py`
can `monkeypatch.setattr(warm_cache, "discover", ...)`; a name only ever bound
inside a function body cannot be monkeypatched from outside it, since the
function's own `from ... import ...` statement would just re-resolve the REAL
symbol on every call). This task's own contract requires monkeypatching
`load_beats_model` (-> a tiny real BEATs instance, never a real checkpoint) and
`iter_target_windows` (-> synthetic windows, never real data), so both are
module-level imports here for the same reason.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import torch

    from rowii.vendor.beats.BEATs import BEATs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rowii.adapt.target_windows import iter_target_windows  # noqa: E402
from rowii.config import Config, load_config  # noqa: E402
from rowii.io.dataset import discover  # noqa: E402
from rowii.pipeline import _BEATS_INSTALL_HINT  # noqa: E402
from rowii.signals.beats_model import BEATS_SAMPLE_RATE_HZ, load_beats_model  # noqa: E402

logger = logging.getLogger(__name__)

_MODE_DEFAULTS: dict[str, tuple[int, float]] = {
    # (epochs, lr) -- spec D4's own mode-specific defaults, applied only when
    # the matching CLI flag is omitted (`_resolve_mode_defaults`).
    "lora": (5, 1e-4),
    "full": (2, 1e-5),
}

_CHECKPOINT_PREFIX: dict[str, str] = {"lora": "beats_lora", "full": "beats_ft"}

_PROXY_NOTE = (
    "adaptation objective is masked-frame reconstruction, a DOCUMENTED PROXY "
    "(spec D1) for BEATs' own unreproducible discrete-acoustic-token pretraining "
    "objective -- not a reproduction of it. Any Step-1/Step-2 result computed "
    "from this checkpoint must restate that caveat wherever it appears."
)


def _import_beats_or_exit() -> None:
    """Mirrors `scripts/warm_cache.py`'s/`scripts/run_step2.py`'s own
    `_import_beats_or_exit` (duplicated, not imported -- `warm_cache.py`'s own
    documented rationale: one script must not depend on a sibling script's
    internals). Unlike `warm_cache.py`, this guard is unconditional (adaptation
    is inherently a torch/beats operation, like `pretrain_tfc.py`'s own
    `_import_torch_or_exit`), so it runs early in `main()`, right after
    argument parsing.
    """
    try:
        import rowii.signals.beats  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"BEATs adaptation needs torch/beats ({exc}); {_BEATS_INSTALL_HINT}"
        ) from exc


def _require_beats_checkpoint(cfg: Config) -> Path:
    """*cfg*'s base BEATs checkpoint, or `SystemExit` with the established
    hint (mirrors `rowii.pipeline._featurizer_for_stream`'s own
    beats-checkpoint-None message shape) if `ROWII_BEATS_CHECKPOINT` is unset.
    """
    checkpoint = cfg.beats_checkpoint
    if checkpoint is None:
        raise SystemExit(
            "adapt_beats needs a base checkpoint: set ROWII_BEATS_CHECKPOINT; "
            f"{_BEATS_INSTALL_HINT}"
        )
    return checkpoint


def _resolve_mode_defaults(mode: str, epochs: int | None, lr: float | None) -> tuple[int, float]:
    """`--epochs`/`--lr` resolution (module docstring / spec D4): an explicit
    CLI flag always wins; an omitted one falls back to *mode*'s own default
    (`_MODE_DEFAULTS`). Pure and parse-level -- no torch, no I/O -- so tests
    can assert the mode-default contract directly, without running `main()`.
    """
    default_epochs, default_lr = _MODE_DEFAULTS[mode]
    resolved_epochs = epochs if epochs is not None else default_epochs
    resolved_lr = lr if lr is not None else default_lr
    return resolved_epochs, resolved_lr


def _prepare_model_for_mode(mode: str, model: BEATs) -> int:
    """Freezes/unfreezes *model* in place for *mode* (task contract):
    `"lora"` -> freeze EVERY parameter of *model* first, THEN inject rank/
    alpha-defaulted adapters into `encoder.layers[*].self_attn.{q,v}_proj`
    (`rowii.adapt.lora.inject_lora`, called on `model.encoder` specifically --
    "the module that contains encoder.layers", per the vendored class:
    `BEATs.__init__` sets `self.encoder = TransformerEncoder(cfg)` and
    `TransformerEncoder.__init__` sets `self.layers = nn.ModuleList(...)`);
    `"full"` -> mark every parameter of *model* trainable.

    ORDER MATTERS for `"lora"`: `model.requires_grad_(False)` MUST run BEFORE
    `inject_lora`, not after. `inject_lora` replaces `q_proj`/`v_proj` with
    fresh `LoraLinear`s whose `lora_a`/`lora_b` are brand-new `nn.Linear`s,
    default-constructed at `requires_grad=True` and therefore untouched by an
    EARLIER blanket freeze; a freeze applied AFTER injection would instead
    also freeze those adapters, breaking LoRA training entirely. `load_beats_
    model`'s own docstring is explicit that loading a checkpoint never itself
    freezes anything, so without this blanket freeze every non-adapter
    parameter (`pos_conv`, per-layer norms, `k_proj`/`out_proj`, FFN,
    `patch_embedding`, ...) would default to trainable too -- this call is
    what makes "only adapter params train" (spec D2) a true statement about
    the model's own `requires_grad` flags, not merely an artifact of which
    parameters happen to end up in the optimizer (`_trainable_params`).

    Returns:
        Count of LoRA adapters injected (0 for `mode="full"`).
    """
    from rowii.adapt.lora import inject_lora

    if mode == "lora":
        model.requires_grad_(False)
        n_injected: int = inject_lora(model.encoder)
        logger.info(
            "adapt_beats: injected %d LoRA adapter(s) (rowii.adapt.lora's own rank/alpha "
            "defaults) into encoder.layers[*].self_attn.{q,v}_proj; every other parameter "
            "frozen",
            n_injected,
        )
        return n_injected
    model.requires_grad_(True)
    logger.info("adapt_beats: mode=full -- every model parameter marked trainable")
    return 0


def _build_frame_bridge(n_mels: int, encoder_dim: int, device: torch.device) -> torch.nn.Linear:
    """Frozen, randomly-initialized `nn.Linear(n_mels, encoder_dim)` --
    module docstring's "frame-preserving bridge", standing in for BEATs' own
    `patch_embedding` (which does not preserve the frame axis this proxy
    objective needs). Frozen so it is never part of either mode's optimizer
    (`_trainable_params`): it is not "the encoder" being adapted, just fixed
    plumbing that makes the proxy objective's shapes well-formed against a
    REAL multi-head self-attention stack. Ordinary default (kaiming-uniform)
    `nn.Linear` init -- drawn from the caller's already-seeded global torch
    RNG (`_train`'s `torch.manual_seed(seed)`), so construction is
    reproducible given a fixed seed and call order, same as everything else
    in `_train`.
    """
    import torch

    bridge = torch.nn.Linear(n_mels, encoder_dim)
    bridge.requires_grad_(False)
    return bridge.to(device)


def _encoder_forward(
    model: BEATs, bridge: torch.nn.Linear
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Builds `masked_patch_loss`'s `encoder_forward` closure: *bridge* (the
    frame-preserving projection) feeds `model.encoder` (the vendored
    `TransformerEncoder` -- "the (adapted) encoder", module docstring)
    directly, bypassing `patch_embedding`/`post_extract_proj`/the top-level
    `layer_norm` entirely. `model.encoder`'s own `forward` returns `(x,
    layer_results)`; only `x` (`(B, frames, encoder_embed_dim)`, the SAME
    `frames` the bridge was given -- self-attention and `pos_conv`'s
    SAME-padded depthwise conv both preserve sequence length, verified by
    reading `rowii/vendor/beats/backbone.py`) is what `masked_patch_loss`
    needs.
    """
    import torch

    def _forward(masked_fbank: torch.Tensor) -> torch.Tensor:
        x = bridge(masked_fbank)
        encoded, _ = model.encoder(x, padding_mask=None)
        return cast(torch.Tensor, encoded)

    return _forward


def _trainable_params(
    mode: str, model: BEATs, head: torch.nn.Linear
) -> list[torch.nn.Parameter]:
    """The EXACT optimizer parameter set for *mode* (task contract, verbatim):
    `"lora"` -> ONLY `rowii.adapt.lora.lora_parameters(model)` + `head.
    parameters()`; `"full"` -> ALL of `model.parameters()` + `head.
    parameters()`. Factored out of `_train` so a test can inspect precisely
    which tensors an optimizer would receive (`tests/test_adapt_beats.py`'s
    "count spy" contract) without constructing an `Adam`/running a step.
    *model* must already have been prepared for *mode*
    (`_prepare_model_for_mode`) -- for `"lora"`, `lora_parameters` only
    yields anything once adapters have actually been injected.
    """
    from rowii.adapt.lora import lora_parameters

    if mode == "lora":
        return list(lora_parameters(model)) + list(head.parameters())
    return list(model.parameters()) + list(head.parameters())


def _train(
    model: BEATs,
    windows: np.ndarray,
    *,
    mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Linear, list[float]]:
    """Trains *model* (already loaded, NOT yet mode-prepared) in place on
    *windows* (`(N, 16000)` float32 16 kHz waveforms, already leakage-filtered
    by the caller via `iter_target_windows`) against `masked_patch_loss`
    (module docstring's proxy objective), for *epochs* full passes in
    *batch_size*-row mini-batches, shuffled each epoch by a *seed*-seeded CPU
    `torch.Generator` -- mirrors `scripts/pretrain_tfc.py`'s own `_train`
    (`_train_autoencoder`'s established shuffle-generator pattern), including
    reusing the SAME generator for `masked_patch_loss`'s per-step masking (a
    DIFFERENT random subset of frames masked every step -- the intended
    behaviour of a masked-reconstruction objective -- while the whole run
    stays exactly reproducible from *seed* alone).

    Self-contained: `torch.manual_seed(seed)` is this function's OWN first
    line (in addition to `main()`'s own call before `load_beats_model`, needed
    there so a monkeypatched loader's own random weight-init is ALSO seeded,
    tested contract (d)) -- re-seeding here makes every draw this function
    itself performs (mode-preparation's LoRA-adapter init, the frame bridge,
    `head`, shuffling, masking) reproducible given *seed* and call order,
    independent of how many draws happened before this function was called.

    Returns:
        `(head, epoch_losses)`: *head* (`nn.Linear(encoder_embed_dim, n_mels)`,
        the reconstruction head trained jointly with the encoder -- needed by
        the caller only for the reload-verification forward pass's device/
        eval-mode bookkeeping is NOT required, since the checkpoint never
        stores *head*; returned so callers/tests can inspect it directly);
        *epoch_losses*: one mean loss per epoch, in order. *model* is left in
        `.eval()` mode (mirrors `_train_autoencoder`'s/`pretrain_tfc._train`'s
        own postcondition).
    """
    import torch

    from rowii.adapt.objective import masked_patch_loss

    torch.manual_seed(seed)
    _prepare_model_for_mode(mode, model)
    model.to(device)
    model.train()

    x_all = torch.from_numpy(windows.astype(np.float32)).to(device)
    n = x_all.shape[0]

    with torch.no_grad():
        probe_fbank = model.preprocess(x_all[: min(4, n)])
    n_mels = int(probe_fbank.shape[-1])
    encoder_dim = int(model.cfg.encoder_embed_dim)

    bridge = _build_frame_bridge(n_mels, encoder_dim, device)
    head = torch.nn.Linear(encoder_dim, n_mels).to(device)
    encoder_forward = _encoder_forward(model, bridge)

    opt = torch.optim.Adam(_trainable_params(mode, model, head), lr=lr)
    gen = torch.Generator().manual_seed(seed)

    epoch_losses: list[float] = []
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            with torch.no_grad():
                fbank = model.preprocess(x_all[idx])
            opt.zero_grad()
            loss = masked_patch_loss(encoder_forward, fbank, head, generator=gen)
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        mean_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(mean_loss)
        logger.info(
            "adapt_beats: epoch %d/%d -- mean loss %.6f", epoch + 1, epochs, mean_loss
        )

    model.eval()
    return head, epoch_losses


def _save_and_verify(
    checkpoint_path: Path, model: BEATs, probe: torch.Tensor, device: torch.device
) -> float:
    """Saves *model* to *checkpoint_path* as `{"cfg": model.cfg.__dict__,
    "model": model.state_dict()}` -- the SAME `{"cfg", "model"}` shape
    `load_beats_model` already reads (D2/D4: "loads it unchanged"); `model.
    cfg.__dict__` (rather than a second raw read of the ORIGINAL base
    checkpoint's own "cfg" dict) is used deliberately: `model.cfg` is the
    EXACT `BEATsConfig` instance that already determined *model*'s real
    architecture (`BEATs.__init__` stores `self.cfg = cfg` and never
    reassigns it), so re-serializing it is architecturally lossless by
    construction and needs no second file read.

    Immediately verifies the round trip empirically, not just structurally:
    a forward pass on *probe* through the IN-MEMORY *model* (already merged
    for `--mode lora`, already `.eval()`'d by `_train`) via `model.
    extract_features` -- BEATs' own PRODUCTION inference path (patch_embedding
    -> encoder -> ...), the SAME method `BeatsFeaturizer`'s `_RealBeatsEncoder.
    extract` drives, deliberately NOT this script's own frame-preserving
    training bridge -- is compared against the SAME forward pass through the
    checkpoint immediately RELOADED from disk via `load_beats_model` (the
    real, established loader; module docstring explains why this name is a
    module-level import rather than lazy, specifically so it stays
    monkeypatchable from outside for the INITIAL load while still exercising
    real `torch.save`/`torch.load`/`load_state_dict` fidelity here, since this
    second call's *checkpoint_path* argument is a file this function itself
    just wrote).

    Returns:
        Max absolute elementwise deviation between the two forward passes
        (also logged here, at INFO level, so callers/tests can assert the
        check ran without threading the return value through their own
        logging).
    """
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": model.cfg.__dict__, "model": model.state_dict()}, checkpoint_path)

    model.eval()
    with torch.no_grad():
        before, _ = model.extract_features(probe)

    reloaded = load_beats_model(checkpoint_path, device)
    with torch.no_grad():
        after, _ = reloaded.extract_features(probe)

    max_deviation = float((before - after).abs().max().item())
    logger.info(
        "adapt_beats: reload-forward deviation check -- %s reloaded via load_beats_model, "
        "max |before - after| = %.3e",
        checkpoint_path, max_deviation,
    )
    return max_deviation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt the vendored BEATs encoder (LoRA or full fine-tune) on ONE run's "
            "leakage-safe target-normal windows against the masked-frame reconstruction "
            "proxy objective, exporting a MERGED checkpoint that reloads through the "
            "existing load_beats_model path (package-5 spec D4, Task 3)."
        )
    )
    parser.add_argument(
        "--mode", required=True, choices=("lora", "full"),
        help="lora: inject+train rank/alpha-defaulted adapters into encoder.layers[*]."
             "self_attn.{q,v}_proj, base frozen. full: every model parameter trainable "
             "(upper-capacity reference).",
    )
    parser.add_argument(
        "--run", required=True, metavar="NAME",
        help="Run name to draw target-normal windows from (e.g. 010726-tu_ph_tu).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Training epochs; default depends on --mode (lora: 5, full: 2) unless given.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Mini-batch size (default: 16).",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Adam learning rate; default depends on --mode (lora: 1e-4, full: 1e-5) "
             "unless given.",
    )
    parser.add_argument(
        "--seed", type=int, default=7,
        help="Seeds weight init (incl. the frame bridge/head), window-shuffling, and "
             "masked-frame selection (default: 7).",
    )
    parser.add_argument(
        "--max-windows", type=int, default=8000,
        help="Cap on target-normal windows drawn from --run (default: 8000).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("models/adapted/"),
        help="Output directory (default: models/adapted/); filename is "
             "beats_lora_<run>.pt (--mode lora) or beats_ft_<run>.pt (--mode full).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    _import_beats_or_exit()

    cfg = load_config()
    base_checkpoint = _require_beats_checkpoint(cfg)

    index = discover(cfg.data_root)
    by_name = {r.name: r for r in index.runs}
    run = by_name.get(args.run)
    if run is None:
        available = ", ".join(sorted(by_name)) or "(none discovered)"
        print(
            f"adapt_beats: unknown --run {args.run!r}; available runs: {available}",
            file=sys.stderr,
        )
        return 2

    epochs, lr = _resolve_mode_defaults(args.mode, args.epochs, args.lr)

    t0 = time.monotonic()
    windows_list = list(
        iter_target_windows(
            run, cfg,
            target_hz=BEATS_SAMPLE_RATE_HZ, seed=args.seed, max_windows=args.max_windows,
        )
    )
    if not windows_list:
        print(
            f"adapt_beats: no target-normal windows found for run {run.name!r} -- "
            "nothing to adapt on",
            file=sys.stderr,
        )
        return 1
    windows = np.stack(windows_list)
    logger.info(
        "adapt_beats: %d target-normal window(s) for run %s (--max-windows %s)",
        windows.shape[0], run.name, args.max_windows,
    )

    from rowii.signals.beats import best_device

    device = best_device()

    import torch

    # Seeded here (BEFORE load_beats_model) so that a monkeypatched loader's
    # own random weight init (tests/test_adapt_beats.py's tiny-instance stub)
    # is ALSO deterministic given --seed; _train re-seeds again with the same
    # value for its own draws (see that function's docstring for why both
    # calls are needed and why the redundancy is harmless).
    torch.manual_seed(args.seed)
    model = load_beats_model(base_checkpoint, device)

    head, epoch_losses = _train(
        model, windows,
        mode=args.mode, epochs=epochs, batch_size=args.batch_size, lr=lr,
        seed=args.seed, device=device,
    )
    del head  # never persisted -- checkpoint format is BEATs' own {"cfg", "model"} only

    if args.mode == "lora":
        from rowii.adapt.lora import merge_lora

        n_merged = merge_lora(model)
        logger.info("adapt_beats: merged %d LoRA adapter(s) back into base Linears", n_merged)

    probe = torch.from_numpy(windows[: min(4, windows.shape[0])].astype(np.float32)).to(device)
    checkpoint_path = args.out / f"{_CHECKPOINT_PREFIX[args.mode]}_{run.name}.pt"
    max_deviation = _save_and_verify(checkpoint_path, model, probe, device)
    elapsed_s = time.monotonic() - t0

    sidecar = {
        "mode": args.mode,
        "run": run.name,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "lr": lr,
        "seed": args.seed,
        "n_windows": int(windows.shape[0]),
        "final_loss": epoch_losses[-1],
        "max_reload_deviation": max_deviation,
        "elapsed_s": elapsed_s,
        "note": _PROXY_NOTE,
    }
    sidecar_path = checkpoint_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(
        f"adapt_beats: saved {checkpoint_path} (--mode {args.mode}, {epochs} epoch(s) over "
        f"{windows.shape[0]} window(s), final mean loss {epoch_losses[-1]:.6f}, max reload "
        f"deviation {max_deviation:.3e}); sidecar {sidecar_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
