"""BEATs adaptation CLI (Step-2 package-5 spec D4, Task 3; objective per spec
D1 as amended by Amendment A1): trains a LoRA-adapted or fully fine-tuned copy
of the vendored BEATs encoder on ONE run's leakage-safe target-normal windows
(`rowii.adapt.target_windows.iter_target_windows`, Task 2) against the native
token-level masked-reconstruction proxy objective
(`rowii.adapt.objective.masked_token_loss`), exporting a MERGED,
format-compatible checkpoint that reloads through the EXISTING
`rowii.signals.beats_model.load_beats_model` path -- D2's integration trick:
pointing `ROWII_BEATS_CHECKPOINT` at the saved file turns every existing
`audio-beats`/`fusion-beats` variant into the adapted evaluation, no new
featurizer code needed downstream.

Usage: `adapt_beats.py --mode lora|full (--run <name> | --runs <a,b,c>)
[--epochs] [--batch-size 16] [--lr] --seed 7 --max-windows 8000 --out
models/adapted/` -> `models/adapted/beats_lora_<run>.pt` / `beats_ft_<run>.pt`
(multi-run: run names joined with `+`, e.g. `beats_lora_<a>+<b>.pt`) + a
sidecar `<same-stem>.json`.

Multi-run pooling (Step-2 package-7 Task 8, spec D6 as amended by A3.11:
`docs/superpowers/specs/2026-07-18-step2-package7-robustness-design.md`):
`--runs a,b,c` (mutually exclusive with `--run`) draws the training windows
ROUND-ROBIN across every named run's own leakage-safe calibration side via
`rowii.adapt.target_windows.iter_target_windows_multi`, with `--max-windows`
acting as the TOTAL budget (A3.11's binding rationale: a sequential chain
would train almost entirely on run 1); the sidecar records the runs list and
per-run window counts. The single-run `--run` path is untouched.

The native token path (READ THIS FIRST -- the train/inference-consistency
guarantee, Amendment A1 in `docs/superpowers/specs/2026-07-16-step2-package5-
adaptation-design.md`): the training forward pass reuses the model's OWN
pre-encoder pipeline, stage for stage -- `model.preprocess` (fbank; hardcoded
128 mels, fixed normalization constants) -> `model.patch_embedding` (strided
Conv2d patchify) -> reshape/transpose to `(B, T, embed_dim)` ->
`model.layer_norm` -> `model.post_extract_proj` (when present -- it is `None`
whenever `embed_dim == encoder_embed_dim`, mirroring `BEATs.__init__`'s own
conditional) -- exactly the stages `BEATs.extract_features` runs before its
encoder (`_native_tokens` below). The resulting NATIVE patch tokens are what
`masked_token_loss` masks and reconstructs, and what the (adapted)
`model.encoder` consumes -- so the adapters train on the SAME token basis the
deployed inference path (`extract_features`, driven downstream by
`BeatsFeaturizer._RealBeatsEncoder.extract`) will feed them. This is Amendment
A1's whole point: the original frame-level objective (`masked_patch_loss`)
cannot consume BEATs' patchified tokens (its frame-count-preserving contract
is structurally unsatisfiable across `patch_embedding` -- a strided Conv2d
that downsamples BOTH fbank axes and flattens the patch grid into ~48 tokens
where 98 frames went in; see that function's docstring), and bridging the gap
with a non-native frame-preserving projection (this script's first, retired
design) trained the adapters on an input distribution DECOUPLED from the
deployed path -- reviewer-proven to perturb native embeddings in an
objective-irrelevant direction. `tests/test_adapt_beats.py::
test_native_tokens_match_extract_features_encoder_input` pins the shared token
basis forever: it captures (via a forward pre-hook) the exact tensor
`extract_features` hands `model.encoder` and asserts `_native_tokens` produces
the identical tensor from the same waveforms.

One deliberate omission from `_native_tokens`: `model.dropout_input`, which
sits between `post_extract_proj` and the encoder in `extract_features`. Its
probability is `BEATsConfig.dropout_input = 0.0` (the class default), making
it an identity even in train mode (`Dropout(p=0)` never drops) -- and at eval
time (where the consistency test runs, and where every downstream inference
call happens under `.eval()`/`torch.no_grad()`) ANY dropout is an identity
regardless. Omitting it keeps the token construction free of a module that
could, under a hypothetical nonzero-p checkpoint, inject global-RNG noise
between the seeded mask decision and the encoder.

Freeze semantics per mode (task contract, unchanged by the rework): in lora
mode the ENTIRE model is frozen first and only injected adapters (+ the
reconstruction head) train, so the token-construction stages are frozen
plumbing; in full mode every model parameter is trainable, and -- unlike the
retired bridge design, which bypassed them entirely -- `patch_embedding`/
`layer_norm`/`post_extract_proj` now genuinely receive gradient through the
UNMASKED token rows the encoder attends to (the masked-target side is
detached inside `masked_token_loss`; see its docstring's stop-gradient
rationale).

Torch import discipline (plan's Global Constraints: eager only in
`rowii.adapt.objective`/`rowii.adapt.lora`/`rowii.adapt.student`'s model part/
`rowii.fusionx.model`; lazy everywhere else): every torch-touching name here is
imported lazily inside the function that needs it, INCLUDING `load_beats_model`/
`iter_target_windows` (and its multi-run sibling `iter_target_windows_multi`)/
`discover`, which is a deliberate exception to "lazy
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

from rowii.adapt.target_windows import (  # noqa: E402
    iter_target_windows,
    iter_target_windows_multi,
)
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
    "adaptation objective is native token-level masked reconstruction (latent-target "
    "MAE on the model's own pre-encoder patch tokens; spec D1 as amended by Amendment "
    "A1), a DOCUMENTED PROXY for BEATs' own unreproducible discrete-acoustic-token "
    "pretraining objective -- not a reproduction of it. Any Step-1/Step-2 result "
    "computed from this checkpoint must restate that caveat wherever it appears."
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


def _parse_runs_or_error(parser: argparse.ArgumentParser, raw: str) -> list[str]:
    """`--runs "a,b,c"` -> `["a", "b", "c"]`, with parse-level validation via
    *parser*`.error` (exit 2, argparse's own convention; P7 spec D6/A3.11):
    whitespace around names is stripped; an empty list (nothing but commas/
    blanks) and duplicate names are both rejected -- a duplicated run would
    occupy TWO slots of the round-robin rotation and silently double-weight
    that run inside the shared `--max-windows` TOTAL budget.
    """
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        parser.error("--runs must name at least one run (comma-separated run names)")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        parser.error(
            f"--runs contains duplicate run name(s): {', '.join(duplicates)} -- each run "
            "may appear once (a duplicate would occupy two round-robin slots and draw "
            "its windows twice under the shared --max-windows total budget)"
        )
    return names


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


def _native_tokens(model: BEATs, fbank: torch.Tensor) -> torch.Tensor:
    """*fbank* `(B, frames, n_mels)` -> the model's NATIVE pre-encoder patch
    tokens `(B, T, encoder_embed_dim)` -- Amendment A1's token construction,
    replicating `BEATs.extract_features`'s own stages between `preprocess`
    and `self.encoder`, in order: `unsqueeze(1)` (the Conv2d's channel dim)
    -> `patch_embedding` -> `reshape(B, embed_dim, -1)` (flattening the
    time-patch x mel-patch grid into one token axis) -> `transpose(1, 2)` ->
    `layer_norm` -> `post_extract_proj` IF the model has one (`BEATs.
    __init__` only constructs it when `embed_dim != encoder_embed_dim`; the
    conditional here mirrors `extract_features`'s own `if self.
    post_extract_proj is not None:` exactly). `model.dropout_input` is
    deliberately NOT applied -- see the module docstring's dedicated
    paragraph.

    The train/inference-consistency guarantee rests on this function staying
    in lockstep with `extract_features` -- pinned by `tests/test_adapt_beats.
    py::test_native_tokens_match_extract_features_encoder_input`, which
    asserts this function's output is IDENTICAL to the tensor
    `extract_features` actually hands `model.encoder` (captured via a
    forward pre-hook) for the same waveforms.

    NOT wrapped in `torch.no_grad()`: in `--mode full` these stages are
    trainable and receive gradient through the unmasked token rows (module
    docstring's freeze-semantics paragraph); in lora mode every stage is
    frozen, so autograd records no parameter graph anyway.
    """
    x = model.patch_embedding(fbank.unsqueeze(1))
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.transpose(1, 2)
    x = model.layer_norm(x)
    if model.post_extract_proj is not None:
        x = model.post_extract_proj(x)
    return cast("torch.Tensor", x)


def _encoder_forward(model: BEATs) -> Callable[[torch.Tensor], torch.Tensor]:
    """Builds `masked_token_loss`'s `encoder_forward` closure over the
    (adapted) `model.encoder` -- the vendored `TransformerEncoder`, the ONLY
    submodule LoRA touches (D2: q/v projections under
    `encoder.layers[*].self_attn`). Its `forward(x, padding_mask=None)`
    returns `(x, layer_results)` (verified in
    `rowii/vendor/beats/backbone.py`); only `x` (`(B, T, encoder_embed_dim)`,
    the same token count in and out -- a transformer encoder consuming its
    own native tokens) is what `masked_token_loss` needs.
    """

    def _forward(masked_tokens: torch.Tensor) -> torch.Tensor:
        encoded, _ = model.encoder(masked_tokens, padding_mask=None)
        return cast("torch.Tensor", encoded)

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
    by the caller via `iter_target_windows`) against `masked_token_loss`
    (module docstring's native token-level proxy objective, Amendment A1),
    for *epochs* full passes in *batch_size*-row mini-batches, shuffled each
    epoch by a *seed*-seeded CPU `torch.Generator` -- mirrors
    `scripts/pretrain_tfc.py`'s own `_train` (`_train_autoencoder`'s
    established shuffle-generator pattern), including reusing the SAME
    generator for `masked_token_loss`'s per-step masking (a DIFFERENT random
    subset of tokens masked every step -- the intended behaviour of a
    masked-reconstruction objective -- while the whole run stays exactly
    reproducible from *seed* alone).

    Per training step: fbank via `model.preprocess` (under `no_grad` -- pure
    parameterless signal processing over constant input data, nothing
    upstream to differentiate) -> `_native_tokens` (under grad: trainable in
    full mode, parameter-graph-free in lora mode where every stage is
    frozen) -> `masked_token_loss(tokens, encoder_forward, head)` -> Adam
    step over `_trainable_params(mode, ...)`.

    Self-contained: `torch.manual_seed(seed)` is this function's OWN first
    line (in addition to `main()`'s own call before `load_beats_model`, needed
    there so a monkeypatched loader's own random weight-init is ALSO seeded,
    tested contract (d)) -- re-seeding here makes every draw this function
    itself performs (mode-preparation's LoRA-adapter init, `head`, shuffling,
    masking) reproducible given *seed* and call order, independent of how
    many draws happened before this function was called.

    Returns:
        `(head, epoch_losses)`: *head* (`nn.Linear(encoder_embed_dim,
        encoder_embed_dim)`, the token-reconstruction head trained jointly
        with the encoder -- never persisted, since the checkpoint format is
        BEATs' own {"cfg", "model"}; returned so callers/tests can inspect it
        directly); *epoch_losses*: one mean loss per epoch, in order. *model*
        is left in `.eval()` mode (mirrors `_train_autoencoder`'s/
        `pretrain_tfc._train`'s own postcondition).
    """
    import torch

    from rowii.adapt.objective import masked_token_loss

    torch.manual_seed(seed)
    _prepare_model_for_mode(mode, model)
    model.to(device)
    model.train()

    x_all = torch.from_numpy(windows.astype(np.float32)).to(device)
    n = x_all.shape[0]

    encoder_dim = int(model.cfg.encoder_embed_dim)
    head = torch.nn.Linear(encoder_dim, encoder_dim).to(device)
    encoder_forward = _encoder_forward(model)

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
            tokens = _native_tokens(model, fbank)
            loss = masked_token_loss(tokens, encoder_forward, head, generator=gen)
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
    -> encoder -> ..., the SAME method `BeatsFeaturizer`'s `_RealBeatsEncoder.
    extract` drives, and since Amendment A1 also the same token basis
    `_native_tokens` trained on) -- is compared against the SAME forward pass
    through the checkpoint immediately RELOADED from disk via `load_beats_model` (the
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
            "leakage-safe target-normal windows against the native token-level "
            "masked-reconstruction proxy objective (spec D1 as amended by Amendment A1), "
            "exporting a MERGED checkpoint that reloads through the existing "
            "load_beats_model path (package-5 spec D4, Task 3)."
        )
    )
    parser.add_argument(
        "--mode", required=True, choices=("lora", "full"),
        help="lora: inject+train rank/alpha-defaulted adapters into encoder.layers[*]."
             "self_attn.{q,v}_proj, base frozen. full: every model parameter trainable "
             "(upper-capacity reference).",
    )
    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument(
        "--run", metavar="NAME",
        help="Run name to draw target-normal windows from (e.g. 010726-tu_ph_tu).",
    )
    run_group.add_argument(
        "--runs", metavar="NAMES",
        help="Comma-separated run names for MULTI-run adaptation (P7 spec D6/A3.11): "
             "target-normal windows are drawn ROUND-ROBIN across every named run's own "
             "leakage-safe calibration side, --max-windows acting as the TOTAL budget "
             "(a sequential chain would train almost entirely on the first run); the "
             "checkpoint name joins the run names with '+' and the sidecar records "
             "per-run window counts. Mutually exclusive with --run.",
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
        help="Seeds weight init (incl. the reconstruction head), window-shuffling, and "
             "masked-token selection (default: 7).",
    )
    parser.add_argument(
        "--max-windows", type=int, default=8000,
        help="Cap on target-normal windows drawn from --run; with --runs, the TOTAL "
             "round-robin budget across all named runs (default: 8000).",
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

    # Parse-level --runs validation (empty/duplicates) runs BEFORE the beats
    # import guard: a malformed flag value is a usage error regardless of the
    # environment's torch install.
    run_names: list[str] = []
    if args.runs is not None:
        run_names = _parse_runs_or_error(parser, args.runs)

    _import_beats_or_exit()

    cfg = load_config()
    base_checkpoint = _require_beats_checkpoint(cfg)

    index = discover(cfg.data_root)
    by_name = {r.name: r for r in index.runs}

    epochs, lr = _resolve_mode_defaults(args.mode, args.epochs, args.lr)

    t0 = time.monotonic()
    windows_per_run: dict[str, int] | None = None
    if args.runs is not None:
        # Multi-run pool (P7 spec D6/A3.11 -- module docstring): round-robin
        # draw across every named run's own calibration side, --max-windows
        # as the TOTAL budget, per-run counts recorded for the sidecar.
        unknown = [name for name in run_names if name not in by_name]
        if unknown:
            available = ", ".join(sorted(by_name)) or "(none discovered)"
            print(
                f"adapt_beats: unknown --runs run(s): {', '.join(unknown)}; "
                f"available runs: {available}",
                file=sys.stderr,
            )
            return 2
        pool_runs = [by_name[name] for name in run_names]
        pairs = list(
            iter_target_windows_multi(
                pool_runs, cfg,
                target_hz=BEATS_SAMPLE_RATE_HZ, seed=args.seed,
                max_windows=args.max_windows, return_run_names=True,
            )
        )
        windows_list = [window for _name, window in pairs]
        # Zero-contribution runs stay visible in the counts (A4.1's
        # coverage-visibility principle: never silently absent).
        windows_per_run = dict.fromkeys(run_names, 0)
        for name, _window in pairs:
            windows_per_run[name] += 1
        label = "+".join(run_names)
        source_desc = f"runs {', '.join(run_names)}"
    else:
        run = by_name.get(args.run)
        if run is None:
            available = ", ".join(sorted(by_name)) or "(none discovered)"
            print(
                f"adapt_beats: unknown --run {args.run!r}; available runs: {available}",
                file=sys.stderr,
            )
            return 2
        windows_list = list(
            iter_target_windows(
                run, cfg,
                target_hz=BEATS_SAMPLE_RATE_HZ, seed=args.seed, max_windows=args.max_windows,
            )
        )
        label = run.name
        source_desc = f"run {run.name!r}"

    if not windows_list:
        print(
            f"adapt_beats: no target-normal windows found for {source_desc} -- "
            "nothing to adapt on",
            file=sys.stderr,
        )
        return 1
    windows = np.stack(windows_list)
    if windows_per_run is None:
        logger.info(
            "adapt_beats: %d target-normal window(s) for run %s (--max-windows %s)",
            windows.shape[0], label, args.max_windows,
        )
    else:
        logger.info(
            "adapt_beats: %d target-normal window(s) pooled round-robin across %d run(s) "
            "(--max-windows %s TOTAL; per-run counts: %s)",
            windows.shape[0], len(run_names), args.max_windows, windows_per_run,
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
    checkpoint_path = args.out / f"{_CHECKPOINT_PREFIX[args.mode]}_{label}.pt"
    max_deviation = _save_and_verify(checkpoint_path, model, probe, device)
    elapsed_s = time.monotonic() - t0

    # Provenance block: the single-run sidecar keeps its exact historical
    # shape ("run"); a multi-run sidecar instead carries the runs list plus
    # the A3.11-required per-run window counts.
    if windows_per_run is None:
        provenance: dict[str, object] = {"run": label}
    else:
        provenance = {"runs": run_names, "windows_per_run": windows_per_run}
    sidecar: dict[str, object] = {
        "mode": args.mode,
        **provenance,
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
