"""Reconstruction anomaly scorers (package-3 spec D2): MLP-AE on any feature
vector; LSTM-AE / Conv-AE on `logmel` patches, reshaped internally so the
window-internal time axis is the sequence (no cross-window contiguity; the
`Scorer` protocol holds unchanged). Score = per-window reconstruction MSE,
higher = more anomalous (explicit polarity by construction, spec D1 note --
an autoencoder trained on normal windows reconstructs a normal window well
(low MSE) and an anomalous one poorly (high MSE), so no sign flip is needed
here unlike this package's own `OcSvmScorer`/`IsolationForestScorer`/
`LofScorer`, which negate an underlying "higher = more normal" sklearn
quantity).

Torch is imported lazily, INSIDE `fit`/`score` only -- importing this module
never requires the `[beats]` extra (matches `rowii.signals.beats`'s own
lazy-import story, the only other torch consumer in this project). Calling
`fit`/`score` without torch installed raises `RuntimeError` with the same
install hint as `scripts/run_step2.py`'s `_import_beats_or_exit` guard.
`_require_torch()` itself returns `None` (not the module, despite the
tempting `torch = _require_torch()` shorthand) -- every method instead
follows it with its OWN plain `import torch` (cheap: guaranteed to hit
`sys.modules` once `_require_torch()` has already succeeded once). This
mirrors `rowii.signals.beats`/`rowii.signals.beats_model`'s own established
per-function local-import convention, and is not just a style choice: with
`mypy --strict`, a shared helper that IMPORTS torch and RETURNS the module
object can only be typed `Any` (torch is a real module, not a class -- there
is no way to spell "the torch module, fully typed" as a return annotation),
and `LstmAeScorer`/`ConvAeScorer` below define a `torch.nn.Module` subclass
per `fit()` call (the model's `forward` needs this module's own
`n_frames`/`n_mels` closed over, and building it lazily is the only way to
keep the module-level "no torch import" promise) -- mypy flatly rejects
subclassing a value of static type `Any` ("Class cannot subclass ... has
type Any"). A real local `import torch` statement, by contrast, gives mypy
torch's own installed type stubs for everything reached from that point in
the function, subclass included -- verified against this exact pattern
before writing the classes below.

`score()` called before `fit()` raises `AssertionError`, not the `ValueError`
`rowii.anomaly.scorers`' five scorers use for the same precondition --
deliberate (orchestrator resolution 6): every `score()` below needs an
`assert self._model is not None` immediately afterward purely to narrow
`self._model`'s type from `torch.nn.Module | None` to `torch.nn.Module` for
`mypy --strict` (no `TYPE_CHECKING`-only trick avoids this), so re-raising
that same assertion as a hand-rolled `ValueError` would just be duplicate
work for an identical guarantee; every class's `score()` docstring calls
this out explicitly.

Device: `_device()` delegates to `rowii.signals.beats.best_device()` --
identical `ROWII_FORCE_CPU` env > mps > cuda > cpu priority as the BEATs
featurizer, so a CPU fallback always works even with no GPU backend present.
Training determinism (spec D2): every class seeds BOTH `torch.manual_seed`
(weight init, called before its model is constructed) and
`_train_autoencoder`'s own shuffle generator from the same `seed` argument.
This is a deterministic GUARANTEE only on CPU -- verified by this module's
own test suite, which forces `ROWII_FORCE_CPU=1` for exactly this reason.
MPS (and, to a lesser extent, CUDA) kernels for some of the ops used here
(`LSTM`, strided `Conv2d`/`ConvTranspose2d`) are not guaranteed
bit-reproducible across runs even with every seed fixed -- a known
upstream PyTorch/Metal limitation, not something this module can paper
over -- so `fit()` on the non-CPU default device is reproducible only up to
that backend's own determinism guarantees, never verified here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from rowii.anomaly.scorers import _check_query, _check_reference

if TYPE_CHECKING:
    import torch

_TORCH_HINT = (
    "reconstruction scorers need torch: pip install -e '.[beats]'"
)


def _require_torch() -> None:
    """Raise `RuntimeError` (with the shared install hint) if torch is not
    importable; a no-op otherwise. See module docstring for why this does
    NOT return the imported module -- every caller follows this with its own
    `import torch` instead.
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(_TORCH_HINT) from e


def _device() -> torch.device:
    from rowii.signals.beats import best_device

    return best_device()


def _train_autoencoder(
    model: torch.nn.Module,
    reference_t: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> None:
    """Shared training loop for `MlpAeScorer`/`LstmAeScorer`/`ConvAeScorer.fit`
    (orchestrator resolution 2 -- the one piece of `fit()` that does NOT vary
    per class; only the encoder/decoder architecture differs). Adam + MSE
    loss, `epochs` full passes over *reference_t*, each pass shuffled into
    `batch_size`-row mini-batches by a *seed*-seeded `torch.Generator`.
    Leaves *model* in `.eval()` mode on return (every `score()` in this
    module reads its model immediately under `torch.no_grad()`).

    Args:
        model: An already-constructed, already-`.to(device)`'d autoencoder
            (any `torch.nn.Module` whose `forward` maps a `(N, F)` batch to
            an `(N, F)` reconstruction of the SAME shape).
        reference_t: `(N, F)` float32 tensor, already on *device* (the
            caller's own `torch.as_tensor(reference, ..., device=device)`).
        epochs: Full passes over *reference_t*.
        lr: Adam learning rate.
        batch_size: Rows per gradient step (the last batch of a pass may be
            smaller, `len(reference_t) % batch_size`).
        seed: Seeds the per-epoch shuffle generator (weight init is seeded
            separately, by the caller, via `torch.manual_seed` BEFORE the
            model is constructed -- shuffling and initialization are two
            independent sources of randomness).
        device: Accepted for signature symmetry with callers, which already
            place both *model* and *reference_t* on it before calling this --
            the loop body never references *device* directly, since every
            tensor operation here inherits its device from *reference_t*/
            *model*'s own parameters.

    Returns:
        None (*model* is mutated in place: trained, then switched to eval
        mode).
    """
    import torch

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    # A plain `torch.Generator()` is CPU-resident by construction (no `device=`
    # given) -- deliberate, not an oversight: indexing a *device* tensor with
    # CPU int64 indices is well-defined regardless of *device* (verified on
    # MPS before writing this), and keeps the shuffle order identical across
    # devices for the same *seed* (`test_deterministic_given_seed`'s
    # guarantee would otherwise depend on which backend happened to run it).
    gen = torch.Generator().manual_seed(seed)
    n = reference_t.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            batch = reference_t[perm[start : start + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
    model.eval()


class MlpAeScorer:
    """MLP autoencoder baseline reconstruction scorer (package-3 spec D2) --
    see module docstring for score polarity/definition and the lazy-torch
    story.

    Works on ANY `(N, F)` feature vector (embeddings, handcrafted stats, or a
    flattened logmel patch) -- unlike `LstmAeScorer`/`ConvAeScorer`, it
    applies no logmel-specific reshape, so the exact same feature matrix can
    be scored by all three reconstruction scorers side by side whenever `F`
    happens to be a flattened logmel patch.
    """

    name: str = "mlpae"

    def __init__(
        self,
        hidden: tuple[int, ...] = (128, 32),
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 256,
        seed: int = 7,
    ) -> None:
        """Args:
            hidden: Encoder layer widths, narrowing from the input width `F`
                down to `hidden[-1]` (the bottleneck) -- e.g. `(128, 32)`
                builds `F -> 128 -> 32` (ReLU after each Linear). The decoder
                mirrors this in reverse (`32 -> 128 -> F`), with NO
                activation after its final layer (raw reconstruction --
                logmel/embedding features are not bounded to `[0, 1]` or
                non-negative, so a final ReLU/Sigmoid would clip attainable
                reconstructions).
            epochs: Full passes over *reference* per `fit()` call.
            lr: Adam learning rate.
            batch_size: Rows per gradient step (`_train_autoencoder`).
            seed: Seeds both `torch.manual_seed` (weight init, in `fit`) and
                the per-epoch shuffle generator (`_train_autoencoder`) --
                `fit()` is deterministic given the same *reference* and
                `seed` (`test_deterministic_given_seed`).
        """
        self.hidden = tuple(hidden)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None

    def fit(self, reference: np.ndarray) -> MlpAeScorer:
        """Fit an MLP autoencoder on *reference* (Adam, MSE loss, shuffled
        mini-batches, `epochs` passes -- `_train_autoencoder`).

        Args:
            reference: `(N, F)` finite matrix of normal reference windows/rows.

        Returns:
            self.

        Raises:
            ValueError: if `reference` is empty or has a non-finite value
                (`_check_reference`).
            RuntimeError: if torch is not installed (`_require_torch`).
        """
        _check_reference(reference)
        _require_torch()
        import torch

        torch.manual_seed(self.seed)
        device = _device()
        f = reference.shape[1]
        dims = [f, *self.hidden]
        enc: list[torch.nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            enc += [torch.nn.Linear(a, b), torch.nn.ReLU()]
        dec: list[torch.nn.Module] = []
        rdims = list(reversed(dims))
        for i, (a, b) in enumerate(zip(rdims[:-1], rdims[1:], strict=True)):
            dec.append(torch.nn.Linear(a, b))
            if i < len(rdims) - 2:
                dec.append(torch.nn.ReLU())
        model: torch.nn.Module = torch.nn.Sequential(*enc, *dec).to(device)
        x = torch.as_tensor(reference, dtype=torch.float32, device=device)
        _train_autoencoder(model, x, self.epochs, self.lr, self.batch_size, self.seed, device)
        self._model, self._device = model, device
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """`(W, F)` windows -> `(W,)` float64 scores, higher = more anomalous
        (mean-squared reconstruction error per row).

        Raises:
            ValueError: if `x` contains non-finite values (`_check_query`).
            RuntimeError: if torch is not installed (`_require_torch`).
            AssertionError: if called before `fit()` (module docstring's
                "score() called before fit()" note).
        """
        _check_query(x)
        _require_torch()
        import torch

        assert self._model is not None, "MlpAeScorer.score() called before fit()"
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float32, device=self._device)
            recon = self._model(t)
            mse = ((recon - t) ** 2).mean(dim=1)
        return np.asarray(mse.cpu().numpy(), dtype=np.float64)


class LstmAeScorer:
    """LSTM autoencoder baseline reconstruction scorer for `logmel` patches
    (package-3 spec D2) -- see module docstring for score polarity/definition
    and the lazy-torch story.

    Reshapes each `(F,)` flattened logmel row back into its own `(n_frames,
    n_mels)` patch (`n_frames = F // n_mels`, frame-major -- `rowii.signals.
    logmel.LogmelFeaturizer`'s own flatten order, Task 2) and treats the
    WINDOW-INTERNAL frame axis as the sequence: an encoder LSTM consumes the
    `n_frames` mel-vectors in order; its final hidden state is repeated
    across `n_frames` steps and fed through a decoder LSTM, then a per-step
    `Linear(hidden, n_mels)` projects back to `(n_frames, n_mels)`, reshaped
    to `(F,)` for the MSE-against-input score. Every window is its OWN short
    sequence here -- consecutive WINDOWS are never chained into one longer
    sequence (the `Scorer` protocol's `fit(reference)`/`score(x)` contract
    has no notion of window-to-window adjacency to begin with).
    """

    name: str = "lstmae"

    def __init__(
        self,
        hidden: int = 64,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 128,
        seed: int = 7,
        n_mels: int = 64,
    ) -> None:
        """Args:
            hidden: Encoder/decoder LSTM hidden width -- also the bottleneck
                width (the encoder's final hidden state, repeated across
                frames, IS the decoder's per-step input).
            epochs: Full passes over *reference* per `fit()` call.
            lr: Adam learning rate.
            batch_size: Rows per gradient step (`_train_autoencoder`).
            seed: Seeds both `torch.manual_seed` (weight init, in `fit`) and
                the per-epoch shuffle generator (`_train_autoencoder`).
            n_mels: Mel-bin count of the logmel geometry *reference*/`x` were
                built from (Task 2 default 64) -- used to recover `n_frames =
                F // n_mels` from the flattened row width; `fit` raises if
                `F` is not an exact multiple (see `fit`).
        """
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.n_mels = n_mels
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None

    def fit(self, reference: np.ndarray) -> LstmAeScorer:
        """Fit an LSTM autoencoder on *reference*'s `(n_frames, n_mels)`
        patches (Adam, MSE loss over the full reconstructed patch, shuffled
        mini-batches, `epochs` passes -- `_train_autoencoder`).

        Args:
            reference: `(N, F)` finite matrix of flattened logmel patches,
                `F` an exact multiple of `n_mels`.

        Returns:
            self.

        Raises:
            ValueError: if `reference` is empty or has a non-finite value
                (`_check_reference`); if `reference.shape[1] % n_mels != 0`
                (checked BEFORE torch is ever imported -- a cheap guard
                against feeding a non-logmel variant, e.g. `audio`/`fusion`/
                `beats` feature vectors whose width has no relationship to
                `n_mels`).
            RuntimeError: if torch is not installed (`_require_torch`).
        """
        _check_reference(reference)
        f = reference.shape[1]
        if f % self.n_mels != 0:
            raise ValueError(
                f"reference width {f} is not a multiple of n_mels={self.n_mels} "
                f"-- LstmAeScorer needs a flattened logmel patch (Task 2's "
                f"frame-major (n_frames, n_mels) layout), not an arbitrary "
                f"feature vector"
            )
        n_frames = f // self.n_mels

        _require_torch()
        import torch

        torch.manual_seed(self.seed)
        device = _device()
        n_mels = self.n_mels
        hidden = self.hidden

        class _LstmAe(torch.nn.Module):
            """Encoder LSTM -> last hidden state -> repeated across frames ->
            decoder LSTM -> per-step Linear projection (class docstring)."""

            def __init__(self) -> None:
                super().__init__()
                self.encoder = torch.nn.LSTM(n_mels, hidden, batch_first=True)
                self.decoder = torch.nn.LSTM(hidden, hidden, batch_first=True)
                self.output = torch.nn.Linear(hidden, n_mels)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n = x.shape[0]
                patches = x.reshape(n, n_frames, n_mels)
                _, (h_n, _c_n) = self.encoder(patches)
                latent = h_n[-1]  # (N, hidden) -- final layer's hidden state
                repeated = latent.unsqueeze(1).repeat(1, n_frames, 1)
                decoded, _ = self.decoder(repeated)
                recon = self.output(decoded)  # (N, n_frames, n_mels)
                return cast(torch.Tensor, recon.reshape(n, n_frames * n_mels))

        model: torch.nn.Module = _LstmAe().to(device)
        x = torch.as_tensor(reference, dtype=torch.float32, device=device)
        _train_autoencoder(model, x, self.epochs, self.lr, self.batch_size, self.seed, device)
        self._model, self._device = model, device
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """`(W, F)` logmel-patch rows -> `(W,)` float64 scores, higher = more
        anomalous (mean-squared reconstruction error per row, over the full
        `(n_frames, n_mels)` patch).

        Raises:
            ValueError: if `x` contains non-finite values (`_check_query`).
            RuntimeError: if torch is not installed (`_require_torch`).
            AssertionError: if called before `fit()` (module docstring's
                "score() called before fit()" note).
        """
        _check_query(x)
        _require_torch()
        import torch

        assert self._model is not None, "LstmAeScorer.score() called before fit()"
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float32, device=self._device)
            recon = self._model(t)
            mse = ((recon - t) ** 2).mean(dim=1)
        return np.asarray(mse.cpu().numpy(), dtype=np.float64)


class ConvAeScorer:
    """2-D convolutional autoencoder baseline reconstruction scorer for
    `logmel` patches (package-3 spec D2) -- see module docstring for score
    polarity/definition and the lazy-torch story.

    Reshapes each `(F,)` flattened logmel row back into its own `(n_frames,
    n_mels)` patch (Task 2's frame-major flatten order, same as
    `LstmAeScorer`) and treats it as a single-channel `(1, n_mels, n_frames)`
    image -- mel (frequency) as height, frame (window-internal time) as
    width. A 2-layer strided `Conv2d` encoder downsamples this to a small
    spatial bottleneck; a mirrored `ConvTranspose2d` decoder upsamples back
    toward the original size. `stride=2` convolutions do not always invert to
    EXACTLY the original spatial size -- `Conv2d`/`ConvTranspose2d`'s own
    output-size formulas can differ from the input size by a pixel or two
    depending on `n_mels`/`n_frames`' parity (verified for both this class's
    test geometry, mels=8/frames=7, and the real logmel geometry,
    mels=64/frames=49) -- so the decoder's raw output is resized to the exact
    `(n_mels, n_frames)` target with `torch.nn.functional.interpolate(...,
    mode="nearest")` before computing the reconstruction loss: an
    unconditional crop/pad step, applied every `forward()` call regardless of
    whether the raw decoder output already matches (in which case it is a
    no-op -- nearest-neighbor resizing to an unchanged target size leaves the
    tensor unchanged).
    """

    name: str = "convae"

    def __init__(
        self,
        channels: tuple[int, int] = (16, 32),
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 128,
        seed: int = 7,
        n_mels: int = 64,
    ) -> None:
        """Args:
            channels: `(c1, c2)` output channel counts for the encoder's two
                `Conv2d` layers (`1 -> c1 -> c2`); the decoder mirrors this in
                reverse (`c2 -> c1 -> 1`).
            epochs: Full passes over *reference* per `fit()` call.
            lr: Adam learning rate.
            batch_size: Rows per gradient step (`_train_autoencoder`).
            seed: Seeds both `torch.manual_seed` (weight init, in `fit`) and
                the per-epoch shuffle generator (`_train_autoencoder`).
            n_mels: Mel-bin count of the logmel geometry *reference*/`x` were
                built from (Task 2 default 64) -- used to recover `n_frames =
                F // n_mels` from the flattened row width; `fit` raises if
                `F` is not an exact multiple (see `fit`).
        """
        self.channels = channels
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.n_mels = n_mels
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None

    def fit(self, reference: np.ndarray) -> ConvAeScorer:
        """Fit a convolutional autoencoder on *reference*'s `(n_mels,
        n_frames)` patches (Adam, MSE loss over the full reconstructed patch,
        shuffled mini-batches, `epochs` passes -- `_train_autoencoder`).

        Args:
            reference: `(N, F)` finite matrix of flattened logmel patches,
                `F` an exact multiple of `n_mels`.

        Returns:
            self.

        Raises:
            ValueError: if `reference` is empty or has a non-finite value
                (`_check_reference`); if `reference.shape[1] % n_mels != 0`
                (checked BEFORE torch is ever imported, same guard/rationale
                as `LstmAeScorer.fit`).
            RuntimeError: if torch is not installed (`_require_torch`).
        """
        _check_reference(reference)
        f = reference.shape[1]
        if f % self.n_mels != 0:
            raise ValueError(
                f"reference width {f} is not a multiple of n_mels={self.n_mels} "
                f"-- ConvAeScorer needs a flattened logmel patch (Task 2's "
                f"frame-major (n_frames, n_mels) layout), not an arbitrary "
                f"feature vector"
            )
        n_frames = f // self.n_mels

        _require_torch()
        import torch

        torch.manual_seed(self.seed)
        device = _device()
        n_mels = self.n_mels
        c1, c2 = self.channels
        target_size = (n_mels, n_frames)

        class _ConvAe(torch.nn.Module):
            """Conv2d/Conv2d encoder -> ConvTranspose2d/ConvTranspose2d
            decoder -> resize-to-exact-shape (class docstring)."""

            def __init__(self) -> None:
                super().__init__()
                self.enc1 = torch.nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1)
                self.enc2 = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
                self.dec1 = torch.nn.ConvTranspose2d(
                    c2, c1, kernel_size=3, stride=2, padding=1, output_padding=1
                )
                self.dec2 = torch.nn.ConvTranspose2d(
                    c1, 1, kernel_size=3, stride=2, padding=1, output_padding=1
                )
                self.relu = torch.nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n = x.shape[0]
                patch = x.reshape(n, n_frames, n_mels).transpose(1, 2).unsqueeze(1)
                h = self.relu(self.enc1(patch))
                h = self.relu(self.enc2(h))
                h = self.relu(self.dec1(h))
                h = self.dec2(h)
                h = torch.nn.functional.interpolate(h, size=target_size, mode="nearest")
                recon = h.squeeze(1).transpose(1, 2).reshape(n, n_frames * n_mels)
                return recon

        model: torch.nn.Module = _ConvAe().to(device)
        x = torch.as_tensor(reference, dtype=torch.float32, device=device)
        _train_autoencoder(model, x, self.epochs, self.lr, self.batch_size, self.seed, device)
        self._model, self._device = model, device
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """`(W, F)` logmel-patch rows -> `(W,)` float64 scores, higher = more
        anomalous (mean-squared reconstruction error per row, over the full
        `(n_mels, n_frames)` patch).

        Raises:
            ValueError: if `x` contains non-finite values (`_check_query`).
            RuntimeError: if torch is not installed (`_require_torch`).
            AssertionError: if called before `fit()` (module docstring's
                "score() called before fit()" note).
        """
        _check_query(x)
        _require_torch()
        import torch

        assert self._model is not None, "ConvAeScorer.score() called before fit()"
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float32, device=self._device)
            recon = self._model(t)
            mse = ((recon - t) ** 2).mean(dim=1)
        return np.asarray(mse.cpu().numpy(), dtype=np.float64)
