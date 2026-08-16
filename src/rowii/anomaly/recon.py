"""Reconstruction anomaly scorers: MLP-AE on any feature
vector; LSTM-AE / Conv-AE on `logmel` patches, reshaped internally so the
window-internal time axis is the sequence (no cross-window contiguity; the
`Scorer` protocol holds unchanged). Score = per-window reconstruction MSE,
higher = more anomalous (explicit polarity by construction:
an autoencoder trained on normal windows reconstructs a normal window well
(low MSE) and an anomalous one poorly (high MSE), so no sign flip is needed
here unlike this package's own `OcSvmScorer`/`IsolationForestScorer`/
`LofScorer`, which negate an underlying "higher = more normal" sklearn
quantity).

Torch never appears at this module's import time: the `torch.nn.Module`
architectures live in `rowii.anomaly._recon_models` (the one module under
`rowii.anomaly` that imports torch at module level), imported lazily INSIDE
`fit()` after `_require_torch()` has confirmed torch is importable -- so
importing THIS module never requires the `[beats]` extra, and calling
`fit`/`score` without torch installed raises `RuntimeError` with the same
install hint as `scripts/run_step2.py`'s `_import_beats_or_exit` guard. The
two-module split mirrors `rowii.signals.beats` (torch-free until called)
delegating to `rowii.signals.beats_model` (eager torch), and is also what
keeps `mypy --strict` clean: a lazily-acquired torch handle can only be typed
`Any`, and mypy rejects subclassing a value of type `Any` ("Class cannot
subclass ... has type Any"), so nn.Module subclasses need a REAL top-level
`import torch` somewhere -- `_recon_models` is that place, and nothing in
THIS module ever needs to subclass torch at all (its own lazy `import torch`
statements resolve against torch's real installed stubs for everything they
touch).

`score()` before `fit()` raises `ValueError("<Class>.score() called before
fit()")` -- the same exception type, precondition, and message shape as
`rowii.anomaly.scorers`' own scorers (`KnnScorer.score`,
`MahalanobisScorer.score`), one convention across the whole Scorer family
(the first cut raised AssertionError here).

Device: `_device()` delegates to `rowii.signals.beats.best_device()` --
identical `ROWII_FORCE_CPU` env > mps > cuda > cpu priority as the BEATs
featurizer, so a CPU fallback always works even with no GPU backend present.
Training determinism: every class seeds BOTH `torch.manual_seed`
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

from typing import TYPE_CHECKING

import numpy as np

from rowii.anomaly.scorers import _check_query, _check_reference

if TYPE_CHECKING:
    import torch

_TORCH_HINT = (
    "reconstruction scorers need torch: pip install -e '.[beats]'"
)


def _require_torch() -> None:
    """Raise `RuntimeError` (with the shared install hint) if torch is not
    importable; a no-op otherwise. Callers follow this with their own local
    `import torch` and/or `from rowii.anomaly import _recon_models` (module
    docstring: the lazy-import story).
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
    (the one piece of `fit()` that does NOT vary
    per class; only the encoder/decoder architecture differs). Adam + MSE
    loss, `epochs` full passes over *reference_t*, each pass shuffled into
    `batch_size`-row mini-batches by a *seed*-seeded `torch.Generator`.
    Leaves *model* in `.eval()` mode on return (every `score()` in this
    module reads its model immediately under `torch.no_grad()`).

    Args:
        model: An already-constructed, already-`.to(device)`'d autoencoder
            (any `torch.nn.Module` whose `forward` maps a `(N, F)` batch to
            an `(N, F)` reconstruction of the SAME shape -- the three
            `rowii.anomaly._recon_models` architectures all do).
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
    """MLP autoencoder baseline reconstruction scorer (architecture:
    `rowii.anomaly._recon_models._MlpAe`) -- see module
    docstring for score polarity/definition and the lazy-torch story.

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

        from rowii.anomaly import _recon_models

        torch.manual_seed(self.seed)
        device = _device()
        model: torch.nn.Module = _recon_models._MlpAe(
            n_features=reference.shape[1], hidden=self.hidden
        ).to(device)
        x = torch.as_tensor(reference, dtype=torch.float32, device=device)
        _train_autoencoder(model, x, self.epochs, self.lr, self.batch_size, self.seed, device)
        self._model, self._device = model, device
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """`(W, F)` windows -> `(W,)` float64 scores, higher = more anomalous
        (mean-squared reconstruction error per row).

        Raises:
            ValueError: if `x` contains non-finite values (`_check_query`),
                or if called before `fit()` (module docstring's consistency
                note).
            RuntimeError: if torch is not installed (`_require_torch`).
        """
        _check_query(x)
        model = self._model
        if model is None:
            raise ValueError("MlpAeScorer.score() called before fit()")
        _require_torch()
        import torch

        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float32, device=self._device)
            recon = model(t)
            mse = ((recon - t) ** 2).mean(dim=1)
        return np.asarray(mse.cpu().numpy(), dtype=np.float64)


class LstmAeScorer:
    """LSTM autoencoder baseline reconstruction scorer for `logmel` patches
    (architecture: `rowii.anomaly._recon_models._LstmAe`)
    -- see module docstring for score polarity/definition and the lazy-torch
    story.

    Reshapes each `(F,)` flattened logmel row back into its own `(n_frames,
    n_mels)` patch (`n_frames = F // n_mels`, frame-major -- `rowii.signals.
    logmel.LogmelFeaturizer`'s own flatten order) and treats the
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
                built from (default 64) -- used to recover `n_frames =
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

        from rowii.anomaly import _recon_models

        torch.manual_seed(self.seed)
        device = _device()
        model: torch.nn.Module = _recon_models._LstmAe(
            n_frames=n_frames, n_mels=self.n_mels, hidden=self.hidden
        ).to(device)
        x = torch.as_tensor(reference, dtype=torch.float32, device=device)
        _train_autoencoder(model, x, self.epochs, self.lr, self.batch_size, self.seed, device)
        self._model, self._device = model, device
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """`(W, F)` logmel-patch rows -> `(W,)` float64 scores, higher = more
        anomalous (mean-squared reconstruction error per row, over the full
        `(n_frames, n_mels)` patch).

        Raises:
            ValueError: if `x` contains non-finite values (`_check_query`),
                or if called before `fit()` (module docstring's consistency
                note).
            RuntimeError: if torch is not installed (`_require_torch`).
        """
        _check_query(x)
        model = self._model
        if model is None:
            raise ValueError("LstmAeScorer.score() called before fit()")
        _require_torch()
        import torch

        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float32, device=self._device)
            recon = model(t)
            mse = ((recon - t) ** 2).mean(dim=1)
        return np.asarray(mse.cpu().numpy(), dtype=np.float64)


class ConvAeScorer:
    """2-D convolutional autoencoder baseline reconstruction scorer for
    `logmel` patches (architecture:
    `rowii.anomaly._recon_models._ConvAe`) -- see module docstring for score
    polarity/definition and the lazy-torch story.

    Reshapes each `(F,)` flattened logmel row back into its own `(n_frames,
    n_mels)` patch (the frame-major flatten order, same as
    `LstmAeScorer`) and treats it as a single-channel `(1, n_mels, n_frames)`
    image -- mel (frequency) as height, frame (window-internal time) as
    width. A 2-layer strided `Conv2d` encoder downsamples this to a small
    spatial bottleneck; a mirrored `ConvTranspose2d` decoder upsamples back
    to EXACTLY the original `(n_mels, n_frames)` shape: stride-2 transposes
    alone do not always invert a stride-2 conv's floor-division downsampling
    (the two layers' output-size formulas disagree by input parity), so each
    decoder layer's `output_padding` is computed per dimension in closed form
    (`rowii.anomaly._recon_models._transpose_output_padding`:
    `output_padding = l_target - (2*l_in - 1)`, always 0 or 1 here) -- no
    interpolate/crop/pad step anywhere, verified exact at both the test
    geometry (8 mels x 7 frames) and the real logmel geometry (64 x 49) by
    `tests/test_recon.py`'s raw-decoder-shape test.
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
                built from (default 64) -- used to recover `n_frames =
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

        from rowii.anomaly import _recon_models

        torch.manual_seed(self.seed)
        device = _device()
        model: torch.nn.Module = _recon_models._ConvAe(
            n_frames=n_frames, n_mels=self.n_mels, channels=self.channels
        ).to(device)
        x = torch.as_tensor(reference, dtype=torch.float32, device=device)
        _train_autoencoder(model, x, self.epochs, self.lr, self.batch_size, self.seed, device)
        self._model, self._device = model, device
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        """`(W, F)` logmel-patch rows -> `(W,)` float64 scores, higher = more
        anomalous (mean-squared reconstruction error per row, over the full
        `(n_mels, n_frames)` patch).

        Raises:
            ValueError: if `x` contains non-finite values (`_check_query`),
                or if called before `fit()` (module docstring's consistency
                note).
            RuntimeError: if torch is not installed (`_require_torch`).
        """
        _check_query(x)
        model = self._model
        if model is None:
            raise ValueError("ConvAeScorer.score() called before fit()")
        _require_torch()
        import torch

        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float32, device=self._device)
            recon = model(t)
            mse = ((recon - t) ** 2).mean(dim=1)
        return np.asarray(mse.cpu().numpy(), dtype=np.float64)
