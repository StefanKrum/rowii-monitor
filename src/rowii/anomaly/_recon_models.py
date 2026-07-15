"""Autoencoder architectures for `rowii.anomaly.recon` (package-3 spec D2,
Task-3 review refinement): the ONE module under `rowii.anomaly` that imports
torch at module level -- so it must only ever be imported lazily, from inside
`recon.py`'s `fit()` methods after `recon._require_torch()` has confirmed
torch is importable, never at another module's import time. The split mirrors
`rowii.signals.beats` (torch-free until called) delegating to
`rowii.signals.beats_model` (eager torch), and is what keeps `mypy --strict`
clean: a lazily-acquired torch handle can only be typed `Any`, and mypy
rejects subclassing a value of type `Any` outright ("Class cannot subclass
... has type Any"), so `torch.nn.Module` subclasses need a REAL top-level
`import torch` behind them -- this module is that place. Plain classes with
explicit constructor arguments (no closures, no per-`fit()`-call class
definitions); `recon.py` owns everything user-facing (input validation,
device placement, training via its `_train_autoencoder`, scoring) and this
module owns architecture only.
"""
from __future__ import annotations

from typing import cast

import torch

_KERNEL = 3
_STRIDE = 2
_PADDING = 1
"""`_ConvAe`'s fixed per-layer geometry (both encoder `Conv2d`s and both
decoder `ConvTranspose2d`s share it): kernel 3, stride 2, padding 1. The
closed-form `output_padding` derivation below is written against these
constants, so they are named once here rather than repeated as magic numbers
per layer."""


def _conv_out_len(l_in: int) -> int:
    """Output length along one spatial dimension of a `Conv2d(kernel_size=3,
    stride=2, padding=1)` layer: `floor((l_in + 2*padding - kernel) / stride)
    + 1`, which for this geometry is `l_in / 2` rounded UP (even `l_in` ->
    `l_in/2`, odd -> `(l_in+1)/2`)."""
    return (l_in + 2 * _PADDING - _KERNEL) // _STRIDE + 1


def _transpose_output_padding(l_in: int, l_target: int) -> int:
    """`output_padding` making a `ConvTranspose2d(kernel_size=3, stride=2,
    padding=1)` layer map length `l_in` to EXACTLY `l_target` along one
    spatial dimension. The transpose's own output-size formula is `l_out =
    (l_in - 1)*stride - 2*padding + kernel + output_padding = 2*l_in - 1 +
    output_padding`, hence `output_padding = l_target - (2*l_in - 1)`. For
    this module's only calling pattern -- inverting one `_conv_out_len` step,
    i.e. `l_in = _conv_out_len(l_target)` -- the result is always valid:
    even `l_target` gives 1, odd gives 0, both inside torch's required
    `[0, stride)` range (asserted below, trust-but-verify -- an out-of-range
    value can only mean a caller passed an `l_in`/`l_target` pair that is NOT
    a `_conv_out_len` inversion). Verified empirically at the test geometry
    (8 mels x 7 frames), the real logmel geometry (64 x 49), and three
    arbitrary mixed-parity geometries before replacing the previous
    interpolate-based resize with this closed form
    (`tests/test_recon.py::TestConvAeDecoderShape` pins the two geometries
    that matter).
    """
    raw_out = (l_in - 1) * _STRIDE - 2 * _PADDING + _KERNEL
    output_padding = l_target - raw_out
    assert 0 <= output_padding < _STRIDE, (
        f"output_padding {output_padding} outside [0, {_STRIDE}) for l_in={l_in}, "
        f"l_target={l_target} -- caller must pass l_in = _conv_out_len(l_target)"
    )
    return output_padding


class _MlpAe(torch.nn.Module):
    """Symmetric MLP autoencoder (`MlpAeScorer`'s architecture): encoder
    `n_features -> hidden[0] -> ... -> hidden[-1]` with ReLU after every
    Linear, mirrored Linear decoder back to `n_features` with ReLU between
    layers but NONE after the final layer (raw-valued reconstruction -- see
    `MlpAeScorer.__init__`'s `hidden` doc for why)."""

    def __init__(self, n_features: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        dims = [n_features, *hidden]
        layers: list[torch.nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
        rdims = list(reversed(dims))
        for i, (a, b) in enumerate(zip(rdims[:-1], rdims[1:], strict=True)):
            layers.append(torch.nn.Linear(a, b))
            if i < len(rdims) - 2:
                layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


class _LstmAe(torch.nn.Module):
    """Sequence autoencoder over one window's frames (`LstmAeScorer`'s
    architecture): encoder LSTM over the `(n_frames, n_mels)` patch -> final
    hidden state -> repeated across `n_frames` steps -> decoder LSTM ->
    per-step `Linear(hidden, n_mels)` projection. Takes and returns the
    FLATTENED `(N, n_frames * n_mels)` layout (frame-major, Task 2's logmel
    flatten order), reshaping internally, so `recon.py`'s shared training/
    scoring paths treat all three architectures identically."""

    def __init__(self, n_frames: int, n_mels: int, hidden: int) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.n_mels = n_mels
        self.encoder = torch.nn.LSTM(n_mels, hidden, batch_first=True)
        self.decoder = torch.nn.LSTM(hidden, hidden, batch_first=True)
        self.output = torch.nn.Linear(hidden, n_mels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        patches = x.reshape(n, self.n_frames, self.n_mels)
        _, (h_n, _c_n) = self.encoder(patches)
        latent = h_n[-1]  # (N, hidden) -- final layer's hidden state
        repeated = latent.unsqueeze(1).repeat(1, self.n_frames, 1)
        decoded, _ = self.decoder(repeated)
        recon = self.output(decoded)  # (N, n_frames, n_mels)
        return cast(torch.Tensor, recon.reshape(n, self.n_frames * self.n_mels))


class _ConvAe(torch.nn.Module):
    """2-D convolutional autoencoder over one window's `(1, n_mels,
    n_frames)` patch image (`ConvAeScorer`'s architecture): two stride-2
    `Conv2d` encoder layers (`1 -> c1 -> c2`, ReLU after each), mirrored
    `ConvTranspose2d` decoder (`c2 -> c1 -> 1`, ReLU between, bare final
    layer) whose per-layer, per-dimension `output_padding` is computed in
    closed form (`_transpose_output_padding`) from the encoder's own
    `_conv_out_len` shape chain -- so the decoder's raw output hits the exact
    `(n_mels, n_frames)` input shape at ANY geometry, with no crop/pad/
    resample step anywhere. Takes and returns the flattened
    `(N, n_frames * n_mels)` layout like `_LstmAe`, transposing internally to
    mel-major image axes (mel = height, frame = width)."""

    def __init__(self, n_frames: int, n_mels: int, channels: tuple[int, int]) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.n_mels = n_mels
        c1, c2 = channels
        h1, w1 = _conv_out_len(n_mels), _conv_out_len(n_frames)
        h2, w2 = _conv_out_len(h1), _conv_out_len(w1)
        self.enc1 = torch.nn.Conv2d(1, c1, kernel_size=_KERNEL, stride=_STRIDE, padding=_PADDING)
        self.enc2 = torch.nn.Conv2d(c1, c2, kernel_size=_KERNEL, stride=_STRIDE, padding=_PADDING)
        self.dec1 = torch.nn.ConvTranspose2d(
            c2, c1, kernel_size=_KERNEL, stride=_STRIDE, padding=_PADDING,
            output_padding=(
                _transpose_output_padding(h2, h1),
                _transpose_output_padding(w2, w1),
            ),
        )
        self.dec2 = torch.nn.ConvTranspose2d(
            c1, 1, kernel_size=_KERNEL, stride=_STRIDE, padding=_PADDING,
            output_padding=(
                _transpose_output_padding(h1, n_mels),
                _transpose_output_padding(w1, n_frames),
            ),
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        patch = x.reshape(n, self.n_frames, self.n_mels).transpose(1, 2).unsqueeze(1)
        h = self.relu(self.enc1(patch))
        h = self.relu(self.enc2(h))
        h = self.relu(self.dec1(h))
        h = self.dec2(h)
        recon = h.squeeze(1).transpose(1, 2).reshape(n, self.n_frames * self.n_mels)
        return cast(torch.Tensor, recon)
