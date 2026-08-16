"""Compact CNN student encoder for `rowii.adapt.student`: the ONE module under
`rowii.adapt` -- alongside `lora.py`/`objective.py` -- that imports torch at
module level, mirroring
`rowii.anomaly._recon_models`'s/`rowii.tfc.model`'s identical role in their own
packages (those modules' own docstrings explain why: a lazily-acquired torch
handle can only be typed `Any`, and mypy rejects subclassing a value of type
`Any`, so an `nn.Module` subclass needs a REAL top-level `import torch` behind
it). This module is therefore imported ONLY lazily, from inside
`rowii.adapt.student`'s functions (`load_student_model`,
`_RealStudentEncoder.embed`) -- never at `student.py`'s own top level, and never
at this package's `__init__.py`.

Architecture (binding contract): three stride-2 `Conv2d` blocks
(`1 -> 32 -> 64 -> 128` channels by default, `kernel=3, stride=2, padding=1`,
`BatchNorm2d` + `ReLU` after each -- mirroring `rowii.tfc.model._Cnn1d`'s own
conv-then-norm-then-activation order, just in 2-D) over the `(1, n_mels,
n_frames)` log-mel patch image, followed by `AdaptiveAvgPool2d(1)` (so, unlike
`rowii.anomaly._recon_models._ConvAe`'s decoder, no closed-form output-size
bookkeeping is needed -- any channel/spatial geometry pools down to one vector)
and a `Linear(channels[-1], out_dim)` head to the 768-d embedding the frozen
BEATs teacher produces. `StudentConfig` (imported FROM `rowii.adapt.student`,
never the reverse -- the same one-directional import convention `tfc/model.py`
uses for `TfcConfig`) fixes `n_mels`/`n_frames`/`out_dim` for every checkpoint
this project trains; only `channels` legitimately varies (tiny/test
architectures).
"""
from __future__ import annotations

from typing import cast

import torch

from rowii.adapt.student import StudentConfig

_KERNEL = 3
_STRIDE = 2
_PADDING = 1


class _StudentNet(torch.nn.Module):
    """Conv2d x `len(cfg.channels)` (default 3: `1 -> 32 -> 64 -> 128`, BN+ReLU
    after each) over the `(cfg.n_mels, cfg.n_frames)` log-mel patch -> adaptive
    average pool -> `Linear(cfg.channels[-1], cfg.out_dim)`.

    Takes and returns the FLATTENED `(B, n_frames * n_mels)` layout (frame-major
    -- `rowii.signals.logmel.LogmelFeaturizer.transform`'s own flatten order),
    reshaping internally to mel-major image axes (mel = height, frame = width),
    the SAME convention `rowii.anomaly._recon_models._ConvAe.forward` uses for
    the identical reason: one `x.reshape(n, n_frames, n_mels).transpose(1, 2)
    .unsqueeze(1)` step turns the flattened patch `StudentFeaturizer.transform`
    (and, for tests, a stub `StudentEncoderProtocol`) hands around into the
    `(B, 1, n_mels, n_frames)` image this network's `Conv2d`s expect.
    """

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.n_mels = cfg.n_mels
        self.n_frames = cfg.n_frames
        layers: list[torch.nn.Module] = []
        in_ch = 1
        for out_ch in cfg.channels:
            layers += [
                torch.nn.Conv2d(
                    in_ch, out_ch, kernel_size=_KERNEL, stride=_STRIDE, padding=_PADDING
                ),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.ReLU(),
            ]
            in_ch = out_ch
        self.body = torch.nn.Sequential(*layers)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(cfg.channels[-1], cfg.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, n_frames * n_mels)` flattened log-mel patch -> `(B, out_dim)`.

        Raises:
            ValueError: *x*'s width does not equal `n_frames * n_mels` -- a
                clear, early failure instead of a cryptic `Conv2d`/`reshape`
                shape error deeper in `body` (e.g. a caller's *rate_hz*/window
                duration producing a differently-shaped log-mel patch than the
                geometry this network was built for; see `StudentFeaturizer`'s
                own docstring on why the real path is geometry-locked to the
                plant's 50 kHz / 1-s default).
        """
        n = x.shape[0]
        width = x.shape[1]
        expected = self.n_frames * self.n_mels
        if width != expected:
            raise ValueError(
                f"_StudentNet.forward expects a flattened (B, {expected}) "
                f"(n_frames={self.n_frames} x n_mels={self.n_mels}) log-mel patch, "
                f"got width {width}"
            )
        patch = x.reshape(n, self.n_frames, self.n_mels).transpose(1, 2).unsqueeze(1)
        h = cast(torch.Tensor, self.body(patch))
        h = self.pool(h).reshape(n, -1)
        return cast(torch.Tensor, self.head(h))
