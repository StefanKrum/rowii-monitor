"""Compact TF-C (Time-Frequency Consistency -- Zhang et al. 2022) model +
loss (package-4 spec D1): the ONE module under `rowii.tfc` that imports
torch at module level, mirroring
`rowii.anomaly._recon_models`'s role in its own package (that module's own
docstring explains why: a lazily-acquired torch handle can only be typed
`Any`, and mypy rejects subclassing a value of type `Any`, so an
`nn.Module` subclass needs a REAL top-level `import torch` behind it). This
module is therefore imported ONLY lazily, from inside `rowii.tfc.wrapper`'s
functions/methods (`load_tfc_model`, `_RealTfcEncoder.embed`) -- never at
`wrapper.py`'s own top level, and never at this package's `__init__.py`.

Architecture: two independent 1-D CNN encoders (`_Cnn1d`, conv-stride-4
blocks -> global average pool -> linear head), one over the raw time-domain
waveform and one over its magnitude spectrum (`freq_view`), each followed by
a small linear projection head. `tfc_loss` is the pretraining objective: a
cross-view NT-Xent (SimCLR-style contrastive loss) treating each window's
own `(z_t, z_f)` pair as the single positive, every other projection in the
batch (both views) as a negative -- the paper's time-frequency CONSISTENCY
idea recast as a contrastive alignment task. This is a documented
simplification of the full TF-C paper, which also contrasts each view
against its own AUGMENTED counterpart (intra-view pairs); this compact
version keeps only the cross-view pair, which is what actually matters for
this project's use of TF-C (a frozen embedding source, not a benchmark
reproduction of the original paper's full pretraining recipe).
"""
from __future__ import annotations

from typing import cast

import torch

from rowii.tfc.wrapper import TfcConfig


class _Cnn1d(torch.nn.Module):
    """Shared 1-D CNN encoder architecture for both the time and frequency
    branches (`TfcModel.time_encoder`/`freq_encoder` are two INDEPENDENT
    instances of this class, never weight-tied): `len(cfg.channels)`
    conv-stride-4 blocks (`Conv1d(kernel=8, stride=4, padding=2)` ->
    `BatchNorm1d` -> `ReLU`), each roughly quartering the sequence length,
    followed by a global average pool over whatever length remains and a
    `Linear` head to `cfg.embed_dim`. Global average pooling (rather than a
    fixed-size flatten) is what lets ONE architecture serve both branches
    despite them seeing different-length inputs (the time view is
    `cfg.n_samples` samples; `freq_view`'s magnitude spectrum is
    `cfg.n_samples // 2 + 1` frequency bins) -- no shape bookkeeping needed
    between the two.
    """

    def __init__(self, cfg: TfcConfig) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_ch = 1
        for out_ch in cfg.channels:
            layers += [
                torch.nn.Conv1d(in_ch, out_ch, kernel_size=8, stride=4, padding=2),
                torch.nn.BatchNorm1d(out_ch),
                torch.nn.ReLU(),
            ]
            in_ch = out_ch
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(cfg.channels[-1], cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, N)` -> `(B, cfg.embed_dim)`, any `N` (see class docstring)."""
        h = cast(torch.Tensor, self.body(x.unsqueeze(1)))  # (B, 1, N) -> (B, C, N')
        h = h.mean(dim=2)  # global average pool -> (B, C)
        return cast(torch.Tensor, self.head(h))


class TfcModel(torch.nn.Module):
    """Time encoder + frequency encoder + projection heads (spec D1).

    `forward` returns both the pooled encoder outputs (`h_t`, `h_f` --
    `cfg.embed_dim`-wide, what `rowii.tfc.wrapper.TfcFeaturizer` actually
    uses as the frozen embedding, concatenated) and their projections (`z_t`,
    `z_f` -- `cfg.proj_dim`-wide, what `tfc_loss` consumes during
    pretraining). Keeping both pairs on one `forward` call, rather than two
    separate encode()/project() calls, avoids ever running the CNN body
    twice for one input.
    """

    def __init__(self, cfg: TfcConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.time_encoder = _Cnn1d(cfg)
        self.freq_encoder = _Cnn1d(cfg)
        self.time_proj = torch.nn.Linear(cfg.embed_dim, cfg.proj_dim)
        self.freq_proj = torch.nn.Linear(cfg.embed_dim, cfg.proj_dim)

    def forward(
        self, x_time: torch.Tensor, x_freq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`x_time`: `(B, cfg.n_samples)` raw waveform. `x_freq`: `(B, N_f)`
        magnitude spectrum, normally `freq_view(x_time)` (kept as a separate
        argument, not computed internally, so callers can standardize/window
        the time view once and derive both from the SAME preprocessed
        tensor -- see `rowii.tfc.wrapper._RealTfcEncoder.embed`).

        Returns: `(h_t, h_f, z_t, z_f)`, each `(B, cfg.embed_dim)` for the
        first two and `(B, cfg.proj_dim)` for the last two.
        """
        h_t = self.time_encoder(x_time)
        h_f = self.freq_encoder(x_freq)
        z_t = cast(torch.Tensor, self.time_proj(h_t))
        z_f = cast(torch.Tensor, self.freq_proj(h_f))
        return h_t, h_f, z_t, z_f


def freq_view(x_time: torch.Tensor) -> torch.Tensor:
    """`(B, N)` time-domain windows -> `(B, N // 2 + 1)` standardized
    magnitude spectrum: `|rfft(x)|`, per-window (row) zero-mean/unit-std
    normalized (`1e-8`-floored std, the same convention as
    `rowii.tfc.wrapper._standardize`) so the frequency view's scale does not
    depend on the time view's own amplitude/units.
    """
    mag = torch.abs(torch.fft.rfft(x_time, dim=1))
    mean = mag.mean(dim=1, keepdim=True)
    std = mag.std(dim=1, keepdim=True).clamp_min(1e-8)
    return (mag - mean) / std


def tfc_loss(z_t: torch.Tensor, z_f: torch.Tensor, temperature: float) -> torch.Tensor:
    """Cross-view NT-Xent (module docstring): for each window `i`, `(z_t_i,
    z_f_i)` is the one positive pair; every other projection in the batch --
    both `z_t_j` and `z_f_j` for `j != i`, AND the other view of `i` itself
    is excluded via the diagonal mask below -- is a negative.

    Args:
        z_t: `(B, D)` time-view projections.
        z_f: `(B, D)` frequency-view projections, paired index-for-index with
            `z_t` (i.e. `z_t[i]`/`z_f[i]` come from the SAME window).
        temperature: NT-Xent softmax temperature (`TfcConfig.temperature`).

    Returns:
        Scalar loss tensor.
    """
    z_t = torch.nn.functional.normalize(z_t, dim=1)
    z_f = torch.nn.functional.normalize(z_f, dim=1)
    b = z_t.shape[0]
    z = torch.cat([z_t, z_f], dim=0)  # (2B, D): rows [0, B) time, [B, 2B) freq
    sim = z @ z.T / temperature  # (2B, 2B) pairwise cosine similarity / T
    sim.fill_diagonal_(float("-inf"))  # exclude self-similarity from the softmax
    # Row i in [0, B) (a time view) has its positive at column B + i (its own
    # frequency view); row B + i has its positive at column i -- symmetric.
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return torch.nn.functional.cross_entropy(sim, targets)
