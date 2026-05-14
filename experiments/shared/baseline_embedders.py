"""Re-implemented prior-work embedders, faithful to the source papers.

Both architectures follow the same I/O contract as ``ResNetMetricEmbedder``
in ``experiments/shared/model.py``::

    forward(x: (B, C, T)) -> emb: (B, embed_dim)   # L2-normalized

so they slot into ``EEGMetricModel`` (denoiser -> embedder) unchanged.

* ``MindIDEmbedder`` follows Zhang et al., "MindID: Person Identification
  from Brain Waves through Attention-based Recurrent Neural Network"
  (arXiv:1711.06149, 2017). The paper's pipeline is

      (i)   Delta-band band-pass filter (0.5-4 Hz, 3rd-order Butterworth),
      (ii)  attention-based Encoder-Decoder LSTM,
      (iii) XGBoost identification head.

  Steps (i) and (ii) are reproduced here; (iii) is replaced by an MLP +
  L2-normalised projection so the embedding plugs into the same metric
  learning machinery as the rest of the pipeline (ArcFace / MultiSim /
  Contrastive / Triplet, dispatched via the trainer).

  The Butterworth IIR of the paper is approximated by a fixed FIR
  (Hamming-windowed sinc) bandpass so the filter runs as a grouped
  Conv1d on GPU without an scipy dependency. This preserves the same
  Delta-band selection while avoiding CPU-only scipy filtering.

* ``BrainNetEmbedder`` follows Fallahi, Strufe, Arias-Cabarcos,
  "BrainNet: Improving Brainwave-based Biometric Recognition with
  Siamese Networks" (IEEE PerCom 2023). The published topology
  (paper Fig. 3) is a 5-layer 1D CNN with *descending* filter counts
  ``[128, 32, 16, 8, 4]``, each followed by dropout and average pooling,
  ending with a flatten + dense projection. The paper trains with a
  FaceNet-style triplet loss; the trainer's ``loss="triplet"`` matches.

For both embedders, the original papers use different datasets,
electrode counts, and identification heads. We keep the *encoder*
faithful and adapt only the head so the architecture can be trained
under the AEP-Hybrid protocol with the same denoiser + loss machinery
as the main model.
"""

from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _design_bandpass_fir(low_hz: float, high_hz: float, sfreq: float,
                         numtaps: int = 101) -> torch.Tensor:
    """Hamming-windowed sinc bandpass FIR. Returns a (numtaps,) tensor.

    Used in place of the paper's 3rd-order Butterworth so the filter can
    run as a grouped Conv1d on GPU. The transition band is wide enough
    for a 0.5-4 Hz Delta selection at 200 Hz.
    """
    if numtaps % 2 == 0:
        numtaps += 1  # symmetric, odd length
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    fL = low_hz / sfreq
    fH = high_hz / sfreq
    # Ideal bandpass impulse response = HP(fL) intersected with LP(fH).
    with np.errstate(invalid="ignore"):
        h = 2 * fH * np.sinc(2 * fH * n) - 2 * fL * np.sinc(2 * fL * n)
    window = np.hamming(numtaps)
    h = h * window
    center = (low_hz + high_hz) / (2.0 * sfreq)
    phase = np.exp(-2j * np.pi * center * np.arange(numtaps))
    gain = np.abs(np.sum(h * phase))
    if gain > 0:
        h = h / gain
    return torch.tensor(h.astype(np.float32), dtype=torch.float32)


class MindIDEmbedder(nn.Module):
    """Attention-based Encoder-Decoder LSTM, MindID-style.

    The Encoder (paper Section 4.4) is

        input -> 3 x FC(164, tanh) -> LSTM(164)

    The attention block (paper Eqs. for ``W'_att``, ``W_att``, ``C_att``)
    is implemented as a second LSTM that takes the same FC-projected
    sequence and produces unnormalised attention scores, which are then
    softmax-normalised and used as elementwise weights on the encoder
    output sequence. The decoder is an MLP that projects the attended
    code to the embedding dimension.

    Because our trainer wants an L2-normalised embedding for the metric
    losses, we replace the paper's XGBoost head with a Linear + BN +
    L2-normalise stack.
    """

    def __init__(
        self,
        in_chans: int = 4,
        hidden: int = 164,
        embed_dim: int = 128,
        dropout: float = 0.3,
        sfreq: float = 200.0,
        delta_low: float = 0.5,
        delta_high: float = 4.0,
        fir_taps: int = 101,
        use_delta: bool = True,
    ):
        super().__init__()
        self.use_delta = use_delta
        self.in_chans = in_chans

        if use_delta:
            fir = _design_bandpass_fir(delta_low, delta_high, sfreq, fir_taps)
            # (in_chans, 1, K) for grouped Conv1d (one filter per channel).
            kernel = fir.view(1, 1, -1).repeat(in_chans, 1, 1)
            self.register_buffer("delta_kernel", kernel)
            self._fir_pad = fir.numel() // 2
        else:
            self.delta_kernel = None
            self._fir_pad = 0

        # 3 FC hidden layers, per-time-step (nn.Linear acts on last dim).
        self.fc_stack = nn.Sequential(
            nn.Linear(in_chans, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        # Encoder LSTM producing the code sequence C.
        self.code_lstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden,
            num_layers=1, batch_first=True,
        )
        # Parallel LSTM producing unnormalised attention weights W'_att.
        self.attn_lstm = nn.LSTM(
            input_size=hidden, hidden_size=hidden,
            num_layers=1, batch_first=True,
        )
        # Decoder MLP -> embedding.
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def _delta_filter(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> grouped Conv1d with one filter per channel.
        return F.conv1d(x, self.delta_kernel, padding=self._fir_pad,
                        groups=self.in_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DC removal + z-score per (sample, channel), before Delta filtering
        # as in paper Sections 4.2 -> 4.3.
        x = x - x.mean(dim=2, keepdim=True)
        x = x / (x.std(dim=2, keepdim=True) + 1e-6)
        if self.use_delta:
            x = self._delta_filter(x)

        # (B, C, T) -> (B, T, C) for per-time-step FCs and LSTMs.
        h = x.transpose(1, 2)
        h = self.fc_stack(h)                       # (B, T, hidden)
        code, _ = self.code_lstm(h)                # (B, T, hidden)
        attn_raw, _ = self.attn_lstm(h)            # (B, T, hidden)

        # Softmax over time, then elementwise weighting (paper's C_att).
        weights = torch.softmax(attn_raw, dim=1)
        c_att = code * weights
        # Pool over time -> fixed-length code, decode to embedding.
        pooled = c_att.sum(dim=1)
        emb = self.decoder(pooled)
        return F.normalize(emb, p=2, dim=1)


class BrainNetEmbedder(nn.Module):
    """1D CNN tower, BrainNet-faithful (Fallahi et al., 2023, Fig. 3).

    Five Conv1d blocks with *descending* filter counts
    ``(128, 32, 16, 8, 4)``. Each block is

        Conv1d -> BatchNorm1d -> ReLU -> Dropout -> AvgPool1d(k=2).

    Followed by a Flatten + LazyLinear projection to the paper's
    compact 32-unit embedding and a BatchNorm1d, then L2 normalisation.
    Kernel size is set to 7 (the paper does not pin it down; 7 is in
    line with similar EEG CNNs and keeps the temporal receptive field
    manageable at 200 Hz).
    """

    def __init__(
        self,
        in_chans: int = 4,
        channels=(128, 32, 16, 8, 4),
        kernel_size: int = 7,
        embed_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        c_prev = in_chans
        for c in channels:
            layers += [
                nn.Conv1d(c_prev, c, kernel_size=kernel_size,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.AvgPool1d(kernel_size=2, stride=2),
            ]
            c_prev = c
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        # LazyLinear sizes itself on the first forward pass; safer than
        # hand-computing T // 2**5 since T can vary across protocols.
        self.fc = nn.LazyLinear(embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.flatten(h)
        emb = self.bn(self.fc(h))
        return F.normalize(emb, p=2, dim=1)
