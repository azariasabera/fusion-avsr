"""MS-TCN (Multi-Stage dilated TCN) decoder, Sheng et al. 2022, Fig. 4.

Per the paper's Fig. 4 (extracted directly from the source PDF text):
"It contains three blocks of Multi-scale dilated TCN, where the dilation
size of each block is 1, 2, 4 respectively. Every block consists of three
TCN layers, whose kernel sizes are 3, 5, 7 respectively."
"""

from __future__ import annotations

import torch
from torch import nn

MS_TCN_BLOCK_DILATIONS = (1, 2, 4)
MS_TCN_LAYER_KERNEL_SIZES = (3, 5, 7)
DEFAULT_MS_TCN_DROPOUT = 0.2


class _MSTCNLayer(nn.Module):
    """One dilated Conv1d layer with 'same' padding, BatchNorm, ReLU, dropout, and a residual connection.

    Dropout is not part of the paper's Fig. 4 description -- added here
    because this decoder is pretrained on GRID's narrow, repetitive
    vocabulary/grammar but needs to generalize to LRS3, a much larger and
    more varied domain, where an unregularized fit to GRID would
    otherwise overfit.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.activation(self.norm(self.conv(x))))


class MSTCN(nn.Module):
    """3 blocks x 3 dilated-Conv1d layers, per Fig. 4's kernel/dilation schedule."""

    def __init__(self, channels: int, dropout: float = DEFAULT_MS_TCN_DROPOUT) -> None:
        """Build the MS-TCN decoder.

        Args:
            channels: Input and output channel count, preserved
                throughout (every layer has a residual connection, so
                channels never change between layers).
            dropout: Dropout probability applied inside every layer (see
                ``_MSTCNLayer``). Defaults to 0.2.
        """
        super().__init__()
        layers = []
        for dilation in MS_TCN_BLOCK_DILATIONS:
            for kernel_size in MS_TCN_LAYER_KERNEL_SIZES:
                layers.append(_MSTCNLayer(channels, kernel_size, dilation, dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """Apply the MS-TCN decoder.

        Args:
            sequence_features: ``(B, C, T)`` feature sequence.

        Returns:
            A ``(B, C, T)`` tensor of the same shape.
        """
        return self.layers(sequence_features)
