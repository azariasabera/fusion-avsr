"""Module 2 -- LMFE (Local Motion Feature Extraction), Sheng et al. 2022, Section III.B.

A lightweight 3-layer 3D CNN, applied independently (shared weights) to
each of the 38 LRLP patch sequences, transforming a ``T x 32 x 32``
grayscale patch sequence into a ``256 x T`` feature vector.

The paper fixes the input/output shapes, the layer count (3), the
temporal receptive field (5 frames), and the output channel count (256,
matching LCFE so the two can be concatenated).
"""

from __future__ import annotations

import torch
from torch import nn

from fusion_avsr.utils.logging import get_logger

logger = get_logger(__name__)

LMFE_OUTPUT_CHANNELS = 256
LMFE_TEMPORAL_RECEPTIVE_FIELD = 5


class LMFE(nn.Module):
    """3-layer 3D CNN extracting a 256-dim feature per frame from each LRLP's patch sequence.

    Implementation note: the paper (Table II) specifies a 3-layer 3D CNN
    with a 5-frame temporal receptive field and 256 output channels. This 
    implementation is a standard, explicitly documented choice satisfying every
    constraint the paper text does fix: 3 conv layers, temporal kernel
    sizes ``(3, 1, 3)`` with 'same' padding give a combined temporal
    receptive field of exactly 5 frames (``1 + 2 + 0 + 2``); spatial
    kernels progressively downsample ``32x32 -> 16x16 -> 8x8 -> 4x4``,
    followed by adaptive average pooling to ``1x1``; the final layer has
    256 output channels. Each conv is followed by BatchNorm3d + ReLU.
    """

    def __init__(self, output_channels: int = LMFE_OUTPUT_CHANNELS) -> None:
        """Build the 3-layer 3D CNN.

        Args:
            output_channels: Number of output feature channels. Defaults
                to 256, matching LCFE's output so the two can be
                concatenated.
        """
        super().__init__()
        mid_channels = output_channels // 2

        self.layers = nn.Sequential(
            nn.Conv3d(1, mid_channels // 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels // 2, mid_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, output_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.output_channels = output_channels

    def forward(self, patch_sequences: torch.Tensor) -> torch.Tensor:
        """Extract per-frame motion features from a batch of LRLP patch sequences.

        Args:
            patch_sequences: ``(B, K, T, H, W)`` grayscale patch
                sequences (as produced by
                ``fusion_avsr.models.landmark.lrlp.extract_lrlp_sequence``,
                batched and converted to a float tensor by the caller),
                with ``H == W == 32`` and ``K == 38``.

        Returns:
            A ``(B, K, output_channels, T)`` tensor of per-landmark,
            per-frame motion features. ``T`` is unchanged from the input
            (every conv layer uses 'same' temporal padding).
        """
        batch_size, num_landmarks, num_frames, height, width = patch_sequences.shape
        x = patch_sequences.reshape(batch_size * num_landmarks, 1, num_frames, height, width)
        x = self.layers(x)
        x = self.spatial_pool(x).squeeze(-1).squeeze(-1)  # (B*K, C, T)
        return x.reshape(batch_size, num_landmarks, self.output_channels, num_frames)
