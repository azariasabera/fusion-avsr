"""Module 3 -- LCFE (Landmark Coordinates Feature Extraction), Sheng et al. 2022, Section III.B.

A lightweight 1D CNN, applied independently (shared weights) to each of
the 38 LRLPs' nose-tip-aligned coordinate sequences, transforming a
``2 x T`` coordinate sequence into a ``256 x T`` feature vector -- the
same shape LMFE (Module 2) produces, so the two can be concatenated.

Nose-tip alignment (``fusion_avsr.models.landmark.lrlp.align_to_nose_tip``)
is this module's own required preprocessing step per the paper ("input
LRLP coordinates" for LCFE are explicitly the aligned ones), but is kept
as a separate function call rather than baked into this ``nn.Module`` --
callers (the dataset/encoder) must align coordinates before calling
``LCFE.forward``, the same way Module 1's raw output stays reusable for
both LMFE and LCFE.
"""

from __future__ import annotations

import torch
from torch import nn

from fusion_avsr.utils.logging import get_logger

logger = get_logger(__name__)

LCFE_OUTPUT_CHANNELS = 256
LCFE_TEMPORAL_RECEPTIVE_FIELD = 5


class LCFE(nn.Module):
    """1D CNN extracting a 256-dim feature per frame from each LRLP's aligned coordinate sequence.

    A single ``Conv1d`` with ``kernel_size=5`` and 'same' padding gives
    exactly the 5-frame temporal receptive field the paper specifies for
    this module (matching LMFE's), in the single 1D conv layer the paper
    text describes.
    """

    def __init__(self, output_channels: int = LCFE_OUTPUT_CHANNELS) -> None:
        """Build the 1D CNN.

        Args:
            output_channels: Number of output feature channels. Defaults
                to 256, matching LMFE's output so the two can be
                concatenated.
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=output_channels,
            kernel_size=LCFE_TEMPORAL_RECEPTIVE_FIELD,
            padding=LCFE_TEMPORAL_RECEPTIVE_FIELD // 2,
        )
        self.norm = nn.BatchNorm1d(output_channels)
        self.activation = nn.ReLU(inplace=True)
        self.output_channels = output_channels

    def forward(self, aligned_coords: torch.Tensor) -> torch.Tensor:
        """Extract per-frame coordinate features from a batch of aligned LRLP coordinate sequences.

        Args:
            aligned_coords: ``(B, K, 2, T)`` nose-tip-relative coordinate
                sequences (as produced by
                ``fusion_avsr.models.landmark.lrlp.align_to_nose_tip``,
                batched and converted to a float tensor by the caller),
                with ``K == 38``.

        Returns:
            A ``(B, K, output_channels, T)`` tensor of per-landmark,
            per-frame coordinate features. ``T`` is unchanged from the
            input ('same' padding).
        """
        batch_size, num_landmarks, _, num_frames = aligned_coords.shape
        x = aligned_coords.reshape(batch_size * num_landmarks, 2, num_frames)
        x = self.activation(self.norm(self.conv(x)))
        return x.reshape(batch_size, num_landmarks, self.output_channels, num_frames)
