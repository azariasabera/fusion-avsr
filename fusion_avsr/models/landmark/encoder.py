"""Landmark (local-stream) encoder: combines Modules 1-5 into one forward pass.

Wires together LMFE (Module 2) and LCFE (Module 3) -- run on Module 1's
patch/coordinate output -- with the Module 4 semantic embedding and the
Module 5 ASST-GCN, following the "Combine" step of the landmark encoder
spec: concatenate LMFE and LCFE per landmark, add the semantic embedding,
feed the result into the ASST-GCN.

This module takes already-batched tensors (patches, aligned coordinates)
as input -- turning one clip's raw video + landmark ``.pkl`` file into
those tensors is Module 1's job
(``fusion_avsr.models.landmark.lrlp.extract_lrlp_sequence`` +
``align_to_nose_tip``), done once per clip in the dataset/dataloader, not
repeated here.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from fusion_avsr.models.landmark.asst_gcn import ASST_GCN_CHANNELS, ASSTGCN
from fusion_avsr.models.landmark.lcfe import LCFE
from fusion_avsr.models.landmark.lmfe import LMFE
from fusion_avsr.models.landmark.semantic_embedding import LRLPSemanticEmbedding


class LandmarkEncoder(nn.Module):
    """The full local-stream landmark encoder: LMFE + LCFE + semantic embedding + ASST-GCN."""

    def __init__(self, output_dim: Optional[int] = None) -> None:
        """Build the landmark encoder.

        Args:
            output_dim: If given, the ASST-GCN's final linear projection
                target -- e.g. the shared fusion dimension used by the
                audio/appearance encoders. If ``None`` (default), the raw
                512-dim (LMFE 256 + LCFE 256) pooled feature is returned
                unprojected.
        """
        super().__init__()
        self.lmfe = LMFE()
        self.lcfe = LCFE()
        self.semantic_embedding = LRLPSemanticEmbedding(embed_dim=self.lmfe.output_channels + self.lcfe.output_channels)
        self.asst_gcn = ASSTGCN(channels=ASST_GCN_CHANNELS, output_dim=output_dim)

    def forward(self, patches: torch.Tensor, aligned_coords: torch.Tensor) -> torch.Tensor:
        """Run the full landmark encoder.

        Args:
            patches: ``(B, K, T, 32, 32)`` grayscale LRLP patch sequences
                (Module 1's ``extract_lrlp_sequence`` output, batched and
                converted to a float tensor), ``K == 38``.
            aligned_coords: ``(B, K, 2, T)`` nose-tip-relative LRLP
                coordinate sequences (Module 1's raw output, aligned via
                ``align_to_nose_tip``, batched and converted to a float
                tensor), ``K == 38``.

        Returns:
            A ``(B, T, output_dim)`` tensor (or ``(B, T, 512)`` if
            ``output_dim`` was not set at construction time) of per-frame
            landmark-encoder features, ready to feed into fusion.

        Raises:
            ValueError: If ``patches`` and ``aligned_coords`` disagree on
                batch size, landmark count, or frame count.
        """
        if patches.shape[0] != aligned_coords.shape[0] or patches.shape[1] != aligned_coords.shape[1] or patches.shape[2] != aligned_coords.shape[3]:
            raise ValueError(
                f"patches shape {tuple(patches.shape)} (B, K, T, H, W) and aligned_coords shape "
                f"{tuple(aligned_coords.shape)} (B, K, 2, T) must agree on (B, K, T)."
            )

        batch_size, _, num_frames, _, _ = patches.shape

        motion_features = self.lmfe(patches)  # (B, K, 256, T)
        coordinate_features = self.lcfe(aligned_coords)  # (B, K, 256, T)
        combined = torch.cat([motion_features, coordinate_features], dim=2)  # (B, K, 512, T)

        semantic = self.semantic_embedding(batch_size=batch_size, num_frames=num_frames)  # (B, K, 512, T)
        combined = combined + semantic

        node_features = combined.permute(0, 3, 1, 2)  # (B, T, K, 512)
        return self.asst_gcn(node_features)  # (B, T, output_dim or 512)
