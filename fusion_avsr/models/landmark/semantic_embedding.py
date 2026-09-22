"""Module 4 -- LRLP Semantic Encoding, Sheng et al. 2022, Section III.B.

A learned embedding keyed purely by landmark index (0..37) -- no image or
coordinate data involved. "The semantic encodings have the same dimension
as the fused features from the LMFE module and LCFE module so that the
two can be summed directly": since LMFE and LCFE each output 256
channels and are concatenated (not summed) to form the combined
per-landmark feature (see ``fusion_avsr.models.landmark.encoder``), the
embedding dimension here defaults to their concatenated width, 512.
"""

from __future__ import annotations

import torch
from torch import nn

from fusion_avsr.models.landmark.lrlp import NUM_LRLPS

DEFAULT_EMBED_DIM = 512  # LMFE (256) + LCFE (256) concatenated width.


class LRLPSemanticEmbedding(nn.Module):
    """Embedding lookup keyed by landmark index (0..37)."""

    def __init__(self, embed_dim: int = DEFAULT_EMBED_DIM) -> None:
        """Build the embedding table.

        Args:
            embed_dim: Embedding dimension. Must match whatever the LMFE
                +LCFE combined feature dimension is, since the two are
                summed elementwise (see
                ``fusion_avsr.models.landmark.encoder``).
        """
        super().__init__()
        self.embedding = nn.Embedding(NUM_LRLPS, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, batch_size: int, num_frames: int) -> torch.Tensor:
        """Return the per-landmark semantic embedding, broadcast over batch and time.

        Args:
            batch_size: Batch size ``B`` to broadcast the embedding over.
            num_frames: Number of frames ``T`` to broadcast the embedding
                over (the embedding is identical at every frame -- it
                encodes landmark identity, not motion).

        Returns:
            A ``(B, K, embed_dim, T)`` tensor, ``K == 38``, where every
            batch element and every frame shares the same per-landmark
            embedding.
        """
        device = self.embedding.weight.device
        landmark_indices = torch.arange(NUM_LRLPS, device=device)
        embeddings = self.embedding(landmark_indices)  # (K, embed_dim)
        embeddings = embeddings.unsqueeze(0).unsqueeze(-1)  # (1, K, embed_dim, 1)
        return embeddings.expand(batch_size, NUM_LRLPS, self.embed_dim, num_frames)
