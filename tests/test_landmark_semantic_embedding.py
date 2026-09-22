"""Unit tests for fusion_avsr.models.landmark.semantic_embedding (Module 4)."""

import torch

from fusion_avsr.models.landmark.lrlp import NUM_LRLPS
from fusion_avsr.models.landmark.semantic_embedding import LRLPSemanticEmbedding


def test_semantic_embedding_output_shape():
    embed_dim = 64
    module = LRLPSemanticEmbedding(embed_dim=embed_dim)

    output = module(batch_size=2, num_frames=5)

    assert output.shape == (2, NUM_LRLPS, embed_dim, 5)


def test_semantic_embedding_identical_across_batch_and_time():
    module = LRLPSemanticEmbedding(embed_dim=16)

    output = module(batch_size=3, num_frames=4)

    # Every batch element and every frame should share the same per-landmark embedding.
    for b in range(1, 3):
        torch.testing.assert_close(output[b], output[0])
    for t in range(1, 4):
        torch.testing.assert_close(output[0, :, :, t], output[0, :, :, 0])


def test_semantic_embedding_distinct_per_landmark():
    module = LRLPSemanticEmbedding(embed_dim=16)

    output = module(batch_size=1, num_frames=1)  # (1, K, D, 1)
    per_landmark = output[0, :, :, 0]  # (K, D)

    # No two (randomly initialized) landmark embeddings should be identical.
    for i in range(per_landmark.shape[0]):
        for j in range(i + 1, per_landmark.shape[0]):
            assert not torch.allclose(per_landmark[i], per_landmark[j])
