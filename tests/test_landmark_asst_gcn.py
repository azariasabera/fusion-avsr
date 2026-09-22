"""Unit tests for fusion_avsr.models.landmark.asst_gcn (Module 5)."""

import torch

from fusion_avsr.models.landmark.asst_gcn import (
    ASST_GCN_CHANNELS,
    ASST_GCN_NUM_LAYERS,
    ASST_GCN_NUM_SUBGRAPHS,
    SEMANTIC_GRAPH_INIT_VALUE,
    ASSTGCN,
    ASSTGCNLayer,
)
from fusion_avsr.models.landmark.lrlp import NUM_LRLPS


def test_asst_gcn_layer_output_shape():
    batch_size, num_frames = 2, 5
    layer = ASSTGCNLayer(channels=64, num_subgraphs=8, num_nodes=NUM_LRLPS)
    x = torch.randn(batch_size, num_frames, NUM_LRLPS, 64)

    output = layer(x)

    assert output.shape == x.shape


def test_asst_gcn_layer_semantic_adjacency_init_value():
    layer = ASSTGCNLayer(channels=32, num_subgraphs=4, num_nodes=NUM_LRLPS)

    assert layer.semantic_adjacency.shape == (4, NUM_LRLPS, NUM_LRLPS)
    assert torch.allclose(layer.semantic_adjacency, torch.full_like(layer.semantic_adjacency, SEMANTIC_GRAPH_INIT_VALUE))


def test_asst_gcn_layer_rejects_non_divisible_channels():
    import pytest
    with pytest.raises(ValueError):
        ASSTGCNLayer(channels=33, num_subgraphs=8, num_nodes=NUM_LRLPS)


def test_asst_gcn_stack_default_config():
    module = ASSTGCN()
    assert len(module.layers) == ASST_GCN_NUM_LAYERS
    assert module.channels == ASST_GCN_CHANNELS
    assert module.layers[0].num_subgraphs == ASST_GCN_NUM_SUBGRAPHS


def test_asst_gcn_output_shape_without_projection():
    batch_size, num_frames = 2, 4
    module = ASSTGCN(num_layers=2, channels=64, num_subgraphs=8, num_nodes=NUM_LRLPS)
    x = torch.randn(batch_size, num_frames, NUM_LRLPS, 64)

    output = module(x)

    # GAP over the landmark (K) dimension: node dim disappears.
    assert output.shape == (batch_size, num_frames, 64)


def test_asst_gcn_output_shape_with_projection():
    module = ASSTGCN(num_layers=2, channels=64, num_subgraphs=8, num_nodes=NUM_LRLPS, output_dim=768)
    x = torch.randn(1, 3, NUM_LRLPS, 64)

    output = module(x)

    assert output.shape == (1, 3, 768)
