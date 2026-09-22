"""Unit tests for fusion_avsr.models.landmark.pretrain."""

import torch

from fusion_avsr.models.landmark.lrlp import NUM_LRLPS
from fusion_avsr.models.landmark.pretrain import GridWordRecognitionModel, masked_mean_pool


def test_masked_mean_pool_ignores_padded_frames():
    # Two real frames of value 2.0, then two padded frames of value 100.0
    # that must not influence the mean.
    sequence = torch.cat([torch.full((1, 2, 4), 2.0), torch.full((1, 2, 4), 100.0)], dim=1)
    mask = torch.tensor([[True, True, False, False]])

    pooled = masked_mean_pool(sequence, mask)

    torch.testing.assert_close(pooled, torch.full((1, 4), 2.0))


def test_masked_mean_pool_no_mask_averages_everything():
    sequence = torch.ones(2, 3, 4)

    pooled = masked_mean_pool(sequence, mask=None)

    torch.testing.assert_close(pooled, torch.ones(2, 4))


def test_grid_word_recognition_model_output_shape():
    batch_size, num_frames, num_classes = 2, 5, 51
    model = GridWordRecognitionModel(num_classes=num_classes)
    patches = torch.randn(batch_size, NUM_LRLPS, num_frames, 32, 32)
    coords = torch.randn(batch_size, NUM_LRLPS, 2, num_frames)
    mask = torch.ones(batch_size, num_frames, dtype=torch.bool)

    logits = model(patches, coords, mask)

    assert logits.shape == (batch_size, num_classes)
