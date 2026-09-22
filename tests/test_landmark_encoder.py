"""Unit tests for fusion_avsr.models.landmark.encoder (combining Modules 1-5)."""

import pytest
import torch

from fusion_avsr.models.landmark.encoder import LandmarkEncoder
from fusion_avsr.models.landmark.lrlp import NUM_LRLPS


def test_landmark_encoder_output_shape_unprojected():
    batch_size, num_frames = 2, 5
    encoder = LandmarkEncoder(output_dim=None)
    patches = torch.randn(batch_size, NUM_LRLPS, num_frames, 32, 32)
    coords = torch.randn(batch_size, NUM_LRLPS, 2, num_frames)

    output = encoder(patches, coords)

    assert output.shape == (batch_size, num_frames, 512)


def test_landmark_encoder_output_shape_projected():
    encoder = LandmarkEncoder(output_dim=768)
    patches = torch.randn(1, NUM_LRLPS, 4, 32, 32)
    coords = torch.randn(1, NUM_LRLPS, 2, 4)

    output = encoder(patches, coords)

    assert output.shape == (1, 4, 768)


def test_landmark_encoder_rejects_mismatched_shapes():
    encoder = LandmarkEncoder()
    patches = torch.randn(1, NUM_LRLPS, 4, 32, 32)
    coords = torch.randn(1, NUM_LRLPS, 2, 5)  # T mismatch

    with pytest.raises(ValueError):
        encoder(patches, coords)
