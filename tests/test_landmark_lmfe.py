"""Unit tests for fusion_avsr.models.landmark.lmfe (Module 2)."""

import torch

from fusion_avsr.models.landmark.lmfe import LMFE, LMFE_OUTPUT_CHANNELS
from fusion_avsr.models.landmark.lrlp import NUM_LRLPS


def test_lmfe_output_shape():
    batch_size, num_frames = 2, 6
    lmfe = LMFE()
    patches = torch.randn(batch_size, NUM_LRLPS, num_frames, 32, 32)

    output = lmfe(patches)

    assert output.shape == (batch_size, NUM_LRLPS, LMFE_OUTPUT_CHANNELS, num_frames)


def test_lmfe_preserves_temporal_length_for_various_t():
    lmfe = LMFE()
    for num_frames in (1, 3, 10):
        patches = torch.randn(1, NUM_LRLPS, num_frames, 32, 32)
        output = lmfe(patches)
        assert output.shape[-1] == num_frames


def test_lmfe_custom_output_channels():
    lmfe = LMFE(output_channels=128)
    patches = torch.randn(1, NUM_LRLPS, 4, 32, 32)

    output = lmfe(patches)

    assert output.shape == (1, NUM_LRLPS, 128, 4)
