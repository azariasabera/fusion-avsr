"""Unit tests for fusion_avsr.models.landmark.lcfe (Module 3)."""

import torch

from fusion_avsr.models.landmark.lcfe import LCFE, LCFE_OUTPUT_CHANNELS
from fusion_avsr.models.landmark.lrlp import NUM_LRLPS


def test_lcfe_output_shape():
    batch_size, num_frames = 2, 6
    lcfe = LCFE()
    coords = torch.randn(batch_size, NUM_LRLPS, 2, num_frames)

    output = lcfe(coords)

    assert output.shape == (batch_size, NUM_LRLPS, LCFE_OUTPUT_CHANNELS, num_frames)


def test_lcfe_preserves_temporal_length_for_various_t():
    lcfe = LCFE()
    for num_frames in (1, 3, 10):
        coords = torch.randn(1, NUM_LRLPS, 2, num_frames)
        output = lcfe(coords)
        assert output.shape[-1] == num_frames


def test_lcfe_matches_lmfe_output_channels_by_default():
    from fusion_avsr.models.landmark.lmfe import LMFE_OUTPUT_CHANNELS

    assert LCFE_OUTPUT_CHANNELS == LMFE_OUTPUT_CHANNELS
