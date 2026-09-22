"""Unit tests for fusion_avsr.models.landmark.ms_tcn."""

import torch

from fusion_avsr.models.landmark.ms_tcn import MSTCN


def test_ms_tcn_output_shape_and_channels_preserved():
    channels, num_frames = 32, 10
    model = MSTCN(channels=channels)
    x = torch.randn(2, channels, num_frames)

    output = model(x)

    assert output.shape == x.shape


def test_ms_tcn_has_nine_layers():
    model = MSTCN(channels=16)
    # 3 blocks x 3 layers each, per Fig. 4's kernel/dilation schedule.
    assert len(model.layers) == 9
