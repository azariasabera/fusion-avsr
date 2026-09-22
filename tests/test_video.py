"""Unit tests for fusion_avsr.utils.video.

Uses a fake decoder that reproduces TorchCodec's len()-overclaim quirk
(reports more frames than it can actually decode) -- no real video file
or torchcodec install needed.
"""

import numpy as np
import pytest

from fusion_avsr.utils.video import decode_all_frames, decode_frame_range, try_decode_frame


class _FakeFrame:
    def __init__(self, value):
        self._value = value

    def numpy(self):
        return np.full((2, 2), self._value, dtype=np.uint8)


class _FakeDecoder:
    """Claims `length` frames via __len__ but only decodes indices < decodable."""

    def __init__(self, length, decodable):
        self._length = length
        self._decodable = decodable

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index >= self._decodable:
            raise RuntimeError("Requested next frame while there are no more frames left to decode")
        return _FakeFrame(index)


def test_decode_all_frames_stops_early_without_raising():
    decoder = _FakeDecoder(length=10, decodable=4)

    frames = decode_all_frames(decoder)

    assert frames.shape[0] == 4


def test_decode_all_frames_raises_if_nothing_decodable():
    decoder = _FakeDecoder(length=5, decodable=0)

    with pytest.raises(RuntimeError):
        decode_all_frames(decoder)


def test_decode_all_frames_full_length_when_fully_decodable():
    decoder = _FakeDecoder(length=6, decodable=6)

    frames = decode_all_frames(decoder)

    assert frames.shape[0] == 6


def test_decode_frame_range_stops_early_within_range():
    decoder = _FakeDecoder(length=10, decodable=5)

    frames = decode_frame_range(decoder, start=2, end=8)

    assert frames.shape[0] == 3  # indices 2, 3, 4 decodable; 5 fails


def test_decode_frame_range_raises_if_start_itself_fails():
    decoder = _FakeDecoder(length=10, decodable=2)

    with pytest.raises(RuntimeError):
        decode_frame_range(decoder, start=5, end=8)


def test_try_decode_frame_returns_none_on_failure():
    decoder = _FakeDecoder(length=10, decodable=3)

    assert try_decode_frame(decoder, 1) is not None
    assert try_decode_frame(decoder, 3) is None
