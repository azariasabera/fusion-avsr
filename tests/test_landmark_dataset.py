"""Unit tests for fusion_avsr.models.landmark.dataset.

GridWordSegmentDataset.__getitem__ needs a real GRID clip/landmark file
and torchcodec, so it is not exercised here (needs Roihu). The
vocabulary builder and collate function are pure and tested directly.
"""

import pandas as pd
import torch

from fusion_avsr.models.landmark.dataset import build_word_vocabulary, collate_word_segments


def test_build_word_vocabulary_is_sorted_and_deterministic():
    word_segments = pd.DataFrame({"word": ["zebra", "apple", "mango", "apple"]})

    vocabulary = build_word_vocabulary(word_segments)

    assert vocabulary == {"apple": 0, "mango": 1, "zebra": 2}


def test_collate_word_segments_pads_to_max_length():
    num_landmarks, patch_size = 38, 32
    sample_short = (
        torch.randn(num_landmarks, 3, patch_size, patch_size),
        torch.randn(num_landmarks, 2, 3),
        5,
    )
    sample_long = (
        torch.randn(num_landmarks, 7, patch_size, patch_size),
        torch.randn(num_landmarks, 2, 7),
        2,
    )

    patches, coords, labels, mask = collate_word_segments([sample_short, sample_long])

    assert patches.shape == (2, num_landmarks, 7, patch_size, patch_size)
    assert coords.shape == (2, num_landmarks, 2, 7)
    assert labels.tolist() == [5, 2]
    assert mask[0].tolist() == [True] * 3 + [False] * 4
    assert mask[1].tolist() == [True] * 7
    # Padded region of the short sample must be exactly zero.
    assert torch.all(patches[0, :, 3:] == 0)
    assert torch.all(coords[0, :, :, 3:] == 0)
