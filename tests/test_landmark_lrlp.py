"""Unit tests for fusion_avsr.models.landmark.lrlp (Module 1).

Uses small synthetic frame/landmark arrays -- no real video or .pkl files
needed. Landmark positions are placed at known pixel coordinates so patch
extraction and nose-tip alignment can be checked exactly, not just by
shape.
"""

import numpy as np
import pytest

from fusion_avsr.models.landmark.lrlp import (
    LRLP_INDICES,
    NOSE_TIP_INDEX,
    NUM_LRLPS,
    PATCH_SIZE,
    _crop_patch,
    _fill_missing_landmarks,
    align_to_nose_tip,
    extract_lrlp_sequence,
)


def _make_landmarks(num_frames: int, base_xy=(50.0, 60.0), drift=(0.0, 0.0)):
    """68-point landmark list where point i sits at (base_x + i, base_y + i), drifting per frame."""
    landmarks = []
    for t in range(num_frames):
        pts = np.array(
            [[base_xy[0] + i + drift[0] * t, base_xy[1] + i + drift[1] * t] for i in range(68)],
            dtype=np.float32,
        )
        landmarks.append(pts)
    return landmarks


# ---------------------------------------------------------------------------
# LRLP_INDICES / constants
# ---------------------------------------------------------------------------

def test_lrlp_indices_count_and_composition():
    assert NUM_LRLPS == 38
    assert len(LRLP_INDICES) == 38
    assert len(set(LRLP_INDICES)) == 38  # no duplicates
    assert set(range(48, 68)).issubset(LRLP_INDICES)  # mouth
    assert set(range(2, 15)).issubset(LRLP_INDICES)  # jaw
    assert set(range(31, 36)).issubset(LRLP_INDICES)  # nose row


def test_nose_tip_index_not_in_lrlps():
    assert NOSE_TIP_INDEX not in LRLP_INDICES


# ---------------------------------------------------------------------------
# _fill_missing_landmarks
# ---------------------------------------------------------------------------

def test_fill_missing_landmarks_no_gaps_returns_unchanged():
    landmarks = _make_landmarks(3)
    filled, valid_mask = _fill_missing_landmarks(landmarks)

    assert filled.shape == (3, 68, 2)
    assert valid_mask.all()
    np.testing.assert_array_equal(filled[0], landmarks[0])


def test_fill_missing_landmarks_forward_fills_middle_gap():
    landmarks = _make_landmarks(3)
    landmarks[1] = None

    filled, valid_mask = _fill_missing_landmarks(landmarks)

    assert list(valid_mask) == [True, False, True]
    np.testing.assert_array_equal(filled[1], landmarks[0])  # forward-filled from frame 0


def test_fill_missing_landmarks_back_fills_leading_gap():
    landmarks = _make_landmarks(3)
    landmarks[0] = None

    filled, valid_mask = _fill_missing_landmarks(landmarks)

    assert list(valid_mask) == [False, True, True]
    np.testing.assert_array_equal(filled[0], landmarks[1])  # back-filled from frame 1


def test_fill_missing_landmarks_all_none_raises():
    with pytest.raises(ValueError):
        _fill_missing_landmarks([None, None])


# ---------------------------------------------------------------------------
# _crop_patch
# ---------------------------------------------------------------------------

def test_crop_patch_centered_interior_point():
    frame = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100) % 256
    patch = _crop_patch(frame, center_xy=np.array([50.0, 50.0]), patch_size=32)

    assert patch.shape == (32, 32)
    np.testing.assert_array_equal(patch, frame[34:66, 34:66])


def test_crop_patch_near_border_is_zero_padded():
    frame = np.ones((40, 40), dtype=np.uint8)
    patch = _crop_patch(frame, center_xy=np.array([1.0, 1.0]), patch_size=32)

    assert patch.shape == (32, 32)
    assert patch[0, 0] == 0  # out-of-bounds region, zero-padded
    assert patch[-1, -1] == 1  # in-bounds region, copied from the real frame


# ---------------------------------------------------------------------------
# extract_lrlp_sequence
# ---------------------------------------------------------------------------

def test_extract_lrlp_sequence_shapes():
    num_frames = 4
    frames = np.random.default_rng(0).integers(0, 255, size=(num_frames, 120, 120), dtype=np.uint8)
    landmarks = _make_landmarks(num_frames, base_xy=(60.0, 60.0))

    patches, raw_coords, valid_mask = extract_lrlp_sequence(frames, landmarks)

    assert patches.shape == (NUM_LRLPS, num_frames, PATCH_SIZE, PATCH_SIZE)
    assert raw_coords.shape == (NUM_LRLPS, num_frames, 2)
    assert valid_mask.shape == (num_frames,)
    assert valid_mask.all()


def test_extract_lrlp_sequence_raw_coords_match_landmark_positions():
    num_frames = 2
    frames = np.zeros((num_frames, 120, 120), dtype=np.uint8)
    landmarks = _make_landmarks(num_frames, base_xy=(60.0, 60.0))

    _patches, raw_coords, _valid_mask = extract_lrlp_sequence(frames, landmarks)

    for k, lrlp_index in enumerate(LRLP_INDICES):
        np.testing.assert_allclose(raw_coords[k, 0], landmarks[0][lrlp_index])


def test_extract_lrlp_sequence_accepts_rgb_input():
    num_frames = 2
    frames = np.zeros((num_frames, 120, 120, 3), dtype=np.uint8)
    landmarks = _make_landmarks(num_frames, base_xy=(60.0, 60.0))

    patches, _raw_coords, _valid_mask = extract_lrlp_sequence(frames, landmarks)

    assert patches.shape == (NUM_LRLPS, num_frames, PATCH_SIZE, PATCH_SIZE)


def test_extract_lrlp_sequence_frame_landmark_length_mismatch_raises():
    frames = np.zeros((3, 120, 120), dtype=np.uint8)
    landmarks = _make_landmarks(2)

    with pytest.raises(ValueError):
        extract_lrlp_sequence(frames, landmarks)


# ---------------------------------------------------------------------------
# align_to_nose_tip
# ---------------------------------------------------------------------------

def test_align_to_nose_tip_subtracts_nose_position():
    num_frames = 2
    frames = np.zeros((num_frames, 120, 120), dtype=np.uint8)
    landmarks = _make_landmarks(num_frames, base_xy=(60.0, 60.0))

    _patches, raw_coords, _valid_mask = extract_lrlp_sequence(frames, landmarks)
    aligned = align_to_nose_tip(raw_coords, landmarks)

    nose_tip_frame0 = landmarks[0][NOSE_TIP_INDEX]
    for k, lrlp_index in enumerate(LRLP_INDICES):
        expected = landmarks[0][lrlp_index] - nose_tip_frame0
        np.testing.assert_allclose(aligned[k, 0], expected)


def test_align_to_nose_tip_shape_mismatch_raises():
    raw_coords = np.zeros((NUM_LRLPS, 3, 2), dtype=np.float32)
    landmarks = _make_landmarks(2)

    with pytest.raises(ValueError):
        align_to_nose_tip(raw_coords, landmarks)
