"""Module 1 -- LRLP patch sequence extraction (Sheng et al. 2022, Section III.B).

Turns one clip's raw video frames + 68-point landmark coordinates into the
two parallel per-landmark sequences the local stream's later modules need:
a 32x32 grayscale pixel patch sequence (consumed by LMFE, Module 2) and the
matching raw (unaligned) (x, y) coordinate sequence (consumed by LCFE,
Module 3, which performs the nose-tip alignment itself -- see
``align_to_nose_tip`` below).

This module operates on the WIDE raw video frame (GRID's 360x288, LRS3's
224x224) -- never the tight 96x96 appearance-encoder mouth crop, which uses
a different, incompatible coordinate space.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from fusion_avsr.utils.logging import get_logger

logger = get_logger(__name__)

PATCH_SIZE = 32

# The 38 Lip Reading related Landmark Points (LRLPs), as a fixed,
# reproducible index list into the standard iBUG 68-point layout. Fixed
# order: mouth, then jaw, then nose -- this order also defines the
# landmark index (0..37) that Module 4's semantic embedding looks up, so
# it must never be reordered once training has started against it.
#
#   Mouth: 48-67 (20 points, all of it)
#   Jaw:   2-14  (13 points -- the jaw contour minus the 2 points nearest
#          each ear, which sit further from the mouth than the rest of
#          the jaw)
#   Nose:  31-35 (5 points -- the horizontal nostril-base row only, NOT
#          the vertical nose bridge 27-30)
LRLP_INDICES: Tuple[int, ...] = tuple(range(48, 68)) + tuple(range(2, 15)) + tuple(range(31, 36))
NUM_LRLPS = len(LRLP_INDICES)
assert NUM_LRLPS == 38, f"expected 38 LRLPs, got {NUM_LRLPS}"

# The datum point used for coordinate alignment (Module 3/LCFE). Per the
# paper: "the tip of nose (one of the 68 facial landmark points) is
# selected to be datum point" -- a single fixed reference point, kept
# distinct from the 38 LRLPs above (it is not one of them). In the
# standard iBUG 68-point layout the nose is annotated as a vertical
# bridge (27-30) followed by a horizontal nostril-base row (31-35); index
# 30 -- the lowest point of the bridge, where the nose actually
# protrudes -- is the point conventionally referred to as "the nose tip".
NOSE_TIP_INDEX = 30


def _fill_missing_landmarks(
    landmarks: Sequence[Optional[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward/back-fill frames with no face detection.

    A clip's landmark ``.pkl`` file is a list of per-frame ``(68, 2)``
    arrays, but some frames may be ``None`` (no face detected). Patch
    extraction and coordinate alignment both need a real position for
    every frame, so missing frames are filled with the nearest earlier
    valid frame's landmarks (or the nearest later one, for missing frames
    before the first detection).
    Args:
        landmarks: One clip's landmark list, length T, each entry either
            a ``(68, 2)`` float array or ``None``.

    Returns:
        A tuple ``(filled, valid_mask)``: ``filled`` is a ``(T, 68, 2)``
        float32 array with no ``None`` entries; ``valid_mask`` is a
        ``(T,)`` bool array, True where the original frame had a real
        detection.

    Raises:
        ValueError: If every frame in ``landmarks`` is ``None`` (nothing
            to forward/back-fill from).
    """
    num_frames = len(landmarks)
    valid_mask = np.array([lm is not None for lm in landmarks], dtype=bool)
    if not valid_mask.any():
        message = "All frames have landmarks=None; nothing to fill from."
        logger.error(message)
        raise ValueError(message)

    filled = np.empty((num_frames, 68, 2), dtype=np.float32)
    last_valid: Optional[np.ndarray] = None
    for t in range(num_frames):
        if landmarks[t] is not None:
            last_valid = np.asarray(landmarks[t], dtype=np.float32)
        filled[t] = last_valid if last_valid is not None else np.nan

    # Back-fill any leading run of missing frames (before the first
    # detection) with the first valid frame's landmarks.
    if not valid_mask[0]:
        first_valid_idx = int(np.argmax(valid_mask))
        filled[:first_valid_idx] = filled[first_valid_idx]

    num_filled = num_frames - int(valid_mask.sum())
    if num_filled:
        logger.warning(
            "Forward/back-filled %d/%d frame(s) with no face detection",
            num_filled, num_frames,
        )
    return filled, valid_mask


def _crop_patch(frame_gray: np.ndarray, center_xy: np.ndarray, patch_size: int) -> np.ndarray:
    """Crop a single ``patch_size x patch_size`` grayscale patch centered on a point.

    The crop is zero-padded where the requested window would fall outside
    the frame (e.g. a landmark near the image border), so the returned
    patch is always exactly ``(patch_size, patch_size)``.

    Args:
        frame_gray: A single grayscale video frame, shape ``(H, W)``.
        center_xy: The ``(x, y)`` pixel coordinate to center the patch on.
        patch_size: Side length of the square patch, in pixels.

    Returns:
        A ``(patch_size, patch_size)`` array with the same dtype as
        ``frame_gray``.
    """
    height, width = frame_gray.shape
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    half = patch_size // 2

    x0, x1 = cx - half, cx - half + patch_size
    y0, y1 = cy - half, cy - half + patch_size

    src_x0, src_x1 = max(x0, 0), min(x1, width)
    src_y0, src_y1 = max(y0, 0), min(y1, height)

    patch = np.zeros((patch_size, patch_size), dtype=frame_gray.dtype)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
        patch[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = (
            frame_gray[src_y0:src_y1, src_x0:src_x1]
        )
    return patch


def extract_lrlp_sequence(
    video_frames: np.ndarray,
    landmarks: Sequence[Optional[np.ndarray]],
    patch_size: int = PATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the LRLP patch and raw-coordinate sequences for one clip.

    For each of the 38 LRLP indices (see ``LRLP_INDICES``) and each
    frame, crops a ``patch_size x patch_size`` grayscale patch centered on
    that landmark's pixel position, and records the same landmark's raw
    ``(x, y)`` pixel coordinate. Frames with no face detection
    (``landmarks[t] is None``) are forward/back-filled first (see
    ``_fill_missing_landmarks``).

    Coordinates are returned RAW (unaligned) -- nose-tip-relative
    alignment is LCFE's (Module 3's) job, via ``align_to_nose_tip``, kept
    separate so this function's output is reusable for either LMFE or
    LCFE input prep.

    Args:
        video_frames: The clip's raw decoded frames, shape ``(T, H, W)``
            (already grayscale) or ``(T, H, W, C)`` (RGB, converted to
            grayscale here via the standard luma weighting).
        landmarks: The clip's landmark list, length T, each entry a
            ``(68, 2)`` array or ``None``. Must be in the same coordinate
            space as ``video_frames`` (the raw, wide frame -- never the
            96x96 appearance-encoder crop).
        patch_size: Side length of each square patch. Defaults to 32, per
            the paper.

    Returns:
        A tuple ``(patches, raw_coords, valid_mask)``:
            - ``patches``: ``(38, T, patch_size, patch_size)`` uint8
              array (or the input dtype, if not uint8).
            - ``raw_coords``: ``(38, T, 2)`` float32 array of raw pixel
              coordinates, NOT nose-tip aligned.
            - ``valid_mask``: ``(T,)`` bool array, True where the frame
              had a real face detection (see ``_fill_missing_landmarks``).

    Raises:
        ValueError: If ``len(video_frames) != len(landmarks)``.
    """
    if len(video_frames) != len(landmarks):
        message = (
            f"video_frames has {len(video_frames)} frames but landmarks has "
            f"{len(landmarks)} -- they must be frame-aligned before calling this."
        )
        logger.error(message)
        raise ValueError(message)

    if video_frames.ndim == 4:
        # Standard luma weighting (ITU-R BT.601), matching
        # torchvision.transforms.Grayscale's default behavior.
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        frames_gray = (video_frames.astype(np.float32) @ weights).astype(video_frames.dtype)
    else:
        frames_gray = video_frames

    filled_landmarks, valid_mask = _fill_missing_landmarks(landmarks)
    num_frames = len(video_frames)

    patches = np.zeros((NUM_LRLPS, num_frames, patch_size, patch_size), dtype=frames_gray.dtype)
    raw_coords = np.empty((NUM_LRLPS, num_frames, 2), dtype=np.float32)

    for k, lrlp_index in enumerate(LRLP_INDICES):
        for t in range(num_frames):
            center_xy = filled_landmarks[t, lrlp_index]
            patches[k, t] = _crop_patch(frames_gray[t], center_xy, patch_size)
            raw_coords[k, t] = center_xy

    return patches, raw_coords, valid_mask


def align_to_nose_tip(
    raw_coords: np.ndarray,
    landmarks: Sequence[Optional[np.ndarray]],
) -> np.ndarray:
    """Align LRLP coordinates relative to the nose tip (Module 3/LCFE's alignment step).

    Subtracts the nose tip's ``(x, y)`` position (``NOSE_TIP_INDEX``,
    forward/back-filled the same way as ``extract_lrlp_sequence``) from
    every LRLP coordinate, frame by frame. This is done as a separate
    step from ``extract_lrlp_sequence`` so its raw output stays reusable
    for both LMFE (which never needs alignment) and LCFE.

    Args:
        raw_coords: ``(38, T, 2)`` raw pixel coordinates, as returned by
            ``extract_lrlp_sequence``.
        landmarks: The same clip's landmark list passed to
            ``extract_lrlp_sequence`` (needed again here to recover the
            nose tip's own position, which is not one of the 38 LRLPs).

    Returns:
        A ``(38, T, 2)`` float32 array of nose-tip-relative coordinates.

    Raises:
        ValueError: If ``raw_coords.shape[1] != len(landmarks)``.
    """
    if raw_coords.shape[1] != len(landmarks):
        message = (
            f"raw_coords has T={raw_coords.shape[1]} but landmarks has "
            f"{len(landmarks)} frames -- they must match."
        )
        logger.error(message)
        raise ValueError(message)

    filled_landmarks, _ = _fill_missing_landmarks(landmarks)
    nose_tip = filled_landmarks[:, NOSE_TIP_INDEX]  # (T, 2)
    return raw_coords - nose_tip[np.newaxis, :, :]
