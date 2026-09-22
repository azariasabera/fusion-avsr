"""Shared TorchCodec frame-decoding helpers.

``VideoDecoder.__len__()`` can overclaim relative to what TorchCodec can
actually decode -- ``decoder[i]`` for some ``i < len(decoder)`` can still
raise ``RuntimeError: Requested next frame while there are no more
frames left to decode``. Every caller that decodes frames by index
should go through the helpers here instead of trusting ``len(decoder)``
as a loop bound directly, so this quirk is only handled in one place.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def iter_decodable_frames(decoder) -> Iterator[object]:
    """Yield ``decoder[0], decoder[1], ...``, stopping cleanly at the first index that fails to decode.

    Args:
        decoder: A ``torchcodec.decoders.VideoDecoder``.

    Yields:
        Each successfully decoded frame, in order, starting from index 0.
    """
    i = 0
    while True:
        try:
            yield decoder[i]
        except RuntimeError:
            return
        i += 1


def decode_all_frames(decoder) -> np.ndarray:
    """Decode every frame TorchCodec will actually give us, as one stacked array.

    Stops silently at the first index that fails to decode, rather than
    raising -- the returned array may have fewer frames than
    ``len(decoder)`` claims. Callers that need to reject a partially
    decoded clip should compare the result's length against
    ``len(decoder)`` themselves (see
    ``scripts/extract_landmarks_grid.py``'s ``_read_video_via_torchcodec``
    for an example that does).

    Args:
        decoder: A ``torchcodec.decoders.VideoDecoder``.

    Returns:
        A ``(T, H, W, C)`` array, ``T <= len(decoder)``.

    Raises:
        RuntimeError: If not even the first frame could be decoded.
    """
    frames = [frame.numpy() for frame in iter_decodable_frames(decoder)]
    if not frames:
        raise RuntimeError("No frames could be decoded")
    return np.stack(frames)


def decode_frame_range(decoder, start: int, end: int) -> np.ndarray:
    """Decode frames ``[start, end)``, stopping early if decoding fails before reaching ``end``.

    Args:
        decoder: A ``torchcodec.decoders.VideoDecoder``.
        start: First frame index to decode (inclusive).
        end: Frame index to stop before (exclusive).

    Returns:
        A ``(T, H, W, C)`` array, ``T <= end - start``.

    Raises:
        RuntimeError: If not even the first frame in the range could be
            decoded.
    """
    frames = []
    for i in range(start, end):
        try:
            frames.append(decoder[i].numpy())
        except RuntimeError:
            break
    if not frames:
        raise RuntimeError(f"No frames could be decoded in range [{start}, {end})")
    return np.stack(frames)


def try_decode_frame(decoder, index: int) -> object:
    """Decode a single frame by index, returning ``None`` instead of raising if it fails.

    Args:
        decoder: A ``torchcodec.decoders.VideoDecoder``.
        index: Frame index to decode.

    Returns:
        The decoded frame, or ``None`` if that index could not be
        decoded (e.g. ``index < len(decoder)`` but past what TorchCodec
        can actually decode).
    """
    try:
        return decoder[index]
    except RuntimeError:
        return None
