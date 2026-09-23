"""Shared TorchCodec frame-decoding helpers.

``VideoDecoder.__len__()`` can overclaim relative to what TorchCodec can
actually decode -- ``decoder[i]`` for some ``i < len(decoder)`` can still
raise ``RuntimeError: Requested next frame while there are no more
frames left to decode``. Every caller that decodes frames by index
should go through the helpers here instead of trusting ``len(decoder)``
as a loop bound directly, so this quirk is only handled in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Union

import numpy as np
import pandas as pd

from fusion_avsr.utils.logging import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


def iter_decodable_frames(decoder) -> Iterator[object]:
    """Yield ``decoder[0], decoder[1], ...``, stopping cleanly at the first index that fails to decode.

    Two distinct failure modes, both meaning "no more real frames here":
        - IndexError: asking for an index at/past len(decoder)
        - RuntimeError: a real decode failure INSIDE the claimed length
    Both are treated identically: stop collecting, keep what decoded.
    
    Args:
        decoder: A ``torchcodec.decoders.VideoDecoder``.

    Yields:
        Each successfully decoded frame, in order, starting from index 0.
    """
    i = 0
    while True:
        try:
            yield decoder[i]
            i += 1
        except (RuntimeError, IndexError):
            return


def decode_all_frames(decoder) -> np.ndarray:
    """Decode every frame TorchCodec will actually give us, as one stacked array.

    Stops silently at the first index that fails to decode, rather than
    raising -- the returned array may have fewer frames than
    ``len(decoder)`` claims. Trusts whatever TorchCodec actually managed
    to decode; only fails if that's zero frames.

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
        except (RuntimeError, IndexError):
            break
    if not frames:
        raise RuntimeError(f"No frames could be decoded in range [{start}, {end})")
    return np.stack(frames)


def compute_real_decodable_frame_counts(
    manifest: pd.DataFrame,
    log_path: Optional[PathLike] = None,
    _open_decoder=None,
) -> Dict[str, int]:
    """Decode every clip in ``manifest`` once and record its real, 
    actually-decodable frame count.

    Args:
        manifest: A per-clip manifest-shaped DataFrame with ``sample_id``
            and ``video_path`` columns.
        log_path: Optional path to an append-only log file. Clips that
            fail to open/decode at all are logged here and given a count
            of 0, rather than crashing this whole batch pass over one
            bad file.
        _open_decoder: Test-only injection point: a ``video_path ->
            decoder`` callable, defaulting to a real
            ``torchcodec.decoders.VideoDecoder``. Lets this function's
            per-clip loop be unit tested with a fake decoder, without
            needing torchcodec installed.

    Returns:
        A dict mapping each ``sample_id`` to its real decodable frame
        count.
    """
    if _open_decoder is None:
        from torchcodec.decoders import VideoDecoder
        _open_decoder = lambda video_path: VideoDecoder(video_path, dimension_order="NHWC")

    counts: Dict[str, int] = {}
    failed_lines = []
    for row in manifest.itertuples():
        try:
            decoder = _open_decoder(row.video_path)
            counts[row.sample_id] = sum(1 for _ in iter_decodable_frames(decoder))
        except Exception as e:
            counts[row.sample_id] = 0
            failed_lines.append(f"{row.sample_id}: failed to decode {row.video_path} ({e})\n")

    if failed_lines:
        logger.warning("compute_real_decodable_frame_counts: %d clip(s) failed to decode", len(failed_lines))
        if log_path is not None:
            with open(log_path, "a") as f:
                f.writelines(failed_lines)

    return counts


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
    except (RuntimeError, IndexError):
        return None
