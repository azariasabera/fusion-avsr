"""Dataset-level pixel mean/variance normalization for landmark-pretrain.

Per Sheng et al. 2022: "The videos are converted to grayscale and all
frames are normalized with respect to the overall mean and variance of
all videos" -- a single (mean, std) pair computed once over the training
set's pixel values, not recomputed per clip. This module computes that
pair (streaming, so the whole dataset is never held in memory at once)
and caches it to a small JSON file, and applies it to grayscale frames
before Module 1's patch extraction.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from fusion_avsr.utils.logging import get_logger
from fusion_avsr.utils.video import try_decode_frame

logger = get_logger(__name__)

PathLike = Union[str, Path]
DEFAULT_SEED = 42


def compute_dataset_pixel_stats(
    video_paths: Sequence[PathLike],
    frames_per_video: int = 10,
    seed: Optional[int] = DEFAULT_SEED,
) -> Tuple[float, float]:
    """Compute the (mean, std) of grayscale pixel values over a set of videos.

    Streams a fixed number of randomly sampled frames per video (rather
    than every frame of every video) so this stays cheap even over tens
    of thousands of clips, using Welford's online algorithm so the whole
    sample never needs to be held in memory at once.

    Args:
        video_paths: Paths to every training-set video to include in the
            statistics (conventionally, every clip in the training
            manifest -- computed once over the training set, per the
            paper, not the val/test sets).
        frames_per_video: Number of frames to randomly sample from each
            video. Defaults to 10.
        seed: Random seed controlling which frames are sampled from each
            video. Defaults to ``DEFAULT_SEED`` (42); pass ``None`` for a
            genuinely unseeded (non-reproducible) run.

    Returns:
        A ``(mean, std)`` tuple of Python floats, over the ``[0, 255]``
        grayscale pixel value range.
    """
    from torchcodec.decoders import VideoDecoder

    rng = random.Random(seed)
    luma_weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)

    count = 0
    mean = 0.0
    m2 = 0.0  # sum of squared differences from the running mean (Welford).

    for video_path in video_paths:
        decoder = VideoDecoder(str(video_path), dimension_order="NHWC")
        num_frames = len(decoder)
        if num_frames == 0:
            continue
        sample_size = min(frames_per_video, num_frames)
        frame_indices = rng.sample(range(num_frames), k=sample_size)

        for frame_index in frame_indices:
            frame = try_decode_frame(decoder, frame_index)
            if frame is None:
                continue
            frame = frame.numpy().astype(np.float64)
            if frame.ndim == 3:
                frame = frame @ luma_weights
            for pixel_value in frame.ravel():
                count += 1
                delta = pixel_value - mean
                mean += delta / count
                m2 += delta * (pixel_value - mean)

    if count == 0:
        message = "No frames sampled across the given video_paths -- cannot compute pixel stats."
        logger.error(message)
        raise ValueError(message)

    variance = m2 / count
    std = float(np.sqrt(variance))
    logger.info(
        "Computed pixel stats over %d sampled frames from %d videos: mean=%.4f std=%.4f",
        count, len(video_paths), mean, std,
    )
    return float(mean), std


def load_or_compute_pixel_stats(
    stats_path: PathLike,
    video_paths: Sequence[PathLike],
    force_recompute: bool = False,
    **kwargs: object,
) -> Tuple[float, float]:
    """Load cached (mean, std) pixel stats, computing and caching them on first use.

    Mirrors ``fusion_avsr.data.manifest_builder.load_or_build_manifest``'s
    build-once-then-cache pattern: the first call for a given
    ``stats_path`` computes the stats via ``compute_dataset_pixel_stats``
    and saves them; every later call just loads the cached JSON file.

    Args:
        stats_path: Path to the cached stats JSON file.
        video_paths: Forwarded to ``compute_dataset_pixel_stats`` if the
            stats need to be computed.
        force_recompute: If True, recompute and overwrite the cached
            stats even if they already exist.
        **kwargs: Forwarded to ``compute_dataset_pixel_stats`` (e.g.
            ``frames_per_video``, ``seed``).

    Returns:
        A ``(mean, std)`` tuple.
    """
    stats_path = Path(stats_path)
    if stats_path.exists() and not force_recompute:
        logger.info("Loading cached pixel stats from %s", stats_path)
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return stats["mean"], stats["std"]

    mean, std = compute_dataset_pixel_stats(video_paths, **kwargs)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"mean": mean, "std": std}, f)
    return mean, std


def normalize_frames(frames: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Normalize grayscale frames (or patches) to zero mean, unit variance.

    Args:
        frames: Grayscale pixel array of any shape (a full clip's frames,
            or Module 1's extracted patch tensor).
        mean: Dataset-level pixel mean, as returned by
            ``compute_dataset_pixel_stats``.
        std: Dataset-level pixel std, as returned by
            ``compute_dataset_pixel_stats``.

    Returns:
        A float32 array of the same shape as ``frames``, normalized.
    """
    return (frames.astype(np.float32) - mean) / std
