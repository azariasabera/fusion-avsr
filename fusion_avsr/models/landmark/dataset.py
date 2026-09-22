"""GRID word-segment dataset for the landmark encoder's word-recognition pretext task.

Loads the (cached, build-once) GRID per-clip manifest and word-segment
table via ``fusion_avsr.data.manifest_builder.load_or_build_manifest``,
and for each word segment, decodes the parent clip's frame range, runs
Module 1's LRLP patch/coordinate extraction and nose-tip alignment, and
applies dataset-level pixel normalization -- producing exactly the
``(patches, aligned_coords, label)`` triple the landmark encoder /
pretraining classifier consume.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fusion_avsr.data.manifest_builder import (
    build_grid_manifest,
    build_grid_word_segments,
    filter_grid_word_segments,
    load_or_build_manifest,
)
from fusion_avsr.data.paths import MANIFEST_DIR
from fusion_avsr.models.landmark.lrlp import align_to_nose_tip, extract_lrlp_sequence
from fusion_avsr.models.landmark.normalization import normalize_frames
from fusion_avsr.utils.logging import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


def build_word_vocabulary(word_segments: pd.DataFrame) -> Dict[str, int]:
    """Build a deterministic word -> class-index mapping from a word-segment table.

    Sorted alphabetically (not by first-seen order) so the mapping is
    reproducible across runs/machines regardless of manifest row order.

    Args:
        word_segments: GRID word-segment table (see
            ``fusion_avsr.data.manifest_builder.build_grid_word_segments``).

    Returns:
        A dict mapping each unique word to a class index in
        ``[0, num_unique_words)``.
    """
    unique_words = sorted(word_segments["word"].unique())
    return {word: index for index, word in enumerate(unique_words)}


class GridWordSegmentDataset(Dataset):
    """One sample = one GRID word segment: (LRLP patches, aligned coordinates, class label)."""

    def __init__(
        self,
        grid_root: PathLike,
        landmarks_root: PathLike,
        audio_output_dir: PathLike,
        pixel_mean: float,
        pixel_std: float,
        manifest_dir: PathLike = MANIFEST_DIR,
        vocabulary: Optional[Dict[str, int]] = None,
        limit: Optional[int] = None,
        force_rebuild_manifests: bool = False,
    ) -> None:
        """Build the dataset, loading/caching GRID's manifests as needed.

        Args:
            grid_root: Path to the GRID dataset root.
            landmarks_root: Path to GRID's landmark ``.pkl`` files.
            audio_output_dir: Path ``scripts/extract_audio.sh`` wrote
                GRID's extracted ``.wav`` files to (needed by
                ``build_grid_manifest``, even though audio itself is
                unused by this pretext task).
            pixel_mean: Dataset-level grayscale pixel mean (see
                ``fusion_avsr.models.landmark.normalization``).
            pixel_std: Dataset-level grayscale pixel std.
            manifest_dir: Directory manifests are cached under. Defaults
                to ``fusion_avsr.data.paths.MANIFEST_DIR``.
            vocabulary: Word -> class-index mapping. If ``None``
                (default), built fresh from this dataset's own word
                segments via ``build_word_vocabulary`` -- pass an
                explicit vocabulary (e.g. from a train split) when
                constructing a val/test split, so class indices line up.
            limit: Forwarded to the manifest builders, for fast smoke
                testing against a handful of clips.
            force_rebuild_manifests: If True, rebuild cached manifests
                even if they already exist on disk.
        """
        manifest_dir = Path(manifest_dir)
        grid_manifest = load_or_build_manifest(
            manifest_dir / "grid_manifest.csv",
            build_grid_manifest,
            force_rebuild=force_rebuild_manifests,
            grid_root=grid_root,
            landmarks_root=landmarks_root,
            audio_output_dir=audio_output_dir,
            limit=limit,
        )
        word_segments = load_or_build_manifest(
            manifest_dir / "grid_word_segments.csv",
            build_grid_word_segments,
            force_rebuild=force_rebuild_manifests,
            grid_root=grid_root,
            limit=limit,
        )
        word_segments = filter_grid_word_segments(word_segments, grid_manifest)

        self._clip_paths: Dict[str, Tuple[str, str]] = {
            row.sample_id: (row.video_path, row.landmark_path) for row in grid_manifest.itertuples()
        }
        self.word_segments = word_segments.reset_index(drop=True)
        self.vocabulary = vocabulary if vocabulary is not None else build_word_vocabulary(self.word_segments)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        logger.info(
            "GridWordSegmentDataset: %d word segments, %d-word vocabulary",
            len(self.word_segments), len(self.vocabulary),
        )

    def __len__(self) -> int:
        return len(self.word_segments)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return one word segment's ``(patches, aligned_coords, label)``.

        Args:
            index: Row index into ``self.word_segments``.

        Returns:
            A tuple ``(patches, aligned_coords, label)``:
                - ``patches``: ``(38, T, 32, 32)`` float tensor,
                  pixel-normalized.
                - ``aligned_coords``: ``(38, 2, T)`` float tensor,
                  nose-tip-relative.
                - ``label``: class index into ``self.vocabulary``.
        """
        from torchcodec.decoders import VideoDecoder

        row = self.word_segments.iloc[index]
        video_path, landmark_path = self._clip_paths[row["sample_id"]]

        with open(landmark_path, "rb") as f:
            clip_landmarks = pickle.load(f)

        decoder = VideoDecoder(video_path, dimension_order="NHWC")
        end_frame = min(int(row["end_frame"]), len(decoder), len(clip_landmarks))
        start_frame = min(int(row["start_frame"]), end_frame)

        frames = np.stack([decoder[t].numpy() for t in range(start_frame, end_frame)])
        landmarks_segment = clip_landmarks[start_frame:end_frame]

        patches, raw_coords, _valid_mask = extract_lrlp_sequence(frames, landmarks_segment)
        aligned_coords = align_to_nose_tip(raw_coords, landmarks_segment)  # (K, T, 2)

        patches = normalize_frames(patches, self.pixel_mean, self.pixel_std)

        patches_tensor = torch.from_numpy(patches).float()
        coords_tensor = torch.from_numpy(aligned_coords).float().permute(0, 2, 1)  # (K, 2, T)
        label = self.vocabulary[row["word"]]
        return patches_tensor, coords_tensor, label


def collate_word_segments(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length word segments into a zero-padded batch.

    Word segments have different durations, so ``T`` varies per sample.
    This pads every sample's patch/coordinate sequence to the batch's max
    ``T`` with zeros, and returns a boolean validity mask so padded
    frames can be excluded from pooling (see
    ``fusion_avsr.models.landmark.pretrain.masked_mean_pool``).

    Args:
        batch: A list of ``(patches, aligned_coords, label)`` tuples, as
            returned by ``GridWordSegmentDataset.__getitem__``.

    Returns:
        A tuple ``(patches, aligned_coords, labels, mask)``:
            - ``patches``: ``(B, K, T_max, 32, 32)`` float tensor.
            - ``aligned_coords``: ``(B, K, 2, T_max)`` float tensor.
            - ``labels``: ``(B,)`` long tensor.
            - ``mask``: ``(B, T_max)`` bool tensor, True for real
              (non-padded) frames.
    """
    max_frames = max(patches.shape[1] for patches, _, _ in batch)
    num_landmarks, patch_size = batch[0][0].shape[0], batch[0][0].shape[-1]

    padded_patches = torch.zeros(len(batch), num_landmarks, max_frames, patch_size, patch_size)
    padded_coords = torch.zeros(len(batch), num_landmarks, 2, max_frames)
    mask = torch.zeros(len(batch), max_frames, dtype=torch.bool)
    labels = torch.zeros(len(batch), dtype=torch.long)

    for i, (patches, coords, label) in enumerate(batch):
        num_frames = patches.shape[1]
        padded_patches[i, :, :num_frames] = patches
        padded_coords[i, :, :, :num_frames] = coords
        mask[i, :num_frames] = True
        labels[i] = label

    return padded_patches, padded_coords, labels, mask
