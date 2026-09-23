"""GRID word-recognition pretraining for the landmark encoder (Modules 1-5).

Trains ``fusion_avsr.models.landmark.pretrain.GridWordRecognitionModel``
(LandmarkEncoder + MS-TCN + 51-word classification head) on GRID's
word-level segments, as a warm start before the landmark encoder joins
main LRS3 fusion training (where it is NOT frozen). After pretraining,
only the encoder's weights are meant to carry forward -- this script
saves the full model checkpoint; downstream code should load just its
``model.encoder.state_dict()``.

Configured via Hydra -- see configs/scripts/pretrain_landmark_grid.yaml
for all available fields. Example invocation:

    python scripts/pretrain_landmark_grid.py \\
        grid_root=/scratch/your_project_name/datasets/kaggle_lipnet/.../data \\
        landmarks_root=/scratch/your_project_name/datasets/grid_landmarks \\
        audio_output_dir=/scratch/your_project_name/datasets/grid_audio
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fusion_avsr.data.manifest_builder import build_grid_manifest, load_or_build_manifest  # noqa: E402
from fusion_avsr.data.paths import MANIFEST_DIR  # noqa: E402
from fusion_avsr.models.landmark.dataset import (  # noqa: E402
    GridWordSegmentDataset,
    collate_word_segments,
)
from fusion_avsr.models.landmark.normalization import load_or_compute_pixel_stats  # noqa: E402
from fusion_avsr.models.landmark.pretrain import GridWordRecognitionModel, evaluate, train_one_epoch  # noqa: E402
from fusion_avsr.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def _split_indices_by_speaker(
    word_segments: pd.DataFrame,
    val_speaker_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Split word-segment row indices into train/val, holding out whole SPEAKERS.

    ``sample_id`` is ``<speaker_id>_<clip_id>``. Splitting at the clip
    level would still let val share a speaker's voice/appearance with
    train; holding out entire speakers instead keeps every word segment
    from a held-out speaker's clips entirely on the val side.

    The number of speakers held out is a FRACTION of however many unique
    speakers are actually present (rounded).

    Args:
        word_segments: The (already clip-filtered) GRID word-segment
            table.
        val_speaker_fraction: Fraction of unique speakers to hold out for
            validation (e.g. 0.1 -> ~10%, rounded).
        seed: Random seed controlling which speakers are held out.

    Returns:
        A tuple ``(train_indices, val_indices)`` of row indices into
        ``word_segments``.

    Raises:
        ValueError: If fewer than 2 unique speakers remain after
            filtering -- there is nothing meaningful to split (should
            only trigger on a pathologically small ``limit``, e.g. 1).
    """
    speaker_ids = word_segments["sample_id"].str.split("_", n=1).str[0]
    unique_speakers = sorted(speaker_ids.unique())

    if len(unique_speakers) < 2:
        message = (
            f"Need at least 2 unique speakers to split train/val, found "
            f"{len(unique_speakers)} -- check that `limit` isn't cutting the dataset "
            f"down to a single speaker."
        )
        logger.error(message)
        raise ValueError(message)

    num_val_speakers = max(1, min(len(unique_speakers) - 1, round(val_speaker_fraction * len(unique_speakers))))

    rng = random.Random(seed)
    rng.shuffle(unique_speakers)
    val_speakers = set(unique_speakers[:num_val_speakers])

    is_val = speaker_ids.isin(val_speakers)
    val_indices = word_segments.index[is_val].tolist()
    train_indices = word_segments.index[~is_val].tolist()
    return train_indices, val_indices


@hydra.main(version_base=None, config_path="../configs/scripts", config_name="pretrain_landmark_grid")
def main(cfg: DictConfig) -> None:
    if cfg.grid_root is None or cfg.landmarks_root is None or cfg.audio_output_dir is None:
        raise ValueError("grid_root, landmarks_root, and audio_output_dir must all be set")

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or "cpu" in cfg.device else "cpu")

    grid_manifest = load_or_build_manifest(
        MANIFEST_DIR / "grid_manifest.csv",
        build_grid_manifest,
        force_rebuild=cfg.force_rebuild_manifest,
        grid_root=cfg.grid_root,
        landmarks_root=cfg.landmarks_root,
        audio_output_dir=cfg.audio_output_dir,
        limit=cfg.limit,
    )

    pixel_mean, pixel_std = load_or_compute_pixel_stats(
        MANIFEST_DIR / "grid_pixel_stats.json",
        video_paths=grid_manifest["video_path"].tolist(),
        frames_per_video=cfg.pixel_stats_frames_per_video,
        seed=cfg.seed,
    )
    logger.info("Using pixel stats: mean=%.4f std=%.4f", pixel_mean, pixel_std)

    full_dataset = GridWordSegmentDataset(
        grid_root=cfg.grid_root,
        landmarks_root=cfg.landmarks_root,
        audio_output_dir=cfg.audio_output_dir,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        limit=cfg.limit,
        force_rebuild_manifests=cfg.force_rebuild_manifest,
    )
    train_indices, val_indices = _split_indices_by_speaker(full_dataset.word_segments, cfg.val_speaker_fraction, cfg.seed)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    logger.info("Train: %d word segments, Val: %d word segments", len(train_dataset), len(val_dataset))

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_word_segments,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_word_segments,
    )

    model = GridWordRecognitionModel(num_classes=len(full_dataset.vocabulary)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    for epoch in range(cfg.num_epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d/%d: train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1, cfg.num_epochs,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
        )
        scheduler.step(val_metrics["loss"])

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            epochs_without_improvement = 0
            checkpoint_path = checkpoint_dir / "best.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocabulary": full_dataset.vocabulary,
                    "pixel_mean": pixel_mean,
                    "pixel_std": pixel_std,
                    "val_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )
            logger.info("New best val_acc=%.4f -- saved checkpoint to %s", best_val_accuracy, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                logger.info(
                    "No val-accuracy improvement for %d epoch(s) (early_stopping_patience=%d) -- "
                    "stopping early after epoch %d/%d",
                    epochs_without_improvement, cfg.early_stopping_patience, epoch + 1, cfg.num_epochs,
                )
                break


if __name__ == "__main__":
    main()
