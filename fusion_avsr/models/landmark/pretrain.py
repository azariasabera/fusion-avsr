"""GRID word-recognition pretraining: model, pooling, and train/eval loops.

The landmark encoder (Modules 1-5) is pretrained via a word-recognition
pretext task on GRID's 51-word vocabulary, as a warm start before joint
training on LRS3 (GRID pretraining only initializes it, it keeps training
on the main task). After pretraining, only ``LandmarkEncoder`` (this module's 
``.encoder`` attribute) carries forward -- the MS-TCN decoder and classification 
head defined here are discarded.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from fusion_avsr.models.landmark.encoder import LandmarkEncoder
from fusion_avsr.models.landmark.ms_tcn import MSTCN
from fusion_avsr.utils.logging import get_logger

logger = get_logger(__name__)

ENCODER_OUTPUT_CHANNELS = 512  # LandmarkEncoder's un-projected output width (LMFE 256 + LCFE 256).
DEFAULT_CLASSIFIER_DROPOUT = 0.2


def masked_mean_pool(sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Average a ``(B, T, C)`` sequence over time, ignoring padded frames.

    Args:
        sequence: ``(B, T, C)`` feature sequence.
        mask: ``(B, T)`` bool tensor, True for real (non-padded) frames.
            If ``None``, every frame is treated as valid (plain mean over
            ``T``).

    Returns:
        A ``(B, C)`` tensor of per-sample pooled features.
    """
    if mask is None:
        return sequence.mean(dim=1)

    mask = mask.unsqueeze(-1).to(sequence.dtype)  # (B, T, 1)
    summed = (sequence * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class GridWordRecognitionModel(nn.Module):
    """LandmarkEncoder -> MS-TCN -> masked mean pool -> 51-word classification head."""

    def __init__(
        self,
        num_classes: int,
        encoder_channels: int = ENCODER_OUTPUT_CHANNELS,
        classifier_dropout: float = DEFAULT_CLASSIFIER_DROPOUT,
    ) -> None:
        """Build the pretext-task model.

        Args:
            num_classes: Number of GRID vocabulary words (51, with
                ``sil``/``sp`` excluded -- see
                ``fusion_avsr.data.manifest_builder.build_grid_word_segments``).
            encoder_channels: Channel width of ``LandmarkEncoder``'s
                output. Defaults to 512 (unprojected).
            classifier_dropout: Dropout probability applied to the pooled
                feature before the final linear layer. Defaults to 0.2 --
                this pretext task is trained on GRID's narrow, repetitive
                vocabulary/grammar but needs to generalize to LRS3, a much
                larger and more varied domain, so the classification head
                (discarded after pretraining, but still shaping how the
                encoder itself learns) is regularized against overfitting
                to GRID specifically.
        """
        super().__init__()
        self.encoder = LandmarkEncoder(output_dim=None)
        self.ms_tcn = MSTCN(channels=encoder_channels)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(encoder_channels, num_classes)

    def forward(
        self,
        patches: torch.Tensor,
        aligned_coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the full pretext-task model.

        Args:
            patches: ``(B, K, T, 32, 32)`` LRLP patch sequences.
            aligned_coords: ``(B, K, 2, T)`` nose-tip-aligned coordinates.
            mask: ``(B, T)`` bool tensor, True for real (non-padded)
                frames (see
                ``fusion_avsr.models.landmark.dataset.collate_word_segments``).

        Returns:
            A ``(B, num_classes)`` tensor of class logits.
        """
        encoded = self.encoder(patches, aligned_coords)  # (B, T, C)
        decoded = self.ms_tcn(encoded.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, C)
        pooled = masked_mean_pool(decoded, mask)  # (B, C)
        return self.classifier(self.dropout(pooled))


def train_one_epoch(
    model: GridWordRecognitionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch over ``dataloader``.

    Args:
        model: The pretext-task model.
        dataloader: Yields ``(patches, aligned_coords, labels, mask)``
            batches, as produced by
            ``fusion_avsr.models.landmark.dataset.collate_word_segments``.
        optimizer: Optimizer already constructed over ``model``'s
            parameters.
        device: Device to run the epoch on.

    Returns:
        A dict with ``"loss"`` and ``"accuracy"``, averaged/computed over
        the whole epoch.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss, total_correct, total_examples = 0.0, 0, 0
    for patches, coords, labels, mask in dataloader:
        patches, coords, labels, mask = (
            patches.to(device), coords.to(device), labels.to(device), mask.to(device)
        )

        optimizer.zero_grad()
        logits = model(patches, coords, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size

    return {"loss": total_loss / total_examples, "accuracy": total_correct / total_examples}


@torch.no_grad()
def evaluate(
    model: GridWordRecognitionModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate ``model`` over ``dataloader`` without updating parameters.

    Args:
        model: The pretext-task model.
        dataloader: Yields ``(patches, aligned_coords, labels, mask)``
            batches.
        device: Device to run evaluation on.

    Returns:
        A dict with ``"loss"`` and ``"accuracy"``.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, total_correct, total_examples = 0.0, 0, 0
    for patches, coords, labels, mask in dataloader:
        patches, coords, labels, mask = (
            patches.to(device), coords.to(device), labels.to(device), mask.to(device)
        )

        logits = model(patches, coords, mask)
        loss = criterion(logits, labels)

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size

    return {"loss": total_loss / total_examples, "accuracy": total_correct / total_examples}
