"""Generate 68-point landmark .pkl files for GRID, using RetinaFace.

GRID's raw .mpg clips are full-frame, uncropped video -- unlike LRS3,
which already ships pre-computed 68-point landmark .pkl files, GRID has
no landmarks at all and they must be generated ourselves.

This script runs the vsr_multilang submodule's RetinaFace-based
LandmarksDetector (the same detector auto-AVSR uses by default) over
every GRID clip, and writes one .pkl file per clip containing a list of
per-frame (68, 2) landmark coordinate arrays -- the same format LRS3's
existing landmark files already use. RetinaFace is used explicitly
(never mediapipe), so that GRID's landmarks follow the same detector
convention as LRS3's.

Configured via Hydra -- see configs/scripts/extract_landmarks_grid.yaml
for all available fields and their meaning. Example invocation:

    python scripts/extract_landmarks_grid.py \\
        grid_root=/scratch/your_project_name/datasets/kaggle_lipnet/.../data \\
        landmarks_root=/scratch/your_project_name/datasets/grid_landmarks

Note: this script only produces landmark COORDINATE files. It does not
do anything else with them (no patch extraction, no nose-alignment --
that is a separate, later step).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

# The vsr_multilang submodule's own code imports as `from pipelines...`,
# which only resolves if the submodule's root directory is on sys.path
# (it is not an installed package). Add it before importing from it.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_VSR_MULTILANG_ROOT = _REPO_ROOT / "external" / "vsr_multilang"
if str(_VSR_MULTILANG_ROOT) not in sys.path:
    sys.path.insert(0, str(_VSR_MULTILANG_ROOT))

# fusion_avsr itself also needs the repo root on sys.path, for the same
# reason (no pyproject.toml/setup.py yet -- see tests/conftest.py).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torchvision.io  # noqa: E402

from fusion_avsr.utils.logging import get_logger  # noqa: E402
from fusion_avsr.utils.video import iter_decodable_frames  # noqa: E402


def _read_video_via_torchcodec(
    filename: str | Path,
    pts_unit: str = "sec",
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Decode a video with TorchCodec using torchvision's return contract.

    This compatibility shim is used by ``LandmarksDetector``.

    Args:
        filename: Path to the video file to decode.
        pts_unit: Timestamp unit accepted by the torchvision API. It is kept
            for API compatibility and is not used by TorchCodec here.
        **kwargs: Additional torchvision arguments accepted for compatibility.

    Returns:
        A tuple containing video frames in ``(T, H, W, C)`` layout, an empty
        audio tensor, and an empty metadata dictionary.

    Raises:
        RuntimeError: If any frame fails to decode. A partially decoded
            clip is rejected rather than returned truncated, since a short
            landmark file would silently misalign with the video.
    """
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(filename, dimension_order="NHWC")
    expected_frames = len(decoder)
    frames: list[torch.Tensor] = list(iter_decodable_frames(decoder))

    if len(frames) < expected_frames:
        raise RuntimeError(
            f"Only decoded {len(frames)}/{expected_frames} frames of {filename} -- "
            f"a partially decoded clip is rejected rather than returned truncated."
        )
    if not frames:
        raise RuntimeError(f"Could not decode any frames from {filename}")

    video_frames = torch.stack(frames)
    return video_frames, torch.empty(0), {}


torchvision.io.read_video = _read_video_via_torchcodec

logger = get_logger(__name__)


def extract_landmarks_for_grid(
    grid_root: Path,
    landmarks_root: Path,
    device: str = "cuda:0",
    model_name: str = "resnet50",
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> List[Path]:
    """Run RetinaFace landmark detection over every GRID clip.

    Walks ``<grid_root>/s<N>_processed/<clip>.mpg``, and for each clip
    writes a landmark file to
    ``<landmarks_root>/s<N>_processed/<clip>.pkl`` -- the same
    ``s<N>_processed/<clip>`` layout GRID's raw video files already use,
    just rooted at ``landmarks_root`` instead of ``grid_root``. This is
    the exact path convention ``fusion_avsr.data.manifest_builder.
    build_grid_manifest`` expects landmark files to live at.

    Args:
        grid_root: Path to the GRID dataset root, containing one
            ``s<N>_processed/`` folder per speaker.
        landmarks_root: Root directory to write landmark ``.pkl`` files
            into. Created automatically if it does not exist.
        device: Device string passed to the RetinaFace face detector and
            FAN landmark predictor (e.g. ``"cuda:0"`` or ``"cpu"``).
        model_name: RetinaFace backbone model name.
        overwrite: If False (default), clips that already have a landmark
            ``.pkl`` file at the expected output path are skipped. If
            True, every clip is reprocessed.
        limit: If given, only the first ``limit`` clips (in sorted
            speaker/clip order) are processed.

    Returns:
        A list of paths to every landmark ``.pkl`` file written (or
        already present and skipped) by this call, in processing order.
    """
    from pipelines.detectors.retinaface.detector import LandmarksDetector

    grid_root = Path(grid_root)
    landmarks_root = Path(landmarks_root)

    mpg_paths = []
    for speaker_dir in sorted(p for p in grid_root.iterdir() if p.is_dir()):
        for mpg_path in sorted(speaker_dir.glob("*.mpg")):
            mpg_paths.append(mpg_path)
    if limit is not None:
        mpg_paths = mpg_paths[:limit]

    logger.info("Found %d GRID clips under %s", len(mpg_paths), grid_root)
    logger.info("Loading RetinaFace landmark detector on device=%s, model_name=%s", device, model_name)
    landmarks_detector = LandmarksDetector(device=device, model_name=model_name)

    output_paths = []
    num_skipped = 0
    num_failed = 0
    for mpg_path in tqdm(mpg_paths, desc="Extracting GRID landmarks"):
        speaker_dir_name = mpg_path.parent.name
        clip_id = mpg_path.stem
        output_path = landmarks_root / speaker_dir_name / f"{clip_id}.pkl"

        if output_path.exists() and not overwrite:
            output_paths.append(output_path)
            num_skipped += 1
            continue

        try:
            landmarks = landmarks_detector(str(mpg_path))
        except Exception as e:
            logger.error("Failed on %s: %s", mpg_path, e)
            landmarks_root.mkdir(parents=True, exist_ok=True)
            with open(landmarks_root / "landmark_errors.log", "a") as f:
                f.write(f"{mpg_path}: {e}\n")
            num_failed += 1
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(landmarks, f)
        output_paths.append(output_path)

    logger.info(
        "Wrote %d landmark files under %s (%d already existed and were skipped, %d failed)",
        len(output_paths) - num_skipped, landmarks_root, num_skipped, num_failed,
    )
    if num_failed:
        logger.warning("%d clips failed; see %s", num_failed, landmarks_root / "landmark_errors.log")
    return output_paths


@hydra.main(version_base=None, config_path="../configs/scripts", config_name="extract_landmarks_grid")
def main(cfg: DictConfig) -> None:
    if cfg.grid_root is None:
        raise ValueError("grid_root must be set, e.g. grid_root=/path/to/grid/data")
    if cfg.landmarks_root is None:
        raise ValueError("landmarks_root must be set, e.g. landmarks_root=/path/to/grid_landmarks")

    extract_landmarks_for_grid(
        grid_root=Path(cfg.grid_root),
        landmarks_root=Path(cfg.landmarks_root),
        device=cfg.device,
        model_name=cfg.model_name,
        overwrite=cfg.overwrite,
        limit=cfg.limit,
    )


if __name__ == "__main__":
    main()
