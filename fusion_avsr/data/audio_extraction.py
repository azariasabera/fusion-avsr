"""Audio extraction utilities.

Produces one real, standalone 16kHz mono ``.wav`` file per clip, for
every data source used in this project (GRID, LRS3-trainval, and
LRS3-test). Downstream code (manifest builders, data loaders)
should never need to know which video container a clip's audio
originally lived in -- after extraction, every clip has a real
``.wav`` file on disk at a known path.

Audio lives inside a video container (``.mpg`` for GRID, ``.mp4`` for
LRS3). Extracting it means demuxing and resampling with ffmpeg, over tens
of thousands of clips -- this is done by the standalone, parallel batch
script ``scripts/extract_audio.sh`` (a shell script, not this module:
running ~65,000 independent ffmpeg subprocesses is an embarrassingly
parallel shell-out job, and ``xargs -P`` handles that more simply than a
Python process pool would). ``get_extracted_wav_path`` resolves the path
that script writes to for a given clip, and checks the script has
actually been run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import soundfile as sf

from fusion_avsr.utils.logging import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]

# The whole project standardizes on 16kHz mono audio: it is what the
# frozen audio encoder (wav2vec2-large-960h-lv60-self) expects.
DEFAULT_SAMPLE_RATE = 16000


def get_extracted_wav_path(output_dir: PathLike, sample_id: str) -> Path:
    """Resolve and verify the path to a clip's already-extracted ``.wav`` file.

    GRID and LRS3-trainval audio is extracted in bulk by
    ``scripts/extract_audio.sh``, which writes one ``<sample_id>.wav``
    file per clip into a given output directory. This function does not
    perform extraction itself -- it just resolves the path that shell
    script writes to (following the exact same ``<sample_id>.wav`` naming
    convention) and confirms the file is actually there, so a manifest
    builder fails immediately with a clear message if the shell script
    has not been run yet, rather than producing a manifest that silently
    points at nonexistent audio files.

    Args:
        output_dir: The directory ``scripts/extract_audio.sh`` was told
            to write ``.wav`` files into.
        sample_id: The clip's sample_id (e.g. ``"<video_id>_<clip_id>"``
            for LRS3-trainval, ``"<speaker>_<clip_id>"`` for GRID) --
            must match the naming convention ``scripts/extract_audio.sh``
            uses.

    Returns:
        The path to the extracted ``.wav`` file.

    Raises:
        FileNotFoundError: If no ``.wav`` file exists at the expected
            path, meaning ``scripts/extract_audio.sh`` has not been run
            for this clip yet.
    """
    wav_path = Path(output_dir) / f"{sample_id}.wav"
    if not wav_path.exists():
        message = (
            f"Expected extracted audio at {wav_path}, but it does not exist. "
            f"Run scripts/extract_audio.sh first to extract GRID/LRS3-trainval audio."
        )
        logger.error(message)
        raise FileNotFoundError(message)

    actual_sample_rate = sf.info(str(wav_path)).samplerate
    if actual_sample_rate != DEFAULT_SAMPLE_RATE:
        logger.warning(
            "%s is at %dHz, expected %dHz -- scripts/extract_audio.sh should "
            "have resampled it; check that script's -ar argument.",
            wav_path, actual_sample_rate, DEFAULT_SAMPLE_RATE,
        )

    return wav_path


def get_wav_duration_sec(wav_path: PathLike) -> float:
    """Return the duration, in seconds, of a ``.wav`` file.

    Used by the manifest builders to fill in each clip's ``duration_sec``
    column cheaply, by reading the already-extracted ``.wav`` file's own
    length rather than re-probing the original source video/dataset row.

    Args:
        wav_path: Path to a ``.wav`` file (as produced by
            ``scripts/extract_audio.sh``).

    Returns:
        Duration of the audio in seconds, as a float.
    """
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate
