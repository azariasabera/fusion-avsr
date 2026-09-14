"""Manifest builders for GRID, LRS3-trainval, and LRS3-test-mattymchen.

Turns each dataset's own raw, source-specific directory/file layout into
ONE consistent per-clip manifest schema. Downstream code (encoders,
fusion, training loops) should only ever need to read a manifest CSV --
it should never need to know GRID's raw layout differs from
LRS3-trainval's, or that LRS3-test-mattymchen is a Hugging Face parquet
dataset rather than a folder of video files.

Per-clip manifest schema (one row per clip, one manifest per
dataset+split):

    sample_id       unique clip identifier
    video_path      path to the raw video file (empty for test-mattymchen)
    audio_path      path to the extracted 16kHz mono .wav (always a real
                    file, for every source, including test-mattymchen)
    landmark_path   path to the 68-point .pkl landmark file (empty for
                    test-mattymchen -- no landmark files exist for it, see
                    build_lrs3_test_mattymchen_manifest's docstring)
    transcript      normalized, lowercased, plain text transcript
    duration_sec    clip duration in seconds
    source          one of "lrs3_trainval", "lrs3_test_mattymchen", "grid"

GRID additionally needs a second, WORD-level table (produced by
``build_grid_word_segments``), because the landmark encoder's
word-recognition pretext task trains on individual word segments sliced
out of each GRID clip, not on whole clips. See that function's docstring
for the exact schema.

This module also provides three reusable consistency-check functions
(``check_file_existence``, ``check_frame_count_vs_duration``, and
``check_video_fps``) that can be re-run against any manifest to catch
missing files, mismatched landmark frame counts, or a video's real frame
rate not actually matching the assumed 25fps -- rather than checked by
hand.
"""

from __future__ import annotations

import pickle
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from fusion_avsr.data.audio_extraction import (
    extract_wav_from_pcm,
    get_extracted_wav_path,
    get_wav_duration_sec,
)
from fusion_avsr.utils.logging import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]

# GRID's .align files express word-boundary timestamps in units of
# 1/25000 second.
GRID_ALIGN_UNITS_PER_SEC = 25000

# Both GRID and LRS3 video is 25fps.
VIDEO_FPS = 25

# GRID's non-vocabulary alignment tokens: silence and short pause. These
# are timing markers, not spoken words, and must never appear as training
# rows in the word-segment table.
GRID_NON_WORD_TOKENS = {"sil", "sp"}

# Manifest columns, in the fixed order used throughout this module. Every
# builder below returns a DataFrame with exactly these columns, in this
# order, regardless of source.
MANIFEST_COLUMNS = [
    "sample_id",
    "video_path",
    "audio_path",
    "landmark_path",
    "transcript",
    "duration_sec",
    "source",
]

# Columns of the GRID word-segment table (see build_grid_word_segments).
WORD_SEGMENT_COLUMNS = ["sample_id", "word", "start_frame", "end_frame"]


def _maybe_write_csv(df: pd.DataFrame, output_csv: Optional[PathLike]) -> None:
    """Write ``df`` to ``output_csv`` if a path was given, else do nothing."""
    if output_csv is None:
        return
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def _parse_lrs3_transcript(txt_path: PathLike) -> str:
    """Parse one LRS3-trainval ``.txt`` transcript file into plain text.

    LRS3-trainval transcript files contain a line of the form
    ``Text:  <TEXT IN CAPS>`` followed by a ``Conf:`` confidence line.
    This function strips the ``Text:`` prefix and lowercases the result,
    so the transcript field is directly comparable across sources (e.g.
    with LRS3-test-mattymchen's labels, which are already plain lowercase
    text with no prefix).

    Args:
        txt_path: Path to one LRS3-trainval ``<clip_id>.txt`` file.

    Returns:
        The clip's transcript as plain lowercase text, with the ``Text:``
        prefix and any leading/trailing whitespace removed.

    Raises:
        ValueError: If no line starting with ``Text:`` is found in the
            file.
    """
    txt_path = Path(txt_path)
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Text:"):
                text = line[len("Text:"):].strip()
                return text.lower()

    message = f"No 'Text:' line found in transcript file: {txt_path}"
    logger.error(message)
    raise ValueError(message)


def _parse_grid_align(align_path: PathLike) -> List[Tuple[int, int, str]]:
    """Parse one GRID ``.align`` file into a list of word-boundary rows.

    Each line of a GRID ``.align`` file has the form
    ``<start_units> <end_units> <word>``, where the timestamps are in
    units of 1/25000 second (see ``GRID_ALIGN_UNITS_PER_SEC``). This
    function does NOT filter out ``sil``/``sp`` tokens -- callers that
    need only real vocabulary words (e.g. ``build_grid_word_segments``)
    are responsible for filtering; callers that want a full plain-text
    rendering including silence markers (e.g. informational transcripts)
    can use the raw list as-is.

    Args:
        align_path: Path to one GRID ``<clip>.align`` file.

    Returns:
        A list of ``(start_units, end_units, word)`` tuples, one per line
        of the ``.align`` file, in file order. ``word`` is lowercased.
    """
    align_path = Path(align_path)
    rows: List[Tuple[int, int, str]] = []
    with open(align_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            start_units, end_units, word = int(parts[0]), int(parts[1]), parts[2]
            rows.append((start_units, end_units, word.lower()))
    return rows


def _align_units_to_frame(units: int) -> int:
    """Convert a GRID ``.align`` timestamp (in 1/25000 sec units) to a frame index.

    First converts the raw timestamp to seconds (divide by
    ``GRID_ALIGN_UNITS_PER_SEC``), then converts seconds to a frame index
    (multiply by ``VIDEO_FPS``). Kept as an explicit two-step function,
    rather than the algebraically-equivalent single division, so the
    conversion mirrors the two natural units involved (seconds, then
    frames) and stays easy to double-check.

    Args:
        units: A raw timestamp from a ``.align`` file, in units of
            1/25000 second.

    Returns:
        The corresponding frame index, rounded to the nearest integer
        frame.
    """
    seconds = units / GRID_ALIGN_UNITS_PER_SEC
    frame = round(seconds * VIDEO_FPS)
    return frame


def build_lrs3_trainval_manifest(
    lrs3_root: PathLike,
    audio_output_dir: PathLike,
    output_csv: Optional[PathLike] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build the per-clip manifest for the LRS3-trainval split.

    Walks ``<lrs3_root>/ainncy/trainval/<video_id>/<clip_id>.mp4`` (and its
    matching ``.txt`` transcript), plus the corresponding landmark file at
    ``<lrs3_root>/landmarks/LRS3_landmarks/trainval/<video_id>/<clip_id>.pkl``.
    Audio is NOT extracted here -- ``scripts/extract_audio.sh`` must be
    run against this split first (see that script's docstring); this
    function only resolves the ``.wav`` path it wrote and fails clearly
    if that has not happened yet.

    Args:
        lrs3_root: Path to the LRS3 dataset root (e.g.
            ``/scratch/project_2020712/datasets/lrs3``).
        audio_output_dir: Directory that ``scripts/extract_audio.sh`` was
            told to write ``.wav`` files into. One ``.wav`` per clip is
            expected at ``<audio_output_dir>/<video_id>_<clip_id>.wav``.
        output_csv: If given, the resulting manifest is also written to
            this path as a CSV file.
        limit: If given, stop after this many clips (in sorted
            video_id/clip_id order), instead of walking the entire
            split. Useful for a quick smoke test against a handful of
            real clips before committing to a full run over all ~32,000.

    Returns:
        A DataFrame with the columns listed in ``MANIFEST_COLUMNS``, one
        row per clip, with ``source`` set to ``"lrs3_trainval"``.
    """
    lrs3_root = Path(lrs3_root)
    audio_output_dir = Path(audio_output_dir)
    video_root = lrs3_root / "ainncy" / "trainval"
    landmarks_root = lrs3_root / "landmarks" / "LRS3_landmarks" / "trainval"

    logger.info("Building LRS3-trainval manifest from %s (limit=%s)", video_root, limit)
    rows = []
    for video_dir in sorted(p for p in video_root.iterdir() if p.is_dir()):
        if limit is not None and len(rows) >= limit:
            break
        video_id = video_dir.name
        for mp4_path in sorted(video_dir.glob("*.mp4")):
            if limit is not None and len(rows) >= limit:
                break
            clip_id = mp4_path.stem
            txt_path = video_dir / f"{clip_id}.txt"
            landmark_path = landmarks_root / video_id / f"{clip_id}.pkl"
            sample_id = f"{video_id}_{clip_id}"

            wav_path = get_extracted_wav_path(audio_output_dir, sample_id)

            rows.append({
                "sample_id": sample_id,
                "video_path": str(mp4_path),
                "audio_path": str(wav_path),
                "landmark_path": str(landmark_path),
                "transcript": _parse_lrs3_transcript(txt_path),
                "duration_sec": get_wav_duration_sec(wav_path),
                "source": "lrs3_trainval",
            })

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    logger.info("Built LRS3-trainval manifest: %d clips", len(manifest))
    _maybe_write_csv(manifest, output_csv)
    return manifest


def build_lrs3_test_mattymchen_manifest(
    parquet_data_dir: PathLike,
    audio_output_dir: PathLike,
    output_csv: Optional[PathLike] = None,
    split: str = "train",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build the per-clip manifest for the LRS3-test-mattymchen test set.

    LRS3-test-mattymchen is NOT a folder of video files -- it is a
    Hugging Face ``datasets``-format parquet dataset at
    ``<lrs3_root>/test-mattymchen/data/*.parquet``, with schema
    ``{idx: int64, audio: List[int16], video: List[List[List[uint8]]],
    label: str}``. Each row's ``video`` field is already grayscale,
    already mouth-cropped to 96x96 -- there is no raw video file to
    reference, so ``video_path`` is left empty for every row. Likewise,
    there are no separate landmark files shipped for this source (unlike
    GRID and LRS3-trainval, which each have a real ``.pkl`` landmark file
    per clip) -- so ``landmark_path`` is also left empty for every row.

    For each row, the embedded ``audio`` PCM array is written out to a
    real ``.wav`` file, so that test-mattymchen clips have a real audio
    file on disk exactly like every other source.

    Args:
        parquet_data_dir: Path to the directory containing the
            ``test-mattymchen`` parquet files (i.e.
            ``<lrs3_root>/test-mattymchen/data``).
        audio_output_dir: Directory to write extracted ``.wav`` files
            into. One ``.wav`` per row is written to
            ``<audio_output_dir>/mattymchen_<idx>.wav``.
        output_csv: If given, the resulting manifest is also written to
            this path as a CSV file.
        split: The Hugging Face ``datasets`` split key to read from the
            loaded parquet dataset. Defaults to ``"train"`` (the default
            split name ``datasets.load_dataset`` assigns when loading a
            directory of parquet files with no explicit split naming
            convention). If ``split`` is not present in the loaded
            dataset, the first available split is used instead.
        limit: If given, only the first ``limit`` rows of the dataset are
            processed, instead of the whole split. Useful for a quick
            smoke test against a handful of real examples before
            committing to a full run over all ~1,321 examples.

    Returns:
        A DataFrame with the columns listed in ``MANIFEST_COLUMNS``, one
        row per example, with ``source`` set to
        ``"lrs3_test_mattymchen"``, ``video_path`` and ``landmark_path``
        left empty for every row.
    """
    from datasets import load_dataset

    parquet_data_dir = Path(parquet_data_dir)
    audio_output_dir = Path(audio_output_dir)

    logger.info("Building LRS3-test-mattymchen manifest from %s (limit=%s)", parquet_data_dir, limit)
    dataset_dict = load_dataset("parquet", data_dir=str(parquet_data_dir))
    dataset = dataset_dict[split] if split in dataset_dict else next(iter(dataset_dict.values()))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    rows = []
    for example in dataset:
        idx = example["idx"]
        sample_id = f"mattymchen_{idx}"

        wav_path = audio_output_dir / f"{sample_id}.wav"
        extract_wav_from_pcm(example["audio"], wav_path)

        rows.append({
            "sample_id": sample_id,
            "video_path": "",
            "audio_path": str(wav_path),
            "landmark_path": "",
            "transcript": example["label"].strip().lower(),
            "duration_sec": get_wav_duration_sec(wav_path),
            "source": "lrs3_test_mattymchen",
        })

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    logger.info("Built LRS3-test-mattymchen manifest: %d clips", len(manifest))
    _maybe_write_csv(manifest, output_csv)
    return manifest


def build_grid_manifest(
    grid_root: PathLike,
    landmarks_root: PathLike,
    audio_output_dir: PathLike,
    output_csv: Optional[PathLike] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build the per-clip manifest for GRID.

    Walks ``<grid_root>/s<N>_processed/<clip>.mpg`` (and its matching
    ``.align`` file at ``<grid_root>/s<N>_processed/align/<clip>.align``).
    Audio is NOT extracted here -- ``scripts/extract_audio.sh`` must be
    run against GRID first (see that script's docstring); this function
    only resolves the ``.wav`` path it wrote and fails clearly if that
    has not happened yet. Landmarks are also not generated here -- this
    function expects a landmark ``.pkl`` file to already exist at
    ``<landmarks_root>/s<N>_processed/<clip>.pkl`` -- i.e. the landmarks
    directory mirrors GRID's own ``s<N>_processed/<clip>`` layout exactly,
    just rooted at ``landmarks_root`` instead of ``grid_root`` and with a
    ``.pkl`` extension instead of ``.mpg``. This is the same path
    convention ``scripts/extract_landmarks_grid.py`` writes to -- the two
    must stay in sync.

    IMPORTANT: this is a per-CLIP manifest, used only for bookkeeping and
    consistency checks (e.g. "does every clip have a landmark file").
    GRID's ``transcript`` field here is a plain-text rendering for humans
    reading the manifest ONLY -- it is NOT what the word-recognition
    pretext task actually trains on. That task trains on individual WORD
    segments, produced by ``build_grid_word_segments`` instead.

    Args:
        grid_root: Path to the GRID dataset root (e.g.
            ``/scratch/project_2020712/datasets/kaggle_lipnet/datasets/
            jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet/
            versions/1/data``).
        landmarks_root: Path to the root directory where GRID landmark
            ``.pkl`` files live (or will live, once
            ``scripts/extract_landmarks_grid.py`` has been run).
        audio_output_dir: Directory that ``scripts/extract_audio.sh`` was
            told to write ``.wav`` files into. One ``.wav`` per clip is
            expected at ``<audio_output_dir>/<speaker>_<clip>.wav``.
        output_csv: If given, the resulting manifest is also written to
            this path as a CSV file.
        limit: If given, stop after this many clips (in sorted
            speaker/clip order), instead of walking all 33 speakers.
            Useful for a quick smoke test against a handful of real
            clips before committing to a full run over all ~33,000.

    Returns:
        A DataFrame with the columns listed in ``MANIFEST_COLUMNS``, one
        row per clip, with ``source`` set to ``"grid"``.
    """
    grid_root = Path(grid_root)
    landmarks_root = Path(landmarks_root)
    audio_output_dir = Path(audio_output_dir)

    logger.info("Building GRID manifest from %s (limit=%s)", grid_root, limit)
    rows = []
    for speaker_dir in sorted(p for p in grid_root.iterdir() if p.is_dir()):
        if limit is not None and len(rows) >= limit:
            break
        speaker = speaker_dir.name.replace("_processed", "")
        align_dir = speaker_dir / "align"

        for mpg_path in sorted(speaker_dir.glob("*.mpg")):
            if limit is not None and len(rows) >= limit:
                break
            clip_id = mpg_path.stem
            align_path = align_dir / f"{clip_id}.align"
            landmark_path = landmarks_root / speaker_dir.name / f"{clip_id}.pkl"
            sample_id = f"{speaker}_{clip_id}"

            wav_path = get_extracted_wav_path(audio_output_dir, sample_id)

            align_rows = _parse_grid_align(align_path)
            words = [word for (_, _, word) in align_rows if word not in GRID_NON_WORD_TOKENS]
            transcript = " ".join(words)

            rows.append({
                "sample_id": sample_id,
                "video_path": str(mpg_path),
                "audio_path": str(wav_path),
                "landmark_path": str(landmark_path),
                "transcript": transcript,
                "duration_sec": get_wav_duration_sec(wav_path),
                "source": "grid",
            })

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    logger.info("Built GRID manifest: %d clips", len(manifest))
    _maybe_write_csv(manifest, output_csv)
    return manifest


def build_grid_word_segments(
    grid_root: PathLike,
    output_csv: Optional[PathLike] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build the word-level segment table used by the word-recognition pretext task.

    GRID is a word-level classification pretext task, not sentence-level
    like LRS3 -- the per-clip GRID manifest (``build_grid_manifest``) is
    bookkeeping only. The real training unit for the landmark encoder's
    word-recognition pretraining is a WORD-level segment, sliced out of
    each clip using its ``.align`` file's word boundaries. This function
    produces exactly that table.

    For each ``.align`` file under
    ``<grid_root>/s<N>_processed/align/<clip>.align``, every word boundary
    line is converted into one output row, EXCEPT ``sil`` (silence) and
    ``sp`` (short pause) segments, which are excluded entirely -- they are
    timing markers, not one of GRID's real vocabulary words, and must
    never appear as training rows.

    Timestamps are converted by dividing the raw ``.align`` timestamp (in
    units of 1/25000 second) by 25000 to get seconds, then multiplying by
    25fps to get a frame index.

    Args:
        grid_root: Path to the GRID dataset root (same as
            ``build_grid_manifest``'s ``grid_root`` argument).
        output_csv: If given, the resulting table is also written to this
            path as a CSV file (conventionally
            ``manifests/grid_word_segments.csv``).
        limit: If given, stop after this many CLIPS (i.e. ``.align``
            files, in sorted speaker/clip order) have been processed --
            note this bounds the number of source clips, not the number
            of output word-segment rows, since one clip produces several
            words. Useful for a quick smoke test before committing to a
            full run over all 33,000 clips.

    Returns:
        A DataFrame with columns ``sample_id``, ``word``, ``start_frame``,
        ``end_frame`` (see ``WORD_SEGMENT_COLUMNS``), one row per word
        segment, across every GRID clip. ``sample_id`` matches the parent
        clip's ``sample_id`` in the per-clip GRID manifest, so the two
        tables can be joined.
    """
    grid_root = Path(grid_root)

    logger.info("Building GRID word-segment table from %s (limit=%s)", grid_root, limit)
    rows = []
    num_clips_processed = 0
    for speaker_dir in sorted(p for p in grid_root.iterdir() if p.is_dir()):
        if limit is not None and num_clips_processed >= limit:
            break
        speaker = speaker_dir.name.replace("_processed", "")
        align_dir = speaker_dir / "align"

        for align_path in sorted(align_dir.glob("*.align")):
            if limit is not None and num_clips_processed >= limit:
                break
            clip_id = align_path.stem
            sample_id = f"{speaker}_{clip_id}"
            num_clips_processed += 1

            for start_units, end_units, word in _parse_grid_align(align_path):
                if word in GRID_NON_WORD_TOKENS:
                    continue
                rows.append({
                    "sample_id": sample_id,
                    "word": word,
                    "start_frame": _align_units_to_frame(start_units),
                    "end_frame": _align_units_to_frame(end_units),
                })

    word_segments = pd.DataFrame(rows, columns=WORD_SEGMENT_COLUMNS)
    logger.info("Built GRID word-segment table: %d word segments", len(word_segments))
    _maybe_write_csv(word_segments, output_csv)
    return word_segments


def check_file_existence(
    manifest: pd.DataFrame,
    path_columns: Tuple[str, ...] = ("video_path", "audio_path", "landmark_path"),
) -> pd.DataFrame:
    """Check that every non-empty path referenced by a manifest actually exists on disk.

    Empty/missing path values (``""`` or ``None``, as used for
    ``video_path``/``landmark_path`` on LRS3-test-mattymchen rows) are
    skipped, since those are expected to be absent for that source -- this
    function only flags paths that SHOULD point to a real file but don't.

    Args:
        manifest: A manifest DataFrame produced by one of the
            ``build_*_manifest`` functions.
        path_columns: Which columns to check for file existence. Defaults
            to all three path-valued manifest columns.

    Returns:
        A DataFrame of problems found, with columns ``sample_id``,
        ``column``, ``path``. Empty (zero rows) if every referenced path
        exists.
    """
    problems = []
    for _, row in manifest.iterrows():
        for column in path_columns:
            path_value = row[column]
            if path_value is None or path_value == "":
                continue
            if not Path(path_value).exists():
                problems.append({
                    "sample_id": row["sample_id"],
                    "column": column,
                    "path": path_value,
                })

    if problems:
        logger.warning("check_file_existence found %d missing path(s)", len(problems))
    else:
        logger.info("check_file_existence: all referenced paths exist")

    return pd.DataFrame(problems, columns=["sample_id", "column", "path"])


def check_frame_count_vs_duration(
    manifest: pd.DataFrame,
    fps: int = VIDEO_FPS,
    tolerance_frames: int = 1,
) -> pd.DataFrame:
    """Check that each clip's landmark frame count matches its audio duration.

    For every row with a non-empty ``landmark_path``, loads the landmark
    ``.pkl`` file, and compares its length (number of per-frame landmark
    entries) against ``duration_sec * fps``, rounded to the nearest frame.

    Args:
        manifest: A manifest DataFrame produced by one of the
            ``build_*_manifest`` functions.
        fps: Video frame rate to use for the expected-frame-count
            calculation. Defaults to 25fps.
        tolerance_frames: Maximum allowed absolute difference between the
            expected and actual frame count before a row is flagged as a
            problem. Defaults to 1, to allow for ordinary floating-point
            rounding at clip boundaries.

    Returns:
        A DataFrame of problems found, with columns ``sample_id``,
        ``landmark_path``, ``expected_frames``, ``actual_frames``,
        ``diff``. Empty (zero rows) if every clip's landmark frame count
        matches its duration within tolerance.
    """
    problems = []
    for _, row in manifest.iterrows():
        landmark_path = row["landmark_path"]
        if landmark_path is None or landmark_path == "":
            continue

        with open(landmark_path, "rb") as f:
            landmarks = pickle.load(f)
        actual_frames = len(landmarks)
        expected_frames = round(row["duration_sec"] * fps)
        diff = actual_frames - expected_frames

        if abs(diff) > tolerance_frames:
            problems.append({
                "sample_id": row["sample_id"],
                "landmark_path": landmark_path,
                "expected_frames": expected_frames,
                "actual_frames": actual_frames,
                "diff": diff,
            })

    if problems:
        logger.warning("check_frame_count_vs_duration found %d mismatched clip(s)", len(problems))
    else:
        logger.info("check_frame_count_vs_duration: all landmark frame counts match")

    return pd.DataFrame(
        problems,
        columns=["sample_id", "landmark_path", "expected_frames", "actual_frames", "diff"],
    )


def _probe_video_fps(video_path: PathLike) -> float:
    """Read a video file's frame rate via ffprobe.

    Reads only the container's stream metadata (the ``r_frame_rate``
    field), not the actual video frames, so this is cheap to run across
    an entire manifest even though it shells out to a subprocess per
    clip.

    Args:
        video_path: Path to a video file.

    Returns:
        The video's frame rate, in frames per second, as a float.

    Raises:
        subprocess.CalledProcessError: If ffprobe exits with a non-zero
            return code (e.g. the file is corrupt/unreadable).
    """
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    # ffprobe reports r_frame_rate as a "<numerator>/<denominator>"
    # fraction string, e.g. "25/1" or "30000/1001".
    numerator, denominator = result.stdout.strip().split("/")
    return float(numerator) / float(denominator)


def check_video_fps(
    manifest: pd.DataFrame,
    expected_fps: int = VIDEO_FPS,
    tolerance: float = 0.5,
) -> pd.DataFrame:
    """Check that every referenced video file's actual frame rate matches the assumed fps.

    ``VIDEO_FPS`` (25fps) is assumed throughout this module (e.g. for the
    GRID ``.align`` timestamp-to-frame conversion, and as the default for
    ``check_frame_count_vs_duration``), but never directly verified
    against the real video files it's applied to. This function closes
    that gap: for every row with a non-empty ``video_path``, it reads the
    file's actual frame rate via ffprobe and flags any clip whose real
    fps doesn't match ``expected_fps``.

    Args:
        manifest: A manifest DataFrame produced by one of the
            ``build_*_manifest`` functions.
        expected_fps: The frame rate every clip is assumed to be at.
            Defaults to ``VIDEO_FPS`` (25fps).
        tolerance: Maximum allowed absolute difference, in fps, between
            the expected and actual frame rate before a row is flagged
            as a problem. Defaults to 0.5, to allow for ordinary
            floating-point rounding in the reported frame rate.

    Returns:
        A DataFrame of problems found, with columns ``sample_id``,
        ``video_path``, ``expected_fps``, ``actual_fps``. Empty (zero
        rows) if every clip's frame rate matches within tolerance.
    """
    problems = []
    for _, row in manifest.iterrows():
        video_path = row["video_path"]
        if video_path is None or video_path == "":
            continue

        actual_fps = _probe_video_fps(video_path)
        if abs(actual_fps - expected_fps) > tolerance:
            problems.append({
                "sample_id": row["sample_id"],
                "video_path": video_path,
                "expected_fps": expected_fps,
                "actual_fps": actual_fps,
            })

    if problems:
        logger.warning("check_video_fps found %d clip(s) with unexpected frame rate", len(problems))
    else:
        logger.info("check_video_fps: all video frame rates match")

    return pd.DataFrame(problems, columns=["sample_id", "video_path", "expected_fps", "actual_fps"])
