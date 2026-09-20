"""Manifest builders for GRID, LRS3-trainval, and LRS3-test.

Turns each dataset's own raw, source-specific directory/file layout into
ONE consistent per-clip manifest schema. Downstream code (encoders,
fusion, training loops) should only ever need to read a manifest CSV --
it should never need to know GRID's raw layout differs from
LRS3-trainval's, or that LRS3-test is a Hugging Face parquet
dataset rather than a folder of video files.

Per-clip manifest schema (one row per clip, one manifest per
dataset+split):

    sample_id       unique clip identifier
    video_path      path to the raw video file (empty for lrs3_test)
    audio_path      path to the extracted 16kHz mono .wav (always a real
                    file, for every source, including lrs3_test)
    landmark_path   path to the 68-point .pkl landmark file (empty for
                    lrs3_test clips whose video_id/clip_id could not be
                    resolved, see build_lrs3_test_manifest's docstring)
    transcript      normalized, lowercased, plain text transcript
    duration_sec    clip duration in seconds
    source          one of "lrs3_trainval", "lrs3_test", "grid"

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

# Placeholder landmark paths for LRS3-test clips that cannot be matched to a
# video_id/clip_id although a mapping CSV was supplied. They point at no real
# file on purpose, so ``clean_manifest`` drops those rows (and the reason is
# visible in the dropped-rows log).
LANDMARK_MARKER_AMBIGUOUS = "AMBIGUOUS_MAPPING"
LANDMARK_MARKER_NOT_FOUND = "NOT_IN_MAPPING"

# Columns of the GRID word-segment table (see build_grid_word_segments).
WORD_SEGMENT_COLUMNS = ["sample_id", "word", "start_frame", "end_frame"]

# Columns of the LRS3-test parquet index side table (see build_lrs3_test_manifest).
PARQUET_INDEX_COLUMNS = ["sample_id", "parquet_idx"]


def _maybe_write_csv(df: pd.DataFrame, output_csv: Optional[PathLike]) -> None:
    """Write ``df`` to ``output_csv`` if a path was given, else do nothing."""
    if output_csv is None:
        return
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def _finalize_manifest(
    manifest: pd.DataFrame,
    output_csv: Optional[PathLike],
    clean: bool,
) -> pd.DataFrame:
    """Optionally clean a freshly built manifest, then write it to CSV.

    The saved CSV is the canonical artifact later stages load instead of
    rebuilding manifests, so the DataFrame returned here is exactly what
    was written (cleaned when ``clean`` is True), never a different,
    uncleaned version.

    Args:
        manifest: Freshly built manifest DataFrame.
        output_csv: Optional CSV destination. If given and ``clean`` is
            True, dropped sample IDs are logged to
            ``<output_csv stem>_dropped.log`` next to it.
        clean: If True, apply ``clean_manifest`` (missing files, landmark
            frame-count mismatches) before writing and returning. Only
            takes effect when ``output_csv`` is given: without a CSV the
            manifest is returned as built, uncleaned.

    Returns:
        The manifest that was written (or would be written).
    """
    if clean and output_csv is not None:
        output_csv = Path(output_csv)
        log_path = output_csv.with_name(f"{output_csv.stem}_dropped.log")
        manifest = clean_manifest(manifest, log_path=log_path)
    _maybe_write_csv(manifest, output_csv)
    return manifest


def _parse_lrs3_transcript(txt_path: PathLike) -> str:
    """Parse one LRS3-trainval ``.txt`` transcript file into plain text.

    LRS3-trainval transcript files contain a line of the form
    ``Text:  <TEXT IN CAPS>`` followed by a ``Conf:`` confidence line.
    This function strips the ``Text:`` prefix and lowercases the result,
    so the transcript field is directly comparable across sources (e.g.
    with LRS3-test labels, which are already plain lowercase
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
    video_root: PathLike,
    audio_output_dir: PathLike,
    landmarks_root: PathLike,
    output_csv: Optional[PathLike] = None,
    limit: Optional[int] = None,
    clean: bool = True,
) -> pd.DataFrame:
    """Build the per-clip manifest for the LRS3-trainval split.

    Walks ``<video_root>/<video_id>/<clip_id>.mp4`` and its matching
    ``.txt`` transcript. Landmark paths are resolved under
    ``<landmarks_root>/<video_id>/<clip_id>.pkl``. Audio is NOT extracted
    here -- ``scripts/extract_audio.sh`` must be run against this split
    first (see that script's docstring); this function only resolves the
    ``.wav`` path it wrote and fails clearly if that has not happened yet.

    Args:
        video_root: Path to the LRS3-trainval directory containing one
            ``<video_id>`` directory per source video, for example
            ``/scratch/your_project_name/datasets/lrs3/ainncy/trainval``.
        audio_output_dir: Directory that ``scripts/extract_audio.sh`` was
            told to write ``.wav`` files into. One ``.wav`` per clip is
            expected at ``<audio_output_dir>/<video_id>_<clip_id>.wav``.
        landmarks_root: Root directory containing the LRS3-trainval
            landmark files, for example
            ``/scratch/your_project_name/datasets/lrs3/landmarks/LRS3_landmarks/trainval``.
        output_csv: If given, the resulting manifest is also written to
            this path as a CSV file.
        limit: If given, stop after this many clips (in sorted
            video_id/clip_id order), instead of walking the entire
            split. Useful for a quick smoke test against a handful of
            real clips before committing to a full run over all ~32,000.
        clean: If True (default) and ``output_csv`` is given, drop rows
            with missing files or landmark frame-count mismatches (see
            ``clean_manifest``) before writing the CSV, so the returned
            DataFrame matches the saved CSV exactly. Dropped sample IDs
            are logged next to ``output_csv``. Ignored without
            ``output_csv``.

    Returns:
        A DataFrame with the columns listed in ``MANIFEST_COLUMNS``, one
        row per clip, with ``source`` set to ``"lrs3_trainval"``.
    """
    video_root = Path(video_root)
    audio_output_dir = Path(audio_output_dir)
    landmarks_root = Path(landmarks_root)

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
    return _finalize_manifest(manifest, output_csv, clean)


def build_lrs3_test_manifest(
    parquet_data_dir: PathLike,
    audio_output_dir: PathLike,
    landmarks_root: PathLike,
    landmark_mapping_csv: Optional[PathLike] = None,
    output_csv: Optional[PathLike] = None,
    split: str = "train",
    limit: Optional[int] = None,
    clean: bool = True,
    parquet_index_csv: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Build the per-clip manifest for the LRS3 test split.
    Also saves audio clips as .wav into dedicated path.

    Backed by a Hugging Face ``datasets``-format parquet dataset.
    Schema ``{idx: int64, audio:List[int16], video: List[List[List[uint8]]], 
    label: str}``. Each row's ``video`` field is already grayscale, already 
    mouth-cropped to 96x96. There is no raw video FILE to reference, so 
    ``video_path`` is always left empty.

    The parquet's own schema has no ``video_id``/``clip_id``
    field, so recovering it requires ``landmark_mapping_csv``: a small,
    committed (``tests/lrs3_test_video_id_mapping_example.csv``) table 
    of ``sample_id -> video_id, clip_id`` derived by matching this 
    parquet's transcripts.

    The transcript in the CSV is matched against the corresponding transcript 
    in the parquet dataset. There are 26 cases where the transcript matches 
    multiple samples. The number of frames disambiguates most of them, 
    but 6 cases remain unresolved.

    When ``landmark_mapping_csv`` is given, a row whose video_id/clip_id
    cannot be resolved gets ``lrs3test_<idx>`` naming and a marker in
    ``landmark_path`` (``LANDMARK_MARKER_AMBIGUOUS`` if its transcript and
    frame count match several mapping rows, ``LANDMARK_MARKER_NOT_FOUND`` if
    they match none). The marker is not a real file, so ``clean_manifest``
    drops the row when the CSV is saved. This keeps every saved test set
    limited to clips that have landmarks, so all models are evaluated on the
    same clips. If ``landmark_mapping_csv`` is omitted entirely, no landmarks
    are expected: rows use ``lrs3test_<idx>`` naming with an empty
    ``landmark_path`` and are kept.
    Still fully functional, just without the trainval-matching naming
    or any landmark-inclusive evaluation.

    Args:
        parquet_data_dir: Path to the directory containing the parquet
            files.
        audio_output_dir: Directory to write extracted ``.wav`` files
            into -- shared with LRS3-trainval's own audio output.
        landmarks_root: Root directory of the official LRS3 test-split
            landmark ``.pkl`` files.
        landmark_mapping_csv: Path to the committed
            ``sample_id -> video_id, clip_id`` mapping. If ``None``,
            every row uses ``lrs3test_<idx>`` naming and no landmarks
            are attached.
        output_csv: If given, the resulting manifest is also written to
            this path as a CSV file.
        split: The Hugging Face ``datasets`` split key to read.
            Defaults to ``"train"``.
        limit: If given, only the first ``limit`` rows are processed.
        clean: If True (default) and ``output_csv`` is given, drop rows
            with missing files or landmark frame-count mismatches (see
            ``clean_manifest``) before writing the CSV, so the returned
            DataFrame matches the saved CSV exactly. Dropped sample IDs
            are logged next to ``output_csv``. Ignored without
            ``output_csv``.
        parquet_index_csv: If given, a side table with columns
            ``PARQUET_INDEX_COLUMNS`` (``sample_id, parquet_idx``) is
            written to this path, mapping each manifest row's
            ``sample_id`` to its ``idx`` (row position) in the parquet
            dataset. It exists because this source's audio and frames live
            in a parquet dataset addressed by row position, while
            ``sample_id`` (``<video_id>_<clip_id>`` where matched,
            ``lrs3test_<idx>`` as fallback) does not equal that position in
            general.
    Returns:
        A DataFrame with the columns listed in ``MANIFEST_COLUMNS``, one
        row per example, ``source`` set to ``"lrs3_test"``.
    """
    from datasets import load_dataset

    parquet_data_dir = Path(parquet_data_dir)
    audio_output_dir = Path(audio_output_dir)
    landmarks_root = Path(landmarks_root)

    logger.info("Building LRS3-test manifest from %s (limit=%s)", parquet_data_dir, limit)
    dataset_dict = load_dataset("parquet", data_dir=str(parquet_data_dir))
    dataset = dataset_dict[split] if split in dataset_dict else next(iter(dataset_dict.values()))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # (transcript, n_frames) -> (video_id, clip_id), built once from the
    # committed mapping file.
    id_lookup = {}
    ambiguous_keys = set()
    if landmark_mapping_csv is not None:
        mapping_df = pd.read_csv(landmark_mapping_csv, dtype={"clip_id": str}) # treat clip_id as string
        dup_mask = mapping_df.duplicated(subset=["transcript", "n_frames"], keep=False)
        num_ambiguous = dup_mask.sum()
        if num_ambiguous:
            logger.warning(
                "%d rows in landmark_mapping_csv share both transcript and "
                "frame count, thus are excluded from matching.",
                num_ambiguous,
            )
        for _, r in mapping_df[dup_mask].iterrows():
            ambiguous_keys.add((r["transcript"], r["n_frames"]))
        for _, r in mapping_df[~dup_mask].iterrows():
            id_lookup[(r["transcript"], r["n_frames"])] = (r["video_id"], r["clip_id"])

    rows = []
    parquet_index_rows = []
    num_matched = 0
    for example in dataset:
        idx = example["idx"]
        transcript = example["label"].strip().lower()
        n_frames = len(example["video"])

        video_id, clip_id = id_lookup.get((transcript, n_frames), (None, None))

        if video_id is not None:
            sample_id = f"{video_id}_{clip_id}"
            landmark_path = str(landmarks_root / video_id / f"{clip_id}.pkl")
            num_matched += 1
        else:
            sample_id = f"lrs3test_{idx}"
            if landmark_mapping_csv is None:
                landmark_path = ""
            elif (transcript, n_frames) in ambiguous_keys:
                landmark_path = LANDMARK_MARKER_AMBIGUOUS
            else:
                landmark_path = LANDMARK_MARKER_NOT_FOUND

        wav_path = audio_output_dir / f"{sample_id}.wav"
        extract_wav_from_pcm(example["audio"], wav_path)

        rows.append({
            "sample_id": sample_id,
            "video_path": "",
            "audio_path": str(wav_path),
            "landmark_path": landmark_path,
            "transcript": transcript,
            "duration_sec": get_wav_duration_sec(wav_path),
            "source": "lrs3_test",
        })
        parquet_index_rows.append({"sample_id": sample_id, "parquet_idx": idx})

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    logger.info(
        "Built LRS3-test manifest: %d clips (%d with resolved video_id, %d fallback)",
        len(manifest), num_matched, len(manifest) - num_matched,
    )
    manifest = _finalize_manifest(manifest, output_csv, clean)

    if parquet_index_csv is not None:
        # Keep only rows that survived cleaning, so the side table matches
        # the returned/saved manifest exactly.
        parquet_index = pd.DataFrame(parquet_index_rows, columns=PARQUET_INDEX_COLUMNS)
        parquet_index = parquet_index[parquet_index["sample_id"].isin(manifest["sample_id"])]
        _maybe_write_csv(parquet_index, parquet_index_csv)
        logger.info("Wrote LRS3-test parquet index (%d rows) to %s", len(parquet_index), parquet_index_csv)
    return manifest


def build_grid_manifest(
    grid_root: PathLike,
    landmarks_root: PathLike,
    audio_output_dir: PathLike,
    output_csv: Optional[PathLike] = None,
    limit: Optional[int] = None,
    clean: bool = True,
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
            ``/scratch/your_project_name/datasets/kaggle_lipnet/datasets/
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
        clean: If True (default) and ``output_csv`` is given, drop rows
            with missing files or landmark frame-count mismatches (see
            ``clean_manifest``) before writing the CSV, so the returned
            DataFrame matches the saved CSV exactly. Dropped sample IDs
            are logged next to ``output_csv``. Ignored without
            ``output_csv``.

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
    return _finalize_manifest(manifest, output_csv, clean)


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
    ``video_path``/``landmark_path`` on lrs3_test rows) are
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
        if not Path(landmark_path).is_file():
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
    """Read a video file's frame rate via torchcodec.

    Args:
        video_path: Path to a video file.

    Returns:
        The video's frame rate, in frames per second, as a float..
    """
    from torchcodec.decoders import VideoDecoder
    decoder = VideoDecoder(str(video_path))
    return float(decoder.metadata.average_fps)


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


def clean_manifest(
    manifest: pd.DataFrame,
    tolerance_frames: int = 3,
    log_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Drop manifest rows with missing files or anomalous frame counts.

    Does two independent checks: missing files (corrupted source videos, like
    GRID's ``s8_processed``) and frame-count differences beyond the requested
    tolerance. Rows without a landmark path, such as unresolved lrs3_test
    rows, are skipped by the frame-count check.

    Args:
        manifest: A manifest DataFrame produced by one of the manifest
            builders. It must contain ``sample_id``, the path columns used by
            ``check_file_existence``, and ``landmark_path``/``duration_sec``
            for frame-count validation.
        tolerance_frames: Maximum allowed absolute difference between the
            landmark frame count and the expected count before a row is
            dropped. Defaults to 3 frames.
        log_path: Optional path to an append-only log file. Dropped sample
            IDs are written here for traceability.

    Returns:
        A copy of ``manifest`` containing only rows that pass both checks,
        with the index reset. The input DataFrame is not modified.
    """
    existence_problems = check_file_existence(manifest)
    frame_problems = check_frame_count_vs_duration(manifest, tolerance_frames=tolerance_frames)
    bad_ids = set(existence_problems["sample_id"]) | set(frame_problems["sample_id"])

    if bad_ids and log_path is not None:
        with open(log_path, "a") as f:
            for sid in sorted(bad_ids):
                missing = existence_problems[existence_problems["sample_id"] == sid]
                reasons = [f"{c}={p}" for c, p in zip(missing["column"], missing["path"])]
                if sid in set(frame_problems["sample_id"]):
                    reasons.append("landmark frame count mismatch")
                f.write(f"{sid}: dropped ({'; '.join(reasons)})\n")

    cleaned = manifest[~manifest["sample_id"].isin(bad_ids)].reset_index(drop=True)
    logger.info("Cleaned manifest: %d -> %d rows (%d dropped)", len(manifest), len(cleaned), len(bad_ids))
    return cleaned

def filter_grid_word_segments(
    word_segments: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Keep word segments whose parent clips are present in a manifest.

    This performs a semi-join on ``sample_id``. It does not check files or
    frame counts itself, so pass the result of ``clean_manifest`` when
    dropped clips must also be removed from the word-segment table.

    Args:
        word_segments: GRID word-segment table produced by
            ``build_grid_word_segments``. It must contain a ``sample_id`` column.
        manifest: Manifest whose ``sample_id`` values define the clips to
            keep. This should normally be a cleaned GRID manifest.

    Returns:
        A copy of ``word_segments`` containing only rows whose ``sample_id``
        occurs in ``manifest``, with the index reset. The input DataFrames
        are not modified.
    """
    valid_ids = set(manifest["sample_id"])
    return word_segments[word_segments["sample_id"].isin(valid_ids)].reset_index(drop=True)
