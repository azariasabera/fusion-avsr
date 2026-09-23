"""Manifest builders for GRID and LRS3 (trainval and test).

Turns each dataset's own raw, source-specific directory/file layout into
ONE consistent per-clip manifest schema. Downstream code (encoders,
fusion, training loops) should only ever need to read a manifest CSV --
it should never need to know GRID's raw layout differs from
LRS3's.

Per-clip manifest schema (one row per clip, one manifest per
dataset+split):

    sample_id       unique clip identifier
    video_path      path to the raw video file
    audio_path      path to the extracted 16kHz mono .wav (always a real
                    file, for every source)
    landmark_path   path to the 68-point .pkl landmark file
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

import json
import pickle
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from fusion_avsr.data.audio_extraction import (
    get_extracted_wav_path,
    get_wav_duration_sec,
)
from fusion_avsr.utils.logging import get_logger
from fusion_avsr.utils.video import compute_real_decodable_frame_counts

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


def _serialize_params(params: dict) -> dict:
    """Make a builder_kwargs dict JSON-safe (Path -> str), for the sidecar params file."""
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in params.items()}


def load_or_build_manifest(
    csv_path: PathLike,
    builder_fn: Callable[..., pd.DataFrame],
    force_rebuild: bool = False,
    **builder_kwargs: object,
) -> pd.DataFrame:
    """Load a cached manifest CSV, building and caching it on first use.

    Manifests are expensive to build (walking tens of thousands of raw
    dataset files) but cheap to load once built. This makes that
    build-once-then-cache pattern uniform across every caller: the first
    call for a given ``csv_path`` builds the manifest and saves it there;
    every later call, from any script or notebook, just loads the saved
    CSV instead of rebuilding it.

    The exact ``builder_kwargs`` used to build the cached CSV are also
    recorded, alongside it, in a ``<name>.params.json`` sidecar file.

    Args:
        csv_path: Path to the manifest CSV, conventionally
            ``fusion_avsr.data.paths.MANIFEST_DIR / "<name>.csv"``.
        builder_fn: One of this module's ``build_*`` functions. Called as
            ``builder_fn(**builder_kwargs, output_csv=csv_path)`` -- it
            must accept an ``output_csv`` keyword argument and both build
            and save the manifest in one call, exactly like every
            ``build_*_manifest``/``build_grid_word_segments`` function in
            this module already does.
        force_rebuild: If True, rebuild and overwrite the cached CSV (and
            its params sidecar) even if it already exists.
        **builder_kwargs: Forwarded to ``builder_fn``, alongside
            ``output_csv``.

    Returns:
        The manifest DataFrame, either loaded from ``csv_path`` or freshly
        built (and now cached at ``csv_path`` for next time).

    Raises:
        ValueError: If a cached manifest exists whose recorded params
            sidecar disagrees with the parameters this call is requesting.
    """
    csv_path = Path(csv_path)
    params_path = csv_path.with_name(f"{csv_path.stem}.params.json")
    requested_params = _serialize_params(builder_kwargs)

    if csv_path.exists() and not force_rebuild:
        if params_path.exists():
            with open(params_path, "r", encoding="utf-8") as f:
                recorded_params = json.load(f)
            if recorded_params != requested_params:
                message = (
                    f"Cached manifest at {csv_path} was built with different parameters than "
                    f"this call is requesting, please compare the difference below:\n"
                    f"  recorded:  {recorded_params}\n  requested: {requested_params}\n"
                    f"Pass force_rebuild=True (or force_rebuild_manifest=True for pretrain_landmark_grid.py), "
                    f"or delete {csv_path} and {params_path}, to rebuild."
                )
                logger.error(message)
                raise ValueError(message)
        else:
            logger.warning(
                "Cached manifest at %s has no %s sidecar to verify parameters against "
                "(likely built before this check existed) -- trusting it as-is.",
                csv_path, params_path,
            )
        logger.info("Loading cached manifest from %s", csv_path)
        return pd.read_csv(csv_path)

    logger.info(
        "No cached manifest at %s (or force_rebuild=True) -- building it via %s",
        csv_path, builder_fn.__name__,
    )
    manifest = builder_fn(**builder_kwargs, output_csv=csv_path)

    params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(requested_params, f, indent=2, sort_keys=True)

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


def _select_grid_clips_round_robin(
    speaker_dirs: List[Path],
    limit: Optional[int],
) -> List[Path]:
    """Select up to ``limit`` GRID clips, round-robin across speakers.

    One pass through all speakers takes each speaker's NEXT unused clip.
    This way, smaller limits still span multiple speakers instead of exhausting one
    speaker's ~1000 clips before ever reaching a second.

    Args:
        speaker_dirs: GRID speaker directories (e.g.
            ``<grid_root>/s10_processed``), in the order round-robin
            visits them each pass.
        limit: Maximum number of clips to select. ``None`` selects every
            clip from every speaker.

    Returns:
        A list of selected ``.mpg`` paths, in round-robin selection
        order. Fewer than ``limit`` if every speaker's clips are
        exhausted first.
    """
    if limit is None:
        return [p for d in speaker_dirs for p in sorted(d.glob("*.mpg"))]

    clips_by_speaker = {d: sorted(d.glob("*.mpg")) for d in speaker_dirs}
    cursor = {d: 0 for d in speaker_dirs}

    selected = []
    while len(selected) < limit:
        made_progress = False
        for d in speaker_dirs:
            if len(selected) >= limit:
                break
            if cursor[d] < len(clips_by_speaker[d]):
                selected.append(clips_by_speaker[d][cursor[d]])
                cursor[d] += 1
                made_progress = True
        if not made_progress:
            break  # every speaker's clips exhausted before reaching limit
    return selected


def _select_lrs3_clips_round_robin(
    video_dirs: List[Path],
    limit: Optional[int],
) -> List[Path]:
    """Select up to ``limit`` LRS3 clips, round-robin across video IDs.

    Same idea as ``_select_grid_clips_round_robin``, grouped by video_id
    instead of speaker: one pass takes each video's NEXT unused clip
    (sorted order, never repeated), so a limit spans multiple videos
    instead of exhausting one video's clips before ever reaching the next.

    Args:
        video_dirs: LRS3 video-id directories (e.g.
            ``<video_root>/<video_id>``), in the order round-robin
            visits them each pass.
        limit: Maximum number of clips to select. ``None`` selects every
            clip from every video.

    Returns:
        A list of selected ``.mp4`` paths, in round-robin selection
        order. Fewer than ``limit`` if every video's clips are exhausted
        first.
    """
    if limit is None:
        return [p for d in video_dirs for p in sorted(d.glob("*.mp4"))]

    clips_by_video = {d: sorted(d.glob("*.mp4")) for d in video_dirs}
    cursor = {d: 0 for d in video_dirs}

    selected = []
    while len(selected) < limit:
        made_progress = False
        for d in video_dirs:
            if len(selected) >= limit:
                break
            if cursor[d] < len(clips_by_video[d]):
                selected.append(clips_by_video[d][cursor[d]])
                cursor[d] += 1
                made_progress = True
        if not made_progress:
            break  # every video's clips exhausted before reaching limit
    return selected


LRS3_SOURCES = ("lrs3_trainval", "lrs3_test")


def build_lrs3_manifest(
    video_root: PathLike,
    audio_output_dir: PathLike,
    landmarks_root: PathLike,
    source: str,
    output_csv: Optional[PathLike] = None,
    limit: Optional[int] = None,
    clean: bool = True,
) -> pd.DataFrame:
    """Build the per-clip manifest for one LRS3 split (trainval or test).

    Both splits share the same layout: ``<video_root>/<video_id>/<clip_id>.mp4``
    with a matching ``.txt`` transcript (``Text:``/``Conf:`` format).
    Landmark paths are resolved under
    ``<landmarks_root>/<video_id>/<clip_id>.pkl``. Audio is NOT extracted
    here -- ``scripts/extract_audio.sh`` must be run against this split
    first (see that script's docstring); this function only resolves the
    ``.wav`` path it wrote and fails clearly if that has not happened yet.

    Args:
        video_root: Path to the split's directory containing one
            ``<video_id>`` directory per source video, for example
            ``/scratch/your_project_name/datasets/lrs3/ainncy/trainval`` or
            ``/scratch/your_project_name/datasets/lrs3/test``.
        audio_output_dir: Directory that ``scripts/extract_audio.sh`` was
            told to write ``.wav`` files into. One ``.wav`` per clip is
            expected at ``<audio_output_dir>/<video_id>_<clip_id>.wav``.
        landmarks_root: Root directory containing the split's landmark
            files, for example
            ``/scratch/your_project_name/datasets/lrs3/landmarks/LRS3_landmarks/trainval``.
        source: Value written to the manifest's ``source`` column; one of
            ``"lrs3_trainval"`` or ``"lrs3_test"``.
        output_csv: If given, the resulting manifest is also written to
            this path as a CSV file.
        limit: If given, select up to this many clips round-robin across
            video IDs (see ``_select_lrs3_clips_round_robin``), instead
            of walking the entire split in sorted video_id/clip_id
            order.
        clean: If True (default) and ``output_csv`` is given, drop rows
            with missing files or landmark frame-count mismatches (see
            ``clean_manifest``) before writing the CSV, so the returned
            DataFrame matches the saved CSV exactly. Dropped sample IDs
            are logged next to ``output_csv``. Ignored without
            ``output_csv``.

    Returns:
        A DataFrame with the columns listed in ``MANIFEST_COLUMNS``, one
        row per clip, with ``source`` set to the given ``source``.

    Raises:
        ValueError: If ``source`` is not one of ``LRS3_SOURCES``.
    """
    if source not in LRS3_SOURCES:
        message = f"source must be one of {LRS3_SOURCES}, got {source!r}"
        logger.error(message)
        raise ValueError(message)

    video_root = Path(video_root)
    audio_output_dir = Path(audio_output_dir)
    landmarks_root = Path(landmarks_root)

    video_dirs = sorted(p for p in video_root.iterdir() if p.is_dir())
    selected_mp4_paths = _select_lrs3_clips_round_robin(video_dirs, limit)

    logger.info("Building %s manifest from %s (limit=%s)", source, video_root, limit)
    rows = []
    for mp4_path in selected_mp4_paths:
        video_dir = mp4_path.parent
        video_id = video_dir.name
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
            "source": source,
        })

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    logger.info("Built %s manifest: %d clips", source, len(manifest))
    return _finalize_manifest(manifest, output_csv, clean)


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
    run against GRID first; this function only resolves the ``.wav`` path
    it wrote and fails clearly if that has not happened yet. Landmarks are
    also not generated here -- this function expects a landmark ``.pkl`` file
    to already exist at ``<landmarks_root>/s<N>_processed/<clip>.pkl``.

    This is NOT what the word-recognition pretext task actually trains on.
    That task trains on individual WORD segments, produced by
    ``build_grid_word_segments`` instead.

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
        limit: If given, select up to this many clips round-robin across
            speakers (see ``_select_grid_clips_round_robin``), instead of
            walking all 33 speakers in sorted order.
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

    speaker_dirs = sorted(p for p in grid_root.iterdir() if p.is_dir())
    selected_mpg_paths = _select_grid_clips_round_robin(speaker_dirs, limit)

    logger.info("Building GRID manifest from %s (limit=%s)", grid_root, limit)
    rows = []
    for mpg_path in selected_mpg_paths:
        speaker_dir = mpg_path.parent
        speaker = speaker_dir.name.replace("_processed", "")
        clip_id = mpg_path.stem
        align_path = speaker_dir / "align" / f"{clip_id}.align"
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


def _clamp_word_segments(
    word_segments: pd.DataFrame,
    frame_counts: Dict[str, int],
    reason: str,
    log_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Clamp each segment's ``end_frame`` to its parent clip's entry 
    in ``frame_counts``, dropping empties.

    Args:
        word_segments: A word-segment table with ``sample_id``, ``word``,
            ``start_frame``, ``end_frame`` columns.
        frame_counts: ``sample_id -> the max frame index (exclusive) this
            clip actually supports``.
        reason: Short label identifying which count this clamp was
            against (e.g. ``"real_frame_count"`` or
            ``"landmark_frame_count"``), included in dropped-row log
            lines and the summary ``logger.warning``.
        log_path: Optional append-only log file for dropped rows.

    Returns:
        A copy of ``word_segments``, ``end_frame`` clamped, rows left
        with no usable frames removed, index reset.
    """
    max_frames = word_segments["sample_id"].map(frame_counts)
    max_frames = max_frames.fillna(word_segments["end_frame"])
    clamped_end_frame = word_segments["end_frame"].where(
        word_segments["end_frame"] <= max_frames, max_frames
    ).astype(int)
    keep_mask = word_segments["start_frame"] < clamped_end_frame

    if (~keep_mask).any():
        dropped = word_segments[~keep_mask]
        dropped_ends = clamped_end_frame[~keep_mask]
        dropped_lines = [
            f"{row.sample_id}: dropped word={row.word!r} (start_frame={row.start_frame} >= "
            f"clamped end_frame={end}, {reason}={frame_counts.get(row.sample_id)})\n"
            for row, end in zip(dropped.itertuples(), dropped_ends)
        ]
        logger.warning("Dropped %d word segment(s) clamping against %s", len(dropped_lines), reason)
        if log_path is not None:
            with open(log_path, "a") as f:
                f.writelines(dropped_lines)

    result = word_segments[keep_mask].copy()
    result["end_frame"] = clamped_end_frame[keep_mask]
    return result.reset_index(drop=True)


def build_grid_word_segments(
    grid_root: PathLike,
    output_csv: Optional[PathLike] = None,
    limit: Optional[int] = None,
    log_path: Optional[PathLike] = None,
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
        limit: If given, select up to this many CLIPS round-robin across
            speakers (see ``_select_grid_clips_round_robin``).
        log_path: Optional path to an append-only log file. Dropped word
            segments (and any clip that failed to decode at all -- see
            ``compute_real_decodable_frame_counts``) are logged here. If
            not given but ``output_csv`` is, a
            ``<output_csv stem>_dropped_segments.log`` path next to it is
            used instead.

    Returns:
        A DataFrame with columns ``sample_id``, ``word``, ``start_frame``,
        ``end_frame`` (see ``WORD_SEGMENT_COLUMNS``), one row per word
        segment, across every GRID clip. ``sample_id`` matches the parent
        clip's ``sample_id`` in the per-clip GRID manifest, so the two
        tables can be joined.
    """
    grid_root = Path(grid_root)

    if log_path is None and output_csv is not None:
        output_csv_path = Path(output_csv)
        log_path = output_csv_path.with_name(f"{output_csv_path.stem}_dropped_segments.log")

    speaker_dirs = sorted(p for p in grid_root.iterdir() if p.is_dir())
    selected_mpg_paths = _select_grid_clips_round_robin(speaker_dirs, limit)

    clip_sample_ids = []
    clip_video_paths = []
    for mpg_path in selected_mpg_paths:
        speaker = mpg_path.parent.name.replace("_processed", "")
        clip_sample_ids.append(f"{speaker}_{mpg_path.stem}")
        clip_video_paths.append(str(mpg_path))
    clip_manifest = pd.DataFrame({"sample_id": clip_sample_ids, "video_path": clip_video_paths})

    logger.info(
        "Decoding %d GRID clip(s) once to find their real frame counts...", len(clip_manifest)
    )
    real_frame_counts = compute_real_decodable_frame_counts(clip_manifest, log_path=log_path)

    logger.info("Building GRID word-segment table from %s (limit=%s)", grid_root, limit)
    rows = []
    for mpg_path in selected_mpg_paths:
        speaker_dir = mpg_path.parent
        speaker = speaker_dir.name.replace("_processed", "")
        clip_id = mpg_path.stem
        align_path = speaker_dir / "align" / f"{clip_id}.align"
        sample_id = f"{speaker}_{clip_id}"

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
    word_segments = _clamp_word_segments(
        word_segments, real_frame_counts, reason="real_frame_count", log_path=log_path
    )
    logger.info("Built GRID word-segment table: %d word segments", len(word_segments))
    _maybe_write_csv(word_segments, output_csv)
    return word_segments


def check_file_existence(
    manifest: pd.DataFrame,
    path_columns: Tuple[str, ...] = ("video_path", "audio_path", "landmark_path"),
) -> pd.DataFrame:
    """Check that every non-empty path referenced by a manifest actually exists on disk.

    Empty/missing path values (``""`` or ``None``) are skipped, since
    those are expected to be absent -- this function only flags paths that
    SHOULD point to a real file but don't.

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
    tolerance. Rows without a landmark path are skipped by the frame-count
    check.

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

def compute_landmark_frame_counts(
    manifest: pd.DataFrame,
    log_path: Optional[PathLike] = None,
) -> Dict[str, int]:
    """Load each clip's landmark ``.pkl`` file once and record its real frame count.

    The landmark-side counterpart to
    ``fusion_avsr.utils.video.compute_real_decodable_frame_counts``: a
    clip's landmark file can be shorter than its ``.align``-implied
    duration too, independently of whether the video itself decodes
    fully -- checked here rather than in ``build_grid_word_segments``,
    since ``landmark_path`` is only available once joined against a
    per-clip manifest (see ``filter_grid_word_segments``).

    Args:
        manifest: A per-clip manifest-shaped DataFrame with ``sample_id``
            and ``landmark_path`` columns.
        log_path: Optional path to an append-only log file. Clips whose
            landmark file fails to load are logged here and given a
            count of 0, rather than crashing this whole batch pass over
            one bad file.

    Returns:
        A dict mapping each ``sample_id`` to the length of its landmark
        list.
    """
    counts: Dict[str, int] = {}
    failed_lines = []
    for row in manifest.itertuples():
        try:
            with open(row.landmark_path, "rb") as f:
                counts[row.sample_id] = len(pickle.load(f))
        except Exception as e:
            counts[row.sample_id] = 0
            failed_lines.append(f"{row.sample_id}: failed to load landmark file {row.landmark_path} ({e})\n")

    if failed_lines:
        logger.warning("compute_landmark_frame_counts: %d clip(s) failed to load", len(failed_lines))
        if log_path is not None:
            with open(log_path, "a") as f:
                f.writelines(failed_lines)

    return counts


def filter_grid_word_segments(
    word_segments: pd.DataFrame,
    manifest: pd.DataFrame,
    log_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Keep word segments whose parent clips are present in a manifest, 
    clamped to real landmark length.

    Two things, both scoped to what ``manifest`` (not ``word_segments``)
    knows about a clip: (1) a semi-join on ``sample_id``, dropping
    segments whose clip isn't in ``manifest`` (e.g. dropped by
    ``clean_manifest``); (2) clamping each remaining segment's
    ``end_frame`` to its parent clip's actual landmark length (see
    ``compute_landmark_frame_counts``) via ``_clamp_word_segments``.

    Args:
        word_segments: GRID word-segment table produced by
            ``build_grid_word_segments``. It must contain a
            ``sample_id`` column.
        manifest: Manifest whose ``sample_id`` values define the clips to
            keep, with a ``landmark_path`` column. This should normally
            be a cleaned GRID manifest.
        log_path: Optional path to an append-only log file. Segments
            dropped by the landmark-length clamp (and clips whose
            landmark file fails to load) are logged here.

    Returns:
        A copy of ``word_segments`` containing only rows whose
        ``sample_id`` occurs in ``manifest`` and whose
        ``[start_frame, end_frame)`` still fits within the parent clip's
        real landmark count, with the index reset. The input DataFrames
        are not modified.
    """
    valid_ids = set(manifest["sample_id"])
    kept = word_segments[word_segments["sample_id"].isin(valid_ids)].reset_index(drop=True)
    if kept.empty:
        return kept

    clip_manifest = manifest[manifest["sample_id"].isin(set(kept["sample_id"]))]
    landmark_frame_counts = compute_landmark_frame_counts(clip_manifest, log_path=log_path)
    return _clamp_word_segments(kept, landmark_frame_counts, reason="landmark_frame_count", log_path=log_path)
