"""Unit tests for fusion_avsr.data.manifest_builder.

Uses small synthetic directory trees under pytest's tmp_path fixture --
no real GRID/LRS3 data is needed. Video/landmark files are empty
placeholders (their contents are never read by the manifest builders,
only their paths are recorded); audio files are real, short synthetic
.wav files written with soundfile, so duration_sec can be computed for
real.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import fusion_avsr.data.manifest_builder as manifest_builder
from fusion_avsr.data.manifest_builder import (
    GRID_NON_WORD_TOKENS,
    LANDMARK_MARKER_AMBIGUOUS,
    LANDMARK_MARKER_NOT_FOUND,
    _align_units_to_frame,
    _parse_grid_align,
    _parse_lrs3_transcript,
    build_grid_manifest,
    build_grid_word_segments,
    build_lrs3_test_manifest,
    build_lrs3_trainval_manifest,
    check_file_existence,
    check_frame_count_vs_duration,
    check_video_fps,
)

SAMPLE_RATE = 16000


def _write_silence_wav(path, duration_sec: float, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a real .wav file of silence, for tests that need a real duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(round(duration_sec * sample_rate))
    samples = np.zeros(num_samples, dtype=np.int16)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")


def _write_grid_align(path, rows) -> None:
    """Write a GRID .align file from a list of (start_units, end_units, word) tuples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for start_units, end_units, word in rows:
            f.write(f"{start_units} {end_units} {word}\n")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def test_parse_lrs3_transcript_strips_prefix_and_lowercases(tmp_path):
    txt_path = tmp_path / "clip.txt"
    txt_path.write_text("Text:  HELLO WORLD\nConf:  1.0\n", encoding="utf-8")

    assert _parse_lrs3_transcript(txt_path) == "hello world"


def test_parse_lrs3_transcript_raises_if_no_text_line(tmp_path):
    txt_path = tmp_path / "clip.txt"
    txt_path.write_text("Conf:  1.0\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _parse_lrs3_transcript(txt_path)


def test_parse_grid_align_reads_all_rows_including_sil_sp(tmp_path):
    align_path = tmp_path / "bbaf2n.align"
    _write_grid_align(align_path, [
        (0, 34400, "sil"),
        (34400, 39800, "place"),
        (39800, 44000, "blue"),
        (44000, 47500, "sp"),
    ])

    rows = _parse_grid_align(align_path)

    assert rows == [
        (0, 34400, "sil"),
        (34400, 39800, "place"),
        (39800, 44000, "blue"),
        (44000, 47500, "sp"),
    ]


@pytest.mark.parametrize("units,expected_frame", [
    (0, 0),
    (25000, 25),       # exactly 1 second -> 25 frames at 25fps
    (12500, 13),       # 0.5s -> 12.5 frames, rounds to 13
    (74500, 75),       # matches the 74500/25000 = 2.98s example from the raw data
])
def test_align_units_to_frame_conversion(units, expected_frame):
    assert _align_units_to_frame(units) == expected_frame


# ---------------------------------------------------------------------------
# build_grid_word_segments
# ---------------------------------------------------------------------------

def test_build_grid_word_segments_excludes_sil_and_sp(tmp_path):
    grid_root = tmp_path / "grid"
    _write_grid_align(grid_root / "s1_processed" / "align" / "bbaf2n.align", [
        (0, 25000, "sil"),
        (25000, 50000, "place"),
        (50000, 62500, "blue"),
        (62500, 65000, "sp"),
        (65000, 90000, "again"),
    ])

    word_segments = build_grid_word_segments(grid_root)

    assert list(word_segments["word"]) == ["place", "blue", "again"]
    assert not set(word_segments["word"]) & GRID_NON_WORD_TOKENS


def test_build_grid_word_segments_frame_conversion_and_sample_id(tmp_path):
    grid_root = tmp_path / "grid"
    _write_grid_align(grid_root / "s3_processed" / "align" / "clip1.align", [
        (25000, 50000, "lay"),
    ])

    word_segments = build_grid_word_segments(grid_root)

    assert len(word_segments) == 1
    row = word_segments.iloc[0]
    assert row["sample_id"] == "s3_clip1"
    assert row["start_frame"] == 25
    assert row["end_frame"] == 50


def test_build_grid_word_segments_limit_caps_number_of_clips(tmp_path):
    grid_root = tmp_path / "grid"
    _write_grid_align(grid_root / "s1_processed" / "align" / "clip1.align", [(0, 25000, "bin")])
    _write_grid_align(grid_root / "s1_processed" / "align" / "clip2.align", [(0, 25000, "lay")])

    word_segments = build_grid_word_segments(grid_root, limit=1)

    assert len(word_segments) == 1
    assert word_segments.iloc[0]["sample_id"] == "s1_clip1"


def test_build_grid_word_segments_writes_csv_when_requested(tmp_path):
    grid_root = tmp_path / "grid"
    _write_grid_align(grid_root / "s1_processed" / "align" / "clip1.align", [
        (0, 25000, "bin"),
    ])
    output_csv = tmp_path / "manifests" / "grid_word_segments.csv"

    build_grid_word_segments(grid_root, output_csv=output_csv)

    assert output_csv.exists()
    reloaded = pd.read_csv(output_csv)
    assert list(reloaded.columns) == ["sample_id", "word", "start_frame", "end_frame"]


# ---------------------------------------------------------------------------
# build_grid_manifest
# ---------------------------------------------------------------------------

def test_build_grid_manifest_one_clip(tmp_path):
    grid_root = tmp_path / "grid"
    landmarks_root = tmp_path / "grid_landmarks"
    audio_output_dir = tmp_path / "audio"

    speaker_dir = grid_root / "s1_processed"
    (speaker_dir).mkdir(parents=True)
    (speaker_dir / "bbaf2n.mpg").write_bytes(b"")  # content never read
    _write_grid_align(speaker_dir / "align" / "bbaf2n.align", [
        (0, 25000, "sil"),
        (25000, 50000, "bin"),
        (50000, 75000, "blue"),
    ])
    _write_silence_wav(audio_output_dir / "s1_bbaf2n.wav", duration_sec=3.0)

    manifest = build_grid_manifest(grid_root, landmarks_root, audio_output_dir)

    assert len(manifest) == 1
    row = manifest.iloc[0]
    assert row["sample_id"] == "s1_bbaf2n"
    assert row["source"] == "grid"
    assert row["transcript"] == "bin blue"  # sil excluded
    assert row["duration_sec"] == pytest.approx(3.0, abs=0.01)
    assert row["landmark_path"] == str(landmarks_root / "s1_processed" / "bbaf2n.pkl")


def test_build_grid_manifest_limit_caps_number_of_clips(tmp_path):
    grid_root = tmp_path / "grid"
    landmarks_root = tmp_path / "grid_landmarks"
    audio_output_dir = tmp_path / "audio"

    for speaker, clip in [("s1_processed", "clip1"), ("s1_processed", "clip2")]:
        speaker_dir = grid_root / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        (speaker_dir / f"{clip}.mpg").write_bytes(b"")
        _write_grid_align(speaker_dir / "align" / f"{clip}.align", [(0, 25000, "bin")])
        _write_silence_wav(audio_output_dir / f"s1_{clip}.wav", duration_sec=1.0)

    manifest = build_grid_manifest(grid_root, landmarks_root, audio_output_dir, limit=1)

    assert len(manifest) == 1


def test_build_grid_manifest_raises_if_audio_not_extracted(tmp_path):
    grid_root = tmp_path / "grid"
    landmarks_root = tmp_path / "grid_landmarks"
    audio_output_dir = tmp_path / "audio"  # left empty: extract_audio.sh "not run yet"

    speaker_dir = grid_root / "s1_processed"
    speaker_dir.mkdir(parents=True)
    (speaker_dir / "bbaf2n.mpg").write_bytes(b"")
    _write_grid_align(speaker_dir / "align" / "bbaf2n.align", [(0, 25000, "bin")])

    with pytest.raises(FileNotFoundError):
        build_grid_manifest(grid_root, landmarks_root, audio_output_dir)


# ---------------------------------------------------------------------------
# build_lrs3_trainval_manifest
# ---------------------------------------------------------------------------

def test_build_lrs3_trainval_manifest_one_clip(tmp_path):
    lrs3_root = tmp_path / "lrs3"
    audio_output_dir = tmp_path / "audio"

    video_dir = lrs3_root / "ainncy" / "trainval" / "abc123"
    video_dir.mkdir(parents=True)
    (video_dir / "00001.mp4").write_bytes(b"")  # content never read
    (video_dir / "00001.txt").write_text("Text:  HELLO THERE\nConf:  0.99\n", encoding="utf-8")
    _write_silence_wav(audio_output_dir / "abc123_00001.wav", duration_sec=2.0)

    video_root = lrs3_root / "ainncy" / "trainval"
    landmarks_root = lrs3_root / "landmarks" / "LRS3_landmarks" / "trainval"
    manifest = build_lrs3_trainval_manifest(video_root, audio_output_dir, landmarks_root)

    assert len(manifest) == 1
    row = manifest.iloc[0]
    assert row["sample_id"] == "abc123_00001"
    assert row["source"] == "lrs3_trainval"
    assert row["transcript"] == "hello there"
    assert row["duration_sec"] == pytest.approx(2.0, abs=0.01)
    assert row["landmark_path"] == str(
        lrs3_root / "landmarks" / "LRS3_landmarks" / "trainval" / "abc123" / "00001.pkl"
    )


# ---------------------------------------------------------------------------
# build_lrs3_test_manifest
# ---------------------------------------------------------------------------

def _write_lrs3_test_parquet(parquet_dir, labels, n_frames=25):
    """Write a tiny lrs3_test-style parquet (idx, audio, video, label) for tests."""
    hf_datasets = pytest.importorskip("datasets")
    pcm = list(np.zeros(SAMPLE_RATE, dtype="int16"))  # 1 second of silence at 16kHz
    video = [[[0]] for _ in range(n_frames)]  # placeholder frames, only len() is read
    dataset = hf_datasets.Dataset.from_dict({
        "idx": list(range(len(labels))),
        "audio": [pcm] * len(labels),
        "video": [video] * len(labels),
        "label": labels,
    })
    parquet_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(parquet_dir / "data.parquet"))


def test_build_lrs3_test_manifest_without_mapping_uses_fallback_ids(tmp_path):
    parquet_dir = tmp_path / "parquet"
    _write_lrs3_test_parquet(parquet_dir, ["HELLO WORLD", "how are you"])

    manifest = build_lrs3_test_manifest(
        parquet_dir, tmp_path / "audio", landmarks_root=tmp_path / "landmarks",
    )

    assert len(manifest) == 2
    first_row = manifest.iloc[0]
    assert first_row["sample_id"] == "lrs3test_0"
    assert first_row["source"] == "lrs3_test"
    assert first_row["video_path"] == ""
    assert first_row["landmark_path"] == ""
    assert first_row["transcript"] == "hello world"
    assert first_row["duration_sec"] == pytest.approx(1.0, abs=0.01)
    assert Path(first_row["audio_path"]).exists()


def test_build_lrs3_test_manifest_resolves_ids_from_mapping(tmp_path):
    parquet_dir = tmp_path / "parquet"
    _write_lrs3_test_parquet(parquet_dir, ["hello world", "how are you"], n_frames=25)
    mapping_csv = tmp_path / "mapping.csv"
    mapping_csv.write_text(
        "video_id,clip_id,n_frames,transcript\nvidA,00001,25,hello world\n",
        encoding="utf-8",
    )
    landmarks_root = tmp_path / "landmarks"

    manifest = build_lrs3_test_manifest(
        parquet_dir, tmp_path / "audio", landmarks_root, landmark_mapping_csv=mapping_csv,
    )

    assert list(manifest["sample_id"]) == ["vidA_00001", "lrs3test_1"]
    assert manifest.iloc[0]["landmark_path"] == str(landmarks_root / "vidA" / "00001.pkl")
    assert manifest.iloc[1]["landmark_path"] == LANDMARK_MARKER_NOT_FOUND


def test_build_lrs3_test_manifest_marks_ambiguous_mapping_rows(tmp_path):
    parquet_dir = tmp_path / "parquet"
    _write_lrs3_test_parquet(parquet_dir, ["hello world"], n_frames=25)
    mapping_csv = tmp_path / "mapping.csv"
    mapping_csv.write_text(
        "video_id,clip_id,n_frames,transcript\n"
        "vidA,00001,25,hello world\nvidB,00002,25,hello world\n",
        encoding="utf-8",
    )

    manifest = build_lrs3_test_manifest(
        parquet_dir, tmp_path / "audio", tmp_path / "landmarks", landmark_mapping_csv=mapping_csv,
    )

    assert manifest.iloc[0]["landmark_path"] == LANDMARK_MARKER_AMBIGUOUS


def test_unresolved_rows_dropped_on_save_only_when_mapping_given(tmp_path):
    parquet_dir = tmp_path / "parquet"
    _write_lrs3_test_parquet(parquet_dir, ["hello world"])
    mapping_csv = tmp_path / "mapping.csv"
    mapping_csv.write_text("video_id,clip_id,n_frames,transcript\n", encoding="utf-8")

    without_mapping = build_lrs3_test_manifest(
        parquet_dir, tmp_path / "audio", tmp_path / "landmarks",
        output_csv=tmp_path / "no_mapping" / "lrs3_test.csv",
    )
    with_mapping = build_lrs3_test_manifest(
        parquet_dir, tmp_path / "audio", tmp_path / "landmarks",
        landmark_mapping_csv=mapping_csv, output_csv=tmp_path / "mapping" / "lrs3_test.csv",
    )

    assert len(without_mapping) == 1  # no mapping given: no landmarks expected, row kept
    assert len(with_mapping) == 0  # mapping given but clip unmatched: dropped
    assert LANDMARK_MARKER_NOT_FOUND in (tmp_path / "mapping" / "lrs3_test_dropped.log").read_text()


def test_saved_csv_is_cleaned_and_matches_returned_manifest(tmp_path):
    grid_root = tmp_path / "grid"
    landmarks_root = tmp_path / "grid_landmarks"
    audio_output_dir = tmp_path / "audio"
    speaker_dir = grid_root / "s1_processed"
    speaker_dir.mkdir(parents=True)
    for clip in ["good", "bad"]:
        (speaker_dir / f"{clip}.mpg").write_bytes(b"")
        _write_grid_align(speaker_dir / "align" / f"{clip}.align", [(0, 25000, "bin")])
        _write_silence_wav(audio_output_dir / f"s1_{clip}.wav", duration_sec=1.0)
    (landmarks_root / "s1_processed").mkdir(parents=True)
    with open(landmarks_root / "s1_processed" / "good.pkl", "wb") as f:
        pickle.dump([np.zeros((68, 2), dtype="float32")] * 25, f)  # 1s * 25fps
    # "bad" has no landmark file, so cleaning must drop it

    output_csv = tmp_path / "manifests" / "grid.csv"
    manifest = build_grid_manifest(
        grid_root, landmarks_root, audio_output_dir, output_csv=output_csv,
    )

    saved = pd.read_csv(output_csv)
    assert list(manifest["sample_id"]) == ["s1_good"]
    assert list(saved["sample_id"]) == ["s1_good"]
    assert "s1_bad" in (tmp_path / "manifests" / "grid_dropped.log").read_text()


# ---------------------------------------------------------------------------
# check_file_existence / check_frame_count_vs_duration
# ---------------------------------------------------------------------------

def test_check_file_existence_flags_missing_and_skips_empty(tmp_path):
    existing_video = tmp_path / "clip.mp4"
    existing_video.write_bytes(b"")

    manifest = pd.DataFrame([
        {
            "sample_id": "a",
            "video_path": str(existing_video),
            "audio_path": str(tmp_path / "missing.wav"),
            "landmark_path": "",  # expected empty for e.g. lrs3_test, must be skipped
        },
    ])

    problems = check_file_existence(manifest)

    assert len(problems) == 1
    assert problems.iloc[0]["sample_id"] == "a"
    assert problems.iloc[0]["column"] == "audio_path"


def test_check_frame_count_vs_duration_flags_mismatch(tmp_path):
    good_landmark_path = tmp_path / "good.pkl"
    with open(good_landmark_path, "wb") as f:
        pickle.dump([np.zeros((68, 2), dtype="float32")] * 50, f)  # 50 frames

    bad_landmark_path = tmp_path / "bad.pkl"
    with open(bad_landmark_path, "wb") as f:
        pickle.dump([np.zeros((68, 2), dtype="float32")] * 10, f)  # way too few frames

    manifest = pd.DataFrame([
        {"sample_id": "good", "landmark_path": str(good_landmark_path), "duration_sec": 2.0},  # 2.0*25=50
        {"sample_id": "bad", "landmark_path": str(bad_landmark_path), "duration_sec": 2.0},
    ])

    problems = check_frame_count_vs_duration(manifest)

    assert list(problems["sample_id"]) == ["bad"]
    assert problems.iloc[0]["expected_frames"] == 50
    assert problems.iloc[0]["actual_frames"] == 10


def test_check_frame_count_vs_duration_skips_missing_landmark(tmp_path):
    manifest = pd.DataFrame([
        {
            "sample_id": "missing",
            "landmark_path": str(tmp_path / "missing.pkl"),
            "duration_sec": 2.0,
        },
    ])

    problems = check_frame_count_vs_duration(manifest)

    assert problems.empty


# ---------------------------------------------------------------------------
# check_video_fps
# ---------------------------------------------------------------------------
#
# _probe_video_fps is monkeypatched instead of using a real video file, so
# these tests don't depend on ffprobe/ffmpeg being installed and run fast.

def test_check_video_fps_flags_mismatch(monkeypatch):
    fake_fps_by_path = {"a.mp4": 25.0, "b.mp4": 30.0}
    monkeypatch.setattr(
        manifest_builder, "_probe_video_fps", lambda path: fake_fps_by_path[path]
    )

    manifest = pd.DataFrame([
        {"sample_id": "a", "video_path": "a.mp4"},
        {"sample_id": "b", "video_path": "b.mp4"},
        {"sample_id": "c", "video_path": ""},  # empty video_path must be skipped
    ])

    problems = check_video_fps(manifest, expected_fps=25)

    assert list(problems["sample_id"]) == ["b"]
    assert problems.iloc[0]["expected_fps"] == 25
    assert problems.iloc[0]["actual_fps"] == 30.0


def test_check_video_fps_empty_when_all_match(monkeypatch):
    monkeypatch.setattr(manifest_builder, "_probe_video_fps", lambda path: 25.0)

    manifest = pd.DataFrame([{"sample_id": "a", "video_path": "a.mp4"}])

    problems = check_video_fps(manifest, expected_fps=25)

    assert len(problems) == 0
