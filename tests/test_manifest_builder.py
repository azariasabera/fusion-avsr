"""Unit tests for fusion_avsr.data.manifest_builder.

Uses small synthetic directory trees under pytest's tmp_path fixture --
no real GRID/LRS3 data is needed. Video/landmark files are empty
placeholders (their contents are never read by the manifest builders,
only their paths are recorded); audio files are real, short synthetic
.wav files written with soundfile, so duration_sec can be computed for
real.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import fusion_avsr.data.manifest_builder as manifest_builder
from fusion_avsr.data.manifest_builder import (
    GRID_NON_WORD_TOKENS,
    _align_units_to_frame,
    _parse_grid_align,
    _parse_lrs3_transcript,
    _select_grid_clips_round_robin,
    _select_lrs3_clips_round_robin,
    build_grid_manifest,
    build_grid_word_segments,
    build_lrs3_manifest,
    check_file_existence,
    check_frame_count_vs_duration,
    check_video_fps,
    load_or_build_manifest,
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


def _write_grid_align_with_mpg(align_path, rows) -> None:
    """Write a GRID .align file, plus the matching empty .mpg placeholder next to its speaker dir.

    _select_grid_clips_round_robin (used by both build_grid_manifest and
    build_grid_word_segments) selects clips from the .mpg files, not the
    .align files directly, so a clip needs a .mpg placeholder present to
    be picked up at all -- matching real GRID's layout, where the two
    always exist together.
    """
    _write_grid_align(align_path, rows)
    speaker_dir = align_path.parent.parent  # align_path is <speaker_dir>/align/<clip>.align
    clip_id = align_path.stem
    (speaker_dir / f"{clip_id}.mpg").write_bytes(b"")


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
# _select_grid_clips_round_robin
# ---------------------------------------------------------------------------

def _make_speaker_dirs_with_clips(root, speaker_to_clips):
    """Create <root>/<speaker>/<clip>.mpg for each speaker's list of clip names."""
    speaker_dirs = []
    for speaker, clips in speaker_to_clips.items():
        speaker_dir = root / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        for clip in clips:
            (speaker_dir / f"{clip}.mpg").write_bytes(b"")
        speaker_dirs.append(speaker_dir)
    return sorted(speaker_dirs)


def test_select_grid_clips_round_robin_spans_speakers_before_repeating(tmp_path):
    speaker_dirs = _make_speaker_dirs_with_clips(tmp_path, {
        "s1_processed": ["clip1", "clip2"],
        "s2_processed": ["clip1", "clip2"],
        "s3_processed": ["clip1", "clip2"],
    })

    selected = _select_grid_clips_round_robin(speaker_dirs, limit=3)

    assert [p.parent.name for p in selected] == ["s1_processed", "s2_processed", "s3_processed"]
    assert [p.stem for p in selected] == ["clip1", "clip1", "clip1"]


def test_select_grid_clips_round_robin_wraps_to_second_clip(tmp_path):
    speaker_dirs = _make_speaker_dirs_with_clips(tmp_path, {
        "s1_processed": ["clip1", "clip2"],
        "s2_processed": ["clip1", "clip2"],
    })

    selected = _select_grid_clips_round_robin(speaker_dirs, limit=4)

    assert [(p.parent.name, p.stem) for p in selected] == [
        ("s1_processed", "clip1"), ("s2_processed", "clip1"),
        ("s1_processed", "clip2"), ("s2_processed", "clip2"),
    ]


def test_select_grid_clips_round_robin_stops_when_every_speaker_exhausted(tmp_path):
    speaker_dirs = _make_speaker_dirs_with_clips(tmp_path, {"s1_processed": ["clip1"]})

    selected = _select_grid_clips_round_robin(speaker_dirs, limit=5)

    assert len(selected) == 1


def test_select_grid_clips_round_robin_none_limit_returns_every_clip_sorted(tmp_path):
    speaker_dirs = _make_speaker_dirs_with_clips(tmp_path, {
        "s2_processed": ["clip2", "clip1"],
        "s1_processed": ["clip1"],
    })

    selected = _select_grid_clips_round_robin(speaker_dirs, limit=None)

    assert [(p.parent.name, p.stem) for p in selected] == [
        ("s1_processed", "clip1"), ("s2_processed", "clip1"), ("s2_processed", "clip2"),
    ]


# ---------------------------------------------------------------------------
# build_grid_word_segments
# ---------------------------------------------------------------------------

def test_build_grid_word_segments_excludes_sil_and_sp(tmp_path):
    grid_root = tmp_path / "grid"
    _write_grid_align_with_mpg(grid_root / "s1_processed" / "align" / "bbaf2n.align", [
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
    _write_grid_align_with_mpg(grid_root / "s3_processed" / "align" / "clip1.align", [
        (25000, 50000, "lay"),
    ])

    word_segments = build_grid_word_segments(grid_root)

    assert len(word_segments) == 1
    row = word_segments.iloc[0]
    assert row["sample_id"] == "s3_clip1"
    assert row["start_frame"] == 25
    assert row["end_frame"] == 50


def test_build_grid_word_segments_drops_zero_length_segments_and_logs(tmp_path):
    grid_root = tmp_path / "grid"
    _write_grid_align_with_mpg(grid_root / "s1_processed" / "align" / "clip1.align", [
        (25000, 50000, "bin"),  # 1 full frame -> kept
        (50000, 50500, "b"),    # rounds to start_frame == end_frame -> dropped
    ])
    log_path = tmp_path / "dropped.log"

    word_segments = build_grid_word_segments(grid_root, log_path=log_path)

    assert list(word_segments["word"]) == ["bin"]
    log_text = log_path.read_text()
    assert "s1_clip1" in log_text
    assert "'b'" in log_text


def test_build_grid_word_segments_limit_caps_number_of_clips(tmp_path):
    grid_root = tmp_path / "grid"
    _write_grid_align_with_mpg(grid_root / "s1_processed" / "align" / "clip1.align", [(0, 25000, "bin")])
    _write_grid_align_with_mpg(grid_root / "s1_processed" / "align" / "clip2.align", [(0, 25000, "lay")])

    word_segments = build_grid_word_segments(grid_root, limit=1)

    assert len(word_segments) == 1
    assert word_segments.iloc[0]["sample_id"] == "s1_clip1"


def test_build_grid_word_segments_writes_csv_when_requested(tmp_path):
    grid_root = tmp_path / "grid"
    _write_grid_align_with_mpg(grid_root / "s1_processed" / "align" / "clip1.align", [
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


def test_build_grid_manifest_and_word_segments_reference_same_clips_with_limit(tmp_path):
    grid_root = tmp_path / "grid"
    landmarks_root = tmp_path / "grid_landmarks"
    audio_output_dir = tmp_path / "audio"

    for speaker in ["s1_processed", "s2_processed"]:
        speaker_dir = grid_root / speaker
        for clip in ["clip1", "clip2"]:
            speaker_dir.mkdir(parents=True, exist_ok=True)
            (speaker_dir / f"{clip}.mpg").write_bytes(b"")
            _write_grid_align(speaker_dir / "align" / f"{clip}.align", [(0, 25000, "bin")])
            sample_id = f"{speaker.replace('_processed', '')}_{clip}"
            _write_silence_wav(audio_output_dir / f"{sample_id}.wav", duration_sec=1.0)

    manifest = build_grid_manifest(grid_root, landmarks_root, audio_output_dir, limit=2)
    word_segments = build_grid_word_segments(grid_root, limit=2)

    assert set(manifest["sample_id"]) == set(word_segments["sample_id"]) == {"s1_clip1", "s2_clip1"}


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
# build_lrs3_manifest
# ---------------------------------------------------------------------------

def test_build_lrs3_manifest_one_clip(tmp_path):
    lrs3_root = tmp_path / "lrs3"
    audio_output_dir = tmp_path / "audio"

    video_dir = lrs3_root / "ainncy" / "trainval" / "abc123"
    video_dir.mkdir(parents=True)
    (video_dir / "00001.mp4").write_bytes(b"")  # content never read
    (video_dir / "00001.txt").write_text("Text:  HELLO THERE\nConf:  0.99\n", encoding="utf-8")
    _write_silence_wav(audio_output_dir / "abc123_00001.wav", duration_sec=2.0)

    video_root = lrs3_root / "ainncy" / "trainval"
    landmarks_root = lrs3_root / "landmarks" / "LRS3_landmarks" / "trainval"
    manifest = build_lrs3_manifest(
        video_root, audio_output_dir, landmarks_root, source="lrs3_trainval",
    )

    assert len(manifest) == 1
    row = manifest.iloc[0]
    assert row["sample_id"] == "abc123_00001"
    assert row["source"] == "lrs3_trainval"
    assert row["transcript"] == "hello there"
    assert row["duration_sec"] == pytest.approx(2.0, abs=0.01)
    assert row["landmark_path"] == str(
        lrs3_root / "landmarks" / "LRS3_landmarks" / "trainval" / "abc123" / "00001.pkl"
    )


def test_build_lrs3_manifest_sets_source_for_test_split(tmp_path):
    video_root = tmp_path / "lrs3" / "test"
    audio_output_dir = tmp_path / "audio"
    video_dir = video_root / "vidA"
    video_dir.mkdir(parents=True)
    (video_dir / "00002.mp4").write_bytes(b"")
    (video_dir / "00002.txt").write_text("Text:  GOOD MORNING\nConf:  0.90\n", encoding="utf-8")
    _write_silence_wav(audio_output_dir / "vidA_00002.wav", duration_sec=1.0)

    manifest = build_lrs3_manifest(
        video_root, audio_output_dir, tmp_path / "landmarks", source="lrs3_test",
    )

    assert list(manifest["sample_id"]) == ["vidA_00002"]
    assert manifest.iloc[0]["source"] == "lrs3_test"
    assert manifest.iloc[0]["video_path"] == str(video_dir / "00002.mp4")


def test_build_lrs3_manifest_rejects_unknown_source(tmp_path):
    with pytest.raises(ValueError):
        build_lrs3_manifest(tmp_path, tmp_path, tmp_path, source="lrs3_pretrain")


def _make_video_dirs_with_clips(root, video_to_clips):
    """Create <root>/<video_id>/<clip>.mp4 for each video's list of clip names."""
    video_dirs = []
    for video_id, clips in video_to_clips.items():
        video_dir = root / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        for clip in clips:
            (video_dir / f"{clip}.mp4").write_bytes(b"")
        video_dirs.append(video_dir)
    return sorted(video_dirs)


def test_select_lrs3_clips_round_robin_spans_videos_before_repeating(tmp_path):
    video_dirs = _make_video_dirs_with_clips(tmp_path, {
        "vidA": ["00001", "00002"],
        "vidB": ["00001", "00002"],
    })

    selected = _select_lrs3_clips_round_robin(video_dirs, limit=2)

    assert [(p.parent.name, p.stem) for p in selected] == [("vidA", "00001"), ("vidB", "00001")]


def test_select_lrs3_clips_round_robin_none_limit_returns_every_clip_sorted(tmp_path):
    video_dirs = _make_video_dirs_with_clips(tmp_path, {
        "vidB": ["00002", "00001"],
        "vidA": ["00001"],
    })

    selected = _select_lrs3_clips_round_robin(video_dirs, limit=None)

    assert [(p.parent.name, p.stem) for p in selected] == [
        ("vidA", "00001"), ("vidB", "00001"), ("vidB", "00002"),
    ]


def test_build_lrs3_manifest_limit_spans_multiple_video_ids(tmp_path):
    lrs3_root = tmp_path / "lrs3"
    audio_output_dir = tmp_path / "audio"

    for video_id in ["vidA", "vidB"]:
        video_dir = lrs3_root / video_id
        for clip in ["00001", "00002"]:
            video_dir.mkdir(parents=True, exist_ok=True)
            (video_dir / f"{clip}.mp4").write_bytes(b"")
            (video_dir / f"{clip}.txt").write_text("Text:  HI\nConf:  0.9\n", encoding="utf-8")
            _write_silence_wav(audio_output_dir / f"{video_id}_{clip}.wav", duration_sec=1.0)

    manifest = build_lrs3_manifest(
        lrs3_root, audio_output_dir, tmp_path / "landmarks", source="lrs3_trainval", limit=2,
    )

    # With sorted-first-N truncation this would have been vidA_00001/vidA_00002
    # (one video exhausted before the next); round-robin spans both videos.
    assert set(manifest["sample_id"]) == {"vidA_00001", "vidB_00001"}


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
            "landmark_path": "",  # empty paths must be skipped
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


# ---------------------------------------------------------------------------
# load_or_build_manifest
# ---------------------------------------------------------------------------

def test_load_or_build_manifest_builds_and_caches_on_first_call(tmp_path):
    csv_path = tmp_path / "manifests" / "fake.csv"
    calls = []

    def fake_builder(foo, output_csv):
        calls.append(foo)
        df = pd.DataFrame([{"sample_id": "a", "foo": foo}])
        df.to_csv(output_csv, index=False)
        return df

    result = load_or_build_manifest(csv_path, fake_builder, foo="bar")

    assert calls == ["bar"]
    assert csv_path.exists()
    assert list(result["sample_id"]) == ["a"]


def test_load_or_build_manifest_loads_cache_on_second_call(tmp_path):
    csv_path = tmp_path / "fake.csv"
    calls = []

    def fake_builder(output_csv):
        calls.append(1)
        df = pd.DataFrame([{"sample_id": "a"}])
        df.to_csv(output_csv, index=False)
        return df

    first = load_or_build_manifest(csv_path, fake_builder)
    second = load_or_build_manifest(csv_path, fake_builder)

    assert calls == [1]  # builder only ran once
    pd.testing.assert_frame_equal(first, second)


def test_load_or_build_manifest_force_rebuild_ignores_cache(tmp_path):
    csv_path = tmp_path / "fake.csv"
    calls = []

    def fake_builder(output_csv):
        calls.append(1)
        df = pd.DataFrame([{"sample_id": "a", "n": len(calls)}])
        df.to_csv(output_csv, index=False)
        return df

    load_or_build_manifest(csv_path, fake_builder)
    result = load_or_build_manifest(csv_path, fake_builder, force_rebuild=True)

    assert calls == [1, 1]
    assert result.iloc[0]["n"] == 2


def _fake_manifest_builder(output_csv, **_kwargs):
    df = pd.DataFrame([{"sample_id": "a"}])
    df.to_csv(output_csv, index=False)
    return df


def test_load_or_build_manifest_writes_params_sidecar(tmp_path):
    csv_path = tmp_path / "fake.csv"

    load_or_build_manifest(csv_path, _fake_manifest_builder, limit=5, grid_root=Path("/some/root"))

    params_path = tmp_path / "fake.params.json"
    assert params_path.exists()
    recorded = json.loads(params_path.read_text())
    assert recorded == {"limit": 5, "grid_root": "/some/root"}  # Path serialized as str


def test_load_or_build_manifest_raises_on_param_mismatch(tmp_path):
    csv_path = tmp_path / "fake.csv"
    load_or_build_manifest(csv_path, _fake_manifest_builder, limit=5)

    with pytest.raises(ValueError, match="different parameters"):
        load_or_build_manifest(csv_path, _fake_manifest_builder, limit=10)


def test_load_or_build_manifest_same_params_loads_cache_fine(tmp_path):
    csv_path = tmp_path / "fake.csv"
    load_or_build_manifest(csv_path, _fake_manifest_builder, limit=5)

    result = load_or_build_manifest(csv_path, _fake_manifest_builder, limit=5)

    assert list(result["sample_id"]) == ["a"]


def test_load_or_build_manifest_missing_sidecar_trusts_cache(tmp_path):
    csv_path = tmp_path / "fake.csv"
    pd.DataFrame([{"sample_id": "a"}]).to_csv(csv_path, index=False)  # no sidecar written

    result = load_or_build_manifest(csv_path, _fake_manifest_builder, limit=5)

    assert list(result["sample_id"]) == ["a"]


def test_load_or_build_manifest_force_rebuild_bypasses_param_mismatch(tmp_path):
    csv_path = tmp_path / "fake.csv"
    load_or_build_manifest(csv_path, _fake_manifest_builder, limit=5)

    result = load_or_build_manifest(csv_path, _fake_manifest_builder, limit=10, force_rebuild=True)

    assert list(result["sample_id"]) == ["a"]
    params_path = tmp_path / "fake.params.json"
    assert json.loads(params_path.read_text()) == {"limit": 10}
