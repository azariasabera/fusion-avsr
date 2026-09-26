"""Unit tests for fusion_avsr.models.landmark.normalization.

compute_dataset_pixel_stats's real-decoder path requires torchcodec and
is not exercised here (needs Roihu), but its accumulation math is
tested via the _open_decoder injection point, with a fake decoder --
same pattern as fusion_avsr.utils.video's tests.
"""

import json

import numpy as np
import pytest

from fusion_avsr.models.landmark.normalization import (
    compute_dataset_pixel_stats,
    load_or_compute_pixel_stats,
    normalize_frames,
)


def test_normalize_frames_zero_mean_unit_std_on_constant_input():
    frames = np.full((4, 8, 8), 100, dtype=np.uint8)

    normalized = normalize_frames(frames, mean=100.0, std=2.0)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, 0.0)


def test_normalize_frames_shape_preserved():
    frames = np.random.default_rng(0).integers(0, 255, size=(3, 5, 32, 32), dtype=np.uint8)

    normalized = normalize_frames(frames, mean=127.0, std=50.0)

    assert normalized.shape == frames.shape


def test_load_or_compute_pixel_stats_uses_cache_without_calling_computer(tmp_path, monkeypatch):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"mean": 42.0, "std": 7.0}))

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("compute_dataset_pixel_stats should not be called when a cache exists")

    monkeypatch.setattr(
        "fusion_avsr.models.landmark.normalization.compute_dataset_pixel_stats", _fail_if_called
    )

    mean, std = load_or_compute_pixel_stats(stats_path, video_paths=["unused.mp4"])

    assert (mean, std) == (42.0, 7.0)


class _FakeFrame:
    def __init__(self, array):
        self._array = array

    def numpy(self):
        return self._array


class _FakeDecoder:
    def __init__(self, frames):
        self._frames = frames

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, index):
        return _FakeFrame(self._frames[index])


def test_compute_dataset_pixel_stats_constant_frame_gives_exact_mean_zero_std():
    frame = np.full((4, 4), 100, dtype=np.uint8)
    decoders_by_path = {"a.mp4": _FakeDecoder([frame, frame, frame])}

    mean, std = compute_dataset_pixel_stats(
        ["a.mp4"], frames_per_video=3, seed=0, _open_decoder=decoders_by_path.get
    )

    assert mean == pytest.approx(100.0)
    assert std == pytest.approx(0.0, abs=1e-6)


def test_compute_dataset_pixel_stats_matches_manual_mean_and_std():
    # Two 2x2 grayscale frames with known pixel values -- verifies the
    # vectorized sum/sum-of-squares accumulation against numpy directly.
    frame_a = np.array([[0, 50], [100, 150]], dtype=np.uint8)
    frame_b = np.array([[200, 210], [220, 230]], dtype=np.uint8)
    decoders_by_path = {"a.mp4": _FakeDecoder([frame_a, frame_b])}

    mean, std = compute_dataset_pixel_stats(
        ["a.mp4"], frames_per_video=2, seed=0, _open_decoder=decoders_by_path.get
    )

    all_pixels = np.concatenate([frame_a.ravel(), frame_b.ravel()]).astype(np.float64)
    assert mean == pytest.approx(all_pixels.mean())
    assert std == pytest.approx(all_pixels.std())


def test_compute_dataset_pixel_stats_raises_if_nothing_sampled():
    decoders_by_path = {"empty.mp4": _FakeDecoder([])}

    with pytest.raises(ValueError):
        compute_dataset_pixel_stats(["empty.mp4"], _open_decoder=decoders_by_path.get)


def test_load_or_compute_pixel_stats_computes_and_caches(tmp_path, monkeypatch):
    stats_path = tmp_path / "nested" / "stats.json"

    monkeypatch.setattr(
        "fusion_avsr.models.landmark.normalization.compute_dataset_pixel_stats",
        lambda video_paths, **kwargs: (10.0, 3.0),
    )

    mean, std = load_or_compute_pixel_stats(stats_path, video_paths=["a.mp4"])

    assert (mean, std) == (10.0, 3.0)
    assert stats_path.exists()
    assert json.loads(stats_path.read_text()) == {"mean": 10.0, "std": 3.0}
