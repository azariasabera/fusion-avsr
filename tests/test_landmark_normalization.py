"""Unit tests for fusion_avsr.models.landmark.normalization.

compute_dataset_pixel_stats requires torchcodec + real video files, so it
is not exercised here (needs Roihu). load_or_compute_pixel_stats's
caching behavior and normalize_frames are pure-numpy/pure-Python and
tested directly.
"""

import json

import numpy as np

from fusion_avsr.models.landmark.normalization import load_or_compute_pixel_stats, normalize_frames


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
