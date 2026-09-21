"""Unit tests for fusion_avsr.data.noise_augmentation.

Builds a small synthetic MUSAN-like directory (a handful of short .wav
files under speech/, music/, noise/) under pytest's tmp_path fixture --
no real MUSAN corpus is needed.
"""

import random

import numpy as np
import pytest
import soundfile as sf

from fusion_avsr.data.noise_augmentation import (
    BABBLE_NUM_SPEAKERS,
    NOISE_CATEGORIES,
    SNR_BUCKETS,
    NoiseMixer,
    _fit_length,
    _load_audio,
    _mix_at_snr,
    _resample_if_needed,
)

SAMPLE_RATE = 16000


def _write_wav(path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples.astype(np.float32), sample_rate)


@pytest.fixture
def musan_root(tmp_path):
    """A small synthetic MUSAN-like directory: a few short noise .wav files per category."""
    rng = np.random.default_rng(0)
    root = tmp_path / "musan"

    for i in range(BABBLE_NUM_SPEAKERS + 2):  # a couple more than one babble mix needs
        _write_wav(root / "speech" / f"speaker{i}.wav", rng.standard_normal(SAMPLE_RATE))

    for i in range(3):
        _write_wav(root / "music" / f"track{i}.wav", rng.standard_normal(SAMPLE_RATE))

    for i in range(3):
        _write_wav(root / "noise" / f"clip{i}.wav", rng.standard_normal(SAMPLE_RATE))

    return root


# ---------------------------------------------------------------------------
# _fit_length
# ---------------------------------------------------------------------------

def test_fit_length_crops_long_audio_to_exact_length():
    audio = np.arange(100, dtype=np.float32)
    fitted = _fit_length(audio, target_length=10, rng=random.Random(0))
    assert len(fitted) == 10


def test_fit_length_tiles_short_audio_to_exact_length():
    audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fitted = _fit_length(audio, target_length=10, rng=random.Random(0))
    assert len(fitted) == 10


def test_fit_length_handles_empty_audio():
    audio = np.array([], dtype=np.float32)
    fitted = _fit_length(audio, target_length=5, rng=random.Random(0))
    assert len(fitted) == 5
    assert np.all(fitted == 0)


# ---------------------------------------------------------------------------
# _mix_at_snr
# ---------------------------------------------------------------------------

def test_mix_at_snr_achieves_target_ratio():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal(SAMPLE_RATE).astype(np.float32)
    noise = rng.standard_normal(SAMPLE_RATE).astype(np.float32)

    for target_snr_db in (20, 10, 0, -5):
        mixed = _mix_at_snr(clean, noise, snr_db=target_snr_db)
        added_noise = mixed - clean

        signal_power = np.mean(clean ** 2)
        added_noise_power = np.mean(added_noise ** 2)
        achieved_snr_db = 10 * np.log10(signal_power / added_noise_power)

        assert achieved_snr_db == pytest.approx(target_snr_db, abs=0.1)


def test_mix_at_snr_handles_silent_clean_signal_without_dividing_by_zero():
    clean = np.zeros(1000, dtype=np.float32)
    noise = np.ones(1000, dtype=np.float32)

    mixed = _mix_at_snr(clean, noise, snr_db=0)

    assert np.all(np.isfinite(mixed))


# ---------------------------------------------------------------------------
# NoiseMixer
# ---------------------------------------------------------------------------

def test_available_categories_excludes_held_out_by_default(musan_root):
    mixer = NoiseMixer(musan_root, held_out_category="noise", seed=0)

    assert "noise" not in mixer.available_categories()
    assert set(mixer.available_categories()) == {"white", "babble", "music"}
    assert set(mixer.available_categories(include_held_out=True)) == set(NOISE_CATEGORIES)


def test_available_categories_all_included_when_held_out_is_none(musan_root):
    mixer = NoiseMixer(musan_root, held_out_category=None, seed=0)

    assert set(mixer.available_categories()) == set(NOISE_CATEGORIES)


def test_mix_clean_bucket_returns_unmodified_copy(musan_root):
    mixer = NoiseMixer(musan_root, seed=0)
    clean = np.ones(1000, dtype=np.float32)

    mixed = mixer.mix(clean, snr_bucket="clean")

    assert np.array_equal(mixed, clean)
    mixed[0] = 999.0
    assert clean[0] == 1.0  # confirms mix() returned a copy, not the same array


@pytest.mark.parametrize("category", ["white", "babble", "music", "noise"])
def test_mix_each_category_produces_correct_length_output(musan_root, category):
    mixer = NoiseMixer(musan_root, seed=0)
    clean = np.random.default_rng(1).standard_normal(4000).astype(np.float32)

    mixed = mixer.mix(clean, snr_bucket=10, category=category)

    assert mixed.shape == clean.shape
    assert mixed.dtype == np.float32


def test_mix_invalid_snr_bucket_raises(musan_root):
    mixer = NoiseMixer(musan_root, seed=0)
    with pytest.raises(ValueError):
        mixer.mix(np.zeros(100, dtype=np.float32), snr_bucket=3)  # 3 is not a valid bucket


def test_mix_invalid_category_raises(musan_root):
    mixer = NoiseMixer(musan_root, seed=0)
    with pytest.raises(ValueError):
        mixer.mix(np.zeros(100, dtype=np.float32), snr_bucket=0, category="not_a_real_category")


def test_sample_bucket_always_returns_a_valid_bucket(musan_root):
    mixer = NoiseMixer(musan_root, seed=0)
    for _ in range(50):
        assert mixer.sample_bucket() in SNR_BUCKETS


def test_resample_if_needed_returns_unchanged_when_rates_already_match():
    audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = _resample_if_needed(audio, sample_rate=16000, target_sample_rate=16000, wav_path="x.wav")
    assert np.array_equal(result, audio)


def test_resample_if_needed_resamples_when_rates_differ():
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(8000).astype(np.float32)  # 1 second at 8kHz

    result = _resample_if_needed(audio, sample_rate=8000, target_sample_rate=16000, wav_path="x.wav")

    assert len(result) == pytest.approx(16000, rel=0.01)  # ~1 second at 16kHz


def test_load_audio_resamples_a_file_at_the_wrong_sample_rate(tmp_path):
    rng = np.random.default_rng(0)
    wav_path = tmp_path / "mismatched_rate.wav"
    _write_wav(wav_path, rng.standard_normal(8000), sample_rate=8000)

    audio = _load_audio(wav_path, target_sample_rate=16000)

    assert len(audio) == pytest.approx(16000, rel=0.01)


def test_babble_works_with_fewer_speech_files_than_babble_num_speakers(tmp_path):
    root = tmp_path / "musan"
    rng = np.random.default_rng(0)
    _write_wav(root / "speech" / "only_one.wav", rng.standard_normal(SAMPLE_RATE))
    _write_wav(root / "music" / "track0.wav", rng.standard_normal(SAMPLE_RATE))
    _write_wav(root / "noise" / "clip0.wav", rng.standard_normal(SAMPLE_RATE))

    mixer = NoiseMixer(root, seed=0)
    clean = np.ones(1000, dtype=np.float32)

    # Must not crash even though there are fewer speech files than
    # BABBLE_NUM_SPEAKERS would normally sample.
    mixed = mixer.mix(clean, snr_bucket=10, category="babble")
    assert mixed.shape == clean.shape
