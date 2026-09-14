"""On-the-fly MUSAN noise mixing at controlled SNR buckets.

Mixes clean audio with noise at a chosen signal-to-noise ratio (SNR).
Mixing happens on-the-fly at data-loading time; nothing is written to
disk here.

Four noise categories are supported:

- ``"white"``: synthetic Gaussian noise, generated on-the-fly (not read
  from any file).
- ``"babble"``: built by summing several randomly chosen clips from
  MUSAN's ``speech/`` folder, a standard way to construct
  cocktail-party-style babble noise from a corpus of individual speech
  recordings.
- ``"music"``: MUSAN's ``music/`` folder, used directly.
- ``"noise"``: MUSAN's ``noise/`` folder, used directly. Intended as the
  unseen noise type for the test set: held out from training by default
  so a model's noise robustness can be evaluated against a category it
  never saw during training.

SNR is controlled via a fixed set of buckets: ``"clean"`` (no noise
added) and 20, 15, 10, 5, 0, -5 dB.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import soundfile as sf

PathLike = Union[str, Path]

# Fixed SNR buckets this module mixes at. "clean" means no noise is
# added; the numeric entries are target SNR in dB.
SNR_BUCKETS = ("clean", 20, 15, 10, 5, 0, -5)

# The four noise categories this module can mix in. See the module
# docstring above for how each one is sourced.
NOISE_CATEGORIES = ("white", "babble", "music", "noise")

# Default category excluded from training by NoiseMixer.available_categories().
DEFAULT_HELD_OUT_CATEGORY = "noise"

# Relative sub-paths, within a MUSAN root directory, that back each
# MUSAN-sourced category. "white" is absent here on purpose: it is
# synthesized, not read from MUSAN.
_MUSAN_CATEGORY_SUBDIRS = {
    "babble": "speech",
    "music": "music",
    "noise": "noise",
}

# Number of individual MUSAN speech clips summed together to build one
# babble noise sample. 5-7 overlapping speakers is a standard choice for
# simulating cocktail-party babble noise; we fix it at 6.
BABBLE_NUM_SPEAKERS = 6


def _list_musan_wavs(musan_root: PathLike, category: str) -> List[Path]:
    """List every ``.wav`` file under a MUSAN category's subdirectory.

    Args:
        musan_root: Path to the MUSAN corpus root (the directory
            containing ``music/``, ``noise/``, ``speech/``).
        category: One of the keys of ``_MUSAN_CATEGORY_SUBDIRS`` (i.e.
            ``"babble"``, ``"music"``, or ``"noise"`` -- NOT ``"white"``,
            which has no MUSAN files).

    Returns:
        A sorted list of paths to every ``.wav`` file found recursively
        under ``<musan_root>/<subdir_for_category>/``.

    Raises:
        KeyError: If ``category`` is ``"white"`` or otherwise not a
            MUSAN-backed category.
    """
    subdir = _MUSAN_CATEGORY_SUBDIRS[category]
    category_root = Path(musan_root) / subdir
    return sorted(category_root.rglob("*.wav"))


def _load_audio(wav_path: PathLike) -> np.ndarray:
    """Load a ``.wav`` file as a 1-D float32 array, downmixed to mono if needed."""
    audio, _sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def _fit_length(audio: np.ndarray, target_length: int, rng: random.Random) -> np.ndarray:
    """Crop or loop-pad ``audio`` to exactly ``target_length`` samples.

    If ``audio`` is longer than ``target_length``, a random contiguous
    crop of the right length is taken. If it is shorter, it is tiled
    (repeated) until it reaches at least ``target_length`` samples, then
    cropped down to exactly that length. This lets a noise clip of any
    length be mixed with a clean clip of any other length.

    Args:
        audio: 1-D audio array to fit.
        target_length: Desired output length, in samples.
        rng: Random number generator used to pick the crop offset (an
            explicit argument, rather than the module-global random
            state, so callers can make mixing reproducible via a seeded
            ``random.Random`` instance).

    Returns:
        A 1-D array of exactly ``target_length`` samples.
    """
    if len(audio) == 0:
        return np.zeros(target_length, dtype=np.float32)

    if len(audio) < target_length:
        num_repeats = target_length // len(audio) + 1
        audio = np.tile(audio, num_repeats)

    max_start = len(audio) - target_length
    start = rng.randint(0, max_start) if max_start > 0 else 0
    return audio[start:start + target_length]


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix ``clean`` and ``noise`` (same length) so the result has the given SNR.

    Scales ``noise`` so that the ratio of clean signal power to scaled
    noise power equals ``snr_db`` (in decibels), then adds it to
    ``clean``. If ``clean`` is silent (zero power), noise is added
    unscaled, since SNR is undefined for a zero-power signal.

    Args:
        clean: 1-D clean audio array.
        noise: 1-D noise audio array, the same length as ``clean``.
        snr_db: Target signal-to-noise ratio, in decibels.

    Returns:
        A 1-D array of the same length as ``clean``, containing the
        noisy mixture.
    """
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if signal_power == 0 or noise_power == 0:
        return clean + noise

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return clean + scale * noise


class NoiseMixer:
    """Mixes clean audio with MUSAN-derived or synthetic noise at controlled SNR.

    One ``NoiseMixer`` instance is constructed per MUSAN root directory,
    and reused across many calls to ``mix`` (it caches the list of
    available ``.wav`` files per category so repeated mixing doesn't
    re-scan the filesystem each time).

    Example:
        >>> mixer = NoiseMixer(musan_root="/path/to/musan")
        >>> noisy = mixer.mix(clean_audio, snr_bucket=5, category="babble")
    """

    def __init__(
        self,
        musan_root: PathLike,
        sample_rate: int = 16000,
        held_out_category: Optional[str] = DEFAULT_HELD_OUT_CATEGORY,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize a ``NoiseMixer``.

        Args:
            musan_root: Path to the MUSAN corpus root (the directory
                containing ``music/``, ``noise/``, ``speech/``).
            sample_rate: Sample rate, in Hz, that all audio passed to
                ``mix`` is assumed to already be at. MUSAN's files are
                not resampled by this class -- callers are responsible
                for ensuring clean audio and MUSAN audio share a sample
                rate.
            held_out_category: The noise category to exclude from
                ``available_categories()`` by default, so training code
                that calls ``mix`` without specifying a category never
                accidentally draws from the unseen-noise eval category.
                Defaults to ``"noise"``. Pass ``None`` to disable
                held-out behavior entirely (all four categories
                available for training).
            seed: Optional random seed, for reproducible noise selection
                and mixing.
        """
        self.musan_root = Path(musan_root)
        self.sample_rate = sample_rate
        self.held_out_category = held_out_category
        self._rng = random.Random(seed)

        self._wavs_by_category = {
            category: _list_musan_wavs(self.musan_root, category)
            for category in _MUSAN_CATEGORY_SUBDIRS
        }

    def available_categories(self, include_held_out: bool = False) -> List[str]:
        """List the noise categories available for use.

        Args:
            include_held_out: If False (default), the held-out category
                (``self.held_out_category``) is excluded -- use this for
                training. If True, all four categories are returned --
                use this for the unseen-noise generalization eval, which
                specifically needs access to the held-out category.

        Returns:
            A list of category name strings, a subset of
            ``NOISE_CATEGORIES``.
        """
        if include_held_out or self.held_out_category is None:
            return list(NOISE_CATEGORIES)
        return [c for c in NOISE_CATEGORIES if c != self.held_out_category]

    def _sample_white_noise(self, length: int) -> np.ndarray:
        """Generate ``length`` samples of synthetic Gaussian white noise."""
        return self._rng_numpy().standard_normal(length).astype(np.float32)

    def _rng_numpy(self) -> np.random.Generator:
        """A numpy Generator seeded from this mixer's Python `random.Random` state.

        Keeps a single source of randomness (`self._rng`) driving both
        Python-level choices (which files to pick) and numpy-level noise
        synthesis, so a fixed `seed` at construction time makes an
        entire mixing session reproducible.
        """
        return np.random.default_rng(self._rng.randint(0, 2**32 - 1))

    def _sample_noise_signal(self, category: str, length: int) -> np.ndarray:
        """Produce ``length`` samples of noise from the given category.

        For ``"white"``, synthesizes Gaussian noise directly. For the
        three MUSAN-backed categories, picks a random file (or, for
        ``"babble"``, several files summed together) and fits it to
        ``length`` via ``_fit_length``.

        Args:
            category: One of ``NOISE_CATEGORIES``.
            length: Desired noise signal length, in samples.

        Returns:
            A 1-D float32 array of exactly ``length`` samples.
        """
        if category == "white":
            return self._sample_white_noise(length)

        if category == "babble":
            speech_wavs = self._wavs_by_category["babble"]
            chosen = self._rng.sample(speech_wavs, k=min(BABBLE_NUM_SPEAKERS, len(speech_wavs)))
            speakers = [_fit_length(_load_audio(p), length, self._rng) for p in chosen]
            return np.sum(speakers, axis=0).astype(np.float32)

        # "music" and "noise" categories: a single randomly-chosen MUSAN file.
        candidates = self._wavs_by_category[category]
        chosen_path = self._rng.choice(candidates)
        return _fit_length(_load_audio(chosen_path), length, self._rng)

    def mix(
        self,
        clean_audio: np.ndarray,
        snr_bucket: Union[str, int, float],
        category: Optional[str] = None,
    ) -> np.ndarray:
        """Mix ``clean_audio`` with noise at the given SNR bucket.

        Args:
            clean_audio: 1-D clean audio array, at ``self.sample_rate``.
            snr_bucket: One of ``SNR_BUCKETS``: either the string
                ``"clean"`` (no noise added, ``clean_audio`` is returned
                unchanged) or a numeric target SNR in dB.
            category: Which noise category to draw from. If ``None``
                (default), a category is chosen uniformly at random from
                ``self.available_categories()`` (i.e. excluding the
                held-out category, unless ``held_out_category`` was set
                to ``None`` at construction time).

        Returns:
            A 1-D float32 array of the same length as ``clean_audio``:
            the noisy mixture (or an unmodified copy of ``clean_audio``,
            if ``snr_bucket == "clean"``).

        Raises:
            ValueError: If ``snr_bucket`` is not one of ``SNR_BUCKETS``,
                or ``category`` is not one of ``NOISE_CATEGORIES``.
        """
        if snr_bucket not in SNR_BUCKETS:
            raise ValueError(f"snr_bucket must be one of {SNR_BUCKETS}, got {snr_bucket!r}")

        if snr_bucket == "clean":
            return np.array(clean_audio, dtype=np.float32, copy=True)

        if category is None:
            category = self._rng.choice(self.available_categories())
        elif category not in NOISE_CATEGORIES:
            raise ValueError(f"category must be one of {NOISE_CATEGORIES}, got {category!r}")

        noise = self._sample_noise_signal(category, length=len(clean_audio))
        return _mix_at_snr(np.asarray(clean_audio, dtype=np.float32), noise, snr_db=float(snr_bucket))

    def sample_bucket(self) -> Union[str, int]:
        """Pick one SNR bucket uniformly at random from ``SNR_BUCKETS``.

        Convenience for training code that wants a random noise condition
        per batch/clip without hand-rolling the random choice each time.

        Returns:
            One element of ``SNR_BUCKETS``.
        """
        return self._rng.choice(SNR_BUCKETS)
