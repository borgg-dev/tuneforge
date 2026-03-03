"""
Structural completeness scorer for TuneForge.

Evaluates whether generated music exhibits coherent large-scale structure —
distinct sections, variety between sections, gentle intros/outros, and
smooth transitions.  Four complementary metrics capture different facets
of structural quality:

* **section_count** — Does the piece contain an appropriate number of sections?
* **section_variety** — Are the sections sufficiently distinct from each other?
* **intro_outro_quality** — Do the opening and closing have gentler dynamics?
* **transition_smoothness** — Are section boundaries smooth rather than abrupt?

All metrics are genre-aware via ``GenreProfile`` and designed to resist gaming
by trivially engineered signals (e.g. pure tones, silence, noise).
"""

import numpy as np
from loguru import logger

from tuneforge.scoring.genre_profiles import GenreProfile, get_genre_profile

# ---------------------------------------------------------------------------
# Sub-metric weights (must sum to 1.0)
# ---------------------------------------------------------------------------

STRUCTURAL_WEIGHTS: dict[str, float] = {
    "section_count": 0.25,
    "section_variety": 0.25,
    "intro_outro_quality": 0.25,
    "transition_smoothness": 0.25,
}


class StructuralCompletenessScorer:
    """Assess structural completeness of generated audio."""

    # Minimum audio duration (seconds) for structural analysis
    _MIN_DURATION: float = 2.0
    # Amplitude below which audio is considered silence
    _SILENCE_THRESHOLD: float = 1e-6

    def score(self, audio: np.ndarray, sr: int, genre: str = "") -> dict[str, float]:
        """
        Compute per-metric structural completeness scores.

        Args:
            audio: Waveform array (1-D or 2-D).
            sr: Sample rate in Hz.
            genre: Optional genre string for genre-aware target adjustment.

        Returns:
            Dict with keys matching ``STRUCTURAL_WEIGHTS``.  All values in [0, 1].
        """
        try:
            import librosa
            from scipy import ndimage, signal

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            # --- Edge-case guards ---
            if np.max(np.abs(audio)) < self._SILENCE_THRESHOLD:
                return {k: 0.0 for k in STRUCTURAL_WEIGHTS}
            if len(audio) / sr < self._MIN_DURATION:
                return {k: 0.0 for k in STRUCTURAL_WEIGHTS}

            profile = get_genre_profile(genre) if genre else GenreProfile(family="default")
            boundaries = _detect_sections(audio, sr, librosa)

            return {
                "section_count": self._score_section_count(
                    audio, sr, boundaries, profile,
                ),
                "section_variety": self._score_section_variety(
                    audio, sr, boundaries, librosa, profile,
                ),
                "intro_outro_quality": self._score_intro_outro_quality(
                    audio, sr, boundaries,
                ),
                "transition_smoothness": self._score_transition_smoothness(
                    audio, sr, boundaries, librosa,
                ),
            }
        except Exception as exc:
            logger.error(f"Structural completeness scoring failed: {exc}")
            return {k: 0.0 for k in STRUCTURAL_WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        """
        Weighted aggregation of per-metric scores.

        Args:
            scores: Dict from ``score()``.

        Returns:
            Aggregate structural completeness score in [0, 1].
        """
        total = 0.0
        for metric, weight in STRUCTURAL_WEIGHTS.items():
            total += scores.get(metric, 0.0) * weight
        return float(np.clip(total, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_section_count(
        audio: np.ndarray,
        sr: int,
        boundaries: list[int],
        profile: GenreProfile,
    ) -> float:
        """
        Score based on detected number of sections vs genre expectation.

        Uses a Gaussian bell curve centred on the genre-specific expected
        section count (scaled by audio duration).  Having exactly the
        expected number of sections scores 1.0; deviations decay
        exponentially.
        """
        try:
            duration = len(audio) / sr
            # Number of sections = number of segments between boundaries
            n_sections = max(len(boundaries) - 1, 1)
            expected = profile.structural_section_target * duration / 30.0
            expected = max(expected, 1.0)

            score = float(
                np.exp(-2.0 * (abs(n_sections - expected) / max(expected, 1)) ** 2)
            )
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_section_variety(
        audio: np.ndarray,
        sr: int,
        boundaries: list[int],
        librosa,
        profile: GenreProfile,
    ) -> float:
        """
        Score based on distinctness of detected sections.

        Computes the mean chroma vector per section, normalises, and
        measures pairwise cosine distances.  The mean distance is compared
        to a genre-specific target via a Gaussian bell curve.
        """
        try:
            n_sections = len(boundaries) - 1
            if n_sections < 2:
                # Only one section — variety is undefined; give a neutral score
                return 0.5

            hop_length = 512
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
            frames_per_sample = sr / hop_length

            # Compute mean chroma per section
            section_chromas = []
            for i in range(n_sections):
                start_frame = int(boundaries[i] / sr * frames_per_sample)
                end_frame = int(boundaries[i + 1] / sr * frames_per_sample)
                start_frame = max(0, min(start_frame, chroma.shape[1] - 1))
                end_frame = max(start_frame + 1, min(end_frame, chroma.shape[1]))
                mean_chroma = chroma[:, start_frame:end_frame].mean(axis=1)
                norm = np.linalg.norm(mean_chroma) + 1e-8
                section_chromas.append(mean_chroma / norm)

            section_chromas = np.array(section_chromas)

            # Pairwise cosine distances
            distances = []
            for i in range(len(section_chromas)):
                for j in range(i + 1, len(section_chromas)):
                    cos_sim = float(np.dot(section_chromas[i], section_chromas[j]))
                    cos_dist = 1.0 - cos_sim
                    distances.append(cos_dist)

            if not distances:
                return 0.5

            mean_distance = float(np.mean(distances))

            # Bell curve centred at genre-specific variety target
            target = profile.structural_variety_target
            score = float(
                np.exp(-8.0 * (mean_distance - target) ** 2)
            )
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_intro_outro_quality(
        audio: np.ndarray,
        sr: int,
        boundaries: list[int],
    ) -> float:
        """
        Score based on whether first/last sections have gentler energy.

        Computes RMS energy for the first section, last section, and the
        average of middle sections.  If an edge section has lower energy
        than the middle (ratio < 0.8), it is considered a good intro or
        outro.  Score = fraction of edges that are gentler.
        """
        try:
            n_sections = len(boundaries) - 1
            if n_sections < 3:
                # Need at least intro + body + outro (3 sections)
                # With fewer sections, give a neutral score
                return 0.5

            # Compute RMS per section
            section_rms = []
            for i in range(n_sections):
                section_audio = audio[boundaries[i]:boundaries[i + 1]]
                if len(section_audio) == 0:
                    section_rms.append(0.0)
                else:
                    rms = float(np.sqrt(np.mean(section_audio ** 2)))
                    section_rms.append(rms)

            first_rms = section_rms[0]
            last_rms = section_rms[-1]
            middle_rms_values = section_rms[1:-1]

            if not middle_rms_values:
                return 0.5

            middle_avg_rms = float(np.mean(middle_rms_values))

            if middle_avg_rms < 1e-8:
                return 0.5

            # Count how many edges (first/last) are gentler than middle
            gentle_count = 0
            if first_rms / middle_avg_rms < 0.8:
                gentle_count += 1
            if last_rms / middle_avg_rms < 0.8:
                gentle_count += 1

            score = gentle_count / 2.0
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_transition_smoothness(
        audio: np.ndarray,
        sr: int,
        boundaries: list[int],
        librosa,
    ) -> float:
        """
        Score based on smoothness of transitions at section boundaries.

        At each detected boundary, the spectral flux (L2 norm of STFT
        frame differences) is measured and compared to the median flux
        across all frames.  Lower boundary flux relative to the median
        indicates smoother transitions.
        """
        try:
            # Need at least one interior boundary
            interior_boundaries = boundaries[1:-1]
            if not interior_boundaries:
                return 0.5

            n_fft = 2048
            hop_length = 512
            S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

            if S.shape[1] < 3:
                return 0.5

            # Compute spectral flux for all frames
            flux = np.linalg.norm(np.diff(S, axis=1), axis=0)

            if len(flux) == 0:
                return 0.5

            median_flux = float(np.median(flux))
            if median_flux < 1e-10:
                return 0.5

            # Spectral flux at boundary frames
            boundary_fluxes = []
            for b in interior_boundaries:
                frame_idx = int(b / hop_length)
                frame_idx = max(0, min(frame_idx, len(flux) - 1))
                boundary_fluxes.append(flux[frame_idx])

            if not boundary_fluxes:
                return 0.5

            mean_boundary_flux = float(np.mean(boundary_fluxes))
            boundary_ratio = mean_boundary_flux / median_flux

            # Lower ratio = smoother transitions
            score = 1.0 - min(boundary_ratio / 3.0, 1.0)
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _detect_sections(
    audio: np.ndarray,
    sr: int,
    librosa,
) -> list[int]:
    """
    Detect section boundaries using chroma self-similarity and a
    checkerboard kernel novelty function.

    Args:
        audio: 1-D float waveform.
        sr: Sample rate in Hz.
        librosa: The lazily-imported librosa module.

    Returns:
        Sorted list of sample-position boundaries.  Always starts with 0
        and ends with ``len(audio)``.
    """
    from scipy import ndimage, signal

    hop_length = 512

    # Compute chroma features and normalise columns
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8
    chroma_norm = chroma / norms

    n_frames = chroma_norm.shape[1]
    if n_frames < 4:
        return [0, len(audio)]

    # Self-similarity matrix
    sim = chroma_norm.T @ chroma_norm  # (n_frames, n_frames)

    # Checkerboard kernel novelty
    kernel_size = min(16, n_frames // 2)
    if kernel_size < 2:
        return [0, len(audio)]

    novelty = np.zeros(n_frames)
    for i in range(kernel_size, n_frames - kernel_size):
        # Same-quadrant blocks (top-left and bottom-right)
        same_tl = sim[i - kernel_size:i, i - kernel_size:i]
        same_br = sim[i:i + kernel_size, i:i + kernel_size]
        same_mean = (np.mean(same_tl) + np.mean(same_br)) / 2.0

        # Cross-quadrant blocks (top-right and bottom-left)
        cross_tr = sim[i - kernel_size:i, i:i + kernel_size]
        cross_bl = sim[i:i + kernel_size, i - kernel_size:i]
        cross_mean = (np.mean(cross_tr) + np.mean(cross_bl)) / 2.0

        novelty[i] = max(same_mean - cross_mean, 0.0)

    # Smooth the novelty curve
    smooth_width = max(3, kernel_size // 2)
    novelty = ndimage.uniform_filter1d(novelty, size=smooth_width)

    # Peak-pick with minimum distance constraint
    duration = len(audio) / sr
    frames_per_second = sr / hop_length
    # Minimum 2 seconds between section boundaries
    min_distance = max(1, int(2.0 * frames_per_second))

    peaks, _ = signal.find_peaks(novelty, distance=min_distance, prominence=0.05)

    # Convert frame indices to sample positions
    boundaries = [0]
    for p in peaks:
        sample_pos = int(p * hop_length)
        if 0 < sample_pos < len(audio):
            boundaries.append(sample_pos)
    boundaries.append(len(audio))

    # Ensure sorted and deduplicated
    boundaries = sorted(set(boundaries))
    return boundaries
