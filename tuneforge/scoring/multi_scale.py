"""
Multi-scale evaluation framework for TuneForge.

Scores audio quality at different temporal granularities, adjusting
scorer weight multipliers based on clip duration.  Short clips emphasise
frame-level metrics (production, quality, perceptual); long clips
emphasise phrase and section metrics (structural, musicality, melody)
and add compositional bonuses.

Usage::

    evaluator = MultiScaleEvaluator()
    multipliers = evaluator.evaluate(audio, sr=44100, duration_seconds=45.0)
    # multipliers["structural"] == 1.8, multipliers["phrase_coherence_bonus"] == 0.03, ...
"""

from __future__ import annotations

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Duration thresholds (seconds)
# ---------------------------------------------------------------------------
_SHORT_THRESHOLD = 10.0
_LONG_THRESHOLD = 30.0

# ---------------------------------------------------------------------------
# Default weight multipliers per scale
# ---------------------------------------------------------------------------
_SHORT_MULTIPLIERS: dict[str, float] = {
    "structural": 0.5,
    "production": 1.5,
    "quality": 1.3,
    "perceptual": 1.3,
    "musicality": 0.7,
    "melody": 0.8,
    "neural_quality": 1.1,
    "neural_codec": 1.2,
    "vocal": 1.0,
    "speed": 1.0,
    "clap": 1.0,
    "preference": 1.0,
    "diversity": 1.0,
    "attribute": 1.0,
}

_MEDIUM_MULTIPLIERS: dict[str, float] = {
    "structural": 1.0,
    "production": 1.0,
    "quality": 1.0,
    "perceptual": 1.0,
    "musicality": 1.0,
    "melody": 1.0,
    "neural_quality": 1.0,
    "neural_codec": 1.0,
    "vocal": 1.0,
    "speed": 1.0,
    "clap": 1.0,
    "preference": 1.0,
    "diversity": 1.0,
    "attribute": 1.0,
}

_LONG_MULTIPLIERS: dict[str, float] = {
    "structural": 1.8,
    "production": 0.8,
    "quality": 0.9,
    "perceptual": 0.9,
    "musicality": 1.3,
    "melody": 1.5,
    "neural_quality": 1.0,
    "neural_codec": 0.8,
    "vocal": 1.0,
    "speed": 0.5,
    "clap": 1.0,
    "preference": 1.0,
    "diversity": 1.0,
    "attribute": 1.1,
}

# Maximum bonuses for long-form compositional analysis
_MAX_PHRASE_COHERENCE_BONUS = 0.05
_MAX_COMPOSITIONAL_ARC_BONUS = 0.05


class MultiScaleEvaluator:
    """Evaluate audio quality at multiple temporal scales.

    Adjusts scorer weight multipliers depending on clip duration and,
    for long clips, computes additional compositional bonuses that are
    added directly to the composite score.
    """

    def evaluate(
        self,
        audio: np.ndarray,
        sr: int,
        duration_seconds: float,
    ) -> dict[str, float]:
        """Return scale-appropriate weight adjustments for the main scorers.

        Args:
            audio: Audio waveform as a 1-D numpy array (mono).  If stereo
                (2-D with shape ``(channels, samples)``), the first channel
                is used.
            sr: Sample rate in Hz.
            duration_seconds: Duration of the clip in seconds.

        Returns:
            A dict of ``scorer_name -> weight_multiplier`` (0.5--2.0 range)
            plus optional bonus keys for long clips:

            * ``phrase_coherence_bonus`` -- up to 0.05
            * ``compositional_arc_bonus`` -- up to 0.05
        """
        audio = self._ensure_mono(audio)

        if duration_seconds < _SHORT_THRESHOLD:
            result = dict(_SHORT_MULTIPLIERS)
        elif duration_seconds < _LONG_THRESHOLD:
            result = self._interpolate_multipliers(duration_seconds)
        else:
            result = dict(_LONG_MULTIPLIERS)
            # Long-form compositional bonuses
            result["phrase_coherence_bonus"] = self._phrase_coherence(
                audio, sr, duration_seconds,
            )
            result["compositional_arc_bonus"] = self._compositional_arc(
                audio, sr, duration_seconds,
            )

        # Clamp all multipliers to the valid range
        for key in list(result):
            if key.endswith("_bonus"):
                continue
            result[key] = float(np.clip(result[key], 0.5, 2.0))

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_mono(audio: np.ndarray) -> np.ndarray:
        """Collapse to mono if multi-channel."""
        if audio.ndim == 2:
            return audio[0]
        return audio

    @staticmethod
    def _interpolate_multipliers(duration: float) -> dict[str, float]:
        """Linearly interpolate between short and long multipliers for medium clips."""
        t = (duration - _SHORT_THRESHOLD) / (_LONG_THRESHOLD - _SHORT_THRESHOLD)
        t = float(np.clip(t, 0.0, 1.0))
        result: dict[str, float] = {}
        all_keys = set(_SHORT_MULTIPLIERS) | set(_LONG_MULTIPLIERS)
        for key in all_keys:
            short_val = _SHORT_MULTIPLIERS.get(key, 1.0)
            long_val = _LONG_MULTIPLIERS.get(key, 1.0)
            result[key] = short_val + t * (long_val - short_val)
        return result

    def _phrase_coherence(
        self,
        audio: np.ndarray,
        sr: int,
        duration_seconds: float,
    ) -> float:
        """Analyse 8-bar phrase coherence using beat tracking and chroma.

        Segments the audio into roughly 8-bar phrases (estimated via beat
        tracking), computes a chroma vector for each phrase, and measures
        inter-phrase similarity.  Higher similarity between alternating
        phrases (e.g. verse--chorus--verse pattern) yields a higher bonus.

        Returns:
            Bonus in [0, _MAX_PHRASE_COHERENCE_BONUS].
        """
        try:
            import librosa
        except ImportError:
            logger.debug("librosa unavailable — skipping phrase coherence")
            return 0.0

        try:
            # Beat tracking to estimate bar length
            tempo_result = librosa.beat.beat_track(y=audio, sr=sr)
            tempo = float(tempo_result[0]) if np.ndim(tempo_result[0]) == 0 else float(tempo_result[0][0])
            beat_frames = tempo_result[1]

            if tempo <= 0 or len(beat_frames) < 16:
                # Not enough beats for phrase analysis
                return 0.0

            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            beats_per_bar = 4  # assume 4/4
            bars_per_phrase = 8
            beats_per_phrase = beats_per_bar * bars_per_phrase

            # Segment into phrases
            phrases: list[np.ndarray] = []
            for i in range(0, len(beat_times) - beats_per_phrase + 1, beats_per_phrase):
                start_time = beat_times[i]
                end_idx = i + beats_per_phrase
                end_time = beat_times[end_idx] if end_idx < len(beat_times) else duration_seconds
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                end_sample = min(end_sample, len(audio))
                if end_sample - start_sample < sr:
                    continue
                phrases.append(audio[start_sample:end_sample])

            if len(phrases) < 2:
                return 0.0

            # Compute mean chroma for each phrase
            chroma_vectors = []
            for phrase_audio in phrases:
                chroma = librosa.feature.chroma_stft(y=phrase_audio, sr=sr)
                chroma_vectors.append(np.mean(chroma, axis=1))

            # Measure inter-phrase cosine similarity
            similarities = []
            for i in range(len(chroma_vectors)):
                for j in range(i + 1, len(chroma_vectors)):
                    a, b = chroma_vectors[i], chroma_vectors[j]
                    norm_a = np.linalg.norm(a)
                    norm_b = np.linalg.norm(b)
                    if norm_a < 1e-8 or norm_b < 1e-8:
                        continue
                    sim = float(np.dot(a, b) / (norm_a * norm_b))
                    similarities.append(sim)

            if not similarities:
                return 0.0

            mean_sim = float(np.mean(similarities))
            # Map similarity [0.5, 1.0] -> bonus [0, max]
            bonus = float(np.clip((mean_sim - 0.5) / 0.5, 0.0, 1.0)) * _MAX_PHRASE_COHERENCE_BONUS
            return round(bonus, 4)

        except Exception as exc:
            logger.warning("Phrase coherence analysis failed: {}", exc)
            return 0.0

    def _compositional_arc(
        self,
        audio: np.ndarray,
        sr: int,
        duration_seconds: float,
    ) -> float:
        """Measure energy contour for build/climax/resolution arc.

        Computes the RMS energy envelope, fits an idealised arc
        (rise--peak--fall), and scores how well the track follows it.
        Tracks with a clear build-up, climax, and resolution earn
        a higher bonus.

        Returns:
            Bonus in [0, _MAX_COMPOSITIONAL_ARC_BONUS].
        """
        try:
            import librosa
        except ImportError:
            logger.debug("librosa unavailable — skipping compositional arc")
            return 0.0

        try:
            # Compute RMS energy in ~1-second windows
            hop_length = sr
            frame_length = sr
            if len(audio) < frame_length * 3:
                return 0.0

            rms = librosa.feature.rms(
                y=audio, frame_length=frame_length, hop_length=hop_length,
            )[0]

            if len(rms) < 3:
                return 0.0

            # Normalise RMS to [0, 1]
            rms_min = float(np.min(rms))
            rms_max = float(np.max(rms))
            if rms_max - rms_min < 1e-8:
                return 0.0
            rms_norm = (rms - rms_min) / (rms_max - rms_min)

            n = len(rms_norm)
            # Generate ideal arc: triangular rise-peak-fall with peak at ~60%
            peak_pos = 0.6
            ideal = np.zeros(n)
            peak_idx = int(n * peak_pos)
            if peak_idx < 1 or peak_idx >= n - 1:
                peak_idx = n // 2

            # Rising phase
            ideal[:peak_idx + 1] = np.linspace(0.2, 1.0, peak_idx + 1)
            # Falling phase
            ideal[peak_idx:] = np.linspace(1.0, 0.3, n - peak_idx)

            # Correlation between actual and ideal arc
            actual_centered = rms_norm - np.mean(rms_norm)
            ideal_centered = ideal - np.mean(ideal)
            norm_actual = np.linalg.norm(actual_centered)
            norm_ideal = np.linalg.norm(ideal_centered)

            if norm_actual < 1e-8 or norm_ideal < 1e-8:
                return 0.0

            correlation = float(np.dot(actual_centered, ideal_centered) / (norm_actual * norm_ideal))

            # Also reward dynamic variation (not flat)
            energy_variation = float(np.std(rms_norm))
            variation_bonus = float(np.clip(energy_variation / 0.25, 0.0, 1.0))

            # Combined score: correlation weighted 70%, variation 30%
            arc_score = max(0.0, correlation) * 0.7 + variation_bonus * 0.3
            bonus = float(np.clip(arc_score, 0.0, 1.0)) * _MAX_COMPOSITIONAL_ARC_BONUS
            return round(bonus, 4)

        except Exception as exc:
            logger.warning("Compositional arc analysis failed: {}", exc)
            return 0.0
