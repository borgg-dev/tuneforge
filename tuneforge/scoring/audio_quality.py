"""
Audio quality scorer for TuneForge.

Analyses spectral and temporal properties of generated audio
to produce per-metric and aggregate quality scores.
"""

import numpy as np
from loguru import logger

from tuneforge.config.scoring_config import QUALITY_WEIGHTS, SILENCE_THRESHOLD


class AudioQualityScorer:
    """Assess audio quality using spectral and temporal analysis."""

    def score(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """
        Compute per-metric quality scores.

        Args:
            audio: 1-D float waveform.
            sr: Sample rate in Hz.

        Returns:
            Dict with keys: clipping, dynamic_range, spectral_quality,
            content_ratio, bandwidth, structure.  All values in [0, 1].
        """
        try:
            import librosa

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            return {
                "clipping": self._score_clipping(audio),
                "dynamic_range": self._score_dynamic_range(audio),
                "spectral_quality": self._score_spectral_quality(audio, sr, librosa),
                "content_ratio": self._score_content_ratio(audio),
                "bandwidth": self._score_bandwidth(audio, sr, librosa),
                "structure": self._score_structure(audio, sr, librosa),
            }
        except Exception as exc:
            logger.error(f"Audio quality scoring failed: {exc}")
            return {k: 0.0 for k in QUALITY_WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        """
        Weighted aggregation of per-metric scores.

        Args:
            scores: Dict from ``score()``.

        Returns:
            Aggregate quality score in [0, 1].
        """
        total = 0.0
        for metric, weight in QUALITY_WEIGHTS.items():
            total += scores.get(metric, 0.0) * weight
        return float(np.clip(total, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_clipping(audio: np.ndarray) -> float:
        """Score inversely proportional to clipping amount."""
        peak = np.max(np.abs(audio))
        if peak < 1e-8:
            return 0.0
        clip_threshold = 0.99
        clipped_samples = np.sum(np.abs(audio) > clip_threshold)
        clip_ratio = clipped_samples / len(audio)
        # 0% clipped → 1.0, ≥5% clipped → 0.0
        return float(np.clip(1.0 - clip_ratio / 0.05, 0.0, 1.0))

    @staticmethod
    def _score_dynamic_range(audio: np.ndarray) -> float:
        """Score based on dynamic range in dB."""
        peak = np.max(np.abs(audio))
        if peak < 1e-8:
            return 0.0
        # Frame-level RMS
        frame_len = 2048
        n_frames = max(1, len(audio) // frame_len)
        frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
        rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1))
        rms_per_frame = rms_per_frame[rms_per_frame > 1e-8]
        if len(rms_per_frame) < 2:
            return 0.0
        rms_db = 20.0 * np.log10(rms_per_frame + 1e-10)
        dynamic_range = float(np.max(rms_db) - np.min(rms_db))
        # 20 dB range → 1.0, 0 dB → 0.0
        return float(np.clip(dynamic_range / 20.0, 0.0, 1.0))

    @staticmethod
    def _score_spectral_quality(audio: np.ndarray, sr: int, librosa) -> float:
        """Score based on spectral centroid and flatness."""
        try:
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            flatness = librosa.feature.spectral_flatness(y=audio)[0]

            avg_centroid = float(np.mean(centroid))
            avg_flatness = float(np.mean(flatness))

            # Musical audio: centroid roughly 500-4000 Hz, low flatness
            centroid_score = float(np.clip(avg_centroid / 4000.0, 0.0, 1.0))
            # Lower flatness → more tonal → higher score
            tonality_score = float(np.clip(1.0 - avg_flatness, 0.0, 1.0))
            return (centroid_score + tonality_score) / 2.0
        except Exception:
            return 0.0

    @staticmethod
    def _score_content_ratio(audio: np.ndarray) -> float:
        """Score based on ratio of non-silent content."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < SILENCE_THRESHOLD:
            return 0.0
        frame_len = 2048
        n_frames = max(1, len(audio) // frame_len)
        frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
        rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1))
        active_frames = np.sum(rms_per_frame > SILENCE_THRESHOLD)
        ratio = active_frames / n_frames
        return float(np.clip(ratio, 0.0, 1.0))

    @staticmethod
    def _score_bandwidth(audio: np.ndarray, sr: int, librosa) -> float:
        """Score based on spectral bandwidth utilisation."""
        try:
            bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            avg_bw = float(np.mean(bw))
            nyquist = sr / 2.0
            # Good music uses at least 30-60% of available bandwidth
            utilisation = avg_bw / nyquist
            return float(np.clip(utilisation / 0.5, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_structure(audio: np.ndarray, sr: int, librosa) -> float:
        """Score based on temporal self-similarity (structural repetition)."""
        try:
            # Chroma features for self-similarity
            hop_length = 512
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)
            if chroma.shape[1] < 4:
                return 0.0
            # Auto-correlation of chroma for structure detection
            chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)
            n_frames = chroma_norm.shape[1]
            mid = n_frames // 2
            first_half = chroma_norm[:, :mid]
            second_half = chroma_norm[:, mid : mid + first_half.shape[1]]
            if first_half.shape[1] == 0:
                return 0.0
            similarity = np.mean(np.sum(first_half * second_half, axis=0))
            return float(np.clip(similarity, 0.0, 1.0))
        except Exception:
            return 0.0
