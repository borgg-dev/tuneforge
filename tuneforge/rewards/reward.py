"""
Production reward model for TuneForge.

Combines five signal sources into a single 0-1 reward per miner response:
- CLAP text-audio similarity  (35%)
- Audio quality metrics        (25%)
- Preference model             (20%)
- Diversity                    (10%)
- Speed                        (10%)

Hard penalties override composite scores (plagiarism, silence, timeout).
"""

import io
import time

import numpy as np
from loguru import logger

from tuneforge.base.protocol import MusicGenerationSynapse
from tuneforge.config.scoring_config import (
    SCORING_WEIGHTS,
    SILENCE_THRESHOLD,
    DURATION_TOLERANCE,
    DURATION_TOLERANCE_MAX,
    SPEED_BEST_SECONDS,
    SPEED_MID_SECONDS,
    SPEED_MID_SCORE,
    SPEED_MAX_SECONDS,
    GENERATION_TIMEOUT,
)
from tuneforge.scoring.audio_quality import AudioQualityScorer
from tuneforge.scoring.clap_scorer import CLAPScorer
from tuneforge.scoring.diversity import DiversityScorer
from tuneforge.scoring.plagiarism import PlagiarismDetector
from tuneforge.scoring.preference_model import PreferenceModel
from tuneforge.settings import Settings


class ProductionRewardModel:
    """Full scoring pipeline combining all signal sources."""

    def __init__(self, config: Settings) -> None:
        self._clap = CLAPScorer(model_name=config.clap_model_name)
        self._quality = AudioQualityScorer()
        self._preference = PreferenceModel(model_path=None)
        self._plagiarism = PlagiarismDetector(db_path="fingerprints.db")
        self._diversity = DiversityScorer()
        self._config = config
        logger.info("ProductionRewardModel initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_response(
        self,
        synapse: MusicGenerationSynapse,
        all_responses: list[MusicGenerationSynapse],
        miner_hotkey: str = "",
    ) -> float:
        """
        Score a single miner response.

        Args:
            synapse: The response synapse from the miner.
            all_responses: All responses in this round (for diversity).
            miner_hotkey: Miner's hotkey for plagiarism tracking.

        Returns:
            Composite reward in [0, 1].
        """
        # --- Decode audio ---
        audio, sr = self._decode_audio(synapse)
        if audio is None:
            logger.debug("No audio in response — score 0")
            return 0.0

        # --- Hard penalties ---
        if self._is_silent(audio):
            logger.debug("Silent audio — score 0")
            return 0.0

        if self._is_timeout(synapse):
            logger.debug("Timeout response — score 0")
            return 0.0

        is_plagiarized, _ = self._plagiarism.check(
            audio, sr, miner_hotkey, synapse.challenge_id
        )
        if is_plagiarized:
            logger.debug("Plagiarized audio — score 0")
            return 0.0

        # --- Component scores ---
        clap_score = self._clap.score(audio, sr, synapse.prompt)
        quality_scores = self._quality.score(audio, sr)
        quality_score = self._quality.aggregate(quality_scores)
        preference_score = self._preference.score(audio, sr)
        speed_score = self._speed_score(synapse)

        # Diversity computed externally per-batch; use 0.5 default for single
        diversity_score = 0.5

        # --- Duration penalty (linear) ---
        duration_penalty = self._duration_penalty(audio, sr, synapse.duration_seconds)

        # --- Weighted composite ---
        composite = (
            SCORING_WEIGHTS["clap"] * clap_score
            + SCORING_WEIGHTS["quality"] * quality_score
            + SCORING_WEIGHTS["preference"] * preference_score
            + SCORING_WEIGHTS["diversity"] * diversity_score
            + SCORING_WEIGHTS["speed"] * speed_score
        )

        final = composite * duration_penalty

        logger.debug(
            f"Scores: clap={clap_score:.3f} quality={quality_score:.3f} "
            f"pref={preference_score:.3f} speed={speed_score:.3f} "
            f"dur_pen={duration_penalty:.3f} → {final:.3f}"
        )
        return float(np.clip(final, 0.0, 1.0))

    def score_batch(
        self,
        synapses: list[MusicGenerationSynapse],
        miner_hotkeys: list[str],
    ) -> list[float]:
        """
        Score a full batch of responses, including diversity.

        Args:
            synapses: All response synapses in this round.
            miner_hotkeys: Corresponding miner hotkeys.

        Returns:
            List of reward scores, aligned with input.
        """
        # Pre-compute diversity scores for the whole batch
        diversity_scores = self._diversity.score_batch(synapses)

        rewards: list[float] = []
        for i, synapse in enumerate(synapses):
            audio, sr = self._decode_audio(synapse)
            hotkey = miner_hotkeys[i] if i < len(miner_hotkeys) else ""

            if audio is None or self._is_silent(audio) or self._is_timeout(synapse):
                rewards.append(0.0)
                continue

            is_plagiarized, _ = self._plagiarism.check(
                audio, sr, hotkey, synapse.challenge_id
            )
            if is_plagiarized:
                rewards.append(0.0)
                continue

            clap_score = self._clap.score(audio, sr, synapse.prompt)
            quality_scores = self._quality.score(audio, sr)
            quality_score = self._quality.aggregate(quality_scores)
            preference_score = self._preference.score(audio, sr)
            speed_score = self._speed_score(synapse)
            diversity_score = diversity_scores[i]

            duration_penalty = self._duration_penalty(audio, sr, synapse.duration_seconds)

            composite = (
                SCORING_WEIGHTS["clap"] * clap_score
                + SCORING_WEIGHTS["quality"] * quality_score
                + SCORING_WEIGHTS["preference"] * preference_score
                + SCORING_WEIGHTS["diversity"] * diversity_score
                + SCORING_WEIGHTS["speed"] * speed_score
            )

            final = float(np.clip(composite * duration_penalty, 0.0, 1.0))
            rewards.append(final)

        # Clear plagiarism round cache
        self._plagiarism.clear_round_cache()

        return rewards

    # ------------------------------------------------------------------
    # Audio decoding
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_audio(synapse: MusicGenerationSynapse) -> tuple[np.ndarray | None, int]:
        """Decode audio bytes from synapse into numpy array."""
        try:
            raw = synapse.deserialize()
            if raw is None:
                return None, 0
            import soundfile as sf

            audio, sr = sf.read(io.BytesIO(raw))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32), sr
        except Exception as exc:
            logger.debug(f"Audio decode failed: {exc}")
            return None, 0

    # ------------------------------------------------------------------
    # Penalty helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_silent(audio: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        return rms < SILENCE_THRESHOLD

    @staticmethod
    def _is_timeout(synapse: MusicGenerationSynapse) -> bool:
        """Check timeout using validator-measured round-trip time."""
        try:
            if synapse.dendrite and synapse.dendrite.process_time is not None:
                return float(synapse.dendrite.process_time) > GENERATION_TIMEOUT
        except (ValueError, TypeError, AttributeError):
            pass
        # Fallback to miner-reported time
        if synapse.generation_time_ms is None:
            return False
        return synapse.generation_time_ms > GENERATION_TIMEOUT * 1000

    @staticmethod
    def _duration_penalty(audio: np.ndarray, sr: int, requested: float) -> float:
        """
        Linear duration penalty.

        ±20% → no penalty.  ±50% → score 0.  Linear between.
        """
        if sr <= 0 or requested <= 0:
            return 1.0
        actual = len(audio) / sr
        deviation = abs(actual - requested) / requested

        if deviation <= DURATION_TOLERANCE:
            return 1.0
        if deviation >= DURATION_TOLERANCE_MAX:
            return 0.0
        # Linear ramp between tolerance and max tolerance
        return 1.0 - (deviation - DURATION_TOLERANCE) / (
            DURATION_TOLERANCE_MAX - DURATION_TOLERANCE
        )

    @staticmethod
    def _speed_score(synapse: MusicGenerationSynapse) -> float:
        """
        Speed score: 5s=1.0, 30s=0.3, >60s=0.0.

        Uses validator-measured round-trip time (synapse.dendrite.process_time)
        instead of miner-reported generation_time_ms to prevent gaming.
        """
        # Prefer validator-measured round-trip time (set by dendrite)
        gen_seconds = None
        try:
            if synapse.dendrite and synapse.dendrite.process_time is not None:
                gen_seconds = float(synapse.dendrite.process_time)
        except (ValueError, TypeError, AttributeError):
            pass

        if gen_seconds is None:
            return 0.5  # no validator timing available — neutral

        if gen_seconds <= SPEED_BEST_SECONDS:
            return 1.0
        if gen_seconds >= SPEED_MAX_SECONDS:
            return 0.0
        if gen_seconds <= SPEED_MID_SECONDS:
            # Linear from 1.0 at BEST to MID_SCORE at MID
            frac = (gen_seconds - SPEED_BEST_SECONDS) / (
                SPEED_MID_SECONDS - SPEED_BEST_SECONDS
            )
            return 1.0 - frac * (1.0 - SPEED_MID_SCORE)
        # Linear from MID_SCORE at MID to 0.0 at MAX
        frac = (gen_seconds - SPEED_MID_SECONDS) / (
            SPEED_MAX_SECONDS - SPEED_MID_SECONDS
        )
        return SPEED_MID_SCORE * (1.0 - frac)
