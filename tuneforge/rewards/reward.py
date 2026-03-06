"""
Production reward model for TuneForge.

Combines signal sources into a single 0-1 reward per miner response.
Quality is the primary driver; prompt adherence is secondary:

Prompt adherence (20%):
- CLAP text-audio similarity   (20%)

Attribute verification (10%):
- Tempo, key, instruments      (10%)

Music quality (54% total):
- Musicality metrics            (10%) — pitch, harmony, rhythm, arrangement, chords
- Neural quality (MERT)         (8%)  — learned music representations
- Melody coherence              (7%)  — melodic intervals, contour, structure
- Structural completeness       (7%)  — section detection, song form
- Harmonic quality              (6%)  — vocal presence, clarity, formant structure
- Production quality metrics    (6%)  — spectral balance, loudness, dynamics, stereo
- Preference model              (6%)  — learned human preference (auto-scales 2-10%)
- Audio quality metrics         (4%)  — signal-level analysis
- Perceptual quality            (4%)  — spectral MOS estimator
- Neural codec quality          (2%)  — EnCodec reconstruction quality

Other (10%):
- Diversity                     (5%)
- Speed                         (5%)

Penalties (applied as multipliers, not weighted components):
- Duration penalty              — linear ramp for off-target duration
- Artifact penalty              — spectral discontinuities, clipping, loops
- FAD penalty                   — per-miner Frechet Audio Distance from real music

Hard penalties override composite scores (plagiarism, silence, timeout).

Anti-gaming: per-round weight perturbation (±30%) with scorer dropout (10%).
"""

import hashlib
import io
import time

import numpy as np
from loguru import logger

from tuneforge.base.protocol import MusicGenerationSynapse
from tuneforge.config.scoring_config import (
    SCORING_WEIGHTS,
    SCORER_DROPOUT_RATE,
    SILENCE_THRESHOLD,
    DURATION_TOLERANCE,
    DURATION_TOLERANCE_MAX,
    SPEED_BEST_SECONDS,
    SPEED_MID_SECONDS,
    SPEED_MID_SCORE,
    SPEED_MAX_SECONDS,
    GENERATION_TIMEOUT,
    WEIGHT_PERTURBATION,
    FAD_WINDOW_SIZE,
    FAD_REFERENCE_STATS_PATH,
    FAD_PENALTY_MIDPOINT,
    FAD_PENALTY_STEEPNESS,
    FAD_PENALTY_FLOOR,
)
from tuneforge.scoring.artifact_detector import ArtifactDetector
from tuneforge.scoring.attribute_verifier import AttributeVerifier
from tuneforge.scoring.audio_quality import AudioQualityScorer
from tuneforge.scoring.clap_scorer import CLAPScorer
from tuneforge.scoring.diversity import DiversityScorer
from tuneforge.scoring.fad_scorer import FADScorer
from tuneforge.scoring.harmonic_quality import HarmonicQualityScorer
from tuneforge.scoring.melody_coherence import MelodyCoherenceScorer
from tuneforge.scoring.musicality import MusicalityScorer
from tuneforge.scoring.neural_codec_quality import NeuralCodecQualityScorer
from tuneforge.scoring.neural_quality import NeuralQualityScorer
from tuneforge.scoring.perceptual_quality import PerceptualQualityScorer
from tuneforge.scoring.plagiarism import PlagiarismDetector
from tuneforge.scoring.preference_model import PreferenceModel
from tuneforge.scoring.production_quality import ProductionQualityScorer
from tuneforge.scoring.structural_completeness import StructuralCompletenessScorer
from tuneforge.settings import Settings


class ProductionRewardModel:
    """Full scoring pipeline combining all signal sources."""

    def __init__(self, config: Settings) -> None:
        self._clap = CLAPScorer(model_name=config.clap_model_name)
        self._quality = AudioQualityScorer()
        self._musicality = MusicalityScorer()
        self._production = ProductionQualityScorer()
        self._melody = MelodyCoherenceScorer()
        self._neural = NeuralQualityScorer()
        self._structural = StructuralCompletenessScorer()
        self._vocal = HarmonicQualityScorer()
        self._artifact = ArtifactDetector()
        self._attribute = AttributeVerifier(clap_scorer=self._clap)
        self._perceptual = PerceptualQualityScorer()
        self._neural_codec = NeuralCodecQualityScorer()
        self._preference = PreferenceModel(
            model_path=config.preference_model_path,
            clap_scorer=self._clap,
            neural_scorer=self._neural,
        )
        self._plagiarism = PlagiarismDetector(
            clap_scorer=self._clap,
            reference_embeddings_path=getattr(config, "reference_embeddings_path", None),
        )
        self._fad = FADScorer(
            window_size=FAD_WINDOW_SIZE,
            reference_stats_path=FAD_REFERENCE_STATS_PATH or None,
            penalty_midpoint=FAD_PENALTY_MIDPOINT,
            penalty_steepness=FAD_PENALTY_STEEPNESS,
            penalty_floor=FAD_PENALTY_FLOOR,
        )
        self._diversity = DiversityScorer()
        self._config = config
        logger.info("ProductionRewardModel initialised (14 scorers + 3 penalties)")

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
        audio, sr, raw_audio = self._decode_audio(synapse)
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

        # --- Component scores (genre-aware) ---
        genre = getattr(synapse, "genre", "") or ""
        challenge_id = getattr(synapse, "challenge_id", "") or ""

        clap_score = self._clap.score(audio, sr, synapse.prompt)
        quality_scores = self._quality.score(audio, sr, genre=genre)
        quality_score = self._quality.aggregate(quality_scores)
        musicality_scores = self._musicality.score(audio, sr, genre=genre)
        musicality_score = self._musicality.aggregate(musicality_scores)
        production_scores = self._production.score(audio, sr, genre=genre, raw_audio=raw_audio)
        production_score = self._production.aggregate(production_scores)
        melody_scores = self._melody.score(audio, sr)
        melody_score = self._melody.aggregate(melody_scores)
        neural_scores = self._neural.score(audio, sr)
        neural_score = self._neural.aggregate(neural_scores)
        structural_scores = self._structural.score(audio, sr, genre=genre)
        structural_score = self._structural.aggregate(structural_scores)
        vocal_scores = self._vocal.score(audio, sr, genre=genre)
        vocal_score = self._vocal.aggregate(vocal_scores)
        attribute_score = self._attribute.verify_all(audio, sr, synapse)
        preference_score = self._preference.score(audio, sr)
        perceptual_scores = self._perceptual.score(audio, sr)
        perceptual_score = self._perceptual.aggregate(perceptual_scores)
        neural_codec_scores = self._neural_codec.score(audio, sr)
        neural_codec_score = self._neural_codec.aggregate(neural_codec_scores)
        speed_score = self._speed_score(synapse)

        # Diversity computed externally per-batch; use 0.5 default for single
        diversity_score = 0.5

        # --- Penalty multipliers ---
        duration_penalty = self._duration_penalty(audio, sr, synapse.duration_seconds)
        artifact_penalty = self._artifact.detect(audio, sr)

        # --- Per-round weight perturbation (anti-gaming) ---
        weights = self._perturb_weights(challenge_id)

        # --- Weighted composite ---
        composite = (
            weights["clap"] * clap_score
            + weights["quality"] * quality_score
            + weights["musicality"] * musicality_score
            + weights["production"] * production_score
            + weights["melody"] * melody_score
            + weights["neural_quality"] * neural_score
            + weights["structural"] * structural_score
            + weights["vocal"] * vocal_score
            + weights["attribute"] * attribute_score
            + weights["preference"] * preference_score
            + weights["perceptual"] * perceptual_score
            + weights["neural_codec"] * neural_codec_score
            + weights["diversity"] * diversity_score
            + weights["speed"] * speed_score
        )

        final = composite * duration_penalty * artifact_penalty

        logger.debug(
            f"Scores: clap={clap_score:.3f} quality={quality_score:.3f} "
            f"musicality={musicality_score:.3f} production={production_score:.3f} "
            f"melody={melody_score:.3f} neural={neural_score:.3f} "
            f"struct={structural_score:.3f} vocal={vocal_score:.3f} "
            f"attr={attribute_score:.3f} pref={preference_score:.3f} "
            f"perceptual={perceptual_score:.3f} codec={neural_codec_score:.3f} "
            f"speed={speed_score:.3f} div={diversity_score:.3f} "
            f"dur_pen={duration_penalty:.3f} art_pen={artifact_penalty:.3f} → {final:.3f}"
        )

        # Clear round cache after single-response scoring to prevent
        # stale embeddings leaking across calls (NEW-02 fix).
        self._plagiarism.clear_round_cache()

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
        # Pre-compute diversity scores for the whole batch (intra-miner)
        diversity_scores = self._diversity.score_batch(synapses, miner_hotkeys)

        rewards: list[float] = []
        try:
            for i, synapse in enumerate(synapses):
                audio, sr, raw_audio = self._decode_audio(synapse)
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

                genre = getattr(synapse, "genre", "") or ""
                challenge_id = getattr(synapse, "challenge_id", "") or ""

                clap_score = self._clap.score(audio, sr, synapse.prompt)
                quality_scores = self._quality.score(audio, sr, genre=genre)
                quality_score = self._quality.aggregate(quality_scores)
                musicality_scores = self._musicality.score(audio, sr, genre=genre)
                musicality_score = self._musicality.aggregate(musicality_scores)
                production_scores = self._production.score(audio, sr, genre=genre, raw_audio=raw_audio)
                production_score = self._production.aggregate(production_scores)
                melody_scores = self._melody.score(audio, sr)
                melody_score = self._melody.aggregate(melody_scores)
                neural_scores = self._neural.score(audio, sr)
                neural_score = self._neural.aggregate(neural_scores)
                structural_scores = self._structural.score(audio, sr, genre=genre)
                structural_score = self._structural.aggregate(structural_scores)
                vocal_scores = self._vocal.score(audio, sr, genre=genre)
                vocal_score = self._vocal.aggregate(vocal_scores)
                attribute_score = self._attribute.verify_all(audio, sr, synapse)
                preference_score = self._preference.score(audio, sr)
                perceptual_scores = self._perceptual.score(audio, sr)
                perceptual_score = self._perceptual.aggregate(perceptual_scores)
                neural_codec_scores = self._neural_codec.score(audio, sr)
                neural_codec_score = self._neural_codec.aggregate(neural_codec_scores)
                speed_score = self._speed_score(synapse)
                diversity_score = diversity_scores[i]

                duration_penalty = self._duration_penalty(audio, sr, synapse.duration_seconds)
                artifact_penalty = self._artifact.detect(audio, sr)

                # FAD: update miner embedding and get penalty
                clap_emb = self._clap.last_embedding
                self._fad.update_miner_embedding(hotkey, clap_emb)
                fad_penalty = self._fad.get_fad_penalty(hotkey)

                weights = self._perturb_weights(challenge_id)

                composite = (
                    weights["clap"] * clap_score
                    + weights["quality"] * quality_score
                    + weights["musicality"] * musicality_score
                    + weights["production"] * production_score
                    + weights["melody"] * melody_score
                    + weights["neural_quality"] * neural_score
                    + weights["structural"] * structural_score
                    + weights["vocal"] * vocal_score
                    + weights["attribute"] * attribute_score
                    + weights["preference"] * preference_score
                    + weights["perceptual"] * perceptual_score
                    + weights["neural_codec"] * neural_codec_score
                    + weights["diversity"] * diversity_score
                    + weights["speed"] * speed_score
                )

                final = float(np.clip(
                    composite * duration_penalty * artifact_penalty * fad_penalty, 0.0, 1.0
                ))

                logger.debug(
                    f"UID scoring: clap={clap_score:.3f} quality={quality_score:.3f} "
                    f"musicality={musicality_score:.3f} production={production_score:.3f} "
                    f"melody={melody_score:.3f} neural={neural_score:.3f} "
                    f"struct={structural_score:.3f} vocal={vocal_score:.3f} "
                    f"attr={attribute_score:.3f} pref={preference_score:.3f} "
                    f"perceptual={perceptual_score:.3f} codec={neural_codec_score:.3f} "
                    f"speed={speed_score:.3f} div={diversity_score:.3f} "
                    f"dur_pen={duration_penalty:.3f} art_pen={artifact_penalty:.3f} "
                    f"fad_pen={fad_penalty:.3f} → {final:.3f}"
                )

                rewards.append(final)
        finally:
            # Always clear plagiarism round cache, even on exception
            self._plagiarism.clear_round_cache()

        # Update intra-miner diversity history after scoring so that the
        # current round's embedding does not inflate its own diversity score.
        self._diversity.update_history(synapses, miner_hotkeys)

        return rewards

    # ------------------------------------------------------------------
    # Audio decoding
    # ------------------------------------------------------------------

    # Maximum raw audio payload size (20 MB) to prevent resource exhaustion
    _MAX_AUDIO_BYTES: int = 20 * 1024 * 1024

    @staticmethod
    def _decode_audio(
        synapse: MusicGenerationSynapse,
    ) -> tuple[np.ndarray | None, int, np.ndarray | None]:
        """Decode audio bytes from synapse into numpy array.

        Returns:
            (mono_audio, sample_rate, raw_multichannel_or_none)
        """
        try:
            raw = synapse.deserialize()
            if raw is None:
                return None, 0, None
            if len(raw) > ProductionRewardModel._MAX_AUDIO_BYTES:
                logger.warning(
                    f"Audio payload too large ({len(raw)} bytes) — rejecting"
                )
                return None, 0, None
            import soundfile as sf

            audio, sr = sf.read(io.BytesIO(raw))

            # Keep raw multichannel for stereo quality scoring
            raw_audio = audio if audio.ndim > 1 else None

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Reject excessively long audio (>120s absolute limit)
            if sr > 0 and len(audio) / sr > 120.0:
                logger.warning(
                    f"Audio too long ({len(audio) / sr:.1f}s) — rejecting"
                )
                return None, 0, None

            audio = audio.astype(np.float32)

            # Reject audio containing NaN or Inf values (malicious payload)
            if not np.all(np.isfinite(audio)):
                logger.warning("Audio contains NaN/Inf values — rejecting")
                return None, 0, None

            return audio, sr, raw_audio
        except Exception as exc:
            logger.debug(f"Audio decode failed: {exc}")
            return None, 0, None

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

    @staticmethod
    def _perturb_weights(challenge_id: str) -> dict[str, float]:
        """
        Per-round weight perturbation for anti-gaming.

        Deterministic per challenge_id (reproducible for verification),
        but unpredictable to miners since challenge_id is generated by
        the validator.  Each weight is perturbed by ±WEIGHT_PERTURBATION
        and the result is renormalized to sum to 1.0.

        Returns SCORING_WEIGHTS unchanged if perturbation is disabled.
        """
        if WEIGHT_PERTURBATION <= 0.0 or not challenge_id:
            return dict(SCORING_WEIGHTS)

        # Deterministic seed from challenge_id
        seed = int(hashlib.md5(challenge_id.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)

        perturbed: dict[str, float] = {}
        for key, base_weight in SCORING_WEIGHTS.items():
            if base_weight == 0.0:
                perturbed[key] = 0.0
                continue
            factor = 1.0 + rng.uniform(-WEIGHT_PERTURBATION, WEIGHT_PERTURBATION)
            perturbed[key] = base_weight * factor

        # Scorer dropout: each non-zero scorer has a chance of being zeroed
        if SCORER_DROPOUT_RATE > 0:
            for key in list(perturbed.keys()):
                if perturbed[key] > 0 and rng.random() < SCORER_DROPOUT_RATE:
                    perturbed[key] = 0.0

        # Renormalize to sum to 1.0
        total = sum(perturbed.values())
        if total > 0:
            perturbed = {k: v / total for k, v in perturbed.items()}

        return perturbed
