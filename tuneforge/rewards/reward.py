"""
Production reward model for TuneForge.

Combines signal sources into a single 0-1 reward per miner response.
Prompt adherence is the primary user-facing signal; quality prevents
technical defects from reaching users.

18 weighted scorers + 4 penalty multipliers + multi-scale evaluation.

Prompt adherence (24%):
- CLAP text-audio similarity   (15%)  — kept at 15% to avoid gaming amplification
- Attribute verification        (9%)  — tempo, key, instruments (concrete, less gameable)

Composition (21%):
- Musicality metrics            (9%)  — pitch, harmony, rhythm, arrangement, chords
- Melody coherence              (6%)  — melodic intervals, contour, structure
- Structural completeness       (6%)  — section detection, song form

Production & fidelity (16%):
- Production quality metrics    (5%)  — spectral balance, loudness, dynamics, stereo
- Neural quality (MERT)         (5%)  — learned music representations
- Harmonic quality              (4%)  — vocal presence, clarity, formant structure
- Audio quality metrics         (2%)  — signal-level analysis

Naturalness & mix (20%):
- Vocal/lyrics quality          (8%)  — vocal clarity, lyrics intelligibility, pitch, sibilance
- Timbral naturalness           (5%)  — spectral envelope, harmonic decay, transients
- Mix separation                (4%)  — spectral clarity, frequency masking, spatial depth
- Multi-resolution quality      (3%)  — multi-resolution perceptual quality estimation

Perceptual quality (2%):
- Perceptual quality            (1%)  — spectral MOS estimator
- Neural codec quality          (1%)  — EnCodec reconstruction quality

Preference (0% bootstrap / 2-20% when trained):
- Preference model              — learned human preference (auto-scales 2-20%)
  Zeroed out in bootstrap mode (no trained model); weight redistributed to other scorers.

Other (10%):
- Diversity                     (8%)  — intra-miner + population-level diversity
- Speed                         (2%)  — duration-relative: gen_time/requested_duration

Penalties (applied as multipliers, not weighted components):
- Duration penalty              — linear ramp for off-target duration
- Artifact penalty              — spectral discontinuities, clipping, loops
- FAD penalty                   — per-miner Frechet Audio Distance from real music
- Soft plagiarism penalty       — smooth cosine penalty for near-matches (0.65-0.72)

Hard penalties override composite scores (plagiarism, silence, timeout).

Multi-scale evaluation adjusts weights based on audio duration:
- Short clips (<10s): emphasize production quality
- Long clips (>30s): emphasize structural coherence + melodic development

Vocals-requested boost: when synapse.vocals_requested is True, vocal_lyrics
weight is doubled and vocal weight is 1.5x (then renormalized).

Anti-gaming:
- Per-round weight perturbation (±30%) with SECRET seed (auto-generated if not set)
- Scorer dropout (10%)
- Canonical-form plagiarism detection (pitch/tempo normalized)
- Population-level diversity bonus
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
    GENERATION_TIMEOUT,
    WEIGHT_PERTURBATION,
    FAD_WINDOW_SIZE,
    FAD_REFERENCE_STATS_PATH,
    FAD_PENALTY_MIDPOINT,
    FAD_PENALTY_STEEPNESS,
    FAD_PENALTY_FLOOR,
    VALIDATOR_PERTURBATION_SECRET,
)
from tuneforge.scoring.artifact_detector import ArtifactDetector
from tuneforge.scoring.attribute_verifier import AttributeVerifier
from tuneforge.scoring.audio_quality import AudioQualityScorer
from tuneforge.scoring.clap_scorer import CLAPScorer
from tuneforge.scoring.diversity import DiversityScorer
from tuneforge.scoring.fad_scorer import FADScorer
from tuneforge.scoring.harmonic_quality import HarmonicQualityScorer
from tuneforge.scoring.learned_mos import LearnedMOSScorer
from tuneforge.scoring.melody_coherence import MelodyCoherenceScorer
from tuneforge.scoring.mix_separation import MixSeparationScorer
from tuneforge.scoring.multi_scale import MultiScaleEvaluator
from tuneforge.scoring.musicality import MusicalityScorer
from tuneforge.scoring.neural_codec_quality import NeuralCodecQualityScorer
from tuneforge.scoring.neural_quality import NeuralQualityScorer
from tuneforge.scoring.perceptual_quality import PerceptualQualityScorer
from tuneforge.scoring.plagiarism import PlagiarismDetector
from tuneforge.scoring.preference_model import PreferenceModel
from tuneforge.scoring.production_quality import ProductionQualityScorer
from tuneforge.scoring.structural_completeness import StructuralCompletenessScorer
from tuneforge.scoring.timbral_naturalness import TimbralNaturalnessScorer
from tuneforge.scoring.vocal_lyrics import VocalLyricsScorer
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
        self._diversity = DiversityScorer(clap_scorer=self._clap)
        self._timbral = TimbralNaturalnessScorer()
        self._vocal_lyrics = VocalLyricsScorer()
        self._mix_separation = MixSeparationScorer()
        self._learned_mos = LearnedMOSScorer()
        self._multi_scale = MultiScaleEvaluator()
        self._config = config
        logger.info("ProductionRewardModel initialised (18 scorers + 4 penalties + multi-scale)")

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
        expected_lyrics = getattr(synapse, "lyrics", "") or ""
        vocals_requested = getattr(synapse, "vocals_requested", False)

        clap_score = self._clap.score(audio, sr, synapse.prompt)
        # Capture CLAP embedding immediately — attribute verifier
        # also calls CLAP and overwrites last_embedding
        clap_emb_for_fad = self._clap.last_embedding
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

        # Naturalness & mix scorers
        timbral_scores = self._timbral.score(audio, sr, genre=genre)
        timbral_score = self._timbral.aggregate(timbral_scores)
        vocal_lyrics_scores = self._vocal_lyrics.score(
            audio, sr, genre=genre, expected_lyrics=expected_lyrics,
        )
        vocal_lyrics_score = self._vocal_lyrics.aggregate(vocal_lyrics_scores)
        mix_sep_scores = self._mix_separation.score(audio, sr, genre=genre)
        mix_sep_score = self._mix_separation.aggregate(mix_sep_scores)
        mos_scores = self._learned_mos.score(audio, sr)
        mos_score = self._learned_mos.aggregate(mos_scores)

        # Diversity computed externally per-batch; use 0.5 default for single
        diversity_score = 0.5

        # --- Penalty multipliers ---
        duration_penalty = self._duration_penalty(audio, sr, synapse.duration_seconds)
        artifact_penalty = self._artifact.detect(audio, sr)

        # Soft plagiarism penalty (computed during check() above, cached)
        soft_plag_penalty = self._plagiarism.get_soft_penalty(audio, sr, miner_hotkey)

        # FAD: update miner embedding and get penalty
        self._fad.update_miner_embedding(miner_hotkey, clap_emb_for_fad)
        fad_penalty = self._fad.get_fad_penalty(miner_hotkey)

        # --- Multi-scale weight adjustment ---
        duration_seconds = getattr(synapse, "duration_seconds", 10.0) or 10.0
        scale_multipliers = self._multi_scale.evaluate(audio, sr, duration_seconds)

        # --- Per-round weight perturbation (anti-gaming) ---
        weights = self._perturb_weights(challenge_id, VALIDATOR_PERTURBATION_SECRET)

        # --- Preference weight: zero in bootstrap, auto-scale when trained ---
        if self._preference.is_bootstrap:
            weights["preference"] = 0.0
        else:
            weights["preference"] = self._preference.get_scaled_weight()

        # --- Vocals-requested boost: increase vocal_lyrics weight when user
        # explicitly wants vocals, decrease instrumental-focused scorers ---
        if vocals_requested:
            weights["vocal_lyrics"] = weights.get("vocal_lyrics", 0) * 2.0
            weights["vocal"] = weights.get("vocal", 0) * 1.5

        # Extract multi-scale bonuses before applying multipliers to weights
        phrase_bonus = scale_multipliers.pop("phrase_coherence_bonus", 0.0)
        arc_bonus = scale_multipliers.pop("compositional_arc_bonus", 0.0)

        # Apply multi-scale multipliers to weights
        for key in weights:
            if key in scale_multipliers:
                weights[key] *= scale_multipliers[key]
        # Renormalize after multi-scale adjustment
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

        # --- Weighted composite ---
        composite = (
            weights.get("clap", 0) * clap_score
            + weights.get("quality", 0) * quality_score
            + weights.get("musicality", 0) * musicality_score
            + weights.get("production", 0) * production_score
            + weights.get("melody", 0) * melody_score
            + weights.get("neural_quality", 0) * neural_score
            + weights.get("structural", 0) * structural_score
            + weights.get("vocal", 0) * vocal_score
            + weights.get("attribute", 0) * attribute_score
            + weights.get("preference", 0) * preference_score
            + weights.get("perceptual", 0) * perceptual_score
            + weights.get("neural_codec", 0) * neural_codec_score
            + weights.get("timbral", 0) * timbral_score
            + weights.get("vocal_lyrics", 0) * vocal_lyrics_score
            + weights.get("mix_separation", 0) * mix_sep_score
            + weights.get("learned_mos", 0) * mos_score
            + weights.get("diversity", 0) * diversity_score
            + weights.get("speed", 0) * speed_score
            + phrase_bonus + arc_bonus
        )

        final = composite * duration_penalty * artifact_penalty * fad_penalty * soft_plag_penalty

        logger.debug(
            f"Scores: clap={clap_score:.3f} quality={quality_score:.3f} "
            f"musicality={musicality_score:.3f} production={production_score:.3f} "
            f"melody={melody_score:.3f} neural={neural_score:.3f} "
            f"struct={structural_score:.3f} vocal={vocal_score:.3f} "
            f"attr={attribute_score:.3f} pref={preference_score:.3f} "
            f"perceptual={perceptual_score:.3f} codec={neural_codec_score:.3f} "
            f"timbral={timbral_score:.3f} vocal_lyrics={vocal_lyrics_score:.3f} "
            f"mix_sep={mix_sep_score:.3f} mos={mos_score:.3f} "
            f"speed={speed_score:.3f} div={diversity_score:.3f} "
            f"dur_pen={duration_penalty:.3f} art_pen={artifact_penalty:.3f} "
            f"fad_pen={fad_penalty:.3f} plag_pen={soft_plag_penalty:.3f} → {final:.3f}"
        )

        # Clear round cache after single-response scoring to prevent
        # stale embeddings leaking across calls.
        # Note: this means cross-miner plagiarism detection only works in
        # score_batch(), not score_response(). This is by design — single
        # response scoring lacks the batch context needed for cross-miner checks.
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
                expected_lyrics = getattr(synapse, "lyrics", "") or ""
                vocals_requested = getattr(synapse, "vocals_requested", False)

                clap_score = self._clap.score(audio, sr, synapse.prompt)
                # Capture CLAP embedding immediately — attribute verifier
                # also calls CLAP and overwrites last_embedding
                clap_emb_for_fad = self._clap.last_embedding
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

                # Naturalness & mix scorers
                timbral_scores = self._timbral.score(audio, sr, genre=genre)
                timbral_score = self._timbral.aggregate(timbral_scores)
                vocal_lyrics_scores = self._vocal_lyrics.score(
                    audio, sr, genre=genre, expected_lyrics=expected_lyrics,
                )
                vocal_lyrics_score = self._vocal_lyrics.aggregate(vocal_lyrics_scores)
                mix_sep_scores = self._mix_separation.score(audio, sr, genre=genre)
                mix_sep_score = self._mix_separation.aggregate(mix_sep_scores)
                mos_scores = self._learned_mos.score(audio, sr)
                mos_score = self._learned_mos.aggregate(mos_scores)

                duration_penalty = self._duration_penalty(audio, sr, synapse.duration_seconds)
                artifact_penalty = self._artifact.detect(audio, sr)

                # Soft plagiarism penalty (partial score reduction for near-matches)
                soft_plag_penalty = self._plagiarism.get_soft_penalty(audio, sr, hotkey)

                # FAD: update miner embedding and get penalty
                self._fad.update_miner_embedding(hotkey, clap_emb_for_fad)
                fad_penalty = self._fad.get_fad_penalty(hotkey)

                # Multi-scale weight adjustment
                duration_seconds = getattr(synapse, "duration_seconds", 10.0) or 10.0
                scale_multipliers = self._multi_scale.evaluate(audio, sr, duration_seconds)

                weights = self._perturb_weights(challenge_id, VALIDATOR_PERTURBATION_SECRET)

                # Preference weight: zero in bootstrap, auto-scale when trained
                if self._preference.is_bootstrap:
                    weights["preference"] = 0.0
                else:
                    weights["preference"] = self._preference.get_scaled_weight()

                # Vocals-requested boost
                if vocals_requested:
                    weights["vocal_lyrics"] = weights.get("vocal_lyrics", 0) * 2.0
                    weights["vocal"] = weights.get("vocal", 0) * 1.5

                # Extract multi-scale bonuses before applying multipliers
                phrase_bonus = scale_multipliers.pop("phrase_coherence_bonus", 0.0)
                arc_bonus = scale_multipliers.pop("compositional_arc_bonus", 0.0)

                # Apply multi-scale multipliers
                for key in weights:
                    if key in scale_multipliers:
                        weights[key] *= scale_multipliers[key]
                total_w = sum(weights.values())
                if total_w > 0:
                    weights = {k: v / total_w for k, v in weights.items()}

                composite = (
                    weights.get("clap", 0) * clap_score
                    + weights.get("quality", 0) * quality_score
                    + weights.get("musicality", 0) * musicality_score
                    + weights.get("production", 0) * production_score
                    + weights.get("melody", 0) * melody_score
                    + weights.get("neural_quality", 0) * neural_score
                    + weights.get("structural", 0) * structural_score
                    + weights.get("vocal", 0) * vocal_score
                    + weights.get("attribute", 0) * attribute_score
                    + weights.get("preference", 0) * preference_score
                    + weights.get("perceptual", 0) * perceptual_score
                    + weights.get("neural_codec", 0) * neural_codec_score
                    + weights.get("timbral", 0) * timbral_score
                    + weights.get("vocal_lyrics", 0) * vocal_lyrics_score
                    + weights.get("mix_separation", 0) * mix_sep_score
                    + weights.get("learned_mos", 0) * mos_score
                    + weights.get("diversity", 0) * diversity_score
                    + weights.get("speed", 0) * speed_score
                    + phrase_bonus + arc_bonus
                )

                final = float(np.clip(
                    composite * duration_penalty * artifact_penalty * fad_penalty * soft_plag_penalty,
                    0.0, 1.0,
                ))

                logger.debug(
                    f"UID scoring: clap={clap_score:.3f} quality={quality_score:.3f} "
                    f"musicality={musicality_score:.3f} production={production_score:.3f} "
                    f"melody={melody_score:.3f} neural={neural_score:.3f} "
                    f"struct={structural_score:.3f} vocal={vocal_score:.3f} "
                    f"attr={attribute_score:.3f} pref={preference_score:.3f} "
                    f"perceptual={perceptual_score:.3f} codec={neural_codec_score:.3f} "
                    f"timbral={timbral_score:.3f} vocal_lyrics={vocal_lyrics_score:.3f} "
                    f"mix_sep={mix_sep_score:.3f} mos={mos_score:.3f} "
                    f"speed={speed_score:.3f} div={diversity_score:.3f} "
                    f"dur_pen={duration_penalty:.3f} art_pen={artifact_penalty:.3f} "
                    f"fad_pen={fad_penalty:.3f} plag_pen={soft_plag_penalty:.3f} → {final:.3f}"
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

            # Reject excessively long audio (>MAX_DURATION absolute limit)
            if sr > 0 and len(audio) / sr > 180.0:
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
        Duration-relative speed score.

        Uses the ratio of generation_time / requested_duration instead of
        absolute thresholds. This avoids penalizing longer generations:
        - ratio <= 1.0 (real-time or faster): score 1.0
        - ratio == 3.0: score 0.3
        - ratio >= 6.0: score 0.0

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

        # Duration-relative: compute generation-to-duration ratio
        requested_duration = getattr(synapse, "duration_seconds", 10.0) or 10.0
        ratio = gen_seconds / max(requested_duration, 1.0)

        # Ratio-based scoring curve:
        # ratio <= 1.0 (real-time or faster): 1.0
        # ratio == 3.0: 0.3
        # ratio >= 6.0: 0.0
        if ratio <= 1.0:
            return 1.0
        if ratio >= 6.0:
            return 0.0
        if ratio <= 3.0:
            # Linear from 1.0 at ratio=1.0 to 0.3 at ratio=3.0
            frac = (ratio - 1.0) / 2.0
            return 1.0 - frac * 0.7
        # Linear from 0.3 at ratio=3.0 to 0.0 at ratio=6.0
        frac = (ratio - 3.0) / 3.0
        return 0.3 * (1.0 - frac)

    @staticmethod
    def _perturb_weights(challenge_id: str, validator_secret: str = "") -> dict[str, float]:
        """
        Per-round weight perturbation for anti-gaming.

        Deterministic per (challenge_id + validator_secret), reproducible for
        verification by the same validator but unpredictable to miners.
        The validator_secret is a private nonce that is NEVER transmitted to
        miners, preventing them from reconstructing the perturbed weights.

        Each weight is perturbed by ±WEIGHT_PERTURBATION and the result is
        renormalized to sum to 1.0.

        Returns SCORING_WEIGHTS unchanged if perturbation is disabled.
        """
        if WEIGHT_PERTURBATION <= 0.0:
            return dict(SCORING_WEIGHTS)
        if not challenge_id:
            logger.warning("Empty challenge_id — perturbation disabled for this round")
            return dict(SCORING_WEIGHTS)

        # Deterministic seed from challenge_id + validator secret
        # The secret is never shared with miners, making weights unpredictable
        seed_material = f"{challenge_id}:{validator_secret}" if validator_secret else challenge_id
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16)
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
