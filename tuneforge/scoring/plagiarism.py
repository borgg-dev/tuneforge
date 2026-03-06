"""
Plagiarism detection for TuneForge.

Uses CLAP audio embeddings for similarity-based detection:
- Reference DB: compare against known copyrighted material embeddings
- Cross-miner: detect near-identical submissions within a round
- Self-plagiarism: detect repeated submissions from the same miner

Hardened with canonical-form comparison: audio is normalized in pitch
and tempo before embedding to prevent evasion via simple transforms.

Supports soft plagiarism zone: similarity between SOFT and HARD thresholds
returns a penalty multiplier rather than a hard zero.
"""

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from loguru import logger

from tuneforge.config.scoring_config import (
    SELF_PLAGIARISM_THRESHOLD,
    SOFT_PLAGIARISM_THRESHOLD,
    CROSS_MINER_PLAGIARISM_THRESHOLD,
)


class PlagiarismDetector:
    """CLAP-embedding based plagiarism and copy detection."""

    def __init__(
        self,
        clap_scorer=None,
        reference_embeddings_path: str | None = None,
        reference_threshold: float = 0.85,
        self_similarity_threshold: float | None = None,
        cross_miner_threshold: float | None = None,
        soft_threshold: float | None = None,
        history_maxlen: int = 50,
    ) -> None:
        self._clap = clap_scorer
        self._ref_threshold = reference_threshold
        self._self_threshold = self_similarity_threshold or SELF_PLAGIARISM_THRESHOLD
        self._cross_threshold = cross_miner_threshold or CROSS_MINER_PLAGIARISM_THRESHOLD
        self._soft_threshold = soft_threshold or SOFT_PLAGIARISM_THRESHOLD
        self._reference_embeddings: np.ndarray | None = None
        self._miner_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_maxlen)
        )
        self._round_embeddings: dict[str, np.ndarray] = {}
        # Also store canonical-form embeddings for transform-invariant detection
        self._miner_canonical_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_maxlen)
        )

        if reference_embeddings_path and Path(reference_embeddings_path).exists():
            try:
                self._reference_embeddings = np.load(reference_embeddings_path)
                if isinstance(self._reference_embeddings, np.lib.npyio.NpzFile):
                    self._reference_embeddings = self._reference_embeddings["embeddings"]
                logger.info(
                    "Loaded {} reference embeddings for plagiarism detection",
                    len(self._reference_embeddings),
                )
            except Exception as exc:
                logger.warning("Failed to load reference embeddings: {}", exc)
                self._reference_embeddings = None

    def check(
        self,
        audio: np.ndarray,
        sr: int,
        miner_hotkey: str,
        challenge_id: str,
    ) -> tuple[bool, float]:
        """Check audio for plagiarism.

        Returns:
            (is_plagiarized, max_similarity)
            is_plagiarized is True for hard plagiarism (score should be 0).
        """
        if self._clap is None:
            return False, 0.0

        embedding = self._clap.get_audio_embedding(audio, sr)
        if embedding is None:
            return False, 0.0

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return False, 0.0
        embedding = embedding / norm

        # Also compute canonical-form embedding (pitch/tempo normalized)
        canonical_emb = self._get_canonical_embedding(audio, sr)

        # 1. Reference DB check
        if self._reference_embeddings is not None and len(self._reference_embeddings) > 0:
            sims = self._reference_embeddings @ embedding
            max_sim = float(np.max(sims))
            if max_sim > self._ref_threshold:
                logger.warning(
                    "Reference plagiarism detected for miner {}: sim={:.3f}",
                    miner_hotkey[:8],
                    max_sim,
                )
                return True, max_sim

        # 2. Cross-miner check (within current round)
        for key, other_emb in self._round_embeddings.items():
            other_hotkey = key.split(":", 1)[1] if ":" in key else ""
            if other_hotkey == miner_hotkey:
                continue
            sim = float(np.dot(embedding, other_emb))
            if sim > self._cross_threshold:
                logger.warning(
                    "Cross-miner copy detected: {} vs {}, sim={:.3f}",
                    miner_hotkey[:8],
                    other_hotkey[:8],
                    sim,
                )
                return True, sim

        # 3. Self-plagiarism check (historical submissions)
        # Check both raw and canonical embeddings for transform-invariant detection
        max_self_sim = 0.0
        for hist_emb in self._miner_history[miner_hotkey]:
            sim = float(np.dot(embedding, hist_emb))
            max_self_sim = max(max_self_sim, sim)
            if sim > self._self_threshold:
                logger.warning(
                    "Self-plagiarism detected for miner {}: sim={:.3f}",
                    miner_hotkey[:8],
                    sim,
                )
                return True, sim

        # 3b. Canonical-form self-plagiarism (catches pitch/tempo transforms)
        if canonical_emb is not None:
            for hist_canonical in self._miner_canonical_history[miner_hotkey]:
                sim = float(np.dot(canonical_emb, hist_canonical))
                max_self_sim = max(max_self_sim, sim)
                if sim > self._self_threshold:
                    logger.warning(
                        "Canonical self-plagiarism detected for miner {}: sim={:.3f}",
                        miner_hotkey[:8],
                        sim,
                    )
                    return True, sim

        # Store embedding for future checks
        self._round_embeddings[f"{challenge_id}:{miner_hotkey}"] = embedding
        self._miner_history[miner_hotkey].append(embedding)
        if canonical_emb is not None:
            self._miner_canonical_history[miner_hotkey].append(canonical_emb)

        return False, max_self_sim

    def get_soft_penalty(
        self,
        audio: np.ndarray,
        sr: int,
        miner_hotkey: str,
    ) -> float:
        """Get soft plagiarism penalty multiplier.

        Returns 1.0 (no penalty) if similarity is below soft threshold.
        Returns 0.3 if similarity is in the soft zone [soft_threshold, hard_threshold).
        Hard plagiarism is handled by check() returning is_plagiarized=True.
        """
        if self._clap is None:
            return 1.0

        embedding = self._clap.get_audio_embedding(audio, sr)
        if embedding is None:
            return 1.0

        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return 1.0
        embedding = embedding / norm

        max_sim = 0.0
        for hist_emb in self._miner_history[miner_hotkey]:
            sim = float(np.dot(embedding, hist_emb))
            max_sim = max(max_sim, sim)

        if max_sim >= self._soft_threshold:
            # Soft penalty zone: linear interpolation from 1.0 at soft_threshold
            # to 0.3 at hard_threshold
            if max_sim >= self._self_threshold:
                return 0.3  # should not reach here (check() catches it)
            t = (max_sim - self._soft_threshold) / (
                self._self_threshold - self._soft_threshold + 1e-8
            )
            return 1.0 - t * 0.7  # 1.0 -> 0.3

        return 1.0

    def _get_canonical_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """Get CLAP embedding of pitch/tempo-normalized audio.

        Normalizes audio to a canonical form to detect plagiarism
        attempts that use pitch shifting or tempo changes.
        """
        try:
            import librosa

            # Normalize tempo: detect tempo, stretch to 120 BPM
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            if hasattr(tempo, '__len__'):
                tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
            tempo = float(tempo)
            if tempo > 0 and abs(tempo - 120.0) > 5.0:
                rate = tempo / 120.0
                # Clamp stretch rate to avoid extreme distortion
                rate = max(0.5, min(2.0, rate))
                audio_stretched = librosa.effects.time_stretch(audio, rate=rate)
            else:
                audio_stretched = audio

            # Normalize pitch: detect dominant pitch class, transpose to C
            chroma = librosa.feature.chroma_cqt(y=audio_stretched, sr=sr)
            mean_chroma = chroma.mean(axis=1)
            dominant_pitch = int(np.argmax(mean_chroma))
            # Shift to C (pitch class 0)
            if dominant_pitch != 0:
                n_steps = -dominant_pitch if dominant_pitch <= 6 else (12 - dominant_pitch)
                audio_canonical = librosa.effects.pitch_shift(
                    audio_stretched, sr=sr, n_steps=n_steps
                )
            else:
                audio_canonical = audio_stretched

            emb = self._clap.get_audio_embedding(audio_canonical.astype(np.float32), sr)
            if emb is None:
                return None
            norm = np.linalg.norm(emb)
            if norm < 1e-8:
                return None
            return emb / norm

        except Exception as exc:
            logger.debug("Canonical embedding failed: {}", exc)
            return None

    def clear_round_cache(self) -> None:
        """Clear per-round embedding cache after scoring completes."""
        self._round_embeddings.clear()
