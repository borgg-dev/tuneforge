"""
Plagiarism detection for TuneForge.

Uses CLAP audio embeddings for similarity-based detection:
- Reference DB: compare against known copyrighted material embeddings
- Cross-miner: detect near-identical submissions within a round
- Self-plagiarism: detect repeated submissions from the same miner
"""

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from loguru import logger

from tuneforge.config.scoring_config import SELF_PLAGIARISM_THRESHOLD


class PlagiarismDetector:
    """CLAP-embedding based plagiarism and copy detection."""

    def __init__(
        self,
        clap_scorer=None,
        reference_embeddings_path: str | None = None,
        reference_threshold: float = 0.85,
        self_similarity_threshold: float | None = None,
        history_maxlen: int = 50,
    ) -> None:
        self._clap = clap_scorer
        self._ref_threshold = reference_threshold
        self._self_threshold = self_similarity_threshold or SELF_PLAGIARISM_THRESHOLD
        self._reference_embeddings: np.ndarray | None = None
        self._miner_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_maxlen)
        )
        self._round_embeddings: dict[str, np.ndarray] = {}

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
            if sim > self._self_threshold:
                logger.warning(
                    "Cross-miner copy detected: {} vs {}, sim={:.3f}",
                    miner_hotkey[:8],
                    other_hotkey[:8],
                    sim,
                )
                return True, sim

        # 3. Self-plagiarism check (historical submissions)
        for hist_emb in self._miner_history[miner_hotkey]:
            sim = float(np.dot(embedding, hist_emb))
            if sim > self._self_threshold:
                logger.warning(
                    "Self-plagiarism detected for miner {}: sim={:.3f}",
                    miner_hotkey[:8],
                    sim,
                )
                return True, sim

        # Store embedding for future checks
        self._round_embeddings[f"{challenge_id}:{miner_hotkey}"] = embedding
        self._miner_history[miner_hotkey].append(embedding)

        return False, 0.0

    def clear_round_cache(self) -> None:
        """Clear per-round embedding cache after scoring completes."""
        self._round_embeddings.clear()
