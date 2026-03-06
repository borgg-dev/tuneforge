"""
Diversity scorer for TuneForge.

Measures both intra-miner diversity (varied outputs across rounds) and
population-level diversity (preventing collective convergence where all
miners produce similar-sounding music).

Uses intra-miner cosine distance against each miner's own submission
history (not inter-miner, which would reward off-topic audio).

Features:
- 50-entry per-miner history
- Population-level diversity bonus (30% weight)
"""

from collections import defaultdict, deque

import numpy as np
from loguru import logger

from tuneforge.config.scoring_config import CLAP_MODEL
from tuneforge.scoring.clap_scorer import CLAPScorer
from tuneforge.base.protocol import MusicGenerationSynapse

# Number of past embeddings to retain per miner
_HISTORY_MAXLEN: int = 50

# Default score for miners with no submission history yet
_DEFAULT_SCORE: float = 0.5

# Population diversity: bonus weight for being different from other miners
_POPULATION_DIVERSITY_WEIGHT: float = 0.3


class DiversityScorer:
    """Score intra-miner and population-level output diversity via CLAP embeddings."""

    def __init__(self, clap_scorer: CLAPScorer | None = None) -> None:
        self._clap = clap_scorer or CLAPScorer(model_name=CLAP_MODEL)
        # Maps hotkey -> deque of past CLAP embeddings (up to _HISTORY_MAXLEN)
        self._miner_history: dict[str, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=_HISTORY_MAXLEN)
        )
        # Cache embeddings from score_batch for reuse in update_history
        self._batch_embeddings: list[np.ndarray | None] = []

    def score_batch(
        self,
        responses: list[MusicGenerationSynapse],
        hotkeys: list[str],
    ) -> list[float]:
        """
        Score diversity for a batch of miner responses.

        Combines:
        - Intra-miner diversity (70%): cosine distance from own history
        - Population diversity (30%): cosine distance from other miners in this batch

        Args:
            responses: Synapses with audio_b64 populated.
            hotkeys:   Miner hotkeys aligned with ``responses``.

        Returns:
            Per-response diversity scores in [0, 1], aligned with input list.
        """
        if len(responses) != len(hotkeys):
            raise ValueError(
                f"responses length ({len(responses)}) must match "
                f"hotkeys length ({len(hotkeys)})"
            )

        # Extract all embeddings first for population diversity
        embeddings: list[np.ndarray | None] = []
        for resp in responses:
            embeddings.append(self._extract_embedding(resp))

        # Cache for reuse in update_history (avoids re-extraction)
        self._batch_embeddings = list(embeddings)

        scores: list[float] = []

        for i, (resp, hotkey) in enumerate(zip(responses, hotkeys)):
            emb = embeddings[i]

            if emb is None:
                scores.append(0.0)
                continue

            # --- Intra-miner diversity ---
            past = self._miner_history[hotkey]
            if len(past) == 0:
                intra_score = _DEFAULT_SCORE
            else:
                intra_score = self._mean_cosine_distance(emb, list(past))

            # --- Population diversity ---
            # Mean cosine distance from other miners' embeddings in this batch
            other_embs = [
                embeddings[j] for j in range(len(embeddings))
                if j != i and embeddings[j] is not None and hotkeys[j] != hotkey
            ]
            if len(other_embs) > 0:
                pop_score = self._mean_cosine_distance(emb, other_embs)
            else:
                pop_score = _DEFAULT_SCORE

            # Combine
            combined = (
                (1.0 - _POPULATION_DIVERSITY_WEIGHT) * intra_score
                + _POPULATION_DIVERSITY_WEIGHT * pop_score
            )
            scores.append(float(np.clip(combined, 0.0, 1.0)))

        return scores

    def update_history(
        self,
        responses: list[MusicGenerationSynapse],
        hotkeys: list[str],
    ) -> None:
        """
        Store current-round embeddings into each miner's history.

        Call this *after* ``score_batch`` so that history is not updated
        before scoring (which would inflate self-similarity scores).

        Reuses cached embeddings from score_batch when available to avoid
        redundant CLAP inference.
        """
        if len(responses) != len(hotkeys):
            raise ValueError(
                f"responses length ({len(responses)}) must match "
                f"hotkeys length ({len(hotkeys)})"
            )

        use_cache = len(self._batch_embeddings) == len(responses)

        for i, (resp, hotkey) in enumerate(zip(responses, hotkeys)):
            emb = self._batch_embeddings[i] if use_cache else self._extract_embedding(resp)
            if emb is not None:
                self._miner_history[hotkey].append(emb)
                logger.debug(
                    f"Updated history for {hotkey}: "
                    f"{len(self._miner_history[hotkey])} embedding(s) stored"
                )

        # Clear cache after use
        self._batch_embeddings = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mean_cosine_distance(
        self,
        current: np.ndarray,
        past: list[np.ndarray],
    ) -> float:
        """
        Compute the mean cosine distance between *current* and each vector in
        *past*.  Returns a value in [0, 1] where 1 means maximally different.
        """
        cur_norm = current / (np.linalg.norm(current) + 1e-8)

        distances: list[float] = []
        for past_emb in past:
            past_norm = past_emb / (np.linalg.norm(past_emb) + 1e-8)
            similarity = float(np.dot(cur_norm, past_norm))
            distance = float(np.clip(1.0 - similarity, 0.0, 1.0))
            distances.append(distance)

        return float(np.mean(distances))

    def _extract_embedding(self, synapse: MusicGenerationSynapse) -> np.ndarray | None:
        """Decode audio from synapse and compute CLAP embedding."""
        try:
            audio_bytes = synapse.deserialize()
            if audio_bytes is None:
                return None

            import soundfile as sf
            import io

            audio, sr = sf.read(io.BytesIO(audio_bytes))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return self._clap.get_audio_embedding(audio.astype(np.float32), sr)
        except Exception as exc:
            logger.debug(f"Diversity embedding extraction failed: {exc}")
            return None
