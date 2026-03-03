"""
Diversity scorer for TuneForge.

Measures intra-miner diversity using CLAP audio embeddings.
Miners who produce varied outputs across rounds score higher than those
who resubmit near-identical audio in successive rounds.

MEDIUM-01 fix: replaced inter-miner cosine distance (which rewarded
off-topic audio and penalised correct consensus) with intra-miner cosine
distance against each miner's own submission history.
"""

from collections import defaultdict, deque

import numpy as np
from loguru import logger

from tuneforge.config.scoring_config import CLAP_MODEL
from tuneforge.scoring.clap_scorer import CLAPScorer
from tuneforge.base.protocol import MusicGenerationSynapse

# Number of past embeddings to retain per miner
_HISTORY_MAXLEN: int = 10

# Default score for miners with no submission history yet
_DEFAULT_SCORE: float = 0.5


class DiversityScorer:
    """Score intra-miner output diversity via CLAP embeddings."""

    def __init__(self) -> None:
        self._clap = CLAPScorer(model_name=CLAP_MODEL)
        # Maps hotkey -> deque of past CLAP embeddings (up to _HISTORY_MAXLEN)
        self._miner_history: dict[str, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=_HISTORY_MAXLEN)
        )

    def score_batch(
        self,
        responses: list[MusicGenerationSynapse],
        hotkeys: list[str],
    ) -> list[float]:
        """
        Score intra-miner diversity for a batch of miner responses.

        Each miner's diversity score is the mean cosine *distance* between
        their current embedding and their own past embeddings stored in
        ``_miner_history``.  Miners with no history receive a neutral default
        score of 0.5.  High scores indicate varied output across rounds;
        low scores indicate repetitive/recycled submissions.

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

        scores: list[float] = []

        for resp, hotkey in zip(responses, hotkeys):
            emb = self._extract_embedding(resp)

            if emb is None:
                # Extraction failed — penalise with a zero score
                scores.append(0.0)
                continue

            past = self._miner_history[hotkey]

            if len(past) == 0:
                # No history yet — return neutral default
                scores.append(_DEFAULT_SCORE)
            else:
                # Compute mean cosine distance from own past embeddings
                score = self._mean_cosine_distance(emb, list(past))
                scores.append(score)

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

        Args:
            responses: Synapses with audio_b64 populated.
            hotkeys:   Miner hotkeys aligned with ``responses``.
        """
        if len(responses) != len(hotkeys):
            raise ValueError(
                f"responses length ({len(responses)}) must match "
                f"hotkeys length ({len(hotkeys)})"
            )

        for resp, hotkey in zip(responses, hotkeys):
            emb = self._extract_embedding(resp)
            if emb is not None:
                self._miner_history[hotkey].append(emb)
                logger.debug(
                    f"Updated history for {hotkey}: "
                    f"{len(self._miner_history[hotkey])} embedding(s) stored"
                )

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

        Args:
            current: The embedding for the current round.
            past:    List of past embeddings for the same miner.

        Returns:
            Mean cosine distance clipped to [0, 1].
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
