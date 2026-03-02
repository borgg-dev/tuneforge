"""
Diversity scorer for TuneForge.

Measures inter-miner diversity using CLAP audio embeddings.
Miners who produce unique outputs score higher than those whose
outputs are near-identical to other miners in the same round.
"""

import numpy as np
from loguru import logger

from tuneforge.config.scoring_config import CLAP_MODEL
from tuneforge.scoring.clap_scorer import CLAPScorer
from tuneforge.base.protocol import MusicGenerationSynapse


class DiversityScorer:
    """Score inter-miner output diversity via CLAP embeddings."""

    def __init__(self) -> None:
        self._clap = CLAPScorer(model_name=CLAP_MODEL)

    def score_batch(self, responses: list[MusicGenerationSynapse]) -> list[float]:
        """
        Score diversity for a batch of miner responses.

        Each miner's diversity score is the mean cosine *distance* from
        all other miners' outputs.  Higher = more unique.

        Args:
            responses: Synapses with audio_b64 populated.

        Returns:
            Per-response diversity scores in [0, 1], aligned with input list.
        """
        if len(responses) <= 1:
            return [1.0] * len(responses)

        embeddings: list[np.ndarray | None] = []
        for resp in responses:
            emb = self._extract_embedding(resp)
            embeddings.append(emb)

        # Collect valid embeddings and their indices
        valid_indices: list[int] = []
        valid_embeds: list[np.ndarray] = []
        for idx, emb in enumerate(embeddings):
            if emb is not None:
                valid_indices.append(idx)
                valid_embeds.append(emb)

        scores = [0.0] * len(responses)

        if len(valid_embeds) <= 1:
            # All or all-but-one failed — give default score
            for i in valid_indices:
                scores[i] = 1.0
            return scores

        # Build normalised embedding matrix
        mat = np.stack(valid_embeds)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        mat_norm = mat / norms

        # Pairwise cosine similarity
        sim_matrix = mat_norm @ mat_norm.T  # shape (N, N)

        for i_pos, i_orig in enumerate(valid_indices):
            # Mean distance to all other valid embeddings
            sims = sim_matrix[i_pos]
            # Exclude self-similarity
            other_sims = np.concatenate([sims[:i_pos], sims[i_pos + 1 :]])
            if len(other_sims) == 0:
                scores[i_orig] = 1.0
                continue
            mean_sim = float(np.mean(other_sims))
            # Convert similarity → distance → [0, 1] score
            diversity = float(np.clip(1.0 - mean_sim, 0.0, 1.0))
            scores[i_orig] = diversity

        return scores

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
