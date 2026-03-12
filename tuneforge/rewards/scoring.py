"""
Task scorer for TuneForge.

Provides batch-level scoring of validation round results,
mapping each miner UID to a final reward score.
"""

from loguru import logger

from tuneforge.base.protocol import MusicGenerationSynapse
from tuneforge.rewards.reward import ProductionRewardModel
from tuneforge.settings import Settings


class TaskScorer:
    """Batch processing of validation results for a round."""

    def __init__(self, config: Settings) -> None:
        self._reward_model = ProductionRewardModel(config)

    def score_round(
        self,
        challenge: dict,
        responses: dict[int, MusicGenerationSynapse],
        hotkeys: dict[int, str] | None = None,
    ) -> dict[int, float]:
        """
        Score all responses from a validation round.

        Args:
            challenge: Challenge parameters dict (prompt, genre, etc.).
            responses: Mapping of miner UID → response synapse.
            hotkeys: Optional mapping of miner UID → hotkey.

        Returns:
            Mapping of miner UID → reward score in [0, 1].
        """
        if not responses:
            logger.info("No responses to score")
            return {}

        uids = sorted(responses.keys())
        synapses = [responses[uid] for uid in uids]
        miner_hotkeys = [
            hotkeys.get(uid, f"unknown-{uid}") if hotkeys else f"unknown-{uid}"
            for uid in uids
        ]

        logger.info(f"Scoring {len(synapses)} responses for challenge {challenge.get('challenge_id', '?')}")

        try:
            rewards = self._reward_model.score_batch(synapses, miner_hotkeys)
        except Exception as exc:
            logger.error(f"Batch scoring failed: {exc}")
            rewards = [0.0] * len(synapses)

        result: dict[int, float] = {}
        for uid, reward in zip(uids, rewards):
            result[uid] = reward
            logger.debug(f"  UID {uid}: reward={reward:.4f}")

        logger.info(
            f"Round scoring complete: "
            f"mean={sum(result.values()) / max(len(result), 1):.4f}, "
            f"max={max(result.values()) if result else 0:.4f}, "
            f"scored={len(result)}"
        )
        return result
