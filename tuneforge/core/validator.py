"""
TuneForge validator neuron.

Extends BaseValidatorNeuron with the full scoring pipeline:
challenge generation → miner querying → multi-signal scoring →
EMA leaderboard → on-chain weight setting.
"""

import asyncio
import time
from typing import Any

import numpy as np
from loguru import logger

from tuneforge.base.dendrite import DendriteResponseEvent
from tuneforge.base.protocol import MusicGenerationSynapse
from tuneforge.base.validator import BaseValidatorNeuron
from tuneforge.rewards.leaderboard import MinerLeaderboard
from tuneforge.rewards.reward import ProductionRewardModel
from tuneforge.rewards.scoring import TaskScorer
from tuneforge.rewards.weight_setter import WeightSetter
from tuneforge.scoring.diversity import DiversityScorer
from tuneforge.settings import Settings, get_settings
from tuneforge.validation.challenge_manager import ChallengeManager
from tuneforge.validation.prompt_generator import PromptGenerator


class TuneForgeValidator(BaseValidatorNeuron):
    """Full-featured TuneForge subnet validator."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        super().__init__(settings=settings)

        # Scoring pipeline
        self._reward_model = ProductionRewardModel(self.settings)
        self._task_scorer = TaskScorer(self.settings)
        self._prompt_generator = PromptGenerator()
        self._leaderboard = MinerLeaderboard()
        self._diversity_scorer = DiversityScorer()
        self._challenge_manager = ChallengeManager()
        self._weight_setter: WeightSetter | None = None  # init after setup

        logger.info("TuneForgeValidator initialised")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Set up validator with weight setter."""
        super().setup()
        self._weight_setter = WeightSetter(
            subtensor=self.subtensor,
            wallet=self.wallet,
            netuid=self.settings.netuid,
            metagraph=self.metagraph,
            update_interval=self.settings.weight_setter_step,
        )
        logger.info("TuneForgeValidator setup complete")

    # ------------------------------------------------------------------
    # Validation round
    # ------------------------------------------------------------------

    async def run_validation_round(self) -> DendriteResponseEvent:
        """Execute one complete validation round."""
        self.round_start_time = time.time()
        self.current_round += 1

        logger.info(f"=== Validation round {self.current_round} ===")

        # 1. Generate challenge prompt
        challenge = self._prompt_generator.generate_challenge()

        # 2. Create synapse
        synapse = MusicGenerationSynapse(
            prompt=challenge["prompt"],
            genre=challenge["genre"],
            mood=challenge["mood"],
            tempo_bpm=challenge["tempo_bpm"],
            duration_seconds=challenge["duration_seconds"],
            key_signature=challenge.get("key_signature"),
            time_signature=challenge.get("time_signature"),
            instruments=challenge.get("instruments"),
            seed=challenge.get("seed"),
            challenge_id=challenge["challenge_id"],
        )

        # 3. Select miner UIDs
        miner_uids = self.get_miner_subset()
        if not miner_uids:
            logger.warning("No miners in subset — empty round")
            return DendriteResponseEvent(
                round_id=challenge["challenge_id"],
                block_number=self.block,
                validator_uid=self.uid or -1,
            )

        logger.info(f"Querying {len(miner_uids)} miners")

        # 4. Query miners via dendrite
        axons = [self.metagraph.axons[uid] for uid in miner_uids]
        try:
            responses: list[MusicGenerationSynapse] = await self.dendrite.forward(
                axons=axons,
                synapse=synapse,
                timeout=self.settings.generation_timeout,
            )
        except Exception as exc:
            logger.error(f"Dendrite forward failed: {exc}")
            responses = []

        # Build response event
        event = DendriteResponseEvent(
            round_id=challenge["challenge_id"],
            block_number=self.block,
            validator_uid=self.uid or -1,
            round_start_time=self.round_start_time,
        )

        valid_responses: list[MusicGenerationSynapse] = []
        valid_uids: list[int] = []
        valid_hotkeys: list[str] = []

        for uid, resp in zip(miner_uids, responses):
            if resp is None:
                continue
            has_audio = resp.audio_b64 is not None
            event.add_generation_response(
                uid=uid,
                response={"audio_b64": resp.audio_b64, "model_id": resp.model_id},
                success=has_audio,
                generation_time_ms=resp.generation_time_ms,
            )
            if has_audio:
                valid_responses.append(resp)
                valid_uids.append(uid)
                try:
                    valid_hotkeys.append(self.metagraph.hotkeys[uid])
                except (IndexError, AttributeError):
                    valid_hotkeys.append(f"uid-{uid}")

        logger.info(
            f"Got {len(valid_responses)}/{len(miner_uids)} valid responses"
        )

        # 5. Score all responses
        if valid_responses:
            try:
                rewards = self._reward_model.score_batch(valid_responses, valid_hotkeys)
            except Exception as exc:
                logger.error(f"Scoring failed: {exc}")
                rewards = [0.0] * len(valid_responses)

            # 6. Update leaderboard
            for uid, reward in zip(valid_uids, rewards):
                self._leaderboard.update(uid, reward)

            # 7. Compute diversity scores (already factored into score_batch,
            #    but log them separately)
            try:
                diversity_scores = self._diversity_scorer.score_batch(valid_responses)
                for uid, div_s in zip(valid_uids, diversity_scores):
                    logger.debug(f"  UID {uid}: diversity={div_s:.3f}")
            except Exception as exc:
                logger.debug(f"Diversity scoring info: {exc}")

            # Update base scores array
            self.update_scores(valid_uids, rewards)

            # Log round results
            for uid, reward in zip(valid_uids, rewards):
                ema = self._leaderboard.get_ema(uid)
                weight = self._leaderboard.get_weight(uid)
                logger.info(
                    f"  UID {uid}: reward={reward:.4f}, "
                    f"ema={ema:.4f}, weight={weight:.4f}"
                )

        # 8. Clear round caches
        # (plagiarism detector caches are cleared in score_batch)

        # 9. Set weights via weight setter
        if self._weight_setter is not None:
            try:
                self._weight_setter.update_metagraph(self.metagraph)
                self._weight_setter.set_weights(self._leaderboard)
            except Exception as exc:
                logger.error(f"Weight setting failed: {exc}")

        # 10. Log summary
        event.round_end_time = time.time()
        lb_summary = self._leaderboard.summary()
        logger.info(
            f"Round {self.current_round} complete: "
            f"{len(valid_responses)} scored, "
            f"leaderboard: {lb_summary.get('warmed_up', 0)} warmed up, "
            f"duration={event.round_end_time - event.round_start_time:.1f}s"
        )

        return event

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def process_round_results(self, response_event: DendriteResponseEvent) -> None:
        """Process round results — scoring is done in run_validation_round."""
        summary = response_event.summary()
        logger.info(f"Round summary: {summary}")

    async def forward(self) -> DendriteResponseEvent:
        """Execute one validation round (async entry point)."""
        return await self.run_validation_round()
