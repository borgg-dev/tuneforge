"""
Base validator neuron for TuneForge subnet.

Extends BaseNeuron with validator-specific functionality:
- Miner querying and availability checking
- Validation round orchestration
- Scoring and weight setting
- Challenge generation with time-synchronised miner subsets
"""

import asyncio
import random
import time
from abc import abstractmethod
from typing import Any

import bittensor as bt
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from tuneforge.base.dendrite import DendriteResponseEvent
from tuneforge.base.neuron import BaseNeuron
from tuneforge.base.protocol import PingSynapse
from tuneforge.settings import Settings, get_settings
from tuneforge import DEFAULT_VALIDATION_INTERVAL, DEFAULT_WEIGHT_UPDATE_INTERVAL


class BaseValidatorNeuron(BaseModel, BaseNeuron):
    """
    Base class for TuneForge validator neurons.

    Validators orchestrate validation rounds, query miners for
    music generation, score their outputs, and set weights on-chain.
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    # Settings (declared for Pydantic; lazy-loaded in BaseNeuron)
    settings: Settings | None = Field(default=None)
    step: int = Field(default=0)
    last_sync_block: int = Field(default=0)

    # Bittensor primitives (private lazy caches)
    _wallet: bt.Wallet | None = None
    _subtensor: bt.Subtensor | None = None
    _metagraph: bt.Metagraph | None = None
    _uid: int | None = None

    # Dendrite for outbound queries
    dendrite: bt.Dendrite | None = Field(default=None)

    # Weight state
    scores: Any = Field(default=None)  # np.ndarray
    weights: Any = Field(default=None)  # np.ndarray
    past_weights: list = Field(default_factory=list)
    _last_weight_set_block: int = 0

    # Round state
    current_round: int = Field(default=0)
    round_start_time: float = Field(default=0.0)
    is_running: bool = Field(default=False)
    should_exit: bool = Field(default=False)
    active_miner_uids: list[int] = Field(default_factory=list)

    def __init__(self, settings: Settings | None = None, **kwargs):
        """Initialise the validator neuron."""
        if settings is None:
            settings = get_settings()
        BaseModel.__init__(self, settings=settings, **kwargs)
        BaseNeuron.__init__(self, settings)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Set up the validator (registration, dendrite, axon, weight arrays)."""
        logger.info("Setting up validator…")

        if not self.check_registered():
            raise RuntimeError("Validator not registered on subnet")

        self.dendrite = self.settings.dendrite
        n_uids = len(self.metagraph.S)
        self.scores = np.zeros(n_uids, dtype=np.float32)
        self.weights = np.zeros(n_uids, dtype=np.float32)

        # Serve axon so the validator's IP is discoverable on the metagraph.
        # The axon itself doesn't handle synapses — it just registers the IP
        # so the platform API can locate the validator's HTTP API.
        if not self.settings.neuron_axon_off:
            self._serve_axon()

        logger.info(f"Validator setup complete. UID: {self.uid}")

    def _serve_axon(self) -> None:
        """Start and register axon on-chain for IP discoverability."""
        if self.settings.axon_port:
            self._axon = bt.Axon(wallet=self.wallet, port=self.settings.axon_port)
        else:
            self._axon = bt.Axon(wallet=self.wallet)
        self._axon.start()
        logger.info(f"Validator axon serving on port {self._axon.port}")

        # Register on-chain with retry
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.subtensor.serve_axon(
                    netuid=self.settings.netuid,
                    axon=self._axon,
                )
                logger.info("Validator axon registered on-chain")
                return
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt * 10
                    logger.warning(f"Axon registration attempt {attempt + 1} failed: {exc}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"Failed to register validator axon after {max_retries} attempts: {exc}")

    # ------------------------------------------------------------------
    # Weight setting
    # ------------------------------------------------------------------

    def should_set_weights(self) -> bool:
        """Check if weights should be set based on blocks elapsed."""
        if self.step == 0:
            return False

        try:
            if not self.metagraph.validator_permit[self.uid]:
                return False
        except Exception:
            return False

        metagraph_last_update = self.metagraph.last_update[self.uid]
        last_update_block = max(metagraph_last_update, self._last_weight_set_block)
        blocks_since_update = self.block - last_update_block
        return blocks_since_update > self.settings.weight_setter_step

    def set_weights(self) -> bool:
        """
        Set weights on-chain.

        Normalises scores, averages with past weights for stability,
        processes through Bittensor weight utilities, and submits.
        """
        logger.info("Setting weights on chain…")

        raw_weights = self.scores.copy()
        raw_weights = np.nan_to_num(raw_weights, nan=0.0)

        # Average with past weights for stability
        self.past_weights.append(raw_weights)
        max_history = getattr(self.settings, "past_weights_count", 5)
        if len(self.past_weights) > max_history:
            self.past_weights.pop(0)

        if self.past_weights:
            raw_weights = np.mean(self.past_weights, axis=0)

        # Normalise
        total = raw_weights.sum()
        if total > 0:
            raw_weights = raw_weights / total

        try:
            processed_uids, processed_weights = (
                bt.utils.weight_utils.process_weights_for_netuid(
                    uids=np.arange(len(raw_weights)),
                    weights=raw_weights,
                    netuid=self.settings.netuid,
                    subtensor=self.subtensor,
                    metagraph=self.metagraph,
                )
            )

            uint_uids, uint_weights = (
                bt.utils.weight_utils.convert_weights_and_uids_for_emit(
                    uids=processed_uids,
                    weights=processed_weights,
                )
            )

            result = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.settings.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_finalization=True,
                wait_for_inclusion=True,
            )

            if result[0]:
                logger.info("Successfully set weights on chain")
                self.weights = raw_weights
                self._last_weight_set_block = self.block
                return True
            else:
                logger.error(f"Failed to set weights on chain: {result}")
                return False

        except Exception as exc:
            logger.error(f"Error setting weights: {exc}")
            return False

    # ------------------------------------------------------------------
    # Miner querying
    # ------------------------------------------------------------------

    async def query_miners_availability(
        self,
        uids: list[int],
        timeout: float = 60.0,
    ) -> dict[int, PingSynapse]:
        """
        Query miners for availability via PingSynapse.

        Returns:
            Mapping of UID → PingSynapse for available miners.
        """
        logger.info(f"Querying {len(uids)} miners for availability…")

        axons = [self.metagraph.axons[uid] for uid in uids]
        synapse = PingSynapse()

        try:
            responses = await self.dendrite.forward(
                axons=axons,
                synapse=synapse,
                timeout=timeout,
            )
        except Exception as exc:
            logger.error(f"Dendrite forward failed: {exc}")
            return {}

        result: dict[int, PingSynapse] = {}
        for uid, response in zip(uids, responses):
            if response is None:
                continue
            if hasattr(response, "is_available") and response.is_available:
                result[uid] = response
                logger.trace(f"Miner {uid} available")

        logger.info(f"Got {len(result)} available miners")
        return result

    def get_miner_subset(self) -> list[int]:
        """
        Get the miners to query this round.

        Returns all available miner UIDs (non-validators) up to the subnet size.
        """
        return self.get_miner_uids()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def update_scores(self, uids: list[int], rewards: list[float]) -> None:
        """
        Update miner scores with exponential moving average.

        Args:
            uids: List of miner UIDs.
            rewards: Corresponding reward values (0.0 – 1.0).
        """
        alpha = 0.1
        for uid, reward in zip(uids, rewards):
            if 0 <= uid < len(self.scores):
                self.scores[uid] = alpha * reward + (1 - alpha) * self.scores[uid]
        logger.debug(f"Updated scores for {len(uids)} miners")

    # ------------------------------------------------------------------
    # Unused BaseNeuron stubs
    # ------------------------------------------------------------------

    def forward(self, synapse: PingSynapse) -> PingSynapse:
        """Validators don't respond to pings — required by BaseNeuron."""
        return synapse

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def run_validation_round(self) -> DendriteResponseEvent:
        """Execute a complete validation round and return collected responses."""
        ...

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main run loop for the validator."""
        logger.info("Starting validator neuron…")
        self.setup()
        self.is_running = True

        # Single event loop for the lifetime of the validator — reused across rounds
        # so the dendrite's aiohttp session stays valid.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while self.is_running and not self.should_exit:
                try:
                    if self.should_sync_metagraph():
                        self.sync()

                    # Run validation round
                    response_event = loop.run_until_complete(
                        self.run_validation_round()
                    )

                    # Process results
                    self.process_round_results(response_event)

                    # Weight setting is handled by WeightSetter in
                    # TuneForgeValidator.run_validation_round() — the
                    # duplicate BaseValidatorNeuron.set_weights() call
                    # was removed because it overwrote the leaderboard's
                    # steepened weights with flat EMA weights (NEW-01 fix).

                    self.log_status()
                    self.step += 1

                    # Wait before next round
                    self._wait_for_next_round()

                except KeyboardInterrupt:
                    break
                except Exception as exc:
                    logger.error(f"Error in validation loop: {exc}")
                    time.sleep(60)
        finally:
            loop.close()
            self.shutdown()

    def process_round_results(self, response_event: DendriteResponseEvent) -> None:
        """
        Process results from a validation round.

        Override in subclasses to implement specific scoring logic.
        """
        logger.debug(f"Processing round results: {response_event.summary()}")

    def _wait_for_next_round(self) -> None:
        """Wait the appropriate interval before the next round."""
        elapsed = time.time() - self.round_start_time
        remaining = self.settings.validation_interval - elapsed
        if remaining > 0:
            logger.info(f"Waiting {remaining:.0f}s until next round…")
            # Sleep in short intervals so Ctrl+C / SIGTERM is responsive
            end_time = time.time() + remaining
            while time.time() < end_time and not self.should_exit:
                time.sleep(1)

    def shutdown(self) -> None:
        """Shut down the validator cleanly."""
        logger.info("Shutting down validator…")
        self.is_running = False
        if hasattr(self, "_axon") and self._axon:
            try:
                self._axon.stop()
            except Exception as exc:
                logger.error(f"Error stopping validator axon: {exc}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
