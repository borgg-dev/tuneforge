"""
Base validator neuron for TuneForge subnet.

Extends BaseNeuron with validator-specific functionality:
- Miner querying and availability checking
- Validation round orchestration
- Scoring and weight setting
- Commit-sync permutation for multi-validator coordination
"""

import asyncio
import hashlib
import time
from abc import abstractmethod
from typing import Any, Optional

import bittensor as bt
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from tuneforge.base.dendrite import DendriteResponseEvent
from tuneforge.base.neuron import BaseNeuron
from tuneforge.base.protocol import PingSynapse
from tuneforge.settings import Settings, get_settings
from tuneforge.validation.commit_sync import CommitSync
from tuneforge import DEFAULT_ROUND_INTERVAL, DEFAULT_WEIGHT_UPDATE_INTERVAL, MAX_ROUNDS_PER_EPOCH, EPOCH_SYNC


class BaseValidatorNeuron(BaseModel, BaseNeuron):
    """
    Base class for TuneForge validator neurons.

    Validators orchestrate validation rounds, query miners for
    music generation, score their outputs, and set weights on-chain.
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
    neuron_type: str = "Validator"

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

    # Commit-sync for multi-validator permutation (initialised in setup)
    _commit_sync: Optional[CommitSync] = None
    # Miner subset assigned by commit-sync for the current round (None = all)
    _current_round_subset: Optional[list[int]] = None
    # Track last completed round to avoid re-running on slow iterations
    _last_completed_round: int = -1
    _last_completed_epoch: int = -1

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
        """Set up the validator (registration, dendrite, weight arrays, commit sync)."""
        logger.info("Setting up validator…")

        if not self.check_registered():
            raise RuntimeError("Validator not registered on subnet")

        self.dendrite = self.settings.dendrite
        n_uids = len(self.metagraph.S)
        self.scores = np.zeros(n_uids, dtype=np.float32)
        self.weights = np.zeros(n_uids, dtype=np.float32)

        # Initialise commit-sync for multi-validator permutation
        self._commit_sync = CommitSync(
            subtensor=self.subtensor,
            wallet=self.wallet,
            netuid=self.settings.netuid,
            metagraph=self.metagraph,
        )

        logger.info(f"Validator setup complete. UID: {self.uid}")

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
        """Get the miners to query this round.

        If commit-sync assigned a permutation subset, return that.
        Otherwise fall back to all miner UIDs (stagger-only mode).
        """
        if self._current_round_subset is not None:
            return self._current_round_subset
        return self.get_miner_uids()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def update_scores(self, uids: list[int], rewards: list[float]) -> None:
        """
        Store raw reward scores for weight setting.

        EMA smoothing is handled by MinerLeaderboard — this array
        just holds the latest raw reward for each UID so the base
        weight-setting logic has something to work with.

        Args:
            uids: List of miner UIDs.
            rewards: Corresponding reward values (0.0 - 1.0).
        """
        for uid, reward in zip(uids, rewards):
            if 0 <= uid < len(self.scores):
                self.scores[uid] = reward
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
        """Main run loop for the validator.

        Runs the validation loop and (optionally) the organic generation
        HTTP server concurrently on the same asyncio event loop.
        """
        logger.info("Starting validator neuron…")
        self.setup()
        self.is_running = True

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._run_async())
        finally:
            loop.close()
            self.shutdown()

    async def _run_async(self) -> None:
        """Run validation loop and organic API server concurrently."""
        tasks: list[asyncio.Task] = []

        # Start organic API server if enabled
        organic_server = await self._start_organic_api()
        if organic_server is not None:
            tasks.append(asyncio.create_task(organic_server.serve()))

        # Run the validation loop
        tasks.append(asyncio.create_task(self._validation_loop()))

        # Start organic score sync (pulls peer scores from platform API)
        if hasattr(self, "run_organic_score_sync"):
            tasks.append(asyncio.create_task(self.run_organic_score_sync()))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _start_organic_api(self):
        """Start the organic generation HTTP server if enabled.

        Returns a uvicorn.Server instance or None.
        """
        if not getattr(self.settings, "organic_api_enabled", False):
            return None

        try:
            import uvicorn
            from tuneforge.api.validator_api import create_validator_api

            api_app = create_validator_api(self)
            port = getattr(self.settings, "organic_api_port", 8090)
            config = uvicorn.Config(
                app=api_app,
                host="0.0.0.0",
                port=port,
                log_level="info",
                access_log=False,
            )
            server = uvicorn.Server(config)
            logger.info("Organic API server will start on port {}", port)
            return server
        except Exception as exc:
            logger.error("Failed to create organic API server: {}", exc)
            return None

    def _compute_stagger_offset(self) -> float:
        """Compute deterministic challenge stagger offset for this validator.

        Spreads validators evenly across the validation interval based on
        metagraph position, so challenge rounds from different validators
        never overlap and cause miner queue congestion / timeouts.
        """
        try:
            validator_uids = sorted(self.get_validator_uids())
            if not validator_uids or self.uid not in validator_uids:
                return 0.0

            position = validator_uids.index(self.uid)
            n_validators = len(validator_uids)
            base_interval = self.settings.round_interval
            offset = (position / n_validators) * base_interval

            logger.info(
                f"Challenge stagger: validator {self.uid} at position "
                f"{position}/{n_validators}, offset {offset:.0f}s "
                f"within {base_interval}s cycle"
            )
            return offset
        except Exception as exc:
            logger.warning(f"Failed to compute stagger offset, using 0: {exc}")
            return 0.0

    def _compute_cycle_jitter(self, cycle_number: int) -> float:
        """Compute small deterministic per-cycle jitter for anti-gaming.

        Uses validator hotkey + cycle number to produce a consistent ±5s
        jitter.  Small enough to never cause overlap between stagger slots
        (minimum slot gap is interval/N, typically 60-200s) but enough to
        prevent miners from predicting exact challenge timing.
        """
        try:
            hotkey = self.wallet.hotkey.ss58_address
            h = hashlib.sha256(f"{hotkey}:{cycle_number}".encode())
            # Map hash to [-0.5, +0.5) then scale to ±5 seconds
            frac = (int.from_bytes(h.digest()[:4], "big") / 0xFFFFFFFF) - 0.5
            return frac * 10.0  # ±5s
        except Exception:
            return 0.0

    async def _validation_loop(self) -> None:
        """Async validation loop with commit-sync permutation.

        Epoch timeline (all derived from the clock):
        |-- epsilon --|-- 64 × round_interval --|-- epsilon --|
          commit-        rounds 0..63              sync
          reveal         (cycle N subsets)          buffer

        All validators agree on epoch boundaries via
        ``time % epoch_interval == 0``.  Within an epoch, elapsed
        time determines the current phase:
        - elapsed < EPOCH_SYNC: commit-reveal phase
        - EPOCH_SYNC <= elapsed < EPOCH_SYNC + 64*round_interval: round phase
        - remainder: cooldown buffer (idle)

        Round boundaries fire when
        ``(elapsed - EPOCH_SYNC) % round_interval == 0``.
        """
        round_interval = self.settings.round_interval
        epoch_interval = self.settings.epoch_interval
        sync_duration = EPOCH_SYNC

        from datetime import datetime, timezone
        remaining = self._seconds_until_next_epoch(epoch_interval)
        logger.info(
            f"Validator is up and running, next epoch starting in ~{remaining}s..."
        )

        loop = asyncio.get_event_loop()
        while self.is_running and not self.should_exit:
            try:
                current_time = int(time.time())
                epoch_start = current_time - (current_time % epoch_interval)
                elapsed = current_time - epoch_start

                # --- Phase 1: Commit-reveal (first sync_duration seconds) ---
                if elapsed < sync_duration:
                    # Only commit once per epoch (at the very start)
                    if (
                        self._commit_sync is not None
                        and self._commit_sync.last_sync_time != epoch_start
                    ):
                        now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                        try:
                            self._commit_sync.update_metagraph(self.metagraph)
                            logger.info(
                                f"📢 Starting new epoch at {now_utc} UTC."
                            )
                            logger.info(
                                f"🐛 Committing Proof of Activity : {epoch_start}"
                            )
                            synced = await self._commit_sync.sync_epoch(
                                sync_time=epoch_start,
                                my_uid=self.uid,
                                all_miner_uids=self.get_miner_uids(),
                                loop=loop,
                            )
                            if not synced:
                                logger.warning(
                                    "❌ Commit-sync failed — skipping epoch"
                                )
                        except Exception as exc:
                            logger.error("❌ Commit-sync failed: {}", exc)

                    await asyncio.sleep(2)
                    continue

                # --- Phase 2: Rounds (after sync, every round_interval) ---
                round_phase_elapsed = elapsed - sync_duration
                if round_phase_elapsed >= 0:
                    round_index = round_phase_elapsed // round_interval
                    if (
                        round_index < MAX_ROUNDS_PER_EPOCH
                        and (round_index != self._last_completed_round
                             or epoch_start != self._last_completed_epoch)
                    ):
                        now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                        self._current_round_subset = None
                        all_miners = self.get_miner_uids()

                        if self._commit_sync is not None:
                            active = self._commit_sync.active_count
                            if active == 0:
                                await asyncio.sleep(2)
                                continue

                            try:
                                cycle_index = round_index % active
                                logger.info(
                                    f"📢 Round {round_index + 1}/{MAX_ROUNDS_PER_EPOCH} "
                                    f"(subset {cycle_index + 1}/{active}) "
                                    f"at {now_utc} UTC."
                                )
                                subset = self._commit_sync.get_miner_subset(
                                    my_uid=self.uid,
                                    all_miner_uids=all_miners,
                                    round_index=cycle_index,
                                )
                                if subset is not None:
                                    self._current_round_subset = subset
                                else:
                                    logger.warning(
                                        "❌ get_miner_subset returned None — skipping round"
                                    )
                                    await asyncio.sleep(2)
                                    continue
                            except Exception as exc:
                                logger.error("❌ Subset selection failed: {}", exc)
                                await asyncio.sleep(2)
                                continue

                        # --- Run the validation round ---
                        response_event = await self.run_validation_round()
                        self.process_round_results(response_event)
                        self._current_round_subset = None

                        # Mark round as completed so we don't re-run it
                        self._last_completed_round = round_index
                        self._last_completed_epoch = epoch_start

                        remaining_rounds = MAX_ROUNDS_PER_EPOCH - round_index - 1
                        if remaining_rounds > 0:
                            next_round = self._seconds_until_next_epoch(round_interval)
                            logger.info(
                                f"🎉 Next round in ~{next_round}s "
                                f"({remaining_rounds} rounds left in epoch)..."
                            )
                        else:
                            cooldown_end = epoch_start + epoch_interval
                            remaining_epoch = cooldown_end - int(time.time())
                            logger.info(
                                f"🏁 All {MAX_ROUNDS_PER_EPOCH} rounds complete. "
                                f"Epoch cooldown ~{max(0, remaining_epoch)}s..."
                            )

                        await loop.run_in_executor(None, self.log_status)
                        self.step += 1

                        await asyncio.sleep(2)
                        continue

                # --- Phase 3: Sync buffer / idle ---
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.error(f"Error in validation loop: {exc}")
                await asyncio.sleep(60)

    @staticmethod
    def _seconds_until_next_epoch(epoch_length: int) -> int:
        """Seconds until the next epoch boundary."""
        now = int(time.time())
        return epoch_length - (now % epoch_length)

    async def _async_wait(self, seconds: float) -> None:
        """Wait in 1-second increments, checking should_exit."""
        end_time = time.time() + seconds
        while time.time() < end_time and not self.should_exit:
            await asyncio.sleep(1)

    def process_round_results(self, response_event: DendriteResponseEvent) -> None:
        """
        Process results from a validation round.

        Override in subclasses to implement specific scoring logic.
        """
        logger.debug(f"Processing round results: {response_event.summary()}")

    def shutdown(self) -> None:
        """Shut down the validator cleanly."""
        logger.info("Shutting down validator…")
        self.is_running = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
