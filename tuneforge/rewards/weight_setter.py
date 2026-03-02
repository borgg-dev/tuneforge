"""
Weight setter for TuneForge.

Periodically submits normalised miner weights to the Bittensor chain
based on leaderboard scores.
"""

import numpy as np
import bittensor as bt
from loguru import logger

from tuneforge.config.scoring_config import WEIGHT_UPDATE_INTERVAL
from tuneforge.rewards.leaderboard import MinerLeaderboard


class WeightSetter:
    """Submit leaderboard weights to chain."""

    def __init__(
        self,
        subtensor: bt.Subtensor,
        wallet: bt.Wallet,
        netuid: int,
        metagraph: bt.Metagraph,
        update_interval: int = WEIGHT_UPDATE_INTERVAL,
    ) -> None:
        self._subtensor = subtensor
        self._wallet = wallet
        self._netuid = netuid
        self._metagraph = metagraph
        self._update_interval = update_interval
        self._last_update_block: int = 0

    def should_update(self) -> bool:
        """Check whether enough blocks have elapsed since last weight set."""
        try:
            current_block = self._subtensor.get_current_block()
            return (current_block - self._last_update_block) >= self._update_interval
        except Exception as exc:
            logger.error(f"Block check failed: {exc}")
            return False

    def update_metagraph(self, metagraph: bt.Metagraph) -> None:
        """Update the metagraph reference."""
        self._metagraph = metagraph

    def set_weights(self, leaderboard: MinerLeaderboard) -> bool:
        """
        Gather weights from leaderboard, normalise, and submit to chain.

        Args:
            leaderboard: The miner leaderboard with current scores.

        Returns:
            True if weights were successfully set on chain.
        """
        if not self.should_update():
            logger.debug("Weight update not due yet")
            return False

        try:
            all_weights = leaderboard.get_all_weights()
            n_neurons = len(self._metagraph.S)

            raw_weights = np.zeros(n_neurons, dtype=np.float32)
            for uid, weight in all_weights.items():
                if 0 <= uid < n_neurons:
                    raw_weights[uid] = weight

            total = raw_weights.sum()
            if total <= 0:
                logger.warning("All weights zero — skipping weight set")
                return False

            raw_weights = raw_weights / total

            uids = np.arange(n_neurons)

            processed_uids, processed_weights = (
                bt.utils.weight_utils.process_weights_for_netuid(
                    uids=uids,
                    weights=raw_weights,
                    netuid=self._netuid,
                    subtensor=self._subtensor,
                    metagraph=self._metagraph,
                )
            )

            uint_uids, uint_weights = (
                bt.utils.weight_utils.convert_weights_and_uids_for_emit(
                    uids=processed_uids,
                    weights=processed_weights,
                )
            )

            result = self._subtensor.set_weights(
                wallet=self._wallet,
                netuid=self._netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_finalization=True,
                wait_for_inclusion=True,
            )

            if result[0]:
                self._last_update_block = self._subtensor.get_current_block()
                logger.info(
                    f"Weights set on chain at block {self._last_update_block} "
                    f"({int(np.count_nonzero(raw_weights))} non-zero)"
                )
                return True
            else:
                logger.error(f"Weight set failed: {result}")
                return False

        except Exception as exc:
            logger.error(f"Weight setting error: {exc}")
            return False
