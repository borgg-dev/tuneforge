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
    """Submit leaderboard weights to chain.

    Supports coverage-gated weight setting for permutation mode:
    weights are set at the end of the round that completes full miner
    coverage (all miners scored at least once), subject to a minimum
    block interval (don't set too early) and a maximum block interval
    (don't wait forever if coverage is never reached).
    """

    # Block interval bounds
    MIN_BLOCK_INTERVAL: int = 115   # ~23 min — never set sooner
    MAX_BLOCK_INTERVAL: int = 690   # ~138 min — force set even without coverage

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

    def should_update(self, coverage_complete: bool = False) -> bool:
        """Check whether weights should be set now.

        Args:
            coverage_complete: True when the validator has scored every
                miner at least once since the last weight set.

        Logic:
            - Always wait at least MIN_BLOCK_INTERVAL blocks.
            - If coverage is complete AND min blocks elapsed → set now.
            - If MAX_BLOCK_INTERVAL blocks elapsed → force set regardless.
        """
        try:
            current_block = self._subtensor.get_current_block()
            blocks_elapsed = current_block - self._last_update_block
        except Exception as exc:
            logger.error(f"Block check failed: {exc}")
            return False

        if blocks_elapsed < self.MIN_BLOCK_INTERVAL:
            return False

        if coverage_complete:
            logger.info(
                f"⚖️ Full miner coverage reached after {blocks_elapsed} blocks — setting weights"
            )
            return True

        if blocks_elapsed >= self.MAX_BLOCK_INTERVAL:
            logger.info(
                f"⚖️ Max interval ({self.MAX_BLOCK_INTERVAL} blocks) reached — "
                f"forcing weight set without full coverage"
            )
            return True

        return False

    def update_metagraph(self, metagraph: bt.Metagraph) -> None:
        """Update the metagraph reference."""
        self._metagraph = metagraph

    def set_weights(
        self, leaderboard: MinerLeaderboard, coverage_complete: bool = False
    ) -> bool:
        """
        Gather weights from leaderboard, normalise, and submit to chain.

        Args:
            leaderboard: The miner leaderboard with current scores.
            coverage_complete: Whether full miner coverage has been reached.

        Returns:
            True if weights were successfully set on chain.
        """
        if not self.should_update(coverage_complete=coverage_complete):
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
