"""
Commit-based multi-validator synchronization and permutation.

Validators commit a timestamp to the Bittensor chain, wait for finality,
then use the shared timestamp to:
1. Discover how many validators are active this round
2. Deterministically partition miners into mutually exclusive subsets
3. Each validator takes its assigned subset — no two validators query the same miner

"""

import hashlib
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import bittensor as bt
from loguru import logger


# How long to wait after committing for blockchain finality (seconds).
# 30s is sufficient for our round cadence.
COMMIT_FINALITY_WAIT: int = 30

# Maximum time allowed for the commit operation itself (seconds).
COMMIT_TIMEOUT: float = 55.0

# Number of commit retry attempts before giving up.
COMMIT_MAX_RETRIES: int = 3

# Minimum stake (TAO) for a validator to be considered active.
# Validators below this threshold are ignored during active discovery.
MIN_VALIDATOR_STAKE: float = 10_000.0


class CommitSync:
    """Handles commit-based validator synchronization and miner partitioning.

    Usage:
        sync = CommitSync(subtensor, wallet, netuid, metagraph)

        # At the start of each round:
        sync_time = sync.commit_timestamp()
        if sync_time is not None:
            subset = sync.get_miner_subset(my_uid, all_miner_uids)
    """

    def __init__(
        self,
        subtensor: bt.Subtensor,
        wallet: bt.Wallet,
        netuid: int,
        metagraph: bt.Metagraph,
        min_stake: float = MIN_VALIDATOR_STAKE,
    ):
        self._subtensor = subtensor
        self._wallet = wallet
        self._netuid = netuid
        self._metagraph = metagraph
        self._min_stake = min_stake

        # State from last successful sync
        self._last_sync_time: Optional[int] = None
        self._active_validator_uids: list[int] = []
        self._active_count: int = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="commit")

    def update_metagraph(self, metagraph: bt.Metagraph) -> None:
        """Update the metagraph reference (call after resync)."""
        self._metagraph = metagraph

    @property
    def active_count(self) -> int:
        """Number of active validators discovered in last sync."""
        return self._active_count

    @property
    def active_validator_uids(self) -> list[int]:
        """UIDs of active validators from last sync."""
        return list(self._active_validator_uids)

    @property
    def last_sync_time(self) -> Optional[int]:
        """Timestamp from last successful commit sync."""
        return self._last_sync_time

    # ------------------------------------------------------------------
    # Step 1: Commit timestamp to chain
    # ------------------------------------------------------------------

    def commit_timestamp(self, sync_time: int) -> bool:
        """Commit a given timestamp to chain as proof of activity.

        The caller is responsible for passing the correct epoch-boundary
        timestamp (i.e. a value where ``sync_time % epoch_length == 0``).
        All active validators hit the same boundary at the same second,
        so they all commit the same value.

        Returns True on success, False if all retries failed.
        """
        for attempt in range(1, COMMIT_MAX_RETRIES + 1):
            try:
                self._subtensor.set_commitment(
                    wallet=self._wallet,
                    netuid=self._netuid,
                    data=str(sync_time),
                )
                logger.info(
                    "✅ Commit succeeded on attempt {} (sync_time={})",
                    attempt, sync_time,
                )
                self._last_sync_time = sync_time
                return True
            except Exception as exc:
                logger.warning(
                    "⚠️ Commit attempt {}/{} failed: {}",
                    attempt, COMMIT_MAX_RETRIES, exc,
                )
                if attempt < COMMIT_MAX_RETRIES:
                    time.sleep(1)

        logger.error("❌ All commit attempts failed. Skipping round.")
        return False

    # ------------------------------------------------------------------
    # Step 2: Discover active validators
    # ------------------------------------------------------------------

    def fetch_active_validators(self, sync_time: int) -> list[int]:
        """Query chain commitments to discover which validators are active.

        Filters by:
        - Matching committed timestamp
        - Has validator_permit
        - Meets minimum stake threshold

        Returns sorted list of active validator UIDs.
        """
        try:
            all_commitments = self._subtensor.get_all_commitments(
                netuid=self._netuid
            )
        except Exception as exc:
            logger.error("Failed to fetch commitments: {}", exc)
            return []

        sync_str = str(sync_time)
        matching_hotkeys = {
            hotkey for hotkey, value in all_commitments.items()
            if value == sync_str
        }

        if not matching_hotkeys:
            logger.warning("No validators committed timestamp {}", sync_time)
            return []

        active_uids: list[int] = []
        try:
            for neuron in self._metagraph.neurons:
                if (
                    neuron.hotkey in matching_hotkeys
                    and neuron.validator_permit
                    and float(self._metagraph.S[neuron.uid]) >= self._min_stake
                ):
                    active_uids.append(neuron.uid)
        except Exception as exc:
            logger.error("Error filtering active validators: {}", exc)
            return []

        active_uids.sort()
        self._active_validator_uids = active_uids
        self._active_count = len(active_uids)

        logger.info(
            "🔍 Number of active validators = {} : {}",
            self._active_count, active_uids,
        )
        return active_uids

    # ------------------------------------------------------------------
    # Step 3: Deterministic miner partitioning
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_seed(sync_time: int) -> int:
        """Compute deterministic seed from sync timestamp.

        All validators with the same sync_time produce the same seed,
        so they all generate the same shuffled miner order.
        """
        h = hashlib.sha256(str(sync_time).encode("utf-8")).hexdigest()
        return int(h, 16) % (2**32)

    @staticmethod
    def _map_to_consecutive(uids: list[int]) -> dict[int, int]:
        """Map sorted UIDs to consecutive indices (0, 1, 2, ...).

        This ensures each active validator gets a unique, stable index
        for selecting its miner subset.
        """
        return {uid: idx for idx, uid in enumerate(sorted(uids))}

    @staticmethod
    def _shuffle_and_partition(
        miner_uids: list[int],
        active_count: int,
        seed: int,
    ) -> list[list[int]]:
        """Deterministically shuffle miners and split into N equal subsets.

        Uses interleaved slicing (uids[i::N]) so subsets are as equal
        as possible when len(miners) % N != 0.
        """
        shuffled = list(miner_uids)
        rng = random.Random(seed)
        rng.shuffle(shuffled)
        return [shuffled[i::active_count] for i in range(active_count)]

    def get_miner_subset(
        self,
        my_uid: int,
        all_miner_uids: list[int],
    ) -> Optional[list[int]]:
        """Get this validator's assigned miner subset for the current round.

        Prerequisites: commit_timestamp() and fetch_active_validators()
        must have been called successfully this round.

        IMPORTANT: The partitioning is computed over ALL neuron UIDs in the
        subnet (0..N-1), NOT the caller's miner list. This ensures all
        validators compute identical partitions even if their metagraph
        snapshots differ slightly. The caller's all_miner_uids is only used
        to filter the final subset (remove validators, offline miners, etc.).

        Returns:
            List of miner UIDs assigned to this validator, or None if
            this validator is not in the active set (fallback needed).
        """
        if not self._active_validator_uids or self._last_sync_time is None:
            logger.warning("Commit sync not ready — no active validators or sync time")
            return None

        if my_uid not in self._active_validator_uids:
            logger.warning(
                "Validator UID {} not in active set {} — not committed?",
                my_uid, self._active_validator_uids,
            )
            return None

        # Derive the full UID space from the metagraph size — this is the
        # same for all validators on the same subnet, regardless of when
        # they last resynced (neuron count only changes on registrations
        # which are visible to all validators after finality).
        try:
            subnet_size = len(self._metagraph.S)
        except Exception:
            subnet_size = max(all_miner_uids) + 1 if all_miner_uids else 256

        # Partition the FULL uid space [0..subnet_size) deterministically.
        # Every validator computes this identically.
        canonical_uids = list(range(subnet_size))
        seed = self._compute_seed(self._last_sync_time)
        subsets = self._shuffle_and_partition(
            canonical_uids, self._active_count, seed
        )

        uid_to_idx = self._map_to_consecutive(self._active_validator_uids)
        my_index = uid_to_idx[my_uid]
        my_raw_subset = subsets[my_index]

        # Filter to only UIDs that are actually queryable miners
        # (removes ourselves, other validators, empty slots, etc.)
        miner_set = set(all_miner_uids)
        my_subset = [uid for uid in my_raw_subset if uid in miner_set]

        logger.info(
            "🎯 Validator {} (index {}/{}) assigned {}/{} miners "
            "(from {} canonical UIDs)",
            my_uid, my_index, self._active_count,
            len(my_subset), len(all_miner_uids), subnet_size,
        )
        return my_subset

    # ------------------------------------------------------------------
    # Combined flow: commit → wait → discover → partition
    # ------------------------------------------------------------------

    async def sync_round(
        self,
        sync_time: int,
        my_uid: int,
        all_miner_uids: list[int],
        loop=None,
    ) -> Optional[list[int]]:
        """Execute the full commit-sync flow for one round.

        1. Commit the epoch-boundary timestamp to chain
        2. Wait for blockchain finality (so all validators' commits are visible)
        3. Query all commitments to discover active validators
        4. Deterministically partition miners and return this validator's subset

        Args:
            sync_time: The epoch-boundary timestamp. ALL validators must pass
                the same value — the caller ensures this by only triggering
                when ``int(time.time()) % epoch_length == 0``.

        Returns miner subset on success, None on failure (caller should
        skip the round).
        """
        import asyncio

        if loop is None:
            loop = asyncio.get_event_loop()

        # Step 1: Commit (blocking chain call → executor)
        success = await loop.run_in_executor(
            self._executor, self.commit_timestamp, sync_time
        )
        if not success:
            return None

        # Step 2: Wait for blockchain finality
        logger.info(
            "⏳ Waiting {}s for blockchain finality...",
            COMMIT_FINALITY_WAIT,
        )
        end_time = time.time() + COMMIT_FINALITY_WAIT
        while time.time() < end_time:
            await asyncio.sleep(1)

        # Step 3: Discover active validators (blocking chain call → executor)
        active_uids = await loop.run_in_executor(
            self._executor, self.fetch_active_validators, sync_time
        )
        if not active_uids:
            logger.warning("❌ No active validators found — skipping round")
            return None

        # Step 4: Partition
        return self.get_miner_subset(my_uid, all_miner_uids)
