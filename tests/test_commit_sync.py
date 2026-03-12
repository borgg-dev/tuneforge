"""Comprehensive tests for commit-based multi-validator synchronization and permutation.

Epoch-aligned commit-sync approach:
- ALL validators poll every 1s and fire when ``time % epoch_length == 0``
- They all commit the SAME timestamp (the epoch boundary value)
- After finality wait, they discover each other and partition miners

Tests cover:
1. Timestamp commitment
2. Active validator discovery
3. Deterministic seed + UID mapping
4. Raw partitioning (mutual exclusivity, coverage, balance)
5. Canonical-UID partitioning with miner filtering
6. End-to-end multi-validator scenarios
7. Edge cases and failure modes
8. Async sync_round integration
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from tuneforge.validation.commit_sync import (
    COMMIT_MAX_RETRIES,
    CommitSync,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metagraph(
    validator_uids: list[int],
    total_uids: int = 0,
    stakes: dict[int, float] | None = None,
):
    if total_uids == 0:
        total_uids = max(validator_uids + [0]) + 5

    metagraph = MagicMock()
    neurons = []
    stake_list = [1000.0] * total_uids

    if stakes:
        for uid, s in stakes.items():
            if uid < total_uids:
                stake_list[uid] = s

    for uid in range(total_uids):
        n = MagicMock()
        n.uid = uid
        n.hotkey = f"5FakeHotkey{uid:04d}"
        n.validator_permit = uid in validator_uids
        neurons.append(n)

    metagraph.neurons = neurons
    metagraph.S = stake_list
    return metagraph


def _make_commit_sync(
    my_uid: int = 0,
    validator_uids: list[int] | None = None,
    total_uids: int = 0,
    stakes: dict[int, float] | None = None,
    min_stake: float = 0.0,
) -> CommitSync:
    if validator_uids is None:
        validator_uids = [0, 1, 2]

    subtensor = MagicMock()
    wallet = MagicMock()
    wallet.hotkey.ss58_address = f"5FakeHotkey{my_uid:04d}"
    metagraph = _make_metagraph(validator_uids, total_uids, stakes)

    return CommitSync(
        subtensor=subtensor, wallet=wallet, netuid=1,
        metagraph=metagraph, min_stake=min_stake,
    )


def _prepare_sync(cs: CommitSync, sync_time: int, active_uids: list[int]):
    cs._last_sync_time = sync_time
    cs._active_validator_uids = sorted(active_uids)
    cs._active_count = len(active_uids)


# ===========================================================================
# 1. COMMIT TIMESTAMP
# ===========================================================================

class TestCommitTimestamp:

    def test_commit_success(self):
        cs = _make_commit_sync(my_uid=0)
        assert cs.commit_timestamp(sync_time=1710000000) is True
        assert cs._subtensor.commit.call_count == 1
        # Verify exact value committed
        call_args = cs._subtensor.commit.call_args
        assert call_args.kwargs["data"] == "1710000000"

    def test_all_validators_commit_same_value(self):
        """When all validators pass the same sync_time, they commit the same data."""
        sync_time = 1710000480  # an epoch boundary
        committed_values = []
        for uid in [0, 5, 10]:
            cs = _make_commit_sync(my_uid=uid)
            cs.commit_timestamp(sync_time=sync_time)
            call_args = cs._subtensor.commit.call_args
            committed_values.append(call_args.kwargs["data"])

        assert all(v == "1710000480" for v in committed_values)

    def test_commit_retries_on_failure(self):
        cs = _make_commit_sync(my_uid=0)
        cs._subtensor.commit.side_effect = [Exception("err"), Exception("err"), None]
        assert cs.commit_timestamp(sync_time=100) is True
        assert cs._subtensor.commit.call_count == 3

    def test_commit_all_retries_fail(self):
        cs = _make_commit_sync(my_uid=0)
        cs._subtensor.commit.side_effect = Exception("permanent failure")
        assert cs.commit_timestamp(sync_time=100) is False
        assert cs._subtensor.commit.call_count == COMMIT_MAX_RETRIES

    def test_commit_stores_sync_time(self):
        cs = _make_commit_sync(my_uid=0)
        cs.commit_timestamp(sync_time=999)
        assert cs.last_sync_time == 999

    def test_commit_failure_does_not_update_sync_time(self):
        cs = _make_commit_sync(my_uid=0)
        cs._subtensor.commit.side_effect = Exception("fail")
        cs.commit_timestamp(sync_time=999)
        assert cs.last_sync_time is None


# ===========================================================================
# 2. EPOCH BOUNDARY (caller's responsibility — test the pattern)
# ===========================================================================

class TestEpochBoundary:
    """Verify that the modulo pattern produces the same timestamp for all."""

    def test_modulo_fires_at_boundary(self):
        epoch_length = 480
        # Simulate 3 validators polling at slightly different ms
        for t in [1710000480, 1710000480, 1710000480]:
            assert t % epoch_length == 0

    def test_modulo_does_not_fire_off_boundary(self):
        epoch_length = 480
        assert 1710000481 % epoch_length != 0
        assert 1710000479 % epoch_length != 0

    def test_all_validators_get_same_sync_time(self):
        """The sync_time passed to commit is current_time when modulo == 0.
        Since all validators check every second, they all see the same second."""
        epoch_length = 480
        boundary = 1710000480
        # All validators call at the same second
        sync_times = [boundary if boundary % epoch_length == 0 else None for _ in range(16)]
        assert all(t == boundary for t in sync_times)


# ===========================================================================
# 3. ACTIVE VALIDATOR DISCOVERY
# ===========================================================================

class TestFetchActiveValidators:

    def test_discovers_matching_validators(self):
        active = [5, 10, 15]
        cs = _make_commit_sync(my_uid=5, validator_uids=active)
        cs._subtensor.get_all_commitments.return_value = {
            f"5FakeHotkey{uid:04d}": "12345" for uid in active
        }
        uids = cs.fetch_active_validators(12345)
        assert sorted(uids) == sorted(active)
        assert cs.active_count == 3

    def test_ignores_non_matching_timestamps(self):
        active = [5, 10, 15]
        cs = _make_commit_sync(my_uid=5, validator_uids=active)
        cs._subtensor.get_all_commitments.return_value = {
            "5FakeHotkey0005": "12345",
            "5FakeHotkey0010": "99999",
            "5FakeHotkey0015": "12345",
        }
        uids = cs.fetch_active_validators(12345)
        assert sorted(uids) == [5, 15]

    def test_ignores_non_validators(self):
        cs = _make_commit_sync(my_uid=5, validator_uids=[5, 10])
        cs._subtensor.get_all_commitments.return_value = {
            "5FakeHotkey0005": "12345",
            "5FakeHotkey0003": "12345",
            "5FakeHotkey0010": "12345",
        }
        uids = cs.fetch_active_validators(12345)
        assert sorted(uids) == [5, 10]

    def test_filters_by_minimum_stake(self):
        cs = _make_commit_sync(
            my_uid=5, validator_uids=[5, 10, 15],
            stakes={5: 1000.0, 10: 50.0, 15: 500.0}, min_stake=100.0,
        )
        cs._subtensor.get_all_commitments.return_value = {
            f"5FakeHotkey{uid:04d}": "999" for uid in [5, 10, 15]
        }
        uids = cs.fetch_active_validators(999)
        assert sorted(uids) == [5, 15]

    def test_no_commitments_returns_empty(self):
        cs = _make_commit_sync(my_uid=0)
        cs._subtensor.get_all_commitments.return_value = {}
        assert cs.fetch_active_validators(12345) == []

    def test_chain_error_returns_empty(self):
        cs = _make_commit_sync(my_uid=0)
        cs._subtensor.get_all_commitments.side_effect = Exception("chain down")
        assert cs.fetch_active_validators(12345) == []

    def test_result_is_sorted(self):
        active = [30, 10, 20, 5]
        cs = _make_commit_sync(my_uid=5, validator_uids=active, total_uids=35)
        cs._subtensor.get_all_commitments.return_value = {
            f"5FakeHotkey{uid:04d}": "42" for uid in active
        }
        assert cs.fetch_active_validators(42) == [5, 10, 20, 30]


# ===========================================================================
# 4. DETERMINISTIC SEED + UID MAPPING
# ===========================================================================

class TestSeedAndMapping:

    def test_same_time_same_seed(self):
        assert CommitSync._compute_seed(100) == CommitSync._compute_seed(100)

    def test_different_time_different_seed(self):
        assert CommitSync._compute_seed(100) != CommitSync._compute_seed(101)

    def test_seed_is_32bit(self):
        for t in [0, 1, 2**40]:
            assert 0 <= CommitSync._compute_seed(t) < 2**32

    def test_consecutive_mapping(self):
        assert CommitSync._map_to_consecutive([30, 10, 50, 20]) == {10: 0, 20: 1, 30: 2, 50: 3}

    def test_single_uid_mapping(self):
        assert CommitSync._map_to_consecutive([42]) == {42: 0}


# ===========================================================================
# 5. RAW PARTITIONING
# ===========================================================================

class TestRawPartitioning:

    def test_deterministic(self):
        uids = list(range(100))
        seed = CommitSync._compute_seed(42)
        assert CommitSync._shuffle_and_partition(uids, 3, seed) == \
               CommitSync._shuffle_and_partition(uids, 3, seed)

    def test_mutually_exclusive(self):
        uids = list(range(200))
        subsets = CommitSync._shuffle_and_partition(uids, 5, seed=999)
        all_assigned = []
        for s in subsets:
            all_assigned.extend(s)
        assert len(all_assigned) == len(set(all_assigned))

    def test_full_coverage(self):
        uids = list(range(150))
        subsets = CommitSync._shuffle_and_partition(uids, 4, seed=42)
        combined = set()
        for s in subsets:
            combined.update(s)
        assert combined == set(uids)

    def test_balanced(self):
        for n_uids in [50, 100, 200, 256]:
            for n_parts in [2, 3, 5, 7, 16]:
                subsets = CommitSync._shuffle_and_partition(list(range(n_uids)), n_parts, seed=n_uids)
                sizes = [len(s) for s in subsets]
                assert max(sizes) - min(sizes) <= 1

    def test_single_validator(self):
        uids = list(range(50))
        subsets = CommitSync._shuffle_and_partition(uids, 1, seed=1)
        assert set(subsets[0]) == set(uids)

    def test_more_validators_than_miners(self):
        subsets = CommitSync._shuffle_and_partition(list(range(3)), 10, seed=7)
        all_assigned = []
        for s in subsets:
            all_assigned.extend(s)
        assert set(all_assigned) == {0, 1, 2}

    def test_empty_list(self):
        subsets = CommitSync._shuffle_and_partition([], 5, seed=42)
        assert all(len(s) == 0 for s in subsets)

    def test_does_not_mutate_input(self):
        uids = list(range(50))
        original = list(uids)
        CommitSync._shuffle_and_partition(uids, 3, seed=42)
        assert uids == original


# ===========================================================================
# 6. CANONICAL-UID PARTITIONING (get_miner_subset)
# ===========================================================================

class TestCanonicalPartitioning:

    def _setup(self, v_uid, validators, total, sync_time):
        cs = _make_commit_sync(my_uid=v_uid, validator_uids=validators, total_uids=total)
        _prepare_sync(cs, sync_time=sync_time, active_uids=validators)
        return cs

    def test_all_agree_disjoint_full_coverage(self):
        validators = [0, 1, 2]
        total = 50
        miners = [uid for uid in range(total) if uid not in validators]
        sync_time = 999

        all_assigned: list[int] = []
        for v in validators:
            subset = self._setup(v, validators, total, sync_time).get_miner_subset(v, miners)
            assert subset is not None
            all_assigned.extend(subset)

        assert len(all_assigned) == len(set(all_assigned)), "Overlap!"
        assert set(all_assigned) == set(miners), "Missing miners!"

    def test_filters_out_validators(self):
        validators = [10, 20, 30]
        total = 40
        miners = [uid for uid in range(total) if uid not in validators]

        for v in validators:
            subset = self._setup(v, validators, total, 123).get_miner_subset(v, miners)
            for uid in subset:
                assert uid not in validators

    def test_stale_metagraph_no_desync(self):
        """Two validators with different miner lists still get disjoint subsets."""
        validators = [0, 1]
        total = 10

        cs0 = self._setup(0, validators, total, 42)
        cs1 = self._setup(1, validators, total, 42)

        # V0 sees UID 8, V1 doesn't
        s0 = cs0.get_miner_subset(0, [2, 3, 4, 5, 6, 7, 8, 9])
        s1 = cs1.get_miner_subset(1, [2, 3, 4, 5, 6, 7, 9])

        assert set(s0).isdisjoint(set(s1))

    def test_not_in_active_set_returns_none(self):
        cs = _make_commit_sync(my_uid=99, validator_uids=[1, 2, 3])
        _prepare_sync(cs, 123, [1, 2, 3])
        assert cs.get_miner_subset(99, list(range(50))) is None

    def test_no_sync_returns_none(self):
        cs = _make_commit_sync(my_uid=1, validator_uids=[1, 2])
        assert cs.get_miner_subset(1, list(range(50))) is None

    def test_empty_miners_returns_empty_list(self):
        cs = _make_commit_sync(my_uid=0, validator_uids=[0], total_uids=5)
        _prepare_sync(cs, 42, [0])
        result = cs.get_miner_subset(0, [])
        assert result == []

    def test_reshuffles_across_rounds(self):
        validators = [1, 2, 3]
        total = 60
        miners = [uid for uid in range(total) if uid not in validators]

        s1 = self._setup(1, validators, total, 1000).get_miner_subset(1, miners)
        s2 = self._setup(1, validators, total, 2000).get_miner_subset(1, miners)
        assert set(s1) != set(s2), "Should differ across rounds"


# ===========================================================================
# 7. END-TO-END SCENARIOS
# ===========================================================================

class TestEndToEnd:

    def _run_round(self, validators, total, miners, sync_time):
        """Simulate one round: all validators partition, verify disjoint + complete."""
        all_assigned: list[int] = []
        for v in validators:
            cs = _make_commit_sync(my_uid=v, validator_uids=validators, total_uids=total)
            _prepare_sync(cs, sync_time, validators)
            subset = cs.get_miner_subset(v, miners)
            assert subset is not None
            all_assigned.extend(subset)

        assert len(all_assigned) == len(set(all_assigned)), "Overlap!"
        assert set(all_assigned) == set(miners), "Missing miners!"

    def test_16_validators_200_miners(self):
        vals = list(range(16))
        miners = list(range(16, 216))
        self._run_round(vals, 220, miners, 1710000480)

    def test_3_validators_90_miners(self):
        vals = [10, 20, 30]
        miners = [uid for uid in range(100) if uid not in vals]
        self._run_round(vals, 100, miners, 42)

    def test_1_validator_all_miners(self):
        miners = list(range(1, 200))
        self._run_round([0], 200, miners, 999)

    def test_64_validators_256_miners(self):
        vals = list(range(64))
        miners = list(range(64, 320))
        self._run_round(vals, 320, miners, 7777)

    def test_validator_joins(self):
        total = 130
        miners = list(range(10, 130))

        # Round 1: 2 validators
        self._run_round([1, 2], total, miners, 1000)
        # Round 2: 3 validators
        self._run_round([1, 2, 3], total, miners, 2000)

    def test_validator_drops(self):
        total = 100
        miners = list(range(10, 100))

        self._run_round([1, 2, 3], total, miners, 5000)
        self._run_round([1, 3], total, miners, 5480)

    def test_independent_instances_agree(self):
        """Separate CommitSync processes must produce identical global partitioning."""
        validators = [10, 20, 30]
        total = 100
        miners = [uid for uid in range(total) if uid not in validators]
        sync_time = 42424242

        subsets: dict[int, set] = {}
        for v in validators:
            cs = _make_commit_sync(my_uid=v, validator_uids=validators, total_uids=total)
            _prepare_sync(cs, sync_time, validators)
            subsets[v] = set(cs.get_miner_subset(v, miners))

        # All pairwise disjoint
        vals = list(subsets.values())
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                assert vals[i].isdisjoint(vals[j])

        # Full coverage
        assert set().union(*vals) == set(miners)


# ===========================================================================
# 8. ASYNC SYNC_ROUND
# ===========================================================================

class TestSyncRound:

    @pytest.fixture
    def event_loop(self):
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    def test_success(self, event_loop):
        validators = [0, 1, 2]
        total = 35
        miners = [uid for uid in range(total) if uid not in validators]
        sync_time = 1710000480

        cs = _make_commit_sync(my_uid=0, validator_uids=validators, total_uids=total)
        cs._subtensor.get_all_commitments.return_value = {
            f"5FakeHotkey{uid:04d}": str(sync_time) for uid in validators
        }

        with patch("tuneforge.validation.commit_sync.COMMIT_FINALITY_WAIT", 0):
            subset = event_loop.run_until_complete(
                cs.sync_round(sync_time=sync_time, my_uid=0,
                              all_miner_uids=miners, loop=event_loop)
            )

        assert subset is not None
        assert all(m in miners for m in subset)
        assert len(subset) > 0

    def test_commit_failure_returns_none(self, event_loop):
        cs = _make_commit_sync(my_uid=0)
        cs._subtensor.commit.side_effect = Exception("chain down")

        with patch("tuneforge.validation.commit_sync.COMMIT_FINALITY_WAIT", 0):
            result = event_loop.run_until_complete(
                cs.sync_round(sync_time=100, my_uid=0,
                              all_miner_uids=list(range(20)), loop=event_loop)
            )
        assert result is None

    def test_no_active_validators_returns_none(self, event_loop):
        cs = _make_commit_sync(my_uid=0, validator_uids=[0])
        cs._subtensor.get_all_commitments.return_value = {}

        with patch("tuneforge.validation.commit_sync.COMMIT_FINALITY_WAIT", 0):
            result = event_loop.run_until_complete(
                cs.sync_round(sync_time=100, my_uid=0,
                              all_miner_uids=list(range(20)), loop=event_loop)
            )
        assert result is None


# ===========================================================================
# 9. UPDATE METAGRAPH
# ===========================================================================

class TestUpdateMetagraph:

    def test_update_metagraph(self):
        cs = _make_commit_sync(my_uid=0, validator_uids=[0])
        new_mg = MagicMock()
        cs.update_metagraph(new_mg)
        assert cs._metagraph is new_mg
