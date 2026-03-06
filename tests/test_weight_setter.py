"""Tests for leaderboard EMA, power-law weighting, and weight setter."""

import numpy as np
import pytest

from tuneforge.rewards.leaderboard import MinerLeaderboard
from tests.mock_subtensor import MockSubtensor, MockWallet


class TestLeaderboardEMA:

    def test_first_update_sets_ema(self):
        lb = MinerLeaderboard(alpha=0.2)
        lb.update(0, 0.8)
        from tuneforge.config.scoring_config import EMA_NEW_MINER_SEED
        expected = 0.2 * 0.8 + 0.8 * EMA_NEW_MINER_SEED
        assert lb.get_ema(0) == pytest.approx(expected)

    def test_ema_formula(self):
        lb = MinerLeaderboard(alpha=0.2)
        lb.update(0, 1.0)
        lb.update(0, 0.5)
        from tuneforge.config.scoring_config import EMA_NEW_MINER_SEED
        r1 = 0.2 * 1.0 + 0.8 * EMA_NEW_MINER_SEED
        r2 = 0.2 * 0.5 + 0.8 * r1
        assert lb.get_ema(0) == pytest.approx(r2)

    def test_ema_converges(self):
        lb = MinerLeaderboard(alpha=0.2)
        for _ in range(100):
            lb.update(0, 0.7)
        assert lb.get_ema(0) == pytest.approx(0.7, abs=0.01)

    def test_multiple_miners(self):
        lb = MinerLeaderboard(alpha=0.2)
        lb.update(0, 0.9)
        lb.update(1, 0.3)
        lb.update(2, 0.6)
        assert lb.get_ema(0) > lb.get_ema(2) > lb.get_ema(1)


class TestLeaderboardPowerLaw:

    def test_zero_ema_zero_weight(self):
        lb = MinerLeaderboard(alpha=0.2, power=2.0)
        assert lb.get_weight(99) == 0.0

    def test_low_ema_gets_weight(self):
        """Even low-scoring miners get some weight (no threshold)."""
        lb = MinerLeaderboard(alpha=0.2, power=2.0)
        for _ in range(50):
            lb.update(0, 0.3)
        weight = lb.get_weight(0)
        assert weight > 0.0

    def test_high_ema_high_weight(self):
        lb = MinerLeaderboard(alpha=0.2, power=2.0)
        for _ in range(50):
            lb.update(0, 0.9)
        weight = lb.get_weight(0)
        assert weight > 0.5

    def test_perfect_score_near_one(self):
        lb = MinerLeaderboard(alpha=0.2, power=2.0)
        for _ in range(100):
            lb.update(0, 1.0)
        weight = lb.get_weight(0)
        assert weight == pytest.approx(1.0, abs=0.05)

    def test_power_law_differentiates(self):
        """Higher power means top scorers get disproportionately more weight."""
        lb = MinerLeaderboard(alpha=0.2, power=2.0)
        for _ in range(50):
            lb.update(0, 0.9)
            lb.update(1, 0.5)
        w0 = lb.get_weight(0)
        w1 = lb.get_weight(1)
        # 0.9^2 / 0.5^2 = 0.81/0.25 = 3.24x
        assert w0 > 3 * w1

    def test_higher_power_more_competitive(self):
        """Power=3 separates miners more than power=2."""
        lb2 = MinerLeaderboard(alpha=0.2, power=2.0)
        lb3 = MinerLeaderboard(alpha=0.2, power=3.0)
        for _ in range(50):
            lb2.update(0, 0.9)
            lb2.update(1, 0.5)
            lb3.update(0, 0.9)
            lb3.update(1, 0.5)
        ratio2 = lb2.get_weight(0) / lb2.get_weight(1)
        ratio3 = lb3.get_weight(0) / lb3.get_weight(1)
        assert ratio3 > ratio2


class TestLeaderboardNewMinerSeed:

    def test_new_miner_starts_at_seed(self):
        from tuneforge.config.scoring_config import EMA_NEW_MINER_SEED
        lb = MinerLeaderboard(alpha=0.2)
        lb.update(0, 0.9)
        expected = 0.2 * 0.9 + 0.8 * EMA_NEW_MINER_SEED
        assert lb.get_ema(0) == pytest.approx(expected)


class TestLeaderboardSummary:

    def test_empty_summary(self):
        lb = MinerLeaderboard()
        s = lb.summary()
        assert s["total_miners"] == 0

    def test_populated_summary(self):
        lb = MinerLeaderboard(alpha=0.2)
        for _ in range(10):
            lb.update(0, 0.8)
            lb.update(1, 0.5)
        s = lb.summary()
        assert s["total_miners"] == 2
        assert s["with_weight"] >= 0


class TestWeightNormalization:

    def test_weights_sum_to_one(self):
        lb = MinerLeaderboard(alpha=0.2, power=2.0)
        for _ in range(15):
            lb.update(0, 0.9)
            lb.update(1, 0.8)
            lb.update(2, 0.7)

        all_weights = lb.get_all_weights()
        nonzero = {uid: w for uid, w in all_weights.items() if w > 0}
        if nonzero:
            total = sum(nonzero.values())
            assert total == pytest.approx(1.0, abs=1e-6)


class TestTieredWeighting:

    def test_elite_gets_80_percent(self):
        """Top 10 miners should get exactly 80% of total weight."""
        lb = MinerLeaderboard(alpha=0.2, power=2.0, elite_k=10, elite_pool=0.80)
        # Create 20 miners with varying quality
        for _ in range(50):
            for uid in range(20):
                lb.update(uid, 0.5 + 0.025 * uid)  # UID 19 best, UID 0 worst

        all_weights = lb.get_all_weights()
        ranked = sorted(all_weights.items(), key=lambda x: x[1], reverse=True)
        elite_weight = sum(w for _, w in ranked[:10])
        rest_weight = sum(w for _, w in ranked[10:])

        assert elite_weight == pytest.approx(0.80, abs=1e-6)
        assert rest_weight == pytest.approx(0.20, abs=1e-6)

    def test_fewer_than_k_miners_get_100(self):
        """If fewer than elite_k miners, all share 100% (no empty rest tier)."""
        lb = MinerLeaderboard(alpha=0.2, power=2.0, elite_k=10, elite_pool=0.80)
        for _ in range(20):
            lb.update(0, 0.9)
            lb.update(1, 0.7)
            lb.update(2, 0.5)

        all_weights = lb.get_all_weights()
        total = sum(all_weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_elite_ordering_by_ema(self):
        """Top miners by EMA should be in the elite tier."""
        lb = MinerLeaderboard(alpha=0.2, power=2.0, elite_k=3, elite_pool=0.80)
        for _ in range(50):
            lb.update(0, 0.90)
            lb.update(1, 0.80)
            lb.update(2, 0.70)
            lb.update(3, 0.60)
            lb.update(4, 0.50)

        all_weights = lb.get_all_weights()
        # Top 3 (UIDs 0,1,2) should have 80% total
        elite_weight = all_weights[0] + all_weights[1] + all_weights[2]
        rest_weight = all_weights[3] + all_weights[4]

        assert elite_weight == pytest.approx(0.80, abs=1e-6)
        assert rest_weight == pytest.approx(0.20, abs=1e-6)
        # Within elite, higher EMA gets more weight
        assert all_weights[0] > all_weights[1] > all_weights[2]
        # Within rest, higher EMA gets more weight
        assert all_weights[3] > all_weights[4]

    def test_quality_still_matters_within_tier(self):
        """Within the elite tier, a much better miner gets more weight."""
        lb = MinerLeaderboard(alpha=0.2, power=2.0, elite_k=10, elite_pool=0.80)
        for _ in range(50):
            lb.update(0, 0.95)  # Best
            lb.update(1, 0.50)  # Weakest elite member
        all_weights = lb.get_all_weights()
        # Both in elite (only 2 miners, < elite_k), but 0 should get more
        assert all_weights[0] > all_weights[1]

    def test_11th_miner_gets_much_less(self):
        """The 11th ranked miner should get less than the 10th due to tier cliff."""
        lb = MinerLeaderboard(alpha=0.2, power=2.0, elite_k=10, elite_pool=0.80)
        for _ in range(50):
            for uid in range(15):
                lb.update(uid, 0.7 + 0.02 * uid)

        all_weights = lb.get_all_weights()
        ranked = sorted(all_weights.items(), key=lambda x: x[1], reverse=True)
        w_10th = ranked[9][1]   # Last elite
        w_11th = ranked[10][1]  # First rest
        # Tier cliff: 10th miner (elite pool) should get more than 11th (rest pool)
        assert w_10th > w_11th
        # The elite pool total should be 4x the rest pool
        elite_total = sum(w for _, w in ranked[:10])
        rest_total = sum(w for _, w in ranked[10:])
        assert elite_total / rest_total == pytest.approx(4.0, abs=0.01)


class TestHotkeyChangeDetection:

    def test_hotkey_change_resets_ema(self):
        """When a UID gets a new hotkey, its EMA should reset to 0."""
        lb = MinerLeaderboard(alpha=0.2, power=2.0)
        for _ in range(20):
            lb.update(0, 0.9)
        assert lb.get_ema(0) > 0.8

        # First call establishes the baseline hotkeys
        lb.check_hotkey_changes({0: "hotkey_A"})

        # Same hotkey — no reset
        lb.check_hotkey_changes({0: "hotkey_A"})
        assert lb.get_ema(0) > 0.8

        # Different hotkey — UID recycled, EMA should reset
        reset = lb.check_hotkey_changes({0: "hotkey_B"})
        assert 0 in reset
        assert lb.get_ema(0) == 0.0

    def test_new_uid_no_reset(self):
        """A UID appearing for the first time should not be reset."""
        lb = MinerLeaderboard(alpha=0.2, power=2.0)
        lb.update(5, 0.7)
        # First time seeing UID 5 hotkey — should not reset
        reset = lb.check_hotkey_changes({5: "hotkey_X"})
        assert 5 not in reset
        assert lb.get_ema(5) > 0.0

    def test_hotkeys_persisted_in_state(self, tmp_path):
        """Hotkeys should be saved and loaded with EMA state."""
        path = str(tmp_path / "ema_state.json")
        lb1 = MinerLeaderboard(alpha=0.2, power=2.0)
        lb1.update(0, 0.8)
        lb1.check_hotkey_changes({0: "hotkey_A"})
        lb1.save_state(path)

        lb2 = MinerLeaderboard(alpha=0.2, power=2.0)
        lb2.load_state(path)
        # Hotkey should be loaded — same hotkey should NOT reset
        reset = lb2.check_hotkey_changes({0: "hotkey_A"})
        assert 0 not in reset
        assert lb2.get_ema(0) > 0.0

        # Different hotkey SHOULD reset
        reset = lb2.check_hotkey_changes({0: "hotkey_B"})
        assert 0 in reset
        assert lb2.get_ema(0) == 0.0


class TestWeightSetterInterval:

    def test_should_not_update_too_soon(self):
        from tuneforge.rewards.weight_setter import WeightSetter
        st = MockSubtensor()
        wallet = MockWallet()
        mg = st.metagraph()

        ws = WeightSetter(st, wallet, netuid=0, metagraph=mg, update_interval=175)
        ws._last_update_block = 100
        assert not ws.should_update()

    def test_should_update_after_interval(self):
        from tuneforge.rewards.weight_setter import WeightSetter
        st = MockSubtensor()
        wallet = MockWallet()
        mg = st.metagraph()

        ws = WeightSetter(st, wallet, netuid=0, metagraph=mg, update_interval=175)
        ws._last_update_block = 0
        st.advance_blocks(175)
        assert ws.should_update()
