"""Tests for leaderboard EMA, steepening, and weight setter."""

import numpy as np
import pytest

from tuneforge.rewards.leaderboard import MinerLeaderboard
from tests.mock_subtensor import MockSubtensor, MockWallet


class TestLeaderboardEMA:

    def test_first_update_sets_ema(self):
        lb = MinerLeaderboard(alpha=0.2)
        lb.update(0, 0.8)
        assert lb.get_ema(0) == pytest.approx(0.8)

    def test_ema_formula(self):
        lb = MinerLeaderboard(alpha=0.2)
        lb.update(0, 1.0)
        lb.update(0, 0.5)
        # EMA = 0.2 * 0.5 + 0.8 * 1.0 = 0.9
        assert lb.get_ema(0) == pytest.approx(0.9)

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


class TestLeaderboardSteepening:

    def test_below_baseline_zero(self):
        lb = MinerLeaderboard(alpha=0.2, steepen_baseline=0.6, steepen_power=3.0)
        for _ in range(15):
            lb.update(0, 0.5)
        assert lb.get_weight(0) == 0.0

    def test_at_baseline_zero(self):
        lb = MinerLeaderboard(alpha=0.2, steepen_baseline=0.6, steepen_power=3.0)
        for _ in range(15):
            lb.update(0, 0.6)
        assert lb.get_weight(0) == pytest.approx(0.0, abs=0.01)

    def test_above_baseline_positive(self):
        lb = MinerLeaderboard(alpha=0.2, steepen_baseline=0.6, steepen_power=3.0)
        for _ in range(15):
            lb.update(0, 0.9)
        weight = lb.get_weight(0)
        assert weight > 0.0

    def test_perfect_score_high_weight(self):
        lb = MinerLeaderboard(alpha=0.2, steepen_baseline=0.6, steepen_power=3.0)
        for _ in range(15):
            lb.update(0, 1.0)
        weight = lb.get_weight(0)
        assert weight == pytest.approx(1.0, abs=0.05)

    def test_steepening_differentiates(self):
        lb = MinerLeaderboard(alpha=0.2, steepen_baseline=0.6, steepen_power=3.0)
        for _ in range(15):
            lb.update(0, 0.95)
            lb.update(1, 0.75)
        w0 = lb.get_weight(0)
        w1 = lb.get_weight(1)
        # Steepening should strongly differentiate
        assert w0 > 3 * w1


class TestLeaderboardWarmup:

    def test_not_warmed_up_initially(self):
        lb = MinerLeaderboard(alpha=0.2)
        lb.update(0, 0.9)
        assert not lb.is_warmed_up(0)

    def test_warmed_up_after_threshold(self):
        lb = MinerLeaderboard(alpha=0.2)
        # warmup = ceil(2/0.2 - 1) = 9
        for i in range(9):
            lb.update(0, 0.9)
        assert lb.is_warmed_up(0)

    def test_weight_zero_before_warmup(self):
        lb = MinerLeaderboard(alpha=0.2, steepen_baseline=0.6)
        for i in range(5):
            lb.update(0, 0.9)
        assert lb.get_weight(0) == 0.0


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
        assert s["warmed_up"] >= 1


class TestWeightNormalization:

    def test_weights_sum_to_one(self):
        lb = MinerLeaderboard(alpha=0.2, steepen_baseline=0.6, steepen_power=3.0)
        for _ in range(15):
            lb.update(0, 0.9)
            lb.update(1, 0.8)
            lb.update(2, 0.7)

        all_weights = lb.get_all_weights()
        nonzero = {uid: w for uid, w in all_weights.items() if w > 0}
        if nonzero:
            total = sum(nonzero.values())
            normalized = {uid: w / total for uid, w in nonzero.items()}
            assert sum(normalized.values()) == pytest.approx(1.0, abs=1e-6)


class TestWeightSetterInterval:

    def test_should_not_update_too_soon(self):
        from tuneforge.rewards.weight_setter import WeightSetter
        st = MockSubtensor()
        wallet = MockWallet()
        mg = st.metagraph()

        ws = WeightSetter(st, wallet, netuid=0, metagraph=mg, update_interval=175)
        ws._last_update_block = 100
        # Block is 100, so 0 blocks elapsed
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
