"""Tests for the reward model and scoring pipeline."""

import base64

import numpy as np
import pytest

from tuneforge.base.protocol import MusicGenerationSynapse
from tuneforge.config.scoring_config import SCORING_WEIGHTS
from tuneforge.rewards.leaderboard import MinerLeaderboard


class TestScoringConfig:

    def test_scoring_weights_sum_to_one(self):
        total = sum(SCORING_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_all_weights_non_negative(self):
        for key, val in SCORING_WEIGHTS.items():
            assert val >= 0, f"{key} weight should be non-negative"


class TestRewardModelResponses:

    def test_empty_response_has_no_audio(self):
        syn = MusicGenerationSynapse(
            prompt="test", genre="rock", mood="happy",
            tempo_bpm=120, duration_seconds=10.0, challenge_id="test-1",
        )
        assert syn.audio_b64 is None
        assert syn.deserialize() is None

    def test_synapse_with_audio_deserializes(self, wav_from_audio, sample_audio_sine, sample_rate):
        wav_bytes = wav_from_audio(sample_audio_sine, sample_rate)
        b64 = base64.b64encode(wav_bytes).decode()
        syn = MusicGenerationSynapse(
            prompt="test", genre="rock", mood="happy",
            tempo_bpm=120, duration_seconds=10.0, challenge_id="test-2",
            audio_b64=b64, sample_rate=sample_rate, generation_time_ms=5000,
        )
        raw = syn.deserialize()
        assert raw is not None
        assert len(raw) > 44


class TestSpeedScoring:
    """Speed scoring uses validator-measured dendrite.process_time (seconds)."""

    @staticmethod
    def _make_synapse_with_process_time(process_time):
        """Create synapse with dendrite.process_time set."""
        from bittensor.core.synapse import TerminalInfo
        syn = MusicGenerationSynapse()
        syn.dendrite = TerminalInfo(process_time=process_time)
        return syn

    def test_fast_generation(self):
        from tuneforge.rewards.reward import ProductionRewardModel as PRM
        syn = self._make_synapse_with_process_time(10.0)
        assert PRM._speed_score(syn) == pytest.approx(1.0)

    def test_slow_generation(self):
        from tuneforge.rewards.reward import ProductionRewardModel as PRM
        syn = self._make_synapse_with_process_time(95.0)
        assert PRM._speed_score(syn) == pytest.approx(0.0)

    def test_mid_generation(self):
        from tuneforge.rewards.reward import ProductionRewardModel as PRM
        # ratio = 30 / 10 = 3.0 → score ≈ 0.30 (duration-relative)
        syn = self._make_synapse_with_process_time(30.0)
        score = PRM._speed_score(syn)
        assert 0.2 <= score <= 0.4

    def test_unknown_generation_time(self):
        from tuneforge.rewards.reward import ProductionRewardModel as PRM
        syn = MusicGenerationSynapse(generation_time_ms=None)
        assert PRM._speed_score(syn) == pytest.approx(0.5)


class TestLeaderboardSteepening:
    """Tests for the steepening function and baseline behavior (FIND-005)."""

    def test_below_baseline_is_zero(self):
        lb = MinerLeaderboard()
        # Manually set EMA below baseline
        lb._ema[0] = 0.30
        lb._rounds[0] = 20  # warmed up
        assert lb.get_weight(0) == 0.0

    def test_above_baseline_positive(self):
        lb = MinerLeaderboard()
        lb._ema[0] = 0.60
        lb._rounds[0] = 20
        w = lb.get_weight(0)
        assert w > 0.0

    def test_gradient_is_monotonic(self):
        lb = MinerLeaderboard()
        weights = []
        for ema in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
            lb._ema[0] = ema
            lb._rounds[0] = 20
            weights.append(lb.get_weight(0))
        # Weights should be strictly non-decreasing
        for i in range(1, len(weights)):
            assert weights[i] >= weights[i - 1]

    def test_quality_improvement_always_rewarded(self):
        """A miner improving quality should always gain weight (FIND-002 regression)."""
        from tuneforge.rewards.reward import ProductionRewardModel as PRM
        from tuneforge.config.scoring_config import SCORING_WEIGHTS

        # Miner A: high quality across all quality dimensions, slow speed
        quality_weight = (SCORING_WEIGHTS["quality"]
                          + SCORING_WEIGHTS["musicality"]
                          + SCORING_WEIGHTS["production"]
                          + SCORING_WEIGHTS.get("melody", 0)
                          + SCORING_WEIGHTS.get("neural_quality", 0)
                          + SCORING_WEIGHTS["preference"]
                          + SCORING_WEIGHTS.get("structural", 0)
                          + SCORING_WEIGHTS.get("vocal", 0))
        comp_a = (SCORING_WEIGHTS["clap"] * 0.60
                  + quality_weight * 0.80
                  + SCORING_WEIGHTS["diversity"] * 0.50
                  + SCORING_WEIGHTS["speed"] * 0.30)
        # Miner B: low quality, fast speed
        comp_b = (SCORING_WEIGHTS["clap"] * 0.60
                  + quality_weight * 0.45
                  + SCORING_WEIGHTS["diversity"] * 0.50
                  + SCORING_WEIGHTS["speed"] * 1.00)
        # Quality miner should outscore speed miner
        assert comp_a > comp_b


class TestDurationPenalty:

    def test_exact_duration_no_penalty(self, sample_rate):
        from tuneforge.rewards.reward import ProductionRewardModel as PRM
        n_samples = int(10.0 * sample_rate)
        audio = np.zeros(n_samples, dtype=np.float32)
        penalty = PRM._duration_penalty(audio, sample_rate, 10.0)
        assert penalty == pytest.approx(1.0)

    def test_way_off_duration_zero(self, sample_rate):
        from tuneforge.rewards.reward import ProductionRewardModel as PRM
        n_samples = int(3.0 * sample_rate)
        audio = np.zeros(n_samples, dtype=np.float32)
        penalty = PRM._duration_penalty(audio, sample_rate, 10.0)
        assert penalty == pytest.approx(0.0)

    def test_within_tolerance_no_penalty(self, sample_rate):
        from tuneforge.rewards.reward import ProductionRewardModel as PRM
        n_samples = int(8.5 * sample_rate)
        audio = np.zeros(n_samples, dtype=np.float32)
        penalty = PRM._duration_penalty(audio, sample_rate, 10.0)
        assert penalty == pytest.approx(1.0)
