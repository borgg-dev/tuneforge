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

    def test_all_weights_positive(self):
        for key, val in SCORING_WEIGHTS.items():
            assert val > 0, f"{key} weight should be positive"


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

    def test_fast_generation(self):
        from tuneforge.rewards.reward import ProductionRewardModel as PRM
        syn = MusicGenerationSynapse(generation_time_ms=3000)
        assert PRM._speed_score(syn) == pytest.approx(1.0)

    def test_slow_generation(self):
        from tuneforge.rewards.reward import ProductionRewardModel as PRM
        syn = MusicGenerationSynapse(generation_time_ms=65000)
        assert PRM._speed_score(syn) == pytest.approx(0.0)

    def test_mid_generation(self):
        from tuneforge.rewards.reward import ProductionRewardModel as PRM
        syn = MusicGenerationSynapse(generation_time_ms=30000)
        score = PRM._speed_score(syn)
        assert 0.2 <= score <= 0.4

    def test_unknown_generation_time(self):
        from tuneforge.rewards.reward import ProductionRewardModel as PRM
        syn = MusicGenerationSynapse(generation_time_ms=None)
        assert PRM._speed_score(syn) == pytest.approx(0.5)


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
