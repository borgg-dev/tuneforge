"""Tests for stereo quality scoring metrics."""

import numpy as np
import pytest

from tuneforge.scoring.stereo_quality import StereoQualityScorer


@pytest.fixture
def scorer():
    return StereoQualityScorer()


@pytest.fixture
def sr():
    return 32000


class TestStereoQualityWeights:

    def test_weights_sum_to_one(self):
        total = sum(StereoQualityScorer.WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestMonoInput:

    def test_mono_1d_returns_penalty(self, scorer, sr):
        """1-D mono audio should get mild penalty scores."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, sr * 2)).astype(np.float32)
        scores = scorer.score(audio, sr)
        assert scores["stereo_width"] == 0.3
        assert scores["phase_coherence"] == 0.8
        assert scores["mid_side_balance"] == 0.3

    def test_mono_2d_single_channel(self, scorer, sr):
        """2-D with shape (N, 1) should be treated as mono."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, sr * 2)).astype(np.float32)
        audio_2d = audio.reshape(-1, 1)
        scores = scorer.score(audio_2d, sr)
        assert scores["stereo_width"] == 0.3


class TestFakeStereo:

    def test_identical_lr_low_width(self, scorer, sr):
        """Identical L/R (mono disguised as stereo) should score low on width."""
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        mono = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        stereo = np.stack([mono, mono], axis=1)  # channels-last
        scores = scorer.score(stereo, sr)
        assert scores["stereo_width"] <= 0.3


class TestGoodStereo:

    def test_different_lr_high_width(self, scorer, sr):
        """Different but correlated L/R should score higher than fake stereo."""
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        # Use different fundamentals so correlation is moderate (~0.3-0.8)
        rng = np.random.default_rng(42)
        left = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * rng.normal(size=len(t))).astype(np.float32)
        right = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * rng.normal(size=len(t))).astype(np.float32)
        stereo = np.stack([left, right], axis=1)
        scores = scorer.score(stereo, sr)
        # Should score higher than fake stereo (0.2) due to decorrelated noise
        assert scores["stereo_width"] > 0.3

    def test_good_stereo_high_phase_coherence(self, scorer, sr):
        """Correlated stereo with shared fundamental should have good phase coherence."""
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        left = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        right = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 1320 * t)).astype(np.float32)
        stereo = np.stack([left, right], axis=1)
        scores = scorer.score(stereo, sr)
        assert scores["phase_coherence"] > 0.5


class TestAntiPhase:

    def test_anti_phase_low_coherence(self, scorer, sr):
        """Perfectly anti-phase L/R should score low on phase coherence."""
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = -left  # Perfect anti-phase
        stereo = np.stack([left, right], axis=1)
        scores = scorer.score(stereo, sr)
        assert scores["phase_coherence"] == 0.0


class TestChannelsFirst:

    def test_channels_first_format(self, scorer, sr):
        """Shape (2, N) should be handled as channels-first."""
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = 0.5 * np.sin(2 * np.pi * 660 * t).astype(np.float32)
        stereo = np.stack([left, right], axis=0)  # (2, N)
        scores = scorer.score(stereo, sr)
        assert set(scores.keys()) == set(StereoQualityScorer.WEIGHTS.keys())
        for v in scores.values():
            assert 0.0 <= v <= 1.0


class TestStereoQualityAggregate:

    def test_aggregate_in_range(self, scorer):
        scores = {"stereo_width": 0.8, "phase_coherence": 0.9, "mid_side_balance": 0.7}
        agg = scorer.aggregate(scores)
        assert 0.0 <= agg <= 1.0

    def test_aggregate_zero(self, scorer):
        scores = {k: 0.0 for k in StereoQualityScorer.WEIGHTS}
        assert scorer.aggregate(scores) == 0.0

    def test_aggregate_max(self, scorer):
        scores = {k: 1.0 for k in StereoQualityScorer.WEIGHTS}
        assert scorer.aggregate(scores) == pytest.approx(1.0, abs=1e-6)


class TestStereoStaticMethods:

    def test_stereo_width_empty(self):
        assert StereoQualityScorer._score_stereo_width(
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        ) == 0.0

    def test_stereo_width_zero_std(self):
        """Constant (DC) signals have zero std, should return 0.0."""
        left = np.ones(1000, dtype=np.float32)
        right = np.ones(1000, dtype=np.float32) * 0.5
        assert StereoQualityScorer._score_stereo_width(left, right) == 0.0

    def test_mid_side_balance_bell_curve(self):
        """Mid ~4x side energy should score near 1.0."""
        # Create signals where mid/side ratio ~ 4
        # mid = (L+R)/2, side = (L-R)/2
        # If L = a*sin, R = b*sin with a,b chosen so mid^2/side^2 ~ 4
        # mid = (a+b)/2, side = (a-b)/2
        # ratio = ((a+b)/(a-b))^2 = 4 => (a+b)/(a-b) = 2 => a+b = 2a-2b => 3b = a
        t = np.linspace(0, 1, 32000, endpoint=False)
        left = (3.0 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        right = (1.0 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        score = StereoQualityScorer._score_mid_side_balance(left, right)
        assert score > 0.8
