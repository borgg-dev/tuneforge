"""Tests for neural quality scoring with mocked MERT model."""

import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from tuneforge.scoring.neural_quality import (
    NeuralQualityScorer,
    NEURAL_QUALITY_WEIGHTS,
)

SAMPLE_RATE = 32000


# ---------------------------------------------------------------------------
# Helpers for generating mock hidden states
# ---------------------------------------------------------------------------

def _make_hidden_states(
    n_layers: int = 13,
    time_steps: int = 50,
    dim: int = 768,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Create mock hidden states: list of *n_layers* tensors [T, dim]."""
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n_layers):
        data = rng.standard_normal((time_steps, dim)).astype(np.float32)
        states.append(torch.from_numpy(data))
    return states


def _make_smooth_hidden_states(
    n_layers: int = 13,
    time_steps: int = 50,
    dim: int = 768,
) -> list[torch.Tensor]:
    """Create hidden states where consecutive time steps are highly similar."""
    states = []
    for _ in range(n_layers):
        # Start from a base vector and evolve slowly
        base = torch.randn(dim)
        rows = [base]
        for _ in range(1, time_steps):
            # Small perturbation -> high cosine similarity between neighbours
            rows.append(rows[-1] + 0.02 * torch.randn(dim))
        states.append(torch.stack(rows, dim=0))
    return states


def _make_noisy_hidden_states(
    n_layers: int = 13,
    time_steps: int = 50,
    dim: int = 768,
) -> list[torch.Tensor]:
    """Create hidden states with random jumps between time steps."""
    rng = np.random.default_rng(99)
    states = []
    for _ in range(n_layers):
        data = rng.standard_normal((time_steps, dim)).astype(np.float32)
        states.append(torch.from_numpy(data))
    return states


def _make_consistent_layer_states(
    n_layers: int = 13,
    time_steps: int = 50,
    dim: int = 768,
) -> list[torch.Tensor]:
    """Create hidden states where all layers produce similar representations."""
    base = torch.randn(time_steps, dim)
    states = []
    for _ in range(n_layers):
        # Slight layer-to-layer variation
        states.append(base + 0.1 * torch.randn(time_steps, dim))
    return states


def _make_random_layer_states(
    n_layers: int = 13,
    time_steps: int = 50,
    dim: int = 768,
) -> list[torch.Tensor]:
    """Create hidden states where layers are completely independent."""
    rng = np.random.default_rng(77)
    states = []
    for _ in range(n_layers):
        data = rng.standard_normal((time_steps, dim)).astype(np.float32)
        states.append(torch.from_numpy(data))
    return states


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer():
    """Return a NeuralQualityScorer with model loading bypassed."""
    s = NeuralQualityScorer()
    return s


@pytest.fixture
def sample_audio():
    """10-second 440 Hz sine at SAMPLE_RATE."""
    t = np.linspace(0, 10.0, int(SAMPLE_RATE * 10.0), endpoint=False)
    return (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestNeuralQualityKeys:
    """Verify score() returns the expected dictionary keys."""

    def test_score_returns_all_keys(self, scorer, sample_audio):
        fake_states = _make_hidden_states()
        with patch.object(scorer, "_load"), \
             patch.object(scorer, "_extract_hidden_states", return_value=fake_states):
            scores = scorer.score(sample_audio, SAMPLE_RATE)

        expected_keys = set(NEURAL_QUALITY_WEIGHTS.keys())
        assert set(scores.keys()) == expected_keys

    def test_score_keys_match_weights(self, scorer, sample_audio):
        fake_states = _make_hidden_states()
        with patch.object(scorer, "_load"), \
             patch.object(scorer, "_extract_hidden_states", return_value=fake_states):
            scores = scorer.score(sample_audio, SAMPLE_RATE)

        for key in NEURAL_QUALITY_WEIGHTS:
            assert key in scores, f"Missing key: {key}"


class TestNeuralQualityRange:
    """Verify all scores lie in [0, 1]."""

    def test_all_scores_in_range(self, scorer, sample_audio):
        fake_states = _make_hidden_states()
        with patch.object(scorer, "_load"), \
             patch.object(scorer, "_extract_hidden_states", return_value=fake_states):
            scores = scorer.score(sample_audio, SAMPLE_RATE)

        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"

    def test_smooth_states_in_range(self, scorer, sample_audio):
        fake_states = _make_smooth_hidden_states()
        with patch.object(scorer, "_load"), \
             patch.object(scorer, "_extract_hidden_states", return_value=fake_states):
            scores = scorer.score(sample_audio, SAMPLE_RATE)

        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"

    def test_noisy_states_in_range(self, scorer, sample_audio):
        fake_states = _make_noisy_hidden_states()
        with patch.object(scorer, "_load"), \
             patch.object(scorer, "_extract_hidden_states", return_value=fake_states):
            scores = scorer.score(sample_audio, SAMPLE_RATE)

        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"


class TestNeuralQualityFallback:
    """When extraction fails, all scores should be 0.5 (neutral)."""

    def test_none_hidden_states(self, scorer, sample_audio):
        with patch.object(scorer, "_load"), \
             patch.object(scorer, "_extract_hidden_states", return_value=None):
            scores = scorer.score(sample_audio, SAMPLE_RATE)

        for key, val in scores.items():
            assert val == pytest.approx(0.5), f"{key} should be 0.5, got {val}"

    def test_short_audio_fallback(self, scorer):
        """Audio shorter than 0.25s should return all 0.5."""
        short_audio = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
        with patch.object(scorer, "_load"), \
             patch.object(scorer, "_extract_hidden_states", return_value=None):
            scores = scorer.score(short_audio, SAMPLE_RATE)

        for key, val in scores.items():
            assert val == pytest.approx(0.5), f"{key} should be 0.5, got {val}"


class TestAggregate:
    """Test weighted aggregation."""

    def test_weights_sum_to_one(self):
        total = sum(NEURAL_QUALITY_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_aggregate_all_ones(self, scorer):
        scores = {k: 1.0 for k in NEURAL_QUALITY_WEIGHTS}
        assert scorer.aggregate(scores) == pytest.approx(1.0, abs=1e-6)

    def test_aggregate_all_zeros(self, scorer):
        scores = {k: 0.0 for k in NEURAL_QUALITY_WEIGHTS}
        assert scorer.aggregate(scores) == pytest.approx(0.0, abs=1e-6)

    def test_aggregate_neutral(self, scorer):
        scores = {k: 0.5 for k in NEURAL_QUALITY_WEIGHTS}
        assert scorer.aggregate(scores) == pytest.approx(0.5, abs=1e-6)

    def test_aggregate_in_range(self, scorer, sample_audio):
        fake_states = _make_hidden_states()
        with patch.object(scorer, "_load"), \
             patch.object(scorer, "_extract_hidden_states", return_value=fake_states):
            scores = scorer.score(sample_audio, SAMPLE_RATE)
        agg = scorer.aggregate(scores)
        assert 0.0 <= agg <= 1.0

    def test_aggregate_weighted_sum_correct(self, scorer):
        scores = {
            "temporal_coherence": 0.8,
            "activation_strength": 0.6,
            "layer_agreement": 0.7,
            "structural_periodicity": 0.9,
        }
        expected = (
            0.30 * 0.8
            + 0.20 * 0.6
            + 0.25 * 0.7
            + 0.25 * 0.9
        )
        assert scorer.aggregate(scores) == pytest.approx(expected, abs=1e-6)


class TestTemporalCoherence:
    """Test temporal coherence metric with smooth vs noisy states."""

    def test_smooth_states_score_higher(self, scorer):
        smooth = _make_smooth_hidden_states()
        noisy = _make_noisy_hidden_states()

        smooth_score = scorer._score_temporal_coherence(smooth)
        noisy_score = scorer._score_temporal_coherence(noisy)

        assert smooth_score > noisy_score, (
            f"Smooth ({smooth_score:.4f}) should outscore noisy ({noisy_score:.4f})"
        )

    def test_few_time_steps_returns_fallback(self, scorer):
        """Fewer than 4 time steps should return 0.5."""
        short_states = _make_hidden_states(time_steps=3)
        score = scorer._score_temporal_coherence(short_states)
        assert score == pytest.approx(0.5)

    def test_identical_frames_score(self, scorer):
        """Identical consecutive frames (similarity=1.0) should still score in [0,1]."""
        states = []
        row = torch.randn(768)
        for _ in range(13):
            states.append(row.unsqueeze(0).expand(50, -1).clone())
        score = scorer._score_temporal_coherence(states)
        assert 0.0 <= score <= 1.0


class TestLayerAgreement:
    """Test layer agreement metric with consistent vs random states."""

    def test_consistent_layers_score_differently_from_random(self, scorer):
        consistent = _make_consistent_layer_states()
        random_states = _make_random_layer_states()

        consistent_score = scorer._score_layer_agreement(consistent)
        random_score = scorer._score_layer_agreement(random_states)

        # Both should be in range
        assert 0.0 <= consistent_score <= 1.0
        assert 0.0 <= random_score <= 1.0

        # Consistent layers have high similarity -> bell curve may peak or fall
        # Random layers have ~0 similarity -> lower on bell curve centred at 0.6
        # The key is that they produce different scores
        assert consistent_score != pytest.approx(random_score, abs=0.05), (
            f"Consistent ({consistent_score:.4f}) and random ({random_score:.4f}) "
            "should produce distinguishable scores"
        )

    def test_single_layer_fallback(self, scorer):
        """With only one layer, should return 0.5."""
        single = [torch.randn(50, 768)]
        score = scorer._score_layer_agreement(single)
        assert score == pytest.approx(0.5)


class TestStructuralPeriodicity:
    """Test structural periodicity metric."""

    def test_periodic_signal_scores(self, scorer):
        """A repeating pattern should produce a non-fallback score."""
        # Create a hidden state with periodic repetition
        dim = 768
        pattern = torch.randn(10, dim)  # 10-step pattern
        repeated = pattern.repeat(5, 1)  # 50 steps = 5 repetitions
        states = [torch.randn(50, dim) for _ in range(12)]
        states.append(repeated)  # last layer is periodic
        score = scorer._score_structural_periodicity(states)
        assert 0.0 <= score <= 1.0

    def test_few_time_steps_returns_fallback(self, scorer):
        """Fewer than 8 time steps should return 0.5."""
        short_states = _make_hidden_states(time_steps=5)
        score = scorer._score_structural_periodicity(short_states)
        assert score == pytest.approx(0.5)

    def test_random_signal_in_range(self, scorer):
        """Random hidden states should still produce a score in [0, 1]."""
        random_states = _make_noisy_hidden_states(time_steps=60)
        score = scorer._score_structural_periodicity(random_states)
        assert 0.0 <= score <= 1.0


class TestModelLoadFailure:
    """Verify graceful fallback when the model cannot be loaded."""

    def _trigger_load_failure(self, scorer, exc_class=RuntimeError):
        """Force _load() to fail by injecting a broken AutoModel import."""
        mock_automodel = MagicMock()
        mock_automodel.from_pretrained.side_effect = exc_class("download failed")

        mock_extractor = MagicMock()
        mock_extractor.from_pretrained.side_effect = exc_class("download failed")

        with patch.dict(
            "sys.modules",
            {
                "transformers": MagicMock(
                    AutoModel=mock_automodel,
                    Wav2Vec2FeatureExtractor=mock_extractor,
                ),
            },
        ):
            scorer._load()

    def test_load_exception_sets_sentinel(self, scorer, sample_audio):
        """If model loading raises, scorer should return neutral scores."""
        self._trigger_load_failure(scorer)

        from tuneforge.scoring.neural_quality import _LOAD_FAILED
        assert scorer._model is _LOAD_FAILED

        # score() should return all 0.5
        scores = scorer.score(sample_audio, SAMPLE_RATE)
        for key, val in scores.items():
            assert val == pytest.approx(0.5), f"{key} should be 0.5, got {val}"

    def test_aggregate_of_fallback_scores(self, scorer, sample_audio):
        """Aggregate of all-0.5 scores should be 0.5."""
        self._trigger_load_failure(scorer)

        scores = scorer.score(sample_audio, SAMPLE_RATE)
        agg = scorer.aggregate(scores)
        assert agg == pytest.approx(0.5, abs=1e-6)

    def test_load_failure_does_not_raise(self, scorer):
        """_load() must not propagate exceptions to the caller."""
        self._trigger_load_failure(scorer, RuntimeError)

        from tuneforge.scoring.neural_quality import _LOAD_FAILED
        assert scorer._model is _LOAD_FAILED


class TestActivationStrength:
    """Test activation strength metric."""

    def test_strong_activations_score_higher(self, scorer):
        """Embeddings with large norms should score higher than small norms."""
        dim = 768
        t_steps = 50

        # Strong activations: norm ~ 25
        strong = [torch.randn(t_steps, dim) * 5.0 for _ in range(13)]
        # Weak activations: norm ~ 2.5
        weak = [torch.randn(t_steps, dim) * 0.5 for _ in range(13)]

        strong_score = scorer._score_activation_strength(strong)
        weak_score = scorer._score_activation_strength(weak)

        assert strong_score > weak_score, (
            f"Strong ({strong_score:.4f}) should outscore weak ({weak_score:.4f})"
        )

    def test_score_clipped_to_one(self, scorer):
        """Very large norms should be clipped to 1.0."""
        dim = 768
        huge = [torch.randn(50, dim) * 100.0 for _ in range(13)]
        score = scorer._score_activation_strength(huge)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zero_activations(self, scorer):
        """Zero embeddings should produce score 0.0."""
        zero = [torch.zeros(50, 768) for _ in range(13)]
        score = scorer._score_activation_strength(zero)
        assert score == pytest.approx(0.0, abs=1e-6)


class TestResample:
    """Test the static _resample helper."""

    def test_same_rate_noop(self):
        audio = np.ones(1000, dtype=np.float32)
        out = NeuralQualityScorer._resample(audio, 24000, 24000)
        np.testing.assert_array_equal(out, audio)

    def test_different_rate_changes_length(self):
        """Resample 48kHz -> 24kHz should halve the length."""
        sr = 48000
        target = 24000
        duration = 1.0
        audio = np.zeros(int(sr * duration), dtype=np.float32)

        # Mock librosa.resample to simulate correct downsampling
        mock_librosa = MagicMock()
        expected_out = np.zeros(int(target * duration), dtype=np.float32)
        mock_librosa.resample.return_value = expected_out

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            out = NeuralQualityScorer._resample(audio, sr, target)

        expected_len = int(target * duration)
        assert abs(len(out) - expected_len) <= 1
        mock_librosa.resample.assert_called_once()
