"""Tests for CLAP scorer with mocked model."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import torch


class TestCLAPScorerMocked:
    """Test CLAP scoring pipeline with mocked model (no real download)."""

    def _make_mock_clap(self):
        """Create a fully mocked CLAPScorer."""
        from tuneforge.scoring.clap_scorer import CLAPScorer
        scorer = CLAPScorer.__new__(CLAPScorer)
        scorer._model_name = "laion/larger_clap_music"

        # Mock model
        mock_model = MagicMock()
        mock_param = torch.zeros(1)
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.eval.return_value = None

        # Text features: random 512-dim vector
        text_emb = torch.randn(1, 512)
        text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
        mock_model.get_text_features.return_value = text_emb

        # Audio features: similar vector (high similarity)
        audio_emb = text_emb + 0.1 * torch.randn(1, 512)
        audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)
        mock_model.get_audio_features.return_value = audio_emb

        scorer._model = mock_model

        # Mock processor
        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
        scorer._processor = mock_processor

        return scorer

    def test_score_returns_float(self, sample_audio_sine, sample_rate):
        scorer = self._make_mock_clap()
        score = scorer.score(sample_audio_sine, sample_rate, "test prompt")
        assert isinstance(score, float)

    def test_score_in_range(self, sample_audio_sine, sample_rate):
        scorer = self._make_mock_clap()
        score = scorer.score(sample_audio_sine, sample_rate, "upbeat rock song")
        assert 0.0 <= score <= 1.0

    def test_get_audio_embedding(self, sample_audio_sine, sample_rate):
        scorer = self._make_mock_clap()
        emb = scorer.get_audio_embedding(sample_audio_sine, sample_rate)
        assert emb is not None
        assert len(emb) == 512

    def test_score_with_different_prompts(self, sample_audio_sine, sample_rate):
        scorer = self._make_mock_clap()
        s1 = scorer.score(sample_audio_sine, sample_rate, "happy pop song")
        s2 = scorer.score(sample_audio_sine, sample_rate, "dark ambient drone")
        # Both should be valid scores (the mock returns similar embeddings)
        assert 0.0 <= s1 <= 1.0
        assert 0.0 <= s2 <= 1.0
