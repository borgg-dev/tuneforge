"""Tests for FAD scorer."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tuneforge.scoring.fad_scorer import FADScorer


@pytest.fixture
def reference_stats(tmp_path):
    """Create reference stats file from normalized embeddings (matching real usage)."""
    dim = 512
    rng = np.random.RandomState(42)

    # Generate embeddings and normalize them (mimicking real pipeline)
    raw = rng.randn(200, dim)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normalized = (raw / norms).astype(np.float64)

    mean = np.mean(normalized, axis=0)
    cov = np.cov(normalized, rowvar=False)
    cov += np.eye(dim) * 1e-6

    path = tmp_path / "ref_stats.npz"
    np.savez(path, mean=mean, cov=cov)
    return str(path)


class TestFADScorer:
    def test_no_reference_stats(self):
        """Without reference stats, penalty is always 1.0."""
        fad = FADScorer(window_size=10)
        emb = np.random.randn(512).astype(np.float64)
        emb /= np.linalg.norm(emb)
        for i in range(20):
            fad.update_miner_embedding("miner1", emb + np.random.randn(512) * 0.01)
        assert fad.get_fad_penalty("miner1") == 1.0

    def test_insufficient_embeddings(self, reference_stats):
        """With too few embeddings, penalty is 1.0."""
        fad = FADScorer(window_size=50, reference_stats_path=reference_stats, min_embeddings=10)
        emb = np.random.randn(512).astype(np.float64)
        for i in range(5):
            fad.update_miner_embedding("miner1", emb)
        assert fad.get_fad_penalty("miner1") == 1.0

    def test_penalty_range(self, reference_stats):
        """Penalty should be in [floor, 1.0]."""
        fad = FADScorer(
            window_size=50,
            reference_stats_path=reference_stats,
            min_embeddings=10,
            penalty_floor=0.5,
        )
        rng = np.random.RandomState(42)
        for i in range(20):
            emb = rng.randn(512)
            fad.update_miner_embedding("miner1", emb)

        penalty = fad.get_fad_penalty("miner1")
        assert 0.5 <= penalty <= 1.0

    def test_similar_distribution_low_penalty(self, reference_stats):
        """Embeddings from same distribution should have low FAD / high penalty."""
        # The reference stats were built from normalized random vectors
        # with seed 42. Generate more from the same distribution.
        fad = FADScorer(
            window_size=200,
            reference_stats_path=reference_stats,
            min_embeddings=10,
            penalty_midpoint=15.0,
        )

        # Use the same generation process as the fixture (random + normalize)
        rng = np.random.RandomState(42)
        for i in range(200):
            emb = rng.randn(512)
            # update_miner_embedding will normalize for us
            fad.update_miner_embedding("good_miner", emb)

        penalty = fad.get_fad_penalty("good_miner")
        # Distribution should closely match reference -> low FAD -> high penalty
        assert penalty > 0.8

    def test_dissimilar_distribution_high_penalty(self, reference_stats):
        """Clustered distribution should differ from uniform sphere -> lower penalty."""
        fad = FADScorer(
            window_size=200,
            reference_stats_path=reference_stats,
            min_embeddings=10,
            penalty_midpoint=0.5,  # Low midpoint to amplify small FAD differences
            penalty_steepness=3.0,
        )

        # Create a highly concentrated distribution: all embeddings near
        # a single direction (first basis vector), very different from the
        # uniform-on-sphere reference distribution.
        base = np.zeros(512)
        base[0] = 1.0
        rng = np.random.RandomState(456)
        for i in range(100):
            emb = base + rng.randn(512) * 0.01
            fad.update_miner_embedding("bad_miner", emb)

        penalty = fad.get_fad_penalty("bad_miner")
        # Concentrated distribution vs uniform sphere -> nonzero FAD -> penalty < 1
        assert penalty < 1.0

    def test_sliding_window(self):
        """Verify sliding window drops old embeddings."""
        fad = FADScorer(window_size=5)
        emb = np.random.randn(512)
        for i in range(10):
            fad.update_miner_embedding("miner1", emb)
        assert len(fad._miner_embeddings["miner1"]) == 5

    def test_null_embedding_ignored(self):
        """None embeddings should be silently ignored."""
        fad = FADScorer(window_size=10)
        fad.update_miner_embedding("miner1", None)
        assert len(fad._miner_embeddings["miner1"]) == 0

    def test_zero_embedding_ignored(self):
        """Zero-norm embeddings should be ignored."""
        fad = FADScorer(window_size=10)
        fad.update_miner_embedding("miner1", np.zeros(512))
        assert len(fad._miner_embeddings["miner1"]) == 0

    def test_load_missing_reference(self):
        """Loading non-existent reference stats should not crash."""
        fad = FADScorer(reference_stats_path="/nonexistent/path.npz")
        assert fad._ref_mean is None
        assert fad.get_fad_penalty("any_miner") == 1.0

    def test_frechet_distance_identical(self):
        """FAD between identical distributions should be ~0."""
        mu = np.random.randn(10)
        cov = np.eye(10) * 0.5
        fd = FADScorer._frechet_distance(mu, cov, mu, cov)
        assert fd < 1e-6
