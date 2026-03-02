"""Tests for plagiarism detection with mocked CLAP."""

import tempfile

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestPlagiarismDetector:
    """Test plagiarism detection with mock CLAP embeddings."""

    def _make_detector(self, tmp_path):
        """Create PlagiarismDetector with mocked CLAP scorer."""
        from tuneforge.scoring.plagiarism import PlagiarismDetector

        db_path = str(tmp_path / "test_fp.db")

        with patch.object(PlagiarismDetector, '__init__', lambda self, *a, **kw: None):
            detector = PlagiarismDetector.__new__(PlagiarismDetector)

        detector._db_path = db_path
        detector._round_embeddings = {}

        # Mock CLAP scorer
        mock_clap = MagicMock()
        mock_clap.get_audio_embedding.return_value = np.random.randn(512).astype(np.float32)
        detector._clap = mock_clap

        # Init DB
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                miner_hotkey TEXT NOT NULL,
                challenge_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fp_fingerprint ON fingerprints(fingerprint)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fp_miner ON fingerprints(miner_hotkey)")
        conn.commit()
        conn.close()

        return detector

    def test_unique_audio_not_plagiarized(self, tmp_path, sample_audio_sine, sample_rate):
        detector = self._make_detector(tmp_path)
        is_plag, sim = detector.check(sample_audio_sine, sample_rate, "miner_a", "challenge_1")
        assert not is_plag
        assert 0.0 <= sim <= 1.0

    def test_embedding_similarity_detection(self, tmp_path, sample_audio_sine, sample_rate):
        detector = self._make_detector(tmp_path)

        # Same embedding for both miners → high similarity
        fixed_emb = np.ones(512, dtype=np.float32)
        detector._clap.get_audio_embedding.return_value = fixed_emb

        is_plag1, _ = detector.check(sample_audio_sine, sample_rate, "miner_a", "challenge_1")
        is_plag2, sim = detector.check(sample_audio_sine, sample_rate, "miner_b", "challenge_1")

        # miner_b's embedding should match miner_a's in the round cache
        assert sim >= 0.95
        assert is_plag2

    def test_round_cache_clear(self, tmp_path, sample_audio_sine, sample_rate):
        detector = self._make_detector(tmp_path)

        fixed_emb = np.ones(512, dtype=np.float32)
        detector._clap.get_audio_embedding.return_value = fixed_emb

        detector.check(sample_audio_sine, sample_rate, "miner_a", "challenge_1")
        assert len(detector._round_embeddings) > 0

        detector.clear_round_cache()
        assert len(detector._round_embeddings) == 0

    def test_different_miners_different_embeddings(self, tmp_path, sample_audio_sine, sample_rate):
        detector = self._make_detector(tmp_path)

        # Different embeddings for each call
        call_count = [0]
        def unique_embedding(*args, **kwargs):
            call_count[0] += 1
            rng = np.random.default_rng(call_count[0])
            return rng.standard_normal(512).astype(np.float32)

        detector._clap.get_audio_embedding.side_effect = unique_embedding

        is_plag1, _ = detector.check(sample_audio_sine, sample_rate, "miner_a", "c1")
        is_plag2, sim = detector.check(sample_audio_sine, sample_rate, "miner_b", "c1")

        assert not is_plag1
        # With random embeddings, similarity should be low
        assert sim < 0.95
