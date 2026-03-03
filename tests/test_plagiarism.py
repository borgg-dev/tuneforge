"""Tests for fingerprint-based plagiarism detection."""

import sqlite3

import numpy as np
import pytest
from unittest.mock import patch


class TestPlagiarismDetector:
    """Test plagiarism detection via Chromaprint fingerprinting."""

    def _make_detector(self, tmp_path):
        """Create PlagiarismDetector with a temp DB."""
        from tuneforge.scoring.plagiarism import PlagiarismDetector

        db_path = str(tmp_path / "test_fp.db")

        with patch.object(PlagiarismDetector, '__init__', lambda self, *a, **kw: None):
            detector = PlagiarismDetector.__new__(PlagiarismDetector)

        detector._db_path = db_path
        detector._round_fingerprints = {}
        detector._store_count = 0
        detector._fpcalc_available = None

        # Init DB
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                miner_hotkey TEXT NOT NULL,
                challenge_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fp_fingerprint ON fingerprints(fingerprint)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fp_miner ON fingerprints(miner_hotkey)")
        conn.commit()
        conn.close()

        return detector

    def test_no_fpcalc_returns_not_plagiarized(self, tmp_path, sample_audio_sine, sample_rate):
        """When fpcalc is not available, nothing is flagged."""
        detector = self._make_detector(tmp_path)
        detector._fpcalc_available = False

        is_plag, sim = detector.check(sample_audio_sine, sample_rate, "miner_a", "challenge_1")
        assert not is_plag
        assert sim == 0.0

    def test_unique_fingerprints_not_plagiarized(self, tmp_path, sample_audio_sine, sample_rate):
        """Different fingerprints from different miners are not flagged."""
        detector = self._make_detector(tmp_path)

        # Mock _get_fingerprint to return unique fingerprints
        call_count = [0]
        def unique_fp(*args, **kwargs):
            call_count[0] += 1
            return f"fingerprint_{call_count[0]}"

        detector._get_fingerprint = unique_fp

        is_plag1, _ = detector.check(sample_audio_sine, sample_rate, "miner_a", "c1")
        is_plag2, sim = detector.check(sample_audio_sine, sample_rate, "miner_b", "c1")

        assert not is_plag1
        assert not is_plag2
        assert sim == 0.0

    def test_cross_miner_exact_replay_detected(self, tmp_path, sample_audio_sine, sample_rate):
        """Exact same fingerprint from different miners is flagged."""
        detector = self._make_detector(tmp_path)

        # Both miners produce the same fingerprint (exact audio replay)
        detector._get_fingerprint = lambda *a, **kw: "identical_fingerprint_123"

        is_plag1, _ = detector.check(sample_audio_sine, sample_rate, "miner_a", "c1")
        is_plag2, sim = detector.check(sample_audio_sine, sample_rate, "miner_b", "c1")

        assert not is_plag1  # first miner is fine
        assert is_plag2  # second miner caught
        assert sim == 1.0

    def test_self_plagiarism_detected(self, tmp_path, sample_audio_sine, sample_rate):
        """Same miner resubmitting same fingerprint across challenges is flagged."""
        detector = self._make_detector(tmp_path)
        detector._get_fingerprint = lambda *a, **kw: "reused_fingerprint_456"

        # First submission — fine
        is_plag1, _ = detector.check(sample_audio_sine, sample_rate, "miner_a", "c1")
        assert not is_plag1

        # Clear round cache (new round)
        detector.clear_round_cache()

        # Second submission with same fingerprint — self-plagiarism
        is_plag2, sim = detector.check(sample_audio_sine, sample_rate, "miner_a", "c2")
        assert is_plag2
        assert sim == 1.0

    def test_round_cache_clear(self, tmp_path, sample_audio_sine, sample_rate):
        """clear_round_cache resets per-round fingerprint cache."""
        detector = self._make_detector(tmp_path)
        detector._get_fingerprint = lambda *a, **kw: "fp_abc"

        detector.check(sample_audio_sine, sample_rate, "miner_a", "c1")
        assert len(detector._round_fingerprints) > 0

        detector.clear_round_cache()
        assert len(detector._round_fingerprints) == 0

    def test_same_miner_same_prompt_not_flagged_in_round(self, tmp_path, sample_audio_sine, sample_rate):
        """Same miner queried twice in the same round is not flagged by cross-miner check."""
        detector = self._make_detector(tmp_path)

        call_count = [0]
        def unique_fp(*args, **kwargs):
            call_count[0] += 1
            return f"fp_{call_count[0]}"

        detector._get_fingerprint = unique_fp

        # Same miner, different fingerprints (different generations)
        is_plag1, _ = detector.check(sample_audio_sine, sample_rate, "miner_a", "c1")
        is_plag2, _ = detector.check(sample_audio_sine, sample_rate, "miner_a", "c2")

        assert not is_plag1
        assert not is_plag2

    def test_no_clap_dependency(self, tmp_path):
        """PlagiarismDetector no longer depends on CLAPScorer."""
        detector = self._make_detector(tmp_path)
        assert not hasattr(detector, '_clap')
        assert not hasattr(detector, '_round_embeddings')
