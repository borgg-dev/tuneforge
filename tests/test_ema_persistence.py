"""Tests for EMA persistence, new miner seed, and scorer dropout."""

import json

import numpy as np
import pytest

from tuneforge.rewards.leaderboard import MinerLeaderboard, EMA_STATE_VERSION


class TestEMAPersistence:
    """Tests for EMA state save/load round-trip."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Create leaderboard, update miners, save, load into new leaderboard, verify."""
        path = str(tmp_path / "ema_state.json")

        lb1 = MinerLeaderboard(alpha=0.2, steepen_baseline=0.5, steepen_power=2.0)
        lb1.update(0, 0.8)
        lb1.update(1, 0.6)
        lb1.update(0, 0.9)  # second update for UID 0
        lb1.save_state(path)

        lb2 = MinerLeaderboard(alpha=0.2, steepen_baseline=0.5, steepen_power=2.0)
        assert lb2.load_state(path) is True

        # EMA values should match
        assert abs(lb2.get_ema(0) - lb1.get_ema(0)) < 1e-6
        assert abs(lb2.get_ema(1) - lb1.get_ema(1)) < 1e-6

        # Weights should match
        assert abs(lb2.get_weight(0) - lb1.get_weight(0)) < 1e-6
        assert abs(lb2.get_weight(1) - lb1.get_weight(1)) < 1e-6

    def test_load_missing_file(self, tmp_path):
        """Loading from a non-existent file returns False gracefully."""
        path = str(tmp_path / "nonexistent.json")
        lb = MinerLeaderboard()
        assert lb.load_state(path) is False

    def test_load_corrupt_file(self, tmp_path):
        """Corrupt primary file falls back to .bak."""
        path = str(tmp_path / "ema_state.json")
        bak_path = str(tmp_path / "ema_state.json.bak")

        # Create a valid backup
        lb1 = MinerLeaderboard()
        lb1.update(0, 0.7)
        lb1.save_state(path)

        # Save again to create .bak (save_state creates .bak from existing file)
        lb1.update(1, 0.5)
        lb1.save_state(path)

        # Now corrupt the primary file
        with open(path, "w") as f:
            f.write("not valid json {{{{")

        # Load should fall back to .bak
        lb2 = MinerLeaderboard()
        assert lb2.load_state(path) is True
        # The backup should have UID 0 but not UID 1
        assert lb2.get_ema(0) > 0

    def test_load_validates_values(self, tmp_path):
        """Out-of-range EMA values are skipped during load."""
        path = str(tmp_path / "ema_state.json")
        data = {
            "version": EMA_STATE_VERSION,
            "alpha": 0.2,
            "baseline": 0.5,
            "power": 2.0,
            "ema": {
                "0": 0.8,      # valid
                "1": 1.5,      # out of range (>1)
                "2": -0.1,     # out of range (<0)
                "-3": 0.5,     # negative UID
                "4": 0.6,      # valid
            },
            "rounds": {"0": 5, "4": 3},
        }
        with open(path, "w") as f:
            json.dump(data, f)

        lb = MinerLeaderboard()
        assert lb.load_state(path) is True

        # Only valid entries should be loaded
        assert abs(lb.get_ema(0) - 0.8) < 1e-6
        assert abs(lb.get_ema(4) - 0.6) < 1e-6
        # Invalid entries should not be present
        assert lb.get_ema(1) == 0.0  # default for missing UID
        assert lb.get_ema(2) == 0.0
        assert lb.get_ema(3) == 0.0  # UID -3 was rejected

    def test_backup_created(self, tmp_path):
        """Saving twice creates a .bak file."""
        path = str(tmp_path / "ema_state.json")
        bak_path = tmp_path / "ema_state.json.bak"

        lb = MinerLeaderboard()
        lb.update(0, 0.5)
        lb.save_state(path)

        assert not bak_path.exists()

        # Save again — the first save should become the backup
        lb.update(0, 0.6)
        lb.save_state(path)

        assert bak_path.exists()

        # Backup should contain valid JSON
        with open(bak_path) as f:
            data = json.load(f)
        assert data["version"] == EMA_STATE_VERSION

    def test_new_miner_seed(self):
        """New miners start at EMA_NEW_MINER_SEED (0.25) instead of 0.0."""
        lb = MinerLeaderboard(alpha=0.2)

        # Before any update, EMA should be 0.0 (default for get_ema on missing UID)
        assert lb.get_ema(99) == 0.0

        # After first update with score 0.0, EMA should reflect the seed
        # EMA = alpha * 0.0 + (1 - alpha) * seed = 0.8 * 0.25 = 0.20
        lb.update(99, 0.0)
        expected = 0.2 * 0.0 + 0.8 * 0.25  # 0.20
        assert abs(lb.get_ema(99) - expected) < 1e-6

    def test_scorer_dropout(self):
        """Verify _perturb_weights can zero some scorers via dropout."""
        from tuneforge.rewards.reward import ProductionRewardModel

        # Run many perturbations and check that at least one key gets zeroed
        found_dropout = False
        for i in range(100):
            challenge_id = f"test-dropout-{i}"
            weights = ProductionRewardModel._perturb_weights(challenge_id)

            # Count how many scorers are zero that were non-zero in base weights
            from tuneforge.config.scoring_config import SCORING_WEIGHTS
            dropped = [
                k for k in SCORING_WEIGHTS
                if SCORING_WEIGHTS[k] > 0 and weights.get(k, 0) == 0.0
            ]
            if dropped:
                found_dropout = True
                break

        assert found_dropout, "Expected at least one scorer to be dropped across 100 trials"

        # Weights should still sum to 1.0 (renormalized)
        total = sum(weights.values())
        if total > 0:
            assert abs(total - 1.0) < 1e-6

    def test_version_mismatch_rejected(self, tmp_path):
        """State file with wrong version is rejected."""
        path = str(tmp_path / "ema_state.json")
        data = {
            "version": 999,
            "alpha": 0.2,
            "baseline": 0.5,
            "power": 2.0,
            "ema": {"0": 0.5},
            "rounds": {"0": 1},
        }
        with open(path, "w") as f:
            json.dump(data, f)

        lb = MinerLeaderboard()
        assert lb.load_state(path) is False

    def test_thread_safety(self):
        """Basic thread safety: concurrent updates don't crash."""
        import threading

        lb = MinerLeaderboard()
        errors = []

        def worker(start_uid):
            try:
                for i in range(50):
                    lb.update(start_uid + (i % 10), float(i % 10) / 10.0)
                    lb.get_ema(start_uid)
                    lb.get_weight(start_uid)
                    lb.get_all_weights()
                    lb.snapshot()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i * 100,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
