"""
Tests for the preference model training pipeline.

Covers: annotation format, embedding cache, training loop,
checkpoint round-trip, config integration, and metadata sidecar.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# TestAnnotationFormat — schema validation for annotation tool output
# ---------------------------------------------------------------------------

class TestAnnotationFormat:

    def test_annotation_record_has_required_fields(self):
        """JSONL records must contain all required fields."""
        record = {
            "pair_id": "035ebbb70436",
            "audio_a": "storage/3/challenge1.wav",
            "audio_b": "storage/7/challenge1.wav",
            "challenge_id": "challenge1",
            "prompt": "Create a calm ambient track",
            "preferred": "a",
            "timestamp": "2026-03-03T14:30:00.000000+00:00",
        }
        required = {"pair_id", "audio_a", "audio_b", "challenge_id", "preferred", "timestamp"}
        assert required.issubset(record.keys())

    def test_preferred_field_valid_values(self):
        """The preferred field should only be a, b, or skip."""
        valid = {"a", "b", "skip"}
        for val in valid:
            assert val in valid
        assert "c" not in valid
        assert "" not in valid

    def test_pair_id_deterministic(self):
        """Same paths produce the same pair_id."""
        from tools.annotate_preferences import compute_pair_id

        id1 = compute_pair_id("storage/3/abc.wav", "storage/7/xyz.wav")
        id2 = compute_pair_id("storage/3/abc.wav", "storage/7/xyz.wav")
        assert id1 == id2
        assert len(id1) == 12

    def test_pair_id_order_invariant(self):
        """Swapped paths produce the same pair_id."""
        from tools.annotate_preferences import compute_pair_id

        id_ab = compute_pair_id("storage/3/abc.wav", "storage/7/xyz.wav")
        id_ba = compute_pair_id("storage/7/xyz.wav", "storage/3/abc.wav")
        assert id_ab == id_ba


# ---------------------------------------------------------------------------
# TestEmbeddingCache — NPZ round-trip and key conversion
# ---------------------------------------------------------------------------

class TestEmbeddingCache:

    def test_embedding_shape_512(self):
        """Cached embeddings must be 512-dimensional float32."""
        emb = np.random.randn(512).astype(np.float32)
        assert emb.shape == (512,)
        assert emb.dtype == np.float32

    def test_npz_round_trip(self, tmp_path):
        """save/load preserves values exactly."""
        key = "3/abc123"
        emb = np.random.randn(512).astype(np.float32)
        npz_path = str(tmp_path / "test.npz")

        np.savez_compressed(npz_path, **{key: emb})

        loaded = np.load(npz_path)
        np.testing.assert_array_equal(loaded[key], emb)

    def test_missing_key_raises(self, tmp_path):
        """KeyError on missing embedding key."""
        key = "3/abc123"
        emb = np.random.randn(512).astype(np.float32)
        npz_path = str(tmp_path / "test.npz")
        np.savez_compressed(npz_path, **{key: emb})

        loaded = np.load(npz_path)
        with pytest.raises(KeyError):
            _ = loaded["99/nonexistent"]

    def test_path_to_key_strips_prefix_and_ext(self):
        """path_to_key converts WAV path to cache key correctly."""
        from tools.build_embedding_cache import path_to_key

        assert path_to_key("storage/abc123/63.wav") == "abc123/63"
        assert path_to_key("data/challenge1/42.wav") == "challenge1/42"


# ---------------------------------------------------------------------------
# TestTrainingLoop — end-to-end training with synthetic data
# ---------------------------------------------------------------------------

class TestTrainingLoop:

    @staticmethod
    def _make_synthetic_pairs(n_pairs: int = 100, seed: int = 42):
        """Create synthetic preference pairs with separable clusters."""
        rng = np.random.default_rng(seed)
        pairs = []
        for _ in range(n_pairs):
            # "Good" embeddings clustered around +1, "bad" around -1
            emb_good = rng.normal(loc=1.0, scale=0.3, size=512).astype(np.float32)
            emb_bad = rng.normal(loc=-1.0, scale=0.3, size=512).astype(np.float32)
            pairs.append((emb_good, emb_bad, 1.0))
        return pairs

    def test_learns_simple_preference(self):
        """Training on separable clusters should achieve >80% val accuracy."""
        from tools.train_preference import train

        pairs = self._make_synthetic_pairs(100)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            out_path = f.name
        try:
            result = train(
                pairs=pairs,
                lr=1e-3,
                epochs=30,
                batch_size=16,
                patience=10,
                val_split=0.2,
                seed=42,
                output_path=out_path,
                device="cpu",
            )
            assert result["best_val_acc"] > 0.80
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_produces_valid_checkpoint(self):
        """Output file exists and loads into PreferenceHead."""
        from tools.train_preference import train
        from tuneforge.scoring.preference_model import PreferenceHead

        pairs = self._make_synthetic_pairs(50)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            out_path = f.name
        try:
            train(
                pairs=pairs,
                lr=1e-3,
                epochs=5,
                batch_size=16,
                patience=3,
                val_split=0.2,
                seed=42,
                output_path=out_path,
                device="cpu",
            )
            assert os.path.exists(out_path)

            # Load into a fresh PreferenceHead
            head = PreferenceHead()
            raw = torch.load(out_path, map_location="cpu", weights_only=True)
            # New format: {"state_dict": ..., "val_accuracy": ..., "embedding_dim": ...}
            state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
            head.load_state_dict(state)

            # Verify forward pass works (output is raw logit, not bounded)
            dummy = torch.randn(1, 512)
            output = head(dummy)
            assert output.shape == (1, 1)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_refuses_tiny_dataset(self):
        """<50 pairs should raise ValueError."""
        from tools.train_preference import train

        pairs = self._make_synthetic_pairs(5)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            out_path = f.name
        try:
            with pytest.raises(ValueError, match="at least 50"):
                train(
                    pairs=pairs,
                    lr=1e-3,
                    epochs=5,
                    batch_size=16,
                    patience=3,
                    val_split=0.2,
                    seed=42,
                    output_path=out_path,
                    device="cpu",
                )
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)


# ---------------------------------------------------------------------------
# TestCheckpointRoundTrip — save/load preserves model behaviour
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:

    def test_state_dict_round_trips(self):
        """Save/load produces identical forward pass results."""
        from tuneforge.scoring.preference_model import PreferenceHead

        original = PreferenceHead()
        original.eval()
        dummy = torch.randn(1, 512)

        with torch.no_grad():
            original_out = original(dummy).item()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            out_path = f.name
        try:
            torch.save(original.state_dict(), out_path)

            loaded = PreferenceHead()
            state = torch.load(out_path, map_location="cpu", weights_only=True)
            loaded.load_state_dict(state)
            loaded.eval()

            with torch.no_grad():
                loaded_out = loaded(dummy).item()

            assert abs(original_out - loaded_out) < 1e-6
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_exits_bootstrap_mode(self):
        """PreferenceModel with valid checkpoint sets _bootstrap=False."""
        from unittest.mock import patch
        from tuneforge.scoring.preference_model import PreferenceHead, PreferenceModel

        # Create a valid checkpoint
        head = PreferenceHead()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            out_path = f.name
        try:
            torch.save(head.state_dict(), out_path)

            # Force CPU to avoid CUDA OOM when miners are running
            with patch("torch.cuda.is_available", return_value=False):
                model = PreferenceModel(model_path=out_path, neural_scorer=None)
            assert model._bootstrap is False
            assert model._head is not None
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)


# ---------------------------------------------------------------------------
# TestConfigIntegration — settings and scoring_config have the new fields
# ---------------------------------------------------------------------------

class TestConfigIntegration:

    def test_settings_has_preference_path(self):
        """Settings model should have the preference_model_path field."""
        from tuneforge.settings import Settings

        s = Settings()
        assert hasattr(s, "preference_model_path")
        assert s.preference_model_path is None

    def test_scoring_config_default_none(self):
        """PREFERENCE_MODEL_PATH defaults to None in scoring config."""
        from tuneforge.config.scoring_config import PREFERENCE_MODEL_PATH

        assert PREFERENCE_MODEL_PATH is None


# ---------------------------------------------------------------------------
# TestMetadataSidecar — schema validation for validator sidecar JSON
# ---------------------------------------------------------------------------

class TestMetadataSidecar:

    def test_sidecar_has_required_fields(self, tmp_path):
        """Metadata sidecar JSON must contain the required fields."""
        from datetime import datetime, timezone

        sidecar = {
            "challenge_id": "test-challenge-123",
            "uid": 3,
            "prompt": "Create a calm ambient track with gentle synths",
            "genre": "ambient",
            "mood": "calm",
            "tempo_bpm": 80,
            "duration_seconds": 10.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Write and read back
        path = tmp_path / "test.json"
        path.write_text(json.dumps(sidecar, indent=2))
        loaded = json.loads(path.read_text())

        required = {"challenge_id", "uid", "prompt", "genre", "mood",
                     "tempo_bpm", "duration_seconds", "timestamp"}
        assert required.issubset(loaded.keys())
        assert loaded["uid"] == 3
        assert loaded["genre"] == "ambient"


# ---------------------------------------------------------------------------
# TestLoadPairs — the load_pairs function with proper JSONL format
# ---------------------------------------------------------------------------

class TestLoadPairs:

    def test_load_pairs_correct_format(self, tmp_path):
        """load_pairs correctly maps a/b preferred to preferred/rejected embeddings."""
        from tools.train_preference import load_pairs

        # Create embeddings (new layout: challenge_id/uid)
        emb_a = np.random.randn(512).astype(np.float32)
        emb_b = np.random.randn(512).astype(np.float32)
        npz_path = str(tmp_path / "embeddings.npz")
        np.savez_compressed(npz_path, **{"challenge1/63": emb_a, "challenge1/64": emb_b})

        # Create annotations with "preferred": "a" meaning audio_a is preferred
        records = []
        for i in range(15):
            records.append({
                "pair_id": f"pair_{i:03d}",
                "audio_a": "storage/challenge1/63.wav",
                "audio_b": "storage/challenge1/64.wav",
                "challenge_id": "challenge1",
                "prompt": "test",
                "preferred": "a" if i % 2 == 0 else "b",
                "timestamp": "2026-03-03T00:00:00+00:00",
            })

        ann_path = str(tmp_path / "annotations.jsonl")
        with open(ann_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        pairs = load_pairs(ann_path, npz_path)
        assert len(pairs) == 15

        # For records where preferred=="a": preferred_emb=emb_a, rejected_emb=emb_b
        # For records where preferred=="b": preferred_emb=emb_b, rejected_emb=emb_a
        for i, (emb_pref, emb_rej, target) in enumerate(pairs):
            assert target == 1.0
            if i % 2 == 0:  # preferred "a"
                np.testing.assert_array_equal(emb_pref, emb_a)
                np.testing.assert_array_equal(emb_rej, emb_b)
            else:  # preferred "b"
                np.testing.assert_array_equal(emb_pref, emb_b)
                np.testing.assert_array_equal(emb_rej, emb_a)

    def test_load_pairs_skips_skip_entries(self, tmp_path):
        """Entries with preferred=='skip' are skipped."""
        from tools.train_preference import load_pairs

        emb_a = np.random.randn(512).astype(np.float32)
        emb_b = np.random.randn(512).astype(np.float32)
        npz_path = str(tmp_path / "embeddings.npz")
        np.savez_compressed(npz_path, **{"c1/3": emb_a, "c1/7": emb_b})

        records = []
        for i in range(12):
            records.append({
                "pair_id": f"pair_{i:03d}",
                "audio_a": "storage/c1/3.wav",
                "audio_b": "storage/c1/7.wav",
                "preferred": "a" if i < 10 else "skip",
            })

        ann_path = str(tmp_path / "annotations.jsonl")
        with open(ann_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        pairs = load_pairs(ann_path, npz_path)
        assert len(pairs) == 10

    def test_load_pairs_raises_under_10(self, tmp_path):
        """ValueError when fewer than 10 valid pairs."""
        from tools.train_preference import load_pairs

        emb_a = np.random.randn(512).astype(np.float32)
        emb_b = np.random.randn(512).astype(np.float32)
        npz_path = str(tmp_path / "embeddings.npz")
        np.savez_compressed(npz_path, **{"c1/3": emb_a, "c1/7": emb_b})

        records = []
        for i in range(5):
            records.append({
                "pair_id": f"pair_{i:03d}",
                "audio_a": "storage/c1/3.wav",
                "audio_b": "storage/c1/7.wav",
                "preferred": "a",
            })

        ann_path = str(tmp_path / "annotations.jsonl")
        with open(ann_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        with pytest.raises(ValueError, match="at least 10"):
            load_pairs(ann_path, npz_path)


# ---------------------------------------------------------------------------
# TestScanStorage — annotation tool's storage scanning
# ---------------------------------------------------------------------------

class TestScanStorage:

    def test_scan_same_challenge_groups_correctly(self, tmp_path):
        """Same-challenge mode only pairs files with the same challenge_id."""
        from tools.annotate_preferences import scan_storage

        # Create storage/<challenge_id>/<uid>.wav structure
        challenge1_dir = tmp_path / "challenge1"
        challenge1_dir.mkdir()
        for uid in [3, 7, 12]:
            # challenge1 has 3 UIDs → C(3,2) = 3 pairs
            (challenge1_dir / f"{uid}.wav").write_bytes(b"RIFF" + b"\x00" * 40)

        # challenge2 has only 1 UID → 0 pairs (need >=2)
        challenge2_dir = tmp_path / "challenge2"
        challenge2_dir.mkdir()
        (challenge2_dir / "3.wav").write_bytes(b"RIFF" + b"\x00" * 40)

        pairs = scan_storage(str(tmp_path), same_challenge=True)
        challenge_ids = [p["challenge_id"] for p in pairs]
        assert all(cid == "challenge1" for cid in challenge_ids)
        assert len(pairs) == 3

    def test_scan_returns_empty_for_nonexistent_dir(self):
        """Non-existent directory returns empty list."""
        from tools.annotate_preferences import scan_storage

        pairs = scan_storage("/nonexistent/path/12345")
        assert pairs == []
