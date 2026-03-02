"""
Plagiarism detection for TuneForge.

Dual approach:
1. Chromaprint audio fingerprinting via fpcalc subprocess
2. CLAP embedding cosine similarity

Fingerprints are stored in a local SQLite database so repeated
submissions can be caught across rounds.
"""

import hashlib
import io
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger

from tuneforge.config.scoring_config import CLAP_MODEL, PLAGIARISM_THRESHOLD
from tuneforge.scoring.clap_scorer import CLAPScorer


class PlagiarismDetector:
    """Detect plagiarised or duplicate audio submissions."""

    def __init__(self, db_path: str = "fingerprints.db") -> None:
        self._db_path = db_path
        self._clap = CLAPScorer(model_name=CLAP_MODEL)
        self._round_embeddings: dict[str, np.ndarray] = {}  # challenge_id:hotkey → embedding
        self._init_db()

    def _init_db(self) -> None:
        """Create fingerprint storage tables if they don't exist."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    miner_hotkey TEXT NOT NULL,
                    challenge_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fp_fingerprint ON fingerprints(fingerprint)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fp_miner ON fingerprints(miner_hotkey)"
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.error(f"Failed to init fingerprint DB: {exc}")

    def check(
        self,
        audio: np.ndarray,
        sr: int,
        miner_hotkey: str,
        challenge_id: str,
    ) -> tuple[bool, float]:
        """
        Check audio for plagiarism.

        Args:
            audio: 1-D float waveform.
            sr: Sample rate in Hz.
            miner_hotkey: Miner hotkey SS58 address.
            challenge_id: Unique challenge identifier.

        Returns:
            (is_plagiarized, max_similarity) — similarity in [0, 1].
        """
        max_sim = 0.0

        # --- Fingerprint check ---
        fp = self._get_fingerprint(audio, sr)
        if fp is not None:
            if self._fingerprint_exists(fp, miner_hotkey):
                logger.warning(f"Fingerprint match for miner {miner_hotkey[:16]}…")
                self._store_fingerprint(fp, miner_hotkey, challenge_id, None)
                return True, 1.0
        else:
            logger.debug("fpcalc not available, relying on CLAP similarity only")

        # --- CLAP embedding check ---
        embedding = self._get_embedding(audio, sr)
        if embedding is not None:
            sim = self._check_embedding_similarity(embedding, miner_hotkey, challenge_id)
            max_sim = max(max_sim, sim)

            # Store for future comparisons
            key = f"{challenge_id}:{miner_hotkey}"
            self._round_embeddings[key] = embedding

        # Store fingerprint for future rounds
        emb_blob = embedding.tobytes() if embedding is not None else None
        if fp is not None:
            self._store_fingerprint(fp, miner_hotkey, challenge_id, emb_blob)
        elif emb_blob is not None:
            self._store_fingerprint("", miner_hotkey, challenge_id, emb_blob)

        is_plagiarized = max_sim >= PLAGIARISM_THRESHOLD
        if is_plagiarized:
            logger.warning(
                f"Plagiarism detected for {miner_hotkey[:16]}… "
                f"(similarity={max_sim:.4f})"
            )
        return is_plagiarized, max_sim

    def clear_round_cache(self) -> None:
        """Clear per-round embedding cache."""
        self._round_embeddings.clear()

    # ------------------------------------------------------------------
    # Fingerprinting via fpcalc (chromaprint)
    # ------------------------------------------------------------------

    def _get_fingerprint(self, audio: np.ndarray, sr: int) -> str | None:
        """Generate Chromaprint fingerprint via fpcalc subprocess."""
        try:
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio.astype(np.float32), sr)
                result = subprocess.run(
                    ["fpcalc", "-raw", tmp.name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    return None
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("FINGERPRINT="):
                        return line.split("=", 1)[1]
            return None
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.debug(f"Fingerprinting failed: {exc}")
            return None

    def _fingerprint_exists(self, fingerprint: str, exclude_hotkey: str) -> bool:
        """Check if fingerprint exists in DB from a different miner."""
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.execute(
                "SELECT COUNT(*) FROM fingerprints "
                "WHERE fingerprint = ? AND miner_hotkey != ? AND fingerprint != ''",
                (fingerprint, exclude_hotkey),
            )
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except Exception as exc:
            logger.error(f"Fingerprint lookup failed: {exc}")
            return False

    def _store_fingerprint(
        self,
        fingerprint: str,
        miner_hotkey: str,
        challenge_id: str,
        embedding_blob: bytes | None,
    ) -> None:
        """Store fingerprint and optional embedding in DB."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT INTO fingerprints (miner_hotkey, challenge_id, fingerprint, embedding) "
                "VALUES (?, ?, ?, ?)",
                (miner_hotkey, challenge_id, fingerprint, embedding_blob),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.error(f"Fingerprint storage failed: {exc}")

    # ------------------------------------------------------------------
    # CLAP embedding similarity
    # ------------------------------------------------------------------

    def _get_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """Get CLAP audio embedding."""
        return self._clap.get_audio_embedding(audio, sr)

    def _check_embedding_similarity(
        self,
        embedding: np.ndarray,
        miner_hotkey: str,
        challenge_id: str,
    ) -> float:
        """
        Check embedding similarity against stored embeddings and round cache.

        Returns maximum cosine similarity found.
        """
        max_sim = 0.0
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Check round cache (other miners in same round)
        for key, stored_emb in self._round_embeddings.items():
            stored_hotkey = key.split(":", 1)[1] if ":" in key else ""
            if stored_hotkey == miner_hotkey:
                continue
            norm_stored = stored_emb / (np.linalg.norm(stored_emb) + 1e-8)
            sim = float(np.dot(norm_emb, norm_stored))
            max_sim = max(max_sim, sim)

        # Check DB embeddings (past rounds)
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.execute(
                "SELECT embedding FROM fingerprints "
                "WHERE miner_hotkey != ? AND embedding IS NOT NULL "
                "ORDER BY id DESC LIMIT 500",
                (miner_hotkey,),
            )
            for (emb_blob,) in cursor.fetchall():
                try:
                    stored_emb = np.frombuffer(emb_blob, dtype=np.float32)
                    if len(stored_emb) != len(embedding):
                        continue
                    norm_stored = stored_emb / (np.linalg.norm(stored_emb) + 1e-8)
                    sim = float(np.dot(norm_emb, norm_stored))
                    max_sim = max(max_sim, sim)
                except Exception:
                    continue
            conn.close()
        except Exception as exc:
            logger.error(f"DB embedding check failed: {exc}")

        return max_sim
