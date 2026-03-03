"""
Plagiarism detection for TuneForge.

Uses Chromaprint audio fingerprinting (via fpcalc subprocess) to detect:
1. Cross-miner exact replay — a miner submits audio identical to another miner's
2. Self-plagiarism — a miner resubmits the same audio across different challenges

CLAP embedding similarity is NOT used for plagiarism because miners responding
to the same prompt will naturally produce semantically similar audio. Penalizing
semantic similarity punishes miners for doing the right thing (matching the prompt).

Fingerprints are stored in a local SQLite database so repeated
submissions can be caught across rounds.
"""

import sqlite3
import subprocess
import tempfile

import numpy as np
from loguru import logger

from tuneforge.config.scoring_config import SELF_PLAGIARISM_THRESHOLD


class PlagiarismDetector:
    """Detect plagiarised or duplicate audio submissions via fingerprinting."""

    def __init__(self, db_path: str = "fingerprints.db") -> None:
        self._db_path = db_path
        self._round_fingerprints: dict[str, str] = {}  # challenge_id:hotkey → fingerprint
        self._store_count: int = 0
        self._fpcalc_available: bool | None = None
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
        Check audio for plagiarism using Chromaprint fingerprinting.

        Args:
            audio: 1-D float waveform.
            sr: Sample rate in Hz.
            miner_hotkey: Miner hotkey SS58 address.
            challenge_id: Unique challenge identifier.

        Returns:
            (is_plagiarized, similarity) — similarity is 1.0 if plagiarized, 0.0 otherwise.
        """
        fp = self._get_fingerprint(audio, sr)
        if fp is None:
            # fpcalc not available — cannot detect plagiarism
            return False, 0.0

        # --- Cross-miner exact replay check ---
        # Check against other miners in the current round
        for key, stored_fp in self._round_fingerprints.items():
            stored_hotkey = key.split(":", 1)[1] if ":" in key else ""
            if stored_hotkey == miner_hotkey:
                continue
            if fp == stored_fp:
                logger.warning(f"Exact fingerprint match (cross-miner) for {miner_hotkey[:16]}…")
                self._store_fingerprint(fp, miner_hotkey, challenge_id)
                return True, 1.0

        # Check against other miners in past rounds (DB)
        if self._fingerprint_exists(fp, miner_hotkey):
            logger.warning(f"Exact fingerprint match (past round) for {miner_hotkey[:16]}…")
            self._store_fingerprint(fp, miner_hotkey, challenge_id)
            return True, 1.0

        # --- Self-plagiarism check ---
        # Check if this miner submitted this exact fingerprint before
        self_sim = self._check_self_plagiarism(fp, miner_hotkey)
        if self_sim >= SELF_PLAGIARISM_THRESHOLD:
            logger.warning(
                f"Self-plagiarism detected for {miner_hotkey[:16]}… "
                f"(fingerprint match)"
            )
            self._store_fingerprint(fp, miner_hotkey, challenge_id)
            return True, self_sim

        # Store for future comparisons
        key = f"{challenge_id}:{miner_hotkey}"
        self._round_fingerprints[key] = fp
        self._store_fingerprint(fp, miner_hotkey, challenge_id)

        return False, 0.0

    def clear_round_cache(self) -> None:
        """Clear per-round fingerprint cache."""
        self._round_fingerprints.clear()

    # ------------------------------------------------------------------
    # Fingerprinting via fpcalc (chromaprint)
    # ------------------------------------------------------------------

    def _get_fingerprint(self, audio: np.ndarray, sr: int) -> str | None:
        """Generate Chromaprint fingerprint via fpcalc subprocess."""
        # Cache fpcalc availability check
        if self._fpcalc_available is False:
            return None

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
                        self._fpcalc_available = True
                        return line.split("=", 1)[1]
            return None
        except FileNotFoundError:
            if self._fpcalc_available is None:
                logger.info("fpcalc not found — fingerprint-based plagiarism detection disabled")
            self._fpcalc_available = False
            return None
        except Exception as exc:
            logger.debug(f"Fingerprinting failed: {exc}")
            return None

    def _fingerprint_exists(self, fingerprint: str, exclude_hotkey: str) -> bool:
        """Check if fingerprint exists in DB from a different miner."""
        if not fingerprint:
            return False
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

    def _check_self_plagiarism(self, fingerprint: str, miner_hotkey: str) -> float:
        """
        Check if miner submitted this exact fingerprint before.

        Returns 1.0 if exact match found, 0.0 otherwise.
        """
        if not fingerprint:
            return 0.0
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.execute(
                "SELECT COUNT(*) FROM fingerprints "
                "WHERE fingerprint = ? AND miner_hotkey = ? AND fingerprint != ''",
                (fingerprint, miner_hotkey),
            )
            count = cursor.fetchone()[0]
            conn.close()
            return 1.0 if count > 0 else 0.0
        except Exception as exc:
            logger.error(f"Self-plagiarism check failed: {exc}")
            return 0.0

    def _store_fingerprint(
        self,
        fingerprint: str,
        miner_hotkey: str,
        challenge_id: str,
    ) -> None:
        """Store fingerprint in DB."""
        if not fingerprint:
            return
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT INTO fingerprints (miner_hotkey, challenge_id, fingerprint) "
                "VALUES (?, ?, ?)",
                (miner_hotkey, challenge_id, fingerprint),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.error(f"Fingerprint storage failed: {exc}")
            return

        self._store_count += 1
        if self._store_count % 100 == 0:
            self._prune_old_entries()

    def _prune_old_entries(self) -> None:
        """Delete entries older than the most recent 10000 rows."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "DELETE FROM fingerprints WHERE id NOT IN ("
                "SELECT id FROM fingerprints ORDER BY id DESC LIMIT 10000"
                ")"
            )
            conn.commit()
            conn.close()
            logger.debug("Pruned old fingerprint entries; kept last 10000.")
        except Exception as exc:
            logger.error(f"Fingerprint pruning failed: {exc}")
