"""
Audio fingerprint scorer — detects copied/plagiarized music.

Uses Chromaprint (via ctypes to libchromaprint) to generate acoustic
fingerprints and checks them against the AcoustID database (~30M tracks).

Returns a penalty multiplier:
- 1.0 = no match found (original / AI-generated)
- 0.0 = high-confidence match to a known song

Also maintains a local submission fingerprint database to detect
cross-miner copying and repeated submissions of the same audio.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import struct
import time
from collections import defaultdict
from threading import Lock

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Chromaprint via ctypes
# ---------------------------------------------------------------------------

_lib = None
_lib_error: str | None = None


def _load_chromaprint():
    """Load libchromaprint shared library."""
    global _lib, _lib_error
    if _lib is not None:
        return _lib
    if _lib_error is not None:
        return None

    lib_path = ctypes.util.find_library("chromaprint")
    if lib_path is None:
        _lib_error = "libchromaprint not found"
        logger.warning(f"Fingerprint scorer disabled: {_lib_error}")
        return None

    try:
        lib = ctypes.CDLL(lib_path)

        # chromaprint_new(algorithm) → ctx*
        lib.chromaprint_new.argtypes = [ctypes.c_int]
        lib.chromaprint_new.restype = ctypes.c_void_p

        # chromaprint_free(ctx*)
        lib.chromaprint_free.argtypes = [ctypes.c_void_p]
        lib.chromaprint_free.restype = None

        # chromaprint_start(ctx*, sample_rate, num_channels) → int
        lib.chromaprint_start.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.chromaprint_start.restype = ctypes.c_int

        # chromaprint_feed(ctx*, data*, size) → int
        lib.chromaprint_feed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int]
        lib.chromaprint_feed.restype = ctypes.c_int

        # chromaprint_finish(ctx*) → int
        lib.chromaprint_finish.argtypes = [ctypes.c_void_p]
        lib.chromaprint_finish.restype = ctypes.c_int

        # chromaprint_get_raw_fingerprint(ctx*, data**, size*) → int
        lib.chromaprint_get_raw_fingerprint.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.chromaprint_get_raw_fingerprint.restype = ctypes.c_int

        # chromaprint_dealloc(ptr*)
        lib.chromaprint_dealloc.argtypes = [ctypes.c_void_p]
        lib.chromaprint_dealloc.restype = None

        _lib = lib
        logger.info("Chromaprint library loaded for fingerprint scoring")
        return lib

    except Exception as exc:
        _lib_error = str(exc)
        logger.warning(f"Fingerprint scorer disabled: {_lib_error}")
        return None


def get_raw_fingerprint(audio: np.ndarray, sr: int) -> list[int] | None:
    """Generate raw Chromaprint fingerprint from audio.

    Args:
        audio: Mono float32 audio in [-1, 1].
        sr: Sample rate.

    Returns:
        List of uint32 fingerprint components, or None on failure.
    """
    lib = _load_chromaprint()
    if lib is None:
        return None

    # Convert to int16
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    # Algorithm 1 = CHROMAPRINT_ALGORITHM_DEFAULT
    ctx = lib.chromaprint_new(1)
    if not ctx:
        return None

    try:
        if not lib.chromaprint_start(ctx, sr, 1):
            return None

        data_ptr = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        if not lib.chromaprint_feed(ctx, data_ptr, len(pcm)):
            return None

        if not lib.chromaprint_finish(ctx):
            return None

        fp_ptr = ctypes.POINTER(ctypes.c_uint32)()
        fp_size = ctypes.c_int()

        if not lib.chromaprint_get_raw_fingerprint(ctx, ctypes.byref(fp_ptr), ctypes.byref(fp_size)):
            return None

        size = fp_size.value
        if size == 0:
            return None

        result = [fp_ptr[i] for i in range(size)]
        lib.chromaprint_dealloc(fp_ptr)
        return result

    finally:
        lib.chromaprint_free(ctx)


def fingerprint_similarity(fp1: list[int], fp2: list[int]) -> float:
    """Compute similarity between two raw fingerprints.

    Uses bit-error rate: fraction of matching bits across aligned
    fingerprint components. Returns 0.0 (no match) to 1.0 (identical).
    """
    if not fp1 or not fp2:
        return 0.0

    # Compare overlapping portion
    min_len = min(len(fp1), len(fp2))
    if min_len == 0:
        return 0.0

    matching_bits = 0
    total_bits = min_len * 32

    for i in range(min_len):
        xor = fp1[i] ^ fp2[i]
        # Count differing bits
        matching_bits += 32 - bin(xor).count("1")

    return matching_bits / total_bits


def fingerprint_hash(fp: list[int]) -> str:
    """Generate a compact hash of a fingerprint for dedup lookups."""
    raw = struct.pack(f">{len(fp)}I", *fp[:64])  # Use first 64 components
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Fingerprint scorer
# ---------------------------------------------------------------------------

# Similarity thresholds
ACOUSTID_MATCH_THRESHOLD = 0.80  # AcoustID match score above which we consider it a known song
LOCAL_DEDUP_THRESHOLD = 0.92     # Local fingerprint similarity above which we flag as duplicate
LOCAL_DEDUP_PENALTY = 0.0        # Score multiplier for local duplicate
ACOUSTID_PENALTY = 0.0           # Score multiplier for known-song match


class FingerprintScorer:
    """Detects copied music via acoustic fingerprinting.

    Two-layer detection:
    1. Local dedup: Maintains a rolling database of recent submission
       fingerprints. If a new submission is too similar to a previous
       one (from any miner), it's flagged.
    2. AcoustID lookup: Checks fingerprint against the AcoustID online
       database (~30M songs). If there's a high-confidence match, the
       submission is flagged as a known song.

    Returns a penalty multiplier in [0.0, 1.0]:
    - 1.0 = no issue detected
    - 0.0 = confirmed match (copied song or duplicate)
    """

    def __init__(
        self,
        local_window: int = 500,
        acoustid_api_key: str = "",
        acoustid_enabled: bool = True,
    ) -> None:
        self._lock = Lock()
        # Rolling window of (fingerprint_hash, raw_fp, miner_hotkey, timestamp)
        self._submissions: list[tuple[str, list[int], str, float]] = []
        self._local_window = local_window
        self._acoustid_api_key = acoustid_api_key
        self._acoustid_enabled = acoustid_enabled and bool(acoustid_api_key)
        # Cache AcoustID results to avoid repeated API calls
        self._acoustid_cache: dict[str, float] = {}

        if not self._acoustid_enabled:
            logger.info(
                "Fingerprint scorer: AcoustID lookup disabled "
                "(set TF_ACOUSTID_API_KEY to enable)"
            )

    def score(
        self,
        audio: np.ndarray,
        sr: int,
        miner_hotkey: str = "",
    ) -> float:
        """Score audio for originality. Returns penalty multiplier [0.0, 1.0]."""
        fp = get_raw_fingerprint(audio, sr)
        if fp is None:
            return 1.0  # Can't fingerprint — no penalty

        fp_hash = fingerprint_hash(fp)

        # Layer 1: Local dedup check
        local_penalty = self._check_local_dedup(fp, fp_hash, miner_hotkey)
        if local_penalty < 1.0:
            logger.warning(
                f"Fingerprint dedup: miner {miner_hotkey[:12]} submitted audio "
                f"matching a previous submission (similarity above {LOCAL_DEDUP_THRESHOLD})"
            )
            return local_penalty

        # Layer 2: AcoustID lookup (known songs database)
        acoustid_penalty = self._check_acoustid(fp, fp_hash, audio, sr)
        if acoustid_penalty < 1.0:
            logger.warning(
                f"Fingerprint match: miner {miner_hotkey[:12]} submitted audio "
                f"matching a known song in AcoustID database"
            )
            return acoustid_penalty

        # Record this submission for future dedup
        self._record_submission(fp, fp_hash, miner_hotkey)

        return 1.0

    def _check_local_dedup(
        self, fp: list[int], fp_hash: str, miner_hotkey: str
    ) -> float:
        """Check if this fingerprint matches any recent submission."""
        with self._lock:
            for prev_hash, prev_fp, prev_hotkey, _ in self._submissions:
                # Quick hash check first
                if prev_hash == fp_hash:
                    # Same miner re-submitting exact same audio — could be
                    # retry, penalize slightly less. Different miner = copying.
                    if prev_hotkey == miner_hotkey:
                        return 0.3  # Self-repeat
                    return LOCAL_DEDUP_PENALTY  # Cross-miner copy

                # Detailed similarity check (only compare first 128 components
                # for speed — covers ~10 seconds of audio)
                sim = fingerprint_similarity(fp[:128], prev_fp[:128])
                if sim >= LOCAL_DEDUP_THRESHOLD:
                    if prev_hotkey == miner_hotkey:
                        return 0.3  # Self-repeat
                    return LOCAL_DEDUP_PENALTY  # Cross-miner copy

        return 1.0

    def _record_submission(
        self, fp: list[int], fp_hash: str, miner_hotkey: str
    ) -> None:
        """Add fingerprint to the local submission database."""
        with self._lock:
            self._submissions.append((fp_hash, fp[:128], miner_hotkey, time.time()))
            # Trim to window size
            if len(self._submissions) > self._local_window:
                self._submissions = self._submissions[-self._local_window:]

    def _check_acoustid(
        self, fp: list[int], fp_hash: str, audio: np.ndarray, sr: int
    ) -> float:
        """Check fingerprint against AcoustID database."""
        if not self._acoustid_enabled:
            return 1.0

        # Check cache
        if fp_hash in self._acoustid_cache:
            return self._acoustid_cache[fp_hash]

        try:
            import httpx

            duration = int(len(audio) / sr)

            # Encode fingerprint for AcoustID API
            # AcoustID expects base64-encoded compressed fingerprint,
            # but the /v2/lookup endpoint also accepts raw fingerprint
            # as comma-separated integers
            fp_str = ",".join(str(x) for x in fp)

            resp = httpx.post(
                "https://api.acoustid.org/v2/lookup",
                data={
                    "client": self._acoustid_api_key,
                    "duration": str(duration),
                    "fingerprint": fp_str,
                    "meta": "",  # We only need match score, not metadata
                },
                timeout=10.0,
            )

            if resp.status_code != 200:
                logger.debug(f"AcoustID API returned {resp.status_code}")
                self._acoustid_cache[fp_hash] = 1.0
                return 1.0

            data = resp.json()
            results = data.get("results", [])

            if not results:
                self._acoustid_cache[fp_hash] = 1.0
                return 1.0

            # Check best match score
            best_score = max(r.get("score", 0.0) for r in results)

            if best_score >= ACOUSTID_MATCH_THRESHOLD:
                logger.info(
                    f"AcoustID match: score={best_score:.3f} "
                    f"(threshold={ACOUSTID_MATCH_THRESHOLD})"
                )
                self._acoustid_cache[fp_hash] = ACOUSTID_PENALTY
                return ACOUSTID_PENALTY

            self._acoustid_cache[fp_hash] = 1.0
            return 1.0

        except Exception as exc:
            logger.debug(f"AcoustID lookup failed: {exc}")
            # Don't penalize on API failure
            self._acoustid_cache[fp_hash] = 1.0
            return 1.0
