"""
Audio fingerprint scorer — detects known/commercial music submissions.

Uses Chromaprint (via ctypes to libchromaprint) to generate acoustic
fingerprints and checks them against the AcoustID database (~30M tracks).

Returns a penalty multiplier:
- 1.0 = no match found (original / AI-generated)
- 0.0 = high-confidence match to a known commercial song

This scorer specifically targets miners who submit existing music instead
of generating it. It does NOT compare miners against each other (which
causes false positives with similar AI models generating similar output).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import struct

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


def fingerprint_hash(fp: list[int]) -> str:
    """Generate a compact hash of a fingerprint for cache lookups."""
    components = fp[:64]
    raw = struct.pack(f">{len(components)}I", *components)  # Use first 64 components
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Fingerprint scorer — AcoustID only (no cross-miner comparison)
# ---------------------------------------------------------------------------

ACOUSTID_MATCH_THRESHOLD = 0.80  # Match score above which we flag as known song
ACOUSTID_PENALTY = 0.0           # Score multiplier for known-song match


class FingerprintScorer:
    """Detects known/commercial music via AcoustID fingerprint lookup.

    Checks submitted audio against the AcoustID database (~30M commercially
    released and catalogued tracks). If a submission matches a known song
    with high confidence, the miner receives a score of 0.

    This does NOT compare miners against each other — doing so causes
    false positives when multiple miners use similar models on the same
    prompt. We only flag submissions that match known, pre-existing music.

    Returns a penalty multiplier in [0.0, 1.0]:
    - 1.0 = no match (original or AI-generated)
    - 0.0 = confirmed match to a known song
    """

    def __init__(
        self,
        acoustid_api_key: str = "",
        **_kwargs,
    ) -> None:
        self._acoustid_api_key = acoustid_api_key
        self._acoustid_enabled = bool(acoustid_api_key)
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
        if not self._acoustid_enabled:
            return 1.0  # No API key — skip entirely

        fp = get_raw_fingerprint(audio, sr)
        if fp is None:
            return 1.0  # Can't fingerprint — no penalty

        fp_hash = fingerprint_hash(fp)

        # AcoustID lookup (known songs database)
        penalty = self._check_acoustid(fp, fp_hash, audio, sr)
        if penalty < 1.0:
            logger.warning(
                f"Known song detected: miner {miner_hotkey[:12]} submitted audio "
                f"matching a known song in AcoustID database (penalty={penalty:.2f})"
            )

        return penalty

    def _check_acoustid(
        self, fp: list[int], fp_hash: str, audio: np.ndarray, sr: int
    ) -> float:
        """Check fingerprint against AcoustID database."""
        # Check cache
        if fp_hash in self._acoustid_cache:
            return self._acoustid_cache[fp_hash]

        try:
            import httpx

            duration = int(len(audio) / sr)

            # AcoustID accepts raw fingerprint as comma-separated integers
            fp_str = ",".join(str(x) for x in fp)

            resp = httpx.post(
                "https://api.acoustid.org/v2/lookup",
                data={
                    "client": self._acoustid_api_key,
                    "duration": str(duration),
                    "fingerprint": fp_str,
                    "meta": "",  # Only need match score
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
