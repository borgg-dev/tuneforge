"""
Scoring and validation configuration constants for TuneForge subnet.

All values are configurable via environment variables with sensible defaults.
"""

import os


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
NETUID: int = _env_int("TF_NETUID", 0)
VERSION: str = _env_str("TF_VERSION", "1.0.0")
BURN_UID: int = _env_int("TF_BURN_UID", 0)
BURN_WEIGHT: float = _env_float("TF_BURN_WEIGHT", 0.0)

# ---------------------------------------------------------------------------
# Scoring composite weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
SCORING_WEIGHTS: dict[str, float] = {
    "clap": _env_float("TF_WEIGHT_CLAP", 0.35),
    "quality": _env_float("TF_WEIGHT_QUALITY", 0.25),
    "preference": _env_float("TF_WEIGHT_PREFERENCE", 0.20),
    "diversity": _env_float("TF_WEIGHT_DIVERSITY", 0.10),
    "speed": _env_float("TF_WEIGHT_SPEED", 0.10),
}

# ---------------------------------------------------------------------------
# Audio quality sub-metric weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
QUALITY_WEIGHTS: dict[str, float] = {
    "clipping": _env_float("TF_QW_CLIPPING", 0.20),
    "dynamic_range": _env_float("TF_QW_DYNAMIC_RANGE", 0.15),
    "spectral_quality": _env_float("TF_QW_SPECTRAL_QUALITY", 0.20),
    "content_ratio": _env_float("TF_QW_CONTENT_RATIO", 0.15),
    "bandwidth": _env_float("TF_QW_BANDWIDTH", 0.15),
    "structure": _env_float("TF_QW_STRUCTURE", 0.15),
}

# ---------------------------------------------------------------------------
# EMA / Leaderboard
# ---------------------------------------------------------------------------
EMA_ALPHA: float = _env_float("TF_EMA_ALPHA", 0.2)
EMA_WARMUP: int = 9  # ceil(2 / EMA_ALPHA - 1)
STEEPEN_BASELINE: float = _env_float("TF_STEEPEN_BASELINE", 0.6)
STEEPEN_POWER: float = _env_float("TF_STEEPEN_POWER", 3.0)

# ---------------------------------------------------------------------------
# Plagiarism / silence thresholds
# ---------------------------------------------------------------------------
PLAGIARISM_THRESHOLD: float = _env_float("TF_PLAGIARISM_THRESHOLD", 0.95)
SILENCE_THRESHOLD: float = _env_float("TF_SILENCE_THRESHOLD", 0.01)

# ---------------------------------------------------------------------------
# Duration tolerance
# ---------------------------------------------------------------------------
DURATION_TOLERANCE: float = _env_float("TF_DURATION_TOLERANCE", 0.20)
DURATION_TOLERANCE_MAX: float = _env_float("TF_DURATION_TOLERANCE_MAX", 0.50)

# ---------------------------------------------------------------------------
# Timing / intervals (seconds or blocks)
# ---------------------------------------------------------------------------
GENERATION_TIMEOUT: int = _env_int("TF_GENERATION_TIMEOUT", 120)
VALIDATION_INTERVAL: int = _env_int("TF_VALIDATION_INTERVAL", 300)
WEIGHT_UPDATE_INTERVAL: int = _env_int("TF_WEIGHT_UPDATE_INTERVAL", 175)
METAGRAPH_SYNC_INTERVAL: int = _env_int("TF_METAGRAPH_SYNC_INTERVAL", 1200)

# ---------------------------------------------------------------------------
# Duration defaults
# ---------------------------------------------------------------------------
DEFAULT_DURATION: float = _env_float("TF_DEFAULT_DURATION", 10.0)
MAX_DURATION: float = _env_float("TF_MAX_DURATION", 60.0)
MIN_DURATION: float = _env_float("TF_MIN_DURATION", 1.0)
DEFAULT_GUIDANCE_SCALE: float = _env_float("TF_DEFAULT_GUIDANCE_SCALE", 3.0)

# ---------------------------------------------------------------------------
# Speed scoring curve
# ---------------------------------------------------------------------------
SPEED_BEST_SECONDS: float = _env_float("TF_SPEED_BEST_SECONDS", 5.0)
SPEED_MID_SECONDS: float = _env_float("TF_SPEED_MID_SECONDS", 30.0)
SPEED_MID_SCORE: float = _env_float("TF_SPEED_MID_SCORE", 0.3)
SPEED_MAX_SECONDS: float = _env_float("TF_SPEED_MAX_SECONDS", 60.0)

# ---------------------------------------------------------------------------
# CLAP model
# ---------------------------------------------------------------------------
CLAP_MODEL: str = _env_str("TF_CLAP_MODEL", "laion/larger_clap_music")
CLAP_SAMPLE_RATE: int = _env_int("TF_CLAP_SAMPLE_RATE", 48000)
