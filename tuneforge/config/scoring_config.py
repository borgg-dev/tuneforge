"""
Scoring and validation configuration constants for TuneForge subnet.

Consensus-critical constants (scoring weights, thresholds, EMA parameters)
are hardcoded and MUST NOT be modified by individual validators.  All
validators must produce identical scores for the same audio to maintain
Bittensor consensus.  Only operational parameters (file paths, network
config, timing intervals) are configurable via environment variables.
"""

import os


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Network (configurable — operational)
# ---------------------------------------------------------------------------
NETUID: int = _env_int("TF_NETUID", 0)
VERSION: str = _env_str("TF_VERSION", "1.0.0")
BURN_UID: int = _env_int("TF_BURN_UID", 0)
BURN_WEIGHT: float = _env_float("TF_BURN_WEIGHT", 0.0)

# ===========================================================================
# CONSENSUS-CRITICAL CONSTANTS
#
# Everything below this line through the "END CONSENSUS-CRITICAL" marker
# is hardcoded.  Changing these values will cause your validator to
# diverge from network consensus and may result in de-registration.
# ===========================================================================

# ---------------------------------------------------------------------------
# Scoring composite weights  (must sum to 1.0)
#
# Balanced across prompt adherence, composition, and production quality.
# Quality is spread across 16 active scorers for gaming resistance.
#   Prompt adherence:  clap + attribute                                = 30%
#   Composition:       musicality + melody + structural                = 21%
#   Production/fidelity: production + neural_quality + vocal + quality = 16%
#   Naturalness/mix:   vocal_lyrics + timbral + mix_sep + learned_mos  = 18%
#   Preference:        preference                              = 0% (bootstrap) / 2-20% (trained)
#   Other:             diversity + speed                               =  8%
#
# NOTE: preference weight is zeroed in bootstrap mode (no trained model).
# Its 7% base weight is redistributed to other scorers via renormalization.
#
# Artifact detection is applied as a penalty multiplier (not in weights).
# Vocals-requested boost: doubles vocal_lyrics weight when vocals are
# explicitly requested, then renormalizes.
# ---------------------------------------------------------------------------
SCORING_WEIGHTS: dict[str, float] = {
    # Prompt adherence (30%)
    "clap": 0.19,
    "attribute": 0.11,
    # Composition (21%)
    "musicality": 0.09,
    "melody": 0.06,
    "structural": 0.06,
    # Production/fidelity (16%)
    "production": 0.05,
    "neural_quality": 0.05,
    "vocal": 0.04,
    "quality": 0.02,
    # Preference (7% base — zeroed in bootstrap, auto-scaled 2-20% when trained)
    "preference": 0.07,
    # Naturalness & mix (18%)
    "vocal_lyrics": 0.08,
    "timbral": 0.03,
    "mix_separation": 0.04,
    "learned_mos": 0.03,
    # Other (8%)
    "diversity": 0.06,
    "speed": 0.02,
}

# ---------------------------------------------------------------------------
# Audio quality sub-metric weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
QUALITY_WEIGHTS: dict[str, float] = {
    "harmonic_ratio": 0.25,
    "onset_quality": 0.20,
    "spectral_contrast": 0.20,
    "dynamic_range": 0.15,
    "temporal_variation": 0.20,
}

# ---------------------------------------------------------------------------
# EMA / Leaderboard
# ---------------------------------------------------------------------------
EMA_ALPHA: float = 0.2
STEEPEN_POWER: float = 2.0
EMA_NEW_MINER_SEED: float = 0.0

# Tiered weight distribution: top ELITE_K miners share ELITE_POOL fraction
# of total weight; remaining miners share (1 - ELITE_POOL).
# This mirrors organic routing (top-10 get queries) and creates a
# highly competitive landscape where miners fight for top-10 slots.
ELITE_K: int = 10
ELITE_POOL: float = 0.80

# ---------------------------------------------------------------------------
# Silence threshold
# ---------------------------------------------------------------------------
SILENCE_THRESHOLD: float = 0.01

# ---------------------------------------------------------------------------
# Duration tolerance
# ---------------------------------------------------------------------------
DURATION_TOLERANCE: float = 0.20
DURATION_TOLERANCE_MAX: float = 0.50

# ---------------------------------------------------------------------------
# Scoring timeout (hard zero penalty if exceeded)
# ---------------------------------------------------------------------------
GENERATION_TIMEOUT: int = 300

# ---------------------------------------------------------------------------
# Duration defaults
# ---------------------------------------------------------------------------
DEFAULT_DURATION: float = 10.0
MAX_DURATION: float = 180.0
MIN_DURATION: float = 1.0

# ---------------------------------------------------------------------------
# CLAP scoring parameters
# ---------------------------------------------------------------------------
CLAP_MODEL: str = "laion/larger_clap_music"
CLAP_SAMPLE_RATE: int = 48000
# Empirical cosine similarity bounds for CLAP music-text pairs.
# Raw cosine similarities are remapped from [floor, ceiling] → [0, 1].
CLAP_SIM_FLOOR: float = 0.15
CLAP_SIM_CEILING: float = 0.75

# ---------------------------------------------------------------------------
# MERT model (neural audio quality)
# ---------------------------------------------------------------------------
MERT_MODEL: str = "m-a-p/MERT-v1-95M"
MERT_SAMPLE_RATE: int = 24000

# MERT bell curve parameters
MERT_TEMPORAL_COHERENCE_CENTER: float = 0.85
MERT_TEMPORAL_COHERENCE_WIDTH: float = 12.5
MERT_LAYER_AGREEMENT_CENTER: float = 0.6
MERT_LAYER_AGREEMENT_WIDTH: float = 8.0
MERT_PERIODICITY_CENTER: float = 0.5
MERT_PERIODICITY_WIDTH: float = 8.0
MERT_EXPECTED_NORM: float = 25.0

# ---------------------------------------------------------------------------
# LUFS loudness (ITU-R BS.1770-4)
# ---------------------------------------------------------------------------
LUFS_TOLERANCE: float = 4.0


# ---------------------------------------------------------------------------
# FAD (Frechet Audio Distance) penalty parameters
# ---------------------------------------------------------------------------
FAD_WINDOW_SIZE: int = 50
FAD_PENALTY_MIDPOINT: float = 15.0
FAD_PENALTY_STEEPNESS: float = 2.0
FAD_PENALTY_FLOOR: float = 0.5

# ---------------------------------------------------------------------------
# Fingerprint anti-plagiarism
# ---------------------------------------------------------------------------
FINGERPRINT_ACOUSTID_THRESHOLD: float = 0.80  # AcoustID match score threshold

# ---------------------------------------------------------------------------
# EnCodec model (neural codec quality)
# ---------------------------------------------------------------------------
ENCODEC_MODEL: str = "facebook/encodec_24khz"

# ---------------------------------------------------------------------------
# Preference weight auto-scaling
# ---------------------------------------------------------------------------
PREFERENCE_WEIGHT_MIN: float = 0.02
PREFERENCE_WEIGHT_MAX: float = 0.20
PREFERENCE_ACCURACY_MIN: float = 0.55
PREFERENCE_ACCURACY_MAX: float = 0.80

# ---------------------------------------------------------------------------
# Annotator reliability
# ---------------------------------------------------------------------------
ANNOTATOR_RELIABILITY_EMA: float = 0.1
ANNOTATOR_RELIABILITY_MIN: float = 0.1
ANNOTATOR_WEIGHTED_THRESHOLD: float = 0.6

# ===========================================================================
# END CONSENSUS-CRITICAL CONSTANTS
# ===========================================================================

# ---------------------------------------------------------------------------
# Configurable operational parameters (paths, timing, secrets)
# These do NOT affect scoring outcomes and can vary between validators.
# ---------------------------------------------------------------------------

# Timing / intervals (seconds or blocks)
ROUND_INTERVAL: int = _env_int("TF_ROUND_INTERVAL", 240)
WEIGHT_UPDATE_INTERVAL: int = _env_int("TF_WEIGHT_UPDATE_INTERVAL", 115)
METAGRAPH_SYNC_INTERVAL: int = _env_int("TF_METAGRAPH_SYNC_INTERVAL", 1200)

# EMA persistence
EMA_STATE_PATH: str = _env_str("TF_EMA_STATE_PATH", "./ema_state.json")
EMA_SAVE_INTERVAL: int = _env_int("TF_EMA_SAVE_INTERVAL", 5)

# Preference model checkpoint path
PREFERENCE_MODEL_PATH: str | None = os.environ.get("TF_PREFERENCE_MODEL_PATH", None) or None

# FAD reference statistics path
FAD_REFERENCE_STATS_PATH: str = _env_str("TF_FAD_REFERENCE_STATS_PATH", "./reference_fad_stats.npz")

# Active learning for annotations
ACTIVE_LEARNING_ENABLED: bool = _env_str("TF_ACTIVE_LEARNING_ENABLED", "true").lower() == "true"
ACTIVE_LEARNING_TOP_K: int = _env_int("TF_ACTIVE_LEARNING_TOP_K", 3)

# Default guidance scale (miner-side, not consensus-critical)
DEFAULT_GUIDANCE_SCALE: float = _env_float("TF_DEFAULT_GUIDANCE_SCALE", 3.0)

