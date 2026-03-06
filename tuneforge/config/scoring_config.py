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
#
# Balanced across prompt adherence, composition, and production quality.
# Quality is spread across 18 independent scorers for gaming resistance.
#   Prompt adherence:  clap + attribute                                        = 24%
#   Composition:       musicality + melody + structural                        = 21%
#   Production/fidelity: production + neural_quality + vocal + quality         = 16%
#   Naturalness/mix:   vocal_lyrics + timbral + mix_sep + learned_mos          = 20%
#   Perceptual:        perceptual + neural_codec                               =  2%
#   Preference:        preference                                      = 0% (bootstrap) / 2-20% (trained)
#   Other:             diversity + speed                                       = 10%
#
# NOTE: preference weight is zeroed in bootstrap mode (no trained model).
# Its 7% base weight is redistributed to other scorers via renormalization.
#
# Artifact detection is applied as a penalty multiplier (not in weights).
# Vocals-requested boost: doubles vocal_lyrics weight when vocals are
# explicitly requested, then renormalizes.
# ---------------------------------------------------------------------------
SCORING_WEIGHTS: dict[str, float] = {
    # Prompt adherence (24% — CLAP kept at 15% to avoid amplifying gaming
    # risk; attribute boosted for concrete, less-gameable prompt checks)
    "clap": _env_float("TF_WEIGHT_CLAP", 0.15),
    "attribute": _env_float("TF_WEIGHT_ATTRIBUTE", 0.09),
    # Composition (21%) + Production/fidelity (16%) + Preference (7% base)
    "musicality": _env_float("TF_WEIGHT_MUSICALITY", 0.09),
    "production": _env_float("TF_WEIGHT_PRODUCTION", 0.05),
    "melody": _env_float("TF_WEIGHT_MELODY", 0.06),
    "neural_quality": _env_float("TF_WEIGHT_NEURAL_QUALITY", 0.05),
    "preference": _env_float("TF_WEIGHT_PREFERENCE", 0.07),
    "structural": _env_float("TF_WEIGHT_STRUCTURAL", 0.06),
    "vocal": _env_float("TF_WEIGHT_VOCAL", 0.04),
    "quality": _env_float("TF_WEIGHT_QUALITY", 0.02),
    # Perceptual quality (2%)
    "perceptual": _env_float("TF_WEIGHT_PERCEPTUAL", 0.01),
    "neural_codec": _env_float("TF_WEIGHT_NEURAL_CODEC", 0.01),
    # Naturalness & mix (20%)
    "timbral": _env_float("TF_WEIGHT_TIMBRAL", 0.05),
    "vocal_lyrics": _env_float("TF_WEIGHT_VOCAL_LYRICS", 0.08),
    "mix_separation": _env_float("TF_WEIGHT_MIX_SEPARATION", 0.04),
    "learned_mos": _env_float("TF_WEIGHT_LEARNED_MOS", 0.03),
    # Other (10% — diversity kept at 8% for anti-convergence defense;
    # speed reduced to 2% since it's now duration-relative)
    "diversity": _env_float("TF_WEIGHT_DIVERSITY", 0.08),
    "speed": _env_float("TF_WEIGHT_SPEED", 0.02),
}

# ---------------------------------------------------------------------------
# Audio quality sub-metric weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
QUALITY_WEIGHTS: dict[str, float] = {
    "harmonic_ratio": _env_float("TF_QW_HARMONIC_RATIO", 0.25),
    "onset_quality": _env_float("TF_QW_ONSET_QUALITY", 0.20),
    "spectral_contrast": _env_float("TF_QW_SPECTRAL_CONTRAST", 0.20),
    "dynamic_range": _env_float("TF_QW_DYNAMIC_RANGE", 0.15),
    "temporal_variation": _env_float("TF_QW_TEMPORAL_VARIATION", 0.20),
}

# ---------------------------------------------------------------------------
# EMA / Leaderboard
# ---------------------------------------------------------------------------
EMA_ALPHA: float = _env_float("TF_EMA_ALPHA", 0.2)
EMA_WARMUP: int = _env_int("TF_EMA_WARMUP", 9)  # kept for reference; no longer used as a gate
STEEPEN_BASELINE: float = _env_float("TF_STEEPEN_BASELINE", 0.45)
STEEPEN_POWER: float = _env_float("TF_STEEPEN_POWER", 2.0)

# ---------------------------------------------------------------------------
# Plagiarism / silence thresholds
# ---------------------------------------------------------------------------
SELF_PLAGIARISM_THRESHOLD: float = _env_float("TF_SELF_PLAGIARISM_THRESHOLD", 0.72)
# Soft plagiarism zone: similarity between soft and hard threshold gets a
# partial penalty multiplier (0.3) instead of hard zero.
SOFT_PLAGIARISM_THRESHOLD: float = _env_float("TF_SOFT_PLAGIARISM_THRESHOLD", 0.65)
# Cross-miner plagiarism uses a slightly stricter threshold
CROSS_MINER_PLAGIARISM_THRESHOLD: float = _env_float("TF_CROSS_MINER_PLAGIARISM_THRESHOLD", 0.70)
SILENCE_THRESHOLD: float = _env_float("TF_SILENCE_THRESHOLD", 0.01)

# ---------------------------------------------------------------------------
# Duration tolerance
# ---------------------------------------------------------------------------
DURATION_TOLERANCE: float = _env_float("TF_DURATION_TOLERANCE", 0.20)
DURATION_TOLERANCE_MAX: float = _env_float("TF_DURATION_TOLERANCE_MAX", 0.50)

# ---------------------------------------------------------------------------
# Timing / intervals (seconds or blocks)
# ---------------------------------------------------------------------------
GENERATION_TIMEOUT: int = _env_int("TF_GENERATION_TIMEOUT", 300)
VALIDATION_INTERVAL: int = _env_int("TF_VALIDATION_INTERVAL", 300)
WEIGHT_UPDATE_INTERVAL: int = _env_int("TF_WEIGHT_UPDATE_INTERVAL", 115)
METAGRAPH_SYNC_INTERVAL: int = _env_int("TF_METAGRAPH_SYNC_INTERVAL", 1200)

# ---------------------------------------------------------------------------
# Duration defaults
# ---------------------------------------------------------------------------
DEFAULT_DURATION: float = _env_float("TF_DEFAULT_DURATION", 10.0)
MAX_DURATION: float = _env_float("TF_MAX_DURATION", 180.0)
MIN_DURATION: float = _env_float("TF_MIN_DURATION", 1.0)
DEFAULT_GUIDANCE_SCALE: float = _env_float("TF_DEFAULT_GUIDANCE_SCALE", 3.0)

# ---------------------------------------------------------------------------
# Speed scoring curve
# ---------------------------------------------------------------------------
SPEED_BEST_SECONDS: float = _env_float("TF_SPEED_BEST_SECONDS", 15.0)
SPEED_MID_SECONDS: float = _env_float("TF_SPEED_MID_SECONDS", 45.0)
SPEED_MID_SCORE: float = _env_float("TF_SPEED_MID_SCORE", 0.3)
SPEED_MAX_SECONDS: float = _env_float("TF_SPEED_MAX_SECONDS", 90.0)

# ---------------------------------------------------------------------------
# CLAP model
# ---------------------------------------------------------------------------
CLAP_MODEL: str = _env_str("TF_CLAP_MODEL", "laion/larger_clap_music")
CLAP_SAMPLE_RATE: int = _env_int("TF_CLAP_SAMPLE_RATE", 48000)
# Empirical cosine similarity bounds for CLAP music-text pairs.
# Raw cosine similarities are remapped from [floor, ceiling] → [0, 1].
# Calibrated on laion/clap-htsat-unfused with MusicGen outputs:
#   unrelated audio ~0.0, weak match ~0.15, good match ~0.40-0.60.
CLAP_SIM_FLOOR: float = _env_float("TF_CLAP_SIM_FLOOR", 0.15)
CLAP_SIM_CEILING: float = _env_float("TF_CLAP_SIM_CEILING", 0.75)

# ---------------------------------------------------------------------------
# MERT model (neural audio quality)
# ---------------------------------------------------------------------------
MERT_MODEL: str = _env_str("TF_MERT_MODEL", "m-a-p/MERT-v1-95M")
MERT_SAMPLE_RATE: int = _env_int("TF_MERT_SAMPLE_RATE", 24000)

# MERT bell curve parameters (calibratable via env or tools/calibrate_mert.py)
MERT_TEMPORAL_COHERENCE_CENTER: float = _env_float("TF_MERT_TEMPORAL_CENTER", 0.85)
MERT_TEMPORAL_COHERENCE_WIDTH: float = _env_float("TF_MERT_TEMPORAL_WIDTH", 12.5)
MERT_LAYER_AGREEMENT_CENTER: float = _env_float("TF_MERT_LAYER_CENTER", 0.6)
MERT_LAYER_AGREEMENT_WIDTH: float = _env_float("TF_MERT_LAYER_WIDTH", 8.0)
MERT_PERIODICITY_CENTER: float = _env_float("TF_MERT_PERIODICITY_CENTER", 0.5)
MERT_PERIODICITY_WIDTH: float = _env_float("TF_MERT_PERIODICITY_WIDTH", 8.0)
MERT_EXPECTED_NORM: float = _env_float("TF_MERT_EXPECTED_NORM", 25.0)

# ---------------------------------------------------------------------------
# LUFS loudness (ITU-R BS.1770-4)
# ---------------------------------------------------------------------------
LUFS_TOLERANCE: float = _env_float("TF_LUFS_TOLERANCE", 4.0)

# ---------------------------------------------------------------------------
# Weight perturbation (anti-gaming)
# Per-round random ±N% adjustment to composite weights, seeded by challenge_id.
# Set to 0.0 to disable perturbation entirely.
# ---------------------------------------------------------------------------
WEIGHT_PERTURBATION: float = _env_float("TF_WEIGHT_PERTURBATION", 0.30)

# ---------------------------------------------------------------------------
# Preference model checkpoint
# Path to a trained PreferenceHead checkpoint (.pt).  When set, the
# preference model switches from the bootstrap heuristic to learned scoring.
# ---------------------------------------------------------------------------
PREFERENCE_MODEL_PATH: str | None = os.environ.get("TF_PREFERENCE_MODEL_PATH", None) or None

# ---------------------------------------------------------------------------
# Scorer dropout rate (anti-gaming)
# ---------------------------------------------------------------------------
SCORER_DROPOUT_RATE: float = _env_float("TF_SCORER_DROPOUT_RATE", 0.10)

# ---------------------------------------------------------------------------
# EMA persistence
# ---------------------------------------------------------------------------
EMA_STATE_PATH: str = _env_str("TF_EMA_STATE_PATH", "./ema_state.json")
EMA_SAVE_INTERVAL: int = _env_int("TF_EMA_SAVE_INTERVAL", 5)
EMA_NEW_MINER_SEED: float = _env_float("TF_EMA_NEW_MINER_SEED", 0.25)

# ---------------------------------------------------------------------------
# FAD (Frechet Audio Distance) scoring
# ---------------------------------------------------------------------------
FAD_WINDOW_SIZE: int = _env_int("TF_FAD_WINDOW_SIZE", 50)
FAD_REFERENCE_STATS_PATH: str = _env_str("TF_FAD_REFERENCE_STATS_PATH", "./reference_fad_stats.npz")
FAD_PENALTY_MIDPOINT: float = _env_float("TF_FAD_PENALTY_MIDPOINT", 15.0)
FAD_PENALTY_STEEPNESS: float = _env_float("TF_FAD_PENALTY_STEEPNESS", 2.0)
FAD_PENALTY_FLOOR: float = _env_float("TF_FAD_PENALTY_FLOOR", 0.5)

# ---------------------------------------------------------------------------
# EnCodec model (neural codec quality)
# ---------------------------------------------------------------------------
ENCODEC_MODEL: str = _env_str("TF_ENCODEC_MODEL", "facebook/encodec_24khz")

# ---------------------------------------------------------------------------
# Preference weight auto-scaling
# ---------------------------------------------------------------------------
PREFERENCE_WEIGHT_MIN: float = _env_float("TF_PREFERENCE_WEIGHT_MIN", 0.02)
PREFERENCE_WEIGHT_MAX: float = _env_float("TF_PREFERENCE_WEIGHT_MAX", 0.20)
PREFERENCE_ACCURACY_MIN: float = _env_float("TF_PREFERENCE_ACCURACY_MIN", 0.55)
PREFERENCE_ACCURACY_MAX: float = _env_float("TF_PREFERENCE_ACCURACY_MAX", 0.80)

# ---------------------------------------------------------------------------
# Active learning for annotations
# ---------------------------------------------------------------------------
ACTIVE_LEARNING_ENABLED: bool = _env_str("TF_ACTIVE_LEARNING_ENABLED", "true").lower() == "true"
ACTIVE_LEARNING_TOP_K: int = _env_int("TF_ACTIVE_LEARNING_TOP_K", 3)

# ---------------------------------------------------------------------------
# Annotator reliability
# ---------------------------------------------------------------------------
ANNOTATOR_RELIABILITY_EMA: float = _env_float("TF_ANNOTATOR_RELIABILITY_EMA", 0.1)
ANNOTATOR_RELIABILITY_MIN: float = _env_float("TF_ANNOTATOR_RELIABILITY_MIN", 0.1)
ANNOTATOR_WEIGHTED_THRESHOLD: float = _env_float("TF_ANNOTATOR_WEIGHTED_THRESHOLD", 0.6)

# ---------------------------------------------------------------------------
# Validator perturbation secret
# A private nonce used to seed weight perturbation. MUST NOT be shared with
# miners. Each validator should set a unique value. If empty, a random secret
# is auto-generated at startup (ephemeral — changes on restart).
# ---------------------------------------------------------------------------
def _default_perturbation_secret() -> str:
    """Auto-generate a random perturbation secret if none is configured."""
    configured = os.environ.get("TF_VALIDATOR_PERTURBATION_SECRET", "")
    if configured:
        return configured
    import secrets
    generated = secrets.token_hex(32)
    import logging
    logging.getLogger("tuneforge.config").warning(
        "TF_VALIDATOR_PERTURBATION_SECRET not set — auto-generated ephemeral secret. "
        "Set this env var for consistent scoring across restarts."
    )
    return generated

VALIDATOR_PERTURBATION_SECRET: str = _default_perturbation_secret()
