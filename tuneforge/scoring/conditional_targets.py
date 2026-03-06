"""
Conditional target deriver for TuneForge.

Derives quality targets dynamically from the challenge prompt text
instead of relying on fixed bell-curve centres.  Uses fast keyword
matching (no ML inference) so it can run in the scoring hot path.

Usage::

    deriver = ConditionalTargetDeriver()
    targets = deriver.derive_targets(
        prompt="an energetic heavy metal track with fast drums",
        genre="metal",
        duration=30.0,
    )
    # targets["onset_density_min"] == 4.0  (higher for energetic)
    # targets["rhythmic_regularity_floor"] == 0.1  (lowered for metal)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Default target values (neutral / balanced)
# ---------------------------------------------------------------------------

@dataclass
class ScorerTargets:
    """Adjustable target parameters derived from prompt context."""

    # Rhythmic regularity: lower floor = more tolerance for irregular rhythms
    rhythmic_regularity_floor: float = 0.3

    # Minimum harmonic variety: number of distinct chroma peaks expected
    harmonic_variety_min: float = 3.0

    # Onset density: expected onsets per second
    onset_density_min: float = 1.5
    onset_density_max: float = 6.0

    # Arrangement contrast: spectral centroid coefficient of variation target
    arrangement_contrast_target: float = 0.15

    # Dynamic range: minimum expected dB range
    dynamic_range_min: float = 4.0
    dynamic_range_target: float = 10.0

    # Section variety: pairwise section dissimilarity target
    section_variety_target: float = 0.35

    # Spectral contrast: expected spectral contrast level
    spectral_contrast_target: float = 0.5

    # Transition smoothness: penalty scale for abrupt transitions
    transition_smoothness_weight: float = 1.0

    # Harmonic progression variety: expected chord change rate
    harmonic_progression_min: float = 2.0

    def to_dict(self) -> dict[str, float]:
        """Serialise all targets to a flat dict."""
        return {
            "rhythmic_regularity_floor": self.rhythmic_regularity_floor,
            "harmonic_variety_min": self.harmonic_variety_min,
            "onset_density_min": self.onset_density_min,
            "onset_density_max": self.onset_density_max,
            "arrangement_contrast_target": self.arrangement_contrast_target,
            "dynamic_range_min": self.dynamic_range_min,
            "dynamic_range_target": self.dynamic_range_target,
            "section_variety_target": self.section_variety_target,
            "spectral_contrast_target": self.spectral_contrast_target,
            "transition_smoothness_weight": self.transition_smoothness_weight,
            "harmonic_progression_min": self.harmonic_progression_min,
        }


# ---------------------------------------------------------------------------
# Keyword categories and their target adjustments
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _KeywordRule:
    """A set of keywords that, when matched, apply target adjustments."""

    keywords: frozenset[str]
    adjustments: dict[str, float]  # field_name -> additive delta


# Each rule's adjustments are *additive* — multiple matching rules stack.
# Final values are clamped to sensible bounds after all rules apply.

_RULES: list[_KeywordRule] = [
    # --- Chaotic / experimental ---
    _KeywordRule(
        keywords=frozenset({
            "chaotic", "experimental", "free jazz", "noise", "atonal",
            "avant-garde", "dissonant", "glitch", "breakcore", "harsh",
            "industrial", "cacophony", "discordant", "frenetic",
        }),
        adjustments={
            "rhythmic_regularity_floor": -0.2,
            "harmonic_variety_min": 2.0,
            "section_variety_target": 0.15,
            "arrangement_contrast_target": 0.10,
            "onset_density_max": 4.0,
        },
    ),
    # --- Minimal / ambient / sparse ---
    _KeywordRule(
        keywords=frozenset({
            "minimal", "ambient", "sparse", "ethereal", "drone",
            "meditative", "contemplative", "space", "spacious",
            "atmospheric", "textural", "pad", "soundscape", "quiet",
            "gentle", "delicate", "subtle", "soft",
        }),
        adjustments={
            "onset_density_min": -1.0,
            "onset_density_max": -2.0,
            "arrangement_contrast_target": -0.08,
            "dynamic_range_min": -2.0,
            "section_variety_target": -0.15,
            "harmonic_progression_min": -1.0,
            "transition_smoothness_weight": 0.3,
        },
    ),
    # --- Energetic / intense / heavy ---
    _KeywordRule(
        keywords=frozenset({
            "energetic", "intense", "heavy", "aggressive", "powerful",
            "driving", "fast", "uptempo", "high-energy", "hard",
            "thrashing", "pounding", "furious", "explosive", "fierce",
            "relentless", "punchy", "banger", "hype",
        }),
        adjustments={
            "dynamic_range_min": 2.0,
            "dynamic_range_target": 2.0,
            "onset_density_min": 2.5,
            "onset_density_max": 4.0,
            "spectral_contrast_target": 0.15,
            "arrangement_contrast_target": 0.05,
        },
    ),
    # --- Smooth / chill / relaxing ---
    _KeywordRule(
        keywords=frozenset({
            "smooth", "chill", "relaxing", "laid-back", "mellow",
            "soothing", "calm", "peaceful", "serene", "easy",
            "warm", "cozy", "lush", "silky", "breezy", "lazy",
            "lounge", "lo-fi",
        }),
        adjustments={
            "spectral_contrast_target": -0.15,
            "transition_smoothness_weight": 0.5,
            "onset_density_min": -0.5,
            "onset_density_max": -1.5,
            "dynamic_range_min": -1.0,
            "arrangement_contrast_target": -0.05,
        },
    ),
    # --- Complex / progressive / intricate ---
    _KeywordRule(
        keywords=frozenset({
            "complex", "progressive", "intricate", "technical",
            "virtuosic", "polyrhythmic", "syncopated", "sophisticated",
            "elaborate", "multi-layered", "layered", "rich",
            "orchestral", "symphonic", "contrapuntal",
        }),
        adjustments={
            "section_variety_target": 0.15,
            "harmonic_variety_min": 2.0,
            "harmonic_progression_min": 2.0,
            "arrangement_contrast_target": 0.08,
            "rhythmic_regularity_floor": -0.1,
        },
    ),
    # --- Repetitive / hypnotic / loop-based ---
    _KeywordRule(
        keywords=frozenset({
            "repetitive", "hypnotic", "looping", "loop", "monotone",
            "trance", "mantric", "cyclical", "ostinato", "steady",
            "four-on-the-floor",
        }),
        adjustments={
            "section_variety_target": -0.15,
            "rhythmic_regularity_floor": 0.15,
            "harmonic_variety_min": -1.0,
            "harmonic_progression_min": -1.0,
            "arrangement_contrast_target": -0.05,
        },
    ),
    # --- Dramatic / cinematic / epic ---
    _KeywordRule(
        keywords=frozenset({
            "dramatic", "cinematic", "epic", "grand", "majestic",
            "sweeping", "triumphant", "heroic", "climactic", "soaring",
            "bombastic", "grandiose",
        }),
        adjustments={
            "dynamic_range_min": 3.0,
            "dynamic_range_target": 4.0,
            "section_variety_target": 0.10,
            "arrangement_contrast_target": 0.10,
            "spectral_contrast_target": 0.10,
        },
    ),
    # --- Dark / moody ---
    _KeywordRule(
        keywords=frozenset({
            "dark", "moody", "brooding", "ominous", "sinister",
            "haunting", "eerie", "menacing", "gothic", "grim",
            "bleak", "somber", "melancholic",
        }),
        adjustments={
            "spectral_contrast_target": -0.05,
            "transition_smoothness_weight": 0.2,
            "dynamic_range_min": 1.0,
        },
    ),
]

# ---------------------------------------------------------------------------
# Clamp bounds for final target values
# ---------------------------------------------------------------------------

_CLAMP_BOUNDS: dict[str, tuple[float, float]] = {
    "rhythmic_regularity_floor": (0.0, 0.8),
    "harmonic_variety_min": (1.0, 8.0),
    "onset_density_min": (0.2, 8.0),
    "onset_density_max": (1.0, 15.0),
    "arrangement_contrast_target": (0.02, 0.40),
    "dynamic_range_min": (1.0, 15.0),
    "dynamic_range_target": (4.0, 20.0),
    "section_variety_target": (0.05, 0.60),
    "spectral_contrast_target": (0.1, 0.9),
    "transition_smoothness_weight": (0.3, 2.0),
    "harmonic_progression_min": (0.5, 8.0),
}


class ConditionalTargetDeriver:
    """Derive scorer targets from challenge prompt context.

    Uses keyword matching for speed — no ML models required.
    Multiple keyword categories can stack additively; final
    values are clamped to sensible bounds.
    """

    def __init__(self) -> None:
        # Pre-compile a combined pattern for each rule for fast matching.
        # Multi-word keywords (e.g. "free jazz") need phrase matching,
        # so we sort longest-first and use re.search.
        self._compiled_rules: list[tuple[re.Pattern[str], dict[str, float]]] = []
        for rule in _RULES:
            # Build alternation pattern, longest keywords first
            sorted_kw = sorted(rule.keywords, key=len, reverse=True)
            escaped = [re.escape(kw) for kw in sorted_kw]
            pattern = re.compile(
                r"\b(?:" + "|".join(escaped) + r")\b",
                re.IGNORECASE,
            )
            self._compiled_rules.append((pattern, rule.adjustments))

    def derive_targets(
        self,
        prompt: str,
        genre: str = "",
        duration: float = 10.0,
    ) -> dict[str, float]:
        """Return adjusted target parameters based on prompt analysis.

        Args:
            prompt: The challenge prompt text.
            genre: Optional genre string (used as additional context).
            duration: Requested duration in seconds.

        Returns:
            Dict with adjusted target values.
        """
        targets = ScorerTargets()

        # Combine prompt and genre for matching
        text = f"{prompt} {genre}".strip()
        if not text:
            return targets.to_dict()

        matched_categories = 0

        for pattern, adjustments in self._compiled_rules:
            if pattern.search(text):
                matched_categories += 1
                for field_name, delta in adjustments.items():
                    current = getattr(targets, field_name, None)
                    if current is not None:
                        setattr(targets, field_name, current + delta)

        # Duration-based adjustments: longer clips need more variety
        if duration > 30.0:
            targets.section_variety_target += 0.05
            targets.harmonic_progression_min += 0.5
        elif duration < 5.0:
            targets.section_variety_target -= 0.10
            targets.harmonic_progression_min -= 0.5

        # Clamp all values to valid bounds
        result = targets.to_dict()
        for key, (lo, hi) in _CLAMP_BOUNDS.items():
            if key in result:
                result[key] = max(lo, min(hi, result[key]))

        # Ensure onset_density_min <= onset_density_max
        if result["onset_density_min"] > result["onset_density_max"]:
            mid = (result["onset_density_min"] + result["onset_density_max"]) / 2
            result["onset_density_min"] = mid
            result["onset_density_max"] = mid

        if matched_categories > 0:
            logger.debug(
                "ConditionalTargets: {} categories matched for prompt '{}'",
                matched_categories,
                prompt[:80],
            )

        return result
