"""
Progressive difficulty system for TuneForge.

Increases challenge complexity as the network's overall quality improves,
tracked via a slow-moving EMA of per-round aggregate scores.

Usage::

    manager = ProgressiveDifficultyManager(state_path="./progressive_state.json")
    manager.load_state()

    # After each validation round:
    manager.update_network_quality([0.45, 0.52, 0.61, 0.38])
    difficulty = manager.get_difficulty_level()
    # difficulty["min_duration"] == 8.0
    # difficulty["quality_floor"] == 0.42
    # difficulty["prompt_complexity"] == "moderate"
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger


_STATE_VERSION = 1

# ---------------------------------------------------------------------------
# Difficulty curve anchors
#
# Each anchor maps a network_quality_ema value to a set of difficulty
# parameters.  Values between anchors are linearly interpolated.
# ---------------------------------------------------------------------------

_DIFFICULTY_ANCHORS: list[tuple[float, dict[str, float]]] = [
    # (ema_threshold, {param: value, ...})
    (0.00, {
        "min_duration": 5.0,
        "max_duration": 15.0,
        "quality_floor": 0.40,
        "attribute_requirements": 0.0,  # none required
        "prompt_complexity_level": 0.0,
    }),
    (0.30, {
        "min_duration": 8.0,
        "max_duration": 30.0,
        "quality_floor": 0.42,
        "attribute_requirements": 0.3,
        "prompt_complexity_level": 0.3,
    }),
    (0.50, {
        "min_duration": 15.0,
        "max_duration": 60.0,
        "quality_floor": 0.47,
        "attribute_requirements": 0.6,
        "prompt_complexity_level": 0.5,
    }),
    (0.65, {
        "min_duration": 25.0,
        "max_duration": 120.0,
        "quality_floor": 0.52,
        "attribute_requirements": 0.8,
        "prompt_complexity_level": 0.7,
    }),
    (0.80, {
        "min_duration": 40.0,
        "max_duration": 180.0,
        "quality_floor": 0.57,
        "attribute_requirements": 1.0,
        "prompt_complexity_level": 0.9,
    }),
    (1.00, {
        "min_duration": 60.0,
        "max_duration": 180.0,
        "quality_floor": 0.60,
        "attribute_requirements": 1.0,
        "prompt_complexity_level": 1.0,
    }),
]

# Prompt complexity level descriptions
_PROMPT_COMPLEXITY_TIERS: list[tuple[float, str]] = [
    (0.0, "simple"),       # e.g. "ambient music"
    (0.25, "basic"),       # e.g. "upbeat electronic track in C major"
    (0.50, "moderate"),    # e.g. "relaxing jazz piano with brushed drums, 90 BPM, Bb minor"
    (0.75, "detailed"),    # e.g. multi-attribute with tempo, key, instruments, mood
    (1.0, "complex"),      # e.g. full specification with structure, dynamics, style
]

# Attribute requirement level descriptions
_ATTRIBUTE_TIERS: list[tuple[float, str]] = [
    (0.0, "none"),                 # no attribute verification required
    (0.3, "genre_only"),           # genre must match
    (0.6, "genre_and_tempo"),      # genre + tempo must match
    (0.8, "genre_tempo_key"),      # genre + tempo + key must match
    (1.0, "full"),                 # genre + tempo + key + instruments all required
]


def _tier_label(level: float, tiers: list[tuple[float, str]]) -> str:
    """Map a continuous level [0, 1] to the closest tier label."""
    best_label = tiers[0][1]
    best_dist = abs(level - tiers[0][0])
    for threshold, label in tiers[1:]:
        dist = abs(level - threshold)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def _interpolate_anchors(ema: float) -> dict[str, float]:
    """Interpolate difficulty parameters from the anchor table."""
    ema = float(np.clip(ema, 0.0, 1.0))

    # Find bounding anchors
    lower = _DIFFICULTY_ANCHORS[0]
    upper = _DIFFICULTY_ANCHORS[-1]

    for i in range(len(_DIFFICULTY_ANCHORS) - 1):
        if _DIFFICULTY_ANCHORS[i][0] <= ema <= _DIFFICULTY_ANCHORS[i + 1][0]:
            lower = _DIFFICULTY_ANCHORS[i]
            upper = _DIFFICULTY_ANCHORS[i + 1]
            break

    # Linear interpolation
    span = upper[0] - lower[0]
    if span < 1e-8:
        t = 0.0
    else:
        t = (ema - lower[0]) / span

    result: dict[str, float] = {}
    all_keys = set(lower[1]) | set(upper[1])
    for key in all_keys:
        lo_val = lower[1].get(key, 0.0)
        hi_val = upper[1].get(key, 0.0)
        result[key] = lo_val + t * (hi_val - lo_val)

    return result


class ProgressiveDifficultyManager:
    """Manage progressive difficulty scaling for challenges.

    Tracks the network's overall quality via a slow-moving EMA and
    maps it to difficulty parameters that control challenge generation.
    State is persisted to JSON with atomic writes and backup rotation.
    """

    def __init__(
        self,
        state_path: str = "./progressive_state.json",
        initial_quality: float = 0.3,
        quality_alpha: float = 0.05,
    ) -> None:
        self._state_path = state_path
        self._network_quality_ema = float(np.clip(initial_quality, 0.0, 1.0))
        self._quality_alpha = quality_alpha
        self._round_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def network_quality_ema(self) -> float:
        """Current network quality EMA value."""
        return self._network_quality_ema

    @property
    def round_count(self) -> int:
        """Total rounds processed."""
        return self._round_count

    def update_network_quality(self, round_scores: list[float]) -> None:
        """Update the network quality EMA from the latest round scores.

        Args:
            round_scores: List of composite scores for all miners in
                the round.  Invalid/NaN values are filtered out.
        """
        if not round_scores:
            logger.debug("ProgressiveDifficulty: empty round_scores — skipping update")
            return

        # Filter invalid values
        valid = [
            float(np.clip(np.nan_to_num(s, nan=0.0), 0.0, 1.0))
            for s in round_scores
        ]
        if not valid:
            return

        # Use the median to resist outliers
        round_quality = float(np.median(valid))

        self._network_quality_ema = (
            self._quality_alpha * round_quality
            + (1.0 - self._quality_alpha) * self._network_quality_ema
        )
        self._network_quality_ema = float(np.clip(self._network_quality_ema, 0.0, 1.0))
        self._round_count += 1

        logger.debug(
            "ProgressiveDifficulty: round_quality={:.4f}, ema={:.4f}, rounds={}",
            round_quality,
            self._network_quality_ema,
            self._round_count,
        )

    def get_difficulty_level(self) -> dict:
        """Return difficulty parameters based on current network quality.

        Returns:
            Dict with keys:

            * ``min_duration`` -- minimum challenge duration in seconds
            * ``max_duration`` -- maximum challenge duration in seconds
            * ``quality_floor`` -- baseline for weight setting
            * ``prompt_complexity`` -- human-readable complexity tier
            * ``prompt_complexity_level`` -- numeric level [0, 1]
            * ``attribute_requirements`` -- human-readable requirement tier
            * ``attribute_requirements_level`` -- numeric level [0, 1]
            * ``network_quality_ema`` -- current EMA value
            * ``round_count`` -- total rounds processed
        """
        params = _interpolate_anchors(self._network_quality_ema)

        complexity_level = params.get("prompt_complexity_level", 0.0)
        attr_level = params.get("attribute_requirements", 0.0)

        return {
            "min_duration": round(params.get("min_duration", 5.0), 1),
            "max_duration": round(params.get("max_duration", 15.0), 1),
            "quality_floor": round(params.get("quality_floor", 0.40), 4),
            "prompt_complexity": _tier_label(complexity_level, _PROMPT_COMPLEXITY_TIERS),
            "prompt_complexity_level": round(complexity_level, 3),
            "attribute_requirements": _tier_label(attr_level, _ATTRIBUTE_TIERS),
            "attribute_requirements_level": round(attr_level, 3),
            "network_quality_ema": round(self._network_quality_ema, 6),
            "round_count": self._round_count,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str | None = None) -> None:
        """Persist progressive difficulty state to JSON.

        Uses atomic write via temp file + rename, with .bak backup
        of the existing file (mirrors leaderboard.py pattern).

        Args:
            path: Override state file path.  Defaults to ``self._state_path``.
        """
        dest = Path(path or self._state_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": _STATE_VERSION,
            "network_quality_ema": self._network_quality_ema,
            "quality_alpha": self._quality_alpha,
            "round_count": self._round_count,
        }

        # Create backup of existing file
        if dest.exists():
            bak = dest.with_suffix(dest.suffix + ".bak")
            try:
                bak.write_text(dest.read_text())
            except Exception as exc:
                logger.warning("Failed to create progressive state backup: {}", exc)

        # Atomic write
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=str(dest.parent), suffix=".tmp", delete=False,
            ) as tmp:
                json.dump(data, tmp, indent=2)
                tmp_path = tmp.name
            Path(tmp_path).rename(dest)
            logger.info(
                "Progressive difficulty state saved to {} (ema={:.4f}, rounds={})",
                dest, self._network_quality_ema, self._round_count,
            )
        except Exception as exc:
            logger.error("Failed to save progressive difficulty state: {}", exc)

    def load_state(self, path: str | None = None) -> bool:
        """Load progressive difficulty state from JSON.

        Falls back to .bak if the primary file is corrupt.

        Args:
            path: Override state file path.  Defaults to ``self._state_path``.

        Returns:
            True if state was loaded successfully, False otherwise.
        """
        dest = Path(path or self._state_path)
        bak = dest.with_suffix(dest.suffix + ".bak")

        for source_name, filepath in [("primary", dest), ("backup", bak)]:
            if not filepath.exists():
                continue
            try:
                raw = filepath.read_text()
                data = json.loads(raw)
                return self._apply_loaded_state(data, source_name)
            except Exception as exc:
                logger.warning(
                    "Failed to load progressive state from {} ({}): {}",
                    filepath, source_name, exc,
                )

        logger.info(
            "No progressive state file found at {} — starting fresh", dest,
        )
        return False

    def _apply_loaded_state(self, data: dict, source: str) -> bool:
        """Validate and apply loaded state data.

        Args:
            data: Parsed JSON state data.
            source: Description of the file source.

        Returns:
            True if the state was applied.
        """
        if not isinstance(data, dict):
            logger.warning("Progressive state ({}) is not a dict — skipping", source)
            return False

        version = data.get("version", 0)
        if version != _STATE_VERSION:
            logger.warning(
                "Progressive state ({}) version {} != {} — skipping",
                source, version, _STATE_VERSION,
            )
            return False

        try:
            ema = float(data["network_quality_ema"])
            round_count = int(data["round_count"])
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Progressive state ({}) missing/invalid fields: {}", source, exc)
            return False

        if not (0.0 <= ema <= 1.0):
            logger.warning(
                "Progressive state ({}) EMA {} out of range — clamping",
                source, ema,
            )
            ema = float(np.clip(ema, 0.0, 1.0))

        if round_count < 0:
            logger.warning("Progressive state ({}) negative round_count — resetting to 0", source)
            round_count = 0

        # Optionally restore alpha if present
        alpha = data.get("quality_alpha")
        if alpha is not None:
            try:
                alpha = float(alpha)
                if 0.0 < alpha < 1.0:
                    self._quality_alpha = alpha
            except (ValueError, TypeError):
                pass

        self._network_quality_ema = ema
        self._round_count = round_count

        logger.info(
            "Loaded progressive state from {} ({}): ema={:.4f}, rounds={}",
            source, source, ema, round_count,
        )
        return True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a summary dict for logging / API exposure."""
        difficulty = self.get_difficulty_level()
        return {
            "network_quality_ema": self._network_quality_ema,
            "round_count": self._round_count,
            "quality_alpha": self._quality_alpha,
            "difficulty": difficulty,
        }
