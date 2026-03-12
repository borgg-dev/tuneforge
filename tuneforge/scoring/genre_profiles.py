"""
Genre-aware scoring profiles for TuneForge.

Maps each genre to a family with appropriate bell-curve targets for
audio quality, musicality, and production metrics.  Genres within the
same family share identical targets so that ambient music is not
penalized for lacking beats, electronic music is not penalized for
narrow dynamic range, and classical music is not penalized for wide
dynamics.

Usage::

    profile = get_genre_profile("ambient")
    target = profile["dynamic_range_target"]  # 12.0 instead of 10.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenreProfile:
    """Per-genre parameter overrides for scoring bell curves."""

    family: str

    # AudioQualityScorer targets
    dynamic_range_target: float = 10.0  # dB — bell curve center
    onset_density_ceiling: float = 4.0  # onsets/sec — score=1.0 threshold

    # MusicalityScorer targets
    pitch_range_tolerance: float = 1.0  # multiplier on pitch-std penalty
    rhythmic_groove_floor: float = 0.0  # minimum groove score (ambient = 0.5)
    arrangement_contrast_target: float = 0.15  # spectral centroid CV target

    # ProductionQualityScorer targets
    spectral_balance_cv_target: float = 1.0  # coefficient of variation center
    loudness_std_target: float = 5.0  # dB — bell curve center (short-term LUFS std)
    dynamic_expressiveness_target: float = 0.05  # normalized derivative std

    # LUFS integrated loudness target (ITU-R BS.1770-4)
    lufs_target: float = -16.0  # streaming standard default

    # StructuralCompletenessScorer targets
    structural_section_target: float = 4.0  # expected sections per 30s
    structural_variety_target: float = 0.35  # pairwise section dissimilarity

    # VocalQualityScorer
    vocal_expected: bool = True  # whether vocals are expected for this genre


# ---------------------------------------------------------------------------
# Genre family definitions
# ---------------------------------------------------------------------------

_ELECTRONIC = GenreProfile(
    family="electronic",
    dynamic_range_target=6.0,
    onset_density_ceiling=8.0,
    pitch_range_tolerance=1.5,
    rhythmic_groove_floor=0.0,
    arrangement_contrast_target=0.10,
    spectral_balance_cv_target=1.2,
    loudness_std_target=3.0,
    dynamic_expressiveness_target=0.04,
    lufs_target=-14.0,
    structural_section_target=4.0,
    structural_variety_target=0.30,
    vocal_expected=False,
)

_ROCK = GenreProfile(
    family="rock",
    dynamic_range_target=8.0,
    onset_density_ceiling=6.0,
    pitch_range_tolerance=1.5,
    rhythmic_groove_floor=0.0,
    arrangement_contrast_target=0.20,
    spectral_balance_cv_target=1.1,
    loudness_std_target=4.0,
    dynamic_expressiveness_target=0.06,
    lufs_target=-14.0,
    structural_section_target=4.0,
    structural_variety_target=0.35,
    vocal_expected=True,
)

_CLASSICAL = GenreProfile(
    family="classical-cinematic",
    dynamic_range_target=14.0,
    onset_density_ceiling=3.0,
    pitch_range_tolerance=2.5,
    rhythmic_groove_floor=0.3,
    arrangement_contrast_target=0.25,
    spectral_balance_cv_target=0.8,
    loudness_std_target=7.0,
    dynamic_expressiveness_target=0.07,
    lufs_target=-23.0,
    structural_section_target=3.0,
    structural_variety_target=0.40,
    vocal_expected=False,
)

_AMBIENT = GenreProfile(
    family="ambient",
    dynamic_range_target=12.0,
    onset_density_ceiling=1.0,
    pitch_range_tolerance=1.0,
    rhythmic_groove_floor=0.5,
    arrangement_contrast_target=0.10,
    spectral_balance_cv_target=0.7,
    loudness_std_target=6.0,
    dynamic_expressiveness_target=0.03,
    lufs_target=-20.0,
    structural_section_target=2.0,
    structural_variety_target=0.20,
    vocal_expected=False,
)

_HIPHOP = GenreProfile(
    family="hip-hop",
    dynamic_range_target=7.0,
    onset_density_ceiling=5.0,
    pitch_range_tolerance=1.0,
    rhythmic_groove_floor=0.0,
    arrangement_contrast_target=0.12,
    spectral_balance_cv_target=1.3,
    loudness_std_target=3.5,
    dynamic_expressiveness_target=0.04,
    lufs_target=-14.0,
    structural_section_target=4.0,
    structural_variety_target=0.30,
    vocal_expected=True,
)

_JAZZ = GenreProfile(
    family="jazz-blues",
    dynamic_range_target=12.0,
    onset_density_ceiling=5.0,
    pitch_range_tolerance=2.0,
    rhythmic_groove_floor=0.0,
    arrangement_contrast_target=0.20,
    spectral_balance_cv_target=0.9,
    loudness_std_target=6.0,
    dynamic_expressiveness_target=0.06,
    lufs_target=-18.0,
    structural_section_target=3.0,
    structural_variety_target=0.35,
    vocal_expected=True,
)

_FOLK = GenreProfile(
    family="folk-acoustic",
    dynamic_range_target=11.0,
    onset_density_ceiling=4.0,
    pitch_range_tolerance=1.5,
    rhythmic_groove_floor=0.1,
    arrangement_contrast_target=0.15,
    spectral_balance_cv_target=0.8,
    loudness_std_target=5.0,
    dynamic_expressiveness_target=0.05,
    lufs_target=-18.0,
    structural_section_target=4.0,
    structural_variety_target=0.35,
    vocal_expected=True,
)

_GROOVE = GenreProfile(
    family="groove-soul",
    dynamic_range_target=8.0,
    onset_density_ceiling=6.0,
    pitch_range_tolerance=1.5,
    rhythmic_groove_floor=0.0,
    arrangement_contrast_target=0.15,
    spectral_balance_cv_target=1.0,
    loudness_std_target=4.0,
    dynamic_expressiveness_target=0.05,
    lufs_target=-14.0,
    structural_section_target=5.0,
    structural_variety_target=0.35,
    vocal_expected=True,
)

_POP = GenreProfile(
    family="pop",
    dynamic_range_target=8.0,
    onset_density_ceiling=5.0,
    pitch_range_tolerance=1.5,
    rhythmic_groove_floor=0.0,
    arrangement_contrast_target=0.18,
    spectral_balance_cv_target=0.9,
    loudness_std_target=4.0,
    dynamic_expressiveness_target=0.05,
    lufs_target=-14.0,
    structural_section_target=5.0,
    structural_variety_target=0.35,
    vocal_expected=True,
)

_WORLD = GenreProfile(
    family="world",
    dynamic_range_target=10.0,
    onset_density_ceiling=6.0,
    pitch_range_tolerance=2.0,
    rhythmic_groove_floor=0.0,
    arrangement_contrast_target=0.15,
    spectral_balance_cv_target=1.0,
    loudness_std_target=5.0,
    dynamic_expressiveness_target=0.05,
    lufs_target=-16.0,
    structural_section_target=3.0,
    structural_variety_target=0.35,
    vocal_expected=True,
)

# Default profile for unknown genres — uses the original hardcoded values
_DEFAULT = GenreProfile(family="default")


# ---------------------------------------------------------------------------
# Genre → family mapping
# ---------------------------------------------------------------------------

_GENRE_MAP: dict[str, GenreProfile] = {
    # Electronic family
    "electronic": _ELECTRONIC,
    "synthwave": _ELECTRONIC,
    "drum-and-bass": _ELECTRONIC,
    "deep-house": _ELECTRONIC,
    "grime": _ELECTRONIC,
    "vaporwave": _ELECTRONIC,
    # Rock family
    "rock": _ROCK,
    "metal": _ROCK,
    "post-rock": _ROCK,
    "shoegaze": _ROCK,
    "progressive-rock": _ROCK,
    "math-rock": _ROCK,
    "garage-rock": _ROCK,
    "psychedelic": _ROCK,
    # Classical / cinematic family
    "classical": _CLASSICAL,
    "cinematic": _CLASSICAL,
    "chamber-pop": _CLASSICAL,
    # Ambient family
    "ambient": _AMBIENT,
    "dark-ambient": _AMBIENT,
    "downtempo": _AMBIENT,
    # Hip-hop family
    "hip-hop": _HIPHOP,
    "trip-hop": _HIPHOP,
    "lo-fi": _HIPHOP,
    # Jazz / blues family
    "jazz": _JAZZ,
    "blues": _JAZZ,
    "bossa-nova": _JAZZ,
    "acid-jazz": _JAZZ,
    "neo-soul": _JAZZ,
    # Folk / acoustic family
    "folk": _FOLK,
    "country": _FOLK,
    "indie-folk": _FOLK,
    # Groove / soul family
    "soul": _GROOVE,
    "funk": _GROOVE,
    "disco": _GROOVE,
    "r&b": _GROOVE,
    "reggae": _GROOVE,
    "afrobeat": _GROOVE,
    # Pop
    "pop": _POP,
    # World / Latin
    "world": _WORLD,
    "latin": _WORLD,
}


def get_genre_profile(genre: str) -> GenreProfile:
    """
    Look up the scoring profile for a genre.

    Args:
        genre: Genre string (case-insensitive, e.g. ``"ambient"``).

    Returns:
        GenreProfile with per-genre parameter overrides.
        Falls back to default profile for unknown genres.
    """
    return _GENRE_MAP.get(genre.lower().strip(), _DEFAULT)


def list_genre_families() -> dict[str, list[str]]:
    """Return a mapping of family name → list of genres for debugging."""
    families: dict[str, list[str]] = {}
    for genre, profile in _GENRE_MAP.items():
        families.setdefault(profile.family, []).append(genre)
    return families
