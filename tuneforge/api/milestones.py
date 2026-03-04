"""
Annotation milestone reward definitions for TuneForge.

Milestones are defined as constants since they change rarely.
Each milestone is earned once based on cumulative trusted annotation count.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MilestoneDef:
    """Definition of an annotation milestone."""

    key: str
    label: str
    threshold: int
    credits: int
    grants_pro_days: int


MILESTONES: tuple[MilestoneDef, ...] = (
    MilestoneDef(key="first_listen", label="First Listen", threshold=5, credits=10, grants_pro_days=0),
    MilestoneDef(key="tuning_in", label="Tuning In", threshold=15, credits=25, grants_pro_days=0),
    MilestoneDef(key="contributor", label="Contributor", threshold=40, credits=75, grants_pro_days=0),
    MilestoneDef(key="trusted_ear", label="Trusted Ear", threshold=80, credits=150, grants_pro_days=0),
    MilestoneDef(key="power_annotator", label="Power Annotator", threshold=150, credits=300, grants_pro_days=0),
    MilestoneDef(key="gold_ear", label="Gold Ear", threshold=250, credits=500, grants_pro_days=30),
)

MILESTONE_BY_KEY: dict[str, MilestoneDef] = {m.key: m for m in MILESTONES}


def get_next_milestone(trusted_count: int, claimed_keys: set[str]) -> MilestoneDef | None:
    """Return the next unclaimed milestone the user is working toward, or None."""
    for m in MILESTONES:
        if m.key not in claimed_keys:
            return m
    return None


def get_newly_unlocked(trusted_count: int, claimed_keys: set[str]) -> list[MilestoneDef]:
    """Return milestones the user qualifies for but hasn't claimed yet."""
    return [m for m in MILESTONES if m.threshold <= trusted_count and m.key not in claimed_keys]


# ---------------------------------------------------------------------------
# Recurring reward system (post-milestone)
# ---------------------------------------------------------------------------

RECURRING_BATCH_SIZE: int = 20
RECURRING_BASE_CREDITS: int = 30
DAILY_STREAK_THRESHOLD: int = 10
TOTAL_MILESTONES: int = len(MILESTONES)


@dataclass(frozen=True)
class StreakTier:
    """Definition of a streak multiplier tier."""

    min_days: int
    max_days: int | None  # None = unbounded
    multiplier: float
    label: str


STREAK_TIERS: tuple[StreakTier, ...] = (
    StreakTier(min_days=1, max_days=2, multiplier=1.0, label="Base"),
    StreakTier(min_days=3, max_days=6, multiplier=1.25, label="Warming Up"),
    StreakTier(min_days=7, max_days=13, multiplier=1.5, label="On Fire"),
    StreakTier(min_days=14, max_days=29, multiplier=1.75, label="Blazing"),
    StreakTier(min_days=30, max_days=None, multiplier=2.0, label="Legendary"),
)


def get_streak_tier(streak_days: int) -> StreakTier:
    """Return the streak tier for the given number of consecutive days."""
    for tier in reversed(STREAK_TIERS):
        if streak_days >= tier.min_days:
            return tier
    return STREAK_TIERS[0]


def get_next_streak_tier(streak_days: int) -> StreakTier | None:
    """Return the next tier the user hasn't reached yet, or None if at max."""
    current = get_streak_tier(streak_days)
    for tier in STREAK_TIERS:
        if tier.min_days > current.min_days:
            return tier
    return None


def compute_recurring_credits(streak_days: int) -> int:
    """Compute credits for one recurring batch at the given streak level."""
    tier = get_streak_tier(streak_days)
    return int(RECURRING_BASE_CREDITS * tier.multiplier)


def is_recurring_eligible(claimed_keys: set[str]) -> bool:
    """Check if a user has claimed all milestones and is eligible for recurring rewards."""
    return len(claimed_keys) >= TOTAL_MILESTONES
