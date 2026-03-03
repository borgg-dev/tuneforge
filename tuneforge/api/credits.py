"""
Credit management service for TuneForge.

Handles credit balance checks, daily resets, spending (charge on success),
and transaction history.
"""

from datetime import datetime, timezone
from enum import IntEnum

from loguru import logger
from sqlalchemy import select, update

from tuneforge.api.database.engine import Database
from tuneforge.api.database.models import CreditRow, CreditTransactionRow


class CreditCost(IntEnum):
    """Credit costs for various operations."""

    GENERATION = 5  # Credits per generation


TIER_DAILY_ALLOWANCE = {
    "free": 50,
    "pro": 0,  # Pro uses monthly credits (Phase 2)
    "premier": 0,  # Premier uses monthly credits (Phase 2)
}


class CreditService:
    """Service for managing user credits."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def get_balance(self, user_id: str) -> CreditRow | None:
        """Get current credit balance, auto-resetting daily if needed."""
        async with self._db.session() as session:
            result = await session.execute(
                select(CreditRow).where(CreditRow.user_id == user_id)
            )
            credit = result.scalar_one_or_none()

        if credit is None:
            return None

        # Check if daily reset is due
        now = datetime.now(timezone.utc)
        if credit.last_daily_reset.date() < now.date():
            credit = await self._reset_daily(user_id, credit)

        return credit

    async def has_sufficient_credits(self, user_id: str, amount: int) -> bool:
        """Check if user has enough credits (read-only, no deduction)."""
        credit = await self.get_balance(user_id)
        if credit is None:
            return False
        return credit.daily_balance >= amount

    async def spend_credits(
        self, user_id: str, amount: int, reference_id: str
    ) -> bool:
        """Deduct credits after a successful generation.

        Returns True if deduction succeeded, False if insufficient balance.
        """
        now = datetime.now(timezone.utc)

        async with self._db.transaction() as session:
            result = await session.execute(
                select(CreditRow).where(CreditRow.user_id == user_id)
            )
            credit = result.scalar_one_or_none()

            if credit is None:
                return False

            # Auto-reset daily if needed
            if credit.last_daily_reset.date() < now.date():
                credit.daily_balance = credit.daily_allowance
                credit.last_daily_reset = now

            if credit.daily_balance < amount:
                return False

            credit.daily_balance -= amount

            tx = CreditTransactionRow(
                user_id=user_id,
                amount=-amount,
                tx_type="spend",
                reference_id=reference_id,
                description=f"Generation {reference_id}",
            )
            session.add(tx)

        logger.debug("Charged {} credits for user {} (ref={})", amount, user_id, reference_id)
        return True

    async def get_next_reset(self) -> datetime:
        """Get the next daily reset time (midnight UTC)."""
        now = datetime.now(timezone.utc)
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if tomorrow <= now:
            from datetime import timedelta

            tomorrow += timedelta(days=1)
        return tomorrow

    async def _reset_daily(self, user_id: str, credit: CreditRow) -> CreditRow:
        """Reset daily balance to allowance."""
        now = datetime.now(timezone.utc)
        async with self._db.transaction() as session:
            await session.execute(
                update(CreditRow)
                .where(CreditRow.user_id == user_id)
                .values(
                    daily_balance=credit.daily_allowance,
                    last_daily_reset=now,
                )
            )
            tx = CreditTransactionRow(
                user_id=user_id,
                amount=credit.daily_allowance,
                tx_type="daily_grant",
                description="Daily credit reset",
            )
            session.add(tx)

        credit.daily_balance = credit.daily_allowance
        credit.last_daily_reset = now
        logger.debug("Reset daily credits for user {}", user_id)
        return credit
