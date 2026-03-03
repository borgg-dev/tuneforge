"""
Credit balance and history endpoints for TuneForge.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from tuneforge.api.auth import require_user
from tuneforge.api.database.models import UserRow
from tuneforge.api.models import (
    CreditBalance,
    CreditHistoryResponse,
    CreditTransaction,
)

router = APIRouter(prefix="/api/v1/credits", tags=["credits"])


@router.get("/balance", response_model=CreditBalance)
async def get_balance(user: UserRow = Depends(require_user)) -> CreditBalance:
    """Get current credit balance."""
    from tuneforge.api.server import app_state

    credit = await app_state.credit_service.get_balance(user.id)
    if credit is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credit record not found.",
        )

    next_reset = await app_state.credit_service.get_next_reset()

    return CreditBalance(
        daily_balance=credit.daily_balance,
        daily_allowance=credit.daily_allowance,
        next_reset=next_reset,
    )


@router.get("/history", response_model=CreditHistoryResponse)
async def get_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    user: UserRow = Depends(require_user),
) -> CreditHistoryResponse:
    """Get credit transaction history."""
    from tuneforge.api.database import get_credit_transactions, total_pages
    from tuneforge.api.server import app_state

    transactions, total = await get_credit_transactions(
        app_state.db, user.id, page=page, page_size=page_size
    )

    return CreditHistoryResponse(
        transactions=[
            CreditTransaction(
                id=tx.id,
                amount=tx.amount,
                tx_type=tx.tx_type,
                reference_id=tx.reference_id,
                description=tx.description,
                created_at=tx.created_at,
            )
            for tx in transactions
        ],
        total=total,
        page=page,
        pages=total_pages(total, page_size),
    )
