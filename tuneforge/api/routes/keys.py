"""
API key management endpoints for TuneForge.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from tuneforge.api.auth import generate_api_key, hash_api_key, require_user
from tuneforge.api.database.models import UserRow
from tuneforge.api.models import ApiKeyCreated, ApiKeyInfo, CreateApiKeyRequest

router = APIRouter(prefix="/api/v1/keys", tags=["api-keys"])


@router.get("/", response_model=list[ApiKeyInfo])
async def list_keys(user: UserRow = Depends(require_user)) -> list[ApiKeyInfo]:
    """List the user's API keys (prefix only, not full key)."""
    from tuneforge.api.database import get_user_api_keys
    from tuneforge.api.server import app_state

    keys = await get_user_api_keys(app_state.db, user.id)
    return [
        ApiKeyInfo(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            last_used_at=k.last_used_at,
            created_at=k.created_at,
        )
        for k in keys
    ]


@router.post("/", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
async def create_key(
    body: CreateApiKeyRequest,
    user: UserRow = Depends(require_user),
) -> ApiKeyCreated:
    """Create a new API key.

    The full key is returned only once — store it securely.
    """
    from tuneforge.api.database import create_api_key
    from tuneforge.api.server import app_state

    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)
    key_prefix = raw_key[:11] + "..."  # "tf_XXXXXXX..."

    row = await create_api_key(
        app_state.db,
        user_id=user.id,
        key_prefix=key_prefix,
        key_hash=key_hash,
        name=body.name,
    )

    logger.info("API key created for user {} ({})", user.id, key_prefix)

    return ApiKeyCreated(
        id=row.id,
        name=row.name,
        key=raw_key,
        key_prefix=key_prefix,
        created_at=row.created_at,
    )


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_key(
    key_id: str,
    user: UserRow = Depends(require_user),
) -> None:
    """Revoke an API key."""
    from tuneforge.api.database import revoke_api_key
    from tuneforge.api.server import app_state

    revoked = await revoke_api_key(app_state.db, key_id, user.id)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found.",
        )

    logger.info("API key {} revoked by user {}", key_id, user.id)
