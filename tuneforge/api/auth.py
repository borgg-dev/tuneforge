"""
Authentication for the TuneForge API.

Supports three auth modes:
1. JWT Bearer token (web frontend)
2. API key Bearer token (developer API, looked up by hash in DB)
3. Legacy env-var keys (existing TF_API_KEYS, backward-compatible)
"""

import hashlib
import os
import secrets
from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

from tuneforge.api.jwt_auth import decode_token

if TYPE_CHECKING:
    from tuneforge.api.database.models import UserRow

_bearer_scheme = HTTPBearer(auto_error=False)

_cached_env_keys: set[str] | None = None


def _load_env_keys() -> set[str]:
    """Load API keys from TF_API_KEYS environment variable."""
    global _cached_env_keys
    if _cached_env_keys is not None:
        return _cached_env_keys
    raw = os.environ.get("TF_API_KEYS", "")
    _cached_env_keys = {k.strip() for k in raw.split(",") if k.strip()}
    if _cached_env_keys:
        logger.info("Loaded {} API key(s) from environment", len(_cached_env_keys))
    return _cached_env_keys


def reload_keys() -> None:
    """Force reload of API keys from environment."""
    global _cached_env_keys
    _cached_env_keys = None
    _load_env_keys()


def verify_api_key(key: str) -> bool:
    """Check whether an API key is valid against env-var keys.

    Uses constant-time comparison to prevent timing attacks.
    """
    env_keys = _load_env_keys()
    return any(secrets.compare_digest(key, k) for k in env_keys)


def generate_api_key() -> str:
    """Generate a new random API key."""
    return f"tf_{secrets.token_urlsafe(32)}"


def hash_api_key(key: str) -> str:
    """Hash an API key with SHA-256 for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> "UserRow | None":
    """Extract authenticated user from JWT or API key.

    Returns None for anonymous access (no auth configured / legacy mode).
    Raises 401 if auth is required but credentials are invalid.

    Auth resolution order:
    1. Try as JWT access token → look up user by ID
    2. Try as API key → look up by hash in DB → find associated user
    3. Try legacy env-var key check → return None (anonymous)
    4. If no keys configured at all → return None (dev mode)
    """
    from tuneforge.api.server import app_state

    settings = app_state.settings
    db = app_state.db

    if credentials is None:
        # No credentials provided
        env_keys = _load_env_keys()
        if not env_keys:
            return None  # No auth configured — anonymous access
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # 1. Try JWT
    payload = decode_token(token, settings.jwt_secret)
    if payload is not None and payload.type == "access":
        if db is None:
            return None
        from tuneforge.api.database import get_user_by_id

        user = await get_user_by_id(db, payload.sub)
        if user is not None and user.is_active:
            return user
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # If it looks like a JWT (has 2 dots) but decode failed → expired/invalid
    if token.count(".") == 2:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired or invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 2. Try API key (DB lookup by hash)
    if db is not None:
        from tuneforge.api.database import get_api_key_by_hash, get_user_by_id, update_api_key_last_used

        key_hash = hash_api_key(token)
        api_key_row = await get_api_key_by_hash(db, key_hash)
        if api_key_row is not None:
            await update_api_key_last_used(db, api_key_row.id)
            user = await get_user_by_id(db, api_key_row.user_id)
            if user is not None and user.is_active:
                return user

    # 3. Legacy env-var key
    if verify_api_key(token):
        return None  # Valid legacy key, but no user object

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid credentials.",
    )


async def require_user(
    user: "UserRow | None" = Depends(get_current_user),
) -> "UserRow":
    """Dependency that requires an authenticated user. Raises 401 if None."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    return user


async def require_admin(
    user: "UserRow" = Depends(require_user),
) -> "UserRow":
    """Dependency that requires an admin user. Raises 403 if not admin."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return user


# Legacy compatibility — kept for existing routes that use get_api_key
async def get_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """Legacy FastAPI dependency for API key validation.

    Returns the API key string on success, "anonymous" if no keys configured.
    """
    env_keys = _load_env_keys()
    if not env_keys:
        return "anonymous"

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide a Bearer token in the Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_api_key(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    return credentials.credentials
