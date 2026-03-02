"""
API key authentication for the TuneForge API.

Supports keys from the TF_API_KEYS environment variable (comma-separated)
and optional database-backed key storage.
"""

import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger


_bearer_scheme = HTTPBearer(auto_error=False)

_cached_keys: set[str] | None = None


def _load_env_keys() -> set[str]:
    """Load API keys from TF_API_KEYS environment variable."""
    global _cached_keys
    if _cached_keys is not None:
        return _cached_keys
    raw = os.environ.get("TF_API_KEYS", "")
    _cached_keys = {k.strip() for k in raw.split(",") if k.strip()}
    if _cached_keys:
        logger.info("Loaded {} API key(s) from environment", len(_cached_keys))
    return _cached_keys


def reload_keys() -> None:
    """Force reload of API keys from environment."""
    global _cached_keys
    _cached_keys = None
    _load_env_keys()


def verify_api_key(key: str) -> bool:
    """Check whether an API key is valid.

    Uses constant-time comparison to prevent timing attacks.
    """
    env_keys = _load_env_keys()
    return any(secrets.compare_digest(key, k) for k in env_keys)


def generate_api_key() -> str:
    """Generate a new random API key."""
    return f"tf_{secrets.token_urlsafe(32)}"


async def get_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """FastAPI dependency that extracts and validates the bearer token.

    When TF_API_KEYS is empty (no keys configured), authentication is
    disabled and all requests are allowed through with a placeholder key.
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
