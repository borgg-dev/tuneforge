"""
Validator service-token authentication.

Simple Bearer token auth for validator → API endpoints.
The token is a shared secret configured via TF_VALIDATOR_SERVICE_TOKEN.
"""

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer_scheme = HTTPBearer(auto_error=True)


async def require_validator_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> str:
    """Validate the validator service token.

    Returns the token on success.
    Raises 401 (missing), 403 (invalid), or 503 (not configured).
    """
    from tuneforge.settings import get_settings

    expected = get_settings().validator_service_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Validator API not configured (no service token set).",
        )

    if not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid validator token.",
        )

    return credentials.credentials
