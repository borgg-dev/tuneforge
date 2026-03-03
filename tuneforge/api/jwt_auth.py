"""
JWT token creation and validation for TuneForge.

Provides access/refresh token management and password hashing
using industry-standard libraries (python-jose, passlib/bcrypt).
"""

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"


class TokenPayload(BaseModel):
    """Decoded JWT payload."""

    sub: str  # user_id
    exp: datetime
    type: str  # "access" or "refresh"


def create_access_token(
    user_id: str, secret: str, expire_minutes: int = 15
) -> str:
    """Create a short-lived access token."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "exp": now + timedelta(minutes=expire_minutes),
        "type": "access",
        "iat": now,
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def create_refresh_token(
    user_id: str, secret: str, expire_days: int = 7
) -> str:
    """Create a long-lived refresh token."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "exp": now + timedelta(days=expire_days),
        "type": "refresh",
        "iat": now,
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_token(token: str, secret: str) -> TokenPayload | None:
    """Decode and validate a JWT token.

    Returns the payload if valid, None if invalid or expired.
    """
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
        return TokenPayload(
            sub=payload["sub"],
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            type=payload.get("type", "access"),
        )
    except JWTError:
        return None


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its bcrypt hash."""
    return _pwd_context.verify(plain_password, hashed_password)
