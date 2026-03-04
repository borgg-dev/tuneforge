"""
Authentication endpoints for the TuneForge API.

Provides registration, login, token refresh, logout, and profile management.
"""

from fastapi import APIRouter, Depends, HTTPException, Response, status
from loguru import logger

from tuneforge.api.auth import require_user
from tuneforge.api.database.models import UserRow
from tuneforge.api.jwt_auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from tuneforge.api.models import (
    AuthResponse,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserProfile,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _user_to_profile(user: UserRow) -> UserProfile:
    return UserProfile(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
        plan_tier=user.plan_tier,
        is_admin=user.is_admin,
        created_at=user.created_at,
    )


@router.post("/register", response_model=AuthResponse)
async def register(body: RegisterRequest) -> AuthResponse:
    """Register a new user with email and password."""
    from tuneforge.api.database import create_credit, create_user, get_user_by_email
    from tuneforge.api.server import app_state

    db = app_state.db
    settings = app_state.settings

    # Check if email is already taken
    existing = await get_user_by_email(db, body.email)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # Create user
    pw_hash = hash_password(body.password)
    user = await create_user(
        db,
        email=body.email,
        password_hash=pw_hash,
        display_name=body.display_name,
    )

    # Create initial credit balance
    await create_credit(db, user_id=user.id, daily_allowance=50)

    # Generate tokens
    access_token = create_access_token(
        user.id, settings.jwt_secret, settings.jwt_access_expire_minutes
    )
    refresh_token = create_refresh_token(
        user.id, settings.jwt_secret, settings.jwt_refresh_expire_days
    )

    logger.info("User registered: {} ({})", user.id, body.email)

    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt_access_expire_minutes * 60,
        user=_user_to_profile(user),
    )


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest) -> AuthResponse:
    """Log in with email and password."""
    from tuneforge.api.database import get_user_by_email
    from tuneforge.api.server import app_state

    db = app_state.db
    settings = app_state.settings

    user = await get_user_by_email(db, body.email)
    if user is None or user.password_hash is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled.",
        )

    access_token = create_access_token(
        user.id, settings.jwt_secret, settings.jwt_access_expire_minutes
    )
    refresh_token = create_refresh_token(
        user.id, settings.jwt_secret, settings.jwt_refresh_expire_days
    )

    logger.info("User logged in: {}", user.id)

    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt_access_expire_minutes * 60,
        user=_user_to_profile(user),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest) -> TokenResponse:
    """Refresh the access token using a refresh token."""
    from tuneforge.api.database import get_user_by_id
    from tuneforge.api.server import app_state

    settings = app_state.settings

    payload = decode_token(body.refresh_token, settings.jwt_secret)
    if payload is None or payload.type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )

    user = await get_user_by_id(app_state.db, payload.sub)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
        )

    new_access = create_access_token(
        user.id, settings.jwt_secret, settings.jwt_access_expire_minutes
    )

    return TokenResponse(
        access_token=new_access,
        expires_in=settings.jwt_access_expire_minutes * 60,
    )


@router.post("/logout")
async def logout(response: Response) -> dict:
    """Log out (clear any server-side state)."""
    # For JWT-based auth, the client just discards the tokens.
    # This endpoint exists for consistency and future session invalidation.
    return {"message": "Logged out successfully."}


@router.get("/me", response_model=UserProfile)
async def get_profile(user: UserRow = Depends(require_user)) -> UserProfile:
    """Get the current user's profile."""
    return _user_to_profile(user)


@router.put("/me", response_model=UserProfile)
async def update_profile(
    body: UpdateProfileRequest,
    user: UserRow = Depends(require_user),
) -> UserProfile:
    """Update the current user's profile."""
    from tuneforge.api.database import update_user
    from tuneforge.api.server import app_state

    updates = {}
    if body.display_name is not None:
        updates["display_name"] = body.display_name
    if body.avatar_url is not None:
        updates["avatar_url"] = body.avatar_url

    if not updates:
        return _user_to_profile(user)

    updated = await update_user(app_state.db, user.id, **updates)
    return _user_to_profile(updated)
