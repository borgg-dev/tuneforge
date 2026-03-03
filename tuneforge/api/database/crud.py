"""
CRUD operations for TuneForge database models.

Provides async functions for creating, reading, updating, and deleting
records across all tables.
"""

import json
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import func, select, update

from tuneforge.api.database.engine import Database
from tuneforge.api.database.models import (
    ApiKeyRow,
    CreditRow,
    CreditTransactionRow,
    GenerationRow,
    TrackRow,
    UserRow,
)


# ---------------------------------------------------------------------------
# Track operations (preserved from original database.py)
# ---------------------------------------------------------------------------


async def create_track(
    db: Database,
    track_id: str,
    request_id: str,
    prompt: str,
    audio_path: str,
    duration_seconds: float,
    fmt: str = "mp3",
    sample_rate: int = 32000,
    generation_time_ms: int = 0,
    miner_hotkey: str = "",
    genre: str | None = None,
    mood: str | None = None,
    tempo_bpm: int | None = None,
    scores: dict[str, float] | None = None,
    user_id: str | None = None,
    generation_id: str | None = None,
) -> TrackRow:
    """Insert a new track record."""
    row = TrackRow(
        id=track_id,
        request_id=request_id,
        prompt=prompt,
        genre=genre,
        mood=mood,
        tempo_bpm=tempo_bpm,
        duration_seconds=duration_seconds,
        audio_path=audio_path,
        format=fmt,
        sample_rate=sample_rate,
        generation_time_ms=generation_time_ms,
        miner_hotkey=miner_hotkey,
        scores_json=json.dumps(scores or {}),
        created_at=datetime.now(timezone.utc),
        user_id=user_id,
        generation_id=generation_id,
    )
    async with db.session() as session:
        session.add(row)
        await session.commit()
        await session.refresh(row)
    logger.debug("Created track {} for request {}", track_id, request_id)
    return row


async def get_track(db: Database, track_id: str) -> TrackRow | None:
    """Fetch a single track by ID."""
    async with db.session() as session:
        return await session.get(TrackRow, track_id)


async def search_tracks(
    db: Database,
    genre: str | None = None,
    mood: str | None = None,
    min_tempo: int | None = None,
    max_tempo: int | None = None,
    user_id: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[TrackRow], int]:
    """Search tracks with optional filters. Returns (rows, total_count)."""
    stmt = select(TrackRow)
    count_stmt = select(func.count()).select_from(TrackRow)

    if genre:
        stmt = stmt.where(TrackRow.genre == genre)
        count_stmt = count_stmt.where(TrackRow.genre == genre)
    if mood:
        stmt = stmt.where(TrackRow.mood == mood)
        count_stmt = count_stmt.where(TrackRow.mood == mood)
    if min_tempo is not None:
        stmt = stmt.where(TrackRow.tempo_bpm >= min_tempo)
        count_stmt = count_stmt.where(TrackRow.tempo_bpm >= min_tempo)
    if max_tempo is not None:
        stmt = stmt.where(TrackRow.tempo_bpm <= max_tempo)
        count_stmt = count_stmt.where(TrackRow.tempo_bpm <= max_tempo)
    if user_id is not None:
        stmt = stmt.where(TrackRow.user_id == user_id)
        count_stmt = count_stmt.where(TrackRow.user_id == user_id)

    stmt = stmt.order_by(TrackRow.created_at.desc())
    offset = (page - 1) * page_size
    stmt = stmt.offset(offset).limit(page_size)

    async with db.session() as session:
        result = await session.execute(stmt)
        rows = list(result.scalars().all())
        count_result = await session.execute(count_stmt)
        total = count_result.scalar() or 0

    return rows, total


async def delete_track(db: Database, track_id: str, user_id: str) -> bool:
    """Delete a track owned by the given user. Returns True if deleted."""
    from sqlalchemy import delete as sa_delete

    async with db.session() as session:
        result = await session.execute(
            sa_delete(TrackRow).where(
                TrackRow.id == track_id,
                TrackRow.user_id == user_id,
            )
        )
        await session.commit()
        return result.rowcount > 0


async def get_track_count(db: Database) -> int:
    """Return total number of tracks."""
    async with db.session() as session:
        result = await session.execute(select(func.count()).select_from(TrackRow))
        return result.scalar() or 0


async def get_tracks_by_generation(db: Database, generation_id: str) -> list[TrackRow]:
    """Fetch all tracks for a generation."""
    async with db.session() as session:
        result = await session.execute(
            select(TrackRow).where(TrackRow.generation_id == generation_id)
        )
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# User operations
# ---------------------------------------------------------------------------


async def create_user(
    db: Database,
    email: str,
    password_hash: str | None = None,
    display_name: str | None = None,
    auth_provider: str = "email",
    google_id: str | None = None,
) -> UserRow:
    """Create a new user account."""
    user = UserRow(
        email=email,
        password_hash=password_hash,
        display_name=display_name or email.split("@")[0],
        auth_provider=auth_provider,
        google_id=google_id,
    )
    async with db.session() as session:
        session.add(user)
        await session.commit()
        await session.refresh(user)
    logger.info("Created user {} ({})", user.id, email)
    return user


async def get_user_by_email(db: Database, email: str) -> UserRow | None:
    """Fetch user by email address."""
    async with db.session() as session:
        result = await session.execute(
            select(UserRow).where(UserRow.email == email)
        )
        return result.scalar_one_or_none()


async def get_user_by_id(db: Database, user_id: str) -> UserRow | None:
    """Fetch user by ID."""
    async with db.session() as session:
        return await session.get(UserRow, user_id)


async def get_user_by_google_id(db: Database, google_id: str) -> UserRow | None:
    """Fetch user by Google ID."""
    async with db.session() as session:
        result = await session.execute(
            select(UserRow).where(UserRow.google_id == google_id)
        )
        return result.scalar_one_or_none()


async def update_user(db: Database, user_id: str, **kwargs) -> UserRow | None:
    """Update user fields."""
    kwargs["updated_at"] = datetime.now(timezone.utc)
    async with db.session() as session:
        await session.execute(
            update(UserRow).where(UserRow.id == user_id).values(**kwargs)
        )
        await session.commit()
        return await session.get(UserRow, user_id)


# ---------------------------------------------------------------------------
# Credit operations
# ---------------------------------------------------------------------------


async def create_credit(db: Database, user_id: str, daily_allowance: int = 50) -> CreditRow:
    """Create a credit record for a user."""
    credit = CreditRow(
        user_id=user_id,
        daily_balance=daily_allowance,
        daily_allowance=daily_allowance,
    )
    async with db.session() as session:
        session.add(credit)
        await session.commit()
        await session.refresh(credit)
    return credit


async def get_credit(db: Database, user_id: str) -> CreditRow | None:
    """Fetch credit balance for a user."""
    async with db.session() as session:
        result = await session.execute(
            select(CreditRow).where(CreditRow.user_id == user_id)
        )
        return result.scalar_one_or_none()


async def create_credit_transaction(
    db: Database,
    user_id: str,
    amount: int,
    tx_type: str,
    reference_id: str | None = None,
    description: str | None = None,
) -> CreditTransactionRow:
    """Record a credit transaction."""
    tx = CreditTransactionRow(
        user_id=user_id,
        amount=amount,
        tx_type=tx_type,
        reference_id=reference_id,
        description=description,
    )
    async with db.session() as session:
        session.add(tx)
        await session.commit()
        await session.refresh(tx)
    return tx


async def get_credit_transactions(
    db: Database,
    user_id: str,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[CreditTransactionRow], int]:
    """Fetch paginated credit transactions for a user."""
    stmt = (
        select(CreditTransactionRow)
        .where(CreditTransactionRow.user_id == user_id)
        .order_by(CreditTransactionRow.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    count_stmt = (
        select(func.count())
        .select_from(CreditTransactionRow)
        .where(CreditTransactionRow.user_id == user_id)
    )
    async with db.session() as session:
        result = await session.execute(stmt)
        rows = list(result.scalars().all())
        total = (await session.execute(count_stmt)).scalar() or 0
    return rows, total


# ---------------------------------------------------------------------------
# Generation operations
# ---------------------------------------------------------------------------


async def create_generation(
    db: Database,
    user_id: str,
    request_id: str,
    prompt: str,
    params_json: str = "{}",
    credits_reserved: int = 0,
) -> GenerationRow:
    """Create a generation record."""
    gen = GenerationRow(
        user_id=user_id,
        request_id=request_id,
        prompt=prompt,
        params_json=params_json,
        credits_reserved=credits_reserved,
    )
    async with db.session() as session:
        session.add(gen)
        await session.commit()
        await session.refresh(gen)
    return gen


async def get_generation(db: Database, request_id: str) -> GenerationRow | None:
    """Fetch generation by request_id."""
    async with db.session() as session:
        result = await session.execute(
            select(GenerationRow).where(GenerationRow.request_id == request_id)
        )
        return result.scalar_one_or_none()


async def update_generation_status(
    db: Database,
    request_id: str,
    status: str,
    error_message: str | None = None,
    credits_spent: int | None = None,
) -> None:
    """Update generation status."""
    values: dict = {"status": status}
    if status in ("completed", "failed"):
        values["completed_at"] = datetime.now(timezone.utc)
    if error_message is not None:
        values["error_message"] = error_message
    if credits_spent is not None:
        values["credits_spent"] = credits_spent

    async with db.session() as session:
        await session.execute(
            update(GenerationRow)
            .where(GenerationRow.request_id == request_id)
            .values(**values)
        )
        await session.commit()


# ---------------------------------------------------------------------------
# API Key operations
# ---------------------------------------------------------------------------


async def create_api_key(
    db: Database,
    user_id: str,
    key_prefix: str,
    key_hash: str,
    name: str = "Default",
) -> ApiKeyRow:
    """Create a new API key record."""
    row = ApiKeyRow(
        user_id=user_id,
        key_prefix=key_prefix,
        key_hash=key_hash,
        name=name,
    )
    async with db.session() as session:
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def get_api_key_by_hash(db: Database, key_hash: str) -> ApiKeyRow | None:
    """Look up an API key by its hash."""
    async with db.session() as session:
        result = await session.execute(
            select(ApiKeyRow).where(
                ApiKeyRow.key_hash == key_hash,
                ApiKeyRow.revoked_at.is_(None),
            )
        )
        return result.scalar_one_or_none()


async def get_user_api_keys(db: Database, user_id: str) -> list[ApiKeyRow]:
    """List all (non-revoked) API keys for a user."""
    async with db.session() as session:
        result = await session.execute(
            select(ApiKeyRow).where(
                ApiKeyRow.user_id == user_id,
                ApiKeyRow.revoked_at.is_(None),
            ).order_by(ApiKeyRow.created_at.desc())
        )
        return list(result.scalars().all())


async def revoke_api_key(db: Database, key_id: str, user_id: str) -> bool:
    """Revoke an API key. Returns True if found and revoked."""
    async with db.session() as session:
        result = await session.execute(
            update(ApiKeyRow)
            .where(ApiKeyRow.id == key_id, ApiKeyRow.user_id == user_id, ApiKeyRow.revoked_at.is_(None))
            .values(revoked_at=datetime.now(timezone.utc))
        )
        await session.commit()
        return result.rowcount > 0


async def update_api_key_last_used(db: Database, key_id: str) -> None:
    """Update the last_used_at timestamp for an API key."""
    async with db.session() as session:
        await session.execute(
            update(ApiKeyRow)
            .where(ApiKeyRow.id == key_id)
            .values(last_used_at=datetime.now(timezone.utc))
        )
        await session.commit()
