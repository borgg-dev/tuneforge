"""
CRUD operations for TuneForge database models.

Provides async functions for creating, reading, updating, and deleting
records across all tables.
"""

import json
from datetime import date, datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import func, select, update

from tuneforge.api.database.engine import Database
from tuneforge.api.database.models import (
    AnnotationMilestoneRow,
    AnnotationRow,
    AnnotationStreakRow,
    AnnotationTaskRow,
    AnnotatorReliabilityRow,
    ApiKeyRow,
    AudioBlobRow,
    CreditRow,
    CreditTransactionRow,
    GenerationRow,
    PreferenceModelRow,
    TrackRow,
    UserRow,
    ValidationAudioRow,
    ValidationRoundRow,
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
    audio_blob_id: str | None = None,
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
        audio_blob_id=audio_blob_id,
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


# ---------------------------------------------------------------------------
# Audio blob operations
# ---------------------------------------------------------------------------


async def create_audio_blob(
    db: Database,
    audio_data: bytes,
    fmt: str = "wav",
    content_type: str = "audio/wav",
    blob_id: str | None = None,
) -> AudioBlobRow:
    """Insert an audio blob into PostgreSQL."""
    import uuid

    row = AudioBlobRow(
        id=blob_id or uuid.uuid4().hex,
        audio_data=audio_data,
        format=fmt,
        size_bytes=len(audio_data),
        content_type=content_type,
    )
    async with db.session() as session:
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def get_audio_blob(db: Database, blob_id: str) -> AudioBlobRow | None:
    """Fetch audio blob by ID."""
    async with db.session() as session:
        return await session.get(AudioBlobRow, blob_id)


async def delete_audio_blob(db: Database, blob_id: str) -> bool:
    """Delete an audio blob. Returns True if deleted."""
    from sqlalchemy import delete as sa_delete

    async with db.session() as session:
        result = await session.execute(
            sa_delete(AudioBlobRow).where(AudioBlobRow.id == blob_id)
        )
        await session.commit()
        return result.rowcount > 0


# ---------------------------------------------------------------------------
# Validation round operations
# ---------------------------------------------------------------------------


async def create_validation_round(
    db: Database,
    challenge_id: str,
    prompt: str,
    genre: str | None = None,
    mood: str | None = None,
    tempo_bpm: int | None = None,
    duration_seconds: float = 0,
    validator_hotkey: str | None = None,
) -> ValidationRoundRow:
    """Insert a validation round."""
    row = ValidationRoundRow(
        challenge_id=challenge_id,
        prompt=prompt,
        genre=genre,
        mood=mood,
        tempo_bpm=tempo_bpm,
        duration_seconds=duration_seconds,
        validator_hotkey=validator_hotkey,
    )
    async with db.session() as session:
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def create_validation_audio(
    db: Database,
    round_id: str,
    miner_uid: int,
    audio_blob_id: str,
    miner_hotkey: str | None = None,
    generation_time_ms: int | None = None,
    score: float | None = None,
) -> ValidationAudioRow:
    """Insert a validation audio entry."""
    row = ValidationAudioRow(
        round_id=round_id,
        miner_uid=miner_uid,
        miner_hotkey=miner_hotkey,
        audio_blob_id=audio_blob_id,
        generation_time_ms=generation_time_ms,
        score=score,
    )
    async with db.session() as session:
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def get_validation_rounds(
    db: Database,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[ValidationRoundRow], int]:
    """Fetch paginated validation rounds."""
    stmt = (
        select(ValidationRoundRow)
        .order_by(ValidationRoundRow.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    count_stmt = select(func.count()).select_from(ValidationRoundRow)
    async with db.session() as session:
        result = await session.execute(stmt)
        rows = list(result.scalars().all())
        total = (await session.execute(count_stmt)).scalar() or 0
    return rows, total


async def get_validation_audio_by_round(
    db: Database,
    round_id: str,
) -> list[ValidationAudioRow]:
    """Fetch all audio entries for a validation round."""
    async with db.session() as session:
        result = await session.execute(
            select(ValidationAudioRow).where(ValidationAudioRow.round_id == round_id)
        )
        return list(result.scalars().all())


async def update_validation_audio_scores(
    db: Database,
    round_id: str,
    scores: list[tuple[int, float]],
) -> int:
    """Batch-update scores for validation audio entries in a round.

    Parameters
    ----------
    scores : list of (miner_uid, score) tuples

    Returns the number of rows updated.
    """
    updated = 0
    async with db.session() as session:
        for miner_uid, score in scores:
            result = await session.execute(
                update(ValidationAudioRow)
                .where(
                    ValidationAudioRow.round_id == round_id,
                    ValidationAudioRow.miner_uid == miner_uid,
                )
                .values(score=score)
            )
            updated += result.rowcount
        await session.commit()
    return updated


# ---------------------------------------------------------------------------
# Annotation task operations
# ---------------------------------------------------------------------------


async def create_annotation_tasks_from_round(
    db: Database,
    round_id: str,
    quorum: int = 5,
) -> tuple[int, int]:
    """Generate all C(n,2) pairwise A/B tasks from audio in a round.

    Pairs are stored in canonical order (audio_a_id < audio_b_id).
    Returns (created_count, skipped_count).
    """
    from itertools import combinations

    audios = await get_validation_audio_by_round(db, round_id)
    if len(audios) < 2:
        return 0, 0

    # Get round metadata for denormalized fields
    async with db.session() as session:
        round_row = await session.get(ValidationRoundRow, round_id)

    sorted_audios = sorted(audios, key=lambda a: a.id)
    created = 0
    skipped = 0

    async with db.session() as session:
        for a, b in combinations(sorted_audios, 2):
            # Canonical order: smaller id is always "a"
            a_id, b_id = (a.id, b.id) if a.id < b.id else (b.id, a.id)
            # Check if pair already exists
            existing = await session.execute(
                select(AnnotationTaskRow).where(
                    AnnotationTaskRow.audio_a_id == a_id,
                    AnnotationTaskRow.audio_b_id == b_id,
                )
            )
            if existing.scalar_one_or_none() is not None:
                skipped += 1
                continue
            task = AnnotationTaskRow(
                round_id=round_id,
                audio_a_id=a_id,
                audio_b_id=b_id,
                quorum=quorum,
            )
            session.add(task)
            created += 1
        await session.commit()

    return created, skipped


async def get_next_annotation_task(
    db: Database,
    user_id: str,
) -> AnnotationTaskRow | None:
    """Fetch the next task, mixing gold tasks at ~20%.

    Gold tasks are served randomly. Regular tasks prioritize fewest votes.
    """
    import random

    async with db.session() as session:
        # Subquery: task IDs this user already voted on
        voted_subq = (
            select(AnnotationRow.task_id)
            .where(AnnotationRow.user_id == user_id)
            .scalar_subquery()
        )

        # Check available gold tasks for this user
        gold_available = (await session.execute(
            select(func.count()).select_from(AnnotationTaskRow)
            .where(
                AnnotationTaskRow.is_gold == True,  # noqa: E712
                AnnotationTaskRow.id.not_in(voted_subq),
            )
        )).scalar() or 0

        # Decide whether to serve gold
        if gold_available >= 1 and random.random() < GOLD_INJECTION_RATE:
            result = await session.execute(
                select(AnnotationTaskRow)
                .where(
                    AnnotationTaskRow.is_gold == True,  # noqa: E712
                    AnnotationTaskRow.id.not_in(voted_subq),
                )
                .order_by(func.random())
                .limit(1)
            )
            task = result.scalar_one_or_none()
            if task is not None:
                return task

        # Regular tasks (exclude gold), ordered by fewest votes
        result = await session.execute(
            select(AnnotationTaskRow)
            .where(
                AnnotationTaskRow.status == "open",
                AnnotationTaskRow.is_gold == False,  # noqa: E712
                AnnotationTaskRow.id.not_in(voted_subq),
            )
            .order_by(AnnotationTaskRow.vote_count.asc(), AnnotationTaskRow.created_at.asc())
            .limit(1)
        )
        return result.scalar_one_or_none()


async def submit_annotation(
    db: Database,
    task_id: str,
    user_id: str,
    choice: str,
    duration_ms: int | None = None,
) -> AnnotationRow:
    """Record a vote with gold/reliability tracking and weighted aggregation."""
    async with db.session() as session:
        task = await session.get(AnnotationTaskRow, task_id)
        if task is None:
            raise ValueError("Task not found")

        # Get or create reliability record
        rel_result = await session.execute(
            select(AnnotatorReliabilityRow)
            .where(AnnotatorReliabilityRow.user_id == user_id)
        )
        reliability = rel_result.scalar_one_or_none()
        if reliability is None:
            reliability = AnnotatorReliabilityRow(user_id=user_id)
            session.add(reliability)
            await session.flush()

        is_gold_task = task.is_gold
        is_annotator_trusted = not reliability.is_flagged

        annotation = AnnotationRow(
            task_id=task_id,
            user_id=user_id,
            choice=choice,
            duration_ms=duration_ms,
            is_gold_response=is_gold_task,
            is_trusted=is_annotator_trusted,
        )
        session.add(annotation)

        # Always increment vote_count
        task.vote_count += 1

        if is_gold_task:
            # Update reliability score
            reliability.gold_total += 1
            if choice == task.gold_answer:
                reliability.gold_correct += 1
            reliability.accuracy = reliability.gold_correct / reliability.gold_total

            # Check flagging threshold
            was_flagged = reliability.is_flagged
            if reliability.gold_total >= MIN_GOLD_FOR_FLAG:
                if reliability.accuracy < FLAG_ACCURACY_THRESHOLD and not reliability.is_flagged:
                    reliability.is_flagged = True
                    reliability.flagged_at = datetime.now(timezone.utc)
                    reliability.unflagged_at = None
                    logger.warning(
                        "Annotator {} flagged: accuracy={:.1%} over {} gold tasks",
                        user_id, reliability.accuracy, reliability.gold_total,
                    )
                elif reliability.accuracy >= FLAG_ACCURACY_THRESHOLD and reliability.is_flagged:
                    reliability.is_flagged = False
                    reliability.unflagged_at = datetime.now(timezone.utc)
                    logger.info(
                        "Annotator {} unflagged: accuracy={:.1%}",
                        user_id, reliability.accuracy,
                    )

        await session.commit()
        await session.refresh(annotation)

    # For regular tasks, check weighted aggregation
    if not is_gold_task:
        await _maybe_resolve_task_weighted(db, task_id)

    # If annotator just got flagged, retroactively mark their past votes
    if is_gold_task and reliability.is_flagged:
        await _mark_user_votes_untrusted(db, user_id)

    return annotation


async def _maybe_resolve_task_weighted(db: Database, task_id: str) -> None:
    """Resolve a task using reliability-weighted votes from trusted annotators."""
    async with db.session() as session:
        task = await session.get(AnnotationTaskRow, task_id)
        if task is None or task.status != "open" or task.is_gold:
            return

        # Count trusted votes for quorum check
        trusted_count = (await session.execute(
            select(func.count()).select_from(AnnotationRow)
            .where(
                AnnotationRow.task_id == task_id,
                AnnotationRow.is_trusted == True,  # noqa: E712
            )
        )).scalar() or 0

        if trusted_count < task.quorum:
            return

        # Get trusted votes with reliability weights
        result = await session.execute(
            select(AnnotationRow.choice, AnnotatorReliabilityRow.accuracy)
            .outerjoin(
                AnnotatorReliabilityRow,
                AnnotationRow.user_id == AnnotatorReliabilityRow.user_id,
            )
            .where(
                AnnotationRow.task_id == task_id,
                AnnotationRow.is_trusted == True,  # noqa: E712
            )
        )
        votes = result.all()

        score_a = 0.0
        score_b = 0.0
        for choice, accuracy in votes:
            weight = accuracy if accuracy is not None else 1.0
            if choice == "a":
                score_a += weight
            else:
                score_b += weight

        if score_a > score_b:
            task.status = "completed"
            task.winner = "a"
        elif score_b > score_a:
            task.status = "completed"
            task.winner = "b"
        else:
            task.status = "discarded"
            task.winner = None

        task.completed_at = datetime.now(timezone.utc)
        await session.commit()


async def get_annotation_task(db: Database, task_id: str) -> AnnotationTaskRow | None:
    """Fetch a single annotation task by ID."""
    async with db.session() as session:
        return await session.get(AnnotationTaskRow, task_id)


async def get_annotation_tasks(
    db: Database,
    status: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[AnnotationTaskRow], int]:
    """Paginated task listing with optional status filter."""
    stmt = select(AnnotationTaskRow)
    count_stmt = select(func.count()).select_from(AnnotationTaskRow)

    if status:
        stmt = stmt.where(AnnotationTaskRow.status == status)
        count_stmt = count_stmt.where(AnnotationTaskRow.status == status)

    stmt = (
        stmt.order_by(AnnotationTaskRow.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    async with db.session() as session:
        result = await session.execute(stmt)
        rows = list(result.scalars().all())
        total = (await session.execute(count_stmt)).scalar() or 0
    return rows, total


async def get_annotation_stats(db: Database) -> dict:
    """Return aggregate annotation statistics including gold/flagged counts."""
    async with db.session() as session:
        # Exclude gold tasks from regular counts
        total = (await session.execute(
            select(func.count()).select_from(AnnotationTaskRow)
            .where(AnnotationTaskRow.is_gold == False)  # noqa: E712
        )).scalar() or 0
        open_tasks = (await session.execute(
            select(func.count()).select_from(AnnotationTaskRow)
            .where(AnnotationTaskRow.status == "open", AnnotationTaskRow.is_gold == False)  # noqa: E712
        )).scalar() or 0
        completed = (await session.execute(
            select(func.count()).select_from(AnnotationTaskRow)
            .where(AnnotationTaskRow.status == "completed")
        )).scalar() or 0
        discarded = (await session.execute(
            select(func.count()).select_from(AnnotationTaskRow)
            .where(AnnotationTaskRow.status == "discarded")
        )).scalar() or 0
        total_annotations = (await session.execute(
            select(func.count()).select_from(AnnotationRow)
        )).scalar() or 0
        annotator_count = (await session.execute(
            select(func.count(func.distinct(AnnotationRow.user_id)))
        )).scalar() or 0
        total_gold = (await session.execute(
            select(func.count()).select_from(AnnotationTaskRow)
            .where(AnnotationTaskRow.is_gold == True)  # noqa: E712
        )).scalar() or 0
        flagged_count = (await session.execute(
            select(func.count()).select_from(AnnotatorReliabilityRow)
            .where(AnnotatorReliabilityRow.is_flagged == True)  # noqa: E712
        )).scalar() or 0

    return {
        "total_tasks": total,
        "open_tasks": open_tasks,
        "completed_tasks": completed,
        "discarded_tasks": discarded,
        "total_annotations": total_annotations,
        "annotator_count": annotator_count,
        "total_gold_tasks": total_gold,
        "flagged_annotator_count": flagged_count,
    }


async def get_user_annotation_progress(
    db: Database,
    user_id: str,
) -> dict:
    """Return annotation progress for a specific user."""
    async with db.session() as session:
        completed_by_user = (await session.execute(
            select(func.count()).select_from(AnnotationRow)
            .where(AnnotationRow.user_id == user_id)
        )).scalar() or 0

        # Count open tasks user hasn't voted on
        voted_subq = (
            select(AnnotationRow.task_id)
            .where(AnnotationRow.user_id == user_id)
            .scalar_subquery()
        )
        remaining = (await session.execute(
            select(func.count()).select_from(AnnotationTaskRow)
            .where(
                AnnotationTaskRow.status == "open",
                AnnotationTaskRow.id.not_in(voted_subq),
            )
        )).scalar() or 0

        total_open = (await session.execute(
            select(func.count()).select_from(AnnotationTaskRow)
            .where(AnnotationTaskRow.status == "open")
        )).scalar() or 0

    return {
        "completed_by_user": completed_by_user,
        "remaining_for_user": remaining,
        "total_open": total_open,
    }


async def export_resolved_annotations(db: Database) -> list[dict]:
    """Export completed non-gold tasks with confidence scores.

    Excludes gold tasks and computes confidence from trusted votes only.
    """
    async with db.session() as session:
        result = await session.execute(
            select(
                AnnotationTaskRow.id,
                AnnotationTaskRow.round_id,
                AnnotationTaskRow.winner,
                AnnotationTaskRow.audio_a_id,
                AnnotationTaskRow.audio_b_id,
                ValidationRoundRow.prompt,
                ValidationRoundRow.genre,
                ValidationRoundRow.challenge_id,
            )
            .join(ValidationRoundRow, AnnotationTaskRow.round_id == ValidationRoundRow.id)
            .where(
                AnnotationTaskRow.status == "completed",
                AnnotationTaskRow.is_gold == False,  # noqa: E712
            )
            .order_by(AnnotationTaskRow.created_at.asc())
        )
        rows = result.all()

    exports = []
    async with db.session() as session:
        for row in rows:
            audio_a = await session.get(ValidationAudioRow, row.audio_a_id)
            audio_b = await session.get(ValidationAudioRow, row.audio_b_id)
            if not audio_a or not audio_b:
                continue

            # Calculate confidence from trusted votes
            votes_result = await session.execute(
                select(AnnotationRow.choice, AnnotatorReliabilityRow.accuracy)
                .outerjoin(
                    AnnotatorReliabilityRow,
                    AnnotationRow.user_id == AnnotatorReliabilityRow.user_id,
                )
                .where(
                    AnnotationRow.task_id == row.id,
                    AnnotationRow.is_trusted == True,  # noqa: E712
                )
            )
            votes = votes_result.all()

            score_a = sum((acc or 1.0) for ch, acc in votes if ch == "a")
            score_b = sum((acc or 1.0) for ch, acc in votes if ch == "b")
            total_score = score_a + score_b
            winning_score = score_a if row.winner == "a" else score_b
            confidence = round(winning_score / total_score, 4) if total_score > 0 else 0.5

            exports.append({
                "pair_id": row.id,
                "audio_a_blob_id": audio_a.audio_blob_id,
                "audio_b_blob_id": audio_b.audio_blob_id,
                "preferred": row.winner,
                "confidence": confidence,
                "n_trusted_votes": len(votes),
                "challenge_id": row.challenge_id,
                "prompt": row.prompt,
                "genre": row.genre,
            })
    return exports


# ---------------------------------------------------------------------------
# Preference model operations
# ---------------------------------------------------------------------------


async def create_preference_model(
    db: Database,
    model_data: bytes,
    sha256: str,
    val_accuracy: float | None = None,
    n_train_pairs: int | None = None,
) -> PreferenceModelRow:
    """Upload a new preference model checkpoint.

    Deactivates all previous models and activates this one.
    """
    async with db.session() as session:
        # Get next version number
        max_ver = (await session.execute(
            select(func.max(PreferenceModelRow.version))
        )).scalar() or 0
        version = max_ver + 1

        # Deactivate all existing models
        await session.execute(
            update(PreferenceModelRow).values(is_active=False)
        )

        row = PreferenceModelRow(
            version=version,
            sha256=sha256,
            val_accuracy=val_accuracy,
            n_train_pairs=n_train_pairs,
            is_active=True,
            model_data=model_data,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)

    logger.info("Uploaded preference model v{} (sha256={})", version, sha256[:12])
    return row


async def get_active_preference_model(db: Database) -> PreferenceModelRow | None:
    """Fetch the currently active preference model."""
    async with db.session() as session:
        result = await session.execute(
            select(PreferenceModelRow).where(PreferenceModelRow.is_active == True)  # noqa: E712
        )
        return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Gold standard quality control
# ---------------------------------------------------------------------------

GOLD_INJECTION_RATE = 0.20
MIN_GOLD_FOR_FLAG = 5
FLAG_ACCURACY_THRESHOLD = 0.70
MIN_LISTEN_DURATION_MS = 3000


async def create_gold_task(
    db: Database,
    round_id: str,
    audio_a_id: str,
    audio_b_id: str,
    gold_answer: str,
) -> AnnotationTaskRow:
    """Create a gold/honeypot task with a known correct answer.

    Gold tasks use quorum=999999 so they never complete via normal aggregation.
    """
    # Canonical ordering: ensure audio_a_id < audio_b_id
    if audio_a_id > audio_b_id:
        audio_a_id, audio_b_id = audio_b_id, audio_a_id
        gold_answer = "b" if gold_answer == "a" else "a"

    task = AnnotationTaskRow(
        round_id=round_id,
        audio_a_id=audio_a_id,
        audio_b_id=audio_b_id,
        quorum=999999,
        is_gold=True,
        gold_answer=gold_answer,
    )
    async with db.session() as session:
        session.add(task)
        await session.commit()
        await session.refresh(task)
    return task


async def get_gold_tasks(
    db: Database,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[dict], int]:
    """List gold tasks with per-task accuracy stats."""
    async with db.session() as session:
        total = (await session.execute(
            select(func.count()).select_from(AnnotationTaskRow)
            .where(AnnotationTaskRow.is_gold == True)  # noqa: E712
        )).scalar() or 0

        result = await session.execute(
            select(AnnotationTaskRow)
            .where(AnnotationTaskRow.is_gold == True)  # noqa: E712
            .order_by(AnnotationTaskRow.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        tasks = result.scalars().all()

        gold_tasks = []
        for task in tasks:
            # Calculate accuracy: how many annotators got it right
            correct_count = (await session.execute(
                select(func.count()).select_from(AnnotationRow)
                .where(
                    AnnotationRow.task_id == task.id,
                    AnnotationRow.choice == task.gold_answer,
                )
            )).scalar() or 0

            accuracy = correct_count / task.vote_count if task.vote_count > 0 else None

            gold_tasks.append({
                "task_id": task.id,
                "audio_a_id": task.audio_a_id,
                "audio_b_id": task.audio_b_id,
                "gold_answer": task.gold_answer,
                "vote_count": task.vote_count,
                "gold_accuracy": round(accuracy, 4) if accuracy is not None else None,
                "created_at": task.created_at,
            })

    return gold_tasks, total


async def delete_gold_task(db: Database, task_id: str) -> bool:
    """Delete a gold task and its associated annotations."""
    async with db.session() as session:
        task = await session.get(AnnotationTaskRow, task_id)
        if task is None or not task.is_gold:
            return False
        await session.delete(task)
        await session.commit()
    return True


async def get_annotator_reliability(
    db: Database,
    user_id: str,
) -> AnnotatorReliabilityRow | None:
    """Fetch reliability record for a user."""
    async with db.session() as session:
        result = await session.execute(
            select(AnnotatorReliabilityRow)
            .where(AnnotatorReliabilityRow.user_id == user_id)
        )
        return result.scalar_one_or_none()


async def get_all_annotator_reliabilities(
    db: Database,
    page: int = 1,
    page_size: int = 50,
    flagged_only: bool = False,
) -> tuple[list[dict], int]:
    """List annotators with reliability scores."""
    async with db.session() as session:
        base_q = select(
            AnnotatorReliabilityRow,
            UserRow.email,
            UserRow.display_name,
        ).join(UserRow, AnnotatorReliabilityRow.user_id == UserRow.id)

        if flagged_only:
            base_q = base_q.where(AnnotatorReliabilityRow.is_flagged == True)  # noqa: E712

        total = (await session.execute(
            select(func.count()).select_from(AnnotatorReliabilityRow)
            .where(AnnotatorReliabilityRow.is_flagged == True if flagged_only else True)  # noqa: E712
        )).scalar() or 0

        result = await session.execute(
            base_q
            .order_by(AnnotatorReliabilityRow.accuracy.asc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        rows = result.all()

        annotators = []
        for reliability, email, display_name in rows:
            # Get total annotations count
            ann_count = (await session.execute(
                select(func.count()).select_from(AnnotationRow)
                .where(AnnotationRow.user_id == reliability.user_id)
            )).scalar() or 0

            annotators.append({
                "user_id": reliability.user_id,
                "email": email,
                "display_name": display_name,
                "gold_total": reliability.gold_total,
                "gold_correct": reliability.gold_correct,
                "accuracy": reliability.accuracy,
                "is_flagged": reliability.is_flagged,
                "flagged_at": reliability.flagged_at,
                "total_annotations": ann_count,
            })

    return annotators, total


async def flag_annotator(db: Database, user_id: str) -> None:
    """Manually flag an annotator and mark their votes untrusted."""
    async with db.session() as session:
        result = await session.execute(
            select(AnnotatorReliabilityRow)
            .where(AnnotatorReliabilityRow.user_id == user_id)
        )
        reliability = result.scalar_one_or_none()
        if reliability is None:
            reliability = AnnotatorReliabilityRow(user_id=user_id)
            session.add(reliability)
        reliability.is_flagged = True
        reliability.flagged_at = datetime.now(timezone.utc)
        reliability.unflagged_at = None
        await session.commit()

    await _mark_user_votes_untrusted(db, user_id)


async def unflag_annotator(db: Database, user_id: str) -> None:
    """Manually unflag an annotator and restore trust on their votes."""
    async with db.session() as session:
        result = await session.execute(
            select(AnnotatorReliabilityRow)
            .where(AnnotatorReliabilityRow.user_id == user_id)
        )
        reliability = result.scalar_one_or_none()
        if reliability is None:
            return
        reliability.is_flagged = False
        reliability.unflagged_at = datetime.now(timezone.utc)
        await session.commit()

    # Restore trust on their votes
    async with db.session() as session:
        await session.execute(
            update(AnnotationRow)
            .where(AnnotationRow.user_id == user_id)
            .values(is_trusted=True)
        )
        await session.commit()


async def _mark_user_votes_untrusted(db: Database, user_id: str) -> None:
    """Mark all votes from a flagged user as untrusted."""
    async with db.session() as session:
        await session.execute(
            update(AnnotationRow)
            .where(AnnotationRow.user_id == user_id)
            .values(is_trusted=False)
        )
        await session.commit()


# ---------------------------------------------------------------------------
# Annotation milestones
# ---------------------------------------------------------------------------


async def get_user_trusted_annotation_count(db: Database, user_id: str) -> int:
    """Count trusted, non-gold annotations for a user (milestone-eligible)."""
    async with db.session() as session:
        count = (await session.execute(
            select(func.count()).select_from(AnnotationRow)
            .where(
                AnnotationRow.user_id == user_id,
                AnnotationRow.is_trusted == True,  # noqa: E712
                AnnotationRow.is_gold_response == False,  # noqa: E712
            )
        )).scalar() or 0
    return count


async def get_claimed_milestone_keys(db: Database, user_id: str) -> set[str]:
    """Return the set of milestone keys already claimed by a user."""
    async with db.session() as session:
        result = await session.execute(
            select(AnnotationMilestoneRow.milestone_key)
            .where(AnnotationMilestoneRow.user_id == user_id)
        )
        return {row[0] for row in result.all()}


async def get_claimed_milestones(
    db: Database, user_id: str,
) -> list[AnnotationMilestoneRow]:
    """Return all milestone records for a user, ordered by claimed_at."""
    async with db.session() as session:
        result = await session.execute(
            select(AnnotationMilestoneRow)
            .where(AnnotationMilestoneRow.user_id == user_id)
            .order_by(AnnotationMilestoneRow.claimed_at)
        )
        return list(result.scalars().all())


async def claim_milestone(
    db: Database,
    user_id: str,
    milestone_key: str,
    credits: int,
    pro_days: int = 0,
) -> AnnotationMilestoneRow:
    """Claim a milestone: insert record, grant credits, optionally upgrade to Pro."""
    from datetime import timedelta

    now = datetime.now(timezone.utc)

    async with db.transaction() as session:
        row = AnnotationMilestoneRow(
            user_id=user_id,
            milestone_key=milestone_key,
            credits_awarded=credits,
            claimed_at=now,
        )
        session.add(row)

        # Add credits to daily balance
        await session.execute(
            update(CreditRow)
            .where(CreditRow.user_id == user_id)
            .values(daily_balance=CreditRow.daily_balance + credits)
        )

        # Record credit transaction
        tx = CreditTransactionRow(
            user_id=user_id,
            amount=credits,
            tx_type="milestone_reward",
            reference_id=milestone_key,
            description=f"Milestone reward: {milestone_key}",
        )
        session.add(tx)

        # Pro upgrade for Gold Ear
        if pro_days > 0:
            expires_at = now + timedelta(days=pro_days)
            await session.execute(
                update(UserRow)
                .where(UserRow.id == user_id)
                .values(plan_tier="pro", pro_expires_at=expires_at)
            )
            logger.info(
                "User {} upgraded to Pro until {} via milestone {}",
                user_id, expires_at.isoformat(), milestone_key,
            )

    logger.info(
        "Milestone '{}' claimed by user {}: +{} credits",
        milestone_key, user_id, credits,
    )
    return row


async def check_and_claim_milestones(
    db: Database, user_id: str,
) -> list[AnnotationMilestoneRow]:
    """Check if user qualifies for new milestones after a vote, and claim them."""
    from tuneforge.api.milestones import get_newly_unlocked

    trusted_count = await get_user_trusted_annotation_count(db, user_id)
    claimed_keys = await get_claimed_milestone_keys(db, user_id)
    newly_unlocked = get_newly_unlocked(trusted_count, claimed_keys)

    claimed = []
    for milestone in newly_unlocked:
        row = await claim_milestone(
            db,
            user_id=user_id,
            milestone_key=milestone.key,
            credits=milestone.credits,
            pro_days=milestone.grants_pro_days,
        )
        claimed.append(row)

    return claimed


# ---------------------------------------------------------------------------
# Annotation streaks & recurring rewards
# ---------------------------------------------------------------------------


async def get_or_create_streak(db: Database, user_id: str) -> AnnotationStreakRow:
    """Get or create the streak row for a user."""
    async with db.session() as session:
        result = await session.execute(
            select(AnnotationStreakRow).where(AnnotationStreakRow.user_id == user_id)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            return row

    async with db.session() as session:
        row = AnnotationStreakRow(user_id=user_id)
        session.add(row)
        await session.commit()
        await session.refresh(row)
    return row


async def get_user_daily_trusted_annotation_count(
    db: Database,
    user_id: str,
    target_date: date,
) -> int:
    """Count trusted non-gold annotations for a user on a specific UTC date."""
    day_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    async with db.session() as session:
        count = (await session.execute(
            select(func.count()).select_from(AnnotationRow)
            .where(
                AnnotationRow.user_id == user_id,
                AnnotationRow.is_trusted == True,  # noqa: E712
                AnnotationRow.is_gold_response == False,  # noqa: E712
                AnnotationRow.created_at >= day_start,
                AnnotationRow.created_at < day_end,
            )
        )).scalar() or 0
    return count


async def update_streak(db: Database, user_id: str) -> AnnotationStreakRow:
    """Update the user's daily streak after a qualifying annotation.

    Checks if today has enough annotations (>= threshold) and updates
    streak days with grace logic. Uses row-level locking.
    """
    from tuneforge.api.milestones import DAILY_STREAK_THRESHOLD

    today = datetime.now(timezone.utc).date()

    # Ensure streak row exists
    await get_or_create_streak(db, user_id)

    today_count = await get_user_daily_trusted_annotation_count(db, user_id, today)

    async with db.session() as session:
        result = await session.execute(
            select(AnnotationStreakRow)
            .where(AnnotationStreakRow.user_id == user_id)
            .with_for_update()
        )
        streak = result.scalar_one()

        # Already updated today
        if streak.last_active_date == today:
            return streak

        # Only activate if threshold met
        if today_count < DAILY_STREAK_THRESHOLD:
            return streak

        if streak.last_active_date is None:
            # First ever qualifying day
            streak.current_streak_days = 1
            streak.grace_used = False
        else:
            gap = (today - streak.last_active_date).days
            if gap == 1:
                # Consecutive day
                streak.current_streak_days += 1
                streak.grace_used = False
            elif gap == 2 and not streak.grace_used:
                # 1-day grace period (missed yesterday but not grace-used)
                streak.current_streak_days += 1
                streak.grace_used = True
            else:
                # Streak broken — restart
                streak.current_streak_days = 1
                streak.grace_used = False

        streak.last_active_date = today
        await session.commit()
        await session.refresh(streak)

    return streak


async def check_and_claim_recurring_reward(
    db: Database,
    user_id: str,
) -> tuple[int, int] | None:
    """Check if the user earned a recurring batch reward.

    Returns (credits_awarded, total_batches) if a batch was claimed, else None.
    Only eligible after all milestones are claimed.
    """
    from tuneforge.api.milestones import (
        RECURRING_BATCH_SIZE,
        compute_recurring_credits,
        is_recurring_eligible,
    )

    claimed_keys = await get_claimed_milestone_keys(db, user_id)
    if not is_recurring_eligible(claimed_keys):
        return None

    async with db.session() as session:
        result = await session.execute(
            select(AnnotationStreakRow)
            .where(AnnotationStreakRow.user_id == user_id)
            .with_for_update()
        )
        streak = result.scalar_one_or_none()
        if streak is None:
            return None

        streak.recurring_annotations_count += 1

        if streak.recurring_annotations_count < RECURRING_BATCH_SIZE:
            await session.commit()
            return None

        # Batch complete — award credits
        credits = compute_recurring_credits(streak.current_streak_days)
        streak.recurring_annotations_count = 0
        streak.total_recurring_credits_earned += credits
        streak.total_recurring_batches_claimed += 1

        # Grant credits
        await session.execute(
            update(CreditRow)
            .where(CreditRow.user_id == user_id)
            .values(daily_balance=CreditRow.daily_balance + credits)
        )

        tx = CreditTransactionRow(
            user_id=user_id,
            amount=credits,
            tx_type="recurring_reward",
            reference_id=f"batch_{streak.total_recurring_batches_claimed}",
            description=f"Recurring annotation reward (batch #{streak.total_recurring_batches_claimed}, {streak.current_streak_days}d streak)",
        )
        session.add(tx)
        await session.commit()

        logger.info(
            "Recurring reward for user {}: +{} credits (batch #{}, streak {}d)",
            user_id, credits, streak.total_recurring_batches_claimed, streak.current_streak_days,
        )
        return credits, streak.total_recurring_batches_claimed


async def get_streak_info(db: Database, user_id: str) -> dict:
    """Return full streak info dict for the API response."""
    from tuneforge.api.milestones import (
        DAILY_STREAK_THRESHOLD,
        RECURRING_BATCH_SIZE,
        compute_recurring_credits,
        get_next_streak_tier,
        get_streak_tier,
        is_recurring_eligible,
    )

    claimed_keys = await get_claimed_milestone_keys(db, user_id)
    eligible = is_recurring_eligible(claimed_keys)

    if not eligible:
        return {"is_recurring_eligible": False}

    streak = await get_or_create_streak(db, user_id)
    today = datetime.now(timezone.utc).date()
    today_count = await get_user_daily_trusted_annotation_count(db, user_id, today)

    tier = get_streak_tier(streak.current_streak_days)
    next_tier = get_next_streak_tier(streak.current_streak_days)
    credits_per_batch = compute_recurring_credits(streak.current_streak_days)
    remaining = RECURRING_BATCH_SIZE - streak.recurring_annotations_count

    # Streak is at risk if today's count is below threshold and the day isn't done
    streak_at_risk = (
        streak.current_streak_days > 0
        and today_count < DAILY_STREAK_THRESHOLD
        and streak.last_active_date != today
    )

    info: dict = {
        "is_recurring_eligible": True,
        "current_streak_days": streak.current_streak_days,
        "streak_multiplier": tier.multiplier,
        "streak_tier_label": tier.label,
        "credits_per_batch": credits_per_batch,
        "batch_size": RECURRING_BATCH_SIZE,
        "annotations_toward_next_batch": streak.recurring_annotations_count,
        "annotations_remaining": remaining,
        "total_recurring_credits_earned": streak.total_recurring_credits_earned,
        "total_recurring_batches_claimed": streak.total_recurring_batches_claimed,
        "today_annotation_count": today_count,
        "daily_streak_threshold": DAILY_STREAK_THRESHOLD,
        "streak_at_risk": streak_at_risk,
        "grace_used": streak.grace_used,
        "next_tier": None,
    }

    if next_tier:
        info["next_tier"] = {
            "label": next_tier.label,
            "min_days": next_tier.min_days,
            "multiplier": next_tier.multiplier,
            "days_remaining": max(0, next_tier.min_days - streak.current_streak_days),
        }

    return info
