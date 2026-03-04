"""
Validator API endpoints for TuneForge.

POST /api/v1/validator/rounds              — Submit a complete validation round
POST /api/v1/validator/rounds/{id}/scores  — Update scores after scoring
GET  /api/v1/validator/rounds              — List validation rounds (paginated)
GET  /api/v1/validator/rounds/{id}         — Get round detail with audio entries
"""

import base64

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from tuneforge.api.validator_auth import require_validator_token
from tuneforge.api.models import (
    SubmitValidationRoundRequest,
    SubmitValidationRoundResponse,
    UpdateScoresRequest,
    UpdateScoresResponse,
    ValidationAudioInfo,
    ValidationRoundDetailResponse,
    ValidationRoundInfo,
    ValidationRoundListResponse,
)

router = APIRouter(
    prefix="/api/v1/validator",
    tags=["validator"],
    dependencies=[Depends(require_validator_token)],
)


@router.post("/rounds", response_model=SubmitValidationRoundResponse, status_code=201)
async def submit_validation_round(
    body: SubmitValidationRoundRequest,
) -> SubmitValidationRoundResponse:
    """Submit a complete validation round with miner audio responses."""
    from tuneforge.api.database.crud import (
        create_audio_blob,
        create_validation_audio,
        create_validation_round,
    )
    from tuneforge.api.server import app_state

    db = app_state.db
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    # Create the validation round
    vr = await create_validation_round(
        db,
        challenge_id=body.challenge_id,
        prompt=body.prompt,
        genre=body.genre,
        mood=body.mood,
        tempo_bpm=body.tempo_bpm,
        duration_seconds=body.duration_seconds,
        validator_hotkey=body.validator_hotkey or None,
    )

    audio_entries: list[ValidationAudioInfo] = []

    for entry in body.responses:
        try:
            audio_bytes = base64.b64decode(entry.audio_b64)
        except Exception:
            logger.warning(
                "Invalid base64 for miner UID {} in round {}",
                entry.miner_uid, body.challenge_id,
            )
            continue

        blob = await create_audio_blob(db, audio_bytes, fmt="wav")
        await create_validation_audio(
            db,
            round_id=vr.id,
            miner_uid=entry.miner_uid,
            audio_blob_id=blob.id,
            miner_hotkey=entry.miner_hotkey,
            generation_time_ms=entry.generation_time_ms,
        )
        audio_entries.append(
            ValidationAudioInfo(
                miner_uid=entry.miner_uid,
                miner_hotkey=entry.miner_hotkey,
                audio_blob_id=blob.id,
                generation_time_ms=entry.generation_time_ms,
                score=None,
            )
        )

    logger.info(
        "Validator {} submitted round {} ({} entries)",
        body.validator_hotkey[:16] if body.validator_hotkey else "unknown",
        vr.id,
        len(audio_entries),
    )

    # Auto-generate A/B annotation tasks from this round
    if len(audio_entries) >= 2:
        from tuneforge.api.database.crud import create_annotation_tasks_from_round

        created, skipped = await create_annotation_tasks_from_round(db, vr.id, quorum=5)
        if created > 0:
            logger.info(
                "Auto-generated {} annotation tasks ({} skipped) for round {}",
                created, skipped, vr.id,
            )

    return SubmitValidationRoundResponse(
        round_id=vr.id,
        challenge_id=body.challenge_id,
        audio_entries=audio_entries,
    )


@router.post("/rounds/{round_id}/scores", response_model=UpdateScoresResponse)
async def update_round_scores(
    round_id: str,
    body: UpdateScoresRequest,
) -> UpdateScoresResponse:
    """Update miner scores for a completed validation round."""
    from tuneforge.api.database.crud import update_validation_audio_scores
    from tuneforge.api.server import app_state

    db = app_state.db
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    scores_tuples = [(s.miner_uid, s.score) for s in body.scores]
    updated = await update_validation_audio_scores(db, round_id, scores_tuples)

    if updated == 0:
        raise HTTPException(status_code=404, detail="Round or scores not found")

    logger.info("Updated {} scores for round {}", updated, round_id)
    return UpdateScoresResponse(updated=updated)


@router.get("/rounds", response_model=ValidationRoundListResponse)
async def list_validation_rounds(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> ValidationRoundListResponse:
    """List validation rounds, newest first."""
    from tuneforge.api.database import total_pages
    from tuneforge.api.database.crud import get_validation_rounds
    from tuneforge.api.server import app_state

    db = app_state.db
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    rows, total = await get_validation_rounds(db, page=page, page_size=page_size)
    return ValidationRoundListResponse(
        rounds=[
            ValidationRoundInfo(
                id=r.id,
                challenge_id=r.challenge_id,
                prompt=r.prompt,
                genre=r.genre,
                mood=r.mood,
                tempo_bpm=r.tempo_bpm,
                duration_seconds=r.duration_seconds,
                validator_hotkey=r.validator_hotkey,
                created_at=r.created_at,
            )
            for r in rows
        ],
        total=total,
        page=page,
        pages=total_pages(total, page_size),
    )


@router.get("/rounds/{round_id}", response_model=ValidationRoundDetailResponse)
async def get_validation_round_detail(
    round_id: str,
) -> ValidationRoundDetailResponse:
    """Get a single validation round with all audio entries."""
    from tuneforge.api.database.crud import get_validation_audio_by_round
    from tuneforge.api.database.models import ValidationRoundRow
    from tuneforge.api.server import app_state

    db = app_state.db
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    async with db.session() as session:
        vr = await session.get(ValidationRoundRow, round_id)

    if vr is None:
        raise HTTPException(status_code=404, detail="Validation round not found")

    audio_rows = await get_validation_audio_by_round(db, round_id)

    return ValidationRoundDetailResponse(
        round=ValidationRoundInfo(
            id=vr.id,
            challenge_id=vr.challenge_id,
            prompt=vr.prompt,
            genre=vr.genre,
            mood=vr.mood,
            tempo_bpm=vr.tempo_bpm,
            duration_seconds=vr.duration_seconds,
            validator_hotkey=vr.validator_hotkey,
            created_at=vr.created_at,
        ),
        audio_entries=[
            ValidationAudioInfo(
                miner_uid=a.miner_uid,
                miner_hotkey=a.miner_hotkey,
                audio_blob_id=a.audio_blob_id,
                generation_time_ms=a.generation_time_ms,
                score=a.score,
            )
            for a in audio_rows
        ],
    )
