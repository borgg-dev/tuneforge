"""
Annotation API endpoints for TuneForge crowd preference labeling.

POST /api/v1/annotations/tasks/generate        — Generate A/B tasks from rounds
GET  /api/v1/annotations/next                  — Get next task for annotator
POST /api/v1/annotations/tasks/{id}/vote       — Submit a preference vote
GET  /api/v1/annotations/tasks                 — List tasks (admin)
GET  /api/v1/annotations/stats                 — Annotation statistics
GET  /api/v1/annotations/export                — Export resolved annotations as JSONL
POST /api/v1/annotations/model                 — Upload trained preference model
GET  /api/v1/annotations/model/latest          — Download latest preference model
"""

import hashlib
import math

from fastapi import APIRouter, Depends, HTTPException, Query, Response, UploadFile, status
from loguru import logger

from tuneforge.api.auth import require_admin, require_user
from tuneforge.api.models import (
    AgreementStats,
    AnnotationExportEntry,
    AnnotationProgress,
    AnnotationTaskDetail,
    AnnotationTaskInfo,
    AnnotationTaskListResponse,
    AnnotatorListResponse,
    AnnotatorReliabilityInfo,
    CreateGoldTaskRequest,
    GenerateAnnotationTasksRequest,
    GenerateAnnotationTasksResponse,
    GoldTaskListResponse,
    GoldTaskResponse,
    MilestoneInfo,
    MilestoneProgressResponse,
    MilestoneUnlocked,
    NextMilestoneInfo,
    NextStreakTierInfo,
    PreferenceModelInfo,
    RecurringRewardEarned,
    StreakInfo,
    SubmitVoteRequest,
    VoteResponse,
)

router = APIRouter(
    prefix="/api/v1/annotations",
    tags=["annotations"],
)


def _get_db():
    """Get the database instance from app state."""
    from tuneforge.api.server import app_state

    db = app_state.db
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


# ---------------------------------------------------------------------------
# Task generation (admin / service token)
# ---------------------------------------------------------------------------


@router.post(
    "/tasks/generate",
    response_model=GenerateAnnotationTasksResponse,
    status_code=201,
)
async def generate_tasks(
    body: GenerateAnnotationTasksRequest,
    user=Depends(require_user),
) -> GenerateAnnotationTasksResponse:
    """Generate A/B annotation tasks from validation rounds."""
    from tuneforge.api.database.crud import (
        create_annotation_tasks_from_round,
        get_validation_rounds,
    )

    db = _get_db()
    total_created = 0
    total_skipped = 0

    if body.round_ids:
        round_ids = body.round_ids
    else:
        # Get all rounds
        rounds, _ = await get_validation_rounds(db, page=1, page_size=10000)
        round_ids = [r.id for r in rounds]

    for round_id in round_ids:
        created, skipped = await create_annotation_tasks_from_round(
            db, round_id, quorum=body.quorum,
        )
        total_created += created
        total_skipped += skipped

    logger.info(
        "Generated {} annotation tasks ({} skipped) from {} rounds",
        total_created, total_skipped, len(round_ids),
    )

    return GenerateAnnotationTasksResponse(
        tasks_created=total_created,
        tasks_skipped=total_skipped,
        total_tasks=total_created + total_skipped,
    )


# ---------------------------------------------------------------------------
# Annotator endpoints
# ---------------------------------------------------------------------------


@router.get("/next")
async def get_next_task(
    user=Depends(require_user),
):
    """Fetch the next open task for the current user."""
    from tuneforge.api.database.crud import (
        get_next_annotation_task,
        get_user_annotation_progress,
    )
    from tuneforge.api.database.models import ValidationAudioRow, ValidationRoundRow

    db = _get_db()
    task = await get_next_annotation_task(db, user.id)

    if task is None:
        return Response(status_code=204)

    # Resolve audio URLs and round metadata
    async with db.session() as session:
        audio_a = await session.get(ValidationAudioRow, task.audio_a_id)
        audio_b = await session.get(ValidationAudioRow, task.audio_b_id)
        round_row = await session.get(ValidationRoundRow, task.round_id)

    if not audio_a or not audio_b or not round_row:
        raise HTTPException(status_code=500, detail="Task references missing data")

    progress_data = await get_user_annotation_progress(db, user.id)

    return AnnotationTaskDetail(
        task_id=task.id,
        prompt=round_row.prompt,
        genre=round_row.genre,
        mood=round_row.mood,
        tempo_bpm=round_row.tempo_bpm,
        duration_seconds=round_row.duration_seconds,
        audio_a_url=f"/api/v1/audio/{audio_a.audio_blob_id}.wav",
        audio_b_url=f"/api/v1/audio/{audio_b.audio_blob_id}.wav",
        progress=AnnotationProgress(**progress_data),
    )


@router.post("/tasks/{task_id}/vote", response_model=VoteResponse)
async def submit_vote(
    task_id: str,
    body: SubmitVoteRequest,
    user=Depends(require_user),
) -> VoteResponse:
    """Submit a preference vote on a task."""
    from sqlalchemy.exc import IntegrityError

    from tuneforge.api.database.crud import (
        MIN_LISTEN_DURATION_MS,
        check_and_claim_milestones,
        get_annotation_task,
        get_next_annotation_task,
        get_user_annotation_progress,
        submit_annotation,
    )
    from tuneforge.api.database.models import ValidationAudioRow, ValidationRoundRow
    from tuneforge.api.milestones import MILESTONE_BY_KEY

    db = _get_db()

    # Minimum listen time enforcement
    if body.duration_ms is not None and body.duration_ms < MIN_LISTEN_DURATION_MS:
        raise HTTPException(
            status_code=422,
            detail=f"Please listen to both audio clips before voting. Minimum time: {MIN_LISTEN_DURATION_MS / 1000:.0f} seconds.",
        )

    # Verify task exists and is open
    task = await get_annotation_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != "open":
        raise HTTPException(status_code=409, detail="Task already resolved")

    try:
        annotation = await submit_annotation(
            db, task_id, user.id, body.choice, body.duration_ms,
        )
    except IntegrityError:
        raise HTTPException(status_code=409, detail="Already voted on this task")

    # Re-fetch task status after vote (aggregation may have changed it)
    task = await get_annotation_task(db, task_id)

    # Milestone + recurring reward check — only for trusted, non-gold votes
    milestone_unlocked = None
    recurring_reward = None
    if annotation.is_trusted and not annotation.is_gold_response:
        claimed = await check_and_claim_milestones(db, user.id)
        if claimed:
            last = claimed[-1]
            m_def = MILESTONE_BY_KEY.get(last.milestone_key)
            if m_def:
                milestone_unlocked = MilestoneUnlocked(
                    key=last.milestone_key,
                    label=m_def.label,
                    credits_awarded=last.credits_awarded,
                    grants_pro=m_def.grants_pro_days > 0,
                )

        # Update daily streak
        from tuneforge.api.database.crud import (
            check_and_claim_recurring_reward,
            update_streak,
        )
        from tuneforge.api.milestones import get_streak_tier

        await update_streak(db, user.id)

        # Check for recurring batch reward
        reward = await check_and_claim_recurring_reward(db, user.id)
        if reward:
            credits_awarded, batches = reward
            from tuneforge.api.database.crud import get_or_create_streak

            streak_row = await get_or_create_streak(db, user.id)
            tier = get_streak_tier(streak_row.current_streak_days)
            recurring_reward = RecurringRewardEarned(
                credits_awarded=credits_awarded,
                batches=batches,
                streak_days=streak_row.current_streak_days,
                multiplier=tier.multiplier,
                tier_label=tier.label,
            )

    # Pre-fetch next task for seamless UX
    next_task_row = await get_next_annotation_task(db, user.id)
    next_task = None
    if next_task_row:
        async with db.session() as session:
            audio_a = await session.get(ValidationAudioRow, next_task_row.audio_a_id)
            audio_b = await session.get(ValidationAudioRow, next_task_row.audio_b_id)
            round_row = await session.get(ValidationRoundRow, next_task_row.round_id)

        if audio_a and audio_b and round_row:
            progress_data = await get_user_annotation_progress(db, user.id)
            next_task = AnnotationTaskDetail(
                task_id=next_task_row.id,
                prompt=round_row.prompt,
                genre=round_row.genre,
                mood=round_row.mood,
                tempo_bpm=round_row.tempo_bpm,
                duration_seconds=round_row.duration_seconds,
                audio_a_url=f"/api/v1/audio/{audio_a.audio_blob_id}.wav",
                audio_b_url=f"/api/v1/audio/{audio_b.audio_blob_id}.wav",
                progress=AnnotationProgress(**progress_data),
            )

    return VoteResponse(
        recorded=True,
        task_status=task.status if task else "open",
        next_task=next_task,
        milestone_unlocked=milestone_unlocked,
        recurring_reward=recurring_reward,
    )


# ---------------------------------------------------------------------------
# Milestone progress
# ---------------------------------------------------------------------------


@router.get("/milestones", response_model=MilestoneProgressResponse)
async def get_milestones(
    user=Depends(require_user),
) -> MilestoneProgressResponse:
    """Get the user's annotation milestone progress."""
    from tuneforge.api.database.crud import (
        get_claimed_milestone_keys,
        get_streak_info,
        get_user_trusted_annotation_count,
    )
    from tuneforge.api.milestones import MILESTONES, get_next_milestone

    db = _get_db()
    trusted_count = await get_user_trusted_annotation_count(db, user.id)
    claimed_keys = await get_claimed_milestone_keys(db, user.id)
    next_ms = get_next_milestone(trusted_count, claimed_keys)

    completed = [
        MilestoneInfo(
            key=m.key,
            label=m.label,
            threshold=m.threshold,
            credits=m.credits,
            grants_pro=m.grants_pro_days > 0,
            is_claimed=True,
        )
        for m in MILESTONES
        if m.key in claimed_keys
    ]

    next_info = None
    if next_ms:
        next_info = NextMilestoneInfo(
            key=next_ms.key,
            label=next_ms.label,
            threshold=next_ms.threshold,
            credits=next_ms.credits,
            grants_pro=next_ms.grants_pro_days > 0,
            current_count=trusted_count,
            remaining=max(0, next_ms.threshold - trusted_count),
        )

    # Streak info for post-milestone users
    streak_data = await get_streak_info(db, user.id)
    streak = None
    if streak_data.get("is_recurring_eligible"):
        next_tier_data = streak_data.get("next_tier")
        next_tier = NextStreakTierInfo(**next_tier_data) if next_tier_data else None
        streak = StreakInfo(
            **{k: v for k, v in streak_data.items() if k != "next_tier"},
            next_tier=next_tier,
        )
    else:
        streak = StreakInfo(is_recurring_eligible=False)

    return MilestoneProgressResponse(
        trusted_annotation_count=trusted_count,
        completed_milestones=completed,
        next_milestone=next_info,
        streak=streak,
    )


# ---------------------------------------------------------------------------
# Admin / stats endpoints
# ---------------------------------------------------------------------------


@router.get("/tasks", response_model=AnnotationTaskListResponse)
async def list_tasks(
    status_filter: str | None = Query(default=None, alias="status"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    user=Depends(require_user),
) -> AnnotationTaskListResponse:
    """List annotation tasks with optional status filter."""
    from tuneforge.api.database.crud import get_annotation_tasks
    from tuneforge.api.database.models import ValidationAudioRow, ValidationRoundRow

    db = _get_db()
    rows, total = await get_annotation_tasks(db, status=status_filter, page=page, page_size=page_size)

    tasks = []
    async with db.session() as session:
        for t in rows:
            audio_a = await session.get(ValidationAudioRow, t.audio_a_id)
            audio_b = await session.get(ValidationAudioRow, t.audio_b_id)
            round_row = await session.get(ValidationRoundRow, t.round_id)
            tasks.append(AnnotationTaskInfo(
                id=t.id,
                round_id=t.round_id,
                prompt=round_row.prompt if round_row else "",
                genre=round_row.genre if round_row else None,
                audio_a_blob_id=audio_a.audio_blob_id if audio_a else "",
                audio_b_blob_id=audio_b.audio_blob_id if audio_b else "",
                quorum=t.quorum,
                vote_count=t.vote_count,
                status=t.status,
                winner=t.winner,
                created_at=t.created_at,
            ))

    return AnnotationTaskListResponse(
        tasks=tasks,
        total=total,
        page=page,
        pages=math.ceil(total / page_size) if total > 0 else 1,
    )


@router.get("/stats", response_model=AgreementStats)
async def get_stats(
    user=Depends(require_user),
) -> AgreementStats:
    """Get annotation statistics."""
    from tuneforge.api.database.crud import get_annotation_stats

    db = _get_db()
    stats = await get_annotation_stats(db)
    return AgreementStats(**stats)


# ---------------------------------------------------------------------------
# Export for training
# ---------------------------------------------------------------------------


@router.get("/export")
async def export_annotations(
    user=Depends(require_user),
):
    """Export resolved annotations as JSON for training pipeline."""
    from tuneforge.api.database.crud import export_resolved_annotations

    db = _get_db()
    entries = await export_resolved_annotations(db)

    return [
        AnnotationExportEntry(
            pair_id=e["pair_id"],
            audio_a=f"/api/v1/audio/{e['audio_a_blob_id']}.wav",
            audio_b=f"/api/v1/audio/{e['audio_b_blob_id']}.wav",
            preferred=e["preferred"],
            confidence=e.get("confidence", 1.0),
            n_trusted_votes=e.get("n_trusted_votes", 0),
            challenge_id=e["challenge_id"],
            prompt=e["prompt"],
            genre=e.get("genre"),
        )
        for e in entries
    ]


# ---------------------------------------------------------------------------
# Preference model management
# ---------------------------------------------------------------------------


@router.post("/model", response_model=PreferenceModelInfo, status_code=201)
async def upload_model(
    file: UploadFile,
    val_accuracy: float | None = None,
    n_train_pairs: int | None = None,
    user=Depends(require_user),
) -> PreferenceModelInfo:
    """Upload a trained preference model checkpoint."""
    from tuneforge.api.database.crud import create_preference_model

    db = _get_db()
    data = await file.read()

    if len(data) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="Model file too large")

    sha = hashlib.sha256(data).hexdigest()
    row = await create_preference_model(
        db,
        model_data=data,
        sha256=sha,
        val_accuracy=val_accuracy,
        n_train_pairs=n_train_pairs,
    )

    return PreferenceModelInfo(
        id=row.id,
        version=row.version,
        sha256=row.sha256,
        val_accuracy=row.val_accuracy,
        n_train_pairs=row.n_train_pairs,
        is_active=row.is_active,
        created_at=row.created_at,
    )


@router.get("/model/latest")
async def get_latest_model(
    user=Depends(require_user),
):
    """Download the latest active preference model checkpoint."""
    from tuneforge.api.database.crud import get_active_preference_model

    db = _get_db()
    model = await get_active_preference_model(db)

    if model is None or model.model_data is None:
        raise HTTPException(status_code=404, detail="No trained model available")

    return Response(
        content=model.model_data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="preference_head_v{model.version}.pt"',
            "X-Model-Version": str(model.version),
            "X-Model-SHA256": model.sha256,
            "Content-Length": str(len(model.model_data)),
        },
    )


# ---------------------------------------------------------------------------
# Gold standard quality control (admin)
# ---------------------------------------------------------------------------


@router.post("/gold", response_model=GoldTaskResponse, status_code=201)
async def create_gold(
    body: CreateGoldTaskRequest,
    user=Depends(require_admin),
) -> GoldTaskResponse:
    """Create a gold/honeypot annotation task with a known answer."""
    from tuneforge.api.database.crud import create_gold_task

    db = _get_db()
    task = await create_gold_task(
        db,
        round_id=body.round_id,
        audio_a_id=body.audio_a_id,
        audio_b_id=body.audio_b_id,
        gold_answer=body.gold_answer,
    )
    return GoldTaskResponse(
        task_id=task.id,
        audio_a_id=task.audio_a_id,
        audio_b_id=task.audio_b_id,
        gold_answer=task.gold_answer,
        vote_count=task.vote_count,
        gold_accuracy=None,
        created_at=task.created_at,
    )


@router.get("/gold", response_model=GoldTaskListResponse)
async def list_gold(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    user=Depends(require_admin),
) -> GoldTaskListResponse:
    """List gold tasks with accuracy stats."""
    from tuneforge.api.database.crud import get_gold_tasks

    db = _get_db()
    tasks, total = await get_gold_tasks(db, page=page, page_size=page_size)
    pages = math.ceil(total / page_size) if total > 0 else 1
    return GoldTaskListResponse(
        tasks=[GoldTaskResponse(**t) for t in tasks],
        total=total,
        page=page,
        pages=pages,
    )


@router.delete("/gold/{task_id}", status_code=204)
async def remove_gold(
    task_id: str,
    user=Depends(require_admin),
):
    """Delete a gold task."""
    from tuneforge.api.database.crud import delete_gold_task

    db = _get_db()
    deleted = await delete_gold_task(db, task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Gold task not found")


# ---------------------------------------------------------------------------
# Admin: Annotator management
# ---------------------------------------------------------------------------


@router.get("/admin/annotators", response_model=AnnotatorListResponse)
async def list_annotators(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    flagged_only: bool = Query(default=False),
    user=Depends(require_admin),
) -> AnnotatorListResponse:
    """List annotators with reliability scores."""
    from tuneforge.api.database.crud import get_all_annotator_reliabilities

    db = _get_db()
    annotators, total = await get_all_annotator_reliabilities(
        db, page=page, page_size=page_size, flagged_only=flagged_only,
    )
    pages = math.ceil(total / page_size) if total > 0 else 1
    return AnnotatorListResponse(
        annotators=[AnnotatorReliabilityInfo(**a) for a in annotators],
        total=total,
        page=page,
        pages=pages,
    )


@router.post("/admin/annotators/{user_id}/flag", status_code=204)
async def flag_user(
    user_id: str,
    user=Depends(require_admin),
):
    """Manually flag an annotator as unreliable."""
    from tuneforge.api.database.crud import flag_annotator

    db = _get_db()
    await flag_annotator(db, user_id)


@router.post("/admin/annotators/{user_id}/unflag", status_code=204)
async def unflag_user(
    user_id: str,
    user=Depends(require_admin),
):
    """Manually unflag an annotator and restore trust."""
    from tuneforge.api.database.crud import unflag_annotator

    db = _get_db()
    await unflag_annotator(db, user_id)
