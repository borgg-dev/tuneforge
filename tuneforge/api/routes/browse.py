"""
Track browsing endpoints for the TuneForge API.

GET /api/v1/tracks    — paginated search with filters (user-scoped when authenticated)
GET /api/v1/tracks/{track_id} — single track detail
"""

import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from tuneforge.api.auth import get_current_user, require_user
from tuneforge.api.database import row_to_scores, total_pages
from tuneforge.api.database.models import UserRow
from tuneforge.api.models import BrowseResponse, TrackMetadata

router = APIRouter(prefix="/api/v1", tags=["browse"])


@router.get("/tracks", response_model=BrowseResponse)
async def browse_tracks(
    genre: str | None = Query(default=None, description="Filter by genre"),
    mood: str | None = Query(default=None, description="Filter by mood"),
    min_tempo: int | None = Query(default=None, ge=20, le=300, description="Minimum tempo BPM"),
    max_tempo: int | None = Query(default=None, ge=20, le=300, description="Maximum tempo BPM"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    user: UserRow | None = Depends(get_current_user),
) -> BrowseResponse:
    """Browse generated tracks with optional filters.

    When authenticated, returns only the user's own tracks.
    """
    from tuneforge.api.database import search_tracks
    from tuneforge.api.server import app_state

    rows, total = await search_tracks(
        app_state.db,
        genre=genre,
        mood=mood,
        min_tempo=min_tempo,
        max_tempo=max_tempo,
        user_id=user.id if user else None,
        page=page,
        page_size=page_size,
    )

    tracks = [
        TrackMetadata(
            track_id=row.id,
            prompt=row.prompt,
            genre=row.genre,
            mood=row.mood,
            tempo_bpm=row.tempo_bpm,
            duration_seconds=row.duration_seconds,
            audio_url=row.audio_path,
            scores=row_to_scores(row),
            miner_hotkey=row.miner_hotkey,
            created_at=row.created_at,
        )
        for row in rows
    ]

    return BrowseResponse(
        tracks=tracks,
        total=total,
        page=page,
        pages=total_pages(total, page_size),
    )


@router.get("/tracks/{track_id}", response_model=TrackMetadata)
async def get_track(
    track_id: str,
    user: UserRow | None = Depends(get_current_user),
) -> TrackMetadata:
    """Get details for a single track."""
    from tuneforge.api.server import app_state

    row = await app_state.db.get_track(track_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Track {track_id} not found.",
        )

    return TrackMetadata(
        track_id=row.id,
        prompt=row.prompt,
        genre=row.genre,
        mood=row.mood,
        tempo_bpm=row.tempo_bpm,
        duration_seconds=row.duration_seconds,
        audio_url=row.audio_path,
        scores=row_to_scores(row),
        miner_hotkey=row.miner_hotkey,
        created_at=row.created_at,
    )


@router.delete("/tracks/{track_id}", status_code=204)
async def delete_track(
    track_id: str,
    user: UserRow = Depends(require_user),
) -> None:
    """Delete a track owned by the authenticated user."""
    from tuneforge.api.database.crud import delete_track as crud_delete_track, get_track
    from tuneforge.api.server import app_state

    # Get track first to find audio blob
    track = await get_track(app_state.db, track_id)
    if track is None or track.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Track not found or not owned by you.",
        )

    deleted = await crud_delete_track(app_state.db, track_id, user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Track not found or not owned by you.",
        )

    # Clean up audio blob
    if track.audio_blob_id and app_state.storage:
        try:
            await app_state.storage.delete(track.audio_blob_id)
        except Exception:
            pass  # Blob cleanup is best-effort
