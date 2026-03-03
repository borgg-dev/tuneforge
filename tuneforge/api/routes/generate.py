"""
Music generation endpoint for the TuneForge API.

POST /api/v1/generate — queries subnet miners and returns generated audio.
GET /api/v1/generate/{request_id}/status — SSE stream for generation progress.
GET /api/v1/generate/{request_id} — poll generation status.
"""

import io
import json
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from loguru import logger

from tuneforge.api.auth import get_current_user, require_user
from tuneforge.api.credits import CreditCost
from tuneforge.api.database.models import UserRow
from tuneforge.api.models import (
    GenerateRequest,
    GenerateResponse,
    GenerationStatusResponse,
    TrackInfo,
)
from tuneforge.api.rate_limiter import check_rate_limit
from tuneforge.base.protocol import MusicGenerationSynapse

router = APIRouter(prefix="/api/v1", tags=["generate"])


def _extract_track_id(url: str) -> str:
    """Pull the hex track ID from a storage URL/path."""
    filename = url.rsplit("/", 1)[-1]
    return filename.rsplit(".", 1)[0]


def _convert_audio(raw_bytes: bytes, target_format: str, sample_rate: int) -> bytes:
    """Convert raw WAV/audio bytes to the requested output format."""
    if target_format == "wav":
        return raw_bytes
    try:
        from pydub import AudioSegment

        segment = AudioSegment.from_file(io.BytesIO(raw_bytes), format="wav")
        buf = io.BytesIO()
        segment.export(buf, format=target_format)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("Audio conversion to {} failed ({}), returning raw bytes", target_format, exc)
        return raw_bytes


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    user: UserRow | None = Depends(get_current_user),
) -> GenerateResponse:
    """Generate music by querying subnet miners.

    When a user is authenticated:
    1. Check credits (no deduction yet)
    2. Create generation record
    3. Publish SSE status updates
    4. Use organic query router
    5. Store and return results
    6. Charge credits only on success

    Falls back to legacy (anonymous) mode when no user is authenticated.
    """
    from tuneforge.api.server import app_state

    request_id = uuid.uuid4().hex
    t_start = time.perf_counter()

    settings = app_state.settings
    storage = app_state.storage
    db = app_state.db

    # --- Credit check (authenticated users only, no deduction) ---
    cost = CreditCost.GENERATION * request.num_variations

    if user is not None and app_state.credit_service is not None:
        has_enough = await app_state.credit_service.has_sufficient_credits(
            user.id, cost
        )
        if not has_enough:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Insufficient credits.",
            )

        # Create generation record
        from tuneforge.api.database import create_generation

        await create_generation(
            db,
            user_id=user.id,
            request_id=request_id,
            prompt=request.prompt,
            params_json=json.dumps({
                "genre": request.genre,
                "mood": request.mood,
                "tempo_bpm": request.tempo_bpm,
                "duration_seconds": request.duration_seconds,
                "key_signature": request.key_signature,
                "instruments": request.instruments,
                "format": request.format,
                "num_variations": request.num_variations,
            }),
            credits_reserved=cost,
        )

    # --- Publish SSE: queued ---
    redis = app_state.redis
    if redis:
        await redis.publish_status(request_id, "queued")

    try:
        # --- Build synapse ---
        synapse = MusicGenerationSynapse(
            prompt=request.prompt,
            genre=request.genre or "",
            mood=request.mood or "",
            tempo_bpm=request.tempo_bpm or 120,
            duration_seconds=request.duration_seconds,
            key_signature=request.key_signature,
            instruments=request.instruments,
            challenge_id=request_id,
            is_organic=True,
        )

        # --- Route generation ---
        tracks: list[TrackInfo] = []

        if app_state.organic_router is not None:
            # Use organic query router
            if redis:
                await redis.publish_status(request_id, "routing")

            if redis:
                await redis.publish_status(request_id, "generating")

            for _ in range(request.num_variations):
                audio_bytes, metadata = await app_state.organic_router.generate(
                    synapse, timeout=settings.generation_timeout
                )

                if audio_bytes is None:
                    continue

                gen_time = metadata.get("generation_time_ms", 0)
                sr = metadata.get("sample_rate", settings.generation_sample_rate)
                hotkey = metadata.get("miner_hotkey", "")

                scores: dict[str, float] = {}
                if gen_time:
                    scores["latency"] = round(gen_time / 1000, 3)

                converted = _convert_audio(audio_bytes, request.format, sr)

                audio_url = await storage.store(
                    converted,
                    request.format,
                    {"request_id": request_id, "miner": hotkey},
                )
                track_id = _extract_track_id(audio_url)

                from tuneforge.api.database import create_track

                await create_track(
                    db,
                    track_id=track_id,
                    request_id=request_id,
                    prompt=request.prompt,
                    audio_path=audio_url,
                    duration_seconds=request.duration_seconds,
                    fmt=request.format,
                    sample_rate=sr,
                    generation_time_ms=gen_time,
                    miner_hotkey=hotkey,
                    genre=request.genre,
                    mood=request.mood,
                    tempo_bpm=request.tempo_bpm,
                    scores=scores,
                    user_id=user.id if user else None,
                )

                tracks.append(
                    TrackInfo(
                        track_id=track_id,
                        audio_url=audio_url,
                        duration_seconds=request.duration_seconds,
                        sample_rate=sr,
                        format=request.format,
                        generation_time_ms=gen_time,
                        miner_hotkey=hotkey,
                        scores=scores,
                    )
                )

        else:
            # Legacy mode: direct dendrite query
            dendrite = app_state.dendrite
            metagraph = app_state.metagraph

            if dendrite is None or metagraph is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Validator not connected to the network yet.",
                )

            uids_incentives = sorted(
                enumerate(metagraph.I.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            candidate_uids = [
                uid for uid, _ in uids_incentives if metagraph.axons[uid].is_serving
            ]
            num_to_query = min(request.num_variations * 2, len(candidate_uids), 16)
            if num_to_query == 0:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="No serving miners available on the network.",
                )
            target_uids = candidate_uids[:num_to_query]
            target_axons = [metagraph.axons[uid] for uid in target_uids]

            logger.info("Generating request={} querying {} miners", request_id, len(target_axons))
            responses = await dendrite.forward(
                axons=target_axons,
                synapse=synapse,
                deserialize=False,
                timeout=settings.generation_timeout,
            )

            for uid, resp in zip(target_uids, responses):
                audio_bytes = resp.deserialize()
                if audio_bytes is None or len(audio_bytes) == 0:
                    continue

                gen_time = resp.generation_time_ms or 0
                sr = resp.sample_rate or settings.generation_sample_rate
                hotkey = metagraph.hotkeys[uid]

                scores: dict[str, float] = {}
                if resp.dendrite and resp.dendrite.process_time:
                    scores["latency"] = round(resp.dendrite.process_time, 3)

                converted = _convert_audio(audio_bytes, request.format, sr)

                audio_url = await storage.store(
                    converted,
                    request.format,
                    {"request_id": request_id, "miner": hotkey},
                )
                track_id = _extract_track_id(audio_url)

                from tuneforge.api.database import create_track

                await create_track(
                    db,
                    track_id=track_id,
                    request_id=request_id,
                    prompt=request.prompt,
                    audio_path=audio_url,
                    duration_seconds=request.duration_seconds,
                    fmt=request.format,
                    sample_rate=sr,
                    generation_time_ms=gen_time,
                    miner_hotkey=hotkey,
                    genre=request.genre,
                    mood=request.mood,
                    tempo_bpm=request.tempo_bpm,
                    scores=scores,
                    user_id=user.id if user else None,
                )

                tracks.append(
                    TrackInfo(
                        track_id=track_id,
                        audio_url=audio_url,
                        duration_seconds=request.duration_seconds,
                        sample_rate=sr,
                        format=request.format,
                        generation_time_ms=gen_time,
                        miner_hotkey=hotkey,
                        scores=scores,
                    )
                )

                if len(tracks) >= request.num_variations:
                    break

        if not tracks:
            if redis:
                await redis.publish_status(request_id, "failed", error="No valid audio generated")

            from tuneforge.api.database import update_generation_status

            if user is not None:
                await update_generation_status(db, request_id, "failed", error_message="No valid audio")

            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="No miners returned valid audio. Try again later.",
            )

        total_ms = int((time.perf_counter() - t_start) * 1000)

        # --- Charge credits only on success ---
        if user is not None and app_state.credit_service is not None:
            await app_state.credit_service.spend_credits(user.id, cost, request_id)

        # Update generation status
        if user is not None:
            from tuneforge.api.database import update_generation_status

            await update_generation_status(
                db, request_id, "completed", credits_spent=cost
            )

        # Publish SSE: completed
        if redis:
            await redis.publish_status(
                request_id, "completed",
                track_ids=[t.track_id for t in tracks],
            )

        # Legacy billing
        total_secs = sum(t.duration_seconds for t in tracks)
        app_state.billing.log_generation(
            user.id if user else "anonymous",
            request_id,
            len(tracks),
            total_secs,
        )

        logger.info("Request {} complete: {} tracks in {}ms", request_id, len(tracks), total_ms)

        return GenerateResponse(
            request_id=request_id,
            tracks=tracks,
            total_time_ms=total_ms,
        )

    except HTTPException as exc:
        if redis:
            await redis.publish_status(request_id, "failed", error=exc.detail)

        if user is not None:
            from tuneforge.api.database import update_generation_status

            await update_generation_status(db, request_id, "failed", error_message=exc.detail)

        raise
    except Exception as exc:
        if redis:
            await redis.publish_status(request_id, "failed", error=str(exc))

        if user is not None:
            from tuneforge.api.database import update_generation_status

            await update_generation_status(db, request_id, "failed", error_message=str(exc))

        logger.opt(exception=exc).error("Generation failed for request {}", request_id)
        raise


# ---------------------------------------------------------------------------
# SSE Generation Status
# ---------------------------------------------------------------------------


@router.get("/generate/{request_id}/status")
async def generation_status_stream(
    request_id: str,
    request: Request,
    token: str | None = Query(default=None, description="JWT token for SSE auth"),
) -> StreamingResponse:
    """SSE stream for real-time generation status updates.

    Since EventSource doesn't support custom headers, the JWT token
    can be passed as a query parameter.
    """
    from tuneforge.api.server import app_state

    # Authenticate via query param token
    if token:
        from tuneforge.api.jwt_auth import decode_token
        from tuneforge.api.database import get_user_by_id

        payload = decode_token(token, app_state.settings.jwt_secret)
        if payload is None or payload.type != "access":
            raise HTTPException(status_code=401, detail="Invalid token")

        user = await get_user_by_id(app_state.db, payload.sub)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")

    # Verify generation exists
    from tuneforge.api.database import get_generation

    generation = await get_generation(app_state.db, request_id)
    if generation is None:
        raise HTTPException(status_code=404, detail="Generation not found")

    if app_state.sse_manager is None:
        raise HTTPException(status_code=503, detail="SSE not available")

    return StreamingResponse(
        app_state.sse_manager.event_stream(request_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/generate/{request_id}", response_model=GenerationStatusResponse)
async def get_generation_status(
    request_id: str,
    user: UserRow = Depends(require_user),
) -> GenerationStatusResponse:
    """Poll generation status (for clients that can't use SSE)."""
    from tuneforge.api.database import get_generation, get_tracks_by_generation, row_to_scores
    from tuneforge.api.server import app_state

    generation = await get_generation(app_state.db, request_id)
    if generation is None or generation.user_id != user.id:
        raise HTTPException(status_code=404, detail="Generation not found")

    tracks: list[TrackInfo] = []
    if generation.status == "completed":
        track_rows = await get_tracks_by_generation(app_state.db, generation.id)
        for row in track_rows:
            tracks.append(
                TrackInfo(
                    track_id=row.id,
                    audio_url=row.audio_path,
                    duration_seconds=row.duration_seconds,
                    sample_rate=row.sample_rate,
                    format=row.format,
                    generation_time_ms=row.generation_time_ms,
                    miner_hotkey=row.miner_hotkey,
                    scores=row_to_scores(row),
                )
            )

    return GenerationStatusResponse(
        request_id=request_id,
        status=generation.status,
        tracks=tracks,
        created_at=generation.created_at,
        completed_at=generation.completed_at,
        error=generation.error_message,
    )
