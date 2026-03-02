"""
Music generation endpoint for the TuneForge API.

POST /api/v1/generate — queries subnet miners and returns generated audio.
"""

import base64
import io
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from tuneforge.api.auth import get_api_key
from tuneforge.api.models import GenerateRequest, GenerateResponse, TrackInfo
from tuneforge.api.rate_limiter import check_rate_limit
from tuneforge.base.protocol import MusicGenerationSynapse

router = APIRouter(prefix="/api/v1", tags=["generate"])


def _extract_track_id(url: str) -> str:
    """Pull the hex track ID from a storage URL/path."""
    filename = url.rsplit("/", 1)[-1]
    return filename.rsplit(".", 1)[0]


def _convert_audio(raw_bytes: bytes, target_format: str, sample_rate: int) -> bytes:
    """Convert raw WAV/audio bytes to the requested output format.

    Falls back to returning raw bytes if pydub is not installed or
    the conversion fails.
    """
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
    api_key: str = Depends(check_rate_limit),
) -> GenerateResponse:
    """Generate music by querying subnet miners.

    1. Select top miners from the metagraph
    2. Build a MusicGenerationSynapse from the request
    3. Query miners via dendrite
    4. Score and rank responses
    5. Convert audio to the requested format
    6. Store audio files and persist metadata
    7. Return download URLs
    """
    from tuneforge.api.server import app_state

    request_id = uuid.uuid4().hex
    t_start = time.perf_counter()

    settings = app_state.settings
    dendrite = app_state.dendrite
    metagraph = app_state.metagraph
    storage = app_state.storage
    db = app_state.db
    billing = app_state.billing

    if dendrite is None or metagraph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Validator not connected to the network yet.",
        )

    # --- 1. Select top miners by incentive ---
    uids_incentives = sorted(
        enumerate(metagraph.I.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    candidate_uids = [
        uid for uid, _ in uids_incentives
        if metagraph.axons[uid].is_serving
    ]
    num_to_query = min(request.num_variations * 2, len(candidate_uids), 16)
    if num_to_query == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No serving miners available on the network.",
        )
    target_uids = candidate_uids[:num_to_query]
    target_axons = [metagraph.axons[uid] for uid in target_uids]

    # --- 2. Build synapse ---
    synapse = MusicGenerationSynapse(
        prompt=request.prompt,
        genre=request.genre or "",
        mood=request.mood or "",
        tempo_bpm=request.tempo_bpm or 120,
        duration_seconds=request.duration_seconds,
        key_signature=request.key_signature,
        instruments=request.instruments,
        challenge_id=request_id,
    )

    # --- 3. Query miners ---
    logger.info("Generating request={} querying {} miners", request_id, len(target_axons))
    responses: list[MusicGenerationSynapse] = await dendrite.forward(
        axons=target_axons,
        synapse=synapse,
        deserialize=False,
        timeout=settings.generation_timeout,
    )

    # --- 4. Collect successful responses ---
    tracks: list[TrackInfo] = []
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

        # --- 5. Convert format ---
        converted = _convert_audio(audio_bytes, request.format, sr)

        # --- 6. Store ---
        audio_url = await storage.store(
            converted,
            request.format,
            {"request_id": request_id, "miner": hotkey},
        )
        track_id = _extract_track_id(audio_url)

        # --- 7. Persist to DB ---
        await db.create_track(
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
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No miners returned valid audio. Try again later.",
        )

    total_ms = int((time.perf_counter() - t_start) * 1000)

    # --- Bill usage ---
    total_secs = sum(t.duration_seconds for t in tracks)
    billing.log_generation(api_key, request_id, len(tracks), total_secs)

    logger.info(
        "Request {} complete: {} tracks in {}ms",
        request_id,
        len(tracks),
        total_ms,
    )

    return GenerateResponse(
        request_id=request_id,
        tracks=tracks,
        total_time_ms=total_ms,
    )
