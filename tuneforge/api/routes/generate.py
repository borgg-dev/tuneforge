"""
Music generation endpoint for the TuneForge Subnet API.

Routes organic generation requests to miners via the organic query router.
No credits, no database, no SSE — that's all handled by the platform API.
"""

import io
import time
import uuid

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from tuneforge.api.models import GenerateRequest, GenerateResponse, TrackInfo
from tuneforge.base.protocol import MusicGenerationSynapse

router = APIRouter(prefix="/api/v1", tags=["generate"])


def _convert_audio(raw_bytes: bytes, target_format: str) -> bytes:
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
        logger.warning("Audio conversion to {} failed ({}), returning raw WAV", target_format, exc)
        return raw_bytes


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate music by routing to subnet miners via the organic query router."""
    from tuneforge.api.server import app_state

    request_id = uuid.uuid4().hex
    t_start = time.perf_counter()
    settings = app_state.settings

    # Build synapse
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

    tracks: list[TrackInfo] = []

    if app_state.organic_router is not None:
        for _ in range(request.num_variations):
            audio_bytes, metadata = await app_state.organic_router.generate(
                synapse, timeout=settings.generation_timeout
            )
            if audio_bytes is None:
                continue

            gen_time = metadata.get("generation_time_ms", 0)
            sr = metadata.get("sample_rate", settings.generation_sample_rate)
            hotkey = metadata.get("miner_hotkey", "")

            converted = _convert_audio(audio_bytes, request.format)

            tracks.append(
                TrackInfo(
                    track_id=uuid.uuid4().hex,
                    audio_url="",  # Platform handles storage
                    duration_seconds=request.duration_seconds,
                    sample_rate=sr,
                    format=request.format,
                    generation_time_ms=gen_time,
                    miner_hotkey=hotkey,
                    scores={"latency": round(gen_time / 1000, 3)} if gen_time else {},
                )
            )
    else:
        # Fallback: direct dendrite query
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

            converted = _convert_audio(audio_bytes, request.format)

            tracks.append(
                TrackInfo(
                    track_id=uuid.uuid4().hex,
                    audio_url="",
                    duration_seconds=request.duration_seconds,
                    sample_rate=sr,
                    format=request.format,
                    generation_time_ms=gen_time,
                    miner_hotkey=hotkey,
                    scores={"latency": round(resp.dendrite.process_time, 3)} if resp.dendrite and resp.dendrite.process_time else {},
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
    logger.info("Request {} complete: {} tracks in {}ms", request_id, len(tracks), total_ms)

    return GenerateResponse(
        request_id=request_id,
        tracks=tracks,
        total_time_ms=total_ms,
    )
