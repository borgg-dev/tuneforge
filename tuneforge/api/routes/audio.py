"""
Audio streaming endpoint for TuneForge.

GET /api/v1/audio/{filename} — stream audio from PostgreSQL with Range support.
"""

from fastapi import APIRouter, HTTPException, Request, Response

router = APIRouter(prefix="/api/v1", tags=["audio"])


@router.get("/audio/{filename}")
async def stream_audio(filename: str, request: Request) -> Response:
    """Stream an audio blob from PostgreSQL."""
    from tuneforge.api.server import app_state
    from tuneforge.api.database.crud import get_audio_blob

    # Parse blob_id from filename (e.g., "abc123def456.mp3")
    parts = filename.rsplit(".", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid filename format")
    blob_id = parts[0]

    blob = await get_audio_blob(app_state.db, blob_id)
    if blob is None:
        raise HTTPException(status_code=404, detail="Audio not found")

    audio_data = blob.audio_data
    content_type = blob.content_type
    size = blob.size_bytes

    # Handle HTTP Range requests for seeking
    range_header = request.headers.get("range")
    if range_header:
        range_spec = range_header.replace("bytes=", "")
        range_parts = range_spec.split("-")
        start = int(range_parts[0]) if range_parts[0] else 0
        end = int(range_parts[1]) if len(range_parts) > 1 and range_parts[1] else size - 1
        end = min(end, size - 1)
        chunk = audio_data[start:end + 1]

        return Response(
            content=chunk,
            status_code=206,
            headers={
                "Content-Type": content_type,
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Content-Length": str(len(chunk)),
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=86400",
            },
        )

    return Response(
        content=audio_data,
        status_code=200,
        headers={
            "Content-Type": content_type,
            "Content-Length": str(size),
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=86400",
        },
    )
