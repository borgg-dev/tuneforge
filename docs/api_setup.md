# TuneForge API Server Setup

## Prerequisites

- Running TuneForge validator (or standalone with network access)
- Python 3.10+

## Installation

```bash
pip install -e .
```

## Configuration

```bash
TF_API_HOST=0.0.0.0
TF_API_PORT=8000
TF_API_RATE_LIMIT=60
TF_STORAGE_BACKEND=local
TF_STORAGE_PATH=./storage
TF_DB_URL=sqlite+aiosqlite:///./tuneforge.db
```

## Running

```bash
python -m tuneforge.api.server
```

Or with uvicorn:
```bash
uvicorn tuneforge.api.server:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /api/v1/generate

Generate music from a text prompt.

**Request:**
```json
{
  "prompt": "upbeat electronic track for a product video",
  "genre": "electronic",
  "mood": "energetic",
  "duration_seconds": 15.0,
  "num_variations": 2,
  "format": "mp3"
}
```

**Response:**
```json
{
  "request_id": "abc123",
  "tracks": [
    {
      "track_id": "def456",
      "audio_url": "/audio/def456.mp3",
      "duration_seconds": 15.0,
      "sample_rate": 32000,
      "format": "mp3",
      "generation_time_ms": 5200,
      "miner_hotkey": "5Hot...",
      "scores": {"latency": 1.23}
    }
  ],
  "total_time_ms": 6100
}
```

### GET /api/v1/tracks

Browse generated tracks with filtering.

**Query params:** `genre`, `mood`, `min_tempo`, `max_tempo`, `page`, `page_size`

### GET /health

Health check endpoint.

## Rate Limiting

Default: 60 requests per minute per API key. Configurable via `TF_API_RATE_LIMIT`.

## Authentication

Set API keys via environment:
```bash
TF_API_KEYS=key1,key2,key3
```

Include in requests as Bearer token:
```bash
curl -H "Authorization: Bearer key1" http://localhost:8000/api/v1/generate
```
