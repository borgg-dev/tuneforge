"""
Audio file storage backends for the TuneForge API.

Supports local filesystem and S3-compatible object storage.
"""

import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger


class StorageBackend(ABC):
    """Abstract interface for audio file storage."""

    @abstractmethod
    async def store(self, audio_bytes: bytes, fmt: str, metadata: dict | None = None) -> str:
        """Store audio data and return a public URL or path."""

    @abstractmethod
    async def get(self, track_id: str) -> bytes:
        """Retrieve raw audio bytes by track ID."""

    @abstractmethod
    async def delete(self, track_id: str) -> None:
        """Delete a stored audio file."""


class LocalStorage(StorageBackend):
    """Store audio files on the local filesystem."""

    def __init__(self, base_path: str = "./storage", base_url: str = "/static/audio") -> None:
        self.base_path = Path(base_path)
        self.base_url = base_url.rstrip("/")
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info("LocalStorage initialized at {}", self.base_path.resolve())

    async def store(self, audio_bytes: bytes, fmt: str, metadata: dict | None = None) -> str:
        track_id = uuid.uuid4().hex
        filename = f"{track_id}.{fmt}"
        filepath = self.base_path / filename
        filepath.write_bytes(audio_bytes)
        url = f"{self.base_url}/{filename}"
        logger.debug("Stored {} ({} bytes) → {}", filename, len(audio_bytes), url)
        return url

    async def get(self, track_id: str) -> bytes:
        matches = list(self.base_path.glob(f"{track_id}.*"))
        if not matches:
            raise FileNotFoundError(f"Track {track_id} not found in local storage")
        return matches[0].read_bytes()

    async def delete(self, track_id: str) -> None:
        for p in self.base_path.glob(f"{track_id}.*"):
            p.unlink()
            logger.debug("Deleted {}", p.name)


class S3Storage(StorageBackend):
    """Store audio files in an S3-compatible bucket."""

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
        prefix: str = "audio/",
        presigned_expiry: int = 3600,
    ) -> None:
        try:
            import aioboto3
        except ImportError as exc:
            raise ImportError("Install aioboto3 for S3 storage: pip install aioboto3") from exc

        self.bucket = bucket
        self.region = region
        self.prefix = prefix
        self.presigned_expiry = presigned_expiry

        session_kwargs: dict = {}
        if access_key and secret_key:
            session_kwargs["aws_access_key_id"] = access_key
            session_kwargs["aws_secret_access_key"] = secret_key
        self._session = aioboto3.Session(**session_kwargs)
        self._region = region
        logger.info("S3Storage initialized for bucket={} region={}", bucket, region)

    async def store(self, audio_bytes: bytes, fmt: str, metadata: dict | None = None) -> str:
        track_id = uuid.uuid4().hex
        key = f"{self.prefix}{track_id}.{fmt}"
        content_type = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
        }.get(fmt, "application/octet-stream")

        async with self._session.client("s3", region_name=self._region) as s3:
            await s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=audio_bytes,
                ContentType=content_type,
                Metadata=metadata or {},
            )
            url = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=self.presigned_expiry,
            )
        logger.debug("Stored s3://{}/{} ({} bytes)", self.bucket, key, len(audio_bytes))
        return url

    async def get(self, track_id: str) -> bytes:
        async with self._session.client("s3", region_name=self._region) as s3:
            # Try common audio extensions
            for ext in ("mp3", "wav", "ogg", "flac"):
                key = f"{self.prefix}{track_id}.{ext}"
                try:
                    resp = await s3.get_object(Bucket=self.bucket, Key=key)
                    return await resp["Body"].read()
                except s3.exceptions.NoSuchKey:
                    continue
        raise FileNotFoundError(f"Track {track_id} not found in S3 bucket {self.bucket}")

    async def delete(self, track_id: str) -> None:
        async with self._session.client("s3", region_name=self._region) as s3:
            for ext in ("mp3", "wav", "ogg", "flac"):
                key = f"{self.prefix}{track_id}.{ext}"
                try:
                    await s3.delete_object(Bucket=self.bucket, Key=key)
                    logger.debug("Deleted s3://{}/{}", self.bucket, key)
                except Exception:
                    pass


class PostgresStorage(StorageBackend):
    """Store audio files as bytea in PostgreSQL via the audio_blobs table."""

    def __init__(self, db, base_url: str = "/api/v1/audio") -> None:
        self._db = db
        self._base_url = base_url.rstrip("/")
        logger.info("PostgresStorage initialized")

    async def store(self, audio_bytes: bytes, fmt: str, metadata: dict | None = None) -> str:
        from tuneforge.api.database.crud import create_audio_blob

        content_type = {
            "mp3": "audio/mpeg", "wav": "audio/wav",
            "ogg": "audio/ogg", "flac": "audio/flac",
        }.get(fmt, "application/octet-stream")

        blob = await create_audio_blob(
            self._db, audio_bytes, fmt=fmt, content_type=content_type,
        )
        url = f"{self._base_url}/{blob.id}.{fmt}"
        logger.debug("Stored audio blob {} ({} bytes)", blob.id, len(audio_bytes))
        return url

    async def get(self, track_id: str) -> bytes:
        from tuneforge.api.database.crud import get_audio_blob

        blob = await get_audio_blob(self._db, track_id)
        if blob is None:
            raise FileNotFoundError(f"Audio blob {track_id} not found")
        return blob.audio_data

    async def delete(self, track_id: str) -> None:
        from tuneforge.api.database.crud import delete_audio_blob

        await delete_audio_blob(self._db, track_id)
        logger.debug("Deleted audio blob {}", track_id)


def get_storage_backend(
    backend: str = "local",
    storage_path: str = "./storage",
    base_url: str = "/static/audio",
    s3_bucket: str | None = None,
    s3_region: str | None = None,
    s3_access_key: str | None = None,
    s3_secret_key: str | None = None,
    db=None,
) -> StorageBackend:
    """Factory: create the appropriate storage backend from config."""
    if backend == "postgres":
        if db is None:
            raise ValueError("PostgreSQL storage requires a Database instance")
        return PostgresStorage(db=db, base_url="/api/v1/audio")
    if backend == "s3":
        if not s3_bucket:
            raise ValueError("S3 storage requires s3_bucket to be configured")
        return S3Storage(
            bucket=s3_bucket,
            region=s3_region or "us-east-1",
            access_key=s3_access_key,
            secret_key=s3_secret_key,
        )
    return LocalStorage(base_path=storage_path, base_url=base_url)
