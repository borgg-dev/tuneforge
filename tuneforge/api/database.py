"""
Async SQLAlchemy database for TuneForge track metadata.

Uses aiosqlite for zero-config local persistence.
"""

import json
import math
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class TrackRow(Base):
    """Persisted track metadata."""

    __tablename__ = "tracks"

    id = Column(String(32), primary_key=True)
    request_id = Column(String(64), index=True, nullable=False)
    prompt = Column(Text, nullable=False)
    genre = Column(String(64), nullable=True)
    mood = Column(String(64), nullable=True)
    tempo_bpm = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=False)
    audio_path = Column(Text, nullable=False)
    format = Column(String(8), nullable=False, default="mp3")
    sample_rate = Column(Integer, nullable=False, default=32000)
    generation_time_ms = Column(Integer, nullable=False, default=0)
    miner_hotkey = Column(String(64), nullable=False, default="")
    scores_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class Database:
    """Async database wrapper for track metadata CRUD operations."""

    def __init__(self, url: str = "sqlite+aiosqlite:///./tuneforge.db") -> None:
        self._engine = create_async_engine(url, echo=False)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init_db(self) -> None:
        """Create tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized")

    async def close(self) -> None:
        """Dispose of the engine connection pool."""
        await self._engine.dispose()

    def _session(self) -> AsyncSession:
        return self._session_factory()

    async def create_track(
        self,
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
        )
        async with self._session() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
        logger.debug("Created track {} for request {}", track_id, request_id)
        return row

    async def get_track(self, track_id: str) -> TrackRow | None:
        """Fetch a single track by ID."""
        async with self._session() as session:
            return await session.get(TrackRow, track_id)

    async def search_tracks(
        self,
        genre: str | None = None,
        mood: str | None = None,
        min_tempo: int | None = None,
        max_tempo: int | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[TrackRow], int]:
        """Search tracks with optional filters.

        Returns (rows, total_count).
        """
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

        stmt = stmt.order_by(TrackRow.created_at.desc())
        offset = (page - 1) * page_size
        stmt = stmt.offset(offset).limit(page_size)

        async with self._session() as session:
            result = await session.execute(stmt)
            rows = list(result.scalars().all())
            count_result = await session.execute(count_stmt)
            total = count_result.scalar() or 0

        return rows, total

    async def get_track_count(self) -> int:
        """Return total number of tracks."""
        async with self._session() as session:
            result = await session.execute(select(func.count()).select_from(TrackRow))
            return result.scalar() or 0


def row_to_scores(row: TrackRow) -> dict[str, float]:
    """Parse the scores_json column into a dict."""
    try:
        return json.loads(row.scores_json)
    except (json.JSONDecodeError, TypeError):
        return {}


def total_pages(total: int, page_size: int) -> int:
    """Compute total page count."""
    return max(1, math.ceil(total / page_size))
