"""
Async database engine and session management for TuneForge.

Uses PostgreSQL (asyncpg) as the database backend.
"""

import math
from contextlib import asynccontextmanager
from typing import AsyncIterator

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tuneforge.api.database.models import Base


class Database:
    """Async database wrapper providing sessions and lifecycle management."""

    def __init__(self, url: str = "postgresql+asyncpg://tuneforge:tuneforge_dev@localhost:5432/tuneforge") -> None:
        self._url = url
        engine_kwargs: dict = {"echo": False, "pool_size": 10, "max_overflow": 20}
        self._engine = create_async_engine(url, **engine_kwargs)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init_db(self, create_tables: bool = True) -> None:
        """Initialise the database.

        In production with Alembic, pass ``create_tables=False``.
        """
        if create_tables:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized (postgresql)")

    async def close(self) -> None:
        """Dispose of the engine connection pool."""
        await self._engine.dispose()

    def session(self) -> AsyncSession:
        """Create a new async session."""
        return self._session_factory()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        """Context manager providing a transactional session.

        Commits on success, rolls back on exception.
        """
        async with self.session() as session:
            async with session.begin():
                yield session


def total_pages(total: int, page_size: int) -> int:
    """Compute total page count."""
    return max(1, math.ceil(total / page_size))
