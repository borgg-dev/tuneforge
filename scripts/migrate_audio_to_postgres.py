#!/usr/bin/env python3
"""
One-time migration: move audio files from local storage into PostgreSQL.

Migrates:
  1. Validation rounds: metadata.json → validation_rounds table
     WAV files → audio_blobs + validation_audio tables
  2. SaaS tracks: MP3 files → audio_blobs, update tracks.audio_blob_id

Usage:
    python scripts/migrate_audio_to_postgres.py [--storage-path ./storage] [--db-url postgresql+asyncpg://...]
    python scripts/migrate_audio_to_postgres.py --dry-run   # preview without writing
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger


async def migrate(storage_path: str, db_url: str, dry_run: bool = False, batch_size: int = 20) -> None:
    from tuneforge.api.database.engine import Database
    from tuneforge.api.database.crud import (
        create_audio_blob,
        create_validation_audio,
        create_validation_round,
    )
    from sqlalchemy import select, update
    from tuneforge.api.database.models import TrackRow

    storage = Path(storage_path)
    if not storage.exists():
        logger.error(f"Storage path {storage} does not exist")
        return

    db = Database(url=db_url)
    await db.init_db(create_tables=False)

    # ---------------------------------------------------------------
    # 1. Migrate validation rounds (challenge dirs with metadata.json)
    # ---------------------------------------------------------------
    challenge_dirs = sorted(
        d for d in storage.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    )
    logger.info(f"Found {len(challenge_dirs)} validation round directories")

    rounds_created = 0
    blobs_created = 0
    va_created = 0
    errors = 0

    for i, cdir in enumerate(challenge_dirs, 1):
        try:
            meta = json.loads((cdir / "metadata.json").read_text())
        except Exception as exc:
            logger.warning(f"Skipping {cdir.name}: bad metadata ({exc})")
            errors += 1
            continue

        challenge_id = meta.get("challenge_id", cdir.name)

        if dry_run:
            wav_files = list(cdir.glob("*.wav"))
            logger.info(f"[DRY RUN] Round {i}/{len(challenge_dirs)}: {challenge_id} — {len(wav_files)} WAVs")
            continue

        # Create validation round
        try:
            vr = await create_validation_round(
                db,
                challenge_id=challenge_id,
                prompt=meta.get("prompt", ""),
                genre=meta.get("genre"),
                mood=meta.get("mood"),
                tempo_bpm=meta.get("tempo_bpm"),
                duration_seconds=meta.get("duration_seconds", 0),
            )
            rounds_created += 1
        except Exception as exc:
            logger.warning(f"Failed to create round for {challenge_id}: {exc}")
            errors += 1
            continue

        # Migrate each WAV file in the challenge dir
        for wav_path in sorted(cdir.glob("*.wav")):
            try:
                uid = int(wav_path.stem)
            except ValueError:
                logger.warning(f"Skipping non-UID wav: {wav_path.name}")
                continue

            try:
                audio_bytes = wav_path.read_bytes()
                blob = await create_audio_blob(db, audio_bytes, fmt="wav")
                blobs_created += 1

                await create_validation_audio(
                    db,
                    round_id=vr.id,
                    miner_uid=uid,
                    audio_blob_id=blob.id,
                    miner_hotkey=f"uid-{uid}",
                )
                va_created += 1
            except Exception as exc:
                logger.warning(f"Failed to migrate {wav_path}: {exc}")
                errors += 1

        if i % batch_size == 0:
            logger.info(f"  Progress: {i}/{len(challenge_dirs)} rounds, {blobs_created} blobs, {va_created} validation_audio")

    logger.info(
        f"Validation rounds done: {rounds_created} rounds, "
        f"{blobs_created} blobs, {va_created} validation_audio, {errors} errors"
    )

    # ---------------------------------------------------------------
    # 2. Migrate SaaS track MP3s (top-level .mp3 files in storage/)
    # ---------------------------------------------------------------
    mp3_files = sorted(storage.glob("*.mp3"))
    logger.info(f"Found {len(mp3_files)} SaaS track MP3 files")

    tracks_updated = 0
    for mp3_path in mp3_files:
        track_id = mp3_path.stem  # filename without extension = track ID

        if dry_run:
            logger.info(f"[DRY RUN] Track MP3: {track_id} ({mp3_path.stat().st_size} bytes)")
            continue

        try:
            audio_bytes = mp3_path.read_bytes()
            blob = await create_audio_blob(
                db, audio_bytes, fmt="mp3", content_type="audio/mpeg"
            )

            # Update the tracks row to point to the new blob
            async with db.session() as session:
                await session.execute(
                    update(TrackRow)
                    .where(TrackRow.id == track_id)
                    .values(
                        audio_blob_id=blob.id,
                        audio_path=f"/api/v1/audio/{blob.id}.mp3",
                    )
                )
                await session.commit()

            tracks_updated += 1
            logger.info(f"Migrated track {track_id} → blob {blob.id}")
        except Exception as exc:
            logger.warning(f"Failed to migrate track {track_id}: {exc}")
            errors += 1

    logger.info(f"Track migration done: {tracks_updated} tracks updated, {errors} total errors")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    total_blobs = blobs_created + tracks_updated
    logger.info(
        f"\n{'=' * 50}\n"
        f"Migration {'(DRY RUN) ' if dry_run else ''}complete:\n"
        f"  Validation rounds: {rounds_created}\n"
        f"  Validation audio:  {va_created}\n"
        f"  Audio blobs:       {total_blobs}\n"
        f"  SaaS tracks updated: {tracks_updated}\n"
        f"  Errors: {errors}\n"
        f"{'=' * 50}"
    )

    await db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate audio from filesystem to PostgreSQL")
    parser.add_argument(
        "--storage-path",
        default="./storage",
        help="Path to local storage directory (default: ./storage)",
    )
    parser.add_argument(
        "--db-url",
        default="postgresql+asyncpg://tuneforge:tuneforge_dev@localhost:5432/tuneforge",
        help="PostgreSQL database URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be migrated without writing to DB",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Log progress every N rounds (default: 20)",
    )
    args = parser.parse_args()

    asyncio.run(migrate(
        storage_path=args.storage_path,
        db_url=args.db_url,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    ))


if __name__ == "__main__":
    main()
