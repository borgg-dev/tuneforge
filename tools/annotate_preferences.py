#!/usr/bin/env python3
"""
A/B preference annotation tool for TuneForge validator audio outputs.

Usage
-----
    python tools/annotate_preferences.py storage/
    python tools/annotate_preferences.py storage/ -o annotations.jsonl --max-pairs 200
    python tools/annotate_preferences.py storage/ --no-same-challenge --player "mpv --no-video"
    python tools/annotate_preferences.py --source db --db-url postgresql+asyncpg://...

The tool scans a storage directory tree (or PostgreSQL database with
``--source db``) for WAV files organized as pairwise A/B comparisons within
each challenge group (or across all files if ``--no-same-challenge`` is
passed), and presents them to the annotator for preference labeling.

Results are appended to a JSONL file one record at a time for crash safety.
Existing annotations are loaded on startup so the session can be resumed
at any time.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Pair identification
# ---------------------------------------------------------------------------

def compute_pair_id(path_a: str, path_b: str) -> str:
    """
    Deterministic, order-invariant pair ID.

    SHA-256 of the two paths sorted alphabetically, truncated to the first
    12 hex characters.
    """
    canonical = "\n".join(sorted([path_a, path_b]))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Storage scanning
# ---------------------------------------------------------------------------

def _load_metadata(wav_path: str) -> dict:
    """Load challenge metadata for a WAV file.

    Looks for metadata.json in the same directory (new layout:
    storage/<challenge_id>/metadata.json).
    """
    parent = Path(wav_path).parent
    meta_path = parent / "metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def scan_storage(storage_dir: str, same_challenge: bool = True) -> list[dict]:
    """
    Scan storage directory for WAV files and generate annotation pairs.

    Each pair dict has keys: audio_a, audio_b, challenge_id, prompt (from
    sidecar JSON if available).

    If *same_challenge* is True, only pairs sharing the same challenge_id
    (filename stem) are generated.  If False, all pairwise combinations
    across the entire storage directory are produced.

    The returned list is shuffled deterministically (seeded by a hash of
    all discovered filenames) so that pair ordering is reproducible.
    """
    root = Path(storage_dir)
    if not root.is_dir():
        return []

    wav_files = sorted(root.rglob("*.wav"))
    if not wav_files:
        return []

    # Group files by challenge_id (parent directory name)
    groups: dict[str, list[Path]] = {}
    for wav in wav_files:
        challenge_id = wav.parent.name
        groups.setdefault(challenge_id, []).append(wav)

    pairs: list[dict] = []

    if same_challenge:
        for challenge_id, files in groups.items():
            if len(files) < 2:
                continue
            for a, b in itertools.combinations(sorted(files), 2):
                # Prefer prompt from A's sidecar, fall back to B's
                meta_a = _load_metadata(str(a))
                meta_b = _load_metadata(str(b))
                prompt = meta_a.get("prompt") or meta_b.get("prompt") or ""
                genre = meta_a.get("genre") or meta_b.get("genre") or ""
                pairs.append({
                    "audio_a": str(a),
                    "audio_b": str(b),
                    "challenge_id": challenge_id,
                    "prompt": prompt,
                    "genre": genre,
                })
    else:
        all_files = sorted(wav_files)
        if len(all_files) >= 2:
            for a, b in itertools.combinations(all_files, 2):
                meta_a = _load_metadata(str(a))
                meta_b = _load_metadata(str(b))
                prompt = meta_a.get("prompt") or meta_b.get("prompt") or ""
                genre = meta_a.get("genre") or meta_b.get("genre") or ""
                # Use challenge_id only when both files share the same parent
                cid_a = a.parent.name
                cid_b = b.parent.name
                pairs.append({
                    "audio_a": str(a),
                    "audio_b": str(b),
                    "challenge_id": cid_a if cid_a == cid_b else "",
                    "prompt": prompt,
                    "genre": genre,
                })

    # Deterministic shuffle seeded by hash of all filenames
    seed_material = "|".join(str(p) for p in sorted(wav_files))
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest(), 16) % (2**32)
    random.Random(seed).shuffle(pairs)

    return pairs


# ---------------------------------------------------------------------------
# Database scanning (--source db)
# ---------------------------------------------------------------------------

def scan_db(db_url: str, same_challenge: bool = True, tmp_dir: str | None = None) -> list[dict]:
    """
    Scan PostgreSQL for validation audio pairs and extract WAV files to a
    temp directory for playback during annotation.

    Returns pairs in the same format as scan_storage().
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    async def _scan() -> list[dict]:
        from tuneforge.api.database.engine import Database
        from tuneforge.api.database.models import AudioBlobRow, ValidationAudioRow, ValidationRoundRow
        from sqlalchemy import select

        db = Database(url=db_url)
        await db.init_db(create_tables=False)

        # Fetch all rounds with their audio
        async with db.session() as session:
            rounds = (await session.execute(
                select(ValidationRoundRow).order_by(ValidationRoundRow.created_at)
            )).scalars().all()

        out_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="tuneforge_annot_"))
        print(f"Extracting audio to: {out_dir}")

        pairs: list[dict] = []
        all_files: list[str] = []

        for vr in rounds:
            async with db.session() as session:
                audios = (await session.execute(
                    select(ValidationAudioRow).where(ValidationAudioRow.round_id == vr.id)
                )).scalars().all()

            if len(audios) < 2 and same_challenge:
                continue

            # Extract WAV files for this round
            round_files: list[tuple[str, ValidationAudioRow]] = []
            for va in audios:
                wav_path = out_dir / vr.challenge_id / f"{va.miner_uid}.wav"
                if not wav_path.exists():
                    async with db.session() as session:
                        blob = await session.get(AudioBlobRow, va.audio_blob_id)
                    if blob is None:
                        continue
                    wav_path.parent.mkdir(parents=True, exist_ok=True)
                    wav_path.write_bytes(blob.audio_data)
                round_files.append((str(wav_path), va))
                all_files.append(str(wav_path))

            if same_challenge and len(round_files) >= 2:
                for (path_a, va_a), (path_b, va_b) in itertools.combinations(sorted(round_files), 2):
                    pairs.append({
                        "audio_a": path_a,
                        "audio_b": path_b,
                        "challenge_id": vr.challenge_id,
                        "prompt": vr.prompt or "",
                        "genre": vr.genre or "",
                    })

        if not same_challenge and len(all_files) >= 2:
            # Cross-challenge pairing not supported for DB source yet
            pass

        await db.close()

        # Deterministic shuffle
        if all_files:
            seed_material = "|".join(sorted(all_files))
            seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest(), 16) % (2**32)
            random.Random(seed).shuffle(pairs)

        return pairs

    return asyncio.run(_scan())


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed(output_path: str) -> set[str]:
    """
    Load set of already-annotated pair_ids from an existing JSONL file.

    Returns an empty set if the file does not exist or is empty.
    Malformed lines are silently skipped.
    """
    completed: set[str] = set()
    path = Path(output_path)
    if not path.exists():
        return completed
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    pid = record.get("pair_id")
                    if pid:
                        completed.add(pid)
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return completed


# ---------------------------------------------------------------------------
# Audio playback
# ---------------------------------------------------------------------------

def _find_player(override: str | None = None) -> list[str]:
    """
    Find an available audio player command.

    If *override* is specified, it is split on whitespace and used directly.
    Otherwise the function probes for ffplay, aplay, and afplay in order.

    Raises ``RuntimeError`` if no player is found.
    """
    if override:
        return override.split()
    for cmd, args in [
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
        ("aplay", []),
        ("afplay", []),
    ]:
        if shutil.which(cmd):
            return [cmd] + args
    raise RuntimeError(
        "No audio player found. Install ffmpeg (provides ffplay) or "
        "alsa-utils (provides aplay)."
    )


def _play_audio(path: str, player_cmd: list[str]) -> None:
    """Play a WAV file via subprocess. Blocks until playback finishes."""
    if not Path(path).exists():
        print(f"  WARNING: file not found: {path}")
        return
    try:
        subprocess.run(
            player_cmd + [path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        print(f"  WARNING: player exited with code {exc.returncode}")
    except FileNotFoundError:
        raise RuntimeError(
            f"Audio player command not found: {player_cmd[0]!r}. "
            "Check your --player setting or install a supported player."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "A/B preference annotation of audio files from TuneForge "
            "validator storage (local filesystem or PostgreSQL)."
        ),
    )
    parser.add_argument(
        "storage_dir",
        nargs="?",
        default="./storage",
        help="Path to storage directory (default: ./storage). Ignored when --source db.",
    )
    parser.add_argument(
        "--source",
        choices=["local", "db"],
        default="local",
        help="Audio source: 'local' for filesystem, 'db' for PostgreSQL (default: local).",
    )
    parser.add_argument(
        "--db-url",
        default="postgresql+asyncpg://tuneforge:tuneforge_dev@localhost:5432/tuneforge",
        help="PostgreSQL database URL (used with --source db).",
    )
    parser.add_argument(
        "--output", "-o",
        default="annotations.jsonl",
        help="Output JSONL path (default: annotations.jsonl).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Maximum number of pairs to present (default: 0 = unlimited).",
    )
    parser.add_argument(
        "--same-challenge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Only pair audio from the same challenge_id (default: True). "
            "Use --no-same-challenge to pair across challenges."
        ),
    )
    parser.add_argument(
        "--player",
        type=str,
        default=None,
        help=(
            "Override audio player command (e.g. 'mpv --no-video'). "
            "By default the tool tries ffplay, aplay, then afplay."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    # Locate audio player early so we fail fast
    try:
        player = _find_player(args.player)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Audio player: {' '.join(player)}")

    # Scan pairs from either local filesystem or database
    if args.source == "db":
        print(f"Loading pairs from database: {args.db_url[:40]}...")
        pairs = scan_db(args.db_url, same_challenge=args.same_challenge)
    else:
        storage_dir = Path(args.storage_dir)
        if not storage_dir.is_dir():
            print(f"Error: {args.storage_dir} is not a directory.", file=sys.stderr)
            return 1
        pairs = scan_storage(str(storage_dir), same_challenge=args.same_challenge)

    if not pairs:
        print("No annotation pairs found. Need at least 2 audio files "
              "sharing a challenge_id.")
        return 0

    # Load previously completed annotations for resume support
    completed = load_completed(args.output)

    # Filter out already-annotated pairs
    pending = [
        p for p in pairs
        if compute_pair_id(p["audio_a"], p["audio_b"]) not in completed
    ]

    if args.max_pairs > 0:
        pending = pending[:args.max_pairs]

    print(f"Found {len(pairs)} total pairs, {len(completed)} already annotated, "
          f"{len(pending)} remaining")

    if not pending:
        print("All pairs have been annotated. Nothing to do.")
        return 0

    annotated = 0

    try:
        for i, pair in enumerate(pending):
            pair_id = compute_pair_id(pair["audio_a"], pair["audio_b"])

            print(f"\n{'=' * 60}")
            print(f"Pair {i + 1}/{len(pending)}")
            if pair.get("prompt"):
                print(f"Prompt: {pair['prompt'][:120]}")
            if pair.get("genre"):
                print(f"Genre: {pair['genre']}")
            if pair.get("challenge_id"):
                print(f"Challenge: {pair['challenge_id'][:20]}")
            print(f"  A: {pair['audio_a']}")
            print(f"  B: {pair['audio_b']}")

            while True:
                print("\nPlaying A...")
                _play_audio(pair["audio_a"], player)
                print("Playing B...")
                _play_audio(pair["audio_b"], player)

                choice = input(
                    "\nWhich sounds better? [1=A / 2=B / r=replay / s=skip / q=quit]: "
                ).strip().lower()

                if choice in ("1", "a"):
                    preferred = "a"
                    break
                elif choice in ("2", "b"):
                    preferred = "b"
                    break
                elif choice == "s":
                    preferred = "skip"
                    break
                elif choice == "r":
                    continue  # replay both clips
                elif choice == "q":
                    print(f"\nQuitting. {annotated} annotations saved to {args.output}")
                    return 0
                else:
                    print("Invalid choice. Enter 1, 2, r, s, or q.")
                    continue

            # Append record to JSONL immediately (crash-safe)
            record = {
                "pair_id": pair_id,
                "audio_a": pair["audio_a"],
                "audio_b": pair["audio_b"],
                "challenge_id": pair.get("challenge_id", ""),
                "prompt": pair.get("prompt", ""),
                "preferred": preferred,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(args.output, "a") as f:
                f.write(json.dumps(record) + "\n")

            annotated += 1
            if preferred != "skip":
                print(f"  -> Preferred: {'A' if preferred == 'a' else 'B'}")
            else:
                print("  -> Skipped")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted. {annotated} annotations saved to {args.output}")
        return 0

    print(f"\nDone! {annotated} annotations saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
