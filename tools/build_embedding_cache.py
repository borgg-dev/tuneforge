#!/usr/bin/env python3
"""
Pre-extract CLAP audio embeddings from WAV files and cache them in a NumPy NPZ.

Usage
-----
    python tools/build_embedding_cache.py annotations.jsonl -o embeddings.npz
    python tools/build_embedding_cache.py annotations.jsonl --skip-existing

CLAP embedding extraction is slow (requires loading a large model and running
inference on every audio file).  This tool pre-computes all embeddings once so
that downstream training scripts can load them instantly from the cache.

The tool reads an annotations JSONL file, collects every unique WAV path from
the ``audio_a`` and ``audio_b`` fields, extracts a 512-dim CLAP embedding for
each file via ``CLAPScorer``, and stores them in a compressed NPZ archive.

NPZ key format
~~~~~~~~~~~~~~
The leading directory component and ``.wav`` extension are stripped::

    storage/3/a1b2c3d4e5f67890.wav  ->  3/a1b2c3d4e5f67890
    data/audio/test.wav              ->  audio/test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Path / key helpers
# ---------------------------------------------------------------------------

def path_to_key(wav_path: str) -> str:
    """Convert WAV file path to NPZ cache key.

    Strips the leading directory component and .wav extension.
    'storage/<challenge_id>/<uid>.wav' -> '<challenge_id>/<uid>'
    """
    p = Path(wav_path)
    parts = p.parts
    if len(parts) >= 2:
        return str(Path(parts[-2]) / p.stem)
    return p.stem


# ---------------------------------------------------------------------------
# Annotations reader
# ---------------------------------------------------------------------------

def collect_wav_paths(annotations_path: str) -> list[str]:
    """Extract unique WAV file paths from annotations JSONL.

    Reads ``audio_a`` and ``audio_b`` fields from each record, deduplicates,
    and returns a sorted list.  Skips records where ``preferred == 'skip'``.

    Note: even though 'skip' records are skipped, a WAV file referenced in a
    skipped record may also appear in a non-skip record.  To ensure nothing is
    missed we collect from *all* records (including skipped ones), because the
    embeddings are useful regardless of the annotation label.
    """
    seen: set[str] = set()
    with open(annotations_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping malformed JSON on line {lineno}: {exc}")
                continue

            for field in ("audio_a", "audio_b"):
                path = record.get(field)
                if path and isinstance(path, str):
                    seen.add(path)

    return sorted(seen)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def build_cache(
    wav_paths: list[str],
    model_name: str = "laion/larger_clap_music",
    existing: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Extract CLAP embeddings for all WAV files.

    Returns a dict mapping cache key to a 512-dim float32 numpy array.
    Prints progress for every file.  Skips files that fail with a warning.

    Parameters
    ----------
    wav_paths:
        List of WAV file paths to process.
    model_name:
        HuggingFace CLAP model identifier.
    existing:
        Optional dict of already-cached embeddings (key -> ndarray).
        Files whose key is already present here are skipped.
    """
    import soundfile as sf
    from tuneforge.scoring.clap_scorer import CLAPScorer

    if existing is None:
        existing = {}

    cache: dict[str, np.ndarray] = dict(existing)
    total = len(wav_paths)
    skipped_existing = 0
    failed = 0

    # Lazy-load the CLAP model only when we know there is work to do
    clap: CLAPScorer | None = None

    for idx, wav_path in enumerate(wav_paths, 1):
        key = path_to_key(wav_path)

        # Skip if already in cache
        if key in cache:
            skipped_existing += 1
            print(f"[{idx}/{total}] {key} -- already cached, skipping")
            continue

        # Check file existence
        if not os.path.isfile(wav_path):
            print(f"[{idx}/{total}] WARNING: file not found: {wav_path}")
            failed += 1
            continue

        # Initialise scorer on first real file
        if clap is None:
            print(f"Loading CLAP model: {model_name} ...")
            clap = CLAPScorer(model_name=model_name)

        try:
            audio, sr = sf.read(wav_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            embedding = clap.get_audio_embedding(audio, sr)

            if embedding is None:
                print(f"[{idx}/{total}] WARNING: embedding extraction returned None for {key}")
                failed += 1
                continue

            cache[key] = embedding.astype(np.float32)
            print(f"[{idx}/{total}] {key} -> {embedding.shape[0]}-dim embedding")

        except Exception as exc:
            print(f"[{idx}/{total}] WARNING: failed to process {wav_path}: {exc}")
            failed += 1

    new_count = len(cache) - len(existing)
    print()
    print(
        f"Cached {len(cache)} embeddings total "
        f"({new_count} new, {skipped_existing} already cached, {failed} failed)"
    )
    return cache


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Pre-extract CLAP audio embeddings from WAV files referenced in "
            "an annotations JSONL file and cache them in a compressed NPZ archive."
        ),
    )
    parser.add_argument(
        "source",
        help="Path to annotations JSONL file (extracts unique WAV paths from it).",
    )
    parser.add_argument(
        "--output", "-o",
        default="embeddings.npz",
        help="Output NPZ path (default: embeddings.npz).",
    )
    parser.add_argument(
        "--model",
        default="laion/larger_clap_music",
        help="CLAP model name (default: laion/larger_clap_music).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help=(
            "Load an existing NPZ file (if present at --output path) and skip "
            "files whose embeddings are already cached."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    annotations_path = os.path.abspath(args.source)
    if not os.path.isfile(annotations_path):
        print(f"Error: annotations file not found: {annotations_path}", file=sys.stderr)
        return 1

    output_path = os.path.abspath(args.output)

    # Collect WAV paths
    print(f"Reading annotations from {annotations_path} ...")
    wav_paths = collect_wav_paths(annotations_path)
    if not wav_paths:
        print("No WAV paths found in annotations file.", file=sys.stderr)
        return 1
    print(f"Found {len(wav_paths)} unique WAV file(s)\n")

    # Optionally load existing cache
    existing: dict[str, np.ndarray] | None = None
    if args.skip_existing and os.path.isfile(output_path):
        print(f"Loading existing cache from {output_path} ...")
        with np.load(output_path) as npz:
            existing = {key: npz[key] for key in npz.files}
        print(f"Loaded {len(existing)} existing embedding(s)\n")

    # Build / extend cache
    cache = build_cache(
        wav_paths=wav_paths,
        model_name=args.model,
        existing=existing,
    )

    if not cache:
        print("No embeddings were produced. Nothing to save.", file=sys.stderr)
        return 1

    # Save
    np.savez_compressed(output_path, **cache)
    print(f"Saved {len(cache)} embedding(s) -> {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
