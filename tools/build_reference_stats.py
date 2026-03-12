#!/usr/bin/env python3
"""
Build FAD reference statistics from a directory of high-quality audio files.

Usage:
    python -m tools.build_reference_stats /path/to/music_dir -o reference_stats.npz

Extracts CLAP embeddings for each audio file, computes mean and covariance,
and saves to a .npz file for use by FADScorer.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Build FAD reference statistics from audio files"
    )
    parser.add_argument("audio_dir", help="Directory containing WAV/MP3/FLAC files")
    parser.add_argument("-o", "--output", default="reference_stats.npz", help="Output .npz path")
    parser.add_argument("--model", default="laion/clap-htsat-unfused", help="CLAP model name")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files (0=all)")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        logger.error("Audio directory not found: {}", audio_dir)
        sys.exit(1)

    # Collect audio files
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = sorted(
        f for f in audio_dir.rglob("*") if f.suffix.lower() in extensions
    )
    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        logger.error("No audio files found in {}", audio_dir)
        sys.exit(1)

    logger.info("Found {} audio files", len(files))

    # Load CLAP model
    from tuneforge.scoring.clap_scorer import CLAPScorer

    clap = CLAPScorer(model_name=args.model)

    # Extract embeddings
    import soundfile as sf

    embeddings = []
    for i, fpath in enumerate(files):
        try:
            audio, sr = sf.read(str(fpath))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            emb = clap.get_audio_embedding(audio, sr)
            if emb is not None:
                norm = np.linalg.norm(emb)
                if norm > 1e-8:
                    embeddings.append(emb / norm)

            if (i + 1) % 10 == 0:
                logger.info("Processed {}/{} files", i + 1, len(files))
        except Exception as exc:
            logger.warning("Failed to process {}: {}", fpath, exc)

    if len(embeddings) < 10:
        logger.error("Too few valid embeddings ({}). Need at least 10.", len(embeddings))
        sys.exit(1)

    emb_array = np.array(embeddings, dtype=np.float64)
    mean = np.mean(emb_array, axis=0)
    cov = np.cov(emb_array, rowvar=False)

    # Regularize covariance
    cov += np.eye(cov.shape[0]) * 1e-6

    np.savez(args.output, mean=mean, cov=cov, n_files=len(embeddings))
    logger.info(
        "Saved reference stats to {} (dim={}, n_files={})",
        args.output,
        len(mean),
        len(embeddings),
    )


if __name__ == "__main__":
    main()
