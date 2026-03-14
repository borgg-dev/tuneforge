#!/usr/bin/env python3
"""
Export crowd annotations from the platform API, build CLAP embeddings,
train the preference model, and upload the checkpoint.

Usage
-----
    python tools/export_and_train.py --api-url http://localhost:8000 --token <jwt_or_service_token>

    # With custom training params:
    python tools/export_and_train.py --api-url http://localhost:8000 --token <token> \
        --lr 1e-3 --epochs 50 --batch-size 32

Steps:
1. GET /api/v1/annotations/export → JSONL annotations
2. Download unique audio blobs to temp dir
3. Build CLAP embedding cache
4. Train PreferenceHead MLP (Bradley-Terry loss)
5. Upload checkpoint via POST /api/v1/annotations/model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import httpx
import numpy as np


def export_annotations(client: httpx.Client) -> list[dict]:
    """Fetch resolved annotations from the API."""
    resp = client.get("/api/v1/annotations/export")
    resp.raise_for_status()
    return resp.json()


def download_audio(
    client: httpx.Client,
    blob_ids: set[str],
    out_dir: Path,
) -> dict[str, Path]:
    """Download audio blobs to temp directory. Returns {blob_id: local_path}."""
    paths: dict[str, Path] = {}
    total = len(blob_ids)
    for i, blob_id in enumerate(sorted(blob_ids), 1):
        out_path = out_dir / f"{blob_id}.wav"
        if out_path.exists():
            paths[blob_id] = out_path
            continue
        print(f"  Downloading {i}/{total}: {blob_id[:12]}...")
        resp = client.get(f"/api/v1/audio/{blob_id}.wav")
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        paths[blob_id] = out_path
    return paths


class _EmbeddingTimeout(Exception):
    pass


def _load_mert_model():
    """Load MERT model for embedding extraction (CPU-friendly, 95M params)."""
    from transformers import AutoModel, AutoProcessor
    import torch

    model_name = "m-a-p/MERT-v1-95M"
    local_only = os.environ.get("HF_HUB_OFFLINE", "") == "1"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_only)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_only)
    model.eval()
    return processor, model


def _extract_mert_embedding(audio: np.ndarray, sr: int, processor, model) -> np.ndarray | None:
    """Extract pooled MERT embedding (768-dim) from audio."""
    import torch
    import librosa

    try:
        # MERT expects 24kHz
        if sr != 24000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=24000)

        inputs = processor(audio, sampling_rate=24000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Pool last hidden layer: mean over time → 768-dim
            last_hidden = outputs.hidden_states[-1]  # [1, T, 768]
            pooled = last_hidden.squeeze(0).mean(dim=0).numpy()  # [768]
        return pooled
    except Exception as exc:
        print(f"  WARNING: MERT extraction failed: {exc}")
        return None


def build_embeddings(
    audio_paths: dict[str, Path],
    cache_path: Path,
    dual: bool = False,
    per_file_timeout: int = 120,
) -> dict[str, np.ndarray]:
    """Extract CLAP (and optionally MERT) embeddings for all audio files.

    When dual=True, stores {blob_id}_clap (512-dim) and {blob_id}_mert (768-dim).
    When dual=False, stores {blob_id} (512-dim CLAP only) for backwards compatibility.

    Uses signal.alarm to timeout individual extractions that hang.
    """
    import signal

    from tuneforge.scoring.clap_scorer import CLAPScorer

    clap_scorer = CLAPScorer()
    mert_processor, mert_model = None, None
    if dual:
        print("  Loading MERT model for dual-mode embeddings...")
        mert_processor, mert_model = _load_mert_model()
        print("  MERT model loaded")

    # Load existing cache if present
    existing: dict[str, np.ndarray] = {}
    if cache_path.exists():
        data = np.load(str(cache_path))
        existing = {k: data[k] for k in data.files}
        print(f"  Loaded {len(existing)} cached embeddings")

    embeddings = dict(existing)
    new_count = 0

    # Count how many need extracting
    if dual:
        to_extract = [bid for bid in sorted(audio_paths)
                      if f"{bid}_clap" not in embeddings or f"{bid}_mert" not in embeddings]
    else:
        to_extract = [bid for bid in sorted(audio_paths) if bid not in embeddings]

    if not to_extract:
        return embeddings

    def _timeout_handler(signum, frame):
        raise _EmbeddingTimeout()

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    try:
        for i, blob_id in enumerate(to_extract, 1):
            path = audio_paths[blob_id]
            try:
                import soundfile as sf

                signal.alarm(per_file_timeout)
                audio, sr = sf.read(str(path))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                clap_emb = clap_scorer.get_audio_embedding(audio, sr)
                if clap_emb is None:
                    signal.alarm(0)
                    continue

                if dual:
                    mert_emb = _extract_mert_embedding(audio, sr, mert_processor, mert_model)
                    signal.alarm(0)
                    if mert_emb is None:
                        continue
                    embeddings[f"{blob_id}_clap"] = clap_emb
                    embeddings[f"{blob_id}_mert"] = mert_emb
                else:
                    signal.alarm(0)
                    embeddings[blob_id] = clap_emb

                new_count += 1
                if new_count % 5 == 0 or i == len(to_extract):
                    print(f"  Extracted {new_count}/{len(to_extract)} embeddings...")
            except _EmbeddingTimeout:
                print(f"  WARNING: Timeout extracting {blob_id} after {per_file_timeout}s — skipping")
            except Exception as exc:
                signal.alarm(0)
                print(f"  WARNING: Failed to extract embedding for {blob_id}: {exc}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if new_count > 0:
        np.savez_compressed(str(cache_path), **embeddings)
        print(f"  Extracted {new_count} new embeddings (total: {len(embeddings)})")

    return embeddings


def build_pairs(
    annotations: list[dict],
    embeddings: dict[str, np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Convert annotations + embeddings into training pairs.

    Auto-detects dual mode: if keys contain '_clap'/'_mert' suffixes,
    concatenates both into 1280-dim vectors. Otherwise uses raw 512-dim.
    """
    # Detect dual mode from key format
    dual = any(k.endswith("_clap") for k in embeddings)

    pairs = []
    skipped = 0

    for entry in annotations:
        audio_a_url = entry.get("audio_a", "")
        audio_b_url = entry.get("audio_b", "")
        preferred = entry.get("preferred", "")

        a_id = audio_a_url.split("/")[-1].replace(".wav", "")
        b_id = audio_b_url.split("/")[-1].replace(".wav", "")

        if dual:
            a_clap, a_mert = f"{a_id}_clap", f"{a_id}_mert"
            b_clap, b_mert = f"{b_id}_clap", f"{b_id}_mert"
            if not {a_clap, a_mert, b_clap, b_mert} <= set(embeddings):
                skipped += 1
                continue
            emb_a = np.concatenate([embeddings[a_clap], embeddings[a_mert]])
            emb_b = np.concatenate([embeddings[b_clap], embeddings[b_mert]])
        else:
            if a_id not in embeddings or b_id not in embeddings:
                skipped += 1
                continue
            emb_a = embeddings[a_id]
            emb_b = embeddings[b_id]

        if preferred == "a":
            pairs.append((emb_a, emb_b, 1.0))
        elif preferred == "b":
            pairs.append((emb_b, emb_a, 1.0))
        else:
            skipped += 1

    if skipped > 0:
        print(f"  Skipped {skipped} annotations (missing embeddings)")

    return pairs


def upload_model(
    client: httpx.Client,
    checkpoint_path: Path,
    val_accuracy: float | None,
    n_train_pairs: int | None,
) -> dict:
    """Upload trained model to the API."""
    data = checkpoint_path.read_bytes()
    files = {"file": ("preference_head.pt", data, "application/octet-stream")}
    params = {}
    if val_accuracy is not None:
        params["val_accuracy"] = str(val_accuracy)
    if n_train_pairs is not None:
        params["n_train_pairs"] = str(n_train_pairs)

    resp = client.post("/api/v1/annotations/model", files=files, params=params)
    resp.raise_for_status()
    return resp.json()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export annotations, train preference model, upload checkpoint."
    )
    parser.add_argument("--api-url", default="http://localhost:8000", help="Platform API URL")
    parser.add_argument("--token", required=True, help="JWT or service token for auth")
    parser.add_argument("--output", "-o", default="preference_head.pt", help="Local checkpoint path")
    parser.add_argument("--cache-dir", default=None, help="Directory for audio/embedding cache")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload", action="store_true", help="Upload model to API after training")
    args = parser.parse_args(argv)

    client = httpx.Client(
        base_url=args.api_url.rstrip("/"),
        headers={"Authorization": f"Bearer {args.token}"},
        timeout=httpx.Timeout(300.0, connect=30.0),
    )

    # 1. Export annotations
    print("Step 1: Exporting annotations...")
    annotations = export_annotations(client)
    print(f"  Got {len(annotations)} resolved annotation pairs")
    if len(annotations) < 10:
        print("ERROR: Need at least 10 resolved annotations to train.", file=sys.stderr)
        return 1

    # 2. Download audio
    print("\nStep 2: Downloading audio...")
    blob_ids: set[str] = set()
    for a in annotations:
        blob_ids.add(a["audio_a"].split("/")[-1].replace(".wav", ""))
        blob_ids.add(a["audio_b"].split("/")[-1].replace(".wav", ""))
    print(f"  {len(blob_ids)} unique audio blobs to fetch")

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp(prefix="tuneforge_train_"))
    audio_dir = cache_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = download_audio(client, blob_ids, audio_dir)
    print(f"  Downloaded {len(audio_paths)} audio files to {audio_dir}")

    # 3. Build embeddings
    print("\nStep 3: Building CLAP embeddings...")
    emb_cache_path = cache_dir / "embeddings.npz"
    embeddings = build_embeddings(audio_paths, emb_cache_path)

    # 4. Build pairs
    print("\nStep 4: Building training pairs...")
    pairs = build_pairs(annotations, embeddings)
    print(f"  {len(pairs)} valid training pairs")

    if len(pairs) < 10:
        print("ERROR: Not enough pairs for training.", file=sys.stderr)
        return 1

    # 5. Train
    print("\nStep 5: Training preference model...")
    from tools.train_preference import train

    result = train(
        pairs=pairs,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        val_split=args.val_split,
        seed=args.seed,
        output_path=args.output,
    )

    print(f"\nTraining complete:")
    print(f"  Best val accuracy: {result['best_val_acc']:.3f}")
    print(f"  Training pairs: {result['n_train']}")
    print(f"  Validation pairs: {result['n_val']}")

    # 6. Upload (optional)
    if args.upload:
        print("\nStep 6: Uploading model to API...")
        checkpoint = Path(args.output)
        if checkpoint.exists():
            info = upload_model(
                client, checkpoint,
                val_accuracy=result.get("best_val_acc"),
                n_train_pairs=result.get("n_train"),
            )
            print(f"  Uploaded as version {info.get('version')} (sha256={info.get('sha256', '')[:12]}...)")
        else:
            print("  WARNING: Checkpoint file not found, skipping upload")

    return 0


if __name__ == "__main__":
    sys.exit(main())
