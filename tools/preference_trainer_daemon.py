#!/usr/bin/env python3
"""
Continuous preference model training daemon.

Runs as a long-lived process (managed by PM2) that periodically:
1. Checks the platform API for new resolved annotations
2. When new data arrives, exports annotations + downloads audio
3. Builds CLAP embeddings and trains the PreferenceHead
4. Uploads the new checkpoint IF validation accuracy improves

Safety gates:
- Won't retrain unless annotation count has grown since last run
- Won't upload a model that regresses in validation accuracy
- Minimum 10 resolved pairs required to train at all
- Cooldown between training runs (default 30 min check, 2h min between trains)

State is persisted to a JSON file so restarts resume correctly.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------

API_URL = os.environ.get("TF_TRAINER_API_URL", "http://localhost:8000")
API_TOKEN = os.environ.get("TF_TRAINER_API_TOKEN", "") or os.environ.get("TF_VALIDATOR_SERVICE_TOKEN", "")
CHECK_INTERVAL = int(os.environ.get("TF_TRAINER_CHECK_INTERVAL", "1800"))  # 30 min
TRAIN_COOLDOWN = int(os.environ.get("TF_TRAINER_COOLDOWN", "7200"))  # 2 hours
MIN_NEW_PAIRS = int(os.environ.get("TF_TRAINER_MIN_NEW_PAIRS", "5"))  # retrain after 5+ new pairs
MIN_TOTAL_PAIRS = int(os.environ.get("TF_TRAINER_MIN_TOTAL_PAIRS", "10"))
CACHE_DIR = os.environ.get("TF_TRAINER_CACHE_DIR", "/tmp/tuneforge_preference_training")
STATE_FILE = os.environ.get("TF_TRAINER_STATE_FILE", "preference_trainer_state.json")
CHECKPOINT_PATH = os.environ.get("TF_TRAINER_CHECKPOINT", "preference_head_latest.pt")

# Training hyperparameters
LR = float(os.environ.get("TF_TRAINER_LR", "1e-3"))
EPOCHS = int(os.environ.get("TF_TRAINER_EPOCHS", "50"))
BATCH_SIZE = int(os.environ.get("TF_TRAINER_BATCH_SIZE", "32"))
PATIENCE = int(os.environ.get("TF_TRAINER_PATIENCE", "5"))
VAL_SPLIT = float(os.environ.get("TF_TRAINER_VAL_SPLIT", "0.2"))
SEED = int(os.environ.get("TF_TRAINER_SEED", "42"))

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    print(f"\n[daemon] Received signal {signum}, shutting down gracefully...")
    _shutdown = True


def load_state(path: str) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {
        "last_train_time": 0,
        "last_annotation_count": 0,
        "last_val_accuracy": 0.0,
        "last_model_version": None,
        "total_trains": 0,
    }


def save_state(path: str, state: dict) -> None:
    Path(path).write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def check_and_train(state: dict) -> dict:
    """Check for new annotations and train if warranted. Returns updated state."""
    import httpx

    client = httpx.Client(
        base_url=API_URL.rstrip("/"),
        headers={"Authorization": f"Bearer {API_TOKEN}"},
        timeout=httpx.Timeout(300.0, connect=30.0),
    )

    # 1. Export annotations and check count
    try:
        resp = client.get("/api/v1/annotations/export")
        resp.raise_for_status()
        annotations = resp.json()
    except Exception as exc:
        print(f"[daemon] Failed to fetch annotations: {exc}")
        return state

    current_count = len(annotations)
    prev_count = state.get("last_annotation_count", 0)
    new_pairs = current_count - prev_count

    print(f"[daemon] Annotations: {current_count} total ({new_pairs} new since last train)")

    if current_count < MIN_TOTAL_PAIRS:
        print(f"[daemon] Need at least {MIN_TOTAL_PAIRS} pairs, only have {current_count}. Skipping.")
        return state

    if new_pairs < MIN_NEW_PAIRS:
        print(f"[daemon] Need at least {MIN_NEW_PAIRS} new pairs to retrain, only {new_pairs}. Skipping.")
        return state

    # Cooldown check
    now = time.time()
    since_last = now - state.get("last_train_time", 0)
    if since_last < TRAIN_COOLDOWN:
        remaining = TRAIN_COOLDOWN - since_last
        print(f"[daemon] Cooldown active, {remaining:.0f}s remaining. Skipping.")
        return state

    print(f"[daemon] Training triggered: {current_count} pairs ({new_pairs} new)")

    # 2. Download audio
    cache_dir = Path(CACHE_DIR)
    audio_dir = cache_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    blob_ids: set[str] = set()
    for a in annotations:
        blob_ids.add(a["audio_a"].split("/")[-1].replace(".wav", ""))
        blob_ids.add(a["audio_b"].split("/")[-1].replace(".wav", ""))

    print(f"[daemon] Downloading {len(blob_ids)} audio blobs...")
    from tools.export_and_train import download_audio
    try:
        audio_paths = download_audio(client, blob_ids, audio_dir)
    except Exception as exc:
        print(f"[daemon] Audio download failed: {exc}")
        return state

    # 3. Build CLAP embeddings
    print("[daemon] Building CLAP embeddings...")
    from tools.export_and_train import build_embeddings
    emb_cache = cache_dir / "embeddings.npz"
    try:
        embeddings = build_embeddings(audio_paths, emb_cache)
    except Exception as exc:
        print(f"[daemon] Embedding extraction failed: {exc}")
        return state

    # 4. Build training pairs
    from tools.export_and_train import build_pairs
    pairs = build_pairs(annotations, embeddings)
    print(f"[daemon] Built {len(pairs)} training pairs")

    if len(pairs) < MIN_TOTAL_PAIRS:
        print(f"[daemon] Not enough valid pairs ({len(pairs)} < {MIN_TOTAL_PAIRS}). Skipping.")
        return state

    # 5. Train
    print("[daemon] Training preference model...")
    from tools.train_preference import train

    try:
        result = train(
            pairs=pairs,
            lr=LR,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            patience=PATIENCE,
            val_split=VAL_SPLIT,
            seed=SEED,
            output_path=CHECKPOINT_PATH,
        )
    except Exception as exc:
        print(f"[daemon] Training failed: {exc}")
        return state

    new_acc = result["best_val_acc"]
    prev_acc = state.get("last_val_accuracy", 0.0)

    print(f"[daemon] Training complete: val_acc={new_acc:.3f} (previous={prev_acc:.3f})")

    # 6. Upload if accuracy improved (or first model)
    if new_acc < prev_acc - 0.02:
        print(
            f"[daemon] Accuracy regressed ({new_acc:.3f} < {prev_acc:.3f} - 0.02 margin). "
            f"NOT uploading. Keeping previous model."
        )
        # Still update state so we don't retrain on the same data
        state["last_train_time"] = time.time()
        state["last_annotation_count"] = current_count
        save_state(STATE_FILE, state)
        return state

    print("[daemon] Uploading model to API...")
    from tools.export_and_train import upload_model
    checkpoint = Path(CHECKPOINT_PATH)
    if not checkpoint.exists():
        print("[daemon] Checkpoint file not found after training!")
        return state

    try:
        info = upload_model(
            client,
            checkpoint,
            val_accuracy=new_acc,
            n_train_pairs=result.get("n_train"),
        )
        version = info.get("version", "?")
        sha = info.get("sha256", "")[:12]
        print(f"[daemon] Uploaded model v{version} (sha256={sha}..., val_acc={new_acc:.3f})")
    except Exception as exc:
        print(f"[daemon] Upload failed: {exc}")
        # Training succeeded, just upload failed — update state so we retry upload next cycle
        state["last_train_time"] = time.time()
        save_state(STATE_FILE, state)
        return state

    # Update state
    state["last_train_time"] = time.time()
    state["last_annotation_count"] = current_count
    state["last_val_accuracy"] = new_acc
    state["last_model_version"] = version
    state["total_trains"] = state.get("total_trains", 0) + 1
    save_state(STATE_FILE, state)

    print(f"[daemon] State saved. Total training runs: {state['total_trains']}")
    return state


def main() -> int:
    global _shutdown

    if not API_TOKEN:
        print("ERROR: TF_TRAINER_API_TOKEN is required", file=sys.stderr)
        return 1

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    state = load_state(STATE_FILE)

    print(f"[daemon] Preference trainer daemon started")
    print(f"[daemon] API: {API_URL}")
    print(f"[daemon] Check interval: {CHECK_INTERVAL}s, Train cooldown: {TRAIN_COOLDOWN}s")
    print(f"[daemon] Min new pairs: {MIN_NEW_PAIRS}, Min total: {MIN_TOTAL_PAIRS}")
    print(f"[daemon] Cache: {CACHE_DIR}")
    print(f"[daemon] State: last_count={state['last_annotation_count']}, "
          f"last_acc={state['last_val_accuracy']:.3f}, "
          f"total_trains={state['total_trains']}")

    while not _shutdown:
        try:
            state = check_and_train(state)
        except Exception as exc:
            print(f"[daemon] Unexpected error in training cycle: {exc}")
            import traceback
            traceback.print_exc()

        # Sleep in small increments so we can respond to signals
        for _ in range(CHECK_INTERVAL):
            if _shutdown:
                break
            time.sleep(1)

    print("[daemon] Shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
