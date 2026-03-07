#!/usr/bin/env python3
"""Import local annotations.jsonl into the VPS database via API.

For each annotation:
1. Submit audio as a validation round (creates round + audio blobs + annotation task)
2. Directly insert the vote into the database via a second API call

Since the vote endpoint requires user auth and listen duration,
we submit rounds via service token, then bulk-insert votes via direct SQL.
"""

import base64
import json
import sys
import time
from pathlib import Path

import httpx

API_URL = sys.argv[1] if len(sys.argv) > 1 else "https://api.tuneforge.io"
API_TOKEN = sys.argv[2] if len(sys.argv) > 2 else ""
ANNOTATIONS_FILE = sys.argv[3] if len(sys.argv) > 3 else "annotations.jsonl"
STORAGE_DIR = sys.argv[4] if len(sys.argv) > 4 else "storage_restored"

if not API_TOKEN:
    print("Usage: python import_annotations.py <api_url> <api_token> [annotations.jsonl] [storage_dir]")
    sys.exit(1)

headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
client = httpx.Client(base_url=API_URL, headers=headers, timeout=120.0)

annotations = []
with open(ANNOTATIONS_FILE) as f:
    for line in f:
        if line.strip():
            annotations.append(json.loads(line.strip()))

print(f"Importing {len(annotations)} annotations from {ANNOTATIONS_FILE}")
print(f"Storage: {STORAGE_DIR}")
print(f"API: {API_URL}")
print()

# Phase 1: Submit all rounds and collect task info
rounds_created = []
errors = 0

for i, ann in enumerate(annotations):
    cid = ann["challenge_id"]
    prompt = ann["prompt"]
    preferred = ann["preferred"]

    audio_a_path = Path(STORAGE_DIR) / cid / "63.wav"
    audio_b_path = Path(STORAGE_DIR) / cid / "64.wav"

    if not audio_a_path.exists() or not audio_b_path.exists():
        print(f"  [{i+1}/{len(annotations)}] SKIP {cid}: audio missing")
        errors += 1
        continue

    # Read metadata
    meta_path = Path(STORAGE_DIR) / cid / "metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass

    audio_a_b64 = base64.b64encode(audio_a_path.read_bytes()).decode("ascii")
    audio_b_b64 = base64.b64encode(audio_b_path.read_bytes()).decode("ascii")

    payload = {
        "challenge_id": cid,
        "prompt": prompt,
        "genre": meta.get("genre", ""),
        "mood": meta.get("mood", ""),
        "tempo_bpm": meta.get("tempo_bpm", 120),
        "duration_seconds": meta.get("duration_seconds", 30),
        "validator_hotkey": "local-import",
        "responses": [
            {
                "miner_uid": 63,
                "miner_hotkey": meta.get("hotkeys", {}).get("63", "miner-63"),
                "audio_b64": audio_a_b64,
                "generation_time_ms": 0,
            },
            {
                "miner_uid": 64,
                "miner_hotkey": meta.get("hotkeys", {}).get("64", "miner-64"),
                "audio_b64": audio_b_b64,
                "generation_time_ms": 0,
            },
        ],
    }

    try:
        resp = client.post("/api/v1/validator/rounds", json=payload)
        if resp.status_code != 200:
            print(f"  [{i+1}/{len(annotations)}] ERROR round: {resp.status_code} {resp.text[:100]}")
            errors += 1
            continue

        data = resp.json()
        round_id = data["round_id"]
        audio_entries = data.get("audio", [])

        if len(audio_entries) < 2:
            print(f"  [{i+1}/{len(annotations)}] ERROR: only {len(audio_entries)} audio entries")
            errors += 1
            continue

        rounds_created.append({
            "round_id": round_id,
            "challenge_id": cid,
            "preferred": preferred,
            "audio_a_blob_id": audio_entries[0]["audio_blob_id"],
            "audio_b_blob_id": audio_entries[1]["audio_blob_id"],
        })
        print(f"  [{i+1}/{len(annotations)}] Round {round_id} created ({cid[:12]}...) preferred={preferred}")

    except Exception as exc:
        print(f"  [{i+1}/{len(annotations)}] EXCEPTION: {exc}")
        errors += 1

    time.sleep(0.1)

print(f"\nPhase 1 complete: {len(rounds_created)} rounds created, {errors} errors")
print(f"\nSaving round mapping to import_rounds.json for Phase 2...")

# Save for phase 2
with open("import_rounds.json", "w") as f:
    json.dump(rounds_created, f, indent=2)

print(f"Saved {len(rounds_created)} rounds to import_rounds.json")
print(f"\nRun Phase 2 on the VPS to insert votes directly into the database.")
