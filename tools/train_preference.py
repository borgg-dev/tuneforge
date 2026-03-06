#!/usr/bin/env python3
"""
Train the PreferenceHead MLP using Bradley-Terry pairwise preference loss.

Usage
-----
    python tools/train_preference.py annotations.jsonl embeddings.npz
    python tools/train_preference.py annotations.jsonl embeddings.npz -o preference_head.pt \
        --lr 1e-3 --epochs 50 --batch-size 32 --patience 5 --val-split 0.2 --seed 42

The script reads paired preference annotations (JSONL) and pre-computed CLAP
embeddings (NPZ), then trains a ``PreferenceHead`` MLP to predict which of two
audio samples a human rater preferred.

The loss function is the Bradley-Terry pairwise loss, implemented as
``BCEWithLogitsLoss(logit_preferred - logit_rejected, target=ones)``.

After training, the full ``PreferenceHead`` state dict is saved so the
checkpoint can be loaded directly via ``PreferenceHead.load_state_dict()``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def path_to_key(wav_path: str) -> str:
    """Convert WAV path to NPZ cache key. Same logic as build_embedding_cache.py.

    'storage/3/abc123.wav' -> '3/abc123'
    """
    p = Path(wav_path)
    # Take the last two path components and strip the extension
    return str(Path(p.parts[-2]) / p.stem)


def load_pairs(
    annotations_path: str,
    embeddings_path: str,
    dual: bool = False,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """
    Load paired training examples from annotations and embedding cache.

    Returns list of (embedding_preferred, embedding_rejected, target) tuples.
    Target is always 1.0 (preferred should score higher).

    When dual=True, expects both ``{key}_clap`` (512-dim) and ``{key}_mert``
    (768-dim) keys in the NPZ and concatenates them into 1280-dim vectors.

    Skips entries where preferred=="skip" or embeddings are missing.
    Raises ValueError if fewer than 10 valid pairs found.
    Prints warning if fewer than 50 valid pairs found.
    """
    embeddings = np.load(embeddings_path)
    available_keys = set(embeddings.files)

    pairs: list[tuple[np.ndarray, np.ndarray, float]] = []
    skipped = 0

    with open(annotations_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: Skipping invalid JSON on line {line_num}: {exc}")
                skipped += 1
                continue

            preferred = entry.get("preferred", "")
            if preferred == "skip":
                skipped += 1
                continue

            audio_a = entry.get("audio_a", "")
            audio_b = entry.get("audio_b", "")

            if not audio_a or not audio_b:
                skipped += 1
                continue

            if preferred == "a":
                preferred_path = audio_a
                rejected_path = audio_b
            elif preferred == "b":
                preferred_path = audio_b
                rejected_path = audio_a
            else:
                skipped += 1
                continue

            pref_key = path_to_key(preferred_path)
            rej_key = path_to_key(rejected_path)

            if dual:
                # Dual mode: need both _clap and _mert suffixed keys
                pref_clap = f"{pref_key}_clap"
                pref_mert = f"{pref_key}_mert"
                rej_clap = f"{rej_key}_clap"
                rej_mert = f"{rej_key}_mert"

                if not {pref_clap, pref_mert, rej_clap, rej_mert} <= available_keys:
                    skipped += 1
                    continue

                emb_pref = np.concatenate([embeddings[pref_clap], embeddings[pref_mert]])
                emb_rej = np.concatenate([embeddings[rej_clap], embeddings[rej_mert]])
            else:
                if pref_key not in available_keys:
                    skipped += 1
                    continue
                if rej_key not in available_keys:
                    skipped += 1
                    continue

                emb_pref = embeddings[pref_key]
                emb_rej = embeddings[rej_key]

            pairs.append((emb_pref, emb_rej, 1.0))

    if skipped > 0:
        print(f"Skipped {skipped} annotations (missing embeddings, invalid, or 'skip')")

    if len(pairs) < 10:
        raise ValueError(
            f"Need at least 10 valid preference pairs, got {len(pairs)}. "
            f"Check that your annotations reference files present in the embedding cache."
        )

    if len(pairs) < 50:
        print(
            f"WARNING: Only {len(pairs)} valid pairs found. "
            f"Recommend 200+ for reliable training."
        )

    return pairs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    pairs: list[tuple[np.ndarray, np.ndarray, float]],
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 5,
    val_split: float = 0.2,
    seed: int = 42,
    output_path: str = "preference_head.pt",
    device: str | None = None,
    dual: bool = False,
) -> dict:
    """
    Train PreferenceHead using Bradley-Terry pairwise loss.

    Returns dict with:
      - best_val_acc: float
      - best_val_loss: float
      - final_epoch: int
      - n_train: int
      - n_val: int
    """
    if len(pairs) < 10:
        raise ValueError(f"Need at least 10 preference pairs, got {len(pairs)}")
    if len(pairs) < 50:
        print(f"WARNING: Only {len(pairs)} pairs. Recommend 200+ for reliable training.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train/val split
    indices = np.random.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_split))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_pairs = [pairs[i] for i in train_indices]
    val_pairs = [pairs[i] for i in val_indices]

    print(f"Training: {len(train_pairs)} pairs, Validation: {len(val_pairs)} pairs")

    # Build model
    if dual:
        from tuneforge.scoring.preference_model import DualPreferenceHead

        model = DualPreferenceHead()
        embedding_dim = 1280
    else:
        from tuneforge.scoring.preference_model import PreferenceHead

        model = PreferenceHead()
        embedding_dim = 512
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)

    # Access raw logit layers (everything except final Sigmoid)
    logit_layers = model.layers[:-1]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        np.random.shuffle(train_pairs)

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for start in range(0, len(train_pairs), batch_size):
            batch = train_pairs[start : start + batch_size]

            emb_pref = torch.tensor(
                np.array([p[0] for p in batch]), dtype=torch.float32, device=device
            )
            emb_rej = torch.tensor(
                np.array([p[1] for p in batch]), dtype=torch.float32, device=device
            )
            targets = torch.ones(len(batch), 1, dtype=torch.float32, device=device)

            logit_pref = logit_layers(emb_pref)
            logit_rej = logit_layers(emb_rej)
            diff = logit_pref - logit_rej

            loss = criterion(diff, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(batch)
            train_correct += (diff > 0).sum().item()
            train_total += len(batch)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for start in range(0, len(val_pairs), batch_size):
                batch = val_pairs[start : start + batch_size]

                emb_pref = torch.tensor(
                    np.array([p[0] for p in batch]),
                    dtype=torch.float32,
                    device=device,
                )
                emb_rej = torch.tensor(
                    np.array([p[1] for p in batch]),
                    dtype=torch.float32,
                    device=device,
                )
                targets = torch.ones(
                    len(batch), 1, dtype=torch.float32, device=device
                )

                logit_pref = logit_layers(emb_pref)
                logit_rej = logit_layers(emb_rej)
                diff = logit_pref - logit_rej

                loss = criterion(diff, targets)
                val_loss_sum += loss.item() * len(batch)
                val_correct += (diff > 0).sum().item()
                val_total += len(batch)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"Epoch {epoch + 1:3d}/{epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        # Early stopping on val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"  * New best val_acc={val_acc:.3f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"  Early stopping at epoch {epoch + 1} (patience={patience})"
                )
                break

    # Save best checkpoint with metadata
    if best_state is not None:
        checkpoint = {
            "state_dict": best_state,
            "val_accuracy": best_val_acc,
            "embedding_dim": embedding_dim,
        }
        torch.save(checkpoint, output_path)
        print(f"\nSaved checkpoint -> {output_path} (val_acc={best_val_acc:.3f}, dim={embedding_dim})")
    else:
        print("\nWARNING: No checkpoint saved (no improvement during training)")

    return {
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_epoch": epoch + 1,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the PreferenceHead MLP using Bradley-Terry pairwise "
            "preference loss on annotated CLAP embeddings."
        ),
    )
    parser.add_argument(
        "annotations",
        help="Path to JSONL annotations file with pairwise preferences.",
    )
    parser.add_argument(
        "embeddings",
        help="Path to NPZ embedding cache (from build_embedding_cache.py).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="preference_head.pt",
        help="Output checkpoint path (default: preference_head.pt).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Adam weight decay (default: 1e-4).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience in epochs (default: 5).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        default=False,
        help="Use dual-embedding mode (CLAP 512 + MERT 768 = 1280-dim input).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    print(f"Loading annotations from {args.annotations}")
    print(f"Loading embeddings from {args.embeddings}")

    try:
        pairs = load_pairs(args.annotations, args.embeddings, dual=args.dual)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(pairs)} valid preference pairs")

    try:
        result = train(
            pairs=pairs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            val_split=args.val_split,
            seed=args.seed,
            output_path=args.output,
            dual=args.dual,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nTraining complete:")
    print(f"  Best validation accuracy: {result['best_val_acc']:.3f}")
    print(f"  Training pairs: {result['n_train']}")
    print(f"  Validation pairs: {result['n_val']}")
    print(f"  Final epoch: {result['final_epoch']}")
    print(f"\nTo deploy, set: TF_PREFERENCE_MODEL_PATH={args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
