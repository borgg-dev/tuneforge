"""
Preference model for TuneForge.

A lightweight MLP on top of CLAP audio embeddings that predicts
human preference.  In bootstrap mode (no trained checkpoint), returns
0.5 and the reward model zeroes its weight so it has no influence.

Supports two architectures:
- Single-embedding (512-dim CLAP only) via ``PreferenceHead``
- Dual-embedding (512-dim CLAP + 768-dim MERT = 1280-dim) via ``DualPreferenceHead``

The architecture is auto-detected from the checkpoint.

Training supports Bradley-Terry pairwise loss for learning from A/B comparisons,
which better matches the annotation format and avoids difficulty of absolute scoring.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from pathlib import Path

from tuneforge.config.scoring_config import CLAP_MODEL
from tuneforge.scoring.clap_scorer import CLAPScorer


class PreferenceHead(nn.Module):
    """MLP: 512-dim CLAP embedding → scalar preference score.

    Note: No final Sigmoid — the raw logit is used for Bradley-Terry
    training (logsigmoid on the difference).  Sigmoid is applied at
    inference time in PreferenceModel.score().
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DualPreferenceHead(nn.Module):
    """MLP for dual-embedding preference: CLAP (512) + MERT (768) = 1280-dim input.

    Note: No final Sigmoid — raw logit for Bradley-Terry training.
    Sigmoid applied at inference in PreferenceModel.score().
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PreferenceWeightScaler:
    """Auto-scale preference weight based on model validation accuracy."""

    def __init__(
        self,
        min_weight: float = 0.02,
        max_weight: float = 0.10,
        min_accuracy: float = 0.55,
        max_accuracy: float = 0.80,
    ) -> None:
        self._min_weight = min_weight
        self._max_weight = max_weight
        self._min_accuracy = min_accuracy
        self._max_accuracy = max_accuracy
        self._current_accuracy: float | None = None

    def update_accuracy(self, val_accuracy: float | None) -> None:
        """Update current model accuracy from checkpoint metadata."""
        self._current_accuracy = val_accuracy

    def get_scaled_weight(self) -> float:
        """Return preference weight scaled by model quality.

        Linear interpolation: accuracy=min->min_weight, accuracy=max->max_weight.
        Returns min_weight when no accuracy data available.
        """
        if self._current_accuracy is None:
            return self._min_weight
        clamped = max(self._min_accuracy, min(self._max_accuracy, self._current_accuracy))
        t = (clamped - self._min_accuracy) / (self._max_accuracy - self._min_accuracy + 1e-8)
        return self._min_weight + t * (self._max_weight - self._min_weight)


class PreferenceModel:
    """
    Predict human preference from audio.

    Uses CLAP audio embeddings fed through a trained MLP head.
    If no trained model path is given, runs in bootstrap mode using
    a multi-feature perceptual quality heuristic.

    Supports dual mode (CLAP + MERT embeddings) when checkpoint was trained
    with 1280-dim input.
    """

    def __init__(
        self,
        model_path: str | None = None,
        clap_scorer=None,
        neural_scorer=None,
    ) -> None:
        self._clap = clap_scorer or CLAPScorer(model_name=CLAP_MODEL)
        self._neural = neural_scorer  # Shared NeuralQualityScorer for MERT features
        self._head: PreferenceHead | DualPreferenceHead | None = None
        self._bootstrap = True
        self._dual_mode = False
        self._model_path = model_path
        self._scaler = PreferenceWeightScaler()

        if model_path is not None and Path(model_path).exists():
            try:
                raw = torch.load(model_path, map_location="cpu", weights_only=True)

                # Support new checkpoint format with metadata
                val_accuracy: float | None = None
                if isinstance(raw, dict) and "state_dict" in raw:
                    state = raw["state_dict"]
                    val_accuracy = raw.get("val_accuracy")
                    embedding_dim = raw.get("embedding_dim")
                else:
                    # Legacy format: raw state dict
                    state = raw
                    embedding_dim = None

                # Auto-detect architecture from first linear layer weight shape
                first_weight_key = None
                for key in state:
                    if key.endswith(".weight") and state[key].ndim == 2:
                        first_weight_key = key
                        break

                if first_weight_key is not None:
                    input_dim = state[first_weight_key].shape[1]
                elif embedding_dim is not None:
                    input_dim = embedding_dim
                else:
                    input_dim = 512  # default

                if input_dim == 1280:
                    self._head = DualPreferenceHead()
                    self._dual_mode = True
                else:
                    self._head = PreferenceHead()
                    self._dual_mode = False

                self._head.load_state_dict(state)
                self._head.eval()
                if torch.cuda.is_available():
                    self._head = self._head.cuda()
                self._bootstrap = False

                if val_accuracy is not None:
                    self._scaler.update_accuracy(val_accuracy)

                mode_str = "dual (CLAP+MERT)" if self._dual_mode else "single (CLAP)"
                logger.info(f"Preference model loaded from {model_path} [{mode_str}]")
            except Exception as exc:
                logger.warning(f"Failed to load preference head, using bootstrap: {exc}")
                self._head = None
                self._bootstrap = True
        else:
            logger.info("No preference model checkpoint — running in bootstrap mode")

    @property
    def is_bootstrap(self) -> bool:
        """True when no trained checkpoint is loaded (returns constant 0.5)."""
        return self._bootstrap

    def score(self, audio: np.ndarray, sr: int) -> float:
        """
        Score audio for predicted human preference.

        Args:
            audio: 1-D float waveform.
            sr: Sample rate in Hz.

        Returns:
            Preference score in [0, 1].
        """
        if self._bootstrap:
            return self._heuristic_score(audio, sr)

        try:
            if self._dual_mode:
                return self._score_dual(audio, sr)
            return self._score_single(audio, sr)

        except Exception as exc:
            logger.error(f"Preference scoring failed: {exc}")
            return self._heuristic_score(audio, sr)

    def _score_single(self, audio: np.ndarray, sr: int) -> float:
        """Score using single CLAP embedding (512-dim)."""
        embedding = self._clap.get_audio_embedding(audio, sr)
        if embedding is None:
            return self._heuristic_score(audio, sr)

        tensor = torch.from_numpy(embedding).float().unsqueeze(0)
        if next(self._head.parameters()).is_cuda:
            tensor = tensor.cuda()

        with torch.no_grad():
            pred = torch.sigmoid(self._head(tensor)).item()

        return float(np.clip(pred, 0.0, 1.0))

    def _score_dual(self, audio: np.ndarray, sr: int) -> float:
        """Score using dual CLAP+MERT embedding (1280-dim)."""
        # Get CLAP embedding (512-dim)
        clap_emb = self._clap.get_audio_embedding(audio, sr)
        if clap_emb is None:
            return self._heuristic_score(audio, sr)

        # Get MERT pooled embedding (768-dim)
        mert_emb = self._get_mert_embedding(audio, sr)
        if mert_emb is None:
            return self._heuristic_score(audio, sr)

        # Concatenate to 1280-dim
        combined = np.concatenate([clap_emb, mert_emb])
        tensor = torch.from_numpy(combined).float().unsqueeze(0)
        if next(self._head.parameters()).is_cuda:
            tensor = tensor.cuda()

        with torch.no_grad():
            pred = torch.sigmoid(self._head(tensor)).item()

        return float(np.clip(pred, 0.0, 1.0))

    def _get_mert_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """Extract pooled MERT embedding (768-dim) from neural scorer."""
        if self._neural is None:
            logger.warning("Dual mode requires neural_scorer but none provided")
            return None

        try:
            self._neural._load()
            if self._neural._model is None or self._neural._model == "LOAD_FAILED":
                return None

            hidden_states = self._neural._extract_hidden_states(audio, sr)
            if hidden_states is None:
                return None

            # Pool last hidden layer: mean over time steps -> 768-dim vector
            last_layer = hidden_states[-1]  # [T, 768]
            pooled = last_layer.mean(dim=0).numpy()  # [768]
            return pooled

        except Exception as exc:
            logger.error(f"MERT embedding extraction failed: {exc}")
            return None

    def get_scaled_weight(self) -> float:
        """Get the dynamically scaled preference weight."""
        return self._scaler.get_scaled_weight()

    def _heuristic_score(self, audio: np.ndarray, sr: int) -> float:
        """Bootstrap fallback: return neutral 0.5 when no trained model.

        The old heuristic extracted 12 features that overlapped with other
        scorers (spectral centroid, RMS, harmonic clarity, etc.), causing
        double-counting. A neutral 0.5 avoids distorting the composite
        score while the preference model is untrained.
        """
        return 0.5


class BradleyTerryLoss(torch.nn.Module):
    """Bradley-Terry pairwise preference loss for training.

    Given embeddings for preferred (A) and rejected (B) audio,
    trains the preference head to satisfy P(A > B) = sigmoid(s(A) - s(B)).

    This loss function better matches the A/B annotation format used in
    TuneForge's crowd annotation system.
    """

    def forward(
        self,
        score_preferred: torch.Tensor,
        score_rejected: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Bradley-Terry loss.

        Args:
            score_preferred: Scalar scores for preferred audio (batch).
            score_rejected: Scalar scores for rejected audio (batch).

        Returns:
            Mean negative log-likelihood of correct ranking.
        """
        # P(A > B) = sigmoid(s(A) - s(B))
        diff = score_preferred - score_rejected
        loss = -torch.nn.functional.logsigmoid(diff)
        return loss.mean()


class PairwisePreferenceTrainer:
    """Train preference model on pairwise A/B comparisons.

    Uses Bradley-Terry loss instead of pointwise MSE regression.
    """

    def __init__(
        self,
        head: PreferenceHead | DualPreferenceHead,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ) -> None:
        self._head = head
        self._loss_fn = BradleyTerryLoss()
        self._optimizer = torch.optim.AdamW(
            head.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train_step(
        self,
        preferred_embedding: torch.Tensor,
        rejected_embedding: torch.Tensor,
    ) -> float:
        """Single training step on a batch of pairwise comparisons.

        Args:
            preferred_embedding: Embeddings for preferred audio (B, D).
            rejected_embedding: Embeddings for rejected audio (B, D).

        Returns:
            Loss value.
        """
        self._head.train()
        self._optimizer.zero_grad()

        score_a = self._head(preferred_embedding).squeeze(-1)
        score_b = self._head(rejected_embedding).squeeze(-1)

        loss = self._loss_fn(score_a, score_b)
        loss.backward()
        self._optimizer.step()

        return float(loss.item())

    def validate(
        self,
        preferred_embeddings: torch.Tensor,
        rejected_embeddings: torch.Tensor,
    ) -> float:
        """Compute pairwise accuracy on validation set.

        Returns fraction of pairs where preferred > rejected.
        """
        self._head.eval()
        with torch.no_grad():
            score_a = self._head(preferred_embeddings).squeeze(-1)
            score_b = self._head(rejected_embeddings).squeeze(-1)
            correct = (score_a > score_b).float().mean()
        return float(correct.item())
