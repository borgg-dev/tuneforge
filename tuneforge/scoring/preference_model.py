"""
Preference model for TuneForge.

A lightweight MLP on top of CLAP audio embeddings that predicts
human preference.  Falls back to a heuristic proxy when no trained
model checkpoint is available (bootstrap mode).
"""

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from pathlib import Path

from tuneforge.config.scoring_config import CLAP_MODEL
from tuneforge.scoring.clap_scorer import CLAPScorer


class PreferenceHead(nn.Module):
    """MLP: 512-dim CLAP embedding → scalar preference score."""

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
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PreferenceModel:
    """
    Predict human preference from audio.

    Uses CLAP audio embeddings fed through a trained MLP head.
    If no trained model path is given, runs in bootstrap mode using
    a simple audio quality heuristic.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._clap = CLAPScorer(model_name=CLAP_MODEL)
        self._head: PreferenceHead | None = None
        self._bootstrap = True
        self._model_path = model_path

        if model_path is not None and Path(model_path).exists():
            try:
                self._head = PreferenceHead()
                state = torch.load(model_path, map_location="cpu", weights_only=True)
                self._head.load_state_dict(state)
                self._head.eval()
                if torch.cuda.is_available():
                    self._head = self._head.cuda()
                self._bootstrap = False
                logger.info(f"Preference model loaded from {model_path}")
            except Exception as exc:
                logger.warning(f"Failed to load preference head, using bootstrap: {exc}")
                self._head = None
                self._bootstrap = True
        else:
            logger.info("No preference model checkpoint — running in bootstrap mode")

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
            embedding = self._clap.get_audio_embedding(audio, sr)
            if embedding is None:
                return self._heuristic_score(audio, sr)

            tensor = torch.from_numpy(embedding).float().unsqueeze(0)
            if next(self._head.parameters()).is_cuda:
                tensor = tensor.cuda()

            with torch.no_grad():
                pred = self._head(tensor).item()

            return float(np.clip(pred, 0.0, 1.0))

        except Exception as exc:
            logger.error(f"Preference scoring failed: {exc}")
            return self._heuristic_score(audio, sr)

    def _heuristic_score(self, audio: np.ndarray, sr: int) -> float:
        """
        Bootstrap heuristic based on audio energy and spectral properties.

        Used when no trained preference head is available.
        """
        try:
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.001:
                return 0.0

            peak = float(np.max(np.abs(audio)))
            if peak < 1e-8:
                return 0.0

            # Crest factor (peak / RMS) — good music ≈ 3–10
            crest = peak / rms
            crest_score = float(np.clip((crest - 1.0) / 9.0, 0.0, 1.0))

            # Energy consistency across frames
            frame_len = 2048
            n_frames = max(1, len(audio) // frame_len)
            frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
            rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1))
            rms_std = float(np.std(rms_per_frame))
            rms_mean = float(np.mean(rms_per_frame))
            consistency = 1.0 - min(rms_std / (rms_mean + 1e-8), 1.0)

            return float(np.clip(0.5 * crest_score + 0.5 * consistency, 0.0, 1.0))

        except Exception as exc:
            logger.error(f"Heuristic scoring failed: {exc}")
            return 0.0
