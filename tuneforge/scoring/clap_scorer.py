"""
CLAP-based text-audio similarity scorer for TuneForge.

Uses laion/larger_clap_music to compute cosine similarity between
a text prompt and generated audio, measuring prompt adherence.
"""

import numpy as np
import torch
from loguru import logger

from tuneforge.config.scoring_config import CLAP_MODEL, CLAP_SAMPLE_RATE, CLAP_SIM_FLOOR, CLAP_SIM_CEILING


class CLAPScorer:
    """Score text-audio relevance via CLAP embeddings."""

    def __init__(self, model_name: str = CLAP_MODEL) -> None:
        self._model_name = model_name
        self._model: torch.nn.Module | None = None
        self._processor = None
        self._last_embedding: np.ndarray | None = None

    @property
    def last_embedding(self) -> np.ndarray | None:
        """Last computed audio embedding (for reuse by FAD/diversity)."""
        return self._last_embedding

    def _load(self) -> None:
        """Lazy-load CLAP model and processor."""
        if self._model is not None:
            return
        try:
            import os
            from transformers import ClapModel, ClapProcessor

            # Use local_files_only when offline env is set or cache exists,
            # to avoid hanging on HuggingFace network calls
            local_only = os.environ.get("HF_HUB_OFFLINE", "") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "") == "1"

            logger.info(f"Loading CLAP model: {self._model_name} (local_only={local_only})")
            self._processor = ClapProcessor.from_pretrained(self._model_name, local_files_only=local_only)
            self._model = ClapModel.from_pretrained(self._model_name, local_files_only=local_only)
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            logger.info("CLAP model loaded")
        except Exception as exc:
            logger.error(f"Failed to load CLAP model: {exc}")
            raise

    def _resample(self, audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if sr == target_sr:
            return audio
        try:
            import librosa

            return librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        except Exception as exc:
            logger.warning(f"Resample failed, using raw audio: {exc}")
            return audio

    def score(self, audio: np.ndarray, sample_rate: int, text_prompt: str) -> float:
        """
        Compute CLAP similarity between audio and text prompt.

        Args:
            audio: Audio waveform as 1-D float array.
            sample_rate: Audio sample rate in Hz.
            text_prompt: Text description of desired music.

        Returns:
            Similarity score in [0, 1].
        """
        try:
            self._load()

            audio_resampled = self._resample(audio, sample_rate, CLAP_SAMPLE_RATE)

            # Ensure mono float32
            if audio_resampled.ndim > 1:
                audio_resampled = audio_resampled.mean(axis=0)
            audio_resampled = audio_resampled.astype(np.float32)

            device = next(self._model.parameters()).device

            # Get text embeddings
            text_inputs = self._processor(
                text=[text_prompt], return_tensors="pt", padding=True
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                text_out = self._model.get_text_features(**text_inputs)
                # Handle both raw tensor and BaseModelOutputWithPooling
                text_embeds = text_out if isinstance(text_out, torch.Tensor) else text_out.pooler_output

            # Get audio embeddings
            audio_inputs = self._processor(
                audio=[audio_resampled],
                sampling_rate=CLAP_SAMPLE_RATE,
                return_tensors="pt",
            )
            audio_inputs = {k: v.to(device) for k, v in audio_inputs.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                audio_out = self._model.get_audio_features(**audio_inputs)
                audio_embeds = audio_out if isinstance(audio_out, torch.Tensor) else audio_out.pooler_output

            # Cache raw audio embedding for reuse by FAD/diversity scorers
            self._last_embedding = audio_embeds.cpu().numpy().flatten()

            # Cosine similarity → [0, 1] via calibrated range mapping
            text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)
            audio_embeds = torch.nn.functional.normalize(audio_embeds, dim=-1)
            cosine_sim = (text_embeds * audio_embeds).sum(dim=-1).item()
            # Remap from empirical CLAP range [floor, ceiling] to [0, 1]
            denom = CLAP_SIM_CEILING - CLAP_SIM_FLOOR
            if denom <= 0:
                score = float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))
            else:
                score = float(np.clip((cosine_sim - CLAP_SIM_FLOOR) / denom, 0.0, 1.0))

            logger.debug(f"CLAP score: {score:.4f} (cosine={cosine_sim:.4f})")
            return score

        except Exception as exc:
            logger.error(f"CLAP scoring failed: {exc}")
            return 0.0

    def get_audio_embedding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
        """
        Get CLAP audio embedding for diversity/FAD checks.

        Returns:
            512-dim embedding vector, or None on failure.
        """
        try:
            self._load()

            audio_resampled = self._resample(audio, sample_rate, CLAP_SAMPLE_RATE)
            if audio_resampled.ndim > 1:
                audio_resampled = audio_resampled.mean(axis=0)
            audio_resampled = audio_resampled.astype(np.float32)

            device = next(self._model.parameters()).device

            audio_inputs = self._processor(
                audio=[audio_resampled],
                sampling_rate=CLAP_SAMPLE_RATE,
                return_tensors="pt",
            )
            audio_inputs = {k: v.to(device) for k, v in audio_inputs.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                audio_out = self._model.get_audio_features(**audio_inputs)
                embeds = audio_out if isinstance(audio_out, torch.Tensor) else audio_out.pooler_output

            return embeds.cpu().numpy().flatten()

        except Exception as exc:
            logger.error(f"CLAP embedding extraction failed: {exc}")
            return None
