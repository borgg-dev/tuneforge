"""
MusicGen backend for TuneForge subnet.

Wraps Meta's MusicGen models (facebook/musicgen-{size}) via the
Hugging Face transformers library for text-to-music generation.
"""

import time
from typing import Any

import numpy as np
import torch
from loguru import logger


class MusicGenBackend:
    """Music generation backend using Meta's MusicGen models.

    Supports small, medium, and large MusicGen variants.
    Generates audio from text prompts using autoregressive decoding.
    """

    SAMPLE_RATE: int = 32000
    SUPPORTS_VOCALS: bool = False

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
    ) -> None:
        """Initialize MusicGen backend.

        Args:
            model_size: Model variant — "small", "medium", or "large".
            device: Torch device string ("cuda", "cuda:0", "cpu").
        """
        self.model_size = model_size
        self.model_id = f"facebook/musicgen-{model_size}"
        self.device = device
        self._processor: Any = None
        self._model: Any = None
        self._loaded = False

        logger.info(f"MusicGenBackend initialized: model={self.model_id}, device={self.device}")

    def load(self) -> None:
        """Load model and processor into memory.

        Called automatically on first generate() call if not already loaded.
        """
        if self._loaded:
            return

        logger.info(f"Loading MusicGen model: {self.model_id}")
        t0 = time.time()

        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_id
            )
            self._model = self._model.to(self.device)
            self._loaded = True
            elapsed = time.time() - t0
            logger.info(f"MusicGen model loaded in {elapsed:.1f}s")
        except Exception as exc:
            logger.error(f"Failed to load MusicGen model {self.model_id}: {exc}")
            raise RuntimeError(f"MusicGen model load failed: {exc}") from exc

    def unload(self) -> None:
        """Unload model from memory and free GPU resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("MusicGen model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._loaded

    def generate(
        self,
        prompt: str,
        duration_seconds: float,
        guidance_scale: float = 3.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        seed: int | None = None,
        **kwargs,  # Accept and ignore extra params (lyrics, etc.)
    ) -> tuple[np.ndarray, int]:
        """Generate audio from a text prompt.

        Args:
            prompt: Natural language description of desired music.
            duration_seconds: Desired audio duration in seconds.
            guidance_scale: Classifier-free guidance scale.
            temperature: Sampling temperature.
            top_k: Top-K sampling parameter.
            top_p: Nucleus sampling parameter (0 = disabled).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (audio_array, sample_rate).
            Audio array is float32 in [-1, 1] range.
        """
        self.load()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # MusicGen generates ~50 tokens per second of audio
        max_new_tokens = min(int(duration_seconds * 50), 1500)

        logger.info(
            f"Generating: prompt='{prompt[:80]}...', "
            f"duration={duration_seconds}s, tokens={max_new_tokens}, "
            f"guidance={guidance_scale}, temp={temperature}"
        )
        t0 = time.time()

        try:
            inputs = self._processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                audio_values = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    guidance_scale=guidance_scale,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p if top_p > 0 else None,
                )

            # Extract audio: shape is (batch, channels, samples) or (batch, samples)
            audio_np = audio_values[0].cpu().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np[0]  # Take first channel

            # Ensure float32 in [-1, 1]
            audio_np = audio_np.astype(np.float32)
            peak = np.max(np.abs(audio_np))
            if peak > 1.0:
                audio_np = audio_np / peak

            elapsed = time.time() - t0
            logger.info(
                f"Generation complete: {len(audio_np)/self.SAMPLE_RATE:.1f}s audio "
                f"in {elapsed:.1f}s ({elapsed/duration_seconds:.2f}x realtime)"
            )

            return audio_np, self.SAMPLE_RATE

        except Exception as exc:
            logger.error(f"MusicGen generation failed: {exc}")
            raise RuntimeError(f"MusicGen generation failed: {exc}") from exc
