"""
Stable Audio Open backend for TuneForge subnet.

Alternative music generation backend using Stability AI's Stable Audio Open
model via the Hugging Face diffusers library.
"""

import time
from typing import Any

import numpy as np
import torch
from loguru import logger


class StableAudioBackend:
    """Music generation backend using Stable Audio Open.

    Uses diffusion-based generation for high-quality audio synthesis
    from text prompts. Requires the `diffusers` library.
    """

    MODEL_ID: str = "stabilityai/stable-audio-open-1.0"
    SAMPLE_RATE: int = 44100

    def __init__(self, device: str = "cuda") -> None:
        """Initialize Stable Audio backend.

        Args:
            device: Torch device string ("cuda", "cuda:0", "cpu").
        """
        self.model_id = self.MODEL_ID
        self.device = device
        self._pipe: Any = None
        self._loaded = False

        logger.info(f"StableAudioBackend initialized: model={self.model_id}, device={self.device}")

    def load(self) -> None:
        """Load the Stable Audio pipeline into memory.

        Called automatically on first generate() call if not already loaded.
        """
        if self._loaded:
            return

        logger.info(f"Loading Stable Audio model: {self.model_id}")
        t0 = time.time()

        try:
            from diffusers import StableAudioPipeline

            self._pipe = StableAudioPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
            )
            self._pipe = self._pipe.to(self.device)
            self._loaded = True
            elapsed = time.time() - t0
            logger.info(f"Stable Audio model loaded in {elapsed:.1f}s")
        except Exception as exc:
            logger.error(f"Failed to load Stable Audio model: {exc}")
            raise RuntimeError(f"Stable Audio model load failed: {exc}") from exc

    def unload(self) -> None:
        """Unload model from memory and free GPU resources."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Stable Audio model unloaded")

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
            temperature: Not directly used by Stable Audio (kept for interface compatibility).
            top_k: Not directly used by Stable Audio (kept for interface compatibility).
            top_p: Not directly used by Stable Audio (kept for interface compatibility).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (audio_array, sample_rate).
            Audio array is float32 in [-1, 1] range.
        """
        self.load()

        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Stable Audio supports negative prompts for quality
        negative_prompt = "low quality, distorted, noise, clipping, static"

        logger.info(
            f"Generating: prompt='{prompt[:80]}...', "
            f"duration={duration_seconds}s, guidance={guidance_scale}"
        )
        t0 = time.time()

        try:
            output = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                audio_end_in_s=duration_seconds,
                num_inference_steps=100,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            # Output is a StableAudioPipelineOutput with .audios
            audio = output.audios[0]

            # Convert to numpy if still a torch tensor
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()

            # May be (channels, samples) — flatten to mono
            if audio.ndim > 1:
                audio = audio.mean(axis=0)

            audio_np = audio.astype(np.float32)
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
            logger.error(f"Stable Audio generation failed: {exc}")
            raise RuntimeError(f"Stable Audio generation failed: {exc}") from exc
