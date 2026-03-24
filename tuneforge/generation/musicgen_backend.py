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

    # MusicGen's hard limit per chunk: 1503 tokens (~30s) from position embeddings
    MAX_TOKENS_PER_CHUNK: int = 1503
    CHUNK_DURATION: float = 30.0
    # Overlap in seconds — tail of previous chunk used as audio prompt for next
    OVERLAP_SECONDS: float = 5.0

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

        For durations > 30s, generates in chunks using the tail of each
        chunk as audio prompt for the next, producing seamless longer audio.

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

        t0 = time.time()

        if duration_seconds <= self.CHUNK_DURATION:
            # Single chunk — simple generation
            audio_np = self._generate_chunk(
                prompt, duration_seconds, guidance_scale, temperature, top_k, top_p
            )
        else:
            # Chunked generation for long audio
            audio_np = self._generate_chunked(
                prompt, duration_seconds, guidance_scale, temperature, top_k, top_p
            )

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

    def _generate_chunk(
        self,
        prompt: str,
        duration_seconds: float,
        guidance_scale: float,
        temperature: float,
        top_k: int,
        top_p: float,
        audio_prompt: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Generate a single chunk of audio (max 30s).

        Args:
            audio_prompt: Optional raw audio tensor [1, samples] to condition on.
                          Passed as ``input_values`` so MusicGen's audio encoder
                          encodes it and the decoder continues from that context.
        """
        max_new_tokens = min(int(duration_seconds * 50), self.MAX_TOKENS_PER_CHUNK)

        try:
            inputs = self._processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            generate_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "guidance_scale": guidance_scale,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "top_k": top_k if top_k > 0 else None,
                "top_p": top_p if top_p > 0 else None,
            }

            # If we have an audio prompt (continuation), pass the raw waveform
            # via input_values so MusicGen encodes it and continues from it.
            if audio_prompt is not None:
                generate_kwargs["input_values"] = audio_prompt

            with torch.no_grad():
                audio_values = self._model.generate(**generate_kwargs)

            audio_np = audio_values[0].cpu().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np[0]

            return audio_np.astype(np.float32)

        except Exception as exc:
            logger.error(f"MusicGen chunk generation failed: {exc}")
            raise RuntimeError(f"MusicGen generation failed: {exc}") from exc

    def _generate_chunked(
        self,
        prompt: str,
        duration_seconds: float,
        guidance_scale: float,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> np.ndarray:
        """Generate long audio by stitching 30s chunks with overlap.

        Each chunk after the first uses the tail of the previous chunk as
        an audio prompt (via input_values) so MusicGen continues the same
        musical idea rather than generating independent segments.
        """
        remaining = duration_seconds
        all_chunks: list[np.ndarray] = []
        overlap_samples = int(self.OVERLAP_SECONDS * self.SAMPLE_RATE)
        chunk_num = 0
        prev_audio_tail: torch.Tensor | None = None

        logger.info(
            f"Chunked generation: {duration_seconds}s total, "
            f"{self.CHUNK_DURATION}s per chunk, {self.OVERLAP_SECONDS}s overlap"
        )

        while remaining > 0:
            chunk_dur = min(remaining + self.OVERLAP_SECONDS, self.CHUNK_DURATION)
            chunk_num += 1

            logger.info(f"  Chunk {chunk_num}: generating {chunk_dur:.1f}s ({remaining:.1f}s remaining)")

            chunk = self._generate_chunk(
                prompt, chunk_dur, guidance_scale, temperature, top_k, top_p,
                audio_prompt=prev_audio_tail,
            )

            # Keep the tail of this chunk as audio context for the next one
            tail_samples = int(self.OVERLAP_SECONDS * self.SAMPLE_RATE)
            if len(chunk) >= tail_samples:
                prev_audio_tail = torch.tensor(
                    chunk[-tail_samples:], dtype=torch.float32,
                ).unsqueeze(0).to(self.device)
            else:
                prev_audio_tail = None

            if all_chunks:
                # Crossfade: blend the overlap region
                chunk = self._crossfade(all_chunks[-1], chunk, overlap_samples)
                all_chunks[-1] = chunk
            else:
                all_chunks.append(chunk)

            remaining -= (self.CHUNK_DURATION - self.OVERLAP_SECONDS)

        # Concatenate all chunks
        audio = np.concatenate(all_chunks) if len(all_chunks) > 1 else all_chunks[0]

        # Trim to exact requested duration
        target_samples = int(duration_seconds * self.SAMPLE_RATE)
        if len(audio) > target_samples:
            audio = audio[:target_samples]

        return audio

    @staticmethod
    def _crossfade(prev: np.ndarray, curr: np.ndarray, overlap_samples: int) -> np.ndarray:
        """Crossfade two audio chunks at the overlap region."""
        if len(prev) < overlap_samples or len(curr) < overlap_samples:
            return np.concatenate([prev, curr])

        # Create fade curves
        fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)

        # Blend the overlap region
        prev_tail = prev[-overlap_samples:] * fade_out
        curr_head = curr[:overlap_samples] * fade_in
        blended = prev_tail + curr_head

        # Stitch: prev (without tail) + blended + curr (without head)
        return np.concatenate([prev[:-overlap_samples], blended, curr[overlap_samples:]])
