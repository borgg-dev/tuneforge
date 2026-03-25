"""
ACE-Step 1.5 backend for TuneForge subnet.

Wraps ACE-Step 1.5 (diffusion-based music generation) for text-to-music
generation with optional lyrics support. Produces 48kHz stereo audio.

Requires the ACE-Step repo on PYTHONPATH and transformers<4.58.
"""

import os
import sys
import time
from typing import Any

import numpy as np
import torch
from loguru import logger


# Default path where ACE-Step repo is cloned (contains the acestep package)
_DEFAULT_ACESTEP_PATH = os.environ.get(
    "ACESTEP_PATH",
    os.path.expanduser("~/ace-step-repo"),
)

# Default path for model checkpoints (separate from code)
_DEFAULT_ACESTEP_CKPT_PATH = os.environ.get(
    "ACESTEP_CKPT_PATH",
    os.path.expanduser("~/ACE-Step-1.5"),
)


def _ensure_acestep_importable(repo_path: str = _DEFAULT_ACESTEP_PATH) -> None:
    """Add ACE-Step repo to sys.path if not already importable."""
    try:
        import acestep  # noqa: F401
    except ImportError:
        if os.path.isdir(repo_path):
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
            logger.info(f"Added ACE-Step to PYTHONPATH: {repo_path}")
        else:
            raise ImportError(
                f"ACE-Step repo not found at {repo_path}. "
                "Clone it: git clone https://github.com/AceStepAI/ACE-Step-1.5 "
                "or set ACESTEP_PATH env var."
            )


class AceStepBackend:
    """Music generation backend using ACE-Step 1.5 (diffusion-based).

    Produces high-quality 48kHz stereo audio from text prompts.
    Supports instrumental generation and lyric-conditioned generation.
    """

    SAMPLE_RATE: int = 48000
    model_id: str = "ACE-Step-1.5"

    def __init__(
        self,
        device: str = "cuda",
        repo_path: str | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        self.device = device
        self._repo_path = repo_path or _DEFAULT_ACESTEP_PATH
        self._ckpt_path = ckpt_path or _DEFAULT_ACESTEP_CKPT_PATH
        self._handler: Any = None
        self._loaded = False

        logger.info(
            f"AceStepBackend initialized: device={self.device}, "
            f"repo={self._repo_path}, ckpt={self._ckpt_path}"
        )

    def load(self) -> None:
        """Load ACE-Step model into memory."""
        if self._loaded:
            return

        logger.info("Loading ACE-Step 1.5 model...")
        t0 = time.time()

        try:
            _ensure_acestep_importable(self._repo_path)
            from acestep.handler import AceStepHandler

            self._handler = AceStepHandler()
            status, ok = self._handler.initialize_service(
                project_root=self._ckpt_path,
                config_path="acestep-v15-turbo",
                device=self.device,
            )
            if not ok:
                raise RuntimeError(f"ACE-Step init failed: {status}")

            self._loaded = True
            elapsed = time.time() - t0
            logger.info(f"ACE-Step model loaded in {elapsed:.1f}s")
        except Exception as exc:
            logger.error(f"Failed to load ACE-Step model: {exc}")
            raise RuntimeError(f"ACE-Step model load failed: {exc}") from exc

    def unload(self) -> None:
        """Unload model from memory and free GPU resources."""
        if self._handler is not None:
            # Clear model references
            for attr in ("model", "vae", "text_encoder"):
                obj = getattr(self._handler, attr, None)
                if obj is not None:
                    del obj
                    setattr(self._handler, attr, None)
            del self._handler
            self._handler = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("ACE-Step model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(
        self,
        prompt: str,
        duration_seconds: float,
        seed: int | None = None,
        lyrics: str | None = None,
        guidance_scale: float = 3.0,
        **kwargs: Any,
    ) -> tuple[np.ndarray, int]:
        """Generate audio from a text prompt.

        Args:
            prompt: Natural language description of desired music.
            duration_seconds: Desired audio duration in seconds.
            seed: Random seed for reproducibility.
            lyrics: Optional lyrics (use "[Instrumental]" for no vocals).
            guidance_scale: Classifier-free guidance scale.
            **kwargs: Additional params (ignored).

        Returns:
            Tuple of (audio_array, sample_rate).
            Audio array is float32 in [-1, 1] range, mono (downmixed from stereo).
        """
        self.load()

        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        # Cap duration to model limits (ACE-Step supports up to ~5 min)
        duration_seconds = min(max(duration_seconds, 5), 300)

        if lyrics is None:
            lyrics = "[Instrumental]"

        logger.info(
            f"Generating: prompt='{prompt[:80]}...', "
            f"duration={duration_seconds}s, seed={seed}"
        )
        t0 = time.time()

        try:
            params = GenerationParams(
                caption=prompt,
                lyrics=lyrics,
                duration=duration_seconds,
                seed=seed if seed is not None else -1,
            )
            config = GenerationConfig(batch_size=1)
            result = generate_music(
                self._handler, None, params, config,
                save_dir=None,  # Don't save to disk, just return tensors
            )

            if not result.success:
                raise RuntimeError(f"ACE-Step generation failed: {result.status_message}")

            if not result.audios:
                raise RuntimeError("ACE-Step returned no audio")

            # result.audios is a list of dicts with "tensor" key (channels, samples)
            audio_entry = result.audios[0]
            audio_tensor = audio_entry.get("tensor") if isinstance(audio_entry, dict) else audio_entry
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = np.array(audio_tensor)

            # Downmix stereo to mono for subnet compatibility
            if audio_np.ndim == 2 and audio_np.shape[0] == 2:
                audio_np = audio_np.mean(axis=0)
            elif audio_np.ndim == 2:
                audio_np = audio_np[0]

            # Ensure float32 in [-1, 1]
            audio_np = audio_np.astype(np.float32)
            peak = np.max(np.abs(audio_np))
            if peak > 0:
                audio_np = audio_np / max(peak, 1.0)

            elapsed = time.time() - t0
            logger.info(
                f"Generation complete: {len(audio_np)/self.SAMPLE_RATE:.1f}s audio "
                f"in {elapsed:.1f}s ({elapsed/duration_seconds:.2f}x realtime)"
            )

            return audio_np, self.SAMPLE_RATE

        except Exception as exc:
            logger.error(f"ACE-Step generation failed: {exc}")
            raise RuntimeError(f"ACE-Step generation failed: {exc}") from exc
