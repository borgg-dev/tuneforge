"""
DiffRhythm backend for TuneForge subnet.

Wraps ASLP-lab's DiffRhythm v1.2 (latent diffusion) for text-to-music
generation. Produces 44.1kHz stereo audio, supports lyrics.

Requires the DiffRhythm repo on PYTHONPATH and espeak-ng system dependency.
"""

import os
import sys
import time
from typing import Any

import numpy as np
import torch
from loguru import logger


# Default path where DiffRhythm repo is cloned
_DEFAULT_DIFFRHYTHM_PATH = os.environ.get(
    "DIFFRHYTHM_PATH",
    os.path.expanduser("~/DiffRhythm"),
)


def _ensure_diffrhythm_importable(repo_path: str = _DEFAULT_DIFFRHYTHM_PATH) -> None:
    """Add DiffRhythm repo to sys.path if not already importable."""
    # We need the infer/ directory importable
    infer_path = os.path.join(repo_path, "infer")
    if not os.path.isdir(repo_path):
        raise ImportError(
            f"DiffRhythm repo not found at {repo_path}. "
            "Clone it: git clone https://github.com/ASLP-lab/DiffRhythm "
            "or set DIFFRHYTHM_PATH env var."
        )
    for p in [repo_path, infer_path]:
        if p not in sys.path:
            sys.path.insert(0, p)
    logger.info(f"Added DiffRhythm to PYTHONPATH: {repo_path}")


class DiffRhythmBackend:
    """Music generation backend using DiffRhythm v1.2 (latent diffusion).

    Produces high-quality 44.1kHz audio from text prompts.
    Supports full-length song generation (up to 4m45s) and lyrics.
    """

    SAMPLE_RATE: int = 44100
    model_id: str = "DiffRhythm-v1.2"

    def __init__(
        self,
        device: str = "cuda",
        repo_path: str | None = None,
        use_full_model: bool = False,
    ) -> None:
        """Initialize DiffRhythm backend.

        Args:
            device: Torch device string.
            repo_path: Path to cloned DiffRhythm repo.
            use_full_model: Use full model (285s/4m45s) vs base (95s/1m35s).
        """
        self.device = device
        self._repo_path = repo_path or _DEFAULT_DIFFRHYTHM_PATH
        self._use_full = use_full_model
        self._max_frames = 6144 if use_full_model else 2048
        self._max_duration = 285.0 if use_full_model else 95.0

        self._cfm: Any = None
        self._tokenizer: Any = None
        self._muq: Any = None
        self._vae: Any = None
        self._loaded = False

        logger.info(
            f"DiffRhythmBackend initialized: device={self.device}, "
            f"model={'full' if use_full_model else 'base'}, "
            f"max_duration={self._max_duration}s"
        )

    def load(self) -> None:
        """Load DiffRhythm model into memory."""
        if self._loaded:
            return

        logger.info(f"Loading DiffRhythm {'full' if self._use_full else 'base'} model...")
        t0 = time.time()

        try:
            _ensure_diffrhythm_importable(self._repo_path)

            # Change to repo dir so relative paths in infer_utils work
            original_cwd = os.getcwd()
            os.chdir(self._repo_path)

            try:
                from infer_utils import prepare_model

                self._cfm, self._tokenizer, self._muq, self._vae = prepare_model(
                    self._max_frames, self.device
                )
            finally:
                os.chdir(original_cwd)

            self._loaded = True
            elapsed = time.time() - t0
            logger.info(f"DiffRhythm model loaded in {elapsed:.1f}s")
        except Exception as exc:
            logger.error(f"Failed to load DiffRhythm model: {exc}")
            raise RuntimeError(f"DiffRhythm model load failed: {exc}") from exc

    def unload(self) -> None:
        """Unload model from memory and free GPU resources."""
        for attr in ("_cfm", "_muq", "_vae", "_tokenizer"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("DiffRhythm model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(
        self,
        prompt: str,
        duration_seconds: float,
        seed: int | None = None,
        lyrics: str | None = None,
        guidance_scale: float = 4.0,
        **kwargs: Any,
    ) -> tuple[np.ndarray, int]:
        """Generate audio from a text prompt.

        Args:
            prompt: Style description (e.g. "Jazzy nightclub vibe, saxophone solo").
            duration_seconds: Desired audio duration in seconds.
            seed: Random seed for reproducibility.
            lyrics: Optional LRC-format lyrics. Use "[Instrumental]" for no vocals.
            guidance_scale: Not directly exposed (hardcoded in DiffRhythm at 4.0).
            **kwargs: Additional params (ignored).

        Returns:
            Tuple of (audio_array, sample_rate).
            Audio array is float32 in [-1, 1] range, mono (downmixed from stereo).
        """
        self.load()

        # Clamp duration to model limits
        duration_seconds = min(max(duration_seconds, 5), self._max_duration)

        # Determine audio_length param (maps to frame count)
        # Base model: exactly 95 (maps to 2048 frames)
        # Full model: 96-285 (maps to 6144 frames)
        if self._use_full:
            audio_length = min(max(int(duration_seconds), 96), 285)
        else:
            audio_length = 95  # Base model is fixed at 95s / 2048 frames

        if lyrics is None:
            lyrics = "[Instrumental]"

        logger.info(
            f"Generating: prompt='{prompt[:80]}...', "
            f"duration={duration_seconds}s, audio_length={audio_length}, seed={seed}"
        )
        t0 = time.time()

        try:
            # Change to repo dir for relative path access
            original_cwd = os.getcwd()
            os.chdir(self._repo_path)

            try:
                from infer import inference
                from infer_utils import (
                    get_lrc_token,
                    get_negative_style_prompt,
                    get_reference_latent,
                    get_style_prompt,
                )

                if seed is not None:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)

                # Style from text prompt
                style_prompt = get_style_prompt(self._muq, prompt=prompt)
                negative_style_prompt = get_negative_style_prompt(self.device)

                # No editing — generate from scratch
                latent_prompt, pred_frames = get_reference_latent(
                    self.device, self._max_frames, False, None, None, None
                )

                # Tokenize lyrics
                lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
                    self._max_frames, lyrics, self._tokenizer, audio_length, self.device
                )

                # Run inference
                with torch.no_grad():
                    results = inference(
                        cfm_model=self._cfm,
                        vae_model=self._vae,
                        cond=latent_prompt,
                        text=lrc_prompt,
                        duration=end_frame,
                        style_prompt=style_prompt,
                        negative_style_prompt=negative_style_prompt,
                        start_time=start_time,
                        pred_frames=pred_frames,
                        batch_infer_num=1,
                        song_duration=song_duration,
                        chunked=True,
                    )
            finally:
                os.chdir(original_cwd)

            # Results is a list of int16 tensors (shape [2, samples])
            audio_tensor = results[0]
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().numpy().astype(np.float32) / 32767.0
            else:
                audio_np = np.array(audio_tensor, dtype=np.float32) / 32767.0

            # Downmix stereo to mono for subnet compatibility
            if audio_np.ndim == 2 and audio_np.shape[0] == 2:
                audio_np = audio_np.mean(axis=0)
            elif audio_np.ndim == 2:
                audio_np = audio_np[0]

            # Trim to requested duration
            target_samples = int(duration_seconds * self.SAMPLE_RATE)
            if len(audio_np) > target_samples:
                audio_np = audio_np[:target_samples]

            # Normalize to [-1, 1]
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
            logger.error(f"DiffRhythm generation failed: {exc}")
            raise RuntimeError(f"DiffRhythm generation failed: {exc}") from exc
