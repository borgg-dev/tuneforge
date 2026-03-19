"""
HeartMuLa backend for TuneForge subnet.

Wraps HeartMuLa (heartlib) for text+lyrics+tags to music generation.
Produces 48kHz audio with vocal/lyrics support. Uses the HeartMuLaGenPipeline.

Requires heartlib installed: git clone + pip install -e .
"""

import os
import tempfile
import time
from typing import Any

import numpy as np
import torch
import torchaudio
from loguru import logger


class HeartMuLaBackend:
    """Music generation backend using HeartMuLa (heartlib).

    Produces high-quality 48kHz audio from tags and lyrics.
    Supports full song generation with vocals up to 4 minutes.
    """

    SAMPLE_RATE: int = 48000
    SUPPORTS_VOCALS: bool = True
    model_id: str = "HeartMuLa-3B"

    def __init__(
        self,
        device: str = "cuda",
        model_path: str | None = None,
        version: str = "3B",
    ) -> None:
        """Initialize HeartMuLa backend.

        Args:
            device: Torch device string.
            model_path: Path to downloaded HeartMuLa checkpoint.
            version: Model version — "3B" or "7B".
        """
        self.device = device
        self._model_path = model_path or os.environ.get(
            "HEARTMULA_MODEL_PATH",
            os.path.expanduser("~/heartmula-ckpt"),
        )
        self._version = version
        self.model_id = f"HeartMuLa-{version}"
        self._pipe: Any = None
        self._loaded = False

        logger.info(
            f"HeartMuLaBackend initialized: version={version}, "
            f"device={self.device}, model_path={self._model_path}"
        )

    def load(self) -> None:
        """Load the HeartMuLa pipeline into memory with optimizations."""
        if self._loaded:
            return

        logger.info(f"Loading HeartMuLa {self._version} model...")
        t0 = time.time()

        try:
            from heartlib import HeartMuLaGenPipeline

            self._pipe = HeartMuLaGenPipeline.from_pretrained(
                self._model_path,
                device={
                    "mula": torch.device(self.device),
                    "codec": torch.device(self.device),
                },
                dtype={
                    "mula": torch.bfloat16,
                    "codec": torch.float32,
                },
                version=self._version,
                lazy_load=False,
            )

            self._loaded = True
            elapsed = time.time() - t0
            logger.info(f"HeartMuLa model loaded in {elapsed:.1f}s")
        except Exception as exc:
            logger.error(f"Failed to load HeartMuLa model: {exc}")
            raise RuntimeError(f"HeartMuLa model load failed: {exc}") from exc

    def _apply_optimizations(self) -> None:
        """Apply torch.compile to the backbone for faster inference.

        torch.compile optimizes the compute graph for ~20-30% speedup
        with zero quality impact. Uses 'max-autotune' mode for best
        performance on repeated generations.
        """
        model = self._pipe.mula

        try:
            model.backbone = torch.compile(
                model.backbone,
                mode="max-autotune",
                fullgraph=False,
            )
            logger.info("torch.compile applied to backbone (max-autotune mode)")
        except Exception as exc:
            logger.warning(f"torch.compile failed (non-fatal): {exc}")

    def unload(self) -> None:
        """Unload model from memory and free GPU resources."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("HeartMuLa model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @staticmethod
    def _build_tags(prompt: str, genre: str | None = None, mood: str | None = None, has_vocals: bool = False) -> str:
        """Build HeartMuLa tags from prompt, genre, and mood.

        HeartMuLa uses comma-separated tags like: 'reggae,chill,guitar,vocal'
        Only adds vocal-related tags when has_vocals is True.
        """
        tags = []

        # Add explicit genre/mood if provided
        if genre:
            tags.append(genre.lower())
        if mood:
            tags.append(mood.lower())

        # Extract additional tags from the prompt
        prompt_lower = prompt.lower()

        # Genre keywords
        genres = [
            "reggae", "rock", "pop", "jazz", "blues", "hip-hop", "rap",
            "electronic", "edm", "house", "techno", "ambient", "classical",
            "folk", "country", "r&b", "soul", "funk", "metal", "punk",
            "lo-fi", "lofi", "cinematic", "orchestral", "latin", "bossa nova",
            "psychedelic", "indie", "alternative", "gospel", "ska",
            "disco", "trap", "synthwave", "drum and bass", "world",
        ]
        for g in genres:
            if g in prompt_lower and g not in tags:
                tags.append(g)

        # Mood keywords
        moods = [
            "energetic", "chill", "dark", "uplifting", "melancholic", "dreamy",
            "aggressive", "peaceful", "romantic", "epic", "happy", "sad",
            "groovy", "mellow", "intense", "hopeful", "nostalgic",
        ]
        for m in moods:
            if m in prompt_lower and m not in tags:
                tags.append(m)

        # Instrument keywords
        instruments = [
            "piano", "guitar", "drums", "bass", "synth", "strings", "brass",
            "flute", "violin", "saxophone", "organ", "trumpet", "harmonica",
        ]
        for i in instruments:
            if i in prompt_lower and i not in tags:
                tags.append(i)

        # Only add vocal tag when explicitly requested — prevents unwanted vocals
        if has_vocals:
            if "vocal" not in tags:
                tags.append("vocal")
        else:
            # Explicitly add instrumental tag to suppress vocals
            tags.append("instrumental")

        # Fallback: use the prompt itself as tags if nothing was extracted
        if not tags:
            words = [w for w in prompt.split()[:5] if len(w) > 2]
            tags = words[:3] if words else ["music"]

        return ",".join(tags)

    @staticmethod
    def _format_lyrics(lyrics: str | None) -> str:
        """Format lyrics for HeartMuLa.

        HeartMuLa expects bracket-delimited sections:
        [Verse] ... [Chorus] ... [Bridge] ... [Outro]
        """
        if not lyrics or lyrics in ("[Instrumental]", "[Vocals]"):
            return ""

        # If already has section markers, return as-is
        if "[Verse]" in lyrics or "[Chorus]" in lyrics:
            return lyrics

        # Auto-add section markers to plain text lyrics
        lines = [l.strip() for l in lyrics.strip().split("\n") if l.strip()]
        if not lines:
            return ""

        # Simple structure: split lines into verse/chorus pattern
        sections = []
        chunk_size = max(2, len(lines) // 4)

        for i in range(0, len(lines), chunk_size):
            chunk = lines[i:i + chunk_size]
            if i == 0:
                sections.append("[Verse]")
            elif i == chunk_size:
                sections.append("\n[Chorus]")
            elif i == chunk_size * 2:
                sections.append("\n[Verse]")
            elif i == chunk_size * 3:
                sections.append("\n[Chorus]")
            else:
                sections.append("\n[Bridge]")
            sections.extend(chunk)

        return "\n".join(sections)

    def generate(
        self,
        prompt: str,
        duration_seconds: float,
        seed: int | None = None,
        lyrics: str | None = None,
        guidance_scale: float = 1.5,
        temperature: float = 1.0,
        top_k: int = 50,
        **kwargs: Any,
    ) -> tuple[np.ndarray, int]:
        """Generate audio from a text prompt.

        Args:
            prompt: Natural language description of desired music.
            duration_seconds: Desired audio duration in seconds.
            seed: Random seed for reproducibility.
            lyrics: Optional lyrics text. None or "[Instrumental]" for instrumental.
            guidance_scale: Classifier-free guidance scale.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            **kwargs: Additional params (genre, mood extracted from kwargs or prompt).

        Returns:
            Tuple of (audio_array, sample_rate).
            Audio array is float32 in [-1, 1] range, mono.
        """
        self.load()

        # Request 1.5x the duration — HeartMuLa is autoregressive and often
        # stops early on short requests. Generating more and trimming avoids
        # both duration penalties (too short) and artifact penalties (from looping).
        overshoot_seconds = min(duration_seconds * 1.5, 240.0)
        max_duration_ms = int(overshoot_seconds * 1000)

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Format lyrics
        formatted_lyrics = self._format_lyrics(lyrics)
        has_vocals = bool(formatted_lyrics)

        # Build tags from prompt — only include vocal tag when vocals actually requested
        genre = kwargs.get("genre")
        mood = kwargs.get("mood")
        tags = self._build_tags(prompt, genre, mood, has_vocals=has_vocals)

        logger.info(
            f"Generating: tags='{tags}', "
            f"vocals={'yes' if has_vocals else 'no'}, "
            f"duration={duration_seconds}s, temp={temperature}, "
            f"cfg={guidance_scale}, seed={seed}"
        )
        t0 = time.time()

        try:
            # HeartMuLa saves to a file, so we use a temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            with torch.no_grad():
                self._pipe(
                    {
                        "lyrics": formatted_lyrics if has_vocals else "",
                        "tags": tags,
                    },
                    max_audio_length_ms=max_duration_ms,
                    save_path=tmp_path,
                    topk=top_k,
                    temperature=temperature,
                    cfg_scale=guidance_scale,
                )

            # Load the generated audio
            audio, sr = torchaudio.load(tmp_path)

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            # Convert to numpy, mono
            audio_np = audio.numpy()
            if audio_np.ndim == 2 and audio_np.shape[0] >= 2:
                audio_np = audio_np.mean(axis=0)
            elif audio_np.ndim == 2:
                audio_np = audio_np[0]

            audio_np = audio_np.astype(np.float32)

            # Trim to exact requested duration (we generated 1.5x overshoot)
            target_samples = int(duration_seconds * sr)
            if len(audio_np) > target_samples:
                audio_np = audio_np[:target_samples]
            elif len(audio_np) < target_samples:
                # Still short despite overshoot — apply musical fade-out
                # to fill remaining time gracefully (sounds intentional)
                shortfall = target_samples - len(audio_np)
                fade_len = min(len(audio_np), int(2.0 * sr))  # 2s fade-out
                if fade_len > 0:
                    fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                    audio_np[-fade_len:] *= fade
                # Pad with silence after fade
                audio_np = np.pad(audio_np, (0, shortfall), mode="constant")
                logger.info(
                    f"Audio {len(audio_np)/sr - shortfall/sr:.1f}s, padded with fade-out to {duration_seconds}s"
                )

            # Normalize
            peak = np.max(np.abs(audio_np))
            if peak > 0:
                audio_np = audio_np / max(peak, 1.0)

            elapsed = time.time() - t0
            logger.info(
                f"Generation complete: {len(audio_np)/sr:.1f}s audio "
                f"in {elapsed:.1f}s ({elapsed/duration_seconds:.2f}x realtime)"
            )

            return audio_np, sr

        except Exception as exc:
            logger.error(f"HeartMuLa generation failed: {exc}")
            # Clean up temp file on error
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise RuntimeError(f"HeartMuLa generation failed: {exc}") from exc
