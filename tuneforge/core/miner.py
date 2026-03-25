"""
TuneForgeMiner — full miner implementation for TuneForge subnet.

Handles music generation requests from validators, processes prompts,
generates audio via configurable backends, and reports health metrics.
"""

import time
from typing import Tuple

import psutil
from loguru import logger

from tuneforge.base.miner import BaseMinerNeuron
from tuneforge.base.protocol import (
    HealthReportSynapse,
    MusicGenerationSynapse,
    PingSynapse,
)
from tuneforge.generation.audio_utils import AudioUtils
from tuneforge.generation.model_manager import ModelManager
from tuneforge.generation.prompt_parser import PromptParser
from tuneforge.settings import Settings, get_settings


# Minimum stake required for generation requests (filters out non-validators)
# Validators below this threshold are blacklisted.
MIN_GENERATION_STAKE: float = 10_000.0


class TuneForgeMiner(BaseMinerNeuron):
    """Full miner implementation for the TuneForge music generation subnet.

    Loads a music generation model, responds to both validator challenges
    and organic (product/API) requests with generated audio, and reports
    capability/health information.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the TuneForgeMiner.

        Args:
            settings: Optional settings override.
        """
        settings = settings or get_settings()
        super().__init__(settings=settings)

        # Parse model backend from model_name
        model_name = self.settings.model_name
        if "heartmula" in model_name.lower() or "heart-mula" in model_name.lower() or "heart_mula" in model_name.lower():
            model_size = "7B" if "7b" in model_name.lower() else "3B"
            backend = "heartmula"
        elif "diffrhythm" in model_name.lower() or "diff-rhythm" in model_name.lower() or "diff_rhythm" in model_name.lower():
            # DiffRhythm: "diffrhythm-full" for 285s, "diffrhythm" for 95s base
            model_size = "full" if "full" in model_name.lower() else "base"
            backend = "diffrhythm"
        elif "musicgen" in model_name:
            parts = model_name.split("-")
            model_size = parts[-1] if parts else "medium"
            backend = "musicgen"
        else:
            model_size = "medium"
            backend = "musicgen"  # Default to MusicGen

        self._model_manager = ModelManager(
            default_backend=backend,
            model_size=model_size,
            device=self.settings.gpu_device,
        )
        self._audio_utils = AudioUtils()
        self._prompt_parser = PromptParser()

        # Lyrics generator — only loaded if active backend supports vocals
        self._lyrics_gen = None

        # Generation tracking
        self._generations_completed: int = 0
        self._errors_last_hour: int = 0
        self._error_timestamps: list[float] = []
        self._generation_times: list[float] = []

        # Preload model so first challenge doesn't timeout on download/load
        logger.info(f"Preloading {backend} model...")
        self._model_manager.preload()

        # Preload lyrics generator only if backend supports vocals
        if self._model_manager.supports_vocals:
            from tuneforge.generation.lyrics_generator import LyricsGenerator
            self._lyrics_gen = LyricsGenerator(device="cpu")  # GPT-2 runs fine on CPU, saves GPU VRAM
            logger.info("Preloading lyrics generator (backend supports vocals)...")
            self._lyrics_gen.load()
        else:
            logger.info("Skipping lyrics generator (backend does not support vocals)")

        logger.info(
            f"TuneForgeMiner initialized: backend={backend}, "
            f"model_size={model_size}, device={self.settings.gpu_device}"
        )

    # ------------------------------------------------------------------
    # Music generation handler
    # ------------------------------------------------------------------

    def forward_generation(
        self, synapse: MusicGenerationSynapse
    ) -> MusicGenerationSynapse:
        """Generate music from either a validator challenge or an organic API query.

        Dispatches to the appropriate handler based on synapse.is_organic.

        Args:
            synapse: Incoming generation request with prompt and parameters.

        Returns:
            Synapse populated with generated audio and metadata.
        """
        if synapse.is_organic:
            return self._handle_organic(synapse)
        return self._handle_challenge(synapse)

    def _handle_challenge(
        self, synapse: MusicGenerationSynapse
    ) -> MusicGenerationSynapse:
        """Handle a validator challenge request (reward mechanism).

        The validator sends structured parameters to test the miner's
        generation quality. Scored for weights.
        """
        t0 = time.time()

        try:
            # For challenges, structured params lead (validator controls the test)
            enhanced_prompt = self._prompt_parser.build_prompt(
                text=synapse.prompt,
                genre=synapse.genre,
                mood=synapse.mood,
                tempo=synapse.tempo_bpm,
                instruments=synapse.instruments,
                key=synapse.key_signature,
                time_sig=synapse.time_signature,
            )
            logger.info(
                f"[CHALLENGE] {synapse.challenge_id}: "
                f"prompt='{enhanced_prompt[:100]}', "
                f"duration={synapse.duration_seconds}s, "
                f"vocals={synapse.vocals_requested}, lyrics={'yes' if synapse.lyrics else 'no'}"
            )

            audio, sr, wav_bytes, elapsed_ms = self._generate_audio(
                enhanced_prompt, synapse, t0
            )

            synapse.audio_b64 = self._audio_utils.to_base64(wav_bytes)
            synapse.sample_rate = sr
            synapse.generation_time_ms = elapsed_ms
            synapse.model_id = self._model_manager.active_model_id

            self._generations_completed += 1
            self._generation_times.append(elapsed_ms)
            if len(self._generation_times) > 100:
                self._generation_times = self._generation_times[-100:]

            logger.info(
                f"[CHALLENGE] {synapse.challenge_id} complete: "
                f"{elapsed_ms}ms, {len(wav_bytes)} bytes"
            )

        except Exception as exc:
            elapsed_ms = int((time.time() - t0) * 1000)
            logger.error(f"[CHALLENGE] Failed {synapse.challenge_id}: {exc}")
            synapse.generation_time_ms = elapsed_ms
            self._record_error()

        return synapse

    def _handle_organic(
        self, synapse: MusicGenerationSynapse
    ) -> MusicGenerationSynapse:
        """Handle an organic (product/API) request from a paying customer.

        The user's free-text prompt is the primary input. Structured
        parameters (genre, mood, etc.) only supplement it.
        This path does NOT affect validator scoring or weight setting.
        """
        t0 = time.time()

        try:
            # For organic: user's prompt leads, structured params supplement
            enhanced_prompt = self._prompt_parser.build_prompt(
                text=synapse.prompt,
                genre=synapse.genre,
                mood=synapse.mood,
                tempo=synapse.tempo_bpm,
                instruments=synapse.instruments,
                key=synapse.key_signature,
                time_sig=synapse.time_signature,
            )
            logger.info(
                f"[ORGANIC] {synapse.challenge_id}: "
                f"prompt='{enhanced_prompt[:100]}', "
                f"duration={synapse.duration_seconds}s"
            )

            audio, sr, wav_bytes, elapsed_ms = self._generate_audio(
                enhanced_prompt, synapse, t0
            )

            synapse.audio_b64 = self._audio_utils.to_base64(wav_bytes)
            synapse.sample_rate = sr
            synapse.generation_time_ms = elapsed_ms
            synapse.model_id = self._model_manager.active_model_id

            logger.info(
                f"[ORGANIC] {synapse.challenge_id} complete: "
                f"{elapsed_ms}ms, {len(wav_bytes)} bytes"
            )

        except Exception as exc:
            elapsed_ms = int((time.time() - t0) * 1000)
            logger.error(f"[ORGANIC] Failed {synapse.challenge_id}: {exc}")
            synapse.generation_time_ms = elapsed_ms
            self._record_error()

        return synapse

    def _generate_audio(
        self, prompt: str, synapse: MusicGenerationSynapse, t0: float
    ) -> Tuple:
        """Shared audio generation logic for both challenge and organic paths.

        Returns:
            Tuple of (audio_array, sample_rate, wav_bytes, elapsed_ms).
        """
        duration = min(
            synapse.duration_seconds,
            float(self.settings.generation_max_duration),
        )

        # Handle lyrics/vocals — only for backends that support them.
        # IMPORTANT: Only generate vocals when EXPLICITLY requested by the
        # caller. Never auto-detect vocal intent — GPT-2 is unreliable as a
        # classifier and generates incoherent lyrics that hurt prompt
        # adherence and vocal quality scores.
        lyrics = None
        vocals_requested = synapse.vocals_requested
        if self._model_manager.supports_vocals and vocals_requested and self._lyrics_gen:
            from tuneforge.generation.lyrics_generator import extract_genre, extract_mood

            lyrics = synapse.lyrics
            if not lyrics:
                # Vocals explicitly requested but no lyrics provided — generate
                logger.info("Generating lyrics from prompt (vocals explicitly requested)")
                genre = extract_genre(prompt) or synapse.genre
                mood = extract_mood(prompt) or synapse.mood
                lyrics = self._lyrics_gen.generate(
                    prompt=prompt,
                    genre=genre,
                    mood=mood,
                    duration_seconds=duration,
                )
                logger.info(f"Generated {len(lyrics.splitlines())} lines of lyrics")
        elif not vocals_requested:
            logger.debug("Vocals not requested — generating instrumental")

        audio, sr = self._model_manager.generate(
            prompt=prompt,
            duration=duration,
            seed=synapse.seed,
            guidance_scale=self.settings.guidance_scale,
            temperature=self.settings.temperature,
            top_k=self.settings.top_k,
            top_p=self.settings.top_p,
            lyrics=lyrics,
            # Forward structured params so backends (HeartMuLa, DiffRhythm)
            # can use them directly instead of lossy keyword extraction.
            genre=synapse.genre,
            mood=synapse.mood,
            tempo_bpm=synapse.tempo_bpm,
            key_signature=synapse.key_signature,
            instruments=synapse.instruments,
        )

        audio = self._audio_utils.normalize(audio)
        audio = self._audio_utils.apply_limiter(audio, threshold=0.95)
        audio = self._audio_utils.fade_edges(audio, sr, fade_ms=50)

        wav_bytes = self._audio_utils.to_wav_bytes(audio, sr)
        elapsed_ms = int((time.time() - t0) * 1000)

        return audio, sr, wav_bytes, elapsed_ms

    # ------------------------------------------------------------------
    # Blacklisting
    # ------------------------------------------------------------------

    def blacklist_generation(
        self, synapse: MusicGenerationSynapse
    ) -> Tuple[bool, str]:
        """Determine if a generation request should be blacklisted.

        Requires the caller to have a validator permit and at least
        ``MIN_GENERATION_STAKE`` alpha stake.

        Args:
            synapse: Incoming generation request.

        Returns:
            Tuple of (should_blacklist, reason).
        """
        caller_hotkey = synapse.dendrite.hotkey
        if not caller_hotkey:
            return True, "No hotkey provided"

        whitelisted = self.get_whitelisted_hotkeys()
        if caller_hotkey not in whitelisted:
            logger.trace(f"Blacklisting generation from {caller_hotkey[:16]}… (not whitelisted)")
            return True, "Not a whitelisted validator"

        logger.trace(f"Allowing generation from whitelisted {caller_hotkey[:16]}…")
        return False, "Whitelisted validator"

    # ------------------------------------------------------------------
    # Priority
    # ------------------------------------------------------------------

    def priority_generation(self, synapse: MusicGenerationSynapse) -> float:
        """Determine request priority.

        Organic (product) requests get a large priority boost over
        validator challenges so paying customers are served first.

        Args:
            synapse: Incoming generation request.

        Returns:
            Priority value.
        """
        # Organic queries from the product API get top priority
        organic_boost = 1_000_000.0 if synapse.is_organic else 0.0

        caller_hotkey = synapse.dendrite.hotkey
        if not caller_hotkey:
            return organic_boost
        try:
            caller_uid = self.metagraph.hotkeys.index(caller_hotkey)
            return organic_boost + float(self.metagraph.S[caller_uid])
        except (ValueError, IndexError):
            return 0.0

    # ------------------------------------------------------------------
    # Ping handler
    # ------------------------------------------------------------------

    def forward_ping(self, synapse: PingSynapse) -> PingSynapse:
        """Report miner availability and capabilities.

        Args:
            synapse: Incoming ping request.

        Returns:
            Synapse populated with capability information.
        """
        from tuneforge import VERSION

        logger.debug("Received ping request")

        synapse.is_available = self.is_running and not self.should_exit
        synapse.version = VERSION
        synapse.supported_genres = [
            "pop", "rock", "classical", "jazz", "electronic", "ambient",
            "hip-hop", "lo-fi", "r&b", "metal", "folk", "country",
            "edm", "house", "techno", "trap", "synthwave", "cinematic",
        ]
        synapse.supported_durations = [5.0, 10.0, 15.0, 20.0, 30.0]
        synapse.max_concurrent = 1

        gpu_info = self._model_manager.get_gpu_info()
        synapse.gpu_model = gpu_info.get("gpu_model", "N/A")

        return synapse

    # ------------------------------------------------------------------
    # Health handler
    # ------------------------------------------------------------------

    def forward_health(self, synapse: HealthReportSynapse) -> HealthReportSynapse:
        """Report miner health and performance metrics.

        Args:
            synapse: Incoming health report request.

        Returns:
            Synapse populated with health metrics.
        """
        logger.debug("Received health report request")

        # GPU metrics
        gpu_info = self._model_manager.get_gpu_info()
        synapse.gpu_utilization = gpu_info.get("utilization_percent", 0.0)
        synapse.gpu_memory_used_mb = gpu_info.get("memory_allocated_mb", 0.0)

        # System metrics
        try:
            synapse.cpu_percent = psutil.cpu_percent(interval=0.1)
            synapse.memory_percent = psutil.virtual_memory().percent
        except Exception as exc:
            logger.warning(f"Failed to get system metrics: {exc}")

        # Generation metrics
        synapse.generations_completed = self._generations_completed
        if self._generation_times:
            synapse.average_generation_time_ms = sum(self._generation_times) / len(
                self._generation_times
            )

        # Uptime
        if hasattr(self, "_start_time"):
            synapse.uptime_seconds = time.time() - self._start_time

        # Error count (prune old errors)
        self._prune_old_errors()
        synapse.errors_last_hour = self._errors_last_hour

        return synapse

    # ------------------------------------------------------------------
    # Forward dispatch (required by BaseNeuron)
    # ------------------------------------------------------------------

    def forward(self, synapse: MusicGenerationSynapse) -> MusicGenerationSynapse:
        """Dispatch to forward_generation (satisfies BaseNeuron abstract)."""
        return self.forward_generation(synapse)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_error(self) -> None:
        """Record an error timestamp for hourly error counting."""
        now = time.time()
        self._error_timestamps.append(now)
        self._prune_old_errors()

    def _prune_old_errors(self) -> None:
        """Remove error timestamps older than 1 hour."""
        cutoff = time.time() - 3600
        self._error_timestamps = [
            ts for ts in self._error_timestamps if ts > cutoff
        ]
        self._errors_last_hour = len(self._error_timestamps)

    def shutdown(self) -> None:
        """Shut down the miner and unload models."""
        logger.info("Shutting down TuneForgeMiner…")
        try:
            self._model_manager.unload_all()
            if self._lyrics_gen is not None:
                self._lyrics_gen.unload()
        except Exception as exc:
            logger.error(f"Error unloading models: {exc}")
        super().shutdown()
