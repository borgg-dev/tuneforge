"""
TuneForgeMiner — full miner implementation for TuneForge subnet.

Handles music generation requests from validators, processes prompts,
generates audio via configurable backends, and reports health metrics.
"""

import time

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
MIN_GENERATION_STAKE: float = 1000.0


class TuneForgeMiner(BaseMinerNeuron):
    """Full miner implementation for the TuneForge music generation subnet.

    Loads a music generation model, responds to validator challenges
    with generated audio, and reports capability/health information.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the TuneForgeMiner.

        Args:
            settings: Optional settings override.
        """
        settings = settings or get_settings()
        super().__init__(settings=settings)

        # Parse model size from model_name (e.g. "facebook/musicgen-medium" → "medium")
        model_name = self.settings.model_name
        if "musicgen" in model_name:
            parts = model_name.split("-")
            model_size = parts[-1] if parts else "medium"
            backend = "musicgen"
        elif "stable-audio" in model_name or "stable_audio" in model_name:
            model_size = "medium"
            backend = "stable_audio"
        else:
            model_size = "medium"
            backend = "musicgen"

        self._model_manager = ModelManager(
            default_backend=backend,
            model_size=model_size,
            device=self.settings.gpu_device,
        )
        self._audio_utils = AudioUtils()
        self._prompt_parser = PromptParser()

        # Generation tracking
        self._generations_completed: int = 0
        self._errors_last_hour: int = 0
        self._error_timestamps: list[float] = []
        self._generation_times: list[float] = []

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
        """Generate music from a validator's challenge synapse.

        Builds an enhanced prompt from structured fields, generates audio
        via the model manager, post-processes the output, and encodes it
        to base64 for transmission back to the validator.

        Args:
            synapse: Incoming generation request with prompt and parameters.

        Returns:
            Synapse populated with generated audio and metadata.
        """
        t0 = time.time()

        try:
            # Build enhanced prompt from structured fields
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
                f"Challenge {synapse.challenge_id}: "
                f"prompt='{enhanced_prompt[:100]}', "
                f"duration={synapse.duration_seconds}s"
            )

            # Clamp duration to configured maximum
            duration = min(
                synapse.duration_seconds,
                float(self.settings.generation_max_duration),
            )

            # Generate audio
            audio, sr = self._model_manager.generate(
                prompt=enhanced_prompt,
                duration=duration,
                seed=synapse.seed,
            )

            # Post-process: normalize, limit, fade
            audio = self._audio_utils.normalize(audio)
            audio = self._audio_utils.apply_limiter(audio, threshold=0.95)
            audio = self._audio_utils.fade_edges(audio, sr, fade_ms=50)

            # Encode to base64 WAV
            wav_bytes = self._audio_utils.to_wav_bytes(audio, sr)
            audio_b64 = self._audio_utils.to_base64(wav_bytes)

            # Fill response fields
            elapsed_ms = int((time.time() - t0) * 1000)
            synapse.audio_b64 = audio_b64
            synapse.sample_rate = sr
            synapse.generation_time_ms = elapsed_ms
            synapse.model_id = self._model_manager.active_model_id

            self._generations_completed += 1
            self._generation_times.append(elapsed_ms)
            # Keep only last 100 times for rolling average
            if len(self._generation_times) > 100:
                self._generation_times = self._generation_times[-100:]

            logger.info(
                f"Challenge {synapse.challenge_id} complete: "
                f"{elapsed_ms}ms, {len(wav_bytes)} bytes"
            )

        except Exception as exc:
            elapsed_ms = int((time.time() - t0) * 1000)
            logger.error(f"Generation failed for challenge {synapse.challenge_id}: {exc}")
            synapse.generation_time_ms = elapsed_ms
            self._record_error()

        return synapse

    # ------------------------------------------------------------------
    # Blacklisting
    # ------------------------------------------------------------------

    async def blacklist_generation(
        self, synapse: MusicGenerationSynapse
    ) -> tuple[bool, str]:
        """Determine if a generation request should be blacklisted.

        Requires the caller to have a validator permit and minimum stake.

        Args:
            synapse: Incoming generation request.

        Returns:
            Tuple of (should_blacklist, reason).
        """
        caller_hotkey = synapse.dendrite.hotkey
        if not caller_hotkey:
            return True, "No hotkey provided"

        try:
            caller_uid = self.metagraph.hotkeys.index(caller_hotkey)
        except ValueError:
            return True, f"Hotkey {caller_hotkey[:16]}… not registered"

        if not self.metagraph.validator_permit[caller_uid]:
            return True, f"UID {caller_uid} has no validator permit"

        stake = float(self.metagraph.S[caller_uid])
        if stake < MIN_GENERATION_STAKE:
            return True, f"UID {caller_uid} stake {stake:.0f} < {MIN_GENERATION_STAKE}"

        return False, f"Allowed: UID {caller_uid}, stake {stake:.0f}"

    # ------------------------------------------------------------------
    # Priority
    # ------------------------------------------------------------------

    def priority_generation(self, synapse: MusicGenerationSynapse) -> float:
        """Determine request priority based on validator stake.

        Higher-staked validators get higher priority in the request queue.

        Args:
            synapse: Incoming generation request.

        Returns:
            Priority value (validator's stake).
        """
        caller_hotkey = synapse.dendrite.hotkey
        if not caller_hotkey:
            return 0.0
        try:
            caller_uid = self.metagraph.hotkeys.index(caller_hotkey)
            return float(self.metagraph.S[caller_uid])
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
        except Exception as exc:
            logger.error(f"Error unloading models: {exc}")
        super().shutdown()
