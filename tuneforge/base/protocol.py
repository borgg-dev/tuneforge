"""
Protocol definitions for TuneForge subnet communication.

Defines Synapse classes for validator-miner communication:
- MusicGenerationSynapse: Music generation requests and responses
- PingSynapse: Availability and capability queries
- HealthReportSynapse: Miner health metric reporting
"""

import base64
from typing import ClassVar

import bittensor as bt
from pydantic import Field


class MusicGenerationSynapse(bt.Synapse):
    """
    Synapse for requesting AI music generation from miners.

    Validators send prompts with musical parameters; miners return
    generated audio as base64-encoded WAV/MP3 data.
    """

    # --- Request fields (validator → miner) ---

    prompt: str = Field(
        default="",
        description="Text prompt describing desired music",
    )
    genre: str = Field(
        default="",
        description="Target genre (pop, rock, classical, jazz, electronic, ambient, …)",
    )
    mood: str = Field(
        default="",
        description="Target mood (happy, sad, energetic, calm, dark, uplifting, …)",
    )
    tempo_bpm: int = Field(
        default=120,
        ge=20,
        le=300,
        description="Desired tempo in BPM",
    )
    duration_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Desired audio duration in seconds",
    )
    key_signature: str | None = Field(
        default=None,
        description="Musical key signature (C major, A minor, …)",
    )
    time_signature: str | None = Field(
        default=None,
        description="Time signature (4/4, 3/4, 6/8, …)",
    )
    instruments: list[str] | None = Field(
        default=None,
        description="Preferred instruments (piano, guitar, drums, …)",
    )
    reference_audio: str | None = Field(
        default=None,
        description="Base64-encoded reference audio for style transfer",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducible generation",
    )
    challenge_id: str = Field(
        default="",
        description="Unique challenge ID for this validation round",
    )

    # --- Response fields (miner → validator) ---

    audio_b64: str | None = Field(
        default=None,
        description="Generated audio data encoded as base64",
    )
    sample_rate: int | None = Field(
        default=None,
        description="Sample rate of generated audio in Hz",
    )
    generation_time_ms: int | None = Field(
        default=None,
        description="Time taken to generate audio in milliseconds",
    )
    model_id: str | None = Field(
        default=None,
        description="Identifier of the model used for generation",
    )

    required_hash_fields_: ClassVar[tuple[str, ...]] = (
        "prompt",
        "genre",
        "mood",
        "tempo_bpm",
        "duration_seconds",
        "challenge_id",
    )

    def deserialize(self) -> bytes | None:
        """Decode base64 audio data into raw bytes.

        Returns:
            Raw audio bytes, or None if no audio was generated.
        """
        if self.audio_b64 is None:
            return None
        try:
            return base64.b64decode(self.audio_b64)
        except Exception:
            return None


class PingSynapse(bt.Synapse):
    """
    Synapse for querying miner availability and capabilities.

    Validators use this to discover online miners and their
    supported generation parameters before sending challenges.
    """

    # --- Request field (validator → miner) ---

    version_check: str | None = Field(
        default=None,
        description="Validator protocol version for compatibility check",
    )

    # --- Response fields (miner → validator) ---

    is_available: bool = Field(
        default=False,
        description="Whether miner is available for generation requests",
    )
    supported_genres: list[str] | None = Field(
        default=None,
        description="List of genres the miner's model supports",
    )
    supported_durations: list[float] | None = Field(
        default=None,
        description="List of supported generation durations in seconds",
    )
    gpu_model: str | None = Field(
        default=None,
        description="GPU model name (e.g. NVIDIA A100)",
    )
    max_concurrent: int | None = Field(
        default=None,
        description="Maximum concurrent generation requests supported",
    )
    version: str | None = Field(
        default=None,
        description="Miner software version",
    )


class HealthReportSynapse(bt.Synapse):
    """
    Synapse for miners to report hardware and performance metrics.

    Validators collect these to monitor miner health and factor
    reliability into scoring.
    """

    # --- Response fields (miner → validator) ---

    gpu_utilization: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="GPU utilization percentage",
    )
    gpu_memory_used_mb: float = Field(
        default=0.0,
        ge=0.0,
        description="GPU memory used in megabytes",
    )
    cpu_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="CPU utilization percentage",
    )
    memory_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="System memory utilization percentage",
    )
    generations_completed: int = Field(
        default=0,
        ge=0,
        description="Total music generations completed since startup",
    )
    average_generation_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Rolling average generation time in milliseconds",
    )
    uptime_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Miner uptime in seconds",
    )
    errors_last_hour: int = Field(
        default=0,
        ge=0,
        description="Number of errors encountered in the last hour",
    )
