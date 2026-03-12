"""Tests for TuneForge Synapse protocol definitions."""

import base64

import numpy as np
import pytest

from tuneforge.base.protocol import (
    HealthReportSynapse,
    MusicGenerationSynapse,
    PingSynapse,
)


class TestMusicGenerationSynapse:
    """MusicGenerationSynapse creation, serialization, deserialization."""

    def test_creation_with_defaults(self):
        syn = MusicGenerationSynapse()
        assert syn.prompt == ""
        assert syn.genre == ""
        assert syn.mood == ""
        assert syn.tempo_bpm == 120
        assert syn.duration_seconds == 10.0
        assert syn.audio_b64 is None
        assert syn.sample_rate is None
        assert syn.generation_time_ms is None
        assert syn.model_id is None
        assert syn.key_signature is None
        assert syn.instruments is None
        assert syn.seed is None
        assert syn.challenge_id == ""

    def test_creation_with_full_fields(self):
        syn = MusicGenerationSynapse(
            prompt="lo-fi chill beat",
            genre="lo-fi",
            mood="calm",
            tempo_bpm=80,
            duration_seconds=15.0,
            key_signature="A minor",
            time_signature="4/4",
            instruments=["piano", "vinyl crackle"],
            seed=42,
            challenge_id="test-challenge-001",
        )
        assert syn.prompt == "lo-fi chill beat"
        assert syn.genre == "lo-fi"
        assert syn.mood == "calm"
        assert syn.tempo_bpm == 80
        assert syn.duration_seconds == 15.0
        assert syn.key_signature == "A minor"
        assert syn.time_signature == "4/4"
        assert syn.instruments == ["piano", "vinyl crackle"]
        assert syn.seed == 42
        assert syn.challenge_id == "test-challenge-001"

    def test_required_hash_fields(self):
        expected = ("prompt", "genre", "mood", "tempo_bpm", "duration_seconds", "challenge_id")
        assert MusicGenerationSynapse.required_hash_fields_ == expected

    def test_deserialize_with_audio(self):
        raw = b"fake_audio_data_for_testing"
        encoded = base64.b64encode(raw).decode()
        syn = MusicGenerationSynapse(audio_b64=encoded)
        result = syn.deserialize()
        assert result == raw

    def test_deserialize_without_audio(self):
        syn = MusicGenerationSynapse()
        assert syn.deserialize() is None

    def test_deserialize_invalid_base64(self):
        syn = MusicGenerationSynapse(audio_b64="not-valid-base64!!!")
        assert syn.deserialize() is None

    def test_response_fields(self):
        syn = MusicGenerationSynapse(
            audio_b64="dGVzdA==",
            sample_rate=32000,
            generation_time_ms=5000,
            model_id="facebook/musicgen-medium",
        )
        assert syn.sample_rate == 32000
        assert syn.generation_time_ms == 5000
        assert syn.model_id == "facebook/musicgen-medium"

    def test_tempo_bounds(self):
        syn = MusicGenerationSynapse(tempo_bpm=60)
        assert syn.tempo_bpm == 60

        syn = MusicGenerationSynapse(tempo_bpm=200)
        assert syn.tempo_bpm == 200


class TestPingSynapse:
    """PingSynapse creation and field verification."""

    def test_creation_defaults(self):
        syn = PingSynapse()
        assert syn.version_check is None
        assert syn.is_available is False
        assert syn.supported_genres is None
        assert syn.gpu_model is None
        assert syn.max_concurrent is None
        assert syn.version is None

    def test_response_fields(self):
        syn = PingSynapse(
            is_available=True,
            supported_genres=["rock", "jazz"],
            supported_durations=[10.0, 20.0],
            gpu_model="NVIDIA A100",
            max_concurrent=2,
            version="1.0.0",
        )
        assert syn.is_available is True
        assert "rock" in syn.supported_genres
        assert syn.gpu_model == "NVIDIA A100"
        assert syn.max_concurrent == 2
        assert syn.version == "1.0.0"


class TestHealthReportSynapse:
    """HealthReportSynapse creation and field verification."""

    def test_creation_defaults(self):
        syn = HealthReportSynapse()
        assert syn.gpu_utilization == 0.0
        assert syn.gpu_memory_used_mb == 0.0
        assert syn.cpu_percent == 0.0
        assert syn.memory_percent == 0.0
        assert syn.generations_completed == 0
        assert syn.average_generation_time_ms == 0.0
        assert syn.uptime_seconds == 0.0
        assert syn.errors_last_hour == 0

    def test_populated_fields(self):
        syn = HealthReportSynapse(
            gpu_utilization=85.0,
            gpu_memory_used_mb=12000.0,
            cpu_percent=45.0,
            memory_percent=60.0,
            generations_completed=100,
            average_generation_time_ms=3500.0,
            uptime_seconds=86400.0,
            errors_last_hour=2,
        )
        assert syn.gpu_utilization == 85.0
        assert syn.generations_completed == 100
        assert syn.errors_last_hour == 2
