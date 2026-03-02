"""
Shared test fixtures for TuneForge test suite.

Provides mock infrastructure, sample audio data, and configuration
fixtures for all test modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.mock_subtensor import (
    MockDendrite,
    MockMetagraph,
    MockSubtensor,
    MockWallet,
)


# ---------------------------------------------------------------------------
# Infrastructure fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_wallet() -> MockWallet:
    return MockWallet()


@pytest.fixture
def mock_subtensor() -> MockSubtensor:
    return MockSubtensor(n_neurons=8)


@pytest.fixture
def mock_metagraph() -> MockMetagraph:
    return MockMetagraph(
        n=8,
        stakes=[10_000.0, 8_000.0, 5_000.0, 2_000.0, 500.0, 100.0, 50.0, 10.0],
        validator_permits=[True, True, True, True, False, False, False, False],
    )


@pytest.fixture
def mock_dendrite() -> MockDendrite:
    return MockDendrite()


# ---------------------------------------------------------------------------
# Audio fixtures — generated programmatically via numpy
# ---------------------------------------------------------------------------

SAMPLE_RATE = 32_000
DURATION = 10.0  # seconds


@pytest.fixture
def sample_rate() -> int:
    return SAMPLE_RATE


@pytest.fixture
def sample_audio_sine() -> np.ndarray:
    """10-second 440 Hz sine wave at 32 kHz."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = 0.8 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_complex() -> np.ndarray:
    """10-second multi-frequency audio with harmonics — approximates real music."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    # Fundamental + harmonics
    audio = (
        0.4 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
        + 0.1 * np.sin(2 * np.pi * 1320 * t)
        + 0.05 * np.sin(2 * np.pi * 1760 * t)
    )
    # Add slow amplitude modulation (tremolo)
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    audio *= modulation
    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_noise() -> np.ndarray:
    """10-second white noise at 32 kHz."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.5, 0.5, int(SAMPLE_RATE * DURATION)).astype(np.float32)


@pytest.fixture
def sample_audio_silence() -> np.ndarray:
    """10-second silence at 32 kHz."""
    return np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.float32)


@pytest.fixture
def sample_audio_clipped() -> np.ndarray:
    """10-second heavily clipped sine wave."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = 2.0 * np.sin(2 * np.pi * 440 * t)  # Intentionally > 1.0
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Helper for WAV encoding in tests
# ---------------------------------------------------------------------------

@pytest.fixture
def wav_from_audio():
    """Factory fixture that creates WAV bytes from audio array."""
    from tuneforge.generation.audio_utils import AudioUtils

    def _make(audio: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
        return AudioUtils.to_wav_bytes(audio, sr)

    return _make
