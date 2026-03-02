"""
Audio processing utilities for TuneForge subnet.

Provides normalization, encoding, analysis, and format conversion
for generated audio data.
"""

import base64
import io
import struct

import numpy as np
from loguru import logger


class AudioUtils:
    """Static utility methods for audio processing."""

    @staticmethod
    def normalize(audio: np.ndarray) -> np.ndarray:
        """Scale audio to [-1, 1] range.

        Args:
            audio: Input audio array.

        Returns:
            Normalized audio array with peak amplitude at 1.0.
        """
        peak = np.max(np.abs(audio))
        if peak < 1e-8:
            return audio
        return audio / peak

    @staticmethod
    def fade_edges(
        audio: np.ndarray, sr: int, fade_ms: int = 50
    ) -> np.ndarray:
        """Apply cosine fade-in and fade-out to audio edges.

        Args:
            audio: Input audio array.
            sr: Sample rate in Hz.
            fade_ms: Fade duration in milliseconds.

        Returns:
            Audio with faded edges.
        """
        fade_samples = int(sr * fade_ms / 1000)
        if fade_samples <= 0 or len(audio) < fade_samples * 2:
            return audio

        result = audio.copy()
        # Cosine fade-in
        fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samples)))
        result[:fade_samples] *= fade_in
        # Cosine fade-out
        fade_out = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_samples)))
        result[-fade_samples:] *= fade_out
        return result

    @staticmethod
    def to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
        """Convert audio array to 16-bit PCM WAV bytes.

        Uses soundfile if available, falls back to manual WAV construction.

        Args:
            audio: Audio array with values in [-1, 1].
            sr: Sample rate in Hz.

        Returns:
            WAV file as bytes.
        """
        try:
            import soundfile as sf

            buf = io.BytesIO()
            sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
            return buf.getvalue()
        except ImportError:
            logger.debug("soundfile not available, using manual WAV encoder")
            return AudioUtils._manual_wav_encode(audio, sr)

    @staticmethod
    def _manual_wav_encode(audio: np.ndarray, sr: int) -> bytes:
        """Manually encode audio to 16-bit PCM WAV format.

        Args:
            audio: Audio array with values in [-1, 1].
            sr: Sample rate in Hz.

        Returns:
            WAV file as bytes.
        """
        if audio.ndim > 1:
            audio = audio.flatten()

        clipped = np.clip(audio, -1.0, 1.0)
        int_data = (clipped * 32767).astype(np.int16)
        raw_bytes = int_data.tobytes()

        num_channels = 1
        bits_per_sample = 16
        byte_rate = sr * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(raw_bytes)

        buf = io.BytesIO()
        # RIFF header
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + data_size))
        buf.write(b"WAVE")
        # fmt chunk
        buf.write(b"fmt ")
        buf.write(struct.pack("<I", 16))
        buf.write(struct.pack("<HHIIHH", 1, num_channels, sr, byte_rate, block_align, bits_per_sample))
        # data chunk
        buf.write(b"data")
        buf.write(struct.pack("<I", data_size))
        buf.write(raw_bytes)
        return buf.getvalue()

    @staticmethod
    def from_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
        """Decode WAV bytes to audio array and sample rate.

        Args:
            wav_bytes: WAV file as bytes.

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        try:
            import soundfile as sf

            buf = io.BytesIO(wav_bytes)
            audio, sr = sf.read(buf, dtype="float32")
            return audio, sr
        except ImportError:
            logger.debug("soundfile not available, using manual WAV decoder")
            return AudioUtils._manual_wav_decode(wav_bytes)

    @staticmethod
    def _manual_wav_decode(wav_bytes: bytes) -> tuple[np.ndarray, int]:
        """Manually decode 16-bit PCM WAV bytes.

        Args:
            wav_bytes: WAV file as bytes.

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        buf = io.BytesIO(wav_bytes)
        riff = buf.read(4)
        if riff != b"RIFF":
            raise ValueError("Not a valid WAV file")
        buf.read(4)  # file size
        wave = buf.read(4)
        if wave != b"WAVE":
            raise ValueError("Not a valid WAV file")

        sr = 44100
        audio_data = b""

        while True:
            chunk_id = buf.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", buf.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = buf.read(chunk_size)
                sr = struct.unpack("<I", fmt_data[4:8])[0]
            elif chunk_id == b"data":
                audio_data = buf.read(chunk_size)
            else:
                buf.read(chunk_size)

        int_data = np.frombuffer(audio_data, dtype=np.int16)
        audio = int_data.astype(np.float32) / 32767.0
        return audio, sr

    @staticmethod
    def to_base64(wav_bytes: bytes) -> str:
        """Encode WAV bytes to base64 string.

        Args:
            wav_bytes: WAV file as bytes.

        Returns:
            Base64-encoded string.
        """
        return base64.b64encode(wav_bytes).decode("utf-8")

    @staticmethod
    def from_base64(b64_str: str) -> bytes:
        """Decode base64 string to raw bytes.

        Args:
            b64_str: Base64-encoded string.

        Returns:
            Decoded bytes.
        """
        return base64.b64decode(b64_str)

    @staticmethod
    def compute_duration(audio: np.ndarray, sr: int) -> float:
        """Compute audio duration in seconds.

        Args:
            audio: Audio array.
            sr: Sample rate in Hz.

        Returns:
            Duration in seconds.
        """
        return len(audio) / sr

    @staticmethod
    def compute_rms(audio: np.ndarray) -> float:
        """Compute root-mean-square amplitude.

        Args:
            audio: Audio array.

        Returns:
            RMS value.
        """
        return float(np.sqrt(np.mean(audio ** 2)))

    @staticmethod
    def compute_peak(audio: np.ndarray) -> float:
        """Compute peak absolute amplitude.

        Args:
            audio: Audio array.

        Returns:
            Peak amplitude.
        """
        return float(np.max(np.abs(audio)))

    @staticmethod
    def apply_limiter(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """Apply soft limiting to prevent clipping.

        Uses tanh-based soft clipping that smoothly compresses
        amplitudes above the threshold.

        Args:
            audio: Input audio array.
            threshold: Amplitude threshold for limiting (0-1).

        Returns:
            Limited audio array.
        """
        if threshold <= 0:
            return np.zeros_like(audio)

        result = audio.copy()
        mask = np.abs(result) > threshold
        if not np.any(mask):
            return result

        # Soft clip using tanh for smooth compression
        over = result[mask]
        sign = np.sign(over)
        magnitude = np.abs(over)
        compressed = threshold + (1.0 - threshold) * np.tanh(
            (magnitude - threshold) / (1.0 - threshold)
        )
        result[mask] = sign * compressed
        return result

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to a different sample rate.

        Uses linear interpolation for simplicity. For higher quality,
        install librosa and use its resampler.

        Args:
            audio: Input audio array.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled audio array.
        """
        if orig_sr == target_sr:
            return audio

        try:
            import librosa

            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Linear interpolation fallback
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(audio.dtype)
