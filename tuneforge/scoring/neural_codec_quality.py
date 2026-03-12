"""
Neural codec quality scorer for TuneForge.

Uses Facebook EnCodec to encode/decode audio and measures reconstruction
error as a quality signal. Natural, well-formed audio has low
reconstruction error.
"""

import numpy as np
from loguru import logger


class NeuralCodecQualityScorer:
    """Assess audio quality via neural codec reconstruction error."""

    WEIGHTS: dict[str, float] = {
        "reconstruction_quality": 0.60,
        "codec_naturalness": 0.40,
    }

    _LOAD_FAILED = "LOAD_FAILED"

    def __init__(self, model_name: str = "facebook/encodec_24khz") -> None:
        self._model_name = model_name
        self._model = None
        self._processor = None

    def _load(self) -> bool:
        """Lazy-load EnCodec model. Returns True on success."""
        if self._model == self._LOAD_FAILED:
            return False
        if self._model is not None:
            return True
        try:
            from transformers import EncodecModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._model_name, use_fast=True)
            self._model = EncodecModel.from_pretrained(self._model_name)
            self._model.eval()
            logger.info("Loaded EnCodec model: {}", self._model_name)
            return True
        except Exception as exc:
            logger.warning("Failed to load EnCodec model (permanent): {}", exc)
            self._model = self._LOAD_FAILED
            return False

    def score(self, audio: np.ndarray, sr: int, genre: str = "") -> dict[str, float]:
        """Compute neural codec quality scores."""
        try:
            if not self._load():
                return {k: 0.5 for k in self.WEIGHTS}

            import torch
            import librosa

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            # Resample to 24kHz (EnCodec native rate)
            target_sr = 24000
            if sr != target_sr:
                audio_24k = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            else:
                audio_24k = audio

            # Limit to 30s to avoid OOM
            max_samples = target_sr * 30
            if len(audio_24k) > max_samples:
                audio_24k = audio_24k[:max_samples]

            # Encode and decode
            inputs = self._processor(
                raw_audio=audio_24k,
                sampling_rate=target_sr,
                return_tensors="pt",
            )

            with torch.no_grad():
                encoder_outputs = self._model.encode(
                    inputs["input_values"],
                    inputs.get("padding_mask"),
                )
                decoded = self._model.decode(
                    encoder_outputs.audio_codes,
                    encoder_outputs.audio_scales,
                    inputs.get("padding_mask"),
                )

            reconstructed = decoded.audio_values.squeeze().numpy()

            # Align lengths
            min_len = min(len(audio_24k), len(reconstructed))
            original = audio_24k[:min_len]
            recon = reconstructed[:min_len]

            return {
                "reconstruction_quality": self._spectral_convergence(original, recon, target_sr),
                "codec_naturalness": self._mel_distance_score(original, recon, target_sr),
            }

        except Exception as exc:
            logger.error("Neural codec quality scoring failed: {}", exc)
            return {k: 0.5 for k in self.WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        total = sum(self.WEIGHTS[k] * scores.get(k, 0.0) for k in self.WEIGHTS)
        return float(np.clip(total, 0.0, 1.0))

    @staticmethod
    def _spectral_convergence(original: np.ndarray, recon: np.ndarray, sr: int) -> float:
        """Multi-resolution STFT spectral convergence. Lower error = higher score."""
        import librosa

        total_error = 0.0
        n_ffts = [512, 1024, 2048]
        for n_fft in n_ffts:
            S_orig = np.abs(librosa.stft(original, n_fft=n_fft))
            S_recon = np.abs(librosa.stft(recon, n_fft=n_fft))
            norm_orig = np.linalg.norm(S_orig) + 1e-8
            error = np.linalg.norm(S_orig - S_recon) / norm_orig
            total_error += error

        avg_error = total_error / len(n_ffts)
        # Map error to score: 0.0 error -> 1.0, 0.5 error -> ~0.0
        return float(np.clip(1.0 - avg_error * 2.0, 0.0, 1.0))

    @staticmethod
    def _mel_distance_score(original: np.ndarray, recon: np.ndarray, sr: int) -> float:
        """Log mel spectrogram L1 distance. Lower = better."""
        import librosa

        S_orig = librosa.feature.melspectrogram(y=original, sr=sr, n_mels=80)
        S_recon = librosa.feature.melspectrogram(y=recon, sr=sr, n_mels=80)

        log_orig = np.log1p(S_orig)
        log_recon = np.log1p(S_recon)

        min_frames = min(log_orig.shape[1], log_recon.shape[1])
        distance = float(np.mean(np.abs(log_orig[:, :min_frames] - log_recon[:, :min_frames])))

        # Map distance to score: 0.0 -> 1.0, 2.0 -> ~0.0
        return float(np.clip(1.0 - distance / 2.0, 0.0, 1.0))
