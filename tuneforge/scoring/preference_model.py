"""
Preference model for TuneForge.

A lightweight MLP on top of CLAP audio embeddings that predicts
human preference.  Falls back to a multi-feature perceptual quality
heuristic when no trained model checkpoint is available (bootstrap mode).

The bootstrap heuristic combines 8 audio features that research shows
correlate with human perception of audio quality: crest factor, spectral
centroid consistency, spectral rolloff position, zero-crossing rate
normality, MFCC diversity, harmonic clarity, temporal smoothness, and
tonal richness.
"""

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from pathlib import Path

from tuneforge.config.scoring_config import CLAP_MODEL
from tuneforge.scoring.clap_scorer import CLAPScorer


class PreferenceHead(nn.Module):
    """MLP: 512-dim CLAP embedding → scalar preference score."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PreferenceModel:
    """
    Predict human preference from audio.

    Uses CLAP audio embeddings fed through a trained MLP head.
    If no trained model path is given, runs in bootstrap mode using
    a multi-feature perceptual quality heuristic.
    """

    def __init__(self, model_path: str | None = None, neural_scorer=None) -> None:
        self._clap = CLAPScorer(model_name=CLAP_MODEL)
        self._neural = neural_scorer  # Shared NeuralQualityScorer for MERT features
        self._head: PreferenceHead | None = None
        self._bootstrap = True
        self._model_path = model_path

        if model_path is not None and Path(model_path).exists():
            try:
                self._head = PreferenceHead()
                state = torch.load(model_path, map_location="cpu", weights_only=True)
                self._head.load_state_dict(state)
                self._head.eval()
                if torch.cuda.is_available():
                    self._head = self._head.cuda()
                self._bootstrap = False
                logger.info(f"Preference model loaded from {model_path}")
            except Exception as exc:
                logger.warning(f"Failed to load preference head, using bootstrap: {exc}")
                self._head = None
                self._bootstrap = True
        else:
            logger.info("No preference model checkpoint — running in bootstrap mode")

    def score(self, audio: np.ndarray, sr: int) -> float:
        """
        Score audio for predicted human preference.

        Args:
            audio: 1-D float waveform.
            sr: Sample rate in Hz.

        Returns:
            Preference score in [0, 1].
        """
        if self._bootstrap:
            return self._heuristic_score(audio, sr)

        try:
            embedding = self._clap.get_audio_embedding(audio, sr)
            if embedding is None:
                return self._heuristic_score(audio, sr)

            tensor = torch.from_numpy(embedding).float().unsqueeze(0)
            if next(self._head.parameters()).is_cuda:
                tensor = tensor.cuda()

            with torch.no_grad():
                pred = self._head(tensor).item()

            return float(np.clip(pred, 0.0, 1.0))

        except Exception as exc:
            logger.error(f"Preference scoring failed: {exc}")
            return self._heuristic_score(audio, sr)

    # Weights for the 12-feature bootstrap heuristic (8 librosa + 4 MERT).
    _HEURISTIC_WEIGHTS = {
        "crest_factor": 0.08,
        "spectral_centroid_consistency": 0.10,
        "spectral_rolloff": 0.08,
        "zcr_normality": 0.06,
        "mfcc_diversity": 0.12,
        "harmonic_clarity": 0.12,
        "temporal_smoothness": 0.12,
        "tonal_richness": 0.12,
        # MERT-derived features (total 0.20)
        "mert_activation_quality": 0.06,
        "mert_temporal_smoothness": 0.06,
        "mert_representation_richness": 0.04,
        "mert_structural_quality": 0.04,
    }

    def _heuristic_score(self, audio: np.ndarray, sr: int) -> float:
        """
        Multi-feature perceptual quality heuristic (bootstrap mode).

        Used when no trained preference head is available.  Computes 8
        audio features that research shows correlate with human perception
        of audio quality and combines them with empirically tuned weights:

        1. Crest Factor Quality (0.10) -- bell curve centred at crest=6.0
        2. Spectral Centroid Consistency (0.12) -- CV bell at 0.15
        3. Spectral Rolloff Position (0.10) -- rolloff ratio bell at 0.45
        4. Zero-Crossing Rate Normality (0.08) -- penalise noise-like ZCR
        5. MFCC Diversity (0.15) -- timbral richness via MFCC std
        6. Harmonic Clarity (0.15) -- harmonic-to-total energy ratio
        7. Temporal Smoothness (0.15) -- low RMS-envelope acceleration
        8. Tonal Richness (0.15) -- chroma entropy at moderate level

        Returns a score in [0, 1].
        """
        try:
            import librosa

            # ----- pre-processing ------------------------------------------
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.001:
                return 0.0

            peak = float(np.max(np.abs(audio)))
            if peak < 1e-8:
                return 0.0

            features: dict[str, float] = {}

            # ----- Feature 1: Crest Factor Quality -------------------------
            crest = peak / rms
            features["crest_factor"] = float(
                np.clip(1.0 - abs(crest - 6.0) / 6.0, 0.0, 1.0)
            )

            # ----- Feature 2: Spectral Centroid Consistency ----------------
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            centroid_mean = float(np.mean(centroid))
            centroid_std = float(np.std(centroid))
            cv = centroid_std / (centroid_mean + 1e-8)
            features["spectral_centroid_consistency"] = float(
                np.clip(1.0 - abs(cv - 0.15) / 0.3, 0.0, 1.0)
            )

            # ----- Feature 3: Spectral Rolloff Position --------------------
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            nyquist = sr / 2.0
            ratio = float(np.mean(rolloff)) / nyquist
            features["spectral_rolloff"] = float(
                np.clip(1.0 - abs(ratio - 0.45) / 0.45, 0.0, 1.0)
            )

            # ----- Feature 4: Zero-Crossing Rate Normality ----------------
            zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
            mean_zcr = float(np.mean(zcr))
            zcr_score = 1.0 - min(mean_zcr / 0.2, 1.0)
            # Penalise pure low-freq drones (very low ZCR)
            if mean_zcr > 0.01:
                zcr_score = max(zcr_score, 0.2)
            features["zcr_normality"] = float(np.clip(zcr_score, 0.0, 1.0))

            # ----- Feature 5: MFCC Diversity -------------------------------
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_std_per_coeff = np.std(mfccs, axis=1)  # shape (13,)
            mean_mfcc_std = float(np.mean(mfcc_std_per_coeff))
            features["mfcc_diversity"] = float(
                np.clip(mean_mfcc_std / 15.0, 0.0, 1.0)
            )

            # ----- Feature 6: Harmonic Clarity -----------------------------
            harmonic = librosa.effects.harmonic(y=audio)
            harmonic_energy = float(np.sum(harmonic ** 2))
            total_energy = float(np.sum(audio ** 2)) + 1e-8
            h_ratio = harmonic_energy / total_energy
            features["harmonic_clarity"] = float(
                np.clip(1.0 - abs(h_ratio - 0.55) / 0.55, 0.0, 1.0)
            )

            # ----- Feature 7: Temporal Smoothness --------------------------
            rms_env = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            if len(rms_env) >= 3:
                accel = np.diff(rms_env, n=2)
                mean_abs_accel = float(np.mean(np.abs(accel)))
                features["temporal_smoothness"] = float(
                    np.clip(1.0 - min(mean_abs_accel / 0.01, 1.0), 0.0, 1.0)
                )
            else:
                features["temporal_smoothness"] = 0.5

            # ----- Feature 8: Tonal Richness -------------------------------
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            mean_chroma = np.mean(chroma, axis=1)  # shape (12,)
            # Normalise to a probability distribution
            chroma_sum = float(np.sum(mean_chroma)) + 1e-8
            chroma_prob = mean_chroma / chroma_sum
            # Entropy
            chroma_prob = chroma_prob[chroma_prob > 0]
            entropy = -float(np.sum(chroma_prob * np.log2(chroma_prob + 1e-12)))
            max_entropy = float(np.log2(12))  # 12 pitch classes
            normalised_entropy = entropy / max_entropy
            features["tonal_richness"] = float(
                np.clip(1.0 - abs(normalised_entropy - 0.7) / 0.7, 0.0, 1.0)
            )

            # ----- MERT-based features (if neural scorer available) --------
            _mert_neutral = 0.5
            if self._neural is not None:
                try:
                    neural_scores = self._neural.score(audio, sr)
                    features["mert_activation_quality"] = neural_scores.get(
                        "activation_strength", _mert_neutral
                    )
                    features["mert_temporal_smoothness"] = neural_scores.get(
                        "temporal_coherence", _mert_neutral
                    )
                    features["mert_representation_richness"] = neural_scores.get(
                        "layer_agreement", _mert_neutral
                    )
                    features["mert_structural_quality"] = neural_scores.get(
                        "structural_periodicity", _mert_neutral
                    )
                except Exception:
                    for k in ("mert_activation_quality", "mert_temporal_smoothness",
                              "mert_representation_richness", "mert_structural_quality"):
                        features[k] = _mert_neutral
            else:
                for k in ("mert_activation_quality", "mert_temporal_smoothness",
                          "mert_representation_richness", "mert_structural_quality"):
                    features[k] = _mert_neutral

            # ----- weighted combination -----------------------------------
            total_score = 0.0
            for feat_name, weight in self._HEURISTIC_WEIGHTS.items():
                total_score += features.get(feat_name, 0.0) * weight

            return float(np.clip(total_score, 0.0, 1.0))

        except Exception as exc:
            logger.error(f"Heuristic scoring failed: {exc}")
            return 0.0
