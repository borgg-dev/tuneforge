"""
Neural audio quality scorer for TuneForge.

Uses the MERT-v1-95M pre-trained music understanding model to assess
audio quality via embedding analysis.  Four metrics are derived from
the model's hidden-state representations:

1. Temporal coherence  – smooth evolution across time steps
2. Activation strength – how strongly the model responds to the input
3. Layer agreement     – consistency of representations across layers
4. Structural periodicity – presence of repeating musical structure

Bell curve parameters are configurable via environment variables
(see ``scoring_config.py``) and can be calibrated against reference
tracks using ``tools/calibrate_mert.py``.
"""

import numpy as np
import torch
from loguru import logger

from tuneforge.config.scoring_config import (
    MERT_MODEL as _MERT_MODEL_CFG,
    MERT_SAMPLE_RATE as _MERT_SR_CFG,
    MERT_EXPECTED_NORM as _MERT_EXPECTED_NORM_CFG,
    MERT_TEMPORAL_COHERENCE_CENTER,
    MERT_TEMPORAL_COHERENCE_WIDTH,
    MERT_LAYER_AGREEMENT_CENTER,
    MERT_LAYER_AGREEMENT_WIDTH,
    MERT_PERIODICITY_CENTER,
    MERT_PERIODICITY_WIDTH,
)

# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------
MERT_MODEL: str = _MERT_MODEL_CFG
MERT_SAMPLE_RATE: int = _MERT_SR_CFG
MERT_CONTEXT_SECONDS: float = 5.0

# ---------------------------------------------------------------------------
# Module-level weights (must sum to 1.0)
# ---------------------------------------------------------------------------
NEURAL_QUALITY_WEIGHTS: dict[str, float] = {
    "temporal_coherence": 0.30,
    "activation_strength": 0.20,
    "layer_agreement": 0.25,
    "structural_periodicity": 0.25,
}

# Sentinel used when model loading fails permanently
_LOAD_FAILED = "LOAD_FAILED"

# Minimum audio duration (seconds) to attempt scoring
_MIN_DURATION: float = 0.25

# Expected L2 norm for a 768-dim MERT embedding vector (configurable)
_EXPECTED_NORM: float = _MERT_EXPECTED_NORM_CFG


class NeuralQualityScorer:
    """Assess audio quality via MERT hidden-state analysis."""

    def __init__(self, model_name: str = MERT_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lazy-load MERT model and feature extractor."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, Wav2Vec2FeatureExtractor

            logger.info(f"Loading MERT model: {self._model_name}")
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self._model_name, trust_remote_code=True,
            )
            self._model = AutoModel.from_pretrained(
                self._model_name, trust_remote_code=True,
            )
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            logger.info("MERT model loaded")
        except Exception as exc:
            logger.error(f"Failed to load MERT model: {exc}")
            self._model = _LOAD_FAILED
            self._processor = None

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to *target_sr* if necessary."""
        if sr == target_sr:
            return audio
        try:
            import librosa

            return librosa.resample(
                audio.astype(np.float32), orig_sr=sr, target_sr=target_sr,
            )
        except Exception as exc:
            logger.warning(f"Resample failed, using raw audio: {exc}")
            return audio

    # ------------------------------------------------------------------
    # Hidden-state extraction
    # ------------------------------------------------------------------

    def _extract_hidden_states(
        self, audio: np.ndarray, sr: int,
    ) -> list[torch.Tensor] | None:
        """
        Run audio through MERT and return per-layer hidden states.

        The audio is resampled to 24 kHz, split into 5-second chunks,
        and the hidden states are averaged across chunks.

        Returns:
            List of 13 tensors (12 transformer layers + input embeddings),
            each of shape ``[time_steps, 768]``, or ``None`` on failure.
        """
        try:
            audio_24k = self._resample(audio, sr, MERT_SAMPLE_RATE)
            chunk_samples = int(MERT_CONTEXT_SECONDS * MERT_SAMPLE_RATE)

            # Split into non-overlapping chunks
            chunks: list[np.ndarray] = []
            for start in range(0, len(audio_24k), chunk_samples):
                chunk = audio_24k[start : start + chunk_samples]
                if len(chunk) < MERT_SAMPLE_RATE // 4:  # skip very short tails
                    continue
                chunks.append(chunk.astype(np.float32))

            if not chunks:
                return None

            device = next(self._model.parameters()).device
            all_hidden: list[list[torch.Tensor]] = []

            for chunk in chunks:
                inputs = self._processor(
                    chunk,
                    sampling_rate=MERT_SAMPLE_RATE,
                    return_tensors="pt",
                )
                inputs = {
                    k: v.to(device)
                    for k, v in inputs.items()
                    if isinstance(v, torch.Tensor)
                }
                with torch.no_grad():
                    try:
                        outputs = self._model(**inputs, output_hidden_states=True)
                    except RuntimeError as gpu_err:
                        # GPU OOM — fall back to CPU
                        if "out of memory" in str(gpu_err).lower():
                            logger.warning("GPU OOM during MERT inference, falling back to CPU")
                            torch.cuda.empty_cache()
                            self._model = self._model.cpu()
                            device = torch.device("cpu")
                            inputs = {k: v.cpu() for k, v in inputs.items()}
                            outputs = self._model(**inputs, output_hidden_states=True)
                        else:
                            raise

                # outputs.hidden_states: tuple of 13 tensors, each [1, T, 768]
                hidden = outputs.hidden_states
                chunk_hidden = [h.squeeze(0).cpu() for h in hidden]
                all_hidden.append(chunk_hidden)

            # Average hidden states across chunks (pad/truncate to min time)
            n_layers = len(all_hidden[0])
            if len(all_hidden) == 1:
                return all_hidden[0]

            averaged: list[torch.Tensor] = []
            for layer_idx in range(n_layers):
                layer_chunks = [ch[layer_idx] for ch in all_hidden]
                min_t = min(lc.shape[0] for lc in layer_chunks)
                stacked = torch.stack([lc[:min_t] for lc in layer_chunks], dim=0)
                averaged.append(stacked.mean(dim=0))

            return averaged

        except Exception as exc:
            logger.error(f"MERT hidden state extraction failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Scoring interface
    # ------------------------------------------------------------------

    def score(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """
        Compute per-metric neural quality scores.

        Args:
            audio: 1-D float waveform.
            sr:    Sample rate in Hz.

        Returns:
            Dict with keys matching ``NEURAL_QUALITY_WEIGHTS``.
            All values in [0, 1].
        """
        try:
            # Very short audio — neutral fallback
            duration = len(audio) / max(sr, 1)
            if duration < _MIN_DURATION:
                return {k: 0.5 for k in NEURAL_QUALITY_WEIGHTS}

            self._load()

            # If model loading failed permanently, return neutral
            if self._model is _LOAD_FAILED:
                return {k: 0.5 for k in NEURAL_QUALITY_WEIGHTS}

            hidden_states = self._extract_hidden_states(audio, sr)
            if hidden_states is None:
                return {k: 0.5 for k in NEURAL_QUALITY_WEIGHTS}

            return {
                "temporal_coherence": self._score_temporal_coherence(hidden_states),
                "activation_strength": self._score_activation_strength(hidden_states),
                "layer_agreement": self._score_layer_agreement(hidden_states),
                "structural_periodicity": self._score_structural_periodicity(hidden_states),
            }

        except Exception as exc:
            logger.error(f"Neural quality scoring failed: {exc}")
            return {k: 0.5 for k in NEURAL_QUALITY_WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        """
        Weighted aggregation of per-metric scores.

        Args:
            scores: Dict from ``score()``.

        Returns:
            Aggregate quality score in [0, 1].
        """
        total = 0.0
        for metric, weight in NEURAL_QUALITY_WEIGHTS.items():
            total += scores.get(metric, 0.0) * weight
        return float(np.clip(total, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_temporal_coherence(hidden_states: list[torch.Tensor]) -> float:
        """
        Measure smoothness of temporal evolution in the last hidden layer.

        Musical audio evolves smoothly; noise/artefacts have abrupt
        discontinuities.  A bell curve centred at 0.85 mean cosine
        similarity rewards natural progression without stagnation.
        """
        try:
            last_layer = hidden_states[-1]  # [T, 768]
            t_steps = last_layer.shape[0]

            if t_steps < 4:
                return 0.5

            # Normalise each time-step embedding
            norms = torch.nn.functional.normalize(last_layer, dim=-1)

            # Cosine similarity between consecutive frames
            sims = (norms[:-1] * norms[1:]).sum(dim=-1)  # [T-1]
            mean_sim = float(sims.mean().item())

            # Configurable bell curve (default: center=0.85, width=12.5)
            score = float(np.exp(
                -MERT_TEMPORAL_COHERENCE_WIDTH * (mean_sim - MERT_TEMPORAL_COHERENCE_CENTER) ** 2
            ))
            return float(np.clip(score, 0.0, 1.0))

        except Exception:
            return 0.5

    @staticmethod
    def _score_activation_strength(hidden_states: list[torch.Tensor]) -> float:
        """
        Measure how strongly the model activates for the input.

        Well-formed music drives strong, consistent activations.
        The mean L2 norm of embeddings at the last layer is compared
        against an empirical expected norm (~25.0 for 768-dim MERT).
        """
        try:
            last_layer = hidden_states[-1]  # [T, 768]

            # L2 norm per time step
            norms = torch.norm(last_layer, dim=-1)  # [T]
            mean_norm = float(norms.mean().item())

            score = min(mean_norm / _EXPECTED_NORM, 1.0)
            return float(np.clip(score, 0.0, 1.0))

        except Exception:
            return 0.5

    @staticmethod
    def _score_layer_agreement(hidden_states: list[torch.Tensor]) -> float:
        """
        Measure consistency across transformer layers.

        Good music produces consistent representations at multiple
        abstraction levels.  A bell curve centred at 0.6 pairwise
        cosine similarity rewards moderate agreement (too high
        indicates trivial input, too low indicates incoherent signal).
        """
        try:
            # Time-reduce each layer to a single 768-dim vector
            layer_vecs: list[torch.Tensor] = []
            for h in hidden_states:
                layer_vecs.append(h.mean(dim=0))  # [768]

            n_layers = len(layer_vecs)
            if n_layers < 2:
                return 0.5

            # Stack and normalise
            stacked = torch.stack(layer_vecs, dim=0)  # [n_layers, 768]
            normed = torch.nn.functional.normalize(stacked, dim=-1)

            # Pairwise cosine similarity (upper triangle)
            sim_matrix = normed @ normed.t()  # [n_layers, n_layers]
            mask = torch.triu(torch.ones(n_layers, n_layers, dtype=torch.bool), diagonal=1)
            pairwise_sims = sim_matrix[mask]
            mean_sim = float(pairwise_sims.mean().item())

            # Configurable bell curve (default: center=0.6, width=8.0)
            score = float(np.exp(
                -MERT_LAYER_AGREEMENT_WIDTH * (mean_sim - MERT_LAYER_AGREEMENT_CENTER) ** 2
            ))
            return float(np.clip(score, 0.0, 1.0))

        except Exception:
            return 0.5

    @staticmethod
    def _score_structural_periodicity(hidden_states: list[torch.Tensor]) -> float:
        """
        Detect periodic structure (beats, phrases, sections) via
        autocorrelation of embedding sequences.

        Musical audio has repeating patterns; noise and random signals
        do not.  A bell curve rewards moderate periodicity — strong but
        not total repetition.
        """
        try:
            last_layer = hidden_states[-1]  # [T, 768]
            t_steps = last_layer.shape[0]

            if t_steps < 8:
                return 0.5

            # Normalise embeddings
            normed = torch.nn.functional.normalize(last_layer, dim=-1)

            # Compute autocorrelation at each lag (dot product of shifted sequences)
            max_lag = t_steps // 2
            autocorr: list[float] = []
            for lag in range(1, max_lag):
                sim = (normed[:-lag] * normed[lag:]).sum(dim=-1).mean()
                autocorr.append(float(sim.item()))

            if not autocorr:
                return 0.5

            # Peak strength = max autocorrelation excluding lag 0
            peak_strength = max(autocorr)

            # Configurable bell curve (default: center=0.5, width=8.0)
            score = float(np.exp(
                -MERT_PERIODICITY_WIDTH * (peak_strength - MERT_PERIODICITY_CENTER) ** 2
            ))
            return float(np.clip(score, 0.0, 1.0))

        except Exception:
            return 0.5
