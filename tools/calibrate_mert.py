#!/usr/bin/env python3
"""
Calibrate MERT neural quality bell curve parameters from reference WAV tracks.

Usage
-----
    python tools/calibrate_mert.py /path/to/reference_tracks/
    python tools/calibrate_mert.py /path/to/reference_tracks/ --sample-rate 44100

The script loads every ``*.wav`` file in the given directory, runs each
through the MERT-v1-95M model (or whichever model is configured via
``TF_MERT_MODEL``), and computes the four neural quality metrics used by
TuneForge's ``NeuralQualityScorer``:

    1. temporal_coherence   -- mean cosine similarity of consecutive
                               last-layer embeddings
    2. activation_strength  -- mean L2 norm of last-layer embeddings
    3. layer_agreement      -- mean pairwise cosine similarity across
                               all 13 layers (time-reduced)
    4. structural_periodicity -- max autocorrelation peak of last-layer
                                embedding sequence

Statistics (mean, std, median, min, max) are printed for each metric,
followed by recommended bell curve parameters suitable for pasting into
a ``.env`` file or shell profile::

    export TF_MERT_TEMPORAL_CENTER=0.85
    export TF_MERT_TEMPORAL_WIDTH=12.5
    ...

The ``center`` is set to the **median** of the observed values and the
``width`` is set to ``1 / (2 * std**2)`` so that the Gaussian bell curve
covers the observed distribution well.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Metric result container
# ---------------------------------------------------------------------------

@dataclass
class MetricStats:
    """Aggregated statistics for a single metric across all tracks."""

    name: str
    values: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def std(self) -> float:
        return float(np.std(self.values))

    @property
    def median(self) -> float:
        return float(np.median(self.values))

    @property
    def min(self) -> float:
        return float(np.min(self.values))

    @property
    def max(self) -> float:
        return float(np.max(self.values))


# ---------------------------------------------------------------------------
# MERT model wrapper (reuses logic from neural_quality.py)
# ---------------------------------------------------------------------------

class MERTExtractor:
    """Thin wrapper around MERT for hidden-state extraction."""

    def __init__(self, model_name: str = "m-a-p/MERT-v1-95M") -> None:
        self._model_name = model_name
        self._model = None
        self._processor = None
        # Internal sample rate expected by MERT
        self._mert_sr: int = 24000
        self._context_seconds: float = 5.0

    def load(self) -> None:
        """Load MERT model and feature extractor."""
        if self._model is not None:
            return

        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        print(f"Loading MERT model: {self._model_name} ...")
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self._model_name, trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            self._model_name, trust_remote_code=True,
        )
        self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        print("MERT model loaded.")

    # ------------------------------------------------------------------

    @staticmethod
    def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
        if sr == target_sr:
            return audio
        import librosa
        return librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)

    # ------------------------------------------------------------------

    def extract_hidden_states(
        self, audio: np.ndarray, sr: int,
    ) -> list[torch.Tensor] | None:
        """
        Run audio through MERT and return per-layer hidden states.

        Returns a list of 13 tensors each of shape ``[time_steps, 768]``,
        or ``None`` on failure.
        """
        self.load()

        audio_24k = self._resample(audio, sr, self._mert_sr)
        chunk_samples = int(self._context_seconds * self._mert_sr)

        # Split into non-overlapping chunks
        chunks: list[np.ndarray] = []
        for start in range(0, len(audio_24k), chunk_samples):
            chunk = audio_24k[start: start + chunk_samples]
            if len(chunk) < self._mert_sr // 4:
                continue
            chunks.append(chunk.astype(np.float32))

        if not chunks:
            return None

        device = next(self._model.parameters()).device
        all_hidden: list[list[torch.Tensor]] = []

        for chunk in chunks:
            inputs = self._processor(
                chunk,
                sampling_rate=self._mert_sr,
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
                    if "out of memory" in str(gpu_err).lower():
                        print("  GPU OOM -- falling back to CPU")
                        self._model = self._model.cpu()
                        inputs = {k: v.cpu() for k, v in inputs.items()}
                        outputs = self._model(**inputs, output_hidden_states=True)
                    else:
                        raise

            chunk_hidden = [h.squeeze(0).cpu() for h in outputs.hidden_states]
            all_hidden.append(chunk_hidden)

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


# ---------------------------------------------------------------------------
# Metric computation (mirrors NeuralQualityScorer but returns raw values)
# ---------------------------------------------------------------------------

def compute_temporal_coherence(hidden_states: list[torch.Tensor]) -> float | None:
    """Mean cosine similarity of consecutive last-layer embeddings."""
    last_layer = hidden_states[-1]  # [T, 768]
    if last_layer.shape[0] < 4:
        return None
    norms = torch.nn.functional.normalize(last_layer, dim=-1)
    sims = (norms[:-1] * norms[1:]).sum(dim=-1)
    return float(sims.mean().item())


def compute_activation_strength(hidden_states: list[torch.Tensor]) -> float | None:
    """Mean L2 norm of last-layer embeddings."""
    last_layer = hidden_states[-1]
    norms = torch.norm(last_layer, dim=-1)
    return float(norms.mean().item())


def compute_layer_agreement(hidden_states: list[torch.Tensor]) -> float | None:
    """Mean pairwise cosine similarity across all 13 layers (time-reduced)."""
    layer_vecs: list[torch.Tensor] = []
    for h in hidden_states:
        layer_vecs.append(h.mean(dim=0))  # [768]

    n_layers = len(layer_vecs)
    if n_layers < 2:
        return None

    stacked = torch.stack(layer_vecs, dim=0)  # [n_layers, 768]
    normed = torch.nn.functional.normalize(stacked, dim=-1)
    sim_matrix = normed @ normed.t()
    mask = torch.triu(torch.ones(n_layers, n_layers, dtype=torch.bool), diagonal=1)
    pairwise_sims = sim_matrix[mask]
    return float(pairwise_sims.mean().item())


def compute_structural_periodicity(hidden_states: list[torch.Tensor]) -> float | None:
    """Max autocorrelation peak of last-layer embedding sequence."""
    last_layer = hidden_states[-1]
    t_steps = last_layer.shape[0]
    if t_steps < 8:
        return None

    normed = torch.nn.functional.normalize(last_layer, dim=-1)
    max_lag = t_steps // 2
    autocorr: list[float] = []
    for lag in range(1, max_lag):
        sim = (normed[:-lag] * normed[lag:]).sum(dim=-1).mean()
        autocorr.append(float(sim.item()))

    if not autocorr:
        return None

    return max(autocorr)


# ---------------------------------------------------------------------------
# WAV loading
# ---------------------------------------------------------------------------

def load_wav(path: str, target_sr: int) -> tuple[np.ndarray, int] | None:
    """Load a WAV file and return (audio_array, sample_rate)."""
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        try:
            import librosa
            audio, sr = librosa.load(path, sr=target_sr, mono=True)
            return audio, sr
        except Exception as exc:
            print(f"  WARNING: could not load {path}: {exc}")
            return None

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio.astype(np.float32), int(sr)


# ---------------------------------------------------------------------------
# Bell curve parameter derivation
# ---------------------------------------------------------------------------

def bell_width(std: float) -> float:
    """
    Compute the bell curve width parameter ``w`` such that the Gaussian
    ``exp(-w * (x - center)^2)`` fits the observed standard deviation.

    Derivation: the Gaussian ``exp(-w * x^2)`` has standard deviation
    ``sigma = 1 / sqrt(2 * w)``, so ``w = 1 / (2 * sigma^2)``.

    A minimum floor is applied to avoid division by zero or extremely
    narrow curves when all samples happen to produce identical values.
    """
    MIN_STD = 1e-6
    std = max(std, MIN_STD)
    return 1.0 / (2.0 * std * std)


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def calibrate(
    directory: str,
    sample_rate: int = 32000,
    model_name: str | None = None,
) -> dict[str, MetricStats] | None:
    """
    Run MERT calibration on all WAV files in *directory*.

    Parameters
    ----------
    directory : str
        Path to a directory containing ``.wav`` reference tracks.
    sample_rate : int
        Sample rate to assume for input WAV files (default 32000).
    model_name : str or None
        MERT model identifier.  Falls back to ``TF_MERT_MODEL`` env var
        or ``m-a-p/MERT-v1-95M``.

    Returns
    -------
    dict[str, MetricStats] or None
        Per-metric statistics, or ``None`` if no tracks were processed.
    """
    wav_files = sorted(glob.glob(os.path.join(directory, "*.wav")))
    if not wav_files:
        print(f"No WAV files found in {directory}")
        return None

    print(f"Found {len(wav_files)} WAV file(s) in {directory}\n")

    if model_name is None:
        model_name = os.environ.get("TF_MERT_MODEL", "m-a-p/MERT-v1-95M")

    extractor = MERTExtractor(model_name=model_name)

    metrics: dict[str, MetricStats] = {
        "temporal_coherence": MetricStats(name="temporal_coherence"),
        "activation_strength": MetricStats(name="activation_strength"),
        "layer_agreement": MetricStats(name="layer_agreement"),
        "structural_periodicity": MetricStats(name="structural_periodicity"),
    }

    compute_fns = {
        "temporal_coherence": compute_temporal_coherence,
        "activation_strength": compute_activation_strength,
        "layer_agreement": compute_layer_agreement,
        "structural_periodicity": compute_structural_periodicity,
    }

    for i, wav_path in enumerate(wav_files, 1):
        basename = os.path.basename(wav_path)
        print(f"[{i}/{len(wav_files)}] Processing {basename} ...")

        result = load_wav(wav_path, target_sr=sample_rate)
        if result is None:
            continue
        audio, sr = result

        duration = len(audio) / max(sr, 1)
        if duration < 0.25:
            print(f"  Skipping (too short: {duration:.2f}s)")
            continue

        hidden_states = extractor.extract_hidden_states(audio, sr)
        if hidden_states is None:
            print("  Skipping (hidden state extraction failed)")
            continue

        for metric_name, fn in compute_fns.items():
            value = fn(hidden_states)
            if value is not None:
                metrics[metric_name].values.append(value)
                print(f"  {metric_name}: {value:.6f}")

        print()

    # Check that we got at least one measurement
    any_data = any(len(m.values) > 0 for m in metrics.values())
    if not any_data:
        print("No valid measurements collected from any track.")
        return None

    return metrics


def print_results(metrics: dict[str, MetricStats]) -> None:
    """Print statistics and recommended env var exports."""

    # ---- Statistics table ----
    print("=" * 72)
    print("METRIC STATISTICS")
    print("=" * 72)

    for m in metrics.values():
        if not m.values:
            print(f"\n{m.name}: no data collected")
            continue
        print(f"\n{m.name} (n={len(m.values)}):")
        print(f"  mean   = {m.mean:.6f}")
        print(f"  std    = {m.std:.6f}")
        print(f"  median = {m.median:.6f}")
        print(f"  min    = {m.min:.6f}")
        print(f"  max    = {m.max:.6f}")

    # ---- Recommended bell curve parameters ----
    print()
    print("=" * 72)
    print("RECOMMENDED BELL CURVE PARAMETERS")
    print("=" * 72)

    env_map = {
        "temporal_coherence": ("TF_MERT_TEMPORAL_CENTER", "TF_MERT_TEMPORAL_WIDTH"),
        "activation_strength": ("TF_MERT_EXPECTED_NORM", None),
        "layer_agreement": ("TF_MERT_LAYER_CENTER", "TF_MERT_LAYER_WIDTH"),
        "structural_periodicity": ("TF_MERT_PERIODICITY_CENTER", "TF_MERT_PERIODICITY_WIDTH"),
    }

    print()
    for metric_name, (center_var, width_var) in env_map.items():
        m = metrics[metric_name]
        if not m.values:
            print(f"# {metric_name}: insufficient data")
            continue

        center = m.median
        width = bell_width(m.std)

        if metric_name == "activation_strength":
            # activation_strength uses a ratio score (mean_norm / expected_norm),
            # so the "center" is the expected norm itself (the median L2 norm).
            print(f"export {center_var}={center:.1f}")
        else:
            print(f"export {center_var}={center:.2f}")
            print(f"export {width_var}={width:.1f}")

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate MERT neural quality bell curve parameters "
            "from a directory of reference WAV tracks."
        ),
    )
    parser.add_argument(
        "directory",
        help="Path to a directory containing reference WAV files.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=32000,
        help="Sample rate of input WAV files (default: 32000).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "MERT model identifier (default: value of TF_MERT_MODEL env var, "
            "or m-a-p/MERT-v1-95M)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    directory = os.path.abspath(args.directory)
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory.", file=sys.stderr)
        return 1

    metrics = calibrate(
        directory=directory,
        sample_rate=args.sample_rate,
        model_name=args.model,
    )

    if metrics is None:
        return 1

    print_results(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
