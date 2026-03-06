"""
Frechet Audio Distance (FAD) scorer for TuneForge.

Computes per-miner FAD over a sliding window of CLAP embeddings,
comparing each miner's distribution to a reference distribution
of real music. Used as a multiplicative penalty in the reward pipeline.

FAD = ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2 * (C_r @ C_g)^{1/2})

Lower FAD = closer to real music distribution = less penalty.
"""

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from loguru import logger


class FADScorer:
    """Batch Frechet Audio Distance scorer per miner."""

    def __init__(
        self,
        window_size: int = 50,
        reference_stats_path: str | None = None,
        penalty_midpoint: float = 15.0,
        penalty_steepness: float = 2.0,
        penalty_floor: float = 0.5,
        min_embeddings: int = 10,
    ) -> None:
        self._window_size = window_size
        self._penalty_midpoint = penalty_midpoint
        self._penalty_steepness = penalty_steepness
        self._penalty_floor = penalty_floor
        self._min_embeddings = min_embeddings

        # Per-miner sliding window of normalized CLAP embeddings
        self._miner_embeddings: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

        # Reference distribution statistics
        self._ref_mean: np.ndarray | None = None
        self._ref_cov: np.ndarray | None = None

        if reference_stats_path:
            self.load_reference_stats(reference_stats_path)

    def load_reference_stats(self, path: str) -> bool:
        """Load precomputed reference mean+covariance from .npz file.

        Expected keys: 'mean' (shape [D,]) and 'cov' (shape [D, D]).
        """
        p = Path(path)
        if not p.exists():
            logger.info("FAD reference stats not found at {} — penalty disabled", path)
            return False

        try:
            data = np.load(p)
            self._ref_mean = data["mean"].astype(np.float64)
            self._ref_cov = data["cov"].astype(np.float64)
            logger.info(
                "Loaded FAD reference stats from {} (dim={})", path, len(self._ref_mean)
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load FAD reference stats: {}", exc)
            return False

    def update_miner_embedding(self, hotkey: str, embedding: np.ndarray) -> None:
        """Add a CLAP embedding to a miner's sliding window."""
        if embedding is None:
            return
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return
        self._miner_embeddings[hotkey].append(
            (embedding / norm).astype(np.float64)
        )

    def compute_miner_fad(self, hotkey: str) -> float | None:
        """Compute FAD between miner's recent embeddings and reference.

        Returns None if insufficient data (fewer than min_embeddings or
        no reference stats loaded).
        """
        if self._ref_mean is None or self._ref_cov is None:
            return None

        embeddings = self._miner_embeddings.get(hotkey)
        if embeddings is None or len(embeddings) < self._min_embeddings:
            return None

        emb_array = np.array(list(embeddings))  # shape [N, D]
        mu_g = np.mean(emb_array, axis=0)
        cov_g = np.cov(emb_array, rowvar=False)

        # Ensure covariance is well-conditioned
        cov_g += np.eye(cov_g.shape[0]) * 1e-6

        return self._frechet_distance(self._ref_mean, self._ref_cov, mu_g, cov_g)

    def get_fad_penalty(self, hotkey: str) -> float:
        """Convert FAD to a 0-1 multiplicative penalty.

        Returns 1.0 (no penalty) when:
        - No reference stats loaded
        - Fewer than min_embeddings for this miner

        Otherwise: sigmoid mapping where FAD=0 -> 1.0, FAD->inf -> penalty_floor.
        """
        fad = self.compute_miner_fad(hotkey)
        if fad is None:
            return 1.0

        # Sigmoid: penalty = floor + (1 - floor) / (1 + (fad / midpoint)^steepness)
        ratio = fad / self._penalty_midpoint
        denominator = 1.0 + ratio ** self._penalty_steepness
        penalty = self._penalty_floor + (1.0 - self._penalty_floor) / denominator

        return float(np.clip(penalty, self._penalty_floor, 1.0))

    @staticmethod
    def _frechet_distance(
        mu1: np.ndarray,
        cov1: np.ndarray,
        mu2: np.ndarray,
        cov2: np.ndarray,
    ) -> float:
        """Compute Frechet distance between two multivariate Gaussians."""
        from scipy.linalg import sqrtm

        diff = mu1 - mu2
        mean_term = float(np.dot(diff, diff))

        # Matrix square root
        product = cov1 @ cov2
        sqrt_product = sqrtm(product)

        # sqrtm can return complex values due to numerical issues
        if np.iscomplexobj(sqrt_product):
            sqrt_product = sqrt_product.real

        trace_term = float(
            np.trace(cov1) + np.trace(cov2) - 2.0 * np.trace(sqrt_product)
        )

        return max(0.0, mean_term + trace_term)
