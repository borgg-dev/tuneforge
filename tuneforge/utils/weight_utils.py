"""
Weight utility functions for TuneForge subnet.

Provides weight normalisation, burn-weight application, and
a convenience helper for processing and submitting weights.
"""

import numpy as np
from loguru import logger

U16_MAX = 65535
U32_MAX = 4294967295


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalise a weight array so it sums to 1.0.

    Args:
        weights: Raw weight array (non-negative).

    Returns:
        Normalised weight array summing to 1.0.
        Falls back to uniform distribution if all zeros.
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 0.0, None)

    total = weights.sum()
    if total > 0:
        return (weights / total).astype(np.float32)

    # Uniform fallback when all weights are zero
    n = len(weights)
    if n == 0:
        return weights.astype(np.float32)
    return (np.ones(n, dtype=np.float64) / n).astype(np.float32)


def normalize_max_weight(
    x: np.ndarray, limit: float = 0.1
) -> np.ndarray:
    """
    Normalise array so that sum(x) == 1 and max(x) <= limit.

    This prevents any single miner from receiving a disproportionate
    share of emissions.

    Args:
        x: Raw weight array.
        limit: Maximum allowed weight for any single entry.

    Returns:
        Normalised array with max weight capped at *limit*.
    """
    epsilon = 1e-7

    weights = x.copy().astype(np.float64)
    values = np.sort(weights)

    if x.sum() == 0 or len(x) * limit <= 1:
        return np.ones_like(x, dtype=np.float32) / x.size

    estimation = values / values.sum()
    if estimation.max() <= limit:
        return (weights / weights.sum()).astype(np.float32)

    cumsum = np.cumsum(estimation, 0)
    estimation_sum = np.array(
        [(len(values) - i - 1) * estimation[i] for i in range(len(values))]
    )
    n_values = (estimation / (estimation_sum + cumsum + epsilon) < limit).sum()

    cutoff_scale = (limit * cumsum[n_values - 1] - epsilon) / (
        1 - (limit * (len(estimation) - n_values))
    )
    cutoff = cutoff_scale * values.sum()

    weights[weights > cutoff] = cutoff
    return (weights / weights.sum()).astype(np.float32)


def apply_burn_weight(
    uids: np.ndarray,
    weights: np.ndarray,
    burn_uid: int = 256,
    burn_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a burn weight to redirect a fraction of emissions.

    Rescales existing weights by (1 - burn_fraction) and appends
    the burn UID with the burned fraction.

    Args:
        uids: Array of miner UIDs.
        weights: Corresponding normalised weights.
        burn_uid: UID to receive the burned fraction.
        burn_fraction: Fraction of total weight to burn (0.0 – 1.0).

    Returns:
        Tuple of (uids_with_burn, weights_with_burn).
    """
    uids = np.asarray(uids)
    weights = np.asarray(weights, dtype=np.float64)

    if burn_fraction <= 0.0:
        return uids, weights.astype(np.float32)

    burn_fraction = min(burn_fraction, 1.0)

    # Remove burn_uid if already present
    mask = uids != burn_uid
    uids = uids[mask]
    weights = weights[mask]

    # Rescale remaining weights
    weights = weights * (1.0 - burn_fraction)

    # Append burn UID
    uids = np.append(uids, burn_uid)
    weights = np.append(weights, burn_fraction)

    return uids, weights.astype(np.float32)


def process_and_submit_weights(
    subtensor,
    wallet,
    netuid: int,
    metagraph,
    raw_weights: np.ndarray,
    burn_uid: int | None = None,
    burn_fraction: float = 0.0,
) -> bool:
    """
    Full pipeline: normalise → apply burn → process → emit → submit.

    Args:
        subtensor: Bittensor subtensor connection.
        wallet: Bittensor wallet.
        netuid: Subnet network UID.
        metagraph: Current metagraph.
        raw_weights: Unnormalised weight array.
        burn_uid: Optional UID for burn weight.
        burn_fraction: Fraction to burn (0.0 – 1.0).

    Returns:
        True if weights were successfully set on chain.
    """
    import bittensor as bt

    try:
        weights = normalize_weights(raw_weights)
        uids = np.arange(len(weights))

        if burn_uid is not None and burn_fraction > 0:
            uids, weights = apply_burn_weight(
                uids, weights, burn_uid=burn_uid, burn_fraction=burn_fraction
            )

        processed_uids, processed_weights = (
            bt.utils.weight_utils.process_weights_for_netuid(
                uids=uids,
                weights=weights,
                netuid=netuid,
                subtensor=subtensor,
                metagraph=metagraph,
            )
        )

        uint_uids, uint_weights = (
            bt.utils.weight_utils.convert_weights_and_uids_for_emit(
                uids=processed_uids,
                weights=processed_weights,
            )
        )

        if not uint_uids:
            logger.warning("No weights to set on chain after processing")
            return False

        result = subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )

        if result[0]:
            logger.info("Successfully set weights on chain")
            return True
        else:
            logger.error(f"Failed to set weights: {result}")
            return False

    except Exception as exc:
        logger.error(f"Error in process_and_submit_weights: {exc}")
        return False
