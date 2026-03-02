"""
Base neuron class for TuneForge subnet.

Provides common functionality for both miners and validators:
- Bittensor chain interaction
- Metagraph synchronization
- Registration verification
- Lifecycle management with graceful shutdown
"""

import signal
import time
from abc import ABC, abstractmethod

import bittensor as bt
from loguru import logger

from tuneforge.settings import Settings, get_settings


class BaseNeuron(ABC):
    """
    Abstract base class for all TuneForge neurons.

    Provides core Bittensor integration and lifecycle management.
    Subclasses (BaseMinerNeuron, BaseValidatorNeuron) extend this
    with role-specific functionality.
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialise the base neuron.

        Args:
            settings: Optional settings override. Uses global settings if None.
        """
        # Only set attributes if not already set (e.g. by Pydantic in subclasses)
        if not hasattr(self, "settings") or self.settings is None:
            self.settings = settings or get_settings()
        elif settings is not None:
            self.settings = settings

        if not hasattr(self, "step"):
            self.step: int = 0
        if not hasattr(self, "last_sync_block"):
            self.last_sync_block: int = 0

        # Bittensor primitives (lazy loaded)
        if not hasattr(self, "_wallet"):
            self._wallet: bt.Wallet | None = None
        if not hasattr(self, "_subtensor"):
            self._subtensor: bt.Subtensor | None = None
        if not hasattr(self, "_metagraph"):
            self._metagraph: bt.Metagraph | None = None
        if not hasattr(self, "_uid"):
            self._uid: int | None = None

        # Lifecycle flags
        if not hasattr(self, "is_running"):
            self.is_running: bool = False
        if not hasattr(self, "should_exit"):
            self.should_exit: bool = False

        # Install signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Initializing {self.__class__.__name__}")

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _signal_handler(self, signum: int, frame: object) -> None:
        """Handle SIGINT/SIGTERM. First press: graceful shutdown. Second: force exit."""
        sig_name = signal.Signals(signum).name
        if self.should_exit:
            logger.warning(f"Received {sig_name} again — forcing exit")
            raise SystemExit(1)
        logger.warning(f"Received {sig_name} — requesting graceful shutdown (press again to force)")
        self.should_exit = True
        self.is_running = False

    # ------------------------------------------------------------------
    # Lazy-loaded Bittensor primitives
    # ------------------------------------------------------------------

    @property
    def wallet(self) -> bt.Wallet:
        """Get Bittensor wallet (lazy loaded from settings)."""
        if self._wallet is None:
            self._wallet = self.settings.wallet
        return self._wallet

    @property
    def subtensor(self) -> bt.Subtensor:
        """Get Bittensor subtensor connection (lazy loaded)."""
        if self._subtensor is None:
            self._subtensor = self.settings.subtensor
        return self._subtensor

    @property
    def metagraph(self) -> bt.Metagraph:
        """Get Bittensor metagraph (auto-synced via settings)."""
        return self.settings.metagraph

    @property
    def uid(self) -> int | None:
        """Get this neuron's UID."""
        if self._uid is None:
            self._uid = self.settings.get_uid()
        return self._uid

    @property
    def block(self) -> int:
        """Get current block number."""
        try:
            return self.subtensor.get_current_block()
        except Exception as exc:
            logger.error(f"Failed to get current block: {exc}")
            return 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Process incoming synapse. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def run(self) -> None:
        """Main run loop. Must be implemented by subclasses."""
        ...

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def check_registered(self) -> bool:
        """
        Check if this neuron is registered on the subnet.

        Returns:
            True if registered, False otherwise.
        """
        try:
            if not self.settings.is_registered():
                logger.error(
                    f"Neuron not registered on subnet {self.settings.netuid}. "
                    f"Hotkey: {self.wallet.hotkey.ss58_address}"
                )
                return False

            self._uid = self.settings.get_uid()
            logger.info(f"Registered with UID: {self._uid}")
            return True
        except Exception as exc:
            logger.error(f"Registration check failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Metagraph sync
    # ------------------------------------------------------------------

    def should_sync_metagraph(self) -> bool:
        """Check if metagraph should be synced (epoch-based)."""
        try:
            current_block = self.block
            blocks_since_sync = current_block - self.last_sync_block
            return blocks_since_sync >= self.settings.neuron_epoch_length
        except Exception as exc:
            logger.error(f"Error checking sync status: {exc}")
            return False

    def sync(self) -> None:
        """Synchronize with the Bittensor network."""
        logger.debug("Syncing with network…")

        if not self.check_registered():
            raise RuntimeError("Neuron not registered")

        self.resync_metagraph()
        self.last_sync_block = self.block
        logger.info(f"Synced at block {self.last_sync_block}")

    def resync_metagraph(self) -> None:
        """Force metagraph resync from chain."""
        logger.debug("Resyncing metagraph…")
        try:
            self._metagraph = self.settings.sync_metagraph()
            self._uid = self.settings.get_uid()
        except Exception as exc:
            logger.error(f"Metagraph resync failed: {exc}")

    # ------------------------------------------------------------------
    # UID helpers
    # ------------------------------------------------------------------

    def get_validator_uids(self) -> list[int]:
        """Get UIDs of all validators (those with validator permit)."""
        validators: list[int] = []
        try:
            for uid_idx in range(len(self.metagraph.S)):
                if self.metagraph.validator_permit[uid_idx]:
                    validators.append(uid_idx)
        except Exception as exc:
            logger.error(f"Error fetching validator UIDs: {exc}")
        return validators

    def get_miner_uids(self) -> list[int]:
        """Get UIDs of all neurons excluding ourselves."""
        try:
            total = len(self.metagraph.S)
            uids = [uid for uid in range(total) if uid != self.uid]
            logger.debug(f"Metagraph: {total} total UIDs, querying {len(uids)}")
            return uids
        except Exception as exc:
            logger.error(f"Error fetching miner UIDs: {exc}")
            return []

    # ------------------------------------------------------------------
    # Status / context manager
    # ------------------------------------------------------------------

    def log_status(self) -> None:
        """Log current neuron status."""
        try:
            stake = self.settings.get_stake()
        except Exception:
            stake = 0.0
        logger.info(
            f"Status: UID={self.uid}, Block={self.block}, "
            f"Step={self.step}, Stake={stake:.4f}"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        logger.info(f"Shutting down {self.__class__.__name__}")
        return False
