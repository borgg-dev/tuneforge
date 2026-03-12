"""
Base miner neuron for TuneForge subnet.

Extends BaseNeuron with miner-specific functionality:
- Axon serving for receiving validator queries
- Blacklisting / priority mechanisms
- Music generation dispatch
- Graceful lifecycle management
"""

import time
from abc import abstractmethod
from typing import Tuple

import bittensor as bt
from loguru import logger
from pydantic import BaseModel, Field

from tuneforge.base.neuron import BaseNeuron
from tuneforge.base.protocol import (
    MusicGenerationSynapse,
    PingSynapse,
    HealthReportSynapse,
)
from tuneforge.settings import Settings, get_settings

# Minimum alpha stake for a validator to be whitelisted on the miner side.
# Validators below this threshold are blacklisted by miners.
MIN_VALIDATOR_STAKE: float = 10_000.0


class BaseMinerNeuron(BaseModel, BaseNeuron):
    """
    Base class for TuneForge miner neurons.

    Miners run music generation models and respond to validator
    challenges. This class handles the Bittensor networking layer
    while subclasses implement the actual generation logic.
    """

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    # Axon for receiving requests
    axon: bt.Axon | None = Field(default=None)

    # State
    is_running: bool = Field(default=False)
    should_exit: bool = Field(default=False)

    def __init__(self, settings: Settings | None = None, **kwargs):
        """Initialise the miner neuron."""
        BaseModel.__init__(self, **kwargs)
        BaseNeuron.__init__(self, settings)

    # ------------------------------------------------------------------
    # Axon setup
    # ------------------------------------------------------------------

    def setup_axon(self) -> bt.Axon:
        """
        Set up the axon for serving requests.

        Attaches handlers for generation, ping, and health synapses
        with appropriate blacklisting and priority functions.

        Note: Bittensor's axon does exact signature matching on attached
        functions. It expects plain functions with a single `synapse`
        parameter (not bound methods with `self`). We wrap our methods
        to satisfy this requirement.
        """
        if self.settings.axon_port:
            self.axon = bt.Axon(wallet=self.wallet, port=self.settings.axon_port)
        else:
            self.axon = bt.Axon(wallet=self.wallet)

        # --- Build plain-function wrappers that match bittensor's expected signatures ---

        def _fwd_generation(synapse: MusicGenerationSynapse) -> MusicGenerationSynapse:
            return self.forward_generation(synapse)

        def _bl_generation(synapse: MusicGenerationSynapse) -> Tuple[bool, str]:
            return self.blacklist_generation(synapse)

        def _pri_generation(synapse: MusicGenerationSynapse) -> float:
            return self.priority_generation(synapse)

        def _fwd_ping(synapse: PingSynapse) -> PingSynapse:
            return self.forward_ping(synapse)

        def _bl_ping(synapse: PingSynapse) -> Tuple[bool, str]:
            return self._check_blacklist(synapse)

        def _pri_ping(synapse: PingSynapse) -> float:
            return self._priority_by_stake(synapse)

        def _fwd_health(synapse: HealthReportSynapse) -> HealthReportSynapse:
            return self.forward_health(synapse)

        def _bl_health(synapse: HealthReportSynapse) -> Tuple[bool, str]:
            return self._check_blacklist(synapse)

        def _pri_health(synapse: HealthReportSynapse) -> float:
            return self._priority_by_stake(synapse)

        # Music generation handler
        self.axon.attach(
            forward_fn=_fwd_generation,
            blacklist_fn=_bl_generation,
            priority_fn=_pri_generation,
        )

        # Ping / availability handler
        self.axon.attach(
            forward_fn=_fwd_ping,
            blacklist_fn=_bl_ping,
            priority_fn=_pri_ping,
        )

        # Health report handler
        self.axon.attach(
            forward_fn=_fwd_health,
            blacklist_fn=_bl_health,
            priority_fn=_pri_health,
        )

        logger.info(f"Axon set up on port {self.axon.port}")
        return self.axon

    # ------------------------------------------------------------------
    # Blacklisting
    # ------------------------------------------------------------------

    def get_whitelisted_hotkeys(self) -> list[str]:
        """Return hotkeys of validators that meet the stake threshold.

        Only neurons with ``validator_permit`` AND stake >=
        ``MIN_VALIDATOR_STAKE`` are allowed to query this miner.
        """
        return [
            neuron.hotkey
            for neuron in self.metagraph.neurons
            if neuron.validator_permit
            and float(self.metagraph.S[neuron.uid]) >= MIN_VALIDATOR_STAKE
        ]

    def _check_blacklist(
        self, synapse: bt.Synapse
    ) -> Tuple[bool, str]:
        """
        Common blacklist logic — only allow whitelisted validators.

        A caller is whitelisted if it has a validator permit AND at least
        ``MIN_VALIDATOR_STAKE`` alpha stake.
        """
        caller_hotkey = synapse.dendrite.hotkey
        if not caller_hotkey:
            return True, "No hotkey provided"

        whitelisted = self.get_whitelisted_hotkeys()
        if caller_hotkey not in whitelisted:
            logger.trace(f"Blacklisting hotkey {caller_hotkey[:16]}… (not whitelisted)")
            return True, "Not a whitelisted validator"

        logger.trace(f"Allowing whitelisted hotkey {caller_hotkey[:16]}…")
        return False, "Whitelisted validator"

    # ------------------------------------------------------------------
    # Priority
    # ------------------------------------------------------------------

    def _priority_by_stake(self, synapse: bt.Synapse) -> float:
        """Return caller's stake as priority value."""
        caller_hotkey = synapse.dendrite.hotkey
        if not caller_hotkey:
            return 0.0
        try:
            caller_uid = self.metagraph.hotkeys.index(caller_hotkey)
            return float(self.metagraph.S[caller_uid])
        except (ValueError, IndexError):
            return 0.0

    # ------------------------------------------------------------------
    # Default forward handlers
    # ------------------------------------------------------------------

    def forward_ping(self, synapse: PingSynapse) -> PingSynapse:
        """
        Handle ping synapse — report availability and capabilities.

        Subclasses may override to customise the response.
        """
        from tuneforge import VERSION

        logger.debug("Received ping request")
        synapse.is_available = self.is_running and not self.should_exit
        synapse.version = VERSION
        return synapse

    def forward_health(self, synapse: HealthReportSynapse) -> HealthReportSynapse:
        """
        Handle health report synapse — report hardware metrics.

        Default implementation returns zeros; subclasses should
        override with real GPU/system metrics.
        """
        synapse.uptime_seconds = time.time() - self._start_time if hasattr(self, "_start_time") else 0.0
        return synapse

    # ------------------------------------------------------------------
    # Abstract methods — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def forward_generation(
        self, synapse: MusicGenerationSynapse
    ) -> MusicGenerationSynapse:
        """Generate music from the synapse prompt and parameters."""
        ...

    @abstractmethod
    def blacklist_generation(
        self, synapse: MusicGenerationSynapse
    ) -> Tuple[bool, str]:
        """Determine if a generation request should be blacklisted."""
        ...

    @abstractmethod
    def priority_generation(
        self, synapse: MusicGenerationSynapse
    ) -> float:
        """Determine request priority for generation synapse."""
        ...

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Main run loop for the miner.

        Checks registration, sets up axon, registers on chain, and
        enters the main loop.
        """
        logger.info("Starting miner neuron…")

        if not self.check_registered():
            raise RuntimeError("Miner not registered on subnet")

        self.setup_axon()
        self.axon.start()
        logger.info(f"Axon serving on port {self.axon.port}")

        # Register axon on-chain with exponential backoff for rate limits
        self._serve_axon_with_retry()

        self.is_running = True
        self._start_time = time.time()

        try:
            while self.is_running and not self.should_exit:
                try:
                    if self.should_sync_metagraph():
                        self.sync()

                    self.log_status()
                    self.step += 1
                    # Sleep in short intervals so Ctrl+C is responsive
                    for _ in range(60):
                        if self.should_exit:
                            break
                        time.sleep(1)

                except KeyboardInterrupt:
                    break
                except Exception as exc:
                    logger.error(f"Error in main loop: {exc}")
                    time.sleep(10)
        finally:
            self.shutdown()

    def _serve_axon_with_retry(self) -> None:
        """Register axon on-chain, retrying on ServingRateLimitExceeded."""
        max_retries = 5
        base_delay = 10
        for attempt in range(max_retries):
            try:
                self.subtensor.serve_axon(
                    netuid=self.settings.netuid,
                    axon=self.axon,
                )
                logger.info("Axon registered on-chain")
                return
            except Exception as exc:
                err = str(exc)
                if "Custom error: 12" in err or "ServingRateLimitExceeded" in err:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Axon serve rate limited, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    for _ in range(delay):
                        if self.should_exit:
                            return
                        time.sleep(1)
                else:
                    raise
        logger.error(
            "Failed to register axon after max retries — "
            "continuing without on-chain registration"
        )

    def shutdown(self) -> None:
        """Shut down the miner cleanly."""
        logger.info("Shutting down miner…")
        self.is_running = False
        if self.axon:
            try:
                self.axon.stop()
            except Exception as exc:
                logger.error(f"Error stopping axon: {exc}")
        logger.info("Miner shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
