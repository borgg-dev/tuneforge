"""
Settings module for TuneForge Subnet.

Provides singleton configuration management with Bittensor integration.
Loads from environment variables and manages wallet/subtensor connections.
"""

from functools import cached_property
from typing import Literal, Optional
import os
import time

import bittensor as bt
from pydantic_settings import BaseSettings
from pydantic import Field

from tuneforge import NETUID, DEFAULT_GENERATION_TIMEOUT, DEFAULT_VALIDATION_INTERVAL, DEFAULT_WEIGHT_UPDATE_INTERVAL


class Settings(BaseSettings):
    """
    Singleton settings for TuneForge subnet neurons.

    Loads configuration from environment variables (TF_ prefix) and provides
    lazy-loaded Bittensor primitives (wallet, subtensor, metagraph).
    """

    model_config = {"extra": "ignore", "env_prefix": "TF_", "env_file": ".env", "env_file_encoding": "utf-8"}

    # Runtime mode
    mode: Literal["miner", "validator"] = "miner"

    # Network configuration
    netuid: int = Field(default=NETUID, description="Subnet network UID")
    subtensor_network: Optional[str] = Field(
        default=None,
        description="Subtensor network (finney, test, local)"
    )
    subtensor_chain_endpoint: Optional[str] = Field(
        default=None,
        description="Custom chain endpoint URL"
    )

    # Wallet configuration
    wallet_name: str = Field(default="default", description="Wallet name")
    wallet_hotkey: str = Field(default="default", description="Hotkey name")
    wallet_path: str = Field(default="~/.bittensor/wallets", description="Wallet path")

    # Neuron configuration
    neuron_epoch_length: int = Field(
        default=100,
        description="Blocks between weight updates"
    )
    neuron_timeout: int = Field(
        default=DEFAULT_GENERATION_TIMEOUT,
        description="Forward timeout in seconds"
    )
    neuron_axon_off: bool = Field(
        default=False,
        description="Disable axon serving"
    )
    axon_port: Optional[int] = Field(
        default=None,
        description="Axon port for serving requests"
    )

    # Music generation configuration
    model_name: str = Field(
        default="ace-step-1.5",
        description="Music generation model (ace-step-1.5, facebook/musicgen-medium, etc.)"
    )
    generation_max_duration: int = Field(
        default=30,
        description="Maximum generation duration in seconds"
    )
    generation_sample_rate: int = Field(
        default=32000,
        description="Audio sample rate in Hz"
    )
    generation_timeout: int = Field(
        default=DEFAULT_GENERATION_TIMEOUT,
        description="Timeout for music generation requests"
    )
    gpu_device: str = Field(
        default="cuda:0",
        description="GPU device for model inference"
    )
    model_precision: Literal["float32", "float16", "bfloat16"] = Field(
        default="float16",
        description="Model inference precision"
    )
    guidance_scale: float = Field(
        default=3.0,
        description="Classifier-free guidance scale for generation"
    )
    temperature: float = Field(
        default=1.0,
        description="Sampling temperature for generation"
    )
    top_k: int = Field(
        default=250,
        description="Top-K sampling parameter (0 = disabled)"
    )
    top_p: float = Field(
        default=0.0,
        description="Nucleus sampling parameter (0 = disabled)"
    )

    # Validation configuration
    validation_interval: int = Field(
        default=DEFAULT_VALIDATION_INTERVAL,
        description="Seconds between validation rounds"
    )
    challenge_batch_size: int = Field(
        default=8,
        description="Number of miners to challenge per round"
    )
    max_concurrent_validations: int = Field(
        default=4,
        description="Maximum concurrent validation tasks"
    )

    # Scoring weights
    audio_quality_weight: float = Field(
        default=0.3,
        description="Weight for audio quality score"
    )
    prompt_adherence_weight: float = Field(
        default=0.4,
        description="Weight for prompt adherence (CLAP) score"
    )
    musicality_weight: float = Field(
        default=0.2,
        description="Weight for musicality score"
    )
    novelty_weight: float = Field(
        default=0.1,
        description="Weight for novelty/diversity score"
    )

    # CLAP scoring
    clap_model_name: str = Field(
        default="laion/larger_clap_music",
        description="CLAP model for text-audio similarity"
    )
    preference_model_path: Optional[str] = Field(
        default=None,
        description="Path to trained preference model checkpoint (.pt file)",
    )
    acoustid_api_key: str = Field(
        default="",
        description="AcoustID API key for known-song fingerprint lookup (optional)",
    )

    # Weight setting
    weight_setter_step: int = Field(
        default=DEFAULT_WEIGHT_UPDATE_INTERVAL,
        description="Blocks between weight setting attempts"
    )

    # Subnet API server configuration
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_max_queue_size: int = Field(default=100, description="Max pending API requests")

    # Validator organic API (HTTP server inside the validator process)
    organic_api_enabled: bool = Field(
        default=True,
        description="Enable organic generation API on the validator",
    )
    organic_api_port: int = Field(
        default=8090,
        description="Port for the validator's organic generation API",
    )

    # Storage (local filesystem for leaderboard snapshots and audio fallback)
    storage_path: str = Field(
        default="./storage",
        description="Local storage path for leaderboard snapshots and audio files"
    )

    # Validator ↔ Platform API integration
    validator_api_url: str = Field(
        default="",
        description="Platform API base URL (e.g. https://tuneforge.io). Required for validators to push data.",
    )
    validator_api_token: str = Field(
        default="",
        description="Bearer token validators send to authenticate with the platform API",
    )

    # Frontend URL (for CORS)
    frontend_url: str = Field(
        default="http://localhost:3000",
        description="Frontend application URL"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_dir: str = Field(default="/tmp/tuneforge", description="Log directory")

    # Monitoring
    wandb_enabled: bool = Field(default=False, description="Enable W&B logging")
    wandb_entity: Optional[str] = Field(default=None, description="W&B entity")
    wandb_project: str = Field(default="tuneforge", description="W&B project")

    # Internal state
    _wallet: Optional["bt.Wallet"] = None
    _subtensor: Optional["bt.Subtensor"] = None
    _metagraph: Optional["bt.Metagraph"] = None
    _metagraph_last_sync: float = 0
    _metagraph_sync_interval: int = 300  # 5 minutes

    @cached_property
    def wallet(self) -> "bt.Wallet":
        """Get or create Bittensor wallet."""
        return bt.Wallet(
            name=self.wallet_name,
            hotkey=self.wallet_hotkey,
            path=os.path.expanduser(self.wallet_path)
        )

    @cached_property
    def subtensor(self) -> "bt.Subtensor":
        """Get or create Bittensor subtensor connection."""
        if self.subtensor_chain_endpoint:
            return bt.Subtensor(chain_endpoint=self.subtensor_chain_endpoint)
        elif self.subtensor_network:
            return bt.Subtensor(network=self.subtensor_network)
        else:
            return bt.Subtensor()

    @property
    def metagraph(self) -> "bt.Metagraph":
        """
        Get metagraph with automatic sync.

        Syncs every 20 minutes to keep data fresh without
        excessive chain queries.
        """
        current_time = time.time()
        if (
            self._metagraph is None
            or current_time - self._metagraph_last_sync > self._metagraph_sync_interval
        ):
            self._metagraph = self.subtensor.metagraph(self.netuid)
            self._metagraph_last_sync = current_time
        return self._metagraph

    def sync_metagraph(self) -> "bt.Metagraph":
        """Force metagraph sync."""
        self._metagraph = self.subtensor.metagraph(self.netuid)
        self._metagraph_last_sync = time.time()
        return self._metagraph

    @cached_property
    def dendrite(self) -> "bt.Dendrite":
        """Get or create dendrite for outbound queries."""
        return bt.Dendrite(wallet=self.wallet)

    @property
    def axon(self) -> "bt.Axon":
        """Create axon for serving requests."""
        if self.axon_port:
            return bt.Axon(wallet=self.wallet, port=self.axon_port)
        return bt.Axon(wallet=self.wallet)

    def get_uid(self) -> Optional[int]:
        """Get this neuron's UID from metagraph."""
        try:
            return self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        except ValueError:
            return None

    def is_registered(self) -> bool:
        """Check if this neuron is registered on the subnet."""
        return self.get_uid() is not None

    def get_stake(self, uid: Optional[int] = None) -> float:
        """Get stake for a UID (defaults to self)."""
        if uid is None:
            uid = self.get_uid()
        if uid is None:
            return 0.0
        return float(self.metagraph.S[uid])

    def is_validator(self, uid: Optional[int] = None) -> bool:
        """Check if UID has validator permit from chain."""
        if uid is None:
            uid = self.get_uid()
        if uid is None:
            return False
        return self.metagraph.validator_permit[uid]


# Global settings singleton
settings: Settings = Settings()


def get_settings() -> Settings:
    """Get the global settings singleton."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings
