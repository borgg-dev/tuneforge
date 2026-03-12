"""
Mock Bittensor infrastructure for TuneForge tests.

Provides mock implementations of Subtensor, Metagraph, Wallet, Axon,
and Dendrite to enable testing without a real chain connection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Mock Wallet
# ---------------------------------------------------------------------------

@dataclass
class _MockHotkey:
    ss58_address: str = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


@dataclass
class _MockColdkey:
    ss58_address: str = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


@dataclass
class MockWallet:
    name: str = "test_wallet"
    hotkey_str: str = "default"
    hotkey: _MockHotkey = field(default_factory=_MockHotkey)
    coldkey: _MockColdkey = field(default_factory=_MockColdkey)
    coldkeypub: _MockColdkey = field(default_factory=_MockColdkey)


# ---------------------------------------------------------------------------
# Mock Axon
# ---------------------------------------------------------------------------

@dataclass
class MockAxon:
    ip: str = "127.0.0.1"
    port: int = 8091
    hotkey: str = ""
    is_serving: bool = True

    def info(self) -> dict[str, Any]:
        return {"ip": self.ip, "port": self.port, "hotkey": self.hotkey}


# ---------------------------------------------------------------------------
# Mock Metagraph
# ---------------------------------------------------------------------------

class MockMetagraph:
    """Simulates a Bittensor metagraph with configurable miners/validators."""

    def __init__(
        self,
        n: int = 8,
        stakes: list[float] | None = None,
        validator_permits: list[bool] | None = None,
    ) -> None:
        self.n = n
        self.uids = np.arange(n)

        if stakes is not None:
            assert len(stakes) == n
            self.S = np.array(stakes, dtype=np.float32)
        else:
            self.S = np.array([10_000.0] * n, dtype=np.float32)

        if validator_permits is not None:
            assert len(validator_permits) == n
            self.validator_permit = list(validator_permits)
        else:
            # First half validators, second half miners
            self.validator_permit = [i < n // 2 for i in range(n)]

        self.hotkeys = [f"5Hot{i:04d}" for i in range(n)]
        self.coldkeys = [f"5Cold{i:04d}" for i in range(n)]
        self.axons = [
            MockAxon(port=8091 + i, hotkey=self.hotkeys[i]) for i in range(n)
        ]
        self.I = np.ones(n, dtype=np.float32) / n
        self.C = np.ones(n, dtype=np.float32)
        self.trust = np.ones(n, dtype=np.float32)
        self.consensus = np.ones(n, dtype=np.float32)
        self.emission = np.ones(n, dtype=np.float32) / n


# ---------------------------------------------------------------------------
# Mock Subtensor
# ---------------------------------------------------------------------------

class MockSubtensor:
    """Simulates chain interactions for weight setting and block tracking."""

    def __init__(self, n_neurons: int = 8) -> None:
        self._block: int = 100
        self._metagraph = MockMetagraph(n=n_neurons)
        self.weight_history: list[dict[str, Any]] = []

    @property
    def block(self) -> int:
        return self._block

    def get_current_block(self) -> int:
        return self._block

    def advance_blocks(self, n: int = 1) -> None:
        """Advance the block counter for testing."""
        self._block += n

    def metagraph(self, netuid: int = 0) -> MockMetagraph:
        return self._metagraph

    def serve_axon(self, netuid: int = 0, axon: Any = None) -> None:
        pass

    def set_weights(
        self,
        wallet: Any = None,
        netuid: int = 0,
        uids: Any = None,
        weights: Any = None,
        wait_for_finalization: bool = True,
        wait_for_inclusion: bool = True,
    ) -> tuple[bool, str]:
        """Record weight submission and return success."""
        self.weight_history.append({
            "block": self._block,
            "uids": uids if uids is None else list(uids),
            "weights": weights if weights is None else list(weights),
        })
        return (True, "Success")


# ---------------------------------------------------------------------------
# Mock Dendrite
# ---------------------------------------------------------------------------

class MockDendrite:
    """Simulates dendrite that returns configurable responses."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self._responses = responses or []
        self.call_count: int = 0

    async def forward(
        self,
        axons: list[Any],
        synapse: Any,
        timeout: float = 120,
        deserialize: bool = False,
    ) -> list[Any]:
        self.call_count += 1
        if self._responses:
            return self._responses[: len(axons)]
        # Return copies of the input synapse (empty responses)
        return [synapse.model_copy() for _ in axons]

    async def __call__(
        self,
        axons: list[Any],
        synapse: Any,
        timeout: float = 120,
    ) -> list[Any]:
        return await self.forward(axons, synapse, timeout)
