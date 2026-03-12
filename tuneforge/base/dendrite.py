"""
Dendrite response event tracking for TuneForge.

Collects and organises responses from miners during
validation rounds for scoring and analysis.
"""

import time
from typing import Any

from pydantic import BaseModel, Field


class DendriteResponseEvent(BaseModel):
    """
    Collects responses from all miners during a validation round.

    Organises data by phase (ping, generation) and provides
    convenience accessors for scoring.
    """

    # Round identification
    round_id: str = Field(default="", description="Unique round identifier")
    block_number: int = Field(default=0, description="Block at round start")
    validator_uid: int = Field(default=-1, description="Validator's UID")
    round_start_time: float = Field(
        default_factory=time.time, description="Round start timestamp"
    )
    round_end_time: float = Field(default=0.0, description="Round end timestamp")

    # Raw responses by phase
    ping_responses: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw ping synapse responses",
    )
    generation_responses: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw generation synapse responses",
    )

    # Indexed by UID for fast lookup
    ping_by_uid: dict[int, dict[str, Any]] = Field(
        default_factory=dict,
        description="Ping responses indexed by miner UID",
    )
    generation_by_uid: dict[int, dict[str, Any]] = Field(
        default_factory=dict,
        description="Generation responses indexed by miner UID",
    )

    # UIDs that participated
    participating_uids: list[int] = Field(
        default_factory=list,
        description="UIDs that participated in this round",
    )

    def add_ping_response(
        self,
        uid: int,
        response: dict[str, Any],
        success: bool = True,
    ) -> None:
        """Record a ping response for a miner."""
        entry = {"response": response, "success": success}
        self.ping_responses.append(entry)
        self.ping_by_uid[uid] = entry
        if uid not in self.participating_uids:
            self.participating_uids.append(uid)

    def add_generation_response(
        self,
        uid: int,
        response: dict[str, Any],
        success: bool = True,
        generation_time_ms: int | None = None,
        audio_size_bytes: int | None = None,
    ) -> None:
        """Record a generation response for a miner."""
        entry: dict[str, Any] = {
            "response": response,
            "success": success,
        }
        if generation_time_ms is not None:
            entry["generation_time_ms"] = generation_time_ms
        if audio_size_bytes is not None:
            entry["audio_size_bytes"] = audio_size_bytes
        self.generation_responses.append(entry)
        self.generation_by_uid[uid] = entry
        if uid not in self.participating_uids:
            self.participating_uids.append(uid)

    def get_available_uids(self) -> list[int]:
        """Get UIDs that responded to ping with availability."""
        available: list[int] = []
        for uid, data in self.ping_by_uid.items():
            if data.get("success") and data.get("response", {}).get("is_available"):
                available.append(uid)
        return available

    def get_successful_generation_uids(self) -> list[int]:
        """Get UIDs that completed generation successfully."""
        successful: list[int] = []
        for uid, data in self.generation_by_uid.items():
            if data.get("success"):
                successful.append(uid)
        return successful

    def summary(self) -> dict[str, Any]:
        """Produce a concise summary of round results."""
        duration = (
            self.round_end_time - self.round_start_time
            if self.round_end_time > 0
            else 0.0
        )
        return {
            "round_id": self.round_id,
            "block": self.block_number,
            "validator_uid": self.validator_uid,
            "total_participants": len(self.participating_uids),
            "available_miners": len(self.get_available_uids()),
            "successful_generations": len(self.get_successful_generation_uids()),
            "duration_seconds": round(duration, 2),
        }
