"""
SubnetsAPI implementation for TuneForge.

Provides a high-level interface for external applications (e.g., other
subnets, notebooks, or scripts) to interact with the TuneForge network
without running a full validator.
"""

from typing import Any

import bittensor as bt
from loguru import logger

from tuneforge import NETUID
from tuneforge.base.protocol import MusicGenerationSynapse


class TuneForgeAPI(bt.SubnetsAPI):
    """External integration API for the TuneForge music generation subnet.

    Usage::

        api = TuneForgeAPI(wallet=my_wallet)
        audio_bytes = await api.generate_music(
            "upbeat jazz piano",
            genre="jazz",
            duration_seconds=15.0,
        )
    """

    def __init__(
        self,
        wallet: "bt.Wallet",
        network: str = "finney",
        netuid: int = NETUID,
    ) -> None:
        super().__init__(wallet=wallet)
        self.netuid = netuid
        self.network = network
        self._dendrite = bt.Dendrite(wallet=wallet)
        self._subtensor = bt.Subtensor(network=network)
        self._metagraph: bt.Metagraph | None = None

    def prepare_synapse(self, prompt: str, **kwargs: Any) -> MusicGenerationSynapse:
        """Build a ``MusicGenerationSynapse`` from user parameters."""
        return MusicGenerationSynapse(
            prompt=prompt,
            genre=kwargs.get("genre", ""),
            mood=kwargs.get("mood", ""),
            tempo_bpm=kwargs.get("tempo_bpm", 120),
            duration_seconds=kwargs.get("duration_seconds", 15.0),
            key_signature=kwargs.get("key_signature"),
            instruments=kwargs.get("instruments"),
        )

    def process_responses(self, responses: list[MusicGenerationSynapse]) -> bytes | None:
        """Select the best response and return decoded audio bytes."""
        best: MusicGenerationSynapse | None = None
        best_time = float("inf")

        for resp in responses:
            audio = resp.deserialize()
            if audio is None or len(audio) == 0:
                continue
            gen_time = resp.generation_time_ms or 999999
            if gen_time < best_time:
                best = resp
                best_time = gen_time

        if best is None:
            return None
        return best.deserialize()

    @property
    def metagraph(self) -> "bt.Metagraph":
        if self._metagraph is None:
            self._metagraph = self._subtensor.metagraph(self.netuid)
        return self._metagraph

    def _select_miners(self, count: int = 4) -> list["bt.AxonInfo"]:
        """Pick the top *count* serving miners by incentive."""
        mg = self.metagraph
        ranked = sorted(
            enumerate(mg.I.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        axons: list[bt.AxonInfo] = []
        for uid, _ in ranked:
            if mg.axons[uid].is_serving:
                axons.append(mg.axons[uid])
            if len(axons) >= count:
                break
        return axons

    async def generate_music(
        self,
        prompt: str,
        timeout: float = 120.0,
        num_miners: int = 4,
        **kwargs: Any,
    ) -> bytes | None:
        """Generate music by querying subnet miners.

        Args:
            prompt: Text description of the desired music.
            timeout: Query timeout in seconds.
            num_miners: Number of miners to query.
            **kwargs: Extra parameters forwarded to ``prepare_synapse``.

        Returns:
            Raw audio bytes from the best miner, or ``None`` if all
            miners failed to produce valid audio.
        """
        synapse = self.prepare_synapse(prompt, **kwargs)
        axons = self._select_miners(count=num_miners)

        if not axons:
            logger.warning("No serving miners available on netuid={}", self.netuid)
            return None

        logger.info("Querying {} miners for '{}'", len(axons), prompt[:60])
        responses: list[MusicGenerationSynapse] = await self._dendrite.forward(
            axons=axons,
            synapse=synapse,
            deserialize=False,
            timeout=timeout,
        )
        return self.process_responses(responses)

    async def get_track(self, track_id: str) -> dict[str, Any] | None:
        """Retrieve track metadata from the API server (if running locally).

        This is a convenience wrapper around the REST API for scripts
        that already hold a ``TuneForgeAPI`` instance.
        """
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(f"http://localhost:8000/api/v1/tracks/{track_id}")
                if resp.status_code == 200:
                    return resp.json()
                return None
        except Exception as exc:
            logger.warning("get_track({}) failed: {}", track_id, exc)
            return None
