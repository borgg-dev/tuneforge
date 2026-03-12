"""
Challenge manager for TuneForge validation rounds.

Orchestrates end-to-end validation: challenge generation, miner querying,
response collection, scoring, and result logging.
"""

import time
import uuid
from typing import Any

from loguru import logger

from tuneforge.base.dendrite import DendriteResponseEvent
from tuneforge.base.protocol import MusicGenerationSynapse
from tuneforge.validation.prompt_generator import PromptGenerator


class ChallengeManager:
    """Manage validation round lifecycle."""

    def __init__(self, seed: int | None = None) -> None:
        self._prompt_generator = PromptGenerator(seed=seed)
        self._round_history: list[dict[str, Any]] = []
        self._current_round: int = 0
        self._max_history: int = 100

    @property
    def current_round(self) -> int:
        return self._current_round

    @property
    def round_history(self) -> list[dict[str, Any]]:
        return list(self._round_history)

    def create_challenge(self) -> tuple[dict, MusicGenerationSynapse]:
        """
        Create a new challenge and its synapse.

        Returns:
            (challenge_dict, synapse) tuple.
        """
        challenge = self._prompt_generator.generate_challenge()

        synapse = MusicGenerationSynapse(
            prompt=challenge["prompt"],
            genre=challenge["genre"],
            mood=challenge["mood"],
            tempo_bpm=challenge["tempo_bpm"],
            duration_seconds=challenge["duration_seconds"],
            key_signature=challenge.get("key_signature"),
            time_signature=challenge.get("time_signature"),
            instruments=challenge.get("instruments"),
            seed=challenge.get("seed"),
            challenge_id=challenge["challenge_id"],
        )

        logger.info(
            f"Challenge created: id={challenge['challenge_id']}, "
            f"genre={challenge['genre']}, mood={challenge['mood']}"
        )
        return challenge, synapse

    async def manage_round(self, validator: Any) -> dict[str, Any]:
        """
        Execute a full validation round.

        Steps:
            1. Generate challenge
            2. Select miner subset
            3. Query miners
            4. Collect responses
            5. Record round stats

        Args:
            validator: The TuneForgeValidator instance.

        Returns:
            Round result dict with challenge, responses, and stats.
        """
        self._current_round += 1
        round_id = uuid.uuid4().hex[:12]
        round_start = time.time()

        logger.info(f"=== Validation round {self._current_round} (id={round_id}) ===")

        # 1. Generate challenge
        challenge, synapse = self.create_challenge()

        # 2. Select miner subset
        miner_uids = validator.get_miner_subset()
        if not miner_uids:
            logger.warning("No miners available — skipping round")
            return self._record_round(round_id, challenge, {}, round_start, "no_miners")

        logger.info(f"Selected {len(miner_uids)} miners for this round")

        # 3. Query miners
        try:
            axons = [validator.metagraph.axons[uid] for uid in miner_uids]
            responses = await validator.dendrite.forward(
                axons=axons,
                synapse=synapse,
                timeout=validator.settings.generation_timeout,
            )
        except Exception as exc:
            logger.error(f"Dendrite query failed: {exc}")
            return self._record_round(round_id, challenge, {}, round_start, "query_failed")

        # 4. Collect responses by UID
        response_map: dict[int, MusicGenerationSynapse] = {}
        for uid, resp in zip(miner_uids, responses):
            if resp is not None and resp.audio_b64 is not None:
                response_map[uid] = resp

        logger.info(
            f"Round {self._current_round}: "
            f"{len(response_map)}/{len(miner_uids)} miners responded with audio"
        )

        # 5. Build response event
        event = DendriteResponseEvent(
            round_id=round_id,
            block_number=validator.block,
            validator_uid=validator.uid or -1,
            round_start_time=round_start,
            round_end_time=time.time(),
        )

        for uid, resp in response_map.items():
            event.add_generation_response(
                uid=uid,
                response={"audio_b64": resp.audio_b64, "model_id": resp.model_id},
                success=True,
                generation_time_ms=resp.generation_time_ms,
            )

        result = self._record_round(round_id, challenge, response_map, round_start, "completed")
        result["event"] = event
        result["response_map"] = response_map
        result["miner_uids"] = miner_uids
        return result

    def _record_round(
        self,
        round_id: str,
        challenge: dict,
        responses: dict,
        start_time: float,
        status: str,
    ) -> dict[str, Any]:
        """Record round stats into history."""
        duration = time.time() - start_time
        record: dict[str, Any] = {
            "round_id": round_id,
            "round_number": self._current_round,
            "challenge_id": challenge.get("challenge_id", ""),
            "genre": challenge.get("genre", ""),
            "mood": challenge.get("mood", ""),
            "n_responses": len(responses),
            "duration_seconds": round(duration, 2),
            "status": status,
            "timestamp": time.time(),
        }

        self._round_history.append(record)
        if len(self._round_history) > self._max_history:
            self._round_history.pop(0)

        logger.info(
            f"Round {self._current_round} {status}: "
            f"{len(responses)} responses in {duration:.1f}s"
        )
        return record

    def get_stats(self) -> dict[str, Any]:
        """Aggregate statistics across round history."""
        if not self._round_history:
            return {"total_rounds": 0}

        completed = [r for r in self._round_history if r["status"] == "completed"]
        response_counts = [r["n_responses"] for r in completed]
        durations = [r["duration_seconds"] for r in completed]

        return {
            "total_rounds": len(self._round_history),
            "completed_rounds": len(completed),
            "avg_responses": sum(response_counts) / max(len(response_counts), 1),
            "avg_duration": sum(durations) / max(len(durations), 1),
            "genres_seen": list({r["genre"] for r in self._round_history}),
        }
