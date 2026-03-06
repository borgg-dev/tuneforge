"""
TuneForge validator neuron.

Extends BaseValidatorNeuron with the full scoring pipeline:
challenge generation → miner querying → multi-signal scoring →
EMA leaderboard → on-chain weight setting.
"""

import asyncio
import base64
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from loguru import logger

from tuneforge.base.dendrite import DendriteResponseEvent
from tuneforge.base.protocol import MusicGenerationSynapse
from tuneforge.base.validator import BaseValidatorNeuron
from tuneforge.config.scoring_config import EMA_STATE_PATH, EMA_SAVE_INTERVAL
from tuneforge.rewards.leaderboard import MinerLeaderboard
from tuneforge.rewards.reward import ProductionRewardModel
from tuneforge.rewards.weight_setter import WeightSetter
from tuneforge.settings import Settings, get_settings
from tuneforge.validation.challenge_manager import ChallengeManager
from tuneforge.validation.prompt_generator import PromptGenerator


class HotkeyAuth(httpx.Auth):
    """httpx Auth flow that signs each request with a Bittensor hotkey."""

    def __init__(self, keypair: Any) -> None:
        self.keypair = keypair

    def auth_flow(self, request: httpx.Request):  # type: ignore[override]
        nonce = str(time.time_ns())
        body_hash = hashlib.sha256(request.content).hexdigest() if request.content else "empty"
        # Use decoded path WITHOUT query string — must match server's request.url.path
        path = request.url.path
        message = f"{nonce}.{self.keypair.ss58_address}.{request.method}.{path}.{body_hash}"
        signature = "0x" + self.keypair.sign(message).hex()
        request.headers["X-Validator-Hotkey"] = self.keypair.ss58_address
        request.headers["X-Validator-Nonce"] = nonce
        request.headers["X-Validator-Signature"] = signature
        yield request


class TuneForgeValidator(BaseValidatorNeuron):
    """Full-featured TuneForge subnet validator."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        super().__init__(settings=settings)

        # Scoring pipeline
        self._reward_model = ProductionRewardModel(self.settings)
        self._prompt_generator = PromptGenerator()
        self._leaderboard = MinerLeaderboard()
        self._leaderboard.load_state(EMA_STATE_PATH)
        self._challenge_manager = ChallengeManager()
        self._weight_setter: WeightSetter | None = None  # init after setup
        self._last_model_check: float = 0.0
        self._current_model_sha: str = ""

        # Concurrency: serialise scoring so challenge rounds and organic
        # requests don't corrupt shared model state (CLAP, MERT, etc.).
        # Scoring runs in a thread pool so it doesn't block the event loop
        # (keeps organic API responsive during challenge scoring).
        self._scoring_lock = asyncio.Lock()
        self._scoring_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scorer")

        logger.info("TuneForgeValidator initialised")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Set up validator with weight setter and API client."""
        super().setup()
        self._weight_setter = WeightSetter(
            subtensor=self.subtensor,
            wallet=self.wallet,
            netuid=self.settings.netuid,
            metagraph=self.metagraph,
            update_interval=self.settings.weight_setter_step,
        )
        # HTTP client for platform API (replaces direct DB access)
        self._api_client: httpx.AsyncClient | None = None
        api_url = self.settings.validator_api_url.rstrip("/")
        api_token = self.settings.validator_api_token

        # Prefer hotkey signing (production), fall back to Bearer token (dev)
        hotkey_available = False
        try:
            _ = self.wallet.hotkey.ss58_address
            hotkey_available = True
        except Exception:
            pass

        if api_url and hotkey_available:
            self._api_client = httpx.AsyncClient(
                base_url=api_url,
                auth=HotkeyAuth(self.wallet.hotkey),
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
            logger.info(f"Validator API client configured with hotkey auth: {api_url}")
        elif api_url and api_token:
            self._api_client = httpx.AsyncClient(
                base_url=api_url,
                headers={"Authorization": f"Bearer {api_token}"},
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
            logger.info(f"Validator API client configured with Bearer token: {api_url}")
        else:
            logger.warning("No validator API URL/credentials — audio saved to filesystem only")
        logger.info("TuneForgeValidator setup complete")

    # ------------------------------------------------------------------
    # Validation round
    # ------------------------------------------------------------------

    async def run_validation_round(self) -> DendriteResponseEvent:
        """Execute one complete validation round."""
        self.round_start_time = time.time()
        self.current_round += 1

        logger.info(f"=== Validation round {self.current_round} ===")

        # 0. Force metagraph resync to get fresh axon data
        # Run in executor — this is a synchronous chain call that would
        # block the event loop and freeze the organic API server.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.resync_metagraph)

        # Check for UID recycling (hotkey changes) and reset EMA for new miners
        uid_to_hotkey = {
            uid: self.metagraph.hotkeys[uid]
            for uid in range(len(self.metagraph.hotkeys))
        }
        reset_uids = self._leaderboard.check_hotkey_changes(uid_to_hotkey)
        if reset_uids:
            logger.info(f"Reset EMA for {len(reset_uids)} recycled UIDs: {reset_uids}")

        # Cache block number once (synchronous chain call) so we don't
        # call self.block multiple times during the round.
        current_block = await loop.run_in_executor(None, lambda: self.block)

        # 1. Generate challenge prompt
        challenge = self._prompt_generator.generate_challenge()

        # 2. Create synapse
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

        # 3. Select miner UIDs
        miner_uids = self.get_miner_subset()
        if not miner_uids:
            logger.warning("No miners in subset — empty round")
            return DendriteResponseEvent(
                round_id=challenge["challenge_id"],
                block_number=current_block,
                validator_uid=self.uid or -1,
            )

        logger.info(f"Querying {len(miner_uids)} miners: UIDs={miner_uids[:20]}{'...' if len(miner_uids) > 20 else ''}")

        # 4. Query miners via dendrite
        axons = [self.metagraph.axons[uid] for uid in miner_uids]
        for uid in miner_uids[:5]:
            ax = self.metagraph.axons[uid]
            logger.debug(f"  UID {uid} axon: {ax.ip}:{ax.port} hotkey={ax.hotkey[:16]}...")
        try:
            responses: list[MusicGenerationSynapse] = await self.dendrite.forward(
                axons=axons,
                synapse=synapse,
                timeout=self.settings.generation_timeout,
                deserialize=False,
            )
            logger.info(f"Dendrite returned {len(responses)} responses")
        except Exception as exc:
            logger.error(f"Dendrite forward failed: {exc}")
            responses = []

        # Log per-response status
        for uid, resp in zip(miner_uids, responses):
            if resp is None:
                logger.debug(f"  UID {uid}: no response (None)")
                continue
            try:
                status_code = resp.axon.status_code if resp.axon else '?'
                status_msg = (resp.axon.status_message or '')[:80] if resp.axon else ''
            except Exception:
                status_code = '?'
                status_msg = ''
            if hasattr(resp, 'audio_b64') and resp.audio_b64 is not None:
                logger.info(f"  UID {uid}: got audio ({len(resp.audio_b64)} chars b64) status={status_code}")
            else:
                logger.debug(f"  UID {uid}: no audio, status={status_code} msg={status_msg}")

        # Build response event
        event = DendriteResponseEvent(
            round_id=challenge["challenge_id"],
            block_number=current_block,
            validator_uid=self.uid or -1,
            round_start_time=self.round_start_time,
        )

        valid_responses: list[MusicGenerationSynapse] = []
        valid_uids: list[int] = []
        valid_hotkeys: list[str] = []

        for uid, resp in zip(miner_uids, responses):
            if resp is None:
                continue
            has_audio = resp.audio_b64 is not None
            event.add_generation_response(
                uid=uid,
                response={"audio_b64": resp.audio_b64, "model_id": resp.model_id},
                success=has_audio,
                generation_time_ms=resp.generation_time_ms,
            )
            if has_audio:
                valid_responses.append(resp)
                valid_uids.append(uid)
                try:
                    valid_hotkeys.append(self.metagraph.hotkeys[uid])
                except (IndexError, AttributeError):
                    valid_hotkeys.append(f"uid-{uid}")

        logger.info(
            f"Got {len(valid_responses)}/{len(miner_uids)} valid responses"
        )

        # 4b. Push validation round to platform API (or filesystem fallback)
        api_round_id: str | None = None
        if self._api_client is not None and valid_responses:
            try:
                validator_hotkey = ""
                try:
                    validator_hotkey = self.wallet.hotkey.ss58_address
                except Exception:
                    pass
                payload = {
                    "challenge_id": challenge["challenge_id"],
                    "prompt": challenge["prompt"],
                    "genre": challenge["genre"],
                    "mood": challenge["mood"],
                    "tempo_bpm": challenge["tempo_bpm"],
                    "duration_seconds": challenge["duration_seconds"],
                    "validator_hotkey": validator_hotkey,
                    "responses": [
                        {
                            "miner_uid": uid,
                            "miner_hotkey": hotkey,
                            "audio_b64": resp.audio_b64,
                            "generation_time_ms": resp.generation_time_ms,
                        }
                        for uid, hotkey, resp in zip(valid_uids, valid_hotkeys, valid_responses)
                    ],
                }
                api_resp = await self._api_client.post("/api/v1/validator/rounds", json=payload)
                api_resp.raise_for_status()
                result = api_resp.json()
                api_round_id = result["round_id"]
                logger.info(
                    f"Submitted round {api_round_id} to API "
                    f"({len(valid_responses)} audio entries)"
                )
            except Exception as exc:
                logger.error(f"Failed to submit validation round to API: {exc}")
        elif valid_responses:
            # Filesystem fallback when no API configured
            round_dir = Path(self.settings.storage_path) / challenge["challenge_id"]
            round_dir.mkdir(parents=True, exist_ok=True)

            meta_path = round_dir / "metadata.json"
            if not meta_path.exists():
                try:
                    meta_path.write_text(json.dumps({
                        "challenge_id": challenge["challenge_id"],
                        "prompt": challenge["prompt"],
                        "genre": challenge["genre"],
                        "mood": challenge["mood"],
                        "tempo_bpm": challenge["tempo_bpm"],
                        "duration_seconds": challenge["duration_seconds"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }, indent=2))
                except Exception as exc:
                    logger.warning(f"Failed to save metadata: {exc}")

            for uid, resp in zip(valid_uids, valid_responses):
                try:
                    audio_bytes = base64.b64decode(resp.audio_b64)
                    out_path = round_dir / f"{uid}.wav"
                    out_path.write_bytes(audio_bytes)
                    logger.info(f"Saved UID {uid} audio to {out_path} ({len(audio_bytes)} bytes)")
                except Exception as exc:
                    logger.warning(f"Failed to save audio for UID {uid}: {exc}")

        # 5. Score all responses (under lock, in thread pool to not block event loop)
        if valid_responses:
            try:
                async with self._scoring_lock:
                    loop = asyncio.get_event_loop()
                    rewards = await loop.run_in_executor(
                        self._scoring_executor,
                        self._reward_model.score_batch,
                        valid_responses,
                        valid_hotkeys,
                    )
            except Exception as exc:
                logger.error(f"Scoring failed: {exc}")
                rewards = [0.0] * len(valid_responses)

            # 6. Update leaderboard
            for uid, reward in zip(valid_uids, rewards):
                self._leaderboard.update(uid, reward)

            # Update base scores array
            self.update_scores(valid_uids, rewards)

            # 6b. Push scores to platform API
            if self._api_client is not None and api_round_id is not None:
                try:
                    score_payload = {
                        "scores": [
                            {"miner_uid": uid, "score": float(reward)}
                            for uid, reward in zip(valid_uids, rewards)
                        ]
                    }
                    score_resp = await self._api_client.post(
                        f"/api/v1/validator/rounds/{api_round_id}/scores",
                        json=score_payload,
                    )
                    score_resp.raise_for_status()
                    logger.info(f"Submitted {len(rewards)} scores for round {api_round_id}")
                except Exception as exc:
                    logger.error(f"Failed to submit scores to API: {exc}")

            # Log round results
            for uid, reward in zip(valid_uids, rewards):
                ema = self._leaderboard.get_ema(uid)
                weight = self._leaderboard.get_weight(uid)
                logger.info(
                    f"  UID {uid}: reward={reward:.4f}, "
                    f"ema={ema:.4f}, weight={weight:.4f}"
                )

        # 8. Save leaderboard snapshot for the organic query router
        try:
            snapshot_path = str(Path(self.settings.storage_path) / "leaderboard.json")
            self._leaderboard.save_snapshot(snapshot_path)
        except Exception as exc:
            logger.warning(f"Failed to save leaderboard snapshot: {exc}")

        # 8b. Periodic EMA state persistence
        if self.current_round % EMA_SAVE_INTERVAL == 0:
            try:
                self._leaderboard.save_state(EMA_STATE_PATH)
            except Exception as exc:
                logger.warning(f"Failed to save EMA state: {exc}")

        # 9. Set weights via weight setter
        # Run in executor — set_weights is a synchronous chain call
        # (wait_for_finalization=True) that blocks for 10-20s.
        if self._weight_setter is not None:
            try:
                self._weight_setter.update_metagraph(self.metagraph)
                await loop.run_in_executor(
                    None, self._weight_setter.set_weights, self._leaderboard
                )
            except Exception as exc:
                logger.error(f"Weight setting failed: {exc}")

        # 10. Check for preference model update
        await self._maybe_update_preference_model()

        # 11. Log summary
        event.round_end_time = time.time()
        lb_summary = self._leaderboard.summary()
        logger.info(
            f"Round {self.current_round} complete: "
            f"{len(valid_responses)} scored, "
            f"leaderboard: {lb_summary.get('with_weight', 0)} with weight, "
            f"duration={event.round_end_time - event.round_start_time:.1f}s"
        )

        return event

    # ------------------------------------------------------------------
    # Preference model auto-update
    # ------------------------------------------------------------------

    async def _maybe_update_preference_model(self) -> None:
        """Check for and download a newer preference model from the API.

        Checks at most once per hour (3600s).
        """
        if self._api_client is None:
            return

        now = time.time()
        if now - self._last_model_check < 3600:
            return
        self._last_model_check = now

        try:
            resp = await self._api_client.get("/api/v1/annotations/model/latest")
            if resp.status_code == 404:
                return  # No model available yet
            resp.raise_for_status()

            remote_sha = resp.headers.get("x-model-sha256", "")
            remote_version = resp.headers.get("x-model-version", "?")

            if remote_sha and remote_sha == self._current_model_sha:
                logger.debug(f"Preference model v{remote_version} is current")
                return

            # Download and write
            model_path = Path(self.settings.preference_model_path or "preference_head.pt")
            model_data = resp.content
            local_sha = hashlib.sha256(model_data).hexdigest()

            if local_sha != remote_sha:
                logger.warning(
                    f"Preference model checksum mismatch: expected {remote_sha[:12]}, got {local_sha[:12]}"
                )
                return

            model_path.write_bytes(model_data)
            self._current_model_sha = remote_sha
            logger.info(
                f"Updated preference model to v{remote_version} "
                f"(sha256={remote_sha[:12]}..., {len(model_data)} bytes)"
            )
        except Exception as exc:
            logger.debug(f"Preference model check failed: {exc}")

    # ------------------------------------------------------------------
    # Organic generation (fan-out → score → rank)
    # ------------------------------------------------------------------

    # Maximum miners to query for organic requests.  Querying fewer
    # miners keeps customer latency low while still getting competitive
    # quality.  The top-K are selected by EMA (best proven quality first).
    ORGANIC_TOP_K: int = 10

    # Shorter dendrite timeout for organic — don't let slow miners hold
    # up the customer.  Good miners respond in 5-15s.
    ORGANIC_TIMEOUT: float = 30.0

    async def run_organic_generation(
        self,
        prompt: str,
        genre: str = "",
        mood: str = "",
        tempo_bpm: int = 120,
        duration_seconds: float = 15.0,
        key_signature: str | None = None,
        instruments: list[str] | None = None,
        request_id: str = "",
    ) -> list[dict[str, Any]]:
        """Fan out an organic request to top-K miners, score, return ranked results.

        Design for low customer latency:
        1. Pick top K miners by EMA (proven quality, fast responders)
        2. Use a short dendrite timeout (30s vs 120s for challenges)
        3. Score only the valid responses that arrived in time
        4. Update leaderboard (organic scores feed the same EMA)
        5. Return results ranked by composite score

        Returns a list of dicts sorted by composite_score descending.
        """
        # Build synapse
        synapse = MusicGenerationSynapse(
            prompt=prompt,
            genre=genre,
            mood=mood,
            tempo_bpm=tempo_bpm,
            duration_seconds=duration_seconds,
            key_signature=key_signature,
            instruments=instruments,
            challenge_id=request_id or hashlib.md5(prompt.encode()).hexdigest()[:16],
            is_organic=True,
        )

        # Select top-K miners by EMA (best quality first)
        miner_uids = self._get_top_miners_by_ema(self.ORGANIC_TOP_K)
        if not miner_uids:
            logger.warning("[ORGANIC] No serving miners available")
            return []

        logger.info(
            "[ORGANIC] Querying top {} miners (of {} serving) for request {}",
            len(miner_uids),
            len(self._get_serving_miners()),
            request_id,
        )

        # Fan out with shorter timeout — don't wait for slow miners
        axons = [self.metagraph.axons[uid] for uid in miner_uids]
        try:
            responses: list[MusicGenerationSynapse] = await self.dendrite.forward(
                axons=axons,
                synapse=synapse,
                timeout=self.ORGANIC_TIMEOUT,
                deserialize=False,
            )
        except Exception as exc:
            logger.error("[ORGANIC] Dendrite fan-out failed: {}", exc)
            return []

        # Collect valid responses
        valid_responses: list[MusicGenerationSynapse] = []
        valid_uids: list[int] = []
        valid_hotkeys: list[str] = []

        for uid, resp in zip(miner_uids, responses):
            if resp is None:
                continue
            if resp.audio_b64 is None:
                continue
            valid_responses.append(resp)
            valid_uids.append(uid)
            try:
                valid_hotkeys.append(self.metagraph.hotkeys[uid])
            except (IndexError, AttributeError):
                valid_hotkeys.append(f"uid-{uid}")

        logger.info(
            "[ORGANIC] Got {}/{} valid responses for request {}",
            len(valid_responses), len(miner_uids), request_id,
        )

        if not valid_responses:
            return []

        # Score valid responses (under lock, in thread pool so event loop stays free)
        try:
            async with self._scoring_lock:
                loop = asyncio.get_event_loop()
                rewards = await loop.run_in_executor(
                    self._scoring_executor,
                    self._reward_model.score_batch,
                    valid_responses,
                    valid_hotkeys,
                )
        except Exception as exc:
            logger.error("[ORGANIC] Scoring failed: {}", exc)
            rewards = [0.0] * len(valid_responses)

        # Update leaderboard — organic scores feed the same EMA
        for uid, reward in zip(valid_uids, rewards):
            self._leaderboard.update(uid, reward)
        self.update_scores(valid_uids, rewards)

        logger.info(
            "[ORGANIC] Leaderboard updated with {} organic scores",
            len(valid_uids),
        )

        # Build ranked results
        results: list[dict[str, Any]] = []
        for uid, hotkey, resp, score in zip(valid_uids, valid_hotkeys, valid_responses, rewards):
            audio_bytes = resp.deserialize()
            if audio_bytes is None:
                continue
            results.append({
                "miner_uid": uid,
                "miner_hotkey": hotkey,
                "audio_bytes": audio_bytes,
                "sample_rate": resp.sample_rate or self.settings.generation_sample_rate,
                "generation_time_ms": resp.generation_time_ms or 0,
                "model_id": resp.model_id,
                "composite_score": round(score, 4),
                "total_queried": len(miner_uids),
                "total_valid": len(valid_responses),
            })

        # Sort by score descending
        results.sort(key=lambda r: r["composite_score"], reverse=True)

        for i, r in enumerate(results[:5]):
            logger.info(
                "[ORGANIC] #{}: UID {} score={:.4f} time={}ms",
                i + 1, r["miner_uid"], r["composite_score"], r["generation_time_ms"],
            )

        return results

    def _get_serving_miners(self) -> list[int]:
        """Get UIDs of all serving miners (exclude self)."""
        uids: list[int] = []
        try:
            total = len(self.metagraph.S)
            for uid in range(total):
                if uid == self.uid:
                    continue
                try:
                    if self.metagraph.axons[uid].is_serving:
                        uids.append(uid)
                except (IndexError, AttributeError):
                    continue
        except Exception as exc:
            logger.error("[ORGANIC] Failed to get serving miners: {}", exc)
        return uids

    def _get_top_miners_by_ema(self, k: int) -> list[int]:
        """Get the top-K serving miners ranked by EMA score.

        Miners with proven quality (high EMA from challenge rounds)
        are most likely to produce good organic results quickly.
        Falls back to all serving miners if leaderboard is empty.
        """
        serving = self._get_serving_miners()
        if not serving:
            return []

        # Rank by EMA descending, then take top K
        scored = [
            (uid, self._leaderboard.get_ema(uid))
            for uid in serving
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        top_k = [uid for uid, _ in scored[:k]]

        if top_k:
            best_ema = self._leaderboard.get_ema(top_k[0])
            worst_ema = self._leaderboard.get_ema(top_k[-1])
            logger.debug(
                "[ORGANIC] Top-{} miners: EMA range [{:.4f}, {:.4f}]",
                len(top_k), worst_ema, best_ema,
            )

        return top_k

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Shut down the validator, saving EMA state."""
        logger.info("TuneForgeValidator shutting down — saving EMA state...")
        try:
            self._leaderboard.save_state(EMA_STATE_PATH)
        except Exception as exc:
            logger.error(f"Failed to save EMA state on shutdown: {exc}")
        super().shutdown()

    def process_round_results(self, response_event: DendriteResponseEvent) -> None:
        """Process round results — scoring is done in run_validation_round."""
        summary = response_event.summary()
        logger.info(f"Round summary: {summary}")

    async def forward(self) -> DendriteResponseEvent:
        """Execute one validation round (async entry point)."""
        return await self.run_validation_round()
