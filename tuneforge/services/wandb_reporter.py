"""
W&B (Weights & Biases) Reporter for TuneForge Validators.

Reports per-miner scoring breakdowns, leaderboard rankings,
and system metrics to W&B for miner transparency and debugging.
Audio is stored in PostgreSQL — W&B logs the challenge_id for cross-reference.

Miners can view their scores at the project dashboard.
"""

import os
import threading
import time
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

from loguru import logger

try:
    import tuneforge
    TUNEFORGE_VERSION = tuneforge.VERSION
except (ImportError, AttributeError):
    TUNEFORGE_VERSION = "unknown"

if TYPE_CHECKING:
    from tuneforge.rewards.reward import ScoringBreakdown
    from tuneforge.rewards.leaderboard import MinerLeaderboard


class WandbReporter:
    """W&B reporter for validator scoring data.

    Logs per-round scoring breakdowns, leaderboard state, audio samples,
    and system-wide metrics. Thread-safe via internal lock.
    """

    def __init__(
        self,
        project: str = "tuneforge",
        entity: Optional[str] = None,
        api_key: Optional[str] = None,
        validator_uid: Optional[int] = None,
        validator_hotkey: Optional[str] = None,
        enabled: bool = True,
        wallet: Optional[Any] = None,
        netuid: Optional[int] = None,
    ):
        self.project = project
        self.entity = entity
        self.validator_uid = validator_uid
        self.validator_hotkey = validator_hotkey
        self.enabled = enabled
        self.wallet = wallet
        self.netuid = netuid
        self._initialized = False
        self._run_id: Optional[str] = None
        self._step = 0
        self._lock = threading.Lock()

        if api_key:
            os.environ["WANDB_API_KEY"] = api_key

        if enabled:
            self._initialize()

    def _initialize(self) -> bool:
        """Initialize or resume W&B run.

        Uses ``resume="allow"`` with a deterministic run ID so the same
        validator process always appends to the same W&B run — even if
        wandb.run drops (network hiccup, token refresh, etc.).  This
        prevents a new run being created every 24h or after transient
        failures.
        """
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed — W&B reporting disabled")
            self.enabled = False
            return False

        # If the run is already alive, nothing to do
        if self._initialized and wandb.run is not None:
            return True

        try:
            run_name = f"validator-{self.validator_uid}"
            if self.validator_hotkey:
                run_name = f"{run_name}-{self.validator_hotkey[:8]}"

            # Deterministic run ID so resume="allow" always reconnects
            # to the same run for this validator, even across restarts.
            run_id = self._run_id
            if run_id is None:
                # First init — generate and remember
                run_id = wandb.util.generate_id()
                self._run_id = run_id

            tags = [
                f"Version: {TUNEFORGE_VERSION}",
            ]
            if self.validator_hotkey:
                tags.append(f"Wallet: {self.validator_hotkey}")
            if self.validator_uid is not None:
                tags.append(f"UID: {self.validator_uid}")
            if self.netuid is not None:
                tags.append(f"Netuid: {self.netuid}")

            config = {
                "validator_uid": self.validator_uid,
                "validator_hotkey": self.validator_hotkey,
                "role": "validator",
                "version": TUNEFORGE_VERSION,
            }
            if self.validator_hotkey:
                config["HOTKEY_SS58"] = self.validator_hotkey
            if self.netuid is not None:
                config["NETUID"] = self.netuid

            run = wandb.init(
                project=self.project,
                entity=self.entity,
                id=run_id,
                name=run_name,
                config=config,
                tags=tags,
                resume="allow",
                # Don't capture console output — it leaks ports,
                # internal IPs, and other operational details to the
                # public W&B dashboard.
                settings=wandb.Settings(console="off"),
            )

            if self.wallet and hasattr(self.wallet, "hotkey") and run:
                try:
                    signature = self.wallet.hotkey.sign(run.id.encode()).hex()
                    run.config.update({"SIGNATURE": signature}, allow_val_change=True)
                except Exception:
                    pass

            self._initialized = True
            logger.info(f"W&B initialized: {self.entity}/{self.project} as {run_name} (id={run_id})")
            return True

        except Exception as exc:
            logger.error(f"Failed to initialize W&B: {exc}")
            return False

    def log_round(
        self,
        challenge: dict,
        breakdowns: list["ScoringBreakdown"],
        uids: list[int],
        hotkeys: list[str],
        leaderboard: "MinerLeaderboard",
    ) -> None:
        """Log a complete scoring round to W&B.

        Audio is NOT stored in W&B — it lives in PostgreSQL. The challenge_id
        is logged so W&B scores can be cross-referenced with DB audio records.

        Args:
            challenge: Challenge metadata (prompt, genre, mood, tempo_bpm, duration_seconds, challenge_id).
            breakdowns: Per-miner scoring breakdowns from score_batch.
            uids: Miner UIDs aligned with breakdowns.
            hotkeys: Miner hotkeys aligned with breakdowns.
            leaderboard: Current leaderboard for EMA/weight data.
        """
        if not self.enabled:
            return

        with self._lock:
            self._step += 1
            step = self._step

        try:
            import wandb
            # Re-establish run if it dropped (network hiccup, token refresh)
            if not self._initialize():
                return

            data: dict[str, Any] = {}

            # Populate UID/hotkey on breakdowns
            for i, bd in enumerate(breakdowns):
                if i < len(uids):
                    bd.uid = uids[i]
                if i < len(hotkeys):
                    bd.hotkey = hotkeys[i]

            # 1. Round identity (cross-reference key for PostgreSQL)
            challenge_id = challenge.get("challenge_id", "")
            data["round/challenge_id"] = challenge_id

            # 2. Round scores table
            table = self._build_round_table(breakdowns, leaderboard, challenge_id)
            if table:
                data["round_scores"] = table

            # 3. Per-miner scalar metrics (time-series)
            data.update(self._build_miner_scalars(breakdowns, leaderboard))

            # 4. System-wide metrics
            data.update(self._build_system_metrics(challenge, breakdowns))

            # 5. Leaderboard state
            data.update(self._build_leaderboard_metrics(leaderboard))

            wandb.log(data, step=step)
            logger.info(f"W&B logged round {challenge_id} at step {step} ({len(breakdowns)} miners)")

        except Exception as exc:
            logger.error(f"W&B logging error: {exc}")

    def log_weight_update(
        self,
        weights: list[float],
        uids: list[int],
        uid_to_hotkey: dict[int, str],
    ) -> None:
        """Log weight setting event."""
        if not self.enabled:
            return

        try:
            import wandb
            if not self._initialize():
                return

            with self._lock:
                step = self._step

            data = {
                "weights/total_miners": len(weights),
                "weights/sum": sum(weights),
                "weights/max": max(weights) if weights else 0,
                "weights/min": min(weights) if weights else 0,
                "weights/avg": sum(weights) / len(weights) if weights else 0,
            }

            if weights:
                data["weights/distribution"] = wandb.Histogram(weights)

            if uids and weights:
                sorted_miners = sorted(zip(uids, weights), key=lambda x: x[1], reverse=True)[:10]
                for rank, (uid, weight) in enumerate(sorted_miners, 1):
                    hk = uid_to_hotkey.get(uid, "?")[:8]
                    data[f"weights/top_{rank}_uid"] = uid
                    data[f"weights/top_{rank}_weight"] = weight

            wandb.log(data, step=step)

        except Exception as exc:
            logger.error(f"W&B weight logging error: {exc}")

    def finish(self) -> None:
        if not self._initialized:
            return
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
            self._initialized = False
            logger.info("W&B run finished")
        except Exception as exc:
            logger.error(f"Error finishing W&B run: {exc}")

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_round_table(
        self,
        breakdowns: list["ScoringBreakdown"],
        leaderboard: "MinerLeaderboard",
        challenge_id: str = "",
    ) -> Optional[Any]:
        try:
            import wandb

            columns = [
                "challenge_id", "uid", "hotkey",
                "clap", "attribute", "musicality", "melody", "structural",
                "production", "neural_quality", "vocal", "quality", "preference",
                "vocal_lyrics", "timbral", "mix_separation", "learned_mos",
                "diversity", "speed",
                "duration_pen", "artifact_pen", "fad_pen", "fingerprint_pen",
                "composite", "final_reward", "ema", "weight",
            ]

            rows = []
            for bd in breakdowns:
                ema = leaderboard.get_ema(bd.uid)
                weight = leaderboard.get_weight(bd.uid)
                rows.append([
                    challenge_id, bd.uid, (bd.hotkey or "")[:16],
                    bd.clap, bd.attribute, bd.musicality, bd.melody, bd.structural,
                    bd.production, bd.neural_quality, bd.vocal, bd.quality, bd.preference,
                    bd.vocal_lyrics, bd.timbral, bd.mix_separation, bd.learned_mos,
                    bd.diversity, bd.speed,
                    bd.duration_penalty, bd.artifact_penalty, bd.fad_penalty, bd.fingerprint_penalty,
                    bd.composite, bd.final_reward, ema, weight,
                ])

            return wandb.Table(columns=columns, data=rows) if rows else None

        except Exception as exc:
            logger.debug(f"Error building round table: {exc}")
            return None

    def _build_miner_scalars(
        self,
        breakdowns: list["ScoringBreakdown"],
        leaderboard: "MinerLeaderboard",
    ) -> dict[str, float]:
        data = {}
        for bd in breakdowns:
            hk = (bd.hotkey or "")[:8]
            prefix = f"miner/{bd.uid}_{hk}"
            data[f"{prefix}/final_reward"] = bd.final_reward
            data[f"{prefix}/ema"] = leaderboard.get_ema(bd.uid)
            data[f"{prefix}/clap"] = bd.clap
            data[f"{prefix}/attribute"] = bd.attribute
            data[f"{prefix}/musicality"] = bd.musicality
            data[f"{prefix}/production"] = bd.production
            data[f"{prefix}/neural_quality"] = bd.neural_quality
            data[f"{prefix}/diversity"] = bd.diversity
            data[f"{prefix}/speed"] = bd.speed
        return data

    def _build_system_metrics(
        self,
        challenge: dict,
        breakdowns: list["ScoringBreakdown"],
    ) -> dict[str, Any]:
        rewards = [bd.final_reward for bd in breakdowns]
        data: dict[str, Any] = {
            "system/challenge_id": challenge.get("challenge_id", ""),
            "system/miners_scored": len(breakdowns),
            "system/avg_reward": sum(rewards) / len(rewards) if rewards else 0,
            "system/max_reward": max(rewards) if rewards else 0,
            "system/min_reward": min(rewards) if rewards else 0,
            "system/prompt": challenge.get("prompt", "")[:128],
            "system/genre": challenge.get("genre", ""),
            "system/mood": challenge.get("mood", ""),
            "system/duration_seconds": challenge.get("duration_seconds", 0),
            "system/tempo_bpm": challenge.get("tempo_bpm", 0),
        }

        if rewards:
            best_idx = int(np.argmax(rewards))
            data["system/best_uid"] = breakdowns[best_idx].uid
            data["system/best_reward"] = rewards[best_idx]

        return data

    def _build_leaderboard_metrics(
        self,
        leaderboard: "MinerLeaderboard",
    ) -> dict[str, Any]:
        data: dict[str, Any] = {}
        try:
            snapshot = leaderboard.snapshot()
            if not snapshot:
                return data

            sorted_miners = sorted(snapshot.items(), key=lambda x: x[1], reverse=True)
            data["leaderboard/total_ranked"] = len(sorted_miners)

            for rank, (uid, weight) in enumerate(sorted_miners[:10], 1):
                data[f"leaderboard/rank_{rank}_uid"] = uid
                data[f"leaderboard/rank_{rank}_ema"] = leaderboard.get_ema(uid)
                data[f"leaderboard/rank_{rank}_weight"] = weight

            all_emas = [leaderboard.get_ema(uid) for uid, _ in sorted_miners]
            if all_emas:
                import wandb
                data["leaderboard/ema_distribution"] = wandb.Histogram(all_emas)
                data["leaderboard/avg_ema"] = sum(all_emas) / len(all_emas)

        except Exception as exc:
            logger.debug(f"Error building leaderboard metrics: {exc}")

        return data


# Need numpy for argmax in system metrics
try:
    import numpy as np
except ImportError:
    pass
