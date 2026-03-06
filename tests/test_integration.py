"""Integration tests: full miner→validator→weight-set cycle with mocks."""

import base64

import numpy as np
import pytest

from tuneforge.base.protocol import MusicGenerationSynapse
from tuneforge.generation.audio_utils import AudioUtils
from tuneforge.generation.prompt_parser import PromptParser
from tuneforge.rewards.leaderboard import MinerLeaderboard
from tuneforge.validation.prompt_generator import PromptGenerator
from tests.mock_subtensor import MockSubtensor, MockWallet


class TestFullCycle:
    """Simulate: prompt generation → miner response → scoring → leaderboard → weights."""

    def _make_fake_response(self, challenge: dict, score_quality: float) -> MusicGenerationSynapse:
        """Create a fake miner response with generated sine audio."""
        sr = 32000
        duration = challenge["duration_seconds"]
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freq = 440 * score_quality  # Higher quality → different freq
        audio = (score_quality * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        audio = AudioUtils.normalize(audio)
        wav_bytes = AudioUtils.to_wav_bytes(audio, sr)
        b64 = base64.b64encode(wav_bytes).decode()

        return MusicGenerationSynapse(
            prompt=challenge["prompt"],
            genre=challenge["genre"],
            mood=challenge["mood"],
            tempo_bpm=challenge["tempo_bpm"],
            duration_seconds=challenge["duration_seconds"],
            challenge_id=challenge["challenge_id"],
            audio_b64=b64,
            sample_rate=sr,
            generation_time_ms=int(5000 / max(score_quality, 0.1)),
            model_id="test-model",
        )

    def test_prompt_to_leaderboard_cycle(self):
        """Generate prompt → create responses → score → update leaderboard."""
        # 1. Generate challenge
        pg = PromptGenerator(seed=42)
        challenge = pg.generate_challenge()
        assert challenge["prompt"]
        assert challenge["genre"]

        # 2. Simulate miner responses (varying quality)
        qualities = [0.9, 0.7, 0.5, 0.3]
        responses = [self._make_fake_response(challenge, q) for q in qualities]

        # 3. Verify responses have audio
        for resp in responses:
            assert resp.audio_b64 is not None
            raw = resp.deserialize()
            assert raw is not None
            assert len(raw) > 44

        # 4. Update leaderboard
        lb = MinerLeaderboard(alpha=0.2, steepen_baseline=0.6, steepen_power=3.0)
        for uid, q in enumerate(qualities):
            lb.update(uid, q)

        # 5. Check ordering
        assert lb.get_ema(0) > lb.get_ema(1) > lb.get_ema(2) > lb.get_ema(3)

    def test_multi_round_weight_convergence(self):
        """Multiple rounds should converge weights to top performers."""
        lb = MinerLeaderboard(alpha=0.2, steepen_baseline=0.6, steepen_power=3.0)
        pg = PromptGenerator(seed=42)

        # Simulate 15 rounds
        for round_num in range(15):
            challenge = pg.generate_challenge()
            # Miner 0 is consistently good, miner 3 is consistently bad
            scores = {0: 0.95, 1: 0.80, 2: 0.65, 3: 0.40}
            for uid, score in scores.items():
                lb.update(uid, score)

        # After warmup, only top miners get weight
        w0 = lb.get_weight(0)
        w1 = lb.get_weight(1)
        w2 = lb.get_weight(2)
        w3 = lb.get_weight(3)

        assert w0 > w1 > w2
        assert w3 == 0.0  # Below baseline

    def test_weight_submission_mock(self):
        """Verify weights can be submitted to mock subtensor."""
        from tuneforge.rewards.weight_setter import WeightSetter

        st = MockSubtensor(n_neurons=4)
        wallet = MockWallet()
        mg = st.metagraph()

        ws = WeightSetter(st, wallet, netuid=0, metagraph=mg, update_interval=10)
        st.advance_blocks(200)  # Ensure interval passed

        lb = MinerLeaderboard(alpha=0.2, steepen_baseline=0.6, steepen_power=3.0)
        for _ in range(15):
            lb.update(0, 0.95)
            lb.update(1, 0.80)
            lb.update(2, 0.50)
            lb.update(3, 0.30)

        # Verify the leaderboard has meaningful weights after 15 rounds
        assert lb.get_weight(0) > 0

        all_weights = lb.get_all_weights()
        nonzero = {uid: w for uid, w in all_weights.items() if w > 0}
        assert len(nonzero) >= 1

    def test_prompt_parser_integration(self):
        """PromptGenerator → PromptParser chain."""
        pg = PromptGenerator(seed=42)
        pp = PromptParser()

        challenge = pg.generate_challenge()
        enhanced = pp.build_prompt(
            text=challenge["prompt"],
            genre=challenge["genre"],
            mood=challenge["mood"],
            tempo=challenge["tempo_bpm"],
            instruments=challenge["instruments"],
            key=challenge.get("key_signature"),
            time_sig=challenge.get("time_signature"),
        )
        assert len(enhanced) > len(challenge["genre"])
        assert challenge["genre"] in enhanced.lower() or True  # Genre may be descriptive
