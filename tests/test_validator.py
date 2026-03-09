"""Tests for validator components: prompt generator, challenge orchestration, staggering."""

import time
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from tuneforge.validation.prompt_generator import (
    GENRES,
    MOODS,
    KEY_SIGNATURES,
    PromptGenerator,
)


class TestPromptGenerator:

    def test_generate_challenge_has_required_keys(self):
        pg = PromptGenerator(seed=42)
        challenge = pg.generate_challenge()

        required_keys = {
            "prompt", "genre", "mood", "tempo_bpm", "duration_seconds",
            "key_signature", "time_signature", "instruments",
            "challenge_id", "seed",
        }
        assert required_keys.issubset(set(challenge.keys()))

    def test_genre_from_vocabulary(self):
        pg = PromptGenerator(seed=42)
        challenge = pg.generate_challenge()
        assert challenge["genre"] in GENRES

    def test_mood_from_vocabulary(self):
        pg = PromptGenerator(seed=42)
        challenge = pg.generate_challenge()
        assert challenge["mood"] in MOODS

    def test_tempo_in_range(self):
        pg = PromptGenerator(seed=42)
        for _ in range(50):
            c = pg.generate_challenge()
            assert 20 <= c["tempo_bpm"] <= 300

    def test_duration_valid(self):
        pg = PromptGenerator(seed=42)
        for _ in range(50):
            c = pg.generate_challenge()
            assert c["duration_seconds"] in [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]

    def test_instruments_not_empty(self):
        pg = PromptGenerator(seed=42)
        for _ in range(50):
            c = pg.generate_challenge()
            assert len(c["instruments"]) >= 2

    def test_key_signature_valid(self):
        pg = PromptGenerator(seed=42)
        challenge = pg.generate_challenge()
        assert challenge["key_signature"] in KEY_SIGNATURES

    def test_challenge_id_unique(self):
        pg = PromptGenerator(seed=42)
        ids = [pg.generate_challenge()["challenge_id"] for _ in range(100)]
        assert len(set(ids)) == 100

    def test_diversity_across_challenges(self):
        pg = PromptGenerator(seed=42)
        genres = set()
        moods = set()
        for _ in range(100):
            c = pg.generate_challenge()
            genres.add(c["genre"])
            moods.add(c["mood"])
        # Should hit many different genres and moods
        assert len(genres) >= 10
        assert len(moods) >= 10

    def test_prompt_is_nonempty_string(self):
        pg = PromptGenerator(seed=42)
        for _ in range(20):
            c = pg.generate_challenge()
            assert isinstance(c["prompt"], str)
            assert len(c["prompt"]) > 10

    def test_deterministic_with_seed(self):
        pg1 = PromptGenerator(seed=123)
        pg2 = PromptGenerator(seed=123)
        c1 = pg1.generate_challenge()
        c2 = pg2.generate_challenge()
        assert c1["genre"] == c2["genre"]
        assert c1["mood"] == c2["mood"]
        assert c1["tempo_bpm"] == c2["tempo_bpm"]


class TestChallengeStagger:
    """Tests for epoch-aligned challenge staggering between validators."""

    def _make_validator_stub(self, uid: int, validator_uids: list[int], interval: int = 600):
        """Create a minimal mock with the methods bound from BaseValidatorNeuron."""
        from tuneforge.base.validator import BaseValidatorNeuron

        stub = MagicMock()
        stub.uid = uid
        stub._uid = uid

        # Mock metagraph with validator_permit
        n_total = max(validator_uids) + 10 if validator_uids else 10
        stub.metagraph.S = list(range(n_total))
        permits = [False] * n_total
        for v in validator_uids:
            permits[v] = True
        stub.metagraph.validator_permit = permits

        stub.settings.validation_interval = interval
        stub.wallet.hotkey.ss58_address = f"5FakeHotkey{uid:04d}"

        # Bind real methods
        stub.get_validator_uids = lambda: [
            u for u in range(n_total) if permits[u]
        ]
        stub._compute_stagger_offset = lambda: BaseValidatorNeuron._compute_stagger_offset(stub)
        stub._compute_cycle_jitter = lambda c: BaseValidatorNeuron._compute_cycle_jitter(stub, c)

        return stub

    def test_offsets_are_evenly_spaced(self):
        """N validators should produce N evenly-spaced offsets covering [0, interval)."""
        vali_uids = [10, 20, 30, 40, 50]
        offsets = []
        for uid in vali_uids:
            stub = self._make_validator_stub(uid, vali_uids)
            offsets.append(stub._compute_stagger_offset())

        expected_gap = 600 / len(vali_uids)  # 120s
        for i in range(1, len(offsets)):
            assert abs(offsets[i] - offsets[i - 1] - expected_gap) < 0.01

    def test_offsets_cover_full_interval(self):
        """First offset should be 0, last should be interval * (N-1)/N."""
        vali_uids = [5, 15, 25]
        offsets = []
        for uid in vali_uids:
            stub = self._make_validator_stub(uid, vali_uids)
            offsets.append(stub._compute_stagger_offset())

        assert offsets[0] == 0.0
        assert abs(offsets[-1] - 600 * 2 / 3) < 0.01

    def test_single_validator_offset_zero(self):
        """A lone validator should get offset 0."""
        stub = self._make_validator_stub(42, [42])
        assert stub._compute_stagger_offset() == 0.0

    def test_offset_stable_across_calls(self):
        """Same validator set should produce same offset every time."""
        vali_uids = [1, 2, 3, 4]
        stub = self._make_validator_stub(2, vali_uids)
        o1 = stub._compute_stagger_offset()
        o2 = stub._compute_stagger_offset()
        assert o1 == o2

    def test_no_overlap_with_jitter(self):
        """Even with ±5s jitter, slots from different validators should never overlap.

        Minimum gap = interval/N - 2*max_jitter.  For N=10, gap=60-10=50s > 0.
        """
        vali_uids = list(range(10))
        offsets = []
        for uid in vali_uids:
            stub = self._make_validator_stub(uid, vali_uids)
            offset = stub._compute_stagger_offset()
            # Worst-case jitter: this validator +5s, next validator -5s
            offsets.append(offset)

        gap = 600 / len(vali_uids)  # 60s
        max_jitter = 5.0
        min_effective_gap = gap - 2 * max_jitter  # 50s
        assert min_effective_gap > 0, "Slots could overlap with 10 validators"

        # Verify actual gaps
        for i in range(1, len(offsets)):
            actual_gap = offsets[i] - offsets[i - 1]
            assert actual_gap > 2 * max_jitter, f"Gap {actual_gap}s too small"

    def test_jitter_deterministic(self):
        """Same hotkey + cycle number should produce same jitter."""
        stub = self._make_validator_stub(1, [1, 2, 3])
        j1 = stub._compute_cycle_jitter(42)
        j2 = stub._compute_cycle_jitter(42)
        assert j1 == j2

    def test_jitter_varies_across_cycles(self):
        """Different cycle numbers should produce different jitter values."""
        stub = self._make_validator_stub(1, [1, 2, 3])
        jitters = {stub._compute_cycle_jitter(c) for c in range(100)}
        assert len(jitters) > 50, "Jitter should vary across cycles"

    def test_jitter_bounded(self):
        """Jitter should always be within ±5s."""
        stub = self._make_validator_stub(1, [1, 2, 3])
        for cycle in range(1000):
            j = stub._compute_cycle_jitter(cycle)
            assert -5.0 <= j <= 5.0, f"Jitter {j} out of bounds at cycle {cycle}"

    def test_unknown_uid_returns_zero(self):
        """If our UID isn't in the validator set, return 0."""
        stub = self._make_validator_stub(99, [1, 2, 3])
        assert stub._compute_stagger_offset() == 0.0

    def test_adapts_to_validator_set_changes(self):
        """Offset should change when the validator set changes."""
        # Initially 3 validators, we're position 1/3
        stub = self._make_validator_stub(20, [10, 20, 30])
        o1 = stub._compute_stagger_offset()

        # A new validator joins, we're now position 1/4
        stub2 = self._make_validator_stub(20, [10, 15, 20, 30])
        o2 = stub2._compute_stagger_offset()

        assert o1 != o2, "Offset should change when validator set changes"
