"""Tests for validator components: prompt generator, challenge orchestration."""

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
