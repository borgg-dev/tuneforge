"""
Prompt generator for TuneForge validation challenges.

Produces randomised music generation challenges with coherent
genre/mood/tempo/instrument combinations for miner evaluation.
"""

import random
import uuid

from loguru import logger

from tuneforge.config.scoring_config import DEFAULT_DURATION, MIN_DURATION, MAX_DURATION


# ---------------------------------------------------------------------------
# Musical vocabulary
# ---------------------------------------------------------------------------

GENRES: list[str] = [
    "pop", "rock", "classical", "jazz", "electronic",
    "ambient", "hip-hop", "r&b", "country", "folk",
    "metal", "blues", "reggae", "latin", "soul",
    "funk", "disco", "lo-fi", "cinematic", "world",
]

MOODS: list[str] = [
    "happy", "sad", "energetic", "calm", "dark",
    "uplifting", "melancholic", "aggressive", "dreamy", "mysterious",
    "romantic", "nostalgic", "triumphant", "tense", "playful",
    "ethereal", "groovy", "peaceful",
]

KEY_SIGNATURES: list[str] = [
    "C major", "C minor", "C# major", "C# minor",
    "D major", "D minor", "Eb major", "Eb minor",
    "E major", "E minor", "F major", "F minor",
    "F# major", "F# minor", "G major", "G minor",
    "Ab major", "Ab minor", "A major", "A minor",
    "Bb major", "Bb minor", "B major", "B minor",
]

TIME_SIGNATURES: list[str] = ["4/4", "3/4", "6/8", "2/4", "5/4", "7/8"]

GENRE_INSTRUMENTS: dict[str, list[str]] = {
    "pop": ["piano", "guitar", "synth", "bass", "drums", "strings"],
    "rock": ["electric guitar", "bass guitar", "drums", "organ"],
    "classical": ["piano", "violin", "cello", "flute", "oboe", "timpani", "harp"],
    "jazz": ["piano", "saxophone", "trumpet", "double bass", "drums", "vibraphone"],
    "electronic": ["synth", "drum machine", "bass synth", "pad", "arpeggiator"],
    "ambient": ["pad", "synth", "piano", "guitar", "field recordings"],
    "hip-hop": ["drum machine", "bass synth", "sampler", "piano", "synth"],
    "r&b": ["piano", "bass", "drums", "synth", "guitar"],
    "country": ["acoustic guitar", "banjo", "fiddle", "pedal steel", "bass", "drums"],
    "folk": ["acoustic guitar", "mandolin", "violin", "harmonica", "banjo"],
    "metal": ["electric guitar", "bass guitar", "drums", "double kick"],
    "blues": ["electric guitar", "harmonica", "piano", "bass", "drums"],
    "reggae": ["guitar", "bass", "drums", "organ", "horns"],
    "latin": ["guitar", "congas", "bongos", "trumpet", "piano", "bass"],
    "soul": ["organ", "piano", "bass", "drums", "horns", "guitar"],
    "funk": ["bass", "guitar", "drums", "clavinet", "horns", "synth"],
    "disco": ["bass", "drums", "strings", "synth", "guitar"],
    "lo-fi": ["piano", "guitar", "drum machine", "synth", "vinyl crackle"],
    "cinematic": ["orchestra", "strings", "brass", "choir", "timpani", "piano"],
    "world": ["sitar", "tabla", "kora", "djembe", "oud", "bamboo flute"],
}

# Tempo ranges per genre (min_bpm, max_bpm)
GENRE_TEMPO_RANGE: dict[str, tuple[int, int]] = {
    "pop": (100, 130),
    "rock": (110, 150),
    "classical": (60, 140),
    "jazz": (80, 160),
    "electronic": (120, 150),
    "ambient": (60, 90),
    "hip-hop": (80, 110),
    "r&b": (70, 110),
    "country": (90, 130),
    "folk": (80, 130),
    "metal": (130, 200),
    "blues": (70, 120),
    "reggae": (70, 100),
    "latin": (90, 140),
    "soul": (70, 110),
    "funk": (100, 130),
    "disco": (110, 130),
    "lo-fi": (70, 95),
    "cinematic": (60, 130),
    "world": (80, 140),
}

DURATIONS: list[float] = [10.0, 15.0, 20.0, 30.0]

# Natural language prompt templates
_TEMPLATES: list[str] = [
    "Create a {mood} {genre} track at {tempo} BPM featuring {instruments}",
    "Generate {genre} music with a {mood} feeling, tempo around {tempo} BPM, using {instruments}",
    "Compose a {mood} {genre} piece at {tempo} BPM with {instruments}",
    "Produce a {tempo} BPM {genre} song that sounds {mood}, featuring {instruments}",
    "Write a short {mood} {genre} composition at {tempo} BPM using {instruments}",
]


class PromptGenerator:
    """Generate randomised validation challenge prompts."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def generate_challenge(self) -> dict:
        """
        Generate a complete validation challenge.

        Returns:
            Dict with all MusicGenerationSynapse fields:
            prompt, genre, mood, tempo_bpm, duration_seconds,
            key_signature, time_signature, instruments, challenge_id, seed.
        """
        genre = self._rng.choice(GENRES)
        mood = self._rng.choice(MOODS)
        tempo_range = GENRE_TEMPO_RANGE.get(genre, (80, 140))
        tempo = self._rng.randint(tempo_range[0], tempo_range[1])
        duration = self._rng.choice(DURATIONS)
        key_sig = self._rng.choice(KEY_SIGNATURES)
        time_sig = self._rng.choice(TIME_SIGNATURES)
        instruments = self._sample_instruments(genre)
        challenge_id = uuid.uuid4().hex[:16]
        seed = self._rng.randint(0, 2**31 - 1)

        prompt = self._build_natural_prompt(genre, mood, tempo, instruments)

        challenge = {
            "prompt": prompt,
            "genre": genre,
            "mood": mood,
            "tempo_bpm": tempo,
            "duration_seconds": duration,
            "key_signature": key_sig,
            "time_signature": time_sig,
            "instruments": instruments,
            "challenge_id": challenge_id,
            "seed": seed,
        }

        logger.debug(
            f"Challenge: genre={genre}, mood={mood}, tempo={tempo}, "
            f"duration={duration}s, id={challenge_id}"
        )
        return challenge

    def _sample_instruments(self, genre: str) -> list[str]:
        """Sample 2-4 instruments appropriate for the genre."""
        pool = GENRE_INSTRUMENTS.get(genre, ["piano", "drums", "bass"])
        count = min(self._rng.randint(2, 4), len(pool))
        return self._rng.sample(pool, count)

    def _build_natural_prompt(
        self,
        genre: str,
        mood: str,
        tempo: int,
        instruments: list[str],
    ) -> str:
        """Build a natural language prompt from parameters."""
        template = self._rng.choice(_TEMPLATES)
        instruments_str = ", ".join(instruments[:-1]) + f" and {instruments[-1]}" if len(instruments) > 1 else instruments[0]
        return template.format(
            genre=genre,
            mood=mood,
            tempo=tempo,
            instruments=instruments_str,
        )
