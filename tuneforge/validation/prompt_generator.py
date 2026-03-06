"""
Prompt generator for TuneForge validation challenges.

Produces randomised music generation challenges with coherent
genre/mood/tempo/instrument combinations for miner evaluation.

Uses expanded genre/mood/duration/template vocabulary with per-challenge
creative constraints, producing 100,000+ unique combinations to make
pre-caching impractical.
"""

import random
import uuid

from loguru import logger

from tuneforge.config.scoring_config import DEFAULT_DURATION, MIN_DURATION, MAX_DURATION


# ---------------------------------------------------------------------------
# Musical vocabulary
# ---------------------------------------------------------------------------

GENRES: list[str] = [
    # Original genres
    "pop", "rock", "classical", "jazz", "electronic",
    "ambient", "hip-hop", "r&b", "country", "folk",
    "metal", "blues", "reggae", "latin", "soul",
    "funk", "disco", "lo-fi", "cinematic", "world",
    # Added subgenres and styles
    "synthwave", "post-rock", "drum-and-bass", "bossa-nova", "trip-hop",
    "shoegaze", "neo-soul", "vaporwave", "dark-ambient", "progressive-rock",
    "deep-house", "indie-folk", "chamber-pop", "math-rock", "afrobeat",
    "grime", "downtempo", "psychedelic", "garage-rock", "acid-jazz",
]

MOODS: list[str] = [
    # Original moods
    "happy", "sad", "energetic", "calm", "dark",
    "uplifting", "melancholic", "aggressive", "dreamy", "mysterious",
    "romantic", "nostalgic", "triumphant", "tense", "playful",
    "ethereal", "groovy", "peaceful",
    # Added moods
    "contemplative", "anxious", "whimsical", "bittersweet", "haunting",
    "euphoric", "introspective", "rebellious", "serene", "chaotic",
    "wistful", "intense", "languid", "raw", "cinematic",
    "hypnotic", "frantic",
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
    # Added subgenres
    "synthwave": ["analog synth", "drum machine", "bass synth", "lead synth", "arpeggiated synth"],
    "post-rock": ["electric guitar", "bass guitar", "drums", "piano", "strings"],
    "drum-and-bass": ["drum machine", "bass synth", "sampler", "pad", "arpeggiator"],
    "bossa-nova": ["nylon guitar", "piano", "double bass", "drums", "flute"],
    "trip-hop": ["drum machine", "bass synth", "sampler", "piano", "strings"],
    "shoegaze": ["electric guitar", "bass guitar", "drums", "reverb guitar", "synth"],
    "neo-soul": ["piano", "bass", "drums", "guitar", "synth", "horns"],
    "vaporwave": ["sampler", "synth", "bass synth", "drum machine", "saxophone"],
    "dark-ambient": ["pad", "synth", "field recordings", "bass drone", "processed guitar"],
    "progressive-rock": ["electric guitar", "bass guitar", "drums", "organ", "synth", "mellotron"],
    "deep-house": ["drum machine", "bass synth", "piano", "pad", "organ"],
    "indie-folk": ["acoustic guitar", "banjo", "violin", "piano", "bass"],
    "chamber-pop": ["piano", "strings", "acoustic guitar", "bass", "clarinet"],
    "math-rock": ["electric guitar", "bass guitar", "drums"],
    "afrobeat": ["guitar", "bass", "drums", "horns", "percussion", "organ"],
    "grime": ["drum machine", "bass synth", "synth", "sampler"],
    "downtempo": ["drum machine", "bass synth", "pad", "piano", "sampler"],
    "psychedelic": ["electric guitar", "bass guitar", "drums", "organ", "theremin"],
    "garage-rock": ["electric guitar", "bass guitar", "drums"],
    "acid-jazz": ["piano", "bass", "drums", "saxophone", "trumpet", "clavinet"],
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
    # Added subgenres
    "synthwave": (100, 130),
    "post-rock": (80, 140),
    "drum-and-bass": (160, 180),
    "bossa-nova": (80, 120),
    "trip-hop": (70, 100),
    "shoegaze": (90, 140),
    "neo-soul": (70, 110),
    "vaporwave": (60, 100),
    "dark-ambient": (50, 80),
    "progressive-rock": (80, 160),
    "deep-house": (120, 130),
    "indie-folk": (75, 120),
    "chamber-pop": (70, 120),
    "math-rock": (100, 160),
    "afrobeat": (100, 130),
    "grime": (130, 145),
    "downtempo": (70, 100),
    "psychedelic": (80, 130),
    "garage-rock": (120, 160),
    "acid-jazz": (90, 130),
}

# Expanded duration options (seconds)
DURATIONS: list[float] = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]

# Creative constraints: emotional arcs
_EMOTIONAL_ARCS: list[str] = [
    "starts calm and builds to intense",
    "begins with high energy and gradually dissolves into stillness",
    "opens mournfully and resolves into hope",
    "starts sparse and swells into a full orchestral texture",
    "builds tension throughout with no release",
    "begins playfully and darkens toward the end",
    "opens with chaos and settles into order",
    "starts introspective and erupts into euphoria",
    "builds slowly to a single climactic peak then fades",
    "oscillates between tension and release throughout",
]

# Creative constraints: structural requirements
_STRUCTURAL_REQUIREMENTS: list[str] = [
    "with a key change in the middle",
    "with a sudden tempo shift halfway through",
    "with a brief a cappella or solo break",
    "with a false ending before the real conclusion",
    "with a gradual metric modulation",
    "with a call-and-response section between two instruments",
    "with an unexpected time signature change",
    "with a recurring motif that transforms each time it appears",
    "with a bridge section that contrasts the main theme",
    "with a coda that recontextualises the opening",
]

# Creative constraints: sonic textures
_SONIC_TEXTURES: list[str] = [
    "with reverb-heavy guitar washed in delay",
    "with heavily distorted and overdriven bass",
    "with intimate close-mic'd acoustic textures",
    "with lush layered vocal harmonics",
    "with dry, punchy, room-filling drums",
    "with airy, high-register flute over a deep drone",
    "with gritty lo-fi tape saturation throughout",
    "with crystalline, clean tones and minimal reverb",
    "with thick analogue synth pads filling the low-mids",
    "with sparse, percussive plucked strings",
]

# Creative constraints: dynamic instructions
_DYNAMIC_INSTRUCTIONS: list[str] = [
    "with a dramatic pause before the finale",
    "with a sudden drop to near-silence at the midpoint",
    "with a sustained crescendo from pianissimo to fortissimo",
    "with staccato bursts punctuating a legato melody",
    "with a whispered intro growing to a wall of sound",
    "with the main theme returning at half volume at the end",
    "with an abrupt full stop followed by a softer reprise",
    "with dynamic accents on every offbeat",
    "with a long decrescendo over the final third",
    "with sharp sforzando hits contrasted against soft passages",
]

# All constraint pools, grouped for weighted selection
_CONSTRAINT_POOLS: list[list[str]] = [
    _EMOTIONAL_ARCS,
    _STRUCTURAL_REQUIREMENTS,
    _SONIC_TEXTURES,
    _DYNAMIC_INSTRUCTIONS,
]

# Natural language prompt templates — varied structural ordering
_TEMPLATES: list[str] = [
    # Original templates
    "Create a {mood} {genre} track at {tempo} BPM featuring {instruments}",
    "Generate {genre} music with a {mood} feeling, tempo around {tempo} BPM, using {instruments}",
    "Compose a {mood} {genre} piece at {tempo} BPM with {instruments}",
    "Produce a {tempo} BPM {genre} song that sounds {mood}, featuring {instruments}",
    "Write a short {mood} {genre} composition at {tempo} BPM using {instruments}",
    # New templates with different structural orderings
    "Using {instruments}, produce a {tempo} BPM {mood} {genre} track",
    "A {genre} composition featuring {instruments} at {tempo} BPM with a {mood} atmosphere",
    "Craft a {mood} piece in the {genre} style, driven by {instruments} at {tempo} BPM",
    "At {tempo} BPM, write a {genre} track with {instruments} that feels {mood}",
    "Design a {mood} {genre} soundscape using {instruments}, set at {tempo} BPM",
    "{instruments} lead this {tempo} BPM {genre} track with a {mood} character",
    "Explore a {mood} interpretation of {genre} at {tempo} BPM through {instruments}",
    "Build a {genre} track around {instruments}, tempo {tempo} BPM, conveying a {mood} mood",
    "Record a {tempo} BPM {mood} {genre} piece where {instruments} carry the arrangement",
    "Evoke a {mood} {genre} feeling at {tempo} BPM by layering {instruments}",
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

        constraint = self._generate_creative_constraint()
        prompt = self._build_natural_prompt(genre, mood, tempo, instruments, constraint)

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

    def _generate_creative_constraint(self) -> str:
        """
        Generate a unique per-challenge creative constraint.

        Selects one constraint at random from the four constraint pools
        (emotional arcs, structural requirements, sonic textures, dynamic
        instructions).  Including this in every prompt makes the effective
        challenge space significantly larger and much harder to pre-cache.

        Returns:
            A natural-language constraint string ready to append to a prompt.
        """
        pool = self._rng.choice(_CONSTRAINT_POOLS)
        return self._rng.choice(pool)

    def _build_natural_prompt(
        self,
        genre: str,
        mood: str,
        tempo: int,
        instruments: list[str],
        constraint: str,
    ) -> str:
        """Build a natural language prompt from parameters."""
        template = self._rng.choice(_TEMPLATES)
        instruments_str = (
            ", ".join(instruments[:-1]) + f" and {instruments[-1]}"
            if len(instruments) > 1
            else instruments[0]
        )
        base = template.format(
            genre=genre,
            mood=mood,
            tempo=tempo,
            instruments=instruments_str,
        )
        return f"{base}, {constraint}."
