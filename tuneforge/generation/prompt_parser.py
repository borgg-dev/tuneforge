"""
Prompt parser for TuneForge music generation.

Converts structured musical parameters into rich natural language prompts
optimized for MusicGen and other text-conditioned music generation models.
"""

from loguru import logger


# Genre-specific descriptive templates
_GENRE_DESCRIPTORS: dict[str, str] = {
    "lo-fi": "lo-fi hip hop beat",
    "lofi": "lo-fi hip hop beat",
    "hip-hop": "hip hop track",
    "hip hop": "hip hop track",
    "pop": "pop song",
    "rock": "rock track",
    "classical": "classical composition",
    "jazz": "jazz piece",
    "electronic": "electronic track",
    "edm": "EDM track",
    "ambient": "ambient soundscape",
    "r&b": "R&B groove",
    "rnb": "R&B groove",
    "soul": "soulful track",
    "blues": "blues piece",
    "country": "country song",
    "folk": "folk song",
    "metal": "metal track",
    "punk": "punk track",
    "reggae": "reggae track",
    "latin": "Latin-influenced track",
    "funk": "funky groove",
    "disco": "disco track",
    "house": "house music track",
    "techno": "techno track",
    "drum and bass": "drum and bass track",
    "dnb": "drum and bass track",
    "trap": "trap beat",
    "synthwave": "synthwave track",
    "cinematic": "cinematic score",
    "orchestral": "orchestral piece",
    "chillhop": "chill hop beat",
    "indie": "indie track",
    "acoustic": "acoustic piece",
    "world": "world music piece",
    "new age": "new age composition",
}

# Mood-specific adjective mappings
_MOOD_ADJECTIVES: dict[str, str] = {
    "happy": "bright and uplifting",
    "sad": "melancholic and sorrowful",
    "melancholic": "melancholic",
    "energetic": "high-energy and driving",
    "calm": "calm and peaceful",
    "dark": "dark and brooding",
    "uplifting": "uplifting and inspiring",
    "aggressive": "aggressive and intense",
    "romantic": "warm and romantic",
    "mysterious": "mysterious and enigmatic",
    "epic": "epic and grandiose",
    "dreamy": "dreamy and ethereal",
    "nostalgic": "nostalgic and wistful",
    "playful": "playful and lighthearted",
    "tense": "tense and suspenseful",
    "triumphant": "triumphant and victorious",
    "peaceful": "serene and tranquil",
    "angry": "fierce and aggressive",
    "chill": "laid-back and relaxed",
    "groovy": "groovy and rhythmic",
    "haunting": "haunting and atmospheric",
    "euphoric": "euphoric and blissful",
    "somber": "somber and reflective",
    "intense": "intense and powerful",
}


class PromptParser:
    """Converts structured music parameters into natural language prompts."""

    @staticmethod
    def build_prompt(
        text: str = "",
        genre: str = "",
        mood: str = "",
        tempo: int = 0,
        instruments: list[str] | None = None,
        key: str | None = None,
        time_sig: str | None = None,
    ) -> str:
        """Build a rich natural language prompt from structured fields.

        Combines the user's free-text prompt with structured musical parameters
        to create a descriptive prompt optimized for music generation models.

        Args:
            text: Free-text prompt describing desired music.
            genre: Target genre (pop, rock, lo-fi, etc.).
            mood: Target mood (happy, sad, energetic, etc.).
            tempo: Desired tempo in BPM (0 = unspecified).
            instruments: List of desired instruments.
            key: Musical key signature (e.g., "C major").
            time_sig: Time signature (e.g., "4/4").

        Returns:
            Descriptive natural language prompt string.

        Example:
            >>> PromptParser.build_prompt(
            ...     genre="lo-fi", mood="melancholic", tempo=75,
            ...     instruments=["piano", "vinyl crackle"]
            ... )
            'melancholic lo-fi hip hop beat at 75 BPM with soft piano chords and vinyl crackle atmosphere'
        """
        # If we have a text prompt but no structured fields, return it directly
        has_structured = bool(genre or mood or tempo or instruments or key or time_sig)
        if text and not has_structured:
            return text.strip()

        parts: list[str] = []

        # Build the core description: [mood] [genre descriptor]
        genre_lower = genre.strip().lower() if genre else ""
        mood_lower = mood.strip().lower() if mood else ""

        genre_desc = _GENRE_DESCRIPTORS.get(genre_lower, f"{genre} track" if genre else "")
        mood_adj = _MOOD_ADJECTIVES.get(mood_lower, mood_lower if mood_lower else "")

        if mood_adj and genre_desc:
            parts.append(f"{mood_adj} {genre_desc}")
        elif genre_desc:
            parts.append(genre_desc)
        elif mood_adj:
            parts.append(f"{mood_adj} music")

        # Tempo
        if tempo and tempo > 0:
            parts.append(f"at {tempo} BPM")

        # Key signature
        if key:
            parts.append(f"in {key}")

        # Time signature
        if time_sig:
            parts.append(f"in {time_sig} time")

        # Instruments with descriptive language
        if instruments:
            instrument_phrases = [
                _describe_instrument(inst.strip()) for inst in instruments
            ]
            if len(instrument_phrases) == 1:
                parts.append(f"with {instrument_phrases[0]}")
            elif len(instrument_phrases) == 2:
                parts.append(f"with {instrument_phrases[0]} and {instrument_phrases[1]}")
            else:
                joined = ", ".join(instrument_phrases[:-1])
                parts.append(f"with {joined}, and {instrument_phrases[-1]}")

        structured_prompt = " ".join(parts)

        # Combine with free-text if both exist
        if text and structured_prompt:
            return f"{structured_prompt}, {text.strip()}"
        elif structured_prompt:
            return structured_prompt
        elif text:
            return text.strip()
        else:
            logger.warning("No prompt parameters provided, using default")
            return "a short piece of instrumental music"


def _describe_instrument(instrument: str) -> str:
    """Add descriptive language to an instrument name.

    Args:
        instrument: Raw instrument name.

    Returns:
        Instrument with descriptive qualifier.
    """
    descriptors: dict[str, str] = {
        "piano": "soft piano chords",
        "guitar": "gentle guitar melodies",
        "acoustic guitar": "warm acoustic guitar picking",
        "electric guitar": "electric guitar riffs",
        "bass": "deep bass lines",
        "drums": "steady drum patterns",
        "synth": "lush synthesizer pads",
        "synthesizer": "lush synthesizer pads",
        "strings": "sweeping string arrangements",
        "violin": "expressive violin lines",
        "cello": "rich cello tones",
        "flute": "airy flute passages",
        "saxophone": "smooth saxophone melodies",
        "trumpet": "bright trumpet accents",
        "organ": "warm organ chords",
        "harp": "delicate harp arpeggios",
        "vinyl crackle": "vinyl crackle atmosphere",
        "rain": "gentle rain ambience",
        "pad": "atmospheric pad textures",
        "bells": "shimmering bell tones",
        "choir": "ethereal choir voices",
        "marimba": "melodic marimba patterns",
        "percussion": "crisp percussion elements",
        "808": "booming 808 bass",
        "hi-hat": "crisp hi-hat patterns",
        "kick": "punchy kick drum",
        "snare": "tight snare hits",
        "woodwinds": "warm woodwind harmonies",
        "brass": "bold brass sections",
    }
    lower = instrument.lower()
    return descriptors.get(lower, instrument)
