"""
Lightweight lyrics generator for TuneForge miners.

Uses a small text model (GPT-2) to generate contextual lyrics from
a music generation prompt. Runs on the miner's GPU alongside the
music model. Also extracts genre/mood from free-text prompts.
"""

import re
import time
from typing import Any

import torch
from loguru import logger


# Genre keywords for extraction
_GENRE_KEYWORDS: dict[str, list[str]] = {
    "reggae": ["reggae", "bob marley", "dancehall", "ska", "dub"],
    "rock": ["rock", "guitar solo", "punk", "grunge", "hard rock"],
    "pop": ["pop", "catchy", "radio hit", "mainstream"],
    "jazz": ["jazz", "swing", "bebop", "smooth jazz", "scat"],
    "blues": ["blues", "12-bar", "delta blues", "bb king"],
    "hip-hop": ["hip-hop", "hip hop", "rap", "trap", "boom bap", "mc"],
    "electronic": ["electronic", "edm", "techno", "house", "trance", "synth"],
    "r&b": ["r&b", "rnb", "soul", "motown", "neo soul"],
    "classical": ["classical", "orchestral", "symphony", "concerto", "chamber"],
    "folk": ["folk", "acoustic", "singer-songwriter", "bluegrass"],
    "metal": ["metal", "heavy metal", "death metal", "thrash"],
    "country": ["country", "nashville", "western", "honky tonk"],
    "ambient": ["ambient", "atmospheric", "drone", "soundscape"],
    "funk": ["funk", "funky", "groove", "bass groove"],
    "latin": ["latin", "salsa", "bossa nova", "samba", "cumbia"],
    "cinematic": ["cinematic", "film score", "epic", "trailer"],
    "lo-fi": ["lo-fi", "lofi", "chill", "study beats"],
}

# Mood keywords for extraction
_MOOD_KEYWORDS: dict[str, list[str]] = {
    "happy": ["happy", "joyful", "upbeat", "cheerful", "bright"],
    "sad": ["sad", "melancholic", "sorrowful", "heartbreak", "crying"],
    "energetic": ["energetic", "powerful", "driving", "intense", "high-energy"],
    "chill": ["chill", "relaxed", "laid-back", "mellow", "calm"],
    "dark": ["dark", "brooding", "ominous", "sinister"],
    "romantic": ["romantic", "love", "sensual", "intimate", "tender"],
    "aggressive": ["aggressive", "angry", "fierce", "rage"],
    "dreamy": ["dreamy", "ethereal", "floating", "hazy"],
    "uplifting": ["uplifting", "inspiring", "hopeful", "triumphant"],
    "nostalgic": ["nostalgic", "wistful", "retro", "memory"],
    "epic": ["epic", "grandiose", "majestic", "heroic"],
    "peaceful": ["peaceful", "serene", "tranquil", "zen"],
}


def extract_genre(prompt: str) -> str | None:
    """Extract genre from a free-text prompt."""
    prompt_lower = prompt.lower()
    for genre, keywords in _GENRE_KEYWORDS.items():
        for kw in keywords:
            if kw in prompt_lower:
                return genre
    return None


def extract_mood(prompt: str) -> str | None:
    """Extract mood from a free-text prompt."""
    prompt_lower = prompt.lower()
    for mood, keywords in _MOOD_KEYWORDS.items():
        for kw in keywords:
            if kw in prompt_lower:
                return mood
    return None


def extract_theme(prompt: str) -> str:
    """Extract the thematic content from a prompt for lyrics generation."""
    # Remove musical instruction words, keep the meaning
    noise_words = [
        "generate", "create", "make", "produce", "compose", "write",
        "song", "track", "music", "beat", "tune", "piece",
        "with a", "in the style of", "style of", "like",
        "bpm", "tempo", "at", "in", "the",
    ]
    text = prompt
    for word in noise_words:
        text = re.sub(rf"\b{re.escape(word)}\b", " ", text, flags=re.IGNORECASE)
    # Clean up
    text = re.sub(r"\d+", "", text)  # Remove numbers (BPM, etc.)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else prompt


class LyricsGenerator:
    """Generates lyrics from a music prompt using GPT-2 small.

    Loads on first use (~500MB VRAM) and stays in memory.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False

    def detect_vocal_intent(self, prompt: str) -> bool:
        """Use GPT-2 to determine if the prompt implies vocals.

        Feeds the prompt into GPT-2 with a classification framing and
        checks if the model's continuation leans toward vocals or instrumental.
        """
        self.load()
        if not self._loaded:
            return False

        try:
            classification_prompt = (
                f'Music prompt: "{prompt}"\n'
                f"Does this music request include vocals or singing? Answer yes or no:"
            )

            inputs = self._tokenizer.encode(classification_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response[len(classification_prompt):].strip().lower()

            wants_vocals = answer.startswith("yes")
            logger.debug(f"Vocal intent detection: '{answer}' → {wants_vocals}")
            return wants_vocals

        except Exception as exc:
            logger.warning(f"Vocal intent detection failed: {exc}")
            return False

    def load(self) -> None:
        """Load GPT-2 small model."""
        if self._loaded:
            return

        logger.info("Loading lyrics generator (GPT-2 small)...")
        t0 = time.time()
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self._model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
            self._model.eval()
            self._loaded = True
            logger.info(f"Lyrics generator loaded in {time.time() - t0:.1f}s")
        except Exception as exc:
            logger.error(f"Failed to load lyrics generator: {exc}")
            self._loaded = False

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        prompt: str,
        genre: str | None = None,
        mood: str | None = None,
        duration_seconds: float = 60.0,
        num_lines: int | None = None,
    ) -> str:
        """Generate lyrics from a music prompt.

        Returns plain text lyrics (not LRC format — the caller handles timestamping).
        """
        self.load()

        if not self._loaded:
            # Fallback if model failed to load
            return self._fallback_lyrics(prompt, genre, duration_seconds)

        # Auto-extract genre/mood if not provided
        if not genre:
            genre = extract_genre(prompt)
        if not mood:
            mood = extract_mood(prompt)

        theme = extract_theme(prompt)

        # Build a prompt that guides GPT-2 to write song lyrics
        gen_prompt = self._build_generation_prompt(theme, genre, mood)

        if num_lines is None:
            num_lines = max(4, int(duration_seconds / 10))

        try:
            inputs = self._tokenizer.encode(gen_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    max_new_tokens=num_lines * 15,  # ~15 tokens per line
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.3,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the generated lyrics (after the prompt)
            lyrics = full_text[len(gen_prompt):].strip()

            # Clean up and limit lines
            lines = self._clean_lyrics(lyrics, num_lines)

            if not lines:
                return self._fallback_lyrics(prompt, genre, duration_seconds)

            return "\n".join(lines)

        except Exception as exc:
            logger.warning(f"Lyrics generation failed: {exc}")
            return self._fallback_lyrics(prompt, genre, duration_seconds)

    @staticmethod
    def _build_generation_prompt(theme: str, genre: str | None, mood: str | None) -> str:
        """Build a prompt that steers GPT-2 to write song lyrics."""
        parts = ["Write song lyrics"]
        if genre:
            parts.append(f"for a {genre} song")
        if mood:
            parts.append(f"with a {mood} feeling")
        if theme:
            parts.append(f"about {theme}")
        parts.append(":\n\n")
        return " ".join(parts)

    @staticmethod
    def _clean_lyrics(raw: str, max_lines: int) -> list[str]:
        """Clean up generated text into usable lyrics lines."""
        lines = []
        for line in raw.split("\n"):
            line = line.strip()
            # Skip empty, too short, or too long lines
            if not line or len(line) < 3 or len(line) > 80:
                continue
            # Skip lines that look like instructions or metadata
            if any(skip in line.lower() for skip in ["verse", "chorus", "bridge", "outro", "intro", "written by", "copyright"]):
                continue
            # Remove leading punctuation/numbers
            line = re.sub(r"^[\d\.\)\-\*]+\s*", "", line)
            if line:
                lines.append(line)
            if len(lines) >= max_lines:
                break
        return lines

    @staticmethod
    def _fallback_lyrics(prompt: str, genre: str | None, duration_seconds: float) -> str:
        """Simple fallback when GPT-2 is unavailable — use the prompt theme as lyrics."""
        theme = extract_theme(prompt)
        words = theme.split()

        # Repeat theme words as simple vocal phrases
        lines = []
        chunk_size = min(4, len(words))
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                lines.append(chunk.strip().capitalize())

        # Pad with simple vocal phrases if too few lines
        fillers = ["Oh oh oh", "Na na na", "La la la", "Yeah yeah"]
        while len(lines) < max(4, int(duration_seconds / 15)):
            lines.append(fillers[len(lines) % len(fillers)])

        return "\n".join(lines[:max(4, int(duration_seconds / 10))])
