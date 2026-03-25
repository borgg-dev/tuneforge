"""
Lyrics generator for TuneForge miners.

Uses Qwen3-0.6B (instruction-following LLM) to generate creative,
contextual lyrics from music prompts. Runs on CPU alongside the
music generation model on GPU.

Also provides genre/mood extraction from free-text prompts.
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


class LyricsGenerator:
    """Generates creative lyrics using Qwen3-0.6B (instruction-following LLM).

    Runs on CPU (~1.2GB RAM) to keep GPU free for music generation.
    """

    MODEL_ID: str = "Qwen/Qwen3-0.6B"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False

    def load(self) -> None:
        """Load Qwen3-0.6B model."""
        if self._loaded:
            return

        logger.info(f"Loading lyrics generator ({self.MODEL_ID})...")
        t0 = time.time()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID, trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            self._model.eval()
            self._loaded = True
            logger.info(f"Lyrics generator loaded in {time.time() - t0:.1f}s")
        except Exception as exc:
            logger.error(f"Failed to load lyrics generator: {exc}")
            self._loaded = False

    def unload(self) -> None:
        """Free memory."""
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
        """Generate creative lyrics from a music prompt.

        Returns lyrics with [Verse]/[Chorus] section markers.
        """
        self.load()

        if not self._loaded:
            return self._fallback_lyrics(prompt, genre, duration_seconds)

        if not genre:
            genre = extract_genre(prompt)
        if not mood:
            mood = extract_mood(prompt)

        if num_lines is None:
            num_lines = max(8, int(duration_seconds / 5))

        # Build chat messages for the instruction-following model
        system_msg = (
            "You are a professional songwriter. Write creative, poetic, "
            "and emotionally resonant song lyrics. Use vivid imagery and "
            "metaphors. Structure with [Verse] and [Chorus] sections. "
            "Output ONLY the lyrics, nothing else."
        )

        user_msg = f"Write {num_lines}-{num_lines + 4} lines of song lyrics"
        if genre:
            user_msg += f" for a {genre} song"
        if mood:
            user_msg += f" with a {mood} mood"
        user_msg += f".\n\nThe song is about: {prompt}"
        user_msg += "\n\nInclude [Verse] and [Chorus] markers. Be creative and poetic — do not just repeat the description."

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=num_lines * 20,
                    temperature=0.85,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode only the new tokens (skip the input)
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            lyrics = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Clean up
            lines = self._clean_lyrics(lyrics, num_lines + 8)
            if not lines:
                return self._fallback_lyrics(prompt, genre, duration_seconds)

            return "\n".join(lines)

        except Exception as exc:
            logger.warning(f"Lyrics generation failed: {exc}")
            return self._fallback_lyrics(prompt, genre, duration_seconds)

    @staticmethod
    def _clean_lyrics(raw: str, max_lines: int) -> list[str]:
        """Clean up generated text into usable lyrics lines."""
        lines = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line or len(line) < 2 or len(line) > 100:
                continue
            # Keep section markers
            if line.startswith("[") and line.endswith("]"):
                lines.append(line)
                continue
            # Skip metadata lines
            if any(skip in line.lower() for skip in [
                "written by", "copyright", "here are", "here's",
                "sure,", "of course", "certainly",
            ]):
                continue
            # Remove leading numbering
            line = re.sub(r"^[\d\.\)\-\*]+\s*", "", line)
            if line:
                lines.append(line)
            if len(lines) >= max_lines:
                break
        return lines

    @staticmethod
    def _fallback_lyrics(prompt: str, genre: str | None, duration_seconds: float) -> str:
        """Simple fallback when model is unavailable."""
        lines = [
            "[Verse]",
            "Dancing through the night",
            "Under neon lights so bright",
            "Feel the rhythm take control",
            "Music flowing through my soul",
            "",
            "[Chorus]",
            "We're alive, we're on fire",
            "Rising higher and higher",
            "Nothing's gonna stop us now",
            "We're making magic somehow",
            "",
            "[Verse]",
            "Lost in the melody",
            "Finding where I need to be",
            "Every beat a story told",
            "Worth more than silver or gold",
            "",
            "[Chorus]",
            "We're alive, we're on fire",
            "Rising higher and higher",
            "Nothing's gonna stop us now",
            "We're making magic somehow",
        ]
        return "\n".join(lines)
