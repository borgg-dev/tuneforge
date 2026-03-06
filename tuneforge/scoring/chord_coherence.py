"""
Chord coherence scorer for TuneForge.

Detects chord progressions via chroma template matching and scores
harmonic quality: clarity, transition smoothness, and variety.
"""

import numpy as np
from loguru import logger


# 12 major + 12 minor triad templates (chroma vectors)
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def _build_chord_templates() -> dict[str, np.ndarray]:
    """Build normalized chord templates for all 24 major/minor triads."""
    templates = {}
    # Major triad intervals: root, major third (+4), perfect fifth (+7)
    # Minor triad intervals: root, minor third (+3), perfect fifth (+7)
    for i, note in enumerate(_NOTE_NAMES):
        major = np.zeros(12)
        major[i] = 1.0
        major[(i + 4) % 12] = 1.0
        major[(i + 7) % 12] = 1.0
        major /= np.linalg.norm(major)
        templates[f"{note}_major"] = major

        minor = np.zeros(12)
        minor[i] = 1.0
        minor[(i + 3) % 12] = 1.0
        minor[(i + 7) % 12] = 1.0
        minor /= np.linalg.norm(minor)
        templates[f"{note}_minor"] = minor

    return templates

CHORD_TEMPLATES = _build_chord_templates()

# Circle of fifths for transition scoring
_COF = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]
_COF_INDEX = {note: i for i, note in enumerate(_COF)}
# Map enharmonics
_COF_INDEX["Db"] = _COF_INDEX["C#"]
_COF_INDEX["Ab"] = _COF_INDEX["G#"]
_COF_INDEX["Eb"] = _COF_INDEX["D#"]
_COF_INDEX["Bb"] = _COF_INDEX["A#"]
_COF_INDEX["Gb"] = _COF_INDEX["F#"]


class ChordCoherenceScorer:
    """Score harmonic progression quality via chord recognition."""

    WEIGHTS: dict[str, float] = {
        "chord_clarity": 0.30,
        "transition_smoothness": 0.35,
        "progression_variety": 0.35,
    }

    def score(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Compute chord coherence sub-scores."""
        try:
            import librosa

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            duration = len(audio) / sr
            if duration < 2.0:
                return {k: 0.0 for k in self.WEIGHTS}

            # Extract chroma features
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=512)

            # Segment into ~2s windows
            frames_per_window = max(1, int(2.0 * sr / 512))
            n_windows = max(1, chroma.shape[1] // frames_per_window)

            chords = []
            confidences = []

            for w in range(n_windows):
                start = w * frames_per_window
                end = min((w + 1) * frames_per_window, chroma.shape[1])
                window_chroma = np.mean(chroma[:, start:end], axis=1)

                chord, confidence = self._match_chord(window_chroma)
                chords.append(chord)
                confidences.append(confidence)

            if not chords:
                return {k: 0.0 for k in self.WEIGHTS}

            return {
                "chord_clarity": self._score_clarity(confidences),
                "transition_smoothness": self._score_transitions(chords),
                "progression_variety": self._score_variety(chords, duration),
            }

        except Exception as exc:
            logger.error("Chord coherence scoring failed: {}", exc)
            return {k: 0.0 for k in self.WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        total = sum(self.WEIGHTS[k] * scores.get(k, 0.0) for k in self.WEIGHTS)
        return float(np.clip(total, 0.0, 1.0))

    @staticmethod
    def _match_chord(chroma_vector: np.ndarray) -> tuple[str, float]:
        """Template matching. Returns (chord_name, confidence)."""
        norm = np.linalg.norm(chroma_vector)
        if norm < 1e-8:
            return "none", 0.0
        normalized = chroma_vector / norm

        best_chord = "none"
        best_sim = -1.0

        for name, template in CHORD_TEMPLATES.items():
            sim = float(np.dot(normalized, template))
            if sim > best_sim:
                best_sim = sim
                best_chord = name

        return best_chord, max(0.0, best_sim)

    @staticmethod
    def _score_clarity(confidences: list[float]) -> float:
        """Average chord detection confidence."""
        if not confidences:
            return 0.0
        avg = float(np.mean(confidences))
        # Remap: 0.5 baseline (random match), 0.9+ is excellent
        return float(np.clip((avg - 0.5) / 0.4, 0.0, 1.0))

    @staticmethod
    def _score_transitions(chords: list[str]) -> float:
        """Score chord transitions by circle-of-fifths proximity."""
        if len(chords) < 2:
            return 0.5

        scores = []
        for i in range(len(chords) - 1):
            a, b = chords[i], chords[i + 1]
            scores.append(ChordCoherenceScorer._transition_quality(a, b))

        return float(np.clip(np.mean(scores), 0.0, 1.0))

    @staticmethod
    def _transition_quality(chord_a: str, chord_b: str) -> float:
        """Score a single chord transition."""
        if chord_a == "none" or chord_b == "none":
            return 0.3
        if chord_a == chord_b:
            return 0.7  # Same chord — ok but not interesting

        root_a = chord_a.split("_")[0]
        root_b = chord_b.split("_")[0]
        mode_a = chord_a.split("_")[1] if "_" in chord_a else "major"
        mode_b = chord_b.split("_")[1] if "_" in chord_b else "major"

        # Relative major/minor
        if root_a == root_b and mode_a != mode_b:
            return 1.0

        idx_a = _COF_INDEX.get(root_a, -1)
        idx_b = _COF_INDEX.get(root_b, -1)
        if idx_a < 0 or idx_b < 0:
            return 0.3

        dist = min(abs(idx_a - idx_b), 12 - abs(idx_a - idx_b))
        # Adjacent on CoF = excellent, far = poor
        if dist <= 1:
            return 1.0
        elif dist <= 2:
            return 0.7
        elif dist <= 3:
            return 0.5
        else:
            return 0.2

    @staticmethod
    def _score_variety(chords: list[str], duration: float) -> float:
        """Score chord progression variety. Bell curve at 4-6 unique chords per 30s."""
        unique = len(set(c for c in chords if c != "none"))
        # Normalize to 30s basis
        if duration > 0:
            normalized = unique * (30.0 / duration)
        else:
            normalized = unique

        # Bell curve centered at 5
        return float(np.clip(np.exp(-0.3 * (normalized - 5.0) ** 2), 0.0, 1.0))
