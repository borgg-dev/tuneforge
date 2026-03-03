"""
Melody coherence scorer for TuneForge.

Evaluates the melodic quality of generated music by analysing pitch content
extracted via fundamental-frequency (f0) tracking.  Four complementary metrics
capture different facets of melodic quality:

* **interval_quality** — Do consecutive pitches form musically natural steps?
* **contour_quality** — Do phrases follow recognisable melodic shapes?
* **repetition_structure** — Is there structural repetition without monotony?
* **melodic_memorability** — Is the pitch material focused yet varied?

All metrics target genuine musical content and are designed to resist gaming
by trivially engineered signals (e.g. pure sine tones, noise bursts).
"""

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Sub-metric weights (must sum to 1.0)
# ---------------------------------------------------------------------------

MELODY_WEIGHTS: dict[str, float] = {
    "interval_quality": 0.30,
    "contour_quality": 0.25,
    "repetition_structure": 0.25,
    "melodic_memorability": 0.20,
}


class MelodyCoherenceScorer:
    """Assess melodic coherence of generated audio."""

    # Minimum audio duration (seconds) to attempt analysis
    _MIN_DURATION: float = 0.5
    # Amplitude below which audio is considered silence
    _SILENCE_THRESHOLD: float = 1e-6

    def score(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """
        Compute per-metric melody coherence scores.

        Args:
            audio: Waveform array (1-D or 2-D).
            sr: Sample rate in Hz.

        Returns:
            Dict with keys matching ``MELODY_WEIGHTS``.  All values in [0, 1].
        """
        try:
            import librosa

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            # --- Edge-case guards ---
            if np.max(np.abs(audio)) < self._SILENCE_THRESHOLD:
                return {k: 0.0 for k in MELODY_WEIGHTS}
            if len(audio) / sr < self._MIN_DURATION:
                return {k: 0.0 for k in MELODY_WEIGHTS}

            return {
                "interval_quality": self._score_interval_quality(audio, sr, librosa),
                "contour_quality": self._score_contour_quality(audio, sr, librosa),
                "repetition_structure": self._score_repetition_structure(audio, sr, librosa),
                "melodic_memorability": self._score_melodic_memorability(audio, sr, librosa),
            }
        except Exception as exc:
            logger.error(f"Melody coherence scoring failed: {exc}")
            return {k: 0.0 for k in MELODY_WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        """
        Weighted aggregation of per-metric scores.

        Args:
            scores: Dict from ``score()``.

        Returns:
            Aggregate melody coherence score in [0, 1].
        """
        total = 0.0
        for metric, weight in MELODY_WEIGHTS.items():
            total += scores.get(metric, 0.0) * weight
        return float(np.clip(total, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_interval_quality(audio: np.ndarray, sr: int, librosa) -> float:
        """
        Score based on the distribution of melodic intervals.

        Musical melodies concentrate on small pitch intervals (steps of 0-2
        semitones) with occasional larger leaps.  The score is the fraction of
        consecutive-frame intervals that fall within the musical range of 0-7
        semitones.

        Returns 0.0 if fewer than 2 voiced frames are detected.
        """
        try:
            fmin = librosa.note_to_hz("C2")
            fmax = librosa.note_to_hz("C7")

            # HPSS: isolate harmonic content for polyphonic audio
            audio_harmonic = librosa.effects.harmonic(y=audio)

            f0, voiced_flag, _ = librosa.pyin(
                audio_harmonic, fmin=fmin, fmax=fmax, sr=sr,
            )

            # Keep only voiced frames
            voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
            if len(voiced) < 2:
                return 0.0

            # Convert to semitones relative to first voiced pitch
            reference = voiced[0]
            semitones = 12.0 * np.log2(voiced / reference + 1e-10)

            # Consecutive intervals (absolute semitone difference)
            intervals = np.abs(np.diff(semitones))

            if len(intervals) == 0:
                return 0.0

            # Fraction within musical range (0-7 semitones)
            musical_fraction = float(np.mean(intervals <= 7.0))
            return float(np.clip(musical_fraction, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_contour_quality(audio: np.ndarray, sr: int, librosa) -> float:
        """
        Score based on phrase-level melodic contour shape.

        Voiced f0 is segmented into phrases (split at unvoiced gaps longer
        than 0.2 seconds).  Each phrase is classified as arch, ascending,
        descending, or flat — all considered valid musical shapes.

        Additionally, phrase-level autocorrelation of the f0 contour is used
        to reward moderate periodicity (0.3-0.7), which indicates musically
        coherent repetition.

        Returns 0.0 if audio is shorter than 0.5 seconds or no phrases are
        found.
        """
        try:
            duration = len(audio) / sr
            if duration < 0.5:
                return 0.0

            fmin = librosa.note_to_hz("C2")
            fmax = librosa.note_to_hz("C7")

            # HPSS: isolate harmonic content for polyphonic audio
            audio_harmonic = librosa.effects.harmonic(y=audio)

            f0, voiced_flag, _ = librosa.pyin(
                audio_harmonic, fmin=fmin, fmax=fmax, sr=sr,
            )

            if f0 is None or len(f0) == 0:
                return 0.0

            # Determine hop length used by pyin (default is sr // 4 in
            # older librosa, 512 in newer — compute from output length)
            hop_duration = duration / len(f0) if len(f0) > 1 else 0.01
            gap_threshold_frames = max(1, int(0.2 / hop_duration))

            # Segment into phrases
            phrases: list[np.ndarray] = []
            current_phrase: list[float] = []
            gap_count = 0

            for i, val in enumerate(f0):
                if np.isnan(val):
                    gap_count += 1
                    if gap_count >= gap_threshold_frames and current_phrase:
                        phrases.append(np.array(current_phrase))
                        current_phrase = []
                else:
                    gap_count = 0
                    current_phrase.append(val)

            if current_phrase:
                phrases.append(np.array(current_phrase))

            if not phrases:
                return 0.0

            # Classify each phrase contour
            identifiable = 0
            for phrase in phrases:
                if len(phrase) < 3:
                    continue
                mid = len(phrase) // 2
                first_half_trend = phrase[mid] - phrase[0]
                second_half_trend = phrase[-1] - phrase[mid]

                # Arch: rises then falls
                is_arch = first_half_trend > 0 and second_half_trend < 0
                # Ascending
                is_ascending = phrase[-1] > phrase[0]
                # Descending
                is_descending = phrase[-1] < phrase[0]
                # Flat: small total variation
                pitch_range = np.max(phrase) - np.min(phrase)
                is_flat = pitch_range < (phrase[0] * 0.05 + 1e-3)

                if is_arch or is_ascending or is_descending or is_flat:
                    identifiable += 1

            shape_score = identifiable / len(phrases) if phrases else 0.0

            # Autocorrelation of the voiced f0 contour for periodicity
            voiced_f0 = f0[~np.isnan(f0)]
            if len(voiced_f0) > 10:
                centred = voiced_f0 - np.mean(voiced_f0)
                autocorr = np.correlate(centred, centred, mode="full")
                autocorr = autocorr[len(autocorr) // 2:]
                if autocorr[0] > 0:
                    autocorr = autocorr / autocorr[0]
                # Find peak autocorrelation in the mid-range
                # (skip lag 0, look at lags 10% to 50% of length)
                start = max(1, len(autocorr) // 10)
                end = len(autocorr) // 2
                if start < end:
                    segment = autocorr[start:end]
                    peak_corr = float(np.max(segment))
                else:
                    peak_corr = 0.0
                # Moderate periodicity (0.3-0.7) scores highest
                periodicity_score = 1.0 - abs(peak_corr - 0.5) / 0.5
                periodicity_score = float(np.clip(periodicity_score, 0.0, 1.0))
            else:
                periodicity_score = 0.0

            combined = 0.6 * shape_score + 0.4 * periodicity_score
            return float(np.clip(combined, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_repetition_structure(audio: np.ndarray, sr: int, librosa) -> float:
        """
        Score based on structural repetition using chroma self-similarity.

        Audio is segmented into ~2-second windows, mean chroma is computed
        per window, and the cosine similarity between all off-diagonal window
        pairs is calculated.  Moderate repetition (20-50% similar pairs)
        scores highest via a Gaussian bell curve centred at 35%.

        Requires at least 2 seconds of audio (at least 2 windows).
        """
        try:
            duration = len(audio) / sr
            if duration < 2.0:
                return 0.0

            # Chroma features
            hop_length = 512
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
            if chroma.shape[1] < 4:
                return 0.0

            # Segment into ~2-second windows
            frames_per_second = sr / hop_length
            window_frames = max(1, int(2.0 * frames_per_second))
            n_windows = chroma.shape[1] // window_frames

            if n_windows < 2:
                return 0.0

            # Mean chroma per window
            window_chromas = []
            for w in range(n_windows):
                start = w * window_frames
                end = start + window_frames
                mean_chroma = chroma[:, start:end].mean(axis=1)
                norm = np.linalg.norm(mean_chroma) + 1e-8
                window_chromas.append(mean_chroma / norm)

            window_chromas = np.array(window_chromas)  # (n_windows, 12)

            # Cosine similarity matrix
            sim_matrix = window_chromas @ window_chromas.T

            # Count off-diagonal pairs with similarity > 0.8
            n_pairs = 0
            n_similar = 0
            for i in range(n_windows):
                for j in range(i + 1, n_windows):
                    n_pairs += 1
                    if sim_matrix[i, j] > 0.8:
                        n_similar += 1

            if n_pairs == 0:
                return 0.0

            ratio = n_similar / n_pairs

            # Bell curve: moderate repetition (35%) scores highest
            score = float(np.exp(-8.0 * (ratio - 0.35) ** 2))
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_melodic_memorability(audio: np.ndarray, sr: int, librosa) -> float:
        """
        Score based on pitch entropy and intervallic consistency.

        * **Pitch entropy:** Voiced f0 values are quantised to MIDI note
          numbers and a discrete entropy is computed.  Moderate entropy
          (~0.5 normalised) scores highest — very low entropy (single note)
          is boring, very high entropy is random.
        * **Intervallic consistency:** The fraction of all intervals
          accounted for by the single most common interval.  Higher
          consistency indicates a stronger recurring motif.

        Combined as 0.5 * entropy_score + 0.5 * consistency_score.
        Returns 0.0 if fewer than 5 voiced frames are detected.
        """
        try:
            fmin = librosa.note_to_hz("C2")
            fmax = librosa.note_to_hz("C7")

            # HPSS: isolate harmonic content for polyphonic audio
            audio_harmonic = librosa.effects.harmonic(y=audio)

            f0, voiced_flag, _ = librosa.pyin(
                audio_harmonic, fmin=fmin, fmax=fmax, sr=sr,
            )

            voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
            if len(voiced) < 5:
                return 0.0

            # --- Pitch entropy ---
            # Quantise to MIDI notes
            midi_notes = np.round(12.0 * np.log2(voiced / 440.0 + 1e-10) + 69).astype(int)
            unique, counts = np.unique(midi_notes, return_counts=True)
            probs = counts / counts.sum()

            entropy = -float(np.sum(probs * np.log2(probs + 1e-10)))
            max_entropy = np.log2(len(unique)) if len(unique) > 1 else 1.0
            normalised_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Bell curve centred at 0.5
            entropy_score = float(np.exp(-8.0 * (normalised_entropy - 0.5) ** 2))

            # --- Intervallic consistency ---
            semitones = 12.0 * np.log2(voiced / voiced[0] + 1e-10)
            intervals = np.round(np.abs(np.diff(semitones))).astype(int)

            if len(intervals) == 0:
                return float(np.clip(0.5 * entropy_score, 0.0, 1.0))

            unique_int, int_counts = np.unique(intervals, return_counts=True)
            most_common_fraction = float(np.max(int_counts)) / len(intervals)
            consistency_score = float(np.clip(most_common_fraction, 0.0, 1.0))

            combined = 0.5 * entropy_score + 0.5 * consistency_score
            return float(np.clip(combined, 0.0, 1.0))
        except Exception:
            return 0.0
