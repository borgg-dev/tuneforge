"""
Musicality scorer for TuneForge.

Measures musical content quality using Music Information Retrieval (MIR)
techniques.  While the companion ``AudioQualityScorer`` validates basic
signal properties (harmonic ratio, onset density, spectral contrast, etc.),
this scorer evaluates higher-level musical structure: pitch coherence,
harmonic progressions, rhythmic groove, and arrangement sophistication.

All metrics target genuine musical content — well-composed music scores
highly while noise, silence, or trivially engineered signals do not.

Genre-aware: accepts an optional genre string to adjust targets via
``GenreProfile`` (e.g. ambient music is not penalized for lacking beats).
"""

import numpy as np
from loguru import logger

from tuneforge.scoring.chord_coherence import ChordCoherenceScorer
from tuneforge.scoring.genre_profiles import GenreProfile, get_genre_profile

# ---------------------------------------------------------------------------
# Sub-metric weights (must sum to 1.0)
# ---------------------------------------------------------------------------
MUSICALITY_WEIGHTS: dict[str, float] = {
    "pitch_stability": 0.25,
    "harmonic_progression": 0.20,
    "chord_coherence": 0.15,
    "rhythmic_groove": 0.22,
    "arrangement_sophistication": 0.18,
}


class MusicalityScorer:
    """Assess musical content quality using MIR analysis."""

    def __init__(self) -> None:
        self._chord = ChordCoherenceScorer()

    def score(self, audio: np.ndarray, sr: int, genre: str = "") -> dict[str, float]:
        """
        Compute per-metric musicality scores.

        Args:
            audio: 1-D (or 2-D multi-channel) float waveform.
            sr: Sample rate in Hz.
            genre: Optional genre string for genre-aware target adjustment.

        Returns:
            Dict with keys matching ``MUSICALITY_WEIGHTS``.
            All values in [0, 1].
        """
        try:
            import librosa

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            profile = get_genre_profile(genre) if genre else GenreProfile(family="default")

            # Compute chord coherence via dedicated scorer
            chord_scores = self._chord.score(audio, sr)
            chord_coherence = self._chord.aggregate(chord_scores)

            return {
                "pitch_stability": self._score_pitch_stability(audio, sr, librosa, profile),
                "harmonic_progression": self._score_harmonic_progression(audio, sr, librosa),
                "chord_coherence": chord_coherence,
                "rhythmic_groove": self._score_rhythmic_groove(audio, sr, librosa, profile),
                "arrangement_sophistication": self._score_arrangement_sophistication(audio, sr, librosa, profile),
            }
        except Exception as exc:
            logger.error(f"Musicality scoring failed: {exc}")
            return {k: 0.0 for k in MUSICALITY_WEIGHTS}

    def aggregate(self, scores: dict[str, float]) -> float:
        """
        Weighted aggregation of per-metric scores.

        Args:
            scores: Dict from ``score()``.

        Returns:
            Aggregate musicality score in [0, 1].
        """
        total = 0.0
        for metric, weight in MUSICALITY_WEIGHTS.items():
            total += scores.get(metric, 0.0) * weight
        return float(np.clip(total, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_pitch_stability(audio: np.ndarray, sr: int, librosa, profile: GenreProfile) -> float:
        """
        Score based on musical pitch quality across the audio signal.

        Uses ``librosa.pyin`` to track f0 and then evaluates two aspects:

        * **Voiced ratio** — fraction of frames with detectable pitch.
          Musical audio has more pitched content than noise.
        * **Interval musicality** — fraction of pitch intervals that fall
          within musically meaningful ranges (0-7 semitones).  Musical
          melodies concentrate on small intervals (steps) with occasional
          larger leaps; noise produces random jumps.

        The ``pitch_range_tolerance`` from the genre profile controls how
        tolerant the metric is of wide pitch ranges — classical and jazz
        allow wider ranges than ambient or hip-hop.

        Returns 0.0 for silence or very short audio (< 0.5 s).
        """
        try:
            duration = len(audio) / sr
            if duration < 0.5:
                return 0.0

            if float(np.max(np.abs(audio))) < 1e-6:
                return 0.0

            # HPSS: isolate harmonic content for better f0 tracking
            # in polyphonic audio (drums/bass confuse monophonic pyin).
            audio_harmonic = librosa.effects.harmonic(y=audio)

            f0, voiced_flag, _ = librosa.pyin(
                audio_harmonic,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )

            if f0 is None or len(f0) == 0:
                return 0.0

            total_frames = len(voiced_flag)
            if total_frames == 0:
                return 0.0

            voiced_count = int(np.sum(voiced_flag))
            voiced_ratio = voiced_count / total_frames

            if voiced_count < 3:
                return 0.0

            # --- Interval musicality ---
            voiced_f0 = f0[voiced_flag]
            # Convert to semitones relative to first note
            semitones = 12.0 * np.log2(voiced_f0 / (voiced_f0[0] + 1e-8) + 1e-8)
            intervals = np.abs(np.diff(semitones))

            if len(intervals) == 0:
                return voiced_ratio * 0.5

            # Musical intervals: 0-7 semitones covers unison through perfect 5th
            # Genre tolerance widens the "acceptable" range
            max_musical_interval = 7.0 * profile.pitch_range_tolerance
            musical_fraction = float(np.mean(intervals <= max_musical_interval))

            # Penalize single-note drones: if all intervals are < 0.5 semitones,
            # the music lacks melodic movement
            near_unison_fraction = float(np.mean(intervals < 0.5))
            drone_penalty = 1.0
            if near_unison_fraction > 0.9:
                drone_penalty = 0.5  # mostly one note

            score = voiced_ratio * musical_fraction * drone_penalty
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_harmonic_progression(audio: np.ndarray, sr: int, librosa) -> float:
        """
        Score based on harmonic (chord) progression quality.

        Uses ``librosa.feature.chroma_cqt`` to extract chroma features, then
        segments them into ~2-second windows.  For each window the dominant
        chroma bin approximates the chord root.

        Two sub-scores are combined equally:

        * **Variety** — number of distinct dominant chromas, scored via a
          bell curve centred at 4-6 distinct chromas per 10 seconds.  Too
          few is monotonous; too many is chaotic.
        * **Smoothness** — mean cosine similarity between consecutive chroma
          segments, scored via a bell curve centred at 0.6.  Too high means
          no progression; too low means random jumps.

        Returns 0.0 for silence or noise with no harmonic content.
        """
        try:
            duration = len(audio) / sr
            if duration < 0.5:
                return 0.0

            if float(np.max(np.abs(audio))) < 1e-6:
                return 0.0

            from scipy.spatial.distance import cosine as cosine_dist

            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
            if chroma.shape[1] < 4:
                return 0.0

            # Segment into ~2-second windows
            hop_length = 512  # librosa default
            frames_per_sec = sr / hop_length
            frames_per_segment = max(1, int(2.0 * frames_per_sec))
            n_frames = chroma.shape[1]

            segments = []
            for start in range(0, n_frames, frames_per_segment):
                end = min(start + frames_per_segment, n_frames)
                seg = chroma[:, start:end].mean(axis=1)
                segments.append(seg)

            if len(segments) < 2:
                return 0.0

            # Dominant chroma per segment
            dominant_chromas = [int(np.argmax(seg)) for seg in segments]
            n_distinct = len(set(dominant_chromas))

            # Variety score: one-sided minimum floor (not bell curve)
            # At least 3 distinct chromas per 10s, capped at 12
            min_expected = max(3.0 * (duration / 10.0), 2.0)
            min_expected = min(min_expected, 12.0)
            # Continuous: ramp from 0 to 0.67 at threshold, then 0.67 to 1.0 above
            ratio = n_distinct / max(min_expected, 1.0)
            variety_score = float(np.clip(ratio / 1.5, 0.0, 1.0))

            # Smoothness: mean cosine similarity between consecutive segments
            similarities = []
            for i in range(len(segments) - 1):
                a = segments[i]
                b = segments[i + 1]
                norm_a = float(np.linalg.norm(a))
                norm_b = float(np.linalg.norm(b))
                if norm_a < 1e-8 or norm_b < 1e-8:
                    similarities.append(0.0)
                else:
                    sim = float(np.dot(a, b) / (norm_a * norm_b))
                    similarities.append(sim)

            if len(similarities) == 0:
                return 0.0

            mean_sim = float(np.mean(similarities))
            # Continuous ramp: 0 at sim=0, 1.0 at sim=0.3, stays high above
            smoothness_score = float(np.clip(mean_sim / 0.3, 0.0, 1.0))

            score = 0.5 * variety_score + 0.5 * smoothness_score
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_rhythmic_groove(audio: np.ndarray, sr: int, librosa, profile: GenreProfile) -> float:
        """
        Score based on rhythmic regularity and beat strength.

        Uses ``librosa.beat.beat_track`` to detect beats and
        ``librosa.onset.onset_strength`` to measure onset energy at beat
        positions.

        Two sub-scores are combined equally:

        * **Regularity** — coefficient of variation (CV) of inter-beat
          intervals.  Lower CV means steadier rhythm.  CV >= 0.3 yields a
          score of 0.0.
        * **Strength** — mean onset strength at beat positions relative to
          the overall onset envelope maximum.

        The ``rhythmic_groove_floor`` from the genre profile sets a minimum
        score for genres where beats are optional (ambient, classical).
        """
        try:
            duration = len(audio) / sr
            if duration < 0.5:
                return 0.0

            if float(np.max(np.abs(audio))) < 1e-6:
                return 0.0

            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            tempo, beat_frames = librosa.beat.beat_track(
                y=audio, sr=sr, onset_envelope=onset_env,
            )

            has_energy = float(np.max(np.abs(audio))) > 1e-4
            genre_floor = profile.rhythmic_groove_floor

            if len(beat_frames) < 2:
                # Genre-aware floor for beat-less audio
                if has_energy:
                    return max(0.15, genre_floor)
                return 0.0

            # Inter-beat intervals (in frames)
            ibis = np.diff(beat_frames).astype(np.float64)
            mean_ibi = float(np.mean(ibis))
            if mean_ibi < 1e-8:
                return 0.0

            cv = float(np.std(ibis) / mean_ibi)

            # Regularity: linear mapping from CV=0 (perfect) → 1.0, CV>=0.3 → 0.0
            regularity_score = max(0.0, 1.0 - cv / 0.3)

            # Beat strength: mean onset strength at beat positions vs max
            max_onset = float(np.max(onset_env)) + 1e-8
            valid_beats = beat_frames[beat_frames < len(onset_env)]
            if len(valid_beats) == 0:
                strength_score = 0.0
            else:
                beat_strengths = onset_env[valid_beats]
                strength_score = float(np.mean(beat_strengths) / max_onset)

            score = 0.5 * regularity_score + 0.5 * strength_score
            # Ensure genre floor is respected
            score = max(score, genre_floor)
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _score_arrangement_sophistication(audio: np.ndarray, sr: int, librosa, profile: GenreProfile) -> float:
        """
        Score based on timbral variation and structural coherence across
        sections of the audio.

        Splits the audio into N equal segments (4-8 depending on duration)
        and computes spectral centroid and MFCC statistics per segment.

        Two sub-scores are combined equally:

        * **Contrast** — standard deviation of spectral centroid means
          across segments, scored via a bell curve centred at moderate
          variation.  Flat signals or totally random spectra score lower.
        * **Coherence** — mean pairwise cosine similarity of MFCC vectors
          across segments, scored via a bell curve centred at 0.5.
          Evolving but related sections score highest.

        Returns 0.0 for silence or very short audio.
        """
        try:
            duration = len(audio) / sr
            if duration < 0.5:
                return 0.0

            if float(np.max(np.abs(audio))) < 1e-6:
                return 0.0

            # Determine number of segments (4-8 based on duration)
            n_segments = min(8, max(4, int(duration / 2.0)))
            seg_len = len(audio) // n_segments

            if seg_len < 1024:
                return 0.0

            centroid_means = []
            mfcc_means = []

            for i in range(n_segments):
                start = i * seg_len
                end = start + seg_len
                segment = audio[start:end]

                # Spectral centroid
                centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
                centroid_means.append(float(np.mean(centroid)))

                # MFCCs
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
                mfcc_means.append(mfcc.mean(axis=1))

            if len(centroid_means) < 2:
                return 0.0

            # Contrast: std of spectral centroid means
            centroid_arr = np.array(centroid_means)
            centroid_mean = float(np.mean(centroid_arr))
            if centroid_mean < 1e-8:
                return 0.0

            # Normalise std by mean to get a relative measure
            relative_std = float(np.std(centroid_arr) / (centroid_mean + 1e-8))
            # One-sided minimum floor: penalize only lack of variation
            # Minimum acceptable contrast varies by genre
            contrast_floor = profile.arrangement_contrast_target * 0.3
            # Continuous: ramp up through floor, continue rising above
            contrast_score = float(np.clip(
                0.5 + relative_std * 2.0 if relative_std >= contrast_floor
                else relative_std / (contrast_floor + 1e-8) * (0.5 + contrast_floor * 2.0),
                0.0, 1.0,
            ))

            # Coherence: mean pairwise cosine similarity of MFCC vectors
            similarities = []
            for i in range(len(mfcc_means)):
                for j in range(i + 1, len(mfcc_means)):
                    a = mfcc_means[i]
                    b = mfcc_means[j]
                    norm_a = float(np.linalg.norm(a))
                    norm_b = float(np.linalg.norm(b))
                    if norm_a < 1e-8 or norm_b < 1e-8:
                        similarities.append(0.0)
                    else:
                        sim = float(np.dot(a, b) / (norm_a * norm_b))
                        similarities.append(sim)

            if len(similarities) == 0:
                return 0.0

            mean_coherence = float(np.mean(similarities))
            # Continuous: ramp from 0 at coherence=0 to 0.6 at 0.2, then up to 1.0
            # At threshold=0.2: 0.4+0.2=0.6, and ramp gives 0.2/0.2*0.6=0.6 → continuous
            if mean_coherence >= 0.2:
                coherence_score = min(1.0, 0.4 + mean_coherence)
            else:
                coherence_score = max(0.0, mean_coherence / 0.2 * 0.6)

            score = 0.5 * contrast_score + 0.5 * coherence_score
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0
