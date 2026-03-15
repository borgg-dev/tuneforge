"""
Vocal and lyrics quality scorer for TuneForge.

Evaluates vocal performance and lyrical content of generated music using
spectral analysis and optional speech recognition.  Five complementary
sub-metrics capture different facets of vocal quality:

* **vocal_clarity** -- Signal-to-noise ratio in vocal band via HPSS.
* **lyrics_intelligibility** -- Speech recognition confidence via Whisper.
* **vocal_pitch_quality** -- Pitch stability, vibrato, and range via pyin.
* **vocal_expressiveness** -- Dynamic variation in the vocal band over time.
* **sibilance_control** -- Penalizes excessive high-frequency sibilant energy.

Genre-aware: instrumental genres (ambient, electronic, classical) receive a
neutral 0.5 score on all metrics so that vocal absence does not penalize
genuinely instrumental music.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from loguru import logger



# ---------------------------------------------------------------------------
# Sub-metric weights (must sum to 1.0)
# ---------------------------------------------------------------------------

VOCAL_LYRICS_WEIGHTS: dict[str, float] = {
    "vocal_clarity": 0.30,
    "lyrics_intelligibility": 0.25,
    "vocal_pitch_quality": 0.20,
    "vocal_expressiveness": 0.15,
    "sibilance_control": 0.10,
}

# Neutral scores returned when vocal evaluation is not applicable
_NEUTRAL: dict[str, float] = {k: 0.5 for k in VOCAL_LYRICS_WEIGHTS}


class VocalLyricsScorer:
    """Assess vocal performance and lyrical quality of generated audio."""

    # Minimum audio duration (seconds) to attempt analysis
    _MIN_DURATION: float = 1.0
    # Amplitude below which audio is considered silence
    _SILENCE_THRESHOLD: float = 1e-6

    # Whisper model lazy-loading state
    _whisper_model: Any = None
    _whisper_lock: threading.Lock = threading.Lock()
    _whisper_available: bool | None = None  # None = not yet checked

    def score(
        self,
        audio: np.ndarray,
        sr: int,
        genre: str = "",
        expected_lyrics: str = "",
        vocals_requested: bool = False,
    ) -> dict[str, float]:
        """
        Compute per-metric vocal/lyrics quality scores.

        Args:
            audio: Waveform array (1-D or 2-D).
            sr: Sample rate in Hz.
            genre: Optional genre string for genre-aware scoring.
            expected_lyrics: Optional reference lyrics for WER evaluation.
            vocals_requested: If True, always evaluate vocals regardless of genre.

        Returns:
            Dict with keys matching ``VOCAL_LYRICS_WEIGHTS``.
            All values in [0, 1].
        """
        try:
            import librosa

            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)

            # --- Edge-case guards ---
            if np.max(np.abs(audio)) < self._SILENCE_THRESHOLD:
                if vocals_requested:
                    return {k: 0.1 for k in VOCAL_LYRICS_WEIGHTS}
                return dict(_NEUTRAL)
            if len(audio) / sr < self._MIN_DURATION:
                if vocals_requested:
                    return {k: 0.1 for k in VOCAL_LYRICS_WEIGHTS}
                return dict(_NEUTRAL)

            # Pre-compute HPSS harmonic signal (shared across metrics)
            harmonic = librosa.effects.hpss(audio)[0]

            # --- Prompt gate ---
            if not vocals_requested:
                # Check if unwanted vocals are present; score 0 if so
                presence = self._detect_vocal_presence(harmonic, sr, librosa)
                if presence > 0.4:
                    return {k: 0.0 for k in VOCAL_LYRICS_WEIGHTS}
                return dict(_NEUTRAL)

            return {
                "vocal_clarity": self._score_vocal_clarity(
                    audio, harmonic, sr, librosa,
                ),
                "lyrics_intelligibility": self._score_lyrics_intelligibility(
                    audio, sr, expected_lyrics,
                ),
                "vocal_pitch_quality": self._score_vocal_pitch_quality(
                    harmonic, sr, librosa,
                ),
                "vocal_expressiveness": self._score_vocal_expressiveness(
                    audio, sr, librosa,
                ),
                "sibilance_control": self._score_sibilance_control(
                    audio, sr, librosa,
                ),
            }
        except Exception as exc:
            logger.error(f"Vocal/lyrics scoring failed: {exc}")
            return dict(_NEUTRAL)

    def aggregate(self, scores: dict[str, float]) -> float:
        """
        Weighted aggregation of per-metric scores.

        Args:
            scores: Dict from ``score()``.

        Returns:
            Aggregate vocal/lyrics quality score in [0, 1].
        """
        total = 0.0
        for metric, weight in VOCAL_LYRICS_WEIGHTS.items():
            total += scores.get(metric, 0.5) * weight
        return float(np.clip(total, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Whisper lazy loader
    # ------------------------------------------------------------------

    @classmethod
    def _load_whisper(cls) -> Any:
        """
        Lazy-load the Whisper model (tiny) on first use.

        Thread-safe via a lock.  If the ``whisper`` package is not
        installed, sets ``_whisper_available = False`` and all future
        calls return ``None`` immediately.
        """
        if cls._whisper_available is False:
            return None
        if cls._whisper_model is not None:
            return cls._whisper_model

        with cls._whisper_lock:
            # Double-check after acquiring lock
            if cls._whisper_model is not None:
                return cls._whisper_model
            if cls._whisper_available is False:
                return None

            try:
                import whisper  # type: ignore[import-untyped]

                cls._whisper_model = whisper.load_model("tiny")
                cls._whisper_available = True
                logger.info("Whisper (tiny) model loaded for lyrics scoring")
                return cls._whisper_model
            except Exception as exc:
                cls._whisper_available = False
                logger.warning(
                    f"Whisper not available, lyrics_intelligibility will "
                    f"return neutral 0.5: {exc}"
                )
                return None

    # ------------------------------------------------------------------
    # Vocal detection (for unwanted-vocal penalty)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_vocal_presence(harmonic: np.ndarray, sr: int, librosa) -> float:
        """Return raw vocal-band energy ratio (0-1). Higher = more vocal content."""
        try:
            n_fft = 2048
            S = np.abs(librosa.stft(harmonic, n_fft=n_fft))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            vocal_mask = (freqs >= 300.0) & (freqs <= 4000.0)
            total_energy = float(np.sum(S ** 2))
            if total_energy < 1e-10:
                return 0.0
            vocal_energy = float(np.sum(S[vocal_mask, :] ** 2))
            return vocal_energy / total_energy
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _score_vocal_clarity(
        audio: np.ndarray,
        harmonic: np.ndarray,
        sr: int,
        librosa: Any,
    ) -> float:
        """
        Vocal band signal-to-noise ratio via HPSS.

        Computes the STFT of both the full mix and the harmonic component,
        then measures the energy ratio of the vocal band (300-4000 Hz)
        in the harmonic signal versus the residual (mix - harmonic) in
        the same band.  Higher SNR in the vocal band indicates clearer
        vocals.
        """
        try:
            n_fft = 2048
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            vocal_mask = (freqs >= 300.0) & (freqs <= 4000.0)

            # Full mix and harmonic STFT magnitudes
            S_full = np.abs(librosa.stft(audio, n_fft=n_fft))
            S_harm = np.abs(librosa.stft(harmonic, n_fft=n_fft))

            # Residual (noise) in vocal band
            S_residual = np.maximum(S_full - S_harm, 0.0)

            vocal_signal_energy = float(np.sum(S_harm[vocal_mask, :] ** 2))
            vocal_noise_energy = float(np.sum(S_residual[vocal_mask, :] ** 2))

            if vocal_signal_energy < 1e-10:
                return 0.3

            # SNR in dB, capped at reasonable range
            snr_db = 10.0 * np.log10(
                vocal_signal_energy / (vocal_noise_energy + 1e-10)
            )

            # Map SNR to [0, 1]: 0 dB -> 0.0, 20 dB -> 1.0
            score = float(np.clip(snr_db / 20.0, 0.0, 1.0))

            # Also factor in vocal band spectral centroid and bandwidth
            # as a secondary indicator of well-mixed vocals
            S_vocal = S_harm[vocal_mask, :]
            vocal_freqs = freqs[vocal_mask]

            if S_vocal.size > 0 and np.sum(S_vocal) > 1e-10:
                # Weighted centroid within vocal band
                power = np.sum(S_vocal ** 2, axis=1)
                total_power = np.sum(power) + 1e-10
                centroid = float(np.sum(vocal_freqs * power) / total_power)

                # Ideal centroid for vocals is ~1000-2500 Hz
                centroid_score = 1.0 - abs(centroid - 1750.0) / 1750.0
                centroid_score = float(np.clip(centroid_score, 0.0, 1.0))

                # Blend: 70% SNR, 30% centroid placement
                score = 0.7 * score + 0.3 * centroid_score

            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.5

    def _score_lyrics_intelligibility(
        self,
        audio: np.ndarray,
        sr: int,
        expected_lyrics: str = "",
    ) -> float:
        """
        Lyrics intelligibility via Whisper transcription.

        Loads the Whisper model lazily, transcribes the audio, and scores
        based on word count, average segment confidence, and speaking rate.
        If ``expected_lyrics`` is provided, also computes word error rate
        (WER) and blends it into the final score.
        """
        try:
            model = self._load_whisper()
            if model is None:
                return 0.5

            import whisper  # type: ignore[import-untyped]

            # Whisper expects 16 kHz mono float32
            if sr != 16000:
                import librosa

                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio

            # Pad or trim to Whisper's expected length
            audio_16k = whisper.pad_or_trim(audio_16k)

            # Transcribe with word-level timestamps for confidence
            result = model.transcribe(
                audio_16k,
                language="en",
                fp16=False,
                word_timestamps=True,
            )

            text = result.get("text", "").strip()
            segments = result.get("segments", [])

            if not text:
                # No speech detected — could be instrumental section
                return 0.3

            # --- Word count score ---
            words = text.split()
            word_count = len(words)
            # Reasonable word count for 30s of singing: 20-80 words
            word_count_score = float(np.clip(word_count / 40.0, 0.0, 1.0))

            # --- Average segment confidence ---
            confidences = []
            for seg in segments:
                # Whisper segments have 'avg_logprob' and optional 'no_speech_prob'
                avg_logprob = seg.get("avg_logprob", -1.0)
                no_speech_prob = seg.get("no_speech_prob", 0.5)
                # Convert log-prob to rough confidence
                conf = float(np.exp(avg_logprob))
                # Penalize segments with high no-speech probability
                conf *= (1.0 - no_speech_prob)
                confidences.append(conf)

            avg_confidence = float(np.mean(confidences)) if confidences else 0.0
            # Confidence typically in [0.0, 0.8] for good speech
            confidence_score = float(np.clip(avg_confidence / 0.6, 0.0, 1.0))

            # --- Speaking rate score ---
            duration_s = len(audio) / sr
            if duration_s > 0:
                words_per_second = word_count / duration_s
                # Natural singing: 1-3 words/sec; speaking: 2-4 words/sec
                # Bell curve centered at 2.0 wps
                rate_score = float(np.exp(-2.0 * (words_per_second - 2.0) ** 2))
            else:
                rate_score = 0.0

            # Combine sub-scores
            score = 0.35 * word_count_score + 0.40 * confidence_score + 0.25 * rate_score

            # --- WER if expected lyrics provided ---
            if expected_lyrics.strip():
                wer = self._compute_wer(expected_lyrics, text)
                # WER of 0.0 = perfect, 1.0+ = terrible
                wer_score = float(np.clip(1.0 - wer, 0.0, 1.0))
                # Blend WER into final: 50% WER, 50% base score
                score = 0.5 * wer_score + 0.5 * score

            return float(np.clip(score, 0.0, 1.0))
        except Exception as exc:
            logger.debug(f"Lyrics intelligibility scoring failed: {exc}")
            return 0.5

    @staticmethod
    def _score_vocal_pitch_quality(
        harmonic: np.ndarray,
        sr: int,
        librosa: Any,
    ) -> float:
        """
        Pitch stability, vibrato, and range from the harmonic signal.

        Uses pyin for f0 extraction on the harmonic component.  Measures:
        - Pitch stability (low jitter = good, but not zero)
        - Vibrato quality (4-7 Hz modulation is natural)
        - Pitch range (reasonable melodic range)

        AI vocals often have either perfectly steady pitch (robotic) or
        erratic pitch; natural vocals sit in between.
        """
        try:
            # Extract f0 using pyin on harmonic signal
            f0, voiced_flag, voiced_probs = librosa.pyin(
                harmonic,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C6"),
                sr=sr,
            )

            # Filter to voiced frames only
            voiced_f0 = f0[voiced_flag]
            if len(voiced_f0) < 10:
                # Not enough voiced frames to evaluate
                return 0.4

            # --- Pitch stability (jitter) ---
            # Frame-to-frame pitch variation in cents
            cents_diff = 1200.0 * np.abs(np.diff(np.log2(voiced_f0 + 1e-10)))
            median_jitter = float(np.median(cents_diff))

            # Natural singing: jitter ~5-30 cents
            # Robotic (AI): jitter < 2 cents
            # Erratic: jitter > 100 cents
            if median_jitter < 2.0:
                # Too stable — robotic
                stability_score = 0.3 + 0.2 * (median_jitter / 2.0)
            elif median_jitter <= 30.0:
                # Natural range
                stability_score = 0.7 + 0.3 * (1.0 - abs(median_jitter - 15.0) / 15.0)
            else:
                # Too erratic
                stability_score = float(np.clip(1.0 - (median_jitter - 30.0) / 100.0, 0.0, 0.5))

            # --- Vibrato detection ---
            # Compute autocorrelation of pitch contour to find ~5 Hz modulation
            hop_time = 512.0 / sr  # default pyin hop
            if len(voiced_f0) > 50:
                pitch_detrended = voiced_f0 - np.convolve(
                    voiced_f0, np.ones(11) / 11.0, mode="same"
                )
                autocorr = np.correlate(pitch_detrended, pitch_detrended, mode="full")
                autocorr = autocorr[len(autocorr) // 2:]
                if autocorr[0] > 0:
                    autocorr = autocorr / autocorr[0]

                # Look for peaks in the vibrato rate range (4-7 Hz)
                min_lag = max(1, int(1.0 / (7.0 * hop_time)))
                max_lag = min(len(autocorr) - 1, int(1.0 / (4.0 * hop_time)))

                if max_lag > min_lag:
                    vibrato_region = autocorr[min_lag:max_lag + 1]
                    vibrato_peak = float(np.max(vibrato_region))
                    # Strong vibrato correlation (0.2-0.6) is natural
                    vibrato_score = float(np.clip(vibrato_peak / 0.5, 0.0, 1.0))
                else:
                    vibrato_score = 0.4
            else:
                vibrato_score = 0.4

            # --- Pitch range ---
            pitch_range_cents = 1200.0 * np.log2(
                (np.percentile(voiced_f0, 95) + 1e-10)
                / (np.percentile(voiced_f0, 5) + 1e-10)
            )
            # Reasonable melodic range: 300-1800 cents (2.5-15 semitones)
            if pitch_range_cents < 100:
                range_score = 0.2
            elif pitch_range_cents <= 1800:
                range_score = float(np.clip(pitch_range_cents / 1200.0, 0.3, 1.0))
            else:
                # Very wide range — still okay but not better than moderate
                range_score = float(np.clip(1.0 - (pitch_range_cents - 1800.0) / 1200.0, 0.5, 1.0))

            # Combine: 40% stability, 30% vibrato, 30% range
            score = 0.40 * stability_score + 0.30 * vibrato_score + 0.30 * range_score
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def _score_vocal_expressiveness(
        audio: np.ndarray,
        sr: int,
        librosa: Any,
    ) -> float:
        """
        Dynamic variation in the vocal frequency band over time.

        Natural singing has crescendos, decrescendos, and emphasis on
        certain words.  Computes the RMS envelope of vocal-band-filtered
        audio and scores based on meaningful dynamic variation.

        Uses a one-sided floor: a minimum acceptable variation threshold
        below which the score drops, but very high variation (random noise)
        is also penalized.
        """
        try:
            from scipy.signal import butter, sosfilt

            # Bandpass filter for vocal range (300-4000 Hz)
            nyquist = sr / 2.0
            low = min(300.0 / nyquist, 0.99)
            high = min(4000.0 / nyquist, 0.99)

            if low >= high:
                return 0.5

            sos = butter(4, [low, high], btype="bandpass", output="sos")
            vocal_band = sosfilt(sos, audio)

            # Compute RMS envelope
            hop_length = 512
            frame_length = 2048
            rms = librosa.feature.rms(
                y=vocal_band, frame_length=frame_length, hop_length=hop_length
            )[0]

            if len(rms) < 4:
                return 0.5

            # Remove near-silence frames for variation computation
            rms_active = rms[rms > np.max(rms) * 0.05]
            if len(rms_active) < 4:
                return 0.3

            # Coefficient of variation of RMS
            mean_rms = float(np.mean(rms_active))
            if mean_rms < 1e-10:
                return 0.3

            cv = float(np.std(rms_active) / mean_rms)

            # Compute derivative of RMS for crescendo/decrescendo detection
            rms_diff = np.diff(rms_active)
            diff_std = float(np.std(rms_diff))
            diff_norm = diff_std / mean_rms

            # --- Scoring ---
            # CV: ideal range 0.15-0.60 for expressive singing
            # Too low (<0.05): flat, unexpressive
            # Too high (>1.0): random/noisy
            if cv < 0.05:
                cv_score = cv / 0.05 * 0.3  # Floor ramp-up
            elif cv <= 0.60:
                cv_score = 0.5 + 0.5 * (cv - 0.05) / 0.55
            else:
                cv_score = float(np.clip(1.0 - (cv - 0.60) / 0.60, 0.3, 1.0))

            # Derivative: presence of crescendos/decrescendos
            # Good range: 0.02-0.15
            if diff_norm < 0.01:
                diff_score = 0.2
            elif diff_norm <= 0.15:
                diff_score = 0.4 + 0.6 * (diff_norm - 0.01) / 0.14
            else:
                diff_score = float(np.clip(1.0 - (diff_norm - 0.15) / 0.20, 0.3, 1.0))

            # Blend: 60% CV, 40% derivative
            score = 0.60 * cv_score + 0.40 * diff_score
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def _score_sibilance_control(
        audio: np.ndarray,
        sr: int,
        librosa: Any,
    ) -> float:
        """
        Penalize excessive sibilance (harsh "s" sounds) in the 5-10 kHz band.

        Measures the energy ratio of the sibilance band (5-10 kHz) to the
        overall vocal energy (300-10000 Hz).  Normal ratio is 0.05-0.15.
        Above 0.25 is harsh.  This is a one-sided penalty: only excessive
        sibilance is penalized.

        Returns 1.0 (perfect) for well-controlled sibilance, lower for
        excessive harshness.
        """
        try:
            n_fft = 2048
            S = np.abs(librosa.stft(audio, n_fft=n_fft)) ** 2
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # Check that we have enough frequency resolution
            max_freq = freqs[-1] if len(freqs) > 0 else 0.0
            if max_freq < 5000.0:
                # Sample rate too low to measure sibilance band
                return 0.7

            # Vocal+sibilance band: 300-10000 Hz
            vocal_full_mask = (freqs >= 300.0) & (freqs <= 10000.0)
            # Sibilance band: 5000-10000 Hz
            sibilance_mask = (freqs >= 5000.0) & (freqs <= 10000.0)

            vocal_energy = float(np.sum(S[vocal_full_mask, :]))
            sibilance_energy = float(np.sum(S[sibilance_mask, :]))

            if vocal_energy < 1e-10:
                return 0.5

            ratio = sibilance_energy / vocal_energy

            # Scoring: one-sided penalty for excessive sibilance
            if ratio <= 0.15:
                # Normal or low sibilance — full score
                score = 1.0
            elif ratio <= 0.25:
                # Slightly elevated — gentle penalty
                score = 1.0 - (ratio - 0.15) / 0.10 * 0.3
            else:
                # Harsh sibilance — steeper penalty
                score = 0.7 - (ratio - 0.25) / 0.25 * 0.5
                score = max(score, 0.1)

            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_wer(reference: str, hypothesis: str) -> float:
        """
        Compute word error rate (WER) using edit distance.

        Args:
            reference: Ground truth text.
            hypothesis: Transcribed text.

        Returns:
            WER as a float (0.0 = perfect, 1.0+ = poor).
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        # Levenshtein distance on word level
        n = len(ref_words)
        m = len(hyp_words)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + cost,  # substitution
                )

        return dp[n][m] / n
