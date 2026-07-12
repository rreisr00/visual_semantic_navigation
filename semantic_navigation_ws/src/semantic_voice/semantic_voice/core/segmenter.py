"""Streaming speech segmenter for continuous (VAD) listening mode.

Pure numpy, deterministic: an RMS energy gate with hangover. Speech starts
when a chunk's RMS crosses ``rms_threshold``; the utterance ends after
``hangover_s`` of continuous silence and is returned as one array. Fine
silence trimming inside the utterance is left to faster-whisper's built-in
silero ``vad_filter``.
"""

from __future__ import annotations

import numpy as np


class SpeechSegmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        rms_threshold: float = 0.015,
        hangover_s: float = 0.8,
        max_utterance_s: float = 12.0,
        min_utterance_s: float = 0.3,
    ) -> None:
        self._sample_rate = sample_rate
        self._rms_threshold = rms_threshold
        self._hangover_samples = int(hangover_s * sample_rate)
        self._max_samples = int(max_utterance_s * sample_rate)
        self._min_samples = int(min_utterance_s * sample_rate)
        self.reset()

    def reset(self) -> None:
        self._buffer: list[np.ndarray] = []
        self._buffered = 0
        self._silence = 0
        self._in_speech = False

    def push(self, chunk: np.ndarray) -> np.ndarray | None:
        """Feed one mono float32 chunk; returns a finished utterance or None."""
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        voiced = rms >= self._rms_threshold

        if not self._in_speech:
            if not voiced:
                return None
            self._in_speech = True

        self._buffer.append(chunk)
        self._buffered += chunk.size
        self._silence = 0 if voiced else self._silence + chunk.size

        if self._silence >= self._hangover_samples or self._buffered >= self._max_samples:
            return self._finalize()
        return None

    def _finalize(self) -> np.ndarray | None:
        utterance = np.concatenate(self._buffer) if self._buffer else np.array([])
        speech_len = utterance.size - self._silence
        self.reset()
        if speech_len < self._min_samples:
            return None
        return utterance
