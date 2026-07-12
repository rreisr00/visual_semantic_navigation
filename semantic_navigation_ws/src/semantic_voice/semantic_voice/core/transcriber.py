"""Thin faster-whisper wrapper with lazy import and CPU degradation.

faster-whisper lives in the project ML venv (.venv-1), injected via
PYTHONPATH by voice.launch.py — importing this module stays cheap so the
parser/segmenter tests never need the venv.

On CUDA failure at load or inference time the model is reloaded once on CPU
(int8), mirroring the visual_encoder degradation strategy: the RTX 2060's
6 GB are shared with SigLIP + YOLO.
"""

from __future__ import annotations

import logging

import numpy as np

_LOG = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str = "en",
        vad_filter: bool = True,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._vad_filter = vad_filter
        self._model = None

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        """Load the model (downloads to ~/.cache/huggingface on first run)."""
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        try:
            self._model = WhisperModel(
                self._model_size, device=self._device,
                compute_type=self._compute_type,
            )
        except Exception as exc:  # CUDA init/OOM → degrade to CPU once
            if self._device == "cpu":
                raise
            _LOG.warning("Whisper load on %s failed (%s); falling back to CPU",
                         self._device, exc)
            self._device, self._compute_type = "cpu", "int8"
            self._model = WhisperModel(
                self._model_size, device="cpu", compute_type="int8",
            )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe mono float32 audio at 16 kHz; returns joined text."""
        self.load()
        if sample_rate != 16000:
            raise ValueError("faster-whisper expects 16 kHz mono float32 input")
        try:
            segments, _info = self._model.transcribe(
                audio,
                language=self._language,
                vad_filter=self._vad_filter,
                beam_size=5,
            )
            return " ".join(s.text.strip() for s in segments).strip()
        except Exception as exc:
            if self._device == "cpu":
                raise
            _LOG.warning("Whisper inference on %s failed (%s); retrying on CPU",
                         self._device, exc)
            self._device, self._compute_type, self._model = "cpu", "int8", None
            return self.transcribe(audio, sample_rate)
