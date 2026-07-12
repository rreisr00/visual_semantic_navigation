"""Microphone capture via sounddevice (lazy import; needs apt libportaudio2).

Produces mono float32 blocks at the configured sample rate — exactly what
faster-whisper consumes, no byte conversion needed.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

import numpy as np


class MicrophoneCapture:
    def __init__(
        self,
        device: str | int | None = None,
        sample_rate: int = 16000,
        block_s: float = 0.03,
    ) -> None:
        self._device = device if device not in ("", None) else None
        self._sample_rate = sample_rate
        self._blocksize = int(block_s * sample_rate)

    def _open_stream(self, on_block):
        import sounddevice as sd

        def _callback(indata, _frames, _time, status):
            if status:
                # over/underruns are logged by the caller via block content
                pass
            on_block(indata[:, 0].copy())

        return sd.InputStream(
            device=self._device,
            channels=1,
            samplerate=self._sample_rate,
            blocksize=self._blocksize,
            dtype="float32",
            callback=_callback,
        )

    def record_toggle(self, stop_event: threading.Event) -> np.ndarray:
        """Record until ``stop_event`` is set (push-to-talk); returns the take."""
        blocks: list[np.ndarray] = []
        with self._open_stream(blocks.append):
            stop_event.wait()
        return np.concatenate(blocks) if blocks else np.array([], dtype=np.float32)

    def stream_chunks(self, stop_event: threading.Event) -> Iterator[np.ndarray]:
        """Yield fixed-size blocks continuously until ``stop_event`` is set (VAD)."""
        q: queue.Queue[np.ndarray] = queue.Queue()
        with self._open_stream(q.put):
            while not stop_event.is_set():
                try:
                    yield q.get(timeout=0.2)
                except queue.Empty:
                    continue
