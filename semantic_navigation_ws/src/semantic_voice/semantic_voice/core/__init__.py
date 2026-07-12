"""Pure-Python core of semantic_voice (no rclpy).

``intent_parser`` and ``segmenter`` are stdlib/numpy-only and always
importable; ``transcriber`` and ``audio_capture`` lazily import
faster-whisper / sounddevice (available in the project ML venv).
"""

from semantic_voice.core.intent_parser import (
    Intent,
    MoveToPosition,
    NoIntent,
    SaveWaypoint,
    SemanticGoal,
    normalize,
    parse,
)
from semantic_voice.core.segmenter import SpeechSegmenter

__all__ = [
    "Intent",
    "MoveToPosition",
    "NoIntent",
    "SaveWaypoint",
    "SemanticGoal",
    "SpeechSegmenter",
    "normalize",
    "parse",
]
