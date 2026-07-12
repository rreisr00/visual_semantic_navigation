"""Rules + regex intent parser for spoken English commands.

Pure Python (stdlib only) so it can be unit-tested without ROS or the ML venv.

Three intents, first match wins:

1. ``MoveToPosition`` — "move to 2.5 3.0", "go to position 1, -2"
   (requires two numerals; coordinates are map-frame meters).
2. ``SaveWaypoint``   — "save waypoint kitchen", "mark this location as sofa"
   (label slugified; empty label allowed, kg_manager auto-names).
3. ``SemanticGoal``   — anything else after stripping leading fillers
   ("go to the sofa" -> query "the sofa"). The query is forwarded verbatim
   to the SigLIP ranking, so descriptive phrases are fine.

Known limitation: spoken number words are not converted ("two point five"),
but Whisper reliably emits digits for numbers in English dictation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MoveToPosition:
    x: float
    y: float
    yaw: float | None = None


@dataclass(frozen=True)
class SaveWaypoint:
    label: str


@dataclass(frozen=True)
class SemanticGoal:
    query: str


@dataclass(frozen=True)
class NoIntent:
    reason: str


Intent = MoveToPosition | SaveWaypoint | SemanticGoal | NoIntent

_NUM = r"(-?\d+(?:\.\d+)?)"

# "move to 2.5 3.0" / "go to position 1, -2.5" / "navigate to point 0 0"
_MOVE_RE = re.compile(
    rf"\b(?:move|go|navigate|drive)\s+to\s+"
    rf"(?:the\s+)?(?:position|coordinates?|point)?\s*"
    rf"{_NUM}\s*[, ]\s*{_NUM}"
)

# "save waypoint kitchen" / "capture this location as living room" /
# "mark waypoint" (label optional -> kg_manager auto-names)
_SAVE_RE = re.compile(
    r"\b(?:save|capture|store|mark)\b"
    r"(?:\s+(?:this|a|the|current|my))*"
    r"\s+(?:"
    r"(?:waypoint|location|place|position|spot)(?:\s+(?:as|called|named))?"
    r"|(?:as|called|named)"
    r")\s*(.*)$"
)

# Leading fillers stripped before treating the rest as a semantic query.
_FILLER_RE = re.compile(
    r"^(?:please\s+|robot\s+|hey\s+|now\s+)*"
    r"(?:(?:go|navigate|drive)\s+to|take\s+me\s+to|bring\s+me\s+to|find|"
    r"look\s+for|where\s+is)?"
    r"\s*(?:the\s+)?"
)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation (keeping '.'/'-' inside numbers), collapse spaces."""
    text = text.lower().strip()
    # Whisper artifacts: trailing sentence punctuation, commas between words.
    text = re.sub(r"[!?¿¡]", "", text)
    # Drop periods NOT between digits (keep "2.5", drop "sofa.").
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    # Keep commas only between digits (coordinate separator), drop elsewhere.
    text = re.sub(r"(?<!\d),|,(?!\s*-?\d)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def parse(text: str) -> Intent:
    """Parse a transcribed utterance into an Intent."""
    text = normalize(text)
    if len(text) < 2:
        return NoIntent(reason="empty or too-short transcript")

    m = _MOVE_RE.search(text)
    if m:
        return MoveToPosition(x=float(m.group(1)), y=float(m.group(2)))

    m = _SAVE_RE.search(text)
    if m:
        return SaveWaypoint(label=_slugify(m.group(1)))

    query = _FILLER_RE.sub("", text, count=1).strip()
    if not query:
        return NoIntent(reason="no description after command words")
    return SemanticGoal(query=query)
