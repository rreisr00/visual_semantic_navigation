"""Room zones: axis-aligned rectangles in the map frame — pure Python.

A Room is a labelled rectangle; waypoints whose (x, y) fall inside it are
considered part of that room (the knowledge graph links them with a
CONTAINS room->waypoint edge). Rectangles come from two RViz clicks, so
``normalized()`` accepts corners in any order. YAML persistence keeps room
definitions per scenario.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class Room:
    room_id: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def normalized(self) -> "Room":
        """Return a Room with min <= max on both axes (clicks in any order)."""
        return Room(
            room_id=self.room_id,
            min_x=min(self.min_x, self.max_x),
            min_y=min(self.min_y, self.max_y),
            max_x=max(self.min_x, self.max_x),
            max_y=max(self.min_y, self.max_y),
        )

    def contains(self, x: float, y: float) -> bool:
        """Inclusive point-in-rectangle test."""
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def corners(self) -> list[tuple[float, float]]:
        """Rectangle corners in draw order (counter-clockwise, not closed)."""
        return [
            (self.min_x, self.min_y),
            (self.max_x, self.min_y),
            (self.max_x, self.max_y),
            (self.min_x, self.max_y),
        ]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)


def room_of_point(x: float, y: float, rooms: Sequence[Room]) -> str | None:
    """Return the room_id of the first room containing (x, y), else None.

    On overlapping rooms the first match in list order wins.
    """
    for room in rooms:
        if room.contains(x, y):
            return room.room_id
    return None


def next_instance_name(
    base: str, existing_names: Iterable[str], width: int = 2
) -> str:
    """Next free ``<base>_<NN>`` given the names already in the graph.

    ``next_instance_name("cocina", ["cocina_01", "cocina_03"])`` → ``cocina_04``
    (max existing suffix + 1, zero-padded; unrelated names are ignored).
    """
    pattern = re.compile(rf"^{re.escape(base)}_(\d+)$")
    highest = 0
    for name in existing_names:
        m = pattern.match(name)
        if m:
            highest = max(highest, int(m.group(1)))
    return f"{base}_{highest + 1:0{width}d}"


def load_rooms(path: str) -> list[Room]:
    """Load rooms from YAML; a missing or empty file yields []."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return [
        Room(
            room_id=str(entry["room_id"]),
            min_x=float(entry["min_x"]),
            min_y=float(entry["min_y"]),
            max_x=float(entry["max_x"]),
            max_y=float(entry["max_y"]),
        ).normalized()
        for entry in data.get("rooms", [])
    ]


def save_rooms(path: str, rooms: Iterable[Room]) -> str:
    """Write rooms to YAML (schema: rooms: [{room_id, min_x, ...}])."""
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "rooms": [
            {
                "room_id": r.room_id,
                "min_x": r.min_x,
                "min_y": r.min_y,
                "max_x": r.max_x,
                "max_y": r.max_y,
            }
            for r in rooms
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return path
