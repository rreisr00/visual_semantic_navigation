"""Polygonal room zones in the map frame — pure Python.

A Room is a labelled polygon; waypoints whose (x, y) fall inside it are
considered part of that room (the knowledge graph links them with a
CONTAINS room->waypoint edge). The legacy rectangle fields remain part of the
public contract, so existing YAML files and the two-click editor keep working.
"""

from __future__ import annotations

import math
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
    polygon: tuple[tuple[float, float], ...] = ()
    transition_width_m: float = 0.5

    @classmethod
    def from_polygon(
        cls,
        room_id: str,
        polygon: Sequence[Sequence[float]],
        transition_width_m: float = 0.5,
    ) -> "Room":
        points = tuple((float(point[0]), float(point[1])) for point in polygon)
        if len(points) < 3:
            raise ValueError("a room polygon requires at least three vertices")
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return cls(
            room_id, min(xs), min(ys), max(xs), max(ys), points,
            max(0.0, float(transition_width_m)),
        )

    def normalized(self) -> "Room":
        """Return a Room with min <= max on both axes (clicks in any order)."""
        if self.polygon:
            return Room.from_polygon(
                self.room_id, self.polygon, self.transition_width_m
            )
        return Room(
            room_id=self.room_id,
            min_x=min(self.min_x, self.max_x),
            min_y=min(self.min_y, self.max_y),
            max_x=max(self.min_x, self.max_x),
            max_y=max(self.min_y, self.max_y),
            transition_width_m=max(0.0, float(self.transition_width_m)),
        )

    def contains(self, x: float, y: float) -> bool:
        """Inclusive point-in-polygon test (rectangle-compatible)."""
        points = self.corners()
        for start, end in zip(points, points[1:] + points[:1]):
            if _point_on_segment(x, y, start, end):
                return True
        inside = False
        previous = points[-1]
        for current in points:
            x1, y1 = previous
            x2, y2 = current
            if ((y1 > y) != (y2 > y)) and (
                x < (x2 - x1) * (y - y1) / (y2 - y1) + x1
            ):
                inside = not inside
            previous = current
        return inside

    def corners(self) -> list[tuple[float, float]]:
        """Rectangle corners in draw order (counter-clockwise, not closed)."""
        if self.polygon:
            return list(self.polygon)
        return [
            (self.min_x, self.min_y),
            (self.max_x, self.min_y),
            (self.max_x, self.max_y),
            (self.min_x, self.max_y),
        ]

    @property
    def center(self) -> tuple[float, float]:
        points = self.corners()
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    def distance_to_boundary(self, x: float, y: float) -> float:
        """Minimum Euclidean distance from a point to the room boundary."""
        points = self.corners()
        return min(
            _distance_to_segment(x, y, start, end)
            for start, end in zip(points, points[1:] + points[:1])
        )

    def in_transition_zone(self, x: float, y: float) -> bool:
        """Whether a point is within the configured band around the boundary."""
        return self.distance_to_boundary(x, y) <= self.transition_width_m


def _point_on_segment(
    x: float,
    y: float,
    start: tuple[float, float],
    end: tuple[float, float],
    tolerance: float = 1e-9,
) -> bool:
    x1, y1 = start
    x2, y2 = end
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return abs(cross) <= tolerance and (
        min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
        and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance
    )


def _distance_to_segment(
    x: float,
    y: float,
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(x - x1, y - y1)
    ratio = max(
        0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_sq)
    )
    return math.hypot(x - (x1 + ratio * dx), y - (y1 + ratio * dy))


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


def next_waypoint_name(existing_names: Iterable[str]) -> str:
    """Return the next compact automatic waypoint id: ``W1``, ``W2``, …"""
    pattern = re.compile(r"^W(\d+)$")
    highest = 0
    for name in existing_names:
        match = pattern.match(str(name))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"W{highest + 1}"


def load_rooms(path: str) -> list[Room]:
    """Load rooms from YAML; a missing or empty file yields []."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rooms: list[Room] = []
    for entry in data.get("rooms", []):
        polygon = entry.get("polygon") or []
        transition = float(entry.get("transition_width_m", 0.5))
        if len(polygon) >= 3:
            rooms.append(Room.from_polygon(entry["room_id"], polygon, transition))
        else:
            rooms.append(Room(
                room_id=str(entry["room_id"]),
                min_x=float(entry["min_x"]),
                min_y=float(entry["min_y"]),
                max_x=float(entry["max_x"]),
                max_y=float(entry["max_y"]),
                transition_width_m=transition,
            ).normalized())
    return rooms


def save_rooms(path: str, rooms: Iterable[Room]) -> str:
    """Write rooms to YAML (schema: rooms: [{room_id, min_x, ...}])."""
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "rooms": [
            ({
                "room_id": r.room_id,
                "min_x": r.min_x,
                "min_y": r.min_y,
                "max_x": r.max_x,
                "max_y": r.max_y,
                "transition_width_m": r.transition_width_m,
                **(
                    {"polygon": [list(point) for point in r.polygon]}
                    if r.polygon else {}
                ),
            })
            for r in rooms
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return path
