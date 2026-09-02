"""Pure helpers for the semantic operator GUI."""

from __future__ import annotations

from collections.abc import Collection
import math
import re


_VIEW_SEPARATOR = re.compile(r'[\s,;]+')
_DIRECTIONS = {'forward', 'back', 'left', 'right'}


def parse_view_angles(text: str) -> list[float]:
    """Parse a comma, semicolon or whitespace-separated list of view angles."""
    stripped = text.strip()
    if not stripped:
        return []
    values: list[float] = []
    for token in _VIEW_SEPARATOR.split(stripped):
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f'invalid view angle: {token!r}') from exc
        if not math.isfinite(value):
            raise ValueError('view angles must be finite')
        if not -360.0 <= value <= 360.0:
            raise ValueError('view angles must be between -360 and 360 degrees')
        values.append(value)
    return values


def motion_from_directions(
    directions: Collection[str],
    linear_speed: float,
    angular_speed: float,
) -> tuple[float, float]:
    """Convert held direction controls into linear and angular velocity."""
    unknown = set(directions) - _DIRECTIONS
    if unknown:
        raise ValueError(f"unknown motion directions: {', '.join(sorted(unknown))}")
    linear = linear_speed * (
        int('forward' in directions) - int('back' in directions)
    )
    angular = angular_speed * (
        int('left' in directions) - int('right' in directions)
    )
    return float(linear), float(angular)


def normalized_room_bounds(
    corner_a: tuple[float, float],
    corner_b: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Return min_x, min_y, max_x, max_y for two arbitrary corners."""
    ax, ay = corner_a
    bx, by = corner_b
    values = (ax, ay, bx, by)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('room corners must be finite')
    min_x, max_x = sorted((float(ax), float(bx)))
    min_y, max_y = sorted((float(ay), float(by)))
    if min_x == max_x or min_y == max_y:
        raise ValueError('room corners must define a non-zero rectangle')
    return min_x, min_y, max_x, max_y
