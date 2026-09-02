"""Navigation path metrics independent of ROS message types."""
from __future__ import annotations

import math
from collections.abc import Iterable


def path_length_2d(points: Iterable[tuple[float, float]]) -> float:
    """Return the accumulated Euclidean length of an ordered 2D path."""
    iterator = iter(points)
    previous = next(iterator, None)
    if previous is None:
        return 0.0
    total = 0.0
    for current in iterator:
        total += math.hypot(current[0] - previous[0], current[1] - previous[1])
        previous = current
    return total


def spl(success: bool, optimal_length: float, actual_length: float) -> float:
    """Success weighted by path length as defined for navigation evaluation."""
    if not success:
        return 0.0
    if not math.isfinite(optimal_length) or optimal_length < 0.0:
        return math.nan
    if not math.isfinite(actual_length) or actual_length < 0.0:
        return math.nan
    denominator = max(optimal_length, actual_length)
    return 1.0 if denominator == 0.0 else optimal_length / denominator
