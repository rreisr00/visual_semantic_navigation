"""Deterministic geometry helpers for readable object-relation diagrams."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence


Rect = tuple[float, float, float, float]
Point = tuple[float, float]


def radial_node_positions(
    identifiers: Sequence[str],
    *,
    centre: Point = (330.0, 215.0),
) -> dict[str, Point]:
    """Place nodes on a spacious ellipse whose size grows with node count."""
    ordered = list(dict.fromkeys(str(value) for value in identifiers))
    count = len(ordered)
    if not count:
        return {}
    if count == 1:
        return {ordered[0]: centre}
    radius_x = max(150.0, min(330.0, 42.0 * count))
    radius_y = max(105.0, min(230.0, 29.0 * count))
    return {
        identifier: (
            centre[0] + math.cos(-math.pi / 2.0 + 2.0 * math.pi * index / count)
            * radius_x,
            centre[1] + math.sin(-math.pi / 2.0 + 2.0 * math.pi * index / count)
            * radius_y,
        )
        for index, identifier in enumerate(ordered)
    }


def relation_lane_offsets(pairs: Sequence[tuple[str, str]]) -> list[float]:
    """Give parallel and reverse relations separate perpendicular lanes."""
    totals = Counter(tuple(sorted(pair)) for pair in pairs)
    consumed: defaultdict[tuple[str, str], int] = defaultdict(int)
    offsets: list[float] = []
    for pair in pairs:
        key = tuple(sorted(pair))
        index = consumed[key]
        consumed[key] += 1
        centred = index - (totals[key] - 1) / 2.0
        offsets.append(centred * 34.0)
    return offsets


def non_overlapping_label_rects(
    desired_centres: Sequence[Point],
    sizes: Sequence[tuple[float, float]],
    *,
    occupied: Iterable[Rect] = (),
) -> list[Rect]:
    """Move labels along a deterministic spiral until rectangles do not meet."""
    placed = list(occupied)
    output: list[Rect] = []
    for desired, size in zip(desired_centres, sizes):
        width, height = size
        selected: Rect | None = None
        for dx, dy in _candidate_displacements():
            candidate = (
                desired[0] - width / 2.0 + dx,
                desired[1] - height / 2.0 + dy,
                width,
                height,
            )
            if not any(_intersects(candidate, other, padding=6.0) for other in placed):
                selected = candidate
                break
        if selected is None:
            selected = (
                desired[0] - width / 2.0,
                desired[1] - height / 2.0 + 36.0 * len(output),
                width,
                height,
            )
        output.append(selected)
        placed.append(selected)
    return output


def _candidate_displacements() -> Iterable[Point]:
    yield 0.0, 0.0
    for ring in range(1, 18):
        distance = 24.0 * ring
        yield 0.0, -distance
        yield distance, 0.0
        yield 0.0, distance
        yield -distance, 0.0
        yield distance, -distance
        yield distance, distance
        yield -distance, distance
        yield -distance, -distance


def _intersects(first: Rect, second: Rect, *, padding: float) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return not (
        ax + aw + padding <= bx
        or bx + bw + padding <= ax
        or ay + ah + padding <= by
        or by + bh + padding <= ay
    )
