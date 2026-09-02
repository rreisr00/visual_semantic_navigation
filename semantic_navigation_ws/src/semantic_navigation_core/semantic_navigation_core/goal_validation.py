"""Occupancy-grid validation and bounded nearest-free goal adjustment."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot
from typing import Sequence


@dataclass(frozen=True)
class GridSpec:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    occupied_threshold: int = 65
    allow_unknown: bool = False


@dataclass(frozen=True)
class GoalValidationResult:
    valid: bool
    x: float
    y: float
    status: str
    adjustment_distance_m: float = 0.0


def world_to_cell(x: float, y: float, grid: GridSpec) -> tuple[int, int]:
    return (
        int((x - grid.origin_x) // grid.resolution),
        int((y - grid.origin_y) // grid.resolution),
    )


def cell_to_world(mx: int, my: int, grid: GridSpec) -> tuple[float, float]:
    return (
        grid.origin_x + (mx + 0.5) * grid.resolution,
        grid.origin_y + (my + 0.5) * grid.resolution,
    )


def _inside(mx: int, my: int, grid: GridSpec) -> bool:
    return 0 <= mx < grid.width and 0 <= my < grid.height


def _free(data: Sequence[int], mx: int, my: int, grid: GridSpec) -> bool:
    if not _inside(mx, my, grid):
        return False
    value = int(data[my * grid.width + mx])
    return (value < 0 and grid.allow_unknown) or 0 <= value < grid.occupied_threshold


def validate_goal(
    x: float,
    y: float,
    data: Sequence[int],
    grid: GridSpec,
    search_radius_m: float = 0.75,
    obstacle_margin_m: float = 0.25,
) -> GoalValidationResult:
    """Validate a goal and, if needed, select the nearest safe free cell."""
    if len(data) != grid.width * grid.height or grid.resolution <= 0.0:
        return GoalValidationResult(False, x, y, "invalid_map")
    start_x, start_y = world_to_cell(x, y, grid)
    search_cells = max(0, int(ceil(search_radius_m / grid.resolution)))
    margin_cells = max(0, int(ceil(obstacle_margin_m / grid.resolution)))

    def safe(mx: int, my: int) -> bool:
        if not _free(data, mx, my, grid):
            return False
        for oy in range(-margin_cells, margin_cells + 1):
            for ox in range(-margin_cells, margin_cells + 1):
                if ox * ox + oy * oy > margin_cells * margin_cells:
                    continue
                if not _free(data, mx + ox, my + oy, grid):
                    return False
        return True

    if safe(start_x, start_y):
        return GoalValidationResult(True, x, y, "valid")

    candidates: list[tuple[float, int, int]] = []
    for my in range(start_y - search_cells, start_y + search_cells + 1):
        for mx in range(start_x - search_cells, start_x + search_cells + 1):
            if safe(mx, my):
                wx, wy = cell_to_world(mx, my, grid)
                candidates.append((hypot(wx - x, wy - y), mx, my))
    if not candidates:
        status = "outside_map" if not _inside(start_x, start_y, grid) else "no_safe_cell"
        return GoalValidationResult(False, x, y, status)
    distance, mx, my = min(candidates)
    wx, wy = cell_to_world(mx, my, grid)
    return GoalValidationResult(True, wx, wy, "adjusted", distance)

