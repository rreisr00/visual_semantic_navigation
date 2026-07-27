from semantic_navigation_core.goal_validation import GridSpec, validate_goal


def _grid():
    return GridSpec(width=7, height=7, resolution=1.0, origin_x=0.0, origin_y=0.0)


def test_accepts_safe_goal_without_adjustment():
    result = validate_goal(3.5, 3.5, [0] * 49, _grid(), obstacle_margin_m=1.0)
    assert result.valid
    assert result.status == "valid"
    assert result.adjustment_distance_m == 0.0


def test_adjusts_occupied_goal_to_nearest_safe_cell():
    data = [0] * 49
    data[3 * 7 + 3] = 100
    result = validate_goal(
        3.5, 3.5, data, _grid(), search_radius_m=3.0, obstacle_margin_m=0.0
    )
    assert result.valid
    assert result.status == "adjusted"
    assert result.adjustment_distance_m == 1.0


def test_rejects_goal_when_no_safe_cell_exists():
    result = validate_goal(
        3.5, 3.5, [100] * 49, _grid(), search_radius_m=2.0, obstacle_margin_m=0.0
    )
    assert not result.valid
    assert result.status == "no_safe_cell"


def test_rejects_malformed_map():
    result = validate_goal(0.0, 0.0, [], _grid())
    assert not result.valid
    assert result.status == "invalid_map"

