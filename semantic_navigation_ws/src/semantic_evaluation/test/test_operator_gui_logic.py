import pytest

from semantic_evaluation.core.operator_gui_logic import (
    motion_from_directions,
    normalized_room_bounds,
    parse_view_angles,
)


def test_parse_view_angles_accepts_common_separators():
    assert parse_view_angles('0, 90; 180 270') == [0.0, 90.0, 180.0, 270.0]
    assert parse_view_angles('  ') == []


@pytest.mark.parametrize('text', ['front', 'nan', '361', '-361'])
def test_parse_view_angles_rejects_invalid_values(text):
    with pytest.raises(ValueError):
        parse_view_angles(text)


def test_motion_from_directions_combines_axes_and_opposites():
    assert motion_from_directions({'forward', 'left'}, 0.4, 0.8) == (0.4, 0.8)
    assert motion_from_directions({'forward', 'back'}, 0.4, 0.8) == (0.0, 0.0)
    assert motion_from_directions(set(), 0.4, 0.8) == (0.0, 0.0)


def test_motion_from_directions_rejects_unknown_direction():
    with pytest.raises(ValueError, match='sideways'):
        motion_from_directions({'sideways'}, 0.4, 0.8)


def test_normalized_room_bounds_accepts_corners_in_any_order():
    assert normalized_room_bounds((4.0, 3.0), (-1.0, -2.0)) == (
        -1.0,
        -2.0,
        4.0,
        3.0,
    )


def test_normalized_room_bounds_rejects_zero_area():
    with pytest.raises(ValueError, match='non-zero'):
        normalized_room_bounds((1.0, 2.0), (1.0, 4.0))
