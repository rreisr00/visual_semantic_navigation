"""Unit tests for the pure rooms module (no ROS required)."""

from semantic_navigation_core.rooms import (
    Room,
    load_rooms,
    next_instance_name,
    next_waypoint_name,
    room_of_point,
    save_rooms,
)


class TestNormalization:
    def test_swapped_corners_normalize(self):
        r = Room("cocina", 3.0, 4.0, 1.0, 2.0).normalized()
        assert (r.min_x, r.min_y, r.max_x, r.max_y) == (1.0, 2.0, 3.0, 4.0)

    def test_already_normalized_unchanged(self):
        r = Room("cocina", 1.0, 2.0, 3.0, 4.0)
        assert r.normalized() == r


class TestContains:
    def test_inside(self):
        assert Room("a", 0.0, 0.0, 2.0, 2.0).contains(1.0, 1.0)

    def test_boundary_inclusive(self):
        r = Room("a", 0.0, 0.0, 2.0, 2.0)
        assert r.contains(0.0, 0.0)
        assert r.contains(2.0, 2.0)
        assert r.contains(0.0, 2.0)

    def test_outside(self):
        r = Room("a", 0.0, 0.0, 2.0, 2.0)
        assert not r.contains(2.1, 1.0)
        assert not r.contains(-0.1, 1.0)

    def test_concave_polygon(self):
        room = Room.from_polygon(
            "l_shape", [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]
        )
        assert room.contains(0.5, 2.0)
        assert not room.contains(2.0, 2.0)
        assert room.contains(1.0, 2.0)  # boundary is inclusive

    def test_transition_zone_uses_boundary_distance(self):
        room = Room.from_polygon("office", [(0, 0), (2, 0), (2, 2), (0, 2)])
        assert room.in_transition_zone(0.25, 1.0)
        assert not room.in_transition_zone(1.0, 1.0)


class TestRoomOfPoint:
    def test_first_match_wins_on_overlap(self):
        rooms = [
            Room("primera", 0.0, 0.0, 4.0, 4.0),
            Room("segunda", 2.0, 2.0, 6.0, 6.0),
        ]
        assert room_of_point(3.0, 3.0, rooms) == "primera"

    def test_no_match(self):
        assert room_of_point(10.0, 10.0, [Room("a", 0.0, 0.0, 1.0, 1.0)]) is None

    def test_empty_rooms(self):
        assert room_of_point(0.0, 0.0, []) is None


class TestYamlIO:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "rooms.yaml")
        rooms = [
            Room("cocina", -3.0, -1.0, 0.0, 2.0),
            Room("sala_estar", 0.5, -2.0, 4.0, 1.5),
        ]
        save_rooms(path, rooms)
        assert load_rooms(path) == rooms

    def test_missing_file_yields_empty(self, tmp_path):
        assert load_rooms(str(tmp_path / "nope.yaml")) == []

    def test_load_normalizes(self, tmp_path):
        path = str(tmp_path / "rooms.yaml")
        save_rooms(path, [Room("x", 5.0, 5.0, 1.0, 1.0)])
        (room,) = load_rooms(path)
        assert (room.min_x, room.max_x) == (1.0, 5.0)

    def test_corners_and_center(self):
        r = Room("a", 0.0, 0.0, 2.0, 4.0)
        assert len(r.corners()) == 4
        assert r.center == (1.0, 2.0)

    def test_polygon_round_trip(self, tmp_path):
        path = str(tmp_path / "rooms.yaml")
        room = Room.from_polygon(
            "office", [(0, 0), (3, 0), (2, 2)], transition_width_m=0.35
        )
        save_rooms(path, [room])
        assert load_rooms(path) == [room]


class TestNextInstanceName:
    def test_first_instance(self):
        assert next_instance_name("cocina", []) == "cocina_01"

    def test_increments_from_max(self):
        assert (
            next_instance_name("cocina", ["cocina_01", "cocina_03"]) == "cocina_04"
        )

    def test_ignores_unrelated_names(self):
        existing = ["cocina_01", "cocina_norte_05", "waypoint_123", "cocina_x"]
        assert next_instance_name("cocina", existing) == "cocina_02"

    def test_multi_token_base(self):
        existing = ["sala_estar_01", "sala_estar_02"]
        assert next_instance_name("sala_estar", existing) == "sala_estar_03"

    def test_zero_padding_growth(self):
        assert next_instance_name("a", ["a_99"]) == "a_100"


class TestNextWaypointName:
    def test_starts_at_one(self):
        assert next_waypoint_name([]) == "W1"

    def test_increments_highest_compact_identifier(self):
        assert next_waypoint_name(["W1", "W3", "salon_04"]) == "W4"

    def test_ignores_descriptive_and_malformed_names(self):
        assert next_waypoint_name(["waypoint_9", "W_room", "w8"]) == "W1"
