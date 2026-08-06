from pathlib import Path

import pytest
import yaml

from semantic_bringup.scene_config import (
    apply_yaml_defaults,
    load_scene_launch_config,
    resolve_scene_config,
)


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_bundled_name_loads_launch_and_parameter_defaults(tmp_path):
    bringup = tmp_path / "semantic_bringup"
    world = _write(bringup / "worlds" / "test.world")
    map_file = _write(tmp_path / "map_pkg" / "maps" / "test.yaml")
    retrieval = _write(tmp_path / "nav_pkg" / "config" / "retrieval.yaml")
    bridge = _write(tmp_path / "robot_pkg" / "config" / "bridge.yaml")
    config = {
        "scene_id": "test_scene",
        "world_package": "map_pkg",
        "world_file": "worlds/test.world",
        "map_file": "package://map_pkg/maps/test.yaml",
        "graph_database": "~/graphs/test.db",
        "localization_mode": "localization",
        "spawn": {"x": -1.0, "y": 2.0, "yaw": 0.5},
        "robot": {
            "package": "robot_pkg",
            "model": "semantic_tall_rgbd",
            "entity_name": "semantic_robot",
            "camera_height_m": 1.05,
            "camera_pitch_rad": -0.1,
            "gz_rgb_topic": "/rgbd/image",
            "gz_depth_topic": "/rgbd/depth_image",
            "gz_camera_info_topic": "/rgbd/camera_info",
            "bridge_config": "package://robot_pkg/config/bridge.yaml",
        },
        "launch": {
            "headless": True,
            "start_rviz": False,
            "start_semantic": True,
            "start_operator_gui": True,
        },
        "parameter_files": {
            "retrieval": "nav_pkg/config/retrieval.yaml",
        },
    }
    config_path = _write(
        bringup / "config" / "scenes" / "test_scene.yaml",
        yaml.safe_dump(config),
    )

    shares = {
        "map_pkg": str(tmp_path / "map_pkg"),
        "nav_pkg": str(tmp_path / "nav_pkg"),
        "robot_pkg": str(tmp_path / "robot_pkg"),
    }
    loaded_path, values = load_scene_launch_config(
        "test_scene",
        bringup,
        package_share_resolver=shares.__getitem__,
    )

    assert loaded_path == config_path.resolve()
    assert values["scene_id"] == "test_scene"
    assert values["world_package"] == "map_pkg"
    assert values["world"] == str(world.resolve())
    assert values["map"] == str(map_file.resolve())
    assert values["retrieval_params_file"] == str(retrieval.resolve())
    assert values["spawn_x"] == "-1.0"
    assert values["spawn_y"] == "2.0"
    assert values["spawn_yaw"] == "0.5"
    assert values["robot_package"] == "robot_pkg"
    assert values["robot_model"] == "semantic_tall_rgbd"
    assert values["robot_entity_name"] == "semantic_robot"
    assert values["camera_mast_z"] == "1.05"
    assert values["camera_pitch_rad"] == "-0.1"
    assert values["robot_gz_rgb_topic"] == "/rgbd/image"
    assert values["robot_gz_depth_topic"] == "/rgbd/depth_image"
    assert values["robot_gz_camera_info_topic"] == "/rgbd/camera_info"
    assert values["robot_bridge_config"] == str(bridge.resolve())
    assert values["headless"] == "true"
    assert values["start_rviz"] == "false"
    assert values["start_operator_gui"] == "true"
    assert values["graph_database"].endswith("/graphs/test.db")


def test_external_relative_resources_resolve_next_to_config(tmp_path):
    config_dir = tmp_path / "external"
    world = _write(config_dir / "custom.world")
    config_path = _write(
        config_dir / "scene.yaml",
        "scene_id: external\nworld_file: custom.world\n",
    )

    _, values = load_scene_launch_config(config_path, tmp_path / "bringup")

    assert values["world"] == str(world.resolve())


def test_explicit_launch_values_win_over_yaml_defaults():
    current = {"headless": "false", "scene_id": "cli_scene"}
    applied = apply_yaml_defaults(
        current,
        {"headless": "true", "scene_id": "yaml_scene", "spawn_x": "1.0"},
    )

    assert applied == ["spawn_x"]
    assert current == {
        "headless": "false",
        "scene_id": "cli_scene",
        "spawn_x": "1.0",
    }


def test_unknown_launch_field_is_rejected(tmp_path):
    config = _write(
        tmp_path / "scene.yaml",
        "scene_id: test\nlaunch:\n  start_rivs: true\n",
    )

    with pytest.raises(ValueError, match="start_rivs"):
        load_scene_launch_config(config, tmp_path)


def test_unknown_robot_field_is_rejected(tmp_path):
    config = _write(
        tmp_path / "scene.yaml",
        "scene_id: test\nrobot:\n  camera_heigth_m: 1.0\n",
    )

    with pytest.raises(ValueError, match="camera_heigth_m"):
        load_scene_launch_config(config, tmp_path)


def test_launch_booleans_must_be_yaml_booleans(tmp_path):
    config = _write(
        tmp_path / "scene.yaml",
        "scene_id: test\nlaunch:\n  headless: 'true'\n",
    )

    with pytest.raises(ValueError, match="YAML boolean"):
        load_scene_launch_config(config, tmp_path)


def test_missing_scene_lists_checked_locations(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        resolve_scene_config("missing", tmp_path)
