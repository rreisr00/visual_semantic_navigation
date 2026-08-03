"""Load simulation launch arguments from a scene YAML file.

The scene files are deliberately not ROS parameter files: they describe the
resources and launch-time choices that select a complete experiment scene.
Node parameter YAML files can be selected from the same document through the
``parameter_files`` mapping.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import yaml


_TOP_LEVEL_FIELDS = {
    "scene_id": "scene_id",
    "world_file": "world",
    "map_file": "map",
    "graph_database": "graph_database",
    "localization_mode": "localization_mode",
    "world_name": "world_name",
    "camera_mast_z": "camera_mast_z",
}

_SPAWN_FIELDS = {
    "x": "spawn_x",
    "y": "spawn_y",
    "yaw": "spawn_yaw",
}

_PARAMETER_FILE_FIELDS = {
    "nav2": "nav2_params_file",
    "rviz": "rviz_config_file",
    "vision": "vision_params_file",
    "knowledge_graph": "knowledge_graph_params_file",
    "retrieval": "retrieval_params_file",
    "mapping": "mapping_params_file",
}

_LAUNCH_FIELDS = {
    "use_sim_time",
    "map",
    "world",
    "camera_mast_z",
    "spawn_x",
    "spawn_y",
    "spawn_yaw",
    "world_name",
    "scene_id",
    "graph_database",
    "start_semantic",
    "start_rviz",
    "start_auto_mapping",
    "headless",
    "localization_mode",
    *_PARAMETER_FILE_FIELDS.values(),
}

_BOOLEAN_FIELDS = {
    "use_sim_time",
    "start_semantic",
    "start_rviz",
    "start_auto_mapping",
    "headless",
}

_RESOURCE_FIELDS = {
    "world",
    "map",
    *_PARAMETER_FILE_FIELDS.values(),
}


def resolve_scene_config(reference: str, bringup_share: str | Path) -> Path:
    """Resolve either a YAML path or a bundled scene name."""
    raw = os.path.expandvars(os.path.expanduser(str(reference).strip()))
    if not raw:
        raise ValueError("scene_config must not be empty")

    requested = Path(raw)
    candidates = [requested]
    if not requested.is_absolute():
        scene_name = requested if requested.suffix else requested.with_suffix(".yaml")
        candidates.append(Path(bringup_share) / "config" / "scenes" / scene_name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"scene configuration '{reference}' was not found; checked: {rendered}"
    )


def load_scene_launch_config(
    reference: str,
    bringup_share: str | Path,
    *,
    package_share_resolver: Callable[[str], str] | None = None,
) -> tuple[Path, dict[str, str]]:
    """Return the scene file path and normalized launch-argument defaults."""
    if package_share_resolver is None:
        package_share_resolver = _ament_package_share
    config_path = resolve_scene_config(reference, bringup_share)
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"scene configuration '{config_path}' must contain a mapping")

    values: dict[str, object] = {}
    for yaml_name, launch_name in _TOP_LEVEL_FIELDS.items():
        if yaml_name in loaded:
            values[launch_name] = loaded[yaml_name]

    spawn = loaded.get("spawn", {})
    if spawn is None:
        spawn = {}
    if not isinstance(spawn, dict):
        raise ValueError("scene configuration field 'spawn' must be a mapping")
    unknown_spawn = sorted(set(spawn) - set(_SPAWN_FIELDS))
    if unknown_spawn:
        raise ValueError(f"unknown spawn fields: {', '.join(unknown_spawn)}")
    for yaml_name, launch_name in _SPAWN_FIELDS.items():
        if yaml_name in spawn:
            values[launch_name] = spawn[yaml_name]

    launch = loaded.get("launch", {})
    if launch is None:
        launch = {}
    if not isinstance(launch, dict):
        raise ValueError("scene configuration field 'launch' must be a mapping")
    unknown_launch = sorted(set(launch) - _LAUNCH_FIELDS)
    if unknown_launch:
        raise ValueError(f"unknown launch fields: {', '.join(unknown_launch)}")
    values.update(launch)

    parameter_files = loaded.get("parameter_files", {})
    if parameter_files is None:
        parameter_files = {}
    if not isinstance(parameter_files, dict):
        raise ValueError(
            "scene configuration field 'parameter_files' must be a mapping"
        )
    unknown_parameter_files = sorted(
        set(parameter_files) - set(_PARAMETER_FILE_FIELDS)
    )
    if unknown_parameter_files:
        raise ValueError(
            "unknown parameter_files fields: " + ", ".join(unknown_parameter_files)
        )
    for yaml_name, launch_name in _PARAMETER_FILE_FIELDS.items():
        if yaml_name in parameter_files:
            values[launch_name] = parameter_files[yaml_name]

    if not str(values.get("scene_id", "")).strip():
        raise ValueError(f"scene configuration '{config_path}' requires scene_id")
    mode = str(values.get("localization_mode", "localization"))
    if mode not in {"localization", "slam"}:
        raise ValueError("localization_mode must be 'localization' or 'slam'")

    normalized: dict[str, str] = {}
    for name, value in values.items():
        if name in _BOOLEAN_FIELDS:
            if not isinstance(value, bool):
                raise ValueError(f"launch field '{name}' must be a YAML boolean")
            normalized[name] = "true" if value else "false"
            continue
        if isinstance(value, (dict, list)) or value is None:
            raise ValueError(f"launch field '{name}' must be a scalar")
        normalized[name] = str(value)

    for name in _RESOURCE_FIELDS & normalized.keys():
        if normalized[name]:
            normalized[name] = str(_resolve_resource(
                normalized[name],
                config_path.parent,
                Path(bringup_share),
                package_share_resolver,
            ))

    if "graph_database" in normalized:
        normalized["graph_database"] = os.path.expandvars(
            os.path.expanduser(normalized["graph_database"])
        )
    return config_path, normalized


def apply_yaml_defaults(
    launch_configurations: dict[str, object],
    yaml_values: dict[str, str],
) -> list[str]:
    """Apply YAML values without replacing parent/CLI launch arguments."""
    applied: list[str] = []
    for name, value in yaml_values.items():
        if name not in launch_configurations:
            launch_configurations[name] = value
            applied.append(name)
    return applied


def _resolve_resource(
    value: str,
    config_directory: Path,
    bringup_share: Path,
    package_share_resolver: Callable[[str], str],
) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if expanded.startswith("package://"):
        package_path = expanded[len("package://"):]
        package, separator, relative = package_path.partition("/")
        if not package or not separator or not relative:
            raise ValueError(f"invalid package resource URI: {value}")
        candidate = Path(package_share_resolver(package)) / relative
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"package resource does not exist: {candidate}")

    requested = Path(expanded)
    candidates: list[Path] = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.append(config_directory / requested)
        if len(requested.parts) > 1:
            try:
                package_share = Path(package_share_resolver(requested.parts[0]))
            except Exception:  # package lookup provides its own detailed error later
                pass
            else:
                candidates.append(package_share.joinpath(*requested.parts[1:]))
        candidates.append(bringup_share / requested)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"resource '{value}' was not found; checked: {rendered}")


def _ament_package_share(package_name: str) -> str:
    """Import ament lazily so path-only YAML validation stays ROS-independent."""
    from ament_index_python.packages import get_package_share_directory

    return get_package_share_directory(package_name)
