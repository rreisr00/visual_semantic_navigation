"""Configuration loading and hard isolation checks for experiments."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from semantic_evaluation.core.experimental_schemas import (
    DatasetSpec,
    SchemaValidationError,
    SimulationSceneSpec,
)

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_OFFLINE_FORBIDDEN = (
    "/.ros/",
    "\\.ros\\",
    "semantic_dataset",
    "ros2_evaluation_results",
    "aws_small_house",
    "turtlebot3_house",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"configuration file not found: {source}")
    with source.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SchemaValidationError(
            str(source), "<root>", type(data).__name__, "a YAML mapping", "replace the root value"
        )
    return data


def expand_path(value: str, repo_root: str | Path) -> tuple[Path | None, list[str]]:
    """Resolve env/user/relative paths and report missing variables."""
    missing = [name for name in _ENV_PATTERN.findall(value) if not os.environ.get(name)]
    if missing:
        return None, missing
    expanded = os.path.expanduser(os.path.expandvars(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve(), []


def _walk_strings(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_strings(child, next_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{prefix}[{index}]")
    elif isinstance(value, str):
        yield prefix, value


def validate_offline_isolation(
    config: Mapping[str, Any],
    source: str = "<offline config>",
    simulation_scene_ids: Iterable[str] = (),
) -> None:
    """Fail when offline configuration references simulation data or storage."""
    forbidden = {token.lower() for token in _OFFLINE_FORBIDDEN}
    forbidden.update(scene.lower() for scene in simulation_scene_ids if scene)
    for field_name, value in _walk_strings(config):
        normalized = os.path.expanduser(value).replace("\\", "/").lower()
        token = next((item for item in forbidden if item in normalized), None)
        if token:
            raise SchemaValidationError(
                source,
                field_name,
                value,
                "a path/value independent of ROS 2 simulation",
                f"move this source to experiments/simulation (matched '{token}')",
            )
        if field_name.endswith("graph_db"):
            raise SchemaValidationError(
                source,
                field_name,
                value,
                "no graph_db field in offline configuration",
                "use a DatasetSpec and keep graph databases in simulation",
            )


def validate_separate_roots(
    offline_config: Mapping[str, Any], simulation_config: Mapping[str, Any]
) -> None:
    offline_paths = set(str(v) for v in offline_config.get("paths", {}).values())
    simulation_paths = set(str(v) for v in simulation_config.get("paths", {}).values())
    overlap = offline_paths & simulation_paths
    if overlap:
        raise SchemaValidationError(
            "offline/simulation configs",
            "paths",
            sorted(overlap),
            "physically distinct cache, results and manifest roots",
            "assign each block its own experiments/<block>/ directory",
        )


def validate_offline_config(config: Mapping[str, Any], source: str) -> None:
    required_sections = (
        "experiment", "models", "runtime", "paths", "datasets",
        "retrieval", "evaluation", "reproducibility",
    )
    for section in required_sections:
        if not isinstance(config.get(section), Mapping):
            raise SchemaValidationError(
                source, section, config.get(section), "a YAML mapping", f"add the '{section}' section"
            )
    validate_offline_isolation(config, source)


def load_dataset_specs(
    config: Mapping[str, Any], repo_root: str | Path
) -> list[DatasetSpec]:
    configs_dir, missing = expand_path(str(config["datasets"]["configs_dir"]), repo_root)
    if missing or configs_dir is None:
        raise SchemaValidationError(
            "offline config", "datasets.configs_dir", missing, "a resolvable path", "set the missing variable"
        )
    enabled = set(str(v) for v in config["datasets"].get("enabled", []))
    specs: list[DatasetSpec] = []
    for dataset_id in sorted(enabled):
        path = configs_dir / f"{dataset_id}.yaml"
        data = load_yaml(path)
        spec = DatasetSpec.from_mapping(data, str(path))
        if spec.dataset_id != dataset_id:
            raise SchemaValidationError(
                str(path), "dataset_id", spec.dataset_id, dataset_id, "align filename and dataset_id"
            )
        specs.append(spec)
    return specs


def load_simulation_scenes(
    config: Mapping[str, Any], repo_root: str | Path
) -> list[SimulationSceneSpec]:
    configs_dir, missing = expand_path(str(config["scenes"]["configs_dir"]), repo_root)
    if missing or configs_dir is None:
        raise SchemaValidationError(
            "simulation config", "scenes.configs_dir", missing, "a resolvable path", "set the missing variable"
        )
    return [
        SimulationSceneSpec.from_mapping(load_yaml(path), str(path))
        for path in sorted(configs_dir.glob("*.yaml"))
    ]


def dataset_availability(spec: DatasetSpec, repo_root: str | Path) -> tuple[bool, str]:
    root, missing = expand_path(spec.root, repo_root)
    if missing:
        names = ", ".join(missing)
        return False, (
            f"dataset '{spec.dataset_id}' omitted: set {names}; expected root from "
            f"{spec.config_path} with adapter '{spec.adapter}'"
        )
    if root is None or not root.is_dir():
        return False, (
            f"dataset '{spec.dataset_id}' omitted: root '{root}' does not exist; "
            f"update {spec.config_path}"
        )
    return True, str(root)

