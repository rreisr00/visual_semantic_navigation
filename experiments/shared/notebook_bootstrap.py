"""Bootstrap for offline and simulation notebooks."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_PURE_PACKAGES = (
    "semantic_navigation_core",
    "semantic_vision_core",
    "semantic_evaluation",
)


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve().parent).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "semantic_navigation_ws" / "src").is_dir():
            return candidate
    raise RuntimeError("repository root with semantic_navigation_ws/src not found")


def _add_project_packages(repo_root: Path) -> None:
    source = repo_root / "semantic_navigation_ws" / "src"
    for package in _PURE_PACKAGES:
        value = str(source / package)
        if value not in sys.path:
            sys.path.insert(0, value)


# Importing this helper is the first operation in every notebook.  Make the
# repository's pure Python packages available immediately so later imports in
# the same cell do not depend on an externally configured PYTHONPATH.
_add_project_packages(find_repo_root())


def _cuda13_fix() -> None:
    site = next((value for value in sys.path if "site-packages" in value), None)
    if site:
        library = Path(site) / "nvidia" / "cu13" / "lib"
        current = os.environ.get("LD_LIBRARY_PATH", "")
        if library.is_dir() and str(library) not in current.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = str(library) + os.pathsep + current


def get_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    from semantic_evaluation.core.config_validation import expand_path
    path, missing = expand_path(value, repo_root)
    if missing or path is None:
        raise RuntimeError(f"path '{value}' requires environment variables {missing}")
    return path


def bootstrap_offline(config_path: str | None = None) -> dict[str, Any]:
    repo_root = find_repo_root()
    _add_project_packages(repo_root)
    _cuda13_fix()
    from semantic_evaluation.core.config_validation import (
        load_dataset_specs,
        load_yaml,
        validate_offline_config,
    )
    from semantic_evaluation.core.embedding_cache import set_seeds

    path = Path(config_path) if config_path else (
        repo_root / "experiments/offline/config/offline_experiment_config.yaml"
    )
    config = load_yaml(path)
    validate_offline_config(config, str(path))
    specs = load_dataset_specs(config, repo_root)
    seed = int(config["experiment"]["seed"])
    set_seeds(seed)
    return {
        "repo_root": repo_root,
        "config_path": path,
        "config": config,
        "dataset_specs": specs,
        "device": get_device(str(config["runtime"].get("device", "auto"))),
    }


def bootstrap_simulation(config_path: str | None = None) -> dict[str, Any]:
    repo_root = find_repo_root()
    _add_project_packages(repo_root)
    from semantic_evaluation.core.config_validation import (
        load_simulation_scenes,
        load_yaml,
    )
    from semantic_evaluation.core.reproducibility import verify_frozen_config

    path = Path(config_path) if config_path else (
        repo_root / "experiments/simulation/config/simulation_experiment_config.yaml"
    )
    config = load_yaml(path)
    scenes = load_simulation_scenes(config, repo_root)
    frozen_path = resolve_repo_path(
        repo_root, str(config["experiment"]["frozen_retrieval_config"])
    )
    expected = config["experiment"].get("expected_config_hash")
    frozen_config, frozen_hash = verify_frozen_config(frozen_path, expected)
    return {
        "repo_root": repo_root,
        "config_path": path,
        "config": config,
        "scenes": scenes,
        "frozen_config": frozen_config,
        "frozen_config_hash": frozen_hash,
    }
