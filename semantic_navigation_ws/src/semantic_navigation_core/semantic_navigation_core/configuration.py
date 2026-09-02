"""Deterministic configuration loading and hashing for reproducible runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


FROZEN_FIELDS = (
    "siglip_checkpoint",
    "yolo_checkpoint",
    "preprocessing",
    "multiview_aggregation",
    "retrieval_method",
    "retrieval_weights",
    "rejection_threshold",
    "category_mappings",
    "relation_extractor_version",
)


def canonical_json(data: Mapping[str, Any]) -> str:
    """Return stable UTF-8 JSON independent of YAML key ordering."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def configuration_hash(data: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical mapping representation."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def load_frozen_config(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load, validate and hash a frozen semantic retrieval configuration."""
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError("frozen retrieval config must contain a YAML mapping")
    missing = [field for field in FROZEN_FIELDS if field not in loaded]
    if missing:
        raise ValueError(f"frozen retrieval config missing fields: {', '.join(missing)}")
    return loaded, configuration_hash(loaded)


def verify_configuration_hash(data: Mapping[str, Any], expected_hash: str) -> str:
    """Return the actual hash or raise when it differs from the expected hash."""
    actual = configuration_hash(data)
    if expected_hash and actual != expected_hash:
        raise ValueError(
            f"configuration hash mismatch: expected {expected_hash}, got {actual}"
        )
    return actual

