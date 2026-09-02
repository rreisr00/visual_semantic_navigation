"""Hashing and immutable retrieval-configuration helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def canonical_config_bytes(config: Mapping[str, Any]) -> bytes:
    return json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_retrieval_config(
    path: str | Path,
    config: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> tuple[Path, Path, str]:
    """Write a selected offline configuration and a sidecar SHA-256.

    A configuration is freezeable only when the caller proves that calibration
    used non-empty offline train/validation data and no simulation source.
    """
    if not calibration.get("validated_offline"):
        raise ValueError("frozen configuration requires validated_offline=true")
    if int(calibration.get("n_validation_queries", 0)) < 1:
        raise ValueError("frozen configuration requires at least one validation query")
    if calibration.get("simulation_sources"):
        raise ValueError("frozen configuration cannot include simulation sources")
    payload = dict(config)
    payload["calibration"] = dict(calibration)
    payload["status"] = "frozen"
    payload["config_hash"] = config_sha256(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    digest = file_sha256(destination)
    sidecar = destination.with_suffix(".sha256")
    sidecar.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return destination, sidecar, digest


def verify_frozen_config(
    path: str | Path, expected_hash: str | None = None
) -> tuple[dict[str, Any], str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"frozen retrieval configuration not found: {source}")
    with source.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("status") != "frozen":
        raise ValueError(f"{source}: status must be 'frozen'")
    digest = file_sha256(source)
    sidecar = source.with_suffix(".sha256")
    if sidecar.is_file():
        recorded = sidecar.read_text(encoding="utf-8").split()[0]
        if recorded != digest:
            raise ValueError(f"{source}: SHA-256 does not match {sidecar}")
    if expected_hash and digest != expected_hash:
        raise ValueError(
            f"{source}: SHA-256 {digest} does not match expected {expected_hash}"
        )
    return config, digest
