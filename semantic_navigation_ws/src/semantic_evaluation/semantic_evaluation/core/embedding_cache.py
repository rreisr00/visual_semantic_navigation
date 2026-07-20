"""Disk cache for embeddings/detections + reproducibility helpers — no rclpy.

Cache keys hash the model identifier plus the exact content (file bytes or
query text), so changing the checkpoint or an image automatically invalidates
its entries. Arrays are stored as ``.npy``, structured data as ``.json``.

Also provides :func:`set_seeds` (Python / NumPy / PyTorch / CUDA) and
:func:`collect_manifest` (git commit, versions, device, config) so every
notebook run is reproducible and self-describing.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np


class EmbeddingCache:
    """Content-addressed cache under ``root`` (created on first use)."""

    def __init__(self, root: str) -> None:
        self.root = os.path.expanduser(root)
        os.makedirs(self.root, exist_ok=True)
        self.hits = 0
        self.misses = 0

    # ── keys ────────────────────────────────────────────────────────────── #

    @staticmethod
    def _digest(*parts: str) -> str:
        h = hashlib.sha1()
        for part in parts:
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    @staticmethod
    def file_hash(path: str) -> str:
        """SHA-1 of a file's bytes (invalidates the cache when it changes)."""
        h = hashlib.sha1()
        with open(os.path.expanduser(path), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def image_key(self, model_id: str, image_path: str) -> str:
        return self._digest("image", model_id, self.file_hash(image_path))

    def text_key(self, model_id: str, text: str) -> str:
        return self._digest("text", model_id, text)

    def detections_key(
        self, model_path: str, image_path: str, confidence: float
    ) -> str:
        model_identity = (
            self.file_hash(model_path) if os.path.isfile(os.path.expanduser(model_path))
            else os.path.basename(model_path)
        )
        return self._digest(
            "detections", model_identity,
            f"{confidence:.3f}", self.file_hash(image_path),
        )

    def crops_key(
        self, model_id: str, image_path: str, detections_key: str
    ) -> str:
        return self._digest(
            "crops", model_id, self.file_hash(image_path), detections_key
        )

    # ── array / json storage ────────────────────────────────────────────── #

    def _path(self, key: str, ext: str) -> str:
        return os.path.join(self.root, f"{key}{ext}")

    def get_array(self, key: str) -> np.ndarray | None:
        path = self._path(key, ".npy")
        if os.path.isfile(path):
            self.hits += 1
            return np.load(path)
        self.misses += 1
        return None

    def put_array(self, key: str, value: np.ndarray) -> None:
        np.save(self._path(key, ".npy"), np.asarray(value))

    def get_arrays(self, key: str) -> list[np.ndarray] | None:
        path = self._path(key, ".npz")
        if not os.path.isfile(path):
            self.misses += 1
            return None
        self.hits += 1
        with np.load(path) as data:
            return [data[k] for k in sorted(data.files)]

    def put_arrays(self, key: str, values: list[np.ndarray]) -> None:
        np.savez(
            self._path(key, ".npz"),
            **{f"a{i:04d}": np.asarray(v) for i, v in enumerate(values)},
        )

    def get_json(self, key: str) -> Any | None:
        path = self._path(key, ".json")
        if not os.path.isfile(path):
            self.misses += 1
            return None
        self.hits += 1
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def put_json(self, key: str, value: Any) -> None:
        with open(self._path(key, ".json"), "w", encoding="utf-8") as fh:
            json.dump(value, fh)

    def clear(self) -> int:
        """Delete every cache entry; returns how many files were removed."""
        removed = 0
        for name in os.listdir(self.root):
            if name.endswith((".npy", ".npz", ".json")):
                os.remove(os.path.join(self.root, name))
                removed += 1
        return removed

    def stats(self) -> dict[str, int]:
        """Cache reuse counters for the current process."""
        return {"hits": self.hits, "misses": self.misses}


# ── Reproducibility ──────────────────────────────────────────────────────── #


def set_seeds(seed: int) -> None:
    """Fix Python, NumPy and (if installed) PyTorch/CUDA seeds."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_commit(repo_dir: str | None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _git_dirty(repo_dir: str | None) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("numpy", "torch", "transformers", "ultralytics", "pandas"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def collect_manifest(
    config: Mapping[str, Any],
    repo_dir: str | None = None,
    device: str = "unknown",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run manifest: everything needed to reproduce this execution."""
    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_dir),
        "git_dirty": _git_dirty(repo_dir),
        "platform": platform.platform(),
        "device": device,
        "versions": _package_versions(),
        "config": dict(config),
    }
    from semantic_evaluation.core.reproducibility import config_sha256

    manifest["config_hash"] = config_sha256(config)
    if extra:
        manifest.update(dict(extra))
    return manifest


def save_manifest(path: str, manifest: Mapping[str, Any]) -> str:
    """Write the manifest as pretty JSON next to the run's results."""
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return path
