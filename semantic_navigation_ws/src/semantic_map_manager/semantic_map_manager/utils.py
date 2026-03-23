from __future__ import annotations

import sys
from pathlib import Path
from typing import Type

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Placeholder cosine similarity implementation using numpy."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _append_workspace_python_paths() -> None:
    current = Path(__file__).resolve()
    src_dir = current.parents[2]
    kg_python_root = src_dir / "knowledge_graph" / "knowledge_graph"

    if kg_python_root.exists():
        kg_python_root_str = str(kg_python_root)
        if kg_python_root_str not in sys.path:
            sys.path.insert(0, kg_python_root_str)


def resolve_graph_manager_class() -> Type:
    """Resolve GraphManager-like class from knowledge_graph, with Jazzy-safe fallback."""
    _append_workspace_python_paths()
    try:
        from knowledge_graph import KnowledgeGraph

        return KnowledgeGraph
    except ImportError:
        print("Warning: Could not import KnowledgeGraph from knowledge_graph package. "
              "Please ensure the knowledge_graph package is installed for full functionality.")


def create_graph_manager_instance():
    """Create a graph manager instance regardless of upstream API flavor."""
    manager_cls = resolve_graph_manager_class()
    if hasattr(manager_cls, "get_instance"):
        return manager_cls.get_instance()
    return manager_cls()