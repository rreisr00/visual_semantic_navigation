"""Adapters from external offline datasets to the existing semantic models.

Large datasets are never downloaded.  Each adapter reports the exact missing
root or normalized index it expects and returns an empty, explicitly skipped
bundle when the source is unavailable.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from semantic_evaluation.core.config_validation import expand_path
from semantic_evaluation.core.evaluation_statistics import (
    normalize_label,
    normalize_predicate,
)
from semantic_evaluation.core.experimental_schemas import DatasetSpec
from semantic_navigation_core.types import (
    ObjectObservation,
    Observation,
    SemanticNode,
    SpatialRelation,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class DatasetBundle:
    dataset_id: str
    split: str
    nodes: list[SemanticNode] = field(default_factory=list)
    topology_edges: list[tuple[str, str]] = field(default_factory=list)
    object_ground_truth: dict[str, list[ObjectObservation]] = field(default_factory=dict)
    relation_ground_truth: dict[str, list[SpatialRelation]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""

    def node_ids(self) -> list[str]:
        return [node.node_id for node in self.nodes]

    @property
    def scene_id(self) -> str:
        """Compatibility identifier for the existing experiment runner."""
        return self.dataset_id

    def room_by_node(self) -> dict[str, str | None]:
        return {node.node_id: node.room_id for node in self.nodes}

    def position_by_node(self) -> dict[str, tuple[float, float]]:
        return {node.node_id: node.position[:2] for node in self.nodes}


class DatasetAdapter:
    adapter_name = "base"

    def load(self, spec: DatasetSpec, repo_root: str | Path) -> DatasetBundle:
        raise NotImplementedError

    @staticmethod
    def _root_or_skipped(
        spec: DatasetSpec, repo_root: str | Path
    ) -> tuple[Path | None, DatasetBundle | None]:
        root, missing = expand_path(spec.root, repo_root)
        if missing:
            variables = ", ".join(missing)
            return None, DatasetBundle(
                spec.dataset_id,
                spec.split.name,
                skipped=True,
                skip_reason=(
                    f"set {variables}; expected the '{spec.adapter}' dataset root "
                    f"configured in {spec.config_path}"
                ),
            )
        if root is None or not root.is_dir():
            return None, DatasetBundle(
                spec.dataset_id,
                spec.split.name,
                skipped=True,
                skip_reason=f"dataset root does not exist: {root}; update {spec.config_path}",
            )
        return root, None


class SiglipRoomsAdapter(DatasetAdapter):
    adapter_name = "siglip_rooms"

    def load(self, spec: DatasetSpec, repo_root: str | Path) -> DatasetBundle:
        root, skipped = self._root_or_skipped(spec, repo_root)
        if skipped:
            return skipped
        nodes: list[SemanticNode] = []
        assert root is not None
        for room_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            images = sorted(
                path for path in room_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            limit = spec.split.max_images_per_room
            if limit is not None:
                images = images[:limit]
            for index, image in enumerate(images, start=1):
                node_id = f"{room_dir.name}_{index:04d}"
                nodes.append(SemanticNode(
                    node_id=node_id,
                    room_id=room_dir.name,
                    observations=[Observation(
                        observation_id=f"{node_id}:view_000",
                        image_path=str(image),
                    )],
                ))
        return DatasetBundle(
            spec.dataset_id,
            spec.split.name,
            nodes=nodes,
            metadata={"root": str(root), "node_semantics": "independent_image"},
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(item)
    return rows


def _box(item: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    raw = item.get("bbox") or item.get("box")
    if raw is not None and len(raw) == 4:
        values = tuple(float(value) for value in raw)
        mode = item.get("bbox_mode", "xyxy")
        if mode == "xywh":
            x, y, width, height = values
            return x, y, x + width, y + height
        return values
    if all(key in item for key in ("x", "y", "w", "h")):
        x, y = float(item["x"]), float(item["y"])
        return x, y, x + float(item["w"]), y + float(item["h"])
    return None


class SunRgbdAdapter(DatasetAdapter):
    """SUN RGB-D adapter using a deterministic normalized JSONL index.

    The index defaults to ``sunrgbd_index.jsonl`` and contains one object per
    image: ``sample_id``, ``image_path``, ``room_label`` and ``objects``.  Each
    object contains ``label`` and a box in ``bbox`` (xyxy) or ``x/y/w/h``.
    This keeps the adapter independent of MATLAB while preserving official
    identifiers and annotations.
    """

    adapter_name = "sunrgbd"

    def load(self, spec: DatasetSpec, repo_root: str | Path) -> DatasetBundle:
        root, skipped = self._root_or_skipped(spec, repo_root)
        if skipped:
            return skipped
        assert root is not None
        metadata_file = spec.annotations.get("metadata_file") or "sunrgbd_index.jsonl"
        index = Path(metadata_file)
        if not index.is_absolute():
            index = root / index
        if not index.is_file():
            return DatasetBundle(
                spec.dataset_id,
                spec.split.name,
                skipped=True,
                skip_reason=(
                    f"SUN RGB-D index missing: {index}. Expected JSONL fields: "
                    "sample_id, image_path, room_label, objects[{label,bbox}]"
                ),
            )
        nodes: list[SemanticNode] = []
        ground_truth: dict[str, list[ObjectObservation]] = {}
        unmapped: set[str] = set()
        mapping = {
            normalize_label(key): normalize_label(value)
            for key, value in dict(spec.annotations.get("class_mapping") or {}).items()
        }
        for item in _read_jsonl(index):
            sample_id = str(item.get("sample_id") or item.get("image_id") or "")
            if not sample_id:
                raise ValueError(f"{index}: row without sample_id")
            image = Path(str(item.get("image_path", "")))
            if not image.is_absolute():
                image = root / image
            observation_id = f"{sample_id}:view_000"
            objects = []
            for annotation in item.get("objects", []):
                label = normalize_label(str(annotation.get("label", "")))
                if not label:
                    continue
                objects.append(ObjectObservation(label=label, box=_box(annotation)))
                if label not in mapping:
                    unmapped.add(label)
            ground_truth[observation_id] = objects
            nodes.append(SemanticNode(
                node_id=sample_id,
                room_id=(str(item["room_label"]) if item.get("room_label") else None),
                observations=[Observation(observation_id, image_path=str(image))],
            ))
        return DatasetBundle(
            spec.dataset_id,
            spec.split.name,
            nodes=nodes,
            object_ground_truth=ground_truth,
            metadata={
                "root": str(root),
                "index": str(index),
                "class_mapping": mapping,
                "unmapped_ground_truth_classes": sorted(unmapped),
            },
        )


class MatterportR2RAdapter(DatasetAdapter):
    """Matterport3D/R2R nodes grouped strictly by ``(scan, viewpoint)``."""

    adapter_name = "matterport3d_r2r"

    def load(self, spec: DatasetSpec, repo_root: str | Path) -> DatasetBundle:
        root, skipped = self._root_or_skipped(spec, repo_root)
        if skipped:
            return skipped
        assert root is not None
        r2r_value = spec.r2r_root or spec.root
        r2r_root, missing = expand_path(r2r_value, repo_root)
        if missing or r2r_root is None or not r2r_root.is_dir():
            return DatasetBundle(
                spec.dataset_id,
                spec.split.name,
                skipped=True,
                skip_reason=(
                    f"R2R root unavailable ({r2r_value}); set {', '.join(missing) or 'R2R_ROOT'} "
                    "and provide connectivity plus views_manifest.jsonl"
                ),
            )
        manifest_value = spec.annotations.get("file") or "views_manifest.jsonl"
        manifest = Path(str(manifest_value))
        if not manifest.is_absolute():
            manifest = root / manifest
        if not manifest.is_file():
            return DatasetBundle(
                spec.dataset_id,
                spec.split.name,
                skipped=True,
                skip_reason=(
                    f"Matterport view manifest missing: {manifest}. Expected JSONL fields: "
                    "scan_id, viewpoint_id, angle_degrees, image_path"
                ),
            )
        allowed_scans = self._scan_ids(spec, r2r_root)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in _read_jsonl(manifest):
            scan_id = str(row.get("scan_id", ""))
            viewpoint_id = str(row.get("viewpoint_id", ""))
            if not scan_id or not viewpoint_id:
                raise ValueError(f"{manifest}: scan_id and viewpoint_id are required")
            if allowed_scans and scan_id not in allowed_scans:
                continue
            grouped[(scan_id, viewpoint_id)].append(row)
        expected_angles = set(spec.split.view_angles_degrees)
        nodes: list[SemanticNode] = []
        annotations: dict[str, Any] = {}
        for (scan_id, viewpoint_id), views in sorted(grouped.items()):
            angles = [int(round(float(view["angle_degrees"]))) % 360 for view in views]
            if len(angles) != len(set(angles)):
                raise ValueError(f"{manifest}: duplicate angle for {scan_id}/{viewpoint_id}")
            selected = [
                view for view, angle in zip(views, angles) if angle in expected_angles
            ]
            if not selected:
                continue
            node_id = f"{scan_id}:{viewpoint_id}"
            observations: list[Observation] = []
            for view in sorted(selected, key=lambda item: float(item["angle_degrees"])):
                image = Path(str(view["image_path"]))
                if not image.is_absolute():
                    image = root / image
                angle = int(round(float(view["angle_degrees"]))) % 360
                observations.append(Observation(
                    observation_id=f"{node_id}:view_{angle:03d}", image_path=str(image)
                ))
            first = views[0]
            nodes.append(SemanticNode(
                node_id=node_id,
                room_id=(str(first["room_label"]) if first.get("room_label") else None),
                observations=observations,
            ))
            annotations[node_id] = {
                "scan_id": scan_id,
                "viewpoint_id": viewpoint_id,
                "objects": list(first.get("objects") or []),
                "relations": list(first.get("relations") or []),
                "queries": list(first.get("queries") or []),
                "valid_node_ids": list(first.get("valid_node_ids") or []),
            }
        edges = self._connectivity_edges(r2r_root, spec, {node.node_id for node in nodes})
        building_split_by_scan = self._building_split_map(spec, r2r_root)
        return DatasetBundle(
            spec.dataset_id,
            spec.split.name,
            nodes=nodes,
            topology_edges=edges,
            metadata={
                "root": str(root), "r2r_root": str(r2r_root),
                "viewpoint_annotations": annotations,
                "split_scan_ids": sorted(allowed_scans),
                "building_split_by_scan": building_split_by_scan,
            },
        )

    @staticmethod
    def _scan_ids(spec: DatasetSpec, r2r_root: Path) -> set[str]:
        if spec.split.scan_ids:
            return set(spec.split.scan_ids)
        if spec.split.split_file:
            path = Path(spec.split.split_file)
            if not path.is_absolute():
                path = r2r_root / path
            if path.is_file():
                if path.suffix == ".json":
                    values = json.loads(path.read_text(encoding="utf-8"))
                else:
                    values = path.read_text(encoding="utf-8").splitlines()
                return {str(value).strip() for value in values if str(value).strip()}
        return set()

    @staticmethod
    def _connectivity_edges(
        r2r_root: Path, spec: DatasetSpec, node_ids: set[str]
    ) -> list[tuple[str, str]]:
        directory = r2r_root / spec.split.connectivity_dir
        scans = sorted({node_id.split(":", 1)[0] for node_id in node_ids})
        edges: set[tuple[str, str]] = set()
        for scan_id in scans:
            path = directory / f"{scan_id}_connectivity.json"
            if not path.is_file():
                continue
            rows = json.loads(path.read_text(encoding="utf-8"))
            included = [row for row in rows if row.get("included", True)]
            for index, row in enumerate(included):
                source = f"{scan_id}:{row['image_id']}"
                unobstructed = row.get("unobstructed", [])
                for target_index, connected in enumerate(unobstructed):
                    if not connected or target_index >= len(rows):
                        continue
                    target = f"{scan_id}:{rows[target_index]['image_id']}"
                    if source in node_ids and target in node_ids and source != target:
                        edges.add(tuple(sorted((source, target))))
        return sorted(edges)

    @staticmethod
    def _building_split_map(spec: DatasetSpec, r2r_root: Path) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for split_name, value in spec.split.building_splits.items():
            path = Path(value)
            if not path.is_absolute():
                path = r2r_root / path
            if not path.is_file():
                continue
            if path.suffix == ".json":
                values = json.loads(path.read_text(encoding="utf-8"))
            else:
                values = path.read_text(encoding="utf-8").splitlines()
            for scan_id in values:
                normalized = str(scan_id).strip()
                if normalized:
                    if normalized in mapping:
                        raise ValueError(
                            f"scan '{normalized}' appears in multiple building splits"
                        )
                    mapping[normalized] = str(split_name)
        return mapping


class VisualGenomeAdapter(DatasetAdapter):
    """Filtered Visual Genome relations; never represented as navigation nodes."""

    adapter_name = "visual_genome"

    def load(self, spec: DatasetSpec, repo_root: str | Path) -> DatasetBundle:
        root, skipped = self._root_or_skipped(spec, repo_root)
        if skipped:
            return skipped
        assert root is not None
        relation_path = root / str(spec.annotations.get("relationships_file", "relationships.json"))
        if not relation_path.is_file():
            return DatasetBundle(
                spec.dataset_id,
                spec.split.name,
                skipped=True,
                skip_reason=f"Visual Genome relationships file missing: {relation_path}",
            )
        allowed_ids = self._indoor_ids(root, spec)
        data = json.loads(relation_path.read_text(encoding="utf-8"))
        relations_by_image: dict[str, list[SpatialRelation]] = {}
        boxes_by_image: dict[str, list[ObjectObservation]] = {}
        ignored_predicates: CounterLike = CounterLike()
        for image_entry in data:
            image_id = str(image_entry.get("image_id", ""))
            if allowed_ids is not None and image_id not in allowed_ids:
                continue
            relations: list[SpatialRelation] = []
            objects: dict[str, ObjectObservation] = {}
            for item in image_entry.get("relationships", []):
                predicate = normalize_predicate(str(item.get("predicate", "")))
                if predicate is None:
                    ignored_predicates.add(str(item.get("predicate", "")))
                    continue
                subject_data = item.get("subject") or {}
                object_data = item.get("object") or {}
                subject = self._object_name(subject_data)
                obj = self._object_name(object_data)
                if not subject or not obj:
                    continue
                relations.append(SpatialRelation(subject, predicate, obj, 1.0))
                objects[f"subject:{item.get('relationship_id', len(objects))}"] = ObjectObservation(
                    subject, box=_box(subject_data)
                )
                objects[f"object:{item.get('relationship_id', len(objects))}"] = ObjectObservation(
                    obj, box=_box(object_data)
                )
            if relations:
                relations_by_image[image_id] = relations
                boxes_by_image[image_id] = list(objects.values())
        return DatasetBundle(
            spec.dataset_id,
            spec.split.name,
            nodes=[],
            object_ground_truth=boxes_by_image,
            relation_ground_truth=relations_by_image,
            metadata={
                "root": str(root),
                "n_relation_images": len(relations_by_image),
                "ignored_predicates": ignored_predicates.as_dict(),
                "navigable": False,
            },
        )

    @staticmethod
    def _object_name(data: Mapping[str, Any]) -> str:
        raw = data.get("name")
        if not raw:
            names = data.get("names") or []
            raw = names[0] if names else ""
        return normalize_label(str(raw)) if raw else ""

    @staticmethod
    def _indoor_ids(root: Path, spec: DatasetSpec) -> set[str] | None:
        value = spec.annotations.get("indoor_image_ids_file")
        if not value:
            return None
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(
                f"Visual Genome indoor image id list not found: {path}"
            )
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


class CounterLike:
    def __init__(self) -> None:
        self._values: dict[str, int] = {}

    def add(self, value: str) -> None:
        key = normalize_label(value) or "<empty>"
        self._values[key] = self._values.get(key, 0) + 1

    def as_dict(self) -> dict[str, int]:
        return dict(sorted(self._values.items()))


_ADAPTERS = {
    adapter.adapter_name: adapter
    for adapter in (
        SiglipRoomsAdapter(), SunRgbdAdapter(), MatterportR2RAdapter(),
        VisualGenomeAdapter(),
    )
}


def get_dataset_adapter(name: str) -> DatasetAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown dataset adapter {name!r}; expected one of {sorted(_ADAPTERS)}"
        ) from exc


def load_dataset(spec: DatasetSpec, repo_root: str | Path) -> DatasetBundle:
    if not spec.enabled:
        return DatasetBundle(
            spec.dataset_id, spec.split.name, skipped=True, skip_reason="disabled in config"
        )
    return get_dataset_adapter(spec.adapter).load(spec, repo_root)


def validate_dataset(bundle: DatasetBundle, check_images: bool = True) -> list[dict[str, str]]:
    """Return structured issues for missing/corrupt/duplicate dataset inputs."""
    issues: list[dict[str, str]] = []
    if bundle.skipped:
        return [{"severity": "skip", "item_id": bundle.dataset_id, "message": bundle.skip_reason}]
    if not bundle.nodes and bundle.metadata.get("navigable", True):
        issues.append({"severity": "error", "item_id": bundle.dataset_id, "message": "dataset has no nodes"})
    seen_nodes: set[str] = set()
    image_hashes: dict[str, str] = {}
    for node in bundle.nodes:
        if node.node_id in seen_nodes:
            issues.append({"severity": "error", "item_id": node.node_id, "message": "duplicate node id"})
        seen_nodes.add(node.node_id)
        if not node.observations:
            issues.append({"severity": "error", "item_id": node.node_id, "message": "node has no observations"})
        for observation in node.observations:
            if not observation.image_path:
                issues.append({"severity": "error", "item_id": observation.observation_id, "message": "missing image path"})
                continue
            path = Path(observation.image_path)
            if not path.is_file():
                issues.append({"severity": "error", "item_id": observation.observation_id,
                               "message": f"image not found: {path}"})
                continue
            if check_images:
                try:
                    from PIL import Image
                    with Image.open(path) as image:
                        image.verify()
                except (OSError, ValueError) as exc:
                    issues.append({"severity": "error", "item_id": observation.observation_id,
                                   "message": f"corrupt image: {exc}"})
                    continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in image_hashes:
                issues.append({"severity": "warning", "item_id": observation.observation_id,
                               "message": f"duplicate image content of {image_hashes[digest]}"})
            else:
                image_hashes[digest] = observation.observation_id
    return issues


def matterport_annotation_template(bundle: DatasetBundle) -> str:
    """YAML template containing only real scan/viewpoint identifiers."""
    if bundle.dataset_id != "matterport3d":
        raise ValueError("Matterport3D bundle required")
    rows = ["viewpoints:"]
    annotations = dict(bundle.metadata.get("viewpoint_annotations") or {})
    for node in bundle.nodes:
        data = annotations.get(node.node_id, {})
        rows.extend([
            f"  - scan_id: {data.get('scan_id', node.node_id.split(':', 1)[0])}",
            f"    node_id: {node.node_id}",
            f"    room_label: {node.room_id if node.room_id is not None else 'null'}",
            "    objects: []",
            "    relations: []",
            "    queries: []",
            "    valid_node_ids: []",
        ])
    return "\n".join(rows) + "\n"
