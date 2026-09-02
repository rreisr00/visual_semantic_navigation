import json

from PIL import Image

from semantic_evaluation.core.dataset_adapters import (
    load_dataset,
    matterport_annotation_template,
    validate_dataset,
)
from semantic_evaluation.core.experimental_schemas import DatasetSpec


def _image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (100, 120, 140)).save(path)


def test_siglip_rooms_images_are_independent_nodes(tmp_path):
    _image(tmp_path / "kitchen" / "a.png")
    _image(tmp_path / "kitchen" / "b.png")
    spec = DatasetSpec.from_mapping({
        "dataset_id": "siglip_rooms", "adapter": "siglip_rooms",
        "root": str(tmp_path), "queries_file": "queries.yaml",
        "split": {"name": "validation"},
    })
    bundle = load_dataset(spec, tmp_path)
    assert len(bundle.nodes) == 2
    assert all(len(node.observations) == 1 for node in bundle.nodes)
    assert bundle.metadata["node_semantics"] == "independent_image"


def test_sunrgbd_class_mapping_records_unmapped_classes(tmp_path):
    _image(tmp_path / "image.png")
    (tmp_path / "sunrgbd_index.jsonl").write_text(json.dumps({
        "sample_id": "sun_1", "image_path": "image.png", "room_label": "office",
        "objects": [
            {"label": "chair", "bbox": [0, 0, 4, 4]},
            {"label": "unknown fixture", "bbox": [1, 1, 3, 3]},
        ],
    }) + "\n", encoding="utf-8")
    spec = DatasetSpec.from_mapping({
        "dataset_id": "sunrgbd", "adapter": "sunrgbd", "root": str(tmp_path),
        "queries_file": "queries.yaml", "split": {"name": "validation"},
        "annotations": {"class_mapping": {"chair": "chair"}},
    })
    bundle = load_dataset(spec, tmp_path)
    assert len(bundle.nodes) == 1
    assert bundle.metadata["unmapped_ground_truth_classes"] == ["unknown fixture"]


def test_matterport_groups_only_same_viewpoint_and_builds_topology(tmp_path):
    root = tmp_path / "matterport"
    r2r = tmp_path / "r2r"
    (r2r / "connectivity").mkdir(parents=True)
    views = []
    for viewpoint in ("v1", "v2"):
        for angle in (0, 90, 180, 270):
            image = root / "images" / f"{viewpoint}_{angle}.png"
            _image(image)
            views.append({"scan_id": "scan1", "viewpoint_id": viewpoint,
                          "angle_degrees": angle,
                          "image_path": str(image.relative_to(root))})
    root.mkdir(exist_ok=True)
    (root / "views_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in views), encoding="utf-8"
    )
    connectivity = [
        {"image_id": "v1", "included": True, "unobstructed": [False, True]},
        {"image_id": "v2", "included": True, "unobstructed": [True, False]},
    ]
    (r2r / "connectivity" / "scan1_connectivity.json").write_text(
        json.dumps(connectivity), encoding="utf-8"
    )
    spec = DatasetSpec.from_mapping({
        "dataset_id": "matterport3d", "adapter": "matterport3d_r2r",
        "root": str(root), "r2r_root": str(r2r), "queries_file": "queries.yaml",
        "split": {"name": "validation", "scan_ids": ["scan1"],
                  "view_angles_degrees": [0, 90, 180, 270]},
    })
    bundle = load_dataset(spec, tmp_path)
    assert {node.node_id for node in bundle.nodes} == {"scan1:v1", "scan1:v2"}
    assert all(len(node.observations) == 4 for node in bundle.nodes)
    assert bundle.topology_edges == [("scan1:v1", "scan1:v2")]
    assert "scan1:v1" in matterport_annotation_template(bundle)


def test_missing_dataset_is_controlled_skip(tmp_path):
    spec = DatasetSpec.from_mapping({
        "dataset_id": "sunrgbd", "adapter": "sunrgbd",
        "root": str(tmp_path / "missing"), "queries_file": "queries.yaml",
    })
    bundle = load_dataset(spec, tmp_path)
    assert bundle.skipped
    assert validate_dataset(bundle)[0]["severity"] == "skip"

