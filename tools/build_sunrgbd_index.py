#!/usr/bin/env python3
"""Build the normalized ``sunrgbd_index.jsonl`` expected by the SUN RGB-D adapter.

The official release ships annotations as MATLAB metadata plus per-sample JSON
polygons, so this script derives one deterministic JSONL row per image with the
fields ``sample_id``, ``image_path``, ``room_label`` and ``objects[{label,bbox}]``.

Sample identifiers keep the official ``sequenceName`` (without the leading
``SUNRGBD/`` component) and boxes come from ``annotation2Dfinal/index.json``,
which annotates every 2D instance in the space of ``image/``.  Polygons drawn
past the frame are clamped to the image, matching what a detector can predict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import scipy.io as sio
from PIL import Image

SPLIT_SOURCES = {
    "train": ("trainvalsplit", "train"),
    "validation": ("trainvalsplit", "val"),
    "test": ("alltest", None),
    "trainval": ("alltrain", None),
}
SEQUENCE_ANCHOR = "SUNRGBD/"
MIN_BOX_SIDE = 2.0


def sequence_names(root: Path, split: str) -> list[str]:
    """Return official ``sequenceName`` values for a split, in file order."""
    if split == "all":
        meta = sio.loadmat(
            root / "SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat",
            squeeze_me=True, struct_as_record=False,
        )["SUNRGBDMeta"]
        return [str(row.sequenceName).rstrip("/") for row in meta]
    variable, field = SPLIT_SOURCES[split]
    data = sio.loadmat(
        root / "SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat",
        squeeze_me=True, struct_as_record=False,
    )[variable]
    paths = getattr(data, field) if field else data
    names = []
    for value in paths:
        text = str(value).rstrip("/")
        index = text.find(SEQUENCE_ANCHOR)
        if index < 0:
            raise ValueError(f"unexpected split entry without {SEQUENCE_ANCHOR!r}: {text}")
        names.append(text[index:])
    return names


def _image_file(sample_dir: Path) -> Path | None:
    candidates = sorted(
        path for path in (sample_dir / "image").glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return candidates[0] if candidates else None


def _polygon_objects(sample_dir: Path, width: int, height: int) -> list[dict[str, Any]]:
    annotation = sample_dir / "annotation2Dfinal" / "index.json"
    if not annotation.is_file():
        return []
    try:
        data = json.loads(annotation.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []
    names = data.get("objects") or []
    frames = data.get("frames") or []
    objects: list[dict[str, Any]] = []
    for polygon in (frames[0].get("polygon") or [] if frames else []):
        xs, ys = polygon.get("x"), polygon.get("y")
        xs = xs if isinstance(xs, list) else ([xs] if isinstance(xs, (int, float)) else [])
        ys = ys if isinstance(ys, list) else ([ys] if isinstance(ys, (int, float)) else [])
        if not xs or not ys:
            continue
        index = polygon.get("object")
        if not isinstance(index, int) or not 0 <= index < len(names):
            continue
        entry = names[index]
        label = str((entry or {}).get("name") or "").strip() if isinstance(entry, dict) else ""
        if not label:
            continue
        x1 = max(0.0, min(float(min(xs)), width))
        y1 = max(0.0, min(float(min(ys)), height))
        x2 = max(0.0, min(float(max(xs)), width))
        y2 = max(0.0, min(float(max(ys)), height))
        if x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE:
            continue
        objects.append({"label": label, "bbox": [x1, y1, x2, y2]})
    return objects


def build_rows(root: Path, split: str) -> Iterator[dict[str, Any]]:
    for sequence in sequence_names(root, split):
        sample_dir = root / sequence
        if not sample_dir.is_dir():
            print(f"omitido (directorio ausente): {sequence}")
            continue
        image = _image_file(sample_dir)
        if image is None:
            print(f"omitido (sin imagen RGB): {sequence}")
            continue
        with Image.open(image) as handle:
            width, height = handle.size
        scene = sample_dir / "scene.txt"
        room_label = scene.read_text(encoding="utf-8").strip() if scene.is_file() else ""
        yield {
            "sample_id": sequence[len(SEQUENCE_ANCHOR):],
            "image_path": str(image.relative_to(root)),
            "room_label": room_label or None,
            "width": width,
            "height": height,
            "objects": _polygon_objects(sample_dir, width, height),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="SUNRGBD_ROOT containing SUNRGBD/ and SUNRGBDtoolbox/")
    parser.add_argument("--split", default="validation",
                        choices=[*SPLIT_SOURCES, "all"],
                        help="official split to export (default: validation)")
    parser.add_argument("--output", type=Path, default=None,
                        help="destination JSONL (default: <root>/sunrgbd_index.jsonl)")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = (args.output or root / "sunrgbd_index.jsonl").expanduser().resolve()
    rows = list(build_rows(root, args.split))
    if not rows:
        raise SystemExit(f"no se generó ninguna fila para el split {args.split!r} en {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    labelled = sum(1 for row in rows if row["objects"])
    boxes = sum(len(row["objects"]) for row in rows)
    rooms = len({row["room_label"] for row in rows if row["room_label"]})
    print(f"{output}: {len(rows)} imágenes, {labelled} con cajas 2D, "
          f"{boxes} instancias, {rooms} categorías de escena")


if __name__ == "__main__":
    main()
