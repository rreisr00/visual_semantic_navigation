#!/usr/bin/env python3
"""Generate the SUN RGB-D query suite from the normalized index.

The natural-language text of every query is authored here; the ground truth is
*derived* from ``sunrgbd_index.jsonl`` so no ``sample_id`` is ever invented.
Room queries carry ``expected_room`` and let ``resolve_valid_nodes`` expand the
whole scene category; object queries pin an explicit ``valid_node_ids`` list
built from the human 2D annotations — never from detector output, which would
make the retrieval evaluation circular.

Selectors work on the annotated labels of each image: ``scenes`` restricts the
scene category and every group in ``all_of`` must contribute at least one label,
so synonym pairs such as sofa/couch stay a single semantic requirement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

DATASET_ID = "sunrgbd"

# ── Room queries: ground truth is the whole scene category ─────────────────── #

ROOM_QUERIES: list[dict[str, Any]] = [
    {"key": "bedroom", "room": "bedroom",
     "es": "ve al dormitorio", "en": "go to the bedroom"},
    {"key": "bathroom", "room": "bathroom",
     "es": "ve al cuarto de baño", "en": "go to the bathroom"},
    {"key": "kitchen", "room": "kitchen",
     "es": "ve a la cocina", "en": "go to the kitchen"},
    {"key": "office", "room": "office",
     "es": "ve a la oficina", "en": "go to the office"},
    {"key": "classroom", "room": "classroom",
     "es": "ve al aula", "en": "go to the classroom"},
    {"key": "library", "room": "library",
     "es": "ve a la biblioteca", "en": "go to the library"},
    {"key": "corridor", "room": "corridor",
     "es": "ve al pasillo", "en": "go to the corridor"},
    {"key": "conference_room", "room": "conference_room",
     "es": "ve a la sala de reuniones", "en": "go to the meeting room"},
    {"key": "living_room", "room": "living_room",
     "es": "ve al salón", "en": "go to the living room"},
    {"key": "dining_room", "room": "dining_room",
     "es": "ve al comedor", "en": "go to the dining room"},
]

# Paraphrases exercise language robustness against the same ground truth.
ROOM_PARAPHRASES: list[dict[str, Any]] = [
    {"key": "kitchen_paraphrase", "room": "kitchen",
     "es": "el lugar donde se cocinan y se preparan los alimentos",
     "en": "the place where meals are cooked and prepared",
     "paraphrase_of": "kitchen"},
    {"key": "bathroom_paraphrase", "room": "bathroom",
     "es": "la estancia para lavarse y ducharse",
     "en": "the room used to wash up and take a shower",
     "paraphrase_of": "bathroom"},
    {"key": "library_paraphrase", "room": "library",
     "es": "la sala silenciosa para leer y estudiar entre estanterías",
     "en": "the quiet room for reading and studying among bookshelves",
     "paraphrase_of": "library"},
]

# ── Object queries: ground truth derived from the 2D annotations ───────────── #

OBJECT_QUERIES: list[dict[str, Any]] = [
    {"key": "bathroom_sink_toilet", "query_type": "multi_object",
     "scenes": ["bathroom"], "all_of": [["sink"], ["toilet"]],
     "expected_room": "bathroom", "expected_objects": ["sink", "toilet"],
     "es": "el baño con lavabo e inodoro",
     "en": "the bathroom with a sink and a toilet"},
    {"key": "bathroom_bathtub", "query_type": "object",
     "scenes": ["bathroom"], "all_of": [["bathtub"]],
     "expected_room": "bathroom", "expected_objects": [],
     "es": "el baño que tiene bañera",
     "en": "the bathroom that has a bathtub"},
    {"key": "bedroom_bed_lamp", "query_type": "multi_object",
     "scenes": ["bedroom"], "all_of": [["bed"], ["lamp"]],
     "expected_room": "bedroom", "expected_objects": ["bed"],
     "es": "el dormitorio con una cama y una lámpara",
     "en": "the bedroom with a bed and a lamp"},
    {"key": "kitchen_sink", "query_type": "object",
     "scenes": ["kitchen", "office_kitchen"], "all_of": [["sink"]],
     "expected_room": None, "expected_objects": ["sink"],
     "es": "la cocina con fregadero",
     "en": "the kitchen with a sink"},
    {"key": "kitchen_microwave", "query_type": "object",
     "scenes": ["kitchen", "office_kitchen"], "all_of": [["microwave"]],
     "expected_room": None, "expected_objects": ["microwave"],
     "es": "la cocina donde hay un microondas",
     "en": "the kitchen where there is a microwave"},
    {"key": "living_room_sofa", "query_type": "object",
     "scenes": ["living_room"], "all_of": [["sofa", "couch"]],
     "expected_room": "living_room", "expected_objects": ["couch"],
     "es": "el salón con un sofá",
     "en": "the living room with a sofa"},
    {"key": "living_room_sofa_tv", "query_type": "functional",
     "scenes": ["living_room"], "all_of": [["sofa", "couch"], ["tv"]],
     "expected_room": "living_room", "expected_objects": ["couch", "tv"],
     "es": "el salón donde puedo ver la televisión sentado en el sofá",
     "en": "the living room where I can watch TV sitting on the sofa"},
    {"key": "workstation_monitor_keyboard", "query_type": "multi_object",
     "scenes": ["office", "home_office", "computer_room"],
     "all_of": [["monitor"], ["keyboard"]],
     "expected_room": None, "expected_objects": ["tv", "keyboard"],
     "es": "el puesto de trabajo con monitor y teclado",
     "en": "the workstation with a monitor and a keyboard"},
    {"key": "library_books_shelves", "query_type": "multi_object",
     "scenes": ["library", "bookstore"],
     "all_of": [["books", "book"], ["shelves", "shelf", "bookshelf"]],
     "expected_room": None, "expected_objects": ["book"],
     "es": "las estanterías llenas de libros",
     "en": "the shelves full of books"},
    {"key": "classroom_board", "query_type": "object",
     "scenes": ["classroom", "lecture_theatre"],
     "all_of": [["whiteboard", "blackboard", "board"]],
     "expected_room": None, "expected_objects": [],
     "es": "el aula con una pizarra",
     "en": "the classroom with a board"},
    {"key": "corridor_door", "query_type": "object",
     "scenes": ["corridor"], "all_of": [["door"]],
     "expected_room": "corridor", "expected_objects": [],
     "es": "el pasillo con una puerta al fondo",
     "en": "the corridor with a door at the end"},
    {"key": "printer", "query_type": "functional",
     "scenes": None, "all_of": [["printer"]],
     "expected_room": None, "expected_objects": [],
     "es": "el sitio donde puedo imprimir un documento",
     "en": "the place where I can print a document"},
]

# ── Negative queries: concepts absent from every scene category ────────────── #

NEGATIVE_QUERIES: list[dict[str, Any]] = [
    {"key": "swimming_pool", "es": "ve a la piscina",
     "en": "go to the swimming pool", "difficulty": "easy"},
    {"key": "garage", "es": "ve al garaje donde están aparcados los coches",
     "en": "go to the garage where the cars are parked", "difficulty": "easy"},
    {"key": "billiard_table", "es": "ve a la mesa de billar",
     "en": "go to the billiard table", "difficulty": "easy"},
    {"key": "escalator", "es": "sube por la escalera mecánica",
     "en": "go up the escalator", "difficulty": "hard"},
]


def load_index(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                item["labels"] = {
                    str(entry["label"]).strip().lower() for entry in item.get("objects", [])
                }
                rows.append(item)
    return rows


def select(rows: Sequence[dict[str, Any]], spec: dict[str, Any]) -> list[str]:
    scenes = spec.get("scenes")
    allowed = set(scenes) if scenes else None
    groups = [set(group) for group in spec.get("all_of", [])]
    return [
        row["sample_id"] for row in rows
        if (allowed is None or row["room_label"] in allowed)
        and all(group & row["labels"] for group in groups)
    ]


def _entry(
    query_id: str, text: str, language: str, query_type: str,
    valid: str, expected_room: str | None, expected_objects: Sequence[str],
    is_negative: bool, metadata: str,
) -> str:
    room = expected_room if expected_room else "null"
    objects = "[" + ", ".join(expected_objects) + "]"
    return "\n".join([
        f"  - query_id: {query_id}",
        f"    text: {text}",
        f"    language: {language}",
        f"    query_type: {query_type}",
        f"    dataset_id: {DATASET_ID}",
        "    scene_id: null",
        f"    valid_node_ids: {valid}",
        f"    expected_room: {room}",
        f"    expected_objects: {objects}",
        "    expected_relations: []",
        f"    is_negative: {'true' if is_negative else 'false'}",
        f"    metadata: {metadata}",
    ])


def _anchored_list(anchor: str, node_ids: Sequence[str]) -> str:
    lines = [f"&{anchor}"] + [f"      - {node_id}" for node_id in node_ids]
    return "\n".join(lines)


def build(rows: Sequence[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    blocks: list[str] = []
    report: list[dict[str, Any]] = []

    for spec in ROOM_QUERIES + ROOM_PARAPHRASES:
        matched = [row["sample_id"] for row in rows if row["room_label"] == spec["room"]]
        if not matched:
            raise SystemExit(f"scene category without images: {spec['room']}")
        origin = spec.get("paraphrase_of")
        for language in ("es", "en"):
            metadata = (f"{{paraphrase_of: {DATASET_ID}_{origin}_{language}, "
                        f"ground_truth: expected_room}}") if origin else (
                        f"{{pair_id: {spec['key']}_direct, ground_truth: expected_room}}")
            blocks.append(_entry(
                f"{DATASET_ID}_{spec['key']}_{language}", spec[language], language,
                "room", "[]", spec["room"], [], False, metadata))
        report.append({"key": spec["key"], "query_type": "room",
                       "n_valid": len(matched), "source": "expected_room"})

    for spec in OBJECT_QUERIES:
        matched = select(rows, spec)
        if not matched:
            raise SystemExit(f"selector without matches: {spec['key']}")
        anchor = f"gt_{spec['key']}"
        for index, language in enumerate(("es", "en")):
            valid = (_anchored_list(anchor, matched) if index == 0 else f"*{anchor}")
            blocks.append(_entry(
                f"{DATASET_ID}_{spec['key']}_{language}", spec[language], language,
                spec["query_type"], valid, spec["expected_room"],
                spec["expected_objects"], False,
                f"{{pair_id: {spec['key']}, ground_truth: annotation2Dfinal}}"))
        report.append({"key": spec["key"], "query_type": spec["query_type"],
                       "n_valid": len(matched), "source": "annotation2Dfinal"})

    for spec in NEGATIVE_QUERIES:
        for language in ("es", "en"):
            blocks.append(_entry(
                f"{DATASET_ID}_{spec['key']}_negative_{language}", spec[language],
                language, "negative", "[]", None, [], True,
                f"{{pair_id: {spec['key']}_negative, difficulty: {spec['difficulty']}}}"))
        report.append({"key": spec["key"], "query_type": "negative",
                       "n_valid": 0, "source": "absent_concept"})

    header = "\n".join([
        "# SUN RGB-D query suite generated by tools/build_sunrgbd_queries.py.",
        "# Room queries resolve their ground truth from expected_room (the scene",
        "# category of scene.txt); object queries pin valid_node_ids derived from the",
        "# human 2D annotations, shared between each es/en pair via a YAML anchor.",
        "# Regenerate after rebuilding sunrgbd_index.jsonl; do not edit ids by hand.",
        "queries:",
    ])
    return header + "\n" + "\n".join(blocks) + "\n", report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path,
                        help="path to sunrgbd_index.jsonl")
    parser.add_argument("--output", required=True, type=Path,
                        help="destination queries YAML")
    args = parser.parse_args()

    rows = load_index(args.index.expanduser().resolve())
    document, report = build(rows)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")

    total = len(rows)
    print(f"{output}: {sum(2 for _ in report)} consultas sobre {total} imágenes")
    for item in report:
        share = f"{item['n_valid'] / total * 100:5.1f}%" if item["n_valid"] else "    —"
        print(f"  {item['query_type']:12s} {item['key']:30s} "
              f"válidos={item['n_valid']:4d} ({share})  {item['source']}")


if __name__ == "__main__":
    main()
