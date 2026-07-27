from semantic_navigation_core.query_semantics import (
    extract_query_semantics,
    normalize_query_text,
)


def test_normalization_handles_accents_and_detector_separators():
    assert normalize_query_text("Taza_a-la IZQUIÉRDA") == "taza a la izquierda"


def test_extracts_spanish_objects_relation_and_room_from_known_vocabulary():
    parsed = extract_query_semantics(
        "Ve a la taza a la izquierda del monitor en sala reunión",
        ["cup", "taza", "monitor"],
        ["sala_reunión", "pasillo"],
    )
    assert parsed.objects == ["taza", "monitor"]
    assert parsed.room == "sala_reunión"
    assert len(parsed.relations) == 1
    assert parsed.relations[0].subject == "taza"
    assert parsed.relations[0].predicate == "LEFT_OF"
    assert parsed.relations[0].obj == "monitor"


def test_does_not_invent_labels_or_relation_without_two_known_objects():
    parsed = extract_query_semantics(
        "Find the red mug left of the unknown appliance",
        ["mug", "chair"],
        ["kitchen"],
    )
    assert parsed.objects == ["mug"]
    assert parsed.relations == []
    assert parsed.room is None


def test_longest_room_match_wins():
    parsed = extract_query_semantics(
        "go to the meeting room",
        [],
        ["room", "meeting_room"],
    )
    assert parsed.room == "meeting_room"


def test_spanish_aliases_map_to_detector_and_room_vocabulary():
    parsed = extract_query_semantics(
        "la taza junto a la silla en la cocina",
        ["cup", "chair"],
        ["kitchen"],
    )
    assert parsed.objects == ["cup", "chair"]
    assert parsed.room == "kitchen"
