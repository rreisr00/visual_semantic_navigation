"""Unit tests for the pure intent parser (no ROS required)."""

import pytest

from semantic_voice.core.intent_parser import (
    MoveToPosition,
    NoIntent,
    SaveWaypoint,
    SemanticGoal,
    parse,
)


class TestMoveToPosition:
    @pytest.mark.parametrize(
        ("text", "x", "y"),
        [
            ("move to 2.5 3.0", 2.5, 3.0),
            ("go to position 1, -2.5", 1.0, -2.5),
            ("navigate to point 0 0", 0.0, 0.0),
            ("drive to coordinates -1.5, -3", -1.5, -3.0),
            ("please move to 4 5", 4.0, 5.0),
            ("Move to 2.5, 3.0.", 2.5, 3.0),  # Whisper capitals + period
            ("go to the position 7 8", 7.0, 8.0),
            ("ve a la posición 1.5, -2", 1.5, -2.0),
            ("navega al punto 0 3", 0.0, 3.0),
        ],
    )
    def test_variants(self, text, x, y):
        intent = parse(text)
        assert intent == MoveToPosition(x=x, y=y)

    def test_single_numeral_is_not_a_move(self):
        # "room 2" has one numeral -> semantic description, not coordinates
        assert isinstance(parse("go to room 2"), SemanticGoal)


class TestSaveWaypoint:
    @pytest.mark.parametrize(
        ("text", "label"),
        [
            ("save waypoint kitchen", "kitchen"),
            ("capture waypoint as kitchen_01", "kitchen_01"),
            ("mark this location as living room", "living_room"),
            ("store the current position as sofa corner", "sofa_corner"),
            ("save this place called bedroom", "bedroom"),
            ("save as kitchen", "kitchen"),  # noun omitted, "as" gates it
            ("Save waypoint Kitchen.", "kitchen"),  # Whisper artifacts
            ("guarda este punto como cocina", "cocina"),
            ("marca la ubicación llamada sala norte", "sala_norte"),
        ],
    )
    def test_variants(self, text, label):
        intent = parse(text)
        assert intent == SaveWaypoint(label=label)

    def test_empty_label_allowed(self):
        # kg_manager auto-names when the label is empty
        assert parse("save waypoint") == SaveWaypoint(label="")

    def test_save_without_noun_or_as_is_semantic(self):
        assert isinstance(parse("save the whales"), SemanticGoal)


class TestSemanticGoal:
    @pytest.mark.parametrize(
        ("text", "query"),
        [
            ("go to the sofa", "sofa"),
            ("take me to the kitchen table", "kitchen table"),
            ("find the red chair", "red chair"),
            ("the red chair near the window", "red chair near the window"),
            ("Go to the sofa.", "sofa"),
            ("please robot go to the fridge", "fridge"),
            ("where is the television", "television"),
            ("navigate to the bathroom sink", "bathroom sink"),
            ("ve al sofá del salón", "sofá del salón"),
            ("por favor busca la taza roja", "taza roja"),
            ("llévame a la mesa junto a la planta", "mesa junto a la planta"),
            ("dónde está la impresora", "impresora"),
        ],
    )
    def test_variants(self, text, query):
        intent = parse(text)
        assert intent == SemanticGoal(query=query)

    def test_bare_description(self):
        assert parse("bookshelf in the corner") == SemanticGoal(
            query="bookshelf in the corner"
        )


class TestNoIntent:
    @pytest.mark.parametrize("text", ["", " ", ".", "a", "go to"])
    def test_rejects(self, text):
        assert isinstance(parse(text), NoIntent)


class TestPrecedence:
    def test_two_numerals_is_move_not_semantic(self):
        assert parse("go to 2 3") == MoveToPosition(x=2.0, y=3.0)

    def test_save_beats_semantic(self):
        assert parse("save waypoint sofa") == SaveWaypoint(label="sofa")
