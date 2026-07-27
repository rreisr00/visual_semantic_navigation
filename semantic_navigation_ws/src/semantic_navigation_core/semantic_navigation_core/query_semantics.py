"""Deterministic semantic hints extracted from English or Spanish queries."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from semantic_navigation_core.types import SpatialRelation


_RELATION_PHRASES: tuple[tuple[str, str], ...] = (
    ("a la izquierda de", "LEFT_OF"),
    ("to the left of", "LEFT_OF"),
    ("left of", "LEFT_OF"),
    ("a la derecha de", "RIGHT_OF"),
    ("to the right of", "RIGHT_OF"),
    ("right of", "RIGHT_OF"),
    ("encima de", "ABOVE"),
    ("above", "ABOVE"),
    ("debajo de", "BELOW"),
    ("below", "BELOW"),
    ("delante de", "IN_FRONT_OF"),
    ("in front of", "IN_FRONT_OF"),
    ("detras de", "BEHIND"),
    ("behind", "BEHIND"),
    ("cerca de", "NEAR"),
    ("al lado de", "NEAR"),
    ("junto a", "NEAR"),
    ("next to", "NEAR"),
    ("near", "NEAR"),
    ("dentro de", "INSIDE"),
    ("inside", "INSIDE"),
    ("sobre", "POSSIBLY_ON_TOP_OF"),
    ("on top of", "POSSIBLY_ON_TOP_OF"),
)

_OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "cup": ("taza",),
    "chair": ("silla",),
    "dining table": ("mesa de comedor", "mesa"),
    "couch": ("sofa", "sillon"),
    "potted plant": ("planta",),
    "tv": ("television", "televisor"),
    "bed": ("cama",),
    "book": ("libro",),
}

_ROOM_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("kitchen", "cocina"),
    ("bedroom", "dormitorio"),
    ("living room", "sala estar", "salon"),
    ("meeting room", "sala reunion"),
    ("print area", "zona impresion"),
    ("open lab", "laboratorio abierto"),
)


@dataclass(frozen=True)
class QuerySemantics:
    objects: list[str] = field(default_factory=list)
    relations: list[SpatialRelation] = field(default_factory=list)
    room: str | None = None


def normalize_query_text(value: str) -> str:
    """Return lowercase, accent-free tokens with stable whitespace."""
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    normalized = " ".join(
        re.sub(r"[^a-zA-Z0-9]+", " ", ascii_text.lower()).split()
    )
    return re.sub(r"\bdel\b", "de", normalized)


def extract_query_semantics(
    text: str,
    object_vocabulary: Iterable[str],
    room_vocabulary: Iterable[str] = (),
) -> QuerySemantics:
    """Extract only labels already present in the scene graph.

    Restricting extraction to the graph vocabulary prevents arbitrary nouns
    from being treated as detector categories. Relations are emitted only when
    a known object occurs on both sides of an explicit relation phrase.
    """
    normalized = normalize_query_text(text)
    padded = f" {normalized} "
    mention_candidates: list[tuple[int, int, str, bool]] = []
    for label in dict.fromkeys(object_vocabulary):
        candidate = normalize_query_text(label)
        if not candidate:
            continue
        candidates = (candidate, *_OBJECT_ALIASES.get(candidate, ()))
        for alias_index, alias in enumerate(candidates):
            match = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized)
            if match is not None:
                mention_candidates.append(
                    (match.start(), match.end(), label, alias_index == 0)
                )
                break

    # A graph can contain both a detector label and a localized synonym (for
    # example ``cup`` and ``taza``).  When both resolve to the same text span,
    # retain the exact vocabulary match instead of emitting the concept twice.
    mentions: list[tuple[int, int, str]] = []
    by_span: dict[tuple[int, int], tuple[int, int, str, bool]] = {}
    for mention in mention_candidates:
        span = mention[:2]
        previous = by_span.get(span)
        if previous is None or (mention[3] and not previous[3]):
            by_span[span] = mention
    mentions = [(start, end, label) for start, end, label, _ in by_span.values()]
    mentions.sort(key=lambda item: item[0])

    relations: list[SpatialRelation] = []
    for phrase, predicate in _RELATION_PHRASES:
        phrase_normalized = normalize_query_text(phrase)
        match = re.search(
            rf"(?<!\w){re.escape(phrase_normalized)}(?!\w)", normalized
        )
        if match is None:
            continue
        before = [item for item in mentions if item[1] <= match.start()]
        after = [item for item in mentions if item[0] >= match.end()]
        if before and after:
            relations.append(
                SpatialRelation(
                    subject=before[-1][2],
                    predicate=predicate,
                    obj=after[0][2],
                    relation_type="query_expectation",
                    reference_frame="language",
                )
            )

    room = None
    room_matches: list[tuple[int, str]] = []
    for room_id in dict.fromkeys(room_vocabulary):
        candidate = normalize_query_text(room_id)
        aliases = {candidate}
        for group in _ROOM_ALIAS_GROUPS:
            if candidate in group:
                aliases.update(group)
        if candidate and any(f" {alias} " in padded for alias in aliases):
            room_matches.append((len(candidate), room_id))
    if room_matches:
        room = max(room_matches, key=lambda item: item[0])[1]

    return QuerySemantics(
        objects=[item[2] for item in mentions],
        relations=relations,
        room=room,
    )
