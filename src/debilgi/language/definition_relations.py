"""Conservative relations from explicit definitions, not encyclopedia hyperlinks."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from debilgi.language.morphology import DEFAULT_MORPHOLOGY

if TYPE_CHECKING:
    from debilgi.language.productive_nlg import ClauseThought


@dataclass(frozen=True, slots=True)
class DefinitionRelation:
    predicate: str
    target: str
    evidence: str
    sense_key: str


def _plain(text: str) -> str:
    # Strip pronunciation stress, retaining ё as a Russian letter.
    return " ".join(text.replace("\u0301", "").casefold().split())


def _phrase(text: str, *, action: bool = False) -> str | None:
    sentence = DEFAULT_MORPHOLOGY.analyze_sentence(text.strip(" ."))
    words = [
        (token, reading)
        for token, reading in zip(sentence.tokens, sentence.readings, strict=True)
        if token[:1].isalnum()
    ]
    if not words or len(words) > 5 or any(reading is None for _, reading in words):
        return None
    parts = [reading.part_of_speech for _, reading in words]
    if action and len(words) == 1 and parts[0] == "verb":
        return words[0][1].lemma
    # Preserve the complete simple noun phrase. Never drop a restrictive clause,
    # location, negation or unknown modifier to obtain a broader claim.
    if parts[-1] != "noun" or any(part != "adjective" for part in parts[:-1]):
        return None
    return " ".join(reading.lemma for _, reading in words)


@lru_cache(maxsize=2048)
def definition_relations(
    subject: str, definition: str, *, require_subject: bool = False
) -> tuple[DefinitionRelation, ...]:
    text = _plain(unicodedata.normalize("NFC", definition))
    subject_key = _plain(subject)
    if not text or not subject_key:
        return ()
    # Only the leading definition is scoped to the headword. Later sentences may
    # introduce different actors or senses and require a stronger discourse parser.
    first = re.split(r"[.!?;]", text, maxsplit=1)[0].strip()
    if re.search(r"\b(?:не|либо|или|возможно|иногда|может)\b", first):
        return ()
    body = first
    match = re.match(r"^(.+?)\s+[—–-]\s+(?:это\s+)?(.+)$", first)
    if match:
        if _plain(match[1]).replace("ё", "е") != subject_key.replace("ё", "е"):
            return ()
        body = match[2]
    elif first.startswith(subject_key + " "):
        body = first[len(subject_key) :].strip()
    elif require_subject:
        return ()
    body = re.sub(r"^это\s+", "", body)
    sense = hashlib.sha256(text.encode()).hexdigest()[:24]
    result = []

    def add(predicate: str, phrase: str, *, action: bool = False):
        target = _phrase(phrase, action=action)
        if target and target != subject_key:
            result.append(DefinitionRelation(predicate, target, definition, sense))

    # Markers describe relations, not response wording.
    patterns = (
        ("part_of", r"^часть\s+(.+)$"),
        ("consists_of", r"^состоит\s+из\s+(.+)$"),
        ("used_for", r"^(?:предназначен[аоы]?|служит|используется)\s+для\s+(.+)$"),
    )
    for predicate, pattern in patterns:
        match = re.match(pattern, body)
        if match:
            add(predicate, match[1], action=predicate == "used_for")
            return tuple(result)
    # A purpose adjunct is kept independently from the class of the object.
    purpose = re.fullmatch(r"(.+?)(?:\s+для|,\s*предназначенн\w*\s+для)\s+(.+)", body)
    if purpose:
        add("is_a", purpose[1])
        add("used_for", purpose[2], action=True)
    else:
        add("is_a", body)
    return tuple(result)


def definition_answer_thought(
    prompt: str,
    meanings: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
) -> ClauseThought | None:
    """Answer an explicit class/purpose question using the selected sense only."""
    from debilgi.language.productive_nlg import (
        ClauseThought,
        ConceptArgument,
        ConceptLexeme,
        SemanticDependency,
    )

    normalized = _plain(prompt)
    if re.search(r"\b(?:не|либо|или)\b", normalized):
        return None
    relation = (
        "used_for"
        if re.search(r"\b(?:зачем|для чего)\b", normalized)
        else "is_a"
        if re.search(r"\bчто (?:такое|значит|означает)\b", normalized)
        else None
    )
    if relation is None:
        return None
    candidates = tuple(
        dict.fromkeys(
            (subject, target)
            for subject, edges in meanings
            for predicate, target in edges
            if predicate == relation
        )
    )
    if len(candidates) != 1:
        return None
    subject, target = candidates[0]
    sentence = DEFAULT_MORPHOLOGY.analyze_sentence(target)
    readings = tuple(item for item in sentence.readings if item is not None)
    if not readings or readings[-1].part_of_speech != "noun":
        return None

    def lexeme(reading):
        return ConceptLexeme(reading.lemma, reading.lemma, reading.part_of_speech)

    argument = ConceptArgument(
        lexeme(readings[-1]),
        modifiers=tuple(lexeme(item) for item in readings[:-1]),
        preposition="для" if relation == "used_for" else None,
        grammatical_case="genitive" if relation == "used_for" else "instrumental",
    )
    return ClauseThought(
        predicate=ConceptLexeme(
            "использоваться" if relation == "used_for" else "являться",
            "используется" if relation == "used_for" else "является",
            "verb",
            (("tense", "present"), ("person", "third"), ("number", "singular")),
        ),
        dependencies=(
            SemanticDependency("subject", ConceptArgument(ConceptLexeme(subject, subject, "noun"))),
            SemanticDependency(
                "topic" if relation == "used_for" else "predicate_complement", argument
            ),
        ),
        confidence=0.86,
        evidence=(f"selected_definition:{subject}:{relation}:{target}",),
    )
