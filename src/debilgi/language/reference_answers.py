"""Bind a reference definition to a requested topic, not incidental word overlap."""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from debilgi.language.content import content_lemmas
from debilgi.language.recovery import lexical_lookup_target

if TYPE_CHECKING:
    from debilgi.language.productive_nlg import ClauseThought

_TOPIC_REQUEST = re.compile(
    r"\b(?:расскажи\s+(?:о|об|про)|кто\s+(?:такой|такая|такие)|"
    r"объясни(?:\s+значение)?(?:\s+слова?)?|"
    r"(?:изучи|узнай|найди|поищи)(?:\s+значение\s+слова)?)\s+"
    r"[«\"']?(?P<topic>[^?!.;,\n]+)",
    re.IGNORECASE,
)
_QUESTION_WORDS = frozenset({"как", "почему", "зачем", "где", "когда", "сколько"})
_DEICTIC = frozenset({"это", "этот", "то", "он", "она", "оно", "они", "ты", "я"})


def definition_requested(prompt: str, title: str) -> bool:
    """A search for a whole question does not request definitions of its words."""
    target = lexical_lookup_target(prompt)
    if target is None:
        match = _TOPIC_REQUEST.search(prompt)
        if match is None:
            return False
        target = match.group("topic").strip(" «»\"'").split(" и ", 1)[0]
        if set(target.casefold().split()) & _QUESTION_WORDS:
            return False
    if target.casefold() in _DEICTIC:
        return False
    requested = set(content_lemmas(target))
    subject = set(content_lemmas(title))
    return bool(subject and subject <= requested)


def same_reference_text(left: str, right: str) -> bool:
    def normalize(value: str) -> str:
        value = unicodedata.normalize("NFC", value).replace("\u0301", "")
        return " ".join(value.casefold().replace("ё", "е").split()).rstrip(".!…")

    return normalize(left) == normalize(right)


def unrequested_definition(
    text: str, prompt: str, definitions: tuple[tuple[str, str], ...]
) -> bool:
    return any(
        same_reference_text(text, definition) and not definition_requested(prompt, title)
        for title, definition in definitions
    )


def definition_head(text: str) -> str | None:
    """Recover a lost dictionary head from old, untyped memory records."""
    match = re.match(r"^\s*([\w\u0301-]+)\s+[—–]\s+", text)
    return match.group(1).replace("\u0301", "") if match else None


def definition_action_thought(
    prompt: str, title: str, definition: str,
) -> ClauseThought | None:
    """Compile an infinitive definition into lexical roles, never a reply string.

    Only a single action and its nominal object are supported. Clauses, negation
    and prepositional restrictions need a richer action frame and are abstained
    from here instead of silently dropping their meaning.
    """
    from debilgi.language.morphology import DEFAULT_MORPHOLOGY
    from debilgi.language.productive_nlg import (
        ClauseThought,
        ConceptArgument,
        ConceptLexeme,
        SemanticDependency,
    )
    from debilgi.language.valency import assess_predicate_valency

    if not definition_requested(prompt, title):
        return None
    body = definition.replace("\u0301", "").strip()
    head = definition_head(body)
    if head:
        if not same_reference_text(head, title):
            return None
        body = re.split(r"\s+[—–]\s+", body, maxsplit=1)[1]
    sentence = DEFAULT_MORPHOLOGY.analyze_sentence(body.rstrip(".!"))
    words = tuple(zip(sentence.tokens, sentence.readings, strict=True))
    if len(words) < 2 or any(reading is None for token, reading in words if token.isalnum()):
        return None
    action = words[0][1]
    if action is None or "INFN" not in action.grammemes:
        return None
    valency = assess_predicate_valency(
        action.lemma, present_roles=frozenset({"subject", "direct_object"}),
    )
    if valency.missing_roles or valency.unexpected_roles:
        return None
    objects = words[1:]
    if any(
        (reading is None and token != ",")
        or (reading is not None and reading.part_of_speech not in {"noun", "adjective", "pronoun"})
        or (reading is not None and reading.lemma in {"который", "каковой", "чей"})
        or (reading is not None and "PRTS" in reading.grammemes)
        for token, reading in objects
    ):
        return None
    object_index = next(
        (i for i, (_, r) in enumerate(objects) if r is not None and r.part_of_speech == "noun"),
        None,
    )
    if object_index is None:
        return None
    if any(r is None or r.part_of_speech != "adjective" for _, r in objects[:object_index]):
        return None

    def lexeme(token, reading):
        return ConceptLexeme(
            reading.lemma if reading else token, token,
            reading.part_of_speech if reading else None,
            tuple(reading.features().items()) if reading else (),
        )

    obj = ConceptArgument(
        lexeme(*objects[object_index]),
        modifiers=tuple(lexeme(*item) for item in objects[:object_index]),
        postmodifiers=tuple(lexeme(*item) for item in objects[object_index + 1:]),
    )
    return ClauseThought(
        predicate=ConceptLexeme("означать", "означает", "verb"),
        dependencies=(
            SemanticDependency("subject", ConceptArgument(ConceptLexeme(
                title, f"«{title}»", "noun", entity_kind="quoted_lexeme",
            ))),
            SemanticDependency("action_complement", ConceptArgument(lexeme(*words[0]))),
            SemanticDependency("direct_object", obj),
        ),
        predicate_class="modal", confidence=0.78,
        evidence=(f"definition_action:{title}:{action.lemma}", definition),
    )
