"""Shared service-word policy for topical terms and graph relation endpoints."""

from __future__ import annotations

from debilgi.language.grammar import role

PARTICLES = frozenset(
    {
        "не",
        "ни",
        "бы",
        "же",
        "ли",
        "ну",
        "вот",
        "даже",
        "уже",
        "ещё",
        "еще",
        "только",
        "просто",
        "именно",
        "типа",
        "типо",
    }
)

CONJUNCTIONS = frozenset(
    {
        "и",
        "а",
        "но",
        "чтобы",
        "или",
        "либо",
        "если",
        "хотя",
        "зато",
        "однако",
        "потому",
        "поэтому",
        "когда",
        "пока",
        "будто",
        "словно",
        "ведь",
        "причем",
        "причём",
    }
)

PREPOSITIONS = frozenset(
    {
        "в",
        "во",
        "на",
        "над",
        "под",
        "за",
        "из",
        "от",
        "до",
        "к",
        "ко",
        "у",
        "о",
        "об",
        "обо",
        "про",
        "для",
        "без",
        "при",
        "через",
        "между",
        "среди",
        "после",
        "перед",
        "вокруг",
        "возле",
        "вместо",
        "внутри",
        "вне",
        "с",
        "со",
        "по",
    }
)

QUESTION_WORDS = frozenset(
    {
        "кто",
        "что",
        "какой",
        "какая",
        "какое",
        "какие",
        "где",
        "куда",
        "откуда",
        "когда",
        "зачем",
        "почему",
        "как",
    }
)

STRUCTURAL_MODIFIERS = frozenset({"самый", "такой", "весь"})

SERVICE_WORD_LEMMAS = frozenset(
    {
        *PARTICLES,
        *CONJUNCTIONS,
        *PREPOSITIONS,
        *QUESTION_WORDS,
        *STRUCTURAL_MODIFIERS,
    }
)
SERVICE_PARTS_OF_SPEECH = frozenset(
    {
        "preposition",
        "conjunction",
        "particle",
        "interjection",
        "prep",
        "conj",
        "prcl",
        "intj",
    }
)


def normalize_service_word(value: str) -> str:
    return value.casefold().replace("ё", "е").strip(".,!?;:—-()[]{}«»…")


def service_word_reason(
    lemma: str,
    *,
    part_of_speech: str | None = None,
    question_context: bool = False,
) -> str | None:
    """Explain why a token is structural rather than a relation endpoint."""

    normalized = normalize_service_word(lemma)
    normalized_part = part_of_speech.casefold() if part_of_speech else None
    if normalized_part in SERVICE_PARTS_OF_SPEECH:
        return f"part_of_speech:{normalized_part}"
    if normalized in PARTICLES:
        return "particle"
    if normalized in CONJUNCTIONS:
        return "conjunction"
    if normalized in PREPOSITIONS:
        return "preposition"
    if normalized in QUESTION_WORDS and (
        question_context or role(normalized) == "question"
    ):
        return "question_word"
    if normalized in STRUCTURAL_MODIFIERS:
        return "structural_modifier"
    return None


def is_service_word(
    lemma: str,
    *,
    part_of_speech: str | None = None,
    question_context: bool = False,
) -> bool:
    return (
        service_word_reason(
            lemma,
            part_of_speech=part_of_speech,
            question_context=question_context,
        )
        is not None
    )


def is_graph_concept(value: str) -> bool:
    """Return whether an already-normalized value may become a concept node."""

    normalized = normalize_service_word(value)
    return bool(normalized) and normalized not in SERVICE_WORD_LEMMAS
