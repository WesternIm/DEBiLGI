"""Structured facts used as semantic invariants across surface realization."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from debilgi.language.content import content_lemmas
from debilgi.language.lexical_matching import same_lexical_family
from debilgi.language.structured_utterance import (
    ClauseFrame,
    StructuredUtterance,
    analyze_structured_utterance,
)


@dataclass(frozen=True, slots=True)
class StructuredFact:
    text: str
    subject: tuple[str, ...]
    predicate: str | None
    object: tuple[str, ...]
    object_quantifier: str
    polarity: str
    certainty: float
    target: str | None
    provenance: tuple[str, ...]

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        certainty: float = 1.0,
        target: str | None = None,
        provenance: tuple[str, ...] = (),
    ) -> StructuredFact:
        structure = analyze_structured_utterance(text)
        frame = structure.clauses[0] if structure.clauses else None
        if frame is None:
            return cls(
                text=text.strip(),
                subject=(),
                predicate=None,
                object=tuple(content_lemmas(text)),
                object_quantifier="all",
                polarity="negative" if _contains_negation(text) else "positive",
                certainty=_clamp(certainty),
                target=target,
                provenance=provenance,
            )
        return _fact_from_frame(
            structure,
            frame,
            text=text,
            certainty=certainty,
            target=target,
            provenance=provenance,
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def ensure_structured_fact(value: str | StructuredFact) -> StructuredFact:
    return value if isinstance(value, StructuredFact) else StructuredFact.from_text(value)


def covers_structured_fact(text: str, required: StructuredFact) -> bool:
    """Require semantic roles and polarity, falling back conservatively when parsing fails."""

    normalized_text = _normalize(text)
    normalized_fact = _normalize(required.text)
    if normalized_fact and normalized_fact in normalized_text:
        return True
    structure = analyze_structured_utterance(text)
    for frame in structure.clauses:
        candidate = _fact_from_frame(
            structure,
            frame,
            text=" ".join(
                token.surface
                for token in structure.tokens[frame.token_start : frame.token_end]
            ),
        )
        if candidate.polarity != required.polarity:
            continue
        if required.predicate is not None and not _same_term(
            required.predicate,
            candidate.predicate,
        ):
            continue
        if required.subject and not _covers_terms(candidate.subject, required.subject):
            continue
        if required.object and not _covers_terms(
            candidate.object,
            required.object,
            quantifier=required.object_quantifier,
        ):
            continue
        if required.target is not None and candidate.target not in {None, required.target}:
            continue
        if required.predicate is not None or required.subject or required.object:
            return True

    # A parser gap must not erase polarity. The lexical fallback is stricter
    # than the previous topical 72% check and preserves explicit negation.
    if _contains_negation(text) != (required.polarity == "negative"):
        return False
    required_terms = set(content_lemmas(required.text))
    text_terms = set(content_lemmas(text))
    return bool(required_terms) and required_terms <= text_terms


def _lemmas(structure, indices: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            structure.tokens[index].lemma
            for index in indices
            if structure.tokens[index].lemma not in {"не"}
        )
    )


def _covers_terms(
    candidate: tuple[str, ...],
    required: tuple[str, ...],
    *,
    quantifier: str = "all",
) -> bool:
    matches = tuple(
        any(same_lexical_family(item, available) for available in candidate)
        for item in required
    )
    return any(matches) if quantifier == "any" else all(matches)


def _fact_from_frame(
    structure: StructuredUtterance,
    frame: ClauseFrame,
    *,
    text: str,
    certainty: float = 1.0,
    target: str | None = None,
    provenance: tuple[str, ...] = (),
) -> StructuredFact:
    predicate = (
        structure.tokens[frame.predicate_index].lemma
        if frame.predicate_index is not None
        else frame.implicit_predicate
    )
    object_indices = {
        *frame.predicate_complement_indices,
        *frame.direct_object_indices,
        *frame.indirect_object_indices,
    }
    return StructuredFact(
        text=text.strip(),
        subject=_lemmas(structure, frame.subject_indices),
        predicate=predicate,
        object=_lemmas(structure, tuple(sorted(object_indices))),
        object_quantifier=(
            "any"
            if {"или", "либо"} & set(_normalize(text).split())
            else "all"
        ),
        polarity="negative" if frame.negated else "positive",
        certainty=_clamp(certainty),
        target=target,
        provenance=tuple(dict.fromkeys(provenance)),
    )


def _same_term(required: str, candidate: str | None) -> bool:
    return candidate is not None and same_lexical_family(required, candidate)


def _contains_negation(text: str) -> bool:
    return "не" in _normalize(text).split()


def _normalize(value: str) -> str:
    return " ".join(value.casefold().replace("ё", "е").split())


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
