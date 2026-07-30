"""Local symbolic NLG from semantic relations and Russian constructions.

The module receives no user-facing answer strings.  It realizes semantic
roles with morphology and grammatical operators, so learned knowledge changes
the content while constructions remain reusable across arbitrary lexemes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from debilgi.language.facts import StructuredFact
from debilgi.language.morphology import DEFAULT_MORPHOLOGY

_CASES = frozenset(
    {
        "nominative",
        "genitive",
        "dative",
        "accusative",
        "instrumental",
        "prepositional",
        "truth_case",
    }
)
_OPERATORS = frozenset(
    {
        "polar_marker",
        "epistemic_marker",
        "comma",
        "sentence_break",
        "possession_preposition",
        "existence_predicate",
        "question_particle",
    }
)
_TRUTH_STATES = frozenset({"true", "false", "unknown", "known"})
_SUPPORTED_RELATIONS = frozenset({"possession"})


@dataclass(frozen=True, slots=True)
class SemanticTerm:
    lemma: str
    surface: str
    part_of_speech: str = "noun"
    number: str | None = None
    entity_kind: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticClause:
    relation: str
    subject: SemanticTerm
    object: SemanticTerm
    truth: bool | None


@dataclass(frozen=True, slots=True)
class ResponseContentPlan:
    response_act: str
    answer_polarity: bool | None
    clauses: tuple[SemanticClause, ...]
    confidence: float
    evidence: tuple[str, ...] = ()
    response_goal: str = ""
    target_id: int | None = None
    topic_ids: tuple[str, ...] = ()
    key_points: tuple[str, ...] = ()
    required_facts: tuple[StructuredFact, ...] = ()
    optional_facts: tuple[StructuredFact, ...] = ()
    unknowns: tuple[str, ...] = ()
    tone: str = "neutral"
    follow_up: str | None = None

    @property
    def speech_act(self) -> str:
        return self.response_act


@dataclass(frozen=True, slots=True)
class ConstructionSurface:
    text: str
    construction: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ConstructionSpec:
    """Validated symbolic construction; it cannot contain authored wording."""

    key: str
    response_act: str
    relation: str
    truth_state: str
    sequence: tuple[str, ...]
    confidence: float = 0.70
    source: str = "learned"

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        source: str = "semantic_consultant",
        confidence: float = 0.70,
    ) -> ConstructionSpec | None:
        key = _identifier(payload.get("key"))
        response_act = _identifier(payload.get("response_act"))
        relation = _identifier(payload.get("relation"))
        truth_state = _identifier(payload.get("truth_state"))
        raw_sequence = payload.get("sequence")
        if not isinstance(raw_sequence, (list, tuple)):
            return None
        sequence = tuple(
            token
            for item in raw_sequence[:16]
            if (token := _symbol(item)) is not None
        )
        spec = cls(
            key=key,
            response_act=response_act,
            relation=relation,
            truth_state=truth_state,
            sequence=sequence,
            confidence=max(0.0, min(0.86, confidence)),
            source=source,
        )
        return spec if validate_construction_spec(spec) else None

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "response_act": self.response_act,
            "relation": self.relation,
            "truth_state": self.truth_state,
            "sequence": list(self.sequence),
        }


_BUILTIN_CONSTRUCTIONS = (
    ConstructionSpec(
        "polar_possession_joined",
        "relation_answer",
        "possession",
        "known",
        (
            "polar_marker",
            "comma",
            "possession_preposition",
            "subject:genitive",
            "existence_predicate",
            "object:truth_case",
        ),
        1.0,
        "builtin",
    ),
    ConstructionSpec(
        "polar_possession_split",
        "relation_answer",
        "possession",
        "known",
        (
            "polar_marker",
            "sentence_break",
            "possession_preposition",
            "subject:genitive",
            "existence_predicate",
            "object:truth_case",
        ),
        0.92,
        "builtin",
    ),
    ConstructionSpec(
        "unknown_possession",
        "relation_answer",
        "possession",
        "unknown",
        (
            "epistemic_marker",
            "comma",
            "existence_predicate",
            "question_particle",
            "possession_preposition",
            "subject:genitive",
            "object:nominative",
        ),
        0.76,
        "builtin",
    ),
)


def realize_content_plan(
    plan: ResponseContentPlan,
    constructions: tuple[ConstructionSpec, ...] = (),
) -> tuple[ConstructionSurface, ...]:
    """Produce grammatical variants without selecting facts or intentions."""

    if len(plan.clauses) != 1:
        return ()
    clause = plan.clauses[0]
    available = tuple(
        dict.fromkeys(
            spec
            for spec in (*_BUILTIN_CONSTRUCTIONS, *constructions)
            if validate_construction_spec(spec)
        )
    )
    surfaces: list[ConstructionSurface] = []
    seen: set[str] = set()
    for spec in available:
        if not _construction_applies(spec, plan, clause):
            continue
        text = _realize_spec(spec, plan, clause)
        normalized = " ".join(text.casefold().split())
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        surfaces.append(
            ConstructionSurface(
                text,
                f"{spec.source}:{spec.key}",
                min(plan.confidence, spec.confidence),
            )
        )
    return tuple(surfaces)


def validate_construction_spec(spec: ConstructionSpec) -> bool:
    """Reject lexical templates and structurally incomplete schemas."""

    if (
        not spec.key
        or not spec.response_act
        or spec.relation not in _SUPPORTED_RELATIONS
        or spec.truth_state not in _TRUTH_STATES
        or not 4 <= len(spec.sequence) <= 16
    ):
        return False
    if any(not _valid_symbol(item) for item in spec.sequence):
        return False
    subject_slots = tuple(item for item in spec.sequence if item.startswith("subject:"))
    object_slots = tuple(item for item in spec.sequence if item.startswith("object:"))
    if len(subject_slots) != 1 or len(object_slots) != 1:
        return False
    if spec.relation == "possession" and not {
        "possession_preposition",
        "existence_predicate",
    } <= set(spec.sequence):
        return False
    if spec.truth_state == "unknown":
        return "epistemic_marker" in spec.sequence and "polar_marker" not in spec.sequence
    return "polar_marker" in spec.sequence


def _construction_applies(
    spec: ConstructionSpec,
    plan: ResponseContentPlan,
    clause: SemanticClause,
) -> bool:
    if spec.response_act != plan.response_act or spec.relation != clause.relation:
        return False
    state = "unknown" if clause.truth is None else "true" if clause.truth else "false"
    return spec.truth_state == state or (spec.truth_state == "known" and state != "unknown")


def _realize_spec(
    spec: ConstructionSpec,
    plan: ResponseContentPlan,
    clause: SemanticClause,
) -> str:
    tokens: list[str] = []
    for symbol in spec.sequence:
        if symbol == "polar_marker":
            if plan.answer_polarity is None:
                return ""
            tokens.append("Да" if plan.answer_polarity else "Нет")
        elif symbol == "epistemic_marker":
            tokens.extend(("Не", "знаю"))
        elif symbol == "comma":
            tokens.append(",")
        elif symbol == "sentence_break":
            tokens.append(".")
        elif symbol == "possession_preposition":
            tokens.append("у")
        elif symbol == "existence_predicate":
            tokens.append("нет" if clause.truth is False else "есть")
        elif symbol == "question_particle":
            tokens.append("ли")
        elif symbol.startswith("subject:"):
            tokens.append(_inflect(clause.subject, case=symbol.partition(":")[2]))
        elif symbol.startswith("object:"):
            case = symbol.partition(":")[2]
            if case == "truth_case":
                case = "genitive" if clause.truth is False else "nominative"
            tokens.append(_inflect(clause.object, case=case))
    return _assemble(tokens)


def _inflect(term: SemanticTerm, *, case: str) -> str:
    requested: dict[str, str] = {"case": case}
    if term.number:
        requested["number"] = term.number
    if term.entity_kind is not None:
        return term.surface
    return DEFAULT_MORPHOLOGY.inflect(term.lemma, **requested) or term.surface


def _finish(text: str) -> str:
    stripped = " ".join(text.split()).strip(" .!?\u2026")
    return stripped[:1].upper() + stripped[1:] + "." if stripped else ""


def _assemble(tokens: list[str]) -> str:
    result = ""
    capitalize = True
    for token in tokens:
        if not token:
            return ""
        if token == ",":
            result = result.rstrip() + ","
            continue
        if token == ".":
            result = result.rstrip() + "."
            capitalize = True
            continue
        word = token[:1].upper() + token[1:] if capitalize else token
        result += ("" if not result or result.endswith(" ") else " ") + word
        capitalize = False
    return _finish(result)


def _identifier(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.casefold().strip()
    return normalized if normalized.replace("_", "").isalnum() else ""


def _symbol(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.casefold().strip()
    return normalized if _valid_symbol(normalized) else None


def _valid_symbol(symbol: str) -> bool:
    if symbol in _OPERATORS:
        return True
    role, separator, case = symbol.partition(":")
    return bool(separator) and role in {"subject", "object"} and case in _CASES
