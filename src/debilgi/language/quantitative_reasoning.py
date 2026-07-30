"""Dimension-aware local reasoning for rate, distance and travel time."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.language.structured_utterance import (
    ClauseFrame,
    StructuredUtterance,
    TokenInterpretation,
)

_TRAVEL_PREDICATES = frozenset(
    {
        "добраться",
        "доехать",
        "дойти",
        "преодолеть",
        "проехать",
        "пройти",
    }
)
_MOTION_PREDICATES = frozenset(
    {"бежать", "двигаться", "ехать", "идти", "лететь", "мчаться", "плыть"}
)
_DISTANCE_FACTORS = {
    "км": Decimal("1000"),
    "километр": Decimal("1000"),
    "м": Decimal("1"),
    "метр": Decimal("1"),
}
_TIME_FACTORS = {
    "ч": Decimal("3600"),
    "час": Decimal("3600"),
    "мин": Decimal("60"),
    "минута": Decimal("60"),
    "с": Decimal("1"),
    "секунда": Decimal("1"),
}


@dataclass(frozen=True, slots=True)
class QuantitativeResolution:
    thought: ClauseThought
    confidence: float
    focus_terms: tuple[str, ...]
    evidence: tuple[str, ...]
    value_seconds: Decimal


def resolve_travel_time_question(
    structure: StructuredUtterance,
) -> QuantitativeResolution | None:
    """Solve ``time = distance / speed`` while preserving physical units."""

    question = next(
        (
            clause
            for clause in structure.clauses
            if clause.clause_kind == "question"
            and clause.predicate_index is not None
            and structure.tokens[clause.predicate_index].lemma in _TRAVEL_PREDICATES
            and _question_requests_time(structure, clause)
        ),
        None,
    )
    if question is None:
        return None
    premise = _rate_premise(structure, question)
    if premise is None:
        return None

    speed = _speed_meters_per_second(structure, premise)
    distance = _distance_meters(structure, question)
    if speed is None or distance is None or speed <= 0 or distance < 0:
        return None
    seconds = distance / speed
    duration = _duration_surface(seconds)
    if duration is None:
        return None

    subject = structure.tokens[premise.subject_indices[0]]
    predicate = structure.tokens[question.predicate_index]
    evidence = (
        f"distance_meters:{_decimal_surface(distance)}",
        f"speed_meters_per_second:{_decimal_surface(speed)}",
        "rate_equation:time=distance/speed",
        f"duration_seconds:{_decimal_surface(seconds)}",
    )
    thought = ClauseThought(
        predicate=_lexeme(predicate),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(_lexeme(subject), grammatical_case="nominative"),
            ),
            SemanticDependency(
                "manner",
                ConceptArgument(
                    ConceptLexeme(
                        duration,
                        duration,
                        "number",
                        entity_kind="duration",
                    ),
                    preposition="за",
                ),
            ),
        ),
        confidence=0.97,
        evidence=evidence,
    )
    return QuantitativeResolution(
        thought,
        0.97,
        (subject.lemma, _decimal_surface(distance), _decimal_surface(speed)),
        evidence,
        seconds,
    )


def _question_requests_time(
    structure: StructuredUtterance,
    clause: ClauseFrame,
) -> bool:
    lemmas = {token.lemma for token in structure.tokens[clause.token_start : clause.token_end]}
    return "когда" in lemmas or ("сколько" in lemmas and bool(lemmas & {"время", *_TIME_FACTORS}))


def _rate_premise(
    structure: StructuredUtterance,
    question: ClauseFrame,
) -> ClauseFrame | None:
    linked = next(
        (
            structure.clauses[link.child_clause_index]
            for link in structure.clause_links
            if link.parent_clause_index == question.clause_index
            and link.relation == "advcl:condition"
        ),
        None,
    )
    candidates = (
        (linked,)
        if linked is not None
        else tuple(
            clause for clause in structure.clauses if clause.clause_index < question.clause_index
        )
    )
    return next(
        (
            clause
            for clause in candidates
            if clause is not None
            and clause.predicate_index is not None
            and clause.subject_indices
            and structure.tokens[clause.predicate_index].lemma in _MOTION_PREDICATES
        ),
        None,
    )


def _speed_meters_per_second(
    structure: StructuredUtterance,
    clause: ClauseFrame,
) -> Decimal | None:
    tokens = structure.tokens[clause.token_start : clause.token_end]
    number = next((token for token in tokens if token.part_of_speech == "number"), None)
    if number is None:
        return None
    number_index = number.index
    distance_unit = next(
        (
            token
            for token in structure.tokens[number_index + 1 : clause.token_end]
            if token.lemma in _DISTANCE_FACTORS
        ),
        None,
    )
    time_unit = next(
        (
            token
            for token in structure.tokens[
                (
                    distance_unit.index + 1 if distance_unit is not None else number_index + 1
                ) : clause.token_end
            ]
            if token.lemma in _TIME_FACTORS
        ),
        None,
    )
    if distance_unit is None or time_unit is None:
        return None
    try:
        value = Decimal(number.lemma.replace(",", "."))
    except InvalidOperation:
        return None
    return value * _DISTANCE_FACTORS[distance_unit.lemma] / _TIME_FACTORS[time_unit.lemma]


def _distance_meters(
    structure: StructuredUtterance,
    clause: ClauseFrame,
) -> Decimal | None:
    tokens = structure.tokens[clause.token_start : clause.token_end]
    for number in tokens:
        if number.part_of_speech != "number":
            continue
        unit = next(
            (
                token
                for token in structure.tokens[number.index + 1 : clause.token_end]
                if token.lemma in _DISTANCE_FACTORS
            ),
            None,
        )
        if unit is None:
            continue
        try:
            value = Decimal(number.lemma.replace(",", "."))
        except InvalidOperation:
            continue
        return value * _DISTANCE_FACTORS[unit.lemma]
    return None


def _duration_surface(seconds: Decimal) -> str | None:
    if not seconds.is_finite() or seconds < 0:
        return None
    rounded = seconds.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    whole_seconds = int(rounded)
    remainder = rounded - Decimal(whole_seconds)
    hours, remaining = divmod(whole_seconds, 3600)
    minutes, seconds_integer = divmod(remaining, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} {_counted_form(hours, 'час', 'часа', 'часов')}")
    if minutes:
        parts.append(f"{minutes} {_counted_form(minutes, 'минута', 'минуты', 'минут')}")
    final_seconds = Decimal(seconds_integer) + remainder
    if final_seconds or not parts:
        surface = _decimal_surface(final_seconds)
        agreement_value = seconds_integer if remainder == 0 else 5
        parts.append(f"{surface} {_counted_form(agreement_value, 'секунда', 'секунды', 'секунд')}")
    return " ".join(parts)


def _counted_form(value: int, one: str, few: str, many: str) -> str:
    tail = abs(value) % 100
    if 11 <= tail <= 14:
        return many
    final = tail % 10
    if final == 1:
        return one
    if 2 <= final <= 4:
        return few
    return many


def _decimal_surface(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f").rstrip("0").rstrip(".").replace(".", ",")


def _lexeme(token: TokenInterpretation) -> ConceptLexeme:
    return ConceptLexeme(
        token.lemma,
        token.corrected_form or token.surface,
        token.part_of_speech,
        token.grammatical_features,
        token.entity_kind,
    )
