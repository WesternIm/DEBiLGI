"""Intention-bound semantic plans for the first clarification-only iteration.

The module deliberately contains no Discord or persistence code.  It accepts an
already selected intention, expresses its required meaning as propositions and
lets the existing productive NLG realize that meaning.  It does not select an
intention and does not retrieve an answer from memory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import product
from types import MappingProxyType

from debilgi.language.content import content_lemmas
from debilgi.language.discourse import topic_keys
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    DependencyPattern,
    ProductiveSurface,
    SemanticDependency,
    realize_thought,
)
from debilgi.language.surface_coherence import evaluate_surface
from debilgi.language.tokenizer import tokenize

EXPRESS_LACK_OF_UNDERSTANDING = "express_lack_of_understanding"
REQUEST_CLARIFICATION = "request_clarification"

_CLARIFICATION_ALLOWED_LEMMAS = frozenset(
    {
        "быть",
        "в",
        "вы",
        "его",
        "иметь",
        "мочь",
        "мысль",
        "не",
        "непонятно",
        "он",
        "понимать",
        "понять",
        "пояснить",
        "про",
        "разобраться",
        "сообщение",
        "смысл",
        "ты",
        "уточнить",
        "что",
        "это",
        "этот",
        "я",
    }
)
_UNDERSTANDING_LEMMAS = frozenset({"понимать", "понять", "разобраться"})
_CLARIFICATION_LEMMAS = frozenset({"пояснить", "уточнить", "объяснить"})


@dataclass(frozen=True, slots=True)
class Proposition:
    """A wording-independent statement or requested operation."""

    subject: str
    predicate: str
    object: str | None = None
    qualifiers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticReplyPlan:
    """Meaning that must survive every surface realization."""

    plan_id: str
    intention_id: str
    speech_act: str
    propositions: tuple[Proposition, ...]
    requested_information: tuple[str, ...]
    target_user_id: int | None
    topic_ids: tuple[str, ...]
    emotional_tone: str | None
    confidence: float
    required_elements: tuple[str, ...]
    optional_elements: tuple[str, ...]
    forbidden_drift_topics: tuple[str, ...]
    supporting_memory_ids: tuple[str, ...]
    causal_hypothesis_ids: tuple[str, ...]
    source_information: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.plan_id or not self.intention_id:
            raise ValueError("semantic reply plan identifiers must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("semantic reply plan confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class PlanValidationResult:
    is_valid: bool
    coverage: float
    contradictions: tuple[str, ...]
    missing_elements: tuple[str, ...]
    semantic_drift: tuple[str, ...]
    unsupported_claims: tuple[str, ...]


class ProcessingOutcomeType(StrEnum):
    REPLIED = "replied"
    INTENTIONAL_SILENCE = "intentional_silence"
    FAILED_TO_FORM_INTENTION = "failed_to_form_intention"
    FAILED_TO_FORM_PLAN = "failed_to_form_plan"
    FAILED_TO_REALIZE_LANGUAGE = "failed_to_realize_language"
    VALIDATION_REJECTED = "validation_rejected"
    INTERNAL_EXCEPTION = "internal_exception"


@dataclass(frozen=True, slots=True)
class ProcessingOutcome:
    """Terminal diagnostic state; intention and implementation failures stay distinct."""

    outcome_type: ProcessingOutcomeType
    intention_id: str | None
    reply_plan_id: str | None
    generated_text: str | None
    failure_stage: str | None
    failure_reason: str | None
    was_intentional: bool
    diagnostic_data: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "diagnostic_data",
            MappingProxyType(dict(self.diagnostic_data)),
        )
        if self.was_intentional and self.failure_stage is not None:
            raise ValueError("an intentional outcome cannot have a technical failure stage")
        if self.outcome_type is ProcessingOutcomeType.INTENTIONAL_SILENCE:
            if not self.was_intentional or self.generated_text is not None:
                raise ValueError("intentional silence must be intentional and text-free")
        elif self.was_intentional:
            raise ValueError("technical and replied outcomes cannot be marked intentional")


@dataclass(frozen=True, slots=True)
class ClarificationCandidate:
    text: str
    source: str
    confidence: float
    validation: PlanValidationResult


@dataclass(frozen=True, slots=True)
class SemanticRealizationResult:
    plan: SemanticReplyPlan
    text: str | None
    validation: PlanValidationResult
    candidates: tuple[ClarificationCandidate, ...]
    outcome: ProcessingOutcome
    used_fallback: bool


def build_clarification_plan(
    *,
    intention_id: str,
    target_user_id: int | None,
    prompt: str,
    confidence: float,
    supporting_memory_ids: tuple[str, ...] = (),
    context_messages: tuple[str, ...] = (),
) -> SemanticReplyPlan:
    """Build the clarification meaning before any candidate text exists."""

    prompt_topics = tuple(dict.fromkeys(topic_keys(prompt)))
    active_topics = set(prompt_topics)
    context_topics = {
        topic for message in context_messages[-8:] for topic in topic_keys(message)
    }
    forbidden_drift = tuple(
        sorted(
            topic
            for topic in context_topics - active_topics
            if topic not in _CLARIFICATION_ALLOWED_LEMMAS
        )
    )
    causal_ids = tuple(
        item
        for item in supporting_memory_ids
        if item.startswith(("causal:", "hypothesis:"))
    )
    return SemanticReplyPlan(
        plan_id=f"semantic-reply:{intention_id}",
        intention_id=intention_id,
        speech_act="clarifying_question",
        propositions=(
            Proposition(
                "self",
                "understand",
                "current_message_meaning",
                (("polarity", "negative"),),
            ),
            Proposition(
                "interlocutor",
                "clarify",
                "intended_meaning",
                (("mode", "request"),),
            ),
        ),
        requested_information=("intended_meaning",),
        target_user_id=target_user_id,
        topic_ids=prompt_topics or ("current_message",),
        emotional_tone="neutral",
        confidence=max(0.0, min(1.0, confidence)),
        required_elements=(
            EXPRESS_LACK_OF_UNDERSTANDING,
            REQUEST_CLARIFICATION,
        ),
        optional_elements=("mention_ambiguous_message",),
        forbidden_drift_topics=forbidden_drift,
        supporting_memory_ids=tuple(dict.fromkeys(supporting_memory_ids)),
        causal_hypothesis_ids=causal_ids,
        source_information=("current_message", *supporting_memory_ids),
    )


def realize_clarification_plan(
    plan: SemanticReplyPlan,
    dependency_patterns: tuple[DependencyPattern, ...] = (),
    *,
    max_attempts: int = 3,
) -> SemanticRealizationResult:
    """Realize and validate bounded variants of one clarification meaning."""

    if plan.speech_act != "clarifying_question":
        validation = _invalid_validation("unsupported_speech_act")
        return SemanticRealizationResult(
            plan,
            None,
            validation,
            (),
            _technical_outcome(
                ProcessingOutcomeType.FAILED_TO_FORM_PLAN,
                plan,
                "semantic_planning",
                "unsupported speech act for clarification realizer",
            ),
            False,
        )

    attempts = max(1, min(max_attempts, 8))
    realization_failure: str | None = None
    try:
        lack_surfaces = _preferred_surfaces(
            realize_thought(_lack_of_understanding_thought(plan), dependency_patterns, limit=4)
        )
        request_surfaces = _preferred_surfaces(
            realize_thought(
                _clarification_request_thought(plan),
                dependency_patterns,
                limit=4,
            )
        )
    except Exception as exc:
        lack_surfaces = ()
        request_surfaces = ()
        realization_failure = f"productive NLG raised {type(exc).__name__}"
    candidates: list[ClarificationCandidate] = []
    for lack, request in product(lack_surfaces, request_surfaces):
        text = f"{lack.text} {request.text}"
        validation = validate_semantic_reply(plan, text)
        candidates.append(
            ClarificationCandidate(
                text=text,
                source=f"{lack.source}+{request.source}",
                confidence=min(plan.confidence, lack.confidence, request.confidence),
                validation=validation,
            )
        )
        if len(candidates) >= attempts:
            break

    valid = tuple(item for item in candidates if item.validation.is_valid)
    if valid:
        selected = max(valid, key=_candidate_score)
        return SemanticRealizationResult(
            plan,
            selected.text,
            selected.validation,
            tuple(candidates),
            ProcessingOutcome(
                ProcessingOutcomeType.REPLIED,
                plan.intention_id,
                plan.plan_id,
                selected.text,
                None,
                None,
                False,
                {
                    "candidate_count": len(candidates),
                    "selected_source": selected.source,
                    "used_fallback": False,
                },
            ),
            False,
        )

    failure_type = (
        ProcessingOutcomeType.FAILED_TO_REALIZE_LANGUAGE
        if not candidates
        else ProcessingOutcomeType.VALIDATION_REJECTED
    )
    failure_reason = realization_failure or (
        "productive NLG produced no candidate"
        if not candidates
        else "all productive candidates failed semantic validation"
    )
    return SemanticRealizationResult(
        plan,
        None,
        _invalid_validation("no_semantic_preserving_surface"),
        tuple(candidates),
        _technical_outcome(
            failure_type,
            plan,
            "language_realization" if not candidates else "semantic_validation",
            failure_reason,
            candidate_count=len(candidates),
        ),
        False,
    )


def validate_semantic_reply(
    plan: SemanticReplyPlan,
    text: str,
) -> PlanValidationResult:
    """Prove that wording still expresses every required clarification element."""

    tokens = tuple(token.casefold().replace("ё", "е") for token in tokenize(text))
    token_set = set(tokens)
    lemmas = set(content_lemmas(text))
    has_negation = "не" in token_set or "непонятно" in token_set
    lacks_understanding = bool(lemmas & _UNDERSTANDING_LEMMAS) and has_negation
    requests_clarification = bool(lemmas & _CLARIFICATION_LEMMAS) and text.rstrip().endswith(
        "?"
    )
    checks = {
        EXPRESS_LACK_OF_UNDERSTANDING: lacks_understanding,
        REQUEST_CLARIFICATION: requests_clarification,
    }
    missing = tuple(item for item in plan.required_elements if not checks.get(item, False))

    contradictions: list[str] = []
    if bool(lemmas & _UNDERSTANDING_LEMMAS) and not has_negation:
        contradictions.append("claims_understanding")
    if plan.speech_act == "clarifying_question" and not text.rstrip().endswith("?"):
        contradictions.append("question_became_statement")

    drift = tuple(sorted(lemmas & set(plan.forbidden_drift_topics)))
    allowed = {
        *_CLARIFICATION_ALLOWED_LEMMAS,
        *plan.topic_ids,
    }
    unsupported = tuple(
        sorted(
            lemma
            for lemma in lemmas - allowed
            if len(lemma) > 2 and lemma not in set(plan.forbidden_drift_topics)
        )
    )
    coverage = (len(plan.required_elements) - len(missing)) / max(
        1, len(plan.required_elements)
    )
    return PlanValidationResult(
        is_valid=(
            bool(text.strip())
            and coverage == 1.0
            and not contradictions
            and not drift
            and not unsupported
        ),
        coverage=coverage,
        contradictions=tuple(contradictions),
        missing_elements=missing,
        semantic_drift=drift,
        unsupported_claims=unsupported,
    )


def intentional_silence_outcome(
    *,
    intention_id: str,
    reason: str,
) -> ProcessingOutcome:
    """Represent a chosen silence without disguising it as a generation failure."""

    return ProcessingOutcome(
        ProcessingOutcomeType.INTENTIONAL_SILENCE,
        intention_id,
        None,
        None,
        None,
        None,
        True,
        {"reason": reason},
    )


def _lack_of_understanding_thought(plan: SemanticReplyPlan) -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme(
            "понимать",
            "понимаю",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "first")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme(
                        "я",
                        "я",
                        "pronoun",
                        (("number", "singular"), ("person", "first")),
                    )
                ),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme(
                        "сообщение",
                        "сообщение",
                        "noun",
                        (("number", "singular"), ("gender", "neuter")),
                    ),
                    modifiers=(ConceptLexeme("этот", "это", "adjective"),),
                ),
            ),
        ),
        clause_kind="statement",
        negated=True,
        focus_role="direct_object",
        confidence=plan.confidence,
        evidence=plan.source_information,
    )


def _clarification_request_thought(plan: SemanticReplyPlan) -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme(
            "мочь",
            "можешь",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "second")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme(
                        "ты",
                        "ты",
                        "pronoun",
                        (("number", "singular"), ("person", "second")),
                    )
                ),
            ),
            SemanticDependency(
                "action_complement",
                ConceptArgument(
                    ConceptLexeme(
                        "пояснить",
                        "пояснить",
                        "verb",
                        (("indeclinable", "true"),),
                    )
                ),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme(
                        "он",
                        "его",
                        "pronoun",
                        (("number", "singular"), ("gender", "neuter")),
                    )
                ),
            ),
        ),
        clause_kind="question",
        focus_role="direct_object",
        confidence=plan.confidence,
        evidence=plan.source_information,
        predicate_class="modal",
    )


def _preferred_surfaces(
    surfaces: tuple[ProductiveSurface, ...],
) -> tuple[ProductiveSurface, ...]:
    return tuple(
        sorted(
            surfaces,
            key=lambda item: (
                item.source != "productive_grammar",
                -item.confidence,
                item.text,
            ),
        )
    )


def _candidate_score(candidate: ClarificationCandidate) -> tuple[float, float, float]:
    surface = evaluate_surface(candidate.text)
    natural_order = float(candidate.source == "productive_grammar+productive_grammar")
    return (
        candidate.validation.coverage,
        surface.score + natural_order * 0.05,
        candidate.confidence,
    )


def _technical_outcome(
    outcome_type: ProcessingOutcomeType,
    plan: SemanticReplyPlan,
    stage: str,
    reason: str,
    **diagnostic_data: object,
) -> ProcessingOutcome:
    return ProcessingOutcome(
        outcome_type,
        plan.intention_id,
        plan.plan_id,
        None,
        stage,
        reason,
        False,
        diagnostic_data,
    )


def _invalid_validation(reason: str) -> PlanValidationResult:
    return PlanValidationResult(False, 0.0, (reason,), (), (), ())
