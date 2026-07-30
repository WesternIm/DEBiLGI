"""Experimental semantic self-questioning over parsed scene roles.

The inquiry chain never stores natural-language prompt/reply pairs.  It asks
typed questions about semantic slots, records where every answer came from,
keeps hypotheses separate from explicit content and only then constructs
``ClauseThought`` objects for productive realization.
"""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    DependencyPattern,
    ProductiveSurface,
    SemanticDependency,
    realize_thought,
)
from debilgi.language.structured_utterance import (
    ClauseFrame,
    StructuredUtterance,
    TokenInterpretation,
)

_PURSUIT_MOTION_PREDICATES = frozenset(
    {
        "бежать",
        "двигаться",
        "ехать",
        "идти",
        "лететь",
        "мчаться",
        "плыть",
        "ползти",
    }
)
_DIRECT_PURSUIT_PREDICATES = frozenset({"гнаться", "догонять", "погнаться", "преследовать"})


@dataclass(frozen=True, slots=True)
class InquiryStep:
    step_id: str
    question_kind: str
    subject_lemma: str | None
    predicate_lemma: str | None
    answer_role: str
    answer_lemma: str | None
    status: str
    confidence: float
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InquiryConclusion:
    thought: ClauseThought
    epistemic_status: str
    confidence: float
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticInquiry:
    inquiry_id: str
    steps: tuple[InquiryStep, ...]
    conclusions: tuple[InquiryConclusion, ...]
    study_terms: tuple[str, ...]
    confidence: float
    evidence: tuple[str, ...]

    @property
    def renderable(self) -> bool:
        return bool(self.conclusions)


def analyze_semantic_inquiry(
    structure: StructuredUtterance,
    *,
    knowledge_facts: tuple[str, ...] = (),
    exploration: float = 0.5,
) -> SemanticInquiry:
    """Decompose clauses, answer explicit slots and derive guarded hypotheses."""

    steps: list[InquiryStep] = []
    conclusions: list[InquiryConclusion] = []
    study_terms: list[str] = []
    evidence: list[str] = ["semantic_inquiry:v1"]

    for clause in structure.clauses:
        subject = _first_token(structure, clause.subject_indices)
        predicate = (
            structure.tokens[clause.predicate_index] if clause.predicate_index is not None else None
        )
        steps.extend(_explicit_steps(structure, clause, subject, predicate))
        pursuit = _pursuit_inquiry(
            structure,
            clause,
            subject,
            predicate,
            knowledge_facts=knowledge_facts,
            exploration=exploration,
        )
        if pursuit is None:
            continue
        pursuit_steps, pursuit_conclusions, pursuit_study, pursuit_evidence = pursuit
        steps.extend(pursuit_steps)
        conclusions.extend(pursuit_conclusions)
        study_terms.extend(pursuit_study)
        evidence.extend(pursuit_evidence)

    confidence = (
        sum(item.confidence for item in conclusions) / len(conclusions)
        if conclusions
        else structure.confidence * 0.72
    )
    return SemanticInquiry(
        inquiry_id=f"semantic-inquiry:{_stable_signature(structure)}",
        steps=tuple(steps),
        conclusions=tuple(conclusions),
        study_terms=tuple(dict.fromkeys(study_terms))[:4],
        confidence=min(0.92, confidence),
        evidence=tuple(dict.fromkeys(evidence)),
    )


def realize_semantic_inquiry(
    inquiry: SemanticInquiry,
    patterns: tuple[DependencyPattern, ...] = (),
) -> tuple[ProductiveSurface, ...]:
    """Realize conclusions, preserving explicit/inferred/hypothetical status."""

    if not inquiry.conclusions:
        return ()
    realized: list[tuple[InquiryConclusion, ProductiveSurface]] = []
    for conclusion in inquiry.conclusions[:3]:
        surfaces = realize_thought(conclusion.thought, patterns, limit=2)
        if surfaces:
            realized.append((conclusion, surfaces[0]))
    if not realized:
        return ()

    bodies: list[str] = []
    combined_order: list[str] = []
    confidence = inquiry.confidence
    for index, (conclusion, surface) in enumerate(realized):
        body = surface.text.rstrip(" .!?…")
        if not body:
            continue
        marker = (
            "Похоже"
            if conclusion.epistemic_status == "inference"
            else "Вероятно"
            if conclusion.epistemic_status == "hypothesis"
            else None
        )
        if marker is not None:
            body = f"{marker}, {body[:1].casefold()}{body[1:]}"
        elif index:
            body = f"{body[:1].upper()}{body[1:]}"
        bodies.append(f"{body}.")
        combined_order.extend(f"conclusion:{index}:{role}" for role in surface.role_order)
        confidence = min(confidence, surface.confidence, conclusion.confidence)
    if not bodies:
        return ()
    return (
        ProductiveSurface(
            " ".join(bodies),
            tuple(combined_order),
            confidence,
            "semantic_inquiry_composition",
        ),
    )


def inquiry_trace(inquiry: SemanticInquiry) -> dict[str, object]:
    """Return a bounded, transparent trace without exposing hidden prose."""

    return {
        "inquiry_id": inquiry.inquiry_id,
        "renderable": inquiry.renderable,
        "confidence": inquiry.confidence,
        "study_terms": inquiry.study_terms,
        "evidence": inquiry.evidence,
        "steps": tuple(
            {
                "step_id": step.step_id,
                "question_kind": step.question_kind,
                "subject": step.subject_lemma,
                "predicate": step.predicate_lemma,
                "answer_role": step.answer_role,
                "answer": step.answer_lemma,
                "status": step.status,
                "confidence": step.confidence,
                "evidence": step.evidence,
            }
            for step in inquiry.steps
        ),
        "conclusions": tuple(
            {
                "epistemic_status": item.epistemic_status,
                "confidence": item.confidence,
                "predicate": item.thought.predicate.lemma,
                "roles": tuple(
                    (dependency.role, dependency.argument.head.lemma)
                    for dependency in item.thought.dependencies
                ),
                "evidence": item.evidence,
            }
            for item in inquiry.conclusions
        ),
    }


def _explicit_steps(
    structure: StructuredUtterance,
    clause: ClauseFrame,
    subject: TokenInterpretation | None,
    predicate: TokenInterpretation | None,
) -> tuple[InquiryStep, ...]:
    subject_lemma = subject.lemma if subject is not None else clause.implicit_subject
    predicate_lemma = predicate.lemma if predicate is not None else clause.implicit_predicate
    steps = [
        InquiryStep(
            f"clause:{clause.clause_index}:subject",
            "actor",
            None,
            predicate_lemma,
            "subject",
            subject_lemma,
            "explicit" if subject_lemma is not None else "unresolved",
            clause.confidence if subject_lemma is not None else 0.0,
            ("clause_subject",),
        ),
        InquiryStep(
            f"clause:{clause.clause_index}:predicate",
            "action",
            subject_lemma,
            None,
            "predicate",
            predicate_lemma,
            "explicit" if predicate_lemma is not None else "unresolved",
            clause.confidence if predicate_lemma is not None else 0.0,
            ("clause_predicate",),
        ),
    ]
    for role_name, indices in (
        ("direct_object", clause.direct_object_indices),
        ("indirect_object", clause.indirect_object_indices),
    ):
        for ordinal, index in enumerate(indices):
            token = structure.tokens[index]
            steps.append(
                InquiryStep(
                    f"clause:{clause.clause_index}:{role_name}:{ordinal}",
                    "argument",
                    subject_lemma,
                    predicate_lemma,
                    role_name,
                    token.lemma,
                    "explicit",
                    clause.confidence,
                    (f"clause_role:{role_name}",),
                )
            )
    for complement_index, complement in enumerate(clause.complements):
        for object_index in complement.object_indices:
            token = structure.tokens[object_index]
            steps.append(
                InquiryStep(
                    f"clause:{clause.clause_index}:complement:{complement_index}",
                    "circumstance",
                    subject_lemma,
                    predicate_lemma,
                    complement.role,
                    token.lemma,
                    "explicit",
                    clause.confidence,
                    (
                        f"clause_complement:{complement.role}",
                        f"preposition:{structure.tokens[complement.preposition_index].lemma}",
                    ),
                )
            )
    return tuple(steps)


def _pursuit_inquiry(
    structure: StructuredUtterance,
    clause: ClauseFrame,
    subject: TokenInterpretation | None,
    predicate: TokenInterpretation | None,
    *,
    knowledge_facts: tuple[str, ...],
    exploration: float,
) -> (
    tuple[
        tuple[InquiryStep, ...],
        tuple[InquiryConclusion, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]
    | None
):
    if subject is None or predicate is None or _animacy(subject) != "animate":
        return None
    target = _pursuit_target(structure, clause, predicate)
    if target is None or _animacy(target) != "animate":
        return None

    construction_evidence = (
        "schema:animate_motion_after_animate_target",
        f"source_predicate:{predicate.lemma}",
        f"source_preposition:{_target_preposition(structure, clause, target.index) or 'direct'}",
    )
    knowledge_support = _knowledge_support(
        knowledge_facts,
        subject.lemma,
        target.lemma,
    )
    relation_confidence = min(
        0.86,
        0.70 + knowledge_support * 0.12 + max(0.0, exploration - 0.5) * 0.08,
    )
    pursuit_thought = ClauseThought(
        predicate=ConceptLexeme(
            "преследовать",
            "преследует",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "third")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(_lexeme(subject), grammatical_case="nominative"),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(_lexeme(target), grammatical_case="accusative"),
            ),
        ),
        confidence=relation_confidence,
        evidence=construction_evidence,
    )
    target_action_confidence = min(
        0.80,
        0.58 + knowledge_support * 0.16 + max(0.0, exploration - 0.4) * 0.10,
    )
    target_action_evidence = (
        *construction_evidence,
        "hypothesis:pursuit_target_moves_away",
    )
    escape_thought = ClauseThought(
        predicate=ConceptLexeme(
            "убегать",
            "убегает",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "third")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(_lexeme(target), grammatical_case="nominative"),
            ),
            SemanticDependency(
                "location",
                ConceptArgument(
                    _lexeme(subject),
                    preposition="от",
                    grammatical_case="genitive",
                ),
            ),
        ),
        confidence=target_action_confidence,
        evidence=target_action_evidence,
    )
    steps = (
        InquiryStep(
            f"clause:{clause.clause_index}:semantic_relation",
            "relation",
            subject.lemma,
            predicate.lemma,
            "pursuit_target",
            target.lemma,
            "inferred",
            relation_confidence,
            construction_evidence,
        ),
        InquiryStep(
            f"clause:{clause.clause_index}:target_action",
            "participant_action",
            target.lemma,
            None,
            "predicate",
            "убегать",
            "hypothesis",
            target_action_confidence,
            target_action_evidence,
        ),
        InquiryStep(
            f"clause:{clause.clause_index}:purpose",
            "purpose",
            subject.lemma,
            predicate.lemma,
            "goal",
            None,
            "unresolved",
            0.0,
            ("purpose_not_expressed",),
        ),
        InquiryStep(
            f"clause:{clause.clause_index}:cause",
            "cause",
            subject.lemma,
            predicate.lemma,
            "cause",
            None,
            "unresolved",
            0.0,
            ("cause_not_expressed",),
        ),
    )
    conclusions = (
        InquiryConclusion(
            pursuit_thought,
            "inference",
            relation_confidence,
            construction_evidence,
        ),
        *(
            (
                InquiryConclusion(
                    escape_thought,
                    "hypothesis",
                    target_action_confidence,
                    target_action_evidence,
                ),
            )
            if exploration >= 0.35
            else ()
        ),
    )
    return (
        steps,
        conclusions,
        (subject.lemma, target.lemma, "преследование"),
        (*construction_evidence, "unresolved:purpose", "unresolved:cause"),
    )


def _pursuit_target(
    structure: StructuredUtterance,
    clause: ClauseFrame,
    predicate: TokenInterpretation,
) -> TokenInterpretation | None:
    if predicate.lemma in _DIRECT_PURSUIT_PREDICATES:
        return _first_token(structure, clause.direct_object_indices)
    if predicate.lemma not in _PURSUIT_MOTION_PREDICATES:
        return None
    for complement in clause.complements:
        preposition = structure.tokens[complement.preposition_index]
        if preposition.lemma != "за":
            continue
        target = _first_token(structure, complement.object_indices)
        if target is not None:
            return target
    return None


def _target_preposition(
    structure: StructuredUtterance,
    clause: ClauseFrame,
    target_index: int,
) -> str | None:
    return next(
        (
            structure.tokens[complement.preposition_index].lemma
            for complement in clause.complements
            if target_index in complement.object_indices
        ),
        None,
    )


def _knowledge_support(
    facts: tuple[str, ...],
    subject: str,
    target: str,
) -> float:
    relevant = tuple(
        fact.casefold().replace("ё", "е")
        for fact in facts
        if subject in fact.casefold().replace("ё", "е")
        or target in fact.casefold().replace("ё", "е")
    )
    if not relevant:
        return 0.0
    pursuit_markers = {
        "добыча",
        "догонять",
        "охотиться",
        "преследовать",
        "убегать",
        "хищник",
    }
    return min(
        1.0,
        sum(0.34 for fact in relevant if any(marker in fact for marker in pursuit_markers)),
    )


def _first_token(
    structure: StructuredUtterance,
    indices: tuple[int, ...],
) -> TokenInterpretation | None:
    return structure.tokens[indices[0]] if indices else None


def _animacy(token: TokenInterpretation) -> str | None:
    return dict(token.grammatical_features).get("animacy")


def _lexeme(token: TokenInterpretation) -> ConceptLexeme:
    return ConceptLexeme(
        token.lemma,
        token.corrected_form or token.surface,
        token.part_of_speech,
        token.grammatical_features,
        token.entity_kind,
    )


def _stable_signature(structure: StructuredUtterance) -> str:
    material = "|".join(
        ":".join(
            (
                token.lemma,
                token.semantic_role,
                str(token.clause_index),
            )
        )
        for token in structure.tokens
        if token.surface[:1].isalnum()
    )
    value = 0
    for character in material:
        value = (value * 131 + ord(character)) % (1 << 48)
    return f"{value:012x}"
