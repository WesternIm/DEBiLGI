"""Turn externally proposed semantics into locally authored Russian surfaces.

The external model never supplies a reply string, surface forms, word order or
style. It may propose predicate lemmas, semantic roles and evidence links.
The local personality ranks those alternatives and the local morphology
realizes them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from debilgi.language.content import content_lemmas
from debilgi.language.lexical_matching import same_lexical_family
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    DependencyPattern,
    SemanticDependency,
    realize_thought,
)
from debilgi.language.semantic_consultant import (
    ConsultantResponseArgument,
    ConsultantResponsePlan,
    ConsultantResponseProposition,
    ConsultationResult,
)
from debilgi.personality.perspective import PersonalityPerspective


@dataclass(frozen=True, slots=True)
class AutonomousResponseCandidate:
    text: str
    move: str
    stance: str
    confidence: float
    personality_fit: float
    evidence_ids: tuple[str, ...]
    proposition_count: int
    grounding_compiled: bool


def realize_autonomous_responses(
    consultation: ConsultationResult | None,
    *,
    perspective: PersonalityPerspective | None,
    allowed_evidence_ids: tuple[str, ...],
    grounding_facts: tuple[tuple[str, str], ...] = (),
    dependency_patterns: tuple[DependencyPattern, ...] = (),
) -> tuple[AutonomousResponseCandidate, ...]:
    """Validate, rank and locally realize semantic alternatives."""

    if consultation is None or not consultation.response_plans:
        return ()
    allowed = set(allowed_evidence_ids)
    grounding = dict(grounding_facts)
    candidates: list[AutonomousResponseCandidate] = []
    for plan in consultation.response_plans:
        normalized_plan = replace(
            plan,
            propositions=tuple(
                replace(
                    proposition,
                    evidence_ids=tuple(
                        _canonical_evidence_id(evidence_id, allowed)
                        for evidence_id in proposition.evidence_ids
                    ),
                )
                for proposition in plan.propositions
            ),
        )
        propositions = tuple(
            dict.fromkeys(
                _compile_grounded_proposition(
                    proposition,
                    grounding=grounding,
                )
                for proposition in normalized_plan.propositions
            )
        )
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for proposition in propositions
                for evidence_id in proposition.evidence_ids
            )
        )
        if any(evidence_id not in allowed for evidence_id in evidence_ids):
            continue
        if _requires_grounding(normalized_plan) and (
            not evidence_ids
            or any(
                not proposition.evidence_ids
                for proposition in propositions
            )
        ):
            continue
        if grounding and any(
            not _proposition_is_grounded(
                proposition,
                move=normalized_plan.move,
                grounding=grounding,
            )
            for proposition in propositions
        ):
            continue
        texts: list[str] = []
        seen_surfaces: set[str] = set()
        surface_confidences: list[float] = []
        for proposition in propositions:
            thought = _thought(proposition, normalized_plan.confidence)
            if thought is None:
                texts = []
                break
            surfaces = realize_thought(
                thought,
                dependency_patterns,
                limit=1,
            )
            if not surfaces:
                texts = []
                break
            normalized_surface = " ".join(
                surfaces[0].text.casefold().split()
            )
            if normalized_surface in seen_surfaces:
                continue
            seen_surfaces.add(normalized_surface)
            texts.append(surfaces[0].text)
            surface_confidences.append(surfaces[0].confidence)
        if not texts:
            continue
        personality_fit = _personality_fit(
            perspective,
            move=normalized_plan.move,
            stance=normalized_plan.stance,
        )
        confidence = min(
            normalized_plan.confidence,
            *surface_confidences,
            0.62 + personality_fit * 0.34,
        )
        candidates.append(
            AutonomousResponseCandidate(
                text=" ".join(texts),
                move=normalized_plan.move,
                stance=normalized_plan.stance,
                confidence=confidence,
                personality_fit=personality_fit,
                evidence_ids=evidence_ids,
                proposition_count=len(propositions),
                grounding_compiled=propositions != plan.propositions,
            )
        )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item.confidence,
            -item.personality_fit,
            item.move,
            item.text,
        ),
    )
    unique: list[AutonomousResponseCandidate] = []
    seen_texts: set[str] = set()
    for candidate in ranked:
        normalized = " ".join(candidate.text.casefold().split())
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        unique.append(candidate)
    return tuple(unique)


def _canonical_evidence_id(value: str, allowed: set[str]) -> str:
    """Accept harmless provider wrappers while preserving the strict allow-list."""

    normalized = value.strip().strip("`\"'")
    candidates = [normalized]
    for prefix in (
        "grounding_facts.id=",
        "grounding_fact.id=",
        "grounding_facts.",
        "evidence_id=",
        "id=",
    ):
        if normalized.startswith(prefix):
            candidates.append(normalized.removeprefix(prefix).strip().strip("`\"'"))
    return next((candidate for candidate in candidates if candidate in allowed), normalized)


def _thought(
    proposition: ConsultantResponseProposition,
    plan_confidence: float,
) -> ClauseThought | None:
    predicate_lemma = _verb_lemma(proposition.predicate)
    if predicate_lemma is None:
        return None
    dependencies: list[SemanticDependency] = []
    seen_roles: set[str] = set()
    for argument in proposition.arguments:
        if argument.role in seen_roles:
            return None
        realized_argument = _validated_argument(argument)
        if realized_argument is None:
            return None
        seen_roles.add(argument.role)
        dependencies.append(
            SemanticDependency(argument.role, realized_argument)
        )
    return ClauseThought(
        predicate=ConceptLexeme(
            predicate_lemma,
            predicate_lemma,
            "verb",
            proposition.predicate_features,
        ),
        dependencies=tuple(dependencies),
        clause_kind=proposition.clause_kind,
        negated=proposition.negated,
        response_polarity=proposition.response_polarity,
        confidence=plan_confidence,
        evidence=proposition.evidence_ids,
        predicate_class=(
            "modal"
            if predicate_lemma in {"мочь", "уметь", "хотеть"}
            and "action_complement" in seen_roles
            else "lexical"
        ),
    )


def _verb_lemma(value: str) -> str | None:
    readings = tuple(
        reading
        for reading in DEFAULT_MORPHOLOGY.parse_word(value)
        if reading.part_of_speech == "verb"
        and "INFN" in reading.grammemes
    )
    if not readings:
        return None
    return max(readings, key=lambda item: item.score).lemma


def _proposition_is_grounded(
    proposition: ConsultantResponseProposition,
    *,
    move: str,
    grounding: dict[str, str],
) -> bool:
    for evidence_id in proposition.evidence_ids:
        value = grounding.get(evidence_id)
        if value is None:
            continue
        structured = _structured_fact(value)
        if structured.get("predicate"):
            if _matches_structured_fact(proposition, structured):
                return True
            continue
        if evidence_id.startswith("personality.") and any(
            marker in move.casefold()
            for marker in ("decide", "position", "engage", "acknowledge")
        ):
            return True
        fact_terms = content_lemmas(value)
        proposition_terms = (
            proposition.predicate,
            *(
                term
                for argument in proposition.arguments
                for term in (argument.lemma, *argument.modifiers)
            ),
        )
        if any(
            same_lexical_family(fact_term, proposition_term)
            for fact_term in fact_terms
            for proposition_term in proposition_terms
        ):
            return True
    return False


def _compile_grounded_proposition(
    proposition: ConsultantResponseProposition,
    *,
    grounding: dict[str, str],
) -> ConsultantResponseProposition:
    structured_facts = tuple(
        (evidence_id, structured)
        for evidence_id in proposition.evidence_ids
        if (
            value := grounding.get(evidence_id)
        ) is not None
        and (structured := _structured_fact(value)).get("predicate")
    )
    if not structured_facts:
        return proposition
    evidence_id, fact = structured_facts[0]
    arguments = tuple(
        _fact_argument(role, fact)
        for role in (
            "subject",
            "direct_object",
            "indirect_object",
            "predicate_complement",
            "action_complement",
            "manner",
            "cause",
            "location",
            "topic",
        )
        if fact.get(role)
    )
    predicate_features = tuple(
        (name, fact[name])
        for name in ("tense", "person", "number", "gender")
        if fact.get(name)
    )
    return ConsultantResponseProposition(
        predicate=fact["predicate"],
        predicate_features=(
            predicate_features or proposition.predicate_features
        ),
        arguments=arguments,
        clause_kind=proposition.clause_kind,
        negated=fact.get("truth", "").casefold() == "false",
        response_polarity=proposition.response_polarity,
        evidence_ids=(evidence_id,),
    )


def _fact_argument(
    role: str,
    fact: dict[str, str],
) -> ConsultantResponseArgument:
    raw_lemma = fact[role]
    lemma = "я" if raw_lemma == "agent" else raw_lemma
    reading = DEFAULT_MORPHOLOGY.best(lemma)
    part_of_speech = (
        "pronoun"
        if lemma in {"я", "ты", "мы", "вы", "он", "она", "оно", "они"}
        else reading.part_of_speech
        if reading is not None
        and reading.part_of_speech
        in {"noun", "verb", "adjective", "adverb", "pronoun", "numeral"}
        else "noun"
    )
    return ConsultantResponseArgument(
        role=role,
        lemma=lemma,
        part_of_speech=part_of_speech,
        grammatical_case=fact.get(f"{role}_case"),
        preposition=fact.get(f"{role}_preposition"),
        modifiers=tuple(
            item.strip()
            for item in fact.get(f"{role}_modifiers", "").split(",")
            if item.strip()
        ),
    )


def _structured_fact(value: str) -> dict[str, str]:
    prefix = value.partition(".")[0]
    pairs: dict[str, str] = {}
    for item in prefix.split(";"):
        key, separator, raw_value = item.partition("=")
        if separator and key.strip() and raw_value.strip():
            pairs[key.strip()] = raw_value.strip()
    return pairs


def _matches_structured_fact(
    proposition: ConsultantResponseProposition,
    fact: dict[str, str],
) -> bool:
    if not same_lexical_family(
        proposition.predicate,
        fact["predicate"],
    ):
        return False
    truth = fact.get("truth", "").casefold()
    if truth == "false" and not proposition.negated:
        return False
    if truth == "true" and proposition.negated:
        return False
    arguments_by_role = {
        argument.role: argument
        for argument in proposition.arguments
    }
    for role in (
        "subject",
        "direct_object",
        "indirect_object",
        "predicate_complement",
        "action_complement",
        "manner",
        "cause",
        "location",
        "topic",
    ):
        expected = fact.get(role)
        if expected is None:
            continue
        argument = arguments_by_role.get(role)
        if argument is None:
            return False
        if expected == "agent":
            if argument.lemma != "я":
                return False
            continue
        expected_terms = content_lemmas(expected)
        actual_terms = (argument.lemma, *argument.modifiers)
        if not any(
            same_lexical_family(expected_term, actual_term)
            for expected_term in expected_terms
            for actual_term in actual_terms
        ):
            return False
    return True


def _validated_argument(
    argument: ConsultantResponseArgument,
) -> ConceptArgument | None:
    pronouns = {"я", "ты", "мы", "вы", "он", "она", "оно", "они"}
    reading = DEFAULT_MORPHOLOGY.best(argument.lemma)
    if argument.lemma in pronouns:
        local_part_of_speech = "pronoun"
        local_lemma = argument.lemma
    elif reading is None:
        local_part_of_speech = "noun"
        local_lemma = argument.lemma
    else:
        local_part_of_speech = reading.part_of_speech
        local_lemma = reading.lemma
        # The semantic contract requires dictionary lemmas.  Accepting an
        # inflected participle as a nominal head produced surfaces such as
        # ``я отвечаю отвечающую``.
        if argument.lemma.casefold().replace("ё", "е") != (
            local_lemma.casefold().replace("ё", "е")
        ):
            return None

    nominal_roles = {
        "subject",
        "direct_object",
        "indirect_object",
        "predicate_complement",
        "cause",
        "location",
        "topic",
    }
    if (
        argument.role in nominal_roles
        and local_part_of_speech
        not in {"noun", "pronoun", "numeral", "adjective"}
    ):
        return None
    if argument.role == "action_complement" and local_part_of_speech != "verb":
        return None
    if (
        argument.role == "manner"
        and local_part_of_speech not in {"adverb", "noun"}
    ):
        return None

    modifiers: list[ConceptLexeme] = []
    for modifier in argument.modifiers:
        modifier_reading = DEFAULT_MORPHOLOGY.best(modifier)
        if (
            modifier_reading is None
            or modifier_reading.part_of_speech
            not in {"adjective", "participle"}
        ):
            return None
        if modifier.casefold().replace("ё", "е") != (
            modifier_reading.lemma.casefold().replace("ё", "е")
        ):
            return None
        modifiers.append(
            ConceptLexeme(
                modifier_reading.lemma,
                modifier_reading.lemma,
                modifier_reading.part_of_speech,
            )
        )

    return ConceptArgument(
        head=ConceptLexeme(
            local_lemma,
            local_lemma,
            local_part_of_speech,
        ),
        modifiers=tuple(modifiers),
        preposition=argument.preposition,
        grammatical_case=argument.grammatical_case,
    )


def _requires_grounding(plan: ConsultantResponsePlan) -> bool:
    normalized = plan.move.casefold().replace("-", "_")
    return any(
        marker in normalized
        for marker in (
            "answer",
            "assert",
            "correct",
            "explain",
            "inform",
            "position",
            "repair",
        )
    )


def _personality_fit(
    perspective: PersonalityPerspective | None,
    *,
    move: str,
    stance: str,
) -> float:
    if perspective is None:
        return 0.5
    normalized_stance = stance.casefold().replace("-", "_")
    stance_fit = {
        "direct": (
            perspective.drive("assertiveness") * 0.62
            + perspective.drive("coherence") * 0.38
        ),
        "warm": (
            perspective.drive("warmth") * 0.58
            + perspective.drive("trust") * 0.24
            + perspective.drive("affinity") * 0.18
        ),
        "playful": (
            perspective.drive("playfulness") * 0.60
            + perspective.drive("expressiveness") * 0.24
            + perspective.drive("arousal") * 0.16
        ),
        "skeptical": (
            perspective.drive("skepticism") * 0.58
            + perspective.drive("caution") * 0.24
            + perspective.drive("independence") * 0.18
        ),
        "curious": (
            perspective.drive("curiosity") * 0.58
            + perspective.drive("exploration") * 0.28
            + perspective.drive("patience") * 0.14
        ),
        "neutral": perspective.drive("coherence"),
    }.get(normalized_stance, perspective.drive("coherence"))
    return max(
        0.0,
        min(
            1.0,
            perspective.candidate_fit(move) * 0.70 + stance_fit * 0.30,
        ),
    )
