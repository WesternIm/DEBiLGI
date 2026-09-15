"""Grounded response meaning plans and a conservative Russian surface realizer."""

from __future__ import annotations

from dataclasses import dataclass, replace

from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.planning import UtterancePlan
from debilgi.language.sense_resolution import (
    UtteranceSenseResolution,
    apply_sense_resolution,
)
from debilgi.language.structured_utterance import (
    ClauseComplement,
    ClauseFrame,
    StructuredUtterance,
    TokenInterpretation,
    analyze_structured_utterance,
)
from debilgi.language.surface_coherence import COMPOSED_SURFACE_SOFT_ISSUES, evaluate_surface
from debilgi.language.valency import (
    assess_predicate_valency,
    governed_object_case,
    resolve_complement_role,
)


@dataclass(frozen=True, slots=True)
class MeaningWord:
    lemma: str
    preferred_surface: str
    part_of_speech: str | None
    grammatical_features: tuple[tuple[str, str], ...]
    grammemes: tuple[str, ...] = ()
    selected_meaning: str | None = None
    entity_kind: str | None = None
    entity_id: int | None = None


@dataclass(frozen=True, slots=True)
class MeaningArgument:
    head: MeaningWord
    modifiers: tuple[MeaningWord, ...] = ()


@dataclass(frozen=True, slots=True)
class MeaningComplement:
    role: str
    preposition: str
    argument: MeaningArgument


@dataclass(frozen=True, slots=True)
class MeaningProposition:
    subject: MeaningArgument | None
    predicate: MeaningWord | None
    predicate_complement: MeaningArgument | None
    direct_object: MeaningArgument | None
    indirect_object: MeaningArgument | None
    complements: tuple[MeaningComplement, ...]
    negated: bool
    implicit_subject: str | None
    implicit_predicate: str | None
    source_fact: str
    confidence: float
    valency_profile: str
    valency_confidence: float


@dataclass(frozen=True, slots=True)
class MeaningPlanRejection:
    source_fact: str
    clause_index: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiscourseLink:
    from_index: int
    to_index: int
    relation: str
    reference_strategy: str
    subject_surface: str | None
    connector: str | None
    shared_lemmas: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResponseMove:
    order: int
    kind: str
    proposition_index: int | None
    relation: str | None
    optional: bool


@dataclass(frozen=True, slots=True)
class ResolvedSenseSlot:
    token_index: int
    lemma: str
    meaning: str
    source: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ResponseMeaningPlan:
    speech_act: str
    objective: str
    propositions: tuple[MeaningProposition, ...]
    discourse_links: tuple[DiscourseLink, ...]
    response_moves: tuple[ResponseMove, ...]
    rejections: tuple[MeaningPlanRejection, ...]
    prompt_structure: StructuredUtterance
    selected_senses: tuple[ResolvedSenseSlot, ...]
    required_facts: tuple[str, ...]
    unresolved_sense_indices: tuple[int, ...]
    confidence: float
    topic_transition: str

    @property
    def renderable(self) -> bool:
        return bool(self.propositions) and all(
            proposition.predicate is not None or proposition.implicit_predicate == "быть"
            for proposition in self.propositions
        )


def build_response_meaning_plan(
    utterance: UtterancePlan,
    prompt_structure: StructuredUtterance,
    sense_resolution: UtteranceSenseResolution,
) -> ResponseMeaningPlan:
    """Create response propositions only from facts already grounded upstream."""

    resolved_prompt = apply_sense_resolution(prompt_structure, sense_resolution)
    facts = tuple(
        dict.fromkeys(
            (
                *((utterance.context.exact_answer,) if utterance.context.exact_answer else ()),
                *utterance.context.required_facts,
            )
        )
    )
    propositions: list[MeaningProposition] = []
    rejections: list[MeaningPlanRejection] = []
    proposition_signatures: set[tuple[object, ...]] = set()
    for fact in facts[:4]:
        structure = analyze_structured_utterance(fact)
        for frame in structure.clauses[:2]:
            proposition, rejection = _proposition_from_frame(structure, frame, fact)
            if rejection is not None:
                rejections.append(rejection)
            if proposition is not None:
                if (
                    proposition.subject is None
                    and proposition.implicit_subject == "previous_clause_subject"
                    and propositions
                    and propositions[-1].source_fact == fact
                    and propositions[-1].subject is not None
                ):
                    proposition = replace(
                        proposition,
                        subject=propositions[-1].subject,
                    )
                signature = _proposition_signature(proposition)
                if signature in proposition_signatures:
                    rejections.append(
                        MeaningPlanRejection(
                            fact,
                            frame.clause_index,
                            ("duplicate_proposition",),
                        )
                    )
                    continue
                proposition_signatures.add(signature)
                propositions.append(proposition)
    discourse_links = _plan_discourse(tuple(propositions))
    response_moves = _response_moves(
        tuple(propositions),
        discourse_links,
        has_unresolved_reference=bool(
            utterance.context.dialogue_state
            and utterance.context.dialogue_state.unresolved_references
        ),
    )
    selected_senses = tuple(
        ResolvedSenseSlot(
            decision.token_index,
            decision.chosen.lemma,
            decision.chosen.meaning,
            decision.chosen.source,
            decision.confidence,
        )
        for decision in sense_resolution.decisions
        if decision.chosen is not None
    )
    proposition_confidence = (
        sum(item.confidence for item in propositions) / len(propositions) if propositions else 0.0
    )
    confidence = min(
        1.0,
        proposition_confidence * 0.58
        + resolved_prompt.confidence * 0.22
        + utterance.thought.confidence * 0.20,
    )
    return ResponseMeaningPlan(
        speech_act=(
            "grounded_answer"
            if propositions and utterance.speech_act == "clarifying_question"
            else utterance.speech_act
        ),
        objective=utterance.thought.objective,
        propositions=tuple(propositions),
        discourse_links=discourse_links,
        response_moves=response_moves,
        rejections=tuple(rejections),
        prompt_structure=resolved_prompt,
        selected_senses=selected_senses,
        required_facts=tuple(dict.fromkeys(item.source_fact for item in propositions)),
        unresolved_sense_indices=sense_resolution.unresolved_indices,
        confidence=confidence,
        topic_transition=(
            utterance.context.dialogue_state.topic.transition
            if utterance.context.dialogue_state is not None
            else "unknown"
        ),
    )


def realize_response_meaning(
    plan: ResponseMeaningPlan, *, allow_rough: bool = False,
) -> str | None:
    """Render complete propositions; return no candidate for an incomplete frame."""

    if not plan.renderable:
        return None
    rendered: list[str] = []
    links_by_target = {item.to_index: item for item in plan.discourse_links}
    proposition_moves = tuple(
        item for item in plan.response_moves if item.proposition_index is not None
    )
    for move in proposition_moves[:3]:
        assert move.proposition_index is not None
        index = move.proposition_index
        proposition = plan.propositions[index]
        link = links_by_target.get(index)
        sentence = _realize_proposition(
            proposition,
            subject_override=link.subject_surface if link is not None else None,
        )
        if sentence is None:
            return None
        if link is not None and link.connector:
            sentence = f"{link.connector} {sentence[:1].lower()}{sentence[1:]}"
        rendered.append(sentence)
    text = " ".join(rendered)
    surface = evaluate_surface(text)
    if allow_rough and not (set(surface.hard_issues) - COMPOSED_SURFACE_SOFT_ISSUES):
        return text or None
    return text if not surface.hard_issues and surface.score >= 0.72 else None


def _proposition_from_frame(
    structure: StructuredUtterance,
    frame: ClauseFrame,
    source_fact: str,
) -> tuple[MeaningProposition | None, MeaningPlanRejection | None]:
    subject = _argument(structure.tokens, frame.subject_indices)
    predicate = (
        _word(structure.tokens[frame.predicate_index])
        if frame.predicate_index is not None
        else None
    )
    predicate_complement = _argument(
        structure.tokens,
        frame.predicate_complement_indices,
    )
    direct_object = _argument(structure.tokens, frame.direct_object_indices)
    indirect_object = _argument(structure.tokens, frame.indirect_object_indices)
    complements = tuple(
        item
        for complement in frame.complements
        if (
            item := _meaning_complement(
                structure.tokens,
                complement,
                predicate.lemma if predicate is not None else frame.implicit_predicate,
            )
        )
        is not None
    )
    if predicate is None and frame.implicit_predicate != "быть":
        return None, _rejection(source_fact, frame, "missing_predicate")
    if predicate is not None and subject is None and frame.implicit_subject is None:
        return None, _rejection(source_fact, frame, "missing_subject")
    if frame.implicit_predicate == "быть" and (subject is None or predicate_complement is None):
        return None, _rejection(source_fact, frame, "incomplete_nominal_predicate")

    present_roles = frozenset(
        role
        for role, argument in (
            ("subject", subject),
            ("predicate_complement", predicate_complement),
            ("direct_object", direct_object),
            ("indirect_object", indirect_object),
        )
        if argument is not None
    )
    if predicate is not None:
        valency = assess_predicate_valency(
            predicate.lemma,
            present_roles=present_roles,
            complement_roles=frozenset(item.role for item in complements),
            implicit_subject=frame.implicit_subject is not None,
            imperative=frame.clause_kind == "command",
        )
        if not valency.complete:
            return None, _rejection(
                source_fact,
                frame,
                *(f"missing_valency:{role}" for role in valency.missing_roles),
            )
        valency_profile = valency.profile_source
        valency_confidence = valency.confidence
    else:
        valency_profile = "implicit_copula"
        valency_confidence = 0.94

    return MeaningProposition(
        subject=subject,
        predicate=predicate,
        predicate_complement=predicate_complement,
        direct_object=direct_object,
        indirect_object=indirect_object,
        complements=complements,
        negated=frame.negated,
        implicit_subject=frame.implicit_subject,
        implicit_predicate=frame.implicit_predicate,
        source_fact=source_fact,
        confidence=min(1.0, frame.confidence * 0.72 + valency_confidence * 0.28),
        valency_profile=valency_profile,
        valency_confidence=valency_confidence,
    ), None


def _argument(
    tokens: tuple[TokenInterpretation, ...],
    indices: tuple[int, ...],
) -> MeaningArgument | None:
    if not indices:
        return None
    head_index = next(
        (
            index
            for index in reversed(indices)
            if tokens[index].part_of_speech in {"noun", "pronoun", "number"}
            or tokens[index].entity_kind is not None
        ),
        indices[-1],
    )
    modifier_indices = [
        index
        for index in indices
        if index != head_index and tokens[index].part_of_speech in {"adjective", "participle"}
    ]
    previous = head_index - 1
    while previous >= 0 and tokens[previous].part_of_speech in {
        "adjective",
        "participle",
    }:
        modifier_indices.append(previous)
        previous -= 1
    modifiers = tuple(_word(tokens[index]) for index in sorted(set(modifier_indices)))
    return MeaningArgument(_word(tokens[head_index]), modifiers)


def _meaning_complement(
    tokens: tuple[TokenInterpretation, ...],
    complement: ClauseComplement,
    predicate_lemma: str | None,
) -> MeaningComplement | None:
    argument = _argument(tokens, complement.object_indices)
    if argument is None:
        return None
    preposition = tokens[complement.preposition_index].surface.casefold()
    return MeaningComplement(
        role=(
            resolve_complement_role(predicate_lemma, preposition, complement.role)
            if predicate_lemma is not None
            else complement.role
        ),
        preposition=preposition,
        argument=argument,
    )


def _word(token: TokenInterpretation) -> MeaningWord:
    return MeaningWord(
        lemma=token.lemma,
        preferred_surface=token.corrected_form or token.surface,
        part_of_speech=token.part_of_speech,
        grammatical_features=token.grammatical_features,
        grammemes=token.grammemes,
        selected_meaning=token.selected_meaning,
        entity_kind=token.entity_kind,
        entity_id=token.entity_id,
    )


def _realize_proposition(
    proposition: MeaningProposition,
    *,
    subject_override: str | None = None,
) -> str | None:
    if proposition.implicit_predicate == "быть":
        if proposition.subject is None or proposition.predicate_complement is None:
            return None
        subject = subject_override or _render_argument(proposition.subject, case="nominative")
        complement = _render_argument(
            proposition.predicate_complement,
            case="nominative",
        )
        if not subject or not complement:
            return None
        parts = [subject, "—", complement]
        for item in proposition.complements:
            case = _feature(item.argument.head, "case")
            rendered = _render_argument(item.argument, case=case)
            if not rendered:
                return None
            parts.extend((item.preposition, rendered))
        return _finish(" ".join(parts))

    if proposition.predicate is None:
        return None
    parts: list[str] = []
    if subject_override is not None:
        parts.append(subject_override)
    elif proposition.subject is not None:
        subject = _render_argument(proposition.subject, case="nominative")
        if not subject:
            return None
        parts.append(subject)
    if proposition.negated:
        parts.append("не")
    predicate = _render_predicate(proposition.predicate)
    if not predicate:
        return None
    parts.append(predicate)
    if proposition.indirect_object is not None:
        indirect = _render_argument(
            proposition.indirect_object,
            case=governed_object_case(
                proposition.predicate.lemma,
                "indirect_object",
                "dative",
            ),
        )
        if not indirect:
            return None
        parts.append(indirect)
    if proposition.direct_object is not None:
        direct = _render_argument(
            proposition.direct_object,
            case=governed_object_case(
                proposition.predicate.lemma,
                "direct_object",
                "accusative",
            ),
        )
        if not direct:
            return None
        parts.append(direct)
    for complement in proposition.complements:
        case = _feature(complement.argument.head, "case")
        rendered = _render_argument(complement.argument, case=case)
        if not rendered:
            return None
        parts.extend((complement.preposition, rendered))
    return _finish(" ".join(parts))


def _render_predicate(word: MeaningWord) -> str:
    features = {
        name: value
        for name, value in word.grammatical_features
        if name in {"tense", "number", "gender", "person"}
    }
    return DEFAULT_MORPHOLOGY.inflect(word.lemma, **features) or word.preferred_surface


def _render_argument(argument: MeaningArgument, *, case: str | None) -> str:
    head_features = _inferred_head_features(argument.head)
    requested = {"case": case} if case else {}
    if head_features.get("number"):
        requested["number"] = head_features["number"]
    head = (
        argument.head.preferred_surface
        if argument.head.entity_kind is not None
        or (case is not None and head_features.get("case") == case)
        else DEFAULT_MORPHOLOGY.inflect(argument.head.lemma, **requested)
        or argument.head.preferred_surface
    )
    modifier_features = dict(requested)
    for name in ("number", "gender", "animacy"):
        if head_features.get(name):
            modifier_features[name] = head_features[name]
    modifiers = tuple(
        DEFAULT_MORPHOLOGY.inflect(item.lemma, **modifier_features) or item.preferred_surface
        for item in argument.modifiers
    )
    return " ".join((*modifiers, head)).strip()


def _inferred_head_features(head: MeaningWord) -> dict[str, str]:
    inferred: dict[str, str] = {}
    reading = DEFAULT_MORPHOLOGY.best(head.lemma)
    if reading is not None:
        inferred.update(reading.features())
    inferred.update(dict(head.grammatical_features))
    return inferred


def _feature(word: MeaningWord, name: str) -> str | None:
    return dict(word.grammatical_features).get(name)


def _finish(text: str) -> str:
    stripped = " ".join(text.split()).strip(" .!?…")
    if not stripped:
        return ""
    return stripped[:1].upper() + stripped[1:] + "."


def _rejection(
    source_fact: str,
    frame: ClauseFrame,
    *reasons: str,
) -> MeaningPlanRejection:
    return MeaningPlanRejection(source_fact, frame.clause_index, tuple(reasons))


def _proposition_signature(proposition: MeaningProposition) -> tuple[object, ...]:
    return (
        _argument_key(proposition.subject),
        proposition.predicate.lemma if proposition.predicate else proposition.implicit_predicate,
        _argument_key(proposition.predicate_complement),
        _argument_key(proposition.direct_object),
        _argument_key(proposition.indirect_object),
        tuple(
            (item.role, item.preposition, _argument_key(item.argument))
            for item in proposition.complements
        ),
        proposition.negated,
    )


def _argument_key(argument: MeaningArgument | None) -> tuple[object, ...] | None:
    if argument is None:
        return None
    entity_key = (
        (argument.head.entity_kind, argument.head.entity_id)
        if argument.head.entity_id is not None
        else None
    )
    return (
        entity_key or argument.head.lemma,
        tuple(item.lemma for item in argument.modifiers),
    )


def _plan_discourse(
    propositions: tuple[MeaningProposition, ...],
) -> tuple[DiscourseLink, ...]:
    links: list[DiscourseLink] = []
    for index in range(1, len(propositions)):
        previous = propositions[index - 1]
        current = propositions[index]
        shared = tuple(sorted(_proposition_lemmas(previous) & _proposition_lemmas(current)))
        same_subject = _same_argument(previous.subject, current.subject)
        subject_surface = _subject_pronoun(current.subject) if same_subject else None
        relation, connector = _explicit_discourse_relation(current.source_fact)
        if relation is None:
            relation = "elaboration" if same_subject else "continuation" if shared else "parallel"
        links.append(
            DiscourseLink(
                from_index=index - 1,
                to_index=index,
                relation=relation,
                reference_strategy="pronoun" if subject_surface else "explicit",
                subject_surface=subject_surface,
                connector=connector,
                shared_lemmas=shared,
            )
        )
    return tuple(links)


def _same_argument(
    left: MeaningArgument | None,
    right: MeaningArgument | None,
) -> bool:
    if left is None or right is None:
        return False
    if left.head.entity_id is not None or right.head.entity_id is not None:
        return (
            left.head.entity_kind,
            left.head.entity_id,
        ) == (
            right.head.entity_kind,
            right.head.entity_id,
        )
    return left.head.lemma == right.head.lemma


def _subject_pronoun(argument: MeaningArgument | None) -> str | None:
    if argument is None or argument.head.part_of_speech == "pronoun":
        return None
    features = dict(argument.head.grammatical_features)
    if features.get("number") == "plural":
        return "они"
    return {
        "masculine": "он",
        "feminine": "она",
        "neuter": "оно",
    }.get(features.get("gender"))


def _proposition_lemmas(proposition: MeaningProposition) -> set[str]:
    lemmas: set[str] = set()
    for argument in (
        proposition.subject,
        proposition.predicate_complement,
        proposition.direct_object,
        proposition.indirect_object,
    ):
        if argument is not None:
            lemmas.add(argument.head.lemma)
            lemmas.update(item.lemma for item in argument.modifiers)
    if proposition.predicate is not None:
        lemmas.add(proposition.predicate.lemma)
    for complement in proposition.complements:
        lemmas.add(complement.argument.head.lemma)
    return lemmas


def _explicit_discourse_relation(source_fact: str) -> tuple[str | None, str | None]:
    normalized = source_fact.lstrip().casefold()
    if normalized.startswith(("но ", "однако ", "зато ")):
        return "contrast", "Однако"
    if normalized.startswith(("поэтому ", "следовательно ")):
        return "consequence", "Поэтому"
    return None, None


def _response_moves(
    propositions: tuple[MeaningProposition, ...],
    links: tuple[DiscourseLink, ...],
    *,
    has_unresolved_reference: bool,
) -> tuple[ResponseMove, ...]:
    moves: list[ResponseMove] = []
    if propositions:
        moves.append(ResponseMove(0, "main_claim", 0, None, False))
    links_by_target = {item.to_index: item for item in links}
    for index in range(1, len(propositions)):
        link = links_by_target.get(index)
        relation = link.relation if link is not None else "continuation"
        kind = {
            "elaboration": "supporting_detail",
            "contrast": "qualification",
            "consequence": "consequence",
        }.get(relation, "supporting_detail")
        moves.append(ResponseMove(len(moves), kind, index, relation, True))
    if has_unresolved_reference:
        moves.append(
            ResponseMove(
                len(moves),
                "reference_clarification",
                None,
                None,
                bool(propositions),
            )
        )
    return tuple(moves)
