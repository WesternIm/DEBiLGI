"""Choose several contextual dialogue moves before productive realization.

The planner stores neither prompt/reply pairs nor complete reply strings.  It
turns the current dialogue state into competing semantic propositions.  The
response arbitrator can therefore choose between moves instead of receiving a
single phrase that was already decided by an act label.
"""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.language.clarification_policy import is_contact_request
from debilgi.language.dialogue_acts import DialogueActFrame
from debilgi.language.dialogue_obligations import DialogueExchange, resolve_dialogue_exchange
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.language.self_state_reasoning import SelfStateFrame
from debilgi.language.social_reasoning import CommunicativeIntention
from debilgi.language.structured_utterance import StructuredUtterance, TokenInterpretation
from debilgi.personality.expression import PersonalityExpression
from debilgi.personality.perspective import (
    PersonalityPerspective,
    PersonalOpinion,
    form_personal_opinion,
)
from debilgi.social.models import SocialReaction

_POSITIVE_CONFIRMATION = frozenset({"да", "же", "ведь", "верно", "правда"})
_CHALLENGE_MARKERS = frozenset({"чё", "че", "что-ли", "что", "ли"})
_CALENDAR_NOUNS = frozenset(
    {
        "день",
        "неделя",
        "выходной",
        "будни",
        "понедельник",
        "вторник",
        "среда",
        "четверг",
        "пятница",
        "суббота",
        "воскресенье",
        "январь",
        "февраль",
        "март",
        "апрель",
        "май",
        "июнь",
        "июль",
        "август",
        "сентябрь",
        "октябрь",
        "ноябрь",
        "декабрь",
    }
)
_TEMPORAL_DEICTICS = frozenset(
    {"сегодня", "завтра", "послезавтра", "вчера", "позавчера"}
)


@dataclass(frozen=True, slots=True)
class DialogueMovePlan:
    """One independently scored semantic option for the current turn."""

    move: str
    strategy: str
    thought: ClauseThought
    confidence: float
    stance: str
    evidence: tuple[str, ...]
    key_points: tuple[str, ...] = ()
    semantic_origin: str = "scaffold"


def plan_dialogue_moves(
    frame: DialogueActFrame | None,
    *,
    self_state: SelfStateFrame | None = None,
    intention: CommunicativeIntention | None = None,
    personality: PersonalityExpression | None = None,
    perspective: PersonalityPerspective | None = None,
    reaction: SocialReaction | None = None,
    structure: StructuredUtterance | None = None,
    prompt: str = "",
    context_messages: tuple[str, ...] = (),
    exchange: DialogueExchange | None = None,
    resolve_from_context: bool = True,
    observed_forms: tuple = (),
    reaction_act: str | None = None,
    allow_hook_question: bool = True,
    alignment: float | None = None,
) -> tuple[DialogueMovePlan, ...]:
    """Build competing moves from state, stance and conversational context."""

    if frame is None:
        return ()
    if reaction_act is not None:
        return _fragment_moves(frame, personality=personality, reaction=reaction, prompt=prompt,
                               observed_forms=observed_forms, reaction_act=reaction_act,
                               allow_hook_question=allow_hook_question, alignment=alignment)
    if is_contact_request(prompt):
        return (DialogueMovePlan(
            move="engage", strategy="answer_contact",
            thought=_reply_thought(0.95, ("current_message:contact_request",)),
            confidence=0.95, stance="engage", evidence=("current_message:contact_request",),
            semantic_origin="derived",
        ),)
    if exchange is None and resolve_from_context:
        exchange = resolve_dialogue_exchange(prompt, context_messages)
    if exchange is not None and exchange.bound_proposition is not None:
        thought = exchange.bound_proposition
        return (DialogueMovePlan(
            move="verify_understanding", strategy="resolve_dialogue_obligation",
            thought=thought, confidence=thought.confidence, stance="curious",
            evidence=thought.evidence,
            key_points=tuple(dep.argument.head.lemma for dep in thought.dependencies),
            semantic_origin="derived",
        ),)
    plans: list[DialogueMovePlan] = []
    if self_state is not None and (
        self_state.state == "self_role_assessment"
        or (self_state.state == "self_assessment" and self_state.clause_kind == "question")
    ):
        plans.extend(
            _self_role_moves(
                frame,
                self_state,
                personality=personality,
                perspective=perspective,
                reaction=reaction,
                prompt=prompt,
                context_messages=context_messages,
            )
        )
    elif (
        self_state is not None
        and self_state.state == "self_reflection"
        and frame.response_obligation == "answer_question"
    ):
        if structure is not None and any(
            structure.tokens[complement.preposition_index].lemma in {"о", "об", "про"}
            and any(structure.tokens[index].part_of_speech == "noun"
                    for index in complement.object_indices)
            for clause in structure.clauses for complement in clause.complements
        ):
            # "What do you think about scissors?" requires a topical appraisal.
            # Describing an active thinking process does not fill that slot.
            return ()
        plans.extend(
            _self_reflection_moves(
                frame,
                self_state,
                perspective=perspective,
            )
        )
    elif not frame.complete or frame.response_obligation == "clarify_fragment":
        relational_plans = _nominal_relation_moves(
            frame,
            structure=structure,
            perspective=perspective,
        )
        plans.extend(
            relational_plans
            or _fragment_moves(
                frame,
                personality=personality,
                reaction=reaction,
                prompt=prompt,
            )
        )
    else:
        content_moves = (
            _statement_content_moves(
                frame,
                structure=structure,
                perspective=perspective,
                reaction=reaction,
            )
            if frame.response_obligation == "continue_topic"
            else ()
        )
        plans.extend(
            content_moves
            or _social_moves(
                frame,
                intention=intention,
                reaction=reaction,
                perspective=perspective,
            )
        )
    return tuple(
        sorted(
            plans,
            key=lambda item: (-item.confidence, item.strategy, item.move),
        )
    )


def _self_reflection_moves(
    frame: DialogueActFrame,
    self_state: SelfStateFrame,
    *,
    perspective: PersonalityPerspective | None,
) -> tuple[DialogueMovePlan, ...]:
    """Answer an underspecified question about thinking from the active process."""

    coherence = perspective.coherence if perspective is not None else 0.5
    exploration = perspective.exploration if perspective is not None else 0.5
    confidence = min(
        0.92,
        frame.confidence * 0.48
        + self_state.confidence * 0.24
        + coherence * 0.14
        + exploration * 0.08
        + 0.10,
    )
    evidence = (
        "self_reflection_question",
        "active_context_analysis",
        f"personality_coherence:{coherence:.3f}",
        f"personality_exploration:{exploration:.3f}",
    )
    return (
        DialogueMovePlan(
            move="describe_current_thinking",
            strategy="ground_self_reflection_in_active_process",
            thought=ClauseThought(
                predicate=ConceptLexeme(
                    "анализировать",
                    "анализирую",
                    "verb",
                    (
                        ("tense", "present"),
                        ("number", "singular"),
                        ("person", "first"),
                    ),
                ),
                dependencies=(
                    _subject(),
                    SemanticDependency(
                        "direct_object",
                        ConceptArgument(
                            ConceptLexeme(
                                "контекст",
                                "контекст",
                                "noun",
                                (
                                    ("case", "accusative"),
                                    ("number", "singular"),
                                    ("gender", "masculine"),
                                ),
                            ),
                            grammatical_case="accusative",
                        ),
                    ),
                ),
                confidence=confidence,
                evidence=evidence,
            ),
            confidence=confidence,
            stance="direct",
            evidence=evidence,
            key_points=("контекст",),
            semantic_origin="derived",
        ),
    )


def _self_role_moves(
    frame: DialogueActFrame,
    self_state: SelfStateFrame,
    *,
    personality: PersonalityExpression | None,
    perspective: PersonalityPerspective | None,
    reaction: SocialReaction | None,
    prompt: str,
    context_messages: tuple[str, ...],
) -> tuple[DialogueMovePlan, ...]:
    target = self_state.target
    if target is None:
        return ()
    opinion = _self_role_opinion(
        target.head.lemma,
        personality=personality,
        perspective=perspective,
        reaction=reaction,
        prompt=prompt,
        context_messages=context_messages,
    )
    stance_score = opinion.score
    evidence = opinion.evidence
    if opinion.stance == "support":
        stance = "support"
        negated = False
        polarity: bool | None = True
    elif opinion.stance == "oppose":
        stance = "oppose"
        negated = True
        polarity = False
    else:
        stance = "inspect"
        negated = False
        polarity = None

    plans: list[DialogueMovePlan] = []
    if stance != "inspect":
        plans.append(
            DialogueMovePlan(
                move="state_position",
                strategy="contextual_self_assessment",
                thought=_self_assessment_thought(
                    target,
                    negated=negated,
                    polarity=polarity,
                    confidence=frame.confidence,
                    evidence=evidence,
                ),
                confidence=min(0.96, frame.confidence * 0.72 + 0.20 + abs(stance_score) * 0.12),
                stance=stance,
                evidence=evidence,
                key_points=(target.head.lemma,),
                semantic_origin="derived",
            )
        )
        plans.append(
            DialogueMovePlan(
                move="accept_or_reject_evaluation",
                strategy="evaluation_stance",
                thought=_evaluation_thought(
                    negated=negated,
                    polarity=polarity,
                    confidence=frame.confidence * 0.96,
                    evidence=evidence,
                ),
                confidence=min(0.90, frame.confidence * 0.70 + 0.17 + abs(stance_score) * 0.08),
                stance=stance,
                evidence=evidence,
                key_points=(target.head.lemma, "оценка"),
                semantic_origin="derived",
            )
        )
    else:
        plans.extend(
            (
                DialogueMovePlan(
                    move="inspect_self_position",
                    strategy="withhold_unsupported_label",
                    thought=_self_assessment_thought(
                        target,
                        negated=True,
                        polarity=False,
                        confidence=frame.confidence,
                        evidence=evidence,
                        qualified=True,
                    ),
                    confidence=min(0.88, frame.confidence * 0.76 + 0.16),
                    stance=stance,
                    evidence=evidence,
                    key_points=(target.head.lemma, "оценка"),
                    semantic_origin="derived",
                ),
                DialogueMovePlan(
                    move="request_basis",
                    strategy="ask_for_evidence",
                    thought=_basis_question(
                        confidence=frame.confidence * 0.94,
                        evidence=evidence,
                    ),
                    confidence=min(0.84, frame.confidence * 0.72 + 0.16),
                    stance=stance,
                    evidence=evidence,
                    key_points=(target.head.lemma,),
                    semantic_origin="derived",
                ),
            )
        )
    return tuple(plans)


def _fragment_moves(
    frame: DialogueActFrame,
    *,
    personality: PersonalityExpression | None,
    reaction: SocialReaction | None,
    prompt: str,
    observed_forms: tuple = (),
    reaction_act: str | None = None,
    allow_hook_question: bool = True,
    alignment: float | None = None,
) -> tuple[DialogueMovePlan, ...]:
    if reaction_act is not None:
        cores = [f for f in observed_forms if f.component == "core" and (
            f.clause_kind != "question" or allow_hook_question)]
        # Input features have already propagated through the learned exchange
        # graph. Bind the winning meaning without a fixed act priority table.
        winner = max(cores, key=lambda f: (f.activation, f.relevance, f.support,
                                           f.rule_id or ""), default=None)
        meanings = [winner.meaning] if winner is not None else []
        confidence = winner.activation if winner and winner.rule_id else 1.0
        return tuple(DialogueMovePlan(
            move=m, strategy="observed_exchange", thought=ClauseThought(
                ConceptLexeme(m, m, "dialogue_act"), (), clause_kind="fragment",
                predicate_class="dialogue_act", confidence=confidence,
                evidence=tuple(dict.fromkeys(e for f in observed_forms if f.meaning == m
                                             for e in f.evidence_ids)),
            ), confidence=confidence, stance="follow_event_valence", evidence=(),
            semantic_origin="derived",
        ) for m in meanings[:1])
    target = frame.objects[0] if frame.objects else ""
    arousal = reaction.vector.arousal if reaction is not None else 0.0
    curiosity = personality.curiosity if personality is not None else 0.5
    expressive_fragment = bool(
        target
        and (
            target.endswith(("ть", "ться"))
            or frame.register in {"colloquial", "expressive"}
        )
    )
    event_confidence = min(
        0.95,
        frame.confidence * 0.62
        + 0.20
        + (0.09 if expressive_fragment else 0.0)
        + arousal * 0.06,
    )
    meaning_confidence = min(
        0.90,
        frame.confidence * 0.60 + 0.13 + curiosity * 0.06,
    )
    evidence = tuple(
        filter(
            None,
            (
                "incomplete_dialogue_act",
                "expressive_fragment" if expressive_fragment else "",
                f"reaction_arousal:{arousal:.3f}",
                f"personality_curiosity:{curiosity:.3f}",
            ),
        )
    )
    plans = [
        DialogueMovePlan(
            move="probe_event",
            strategy="ask_what_happened",
            thought=_event_question("произойти", "произошло", event_confidence, evidence),
            confidence=event_confidence,
            stance="curious",
            evidence=evidence,
        ),
        DialogueMovePlan(
            move="probe_reaction_source",
            strategy="ask_what_triggered_reaction",
            thought=_reaction_source_question(
                min(0.91, event_confidence * 0.97),
                evidence,
            ),
            confidence=min(0.91, event_confidence * 0.97),
            stance="engage",
            evidence=evidence,
        ),
    ]
    # A bare expressive infinitive is normally a reaction in Russian, not a
    # request to define the word.  A semantic-scope probe remains available
    # for genuinely ambiguous non-expressive fragments.
    if target and target not in {"это", "этот", "тот", "такой"} and not expressive_fragment:
        plans.append(
            DialogueMovePlan(
                move="clarify_intended_meaning",
                strategy="ask_semantic_scope",
                thought=_meaning_question(target, meaning_confidence, evidence),
                confidence=meaning_confidence,
                stance="curious",
                evidence=evidence,
                key_points=(target,),
            )
        )
    return tuple(plans)


def _nominal_relation_moves(
    frame: DialogueActFrame,
    *,
    structure: StructuredUtterance | None,
    perspective: PersonalityPerspective | None,
) -> tuple[DialogueMovePlan, ...]:
    """Digest a known nominal relation instead of asking for a definition."""

    if structure is None or len(structure.clauses) != 1:
        return ()
    clause = structure.clauses[0]
    if (
        clause.predicate_index is not None
        or len(clause.subject_indices) != 1
        or not clause.complements
    ):
        return ()
    complement = next(
        (
            item
            for item in clause.complements
            if item.object_indices
            and item.role in {"location", "destination", "location_or_destination"}
        ),
        None,
    )
    if complement is None:
        return ()
    subject_token = structure.tokens[clause.subject_indices[0]]
    object_token = structure.tokens[complement.object_indices[0]]
    preposition = structure.tokens[complement.preposition_index].lemma
    subject = _token_argument(subject_token, grammatical_case="nominative")
    direct_object = _token_argument(subject_token, grammatical_case="accusative")
    location = _token_argument(
        object_token,
        grammatical_case=(
            dict(object_token.grammatical_features).get("case") or "prepositional"
        ),
        preposition=preposition,
    )
    playfulness = perspective.playfulness if perspective is not None else 0.5
    curiosity = perspective.curiosity if perspective is not None else 0.5
    evidence = (
        "known_nominal_relation",
        f"relation:{complement.role}",
        f"preposition:{preposition}",
        f"personality_playfulness:{playfulness:.3f}",
        f"personality_curiosity:{curiosity:.3f}",
    )
    integrate_confidence = min(
        0.93,
        frame.confidence * 0.56 + structure.confidence * 0.20 + playfulness * 0.14 + 0.10,
    )
    inspect_confidence = min(
        0.90,
        frame.confidence * 0.52 + structure.confidence * 0.20 + curiosity * 0.14 + 0.08,
    )
    return (
        DialogueMovePlan(
            move="integrate_scene",
            strategy="integrate_nominal_relation",
            thought=ClauseThought(
                predicate=_verb("представлять", "представляю"),
                dependencies=(
                    _subject(),
                    SemanticDependency("direct_object", direct_object),
                    SemanticDependency("location", location),
                ),
                confidence=integrate_confidence,
                evidence=evidence,
            ),
            confidence=integrate_confidence,
            stance="engage",
            evidence=evidence,
            key_points=(subject_token.lemma, object_token.lemma),
            semantic_origin="derived",
        ),
        DialogueMovePlan(
            move="inspect_relation",
            strategy="verify_nominal_relation",
            thought=ClauseThought(
                predicate=ConceptLexeme(
                    "находиться",
                    "находится",
                    "verb",
                    (
                        ("tense", "present"),
                        ("number", "singular"),
                        ("person", "third"),
                    ),
                ),
                dependencies=(
                    SemanticDependency("subject", subject),
                    SemanticDependency("location", location),
                ),
                clause_kind="question",
                focus_role="location",
                confidence=inspect_confidence,
                evidence=evidence,
            ),
            confidence=inspect_confidence,
            stance="curious",
            evidence=evidence,
            key_points=(subject_token.lemma, object_token.lemma),
            semantic_origin="derived",
        ),
    )


def _social_moves(
    frame: DialogueActFrame,
    *,
    intention: CommunicativeIntention | None,
    reaction: SocialReaction | None,
    perspective: PersonalityPerspective | None,
) -> tuple[DialogueMovePlan, ...]:
    obligation = frame.response_obligation
    confidence = frame.confidence
    evidence = (
        f"dialogue_obligation:{obligation}",
        *(
            (f"social_move:{intention.primary_move}",)
            if intention is not None
            else ()
        ),
    )
    specifications: dict[str, tuple[tuple[str, str, ClauseThought], ...]] = {
        "acknowledge_attention": (
            ("engage", "active_listening", _listen_thought(confidence, evidence)),
            ("engage", "hold_attention", _attention_thought(confidence * 0.96, evidence)),
        ),
        "acknowledge_understanding": (
            (
                "acknowledge",
                "register_understanding",
                _understanding_thought("понять", "понял", confidence, evidence),
            ),
            (
                "acknowledge",
                "register_signal",
                _understanding_thought(
                    "уловить", "уловил", confidence * 0.97, evidence
                ),
            ),
        ),
        "acknowledge_evaluation": (
            (
                "acknowledge",
                "receive_evaluation",
                _evaluation_thought(False, True, confidence, evidence),
            ),
            ("reciprocate", "notice_support", _support_thought(confidence * 0.96, evidence)),
        ),
        "address_negative_evaluation": (
            ("targeted_repair", "locate_failure", _failure_question(confidence, evidence)),
            (
                "ask_specific_critique",
                "request_failed_part",
                _failed_part_question(confidence * 0.97, evidence),
            ),
        ),
        "repair_previous_answer": (
            ("targeted_repair", "reconsider_answer", _reconsider_thought(confidence, evidence)),
            (
                "clarify_reference",
                "recheck_question",
                _recheck_thought(confidence * 0.97, evidence),
            ),
        ),
        "continue_topic": (
            ("continue_topic", "continue_thought", _continue_thought(confidence, evidence)),
            ("acknowledge", "retain_topic", _retain_topic_thought(confidence * 0.96, evidence)),
        ),
        "acknowledge_reference": (
            ("acknowledge", "retain_reference", _retain_topic_thought(confidence, evidence)),
            ("continue_topic", "continue_thought", _continue_thought(confidence * 0.96, evidence)),
        ),
    }
    plans: list[DialogueMovePlan] = []
    derived_obligation = obligation in {
        "acknowledge_attention",
        "acknowledge_understanding",
        "acknowledge_evaluation",
        "repair_previous_answer",
        "repair_understanding",
    }
    for index, (move, strategy, thought) in enumerate(specifications.get(obligation, ())):
        reaction_fit = 0.0
        if reaction is not None:
            reaction_fit = (
                reaction.vector.curiosity * 0.03
                if move in {"engage", "ask_specific_critique", "continue_topic"}
                else reaction.vector.trust * 0.02
            )
        personality_fit = perspective.candidate_fit(move) if perspective is not None else 0.5
        plans.append(
            DialogueMovePlan(
                move=move,
                strategy=strategy,
                thought=thought,
                confidence=min(
                    0.95,
                    confidence * 0.68
                    + 0.10
                    - index * 0.025
                    + reaction_fit
                    + personality_fit * 0.12,
                ),
                stance=intention.stance_mode if intention is not None else "engage",
                evidence=evidence,
                key_points=frame.objects[:2],
                semantic_origin="derived" if derived_obligation else "scaffold",
            )
        )
    return tuple(plans)


def _statement_content_moves(
    frame: DialogueActFrame,
    *,
    structure: StructuredUtterance | None,
    perspective: PersonalityPerspective | None,
    reaction: SocialReaction | None,
) -> tuple[DialogueMovePlan, ...]:
    """Continue an assertion through its content, never through dialogue meta-talk."""

    if structure is None:
        return ()
    target = _salient_statement_target(structure)
    if target is None:
        return ()
    curiosity = perspective.curiosity if perspective is not None else 0.5
    playfulness = perspective.playfulness if perspective is not None else 0.5
    social_curiosity = (
        reaction.vector.curiosity * reaction.confidence
        if reaction is not None
        else 0.0
    )
    evidence = (
        "complete_assertion",
        f"assertion_predicate:{frame.predicate or 'implicit'}",
        f"content_target:{target.lemma}",
        f"personality_curiosity:{curiosity:.3f}",
        f"personality_playfulness:{playfulness:.3f}",
    )
    inspect_confidence = min(
        0.93,
        frame.confidence * 0.62
        + curiosity * 0.16
        + social_curiosity * 0.08
        + 0.15,
    )
    plans = [
        DialogueMovePlan(
            move="explore_assertion",
            strategy="ask_owned_topic_view",
            thought=_topic_view_question(
                target,
                confidence=inspect_confidence,
                evidence=evidence,
            ),
            confidence=inspect_confidence,
            stance="curious",
            evidence=evidence,
            key_points=(target.lemma,),
        )
    ]
    if _is_temporal_target(target, structure):
        planning_confidence = min(
            0.95,
            frame.confidence * 0.64
            + curiosity * 0.14
            + playfulness * 0.06
            + 0.17,
        )
        plans.append(
            DialogueMovePlan(
                move="connect_assertion_to_personal_future",
                strategy="ask_temporal_intention",
                thought=_temporal_intention_question(
                    target,
                    confidence=planning_confidence,
                    evidence=(*evidence, "calendar_semantic_class"),
                ),
                confidence=planning_confidence,
                stance="engage",
                evidence=(*evidence, "calendar_semantic_class"),
                key_points=(target.lemma,),
            )
        )
    return tuple(plans)


def _salient_statement_target(
    structure: StructuredUtterance,
) -> TokenInterpretation | None:
    nominal_priority: list[int] = []
    descriptive_fallback: list[int] = []
    for clause in structure.clauses:
        nominal_priority.extend(reversed(clause.direct_object_indices))
        nominal_priority.extend(
            index
            for complement in clause.complements
            for index in reversed(complement.object_indices)
        )
        if clause.implicit_predicate == "быть" and clause.subject_indices:
            # In a copular assertion the subject is the entity under
            # discussion; the complement only classifies or evaluates it.
            nominal_priority.extend(reversed(clause.subject_indices))
            nominal_priority.extend(reversed(clause.predicate_complement_indices))
        else:
            nominal_priority.extend(reversed(clause.predicate_complement_indices))
            nominal_priority.extend(reversed(clause.subject_indices))
        descriptive_fallback.extend(reversed(clause.predicate_complement_indices))
    indices = tuple(dict.fromkeys((*nominal_priority, *descriptive_fallback)))
    return next(
        (
            structure.tokens[index]
            for index in indices
            if structure.tokens[index].part_of_speech
            in {"noun", "pronoun", "number"}
            and structure.tokens[index].lemma not in {"я", "ты", "вы", "мы", "это"}
        ),
        next(
            (
                structure.tokens[index]
                for index in indices
                if structure.tokens[index].part_of_speech in {"adjective", "participle"}
            ),
            None,
        ),
    )


def _is_temporal_target(
    target: TokenInterpretation,
    structure: StructuredUtterance,
) -> bool:
    return target.lemma in _CALENDAR_NOUNS or any(
        token.lemma in _TEMPORAL_DEICTICS for token in structure.tokens
    )


def _topic_view_question(
    target: TokenInterpretation,
    *,
    confidence: float,
    evidence: tuple[str, ...],
) -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme(
            "думать",
            "думаешь",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "second")),
        ),
        dependencies=(
            _subject("ты", "ты"),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("что", "что", "pronoun"),
                    grammatical_case="accusative",
                ),
            ),
            SemanticDependency(
                "topic",
                _token_argument(
                    target,
                    grammatical_case="prepositional",
                    preposition="о",
                ),
            ),
        ),
        clause_kind="question",
        focus_role="direct_object",
        confidence=confidence,
        evidence=evidence,
    )


def _temporal_intention_question(
    target: TokenInterpretation,
    *,
    confidence: float,
    evidence: tuple[str, ...],
) -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme(
            "планировать",
            "планируешь",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "second")),
        ),
        dependencies=(
            _subject("ты", "ты"),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("что", "что", "pronoun"),
                    grammatical_case="accusative",
                ),
            ),
            SemanticDependency(
                "topic",
                _token_argument(
                    target,
                    grammatical_case="accusative",
                    preposition="на",
                ),
            ),
        ),
        clause_kind="question",
        focus_role="direct_object",
        confidence=confidence,
        evidence=evidence,
    )


def _token_argument(
    token: TokenInterpretation,
    *,
    grammatical_case: str,
    preposition: str | None = None,
) -> ConceptArgument:
    return ConceptArgument(
        ConceptLexeme(
            token.lemma,
            token.corrected_form or token.surface,
            token.part_of_speech,
            token.grammatical_features,
            token.entity_kind,
        ),
        preposition=preposition,
        grammatical_case=grammatical_case,
    )


def _self_role_opinion(
    target: str,
    *,
    personality: PersonalityExpression | None,
    perspective: PersonalityPerspective | None,
    reaction: SocialReaction | None,
    prompt: str,
    context_messages: tuple[str, ...],
) -> PersonalOpinion:
    # Conversational approval and repeated mentions are not evidence that a
    # property applies to the speaker. Only a semantic value appraisal may
    # support an owned opinion; absent a link, preserve uncertainty.
    return form_personal_opinion("self", target, perspective=perspective)


def _subject(lemma: str = "я", surface: str | None = None) -> SemanticDependency:
    return SemanticDependency(
        "subject",
        ConceptArgument(
            ConceptLexeme(
                lemma,
                surface or lemma,
                "pronoun",
                (("number", "singular"), ("person", "first" if lemma == "я" else "second")),
            )
        ),
    )


def _noun(
    lemma: str,
    *,
    surface: str | None = None,
    case: str | None = None,
    modifiers: tuple[ConceptLexeme, ...] = (),
    preposition: str | None = None,
    entity_kind: str | None = None,
) -> ConceptArgument:
    return ConceptArgument(
        ConceptLexeme(lemma, surface or lemma, "noun", entity_kind=entity_kind),
        modifiers,
        preposition,
        case,
    )


def _verb(lemma: str, surface: str, *, past: bool = False) -> ConceptLexeme:
    features = (
        (("tense", "past"), ("number", "singular"), ("gender", "masculine"))
        if past
        else (("tense", "present"), ("number", "singular"), ("person", "first"))
    )
    return ConceptLexeme(lemma, surface, "verb", features)


def _self_assessment_thought(
    target: ConceptArgument,
    *,
    negated: bool,
    polarity: bool | None,
    confidence: float,
    evidence: tuple[str, ...],
    qualified: bool = False,
) -> ClauseThought:
    dependencies: list[SemanticDependency] = [
        _subject(),
        SemanticDependency(
            "direct_object",
            ConceptArgument(
                ConceptLexeme(
                    "себя",
                    "себя",
                    "pronoun",
                    (("case", "accusative"),),
                ),
                grammatical_case="accusative",
            ),
        ),
        SemanticDependency(
            "predicate_complement",
            ConceptArgument(
                target.head,
                target.modifiers,
                target.preposition,
                "instrumental",
            ),
        ),
    ]
    if qualified:
        dependencies.append(
            SemanticDependency(
                "manner",
                ConceptArgument(ConceptLexeme("пока", "пока", "adverb")),
            )
        )
    return ClauseThought(
        predicate=_verb("считать", "считаю"),
        dependencies=tuple(dependencies),
        negated=negated,
        response_polarity=polarity,
        confidence=confidence,
        evidence=evidence,
    )


def _evaluation_thought(
    negated: bool,
    polarity: bool | None,
    confidence: float,
    evidence: tuple[str, ...],
) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("принимать", "принимаю"),
        dependencies=(
            _subject(),
            SemanticDependency(
                "direct_object",
                _noun(
                    "оценка",
                    case="accusative",
                    modifiers=(ConceptLexeme("этот", "эту", "adjective"),),
                ),
            ),
        ),
        negated=negated,
        response_polarity=polarity,
        confidence=confidence,
        evidence=evidence,
    )


def _basis_question(confidence: float, evidence: tuple[str, ...]) -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme(
            "подтверждать",
            "подтверждает",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "third")),
        ),
        dependencies=(
            _subject("что", "что"),
            SemanticDependency(
                "direct_object",
                _noun(
                    "оценка",
                    case="accusative",
                    modifiers=(ConceptLexeme("такой", "такую", "adjective"),),
                ),
            ),
        ),
        clause_kind="question",
        focus_role="subject",
        confidence=confidence,
        evidence=evidence,
    )


def _event_question(
    lemma: str,
    surface: str,
    confidence: float,
    evidence: tuple[str, ...],
) -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme(
            lemma,
            surface,
            "verb",
            (("tense", "past"), ("number", "singular"), ("gender", "neuter")),
        ),
        dependencies=(_subject("что", "что"),),
        clause_kind="question",
        focus_role="subject",
        confidence=confidence,
        evidence=evidence,
    )


def _reaction_source_question(
    confidence: float,
    evidence: tuple[str, ...],
) -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme(
            "вызвать",
            "вызвало",
            "verb",
            (("tense", "past"), ("number", "singular"), ("gender", "neuter")),
        ),
        dependencies=(
            _subject("что", "что"),
            SemanticDependency(
                "direct_object",
                _noun(
                    "реакция",
                    case="accusative",
                    modifiers=(ConceptLexeme("такой", "такую", "adjective"),),
                ),
            ),
        ),
        clause_kind="question",
        focus_role="subject",
        confidence=confidence,
        evidence=evidence,
    )


def _meaning_question(
    target: str,
    confidence: float,
    evidence: tuple[str, ...],
) -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme(
            "подразумевать",
            "подразумеваешь",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "second")),
        ),
        dependencies=(
            _subject("ты", "ты"),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("что", "что", "pronoun"),
                    grammatical_case="accusative",
                ),
            ),
            SemanticDependency(
                "topic",
                _noun(
                    target,
                    case="instrumental",
                    preposition="под",
                    entity_kind="quoted_lexeme",
                ),
            ),
        ),
        clause_kind="question",
        focus_role="direct_object",
        confidence=confidence,
        evidence=evidence,
    )


def _listen_thought(confidence: float, evidence: tuple[str, ...]) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("слушать", "слушаю"),
        dependencies=(
            _subject(),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("ты", "тебя", "pronoun"),
                    grammatical_case="accusative",
                ),
            ),
        ),
        confidence=confidence,
        evidence=evidence,
    )


def _reply_thought(confidence: float, evidence: tuple[str, ...]) -> ClauseThought:
    """Represent taking the conversational turn as an action, before wording."""

    return ClauseThought(
        predicate=_verb("отвечать", "отвечаю"),
        dependencies=(
            _subject(),
            SemanticDependency(
                "indirect_object",
                ConceptArgument(
                    ConceptLexeme("ты", "тебе", "pronoun"),
                    grammatical_case="dative",
                ),
            ),
        ),
        confidence=confidence,
        evidence=evidence,
    )


def _attention_thought(confidence: float, evidence: tuple[str, ...]) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("держать", "держу"),
        dependencies=(
            _subject(),
            SemanticDependency(
                "direct_object",
                _noun("внимание", case="accusative"),
            ),
        ),
        confidence=confidence,
        evidence=evidence,
    )


def _understanding_thought(
    lemma: str,
    surface: str,
    confidence: float,
    evidence: tuple[str, ...],
) -> ClauseThought:
    return ClauseThought(
        predicate=_verb(lemma, surface, past=True),
        dependencies=(
            _subject(),
            SemanticDependency("direct_object", _noun("мысль", case="accusative")),
        ),
        response_polarity=True,
        confidence=confidence,
        evidence=evidence,
    )


def _support_thought(confidence: float, evidence: tuple[str, ...]) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("замечать", "замечаю"),
        dependencies=(
            _subject(),
            SemanticDependency("direct_object", _noun("поддержка", case="accusative")),
        ),
        confidence=confidence,
        evidence=evidence,
    )


def _failure_question(confidence: float, evidence: tuple[str, ...]) -> ClauseThought:
    thought = _event_question("сработать", "сработало", confidence, evidence)
    return ClauseThought(
        predicate=thought.predicate,
        dependencies=thought.dependencies,
        clause_kind="question",
        negated=True,
        focus_role="subject",
        confidence=confidence,
        evidence=evidence,
    )


def _failed_part_question(
    confidence: float,
    evidence: tuple[str, ...],
) -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme(
            "промахнуться",
            "промахнулся",
            "verb",
            (("tense", "past"), ("number", "singular"), ("gender", "masculine")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                _noun("ответ", modifiers=(ConceptLexeme("какой", "какая", "adjective"),)),
            ),
        ),
        clause_kind="question",
        focus_role="subject",
        confidence=confidence,
        evidence=evidence,
    )


def _reconsider_thought(confidence: float, evidence: tuple[str, ...]) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("пересматривать", "пересматриваю"),
        dependencies=(
            _subject(),
            SemanticDependency(
                "direct_object",
                _noun(
                    "ответ",
                    case="accusative",
                    modifiers=(ConceptLexeme("предыдущий", "предыдущий", "adjective"),),
                ),
            ),
        ),
        confidence=confidence,
        evidence=evidence,
    )


def _recheck_thought(confidence: float, evidence: tuple[str, ...]) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("уточнять", "уточняю"),
        dependencies=(
            _subject(),
            SemanticDependency("direct_object", _noun("вопрос", case="accusative")),
        ),
        confidence=confidence,
        evidence=evidence,
    )


def _continue_thought(confidence: float, evidence: tuple[str, ...]) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("продолжать", "продолжаю"),
        dependencies=(
            _subject(),
            SemanticDependency("direct_object", _noun("мысль", case="accusative")),
        ),
        confidence=confidence,
        evidence=evidence,
    )


def _retain_topic_thought(confidence: float, evidence: tuple[str, ...]) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("держать", "держу"),
        dependencies=(
            _subject(),
            SemanticDependency("direct_object", _noun("тема", case="accusative")),
        ),
        confidence=confidence,
        evidence=evidence,
    )
