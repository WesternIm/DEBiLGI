"""Grounded capability reasoning for explicit modal questions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from debilgi.language.content import content_lemmas
from debilgi.language.lexical_matching import same_lexical_family
from debilgi.language.lexicon import WordAnalysis
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    DependencyPattern,
    SemanticDependency,
    realize_thought,
)
from debilgi.language.relation_reasoning import classify_object_terms
from debilgi.language.structured_utterance import (
    StructuredUtterance,
    analyze_structured_utterance,
)
from debilgi.language.tokenizer import tokenize

_CAPABILITY_PREDICATES = frozenset({"мочь", "уметь"})
_COMPUTATIONAL_ACTIONS = frozenset(
    {
        "анализировать",
        "выбирать",
        "думать",
        "оценивать",
        "оценить",
        "определять",
        "определить",
        "изучать",
        "искать",
        "объяснять",
        "обрабатывать",
        "обсуждать",
        "обсудить",
        "описывать",
        "описать",
        "отвечать",
        "писать",
        "написать",
        "помнить",
        "понимать",
        "понять",
        "продолжать",
        "продолжить",
        "проверять",
        "прочитать",
        "рассуждать",
        "рассказывать",
        "рассказать",
        "решать",
        "сказать",
        "сравнивать",
        "считать",
        "сообщать",
        "сообщить",
        "устанавливать",
        "установить",
        "строить",
        "учиться",
        "читать",
    }
)
_UNASSISTED_HUMAN_LIMITS = frozenset({"летать"})
_KNOWN_HUMAN_SUBJECTS = frozenset(
    {
        "человек",
        "люди",
        "астроном",
        "космонавт",
        "учёный",
        "врач",
        "учитель",
        "пилот",
    }
)
_NON_ACTION_STATES = frozenset(
    {
        "быть",
        "верить",
        "знать",
        "иметь",
        "казаться",
        "любить",
        "нравиться",
        "помнить",
        "понимать",
        "считать",
        "хотеть",
    }
)
_ABSTRACT_INTERNAL_OBJECTS = frozenset(
    {
        "взгляд",
        "идея",
        "мнение",
        "отношение",
        "оценка",
        "позиция",
        "предпочтение",
        "решение",
        "смысл",
    }
)
_EXTERNAL_ACTIONS = frozenset(
    {
        "бежать",
        "бить",
        "брать",
        "взять",
        "выкинуть",
        "держать",
        "есть",
        "закрыть",
        "идти",
        "играть",
        "копать",
        "летать",
        "нести",
        "носить",
        "открыть",
        "пить",
        "плыть",
        "поиграть",
        "поднять",
        "положить",
        "разбить",
        "съесть",
        "сыграть",
        "трогать",
        "увидеть",
        "упасть",
        "хватать",
    }
)
_OBJECT_SENSITIVE_EXTERNAL_ACTIONS = frozenset(
    {
        "купить",
        "покупать",
        "потерять",
        "ронять",
        "снять",
        "снимать",
        "терять",
        "уронить",
    }
)
_RESOURCE_USE_ACTIONS = frozenset(
    {
        "использовать",
        "пользоваться",
        "применять",
        "юзать",
    }
)
_LEARNED_INTERNAL_CAPABILITY_MARKERS = frozenset(
    {
        "анализ",
        "данные",
        "информация",
        "литературный",
        "модель",
        "программа",
        "слово",
        "стих",
        "текст",
        "цифровой",
        "язык",
    }
)
_LEARNED_EXTERNAL_CAPABILITY_MARKERS = frozenset(
    {
        "атака",
        "выпас",
        "группа",
        "животное",
        "нападение",
        "организация",
        "организовать",
        "рейд",
        "событие",
        "стадо",
        "территория",
        "участник",
        "физический",
    }
)


@dataclass(frozen=True, slots=True)
class CapabilityAssessment:
    subject_lemma: str
    subject_surface: str
    subject_kind: str
    action_lemma: str
    supported: bool
    reason: str
    confidence: float
    subject_number: str | None = None
    mode: str = "capability"
    action_object: ConceptArgument | None = None

    @property
    def patterns(self) -> tuple[DependencyPattern, ...]:
        if self.action_object is None or self.mode in {
            "experience",
            "activity",
            "hypothetical_desire",
        }:
            return ()
        roles = (
            ("subject", "negation", "predicate", "action_complement", "direct_object")
            if not self.supported
            else ("subject", "predicate", "action_complement", "direct_object")
        )
        return (
            DependencyPattern(
                roles,
                confidence=0.97,
                evidence_count=8,
                source="russian_modal_action_structure",
                predicate_class="modal",
                negation_mode="required" if not self.supported else "forbidden",
            ),
        )

    @property
    def thought(self) -> ClauseThought:
        subject_lemma = "я" if self.subject_kind == "bot" else self.subject_lemma
        subject_surface = "я" if self.subject_kind == "bot" else self.subject_surface
        subject = ConceptArgument(
            ConceptLexeme(
                subject_lemma,
                subject_surface,
                "pronoun" if subject_lemma == "я" else "noun",
                (("number", "singular"), ("person", "first"))
                if subject_lemma == "я"
                else (
                    ("number", self.subject_number or "singular"),
                    ("person", "third"),
                ),
            )
        )
        if self.mode == "experience":
            dependencies = [SemanticDependency("subject", subject)]
            if self.action_object is not None:
                dependencies.append(SemanticDependency("direct_object", self.action_object))
            return ClauseThought(
                predicate=ConceptLexeme(
                    self.action_lemma,
                    self.action_lemma,
                    "verb",
                    (
                        ("tense", "past"),
                        ("number", "singular"),
                        ("gender", "masculine"),
                    ),
                ),
                dependencies=tuple(dependencies),
                negated=not self.supported,
                response_polarity=self.supported,
                confidence=self.confidence,
                evidence=(self.reason,),
            )
        if self.mode == "activity":
            dependencies = [SemanticDependency("subject", subject)]
            if self.action_object is not None:
                dependencies.append(SemanticDependency("direct_object", self.action_object))
            return ClauseThought(
                predicate=ConceptLexeme(
                    self.action_lemma,
                    self.action_lemma,
                    "verb",
                    (
                        ("tense", "present"),
                        ("number", "singular"),
                        ("person", "first"),
                    ),
                ),
                dependencies=tuple(dependencies),
                negated=not self.supported,
                response_polarity=self.supported,
                confidence=self.confidence,
                evidence=(self.reason,),
            )
        if self.mode == "hypothetical_desire":
            dependencies = [
                SemanticDependency("subject", subject),
                SemanticDependency(
                    "action_complement",
                    ConceptArgument(ConceptLexeme(self.action_lemma, self.action_lemma, "verb")),
                ),
            ]
            if self.action_object is not None:
                dependencies.append(SemanticDependency("direct_object", self.action_object))
            return ClauseThought(
                predicate=ConceptLexeme(
                    "хотеть",
                    "хочу",
                    "verb",
                    (
                        ("tense", "present"),
                        ("number", "singular"),
                        ("person", "first"),
                    ),
                ),
                dependencies=tuple(dependencies),
                response_polarity=True,
                confidence=self.confidence,
                evidence=(self.reason,),
                predicate_class="modal",
            )
        predicate_lemma = (
            "мочь"
            if self.mode == "directive"
            else "уметь"
            if self.subject_kind == "bot"
            else "мочь"
        )
        predicate_surface = (
            "могу"
            if self.mode == "directive"
            else "умею"
            if self.subject_kind == "bot"
            else "может"
        )
        action = ConceptArgument(ConceptLexeme(self.action_lemma, self.action_lemma, "verb"))
        dependencies = [
            SemanticDependency("subject", subject),
            SemanticDependency("action_complement", action),
        ]
        if self.action_object is not None:
            dependencies.append(SemanticDependency("direct_object", self.action_object))
        return ClauseThought(
            predicate=ConceptLexeme(
                predicate_lemma,
                predicate_surface,
                "verb",
                (
                    ("tense", "present"),
                    ("number", self.subject_number or "singular"),
                ),
            ),
            dependencies=tuple(dependencies),
            negated=not self.supported,
            response_polarity=(self.supported if self.mode != "directive" else None),
            confidence=self.confidence,
            evidence=(self.reason,),
            predicate_class="modal",
        )


def assess_capability_question(
    structure: StructuredUtterance,
    *,
    analyses: Sequence[WordAnalysis] = (),
    context_messages: tuple[str, ...] = (),
    knowledge_facts: Sequence[str] = (),
    study_completed: bool = False,
) -> CapabilityAssessment | None:
    """Ground self-directed capability, experience and external-action requests."""

    for frame in reversed(structure.clauses):
        if frame.predicate_index is None:
            continue
        predicate = structure.tokens[frame.predicate_index]
        subject = structure.tokens[frame.subject_indices[-1]] if frame.subject_indices else None
        addressed_bot = frame.implicit_subject == "addressee" or bool(
            subject is not None and subject.lemma in {"ты", "вы"}
        )
        if (
            not addressed_bot
            and context_messages
            and frame.clause_kind == "question"
            and predicate.lemma == "хотеть"
            and any(
                token.lemma == "бы"
                for token in structure.tokens[frame.token_start : frame.token_end]
            )
        ):
            # A bare ``а хотел бы?`` continues the interlocutor-directed
            # question in the previous turn; Russian normally omits ``ты``.
            addressed_bot = True
        if (
            not addressed_bot
            and frame.clause_kind == "question"
            and frame.implicit_subject == "previous_clause_subject"
            and predicate.lemma == "хотеть"
        ):
            addressed_bot = True
        action_object = _action_object(
            structure,
            frame,
            subject.index if subject is not None else None,
        )
        resource_use_supported = _resource_use_support(
            predicate.lemma,
            action_object,
            analyses,
            study_completed=study_completed,
        )
        learned_predicate_support = _learned_capability_support(
            predicate.lemma,
            action_object,
            knowledge_facts,
        )
        if (
            addressed_bot
            and frame.clause_kind == "question"
            and predicate.lemma == "хотеть"
            and any(
                token.lemma == "бы"
                for token in structure.tokens[frame.token_start : frame.token_end]
            )
        ):
            action = next(
                (structure.tokens[index] for index in frame.action_complement_indices),
                None,
            )
            context_action, context_object = _context_action(context_messages)
            action_lemma = action.lemma if action is not None else context_action
            if action_lemma is not None:
                return CapabilityAssessment(
                    "ты",
                    "ты",
                    "bot",
                    action_lemma,
                    True,
                    "hypothetical_desire_can_be_imagined",
                    0.88,
                    "singular",
                    "hypothetical_desire",
                    action_object or context_object or _context_action_object(context_messages),
                )
        requires_external_agency = (
            action_requires_external_agency(predicate.lemma)
            or _object_sensitive_external_action(
                predicate.lemma,
                action_object,
                analyses,
            )
            or resource_use_supported is False
            or learned_predicate_support is False
        )
        if (
            addressed_bot
            and frame.clause_kind == "question"
            and predicate.lemma
            not in (_CAPABILITY_PREDICATES | _NON_ACTION_STATES | _COMPUTATIONAL_ACTIONS)
            and study_completed
        ):
            # A question about the bot's current or past activity is a closed
            # self-state once its action was researched.  Lack of an active
            # grounded operation is a negative answer, not a knowledge gap.
            requires_external_agency = True
        if predicate.lemma in _CAPABILITY_PREDICATES:
            action = next(
                (
                    token
                    for token in structure.tokens[frame.predicate_index + 1 : frame.token_end]
                    if token.part_of_speech == "verb" and "INFN" in token.grammemes
                ),
                None,
            )
            if action is None or subject is None:
                continue
            action_object = _action_object(structure, frame, subject.index)
        elif (
            addressed_bot
            and frame.clause_kind == "command"
            and action_requires_external_agency(predicate.lemma)
        ):
            return CapabilityAssessment(
                "ты",
                "ты",
                "bot",
                predicate.lemma,
                False,
                "directive_requires_unavailable_external_agency",
                0.94,
                "singular",
                "directive",
                _action_object(structure, frame, None),
            )
        elif (
            addressed_bot
            and frame.clause_kind == "question"
            and dict(predicate.grammatical_features).get("tense") == "past"
            and (
                requires_external_agency
                or predicate.lemma
                not in (
                    _NON_ACTION_STATES | _COMPUTATIONAL_ACTIONS | _OBJECT_SENSITIVE_EXTERNAL_ACTIONS
                )
            )
        ):
            return CapabilityAssessment(
                "ты",
                "ты",
                "bot",
                predicate.lemma,
                False,
                "past_action_requires_body_or_external_agency",
                0.95,
                "singular",
                "experience",
                action_object,
            )
        elif (
            addressed_bot
            and frame.clause_kind == "question"
            and dict(predicate.grammatical_features).get("tense") == "present"
            and requires_external_agency
        ):
            return CapabilityAssessment(
                "ты",
                "ты",
                "bot",
                predicate.lemma,
                False,
                "ongoing_action_requires_unavailable_external_agency",
                0.93,
                "singular",
                "activity",
                action_object,
            )
        elif (
            addressed_bot
            and frame.clause_kind == "question"
            and dict(predicate.grammatical_features).get("tense") in {"present", "future"}
            and requires_external_agency
        ):
            # Russian requests are often phrased as a bare future-tense
            # question: ``откроешь бутылку?``.  This is an action decision,
            # not a request to retrieve some old fact containing ``открыл``.
            return CapabilityAssessment(
                "ты",
                "ты",
                "bot",
                predicate.lemma,
                False,
                "prospective_action_requires_unavailable_external_agency",
                0.95,
                "singular",
                "directive",
                action_object,
            )
        else:
            continue
        if subject.lemma in {"ты", "вы"}:
            abstract_internal_state = bool(
                action.lemma in _NON_ACTION_STATES
                and action_object is not None
                and action_object.head.lemma in _ABSTRACT_INTERNAL_OBJECTS
            )
            if action.lemma in _COMPUTATIONAL_ACTIONS:
                supported = True
                reason = "internal_computational_action"
            elif abstract_internal_state:
                supported = True
                reason = "internal_cognitive_or_evaluative_state"
            elif action_requires_external_agency(action.lemma):
                supported = False
                reason = "action_requires_unavailable_external_agency"
            elif (
                learned_support := _learned_capability_support(
                    action.lemma,
                    action_object,
                    knowledge_facts,
                )
            ) is not None:
                supported = learned_support
                reason = "learned_internal_action" if learned_support else "learned_external_action"
            elif study_completed:
                # Capability is a closed self-state.  After all foreground
                # sources were actually awaited, an unregistered action is
                # unsupported rather than an excuse to emit "no data".
                supported = False
                reason = "capability_unavailable_after_foreground_study"
            else:
                # An unknown action is uncertainty, not evidence that a body
                # is required.  Let other candidates or study resolve it.
                continue
            return CapabilityAssessment(
                subject.lemma,
                subject.surface,
                "bot",
                action.lemma,
                supported,
                reason,
                0.96 if supported else 0.92,
                dict(subject.grammatical_features).get("number"),
                "capability",
                action_object,
            )
        if subject.lemma in _KNOWN_HUMAN_SUBJECTS and action.lemma in _UNASSISTED_HUMAN_LIMITS:
            return CapabilityAssessment(
                subject.lemma,
                subject.surface,
                "person",
                action.lemma,
                False,
                "human_requires_technology",
                0.94,
                dict(subject.grammatical_features).get("number"),
                "capability",
                action_object,
            )
    return None


def capability_study_terms(structure: StructuredUtterance) -> tuple[str, ...]:
    """Return action terms blocking a grounded answer about the bot itself."""

    for frame in reversed(structure.clauses):
        if frame.predicate_index is None or frame.clause_kind != "question":
            continue
        predicate = structure.tokens[frame.predicate_index]
        subject = structure.tokens[frame.subject_indices[-1]] if frame.subject_indices else None
        if subject is None or subject.lemma not in {"ты", "вы"}:
            continue
        action = (
            next(
                (
                    token
                    for token in structure.tokens[frame.predicate_index + 1 : frame.token_end]
                    if token.part_of_speech == "verb" and "INFN" in token.grammemes
                ),
                None,
            )
            if predicate.lemma in _CAPABILITY_PREDICATES
            else predicate
            if predicate.lemma not in (_NON_ACTION_STATES | _COMPUTATIONAL_ACTIONS)
            else None
        )
        if action is None:
            continue
        action_object = _action_object(structure, frame, subject.index)
        if action.lemma in _RESOURCE_USE_ACTIONS and action_object is not None:
            return (action_object.head.lemma,)
        return tuple(
            dict.fromkeys(
                (
                    action.lemma,
                    *((action_object.head.lemma,) if action_object is not None else ()),
                )
            )
        )
    return ()


def realize_capability_assessment(assessment: CapabilityAssessment | None) -> str | None:
    if assessment is None:
        return None
    surfaces = realize_thought(assessment.thought, assessment.patterns)
    return surfaces[0].text if surfaces else None


def action_requires_external_agency(action_lemma: str) -> bool:
    """Separate internal language work from bodily or world-changing actions."""

    return action_lemma.casefold() in _EXTERNAL_ACTIONS if action_lemma else False


def _object_sensitive_external_action(
    action_lemma: str,
    action_object: ConceptArgument | None,
    analyses: Sequence[WordAnalysis],
) -> bool:
    """Resolve embodied/polysemous actions from the object's learned type."""

    if action_lemma.casefold() not in _OBJECT_SENSITIVE_EXTERNAL_ACTIONS or action_object is None:
        return False
    object_lemma = action_object.head.lemma.casefold()
    terms: set[str] = set()
    for analysis in analyses:
        if (analysis.lemma or analysis.normalized_form).casefold() != object_lemma:
            continue
        explanations = (
            *analysis.dictionary_meanings,
            analysis.selected_meaning,
            analysis.local_meaning,
            analysis.personal_interpretation,
        )
        terms.update(
            term
            for explanation in explanations
            if explanation
            for term in content_lemmas(explanation)
        )
    object_class, _, _ = classify_object_terms(terms)
    return object_class == "physical"


def _learned_capability_support(
    action_lemma: str,
    action_object: ConceptArgument | None,
    knowledge_facts: Sequence[str],
) -> bool | None:
    """Classify an unknown action from facts obtained before reply generation."""

    anchors = {action_lemma.casefold()}
    if action_object is not None:
        anchors.add(action_object.head.lemma.casefold())
        anchors.update(content_lemmas(action_object.head.lemma))
    relevant_terms: set[str] = set()
    for fact in knowledge_facts:
        fact_terms = set(content_lemmas(fact))
        if not fact_terms or not any(
            same_lexical_family(anchor, term) for anchor in anchors for term in fact_terms
        ):
            continue
        relevant_terms.update(fact_terms)
    if relevant_terms & _LEARNED_EXTERNAL_CAPABILITY_MARKERS:
        return False
    if relevant_terms & _LEARNED_INTERNAL_CAPABILITY_MARKERS:
        return True
    return None


def _action_object(structure, frame, subject_index: int | None) -> ConceptArgument | None:
    indices = frame.direct_object_indices
    preposition = None
    if not indices:
        complement = next(
            (item for item in frame.complements if item.object_indices),
            None,
        )
        if complement is not None:
            indices = complement.object_indices
            preposition = structure.tokens[complement.preposition_index].lemma
    if not indices:
        indices = tuple(
            token.index
            for token in structure.tokens[frame.token_start : frame.token_end]
            if token.index != subject_index
            and token.part_of_speech == "noun"
            and token.index != frame.predicate_index
        )
    if not indices:
        # Chat names, abbreviations and borrowed compounds are frequently
        # tagged as interjections or remain unclassified.  A content token
        # governed by the verb still fills its object slot.
        indices = tuple(
            token.index
            for token in structure.tokens[frame.token_start : frame.token_end]
            if token.index != subject_index
            and token.index != frame.predicate_index
            and (token.corrected_form or token.surface)[:1].isalnum()
            and token.part_of_speech
            not in {
                "adverb",
                "conjunction",
                "particle",
                "predicate",
                "preposition",
                "pronoun",
                "verb",
            }
        )
    if not indices:
        return None
    token = structure.tokens[indices[0]]
    name_indices = [indices[0]]
    for index in indices[1:]:
        continuation = structure.tokens[index]
        surface = continuation.corrected_form or continuation.surface
        if surface.isdigit() or continuation.part_of_speech is None:
            name_indices.append(index)
            continue
        break
    name_tokens = tuple(structure.tokens[index] for index in name_indices)
    lemma = " ".join(item.lemma for item in name_tokens)
    surface = " ".join(item.corrected_form or item.surface for item in name_tokens)
    return ConceptArgument(
        ConceptLexeme(
            lemma,
            surface,
            token.part_of_speech,
            token.grammatical_features,
            token.entity_kind,
        ),
        preposition=preposition,
        grammatical_case=(dict(token.grammatical_features).get("case") or "accusative"),
    )


def _resource_use_support(
    action_lemma: str,
    action_object: ConceptArgument | None,
    analyses: Sequence[WordAnalysis],
    *,
    study_completed: bool,
) -> bool | None:
    """Map a learned resource type onto the agent's embodiment boundary."""

    if action_lemma.casefold() not in _RESOURCE_USE_ACTIONS or action_object is None:
        return None
    object_lemma = action_object.head.lemma.casefold()
    terms: set[str] = set()
    for analysis in analyses:
        if (analysis.lemma or analysis.normalized_form).casefold() != object_lemma:
            continue
        explanations = (
            *analysis.dictionary_meanings,
            analysis.selected_meaning,
            analysis.local_meaning,
            analysis.personal_interpretation,
        )
        terms.update(
            term
            for explanation in explanations
            if explanation
            for term in content_lemmas(explanation)
        )
    object_class, _, _ = classify_object_terms(terms)
    if object_class == "physical":
        return False
    # If every foreground source was tried and still supplied no usable type,
    # the agent can still report that the resource is not among its observed
    # active tools.  This is a closed self-state after research, not a claim
    # that the word itself is unknowable.
    return False if object_class is None and study_completed else None


def _context_action_object(
    context_messages: tuple[str, ...],
) -> ConceptArgument | None:
    ignored = (
        _CAPABILITY_PREDICATES
        | _COMPUTATIONAL_ACTIONS
        | _NON_ACTION_STATES
        | _EXTERNAL_ACTIONS
        | {
            "подтвердить",
            "установить",
            "ответ",
            "вопрос",
            "интересно",
            "это",
        }
    )
    for message in reversed(context_messages[-8:]):
        raw_tokens = tuple(
            token.casefold().replace("ё", "е") for token in tokenize(message) if token[:1].isalnum()
        )
        numbered_topic = next(
            (
                f"{raw_tokens[index - 1]} {raw_tokens[index]}"
                for index in range(len(raw_tokens) - 1, 0, -1)
                if raw_tokens[index].isdigit()
                and len(raw_tokens[index - 1]) > 1
                and raw_tokens[index - 1] not in ignored
            ),
            None,
        )
        if numbered_topic is not None:
            return ConceptArgument(
                ConceptLexeme(numbered_topic, numbered_topic, "noun"),
                preposition="в",
                grammatical_case="accusative",
            )
        topic = next(
            (
                term
                for term in reversed(content_lemmas(message))
                if term not in ignored and len(term) > 1
            ),
            None,
        )
        if topic is None:
            continue
        return ConceptArgument(
            ConceptLexeme(topic, topic, "noun"),
            preposition="в",
            grammatical_case="accusative",
        )
    return None


def _context_action(
    context_messages: tuple[str, ...],
) -> tuple[str | None, ConceptArgument | None]:
    """Recover an omitted action and its governed object from recent turns."""

    latest_action: str | None = None
    for message in reversed(context_messages[-8:]):
        speaker, separator, body = message.partition(":")
        text = (
            body.strip()
            if separator and speaker.casefold() in {"agent", "assistant", "bot", "user"}
            else message
        )
        structure = analyze_structured_utterance(text)
        for frame in reversed(structure.clauses):
            if frame.predicate_index is None:
                continue
            predicate = structure.tokens[frame.predicate_index]
            action = (
                next(
                    (structure.tokens[index] for index in frame.action_complement_indices),
                    None,
                )
                if predicate.lemma in (_CAPABILITY_PREDICATES | {"хотеть"})
                else predicate
            )
            if (
                action is None
                or action.part_of_speech != "verb"
                or action.lemma in _NON_ACTION_STATES
            ):
                continue
            if latest_action is None:
                latest_action = action.lemma
            elif not same_lexical_family(latest_action, action.lemma):
                continue
            subject_index = frame.subject_indices[-1] if frame.subject_indices else None
            action_object = _action_object(structure, frame, subject_index)
            if action_object is not None:
                return latest_action, action_object
    return latest_action, None
