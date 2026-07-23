"""General dialogue acts and response obligations above words and templates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from debilgi.language.content import content_lemmas, content_tokens
from debilgi.language.pragmatics import PragmaticAnalysis, analyze_pragmatics
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
    realize_thought,
)
from debilgi.language.question_reasoning import SemanticQuestionFrame
from debilgi.language.relation_reasoning import RelationQuestionFrame
from debilgi.language.self_state_reasoning import SelfStateFrame
from debilgi.language.structured_utterance import StructuredUtterance
from debilgi.language.tokenizer import tokenize
from debilgi.social.models import SocialReaction
from debilgi.social.signals import evidence_counts

_EVALUATIVE_QUALITY_LEMMAS = frozenset(
    {
        "адекватный",
        "бесполезный",
        "важный",
        "вредный",
        "гений",
        "глупый",
        "добрый",
        "достойный",
        "злой",
        "идиот",
        "интересный",
        "красивый",
        "логичный",
        "лох",
        "молодец",
        "надежный",
        "нормальный",
        "плохой",
        "полезный",
        "правильный",
        "разумный",
        "странный",
        "умный",
        "умник",
        "убедительный",
        "хороший",
        "честный",
    }
)
_DEICTIC_SUBJECTS = frozenset({"addressee", "speaker", "я", "ты", "вы"})
_STUDY_PREDICATES = frozenset({"изучить", "исследовать", "узнать", "разобраться"})
_POSITION_EXPRESSION_PREDICATES = frozenset(
    {"сказать", "высказать", "выразить", "поделиться"}
)
_ADDRESSEE_POSSESSIVES = frozenset({"твой", "ваш"})
_DIALOGUE_REFERENCE_NOUNS = frozenset(
    {"ответ", "реплика", "сообщение", "слово", "фраза", "вопрос"}
)


@dataclass(frozen=True, slots=True)
class DialogueActFrame:
    kind: str
    response_obligation: str
    subject: str | None
    predicate: str | None
    objects: tuple[str, ...]
    polarity: float
    register: str
    complete: bool
    confidence: float
    evidence: tuple[str, ...]


def analyze_dialogue_act(
    structure: StructuredUtterance,
    *,
    pragmatics: PragmaticAnalysis | None = None,
    reaction: SocialReaction | None = None,
    semantic_question: SemanticQuestionFrame | None = None,
    relation_question: RelationQuestionFrame | None = None,
    self_state: SelfStateFrame | None = None,
) -> DialogueActFrame:
    """Turn syntax, social evidence and pragmatics into one response contract."""

    pragmatics = pragmatics or analyze_pragmatics(structure.text)
    words = tuple(
        token.casefold().replace("ё", "е")
        for token in tokenize(structure.text)
        if token[:1].isalnum()
    )
    counts = evidence_counts(words)
    positive = min(1.0, counts["approval"] * 0.72 + counts["affirmation"] * 0.28)
    negative = min(
        1.0,
        counts["disapproval"] * 0.64
        + counts["aversion"] * 0.72
        + counts["refusal"] * 0.20,
    )
    if reaction is not None:
        positive = max(positive, reaction.vector.approval * reaction.confidence)
        negative = max(
            negative,
            max(reaction.vector.aversion, reaction.vector.resistance)
            * reaction.confidence,
        )
    register = (
        "coarse"
        if "coarse_emphasis" in pragmatics.functions
        else "emotional"
        if pragmatics.emotional_intensity >= 0.58
        else "neutral"
    )
    embedded_question_indices = {
        link.child_clause_index
        for link in structure.clause_links
        if link.relation.startswith("ccomp")
    }
    question_clause = next(
        (
            item
            for item in structure.clauses
            if item.clause_kind == "question"
            and item.clause_index not in embedded_question_indices
        ),
        None,
    )
    support_clause = max(
        (
            item
            for item in structure.clauses
            if item.predicate_index is not None or item.implicit_predicate is not None
        ),
        key=lambda item: item.confidence,
        default=None,
    )
    clause = question_clause or support_clause
    semantic_clause = (
        clause
        if clause is not None
        and (clause.predicate_index is not None or clause.implicit_predicate is not None)
        else support_clause
    )
    subject = (
        (
            _first_lemma(structure, semantic_clause.subject_indices)
            or semantic_clause.implicit_subject
        )
        if semantic_clause
        else None
    )
    predicate = (
        structure.tokens[semantic_clause.predicate_index].lemma
        if semantic_clause is not None and semantic_clause.predicate_index is not None
        else semantic_clause.implicit_predicate
        if semantic_clause is not None
        else None
    )
    objects = (
        tuple(
            dict.fromkeys(
                structure.tokens[index].surface.casefold()
            for index in (
                *semantic_clause.direct_object_indices,
                *semantic_clause.indirect_object_indices,
                *semantic_clause.predicate_complement_indices,
                    *(
                        index
                        for complement in semantic_clause.complements
                        for index in complement.object_indices
                        if structure.tokens[index].semantic_role
                        == "unassigned_content"
                    ),
            )
            )
        )
        if semantic_clause is not None
        else ()
    )

    # Conversational constructions carry a stronger obligation than the
    # isolated lexical items inside them.  This is deliberately evaluated
    # before ordinary question routing: ``как думаешь`` asks for a position,
    # while ``ты что ...`` usually challenges the preceding answer rather
    # than requesting a dictionary fact about its last noun.
    pragmatic_functions = pragmatics.functions
    topical = content_lemmas(structure.text)
    understanding_challenge = _understanding_challenge(structure)
    if understanding_challenge:
        return DialogueActFrame(
            kind="understanding_challenge",
            response_obligation="acknowledge_understanding",
            subject="addressee",
            predicate="понять",
            objects=("намёк",),
            polarity=0.0,
            register=register,
            complete=True,
            confidence=max(0.86, structure.confidence * 0.94),
            evidence=("construction:do_not_pretend_not_to_understand",),
        )
    if "rhetorical_challenge" in pragmatic_functions and predicate is not None:
        return DialogueActFrame(
            kind="rhetorical_challenge",
            response_obligation="repair_previous_answer",
            subject="addressee",
            predicate=predicate,
            objects=topical,
            polarity=-0.72,
            register=register,
            complete=True,
            confidence=max(0.82, structure.confidence * 0.90),
            evidence=("construction:rhetorical_challenge",),
        )
    if "ellipsis_alternative_request" in pragmatic_functions:
        return DialogueActFrame(
            kind="elliptical_follow_up",
            response_obligation="repair_previous_answer",
            subject="addressee",
            predicate=predicate,
            objects=topical,
            polarity=0.0,
            register=register,
            complete=True,
            confidence=max(0.84, structure.confidence * 0.92),
            evidence=("construction:ellipsis_alternative_request",),
        )
    if pragmatic_functions & {"opinion_request", "binary_answer_demand"}:
        opinion_objects = content_tokens(structure.text)
        return DialogueActFrame(
            kind="opinion_question",
            response_obligation="express_position",
            subject=subject or (clause.implicit_subject if clause else None),
            predicate=predicate,
            objects=opinion_objects,
            polarity=0.0,
            register=register,
            complete=bool(opinion_objects),
            confidence=max(0.84, structure.confidence * 0.92),
            evidence=tuple(
                f"construction:{item}"
                for item in ("opinion_request", "binary_answer_demand")
                if item in pragmatic_functions
            ),
        )

    # A Russian zero-copula question can request an evaluation without an
    # explicit ``как думаешь``: ``лохи хорошие люди?``.  Treat the evaluative
    # complement as a position slot, while ordinary properties such as
    # ``стол красный?`` remain factual questions.
    if (
        clause is not None
        and clause.clause_kind == "question"
        and clause.implicit_predicate == "быть"
        and clause.subject_indices
        and subject not in _DEICTIC_SUBJECTS
        and clause.predicate_complement_indices
    ):
        quality_index = next(
            (
                index
                for index in clause.predicate_complement_indices
                if structure.tokens[index].lemma in _EVALUATIVE_QUALITY_LEMMAS
            ),
            None,
        )
        if quality_index is not None:
            target_index = clause.subject_indices[0]
            return DialogueActFrame(
                kind="opinion_question",
                response_obligation="express_position",
                subject=subject,
                predicate="считать",
                objects=(
                    structure.tokens[target_index].surface.casefold(),
                    structure.tokens[quality_index].surface.casefold(),
                ),
                polarity=0.0,
                register=register,
                complete=True,
                confidence=max(0.82, structure.confidence * 0.90),
                evidence=("construction:evaluative_copular_question",),
            )

    if semantic_question is not None and semantic_question.complete:
        return DialogueActFrame(
            kind=(
                "alternative_choice"
                if semantic_question.kind == "choice"
                else "alternative_comparison"
            ),
            response_obligation="select_or_compare",
            subject=subject or (clause.implicit_subject if clause else None),
            predicate=predicate,
            objects=semantic_question.alternative_surfaces,
            polarity=0.0,
            register=register,
            complete=True,
            confidence=semantic_question.confidence,
            evidence=("alternative_pair", "question"),
        )

    if relation_question is not None and relation_question.complete:
        return DialogueActFrame(
            kind="relation_question",
            response_obligation="resolve_relation",
            subject=relation_question.holder_lemma,
            predicate=predicate,
            objects=(relation_question.object_surface,),
            polarity=0.0,
            register=register,
            complete=True,
            confidence=relation_question.confidence,
            evidence=(f"relation:{relation_question.relation}",),
        )
    if (
        self_state is not None
        and self_state.complete
        and self_state.clause_kind == "question"
        and self_state.experiencer_kind == "self"
    ):
        return DialogueActFrame(
            kind=(
                "self_reflection_question"
                if self_state.state == "self_reflection"
                else "self_state_question"
            ),
            response_obligation=(
                "express_position"
                if self_state.state == "self_reflection"
                else "resolve_self_state"
            ),
            subject=self_state.experiencer_kind,
            predicate=self_state.predicate_lemma,
            objects=self_state.target_lemmas,
            polarity=0.0,
            register=register,
            complete=True,
            confidence=self_state.confidence,
            evidence=(f"self_state:{self_state.state}",),
        )
    if (
        self_state is not None
        and self_state.complete
        and self_state.clause_kind == "statement"
        and self_state.experiencer_kind == "speaker"
        and self_state.state == "understanding"
        and self_state.negated
    ):
        return DialogueActFrame(
            kind="understanding_failure_report",
            response_obligation="repair_understanding",
            subject="speaker",
            predicate=self_state.predicate_lemma,
            objects=self_state.target_lemmas,
            polarity=-1.0,
            register=register,
            complete=True,
            confidence=self_state.confidence,
            evidence=("speaker_reported_misunderstanding",),
        )
    is_question = question_clause is not None
    if is_question:
        complete = bool(
            semantic_clause is not None
            and (
                semantic_clause.predicate_index is not None
                or semantic_clause.complements
                or semantic_clause.implicit_predicate is not None
            )
            or question_clause.question_target_index is not None
        )
        question_target = _semantic_question_target(structure, question_clause)
        return DialogueActFrame(
            kind="information_question",
            response_obligation="answer_question",
            subject=subject or question_clause.implicit_subject,
            predicate=predicate,
            objects=objects,
            polarity=0.0,
            register=register,
            complete=complete,
            confidence=structure.confidence * (0.94 if complete else 0.68),
            evidence=(
                "question_clause",
                *(("predicate",) if predicate else ()),
                *((f"question_target:{question_target}",) if question_target else ()),
            ),
        )

    if positive >= 0.46 and positive > negative + 0.12:
        return DialogueActFrame(
            kind="positive_evaluation",
            response_obligation="acknowledge_evaluation",
            subject="addressee",
            predicate=predicate,
            objects=objects,
            polarity=positive,
            register=register,
            complete=True,
            confidence=max(0.72, structure.confidence * 0.80 + positive * 0.20),
            evidence=("positive_social_evidence",),
        )
    if negative >= 0.46 and negative > positive + 0.12:
        return DialogueActFrame(
            kind="negative_evaluation",
            response_obligation="address_negative_evaluation",
            subject="addressee",
            predicate=predicate,
            objects=objects,
            polarity=-negative,
            register=register,
            complete=True,
            confidence=max(0.70, structure.confidence * 0.78 + negative * 0.22),
            evidence=("negative_social_evidence",),
        )
    reference = next(
        (
            token
            for token in structure.tokens
            if token.lemma in _DIALOGUE_REFERENCE_NOUNS
        ),
        None,
    )
    if (
        predicate is None
        and reference is not None
        and any(token.lemma in _ADDRESSEE_POSSESSIVES for token in structure.tokens)
    ):
        return DialogueActFrame(
            kind="dialogue_reference_fragment",
            response_obligation="acknowledge_reference",
            subject="addressee",
            predicate=None,
            objects=(reference.surface.casefold(),),
            polarity=0.0,
            register=register,
            complete=True,
            confidence=max(0.70, structure.confidence * 0.86),
            evidence=("elliptical_reference_to_agent_utterance",),
        )
    compound_directive = _compound_study_position_directive(structure, register)
    if compound_directive is not None:
        return compound_directive
    study_directive = _study_directive(structure, register)
    if study_directive is not None:
        return study_directive
    if clause is not None and clause.clause_kind == "command":
        return DialogueActFrame(
            kind="directive",
            response_obligation="handle_directive",
            subject=clause.implicit_subject,
            predicate=predicate,
            objects=objects,
            polarity=0.0,
            register=register,
            complete=predicate is not None,
            confidence=structure.confidence,
            evidence=("command_clause",),
        )
    return DialogueActFrame(
        kind="statement" if predicate else "fragment",
        response_obligation="continue_topic" if predicate else "clarify_fragment",
        subject=subject,
        predicate=predicate,
        objects=objects,
        polarity=positive - negative,
        register=register,
        complete=predicate is not None,
        confidence=structure.confidence * (0.82 if predicate else 0.56),
        evidence=("predicate",) if predicate else ("fragment",),
    )


def refine_dialogue_act(
    frame: DialogueActFrame,
    *,
    external_intent: str,
    confidence: float,
    external_response_obligation: str | None = None,
    expected_response_acts: tuple[str, ...] = (),
) -> DialogueActFrame:
    """Use teacher semantics to repair a generic act, never its wording."""

    intent = external_intent.casefold().replace("-", "_")
    obligation = (external_response_obligation or "").casefold().replace("-", "_")
    expected = {
        item.casefold().replace("-", "_") for item in expected_response_acts
    }
    generic_question = frame.kind == "information_question"
    if generic_question and confidence >= 0.62 and (
        any(marker in intent for marker in ("opinion", "reflection", "self_assessment"))
        or obligation in {"express_position", "decide"}
        or bool(expected & {"express_position", "decide"})
    ):
        return replace(
            frame,
            kind="opinion_question",
            response_obligation="express_position",
            complete=True,
            confidence=min(0.86, max(frame.confidence, confidence * 0.90)),
            evidence=(*frame.evidence, "external_intent:position_request"),
        )
    if frame.complete:
        return frame
    if any(marker in intent for marker in ("praise", "compliment", "approval")):
        return replace(
            frame,
            kind="positive_evaluation",
            response_obligation="acknowledge_evaluation",
            subject="addressee",
            polarity=max(0.7, confidence),
            complete=True,
            confidence=min(0.86, max(frame.confidence, confidence * 0.88)),
            evidence=(*frame.evidence, "external_intent:praise"),
        )
    if any(marker in intent for marker in ("insult", "mockery", "disapproval")):
        return replace(
            frame,
            kind="negative_evaluation",
            response_obligation="address_negative_evaluation",
            subject="addressee",
            polarity=-max(0.7, confidence),
            complete=True,
            confidence=min(0.86, max(frame.confidence, confidence * 0.88)),
            evidence=(*frame.evidence, "external_intent:negative_evaluation"),
        )
    return frame


def realize_dialogue_act(frame: DialogueActFrame) -> str | None:
    """Compatibility realizer backed by semantic roles, not reply strings."""

    thought = dialogue_act_thought(frame)
    surfaces = realize_thought(thought) if thought is not None else ()
    return surfaces[0].text if surfaces else None


def dialogue_act_thought(frame: DialogueActFrame) -> ClauseThought | None:
    """Turn a social obligation into a proposition for productive grammar."""

    if not frame.complete:
        return None
    if frame.response_obligation == "acknowledge_understanding":
        return ClauseThought(
            predicate=ConceptLexeme(
                "понять",
                "понял",
                "verb",
                (("tense", "past"), ("number", "singular"), ("gender", "masculine")),
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
                            "намёк",
                            "намёк",
                            "noun",
                            (
                                ("case", "accusative"),
                                ("number", "singular"),
                                ("gender", "masculine"),
                                ("animacy", "inanimate"),
                            ),
                        ),
                        grammatical_case="accusative",
                    ),
                ),
            ),
            response_polarity=True,
            confidence=frame.confidence,
            evidence=("acknowledge_understanding",),
        )
    if frame.response_obligation == "acknowledge_evaluation":
        return ClauseThought(
            predicate=ConceptLexeme(
                "стараться",
                "стараюсь",
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
            ),
            response_polarity=True,
            confidence=frame.confidence,
            evidence=("acknowledge_evaluation",),
        )
    if frame.response_obligation == "address_negative_evaluation":
        return ClauseThought(
            predicate=ConceptLexeme(
                "сработать",
                "сработало",
                "verb",
                (("tense", "past"), ("number", "singular"), ("gender", "neuter")),
            ),
            dependencies=(
                SemanticDependency(
                    "subject",
                    ConceptArgument(
                        ConceptLexeme(
                            "что",
                            "что",
                            "pronoun",
                            (("number", "singular"), ("gender", "neuter")),
                        )
                    ),
                ),
            ),
            clause_kind="question",
            negated=True,
            focus_role="subject",
            confidence=frame.confidence,
            evidence=("request_specific_negative_feedback",),
        )
    if frame.response_obligation == "repair_previous_answer":
        return ClauseThought(
            predicate=ConceptLexeme(
                "понять",
                "понял",
                "verb",
                (("tense", "past"), ("number", "singular"), ("gender", "masculine")),
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
                            "вопрос",
                            "вопрос",
                            "noun",
                            (
                                ("number", "singular"),
                                ("gender", "masculine"),
                                ("animacy", "inanimate"),
                            ),
                        ),
                        (ConceptLexeme("предыдущий", "предыдущий", "adjective"),),
                    ),
                ),
                SemanticDependency(
                    "manner",
                    ConceptArgument(ConceptLexeme("неправильно", "неправильно", "adverb")),
                ),
            ),
            confidence=frame.confidence,
            evidence=("repair_previous_answer",),
        )
    if frame.response_obligation in {"continue_topic", "acknowledge_reference"}:
        return ClauseThought(
            predicate=ConceptLexeme(
                "понять",
                "понял",
                "verb",
                (("tense", "past"), ("number", "singular"), ("gender", "masculine")),
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
                            "ты",
                            "тебя",
                            "pronoun",
                            (("number", "singular"), ("person", "second")),
                        ),
                        grammatical_case="accusative",
                    ),
                ),
            ),
            confidence=frame.confidence,
            evidence=(
                (
                    "acknowledge_dialogue_reference",
                )
                if frame.response_obligation == "acknowledge_reference"
                else ("continue_understood_topic",)
            ),
        )
    return None


def _understanding_challenge(structure: StructuredUtterance) -> bool:
    """Recognize ``не делай вид, что не понял`` as a social correction."""

    clauses = {clause.clause_index: clause for clause in structure.clauses}
    for link in structure.clause_links:
        if not link.relation.startswith("ccomp"):
            continue
        parent = clauses.get(link.parent_clause_index)
        child = clauses.get(link.child_clause_index)
        if parent is None or child is None:
            continue
        parent_predicate = (
            structure.tokens[parent.predicate_index].lemma
            if parent.predicate_index is not None
            else parent.implicit_predicate
        )
        child_predicate = (
            structure.tokens[child.predicate_index].lemma
            if child.predicate_index is not None
            else child.implicit_predicate
        )
        parent_objects = {
            structure.tokens[index].lemma
            for index in (
                *parent.direct_object_indices,
                *parent.predicate_complement_indices,
            )
        }
        if (
            parent.clause_kind == "command"
            and parent.negated
            and parent_predicate == "делать"
            and "вид" in parent_objects
            and child_predicate == "понять"
            and child.negated
        ):
            return True
    return False


def candidate_obligation_fit(
    frame: DialogueActFrame,
    text: str,
    metadata: Mapping[str, object] | None = None,
) -> float:
    """Score whether candidate meaning fulfils the turn, independent of source.

    A generator name is provenance, not semantics.  The old source whitelist
    could discard a perfectly grounded answer merely because it came from a
    newer branch.  This score instead looks at speech shape, topical overlap
    and the declared communicative move.
    """

    if not frame.complete:
        return 1.0
    metadata = metadata or {}
    lemmas = frozenset(content_lemmas(text))
    expected_items = (
        *((frame.subject,) if frame.subject and frame.subject not in _DEICTIC_SUBJECTS else ()),
        *frame.objects,
        *((frame.predicate,) if frame.predicate else ()),
    )
    expected = frozenset(
        lemma
        for item in expected_items
        for lemma in (
            content_lemmas(item)
            or (item.casefold().replace("ё", "е"),)
        )
        if lemma
    )
    overlap = len(lemmas & expected) / max(1, len(expected))
    words = frozenset(
        token.casefold().replace("ё", "е")
        for token in tokenize(text)
        if token[:1].isalnum()
    )
    question = "?" in text
    first_person = bool(words & {"я", "мне", "меня", "мой", "моя", "мои"})
    move = str(
        metadata.get("response_move") or metadata.get("social_move") or ""
    ).casefold().replace("-", "_")
    obligation = frame.response_obligation

    if obligation in {"repair_previous_answer", "repair_understanding"}:
        semantic_repair = bool(
            lemmas
            & {"понять", "ошибиться", "исправить", "пересобрать", "непонятный"}
        )
        repair_shape = question or bool(words & {"неправильно", "иначе"})
        return min(
            1.0,
            0.03
            + semantic_repair * 0.12
            + repair_shape * 0.58
            + (move == "targeted_repair") * 0.22,
        )
    if obligation == "express_position":
        stance = bool(lemmas & {"считать", "думать", "полагать", "выбирать"})
        direct_position = bool(words & {"да", "нет"}) and overlap > 0.0
        return min(
            1.0,
            0.12
            + first_person * 0.28
            + stance * 0.34
            + direct_position * 0.42
            + overlap * 0.30
            - question * 0.18,
        )
    if obligation == "select_or_compare":
        stance = bool(
            lemmas
            & {"выбрать", "выбирать", "предпочитать", "считать", "думать"}
        )
        return min(1.0, 0.05 + overlap * 0.50 + stance * 0.28 + (not question) * 0.10)
    if obligation == "study_and_express_position":
        completed = {
            str(item).casefold().replace("-", "_")
            for item in metadata.get("completed_subgoals", ())
            if isinstance(item, str)
        }
        has_study = "study" in completed
        has_position = "express_position" in completed
        lexical_content = bool(metadata.get("knowledge_component"))
        position_content = bool(metadata.get("position_component"))
        return min(
            1.0,
            0.04
            + has_study * 0.31
            + has_position * 0.31
            + lexical_content * 0.12
            + position_content * 0.12
            + overlap * 0.10,
        )
    if obligation == "study_and_answer":
        completed = {
            str(item).casefold().replace("-", "_")
            for item in metadata.get("completed_subgoals", ())
            if isinstance(item, str)
        }
        has_study = "study" in completed
        has_answer = "answer" in completed
        knowledge_content = bool(metadata.get("knowledge_component"))
        return min(
            1.0,
            0.04
            + has_study * 0.34
            + has_answer * 0.34
            + knowledge_content * 0.18
            + overlap * 0.10,
        )
    if obligation == "resolve_self_state":
        if not first_person:
            return 0.08 + (move == "answer") * 0.08
        return min(1.0, 0.12 + first_person * 0.38 + overlap * 0.40 + (not question) * 0.10)
    if obligation == "resolve_relation":
        relation_answer = bool(
            words & {"да", "нет", "есть"}
            or lemmas & {"иметь", "иметься", "отсутствовать", "принадлежать"}
        )
        if not relation_answer:
            return 0.06 + overlap * 0.08
        return min(1.0, 0.16 + overlap * 0.42 + relation_answer * 0.28 + (not question) * 0.10)
    if obligation == "answer_question":
        if move == "clarify_reference":
            return 0.92
        question_target = next(
            (
                item.split(":", 1)[1]
                for item in frame.evidence
                if item.startswith("question_target:")
            ),
            None,
        )
        slot_fit = _question_slot_fit(question_target, words, lemmas, metadata)
        answer_move = move in {"answer", "inform"}
        direct_marker = bool(words & {"да", "нет"})
        if question_target is not None:
            if slot_fit < 0.50:
                # Repeating the proposition from a WH-question does not fill
                # its missing cause, place, time or manner, even with perfect
                # lexical overlap and an ``answer`` metadata label.
                return min(0.34, 0.06 + overlap * 0.18 + slot_fit * 0.18)
            return min(
                1.0,
                0.06
                + overlap * 0.34
                + slot_fit * 0.42
                + answer_move * 0.16
                + (not question) * 0.08,
            )
        return min(
            1.0,
            0.06
            + overlap * 0.48
            + first_person * 0.08
            + answer_move * 0.24
            + direct_marker * 0.22
            + (not question) * 0.06,
        )
    if obligation in {"acknowledge_evaluation", "address_negative_evaluation"}:
        appropriate_move = move in {"acknowledge", "targeted_repair"}
        socially_bound = bool(lemmas & {"стараться", "сработать", "понять", "ошибиться"})
        return min(1.0, 0.16 + appropriate_move * 0.44 + socially_bound * 0.34)
    return min(1.0, 0.44 + overlap * 0.42 + (not question) * 0.14)


def candidate_fulfills_obligation(
    frame: DialogueActFrame,
    source: str,
    *,
    text: str = "",
    metadata: Mapping[str, object] | None = None,
) -> bool:
    """Compatibility boolean over semantic fit; ``source`` is only provenance."""

    del source
    minimum = {
        "repair_previous_answer": 0.52,
        "repair_understanding": 0.52,
        "express_position": 0.56,
        "select_or_compare": 0.48,
        "resolve_self_state": 0.52,
        "resolve_relation": 0.48,
        "answer_question": 0.54,
        "study_and_express_position": 0.72,
        "study_and_answer": 0.72,
        "acknowledge_evaluation": 0.48,
        "address_negative_evaluation": 0.48,
    }.get(frame.response_obligation, 0.42)
    return candidate_obligation_fit(frame, text, metadata) >= minimum


def _question_slot_fit(
    target: str | None,
    words: frozenset[str],
    lemmas: frozenset[str],
    metadata: Mapping[str, object],
) -> float:
    if target is None:
        return 1.0
    declared = str(metadata.get("answer_slot") or "").casefold()
    if declared == target:
        return 1.0
    if target in {"где", "куда", "откуда"}:
        cues = {
            "в",
            "на",
            "из",
            "с",
            "к",
            "около",
            "возле",
            "внутри",
            "снаружи",
            "там",
            "тут",
            "здесь",
            "туда",
            "сюда",
            "домой",
        }
        return 1.0 if words & cues else 0.0
    if target in {"почему", "отчего", "зачем"}:
        return 1.0 if words & {"потому", "причина", "из-за", "ради", "чтобы"} else 0.0
    if target == "когда":
        time_cues = {"когда", "тогда", "сейчас", "потом", "раньше", "после", "до"}
        return 1.0 if words & time_cues else 0.0
    if target == "как":
        return 1.0 if words & {"так", "через", "помощью", "способ", "метод"} else 0.45
    if target in {"кто", "что", "какой", "сколько"}:
        return 0.88 if lemmas - {target} else 0.0
    return 0.55


def _semantic_question_target(structure: StructuredUtterance, clause: object) -> str | None:
    index = getattr(clause, "question_target_index", None)
    target = structure.tokens[index].lemma if index is not None else None
    lemmas = {token.lemma for token in structure.tokens}
    # Russian often asks for a cause through a nominal construction rather
    # than the adverb ``почему``: ``какая причина того, что ...``.  The noun
    # describes the requested answer slot and must never become an argument
    # of the embedded predicate.
    if target in {"какой", "что"} and lemmas & {"причина", "основание", "повод"}:
        return "почему"
    if target in {"какой", "что"} and lemmas & {"цель", "смысл"}:
        return "зачем"
    return target


def _compound_study_position_directive(
    structure: StructuredUtterance,
    register: str,
) -> DialogueActFrame | None:
    commands = tuple(
        clause for clause in structure.clauses if clause.clause_kind == "command"
    )
    predicates = {
        structure.tokens[clause.predicate_index].lemma
        for clause in commands
        if clause.predicate_index is not None
    }
    lemmas = {token.lemma for token in structure.tokens}
    if not (
        predicates & _STUDY_PREDICATES
        and predicates & _POSITION_EXPRESSION_PREDICATES
        and "мнение" in lemmas
    ):
        return None
    topics = tuple(
        dict.fromkeys(
            structure.tokens[index].surface.casefold()
            for clause in commands
            if clause.predicate_index is not None
            and structure.tokens[clause.predicate_index].lemma in _STUDY_PREDICATES
            for index in _study_topic_indices(structure, clause.clause_index)
            if structure.tokens[index].lemma != "мнение"
        )
    )
    return DialogueActFrame(
        kind="compound_directive",
        response_obligation="study_and_express_position",
        subject="addressee",
        predicate="сказать",
        objects=topics,
        polarity=0.0,
        register=register,
        complete=bool(topics),
        confidence=max(0.86, structure.confidence * 0.94),
        evidence=("subgoal:study", "subgoal:express_position"),
    )


def _study_directive(
    structure: StructuredUtterance,
    register: str,
) -> DialogueActFrame | None:
    commands = tuple(
        clause
        for clause in structure.clauses
        if clause.clause_kind == "command"
        and clause.predicate_index is not None
        and structure.tokens[clause.predicate_index].lemma in _STUDY_PREDICATES
    )
    if not commands:
        return None
    topics = tuple(
        dict.fromkeys(
            structure.tokens[index].surface.casefold()
            for clause in commands
            for index in _study_topic_indices(structure, clause.clause_index)
            if structure.tokens[index].lemma
            not in {
                "что",
                "такой",
                "ответить",
                "рассказать",
                "объяснить",
                "сказать",
            }
        )
    )
    return DialogueActFrame(
        kind="study_directive",
        response_obligation="study_and_answer",
        subject="addressee",
        predicate="изучить",
        objects=topics,
        polarity=0.0,
        register=register,
        complete=bool(topics),
        confidence=max(0.86, structure.confidence * 0.94),
        evidence=("subgoal:study", "subgoal:answer"),
    )


def _study_topic_indices(
    structure: StructuredUtterance,
    study_clause_index: int,
) -> tuple[int, ...]:
    clause = structure.clauses[study_clause_index]
    direct = clause.direct_object_indices
    linked = tuple(
        structure.clauses[link.child_clause_index]
        for link in structure.clause_links
        if link.parent_clause_index == study_clause_index
        and link.relation.startswith("ccomp")
    )
    subordinate_arguments = tuple(
        index
        for child in linked
        for index in (
            *child.subject_indices,
            *child.direct_object_indices,
            *child.predicate_complement_indices,
        )
        if structure.tokens[index].semantic_role not in {"question_target", "function_word"}
    )
    unassigned_content = tuple(
        token.index
        for token in structure.tokens[clause.token_start : clause.token_end]
        if token.semantic_role == "unassigned_content"
    )
    return tuple(
        dict.fromkeys((*direct, *subordinate_arguments, *unassigned_content))
    )


def _first_lemma(structure: StructuredUtterance, indices: tuple[int, ...]) -> str | None:
    return structure.tokens[indices[0]].lemma if indices else None
