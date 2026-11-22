"""Boundary between a structured decision and any future surface wording."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from debilgi.language.generative_planner import ResponseDecision

from debilgi.cognition.dialogue_tasks import DialogueSituation
from debilgi.cognition.models import ActionIntent, ActionType
from debilgi.language.assertion_memory import AssertionMemory, ContinuationPlan
from debilgi.language.canonical_claims import Known, Unknown
from debilgi.language.construction_nlg import ConstructionSpec
from debilgi.language.dialogue_acts import TurnForm
from debilgi.language.dialogue_obligations import BoundDialogueReport
from debilgi.language.dialogue_state import DialogueStateAnalysis
from debilgi.language.productive_nlg import DependencyPattern
from debilgi.language.semantic_graph import MeaningGoal, MeaningRecord
from debilgi.language.social_reasoning import SocialPolicy
from debilgi.language.structured_utterance import StructuredUtterance, analyze_structured_utterance
from debilgi.language.usage_learning import UsageGuidance
from debilgi.language.word_relations import WordRelation
from debilgi.personality.expression import PersonalityExpression
from debilgi.personality.perspective import PersonalityPerspective
from debilgi.reflection.models import ReflectionGuidance
from debilgi.social.models import RelationState, SocialReaction


@dataclass(frozen=True, slots=True)
class StructuredThought:
    action_type: ActionType
    target_id: int | None
    objective: str
    confidence: float
    evidence: tuple[str, ...]
    intention_id: str = ""
    state_features: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class ReplyContext:
    """Relevant memory supplied to wording without changing the chosen action."""

    prompt: str = ""
    turn_prompt: str = ""
    response_options: tuple[str, ...] = ()
    focus_reason: str | None = None
    memory_messages: tuple[str, ...] = ()
    dialogue_messages: tuple[str, ...] = ()
    # The optional teacher receives labelled text. Local assertion reasoning
    # retains event provenance and participant bindings separately below.
    consultation_messages: tuple[str, ...] = ()
    reply_heads: tuple[tuple[str, ...], ...] = ()
    knowledge_facts: tuple[str, ...] = ()
    # Scoped human statements retained in SQLite, with their source event IDs.
    proposition_reports: tuple[tuple[str, str], ...] = ()
    claim_result: Known | Unknown | None = None
    dialogue_reports: tuple[BoundDialogueReport, ...] = ()
    meaning_records: tuple[MeaningRecord, ...] = ()
    meaning_goals: tuple[MeaningGoal, ...] = ()
    source_event_id: str = ""
    affect_features: tuple[tuple[str, float], ...] = ()
    dialogue_situation: DialogueSituation | None = None
    # Raw human language remains useful when a proposition cannot be parsed.
    corpus_utterances: tuple[tuple[str, str], ...] = ()
    word_relations: tuple[WordRelation, ...] = ()
    reference_definitions: tuple[tuple[str, str], ...] = ()
    required_facts: tuple[str, ...] = ()
    seed: str = ""
    personality: PersonalityExpression | None = None
    perspective: PersonalityPerspective | None = None
    reaction: SocialReaction | None = None
    relation: RelationState | None = None
    reflection: ReflectionGuidance | None = None
    exact_answer: str | None = None
    usage_hints: tuple[UsageGuidance, ...] = ()
    learned_syntax_patterns: tuple[tuple[str, ...], ...] = ()
    learned_constructions: tuple[ConstructionSpec, ...] = ()
    dependency_patterns: tuple[DependencyPattern, ...] = ()
    expected_response_acts: tuple[str, ...] = ()
    learned_registers: tuple[str, ...] = ()
    social_policies: tuple[SocialPolicy, ...] = ()
    dialogue_state: DialogueStateAnalysis | None = None
    assertion_memory: AssertionMemory = field(default_factory=AssertionMemory)
    assertion_continuation: ContinuationPlan | None = None
    assertion_feedback_counts: tuple[int, int] = (0, 0)
    input_meaning: StructuredUtterance | None = None
    response_decision: ResponseDecision | None = None
    turn_form: TurnForm | None = None
    prior_bot_turn: dict | None = None

    @property
    def structure(self) -> StructuredUtterance:
        # Focusing an earlier request invalidates only that projection.
        if self.input_meaning is not None and self.input_meaning.text == self.prompt:
            return self.input_meaning
        return analyze_structured_utterance(self.prompt)


@dataclass(frozen=True, slots=True)
class UtterancePlan:
    speech_act: str
    max_tokens: int
    thought: StructuredThought
    context: ReplyContext = field(default_factory=ReplyContext)


def thought_from_intent(intent: ActionIntent) -> StructuredThought:
    return StructuredThought(
        action_type=intent.action_type,
        target_id=intent.target_id,
        objective=str(intent.expected_effect.get("hypothesis", "observe")),
        confidence=intent.confidence,
        evidence=intent.evidence,
        intention_id=intent.intent_id,
        state_features=tuple(
            (key, float(value))
            for key, value in intent.decision_context.items()
            if key.startswith("state_") and isinstance(value, (int, float))
        ),
    )


def utterance_plan(
    thought: StructuredThought, *, context: ReplyContext | None = None
) -> UtterancePlan | None:
    speech_acts = {
        ActionType.SHORT_REPLY: "brief_statement",
        ActionType.ASK_QUESTION: "clarifying_question",
        ActionType.CONTINUE_TOPIC: "topic_continuation",
    }
    speech_act = speech_acts.get(thought.action_type)
    reaction = context.reaction if context is not None else None
    if speech_act and speech_act != "clarifying_question" and reaction is not None:
        vector = reaction.vector
        positive = max(vector.agreement, vector.approval)
        negative = max(vector.resistance, vector.aversion)
        if positive >= 0.45 and negative >= 0.45 and abs(positive - negative) <= 0.28:
            speech_act = "qualified_position"
        elif vector.resistance >= 0.58 and vector.resistance > vector.agreement:
            speech_act = "boundary_statement"
        elif vector.aversion >= 0.58:
            speech_act = "aversive_statement"
        elif thought.confidence < 0.5 and positive >= 0.36:
            speech_act = "cautious_position"
        elif vector.approval >= 0.55:
            speech_act = "approving_statement"
        elif vector.agreement >= 0.55:
            speech_act = "agreeing_statement"
    return (
        UtterancePlan(
            speech_act=speech_act,
            max_tokens=72,
            thought=thought,
            context=context or ReplyContext(),
        )
        if speech_act
        else None
    )
