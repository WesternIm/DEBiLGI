"""Narrow opt-in bridge from a stored intent to a harmless text reply."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime
from time import perf_counter

from debilgi.autonomy.policy import InitiativePolicy
from debilgi.cognition.continuous import ContinuousCognitionService
from debilgi.cognition.core_repository import CognitiveRepository
from debilgi.cognition.models import ActionIntent, ActionType
from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.events.models import Event
from debilgi.experience.consolidation import ExperienceConsolidator
from debilgi.experience.models import PreparedReply
from debilgi.knowledge.models import KnowledgeGrounding
from debilgi.knowledge.policy import KnowledgeQueryPolicy
from debilgi.knowledge.service import KnowledgeAcquisitionService, concise_reference
from debilgi.language.abstract_reasoning import (
    AbstractReasoningService,
    build_mental_model,
    resolve_abstract_question,
)
from debilgi.language.addressing import DEFAULT_ADDRESSING
from debilgi.language.arbitration import (
    ArbitrationDecision,
    ArbitrationProfile,
    ResponseArbitrator,
    ResponseCandidate,
    merge_recovery_decision,
)
from debilgi.language.arbitration_repository import ArbitrationRepository
from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.assertion_experience import (
    AssertionExperienceRepository,
    experience_proposal,
)
from debilgi.language.assertion_memory import (
    build_assertion_memory,
    event_followup_thought,
    plan_assertion_continuation,
    recall_event_slot,
)
from debilgi.language.autonomous_response import realize_autonomous_responses
from debilgi.language.capabilities import (
    action_requires_external_agency,
    assess_capability_question,
    capability_study_terms,
)
from debilgi.language.causal_reasoning import (
    CausalReasoningRepository,
    analyze_causal_query,
)
from debilgi.language.clarification_policy import (
    is_contact_request,
    lexical_question_target,
    requests_word_definition,
)
from debilgi.language.communication import MessagePlan, PendingText, ResponseComposer, reader_goal
from debilgi.language.content import content_lemmas, content_tokens
from debilgi.language.context_gate import WorkingMemoryContextGate
from debilgi.language.contextual_question_reasoning import (
    epistemic_gap_candidates_for_question,
    epistemic_gap_for_question,
    reason_about_contextual_question,
)
from debilgi.language.conversation import (
    ConversationRepository,
    ResponsePlan,
    ResponsePlanRepository,
    validate_response,
)
from debilgi.language.conversational_repair import (
    repair_for_utterance,
    unknown_meaning_repair,
)
from debilgi.language.definition_relations import definition_answer_thought
from debilgi.language.dialogue_acts import (
    DialogueActFrame,
    analyze_dialogue_act,
    candidate_fulfills_obligation,
    candidate_obligation_fit,
    realize_dialogue_act,
    refine_dialogue_act,
)
from debilgi.language.dialogue_move_planner import plan_dialogue_moves
from debilgi.language.dialogue_obligations import exchange_from_events
from debilgi.language.dialogue_state import (
    analyze_dialogue_state,
    realize_reference_clarification,
)
from debilgi.language.discourse import DiscourseContext, DiscourseMemoryRepository, topic_keys
from debilgi.language.entities import (
    EntityAnalysis,
    analyze_entities,
    identity_reply,
    identity_thought,
    render_message_content,
)
from debilgi.language.grounded_repair import (
    grounded_repair_thought,
    prior_claim_review_thought,
    unmet_requirement_thought,
    unresolved_answer_thought,
)
from debilgi.language.lexical_matching import same_lexical_family
from debilgi.language.lexicon import (
    LexiconService,
    UnknownKind,
    WordAnalysis,
    is_substantive_definition,
)
from debilgi.language.meaning_plan import (
    build_response_meaning_plan,
    realize_response_meaning,
)
from debilgi.language.open_questions import OpenLanguageQuestionService
from debilgi.language.personal_action_reasoning import (
    analyze_personal_action,
    reason_about_personal_action,
)
from debilgi.language.planning import (
    PhraseGenerator,
    ReplyContext,
    StructuredThought,
    thought_from_intent,
    utterance_plan,
)
from debilgi.language.position_reasoning import (
    PositionDecision,
    PositionReasoningService,
    realize_position,
)
from debilgi.language.pragmatics import PragmaticAnalysis, analyze_pragmatics
from debilgi.language.preference_reasoning import (
    appraise_preference,
    preference_position_thought,
    preference_reason_thought,
)
from debilgi.language.premise_reasoning import infer_from_premises, realize_premise_inference
from debilgi.language.processing_outcomes import ProcessingOutcomeRepository
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
    realize_thought,
)
from debilgi.language.quantitative_reasoning import resolve_travel_time_question
from debilgi.language.question_reasoning import (
    analyze_semantic_question,
    reason_from_internal_knowledge,
)
from debilgi.language.recovery import (
    RecoveryEvidence,
    compose_dialogue_reply,
    compose_recovery,
    is_clarification_request,
    is_identity_question,
    is_social_routine,
    lexical_lookup_target,
)
from debilgi.language.reference_answers import (
    definition_action_thought,
    definition_head,
    definition_requested,
    unrequested_definition,
)
from debilgi.language.relation_reasoning import (
    analyze_relation_question,
    reason_about_relation,
)
from debilgi.language.research_wait import LexicalResearchPending
from debilgi.language.response_alignment import assess_response_alignment
from debilgi.language.response_requirements import response_requirement
from debilgi.language.response_trace import ResponseTraceRepository
from debilgi.language.runtime_state_reasoning import (
    analyze_runtime_state,
    reason_about_runtime_state,
)
from debilgi.language.russian_reference import RussianReferenceService
from debilgi.language.ruwordnet_reasoning import ruwordnet_answer_thought
from debilgi.language.self_state_reasoning import (
    analyze_self_state,
    reason_about_self_state,
)
from debilgi.language.semantic_assimilation import (
    SemanticAssimilationService,
    SemanticLessonGuidance,
)
from debilgi.language.semantic_consultant import (
    ConsultationRequest,
    SemanticConsultationService,
)
from debilgi.language.semantic_inquiry import (
    analyze_semantic_inquiry,
    inquiry_trace,
    realize_semantic_inquiry,
)
from debilgi.language.semantic_repeat import repeats_proposition
from debilgi.language.semantic_reply_plan import (
    build_clarification_plan,
    realize_clarification_plan,
)
from debilgi.language.semantics import SemanticMemoryRepository
from debilgi.language.sense_resolution import (
    UtteranceSenseResolution,
    resolve_contextual_senses,
)
from debilgi.language.sentence_understanding import (
    UtteranceUnderstanding,
    UtteranceUnderstandingRepository,
    understand_utterance,
)
from debilgi.language.short_acts import ShortSpeechActService
from debilgi.language.social_reasoning import (
    candidate_social_features,
    choose_communicative_intention,
)
from debilgi.language.spatial_dynamics import (
    realize_boundary_entry_explanation,
    resolve_boundary_entry_explanation,
    resolve_boundary_entry_question,
)
from debilgi.language.structured_utterance import (
    StructuredUtterance,
    analyze_structured_utterance,
)
from debilgi.language.surface_coherence import COMPOSED_SURFACE_SOFT_ISSUES, evaluate_surface
from debilgi.language.surface_policy import internal_surface_reason
from debilgi.language.tokenizer import tokenize
from debilgi.language.usage_learning import (
    LexicalStudyMaterial,
    LexicalUsageLearningService,
    UsageGuidance,
)
from debilgi.literacy.engine import LiteracyEngine
from debilgi.memory.long_term import LongTermMemoryRepository
from debilgi.memory.short_term import ShortTermMemory
from debilgi.personality.expression import PersonalityExpressionEngine
from debilgi.personality.perspective import (
    PersonalityPerspective,
    build_personality_perspective,
)
from debilgi.reflection.engine import ReflectionEngine
from debilgi.social.models import RelationState, SocialReaction
from debilgi.social.repository import SocialReactionRepository

_WHITESPACE_RE = re.compile(r"\s+")
_TECHNICAL_TOKEN_RE = re.compile(r"<(?:user|role|channel|url|emoji:[^>]+)>", re.IGNORECASE)
_WORD_SURFACE_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_-]+")
_QUOTATION_REQUEST_RE = re.compile(
    r"\b(?:процитир\w*|цитат\w*|дословно|точными\s+словами|повтори\s+фразу)\b",
    re.IGNORECASE,
)
_BARE_STANCE_ANSWERS = frozenset({"да", "нет", "сейчас будет", "не хочу"})
_BROKEN_OPENING_WORDS = frozenset({"ответа", "значит", "контекста", "смысла", "темы"})
_QUESTION_OBJECTION_RE = re.compile(
    r"\b(?:специально\s+зада\w*\s+вопрос\w*|хватит\s+зада\w*\s+вопрос\w*|"
    r"не\s+зада\w*\s+вопрос\w*|зачем\s+(?:ты\s+)?(?:мне\s+)?зада\w*\s+вопрос\w*)\b",
    re.IGNORECASE,
)
_DIRECT_MEMORY_SURFACE_SOURCES = frozenset(
    {"grounded_surface", "knowledge_retrieval", "lexical_reference", "meaning_plan"}
)
_CONVERSATIONAL_REASONING_SOURCES = frozenset(
    {
        "abstract_reasoning",
        "autonomous_semantic_plan",
        "causal_reasoning",
        "dialogue_act",
        "dialogue_move",
        "dialogue_context",
        "dialogue_rule",
        "personal_action",
        "personality_desire",
        "position_reasoning",
        "quantitative_reasoning",
        "relation_reasoning",
        "runtime_state",
        "semantic_reasoning",
        "semantic_inquiry",
        "semantic_synthesis",
        "semantic_thought",
        "spatial_dynamics",
    }
)
_AUTONOMOUS_LOCAL_PRODUCTIVE_SOURCES = frozenset(
    {
        "discord_identity",
        "abstract_reasoning",
        "causal_reasoning",
        "dialogue_context",
        "dialogue_move",
        "personal_action",
        "personality_desire",
        "position_reasoning",
        "quantitative_reasoning",
        "relation_reasoning",
        "runtime_state",
        "semantic_reasoning",
        "semantic_inquiry",
        "semantic_synthesis",
        "semantic_thought",
        "spatial_dynamics",
    }
)
_INFORMATION_DIRECTIVE_PREDICATES = frozenset(
    {
        "объяснить",
        "описать",
        "показать",
        "рассказать",
        "напомнить",
        "назвать",
        "изучить",
        "исследовать",
        "узнать",
        "разобраться",
    }
)
# This is the only fixed transport-level reply.  Content fallbacks belong to
# semantic reasoners or evidence-backed recovery, never to a rotating phrase bank.
_TERMINAL_PROCESSING_FAILURE_REPLY = "Повтори запрос: сейчас ответ не отправился."
_failure_logger = logging.getLogger("debilgi.reply_failures")
_topic_logger = logging.getLogger("debilgi.topic_continuity")
_identity_logger = logging.getLogger("debilgi.identity_replies")
_understanding_logger = logging.getLogger("debilgi.utterance_understanding")
_structured_logger = logging.getLogger("debilgi.structured_utterance")
_sense_logger = logging.getLogger("debilgi.sense_resolution")


def _dialogue_act_requests_memory_fact(frame: DialogueActFrame | None) -> bool:
    """Keep remembered facts as evidence unless the user explicitly requests one."""

    if frame is None:
        return False
    if frame.response_obligation in {"answer_question", "resolve_relation"}:
        return True
    return (
        frame.response_obligation
        in {
            "handle_directive",
            "study_and_answer",
            "study_and_express_position",
        }
        and frame.predicate in _INFORMATION_DIRECTIVE_PREDICATES
    )


_meaning_logger = logging.getLogger("debilgi.meaning_plan")
_semantic_reply_logger = logging.getLogger("debilgi.semantic_reply_plan")
_dialogue_state_logger = logging.getLogger("debilgi.dialogue_state")
_contextual_question_logger = logging.getLogger("debilgi.contextual_question_reasoning")


def _quotation_requested(prompt: str) -> bool:
    """Return whether verbatim stored wording is the requested payload."""

    return _QUOTATION_REQUEST_RE.search(prompt) is not None


def _question_addresses_agent(
    dialogue_act: DialogueActFrame | None,
    structure: StructuredUtterance | None,
) -> bool:
    """Identify questions about the interlocutor, including omitted ``ты``."""

    if dialogue_act is not None and dialogue_act.subject in {
        "addressee",
        "self",
        "ты",
        "вы",
    }:
        return True
    if structure is None:
        return False
    for frame in structure.clauses:
        if frame.clause_kind != "question":
            continue
        if frame.implicit_subject == "addressee":
            return True
        if any(structure.tokens[index].lemma in {"ты", "вы"} for index in frame.subject_indices):
            return True
        clause_tokens = structure.tokens[frame.token_start : frame.token_end]
        if any(
            token.lemma in {"ты", "вы"} and index > 0 and clause_tokens[index - 1].lemma == "у"
            for index, token in enumerate(clause_tokens)
        ):
            return True
    return False


def _research_fact_bound_to_prompt(fact: str, prompt: str) -> bool:
    """Require researched evidence to mention the actual question topic."""

    stop_terms = {
        "быть",
        "делать",
        "идти",
        "иметь",
        "какой",
        "сейчас",
        "стоить",
        "теперь",
        "этот",
    }
    prompt_terms = tuple(term for term in content_lemmas(prompt) if term not in stop_terms)
    fact_terms = content_lemmas(fact)
    return bool(
        prompt_terms
        and any(
            _same_lexical_family(prompt_term, fact_term)
            for prompt_term in prompt_terms
            for fact_term in fact_terms
        )
    )


def _latest_user_consultation_prompt(
    consultation_messages: tuple[str, ...],
) -> str | None:
    for message in reversed(consultation_messages):
        role, separator, content = message.partition(": ")
        if separator and role.casefold() == "user" and content.strip():
            return content.strip()
    return None


def _previous_answer_grounding(
    consultation_messages: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Compile the latest agent surface back into reusable proposition facts."""

    previous = next(
        (
            content.strip()
            for message in reversed(consultation_messages)
            if (parts := message.partition(": "))[1]
            if parts[0].casefold() == "agent"
            if (content := parts[2]).strip()
        ),
        None,
    )
    if previous is None:
        return ()
    structure = analyze_structured_utterance(previous)
    clause = max(structure.clauses, key=lambda item: item.confidence, default=None)
    if (clause is None or clause.predicate_index is None
            or clause.clause_kind not in {"statement", "nominal_statement"}):
        return ()
    predicate = structure.tokens[clause.predicate_index]
    action = (
        structure.tokens[clause.action_complement_indices[0]]
        if clause.action_complement_indices
        else None
    )
    direct_object = (
        structure.tokens[clause.direct_object_indices[0]]
        if clause.direct_object_indices
        else None
    )
    negated = clause.negated or any(
        token.lemma == "не"
        for token in structure.tokens[clause.token_start : clause.token_end]
    )
    features = dict(predicate.grammatical_features)
    statement = (
        f"predicate={predicate.lemma};subject=agent;"
        + (f"action_complement={action.lemma};" if action is not None else "")
        + (f"direct_object={direct_object.lemma};" if direct_object is not None else "")
        + (f"tense={features['tense']};" if features.get("tense") else "")
        + f"asserted_polarity={str(not negated).lower()};epistemic_status=unverified_claim. "
        + f"Предыдущий ответ агента: {previous}"
    )
    facts = [("previous_answer.statement", statement)]
    if negated and action is not None and action_requires_external_agency(action.lemma):
        facts.append(
            (
                "previous_answer.capability_boundary",
                "predicate=иметь;subject=agent;direct_object=тело;"
                "tense=present;truth=false. Отказ связан с отсутствием физического тела.",
            )
        )
    return tuple(facts)


def _looks_like_lexical_definition(text: str) -> bool:
    """Reject dictionary metadata when the requested payload is not a definition."""

    normalized = _WHITESPACE_RE.sub(" ", text.casefold().replace("ё", "е")).strip()
    lexical_labels = (
        "глагол",
        "существительное",
        "прилагательное",
        "наречие",
        "местоимение",
        "предлог",
        "союз",
        "частица",
        "междометие",
        "слово",
        "термин",
    )
    return bool(
        re.search(
            rf"(?:—|-|это)\s+(?:\w+\s+)?(?:{'|'.join(lexical_labels)})\b",
            normalized,
        )
        or re.search(r"\bслово\s+.+\s+(?:значит|означает)\b", normalized)
    )


def _research_facts_may_surface(
    *,
    dialogue_act: DialogueActFrame | None,
    structure: StructuredUtterance | None,
    explicit_lookup: bool,
) -> bool:
    """Keep research as reasoning evidence unless facts are the requested answer."""

    if explicit_lookup:
        return True
    if dialogue_act is None:
        return False
    if _question_addresses_agent(dialogue_act, structure):
        return False
    return dialogue_act.response_obligation in {
        "answer_question",
        "resolve_relation",
        "study_and_answer",
    }


def _allows_epistemic_recovery(
    dialogue_act: DialogueActFrame | None,
    structure: StructuredUtterance | None,
) -> bool:
    """Reserve FAQ-like evidence-gap replies for actual world-fact questions.

    A dialogue act is evidence about the user's move, not a response mode.
    Questions addressed to the bot, social challenges, positions and local
    follow-ups must remain conversational even when no stored fact exists.
    """

    if dialogue_act is None or structure is None:
        return False
    if _question_addresses_agent(dialogue_act, structure):
        return False
    if dialogue_act.kind in {
        "elliptical_follow_up",
        "negative_evaluation",
        "opinion_question",
        "positive_evaluation",
        "rhetorical_challenge",
        "self_reflection_question",
        "self_state_question",
        "understanding_challenge",
    }:
        return False
    question_clauses = tuple(
        clause for clause in structure.clauses if clause.clause_kind == "question"
    )
    if (
        question_clauses
        and all(clause.elliptical for clause in question_clauses)
        and not content_lemmas(structure.text)
    ):
        return False
    if not dialogue_act.complete:
        question_words = {
            "а",
            "где",
            "зачем",
            "как",
            "какой",
            "когда",
            "кто",
            "куда",
            "откуда",
            "почему",
            "сколько",
            "что",
            "так",
            "тогда",
            "то",
            "это",
        }
        substantive_terms = set(content_lemmas(structure.text)) - question_words
        if not substantive_terms:
            return False
    return dialogue_act.response_obligation in {
        "answer_question",
        "resolve_relation",
        "study_and_answer",
    }


def _decision_trace_details(decision: ArbitrationDecision) -> dict[str, object]:
    return {
        "decision_id": decision.decision_id,
        "winner_candidate_id": (
            decision.winner.candidate_id if decision.winner is not None else None
        ),
        "winner_source": decision.winner.source if decision.winner is not None else None,
        "winner_text": decision.winner.text if decision.winner is not None else None,
        "minimum_score": decision.minimum_score,
        "used_fallback": decision.used_fallback,
        "selection_reason": decision.selection_reason,
        "latency_ms": decision.latency_ms,
        "scores": tuple(
            {
                "candidate_id": score.candidate.candidate_id,
                "source": score.candidate.source,
                "renderer": score.candidate.renderer,
                "text": score.candidate.text,
                "total": score.total,
                "admission_score": score.admission_score,
                "answer_status": score.answer_status,
                "eligible": score.eligible,
                "rejection_reasons": score.rejection_reasons,
                "features": score.features.as_dict(),
                "weighted_components": dict(score.weighted_components),
            }
            for score in decision.scores
        ),
        "content_scores": tuple(
            {
                "proposal_id": score.proposal.proposal_id,
                "source": score.proposal.source,
                "response_goal": score.proposal.plan.response_goal,
                "total": score.total,
                "eligible": score.eligible,
                "rejection_reasons": score.rejection_reasons,
                "components": dict(score.components),
            }
            for score in decision.content_scores
        ),
    }


class SafeMentionReplyService:
    def __init__(
        self,
        repository: CognitiveRepository,
        memory: ShortTermMemory,
        archive_repository: ArchiveRepository,
        expression_engine: PersonalityExpressionEngine,
        experience_consolidator: ExperienceConsolidator,
        initiative_policy: InitiativePolicy,
        generator: PhraseGenerator,
        *,
        social_reactions: SocialReactionRepository | None = None,
        reflection: ReflectionEngine | None = None,
        literacy: LiteracyEngine | None = None,
        discourse_memory: DiscourseMemoryRepository | None = None,
        semantic_memory: SemanticMemoryRepository | None = None,
        causal_reasoning: CausalReasoningRepository | None = None,
        knowledge_service: KnowledgeAcquisitionService | None = None,
        lexicon: LexiconService | None = None,
        conversation_memory: ConversationRepository | None = None,
        response_plans: ResponsePlanRepository | None = None,
        long_term_memory: LongTermMemoryRepository | None = None,
        self_dataset: SelfDatasetRepository | None = None,
        russian_reference: RussianReferenceService | None = None,
        open_questions: OpenLanguageQuestionService | None = None,
        short_speech_acts: ShortSpeechActService | None = None,
        usage_learning: LexicalUsageLearningService | None = None,
        understanding_repository: UtteranceUnderstandingRepository | None = None,
        arbitration_repository: ArbitrationRepository | None = None,
        context_gate: WorkingMemoryContextGate | None = None,
        semantic_consultant: SemanticConsultationService | None = None,
        semantic_assimilation: SemanticAssimilationService | None = None,
        position_reasoning: PositionReasoningService | None = None,
        abstract_reasoning: AbstractReasoningService | None = None,
        processing_outcomes: ProcessingOutcomeRepository | None = None,
        response_traces: ResponseTraceRepository | None = None,
        continuous_cognition: ContinuousCognitionService | None = None,
        assertion_experiences: AssertionExperienceRepository | None = None,
        arbitration_mode: str = "off",
        arbitration_minimum_score: float = 0.45,
        arbitration_max_total_candidates: int = 48,
        response_confidence_threshold: float = 0.45,
        response_plan_retry_limit: int = 2,
        context_character_limit: int = 600,
        productive_only_replies: bool = False,
        enabled: bool = False,
    ) -> None:
        if context_character_limit <= 0:
            raise ValueError("context_character_limit must be positive")
        self._repository = repository
        self._assertion_experiences = assertion_experiences
        self._memory = memory
        self._archive_repository = archive_repository
        self._expression_engine = expression_engine
        self._experience_consolidator = experience_consolidator
        self._initiative_policy = initiative_policy
        self._generator = generator
        self._social_reactions = social_reactions
        self._reflection = reflection
        self._literacy = literacy
        self._discourse_memory = discourse_memory
        self._semantic_memory = semantic_memory
        self._causal_reasoning = causal_reasoning
        self._knowledge_service = knowledge_service
        self._lexicon = lexicon
        self._conversation_memory = conversation_memory
        self._response_plans = response_plans
        self._long_term_memory = long_term_memory
        self._self_dataset = self_dataset
        self._russian_reference = russian_reference
        self._open_questions = open_questions
        self._short_speech_acts = short_speech_acts
        self._usage_learning = usage_learning
        self._understanding_repository = understanding_repository
        self._arbitration_repository = arbitration_repository
        self._context_gate = context_gate
        self._semantic_consultant = semantic_consultant
        self._autonomous_planning_required = bool(
            getattr(semantic_consultant, "autonomous_planning", False)
        )
        self._semantic_assimilation = semantic_assimilation
        self._position_reasoning = position_reasoning or PositionReasoningService()
        self._abstract_reasoning = abstract_reasoning
        self._processing_outcomes = processing_outcomes
        self._response_traces = response_traces
        self._continuous_cognition = continuous_cognition
        self._arbitration_mode = arbitration_mode
        self._arbitration_max_total_candidates = max(4, min(arbitration_max_total_candidates, 96))
        self._arbitrator = ResponseArbitrator(minimum_score=arbitration_minimum_score)
        self._response_confidence_threshold = response_confidence_threshold
        self._response_plan_retry_limit = response_plan_retry_limit
        self._context_character_limit = context_character_limit
        self._productive_only_replies = productive_only_replies
        self._stable_reply_cache: OrderedDict[str, str] = OrderedDict()
        self._pending_concepts: OrderedDict[str, str] = OrderedDict()
        self._candidate_sources: OrderedDict[str, str] = OrderedDict()
        self._arbitration_decisions: OrderedDict[str, ArbitrationDecision] = OrderedDict()
        self._pending_positions: OrderedDict[str, PositionDecision] = OrderedDict()
        self.enabled = enabled

    @property
    def productive_only_replies(self) -> bool:
        return self._productive_only_replies

    async def response_for(self, event: Event) -> str | None:
        prepared = await self.prepare_reply(event)
        return prepared.text if prepared else None

    async def _trace_begin(self, event: Event) -> None:
        if self._response_traces is None:
            return
        await self._response_traces.begin(
            event,
            prompt=self._clean_content(event),
        )

    async def _trace_step(
        self,
        event_id: str,
        *,
        stage: str,
        component: str,
        status: str,
        summary: str,
        details: dict[str, object] | None = None,
    ) -> None:
        if self._response_traces is None:
            return
        await self._response_traces.step(
            event_id,
            stage=stage,
            component=component,
            status=status,
            summary=summary,
            details=details,
        )

    async def prepare_guaranteed_reply(
        self,
        event: Event,
        *,
        upstream_failure: str | None = None,
    ) -> PreparedReply | None:
        """Guarantee a user-facing response for every direct, enabled mention."""

        if (
            not self.enabled
            or event.payload.get("bot_mentioned") is not True
            or event.payload.get("author_is_bot") is True
        ):
            return None
        await self._trace_begin(event)
        if upstream_failure is not None:
            await self._trace_step(
                event.event_id,
                stage="upstream",
                component="event_pipeline",
                status="failed",
                summary="event processing failed before reply generation",
                details={"failure": upstream_failure},
            )
            self._log_reply_failure(event, upstream_failure)
            if self._productive_only_replies or self._autonomous_planning_required:
                return await self._withhold_non_productive_failure(event, upstream_failure)
            return await self._failure_reply(event)
        try:
            prepared = await self.prepare_reply(event)
        except asyncio.CancelledError:
            raise
        except LexicalResearchPending:
            raise
        except Exception as exc:
            await self._trace_step(
                event.event_id,
                stage="generation",
                component="safe_reply_service",
                status="failed",
                summary="reply generation raised an exception",
                details={"exception_type": type(exc).__name__, "exception": str(exc)},
            )
            self._log_reply_failure(event, "generation_exception", exc)
            if self._productive_only_replies or self._autonomous_planning_required:
                return await self._withhold_non_productive_failure(
                    event,
                    "generation_exception",
                )
            return await self._failure_reply(event)
        if prepared is not None and prepared.text.strip():
            return prepared
        self._log_reply_failure(event, "empty_generation_result")
        await self._trace_step(
            event.event_id,
            stage="generation",
            component="safe_reply_service",
            status="failed",
            summary="reply generation returned no usable text",
        )
        if self._productive_only_replies or self._autonomous_planning_required:
            return await self._withhold_non_productive_failure(
                event,
                (
                    "autonomous_planning_no_surface"
                    if self._autonomous_planning_required
                    else "productive_only_no_surface"
                ),
            )
        return await self._failure_reply(event)

    async def prepare_reply(self, event: Event) -> PreparedReply | None:
        if (
            not self.enabled
            or event.payload.get("bot_mentioned") is not True
            or event.payload.get("author_is_bot") is True
        ):
            return None
        await self._trace_begin(event)
        intent = await self._repository.intent_for_event(event.event_id)
        if intent is None:
            await self._trace_step(
                event.event_id,
                stage="intent",
                component="cognitive_repository",
                status="missing",
                summary="no action intent was available",
            )
            return None
        await self._trace_step(
            event.event_id,
            stage="intent",
            component="cognitive_repository",
            status="selected",
            summary="action intent selected",
            details={
                "intent_id": intent.intent_id,
                "action_type": intent.action_type.value,
                "confidence": intent.confidence,
                "evidence": intent.evidence,
                "reasoning_trace": intent.reasoning_trace,
            },
        )
        context = await self._reply_context(event, intent)
        await self._trace_step(
            event.event_id,
            stage="context",
            component="reply_context",
            status="assembled",
            summary="conversation and knowledge context assembled",
            details={
                "prompt": context.prompt,
                "dialogue_messages": context.dialogue_messages,
                "consultation_messages": context.consultation_messages,
                "exact_answer": context.exact_answer,
                "required_facts": context.required_facts,
                "knowledge_facts": context.knowledge_facts,
                "dialogue_state": context.dialogue_state,
            },
        )
        plan = utterance_plan(thought_from_intent(intent), context=context)
        if plan is None:
            await self._trace_step(
                event.event_id,
                stage="planning",
                component="utterance_plan",
                status="failed",
                summary="utterance plan could not be created",
            )
            return None
        await self._trace_step(
            event.event_id,
            stage="planning",
            component="utterance_plan",
            status="created",
            summary="meaning-level utterance plan created",
            details={
                "speech_act": plan.speech_act,
                "objective": plan.thought.objective,
                "confidence": plan.thought.confidence,
            },
        )
        semantic_reply = self._prepare_semantic_clarification(event, intent, plan)
        if semantic_reply is not None and not requests_word_definition(semantic_reply.text):
            return semantic_reply
        text = await self._generate_checked(event, plan)
        research_gaps = await self._defer_pending_research(
            event, plan.context.prompt, plan.context.dialogue_messages,
            plan.context.dialogue_state.resolved_bindings if plan.context.dialogue_state else (),
        )
        decision = self._arbitration_decisions.get(event.event_id)
        grounded_answer = bool(decision and decision.winner and decision.winner.source in {
            "semantic_thought", "semantic_reasoning", "discord_identity", "dialogue_context",
            "personal_action", "runtime_state", "relation_reasoning",
        })
        if unrequested_definition(
            text, plan.context.prompt, plan.context.reference_definitions
        ) or requests_word_definition(text) or (
            research_gaps and text.rstrip().endswith("?") and not grounded_answer
        ):
            text = realize_thought(unresolved_answer_thought(), limit=1)[0].text
            self._remember_candidate_source(event.event_id, "semantic_recovery")
            self._arbitration_decisions.pop(event.event_id, None)
        if not text or context.personality is None:
            return None
        arbitration = self._arbitration_decisions.pop(event.event_id, None)
        legacy_source = self._candidate_sources.pop(event.event_id, "graph")
        pending_concept = self._pending_concepts.pop(event.event_id, None)
        prepared = PreparedReply(
            text=text,
            intent_id=intent.intent_id,
            tokens=tokenize(text),
            genome_signature=context.personality.genome_signature,
            candidate_count=(
                len(arbitration.scores)
                if arbitration is not None
                else context.personality.candidate_count
            ),
            fallback=(
                arbitration.used_fallback
                if arbitration is not None
                else legacy_source == "fallback"
            ),
            pending_concept=pending_concept if requests_word_definition(text) else None,
            response_source=(
                f"arbitrated:{arbitration.winner.source}"
                if arbitration is not None and arbitration.winner is not None
                else legacy_source
            ),
            arbitration_decision_id=(arbitration.decision_id if arbitration is not None else None),
            winner_candidate_id=(
                arbitration.winner.candidate_id
                if arbitration is not None and arbitration.winner is not None
                else None
            ),
            arbitration_weight_version=(
                arbitration.weight_version if arbitration is not None else 0
            ),
            assertion_experience=(
                arbitration.winner.metadata.get("assertion_experience")
                if arbitration is not None and arbitration.winner is not None
                and arbitration.winner.text == text else None
            ),
        )
        await self._trace_step(
            event.event_id,
            stage="prepared",
            component="safe_reply_service",
            status="ready",
            summary="reply is ready for the context gate",
            details={
                "text": prepared.text,
                "response_source": prepared.response_source,
                "fallback": prepared.fallback,
                "candidate_count": prepared.candidate_count,
                "arbitration_decision_id": prepared.arbitration_decision_id,
                "winner_candidate_id": prepared.winner_candidate_id,
            },
        )
        return prepared

    def _prepare_semantic_clarification(
        self,
        event: Event,
        intent: ActionIntent,
        utterance,
    ) -> PreparedReply | None:
        if (
            intent.action_type is not ActionType.ASK_QUESTION
            or utterance.speech_act != "clarifying_question"
            or utterance.context.exact_answer is not None
            or utterance.context.required_facts
            or utterance.context.personality is None
        ):
            return None
        plan = build_clarification_plan(
            intention_id=intent.intent_id,
            target_user_id=intent.target_id or event.actor_id,
            prompt=utterance.context.prompt,
            confidence=intent.confidence,
            supporting_memory_ids=intent.evidence,
            context_messages=utterance.context.dialogue_messages,
        )
        result = realize_clarification_plan(
            plan,
            utterance.context.dependency_patterns,
            max_attempts=max(2, self._response_plan_retry_limit),
        )
        if result.text is None and self._productive_only_replies:
            return None
        text = result.text or "Не поняла смысл. Пояснишь?"
        fallback = result.text is None
        _semantic_reply_logger.info(
            "event=%s intent=%s plan=%s outcome=%s candidates=%s fallback=%s "
            "failure_stage=%s failure_reason=%s",
            event.event_id,
            intent.intent_id,
            plan.plan_id,
            result.outcome.outcome_type.value,
            len(result.candidates),
            fallback,
            result.outcome.failure_stage,
            result.outcome.failure_reason,
        )
        return PreparedReply(
            text=text,
            intent_id=intent.intent_id,
            tokens=tokenize(text),
            genome_signature=utterance.context.personality.genome_signature,
            candidate_count=len(result.candidates),
            fallback=fallback,
            response_source=(
                "semantic_reply_plan" if not fallback else "semantic_reply_plan_fallback"
            ),
            semantic_reply_plan_id=plan.plan_id,
            processing_outcome=result.outcome,
        )

    async def prepare_bound_explanation(self, event: Event) -> PreparedReply | None:
        if self._productive_only_replies or self._open_questions is None:
            return None
        resolution = await self._open_questions.resolution_for_event(event.event_id)
        if resolution is None:
            return None
        text = resolution.confirmation
        return PreparedReply(
            text=text,
            intent_id=f"contextual-learning:{resolution.question_id}",
            tokens=tokenize(text),
            genome_signature=self._expression_engine.genome.signature,
            candidate_count=0,
            contextual_learning=True,
            response_source="contextual_learning",
        )

    async def prepare_bound_short_act(self, event: Event) -> PreparedReply | None:
        if self._productive_only_replies or self._short_speech_acts is None:
            return None
        resolution = await self._short_speech_acts.resolution_for_event(event.event_id)
        if resolution is None:
            return None
        text = resolution.reply_text()
        if text is None:
            return None
        return PreparedReply(
            text=text,
            intent_id=f"short-act:{resolution.event_id}",
            tokens=tokenize(text),
            genome_signature=self._expression_engine.genome.signature,
            candidate_count=0,
            contextual_learning=True,
            response_source="short_act",
        )

    async def prepare_initiative(self, event: Event) -> PreparedReply | None:
        intent = await self._repository.intent_for_event(event.event_id)
        if intent is None:
            return None
        decision = await self._initiative_policy.evaluate(event, intent)
        if not decision.should_initiate or decision.goal_id is None:
            return None
        context = await self._reply_context(event, intent)
        if context.exact_answer is None:
            curiosity_question = await self._curiosity_question(event)
            if curiosity_question:
                context = replace(context, exact_answer=curiosity_question)
        thought = StructuredThought(
            action_type=ActionType.ASK_QUESTION,
            target_id=event.actor_id,
            objective="reduce uncertainty through voluntary inquiry",
            confidence=intent.confidence,
            evidence=(*intent.evidence, decision.goal_id),
        )
        plan = utterance_plan(thought, context=context)
        if plan is None:
            return None
        text = await self._generate_checked(event, plan)
        if text and not text.rstrip().endswith("?"):
            repair = repair_for_utterance(
                analyze_structured_utterance(context.prompt),
                analyze_pragmatics(context.prompt),
            )
            text = repair.realize() or text
        if not text or context.personality is None:
            return None
        return PreparedReply(
            text=text,
            intent_id=intent.intent_id,
            tokens=tokenize(text),
            genome_signature=context.personality.genome_signature,
            candidate_count=context.personality.candidate_count,
            initiative=True,
            goal_id=decision.goal_id,
            initiative_drive=decision.drive,
            initiative_reasoning=decision.reasoning,
            response_source="initiative_graph",
        )

    async def record_sent(
        self,
        prompt: Event,
        *,
        response_message_id: int,
        reply: PreparedReply,
        created_at: datetime,
    ) -> None:
        if self._assertion_experiences is not None and reply.assertion_experience is not None:
            await self._assertion_experiences.record_sent(
                prompt, response_message_id=response_message_id,
                proposal=reply.assertion_experience, created_at=created_at,
            )
        prompt_text = self._clean_content(prompt)
        if reply.processing_outcome is not None and self._processing_outcomes is not None:
            await self._processing_outcomes.save(
                prompt,
                reply.processing_outcome,
                response_message_id=response_message_id,
                created_at=created_at,
            )
        if reply.arbitration_decision_id is not None and self._arbitration_repository is not None:
            await self._arbitration_repository.mark_sent(
                reply.arbitration_decision_id,
                response_message_id=response_message_id,
                sent_at=created_at,
            )
        expressed_position = self._pending_positions.pop(prompt.event_id, None)
        if expressed_position is not None and self._position_reasoning.repository is not None:
            await self._position_reasoning.repository.record_expressed(
                expressed_position,
                guild_id=prompt.guild_id,
                source_event_id=prompt.event_id,
                candidate_id=reply.winner_candidate_id,
                response_message_id=response_message_id,
                observed_at=created_at,
            )
        if is_identity_question(prompt_text):
            _identity_logger.info(
                "event=%s guild=%s channel=%s actor=%s prompt=%r response=%r",
                prompt.event_id,
                prompt.guild_id,
                prompt.channel_id,
                prompt.actor_id,
                _one_line(prompt_text),
                _one_line(reply.text),
            )
        pending_concept = reply.pending_concept or lexical_question_target(reply.text)
        if (pending_concept is not None and self._open_questions is not None
                and requests_word_definition(reply.text)):
            await self._open_questions.open_for(
                prompt,
                concept=pending_concept,
                question_message_id=response_message_id,
                asked_at=created_at,
            )
        if (
            reply.fallback
            or reply.contextual_learning
            or not evaluate_surface(reply.text).learnable
        ):
            return
        await self._experience_consolidator.register_sent(
            prompt,
            response_message_id=response_message_id,
            reply=reply,
            created_at=created_at,
        )
        if self._self_dataset is not None:
            await self._self_dataset.capture_exchange(
                prompt,
                response_message_id=response_message_id,
                reply=reply,
                created_at=created_at,
            )
        if reply.initiative:
            await self._initiative_policy.autonomy.start_goal(
                prompt,
                response_message_id=response_message_id,
                reply=reply,
                created_at=created_at,
            )

    async def _reply_context(self, event: Event, intent: ActionIntent) -> ReplyContext:
        prompt = self._clean_content(event)
        personality = self._expression_engine.express(intent)
        if event.actor_id is None or event.channel_id is None:
            return ReplyContext(
                prompt=prompt,
                seed=_stable_response_seed(prompt, scope=str(event.guild_id)),
                personality=personality,
                perspective=build_personality_perspective(personality),
            )
        (
            working_events,
            reply_heads,
            memory_ngrams,
            experience_transitions,
            discourse,
            semantic_facts,
            knowledge_grounding,
            reaction,
            relation,
            reflection,
            literacy_facts,
            conversation_summaries,
            long_term_summaries,
            usage_hints,
            semantic_lesson_guidance,
        ) = await asyncio.gather(
            self._memory.relevant_context(
                event.guild_id,
                channel_id=event.channel_id,
                user_id=event.actor_id,
                prompt=prompt,
            ),
            self._archive_repository.matching_reply_heads(event.guild_id, tokenize(prompt)),
            self._archive_repository.common_ngrams(event.guild_id),
            self._experience_consolidator.transition_adjustments(event.guild_id),
            self._discourse_context(event, prompt),
            self._semantic_context(event, prompt),
            self._knowledge_context(event, prompt),
            self._reaction_context(event),
            self._relation_context(event),
            self._reflection_context(event),
            self._literacy_context(prompt),
            self._conversation_context(event, prompt),
            self._long_term_context(event, prompt),
            self._usage_context(event, prompt),
            self._semantic_lesson_context(event, prompt),
        )
        # A deferred reply still interprets the original turn. Later messages
        # must not become antecedents merely because dictionary lookup took time.
        working_events = tuple(item for item in working_events
                               if item.utc_timestamp <= event.utc_timestamp)
        recent_working_messages = self._recent_texts(
            (item for item in working_events if item.event_id != event.event_id),
            limit=self._memory.limits.max_context_events,
        )
        recent_dialogue_messages = self._recent_texts(
            (item for item in working_events if item.event_id != event.event_id),
            limit=self._memory.limits.max_context_events,
            include_bot=True,
        )
        consultation_messages = self._recent_consultation_turns(
            (item for item in working_events if item.event_id != event.event_id),
            limit=12,
        )
        dialogue_chain = tuple(discourse.messages[: discourse.reply_depth])
        dialogue_messages = (
            *(
                message
                for message in recent_dialogue_messages[-12:]
                if message not in dialogue_chain
            ),
            *dialogue_chain,
        )
        dialogue_state = analyze_dialogue_state(prompt, dialogue_messages)
        dialogue_state = replace(
            dialogue_state, exchange=exchange_from_events(tuple(working_events), event)
        )
        assertion_memory = build_assertion_memory(tuple(working_events), event)
        assertion_continuation = plan_assertion_continuation(
            prompt, assertion_memory, tuple(working_events), event
        )
        assertion_feedback_counts = (0, 0)
        if assertion_continuation is not None and self._assertion_experiences is not None:
            assertion_continuation, guidance = await self._assertion_experiences.adapt(
                event, prompt, assertion_continuation
            )
            assertion_feedback_counts = (guidance.confirmations, guidance.rejections)
        knowledge_facts = knowledge_grounding.facts
        knowledge_answer = knowledge_grounding.exact_answer
        reference_definitions = knowledge_grounding.definitions
        if dialogue_state.added_context_terms:
            resolved_semantic, resolved_grounding = await asyncio.gather(
                self._semantic_context(event, dialogue_state.retrieval_query),
                self._knowledge_context(event, dialogue_state.retrieval_query),
            )
            resolved_facts = resolved_grounding.facts
            resolved_answer = resolved_grounding.exact_answer
            reference_definitions = tuple(dict.fromkeys(
                (*reference_definitions, *resolved_grounding.definitions)
            ))
            semantic_facts = tuple(dict.fromkeys((*semantic_facts, *resolved_semantic)))
            knowledge_facts = tuple(dict.fromkeys((*knowledge_facts, *resolved_facts)))
            knowledge_answer = knowledge_answer or resolved_answer
        _dialogue_state_logger.info(
            "event=%s transition=%s reason=%s active=%s references=%s "
            "added_terms=%s repair_target=%r retrieval_query=%r confidence=%.3f",
            event.event_id,
            dialogue_state.topic.transition,
            dialogue_state.topic.reason,
            dialogue_state.topic.active_concepts,
            tuple(
                (
                    item.kind,
                    item.surface,
                    item.chosen.lemma if item.chosen else None,
                    tuple(candidate.lemma for candidate in item.alternatives),
                    item.reason,
                )
                for item in dialogue_state.references
            ),
            dialogue_state.added_context_terms,
            dialogue_state.repair_target,
            dialogue_state.retrieval_query,
            dialogue_state.confidence,
        )
        prompt_content_terms = set(content_lemmas(prompt))
        semantic_facts = tuple(
            fact for fact in semantic_facts if not set(content_lemmas(fact)) <= prompt_content_terms
        )
        required_facts = (
            *(fact for fact in knowledge_facts if not unrequested_definition(
                fact, dialogue_state.repair_target or prompt, reference_definitions
            )),
            *(fact.text for fact in literacy_facts),
        )
        exact_answer = knowledge_answer or next(
            (fact.text for fact in literacy_facts if fact.exact), None
        )
        if exact_answer and unrequested_definition(
            exact_answer, dialogue_state.repair_target or prompt, reference_definitions
        ):
            exact_answer = None
        perspective = build_personality_perspective(
            personality,
            reaction=reaction,
            relation=relation,
            reflection=reflection,
        )
        return ReplyContext(
            prompt=prompt,
            reference_definitions=reference_definitions,
            memory_messages=(
                *recent_working_messages,
                *conversation_summaries,
                *long_term_summaries,
            ),
            dialogue_messages=dialogue_messages,
            consultation_messages=consultation_messages,
            memory_ngrams=memory_ngrams,
            experience_transitions=experience_transitions,
            reply_heads=(*discourse.reply_heads, *reply_heads),
            knowledge_facts=(
                *knowledge_facts,
                *(fact.text for fact in literacy_facts),
                *semantic_facts,
                *semantic_lesson_guidance.facts,
            ),
            required_facts=required_facts,
            seed=_stable_response_seed(
                prompt,
                dialogue_chain,
                (*required_facts, *((exact_answer,) if exact_answer else ())),
                scope=f"{event.guild_id}:{event.channel_id}",
            ),
            personality=personality,
            perspective=perspective,
            reaction=reaction,
            relation=relation,
            reflection=reflection,
            exact_answer=exact_answer,
            usage_hints=usage_hints,
            learned_syntax_patterns=semantic_lesson_guidance.syntax_patterns,
            learned_constructions=semantic_lesson_guidance.constructions,
            dependency_patterns=semantic_lesson_guidance.dependency_patterns,
            expected_response_acts=semantic_lesson_guidance.expected_response_acts,
            learned_registers=semantic_lesson_guidance.registers,
            social_policies=semantic_lesson_guidance.social_policies,
            dialogue_state=dialogue_state,
            assertion_memory=assertion_memory,
            assertion_continuation=assertion_continuation,
            assertion_feedback_counts=assertion_feedback_counts,
        )

    async def _usage_context(self, event: Event, prompt: str) -> tuple[UsageGuidance, ...]:
        if self._usage_learning is None:
            return ()
        return await self._usage_learning.guidance_for_text(event.guild_id, prompt)

    async def _semantic_lesson_context(self, event: Event, prompt: str) -> SemanticLessonGuidance:
        if self._semantic_assimilation is None:
            return SemanticLessonGuidance()
        return await self._semantic_assimilation.guidance_for_prompt(event.guild_id, prompt)

    async def _generate_checked(self, event: Event, utterance) -> str:
        dialogue_state = utterance.context.dialogue_state
        if (
            dialogue_state is not None
            and dialogue_state.repair_target is not None
            and response_requirement(utterance.context.prompt) is None
        ):
            repair_target = dialogue_state.repair_target
            utterance = replace(
                utterance,
                context=replace(
                    utterance.context,
                    prompt=repair_target,
                    seed=_stable_response_seed(
                        repair_target,
                        utterance.context.dialogue_messages,
                        utterance.context.required_facts,
                        scope=f"{event.guild_id}:{event.channel_id}:repair",
                    ),
                ),
            )
        if self._productive_only_replies or self._arbitration_mode == "active":
            return await self._generate_arbitrated(event, utterance, mode="active")
        legacy_text = await self._generate_legacy_checked(event, utterance)
        if self._arbitration_mode == "shadow":
            legacy_source = self._candidate_sources.get(event.event_id, "graph")
            try:
                await self._generate_arbitrated(
                    event,
                    utterance,
                    mode="shadow",
                    legacy_text=legacy_text,
                    legacy_source=legacy_source,
                    remember=False,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.getLogger("debilgi.response_arbitration").exception(
                    "Shadow arbitration failed for event=%s", event.event_id
                )
            finally:
                self._candidate_sources[event.event_id] = legacy_source
                self._arbitration_decisions.pop(event.event_id, None)
        return legacy_text

    async def _generate_legacy_checked(self, event: Event, utterance) -> str:
        self._remember_candidate_source(
            event.event_id,
            "retrieval" if utterance.context.exact_answer is not None else "graph",
        )
        resolved_identity = identity_reply(utterance.context.prompt, event.payload)
        if resolved_identity is not None:
            self._remember_candidate_source(event.event_id, "discord_identity")
            return resolved_identity
        entity_analysis = analyze_entities(utterance.context.prompt, event.payload)
        analyses = await self._word_analyses(event, utterance.context.prompt)
        resolved_bindings = (
            utterance.context.dialogue_state.resolved_bindings
            if utterance.context.dialogue_state is not None
            else ()
        )
        initial_understanding = understand_utterance(
            utterance.context.prompt,
            analyses,
            utterance.context.dialogue_messages,
            entity_analysis,
            resolved_bindings,
        )
        analyses = await self._resolve_unknowns_autonomously(
            event,
            utterance.context.prompt,
            analyses,
            initial_understanding,
        )
        understanding = understand_utterance(
            utterance.context.prompt,
            analyses,
            utterance.context.dialogue_messages,
            entity_analysis,
            resolved_bindings,
        )
        await self._save_understanding(event, understanding)
        self._log_structured_analysis(
            event,
            utterance.context.prompt,
            analyses,
            entity_analysis,
        )
        await self._queue_unknowns(
            event,
            analyses,
            protected_terms=entity_analysis.identity_terms,
        )
        evidence = RecoveryEvidence(
            prompt=utterance.context.prompt,
            analyses=analyses,
            context_messages=utterance.context.dialogue_messages,
        )
        unknown_messages = (
            (self._phrase_uncertainty(understanding),)
            if understanding.unknown_terms and not understanding.sufficient
            else ()
        )
        stored_plan = await self._create_response_plan(event, utterance, analyses)
        recent_self_replies = await self._recent_self_replies(event)
        cached_reply = self._stable_reply_cache.get(utterance.context.seed)
        if cached_reply is not None:
            self._stable_reply_cache.move_to_end(utterance.context.seed)
            if stored_plan is not None:
                validation = validate_response(stored_plan, cached_reply)
                await self._response_plans.complete(stored_plan, validation)  # type: ignore[union-attr]
            return cached_reply
        direct_reply = compose_dialogue_reply(evidence)
        lexical_reply = (
            await self._lexical_reference_reply(utterance.context.prompt, analyses)
            if direct_reply is None
            else None
        )
        if (
            direct_reply is not None
            or lexical_reply is not None
            or is_social_routine(utterance.context.prompt)
        ):
            self._remember_candidate_source(
                event.event_id,
                "retrieval" if lexical_reply is not None else "dialogue_rule",
            )
            text = direct_reply or lexical_reply or compose_recovery(evidence)
            if stored_plan is not None:
                validation = validate_response(stored_plan, text)
                await self._response_plans.complete(stored_plan, validation)  # type: ignore[union-attr]
            self._remember_stable_reply(utterance.context.seed, text)
            return text
        has_low_confidence_unknown = any(
            item.unknown_kind in {UnknownKind.UNKNOWN_FORM, UnknownKind.UNKNOWN_MEANING}
            and item.response_confidence < self._response_confidence_threshold
            for item in analyses
        )
        if has_low_confidence_unknown and not understanding.sufficient:
            self._remember_candidate_source(event.event_id, "fallback")
            text = await self._recover_response(
                event,
                utterance,
                analyses,
                understanding=understanding,
            )
            if stored_plan is not None:
                validation = validate_response(stored_plan, text)
                await self._response_plans.complete(stored_plan, validation)  # type: ignore[union-attr]
            self._remember_stable_reply(utterance.context.seed, text)
            return text
        validation = None
        text = ""
        surface_issue: str | None = None
        for attempt in range(self._response_plan_retry_limit + 1):
            attempt_context = replace(
                utterance.context,
                seed=f"{utterance.context.seed}|response-plan-attempt:{attempt}",
            )
            text = self._generator.generate(replace(utterance, context=attempt_context))
            if utterance.context.exact_answer is None:
                text = self._trim_unrelated_tail(text, utterance.context.prompt)
            surface_issue = self._surface_quality_issue(
                text,
                prompt=utterance.context.prompt,
                recent_self_replies=recent_self_replies,
            )
            if surface_issue is None:
                surface_issue = self._topic_continuity_issue(
                    text,
                    prompt=utterance.context.prompt,
                    dialogue_messages=utterance.context.dialogue_messages,
                    required_facts=utterance.context.required_facts,
                )
            if surface_issue is not None:
                if surface_issue == "topic_discontinuity":
                    self._log_topic_rejection(event, utterance.context, text)
                continue
            if stored_plan is None:
                return text
            validation = validate_response(stored_plan, text)
            if validation.valid:
                break
        if stored_plan is None:
            if surface_issue is None:
                self._remember_stable_reply(utterance.context.seed, text)
                return text
            text = await self._recover_response(
                event,
                utterance,
                analyses,
                understanding=understanding,
            )
            self._remember_candidate_source(event.event_id, "fallback")
            self._remember_stable_reply(utterance.context.seed, text)
            return text
        if stored_plan is not None and (validation is None or not validation.valid):
            text = await self._recover_response(
                event,
                utterance,
                analyses,
                understanding=understanding,
            )
            self._remember_candidate_source(event.event_id, "fallback")
            validation = validate_response(stored_plan, text)
        if (
            self._surface_quality_issue(
                text,
                prompt=utterance.context.prompt,
                recent_self_replies=recent_self_replies,
            )
            is not None
        ):
            text = self._fallback_for_plan(
                stored_plan,
                unknown_messages,
                prompt=utterance.context.prompt,
                dialogue_messages=utterance.context.dialogue_messages,
            )
            self._remember_candidate_source(event.event_id, "fallback")
            validation = validate_response(stored_plan, text)
        if (
            self._topic_continuity_issue(
                text,
                prompt=utterance.context.prompt,
                dialogue_messages=utterance.context.dialogue_messages,
                required_facts=utterance.context.required_facts,
            )
            is not None
        ):
            self._log_topic_rejection(event, utterance.context, text)
            text = self._fallback_for_plan(
                stored_plan,
                unknown_messages,
                prompt=utterance.context.prompt,
                dialogue_messages=utterance.context.dialogue_messages,
            )
            self._remember_candidate_source(event.event_id, "fallback")
            validation = validate_response(stored_plan, text)
        if stored_plan is not None and validation is not None:
            await self._response_plans.complete(stored_plan, validation)  # type: ignore[union-attr]
        self._remember_stable_reply(utterance.context.seed, text)
        return text

    async def _generate_arbitrated(
        self,
        event: Event,
        utterance,
        *,
        mode: str,
        legacy_text: str | None = None,
        legacy_source: str | None = None,
        remember: bool = True,
    ) -> str:
        """Collect applicable sources before selecting a cross-source winner."""

        started = perf_counter()
        reference_definitions = list(utterance.context.reference_definitions)
        prompt = utterance.context.prompt
        entity_analysis = analyze_entities(prompt, event.payload)
        analyses = await self._word_analyses(event, prompt)
        resolved_bindings = (
            utterance.context.dialogue_state.resolved_bindings
            if utterance.context.dialogue_state is not None
            else ()
        )
        initial_understanding = understand_utterance(
            prompt,
            analyses,
            utterance.context.dialogue_messages,
            entity_analysis,
            resolved_bindings,
        )
        analyses = await self._resolve_unknowns_autonomously(
            event,
            prompt,
            analyses,
            initial_understanding,
        )
        understanding = understand_utterance(
            prompt,
            analyses,
            utterance.context.dialogue_messages,
            entity_analysis,
            resolved_bindings,
        )
        if understanding.task is not None:
            task = understanding.task
            frontier = task.frontier
            await self._trace_step(
                event.event_id, stage="planning", component="comprehension_task",
                status="interpreted" if task.interpretation_ready else "incomplete",
                summary="interpretation dependencies are separate from answer evidence",
                details={
                    "goal": task.goal, "answer_slot": task.answer_slot,
                    "interpretation_ready": task.interpretation_ready,
                    "answer_ready": task.answer_ready, "pending": frontier.pending,
                    "lexical_states": task.lexical_states, "lookup_terms": task.lookup_terms,
                    "incidental_terms": task.incidental_terms,
                    "cycles": frontier.cycles, "limited": frontier.limited,
                },
            )
        structured_bundle = self._log_structured_analysis(
            event,
            prompt,
            analyses,
            entity_analysis,
        )
        composer = ResponseComposer(
            structured_bundle[0] if structured_bundle else analyze_structured_utterance(prompt)
        )
        composer.deferred = self._productive_only_replies or self._autonomous_planning_required
        spatial_dynamics_resolution = (
            resolve_boundary_entry_question(structured_bundle[0])
            if structured_bundle is not None
            else None
        )
        spatial_dynamics_explanation = (
            resolve_boundary_entry_explanation(
                structured_bundle[0],
                utterance.context.dialogue_messages,
            )
            if structured_bundle is not None
            else None
        )
        quantitative_resolution = (
            resolve_travel_time_question(structured_bundle[0])
            if structured_bundle is not None
            else None
        )
        abstract_model = build_mental_model(
            prompt,
            guild_id=event.guild_id,
            channel_id=event.channel_id,
            source_event_id=event.event_id,
            context_messages=utterance.context.dialogue_messages,
        )
        if abstract_model is not None and self._abstract_reasoning is not None and remember:
            await self._abstract_reasoning.remember(
                abstract_model,
                observed_at=event.utc_timestamp,
            )
        abstract_resolution = resolve_abstract_question(abstract_model)
        if structured_bundle is not None and self._semantic_assimilation is not None and remember:
            await self._semantic_assimilation.observe_local_structure(
                guild_id=event.guild_id,
                structure=structured_bundle[0],
                observed_at=event.utc_timestamp,
            )
        response_meaning_plan = None
        meaning_plan_surface = None
        semantic_question = (
            analyze_semantic_question(structured_bundle[0])
            if structured_bundle is not None
            else None
        )
        relation_question = (
            analyze_relation_question(structured_bundle[0])
            if structured_bundle is not None
            else None
        )
        replied_to_agent = bool(
            event.payload.get("reply_to_message_id") is not None
            and isinstance(event.payload.get("bot_user_id"), int)
            and event.target_id == event.payload.get("bot_user_id")
        )
        self_state_frame = (
            analyze_self_state(
                structured_bundle[0],
                resolved_bindings=resolved_bindings,
                replied_to_agent=replied_to_agent,
                context_messages=utterance.context.dialogue_messages,
            )
            if structured_bundle is not None
            else None
        )
        personal_action_frame = (
            analyze_personal_action(structured_bundle[0]) if structured_bundle is not None else None
        )
        runtime_state_frame = (
            analyze_runtime_state(structured_bundle[0]) if structured_bundle is not None else None
        )
        causal_query = analyze_causal_query(
            prompt,
            utterance.context.dialogue_messages,
        )
        causal_inference = (
            await self._causal_reasoning.infer(event.guild_id, causal_query)
            if self._causal_reasoning is not None and causal_query is not None
            else None
        )
        pragmatics = analyze_pragmatics(prompt)
        dialogue_act = (
            analyze_dialogue_act(
                structured_bundle[0],
                pragmatics=pragmatics,
                reaction=utterance.context.reaction,
                semantic_question=semantic_question,
                relation_question=relation_question,
                self_state=self_state_frame,
            )
            if structured_bundle is not None
            else None
        )
        explicit_lexical_lookup = lexical_lookup_target(prompt) is not None
        for word in analyses:
            title = word.lemma or word.normalized_form
            for meaning in word.dictionary_meanings:
                reference_definitions.extend(((title, meaning), (title, f"{title} — {meaning}")))
        meaning_utterance = utterance
        if not explicit_lexical_lookup:
            def unsuitable_meaning_fact(fact: str) -> bool:
                return unrequested_definition(fact, prompt, tuple(reference_definitions)) or (
                    (head := definition_head(fact)) is not None
                    and not definition_requested(prompt, head)
                )

            meaning_context = replace(
                utterance.context,
                exact_answer=(
                    None
                    if utterance.context.exact_answer
                    and unsuitable_meaning_fact(utterance.context.exact_answer)
                    else utterance.context.exact_answer
                ),
                required_facts=tuple(
                    fact
                    for fact in utterance.context.required_facts
                    if not unsuitable_meaning_fact(fact)
                ),
            )
            meaning_utterance = replace(utterance, context=meaning_context)
        response_meaning_plan = (
            build_response_meaning_plan(
                meaning_utterance,
                structured_bundle[0],
                structured_bundle[1],
            )
            if structured_bundle is not None
            else None
        )
        meaning_plan_surface = (
            realize_response_meaning(
                response_meaning_plan, allow_rough=self._productive_only_replies,
            )
            if response_meaning_plan is not None
            else None
        )
        await self._trace_step(
            event.event_id,
            stage="interpretation",
            component="language_understanding",
            status="completed",
            summary="structure, pragmatics and conversational intent were interpreted",
            details={
                "understanding": {
                    "content_terms": understanding.content_terms,
                    "known_terms": understanding.known_terms,
                    "unknown_terms": understanding.unknown_terms,
                    "blocking_unknown_terms": understanding.blocking_unknown_terms,
                    "referential_unknown_terms": understanding.referential_unknown_terms,
                    "nonblocking_unknown_terms": understanding.nonblocking_unknown_terms,
                    "context_links": understanding.context_links,
                    "comprehension_score": understanding.comprehension_score,
                    "sufficient": understanding.sufficient,
                    "task": understanding.task,
                },
                "structure": (
                    {
                        "confidence": structured_bundle[0].confidence,
                        "tokens": tuple(
                            {
                                "surface": token.surface,
                                "lemma": token.lemma,
                                "part_of_speech": token.part_of_speech,
                                "semantic_role": token.semantic_role,
                                "confidence": token.confidence,
                            }
                            for token in structured_bundle[0].tokens
                        ),
                        "clauses": tuple(
                            {
                                "kind": clause.clause_kind,
                                "subject_indices": clause.subject_indices,
                                "predicate_index": clause.predicate_index,
                                "predicate_complement_indices": (
                                    clause.predicate_complement_indices
                                ),
                                "question_target_index": clause.question_target_index,
                                "implicit_subject": clause.implicit_subject,
                                "implicit_predicate": clause.implicit_predicate,
                                "elliptical": clause.elliptical,
                                "confidence": clause.confidence,
                            }
                            for clause in structured_bundle[0].clauses
                        ),
                        "ambiguities": structured_bundle[0].ambiguities,
                    }
                    if structured_bundle is not None
                    else None
                ),
                "pragmatic_cues": tuple(
                    {
                        "surface": cue.surface,
                        "category": cue.category,
                        "function": cue.function,
                        "confidence": cue.confidence,
                        "intensity": cue.intensity,
                    }
                    for cue in pragmatics.cues
                ),
                "dialogue_act": (
                    {
                        "kind": dialogue_act.kind,
                        "response_obligation": dialogue_act.response_obligation,
                        "subject": dialogue_act.subject,
                        "predicate": dialogue_act.predicate,
                        "objects": dialogue_act.objects,
                        "register": dialogue_act.register,
                        "complete": dialogue_act.complete,
                        "confidence": dialogue_act.confidence,
                        "evidence": dialogue_act.evidence,
                    }
                    if dialogue_act is not None
                    else None
                ),
                "personality_perspective": (
                    utterance.context.perspective.trace()
                    if utterance.context.perspective is not None
                    else None
                ),
                "self_state_frame": self_state_frame,
                "personal_action_frame": personal_action_frame,
                "runtime_state_frame": runtime_state_frame,
                "meaning_plan": {
                    "renderable": (
                        response_meaning_plan.renderable
                        if response_meaning_plan is not None
                        else False
                    ),
                    "required_facts": (
                        response_meaning_plan.required_facts
                        if response_meaning_plan is not None
                        else ()
                    ),
                    "surface": meaning_plan_surface,
                },
            },
        )
        continuous_facts = (
            await self._continuous_cognition.relevant_facts(
                event.guild_id,
                content_lemmas(prompt),
            )
            if self._continuous_cognition is not None
            else ()
        )
        internal_facts = tuple(
            dict.fromkeys(
                (
                    *utterance.context.required_facts,
                    *utterance.context.knowledge_facts,
                    *continuous_facts,
                )
            )
        )
        communicative_intention = (
            choose_communicative_intention(
                dialogue_act,
                utterance.context.social_policies,
                grounding_available=bool(internal_facts),
            )
            if dialogue_act is not None
            else None
        )
        semantic_inquiry = (
            analyze_semantic_inquiry(
                structured_bundle[0],
                knowledge_facts=internal_facts,
                exploration=(
                    utterance.context.perspective.exploration
                    if utterance.context.perspective is not None
                    else 0.5
                ),
            )
            if structured_bundle is not None
            else None
        )
        semantic_inquiry_surfaces = (
            realize_semantic_inquiry(
                semantic_inquiry,
                utterance.context.dependency_patterns,
            )
            if semantic_inquiry is not None
            else ()
        )
        if semantic_inquiry is not None:
            await self._trace_step(
                event.event_id,
                stage="inquiry",
                component="semantic_inquiry",
                status=("concluded" if semantic_inquiry.renderable else "decomposed"),
                summary=(
                    "semantic roles were decomposed into explicit answers, "
                    "inferences, hypotheses and unresolved gaps"
                ),
                details=inquiry_trace(semantic_inquiry),
            )
        dialogue_move_plans = (
            plan_dialogue_moves(
                dialogue_act,
                self_state=self_state_frame,
                intention=communicative_intention,
                personality=utterance.context.personality,
                perspective=utterance.context.perspective,
                reaction=utterance.context.reaction,
                structure=(structured_bundle[0] if structured_bundle is not None else None),
                prompt=prompt,
                context_messages=utterance.context.dialogue_messages,
                exchange=(utterance.context.dialogue_state.exchange
                          if utterance.context.dialogue_state is not None else None),
                resolve_from_context=False,
            )
            if self._productive_only_replies
            else ()
        )
        position_decision = await self._position_reasoning.resolve(
            event.guild_id,
            dialogue_act,
            facts=internal_facts,
            context_messages=utterance.context.dialogue_messages,
            perspective=utterance.context.perspective,
        )
        position_surfaces = (
            realize_position(
                position_decision,
                utterance.context.dependency_patterns,
                composer=composer,
            )
            if position_decision is not None
            else ()
        )
        dialogue_act_surface = (
            realize_dialogue_act(dialogue_act)
            if dialogue_act is not None
            and dialogue_act.complete
            and not self._productive_only_replies
            else None
        )
        internal_reasoning = (
            reason_from_internal_knowledge(
                semantic_question,
                analyses,
                internal_facts,
            )
            if semantic_question is not None and semantic_question.complete
            else None
        )
        relation_reasoning = (
            reason_about_relation(relation_question, analyses)
            if relation_question is not None and relation_question.complete
            else None
        )
        semantic_study_terms = (
            semantic_question.study_terms
            if internal_reasoning is not None and internal_reasoning.needs_study
            else ()
        )
        relation_study_terms = (
            relation_question.study_terms
            if relation_reasoning is not None and relation_reasoning.needs_study
            else ()
        )
        lookup_requested, policy_lookup_target = KnowledgeQueryPolicy.followup_query(prompt)
        foreground_lookup_target = lexical_lookup_target(prompt) or policy_lookup_target
        if (
            lookup_requested
            and foreground_lookup_target is None
            and utterance.context.dialogue_state is not None
        ):
            foreground_lookup_target = next(
                (
                    lemma
                    for role, lemma in utterance.context.dialogue_state.resolved_bindings
                    if role == "direct_object"
                ),
                None,
            )
        lexical_study_terms = tuple(
            dict.fromkeys(
                item.lemma or item.normalized_form
                for item in analyses
                if item.requires_lexical_study
                and (item.lemma or item.normalized_form)
                not in understanding.nonblocking_unknown_terms
            )
        )
        initial_capability_assessment = (
            assess_capability_question(
                structured_bundle[0],
                analyses=analyses,
                context_messages=utterance.context.dialogue_messages,
                knowledge_facts=internal_facts,
                perspective=utterance.context.perspective,
            )
            if structured_bundle is not None
            else None
        )
        capability_gap_terms = (
            capability_study_terms(structured_bundle[0])
            if structured_bundle is not None and initial_capability_assessment is None
            else ()
        )
        recalled_event = (
            recall_event_slot(prompt, utterance.context.assertion_memory, event)
            if utterance.context.assertion_memory is not None else None
        )
        study_terms = tuple(
            dict.fromkeys(
                (
                    *semantic_study_terms,
                    *relation_study_terms,
                    *lexical_study_terms,
                    *capability_gap_terms,
                    *((foreground_lookup_target,) if foreground_lookup_target is not None else ()),
                )
            )
        )
        if understanding.task is not None and not understanding.task.research_allowed:
            # A missing referent or speech act cannot be repaired by dictionary lookup.
            study_terms = ()
        exchange = (
            utterance.context.dialogue_state.exchange if utterance.context.dialogue_state else None
        )
        if foreground_lookup_target is None:
            study_terms = (() if recalled_event is not None else tuple(
                term for term in study_terms if term not in understanding.nonblocking_unknown_terms
            ))
        if (
            exchange is not None and exchange.status == "matched"
            and foreground_lookup_target is None
        ):
            # A compatible answer closes the conversational gap. Confirming its
            # interpretation does not require researching every word in that answer.
            study_terms = ()
        foreground_study_attempted = bool(
            study_terms
            and (self._knowledge_service is not None or self._semantic_consultant is not None)
        )
        studied_concepts = ()
        if study_terms and self._lexicon is not None:
            studied_concepts = (
                await self._knowledge_service.internalize_for_reasoning(
                    event,
                    study_terms,
                    limit=max(6, len(study_terms)),
                )
                if self._knowledge_service is not None
                else ()
            )
            reference_definitions.extend(
                (concept.title, concise_reference(concept.definition))
                for concept in studied_concepts
            )
            # Reference sources teach the local lexicon.  Re-run the same input
            # after that write so generation sees the newly internalized senses.
            analyses = await self._word_analyses(event, prompt)
            learned_terms = {
                item.lemma or item.normalized_form
                for item in analyses
                if item.dictionary_meanings
                or item.selected_meaning
                or item.local_meaning
                or item.personal_interpretation
            }
            learned_terms.update(
                term for concept in studied_concepts for term in content_lemmas(concept.title)
            )
            await self._lexicon.queue_semantic_study(
                event,
                tuple(term for term in study_terms if term not in learned_terms),
                reason=(
                    f"complete_{semantic_question.kind}_knowledge_gap"
                    if semantic_study_terms and semantic_question is not None
                    else f"complete_{relation_question.relation}_knowledge_gap"
                    if relation_study_terms and relation_question is not None
                    else "complete_lexical_knowledge_gap"
                ),
                priority=0.86,
            )
            understanding = understand_utterance(
                prompt,
                analyses,
                utterance.context.dialogue_messages,
                entity_analysis,
                resolved_bindings,
            )
            structured_bundle = self._log_structured_analysis(
                event,
                prompt,
                analyses,
                entity_analysis,
            )
            if structured_bundle is not None:
                composer.goal = reader_goal(structured_bundle[0])
            response_meaning_plan = (
                build_response_meaning_plan(
                    meaning_utterance,
                    structured_bundle[0],
                    structured_bundle[1],
                )
                if structured_bundle is not None
                else None
            )
            meaning_plan_surface = (
                realize_response_meaning(
                    response_meaning_plan, allow_rough=self._productive_only_replies,
                )
                if response_meaning_plan is not None
                else None
            )
            semantic_question = (
                analyze_semantic_question(structured_bundle[0])
                if structured_bundle is not None
                else semantic_question
            )
            relation_question = (
                analyze_relation_question(structured_bundle[0])
                if structured_bundle is not None
                else relation_question
            )
            self_state_frame = (
                analyze_self_state(
                    structured_bundle[0],
                    resolved_bindings=resolved_bindings,
                    replied_to_agent=replied_to_agent,
                    context_messages=utterance.context.dialogue_messages,
                )
                if structured_bundle is not None
                else self_state_frame
            )
            internal_facts = tuple(
                dict.fromkeys(
                    (
                        *internal_facts,
                        *(concept.definition for concept in studied_concepts),
                    )
                )
            )
            internal_reasoning = (
                reason_from_internal_knowledge(
                    semantic_question,
                    analyses,
                    internal_facts,
                    study_completed=True,
                )
                if semantic_question is not None and semantic_question.complete
                else None
            )
            relation_reasoning = (
                reason_about_relation(
                    relation_question,
                    analyses,
                    tuple((concept.title, concept.definition) for concept in studied_concepts),
                    study_completed=True,
                )
                if relation_question is not None and relation_question.complete
                else None
            )
        consulted_facts: tuple[str, ...] = ()
        research = (
            getattr(self._semantic_consultant, "research_lexeme", None)
            if self._semantic_consultant is not None
            else None
        )
        if study_terms and callable(research):
            covered_terms = {
                term
                for term in study_terms
                if any(_same_lexical_family(term, concept.title) for concept in studied_concepts)
                or any(
                    term in {item.lemma, item.normalized_form}
                    and bool(
                        item.dictionary_meanings
                        or item.selected_meaning
                        or item.local_meaning
                        or item.personal_interpretation
                    )
                    for item in analyses
                )
            }
            research_terms = tuple(term for term in study_terms if term not in covered_terms)[:3]
            researched = await asyncio.gather(
                *(
                    research(
                        event_id=event.event_id,
                        guild_id=event.guild_id,
                        channel_id=event.channel_id,
                        term=term,
                        context_messages=(
                            *utterance.context.consultation_messages[-7:],
                            f"user: {prompt}",
                        ),
                    )
                    for term in research_terms
                )
            )
            consulted_facts = tuple(
                dict.fromkeys(
                    fact
                    for facts in researched
                    for fact in facts
                    if fact.strip() and internal_surface_reason(fact) is None
                )
            )
            if consulted_facts:
                internal_facts = tuple(dict.fromkeys((*internal_facts, *consulted_facts)))
                persist = (
                    getattr(
                        self._knowledge_service,
                        "internalize_consulted_fact",
                        None,
                    )
                    if self._knowledge_service is not None
                    else None
                )
                if callable(persist):
                    for term, facts in zip(research_terms, researched, strict=True):
                        if not facts:
                            continue
                        await persist(
                            event,
                            query=term,
                            definition=facts[0],
                            confidence=0.68,
                        )
        self_state_resolution = (
            reason_about_self_state(
                self_state_frame,
                understanding,
                context_messages=utterance.context.dialogue_messages,
                knowledge_facts=internal_facts,
            )
            if self_state_frame is not None
            else None
        )
        personal_action_frame = (
            analyze_personal_action(structured_bundle[0]) if structured_bundle is not None else None
        )
        personal_action_resolution = reason_about_personal_action(personal_action_frame)
        repairs_runtime_answer = False
        runtime_state_frame = (
            analyze_runtime_state(structured_bundle[0]) if structured_bundle is not None else None
        )
        if runtime_state_frame is None and is_clarification_request(prompt):
            previous_user_prompt = _latest_user_consultation_prompt(
                utterance.context.consultation_messages
            )
            if previous_user_prompt is not None:
                runtime_state_frame = analyze_runtime_state(
                    analyze_structured_utterance(previous_user_prompt)
                )
                repairs_runtime_answer = runtime_state_frame is not None
        runtime_state_resolution = reason_about_runtime_state(
            runtime_state_frame,
            repairs_previous_answer=repairs_runtime_answer,
        )
        await self._save_understanding(event, understanding)
        capability_assessment = (
            assess_capability_question(
                structured_bundle[0],
                analyses=analyses,
                context_messages=utterance.context.dialogue_messages,
                knowledge_facts=internal_facts,
                study_completed=bool(capability_gap_terms and foreground_study_attempted),
                perspective=utterance.context.perspective,
            )
            if structured_bundle is not None and personal_action_resolution is None
            else None
        )
        capability_surfaces = (
            composer.express(
                capability_assessment.thought,
                (
                    *capability_assessment.patterns,
                    *utterance.context.dependency_patterns,
                ),
            )
            if capability_assessment is not None
            else ()
        )
        contextual_question = (
            reason_about_contextual_question(
                structured_bundle[0],
                utterance.context.dialogue_messages,
            )
            if structured_bundle is not None
            else None
        )
        contextual_question_surfaces = (
            composer.express(
                contextual_question.thought,
                (
                    *contextual_question.patterns,
                    *utterance.context.dependency_patterns,
                ),
            )
            if contextual_question is not None
            else ()
        )
        if contextual_question is not None:
            _contextual_question_logger.info(
                "event=%s kind=%s antecedent=%r targets=%s evidence=%s confidence=%.3f surfaces=%s",
                event.event_id,
                contextual_question.kind,
                contextual_question.antecedent,
                contextual_question.target_terms,
                contextual_question.evidence,
                contextual_question.confidence,
                tuple(item.text for item in contextual_question_surfaces),
            )
        dialogue_clarification = (
            realize_reference_clarification(utterance.context.dialogue_state)
            if utterance.context.dialogue_state is not None
            else None
        )
        if response_meaning_plan is not None:
            _meaning_logger.info(
                "event=%s propositions=%s renderable=%s confidence=%.3f "
                "selected_senses=%s unresolved=%s topic_transition=%s moves=%s "
                "discourse=%s rejected=%s surface=%r",
                event.event_id,
                len(response_meaning_plan.propositions),
                response_meaning_plan.renderable,
                response_meaning_plan.confidence,
                tuple(
                    (item.lemma, item.source, round(item.confidence, 3))
                    for item in response_meaning_plan.selected_senses
                ),
                response_meaning_plan.unresolved_sense_indices,
                response_meaning_plan.topic_transition,
                tuple(
                    (item.order, item.kind, item.proposition_index, item.optional)
                    for item in response_meaning_plan.response_moves
                ),
                tuple(
                    (
                        item.from_index,
                        item.to_index,
                        item.relation,
                        item.reference_strategy,
                    )
                    for item in response_meaning_plan.discourse_links
                ),
                tuple(
                    (item.source_fact, item.clause_index, item.reasons)
                    for item in response_meaning_plan.rejections
                ),
                meaning_plan_surface,
            )
        await self._queue_unknowns(
            event,
            analyses,
            protected_terms=entity_analysis.identity_terms,
        )
        if (
            remember
            and semantic_inquiry is not None
            and semantic_inquiry.study_terms
            and self._lexicon is not None
        ):
            await self._lexicon.queue_semantic_study(
                event,
                semantic_inquiry.study_terms,
                reason="semantic_inquiry_unresolved_slots",
                priority=0.64,
            )
        evidence = RecoveryEvidence(
            prompt=prompt,
            analyses=analyses,
            context_messages=utterance.context.dialogue_messages,
        )
        repair_target = (
            utterance.context.dialogue_state.repair_target
            if utterance.context.dialogue_state is not None
            else None
        )
        repair_surface = (
            compose_dialogue_reply(
                RecoveryEvidence(
                    prompt=repair_target,
                    analyses=analyses,
                    facts=utterance.context.required_facts,
                    context_messages=utterance.context.dialogue_messages,
                )
            )
            if repair_target is not None
            else None
        )
        stored_plan = (
            await self._create_response_plan(
                event,
                utterance,
                analyses,
                dialogue_act=dialogue_act,
            )
            if remember
            else None
        )
        recent_self_replies = await self._recent_self_replies(event)
        profile = (
            await self._arbitration_repository.profile(event.guild_id)
            if self._arbitration_repository is not None
            else ArbitrationProfile.baseline(event.guild_id)
        )
        structural_weights = (
            await self._arbitration_repository.structural_weights(event.guild_id)
            if self._arbitration_repository is not None else {}
        )
        candidates: list[ResponseCandidate] = []
        pending_content: list[tuple[PendingText, dict]] = []
        selected_message_key: str | None = None
        allowed_clarifications: tuple[str, ...] = ()
        suppressed_precomposed_sources: list[str] = []
        global_required_facts: tuple[str, ...] = ()

        def add(
            text: str | PendingText | None,
            *,
            source: str,
            renderer: str,
            confidence: float,
            required_facts: tuple[str, ...] = (),
            key_points: tuple[str, ...] = (),
            metadata: dict[str, object] | None = None,
        ) -> None:
            if isinstance(text, PendingText):
                pending_content.append((text, {
                    "source": source, "renderer": renderer, "confidence": confidence,
                    "required_facts": required_facts, "key_points": key_points,
                    "metadata": dict(metadata or {}),
                }))
                return
            if not text:
                return
            # Legacy strings cannot consume the selected meaning's realization
            # budget. Each pending plan already bounds its own surface variants.
            if (len(candidates) >= self._arbitration_max_total_candidates
                    and not composer.metadata_for(text)):
                return
            candidate_metadata = {**composer.metadata_for(text), **dict(metadata or {})}
            composed = bool(
                candidate_metadata.get("authored_reply_template") is False
                and (
                    candidate_metadata.get("role_order")
                    or candidate_metadata.get("proposition_count", 0) > 0
                )
            )
            candidate_metadata["locally_composed"] = composed
            candidate_metadata["allow_rough_surface"] = self._productive_only_replies and composed
            if unrequested_definition(text, prompt, tuple(reference_definitions)):
                candidate_metadata["hard_rejection"] = "lexical_definition_not_requested"
            locally_realized_semantics = bool(
                source in _AUTONOMOUS_LOCAL_PRODUCTIVE_SOURCES
                and candidate_metadata.get("authored_reply_template") is False
                and candidate_metadata.get("role_order")
                and candidate_metadata.get("semantic_scaffold") is not True
            )
            if (
                self._autonomous_planning_required
                and source != "autonomous_semantic_plan"
                and not locally_realized_semantics
            ):
                if source not in suppressed_precomposed_sources:
                    suppressed_precomposed_sources.append(source)
                return
            if source in _DIRECT_MEMORY_SURFACE_SOURCES and not _dialogue_act_requests_memory_fact(
                dialogue_act
            ):
                candidate_metadata["hard_rejection"] = "direct_memory_surface_not_requested"
            unbound_surface = source in {"graph", "grounded_surface", "cache"}
            if source == "grounded_surface":
                candidate_metadata.setdefault(
                    "surface_provenance",
                    "grounded_local_fact",
                )
                candidate_metadata.setdefault("reference_copy_penalty", 0.12)
            direct_surface_reuse = source == "cache" or (
                source == "graph" and renderer.startswith("legacy_generator")
            )
            if direct_surface_reuse and not _quotation_requested(prompt):
                candidate_metadata["hard_rejection"] = "verbatim_surface_reuse_requires_quotation"
                candidate_metadata["surface_provenance"] = "stored_complete_phrase"
            elif unbound_surface and causal_query is not None:
                candidate_metadata["hard_rejection"] = "causal_query_requires_causal_inference"
            elif unbound_surface and repair_target is not None:
                candidate_metadata["hard_rejection"] = "repair_request_requires_target_answer"
            elif unbound_surface and structured_bundle is not None:
                alignment = assess_response_alignment(structured_bundle[0], text)
                candidate_metadata["semantic_alignment"] = {
                    "expected_subjects": alignment.expected_subjects,
                    "candidate_subjects": alignment.candidate_subjects,
                    "question_focus": alignment.question_focus,
                    "matched_focus": alignment.matched_focus,
                }
                if not alignment.allowed:
                    candidate_metadata["hard_rejection"] = ",".join(alignment.reasons)
            if self._autonomous_planning_required:
                question_issue = self._question_response_issue(
                    text,
                    response_obligation=(
                        dialogue_act.response_obligation if dialogue_act is not None else None
                    ),
                    dialogue_messages=utterance.context.dialogue_messages,
                    recent_self_replies=recent_self_replies,
                    metadata=candidate_metadata,
                )
                if question_issue is not None:
                    candidate_metadata["hard_rejection"] = question_issue
            issue = self._surface_quality_issue(
                text,
                prompt=prompt,
                recent_self_replies=recent_self_replies,
                allow_rough=self._productive_only_replies and composed,
            )
            if (
                issue == "unexpected_valency_role"
                and not self._productive_only_replies
                and source == "semantic_synthesis"
                and candidate_metadata.get("foreground_research") is True
                and required_facts
            ):
                # A compound researched proposition can exceed the local
                # single-clause valency model.  Keep all other coherence gates,
                # but do not turn that parser limitation into "no data".
                issue = None
            if (
                issue == "partial_prompt_echo"
                and dialogue_act is not None
                and (
                    (
                        dialogue_act.response_obligation == "select_or_compare"
                        and source == "semantic_reasoning"
                    )
                    or (
                        dialogue_act.response_obligation == "express_position"
                        and source
                        in {
                            "autonomous_semantic_plan",
                            "semantic_reasoning",
                            "semantic_thought",
                            "position_reasoning",
                        }
                    )
                    or (
                        dialogue_act.response_obligation
                        in {"answer_question", "resolve_relation", "resolve_self_state"}
                        and source
                        in {
                            "autonomous_semantic_plan",
                            "dialogue_context",
                            "discord_identity",
                            "knowledge_retrieval",
                            "meaning_plan",
                            "quantitative_reasoning",
                            "dialogue_rule",
                            "semantic_inquiry",
                            "semantic_reasoning",
                            "abstract_reasoning",
                            "causal_reasoning",
                            "relation_reasoning",
                            "runtime_state",
                            "semantic_synthesis",
                            "semantic_thought",
                            "spatial_dynamics",
                        }
                    )
                )
            ):
                # Reusing the questioned entity is required grounding, not
                # parroting.  The echo gate remains strict for unbound graph
                # surfaces.
                issue = None
            if (
                issue in {"recent_exact_repeat", "recent_semantic_repeat"}
                and dialogue_act is not None
                and dialogue_act.response_obligation
                in {
                    "answer_question",
                    "resolve_relation",
                    "resolve_self_state",
                    "express_position",
                }
                and source
                in {
                    "autonomous_semantic_plan",
                    "dialogue_context",
                    "discord_identity",
                    "knowledge_retrieval",
                    "meaning_plan",
                    "quantitative_reasoning",
                    "semantic_inquiry",
                    "semantic_reasoning",
                    "abstract_reasoning",
                    "causal_reasoning",
                    "relation_reasoning",
                    "runtime_state",
                    "semantic_synthesis",
                    "semantic_thought",
                    "spatial_dynamics",
                    "position_reasoning",
                }
            ):
                # A repeated question may require the same factually correct
                # answer. Repetition remains a numeric penalty in arbitration,
                # but is not a hard semantic failure for a bound answer.
                issue = None
            if issue is not None:
                candidate_metadata["hard_rejection"] = issue
            effective_required_facts = tuple(
                dict.fromkeys((*required_facts, *global_required_facts))
            )
            if communicative_intention is not None:
                candidate_metadata.update(
                    candidate_social_features(
                        intention=communicative_intention,
                        source=source,
                        text=text,
                        metadata=candidate_metadata,
                        required_facts=effective_required_facts,
                        source_confidence=confidence,
                    )
                )
            perspective = utterance.context.perspective
            if perspective is not None:
                declared_move = str(
                    candidate_metadata.get("response_move")
                    or candidate_metadata.get("social_move")
                    or source
                )
                declared_personality_fit = candidate_metadata.get("personality_fit")
                personality_fit = (
                    float(declared_personality_fit)
                    if (
                        source == "autonomous_semantic_plan"
                        and isinstance(
                            declared_personality_fit,
                            int | float,
                        )
                        and not isinstance(declared_personality_fit, bool)
                    )
                    else perspective.candidate_fit(declared_move)
                )
                candidate_metadata["personality_fit"] = personality_fit
                candidate_metadata["personality_evidence"] = perspective.evidence
                prior_stance_fit = candidate_metadata.get("stance_consistency", 0.55)
                stance_fit = (
                    float(prior_stance_fit)
                    if isinstance(prior_stance_fit, int | float)
                    and not isinstance(prior_stance_fit, bool)
                    else 0.55
                )
                prior_social_fit = candidate_metadata.get("social_coherence", 0.55)
                social_fit = (
                    float(prior_social_fit)
                    if isinstance(prior_social_fit, int | float)
                    and not isinstance(prior_social_fit, bool)
                    else 0.55
                )
                candidate_metadata["stance_consistency"] = (
                    stance_fit * 0.72 + personality_fit * 0.28
                )
                candidate_metadata["social_coherence"] = social_fit * 0.78 + personality_fit * 0.22
            if dialogue_act is not None:
                candidate_metadata["response_obligation"] = dialogue_act.response_obligation
                obligation_fit = candidate_obligation_fit(
                    dialogue_act,
                    text,
                    candidate_metadata,
                )
                candidate_metadata["obligation_fit"] = obligation_fit
                prior_frame_fit = float(
                    candidate_metadata.get("semantic_frame_fit", obligation_fit)
                )
                candidate_metadata["semantic_frame_fit"] = (
                    prior_frame_fit * 0.65 + obligation_fit * 0.35
                )
                if not candidate_fulfills_obligation(
                    dialogue_act,
                    source,
                    text=text,
                    metadata=candidate_metadata,
                ):
                    mismatch = f"response_signal_mismatch:{dialogue_act.response_obligation}"
                    if source in _CONVERSATIONAL_REASONING_SOURCES:
                        # A conversational interpretation is a scoring signal,
                        # not a rigid answer mode. Keep locally reasoned moves
                        # in arbitration even when the coarse frame validator
                        # cannot recognize their exact wording.
                        candidate_metadata["response_signal_mismatch"] = mismatch
                    else:
                        candidate_metadata["hard_rejection"] = mismatch
            if (
                self._productive_only_replies
                and not composed
            ):
                rejection = "productive_only:surface_not_productively_realized"
                candidate_metadata["productive_only_rejection"] = rejection
                candidate_metadata.setdefault("hard_rejection", rejection)
            candidates.append(
                ResponseCandidate.create(
                    event_id=event.event_id,
                    source=source,
                    renderer=renderer,
                    text=text,
                    speech_act=utterance.speech_act,
                    source_confidence=confidence,
                    required_facts=effective_required_facts,
                    key_points=key_points,
                    metadata=candidate_metadata,
                )
            )

        def select_and_express_content() -> None:
            nonlocal selected_message_key
            ordered = sorted(pending_content, key=lambda item: composer.priority(
                item[0], item[1]["metadata"], item[1]["confidence"],
            ), reverse=True)
            pending_content.clear()
            for index, (request, parameters) in enumerate(ordered):
                if parameters["metadata"].get("hard_rejection"):
                    continue
                first = len(candidates)
                # This is the first point where this proposed meaning receives
                # words. A failed realization advances to the next meaning.
                for surface in composer.realize(request):
                    add(surface.text, **{
                        **parameters,
                        "confidence": min(parameters["confidence"], surface.confidence),
                        "renderer": parameters["renderer"].replace("planned", surface.source),
                        "metadata": {
                            **{key: surface.text if isinstance(value, PendingText) else value
                               for key, value in parameters["metadata"].items()},
                            "role_order": surface.role_order,
                            "construction_id": surface.construction_id,
                        },
                    })
                check = self._arbitrator.decide(
                    event_id=event.event_id, prompt=prompt, focus_concepts=(),
                    candidates=tuple(candidates[first:]), recent_replies=recent_self_replies,
                    profile=ArbitrationProfile.baseline(event.guild_id),
                    clarifiable_terms=allowed_clarifications,
                )
                if check.winner is not None:
                    selected_message_key = request.plan.key
                    composer.records.append({
                        "status": "content_selected", "plan_key": selected_message_key,
                        "unrealized_alternatives": len(ordered) - index - 1,
                    })
                    break

        premise_inference = (
            infer_from_premises(structured_bundle[0], analyses)
            if structured_bundle is not None else None
        )
        if premise_inference is not None:
            inference_surface = realize_premise_inference(premise_inference, composer=composer)
            if inference_surface is not None:
                add(
                    inference_surface.text, source="semantic_reasoning",
                    renderer=f"premise_chain:{premise_inference.kind}",
                    confidence=inference_surface.confidence,
                    key_points=(premise_inference.target,),
                    metadata={
                        "authored_reply_template": False,
                        "role_order": inference_surface.role_order,
                        "reasoning_evidence": premise_inference.evidence,
                        "conditional_assumption": premise_inference.assumption is not None,
                        "response_move": "answer", "semantic_frame_fit": 0.95,
                    },
                )
            await self._trace_step(
                event.event_id, stage="inquiry", component="premise_inference",
                status="concluded" if inference_surface else "unrealized",
                summary="lexical premises were connected to a response conclusion",
                details={"inference": premise_inference, "surface": inference_surface},
            )
        if recalled_event is not None:
            for surface in composer.express(recalled_event, limit=1):
                add(surface.text, source="semantic_reasoning", renderer="event_slot_recall",
                    confidence=0.94, metadata={
                        "authored_reply_template": False, "role_order": surface.role_order,
                        "reasoning_evidence": recalled_event.evidence,
                        "response_move": "answer", "truth_grounding": 0.94,
                        "semantic_frame_fit": 0.98, "claim_source": "user_report",
                    })
        definition_thought = (
            ruwordnet_answer_thought(prompt, analyses) or definition_answer_thought(
                prompt, tuple((item.lemma or item.normalized_form, item.semantic_relations)
                              for item in analyses),
            )
        )
        if definition_thought is not None:
            for surface in composer.express(definition_thought, limit=1):
                add(surface.text, source="semantic_thought", renderer="definition_relations",
                    confidence=surface.confidence,
                    metadata={"authored_reply_template": False, "role_order": surface.role_order,
                              "reasoning_evidence": definition_thought.evidence,
                              "semantic_frame_fit": 0.96, "truth_grounding": 0.9})

        continuation = utterance.context.assertion_continuation
        await self._trace_step(
            event.event_id, stage="reasoning", component="assertion_memory",
            status="resolved" if continuation is not None else "observed",
            summary="attributed claims and cross-turn links evaluated locally",
            details={
                "assertions": tuple({
                    "source_event_id": claim.source_event_id,
                    "speaker": claim.speaker,
                    "predicate": claim.thought.predicate.lemma,
                    "negated": claim.thought.negated,
                    "temporarily_bounded": claim.temporarily_bounded,
                    "parsing_confidence": claim.parsing_confidence,
                } for claim in utterance.context.assertion_memory.assertions),
                "links": tuple({
                    "kind": link.kind, "earlier_event_id": link.earlier_event_id,
                    "later_event_id": link.later_event_id,
                } for link in utterance.context.assertion_memory.links),
                "continuation_kind": continuation.kind if continuation else None,
                "feedback_counts": utterance.context.assertion_feedback_counts,
                "source_event_ids": continuation.source_event_ids if continuation else (),
            },
        )
        if continuation is not None:
            for surface in composer.express(continuation.thought, limit=1):
                add(
                    surface.text, source="dialogue_context",
                    renderer=f"assertion_memory:{continuation.kind}",
                    confidence=continuation.confidence,
                    metadata={
                        "authored_reply_template": False,
                        "role_order": surface.role_order,
                        "reasoning_evidence": continuation.thought.evidence,
                        "source_event_ids": continuation.source_event_ids,
                        "response_move": "check_temporal_change",
                        "context_relevance": 0.95,
                        "semantic_frame_fit": 0.92,
                        "hypothesis_only": True,
                        "assertion_experience": experience_proposal(prompt, continuation),
                    },
                )

        identity_text = identity_reply(prompt, event.payload)
        direct_reply = compose_dialogue_reply(evidence)
        preference_thought = _preference_question_thought(
            prompt,
            understanding,
            analyses=analyses,
            facts=(*internal_facts, *utterance.context.dialogue_messages[-8:]),
            perspective=utterance.context.perspective,
        )
        preference_reason = _preference_reason(
            prompt,
            understanding,
            analyses=analyses,
            facts=(*internal_facts, *utterance.context.dialogue_messages[-8:]),
            perspective=utterance.context.perspective,
        )
        social_routine = is_social_routine(prompt)
        identity_proposition = identity_thought(prompt, event.payload)
        if identity_proposition is not None:
            for surface in composer.express(identity_proposition, limit=1):
                add(surface.text, source="discord_identity", renderer="identity_proposition",
                    confidence=0.99, metadata={
                        "authored_reply_template": False, "role_order": surface.role_order,
                        "response_move": "answer", "answer_slot": "как",
                        "reasoning_evidence": identity_proposition.evidence,
                        "context_relevance": 1.0, "truth_grounding": 1.0,
                        "semantic_frame_fit": 1.0,
                    })
        else:
            add(identity_text, source="discord_identity", renderer="identity_fact",
                confidence=0.99, key_points=(identity_text,) if identity_text else ())

        exact_answer = utterance.context.exact_answer
        retrieval_rejection = None
        if identity_text is not None:
            retrieval_rejection = "identity_answer_has_priority"
        elif personal_action_resolution is not None:
            retrieval_rejection = "retrieval_not_applicable_to_personal_action"
        elif runtime_state_resolution is not None:
            retrieval_rejection = "retrieval_not_applicable_to_runtime_state"
        elif preference_thought is not None:
            retrieval_rejection = "retrieval_not_applicable_to_preference_act"
        elif (
            exact_answer is not None
            and _looks_like_lexical_definition(exact_answer)
            and not explicit_lexical_lookup
        ):
            retrieval_rejection = "lexical_definition_not_requested"
        elif dialogue_act is not None and dialogue_act.response_obligation in {
            "acknowledge_evaluation",
            "address_negative_evaluation",
            "select_or_compare",
        }:
            retrieval_rejection = "retrieval_not_applicable_to_dialogue_act"
        elif not _dialogue_act_requests_memory_fact(dialogue_act):
            retrieval_rejection = "direct_memory_surface_not_requested"
        elif social_routine:
            retrieval_rejection = "retrieval_not_applicable_to_social_act"
        add(
            exact_answer,
            source="knowledge_retrieval",
            renderer="exact_fact",
            confidence=0.97,
            required_facts=(exact_answer,) if exact_answer else (),
            key_points=(exact_answer,) if exact_answer else (),
            metadata=(
                {"hard_rejection": retrieval_rejection}
                if retrieval_rejection
                else {
                    "response_move": "answer",
                    "completed_subgoals": (
                        ("study", "answer")
                        if dialogue_act is not None
                        and dialogue_act.response_obligation == "study_and_answer"
                        else ("answer",)
                    ),
                    "knowledge_component": exact_answer,
                }
            ),
        )
        if (exact_answer is not None and retrieval_rejection is None
                and not self._productive_only_replies):
            global_required_facts = tuple(dict.fromkeys((*global_required_facts, exact_answer)))

        add(
            meaning_plan_surface,
            source="meaning_plan",
            renderer="structured_russian",
            confidence=(
                response_meaning_plan.confidence if response_meaning_plan is not None else 0.0
            ),
            required_facts=(
                response_meaning_plan.required_facts if response_meaning_plan is not None else ()
            ),
            key_points=(
                response_meaning_plan.required_facts[:2]
                if response_meaning_plan is not None
                else ()
            ),
            metadata=(
                {"hard_rejection": retrieval_rejection}
                if retrieval_rejection is not None
                else {
                    "semantic_plan": True,
                    "proposition_count": len(response_meaning_plan.propositions)
                    if response_meaning_plan is not None else 0,
                    "authored_reply_template": False,
                }
            ),
        )

        add(
            (abstract_resolution.text if abstract_resolution is not None else None),
            source="abstract_reasoning",
            renderer="mental_model_spatial_resolution",
            confidence=(abstract_resolution.confidence if abstract_resolution is not None else 0.0),
            key_points=(abstract_resolution.focus_terms if abstract_resolution is not None else ()),
            metadata=(
                {
                    "response_move": "answer",
                    "answer_slot": "где",
                    "mental_model_mode": abstract_resolution.mode,
                    "reasoning_evidence": abstract_resolution.evidence,
                    "truth_grounding": 1.0,
                    "semantic_frame_fit": 0.97,
                    "authored_reply_template": False,
                }
                if abstract_resolution is not None
                else None
            ),
        )
        spatial_dynamics_surfaces = (
            composer.express(
                spatial_dynamics_resolution.thought,
                utterance.context.dependency_patterns,
            )
            if spatial_dynamics_resolution is not None
            else ()
        )
        for surface_index, surface in enumerate(spatial_dynamics_surfaces):
            add(
                surface.text,
                source="spatial_dynamics",
                renderer=f"boundary_entry:{surface.source}:{surface_index}",
                confidence=min(
                    spatial_dynamics_resolution.confidence,
                    surface.confidence,
                ),
                key_points=spatial_dynamics_resolution.focus_terms,
                metadata={
                    "response_move": "answer",
                    "answer_slot": "откуда",
                    "reasoning_evidence": spatial_dynamics_resolution.evidence,
                    "truth_grounding": 1.0,
                    "semantic_frame_fit": 0.98,
                    "role_order": surface.role_order,
                    "construction_id": surface.construction_id,
                    "authored_reply_template": False,
                },
            )
        spatial_explanation_surfaces = (
            realize_boundary_entry_explanation(
                spatial_dynamics_explanation,
                utterance.context.dependency_patterns,
            )
            if spatial_dynamics_explanation is not None
            else ()
        )
        for surface_index, surface in enumerate(spatial_explanation_surfaces):
            add(
                surface.text,
                source="spatial_dynamics",
                renderer=f"boundary_reason:{surface.source}:{surface_index}",
                confidence=min(
                    spatial_dynamics_explanation.confidence,
                    surface.confidence,
                ),
                key_points=spatial_dynamics_explanation.focus_terms,
                metadata={
                    "response_move": "explain",
                    "answer_slot": "почему",
                    "reasoning_evidence": spatial_dynamics_explanation.evidence,
                    "truth_grounding": 1.0,
                    "semantic_frame_fit": 0.98,
                    "role_order": surface.role_order,
                    "construction_id": surface.construction_id,
                    "authored_reply_template": False,
                },
            )
        quantitative_surfaces = (
            composer.express(
                quantitative_resolution.thought,
                utterance.context.dependency_patterns,
            )
            if quantitative_resolution is not None
            else ()
        )
        for surface_index, surface in enumerate(quantitative_surfaces):
            add(
                surface.text,
                source="quantitative_reasoning",
                renderer=f"rate_time:{surface.source}:{surface_index}",
                confidence=min(quantitative_resolution.confidence, surface.confidence),
                key_points=quantitative_resolution.focus_terms,
                metadata={
                    "response_move": "answer",
                    "answer_slot": "когда",
                    "reasoning_evidence": quantitative_resolution.evidence,
                    "truth_grounding": 1.0,
                    "semantic_frame_fit": 0.99,
                    "role_order": surface.role_order,
                    "construction_id": surface.construction_id,
                    "authored_reply_template": False,
                },
            )

        internal_reasoning_surfaces = (
            composer.express(
                internal_reasoning.content_thought,
                utterance.context.dependency_patterns,
            )
            if (
                personal_action_resolution is None
                and internal_reasoning is not None
                and internal_reasoning.content_thought is not None
            )
            else ()
        )
        for reasoning_surface in internal_reasoning_surfaces:
            add(
                reasoning_surface.text,
                source="semantic_reasoning",
                renderer=f"productive_dependencies:{reasoning_surface.source}",
                confidence=min(internal_reasoning.confidence, reasoning_surface.confidence),
                key_points=(
                    tuple(semantic_question.alternatives) if semantic_question is not None else ()
                ),
                metadata=(
                    {
                        "question_kind": semantic_question.kind,
                        "knowledge_internalized": not internal_reasoning.needs_study,
                        "used_evidence": internal_reasoning.used_evidence,
                        "role_order": reasoning_surface.role_order,
                        "construction_id": reasoning_surface.construction_id,
                        "authored_reply_template": False,
                    }
                    if semantic_question is not None
                    else None
                ),
            )

        thought_surfaces = (
            composer.express(
                self_state_resolution.thought,
                utterance.context.dependency_patterns,
            )
            if self_state_resolution is not None and self_state_resolution.thought is not None
            else ()
        )
        for thought_surface in thought_surfaces:
            add(
                thought_surface.text,
                source="semantic_thought",
                renderer=f"productive_dependencies:{thought_surface.source}",
                confidence=min(
                    self_state_resolution.confidence,
                    thought_surface.confidence,
                ),
                key_points=(self_state_frame.target_lemmas if self_state_frame is not None else ()),
                metadata={
                    "state": self_state_resolution.state,
                    "actual": self_state_resolution.actual,
                    "reasoning_evidence": self_state_resolution.evidence,
                    "role_order": thought_surface.role_order,
                    "construction_id": thought_surface.construction_id,
                    "authored_reply_template": False,
                },
            )

        personal_action_surfaces = (
            composer.express(
                personal_action_resolution.thought,
                utterance.context.dependency_patterns,
            )
            if personal_action_resolution is not None
            and personal_action_resolution.thought is not None
            else ()
        )
        for action_surface in personal_action_surfaces:
            add(
                action_surface.text,
                source="personal_action",
                renderer=f"owned_action_history:{action_surface.source}",
                confidence=min(
                    personal_action_resolution.confidence,
                    action_surface.confidence,
                ),
                key_points=personal_action_resolution.frame.key_points,
                metadata={
                    "response_move": "answer",
                    "actual": personal_action_resolution.actual,
                    "reasoning_evidence": personal_action_resolution.evidence,
                    "role_order": action_surface.role_order,
                    "construction_id": action_surface.construction_id,
                    "truth_grounding": personal_action_resolution.confidence,
                    "semantic_frame_fit": 0.96,
                    "authored_reply_template": False,
                },
            )

        relation_surfaces = (
            composer.express(
                relation_reasoning.productive_thought,
                utterance.context.dependency_patterns,
            )
            if relation_reasoning is not None and relation_reasoning.productive_thought is not None
            else ()
        )
        for relation_surface in relation_surfaces:
            add(
                relation_surface.text,
                source="relation_reasoning",
                renderer=f"productive_dependencies:{relation_surface.source}",
                confidence=min(
                    relation_reasoning.confidence,
                    relation_surface.confidence,
                ),
                key_points=(
                    (relation_question.object_surface,) if relation_question is not None else ()
                ),
                metadata={
                    "relation": relation_reasoning.relation,
                    "object_lemma": relation_reasoning.object_lemma,
                    "object_class": relation_reasoning.object_class,
                    "knowledge_internalized": not relation_reasoning.needs_study,
                    "used_evidence": relation_reasoning.evidence_terms,
                    "content_plan": relation_reasoning.content_plan.response_act,
                    "role_order": relation_surface.role_order,
                    "construction_id": relation_surface.construction_id,
                    "authored_reply_template": False,
                },
            )

        for variant, contextual_surface in enumerate(contextual_question_surfaces):
            add(
                contextual_surface.text,
                source="dialogue_context",
                renderer=(
                    f"contextual_question:{contextual_question.kind}:"
                    f"{contextual_surface.source}:{variant}"
                ),
                confidence=min(
                    contextual_question.confidence,
                    contextual_surface.confidence,
                ),
                key_points=contextual_question.target_terms,
                metadata={
                    "response_move": "answer",
                    "answer_slot": (
                        "почему"
                        if contextual_question.kind
                        in {
                            "embodiment_cause",
                            "epistemic_cause",
                            "epistemic_purpose",
                            "internal_capability_correction",
                            "overcautious_response_cause",
                            "preference_cause_unestablished",
                            "self_assessment_cause",
                        }
                        else "куда"
                        if contextual_question.kind == "antecedent_location_unspecified"
                        else "как"
                        if contextual_question.kind == "cognitive_method"
                        else ""
                    ),
                    "contextual_question_kind": contextual_question.kind,
                    "antecedent": contextual_question.antecedent,
                    "reasoning_evidence": contextual_question.evidence,
                    "role_order": contextual_surface.role_order,
                    "construction_id": contextual_surface.construction_id,
                    "authored_reply_template": False,
                },
            )

        add(
            dialogue_clarification,
            source="dialogue_context",
            renderer="reference_clarification",
            confidence=(
                utterance.context.dialogue_state.confidence
                if utterance.context.dialogue_state is not None
                else 0.0
            ),
            key_points=(dialogue_clarification,) if dialogue_clarification else (),
            metadata={"unresolved_reference": True},
        )

        add(
            repair_surface,
            source="dialogue_context",
            renderer="answer_repair",
            confidence=0.96 if repair_surface else 0.0,
            key_points=(repair_target,) if repair_target else (),
            metadata={"repair_target": repair_target} if repair_target else None,
        )

        for capability_surface in capability_surfaces:
            add(
                capability_surface.text,
                source=(
                    "personality_desire"
                    if capability_assessment is not None
                    and capability_assessment.mode == "personal_desire"
                    else "semantic_reasoning"
                ),
                renderer=f"productive_dependencies:{capability_surface.source}",
                confidence=(
                    min(capability_assessment.confidence, capability_surface.confidence)
                    if capability_assessment is not None
                    else 0.0
                ),
                key_points=(
                    (capability_assessment.action_lemma,)
                    if capability_assessment is not None
                    else ()
                ),
                metadata=(
                    {
                        "capability_reason": capability_assessment.reason,
                        "reasoning_evidence": capability_assessment.evidence,
                        "role_order": capability_surface.role_order,
                        "construction_id": capability_surface.construction_id,
                        "authored_reply_template": False,
                    }
                    if capability_assessment is not None
                    else None
                ),
            )

        for position_surface in position_surfaces:
            add(
                position_surface.text,
                source="position_reasoning",
                renderer=f"position_dependencies:{position_surface.source}",
                confidence=(
                    min(position_decision.confidence, position_surface.confidence)
                    if position_decision is not None
                    else 0.0
                ),
                key_points=(
                    (position_decision.target, position_decision.quality)
                    if position_decision is not None
                    else ()
                ),
                metadata=(
                    {
                        "position_key": position_decision.position_key,
                        "position_stance": position_decision.stance,
                        "position_score": position_decision.score,
                        "position_confidence": position_decision.confidence,
                        "position_evidence": position_decision.evidence,
                        "expression_mode": position_decision.expression_mode,
                        "response_move": "state_position",
                        "role_order": position_surface.role_order,
                        "construction_id": position_surface.construction_id,
                        "authored_reply_template": False,
                    }
                    if position_decision is not None
                    else None
                ),
            )

        for surface_index, inquiry_surface in enumerate(semantic_inquiry_surfaces):
            add(
                inquiry_surface.text,
                source="semantic_inquiry",
                renderer=(f"semantic_inquiry:{inquiry_surface.source}:{surface_index}"),
                confidence=(
                    min(semantic_inquiry.confidence, inquiry_surface.confidence)
                    if semantic_inquiry is not None
                    else 0.0
                ),
                key_points=(
                    tuple(
                        dict.fromkeys(
                            step.answer_lemma
                            for step in semantic_inquiry.steps
                            if step.answer_lemma is not None
                        )
                    )[:6]
                    if semantic_inquiry is not None
                    else ()
                ),
                metadata=(
                    {
                        "inquiry_id": semantic_inquiry.inquiry_id,
                        "response_move": "interpret_scene",
                        "reasoning_evidence": semantic_inquiry.evidence,
                        "inquiry_step_count": len(semantic_inquiry.steps),
                        "inquiry_unresolved": tuple(
                            step.answer_role
                            for step in semantic_inquiry.steps
                            if step.status == "unresolved"
                        ),
                        "study_terms": semantic_inquiry.study_terms,
                        "context_relevance": semantic_inquiry.confidence,
                        "reply_binding": semantic_inquiry.confidence,
                        "semantic_frame_fit": 0.90,
                        "truth_grounding": 0.68,
                        "epistemic_status": "calibrated_hypothesis",
                        "role_order": inquiry_surface.role_order,
                        "construction_id": inquiry_surface.construction_id,
                        "authored_reply_template": False,
                        "completed_subgoals": (
                            (dialogue_act.response_obligation,) if dialogue_act is not None else ()
                        ),
                    }
                    if semantic_inquiry is not None
                    else None
                ),
            )

        for move_plan in dialogue_move_plans:
            move_surfaces = composer.express(
                move_plan.thought,
                utterance.context.dependency_patterns,
                limit=3,
            )
            for variant, move_surface in enumerate(move_surfaces):
                add(
                    move_surface.text,
                    source="dialogue_move",
                    renderer=(
                        f"contextual_move:{move_plan.strategy}:{move_surface.source}:{variant}"
                    ),
                    confidence=min(move_plan.confidence, move_surface.confidence),
                    key_points=move_plan.key_points,
                    metadata={
                        "dialogue_act": (dialogue_act.kind if dialogue_act is not None else None),
                        "response_obligation": (
                            dialogue_act.response_obligation if dialogue_act is not None else None
                        ),
                        "response_move": move_plan.move,
                        "move_strategy": move_plan.strategy,
                        "move_stance": move_plan.stance,
                        "reasoning_evidence": move_plan.evidence,
                        "context_relevance": move_plan.confidence,
                        "reply_binding": move_plan.confidence,
                        "semantic_frame_fit": move_plan.confidence,
                        "social_policy_ids": (
                            communicative_intention.matched_policy_ids
                            if communicative_intention is not None
                            else ()
                        ),
                        "role_order": move_surface.role_order,
                        "construction_id": move_surface.construction_id,
                        "authored_reply_template": False,
                        "semantic_origin": move_plan.semantic_origin,
                        "semantic_scaffold": move_plan.semantic_origin != "derived",
                        "completed_subgoals": (
                            (dialogue_act.response_obligation,) if dialogue_act is not None else ()
                        ),
                    },
                )

        add(
            (
                dialogue_act_surface
                if (self_state_resolution is None or self_state_resolution.thought is None)
                and not (
                    dialogue_act is not None
                    and dialogue_act.response_obligation == "continue_topic"
                    and dialogue_act.subject == "speaker"
                    and not dialogue_act.objects
                )
                else None
            ),
            source="dialogue_act",
            renderer="response_obligation_realizer",
            confidence=dialogue_act.confidence if dialogue_act is not None else 0.0,
            key_points=dialogue_act.objects[:2] if dialogue_act is not None else (),
            metadata=(
                {
                    "dialogue_act": dialogue_act.kind,
                    "response_obligation": dialogue_act.response_obligation,
                    "act_evidence": dialogue_act.evidence,
                    "response_move": (
                        communicative_intention.primary_move
                        if communicative_intention is not None
                        else "engage"
                    ),
                    "social_policy_ids": (
                        communicative_intention.matched_policy_ids
                        if communicative_intention is not None
                        else ()
                    ),
                    "authored_reply_template": False,
                }
                if dialogue_act is not None
                else None
            ),
        )

        add(
            direct_reply,
            source="dialogue_rule",
            renderer="dialogue_plan",
            confidence=0.88,
        )
        if response_requirement(prompt) == "review_prior_claim":
            reviewed = prior_claim_review_thought(prompt, utterance.context.consultation_messages)
            if reviewed is not None:
                acknowledgment, claim = reviewed
                surfaces = composer.compose(MessagePlan(
                    composer.goal, acknowledgment,
                    follow_up=unmet_requirement_thought("review_prior_claim"),
                ), limit=1)
                if surfaces:
                    add(
                        surfaces[0].text, source="semantic_reasoning",
                        renderer="review_own_claim", confidence=0.86,
                        key_points=tuple(content_lemmas(claim)),
                        metadata={
                            "reviewed_agent_claim": claim, "claim_source": "dialogue",
                            "response_move": "targeted_repair",
                            "reasoning_evidence": acknowledgment.evidence,
                            "role_order": surfaces[0].role_order,
                            "authored_reply_template": False, "semantic_origin": "derived",
                            "truth_grounding": 0.9, "semantic_frame_fit": 0.95,
                        },
                    )
        add(
            causal_inference.text if causal_inference is not None else None,
            source="causal_reasoning",
            renderer="causal_path",
            confidence=causal_inference.confidence if causal_inference is not None else 0.0,
            key_points=(causal_query.focus_terms if causal_query is not None else ()),
            metadata=(
                {
                    "response_move": "answer",
                    "answer_slot": ("почему" if causal_query.direction == "cause" else "что"),
                    "causal_direction": causal_query.direction,
                    "causal_path": causal_inference.evidence,
                    "truth_grounding": causal_inference.confidence,
                    "semantic_frame_fit": 0.94,
                    "completed_subgoals": ("answer_question", "explain_causality"),
                }
                if causal_inference is not None and causal_query is not None
                else None
            ),
        )
        preference_surfaces = (
            composer.express(preference_thought, utterance.context.dependency_patterns)
            if preference_thought is not None
            else ()
        )
        for variant, surface in enumerate(preference_surfaces):
            add(
                surface.text,
                source="semantic_thought",
                renderer=f"productive_dependencies:{surface.source}:{variant}",
                confidence=surface.confidence,
                key_points=tuple(understanding.content_terms[:2]),
                metadata={
                    "role_order": surface.role_order,
                    "construction_id": surface.construction_id,
                    "authored_reply_template": False,
                    "preference_state": True,
                    "completed_subgoals": ("express_position",),
                    "position_component": surface.text,
                },
            )
        preference_reason_surfaces = (
            composer.express(
                preference_reason,
                utterance.context.dependency_patterns,
            )
            if preference_reason is not None
            else ()
        )
        for variant, surface in enumerate(preference_reason_surfaces):
            add(
                surface.text,
                source="semantic_thought",
                renderer=f"preference_reason:{surface.source}:{variant}",
                confidence=surface.confidence,
                key_points=tuple(understanding.context_links[:2]),
                metadata={
                    "response_move": "answer",
                    "answer_slot": "почему",
                    "role_order": surface.role_order,
                    "construction_id": surface.construction_id,
                    "authored_reply_template": False,
                    "preference_reason": True,
                },
            )
        answer_reference = (
            None if repair_target is not None else _answer_reference_thought(understanding)
        )
        for surface in composer.express(answer_reference, limit=3) if answer_reference else ():
            add(
                surface.text,
                source="dialogue_rule",
                renderer=surface.source,
                confidence=surface.confidence,
                key_points=tuple(understanding.content_terms[:3]),
                metadata={
                    "response_move": "answer",
                    "authored_reply_template": False,
                    "semantic_answer_reference": True,
                    "role_order": surface.role_order,
                    "semantic_frame_fit": 0.98,
                },
            )
        lexical_reply = await self._lexical_reference_reply(prompt, analyses)
        lexical_subgoals: tuple[str, ...] = ()
        if lexical_reply is not None:
            lexical_subgoals = (
                ("study", "answer")
                if dialogue_act is not None
                and dialogue_act.response_obligation == "study_and_answer"
                else ("answer",)
            )
        add(
            lexical_reply,
            source="lexical_reference",
            renderer="lexical_fact",
            confidence=0.95,
            required_facts=(lexical_reply,) if lexical_reply else (),
            metadata=(
                {
                    "response_move": "answer",
                    "completed_subgoals": lexical_subgoals,
                    "knowledge_component": lexical_reply,
                }
                if lexical_reply is not None
                else None
            ),
        )
        if (
            dialogue_act is not None
            and dialogue_act.response_obligation == "study_and_express_position"
            and lexical_reply is not None
        ):
            studied_preference = _preference_question_thought(
                prompt,
                understanding,
                analyses=analyses,
                facts=(*internal_facts, lexical_reply),
                perspective=utterance.context.perspective,
            )
            studied_surfaces = (
                composer.express(
                    studied_preference,
                    utterance.context.dependency_patterns,
                )
                if studied_preference is not None
                else ()
            )
            position_surface = studied_surfaces[0].text if studied_surfaces else None
        else:
            position_surface = None
        if lexical_reply is not None and position_surface is not None:
            add(
                f"{lexical_reply} {position_surface}",
                source="semantic_synthesis",
                renderer="multi_act_knowledge_then_position",
                confidence=min(0.91, dialogue_act.confidence),
                required_facts=(lexical_reply,),
                key_points=dialogue_act.objects,
                metadata={
                    "completed_subgoals": ("study", "express_position"),
                    "knowledge_component": lexical_reply,
                    "position_component": position_surface,
                    "authored_reply_template": False,
                },
            )
        if lexical_reply is not None and not self._productive_only_replies:
            global_required_facts = tuple(dict.fromkeys((*global_required_facts, lexical_reply)))

        consulted_lookup_answer = (
            _realize_consulted_definition(
                foreground_lookup_target,
                consulted_facts[0],
            )
            if foreground_lookup_target is not None and consulted_facts
            else None
        )
        add(
            consulted_lookup_answer,
            source="semantic_synthesis",
            renderer="internalized_lexical_fact",
            confidence=0.72,
            required_facts=(
                (consulted_lookup_answer,) if consulted_lookup_answer is not None else ()
            ),
            key_points=(
                (foreground_lookup_target,) if foreground_lookup_target is not None else ()
            ),
            metadata=(
                {
                    "response_move": "answer",
                    "completed_subgoals": ("study", "answer"),
                    "knowledge_component": consulted_lookup_answer,
                    "authored_reply_template": False,
                    "teacher_fact_recomposed": True,
                }
                if consulted_lookup_answer is not None
                else None
            ),
        )
        if consulted_lookup_answer is not None and not self._productive_only_replies:
            global_required_facts = tuple(
                dict.fromkeys((*global_required_facts, consulted_lookup_answer))
            )

        cached_reply = self._stable_reply_cache.get(utterance.context.seed)
        add(
            cached_reply,
            source="cache",
            renderer="stable_cache",
            confidence=0.76,
        )

        generator_candidates = getattr(self._generator, "generate_candidates", None)
        if callable(generator_candidates):
            generated = generator_candidates(
                utterance,
                limit=min(
                    3 if semantic_question is not None else 10,
                    self._arbitration_max_total_candidates,
                ),
            )
            for item in generated:
                add(
                    item.text,
                    source=getattr(item, "source", "graph"),
                    renderer=item.renderer,
                    confidence=max(0.35, min(0.80, utterance.thought.confidence)),
                    key_points=tuple(utterance.context.required_facts[:2]),
                    metadata={"variant": item.variant},
                )
        else:
            for attempt in range(self._response_plan_retry_limit + 1):
                attempt_context = replace(
                    utterance.context,
                    seed=f"{utterance.context.seed}|arbitration-attempt:{attempt}",
                )
                add(
                    self._generator.generate(replace(utterance, context=attempt_context)),
                    source="graph",
                    renderer=f"legacy_generator:{attempt}",
                    confidence=max(0.35, min(0.80, utterance.thought.confidence)),
                    key_points=tuple(utterance.context.required_facts[:2]),
                )

        if social_routine:
            recovery_surface = compose_recovery(evidence)
            add(
                recovery_surface,
                source="recovery",
                renderer="social_recovery",
                confidence=0.78,
            )
        elif understanding.unknown_terms and not understanding.sufficient:
            recovered = await self._recover_response(
                event,
                utterance,
                analyses,
                understanding=understanding,
            )
            add(
                recovered,
                source="recovery",
                renderer="knowledge_gap",
                confidence=0.58,
            )

        if exact_answer is not None and retrieval_rejection is not None:
            rejected_surface = " ".join(exact_answer.casefold().split())
            candidates = [
                replace(
                    candidate,
                    metadata={
                        **candidate.metadata,
                        "hard_rejection": retrieval_rejection,
                    },
                )
                if " ".join(candidate.text.casefold().split()) == rejected_surface
                and candidate.source != "discord_identity"
                else candidate
                for candidate in candidates
            ]

        select_and_express_content()
        has_bound_answer = any(
            candidate.source in _CONVERSATIONAL_REASONING_SOURCES
            and candidate.source_confidence >= self._response_confidence_threshold
            and not candidate.metadata.get("hard_rejection")
            and (
                candidate.metadata.get("communicative_status") == "inform"
                and candidate.metadata.get("reader_checked") is True
                or candidate.required_facts
                or candidate.metadata.get("answer_slot")
                and candidate.metadata.get("reasoning_evidence")
            )
            for candidate in candidates
        )
        local_productive_threshold = (
            min(self._response_confidence_threshold, 0.34)
            if dialogue_act is not None and not dialogue_act.complete
            else self._response_confidence_threshold
        )
        local_decisive_answer = any(
            candidate.source in _AUTONOMOUS_LOCAL_PRODUCTIVE_SOURCES
            and candidate.source_confidence >= local_productive_threshold
            and candidate.metadata.get("authored_reply_template") is False
            and candidate.metadata.get("role_order")
            and candidate.metadata.get("semantic_scaffold") is not True
            and candidate.metadata.get("communicative_status") != "express_limit"
            and not candidate.metadata.get("hard_rejection")
            for candidate in candidates
        )
        answer_research_required = bool(
            not social_routine
            and not has_bound_answer
            and (understanding.task is None or understanding.task.research_allowed)
            and (
                explicit_lexical_lookup
                or _allows_epistemic_recovery(
                    dialogue_act,
                    structured_bundle[0] if structured_bundle is not None else None,
                )
            )
        )
        consultation_required = bool(
            self._semantic_consultant is not None
            and not self._autonomous_planning_required
            and not social_routine
            and self._semantic_consultant.should_consult(
                local_confidence=understanding.comprehension_score,
                unknown_terms=understanding.unknown_terms,
                has_bound_answer=has_bound_answer,
            )
        )
        consultation = None
        semantic_facts: tuple[str, ...] = ()
        if consultation_required or answer_research_required:
            reason_codes = tuple(
                code
                for code, applies in (
                    ("no_reply_bound_local_candidate", not has_bound_answer),
                    ("unknown_terms", bool(understanding.unknown_terms)),
                    (
                        "low_comprehension",
                        self._semantic_consultant is not None
                        and understanding.comprehension_score
                        < self._semantic_consultant.trigger_threshold,
                    ),
                )
                if applies
            )
            consultation_messages = (
                utterance.context.consultation_messages[-8:]
                or utterance.context.dialogue_messages[-8:]
            )
            local_relations = tuple(
                (item.source, item.relation, item.target) for item in understanding.relations
            )
            structured_research_terms = (
                tuple(
                    token.lemma
                    for token in structured_bundle[0].tokens
                    if token.part_of_speech in {"noun", "adjective", "numeral"}
                )
                if structured_bundle is not None
                else ()
            )
            research_terms = tuple(
                dict.fromkeys(
                    (
                        *structured_research_terms,
                        *content_lemmas(prompt),
                    )
                )
            )[:4]
            internalize = (
                getattr(self._knowledge_service, "internalize_for_reasoning", None)
                if self._knowledge_service is not None
                else None
            )
            research_question = (
                getattr(self._semantic_consultant, "research_question", None)
                if self._semantic_consultant is not None
                else None
            )
            consultation, researched_concepts, researched_facts = await asyncio.gather(
                self._semantic_consultant.consult(
                    event_id=event.event_id,
                    guild_id=event.guild_id,
                    channel_id=event.channel_id,
                    request=ConsultationRequest(
                        prompt=prompt,
                        context_messages=consultation_messages,
                        known_terms=understanding.known_terms,
                        unknown_terms=understanding.unknown_terms,
                        local_relations=local_relations,
                        local_confidence=understanding.comprehension_score,
                        reason_codes=reason_codes,
                    ),
                )
                if consultation_required and self._semantic_consultant is not None
                else asyncio.sleep(0, result=None),
                internalize(
                    event,
                    research_terms,
                    limit=max(4, len(research_terms)),
                )
                if answer_research_required and research_terms and callable(internalize)
                else asyncio.sleep(0, result=()),
                research_question(
                    event_id=event.event_id,
                    guild_id=event.guild_id,
                    channel_id=event.channel_id,
                    prompt=prompt,
                    context_messages=consultation_messages,
                    subject_terms=research_terms,
                    local_relations=local_relations,
                )
                if answer_research_required and callable(research_question)
                else asyncio.sleep(0, result=()),
            )
            reference_facts = tuple(
                dict.fromkeys(
                    concise_reference(str(concept.definition))
                    for concept in researched_concepts
                    if getattr(concept, "definition", None)
                )
            )
            reference_definitions.extend(
                (concept.title, concise_reference(concept.definition))
                for concept in researched_concepts
                if getattr(concept, "definition", None)
            )
            if consultation is not None and dialogue_act is not None:
                refined_act = refine_dialogue_act(
                    dialogue_act,
                    external_intent=consultation.intent,
                    confidence=consultation.confidence,
                    external_response_obligation=(
                        consultation.social_frame.response_obligation
                        if consultation.social_frame is not None
                        else None
                    ),
                    expected_response_acts=consultation.expected_response_acts,
                )
                if refined_act != dialogue_act:
                    # The consultation is a semantic correction, not merely
                    # another candidate generator.  Make the corrected act the
                    # active response contract so every later gate, renderer
                    # and persisted plan evaluates the actual user request.
                    dialogue_act = refined_act
                    communicative_intention = choose_communicative_intention(
                        dialogue_act,
                        utterance.context.social_policies,
                        grounding_available=bool(internal_facts),
                    )
                    reframe_plan = (
                        getattr(self._response_plans, "reframe", None)
                        if self._response_plans is not None
                        else None
                    )
                    if stored_plan is not None and callable(reframe_plan):
                        stored_plan = await reframe_plan(
                            stored_plan,
                            message_intent=dialogue_act.kind,
                            response_goal=dialogue_act.response_obligation,
                            key_points=dialogue_act.objects,
                            confidence=dialogue_act.confidence,
                        )
                    add(
                        realize_dialogue_act(refined_act),
                        source="dialogue_act",
                        renderer="consulted_intent_obligation_realizer",
                        confidence=refined_act.confidence,
                        key_points=refined_act.objects[:2],
                        metadata={
                            "dialogue_act": refined_act.kind,
                            "response_obligation": refined_act.response_obligation,
                            "external_intent_only": consultation.intent,
                            "authored_reply_template": False,
                        },
                    )
            semantic_facts = tuple(
                dict.fromkeys(
                    fact.strip()
                    for fact in (
                        *researched_facts,
                        *reference_facts,
                        *(consultation.facts[:5] if consultation is not None else ()),
                    )
                    if fact.strip() and internal_surface_reason(fact) is None
                )
            )
            if (
                dialogue_act is not None
                and dialogue_act.response_obligation == "express_position"
                and not any(
                    candidate.source == "position_reasoning"
                    and not candidate.metadata.get("hard_rejection")
                    for candidate in candidates
                )
            ):
                # Research supplies evidence; it is not itself the answer to a
                # subjective request.  Re-run owned position reasoning after
                # intent refinement and fact collection, then realize a local
                # first-person stance.  With no decisive evidence this yields
                # a qualified, revisable position instead of a fake epistemic
                # failure such as "недостаточно данных".
                researched_position = await self._position_reasoning.resolve(
                    event.guild_id,
                    dialogue_act,
                    facts=tuple(dict.fromkeys((*internal_facts, *semantic_facts))),
                    context_messages=utterance.context.dialogue_messages,
                    perspective=utterance.context.perspective,
                )
                researched_position_surfaces = (
                    realize_position(
                        researched_position,
                        utterance.context.dependency_patterns,
                        composer=composer,
                    )
                    if researched_position is not None
                    else ()
                )
                for researched_surface in researched_position_surfaces:
                    add(
                        researched_surface.text,
                        source="position_reasoning",
                        renderer=(f"position_dependencies:consulted_{researched_surface.source}"),
                        confidence=min(
                            researched_position.confidence,
                            researched_surface.confidence,
                        ),
                        key_points=(
                            researched_position.target,
                            researched_position.quality,
                        ),
                        metadata={
                            "position_key": researched_position.position_key,
                            "position_stance": researched_position.stance,
                            "position_score": researched_position.score,
                            "position_confidence": researched_position.confidence,
                            "position_evidence": researched_position.evidence,
                            "expression_mode": researched_position.expression_mode,
                            "response_move": "state_position",
                            "role_order": researched_surface.role_order,
                            "construction_id": researched_surface.construction_id,
                            "consulted_intent_refinement": True,
                            "authored_reply_template": False,
                        },
                    )
                if researched_position_surfaces:
                    position_decision = researched_position
            persist = (
                getattr(
                    self._knowledge_service,
                    "internalize_consulted_fact",
                    None,
                )
                if self._knowledge_service is not None
                else None
            )
            if researched_facts and research_terms and callable(persist):
                for fact in researched_facts:
                    fact_terms = content_lemmas(fact)
                    matching_queries = tuple(
                        term
                        for term in research_terms
                        if any(_same_lexical_family(term, fact_term) for fact_term in fact_terms)
                    )
                    if not matching_queries:
                        continue
                    await persist(
                        event,
                        query=matching_queries[0],
                        definition=fact,
                        confidence=0.70,
                    )
            answer_facts = tuple(dict.fromkeys((*researched_facts, *reference_facts)))
            explicit_lookup = foreground_lookup_target is not None
            research_facts_may_surface = _research_facts_may_surface(
                dialogue_act=dialogue_act,
                structure=structured_bundle[0] if structured_bundle is not None else None,
                explicit_lookup=explicit_lookup,
            )
            answer_slot = (
                next(
                    (
                        item.partition(":")[2]
                        for item in dialogue_act.evidence
                        if item.startswith("question_target:")
                    ),
                    None,
                )
                if dialogue_act is not None
                else None
            )
            for fact_index, fact in enumerate(
                answer_facts[:3] if not self._productive_only_replies else ()
            ):
                if (
                    not research_facts_may_surface
                    or unrequested_definition(fact, prompt, tuple(reference_definitions))
                    or (_looks_like_lexical_definition(fact) and not explicit_lookup)
                    or (not explicit_lookup and not _research_fact_bound_to_prompt(fact, prompt))
                ):
                    continue
                add(
                    concise_reference(fact),
                    source="semantic_synthesis",
                    renderer=f"foreground_researched_fact:{fact_index}",
                    confidence=0.72 if fact in researched_facts else 0.66,
                    required_facts=(fact,),
                    key_points=content_lemmas(fact)[:4],
                    metadata={
                        "reply_binding": 0.88,
                        "foreground_research": True,
                        "research_sources": (
                            "semantic_teacher" if fact in researched_facts else "reference"
                        ),
                        "response_move": "answer",
                        "answer_slot": answer_slot,
                        "completed_subgoals": ("study", "answer"),
                        "knowledge_component": fact,
                        "authored_reply_template": False,
                    },
                )
            surface_semantic_facts = tuple(
                fact
                for fact in semantic_facts
                if not unrequested_definition(fact, prompt, tuple(reference_definitions))
                and ((head := definition_head(fact)) is None or definition_requested(prompt, head))
                and (
                    explicit_lookup
                    or (
                        not _looks_like_lexical_definition(fact)
                        and _research_fact_bound_to_prompt(fact, prompt)
                    )
                )
            )
            for title, definition in dict.fromkeys(reference_definitions):
                action_thought = definition_action_thought(prompt, title, definition)
                for surface in composer.express(action_thought, limit=1) if action_thought else ():
                    add(
                        surface.text, source="semantic_synthesis",
                        renderer="definition_action_roles", confidence=surface.confidence,
                        metadata={
                            "authored_reply_template": False,
                            "role_order": surface.role_order,
                            "reasoning_evidence": action_thought.evidence,
                            "response_move": "answer", "answer_slot": answer_slot,
                            "completed_subgoals": ("study", "answer"),
                            "foreground_research": True,
                        },
                    )
            if (
                research_facts_may_surface
                and surface_semantic_facts
                and structured_bundle is not None
            ):
                consulted_context = replace(
                    utterance.context,
                    exact_answer=None,
                    required_facts=tuple(
                        dict.fromkeys(
                            (
                                *utterance.context.required_facts,
                                *surface_semantic_facts,
                            )
                        )
                    ),
                    seed=f"{utterance.context.seed}|internal-semantic-plan",
                )
                consulted_utterance = replace(utterance, context=consulted_context)
                consulted_meaning = build_response_meaning_plan(
                    consulted_utterance,
                    structured_bundle[0],
                    structured_bundle[1],
                )
                consultation_confidence = (
                    consultation.confidence if consultation is not None else 0.72
                )
                add(
                    realize_response_meaning(
                        consulted_meaning, allow_rough=self._productive_only_replies,
                    ),
                    source="semantic_synthesis",
                    renderer="internal_meaning_plan_after_foreground_research",
                    confidence=max(
                        0.35,
                        min(0.76, consultation_confidence * 0.84),
                    ),
                    required_facts=surface_semantic_facts,
                    key_points=surface_semantic_facts[:4],
                    metadata={
                        "reply_binding": 0.82,
                        "provider": (
                            consultation.provider
                            if consultation is not None
                            else "foreground-research"
                        ),
                        "model": (
                            consultation.model
                            if consultation is not None
                            else "reference-and-semantic-teacher"
                        ),
                        "intent": (
                            consultation.intent
                            if consultation is not None
                            else dialogue_act.kind
                            if dialogue_act is not None
                            else "answer"
                        ),
                        "cached": (consultation.cached if consultation is not None else False),
                        "foreground_research": answer_research_required,
                        "response_move": "answer",
                        "answer_slot": answer_slot,
                        "completed_subgoals": ("study", "answer"),
                        "knowledge_component": surface_semantic_facts[0],
                        "proposition_count": len(consulted_meaning.propositions),
                        "authored_reply_template": False,
                    },
                )

        autonomous_planning_result = None
        autonomous_grounding: list[tuple[str, str]] = []

        if runtime_state_resolution is not None:
            autonomous_grounding.extend(
                (observation.fact_id, observation.value)
                for observation in runtime_state_resolution.observations
            )
        if personal_action_resolution is not None:
            action_object = (
                personal_action_resolution.frame.object.head.lemma
                if personal_action_resolution.frame.object is not None
                else None
            )
            autonomous_grounding.append(
                (
                    "agent.action_history",
                    f"predicate={personal_action_resolution.frame.action_lemma};"
                    "subject=agent;"
                    + (f"direct_object={action_object};" if action_object is not None else "")
                    + f"tense={personal_action_resolution.frame.tense};"
                    f"truth={str(bool(personal_action_resolution.actual)).lower()}."
                    " Локальная история действий агента является "
                    "авторитетным источником этого факта. "
                    f"confidence={personal_action_resolution.confidence:.3f}",
                )
            )
        if self_state_resolution is not None:
            autonomous_grounding.append(
                (
                    "agent.self_state",
                    f"state={self_state_resolution.state};"
                    f"actual={self_state_resolution.actual!r};"
                    f"confidence={self_state_resolution.confidence:.3f}",
                )
            )
        if capability_assessment is not None:
            capability_object = (
                capability_assessment.action_object.head.lemma
                if capability_assessment.action_object is not None
                else None
            )
            capability_predicate = (
                capability_assessment.action_lemma
                if capability_assessment.mode in {"experience", "activity", "commitment"}
                else "хотеть"
                if capability_assessment.mode in {"hypothetical_desire", "personal_desire"}
                else "мочь"
                if capability_assessment.mode == "directive"
                else "уметь"
            )
            capability_action_role = (
                ""
                if capability_assessment.mode in {"experience", "activity", "commitment"}
                else f"action_complement={capability_assessment.action_lemma};"
            )
            capability_tense = (
                "past"
                if capability_assessment.mode == "experience"
                else "future"
                if capability_assessment.mode == "commitment"
                else "present"
            )
            autonomous_grounding.append(
                (
                    "agent.capability",
                    f"predicate={capability_predicate};"
                    "subject=agent;"
                    f"{capability_action_role}"
                    + (
                        f"direct_object={capability_object};"
                        if capability_object is not None
                        else ""
                    )
                    + f"tense={capability_tense};"
                    f"truth={str(capability_assessment.supported).lower()}."
                    f" mode={capability_assessment.mode};"
                    f"reason={capability_assessment.reason};"
                    f"evidence={','.join(capability_assessment.evidence[:6])};"
                    f"confidence={capability_assessment.confidence:.3f}",
                )
            )
        if contextual_question is not None:
            autonomous_grounding.append(
                (
                    "dialogue.context_resolution",
                    f"kind={contextual_question.kind};"
                    f"antecedent={contextual_question.antecedent or '_'};"
                    f"targets={','.join(contextual_question.target_terms) or '_'};"
                    f"confidence={contextual_question.confidence:.3f}",
                )
            )
        if relation_reasoning is not None:
            relation_truth = (
                relation_reasoning.content_plan.clauses[0].truth
                if relation_reasoning.content_plan is not None
                and relation_reasoning.content_plan.clauses
                else None
            )
            autonomous_grounding.append(
                (
                    "agent.relation_state",
                    f"relation={relation_reasoning.relation};"
                    f"object={relation_reasoning.object_lemma or '_'};"
                    f"object_class={relation_reasoning.object_class or '_'};"
                    f"truth={relation_truth!r};"
                    f"confidence={relation_reasoning.confidence:.3f}",
                )
            )
        if position_decision is not None:
            autonomous_grounding.append(
                (
                    "personality.position",
                    f"target={position_decision.target};"
                    f"quality={position_decision.quality};"
                    f"stance={position_decision.stance};"
                    f"score={position_decision.score:.3f};"
                    f"confidence={position_decision.confidence:.3f}",
                )
            )
        if causal_inference is not None:
            causal_links = "|".join(
                (
                    f"{link.relation}:"
                    f"{link.cause.subject_lemma or '_'}:"
                    f"{link.cause.predicate_lemma or '_'}:"
                    f"{link.cause.object_lemma or '_'}>"
                    f"{link.effect.subject_lemma or '_'}:"
                    f"{link.effect.predicate_lemma or '_'}:"
                    f"{link.effect.object_lemma or '_'}"
                )
                for link in causal_inference.path
            )
            if causal_links:
                autonomous_grounding.append(
                    (
                        "reasoning.causal_path",
                        f"direction={causal_inference.direction};"
                        f"links={causal_links};"
                        f"confidence={causal_inference.confidence:.3f}",
                    )
                )
        relevant_grounding_facts = tuple(
            dict.fromkeys(
                fact
                for fact in (*internal_facts, *consulted_facts, *semantic_facts)
                if explicit_lexical_lookup or _research_fact_bound_to_prompt(fact, prompt)
            )
        )
        autonomous_grounding.extend(
            (
                f"knowledge.{hashlib.sha256(fact.encode('utf-8')).hexdigest()[:12]}",
                fact,
            )
            for fact in relevant_grounding_facts[:16]
        )
        autonomous_grounding.extend(
            _previous_answer_grounding(utterance.context.consultation_messages)
        )
        autonomous_grounding.append(
            (
                "personality.current",
                "personality_drives=authoritative_current_state",
            )
        )
        perspective = utterance.context.perspective
        personality_drives = (
            tuple(
                (name, perspective.drive(name))
                for name in (
                    "sociability",
                    "curiosity",
                    "expressiveness",
                    "novelty",
                    "coherence",
                    "assertiveness",
                    "independence",
                    "skepticism",
                    "patience",
                    "warmth",
                    "playfulness",
                    "trust",
                    "affinity",
                    "resistance",
                    "arousal",
                    "caution",
                    "exploration",
                )
            )
            if perspective is not None
            else ()
        )
        plan_response = (
            getattr(self._semantic_consultant, "plan_response", None)
            if self._semantic_consultant is not None
            else None
        )
        if (
            self._autonomous_planning_required
            and not local_decisive_answer
            and callable(plan_response)
        ):
            autonomous_planning_result = await plan_response(
                event_id=event.event_id,
                guild_id=event.guild_id,
                channel_id=event.channel_id,
                prompt=prompt,
                context_messages=(
                    utterance.context.consultation_messages[-8:]
                    or utterance.context.dialogue_messages[-8:]
                ),
                known_terms=understanding.known_terms,
                local_relations=tuple(
                    (item.source, item.relation, item.target) for item in understanding.relations
                ),
                grounding_facts=tuple(autonomous_grounding),
                personality_drives=personality_drives,
                response_obligation=(
                    dialogue_act.response_obligation
                    if dialogue_act is not None
                    else utterance.thought.speech_act
                ),
                local_confidence=understanding.comprehension_score,
            )
        autonomous_surfaces = realize_autonomous_responses(
            autonomous_planning_result,
            perspective=perspective,
            allowed_evidence_ids=tuple(fact_id for fact_id, _ in autonomous_grounding),
            grounding_facts=tuple(autonomous_grounding),
            dependency_patterns=utterance.context.dependency_patterns,
        )
        for variant, autonomous_surface in enumerate(autonomous_surfaces):
            add(
                autonomous_surface.text,
                source="autonomous_semantic_plan",
                renderer=f"local_morphology:{variant}",
                confidence=autonomous_surface.confidence,
                key_points=tuple(
                    dict.fromkeys(
                        (
                            *(dialogue_act.objects if dialogue_act is not None else ()),
                            *understanding.content_terms[:4],
                        )
                    )
                ),
                metadata={
                    "response_move": autonomous_surface.move,
                    "stance": autonomous_surface.stance,
                    "personality_fit": autonomous_surface.personality_fit,
                    "reasoning_evidence": autonomous_surface.evidence_ids,
                    "proposition_count": autonomous_surface.proposition_count,
                    "grounding_compiled": (autonomous_surface.grounding_compiled),
                    "external_surface_used": False,
                    "authored_reply_template": False,
                    "truth_grounding": autonomous_surface.confidence,
                    "semantic_frame_fit": 0.98,
                },
            )

        await self._trace_step(
            event.event_id,
            stage="knowledge",
            component="knowledge_and_consultation",
            status="completed",
            summary="internal and consulted evidence collection completed",
            details={
                "internal_facts": internal_facts,
                "consulted_facts": consulted_facts,
                "study_terms": study_terms,
                "foreground_lookup_target": foreground_lookup_target,
                "foreground_study_attempted": foreground_study_attempted,
                "has_bound_answer": has_bound_answer,
                "answer_research_required": answer_research_required,
                "consultation_required": consultation_required,
                "autonomous_planning_required": (self._autonomous_planning_required),
                "autonomous_grounding": tuple(autonomous_grounding),
                "autonomous_plan_count": (
                    len(autonomous_planning_result.response_plans)
                    if autonomous_planning_result is not None
                    else 0
                ),
                "autonomous_plans": (
                    tuple(
                        {
                            "move": plan.move,
                            "stance": plan.stance,
                            "confidence": plan.confidence,
                            "propositions": tuple(
                                {
                                    "predicate": proposition.predicate,
                                    "predicate_features": dict(proposition.predicate_features),
                                    "arguments": tuple(
                                        {
                                            "role": argument.role,
                                            "lemma": argument.lemma,
                                            "part_of_speech": (argument.part_of_speech),
                                            "grammatical_case": (argument.grammatical_case),
                                            "preposition": argument.preposition,
                                            "modifiers": argument.modifiers,
                                        }
                                        for argument in proposition.arguments
                                    ),
                                    "clause_kind": proposition.clause_kind,
                                    "negated": proposition.negated,
                                    "response_polarity": (proposition.response_polarity),
                                    "evidence_ids": proposition.evidence_ids,
                                }
                                for proposition in plan.propositions
                            ),
                        }
                        for plan in autonomous_planning_result.response_plans
                    )
                    if autonomous_planning_result is not None
                    else ()
                ),
                "autonomous_surface_count": len(autonomous_surfaces),
                "suppressed_precomposed_sources": tuple(suppressed_precomposed_sources),
                "productive_only_replies": self._productive_only_replies,
            },
        )
        select_and_express_content()
        composer.deferred = False
        if selected_message_key is not None:
            candidates = [replace(candidate, metadata={
                **candidate.metadata, "hard_rejection": "outside_selected_message_plan",
            }) if candidate.metadata.get("message_plan_key") != selected_message_key
            else candidate for candidate in candidates]
        # Foreground research can identify a definition after an older cached
        # candidate has already been generated. Apply the same boundary to all
        # candidates before arbitration, including legacy and memory surfaces.
        candidates = [
            replace(candidate, metadata={
                **candidate.metadata, "hard_rejection": "lexical_definition_not_requested",
            }) if unrequested_definition(
                candidate.text, prompt, tuple(reference_definitions)
            ) else candidate
            for candidate in candidates
        ]
        await self._trace_step(
            event.event_id, stage="planning", component="communicative_composition",
            status="checked", summary="reader goal and thoughts precede wording and role checks",
            details={"goal": composer.goal, "attempts": composer.records},
        )
        await self._trace_step(
            event.event_id,
            stage="candidate_generation",
            component="response_sources",
            status="completed",
            summary="all response sources produced their candidates",
            details={
                "candidate_count": len(candidates),
                "candidates": tuple(
                    {
                        "candidate_id": candidate.candidate_id,
                        "source": candidate.source,
                        "renderer": candidate.renderer,
                        "text": candidate.text,
                        "source_confidence": candidate.source_confidence,
                        "required_facts": tuple(fact.text for fact in candidate.required_facts),
                        "hard_rejection": candidate.metadata.get("hard_rejection"),
                        "response_obligation": candidate.metadata.get("response_obligation"),
                        "metadata": dict(candidate.metadata),
                    }
                    for candidate in candidates
                ),
            },
        )

        focus_concepts: tuple[str, ...] = ()
        if self._context_gate is not None and event.channel_id is not None:
            relevance = await self._context_gate.measure_candidates(
                guild_id=event.guild_id,
                channel_id=event.channel_id,
                candidate_texts=tuple(item.text for item in candidates),
            )
            measured_candidates: list[ResponseCandidate] = []
            prompt_has_content = bool(content_lemmas(prompt))
            for candidate, measured in zip(candidates, relevance, strict=True):
                metadata = {**candidate.metadata, "context_relevance": measured.score}
                candidate_words = tuple(_WORD_SURFACE_RE.findall(candidate.text))
                if (
                    candidate.source == "graph"
                    and prompt_has_content
                    and measured.score < 0.08
                    and len(candidate_words) > 4
                    and not metadata.get("hard_rejection")
                ):
                    metadata["hard_rejection"] = "graph_without_prompt_relevance"
                measured_candidates.append(replace(candidate, metadata=metadata))
            candidates = measured_candidates
            focus_concepts = relevance[0].focus_concepts if relevance else ()
        decision = self._arbitrator.decide(
            clarifiable_terms=allowed_clarifications,
            event_id=event.event_id,
            prompt=prompt,
            focus_concepts=focus_concepts,
            candidates=tuple(candidates),
            recent_replies=recent_self_replies,
            profile=profile,
            structural_weights=structural_weights,
        )
        await self._trace_step(
            event.event_id,
            stage="arbitration",
            component="response_arbitrator",
            status="selected" if decision.winner is not None else "no_winner",
            summary=(
                "an initial response candidate won"
                if decision.winner is not None
                else "all initial response candidates were absent or rejected"
            ),
            details=_decision_trace_details(decision),
        )
        if decision.winner is None:
            unknown_messages = (
                (self._phrase_uncertainty(understanding),)
                if understanding.unknown_terms and not understanding.sufficient
                else ()
            )
            recovery_candidates: list[ResponseCandidate] = []
            gap_plans = (
                epistemic_gap_candidates_for_question(structured_bundle[0])
                if (
                    structured_bundle is not None
                    and _allows_epistemic_recovery(dialogue_act, structured_bundle[0])
                )
                else ()
            )
            await self._trace_step(
                event.event_id,
                stage="recovery",
                component="semantic_recovery",
                status="started",
                summary="initial arbitration had no winner; recovery was requested",
                details={
                    "unknown_messages": unknown_messages,
                    "gap_plans": tuple(
                        {
                            "kind": gap.kind,
                            "strategy": gap.strategy,
                            "confidence": gap.confidence,
                            "evidence": gap.evidence,
                            "target_terms": gap.target_terms,
                        }
                        for gap in gap_plans
                    ),
                    "dialogue_act": (
                        {
                            "kind": dialogue_act.kind,
                            "response_obligation": dialogue_act.response_obligation,
                            "subject": dialogue_act.subject,
                        }
                        if dialogue_act is not None
                        else None
                    ),
                },
            )
            for plan_index, gap_plan in enumerate(gap_plans):
                surfaces = composer.express(
                    gap_plan.thought,
                    (*gap_plan.patterns, *utterance.context.dependency_patterns),
                    limit=3,
                )
                for surface_index, surface in enumerate(surfaces):
                    coherence = evaluate_surface(surface.text)
                    if not coherence.learnable or internal_surface_reason(surface.text) is not None:
                        continue
                    recovery_candidates.append(
                        ResponseCandidate.create(
                            event_id=event.event_id,
                            source="fallback",
                            renderer=(
                                f"semantic_recovery:{gap_plan.kind}:{gap_plan.strategy}:"
                                f"{plan_index}:{surface_index}"
                            ),
                            text=surface.text,
                            speech_act=utterance.speech_act,
                            source_confidence=min(gap_plan.confidence, surface.confidence),
                            key_points=gap_plan.target_terms,
                            metadata={
                                "semantic_recovery": True,
                                "gap_kind": gap_plan.kind,
                                "gap_strategy": gap_plan.strategy,
                                "reasoning_evidence": gap_plan.evidence,
                                "role_order": surface.role_order,
                                "construction_id": surface.construction_id,
                                "authored_reply_template": False,
                            },
                        )
                    )
            if (
                not recovery_candidates
                and not self._productive_only_replies
                and not self._autonomous_planning_required
            ):
                fallback_text = self._fallback_for_plan(
                    stored_plan,
                    unknown_messages,
                    prompt=prompt,
                    dialogue_messages=utterance.context.dialogue_messages,
                )
                recovery_candidates.append(
                    ResponseCandidate.create(
                        event_id=event.event_id,
                        source="fallback",
                        renderer="conversational_repair",
                        text=fallback_text,
                        speech_act=utterance.speech_act,
                        source_confidence=0.45,
                    )
                )
            fallback_decision = ResponseArbitrator(
                minimum_score=self._arbitrator.minimum_score
            ).decide(
                clarifiable_terms=allowed_clarifications,
                event_id=event.event_id,
                prompt=prompt,
                focus_concepts=focus_concepts,
                candidates=tuple((*candidates, *recovery_candidates)),
                recent_replies=recent_self_replies,
                profile=profile,
                structural_weights=structural_weights,
                used_fallback=True,
            )
            if fallback_decision.winner is None and (
                self._autonomous_planning_required or self._productive_only_replies
                or response_requirement(prompt) is not None
            ):
                semantic_recovery_attempt = fallback_decision
                required_goal = response_requirement(prompt)
                repair_thought = (
                    unmet_requirement_thought(required_goal) if required_goal is not None
                    else (event_followup_thought(event)
                          if understanding.referential_unknown_terms else None)
                    or grounded_repair_thought(
                        structured_bundle[0], clarifiable_terms=allowed_clarifications,
                    )
                    if structured_bundle is not None else None
                )
                grounded_repair_candidates = tuple(
                    ResponseCandidate.create(
                        event_id=event.event_id,
                        source="semantic_recovery",
                        renderer=f"grounded_repair:{surface.source}:{surface_index}",
                        text=surface.text,
                        speech_act=utterance.speech_act,
                        source_confidence=min(0.86, surface.confidence),
                        metadata={
                            "semantic_recovery": True,
                            "unresolved_requirement": required_goal,
                            "gap_kind": "contextual_clarification",
                            "reasoning_evidence": repair_thought.evidence,
                            "context_relevance": 0.95,
                            "truth_grounding": 1.0,
                            "semantic_frame_fit": 0.92,
                            "role_order": surface.role_order,
                            "construction_id": surface.construction_id,
                            "authored_reply_template": False,
                        },
                    )
                    for surface_index, surface in enumerate(
                        composer.express(repair_thought, limit=1)
                        if repair_thought is not None else ()
                    )
                )
                fallback_decision = merge_recovery_decision(
                    semantic_recovery_attempt,
                    ResponseArbitrator(
                        minimum_score=self._arbitrator.minimum_score
                    ).decide(
                        clarifiable_terms=allowed_clarifications,
                        event_id=event.event_id,
                        prompt=prompt,
                        focus_concepts=focus_concepts,
                        candidates=grounded_repair_candidates,
                        recent_replies=recent_self_replies,
                        # Clarify an actual unresolved input slot. Learned dislike
                        # of past failed answers is not evidence against that slot.
                        # Grammar and other hard rejections still apply.
                        profile=ArbitrationProfile.baseline(event.guild_id),
                        structural_weights={},
                        used_fallback=True,
                    ),
                    selection_reason="grounded_clarification_after_recovery",
                )
            decision = merge_recovery_decision(decision, fallback_decision)
        decision = replace(decision, latency_ms=(perf_counter() - started) * 1000.0)
        if understanding.task is not None:
            task = understanding.task
            winner = decision.winner
            answer_evidence: tuple[str, ...] = ()
            if winner is not None and not winner.metadata.get("unresolved_requirement"):
                if winner.source in {
                    "semantic_thought", "semantic_reasoning", "discord_identity",
                    "dialogue_context", "personal_action", "runtime_state", "relation_reasoning",
                    "causal_reasoning",
                }:
                    answer_evidence = tuple(winner.metadata.get("reasoning_evidence", ()))
                elif task.goal == "define":
                    answer_evidence = tuple(f.text for f in winner.required_facts)
            completed_task = task.with_answer_evidence(answer_evidence)
            await self._trace_step(
                event.event_id, stage="reasoning", component="comprehension_task_result",
                status="resolved" if completed_task.answer_ready else "unresolved",
                summary="answer readiness requires applicable evidence, not familiar vocabulary",
                details={
                    "goal": completed_task.goal,
                    "interpretation_ready": completed_task.interpretation_ready,
                    "answer_ready": completed_task.answer_ready,
                    "answer_evidence": answer_evidence,
                    "pending": completed_task.frontier.pending,
                },
            )
        await self._trace_step(
            event.event_id,
            stage="selection",
            component="response_arbitrator",
            status="selected" if decision.winner is not None else "failed",
            summary=(
                "final response candidate selected"
                if decision.winner is not None
                else "no final response candidate selected"
            ),
            details=_decision_trace_details(decision),
        )
        if self._arbitration_repository is not None:
            try:
                await self._arbitration_repository.save_decision(
                    event,
                    decision,
                    mode=mode,
                    prompt=prompt,
                    legacy_text=legacy_text,
                    legacy_source=legacy_source,
                )
            except Exception:
                # Arbitration persistence is observability and learning state,
                # not a prerequisite for sending an already validated reply.
                _failure_logger.exception(
                    "event=%s guild=%s channel=%s actor=%s "
                    "reason=arbitration_persistence_failed decision=%s",
                    event.event_id,
                    event.guild_id,
                    event.channel_id,
                    event.actor_id,
                    decision.decision_id,
                )
        self._log_arbitration(
            event,
            decision,
            mode=mode,
            legacy_text=legacy_text,
            legacy_source=legacy_source,
        )
        if remember:
            self._remember_arbitration(event.event_id, decision)
        winner = decision.winner
        if winner is None:
            return ""
        if remember:
            self._remember_candidate_source(event.event_id, f"arbitrated:{winner.source}")
            self._remember_stable_reply(utterance.context.seed, winner.text)
            if winner.source == "position_reasoning" and position_decision is not None:
                self._pending_positions[event.event_id] = position_decision
                self._pending_positions.move_to_end(event.event_id)
                while len(self._pending_positions) > 512:
                    self._pending_positions.popitem(last=False)
            else:
                self._pending_positions.pop(event.event_id, None)
        if stored_plan is not None:
            validation = validate_response(stored_plan, winner.text)
            await self._response_plans.complete(stored_plan, validation)  # type: ignore[union-attr]
        return winner.text

    def _remember_stable_reply(self, seed: str, text: str) -> None:
        if not seed or not text:
            return
        self._stable_reply_cache[seed] = text
        self._stable_reply_cache.move_to_end(seed)
        while len(self._stable_reply_cache) > 256:
            self._stable_reply_cache.popitem(last=False)

    async def _lexical_reference_reply(
        self,
        prompt: str,
        analyses: tuple[WordAnalysis, ...],
    ) -> str | None:
        target = lexical_lookup_target(prompt)
        if target is None:
            return None
        cached = next(
            (
                item
                for item in analyses
                if target in {item.normalized_form, item.lemma}
                and (
                    item.local_meaning
                    or item.selected_meaning
                    or item.personal_interpretation
                    or item.dictionary_meanings
                )
            ),
            None,
        )
        if cached is not None and cached.dictionary_meanings:
            meaning = cached.selected_meaning or cached.dictionary_meanings[0]
            public_labels = tuple(
                label for label in cached.style_labels if label != "wiktionary_reference"
            )
            labels = f" ({', '.join(public_labels)})" if public_labels else ""
            concise = concise_reference(meaning).rstrip(". ")
            return f"«{cached.lemma or target}»{labels} — {concise}."
        if self._russian_reference is not None:
            remote = await self._russian_reference.refresh_word(target)
            if remote is not None:
                labels = f" ({', '.join(remote.style_labels)})" if remote.style_labels else ""
                concise = concise_reference(remote.meanings[0]).rstrip(". ")
                return f"«{remote.word}»{labels} — {concise}."
        if cached is None:
            return None
        local_meaning = (
            cached.local_meaning or cached.personal_interpretation or cached.selected_meaning
        )
        if not local_meaning or not is_substantive_definition(target, local_meaning):
            return None
        label = "локальный сленг сервера" if cached.local_meaning else "объяснение собеседника"
        concise = concise_reference(local_meaning).rstrip(". ")
        return f"«{cached.lemma or target}» ({label}, не словарный эталон) — {concise}."

    async def _recent_self_replies(self, event: Event) -> tuple[str, ...]:
        if event.channel_id is None:
            return ()
        context = await self._memory.channel_context(event.guild_id, event.channel_id)
        return tuple(
            content
            for item in context[-20:]
            if item.payload.get("author_is_bot") is True
            if isinstance((content := item.payload.get("content")), str)
            if content.strip()
        )[-6:]

    @staticmethod
    def _question_response_issue(
        text: str,
        *,
        response_obligation: str | None,
        dialogue_messages: tuple[str, ...],
        recent_self_replies: tuple[str, ...],
        metadata: dict[str, object],
    ) -> str | None:
        """Reject question-shaped evasions when the turn requires an answer or repair."""

        if not text.rstrip().endswith("?"):
            return None
        response_move = metadata.get("response_move")
        genuine_clarification = bool(
            metadata.get("unresolved_reference")
            or response_move in {"clarify_reference", "clarify_intended_meaning"}
        )
        if response_obligation in {
            "answer_question",
            "repair_previous_answer",
            "repair_understanding",
            "address_negative_evaluation",
            "resolve_relation",
            "resolve_self_state",
        } and not genuine_clarification:
            return "question_does_not_fulfill_answer_or_repair"
        if any(_QUESTION_OBJECTION_RE.search(message) for message in dialogue_messages[-8:]):
            return "question_loop_after_user_objection"
        if recent_self_replies and recent_self_replies[-1].rstrip().endswith("?"):
            return "consecutive_bot_question"
        return None

    @staticmethod
    def _surface_quality_issue(
        text: str,
        *,
        prompt: str,
        recent_self_replies: tuple[str, ...],
        allow_rough: bool = False,
    ) -> str | None:
        normalized = " ".join(text.casefold().split())
        if not normalized:
            return "empty_surface"
        if text.lstrip().startswith((".", "!", "?", "…", ")", "]", "}", ",", ";", ":")):
            return "broken_opening"
        bare_stance = normalized.strip(" .,!?…")
        if bare_stance in _BARE_STANCE_ANSWERS:
            return "bare_stance_without_reason"
        if _TECHNICAL_TOKEN_RE.search(text):
            return "technical_token_leak"
        if internal_surface_reason(normalized) is not None:
            return "internal_diagnostic_leak"
        if any(normalized == " ".join(item.casefold().split()) for item in recent_self_replies):
            return "recent_exact_repeat"
        if repeats_proposition(text, recent_self_replies):
            return "recent_semantic_repeat"
        prompt_normalized = prompt.casefold()
        prompt_words = tuple(_WORD_SURFACE_RE.findall(prompt_normalized))
        response_words = tuple(_WORD_SURFACE_RE.findall(normalized))
        if len(prompt_words) >= 2:
            prompt_sequence = " ".join(prompt_words)
            response_sequence = " ".join(response_words)
            prompt_topics = set(topic_keys(prompt))
            response_topics = set(topic_keys(text))
            if response_sequence == prompt_sequence:
                return "prompt_echo"
            if (
                response_sequence.endswith(prompt_sequence)
                and len(response_words) <= len(prompt_words) + 5
                and not (response_topics - prompt_topics)
            ):
                return "prompt_echo"
            if (
                prompt_topics
                and response_topics
                and all(
                    any(_same_lexical_family(response, source) for source in prompt_topics)
                    for response in response_topics
                )
                and len(response_words) <= len(prompt_words) + 1
            ):
                return "partial_prompt_echo"
        sentences = tuple(
            sentence.strip() for sentence in re.split(r"[.!?…]+", normalized) if sentence.strip()
        )
        if len(sentences) >= 2:
            opening_words = _WORD_SURFACE_RE.findall(sentences[0])
            if len(opening_words) == 1 and opening_words[0] in _BROKEN_OPENING_WORDS:
                return "broken_opening_fragment"
        if any(
            len(words := _WORD_SURFACE_RE.findall(sentence)) == 2
            and words[0] in {"это", "так", "только"}
            and words[1].endswith(("ть", "ться"))
            for sentence in sentences
        ):
            return "broken_fragment"
        coherence = evaluate_surface(text)
        issues = tuple(issue for issue in coherence.hard_issues
                       if not (allow_rough and issue in COMPOSED_SURFACE_SOFT_ISSUES))
        if issues:
            return issues[0]
        return None

    async def _resolve_unknowns_autonomously(
        self,
        event: Event,
        prompt: str,
        analyses: tuple[WordAnalysis, ...],
        understanding: UtteranceUnderstanding,
    ) -> tuple[WordAnalysis, ...]:
        """Resolve missing meanings before composing the answer, without asking a speaker."""

        if (self._russian_reference is None or is_contact_request(prompt)
                or is_identity_question(prompt)):
            return analyses
        unknown_by_term = {
            item.lemma or item.normalized_form: item
            for item in analyses
            if not item.has_usable_meaning
            if item.requires_lexical_study or item.part_of_speech in {
                "NOUN", "VERB", "INFN", "ADJF", "noun", "verb", "adjective",
            }
            if (item.lemma or item.normalized_form) not in understanding.nonblocking_unknown_terms
            if understanding.task is None
            or (item.lemma or item.normalized_form) in understanding.task.lookup_terms
        }
        if not unknown_by_term:
            return analyses
        relation_terms = {term for relation in understanding.relations for term in relation.terms()}
        explicit_target = lexical_lookup_target(prompt)
        ranked = sorted(
            unknown_by_term.values(),
            key=lambda item: (
                item.normalized_form != explicit_target,
                (item.lemma or item.normalized_form) not in relation_terms,
                -len(item.normalized_form),
            ),
        )

        async def resolve(item: WordAnalysis) -> bool:
            try:
                resolver = getattr(
                    self._russian_reference,
                    "resolve_for_context",
                    self._russian_reference.refresh_word,
                )
                lookup_term = item.lemma or item.normalized_form
                remote = await asyncio.wait_for(resolver(lookup_term), timeout=6)
                if remote is None and lookup_term != item.normalized_form:
                    remote = await asyncio.wait_for(resolver(item.normalized_form), timeout=6)
                if remote is None:
                    return False
                if self._usage_learning is not None:
                    await self._usage_learning.learn(
                        event.guild_id,
                        LexicalStudyMaterial(
                            word=item.normalized_form,
                            lemma=remote.word,
                            definition=remote.meanings[0],
                            examples=remote.examples[:5],
                            source="wiktionary",
                        ),
                        observed_at=event.utc_timestamp,
                    )
                return True
            except Exception as exc:
                _understanding_logger.warning(
                    "event=%s word=%r autonomous_lookup_failed=%s",
                    event.event_id,
                    item.normalized_form,
                    type(exc).__name__,
                )
                return False

        enqueue = getattr(self._russian_reference, "enqueue", None)
        if enqueue is not None:
            for item in ranked[4:]:
                await enqueue(item.lemma or item.normalized_form)
        learned = sum(await asyncio.gather(*(resolve(item) for item in ranked[:4])))
        if learned:
            return await self._word_analyses(event, prompt)
        return analyses

    async def _defer_pending_research(
        self, event: Event, prompt: str, context_messages: tuple[str, ...] = (),
        resolved_bindings: tuple[tuple[str, str], ...] = (),
    ) -> tuple[str, ...]:
        reference = self._russian_reference
        retry_delay = getattr(reference, "retry_delay_for", None)
        if is_contact_request(prompt) or is_identity_question(prompt):
            return ()
        analyses = await self._word_analyses(event, prompt)
        understanding = understand_utterance(
            prompt, analyses, context_messages, resolved_bindings=resolved_bindings
        )
        terms = tuple(dict.fromkeys(
            item.lemma or item.normalized_form for item in analyses
            if not item.has_usable_meaning
            if (item.lemma or item.normalized_form) not in understanding.nonblocking_unknown_terms
            if item.requires_lexical_study or item.part_of_speech in {
                "NOUN", "VERB", "INFN", "ADJF", "noun", "verb", "adjective",
            }
            if understanding.task is None or (item.lemma or item.normalized_form)
            in understanding.task.lookup_terms
        ))
        delay = await retry_delay(terms) if terms and retry_delay is not None else None
        if delay is not None:
            await self._trace_step(
                event.event_id, stage="research", component="russian_reference", status="waiting",
                summary="reply awaits reference lookup; delivery will resume the original event",
                details={"terms": terms, "retry_after": delay},
            )
            raise LexicalResearchPending(terms, delay)
        return terms

    async def _save_understanding(
        self, event: Event, understanding: UtteranceUnderstanding
    ) -> None:
        if self._understanding_repository is None:
            return
        try:
            await self._understanding_repository.save(
                event_id=event.event_id,
                guild_id=event.guild_id,
                channel_id=event.channel_id,
                understanding=understanding,
                observed_at=event.utc_timestamp,
            )
        except Exception as exc:
            _understanding_logger.warning(
                "event=%s save_failed=%s", event.event_id, type(exc).__name__
            )
        else:
            _understanding_logger.info(
                "event=%s known=%s unknown=%s relations=%s context_links=%s "
                "score=%.3f sufficient=%s",
                event.event_id,
                understanding.known_terms,
                understanding.unknown_terms,
                tuple(
                    (item.source, item.relation, item.target) for item in understanding.relations
                ),
                understanding.context_links,
                understanding.comprehension_score,
                understanding.sufficient,
            )

    async def _queue_unknowns(
        self,
        event: Event,
        analyses: tuple[WordAnalysis, ...],
        *,
        protected_terms: frozenset[str] = frozenset(),
    ) -> None:
        if self._lexicon is None:
            return
        for item in analyses[:20]:
            if item.unknown_kind is None:
                continue
            if (item.lemma or item.normalized_form) in protected_terms:
                continue
            try:
                await self._lexicon.ensure_study_queue(event, item)
            except Exception as exc:
                _understanding_logger.warning(
                    "event=%s word=%r queue_failed=%s",
                    event.event_id,
                    item.normalized_form,
                    type(exc).__name__,
                )

    @staticmethod
    def _phrase_uncertainty(understanding: UtteranceUnderstanding) -> str:
        word = next(iter(understanding.unknown_terms), "это")
        return unknown_meaning_repair(word).realize() or _TERMINAL_PROCESSING_FAILURE_REPLY

    async def _recover_response(
        self,
        event: Event,
        utterance,
        analyses: tuple[WordAnalysis, ...],
        *,
        understanding: UtteranceUnderstanding,
    ) -> str:
        """Investigate a stalled subject and compose a grounded, user-facing recovery."""

        prompt = utterance.context.prompt
        facts = tuple(utterance.context.required_facts)
        unknowns = tuple(item for item in analyses if item.unknown_kind is not None)
        similar_words: list[tuple[str, tuple[str, ...]]] = []
        if self._lexicon is not None:
            for item in unknowns[:2]:
                similar_words.append(
                    (
                        item.normalized_form,
                        await self._lexicon.find_similar_words(item.normalized_form, limit=3),
                    )
                )
        reference_target = lexical_lookup_target(prompt)
        if (not facts and not is_social_routine(prompt) and self._knowledge_service is not None
                and (understanding.task is None or understanding.task.research_allowed)):
            query = self._recovery_query(analyses, prompt, understanding)
            if query:
                facts = await self._knowledge_service.investigate(event, query)
        if reference_target and unknowns and not facts:
            self._remember_pending_concept(
                event.event_id,
                reference_target,
            )
        if unknowns and not facts and reference_target:
            target = next(
                (
                    item
                    for item in unknowns
                    if reference_target in {item.normalized_form, item.lemma}
                ),
                unknowns[0],
            )
            return (
                unknown_meaning_repair(target.surface_form).realize()
                or _TERMINAL_PROCESSING_FAILURE_REPLY
            )
        if unknowns and not facts:
            if understanding.sufficient:
                repair = repair_for_utterance(
                    analyze_structured_utterance(prompt),
                    analyze_pragmatics(prompt),
                )
                surface = repair.realize()
                if surface is not None:
                    return surface
            return self._phrase_uncertainty(understanding)
        return compose_recovery(
            RecoveryEvidence(
                prompt=prompt,
                analyses=analyses,
                facts=facts,
                similar_words=tuple(similar_words),
                context_messages=utterance.context.dialogue_messages,
            )
        )

    @staticmethod
    def _log_structured_analysis(
        event: Event,
        prompt: str,
        analyses: tuple[WordAnalysis, ...],
        entity_analysis: EntityAnalysis,
    ) -> tuple[StructuredUtterance, UtteranceSenseResolution] | None:
        """Run the new meaning structure in shadow mode; never choose the reply yet."""

        try:
            structured = analyze_structured_utterance(
                prompt,
                analyses,
                entity_analysis,
            )
        except Exception:
            _structured_logger.exception(
                "event=%s structured_analysis_failed prompt=%r",
                event.event_id,
                _one_line(prompt),
            )
            return None
        resolution = resolve_contextual_senses(structured)
        token_summary = tuple(
            (
                item.surface,
                item.corrected_form,
                item.lemma,
                item.part_of_speech,
                item.semantic_role,
                item.entity_kind,
                item.lexical_source,
            )
            for item in structured.tokens
            if item.surface[:1].isalnum()
        )
        clause_summary = tuple(
            (
                item.clause_kind,
                item.subject_indices,
                item.predicate_index,
                item.action_complement_indices,
                item.predicate_complement_indices,
                item.direct_object_indices,
                item.indirect_object_indices,
                tuple(
                    (complement.role, complement.object_indices) for complement in item.complements
                ),
                item.implicit_subject,
                item.implicit_predicate,
                item.elliptical,
                round(item.confidence, 3),
            )
            for item in structured.clauses
        )
        dependency_summary = tuple(
            (
                item.clause_index,
                item.head_index,
                item.dependent_index,
                item.relation,
            )
            for item in structured.dependencies
        )
        clause_link_summary = tuple(
            (
                item.parent_clause_index,
                item.child_clause_index,
                item.relation,
                item.marker_index,
            )
            for item in structured.clause_links
        )
        _structured_logger.info(
            "event=%s guild=%s channel=%s actor=%s confidence=%.3f "
            "tokens=%s clauses=%s dependencies=%s clause_links=%s niche=%s ambiguities=%s",
            event.event_id,
            event.guild_id,
            event.channel_id,
            event.actor_id,
            structured.confidence,
            token_summary,
            clause_summary,
            dependency_summary,
            clause_link_summary,
            structured.niche_signals,
            structured.ambiguities,
        )
        for decision in resolution.decisions:
            _sense_logger.info(
                "event=%s token_index=%s surface=%r chosen=%s confidence=%.3f "
                "margin=%.3f reason=%s candidates=%s",
                event.event_id,
                decision.token_index,
                decision.surface,
                decision.chosen.candidate_id if decision.chosen else None,
                decision.confidence,
                decision.margin,
                decision.reason,
                tuple(
                    (
                        item.candidate.candidate_id,
                        item.candidate.source,
                        item.candidate.scope,
                        round(item.total, 4),
                        item.components,
                        item.rejection_reasons,
                        _one_line(item.candidate.meaning, 240),
                    )
                    for item in decision.scores
                ),
            )
        return structured, resolution

    @staticmethod
    def _recovery_query(
        analyses: tuple[WordAnalysis, ...],
        prompt: str,
        understanding: UtteranceUnderstanding,
    ) -> str | None:
        explicit_target = lexical_lookup_target(prompt)
        if explicit_target:
            return explicit_target
        if len(understanding.unknown_terms) == 1:
            return understanding.unknown_terms[0]
        if understanding.unknown_terms and understanding.known_terms:
            return f"{understanding.unknown_terms[0]} {understanding.known_terms[0]}"
        noun = next(
            (
                item.lemma or item.normalized_form
                for item in analyses
                if item.part_of_speech in {"NOUN", "noun"}
            ),
            None,
        )
        if noun:
            return noun
        topics = topic_keys(prompt)
        return topics[-1] if topics and prompt.rstrip().endswith("?") else None

    @staticmethod
    def _trim_unrelated_tail(text: str, prompt: str) -> str:
        """Drop a newly introduced sentence when it loses the active subject."""

        parts = tuple(part for part in re.split(r"(?<=[.!?…])\s+", text.strip()) if part.strip())
        if len(parts) < 2:
            return text
        prompt_terms = set(topic_keys(prompt)) | {
            word.casefold() for word in _WORD_SURFACE_RE.findall(prompt) if len(word) > 2
        }
        first_terms = set(topic_keys(parts[0])) | {
            word.casefold() for word in _WORD_SURFACE_RE.findall(parts[0]) if len(word) > 2
        }
        if prompt_terms and not prompt_terms & first_terms:
            return text
        anchors = prompt_terms | first_terms
        kept = [parts[0]]
        link_pronouns = {"он", "она", "оно", "они", "его", "её", "ее", "их", "это", "этот"}
        for part in parts[1:]:
            words = {word.casefold() for word in _WORD_SURFACE_RE.findall(part)}
            terms = set(topic_keys(part)) | {word for word in words if len(word) > 2}
            if not (anchors & terms or link_pronouns & words):
                break
            kept.append(part)
            anchors.update(terms)
        return " ".join(kept)

    async def _create_response_plan(
        self,
        event: Event,
        utterance,
        analyses: tuple[WordAnalysis, ...],
        *,
        dialogue_act: DialogueActFrame | None = None,
    ) -> ResponsePlan | None:
        if self._response_plans is None:
            return None
        pragmatics = analyze_pragmatics(utterance.context.prompt)
        topics = (
            ()
            if is_social_routine(utterance.context.prompt)
            else self._content_topics(utterance.context.prompt, pragmatics)
        )
        state = (
            await self._conversation_memory.active(event.guild_id, event.channel_id)
            if self._conversation_memory is not None and event.channel_id is not None
            else None
        )
        unknowns = tuple(
            dict.fromkeys(
                item.normalized_form for item in analyses if item.unknown_kind is not None
            )
        )
        key_points: tuple[str, ...]
        if utterance.context.exact_answer:
            key_points = (utterance.context.exact_answer,)
        elif dialogue_act is not None and dialogue_act.objects:
            key_points = dialogue_act.objects[:4]
        else:
            key_points = tuple(utterance.context.required_facts[:2])
        tone = "cautious" if unknowns or utterance.thought.confidence < 0.5 else "interested"
        return await self._response_plans.create(
            source_event_id=event.event_id,
            author_id=event.actor_id,
            conversation_id=state.conversation_id if state else None,
            topic_ids=topics,
            message_intent=(
                dialogue_act.kind if dialogue_act is not None else utterance.speech_act
            ),
            response_goal=(
                dialogue_act.response_obligation
                if dialogue_act is not None
                else utterance.thought.objective
            ),
            key_points=key_points,
            tone=tone,
            unknowns=unknowns,
            confidence=utterance.thought.confidence,
            now=event.utc_timestamp,
        )

    async def _word_analyses(self, event: Event, prompt: str) -> tuple[WordAnalysis, ...]:
        if self._lexicon is None:
            return ()
        words = content_tokens(prompt)[:20]
        return tuple(
            await asyncio.gather(
                *(
                    self._lexicon.analyze_word(
                        word,
                        guild_id=event.guild_id,
                        channel_id=event.channel_id,
                        context=prompt,
                    )
                    for word in words
                )
            )
        )

    @staticmethod
    def _fallback_for_plan(
        plan: ResponsePlan | None,
        unknown_messages: tuple[str, ...],
        *,
        prompt: str,
        dialogue_messages: tuple[str, ...] = (),
    ) -> str:
        pragmatics = analyze_pragmatics(prompt)
        structure = analyze_structured_utterance(prompt)
        contextual_question = reason_about_contextual_question(
            structure,
            dialogue_messages,
        )
        if contextual_question is not None:
            surfaces = realize_thought(
                contextual_question.thought,
                contextual_question.patterns,
            )
            if surfaces:
                return surfaces[0].text
        semantic_question = analyze_semantic_question(structure)
        dialogue_act = analyze_dialogue_act(
            structure,
            pragmatics=pragmatics,
            semantic_question=semantic_question,
            relation_question=analyze_relation_question(structure),
            self_state=analyze_self_state(structure),
        )
        obligation_surface = realize_dialogue_act(dialogue_act)
        if (
            dialogue_act.complete
            and obligation_surface is not None
            and dialogue_act.response_obligation != "continue_topic"
        ):
            return obligation_surface
        if _allows_epistemic_recovery(dialogue_act, structure):
            epistemic_gap = epistemic_gap_for_question(structure)
            surfaces = realize_thought(epistemic_gap.thought) if epistemic_gap is not None else ()
            if surfaces:
                return surfaces[0].text
        if unknown_messages:
            return unknown_messages[0]
        if plan is not None and plan.key_points and _quotation_requested(prompt):
            return " ".join(point.rstrip(". ") + "." for point in plan.key_points)
        repair = repair_for_utterance(structure, pragmatics)
        return repair.realize() or _TERMINAL_PROCESSING_FAILURE_REPLY

    @staticmethod
    def _content_topics(prompt: str, pragmatics: PragmaticAnalysis) -> tuple[str, ...]:
        topics = topic_keys(prompt)
        cue_surfaces = {
            word
            for cue in pragmatics.cues
            for word in _WORD_SURFACE_RE.findall(cue.surface.casefold())
        }
        return tuple(topic for topic in topics if topic not in cue_surfaces)

    @staticmethod
    def _topic_continuity_issue(
        text: str,
        *,
        prompt: str,
        dialogue_messages: tuple[str, ...],
        required_facts: tuple[str, ...],
    ) -> str | None:
        candidate_topics = set(content_lemmas(text))
        prompt_topics = set(content_lemmas(prompt))
        if not candidate_topics:
            return None
        recent_topics = {
            topic for message in dialogue_messages[-5:] for topic in content_lemmas(message)
        }
        active_topics = prompt_topics or recent_topics
        grounded_topics = {topic for fact in required_facts[:3] for topic in content_lemmas(fact)}
        allowed_topics = active_topics | grounded_topics
        if not allowed_topics:
            return None
        connected = sum(
            any(_same_lexical_family(candidate, active) for active in allowed_topics)
            for candidate in candidate_topics
        )
        threshold = max(1, (len(candidate_topics) + 4) // 5)
        return "topic_discontinuity" if connected < threshold else None

    async def _withhold_non_productive_failure(
        self,
        event: Event,
        reason: str,
    ) -> None:
        await self._trace_step(
            event.event_id,
            stage="recovery",
            component="productive_only_policy",
            status="withheld",
            summary="no productively realized surface was available",
            details={
                "reason": reason,
                "fixed_transport_fallback_disabled": True,
            },
        )
        return None

    async def _failure_reply(self, event: Event) -> PreparedReply | None:
        if self._productive_only_replies:
            return await self._withhold_non_productive_failure(
                event,
                "fixed_failure_surface_disabled",
            )
        text = _TERMINAL_PROCESSING_FAILURE_REPLY
        prompt = self._clean_content(event)
        unknowns = await self._unknowns_for_event(event)
        pending_concept: str | None = None
        if unknowns:
            understanding = understand_utterance(prompt, unknowns)
            await self._save_understanding(event, understanding)
            await self._queue_unknowns(event, unknowns)
            explicit_target = lexical_lookup_target(prompt)
            if explicit_target:
                unknown = next(
                    (
                        item
                        for item in unknowns
                        if explicit_target in {item.normalized_form, item.lemma}
                    ),
                    unknowns[0],
                )
                text = (
                    unknown_meaning_repair(unknown.surface_form).realize()
                    or _TERMINAL_PROCESSING_FAILURE_REPLY
                )
                pending_concept = explicit_target
            elif not understanding.sufficient:
                text = self._phrase_uncertainty(understanding)
        return PreparedReply(
            text=text,
            intent_id=f"fallback:{event.event_id}",
            tokens=tokenize(text),
            genome_signature=self._expression_engine.genome.signature,
            candidate_count=0,
            fallback=True,
            pending_concept=pending_concept,
            response_source="fallback",
        )

    async def _unknowns_for_event(self, event: Event) -> tuple[WordAnalysis, ...]:
        if self._lexicon is None:
            return ()
        prompt = self._clean_content(event)
        unknowns: list[WordAnalysis] = []
        try:
            for word in content_tokens(prompt)[:20]:
                analysis = await self._lexicon.analyze_word(
                    word,
                    guild_id=event.guild_id,
                    channel_id=event.channel_id,
                    context=prompt,
                )
                if analysis.unknown_kind is not None:
                    unknowns.append(analysis)
        except Exception as exc:
            self._log_reply_failure(event, "fallback_word_analysis_failed", exc)
        return tuple(unknowns)

    def _remember_candidate_source(self, event_id: str, source: str) -> None:
        if not event_id:
            return
        self._candidate_sources[event_id] = source
        self._candidate_sources.move_to_end(event_id)
        while len(self._candidate_sources) > 256:
            self._candidate_sources.popitem(last=False)

    def _remember_arbitration(self, event_id: str, decision: ArbitrationDecision) -> None:
        self._arbitration_decisions[event_id] = decision
        self._arbitration_decisions.move_to_end(event_id)
        while len(self._arbitration_decisions) > 512:
            self._arbitration_decisions.popitem(last=False)

    @staticmethod
    def _log_arbitration(
        event: Event,
        decision: ArbitrationDecision,
        *,
        mode: str,
        legacy_text: str | None,
        legacy_source: str | None,
    ) -> None:
        logger = logging.getLogger("debilgi.response_arbitration")
        winner_id = decision.winner.candidate_id if decision.winner else None
        disagreed = bool(
            legacy_text is not None
            and decision.winner is not None
            and " ".join(legacy_text.casefold().split())
            != " ".join(decision.winner.text.casefold().split())
        )
        logger.info(
            "event=%s mode=%s decision=%s winner=%s source=%s candidates=%s eligible=%s "
            "fallback=%s reason=%s weights=%s latency_ms=%.2f legacy_source=%s "
            "shadow_disagreed=%s",
            event.event_id,
            mode,
            decision.decision_id,
            winner_id,
            decision.winner.source if decision.winner else None,
            len(decision.scores),
            sum(item.eligible for item in decision.scores),
            decision.used_fallback,
            decision.selection_reason,
            decision.weight_version,
            decision.latency_ms,
            legacy_source,
            disagreed,
        )
        for item in decision.scores:
            logger.info(
                "decision=%s candidate=%s source=%s renderer=%s score=%.4f "
                "eligible=%s features=%s rejected=%s text=%r",
                decision.decision_id,
                item.candidate.candidate_id,
                item.candidate.source,
                item.candidate.renderer,
                item.total,
                item.eligible,
                item.features.as_dict(),
                item.rejection_reasons,
                _one_line(item.candidate.text, 240),
            )

    def _remember_pending_concept(self, event_id: str, concept: str) -> None:
        if not event_id or not concept:
            return
        self._pending_concepts[event_id] = concept
        self._pending_concepts.move_to_end(event_id)
        while len(self._pending_concepts) > 256:
            self._pending_concepts.popitem(last=False)

    @staticmethod
    def _log_reply_failure(
        event: Event,
        reason: str,
        error: Exception | None = None,
    ) -> None:
        _failure_logger.error(
            "event=%s guild=%s channel=%s actor=%s reason=%s error=%s input=%r",
            event.event_id,
            event.guild_id,
            event.channel_id,
            event.actor_id,
            reason,
            f"{type(error).__name__}: {error}" if error is not None else "none",
            _one_line(str(event.payload.get("content", ""))),
            exc_info=error is not None,
        )

    @staticmethod
    def _log_topic_rejection(event: Event, context: ReplyContext, text: str) -> None:
        _topic_logger.info(
            "event=%s guild=%s channel=%s prompt_topics=%s candidate_topics=%s "
            "rejected_candidate=%r reason=no_connected_topic",
            event.event_id,
            event.guild_id,
            event.channel_id,
            sorted(set(content_lemmas(context.prompt))),
            sorted(set(content_lemmas(text))),
            _one_line(text),
        )

    async def _knowledge_context(
        self, event: Event, prompt: str
    ) -> KnowledgeGrounding:
        if self._knowledge_service is None:
            return KnowledgeGrounding()
        evidence_for_event = getattr(self._knowledge_service, "grounding_evidence_for_event", None)
        if callable(evidence_for_event):
            return await evidence_for_event(event, prompt)
        facts, exact = await self._knowledge_service.grounding_for_event(event, prompt)
        return KnowledgeGrounding(facts=facts, exact_answer=exact)

    async def _semantic_context(self, event: Event, prompt: str) -> tuple[str, ...]:
        if self._semantic_memory is None:
            return ()
        terms = tokenize(prompt)
        if self._lexicon is not None:
            terms = (
                *terms,
                *(await self._lexicon.semantic_terms(prompt, guild_id=event.guild_id)),
            )
        return await self._semantic_memory.context(event.guild_id, terms)

    async def _conversation_context(self, event: Event, prompt: str) -> tuple[str, ...]:
        if self._conversation_memory is None or event.channel_id is None:
            return ()
        return await self._conversation_memory.contextual_summaries(
            event.guild_id, event.channel_id, topic_keys(prompt)
        )

    async def _long_term_context(self, event: Event, prompt: str) -> tuple[str, ...]:
        if self._long_term_memory is None:
            return ()
        capsules = await self._long_term_memory.recall(
            event.guild_id,
            prompt,
            channel_id=event.channel_id,
            user_id=event.actor_id,
        )
        return tuple(item.summary for item in capsules)

    async def _curiosity_question(self, event: Event) -> str | None:
        # Lexical curiosity is serviced by the study queue, never by quizzing a user.
        return None

    async def _discourse_context(self, event: Event, prompt: str) -> DiscourseContext:
        if self._discourse_memory is None:
            return DiscourseContext((), (), (), None, 0)
        return await self._discourse_memory.context_for(event, prompt)

    async def _reaction_context(self, event: Event) -> SocialReaction | None:
        if self._social_reactions is None:
            return None
        return await self._social_reactions.reaction_for_event(event.event_id)

    async def _relation_context(self, event: Event) -> RelationState | None:
        if self._social_reactions is None or event.actor_id is None:
            return None
        return await self._social_reactions.relation_state(event.guild_id, event.actor_id)

    async def _reflection_context(self, event: Event):
        if self._reflection is None:
            return None
        return await self._reflection.guidance(event.guild_id)

    async def _literacy_context(self, prompt: str):
        return self._literacy.context_for_text(prompt) if self._literacy else ()

    def _recent_texts(
        self,
        events: Iterable[Event],
        limit: int,
        *,
        include_bot: bool = False,
    ) -> tuple[str, ...]:
        texts = [
            text
            for item in events
            if include_bot or item.payload.get("author_is_bot") is not True
            if (text := self._clean_content(item))
        ]
        return tuple(texts[-limit:])

    def _recent_consultation_turns(
        self,
        events: Iterable[Event],
        limit: int,
    ) -> tuple[str, ...]:
        """Preserve speaker roles for deixis and cross-turn referents.

        A plain list of strings makes ``я`` and ``ты`` ambiguous to an
        external teacher.  The role prefix is metadata, not authored dialogue
        and never enters the local response generator.
        """

        turns = [
            f"{'agent' if item.payload.get('author_is_bot') is True else 'user'}: {text}"
            for item in events
            if (text := self._clean_content(item))
        ]
        return tuple(turns[-limit:])

    def _clean_content(self, event: Event) -> str:
        content = event.payload.get("content")
        if not isinstance(content, str):
            return ""
        rendered = render_message_content(content, event.payload)
        cleaned = _WHITESPACE_RE.sub(" ", rendered).strip()
        return DEFAULT_ADDRESSING.analyze(cleaned).content_text[: self._context_character_limit]


def _same_lexical_family(left: str, right: str) -> bool:
    """Catch shallow echoes that merely swap aspect or inflection of the same verb."""

    return same_lexical_family(left, right)


def _realize_consulted_definition(term: str, fact: str) -> str:
    """Turn an internal teacher fact into a local lexical construction."""

    concise = concise_reference(fact).strip()
    normalized_term = " ".join(term.casefold().replace("ё", "е").split())
    opening = " ".join(concise.casefold().replace("ё", "е").split())
    if opening.startswith(f"{normalized_term} —") or opening.startswith(f"«{normalized_term}» —"):
        return concise
    return f"«{normalized_term}» — {concise.rstrip('. ')}."


def _one_line(value: str, limit: int = 600) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()[:limit]


def _stable_response_seed(
    prompt: str,
    dialogue_chain: tuple[str, ...] = (),
    facts: tuple[str, ...] = (),
    *,
    scope: str = "",
) -> str:
    """Keep wording stable for the same semantic input, independent of Discord IDs."""

    normalized_prompt = " ".join(tokenize(prompt))
    normalized_chain = (" ".join(tokenize(item)) for item in dialogue_chain[-6:])
    normalized_facts = (" ".join(tokenize(item)) for item in facts[:3])
    material = "\n".join((scope, normalized_prompt, *normalized_chain, *normalized_facts))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _preference_question_thought(
    prompt: str,
    understanding: UtteranceUnderstanding,
    *,
    analyses: tuple[WordAnalysis, ...] = (),
    facts: tuple[str, ...] = (),
    perspective: PersonalityPerspective | None = None,
) -> ClauseThought | None:
    """Build a stable self-preference as semantics, leaving wording to NLG."""

    preference_actions = {"любить", "нравиться", "предпочитать", "увлекаться"}
    structure = analyze_structured_utterance(prompt)
    question_clauses = tuple(
        clause for clause in structure.clauses if clause.clause_kind == "question"
    )
    binary_preference_question = bool(question_clauses) and not any(
        clause.question_target_index is not None for clause in question_clauses
    )
    predicate_lemmas = {
        structure.tokens[clause.predicate_index].lemma
        for clause in structure.clauses
        if clause.predicate_index is not None
    }
    explicit_opinion_directive = (
        "мнение" in {token.lemma for token in structure.tokens}
        and bool(predicate_lemmas & {"сказать", "высказать", "выразить", "поделиться"})
        and bool(predicate_lemmas & {"изучить", "исследовать", "узнать", "разобраться"})
    )
    # A preference proposition answers a polar question only.  Previously the
    # presence of ``любить`` also captured ``почему не любишь`` and even
    # ``какая причина``, turning the requested reason into the loved object.
    if not binary_preference_question and not explicit_opinion_directive:
        return None
    relation = next(
        (
            item
            for item in understanding.relations
            if item.source in {"ты", "вы"}
            and item.relation in preference_actions
            and item.target not in {"я", "ты", "вы"}
        ),
        None,
    )
    topic = relation.target if relation is not None else None
    if topic is None and explicit_opinion_directive:
        topic = _explicit_study_topic(structure)
    if topic is None:
        prompt_words = set(tokenize(prompt))
        implicit_second_person = (
            binary_preference_question
            and not prompt_words & {"я", "мы"}
            and bool(set(understanding.content_terms) & preference_actions)
        )
        if not implicit_second_person:
            return None
        topic = next(
            (
                term
                for term in understanding.content_terms
                if term not in preference_actions
                and term not in {"я", "мы", "ты", "вы", "что", "кто", "как"}
            ),
            None,
        )
    if topic is None:
        return None
    appraisal = appraise_preference(
        topic,
        _preference_evidence(topic, analyses, facts),
        perspective=perspective,
    )
    return preference_position_thought(appraisal)


def _explicit_study_topic(structure: StructuredUtterance) -> str | None:
    study_predicates = {"изучить", "исследовать", "узнать", "разобраться"}
    for clause in structure.clauses:
        if clause.predicate_index is None:
            continue
        if structure.tokens[clause.predicate_index].lemma not in study_predicates:
            continue
        candidate_indices = list(clause.direct_object_indices)
        for link in structure.clause_links:
            if link.parent_clause_index != clause.clause_index or not link.relation.startswith(
                "ccomp"
            ):
                continue
            child = structure.clauses[link.child_clause_index]
            candidate_indices.extend(
                (
                    *child.subject_indices,
                    *child.direct_object_indices,
                    *child.predicate_complement_indices,
                )
            )
        topic = next(
            (
                structure.tokens[index].lemma
                for index in dict.fromkeys(candidate_indices)
                if structure.tokens[index].lemma != "мнение"
                and structure.tokens[index].semantic_role
                not in {"question_target", "function_word"}
            ),
            None,
        )
        if topic is not None:
            return topic
    return None


def _preference_reason(
    prompt: str,
    understanding: UtteranceUnderstanding,
    *,
    analyses: tuple[WordAnalysis, ...] = (),
    facts: tuple[str, ...] = (),
    perspective: PersonalityPerspective | None = None,
) -> ClauseThought | None:
    structure = analyze_structured_utterance(prompt)
    cause_question = any(
        clause.clause_kind == "question"
        and clause.question_target_index is not None
        and structure.tokens[clause.question_target_index].lemma in {"почему", "отчего"}
        for clause in structure.clauses
    ) or (
        {token.lemma for token in structure.tokens} & {"причина", "основание", "повод"}
        and any(clause.clause_kind == "question" for clause in structure.clauses)
    )
    if not cause_question:
        return None
    preference_predicates = {"любить", "нравиться", "предпочитать"}
    if not any(
        clause.predicate_index is not None
        and structure.tokens[clause.predicate_index].lemma in preference_predicates
        for clause in structure.clauses
    ):
        return None
    topic = next(
        (
            structure.tokens[index].lemma
            for clause in structure.clauses
            if clause.predicate_index is not None
            and structure.tokens[clause.predicate_index].lemma in preference_predicates
            for index in clause.direct_object_indices
        ),
        None,
    )
    if topic is None:
        topic = next(
            (
                term
                for term in reversed(understanding.context_links)
                if term not in preference_predicates
            ),
            None,
        )
    if topic is None:
        return None
    appraisal = appraise_preference(
        topic,
        _preference_evidence(topic, analyses, facts),
        perspective=perspective,
    )
    words = {item.casefold().replace("ё", "е") for item in tokenize(prompt)}
    return preference_reason_thought(
        appraisal,
        correcting_negative_premise="не" in words,
    )


def _preference_evidence(
    topic: str,
    analyses: tuple[WordAnalysis, ...],
    facts: tuple[str, ...],
) -> tuple[str, ...]:
    semantic_descriptions = tuple(
        description
        for item in analyses
        if topic in {item.lemma, item.normalized_form}
        for description in (
            item.selected_meaning,
            item.local_meaning,
            item.personal_interpretation,
            *item.dictionary_meanings,
        )
        if description
    )
    relevant_facts = tuple(
        fact for fact in facts if topic in content_lemmas(fact) or topic in content_tokens(fact)
    )
    return tuple(dict.fromkeys((*semantic_descriptions, *relevant_facts)))


def _answer_reference_thought(
    understanding: UtteranceUnderstanding,
) -> ClauseThought | None:
    """Compose a first-person answer about checking an answer against its question."""

    terms = set(understanding.content_terms)
    if not terms.intersection({"ответ", "ответить"}) or "вопрос" not in terms:
        return None
    predicate = next(
        (
            term
            for term in understanding.content_terms
            if term in {"сверить", "сравнить", "сопоставить", "проверить"}
        ),
        "сопоставить",
    )
    return ClauseThought(
        predicate=ConceptLexeme(
            predicate,
            predicate,
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
                        "ответ",
                        "ответ",
                        "noun",
                        (("number", "singular"), ("gender", "masculine")),
                    ),
                    grammatical_case="accusative",
                ),
            ),
            SemanticDependency(
                "topic",
                ConceptArgument(
                    ConceptLexeme(
                        "вопрос",
                        "вопрос",
                        "noun",
                        (("number", "singular"), ("gender", "masculine")),
                    ),
                    preposition="с",
                    grammatical_case="instrumental",
                ),
            ),
        ),
        response_polarity=True,
        confidence=max(0.96, understanding.comprehension_score),
        evidence=("answer_question_reference",),
    )
