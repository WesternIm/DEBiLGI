"""Narrow opt-in bridge from a stored intent to a harmless text reply."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from time import perf_counter

from debilgi.autonomy.policy import InitiativePolicy
from debilgi.cognition.affect import AffectRepository
from debilgi.cognition.core_repository import CognitiveRepository
from debilgi.cognition.dialogue_tasks import DialogueTaskRepository, task_candidate_metadata
from debilgi.cognition.models import ActionIntent, ActionType
from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.events.models import Event, EventType
from debilgi.experience.consolidation import ExperienceConsolidator
from debilgi.experience.meaning_intents import MeaningIntentRepository, delivered_basis
from debilgi.experience.models import PreparedReply
from debilgi.knowledge.models import KnowledgeGrounding
from debilgi.knowledge.service import KnowledgeAcquisitionService
from debilgi.language.addressing import DEFAULT_ADDRESSING
from debilgi.language.arbitration import (
    ArbitrationDecision,
    ArbitrationProfile,
    ResponseArbitrator,
    ResponseCandidate,
)
from debilgi.language.arbitration_repository import ArbitrationRepository
from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.assertion_experience import AssertionExperienceRepository
from debilgi.language.assertion_memory import build_assertion_memory, plan_assertion_continuation
from debilgi.language.clarification_policy import lexical_question_target, requests_word_definition
from debilgi.language.communication import ResponseComposer, plan_issues
from debilgi.language.content import content_lemmas, content_tokens
from debilgi.language.context_gate import WorkingMemoryContextGate
from debilgi.language.conversation import ConversationRepository
from debilgi.language.conversational_repair import repair_for_utterance
from debilgi.language.dialogue_obligations import exchange_from_events
from debilgi.language.dialogue_state import analyze_dialogue_state
from debilgi.language.discourse import DiscourseContext, DiscourseMemoryRepository, topic_keys
from debilgi.language.entities import render_message_content
from debilgi.language.generative_planner import (
    compatible_response_decisions,
    decide_response_mode,
    knowledge_limit_plan,
    plan_from_evidence,
    resolve_evidence_context,
    response_act_plan,
    response_self_questions,
)
from debilgi.language.interlocutor_intent import hypothesis_data
from debilgi.language.lexicon import LexiconService, WordAnalysis
from debilgi.language.open_questions import OpenLanguageQuestionService
from debilgi.language.planning import (
    ReplyContext,
    StructuredThought,
    thought_from_intent,
    utterance_plan,
)
from debilgi.language.pragmatics import analyze_pragmatics
from debilgi.language.processing_outcomes import ProcessingOutcomeRepository
from debilgi.language.proposition_memory import PropositionMemory
from debilgi.language.recovery import is_identity_question
from debilgi.language.reference_answers import unrequested_definition
from debilgi.language.request_focus import supported_alternative
from debilgi.language.research_wait import LexicalResearchPending
from debilgi.language.response_requirements import response_requirement
from debilgi.language.response_trace import ResponseTraceRepository
from debilgi.language.semantic_assimilation import (
    SemanticAssimilationService,
    SemanticLessonGuidance,
)
from debilgi.language.semantics import SemanticMemoryRepository
from debilgi.language.surface_coherence import evaluate_surface
from debilgi.language.tokenizer import tokenize
from debilgi.language.usage_learning import LexicalUsageLearningService, UsageGuidance
from debilgi.literacy.engine import LiteracyEngine
from debilgi.memory.long_term import LongTermMemoryRepository
from debilgi.memory.short_term import ShortTermMemory
from debilgi.personality.expression import PersonalityExpressionEngine
from debilgi.personality.perspective import build_personality_perspective
from debilgi.reflection.engine import ReflectionEngine
from debilgi.social.models import RelationState, SocialReaction
from debilgi.social.repository import SocialReactionRepository

_WHITESPACE_RE = re.compile("\\s+")
_failure_logger = logging.getLogger("debilgi.reply_failures")
_identity_logger = logging.getLogger("debilgi.identity_replies")
_dialogue_state_logger = logging.getLogger("debilgi.dialogue_state")


def _decision_trace_details(decision: ArbitrationDecision) -> dict[str, object]:
    return {
        "decision_id": decision.decision_id,
        "winner_candidate_id": decision.winner.candidate_id
        if decision.winner is not None
        else None,
        "sleep_hypothesis_id": decision.winner.metadata.get("sleep_hypothesis_id")
        if decision.winner else None,
        "winner_source": decision.winner.source if decision.winner is not None else None,
        "winner_text": decision.winner.text if decision.winner is not None else None,
        "diagnostic": dict(decision.winner.metadata["diagnostic"]) if decision.winner else None,
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
                "diagnostic": dict(score.candidate.metadata["diagnostic"]),
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



def elided_subject(proposal):
    """The same plan with the subject carried by the verb form («не знаю», not «я не знаю»)."""
    from debilgi.language.productive_nlg import ConceptArgument, SemanticDependency

    claim = proposal.plan.claim
    dependencies = tuple(
        SemanticDependency(d.role, ConceptArgument(
            replace(d.argument.head, features=(*d.argument.head.features, ("elided", "true"))),
            d.argument.modifiers, d.argument.preposition, d.argument.grammatical_case,
            d.argument.postmodifiers))
        if d.role == "subject" and d.argument.head.lemma in {"я", "мы"} else d
        for d in claim.dependencies)
    return replace(proposal, plan=replace(proposal.plan, claim=replace(
        claim, dependencies=dependencies)))

class _InnerFailure:
    """The first pass failed inside (4.40): the reason, for the repair pass."""

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason


class SafeMentionReplyService:
    def __init__(
        self,
        repository: CognitiveRepository,
        memory: ShortTermMemory,
        archive_repository: ArchiveRepository,
        expression_engine: PersonalityExpressionEngine,
        experience_consolidator: ExperienceConsolidator,
        initiative_policy: InitiativePolicy,
        *,
        social_reactions: SocialReactionRepository | None = None,
        reflection: ReflectionEngine | None = None,
        literacy: LiteracyEngine | None = None,
        discourse_memory: DiscourseMemoryRepository | None = None,
        semantic_memory: SemanticMemoryRepository | None = None,
        knowledge_service: KnowledgeAcquisitionService | None = None,
        lexicon: LexiconService | None = None,
        conversation_memory: ConversationRepository | None = None,
        long_term_memory: LongTermMemoryRepository | None = None,
        self_dataset: SelfDatasetRepository | None = None,
        open_questions: OpenLanguageQuestionService | None = None,
        usage_learning: LexicalUsageLearningService | None = None,
        arbitration_repository: ArbitrationRepository | None = None,
        context_gate: WorkingMemoryContextGate | None = None,
        semantic_assimilation: SemanticAssimilationService | None = None,
        processing_outcomes: ProcessingOutcomeRepository | None = None,
        response_traces: ResponseTraceRepository | None = None,
        assertion_experiences: AssertionExperienceRepository | None = None,
        arbitration_minimum_score: float = 0.45,
        context_character_limit: int = 600,
        productive_only_replies: bool = False,
        response_deliberation_enabled: bool = True,
        proposition_memory: PropositionMemory | None = None,
        meaning_intents: MeaningIntentRepository | None = None,
        affect: AffectRepository | None = None,
        dialogue_tasks: DialogueTaskRepository | None = None,
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
        self._meaning_intents = meaning_intents
        self._interlocutor_withhold_reasons: OrderedDict[str, str] = OrderedDict()
        self._held_steps: dict[str, list[dict]] = {}
        self._affect = affect
        self._dialogue_tasks = dialogue_tasks
        self._initiative_policy = initiative_policy
        self._social_reactions = social_reactions
        self._reflection = reflection
        self._literacy = literacy
        self._discourse_memory = discourse_memory
        self._semantic_memory = semantic_memory
        self._knowledge_service = knowledge_service
        self._lexicon = lexicon
        self._conversation_memory = conversation_memory
        self._long_term_memory = long_term_memory
        self._self_dataset = self_dataset
        self._open_questions = open_questions
        self._usage_learning = usage_learning
        self._arbitration_repository = arbitration_repository
        self._context_gate = context_gate
        self._semantic_assimilation = semantic_assimilation
        self._processing_outcomes = processing_outcomes
        self._response_traces = response_traces
        self._arbitrator = ResponseArbitrator(minimum_score=arbitration_minimum_score)
        self._context_character_limit = context_character_limit
        self._productive_only_replies = productive_only_replies
        self._response_deliberation_enabled = response_deliberation_enabled
        self._proposition_memory = proposition_memory
        self._candidate_sources: OrderedDict[str, str] = OrderedDict()
        self._arbitration_decisions: OrderedDict[str, ArbitrationDecision] = OrderedDict()
        self.enabled = enabled
        self.sleep_cognition = None

    @property
    def productive_only_replies(self) -> bool:
        return self._productive_only_replies

    async def response_for(self, event: Event) -> str | None:
        prepared = await self.prepare_reply(event)
        return prepared.text if prepared else None

    async def _trace_begin(self, event: Event) -> None:
        if self._response_traces is None:
            return
        await self._response_traces.begin(event, prompt=self._clean_content(event))

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
        held = self._held_steps.get(event_id)
        if held is not None:
            held.append(dict(event_id=event_id, stage=stage, component=component, status=status,
                             summary=summary, details=details))
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
        self, event: Event, *, upstream_failure: str | None = None
    ) -> PreparedReply | None:
        """Guarantee a user-facing response for every direct, enabled mention."""
        if not self.enabled:
            await self._trace_begin(event)
            await self._trace_step(
                event.event_id,
                stage="preparation",
                component="safe_reply_service",
                status="withheld",
                summary="reply service is disabled",
                details={"reason": "replies_disabled"},
            )
            return None
        if (
            event.payload.get("bot_mentioned") is not True
            or event.payload.get("author_is_bot") is True
        ):
            await self._trace_begin(event)
            await self._trace_step(
                event.event_id,
                stage="addressing",
                component="dialogue_loop_guard",
                status="withheld",
                summary="the turn does not require an agent reply",
                details={
                    "reason": "bot_author_loop_prevented"
                    if event.payload.get("author_is_bot")
                    else "not_addressed"
                },
            )
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
            return await self._prepare_data_exploration(event)
        try:
            prepared = await self.prepare_reply(event)
        except asyncio.CancelledError:
            raise
        except LexicalResearchPending:
            return await self._prepare_data_exploration(event)
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
            return await self._prepare_data_exploration(event)
        if prepared is not None and prepared.text.strip():
            return prepared
        return await self._prepare_data_exploration(event)

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
            intent = self._direct_reply_intent(event)
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
                "dialogue_situation": context.dialogue_situation,
                "affect_features": context.affect_features,
            },
        )
        plan = utterance_plan(thought_from_intent(intent), context=context)
        if plan is None:
            intent = self._direct_reply_intent(event)
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
        text = await self._generate_evidence_reply(event, plan)
        if unrequested_definition(
            text, plan.context.prompt, plan.context.reference_definitions
        ) or requests_word_definition(text):
            self._arbitration_decisions.pop(event.event_id, None)
            text = ""
            if requests_word_definition(text) or unrequested_definition(
                text, plan.context.prompt, plan.context.reference_definitions
            ):
                text = ""
                self._arbitration_decisions.pop(event.event_id, None)
        if not text or context.personality is None:
            return None
        arbitration = self._arbitration_decisions.pop(event.event_id, None)
        candidate_source = self._candidate_sources.pop(event.event_id, "graph")
        prepared = PreparedReply(
            text=text,
            intent_id=intent.intent_id,
            tokens=tokenize(text),
            genome_signature=context.personality.genome_signature,
            candidate_count=len(arbitration.scores)
            if arbitration is not None
            else context.personality.candidate_count,
            fallback=arbitration.used_fallback
            if arbitration is not None
            else candidate_source == "fallback",
            response_source=f"arbitrated:{arbitration.winner.source}"
            if arbitration is not None and arbitration.winner is not None
            else candidate_source,
            arbitration_decision_id=arbitration.decision_id if arbitration is not None else None,
            winner_candidate_id=arbitration.winner.candidate_id
            if arbitration is not None and arbitration.winner is not None
            else None,
            arbitration_weight_version=arbitration.weight_version if arbitration is not None else 0,
            assertion_experience=arbitration.winner.metadata.get("assertion_experience")
            if arbitration is not None
            and arbitration.winner is not None
            and (arbitration.winner.text == text)
            else None,
            meaning_intention_id=arbitration.winner.metadata.get("meaning_intention_id")
            if arbitration is not None
            and arbitration.winner is not None
            and (arbitration.winner.text == text)
            else None,
            dialogue_task_id=str(arbitration.winner.metadata["dialogue_task_id"])
            if arbitration is not None
            and arbitration.winner is not None
            and (arbitration.winner.text == text)
            and arbitration.winner.metadata.get("dialogue_task_id")
            else None,
            dialogue_task_progress=float(arbitration.winner.metadata["dialogue_task_progress"])
            if arbitration is not None
            and arbitration.winner is not None
            and (arbitration.winner.text == text)
            and isinstance(arbitration.winner.metadata.get("dialogue_task_progress"), int | float)
            else None,
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

    async def prepare_initiative(self, event: Event) -> PreparedReply | None:
        if event.payload.get("author_is_bot"):
            return None
        intent = await self._repository.intent_for_event(event.event_id)
        if intent is None:
            return None
        decision = await self._initiative_policy.evaluate(event, intent)
        if not decision.should_initiate or decision.goal_id is None:
            return None
        context = await self._reply_context(event, intent)
        from debilgi.config import thought_content_enabled

        if context.exact_answer is None and not thought_content_enabled():
            # (4.38: under the flag a question about the unknown is a plan, ``curiosity_plans``)
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
        text = await self._generate_evidence_reply(event, plan)
        if text and (not text.rstrip().endswith("?")):
            repair = repair_for_utterance(context.structure, analyze_pragmatics(context.prompt))
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
        self, prompt: Event, *, response_message_id: int, reply: PreparedReply, created_at: datetime
    ) -> None:
        if self._proposition_memory is not None:
            from debilgi.language.canonical_claims import activity_thought

            await self._proposition_memory.record_agent_activity(
                prompt,
                source_key=f"receipt:{prompt.channel_id}:{response_message_id}",
                operation="send",
                thought=activity_thought("отправить", "сообщение"),
                occurred_at=created_at,
                receipt_id=response_message_id,
            )
        if self._dialogue_tasks is not None and reply.dialogue_task_id is not None:
            await self._dialogue_tasks.record_attempt(
                prompt,
                task_id=reply.dialogue_task_id,
                response_message_id=response_message_id,
                text=reply.text,
                created_at=created_at,
                predicted_progress=reply.dialogue_task_progress,
            )
        if self._meaning_intents is not None and reply.meaning_intention_id is not None:
            await self._meaning_intents.record_sent(
                prompt,
                intention_id=reply.meaning_intention_id,
                response_message_id=response_message_id,
                created_at=created_at,
                text=reply.text,
            )
        if self._assertion_experiences is not None and reply.assertion_experience is not None:
            await self._assertion_experiences.record_sent(
                prompt,
                response_message_id=response_message_id,
                proposal=reply.assertion_experience,
                created_at=created_at,
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
        if (
            pending_concept is not None
            and self._open_questions is not None
            and requests_word_definition(reply.text)
        ):
            await self._open_questions.open_for(
                prompt,
                concept=pending_concept,
                question_message_id=response_message_id,
                asked_at=created_at,
            )
        if (
            reply.fallback
            or reply.contextual_learning
            or (not evaluate_surface(reply.text).learnable)
        ):
            return
        await self._experience_consolidator.register_sent(
            prompt, response_message_id=response_message_id, reply=reply, created_at=created_at
        )
        if self._self_dataset is not None:
            await self._self_dataset.capture_exchange(
                prompt, response_message_id=response_message_id, reply=reply, created_at=created_at
            )
        if reply.initiative:
            await self._initiative_policy.autonomy.start_goal(
                prompt, response_message_id=response_message_id, reply=reply, created_at=created_at
            )

    async def _reply_context(self, event: Event, intent: ActionIntent) -> ReplyContext:
        turn_form = event.turn_form
        prompt = self._clean_content(event)
        affect_features: tuple[tuple[str, float], ...] = ()
        if self._affect is not None and event.actor_id is not None:
            affect_features = (
                await self._affect.state(event.guild_id, event.actor_id, at=event.utc_timestamp)
            ).features
        expressive_intent = replace(
            intent, decision_context={**intent.decision_context, **dict(affect_features)}
        )
        personality = (
            self._expression_engine.express(expressive_intent)
            if self._proposition_memory is None
            or event.actor_id is None
            or event.channel_id is None
            else None
        )
        if event.actor_id is None or event.channel_id is None:
            return ReplyContext(
                prompt=prompt,
                turn_form=turn_form,
                source_event_id=event.event_id,
                input_meaning=event.meaning,
                seed=_stable_response_seed(prompt, scope=str(event.guild_id)),
                personality=personality,
                perspective=build_personality_perspective(personality),
                affect_features=affect_features,
            )
        if self._proposition_memory is not None:
            turns, lesson, reaction, relation, grounding = await asyncio.gather(
                self._proposition_memory.recent_conversation(event),
                self._semantic_lesson_context(event, prompt),
                self._reaction_context(event),
                self._relation_context(event),
                self._knowledge_context(event, prompt),
            )
            addressed = tuple(
                t
                for t in turns
                if t.actor_id == event.actor_id
                and (not t.payload.get("author_is_bot"))
                or (
                    t.payload.get("author_is_bot")
                    and t.actor_id == event.payload.get("bot_user_id")
                    and (t.target_id == event.actor_id)
                )
            )
            messages = tuple(self._clean_content(t) for t in addressed)
            consultation = self._recent_consultation_turns(addressed, limit=100)
            state = replace(
                analyze_dialogue_state(prompt, messages, structure=event.meaning),
                exchange=exchange_from_events(turns, event),
            )
            situation = (
                await self._dialogue_tasks.deliberate(
                    event, prompt=prompt, consultation_messages=consultation, dialogue_state=state
                )
                if self._dialogue_tasks is not None
                else None
            )
            expressive_intent = replace(
                intent,
                decision_context={
                    **intent.decision_context,
                    **dict(affect_features),
                    **(dict(situation.features) if situation else {}),
                },
            )
            personality = self._expression_engine.express(expressive_intent)
            return ReplyContext(
                prompt=prompt,
                turn_form=turn_form,
                source_event_id=event.event_id,
                input_meaning=event.meaning,
                seed=_stable_response_seed(prompt, scope=f"{event.guild_id}:{event.channel_id}"),
                personality=personality,
                perspective=build_personality_perspective(
                    personality, reaction=reaction, relation=relation
                ),
                reaction=reaction,
                relation=relation,
                dialogue_messages=messages,
                consultation_messages=consultation,
                dialogue_state=state,
                dialogue_situation=situation,
                affect_features=affect_features,
                assertion_memory=build_assertion_memory(turns, event),
                dependency_patterns=lesson.dependency_patterns,
                learned_syntax_patterns=lesson.syntax_patterns,
                learned_constructions=lesson.constructions,
                knowledge_facts=grounding.facts,
                required_facts=grounding.facts,
                exact_answer=grounding.exact_answer,
            )
        (
            working_events,
            reply_heads,
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
                event.guild_id, channel_id=event.channel_id, user_id=event.actor_id, prompt=prompt
            ),
            self._archive_repository.matching_reply_heads(event.guild_id, tokenize(prompt)),
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
        working_events = tuple(
            item for item in working_events if item.utc_timestamp <= event.utc_timestamp
        )
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
            (
                item
                for item in working_events
                if item.event_id != event.event_id
                and (
                    item.actor_id == event.actor_id
                    or (
                        item.event_type == EventType.BOT_REPLIED
                        and item.target_id == event.actor_id
                    )
                    or (item.payload.get("author_is_bot") and item.target_id == event.actor_id)
                )
            ),
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
        dialogue_state = analyze_dialogue_state(prompt, dialogue_messages, structure=event.meaning)
        dialogue_state = replace(
            dialogue_state, exchange=exchange_from_events(tuple(working_events), event)
        )
        dialogue_situation = (
            await self._dialogue_tasks.deliberate(
                event,
                prompt=prompt,
                consultation_messages=consultation_messages,
                dialogue_state=dialogue_state,
            )
            if self._dialogue_tasks is not None
            else None
        )
        if dialogue_situation is not None:
            expressive_intent = replace(
                intent,
                decision_context={
                    **intent.decision_context,
                    **dict(affect_features),
                    **dict(dialogue_situation.features),
                },
            )
            personality = self._expression_engine.express(expressive_intent)
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
            reference_definitions = tuple(
                dict.fromkeys((*reference_definitions, *resolved_grounding.definitions))
            )
            semantic_facts = tuple(dict.fromkeys((*semantic_facts, *resolved_semantic)))
            knowledge_facts = tuple(dict.fromkeys((*knowledge_facts, *resolved_facts)))
            knowledge_answer = knowledge_answer or resolved_answer
        _dialogue_state_logger.info(
            "event=%s transition=%s reason=%s active=%s references=%s added_terms=%s "
            "repair_target=%r retrieval_query=%r confidence=%.3f",
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
            *(
                fact
                for fact in knowledge_facts
                if not unrequested_definition(
                    fact, dialogue_state.repair_target or prompt, reference_definitions
                )
            ),
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
            personality, reaction=reaction, relation=relation, reflection=reflection
        )
        return ReplyContext(
            prompt=prompt,
            turn_form=turn_form,
            source_event_id=event.event_id,
            input_meaning=event.meaning,
            reference_definitions=reference_definitions,
            memory_messages=(
                *recent_working_messages,
                *conversation_summaries,
                *long_term_summaries,
            ),
            dialogue_messages=dialogue_messages,
            consultation_messages=consultation_messages,
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
            affect_features=affect_features,
            dialogue_situation=dialogue_situation,
        )

    async def _usage_context(self, event: Event, prompt: str) -> tuple[UsageGuidance, ...]:
        if self._usage_learning is None:
            return ()
        return await self._usage_learning.guidance_for_text(event.guild_id, prompt)

    async def _semantic_lesson_context(self, event: Event, prompt: str) -> SemanticLessonGuidance:
        if self._semantic_assimilation is None:
            return SemanticLessonGuidance()
        return await self._semantic_assimilation.guidance_for_prompt(event.guild_id, prompt)

    @staticmethod
    def _direct_reply_intent(event: Event) -> ActionIntent:
        """An observed direct address permits a reply even without a stored intent."""
        return ActionIntent(
            intent_id=f"local-response:{event.event_id}",
            guild_id=event.guild_id,
            action_type=ActionType.SHORT_REPLY,
            target_id=event.actor_id,
            channel_id=event.channel_id,
            urgency=0.5,
            confidence=0.25,
            expected_effect={"hypothesis": "respond_to_observed_message"},
            reasoning_trace=({"source": "direct_address", "event_id": event.event_id},),
            evidence=(event.event_id,),
            decision_context={"state_uncertainty": 1.0},
            created_at=event.utc_timestamp,
            expires_at=event.utc_timestamp + timedelta(minutes=5),
        )

    async def _prepare_data_exploration(self, event: Event) -> PreparedReply | None:
        """Enter the same meaning policy once when upstream preparation failed."""
        if event.event_id in self._arbitration_decisions:
            self._arbitration_decisions.pop(event.event_id, None)
            return await self._withhold_non_productive_failure(
                event,
                self._interlocutor_withhold_reasons.pop(
                    event.event_id, "unrealizable_selected_plan"
                ),
            )
        intent = self._direct_reply_intent(event)
        affect_features: tuple[tuple[str, float], ...] = ()
        if self._affect is not None and event.actor_id is not None:
            affect_features = (
                await self._affect.state(event.guild_id, event.actor_id, at=event.utc_timestamp)
            ).features
            intent = replace(
                intent, decision_context={**intent.decision_context, **dict(affect_features)}
            )
        dialogue_situation = (
            await self._dialogue_tasks.situation(event.guild_id, event.channel_id, event.actor_id)
            if self._dialogue_tasks is not None
            and event.channel_id is not None
            and (event.actor_id is not None)
            else None
        )
        context = ReplyContext(
            prompt=self._clean_content(event),
            seed=event.event_id,
            personality=self._expression_engine.express(intent),
            source_event_id=event.event_id,
            input_meaning=event.meaning,
            dialogue_situation=dialogue_situation,
            affect_features=affect_features,
        )
        plan = utterance_plan(thought_from_intent(intent), context=context)
        try:
            await self._generate_evidence_reply(event, plan)
        except Exception:
            _failure_logger.exception("event=%s reason=query_execution_failed", event.event_id)
            return await self._withhold_non_productive_failure(event, "query_execution_failed")
        decision = self._arbitration_decisions.get(event.event_id)
        if decision is None:
            return await self._withhold_non_productive_failure(event, "unrealizable_selected_plan")
        winner = decision.winner
        self._arbitration_decisions.pop(event.event_id, None)
        self._candidate_sources.pop(event.event_id, None)
        if winner is None:
            return await self._withhold_non_productive_failure(
                event,
                self._interlocutor_withhold_reasons.pop(
                    event.event_id, "exploration_did_not_fulfill_dialogue_task"
                ),
            )
        return PreparedReply(
            text=winner.text,
            intent_id=intent.intent_id,
            tokens=tokenize(winner.text),
            genome_signature=context.personality.genome_signature,
            candidate_count=len(decision.scores),
            fallback=False,
            response_source=f"arbitrated:{winner.source}",
            meaning_intention_id=winner.metadata.get("meaning_intention_id"),
            arbitration_decision_id=decision.decision_id,
            winner_candidate_id=winner.candidate_id,
            arbitration_weight_version=decision.weight_version,
            dialogue_task_id=str(winner.metadata["dialogue_task_id"])
            if winner.metadata.get("dialogue_task_id")
            else None,
            dialogue_task_progress=float(winner.metadata["dialogue_task_progress"])
            if isinstance(winner.metadata.get("dialogue_task_progress"), int | float)
            else None,
        )

    async def _generate_evidence_reply(
        self, event: Event, utterance, *, remember: bool = True
    ) -> str:
        from debilgi.language.semantic_graph import canonical_meanings

        canon = await self._lexicon.canonical_lemmas(event.guild_id) if self._lexicon else {}
        with canonical_meanings(canon):
            return await self._generate_canonical_reply(event, utterance, remember=remember)

    async def _generate_canonical_reply(
        self, event: Event, utterance, *, remember: bool = True
    ) -> str:
        """One local evidence path; no fallback to act-specific semantic scripts.

        With the flag (4.40) an inner failure of the first pass (no observed social form, an
        unrealizable plan) is not the end: a second pass offers only a question about what was
        not understood. The trace keeps one decision per event: the steps of a first pass that
        is repeated are not written.
        """
        from debilgi.config import thought_content_enabled

        if not thought_content_enabled():
            return await self._canonical_pass(event, utterance, remember=remember)
        held = self._held_steps[event.event_id] = []
        try:
            result = await self._canonical_pass(event, utterance, remember=remember,
                                                may_repair=True)
        except BaseException:
            self._held_steps.pop(event.event_id, None)
            await self._flush_steps(held)
            raise
        self._held_steps.pop(event.event_id, None)
        if isinstance(result, _InnerFailure):
            return await self._canonical_pass(event, utterance, remember=remember,
                                              repair=result.reason)
        await self._flush_steps(held)
        return result

    async def _repair_possible(self, event, context, failure, irritation, boredom, allowed) -> bool:
        learning = self._semantic_assimilation.dialogue_learning
        return bool(await learning.repair_plans(
            event, context, failure, allowed=allowed, irritation=irritation, boredom=boredom))

    async def _flush_steps(self, held: list) -> None:
        for step in held:
            await self._trace_step(**step)

    async def _canonical_pass(
        self, event: Event, utterance, *, remember: bool = True, repair: str | None = None,
        may_repair: bool = False,
    ) -> str:
        """One pass of the evidence path (see ``_generate_canonical_reply``)."""
        started = perf_counter()
        context = utterance.context
        if self._semantic_assimilation is not None:
            context = replace(
                context,
                turn_form=await self._semantic_assimilation.dialogue_learning.turn_form_for(event),
            )
            context = replace(
                context,
                prior_bot_turn=await (
                    self._semantic_assimilation.dialogue_learning.prior_response_for(event)
                ),
            )
        elif context.turn_form is None:
            context = replace(context, turn_form=event.turn_form)
        await self._trace_step(
            event.event_id,
            stage="input",
            component="turn_form",
            status="observed",
            summary="turn shape and address precede lexical evidence retrieval",
            details={"form": context.turn_form},
        )
        context = resolve_evidence_context(context)
        if (
            context.turn_form
            and context.turn_form.kind == "repair"
            and context.prior_bot_turn
            and (
                context.focus_reason
                not in {"answer_control_keeps_request", "open_request_continuation"}
            )
        ):
            context = replace(
                context,
                prompt=self._clean_content(event),
                input_meaning=event.meaning,
                response_options=(),
                focus_reason="own_delivered_turn_repair",
            )
        context = replace(
            context,
            source_event_id=event.event_id,
            meaning_goals=await self._meaning_intents.learned_goals(event)
            if self._meaning_intents is not None
            else (),
        )
        partner_hypotheses = ()
        partner_repair_allowed = True
        partner_rejected_relations = frozenset()
        if self._semantic_assimilation is not None:
            partner = self._semantic_assimilation.dialogue_learning.interlocutor
            partner_hypotheses = await partner.infer(event, context.prompt)
            partner_repair_allowed = await partner.clarification_allowed(event)
            partner_rejected_relations = await partner.rejected_relations(event)
        if self._proposition_memory is not None:
            query = (
                context.dialogue_state.retrieval_query if context.dialogue_state else context.prompt
            )
            gathered = await asyncio.gather(
                self._proposition_memory.retrieve_claims(
                    event, context.prompt, hypotheses=partner_hypotheses
                ),
                self._proposition_memory.relevant_utterances(event, query),
                self._proposition_memory.learned_word_relations(event),
                self._proposition_memory.relevant_dialogue_reports(event, query),
                self._proposition_memory.learned_meanings(event),
                self._proposition_memory.relevant_reports(event, query, current_only=True),
            )
            claims, corpus, relations, dialogue_reports, meanings, reports = gathered
            context = replace(
                context,
                proposition_reports=reports,
                claim_result=claims,
                corpus_utterances=corpus,
                word_relations=relations,
                dialogue_reports=dialogue_reports,
                meaning_records=meanings,
            )
        utterance = replace(utterance, context=context)
        analyses = await self._word_analyses(event, context.prompt)
        planning = plan_from_evidence(
            context, analyses, explore_meanings=self._meaning_intents is not None
        )
        if self._semantic_assimilation is not None:
            # «Почему?» after the agent's own play: own up to it or go on with the game.
            play_followup = await (
                self._semantic_assimilation.dialogue_learning.fantasy_followup_plans(
                    event, context
                )
            )
            variable = None if play_followup else await (
                self._semantic_assimilation.dialogue_learning.variable_plan(event, context)
            )
            reported_check = None if variable is not None or play_followup else await (
                self._semantic_assimilation.dialogue_learning.verify_reported_position(
                    event, context
                )
            )
            if play_followup:
                planning = replace(planning, proposals=play_followup)
            elif variable is not None:
                planning = replace(planning, proposals=(variable,))
            elif reported_check is not None:
                planning = replace(planning, proposals=(reported_check,))
            elif held_position := await (
                self._semantic_assimilation.dialogue_learning.own_position_plans(
                    event, context, planning
                )
            ):
                planning = replace(planning, proposals=held_position)
            elif (opined := await self._opinion_plans(event, context, planning)):
                # A view of its own (4.45) next to the honest limit, which stays if it loses —
                # but not once the stance is held: it has said what it thinks, "I do not know"
                # would be untrue.
                held_view = any(json.loads(dict(p.plan.hypothesis).get("basis") or "{}").get("held")
                                for p in opined)
                planning = replace(planning, proposals=opined if held_view
                                   else (*opined, *planning.proposals))
            elif supposed := await (
                self._semantic_assimilation.dialogue_learning.supposition_plans(
                    event, context, planning
                )
            ):
                planning = replace(planning, proposals=supposed)
            elif made_up := await (
                self._semantic_assimilation.dialogue_learning.fantasy_plans(
                    event, context, planning
                )
            ):
                # The game competes with the honest limit, which stays if it loses.
                planning = replace(planning, proposals=(*made_up, *planning.proposals))
            elif (
                context.turn_form and context.turn_form.kind == "repair" and context.prior_bot_turn
            ):
                from debilgi.language.dialogue_learning import own_turn_plan

                delivered_plan = own_turn_plan(context, None)
                if delivered_plan is not None:
                    planning = replace(planning, proposals=(delivered_plan,))
        if self._semantic_assimilation is not None:
            await self._trace_step(
                event.event_id,
                stage="intent",
                component="interlocutor_model",
                status="inferred" if partner_hypotheses else "unresolved",
                summary="partner goals inferred from observed response graphs before realization",
                details={
                    "hypotheses": hypothesis_data(partner_hypotheses),
                    "graph_query": context.claim_result.query if context.claim_result else None,
                    "max_inference_depth": 1,
                    "max_examples": 2048,
                    "max_neighbours": 32,
                },
            )
        observed_forms, recent_reactions, recent_surfaces = ((), (), ())
        accepting_report = False
        # Irritation R (4.34, flag only): what this person did to the agent, by journal and clock.
        irritation = (await self._meaning_intents.irritation(event, context.prompt)
                      if self._meaning_intents is not None
                      and context.turn_form is not None and context.turn_form.addressed
                      else None)
        hostile = bool(irritation is not None and irritation.hostile)
        if self._semantic_assimilation is not None:
            learning = self._semantic_assimilation.dialogue_learning
            from debilgi.language.morphology import DEFAULT_MORPHOLOGY

            accepting_report = bool(
                context.dialogue_situation
                and any(
                    t.objective.response_options
                    and set(content_lemmas(context.prompt))
                    & {DEFAULT_MORPHOLOGY.lemma(o) for o in t.objective.response_options}
                    for t in context.dialogue_situation.open_tasks
                )
                and (
                    not any(
                        c.clause_kind in {"question", "command"} for c in context.structure.clauses
                    )
                )
            )
            from debilgi.language.canonical_claims import self_query

            # A scoped self Unknown commits to the existing epistemic reader
            # before reaction forms are considered. Avoid hydrating the whole
            # reaction projection when that branch cannot consume it.
            scoped_unknown = (
                context.claim_result is not None
                and context.claim_result.status == "Unknown"
                and self_query(context.claim_result.query)
                and decide_response_mode(context, planning, partner_hypotheses).act
                == "express_limit"
            )
            if not scoped_unknown:
                observed_forms = await learning.forms_for(
                    event, context.prompt, accepting_report=accepting_report,
                    hostile=hostile,
                )
            recent_reactions = await learning.recent_reactions(event)
            recent_surfaces = await learning.recent_reactions(
                event, minutes=60, include_repairs=True)
        hypothesis_allowed = (
            await learning.hypothesis_allowed(event)
            if self._semantic_assimilation is not None
            else False
        )
        from debilgi.language.dialogue_learning import reactive_act

        # Boredom (4.33, flag only): a function of the agent's own delivered turns and the clock.
        boredom = (await self._meaning_intents.boredom_ledger(event)
                   if self._meaning_intents is not None else None)
        hypothesis_allowed = hypothesis_allowed or (
            context.turn_form is not None
            and context.turn_form.addressed
            and (accepting_report or reactive_act(context.prompt) == "acknowledgment")
        )
        if boredom is not None and any(
                p.operation == "ask_unbound_variable" for p in planning.proposals) and not (
                partner_repair_allowed and await learning.repair_rate_ok(event, boredom)):
            # The question about the unbound variable is a repair too (4.40): it obeys the same
            # budget (not twice in a row) and the same rate (not above people's own).
            planning = replace(planning, proposals=tuple(
                p for p in planning.proposals if p.operation != "ask_unbound_variable"))
        # Curiosity (4.38, flag only): a question about the unknown topic competes with the limit.
        curious = (
            await self._semantic_assimilation.dialogue_learning.curiosity_plans(
                event, context, planning, irritation=irritation, boredom=boredom)
            if self._semantic_assimilation is not None and boredom is not None else ()
        )
        mode_decision = decide_response_mode(
            context,
            planning,
            partner_hypotheses,
            clarification_allowed=partner_repair_allowed,
            observed_reaction=any(
                f.component == "core"
                and (
                    f.clause_kind != "question"
                    or not recent_reactions
                    or (not recent_reactions[0][2])
                )
                for f in observed_forms
            ),
            hook_allowed=not recent_reactions or not recent_reactions[0][2],
            observed_forms=observed_forms,
            hypothesis_allowed=hypothesis_allowed,
        )
        # Past its patience (the genome's 1 - patience = R's own gain) the agent stops giving
        # social reactions to this person; a turn with something to answer is never ignored.
        spent_patience = irritation is not None and irritation.value > irritation.sensitivity
        if (hostile or spent_patience) and (
            hostile and not (
                mode_decision.mode == "free_form" and mode_decision.target_meaning == "retort")
            or mode_decision.mode == "free_form" and mode_decision.target_meaning != "retort"
        ):
            # An attack is answered only by a form people on this server used against an attack;
            # otherwise ignoring it is the agent's own choice, and R is its reason.
            mode_decision = replace(
                mode_decision, mode="silence_or_unknown", act="withhold",
                reason="irritation_ignored", confidence_level="silence", target_meaning=None,
                rule_id=None, allow_hook_question=False, known_slots=(), hypothesis=(),
                compatible_supports=(),
            )
        if repair is not None and self._semantic_assimilation is not None:
            repairs = await learning.repair_plans(
                event, context, repair, allowed=partner_repair_allowed,
                irritation=irritation, boredom=boredom)
            if repairs:
                mode_decision = replace(
                    mode_decision, mode="answer", act="clarify_variable",
                    reason="repair_of_failure", confidence_level="answer", target_meaning=None,
                    rule_id=None, allow_hook_question=False, known_slots=(), hypothesis=(),
                    compatible_supports=())
                planning = replace(planning, proposals=repairs)
                curious = ()
        if repair is None and self._semantic_assimilation is not None:
            received = await learning.reception_plans(event, context, irritation=irritation)
            if received:
                # The reply to the agent's own question is received (4.44), not a new topic.
                mode_decision = replace(
                    mode_decision, mode="answer", act="acknowledge_report",
                    reason="report_received", confidence_level="answer", target_meaning=None,
                    rule_id=None, allow_hook_question=False, known_slots=(), hypothesis=(),
                    compatible_supports=())
                planning = replace(planning, proposals=received)
                curious = ()
        if not mode_decision.input_features:
            from debilgi.language.dialogue_learning import turn_features

            mode_decision = replace(
                mode_decision, input_features=tuple(sorted(turn_features(context.prompt)))
            )
        requested_options = context.response_options
        context = replace(
            context,
            response_decision=mode_decision,
            response_options=() if mode_decision.mode == "free_form" else context.response_options,
        )
        utterance = replace(utterance, context=context)
        proposals = tuple(
            p
            for p in planning.proposals
            if p.plan.claim.clause_kind != "question"
            or p.operation in {"ask_unbound_variable", "repair"}
        )
        if mode_decision.mode in {"free_form", "clarification"}:
            act_plan = response_act_plan(context, mode_decision, observed_forms)
            proposals = (act_plan,) if act_plan is not None else ()
        elif mode_decision.mode == "silence_or_unknown":
            limit = knowledge_limit_plan(context) if mode_decision.act == "express_limit" else None
            if limit is not None and self._semantic_assimilation is not None:
                limit = await learning.remember_position(event, context, limit)  # (4.44)
            proposals = (limit,) if limit else ()
            if limit and curious:
                proposals = (*proposals, *curious)
        composer = ResponseComposer(context.structure)
        source_answers = (
            await learning.source_answers(event)
            if mode_decision.mode == "free_form" and observed_forms
            else frozenset()
        )
        checked_surfaces, plan_decisions = ({}, {})
        deliberation_started = perf_counter()
        questions = (
            response_self_questions(context, observed_forms, proposals)
            if self._response_deliberation_enabled
            else ()
        )
        examined = []
        if mode_decision.mode == "free_form":
            plans = []
            extra_started = None
            for option in compatible_response_decisions(
                mode_decision,
                observed_forms,
                hypothesis_allowed=hypothesis_allowed,
                limit=4 if self._response_deliberation_enabled else 1,
            ):
                if examined and extra_started is None:
                    extra_started = perf_counter()
                if extra_started is not None and perf_counter() - extra_started >= 1.0:
                    break
                forms = tuple(
                    f
                    for f in observed_forms
                    if f.component != "core"
                    or (
                        f.meaning == option.target_meaning
                        and f.hypothesis == option.hypothesis
                        and (f.confidence_level == option.confidence_level)
                    )
                )
                proposal = response_act_plan(context, option, forms)
                if proposal is None:
                    continue
                surfaces = composer.compose(
                    proposal.plan,
                    (*proposal.patterns, *context.dependency_patterns),
                    exchange_forms=forms,
                    source_answers=source_answers,
                    recent_texts=tuple((t for t, _, _ in recent_surfaces)),
                    variation_key="" if boredom is not None else event.event_id,
                    limit=12,
                )
                examined.append(
                    {
                        "plan_key": proposal.plan.key,
                        "meaning": option.target_meaning,
                        "activation": option.activation,
                        "passed": bool(surfaces),
                    }
                )
                if surfaces:
                    plans.append(proposal)
                    checked_surfaces[proposal.plan.key] = surfaces
                    plan_decisions[proposal.plan.key] = option
            proposals = tuple(plans)
            if boredom is not None and proposals and self._meaning_intents is not None:
                # What this person pressed again after this operation (4.41) meets alternatives:
                # the honest limit about the very request, else a question about it (4.40).
                left_open = await self._meaning_intents.unclosed_by_operation(
                    event, planning.objective)
                # ... and what this person is predicted to meet with laughter, hostility or
                # rejection (4.42: the person's own tested reactions, never another person's)
                from debilgi.experience.meaning_intents import move_type

                unwelcome = await self._meaning_intents.unwelcome_by_operation(event)
                if any(p.operation in left_open
                       or move_type(p.operation, dict(p.plan.hypothesis).get("kind")) in unwelcome
                       for p in proposals):
                    extra = knowledge_limit_plan(context)
                    if extra is None and self._semantic_assimilation is not None:
                        extra = next(iter(await learning.repair_plans(
                            event, context, "demand_left_open", allowed=partner_repair_allowed,
                            irritation=irritation, boredom=boredom)), None)
                    if extra is not None and extra.plan.knowledge_limit is not None and not (
                            context.word_relations or context.dependency_patterns
                            or context.corpus_utterances):
                        extra = None  # (a limit with no observed wording would fail to be said)
                    if extra is not None and extra.plan.key not in {p.plan.key for p in proposals}:
                        surfaces = composer.compose(
                            extra.plan, (*extra.patterns, *context.dependency_patterns))
                        if surfaces:
                            proposals = (*proposals, extra)
                            checked_surfaces[extra.plan.key] = surfaces
        recent_reply_texts = tuple((t for t, _, _ in recent_surfaces))
        if not recent_reply_texts:
            recent_reply_texts = await self._recent_self_replies(event)
        sole_limit = [p for p in proposals if p.plan.knowledge_limit is not None]
        if len(sole_limit) == 1 and all(
            p.plan.knowledge_limit is not None
            or dict(p.plan.hypothesis).get("kind") == "fantasy"
            or p.operation in {"curiosity", "opinion"}
            for p in proposals
        ):
            from debilgi.language.dialogue_learning import response_repetition
            from debilgi.language.productive_nlg import unresolved_answer_thought

            proposal = sole_limit[0]
            surfaces = composer.compose(
                proposal.plan, (*proposal.patterns, *context.dependency_patterns)
            )

            from debilgi.language.communication import construction_repetition

            # The same words, or the same clause skeleton, as a recent reply: other forms of
            # not knowing compete (the compact limit; the limit with the person carried by
            # the verb, when people of this server are seen to say it so).
            playing = any(dict(p.plan.hypothesis).get("kind") == "fantasy" or p.operation
                          == "opinion" for p in proposals)
            if surfaces and (playing or any(
                response_repetition(s.text, recent_reply_texts) > 0
                or construction_repetition(s.text, recent_reply_texts) >= 0.25
                or boredom is not None and boredom.reading(text=s.text).surface > 0
                for s in surfaces
            )):
                limit = replace(proposal.plan.knowledge_limit, question=None)
                claim = replace(
                    unresolved_answer_thought(confidence=proposal.confidence),
                    evidence=proposal.plan.claim.evidence,
                )
                alternative = replace(
                    proposal, plan=replace(proposal.plan, claim=claim, knowledge_limit=limit)
                )
                if alternative.plan.key == proposal.plan.key:
                    claim = replace(
                        claim,
                        dependencies=tuple(d for d in claim.dependencies if d.role != "topic"),
                    )
                    alternative = replace(alternative, plan=replace(alternative.plan, claim=claim))
                forms = [alternative]
                if self._semantic_assimilation is not None and await (
                    self._semantic_assimilation.dialogue_learning.elided_limit_observed(event)
                ):
                    forms = [alternative, *(
                        elided_subject(item) for item in (proposal, alternative))]
                proposals = (*proposals, *forms)
            if boredom is not None:
                proposals = (*proposals, *self._tired_of_limit(boredom, proposal, context))
        deliberation = {
            "self_questions": questions,
            "candidates": examined,
            "candidate_count": len(examined) or len(proposals),
            "rejected_plans": sum(not p["passed"] for p in examined),
            "budget": {"max_plans": 4, "extra_seconds": 1.0, "primary_preserved": True},
            "elapsed_ms": (perf_counter() - deliberation_started) * 1000,
        }
        await self._trace_step(
            event.event_id,
            stage="intent",
            component="response_activation",
            status="selected",
            summary="response act committed before generation",
            details={
                "decision": mode_decision,
                "query": context.claim_result.query if context.claim_result is not None else None,
                "requested_response_options": requested_options,
                "internal_requests": planning.unresolved,
                "max_inference_depth": 1,
                "activation_graph": {
                    "decay": 0.78,
                    "threshold": 0.025,
                    "memory_depth": 3,
                    "rule_depth": 2,
                    "total_depth": 5,
                },
            },
        )
        selection = None
        if self._meaning_intents is not None:
            proposals = tuple(
                proposal
                for proposal in proposals
                if not plan_issues(proposal.plan)
                and (
                    proposal.plan.status == "express_limit"
                    or not context.response_options
                    or supported_alternative(
                        context.prompt, context.response_options, proposal.plan.claim
                    )
                    is not None
                )
            )
            selection = await self._meaning_intents.choose(
                event,
                proposals,
                planning.objective,
                request_text=context.prompt,
                query_result=context.claim_result,
                response_decision=mode_decision,
                parent_intent_id=utterance.thought.intention_id or "direct-reply:" + event.event_id,
                state_features=tuple(
                    dict(
                        (
                            *utterance.thought.state_features,
                            *context.affect_features,
                            *((("irritation", irritation.value),)
                              if irritation is not None else ()),
                            *(
                                context.dialogue_situation.features
                                if context.dialogue_situation is not None
                                else ()
                            ),
                        )
                    ).items()
                ),
            )
            admitted_keys = {a["plan_key"] for a in selection.alternatives} if selection else set()
            proposals = tuple(p for p in proposals if p.plan.key in admitted_keys)
            if curious and selection is not None and selection.proposal.operation != "curiosity":
                # The policy kept the limit: the inquiry did not win the drives, it is not offered.
                proposals = tuple(p for p in proposals if p.operation != "curiosity")
            if selection is not None and selection.proposal.operation != "opinion":
                # (4.45) the policy, with what this person is predicted to do to a view of its own
                # (H) and who they are for the agent (P), kept another move: no view is offered.
                proposals = tuple(p for p in proposals if p.operation != "opinion")
            if selection is not None:
                mode_decision = plan_decisions.get(selection.proposal.plan.key, mode_decision)
                is_question = selection.proposal.plan.claim.clause_kind == "question"
                utterance = replace(
                    utterance,
                    speech_act="clarifying_question" if is_question else "brief_statement",
                    thought=replace(
                        utterance.thought,
                        intention_id=selection.intention_id,
                        action_type=ActionType.ASK_QUESTION
                        if is_question
                        else ActionType.SHORT_REPLY,
                        objective=selection.expected_outcome or selection.proposal.operation,
                    ),
                )
            await self._trace_step(
                event.event_id,
                stage="intent",
                component="learned_meaning_policy",
                status="selected" if selection is not None else "unresolved",
                summary="a semantic intention was selected before wording from actual experience",
                details={
                    "selection": selection,
                    "learned_goals": context.meaning_goals,
                    "script_fallback_enabled": False,
                },
            )
        if irritation is not None:
            await self._trace_step(
                event.event_id,
                stage="intent",
                component="irritation",
                status="ignored" if mode_decision.reason == "irritation_ignored" else "read",
                summary="what this person did to the agent, decayed by the clock",
                details=irritation.as_dict(),
            )
        if boredom is not None:
            await self._trace_step(
                event.event_id,
                stage="intent",
                component="boredom",
                status="read",
                summary="repetition of the agent's own delivered turns, decayed by the clock",
                details={
                    "rows": len(boredom.rows),
                    "sensitivity": boredom.sensitivity,
                    "half_life_seconds": boredom.half_life,
                    "alternatives": [
                        {"plan_key": a["plan_key"], "operation": a["operation"],
                         "boredom": a.get("boredom")}
                        for a in (selection.alternatives if selection is not None else ())],
                },
            )
        await self._trace_step(
            event.event_id,
            stage="reasoning",
            component="evidence_planner",
            status="derived" if planning.proposals else "unresolved",
            summary="local response meanings retain their actual source propositions",
            details={
                "planning": planning,
                "input_parse": {
                    "parser_version": context.structure.parser_version,
                    "supported": context.structure.supported,
                    "participants": context.structure.participants,
                    "unresolved_slots": context.structure.unresolved_slots,
                    "corrections": context.structure.corrections,
                    "clause_count": len(context.structure.clauses),
                    "ambiguities": context.structure.ambiguities,
                    "has_tokens": bool(context.structure.tokens),
                },
                "script_fallback_enabled": False,
                "turn_prompt": context.turn_prompt,
                "focused_request": context.prompt,
                "focus_reason": context.focus_reason,
                "response_options": context.response_options,
            },
        )
        from debilgi.language.word_relations import observe_word_relations

        candidates = []
        expression_confidences = []
        active_task = (
            context.dialogue_situation.active_task
            if context.dialogue_situation is not None
            else None
        )
        for proposal in proposals:
            proposal_decision = plan_decisions.get(proposal.plan.key, mode_decision)
            alternative = (
                next((a for a in selection.alternatives if a["plan_key"] == proposal.plan.key), {})
                if selection
                else {}
            )
            preference_index = (
                next(
                    (
                        i
                        for i, a in enumerate(selection.alternatives)
                        if a["plan_key"] == proposal.plan.key
                    ),
                    0,
                )
                if selection
                else 0
            )
            totals = selection.choice.get("totals", ()) if selection else ()
            preference = totals[preference_index] if preference_index < len(totals) else 0.0
            if proposal.plan.knowledge_limit and (
                not (
                    context.word_relations
                    or context.dependency_patterns
                    or context.corpus_utterances
                )
            ):
                continue
            answer_option = (
                supported_alternative(context.prompt, context.response_options, proposal.plan.claim)
                if context.response_options
                else None
            )
            if (
                context.response_options
                and answer_option is None
                and (proposal.plan.status != "express_limit")
            ):
                continue
            message = replace(proposal.plan, answer_shape="sentence", focus_role=None)
            is_fantasy = dict(proposal.plan.hypothesis).get("kind") == "fantasy"
            for surface in checked_surfaces.get(proposal.plan.key, ()) or composer.compose(
                message,
                (*proposal.patterns, *context.dependency_patterns),
                **{
                    "exchange_forms": observed_forms,
                    "source_answers": source_answers,
                    "recent_texts": tuple((t for t, _, _ in recent_surfaces)),
                    "variation_key": "" if boredom is not None else event.event_id,
                    "limit": 12,
                }
                if mode_decision.mode == "free_form"
                else {},
            ):
                expression_confidences.append(surface.confidence)
                text = (
                    answer_option[:1].upper() + answer_option[1:] if answer_option else surface.text
                )
                if unrequested_definition(text, context.prompt, context.reference_definitions):
                    continue
                if is_fantasy and await (
                    self._semantic_assimilation.dialogue_learning.is_human_utterance(event, text)
                ):
                    continue  # make-believe is composed, never a person's words repeated
                bored = (self._boredom_of(boredom, proposal, alternative, text)
                         if boredom is not None else None)
                candidates.append(
                    ResponseCandidate.create(
                        event_id=event.event_id,
                        source="learned_dialogue_structure"
                        if mode_decision.mode == "free_form"
                        else "semantic_reasoning",
                        renderer=f"response_composer:{proposal.operation}"
                        if mode_decision.mode == "free_form"
                        else f"evidence_planner:{proposal.operation}",
                        text=text,
                        speech_act=utterance.speech_act,
                        source_confidence=surface.confidence
                        if mode_decision.mode == "free_form"
                        else proposal.confidence,
                        key_points=planning.objective.topics,
                        metadata=task_candidate_metadata(
                            active_task,
                            text,
                            {
                                **composer.metadata_for(surface.text, plan_key=message.key),
                                "authored_reply_template": False,
                                "locally_composed": True,
                                "learned_response_act": mode_decision.act
                                if mode_decision.mode == "free_form"
                                else None,
                                "role_order": surface.role_order,
                                "meaning_origin": "derived",
                                "semantic_origin": "derived",
                                "semantic_scaffold": False,
                                "reasoning_evidence": tuple(
                                    p.evidence_id for p in proposal.premises
                                ),
                                "premises": proposal.premises,
                                "derivation": proposal.operation,
                                "claim_query_event_id": context.claim_result.query.exclude_event_id
                                if context.claim_result is not None
                                else None,
                                "hard_rejection": "corrected_lexical_exception"
                                if not message.knowledge_limit
                                and any(
                                    (edge.governor, edge.dependent, edge.relation)
                                    in partner_rejected_relations
                                    for edge in observe_word_relations(text)
                                )
                                else None,
                                "answer_slot": proposal.answer_slot,
                                "response_decision": asdict(proposal_decision),
                                "deliberation": deliberation,
                                "response_mixture": proposal_decision.mixture,
                                "known_slots": proposal.plan.known_slots
                                or proposal_decision.known_slots,
                                "hypothesis": proposal_decision.hypothesis,
                                "ambiguous_slot": proposal_decision.ambiguous_slot,
                                "alternatives": proposal_decision.alternatives,
                                "expected_information_gain": (
                                    proposal_decision.expected_information_gain
                                ),
                                "genuine_clarification": mode_decision.mode == "clarification"
                                or proposal.operation in {"ask_unbound_variable", "repair"},
                                "response_move": proposal_decision.act,
                                "epistemic_query": proposal.plan.knowledge_limit,
                                "unresolved_requirement": "explain_reason"
                                if proposal.operation
                                in {"ask_unbound_variable", "bind_delivered_variable", "repair"}
                                else "review_prior_claim"
                                if proposal.operation == "verify_reported_position"
                                and not proposal.premises
                                else None,
                                # The review of a prior claim the agent actually made.
                                "reviewed_agent_claim": proposal.operation
                                == "verify_reported_position" and bool(proposal.premises),
                                "claim_source": "dialogue"
                                if proposal.operation == "verify_reported_position" else None,
                                # A marked supposition on the agent's own record: a
                                # hypothesis, never a fact of the agent or of a person.
                                "supposition_licensed": proposal.operation == "supposition"
                                and not is_fantasy,
                                # A view of its own (4.45): a stance of the agent, never a claim,
                                # a fact or a hypothesis of the world; its ground goes to basis.
                                "opinion_licensed": proposal.operation == "opinion",
                                "opinion": json.loads(dict(proposal.plan.hypothesis).get(
                                    "basis") or "null") if proposal.operation == "opinion"
                                else None,
                                # A question about the unknown topic: never a claim of anyone.
                                "curiosity_licensed": proposal.operation == "curiosity",
                                "repair_licensed": proposal.operation == "repair",
                                "report_licensed": proposal.operation == "acknowledge_report",
                                "curiosity_target": proposal.premises[0].text
                                if proposal.operation == "curiosity" and proposal.premises
                                else None,
                                # Make-believe: never a hypothesis, a fact or a position.
                                "fantasy_licensed": is_fantasy,
                                "fantasy": is_fantasy,
                                "fantasy_continuation": dict(proposal.plan.hypothesis).get(
                                    "continues") == "play",
                                "epistemic_status": "fantasy"
                                if is_fantasy
                                else "hypothesis"
                                if proposal.operation == "supposition"
                                else "opinion" if proposal.operation == "opinion" else None,
                                "obligation_fit": sum(
                                    (
                                        (role, lemma)
                                        in {
                                            (d.role, d.argument.head.lemma)
                                            for d in message.claim.dependencies
                                        }
                                        for role, lemma in message.goal.known_roles
                                    )
                                )
                                / len(message.goal.known_roles)
                                if message.goal.known_roles and (not message.knowledge_limit)
                                else surface.confidence,
                                "meaning_activation": proposal_decision.activation
                                or proposal.confidence,
                                "meaning_preference": 0.5
                                + preference / (2 * (1 + abs(preference))),
                                "meaning_expected_reward": alternative.get("expected_reward", 0.0),
                                "choice_context": {
                                    **(proposal_decision.selection_context or {}),
                                    "enabled": self._response_deliberation_enabled,
                                },
                                "response_options": context.response_options,
                                "focused_request": context.prompt,
                                "choice_is_provisional": False,
                                "task_resolution": proposal.operation.startswith("task_"),
                                "meaning_intention_id": selection.intention_id
                                if selection is not None
                                else None,
                                **({"boredom": bored} if bored is not None else {}),
                                "intention_plan_key": proposal.plan.key,
                                "expected_outcome": alternative.get("expected_outcome"),
                            },
                        ),
                    )
                )
        if (
            active_task is not None
            and active_task.goal == "perform_utterance"
            and active_task.objective.required_content
        ):
            requested = active_task.objective.required_content.strip()
            text = requested[:1].upper() + requested[1:]
            if text and text[-1].isalnum():
                text += "."
            candidates.append(
                ResponseCandidate.create(
                    event_id=event.event_id,
                    source="dialogue_task",
                    renderer="requested_utterance_effect",
                    text=text,
                    speech_act="brief_statement",
                    source_confidence=active_task.confidence,
                    key_points=active_task.objective.focus_terms,
                    metadata=task_candidate_metadata(
                        active_task,
                        text,
                        {
                            "authored_reply_template": False,
                            "locally_composed": True,
                            "meaning_origin": "dialogue_task_effect",
                            "semantic_origin": "current_turn_request",
                            "semantic_scaffold": False,
                            "task_resolution": "perform_utterance",
                            "obligation_fit": float(
                                text.rstrip(".").casefold() == requested.rstrip(".").casefold()
                            ),
                            "semantic_frame_fit": active_task.confidence,
                            "focused_request": context.prompt,
                        },
                    ),
                )
            )
        if (self.sleep_cognition is not None and
                (context.claim_result is None or context.claim_result.status != "Known")):
            for candidate in await self.sleep_cognition.sleep_candidates(
                    event, context.prompt, context=context):
                if self._context_gate is not None and event.channel_id is not None:
                    gate = await self._context_gate.evaluate(
                        guild_id=event.guild_id, channel_id=event.channel_id,
                        candidate_text=candidate.text, source="arbitrated:semantic_reasoning",
                        now=event.utc_timestamp)
                    if not gate.allowed or gate.text != candidate.text:
                        candidate = replace(candidate, metadata={
                            **candidate.metadata, "hard_rejection": "sleep_context:" + gate.reason})
                candidates.append(candidate)
        await self._trace_step(
            event.event_id,
            stage="candidate_generation",
            component="evidence_planner",
            status="completed",
            summary="only evidence-derived meanings were realized",
            details={
                "candidate_count": len(candidates),
                "candidates": tuple(
                    {
                        "source": c.source,
                        "renderer": c.renderer,
                        "text": c.text,
                        "metadata": dict(c.metadata),
                    }
                    for c in candidates
                ),
                "composition": composer.records,
                "deliberation": {
                    **deliberation,
                    "reverse_rejections": sum(
                        bool(c["issues"]) for r in composer.records for c in r["surface_checks"]
                    ),
                },
                "confidence_dimensions": {
                    "parse": mode_decision.parse_confidence,
                    "intent": mode_decision.intent_confidence,
                    "fact": mode_decision.fact_confidence,
                    "expression": max(expression_confidences, default=0.0),
                },
            },
        )
        if self._context_gate is not None and event.channel_id is not None and candidates:
            relevance = await self._context_gate.measure_candidates(
                guild_id=event.guild_id,
                channel_id=event.channel_id,
                candidate_texts=tuple(c.text for c in candidates),
            )
            candidates = [
                replace(
                    c,
                    metadata={
                        **c.metadata,
                        "context_relevance": r.score,
                        "lexical_context_relevance": r.score,
                    },
                )
                for c, r in zip(candidates, relevance, strict=True)
            ]
        profile = (
            await self._arbitration_repository.profile(event.guild_id)
            if self._arbitration_repository is not None
            else ArbitrationProfile.baseline(event.guild_id)
        )
        structural_weights = (
            await self._arbitration_repository.structural_weights(event.guild_id)
            if self._arbitration_repository is not None
            else {}
        )
        from debilgi.config import thought_content_enabled

        select = (
            self._arbitrator.decide_answer_domain
            if context.response_options
            else self._arbitrator.decide
        )
        recent_replies = recent_reply_texts
        decision = select(
            **{"options": context.response_options} if context.response_options else {},
            event_id=event.event_id,
            prompt=context.prompt,
            focus_concepts=planning.objective.topics,
            candidates=tuple(candidates),
            recent_replies=recent_replies,
            profile=profile,
            structural_weights=structural_weights,
            **({"waive_floor": "repair" if repair is not None else "limit"}
               if thought_content_enabled() and not hostile and not (
                   irritation is not None and irritation.value > irritation.sensitivity)
               else {}),
        )
        if decision.winner is not None and selection is not None:
            key = decision.winner.metadata.get("intention_plan_key")
            final_proposal = next((p for p in proposals if p.plan.key == key), None)
            if final_proposal is not None:
                mode_decision = plan_decisions.get(final_proposal.plan.key, mode_decision)
                selection = await self._meaning_intents.finalize_selection(
                    selection,
                    final_proposal,
                    decision=mode_decision,
                    pattern_id=decision.winner.metadata.get("exchange_pattern_id"),
                    basis=delivered_basis(
                        final_proposal.premises, decision.winner.metadata
                    ) if (decision.winner.metadata.get("fantasy")
                          or decision.winner.metadata.get("opinion_licensed")) else None,
                )
            elif (
                decision.winner.metadata.get("sleep_hypothesis_id")
                and decision.winner.metadata.get("message_plan")
            ):
                # A night hypothesis said by day is the bot's own turn: it is remembered
                # as the delivered plan (never as a fact), so the next turn, the
                # repetition check and the contradiction check can see it.
                selection = await self._meaning_intents.adopt_hypothesis(
                    selection,
                    decision.winner.metadata["message_plan"],
                    decision.winner.metadata["sleep_hypothesis_id"],
                    decision.winner.metadata.get("message_plan_key"),
                    decision=mode_decision,
                    basis=delivered_basis(metadata=decision.winner.metadata),
                )
                decision = replace(
                    decision,
                    winner=replace(
                        decision.winner,
                        metadata={
                            **decision.winner.metadata,
                            "meaning_intention_id": selection.intention_id,
                        },
                    ),
                )
        await self._trace_step(
            event.event_id,
            stage="intent",
            component="response_mode",
            status="selected",
            summary="compatible response plan after bounded reader checks",
            details={"decision": mode_decision, "deliberation": deliberation},
        )
        if decision.winner is None:
            failure = (
                mode_decision.reason
                if mode_decision.act == "withhold"
                else "no_observed_social_form"
                if mode_decision.mode == "free_form"
                else "unrealizable_selected_plan"
            )
            if (may_repair and boredom is not None and self._semantic_assimilation is not None
                    and failure in {"no_observed_social_form", "unrealizable_selected_plan"}
                    and await self._repair_possible(event, context, failure, irritation,
                                                    boredom, partner_repair_allowed)):
                # An inner failure is not a decision to be silent: ask what was not understood.
                return _InnerFailure(failure)
            self._interlocutor_withhold_reasons[event.event_id] = failure
        decision = replace(decision, latency_ms=(perf_counter() - started) * 1000.0)
        await self._trace_step(
            event.event_id,
            stage="selection",
            component="response_arbitrator",
            status="selected" if decision.winner is not None else "failed",
            summary="local evidence selection completed without scripted recovery",
            details=_decision_trace_details(decision),
        )
        if self._arbitration_repository is not None:
            try:
                await self._arbitration_repository.save_decision(
                    event,
                    decision,
                    mode="active",
                    prompt=context.prompt,
                    legacy_text=None,
                    legacy_source=None,
                )
            except Exception:
                _failure_logger.exception(
                    "event=%s reason=arbitration_persistence_failed decision=%s",
                    event.event_id,
                    decision.decision_id,
                )
        self._log_arbitration(event, decision, mode="active", legacy_text=None, legacy_source=None)
        if remember:
            self._remember_arbitration(event.event_id, decision)
        if decision.winner is None:
            return ""
        if remember:
            self._remember_candidate_source(event.event_id, f"arbitrated:{decision.winner.source}")
        return decision.winner.text

    @staticmethod
    def _boredom_of(ledger, proposal, alternative, text) -> dict:
        """B of one candidate wording, from the journal: what is repeated and whether it is spent.

        A not-knowing of the same question said ``TIRED_REPEATS`` times lately is spent, and so
        is a report of the agent's own words that was already made (in any word order: it may
        not be said twice). The reading and its sources are kept in the candidate and in the trace.
        """
        from debilgi.experience.meaning_intents import (
            TIRED_REPEATS,
            claim_terms,
            goal_material,
            limit_scope,
        )

        limit = proposal.plan.knowledge_limit
        reading = ledger.reading(
            behavior_key=alternative.get("behavior_key"), operation=proposal.operation,
            terms=claim_terms(proposal.plan.claim), text=text,
            scope=limit_scope(limit.query) if limit is not None else None,
            material=goal_material(goal) if (goal := getattr(proposal.plan, "goal", None))
            else frozenset())
        report = proposal.operation == "verify_reported_position" and any(
            p.kind == "sent_turn" for p in proposal.premises)
        # A repair question said again in the same words to the same person is spent as a
        # limit is (4.40): the third time within the journal's memory is no longer asked.
        asked_again = proposal.operation == "repair" and reading.surface >= (
            1.0 - math.exp(-ledger.sensitivity * TIRED_REPEATS))
        spent = (limit is not None and reading.repeats >= TIRED_REPEATS) or (
            report and reading.same_plan > 0) or asked_again
        return {**reading.as_data(), "spent": spent}

    @staticmethod
    def _tired_of_limit(ledger, proposal, context) -> tuple:
        """Having said «не знаю» about this twice already, the agent says that it did (a report)."""
        from debilgi.experience.meaning_intents import TIRED_REPEATS, limit_scope
        from debilgi.language.communication import reader_goal
        from debilgi.language.dialogue_learning import delivered_report_plan

        limit = proposal.plan.knowledge_limit
        scope = limit_scope(limit.query)
        said = [r for r in ledger.rows if r.scope == scope and r.operation == proposal.operation]
        if len(said) < 1 or ledger.reading(
                operation=proposal.operation, scope=scope).repeats < TIRED_REPEATS:
            return ()
        row = said[0]
        return (delivered_report_plan(reader_goal(context.structure), {
            "receipt_id": row.receipt_id, "text": row.text}),)

    async def _opinion_plans(self, event: Event, context, planning) -> tuple:
        """A view of the agent's own on what is asked (4.45, flag only): its ground S and the
        kind of move it is (H, P) are read by `MeaningIntentRepository`; the plan by learning."""
        from debilgi.config import thought_content_enabled

        if (not thought_content_enabled() or self._meaning_intents is None
                or self._semantic_assimilation is None):
            return ()
        irritation = (await self._meaning_intents.irritation(event, context.prompt)
                      if context.turn_form is not None and context.turn_form.addressed else None)
        return await self._semantic_assimilation.dialogue_learning.opinion_plans(
            event, context, planning, sympathy=self._meaning_intents.sympathy,
            irritation=irritation)

    async def _recent_self_replies(self, event: Event) -> tuple[str, ...]:
        if event.channel_id is None:
            return ()
        context = await self._memory.channel_context(event.guild_id, event.channel_id)
        return tuple(
            content
            for item in context[-40:]
            if item.payload.get("author_is_bot") is True
            and item.target_id == event.actor_id
            and (item.utc_timestamp < event.utc_timestamp)
            and isinstance((content := item.payload.get("content")), str)
            and content.strip()
        )[-6:][::-1]

    async def _word_analyses(self, event: Event, prompt: str) -> tuple[WordAnalysis, ...]:
        if self._lexicon is None:
            return ()
        words = content_tokens(prompt)[:20]
        return tuple(
            await asyncio.gather(
                *(
                    self._lexicon.analyze_word(
                        word, guild_id=event.guild_id, channel_id=event.channel_id, context=prompt
                    )
                    for word in words
                )
            )
        )

    async def _withhold_non_productive_failure(self, event: Event, reason: str) -> None:
        await self._trace_step(
            event.event_id,
            stage="recovery",
            component="productive_only_policy",
            status="withheld",
            summary="no productively realized surface was available",
            details={"reason": reason, "fixed_transport_fallback_disabled": True},
        )
        return None

    def _remember_candidate_source(self, event_id: str, source: str) -> None:
        if not event_id:
            return
        self._candidate_sources[event_id] = source
        self._candidate_sources.move_to_end(event_id)
        while len(self._candidate_sources) > 256:
            self._candidate_sources.popitem(last=False)

    async def replay_sleep_episode(
        self, event: Event, *, variation: dict | None = None
    ) -> tuple[dict, ...]:
        """Cross local episode plans and observed forms, without running a reply.

        No consultant, knowledge fetch, send, intention selection, or fact write
        is reachable here. The same composer and reader check every variant.
        A revisit (``variation["visit"]`` > 0) widens the same windows over
        existing pieces — more neighbour forms, human donors and compatible
        acts — and skips plans already tried; examined plan keys are reported
        back in ``variation["examined"]``. Visit 0 is the original replay.
        """
        variation = variation if variation is not None else {}
        # Windows grow with each visit, up to a bounded unit (the 120 s budget).
        breadth = 1 + min(5, max(0, int(variation.get("visit", 0))))
        tried = set(variation.get("tried", ()))
        from debilgi.language.dialogue_learning import own_turn_plan
        from debilgi.language.productive_nlg import unresolved_answer_thought

        if self._semantic_assimilation is None or self._proposition_memory is None:
            return ()
        learning = self._semantic_assimilation.dialogue_learning
        prompt = self._clean_content(event)
        lesson = await self._semantic_lesson_context(event, prompt)
        context = ReplyContext(
            prompt=prompt, source_event_id=event.event_id, input_meaning=event.meaning,
            turn_form=event.turn_form, dependency_patterns=lesson.dependency_patterns,
            prior_bot_turn=await learning.prior_response_for(event),
            claim_result=await self._proposition_memory.retrieve_claims(
                event, prompt, reconcile=False),
        )
        context = resolve_evidence_context(context)
        forms = await learning.forms_for(event, context.prompt)
        # Adjacent human turns may lend a question move, while the current
        # scoped query supplies its arguments. They supply grammar, not facts.
        neighbours = await self._proposition_memory.recent_conversation(event)
        for neighbour in [n for n in neighbours if n.actor_id == event.actor_id
                          and n.event_id != event.event_id
                          and not n.payload.get("author_is_bot")][-2 * breadth:]:
            nearby = await learning.forms_for(event, self._clean_content(neighbour))
            forms += tuple(f for f in nearby if f.clause_kind == "question")
        planning = plan_from_evidence(context, ())
        human_proposals = await learning.sleep_answer_plans(event, context,
                                                            breadth=3 * breadth)
        repair = await learning.sleep_repair_plan(event, context, neighbours)
        if repair is not None:
            human_proposals = (repair, *human_proposals)
        proposals = []
        variable = await learning.variable_plan(event, context)
        if variable is not None:
            proposals.append(variable)
        delivered = (
            own_turn_plan(context, None)
            if context.prior_bot_turn and event.turn_form.kind == "repair"
            else None
        )
        if delivered is not None:
            proposals.append(delivered)
        limit = knowledge_limit_plan(context)
        if limit is not None:
            # Cross the scoped query with the already used compact epistemic
            # operator; the original daytime answer remains a competitor.
            compact = replace(limit.plan.knowledge_limit, question=None)
            claim = replace(unresolved_answer_thought(confidence=limit.confidence),
                            evidence=limit.plan.claim.evidence)
            proposals.append(replace(limit, plan=replace(
                limit.plan, claim=claim, knowledge_limit=compact)))
        decision = decide_response_mode(context, planning, (), observed_forms=forms,
                                        hypothesis_allowed=True)
        for option in compatible_response_decisions(
            decision, forms, hypothesis_allowed=True, limit=3 * breadth
        ):
            proposal = response_act_plan(context, option, forms)
            # Unconfirmed declarative world/user claims are never realized as
            # factual assertions. Experiments may ask, or admit an epistemic limit.
            if proposal and (proposal.plan.status in {"clarify", "express_limit"}
                             or proposal.plan.claim.predicate_class == "dialogue_act"):
                proposals.append(proposal)
        question_forms = tuple(f for f in forms if f.clause_kind == "question")
        if question_forms:
            exploratory = replace(decision, mode="free_form", act="question",
                                  target_meaning=None, allow_hook_question=True)
            for option in compatible_response_decisions(
                exploratory, question_forms, hypothesis_allowed=True, limit=3 * breadth
            ):
                proposal = response_act_plan(context, option, question_forms)
                if proposal is not None:
                    proposals.append(proposal)
        if tried:
            proposals = [p for p in proposals if p.plan.key not in tried]
            human_proposals = tuple(h for h in human_proposals if h[0].plan.key not in tried)
        proposals.sort(key=lambda p: p.plan.status == "express_limit")
        limits = [p for p in proposals if p.plan.status == "express_limit"]
        bounded = proposals[:3] + limits[-1:] if limits else proposals[:4]
        result = []
        old_trace = await learning._database.fetchone(
            "SELECT final_text FROM language_response_traces WHERE source_event_id=?",
            (event.event_id,))
        previous = old_trace[0] if old_trace else None
        from debilgi.cognition.continuous import ContinuousCognitionService

        if previous and ContinuousCognitionService.admits_not_knowing(previous):
            # An honest admission is not replaced by an unevidenced guess made
            # only of the question's own thought; it needs a human donor.
            human_proposals = tuple(h for h in human_proposals
                                    if not h[1].get("sleep_composed"))
        checked = []
        accepted_plans = set()
        examined = variation.setdefault("examined", [])
        for proposal, extra in (*human_proposals, *((p, {}) for p in bounded)):
            if proposal.plan.key not in examined:
                examined.append(proposal.plan.key)
            composer = ResponseComposer(context.structure)
            surfaces = await asyncio.to_thread(
                composer.compose, proposal.plan,
                (*proposal.patterns, *context.dependency_patterns),
                exchange_forms=forms, limit=3,
            )
            for surface in surfaces[:2]:
                metadata = {
                    **composer.metadata_for(surface.text, plan_key=proposal.plan.key),
                    "authored_reply_template": False, "locally_composed": True,
                    "meaning_origin": "derived", "semantic_origin": "derived",
                    "semantic_scaffold": False, "derivation": proposal.operation,
                    "reasoning_evidence": tuple(p.evidence_id for p in proposal.premises),
                    "intention_plan_key": proposal.plan.key,
                    "genuine_clarification": proposal.plan.status == "clarify",
                    "communicative_status": proposal.plan.status,
                    "obligation_fit": surface.confidence,
                    "semantic_frame_fit": surface.confidence,
                    "sleep_previous_text": previous,
                    # An honest limit leaves a required explanation unresolved,
                    # exactly as the daytime variable question does.
                    "unresolved_requirement": response_requirement(context.prompt)
                    if proposal.plan.status == "express_limit" else None,
                    **extra,
                }
                if extra.get("human_answer_match"):
                    # Relevance to the request and confidence in its personal
                    # truth are different measurements. Keep the latter low.
                    metadata["obligation_fit"] = extra["human_answer_match"]
                    metadata["semantic_frame_fit"] = extra["human_answer_match"]
                    metadata["answer_slot"] = extra.get("sleep_answer_slot", False)
                candidate = {"renderer": "sleep_replay:" + proposal.operation,
                               "text": surface.text, "speech_act": proposal.plan.speech_act
                               or "brief_statement",
                               "source_confidence": min(
                                   0.6, extra.get("human_answer_match", surface.confidence)),
                               "key_points": list(planning.objective.topics),
                               "metadata": metadata}
                from debilgi.cognition.continuous import ContinuousCognitionService
                rejection = ContinuousCognitionService.sleep_rejection(event, candidate)
                checked.append({"text": surface.text, "operation": proposal.operation,
                                "rejection": rejection})
                if rejection is None and proposal.plan.key not in accepted_plans:
                    result.append(candidate)
                    accepted_plans.add(proposal.plan.key)
        await self._trace_step(
            event.event_id, stage="sleep_replay", component="meaning_reader", status="completed",
            summary="candidate meanings checked against the original hole",
            details={"candidates": checked, "composition": composer.records if checked else []})
        return tuple(result[:3])

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
            and (
                " ".join(legacy_text.casefold().split())
                != " ".join(decision.winner.text.casefold().split())
            )
        )
        logger.info(
            "event=%s mode=%s decision=%s winner=%s source=%s candidates=%s eligible=%s "
            "fallback=%s reason=%s weights=%s latency_ms=%.2f legacy_source=%s shadow_disagreed=%s",
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
                "decision=%s candidate=%s source=%s renderer=%s score=%.4f eligible=%s "
                "features=%s rejected=%s text=%r",
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

    @staticmethod
    def _log_reply_failure(event: Event, reason: str, error: Exception | None = None) -> None:
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

    async def _knowledge_context(self, event: Event, prompt: str) -> KnowledgeGrounding:
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
            terms = (*terms, *await self._lexicon.semantic_terms(prompt, guild_id=event.guild_id))
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
            event.guild_id, prompt, channel_id=event.channel_id, user_id=event.actor_id
        )
        return tuple(item.summary for item in capsules)

    async def _curiosity_question(self, event: Event) -> str | None:
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
        self, events: Iterable[Event], limit: int, *, include_bot: bool = False
    ) -> tuple[str, ...]:
        texts = [
            text
            for item in events
            if include_bot or item.payload.get("author_is_bot") is not True
            if (text := self._clean_content(item))
        ]
        return tuple(texts[-limit:])

    def _recent_consultation_turns(self, events: Iterable[Event], limit: int) -> tuple[str, ...]:
        """Preserve speaker roles for deixis and cross-turn referents.

        A plain list of strings makes ``я`` and ``ты`` ambiguous to an
        external teacher.  The role prefix is metadata, not authored dialogue
        and never enters the local response generator.
        """
        turns = [
            f"{('agent' if item.payload.get('author_is_bot') is True else 'user')}: {text}"
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
