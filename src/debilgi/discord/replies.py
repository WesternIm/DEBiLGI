"""Narrow opt-in bridge from a stored intent to a harmless text reply."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from datetime import datetime

from debilgi.autonomy.policy import InitiativePolicy
from debilgi.cognition.core_repository import CognitiveRepository
from debilgi.cognition.models import ActionIntent, ActionType
from debilgi.events.models import Event
from debilgi.experience.consolidation import ExperienceConsolidator
from debilgi.experience.models import PreparedReply
from debilgi.knowledge.service import KnowledgeAcquisitionService
from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.discourse import DiscourseContext, DiscourseMemoryRepository
from debilgi.language.planning import (
    PhraseGenerator,
    ReplyContext,
    StructuredThought,
    thought_from_intent,
    utterance_plan,
)
from debilgi.language.semantics import SemanticMemoryRepository
from debilgi.language.tokenizer import tokenize
from debilgi.literacy.engine import LiteracyEngine
from debilgi.memory.short_term import ShortTermMemory
from debilgi.personality.expression import PersonalityExpressionEngine
from debilgi.reflection.engine import ReflectionEngine
from debilgi.social.models import SocialReaction
from debilgi.social.repository import SocialReactionRepository

_USER_MENTION_RE = re.compile(r"<@!?\d+>")
_WHITESPACE_RE = re.compile(r"\s+")


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
        knowledge_service: KnowledgeAcquisitionService | None = None,
        context_character_limit: int = 600,
        enabled: bool = False,
    ) -> None:
        if context_character_limit <= 0:
            raise ValueError("context_character_limit must be positive")
        self._repository = repository
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
        self._knowledge_service = knowledge_service
        self._context_character_limit = context_character_limit
        self.enabled = enabled

    async def response_for(self, event: Event) -> str | None:
        prepared = await self.prepare_reply(event)
        return prepared.text if prepared else None

    async def prepare_reply(self, event: Event) -> PreparedReply | None:
        if (
            not self.enabled
            or event.payload.get("bot_mentioned") is not True
            or event.payload.get("author_is_bot") is True
        ):
            return None
        intent = await self._repository.intent_for_event(event.event_id)
        if intent is None:
            return None
        context = await self._reply_context(event, intent)
        plan = utterance_plan(thought_from_intent(intent), context=context)
        if plan is None:
            return None
        text = self._generator.generate(plan)
        if not text or context.personality is None:
            return None
        return PreparedReply(
            text=text,
            intent_id=intent.intent_id,
            tokens=tokenize(text),
            genome_signature=context.personality.genome_signature,
            candidate_count=context.personality.candidate_count,
        )

    async def prepare_initiative(self, event: Event) -> PreparedReply | None:
        intent = await self._repository.intent_for_event(event.event_id)
        if intent is None:
            return None
        decision = await self._initiative_policy.evaluate(event, intent)
        if not decision.should_initiate or decision.goal_id is None:
            return None
        context = await self._reply_context(event, intent)
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
        text = self._generator.generate(plan)
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
        )

    async def record_sent(
        self,
        prompt: Event,
        *,
        response_message_id: int,
        reply: PreparedReply,
        created_at: datetime,
    ) -> None:
        await self._experience_consolidator.register_sent(
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
        if event.actor_id is None or event.channel_id is None:
            return ReplyContext(
                prompt=prompt,
                seed=event.event_id,
                personality=self._expression_engine.express(intent),
            )
        (
            user_events,
            channel_events,
            archived_texts,
            reply_heads,
            memory_ngrams,
            experience_transitions,
            discourse,
            semantic_facts,
            knowledge_facts,
            reaction,
            reflection,
            literacy_facts,
        ) = await asyncio.gather(
            self._memory.user_context(event.guild_id, event.actor_id),
            self._memory.channel_context(event.guild_id, event.channel_id),
            self._archive_repository.recent_texts(event.guild_id, event.channel_id),
            self._archive_repository.matching_reply_heads(event.guild_id, tokenize(prompt)),
            self._archive_repository.common_ngrams(event.guild_id),
            self._experience_consolidator.transition_adjustments(event.guild_id),
            self._discourse_context(event, prompt),
            self._semantic_context(event, prompt),
            self._knowledge_context(event, prompt),
            self._reaction_context(event),
            self._reflection_context(event),
            self._literacy_context(prompt),
        )
        previous_user_events = tuple(
            item for item in user_events if item.event_id != event.event_id
        )
        prior_user_messages = self._recent_texts(previous_user_events, limit=30)
        recent_channel_messages = self._recent_texts(
            (
                item
                for item in channel_events
                if item.event_id != event.event_id and item.actor_id != event.actor_id
            ),
            limit=50,
        )
        return ReplyContext(
            prompt=prompt,
            memory_messages=(
                *archived_texts,
                *recent_channel_messages,
                *prior_user_messages,
                *discourse.messages,
                *discourse.open_questions,
            ),
            memory_ngrams=memory_ngrams,
            experience_transitions=experience_transitions,
            reply_heads=(*discourse.reply_heads, *reply_heads),
            knowledge_facts=(
                *(fact.text for fact in literacy_facts),
                *semantic_facts,
                *knowledge_facts,
            ),
            seed=event.event_id,
            personality=self._expression_engine.express(intent),
            reaction=reaction,
            reflection=reflection,
            exact_answer=next((fact.text for fact in literacy_facts if fact.exact), None),
        )

    async def _knowledge_context(self, event: Event, prompt: str) -> tuple[str, ...]:
        if self._knowledge_service is None:
            return ()
        await self._knowledge_service.wait_for_event(event.event_id)
        return await self._knowledge_service.context_for_text(prompt)

    async def _semantic_context(self, event: Event, prompt: str) -> tuple[str, ...]:
        if self._semantic_memory is None:
            return ()
        return await self._semantic_memory.context(event.guild_id, tokenize(prompt))

    async def _discourse_context(self, event: Event, prompt: str) -> DiscourseContext:
        if self._discourse_memory is None:
            return DiscourseContext((), (), (), None, 0)
        return await self._discourse_memory.context_for(event, prompt)

    async def _reaction_context(self, event: Event) -> SocialReaction | None:
        if self._social_reactions is None:
            return None
        return await self._social_reactions.reaction_for_event(event.event_id)

    async def _reflection_context(self, event: Event):
        if self._reflection is None:
            return None
        return await self._reflection.guidance(event.guild_id)

    async def _literacy_context(self, prompt: str):
        return self._literacy.context_for_text(prompt) if self._literacy else ()

    def _recent_texts(self, events: Iterable[Event], limit: int) -> tuple[str, ...]:
        texts = [
            text
            for item in events
            if item.payload.get("author_is_bot") is not True
            if (text := self._clean_content(item))
        ]
        return tuple(texts[-limit:])

    def _clean_content(self, event: Event) -> str:
        content = event.payload.get("content")
        if not isinstance(content, str):
            return ""
        without_mentions = _USER_MENTION_RE.sub("", content)
        return _WHITESPACE_RE.sub(" ", without_mentions).strip()[: self._context_character_limit]
