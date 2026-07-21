"""Observe consequences and slowly reinforce or weaken generated transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from debilgi.events.models import Event, EventType
from debilgi.experience.models import PreparedReply, UtteranceExperience
from debilgi.experience.repository import ExperienceRepository
from debilgi.reflection.engine import ReflectionEngine

_NEGATIVE_REACTIONS = {"👎", "🤮", "💩", "❌"}


@dataclass(frozen=True, slots=True)
class ConsolidationConfig:
    feedback_window: timedelta = timedelta(minutes=10)
    direct_reply_reward: float = 0.90
    reaction_reward: float = 0.65
    negative_reaction_reward: float = -0.65
    reaction_removed_reward: float = -0.35
    continuation_reward: float = 0.35
    ignored_reward: float = -0.18


class ExperienceConsolidator:
    def __init__(
        self,
        repository: ExperienceRepository,
        config: ConsolidationConfig | None = None,
        reflection: ReflectionEngine | None = None,
    ) -> None:
        self.repository = repository
        self.config = config or ConsolidationConfig()
        self._reflection = reflection

    async def register_sent(
        self,
        prompt: Event,
        *,
        response_message_id: int,
        reply: PreparedReply,
        created_at: datetime,
    ) -> None:
        if prompt.channel_id is None:
            return
        expected_reward = (
            await self._reflection.predict(prompt.guild_id, reply) if self._reflection else 0.0
        )
        await self.repository.register(
            guild_id=prompt.guild_id,
            channel_id=prompt.channel_id,
            user_id=prompt.actor_id,
            prompt_event_id=prompt.event_id,
            response_message_id=response_message_id,
            reply=reply,
            created_at=created_at,
            expected_reward=expected_reward,
        )

    async def observe(self, event: Event) -> None:
        await self._expire(event)
        if event.payload.get("author_is_bot") is True:
            return
        # Short reactions are resolved against an exact bot turn by the
        # pre-parser service.  Treating them as generic continuation here
        # would turn even "нет же" into positive feedback.
        if event.payload.get("suppress_full_parse") is True:
            return
        experience = None
        outcome = ""
        reward = 0.0
        if event.event_type in {EventType.REACTION_ADDED, EventType.REACTION_REMOVED}:
            message_id = _payload_int(event, "message_id")
            if message_id is not None:
                experience = await self.repository.pending_for_response_message(
                    event.guild_id, message_id
                )
                outcome = event.event_type.value.lower()
                if event.event_type is EventType.REACTION_ADDED:
                    emoji = event.payload.get("emoji")
                    reward = (
                        self.config.negative_reaction_reward
                        if isinstance(emoji, str) and emoji in _NEGATIVE_REACTIONS
                        else self.config.reaction_reward
                    )
                else:
                    reward = self.config.reaction_removed_reward
        elif event.event_type in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}:
            reply_to = _payload_int(event, "reply_to_message_id")
            if reply_to is not None:
                experience = await self.repository.pending_for_response_message(
                    event.guild_id, reply_to
                )
            if experience is not None:
                outcome = "direct_reply"
                reward = self.config.direct_reply_reward
            elif event.actor_id is not None and event.channel_id is not None:
                experience = await self.repository.latest_pending_for_user(
                    event.guild_id, event.channel_id, event.actor_id
                )
                outcome = "conversation_continued"
                reward = self.config.continuation_reward
        if experience is not None:
            await self._resolve(
                experience,
                outcome_type=outcome,
                reward=reward,
                resolved_at=event.utc_timestamp,
            )

    async def apply_short_act(
        self,
        event: Event,
        *,
        response_message_id: int,
        accepted: bool,
    ) -> bool:
        experience = await self.repository.pending_for_response_message(
            event.guild_id, response_message_id
        )
        if experience is None:
            return False
        await self._resolve(
            experience,
            outcome_type="short_act_confirmation" if accepted else "short_act_rejection",
            reward=0.80 if accepted else -0.85,
            resolved_at=event.utc_timestamp,
        )
        return True

    async def transition_adjustments(
        self, guild_id: int
    ) -> tuple[tuple[tuple[str, ...], float], ...]:
        return await self.repository.transition_adjustments(guild_id)

    async def _expire(self, event: Event) -> None:
        cutoff = event.utc_timestamp - self.config.feedback_window
        for experience in await self.repository.pending_before(cutoff):
            await self._resolve(
                experience,
                outcome_type="ignored",
                reward=self.config.ignored_reward,
                resolved_at=event.utc_timestamp,
                expired=True,
            )

    async def _resolve(
        self,
        experience: UtteranceExperience,
        *,
        outcome_type: str,
        reward: float,
        resolved_at: datetime,
        expired: bool = False,
    ) -> None:
        resolved = await self.repository.resolve(
            experience,
            outcome_type=outcome_type,
            reward=reward,
            resolved_at=resolved_at,
            expired=expired,
        )
        if resolved and self._reflection is not None:
            await self._reflection.reflect(
                experience,
                outcome_type=outcome_type,
                actual_reward=reward,
                created_at=resolved_at,
            )


def _payload_int(event: Event, key: str) -> int | None:
    value = event.payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None
