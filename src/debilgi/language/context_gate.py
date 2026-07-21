"""One working-focus state and relevance gate for every conversational send path."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.events.models import Event, EventType
from debilgi.language.content import content_lemmas
from debilgi.language.lexical_matching import same_lexical_family

logger = logging.getLogger("debilgi.context_gate")

_MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
_BOUND_SOURCES = frozenset({"fallback", "short_act", "contextual_learning"})


@dataclass(frozen=True, slots=True)
class WorkingTurn:
    source_id: str
    message_id: int | None
    text: str
    concepts: tuple[str, ...]
    actor_id: int | None
    target_user_id: int | None
    is_bot: bool
    addressed_to_bot: bool
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class WorkingFocus:
    guild_id: int
    channel_id: int
    turns: tuple[WorkingTurn, ...]
    concepts: tuple[str, ...]
    pending_concepts: tuple[str, ...]
    last_actor_id: int | None
    last_addressed_to_bot: bool


@dataclass(frozen=True, slots=True)
class GateDecision:
    action: str
    text: str | None
    source: str
    candidate_concepts: tuple[str, ...]
    focus_concepts: tuple[str, ...]
    matched_concepts: tuple[str, ...]
    reason: str

    @property
    def allowed(self) -> bool:
        return self.action in {"passed", "replaced"} and bool(self.text)


@dataclass(frozen=True, slots=True)
class ContextRelevance:
    score: float
    candidate_concepts: tuple[str, ...]
    focus_concepts: tuple[str, ...]
    matched_concepts: tuple[str, ...]
    related_concepts: tuple[str, ...]


class ContextGateDeferred(RuntimeError):
    """Raised by a transport when a background initiative should stay queued."""


class WorkingMemoryContextGate:
    def __init__(
        self,
        dataset: SelfDatasetRepository,
        *,
        window_size: int = 5,
        minimum_matches: int = 1,
        initiative_idle: timedelta = timedelta(minutes=5),
    ) -> None:
        if window_size <= 0 or minimum_matches <= 0 or initiative_idle <= timedelta(0):
            raise ValueError("context gate limits must be positive")
        self._dataset = dataset
        self.window_size = window_size
        self.minimum_matches = minimum_matches
        self.initiative_idle = initiative_idle
        self._lock = asyncio.Lock()
        self._channels: dict[tuple[int, int], deque[WorkingTurn]] = {}
        self._pending: dict[tuple[int, int, int], str] = {}

    async def observe(self, event: Event) -> None:
        if event.event_type not in _MESSAGE_TYPES or event.channel_id is None:
            return
        content = event.payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return
        turn = WorkingTurn(
            source_id=event.event_id,
            message_id=_payload_int(event, "message_id"),
            text=content.strip(),
            concepts=content_lemmas(content),
            actor_id=event.actor_id,
            target_user_id=_payload_int(event, "reply_to_author_id"),
            is_bot=event.payload.get("author_is_bot") is True,
            addressed_to_bot=event.payload.get("bot_mentioned") is True,
            observed_at=event.utc_timestamp,
        )
        await self._remember(event.guild_id, event.channel_id, turn)

    async def commit_outgoing(
        self,
        *,
        guild_id: int,
        channel_id: int,
        message_id: int,
        text: str,
        bot_user_id: int | None,
        target_user_id: int | None,
        observed_at: datetime,
    ) -> None:
        await self._remember(
            guild_id,
            channel_id,
            WorkingTurn(
                source_id=f"context-gate:outgoing:{message_id}",
                message_id=message_id,
                text=text.strip(),
                concepts=content_lemmas(text),
                actor_id=bot_user_id,
                target_user_id=target_user_id,
                is_bot=True,
                addressed_to_bot=False,
                observed_at=_aware(observed_at),
            ),
        )

    async def set_pending_question(
        self, guild_id: int, channel_id: int, user_id: int, concept: str
    ) -> None:
        async with self._lock:
            self._pending[(guild_id, channel_id, user_id)] = concept

    async def clear_pending_question(
        self,
        guild_id: int,
        channel_id: int,
        user_id: int,
        *,
        concept: str | None = None,
    ) -> None:
        key = (guild_id, channel_id, user_id)
        async with self._lock:
            current = self._pending.get(key)
            if current is not None and (concept is None or current == concept):
                self._pending.pop(key, None)

    async def pending_question(
        self, guild_id: int, channel_id: int, user_id: int
    ) -> str | None:
        async with self._lock:
            return self._pending.get((guild_id, channel_id, user_id))

    async def snapshot(self, guild_id: int, channel_id: int) -> WorkingFocus:
        async with self._lock:
            turns = tuple(self._channels.get((guild_id, channel_id), ()))
            pending = tuple(
                dict.fromkeys(
                    concept
                    for (item_guild, item_channel, _), concept in self._pending.items()
                    if (item_guild, item_channel) == (guild_id, channel_id)
                )
            )
        concepts = tuple(
            dict.fromkeys(
                concept for turn in turns for concept in turn.concepts
            )
        )
        combined = tuple(dict.fromkeys((*concepts, *pending)))
        last = turns[-1] if turns else None
        return WorkingFocus(
            guild_id=guild_id,
            channel_id=channel_id,
            turns=turns,
            concepts=combined,
            pending_concepts=pending,
            last_actor_id=last.actor_id if last else None,
            last_addressed_to_bot=last.addressed_to_bot if last else False,
        )

    async def last_bot_turn(
        self, guild_id: int, channel_id: int, *, user_id: int | None = None
    ) -> WorkingTurn | None:
        focus = await self.snapshot(guild_id, channel_id)
        targeted = tuple(
            turn
            for turn in reversed(focus.turns)
            if turn.is_bot and (user_id is None or turn.target_user_id in {None, user_id})
        )
        return targeted[0] if targeted else None

    async def evaluate(
        self,
        *,
        guild_id: int,
        channel_id: int,
        candidate_text: str,
        source: str,
        now: datetime | None = None,
    ) -> GateDecision:
        focus = await self.snapshot(guild_id, channel_id)
        candidate = content_lemmas(candidate_text)
        focus_concepts = focus.concepts
        matched = tuple(
            concept
            for concept in candidate
            if any(_same_family(concept, active) for active in focus_concepts)
        )
        if len(matched) < self.minimum_matches and candidate and focus_concepts:
            related = await self._related_focus(guild_id, focus_concepts)
            matched = tuple(
                dict.fromkeys(
                    (*matched, *(concept for concept in candidate if concept in related))
                )
            )

        if source in _BOUND_SOURCES or source.startswith("arbitrated"):
            return self._decision(
                "passed", candidate_text, source, candidate, focus_concepts, matched,
                "pre_scored_by_arbitrator"
                if source.startswith("arbitrated")
                else "bound_to_current_turn",
            )
        if not focus.turns or not focus_concepts:
            return self._decision(
                "passed", candidate_text, source, candidate, focus_concepts, matched,
                "no_active_focus",
            )
        if len(matched) >= self.minimum_matches:
            return self._decision(
                "passed", candidate_text, source, candidate, focus_concepts, matched,
                "concept_match",
            )

        current_time = _aware(now or datetime.now(UTC))
        last_observed = max(turn.observed_at for turn in focus.turns)
        if source == "background":
            if current_time - last_observed >= self.initiative_idle:
                return self._decision(
                    "passed", candidate_text, source, candidate, focus_concepts, matched,
                    "channel_idle",
                )
            return self._decision(
                "deferred", None, source, candidate, focus_concepts, matched,
                "background_unrelated_to_active_focus",
            )
        if source.startswith("initiative"):
            return self._decision(
                "deferred", None, source, candidate, focus_concepts, matched,
                "initiative_unrelated_to_active_focus",
            )
        topic = _latest_human_topic(focus) or focus_concepts[-1]
        replacement = (
            f"Не могу надёжно связать ответ с текущей темой «{topic}». "
            "Надо разобраться."
        )
        return self._decision(
            "replaced", replacement, source, candidate, focus_concepts, matched,
            "reactive_candidate_unrelated",
        )

    async def measure_relevance(
        self,
        *,
        guild_id: int,
        channel_id: int,
        candidate_text: str,
    ) -> ContextRelevance:
        """Return graded relevance for candidate arbitration without replacing its text."""

        return (
            await self.measure_candidates(
                guild_id=guild_id,
                channel_id=channel_id,
                candidate_texts=(candidate_text,),
            )
        )[0]

    async def measure_candidates(
        self,
        *,
        guild_id: int,
        channel_id: int,
        candidate_texts: tuple[str, ...],
    ) -> tuple[ContextRelevance, ...]:
        """Measure a candidate pool against one shared focus and graph lookup."""

        focus = await self.snapshot(guild_id, channel_id)
        related_set = (
            await self._related_focus(guild_id, focus.concepts) if focus.concepts else set()
        )
        return tuple(
            self._measure_text(text, focus=focus, related_set=related_set)
            for text in candidate_texts
        )

    @staticmethod
    def _measure_text(
        candidate_text: str,
        *,
        focus: WorkingFocus,
        related_set: set[str],
    ) -> ContextRelevance:
        candidate = content_lemmas(candidate_text)
        direct = tuple(
            concept
            for concept in candidate
            if any(_same_family(concept, active) for active in focus.concepts)
        )
        related = tuple(
            concept for concept in candidate if concept not in direct and concept in related_set
        )
        if not focus.concepts:
            score = 0.75
        elif not candidate:
            score = 0.0
        else:
            direct_ratio = len(direct) / len(candidate)
            related_ratio = len(related) / len(candidate)
            score = min(1.0, direct_ratio + related_ratio * 0.65)
        return ContextRelevance(
            score=score,
            candidate_concepts=candidate,
            focus_concepts=focus.concepts,
            matched_concepts=direct,
            related_concepts=related,
        )

    async def _remember(
        self, guild_id: int, channel_id: int, turn: WorkingTurn
    ) -> None:
        key = (guild_id, channel_id)
        async with self._lock:
            window = self._channels.setdefault(key, deque(maxlen=self.window_size))
            if turn.message_id is not None:
                retained = [item for item in window if item.message_id != turn.message_id]
                if len(retained) != len(window):
                    window.clear()
                    window.extend(retained)
            window.append(turn)

    async def _related_focus(
        self, guild_id: int, focus_concepts: tuple[str, ...]
    ) -> set[str]:
        related: set[str] = set()
        for concept in focus_concepts[-12:]:
            try:
                related.update(await self._dataset.related_concepts(guild_id, concept, limit=16))
            except Exception:
                logger.exception(
                    "related-concept lookup failed guild=%s concept=%r",
                    guild_id,
                    concept,
                )
        return related

    @staticmethod
    def _decision(
        action: str,
        text: str | None,
        source: str,
        candidate: tuple[str, ...],
        focus: tuple[str, ...],
        matched: tuple[str, ...],
        reason: str,
    ) -> GateDecision:
        decision = GateDecision(action, text, source, candidate, focus, matched, reason)
        logger.info(
            "source=%s candidate_concepts=%s focus_concepts=%s matched=%s result=%s reason=%s",
            source,
            list(candidate),
            list(focus),
            list(matched),
            action,
            reason,
        )
        return decision


def _latest_human_topic(focus: WorkingFocus) -> str | None:
    for turn in reversed(focus.turns):
        if not turn.is_bot and turn.concepts:
            return turn.concepts[-1]
    return None


def _same_family(left: str, right: str) -> bool:
    return same_lexical_family(
        left,
        right,
        minimum_prefix=4,
        minimum_similarity=0.62,
    )


def _payload_int(event: Event, key: str) -> int | None:
    value = event.payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
