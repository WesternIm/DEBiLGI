"""Active conversational state, transient response plans and atomic consolidation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from debilgi.events.models import Event, EventType
from debilgi.language.content import content_lemmas
from debilgi.language.discourse import classify_utterance, topic_keys
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database

_MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
_QUESTION_WORDS = {"кто", "что", "где", "когда", "как", "почему", "зачем", "какой"}
_CAUSAL_MARKERS = {"потому", "поэтому", "чтобы", "из-за", "так", "значит"}
_CONCLUSION_MARKERS = {"решили", "вывод", "итого", "согласны", "договорились"}
_EXPLICIT_ENDINGS = {
    "пока",
    "до свидания",
    "на этом всё",
    "на этом все",
    "спасибо, всё",
    "спасибо, все",
    "закрываем тему",
}
@dataclass(frozen=True, slots=True)
class ConversationState:
    conversation_id: str
    guild_id: int
    channel_id: int
    started_at: datetime
    last_activity_at: datetime
    participants: tuple[int, ...]
    main_topic: str | None
    topic_history: tuple[str, ...]
    current_summary: str
    accepted_conclusions: tuple[str, ...]
    active_unknowns: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    recent_message_ids: tuple[int, ...]
    last_response_plan_id: str | None
    message_count: int


@dataclass(frozen=True, slots=True)
class ConversationEpisode:
    episode_id: str
    guild_id: int
    channel_id: int
    main_topic: str | None
    summary: str
    development_path: tuple[str, ...]
    conclusions: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    importance_score: float
    confidence: float


@dataclass(frozen=True, slots=True)
class ResponsePlan:
    plan_id: str
    source_event_id: str
    conversation_id: str | None
    author_id: int | None
    topic_ids: tuple[str, ...]
    message_intent: str
    response_goal: str
    key_points: tuple[str, ...]
    tone: str
    unknowns: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class PlanValidation:
    valid: bool
    covered_points: tuple[str, ...]
    missing_points: tuple[str, ...]
    topic_preserved: bool
    unsupported_certainty: bool
    reason: str


class ConversationRepository:
    def __init__(
        self,
        database: Database,
        *,
        idle_timeout: timedelta = timedelta(minutes=45),
        max_idle_timeout: timedelta = timedelta(minutes=60),
        summary_character_limit: int = 1800,
    ) -> None:
        if idle_timeout <= timedelta(0) or max_idle_timeout < idle_timeout:
            raise ValueError("conversation timeouts are invalid")
        self._database = database
        self.idle_timeout = idle_timeout
        self.max_idle_timeout = max_idle_timeout
        self.summary_character_limit = summary_character_limit

    async def observe(self, event: Event) -> None:
        if event.event_type not in _MESSAGE_TYPES or event.channel_id is None:
            return
        content = event.payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return
        await self.close_stale(event.utc_timestamp)
        await self._upsert_turn(event, _clean(content))
        if _clean(content).casefold().rstrip(".!? ") in _EXPLICIT_ENDINGS:
            await self.close_channel(event.guild_id, event.channel_id, now=event.utc_timestamp)

    async def active(self, guild_id: int, channel_id: int) -> ConversationState | None:
        row = await self._database.fetchone(
            """SELECT * FROM language_conversation_sessions
               WHERE guild_id = ? AND channel_id = ? AND status = 'active'""",
            (guild_id, channel_id),
        )
        return await self._state_from_row(row) if row else None

    async def contextual_summaries(
        self, guild_id: int, channel_id: int, topics: tuple[str, ...], limit: int = 3
    ) -> tuple[str, ...]:
        active = await self.active(guild_id, channel_id)
        summaries: list[str] = []
        if active and active.current_summary:
            summaries.append(active.current_summary)
        rows = await self._database.fetchall(
            """SELECT summary, main_topic, development_path_json
               FROM language_conversation_episodes
               WHERE guild_id = ? ORDER BY ended_at DESC LIMIT 100""",
            (guild_id,),
        )
        topic_set = set(topics)
        ranked: list[tuple[int, int, str]] = []
        for index, row in enumerate(rows):
            text = str(row["summary"])
            terms = set(topic_keys(f"{row['main_topic'] or ''} {text}"))
            ranked.append((len(topic_set & terms), -index, text))
        for overlap, _, text in sorted(ranked, reverse=True):
            if topic_set and overlap == 0:
                continue
            if text not in summaries:
                summaries.append(text)
            if len(summaries) >= max(1, limit):
                break
        return tuple(summaries)

    async def close_stale(self, now: datetime | None = None) -> int:
        moment = now or datetime.now(UTC)
        rows = await self._database.fetchall(
            "SELECT * FROM language_conversation_sessions WHERE status = 'active'"
        )
        closed = 0
        for row in rows:
            last = datetime.fromisoformat(str(row["last_activity_at"]))
            messages = int(row["message_count"])
            dynamic = self.idle_timeout + timedelta(minutes=min(15, messages // 8 * 3))
            if moment - last < min(self.max_idle_timeout, dynamic):
                continue
            await self._consolidate(str(row["conversation_id"]), moment)
            closed += 1
        return closed

    async def close_channel(
        self, guild_id: int, channel_id: int, *, now: datetime | None = None
    ) -> ConversationEpisode | None:
        state = await self.active(guild_id, channel_id)
        if state is None:
            return None
        return await self._consolidate(state.conversation_id, now or datetime.now(UTC))

    async def prune_episodes(self, guild_id: int, max_episodes: int = 5_000) -> int:
        """Bound episodic growth while keeping the most recent and important memories."""

        cursor = await self._database.execute(
            """DELETE FROM language_conversation_episodes
               WHERE guild_id = ? AND episode_id NOT IN (
                   SELECT episode_id FROM language_conversation_episodes
                   WHERE guild_id = ?
                   ORDER BY importance_score DESC, ended_at DESC LIMIT ?
               )""",
            (guild_id, guild_id, max(1, max_episodes)),
        )
        return cursor.rowcount

    async def _upsert_turn(self, event: Event, content: str) -> None:
        assert event.channel_id is not None
        observed = event.utc_timestamp.isoformat()
        topics = topic_keys(content)
        message_id = int(event.payload.get("message_id", 0))
        question = classify_utterance(content) == "question"
        async with self._database.transaction() as connection:
            row = await (
                await connection.execute(
                    """SELECT * FROM language_conversation_sessions
                       WHERE guild_id = ? AND channel_id = ?""",
                    (event.guild_id, event.channel_id),
                )
            ).fetchone()
            if row is None:
                restored = await self._restore_context(connection, event.guild_id, topics)
                conversation_id = f"conversation:{uuid4()}"
                summary = _summary_update(restored, content, topics, self.summary_character_limit)
                await connection.execute(
                    """INSERT INTO language_conversation_sessions (
                        conversation_id, guild_id, channel_id, started_at, last_activity_at,
                        main_topic, topic_history_json, current_summary,
                        unresolved_questions_json, recent_message_ids_json, message_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                    (
                        conversation_id,
                        event.guild_id,
                        event.channel_id,
                        observed,
                        observed,
                        topics[0] if topics else None,
                        json.dumps(topics, ensure_ascii=False),
                        summary,
                        json.dumps((content,) if question else (), ensure_ascii=False),
                        json.dumps((message_id,) if message_id else (), ensure_ascii=False),
                    ),
                )
                conclusions: tuple[str, ...] = ()
            else:
                conversation_id = str(row["conversation_id"])
                history = _bounded_unique(_json_tuple(row["topic_history_json"]), topics, 24)
                recent_ids = _bounded_unique(
                    tuple(int(item) for item in _json_tuple(row["recent_message_ids_json"])),
                    (message_id,) if message_id else (),
                    80,
                )
                unresolved = list(_json_tuple(row["unresolved_questions_json"]))
                if question:
                    unresolved.append(content)
                elif unresolved and _looks_like_answer(content):
                    unresolved.pop(0)
                conclusions = tuple(_json_tuple(row["accepted_conclusions_json"]))
                if set(words_from_sentence(content)) & _CONCLUSION_MARKERS:
                    conclusions = _bounded_unique(conclusions, (content,), 12)
                summary = _summary_update(
                    str(row["current_summary"]), content, topics, self.summary_character_limit
                )
                await connection.execute(
                    """UPDATE language_conversation_sessions SET
                        last_activity_at = ?, main_topic = COALESCE(?, main_topic),
                        topic_history_json = ?, current_summary = ?,
                        accepted_conclusions_json = ?, unresolved_questions_json = ?,
                        recent_message_ids_json = ?, message_count = message_count + 1
                       WHERE conversation_id = ?""",
                    (
                        observed,
                        topics[0] if topics else None,
                        json.dumps(history, ensure_ascii=False),
                        summary,
                        json.dumps(conclusions, ensure_ascii=False),
                        json.dumps(tuple(unresolved[-20:]), ensure_ascii=False),
                        json.dumps(recent_ids, ensure_ascii=False),
                        conversation_id,
                    ),
                )
            if event.actor_id is not None:
                await connection.execute(
                    """INSERT INTO language_conversation_participants (
                        conversation_id, participant_id, position_summary,
                        first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (conversation_id, participant_id) DO UPDATE SET
                        position_summary = excluded.position_summary,
                        last_seen_at = excluded.last_seen_at""",
                    (conversation_id, event.actor_id, content[:500], observed, observed),
                )

    async def _consolidate(
        self, conversation_id: str, ended_at: datetime
    ) -> ConversationEpisode | None:
        async with self._database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT * FROM language_conversation_sessions WHERE conversation_id = ?",
                    (conversation_id,),
                )
            ).fetchone()
            if row is None:
                return None
            participants_rows = await (
                await connection.execute(
                    """SELECT participant_id FROM language_conversation_participants
                       WHERE conversation_id = ? ORDER BY first_seen_at""",
                    (conversation_id,),
                )
            ).fetchall()
            topics = _json_tuple(row["topic_history_json"])
            conclusions = _json_tuple(row["accepted_conclusions_json"])
            unknowns = _json_tuple(row["active_unknowns_json"])
            questions = _json_tuple(row["unresolved_questions_json"])
            messages = tuple(int(item) for item in _json_tuple(row["recent_message_ids_json"]))
            summary = (
                str(row["current_summary"]).strip() or "Разговор завершён без устойчивого вывода."
            )
            episode_id = f"conversation-episode:{uuid4()}"
            importance = min(
                1.0,
                0.15
                + int(row["message_count"]) * 0.025
                + len(conclusions) * 0.08
                + len(unknowns) * 0.04,
            )
            confidence = min(0.92, 0.45 + int(row["message_count"]) * 0.02)
            await connection.execute(
                """INSERT INTO language_conversation_episodes (
                    episode_id, guild_id, channel_id, started_at, ended_at, participants_json,
                    main_topic, summary, development_path_json, conclusions_json,
                    new_concepts_json, unresolved_questions_json, important_message_ids_json,
                    importance_score, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    episode_id,
                    int(row["guild_id"]),
                    int(row["channel_id"]),
                    str(row["started_at"]),
                    ended_at.astimezone(UTC).isoformat(),
                    json.dumps(tuple(int(item[0]) for item in participants_rows)),
                    row["main_topic"],
                    summary,
                    json.dumps(topics, ensure_ascii=False),
                    json.dumps(conclusions, ensure_ascii=False),
                    json.dumps(unknowns, ensure_ascii=False),
                    json.dumps(questions, ensure_ascii=False),
                    json.dumps(messages),
                    importance,
                    confidence,
                ),
            )
            await connection.execute(
                "DELETE FROM language_conversation_sessions WHERE conversation_id = ?",
                (conversation_id,),
            )
        return ConversationEpisode(
            episode_id,
            int(row["guild_id"]),
            int(row["channel_id"]),
            str(row["main_topic"]) if row["main_topic"] else None,
            summary,
            topics,
            conclusions,
            questions,
            importance,
            confidence,
        )

    async def _restore_context(self, connection, guild_id: int, topics: tuple[str, ...]) -> str:
        rows = await (
            await connection.execute(
                """SELECT main_topic, summary FROM language_conversation_episodes
                   WHERE guild_id = ? ORDER BY ended_at DESC LIMIT 30""",
                (guild_id,),
            )
        ).fetchall()
        topic_set = set(topics)
        for row in rows:
            remembered = set(topic_keys(f"{row['main_topic'] or ''} {row['summary']}"))
            if topic_set & remembered:
                return f"Ранее по этой теме: {row['summary']}"
        return ""

    async def _state_from_row(self, row) -> ConversationState:
        people = await self._database.fetchall(
            """SELECT participant_id FROM language_conversation_participants
               WHERE conversation_id = ? ORDER BY first_seen_at""",
            (str(row["conversation_id"]),),
        )
        return ConversationState(
            conversation_id=str(row["conversation_id"]),
            guild_id=int(row["guild_id"]),
            channel_id=int(row["channel_id"]),
            started_at=datetime.fromisoformat(str(row["started_at"])),
            last_activity_at=datetime.fromisoformat(str(row["last_activity_at"])),
            participants=tuple(int(item["participant_id"]) for item in people),
            main_topic=str(row["main_topic"]) if row["main_topic"] else None,
            topic_history=_json_tuple(row["topic_history_json"]),
            current_summary=str(row["current_summary"]),
            accepted_conclusions=_json_tuple(row["accepted_conclusions_json"]),
            active_unknowns=_json_tuple(row["active_unknowns_json"]),
            unresolved_questions=_json_tuple(row["unresolved_questions_json"]),
            recent_message_ids=tuple(
                int(item) for item in _json_tuple(row["recent_message_ids_json"])
            ),
            last_response_plan_id=(
                str(row["last_response_plan_id"]) if row["last_response_plan_id"] else None
            ),
            message_count=int(row["message_count"]),
        )


class ResponsePlanRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def create(
        self,
        *,
        source_event_id: str,
        author_id: int | None,
        conversation_id: str | None,
        topic_ids: tuple[str, ...],
        message_intent: str,
        response_goal: str,
        key_points: tuple[str, ...],
        tone: str,
        unknowns: tuple[str, ...],
        confidence: float,
        now: datetime | None = None,
    ) -> ResponsePlan:
        plan = ResponsePlan(
            plan_id=f"response-plan:{uuid4()}",
            source_event_id=source_event_id,
            conversation_id=conversation_id,
            author_id=author_id,
            topic_ids=topic_ids[:12],
            message_intent=message_intent,
            response_goal=response_goal,
            key_points=key_points[:8],
            tone=tone,
            unknowns=unknowns[:12],
            confidence=max(0.0, min(1.0, confidence)),
        )
        created = (now or datetime.now(UTC)).isoformat()
        async with self._database.transaction() as connection:
            await connection.execute(
                """INSERT INTO language_response_plans (
                    plan_id, conversation_id, source_event_id, author_id, topic_ids_json,
                    message_intent, response_goal, key_points_json, tone, unknowns_json,
                    confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    plan.plan_id,
                    plan.conversation_id,
                    plan.source_event_id,
                    plan.author_id,
                    json.dumps(plan.topic_ids, ensure_ascii=False),
                    plan.message_intent,
                    plan.response_goal,
                    json.dumps(plan.key_points, ensure_ascii=False),
                    plan.tone,
                    json.dumps(plan.unknowns, ensure_ascii=False),
                    plan.confidence,
                    created,
                ),
            )
            if conversation_id:
                await connection.execute(
                    """UPDATE language_conversation_sessions SET last_response_plan_id = ?
                       WHERE conversation_id = ?""",
                    (plan.plan_id, conversation_id),
                )
        return plan

    async def complete(self, plan: ResponsePlan, validation: PlanValidation) -> None:
        await self._database.execute(
            """UPDATE language_response_plans SET status = ?, validation_json = ?,
               completed_at = CURRENT_TIMESTAMP WHERE plan_id = ?""",
            (
                "used" if validation.valid else "failed",
                json.dumps(
                    {
                        "covered_points": validation.covered_points,
                        "missing_points": validation.missing_points,
                        "topic_preserved": validation.topic_preserved,
                        "unsupported_certainty": validation.unsupported_certainty,
                        "reason": validation.reason,
                    },
                    ensure_ascii=False,
                ),
                plan.plan_id,
            ),
        )

    async def reframe(
        self,
        plan: ResponsePlan,
        *,
        message_intent: str,
        response_goal: str,
        key_points: tuple[str, ...],
        confidence: float,
    ) -> ResponsePlan:
        """Persist a stronger semantic reading discovered after plan creation."""

        reframed = replace(
            plan,
            message_intent=message_intent,
            response_goal=response_goal,
            key_points=key_points[:8],
            confidence=max(plan.confidence, max(0.0, min(1.0, confidence))),
        )
        await self._database.execute(
            """UPDATE language_response_plans
               SET message_intent = ?, response_goal = ?, key_points_json = ?,
                   confidence = ?
               WHERE plan_id = ?""",
            (
                reframed.message_intent,
                reframed.response_goal,
                json.dumps(reframed.key_points, ensure_ascii=False),
                reframed.confidence,
                reframed.plan_id,
            ),
        )
        return reframed

    async def cleanup(self, before: datetime) -> int:
        cursor = await self._database.execute(
            "DELETE FROM language_response_plans WHERE status != 'pending' AND created_at < ?",
            (before.astimezone(UTC).isoformat(),),
        )
        return cursor.rowcount


def validate_response(plan: ResponsePlan, text: str) -> PlanValidation:
    response_terms = _content_terms(text)
    covered: list[str] = []
    missing: list[str] = []
    for point in plan.key_points:
        point_terms = _content_terms(point)
        coverage = len(point_terms & response_terms) / max(1, len(point_terms))
        (covered if coverage >= 0.35 else missing).append(point)
    topic_preserved = (
        not plan.topic_ids
        or bool(set(plan.topic_ids) & response_terms)
        or bool(plan.key_points and not missing)
    )
    unsupported = plan.confidence < 0.45 and not plan.unknowns and _sounds_certain(text)
    valid = bool(text.strip()) and not missing and topic_preserved and not unsupported
    reasons = []
    if missing:
        reasons.append("missing key points")
    if not topic_preserved:
        reasons.append("topic drift")
    if unsupported:
        reasons.append("unsupported certainty")
    if not text.strip():
        reasons.append("empty response")
    return PlanValidation(
        valid,
        tuple(covered),
        tuple(missing),
        topic_preserved,
        unsupported,
        ", ".join(reasons) or "plan covered",
    )


def _summary_update(previous: str, content: str, topics: tuple[str, ...], limit: int) -> str:
    sentence = content.strip()[:500]
    words = set(words_from_sentence(sentence))
    if not previous:
        return f"Разговор начался с темы «{topics[0]}»: {sentence}"[:limit] if topics else sentence
    connector = " Поэтому " if words & _CAUSAL_MARKERS else " Затем "
    combined = f"{previous.rstrip('.')} .{connector}{sentence}".replace(" .", ".")
    if len(combined) <= limit:
        return combined
    return f"…{combined[-(limit - 1) :]}"


def _clean(text: str) -> str:
    return " ".join(text.split())[:1500]


def _json_tuple(value) -> tuple[str, ...]:
    return tuple(str(item) for item in json.loads(str(value)))


def _bounded_unique(left: tuple, right: tuple, limit: int) -> tuple:
    values = list(left)
    for item in right:
        if item not in values:
            values.append(item)
    return tuple(values[-limit:])


def words_from_sentence(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in tokenize(text) if token[:1].isalnum())


def _content_terms(text: str) -> set[str]:
    return set(content_lemmas(text))


def _looks_like_answer(text: str) -> bool:
    tokens = set(words_from_sentence(text))
    return not bool(tokens & _QUESTION_WORDS) and not text.rstrip().endswith("?")


def _sounds_certain(text: str) -> bool:
    normalized = text.casefold()
    hedges = ("не знаю", "не уверен", "кажется", "возможно", "может быть")
    return not any(hedge in normalized for hedge in hedges)
