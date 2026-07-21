"""Long-term conversational links: turns, addressees, topics and open questions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import log1p
from typing import Any

import aiosqlite

from debilgi.events.models import Event, EventType
from debilgi.language.addressing import DEFAULT_ADDRESSING
from debilgi.language.content import content_lemmas
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database

_MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
_QUESTION_WORDS = frozenset(
    {"кто", "что", "где", "куда", "откуда", "когда", "зачем", "почему", "как", "какой"}
)
_LEADING_FILLERS = frozenset({"а", "да", "давай", "ладно", "ну", "так", "эй"})
_NON_COMMAND_DETERMINERS = frozenset(
    {"мой", "моя", "мое", "моё", "мои", "твой", "твоя", "свой", "своя"}
)
_MENTION_RE = re.compile(r"<@!?\d+>")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class DiscourseContext:
    messages: tuple[str, ...]
    open_questions: tuple[str, ...]
    reply_heads: tuple[tuple[str, ...], ...]
    addressed_to_id: int | None
    reply_depth: int


@dataclass(frozen=True, slots=True)
class DiscourseStats:
    turns: int
    open_questions: int
    answered_questions: int


def classify_utterance(content: str) -> str:
    tokens = DEFAULT_ADDRESSING.analyze(content).content_tokens
    words = tuple(
        token.casefold().replace("ё", "е")
        for token in tokens
        if any(character.isalnum() for character in token)
    )
    first = words[0].replace("ё", "е") if words else ""
    if content.rstrip().endswith("?") or first in _QUESTION_WORDS:
        return "question"
    command_index = 0
    while command_index < len(words) and words[command_index] in _LEADING_FILLERS:
        command_index += 1
    command = words[command_index] if command_index < len(words) else ""
    if (
        command
        and command not in _NON_COMMAND_DETERMINERS
        and DEFAULT_ADDRESSING.is_imperative(command)
    ):
        return "command"
    return "assertion"


def topic_keys(content: str) -> tuple[str, ...]:
    addressed = DEFAULT_ADDRESSING.analyze(content)
    return content_lemmas(addressed.content_tokens)[:24]


async def store_discourse_turn(
    connection: aiosqlite.Connection,
    *,
    source_id: str,
    guild_id: int,
    channel_id: int,
    message_id: int,
    parent_message_id: int | None,
    actor_id: int | None,
    addressed_to_id: int | None,
    content: str,
    observed_at: str,
) -> bool:
    if addressed_to_id is None and parent_message_id is not None:
        parent = await (
            await connection.execute(
                """
                SELECT actor_id FROM language_discourse_turns
                WHERE guild_id = ? AND channel_id = ? AND message_id = ?
                """,
                (guild_id, channel_id, parent_message_id),
            )
        ).fetchone()
        addressed_to_id = int(parent[0]) if parent is not None and parent[0] is not None else None
    utterance_type = classify_utterance(content)
    cursor = await connection.execute(
        """
        INSERT OR IGNORE INTO language_discourse_turns (
            source_id, guild_id, channel_id, message_id, parent_message_id,
            actor_id, addressed_to_id, utterance_type, question_status,
            content, topics_json, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source_id,
            guild_id,
            channel_id,
            message_id,
            parent_message_id,
            actor_id,
            addressed_to_id,
            utterance_type,
            "open" if utterance_type == "question" else "not_question",
            content,
            json.dumps(topic_keys(content), ensure_ascii=False, separators=(",", ":")),
            observed_at,
        ),
    )
    if parent_message_id is not None:
        await connection.execute(
            """
            UPDATE language_discourse_turns
            SET question_status = 'answered', answered_by_source_id = ?
            WHERE guild_id = ? AND channel_id = ? AND message_id = ?
              AND question_status = 'open'
            """,
            (source_id, guild_id, channel_id, parent_message_id),
        )
    return cursor.rowcount == 1


class DiscourseMemoryRepository:
    def __init__(
        self,
        database: Database,
        *,
        max_turns_per_guild: int = 20_000,
        context_turn_limit: int = 80,
        content_character_limit: int = 600,
    ) -> None:
        if min(max_turns_per_guild, context_turn_limit, content_character_limit) <= 0:
            raise ValueError("discourse memory limits must be positive")
        self._database = database
        self.max_turns_per_guild = max_turns_per_guild
        self.context_turn_limit = context_turn_limit
        self.content_character_limit = content_character_limit

    async def observe(self, event: Event) -> None:
        if event.event_type not in _MESSAGE_TYPES or event.channel_id is None:
            return
        message_id = self._payload_int(event, "message_id")
        content = event.payload.get("content")
        if message_id is None or not isinstance(content, str):
            return
        cleaned = self._clean(content)
        if not cleaned:
            return
        parent_message_id = self._payload_int(event, "reply_to_message_id")
        addressed_to = self._addressed_to(event)

        async with self._database.transaction() as connection:
            inserted = await store_discourse_turn(
                connection,
                source_id=event.event_id,
                guild_id=event.guild_id,
                channel_id=event.channel_id,
                message_id=message_id,
                parent_message_id=parent_message_id,
                actor_id=event.actor_id,
                addressed_to_id=addressed_to,
                content=cleaned,
                observed_at=event.utc_timestamp.isoformat(),
            )
            if inserted and event.payload.get("suppress_full_parse") is True:
                # Keep the turn as conversational context, but never let a
                # bare reaction such as "ты понял" become a graph topic.
                await connection.execute(
                    """UPDATE language_discourse_turns
                       SET topics_json = '[]', utterance_type = 'assertion',
                           question_status = 'not_question'
                       WHERE source_id = ?""",
                    (event.event_id,),
                )
            if inserted and parent_message_id is None and event.actor_id is not None:
                await connection.execute(
                    """
                    UPDATE language_discourse_turns
                    SET question_status = 'answered', answered_by_source_id = ?
                    WHERE source_id = (
                        SELECT source_id FROM language_discourse_turns
                        WHERE guild_id = ? AND channel_id = ?
                          AND question_status = 'open'
                          AND addressed_to_id = ? AND source_id <> ?
                          AND observed_at < ?
                        ORDER BY observed_at DESC LIMIT 1
                    )
                    """,
                    (
                        event.event_id,
                        event.guild_id,
                        event.channel_id,
                        event.actor_id,
                        event.event_id,
                        event.utc_timestamp.isoformat(),
                    ),
                )
            await connection.execute(
                """
                DELETE FROM language_discourse_turns
                WHERE guild_id = ? AND source_id IN (
                    SELECT source_id FROM language_discourse_turns
                    WHERE guild_id = ?
                    ORDER BY observed_at DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (event.guild_id, event.guild_id, self.max_turns_per_guild),
            )

    async def rebuild_archive(self, guild_id: int) -> int:
        rows = await self._database.fetchall(
            """
            SELECT message_id, channel_id, author_id, content,
                   reply_to_message_id, created_at
            FROM archive_messages
            WHERE guild_id = ? AND content IS NOT NULL
            ORDER BY created_at ASC, message_id ASC
            """,
            (guild_id,),
        )
        async with self._database.transaction() as connection:
            await connection.execute(
                """
                DELETE FROM language_discourse_turns
                WHERE guild_id = ? AND source_id LIKE 'archive:%'
                """,
                (guild_id,),
            )
            inserted = 0
            for row in rows:
                inserted += await store_discourse_turn(
                    connection,
                    source_id=f"archive:{int(row['message_id'])}",
                    guild_id=guild_id,
                    channel_id=int(row["channel_id"]),
                    message_id=int(row["message_id"]),
                    parent_message_id=(
                        int(row["reply_to_message_id"])
                        if row["reply_to_message_id"] is not None
                        else None
                    ),
                    actor_id=int(row["author_id"]),
                    addressed_to_id=None,
                    content=self._clean(str(row["content"])),
                    observed_at=str(row["created_at"]),
                )
        return inserted

    async def context_for(self, event: Event, prompt: str) -> DiscourseContext:
        if event.channel_id is None:
            return DiscourseContext((), (), (), None, 0)
        prompt_topics = set(self._topics(prompt))
        addressed_to = self._addressed_to(event)
        chain, reply_depth = await self._reply_chain(event)
        chain_ids = {source_id for source_id, _ in chain}

        rows = await self._database.fetchall(
            """
            SELECT * FROM language_discourse_turns
            WHERE guild_id = ? AND channel_id = ? AND source_id <> ?
            ORDER BY observed_at DESC
            LIMIT 800
            """,
            (event.guild_id, event.channel_id, event.event_id),
        )
        topic_frequency: dict[str, int] = {}
        parsed_topics: dict[str, set[str]] = {}
        for row in rows:
            topics = set(self._row_topics(row))
            parsed_topics[str(row["source_id"])] = topics
            for topic in topics:
                topic_frequency[topic] = topic_frequency.get(topic, 0) + 1

        ranked: list[tuple[float, str, str]] = []
        total = max(1, len(rows))
        for index, row in enumerate(rows):
            source_id = str(row["source_id"])
            topics = parsed_topics[source_id]
            overlap = prompt_topics & topics
            specificity = sum(log1p(total / topic_frequency[topic]) for topic in overlap)
            score = specificity * 2.3 + (total - index) / total * 0.35
            score += 4.0 if source_id in chain_ids else 0.0
            score += (
                1.1 if event.actor_id is not None and row["actor_id"] == event.actor_id else 0.0
            )
            score += (
                1.8
                if event.actor_id is not None and row["addressed_to_id"] == event.actor_id
                else 0.0
            )
            if str(row["question_status"]) == "open":
                score += 0.6
            ranked.append((score, str(row["observed_at"]), str(row["content"])))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))

        messages: list[str] = [content for _, content in chain]
        for _, _, content in ranked:
            if content not in messages:
                messages.append(content)
            if len(messages) >= self.context_turn_limit:
                break
        questions = await self._open_questions(event, prompt_topics)
        reply_heads = await self._matching_reply_heads(event.guild_id, prompt_topics)
        return DiscourseContext(
            messages=tuple(messages),
            open_questions=questions,
            reply_heads=reply_heads,
            addressed_to_id=addressed_to,
            reply_depth=reply_depth,
        )

    async def stats(self, guild_id: int) -> DiscourseStats:
        row = await self._database.fetchone(
            """
            SELECT COUNT(*) AS turns,
                   SUM(question_status = 'open') AS open_questions,
                   SUM(question_status = 'answered') AS answered_questions
            FROM language_discourse_turns WHERE guild_id = ?
            """,
            (guild_id,),
        )
        return DiscourseStats(
            turns=int(row["turns"] or 0) if row else 0,
            open_questions=int(row["open_questions"] or 0) if row else 0,
            answered_questions=int(row["answered_questions"] or 0) if row else 0,
        )

    async def _reply_chain(self, event: Event) -> tuple[list[tuple[str, str]], int]:
        if event.channel_id is None:
            return [], 0
        parent = self._payload_int(event, "reply_to_message_id")
        chain: list[tuple[str, str]] = []
        if parent is None:
            implicit = await self._database.fetchone(
                """
                SELECT source_id, content FROM language_discourse_turns
                WHERE guild_id = ? AND channel_id = ? AND answered_by_source_id = ?
                ORDER BY observed_at DESC LIMIT 1
                """,
                (event.guild_id, event.channel_id, event.event_id),
            )
            if implicit is not None:
                chain.append((str(implicit["source_id"]), str(implicit["content"])))
        visited: set[int] = set()
        while parent is not None and parent not in visited and len(chain) < 30:
            visited.add(parent)
            row = await self._database.fetchone(
                """
                SELECT source_id, content, parent_message_id
                FROM language_discourse_turns
                WHERE guild_id = ? AND channel_id = ? AND message_id = ?
                """,
                (event.guild_id, event.channel_id, parent),
            )
            if row is None:
                break
            chain.append((str(row["source_id"]), str(row["content"])))
            parent = int(row["parent_message_id"]) if row["parent_message_id"] else None
        chain.reverse()
        return chain, len(chain)

    async def _open_questions(self, event: Event, prompt_topics: set[str]) -> tuple[str, ...]:
        rows = await self._database.fetchall(
            """
            SELECT content, topics_json, addressed_to_id, actor_id
            FROM language_discourse_turns
            WHERE guild_id = ? AND channel_id = ? AND question_status = 'open'
              AND source_id <> ?
            ORDER BY observed_at DESC LIMIT 30
            """,
            (event.guild_id, event.channel_id, event.event_id),
        )
        ranked: list[tuple[float, str]] = []
        for row in rows:
            topics = set(self._row_topics(row))
            overlap = len(prompt_topics & topics)
            addressed = event.actor_id is not None and row["addressed_to_id"] == event.actor_id
            same_author = event.actor_id is not None and row["actor_id"] == event.actor_id
            ranked.append(
                (overlap * 2.0 + float(addressed) * 3.0 + float(same_author), str(row["content"]))
            )
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(content for _, content in ranked[:8])

    async def _matching_reply_heads(
        self, guild_id: int, prompt_topics: set[str]
    ) -> tuple[tuple[str, ...], ...]:
        rows = await self._database.fetchall(
            """
            SELECT question.topics_json, answer.content
            FROM language_discourse_turns AS question
            JOIN language_discourse_turns AS answer
              ON answer.source_id = question.answered_by_source_id
            WHERE question.guild_id = ? AND question.question_status = 'answered'
            ORDER BY question.observed_at DESC LIMIT 300
            """,
            (guild_id,),
        )
        ranked: list[tuple[int, tuple[str, ...]]] = []
        for row in rows:
            topics = set(json.loads(str(row["topics_json"])))
            head = tokenize(str(row["content"]))[:4]
            if head:
                ranked.append((len(prompt_topics & topics), head))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        unique: list[tuple[str, ...]] = []
        for overlap, head in ranked:
            if overlap <= 0 and unique:
                break
            if head not in unique:
                unique.append(head)
            if len(unique) >= 8:
                break
        return tuple(unique)

    def _clean(self, content: str) -> str:
        return _SPACE_RE.sub(" ", _MENTION_RE.sub("", content)).strip()[
            : self.content_character_limit
        ]

    @staticmethod
    def _topics(content: str) -> tuple[str, ...]:
        return topic_keys(content)

    @staticmethod
    def _row_topics(row: Any) -> tuple[str, ...]:
        return tuple(str(topic) for topic in json.loads(str(row["topics_json"])))

    @classmethod
    def _addressed_to(cls, event: Event) -> int | None:
        if event.target_id is not None:
            return event.target_id
        mentions = event.payload.get("mentioned_user_ids")
        if isinstance(mentions, list):
            return next(
                (
                    value
                    for value in mentions
                    if isinstance(value, int)
                    and not isinstance(value, bool)
                    and value != event.actor_id
                ),
                None,
            )
        return None

    @staticmethod
    def _payload_int(event: Event, key: str) -> int | None:
        value = event.payload.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None
