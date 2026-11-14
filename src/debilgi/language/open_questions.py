"""Persistent per-user binding between lexical questions and later explanations."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.events.models import Event, EventType
from debilgi.language.content import content_lemmas
from debilgi.language.context_gate import WorkingMemoryContextGate
from debilgi.language.lexicon import (
    LexiconService,
    is_substantive_definition,
    normalize_word,
)
from debilgi.storage.database import Database

logger = logging.getLogger("debilgi.open_questions")

_MENTION_RE = re.compile(r"<@!?\d+>")
_EXPLANATION_RE = re.compile(
    r"\b(?:ну\s+)?(?:это\s+(?:когда|типа|значит|из|от)|"
    r"значит|то\s+есть|под\s+этим\s+я\s+имею\s+в\s+виду|"
    r"я\s+называю\s+так|это\s+слово)",
    re.IGNORECASE,
)
_TOPIC_SHIFT_RE = re.compile(
    r"^(?:кстати|ладно[,.]?\s+(?:другой|новый)|другой\s+вопрос|"
    r"забудь|не\s+об\s+этом|сменим\s+тему)\b",
    re.IGNORECASE,
)
_QUESTION_START_RE = re.compile(
    r"^(?:а\s+)?(?:кто|что|где|когда|почему|зачем|как|какой|куда)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё-]{2,64}")


@dataclass(frozen=True, slots=True)
class OpenQuestion:
    question_id: str
    guild_id: int
    channel_id: int
    user_id: int
    concept_key: str
    prompt_event_id: str
    question_message_id: int
    asked_at: datetime
    turns_remaining: int


@dataclass(frozen=True, slots=True)
class ExplanationResolution:
    question_id: str
    concept_key: str
    explanation: str
    relevance_reason: str
    confirmation: str


class OpenLanguageQuestionService:
    def __init__(
        self,
        database: Database,
        lexicon: LexiconService,
        dataset: SelfDatasetRepository,
        *,
        ttl_messages: int = 2,
        working_context: WorkingMemoryContextGate | None = None,
    ) -> None:
        if ttl_messages <= 0:
            raise ValueError("open-question TTL must be positive")
        self._database = database
        self._lexicon = lexicon
        self._dataset = dataset
        self.ttl_messages = ttl_messages
        self._working_context = working_context
        self._lock = asyncio.Lock()

    async def open_for(
        self,
        prompt: Event,
        *,
        concept: str,
        question_message_id: int,
        asked_at: datetime,
    ) -> str | None:
        if prompt.channel_id is None or prompt.actor_id is None:
            return None
        concept_key = normalize_word(concept)
        if not concept_key:
            return None
        identity = (
            f"{prompt.guild_id}:{prompt.channel_id}:{prompt.actor_id}:"
            f"{question_message_id}:{concept_key}"
        )
        question_id = f"open-question:{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        replaced: list[tuple[str, str]] = []
        async with self._lock, self._database.transaction() as connection:
            existing = await (await connection.execute(
                "SELECT question_id FROM language_open_questions WHERE question_id = ?",
                (question_id,),
            )).fetchone()
            if existing is not None:
                return question_id
            rows = await (
                await connection.execute(
                    """SELECT question_id, concept_key FROM language_open_questions
                       WHERE guild_id = ? AND channel_id = ? AND user_id = ?
                         AND status = 'active'""",
                    (prompt.guild_id, prompt.channel_id, prompt.actor_id),
                )
            ).fetchall()
            replaced = [(str(row["question_id"]), str(row["concept_key"])) for row in rows]
            await connection.execute(
                """UPDATE language_open_questions
                   SET status = 'stale', relevance_reason = 'replaced_by_new_question',
                       resolved_at = ?
                   WHERE guild_id = ? AND channel_id = ? AND user_id = ?
                     AND status = 'active'""",
                (
                    asked_at.isoformat(),
                    prompt.guild_id,
                    prompt.channel_id,
                    prompt.actor_id,
                ),
            )
            await connection.execute(
                """INSERT INTO language_open_questions (
                    question_id, guild_id, channel_id, user_id, concept_key,
                    prompt_event_id, question_message_id, asked_at, turns_remaining
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    question_id,
                    prompt.guild_id,
                    prompt.channel_id,
                    prompt.actor_id,
                    concept_key,
                    prompt.event_id,
                    question_message_id,
                    asked_at.isoformat(),
                    self.ttl_messages,
                ),
            )
        for old_id, old_concept in replaced:
            self._log_final(
                old_id,
                old_concept,
                answer_received=False,
                accepted=False,
                status="stale",
                reason="replaced_by_new_question",
            )
        logger.info(
            "question=%s word=%r answer_received=false accepted=false status=active ttl=%d",
            question_id,
            concept_key,
            self.ttl_messages,
        )
        if self._working_context is not None:
            await self._working_context.set_pending_question(
                prompt.guild_id,
                prompt.channel_id,
                prompt.actor_id,
                concept_key,
            )
        return question_id

    async def restore_working_context(self) -> int:
        if self._working_context is None:
            return 0
        rows = await self._database.fetchall(
            """SELECT guild_id, channel_id, user_id, concept_key
               FROM language_open_questions WHERE status = 'active'"""
        )
        for row in rows:
            await self._working_context.set_pending_question(
                int(row["guild_id"]),
                int(row["channel_id"]),
                int(row["user_id"]),
                str(row["concept_key"]),
            )
        return len(rows)

    async def observe(self, event: Event) -> None:
        if (
            event.event_type not in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
            or event.payload.get("author_is_bot") is True
            or event.channel_id is None
            or event.actor_id is None
        ):
            return
        content = event.payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return
        async with self._lock:
            question = await self.active_for(
                event.guild_id,
                event.channel_id,
                event.actor_id,
            )
            if question is None:
                return
            if event.payload.get("suppress_full_parse") is True:
                # A bare "да/нет/ты понял" is feedback about the bot turn,
                # not a definition of the word the bot asked about.
                await self._decrement(question, event)
                return
            decision, reason = await self._classify(question, event, content)
            if decision == "learned":
                await self._accept(question, event, content, reason)
            elif decision == "topic_changed":
                await self._finish(
                    question,
                    status="topic_changed",
                    event=event,
                    response_text=content,
                    reason=reason,
                )
            else:
                await self._decrement(question, event)

    async def active_for(
        self,
        guild_id: int,
        channel_id: int,
        user_id: int,
    ) -> OpenQuestion | None:
        row = await self._database.fetchone(
            """SELECT * FROM language_open_questions
               WHERE guild_id = ? AND channel_id = ? AND user_id = ? AND status = 'active'
               ORDER BY asked_at DESC LIMIT 1""",
            (guild_id, channel_id, user_id),
        )
        question = _question_from_row(row) if row is not None else None
        if question is not None and self._working_context is not None:
            await self._working_context.set_pending_question(
                question.guild_id,
                question.channel_id,
                question.user_id,
                question.concept_key,
            )
        return question

    async def resolution_for_event(self, event_id: str) -> ExplanationResolution | None:
        row = await self._database.fetchone(
            """SELECT * FROM language_open_questions
               WHERE response_event_id = ? AND status = 'learned' LIMIT 1""",
            (event_id,),
        )
        if row is None:
            return None
        concept = str(row["concept_key"])
        explanation = str(row["response_text"] or "")
        reason = str(row["relevance_reason"] or "semantic_match")
        return ExplanationResolution(
            question_id=str(row["question_id"]),
            concept_key=concept,
            explanation=explanation,
            relevance_reason=reason,
            confirmation=_confirmation(concept, explanation),
        )

    async def _classify(
        self,
        question: OpenQuestion,
        event: Event,
        content: str,
    ) -> tuple[str, str]:
        cleaned = _clean(content)
        message_terms = set(content_lemmas(cleaned))
        concept_terms = set(content_lemmas(question.concept_key)) | {question.concept_key}
        related_terms = set(
            await self._dataset.related_concepts(
                question.guild_id,
                question.concept_key,
                limit=32,
            )
        )
        direct_word = _contains_word(cleaned, question.concept_key)
        related = bool(message_terms & (concept_terms | related_terms))
        explanation_trigger = bool(_EXPLANATION_RE.search(cleaned)) and bool(
            message_terms - concept_terms
        )
        has_definition = is_substantive_definition(question.concept_key, cleaned)
        direct_reply = event.payload.get("reply_to_message_id") == question.question_message_id
        mentions = event.payload.get("mentioned_user_ids")
        bot_user_id = event.payload.get("bot_user_id")
        other_user_mentioned = isinstance(mentions, list) and any(
            mentioned != bot_user_id for mentioned in mentions
        )
        topic_shift = bool(
            other_user_mentioned
            or _TOPIC_SHIFT_RE.search(cleaned)
            or (
                cleaned.rstrip().endswith("?")
                and _QUESTION_START_RE.search(cleaned)
                and not (direct_word or related or explanation_trigger or direct_reply)
            )
        )
        if topic_shift:
            return "topic_changed", "explicit_topic_shift"
        if direct_reply and has_definition:
            return "learned", "reply_to_question"
        if direct_word and has_definition:
            return "learned", "concept_mentioned"
        if related and has_definition:
            return "learned", "graph_relation"
        if explanation_trigger and has_definition:
            return "learned", "explanation_trigger"
        if direct_reply or direct_word or related or explanation_trigger:
            return "ambiguous", "meta_answer_without_definition"
        return "ambiguous", "no_reliable_link"

    async def _accept(
        self,
        question: OpenQuestion,
        event: Event,
        content: str,
        reason: str,
    ) -> None:
        cleaned = _clean(content)[:1000]
        associations = tuple(
            term
            for term in content_lemmas(cleaned)
            if term not in set(content_lemmas(question.concept_key))
        )[:24]
        await self._lexicon.teach_personal_interpretation(
            guild_id=question.guild_id,
            word=question.concept_key,
            interpretation=cleaned,
            associations=associations,
            confidence=0.72,
            source_event_ids=(event.event_id,),
            now=event.utc_timestamp,
        )
        await self._lexicon.resolve_unknown(
            question.guild_id,
            question.channel_id,
            question.concept_key,
        )
        await self._dataset.link_explanation(
            guild_id=question.guild_id,
            concept=question.concept_key,
            explanation=cleaned,
            source_event_id=event.event_id,
            observed_at=event.utc_timestamp,
            confidence=0.72,
        )
        await self._finish(
            question,
            status="learned",
            event=event,
            response_text=cleaned,
            reason=reason,
        )

    async def _decrement(self, question: OpenQuestion, event: Event) -> None:
        remaining = max(0, question.turns_remaining - 1)
        if remaining == 0:
            await self._finish(
                question,
                status="stale",
                event=event,
                response_text=None,
                reason="ttl_exhausted",
            )
            return
        await self._database.execute(
            """UPDATE language_open_questions SET turns_remaining = ?
               WHERE question_id = ? AND status = 'active'""",
            (remaining, question.question_id),
        )
        logger.info(
            "question=%s word=%r answer_received=false accepted=false "
            "status=active reason=no_reliable_link ttl=%d",
            question.question_id,
            question.concept_key,
            remaining,
        )

    async def _finish(
        self,
        question: OpenQuestion,
        *,
        status: str,
        event: Event,
        response_text: str | None,
        reason: str,
    ) -> None:
        await self._database.execute(
            """UPDATE language_open_questions SET
                status = ?, turns_remaining = 0, response_event_id = ?, response_text = ?,
                relevance_reason = ?, resolved_at = ?
               WHERE question_id = ? AND status = 'active'""",
            (
                status,
                event.event_id,
                response_text,
                reason,
                event.utc_timestamp.isoformat(),
                question.question_id,
            ),
        )
        if self._working_context is not None:
            await self._working_context.clear_pending_question(
                question.guild_id,
                question.channel_id,
                question.user_id,
                concept=question.concept_key,
            )
        self._log_final(
            question.question_id,
            question.concept_key,
            answer_received=status in {"learned", "topic_changed"},
            accepted=status == "learned",
            status=status,
            reason=reason,
        )

    @staticmethod
    def _log_final(
        question_id: str,
        concept: str,
        *,
        answer_received: bool,
        accepted: bool,
        status: str,
        reason: str,
    ) -> None:
        logger.info(
            "question=%s word=%r answer_received=%s accepted=%s status=%s reason=%s",
            question_id,
            concept,
            str(answer_received).lower(),
            str(accepted).lower(),
            status,
            reason,
        )


def _question_from_row(row: Any) -> OpenQuestion:
    return OpenQuestion(
        question_id=str(row["question_id"]),
        guild_id=int(row["guild_id"]),
        channel_id=int(row["channel_id"]),
        user_id=int(row["user_id"]),
        concept_key=str(row["concept_key"]),
        prompt_event_id=str(row["prompt_event_id"]),
        question_message_id=int(row["question_message_id"]),
        asked_at=datetime.fromisoformat(str(row["asked_at"])),
        turns_remaining=int(row["turns_remaining"]),
    )


def _clean(content: str) -> str:
    return " ".join(_MENTION_RE.sub("", content).split()).strip()


def _contains_word(content: str, word: str) -> bool:
    normalized = normalize_word(content)
    target = normalize_word(word)
    return target in {normalize_word(item.group(0)) for item in _WORD_RE.finditer(normalized)}


def _confirmation(concept: str, explanation: str) -> str:
    cleaned = _clean(explanation).strip(" .!?…")
    origin = re.search(r"(?:из|от)\s+слов[ао]\s+[«\"']?([а-яёa-z-]+)", cleaned, re.I)
    rejected = re.search(r"а\s+не\s+[«\"']?([а-яёa-z-]+)", cleaned, re.I)
    if origin:
        text = f"Понял: «{concept}» здесь связано со словом «{origin.group(1)}»"
        if rejected:
            text += f", а разбор как «{rejected.group(1)}» был ошибочным"
        return text + "."
    body = re.sub(
        r"^(?:ну\s+)?(?:это\s+(?:когда|типа|значит)?|значит|то\s+есть)\s*",
        "",
        cleaned,
        flags=re.I,
    ).strip(" ,:—-")
    if len(body) > 260:
        body = body[:260].rsplit(" ", 1)[0].rstrip(" ,;:—-") + "…"
    return f"Понял: «{concept}» — это примерно {body or cleaned}."
