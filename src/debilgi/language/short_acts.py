"""Fast pre-parser for short conversational reactions and their antecedents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.events.models import Event, EventType
from debilgi.experience.consolidation import ExperienceConsolidator
from debilgi.language.context_gate import WorkingMemoryContextGate
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.tokenizer import canonical_chat_tokens
from debilgi.storage.database import Database

SHORT_ACT_PATTERNS: dict[str, frozenset[str]] = {
    "confirmation": frozenset(
        {
            "да",
            "ага",
            "угу",
            "именно",
            "точно",
            "верно",
            "правильно",
            "вот именно",
            "так и есть",
            "ты понял",
            "понял",
            "понятно",
            "согласен",
            "согласна",
        }
    ),
    "rejection": frozenset(
        {
            "нет",
            "нет же",
            "неа",
            "не то",
            "не так",
            "неправильно",
            "совсем не то",
            "ты не понял",
            "ты не поняла",
        }
    ),
    "continue": frozenset(
        {
            "и?",
            "ну и?",
            "дальше",
            "продолжай",
            "а дальше?",
            "что дальше?",
        }
    ),
    "reaction": frozenset(
        {
            "ясно",
            "хм",
            "хмм",
            "ладно",
            "ок",
            "окей",
            "интересно",
            "смешно",
            "красиво",
        }
    ),
}

_MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
_MENTION_RE = re.compile(r"<@!?\d+>")
_TRAILING_PUNCTUATION_RE = re.compile(r"[.!…]+$")
_PRONOUN_REFERENCES = frozenset({"ты", "тебя", "тебе", "это", "то", "так"})
_EXPLICIT_SUBJECTS = frozenset({"я", "мы", "он", "она", "оно", "они"})
_SCAFFOLDING = frozenset(
    {
        "а",
        "и",
        "ну",
        "же",
        "это",
        "то",
        "так",
        "ты",
        "тебя",
        "тебе",
        "чего",
        "что",
        "там",
        "тут",
    }
)


@dataclass(frozen=True, slots=True)
class ShortActDetection:
    kind: str
    matched_pattern: str | None
    recognized: bool
    suppress_full_parse: bool
    coreference_to_bot: bool
    interpretation: str
    word_count: int


@dataclass(frozen=True, slots=True)
class ShortActResolution:
    event_id: str
    kind: str
    recognized: bool
    coreference_to_bot: bool
    target_message_id: int | None
    target_source_id: str | None
    target_content: str | None
    interpretation: str

    def reply_text(self) -> str:
        if self.kind == "confirmation":
            return "Понял. Значит, предыдущая формулировка была принята собеседником."
        if self.kind == "rejection":
            return "Принял поправку: предыдущая формулировка не подошла. Пересоберу вывод."
        if self.kind == "continue":
            if self.target_content:
                excerpt = " ".join(self.target_content.split())[:180].rstrip(" ,;:—-")
                return (
                    f"Продолжу от предыдущей мысли: «{excerpt}». "
                    "Дальше могу раскрыть причину, пример или последствия."
                )
            return "Продолжу, но сначала нужен предыдущий предмет разговора."
        if self.kind == "reaction":
            return "Услышал реакцию. Предыдущую мысль пока не меняю."
        if self.coreference_to_bot and self.target_message_id is not None:
            return "Ты про мой предыдущий ответ?"
        return "Это продолжение предыдущей мысли?"


def detect_short_speech_act(text: str, *, maximum_words: int = 3) -> ShortActDetection | None:
    """Classify a structurally short reaction before morphology or graph extraction."""

    if maximum_words <= 0:
        raise ValueError("maximum_words must be positive")
    cleaned = " ".join(_MENTION_RE.sub("", text).casefold().replace("ё", "е").split())
    canonical_tokens = canonical_chat_tokens(cleaned)
    words = tuple(token for token in canonical_tokens if token[:1].isalnum())
    if not words or len(words) > maximum_words:
        return None
    key_text = " ".join(words) + ("?" if "?" in canonical_tokens else "")
    key = _pattern_key(key_text)
    for kind, patterns in SHORT_ACT_PATTERNS.items():
        if key in patterns:
            coreference = bool(set(words) & _PRONOUN_REFERENCES) or kind in {
                "confirmation",
                "rejection",
                "continue",
            }
            return ShortActDetection(
                kind=kind,
                matched_pattern=key,
                recognized=True,
                suppress_full_parse=True,
                coreference_to_bot=coreference,
                interpretation=_interpretation(kind),
                word_count=len(words),
            )
    if set(words) & _EXPLICIT_SUBJECTS or _has_concrete_noun(words):
        return None
    if len(words) == 1 and words[0] not in _SCAFFOLDING:
        reading = DEFAULT_MORPHOLOGY.best(words[0])
        if reading is None:
            return None
    coreference = bool(set(words) & _PRONOUN_REFERENCES)
    if coreference and any(word not in _SCAFFOLDING for word in words):
        return None
    structurally_incomplete = coreference or all(word in _SCAFFOLDING for word in words)
    if not structurally_incomplete:
        return None
    return ShortActDetection(
        kind="ambiguous",
        matched_pattern=None,
        recognized=False,
        suppress_full_parse=True,
        coreference_to_bot=coreference,
        interpretation=(
            "ambiguous_reference_to_last_bot_action"
            if coreference
            else "ambiguous_short_utterance"
        ),
        word_count=len(words),
    )


class ShortSpeechActService:
    def __init__(
        self,
        database: Database,
        experience: ExperienceConsolidator,
        dataset: SelfDatasetRepository,
        *,
        binding_window: timedelta = timedelta(minutes=10),
        working_context: WorkingMemoryContextGate | None = None,
    ) -> None:
        self._database = database
        self._experience = experience
        self._dataset = dataset
        self._working_context = working_context
        self.binding_window = binding_window

    async def observe(self, event: Event) -> None:
        if (
            event.event_type not in _MESSAGE_TYPES
            or event.payload.get("author_is_bot") is True
            or event.channel_id is None
        ):
            return
        kind = event.payload.get("short_speech_act")
        if not isinstance(kind, str):
            return
        target = await self._last_bot_turn(event)
        target_message_id = int(target["message_id"]) if target is not None else None
        target_source_id = str(target["source_id"]) if target is not None else None
        target_content = str(target["content"]) if target is not None else None
        cursor = await self._database.execute(
            """INSERT OR IGNORE INTO language_short_speech_acts (
                event_id, guild_id, channel_id, user_id, act_kind, matched_pattern,
                recognized, coreference_to_bot, target_message_id, target_source_id,
                target_content, interpretation, word_count, observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.guild_id,
                event.channel_id,
                event.actor_id,
                kind,
                event.payload.get("short_act_pattern"),
                int(event.payload.get("short_act_recognized") is True),
                int(event.payload.get("short_act_coreference") is True),
                target_message_id,
                target_source_id,
                target_content,
                str(event.payload.get("short_act_interpretation", "short_act")),
                int(event.payload.get("short_act_word_count", 0)),
                event.utc_timestamp.isoformat(),
            ),
        )
        if cursor.rowcount != 1:
            return
        recognized = event.payload.get("short_act_recognized") is True
        interpretation = str(event.payload.get("short_act_interpretation", "short_act"))
        _log_short_act(
            event,
            recognized=recognized,
            kind=kind,
            target_message_id=target_message_id,
            interpretation=interpretation,
        )
        if target_message_id is None or kind not in {"confirmation", "rejection"}:
            return
        accepted = kind == "confirmation"
        await self._experience.apply_short_act(
            event,
            response_message_id=target_message_id,
            accepted=accepted,
        )
        await self._dataset.record_short_act_feedback(
            event,
            response_message_id=target_message_id,
            accepted=accepted,
        )

    async def resolution_for_event(self, event_id: str) -> ShortActResolution | None:
        row = await self._database.fetchone(
            "SELECT * FROM language_short_speech_acts WHERE event_id = ?",
            (event_id,),
        )
        if row is None:
            return None
        return ShortActResolution(
            event_id=str(row["event_id"]),
            kind=str(row["act_kind"]),
            recognized=bool(row["recognized"]),
            coreference_to_bot=bool(row["coreference_to_bot"]),
            target_message_id=(
                int(row["target_message_id"]) if row["target_message_id"] is not None else None
            ),
            target_source_id=(
                str(row["target_source_id"]) if row["target_source_id"] else None
            ),
            target_content=str(row["target_content"]) if row["target_content"] else None,
            interpretation=str(row["interpretation"]),
        )

    async def _last_bot_turn(self, event: Event) -> Any | None:
        if self._working_context is not None and event.channel_id is not None:
            working_turn = await self._working_context.last_bot_turn(
                event.guild_id,
                event.channel_id,
                user_id=event.actor_id,
            )
            if working_turn is not None and working_turn.message_id is not None:
                return {
                    "source_id": working_turn.source_id,
                    "message_id": working_turn.message_id,
                    "content": working_turn.text,
                }
        reply_to = event.payload.get("reply_to_message_id")
        if isinstance(reply_to, int) and not isinstance(reply_to, bool):
            exact = await self._database.fetchone(
                """SELECT turn.source_id, turn.message_id, turn.content
                   FROM language_discourse_turns AS turn
                   JOIN events AS source ON source.event_id = turn.source_id
                   WHERE turn.guild_id = ? AND turn.channel_id = ? AND turn.message_id = ?
                     AND json_extract(source.payload_json, '$.author_is_bot') = 1""",
                (event.guild_id, event.channel_id, reply_to),
            )
            if exact is not None:
                return exact
        cutoff = (event.utc_timestamp - self.binding_window).isoformat()
        return await self._database.fetchone(
            """SELECT turn.source_id, turn.message_id, turn.content
               FROM language_discourse_turns AS turn
               JOIN events AS source ON source.event_id = turn.source_id
               WHERE turn.guild_id = ? AND turn.channel_id = ?
                 AND turn.observed_at >= ? AND turn.observed_at < ?
                 AND json_extract(source.payload_json, '$.author_is_bot') = 1
                 AND (turn.addressed_to_id = ? OR turn.addressed_to_id IS NULL)
               ORDER BY CASE WHEN turn.addressed_to_id = ? THEN 0 ELSE 1 END,
                        turn.observed_at DESC
               LIMIT 1""",
            (
                event.guild_id,
                event.channel_id,
                cutoff,
                event.utc_timestamp.isoformat(),
                event.actor_id,
                event.actor_id,
            ),
        )


def _pattern_key(cleaned: str) -> str:
    without_terminal = _TRAILING_PUNCTUATION_RE.sub("", cleaned).strip(" ,")
    question = without_terminal.endswith("?")
    base = without_terminal.rstrip("?").strip(" ,")
    return f"{base}?" if question and base in {"и", "ну и", "а дальше", "что дальше"} else base


def _has_concrete_noun(words: tuple[str, ...]) -> bool:
    return any(
        reading is not None and reading.part_of_speech in {"noun", "NOUN"}
        for word in words
        if (reading := DEFAULT_MORPHOLOGY.best(word)) is not None
    )


def _interpretation(kind: str) -> str:
    return {
        "confirmation": "accepted_previous_bot_response",
        "rejection": "rejected_previous_bot_response",
        "continue": "continue_previous_bot_response",
        "reaction": "reaction_to_previous_bot_response",
    }[kind]


def _log_short_act(
    event: Event,
    *,
    recognized: bool,
    kind: str,
    target_message_id: int | None,
    interpretation: str,
) -> None:
    import logging

    logging.getLogger("debilgi.short_speech_acts").info(
        "event=%s guild=%s channel=%s actor=%s recognized=%s act=%s "
        "target_message_id=%s interpretation=%s input=%r",
        event.event_id,
        event.guild_id,
        event.channel_id,
        event.actor_id,
        str(recognized).lower(),
        kind,
        target_message_id,
        interpretation,
        " ".join(str(event.payload.get("content", "")).split())[:400],
    )
