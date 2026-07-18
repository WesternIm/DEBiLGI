"""Small learned semantic frames extracted without authored response templates."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import aiosqlite

from debilgi.events.models import Event, EventType
from debilgi.language.grammar import role
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database

_MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
_CLAUSE_END = frozenset(".!?;:…")
_CLAUSE_LINKS = frozenset({"и", "а", "но", "затем", "потом", "поэтому"})
_LOCATION_QUESTIONS = frozenset({"где", "куда", "откуда"})
_STATE_ACTIONS = frozenset(
    {
        "находится",
        "находятся",
        "лежит",
        "лежат",
        "стоит",
        "стоят",
        "сидит",
        "сидят",
        "остается",
        "остаются",
        "осталась",
        "остались",
    }
)
_PREPOSITIONS = frozenset(
    {
        "в",
        "во",
        "на",
        "над",
        "под",
        "за",
        "из",
        "от",
        "к",
        "у",
        "около",
        "возле",
        "внутри",
        "между",
        "через",
        "с",
        "со",
        "без",
        "после",
        "перед",
    }
)
_PAST_SUFFIXES = (
    "нулась",
    "нулись",
    "ился",
    "илась",
    "ились",
    "ался",
    "алась",
    "ались",
    "яла",
    "яли",
    "ял",
    "ела",
    "ели",
    "ел",
    "ила",
    "или",
    "ил",
    "ыла",
    "ыли",
    "ыл",
    "ула",
    "ули",
    "ул",
)
_NOUN_ENDINGS = (
    "иями",
    "ями",
    "ами",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
    "иях",
    "ах",
    "ях",
    "ов",
    "ев",
    "ей",
    "ой",
    "ам",
    "ям",
    "ом",
    "ем",
    "ою",
    "ею",
    "а",
    "я",
    "ы",
    "и",
    "у",
    "ю",
    "е",
    "о",
)


@dataclass(frozen=True, slots=True)
class SemanticFrame:
    actor: str
    action: str
    target: str | None
    modifiers: tuple[tuple[str, str], ...]
    negated: bool
    confidence: float

    def surface_text(self) -> str:
        parts = [self.actor]
        if self.negated:
            parts.append("не")
        parts.append(self.action)
        if self.target:
            parts.append(self.target)
        for link, value in self.modifiers:
            parts.extend((link, value))
        return " ".join(parts)

    def terms(self) -> frozenset[str]:
        values = [self.actor, self.action, self.target or ""]
        values.extend(value for pair in self.modifiers for value in pair)
        return frozenset(
            key for value in values for token in tokenize(value) if (key := lexeme_key(token))
        )


def extract_semantic_frames(text: str) -> tuple[SemanticFrame, ...]:
    """Infer loose actor/action/object frames; uncertainty is kept as confidence."""

    if text.rstrip().endswith("?"):
        return ()
    tokens = tuple(token.casefold() for token in tokenize(text))
    clauses: list[tuple[str, ...]] = []
    current: list[str] = []
    for token in tokens:
        if token in _CLAUSE_END or token in _CLAUSE_LINKS:
            if current:
                clauses.append(tuple(current))
                current = []
            continue
        current.append(token)
    if current:
        clauses.append(tuple(current))

    frames: list[SemanticFrame] = []
    inherited_actor: str | None = None
    for clause in clauses:
        frame = _frame_from_clause(clause, inherited_actor)
        if frame is None:
            continue
        frames.append(frame)
        inherited_actor = frame.actor
    return tuple(frames)


def lexeme_key(value: str) -> str:
    """Cheap inflection-resistant key; the original surface word is still retained."""

    normalized = value.casefold().replace("ё", "е").strip(".,!?;:—-()[]{}«»…")
    if len(normalized) <= 4:
        return normalized
    for ending in _NOUN_ENDINGS:
        if normalized.endswith(ending) and len(normalized) - len(ending) >= 4:
            return normalized[: -len(ending)]
    return normalized


async def store_frames(
    connection: aiosqlite.Connection,
    *,
    source_id: str,
    source_kind: str,
    guild_id: int,
    channel_id: int | None,
    actor_id: int | None,
    text: str,
    observed_at: str,
) -> int:
    frames = extract_semantic_frames(text)
    if not frames:
        return 0
    await connection.executemany(
        """
        INSERT OR IGNORE INTO language_semantic_frames (
            source_id, frame_index, source_kind, guild_id, channel_id, actor_id,
            actor, action, target, modifiers_json, negated, confidence, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                source_id,
                index,
                source_kind,
                guild_id,
                channel_id,
                actor_id,
                frame.actor,
                frame.action,
                frame.target,
                json.dumps(frame.modifiers, ensure_ascii=False, separators=(",", ":")),
                int(frame.negated),
                frame.confidence,
                observed_at,
            )
            for index, frame in enumerate(frames)
        ),
    )
    return len(frames)


class SemanticMemoryRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def observe(self, event: Event) -> None:
        if event.event_type not in _MESSAGE_TYPES or event.payload.get("author_is_bot") is True:
            return
        content = event.payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return
        async with self._database.transaction() as connection:
            await store_frames(
                connection,
                source_id=event.event_id,
                source_kind="event",
                guild_id=event.guild_id,
                channel_id=event.channel_id,
                actor_id=event.actor_id,
                text=content,
                observed_at=event.utc_timestamp.isoformat(),
            )

    async def context(
        self, guild_id: int, prompt_tokens: Iterable[str], limit: int = 12
    ) -> tuple[str, ...]:
        rows = await self._database.fetchall(
            """
            SELECT actor, action, target, modifiers_json, negated, confidence, observed_at
            FROM language_semantic_frames
            WHERE guild_id = ?
            ORDER BY observed_at DESC, source_id DESC, frame_index DESC
            LIMIT 2000
            """,
            (guild_id,),
        )
        prompt_token_tuple = tuple(token.casefold().replace("ё", "е") for token in prompt_tokens)
        prompt_terms = {key for token in prompt_token_tuple if (key := lexeme_key(token))}
        ranked: list[tuple[float, int, str]] = []
        seen_counts: dict[str, int] = {}
        total = max(1, len(rows))
        for index, row in enumerate(rows):
            frame = self._from_row(row)
            rendered = frame.surface_text()
            seen_counts[rendered] = seen_counts.get(rendered, 0) + 1
            overlap = len(prompt_terms & frame.terms())
            recency = (total - index) / total
            question_fit = self._question_fit(prompt_token_tuple, frame)
            score = overlap * 3.0 + frame.confidence + recency * 0.20 + question_fit
            ranked.append((score, index, rendered))
        ranked.sort(
            key=lambda item: (
                -item[0] - min(1.0, (seen_counts[item[2]] - 1) * 0.12),
                item[1],
                item[2],
            )
        )
        selected: list[str] = []
        for _, _, rendered in ranked:
            if rendered not in selected:
                selected.append(rendered)
            if len(selected) >= max(1, min(limit, 50)):
                break
        return tuple(selected)

    async def count(self, guild_id: int) -> int:
        row = await self._database.fetchone(
            "SELECT COUNT(*) AS count FROM language_semantic_frames WHERE guild_id = ?",
            (guild_id,),
        )
        return int(row["count"]) if row else 0

    async def rebuild_archive(self, guild_id: int) -> int:
        """Reparse retained archive text after the extractor gains new capabilities."""

        rows = await self._database.fetchall(
            """
            SELECT message_id, channel_id, author_id, content, created_at
            FROM archive_messages
            WHERE guild_id = ? AND content IS NOT NULL
            ORDER BY created_at ASC, message_id ASC
            """,
            (guild_id,),
        )
        async with self._database.transaction() as connection:
            await connection.execute(
                """
                DELETE FROM language_semantic_frames
                WHERE guild_id = ? AND source_kind = 'archive'
                """,
                (guild_id,),
            )
            stored = 0
            for row in rows:
                stored += await store_frames(
                    connection,
                    source_id=f"archive:{int(row['message_id'])}",
                    source_kind="archive",
                    guild_id=guild_id,
                    channel_id=int(row["channel_id"]),
                    actor_id=int(row["author_id"]),
                    text=str(row["content"]),
                    observed_at=str(row["created_at"]),
                )
        return stored

    @staticmethod
    def _from_row(row: Any) -> SemanticFrame:
        modifiers = tuple(
            (str(pair[0]), str(pair[1]))
            for pair in json.loads(str(row["modifiers_json"]))
            if isinstance(pair, list) and len(pair) == 2
        )
        return SemanticFrame(
            actor=str(row["actor"]),
            action=str(row["action"]),
            target=str(row["target"]) if row["target"] is not None else None,
            modifiers=modifiers,
            negated=bool(row["negated"]),
            confidence=float(row["confidence"]),
        )

    @staticmethod
    def _question_fit(prompt_tokens: tuple[str, ...], frame: SemanticFrame) -> float:
        if not (_LOCATION_QUESTIONS & set(prompt_tokens)):
            return 0.0
        normalized_action = frame.action.replace("ё", "е")
        state_bonus = 2.0 if normalized_action in _STATE_ACTIONS else 0.0
        location_bonus = 0.65 if frame.modifiers else 0.0
        return state_bonus + location_bonus


def _frame_from_clause(
    tokens: tuple[str, ...], inherited_actor: str | None
) -> SemanticFrame | None:
    action_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if role(token) == "verb" or _looks_like_past_verb(token)
        ),
        None,
    )
    if action_index is None:
        return None

    actor_index = next(
        (
            index
            for index in range(action_index - 1, -1, -1)
            if role(tokens[index]) in {"content", "pronoun"}
        ),
        None,
    )
    actor = _noun_phrase(tokens, actor_index) if actor_index is not None else inherited_actor
    if actor is None:
        return None

    modifiers: list[tuple[str, str]] = []
    modifier_values: set[int] = set()
    for index in range(action_index + 1, len(tokens) - 1):
        if tokens[index] not in _PREPOSITIONS:
            continue
        value_index = next(
            (
                following
                for following in range(index + 1, len(tokens))
                if role(tokens[following]) in {"content", "pronoun", "number"}
            ),
            None,
        )
        if value_index is not None:
            modifiers.append((tokens[index], _noun_phrase(tokens, value_index)))
            modifier_values.add(value_index)

    target_index = next(
        (
            index
            for index, token in enumerate(tokens[action_index + 1 :], action_index + 1)
            if index not in modifier_values
            and role(token) in {"content", "pronoun", "number"}
            and (index == 0 or tokens[index - 1] not in _PREPOSITIONS)
        ),
        None,
    )
    target = _noun_phrase(tokens, target_index) if target_index is not None else None
    negated = "не" in tokens[max(0, action_index - 2) : action_index]
    confidence = 0.55 + 0.15 * (target is not None) + min(0.15, len(modifiers) * 0.075)
    if inherited_actor is not None and actor == inherited_actor and action_index == 0:
        confidence -= 0.08
    return SemanticFrame(
        actor=actor,
        action=tokens[action_index],
        target=target,
        modifiers=tuple(modifiers),
        negated=negated,
        confidence=max(0.35, min(0.95, confidence)),
    )


def _looks_like_past_verb(token: str) -> bool:
    normalized = token.casefold().replace("ё", "е")
    return len(normalized) >= 5 and normalized.endswith(_PAST_SUFFIXES)


def _noun_phrase(tokens: tuple[str, ...], index: int) -> str:
    start = index
    while start > 0 and role(tokens[start - 1]) == "adjective":
        start -= 1
    return " ".join(tokens[start : index + 1])
