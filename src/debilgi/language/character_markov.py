"""Bounded character-level Markov model for word forms, never reply meaning."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher

from debilgi.events.models import Event, EventType
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database

WORD_START = "<WORD_START>"
WORD_END = "<WORD_END>"
_MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
_WORD_RE = re.compile(r"^[^\W_]+(?:[-'][^\W_]+)*$", re.UNICODE)


@dataclass(frozen=True, slots=True)
class CharacterMarkovConfig:
    min_order: int = 2
    max_order: int = 8
    min_word_length: int = 3
    max_word_length: int = 32
    generation_attempts: int = 24
    max_repeated_characters: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.min_order <= self.max_order <= 16:
            raise ValueError("character Markov orders must satisfy 1 <= min <= max <= 16")
        if not 1 <= self.min_word_length <= self.max_word_length:
            raise ValueError("word length bounds are invalid")
        if self.generation_attempts <= 0 or self.max_repeated_characters <= 0:
            raise ValueError("generation limits must be positive")


class CharacterMarkovModel:
    """Learn and assess individual words with explicit boundaries and backoff."""

    def __init__(self, config: CharacterMarkovConfig | None = None) -> None:
        self.config = config or CharacterMarkovConfig()
        self.transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.words: set[str] = set()

    def train(self, words: Iterable[str]) -> int:
        learned = 0
        for raw_word in words:
            word = normalize_word(raw_word)
            if not is_word_form(word) or len(word) > self.config.max_word_length:
                continue
            self.words.add(word)
            sequence = (WORD_START,) * self.config.max_order + tuple(word) + (WORD_END,)
            for position in range(self.config.max_order, len(sequence)):
                following = sequence[position]
                for order in range(self.config.min_order, self.config.max_order + 1):
                    prefix = sequence[position - order : position]
                    self.transitions[prefix][following] += 1
            learned += 1
        return learned

    def plausibility(self, word: str) -> float:
        normalized = normalize_word(word)
        if not is_word_form(normalized):
            return 0.0
        sequence = (WORD_START,) * self.config.max_order + tuple(normalized) + (WORD_END,)
        known = 0.0
        checked = 0
        for position in range(self.config.max_order, len(sequence)):
            checked += 1
            following = sequence[position]
            match = self._options(sequence[:position])
            if match is None:
                continue
            _, options = match
            known += options.get(following, 0) / max(1, sum(options.values()))
        return min(1.0, known / max(1, checked))

    def generate(
        self,
        *,
        seed: str = "",
        existing_words: Iterable[str] = (),
        similar_threshold: float = 0.84,
    ) -> str | None:
        known = {normalize_word(word) for word in (*self.words, *tuple(existing_words))}
        if not self.transitions:
            return None
        for attempt in range(self.config.generation_attempts):
            rng = random.Random(_seed_number(f"{seed}:{attempt}"))
            history: list[str] = [WORD_START] * self.config.max_order
            output: list[str] = []
            reached_end = False
            while len(output) < self.config.max_word_length:
                match = self._options(tuple(history))
                if match is None:
                    break
                _, options = match
                following = _weighted_pick(options, rng)
                if following == WORD_END:
                    reached_end = True
                    break
                if following == WORD_START:
                    break
                output.append(following)
                history.append(following)
                if _has_excessive_repeat(output, self.config.max_repeated_characters):
                    break
            candidate = "".join(output)
            if not reached_end or not self._acceptable(candidate, known, similar_threshold):
                continue
            return candidate
        return None

    def _options(
        self, history: tuple[str, ...]
    ) -> tuple[tuple[str, ...], Mapping[str, int]] | None:
        for order in range(min(self.config.max_order, len(history)), self.config.min_order - 1, -1):
            prefix = history[-order:]
            options = self.transitions.get(prefix)
            if options:
                return prefix, options
        return None

    def _acceptable(self, candidate: str, known: set[str], similar_threshold: float) -> bool:
        if not self.config.min_word_length <= len(candidate) <= self.config.max_word_length:
            return False
        if not is_word_form(candidate) or _has_excessive_repeat(
            list(candidate), self.config.max_repeated_characters
        ):
            return False
        if candidate in known or self.plausibility(candidate) < 0.12:
            return False
        return all(
            SequenceMatcher(None, candidate, word).ratio() < similar_threshold
            for word in known
            if abs(len(candidate) - len(word)) <= 2
        )


class CharacterMarkovRepository:
    """Persist character transitions learned from real, separately bounded words."""

    def __init__(self, database: Database, config: CharacterMarkovConfig | None = None) -> None:
        self._database = database
        self.config = config or CharacterMarkovConfig()

    async def observe(self, event: Event) -> None:
        if event.event_type not in _MESSAGE_TYPES or event.payload.get("author_is_bot") is True:
            return
        if event.payload.get("suppress_full_parse") is True:
            return
        content = event.payload.get("content")
        if not isinstance(content, str):
            return
        await self.train_words(event.guild_id, words_from_text(content))

    async def train_words(self, guild_id: int, words: Iterable[str]) -> int:
        rows, learned = character_transition_rows(guild_id, words, self.config)
        if rows:
            async with self._database.transaction() as connection:
                await connection.executemany(
                    """
                    INSERT INTO language_character_transitions (
                        guild_id, order_size, prefix, next_character, count
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (guild_id, order_size, prefix, next_character)
                    DO UPDATE SET count = count + excluded.count
                    """,
                    rows,
                )
        return learned

    async def model(self, guild_id: int) -> CharacterMarkovModel:
        rows = await self._database.fetchall(
            """
            SELECT prefix, next_character, count
            FROM language_character_transitions
            WHERE guild_id = ? AND order_size BETWEEN ? AND ?
            """,
            (guild_id, self.config.min_order, self.config.max_order),
        )
        model = CharacterMarkovModel(self.config)
        for row in rows:
            prefix = tuple(str(part) for part in json.loads(str(row["prefix"])))
            model.transitions[prefix][str(row["next_character"])] += int(row["count"])
        return model


def normalize_word(word: str) -> str:
    return word.casefold().replace("’", "'").strip("-' ")


def is_word_form(word: str) -> bool:
    return bool(word and _WORD_RE.fullmatch(word))


def words_from_text(text: str) -> tuple[str, ...]:
    return tuple(word for token in tokenize(text) if is_word_form(word := normalize_word(token)))


def character_transition_rows(
    guild_id: int,
    words: Iterable[str],
    config: CharacterMarkovConfig | None = None,
) -> tuple[list[tuple[int, int, str, str, int]], int]:
    model = CharacterMarkovModel(config)
    learned = model.train(words)
    rows: list[tuple[int, int, str, str, int]] = []
    for prefix, options in model.transitions.items():
        serialized = json.dumps(prefix, ensure_ascii=False, separators=(",", ":"))
        rows.extend(
            (guild_id, len(prefix), serialized, following, count)
            for following, count in options.items()
        )
    return rows, learned


def _seed_number(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _weighted_pick(options: Mapping[str, int], rng: random.Random) -> str:
    total = sum(max(0, int(weight)) for weight in options.values())
    if total <= 0:
        return WORD_END
    point = rng.randrange(total)
    cumulative = 0
    for value, weight in sorted(options.items()):
        cumulative += max(0, int(weight))
        if point < cumulative:
            return value
    return WORD_END


def _has_excessive_repeat(characters: list[str], maximum: int) -> bool:
    if len(characters) <= maximum:
        return False
    return len(set(characters[-(maximum + 1) :])) == 1
