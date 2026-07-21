"""Russian direct-address recognition separated from sentence content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from debilgi.language.tokenizer import tokenize

if TYPE_CHECKING:
    from debilgi.language.morphology import MorphologicalReading, RussianMorphology

_TERMINAL = frozenset(".!?…")
_QUOTES = {"«": "»", "“": "”", '"': '"'}
_MARKUP = frozenset({"<user>", "<role>", "<channel>", "<url>"})
_ADDRESS_LEMMAS = frozenset(
    {
        "брат",
        "братишка",
        "гений",
        "господин",
        "госпожа",
        "девушка",
        "дебил",
        "друг",
        "дружище",
        "идиот",
        "коллега",
        "мадам",
        "мальчик",
        "народ",
        "парень",
        "подруга",
        "ребята",
        "товарищ",
        "умник",
        "чувак",
    }
)
_IMPERATIVE_HINTS = frozenset(
    {
        "давай",
        "запомни",
        "найди",
        "объясни",
        "ответь",
        "поищи",
        "покажи",
        "проверь",
        "расскажи",
        "скажи",
        "смотри",
        "слушай",
        "узнай",
    }
)
_QUESTION_HINTS = frozenset(
    {"где", "зачем", "как", "какой", "когда", "кто", "куда", "откуда", "почему", "что"}
)
_LEADING_FILLERS = frozenset({"а", "ладно", "ну", "так", "эй"})
_COMMAND_FILLERS = frozenset({"быстро", "же", "ка", "пожалуйста", "сейчас"})
_SECOND_PERSON_PRONOUNS = frozenset({"ты", "вы"})
_NO_SPACE_BEFORE = frozenset(",.!?:;)]}»…")
_NO_SPACE_AFTER = frozenset("([{«")


@dataclass(frozen=True, slots=True)
class AddressSpan:
    text: str
    token_start: int
    token_end: int
    position: str
    confidence: float
    lemmas: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AddressAnalysis:
    text: str
    tokens: tuple[str, ...]
    content_tokens: tuple[str, ...]
    addresses: tuple[AddressSpan, ...]

    @property
    def content_text(self) -> str:
        return _detokenize(self.content_tokens)

    @property
    def address_terms(self) -> frozenset[str]:
        return frozenset(lemma for span in self.addresses for lemma in span.lemmas)


class RussianAddressing:
    """Identify vocative-like spans without confusing them with objects or subjects."""

    def __init__(self, morphology: RussianMorphology | None = None) -> None:
        if morphology is None:
            from debilgi.language.morphology import RussianMorphology

            morphology = RussianMorphology()
        self._morphology = morphology

    def analyze(self, text: str) -> AddressAnalysis:
        tokens = tokenize(text)
        if not tokens:
            return AddressAnalysis(text, (), (), ())
        quoted = self._quoted_indices(tokens)
        spans: list[AddressSpan] = []
        commas = tuple(index for index, token in enumerate(tokens) if token == ",")

        if commas:
            first = commas[0]
            self._append_candidate(spans, tokens, 0, first, "initial", 0.96, quoted)
            last = commas[-1]
            self._append_candidate(
                spans,
                tokens,
                last + 1,
                self._terminal_start(tokens),
                "final",
                0.96,
                quoted,
            )
            for left, right in zip(commas, commas[1:], strict=False):
                self._append_candidate(
                    spans, tokens, left + 1, right, "parenthetical", 0.98, quoted
                )

        word_indices = tuple(
            index
            for index, token in enumerate(tokens)
            if token[:1].isalnum() and index not in quoted
        )
        if not spans and len(word_indices) >= 2:
            self._append_unpunctuated(spans, tokens, word_indices, quoted)

        unique = tuple(
            sorted(
                {(span.token_start, span.token_end): span for span in spans}.values(),
                key=lambda span: (span.token_start, span.token_end),
            )
        )
        content = self._content_without_addresses(tokens, unique)
        return AddressAnalysis(text, tokens, content, unique)

    def _append_candidate(
        self,
        spans: list[AddressSpan],
        tokens: tuple[str, ...],
        start: int,
        end: int,
        position: str,
        confidence: float,
        quoted: frozenset[int],
    ) -> bool:
        while start < end and tokens[start] in _MARKUP:
            start += 1
        while start < end and tokens[end - 1] in _TERMINAL:
            end -= 1
        if start >= end or any(index in quoted for index in range(start, end)):
            return False
        words = tuple(token for token in tokens[start:end] if token[:1].isalnum())
        if not 1 <= len(words) <= 4 or not self._is_address_phrase(words):
            return False
        lemmas = tuple(self._address_lemma(word) for word in words)
        spans.append(
            AddressSpan(
                text=_detokenize(tokens[start:end]),
                token_start=start,
                token_end=end,
                position=position,
                confidence=confidence,
                lemmas=lemmas,
            )
        )
        return True

    def _append_unpunctuated(
        self,
        spans: list[AddressSpan],
        tokens: tuple[str, ...],
        word_indices: tuple[int, ...],
        quoted: frozenset[int],
    ) -> None:
        words = tuple(tokens[index] for index in word_indices)
        address_start = 0
        while address_start < len(words) and words[address_start] in _LEADING_FILLERS:
            address_start += 1
        maximum = min(4, len(words) - address_start - 1)
        for size in range(maximum, 0, -1):
            rest = words[address_start + size :]
            if not self._starts_directed_clause(rest):
                continue
            if self._append_candidate(
                spans,
                tokens,
                word_indices[address_start],
                word_indices[address_start + size - 1] + 1,
                "initial_unpunctuated",
                0.82,
                quoted,
            ):
                return

        command_position = 0
        while command_position < len(words) and words[command_position] in _LEADING_FILLERS:
            command_position += 1
        if command_position >= len(words) or not self._is_imperative(words[command_position]):
            return
        address_start = command_position + 1
        while address_start < len(words) and words[address_start] in _COMMAND_FILLERS:
            address_start += 1
        maximum = min(4, len(words) - address_start)
        for size in range(maximum, 0, -1):
            remaining = words[address_start + size :]
            position = "medial_unpunctuated" if remaining else "final_unpunctuated"
            confidence = 0.76 if remaining else 0.74
            if self._append_candidate(
                spans,
                tokens,
                word_indices[address_start],
                word_indices[address_start + size - 1] + 1,
                position,
                confidence,
                quoted,
            ):
                return

    def _starts_directed_clause(self, words: tuple[str, ...]) -> bool:
        offset = 0
        while offset < len(words) and words[offset] in _LEADING_FILLERS:
            offset += 1
        if offset >= len(words):
            return False
        first = words[offset]
        if self._is_imperative(first) or first in _QUESTION_HINTS:
            return True
        if first not in _SECOND_PERSON_PRONOUNS or offset + 1 >= len(words):
            return False
        return any(
            reading.part_of_speech == "verb" and reading.person == "second"
            for reading in self._morphology.parse_word(words[offset + 1])
        )

    def _is_address_phrase(self, words: tuple[str, ...]) -> bool:
        readings = tuple(self._morphology.parse_word(word) for word in words)
        head_options = readings[-1]
        head = next((item for item in head_options if self._address_reading(item)), None)
        if head is None:
            return False
        for options in readings[:-1]:
            if not any(
                item.part_of_speech in {"adjective", "participle"}
                or self._address_reading(item)
                for item in options
            ):
                return False
        return True

    @staticmethod
    def _address_reading(reading: MorphologicalReading) -> bool:
        named = bool(reading.grammemes & {"Name", "Surn", "Patr"}) and reading.case in {
            "nominative",
            "vocative",
        }
        lexical = (
            reading.lemma in _ADDRESS_LEMMAS
            and reading.case in {None, "nominative", "vocative"}
        )
        grammatical = (
            reading.part_of_speech == "noun"
            and reading.animacy == "animate"
            and reading.case in {"nominative", "vocative"}
        )
        return named or lexical or grammatical

    def _address_lemma(self, word: str) -> str:
        options = self._morphology.parse_word(word)
        preferred = next(
            (
                item
                for item in options
                if item.part_of_speech in {"adjective", "participle"}
                or self._address_reading(item)
            ),
            None,
        )
        return preferred.lemma if preferred is not None else self._morphology.lemma(word)

    def _is_imperative(self, word: str) -> bool:
        if word in _IMPERATIVE_HINTS:
            return True
        return any("impr" in reading.grammemes for reading in self._morphology.parse_word(word))

    def is_imperative(self, word: str) -> bool:
        """Expose the morphology-backed imperative check to dialogue classifiers."""

        return self._is_imperative(word.casefold().replace("ё", "е"))

    @staticmethod
    def _quoted_indices(tokens: tuple[str, ...]) -> frozenset[int]:
        quoted: set[int] = set()
        expected: str | None = None
        for index, token in enumerate(tokens):
            if expected is None and token in _QUOTES:
                expected = _QUOTES[token]
                quoted.add(index)
            elif expected is not None:
                quoted.add(index)
                if token == expected:
                    expected = None
        return frozenset(quoted)

    @staticmethod
    def _terminal_start(tokens: tuple[str, ...]) -> int:
        end = len(tokens)
        while end and tokens[end - 1] in _TERMINAL:
            end -= 1
        return end

    @staticmethod
    def _content_without_addresses(
        tokens: tuple[str, ...], spans: tuple[AddressSpan, ...]
    ) -> tuple[str, ...]:
        removed = {
            index
            for span in spans
            for index in range(span.token_start, span.token_end)
        }
        kept = [token for index, token in enumerate(tokens) if index not in removed]
        kept = [token for token in kept if token not in _MARKUP]
        while kept and kept[0] == ",":
            kept.pop(0)
        while kept and kept[-1] == ",":
            kept.pop()
        compact: list[str] = []
        for token in kept:
            if token == "," and (not compact or compact[-1] == ","):
                continue
            compact.append(token)
        return tuple(compact)


def _detokenize(tokens: tuple[str, ...]) -> str:
    result = ""
    previous = ""
    for token in tokens:
        if not result or token in _NO_SPACE_BEFORE or previous in _NO_SPACE_AFTER:
            result += token
        else:
            result += f" {token}"
        previous = token
    return result.strip(" ,")


DEFAULT_ADDRESSING = RussianAddressing()
