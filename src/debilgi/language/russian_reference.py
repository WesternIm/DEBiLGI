"""Cached background lookup against the public Russian Wiktionary API."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import socket
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from typing import Any

import aiohttp

from debilgi.events.models import Event, EventType
from debilgi.language.lexical_matching import fuzzy_match_allowed
from debilgi.language.lexicon import LexiconService
from debilgi.language.tokenizer import normalize_word, words_from_text
from debilgi.storage.database import Database

logger = logging.getLogger(__name__)
_MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
_CYRILLIC_WORD_RE = re.compile(r"^[а-яё-]{2,48}$", re.IGNORECASE)
_LANGUAGE_HEADING_RE = re.compile(r"(?m)^=\s*\{\{-[a-z-]+-\}\}\s*=$")
_POS_HEADING_RE = re.compile(r"(?m)^==\s*\{\{з\|\(([^)]+)\)\}\}\s*==$")
_MODERN_POS_PATTERNS = (
    (re.compile(r"\{\{\s*сущ-ru\b", re.IGNORECASE), "NOUN"),
    (re.compile(r"\{\{\s*прил(?:-ru)?\b", re.IGNORECASE), "ADJF"),
    (re.compile(r"\{\{\s*гл(?:-ru)?\b", re.IGNORECASE), "VERB"),
    (re.compile(r"\{\{\s*нареч(?:ие)?(?:-ru)?\b", re.IGNORECASE), "ADVB"),
    (re.compile(r"\{\{\s*частиц(?:а)?(?:-ru)?\b", re.IGNORECASE), "PRCL"),
    (re.compile(r"\{\{\s*междомет(?:ие)?(?:-ru)?\b", re.IGNORECASE), "INTJ"),
)
_MEANING_HEADING_RE = re.compile(r"(?m)^====\s*Значение\s*====\s*$")
_SUBHEADING_RE = re.compile(r"(?m)^====\s*[^=].*?====\s*$")
_DEFINITION_RE = re.compile(r"(?m)^#(?![#*:])\s*(.+)$")
_WIKILINK_RE = re.compile(r"\[\[(?:[^]|]+\|)?([^]]+)\]\]")
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
_TAG_RE = re.compile(r"<[^>]+>")
_STYLE_TEMPLATE_RE = re.compile(r"\{\{\s*([^|}]+)", re.IGNORECASE)
_EXPLICIT_LOOKUP_RE = re.compile(
    r"\b(?:что\s+(?:значит|означает)(?:\s+слово)?|значение\s+слова|"
    r"что\s+такое(?:\s+слово)?)\s+[«\"']?([а-яё-]{2,48})",
    re.IGNORECASE,
)
_POS_NAMES = {
    "междометие": "INTJ",
    "частица": "PRCL",
    "союз": "CONJ",
    "предлог": "PREP",
    "наречие": "ADVB",
    "существительное": "NOUN",
    "прилагательное": "ADJF",
    "глагол": "VERB",
    "местоимение": "NPRO",
    "числительное": "NUMR",
}
_STYLE_LABELS = {
    "бран.": "бранное",
    "вульг.": "вульгарное",
    "диал.": "диалектное",
    "жарг.": "жаргонное",
    "интернет.": "интернет-сленг",
    "мол.": "молодёжное",
    "неол.": "неологизм",
    "прост.": "просторечное",
    "разг.": "разговорное",
    "сленг.": "сленг",
}


@dataclass(frozen=True, slots=True)
class RemoteLexeme:
    word: str
    part_of_speech: str | None
    meanings: tuple[str, ...]
    source_url: str
    style_labels: tuple[str, ...] = ()
    derived_from: str | None = None
    examples: tuple[str, ...] = ()


class WiktionaryUnavailable(aiohttp.ClientError):
    """An operational failure, never evidence that a word is absent."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _retry_after_seconds(value: str | None, default: float = 60.0) -> float:
    if value:
        try:
            return max(default, float(int(value)))
        except ValueError:
            try:
                return max(
                    default, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
                )
            except (ValueError, TypeError, OverflowError):
                pass
    return default


class WiktionaryClient:
    def __init__(
        self,
        *,
        user_agent: str,
        timeout_seconds: int = 5,
    ) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self._session: aiohttp.ClientSession | None = None
        self._request_lock = asyncio.Lock()
        self._next_request_at = 0.0
        self._retry_at = 0.0
        self._failure: WiktionaryUnavailable | None = None

    @property
    def endpoint(self) -> str:
        return "https://ru.wiktionary.org/w/api.php"

    async def start(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                connector=aiohttp.TCPConnector(family=socket.AF_INET),
                headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            )

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def lookup(self, word: str) -> RemoteLexeme | None:
        """Try an exact page, then a tightly matched Wiktionary search suggestion."""

        await self.start()
        assert self._session is not None
        normalized = normalize_word(word)
        result = await self._lookup_page(normalized)
        if result is not None:
            return result
        variants = (normalized, *_safe_spelling_variants(normalized))
        for variant in variants[1:]:
            result = await self._lookup_page(variant)
            if result is not None:
                return result
        candidate = await self._search_candidate(normalized, variants)
        result = await self._lookup_page(candidate) if candidate else None
        if result is not None:
            return result
        for base_word in derivational_base_candidates(normalized):
            base = await self._lookup_page(base_word)
            if base is None or not base.meanings:
                continue
            definition = (
                f"женское название лица, соответствующее слову «{base.word}»; "
                f"базовое словарное значение: {base.meanings[0]}"
            )
            return RemoteLexeme(
                word=normalized,
                part_of_speech="NOUN",
                meanings=(definition,),
                source_url=base.source_url,
                style_labels=("словообразовательный вывод", *base.style_labels),
                derived_from=base.word,
                examples=(),
            )
        return None

    async def _lookup_page(self, word: str) -> RemoteLexeme | None:
        assert self._session is not None
        parameters = {
            "action": "parse",
            "page": word,
            "prop": "wikitext",
            "format": "json",
            "formatversion": "2",
            "utf8": "1",
            "maxlag": "5",
        }
        payload = await self._request(parameters)
        return parse_wiktionary_payload(word, payload)

    async def _request(self, parameters: dict[str, str]) -> dict[str, Any]:
        """Serialize all workers and share server backoff across words and actions."""
        assert self._session is not None
        async with self._request_lock:
            if self._failure is not None and time.monotonic() < self._retry_at:
                raise WiktionaryUnavailable(str(self._failure), status=self._failure.status)
            delay = self._next_request_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                async with self._session.get(self.endpoint, params=parameters) as response:
                    if response.status == 403:
                        self._retry_at = float("inf")
                        raise WiktionaryUnavailable(
                            "Wikimedia denied access (403). Check the bot User-Agent/contact "
                            "and Wikimedia robot policy; resolve access before restarting.",
                            status=403,
                        )
                    if response.status == 429 or response.status >= 500:
                        self._retry_at = time.monotonic() + _retry_after_seconds(
                            response.headers.get("Retry-After")
                        )
                        raise WiktionaryUnavailable(
                            "Wikimedia requested backoff", status=response.status
                        )
                    response.raise_for_status()
                    payload = await response.json(content_type=None)
                    if not isinstance(payload, dict):
                        raise WiktionaryUnavailable("Invalid Wikimedia API response")
                    error = payload.get("error")
                    if isinstance(error, dict):
                        code = error.get("code")
                        if code == "missingtitle" and parameters.get("action") == "parse":
                            return payload
                        self._retry_at = time.monotonic() + _retry_after_seconds(
                            response.headers.get("Retry-After")
                        )
                        raise WiktionaryUnavailable(f"Wikimedia API error: {code}")
                    expected = "parse" if parameters.get("action") == "parse" else "query"
                    if not isinstance(payload.get(expected), dict):
                        raise WiktionaryUnavailable(f"Missing Wikimedia API field: {expected}")
                    self._failure = None
                    return payload
            except (aiohttp.ClientError, TimeoutError, OSError, ValueError) as exc:
                failure = (
                    exc
                    if isinstance(exc, WiktionaryUnavailable)
                    else WiktionaryUnavailable(str(exc), status=getattr(exc, "status", None))
                )
                self._failure = failure
                self._retry_at = max(self._retry_at, time.monotonic() + 60.0)
                raise failure from None
            finally:
                self._next_request_at = time.monotonic() + 1.0

    async def _search_candidate(self, word: str, search_terms: tuple[str, ...]) -> str | None:
        assert self._session is not None
        for search_term in search_terms:
            parameters = {
                "action": "query",
                "list": "search",
                "srsearch": search_term,
                "srnamespace": "0",
                "srlimit": "8",
                "format": "json",
                "formatversion": "2",
                "utf8": "1",
                "maxlag": "5",
            }
            payload = await self._request(parameters)
            query = payload.get("query") if isinstance(payload, dict) else None
            raw_results = query.get("search") if isinstance(query, dict) else None
            titles = (
                [item.get("title") for item in raw_results if isinstance(item, dict)]
                if isinstance(raw_results, list)
                else ()
            )
            candidate = best_wiktionary_candidate(word, titles)
            if candidate:
                return candidate
        return None


class RussianReferenceService:
    """Queue rare lookups and import successful results into the local lexicon."""

    def __init__(
        self,
        database: Database,
        lexicon: LexiconService,
        client: WiktionaryClient,
        *,
        enabled: bool = False,
        refresh_after: timedelta = timedelta(days=30),
        queue_limit: int = 128,
        worker_count: int = 4,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("reference worker count must be positive")
        self._database = database
        self._lexicon = lexicon
        self._client = client
        self.enabled = enabled
        self.refresh_after = refresh_after
        self.worker_count = min(worker_count, 16)
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max(1, queue_limit))
        self._queued: set[str] = set()
        self._workers: list[asyncio.Task[None]] = []
        self._inflight: dict[str, asyncio.Task[RemoteLexeme | None]] = {}

    async def start(self) -> None:
        if self.enabled and not self._workers:
            await self._client.start()
            self._workers = [
                asyncio.create_task(
                    self._run(),
                    name=f"russian-reference-worker-{index}",
                )
                for index in range(self.worker_count)
            ]

    async def stop(self) -> None:
        workers, self._workers = self._workers, []
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        inflight = tuple(self._inflight.values())
        for task in inflight:
            task.cancel()
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
        self._inflight.clear()
        await self._client.close()

    async def observe(self, event: Event) -> None:
        if (
            not self.enabled
            or event.event_type not in _MESSAGE_TYPES
            or event.payload.get("author_is_bot") is True
            or event.payload.get("suppress_full_parse") is True
        ):
            return
        content = event.payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return
        explicit = reference_lookup_target(content)
        candidates = [explicit] if explicit else []
        if event.payload.get("bot_mentioned") is True:
            candidates.extend(words_from_text(content))
        queued = 0
        for word in dict.fromkeys(candidates):
            if queued >= 3:
                break
            analysis = await self._lexicon.analyze_word(
                word,
                guild_id=event.guild_id,
                channel_id=event.channel_id,
                context=content,
            )
            needs_meaning = not analysis.dictionary_meanings and (
                analysis.unknown_kind is not None
                or analysis.part_of_speech in {"PRCL", "INTJ", "CONJ", "PREP"}
            )
            if needs_meaning and await self.enqueue(word):
                queued += 1

    async def enqueue(self, word: str) -> bool:
        normalized = normalize_word(word)
        if not self.enabled or not _CYRILLIC_WORD_RE.fullmatch(normalized):
            return False
        if normalized in self._queued or not await self._needs_refresh(normalized):
            return False
        try:
            self._queue.put_nowait(normalized)
        except asyncio.QueueFull:
            return False
        self._queued.add(normalized)
        return True

    async def refresh_word(self, word: str) -> RemoteLexeme | None:
        """Resolve one word now; useful for explicit diagnostics and deterministic tests."""

        normalized = normalize_word(word)
        if not _CYRILLIC_WORD_RE.fullmatch(normalized):
            return None
        try:
            result = await self._client.lookup(normalized)
        except (aiohttp.ClientError, TimeoutError, OSError) as exc:
            logger.warning(
                "Dictionary lookup failed for %r: %s status=%s: %s",
                normalized,
                type(exc).__name__,
                getattr(exc, "status", None),
                exc,
            )
            await self._record(normalized, "error", None)
            return None
        if result is None or not result.meanings:
            await self._record(normalized, "missing", result)
            return None
        await self._lexicon.import_dictionary_entry(
            lemma=result.word,
            part_of_speech=result.part_of_speech,
            meanings=result.meanings,
            forms={normalized: {}, result.word: {}},
            style_labels=("wiktionary_reference", *result.style_labels),
            source="ru_wiktionary_derived" if result.derived_from else "ru_wiktionary",
            confidence=0.64 if result.derived_from else 0.76,
        )
        await self._record(normalized, "found", result)
        return result

    async def resolve_for_context(self, word: str) -> RemoteLexeme | None:
        """Use the durable lookup cache before making an autonomous network request."""

        normalized = normalize_word(word)
        if not _CYRILLIC_WORD_RE.fullmatch(normalized):
            return None
        if self.enabled and await self._needs_refresh(normalized):
            task = self._inflight.get(normalized)
            if task is None:
                task = asyncio.create_task(self.refresh_word(normalized))
                self._inflight[normalized] = task
                task.add_done_callback(lambda done: self._finish_lookup(normalized, done))
            # A foreground budget expiring must not cancel the shared lookup.
            return await asyncio.shield(task)
        row = await self._database.fetchone(
            """SELECT source, status, source_url, part_of_speech, meanings_json
               FROM language_reference_lookups WHERE normalized_form = ?""",
            (normalized,),
        )
        if row is None or str(row["status"]) != "found":
            return None
        meanings = tuple(str(item) for item in json.loads(str(row["meanings_json"])))
        if not meanings:
            return None
        return RemoteLexeme(
            word=normalized,
            part_of_speech=(str(row["part_of_speech"]) if row["part_of_speech"] else None),
            meanings=meanings,
            source_url=str(row["source_url"] or ""),
        )

    def _finish_lookup(self, word: str, task: asyncio.Task) -> None:
        if self._inflight.get(word) is task:
            self._inflight.pop(word, None)
        if not task.cancelled():
            task.exception()  # observe an exception even if all waiters timed out

    async def retry_delay_for(self, words: tuple[str, ...]) -> int | None:
        """Expose pending/error state without treating a missing entry as an outage."""
        if not self.enabled:
            return None
        delays = []
        for word in words:
            normalized = normalize_word(word)
            if normalized in self._inflight or normalized in self._queued:
                delays.append(5)
                continue
            row = await self._database.fetchone(
                "SELECT status, checked_at FROM language_reference_lookups WHERE normalized_form=?",
                (normalized,),
            )
            if row is not None and row["status"] == "error":
                checked = datetime.fromisoformat(str(row["checked_at"]))
                if checked.tzinfo is None:
                    checked = checked.replace(tzinfo=UTC)
                remaining = 300 - (datetime.now(UTC) - checked).total_seconds()
                delays.append(max(1, int(remaining) + 1))
        return min(delays) if delays else None

    async def _run(self) -> None:
        while True:
            word = await self._queue.get()
            try:
                await self.resolve_for_context(word)
            finally:
                self._queued.discard(word)
                self._queue.task_done()

    async def _needs_refresh(self, normalized: str) -> bool:
        row = await self._database.fetchone(
            "SELECT checked_at, status FROM language_reference_lookups WHERE normalized_form = ?",
            (normalized,),
        )
        if row is None:
            return True
        checked_at = datetime.fromisoformat(str(row["checked_at"]))
        if checked_at.tzinfo is None:
            checked_at = checked_at.replace(tzinfo=UTC)
        refresh_after = (
            min(self.refresh_after, timedelta(minutes=5))
            if row["status"] == "error"
            else min(self.refresh_after, timedelta(days=1))
            if row["status"] == "missing"
            else self.refresh_after
        )
        return datetime.now(UTC) - checked_at >= refresh_after

    async def _record(self, normalized: str, status: str, result: RemoteLexeme | None) -> None:
        source = "ru_wiktionary_derived" if result and result.derived_from else "ru_wiktionary"
        await self._database.execute(
            """INSERT INTO language_reference_lookups (
                   normalized_form, source, status, source_url,
                   part_of_speech, meanings_json, checked_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (normalized_form) DO UPDATE SET
                   status = excluded.status,
                   source_url = excluded.source_url,
                   part_of_speech = excluded.part_of_speech,
                   meanings_json = excluded.meanings_json,
                   checked_at = excluded.checked_at""",
            (
                normalized,
                source,
                status,
                result.source_url if result else None,
                result.part_of_speech if result else None,
                json.dumps(result.meanings if result else (), ensure_ascii=False),
                datetime.now(UTC).isoformat(),
            ),
        )


def reference_lookup_target(content: str) -> str | None:
    match = _EXPLICIT_LOOKUP_RE.search(content)
    return normalize_word(match.group(1)) if match else None


def derivational_base_candidates(word: str) -> tuple[str, ...]:
    """Return conservative bases for productive feminine person-name suffixes."""

    normalized = normalize_word(word)
    if normalized.endswith("есса") and len(normalized) >= 7:
        return (normalized[:-4],)
    return ()


def parse_wiktionary_payload(word: str, payload: Any) -> RemoteLexeme | None:
    if not isinstance(payload, dict) or isinstance(payload.get("error"), dict):
        return None
    parsed = payload.get("parse")
    if not isinstance(parsed, dict):
        return None
    wikitext = parsed.get("wikitext")
    if not isinstance(wikitext, str):
        return None
    russian_start = wikitext.find("{{-ru-}}")
    if russian_start < 0:
        return None
    russian = wikitext[russian_start:]
    next_language = _LANGUAGE_HEADING_RE.search(russian, pos=8)
    if next_language:
        russian = russian[: next_language.start()]
    headings = tuple(_POS_HEADING_RE.finditer(russian))
    meanings: list[str] = []
    examples: list[str] = []
    style_labels: list[str] = []
    chosen_pos = _modern_part_of_speech(russian)
    sections: list[tuple[str | None, str]] = []
    if headings:
        for index, heading in enumerate(headings):
            section_end = headings[index + 1].start() if index + 1 < len(headings) else len(russian)
            sections.append(
                (heading.group(1).casefold().strip(), russian[heading.end() : section_end])
            )
    else:
        sections.append((None, russian))
    for raw_pos, section in sections:
        meaning_heading = _MEANING_HEADING_RE.search(section)
        if meaning_heading is None:
            continue
        meaning_section = section[meaning_heading.end() :]
        next_subheading = _SUBHEADING_RE.search(meaning_section)
        if next_subheading:
            meaning_section = meaning_section[: next_subheading.start()]
        raw_definitions = tuple(_DEFINITION_RE.findall(meaning_section))
        if raw_definitions and chosen_pos is None and raw_pos is not None:
            chosen_pos = _POS_NAMES.get(raw_pos)
        for raw in raw_definitions:
            for example in _extract_usage_examples(raw):
                if len(examples) >= 5:
                    break
                if example not in examples:
                    examples.append(example)
            definition = _clean_wikitext(raw)
            if not definition:
                continue
            for label in _definition_style_labels(raw):
                if label not in style_labels:
                    style_labels.append(label)
            if definition not in meanings:
                meanings.append(definition)
            if len(meanings) >= 4:
                break
        if len(meanings) >= 4:
            break
    normalized = normalize_word(word)
    if not meanings:
        return None
    return RemoteLexeme(
        normalized,
        chosen_pos,
        tuple(meanings),
        f"https://ru.wiktionary.org/wiki/{normalized}",
        tuple(style_labels),
        None,
        tuple(examples[:5]),
    )


def best_wiktionary_candidate(word: str, titles: Any) -> str | None:
    """Accept only a close one-word suggestion, avoiding unrelated search results."""

    normalized = normalize_word(word)
    if not isinstance(titles, list | tuple):
        return None
    candidates: list[tuple[float, str]] = []
    for title in titles:
        if not isinstance(title, str):
            continue
        candidate = normalize_word(title)
        if not _CYRILLIC_WORD_RE.fullmatch(candidate) or ":" in title:
            continue
        if candidate == normalized:
            return candidate
        if not fuzzy_match_allowed(normalized, candidate):
            continue
        if abs(len(candidate) - len(normalized)) > 2:
            continue
        similarity = SequenceMatcher(None, normalized, candidate).ratio()
        if similarity >= 0.78:
            candidates.append((similarity, candidate))
    return max(candidates, default=(0.0, ""))[1] or None


def _safe_spelling_variants(word: str) -> tuple[str, ...]:
    if len(word) >= 3 and word[-1] == word[-2]:
        return (word[:-1],)
    return ()


def _definition_style_labels(value: str) -> tuple[str, ...]:
    labels: list[str] = []
    for raw in _STYLE_TEMPLATE_RE.findall(value):
        label = _STYLE_LABELS.get(raw.casefold().strip())
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


def _modern_part_of_speech(value: str) -> str | None:
    return next((part for pattern, part in _MODERN_POS_PATTERNS if pattern.search(value)), None)


def _clean_wikitext(value: str) -> str:
    text = value.split("{{пример", 1)[0]
    text = _WIKILINK_RE.sub(r"\1", text)
    previous = None
    while previous != text:
        previous = text
        text = _TEMPLATE_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = re.sub(r"'{2,}", "", text)
    return " ".join(text.split()).strip(" —;,.")[:500]


def _extract_usage_examples(value: str) -> tuple[str, ...]:
    """Read nested ``{{пример|...}}`` templates and discard template metadata."""

    examples: list[str] = []
    cursor = 0
    while len(examples) < 5:
        start = value.casefold().find("{{пример", cursor)
        if start < 0:
            break
        end = _balanced_template_end(value, start)
        if end is None:
            break
        parts = _split_template_arguments(value[start + 2 : end - 2])
        if parts and parts[0].casefold().strip() == "пример" and len(parts) > 1:
            example = _clean_example_wikitext(parts[1])
            if example and example not in examples:
                examples.append(example)
        cursor = end
    return tuple(examples)


def _balanced_template_end(value: str, start: int) -> int | None:
    depth = 0
    index = start
    while index < len(value) - 1:
        pair = value[index : index + 2]
        if pair == "{{":
            depth += 1
            index += 2
            continue
        if pair == "}}":
            depth -= 1
            index += 2
            if depth == 0:
                return index
            continue
        index += 1
    return None


def _split_template_arguments(value: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    template_depth = 0
    link_depth = 0
    index = 0
    while index < len(value):
        pair = value[index : index + 2]
        if pair == "{{":
            template_depth += 1
            index += 2
            continue
        if pair == "}}" and template_depth:
            template_depth -= 1
            index += 2
            continue
        if pair == "[[":
            link_depth += 1
            index += 2
            continue
        if pair == "]]" and link_depth:
            link_depth -= 1
            index += 2
            continue
        if value[index] == "|" and template_depth == 0 and link_depth == 0:
            parts.append(value[start:index])
            start = index + 1
        index += 1
    parts.append(value[start:])
    return tuple(parts)


def _clean_example_wikitext(value: str) -> str:
    text = re.sub(r"\{\{\s*выдел\s*\|\s*([^{}|]+?)\s*\}\}", r"\1", value)
    text = _WIKILINK_RE.sub(r"\1", text)
    previous = None
    while previous != text:
        previous = text
        text = _TEMPLATE_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = re.sub(r"'{2,}", "", text)
    return " ".join(text.split()).strip(" —")[:700]
