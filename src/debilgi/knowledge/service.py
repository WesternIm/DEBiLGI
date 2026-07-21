"""Background acquisition service that keeps network latency outside the event queue."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from debilgi.cognition.core_repository import CognitiveRepository
from debilgi.events.models import Event
from debilgi.knowledge.models import KnowledgeConcept, KnowledgeDecision, WikipediaResult
from debilgi.knowledge.policy import (
    MEDIA_ARTICLE_MARKERS,
    KnowledgeQueryPolicy,
    canonical_lexical_query,
    media_article_reasons,
)
from debilgi.knowledge.repository import KnowledgeRepository
from debilgi.knowledge.text import normalize_key
from debilgi.language.content import content_lemmas

logger = logging.getLogger(__name__)
selection_logger = logging.getLogger("debilgi.knowledge_selection")
_REFERENCE_MAX_CHARS = 420
_REFERENCE_MORE_NOTE = " Если нужно подробнее — уточни."


class KnowledgeSource(Protocol):
    language: str

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def search(self, query: str) -> WikipediaResult | None: ...


class DictionaryLexeme(Protocol):
    word: str
    meanings: tuple[str, ...]
    source_url: str


class DictionarySource(Protocol):
    enabled: bool

    async def resolve_for_context(self, word: str) -> DictionaryLexeme | None: ...


@dataclass(frozen=True, slots=True)
class LookupSignals:
    surface: str
    lowercase_surface: bool
    media_context: bool
    dictionary_priority: bool
    conflict_lemma: bool


LEXICAL_MEDIA_CONFLICTS = frozenset({"умник"})
_MEDIA_CONTEXT_TERMS = frozenset(
    {
        "фильм",
        "сериал",
        "телесериал",
        "мультфильм",
        "игра",
        "альбом",
        "песня",
        "книга",
        "роман",
        "шоу",
        "аниме",
    }
)
class KnowledgeAcquisitionService:
    source_name = "wikipedia"

    def __init__(
        self,
        repository: KnowledgeRepository,
        cognitive_repository: CognitiveRepository,
        policy: KnowledgeQueryPolicy,
        source: KnowledgeSource,
        *,
        dictionary: DictionarySource | None = None,
        reply_wait_seconds: float = 1.8,
        queue_size: int = 100,
    ) -> None:
        self.repository = repository
        self.cognitive_repository = cognitive_repository
        self.policy = policy
        self.source = source
        self.dictionary = dictionary
        self.reply_wait_seconds = reply_wait_seconds
        self._queue: asyncio.Queue[tuple[str, str, LookupSignals]] = asyncio.Queue(
            maxsize=queue_size
        )
        self._worker: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return self.policy.enabled

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if not self.enabled or self._worker is not None:
            return
        await self.source.start()
        self._worker = asyncio.create_task(self._run(), name="debilgi-wikipedia-worker")

    async def stop(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=3.0)
            except TimeoutError:
                logger.warning("Wikipedia queue did not drain before shutdown")
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        await self.source.close()

    async def observe(self, event: Event) -> None:
        if not self.enabled or event.payload.get("suppress_full_parse") is True:
            return
        await self.repository.observe(event)
        await self.repository.remember_event_concepts(event)
        intent = await self.cognitive_repository.intent_for_event(event.event_id)
        decision = self.policy.decide(event, intent)
        if (
            not decision.should_query
            and decision.reason == "followup"
            and event.channel_id is not None
        ):
            recent_query = await self.repository.recent_explicit_query(
                event.guild_id,
                event.channel_id,
                event.actor_id,
                before=event.utc_timestamp,
            )
            if recent_query:
                decision = KnowledgeDecision(
                    True,
                    recent_query,
                    "explicit",
                    decision.uncertainty,
                )
        if not decision.should_query or decision.query is None or decision.reason is None:
            return
        raw_content = event.payload.get("content")
        signals = _lookup_signals(
            str(raw_content) if isinstance(raw_content, str) else decision.query,
            decision.query,
            dictionary_available=self._dictionary_enabled,
        )
        preferred_sources = ("ru_wiktionary", "ru_wiktionary_derived")
        cached = await self.repository.concept_for_alias(
            decision.query,
            preferred_sources=preferred_sources if signals.dictionary_priority else (),
        )
        cached_allowed = (
            cached is not None
            and _title_matches_query(decision.query, cached.title)
            and not (_concept_is_media(cached) and not signals.media_context)
            and not (
                signals.dictionary_priority
                and self._dictionary_enabled
                and not cached.source.startswith("ru_wiktionary")
            )
        )
        if cached_allowed and cached is not None:
            await self.repository.record_cache_hit(
                event,
                query=decision.query,
                reason=decision.reason,
                uncertainty=decision.uncertainty,
                concept_id=cached.concept_id,
                source=cached.source,
                language=self.source.language,
            )
            selection_logger.info(
                "query=%r selected_source=%s title=%r reason=preferred_cache "
                "lowercase=%s media_context=%s conflict=%s",
                decision.query,
                cached.source,
                cached.title,
                signals.lowercase_surface,
                signals.media_context,
                signals.conflict_lemma,
            )
            return
        if cached is not None:
            selection_logger.info(
                "query=%r rejected_cache_source=%s title=%r reason=%s",
                decision.query,
                cached.source,
                cached.title,
                "dictionary_priority"
                if signals.dictionary_priority and self._dictionary_enabled
                else "media_without_context",
            )
        query_id = await self.repository.reserve_query(
            event,
            query=decision.query,
            reason=decision.reason,
            uncertainty=decision.uncertainty,
            source=self.source_name,
            language=self.source.language,
        )
        if query_id is None:
            return
        try:
            self._queue.put_nowait((query_id, decision.query, signals))
        except asyncio.QueueFull:
            await self.repository.finish_without_result(query_id, error="acquisition queue is full")

    async def wait_for_event(self, event_id: str) -> str | None:
        if not self.enabled:
            return None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.reply_wait_seconds
        while True:
            status = await self.repository.query_status_for_event(event_id)
            if status is None or status != "pending":
                return status
            remaining = deadline - loop.time()
            if remaining <= 0:
                return status
            await asyncio.sleep(min(0.04, remaining))

    async def context_for_text(self, value: str) -> tuple[str, ...]:
        if not self.enabled:
            return ()
        signals = _lookup_signals(
            value,
            next(iter(content_lemmas(value)), value),
            dictionary_available=self._dictionary_enabled,
        )
        concepts = await self.repository.concepts_for_text(
            value,
            preferred_sources=("ru_wiktionary", "ru_wiktionary_derived")
            if signals.dictionary_priority
            else (),
        )
        terms = content_lemmas(value)
        relevant = tuple(
            concept
            for concept in concepts
            if any(_title_matches_query(term, concept.title) for term in terms)
            if signals.media_context or not _concept_is_media(concept)
        )
        return self._format_concepts(relevant)

    async def grounding_for_event(
        self, event: Event, prompt: str
    ) -> tuple[tuple[str, ...], str | None]:
        """Return facts plus an exact answer only for an intentional lookup event."""

        if not self.enabled:
            return (), None
        await self.wait_for_event(event.event_id)
        looked_up = await self.repository.resolved_concept_for_event(event.event_id)
        if looked_up is not None:
            facts = self._format_concepts((looked_up,))
            return facts, (facts[0] if facts else None)
        return await self.context_for_text(prompt), None

    async def investigate(self, event: Event, query: str) -> tuple[str, ...]:
        """Use cached or newly acquired evidence when response generation reaches an impasse."""

        if not self.enabled or event.channel_id is None or not query.strip():
            return ()
        raw_content = event.payload.get("content")
        context = str(raw_content) if isinstance(raw_content, str) else query
        query = canonical_lexical_query(query, context)
        await self.wait_for_event(event.event_id)
        learned = await self.repository.any_resolved_concept_for_event(event.event_id)
        if learned is not None:
            return self._format_concepts((learned,))
        signals = _lookup_signals(
            context,
            query,
            dictionary_available=self._dictionary_enabled,
        )
        cached = await self.repository.concept_for_alias(
            query,
            preferred_sources=("ru_wiktionary", "ru_wiktionary_derived")
            if signals.dictionary_priority
            else (),
        )
        cached_allowed = (
            cached is not None
            and _title_matches_query(query, cached.title)
            and not (_concept_is_media(cached) and not signals.media_context)
            and not (
                signals.dictionary_priority
                and self._dictionary_enabled
                and not cached.source.startswith("ru_wiktionary")
            )
        )
        if cached_allowed and cached is not None:
            await self.repository.record_cache_hit(
                event,
                query=query,
                reason="uncertainty",
                uncertainty=1.0,
                concept_id=cached.concept_id,
                source=cached.source,
                language=self.source.language,
            )
            return self._format_concepts((cached,))
        query_id = await self.repository.reserve_query(
            event,
            query=query,
            reason="uncertainty",
            uncertainty=1.0,
            source=self.source_name,
            language=self.source.language,
        )
        if query_id is None:
            return ()
        try:
            resolved = await asyncio.wait_for(
                self._resolve_lookup(query, signals),
                timeout=max(0.5, self.reply_wait_seconds),
            )
            if resolved is None:
                await self.repository.finish_without_result(query_id)
                return ()
            result, source, language, _ = resolved
            concept = await self.repository.complete_query(
                query_id,
                result,
                source=source,
                language=language,
            )
            logger.info("Autonomously resolved stalled concept %r", query)
            return self._format_concepts((concept,))
        except TimeoutError:
            await self.repository.finish_without_result(query_id, error="recovery lookup timed out")
            return ()
        except Exception as exc:
            logger.warning("Recovery lookup failed for %r: %s", query, exc)
            await self.repository.finish_without_result(
                query_id, error=f"{type(exc).__name__}: {exc}"
            )
            return ()

    async def internalize_for_reasoning(
        self,
        event: Event,
        terms: tuple[str, ...],
        *,
        limit: int = 6,
    ) -> tuple[KnowledgeConcept, ...]:
        """Study missing terms before generation, without returning source prose.

        The resulting concepts are persisted as internal evidence.  No explicit
        lookup ledger entry is created, so a dictionary or encyclopedia extract
        cannot accidentally become the response to the triggering message.
        """

        if not self.enabled or event.channel_id is None:
            return ()
        raw_content = event.payload.get("content")
        context = str(raw_content) if isinstance(raw_content, str) else " ".join(terms)
        normalized_terms = tuple(
            dict.fromkeys(
                canonical_lexical_query(term, context)
                for term in terms
                if term.strip()
            )
        )[: max(1, limit)]

        async def study(term: str) -> KnowledgeConcept | None:
            signals = _lookup_signals(
                context,
                term,
                dictionary_available=self._dictionary_enabled,
            )
            cached = await self.repository.concept_for_alias(
                term,
                preferred_sources=("ru_wiktionary", "ru_wiktionary_derived")
                if signals.dictionary_priority
                else (),
            )
            cached_allowed = (
                cached is not None
                and _title_matches_query(term, cached.title)
                and not (_concept_is_media(cached) and not signals.media_context)
                and not (
                    signals.dictionary_priority
                    and self._dictionary_enabled
                    and not cached.source.startswith("ru_wiktionary")
                )
            )
            if cached_allowed:
                selection_logger.info(
                    "query=%r selected_source=%s title=%r reason=internal_study_cache",
                    term,
                    cached.source,
                    cached.title,
                )
                return cached
            try:
                resolved = await asyncio.wait_for(
                    self._resolve_lookup(term, signals),
                    timeout=max(6.0, self.reply_wait_seconds),
                )
            except TimeoutError:
                logger.info("Internal study timed out for %r", term)
                return None
            except Exception as exc:
                logger.warning("Internal study failed for %r: %s", term, exc)
                return None
            if resolved is None:
                return None
            result, source, language, reason = resolved
            concept = await self.repository.store_internalized_concept(
                event,
                query=term,
                result=result,
                source=source,
                language=language,
            )
            logger.info(
                "Internally studied concept %r from %s before response (%s)",
                term,
                source,
                reason,
            )
            return concept

        studied = await asyncio.gather(*(study(term) for term in normalized_terms))
        return tuple(concept for concept in studied if concept is not None)

    @property
    def _dictionary_enabled(self) -> bool:
        return self.dictionary is not None and self.dictionary.enabled

    async def _resolve_lookup(
        self, query: str, signals: LookupSignals
    ) -> tuple[WikipediaResult, str, str, str] | None:
        if signals.dictionary_priority and self.dictionary is not None:
            lexeme = await self.dictionary.resolve_for_context(query)
            if lexeme is not None and lexeme.meanings:
                result = _dictionary_result(lexeme)
                reason = (
                    "manual_conflict_dictionary_priority"
                    if signals.conflict_lemma
                    else "lowercase_dictionary_priority"
                    if signals.lowercase_surface
                    else "dictionary_first"
                )
                selection_logger.info(
                    "query=%r surface=%r selected_source=ru_wiktionary title=%r "
                    "reason=%s media_context=%s",
                    query,
                    signals.surface,
                    result.title,
                    reason,
                    signals.media_context,
                )
                return result, "ru_wiktionary", "ru", reason
            selection_logger.info(
                "query=%r surface=%r dictionary_result=missing reason=fallback_to_wikipedia",
                query,
                signals.surface,
            )
        result = await self._search_encyclopedia(query, signals)
        if result is None:
            selection_logger.info(
                "query=%r selected_source=none reason=no_applicable_dictionary_or_article",
                query,
            )
            return None
        reason = (
            "explicit_media_context"
            if signals.media_context
            else "capitalized_title_fallback"
            if media_article_reasons(result) and not signals.lowercase_surface
            else "dictionary_miss"
        )
        selection_logger.info(
            "query=%r surface=%r selected_source=wikipedia title=%r reason=%s "
            "categories=%s lowercase=%s conflict=%s",
            query,
            signals.surface,
            result.title,
            reason,
            result.categories,
            signals.lowercase_surface,
            signals.conflict_lemma,
        )
        return result, self.source_name, self.source.language, reason

    async def _search_encyclopedia(
        self, query: str, signals: LookupSignals
    ) -> WikipediaResult | None:
        candidate_search = getattr(self.source, "search_candidates", None)
        if callable(candidate_search):
            candidates = tuple(await candidate_search(query, limit=5))
        else:
            single = await self.source.search(query)
            candidates = (single,) if single is not None else ()
        capitalized_media_fallback: WikipediaResult | None = None
        for result in candidates:
            if not result.extract.strip() or not _title_matches_query(query, result.title):
                selection_logger.info(
                    "query=%r rejected_article=%r reason=title_or_extract_mismatch",
                    query,
                    result.title,
                )
                continue
            media_reasons = media_article_reasons(result)
            if media_reasons and not signals.media_context:
                if not signals.lowercase_surface and not signals.conflict_lemma:
                    capitalized_media_fallback = capitalized_media_fallback or result
                    selection_logger.info(
                        "query=%r deferred_article=%r reason=capitalized_media_fallback "
                        "media_markers=%s categories=%s",
                        query,
                        result.title,
                        media_reasons,
                        result.categories,
                    )
                    continue
                selection_logger.info(
                    "query=%r rejected_article=%r reason=media_without_context "
                    "media_markers=%s categories=%s",
                    query,
                    result.title,
                    media_reasons,
                    result.categories,
                )
                continue
            return result
        return capitalized_media_fallback

    @staticmethod
    def _format_concepts(concepts: tuple[KnowledgeConcept, ...]) -> tuple[str, ...]:
        return tuple(
            concise_reference(concept.definition)
            for concept in concepts
            if concept.definition.strip()
        )

    async def _run(self) -> None:
        while True:
            query_id, query, signals = await self._queue.get()
            try:
                resolved = await self._resolve_lookup(query, signals)
                if resolved is None:
                    await self.repository.finish_without_result(query_id)
                else:
                    result, source, language, _ = resolved
                    await self.repository.complete_query(
                        query_id,
                        result,
                        source=source,
                        language=language,
                    )
                    logger.info("Learned %s concept %r", source, result.title)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Wikipedia lookup failed for %r: %s", query, exc)
                await self.repository.finish_without_result(
                    query_id, error=f"{type(exc).__name__}: {exc}"
                )
            finally:
                self._queue.task_done()


def _lookup_signals(
    content: str,
    query: str,
    *,
    dictionary_available: bool,
) -> LookupSignals:
    normalized_query = normalize_key(query)
    match = re.search(
        rf"(?<![\w-]){re.escape(normalized_query)}(?![\w-])",
        content,
        re.IGNORECASE,
    )
    surface = match.group(0) if match is not None else query.strip()
    first_letter = next((character for character in surface if character.isalpha()), "")
    lowercase_surface = bool(first_letter) and first_letter.islower()
    normalized_content = normalize_key(content)
    media_context = bool(
        set(content_lemmas(normalized_content)) & _MEDIA_CONTEXT_TERMS
        or any(
            re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", normalized_content)
            for term in _MEDIA_CONTEXT_TERMS
        )
    )
    conflict_lemma = normalized_query in LEXICAL_MEDIA_CONFLICTS
    dictionary_priority = bool(
        dictionary_available
        and len(normalized_query.split()) == 1
        and not media_context
    )
    return LookupSignals(
        surface=surface,
        lowercase_surface=lowercase_surface,
        media_context=media_context,
        dictionary_priority=dictionary_priority,
        conflict_lemma=conflict_lemma,
    )


def _dictionary_result(lexeme: DictionaryLexeme) -> WikipediaResult:
    definition = " ".join(lexeme.meanings[0].split())
    word = " ".join(lexeme.word.split())
    if not normalize_key(definition).startswith(normalize_key(word)):
        definition = f"{word} — {definition}"
    page_id = int.from_bytes(
        hashlib.sha256(normalize_key(word).encode("utf-8")).digest()[:8], "big"
    ) & 0x7FFF_FFFF_FFFF_FFFF
    return WikipediaResult(
        page_id=page_id,
        title=word,
        extract=definition,
        canonical_url=lexeme.source_url,
        wikidata_id=None,
        revision_id=None,
        related_titles=(),
        categories=("dictionary",),
    )


def _concept_is_media(concept: KnowledgeConcept) -> bool:
    evidence = normalize_key(f"{concept.title} {concept.definition}")
    return any(marker in evidence for marker in MEDIA_ARTICLE_MARKERS)


def concise_reference(definition: str) -> str:
    cleaned = " ".join(definition.split())
    sentences = tuple(
        match.group(0).strip()
        for match in re.finditer(r".+?(?:[.!?](?=\s|$)|$)", cleaned)
        if match.group(0).strip()
    )
    selected = " ".join(sentences[:2]) if sentences else cleaned
    truncated = len(sentences) > 2 or len(selected) < len(cleaned)
    if len(selected) > _REFERENCE_MAX_CHARS:
        truncated = True
    if not truncated:
        return selected
    budget = _REFERENCE_MAX_CHARS - len(_REFERENCE_MORE_NOTE)
    if len(selected) <= budget:
        return selected.rstrip() + _REFERENCE_MORE_NOTE
    prefix = selected[:budget].rstrip()
    sentence_end = max(prefix.rfind(mark) for mark in ".!?")
    if sentence_end >= max(40, budget // 3):
        prefix = prefix[: sentence_end + 1]
    elif (word_end := prefix.rfind(" ")) > 0:
        prefix = prefix[:word_end].rstrip(" ,;:—-") + "…"
    return prefix.rstrip() + _REFERENCE_MORE_NOTE


def _title_matches_query(query: str, title: str) -> bool:
    """Reject search hits where a common word only occurs inside an unrelated title."""

    query_key = normalize_key(query)
    title_key = normalize_key(title)
    variants = tuple(dict.fromkeys((query_key, *content_lemmas(query_key))))
    for variant in variants:
        if not variant:
            continue
        if title_key == variant or title_key.startswith(f"{variant} "):
            return True
    if len(query_key.split()) > 1:
        query_words = set(query_key.split())
        title_words = set(title_key.split())
        return len(query_words & title_words) / max(1, len(query_words)) >= 0.8
    return False
