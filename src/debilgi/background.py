"""Scheduled learning, graph maintenance, self-checks and restrained initiative."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from typing import Any
from uuid import uuid4

import aiohttp

from debilgi.autonomy.repository import AutonomyRepository
from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.knowledge.policy import media_article_reasons
from debilgi.knowledge.wikipedia import WikipediaClient
from debilgi.language.causal_reasoning import CausalReasoningRepository
from debilgi.language.character_markov import normalize_word
from debilgi.language.content import content_lemmas
from debilgi.language.context_gate import ContextGateDeferred
from debilgi.language.lexical_matching import fuzzy_match_allowed
from debilgi.language.lexicon import LexiconService
from debilgi.language.russian_reference import RemoteLexeme, RussianReferenceService
from debilgi.language.semantic_consultant import (
    ConsultationRequest,
    SemanticConsultationService,
)
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.language.usage_learning import (
    LexicalStudyMaterial,
    LexicalUsageLearningService,
)
from debilgi.memory.long_term import LongTermMemoryRepository
from debilgi.storage.database import Database

BackgroundSender = Callable[[int, int, str], Awaitable[int | None]]

logger = logging.getLogger("debilgi.background")
identity_logger = logging.getLogger("debilgi.identity_replies")

_IDENTITY_PROMPTS = ("кто ты", "как тебя зовут")
_FUNCTIONAL_RELATIONS = frozenset(
    {"являться", "значить", "означать", "звать", "находиться", "называться"}
)
_OPPOSITE_RELATIONS = {
    "синоним": "антоним",
    "антоним": "синоним",
    "подтверждает": "опровергает",
    "опровергает": "подтверждает",
    "is": "is_not",
    "is_not": "is",
}
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_LEXICAL_STUDY_RE = re.compile(r"^[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё-]{1,47}$")


@dataclass(frozen=True, slots=True)
class BackgroundCycleStats:
    cycle_id: str
    words_processed: int = 0
    words_found: int = 0
    words_missing: int = 0
    decayed_edges: int = 0
    memory_items_maintained: int = 0
    conflicts_found: int = 0
    identity_checks: int = 0
    initiative_queued: int = 0
    initiative_sent: int = 0


@dataclass(frozen=True, slots=True)
class LearnedConcept:
    guild_id: int
    channel_id: int
    word: str
    explanation: str
    source: str
    examples_found: int = 0
    usage_pattern: str = "uncertain"
    generated_example: str = ""


class BackgroundLearningService:
    """Run autonomous maintenance without manufacturing Discord input events."""

    def __init__(
        self,
        database: Database,
        lexicon: LexiconService,
        russian_reference: RussianReferenceService,
        wikipedia: WikipediaClient,
        dataset: SelfDatasetRepository,
        long_term_memory: LongTermMemoryRepository,
        autonomy: AutonomyRepository,
        *,
        sender: BackgroundSender | None = None,
        usage_learning: LexicalUsageLearningService | None = None,
        enabled: bool = False,
        interval: timedelta = timedelta(minutes=10),
        words_per_cycle: int = 5,
        maximum_study_attempts: int = 4,
        wikipedia_enabled: bool = False,
        semantic_consultant: SemanticConsultationService | None = None,
        causal_reasoning: CausalReasoningRepository | None = None,
        graph_decay_daily_rate: float = 0.015,
        self_check_every_cycles: int = 6,
        initiative_enabled: bool = False,
        initiative_probability: float = 0.25,
        initiative_cooldown: timedelta = timedelta(minutes=60),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if interval.total_seconds() <= 0 or initiative_cooldown.total_seconds() <= 0:
            raise ValueError("background timing values must be positive")
        if words_per_cycle <= 0 or maximum_study_attempts <= 0 or self_check_every_cycles <= 0:
            raise ValueError("background cycle limits must be positive")
        if not 0.0 <= graph_decay_daily_rate <= 1.0:
            raise ValueError("graph decay rate must be between zero and one")
        if not 0.0 <= initiative_probability <= 1.0:
            raise ValueError("initiative probability must be between zero and one")
        self._database = database
        self._lexicon = lexicon
        self._russian_reference = russian_reference
        self._wikipedia = wikipedia
        self._dataset = dataset
        self._long_term_memory = long_term_memory
        self._autonomy = autonomy
        self._sender = sender
        self._usage_learning = usage_learning or LexicalUsageLearningService(database, dataset)
        self.enabled = enabled
        self.interval = interval
        self.words_per_cycle = words_per_cycle
        self.maximum_study_attempts = maximum_study_attempts
        self.wikipedia_enabled = wikipedia_enabled
        self._semantic_consultant = semantic_consultant
        self._causal_reasoning = causal_reasoning
        self.graph_decay_daily_rate = graph_decay_daily_rate
        self.self_check_every_cycles = self_check_every_cycles
        self.initiative_enabled = initiative_enabled
        self.initiative_probability = initiative_probability
        self.initiative_cooldown = initiative_cooldown
        self._clock = clock or (lambda: datetime.now(UTC))
        self._task: asyncio.Task[None] | None = None
        self._run_lock = asyncio.Lock()
        self._study_wake = asyncio.Event()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if not self.enabled or self.running:
            return
        self._task = asyncio.create_task(self._run_loop(), name="debilgi-background-learning")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    def wake_for_study(self) -> None:
        """Interrupt the idle wait after new lexical/semantic work is queued."""

        self._study_wake.set()

    async def run_once(self, now: datetime | None = None) -> BackgroundCycleStats:
        async with self._run_lock:
            started = self._aware(now or self._clock())
            cycle_id = f"background:{uuid4()}"
            await self._database.execute(
                "INSERT INTO background_cycle_runs (cycle_id, started_at) VALUES (?, ?)",
                (cycle_id, started.isoformat()),
            )
            logger.info("cycle=%s started_at=%s status=started", cycle_id, started.isoformat())
            try:
                learned, processed, missing = await self._study_unknowns(cycle_id, started)
                semantic_learned, semantic_processed, semantic_missing = (
                    await self._study_semantic_targets(
                        cycle_id,
                        started,
                        limit=max(1, self.words_per_cycle - processed),
                    )
                )
                learned.extend(semantic_learned)
                processed += semantic_processed
                missing += semantic_missing
                await self._study_dialogue_constructions(cycle_id, started)
                if self._causal_reasoning is not None:
                    await self._causal_reasoning.rebuild_knowledge(
                        limit=max(100, self.words_per_cycle * 20)
                    )
                decayed = await self._decay_graph(started)
                memory_items = await self._long_term_memory.maintain(started)
                conflicts = await self._find_conflicts(started)
                cycle_number = await self._completed_cycle_count()
                identity_checks = 0
                if cycle_number % self.self_check_every_cycles == 0:
                    identity_checks = await self._identity_self_check(cycle_id, started)
                queued = await self._queue_initiatives(cycle_id, learned, started)
                sent = await self._dispatch_one_initiative(started)
                stats = BackgroundCycleStats(
                    cycle_id=cycle_id,
                    words_processed=processed,
                    words_found=len(learned),
                    words_missing=missing,
                    decayed_edges=decayed,
                    memory_items_maintained=memory_items,
                    conflicts_found=conflicts,
                    identity_checks=identity_checks,
                    initiative_queued=queued,
                    initiative_sent=sent,
                )
                await self._finish_cycle(stats, started, "completed")
                logger.info(
                    "cycle=%s status=completed processed=%d found=%d missing=%d "
                    "decayed_edges=%d memory_items=%d conflicts=%d identity_checks=%d "
                    "initiative_queued=%d initiative_sent=%d",
                    cycle_id,
                    processed,
                    len(learned),
                    missing,
                    decayed,
                    memory_items,
                    conflicts,
                    identity_checks,
                    queued,
                    sent,
                )
                return stats
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._database.execute(
                    """UPDATE background_cycle_runs SET finished_at = ?, status = 'failed',
                              error_type = ?, error_message = ? WHERE cycle_id = ?""",
                    (
                        self._aware(self._clock()).isoformat(),
                        type(exc).__name__,
                        str(exc)[:1000],
                        cycle_id,
                    ),
                )
                logger.exception("cycle=%s status=failed", cycle_id)
                raise

    async def _run_loop(self) -> None:
        while True:
            self._study_wake.clear()
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # The failed cycle is durable and logged; the scheduler stays alive.
                pass
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._study_wake.wait(),
                    timeout=self.interval.total_seconds(),
                )

    async def _study_unknowns(
        self, cycle_id: str, now: datetime
    ) -> tuple[list[LearnedConcept], int, int]:
        rows = await self._database.fetchall(
            """SELECT guild_id, channel_id, concept_key, study_attempts
               FROM language_unknown_concepts
               WHERE resolved = 0 AND study_status = 'pending' AND study_attempts < ?
               ORDER BY curiosity_score DESC, occurrence_count DESC, last_seen_at DESC
               LIMIT ?""",
            (self.maximum_study_attempts, self.words_per_cycle),
        )
        learned: list[LearnedConcept] = []
        missing = 0
        for row in rows:
            guild_id = int(row["guild_id"])
            channel_id = int(row["channel_id"])
            word = str(row["concept_key"])
            stored_attempts = int(row["study_attempts"])
            attempts = stored_attempts + 1
            try:
                result = await self._resolve_concept(word)
            except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                logger.warning(
                    "cycle=%s word=%r guild=%d channel=%d result=network_error error=%s",
                    cycle_id,
                    word,
                    guild_id,
                    channel_id,
                    type(exc).__name__,
                )
                result = None
            if result is None:
                consulted = await self._consult_unknown_concept(
                    word,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    reason="background_lexical_sources_exhausted",
                )
                missing += 1
                provider_unavailable = consulted is None
                if provider_unavailable:
                    attempts = stored_attempts
                status = "exhausted" if attempts >= self.maximum_study_attempts else "pending"
                await self._database.execute(
                    """UPDATE language_unknown_concepts
                       SET study_attempts = ?, last_study_attempt_at = ?, study_status = ?
                       WHERE guild_id = ? AND channel_id = ? AND concept_key = ?""",
                    (attempts, now.isoformat(), status, guild_id, channel_id, word),
                )
                logger.info(
                    "cycle=%s word=%r guild=%d channel=%d result=missing attempt=%d status=%s",
                    cycle_id,
                    word,
                    guild_id,
                    channel_id,
                    attempts,
                    status,
                )
                if consulted:
                    logger.info(
                        "cycle=%s word=%r result=provisional_consultation_saved",
                        cycle_id,
                        word,
                    )
                continue
            lesson = await self._usage_learning.learn(
                guild_id,
                result,
                observed_at=now,
            )
            await self._database.execute(
                """UPDATE language_unknown_concepts
                   SET resolved = 1, curiosity_score = 0.0, study_attempts = ?,
                       last_study_attempt_at = ?, study_status = 'learned'
                   WHERE guild_id = ? AND channel_id = ? AND concept_key = ?""",
                (attempts, now.isoformat(), guild_id, channel_id, word),
            )
            pattern_summary = _pattern_summary(lesson.pattern)
            learned.append(
                LearnedConcept(
                    guild_id,
                    channel_id,
                    word,
                    result.definition,
                    result.source,
                    len(result.examples),
                    pattern_summary,
                    lesson.generated_example,
                )
            )
            logger.info(
                "cycle=%s word=%r guild=%d channel=%d result=learned source=%s attempt=%d "
                "definition_found=%s examples_found=%d usage_pattern=%r generated_example=%r",
                cycle_id,
                word,
                guild_id,
                channel_id,
                result.source,
                attempts,
                bool(result.definition),
                len(result.examples),
                pattern_summary,
                lesson.generated_example,
            )
        return learned, len(rows), missing

    async def _resolve_concept(self, word: str) -> LexicalStudyMaterial | None:
        if _LEXICAL_STUDY_RE.fullmatch(word) is None:
            return None
        local = await self._lexicon.analyze_word(word)
        local_definition = (
            _concise(local.dictionary_meanings[0]) if local.dictionary_meanings else ""
        )
        remote: RemoteLexeme | None = None
        if self._russian_reference.enabled:
            remote = await self._russian_reference.refresh_word(word)
        if remote is not None and remote.meanings:
            return LexicalStudyMaterial(
                word=word,
                lemma=remote.word,
                definition=_concise(remote.meanings[0]),
                examples=remote.examples[:5],
                source="wiktionary",
            )
        if local_definition:
            return LexicalStudyMaterial(
                word=word,
                lemma=local.lemma or normalize_word(word),
                definition=local_definition,
                examples=(),
                source="local_dictionary",
            )
        if not self.wikipedia_enabled:
            return None
        wikipedia = await self._wikipedia.search(word)
        if wikipedia is None or not wikipedia.extract.strip():
            return None
        media_reasons = media_article_reasons(wikipedia)
        if media_reasons:
            logger.info(
                "Rejected background Wikipedia article %r for common word %r: "
                "media_without_context markers=%s categories=%s",
                wikipedia.title,
                word,
                media_reasons,
                wikipedia.categories,
            )
            return None
        normalized_word = normalize_word(word)
        normalized_title = normalize_word(wikipedia.title)
        if not fuzzy_match_allowed(normalized_word, normalized_title):
            logger.info(
                "Rejected background Wikipedia article %r for %r: protected_false_friend",
                wikipedia.title,
                word,
            )
            return None
        similarity = SequenceMatcher(None, normalized_word, normalized_title).ratio()
        if normalized_word not in normalized_title and similarity < 0.72:
            return None
        explanation = _concise(wikipedia.extract)
        if not explanation:
            return None
        await self._lexicon.import_dictionary_entry(
            lemma=word,
            part_of_speech=None,
            meanings=(explanation,),
            forms={word: {}},
            style_labels=("wikipedia_reference",),
            source="ru_wikipedia",
            confidence=0.66,
        )
        return LexicalStudyMaterial(
            word=word,
            lemma=local.lemma or normalize_word(word),
            definition=explanation,
            examples=(),
            source="wikipedia",
        )

    async def _study_semantic_targets(
        self,
        cycle_id: str,
        now: datetime,
        *,
        limit: int,
    ) -> tuple[list[LearnedConcept], int, int]:
        """Study understood word forms that still lack usable semantic properties."""

        rows = await self._database.fetchall(
            """SELECT guild_id, channel_id, concept_key, study_attempts, reason
               FROM language_semantic_study_queue
               WHERE study_status = 'pending' AND study_attempts < ?
               ORDER BY priority DESC, occurrence_count DESC, last_seen_at DESC
               LIMIT ?""",
            (self.maximum_study_attempts, max(1, limit)),
        )
        learned: list[LearnedConcept] = []
        missing = 0
        for row in rows:
            guild_id = int(row["guild_id"])
            channel_id = int(row["channel_id"])
            word = str(row["concept_key"])
            reason = str(row["reason"])
            stored_attempts = int(row["study_attempts"])
            attempts = stored_attempts + 1
            try:
                result = await self._resolve_concept(word)
            except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                logger.warning(
                    "cycle=%s semantic_word=%r result=network_error error=%s",
                    cycle_id,
                    word,
                    type(exc).__name__,
                )
                result = None
            if result is None:
                consulted = await self._consult_unknown_concept(
                    word,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    reason=f"background_semantic_sources_exhausted:{reason}",
                )
                missing += 1
                provider_unavailable = consulted is None
                if provider_unavailable:
                    attempts = stored_attempts
                status = "exhausted" if attempts >= self.maximum_study_attempts else "pending"
                await self._database.execute(
                    """UPDATE language_semantic_study_queue
                       SET study_attempts = ?, last_study_attempt_at = ?, study_status = ?
                       WHERE guild_id = ? AND channel_id = ? AND concept_key = ?""",
                    (attempts, now.isoformat(), status, guild_id, channel_id, word),
                )
                logger.info(
                    "cycle=%s semantic_word=%r reason=%s result=%s attempt=%d status=%s",
                    cycle_id,
                    word,
                    reason,
                    "provider_unavailable" if provider_unavailable else "missing",
                    attempts,
                    status,
                )
                if consulted:
                    logger.info(
                        "cycle=%s semantic_word=%r result=provisional_consultation_saved",
                        cycle_id,
                        word,
                    )
                continue
            lesson = await self._usage_learning.learn(
                guild_id,
                result,
                observed_at=now,
            )
            await self._database.execute(
                """UPDATE language_semantic_study_queue
                   SET study_attempts = ?, last_study_attempt_at = ?, study_status = 'learned'
                   WHERE guild_id = ? AND channel_id = ? AND concept_key = ?""",
                (attempts, now.isoformat(), guild_id, channel_id, word),
            )
            pattern_summary = _pattern_summary(lesson.pattern)
            learned.append(
                LearnedConcept(
                    guild_id,
                    channel_id,
                    word,
                    result.definition,
                    result.source,
                    len(result.examples),
                    pattern_summary,
                    lesson.generated_example,
                )
            )
            logger.info(
                "cycle=%s semantic_word=%r reason=%s result=learned source=%s "
                "attempt=%d concepts=%s usage_pattern=%r",
                cycle_id,
                word,
                reason,
                result.source,
                attempts,
                lesson.concepts,
                pattern_summary,
            )
        return learned, len(rows), missing

    async def _consult_unknown_concept(
        self,
        word: str,
        *,
        guild_id: int,
        channel_id: int,
        reason: str,
    ) -> bool | None:
        if self._semantic_consultant is None or _LEXICAL_STUDY_RE.fullmatch(word) is None:
            return False
        result = await self._semantic_consultant.consult(
            event_id=None,
            guild_id=guild_id,
            channel_id=channel_id,
            request=ConsultationRequest(
                prompt=f"Что означает слово «{word}» и как оно употребляется?",
                context_messages=(),
                known_terms=(),
                unknown_terms=(word,),
                local_relations=(),
                local_confidence=0.0,
                reason_codes=(reason,),
                mode="lexical",
            ),
        )
        if result is None and self._semantic_consultant.last_failure_retryable:
            return None
        return result is not None

    async def _study_dialogue_constructions(
        self,
        cycle_id: str,
        now: datetime,
    ) -> None:
        """Let the external teacher label dependency grammar while idle."""

        if self._semantic_consultant is None or not getattr(
            self._semantic_consultant,
            "enabled",
            True,
        ):
            return
        examples = await self._database.fetchall(
            """SELECT example.guild_id, example.channel_id, example.source_event_id,
                      example.prompt_hash, example.instruction_text, example.created_at
               FROM learning_instruction_examples AS example
               LEFT JOIN language_construction_study_queue AS queued
                 ON queued.guild_id = example.guild_id
                AND queued.source_event_id = example.source_event_id
               WHERE queued.source_event_id IS NULL
                 AND example.quality_score >= 0.10
                 AND length(example.instruction_text) BETWEEN 3 AND 400
               ORDER BY example.updated_at DESC LIMIT 40"""
        )
        for row in examples:
            await self._database.execute(
                """INSERT OR IGNORE INTO language_construction_study_queue (
                       guild_id, source_event_id, channel_id, prompt_hash, queued_at
                   ) VALUES (?, ?, ?, ?, ?)""",
                (
                    int(row["guild_id"]),
                    str(row["source_event_id"]),
                    int(row["channel_id"]) if row["channel_id"] is not None else None,
                    str(row["prompt_hash"]),
                    str(row["created_at"]),
                ),
            )

        rows = await self._database.fetchall(
            """SELECT queue.guild_id, queue.channel_id, queue.source_event_id,
                      queue.study_attempts, example.instruction_text
               FROM language_construction_study_queue AS queue
               JOIN learning_instruction_examples AS example
                 ON example.guild_id = queue.guild_id
                AND example.source_event_id = queue.source_event_id
               WHERE queue.status = 'pending' AND queue.study_attempts < ?
               ORDER BY queue.queued_at ASC LIMIT 1""",
            (self.maximum_study_attempts,),
        )
        if not rows:
            return
        row = rows[0]
        prompt = str(row["instruction_text"])
        structure = analyze_structured_utterance(prompt)
        clause = max(structure.clauses, key=lambda item: item.confidence, default=None)
        local_confidence = clause.confidence if clause is not None else structure.confidence
        stored_attempts = int(row["study_attempts"])
        attempts = stored_attempts + 1
        result = await self._semantic_consultant.consult(
            event_id=str(row["source_event_id"]),
            guild_id=int(row["guild_id"]),
            channel_id=(
                int(row["channel_id"]) if row["channel_id"] is not None else None
            ),
            request=ConsultationRequest(
                prompt=prompt,
                context_messages=(),
                known_terms=tuple(dict.fromkeys(content_lemmas(prompt)))[:24],
                unknown_terms=(),
                local_relations=(),
                local_confidence=local_confidence,
                reason_codes=("background_dependency_induction",),
                mode="utterance",
            ),
        )
        provider_unavailable = (
            result is None and self._semantic_consultant.last_failure_retryable
        )
        if provider_unavailable:
            attempts = stored_attempts
        status = (
            "learned"
            if result is not None
            else "exhausted"
            if attempts >= self.maximum_study_attempts
            else "pending"
        )
        await self._database.execute(
            """UPDATE language_construction_study_queue
               SET study_attempts = ?, status = ?, last_attempt_at = ?
               WHERE guild_id = ? AND source_event_id = ?""",
            (
                attempts,
                status,
                now.isoformat(),
                int(row["guild_id"]),
                str(row["source_event_id"]),
            ),
        )
        logger.info(
            "cycle=%s syntax_event=%s result=%s attempts=%d role_count=%d",
            cycle_id,
            row["source_event_id"],
            "provider_unavailable" if provider_unavailable else status,
            attempts,
            len(result.sentence_pattern) if result is not None else 0,
        )

    async def _decay_graph(self, now: datetime) -> int:
        rows = await self._database.fetchall(
            """SELECT edge_id, confidence, last_seen_at,
                      COALESCE(last_decay_at, last_seen_at) AS decay_from
               FROM learning_graph_edges"""
        )
        changed: list[tuple[float, str, str]] = []
        for row in rows:
            decay_from = self._aware(datetime.fromisoformat(str(row["decay_from"])))
            days = max(0.0, (now - decay_from).total_seconds() / 86400.0)
            if days <= 0:
                continue
            last_seen = self._aware(datetime.fromisoformat(str(row["last_seen_at"])))
            idle_days = max(0.0, (now - last_seen).total_seconds() / 86400.0)
            if idle_days < 1.0:
                continue
            confidence = float(row["confidence"]) * math.exp(
                -self.graph_decay_daily_rate * days
            )
            changed.append((max(0.03, confidence), now.isoformat(), str(row["edge_id"])))
        if changed:
            async with self._database.transaction() as connection:
                await connection.executemany(
                    """UPDATE learning_graph_edges
                       SET confidence = ?, last_decay_at = ? WHERE edge_id = ?""",
                    changed,
                )
        return len(changed)

    async def _find_conflicts(self, now: datetime) -> int:
        rows = await self._database.fetchall(
            """SELECT edge.guild_id, edge.source_node_id, edge.relation,
                      edge.target_node_id, edge.properties_json,
                      target.label AS target_label
               FROM learning_graph_edges AS edge
               JOIN learning_graph_nodes AS target ON target.node_id = edge.target_node_id
               WHERE edge.confidence >= 0.35
               ORDER BY edge.guild_id, edge.source_node_id, edge.relation, edge.target_node_id"""
        )
        grouped: dict[tuple[int, str], list[Any]] = {}
        for row in rows:
            grouped.setdefault((int(row["guild_id"]), str(row["source_node_id"])), []).append(row)
        conflicts: list[tuple[Any, Any, str]] = []
        for edges in grouped.values():
            for index, left in enumerate(edges):
                for right in edges[index + 1 :]:
                    reason = _conflict_reason(left, right)
                    if reason is not None:
                        conflicts.append((left, right, reason))
        for left, right, reason in conflicts:
            values = _ordered_conflict(left, right)
            fingerprint = ":".join(str(item) for item in (*values, reason))
            conflict_id = f"conflict:{hashlib.sha256(fingerprint.encode()).hexdigest()[:28]}"
            await self._database.execute(
                """INSERT INTO background_graph_conflicts (
                       conflict_id, guild_id, source_node_id, relation_a, target_a_node_id,
                       relation_b, target_b_node_id, reason, first_detected_at, last_detected_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (
                       guild_id, source_node_id, relation_a, target_a_node_id,
                       relation_b, target_b_node_id, reason
                   ) DO UPDATE SET last_detected_at = excluded.last_detected_at,
                                   occurrence_count = occurrence_count + 1""",
                (conflict_id, *values, reason, now.isoformat(), now.isoformat()),
            )
            logger.info(
                "conflict=%s guild=%s source=%s relation_a=%s target_a=%s "
                "relation_b=%s target_b=%s reason=%s status=review",
                conflict_id,
                *values,
                reason,
            )
        return len(conflicts)

    async def _identity_self_check(self, cycle_id: str, now: datetime) -> int:
        guild_rows = await self._database.fetchall(
            """SELECT DISTINCT guild_id FROM events
               UNION SELECT DISTINCT guild_id FROM learning_graph_nodes"""
        )
        count = 0
        for guild_row in guild_rows:
            guild_id = int(guild_row["guild_id"])
            for prompt in _IDENTITY_PROMPTS:
                snapshot, source = await self._identity_snapshot(guild_id, prompt)
                await self._database.execute(
                    """INSERT INTO background_identity_checks (
                           check_id, cycle_id, guild_id, prompt, response_snapshot,
                           evidence_source, observed_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"identity-check:{uuid4()}",
                        cycle_id,
                        guild_id,
                        prompt,
                        snapshot,
                        source,
                        now.isoformat(),
                    ),
                )
                identity_logger.info(
                    "background=true cycle=%s guild=%d prompt=%r response=%r source=%s",
                    cycle_id,
                    guild_id,
                    prompt,
                    snapshot,
                    source,
                )
                count += 1
        return count

    async def _identity_snapshot(self, guild_id: int, prompt: str) -> tuple[str, str]:
        rows = await self._database.fetchall(
            """SELECT instruction_text, answer_text, status, quality_score, updated_at
               FROM learning_instruction_examples
               WHERE guild_id = ? AND status <> 'rejected'
               ORDER BY CASE status WHEN 'accepted' THEN 0 ELSE 1 END,
                        quality_score DESC, updated_at DESC LIMIT 500""",
            (guild_id,),
        )
        normalized_prompt = _normalize_phrase(prompt)
        matches = [
            row
            for row in rows
            if SequenceMatcher(
                None, normalized_prompt, _normalize_phrase(str(row["instruction_text"]))
            ).ratio()
            >= 0.72
        ]
        if matches:
            return _concise(str(matches[0]["answer_text"])), "instruction_dataset"
        nodes = await self._database.fetchall(
            """SELECT label FROM learning_graph_nodes
               WHERE guild_id = ? AND node_type IN ('concept', 'answer', 'explanation')
                 AND (node_key LIKE '%debilgi%' OR node_key LIKE '%деби%'
                      OR node_key LIKE '%имя%' OR node_key LIKE '%себя%')
               ORDER BY confidence DESC, occurrence_count DESC LIMIT 3""",
            (guild_id,),
        )
        if nodes:
            return _concise("; ".join(str(row["label"]) for row in nodes)), "concept_graph"
        return "В текущем графе пока нет устойчивого ответа.", "insufficient_graph_evidence"

    async def _queue_initiatives(
        self, cycle_id: str, learned: list[LearnedConcept], now: datetime
    ) -> int:
        if not self.initiative_enabled:
            return 0
        queued = 0
        for item in learned:
            if item.channel_id <= 0:
                continue
            probability_draw = _stable_fraction(
                f"{cycle_id}:{item.guild_id}:{item.word}"
            )
            if probability_draw > self.initiative_probability:
                continue
            text = f"Кстати, разобрался, что значит «{item.word}»: {item.explanation}"
            await self._database.execute(
                """INSERT INTO background_initiative_queue (
                       initiative_id, cycle_id, guild_id, channel_id, concept_key,
                       message_text, queued_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"background-initiative:{uuid4()}",
                    cycle_id,
                    item.guild_id,
                    item.channel_id,
                    item.word,
                    text[:1200],
                    now.isoformat(),
                ),
            )
            queued += 1
        return queued

    async def _dispatch_one_initiative(self, now: datetime) -> int:
        if not self.initiative_enabled or self._sender is None:
            return 0
        rows = await self._database.fetchall(
            """SELECT * FROM background_initiative_queue
               WHERE status = 'pending' ORDER BY queued_at ASC LIMIT 25"""
        )
        for row in rows:
            guild_id = int(row["guild_id"])
            channel_id = int(row["channel_id"])
            recent = await self._database.fetchone(
                """SELECT sent_at FROM background_initiative_queue
                   WHERE guild_id = ? AND channel_id = ? AND status = 'sent'
                   ORDER BY sent_at DESC LIMIT 1""",
                (guild_id, channel_id),
            )
            if recent is not None:
                last_sent = self._aware(datetime.fromisoformat(str(recent["sent_at"])))
                if now - last_sent < self.initiative_cooldown:
                    continue
            if not await self._autonomy.background_initiative_allowed(guild_id, channel_id):
                continue
            initiative_id = str(row["initiative_id"])
            try:
                message_id = await self._sender(guild_id, channel_id, str(row["message_text"]))
            except ContextGateDeferred as exc:
                await self._database.execute(
                    """UPDATE background_initiative_queue
                       SET attempted_at = ?, failure_reason = ? WHERE initiative_id = ?""",
                    (now.isoformat(), f"deferred: {exc}"[:500], initiative_id),
                )
                logger.info(
                    "initiative=%s guild=%d channel=%d status=deferred reason=%s",
                    initiative_id,
                    guild_id,
                    channel_id,
                    exc,
                )
                return 0
            except Exception as exc:
                await self._database.execute(
                    """UPDATE background_initiative_queue
                       SET status = 'failed', attempts = attempts + 1, attempted_at = ?,
                           failure_reason = ? WHERE initiative_id = ?""",
                    (now.isoformat(), f"{type(exc).__name__}: {exc}"[:500], initiative_id),
                )
                logger.exception("initiative=%s status=failed", initiative_id)
                continue
            if message_id is None:
                await self._database.execute(
                    """UPDATE background_initiative_queue
                       SET status = 'failed', attempts = attempts + 1, attempted_at = ?,
                           failure_reason = 'channel_unavailable' WHERE initiative_id = ?""",
                    (now.isoformat(), initiative_id),
                )
                continue
            await self._database.execute(
                """UPDATE background_initiative_queue
                   SET status = 'sent', attempts = attempts + 1, attempted_at = ?, sent_at = ?,
                       response_message_id = ? WHERE initiative_id = ?""",
                (now.isoformat(), now.isoformat(), message_id, initiative_id),
            )
            await self._autonomy.record_background_initiative(
                guild_id, channel_id, created_at=now
            )
            logger.info(
                "initiative=%s guild=%d channel=%d message_id=%d status=sent",
                initiative_id,
                guild_id,
                channel_id,
                message_id,
            )
            return 1
        return 0

    async def _completed_cycle_count(self) -> int:
        row = await self._database.fetchone(
            "SELECT COUNT(*) AS count FROM background_cycle_runs WHERE status = 'completed'"
        )
        return int(row["count"]) if row else 0

    async def _finish_cycle(
        self, stats: BackgroundCycleStats, started: datetime, status: str
    ) -> None:
        finished = self._aware(self._clock())
        await self._database.execute(
            """UPDATE background_cycle_runs SET
                   finished_at = ?, words_processed = ?, words_found = ?, words_missing = ?,
                   decayed_edges = ?, memory_items_maintained = ?, conflicts_found = ?,
                   identity_checks = ?, initiative_queued = ?, initiative_sent = ?, status = ?
               WHERE cycle_id = ?""",
            (
                max(finished, started).isoformat(),
                stats.words_processed,
                stats.words_found,
                stats.words_missing,
                stats.decayed_edges,
                stats.memory_items_maintained,
                stats.conflicts_found,
                stats.identity_checks,
                stats.initiative_queued,
                stats.initiative_sent,
                status,
                stats.cycle_id,
            ),
        )

    @staticmethod
    def _aware(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def _conflict_reason(left: Any, right: Any) -> str | None:
    left_relation = str(left["relation"])
    right_relation = str(right["relation"])
    left_target = str(left["target_node_id"])
    right_target = str(right["target_node_id"])
    if _OPPOSITE_RELATIONS.get(left_relation) == right_relation and left_target == right_target:
        return "opposite_relations_same_target"
    if (
        left_relation == right_relation
        and left_relation in _FUNCTIONAL_RELATIONS
        and left_target != right_target
    ):
        return "multiple_targets_for_functional_relation"
    if left_relation == right_relation == "explained_by" and left_target != right_target:
        similarity = SequenceMatcher(
            None,
            _normalize_phrase(str(left["target_label"])),
            _normalize_phrase(str(right["target_label"])),
        ).ratio()
        # Explanations that share only a generic opening ("X — это ...")
        # still score around 0.5 character-wise, so keep this review threshold
        # deliberately conservative but above that boilerplate overlap.
        if similarity < 0.58:
            return "divergent_explanations"
    if left_relation == right_relation and left_target == right_target:
        try:
            left_negated = bool(json.loads(str(left["properties_json"])).get("negated"))
            right_negated = bool(json.loads(str(right["properties_json"])).get("negated"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if left_negated != right_negated:
            return "negated_and_positive_evidence"
    return None


def _ordered_conflict(left: Any, right: Any) -> tuple[int, str, str, str, str, str]:
    guild_id = int(left["guild_id"])
    source = str(left["source_node_id"])
    first = (str(left["relation"]), str(left["target_node_id"]))
    second = (str(right["relation"]), str(right["target_node_id"]))
    ordered = (*first, *second) if first <= second else (*second, *first)
    relation_a, target_a, relation_b, target_b = ordered
    return guild_id, source, relation_a, target_a, relation_b, target_b


def _concise(text: str, *, maximum: int = 500) -> str:
    cleaned = " ".join(text.split())
    sentences = _SENTENCE_BOUNDARY_RE.split(cleaned)
    result = " ".join(sentences[:2]).strip()
    if len(result) <= maximum:
        return result
    boundary = max(
        result.rfind(".", 0, maximum),
        result.rfind("!", 0, maximum),
        result.rfind("?", 0, maximum),
    )
    return result[: boundary + 1 if boundary >= 40 else maximum].rstrip(" ,;:—-")


def _normalize_phrase(text: str) -> str:
    return " ".join(normalize_word(part) for part in text.split() if normalize_word(part))


def _pattern_summary(pattern: Any) -> str:
    if pattern.status != "stable":
        return "сочетаемость неопределена"
    parts = tuple(
        value
        for value in (
            pattern.preferred_case,
            pattern.preferred_number,
            pattern.preferred_role,
            *(f"neighbor:{item}" for item in pattern.neighbors),
        )
        if value
    )
    return ", ".join(parts) or "сочетаемость неопределена"


def _stable_fraction(value: str) -> float:
    integer = int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")
    return integer / ((1 << 64) - 1)
