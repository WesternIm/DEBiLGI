import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from debilgi.autonomy.repository import AutonomyRepository
from debilgi.background import BackgroundLearningService
from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.events.models import Event, EventType
from debilgi.knowledge.models import WikipediaResult
from debilgi.language.context_gate import ContextGateDeferred
from debilgi.language.lexicon import LexiconService
from debilgi.language.russian_reference import RemoteLexeme
from debilgi.language.semantic_assimilation import SemanticAssimilationService
from debilgi.language.semantic_consultant import (
    ConsultationResult,
    SemanticConsultationService,
)
from debilgi.memory.long_term import LongTermMemoryRepository
from debilgi.memory.short_term import ShortTermMemory
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 18, 15, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "background.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


class _Reference:
    enabled = True

    def __init__(self, results: dict[str, RemoteLexeme | None]) -> None:
        self.results = results
        self.lookups: list[str] = []

    async def refresh_word(self, word: str) -> RemoteLexeme | None:
        self.lookups.append(word)
        return self.results.get(word)


class _Wikipedia:
    async def search(self, query: str) -> None:
        return None


class _MediaWikipedia:
    def __init__(self) -> None:
        self.lookups: list[str] = []

    async def search(self, query: str) -> WikipediaResult:
        self.lookups.append(query)
        return WikipediaResult(
            page_id=900,
            title="Умник",
            extract="Умник — американский телесериал 1987 года.",
            canonical_url="https://ru.wikipedia.org/wiki/Умник_(телесериал)",
            wikidata_id=None,
            revision_id=None,
            related_titles=(),
            categories=("Категория:Телесериалы США",),
        )


class _Sender:
    def __init__(self) -> None:
        self.messages: list[tuple[int, int, str]] = []

    async def __call__(self, guild_id: int, channel_id: int, text: str) -> int:
        self.messages.append((guild_id, channel_id, text))
        return 9000 + len(self.messages)


class _DeferredSender:
    async def __call__(self, guild_id: int, channel_id: int, text: str) -> int:
        del guild_id, channel_id, text
        raise ContextGateDeferred("background_unrelated_to_active_focus")


class _SemanticConsultant:
    def __init__(self) -> None:
        self.requests = []

    async def consult(self, **kwargs) -> ConsultationResult:
        self.requests.append(kwargs)
        word = kwargs["request"].unknown_terms[0]
        return ConsultationResult.from_payload(
            {
                "interpretation": f"Запрос о значении слова {word}.",
                "intent": "lexical_study",
                "entities": [{"surface": word, "lemma": word, "type": "unknown"}],
                "relations": [],
                "ambiguities": [],
                "facts": [f"Предварительное значение слова {word}."],
                "response_text": None,
                "confidence": 0.72,
            },
            provider="fake-puter",
            model="fake-model",
        )


class _TemporarilyUnavailableConsultant:
    enabled = True
    last_failure_retryable = True

    async def consult(self, **kwargs) -> None:
        del kwargs
        return None


class _ConstructionProvider:
    def __init__(self) -> None:
        self.requests = []

    async def consult(self, request) -> ConsultationResult:
        self.requests.append(request)
        return ConsultationResult.from_payload(
            {
                "interpretation": "Полярный вопрос о наличии объекта.",
                "intent": "possession_question",
                "relations": [],
                "facts": ["Реплика выражает вопрос о наличии."],
                "constructions": [
                    {
                        "key": "object_first_possession",
                        "response_act": "relation_answer",
                        "relation": "possession",
                        "truth_state": "known",
                        "sequence": [
                            "polar_marker",
                            "comma",
                            "object:truth_case",
                            "existence_predicate",
                            "possession_preposition",
                            "subject:genitive",
                        ],
                    }
                ],
                "confidence": 0.86,
            },
            provider="fake-puter",
            model="fake-model",
        )

    async def close(self) -> None:
        return None


def _service(
    database: Database,
    reference: Any,
    *,
    sender: _Sender | None = None,
    attempts: int = 4,
    wikipedia: Any | None = None,
    wikipedia_enabled: bool = False,
    semantic_consultant: Any | None = None,
) -> BackgroundLearningService:
    return BackgroundLearningService(
        database,
        LexiconService(database),
        reference,
        wikipedia or _Wikipedia(),
        SelfDatasetRepository(database),
        LongTermMemoryRepository(database, ShortTermMemory()),
        AutonomyRepository(database),
        sender=sender,
        enabled=True,
        interval=timedelta(minutes=10),
        maximum_study_attempts=attempts,
        wikipedia_enabled=wikipedia_enabled,
        semantic_consultant=semantic_consultant,
        self_check_every_cycles=1,
        initiative_enabled=sender is not None,
        initiative_probability=1.0,
        clock=lambda: NOW,
    )


async def _unknown(database: Database, word: str, *, channel_id: int = 10) -> None:
    await database.execute(
        """INSERT INTO language_unknown_concepts (
               guild_id, channel_id, concept_key, unknown_kind, first_seen_at,
               last_seen_at, curiosity_score
           ) VALUES (1, ?, ?, 'unknown_form', ?, ?, 0.9)""",
        (channel_id, word, (NOW - timedelta(minutes=5)).isoformat(), NOW.isoformat()),
    )


async def _instruction_example(
    database: Database,
    *,
    event_id: str,
    response_id: int,
    prompt: str,
) -> None:
    await database.execute(
        """INSERT INTO learning_instruction_examples (
               example_id, guild_id, channel_id, user_id, source_event_id,
               response_message_id, instruction_text, input_text, answer_text,
               reasoning_json, topic_terms_json, prompt_hash, answer_hash,
               quality_score, status, created_at, updated_at
           ) VALUES (?, 1, 10, 20, ?, ?, ?, '', 'локальный ответ', '{}', '[]',
                     ?, ?, 0.8, 'candidate', ?, ?)""",
        (
            f"example:{event_id}",
            event_id,
            response_id,
            prompt,
            f"prompt:{event_id}",
            f"answer:{event_id}",
            NOW.isoformat(),
            NOW.isoformat(),
        ),
    )


async def test_cycle_learns_unknown_grows_graph_and_sends_one_initiative(
    database: Database,
) -> None:
    await _unknown(database, "стромба")
    old = NOW - timedelta(days=20)
    dataset = SelfDatasetRepository(database)
    await dataset.link_explanation(
        guild_id=1,
        concept="образ",
        explanation="образ — это мысленное представление",
        source_event_id="old:one",
        observed_at=old,
    )
    await dataset.link_explanation(
        guild_id=1,
        concept="образ",
        explanation="образ — это файл операционной системы на диске",
        source_event_id="old:two",
        observed_at=old,
    )
    reference = _Reference(
        {
            "стромба": RemoteLexeme(
                "стромба",
                "NOUN",
                ("разговорное название вымышленного предмета.",),
                "https://example.test/стромба",
            )
        }
    )
    sender = _Sender()
    service = _service(database, reference, sender=sender)

    stats = await service.run_once(NOW)

    assert stats.words_processed == 1
    assert stats.words_found == 1
    assert stats.words_missing == 0
    assert stats.decayed_edges > 0
    assert stats.conflicts_found >= 1
    assert stats.identity_checks == 2
    assert stats.initiative_queued == 1
    assert stats.initiative_sent == 1
    assert sender.messages and "стромба" in sender.messages[0][2]
    unknown = await database.fetchone(
        "SELECT resolved, study_status, study_attempts FROM language_unknown_concepts"
    )
    assert unknown is not None
    assert (int(unknown["resolved"]), str(unknown["study_status"])) == (1, "learned")
    assert int(unknown["study_attempts"]) == 1
    concept = await database.fetchone(
        """SELECT 1 FROM learning_graph_nodes
           WHERE guild_id = 1 AND node_type = 'concept' AND node_key = 'стромба'"""
    )
    assert concept is not None


async def test_missing_word_is_exhausted_after_bounded_attempts(database: Database) -> None:
    await _unknown(database, "неотыскиваемка")
    service = _service(database, _Reference({}), attempts=1)

    stats = await service.run_once(NOW)

    assert stats.words_missing == 1
    row = await database.fetchone(
        "SELECT study_status, study_attempts, resolved FROM language_unknown_concepts"
    )
    assert row is not None
    assert str(row["study_status"]) == "exhausted"
    assert int(row["study_attempts"]) == 1
    assert int(row["resolved"]) == 0


async def test_background_asks_consultant_only_after_reference_sources_fail(
    database: Database,
) -> None:
    await _unknown(database, "неотыскиваемка")
    consultant = _SemanticConsultant()
    service = _service(
        database,
        _Reference({}),
        attempts=1,
        semantic_consultant=consultant,
    )

    stats = await service.run_once(NOW)

    assert stats.words_missing == 1
    assert len(consultant.requests) == 1
    request = consultant.requests[0]["request"]
    assert request.mode == "lexical"
    assert request.unknown_terms == ("неотыскиваемка",)


async def test_idle_worker_induces_and_activates_reusable_construction(
    database: Database,
) -> None:
    await _instruction_example(
        database,
        event_id="discord:construction:one",
        response_id=8001,
        prompt="У тебя есть ключ?",
    )
    await _instruction_example(
        database,
        event_id="discord:construction:two",
        response_id=8002,
        prompt="Есть ли у тебя куртка?",
    )
    provider = _ConstructionProvider()
    assimilation = SemanticAssimilationService(
        database,
        SelfDatasetRepository(database),
    )
    consultant = SemanticConsultationService(
        database,
        provider,
        enabled=True,
        assimilator=assimilation,
    )
    service = _service(
        database,
        _Reference({}),
        semantic_consultant=consultant,
    )

    await service._study_dialogue_constructions("cycle:one", NOW)
    await service._study_dialogue_constructions("cycle:two", NOW + timedelta(minutes=10))

    statuses = await database.fetchall(
        "SELECT status FROM language_construction_study_queue ORDER BY source_event_id"
    )
    guidance = await assimilation.guidance_for_prompt(1, "У тебя есть носки?")
    assert [str(row["status"]) for row in statuses] == ["learned", "learned"]
    assert len(provider.requests) == 2
    assert all(request.mode == "utterance" for request in provider.requests)
    assert any(item.key == "object_first_possession" for item in guidance.constructions)


async def test_complete_question_can_queue_known_terms_for_silent_study(
    database: Database,
) -> None:
    lexicon = LexiconService(database)
    event = Event(
        event_id="discord:semantic-study",
        guild_id=1,
        channel_id=10,
        actor_id=20,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={"content": "Что полезнее: роботы или люди?", "bot_mentioned": True},
        source="discord",
    )
    queued = await lexicon.queue_semantic_study(
        event,
        ("робот",),
        reason="complete_comparison_knowledge_gap",
        priority=0.9,
    )
    reference = _Reference(
        {
            "робот": RemoteLexeme(
                "робот",
                "NOUN",
                ("автоматическое устройство, выполняющее заданные действия",),
                "https://example.test/робот",
            )
        }
    )

    stats = await _service(database, reference).run_once(NOW)

    assert queued == 1
    assert stats.words_processed == 1
    assert stats.words_found == 1
    row = await database.fetchone(
        """SELECT study_status, study_attempts, reason
           FROM language_semantic_study_queue WHERE concept_key = 'робот'"""
    )
    assert row is not None
    assert str(row["study_status"]) == "learned"
    assert int(row["study_attempts"]) == 1
    assert str(row["reason"]) == "complete_comparison_knowledge_gap"


async def test_provider_outage_does_not_consume_a_semantic_study_attempt(
    database: Database,
) -> None:
    lexicon = LexiconService(database)
    event = Event(
        event_id="discord:semantic-provider-outage",
        guild_id=1,
        channel_id=10,
        actor_id=20,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={"content": "Что значит заведомо?", "bot_mentioned": True},
        source="discord",
    )
    await lexicon.queue_semantic_study(
        event,
        ("заведомо",),
        reason="knowledge_gap",
        priority=0.9,
    )
    service = _service(
        database,
        _Reference({"заведомо": None}),
        semantic_consultant=_TemporarilyUnavailableConsultant(),
    )

    await service.run_once(NOW)

    row = await database.fetchone(
        """SELECT study_status, study_attempts FROM language_semantic_study_queue
           WHERE concept_key = 'заведомо'"""
    )
    assert row is not None
    assert str(row["study_status"]) == "pending"
    assert int(row["study_attempts"]) == 0


async def test_idle_worker_wakes_immediately_for_semantic_study(
    database: Database,
) -> None:
    reference = _Reference(
        {
            "робот": RemoteLexeme(
                "робот",
                "NOUN",
                ("автоматическое устройство, выполняющее заданные действия",),
                "https://example.test/робот",
            )
        }
    )
    service = _service(database, reference)
    service._lexicon.set_semantic_study_notifier(service.wake_for_study)
    await service.start()
    try:
        async def completed_cycles() -> int:
            row = await database.fetchone(
                "SELECT COUNT(*) AS count FROM background_cycle_runs WHERE status = 'completed'"
            )
            return int(row["count"]) if row is not None else 0

        async def wait_for_cycles(minimum: int) -> None:
            async def poll() -> None:
                while await completed_cycles() < minimum:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(poll(), timeout=2.0)

        await wait_for_cycles(1)
        current = Event(
            event_id="discord:idle-study",
            guild_id=1,
            channel_id=10,
            actor_id=20,
            target_id=None,
            event_type=EventType.MESSAGE_CREATED,
            timestamp=NOW,
            payload={"content": "роботы или люди?", "bot_mentioned": True},
            source="discord",
        )
        await service._lexicon.queue_semantic_study(
            current,
            ("робот",),
            reason="idle_semantic_gap",
            priority=0.9,
        )

        await wait_for_cycles(2)

        row = await database.fetchone(
            """SELECT study_status FROM language_semantic_study_queue
               WHERE concept_key = 'робот'"""
        )
        assert row is not None
        assert str(row["study_status"]) == "learned"
        assert reference.lookups == ["робот"]
    finally:
        await service.stop()


async def test_numeric_formula_tokens_do_not_enter_lexical_retrieval(database: Database) -> None:
    reference = _Reference({})
    service = _service(database, reference)

    result = await service._resolve_concept("707")

    assert result is None
    assert reference.lookups == []


async def test_background_learning_rejects_media_title_for_common_word(
    database: Database, caplog: pytest.LogCaptureFixture
) -> None:
    wikipedia = _MediaWikipedia()
    service = _service(
        database,
        _Reference({}),
        wikipedia=wikipedia,
        wikipedia_enabled=True,
    )

    result = await service._resolve_concept("умник")

    assert result is None
    assert wikipedia.lookups == ["умник"]


async def test_context_gate_defer_keeps_background_initiative_pending(
    database: Database,
) -> None:
    await _unknown(database, "стромба")
    reference = _Reference(
        {
            "стромба": RemoteLexeme(
                "стромба",
                "NOUN",
                ("вымышленный предмет",),
                "https://example.test/стромба",
            )
        }
    )
    service = _service(database, reference, sender=_DeferredSender())  # type: ignore[arg-type]

    stats = await service.run_once(NOW)

    row = await database.fetchone(
        "SELECT status, attempts, failure_reason FROM background_initiative_queue"
    )
    assert stats.initiative_queued == 1
    assert stats.initiative_sent == 0
    assert row is not None
    assert str(row["status"]) == "pending"
    assert int(row["attempts"]) == 0
    assert str(row["failure_reason"]).startswith("deferred:")


async def test_scheduler_start_is_non_blocking_and_stop_is_clean(database: Database) -> None:
    service = _service(database, _Reference({}))

    await service.start()
    assert service.running is True
    await service.stop()
    assert service.running is False
