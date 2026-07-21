import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.cognition.models import ActionIntent, ActionType
from debilgi.events.models import Event, EventType
from debilgi.knowledge.models import WikipediaResult
from debilgi.knowledge.policy import KnowledgeQueryPolicy
from debilgi.knowledge.repository import KnowledgeRepository
from debilgi.knowledge.service import KnowledgeAcquisitionService
from debilgi.language.russian_reference import RemoteLexeme
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "knowledge.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def event(number: int, content: str) -> Event:
    return Event(
        event_id=f"discord:message:{number}",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW + timedelta(seconds=number),
        payload={
            "message_id": number,
            "content": content,
            "author_is_bot": False,
            "bot_mentioned": True,
        },
        source="test",
    )


def intent(event_id: str) -> ActionIntent:
    return ActionIntent(
        intent_id=f"intent:{event_id}",
        guild_id=1,
        action_type=ActionType.SHORT_REPLY,
        target_id=100,
        channel_id=10,
        urgency=0.7,
        confidence=0.8,
        expected_effect={"hypothesis": "answer"},
        reasoning_trace=(),
        evidence=(event_id,),
        decision_context={
            "state_uncertainty": 0.9,
            "state_interest": 0.8,
            "state_unexpectedness": 0.7,
        },
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=10),
    )


class FakeCognitiveRepository:
    async def intent_for_event(self, event_id: str) -> ActionIntent:
        return intent(event_id)


class FakeWikipedia:
    language = "ru"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def search(self, query: str) -> WikipediaResult | None:
        self.calls.append(query)
        if query == "мифология":
            return WikipediaResult(
                page_id=456,
                title="Мифология",
                extract="Мифология — совокупность мифов определённой культуры.",
                canonical_url="https://ru.wikipedia.org/wiki/Мифология",
                wikidata_id="Q9134",
                revision_id=778,
                related_titles=("Легенда",),
            )
        if query == "неопределенность":
            return WikipediaResult(
                page_id=789,
                title="Неопределённость",
                extract="Неопределённость — состояние неполной информации.",
                canonical_url="https://ru.wikipedia.org/wiki/Неопределённость",
                wikidata_id=None,
                revision_id=779,
                related_titles=(),
            )
        if query == "обманщик":
            return WikipediaResult(
                page_id=1544091,
                title="Тартюф, или Обманщик",
                extract="Комедийная пьеса Мольера.",
                canonical_url="https://ru.wikipedia.org/wiki/Тартюф",
                wikidata_id=None,
                revision_id=780,
                related_titles=(),
            )
        return WikipediaResult(
            page_id=123,
            title="Дракон",
            extract="Дракон — собирательное название мифического существа.",
            canonical_url="https://ru.wikipedia.org/wiki/Дракон",
            wikidata_id="Q7559",
            revision_id=777,
            related_titles=("Мифология", "Змей"),
        )


class FakeDictionary:
    enabled = True

    def __init__(self, result: RemoteLexeme | None) -> None:
        self.result = result
        self.calls: list[str] = []

    async def resolve_for_context(self, word: str) -> RemoteLexeme | None:
        self.calls.append(word)
        return self.result


class FakeMediaWikipedia:
    language = "ru"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def search(self, query: str) -> WikipediaResult | None:
        return (await self.search_candidates(query))[0]

    async def search_candidates(
        self, query: str, *, limit: int = 5
    ) -> tuple[WikipediaResult, ...]:
        del limit
        self.calls.append(query)
        return (
            WikipediaResult(
                page_id=900,
                title="Умник",
                extract="Умник — американский телесериал 1987 года.",
                canonical_url="https://ru.wikipedia.org/wiki/Умник_(телесериал)",
                wikidata_id=None,
                revision_id=1,
                related_titles=(),
                categories=("Категория:Телесериалы США",),
            ),
        )


async def test_explicit_question_is_learned_and_reused_from_cache(database: Database) -> None:
    repository = KnowledgeRepository(database)
    source = FakeWikipedia()
    service = KnowledgeAcquisitionService(
        repository,
        FakeCognitiveRepository(),  # type: ignore[arg-type]
        KnowledgeQueryPolicy(enabled=True),
        source,
        reply_wait_seconds=1.0,
    )
    await service.start()
    try:
        first = event(1, "что такое дракон?")
        await service.observe(first)
        assert await service.wait_for_event(first.event_id) == "resolved"

        concepts = await repository.concepts_for_text("этот дракон умеет летать")
        assert len(concepts) == 1
        assert concepts[0].title == "Дракон"
        assert concepts[0].wikidata_id == "Q7559"
        assert concepts[0].revision_id == 777
        assert "мифического существа" in concepts[0].definition

        second = event(2, "объясни что такое дракон")
        await service.observe(second)
        assert await service.wait_for_event(second.event_id) == "cached"
        assert source.calls == ["дракон"]

        third = event(4, "что такое мифология")
        await service.observe(third)
        assert await service.wait_for_event(third.event_id) == "resolved"
        assert source.calls == ["дракон", "мифология"]

        stats = await repository.stats(1)
        assert stats.concepts == 2
        assert stats.resolved == 2
        assert stats.cached == 1
        assert stats.event_links == 3
        assert stats.context_associations == 6
        relation_rows = await database.fetchall(
            """
            SELECT object_title, object_concept_id FROM knowledge_relations
            WHERE subject_concept_id = 'wikipedia:ru:123' ORDER BY object_title
            """
        )
        relations = {str(row["object_title"]): row["object_concept_id"] for row in relation_rows}
        assert set(relations) == {"Мифология", "Змей"}
        assert relations["Мифология"] == "wikipedia:ru:456"
        association = await database.fetchone(
            """
            SELECT mention_count, explicit_query_count
            FROM knowledge_context_associations
            WHERE scope_type = 'channel' AND scope_id = 10
                AND concept_id = 'wikipedia:ru:123'
            """
        )
        assert association is not None
        assert int(association["mention_count"]) == 2
        assert int(association["explicit_query_count"]) == 2
    finally:
        await service.stop()
    assert source.started
    assert source.closed


async def test_lookup_followup_reuses_the_previous_explicit_subject(database: Database) -> None:
    repository = KnowledgeRepository(database)
    source = FakeWikipedia()
    service = KnowledgeAcquisitionService(
        repository,
        FakeCognitiveRepository(),  # type: ignore[arg-type]
        KnowledgeQueryPolicy(enabled=True),
        source,
        reply_wait_seconds=1.0,
    )
    await service.start()
    try:
        first = event(1, "что такое дракон?")
        await service.observe(first)
        assert await service.wait_for_event(first.event_id) == "resolved"
        assert (
            await repository.recent_explicit_query(
                1, 10, 100, before=event(2, "placeholder").utc_timestamp
            )
            == "дракон"
        )

        second = event(2, "узнай, умник")
        await service.observe(second)
        assert await service.wait_for_event(second.event_id) == "cached"

        facts, exact = await service.grounding_for_event(second, "узнай, умник")
        assert source.calls == ["дракон"]
        assert exact == facts[0]
        assert "мифического существа" in exact
        row = await database.fetchone(
            "SELECT query_text, reason FROM knowledge_queries WHERE event_id = ?",
            (second.event_id,),
        )
        assert row is not None
        assert (str(row["query_text"]), str(row["reason"])) == ("дракон", "explicit")
    finally:
        await service.stop()


async def test_autonomous_lookup_spends_event_balance(database: Database) -> None:
    repository = KnowledgeRepository(database)
    source = FakeWikipedia()
    service = KnowledgeAcquisitionService(
        repository,
        FakeCognitiveRepository(),  # type: ignore[arg-type]
        KnowledgeQueryPolicy(enabled=True),
        source,
        reply_wait_seconds=1.0,
    )
    await service.start()
    try:
        current = event(3, "неопределенность неожиданно изменилась")
        await service.observe(current)
        assert await service.wait_for_event(current.event_id) == "resolved"
        state = await repository.state(1, 10)
        assert state.queries_started == 1
        assert state.evidence_balance == pytest.approx(0.28)
        assert source.calls == ["неопределенность"]
    finally:
        await service.stop()


async def test_stalled_answer_can_investigate_and_persist_a_subject(database: Database) -> None:
    repository = KnowledgeRepository(database)
    source = FakeWikipedia()
    service = KnowledgeAcquisitionService(
        repository,
        FakeCognitiveRepository(),  # type: ignore[arg-type]
        KnowledgeQueryPolicy(enabled=True),
        source,
        reply_wait_seconds=1.0,
    )
    await service.start()
    try:
        current = event(5, "а что означает дракон?")

        facts = await service.investigate(current, "дракон")

        assert "мифического существа" in facts[0]
        assert source.calls == ["дракон"]
        learned = await repository.any_resolved_concept_for_event(current.event_id)
        assert learned is not None
        assert learned.title == "Дракон"
    finally:
        await service.stop()


async def test_search_rejects_a_title_that_only_contains_the_requested_word(
    database: Database,
) -> None:
    repository = KnowledgeRepository(database)
    source = FakeWikipedia()
    service = KnowledgeAcquisitionService(
        repository,
        FakeCognitiveRepository(),  # type: ignore[arg-type]
        KnowledgeQueryPolicy(enabled=True),
        source,
        reply_wait_seconds=1.0,
    )
    await service.start()
    try:
        current = event(6, "ты обманщик?")

        facts = await service.investigate(current, "обманщик")

        assert facts == ()
        assert await repository.any_resolved_concept_for_event(current.event_id) is None
        assert await service.context_for_text("ты обманщик?") == ()
    finally:
        await service.stop()


async def test_dictionary_definition_wins_over_conflicting_media_title(
    database: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="debilgi.knowledge_selection")
    repository = KnowledgeRepository(database)
    dictionary = FakeDictionary(
        RemoteLexeme(
            word="умник",
            part_of_speech="NOUN",
            meanings=("Разговорное обозначение умного или самоуверенного человека.",),
            source_url="https://ru.wiktionary.org/wiki/умник",
            style_labels=("разговорное",),
        )
    )
    source = FakeMediaWikipedia()
    service = KnowledgeAcquisitionService(
        repository,
        FakeCognitiveRepository(),  # type: ignore[arg-type]
        KnowledgeQueryPolicy(enabled=True),
        source,
        dictionary=dictionary,
        reply_wait_seconds=1.0,
    )
    await service.start()
    try:
        current = event(7, "что такое умник?")

        await service.observe(current)

        assert await service.wait_for_event(current.event_id) == "resolved"
        learned = await repository.resolved_concept_for_event(current.event_id)
        assert learned is not None
        assert learned.source == "ru_wiktionary"
        assert "самоуверенного человека" in learned.definition
        assert dictionary.calls == ["умник"]
        assert source.calls == []
        assert "manual_conflict_dictionary_priority" in caplog.text
    finally:
        await service.stop()


async def test_internal_study_persists_evidence_without_creating_exact_answer(
    database: Database,
) -> None:
    repository = KnowledgeRepository(database)
    dictionary = FakeDictionary(
        RemoteLexeme(
            word="робот",
            part_of_speech="NOUN",
            meanings=("Автоматическое устройство, выполняющее заданные действия.",),
            source_url="https://ru.wiktionary.org/wiki/робот",
        )
    )
    service = KnowledgeAcquisitionService(
        repository,
        FakeCognitiveRepository(),  # type: ignore[arg-type]
        KnowledgeQueryPolicy(enabled=True),
        FakeWikipedia(),
        dictionary=dictionary,
        reply_wait_seconds=1.0,
    )
    current = event(70, "Что полезнее: роботы или люди?")

    concepts = await service.internalize_for_reasoning(current, ("робот",))

    assert len(concepts) == 1
    assert concepts[0].source == "ru_wiktionary"
    assert "устройство" in concepts[0].definition
    assert await repository.concept_for_alias("робот") is not None
    assert await repository.resolved_concept_for_event(current.event_id) is None
    assert await repository.any_resolved_concept_for_event(current.event_id) is None
    query_count = await database.fetchone("SELECT COUNT(*) AS count FROM knowledge_queries")
    assert query_count is not None
    assert int(query_count["count"]) == 0


async def test_lowercase_common_word_rejects_media_after_dictionary_miss(
    database: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="debilgi.knowledge_selection")
    repository = KnowledgeRepository(database)
    dictionary = FakeDictionary(None)
    source = FakeMediaWikipedia()
    service = KnowledgeAcquisitionService(
        repository,
        FakeCognitiveRepository(),  # type: ignore[arg-type]
        KnowledgeQueryPolicy(enabled=True),
        source,
        dictionary=dictionary,
        reply_wait_seconds=1.0,
    )
    await service.start()
    try:
        current = event(8, "что такое умник?")

        await service.observe(current)

        assert await service.wait_for_event(current.event_id) == "no_result"
        assert await repository.resolved_concept_for_event(current.event_id) is None
        assert dictionary.calls == ["умник"]
        assert source.calls == ["умник"]
        assert "media_without_context" in caplog.text
    finally:
        await service.stop()


async def test_explicit_media_context_allows_series_and_skips_dictionary(
    database: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="debilgi.knowledge_selection")
    repository = KnowledgeRepository(database)
    dictionary = FakeDictionary(None)
    source = FakeMediaWikipedia()
    service = KnowledgeAcquisitionService(
        repository,
        FakeCognitiveRepository(),  # type: ignore[arg-type]
        KnowledgeQueryPolicy(enabled=True),
        source,
        dictionary=dictionary,
        reply_wait_seconds=1.0,
    )
    await service.start()
    try:
        current = event(9, "расскажи про сериал Умник")

        facts = await service.investigate(current, "умник")

        assert facts and "телесериал" in facts[0]
        learned = await repository.any_resolved_concept_for_event(current.event_id)
        assert learned is not None
        assert learned.source == "wikipedia"
        assert dictionary.calls == []
        assert source.calls == ["умник"]
        assert "explicit_media_context" in caplog.text
    finally:
        await service.stop()


async def test_explicit_typo_is_canonicalized_before_dictionary_lookup(
    database: Database,
) -> None:
    repository = KnowledgeRepository(database)
    dictionary = FakeDictionary(
        RemoteLexeme(
            word="тарелка",
            part_of_speech="NOUN",
            meanings=("Плоский столовый сосуд для еды.",),
            source_url="https://ru.wiktionary.org/wiki/тарелка",
        )
    )
    source = FakeMediaWikipedia()
    service = KnowledgeAcquisitionService(
        repository,
        FakeCognitiveRepository(),  # type: ignore[arg-type]
        KnowledgeQueryPolicy(enabled=True),
        source,
        dictionary=dictionary,
        reply_wait_seconds=1.0,
    )
    await service.start()
    try:
        current = event(10, "что такое таоелка?")

        await service.observe(current)

        assert await service.wait_for_event(current.event_id) == "resolved"
        learned = await repository.resolved_concept_for_event(current.event_id)
        assert learned is not None
        assert learned.title == "тарелка"
        assert dictionary.calls == ["тарелка"]
        assert source.calls == []
    finally:
        await service.stop()
