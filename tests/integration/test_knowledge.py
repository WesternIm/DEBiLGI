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
        return WikipediaResult(
            page_id=123,
            title="Дракон",
            extract="Дракон — собирательное название мифического существа.",
            canonical_url="https://ru.wikipedia.org/wiki/Дракон",
            wikidata_id="Q7559",
            revision_id=777,
            related_titles=("Мифология", "Змей"),
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
