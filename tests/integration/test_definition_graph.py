from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.events.normalizer import normalize_message_created
from debilgi.events.repository import EventRepository
from debilgi.knowledge.models import WikipediaResult
from debilgi.knowledge.repository import KnowledgeRepository
from debilgi.language.lexicon import LexiconService, UnknownKind
from debilgi.language.sentence_understanding import understand_utterance
from debilgi.storage.database import Database


@pytest.fixture
async def db(tmp_path):
    database = Database(
        tmp_path / "definitions.sqlite3", Path(__file__).resolve().parents[2] / "migrations"
    )
    await database.initialize()
    try:
        yield database
    finally:
        await database.close()


async def save_concept(db, definition="Ручка — письменный инструмент для письма.", title="Ручка"):
    event = normalize_message_created(
        message_id=1,
        guild_id=1,
        channel_id=10,
        author_id=100,
        created_at=datetime.now(UTC),
        content="ручка",
        author_is_bot=False,
    )
    await EventRepository(db).append(event)
    result = WikipediaResult(
        1, title, definition, "https://ru.wikipedia.org/wiki/Ручка", None, 2, ("Чернила",)
    )
    repo = KnowledgeRepository(db)
    return await repo.store_internalized_concept(
        event, query="ручка", result=result, source="wikipedia", language="ru"
    )


async def test_acquired_graph_reaches_word_understanding_and_survives_restart(db):
    concept = await save_concept(db)
    edges = await db.fetchall("SELECT * FROM knowledge_definition_relations")
    assert {(e["predicate"], e["target"]) for e in edges} == {
        ("is_a", "письменный инструмент"),
        ("used_for", "письмо"),
    }
    assert all(
        e["concept_id"] == concept.concept_id and e["evidence_text"] == concept.definition
        for e in edges
    )
    await db.close()
    await db.initialize()
    analysis = await LexiconService(db).analyze_word("ручкой", context="пишу ручкой")
    assert analysis.has_usable_meaning
    assert ("used_for", "письмо") in analysis.semantic_relations
    understanding = understand_utterance("Для чего ручка?", (analysis,))
    assert any(
        r.source == "ручка"
        and r.relation == "used_for"
        and r.target == "письмо"
        and r.kind == "lexical_definition"
        for r in understanding.relations
    )
    assert all(r.target != "Чернила" for r in understanding.relations)


async def test_definition_update_replaces_stale_edges_and_cache_upgrade_is_idempotent(db):
    await save_concept(db)
    concept = await save_concept(db, "Ручка — часть двери.")
    repo = KnowledgeRepository(db)
    await repo.ensure_definition_relations(concept)
    await repo.ensure_definition_relations(concept)
    edges = await db.fetchall("SELECT predicate,target FROM knowledge_definition_relations")
    assert [(r["predicate"], r["target"]) for r in edges] == [("part_of", "дверь")]
    await db.execute("DELETE FROM knowledge_definition_relations")
    assert await repo.concept_for_alias("ручка") is not None
    assert len(await db.fetchall("SELECT * FROM knowledge_definition_relations")) == 1


async def test_poisoned_partial_alias_cannot_become_a_words_dictionary_meaning(db):
    await save_concept(db, "Надежда Ручка — певица.", "Ручка, Надежда Александровна")
    analysis = await LexiconService(db).analyze_word("ручка")
    assert not any("певица" in meaning for meaning in analysis.dictionary_meanings)
    assert not analysis.semantic_relations


async def test_polysemy_selects_contextual_definition_and_keeps_other_sense_out(db):
    lexicon = LexiconService(db)
    await lexicon.import_dictionary_entry(
        lemma="ручка",
        part_of_speech="NOUN",
        meanings=(
            "инструмент для письма",
            "часть двери",
        ),
    )
    ambiguous = await lexicon.analyze_word("ручка", context="Что такое ручка?")
    assert ambiguous.unknown_kind == UnknownKind.AMBIGUOUS_CONTEXT
    assert not ambiguous.semantic_relations
    writing = await lexicon.analyze_word("ручка", context="ручка для письма")
    door = await lexicon.analyze_word("ручка", context="ручка двери")
    assert ("used_for", "письмо") in writing.semantic_relations
    assert door.semantic_relations == (("part_of", "дверь"),)
    assert "дверь" not in {target for _, target in writing.semantic_relations}
