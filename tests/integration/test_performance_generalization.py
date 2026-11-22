"""Regression proofs for 4.13b: maintenance, provenance and parser projections."""

import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language import dialogue_learning as dialogue
from debilgi.language.causal_reasoning import CausalReasoningRepository
from debilgi.language.lexicon import LexiconService
from debilgi.language.proposition_memory import PropositionMemory, count_meaning_evidence
from debilgi.language.tokenizer import normalize_chat_token
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 16, 12, tzinfo=UTC)


def event(n, text, actor=10):
    return Event(str(n), 1, 2, actor, None, EventType.MESSAGE_CREATED,
                 NOW + timedelta(seconds=n), {"content": text}, "discord")


@pytest.fixture
async def database(tmp_path):
    db = Database(tmp_path / "performance.sqlite3", Path("migrations"))
    await db.initialize()
    yield db
    await db.close()


async def observe(db, n, text, actor=10):
    item = event(n, text, actor)
    await EventRepository(db).append(item)
    await PropositionMemory.for_meaning(db).observe(item)
    return item


async def test_import_rekeys_existing_human_evidence(database):
    await observe(database, 1, "Мишка ест рыбу.")
    await observe(database, 2, "Медведь ест рыбу.", 11)
    lexicon = LexiconService(database)
    await lexicon.import_sense(lemma="медведь", definition="Лесное животное",
                               context_terms=("лес",), synonyms=("мишка",), guild_id=1)
    rows = await database.fetchall("SELECT DISTINCT meaning_key FROM meaning_evidence")
    assert len(rows) == 1
    memory = PropositionMemory.for_meaning(database)
    assert (await memory.support(1, rows[0][0], public=True)).supports == 2
    result = await memory.retrieve_claims(event(100, "Мишка ест рыбу?", 99))
    assert result.status == "Known"
    assert await database.fetchone("SELECT 1 FROM meaning_positions WHERE actor_id=11")


async def test_global_import_does_not_confirm_server_synonyms_or_facts(database):
    await observe(database, 1, "Мишка ест рыбу.")
    await observe(database, 2, "Медведь ест рыбу.", 11)
    lexicon = LexiconService(database)
    await lexicon.import_sense(lemma="медведь", definition="Лесное животное",
                               context_terms=("лес",), synonyms=("мишка",), guild_id=0)
    assert not await lexicon.canonical_lemmas(1)
    memory = PropositionMemory.for_meaning(database)
    result = await memory.retrieve_claims(event(100, "Мишка ест рыбу?", 99))
    assert result.status == "Unknown"
    # The same dictionary class can subsequently be confirmed by server humans.
    for n, actor in ((3, 10), (4, 10), (5, 11)):
        item = event(n, "мишка это медведь", actor)
        await EventRepository(database).append(item)
        await lexicon.observe_synonym(item, "мишка", "медведь")
    assert await lexicon.canonical_lemmas(1)


async def test_candidates_are_deferred_ranked_and_repeatable(database):
    for governor, left, right in (("видеть", "медведь", "мишка"),
                                  ("кормить", "медведь", "мишка"),
                                  ("видеть", "кошка", "кот")):
        for word in (right, left):
            await database.execute(
                "INSERT INTO language_generation_word_relations VALUES "
                "(1,2,?,?,'direct_object','noun','accusative',3)", (governor, word))
    lexicon = LexiconService(database)
    item = event(1, "Вижу медведя.")
    await EventRepository(database).append(item)
    await lexicon.observe_synonym_text(item, item.payload["content"])
    assert not await database.fetchone("SELECT 1 FROM continuous_cognition_hypotheses")
    first = await lexicon.discover_synonym_candidates()
    assert first[0] == (1, "медведь", "мишка", 2)
    await lexicon.observe_synonym_text(item, item.payload["content"])
    assert first == await lexicon.discover_synonym_candidates()
    await lexicon.stop_synonym_candidates()
    assert not await lexicon.canonical_lemmas(1)
    assert not await database.fetchone(
        "SELECT 1 FROM continuous_cognition_hypotheses WHERE positive_evidence>0")


async def test_causal_observer_never_appends_and_updates_people_incrementally(database):
    repository = CausalReasoningRepository(database)
    statements = ((1, "Дождь вызывает наводнение.", 10),
                  (2, "Дождь вызывает наводнение.", 10),
                  (3, "Дождь вызывает наводнение.", 11),
                  (4, "Дождь не вызывает наводнение.", 10))
    statements_sql = []
    await database.connection.set_trace_callback(statements_sql.append)
    for n, text, actor in statements:
        await repository.observe(event(n, text, actor))
    assert not await database.fetchone("SELECT 1 FROM events")
    row = await database.fetchone("SELECT * FROM language_causal_links")
    assert (row["positive_evidence"], row["negative_evidence"]) == (1, 1)
    # An old observation cannot overwrite a person's latest stance.
    await repository.observe(event(0, statements[0][1], 10))
    await repository.observe(event(4, statements[-1][1], 10))
    row = await database.fetchone("SELECT * FROM language_causal_links")
    assert (row["positive_evidence"], row["negative_evidence"]) == (1, 1)
    assert not any("JOIN events e ON e.event_id=x.source_event_id" in sql
                   for sql in statements_sql)


async def test_positive_event_counts_migrate_to_people_without_data_loss(tmp_path):
    old_migrations = tmp_path / "old_migrations"
    old_migrations.mkdir()
    for path in Path("migrations").glob("*.sql"):
        if int(path.name.split("_", 1)[0]) < 64:
            shutil.copyfile(path, old_migrations / path.name)
    path = tmp_path / "old.sqlite3"
    db = Database(path, old_migrations)
    await db.initialize()
    stamp = NOW.isoformat()
    await db.execute(
        "INSERT INTO continuous_cognition_hypotheses "
        "(hypothesis_id,guild_id,signature,predicate_key,epistemic_status,confidence,"
        "positive_evidence,first_seen_at,last_reviewed_at) "
        "VALUES ('old',1,'response:old','есть','hypothesis',0.7,7,?,?)", (stamp, stamp))
    for n in range(1, 8):
        item = event(n, "да", 10 if n < 7 else 11)
        await EventRepository(db).append(item)
        await db.execute("INSERT INTO continuous_cognition_evidence VALUES "
                         "(?,'old',?,'supports',0.7,'{}',?)",
                         (str(n), item.event_id, item.utc_timestamp.isoformat()))
    await db.close()
    db = Database(path, Path("migrations"))
    await db.initialize()
    try:
        row = await db.fetchone("SELECT * FROM continuous_cognition_hypotheses")
        assert (row["positive_evidence"], row["legacy_positive_events"]) == (2, 7)
        assert (await db.fetchone("SELECT COUNT(*) FROM continuous_cognition_evidence"))[0] == 7
    finally:
        await db.close()


async def test_corpus_does_not_search_candidates_or_cast_votes(database):
    statements = []
    await database.connection.set_trace_callback(statements.append)
    item = replace(event(1, "Медведь ест рыбу."), source="seed_corpus_toloka")
    await EventRepository(database).append(item)
    await PropositionMemory(database).observe(item)
    assert not any("SELECT DISTINCT c.claim_id,c.thought_json" in sql for sql in statements)
    assert not await database.fetchone("SELECT 1 FROM meaning_evidence WHERE voter=1")


async def test_parser_cache_survives_restart_and_rejects_old_version(database, monkeypatch):
    learning = dialogue.DialogueStructureLearning(database, None)
    prompt, answer = "Ну привет!", "Привет, как дела?"
    expected = dialogue.exchange_analysis(prompt)
    forms = dialogue.cached_response_forms(prompt, answer)
    await learning._persist_analysis(prompt, answer)
    path = database.path
    await database.close()
    dialogue._ANALYSIS_PAYLOADS.clear()
    dialogue.exchange_analysis.cache_clear()
    dialogue.cached_response_forms.cache_clear()
    await database.initialize()
    restored = dialogue.DialogueStructureLearning(database, None)
    await restored._load_analysis_cache()
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise AssertionError("parser called despite a persisted projection")
        patch.setattr(dialogue, "analyze_structured_utterance", fail)
        patch.setattr(dialogue, "response_forms", fail)
        assert dialogue.exchange_analysis(prompt) == expected
        assert dialogue.cached_response_forms(prompt, answer) == forms
    assert database.path == path
    dialogue._ANALYSIS_PAYLOADS.clear()
    dialogue.exchange_analysis.cache_clear()
    await database.execute("UPDATE language_exchange_analysis SET parser_version='obsolete'")
    await dialogue.DialogueStructureLearning(database, None)._load_analysis_cache()
    assert not dialogue._ANALYSIS_PAYLOADS
    assert dialogue.exchange_analysis(prompt) == expected


def test_mutable_dictionary_callback_is_never_cached():
    known = {"касса"}
    callback = known.__contains__
    first = normalize_chat_token("касссса", is_known=callback)
    known.clear()
    second = normalize_chat_token("касссса", is_known=callback)
    assert first.canonical != second.canonical


async def test_materialized_support_matches_history_and_reliability_invalidates(database):
    memory = PropositionMemory.for_meaning(database)
    for n, actor in enumerate((10, 11, 12), 1):
        await observe(database, n, "Медведь ест рыбу.", actor)
    key = (await database.fetchone("SELECT meaning_key FROM meaning_evidence"))[0]
    assert await database.fetchone("SELECT 1 FROM meaning_support")
    rows = await database.fetchall("SELECT * FROM meaning_evidence WHERE voter=1")
    assert await memory.support(1, key, public=True) == count_meaning_evidence(rows)
    await memory.source_reliability(1, 10)
    for n, actor in enumerate((10, 11, 12), 4):
        await observe(database, n, "Кошка ест мясо." if actor != 10 else
                      "Кошка не ест мясо.", actor)
    assert await memory.source_reliability(1, 10) < 1
    for cutoff in (2, 4, 7):
        snapshot = event(cutoff, "").utc_timestamp.isoformat()
        expected = count_meaning_evidence([r for r in rows if r["observed_at"] < snapshot])
        actual = await memory.support(1, key, snapshot=snapshot, public=True)
        assert (actual.supports, actual.contradicts, actual.incumbent) == (
            expected.supports, expected.contradicts, expected.incumbent)


def test_incremental_aggregates_preserve_order_and_distinct_witness_gates():
    def row(n, prompt, answer, actor):
        forms, links = dialogue.cached_response_forms(prompt, answer)
        return dict(answer_event_id=str(n), prompt_event_id=f"p:{n}", prompt=prompt,
                    answer=answer, prompt_actor_id=actor,
                    response_forms_json=json.dumps(forms), topic_links_json=json.dumps(links))
    rows = [row(1, "Я инженер", "Ого", 10), row(2, "Я полицейский", "Ого", 10)]
    aggregate = dialogue._ExchangeRows()
    aggregate.update(rows)
    graph, totals = aggregate.question_projection(), aggregate.projection_totals()
    aggregate.update([dict(r) for r in rows])
    assert aggregate.question_projection() is graph
    assert aggregate.projection_totals() is totals
    aggregate.update(list(reversed(rows)))
    assert aggregate.question_projection() is not graph
    assert [r["answer_event_id"] for r in aggregate] == ["2", "1"]
    assert not dialogue.evaluation_tone_forms("лошара", aggregate)
    rows.insert(0, row(3, "У тебя есть кошка?", "Ого", 11))
    aggregate.update(rows)
    assert dialogue.evaluation_tone_forms("лошара", aggregate) == (
        dialogue.evaluation_tone_forms("лошара", rows))
    assert dialogue.evaluation_tone_forms("лошара", aggregate)
    aggregate.update(rows[:2])
    assert not dialogue.evaluation_tone_forms("лошара", aggregate)


async def test_support_reads_ready_counts_once_after_history_warmup(database):
    memory = PropositionMemory.for_meaning(database)
    for n, actor in enumerate((10, 11, 12), 1):
        await observe(database, n, "Медведь ест рыбу.", actor)
    key = (await database.fetchone("SELECT meaning_key FROM meaning_evidence"))[0]
    await memory.source_reliability(1, 10)
    count = database.read_queries
    assert (await memory.support(1, key)).supports == 3
    assert database.read_queries - count == 1


async def test_background_warmup_returns_and_is_cancelled_before_database_close(
    database, monkeypatch,
):
    import asyncio

    entered, cancelled = asyncio.Event(), asyncio.Event()
    learning = dialogue.DialogueStructureLearning(database, None)

    async def warming():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(learning, "warm_analysis_cache", warming)
    learning.start_analysis_warmup()
    await asyncio.wait_for(entered.wait(), 1)
    assert not learning._warm_task.done()
    await database.close()
    assert cancelled.is_set()


def test_parser_backend_invalidates_analysis_and_auxiliary_caches():
    previous = dialogue.ANALYSIS_VERSION.split(":", 1)[1]
    dialogue.exchange_analysis("Медведь ест рыбу.")
    dialogue.configure_analysis_cache("test-backend")
    assert dialogue.ANALYSIS_VERSION.endswith(":test-backend")
    assert dialogue.exchange_analysis.cache_info().currsize == 0
    dialogue.configure_analysis_cache(previous)
