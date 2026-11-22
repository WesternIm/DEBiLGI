from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.canonical_claims import Known, Unknown, answer_thought, query_from_event
from debilgi.language.generative_planner import plan_from_evidence
from debilgi.language.planning import ReplyContext
from debilgi.language.proposition_memory import PropositionMemory
from debilgi.language.semantic_graph import extract_meanings, infer_meanings
from debilgi.storage.database import Database

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
NOW = datetime(2026, 11, 14, 12, tzinfo=UTC)


def message(event_id, text, *, seconds=0, **changes):
    return replace(
        Event(
            event_id,
            1,
            2,
            3,
            99,
            EventType.MESSAGE_CREATED,
            NOW + timedelta(seconds=seconds),
            {"content": text},
            "test",
        ),
        **changes,
    )


@pytest.fixture
async def memory(tmp_path):
    database = Database(tmp_path / "claims.sqlite3", MIGRATIONS)
    await database.initialize()
    try:
        yield PropositionMemory(database, timezone=ZoneInfo("Asia/Almaty")), database
    finally:
        await database.close()


async def observe(memory, database, *events):
    for event in events:
        await EventRepository(database).append(event)
        await memory.observe(event)


def object_of(claim):
    return next(
        d.argument.head.lemma for d in claim.thought.dependencies if d.role == "direct_object"
    )


async def snapshot(database):
    return {
        table: [tuple(r) for r in await database.fetchall(f"SELECT * FROM {table} ORDER BY 1, 2")]
        for table in (
            "meaning_claims",
            "meaning_arguments",
            "meaning_evidence",
            "meaning_supersedes",
            "meaning_claim_projections",
        )
    }


async def test_correction_history_proof_and_raw_redaction_restart(memory):
    memory, database = memory
    book = message("book", "я читаю книгу.")
    correction = message("journal", "нет, я читаю журнал.", seconds=20)
    await observe(memory, database, book, correction)
    question = message("q", "что я читаю?", seconds=40)
    result = await memory.retrieve_claims(question)
    assert isinstance(result, Known) and len(result.claims) == 1
    claim = result.claims[0]
    assert object_of(claim) == "журнал" and claim.polarity == 1
    assert claim.subject_ref == "discord:user:3" and claim.source_kind == "self_report"
    assert claim.proof.claim_ids == (claim.claim_id,)
    assert claim.proof.source_event_ids == ("journal",)
    history = await memory.query_claims(replace(result.query, include_history=True))
    assert {object_of(c) for c in history.claims} == {"книга", "журнал"}
    assert claim.supersedes == (
        next(c.claim_id for c in history.claims if object_of(c) == "книга"),
    )
    before = await memory.retrieve_claims(replace(question, timestamp=NOW + timedelta(seconds=10)))
    assert [object_of(c) for c in before.claims] == ["книга"]
    plan = plan_from_evidence(
        ReplyContext(
            prompt=question.payload["content"], input_meaning=question.meaning, claim_result=result
        )
    )
    assert {object_of(replace(claim, thought=p.plan.claim)) for p in plan.proposals} == {"журнал"}
    assert all(p.operation == "retrieve_proposition" for p in plan.proposals)
    assert claim.claim_id in plan.proposals[0].plan.claim.evidence
    retained = await snapshot(database)
    for _ in range(2):
        await memory.restore_from_events()
        await memory.observe(book)
        await memory.observe(correction)
    assert await snapshot(database) == retained
    await database.execute(
        "UPDATE events SET payload_json = json_remove(payload_json, '$.content')"
    )
    await database.close()
    await database.initialize()
    restored = PropositionMemory(database, timezone=ZoneInfo("Asia/Almaty"))
    await restored.restore_from_events()
    assert await snapshot(database) == retained
    assert await restored.retrieve_claims(question) == result


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 0, 2)])
async def test_out_of_order_replay_has_same_correction_chain(memory, order):
    memory, database = memory
    events = (
        message("book", "я читаю книгу."),
        message("journal", "нет, я читаю журнал.", seconds=20),
        message("letter", "нет, я читаю письмо.", seconds=30),
    )
    for i in order:
        await observe(memory, database, events[i])
    result = await memory.retrieve_claims(message("q", "что я читаю?", seconds=40))
    assert [object_of(c) for c in result.claims] == ["письмо"]
    before = await snapshot(database)
    await database.execute("DELETE FROM meaning_supersedes")
    await database.execute("DELETE FROM meaning_evidence")
    await database.execute("DELETE FROM meaning_arguments")
    await database.execute("DELETE FROM meaning_claims")
    await database.execute("DELETE FROM meaning_claim_projections")
    await memory.backfill_claims(batch_size=1)
    assert await snapshot(database) == before


async def test_reply_id_targets_older_fact_and_never_other_author(memory):
    memory, database = memory
    book = message("discord:message:10", "я читаю книгу.")
    letter = message("letter", "я читаю письмо.", seconds=10)
    correction = message(
        "journal",
        "нет, я читаю журнал.",
        seconds=20,
        payload={"content": "нет, я читаю журнал.", "reply_to_message_id": 10},
    )
    await observe(memory, database, book, letter, correction)
    result = await memory.retrieve_claims(message("q", "что я читаю?", seconds=40))
    assert {object_of(c) for c in result.claims} == {"журнал", "письмо"}
    other = replace(correction, event_id="other", actor_id=4, timestamp=NOW + timedelta(seconds=30))
    await observe(memory, database, other)
    assert (
        not (await memory.retrieve_claims(message("q2", "что я читаю?", seconds=40, actor_id=4)))
        .claims[0]
        .supersedes
    )
    assert await memory.retrieve_claims(message("q", "что я читаю?", seconds=40)) == result


@pytest.mark.parametrize(
    "change",
    [
        {"actor_id": 4},
        {"channel_id": 4},
        {"guild_id": 4},
        {"timestamp": NOW - timedelta(seconds=1)},
    ],
)
async def test_scope_and_snapshot_do_not_leak(memory, change):
    memory, database = memory
    await observe(memory, database, message("fact", "я читаю книгу."))
    query = message("q", "что я читаю?", seconds=20, **change)
    assert isinstance(await memory.retrieve_claims(query), Unknown)


async def test_private_third_person_reports_keep_attributions_without_disclosure(memory):
    memory, database = memory
    await observe(
        memory,
        database,
        message("positive", "ты читаешь книгу.", target_id=99),
        message("negative", "ты не читаешь книгу.", target_id=99, actor_id=4),
        message("update", "нет, ты читаешь журнал.", target_id=99, seconds=10),
    )
    first = await memory.retrieve_claims(message("q", "что ты читаешь?", seconds=20))
    second = await memory.retrieve_claims(message("q", "ты читаешь книгу?", seconds=20, actor_id=4))
    assert isinstance(first, Unknown) and isinstance(second, Unknown)
    rows = await database.fetchall(
        "SELECT * FROM meaning_claims WHERE subject_ref='discord:user:99'"
    )
    assert len(rows) == 3 and {r["speaker_id"] for r in rows} == {3, 4}
    negative = next(r for r in rows if r["source_event_id"] == "negative")
    assert negative["polarity"] == -1 and negative["source_kind"] == "other_report"
    assert not await database.fetchall(
        "SELECT * FROM meaning_supersedes WHERE claim_id=?", (negative["claim_id"],)
    )


@pytest.mark.parametrize(
    "timezone, source",
    [
        ("Asia/Almaty", datetime(2026, 11, 14, 23, tzinfo=UTC)),
        ("America/New_York", datetime(2026, 11, 1, 7, tzinfo=UTC)),
    ],
)
async def test_relative_dates_use_source_day_and_server_zone(memory, timezone, source):
    _, database = memory
    memory = PropositionMemory(database, timezone=ZoneInfo(timezone))
    book = message("book", "я сегодня читал книгу.", timestamp=source)
    journal = message("journal", "я сегодня читал журнал.", timestamp=source + timedelta(days=1))
    await observe(memory, database, book, journal)
    question = message(
        "q", "что я вчера читал?", timestamp=journal.timestamp + timedelta(seconds=20)
    )
    yesterday = await memory.retrieve_claims(question)
    today = await memory.retrieve_claims(
        replace(question, payload={"content": "что я сегодня читал?"})
    )
    assert [object_of(c) for c in yesterday.claims] == ["книга"]
    assert [object_of(c) for c in today.claims] == ["журнал"]
    first = yesterday.claims[0]
    assert first.source_time == source.isoformat()
    assert datetime.fromisoformat(first.valid_from).astimezone(ZoneInfo(timezone)).hour == 0
    assert datetime.fromisoformat(first.valid_to).astimezone(ZoneInfo(timezone)).hour == 0
    rendered = answer_thought(first, yesterday.query)
    assert rendered.semantic_time.deictic == "вчера"
    assert "сегодня" not in {d.argument.head.lemma for d in rendered.dependencies}
    future = replace(question, timestamp=source + timedelta(days=4))
    assert isinstance(await memory.retrieve_claims(future), Unknown)


@pytest.mark.parametrize(
    "text", ["что я покупаю?", "я покупаю журнал?", "что ты читаешь?", "что я читал вчера?"]
)
async def test_missing_fact_is_unknown_with_query_receipt(memory, text):
    memory, database = memory
    await observe(memory, database, message("fact", "я читаю книгу."))
    question = message("q", text, seconds=20)
    result = await memory.retrieve_claims(question)
    assert isinstance(result, Unknown) and result.claims == ()
    assert result.query == query_from_event(question, timezone=ZoneInfo("Asia/Almaty"))
    # A failed query never writes a negative claim or creates training evidence.
    assert len(await database.fetchall("SELECT * FROM meaning_claims")) == 1


async def test_modality_is_retained_and_not_promoted_to_actual_action(memory):
    memory, database = memory
    await observe(memory, database, message("can", "я могу читать книгу."))
    result = await memory.retrieve_claims(message("q", "я могу читать книгу?", seconds=20))
    assert isinstance(result, Known)
    assert result.claims[0].thought.scopes[0].operator == "мочь"
    assert isinstance(
        await memory.retrieve_claims(message("q", "что я читаю?", seconds=20)), Unknown
    )


async def test_negative_correction_keeps_embedded_negation_and_open_world(memory):
    memory, database = memory
    await observe(
        memory,
        database,
        message("positive", "я читаю книгу."),
        message("negative", "нет, я не читаю книгу.", seconds=10),
    )
    result = await memory.retrieve_claims(message("q", "я читаю книгу?", seconds=20))
    assert len(result.claims) == 1 and result.claims[0].polarity == -1
    assert result.claims[0].thought.negated
    assert len(result.claims[0].supersedes) == 1
    absent = await memory.retrieve_claims(message("q", "я читаю журнал?", seconds=20))
    assert isinstance(absent, Unknown)


async def test_future_valid_time_is_not_current_activity(memory):
    memory, database = memory
    await observe(memory, database, message("tomorrow", "я завтра читаю книгу."))
    current = message("q", "что я читаю?", seconds=20)
    assert isinstance(await memory.retrieve_claims(current), Unknown)
    tomorrow = replace(current, payload={"content": "что я завтра читаю?"})
    assert [object_of(c) for c in (await memory.retrieve_claims(tomorrow)).claims] == ["книга"]


async def test_past_query_without_date_does_not_repeat_stale_today(memory):
    memory, database = memory
    await observe(memory, database, message("read", "я сегодня читал книгу."))
    question = message("q", "что я читал?", timestamp=NOW + timedelta(days=1))
    result = await memory.retrieve_claims(question)
    thought = answer_thought(result.claims[0], result.query)
    assert thought.semantic_time.deictic == "вчера"
    assert "сегодня" not in {d.argument.head.lemma for d in thought.dependencies}


async def test_arguments_are_roles_not_unordered_words(memory):
    memory, database = memory
    await observe(memory, database, message("give", "я дарю тебе книгу.", target_id=99))
    correct = await memory.retrieve_claims(message("q", "что я дарю тебе?", seconds=20))
    reversed_roles = await memory.retrieve_claims(message("q", "что ты даришь мне?", seconds=20))
    assert isinstance(correct, Known) and object_of(correct.claims[0]) == "книга"
    assert isinstance(reversed_roles, Unknown)


async def test_existing_rule_proofs_keep_canonical_claim_ids(memory):
    memory, database = memory
    await observe(memory, database, message("cause", "Робот скучает, потому что человек молчит."))
    question = message("q", "Почему робот скучает?", seconds=20)
    meanings = await memory.learned_meanings(question)
    claim_ids = {c for r in meanings for c in r.claim_ids}
    assert len(claim_ids) == 2
    assert all(set(p.claim_ids) <= claim_ids for p in infer_meanings(meanings))
    result = plan_from_evidence(
        ReplyContext(
            prompt=question.payload["content"],
            input_meaning=question.meaning,
            claim_result=await memory.retrieve_claims(question),
            meaning_records=meanings,
        )
    )
    assert result.proposals[0].plan.reason is not None
    assert claim_ids <= set(result.proposals[0].plan.claim.evidence)


async def test_redacted_multi_sentence_evidence_rebuilds_same_claim_ids(memory):
    memory, database = memory
    await observe(memory, database, message("multi", "я читаю книгу. я пишу письмо."))
    before = await snapshot(database)
    assert len(before["meaning_claims"]) == 2
    await database.execute(
        "UPDATE events SET payload_json = json_remove(payload_json, '$.content')"
    )
    for table in (
        "meaning_supersedes",
        "meaning_evidence",
        "meaning_arguments",
        "meaning_claims",
        "meaning_claim_projections",
    ):
        await database.execute(f"DELETE FROM {table}")
    await memory.backfill_claims()
    assert await snapshot(database) == before


async def test_raw_dialogue_backfill_matches_live_two_source_claim(memory):
    memory, database = memory
    question = message(
        "discord:message:1",
        "Ты нашёл ключ?",
        actor_id=99,
        target_id=3,
        payload={"content": "Ты нашёл ключ?", "author_is_bot": True, "message_id": 1},
    )
    answer = message(
        "answer",
        "Нет",
        seconds=1,
        payload={"content": "Нет", "reply_to_message_id": 1, "bot_user_id": 99},
    )
    await observe(memory, database, question, answer)
    before = await snapshot(database)
    assert len(before["meaning_claims"]) == 1
    for table in (
        "meaning_supersedes",
        "meaning_evidence",
        "meaning_arguments",
        "meaning_claims",
        "meaning_claim_projections",
        "language_dialogue_reports",
    ):
        await database.execute(f"DELETE FROM {table}")
    await memory.backfill_claims(batch_size=1)
    assert await snapshot(database) == before


async def test_corpus_and_bot_never_supply_personal_claims(memory):
    memory, database = memory
    await observe(
        memory,
        database,
        message("corpus", "я читаю книгу.", source="seed_corpus_toloka"),
        message(
            "agent",
            "я читаю книгу.",
            actor_id=99,
            payload={"content": "я читаю книгу.", "author_is_bot": True},
        ),
    )
    assert isinstance(
        await memory.retrieve_claims(message("q", "что я читаю?", seconds=20)), Unknown
    )
    assert await memory.learned_meanings(message("q", "что я читаю?", seconds=20)) == ()
    rows = await database.fetchall("SELECT subject_ref, source_kind FROM meaning_claims")
    assert len(rows) == 1 and rows[0]["subject_ref"].startswith("corpus:")
    assert rows[0]["source_kind"] == "corpus_form"


async def test_legacy_migration_backfills_redacted_reports_and_preserves_evidence(tmp_path):
    old = tmp_path / "old_migrations"
    old.mkdir()
    for path in MIGRATIONS.glob("*.sql"):
        if int(path.name.split("_")[0]) < 60:
            (old / path.name).write_text(path.read_text())
    database = Database(tmp_path / "legacy.sqlite3", old)
    await database.initialize()
    events = (
        message("book", "я читаю книгу."),
        message("corpus", "я читаю журнал.", source="seed_corpus_toloka"),
    )
    import json

    for event in events:
        await EventRepository(database).append(event)
        await database.execute(
            "INSERT INTO language_proposition_reports VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.event_id,
                1,
                2,
                3,
                event.payload["content"],
                event.utc_timestamp.isoformat(),
            ),
        )
        for record in extract_meanings(event.meaning, event.event_id):
            await database.execute(
                "INSERT INTO language_meaning_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.meaning_id,
                    event.event_id,
                    1,
                    2,
                    3,
                    record.kind,
                    json.dumps(asdict(record)),
                    event.utc_timestamp.isoformat(),
                ),
            )
    await database.execute("UPDATE events SET payload_json = '{}'")
    await database.close()
    database = Database(database.path, MIGRATIONS)
    await database.initialize()
    try:
        memory = PropositionMemory(database)
        await memory.backfill_claims(batch_size=1)
        query = message("q", "что я читаю?", seconds=20)
        assert [object_of(c) for c in (await memory.retrieve_claims(query)).claims] == ["книга"]
        assert len(await database.fetchall("SELECT * FROM language_proposition_reports")) == 2
        before = await snapshot(database)
        await memory.restore_from_events()
        assert await snapshot(database) == before
    finally:
        await database.close()


def test_past_polar_question_without_subject_queries_the_agent():
    from datetime import UTC, datetime

    from debilgi.events.normalizer import normalize_message_created
    from debilgi.language.canonical_claims import query_from_event

    event = normalize_message_created(
        message_id=1, guild_id=2, channel_id=3, author_id=4,
        created_at=datetime(2026, 11, 16, tzinfo=UTC), content="понял?",
        author_is_bot=False, bot_user_id=99, bot_mentioned=True)
    query = query_from_event(event)
    assert query.predicate == "понять"
    assert ("subject", "discord:agent:99") in query.arguments
    assert ("subject", "discord:user:4") not in query.arguments


async def test_past_polar_query_does_not_borrow_the_users_self_report(memory):
    graph, database = memory
    report = message("human-report", "я не понял тебя")
    report = replace(report, payload={**report.payload, "bot_user_id": 99,
                                       "bot_mentioned": True, "author_is_bot": False})
    await observe(graph, database, report)
    stored = await database.fetchone(
        "SELECT subject_ref FROM meaning_claims WHERE source_event_id=? AND predicate='понять'",
        (report.event_id,))
    assert stored and stored["subject_ref"] == "discord:user:3"
    query = message("agent-understanding", "понял?", seconds=20)
    query = replace(query, payload={**query.payload, "bot_user_id": 99, "bot_mentioned": True})
    result = await graph.retrieve_claims(query)
    assert isinstance(result, Unknown)
    assert ("subject", "discord:agent:99") in result.query.arguments
    assert not result.claims
