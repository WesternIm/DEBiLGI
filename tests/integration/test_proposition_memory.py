from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.proposition_memory import PropositionMemory
from debilgi.storage.database import Database

NOW = datetime(2026, 9, 18, 12, tzinfo=UTC)
MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"


def message(event_id, text):
    return Event(
        event_id,
        1,
        2,
        3,
        None,
        EventType.MESSAGE_CREATED,
        NOW - timedelta(days=1),
        {"content": text},
        "test",
    )


async def test_semantic_rules_survive_restart_raw_redaction_and_are_scoped(tmp_path):
    path = tmp_path / "meanings.sqlite3"
    database = Database(path, MIGRATIONS)
    await database.initialize()
    definition = message(
        "definition", "Быть выключенным для тебя — это когда ты не можешь отвечать мне."
    )
    examples = (
        definition,
        replace(definition, event_id="bot", payload={**definition.payload, "author_is_bot": True}),
        replace(definition, event_id="elsewhere", channel_id=99),
        replace(definition, event_id="other", actor_id=99),
        replace(definition, event_id="future", timestamp=NOW + timedelta(days=1)),
    )
    memory = PropositionMemory(database)
    for item in examples:
        await EventRepository(database).append(item)
        await memory.observe(item)
        await memory.observe(item)
    assert len(await database.fetchall("SELECT * FROM language_meaning_records")) == 4
    await database.execute(
        "UPDATE events SET payload_json = json_remove(payload_json, '$.content')"
    )
    await database.close()
    database = Database(path, MIGRATIONS)
    await database.initialize()
    try:
        memory = PropositionMemory(database)
        await memory.restore_from_events()
        current = replace(message("question", "Ты хочешь быть выключенным?"), timestamp=NOW)
        retained = await memory.learned_meanings(current)
        assert len(retained) == 1 and retained[0].source_event_id == "definition"
        assert retained[0].kind == "definition"
    finally:
        await database.close()


async def test_current_states_expire_and_explicit_updates_replace_older_polarity(tmp_path):
    database = Database(tmp_path / "states.sqlite3", MIGRATIONS)
    await database.initialize()
    try:
        memory = PropositionMemory(database)
        examples = (
            message("old", "Бот заряжен."),
            replace(message("state", "Бот выключен."), timestamp=NOW - timedelta(minutes=4)),
            replace(message("update", "Бот не выключен."), timestamp=NOW - timedelta(minutes=2)),
        )
        for item in examples:
            await EventRepository(database).append(item)
            await memory.observe(item)
        current = replace(message("question", "Бот выключен?"), timestamp=NOW)
        meanings = await memory.learned_meanings(current)
        assert len(meanings) == 1
        assert meanings[0].source_event_id == "update"
        assert meanings[0].conclusions[0].negated
        assert (
            await memory.learned_meanings(replace(current, timestamp=NOW + timedelta(hours=2)))
            == ()
        )
    finally:
        await database.close()


async def test_historical_reports_keep_scope_provenance_and_polarity(tmp_path):
    database = Database(tmp_path / "reports.sqlite3", MIGRATIONS)
    await database.initialize()
    try:
        events = EventRepository(database)
        own = message("human", "Маша не нашла ключ.")
        bot = replace(
            message("bot", "Маша нашла ключ."),
            payload={"content": "Маша нашла ключ.", "author_is_bot": True},
        )
        others = [
            replace(message("other-author", "Маша нашла ключ."), actor_id=4),
            replace(message("other-channel", "Маша нашла ключ."), channel_id=5),
            replace(message("other-server", "Маша нашла ключ."), guild_id=6),
            replace(message("future", "Маша нашла ключ."), timestamp=NOW + timedelta(days=1)),
            message("qualified", "Маша, возможно, нашла ключ."),
            message("unresolved", "Она нашла ключ."),
        ]
        for item in (own, bot, *others):
            await events.append(item)
        memory = PropositionMemory(database)
        await memory.restore_from_events()
        await memory.restore_from_events()
        current = replace(message("question", "Кто нашёл ключ?"), timestamp=NOW)
        reports = await memory.relevant_reports(current, "Кто нашёл ключ?")
        assert reports == (("human", "Маша не нашла ключ."),)
        forbidden = await database.fetchall(
            "SELECT source_event_id FROM language_proposition_reports "
            "WHERE source_event_id IN ('bot', 'qualified', 'unresolved')"
        )
        assert forbidden == []
    finally:
        await database.close()


async def test_unparsed_language_survives_restart_and_raw_redaction_without_self_training(tmp_path):
    database_path = tmp_path / "language.sqlite3"
    database = Database(database_path, MIGRATIONS)
    await database.initialize()
    own = message("own", "ебать, а я ногти грызу")
    fragment = message("fragment", "привееет 😂")
    bot = replace(
        message("bot", "служебная заготовка"),
        payload={"content": "служебная заготовка", "author_is_bot": True},
    )
    elsewhere = replace(message("elsewhere", "межканальный текст"), channel_id=99)
    memory = PropositionMemory(database)
    for item in (own, fragment, bot, elsewhere):
        await EventRepository(database).append(item)
        await memory.observe(item)
    grams = await memory.learned_ngrams(own)
    assert grams
    assert any("привет" in gram for gram, _ in grams)
    raw = await database.fetchall("SELECT gram_json FROM language_generation_ngrams")
    assert any("привееет" in row["gram_json"] for row in raw)
    assert not any("служебная" in gram or "межканальный" in gram for gram, _ in grams)
    await memory.restore_from_events()
    assert grams == await memory.learned_ngrams(own)
    current = replace(message("current", "ногти"), timestamp=NOW)
    corpus = await memory.relevant_utterances(current, "ногти")
    assert corpus[0] == (own.event_id, own.payload["content"])
    assert set(event_id for event_id, _ in corpus) == {"own", "fragment"}
    reports = await memory.relevant_reports(current, "ногти")
    assert reports == (("own", own.payload["content"]),)
    await database.execute(
        "UPDATE events SET payload_json = json_remove(payload_json, '$.content')"
    )
    await database.close()
    database = Database(database_path, MIGRATIONS)
    await database.initialize()
    try:
        assert await PropositionMemory(database).learned_ngrams(own) == grams
        assert await PropositionMemory(database).relevant_utterances(current, "ногти") == ()
    finally:
        await database.close()


async def test_lexical_dependencies_are_scoped_idempotent_and_survive_raw_redaction(tmp_path):
    path = tmp_path / "relations.sqlite3"
    database = Database(path, MIGRATIONS)
    await database.initialize()
    own = message("own", "Я пишу другу. Я читаю книгу.")
    bot = replace(
        message("bot", "Я дарю слону подарок."),
        payload={"content": "Я дарю слону подарок.", "author_is_bot": True},
    )
    elsewhere = replace(message("elsewhere", "Я звоню сестре."), channel_id=99)
    memory = PropositionMemory(database)
    for item in (own, bot, elsewhere):
        await EventRepository(database).append(item)
        await memory.observe(item)
    links = await memory.learned_word_relations(own)
    assert any(
        link.governor == "писать" and link.dependent == "друг" and link.case == "dative"
        for link in links
    )
    assert any(
        link.governor == "читать" and link.dependent == "книга" and link.case == "accusative"
        for link in links
    )
    assert not any(link.governor in {"дарить", "звонить"} for link in links)
    grams = await memory.learned_ngrams(own)
    assert any(gram == ("ты", "пишешь") for gram, _ in grams)
    assert not any("я" in gram for gram, _ in grams)
    assert not any("." in gram[:-1] and gram[-1] not in {"\0end"} for gram, _ in grams)
    await memory.restore_from_events()
    assert links == await memory.learned_word_relations(own)
    assert grams == await memory.learned_ngrams(own)
    await database.execute(
        "UPDATE events SET payload_json = json_remove(payload_json, '$.content')"
    )
    await database.close()
    database = Database(path, MIGRATIONS)
    await database.initialize()
    try:
        memory = PropositionMemory(database)
        assert links == await memory.learned_word_relations(own)
        assert grams == await memory.learned_ngrams(own)
    finally:
        await database.close()


async def test_answer_reports_keep_two_event_sources_after_restart_and_raw_redaction(tmp_path):
    from debilgi.language.generative_planner import plan_from_evidence
    from debilgi.language.planning import ReplyContext

    path = tmp_path / "answers.sqlite3"
    database = Database(path, MIGRATIONS)
    await database.initialize()
    question = replace(
        message("q", "Зачем ты открыл дверь?"),
        actor_id=99,
        target_id=3,
        timestamp=NOW - timedelta(minutes=1),
        payload={"content": "Зачем ты открыл дверь?", "author_is_bot": True, "message_id": 1},
    )
    answer = replace(
        message("a", "Для проверки"),
        timestamp=question.timestamp + timedelta(seconds=1),
        payload={"content": "Для проверки", "bot_user_id": 99, "reply_to_message_id": 1},
    )
    polarity_question = replace(
        question,
        event_id="pq",
        timestamp=NOW - timedelta(seconds=30),
        payload={"content": "Ты нашёл ключ?", "author_is_bot": True, "message_id": 2},
    )
    polarity_answer = replace(
        answer,
        event_id="pa",
        timestamp=NOW - timedelta(seconds=29),
        payload={"content": "Нет", "bot_user_id": 99, "reply_to_message_id": 2},
    )
    unknown_question = replace(
        polarity_question,
        event_id="uq",
        timestamp=NOW - timedelta(seconds=20),
        payload={"content": "Ты взял книгу?", "author_is_bot": True, "message_id": 3},
    )
    unknown_answer = replace(
        polarity_answer,
        event_id="ua",
        timestamp=NOW - timedelta(seconds=19),
        payload={"content": "Не знаю", "bot_user_id": 99, "reply_to_message_id": 3},
    )
    memory = PropositionMemory(database)
    repository = EventRepository(database)
    for item in (
        question,
        answer,
        polarity_question,
        polarity_answer,
        unknown_question,
        unknown_answer,
    ):
        await repository.append(item)
        await memory.observe(item)
    await memory.restore_from_events()
    current = replace(message("new", "Почему я открыл дверь?"), timestamp=NOW)
    reports = await memory.relevant_dialogue_reports(current, current.payload["content"])
    assert len(reports) == 1
    assert (reports[0].question_event_id, reports[0].answer_event_id) == ("q", "a")
    assert reports[0].answer_text == "Для проверки"
    result = plan_from_evidence(
        ReplyContext(prompt=current.payload["content"], dialogue_reports=reports)
    )
    assert result.proposals and result.proposals[0].operation == "retrieve_proposition"
    assert {"q", "a"} <= set(result.proposals[0].plan.claim.evidence)
    negative = await memory.relevant_dialogue_reports(current, "Я нашёл ключ?")
    assert negative[0].thought.negated
    assert await memory.relevant_dialogue_reports(current, "Я взял книгу?") == ()
    for changed in (
        replace(current, actor_id=4),
        replace(current, channel_id=7),
        replace(current, guild_id=8),
        replace(current, timestamp=NOW - timedelta(days=1)),
    ):
        assert await memory.relevant_dialogue_reports(changed, current.payload["content"]) == ()
    await database.execute(
        "UPDATE events SET payload_json = json_remove(payload_json, '$.content')"
    )
    await database.close()
    database = Database(path, MIGRATIONS)
    await database.initialize()
    try:
        restored = PropositionMemory(database)
        assert (
            await restored.relevant_dialogue_reports(current, current.payload["content"]) == reports
        )
        assert await restored.relevant_utterances(current, "проверка") == ()
        # Reconstructed propositions never become statistical speech training.
        grams = await restored.learned_ngrams(current)
        assert not any("дверь" in gram or "выключить" in gram for gram, _ in grams)
    finally:
        await database.close()
