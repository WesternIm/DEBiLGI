from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.events.normalizer import normalize_message_created
from debilgi.language.context_gate import WorkingMemoryContextGate
from debilgi.language.definition_relations import definition_answer_thought
from debilgi.language.lexicon import LexiconService
from debilgi.language.open_questions import OpenLanguageQuestionService
from debilgi.language.productive_nlg import realize_thought
from debilgi.storage.database import Database

NOW = datetime(2026, 7, 18, 10, 0, tzinfo=UTC)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "open-questions.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def message(
    message_id: int,
    *,
    user_id: int,
    content: str,
    reply_to_message_id: int | None = None,
    mentioned_user_ids: tuple[int, ...] = (),
):
    return normalize_message_created(
        message_id=message_id,
        guild_id=1,
        channel_id=10,
        author_id=user_id,
        created_at=NOW + timedelta(seconds=message_id),
        content=content,
        author_is_bot=False,
        reply_to_message_id=reply_to_message_id,
        bot_mentioned=False,
        bot_user_id=999,
        mentioned_user_ids=mentioned_user_ids,
    )


def services(
    database: Database,
) -> tuple[LexiconService, SelfDatasetRepository, OpenLanguageQuestionService]:
    lexicon = LexiconService(database)
    dataset = SelfDatasetRepository(database)
    return lexicon, dataset, OpenLanguageQuestionService(database, lexicon, dataset)


async def test_lesson_survives_restart_and_supplies_meaning_to_inflected_word(database):
    lexicon, _, questions = services(database)
    prompt = message(1, user_id=20, content="салфетка")
    await questions.open_for(prompt, concept="салфетка", question_message_id=500, asked_at=NOW)
    lesson = message(2, user_id=20, content="Это бумага для гигиены", reply_to_message_id=500)
    await questions.observe(lesson)
    assert await questions.resolution_for_event(lesson.event_id) is not None
    await database.close()
    await database.initialize()
    lexicon = LexiconService(database)
    learned = await lexicon.analyze_word("салфеткой", guild_id=1, context="Пользуюсь салфеткой")
    assert learned.source == "personal_lexicon"
    assert learned.selected_meaning == "Это бумага для гигиены"
    assert learned.grammatical_features["case"] == "instrumental"
    assert ("is_a", "бумага") in learned.semantic_relations
    thought = definition_answer_thought("Что такое салфетка?", ((
        learned.lemma, learned.semantic_relations,
    ),))
    assert thought is not None
    assert "бумагой" in realize_thought(thought, limit=1)[0].text
    other = await lexicon.analyze_word("салфеткой", guild_id=2)
    assert other.personal_interpretation is None


async def test_other_speaker_cannot_supply_the_pending_lesson(database):
    lexicon, _, questions = services(database)
    prompt = message(1, user_id=20, content="салфетка")
    await questions.open_for(prompt, concept="салфетка", question_message_id=500, asked_at=NOW)
    await questions.observe(message(2, user_id=30, content="Это оружие", reply_to_message_id=500))
    assert await questions.active_for(1, 10, 20) is not None
    assert (await lexicon.analyze_word("салфетка", guild_id=1)).source != "personal_lexicon"


async def test_explanation_is_bound_learned_and_confirmed(database: Database) -> None:
    lexicon, dataset, questions = services(database)
    prompt = message(1, user_id=20, content="Я сужу по твоим ответам")
    await questions.open_for(
        prompt,
        concept="сужу",
        question_message_id=500,
        asked_at=NOW,
    )
    await questions.open_for(
        prompt, concept="сужу", question_message_id=500, asked_at=NOW,
    )
    rows = await database.fetchall("SELECT status FROM language_open_questions")
    assert len(rows) == 1
    assert rows[0]['status'] == 'active'
    explanation = message(
        2,
        user_id=20,
        content="Сужу — это из слова суд, а не сузить",
    )

    await questions.observe(explanation)

    resolution = await questions.resolution_for_event(explanation.event_id)
    learned = await lexicon.analyze_word("сужу", guild_id=1, channel_id=10)
    related = await dataset.related_concepts(1, "сужу")
    assert resolution is not None
    assert resolution.confirmation == (
        "Понял: «сужу» здесь связано со словом «суд», "
        "а разбор как «сузить» был ошибочным."
    )
    assert learned.personal_interpretation == "Сужу — это из слова суд, а не сузить"
    assert "суд" in related
    assert await questions.active_for(1, 10, 20) is None

    await dataset.rebuild_graph(1)
    assert "суд" in await dataset.related_concepts(1, "сужу")


async def test_ambiguous_messages_expire_only_the_same_users_marker(database: Database) -> None:
    _, _, questions = services(database)
    first = message(10, user_id=20, content="стромба тут")
    second = message(11, user_id=21, content="клюмба тут")
    await questions.open_for(first, concept="стромба", question_message_id=510, asked_at=NOW)
    await questions.open_for(second, concept="клюмба", question_message_id=511, asked_at=NOW)

    await questions.observe(message(12, user_id=21, content="хмм пока не знаю"))
    untouched = await questions.active_for(1, 10, 20)
    decremented = await questions.active_for(1, 10, 21)
    assert untouched is not None and untouched.turns_remaining == 2
    assert decremented is not None and decremented.turns_remaining == 1

    await questions.observe(message(13, user_id=21, content="потом разберёмся"))

    assert await questions.active_for(1, 10, 20) is not None
    assert await questions.active_for(1, 10, 21) is None
    row = await database.fetchone(
        """SELECT status, relevance_reason FROM language_open_questions
           WHERE guild_id = 1 AND user_id = 21"""
    )
    assert row is not None
    assert (row["status"], row["relevance_reason"]) == ("stale", "ttl_exhausted")


async def test_explicit_topic_shift_drops_marker_without_learning(database: Database) -> None:
    lexicon, _, questions = services(database)
    prompt = message(20, user_id=30, content="что такое стромба")
    await questions.open_for(
        prompt,
        concept="стромба",
        question_message_id=520,
        asked_at=NOW,
    )

    shifted = message(21, user_id=30, content="Кстати, почему сервер сегодня упал?")
    await questions.observe(shifted)

    assert await questions.active_for(1, 10, 30) is None
    analysis = await lexicon.analyze_word("стромба", guild_id=1, channel_id=10)
    assert analysis.personal_interpretation is None
    row = await database.fetchone(
        """SELECT status, relevance_reason FROM language_open_questions
           WHERE guild_id = 1 AND user_id = 30"""
    )
    assert row is not None
    assert (row["status"], row["relevance_reason"]) == (
        "topic_changed",
        "explicit_topic_shift",
    )


async def test_reply_to_exact_question_is_accepted_without_trigger(database: Database) -> None:
    _, _, questions = services(database)
    prompt = message(30, user_id=40, content="непонятное слово")
    await questions.open_for(
        prompt,
        concept="стромба",
        question_message_id=530,
        asked_at=NOW,
    )
    reply = message(
        31,
        user_id=40,
        content="хаотичное повторение одной ошибки",
        reply_to_message_id=530,
    )

    await questions.observe(reply)

    resolution = await questions.resolution_for_event(reply.event_id)
    assert resolution is not None
    assert resolution.relevance_reason == "reply_to_question"


async def test_meta_answer_about_separate_meaning_is_not_learned_as_definition(
    database: Database,
) -> None:
    lexicon, _, questions = services(database)
    prompt = message(40, user_id=50, content="что означает клоунесса")
    await questions.open_for(
        prompt,
        concept="клоунесса",
        question_message_id=540,
        asked_at=NOW,
    )
    meta_answer = message(
        41,
        user_id=50,
        content="Есть отдельное значение",
        reply_to_message_id=540,
    )

    await questions.observe(meta_answer)

    analysis = await lexicon.analyze_word("клоунесса", guild_id=1, channel_id=10)
    active = await questions.active_for(1, 10, 50)
    assert analysis.personal_interpretation is None
    assert active is not None and active.turns_remaining == 1
    assert await questions.resolution_for_event(meta_answer.event_id) is None


async def test_open_question_marker_uses_and_clears_shared_working_context(
    database: Database,
) -> None:
    lexicon = LexiconService(database)
    dataset = SelfDatasetRepository(database)
    working = WorkingMemoryContextGate(dataset)
    questions = OpenLanguageQuestionService(
        database,
        lexicon,
        dataset,
        working_context=working,
    )
    prompt = message(50, user_id=60, content="что означает фырбуль")
    await questions.open_for(
        prompt,
        concept="фырбуль",
        question_message_id=550,
        asked_at=NOW,
    )

    assert await working.pending_question(1, 10, 60) == "фырбуль"
    explanation = message(
        51,
        user_id=60,
        content="Фырбуль — это маленький складной фонарь",
        reply_to_message_id=550,
    )
    await questions.observe(explanation)

    assert await working.pending_question(1, 10, 60) is None
