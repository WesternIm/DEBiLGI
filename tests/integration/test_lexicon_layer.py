from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.events.normalizer import normalize_message_created
from debilgi.language.lexicon import LexiconService, NeologismRepository, UnknownKind
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "lexicon.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


async def test_morphology_and_contextual_polysemy(database: Database) -> None:
    service = LexiconService(database)

    stick = await service.analyze_word("палкой", context="ударила палкой")
    found = await service.analyze_word("нашла", context="она нашла ключ")
    weapon = await service.analyze_word("лук", context="натянул тетиву и выпустил стрелу")
    food = await service.analyze_word("лук", context="нарезал овощ в салат")
    ambiguous = await service.analyze_word("лук", context="я увидел лук")

    assert stick.lemma == "палка"
    assert stick.grammatical_features["case"] in {"instrumental", "ablt"}
    assert found.lemma == "найти"
    assert found.grammatical_features["gender"] in {"feminine", "femn"}
    assert found.grammatical_features["tense"] == "past"
    assert weapon.selected_meaning == "оружие для стрельбы стрелами"
    assert food.selected_meaning == "огородное растение"
    assert ambiguous.unknown_kind is UnknownKind.AMBIGUOUS_CONTEXT


async def test_builtin_meaning_is_available_through_an_inflected_form(
    database: Database,
) -> None:
    service = LexiconService(database)

    analysis = await service.analyze_word("обманщика", context="ты не так понял обманщика")

    assert analysis.lemma == "обманщик"
    assert analysis.selected_meaning == (
        "человек, который намеренно вводит других в заблуждение"
    )


async def test_unique_typo_resolves_to_dictionary_lemma_without_becoming_unknown(
    database: Database,
) -> None:
    service = LexiconService(database)

    analysis = await service.analyze_word("таоелка", context="положи таоелку на стол")

    assert analysis.lemma == "тарелка"
    assert analysis.source == "opencorpora_spelling"
    assert analysis.unknown_kind is None
    assert await service.find_similar_words("таоелка") == ("тарелка",)


async def test_expressive_spelling_resolves_before_dictionary_lookup(
    database: Database,
) -> None:
    service = LexiconService(database)

    analysis = await service.analyze_word("привеееет", context="привеееет")

    assert analysis.normalized_form == "привет"
    assert analysis.lemma == "привет"
    assert analysis.unknown_kind is None


async def test_server_slang_has_priority_over_dictionary(database: Database) -> None:
    service = LexiconService(database)
    await service.register_server_slang(
        guild_id=1,
        word="суп",
        meaning="бот занят и отказывается выполнять просьбу",
        confidence=0.8,
    )

    analysis = await service.analyze_word("суп", guild_id=1, channel_id=10)

    assert analysis.source == "server_slang"
    assert analysis.local_meaning == "бот занят и отказывается выполнять просьбу"


async def test_local_dictionary_import_exposes_forms_meanings_and_style(database: Database) -> None:
    service = LexiconService(database)
    await service.import_dictionary_entry(
        lemma="краш",
        part_of_speech="NOUN",
        meanings=("аварийное завершение программы",),
        forms={"крашем": {"case": "instrumental"}},
        style_labels=("computing", "colloquial"),
    )

    analysis = await service.analyze_word("крашем")

    assert analysis.lemma == "краш"
    assert analysis.dictionary_meanings == ("аварийное завершение программы",)
    assert analysis.style_labels == ("computing", "colloquial")
    assert analysis.grammatical_features == {"case": "instrumental"}


async def test_lower_confidence_reference_does_not_replace_stronger_entry(
    database: Database,
) -> None:
    service = LexiconService(database)
    await service.import_dictionary_entry(
        lemma="ну",
        part_of_speech="INTJ",
        meanings=("контекстно неоднозначное междометие или частица",),
        source="normative_curriculum",
        confidence=0.94,
    )
    await service.import_dictionary_entry(
        lemma="ну",
        part_of_speech="PRCL",
        meanings=("одно значение из сетевого словаря",),
        source="remote_reference",
        confidence=0.76,
    )

    analysis = await service.analyze_word("ну")

    assert analysis.part_of_speech == "INTJ"
    assert analysis.dictionary_meanings == (
        "контекстно неоднозначное междометие или частица",
    )


async def test_unknown_word_is_remembered_without_invented_definition(database: Database) -> None:
    service = LexiconService(database, curiosity_threshold=0.4)
    for index in range(3):
        event = normalize_message_created(
            message_id=100 + index,
            guild_id=1,
            channel_id=10,
            author_id=20,
            created_at=NOW + timedelta(seconds=index),
            content="стромба мешает теме",
            author_is_bot=False,
            bot_mentioned=True,
        )
        await service.observe(event)

    analysis = await service.analyze_word("стромба", guild_id=1, channel_id=10)
    assert await service.ensure_study_queue(event, analysis)
    unknowns = await service.top_unknowns(1, channel_id=10)

    assert analysis.selected_meaning is None
    assert analysis.unknown_kind in {UnknownKind.UNKNOWN_FORM, UnknownKind.UNKNOWN_MEANING}
    queued = next(item for item in unknowns if item.concept_key == "стромба")
    assert queued.occurrence_count == 3
    assert await service.next_curiosity_question(1, 10, now=NOW + timedelta(minutes=1)) is None
    await database.execute(
        """UPDATE language_unknown_concepts
           SET study_status = 'exhausted', study_attempts = 4
           WHERE guild_id = 1 AND channel_id = 10 AND concept_key = 'стромба'"""
    )
    question = await service.next_curiosity_question(1, 10, now=NOW + timedelta(minutes=1))
    assert question is not None
    assert "сам несколько раз искал" in question
    assert "целой фразе" in question
    assert await service.next_curiosity_question(1, 10, now=NOW + timedelta(minutes=2)) is None


async def test_explicit_definition_replaces_a_meta_placeholder(database: Database) -> None:
    service = LexiconService(database)
    await service.teach_personal_interpretation(
        guild_id=1,
        word="клоунесса",
        interpretation="Есть отдельное значение",
        confidence=0.72,
    )
    correction = normalize_message_created(
        message_id=150,
        guild_id=1,
        channel_id=10,
        author_id=20,
        created_at=NOW,
        content=(
            "+3 Клоунесса — это артистка цирка или театра, "
            "исполняющая комические роли."
        ),
        author_is_bot=False,
        bot_mentioned=False,
    )

    await service.observe(correction)

    analysis = await service.analyze_word("клоунесса", guild_id=1, channel_id=10)
    assert analysis.personal_interpretation is not None
    assert analysis.personal_interpretation.startswith("артистка цирка или театра")
    assert analysis.personal_interpretation != "Есть отдельное значение"


async def test_neologism_stays_quarantined_until_human_repetition(database: Database) -> None:
    repository = NeologismRepository(database)
    candidate = await repository.create_candidate(
        guild_id=1,
        channel_id=10,
        word="стромбеж",
        creator_context="шуточное обсуждение повторяющейся ошибки",
        source_components=("стромба", "движ"),
        intended_meaning="хаотичное повторение одной ошибки",
        example_usage="опять начался стромбеж",
        now=NOW,
    )

    assert candidate is not None
    assert candidate.status == "experimental"
    assert await repository.record_feedback(candidate.candidate_id, used=True) == "experimental"
    await repository.record_feedback(candidate.candidate_id, human_repeat=True, positive=True)
    status = await repository.record_feedback(
        candidate.candidate_id, human_repeat=True, positive=True
    )

    assert status == "server_slang"
    analysis = await LexiconService(database).analyze_word("стромбеж", guild_id=1, channel_id=10)
    assert analysis.source == "server_slang"


async def test_neologism_has_server_and_global_response_cooldowns(database: Database) -> None:
    repository = NeologismRepository(database, min_response_interval=50)
    first = await repository.create_candidate(
        guild_id=1,
        channel_id=10,
        word="первослов",
        creator_context="creative",
        source_components=("первый", "слово"),
        intended_meaning="первый кандидат",
        example_usage="первослов появился",
        response_number=100,
        global_response_number=1000,
        now=NOW,
    )
    blocked = await repository.create_candidate(
        guild_id=1,
        channel_id=10,
        word="второслов",
        creator_context="creative",
        source_components=("второй", "слово"),
        intended_meaning="слишком ранний кандидат",
        example_usage="второслов появился",
        response_number=120,
        global_response_number=1020,
        now=NOW,
    )

    assert first is not None
    assert blocked is None
    assert await repository.cooldown_allows(1, response_number=150, global_response_number=1050)
