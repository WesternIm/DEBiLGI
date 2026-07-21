"""Pretrain the active local database to use the symbolic language-layer capabilities."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from debilgi.cognition.models import ActionType
from debilgi.config import Settings
from debilgi.events.normalizer import normalize_message_created
from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.character_markov import CharacterMarkovConfig, CharacterMarkovRepository
from debilgi.language.conversation import ConversationRepository
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.language.language_layer_curriculum import SCENES, total_turns
from debilgi.language.lexicon import LexiconService
from debilgi.language.memory_based import MemoryPhraseGenerator
from debilgi.language.models import ArchivedMessage
from debilgi.language.planning import ReplyContext, StructuredThought, UtterancePlan
from debilgi.language.semantics import SemanticMemoryRepository
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database

DEFAULT_GUILD_ID = 1_371_151_602_559_225_978
CHANNEL_ID = 1_371_151_715_184_676_878
EPISODE_CHANNEL_BASE = 1_371_151_715_185_000_000
BOT_ID = 1_527_681_283_470_135_466
TEACHER_ID = 9_300_000_000_000_001
BASE_MESSAGE_ID = 8_300_000_000_000_000_000
STARTED_AT = datetime(2026, 7, 18, 15, 0, tzinfo=UTC)

DICTIONARY_LESSONS: tuple[
    tuple[str, str, tuple[str, ...], dict[str, dict[str, str]], tuple[str, ...]], ...
] = (
    (
        "словарь",
        "NOUN",
        ("собрание слов с пояснениями или переводом",),
        {"словарём": {"case": "instrumental"}, "словаря": {"case": "genitive"}},
        ("linguistics",),
    ),
    (
        "контекст",
        "NOUN",
        ("окружение высказывания, необходимое для понимания его смысла",),
        {"контексте": {"case": "prepositional"}, "контекстом": {"case": "instrumental"}},
        ("linguistics",),
    ),
    (
        "интерпретация",
        "NOUN",
        ("раскрытие или объяснение смысла",),
        {"интерпретации": {"case": "genitive"}},
        ("semantics",),
    ),
    (
        "неологизм",
        "NOUN",
        ("новое слово или новое значение существующего слова",),
        {"неологизма": {"case": "genitive"}, "неологизмы": {"number": "plural"}},
        ("linguistics",),
    ),
    (
        "многозначность",
        "NOUN",
        ("наличие у языковой единицы нескольких связанных значений",),
        {"многозначности": {"case": "genitive"}},
        ("linguistics",),
    ),
    (
        "уверенность",
        "NOUN",
        ("степень отсутствия сомнения в оценке или выводе",),
        {"уверенности": {"case": "genitive"}},
        ("cognition",),
    ),
    (
        "любопытство",
        "NOUN",
        ("стремление узнать или понять неизвестное",),
        {"любопытства": {"case": "genitive"}},
        ("cognition",),
    ),
    (
        "разговор",
        "NOUN",
        ("словесное общение между участниками",),
        {"разговора": {"case": "genitive"}, "разговоре": {"case": "prepositional"}},
        ("communication",),
    ),
    (
        "причина",
        "NOUN",
        ("обстоятельство, вызывающее другое явление или следствие",),
        {"причины": {"case": "genitive"}, "причиной": {"case": "instrumental"}},
        ("logic",),
    ),
    (
        "вывод",
        "NOUN",
        ("результат рассуждения, полученный из наблюдений или посылок",),
        {"вывода": {"case": "genitive"}, "выводы": {"number": "plural"}},
        ("logic",),
    ),
)

PERSONAL_LESSONS = (
    (
        "словарь",
        "эталон человеческого языка, с которым можно сравнивать личный опыт и локальный сленг",
        ("эталон", "понятность", "значение", "сленг"),
    ),
    (
        "неологизм",
        "редкая экспериментальная форма, которая сначала проходит карантин и проверку людьми",
        ("новое слово", "карантин", "повторение", "реакция"),
    ),
    (
        "любопытство",
        "ограниченное желание закрыть важный пробел знания, не разрушая активный разговор",
        ("неизвестность", "вопрос", "важность", "время ожидания"),
    ),
    (
        "разговор",
        "развивающаяся история с темой, причинами, позициями, выводами и вопросами",
        ("тема", "причина", "вывод", "участники"),
    ),
)

PROBES = (
    "зачем нужен эталонный словарь",
    "что делать с неизвестным словом",
    "какое значение имеет лук без контекста",
    "можно ли задавать вопросы подряд",
    "кто определяет смысл ответа",
    "когда новое слово становится сленгом",
    "что сохраняется после завершения разговора",
)


async def train(*, evaluate_only: bool = False) -> None:
    settings = Settings.from_env()
    database_path = settings.database_path.resolve()
    guild_id = settings.dev_guild_id or DEFAULT_GUILD_ID
    backup_path = None if evaluate_only else _backup(database_path)
    character_config = CharacterMarkovConfig(
        min_order=settings.character_markov_min_order,
        max_order=settings.character_markov_max_order,
        max_word_length=settings.character_markov_max_word_length,
        generation_attempts=settings.character_markov_generation_attempts,
    )
    database = Database(database_path, settings.migrations_path)
    await database.initialize()
    try:
        archive = ArchiveRepository(database, character_markov_config=character_config)
        character_markov = CharacterMarkovRepository(database, character_config)
        discourse = DiscourseMemoryRepository(database)
        semantic = SemanticMemoryRepository(database)
        lexicon = LexiconService(database)
        conversations = ConversationRepository(
            database,
            idle_timeout=timedelta(minutes=settings.conversation_idle_timeout_minutes),
            max_idle_timeout=timedelta(minutes=settings.conversation_max_idle_timeout_minutes),
        )

        last_message_id = BASE_MESSAGE_ID + total_turns() - 1
        complete = await database.fetchone(
            "SELECT 1 FROM archive_messages WHERE message_id = ?", (last_message_id,)
        )
        if not evaluate_only:
            await _install_dictionary(lexicon, guild_id)
            if complete is None:
                print(f"language_training_messages={await _ingest(archive, guild_id)}")
            else:
                print("language_training_messages=already_present")
            episodes_added = await _build_episodes(database, conversations, guild_id)
            print(f"conversation_episodes_added={episodes_added}")
            print(f"semantic_rebuilt={await semantic.rebuild_archive(guild_id)}")
            print(f"discourse_rebuilt={await discourse.rebuild_archive(guild_id)}")
        elif complete is None:
            print("language_training_messages=not_present")
        else:
            print("language_training_messages=already_present")

        await _report(
            database,
            archive,
            character_markov,
            discourse,
            semantic,
            lexicon,
            guild_id,
            settings,
            backup_path,
        )
    finally:
        await database.close()


async def _install_dictionary(lexicon: LexiconService, guild_id: int) -> None:
    for lemma, part, meanings, forms, labels in DICTIONARY_LESSONS:
        await lexicon.import_dictionary_entry(
            lemma=lemma,
            part_of_speech=part,
            meanings=meanings,
            forms={lemma: {}, **forms},
            style_labels=labels,
            source="language_layer_curriculum",
            confidence=0.94,
        )
    for word, interpretation, associations in PERSONAL_LESSONS:
        await lexicon.teach_personal_interpretation(
            guild_id=guild_id,
            word=word,
            interpretation=interpretation,
            associations=associations,
            confidence=0.72,
            source_event_ids=("curriculum:language-layer",),
            now=STARTED_AT,
        )


async def _ingest(repository: ArchiveRepository, guild_id: int) -> int:
    job = await repository.create_job(
        guild_id=guild_id,
        channel_id=CHANNEL_ID,
        requested_by=TEACHER_ID,
        after_at=STARTED_AT - timedelta(days=1),
        before_at=STARTED_AT + timedelta(days=2),
        max_messages=total_turns(),
        delete_raw_after=False,
    )
    await repository.mark_running(job.job_id)
    inserted = 0
    offset = 0
    for scene in SCENES:
        previous_id: int | None = None
        for turn in scene.turns:
            message_id = BASE_MESSAGE_ID + offset
            observed_at = STARTED_AT + timedelta(minutes=offset)
            is_bot = turn.speaker == "bot"
            inserted += await repository.ingest_message(
                job.job_id,
                ArchivedMessage(
                    message_id=message_id,
                    guild_id=guild_id,
                    channel_id=CHANNEL_ID,
                    author_id=BOT_ID if is_bot else TEACHER_ID,
                    created_at=observed_at,
                    content=turn.text,
                    reply_to_message_id=previous_id,
                    author_is_bot=is_bot,
                ),
            )
            previous_id = message_id
            offset += 1
    await repository.advance_job(
        job.job_id,
        last_message_id=BASE_MESSAGE_ID + total_turns() - 1,
        seen=total_turns(),
        processed=inserted,
    )
    await repository.complete_job(job.job_id)
    return inserted


async def _build_episodes(
    database: Database,
    conversations: ConversationRepository,
    guild_id: int,
) -> int:
    added = 0
    offset = 0
    for scene_index, scene in enumerate(SCENES):
        first_id = BASE_MESSAGE_ID + offset
        existing = await database.fetchone(
            """SELECT 1 FROM language_conversation_episodes AS episode,
                      json_each(episode.important_message_ids_json) AS message
               WHERE episode.guild_id = ? AND CAST(message.value AS INTEGER) = ? LIMIT 1""",
            (guild_id, first_id),
        )
        channel_id = EPISODE_CHANNEL_BASE + scene_index
        previous_id: int | None = None
        for turn in scene.turns:
            if existing is None:
                message_id = BASE_MESSAGE_ID + offset
                is_bot = turn.speaker == "bot"
                await conversations.observe(
                    normalize_message_created(
                        message_id=message_id,
                        guild_id=guild_id,
                        channel_id=channel_id,
                        author_id=BOT_ID if is_bot else TEACHER_ID,
                        created_at=STARTED_AT + timedelta(minutes=offset),
                        content=turn.text,
                        author_is_bot=is_bot,
                        reply_to_message_id=previous_id,
                    )
                )
                previous_id = message_id
            offset += 1
        if existing is None:
            episode = await conversations.close_channel(
                guild_id,
                channel_id,
                now=STARTED_AT + timedelta(minutes=offset, seconds=1),
            )
            added += int(episode is not None)
    return added


async def _report(
    database: Database,
    archive: ArchiveRepository,
    character_markov: CharacterMarkovRepository,
    discourse: DiscourseMemoryRepository,
    semantic: SemanticMemoryRepository,
    lexicon: LexiconService,
    guild_id: int,
    settings: Settings,
    backup_path: Path | None,
) -> None:
    stats = await archive.language_stats(guild_id)
    discourse_stats = await discourse.stats(guild_id)
    skills = Counter(skill for scene in SCENES for skill in scene.skills)
    dictionary_row = await database.fetchone(
        "SELECT COUNT(*) AS count FROM language_dictionary_entries"
    )
    personal_row = await database.fetchone(
        """SELECT COUNT(*) AS count FROM language_personal_interpretations
           WHERE guild_id = ?""",
        (guild_id,),
    )
    character_row = await database.fetchone(
        """SELECT COUNT(*) AS count FROM language_character_transitions
           WHERE guild_id = ?""",
        (guild_id,),
    )
    episode_row = await database.fetchone(
        """SELECT COUNT(*) AS count FROM language_conversation_episodes
           WHERE guild_id = ? AND channel_id >= ? AND channel_id < ?""",
        (guild_id, EPISODE_CHANNEL_BASE, EPISODE_CHANNEL_BASE + len(SCENES)),
    )
    print(f"backup={backup_path}")
    print(f"schema={await database.schema_version()}")
    print(f"language_scenes={len(SCENES)}")
    print(f"language_turns={total_turns()}")
    print(f"language_skills={len(skills)}")
    print(f"raw_messages={stats.raw_messages}")
    print(f"reply_patterns={stats.reply_patterns}")
    print(f"semantic_frames={stats.semantic_frames}")
    print(f"discourse_turns={discourse_stats.turns}")
    print(f"dictionary_entries={int(dictionary_row['count']) if dictionary_row else 0}")
    print(f"personal_interpretations={int(personal_row['count']) if personal_row else 0}")
    print(f"character_transitions={int(character_row['count']) if character_row else 0}")
    print(f"language_episodes={int(episode_row['count']) if episode_row else 0}")

    print("\nword analysis:")
    for word, context in (
        ("словарём", "проверить значение словарём"),
        ("лук", "лучник натянул лук и выпустил стрелу"),
        ("лук", "повар нарезал лук в салат"),
    ):
        analysis = await lexicon.analyze_word(word, guild_id=guild_id, context=context)
        print(
            f"- {word}: lemma={analysis.lemma} meaning={analysis.selected_meaning} "
            f"source={analysis.source} response_confidence={analysis.response_confidence:.2f}"
        )

    model = await character_markov.model(guild_id)
    print(f"character_plausibility[неологизм]={model.plausibility('неологизм'):.3f}")
    print("\ngenerated replies:")
    for index, probe in enumerate(PROBES):
        print(
            f"- {probe} -> {await _generate(archive, semantic, guild_id, probe, index, settings)}"
        )


async def _generate(
    archive: ArchiveRepository,
    semantic: SemanticMemoryRepository,
    guild_id: int,
    prompt: str,
    index: int,
    settings: Settings,
) -> str:
    tokens = tokenize(prompt)
    context = ReplyContext(
        prompt=prompt,
        memory_messages=await archive.recent_texts(guild_id, CHANNEL_ID, 200),
        memory_ngrams=await archive.common_ngrams(guild_id),
        reply_heads=await archive.matching_reply_heads(guild_id, tokens),
        knowledge_facts=(
            *(await archive.matching_replies(guild_id, tokens)),
            *(await semantic.context(guild_id, tokens, limit=6)),
        ),
        exact_answer=await archive.exact_learned_reply(guild_id, tokens),
        seed=f"language-layer-probe:{index}",
    )
    plan = UtterancePlan(
        speech_act="brief_statement",
        max_tokens=settings.language_max_output_tokens,
        thought=StructuredThought(ActionType.SHORT_REPLY, None, "explain", 0.82, ()),
        context=context,
    )
    return MemoryPhraseGenerator(
        max_output_tokens=settings.language_max_output_tokens,
        max_output_chars=settings.language_max_output_chars,
    ).generate(plan)


def _backup(database_path: Path) -> Path | None:
    if not database_path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup = database_path.with_name(f"{database_path.stem}-before-language-layer-{stamp}.sqlite3")
    with sqlite3.connect(database_path) as source, sqlite3.connect(backup) as target:
        source.backup(target)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-only", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(train(evaluate_only=arguments.evaluate_only))


if __name__ == "__main__":
    main()
