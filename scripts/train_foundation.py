"""Teach the active local database basic actors, actions, objects and consequences."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from debilgi.cognition.models import ActionType
from debilgi.config import Settings
from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.language.memory_based import MemoryPhraseGenerator
from debilgi.language.models import ArchivedMessage
from debilgi.language.planning import ReplyContext, StructuredThought, UtterancePlan
from debilgi.language.semantics import SemanticMemoryRepository
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database

DEFAULT_GUILD_ID = 1371151602559225978
CHANNEL_ID = 1371151715184676878
BOT_ID = 1527681283470135466
TEACHER_ID = 9_100_000_000_000_001
BASE_MESSAGE_ID = 8_100_000_000_000_000_000

# These are observations and replies used as memory, never runtime response templates.
LESSONS: tuple[tuple[str, str], ...] = (
    ("Лиса взяла медный ключ со стола.", "Теперь медный ключ находится у лисы."),
    ("Лиса положила медный ключ в деревянный ящик.", "Теперь ключ лежит внутри ящика."),
    ("Где находится медный ключ?", "Медный ключ находится в деревянном ящике."),
    ("Кто взял медный ключ?", "Медный ключ взяла лиса."),
    ("Ворон открыл деревянный ящик.", "После действия ворона ящик стал открытым."),
    ("Ворон достал медный ключ из ящика.", "После этого ключ находится у ворона."),
    ("Кот толкнул синюю чашку лапой.", "Синяя чашка упала со стола."),
    ("Почему упала синяя чашка?", "Чашка упала, потому что кот толкнул её лапой."),
    ("Кот не трогал красную чашку.", "Красная чашка осталась на столе."),
    ("Какую чашку толкнул кот?", "Кот толкнул синюю чашку."),
    ("Робот нажал зелёную кнопку на панели.", "После нажатия зелёной кнопки лампа загорелась."),
    ("Зелёная кнопка включает лампу.", "Лампа получает свет после нажатия кнопки."),
    ("Робот нажал красную кнопку.", "Красная кнопка выключила лампу."),
    ("Что изменилось после красной кнопки?", "После красной кнопки лампа перестала светиться."),
    ("Лампа стояла над дверью.", "Дверь находилась под лампой."),
    ("Когда лампа погасла, коридор стал тёмным.", "Темнота появилась после выключения лампы."),
    ("Мира открыла дверь железным ключом.", "Железный ключ подошёл к замку двери."),
    ("Мира вошла в комнату через открытую дверь.", "После входа Мира находится внутри комнаты."),
    ("Дверь закрылась за Мирой.", "Теперь дверь закрыта, а Мира остаётся в комнате."),
    ("Мира не выходила из комнаты.", "Значит Мира всё ещё находится внутри комнаты."),
    ("Камень тяжелее листа.", "Лист легче камня."),
    ("Ветер поднял лёгкий лист.", "Лист полетел над землёй из-за ветра."),
    ("Ветер не поднял тяжёлый камень.", "Камень остался лежать на земле."),
    ("Что может поднять ветер?", "Ветер может поднять лёгкий лист."),
    ("Собака принесла мяч хозяину.", "После этого мяч оказался у хозяина."),
    ("Хозяин бросил мяч во двор.", "Мяч полетел от хозяина во двор."),
    ("Собака побежала за мячом.", "Собака ищет объект, который бросил хозяин."),
    ("Собака нашла мяч возле дерева.", "Теперь собака и мяч находятся возле дерева."),
    ("Аня дала книгу Борису.", "После передачи книга принадлежит Борису."),
    ("Борис положил книгу на полку.", "Книга находится на полке."),
    ("Борис взял книгу с полки.", "Книга больше не находится на полке."),
    ("Кому Аня дала книгу?", "Аня передала книгу Борису."),
    ("Дождь намочил сухую дорогу.", "После дождя дорога стала мокрой."),
    ("Солнце нагрело мокрую дорогу.", "Тёплая дорога начала высыхать."),
    ("Дорога высохла после солнца.", "Состояние дороги изменилось с мокрого на сухое."),
    ("Почему дорога стала мокрой?", "Дорога стала мокрой из-за дождя."),
    ("Сначала Ника собрала детали.", "Сбор деталей произошёл до сборки механизма."),
    ("Потом Ника собрала механизм из деталей.", "Готовый механизм появился после сборки."),
    ("Ника включила готовый механизм.", "Механизм начал работать после включения."),
    ("Механизм остановился без энергии.", "Отсутствие энергии остановило работу механизма."),
    (
        "Что было раньше: детали или механизм?",
        "Сначала существовали детали, потом появился механизм.",
    ),
    ("Птица построила гнездо на высокой ветке.", "Гнездо находится на ветке дерева."),
    ("Птица принесла ветку в гнездо.", "Ветка стала частью гнезда."),
    ("Птенец сидит внутри гнезда.", "Гнездо окружает и защищает птенца."),
    ("Река течёт через долину.", "Вода движется по руслу реки через долину."),
    ("Мост соединяет два берега над рекой.", "По мосту можно перейти с одного берега на другой."),
    ("Машина проехала по мосту.", "Машина переместилась на другой берег."),
    ("Если мост закрыт, машина не проезжает.", "Закрытый мост препятствует движению машины."),
    ("Какой ключ находится у ворона?", "Медный ключ находится у ворона."),
    ("Что происходит без энергии?", "Без энергии готовый механизм останавливается."),
    ("Ты согласен с выводом?", "Да, вывод совпадает с наблюдениями."),
    ("Ты не согласен?", "Нет, вывод противоречит наблюдениям."),
    ("Тебе нравится результат?", "Да, результат хороший и заслуживает одобрения."),
    ("Это плохой результат?", "Результат плохой, одобрения нет."),
    ("Ты будешь это делать?", "Нет, я отказываюсь это делать."),
    ("Почему ты отказался?", "Отказ появился из-за сильного сопротивления."),
    ("Тебе интересно, что произошло?", "Да, интересно понять причину."),
    ("Что вызывает любопытство?", "Неизвестная причина усиливает интерес."),
    ("Тебе противно?", "Фу, это мерзко и вызывает отвращение."),
    (
        "Может быть мерзко и интересно одновременно?",
        "Да, любопытство может сочетаться с отвращением.",
    ),
    (
        "Можно одобрять и сомневаться одновременно?",
        "Да, одобрение может сочетаться с настороженностью.",
    ),
    ("Ты доверяешь этому?", "Доверие растёт, когда исходы повторяются."),
    (
        "Можно согласиться, но сопротивляться?",
        "Согласие и сопротивление могут существовать одновременно.",
    ),
    (
        "Как рождается новая реакция?",
        "Новая реакция складывается из нескольких базовых сил.",
    ),
)

PROBES = (
    "где сейчас медный ключ",
    "почему упала синяя чашка",
    "что сделала красная кнопка",
    "где осталась Мира",
    "кому принадлежит книга",
    "что случается без энергии",
)


async def train(*, evaluate_only: bool = False) -> None:
    settings = Settings.from_env()
    database_path = settings.database_path.resolve()
    guild_id = settings.dev_guild_id or DEFAULT_GUILD_ID
    backup_path = None if evaluate_only else _backup(database_path)

    database = Database(database_path, settings.migrations_path)
    await database.initialize()
    try:
        repository = ArchiveRepository(database)
        discourse = DiscourseMemoryRepository(database)
        semantic = SemanticMemoryRepository(database)
        already = await database.fetchone(
            "SELECT 1 FROM archive_messages WHERE message_id = ?", (BASE_MESSAGE_ID,)
        )
        last_message_id = BASE_MESSAGE_ID + len(LESSONS) * 2 - 1
        complete = await database.fetchone(
            "SELECT 1 FROM archive_messages WHERE message_id = ?", (last_message_id,)
        )
        if complete is None and not evaluate_only:
            inserted = await _ingest(repository, guild_id)
            print(f"training_messages={inserted}")
        elif already is None:
            print("training_messages=not_present")
        else:
            print("training_messages=already_present")

        if not evaluate_only:
            rebuilt = await semantic.rebuild_archive(guild_id)
            print(f"semantic_rebuilt={rebuilt}")
            discourse_rebuilt = await discourse.rebuild_archive(guild_id)
            print(f"discourse_rebuilt={discourse_rebuilt}")

        stats = await repository.language_stats(guild_id)
        discourse_stats = await discourse.stats(guild_id)
        print(f"backup={backup_path}")
        print(f"schema={await database.schema_version()}")
        print(f"raw_messages={stats.raw_messages}")
        print(f"semantic_frames={stats.semantic_frames}")
        print(f"discourse_turns={discourse_stats.turns}")
        print(f"open_questions={discourse_stats.open_questions}")
        print(f"answered_questions={discourse_stats.answered_questions}")
        print("\nsemantic recall:")
        for probe in PROBES:
            facts = await semantic.context(guild_id, tokenize(probe), limit=3)
            print(f"- {probe}: {' | '.join(facts)}")

        print("\ngenerated replies:")
        for index, probe in enumerate(PROBES):
            reply = await _generate(repository, semantic, guild_id, probe, index, settings)
            print(f"- {probe} -> {reply}")
    finally:
        await database.close()


async def _ingest(repository: ArchiveRepository, guild_id: int) -> int:
    started = datetime(2026, 7, 18, 9, 0, tzinfo=UTC)
    job = await repository.create_job(
        guild_id=guild_id,
        channel_id=CHANNEL_ID,
        requested_by=TEACHER_ID,
        after_at=started - timedelta(days=1),
        before_at=started + timedelta(days=2),
        max_messages=len(LESSONS) * 2,
        delete_raw_after=False,
    )
    await repository.mark_running(job.job_id)
    inserted = 0
    for index, (prompt, response) in enumerate(LESSONS):
        prompt_id = BASE_MESSAGE_ID + index * 2
        response_id = prompt_id + 1
        observed_at = started + timedelta(minutes=index * 2)
        inserted += await repository.ingest_message(
            job.job_id,
            ArchivedMessage(
                message_id=prompt_id,
                guild_id=guild_id,
                channel_id=CHANNEL_ID,
                author_id=TEACHER_ID,
                created_at=observed_at,
                content=prompt,
                reply_to_message_id=None,
                author_is_bot=False,
            ),
        )
        inserted += await repository.ingest_message(
            job.job_id,
            ArchivedMessage(
                message_id=response_id,
                guild_id=guild_id,
                channel_id=CHANNEL_ID,
                author_id=BOT_ID,
                created_at=observed_at + timedelta(seconds=15),
                content=response,
                reply_to_message_id=prompt_id,
                author_is_bot=True,
            ),
        )
    await repository.advance_job(
        job.job_id,
        last_message_id=BASE_MESSAGE_ID + len(LESSONS) * 2 - 1,
        seen=inserted,
        processed=inserted,
    )
    await repository.complete_job(job.job_id)
    return inserted


async def _generate(
    repository: ArchiveRepository,
    semantic: SemanticMemoryRepository,
    guild_id: int,
    prompt: str,
    index: int,
    settings: Settings,
) -> str:
    tokens = tokenize(prompt)
    context = ReplyContext(
        prompt=prompt,
        memory_messages=await repository.recent_texts(guild_id, CHANNEL_ID, 120),
        memory_ngrams=await repository.common_ngrams(guild_id),
        reply_heads=await repository.matching_reply_heads(guild_id, tokens),
        knowledge_facts=await semantic.context(guild_id, tokens),
        seed=f"foundation-probe:{index}",
    )
    plan = UtterancePlan(
        speech_act="brief_statement",
        max_tokens=settings.language_max_output_tokens,
        thought=StructuredThought(ActionType.SHORT_REPLY, None, "recall", 0.8, ()),
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
    backup = database_path.with_name(f"{database_path.stem}-before-foundation-{stamp}.sqlite3")
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
