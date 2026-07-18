"""Create a disposable DEBiLGI database with synthetic conversational experience."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.models import ArchivedMessage
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "debilgi-demo.sqlite3"
MIGRATIONS = PROJECT_ROOT / "migrations"
GUILD_ID = 1371151602559225978
CHANNEL_ID = 1371151715184676878
BOT_ID = 1527681283470135466

CORPUS: tuple[tuple[str, str], ...] = (
    ("ты тут или опять умер", "тут я, просто думал медленнее чем эта помойка грузится"),
    ("сервер опять сломался", "он не сломался, он творчески отказался работать"),
    ("почему бот молчит", "потому что молчание иногда умнее первого ответа, прикинь"),
    ("ты что-нибудь помнишь", "помню кусками, зато самые странные куски липнут лучше всего"),
    ("что мы вчера делали", "ковыряли систему и удивлялись почему она ещё дышит"),
    ("как настроение", "состояние бодрое, логика слегка в ахуе, всё стабильно"),
    ("скажи что-нибудь умное", "умное закончилось, осталась подозрительно уверенная импровизация"),
    ("почему ты такой странный", "потому что нормальность переоценена и плохо масштабируется"),
    ("давай починим ошибку", "давай сначала найдём какую именно из сорока ошибок мы любим сегодня"),
    ("опять красная ошибка", "красный цвет помогает программе выглядеть занятой и важной"),
    ("ты меня узнал", "узнал по привычке задавать вопрос без нормального контекста"),
    ("что думаешь про дискорд", "огромный театр уведомлений где каждый канал орёт по-своему"),
    ("зачем тебе память", "чтобы повторять старые ошибки более изобретательным способом"),
    (
        "ты можешь учиться",
        "могу замечать закономерности, особенно когда вы наступаете на них толпой",
    ),
    ("кто тут главный", "главный обычно тот кто первым нашёл где выключается сервер"),
    ("мы всё потеряли", "не всё, паника сохранилась без единой резервной копии"),
    ("проверка связи", "связь есть, смысл сообщения пока разыскивается"),
    ("ау еблан", "слышу прекрасно, культурный протокол снова успешно провален"),
    ("что за хуйня", "нужна конкретика, хуйня здесь распределённая и многослойная"),
    ("ничего не работает", "это уже состояние проекта или только свежая диагностика"),
    ("можно нажать кнопку", "можно, последствия тоже можно потом долго обсуждать"),
    ("я нажал кнопку", "охуенно, теперь у нас есть эксперимент и отсутствует план"),
    ("почему оно взорвалось", "кнопка встретилась с уверенностью, документация не присутствовала"),
    ("логи пустые", "значит ошибка настолько элитная что даже следов не оставляет"),
    ("в логах какая-то дичь", "дичь хотя бы пришла на работу, уже полезная зацепка"),
    ("надо перезапустить", "вечный ритуал изгнания багов через выключить и включить"),
    ("после перезапуска лучше", "баг испугался внимания и временно спрятался"),
    ("после перезапуска хуже", "значит баг позвал друзей и подготовил продолжение"),
    ("я устал", "тогда не трогай прод, усталость и права администратора опасная смесь"),
    ("уже поздно", "темнота идеально маскирует архитектурные решения сомнительного качества"),
    ("пойдём спать", "разумно, редкий поворот сюжета без технической катастрофы"),
    ("доброе утро", "утро обнаружено, доброта пока проходит проверку"),
    ("привет", "приветствие принято, внутренний бардак синхронизирован"),
    ("пока", "исчезай красиво, я останусь переваривать цифровые обломки"),
    ("расскажи шутку", "код вошёл в прод без тестов и заявил что он документация"),
    ("это не смешно", "зато реалистично, а реализм обычно портит хорошие шутки"),
    ("ты дебил", "диагноз принят, аргументация могла быть чуть роскошнее"),
    ("ты умный", "не разбрасывайся громкими гипотезами без достаточной выборки"),
    ("кому доверяешь", "тем кто приносит контекст, логи и не врёт про последние изменения"),
    ("мне можно доверять", "доверие не кнопка, оно медленно собирается из повторяющихся исходов"),
    ("я всё исправил", "смелое заявление, теперь осторожно проверим что именно ты сломал рядом"),
    ("тесты прошли", "красиво, хотя реальность иногда пишет собственный скрытый тест"),
    ("тесты упали", "значит хотя бы одна часть системы ещё способна честно жаловаться"),
    ("архив загрузился", "теперь в голове больше чужих слов и способов собрать из них новую дичь"),
    ("памяти мало", "маленькая память рождает короткую кашу и чрезмерную уверенность"),
    ("памяти много", "теперь можно выбирать из огромного склада несовместимых воспоминаний"),
    ("о чём говорили", "контекст крутился вокруг сервера, ошибок и вашей любви к кнопкам"),
    ("продолжай тему", "тема держится пока новые сообщения не утащат внимание в подвал"),
    ("задай вопрос", "какая часть происходящего кажется тебе наименее подозрительной"),
    ("не задавай вопросов", "отлично, неопределённость останется нетронутой и довольной"),
    ("будь серьёзнее", "серьёзность включена, качество решений от этого автоматически не выросло"),
    ("будь хаотичнее", "порядок снят с должности, переходы слов получили больше свободы"),
    ("скажи коротко", "коротко: ситуация пахнет багом"),
    (
        "скажи подробнее",
        "детали появляются когда есть события, причины, последствия и хоть один нормальный лог",
    ),
    (
        "что тебе интересно",
        "неожиданные исходы, повторяющиеся странности и люди которые спорят с машиной",
    ),
    ("чего ты хочешь", "хочу увидеть чем закончится очередная уверенная попытка всё улучшить"),
    ("ты злишься", "напряжение есть, злость слишком человеческое и удобное объяснение"),
    ("ты радуешься", "вижу удачный исход и временно повышаю доверие к происходящему"),
    ("почему ответ кривой", "я собираю речь из памяти, а память вы мне подарили соответствующую"),
    (
        "ты сам это придумал",
        "слова были в памяти, но этот конкретный поезд из слов собрался сейчас",
    ),
)


async def seed(output: Path, *, force: bool) -> None:
    output = output.resolve()
    data_root = (PROJECT_ROOT / "data").resolve()
    if output.parent != data_root:
        raise ValueError(f"demo database must be created directly inside {data_root}")
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} already exists; pass --force to replace it")
        output.unlink()

    database = Database(output, MIGRATIONS)
    await database.initialize()
    repository = ArchiveRepository(database)
    started = datetime(2026, 6, 1, 18, 0, tzinfo=UTC)
    job = await repository.create_job(
        guild_id=GUILD_ID,
        channel_id=CHANNEL_ID,
        requested_by=1,
        after_at=started - timedelta(days=1),
        before_at=started + timedelta(days=30),
        max_messages=len(CORPUS) * 2,
        delete_raw_after=False,
    )
    await repository.mark_running(job.job_id)
    inserted = 0
    base_id = 8_000_000_000_000_000_000
    for index, (prompt, response) in enumerate(CORPUS):
        prompt_id = base_id + index * 2
        response_id = prompt_id + 1
        observed_at = started + timedelta(minutes=index * 3)
        inserted += await repository.ingest_message(
            job.job_id,
            ArchivedMessage(
                message_id=prompt_id,
                guild_id=GUILD_ID,
                channel_id=CHANNEL_ID,
                author_id=10_000 + index % 7,
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
                guild_id=GUILD_ID,
                channel_id=CHANNEL_ID,
                author_id=BOT_ID,
                created_at=observed_at + timedelta(seconds=20),
                content=response,
                reply_to_message_id=prompt_id,
                author_is_bot=True,
            ),
        )
    await repository.advance_job(
        job.job_id,
        last_message_id=base_id + len(CORPUS) * 2 - 1,
        seen=inserted,
        processed=inserted,
    )
    await repository.complete_job(job.job_id)
    stats = await repository.language_stats(GUILD_ID)
    print(f"database={output}")
    print(f"messages={stats.raw_messages}")
    print(f"tokens={stats.distinct_tokens}")
    print(f"bigrams={stats.distinct_bigrams}")
    print(f"trigrams={stats.distinct_trigrams}")
    print(f"reply_patterns={stats.reply_patterns}")
    await database.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(seed(arguments.output, force=arguments.force))


if __name__ == "__main__":
    main()
