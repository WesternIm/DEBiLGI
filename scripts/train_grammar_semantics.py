# ruff: noqa: E501
"""Train the active database on Russian grammar and contextual lexical semantics."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from debilgi.config import Settings
from debilgi.events.normalizer import normalize_message_created
from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.language.grammar_semantics_curriculum import SCENES, covered_skills, total_turns
from debilgi.language.lexicon import LexiconService
from debilgi.language.models import ArchivedMessage
from debilgi.language.morphology import MorphologyRepository, RussianMorphology
from debilgi.language.semantics import SemanticMemoryRepository
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database

DEFAULT_GUILD_ID = 1_371_151_602_559_225_978
CHANNEL_ID = 1_371_151_715_184_676_879
BOT_ID = 1_527_681_283_470_135_466
TEACHER_ID = 9_300_000_000_000_002
BASE_MESSAGE_ID = 8_400_000_000_000_000_000
STARTED_AT = datetime(2026, 7, 18, 18, 0, tzinfo=UTC)

SENSES = (
    ("ключ", "предмет для отпирания замка", ("замок", "дверь", "открыть", "отпереть"), ("отмычка",), "ключ"),
    ("ключ", "родник, выход подземной воды", ("вода", "родник", "земля", "лес", "пить"), ("родник", "источник"), "ключ"),
    ("коса", "заплетённые вместе длинные волосы", ("волосы", "девушка", "заплести", "причёска"), ("причёска",), "коса"),
    ("коса", "инструмент для срезания травы", ("трава", "поле", "косить", "точить"), ("косилка",), "коса"),
    ("коса", "узкая полоса суши у берега", ("берег", "море", "песок", "суша"), ("отмель",), "коса"),
    ("лук", "оружие для стрельбы стрелами", ("стрела", "тетива", "лучник", "выстрел"), ("оружие",), "лук"),
    ("лук", "огородное растение, употребляемое в пищу", ("салат", "овощ", "суп", "повар"), ("овощ",), "лук"),
    ("мир", "состояние без войны", ("война", "договор", "страна", "перемирие"), ("перемирие",), "мир"),
    ("мир", "совокупность окружающей действительности", ("планета", "вселенная", "жизнь", "общество"), ("вселенная",), "мир"),
    ("быстрый", "движущийся или происходящий с большой скоростью", ("скорость", "движение", "время"), ("скорый", "стремительный"), None),
    ("говорить", "выражать мысли словами", ("речь", "слово", "разговор"), ("сказать", "произносить"), None),
    ("ошибка", "неправильность в действии, мысли или результате", ("неверно", "исправить", "проверить"), ("промах", "неточность"), None),
    ("огонь", "горящие газы, выделяющие свет и тепло", ("пламя", "тепло", "гореть"), ("пламя",), None),
)

FUNCTION_WORDS = (
    ("не", "PRCL", "частица отрицания", ("negation",)),
    ("ни", "PRCL", "частица усиленного отрицания или обобщения", ("negation",)),
    ("бы", "PRCL", "частица условности, желательности или смягчения", ("modality",)),
    ("же", "PRCL", "частица смыслового выделения или напоминания известного", ("focus",)),
    ("ли", "PRCL", "вопросительная частица или показатель сомнения", ("question",)),
    ("разве", "PRCL", "частица вопроса, сомнения или возражения", ("question",)),
    ("неужели", "PRCL", "частица удивлённого вопроса или недоверия", ("question",)),
    ("только", "PRCL", "ограничительная или выделительная частица", ("restriction",)),
    ("именно", "PRCL", "частица точного смыслового выделения", ("focus",)),
    ("ведь", "PRCL", "частица напоминания основания, известного собеседникам", ("reason",)),
    ("ну", "INTJ", "междометие или частица побуждения, реакции и управления репликой", ("colloquial", "ambiguous")),
    ("ой", "INTJ", "междометие испуга, боли, неожиданности или неловкости", ("emotion",)),
    ("ах", "INTJ", "междометие удивления, восхищения или сожаления", ("emotion",)),
    ("увы", "INTJ", "междометие сожаления", ("emotion",)),
    ("фу", "INTJ", "междометие отвращения или резкого неодобрения", ("emotion",)),
    ("ого", "INTJ", "междометие сильного удивления или восхищения", ("emotion",)),
    ("хм", "INTJ", "междометие раздумья, сомнения или заминки", ("emotion",)),
    ("мда", "INTJ", "разговорное междометие скепсиса или разочарования", ("colloquial",)),
)

PROBES = (
    "Какой падеж нужен после без?",
    "Исправь: девушка прочитал сообщение.",
    "Чем различаются быстрый и стремительный?",
    "Я открыл дверь ключом.",
    "Из-под камня бил холодный ключ.",
    "Я увидел косу. Что это значит?",
    "Локальный смысл заменит словарный?",
)


async def train(*, evaluate_only: bool = False) -> None:
    settings = Settings.from_env()
    database_path = settings.database_path.resolve()
    guild_id = settings.dev_guild_id or DEFAULT_GUILD_ID
    backup = None if evaluate_only else _backup(database_path)
    database = Database(database_path, settings.migrations_path)
    await database.initialize()
    try:
        archive = ArchiveRepository(database)
        lexicon = LexiconService(database)
        morphology = MorphologyRepository(database)
        semantic = SemanticMemoryRepository(database)
        discourse = DiscourseMemoryRepository(database)
        last_id = BASE_MESSAGE_ID + total_turns() - 1
        complete = await database.fetchone(
            "SELECT 1 FROM archive_messages WHERE message_id = ?", (last_id,)
        )
        if not evaluate_only:
            await _install_senses(lexicon)
            if complete is None:
                inserted = await _ingest(archive, morphology, guild_id)
                print(f"grammar_training_messages={inserted}")
            else:
                print("grammar_training_messages=already_present")
            print(f"semantic_rebuilt={await semantic.rebuild_archive(guild_id)}")
            print(f"discourse_rebuilt={await discourse.rebuild_archive(guild_id)}")
        else:
            print(f"grammar_training_messages={'already_present' if complete else 'not_present'}")
        await _report(database, archive, lexicon, guild_id, backup)
    finally:
        await database.close()


async def _install_senses(lexicon: LexiconService) -> None:
    for lemma, definition, terms, synonyms, homonym_group in SENSES:
        await lexicon.import_sense(
            lemma=lemma,
            definition=definition,
            context_terms=terms,
            synonyms=synonyms,
            homonym_group=homonym_group,
            part_of_speech="NOUN" if lemma not in {"быстрый", "говорить"} else None,
            source="grammar_semantics_curriculum",
            confidence=0.95,
        )
    for lemma, part, meaning, labels in FUNCTION_WORDS:
        await lexicon.import_dictionary_entry(
            lemma=lemma,
            part_of_speech=part,
            meanings=(meaning,),
            style_labels=labels,
            source="gramota_opencorpora_curriculum",
            confidence=0.94,
        )


async def _ingest(
    archive: ArchiveRepository, morphology: MorphologyRepository, guild_id: int
) -> int:
    job = await archive.create_job(
        guild_id=guild_id,
        channel_id=CHANNEL_ID,
        requested_by=TEACHER_ID,
        after_at=STARTED_AT - timedelta(days=1),
        before_at=STARTED_AT + timedelta(days=2),
        max_messages=total_turns(),
        delete_raw_after=False,
    )
    await archive.mark_running(job.job_id)
    inserted = 0
    offset = 0
    for scene in SCENES:
        previous_id: int | None = None
        for turn in scene.turns:
            message_id = BASE_MESSAGE_ID + offset
            moment = STARTED_AT + timedelta(minutes=offset)
            is_bot = turn.speaker == "bot"
            added = await archive.ingest_message(
                job.job_id,
                ArchivedMessage(
                    message_id=message_id,
                    guild_id=guild_id,
                    channel_id=CHANNEL_ID,
                    author_id=BOT_ID if is_bot else TEACHER_ID,
                    created_at=moment,
                    content=turn.text,
                    reply_to_message_id=previous_id,
                    author_is_bot=is_bot,
                ),
            )
            inserted += int(added)
            if added:
                await morphology.observe(
                    normalize_message_created(
                        message_id=message_id,
                        guild_id=guild_id,
                        channel_id=CHANNEL_ID,
                        author_id=TEACHER_ID,
                        created_at=moment,
                        content=turn.text,
                        author_is_bot=False,
                        reply_to_message_id=previous_id,
                    )
                )
            previous_id = message_id
            offset += 1
    await archive.advance_job(
        job.job_id,
        last_message_id=BASE_MESSAGE_ID + total_turns() - 1,
        seen=total_turns(),
        processed=inserted,
    )
    await archive.complete_job(job.job_id)
    return inserted


async def _report(
    database: Database,
    archive: ArchiveRepository,
    lexicon: LexiconService,
    guild_id: int,
    backup: Path | None,
) -> None:
    counters = {}
    for table in (
        "language_dictionary_senses",
        "language_synonym_links",
        "language_homonym_groups",
        "language_morphology_observations",
        "language_grammar_patterns",
    ):
        row = await database.fetchone(f"SELECT COUNT(*) AS count FROM {table}")
        counters[table] = int(row["count"]) if row else 0
    print(f"backup={backup}")
    print(f"schema={await database.schema_version()}")
    print(f"grammar_scenes={len(SCENES)}")
    print(f"grammar_turns={total_turns()}")
    print(f"grammar_skills={len(covered_skills())}")
    for table, count in counters.items():
        print(f"{table.removeprefix('language_')}={count}")

    morphology = RussianMorphology()
    print("case_probes:")
    for phrase in ("без словаря", "к другу", "вижу стол", "со словарём", "о речи"):
        analysis = morphology.analyze_sentence(phrase)
        cases = tuple(reading.case for reading in analysis.readings if reading and reading.case)
        print(f"- {phrase}: {cases}")
    print("homonym_probes:")
    for word, context in (
        ("ключ", "я открыл дверь ключом"),
        ("ключ", "из земли бьёт холодная вода"),
        ("коса", "девушка заплела волосы в косу"),
        ("коса", "наточил косу и вышел косить траву"),
        ("коса", "я увидел косу"),
    ):
        sense = await lexicon.resolve_sense(word, context)
        print(f"- {context} -> {sense.definition if sense else 'AMBIGUOUS'}")
    print("learned_reply_probes:")
    for probe in PROBES:
        answer = await archive.exact_learned_reply(guild_id, tokenize(probe))
        if answer is None:
            matches = await archive.matching_replies(guild_id, tokenize(probe), limit=1)
            answer = matches[0] if matches else None
        print(f"- {probe} -> {answer}")


def _backup(database_path: Path) -> Path | None:
    if not database_path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup = database_path.with_name(
        f"{database_path.stem}-before-grammar-semantics-{stamp}.sqlite3"
    )
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
