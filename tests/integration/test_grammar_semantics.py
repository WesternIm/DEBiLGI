from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from debilgi.language.lexicon import LexiconService
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "grammar-semantics.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


async def test_context_resolves_general_homonyms(database: Database) -> None:
    service = LexiconService(database)
    await service.import_sense(
        lemma="ключ",
        definition="предмет для отпирания замка",
        context_terms=("замок", "дверь", "отпереть", "металл"),
        synonyms=("отмычка",),
        homonym_group="ключ",
        part_of_speech="NOUN",
    )
    await service.import_sense(
        lemma="ключ",
        definition="родник, выход подземной воды",
        context_terms=("вода", "родник", "лес", "пить"),
        synonyms=("родник", "источник"),
        homonym_group="ключ",
        part_of_speech="NOUN",
    )

    door = await service.resolve_sense("ключом", "Я открыл ключом дверной замок")
    spring = await service.resolve_sense("ключ", "В лесу из земли бьёт чистая вода")
    ambiguous = await service.resolve_sense("ключ", "Я увидел ключ")

    assert door and "отпирания" in door.definition
    assert spring and "воды" in spring.definition
    assert ambiguous is None


async def test_synonyms_expand_semantic_retrieval_terms(database: Database) -> None:
    service = LexiconService(database)
    await service.import_sense(
        lemma="быстрый",
        definition="движущийся с большой скоростью",
        context_terms=("скорость", "движение"),
        synonyms=("скорый", "стремительный"),
        part_of_speech="ADJF",
    )

    synonyms = await service.lookup_synonyms("быстрое")
    terms = await service.semantic_terms("быстрое движение")

    assert {"скорый", "стремительный"} <= set(synonyms)
    assert {"быстрый", "скорый", "стремительный"} <= set(terms)


async def test_reference_personal_and_local_meanings_remain_separate(database: Database) -> None:
    service = LexiconService(database)
    await service.import_sense(
        lemma="огонь",
        definition="горящие газы, выделяющие свет и тепло",
        context_terms=("пламя", "тепло", "гореть"),
        synonyms=("пламя",),
        part_of_speech="NOUN",
    )
    await service.teach_personal_interpretation(
        guild_id=7,
        word="огонь",
        interpretation="что-то очень удачное и впечатляющее",
    )
    await service.register_server_slang(
        guild_id=7,
        word="огонь",
        meaning="высшая локальная оценка результата",
    )

    comparison = await service.compare_with_reference(
        "огонь", guild_id=7, context="Этот результат просто огонь"
    )

    assert comparison.reference_meaning == "горящие газы, выделяющие свет и тепло"
    assert comparison.personal_meaning == "что-то очень удачное и впечатляющее"
    assert comparison.local_meaning == "высшая локальная оценка результата"
    assert comparison.relation == "different"
