import hashlib
import sqlite3
from pathlib import Path

import pytest

from debilgi.language.lexicon import LexiconService, UnknownKind
from debilgi.language.productive_nlg import realize_thought
from debilgi.language.ruwordnet import RuWordNet
from debilgi.language.ruwordnet_reasoning import ruwordnet_answer_thought
from debilgi.language.sentence_understanding import understand_utterance
from debilgi.storage.database import Database


@pytest.fixture
async def reference_lexicon(tmp_path):
    path = tmp_path / "ruwordnet.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE synset(
                id TEXT PRIMARY KEY, title TEXT, definition TEXT, part_of_speech TEXT);
            CREATE TABLE sense(id TEXT PRIMARY KEY, name TEXT, lemma TEXT, synset_id TEXT);
            CREATE TABLE hypernym_relation(hyponym_id TEXT, hypernym_id TEXT);
            CREATE TABLE meronymy_relation(meronym_id TEXT, holonym_id TEXT);
        """)
        rows = (
            ("lock-N", "ЗАМОК ДЛЯ ЗАПИРАНИЯ", "устройство для запирания ключом", "N", "ЗАМОК"),
            ("castle-N", "СРЕДНЕВЕКОВЫЙ ЗАМОК", "укреплённое жилище короля", "N", "ЗАМОК"),
            ("asparagus-N", "СПАРЖА", "", "N", "СПАРЖА"),
            ("vegetable-N", "ОВОЩ", "", "N", "ОВОЩ"),
            ("plant-N", "РАСТЕНИЕ", "", "N", "РАСТЕНИЕ"),
            ("wheel-N", "КОЛЕСО", "", "N", "КОЛЕСО"),
            ("car-N", "МАШИНА", "", "N", "МАШИНА"),
            ("stove-N", "ПЕЧЬ", "устройство для нагревания", "N", "ПЕЧЬ"),
            ("bake-V", "ВЫПЕКАТЬ", "готовить в духовке", "V", "ПЕЧЬ"),
        )
        for sid, title, definition, pos, word in rows:
            db.execute("INSERT INTO synset VALUES (?,?,?,?)", (sid, title, definition, pos))
            db.execute("INSERT INTO sense VALUES (?,?,?,?)", (sid + "-1", word, word, sid))
        db.execute("INSERT INTO sense VALUES ('car-2','АВТОМОБИЛЬ','АВТОМОБИЛЬ','car-N')")
        db.executemany("INSERT INTO hypernym_relation VALUES (?,?)", (
            ("asparagus-N", "vegetable-N"), ("asparagus-N", "plant-N"),
        ))
        db.execute("INSERT INTO meronymy_relation VALUES ('wheel-N','car-N')")
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    runtime = Database(tmp_path / "runtime.db", Path(__file__).resolve().parents[2] / "migrations")
    await runtime.initialize()
    yield LexiconService(runtime, ruwordnet=RuWordNet(path))
    await runtime.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash


async def test_ruwordnet_supplies_sense_scoped_relations_and_productive_definition(
    reference_lexicon,
):
    word = await reference_lexicon.analyze_word("спаржа", context="Что такое спаржа?")
    assert word.source == "ruwordnet"
    assert word.sense_id == "asparagus-N"
    assert not word.requires_lexical_study
    assert ("is_a", "овощ") in word.semantic_relations
    understanding = understand_utterance("Что такое спаржа?", (word,))
    edge = next(r for r in understanding.relations if r.target == "овощ")
    assert (edge.source_sense_id, edge.target_sense_id) == ("asparagus-N", "vegetable-N")
    thought = ruwordnet_answer_thought("Что такое спаржа?", (word,))
    assert thought is not None
    assert realize_thought(thought, limit=1)[0].text == "Спаржа является овощем."


@pytest.mark.parametrize("context,expected", [
    ("Замок открывают ключом", "lock-N"),
    ("Средневековый замок короля", "castle-N"),
    ("замок", None), ("ключ и король", None),
    ("Это не средневековый замок", None),
])
async def test_context_selects_a_sense_without_merging_alternatives(
    reference_lexicon, context, expected,
):
    word = await reference_lexicon.analyze_word("замок", context=context)
    assert word.sense_id == expected
    if expected is None:
        assert word.unknown_kind == UnknownKind.AMBIGUOUS_CONTEXT
        assert word.dictionary_meanings == ()
        assert word.semantic_relations == ()
        assert len(word.sense_options) == 2
        assert not word.requires_lexical_study


async def test_ambiguity_answer_names_available_senses_without_asking_user(reference_lexicon):
    word = await reference_lexicon.analyze_word("замок", context="Что такое замок?")
    thought = ruwordnet_answer_thought("Что такое замок?", (word,))
    assert thought is not None
    text = realize_thought(thought, limit=1)[0].text
    assert "замок для запирания" in text and "средневековый замок" in text
    assert " или " in text and text.endswith(".")
    assert "не знаю" not in text


@pytest.mark.parametrize("word,context,expected,case", [
    ("печь", "Буду печь пирог", "bake-V", None),
    ("печь", "В доме стоит печь", "stove-N", "nominative"),
    ("спаржей", "Салат со спаржей", "asparagus-N", "instrumental"),
])
async def test_context_morphology_constrains_thesaurus_senses(
    reference_lexicon, word, context, expected, case,
):
    result = await reference_lexicon.analyze_word(word, context=context)
    assert result.sense_id == expected
    if case:
        assert result.grammatical_features["case"] == case


async def test_relation_directions_and_synonyms_survive_into_graph(reference_lexicon):
    wheel = await reference_lexicon.analyze_word("колесо")
    car = await reference_lexicon.analyze_word("машина")
    assert ("part_of", "машина") in wheel.semantic_relations
    assert ("has_part", "колесо") in car.semantic_relations
    assert ("part_of", "колесо") not in car.semantic_relations
    assert ("synonym", "автомобиль") in car.semantic_relations
    assert any(link.relation == "synonym" and link.target_id == car.sense_id
               for link in car.typed_relations)


async def test_local_dictionary_definition_keeps_priority(reference_lexicon):
    await reference_lexicon.import_dictionary_entry(
        lemma="замок", part_of_speech="NOUN", meanings=("условный термин нашей игры",),
    )
    word = await reference_lexicon.analyze_word("замок", context="замок")
    assert word.source == "dictionary_cache"
    assert word.dictionary_meanings == ("условный термин нашей игры",)
    assert word.sense_id is None


async def test_explicit_lesson_overrides_thesaurus_without_importing_wrong_sense_links(
    reference_lexicon,
):
    await reference_lexicon.teach_personal_interpretation(
        guild_id=1, word="машина", interpretation="вычислительное устройство",
    )
    word = await reference_lexicon.analyze_word("машиной", guild_id=1, context="Работаю с машиной")
    assert word.source == "personal_lexicon"
    assert word.selected_meaning == "вычислительное устройство"
    assert word.grammatical_features["case"] == "instrumental"
    assert word.sense_id is None
    assert ("has_part", "колесо") not in word.semantic_relations
    assert ("is_a", "вычислительный устройство") in word.semantic_relations
