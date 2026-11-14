from pathlib import Path

import pytest

from debilgi.language.lexicon import LexiconService
from debilgi.language.morphology import DEFAULT_MORPHOLOGY, POS_NAMES
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.storage.database import Database


@pytest.fixture
async def lexicon(tmp_path):
    db = Database(tmp_path / "context.sqlite3", Path(__file__).resolve().parents[2] / "migrations")
    await db.initialize()
    service = LexiconService(db)
    for lemma, part, meaning, forms in (
        ("сталь", "NOUN", "сплав железа с углеродом", {"стали": {"case": "dative"}}),
        ("стать", "VERB", "перейти к действию или в другое состояние", {}),
        ("печь", "NOUN", "устройство для нагревания", {}),
        ("ручка", "NOUN", "инструмент для письма", {}),
    ):
        await service.import_dictionary_entry(
            lemma=lemma,
            part_of_speech=part,
            meanings=(meaning,),
            forms=forms,
        )
    yield service
    await db.close()


@pytest.mark.parametrize(
    "word,prompt,lemma,pos,feature,value",
    [
        ("стали", "Они стали работать", "стать", "verb", "tense", "past"),
        ("стали", "Нож сделан из стали", "сталь", "noun", "case", "genitive"),
        ("печь", "Буду печь пирог", "печь", "verb", "aspect", "imperfective"),
        ("печь", "В доме стоит печь", "печь", "noun", "case", "nominative"),
        ("ручкой", "Я пишу ручкой", "ручка", "noun", "case", "instrumental"),
    ],
)
async def test_dictionary_follows_sentence_morphology(
    lexicon, word, prompt, lemma, pos, feature, value
):
    analysis = await lexicon.analyze_word(word, context=prompt)
    assert analysis.lemma == lemma
    assert POS_NAMES.get(analysis.part_of_speech, analysis.part_of_speech) == pos
    assert analysis.grammatical_features[feature] == value
    if word == "печь" and pos == "verb":
        assert "устройство для нагревания" not in analysis.dictionary_meanings
    else:
        assert analysis.dictionary_meanings
    structure = analyze_structured_utterance(prompt)
    token = next(token for token in structure.tokens if token.surface.casefold() == word)
    assert token.lemma == analysis.lemma


def test_sentence_retains_alternative_parts_of_speech():
    sentence = DEFAULT_MORPHOLOGY.analyze_sentence("Они стали работать")
    index = sentence.tokens.index("стали")
    assert {item.part_of_speech for item in sentence.alternatives[index]} >= {"noun", "verb"}
    assert sentence.readings[index].lemma == "стать"
