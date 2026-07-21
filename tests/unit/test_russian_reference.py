from debilgi.language.russian_reference import (
    RemoteLexeme,
    WiktionaryClient,
    best_wiktionary_candidate,
    derivational_base_candidates,
    parse_wiktionary_payload,
    reference_lookup_target,
)


def test_reference_lookup_ignores_the_optional_word_marker() -> None:
    assert reference_lookup_target("Что значит слово «кринж»?") == "кринж"
    assert reference_lookup_target("Что такое слово фырбуль?") == "фырбуль"


def test_parses_russian_wiktionary_particle_and_meanings() -> None:
    payload = {
        "parse": {
            "wikitext": """
= {{-ru-}} =
== {{з|(частица)}} ==
==== Значение ====
# выражает [[сомнение]] или недоверие {{пример|Ну, не может быть!}}
# вводит [[итог]] рассуждения
==== Синонимы ====
# [[итак]]
== {{з|(междометие)}} ==
==== Значение ====
# выражает [[побуждение]]
= {{-en-}} =
== English ==
"""
        }
    }

    result = parse_wiktionary_payload("ну", payload)

    assert result is not None
    assert result.part_of_speech == "PRCL"
    assert result.meanings == (
        "выражает сомнение или недоверие",
        "вводит итог рассуждения",
        "выражает побуждение",
    )
    assert result.examples == ("Ну, не может быть!",)
    assert result.source_url.endswith("/ну")


def test_extracts_nested_usage_examples_without_template_metadata() -> None:
    payload = {
        "parse": {
            "wikitext": """
= {{-ru-}} =
== {{з|(существительное)}} ==
==== Значение ====
# артист цирка {{пример|Весёлый {{выдел|клоун}} вышел на арену.|source=НКРЯ}}
# человек, который смешит {{пример|Этот [[клоун]] рассмешил публику.|год=2001}}
==== Синонимы ====
"""
        }
    }

    result = parse_wiktionary_payload("клоун", payload)

    assert result is not None
    assert result.examples == (
        "Весёлый клоун вышел на арену.",
        "Этот клоун рассмешил публику.",
    )
    assert all("НКРЯ" not in example for example in result.examples)


def test_rejects_page_without_russian_section_or_definitions() -> None:
    assert parse_wiktionary_payload("test", {"parse": {"wikitext": "= {{-en-}} ="}}) is None
    assert parse_wiktionary_payload("слово", {"error": {"code": "missingtitle"}}) is None


def test_preserves_conversational_and_slang_usage_labels() -> None:
    payload = {
        "parse": {
            "wikitext": """
= {{-ru-}} =
=== Морфологические и синтаксические свойства ===
{{сущ-ru|кринж|м 4a}}
=== Семантические свойства ===
==== Значение ====
# {{интернет.}} {{разг.}} сильное чувство неловкости
==== Синонимы ====
"""
        }
    }

    result = parse_wiktionary_payload("кринж", payload)

    assert result is not None
    assert result.part_of_speech == "NOUN"
    assert result.meanings == ("сильное чувство неловкости",)
    assert result.style_labels == ("интернет-сленг", "разговорное")


def test_flexible_search_accepts_only_close_single_word_candidates() -> None:
    assert best_wiktionary_candidate("кринжж", ["Кринж", "Кринж-культура"]) == "кринж"
    assert best_wiktionary_candidate("обманщик", ["Тартюф, или Обманщик"]) is None
    assert best_wiktionary_candidate("варенье", ["Варение"]) is None


def test_conservative_feminine_derivation_points_to_dictionary_base() -> None:
    assert derivational_base_candidates("клоунесса") == ("клоун",)
    assert derivational_base_candidates("принцесса") == ("принц",)
    assert derivational_base_candidates("актриса") == ()


def test_remote_lexeme_can_mark_a_word_formation_inference() -> None:
    result = RemoteLexeme(
        word="клоунесса",
        part_of_speech="NOUN",
        meanings=("женское название лица, соответствующее слову «клоун»",),
        source_url="https://ru.wiktionary.org/wiki/клоун",
        style_labels=("словообразовательный вывод",),
        derived_from="клоун",
    )

    assert result.derived_from == "клоун"
    assert "словообразовательный" in result.style_labels[0]


async def test_lookup_falls_back_to_a_grounded_feminine_derivation() -> None:
    class ReferenceStub(WiktionaryClient):
        async def start(self) -> None:
            self._session = object()  # type: ignore[assignment]

        async def _lookup_page(self, word: str) -> RemoteLexeme | None:
            if word != "клоун":
                return None
            return RemoteLexeme(
                word="клоун",
                part_of_speech="NOUN",
                meanings=("комический артист (обычно в цирке)",),
                source_url="https://ru.wiktionary.org/wiki/клоун",
            )

        async def _search_candidate(
            self, word: str, search_terms: tuple[str, ...]
        ) -> str | None:
            return None

    result = await ReferenceStub(user_agent="test").lookup("клоунесса")

    assert result is not None
    assert result.word == "клоунесса"
    assert result.derived_from == "клоун"
    assert "комический артист" in result.meanings[0]
