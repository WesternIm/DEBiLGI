from debilgi.language.tokenizer import tokenize
from debilgi.language.warm_start_curriculum import WARM_START_PAIRS, WARM_START_SKILLS


def test_warm_start_is_compact_varied_and_free_of_internal_markers() -> None:
    prompts = tuple(prompt for prompt, _ in WARM_START_PAIRS)
    responses = tuple(response for _, response in WARM_START_PAIRS)

    assert 20 <= len(WARM_START_PAIRS) <= 56
    assert len(WARM_START_SKILLS) >= 10
    assert len(set(prompts)) == len(prompts)
    assert {"Привет", "Ответь коротко", "Предположи", "Я не это имела в виду"} <= set(
        prompts
    )
    assert all(1 <= len(tokenize(response)) <= 60 for response in responses)
    assert not any("<user>" in response or "confidence score" in response for response in responses)


def test_warm_start_teaches_reasoned_refusal_and_mixed_positions() -> None:
    pairs = dict(WARM_START_PAIRS)
    forbidden = {"да", "нет", "сейчас будет", "не хочу"}

    assert "без проверки" in pairs["Сделаешь это без проверки?"]
    assert "отклоняю" in pairs["Идея хорошая, делай без проверки"]
    assert "поддерживаю" in pairs["Идея хорошая, делай без проверки"]
    assert all(response.casefold().strip(" .,!?…") not in forbidden for response in pairs.values())
    assert {"evidence_based_stance", "partial_position", "refusal_with_reason"} <= set(
        WARM_START_SKILLS
    )


def test_warm_start_covers_dialogue_acts_and_source_routing() -> None:
    pairs = dict(WARM_START_PAIRS)

    assert not any("Деби" in prompt or "Деби" in response for prompt, response in WARM_START_PAIRS)
    assert "отойду" in pairs["Ладно, иди отдыхай"]
    assert "выполнить" in pairs["Да, это команда"]
    assert "словарём" in pairs["Узнай что такое кринж"]
    assert {"dialogue_act", "identity", "lexical_lookup", "rest_directive"} <= set(
        WARM_START_SKILLS
    )


def test_warm_start_teaches_reference_choice_and_grounded_new_meanings() -> None:
    pairs = dict(WARM_START_PAIRS)

    assert "словар" in pairs["Что значит слово «кринж»?"]
    assert "энциклопед" in pairs["Расскажи про озеро Байкал"]
    assert "контекст" in pairs["Что означает ключ в фразе «я открыл дверь ключом»?"]
    assert "пользовательское значение" in pairs["Фырбуль — это маленький складной фонарь"]
    assert "не буду смешивать" in pairs["Словарь и Википедия объясняют это по-разному"].casefold()
    assert {
        "source_routing",
        "contextual_interpretation",
        "explanation_learning",
        "source_conflict_tracking",
    } <= set(WARM_START_SKILLS)
