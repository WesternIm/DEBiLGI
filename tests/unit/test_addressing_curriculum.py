from debilgi.language.addressing_curriculum import ADDRESSING_PAIRS, ADDRESSING_SKILLS


def test_addressing_curriculum_covers_ambiguous_roles() -> None:
    prompts = {prompt for prompt, _ in ADDRESSING_PAIRS}

    assert len(ADDRESSING_PAIRS) >= 20
    assert len(ADDRESSING_SKILLS) >= 12
    assert "Узнай, умник." in prompts
    assert "Найди умника." in prompts
    assert "Умник ответил на вопрос." in prompts
    assert "Что значит слово «умник»?" in prompts
    assert "Умник узнай что такое мороженое." in prompts
    assert "Скажи умник где лежит ключ." in prompts
