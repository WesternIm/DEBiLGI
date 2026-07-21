from debilgi.language.self_dialogue_curriculum import SELF_DIALOGUE_PAIRS


def test_self_dialogue_is_natural_and_covers_failure_modes() -> None:
    prompts = {prompt for prompt, _ in SELF_DIALOGUE_PAIRS}
    answers = " ".join(answer.casefold() for _, answer in SELF_DIALOGUE_PAIRS)

    assert len(SELF_DIALOGUE_PAIRS) >= 10
    assert "ты разумный?" in prompts
    assert "а почему ты разумный?" in prompts
    assert "почему ты повторяешь одно и то же?" in prompts
    assert "рабочей памяти" in answers
    assert "повышаю доверие" not in answers
    assert "коэффициент" not in answers
