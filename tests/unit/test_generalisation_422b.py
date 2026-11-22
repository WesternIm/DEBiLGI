"""Stage 4.22b: deixis, anaphora, aspect pairs and response control as grammar mechanisms.

Every case uses words and forms that did not occur in the live dialogue, to show that the
mechanism (morphological features), not a word, does the work.
"""

import pytest

from debilgi.language.dialogue_state import (
    has_antecedent,
    is_anaphoric_reference,
    unbound_anaphors,
)
from debilgi.language.lexical_matching import aspect_pair
from debilgi.language.morphology import (
    DEFAULT_MORPHOLOGY,
    DEIXIS_CELLS,
    deictic_counterpart,
    person_cell,
)
from debilgi.language.productive_nlg import realize_thought
from debilgi.language.proposition_memory import meaning_key
from debilgi.language.request_focus import is_response_control
from debilgi.language.semantic_graph import parse_proposition
from debilgi.language.structured_utterance import analyze_structured_utterance


def retold(text, **flags):
    thought = parse_proposition(text, allow_question=True, **flags)
    assert thought is not None, text
    return realize_thought(thought, limit=1)[0].text


# --- deixis: person and number are features, forms and cases are inflected -------------------
@pytest.mark.parametrize(
    "heard, said",
    [
        ("ты видел меня?", "Я видел тебя."),  # accusative
        ("ты дашь мне книгу?", "Я дам тебе книгу."),  # dative
        ("ты пойдёшь со мной?", "Я пойду с тобой."),  # instrumental
        ("ты слышал о нас?", "Я слышал о вас."),  # plural personal
        ("я видел твою кошку?", "Ты видел мою кошку."),  # 2nd → 1st possessive
        ("ты любишь мою сестру?", "Я люблю твою сестру."),  # feminine accusative
        ("ты знаешь моих родителей?", "Я знаю твоих родителей."),  # plural animate
        ("ты читал мои письма?", "Я читал твои письма."),  # plural inanimate
        ("ты поедешь на моей машине?", "Я поеду на твоей машине."),  # prepositional
        ("ты дружишь с моим братом?", "Я дружу с твоим братом."),  # instrumental masc
        ("ты ответишь на мой вопрос?", "Я отвечу на твой вопрос."),
    ],
)
def test_person_swaps_across_pronoun_and_possessive_forms(heard, said):
    assert retold(heard, possessive_deixis=True) == said


@pytest.mark.parametrize(
    "heard",
    ["ты знаешь своего создателя?", "ты любишь наш сервер?", "ты видел наших друзей?"],
)
def test_reflexive_and_inclusive_possessives_stay_without_a_context_signal(heard):
    kept = retold(heard, possessive_deixis=True)
    assert kept == retold(heard)  # same surface as the default reading
    assert any(word in kept for word in ("своего", "наш", "наших"))


def test_default_reading_and_fact_keys_do_not_change():
    thought = parse_proposition("я кормлю мою кошку")
    flagged = parse_proposition("я кормлю мою кошку", possessive_deixis=True)
    assert "мой" in meaning_key(thought) and "твой" not in meaning_key(thought)
    assert "твой" in meaning_key(flagged)  # the flag is the only difference
    # the personal pronoun swap of the default reader is untouched
    assert retold("я вчера отдыхал?") == "Ты отдыхал вчера."


def test_cells_are_checked_against_the_analyzer():
    for (person, number), (pronoun, possessive) in DEIXIS_CELLS.items():
        assert person_cell(pronoun) == (person, number, "personal")
        assert person_cell(possessive) == (person, number, "possessive")
        reading = DEFAULT_MORPHOLOGY._analyzer.parse(pronoun)[0].tag
        assert ("1per" if person == "first" else "2per") in reading
    for lemma in ("он", "они", "кто", "свой", "этот", "кошка", "бежать"):
        assert deictic_counterpart(lemma) is None


@pytest.mark.parametrize("lemma", ["я", "ты", "мы", "вы", "мой", "твой"])
def test_person_swap_is_an_involution(lemma):
    other = deictic_counterpart(lemma)
    assert other and deictic_counterpart(other) == lemma


# --- anaphora ---------------------------------------------------------------------------
def anaphors(answer, prompt):
    return [t.surface for t in unbound_anaphors(
        analyze_structured_utterance(answer), analyze_structured_utterance(prompt))]


@pytest.mark.parametrize(
    "answer, prompt, dangling",
    [
        ("кто их не любит", "любишь пироги?", []),
        ("кто их не любит", "ты бы хотел полететь?", ["их"]),
        ("кто её не любит", "любишь кошку?", []),
        ("кто её не любит", "любишь кота?", ["её"]),  # gender disagrees
        ("а кто его не любит", "любишь кота?", []),
        ("а кто его не любит", "любишь кошек?", ["его"]),
        ("им это нужно", "видел котов?", []),
        ("им это нужно", "как дела?", ["им"]),  # dative plural, no antecedent
        ("ими не пользуются", "любишь ножи и вилки?", []),  # instrumental plural
        ("ими не пользуются", "любишь нож?", ["ими"]),
        ("она красивая", "смотрел фильм?", ["она"]),
        ("кто тот", "видел кота?", []),  # demonstrative without noun
        ("кто тот", "как дела?", ["тот"]),
        ("кто себя любит", "как дела?", []),  # reflexive binds inside its clause
        ("это круто", "как дела?", []),  # neuter demonstrative refers to the situation
        ("кошка, а её не видно", "как дела?", []),  # antecedent earlier in the same text
    ],
)
def test_dangling_anaphora_of_any_pronoun(answer, prompt, dangling):
    assert anaphors(answer, prompt) == dangling


def test_anaphor_class_comes_from_morphology_not_words():
    flags = {
        t.surface: is_anaphoric_reference(t)
        for t in analyze_structured_utterance("им ими нему неё ихний кто что мне себя свой").tokens
    }
    assert flags["им"] and flags["ими"] and flags["нему"] and flags["неё"]
    assert not flags["мне"] and not flags["себя"] and not flags["свой"] and not flags["кто"]
    two_nouns = analyze_structured_utterance("кошка и собака")
    anaphor = next(t for t in analyze_structured_utterance("они дома").tokens
                   if t.lemma == "они")
    assert has_antecedent(anaphor, tuple(two_nouns.tokens))  # coordination is plural


# --- aspect pairs by morphology -----------------------------------------------------------
@pytest.mark.parametrize(
    "left, right",
    [
        ("просыпаться", "проснуться"), ("отвечать", "ответить"), ("делать", "сделать"),
        ("понимать", "понять"), ("решать", "решить"), ("закрывать", "закрыть"),
        ("рисовать", "нарисовать"), ("будить", "разбудить"), ("засыпать", "заснуть"),
        ("одеваться", "одеться"), ("покупать", "купить"), ("вспоминать", "вспомнить"),
        ("купаться", "искупаться"), ("пробуждаться", "пробудиться"),
    ],
)
def test_aspect_pairs_are_found_in_both_directions(left, right):
    assert aspect_pair(left, right) and aspect_pair(right, left)


@pytest.mark.parametrize(
    "left, right",
    [
        ("хотеть", "мочь"), ("любить", "ловить"), ("писать", "плакать"), ("мыть", "мыться"),
        ("брать", "взять"), ("делать", "делать"), ("просыпаться", "просить"),
        ("отвечать", "отдохнуть"), ("класть", "красть"),
    ],
)
def test_unrelated_or_identical_verbs_are_not_aspect_pairs(left, right):
    assert not aspect_pair(left, right)


# --- response control -----------------------------------------------------------------
@pytest.mark.parametrize(
    "text, control",
    [
        ("ты ответишь на мой вопрос?", True),
        ("вы ответите на наш вопрос?", True),
        ("ты ответишь на этот вопрос?", True),
        ("ты скажешь свой ответ?", True),
        ("ты ответишь на мой вопрос про кошку?", False),
        ("ты ответишь на мой вопрос про собаку?", False),
        ("ты расскажешь про кошку?", False),
    ],
)
def test_pronouns_and_discourse_nouns_do_not_make_new_content(text, control):
    assert is_response_control(text) is control


# --- «бы» in one clause ----------------------------------------------------------------
@pytest.mark.parametrize(
    "text, said",
    [
        ("ты бы хотел полететь?", "Я хотел бы полететь."),
        ("ты бы ответил на мой вопрос?", "Я ответил бы на твой вопрос."),
        ("ты бы прочитал мои письма?", "Я прочитал бы твои письма."),
    ],
)
def test_irrealis_reading_is_opt_in_and_marks_the_mood(text, said):
    assert parse_proposition(text, allow_question=True) is None  # the default reader refuses
    thought = parse_proposition(text, allow_question=True, irrealis=True,
                                possessive_deixis=True)
    assert dict(thought.predicate.features)["mood"] == "conditional"
    assert realize_thought(thought, limit=1)[0].text == said


# --- the task that asks the agent to say something is met from the agent's side -----------
def test_requested_content_is_realized_in_the_agents_person():
    from debilgi.cognition.dialogue_tasks import _from_agent_side, _normalized_words

    for asked, said in [
        ("на мой вопрос про кошку", ("на", "твой", "вопрос", "про", "кошку")),
        ("со мной и моей сестрой", ("со", "тобой", "и", "твоей", "сестрой")),
        ("меня тебя", ("тебя", "меня")),
        ("нашу книгу", ("нашу", "книгу")),  # inclusive «наш» stays without a context signal
    ]:
        assert _from_agent_side(_normalized_words(asked)) == said
