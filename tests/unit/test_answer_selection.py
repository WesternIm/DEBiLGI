from debilgi.language.answer_selection import select_answer_shape
from debilgi.language.generative_planner import plan_from_evidence
from debilgi.language.planning import ReplyContext


def test_focused_question_can_use_a_phrase_without_copying_a_reply():
    result = plan_from_evidence(
        ReplyContext(prompt="Кто нашёл ключ?", knowledge_facts=("Маша нашла ключ.",))
    )
    shape = select_answer_shape(result.objective, result.proposals[0].plan.claim)
    assert shape.mode == "fragment"
    assert shape.focus_role == "subject"


def test_content_question_keeps_the_proposition():
    result = plan_from_evidence(
        ReplyContext(
            prompt="Что знаешь про кварц?",
            knowledge_facts=("Кварц является минералом.",),
        )
    )
    shape = select_answer_shape(result.objective, result.proposals[0].plan.claim)
    assert shape.mode == "sentence"
