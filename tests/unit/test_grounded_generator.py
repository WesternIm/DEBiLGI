from debilgi.cognition.models import ActionType
from debilgi.language.grounded_generator import GroundedConstructionGenerator
from debilgi.language.planning import ReplyContext, StructuredThought, utterance_plan


def _plan(context: ReplyContext):
    thought = StructuredThought(ActionType.SHORT_REPLY, None, "answer", 0.8, ())
    plan = utterance_plan(thought, context=context)
    assert plan is not None
    return plan


def test_uses_a_complete_relevant_local_fact_without_token_synthesis() -> None:
    plan = _plan(
        ReplyContext(
            prompt="Где лежит ключ?",
            knowledge_facts=(
                "Дракон летит над городом.",
                "Ключ лежит в ящике.",
            ),
        )
    )

    candidates = GroundedConstructionGenerator().generate_candidates(plan)

    assert tuple(item.text for item in candidates) == ("Ключ лежит в ящике.",)
    assert candidates[0].source == "grounded_surface"


def test_does_not_splice_memory_when_no_grounded_fact_exists() -> None:
    plan = _plan(
        ReplyContext(
            prompt="Что ты об этом думаешь?",
            memory_messages=(
                "разговора выводы вывод",
                "случайные слова из старого сообщения",
            ),
        )
    )

    assert GroundedConstructionGenerator().generate(plan) == ""


def test_rejects_a_memory_fragment_with_missing_predicate_argument() -> None:
    plan = _plan(
        ReplyContext(
            prompt="Что даст большая база?",
            knowledge_facts=(
                "Большая база даст тебе.",
                "Большая база даст тебе примеры.",
            ),
        )
    )

    candidates = GroundedConstructionGenerator().generate_candidates(plan)

    assert tuple(item.text for item in candidates) == (
        "Большая база даст тебе примеры.",
    )
