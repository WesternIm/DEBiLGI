from debilgi.language.dialogue_acts import DialogueActFrame
from debilgi.language.position_reasoning import PositionReasoningService, realize_position


def _frame(*, predicate: str | None = None) -> DialogueActFrame:
    return DialogueActFrame(
        kind="opinion_question",
        response_obligation="express_position",
        subject=None,
        predicate=predicate,
        objects=("бобун", "лох"),
        polarity=0.0,
        register="neutral",
        complete=True,
        confidence=0.88,
        evidence=("opinion_request",),
    )


async def test_missing_evidence_creates_revisable_non_endorsement() -> None:
    decision = await PositionReasoningService().resolve(1, _frame())

    assert decision is not None
    assert decision.stance == "withhold"
    assert decision.expression_mode == "qualified"
    assert "no_supporting_evidence" in decision.evidence
    assert realize_position(decision)[0].text == "Нет, я пока не считаю его лохом."


async def test_grounded_claim_changes_the_position_instead_of_choosing_a_default() -> None:
    service = PositionReasoningService()
    support = await service.resolve(1, _frame(), facts=("Бобун — лох.",))
    opposition = await service.resolve(1, _frame(), facts=("Бобун не лох.",))

    assert support is not None and opposition is not None
    assert support.stance == "support"
    assert opposition.stance == "oppose"
    assert realize_position(support)[0].text == "Да, я считаю его лохом."
    assert realize_position(opposition)[0].text == "Нет, я не считаю его лохом."


async def test_opinion_wrapper_does_not_replace_a_concrete_capability_reasoner() -> None:
    decision = await PositionReasoningService().resolve(1, _frame(predicate="уметь"))

    assert decision is None
