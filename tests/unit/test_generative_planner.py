from dataclasses import replace

import pytest

from debilgi.language.communication import MessagePlan, ResponseComposer, plan_issues
from debilgi.language.generative_planner import (
    plan_from_evidence,
    resolve_evidence_context,
    statement_thought,
)
from debilgi.language.grounded_repair import unresolved_answer_thought
from debilgi.language.planning import ReplyContext
from debilgi.language.structured_utterance import analyze_structured_utterance


@pytest.mark.parametrize(
    "topic,fact,predicate",
    [
        ("кварц", "Кварц является минералом.", "являться"),
        ("Марс", "Марс вращается.", "вращаться"),
        ("Машу", "Маша нашла ключ.", "найти"),
    ],
)
def test_content_is_derived_from_supplied_fact(topic, fact, predicate):
    context = ReplyContext(prompt=f"Что знаешь про {topic}?", knowledge_facts=(fact,))
    result = plan_from_evidence(context)
    assert result.objective.kind == "content"
    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.plan.claim.predicate.lemma == predicate
    assert proposal.plan.claim.response_polarity is None
    assert proposal.premises[0].text == fact
    composer = ResponseComposer(analyze_structured_utterance(context.prompt))
    assert composer.compose(proposal.plan)
    # Removing the premise must remove the claim, not switch to a negative answer.
    missing = plan_from_evidence(replace(context, knowledge_facts=()))
    assert missing.proposals == ()
    assert missing.unresolved


def test_slot_keeps_actor_and_patient():
    result = plan_from_evidence(
        ReplyContext(
            prompt="Кто нашёл ключ?",
            knowledge_facts=("Маша нашла ключ.", "Маша нашла книгу."),
        )
    )
    assert len(result.proposals) == 1
    roles = {d.role: d.argument.head.lemma for d in result.proposals[0].plan.claim.dependencies}
    assert roles == {"subject": "маша", "direct_object": "ключ"}


@pytest.mark.parametrize(
    "fact",
    [
        "Маша, возможно, нашла ключ.",
        "Если Маша нашла ключ, дверь открыта.",
        "Маша якобы нашла ключ.",
    ],
)
def test_unsupported_qualifications_are_not_silently_asserted(fact):
    assert statement_thought(fact) is None


def test_conflicting_premises_do_not_choose_arbitrary_truth():
    result = plan_from_evidence(
        ReplyContext(
            prompt="Что знаешь про Машу?",
            knowledge_facts=("Маша нашла ключ.", "Маша не нашла ключ."),
        )
    )
    assert result.proposals == ()


def test_dialogue_report_uses_actual_user_turn_not_unrelated_knowledge():
    result = plan_from_evidence(
        ReplyContext(
            prompt="И что услышал",
            consultation_messages=("user: Я нашёл ключ.", "agent: Слушаю."),
            knowledge_facts=("Марс вращается.",),
        )
    )
    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.operation == "attribute_dialogue_turn"
    assert proposal.premises[0].text == "Я нашёл ключ."
    assert ResponseComposer(analyze_structured_utterance("И что услышал")).compose(proposal.plan)


def test_planner_failure_is_not_conversational_content():
    composer = ResponseComposer(analyze_structured_utterance("Как это работает?"))
    plan = MessagePlan(composer.goal, unresolved_answer_thought())
    assert "internal_outcome_is_not_response_content" in plan_issues(plan)
    assert composer.compose(plan) == ()


def test_topic_quantifier_ability_statement_is_reconstructed():
    prompt = "а машины то некоторые умеют летать"
    result = plan_from_evidence(ReplyContext(prompt=prompt))
    claim = statement_thought(prompt, speaker="user")
    assert claim is not None
    subject = next(item.argument for item in claim.dependencies if item.role == "subject")
    assert subject.head.lemma == "машина"
    assert dict(subject.head.features)["number"] == "plural"
    assert result.proposals[0].operation == "resolve_open_group"


def test_same_dependency_recovery_handles_a_new_action_verb():
    prompt = "а машины то некоторые умеют плавать"
    claim = statement_thought(prompt, speaker="user")
    assert claim is not None
    assert claim.predicate.lemma == "уметь"
    action = next(
        item.argument.head.lemma for item in claim.dependencies if item.role == "action_complement"
    )
    assert action == "плавать"


def test_internal_self_state_facts_do_not_become_missing_argument_evidence():
    result = plan_from_evidence(
        ReplyContext(
            prompt="не знаю",
            knowledge_facts=("я знаю нормативное слово", "я не знаю этого слова"),
        )
    )

    assert result.proposals == ()
    assert all(item.kind != "knowledge" for item in result.evidence)


@pytest.mark.parametrize(
    "topic,statement,fact",
    [
        ("зарядка", "Зарядка работает.", "Зарядка заряжает телефон."),
        ("вертолёт", "а вертолет то летит", "Вертолёт перевозит людей."),
    ],
)
def test_statement_continues_from_a_connected_memory(topic, statement, fact):
    result = plan_from_evidence(
        ReplyContext(
            prompt=statement,
            knowledge_facts=(fact, "Марс вращается."),
        )
    )
    assert result.proposals
    assert all(p.operation == "continue_from_memory" for p in result.proposals)
    assert result.proposals[0].premises[-1].text == fact
    missing = plan_from_evidence(ReplyContext(prompt=statement))
    assert missing.proposals
    assert all(p.operation == "query_unobserved_relation" for p in missing.proposals)
    assert all(p.plan.claim.clause_kind == "question" for p in missing.proposals)


def test_resolved_subject_selects_the_actual_remembered_proposition():
    result = plan_from_evidence(
        ReplyContext(
            prompt="Где она лежит?",
            dialogue_messages=("Книга лежит на столе.",),
            knowledge_facts=("Книга лежит на столе.", "Маша лежит на диване."),
        )
    )
    assert len(result.proposals) == 1
    assert result.proposals[0].premises[0].text == "Книга лежит на столе."


def test_omitted_content_request_keeps_the_previous_user_topic():
    result = plan_from_evidence(
        ReplyContext(
            prompt="расскажи",
            consultation_messages=(
                "user: Что знаешь про кварц?",
                "agent: Я знаю кварц.",
            ),
            knowledge_facts=("Кварц является минералом.",),
        )
    )
    assert len(result.proposals) == 1
    assert result.objective.topics == ("кварц",)


def test_why_requires_cause_instead_of_repeating_the_assertion():
    result = plan_from_evidence(
        ReplyContext(
            prompt="Почему бот молчит?",
            knowledge_facts=("Бот молчит.",),
        )
    )
    assert result.proposals == ()


def test_frequency_modifier_survives_proposition_reconstruction():
    thought = statement_thought("Ты периодически не отвечаешь.", speaker="user")
    assert thought is not None and thought.negated
    assert any(
        d.role == "manner" and d.argument.head.lemma == "периодически" for d in thought.dependencies
    )


def test_contact_opens_new_episode_instead_of_resuming_a_content_request():
    result = plan_from_evidence(
        ReplyContext(
            prompt="але",
            consultation_messages=("user: Что знаешь про кварц?",),
            knowledge_facts=("Кварц является минералом.",),
        )
    )
    assert not result.proposals
    assert result.objective.predicate is None


def test_connected_memory_conflict_is_not_presented_as_a_continuation():
    result = plan_from_evidence(
        ReplyContext(
            prompt="Маша нашла ключ.",
            knowledge_facts=(
                "Маша читает книгу.",
                "Маша не читает книгу.",
            ),
        )
    )
    assert result.proposals
    assert all(p.plan.claim.clause_kind == "question" for p in result.proposals)
    assert all(p.operation != "continue_from_memory" for p in result.proposals)


def test_unattributed_self_claim_is_not_evidence_of_bot_attention():
    result = plan_from_evidence(
        ReplyContext(
            prompt="Ты слушаешь?",
            knowledge_facts=("Я слушаю тебя.",),
        )
    )
    assert result.proposals == ()


@pytest.mark.parametrize(
    "prompt,expected_object",
    (
        ("У тебя есть бабушки?", "бабушки"),
        ("У тебя сколько бабушек?", "бабушек"),
    ),
)
def test_relation_question_is_answered_by_general_subject_ontology(
    prompt, expected_object
):
    context = ReplyContext(
        prompt=prompt,
        reference_definitions=(
            ("бабушка", "мать отца или матери, родственница и предок человека"),
        ),
    )

    result = plan_from_evidence(context)

    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.operation == "resolve_relation"
    assert proposal.premises[0].kind == "relation_reasoning"
    surfaces = ResponseComposer(analyze_structured_utterance(prompt)).compose(proposal.plan)
    assert surfaces
    surface = surfaces[0].text.casefold()
    assert "у меня нет" in surface
    assert expected_object in surface


@pytest.mark.parametrize(
    "prompt,dialogue_messages",
    (
        ("Ты машиной управлять можешь?", ()),
        ("А ты можешь?", ("Ты машиной управлять можешь?",)),
    ),
)
def test_self_capability_is_planned_before_distributional_wording(
    prompt, dialogue_messages
):
    result = plan_from_evidence(
        ReplyContext(prompt=prompt, dialogue_messages=dialogue_messages)
    )

    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.operation == "resolve_capability"
    assert proposal.premises[0].kind == "capability_reasoning"
    surfaces = ResponseComposer(analyze_structured_utterance(prompt)).compose(
        proposal.plan,
        proposal.patterns,
    )
    assert surfaces
    surface = surfaces[0].text.casefold()
    assert "управлять" in surface
    assert proposal.plan.claim.negated


def test_contact_does_not_resume_an_unanswered_capability() -> None:
    question = "Ты машиной управлять можешь?"
    result = plan_from_evidence(
        ReplyContext(
            prompt="ауу",
            consultation_messages=(f"user: {question}",),
            dialogue_messages=(question,),
        )
    )

    assert not result.proposals
    assert result.objective.kind == "respond"


@pytest.mark.parametrize("prompt", ["ну так что? Можешь?", "але", "аууу"])
def test_elliptical_modal_and_contact_skip_incomplete_turns_to_recover_action(
    prompt,
) -> None:
    question = "Ты машиной управлять можешь?"
    consultation = (
        f"user: {question}",
        "user: Не знаю, а ты можешь?",
        "agent: Не знаю, чего хочешь, как я обычно решаю, что делать дальше?",
        "user: ну так что? Можешь?",
    )
    if prompt == "ну так что? Можешь?":
        consultation = consultation[:-1]
    context = ReplyContext(
        prompt=prompt,
        consultation_messages=consultation,
        dialogue_messages=tuple(turn.partition(": ")[2] for turn in consultation),
    )

    resolved = resolve_evidence_context(context)
    if prompt in {"але", "аууу"}:
        assert resolved.prompt == prompt
        assert not plan_from_evidence(context).proposals
        return
    assert resolved.prompt == question
    result = plan_from_evidence(context)
    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.operation == "resolve_capability"
    assert proposal.plan.claim.negated
    assert any(
        dependency.role == "action_complement"
        and dependency.argument.head.lemma == "управлять"
        for dependency in proposal.plan.claim.dependencies
    )


def test_report_keeps_possession_in_the_speakers_perspective():
    thought = statement_thought("Моя машина работает.", speaker="user")
    assert thought is not None
    subject = next(d.argument for d in thought.dependencies if d.role == "subject")
    assert subject.modifiers[0].lemma == "твой"
    composer = ResponseComposer(analyze_structured_utterance("Что знаешь про машину?"))
    assert "твоя" in composer.express(thought)[0].text.casefold()


@pytest.mark.parametrize("prompt", ["ебать, а я ногти грызу", "ого, а я яблоко ем"])
def test_detached_emotion_does_not_erase_a_complete_report(prompt):
    thought = statement_thought(prompt, speaker="user")
    assert thought is not None
    result = plan_from_evidence(ReplyContext(prompt=prompt))
    assert result.proposals
    assert result.proposals[0].operation == "query_unobserved_relation"
    assert result.proposals[0].premises[0].text == prompt


def test_dependency_guess_cannot_change_an_omitted_addressee_into_a_noun_subject(monkeypatch):
    from debilgi.language import syntax_backend
    from debilgi.language.syntax_backend import SyntaxEdge

    monkeypatch.setattr(
        syntax_backend,
        "syntax_edges",
        lambda _: (
            SyntaxEdge(1, 0, "nsubj"),
            SyntaxEdge(-1, 1, "root"),
        ),
    )
    assert statement_thought("ногти потеряешь") is None


def test_bare_elliptical_question_keeps_the_last_human_request_without_bot_invention():
    from debilgi.language.generative_planner import resolve_evidence_context

    context = ReplyContext(
        prompt="Чего?",
        consultation_messages=(
            "user: Почему я тупой?",
            "agent: Я считаю себя тупым.",
        ),
    )
    resolved = resolve_evidence_context(context)
    assert resolved.prompt == "Почему я тупой?"
    assert (
        resolve_evidence_context(replace(context, prompt="Кто нашёл ключ?")).prompt
        == "Кто нашёл ключ?"
    )


def test_short_purpose_answer_is_integrated_instead_of_replacing_it_with_old_request():
    from debilgi.language.dialogue_state import analyze_dialogue_state

    context = ReplyContext(
        prompt="Для эксперимента",
        dialogue_state=analyze_dialogue_state(
            "Для эксперимента", ("Почему ты сейчас меня выключишь?",)
        ),
        consultation_messages=("user: Мне тебя выключить?",),
    )
    result = plan_from_evidence(context)
    assert result.proposals[0].operation == "integrate_user_answer"
    thought = result.proposals[0].plan.claim
    assert thought.clause_kind == "statement"
    assert any(
        dep.role == "cause" and dep.argument.head.lemma == "эксперимент"
        for dep in thought.dependencies
    )
    assert result.proposals[0].premises[0].kind == "bound_user_answer"
    surfaces = ResponseComposer(analyze_structured_utterance(context.prompt)).compose(
        result.proposals[0].plan
    )
    assert surfaces and "эксперимента" in surfaces[0].text


def test_summary_marker_preserves_human_action_instead_of_copying_an_answer_control():
    result = plan_from_evidence(ReplyContext(prompt="Короче выключу тебя"))
    assert result.proposals and result.proposals[0].operation == "query_unobserved_relation"
    thought = result.proposals[0].plan.claim
    assert thought.predicate.lemma == "выключить"
    roles = {dep.role: dep.argument.head.lemma for dep in thought.dependencies}
    assert roles["subject"] == "ты" and roles["direct_object"] == "я"
