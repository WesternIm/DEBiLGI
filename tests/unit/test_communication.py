from dataclasses import replace
from unittest.mock import Mock

import pytest

import debilgi.language.communication as communication
from debilgi.language.communication import (
    MessagePlan,
    PendingText,
    ResponseComposer,
    plan_issues,
    reader_goal,
    surface_issues,
)
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.language.structured_utterance import analyze_structured_utterance


def thought(subject="маша", obj="петя", *, negated=False):
    return ClauseThought(
        ConceptLexeme("видеть", "видеть", "verb"),
        tuple(SemanticDependency(role, ConceptArgument(ConceptLexeme(lemma, lemma, "noun")))
              for role, lemma in (("subject", subject), ("direct_object", obj))),
        negated=negated, confidence=.85, evidence=("event:observed",),
    )


def test_content_can_be_compared_and_selected_without_calling_nlg(monkeypatch):
    composer = ResponseComposer(analyze_structured_utterance("Маша видит Петю?"))
    composer.deferred = True
    nlg = Mock(wraps=communication.realize_thought)
    monkeypatch.setattr(communication, "realize_thought", nlg)
    requested = composer.express(thought())[0].text
    unrelated = composer.express(thought("коля", "дима"))[0].text
    assert isinstance(requested, PendingText)
    # Fluency preferences and an unrelated producer's confidence cannot choose
    # the topic. No candidate has received words at this stage.
    assert composer.priority(requested, {}, .5) > composer.priority(unrelated, {}, .99)
    nlg.assert_not_called()
    result = composer.realize(requested)
    assert result
    assert nlg.call_count == 1
    metadata = composer.metadata_for(result[0].text)
    assert metadata["message_plan_key"] == requested.plan.key
    assert metadata["reader_checked"]


@pytest.mark.parametrize("text,issue", [
    ("Петя видит Машу.", "swapped_actor_and_object"),
    ("Маша не видит Петю.", "changed_negation"),
    ("Маша видит.", "missing_direct_object"),
    ("Маша видит Диму.", "missing_direct_object"),
    ("Маша слышит Петю.", "missing_predicate"),
])
def test_reader_check_catches_lost_or_changed_propositions(text, issue):
    assert issue in surface_issues(thought(), text)


def test_word_order_can_change_without_changing_intended_roles():
    assert not surface_issues(thought(), "Петю видит Маша.")


def test_actor_object_and_negation_are_part_of_message_identity():
    composer = ResponseComposer(analyze_structured_utterance("Маша видит Петю?"))
    plan = MessagePlan(composer.goal, thought())
    assert plan.key != replace(plan, claim=thought("петя", "маша")).key
    assert plan.key != replace(plan, claim=thought(negated=True)).key
    assert plan.key == replace(plan, claim=replace(plan.claim, confidence=.1)).key


def test_content_selection_compares_roles_not_a_bag_of_identical_words():
    composer = ResponseComposer(analyze_structured_utterance("Маша видит Петю?"))
    composer.deferred = True
    correct = composer.express(thought())[0].text
    reversed_roles = composer.express(thought("петя", "маша"))[0].text
    assert composer.priority(correct, {}, .5) > composer.priority(reversed_roles, {}, .99)


def test_known_question_arguments_cannot_be_swapped_before_generation():
    composer = ResponseComposer(analyze_structured_utterance("Кого видит Маша?"))
    assert "question_argument_changed_role" in plan_issues(
        MessagePlan(composer.goal, thought("петя", "маша")),
    )


@pytest.mark.parametrize("subject", ["погода", "машина", "музыка"])
def test_nominal_question_subject_cannot_become_an_assessed_quality(subject):
    composer = ResponseComposer(analyze_structured_utterance(f"Какая сегодня {subject}?"))
    malformed = ClauseThought(
        ConceptLexeme("считать", "считать", "verb"),
        (
            SemanticDependency("subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))),
            SemanticDependency("direct_object", ConceptArgument(
                ConceptLexeme("сегодня", "сегодня", "adverb"),
            )),
            SemanticDependency("predicate_complement", ConceptArgument(
                ConceptLexeme(subject, subject, "noun"),
            )),
        ), confidence=.95,
    )
    assert "question_subject_became_property" in plan_issues(MessagePlan(composer.goal, malformed))


def test_incomplete_meaning_is_rejected_before_surface_generation(monkeypatch):
    composer = ResponseComposer(analyze_structured_utterance("Кто видит Петю?"))
    malformed = replace(thought(), dependencies=thought().dependencies[1:])
    plan = MessagePlan(composer.goal, malformed)
    assert "missing_subject" in plan_issues(plan)
    nlg = Mock()
    monkeypatch.setattr(communication, "realize_thought", nlg)
    assert composer.compose(plan) == ()
    nlg.assert_not_called()


def test_reader_goal_recovers_quantity_word_when_parser_did_not_mark_target():
    goal = reader_goal(analyze_structured_utterance("У тебя сколько бабушек?"))

    assert goal.kind == "answer_slot"
    assert goal.question_word == "сколько"
    assert "бабушка" in goal.topics


def test_failed_verbalization_does_not_silently_drop_the_reason(monkeypatch):
    composer = ResponseComposer(analyze_structured_utterance("Маша видит Петю?"))
    nlg = communication.realize_thought
    reason = thought("коля", "дима")
    monkeypatch.setattr(communication, "realize_thought",
                        lambda t, *a, **k: () if t is reason else nlg(t, *a, **k))
    assert composer.compose(MessagePlan(composer.goal, thought(), reason=reason)) == ()


def test_same_unknown_words_keep_their_independent_query_metadata():
    from debilgi.language.canonical_claims import ClaimQuery
    from debilgi.language.communication import KnowledgeLimit
    from debilgi.language.productive_nlg import unresolved_answer_thought

    structure = analyze_structured_utterance("Кто бежит?")
    composer = ResponseComposer(structure)
    query = ClaimQuery(1, 2, 3, "2026-11-16", "question", predicate="бежать",
                       arguments=(("subject", "concept:медведь"),))
    first = MessagePlan(reader_goal(structure), unresolved_answer_thought(),
                        knowledge_limit=KnowledgeLimit(query, None, "кто"))
    second = replace(first, knowledge_limit=replace(first.knowledge_limit,
                     query=replace(query, arguments=(("subject", "concept:муравей"),))))
    a, b = composer.compose(first)[0], composer.compose(second)[0]
    assert a.text == b.text and first.key != second.key
    for plan in (first, second):
        metadata = composer.metadata_for(a.text, plan_key=plan.key)
        assert metadata["message_plan_key"] == plan.key
        assert metadata["message_plan"]["knowledge_limit"]["query"]["arguments"] == (
            plan.knowledge_limit.query.arguments)


def _embedded_question_surface(text: str, complement=None) -> str:
    from debilgi.language.canonical_claims import ClaimQuery
    from debilgi.language.communication import KnowledgeLimit
    from debilgi.language.productive_nlg import unresolved_answer_thought
    from debilgi.language.semantic_graph import parse_proposition

    structure = analyze_structured_utterance(text)
    parsed = parse_proposition(structure, perspective="user", allow_question=True)
    question = replace(parsed, clause_kind="question")
    if complement is not None:
        question = replace(question, dependencies=tuple(
            replace(d, argument=complement) if d.role == "predicate_complement" else d
            for d in question.dependencies))
    query = ClaimQuery(1, 2, 3, "2026-11-16", "question", predicate="быть")
    assertion = unresolved_answer_thought(confidence=0.95)
    assertion = replace(assertion, dependencies=tuple(
        d for d in assertion.dependencies if d.role == "subject"))
    plan = MessagePlan(reader_goal(structure), assertion,
                       knowledge_limit=KnowledgeLimit(query, question, None))
    return ResponseComposer(structure).compose(plan)[0].text


def test_particle_li_follows_the_nominal_predicate_not_the_subject():
    assert _embedded_question_surface("ты рад?").endswith(", рад ли я.")
    assert _embedded_question_surface("ты веселый?").endswith(", весёлый ли я.")
    assert _embedded_question_surface("ты не рад?").endswith(
        ", рад ли я.")  # the negation of a polar question is its bias
    from debilgi.language.productive_nlg import ConceptArgument, ConceptLexeme

    phrase = ConceptArgument(ConceptLexeme("математик", "математик", "noun", ()),
                             (ConceptLexeme("крутой", "крутой", "adjective", ()),),
                             grammatical_case="nominative")
    assert _embedded_question_surface("ты веселый?", phrase).endswith(", крутой ли я математик.")


def test_particle_li_keeps_verbal_question_order():
    assert _embedded_question_surface("ты хочешь бегать?").endswith(", хочу ли я бегать.")
    assert _embedded_question_surface("ты бегаешь по утрам?").endswith(", бегаю ли я по утрам.")


def test_witnessed_typo_is_spoken_in_dictionary_form_and_rare_unknown_words_are_withheld():
    from debilgi.language.communication import _witnessed_word_surface

    assert _witnessed_word_surface("здравтвуй", 1) == "здравствуй"
    assert _witnessed_word_surface("дадно", 1) is None
    assert _witnessed_word_surface("оооо", 1) == "оооо"
    assert _witnessed_word_surface("!)", 1) == "!)"


def test_repeating_laughter_is_not_mistaken_for_a_typo():
    from debilgi.language.communication import _witnessed_word_surface

    assert _witnessed_word_surface("ахаха", 1) == "ахаха"
    assert _witnessed_word_surface("хехе", 1) == "хехе"


def test_elliptical_why_without_a_delivered_turn_is_not_replaced_by_the_earlier_question():
    from debilgi.language.dialogue_acts import analyze_turn_form
    from debilgi.language.generative_planner import resolve_evidence_context
    from debilgi.language.planning import ReplyContext

    form = analyze_turn_form("почему?", addressed=True, follows_bot=True,
                             previous_bot_text="Наверное, я рад.")
    context = ReplyContext(prompt="почему?", turn_prompt="почему?", turn_form=form,
                           consultation_messages=("user: так ты рад?",), prior_bot_turn=None)
    resolved = resolve_evidence_context(context)
    assert resolved.prompt == "почему?" and resolved.focus_reason == "elliptical_request_recovery"


def test_alternative_meanings_need_independent_support_against_a_repeated_reaction():
    from types import SimpleNamespace

    from debilgi.language.generative_planner import ResponseDecision, compatible_response_decisions

    def form(meaning, support):
        return SimpleNamespace(component="core", meaning=meaning, support=support, activation=0.5,
                               recovery_only=False, hypothesis=(), clause_kind="fragment",
                               rule_id=meaning, evidence_ids=(), pattern_id=meaning,
                               confidence_level="answer", rule_weight=0.7, input_features=(),
                               target_features=(), mixture=(), known_slots=())

    decision = ResponseDecision("free_form", "activated_transformation", "greeting", 0.9, 0.9,
                                None, target_meaning="return_greeting", selection_context={})
    forms = (form("return_greeting", 30), form("acknowledge", 1), form("appraise", 3))
    meanings = [d.target_meaning for d in compatible_response_decisions(decision, forms, limit=4)]
    assert "acknowledge" not in meanings and "return_greeting" in meanings
