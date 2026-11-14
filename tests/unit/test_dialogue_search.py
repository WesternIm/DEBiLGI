from debilgi.language.communication import reader_goal
from debilgi.language.dialogue_search import query_unobserved_relation, search_continuations
from debilgi.language.generative_planner import statement_thought
from debilgi.language.structured_utterance import analyze_structured_utterance


def test_search_returns_a_semantic_continuation_without_a_reply_template():
    prompt = "а машины то некоторые умеют летать"
    claim = statement_thought(prompt, speaker="user")
    assert claim is not None
    candidates = search_continuations(
        reader_goal(analyze_structured_utterance(prompt)), claim, evidence_id="event:1"
    )
    assert candidates
    candidate = candidates[0]
    assert candidate.operation == "resolve_open_group"
    assert candidate.plan.claim.clause_kind == "question"
    assert candidate.plan.claim.evidence == ("event:1",)
    assert not hasattr(candidate, "text")


def test_complete_personal_report_can_have_an_unobserved_relation_without_a_noun_object():
    prompt = "Я отвечаю."
    claim = statement_thought(prompt, speaker="user")
    assert claim is not None
    candidates = query_unobserved_relation(
        reader_goal(analyze_structured_utterance(prompt)),
        claim,
        evidence_id="report:1",
    )
    assert candidates
    assert candidates[0].plan.claim.clause_kind == "question"
    assert candidates[0].plan.claim.predicate.lemma == claim.predicate.lemma
    assert all(dep in candidates[0].plan.claim.dependencies for dep in claim.dependencies)
