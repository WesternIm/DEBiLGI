from debilgi.language.arbitration import (
    ArbitrationProfile,
    ResponseArbitrator,
    ResponseCandidate,
)


def candidate(
    text: str,
    *,
    source: str,
    confidence: float = 0.8,
    renderer: str = "test",
    metadata: dict[str, object] | None = None,
) -> ResponseCandidate:
    return ResponseCandidate.create(
        event_id="event:one",
        source=source,
        renderer=renderer,
        text=text,
        speech_act="question",
        source_confidence=confidence,
        metadata=metadata,
    )


def decide(*candidates: ResponseCandidate):
    return ResponseArbitrator(minimum_score=0.0).decide(
        event_id="event:one",
        prompt="А футбол любишь?",
        focus_concepts=("футбол",),
        candidates=tuple(candidates),
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )


def test_winner_does_not_depend_on_generator_order() -> None:
    relevant = candidate(
        "Да, футбол мне интересен. А тебе какой клуб нравится?",
        source="dialogue_rule",
    )
    unrelated = candidate(
        "Умник — американский телесериал 1987 года.",
        source="knowledge_retrieval",
        confidence=0.97,
    )

    first = decide(relevant, unrelated)
    reversed_order = decide(unrelated, relevant)

    assert first.winner == relevant
    assert reversed_order.winner == relevant


def test_inapplicable_retrieval_is_logged_but_cannot_win() -> None:
    retrieval = candidate(
        "Умник — американский телесериал 1987 года.",
        source="knowledge_retrieval",
        confidence=0.99,
        metadata={"hard_rejection": "retrieval_not_applicable_to_social_act"},
    )
    dialogue = candidate(
        "Да ладно, просто ответила по делу.",
        source="recovery",
        confidence=0.65,
    )

    decision = ResponseArbitrator(minimum_score=0.0).decide(
        event_id="event:one",
        prompt="Ты что, умник теперь?",
        focus_concepts=("умник",),
        candidates=(retrieval, dialogue),
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )
    retrieval_score = next(
        item for item in decision.scores if item.candidate.candidate_id == retrieval.candidate_id
    )

    assert decision.winner == dialogue
    assert not retrieval_score.eligible
    assert "retrieval_not_applicable_to_social_act" in retrieval_score.rejection_reasons


def test_duplicate_surfaces_compete_once_and_keep_stronger_source() -> None:
    weak = candidate("Поняла тебя.", source="graph", confidence=0.4, renderer="weak")
    strong = candidate("Поняла тебя.", source="dialogue_rule", confidence=0.9, renderer="strong")

    decision = decide(weak, strong)

    assert len(decision.scores) == 1
    assert decision.scores[0].candidate == strong


def test_deduplication_keeps_an_eligible_surface_over_a_rejected_copy() -> None:
    rejected = candidate(
        "Поняла тебя.",
        source="knowledge_retrieval",
        confidence=0.99,
        renderer="rejected",
        metadata={"hard_rejection": "not_applicable"},
    )
    eligible = candidate(
        "Поняла тебя.",
        source="dialogue_rule",
        confidence=0.7,
        renderer="eligible",
    )

    decision = decide(rejected, eligible)

    assert len(decision.scores) == 1
    assert decision.scores[0].candidate == eligible
    assert decision.scores[0].eligible


def test_internal_diagnostic_candidate_is_hard_rejected() -> None:
    leaked = candidate(
        "Внутренний разбор: кандидат ответа получил confidence score 0.8.",
        source="graph",
    )

    decision = decide(leaked)

    assert decision.winner is None
    assert "internal_diagnostic_leak" in decision.scores[0].rejection_reasons


def test_learned_command_like_diagnostic_is_never_allowed_back_into_chat() -> None:
    leaked = candidate(
        "команды узнай названного объекта.",
        source="graph",
        confidence=0.99,
    )

    decision = decide(leaked)

    assert decision.winner is None
    assert not decision.scores[0].eligible
    assert "internal_diagnostic_leak" in decision.scores[0].rejection_reasons


def test_logged_incoherent_graph_surface_is_rejected_instead_of_scoring_as_grammar() -> None:
    broken = candidate(
        "узнай?. затем ошибку разобрать. значение слова зависит от контекста: оно может быть.",
        source="graph",
    )

    decision = decide(broken)

    assert decision.winner is None
    assert not decision.scores[0].eligible
    assert decision.scores[0].features.grammar_fit <= 0.20
    assert "surface:malformed_punctuation" in decision.scores[0].rejection_reasons
