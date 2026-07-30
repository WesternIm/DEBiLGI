from debilgi.language.arbitration import (
    ArbitrationProfile,
    ResponseArbitrator,
    ResponseCandidate,
    extract_features,
    merge_recovery_decision,
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


def test_duplicate_sentences_are_a_hard_surface_failure() -> None:
    broken = candidate(
        "Я отвечаю. Я отвечаю.",
        source="autonomous_semantic_plan",
    )

    decision = decide(broken)

    assert decision.winner is None
    assert "surface:duplicate_sentence" in decision.scores[0].rejection_reasons


def test_predicate_participle_cannot_pass_as_its_object() -> None:
    broken = candidate(
        "Я отвечаю отвечающую.",
        source="autonomous_semantic_plan",
    )

    decision = decide(broken)

    assert decision.winner is None
    assert (
        "surface:predicate_participle_as_object"
        in decision.scores[0].rejection_reasons
    )


def test_unknown_features_stay_unknown_instead_of_becoming_middle_scores() -> None:
    features = extract_features(
        candidate("Поняла тебя.", source="graph"),
        prompt="Расскажи подробнее",
        focus_concepts=(),
        recent_replies=(),
    )

    assert features.applicability is None
    assert features.reply_binding is None
    assert features.fact_grounding is None
    assert features.stance_consistency is None
    assert features.social_coherence is None


def test_required_fact_polarity_is_a_hard_invariant() -> None:
    contradicted = ResponseCandidate.create(
        event_id="event:polarity",
        source="semantic_reasoning",
        renderer="test",
        text="Макс не любит футбол.",
        speech_act="statement",
        source_confidence=0.9,
        required_facts=("Макс любит футбол.",),
    )

    decision = ResponseArbitrator(minimum_score=0.0).decide(
        event_id="event:polarity",
        prompt="Макс любит футбол?",
        focus_concepts=("макс", "футбол"),
        candidates=(contradicted,),
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )

    assert decision.winner is None
    assert not decision.scores[0].eligible
    assert any(
        reason.startswith("missing_required_fact:")
        for reason in decision.scores[0].rejection_reasons
    )


def test_structured_required_fact_is_persistable_metadata() -> None:
    grounded = ResponseCandidate.create(
        event_id="event:structured",
        source="semantic_reasoning",
        renderer="test",
        text="Макс не любит футбол.",
        speech_act="statement",
        source_confidence=0.9,
        required_facts=("Макс не любит футбол.",),
    )

    fact = grounded.required_facts[0]
    assert fact.subject
    assert fact.predicate == "любить"
    assert fact.object
    assert fact.polarity == "negative"


def test_content_is_ranked_before_its_surface_wording() -> None:
    first = candidate(
        "Футбол мне интересен.",
        source="dialogue_rule",
        metadata={"response_obligation": "express_position"},
    )
    second = candidate(
        "Да, мне интересен футбол.",
        source="dialogue_rule",
        renderer="alternative",
        metadata={"response_obligation": "express_position"},
    )

    decision = decide(first, second)

    assert len(decision.content_scores) == 1
    assert decision.selected_proposal_ids == (
        first.proposal.proposal_id,
    )
    assert first.proposal == second.proposal
    assert not hasattr(first.proposal.plan, "text")


def test_recovery_merge_keeps_content_for_every_persisted_surface() -> None:
    primary_candidate = candidate(
        "Собачки умеют бегать.",
        source="semantic_reasoning",
    )
    primary = ResponseArbitrator(minimum_score=100.0).decide(
        event_id="event:one",
        prompt="Собачки умеют бегать?",
        focus_concepts=("собачка", "бегать"),
        candidates=(primary_candidate,),
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )
    recovery_candidate = candidate(
        "Уточни, о каких собачках речь?",
        source="fallback",
        renderer="conversational_repair",
    )
    recovery = ResponseArbitrator(minimum_score=-100.0).decide(
        event_id="event:one",
        prompt="Собачки умеют бегать?",
        focus_concepts=("собачка", "бегать"),
        candidates=(recovery_candidate,),
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
        used_fallback=True,
    )

    merged = merge_recovery_decision(primary, recovery)
    persisted_proposals = {
        item.proposal.proposal_id for item in merged.content_scores
    }
    referenced_proposals = {
        item.candidate.proposal.proposal_id for item in merged.scores
    }

    assert merged.winner == recovery.winner
    assert merged.used_fallback
    assert len(merged.scores) == 2
    assert referenced_proposals <= persisted_proposals
