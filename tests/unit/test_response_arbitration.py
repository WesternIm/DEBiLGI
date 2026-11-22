import pytest

from debilgi.language.arbitration import (
    ArbitrationProfile,
    ResponseArbitrator,
    ResponseCandidate,
    extract_features,
    merge_recovery_decision,
)
from debilgi.language.communication import ResponseComposer
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.language.structured_utterance import analyze_structured_utterance


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


def grounded_answer(text="Лиса взяла ключ."):
    return ResponseCandidate.create(
        event_id="event:one", source="semantic_reasoning", renderer="productive_roles",
        text=text, speech_act="statement", source_confidence=0.86,
        required_facts=("Лиса взяла ключ.",), metadata={
            "authored_reply_template": False,
            "role_order": ("subject", "predicate", "direct_object"),
            "reasoning_evidence": ("event:reported_action",),
        },
    )


@pytest.mark.parametrize("source", [
    "semantic_reasoning", "proposition_memory", "learned_distribution",
    "learned_dialogue_structure", "interlocutor_clarification", "discord_identity",
    "affective_stance", "new_independent_producer",
])
def test_all_candidates_have_provenance_without_changing_reference_arbitration(source):
    from types import MappingProxyType

    reference = grounded_answer()
    alternative = candidate("А какие собаки?", source=source, metadata={
        "corpus_source_event_ids": ("human:dogs",),
    })
    before = decide(reference, alternative)
    for item in (reference, alternative):
        diagnostic = item.metadata["diagnostic"]
        assert diagnostic["content_source"] and diagnostic["origin"]
        assert diagnostic["plan_id"] and diagnostic["meaning_id"]
        assert diagnostic["plan_basis"] == "surface_only"
        # Reconstruct metadata before stage 0 without rebuilding candidate IDs,
        # content proposals, evidence or any scoring inputs.
        object.__setattr__(item, "metadata", MappingProxyType({
            key: value for key, value in item.metadata.items() if key != "diagnostic"
        }))
    after = decide(reference, alternative)
    assert before.winner == after.winner
    assert before.selection_reason == after.selection_reason
    assert [(s.total, s.features, s.eligible, s.rejection_reasons) for s in before.scores] == [
        (s.total, s.features, s.eligible, s.rejection_reasons) for s in after.scores
    ]


def test_semantic_plan_ids_survive_surface_variants():
    first = candidate("Ты читаешь книгу.", source="semantic_reasoning", metadata={
        "intention_plan_key": "existing-semantic-key", "meaning_intention_id": "intent:one",
    })
    second = candidate("Книгу ты читаешь.", source="semantic_reasoning", metadata={
        "intention_plan_key": "existing-semantic-key", "meaning_intention_id": "intent:two",
    })
    assert first.candidate_id != second.candidate_id
    assert first.metadata["diagnostic"]["plan_id"] == second.metadata["diagnostic"]["plan_id"]
    assert first.metadata["diagnostic"]["meaning_id"] == second.metadata["diagnostic"]["meaning_id"]
    assert first.metadata["diagnostic"]["plan_basis"] == "semantic_plan"


def composed_question(prompt, predicate):
    composer = ResponseComposer(analyze_structured_utterance(prompt))
    thought = ClauseThought(
        ConceptLexeme(predicate, predicate, "verb", (("tense", "past"),)),
        (SemanticDependency("subject", ConceptArgument(ConceptLexeme("что", "что", "pronoun"))),),
        clause_kind="question", focus_role="subject", confidence=0.8,
        evidence=("current_turn:unresolved_event",),
    )
    surface = composer.express(thought, limit=1)[0]
    return surface.text, composer.metadata_for(surface.text)


@pytest.mark.parametrize("predicate", ["произойти", "измениться", "случиться"])
@pytest.mark.parametrize("source", ["dialogue_move", "new_independent_producer"])
def test_checked_dialogue_move_survives_zero_word_overlap_and_learned_penalties(predicate, source):
    text, plan_metadata = composed_question("пиздээ", predicate)
    answer = candidate(text, source=source, confidence=0.43,
        metadata={
            **plan_metadata,
            "reader_checked": True, "locally_composed": True,
            "reasoning_evidence": ("incomplete_dialogue_act",), "obligation_fit": 1.0,
            "context_relevance": 0.0, "semantic_frame_fit": 0.73,
        })
    decision = ResponseArbitrator(minimum_score=0.45).decide(
        event_id="event:one", prompt="пиздээ", focus_concepts=("пиздээ",),
        candidates=(answer,), recent_replies=(), profile=ArbitrationProfile.baseline(1),
        structural_weights={f"source:{source}": -100.0},
    )
    assert decision.winner == answer
    assert decision.scores[0].answer_status == "dialogue_response"
    assert decision.scores[0].features.context_relevance == 1.0


def test_dialogue_move_cannot_replace_a_factual_answer_by_claiming_high_fit():
    text, plan_metadata = composed_question("Сколько стоит гречка?", "произойти")
    answer = candidate(text, source="dialogue_move", confidence=0.43,
        metadata={
            **plan_metadata,
            "response_obligation": "answer_question", "response_move": "probe_event",
            "reader_checked": True, "locally_composed": True,
            "reasoning_evidence": ("incomplete_dialogue_act",), "obligation_fit": 1.0,
            "context_relevance": 0.0,
        })
    features = extract_features(answer, prompt="Сколько стоит гречка?",
                                focus_concepts=(), recent_replies=())
    assert features.context_relevance == 0.0


def test_negative_wording_feedback_cannot_suppress_a_grounded_answer():
    answer = grounded_answer()
    kwargs = dict(
        event_id="event:one", prompt="Что взяла лиса?", focus_concepts=("лиса",),
        candidates=(answer,), recent_replies=(), profile=ArbitrationProfile.baseline(1),
    )
    initial = ResponseArbitrator(minimum_score=0.0).decide(**kwargs)
    threshold = initial.scores[0].total - 0.03
    learned = ResponseArbitrator(minimum_score=threshold).decide(
        **kwargs, structural_weights={"source:semantic_reasoning": -100.0},
    )
    assert learned.scores[0].total < threshold
    assert learned.scores[0].admission_score >= threshold
    assert learned.winner == answer
    assert learned.selection_reason == "grounded_answer_then_surface"


def test_recovery_self_confidence_cannot_outbid_a_grounded_answer():
    answer = grounded_answer()
    recovery = candidate("Я не знаю ответа.", source="semantic_recovery", confidence=1.0,
                         metadata={"context_relevance": 1.0, "truth_grounding": 1.0,
                                   "semantic_frame_fit": 1.0})
    decision = ResponseArbitrator(minimum_score=0.0).decide(
        event_id="event:one", prompt="Что взяла лиса?", focus_concepts=("лиса",),
        candidates=(answer, recovery), recent_replies=(), profile=ArbitrationProfile.baseline(1),
        structural_weights={"source:semantic_reasoning": -100.0,
                            "source:semantic_recovery": 100.0},
    )
    assert decision.winner == answer
    statuses = {score.candidate.source: score.answer_status for score in decision.scores}
    assert statuses == {"semantic_reasoning": "grounded_answer", "semantic_recovery": "recovery"}


def test_inapplicable_high_confidence_duplicate_does_not_hide_valid_answer():
    invalid = candidate("Лиса взяла ключ.", source="knowledge_retrieval", confidence=1.0,
                        metadata={"applicability": 0.0})
    answer = grounded_answer()
    assert decide(invalid, answer).winner == answer


def test_hard_semantic_failure_cannot_be_rescued_by_high_scores():
    decision = ResponseArbitrator(minimum_score=0.0).decide(
        event_id="event:one", prompt="Что взяла лиса?", focus_concepts=("лиса",),
        candidates=(grounded_answer("Лиса не взяла ключ."),), recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
        structural_weights={"source:semantic_reasoning": 100.0},
    )
    assert decision.winner is None
    assert decision.scores[0].admission_score < 0.0


def test_content_preselection_cannot_hide_the_only_valid_answer():
    rejected = tuple(candidate(
        "Я отвечаю.", source=f"source_{i}", confidence=1.0,
        metadata={"hard_rejection": "test_invalid", "response_obligation": "answer_question"},
    ) for i in range(13))
    # Distinct surfaces prevent deduplication from masking the pruning problem.
    from dataclasses import replace
    rejected = tuple(replace(c, text=f"Ответ номер {i}.") for i, c in enumerate(rejected))
    answer = grounded_answer()
    decision = ResponseArbitrator(minimum_score=0.0).decide(
        event_id="event:one", prompt="Что взяла лиса?", focus_concepts=("лиса",),
        candidates=(*rejected, answer), recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )
    assert len(decision.scores) == 14
    assert decision.winner == answer


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


def test_composed_case_error_is_a_penalty_without_becoming_a_learning_example():
    from debilgi.language.surface_coherence import evaluate_surface

    rough = "Я управляю машину."
    assert "governed_case_mismatch" in evaluate_surface(rough).hard_issues
    composed = candidate(rough, source="semantic_reasoning", metadata={
        "locally_composed": True, "allow_rough_surface": True,
        "role_order": ("subject", "predicate", "direct_object"),
    })
    copied = candidate(rough, source="semantic_synthesis", metadata={
        "authored_reply_template": False, "allow_rough_surface": True,
    })
    assert ResponseArbitrator._hard_rejections(composed) == ()
    assert "surface:governed_case_mismatch" in ResponseArbitrator._hard_rejections(copied)
    decision = decide(composed)
    assert decision.winner == composed
    assert decision.scores[0].features.broken_surface_penalty == 0.15
    assert not evaluate_surface(rough).learnable


def test_rough_composition_does_not_excuse_invalid_semantic_roles():
    invalid = candidate("Я поперхнулся мозг.", source="semantic_reasoning", metadata={
        "locally_composed": True, "allow_rough_surface": True,
        "role_order": ("subject", "predicate", "direct_object"),
    })
    assert "surface:unexpected_valency_role" in ResponseArbitrator._hard_rejections(invalid)


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


def test_recent_reply_penalty_is_separate_from_reusing_words_inside_phrase():
    from debilgi.language.arbitration import extract_features

    item = candidate("Да да!", source="semantic_reasoning")
    fresh = extract_features(item, prompt="согласен?", focus_concepts=(), recent_replies=())
    repeated = extract_features(item, prompt="согласен?", focus_concepts=(),
                                recent_replies=("да да.",))
    assert fresh.intra_phrase_repetition > 0 and fresh.repetition_penalty == 0
    assert repeated.repetition_penalty == 1
    assert repeated.intra_phrase_repetition == fresh.intra_phrase_repetition


def test_neutral_subject_order_beats_marked_final_personal_subject():
    neutral = candidate("Я сказал «Привет».", source="semantic_reasoning", metadata={
        "role_order": ("subject", "predicate", "direct_object"),
        "choice_context": {"enabled": True, "personality": {"patience": .5}},
    })
    marked = candidate("Сказал «Привет» я.", source="semantic_reasoning", metadata={
        "role_order": ("predicate", "direct_object", "subject"),
        "choice_context": {"enabled": True, "personality": {"patience": .5}},
    })
    decision = decide(marked, neutral)
    assert decision.winner.text == neutral.text
    assert decision.winner.metadata["surface_choice"]["considered"] == 2


def test_receipt_recency_distinguishes_two_previously_used_surfaces():
    from debilgi.language.dialogue_learning import response_repetition

    recent = ("Я не знаю ответа.", "Я не знаю ответа на этот вопрос.")
    assert response_repetition(recent[0], recent) > response_repetition(recent[1], recent)
    assert response_repetition("Я знаю ответ.", recent) == 0


def test_hedge_operator_calibrates_affirmed_and_negated_guesses_alike():
    plan = {"hypothesis": [["kind", "experience_opinion"], ["operator", "наверное"]]}

    def calibration(text):
        item = candidate(text, source="semantic_reasoning", confidence=0.5,
                         metadata={"message_plan": plan})
        return extract_features(item, prompt="так ты рад?", focus_concepts=(),
                                recent_replies=()).uncertainty_calibration

    assert calibration("Наверное, я рад.") == calibration("Наверное, я не рад.") == 1.0
