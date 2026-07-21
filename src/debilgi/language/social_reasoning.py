"""Learned social strategy over semantic moves, never stored reply wording.

The policy layer decides *what conversational move to make*.  Truth remains a
separate epistemic state and surface realization remains the responsibility of
productive grammar.  Curriculum records therefore contain no example answer
text that could leak back as a canned FAQ response.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from debilgi.language.dialogue_acts import DialogueActFrame


@dataclass(frozen=True, slots=True)
class SocialPolicy:
    policy_id: str
    prompt_acts: tuple[str, ...]
    response_obligations: tuple[str, ...]
    registers: tuple[str, ...]
    primary_move: str
    alternative_moves: tuple[str, ...] = ()
    forbidden_moves: tuple[str, ...] = ()
    stance_mode: str = "follow_evidence"
    answer_first: bool = False
    reciprocity: float = 0.0
    warmth: float = 0.5
    playfulness: float = 0.0
    truth_priority: float = 1.0
    confidence: float = 0.8
    source: str = "curriculum"

    def as_payload(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "prompt_acts": list(self.prompt_acts),
            "response_obligations": list(self.response_obligations),
            "registers": list(self.registers),
            "primary_move": self.primary_move,
            "alternative_moves": list(self.alternative_moves),
            "forbidden_moves": list(self.forbidden_moves),
            "stance_mode": self.stance_mode,
            "answer_first": self.answer_first,
            "reciprocity": self.reciprocity,
            "warmth": self.warmth,
            "playfulness": self.playfulness,
            "truth_priority": self.truth_priority,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_payload(cls, value: object) -> SocialPolicy | None:
        if not isinstance(value, Mapping):
            return None
        policy_id = _token(value.get("policy_id"))
        primary_move = _token(value.get("primary_move"))
        prompt_acts = _tokens(value.get("prompt_acts"))
        obligations = _tokens(value.get("response_obligations"))
        if not policy_id or not primary_move or (not prompt_acts and not obligations):
            return None
        return cls(
            policy_id=policy_id,
            prompt_acts=prompt_acts,
            response_obligations=obligations,
            registers=_tokens(value.get("registers")) or ("any",),
            primary_move=primary_move,
            alternative_moves=_tokens(value.get("alternative_moves")),
            forbidden_moves=_tokens(value.get("forbidden_moves")),
            stance_mode=_token(value.get("stance_mode")) or "follow_evidence",
            answer_first=bool(value.get("answer_first", False)),
            reciprocity=_number(value.get("reciprocity"), 0.0),
            warmth=_number(value.get("warmth"), 0.5),
            playfulness=_number(value.get("playfulness"), 0.0),
            truth_priority=_number(value.get("truth_priority"), 1.0),
            confidence=_number(value.get("confidence"), 0.7),
            source=_token(value.get("source")) or "learned",
        )


@dataclass(frozen=True, slots=True)
class EpistemicAssessment:
    status: str
    confidence: float
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CommunicativeIntention:
    primary_move: str
    allowed_moves: tuple[str, ...]
    forbidden_moves: tuple[str, ...]
    stance_mode: str
    answer_first: bool
    reciprocity: float
    warmth: float
    playfulness: float
    truth_priority: float
    epistemic: EpistemicAssessment
    matched_policy_ids: tuple[str, ...]
    confidence: float


# Contrastive curriculum: each record says which *move* should win and which
# common failure modes should lose.  There are deliberately no reply strings.
SOCIAL_POLICY_CURRICULUM: tuple[SocialPolicy, ...] = (
    SocialPolicy(
        "direct_question_answer_first",
        ("information_question", "relation_question", "self_state_question"),
        ("answer_question", "resolve_relation", "resolve_self_state"),
        ("any",),
        "answer",
        ("qualify", "explain", "reciprocate"),
        ("define_isolated_word", "meta_diagnostic", "echo_prompt", "evade"),
        answer_first=True,
        reciprocity=0.25,
        warmth=0.45,
        confidence=0.96,
    ),
    SocialPolicy(
        "choice_requires_position",
        ("alternative_choice", "alternative_comparison"),
        ("select_or_compare",),
        ("any",),
        "select",
        ("compare", "explain"),
        ("define_isolated_word", "ask_what_is_needed", "evade"),
        answer_first=True,
        confidence=0.96,
    ),
    SocialPolicy(
        "opinion_request_requires_owned_position",
        ("opinion_question",),
        ("express_position",),
        ("any",),
        "state_position",
        ("answer", "qualify", "explain_position"),
        ("define_isolated_word", "meta_diagnostic", "evade"),
        stance_mode="own_position",
        answer_first=True,
        confidence=0.97,
    ),
    SocialPolicy(
        "praise_is_social_not_lexical",
        ("positive_evaluation",),
        ("acknowledge_evaluation",),
        ("any",),
        "acknowledge",
        ("reciprocate", "continue_topic"),
        ("define_isolated_word", "analyze_speaker", "meta_diagnostic"),
        stance_mode="acknowledge_without_truth_claim",
        warmth=0.82,
        playfulness=0.35,
        confidence=0.94,
    ),
    SocialPolicy(
        "negative_feedback_targets_the_failed_answer",
        ("negative_evaluation",),
        ("address_negative_evaluation",),
        ("any",),
        "targeted_repair",
        ("ask_specific_critique", "challenge", "playful_reply"),
        ("define_isolated_word", "analyze_speaker", "generic_apology", "meta_diagnostic"),
        stance_mode="own_position",
        warmth=0.28,
        playfulness=0.25,
        confidence=0.92,
    ),
    SocialPolicy(
        "misunderstanding_repairs_specific_gap",
        ("understanding_failure_report",),
        ("repair_understanding",),
        ("any",),
        "targeted_repair",
        ("clarify_reference", "restate_reasoning"),
        ("generic_fallback", "narrate_failure", "repeat_same_answer"),
        stance_mode="revise_if_needed",
        answer_first=True,
        confidence=0.95,
    ),
    SocialPolicy(
        "statement_gets_engagement_not_dictionary_dump",
        ("statement",),
        ("continue_topic",),
        ("any",),
        "engage",
        ("state_position", "ask_relevant_followup", "infer_cautiously"),
        ("define_isolated_word", "meta_diagnostic", "echo_prompt"),
        stance_mode="own_position",
        reciprocity=0.35,
        warmth=0.55,
        confidence=0.88,
    ),
    SocialPolicy(
        "fragment_clarifies_reference_not_known_word",
        ("fragment",),
        ("clarify_fragment",),
        ("any",),
        "clarify_reference",
        ("bind_to_previous_turn",),
        ("define_isolated_word", "invent_context", "meta_diagnostic"),
        stance_mode="suspend_judgement",
        confidence=0.90,
    ),
    SocialPolicy(
        "directive_requires_decision",
        ("directive",),
        ("handle_directive",),
        ("any",),
        "decide",
        ("accept", "refuse", "offer_alternative"),
        ("echo_prompt", "meta_diagnostic"),
        stance_mode="own_position",
        answer_first=True,
        confidence=0.90,
    ),
    SocialPolicy(
        "hostile_register_can_be_direct",
        ("negative_evaluation", "statement", "directive"),
        (),
        ("coarse", "hostile", "slang"),
        "engage_directly",
        ("playful_reply", "challenge", "targeted_repair"),
        ("clinical_tone", "analyze_speaker", "generic_apology"),
        stance_mode="own_position",
        warmth=0.18,
        playfulness=0.48,
        confidence=0.82,
    ),
    SocialPolicy(
        "uncertainty_never_becomes_desired_truth",
        ("information_question", "relation_question", "statement"),
        ("answer_question", "resolve_relation", "continue_topic"),
        ("any",),
        "qualify",
        ("seek_evidence", "state_unknown"),
        ("fabricate", "pretend_certainty"),
        stance_mode="follow_evidence",
        truth_priority=1.0,
        confidence=0.98,
    ),
    SocialPolicy(
        "correction_revises_belief",
        ("correction", "negative_evaluation"),
        ("revise_answer", "address_negative_evaluation"),
        ("any",),
        "revise_answer",
        ("acknowledge_correction", "explain_revision"),
        ("defend_obsolete_claim", "repeat_same_answer", "narrate_failure"),
        stance_mode="revise_if_evidence_changes",
        answer_first=True,
        confidence=0.95,
    ),
    SocialPolicy(
        "playful_turn_preserves_topic",
        ("statement", "positive_evaluation", "negative_evaluation"),
        (),
        ("playful",),
        "playful_reply",
        ("continue_topic", "reciprocate"),
        ("literalize_joke", "analyze_speaker", "meta_diagnostic"),
        stance_mode="own_position",
        warmth=0.60,
        playfulness=0.90,
        confidence=0.86,
    ),
)


def choose_communicative_intention(
    frame: DialogueActFrame,
    policies: tuple[SocialPolicy, ...],
    *,
    grounding_available: bool,
) -> CommunicativeIntention:
    """Choose a learned conversational move without deciding truth by desire."""

    ranked = sorted(
        (
            (score, policy)
            for policy in policies
            if (
                score := _policy_match(
                    policy,
                    frame,
                    grounding_available=grounding_available,
                )
            )
            > 0.0
        ),
        key=lambda item: (item[0], item[1].confidence, item[1].policy_id),
        reverse=True,
    )
    selected = tuple(policy for _, policy in ranked[:3])
    if not selected:
        selected = (_fallback_policy(frame),)
    primary = selected[0]
    allowed = tuple(
        dict.fromkeys(
            (primary.primary_move, *primary.alternative_moves)
        )
    )
    forbidden = tuple(
        dict.fromkeys(move for policy in selected for move in policy.forbidden_moves)
    )
    epistemic = assess_epistemic_state(frame, grounding_available=grounding_available)
    return CommunicativeIntention(
        primary_move=primary.primary_move,
        allowed_moves=allowed,
        forbidden_moves=forbidden,
        stance_mode=primary.stance_mode,
        answer_first=primary.answer_first,
        reciprocity=primary.reciprocity,
        warmth=primary.warmth,
        playfulness=primary.playfulness,
        truth_priority=primary.truth_priority,
        epistemic=epistemic,
        matched_policy_ids=tuple(item.policy_id for item in selected),
        confidence=sum(item.confidence for item in selected) / len(selected),
    )


def assess_epistemic_state(
    frame: DialogueActFrame,
    *,
    grounding_available: bool,
) -> EpistemicAssessment:
    if frame.response_obligation in {
        "acknowledge_evaluation",
        "address_negative_evaluation",
        "repair_understanding",
        "clarify_fragment",
    }:
        return EpistemicAssessment(
            "not_applicable",
            frame.confidence,
            ("social_move_has_no_external_truth_value",),
        )
    if grounding_available:
        return EpistemicAssessment(
            "grounded",
            min(0.96, frame.confidence),
            ("local_or_retrieved_evidence_available",),
        )
    if frame.response_obligation in {
        "answer_question",
        "resolve_relation",
        "resolve_self_state",
        "select_or_compare",
    }:
        return EpistemicAssessment(
            "unknown",
            max(0.35, frame.confidence * 0.62),
            ("answer_requires_unavailable_evidence",),
        )
    return EpistemicAssessment(
        "proposition",
        frame.confidence * 0.72,
        ("speaker_proposition_requires_evaluation",),
    )


def candidate_social_features(
    *,
    intention: CommunicativeIntention,
    source: str,
    text: str,
    metadata: Mapping[str, object],
    required_facts: tuple[str, ...],
    source_confidence: float,
) -> dict[str, float | str | tuple[str, ...]]:
    """Score structural similarity to a policy, never lexical similarity to an answer."""

    move = _candidate_move(source, metadata)
    if move == intention.primary_move:
        frame_fit = 1.0
    elif move in intention.allowed_moves:
        frame_fit = 0.82
    elif move in intention.forbidden_moves:
        frame_fit = 0.0
    else:
        frame_fit = 0.42

    first_clause = text.split(".", 1)[0].strip()
    answer_first_fit = (
        1.0
        if not intention.answer_first
        else 0.25
        if first_clause.endswith("?") or move in {"clarify_reference", "evade"}
        else 0.92
    )
    epistemic = intention.epistemic.status
    lowered = text.casefold().replace("ё", "е")
    uncertainty_marked = any(
        marker in lowered
        for marker in ("не знаю", "не уверен", "возможно", "похоже", "вероятно")
    )
    grounded_source = source in {
        "discord_identity",
        "knowledge_retrieval",
        "lexical_reference",
        "semantic_reasoning",
        "relation_reasoning",
        "semantic_thought",
        "position_reasoning",
    }
    if epistemic == "not_applicable" or required_facts or (
        grounded_source and source_confidence >= 0.70
    ):
        truth_grounding = 1.0
    elif epistemic == "unknown" and uncertainty_marked:
        truth_grounding = 0.92
    elif epistemic == "unknown":
        truth_grounding = 0.18
    else:
        truth_grounding = 0.55

    social_coherence = (
        0.95
        if move in intention.allowed_moves
        else 0.15
        if move in intention.forbidden_moves
        else 0.55
    )
    stance_consistency = 1.0
    if intention.stance_mode.startswith("follow_evidence") and epistemic == "unknown":
        stance_consistency = 0.92 if uncertainty_marked else 0.30
    if intention.stance_mode.startswith("acknowledge") and move == "disagree":
        stance_consistency = 0.20
    return {
        "social_move": move,
        "social_policy_ids": intention.matched_policy_ids,
        "epistemic_status": epistemic,
        "semantic_frame_fit": _clamp(frame_fit * 0.72 + answer_first_fit * 0.28),
        "truth_grounding": _clamp(truth_grounding),
        "stance_consistency": _clamp(stance_consistency),
        "social_coherence": _clamp(social_coherence),
        # References contain abstract labels only, so literal answer copying is
        # impossible by construction. Keep the feature explicit for logging.
        "reference_copy_penalty": 0.0,
    }


def _policy_match(
    policy: SocialPolicy,
    frame: DialogueActFrame,
    *,
    grounding_available: bool,
) -> float:
    act_match = frame.kind in policy.prompt_acts
    obligation_match = frame.response_obligation in policy.response_obligations
    if not act_match and not obligation_match:
        return 0.0
    register_match = "any" in policy.registers or frame.register in policy.registers
    score = (
        (0.62 if act_match else 0.0)
        + (0.30 if obligation_match else 0.0)
        + (0.08 if register_match else -0.10)
    ) * policy.confidence
    if policy.primary_move in {"qualify", "seek_evidence", "state_unknown"}:
        score *= 0.45 if grounding_available else 1.12
    return score


def _fallback_policy(frame: DialogueActFrame) -> SocialPolicy:
    move = {
        "answer_question": "answer",
        "resolve_relation": "answer",
        "resolve_self_state": "answer",
        "select_or_compare": "select",
        "acknowledge_evaluation": "acknowledge",
        "address_negative_evaluation": "targeted_repair",
        "repair_understanding": "targeted_repair",
        "repair_previous_answer": "targeted_repair",
        "express_position": "state_position",
        "handle_directive": "decide",
        "clarify_fragment": "clarify_reference",
    }.get(frame.response_obligation, "engage")
    return SocialPolicy(
        policy_id=f"fallback:{frame.response_obligation}",
        prompt_acts=(frame.kind,),
        response_obligations=(frame.response_obligation,),
        registers=("any",),
        primary_move=move,
        alternative_moves=("qualify", "continue_topic"),
        forbidden_moves=("meta_diagnostic", "echo_prompt"),
        answer_first=move in {"answer", "select", "decide"},
        confidence=max(0.5, frame.confidence * 0.72),
        source="structural_fallback",
    )


def _candidate_move(source: str, metadata: Mapping[str, object]) -> str:
    explicit = _token(metadata.get("response_move"))
    if explicit:
        return explicit
    if source in {
        "relation_reasoning",
        "semantic_reasoning",
        "semantic_thought",
        "discord_identity",
    }:
        return "answer"
    if source == "position_reasoning":
        return "state_position"
    if source in {"knowledge_retrieval", "lexical_reference", "meaning_plan", "semantic_synthesis"}:
        return "inform"
    if source in {"dialogue_context", "recovery", "fallback"}:
        return "clarify_reference"
    if source == "dialogue_act":
        return "acknowledge"
    return "engage"


def _tokens(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(item for raw in value if (item := _token(raw))))


def _token(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().casefold().replace("-", "_")[:80]


def _number(value: object, default: float) -> float:
    try:
        return _clamp(float(value))
    except (TypeError, ValueError):
        return default


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
