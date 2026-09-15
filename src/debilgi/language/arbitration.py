"""Cross-source response candidates, common scoring and deterministic arbitration."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from types import MappingProxyType
from uuid import uuid4

from debilgi.language.clarification_policy import clarification_rejection
from debilgi.language.construction_nlg import ResponseContentPlan
from debilgi.language.content import content_lemmas
from debilgi.language.facts import (
    StructuredFact,
    covers_structured_fact,
)
from debilgi.language.lexical_gaps import irrelevant_lexical_question
from debilgi.language.response_requirements import requirement_rejection
from debilgi.language.structural_learning import (
    learned_structural_bonus,
    structural_response_features,
)
from debilgi.language.surface_coherence import COMPOSED_SURFACE_SOFT_ISSUES, evaluate_surface
from debilgi.language.surface_policy import internal_surface_reason
from debilgi.language.tokenizer import tokenize

_TECHNICAL_RE = re.compile(r"<(?:user|role|channel|url|emoji:[^>]+)>", re.IGNORECASE)
_UNCERTAINTY_MARKERS = frozenset(
    {"похоже", "возможно", "вероятно", "кажется", "не", "уверен", "точно"}
)


@dataclass(frozen=True, slots=True)
class ContentProposal:
    proposal_id: str
    source: str
    plan: ResponseContentPlan
    applicability: float | None
    source_confidence: float
    evidence_ids: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        source: str,
        plan: ResponseContentPlan,
        source_confidence: float,
        applicability: float | None = None,
        evidence_ids: tuple[str, ...] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> ContentProposal:
        material = "\x1f".join(
            (
                event_id,
                source,
                plan.speech_act,
                plan.response_goal or plan.response_act,
                "|".join(fact.text for fact in plan.required_facts),
                "|".join(plan.key_points),
                str((metadata or {}).get("message_plan_key", "")),
            )
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        return cls(
            proposal_id=f"proposal:{digest}",
            source=source,
            plan=plan,
            applicability=_clamp(applicability) if applicability is not None else None,
            source_confidence=_clamp(source_confidence),
            evidence_ids=tuple(dict.fromkeys(evidence_ids)),
            metadata=MappingProxyType(dict(metadata or {})),
        )


@dataclass(frozen=True, slots=True)
class SurfaceCandidate:
    candidate_id: str
    proposal: ContentProposal
    source: str
    renderer: str
    text: str
    speech_act: str
    source_confidence: float
    required_facts: tuple[StructuredFact, ...] = ()
    key_points: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        source: str,
        renderer: str,
        text: str,
        speech_act: str,
        source_confidence: float,
        required_facts: tuple[str | StructuredFact, ...] = (),
        key_points: tuple[str, ...] = (),
        metadata: Mapping[str, object] | None = None,
        content_plan: ResponseContentPlan | None = None,
        proposal: ContentProposal | None = None,
    ) -> SurfaceCandidate:
        candidate_metadata = MappingProxyType(dict(metadata or {}))
        fact_target = (
            f"user:{candidate_metadata['target_id']}"
            if isinstance(candidate_metadata.get("target_id"), int)
            and not isinstance(candidate_metadata.get("target_id"), bool)
            else None
        )
        structured_facts = tuple(
            fact
            if isinstance(fact, StructuredFact)
            else StructuredFact.from_text(
                fact,
                certainty=source_confidence,
                target=fact_target,
                provenance=(event_id, source),
            )
            for fact in required_facts
        )
        if proposal is None:
            measured_applicability = candidate_metadata.get("applicability")
            applicability = (
                float(measured_applicability)
                if isinstance(measured_applicability, int | float)
                and not isinstance(measured_applicability, bool)
                else None
            )
            plan = content_plan or ResponseContentPlan(
                response_act=speech_act,
                answer_polarity=None,
                clauses=(),
                confidence=_clamp(source_confidence),
                response_goal=str(candidate_metadata.get("response_obligation", speech_act)),
                target_id=(
                    int(candidate_metadata["target_id"])
                    if isinstance(candidate_metadata.get("target_id"), int)
                    and not isinstance(candidate_metadata.get("target_id"), bool)
                    else None
                ),
                topic_ids=tuple(
                    dict.fromkeys(
                        term
                        for value in (*key_points, *(fact.text for fact in structured_facts))
                        for term in content_lemmas(value)
                    )
                ),
                key_points=key_points,
                required_facts=structured_facts,
                evidence=(event_id, source),
            )
            proposal = ContentProposal.create(
                event_id=event_id,
                source=source,
                plan=plan,
                source_confidence=source_confidence,
                applicability=applicability,
                evidence_ids=tuple(
                    evidence for fact in structured_facts for evidence in fact.provenance
                ),
                metadata=candidate_metadata,
            )
        material = "\x1f".join((event_id, source, renderer, " ".join(text.split())))
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        return cls(
            candidate_id=f"candidate:{digest}",
            proposal=proposal,
            source=source,
            renderer=renderer,
            text=text.strip(),
            speech_act=speech_act,
            source_confidence=_clamp(source_confidence),
            required_facts=structured_facts,
            key_points=key_points,
            metadata=candidate_metadata,
        )


# Compatibility name for integrations built before the content/surface split.
ResponseCandidate = SurfaceCandidate


@dataclass(frozen=True, slots=True)
class CandidateFeatures:
    applicability: float | None
    context_relevance: float
    source_confidence: float
    reply_binding: float | None
    speech_act_fit: float
    plan_coverage: float
    fact_grounding: float | None
    topic_continuity: float
    grammar_fit: float
    specificity: float
    length_fit: float
    non_repetition: float
    uncertainty_calibration: float
    truth_grounding: float | None
    stance_consistency: float | None
    semantic_frame_fit: float
    social_coherence: float | None
    prompt_echo_penalty: float
    reference_copy_penalty: float
    retrieval_dump_penalty: float
    repetition_penalty: float
    broken_surface_penalty: float

    def as_dict(self) -> dict[str, float | None]:
        return {
            name: float(value) if value is not None else None
            for name in self.__dataclass_fields__
            for value in (getattr(self, name),)
        }


DEFAULT_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "applicability": 1.40,
        "context_relevance": 1.35,
        "source_confidence": 1.15,
        "reply_binding": 0.65,
        "speech_act_fit": 1.10,
        "plan_coverage": 1.25,
        "fact_grounding": 1.25,
        "topic_continuity": 1.10,
        "grammar_fit": 0.90,
        "specificity": 0.75,
        "length_fit": 0.55,
        "non_repetition": 0.70,
        "uncertainty_calibration": 0.60,
        "truth_grounding": 1.35,
        "stance_consistency": 0.90,
        "semantic_frame_fit": 1.25,
        "social_coherence": 1.05,
        "prompt_echo_penalty": 1.50,
        "reference_copy_penalty": 1.20,
        "retrieval_dump_penalty": 0.80,
        "repetition_penalty": 1.10,
        "broken_surface_penalty": 2.00,
    }
)

DEFAULT_SOURCE_BIASES: Mapping[str, float] = MappingProxyType(
    {
        "discord_identity": 0.45,
        "autonomous_semantic_plan": 0.48,
        "contextual_learning": 0.35,
        "short_act": 0.30,
        "exact_archive": 0.25,
        "lexical_reference": 0.25,
        "knowledge_retrieval": 0.20,
        "meaning_plan": 0.22,
        "dialogue_context": 0.24,
        "personal_action": 0.30,
        "semantic_reasoning": 0.23,
        "semantic_recovery": 0.04,
        "semantic_inquiry": 0.31,
        "abstract_reasoning": 0.32,
        "relation_reasoning": 0.28,
        "runtime_state": 0.34,
        "causal_reasoning": 0.30,
        "dialogue_act": 0.26,
        "semantic_synthesis": 0.08,
        "dialogue_rule": 0.08,
        "cache": 0.02,
        "graph": 0.0,
        "recovery": -0.15,
        "fallback": -0.40,
    }
)

_PENALTY_NAMES = frozenset(
    {
        "prompt_echo_penalty",
        "reference_copy_penalty",
        "retrieval_dump_penalty",
        "repetition_penalty",
        "broken_surface_penalty",
    }
)


@dataclass(frozen=True, slots=True)
class ArbitrationProfile:
    profile_id: str
    guild_id: int
    weights: Mapping[str, float]
    source_biases: Mapping[str, float]
    observations: int = 0
    confidence: float = 0.0
    version: int = 1

    @classmethod
    def baseline(cls, guild_id: int) -> ArbitrationProfile:
        return cls(
            profile_id=f"arbitration-profile:guild:{guild_id}",
            guild_id=guild_id,
            weights=DEFAULT_WEIGHTS,
            source_biases=DEFAULT_SOURCE_BIASES,
        )


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    candidate: SurfaceCandidate
    features: CandidateFeatures
    weighted_components: Mapping[str, float]
    total: float
    eligible: bool
    rejection_reasons: tuple[str, ...]
    admission_score: float = 0.0
    answer_status: str = "candidate"


@dataclass(frozen=True, slots=True)
class ArbitrationDecision:
    decision_id: str
    event_id: str
    winner: SurfaceCandidate | None
    scores: tuple[ScoredCandidate, ...]
    profile_id: str
    weight_version: int
    minimum_score: float
    used_fallback: bool
    selection_reason: str
    content_scores: tuple[ScoredContentProposal, ...] = ()
    selected_proposal_ids: tuple[str, ...] = ()
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class ScoredContentProposal:
    proposal: ContentProposal
    total: float
    eligible: bool
    rejection_reasons: tuple[str, ...]
    components: Mapping[str, float]


def merge_recovery_decision(
    primary: ArbitrationDecision,
    recovery: ArbitrationDecision,
    *,
    selection_reason: str = "semantic_recovery_after_below_threshold",
) -> ArbitrationDecision:
    """Attach rejected primary surfaces without orphaning their content proposals."""

    scores = {item.candidate.candidate_id: item for item in (*recovery.scores, *primary.scores)}
    content_scores = {
        item.proposal.proposal_id: item
        for item in (*recovery.content_scores, *primary.content_scores)
    }
    return replace(
        recovery,
        scores=tuple(scores.values()),
        content_scores=tuple(content_scores.values()),
        selected_proposal_ids=tuple(
            dict.fromkeys(
                (
                    *recovery.selected_proposal_ids,
                    *primary.selected_proposal_ids,
                )
            )
        ),
        used_fallback=True,
        selection_reason=selection_reason,
    )


class ResponseArbitrator:
    def __init__(self, *, minimum_score: float = 0.45) -> None:
        self.minimum_score = minimum_score

    def decide(
        self,
        *,
        event_id: str,
        prompt: str,
        focus_concepts: tuple[str, ...],
        candidates: tuple[ResponseCandidate, ...],
        recent_replies: tuple[str, ...],
        profile: ArbitrationProfile,
        structural_weights: Mapping[str, float] | None = None,
        used_fallback: bool = False,
        clarifiable_terms: tuple[str, ...] | None = None,
    ) -> ArbitrationDecision:
        unique = self._deduplicate(candidates)
        proposals = {candidate.proposal.proposal_id: candidate.proposal for candidate in unique}
        content_scores = tuple(
            sorted(
                (self._score_content(proposal, profile=profile) for proposal in proposals.values()),
                key=lambda item: (
                    item.total,
                    _stable_number(f"{event_id}|{item.proposal.proposal_id}"),
                ),
                reverse=True,
            )
        )
        selected_proposal_ids = tuple(
            item.proposal.proposal_id for item in content_scores if item.eligible
        )
        scores = tuple(
            self._score(
                candidate,
                clarifiable_terms=clarifiable_terms,
                prompt=prompt,
                focus_concepts=focus_concepts,
                recent_replies=recent_replies,
                profile=profile,
                structural_weights=structural_weights or {},
            )
            # Confidence-only pruning used to hide potentially useful answers
            # before their relevance or actual surface had even been assessed.
            for candidate in unique
        )
        eligible = tuple(item for item in scores if item.eligible)
        admitted = tuple(item for item in eligible if item.admission_score >= self.minimum_score)
        winner_score = max(
            admitted,
            key=lambda item: (
                # Preserve competition between different valid answer methods;
                # only a no-answer recovery belongs to a lower priority pool.
                int(item.answer_status != "recovery"),
                item.total,
                _stable_number(f"{event_id}|{item.candidate.candidate_id}"),
            ),
            default=None,
        )
        winner = (
            winner_score.candidate
            if winner_score is not None
            else None
        )
        reason = (
            "grounded_answer_then_surface"
            if winner_score is not None and winner_score.answer_status == "grounded_answer"
            else "highest_score"
            if winner is not None
            else "below_minimum_score"
            if eligible
            else "no_eligible_candidates"
        )
        return ArbitrationDecision(
            decision_id=f"arbitration:{uuid4()}",
            event_id=event_id,
            winner=winner,
            scores=tuple(sorted(scores, key=lambda item: item.total, reverse=True)),
            profile_id=profile.profile_id,
            weight_version=profile.version,
            minimum_score=self.minimum_score,
            used_fallback=used_fallback,
            selection_reason=reason,
            content_scores=content_scores,
            selected_proposal_ids=selected_proposal_ids,
        )

    @staticmethod
    def _score_content(
        proposal: ContentProposal,
        *,
        profile: ArbitrationProfile,
    ) -> ScoredContentProposal:
        reasons: list[str] = []
        if proposal.applicability is not None and proposal.applicability <= 0.0:
            reasons.append("not_applicable")
        plan = proposal.plan
        response_goal = plan.response_goal or plan.response_act
        if not response_goal.strip():
            reasons.append("missing_response_goal")
        components = {
            "source_confidence": proposal.source_confidence * 1.15,
            "semantic_specificity": min(
                1.0,
                (len(plan.required_facts) * 2 + len(plan.key_points)) / 4.0,
            )
            * 0.85,
            "goal_specificity": float(response_goal not in {"", "statement", "question"}) * 0.55,
            "source_bias": float(
                profile.source_biases.get(
                    proposal.source,
                    DEFAULT_SOURCE_BIASES.get(proposal.source, 0.0),
                )
            ),
        }
        denominator = 3.55
        if proposal.applicability is not None:
            components["applicability"] = proposal.applicability * 1.40
            denominator += 1.40
        total = sum(components.values()) / denominator
        if reasons:
            total = min(total, -10.0)
        return ScoredContentProposal(
            proposal=proposal,
            total=total,
            eligible=not reasons,
            rejection_reasons=tuple(reasons),
            components=MappingProxyType(components),
        )

    def _score(
        self,
        candidate: ResponseCandidate,
        *,
        prompt: str,
        focus_concepts: tuple[str, ...],
        recent_replies: tuple[str, ...],
        profile: ArbitrationProfile,
        structural_weights: Mapping[str, float],
        clarifiable_terms: tuple[str, ...] | None = None,
    ) -> ScoredCandidate:
        reasons = self._hard_rejections(candidate)
        if clarifiable_terms is not None and (
            issue := clarification_rejection(candidate.text, clarifiable_terms)
        ):
            reasons = (*reasons, issue)
        if irrelevant_lexical_question(prompt, candidate.text):
            reasons = (*reasons, "definition_not_required_for_event")
        if unmet := requirement_rejection(prompt, candidate.metadata):
            reasons = (*reasons, unmet)
        features = extract_features(
            candidate,
            prompt=prompt,
            focus_concepts=focus_concepts,
            recent_replies=recent_replies,
        )
        components: dict[str, float] = {}
        measured_features = {
            name: value for name, value in features.as_dict().items() if value is not None
        }
        for name, value in measured_features.items():
            assert value is not None
            weight = float(profile.weights.get(name, DEFAULT_WEIGHTS.get(name, 0.0)))
            components[name] = -weight * value if name in _PENALTY_NAMES else weight * value
        source_bias = float(
            profile.source_biases.get(
                candidate.source,
                DEFAULT_SOURCE_BIASES.get(candidate.source, 0.0),
            )
        )
        components["source_bias"] = source_bias
        maximum = (
            sum(
                float(profile.weights.get(name, DEFAULT_WEIGHTS.get(name, 0.0)))
                for name in measured_features
                if name not in _PENALTY_NAMES
            )
            + 1.0
        )
        structural_features = structural_response_features(
            prompt=prompt,
            candidate_text=candidate.text,
            source=candidate.source,
            speech_act=candidate.speech_act,
            metadata=candidate.metadata,
        )
        structural_bonus = learned_structural_bonus(
            structural_features,
            structural_weights,
        )
        components["learned_structure"] = structural_bonus * maximum
        total = sum(components.values()) / max(1.0, maximum)
        answer_status = _answer_status(candidate, features, prompt, focus_concepts)
        # Feedback about wording ranks admitted answers, but cannot suppress an
        # otherwise grounded answer or turn lack of an answer into evidence.
        admission_score = total
        if answer_status == "grounded_answer":
            stable_weights = {
                name: DEFAULT_WEIGHTS.get(name, 0.0) for name in measured_features
            }
            admission_score = sum(
                (-1 if name in _PENALTY_NAMES else 1) * stable_weights[name] * value
                for name, value in measured_features.items()
            ) / max(1.0, sum(
                weight for name, weight in stable_weights.items() if name not in _PENALTY_NAMES
            ))
        if reasons:
            total = min(total, -10.0)
            admission_score = min(admission_score, -10.0)
        return ScoredCandidate(
            candidate=candidate,
            features=features,
            weighted_components=MappingProxyType(components),
            total=total,
            eligible=not reasons,
            rejection_reasons=reasons,
            admission_score=admission_score,
            answer_status=answer_status,
        )

    @staticmethod
    def _hard_rejections(candidate: ResponseCandidate) -> tuple[str, ...]:
        reasons: list[str] = []
        if candidate.proposal.applicability is not None and candidate.proposal.applicability <= 0.0:
            reasons.append("not_applicable")
        text = candidate.text.strip()
        normalized = text.casefold()
        if not text:
            reasons.append("empty")
        if _TECHNICAL_RE.search(text):
            reasons.append("technical_token")
        if internal_surface_reason(normalized) is not None:
            reasons.append("internal_diagnostic_leak")
        surface = evaluate_surface(text)
        researched_fact_surface = bool(
            candidate.metadata.get("foreground_research") is True
            and candidate.metadata.get("allow_rough_surface") is not True
            and candidate.required_facts
            and all(_covers_fact(text, fact) for fact in candidate.required_facts)
        )
        reasons.extend(
            f"surface:{issue}"
            for issue in surface.hard_issues
            if not (
                (researched_fact_surface and issue == "unexpected_valency_role")
                or (candidate.metadata.get("allow_rough_surface") is True
                    and candidate.metadata.get("locally_composed") is True
                    and issue in COMPOSED_SURFACE_SOFT_ISSUES)
            )
        )
        explicit_rejection = candidate.metadata.get("hard_rejection")
        if isinstance(explicit_rejection, str) and explicit_rejection:
            reasons.append(explicit_rejection)
        for fact in candidate.required_facts:
            if not _covers_fact(text, fact):
                reasons.append(f"missing_required_fact:{_fact_key(fact)}")
        return tuple(dict.fromkeys(reasons))

    @staticmethod
    def _deduplicate(
        candidates: tuple[ResponseCandidate, ...],
    ) -> tuple[ResponseCandidate, ...]:
        by_text: dict[str, ResponseCandidate] = {}
        for candidate in candidates:
            key = " ".join(candidate.text.casefold().split())
            current = by_text.get(key)
            candidate_rejected = bool(ResponseArbitrator._hard_rejections(candidate))
            current_rejected = (
                bool(ResponseArbitrator._hard_rejections(current)) if current is not None else True
            )
            if (
                current is None
                or (current_rejected and not candidate_rejected)
                or (
                    current_rejected == candidate_rejected
                    and candidate.source_confidence > current.source_confidence
                )
            ):
                by_text[key] = candidate
        return tuple(by_text.values())


def extract_features(
    candidate: ResponseCandidate,
    *,
    prompt: str,
    focus_concepts: tuple[str, ...],
    recent_replies: tuple[str, ...],
) -> CandidateFeatures:
    text = candidate.text
    prompt_terms = set(content_lemmas(prompt))
    candidate_terms = set(content_lemmas(text))
    focus_terms = set(focus_concepts)
    active_terms = prompt_terms | focus_terms
    measured_relevance = candidate.metadata.get("context_relevance")
    measured_applicability = candidate.metadata.get("applicability")
    applicability = (
        float(measured_applicability)
        if isinstance(measured_applicability, int | float)
        and not isinstance(measured_applicability, bool)
        else None
    )
    measured_binding = candidate.metadata.get("reply_binding")
    reply_binding = (
        float(measured_binding)
        if isinstance(measured_binding, int | float) and not isinstance(measured_binding, bool)
        else None
    )
    context_relevance = (
        float(measured_relevance)
        if isinstance(measured_relevance, int | float) and not isinstance(measured_relevance, bool)
        else _overlap(candidate_terms, active_terms)
    )
    topic_continuity = _overlap(candidate_terms, prompt_terms or focus_terms)
    covered = sum(_covers_fact(text, fact) for fact in candidate.required_facts)
    covered += sum(_covers_key_point(text, point) for point in candidate.key_points)
    required_count = len(candidate.required_facts) + len(candidate.key_points)
    plan_coverage = covered / required_count if required_count else context_relevance
    required_terms = {
        term for fact in candidate.required_facts for term in content_lemmas(fact.text)
    }
    fact_grounding = _overlap(candidate_terms, required_terms) if required_terms else None
    tokens = tokenize(text)
    words = tuple(token for token in tokens if token[:1].isalnum())
    surface = evaluate_surface(text)
    grammar_fit = surface.score
    specificity = min(1.0, len(candidate_terms) / 6.0)
    target_length = 16
    length_fit = 1.0 - min(1.0, abs(len(words) - target_length) / target_length)
    recent_similarity = max(
        (
            SequenceMatcher(None, text.casefold(), item.casefold()).ratio()
            for item in recent_replies
        ),
        default=0.0,
    )
    non_repetition = 1.0 - recent_similarity
    speech_act_fit = _speech_act_fit(candidate.speech_act, text)
    uncertainty_words = set(word.casefold() for word in words) & _UNCERTAINTY_MARKERS
    uncertainty_calibration = (
        1.0
        if candidate.source_confidence >= 0.70 and not uncertainty_words
        else 1.0
        if candidate.source_confidence < 0.70 and uncertainty_words
        else 0.55
    )
    truth_grounding = _metadata_score(candidate, "truth_grounding", fact_grounding)
    stance_consistency = _metadata_score(candidate, "stance_consistency", None)
    semantic_frame_fit = _metadata_score(candidate, "semantic_frame_fit", speech_act_fit)
    social_coherence = _metadata_score(candidate, "social_coherence", None)
    prompt_similarity = SequenceMatcher(None, " ".join(tokenize(prompt)), " ".join(tokens)).ratio()
    prompt_echo_penalty = max(0.0, (prompt_similarity - 0.62) / 0.38)
    reference_copy_penalty = _metadata_score(candidate, "reference_copy_penalty", 0.0)
    retrieval_dump_penalty = (
        min(1.0, max(0, len(words) - 48) / 64.0)
        if candidate.source in {"exact_archive", "lexical_reference", "knowledge_retrieval"}
        else 0.0
    )
    repetition_penalty = 1.0 - len(set(words)) / max(1, len(words))
    soft_surface = bool(
        candidate.metadata.get("allow_rough_surface") is True
        and candidate.metadata.get("locally_composed") is True
        and surface.hard_issues
        and not (set(surface.hard_issues) - COMPOSED_SURFACE_SOFT_ISSUES)
    )
    broken_surface_penalty = 0.15 if soft_surface else float(bool(surface.hard_issues))
    return CandidateFeatures(
        applicability=_clamp(applicability) if applicability is not None else None,
        context_relevance=_clamp(context_relevance),
        source_confidence=candidate.source_confidence,
        reply_binding=_clamp(reply_binding) if reply_binding is not None else None,
        speech_act_fit=_clamp(speech_act_fit),
        plan_coverage=_clamp(plan_coverage),
        fact_grounding=_clamp(fact_grounding) if fact_grounding is not None else None,
        topic_continuity=_clamp(topic_continuity),
        grammar_fit=_clamp(grammar_fit),
        specificity=_clamp(specificity),
        length_fit=_clamp(length_fit),
        non_repetition=_clamp(non_repetition),
        uncertainty_calibration=_clamp(uncertainty_calibration),
        truth_grounding=_clamp(truth_grounding) if truth_grounding is not None else None,
        stance_consistency=(_clamp(stance_consistency) if stance_consistency is not None else None),
        semantic_frame_fit=_clamp(semantic_frame_fit),
        social_coherence=_clamp(social_coherence) if social_coherence is not None else None,
        prompt_echo_penalty=_clamp(prompt_echo_penalty),
        reference_copy_penalty=_clamp(reference_copy_penalty),
        retrieval_dump_penalty=_clamp(retrieval_dump_penalty),
        repetition_penalty=_clamp(repetition_penalty),
        broken_surface_penalty=_clamp(broken_surface_penalty),
    )


def _answer_status(
    candidate: ResponseCandidate, features: CandidateFeatures,
    prompt: str, focus_concepts: tuple[str, ...],
) -> str:
    metadata = candidate.metadata
    if (candidate.source in {"fallback", "semantic_recovery"}
            or metadata.get("semantic_recovery") is True):
        return "recovery"
    composed = bool(metadata.get("locally_composed") or (
        metadata.get("authored_reply_template") is False
        and (metadata.get("role_order") or metadata.get("proposition_count"))
    ))
    evidence = bool(candidate.required_facts or any(metadata.get(key) for key in (
        "reasoning_evidence", "position_evidence", "used_evidence", "causal_path",
    )))
    # Binding is checked against actual words too; a producer's numerical
    # confidence alone cannot make "I analyze context" answer any topic.
    topics = set(content_lemmas(prompt)) | set(focus_concepts)
    bound = bool(topics & set(content_lemmas(candidate.text)))
    if not bound:
        bound = bool(metadata.get("answer_slot") and features.semantic_frame_fit >= 0.8
                     and metadata.get("obligation_fit", 0.0) >= 0.8)
    if (composed and evidence and bound
            and not metadata.get("response_signal_mismatch")
            and metadata.get("semantic_scaffold") is not True):
        return "grounded_answer"
    return "candidate"


def _speech_act_fit(speech_act: str, text: str) -> float:
    is_question = text.rstrip().endswith("?")
    if speech_act in {"clarifying_question", "question", "invite_continuation"}:
        return 1.0 if is_question else 0.25
    return 0.85 if not is_question else 0.55


def _metadata_score(
    candidate: ResponseCandidate,
    name: str,
    default: float | None,
) -> float | None:
    value = candidate.metadata.get(name)
    return (
        float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default
    )


def _covers_fact(text: str, fact: StructuredFact) -> bool:
    return covers_structured_fact(text, fact)


def _covers_key_point(text: str, point: str) -> bool:
    point_terms = set(content_lemmas(point))
    if not point_terms:
        return False
    text_terms = set(content_lemmas(text))
    return len(text_terms & point_terms) / len(point_terms) >= 0.72


def _fact_key(fact: StructuredFact) -> str:
    return hashlib.sha256(fact.text.encode("utf-8")).hexdigest()[:10]


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _stable_number(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
