"""Cross-source response candidates, common scoring and deterministic arbitration."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from types import MappingProxyType
from uuid import uuid4

from debilgi.language.content import content_lemmas
from debilgi.language.structural_learning import (
    learned_structural_bonus,
    structural_response_features,
)
from debilgi.language.surface_coherence import evaluate_surface
from debilgi.language.surface_policy import internal_surface_reason
from debilgi.language.tokenizer import tokenize

_TECHNICAL_RE = re.compile(r"<(?:user|role|channel|url|emoji:[^>]+)>", re.IGNORECASE)
_UNCERTAINTY_MARKERS = frozenset(
    {"похоже", "возможно", "вероятно", "кажется", "не", "уверен", "точно"}
)


@dataclass(frozen=True, slots=True)
class ResponseCandidate:
    candidate_id: str
    source: str
    renderer: str
    text: str
    speech_act: str
    source_confidence: float
    required_facts: tuple[str, ...] = ()
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
        required_facts: tuple[str, ...] = (),
        key_points: tuple[str, ...] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> ResponseCandidate:
        material = "\x1f".join((event_id, source, renderer, " ".join(text.split())))
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        return cls(
            candidate_id=f"candidate:{digest}",
            source=source,
            renderer=renderer,
            text=text.strip(),
            speech_act=speech_act,
            source_confidence=_clamp(source_confidence),
            required_facts=required_facts,
            key_points=key_points,
            metadata=MappingProxyType(dict(metadata or {})),
        )


@dataclass(frozen=True, slots=True)
class CandidateFeatures:
    applicability: float
    context_relevance: float
    source_confidence: float
    reply_binding: float
    speech_act_fit: float
    plan_coverage: float
    fact_grounding: float
    topic_continuity: float
    grammar_fit: float
    specificity: float
    length_fit: float
    non_repetition: float
    uncertainty_calibration: float
    truth_grounding: float
    stance_consistency: float
    semantic_frame_fit: float
    social_coherence: float
    prompt_echo_penalty: float
    reference_copy_penalty: float
    retrieval_dump_penalty: float
    repetition_penalty: float
    broken_surface_penalty: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


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
        "contextual_learning": 0.35,
        "short_act": 0.30,
        "exact_archive": 0.25,
        "lexical_reference": 0.25,
        "knowledge_retrieval": 0.20,
        "meaning_plan": 0.22,
        "dialogue_context": 0.24,
        "semantic_reasoning": 0.23,
        "relation_reasoning": 0.28,
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
    candidate: ResponseCandidate
    features: CandidateFeatures
    weighted_components: Mapping[str, float]
    total: float
    eligible: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArbitrationDecision:
    decision_id: str
    event_id: str
    winner: ResponseCandidate | None
    scores: tuple[ScoredCandidate, ...]
    profile_id: str
    weight_version: int
    minimum_score: float
    used_fallback: bool
    selection_reason: str
    latency_ms: float = 0.0


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
    ) -> ArbitrationDecision:
        unique = self._deduplicate(candidates)
        scores = tuple(
            self._score(
                candidate,
                prompt=prompt,
                focus_concepts=focus_concepts,
                recent_replies=recent_replies,
                profile=profile,
                structural_weights=structural_weights or {},
            )
            for candidate in unique
        )
        eligible = tuple(item for item in scores if item.eligible)
        winner_score = max(
            eligible,
            key=lambda item: (
                item.total,
                _stable_number(f"{event_id}|{item.candidate.candidate_id}"),
            ),
            default=None,
        )
        winner = (
            winner_score.candidate
            if winner_score is not None and winner_score.total >= self.minimum_score
            else None
        )
        reason = (
            "highest_score"
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
    ) -> ScoredCandidate:
        reasons = self._hard_rejections(candidate)
        features = extract_features(
            candidate,
            prompt=prompt,
            focus_concepts=focus_concepts,
            recent_replies=recent_replies,
        )
        components: dict[str, float] = {}
        for name, value in features.as_dict().items():
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
                for name in features.as_dict()
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
        if reasons:
            total = min(total, -10.0)
        return ScoredCandidate(
            candidate=candidate,
            features=features,
            weighted_components=MappingProxyType(components),
            total=total,
            eligible=not reasons,
            rejection_reasons=reasons,
        )

    @staticmethod
    def _hard_rejections(candidate: ResponseCandidate) -> tuple[str, ...]:
        reasons: list[str] = []
        text = candidate.text.strip()
        normalized = text.casefold()
        if not text:
            reasons.append("empty")
        if _TECHNICAL_RE.search(text):
            reasons.append("technical_token")
        if internal_surface_reason(normalized) is not None:
            reasons.append("internal_diagnostic_leak")
        surface = evaluate_surface(text)
        reasons.extend(f"surface:{issue}" for issue in surface.hard_issues)
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
        else 1.0
    )
    measured_binding = candidate.metadata.get("reply_binding")
    reply_binding = (
        float(measured_binding)
        if isinstance(measured_binding, int | float) and not isinstance(measured_binding, bool)
        else 0.5
    )
    context_relevance = (
        float(measured_relevance)
        if isinstance(measured_relevance, int | float) and not isinstance(measured_relevance, bool)
        else _overlap(candidate_terms, active_terms)
    )
    topic_continuity = _overlap(candidate_terms, prompt_terms or focus_terms)
    required = tuple((*candidate.required_facts, *candidate.key_points))
    covered = sum(_covers_fact(text, fact) for fact in required)
    plan_coverage = covered / max(1, len(required)) if required else context_relevance
    required_terms = {term for fact in candidate.required_facts for term in content_lemmas(fact)}
    fact_grounding = _overlap(candidate_terms, required_terms) if required_terms else 0.5
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
    stance_consistency = _metadata_score(candidate, "stance_consistency", 0.65)
    semantic_frame_fit = _metadata_score(candidate, "semantic_frame_fit", speech_act_fit)
    social_coherence = _metadata_score(candidate, "social_coherence", 0.60)
    prompt_similarity = SequenceMatcher(None, " ".join(tokenize(prompt)), " ".join(tokens)).ratio()
    prompt_echo_penalty = max(0.0, (prompt_similarity - 0.62) / 0.38)
    reference_copy_penalty = _metadata_score(candidate, "reference_copy_penalty", 0.0)
    retrieval_dump_penalty = (
        min(1.0, max(0, len(words) - 48) / 64.0)
        if candidate.source in {"exact_archive", "lexical_reference", "knowledge_retrieval"}
        else 0.0
    )
    repetition_penalty = 1.0 - len(set(words)) / max(1, len(words))
    broken_surface_penalty = float(bool(surface.hard_issues))
    return CandidateFeatures(
        applicability=_clamp(applicability),
        context_relevance=_clamp(context_relevance),
        source_confidence=candidate.source_confidence,
        reply_binding=_clamp(reply_binding),
        speech_act_fit=_clamp(speech_act_fit),
        plan_coverage=_clamp(plan_coverage),
        fact_grounding=_clamp(fact_grounding),
        topic_continuity=_clamp(topic_continuity),
        grammar_fit=_clamp(grammar_fit),
        specificity=_clamp(specificity),
        length_fit=_clamp(length_fit),
        non_repetition=_clamp(non_repetition),
        uncertainty_calibration=_clamp(uncertainty_calibration),
        truth_grounding=_clamp(truth_grounding),
        stance_consistency=_clamp(stance_consistency),
        semantic_frame_fit=_clamp(semantic_frame_fit),
        social_coherence=_clamp(social_coherence),
        prompt_echo_penalty=_clamp(prompt_echo_penalty),
        reference_copy_penalty=_clamp(reference_copy_penalty),
        retrieval_dump_penalty=_clamp(retrieval_dump_penalty),
        repetition_penalty=_clamp(repetition_penalty),
        broken_surface_penalty=_clamp(broken_surface_penalty),
    )


def _speech_act_fit(speech_act: str, text: str) -> float:
    is_question = text.rstrip().endswith("?")
    if speech_act in {"clarifying_question", "question", "invite_continuation"}:
        return 1.0 if is_question else 0.25
    return 0.85 if not is_question else 0.55


def _metadata_score(candidate: ResponseCandidate, name: str, default: float) -> float:
    value = candidate.metadata.get(name)
    return (
        float(value)
        if isinstance(value, int | float) and not isinstance(value, bool)
        else default
    )


def _covers_fact(text: str, fact: str) -> bool:
    normalized_text = " ".join(text.casefold().replace("ё", "е").split())
    normalized_fact = " ".join(fact.casefold().replace("ё", "е").split())
    if normalized_fact and normalized_fact in normalized_text:
        return True
    fact_terms = set(content_lemmas(fact))
    if not fact_terms:
        return False
    text_terms = set(content_lemmas(text))
    # Required-fact coverage is directional: other grounded sentences in the
    # same answer must not make an otherwise covered fact look less complete.
    return len(text_terms & fact_terms) / len(fact_terms) >= 0.72


def _fact_key(fact: str) -> str:
    return hashlib.sha256(fact.encode("utf-8")).hexdigest()[:10]


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _stable_number(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
