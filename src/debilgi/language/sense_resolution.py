"""Unified contextual arbitration for lexical, local and entity meanings."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from debilgi.language.content import content_lemmas
from debilgi.language.structured_utterance import StructuredUtterance, TokenInterpretation

_MEDIA_MARKERS = frozenset(
    {
        "альбом",
        "аниме",
        "игра",
        "книга",
        "комикс",
        "манга",
        "песня",
        "произведение",
        "сериал",
        "телесериал",
        "фильм",
    }
)
_MEDIA_CONTEXT = frozenset(
    {
        *_MEDIA_MARKERS,
        "актёр",
        "актер",
        "выйти",
        "жанр",
        "режиссёр",
        "режиссер",
        "сезон",
        "серия",
        "смотреть",
        "эпизод",
    }
)
_PERSON_MARKERS = frozenset({"человек", "лицо", "собеседник", "мужчина", "женщина", "тот"})
_PERSON_ROLES = frozenset({"subject", "vocative", "predicate_complement"})
_OBJECT_ROLES = frozenset(
    {
        "direct_object",
        "indirect_object",
        "predicate_complement",
        "unassigned_content",
    }
)


@dataclass(frozen=True, slots=True)
class SenseCandidate:
    candidate_id: str
    token_index: int
    lemma: str
    meaning: str
    source: str
    scope: str
    confidence: float
    context_terms: tuple[str, ...]
    style_labels: tuple[str, ...] = ()
    entity_kind: str | None = None


@dataclass(frozen=True, slots=True)
class ScoredSense:
    candidate: SenseCandidate
    total: float
    components: tuple[tuple[str, float], ...]
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SenseDecision:
    token_index: int
    surface: str
    chosen: SenseCandidate | None
    scores: tuple[ScoredSense, ...]
    confidence: float
    margin: float
    reason: str


@dataclass(frozen=True, slots=True)
class UtteranceSenseResolution:
    decisions: tuple[SenseDecision, ...]

    @property
    def unresolved_indices(self) -> tuple[int, ...]:
        return tuple(item.token_index for item in self.decisions if item.chosen is None)


def resolve_contextual_senses(
    utterance: StructuredUtterance,
) -> UtteranceSenseResolution:
    """Score every available sense instead of accepting the first source branch."""

    active_terms = set(content_lemmas(utterance.text))
    media_context = bool(active_terms & _MEDIA_CONTEXT)
    decisions: list[SenseDecision] = []
    for token in utterance.tokens:
        candidates = _candidates_for(token)
        if not candidates:
            continue
        scored = tuple(
            sorted(
                (
                    _score_candidate(
                        candidate,
                        token=token,
                        active_terms=active_terms,
                        media_context=media_context,
                    )
                    for candidate in candidates
                ),
                key=lambda item: (item.total, item.candidate.candidate_id),
                reverse=True,
            )
        )
        eligible = tuple(item for item in scored if not item.rejection_reasons)
        top = eligible[0] if eligible else None
        runner_up = eligible[1] if len(eligible) > 1 else None
        margin = top.total - runner_up.total if top and runner_up else top.total if top else 0.0
        privileged = bool(top and top.candidate.scope in {"entity", "server", "channel"})
        chosen = (
            top.candidate
            if top is not None
            and top.total >= 0.30
            and (len(eligible) == 1 or margin >= 0.06 or privileged)
            else None
        )
        reason = (
            "highest_contextual_score"
            if chosen is not None
            else "no_applicable_sense"
            if not eligible
            else "below_minimum_score"
            if top is not None and top.total < 0.30
            else "ambiguous_margin"
        )
        decision_confidence = (
            min(1.0, max(0.0, top.total) * 0.75 + max(0.0, margin) * 0.25)
            if top is not None
            else 0.0
        )
        decisions.append(
            SenseDecision(
                token.index,
                token.surface,
                chosen,
                scored,
                decision_confidence,
                margin,
                reason,
            )
        )
    return UtteranceSenseResolution(tuple(decisions))


def apply_sense_resolution(
    utterance: StructuredUtterance,
    resolution: UtteranceSenseResolution,
) -> StructuredUtterance:
    """Return a structured copy carrying the resolver winner for future planners."""

    chosen_by_index = {
        item.token_index: item.chosen for item in resolution.decisions if item.chosen is not None
    }
    tokens = tuple(
        replace(
            token,
            selected_meaning=chosen.meaning,
            lexical_source=chosen.source,
            meaning_confidence=next(
                (
                    decision.confidence
                    for decision in resolution.decisions
                    if decision.token_index == token.index
                ),
                token.meaning_confidence,
            ),
        )
        if (chosen := chosen_by_index.get(token.index)) is not None
        else token
        for token in utterance.tokens
    )
    return replace(utterance, tokens=tokens)


def _candidates_for(token: TokenInterpretation) -> tuple[SenseCandidate, ...]:
    raw: list[SenseCandidate] = []
    if token.entity_kind is not None:
        raw.append(
            _candidate(
                token,
                meaning=f"entity:{token.entity_kind}:{token.lemma}",
                source="entity",
                scope="entity",
                confidence=max(token.confidence, 0.72),
                entity_kind=token.entity_kind,
            )
        )
    if token.local_meaning:
        raw.append(
            _candidate(
                token,
                meaning=token.local_meaning,
                source=token.lexical_source or "server_slang",
                scope="channel" if "channel" in token.style_labels else "server",
                confidence=max(token.meaning_confidence, 0.68),
            )
        )
    if token.personal_meaning and token.personal_meaning != token.local_meaning:
        raw.append(
            _candidate(
                token,
                meaning=token.personal_meaning,
                source="personal_lexicon",
                scope="user",
                confidence=max(token.meaning_confidence * 0.92, 0.48),
            )
        )
    if token.selected_meaning and token.selected_meaning not in {
        token.local_meaning,
        token.personal_meaning,
    }:
        raw.append(
            _candidate(
                token,
                meaning=token.selected_meaning,
                source=token.lexical_source or "selected_reference",
                scope="dictionary",
                confidence=max(token.meaning_confidence, token.context_fit, 0.52),
            )
        )
    for meaning in token.meaning_candidates:
        if meaning and meaning not in {
            token.local_meaning,
            token.personal_meaning,
            token.selected_meaning,
        }:
            raw.append(
                _candidate(
                    token,
                    meaning=meaning,
                    source=f"dictionary:{token.lexical_source or 'reference'}",
                    scope="dictionary",
                    confidence=max(token.meaning_confidence, 0.45),
                )
            )
    by_key: dict[tuple[str, str], SenseCandidate] = {}
    for candidate in raw:
        key = (" ".join(candidate.meaning.casefold().split()), candidate.scope)
        current = by_key.get(key)
        if current is None or candidate.confidence > current.confidence:
            by_key[key] = candidate
    return tuple(by_key.values())


def _candidate(
    token: TokenInterpretation,
    *,
    meaning: str,
    source: str,
    scope: str,
    confidence: float,
    entity_kind: str | None = None,
) -> SenseCandidate:
    material = "\x1f".join((str(token.index), token.lemma, meaning, source, scope))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return SenseCandidate(
        candidate_id=f"sense:{digest}",
        token_index=token.index,
        lemma=token.lemma,
        meaning=meaning,
        source=source,
        scope=scope,
        confidence=max(0.0, min(1.0, confidence)),
        context_terms=content_lemmas(meaning),
        style_labels=token.style_labels,
        entity_kind=entity_kind,
    )


def _score_candidate(
    candidate: SenseCandidate,
    *,
    token: TokenInterpretation,
    active_terms: set[str],
    media_context: bool,
) -> ScoredSense:
    definition_terms = set(candidate.context_terms) - {token.lemma}
    overlap_count = len(definition_terms & active_terms)
    components: dict[str, float] = {
        "source_confidence": candidate.confidence * 0.38,
        "scope_fit": {
            "entity": 0.32,
            "channel": 0.27,
            "server": 0.24,
            "user": 0.15,
            "dictionary": 0.05,
        }.get(candidate.scope, 0.0),
        "context_overlap": min(0.34, overlap_count * 0.12),
        "role_fit": _role_fit(candidate, token),
        "media_context": 0.0,
        "surface_case": 0.0,
    }
    media_sense = bool(definition_terms & _MEDIA_MARKERS) or bool(
        set(candidate.style_labels) & _MEDIA_MARKERS
    )
    if media_sense:
        components["media_context"] = 0.30 if media_context else -0.32
        if token.surface[:1].isupper():
            components["surface_case"] = 0.04 if token.index == 0 else 0.10
    elif token.semantic_role in _PERSON_ROLES and definition_terms & _PERSON_MARKERS:
        components["role_fit"] += 0.18
    if candidate.source == token.lexical_source and token.selected_meaning == candidate.meaning:
        components["preselected_context"] = min(0.12, token.context_fit * 0.12)
    total = max(0.0, min(1.0, sum(components.values())))
    rejection_reasons: list[str] = []
    if candidate.entity_kind is not None and token.entity_kind != candidate.entity_kind:
        rejection_reasons.append("entity_kind_mismatch")
    return ScoredSense(
        candidate,
        total,
        tuple(sorted(components.items())),
        tuple(rejection_reasons),
    )


def _role_fit(candidate: SenseCandidate, token: TokenInterpretation) -> float:
    if candidate.entity_kind is not None:
        return 0.24 if token.semantic_role in _PERSON_ROLES else 0.10
    if token.semantic_role == "predicate" and token.part_of_speech == "verb":
        return 0.14
    if token.semantic_role in _OBJECT_ROLES and token.part_of_speech in {
        "noun",
        "pronoun",
        "adjective",
    }:
        return 0.08
    return 0.02
