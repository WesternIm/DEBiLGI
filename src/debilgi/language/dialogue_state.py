"""Topic continuity and cross-turn reference resolution.

This module enriches a retrieval query with context that can be justified from
recent turns.  It never rewrites the user's original utterance or invents a
reply fact.
"""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.language.content import content_lemmas
from debilgi.language.lexical_matching import same_lexical_family
from debilgi.language.structured_utterance import (
    StructuredUtterance,
    TokenInterpretation,
    analyze_structured_utterance,
)
from debilgi.language.valency import predicate_valency

_SHIFT_MARKERS = (
    "кстати",
    "другая тема",
    "сменим тему",
    "теперь о",
    "а теперь о",
)
_RETURN_MARKERS = (
    "вернемся к",
    "вернёмся к",
    "возвращаясь к",
    "снова о",
)
_REFERENCE_PRONOUNS = frozenset(
    {
        "он",
        "она",
        "оно",
        "они",
        "его",
        "её",
        "ее",
        "их",
        "ему",
        "ей",
        "им",
        "это",
    }
)
_ASK_PREDICATES = frozenset({"спрашивать", "спросить", "задавать"})


@dataclass(frozen=True, slots=True)
class TopicState:
    transition: str
    current_concepts: tuple[str, ...]
    active_concepts: tuple[str, ...]
    previous_concepts: tuple[str, ...]
    anchor_message: str | None
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class ReferenceCandidate:
    lemma: str
    surface: str
    semantic_role: str
    source_message: str
    source_distance: int
    gender: str | None
    number: str | None
    animacy: str | None
    score: float


@dataclass(frozen=True, slots=True)
class ReferenceResolution:
    kind: str
    token_index: int | None
    surface: str
    semantic_role: str
    chosen: ReferenceCandidate | None
    alternatives: tuple[ReferenceCandidate, ...]
    confidence: float
    reason: str


@dataclass(frozen=True, slots=True)
class DialogueStateAnalysis:
    topic: TopicState
    references: tuple[ReferenceResolution, ...]
    retrieval_query: str
    added_context_terms: tuple[str, ...]
    confidence: float
    repair_target: str | None

    @property
    def unresolved_references(self) -> tuple[ReferenceResolution, ...]:
        return tuple(item for item in self.references if item.chosen is None)

    @property
    def resolved_bindings(self) -> tuple[tuple[str, str], ...]:
        """Semantic role bindings safe to pass beyond retrieval."""

        return tuple(
            dict.fromkeys(
                (item.semantic_role, item.chosen.lemma)
                for item in self.references
                if item.chosen is not None
            )
        )


def analyze_dialogue_state(
    prompt: str,
    context_messages: tuple[str, ...] = (),
) -> DialogueStateAnalysis:
    """Resolve topic movement and references against a bounded recent window."""

    recent_messages = tuple(message.strip() for message in context_messages[-8:] if message.strip())
    prompt_structure = analyze_structured_utterance(prompt)
    topic = _topic_state(prompt, recent_messages, prompt_structure)
    repair_target = _repair_target(prompt, recent_messages)
    candidates = _reference_candidates(recent_messages)
    references = _resolve_references(prompt_structure, candidates)
    added_terms = tuple(
        dict.fromkeys(
            item.chosen.lemma
            for item in references
            if item.chosen is not None and item.chosen.lemma not in set(content_lemmas(prompt))
        )
    )
    # A short elliptical continuation can have no overt pronoun.  Supplying the
    # active topic to retrieval is safe only when continuity itself is strong.
    if (
        not added_terms
        and topic.transition == "continuation"
        and topic.confidence >= 0.78
        and len(topic.current_concepts) <= 2
        and any(frame.elliptical for frame in prompt_structure.clauses)
    ):
        added_terms = tuple(
            concept
            for concept in topic.active_concepts[:2]
            if concept not in set(topic.current_concepts)
        )
    if repair_target is not None:
        repair_terms = tuple(
            term
            for term in content_lemmas(repair_target)
            if term not in set(content_lemmas(prompt))
        )
        added_terms = tuple(dict.fromkeys((*added_terms, *repair_terms[:6])))
        retrieval_query = repair_target
    else:
        retrieval_query = " ".join((prompt.strip(), *added_terms)).strip()
    reference_confidence = (
        sum(item.confidence for item in references) / len(references) if references else 0.76
    )
    return DialogueStateAnalysis(
        topic=topic,
        references=references,
        retrieval_query=retrieval_query,
        added_context_terms=added_terms,
        confidence=min(1.0, topic.confidence * 0.56 + reference_confidence * 0.44),
        repair_target=repair_target,
    )


def realize_reference_clarification(analysis: DialogueStateAnalysis) -> str | None:
    """Ask one grounded clarification only when a reference is unresolved."""

    unresolved = next(iter(analysis.unresolved_references), None)
    if unresolved is None or unresolved.kind != "explicit_pronoun":
        return None
    options = tuple(dict.fromkeys(item.lemma for item in unresolved.alternatives[:3]))
    if len(options) >= 2:
        joined = " или ".join(f"«{item}»" for item in options[:2])
        return f"Под «{unresolved.surface}» имеется в виду {joined}?"
    return f"К кому или к чему здесь относится «{unresolved.surface}»?"


def _topic_state(
    prompt: str,
    messages: tuple[str, ...],
    structure: StructuredUtterance,
) -> TopicState:
    current = tuple(dict.fromkeys(content_lemmas(prompt)))
    histories = tuple(tuple(dict.fromkeys(content_lemmas(message))) for message in messages)
    histories = tuple(item for item in histories if item)
    normalized = " ".join(prompt.casefold().replace("ё", "е").split())
    previous = histories[-1] if histories else ()
    if not histories:
        return TopicState("new", current, current, (), None, 0.90, "no_previous_turn")

    latest_overlap = _concept_overlap(current, previous)
    older_match = next(
        (
            (index, concepts)
            for index, concepts in enumerate(reversed(histories[:-1]), start=1)
            if _concept_overlap(current, concepts) >= 0.34
        ),
        None,
    )
    explicit_return = normalized.startswith(_RETURN_MARKERS)
    explicit_shift = normalized.startswith(_SHIFT_MARKERS)
    has_explicit_reference = any(
        token.part_of_speech == "pronoun" and token.lemma in _REFERENCE_PRONOUNS
        for token in structure.tokens
    )
    short_follow_up = (
        len(current) <= 2
        and bool(structure.clauses)
        and (
            has_explicit_reference
            or structure.clauses[0].question_target_index is not None
            or (structure.clauses[0].elliptical and not current)
        )
    )
    if explicit_return and older_match is not None:
        _, anchor = older_match
        return TopicState(
            "return",
            current,
            tuple(dict.fromkeys((*anchor, *current))),
            previous,
            _message_for_concepts(messages, anchor),
            0.94,
            "explicit_return_marker",
        )
    if explicit_shift:
        return TopicState(
            "shift",
            current,
            current,
            previous,
            messages[-1],
            0.96,
            "explicit_shift_marker",
        )
    if latest_overlap >= 0.28:
        return TopicState(
            "continuation",
            current,
            tuple(dict.fromkeys((*previous, *current))),
            previous,
            messages[-1],
            min(0.98, 0.70 + latest_overlap * 0.28),
            "latest_topic_overlap",
        )
    if older_match is not None and current:
        _, anchor = older_match
        return TopicState(
            "return",
            current,
            tuple(dict.fromkeys((*anchor, *current))),
            previous,
            _message_for_concepts(messages, anchor),
            0.84,
            "older_topic_overlap",
        )
    if short_follow_up:
        return TopicState(
            "continuation",
            current,
            tuple(dict.fromkeys((*previous, *current))),
            previous,
            messages[-1],
            0.80,
            "short_elliptical_follow_up",
        )
    return TopicState(
        "shift",
        current,
        current,
        previous,
        messages[-1],
        0.74 if current else 0.55,
        "new_unconnected_content" if current else "empty_content",
    )


def _reference_candidates(messages: tuple[str, ...]) -> tuple[ReferenceCandidate, ...]:
    result: list[ReferenceCandidate] = []
    seen: set[tuple[str, str, int]] = set()
    for distance, message in enumerate(reversed(messages), start=1):
        structure = analyze_structured_utterance(message)
        for token in structure.tokens:
            if token.part_of_speech != "noun" and token.entity_kind is None:
                continue
            if token.semantic_role in {"function_word", "unassigned_content", "vocative"}:
                continue
            key = (token.lemma, token.semantic_role, distance)
            if key in seen:
                continue
            seen.add(key)
            features = dict(token.grammatical_features)
            role_bonus = (
                0.18
                if token.semantic_role == "subject"
                else 0.15
                if token.semantic_role == "direct_object"
                else 0.08
            )
            score = max(0.10, 0.78 - (distance - 1) * 0.10 + role_bonus)
            result.append(
                ReferenceCandidate(
                    lemma=token.lemma,
                    surface=token.corrected_form or token.surface,
                    semantic_role=token.semantic_role,
                    source_message=message,
                    source_distance=distance,
                    gender=features.get("gender"),
                    number=features.get("number"),
                    animacy=features.get("animacy"),
                    score=min(1.0, score),
                )
            )
    return tuple(result)


def _resolve_references(
    structure: StructuredUtterance,
    candidates: tuple[ReferenceCandidate, ...],
) -> tuple[ReferenceResolution, ...]:
    resolutions: list[ReferenceResolution] = []
    for token in structure.tokens:
        if token.part_of_speech != "pronoun" or token.lemma not in _REFERENCE_PRONOUNS:
            continue
        ranked = _unique_concept_rankings(_rank_candidates(token, candidates))
        resolutions.append(
            _choose_reference(
                kind="explicit_pronoun",
                token_index=token.index,
                surface=token.surface,
                semantic_role=token.semantic_role,
                ranked=ranked,
            )
        )

    for frame in structure.clauses:
        if frame.predicate_index is None or frame.direct_object_indices:
            continue
        if any(
            link.parent_clause_index == frame.clause_index
            and link.relation.startswith("ccomp")
            for link in structure.clause_links
        ):
            # The predicate already has a whole subordinate proposition as
            # its content (``изучи, что такое небо``).  Do not fill it with an
            # unrelated noun from an older turn merely because no overt
            # accusative object exists in the matrix clause.
            continue
        predicate = structure.tokens[frame.predicate_index]
        profile = predicate_valency(predicate.lemma)
        requires_object = bool(
            profile
            and any(
                slot.role == "direct_object" and slot.required_in_isolated_fact
                for slot in profile.slots
            )
        ) or dict(predicate.grammatical_features).get("transitivity") == "transitive"
        if not requires_object:
            continue
        ranked = _unique_concept_rankings(tuple(
            sorted(
                (
                    (
                        item.score + (0.16 if item.semantic_role == "direct_object" else 0.0),
                        item,
                    )
                    for item in candidates
                ),
                key=lambda pair: (-pair[0], pair[1].source_distance, pair[1].lemma),
            )
        ))
        resolutions.append(
            _choose_reference(
                kind="implicit_direct_object",
                token_index=None,
                surface=predicate.surface,
                semantic_role="direct_object",
                ranked=ranked,
            )
        )
    return tuple(resolutions)


def _rank_candidates(
    token: TokenInterpretation,
    candidates: tuple[ReferenceCandidate, ...],
) -> tuple[tuple[float, ReferenceCandidate], ...]:
    features = dict(token.grammatical_features)
    ranked: list[tuple[float, ReferenceCandidate]] = []
    for candidate in candidates:
        # Demonstrative ``это`` points at a proposition or discourse topic and
        # does not inherit the neuter agreement restrictions of nominal
        # ``оно``.  Applying gender here made ``что это`` reject a masculine
        # antecedent such as ``рэпер`` before topic ranking even began.
        if token.lemma != "это" and not _compatible(features, candidate):
            continue
        score = candidate.score
        if token.semantic_role == "subject":
            score += 0.08 if candidate.semantic_role == "subject" else 0.0
            score += 0.10 if candidate.animacy == "animate" else 0.0
        if token.semantic_role == "direct_object":
            score += 0.08 if candidate.semantic_role == "direct_object" else 0.0
        if token.lemma == "это":
            score += 0.12 if candidate.semantic_role == "direct_object" else 0.0
        ranked.append((min(1.0, score), candidate))
    return tuple(
        sorted(ranked, key=lambda pair: (-pair[0], pair[1].source_distance, pair[1].lemma))
    )


def _unique_concept_rankings(
    ranked: tuple[tuple[float, ReferenceCandidate], ...],
) -> tuple[tuple[float, ReferenceCandidate], ...]:
    """Compare distinct referents, not repeated mentions of one referent.

    Repeating ``небо`` across several recent turns is supporting continuity,
    not ambiguity between several different skies.  The previous margin check
    compared those duplicate mentions against each other and therefore refused
    to resolve an otherwise unique omitted object.
    """

    best_by_lemma: dict[str, tuple[float, ReferenceCandidate]] = {}
    for score, candidate in ranked:
        current = best_by_lemma.get(candidate.lemma)
        if current is None or score > current[0]:
            best_by_lemma[candidate.lemma] = (score, candidate)
    return tuple(
        sorted(
            best_by_lemma.values(),
            key=lambda pair: (-pair[0], pair[1].source_distance, pair[1].lemma),
        )
    )


def _choose_reference(
    *,
    kind: str,
    token_index: int | None,
    surface: str,
    semantic_role: str,
    ranked: tuple[tuple[float, ReferenceCandidate], ...],
) -> ReferenceResolution:
    alternatives = tuple(item for _, item in ranked[:3])
    if not ranked:
        return ReferenceResolution(
            kind,
            token_index,
            surface,
            semantic_role,
            None,
            (),
            0.0,
            "no_compatible_antecedent",
        )
    best_score, best = ranked[0]
    margin = best_score - ranked[1][0] if len(ranked) > 1 else best_score
    threshold = 0.68 if kind == "explicit_pronoun" else 0.76
    required_margin = 0.13 if len(ranked) > 1 else 0.0
    if best_score >= threshold and margin >= required_margin:
        return ReferenceResolution(
            kind,
            token_index,
            surface,
            semantic_role,
            best,
            alternatives,
            min(1.0, best_score * 0.76 + min(1.0, margin) * 0.24),
            "unique_compatible_antecedent",
        )
    return ReferenceResolution(
        kind,
        token_index,
        surface,
        semantic_role,
        None,
        alternatives,
        max(0.0, min(1.0, best_score - 0.20)),
        "ambiguous_antecedent",
    )


def _compatible(features: dict[str, str], candidate: ReferenceCandidate) -> bool:
    for name in ("gender", "number"):
        expected = features.get(name)
        actual = getattr(candidate, name)
        if expected is not None and actual is not None and expected != actual:
            return False
    return True


def _concept_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    matched = sum(
        any(
            same_lexical_family(
                item,
                candidate,
                minimum_prefix=4,
                minimum_similarity=0.68,
            )
            for candidate in right
        )
        for item in left
    )
    return matched / max(1, min(len(left), len(right)))


def _message_for_concepts(messages: tuple[str, ...], concepts: tuple[str, ...]) -> str | None:
    target = set(concepts)
    return next(
        (message for message in reversed(messages) if target & set(content_lemmas(message))),
        None,
    )


def _repair_target(prompt: str, messages: tuple[str, ...]) -> str | None:
    normalized = " ".join(prompt.casefold().replace("ё", "е").split())
    terms = set(content_lemmas(prompt))
    repair_vocabulary = {
        "ответить",
        "ответ",
        "вопрос",
        "нормально",
        "предыдущий",
        "прошлый",
        "ты",
        "вы",
    }
    explicit_failure = "не ответ" in normalized or "без ответа" in normalized
    retry_command = "ответить" in terms and bool(terms & {"вопрос", "нормально"})
    structure = analyze_structured_utterance(prompt)
    meta_question_reference = any(
        frame.predicate_index is not None
        and structure.tokens[frame.predicate_index].lemma in _ASK_PREDICATES
        and bool(frame.subject_indices)
        and structure.tokens[frame.subject_indices[0]].lemma in {"я", "мы"}
        for frame in structure.clauses
    )
    if not (explicit_failure or retry_command or meta_question_reference):
        return None
    if not meta_question_reference and terms - repair_vocabulary:
        return None
    return next(
        (
            message
            for message in reversed(messages)
            if message.rstrip().endswith("?")
            and not ({"ответить", "вопрос"} <= set(content_lemmas(message)))
        ),
        None,
    )
