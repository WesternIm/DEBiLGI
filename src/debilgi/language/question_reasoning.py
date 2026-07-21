"""Sentence-level question frames and replies grounded in internalized knowledge.

Reference sources are teachers for the lexicon and semantic graph.  This module
never exposes a dictionary or encyclopedia extract as a reply.  It only uses
relations that have already been internalized into lexical analyses or memory
facts, and otherwise records that the *knowledge* is incomplete rather than
pretending that a complete question was malformed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from debilgi.language.content import content_lemmas
from debilgi.language.lexicon import WordAnalysis
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
    realize_thought,
)
from debilgi.language.structured_utterance import StructuredUtterance, TokenInterpretation

_STANCE_LEMMAS = frozenset({"думать", "полагать", "считать", "судить", "казаться"})
_DIRECTIVE_LEMMAS = frozenset(
    {
        "брать",
        "выбирать",
        "выбрать",
        "назвать",
        "объяснить",
        "ответить",
        "предпочитать",
        "рассказать",
        "сказать",
    }
)
_QUESTION_SCAFFOLD = frozenset(
    {
        "кто",
        "что",
        "как",
        "какой",
        "почему",
        "один",
        "вариант",
        "причина",
    }
)
_ARGUMENT_PARTS = frozenset({"noun", "pronoun"})
_CONTEXT_PARTS = frozenset({"noun", "verb", "adjective", "participle", "adverb"})


@dataclass(frozen=True, slots=True)
class SemanticQuestionFrame:
    kind: str
    focus: str | None
    alternatives: tuple[str, ...]
    alternative_surfaces: tuple[str, ...]
    context_terms: tuple[str, ...]
    study_terms: tuple[str, ...]
    complete: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class InternalReasoningResult:
    content_thought: ClauseThought | None
    confidence: float
    used_evidence: tuple[str, ...]
    needs_study: bool

    @property
    def text(self) -> str:
        surfaces = realize_thought(self.content_thought) if self.content_thought else ()
        return surfaces[0].text if surfaces else ""


def analyze_semantic_question(structure: StructuredUtterance) -> SemanticQuestionFrame | None:
    """Extract a general choice/comparison frame, ignoring stance scaffolding."""

    if not any(frame.clause_kind == "question" for frame in structure.clauses):
        return None
    pair = _alternative_pair(structure)
    if pair is None:
        return None
    left, right, conjunction_index = pair
    focus = _comparison_focus(structure.tokens, conjunction_index)
    alternatives = (left.lemma, right.lemma)
    context_terms = tuple(
        dict.fromkeys(
            token.lemma
            for token in structure.tokens
            if token.part_of_speech in _CONTEXT_PARTS
            if token.lemma not in set(alternatives)
            if token.lemma != focus
            if token.lemma not in _STANCE_LEMMAS
            if token.lemma not in _DIRECTIVE_LEMMAS
            if token.lemma not in _QUESTION_SCAFFOLD
            if not token.semantic_role.startswith("function_word")
        )
    )
    study_terms = tuple(
        dict.fromkeys((*alternatives, *((focus,) if focus else ()), *context_terms))
    )
    # A bare alternative question ("чай или кофе?", "краба выбираешь или
    # рака?") is already complete.  It asks for a preference, not for an
    # objective comparison criterion that must be recovered from elsewhere.
    complete = len(set(alternatives)) == 2
    confidence = min(
        0.97,
        structure.confidence * 0.62
        + min(left.confidence, right.confidence) * 0.20
        + (0.18 if complete else 0.05),
    )
    return SemanticQuestionFrame(
        kind="comparison" if focus is not None else "choice",
        focus=focus,
        alternatives=alternatives,
        alternative_surfaces=(left.surface.casefold(), right.surface.casefold()),
        context_terms=context_terms,
        study_terms=study_terms,
        complete=complete,
        confidence=confidence,
    )


def reason_from_internal_knowledge(
    frame: SemanticQuestionFrame,
    analyses: tuple[WordAnalysis, ...],
    memory_facts: tuple[str, ...] = (),
    *,
    study_completed: bool = False,
) -> InternalReasoningResult:
    """Compare alternatives using internal meanings and relations, never source prose."""

    if frame.kind == "choice" and not frame.context_terms:
        ranked_alternatives = tuple(
            sorted(zip(frame.alternatives, frame.alternative_surfaces, strict=True))
        )
        material = "|".join(item[0] for item in ranked_alternatives).encode("utf-8")
        selected_index = (
            int.from_bytes(hashlib.sha256(material).digest()[:2], "big")
            % len(ranked_alternatives)
        )
        selected_lemma, selected = ranked_alternatives[selected_index]
        confidence = max(0.72, frame.confidence * 0.88)
        return InternalReasoningResult(
            content_thought=_choice_thought(
                selected_lemma,
                selected,
                confidence=confidence,
            ),
            confidence=confidence,
            used_evidence=(selected_lemma,),
            needs_study=False,
        )

    anchors = set(frame.context_terms)
    if frame.focus:
        anchors.add(frame.focus)
    evidence_by_alternative: list[tuple[float, tuple[str, ...]]] = []
    salient_by_alternative: list[tuple[str, ...]] = []
    evidence_available: list[bool] = []
    for alternative in frame.alternatives:
        evidence_terms: set[str] = set()
        salient_terms: set[str] = set()
        score = 0.0
        lexical = next(
            (
                item
                for item in analyses
                if alternative in {item.lemma, item.normalized_form}
            ),
            None,
        )
        if lexical is not None:
            meanings = tuple(
                dict.fromkeys(
                    meaning
                    for meaning in (
                        lexical.selected_meaning,
                        lexical.local_meaning,
                        lexical.personal_interpretation,
                        *lexical.dictionary_meanings,
                    )
                    if meaning
                )
            )
            lexical_terms = {
                term for meaning in meanings[:3] for term in content_lemmas(meaning)
            } | set(lexical.associations)
            salient_terms.update(lexical_terms - {alternative})
            matched = lexical_terms & anchors
            evidence_terms.update(matched)
            score += len(matched) * 0.34
            score += min(0.22, len(lexical_terms) * 0.015) if meanings else 0.0
        for fact in memory_facts:
            fact_terms = set(content_lemmas(fact))
            if alternative not in fact_terms:
                continue
            matched = fact_terms & anchors
            evidence_terms.update(matched)
            salient_terms.update(fact_terms - {alternative})
            score += 0.25 + len(matched) * 0.42
        evidence_by_alternative.append((score, tuple(sorted(evidence_terms))))
        salient_by_alternative.append(tuple(sorted(salient_terms))[:5])
        evidence_available.append(bool(salient_terms))

    ranked = sorted(
        enumerate(evidence_by_alternative),
        key=lambda item: (item[1][0], -item[0]),
        reverse=True,
    )
    best_index, (best_score, best_terms) = ranked[0]
    second_score = ranked[1][1][0]
    if best_score >= 0.48 and best_score - second_score >= 0.18:
        selected = frame.alternative_surfaces[best_index]
        selected_lemma = frame.alternatives[best_index]
        confidence = min(0.88, 0.64 + best_score * 0.12)
        return InternalReasoningResult(
            content_thought=_choice_thought(
                selected_lemma,
                selected,
                confidence=confidence,
                evidence=best_terms,
            ),
            confidence=confidence,
            used_evidence=best_terms,
            needs_study=False,
        )

    if study_completed and any(evidence_available):
        selected = frame.alternative_surfaces[best_index]
        selected_lemma = frame.alternatives[best_index]
        basis_terms = best_terms or salient_by_alternative[best_index]
        confidence = max(0.64, frame.confidence * 0.82)
        return InternalReasoningResult(
            content_thought=_choice_thought(
                selected_lemma,
                selected,
                confidence=confidence,
                evidence=basis_terms,
            ),
            confidence=confidence,
            used_evidence=basis_terms,
            needs_study=False,
        )

    return InternalReasoningResult(
        content_thought=None,
        confidence=max(0.62, frame.confidence * 0.78),
        used_evidence=(),
        needs_study=True,
    )


def _choice_thought(
    lemma: str,
    surface: str,
    *,
    confidence: float,
    evidence: tuple[str, ...] = (),
) -> ClauseThought:
    reading = next(
        (
            item
            for item in DEFAULT_MORPHOLOGY.parse_word(surface)
            if item.lemma == lemma and item.part_of_speech == "noun"
        ),
        None,
    )
    argument_features = tuple(
        (name, value)
        for name, value in (
            ("number", reading.number if reading is not None else None),
            ("gender", reading.gender if reading is not None else None),
            ("animacy", reading.animacy if reading is not None else None),
        )
        if value is not None
    )
    return ClauseThought(
        predicate=ConceptLexeme(
            "выбирать",
            "выбираю",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "first")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme(
                        "я",
                        "я",
                        "pronoun",
                        (("number", "singular"), ("person", "first")),
                    )
                ),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme(lemma, surface, "noun", argument_features),
                    grammatical_case="accusative",
                ),
            ),
        ),
        focus_role="direct_object",
        confidence=confidence,
        evidence=evidence,
    )


def _alternative_pair(
    structure: StructuredUtterance,
) -> tuple[TokenInterpretation, TokenInterpretation, int] | None:
    tokens = structure.tokens
    structural_pair = next(
        (
            coordination
            for clause in structure.clauses
            for coordination in clause.coordinations
            if coordination.relation == "alternative"
            and len(coordination.member_indices) == 2
        ),
        None,
    )
    if structural_pair is not None:
        left_index, right_index = structural_pair.member_indices
        return tokens[left_index], tokens[right_index], structural_pair.conjunction_index

    # Compatibility fallback for low-confidence morphology. It still requires
    # an explicit alternative operator and two argument-like neighbours.
    for index, token in enumerate(tokens):
        if token.lemma != "или":
            continue
        left = next(
            (
                candidate
                for candidate in reversed(tokens[:index])
                if candidate.part_of_speech in _ARGUMENT_PARTS
                and candidate.lemma not in _QUESTION_SCAFFOLD
            ),
            None,
        )
        right = next(
            (
                candidate
                for candidate in tokens[index + 1 :]
                if candidate.part_of_speech in _ARGUMENT_PARTS
                and candidate.lemma not in _QUESTION_SCAFFOLD
            ),
            None,
        )
        if left is not None and right is not None and left.lemma != right.lemma:
            return left, right, index
    return None


def _comparison_focus(
    tokens: tuple[TokenInterpretation, ...], conjunction_index: int
) -> str | None:
    adjectives = tuple(
        token
        for token in tokens[:conjunction_index]
        if token.part_of_speech == "adjective"
        and token.lemma not in _QUESTION_SCAFFOLD
    )
    if adjectives:
        preferred = tuple(
            token
            for token in adjectives
            if token.semantic_role in {"predicate_complement", "unassigned_content"}
        )
        return (preferred or adjectives)[-1].lemma
    return None

