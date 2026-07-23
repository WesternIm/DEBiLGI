"""Semantic role alignment between a question and a generated candidate."""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.language.content import content_lemmas
from debilgi.language.lexical_matching import same_lexical_family
from debilgi.language.structured_utterance import (
    ClauseFrame,
    StructuredUtterance,
    analyze_structured_utterance,
)
from debilgi.language.tokenizer import tokenize

_FIRST_PERSON = frozenset({"я", "мы"})
_SECOND_PERSON = frozenset({"ты", "вы"})
_POLAR_MARKERS = frozenset({"да", "нет", "не", "верно", "неверно"})
_FOCUS_PARTS = frozenset({"noun", "adjective", "participle", "verb", "predicate"})
_FIXED_PRONOUN_FOCUS = frozenset({"себя"})


@dataclass(frozen=True, slots=True)
class ResponseAlignment:
    allowed: bool
    reasons: tuple[str, ...]
    expected_subjects: tuple[str, ...]
    candidate_subjects: tuple[str, ...]
    question_focus: tuple[str, ...]
    matched_focus: tuple[str, ...]


def assess_response_alignment(
    prompt: StructuredUtterance,
    candidate_text: str,
) -> ResponseAlignment:
    """Reject fluent graph text that answers another subject or predicate."""

    frame = _answer_target_frame(prompt)
    if frame is None:
        return ResponseAlignment(True, (), (), (), (), ())
    candidate = analyze_structured_utterance(candidate_text)
    candidate_frame = next(
        (
            item
            for item in candidate.clauses
            if item.predicate_index is not None or item.implicit_predicate is not None
        ),
        None,
    )
    expected_subjects = _subjects(prompt, frame)
    candidate_subjects = _subjects(candidate, candidate_frame) if candidate_frame else ()
    focus = _question_focus(prompt, frame)
    candidate_terms = set(content_lemmas(candidate_text))
    candidate_terms.update(
        token.casefold().replace("ё", "е")
        for token in tokenize(candidate_text)
        if (
            token.casefold().replace("ё", "е") in _FIXED_PRONOUN_FOCUS
            or token.isdigit()
            or token.isalnum() and len(token) <= 5
        )
    )
    matched_focus = tuple(
        item
        for item in focus
        if any(
            same_lexical_family(
                item,
                term,
                minimum_prefix=4,
                minimum_similarity=0.70,
            )
            for term in candidate_terms
        )
    )
    words = tuple(token.casefold().replace("ё", "е") for token in tokenize(candidate_text))
    polar_short = len(words) <= 4 and bool(set(words) & _POLAR_MARKERS)
    reasons: list[str] = []
    if (
        expected_subjects
        and candidate_subjects
        and not _subjects_align(
            expected_subjects,
            candidate_subjects,
            candidate_terms,
        )
    ):
        reasons.append("answer_subject_mismatch")
    if focus and not matched_focus and not polar_short:
        reasons.append("answer_focus_missing")
    return ResponseAlignment(
        not reasons,
        tuple(reasons),
        expected_subjects,
        candidate_subjects,
        focus,
        matched_focus,
    )


def _answer_target_frame(structure: StructuredUtterance) -> ClauseFrame | None:
    questions = tuple(item for item in structure.clauses if item.clause_kind == "question")
    complete = tuple(
        item
        for item in questions
        if item.predicate_index is not None or item.implicit_predicate is not None
    )
    return complete[-1] if complete else questions[-1] if questions else None


def _subjects(
    structure: StructuredUtterance,
    frame: ClauseFrame | None,
) -> tuple[str, ...]:
    if frame is None:
        return ()
    return tuple(dict.fromkeys(structure.tokens[index].lemma for index in frame.subject_indices))


def _question_focus(
    structure: StructuredUtterance,
    frame: ClauseFrame,
) -> tuple[str, ...]:
    if frame.predicate_index is None:
        return tuple(structure.tokens[index].lemma for index in frame.predicate_complement_indices)
    subject_indices = set(frame.subject_indices)
    focus = tuple(
        token.lemma
        for token in structure.tokens[frame.predicate_index + 1 : frame.token_end]
        if token.index not in subject_indices
        and (
            token.part_of_speech in _FOCUS_PARTS
            or token.lemma in _FIXED_PRONOUN_FOCUS
            or token.semantic_role == "unassigned_content"
            or (
                token.part_of_speech is None
                and token.semantic_role.startswith("complement:")
            )
        )
        and token.semantic_role not in {"function_word", "vocative"}
    )
    return tuple(dict.fromkeys(focus or (structure.tokens[frame.predicate_index].lemma,)))


def _subjects_align(
    expected: tuple[str, ...],
    actual: tuple[str, ...],
    candidate_terms: set[str],
) -> bool:
    expected_set = set(expected)
    actual_set = set(actual)
    if expected_set & _SECOND_PERSON:
        return bool(actual_set & _FIRST_PERSON) or bool(expected_set & actual_set)
    if expected_set & _FIRST_PERSON:
        return bool(actual_set & _SECOND_PERSON) or bool(expected_set & actual_set)
    return bool(expected_set & (actual_set | candidate_terms))
