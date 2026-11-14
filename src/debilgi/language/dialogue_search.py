"""Search semantic continuations for a user assertion.

The search returns meaning graphs only. It has no response phrase table and no
mapping from a particular input sentence to a particular answer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from debilgi.language.communication import MessagePlan, ReaderGoal
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)


@dataclass(frozen=True, slots=True)
class DialogueSearchCandidate:
    plan: MessagePlan
    operation: str
    score: float
    evidence: tuple[str, ...]


def search_continuations(
    goal: ReaderGoal,
    claim: ClauseThought,
    *,
    evidence_id: str,
) -> tuple[DialogueSearchCandidate, ...]:
    """Return independently scored next semantic moves for one assertion."""
    candidates: list[DialogueSearchCandidate] = []
    # A modifier that is itself a pronoun represents an unresolved reference
    # or quantity. Replace it with a question concept while preserving the
    # original noun, predicate, action and all grammatical relations.
    subject_item = next((item for item in claim.dependencies if item.role == "subject"), None)
    if subject_item is not None:
        quantifier = next(
            (item for item in subject_item.argument.modifiers if item.part_of_speech == "pronoun"),
            None,
        )
        if quantifier is not None:
            question_word = (
                "кто" if subject_item.argument.head.part_of_speech == "pronoun" else "какие"
            )
            question_subject = replace(
                subject_item.argument,
                modifiers=(ConceptLexeme(question_word, question_word, "pronoun"),),
            )
            question = replace(
                claim,
                clause_kind="question",
                response_polarity=None,
                focus_role="subject",
                dependencies=tuple(
                    SemanticDependency("subject", question_subject)
                    if item.role == "subject"
                    else item
                    for item in claim.dependencies
                ),
                evidence=(evidence_id,),
            )
            candidates.append(
                DialogueSearchCandidate(
                    MessagePlan(goal, question),
                    "resolve_open_group",
                    0.58,
                    (evidence_id, "unresolved_group_reference"),
                )
            )
    return tuple(sorted(candidates, key=lambda item: (-item.score, item.operation)))


def query_unobserved_relation(
    goal: ReaderGoal,
    claim: ClauseThought,
    *,
    evidence_id: str,
) -> tuple[DialogueSearchCandidate, ...]:
    """Ask for an absent relation of an actual report, preserving its participants.

    A complete assertion is already usable evidence for a question about that
    assertion. No explanation or new event is asserted without a premise.
    """
    subject = next((d.argument.head for d in claim.dependencies if d.role == "subject"), None)
    if subject is None or subject.lemma in {"я", "мы"} or claim.clause_kind != "statement":
        return ()
    if any(d.role == "cause" for d in claim.dependencies):
        return ()
    question = replace(
        claim,
        clause_kind="question",
        focus_role="cause",
        response_polarity=None,
        dependencies=(
            *claim.dependencies,
            SemanticDependency(
                "cause",
                ConceptArgument(ConceptLexeme("почему", "почему", "adverb")),
            ),
        ),
        evidence=(evidence_id,),
    )
    return (
        DialogueSearchCandidate(
            MessagePlan(goal, question),
            "query_unobserved_relation",
            claim.confidence,
            (evidence_id, "unobserved_cause"),
        ),
    )
