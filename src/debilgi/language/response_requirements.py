"""Mandatory semantic goals, shared by normal generation and recovery."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

from debilgi.language.repair_signals import signals_prior_answer_correction
from debilgi.language.structured_utterance import analyze_structured_utterance


@lru_cache(maxsize=512)
def response_requirement(prompt: str) -> str | None:
    if signals_prior_answer_correction(prompt):
        return "review_prior_claim"
    structure = analyze_structured_utterance(prompt)
    for clause in structure.clauses:
        if clause.question_target_index is not None:
            target = structure.tokens[clause.question_target_index]
            if target.lemma == "почему":
                return "explain_reason"
    return None


def requirement_rejection(
    prompt: str, metadata: Mapping[str, object], *, text: str = "",
) -> str | None:
    requirement = response_requirement(prompt)
    if requirement and metadata.get("reaction_meaning") == "express_limit":
        from debilgi.language.dialogue_learning import observed_reaction_binding

        if observed_reaction_binding(prompt, text, metadata):
            return None
    if requirement == "explain_reason":
        if metadata.get("answer_slot") == "почему" and metadata.get("reasoning_evidence"):
            return None
        if metadata.get("causal_path") and metadata.get("causal_direction") == "cause":
            return None
        if metadata.get("unresolved_requirement") == requirement:
            return None
        return "reason_required_not_polarity_or_word_meaning"
    if requirement == "review_prior_claim":
        if metadata.get("reviewed_agent_claim") and metadata.get("claim_source") == "dialogue":
            return None
        if metadata.get("unresolved_requirement") == requirement:
            return None
        return "prior_claim_review_required"
    return None
