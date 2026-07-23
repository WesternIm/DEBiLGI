"""Structural warm-start policy without authored prompt-to-reply pairs.

Earlier versions stored complete conversational answers here and later exposed
them through exact archive retrieval.  That made the bot select a prepared
surface instead of composing an answer from the current semantic state.
Language bootstrapping now uses the structural curricula and productive NLG;
therefore this compatibility module intentionally contains no reply pairs.
"""

from __future__ import annotations

WARM_START_PAIRS: tuple[tuple[str, str], ...] = ()

WARM_START_SKILLS = (
    "brief_answer",
    "clarification",
    "comparison",
    "concept_definition",
    "context_repair",
    "contextual_followup",
    "continuation",
    "evidence_awareness",
    "dialogue_act",
    "example_request",
    "greeting",
    "identity",
    "lexical_lookup",
    "memory_recall",
    "response_stability",
    "self_correction",
    "negative_imperative",
    "evidence_based_stance",
    "partial_position",
    "rest_directive",
    "paraphrase",
    "stepwise_explanation",
    "thanks",
    "source_routing",
    "contextual_interpretation",
    "explanation_learning",
    "source_conflict_tracking",
)
