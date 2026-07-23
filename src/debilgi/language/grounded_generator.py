"""Conservative runtime fallback without Markov or external text generation.

Semantic answer modules normally produce ``ResponseContentPlan`` instances
before this adapter is reached.  The adapter only exposes already grounded,
complete local facts; it never joins fragments, predicts a next token or asks
an external model to write a reply.
"""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.language.content import content_lemmas
from debilgi.language.planning import UtterancePlan
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.language.surface_coherence import evaluate_surface
from debilgi.language.surface_policy import internal_surface_reason
from debilgi.language.valency import assess_predicate_valency


@dataclass(frozen=True, slots=True)
class GroundedUtterance:
    text: str
    variant: int
    renderer: str = "grounded_local_fact"
    source: str = "grounded_surface"


class GroundedConstructionGenerator:
    """Expose only locally grounded complete surfaces as a final backstop."""

    def __init__(self, *, max_output_chars: int = 1200) -> None:
        if max_output_chars <= 0:
            raise ValueError("output limit must be positive")
        self.max_output_chars = max_output_chars

    def generate_candidates(
        self,
        plan: UtterancePlan,
        *,
        limit: int = 8,
    ) -> tuple[GroundedUtterance, ...]:
        prompt_terms = set(content_lemmas(plan.context.prompt))
        required = set(plan.context.required_facts)
        facts = (
            *((plan.context.exact_answer,) if plan.context.exact_answer else ()),
            *plan.context.required_facts,
            *plan.context.knowledge_facts,
        )
        result: list[GroundedUtterance] = []
        seen: set[str] = set()
        for fact in facts:
            text = " ".join(fact.split()).strip()
            normalized = text.casefold()
            if not text or normalized in seen or len(text) > self.max_output_chars:
                continue
            fact_terms = set(content_lemmas(text))
            if fact not in required and prompt_terms and not prompt_terms & fact_terms:
                continue
            surface = evaluate_surface(text)
            if internal_surface_reason(text) is not None:
                continue
            if surface.hard_issues or surface.score < 0.72:
                continue
            if not _is_complete_grounded_fact(text):
                continue
            seen.add(normalized)
            result.append(GroundedUtterance(text, len(result)))
            if len(result) >= max(1, min(limit, 24)):
                break
        return tuple(result)

    def generate(self, plan: UtterancePlan) -> str:
        candidates = self.generate_candidates(plan, limit=1)
        return candidates[0].text if candidates else ""


def _is_complete_grounded_fact(text: str) -> bool:
    """Reject memory fragments that only resemble complete propositions."""

    structure = analyze_structured_utterance(text)
    found_proposition = False
    for clause in structure.clauses:
        if clause.predicate_index is None:
            if (
                clause.implicit_predicate == "быть"
                and clause.subject_indices
                and clause.predicate_complement_indices
            ):
                found_proposition = True
                continue
            return False
        found_proposition = True
        predicate = structure.tokens[clause.predicate_index].lemma
        present_roles = frozenset(
            role
            for indices, role in (
                (clause.subject_indices, "subject"),
                (clause.direct_object_indices, "direct_object"),
                (clause.indirect_object_indices, "indirect_object"),
                (clause.predicate_complement_indices, "predicate_complement"),
                (clause.action_complement_indices, "action_complement"),
            )
            if indices
        )
        valency = assess_predicate_valency(
            predicate,
            present_roles=present_roles,
            complement_roles=frozenset(item.role for item in clause.complements),
            implicit_subject=clause.implicit_subject is not None,
            imperative=clause.clause_kind == "command",
        )
        if valency.missing_roles or valency.unexpected_roles:
            return False
    return found_proposition
