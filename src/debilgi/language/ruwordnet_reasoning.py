"""Productive definitions from an explicitly selected thesaurus sense."""

from __future__ import annotations

from dataclasses import replace

from debilgi.language.content import content_lemmas
from debilgi.language.definition_relations import definition_answer_thought
from debilgi.language.lexicon import WordAnalysis
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.language.recovery import lexical_lookup_target


def ruwordnet_answer_thought(
    prompt: str, analyses: tuple[WordAnalysis, ...],
) -> ClauseThought | None:
    target = lexical_lookup_target(prompt)
    if target is None:
        return None
    focus = set(content_lemmas(target))
    matched = [a for a in analyses if a.source == "ruwordnet" and a.lemma in focus]
    if len(matched) != 1:
        return None
    chosen = matched[0]
    if chosen.sense_id is None:
        if not 2 <= len(chosen.sense_options) <= 4:
            return None
        alternatives = " или ".join(f"«{label}»" for _, label in chosen.sense_options)
        if len(alternatives) > 240:
            return None
        return ClauseThought(
            predicate=ConceptLexeme("означать", "означает", "verb", (("tense", "present"),)),
            dependencies=(
                SemanticDependency("subject", ConceptArgument(
                    ConceptLexeme(chosen.lemma, f"«{chosen.lemma}»", "noun",
                                  entity_kind="quoted_lexeme"),
                )),
                SemanticDependency("direct_object", ConceptArgument(ConceptLexeme(
                    chosen.lemma, alternatives, "noun", entity_kind="quoted_sense_options",
                ))),
            ), clause_kind="statement", confidence=0.85,
            evidence=tuple(f"ruwordnet:unresolved_sense:{sid}" for sid in chosen.sense_candidates),
        )
    # Several direct hypernyms of one selected synset are compatible facts,
    # unlike definitions from competing senses. Realize one compact relation.
    links = sorted((link for link in chosen.typed_relations if link.relation == "is_a"),
                   key=lambda link: (len(link.target.split()), len(link.target), link.target_id))
    for link in links:
        thought = definition_answer_thought(
            prompt, ((chosen.lemma, (("is_a", link.target),)),),
        )
        if thought is not None:
            return replace(thought, evidence=(
                *chosen.relation_evidence,
                f"ruwordnet_selected_edge:{chosen.sense_id}:is_a:{link.target_id}",
            ))
    return None
