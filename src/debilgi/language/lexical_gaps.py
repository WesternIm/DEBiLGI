"""Separate missing definitions from information needed for the current task."""

from __future__ import annotations

from functools import lru_cache

from debilgi.language.structured_utterance import StructuredUtterance, analyze_structured_utterance


def nonblocking_event_terms(
    structure: StructuredUtterance, unknown_terms: tuple[str, ...],
) -> tuple[str, ...]:
    if len(structure.clauses) != 1 or structure.clause_links:
        return ()
    if any(t.lemma in {"если", "бы", "возможно", "наверное", "кажется"}
           for t in structure.tokens):
        return ()
    frame = structure.clauses[0]
    if (frame.predicate_index is None or frame.modal_lemmas or frame.coordinations
            or frame.action_complement_indices or frame.predicate_complement_indices
            or not frame.subject_indices):
        return ()
    predicate = structure.tokens[frame.predicate_index]
    if predicate.part_of_speech != "verb" or not predicate.morphology_known:
        return ()
    # Literal event reports and requests for event slots can preserve a name
    # without knowing its definition. Causes, properties, permission and
    # classifications need semantic evidence instead.
    target = (structure.tokens[frame.question_target_index].lemma
              if frame.question_target_index is not None else None)
    if frame.clause_kind == "question" and target not in {"когда", "где", "куда", "кто", "что"}:
        return ()
    if frame.clause_kind not in {"statement", "question"}:
        return ()
    if dict(predicate.grammatical_features).get("tense") != "past":
        return ()
    indices = {
        *frame.subject_indices, *frame.direct_object_indices,
        *(index for complement in frame.complements
          if complement.role in {"location", "location_or_destination"}
          for index in complement.object_indices),
    }
    references = {
        structure.tokens[i].lemma for i in indices
        if structure.tokens[i].part_of_speech == "noun"
    }
    # A literal event slot can be recalled without resolving every sense of
    # the action verb. This is structural recognition, never a definition.
    references.add(predicate.lemma)
    references.update(t.lemma for t in structure.tokens
                      if t.surface in {"вчера", "сегодня", "позавчера", "утром", "вечером"})
    forms = {t.surface for t in structure.tokens if t.lemma in references}
    return tuple(term for term in unknown_terms if term in references or term in forms)


def nonblocking_reference_terms(
    structure: StructuredUtterance, unknown_terms: tuple[str, ...],
) -> tuple[str, ...]:
    allowed = nonblocking_event_terms(structure, unknown_terms)
    noun_forms = {form for t in structure.tokens if t.part_of_speech == "noun"
                  for form in (t.lemma, t.surface)}
    return tuple(term for term in allowed if term in noun_forms)


@lru_cache(maxsize=512)
def irrelevant_lexical_question(prompt: str, response: str) -> bool:
    if not response.rstrip().endswith("?"):
        return False
    question = analyze_structured_utterance(response)
    if not any(t.lemma in {"подразумевать", "означать", "значить"} for t in question.tokens):
        return False
    structure = analyze_structured_utterance(prompt)
    allowed = nonblocking_event_terms(structure, tuple(t.lemma for t in structure.tokens))
    return any(t.lemma in allowed for t in question.tokens)
