"""Detect reorderings of the same proposition without merging reversed roles."""

from functools import lru_cache

from debilgi.language.structured_utterance import analyze_structured_utterance


@lru_cache(maxsize=1024)
def proposition_signature(text: str) -> tuple:
    structure = analyze_structured_utterance(text)
    result = []
    for clause in structure.clauses:
        if clause.predicate_index is None:
            return ()
        roles = tuple(
            tuple(sorted(structure.tokens[i].lemma for i in indices))
            for indices in (
                clause.subject_indices,
                clause.direct_object_indices,
                clause.indirect_object_indices,
                clause.predicate_complement_indices,
                clause.action_complement_indices,
            )
        )
        # Include every content word, so changed modifiers/locations are new information.
        content = tuple(
            sorted(
                t.lemma
                for t in structure.tokens[clause.token_start : clause.token_end]
                if t.part_of_speech not in {None, "particle", "conjunction"}
            )
        )
        predicate = structure.tokens[clause.predicate_index]
        result.append(
            (
                clause.clause_kind,
                clause.negated,
                predicate.lemma,
                tuple(
                    (name, value)
                    for name, value in predicate.grammatical_features
                    if name in {"tense", "mood", "voice"}
                ),
                roles,
                content,
            )
        )
    return tuple(result)


def repeats_proposition(text: str, previous: tuple[str, ...]) -> bool:
    signature = proposition_signature(text)
    return bool(signature) and any(signature == proposition_signature(item) for item in previous)
