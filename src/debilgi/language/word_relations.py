"""Learn lexical dependencies and preserve grammatical participants in speech.

The stored objects are word relations, never input/answer pairs or reply text.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from debilgi.language.morphology import DEFAULT_MORPHOLOGY, deictic_counterpart
from debilgi.language.structured_utterance import StructuredUtterance, analyze_structured_utterance


@dataclass(frozen=True, slots=True)
class WordRelation:
    governor: str
    dependent: str
    relation: str
    dependent_pos: str
    case: str
    count: int = 1


@dataclass(frozen=True, slots=True)
class RelationCoherence:
    """How much of a generated dependency graph was actually learned.

    ``unsupported`` means the governor has never been observed and therefore
    cannot be judged yet. ``conflicts`` means the governor is known, but the
    generated edge gives it a role which the language memory has never seen.
    The latter is the dangerous Markov-chain splice (for example, turning an
    intransitive predicate into a verb with a direct object).
    """

    score: float
    supported: int
    generalized: int
    unsupported: int
    conflicts: int


@lru_cache(maxsize=2048)
def observe_word_relations(
    text: str | StructuredUtterance,
    *,
    human_perspective: bool = False,
) -> tuple[WordRelation, ...]:
    structure = (
        text if isinstance(text, StructuredUtterance) else analyze_structured_utterance(text)
    )
    quoted = _quoted_indices(structure.tokens)
    reported = {
        link.child_clause_index
        for link in structure.clause_links
        if link.relation.startswith("ccomp")
    }

    def lemma(token):
        if (
            human_perspective
            and token.index not in quoted
            and token.clause_index not in reported
            and structure.clauses[token.clause_index].clause_kind != "command"
        ):
            return (
                deictic_counterpart(token.lemma) or token.lemma
            )
        return token.lemma

    result = []
    for arc in structure.dependencies:
        head, child = structure.tokens[arc.head_index], structure.tokens[arc.dependent_index]
        if head.semantic_role == "pragmatic_marker" or child.semantic_role == "pragmatic_marker":
            continue
        if not head.part_of_speech or not child.part_of_speech:
            continue
        features = dict(child.grammatical_features)
        head_features = dict(head.grammatical_features)
        if (
            arc.relation == "nsubj"
            and head_features.get("person") in {"first", "second"}
            and child.part_of_speech == "noun"
        ):
            continue
        result.append(
            WordRelation(
                lemma(head),
                lemma(child),
                arc.relation,
                child.part_of_speech,
                features.get("case", ""),
            )
        )
    return tuple(result)


def _quoted_indices(tokens) -> set[int]:
    inside = False
    indices = set()
    for token in tokens:
        if token.surface in {'"', "«", "»", "“", "”", "„"}:
            inside = not inside
        elif inside:
            indices.add(token.index)
    return indices


@lru_cache(maxsize=2048)
def grammatical_tokens(
    text: str | StructuredUtterance,
    *,
    human_perspective: bool = False,
    prune_unbound: bool = False,
) -> tuple[str, ...]:
    """Revoice participants and agree forms without changing lexical concepts."""
    structure = (
        text if isinstance(text, StructuredUtterance) else analyze_structured_utterance(text)
    )
    output = [token.corrected_form or token.surface.casefold() for token in structure.tokens]
    quoted = _quoted_indices(structure.tokens)
    omitted = set()
    # Reported clauses have their own speaker; do not invert embedded quotations.
    reported = {
        link.child_clause_index
        for link in structure.clause_links
        if link.relation.startswith("ccomp")
    }
    for clause in structure.clauses:
        if clause.clause_index in reported or clause.clause_kind == "command":
            continue
        if (
            prune_unbound
            and clause.subject_indices
            and clause.predicate_index is not None
            and not clause.coordinations
            and not structure.clause_links
        ):
            omitted.update(
                token.index
                for token in structure.tokens[clause.token_start : clause.token_end]
                if token.index not in quoted
                and token.index != clause.question_target_index
                and token.semantic_role == "unassigned_content"
                and token.part_of_speech == "pronoun"
                and dict(token.grammatical_features).get("case") == "nominative"
            )
        for token in structure.tokens[clause.token_start : clause.token_end]:
            if not human_perspective or token.index in quoted:
                continue
            mapped = deictic_counterpart(token.lemma)
            if mapped:
                features = {
                    name: value
                    for name, value in token.grammatical_features
                    if name in {"case", "number", "gender"}
                }
                output[token.index] = DEFAULT_MORPHOLOGY.inflect(mapped, **features) or mapped
        if clause.predicate_index is None or clause.predicate_index in quoted:
            continue
        predicate = structure.tokens[clause.predicate_index]
        features = dict(predicate.grammatical_features)
        if predicate.part_of_speech != "verb" or "INFN" in predicate.grammemes:
            continue
        subject = next(
            (
                structure.tokens[index]
                for index in clause.subject_indices
                if index not in quoted
                and structure.tokens[index].part_of_speech in {"noun", "pronoun"}
            ),
            None,
        )
        explicit_person = next(
            (
                token
                for token in structure.tokens[clause.token_start : clause.token_end]
                if token.index not in quoted
                and deictic_counterpart(token.lemma, possessives=False) is not None
                and dict(token.grammatical_features).get("case") == "nominative"
            ),
            None,
        )
        if explicit_person is not None:
            subject = explicit_person
        elif (
            subject is not None
            and subject.part_of_speech == "noun"
            and (features.get("person") in {"first", "second"})
        ):
            # "ногти потеряешь" has an omitted addressee, not acting nails.
            subject = None
        agreement = {}
        if subject is not None:
            reading = DEFAULT_MORPHOLOGY.best(output[subject.index])
            if reading and any(
                item.case == "nominative"
                for item in DEFAULT_MORPHOLOGY.parse_word(output[subject.index])
            ):
                agreement = {"number": reading.number or "singular"}
                if features.get("tense") == "past" and agreement["number"] == "singular":
                    gender = reading.gender or features.get("gender")
                    if gender:
                        agreement["gender"] = gender
                elif features.get("tense") != "past":
                    agreement["person"] = reading.person or "third"
        elif human_perspective and features.get("person") in {"first", "second"}:
            agreement["person"] = "second" if features["person"] == "first" else "first"
            agreement["number"] = features.get("number", "singular")
        if agreement:
            if features.get("tense"):
                agreement["tense"] = features["tense"]
            output[predicate.index] = (
                DEFAULT_MORPHOLOGY.inflect(
                    predicate.lemma,
                    **agreement,
                )
                or output[predicate.index]
            )
    return tuple(token for index, token in enumerate(output) if index not in omitted)


def relation_fit(text: str, learned: tuple[WordRelation, ...]) -> float:
    """Score lexical partners and case usage, with generalization to new nouns."""
    if not learned:
        return 0.0
    observations = observe_word_relations(text)
    scores = []
    for actual in observations:
        relevant = tuple(
            link
            for link in learned
            if link.governor == actual.governor and link.relation == actual.relation
        )
        if not relevant:
            if any(link.governor == actual.governor for link in learned):
                scores.append(0.0)
            continue
        total = sum(link.count for link in relevant)
        lexical = sum(link.count for link in relevant if link.dependent == actual.dependent) / total
        pos = (
            sum(link.count for link in relevant if link.dependent_pos == actual.dependent_pos)
            / total
        )
        cases = Counter()
        for link in relevant:
            if link.case:
                cases[link.case] += link.count
        case = cases[actual.case] / sum(cases.values()) if cases and actual.case else 0.5
        scores.append(lexical * 0.5 + pos * 0.3 + case * 0.2)
    return sum(scores) / len(scores) if scores else 0.0


def relation_coherence(text: str, learned: tuple[WordRelation, ...]) -> RelationCoherence:
    """Validate new wording against observed governor/role combinations.

    A different noun may occupy a learned slot when its part of speech and
    case match. This permits productive language. A known governor acquiring
    a completely new dependency role is a conflict and must not be mistaken
    for creativity merely because both words occurred somewhere in the DB.
    """

    observations = observe_word_relations(text)
    if not observations or not learned:
        return RelationCoherence(0.0, 0, 0, len(observations), 0)

    by_governor: dict[str, list[WordRelation]] = {}
    for link in learned:
        by_governor.setdefault(link.governor, []).append(link)

    supported = generalized = unsupported = conflicts = 0
    for actual in observations:
        governor_links = by_governor.get(actual.governor)
        if not governor_links:
            unsupported += 1
            continue
        role_links = tuple(link for link in governor_links if link.relation == actual.relation)
        if not role_links:
            conflicts += 1
            continue
        if any(link.dependent == actual.dependent for link in role_links):
            supported += 1
            continue
        compatible = any(
            link.dependent_pos == actual.dependent_pos
            and (not link.case or not actual.case or link.case == actual.case)
            for link in role_links
        )
        if compatible:
            generalized += 1
        else:
            conflicts += 1

    judged = supported + generalized + conflicts
    score = (supported + generalized * 0.72) / judged if judged else 0.0
    return RelationCoherence(score, supported, generalized, unsupported, conflicts)
