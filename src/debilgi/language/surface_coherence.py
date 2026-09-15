"""Structural validation for generated response surfaces."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from debilgi.language.grammar import role
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.tokenizer import tokenize
from debilgi.language.valency import assess_predicate_valency, governed_object_case

_INVALID_OPENING = frozenset(".!?…,;:)]}")
_TERMINAL = frozenset(".!?…")
_CONTINUATION_MARKERS = frozenset(
    {"затем", "поэтому", "потом", "следовательно", "причём", "причем"}
)
_CONNECTORS = frozenset(
    {"но", "затем", "потом", "поэтому", "потому"}
)
_SAFE_SHORT_ROLES = frozenset({"interjection", "predicate", "adverb"})
_MALFORMED_PUNCTUATION_RE = re.compile(r"(?:\?\.|!\.|\.\?|\.!|,\.|\.,|;\.|:\.)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")

# A case ending may be rough while the composed semantic roles remain valid.
COMPOSED_SURFACE_SOFT_ISSUES = frozenset({"governed_case_mismatch"})


@dataclass(frozen=True, slots=True)
class SurfaceCoherence:
    score: float
    hard_issues: tuple[str, ...]
    sentence_completeness: float
    agreement: float

    @property
    def learnable(self) -> bool:
        return not self.hard_issues and self.score >= 0.72


def evaluate_surface(text: str) -> SurfaceCoherence:
    """Measure readable sentence structure independently of source and topic."""

    stripped = text.strip()
    if not stripped:
        return SurfaceCoherence(0.0, ("empty",), 0.0, 0.0)

    issues: list[str] = []
    if stripped[0] in _INVALID_OPENING:
        issues.append("broken_opening")
    if _MALFORMED_PUNCTUATION_RE.search(stripped):
        issues.append("malformed_punctuation")
    if not _balanced_delimiters(stripped):
        issues.append("unbalanced_delimiters")

    tokens = tokenize(stripped)
    lexical = tuple(token for token in tokens if token[:1].isalnum())
    if not lexical:
        issues.append("no_lexical_content")
    if tokens and role(tokens[-1]) in {"preposition", "conjunction", "question"}:
        issues.append("unfinished_ending")

    connector_counts = Counter(
        token.casefold() for token in lexical if token.casefold() in _CONNECTORS
    )
    if len(lexical) <= 14 and any(count >= 2 for count in connector_counts.values()):
        issues.append("repeated_connector")

    # Read the generated sentence back into semantic roles.  Morphological
    # agreement alone cannot notice that an intransitive predicate received a
    # direct object (``поперхнулся мозг``), while reverse valency can.
    from debilgi.language.structured_utterance import analyze_structured_utterance

    structure = analyze_structured_utterance(stripped)
    for clause in structure.clauses:
        if clause.predicate_index is None:
            continue
        predicate = structure.tokens[clause.predicate_index].lemma
        direct_object_indices = tuple(
            index
            for index in clause.direct_object_indices
            if not _continues_prepositional_complement(structure, clause, index)
        )
        inferred_subject = False
        if (
            predicate.endswith(("ся", "сь"))
            and not clause.subject_indices
            and len(direct_object_indices) == 1
            and direct_object_indices[0] < clause.predicate_index
        ):
            possible_subject = structure.tokens[direct_object_indices[0]]
            subject_features = dict(possible_subject.grammatical_features)
            predicate_features = dict(
                structure.tokens[clause.predicate_index].grammatical_features
            )
            inferred_subject = (
                subject_features.get("case") == "nominative"
                or (
                    subject_features.get("case") == "accusative"
                    and subject_features.get("animacy") == "inanimate"
                )
            ) and all(
                not subject_features.get(feature)
                or not predicate_features.get(feature)
                or subject_features[feature] == predicate_features[feature]
                for feature in ("number", "gender")
            )
            if inferred_subject:
                direct_object_indices = ()
        present_roles = frozenset(
            role_name
            for indices, role_name in (
                (clause.subject_indices or ((-1,) if inferred_subject else ()), "subject"),
                (direct_object_indices, "direct_object"),
                (clause.indirect_object_indices, "indirect_object"),
                (clause.predicate_complement_indices, "predicate_complement"),
                (clause.action_complement_indices, "action_complement"),
            )
            if indices
        )
        if clause.action_complement_indices and direct_object_indices:
            last_action = clause.action_complement_indices[-1]
            if all(index > last_action for index in direct_object_indices):
                present_roles = present_roles - {"direct_object"}
        complement_roles = frozenset(item.role for item in clause.complements)
        valency = assess_predicate_valency(
            predicate,
            present_roles=present_roles,
            complement_roles=complement_roles,
            implicit_subject=clause.implicit_subject is not None,
            imperative=clause.clause_kind == "command",
        )
        if valency.unexpected_roles:
            issues.append("unexpected_valency_role")
        for index in range(clause.token_start, clause.token_end):
            object_token = structure.tokens[index]
            if (
                index != clause.predicate_index
                and
                object_token.part_of_speech == "participle"
                and object_token.lemma.casefold().replace("ё", "е")
                == predicate.casefold().replace("ё", "е")
            ):
                issues.append("predicate_participle_as_object")
        for action_index in clause.action_complement_indices:
            action = structure.tokens[action_index].lemma
            action_objects = tuple(
                index for index in direct_object_indices if index > action_index
            )
            action_valency = assess_predicate_valency(
                action,
                present_roles=frozenset(
                    {
                        "subject",
                        *("direct_object" for _ in action_objects),
                    }
                ),
                implicit_subject=True,
            )
            if action_valency.unexpected_roles or action_valency.missing_roles:
                issues.append("action_complement_valency_mismatch")
        for indices, role_name, default_case in (
            (direct_object_indices, "direct_object", "accusative"),
            (clause.indirect_object_indices, "indirect_object", "dative"),
        ):
            if not indices or role_name in valency.unexpected_roles:
                continue
            expected_case = governed_object_case(predicate, role_name, default_case)
            head = structure.tokens[indices[-1]]
            head_features = dict(head.grammatical_features)
            actual_case = head_features.get("case")
            surface_supports_expected_case = bool(
                expected_case
                and any(
                    reading.features().get("case") == expected_case
                    for reading in DEFAULT_MORPHOLOGY.parse_word(head.surface)
                )
            )
            syncretic_accusative = (
                expected_case == "accusative"
                and (
                    (
                        actual_case == "nominative"
                        and head_features.get("animacy") == "inanimate"
                    )
                    or (
                        actual_case == "genitive"
                        and head_features.get("animacy") == "animate"
                    )
                )
            )
            if (
                valency.profile_source == "lexical_profile"
                and actual_case
                and expected_case
                and actual_case != expected_case
                and not (clause.negated and expected_case == "accusative"
                         and actual_case == "genitive")
                and not syncretic_accusative
                and not surface_supports_expected_case
            ):
                issues.append("governed_case_mismatch")

    sentences = tuple(
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_RE.split(stripped)
        if sentence.strip()
    ) or (stripped,)
    normalized_sentences = tuple(
        " ".join(sentence.casefold().strip(" .!?…").split())
        for sentence in sentences
    )
    if len(set(normalized_sentences)) != len(normalized_sentences):
        issues.append("duplicate_sentence")
    completeness: list[float] = []
    agreement_scores: list[float] = []
    for sentence in sentences:
        words = tuple(token for token in tokenize(sentence) if token[:1].isalnum())
        if not words:
            completeness.append(0.0)
            continue
        readings = tuple(DEFAULT_MORPHOLOGY.best(word) for word in words)
        roles = tuple(role(word) for word in words)
        finite_predicate = any(
            reading is not None
            and (
                reading.part_of_speech == "predicate"
                or (
                    reading.part_of_speech == "verb"
                    and "INFN" not in reading.grammemes
                )
            )
            for reading in readings
        )
        has_infinitive = any(
            reading is not None and "INFN" in reading.grammemes for reading in readings
        )
        elliptical = (
            "—" in sentence
            or ":" in sentence
            or "question" in roles
            or (len(words) <= 3 and bool(set(roles) & _SAFE_SHORT_ROLES))
            or (len(words) <= 3 and "pronoun" in roles and "adverb" in roles)
        )
        complete = finite_predicate or elliptical
        completeness.append(1.0 if complete else 0.45 if len(words) >= 3 else 0.15)

        first = words[0].casefold().replace("ё", "е")
        if first in _CONTINUATION_MARKERS and not finite_predicate:
            issues.append("dangling_continuation")
        if len(words) == 1 and roles[0] in {"content", "adjective", "preposition"}:
            issues.append("nominal_fragment")
        if has_infinitive and not finite_predicate and first in _CONTINUATION_MARKERS:
            issues.append("infinitive_fragment")

        agreement_scores.append(DEFAULT_MORPHOLOGY.analyze_sentence(sentence).agreement_score)

    sentence_completeness = sum(completeness) / max(1, len(completeness))
    agreement = sum(agreement_scores) / max(1, len(agreement_scores))
    score = (
        float(stripped[-1] in _TERMINAL) * 0.20
        + float(stripped[0] not in _INVALID_OPENING) * 0.15
        + sentence_completeness * 0.40
        + agreement * 0.25
    )
    unique_issues = tuple(dict.fromkeys(issues))
    if unique_issues:
        score = min(score, 0.20)
    return SurfaceCoherence(
        max(0.0, min(1.0, score)),
        unique_issues,
        sentence_completeness,
        agreement,
    )


def _balanced_delimiters(text: str) -> bool:
    pairs = (("(", ")"), ("[", "]"), ("{", "}"), ("«", "»"))
    return all(text.count(opening) == text.count(closing) for opening, closing in pairs)


def _continues_prepositional_complement(structure, clause, token_index: int) -> bool:
    """Recover a complement head split off by an ambiguous adjective parse."""

    token = structure.tokens[token_index]
    actual_case = dict(token.grammatical_features).get("case")
    if actual_case is None:
        return False
    for complement in clause.complements:
        if not complement.object_indices:
            continue
        tail = max(complement.object_indices)
        if token_index != tail + 1:
            continue
        preposition = structure.tokens[complement.preposition_index].lemma
        expected_cases = DEFAULT_MORPHOLOGY.expected_cases(preposition)
        tail_case = dict(structure.tokens[tail].grammatical_features).get("case")
        if actual_case in expected_cases or (
            tail_case is not None and actual_case == tail_case
        ):
            return True
    return False
