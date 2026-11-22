"""Build a fresh stance by transforming the current proposition.

The module contains no reply strings.  It reads grammatical roles from the
current turn, revoices its participants, and lets persistent mood/emotion plus
the expressed personality weight alignment against resistance.  The result is
a new proposition assembled from lexical material in the turn rather than a
remembered answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.language.content import content_lemmas
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.language.word_relations import grammatical_tokens
from debilgi.personality.expression import PersonalityExpression


@dataclass(frozen=True, slots=True)
class AffectiveStance:
    text: str
    operation: str
    aligned: bool
    weight: float
    predicate: str
    topics: tuple[str, ...]


def reaction_alignment(*, form=None, affect_features=(), personality=None) -> float:
    """The same stance policy applies without an understood proposition.

    Habitual coarse punctuation is a manner; a change in that manner is an
    additional resistance cue. Neither licenses a factual assertion.
    """
    value = _alignment_weight(affect_features, personality)
    if form is not None:
        value -= 0.16 * ("coarse" in form.tone_change)
        value -= 0.08 * ("caps" in form.tone_change)
    return max(0.08, min(0.92, value))


def affective_stances(
    text: str,
    *,
    affect_features: tuple[tuple[str, float], ...] = (),
    personality: PersonalityExpression | None = None,
) -> tuple[AffectiveStance, ...]:
    """Return independently weighted positions for a simple attitude statement.

    Russian attitude predicates such as ``мне плевать на ...`` use a dative
    experiencer and an infinitive.  They were previously reduced to a bag of
    topic words, after which an unrelated corpus fragment could win.  Here the
    experiencer, predicate, polarity and complements remain explicit.
    """

    structure = analyze_structured_utterance(text)
    if len(structure.clauses) != 1 or structure.clause_links:
        return ()
    clause = structure.clauses[0]
    if (
        clause.clause_kind in {"question", "command"}
        or clause.predicate_index is None
        or not clause.indirect_object_indices
    ):
        return ()
    predicate = structure.tokens[clause.predicate_index]
    if predicate.part_of_speech != "verb" or "INFN" not in predicate.grammemes:
        return ()
    experiencer = next(
        (
            structure.tokens[index]
            for index in clause.indirect_object_indices
            if structure.tokens[index].lemma in {"я", "ты", "мы", "вы"}
        ),
        None,
    )
    if experiencer is None:
        return ()

    revoiced = grammatical_tokens(text, human_perspective=True)
    if len(revoiced) != len(structure.tokens):
        return ()
    complement_indices: set[int] = set()
    for complement in clause.complements:
        if not complement.object_indices:
            continue
        beginning = complement.preposition_index
        ending = max(complement.object_indices)
        complement_indices.update(range(beginning, ending + 1))
    if not complement_indices:
        return ()
    complement = tuple(revoiced[index] for index in sorted(complement_indices))
    if not any(token[:1].isalnum() for token in complement):
        return ()

    singular = dict(experiencer.grammatical_features).get("number") != "plural"
    first_person = "я" if singular else "мы"
    experiencer_surface = DEFAULT_MORPHOLOGY.inflect(
        first_person,
        case="dative",
        number="singular" if singular else "plural",
    ) or ("мне" if singular else "нам")
    predicate_surface = revoiced[clause.predicate_index]
    aligned_weight = _alignment_weight(affect_features, personality)
    topic_terms = tuple(
        dict.fromkeys(
            lemma
            for index in sorted(complement_indices)
            for lemma in content_lemmas((revoiced[index],))
        )
    )

    variants = (
        (
            True,
            aligned_weight,
            (
                *complement,
                experiencer_surface,
                *(('не',) if clause.negated else ()),
                "тоже",
                predicate_surface,
            ),
        ),
        (
            False,
            1.0 - aligned_weight,
            (
                *complement,
                experiencer_surface,
                *(("не",) if not clause.negated else ()),
                predicate_surface,
            ),
        ),
    )
    result: list[AffectiveStance] = []
    seen: set[str] = set()
    for aligned, weight, tokens in variants:
        surface = _detokenize(tokens)
        key = " ".join(surface.casefold().split())
        if not surface or key in seen:
            continue
        seen.add(key)
        result.append(
            AffectiveStance(
                surface,
                "align_attitude" if aligned else "resist_attitude",
                aligned,
                weight,
                predicate.lemma,
                topic_terms,
            )
        )
    return tuple(sorted(result, key=lambda item: (item.weight, item.text), reverse=True))


def _alignment_weight(
    affect_features: tuple[tuple[str, float], ...],
    personality: PersonalityExpression | None,
) -> float:
    affect = dict(affect_features)

    def blended(name: str) -> float:
        return max(
            -1.0,
            min(
                1.0,
                affect.get(f"emotion_{name}", 0.0) * 0.72
                + affect.get(f"mood_{name}", 0.0) * 0.28,
            ),
        )

    affiliation = blended("affiliation")
    valence = blended("valence")
    dominance = blended("dominance")
    warmth = personality.warmth if personality is not None else 0.5
    suspicion = personality.suspicion if personality is not None else 0.5
    assertiveness = personality.assertiveness if personality is not None else 0.5
    value = (
        0.50
        + affiliation * 0.25
        + valence * 0.16
        - dominance * 0.08
        + (warmth - 0.5) * 0.14
        - (suspicion - 0.5) * 0.18
        - (assertiveness - 0.5) * 0.10
    )
    return max(0.08, min(0.92, value))


def _detokenize(tokens: tuple[str, ...]) -> str:
    body = ""
    for token in tokens:
        if not token:
            continue
        if not body:
            body = token
        elif token in {",", ".", "!", "?", ":", ";", "…"}:
            body += token
        else:
            body += " " + token
    body = body.strip(" .!?…")
    return body[:1].upper() + body[1:] + "." if body else ""
