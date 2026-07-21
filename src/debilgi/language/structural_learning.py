"""Sparse online features for learning response structure without an LLM.

The model deliberately learns abstract form: character shape, POS/dependency
sequences, clause frames and their interaction with the candidate source.  It
does not memorize whole prompts or answer strings, so feedback can transfer to
new words and new topics.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from debilgi.language.content import content_lemmas
from debilgi.language.structured_utterance import StructuredUtterance, analyze_structured_utterance

_ELONGATED_RE = re.compile(r"([а-яёa-z])\1{2,}", re.IGNORECASE)
_EMOJI_RE = re.compile(r"[^\w\s.,!?;:()\[\]{}'\"«»—–-]", re.UNICODE)


def structural_response_features(
    *,
    prompt: str,
    candidate_text: str,
    source: str,
    speech_act: str,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Extract bounded, reusable sparse features for a candidate pair."""

    prompt_structure = analyze_structured_utterance(prompt)
    candidate_structure = analyze_structured_utterance(candidate_text)
    features: dict[str, float] = {f"source:{source}": 1.0}
    if speech_act:
        features[f"speech_act:{speech_act}"] = 1.0
    _shape_features(features, "prompt", prompt)
    _shape_features(features, "candidate", candidate_text)
    _structure_features(features, "prompt", prompt_structure)
    _structure_features(features, "candidate", candidate_structure)

    prompt_kind = _main_clause_kind(prompt_structure)
    candidate_kind = _main_clause_kind(candidate_structure)
    features[f"cross:clause:{prompt_kind}>{candidate_kind}"] = 1.0
    features[f"cross:prompt_clause_source:{prompt_kind}>{source}"] = 1.0

    prompt_terms = set(content_lemmas(prompt))
    candidate_terms = set(content_lemmas(candidate_text))
    overlap = len(prompt_terms & candidate_terms) / max(1, len(prompt_terms | candidate_terms))
    features[f"cross:lemma_overlap:{_bucket(overlap, (0.0, 0.15, 0.35, 0.60))}"] = 1.0

    obligation = (metadata or {}).get("response_obligation")
    if isinstance(obligation, str) and obligation:
        features[f"cross:obligation_source:{obligation}>{source}"] = 1.0
        features[f"cross:obligation_clause:{obligation}>{candidate_kind}"] = 1.0
    move = (metadata or {}).get("response_move")
    if isinstance(move, str) and move:
        features[f"cross:move_source:{move}>{source}"] = 1.0
    return features


def learned_structural_bonus(
    features: Mapping[str, float],
    weights: Mapping[str, float],
    *,
    maximum_bonus: float = 0.12,
) -> float:
    """Return a bounded additive arbitration bonus."""

    raw = sum(float(weights.get(key, 0.0)) * value for key, value in features.items())
    return math.tanh(raw) * maximum_bonus


def _shape_features(features: dict[str, float], prefix: str, text: str) -> None:
    letters = sum(character.isalpha() for character in text)
    digits = sum(character.isdigit() for character in text)
    features[f"{prefix}:length:{_length_bucket(len(text))}"] = 1.0
    if digits:
        features[f"{prefix}:has_digit"] = 1.0
        ratio_bucket = _bucket(digits / max(1, len(text)), (0.0, 0.08, 0.25))
        features[f"{prefix}:digit_ratio:{ratio_bucket}"] = 1.0
    if _ELONGATED_RE.search(text):
        features[f"{prefix}:elongated_word"] = 1.0
    if _EMOJI_RE.search(text):
        features[f"{prefix}:has_symbol_or_emoji"] = 1.0
    if "?" in text:
        features[f"{prefix}:question_mark"] = 1.0
    if "!" in text:
        features[f"{prefix}:exclamation_mark"] = 1.0
    if letters >= 3 and text.upper() == text and text.lower() != text:
        features[f"{prefix}:all_caps"] = 1.0


def _structure_features(
    features: dict[str, float],
    prefix: str,
    structure: StructuredUtterance,
) -> None:
    parts = tuple(token.part_of_speech or "unknown" for token in structure.tokens)
    roles = tuple(token.semantic_role or "unknown" for token in structure.tokens)
    for part in dict.fromkeys(parts):
        features[f"{prefix}:pos:{part}"] = 1.0
    for left, right in zip(parts, parts[1:], strict=False):
        features[f"{prefix}:pos_bigram:{left}>{right}"] = 1.0
    for role in dict.fromkeys(roles):
        features[f"{prefix}:role:{role}"] = 1.0
    for left, right in zip(roles, roles[1:], strict=False):
        features[f"{prefix}:role_bigram:{left}>{right}"] = 1.0
    for clause in structure.clauses:
        features[f"{prefix}:clause:{clause.clause_kind}"] = 1.0
        if clause.negated:
            features[f"{prefix}:negated"] = 1.0
        if clause.elliptical:
            features[f"{prefix}:elliptical"] = 1.0
        if clause.question_target_index is not None:
            features[f"{prefix}:question_target"] = 1.0
        if clause.modal_lemmas:
            features[f"{prefix}:modal"] = 1.0
    for dependency in structure.dependencies:
        features[f"{prefix}:dependency:{dependency.relation}"] = 1.0
    for link in structure.clause_links:
        features[f"{prefix}:clause_link:{link.relation}"] = 1.0


def _main_clause_kind(structure: StructuredUtterance) -> str:
    return structure.clauses[0].clause_kind if structure.clauses else "empty"


def _length_bucket(length: int) -> str:
    if length <= 8:
        return "tiny"
    if length <= 24:
        return "short"
    if length <= 80:
        return "medium"
    return "long"


def _bucket(value: float, boundaries: tuple[float, ...]) -> str:
    return str(sum(value > boundary for boundary in boundaries))
