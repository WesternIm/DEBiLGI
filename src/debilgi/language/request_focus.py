"""Keep a request and its answer domain separate from conversation controls.

This layer selects observed meaning-bearing clauses, never an answer phrase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from debilgi.language.content import content_lemmas
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.productive_nlg import ClauseThought
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.language.tokenizer import tokenize

# Lexical speech operations and their arguments, rather than complete inputs.
_RESPONSE_OPERATIONS = frozenset({"ответить", "отвечать", "выбрать", "выбирать", "сказать"})
_DISCOURSE_ARGUMENTS = frozenset({"ответ", "тема", "одно", "один", "бинарный"})
_POLARITY = {"да": True, "нет": False}


@dataclass(frozen=True, slots=True)
class RequestFocus:
    turn_text: str
    request_text: str
    alternatives: tuple[str, ...] = ()
    reason: str = "current_turn"


def polarity_of(word: str) -> bool | None:
    return _POLARITY.get(word.casefold())


def supported_alternative(
    request: str, options: tuple[str, ...], thought: ClauseThought
) -> str | None:
    if thought.clause_kind != "statement":
        return None
    structure = analyze_structured_utterance(request)
    if options and all(word in _POLARITY for word in options):
        if len(structure.clauses) != 1 or structure.clauses[0].clause_kind != "question":
            return None
        clause = structure.clauses[0]
        if clause.coordinations or clause.complements:
            return None
        if clause.modal_lemmas:
            from debilgi.language.semantic_graph import parse_proposition, proposition_key

            query = parse_proposition(request, allow_question=True)
            if query is None or query.negated or clause.question_target_index is not None:
                return None
            if proposition_key(query, signed=False) != proposition_key(thought, signed=False):
                return None
            return next(
                (option for option in options if polarity_of(option) == (not thought.negated)), None
            )
        if clause.predicate_index is None or clause.question_target_index is not None:
            return None
        if structure.tokens[clause.predicate_index].lemma != thought.predicate.lemma:
            return None
        if dict(thought.predicate.features).get("tense") != dict(
            structure.tokens[clause.predicate_index].grammatical_features
        ).get("tense"):
            return None
        # A negative question makes a bare confirmation ambiguous in Russian.
        if clause.negated:
            return None
        roles = {dep.role: dep.argument.head.lemma for dep in thought.dependencies}
        flip = {"я": "ты", "ты": "я", "мы": "вы", "вы": "мы"}
        for role, indices in (
            ("subject", clause.subject_indices),
            ("direct_object", clause.direct_object_indices),
            ("indirect_object", clause.indirect_object_indices),
            ("action_complement", clause.action_complement_indices),
            ("predicate_complement", clause.predicate_complement_indices),
        ):
            if len(indices) > 1:
                return None
            for index in indices:
                lemma = structure.tokens[index].lemma
                if roles.get(role) != flip.get(lemma, lemma):
                    return None
        supplied = {
            clause.predicate_index,
            *clause.subject_indices,
            *clause.direct_object_indices,
            *clause.indirect_object_indices,
            *clause.action_complement_indices,
            *clause.predicate_complement_indices,
        }
        if any(
            token.index not in supplied
            and token.surface[:1].isalnum()
            and token.semantic_role not in {"function_word", "pragmatic_marker"}
            for token in structure.tokens
        ):
            return None
        return next(
            (option for option in options if polarity_of(option) == (not thought.negated)), None
        )
    heads = {dep.argument.head.lemma for dep in thought.dependencies if dep.role != "subject"}
    matching = tuple(option for option in options if DEFAULT_MORPHOLOGY.lemma(option) in heads)
    return matching[0] if len(matching) == 1 else None


def response_alternatives(text: str) -> tuple[str, ...]:
    tokens = tokenize(text)
    words = tuple(token for token in tokens if token[:1].isalnum())
    if set(words) == set(_POLARITY):
        return tuple(dict.fromkeys(word for word in words if word in _POLARITY))
    pairs = tuple(
        (tokens[index - 1], tokens[index + 1])
        for index, token in enumerate(tokens)
        if token in {"или", "либо"}
        and 0 < index < len(tokens) - 1
        and tokens[index - 1][:1].isalnum()
        and tokens[index + 1][:1].isalnum()
    )
    return tuple(dict.fromkeys(word for pair in pairs for word in pair)) if len(pairs) == 1 else ()


def question_focus(text: str) -> str | None:
    structure = analyze_structured_utterance(text)
    embedded = {link.child_clause_index for link in structure.clause_links}
    clauses = tuple(
        clause
        for clause in structure.clauses
        if clause.clause_kind == "question"
        and clause.clause_index not in embedded
        and (
            clause.predicate_index is not None
            or content_lemmas(
                tuple(
                    token.surface
                    for token in structure.tokens[clause.token_start : clause.token_end]
                )
            )
        )
    )
    if not clauses:
        return None
    clause = clauses[-1]
    if len(structure.clauses) == 1 or structure.clause_links:
        return text
    end = clause.token_end
    if end < len(structure.tokens) and structure.tokens[end].surface in {"?", "!", "."}:
        end += 1
    rendered = " ".join(token.surface for token in structure.tokens[clause.token_start : end])
    return re.sub(r"\s+([,.!?;:])", r"\1", rendered)


def is_response_control(text: str) -> bool:
    structure = analyze_structured_utterance(text)
    commands = any(clause.clause_kind == "command" for clause in structure.clauses)
    alternatives = response_alternatives(text)
    if alternatives and (all(word in _POLARITY for word in alternatives) or commands):
        return True
    if not commands:
        return False
    clause = structure.clauses[-1]
    if clause.clause_kind != "command":
        return False
    tokens = structure.tokens[clause.token_start : clause.token_end]
    # A detached address (including an adjective) isn't a new requested topic.
    terms = set(
        content_lemmas(
            tuple(
                token.surface
                for token in tokens
                if not (
                    token.part_of_speech == "adverb"
                    or (
                    token.part_of_speech == "adjective"
                    and token.index > clause.token_start
                    and structure.tokens[token.index - 1].surface == ","
                    )
                )
            )
        )
    )
    verbs = {token.lemma for token in tokens if token.part_of_speech == "verb"}
    if terms - verbs - _DISCOURSE_ARGUMENTS:
        return False
    return bool(verbs & _RESPONSE_OPERATIONS or terms & _DISCOURSE_ARGUMENTS)


def focus_request(text: str, turns: tuple[str, ...]) -> RequestFocus:
    alternatives = response_alternatives(text)
    control = is_response_control(text)
    if not control and (question := question_focus(text)):
        return RequestFocus(text, question, alternatives, "current_question")
    if control:
        for turn in reversed(turns):
            speaker, separator, previous = turn.partition(": ")
            if not separator or speaker != "user":
                continue
            if is_response_control(previous):
                continue
            if question := question_focus(previous):
                return RequestFocus(text, question, alternatives, "answer_control_keeps_request")
            if not is_response_control(previous) and content_lemmas(previous):
                break
    return RequestFocus(text, text, alternatives if control else ())
