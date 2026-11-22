"""Keep a request and its answer domain separate from conversation controls.

This layer selects observed meaning-bearing clauses, never an answer phrase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from debilgi.config import thought_content_enabled
from debilgi.language.content import content_lemmas
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.productive_nlg import ClauseThought
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.language.tokenizer import tokenize

# Lexical speech operations and their arguments, rather than complete inputs.
_RESPONSE_OPERATIONS = frozenset(
    {"ответить", "отвечать", "выбрать", "выбирать", "сказать", "предположить", "прикинуть"}
)
_DISCOURSE_ARGUMENTS = frozenset({"ответ", "тема", "одно", "один", "бинарный", "вопрос"})
# Operations of making something up or telling something freely (perfective base lemmas; the
# imperfective partner of each is recognised by the aspect pair, not listed).
_INVENTION_OPERATIONS = frozenset(
    {"придумать", "выдумать", "сочинить", "пошутить", "пофантазировать", "рассказать"}
)
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
    # A following alternative supplies a domain for the preceding question.
    if len(clauses) > 1 and response_alternatives(text):
        return text
    clause = clauses[-1]
    if len(structure.clauses) == 1 or structure.clause_links:
        return text
    end = clause.token_end
    if end < len(structure.tokens) and structure.tokens[end].surface in {"?", "!", "."}:
        end += 1
    rendered = " ".join(token.surface for token in structure.tokens[clause.token_start : end])
    return re.sub(r"\s+([,.!?;:])", r"\1", rendered)


def is_fragment_reaction(text: str) -> bool:
    """A reaction without a predicate, an interrogative word or a choice («серьёзно??», «правда?»).

    It can carry a question mark, but it does not state a request of its own: it reacts to the
    turn before it, so an explicit command after it («а ты предположи») still means that turn.
    """
    structure = analyze_structured_utterance(text)
    if response_alternatives(text):
        return False
    return not any(c.predicate_index is not None for c in structure.clauses) and not any(
        t.semantic_role == "question_target" for t in structure.tokens
    )


def is_response_control(text: str) -> bool:
    structure = analyze_structured_utterance(text)
    commands = any(clause.clause_kind == "command" for clause in structure.clauses)
    alternatives = response_alternatives(text)
    if alternatives and (all(word in _POLARITY for word in alternatives) or commands):
        return True
    if not commands:
        verbs = {t.lemma for t in structure.tokens if t.part_of_speech == "verb"}
        # Pronouns (personal, possessive, interrogative) carry no new topic; they are
        # recognised by morphology, as are the discourse nouns about the exchange itself.
        pronouns = {
            t.lemma
            for t in structure.tokens
            if t.part_of_speech == "pronoun" or "Apro" in t.grammemes
        }
        content = set(content_lemmas(text)) - verbs - pronouns - _DISCOURSE_ARGUMENTS
        if verbs & _RESPONSE_OPERATIONS and not content:
            return True
        terms = set(content_lemmas(text)) - pronouns
        return bool(terms & {"ответ", "вопрос"}) and not (terms - _DISCOURSE_ARGUMENTS)
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
    if terms - verbs - _DISCOURSE_ARGUMENTS - {"я", "ты", "мы", "вы"}:
        return False
    return bool(verbs & _RESPONSE_OPERATIONS or terms & _DISCOURSE_ARGUMENTS)


def is_invention_request(text: str, *, bare: bool = True) -> bool:
    """A command to make something up («пошути», «придумай что-нибудь», «сочини стих»).

    Licenses play the way a request to answer licenses a guess: the verb is an invention
    operation (either aspect) and the request carries no topic of its own — only pronouns,
    qualities, adverbs or the product it names as a direct object («стих»). A question, or a
    topic complement («расскажи про кошку»), is a request for information, not for play.
    ``bare=False`` also demands something besides the verb («расскажи» alone may only ask to go
    on with the answer of the question before it).
    """
    from debilgi.language.lexical_matching import aspect_pair

    structure = analyze_structured_utterance(text)
    if any(c.clause_kind in {"question", "statement"} for c in structure.clauses):
        return False
    command = next((c for c in structure.clauses if c.clause_kind == "command"
                    and c.predicate_index is not None), None)
    if command is None:
        return False
    tokens = structure.tokens[command.token_start:command.token_end]
    verbs = [t for t in tokens if t.part_of_speech == "verb"]
    # «давай пошути»: the hortative is the predicate, the operation is the verb it governs.
    if bare is False and len([t for t in tokens if t.part_of_speech not in {None}]) < 2:
        return False
    if len(verbs) > 2 or not any(
        v.lemma == base or aspect_pair(v.lemma, base)
        for v in verbs for base in _INVENTION_OPERATIONS
    ):
        return False
    return all(
        token.part_of_speech == "verb"
        or token.part_of_speech
        in {"pronoun", "adjective", "adverb", "particle", "preposition", None}
        and not token.semantic_role.startswith("complement_link")
        or token.part_of_speech == "noun" and token.semantic_role == "direct_object"
        or token.semantic_role in {"punctuation", "pragmatic_marker", "vocative"}
        for token in tokens
    )


def focus_request(
    text: str, turns: tuple[str, ...], *, open_request: str | None = None
) -> RequestFocus:
    """One continuation rule for controls, omitted domains and clarification fills."""
    alternatives = response_alternatives(text)
    control = is_response_control(text)
    question = question_focus(text)
    previous = open_request
    if previous is None:
        for turn in reversed(turns):
            speaker, separator, candidate = turn.partition(": ")
            if not separator or speaker != "user" or is_response_control(candidate):
                continue
            if question_focus(candidate):
                if thought_content_enabled() and is_fragment_reaction(candidate):
                    continue  # a reaction is not the open request (stage 4.32)
                previous = candidate
                break
            if content_lemmas(candidate):
                break
    prior_options = response_alternatives(previous) if previous else ()
    terms = set(content_lemmas(text))
    prior_terms = set(content_lemmas(previous or ""))
    elliptical = bool(
        previous
        and prior_options
        and (
            question
            and terms <= prior_terms
            or alternatives
            and set(alternatives) == set(prior_options)
            or not question
            and terms
            and terms <= {"оба", "весь"}
        )
    )
    if previous and (control or elliptical):
        return RequestFocus(
            text,
            previous,
            alternatives or prior_options,
            "answer_control_keeps_request" if control else "open_request_continuation",
        )
    if not control and question:
        return RequestFocus(text, question, alternatives, "current_question")
    return RequestFocus(text, text, alternatives if control else ())
