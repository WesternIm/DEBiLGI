"""Evaluative and definitional readings of one clause (stage 4.37, opt-in, read only).

``parse_proposition(..., evaluation=True, definition=True)`` asks this module first. It reads
what the default reader drops or refuses, from the structure the analyzer already built (no
word lists; only parts of speech, grammemes and agreement):

* a short adjective that the dictionary ranks as a noun («Гитлер был прав» — «право»);
* an evaluative adjective in front of a predicate noun («был плохим человеком»);
* a definition «X — это Y», «X это Y», «X — Y» with the dependents of Y kept as tree links
  (``head="predicate_complement"``): prepositional complements, genitive chains and a relative
  clause («который начал войну»);
* a comparison «кошки лучше собак» / «лучше, чем собаки».

The result is an ordinary ``ClauseThought`` (a copular «быть»). Nothing here writes memory,
changes a plan or an answer: the thought is produced only for a caller that asks for it.
"""

from __future__ import annotations

from dataclasses import replace

from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)

_DASHES = {"—", "–", "-", "−"}
_SOFT = {",", ".", "!", "?", "…"}
_QUESTION_WORDS = {"кто", "что", "какой", "где", "когда", "почему", "зачем", "как", "сколько"}


def _features(token) -> dict[str, str]:
    return dict(token.grammatical_features)


def _is_punct(token) -> bool:
    return token.part_of_speech is None and not token.surface[:1].isalnum()


def _is_verb(token) -> bool:
    return token.part_of_speech == "verb" and "VERB" in token.grammemes


def _is_noun(token) -> bool:
    return token.part_of_speech in {"noun", "pronoun"} and token.lemma not in _QUESTION_WORDS


def _is_adjective(token) -> bool:
    return token.part_of_speech in {"adjective", "participle"}


def _lexeme(token, perspective, possessive_deixis):
    from debilgi.language.semantic_graph import _lexeme as build

    return build(token, perspective, possessive_deixis)


def _agrees(subject_token, reading) -> bool:
    subject = _features(subject_token)
    number = subject.get("number")
    if number and reading.number and number != reading.number:
        return False
    gender = subject.get("gender")
    return not (gender and reading.gender and gender != reading.gender)


def _short_adjective(token, subject_token):
    """The short-adjective reading of a word the dictionary ranks as a noun («прав»)."""
    from debilgi.language.morphology import DEFAULT_MORPHOLOGY

    if token.part_of_speech != "noun" or _features(token).get("case") in {"nominative", None}:
        return None
    options = [
        r for r in DEFAULT_MORPHOLOGY.parse_word(token.surface)
        if r.part_of_speech == "adjective" and "ADJS" in r.grammemes
        and _agrees(subject_token, r)
    ]
    return max(options, key=lambda r: r.score) if options else None


def _adjective_lexeme(token, perspective, possessive_deixis, *, reading=None, degree=None):
    lexeme = _lexeme(token, perspective, possessive_deixis)
    if reading is not None:
        lexeme = replace(
            lexeme, lemma=reading.lemma, part_of_speech="adjective",
            features=tuple(
                (k, v) for k, v in reading.features().items() if k in {"gender", "number"}
            ),
        )
    if degree:
        lexeme = replace(lexeme, features=(*lexeme.features, ("degree", degree)))
    return lexeme


def _noun_phrase(tokens, i, perspective, possessive_deixis, *, post: bool):
    """Adjectives, one head noun and (``post``) genitive nouns after it → (argument, next)."""
    modifiers = []
    while i < len(tokens) and _is_adjective(tokens[i]) and tokens[i].lemma != "который":
        if modifiers and tokens[i].index != modifiers[-1].index + 1:
            return None
        modifiers.append(tokens[i])
        i += 1
    if i >= len(tokens) or not _is_noun(tokens[i]):
        return None
    head = tokens[i]
    if modifiers and head.index != modifiers[-1].index + 1:
        return None
    i += 1
    chain = []
    while post and i < len(tokens) and _is_noun(tokens[i]) \
            and tokens[i].index == tokens[i - 1].index + 1 \
            and _features(tokens[i]).get("case") == "genitive":
        chain.append(tokens[i])
        i += 1
    argument = ConceptArgument(
        _lexeme(head, perspective, possessive_deixis),
        tuple(_lexeme(t, perspective, possessive_deixis) for t in modifiers),
        grammatical_case=_features(head).get("case"),
        postmodifiers=tuple(_lexeme(t, perspective, possessive_deixis) for t in chain),
    )
    return argument, i


def _dependents(tokens, i, perspective, possessive_deixis, complements):
    """What stands after the predicate noun: prepositional phrases, a relative clause."""
    found = []
    while i < len(tokens):
        token = tokens[i]
        if token.part_of_speech == "preposition":
            phrase = _noun_phrase(tokens, i + 1, perspective, possessive_deixis, post=True)
            if phrase is None:
                return None
            argument, i = phrase
            role = "location" if complements.get(token.index, "").startswith("location") \
                else "topic"
            found.append(SemanticDependency(
                role, replace(argument, preposition=token.lemma), head="predicate_complement"
            ))
        elif token.lemma == "который" and i + 1 < len(tokens) and _is_verb(tokens[i + 1]):
            verb = tokens[i + 1]
            found.append(SemanticDependency(
                "relative_action", ConceptArgument(_lexeme(verb, perspective, possessive_deixis)),
                head="predicate_complement",
            ))
            i += 2
            if i < len(tokens):
                phrase = _noun_phrase(tokens, i, perspective, possessive_deixis, post=True)
                if phrase is None:
                    return None
                argument, i = phrase
                found.append(SemanticDependency(
                    "direct_object", argument, head="relative_action"
                ))
        else:
            return None
    return found


def read_evaluative(
    structure,
    *,
    perspective: str,
    possessive_deixis: bool,
    evaluation: bool,
    definition: bool,
    allow_question: bool,
    fallback_subject: ConceptArgument | None = None,
) -> ClauseThought | None:
    tokens_all = structure.tokens
    clause = structure.clauses[0]
    dashed = any(t.surface in _DASHES for t in tokens_all)
    commas = [t.index for t in tokens_all if t.surface == ","]
    if any(_is_punct(t) and t.surface not in _SOFT and t.surface not in _DASHES
           for t in tokens_all):
        return None
    tokens = [t for t in tokens_all if not _is_punct(t)]
    if not tokens or not any(t.index > 0 for t in tokens):
        return None
    if tokens_all[0].surface in _DASHES:
        return None  # a dialogue line, not a definition
    if any(t.semantic_role == "question_target" or t.lemma in _QUESTION_WORDS for t in tokens):
        return None
    complements = {
        c.preposition_index: c.role for c in clause.complements
    } if len(structure.clauses) == 1 else {}

    i = 0
    subject_token = None
    subject_arg = None
    phrase = _noun_phrase(tokens, 0, perspective, possessive_deixis, post=False)
    if phrase is not None and not (tokens[0].lemma == "этот" and tokens[0].part_of_speech
                                   == "adjective" and phrase[1] == 1):
        subject_arg, i = phrase
        subject_token = tokens[i - 1]
    elif fallback_subject is not None:
        subject_arg = fallback_subject
    else:
        return None
    if subject_token is not None and subject_token.lemma in {"это", "этот"} and not dashed:
        return None

    tense = ("tense", "present")
    marker = dashed
    negated = False
    while i < len(tokens):
        token = tokens[i]
        if token.lemma == "быть" and _is_verb(token):
            marker = True
            tense = ("tense", _features(token).get("tense", "present"))
        elif token.lemma == "этот" and token.part_of_speech in {"adjective", "pronoun"} \
                and "Anph" in token.grammemes and subject_token is not None:
            marker = True
        elif token.lemma == "не":
            negated = True
        else:
            break
        i += 1
    if i >= len(tokens):
        return None
    relative = any(t.lemma == "который" for t in tokens[i:])
    if not relative and any(_is_verb(t) for t in tokens[i:]):
        return None
    pronoun_subject = subject_token is not None and subject_token.part_of_speech == "pronoun"
    predicate = tokens[i]
    if subject_token is not None and any(
        subject_token.index < c < predicate.index for c in commas
    ) and predicate.lemma != "чем":
        return None

    def finish(dependencies, **kw):
        dependencies = [SemanticDependency("subject", subject_arg), *dependencies]
        lexeme = ConceptLexeme("быть", "быть", "verb", (tense,))
        question = clause.clause_kind == "question" or any(t.surface == "?" for t in tokens_all)
        if question and not allow_question:
            return None
        return ClauseThought(
            lexeme, tuple(dependencies), negated=negated, confidence=0.86,
            predicate_class="copular",
            semantic_time=clause.semantic_time if len(structure.clauses) == 1
            else replace(clause.semantic_time),
            parser_version=structure.parser_version,
            source_event_id=structure.source_event_id, **kw,
        )

    def complement(lexeme_or_arg, case="nominative"):
        argument = lexeme_or_arg if isinstance(lexeme_or_arg, ConceptArgument) else \
            ConceptArgument(lexeme_or_arg, grammatical_case=case)
        return SemanticDependency("predicate_complement", argument)

    # comparison: «кошки лучше собак», «лучше, чем собаки»
    if evaluation and _is_adjective(predicate) and "COMP" in predicate.grammemes \
            and (subject_token is not None or fallback_subject is not None):
        deps = [complement(_adjective_lexeme(predicate, perspective, possessive_deixis,
                                              degree="comparative"))]
        rest = i + 1
        if rest < len(tokens) and tokens[rest].lemma == "чем":
            standard = _noun_phrase(tokens, rest + 1, perspective, possessive_deixis, post=False)
            marker_word = "чем"
        elif rest < len(tokens):
            standard = _noun_phrase(tokens, rest, perspective, possessive_deixis, post=False)
            marker_word = None
        else:
            standard = marker_word = None
        if rest < len(tokens):
            if standard is None or standard[1] != len(tokens):
                return None
            deps.append(SemanticDependency(
                "comparison", replace(standard[0], preposition=marker_word),
                head="predicate_complement",
            ))
        return finish(deps)

    if not (marker or pronoun_subject or evaluation):
        return None

    # a single adjective: «прав», «неправ», «был плохим», «ты уверен»
    if evaluation and i == len(tokens) - 1 and subject_token is not None:
        if _is_adjective(predicate) and "COMP" not in predicate.grammemes:
            quality = _adjective_lexeme(predicate, perspective, possessive_deixis)
            return finish([complement(quality)])
        reading = _short_adjective(predicate, subject_token)
        if reading is not None and (marker or pronoun_subject):
            return finish([complement(_adjective_lexeme(
                predicate, perspective, possessive_deixis, reading=reading))])

    # a predicate noun with what hangs on it
    phrase = _noun_phrase(tokens, i, perspective, possessive_deixis, post=True)
    if phrase is None:
        return None
    argument, j = phrase
    evaluative = bool(argument.modifiers) and evaluation
    if not (definition or evaluative) or not (marker or (pronoun_subject and evaluative)):
        return None
    if argument.grammatical_case not in {"nominative", "instrumental", "accusative", None}:
        return None
    if argument.head.lemma in {"это", "этот"}:
        return None
    dependents = _dependents(tokens, j, perspective, possessive_deixis, complements)
    if dependents is None:
        return None
    if dependents and not definition:
        return None
    return finish([complement(argument), *dependents])


def evaluative_qualities(
    thought: ClauseThought,
) -> tuple[tuple[str, ConceptLexeme, ConceptLexeme | None], ...]:
    """(subject lemma, quality, standard of comparison) in a thought read in evaluation mode.

    The quality is the adjective of a copular thought («прав», «плохой» in «плохой человек»);
    the standard is the compared object of «лучше собак». A thought with no quality gives ().
    """
    if thought.predicate_class != "copular":
        return ()
    subject = next((d.argument for d in thought.dependencies if d.role == "subject"), None)
    complement = next(
        (d.argument for d in thought.dependencies if d.role == "predicate_complement"), None
    )
    if subject is None or complement is None:
        return ()
    standard = next(
        (d.argument.head for d in thought.dependencies if d.role == "comparison"), None
    )
    qualities = []
    if complement.head.part_of_speech in {"adjective", "participle"}:
        qualities.append(complement.head)
    qualities.extend(m for m in complement.modifiers if m.part_of_speech == "adjective")
    return tuple((subject.head.lemma, quality, standard) for quality in qualities)
