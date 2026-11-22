"""Human propositions and scoped implications, independent of reply wording.

Only linguistic operators are built in. Predicates, states and consequences
come from observed messages. Forward chaining requires every premise; it never
uses missing evidence as negation or treats a capability as an actual action.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import lru_cache

from debilgi.language.dialogue_obligations import thought_from_data
from debilgi.language.morphology import (
    DEFAULT_MORPHOLOGY,
    deictic_counterpart,
    deictic_person,
    person_cell,
)
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
    identity_graph,
)
from debilgi.language.structured_utterance import (
    StructuredUtterance,
    TokenInterpretation,
    analyze_structured_utterance,
)
from debilgi.language.tokenizer import tokenize

_UNSUPPORTED = {
    "бы",
    "либо",
    "или",
    "возможно",
    "наверное",
    "якобы",
    "вероятно",
    "кажется",
    "только",
    "вряд",
    "почти",
    "едва",
    "разве",
    "если",
    "когда",
}
_RELATIVE = {
    "сейчас",
    "сегодня",
    "вчера",
    "завтра",
    "позавчера",
    "послезавтра",
    "тут",
    "там",
    "здесь",
}
_MODALS = {"мочь", "уметь", "хотеть"}
_THIRD_PERSON = {"он", "она", "оно", "они"}


@dataclass(frozen=True, slots=True)
class MeaningRecord:
    meaning_id: str
    source_event_id: str
    text: str
    kind: str
    premises: tuple[ClauseThought, ...]
    conclusions: tuple[ClauseThought, ...]
    claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MeaningProof:
    thought: ClauseThought
    source_event_ids: tuple[str, ...]
    rule_ids: tuple[str, ...] = ()
    premise_keys: tuple[tuple, ...] = ()
    claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MeaningQuery:
    kind: str
    proposition: ClauseThought | None
    assumption: ClauseThought | None = None


@dataclass(frozen=True, slots=True)
class MeaningGoal:
    action: ClauseThought
    evidence_id: str
    weight: float = 1.0


def spread_activation(seeds, edges, *, decay=0.78, threshold=0.025, max_depth=3):
    """Bounded synchronous currents over memory edges; never revisit a node.

    Incoming currents sum, outgoing currents decay at every hop. Cycles cannot
    manufacture evidence. The caller supplies conjunctive rule gates as edges.
    This operation does not materialize inferred claims or training examples.
    """
    frontier = dict(seeds)
    activated = dict(seeds)
    visited = set(seeds)
    for _ in range(max_depth):
        following = {}
        for source, target, weight in edges:
            if source in frontier and target not in visited:
                following[target] = following.get(target, 0.0) + frontier[source] * weight * decay
        frontier = {k: min(1.0, v) for k, v in following.items() if v >= threshold}
        if not frontier:
            break
        activated.update(frontier)
        visited.update(frontier)
    return activated


def _lexeme(
    token: TokenInterpretation, perspective: str, possessive_deixis: bool = False
) -> ConceptLexeme:
    lemma = token.lemma
    surface = token.corrected_form or token.surface
    features = dict(token.grammatical_features)
    if perspective == "user":
        counterpart = deictic_counterpart(
            lemma, possessives=possessive_deixis, inclusive_possessives=False
        )
        if counterpart:
            lemma = surface = counterpart
            cell = person_cell(counterpart)
            if cell and cell[2] == "personal":
                features["person"] = cell[0]
    return ConceptLexeme(
        lemma,
        surface,
        token.part_of_speech,
        tuple(features.items()),
        token.entity_kind,
        token.participant.ref if token.participant else None,
    )


def _toward_speaker(personal: str, perspective: str) -> str:
    if perspective != "user":
        return personal
    return deictic_counterpart(personal, possessives=False) or personal


def subject_of(thought: ClauseThought) -> ConceptArgument:
    return next(dep.argument for dep in thought.dependencies if dep.role == "subject")


def with_subject(thought: ClauseThought, subject: ConceptArgument) -> ClauseThought:
    subject = replace(subject, grammatical_case="nominative")
    return replace(
        thought,
        dependencies=tuple(
            replace(dep, argument=subject) if dep.role == "subject" else dep
            for dep in thought.dependencies
        ),
    )


def variable_subject() -> ConceptArgument:
    return ConceptArgument(
        ConceptLexeme("@subject", "@subject", "pronoun", entity_kind="semantic_variable"),
        grammatical_case="nominative",
    )


@lru_cache(maxsize=2048)
def parse_proposition(
    text: str | StructuredUtterance,
    *,
    perspective: str = "user",
    fallback_subject: ConceptArgument | None = None,
    allow_question: bool = False,
    future_auxiliary: bool = False,
    possessive_deixis: bool = False,
    irrealis: bool = False,
    tree: bool = False,
    evaluation: bool = False,
    definition: bool = False,
) -> ClauseThought | None:
    """Parse a complete simple proposition, including zero copula and modality.

    ``future_auxiliary`` reads «буду/будешь + infinitive» as the infinitive's
    future event. It is opt-in so that existing daytime readers are unchanged.
    ``possessive_deixis`` also turns the possessives of the speaker into those of the
    addressee («мой»→«твой») when a clause is retold; plural possessives stay (inclusive
    «наш» needs a context signal). Fact keys use the default (off). ``irrealis`` accepts the
    particle «бы» and marks the predicate ``mood=conditional`` (the 4.22 mood feature) instead
    of refusing the clause; only the retelling of the agent's own limit uses it.
    ``tree`` (stage 4.24, opt-in) keeps the attachments the default reading flattens: «как X»
    (X not in the asked slot) becomes a ``comparison`` dependency, and the arguments that stand
    behind the infinitive of a modal get ``head="action_complement"`` (their governor is the
    embedded verb, not the modal). Facts and every other reader keep the default.
    ``evaluation`` / ``definition`` (stage 4.37, opt-in, only with ``DEBILGI_THOUGHT_CONTENT``)
    first try ``evaluative_reading``: the short adjective («прав»), the evaluative adjective of a
    predicate noun, the comparison («лучше собак») and the definition «X — это Y».
    """
    structure = (
        text if isinstance(text, StructuredUtterance) else analyze_structured_utterance(text)
    )
    text = structure.text
    if structure.corrections or any(
        reason != "fragment" or fallback_subject is None
        for c in structure.clauses
        for reason in c.unsupported
    ):
        return None
    if len(structure.clauses) != 1 or structure.clause_links:
        return None
    c = structure.clauses[0]
    tokens = structure.tokens
    if (
        not tokens
        or (("?" in text or c.clause_kind == "question") and not allow_question)
        or c.clause_kind == "command"
    ):
        return None
    unsupported = _UNSUPPORTED - {"бы"} if irrealis else _UNSUPPORTED
    if any(t.lemma in unsupported or t.surface in {"«", "»", '"'} for t in tokens):
        return None
    from debilgi.config import thought_content_enabled

    if (evaluation or definition) and thought_content_enabled():
        from debilgi.language.evaluative_reading import read_evaluative

        read = read_evaluative(
            structure, perspective=perspective, possessive_deixis=possessive_deixis,
            evaluation=evaluation, definition=definition, allow_question=allow_question,
            fallback_subject=fallback_subject,
        )
        if read is not None:
            return read
    # a verb the structure reads as a pragmatic marker («значит», «смотри») is not the predicate
    # (4.35, flag only): «ты значит честный?» is the question «ты честный?»
    marker_free = thought_content_enabled()
    finite = [t for t in tokens if t.part_of_speech == "verb" and "VERB" in t.grammemes
              and not (marker_free and t.semantic_role == "pragmatic_marker")]
    if len(finite) > 1:
        return None
    predicate = finite[0] if finite else None
    auxiliary = None
    if future_auxiliary and predicate is not None and predicate.lemma == "быть":
        actions = [t for t in tokens if "INFN" in t.grammemes and t.lemma != "быть"]
        if len(actions) == 1 and dict(predicate.grammatical_features).get("tense") == "future":
            auxiliary, predicate = predicate, actions[0]
    person_source = auxiliary or predicate
    subjects = [
        tokens[i]
        for i in c.subject_indices
        if tokens[i].part_of_speech in {"noun", "pronoun"} and tokens[i].lemma not in {"кто", "что"}
    ]
    if not subjects:
        subjects = [
            t
            for t in tokens
            if t.part_of_speech in {"noun", "pronoun"}
            and dict(t.grammatical_features).get("case") == "nominative"
            and t.lemma not in {"это", "что", "кто"}
        ]
    if len(subjects) > 1:
        return None
    used = set()
    if subjects:
        subject_token = subjects[0]
        subject = ConceptArgument(
            _lexeme(subject_token, perspective, possessive_deixis), grammatical_case="nominative"
        )
        used.add(subject_token.index)
        if (
            person_source
            and dict(person_source.grammatical_features).get("person") in {"first", "second"}
            and subject_token.part_of_speech != "pronoun"
        ):
            return None
    elif fallback_subject:
        subject = fallback_subject
    elif person_source and dict(person_source.grammatical_features).get("person") in {
        "first", "second"
    }:
        features = dict(person_source.grammatical_features)
        personal = deictic_person(features.get("person"), features.get("number"))
        if not personal:
            return None
        personal = _toward_speaker(personal, perspective)
        subject = ConceptArgument(ConceptLexeme(personal, personal, "pronoun"))
    else:
        return None
    dependencies = [SemanticDependency("subject", subject)]
    if auxiliary is not None:
        used.update((auxiliary.index, predicate.index))
        predicate_lexeme = replace(
            _lexeme(predicate, perspective, possessive_deixis), features=(("tense", "future"),)
        )
        predicate_class = "modal" if predicate.lemma in _MODALS else "lexical"
    elif predicate:
        used.add(predicate.index)
        predicate_lexeme = _lexeme(predicate, perspective, possessive_deixis)
        predicate_class = "modal" if predicate.lemma in _MODALS else "lexical"
    else:
        predicate_lexeme = ConceptLexeme("быть", "быть", "verb", (("tense", "present"),))
        predicate_class = "copular"
    infinitives = [t for t in tokens if "INFN" in t.grammemes and t.index not in used]
    action_index: int | None = None
    # Negation of the embedded action isn't negation of ability/desire.
    # Keep unsupported nested scope out of the graph rather than changing it.
    if predicate and any(t.lemma == "не" and t.index > predicate.index for t in tokens):
        return None
    if predicate_class == "copular":
        if any(t.lemma != "быть" for t in infinitives):
            return None
        used.update(t.index for t in infinitives)
    elif predicate_class == "modal":
        if len(infinitives) != 1:
            return None
        action = infinitives[0]
        dependencies.append(
            SemanticDependency(
                "action_complement",
                ConceptArgument(_lexeme(action, perspective, possessive_deixis)),
            )
        )
        used.add(action.index)
        action_index = action.index
    elif infinitives:
        return None
    positions: dict[str, int] = {}

    def argument(indices: tuple[int, ...], role: str, preposition: str | None = None) -> bool:
        indices = tuple(i for i in indices if i not in used)
        if not indices:
            return True
        heads = [
            tokens[i] for i in indices if tokens[i].part_of_speech in {"noun", "pronoun", "number"}
        ]
        if len(heads) != 1:
            return False
        head = heads[0]
        modifiers = [tokens[i] for i in indices if i != head.index]
        before = head.index - 1
        while before >= 0 and tokens[before].part_of_speech == "adjective" and before not in used:
            if tokens[before] not in modifiers:
                modifiers.append(tokens[before])
            before -= 1
        if any(t.part_of_speech != "adjective" for t in modifiers):
            return False
        dependencies.append(
            SemanticDependency(
                role,
                ConceptArgument(
                    _lexeme(head, perspective, possessive_deixis),
                    tuple(_lexeme(t, perspective, possessive_deixis) for t in modifiers),
                    preposition=preposition,
                    grammatical_case=dict(head.grammatical_features).get("case"),
                ),
            )
        )
        used.update((head.index, *(t.index for t in modifiers)))
        positions[role] = head.index
        return True

    if tree:
        likeness = _likeness_indices(tokens, used)
        if likeness is not None and predicate_class == "lexical" and predicate is not None:
            # A transitive verb with no object of its own («знаешь как дела») takes «как …» as
            # what it is told or asked, not as a standard of comparison.
            reading = DEFAULT_MORPHOLOGY.best(predicate.lemma)
            if (
                reading is not None
                and reading.features().get("transitivity") == "transitive"
                and not [i for i in c.direct_object_indices if i not in likeness[1]]
            ):
                likeness = None
        if likeness is not None:
            marker_index, phrase = likeness
            if not argument(phrase, "comparison"):
                return None
            used.add(marker_index)
            last = dependencies[-1]
            dependencies[-1] = SemanticDependency(
                "comparison",
                replace(
                    last.argument, preposition=tokens[marker_index].lemma,
                    grammatical_case="nominative",
                ),
                head="action_complement" if action_index is not None else None,
            )
    for indices, role in (
        (c.direct_object_indices, "direct_object"),
        (c.indirect_object_indices, "indirect_object"),
    ):
        if not argument(indices, role):
            return None
    for complement in c.complements:
        prep = tokens[complement.preposition_index]
        if not argument(
            complement.object_indices,
            "location" if complement.role.startswith("location") else "topic",
            prep.lemma,
        ):
            return None
        used.add(prep.index)
    properties = [
        t for t in tokens if t.index not in used and t.part_of_speech in {"adjective", "participle"}
    ]
    if properties:
        copular_action = any(
            d.role == "action_complement" and d.argument.head.lemma == "быть" for d in dependencies
        )
        if len(properties) != 1 or (predicate_class != "copular" and not copular_action):
            return None
        prop = properties[0]
        dependencies.append(
            SemanticDependency(
                "predicate_complement",
                ConceptArgument(
                    _lexeme(prop, perspective, possessive_deixis),
                    grammatical_case="nominative"
                    if predicate_class == "copular"
                    else "instrumental",
                ),
            )
        )
        used.add(prop.index)
    if predicate_class == "copular" and not properties:
        return None
    for token in tokens:
        if token.index in used or not token.surface[:1].isalnum():
            continue
        if allow_question and token.semantic_role == "question_target":
            continue
        if token.part_of_speech == "adverb" and token.semantic_role == "unassigned_content":
            dependencies.append(
                SemanticDependency(
                    "manner", ConceptArgument(_lexeme(token, perspective, possessive_deixis))
                )
            )
        elif token.lemma == "не" or token.semantic_role in {"function_word", "pragmatic_marker"}:
            continue
        else:
            return None
    if tree and action_index is not None:
        dependencies = [
            replace(dep, head="action_complement")
            if dep.head is None and dep.role in _EMBEDDED_ROLES
            and positions.get(dep.role, -1) > action_index
            else dep
            for dep in dependencies
        ]
    if irrealis and any(t.lemma == "бы" for t in tokens):
        predicate_lexeme = replace(
            predicate_lexeme,
            features=(
                *(f for f in predicate_lexeme.features if f[0] != "mood"),
                ("mood", "conditional"),
            ),
        )
    return ClauseThought(
        predicate_lexeme,
        tuple(dependencies),
        negated=any(t.lemma == "не" for t in tokens),
        confidence=0.86,
        predicate_class=predicate_class,
        semantic_time=c.semantic_time,
        scopes=c.scopes,
        parser_version=structure.parser_version,
        source_event_id=structure.source_event_id,
    )


# Roles whose word, standing behind the infinitive of a modal, belongs to that infinitive.
_EMBEDDED_ROLES = frozenset(
    {"direct_object", "indirect_object", "predicate_complement", "location", "topic"}
)


def _likeness_indices(tokens, used: set[int]) -> tuple[int, tuple[int, ...]] | None:
    """«как X»: the conjunction «как» that compares (it follows a word, it asks nothing) and the
    noun phrase it introduces, or ``None``. Position and part of speech decide, no word list
    beyond the relation's own conjunction."""
    for token in tokens:
        if token.lemma != "как" or token.part_of_speech != "conjunction" or token.index == 0:
            continue
        before = tokens[token.index - 1]
        if not before.surface[:1].isalnum():
            continue
        phrase: list[int] = []
        for later in tokens[token.index + 1:]:
            if later.part_of_speech == "adjective":
                phrase.append(later.index)
            elif later.part_of_speech in {"noun", "pronoun", "number"}:
                phrase.append(later.index)
                break
            else:
                break
        else:
            return None
        if (
            phrase
            and tokens[phrase[-1]].part_of_speech in {"noun", "pronoun", "number"}
            and not set(phrase) & used
            # «как ты спишь», «как мне быть»: a verb after the phrase makes «как» a question
            # about manner, and the phrase its subject, not a standard of comparison.
            and not any(t.part_of_speech == "verb" for t in tokens[phrase[-1] + 1:])
        ):
            return token.index, tuple(phrase)
    return None


def complete_nested_complement(
    thought: ClauseThought, earlier_texts: tuple[str, ...]
) -> ClauseThought:
    """«а хотел бы быть?»: an embedded verb with nothing attached takes what the earlier
    question named for the same verb («ты хочешь быть как Гитлер?» → «… как Гитлер»).

    ``earlier_texts`` are earlier turns of people, most recent first (never the agent's own
    words). The first earlier turn that reads, as a modal with the same embedded verb, decides:
    its attached arguments are copied (marked ``elliptical_complement`` in ``evidence`` so a
    planner can screen what it borrowed); a turn with that verb and nothing attached ends the
    search. Everything else is skipped. Nothing is invented: no earlier turn, no change.
    """
    if thought.predicate_class != "modal":
        return thought
    action = next((d for d in thought.dependencies if d.role == "action_complement"), None)
    if action is None or any(d.head == "action_complement" for d in thought.dependencies):
        return thought
    for text in earlier_texts:
        earlier = parse_proposition(
            text, allow_question=True, possessive_deixis=True, irrealis=True, tree=True
        )
        if earlier is None or earlier.predicate_class != "modal":
            continue
        verb = next((d for d in earlier.dependencies if d.role == "action_complement"), None)
        if verb is None or verb.argument.head.lemma != action.argument.head.lemma:
            continue
        attached = tuple(d for d in earlier.dependencies if d.head == "action_complement")
        if not attached:
            break
        return replace(
            thought,
            dependencies=(*thought.dependencies, *attached),
            evidence=(*thought.evidence, "elliptical_complement"),
        )
    return thought


# The speech-report predicates the agent already uses to quote itself (own_turn_plan);
# other report verbs («думать», «утверждать») are deliberately not covered.
SPEECH_REPORT_LEMMAS = frozenset({"сказать", "говорить"})


@dataclass(frozen=True, slots=True)
class ReportedProposition:
    """What an interlocutor attributes to the addressee: «ты говорил, что P»."""

    content: ClauseThought
    tense: str
    question: bool


@lru_cache(maxsize=2048)
def reported_proposition(
    text: str | StructuredUtterance, *, perspective: str = "user"
) -> ReportedProposition | None:
    """Read «<addressee> <say-verb>, что P» as the addressee's reported statement.

    Opt-in reader: ``parse_proposition`` is unchanged. Exactly one ``ccomp`` link, the
    matrix subject is the addressee («ты/вы»), the matrix predicate a speech-report
    anchor, and the complement parses by itself, without the conjunction.
    """
    structure = (
        text if isinstance(text, StructuredUtterance) else analyze_structured_utterance(text)
    )
    links = [link for link in structure.clause_links if link.relation == "ccomp"]
    if len(links) != 1 or len(structure.clause_links) != 1 or len(structure.clauses) != 2:
        return None
    link = links[0]
    matrix = structure.clauses[link.parent_clause_index]
    child = structure.clauses[link.child_clause_index]
    tokens = structure.tokens
    if matrix.clause_kind != "statement" or matrix.predicate_index is None or matrix.negated:
        return None
    predicate = tokens[matrix.predicate_index]
    if predicate.lemma not in SPEECH_REPORT_LEMMAS:
        return None
    if not any(tokens[i].lemma in {"ты", "вы"} for i in matrix.subject_indices):
        return None
    inner = " ".join(
        t.surface
        for t in tokens[child.token_start:child.token_end]
        if t.index != link.marker_index and t.surface[:1].isalnum()
    )
    content = parse_proposition(inner, perspective=perspective) if inner else None
    if content is None:
        return None
    tense = dict(predicate.grammatical_features).get("tense", "present")
    return ReportedProposition(content, tense, "?" in structure.text)


@dataclass(frozen=True, slots=True)
class ConditionalProposition:
    """«если C, то P»: what is asked under a condition, and whether it is only imagined."""

    condition: ClauseThought | None
    consequent: ClauseThought
    irrealis: bool


@lru_cache(maxsize=2048)
def conditional_proposition(
    text: str | StructuredUtterance,
    *,
    perspective: str = "user",
    possessive_deixis: bool = False,
) -> ConditionalProposition | None:
    """Read one ``advcl:condition`` link as a condition and its consequent.

    Opt-in reader: ``parse_proposition`` is unchanged (it refuses clause links and the
    conditional words). Each clause is parsed alone by its own token range, without the
    conditional particles; a clause without a subject takes the participant its structure
    names. The consequent must parse; the condition may stay unread (a modal without its
    infinitive is not a proposition), then it is ``None``. ``irrealis`` is the particle «бы»
    among the consequent's operators.
    """
    structure = (
        text if isinstance(text, StructuredUtterance) else analyze_structured_utterance(text)
    )
    links = [link for link in structure.clause_links if link.relation == "advcl:condition"]
    if len(links) != 1 or len(structure.clause_links) != 1 or len(structure.clauses) != 2:
        return None
    link = links[0]
    tokens = structure.tokens
    noise = {"бы", "то", "если", "ежели", "а", "но", "и"}

    def read(clause):
        words = [
            t for t in tokens[clause.token_start:clause.token_end]
            if t.surface[:1].isalnum() and t.lemma not in noise
        ]
        if not words:
            return None
        text_ = " ".join(t.surface for t in words)
        if not clause.subject_indices and clause.implicit_subject:
            text_ = ("ты " if clause.implicit_subject == "addressee" else "я ") + text_
        return parse_proposition(
            text_, perspective=perspective, possessive_deixis=possessive_deixis
        )

    consequent_clause = structure.clauses[link.parent_clause_index]
    consequent = read(consequent_clause)
    if consequent is None:
        return None
    irrealis = any(
        t.lemma == "бы" for t in tokens[consequent_clause.token_start:consequent_clause.token_end]
    )
    return ConditionalProposition(
        read(structure.clauses[link.child_clause_index]), consequent, irrealis
    )


def _parts(text: str) -> tuple[str, ...]:
    # Split only conjunctions/commas whose two sides independently parse.
    pieces = tuple(
        part.strip(" ,.;") for part in re.split(r",|;|\bи\b", text) if part.strip(" ,.;")
    )
    return pieces if len(pieces) > 1 else (text,)


def _group(
    text: str,
    perspective: str,
    antecedent: ClauseThought | None = None,
    structure: StructuredUtterance | None = None,
) -> tuple[ClauseThought, ...]:
    fallback = subject_of(antecedent) if antecedent else None
    result = []
    parts = _parts(text)
    for part in parts:
        thought = parse_proposition(
            structure if structure is not None and len(parts) == 1 else part,
            perspective=perspective,
            fallback_subject=fallback,
        )
        if thought is None:
            return ()
        subject = subject_of(thought)
        if subject.head.lemma in _THIRD_PERSON:
            if fallback is None:
                return ()
            # Don't guess a referent when the previous action names two actors.
            if antecedent and any(
                d.role != "subject" and d.argument.head.part_of_speech in {"noun", "pronoun"}
                for d in antecedent.dependencies
            ):
                return ()
            thought = with_subject(thought, fallback)
        result.append(thought)
        fallback = subject_of(thought)
    return tuple(result)


def _record(
    source: str,
    text: str,
    kind: str,
    premises: tuple[ClauseThought, ...],
    conclusions: tuple[ClauseThought, ...],
) -> MeaningRecord:
    key = json.dumps(
        (
            kind,
            tuple(identity_graph(t) for t in premises),
            tuple(identity_graph(t) for t in conclusions),
        ),
        ensure_ascii=False,
        sort_keys=True,
    )
    meaning_id = hashlib.sha256((source + "|" + key).encode()).hexdigest()
    return MeaningRecord(meaning_id, source, text, kind, premises, conclusions)


def extract_meanings(
    text: str | StructuredUtterance,
    source_event_id: str,
    *,
    perspective: str = "user",
    structure: StructuredUtterance | None = None,
) -> tuple[MeaningRecord, ...]:
    if isinstance(text, StructuredUtterance):
        structure, text = text, text.text
    results = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if (
            not sentence
            or "?" in sentence
            or any(word in _UNSUPPORTED - {"если", "когда"} for word in tokenize(sentence))
        ):
            continue
        if any(mark in sentence for mark in ('"', "«", "»")):
            continue
        lower = sentence.casefold().strip(" .!")
        if lower.startswith("если "):
            condition, separator, effect = lower[5:].partition(",")
            premises = _group(condition, perspective)
            conclusions = (
                _group(effect, perspective, premises[0] if len(premises) == 1 else None)
                if separator
                else ()
            )
            if premises and conclusions:
                results.append(
                    _record(source_event_id, sentence, "condition", premises, conclusions)
                )
            continue
        match = re.search(r"\b(?:это\s+когда|означает,?\s+что|значит,?\s+что)\b", lower)
        if match:
            left = lower[: match.start()].strip(" ,—–-")
            right = lower[match.end() :].strip(" ,—–-")
            scope = re.search(r"\bдля\s+(тебя|меня|вас|нас)\s*$", left)
            subject = None
            if scope:
                personal = {"тебя": "ты", "меня": "я", "вас": "вы", "нас": "мы"}[scope[1]]
                personal = _toward_speaker(personal, perspective)
                subject = ConceptArgument(ConceptLexeme(personal, personal, "pronoun"))
                left = left[: scope.start()].strip(" ,—–-")
            premise = parse_proposition(
                left, perspective=perspective, fallback_subject=subject or variable_subject()
            )
            if premise:
                conclusions = _group(right, perspective, premise)
                if conclusions:
                    if subject_of(premise).head.entity_kind == "semantic_variable":
                        head = subject_of(conclusions[0])
                        if head.head.entity_kind != "semantic_variable":
                            premise = with_subject(premise, head)
                    if all(
                        subject_of(t).head.lemma == subject_of(premise).head.lemma
                        for t in conclusions
                    ):
                        results.append(
                            _record(
                                source_event_id, sentence, "definition", (premise,), conclusions
                            )
                        )
            continue
        match = re.search(r",?\s+потому\s+что\s+", lower)
        if match:
            effect = _group(lower[: match.start()], perspective)
            cause = _group(
                lower[match.end() :], perspective, effect[0] if len(effect) == 1 else None
            )
            if effect and cause:
                results.append(_record(source_event_id, sentence, "cause", cause, effect))
                results.append(_record(source_event_id, sentence, "fact", (), (*cause, *effect)))
            continue
        thoughts = _group(sentence, perspective, structure=structure if sentence == text else None)
        if thoughts and not any(
            subject_of(t).head.entity_kind == "semantic_variable" for t in thoughts
        ):
            results.append(_record(source_event_id, sentence, "fact", (), thoughts))
    return tuple(results)


def meaning_from_data(data: dict) -> MeaningRecord:
    return MeaningRecord(
        data["meaning_id"],
        data["source_event_id"],
        data["text"],
        data["kind"],
        tuple(thought_from_data(t) for t in data["premises"]),
        tuple(thought_from_data(t) for t in data["conclusions"]),
        tuple(data.get("claim_ids", ())),
    )


@lru_cache(maxsize=8192)
def _raw_argument_key(argument: ConceptArgument) -> tuple:
    head = argument.head
    lemma, pos, voice = head.lemma, head.part_of_speech, dict(head.features).get("voice")
    if pos in {"adjective", "participle"}:
        features = dict(head.features)
        short_features = {
            "form": "short_adjective" if pos == "adjective" else "short_participle",
            "number": features.get("number", "singular"),
        }
        if short_features["number"] != "plural":
            short_features["gender"] = features.get("gender", "masculine")
        short = DEFAULT_MORPHOLOGY.inflect(head.surface, **short_features)
        reading = DEFAULT_MORPHOLOGY.best(short) if short else None
        if reading:
            lemma, pos = reading.lemma, reading.part_of_speech
            voice = reading.features().get("voice")
    return (
        lemma,
        pos,
        argument.preposition,
        dict(head.features).get("number", "singular"),
        voice,
        tuple(m.lemma for m in argument.modifiers),
        tuple(m.lemma for m in argument.postmodifiers),
    )


_MEANING_CANON = ContextVar("meaning_canon", default=None)


def _argument_key(argument: ConceptArgument, canonical=None) -> tuple:
    raw = _raw_argument_key(argument)
    canonical = _MEANING_CANON.get() if canonical is None else canonical
    if not canonical:
        return raw
    def lemma(word):
        return canonical.get(word, word)
    return (lemma(raw[0]), *raw[1:5], tuple(map(lemma, raw[5])), tuple(map(lemma, raw[6])))


@contextmanager
def canonical_meanings(canonical):
    """A server's canon is local to this operation, including concurrent replies."""
    token = _MEANING_CANON.set(canonical)
    try:
        yield
    finally:
        _MEANING_CANON.reset(token)


def proposition_key(thought: ClauseThought, *, signed: bool = True, canonical=None) -> tuple:
    # The empty canon must preserve the historical tuple byte for byte.
    if canonical is None:
        canonical = _MEANING_CANON.get()
    key = (
        thought.predicate.lemma,
        thought.predicate_class,
        dict(thought.predicate.features).get("tense", "present"),
        tuple(
            sorted(
                (dep.role, _argument_key(dep.argument, canonical)) for dep in thought.dependencies
            )
        ),
        thought.negated if signed else None,
    )
    if not canonical:
        return key

    return (canonical.get(key[0], key[0]), *key[1:])


def contextual(thought: ClauseThought) -> bool:
    return any(dep.argument.head.lemma in _RELATIVE | _THIRD_PERSON for dep in thought.dependencies)


def _match(
    pattern: ClauseThought, actual: ClauseThought, bindings: dict[str, ConceptArgument]
) -> dict[str, ConceptArgument] | None:
    canon = _MEANING_CANON.get() or {}
    if (
        canon.get(pattern.predicate.lemma, pattern.predicate.lemma)
        != canon.get(actual.predicate.lemma, actual.predicate.lemma)
        or pattern.predicate_class != actual.predicate_class
        or pattern.negated != actual.negated
        or dict(pattern.predicate.features).get("tense", "present")
        != dict(actual.predicate.features).get("tense", "present")
    ):
        return None
    desired = {dep.role: dep.argument for dep in pattern.dependencies}
    supplied = {dep.role: dep.argument for dep in actual.dependencies}
    if desired.keys() != supplied.keys():
        return None
    result = dict(bindings)
    for role, argument in desired.items():
        value = supplied[role]
        if argument.head.entity_kind == "semantic_variable":
            previous = result.get(argument.head.lemma)
            if previous is not None and _argument_key(previous) != _argument_key(value):
                return None
            result[argument.head.lemma] = value
        elif _argument_key(argument) != _argument_key(value):
            return None
    return result


def _instantiate(
    thought: ClauseThought, bindings: dict[str, ConceptArgument]
) -> ClauseThought | None:
    if any(
        d.argument.head.entity_kind == "semantic_variable" and d.argument.head.lemma not in bindings
        for d in thought.dependencies
    ):
        return None
    return replace(
        thought,
        dependencies=tuple(
            replace(dep, argument=bindings.get(dep.argument.head.lemma, dep.argument))
            for dep in thought.dependencies
        ),
    )


def matches_proposition(pattern: ClauseThought, actual: ClauseThought) -> bool:
    return _match(pattern, actual, {}) is not None


def infer_meanings(
    records: tuple[MeaningRecord, ...], *, assumptions: tuple[ClauseThought, ...] = ()
) -> tuple[MeaningProof, ...]:
    proofs: dict[tuple, MeaningProof] = {}
    rules = []
    for record in records:
        if record.kind == "fact":
            for thought in record.conclusions:
                proofs.setdefault(
                    proposition_key(thought),
                    MeaningProof(thought, (record.source_event_id,), claim_ids=record.claim_ids),
                )
        else:
            rules.append((record, record.premises, record.conclusions))
            if record.kind == "definition":
                # A conjunction in a definition is reversed only as a whole.
                rules.append((record, record.conclusions, record.premises))
    for thought in assumptions:
        proofs[proposition_key(thought)] = MeaningProof(thought, tuple(thought.evidence))
    for _ in range(5):
        snapshot = tuple(proofs.values())
        conflicts = {
            proposition_key(p.thought, signed=False)
            for p in snapshot
            if proposition_key(replace(p.thought, negated=not p.thought.negated)) in proofs
        }
        changed = False
        for record, premises, conclusions in rules[:256]:
            matches: list[tuple[dict, tuple[MeaningProof, ...]]] = [({}, ())]
            for premise in premises:
                following = []
                for bindings, sources in matches:
                    for proof in snapshot:
                        if proposition_key(proof.thought, signed=False) in conflicts:
                            continue
                        bound = _match(premise, proof.thought, bindings)
                        if bound is not None:
                            following.append((bound, (*sources, proof)))
                matches = following[:64]
            for bindings, sources in matches:
                events = tuple(
                    dict.fromkeys(
                        (*[e for p in sources for e in p.source_event_ids], record.source_event_id)
                    )
                )
                path = tuple(
                    dict.fromkeys((*[r for p in sources for r in p.rule_ids], record.meaning_id))
                )
                for conclusion in conclusions:
                    thought = _instantiate(conclusion, bindings)
                    if thought is None:
                        continue
                    thought = replace(thought, evidence=events, confidence=0.84)
                    key = proposition_key(thought)
                    if key not in proofs:
                        proofs[key] = MeaningProof(
                            thought,
                            events,
                            path,
                            tuple(proposition_key(p.thought) for p in sources),
                            tuple(dict.fromkeys(c for p in sources for c in p.claim_ids)),
                        )
                        changed = True
            if len(proofs) >= 128:
                break
        if not changed or len(proofs) >= 128:
            break
    conflicts = {
        proposition_key(p.thought, signed=False)
        for p in proofs.values()
        if proposition_key(replace(p.thought, negated=not p.thought.negated)) in proofs
    }
    invalid = {
        key for key, p in proofs.items() if proposition_key(p.thought, signed=False) in conflicts
    }
    # A later conflicting rule also retracts descendants of the disputed fact.
    for _ in range(len(proofs)):
        following = invalid | {
            key for key, p in proofs.items() if any(parent in invalid for parent in p.premise_keys)
        }
        if following == invalid:
            break
        invalid = following
    return tuple(p for key, p in proofs.items() if key not in invalid)


def meaning_query(text: str) -> MeaningQuery:
    tokens = tokenize(text)
    if "кто" in tokens:
        thought = parse_proposition(text, allow_question=True, fallback_subject=variable_subject())
        if thought and subject_of(thought).head.entity_kind == "semantic_variable":
            return MeaningQuery("subject_slot", thought)
    if "если" in tokens:
        index = tokens.index("если")
        assumption = parse_proposition(" ".join(tokens[index + 1 :]).rstrip("?"))
        return MeaningQuery("consequence", None, assumption)
    if "что" in tokens and any(word in tokens for word in {"значит", "означает", "такое"}):
        marker = next(i for i, word in enumerate(tokens) if word in {"значит", "означает", "такое"})
        state = parse_proposition(
            " ".join(tokens[marker + 1 :]).rstrip("?"), fallback_subject=variable_subject()
        )
        return MeaningQuery("definition", state, state)
    thought = parse_proposition(text, allow_question=True)
    if "почему" in tokens:
        return MeaningQuery("cause", thought)
    if thought and thought.predicate.lemma == "хотеть":
        actions = [d for d in thought.dependencies if d.role == "action_complement"]
        if actions and actions[0].argument.head.lemma == "быть":
            state = replace(
                thought,
                predicate=ConceptLexeme("быть", "быть", "verb", (("tense", "present"),)),
                dependencies=tuple(
                    replace(d, argument=replace(d.argument, grammatical_case="nominative"))
                    if d.role == "predicate_complement"
                    else d
                    for d in thought.dependencies
                    if d.role != "action_complement"
                ),
                negated=False,
                predicate_class="copular",
            )
            return MeaningQuery("preference", thought, state)
        return MeaningQuery("preference", thought)
    return MeaningQuery("polar" if "?" in text else "statement", thought)


def goal_effect(thought: ClauseThought, goal: MeaningGoal) -> float:
    """Evaluate a learned capability against the current task, not a fixed state preference."""
    if thought.predicate.lemma not in {"мочь", "уметь"}:
        return 0.0
    roles = {d.role: d.argument for d in thought.dependencies}
    action = roles.get("action_complement")
    if not action or action.head.lemma != goal.action.predicate.lemma:
        return 0.0
    if subject_of(thought).head.lemma != subject_of(goal.action).head.lemma:
        return 0.0
    desired = {d.role: d.argument for d in goal.action.dependencies}
    for role, argument in roles.items():
        if role in {"subject", "action_complement"}:
            continue
        if role not in desired or _argument_key(argument) != _argument_key(desired[role]):
            return 0.0
    return (-1.0 if thought.negated != goal.action.negated else 1.0) * goal.weight


def learned_result_link(
    action: ClauseThought, records: tuple[MeaningRecord, ...], source_event_id: str, text: str
) -> MeaningRecord | None:
    """A completed transitive event supplies its passive result for the patient.

    Only connect to a passive state actually present in learned rules. No named
    action/state mapping is supplied by the program. Keep the original event as
    the premise so the answer never loses its actor or condition.
    """
    features = dict(action.predicate.features)
    roles = {d.role: d.argument for d in action.dependencies}
    if (
        action.negated
        or action.predicate_class != "lexical"
        or features.get("aspect") != "perfective"
        or features.get("tense") not in {"future", "past"}
        or roles.keys() != {"subject", "direct_object"}
    ):
        return None
    for record in records:
        for state in record.premises:
            prop = next(
                (d.argument for d in state.dependencies if d.role == "predicate_complement"), None
            )
            if (
                state.predicate_class == "copular"
                and not state.negated
                and prop
                and prop.head.lemma == action.predicate.lemma
                and dict(prop.head.features).get("voice") == "passive"
                and {d.role for d in state.dependencies} == {"subject", "predicate_complement"}
            ):
                return _record(
                    source_event_id,
                    text,
                    "event_result",
                    (action,),
                    (with_subject(state, roles["direct_object"]),),
                )
    return None
